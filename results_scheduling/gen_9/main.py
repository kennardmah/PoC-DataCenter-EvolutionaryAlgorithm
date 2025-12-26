# EVOLVE-BLOCK-START
"""Modular cost-aware job-shop scheduler for government data center"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import math

TIME_HORIZON = 24

# =============================================================================
# DATA MODEL
# =============================================================================

@dataclass
class Server:
    id: str
    server_type: str
    capacity: int
    is_cloud: bool = False

@dataclass
class Operation:
    server_type: str
    duration: int
    capacity_required: int = 1

@dataclass
class Job:
    id: str
    description: str
    priority: str
    deadline: int
    delay_weight: float
    operations: List[Operation]

@dataclass
class MachineFailure:
    machine_id: str
    failure_start: int
    failure_end: int
    capacity_reduction: int
    failure_rate: float = 1.0
    recovery_rate: float = 1.0

@dataclass
class Instance:
    servers: Dict[str, Server]
    jobs: Dict[str, Job]
    server_cost: Dict[str, Dict[int, float]]
    time_horizon: int
    weights: Dict[str, float]
    machine_failures: Optional[List[MachineFailure]]

# =============================================================================
# INSTANCE LOADER
# =============================================================================

def create_instance(
    scenario: str = "default",
    failure_scenario: Optional[str] = None
) -> Instance:
    # For brevity, only default scenario is shown; CSV loader could be plugged in here
    SERVER_COST = {
        "M1": {**{t:80 for t in range(6)}, **{t:120 for t in range(6,10)},
               **{t:100 for t in range(10,17)}, **{t:140 for t in range(17,22)}, **{t:90 for t in (22,23)}},
        "M2": {**{t:20 for t in range(6)}, **{t:45 for t in range(6,10)},
               **{t:30 for t in range(10,17)}, **{t:55 for t in range(17,22)}, **{t:25 for t in (22,23)}},
        "M3": {**{t:60 for t in range(6)}, **{t:95 for t in range(6,10)},
               **{t:75 for t in range(10,17)}, **{t:110 for t in range(17,22)}, **{t:70 for t in (22,23)}},
        "M4": {**{t:50 for t in range(6)}, **{t:70 for t in range(6,10)},
               **{t:60 for t in range(10,17)}, **{t:80 for t in range(17,22)}, **{t:55 for t in (22,23)}},
        "M5": {**{t:15 for t in range(6)}, **{t:30 for t in range(6,10)},
               **{t:22 for t in range(10,17)}, **{t:35 for t in range(17,22)}, **{t:18 for t in (22,23)}},
    }
    # Cloud costs = 5x local
    for ci, mt in [("C1","M1"),("C2","M2"),("C3","M3"),("C4","M4"),("C5","M5")]:
        SERVER_COST[ci] = {t: SERVER_COST[mt][t]*5 for t in SERVER_COST[mt]}

    servers = {
        "M1": Server("M1","GPU",20), "M2": Server("M2","CPU",40),
        "M3": Server("M3","HighMem",10), "M4": Server("M4","Secure",5),
        "M5": Server("M5","Ingest",30),
        "C1": Server("C1","GPU",10**9,True),
        "C2": Server("C2","CPU",10**9,True),
        "C3": Server("C3","HighMem",10**9,True),
        "C4": Server("C4","Secure",10**9,True),
        "C5": Server("C5","Ingest",10**9,True),
    }
    jobs = {
        "J1": Job("J1","National census data processing","HIGH",12,10.0,
                  [Operation("Ingest",2,5), Operation("HighMem",3,3), Operation("CPU",2,4)]),
        "J2": Job("J2","Tax fraud detection ML model","HIGH",10,8.0,
                  [Operation("Ingest",1,3), Operation("GPU",3,8), Operation("CPU",1,2)]),
        "J3": Job("J3","Emergency response coordination","CRITICAL",6,15.0,
                  [Operation("Secure",2,2), Operation("CPU",1,3), Operation("HighMem",1,2)]),
        "J4": Job("J4","Geospatial infrastructure analysis","MEDIUM",16,5.0,
                  [Operation("Ingest",2,4), Operation("HighMem",4,5)]),
        "J5": Job("J5","Classified intelligence processing","CRITICAL",8,12.0,
                  [Operation("Secure",3,3), Operation("Secure",2,2)]),
        "J6": Job("J6","Public health data aggregation","MEDIUM",18,4.0,
                  [Operation("Ingest",1,2), Operation("CPU",2,5), Operation("HighMem",2,3)]),
        "J7": Job("J7","Climate model training","LOW",20,2.0,
                  [Operation("Ingest",1,3), Operation("GPU",4,10)]),
        "J8": Job("J8","Social services optimization","MEDIUM",14,6.0,
                  [Operation("Ingest",1,3), Operation("CPU",2,6), Operation("HighMem",2,4)]),
    }
    return Instance(servers, jobs, SERVER_COST, TIME_HORIZON,
                    {"compute_cost":1.0,"delay_penalty":5.0}, machine_failures=None)

# =============================================================================
# EVALUATOR
# =============================================================================

class Evaluator:
    def __init__(self, inst: Instance):
        self.inst = inst

    def get_available_capacity(self, server_id: str, time: int) -> int:
        base = self.inst.servers[server_id].capacity
        if not self.inst.machine_failures:
            return base
        red = 0
        for f in self.inst.machine_failures:
            if f.machine_id != server_id: continue
            if time < f.failure_start: continue
            if time >= f.failure_end:
                tau = time - f.failure_end
                rec = 1-math.exp(-f.recovery_rate*tau)
                red += f.capacity_reduction*(1-rec)
            else:
                tau = time - f.failure_start
                fail = 1-math.exp(-f.failure_rate*tau)
                red += f.capacity_reduction*fail
        return max(0, base-int(red))

    def evaluate(self, schedule: Dict[str,List[Dict[str,int]]]) -> Any:
        from collections import namedtuple
        Fitness = namedtuple("Fitness",["score","compute_cost","delay_penalty","violations","job_completion_times"])
        cap_usage = {s:{} for s in self.inst.servers}
        comp=0; delay_pen=0; viol=[]
        job_ct = {}
        for jid,ops in schedule.items():
            job=self.inst.jobs[jid]
            if len(ops)!=len(job.operations):
                viol.append(f"{jid}: wrong op count"); continue
            prev_end=0
            for idx,assign in enumerate(ops):
                srv, st = assign["server"], assign["start"]
                op=job.operations[idx]
                if st<prev_end: viol.append(f"{jid}: seq error op{idx}")
                prev_end=st+op.duration
                if self.inst.servers[srv].server_type!=op.server_type:
                    viol.append(f"{jid}: wrong type op{idx}")
                for t in range(st,st+op.duration):
                    cap_usage[srv][t]=cap_usage[srv].get(t,0)+op.capacity_required
                    comp+=self.inst.server_cost[srv][t%self.inst.time_horizon]
            job_ct[jid]=prev_end
            lateness=max(0,prev_end-job.deadline)
            delay_pen+=lateness*job.delay_weight
        for srv,usage in cap_usage.items():
            for t,used in usage.items():
                av=self.get_available_capacity(srv,t)
                if used>av:
                    viol.append(f"{srv}@{t}: cap {used}>{av}")
        score = (self.inst.weights["compute_cost"]*comp+
                 self.inst.weights["delay_penalty"]*delay_pen+
                 1e4*len(viol))
        return Fitness(round(score,2),round(comp,2),round(delay_pen,2),viol,job_ct)

# =============================================================================
# SCHEDULER
# =============================================================================

class Scheduler:
    def __init__(self, inst: Instance):
        self.inst = inst
        self.time_horizon = inst.time_horizon

    def order_jobs(self) -> List[str]:
        # priority map
        pmap={"CRITICAL":0,"HIGH":1,"MEDIUM":2,"LOW":3}
        slack = {j.id:(j.deadline - sum(op.duration for op in j.operations)) 
                 for j in self.inst.jobs.values()}
        return sorted(self.inst.jobs.keys(),
            key=lambda j: (pmap[self.inst.jobs[j].priority],
                           slack[j]/(self.inst.jobs[j].delay_weight+1e-6),
                           self.inst.jobs[j].deadline))

    def schedule(self) -> Dict[str,List[Dict[str,int]]]:
        cap_usage={s:{} for s in self.inst.servers}
        job_ready={j:0 for j in self.inst.jobs}
        schedule={}
        for jid in self.order_jobs():
            job=self.inst.jobs[jid]
            schedule[jid]=[]
            for op in job.operations:
                best = None  # (cost, server, start)
                earliest = job_ready[jid]
                # consider each local then cloud
                cands=[s for s in self.inst.servers.values() 
                       if s.server_type==op.server_type]
                # local first
                cands.sort(key=lambda s: s.is_cloud)
                for srv in cands:
                    # allowable window
                    latest = job.deadline - op.duration
                    if latest < earliest: latest = self.time_horizon-op.duration
                    for st in range(earliest, latest+1):
                        # check capacity
                        ok=True
                        for t in range(st,st+op.duration):
                            used=cap_usage[srv.id].get(t,0)
                            av = cap_usage[srv.id]  # placeholder
                            # simple: cloud infinite, local check base
                            if not srv.is_cloud and used+op.capacity_required>srv.capacity:
                                ok=False; break
                        if not ok: continue
                        # cost
                        csum = sum(self.inst.server_cost[srv.id][t%self.time_horizon]
                                   for t in range(st,st+op.duration))
                        # choose minimal cost
                        if best is None or csum<best[0]:
                            best=(csum,srv,st)
                        if srv.is_cloud:
                            # for cloud, only need earliest
                            break
                    if best and srv.is_cloud:
                        break
                if best is None:
                    # fallback: first local earliest
                    srv=cands[0]; st=earliest; best=(0,srv,st)
                _,srv,st = best
                # reserve
                for t in range(st,st+op.duration):
                    cap_usage[srv.id][t]=cap_usage[srv.id].get(t,0)+op.capacity_required
                schedule[jid].append({"server":srv.id,"start":st})
                job_ready[jid]=st+op.duration
        return schedule

# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_scheduling():
    """Run the schedule constructor and return results."""
    inst = create_instance()
    schedule = construct_schedule(inst)
    fitness = evaluate(inst, schedule)
    return schedule, fitness, float(fitness.score)
