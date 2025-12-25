"""
Initial Strategies (Baseline Solvers)
=====================================
Baseline scheduling algorithms for the job-shop problem.
"""

from typing import List, Callable
from problem import Instance, Schedule


# Type alias: Solver takes Instance, returns Schedule
Solver = Callable[[Instance], Schedule]


def get_servers_by_type(inst: Instance, server_type: str) -> List[str]:
    """Get all server IDs of a given type."""
    return [s.id for s in inst.servers.values() if s.server_type == server_type]


def fifo_solver(inst: Instance) -> Schedule:
    """FIFO baseline: schedule jobs in order, earliest possible time."""
    server_available = {s: 0 for s in inst.servers}
    job_available = {j: 0 for j in inst.jobs}
    schedule = {}

    for job_id in sorted(inst.jobs.keys()):
        job = inst.jobs[job_id]
        schedule[job_id] = []

        for op in job.operations:
            compatible = get_servers_by_type(inst, op.server_type)
            chosen = min(compatible, key=lambda s: server_available[s])
            start = max(job_available[job_id], server_available[chosen])
            
            schedule[job_id].append({"server": chosen, "start": start})
            
            finish = start + op.duration
            job_available[job_id] = finish
            server_available[chosen] = finish

    return schedule


def energy_greedy_solver(inst: Instance) -> Schedule:
    """Energy-greedy: prefer low-cost time slots."""
    server_available = {s: 0 for s in inst.servers}
    job_available = {j: 0 for j in inst.jobs}
    schedule = {}

    def op_energy_cost(server_id: str, start: int, duration: int) -> float:
        power = inst.servers[server_id].power_kw
        return sum(power * inst.energy_price[t % inst.time_horizon] 
                   for t in range(start, start + duration))

    for job_id in sorted(inst.jobs.keys()):
        job = inst.jobs[job_id]
        schedule[job_id] = []

        for op in job.operations:
            compatible = get_servers_by_type(inst, op.server_type)
            best_choice, best_cost = None, float("inf")

            for server_id in compatible:
                earliest = max(job_available[job_id], server_available[server_id])
                # Try a range of start times (at least try earliest)
                max_start = max(earliest + 1, inst.time_horizon - op.duration + 1)
                for start in range(earliest, max_start):
                    cost = op_energy_cost(server_id, start, op.duration)
                    if cost < best_cost:
                        best_cost = cost
                        best_choice = (server_id, start)
            
            # Fallback if no choice found
            if best_choice is None:
                server_id = compatible[0]
                start = max(job_available[job_id], server_available[server_id])
            else:
                server_id, start = best_choice
            
            schedule[job_id].append({"server": server_id, "start": start})
            
            finish = start + op.duration
            job_available[job_id] = finish
            server_available[server_id] = finish

    return schedule


def priority_solver(inst: Instance) -> Schedule:
    """Priority-first: schedule high-priority jobs first."""
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    sorted_jobs = sorted(inst.jobs.keys(), 
                        key=lambda j: (priority_order[inst.jobs[j].priority], j))
    
    server_available = {s: 0 for s in inst.servers}
    job_available = {j: 0 for j in inst.jobs}
    schedule = {}

    for job_id in sorted_jobs:
        job = inst.jobs[job_id]
        schedule[job_id] = []

        for op in job.operations:
            compatible = get_servers_by_type(inst, op.server_type)
            chosen = min(compatible, key=lambda s: server_available[s])
            start = max(job_available[job_id], server_available[chosen])
            
            schedule[job_id].append({"server": chosen, "start": start})
            
            finish = start + op.duration
            job_available[job_id] = finish
            server_available[chosen] = finish

    return schedule


# List of all baseline solvers
BASELINE_SOLVERS = [fifo_solver, energy_greedy_solver, priority_solver]
