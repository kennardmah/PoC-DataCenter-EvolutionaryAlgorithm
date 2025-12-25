# EVOLVE-BLOCK-START
"""Constructor-based job-shop scheduling for government data center"""

from dataclasses import dataclass
from typing import Dict, List, Any
import copy


# =============================================================================
# PROBLEM DEFINITION
# =============================================================================

TIME_HORIZON = 24

# Direct server cost per hour (¥/hour) - like on-demand compute pricing
# Cost varies by time slot (demand-based pricing)
SERVER_COST = {
    "S1": {  # CPU Server 1
        0: 16, 1: 16, 2: 16, 3: 16, 4: 16, 5: 16,      # off-peak (cheap)
        6: 36, 7: 36, 8: 36, 9: 36,                    # morning peak
        10: 24, 11: 24, 12: 24, 13: 24, 14: 24, 15: 24, 16: 24,  # midday
        17: 44, 18: 44, 19: 44, 20: 44, 21: 44,        # evening peak (expensive)
        22: 20, 23: 20                                  # late night
    },
    "S2": {  # CPU Server 2 (same as S1)
        0: 16, 1: 16, 2: 16, 3: 16, 4: 16, 5: 16,
        6: 36, 7: 36, 8: 36, 9: 36,
        10: 24, 11: 24, 12: 24, 13: 24, 14: 24, 15: 24, 16: 24,
        17: 44, 18: 44, 19: 44, 20: 44, 21: 44,
        22: 20, 23: 20
    },
    "S3": {  # GPU Server (more expensive)
        0: 36, 1: 36, 2: 36, 3: 36, 4: 36, 5: 36,
        6: 81, 7: 81, 8: 81, 9: 81,
        10: 54, 11: 54, 12: 54, 13: 54, 14: 54, 15: 54, 16: 54,
        17: 99, 18: 99, 19: 99, 20: 99, 21: 99,
        22: 45, 23: 45
    },
    "S4": {  # Storage Server (cheapest)
        0: 10, 1: 10, 2: 10, 3: 10, 4: 10, 5: 10,
        6: 22, 7: 22, 8: 22, 9: 22,
        10: 14, 11: 14, 12: 14, 13: 14, 14: 14, 15: 14, 16: 14,
        17: 26, 18: 26, 19: 26, 20: 26, 21: 26,
        22: 12, 23: 12
    },
}


@dataclass
class Server:
    id: str
    server_type: str


@dataclass
class Operation:
    server_type: str
    duration: int


@dataclass
class Job:
    id: str
    description: str
    priority: str
    deadline: int
    delay_weight: float
    operations: List[Operation]


@dataclass
class Instance:
    servers: Dict[str, Server]
    jobs: Dict[str, Job]
    server_cost: Dict[str, Dict[int, float]]  # server_id -> {hour: cost}
    time_horizon: int
    weights: Dict[str, float]


Schedule = Dict[str, List[Dict[str, Any]]]


def create_instance() -> Instance:
    """Create the problem instance."""
    servers = {
        "S1": Server("S1", "CPU"),
        "S2": Server("S2", "CPU"),
        "S3": Server("S3", "GPU"),
        "S4": Server("S4", "Storage"),
    }
    jobs = {
        "J1": Job("J1", "Emergency response analytics", "HIGH", 8, 10.0, [
            Operation("Storage", 1), Operation("CPU", 2), Operation("GPU", 1)
        ]),
        "J2": Job("J2", "Disaster damage estimation", "HIGH", 10, 8.0, [
            Operation("Storage", 1), Operation("GPU", 2), Operation("CPU", 1)
        ]),
        "J3": Job("J3", "Tax data aggregation", "MEDIUM", 18, 4.0, [
            Operation("CPU", 2), Operation("Storage", 1), Operation("CPU", 1)
        ]),
        "J4": Job("J4", "Census preprocessing", "MEDIUM", 20, 3.0, [
            Operation("Storage", 1), Operation("CPU", 2)
        ]),
        "J5": Job("J5", "Open data batch update", "LOW", 22, 1.0, [
            Operation("CPU", 1), Operation("Storage", 1)
        ]),
        "J6": Job("J6", "Archive and backup", "LOW", 24, 0.5, [
            Operation("Storage", 2), Operation("CPU", 2)
        ]),
    }
    return Instance(servers, jobs, SERVER_COST, TIME_HORIZON,
                   {"compute_cost": 1.0, "delay_penalty": 5.0})


# =============================================================================
# FITNESS EVALUATION
# =============================================================================

@dataclass
class Fitness:
    score: float
    compute_cost: float
    delay_penalty: float
    violations: List[str]
    job_completion_times: Dict[str, int]

    def is_valid(self) -> bool:
        return len(self.violations) == 0


def evaluate(inst: Instance, schedule: Schedule) -> Fitness:
    """Evaluate a schedule and return fitness."""
    server_usage = {s: [] for s in inst.servers}
    job_completion_times = {}
    total_compute_cost = 0.0
    total_delay_penalty = 0.0
    violations = []

    for job_id, ops in schedule.items():
        job = inst.jobs[job_id]
        if len(ops) != len(job.operations):
            violations.append(f"{job_id}: wrong number of operations")
            continue

        prev_end = None
        for idx, op in enumerate(ops):
            server_id = op["server"]
            start = op["start"]
            duration = job.operations[idx].duration
            end = start + duration

            if prev_end is not None and start < prev_end:
                violations.append(f"{job_id}: op {idx} starts before previous ends")
            prev_end = end

            required_type = job.operations[idx].server_type
            actual_type = inst.servers[server_id].server_type
            if required_type != actual_type:
                violations.append(f"{job_id}: op {idx} requires {required_type}, got {actual_type}")

            server_usage[server_id].append((start, end, job_id, idx))
            
            # Direct cost lookup
            for t in range(start, end):
                total_compute_cost += inst.server_cost[server_id][t % inst.time_horizon]

        completion = prev_end if prev_end else 0
        job_completion_times[job_id] = completion
        delay = max(0, completion - job.deadline)
        total_delay_penalty += delay * job.delay_weight

    for server_id, intervals in server_usage.items():
        intervals.sort()
        for i in range(len(intervals) - 1):
            _, end1, job1, _ = intervals[i]
            start2, _, job2, _ = intervals[i + 1]
            if start2 < end1:
                violations.append(f"{server_id}: overlap between {job1} and {job2}")

    score = (inst.weights["compute_cost"] * total_compute_cost +
             inst.weights["delay_penalty"] * total_delay_penalty +
             len(violations) * 10000)

    return Fitness(round(score, 2), round(total_compute_cost, 2),
                   round(total_delay_penalty, 2), violations, job_completion_times)


# =============================================================================
# SCHEDULE CONSTRUCTOR (TO BE EVOLVED)
# =============================================================================

def construct_schedule(inst: Instance) -> Schedule:
    """
    Constructs a schedule for the job-shop scheduling problem.
    This function is evolved by ShinkaEvolve/AlphaEvolve.

    Args:
        inst: Problem instance with servers, jobs, server costs

    Returns:
        Schedule: Dict mapping job_id -> list of {"server": str, "start": int}
    """
    def get_servers_by_type(server_type: str) -> List[str]:
        return [s.id for s in inst.servers.values() if s.server_type == server_type]

    def op_cost(server_id: str, start: int, duration: int) -> float:
        """Compute cost of running operation on server at given time."""
        return sum(inst.server_cost[server_id][t % inst.time_horizon]
                   for t in range(start, start + duration))

    # --- Initial solution: Priority-based with cost awareness ---
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    sorted_jobs = sorted(inst.jobs.keys(),
                        key=lambda j: (priority_order[inst.jobs[j].priority],
                                      inst.jobs[j].deadline, j))

    server_available = {s: 0 for s in inst.servers}
    job_available = {j: 0 for j in inst.jobs}
    schedule = {}

    for job_id in sorted_jobs:
        job = inst.jobs[job_id]
        schedule[job_id] = []

        for op in job.operations:
            compatible = get_servers_by_type(op.server_type)
            best_choice, best_score = None, float("inf")

            for server_id in compatible:
                earliest = max(job_available[job_id], server_available[server_id])

                # Try a few start times to find cost-efficient slot
                for start in range(earliest, min(earliest + 6, inst.time_horizon - op.duration + 1)):
                    cost = op_cost(server_id, start, op.duration)
                    # Balance compute cost with completion time
                    delay_risk = max(0, start + op.duration - job.deadline) * job.delay_weight
                    score = cost + delay_risk * 2

                    if score < best_score:
                        best_score = score
                        best_choice = (server_id, start)

            if best_choice is None:
                server_id = compatible[0]
                start = max(job_available[job_id], server_available[server_id])
            else:
                server_id, start = best_choice

            schedule[job_id].append({"server": server_id, "start": start})
            finish = start + op.duration
            job_available[job_id] = finish
            server_available[server_id] = finish

    # --- Local search improvement ---
    for _ in range(50):
        improved = False
        current_fit = evaluate(inst, schedule)

        for job_id in inst.jobs:
            job = inst.jobs[job_id]
            for op_idx in range(len(job.operations)):
                op = job.operations[op_idx]
                compatible = get_servers_by_type(op.server_type)

                # Try swapping to different server or shifting time
                for new_server in compatible:
                    for delta in range(-3, 4):
                        new_start = max(0, schedule[job_id][op_idx]["start"] + delta)

                        # Create candidate schedule
                        candidate = copy.deepcopy(schedule)
                        candidate[job_id][op_idx] = {"server": new_server, "start": new_start}

                        new_fit = evaluate(inst, candidate)
                        if new_fit.score < current_fit.score:
                            schedule = candidate
                            current_fit = new_fit
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

        if not improved:
            break

    return schedule


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_scheduling():
    """Run the schedule constructor and return results."""
    inst = create_instance()
    schedule = construct_schedule(inst)
    fitness = evaluate(inst, schedule)
    return schedule, fitness, float(fitness.score)
