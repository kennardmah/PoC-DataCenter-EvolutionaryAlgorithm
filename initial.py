# EVOLVE-BLOCK-START
"""Constructor-based job-shop scheduling for government data center"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import copy
import os


# =============================================================================
# PROBLEM DEFINITION
# =============================================================================

TIME_HORIZON = 24

# Direct server cost per hour (¥/hour) - like on-demand compute pricing
# Cost varies by time slot (demand-based pricing)
SERVER_COST = {
    "M1": {  # GPU Compute - most expensive (training/inference)
        0: 80, 1: 80, 2: 80, 3: 80, 4: 80, 5: 80,      # off-peak
        6: 120, 7: 120, 8: 120, 9: 120,                 # morning peak
        10: 100, 11: 100, 12: 100, 13: 100, 14: 100, 15: 100, 16: 100,  # midday
        17: 140, 18: 140, 19: 140, 20: 140, 21: 140,   # evening peak (most expensive)
        22: 90, 23: 90                                  # late night
    },
    "M2": {  # CPU Compute - moderate cost (control logic, feature engineering)
        0: 20, 1: 20, 2: 20, 3: 20, 4: 20, 5: 20,
        6: 45, 7: 45, 8: 45, 9: 45,
        10: 30, 11: 30, 12: 30, 13: 30, 14: 30, 15: 30, 16: 30,
        17: 55, 18: 55, 19: 55, 20: 55, 21: 55,
        22: 25, 23: 25
    },
    "M3": {  # High-Memory Analytics - expensive (graphs, large matrices)
        0: 60, 1: 60, 2: 60, 3: 60, 4: 60, 5: 60,
        6: 95, 7: 95, 8: 95, 9: 95,
        10: 75, 11: 75, 12: 75, 13: 75, 14: 75, 15: 75, 16: 75,
        17: 110, 18: 110, 19: 110, 20: 110, 21: 110,
        22: 70, 23: 70
    },
    "M4": {  # Secure/Isolated Compute - premium cost (gov-only workloads)
        0: 50, 1: 50, 2: 50, 3: 50, 4: 50, 5: 50,
        6: 70, 7: 70, 8: 70, 9: 70,
        10: 60, 11: 60, 12: 60, 13: 60, 14: 60, 15: 60, 16: 60,
        17: 80, 18: 80, 19: 80, 20: 80, 21: 80,
        22: 55, 23: 55
    },
    "M5": {  # Data Ingest & Preprocessing - cheapest (IO-heavy, staging)
        0: 15, 1: 15, 2: 15, 3: 15, 4: 15, 5: 15,
        6: 30, 7: 30, 8: 30, 9: 30,
        10: 22, 11: 22, 12: 22, 13: 22, 14: 22, 15: 22, 16: 22,
        17: 35, 18: 35, 19: 35, 20: 35, 21: 35,
        22: 18, 23: 18
    },
    # Cloud machines - 5x local cost, infinite capacity
    "C1": {  # Cloud GPU
        0: 400, 1: 400, 2: 400, 3: 400, 4: 400, 5: 400,
        6: 600, 7: 600, 8: 600, 9: 600,
        10: 500, 11: 500, 12: 500, 13: 500, 14: 500, 15: 500, 16: 500,
        17: 700, 18: 700, 19: 700, 20: 700, 21: 700,
        22: 450, 23: 450
    },
    "C2": {  # Cloud CPU
        0: 100, 1: 100, 2: 100, 3: 100, 4: 100, 5: 100,
        6: 225, 7: 225, 8: 225, 9: 225,
        10: 150, 11: 150, 12: 150, 13: 150, 14: 150, 15: 150, 16: 150,
        17: 275, 18: 275, 19: 275, 20: 275, 21: 275,
        22: 125, 23: 125
    },
    "C3": {  # Cloud HighMem
        0: 300, 1: 300, 2: 300, 3: 300, 4: 300, 5: 300,
        6: 475, 7: 475, 8: 475, 9: 475,
        10: 375, 11: 375, 12: 375, 13: 375, 14: 375, 15: 375, 16: 375,
        17: 550, 18: 550, 19: 550, 20: 550, 21: 550,
        22: 350, 23: 350
    },
    "C4": {  # Cloud Secure
        0: 250, 1: 250, 2: 250, 3: 250, 4: 250, 5: 250,
        6: 350, 7: 350, 8: 350, 9: 350,
        10: 300, 11: 300, 12: 300, 13: 300, 14: 300, 15: 300, 16: 300,
        17: 400, 18: 400, 19: 400, 20: 400, 21: 400,
        22: 275, 23: 275
    },
    "C5": {  # Cloud Ingest
        0: 75, 1: 75, 2: 75, 3: 75, 4: 75, 5: 75,
        6: 150, 7: 150, 8: 150, 9: 150,
        10: 110, 11: 110, 12: 110, 13: 110, 14: 110, 15: 110, 16: 110,
        17: 175, 18: 175, 19: 175, 20: 175, 21: 175,
        22: 90, 23: 90
    },
}


@dataclass
class Server:
    id: str
    server_type: str
    is_cloud: bool = False  # Cloud machines have infinite capacity but 5x cost


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
class MachineFailure:
    machine_id: str
    failure_start: int
    failure_end: int
    reason: str  # Description of why machine is unavailable


@dataclass
class Instance:
    servers: Dict[str, Server]
    jobs: Dict[str, Job]
    server_cost: Dict[str, Dict[int, float]]
    time_horizon: int
    weights: Dict[str, float]
    machine_failures: Optional[List[MachineFailure]] = None


def create_instance(
    scenario: str = "default",
    failure_scenario: Optional[str] = None
) -> Instance:
    """
    Create the problem instance.
    
    Args:
        scenario: Job scenario to load ('baseline', 'peak_load', 'emergency', 'light_load')
                 or 'default' to use hardcoded baseline
        failure_scenario: Machine failure scenario ('earthquake_moderate', 'tsunami_warning', etc.)
                         or None for no failures
    
    Returns:
        Instance object with jobs, servers, and optional machine failures
    """
    # Try to load from CSV if available
    if scenario != "default":
        try:
            from scenario_loader import create_instance_from_csv
            return create_instance_from_csv(scenario, failure_scenario)
        except (ImportError, FileNotFoundError) as e:
            print(f"Warning: Could not load scenario from CSV ({e}). Using default hardcoded instance.")
    
    # Fallback to hardcoded baseline (same as before)
    servers = {
        "M1a": Server("M1a", "GPU", is_cloud=False),
        "M1b": Server("M1b", "GPU", is_cloud=False),
        "M2a": Server("M2a", "CPU", is_cloud=False),
        "M2b": Server("M2b", "CPU", is_cloud=False),
        "M3a": Server("M3a", "HighMem", is_cloud=False),
        "M3b": Server("M3b", "HighMem", is_cloud=False),
        "M4": Server("M4", "Secure", is_cloud=False),
        "M5a": Server("M5a", "Ingest", is_cloud=False),
        "M5b": Server("M5b", "Ingest", is_cloud=False),
        # Cloud overflow machines (5x cost, infinite capacity)
        "C1": Server("C1", "GPU", is_cloud=True),
        "C2": Server("C2", "CPU", is_cloud=True),
        "C3": Server("C3", "HighMem", is_cloud=True),
        "C4": Server("C4", "Secure", is_cloud=True),
        "C5": Server("C5", "Ingest", is_cloud=True),
    }
    jobs = {
        "J1": Job("J1", "National census data processing", "HIGH", 12, 10.0, [
            Operation("Ingest", 2), Operation("HighMem", 3), Operation("CPU", 2)
        ]),
        "J2": Job("J2", "Tax fraud detection ML model", "HIGH", 10, 8.0, [
            Operation("Ingest", 1), Operation("GPU", 3), Operation("CPU", 1)
        ]),
        "J3": Job("J3", "Emergency response coordination", "CRITICAL", 6, 15.0, [
            Operation("Secure", 2), Operation("CPU", 1), Operation("HighMem", 1)
        ]),
        "J4": Job("J4", "Geospatial infrastructure analysis", "MEDIUM", 16, 5.0, [
            Operation("Ingest", 2), Operation("HighMem", 4)
        ]),
        "J5": Job("J5", "Classified intelligence processing", "CRITICAL", 8, 12.0, [
            Operation("Secure", 3), Operation("Secure", 2)
        ]),
        "J6": Job("J6", "Public health data aggregation", "MEDIUM", 18, 4.0, [
            Operation("Ingest", 1), Operation("CPU", 2), Operation("HighMem", 2)
        ]),
        "J7": Job("J7", "Climate model training", "LOW", 20, 2.0, [
            Operation("Ingest", 1), Operation("GPU", 4)
        ]),
        "J8": Job("J8", "Social services optimization", "MEDIUM", 14, 6.0, [
            Operation("Ingest", 1), Operation("CPU", 2), Operation("HighMem", 2)
        ]),
    }
    return Instance(servers, jobs, SERVER_COST, TIME_HORIZON,
                   {"compute_cost": 1.0, "delay_penalty": 5.0}, machine_failures=None)


# =============================================================================
# FITNESS EVALUATION
# =============================================================================

Schedule = Dict[str, List[Dict[str, Any]]]  # job_id -> [{server, start}, ...]
Fitness = Any  # namedtuple with score, compute_cost, delay_penalty, violations


def evaluate(inst: Instance, schedule: Schedule) -> Fitness:
    """Evaluate a schedule and return fitness metrics."""
    from collections import namedtuple
    
    FitnessTuple = namedtuple('Fitness', ['score', 'compute_cost', 'delay_penalty', 'violations', 'job_completion_times'])
    
    violations = []
    total_compute_cost = 0
    total_delay_penalty = 0
    job_completion_times = {}
    server_usage = {s: [] for s in inst.servers}
    
    # Build machine failure lookup for fast checking
    machine_unavailable = {}
    if inst.machine_failures:
        for failure in inst.machine_failures:
            if failure.machine_id not in machine_unavailable:
                machine_unavailable[failure.machine_id] = []
            machine_unavailable[failure.machine_id].append(
                (failure.failure_start, failure.failure_end)
            )

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

            # Check for machine failures during operation
            if server_id in machine_unavailable:
                for fail_start, fail_end in machine_unavailable[server_id]:
                    # Check if operation overlaps with failure window
                    if start < fail_end and end > fail_start:
                        violations.append(
                            f"{job_id}: op {idx} on {server_id} conflicts with machine failure "
                            f"(op: t={start}-{end}, failure: t={fail_start}-{fail_end})"
                        )

            server_usage[server_id].append((start, end, job_id, idx))
            
            # Direct cost lookup
            for t in range(start, end):
                total_compute_cost += inst.server_cost[server_id][t % inst.time_horizon]

        completion = prev_end if prev_end else 0
        job_completion_times[job_id] = completion
        delay = max(0, completion - job.deadline)
        total_delay_penalty += delay * job.delay_weight

    for server_id, intervals in server_usage.items():
        # Skip overlap check for cloud machines (infinite capacity)
        if inst.servers[server_id].is_cloud:
            continue
            
        intervals.sort()
        for i in range(len(intervals) - 1):
            _, end1, job1, _ = intervals[i]
            start2, _, job2, _ = intervals[i + 1]
            if start2 < end1:
                violations.append(f"{server_id}: overlap between {job1} and {job2}")

    score = (inst.weights["compute_cost"] * total_compute_cost +
             inst.weights["delay_penalty"] * total_delay_penalty +
             len(violations) * 10000)

    return FitnessTuple(round(score, 2), round(total_compute_cost, 2),
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
    priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
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
