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
    capacity: int  # Available capacity units
    is_cloud: bool = False  # Cloud machines have infinite capacity but 5x cost


@dataclass
class Operation:
    server_type: str
    duration: int
    capacity_required: int = 1  # Capacity units consumed during execution


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
    capacity_reduction: int  # Amount of capacity lost during failure
    failure_rate: float = 1.0  # Exponential failure rate (1.0 = instant, <1.0 = gradual)
    recovery_rate: float = 1.0  # Exponential recovery rate (1.0 = instant, <1.0 = gradual)
    reason: str = ""  # Description of failure


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
        # Local machines with capacity
        "M1": Server("M1", "GPU", capacity=20, is_cloud=False),
        "M2": Server("M2", "CPU", capacity=40, is_cloud=False),
        "M3": Server("M3", "HighMem", capacity=10, is_cloud=False),
        "M4": Server("M4", "Secure", capacity=5, is_cloud=False),
        "M5": Server("M5", "Ingest", capacity=30, is_cloud=False),
        # Cloud overflow machines (infinite capacity, 5x cost)
        "C1": Server("C1", "GPU", capacity=999999, is_cloud=True),
        "C2": Server("C2", "CPU", capacity=999999, is_cloud=True),
        "C3": Server("C3", "HighMem", capacity=999999, is_cloud=True),
        "C4": Server("C4", "Secure", capacity=999999, is_cloud=True),
        "C5": Server("C5", "Ingest", capacity=999999, is_cloud=True),
    }
    jobs = {
        "J1": Job("J1", "National census data processing", "HIGH", 12, 10.0, [
            Operation("Ingest", 2, 5), Operation("HighMem", 3, 3), Operation("CPU", 2, 4)
        ]),
        "J2": Job("J2", "Tax fraud detection ML model", "HIGH", 10, 8.0, [
            Operation("Ingest", 1, 3), Operation("GPU", 3, 8), Operation("CPU", 1, 2)
        ]),
        "J3": Job("J3", "Emergency response coordination", "CRITICAL", 6, 15.0, [
            Operation("Secure", 2, 2), Operation("CPU", 1, 3), Operation("HighMem", 1, 2)
        ]),
        "J4": Job("J4", "Geospatial infrastructure analysis", "MEDIUM", 16, 5.0, [
            Operation("Ingest", 2, 4), Operation("HighMem", 4, 5)
        ]),
        "J5": Job("J5", "Classified intelligence processing", "CRITICAL", 8, 12.0, [
            Operation("Secure", 3, 3), Operation("Secure", 2, 2)
        ]),
        "J6": Job("J6", "Public health data aggregation", "MEDIUM", 18, 4.0, [
            Operation("Ingest", 1, 2), Operation("CPU", 2, 5), Operation("HighMem", 2, 3)
        ]),
        "J7": Job("J7", "Climate model training", "LOW", 20, 2.0, [
            Operation("Ingest", 1, 3), Operation("GPU", 4, 10)
        ]),
        "J8": Job("J8", "Social services optimization", "MEDIUM", 14, 6.0, [
            Operation("Ingest", 1, 3), Operation("CPU", 2, 6), Operation("HighMem", 2, 4)
        ]),
    }
    return Instance(servers, jobs, SERVER_COST, TIME_HORIZON,
                   {"compute_cost": 1.0, "delay_penalty": 5.0}, machine_failures=None)


# =============================================================================
# FITNESS EVALUATION
# =============================================================================

Schedule = Dict[str, List[Dict[str, Any]]]  # job_id -> [{server, start}, ...]


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
    """Evaluate a schedule with capacity constraints and exponential failure model."""
    import math

    # Track capacity usage per machine per timestep
    # capacity_usage[server_id][time] = total capacity used
    capacity_usage = {s: {} for s in inst.servers}

    # Helper function to calculate available capacity with exponential failure/recovery
    def get_available_capacity(server_id: str, time: int) -> int:
        """Calculate available capacity considering exponential failure/recovery."""
        base_capacity = inst.servers[server_id].capacity

        if not inst.machine_failures:
            return base_capacity

        # Apply all failures affecting this machine at this time
        total_reduction = 0
        for failure in inst.machine_failures:
            if failure.machine_id != server_id:
                continue

            # Before failure window
            if time < failure.failure_start:
                continue
            # After failure window
            elif time >= failure.failure_end:
                # Exponential recovery
                time_since_end = time - failure.failure_end
                recovery_factor = 1.0 - math.exp(-failure.recovery_rate * time_since_end)
                reduction = failure.capacity_reduction * (1.0 - recovery_factor)
                total_reduction += reduction
            # During failure window
            else:
                # Exponential failure onset
                time_since_start = time - failure.failure_start
                failure_factor = 1.0 - math.exp(-failure.failure_rate * time_since_start)
                reduction = failure.capacity_reduction * failure_factor
                total_reduction += reduction

        available = max(0, base_capacity - int(total_reduction))
        return available

    # Validate schedule
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
            capacity_req = job.operations[idx].capacity_required
            end = start + duration

            if prev_end is not None and start < prev_end:
                violations.append(f"{job_id}: op {idx} starts before previous ends")
            prev_end = end

            required_type = job.operations[idx].server_type
            actual_type = inst.servers[server_id].server_type
            if required_type != actual_type:
                violations.append(f"{job_id}: op {idx} requires {required_type}, got {actual_type}")

            # Track capacity usage for each timestep
            for t in range(start, end):
                if t not in capacity_usage[server_id]:
                    capacity_usage[server_id][t] = 0
                capacity_usage[server_id][t] += capacity_req

            # Calculate cost
            for t in range(start, end):
                total_compute_cost += inst.server_cost[server_id][t % inst.time_horizon]

        completion = prev_end if prev_end else 0
        job_completion_times[job_id] = completion
        delay = max(0, completion - job.deadline)
        total_delay_penalty += delay * job.delay_weight

    # Check capacity constraints for each machine at each timestep
    for server_id, usage_by_time in capacity_usage.items():
        for time, used_capacity in usage_by_time.items():
            available_capacity = get_available_capacity(server_id, time)
            if used_capacity > available_capacity:
                violations.append(
                    f"{server_id}: capacity exceeded at t={time} "
                    f"(used: {used_capacity}, available: {available_capacity})"
                )

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
    Cost-aware, slack-based scheduler with look-ahead and cloud trade-off.
    - Orders jobs by priority, lateness risk (negative slack), deadline, and penalty weight.
    - For each operation, scans a 24h window from earliest feasible start across local and cloud
      and selects (server, start) minimizing: compute_cost + weights['delay_penalty'] × delay_penalty_est.
    - Prefers local on ties, shifts into off-peak when slack allows, and respects failures/capacity.
    """
    import math

    def get_servers_by_type(server_type: str, prefer_local: bool = True) -> List[str]:
        """Get servers of a given type, optionally preferring local over cloud."""
        servers = [s.id for s in inst.servers.values() if s.server_type == server_type]
        if prefer_local:
            servers.sort(key=lambda s: inst.servers[s].is_cloud)  # local first
        return servers

    def get_available_capacity(server_id: str, time: int) -> int:
        """Calculate available capacity considering exponential failure/recovery."""
        base_capacity = inst.servers[server_id].capacity
        if not inst.machine_failures:
            return base_capacity

        total_reduction = 0
        for failure in inst.machine_failures:
            if failure.machine_id != server_id:
                continue
            if time < failure.failure_start:
                continue
            elif time >= failure.failure_end:
                time_since_end = time - failure.failure_end
                recovery_factor = 1.0 - math.exp(-failure.recovery_rate * time_since_end)
                reduction = failure.capacity_reduction * (1.0 - recovery_factor)
                total_reduction += reduction
            else:
                time_since_start = time - failure.failure_start
                failure_factor = 1.0 - math.exp(-failure.failure_rate * time_since_start)
                reduction = failure.capacity_reduction * failure_factor
                total_reduction += reduction

        return max(0, base_capacity - int(total_reduction))

    def has_capacity(server_id: str, start: int, duration: int, capacity_req: int) -> bool:
        """Check if server has enough capacity for the entire operation duration."""
        for t in range(start, start + duration):
            used = capacity_usage[server_id].get(t, 0)
            available = get_available_capacity(server_id, t)
            if used + capacity_req > available:
                return False
        return True

    def op_cost(server_id: str, start: int, duration: int) -> float:
        """Compute direct compute cost for running [start, start+duration) on server_id."""
        H = inst.time_horizon
        return sum(inst.server_cost[server_id][(t % H)] for t in range(start, start + duration))

    # Track capacity usage per machine per timestep
    capacity_usage = {s: {} for s in inst.servers}
    # Track when each job can start (after its previous operation finishes)
    job_available_time = {j: 0 for j in inst.jobs}
    schedule: Schedule = {}

    # Priority and smarter ordering: priority -> lateness risk (negative slack) -> deadline -> higher penalty weight
    priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    def job_sort_key(jid: str):
        job = inst.jobs[jid]
        total_pt = sum(op.duration for op in job.operations)
        slack = job.deadline - total_pt
        neg_slack_flag = 0 if slack < 0 else 1  # negative slack first
        return (priority_order[job.priority], neg_slack_flag, job.deadline, -job.delay_weight, jid)

    sorted_jobs = sorted(inst.jobs.keys(), key=job_sort_key)

    # Build the schedule
    for job_id in sorted_jobs:
        job = inst.jobs[job_id]
        schedule[job_id] = []
        n_ops = len(job.operations)

        for idx, oper in enumerate(job.operations):
            compatible = get_servers_by_type(oper.server_type, prefer_local=True)
            earliest_start = job_available_time[job_id]
            remaining_after = sum(job.operations[k].duration for k in range(idx + 1, n_ops))

            # Look-ahead window: explore next 24 hours from earliest feasible start
            lookahead = 24
            time_window_end = earliest_start + lookahead

            best_choice = None  # tuple(order_value, is_cloud, finish_est, comp_cost, server_id, start_time)
            for server_id in compatible:
                is_cloud = inst.servers[server_id].is_cloud
                for start_time in range(earliest_start, time_window_end + 1):
                    # Capacity check (cloud effectively huge capacity)
                    if not is_cloud and not has_capacity(server_id, start_time, oper.duration, oper.capacity_required):
                        continue

                    comp_cost = op_cost(server_id, start_time, oper.duration)
                    finish_est = start_time + oper.duration + remaining_after
                    # Estimated delay penalty (forward-looking)
                    delay_hours = max(0, finish_est - job.deadline)
                    penalty_est = delay_hours * job.delay_weight * inst.weights["delay_penalty"]

                    # Objective blends compute cost and estimated delay (compute_cost weight is 1.0)
                    objective = comp_cost * inst.weights["compute_cost"] + penalty_est

                    candidate = (objective, is_cloud, finish_est, comp_cost, server_id, start_time)
                    if best_choice is None or candidate < best_choice:
                        best_choice = candidate

            # Fallback strategies if nothing found in window (should be rare):
            if best_choice is None:
                # Try cloud at earliest_start
                cloud_candidates = [sid for sid in compatible if inst.servers[sid].is_cloud]
                if cloud_candidates:
                    sid = cloud_candidates[0]
                    best_choice = (op_cost(sid, earliest_start, oper.duration),
                                   True,
                                   earliest_start + oper.duration + remaining_after,
                                   op_cost(sid, earliest_start, oper.duration),
                                   sid,
                                   earliest_start)
                else:
                    # Place on local at the earliest feasible time we can find by expanding search
                    sid = compatible[0]
                    start_time = earliest_start
                    # Expand search up to +48 hours if needed
                    found = False
                    for t in range(earliest_start, earliest_start + 48):
                        if has_capacity(sid, t, oper.duration, oper.capacity_required):
                            start_time = t
                            found = True
                            break
                    if not found:
                        # As last resort, place at earliest_start ignoring capacity (avoid undefined schedule)
                        start_time = earliest_start
                    comp_cost = op_cost(sid, start_time, oper.duration)
                    best_choice = (comp_cost, False, start_time + oper.duration + remaining_after, comp_cost, sid, start_time)

            _, is_cloud_sel, _, comp_cost_sel, server_sel, start_sel = best_choice

            # Reserve capacity and commit schedule
            for t in range(start_sel, start_sel + oper.duration):
                if t not in capacity_usage[server_sel]:
                    capacity_usage[server_sel][t] = 0
                capacity_usage[server_sel][t] += oper.capacity_required

            schedule[job_id].append({"server": server_sel, "start": start_sel})
            job_available_time[job_id] = start_sel + oper.duration

    return schedule


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_scheduling():
    """Run the schedule constructor and return results."""
    inst = create_instance()
    schedule = construct_schedule(inst)
    fitness = evaluate(inst, schedule)
    return schedule, fitness, float(fitness.score)