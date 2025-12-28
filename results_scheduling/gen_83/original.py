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
    arrival_time: int = 0  # Time when job becomes available for scheduling

    # These will be populated during schedule construction
    total_duration: int = 0
    priority_score: int = 0


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
    scenario: str = "baseline",
    failure_scenario: Optional[str] = None
) -> Instance:
    """
    Create the problem instance from CSV files.

    Args:
        scenario: Job scenario to load from CSV ('baseline', 'peak_load', 'emergency', 'light_load')
        failure_scenario: Machine failure scenario ('earthquake_moderate', 'tsunami_warning', etc.)
                         or None for no failures

    Returns:
        Instance object with jobs, servers, and optional machine failures from CSV
    """
    from helpers.scenario_loader import create_instance_from_csv
    return create_instance_from_csv(scenario, failure_scenario)


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
    cloud_relocations: List[str] = None

    def __post_init__(self):
        if self.cloud_relocations is None:
            self.cloud_relocations = []

    def is_valid(self) -> bool:
        return len(self.violations) == 0


def evaluate(inst: Instance, schedule: Schedule) -> Fitness:
    """
    Evaluate a schedule with capacity constraints and automatic cloud failover.

    When capacity violations occur (e.g., due to failures during job execution),
    affected operations are automatically relocated to cloud backup resources.
    This maintains validity but incurs higher costs (5x multiplier for cloud).
    """
    import math

    violations = []
    total_compute_cost = 0
    total_delay_penalty = 0
    job_completion_times = {}
    cloud_relocations = []  # Track which operations were moved to cloud

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

    def get_cloud_server(server_type: str) -> Optional[str]:
        """Find a cloud server of the given type."""
        for server_id, server in inst.servers.items():
            if server.is_cloud and server.server_type == server_type:
                return server_id
        return None

    # First pass: validate basic structure and track capacity usage
    for job_id, ops in schedule.items():
        job = inst.jobs[job_id]
        if len(ops) != len(job.operations):
            violations.append(f"{job_id}: wrong number of operations")
            continue

        # Check first operation doesn't start before arrival
        if ops and ops[0]["start"] < job.arrival_time:
            violations.append(f"{job_id}: starts at {ops[0]['start']} before arrival at {job.arrival_time}")

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

        completion = prev_end if prev_end else 0
        job_completion_times[job_id] = completion
        delay = max(0, completion - job.deadline)
        total_delay_penalty += delay * job.delay_weight

    # Second pass: detect capacity violations and relocate to cloud
    capacity_violations = {}  # (server_id, time) -> exceeded amount

    for server_id, usage_by_time in capacity_usage.items():
        # Skip cloud servers (they have unlimited capacity)
        if inst.servers[server_id].is_cloud:
            continue

        for time, used_capacity in usage_by_time.items():
            available_capacity = get_available_capacity(server_id, time)
            if used_capacity > available_capacity:
                excess = used_capacity - available_capacity
                capacity_violations[(server_id, time)] = excess

    # If violations detected, relocate affected operations to cloud
    if capacity_violations:
        # Find operations that overlap with violation times
        for job_id, ops in schedule.items():
            job = inst.jobs[job_id]
            for idx, op in enumerate(ops):
                server_id = op["server"]
                start = op["start"]
                duration = job.operations[idx].duration
                end = start + duration

                # Skip if already on cloud
                if inst.servers[server_id].is_cloud:
                    continue

                # Check if this operation overlaps with a capacity violation
                needs_relocation = False
                for t in range(start, end):
                    if (server_id, t) in capacity_violations:
                        needs_relocation = True
                        break

                if needs_relocation:
                    # Find cloud backup of same type
                    cloud_server = get_cloud_server(inst.servers[server_id].server_type)
                    if cloud_server:
                        # Remove from local capacity tracking
                        for t in range(start, end):
                            if t in capacity_usage[server_id]:
                                capacity_usage[server_id][t] -= job.operations[idx].capacity_required

                        # Add to cloud capacity tracking
                        for t in range(start, end):
                            if t not in capacity_usage[cloud_server]:
                                capacity_usage[cloud_server][t] = 0
                            capacity_usage[cloud_server][t] += job.operations[idx].capacity_required

                        # Update the schedule to reflect cloud relocation
                        op["server"] = cloud_server
                        op["relocated_to_cloud"] = True

                        cloud_relocations.append(
                            f"{job_id} op{idx}: {server_id} → {cloud_server} (failure during execution)"
                        )

    # Calculate costs (now including relocated operations on cloud)
    for job_id, ops in schedule.items():
        job = inst.jobs[job_id]
        for idx, op in enumerate(ops):
            server_id = op["server"]
            start = op["start"]
            duration = job.operations[idx].duration

            # Calculate cost
            for t in range(start, start + duration):
                total_compute_cost += inst.server_cost[server_id][t % inst.time_horizon]

    # Final validation: check if capacity violations were resolved
    for server_id, usage_by_time in capacity_usage.items():
        if inst.servers[server_id].is_cloud:
            continue  # Cloud has unlimited capacity

        for time, used_capacity in usage_by_time.items():
            available_capacity = get_available_capacity(server_id, time)
            if used_capacity > available_capacity:
                violations.append(
                    f"{server_id}: capacity exceeded at t={time} even after cloud relocation "
                    f"(used: {used_capacity}, available: {available_capacity})"
                )

    score = (inst.weights["compute_cost"] * total_compute_cost +
             inst.weights["delay_penalty"] * total_delay_penalty +
             len(violations) * 10000)

    return Fitness(round(score, 2), round(total_compute_cost, 2),
                   round(total_delay_penalty, 2), violations, job_completion_times,
                   cloud_relocations)


# =============================================================================
# SCHEDULE CONSTRUCTOR (TO BE EVOLVED)
# =============================================================================

def construct_schedule(inst: Instance) -> Schedule:
    """
    Schedules jobs with a priority-aware, greedy earliest-available-slot heuristic.
    It prioritizes jobs based on a composite urgency metric and attempts to schedule
    operations on local machines first for cost efficiency. If local capacity is
    insufficient or would cause significant delay, it falls back to cloud resources.

    Args:
        inst: Problem instance with servers, jobs, server costs

    Returns:
        Schedule: Dict mapping job_id -> list of {"server": str, "start": int}
    """
    import math

    def get_servers_by_type(server_type: str, prefer_local: bool = True) -> List[str]:
        """Get servers of a given type, optionally preferring local over cloud."""
        servers = [s.id for s in inst.servers.values() if s.server_type == server_type]
        if prefer_local:
            # Sort: local machines first, then cloud
            servers.sort(key=lambda s: inst.servers[s].is_cloud)
        return servers

    def get_available_capacity(server_id: str, time: int) -> int:
        """
        Calculate available capacity at a specific time, considering machine failures.
        """
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

        available = max(0, base_capacity - int(total_reduction))
        return available

    def has_capacity(server_id: str, start: int, duration: int, capacity_req: int) -> bool:
        """Check if server has enough capacity for the entire operation duration."""
        for t in range(start, start + duration):
            used = capacity_usage[server_id].get(t, 0)
            available = get_available_capacity(server_id, t)
            if used + capacity_req > available:
                return False
        return True

    # Track capacity usage per machine per timestep
    capacity_usage = {s: {} for s in inst.servers}

    # Track when each job can start (max of arrival time and previous operation finish time)
    job_available_time = {j: inst.jobs[j].arrival_time for j in inst.jobs}

    schedule = {}

    # Priority mapping for jobs (lower value = higher priority)
    PRIORITY_MAP = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}

    # Pre-calculate total duration and priority score for each job
    for job_id, job in inst.jobs.items():
        job.total_duration = sum(op.duration for op in job.operations)
        job.priority_score = PRIORITY_MAP.get(job.priority, 99) # Default to lowest priority

    # Prioritize jobs based on a multi-criteria heuristic:
    # 1. Job Priority (CRITICAL, HIGH, MEDIUM, LOW) - lower integer value is higher priority
    # 2. Effective deadline (deadline - total_duration - arrival_time) - tighter deadlines first
    # 3. Delay weight (higher delay weight first) - multiplied by -1 to sort descending
    # 4. Arrival time (earlier is better, for secondary tie-breaking)
    # 5. Job ID (for deterministic tie-breaking)
    sorted_jobs = sorted(inst.jobs.keys(),
                         key=lambda j_id: (
                             inst.jobs[j_id].priority_score,
                             inst.jobs[j_id].deadline - inst.jobs[j_id].total_duration - inst.jobs[j_id].arrival_time,
                             -inst.jobs[j_id].delay_weight, # Sort by higher delay weight first (negate for ascending sort)
                             inst.jobs[j_id].arrival_time,
                             j_id
                         ))

    for job_id in sorted_jobs:
        job = inst.jobs[job_id]
        schedule[job_id] = []

        current_op_start_time = job_available_time[job_id]

        for op_idx, op in enumerate(job.operations):
            # Get compatible servers (local first, then cloud)
            compatible_servers = get_servers_by_type(op.server_type, prefer_local=True)

            scheduled = False
            best_local_start = None
            best_local_server = None

            # Attempt to schedule on local machines first
            for server_id in compatible_servers:
                if inst.servers[server_id].is_cloud:
                    continue  # Skip cloud for now, only consider local

                # Find the earliest time this local server can accommodate the operation
                for start_time in range(current_op_start_time, inst.time_horizon):
                    if has_capacity(server_id, start_time, op.duration, op.capacity_required):
                        if best_local_start is None or start_time < best_local_start:
                            best_local_start = start_time
                            best_local_server = server_id
                        break  # Found earliest slot on this server, move to next local server if any
                               # or could potentially explore more to find cheapest, but sticking to earliest for now.

            # If a local slot was found, use it
            if best_local_server is not None:
                final_server = best_local_server
                final_start = best_local_start
                scheduled = True
            else:
                # No local machine available within time horizon from current_op_start_time.
                # Use cloud immediately to avoid delays.
                for server_id in compatible_servers:
                    if inst.servers[server_id].is_cloud:
                        final_server = server_id
                        final_start = current_op_start_time # Cloud has infinite capacity, schedule immediately
                        scheduled = True
                        break # Found a cloud server, no need to check others

            # Fallback for extreme cases (should not happen if cloud servers exist for all types)
            if not scheduled:
                # If no cloud server was found or accessible, which implies a problem with instance setup,
                # we'll still attempt to assign it to the first compatible server found, forcing it.
                # This ensures the schedule is always generated, though it might violate constraints.
                # The fitness function will then penalize for violations.
                final_server = compatible_servers[0] if compatible_servers else list(inst.servers.keys())[0] # Very last resort
                final_start = current_op_start_time
                # Note: this case is a last resort and likely indicates a severe issue if cloud isn't chosen.

            # Reserve capacity for the chosen server and start time
            for t in range(final_start, final_start + op.duration):
                if t not in capacity_usage[final_server]:
                    capacity_usage[final_server][t] = 0
                capacity_usage[final_server][t] += op.capacity_required

            schedule[job_id].append({"server": final_server, "start": final_start})
            current_op_start_time = final_start + op.duration
            job_available_time[job_id] = current_op_start_time # Update for next operations of this job

    return schedule


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_scheduling(
    scenario: str = "baseline",
    failure_scenario: Optional[str] = None,
    multi_scenario: bool = False,
    scenarios: Optional[List[str]] = None,
    scenario_weights: Optional[Dict[str, float]] = None,
    failure_scenarios: Optional[Dict[str, str]] = None
):
    """
    Run the schedule constructor and return results.

    Args:
        scenario: Job scenario to load from CSV (for single-scenario mode)
        failure_scenario: Machine failure scenario from CSV or None (for single-scenario mode)
        multi_scenario: If True, evaluate across multiple scenarios
        scenarios: List of scenario names to evaluate (for multi-scenario mode)
        scenario_weights: Weight for each scenario in final score (for multi-scenario mode)
        failure_scenarios: Mapping of scenario name to failure scenario (for multi-scenario mode)
    """
    from helpers.scenario_loader import create_instance_from_csv

    if multi_scenario and scenarios:
        # Multi-scenario evaluation
        results = {}
        weighted_score = 0.0

        for scenario_name in scenarios:
            # Get failure scenario for this scenario (if any)
            fail_scenario = None
            if failure_scenarios and scenario_name in failure_scenarios:
                fail_scenario = failure_scenarios[scenario_name]

            # Create instance and run scheduling
            inst = create_instance_from_csv(scenario=scenario_name, failure_scenario=fail_scenario)
            schedule = construct_schedule(inst)
            fitness = evaluate(inst, schedule)
            score = float(fitness.score)

            # Store results
            results[scenario_name] = (schedule, fitness, score)

            # Add weighted contribution
            weight = scenario_weights.get(scenario_name, 1.0) if scenario_weights else 1.0
            weighted_score += weight * score

        # Return multi-scenario results
        return {
            "scenarios": results,
            "weighted_score": weighted_score,
            "weights": scenario_weights or {s: 1.0 for s in scenarios}
        }
    else:
        # Single scenario evaluation (backward compatibility)
        inst = create_instance_from_csv(scenario=scenario, failure_scenario=failure_scenario)
        schedule = construct_schedule(inst)
        fitness = evaluate(inst, schedule)
        return schedule, fitness, float(fitness.score)