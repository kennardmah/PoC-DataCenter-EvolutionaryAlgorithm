# EVOLVE-BLOCK-START
"""Constructor-based job-shop scheduling for government data center"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import copy
import os
import math # Added for exponential functions in get_available_capacity

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
    Constructs a schedule for jobs, prioritizing based on a multi-criteria heuristic
    and making local/cloud placement decisions by evaluating the full objective function
    (compute cost + weighted delay penalty).

    Args:
        inst: Problem instance with servers, jobs, server costs

    Returns:
        Schedule: Dict mapping job_id -> list of {"server": str, "start": int}
    """
    # math is already imported at the top of the EVOLVE-BLOCK

    def get_available_capacity(server_id: str, time: int) -> int:
        """
        Calculate available capacity at a specific time, considering machine failures.
        This is a proactive check, assuming the scheduler has knowledge of future failures.
        """
        base_capacity = inst.servers[server_id].capacity

        if not inst.machine_failures:
            return base_capacity

        total_reduction = 0
        for failure in inst.machine_failures:
            if failure.machine_id != server_id:
                continue

            # Before failure window
            if time < failure.failure_start:
                continue
            # After failure window
            elif time >= failure.failure_end:
                time_since_end = time - failure.failure_end
                recovery_factor = 1.0 - math.exp(-failure.recovery_rate * time_since_end)
                reduction = failure.capacity_reduction * (1.0 - recovery_factor)
                total_reduction += reduction
            # During failure window
            else:
                time_since_start = time - failure.failure_start
                failure_factor = 1.0 - math.exp(-failure.failure_rate * time_since_start)
                reduction = failure.capacity_reduction * failure_factor
                total_reduction += reduction

        return max(0, base_capacity - int(total_reduction))

    def has_capacity(server_id: str, start: int, duration: int, capacity_req: int) -> bool:
        """Check if server has enough capacity for the entire operation duration."""
        server = inst.servers[server_id]
        if server.is_cloud:
            return True # Cloud has effectively infinite capacity for scheduling purposes

        for t in range(start, start + duration):
            # If current time 't' is beyond the instance time horizon, it's invalid
            if t >= inst.time_horizon:
                return False
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

    # Pre-calculate total duration for each job and priority scores
    priority_map = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    for job_id, job in inst.jobs.items():
        job.total_duration = sum(op.duration for op in job.operations)
        job.priority_score = priority_map.get(job.priority, 99) # Lower score for higher priority

    # Pre-process server types for quick access
    server_types_map: Dict[str, List[str]] = {}
    cloud_servers_by_type: Dict[str, str] = {}
    for server_id, server in inst.servers.items():
        if server.server_type not in server_types_map:
            server_types_map[server.server_type] = []
        if server.is_cloud:
            cloud_servers_by_type[server.server_type] = server_id
        else:
            server_types_map[server.server_type].append(server_id)

    def _get_op_schedule_cost(server_id: str, start_time: int, op_duration: int, capacity_req: int, job: Job, current_op_index: int) -> float:
        """
        Calculates the estimated cost of scheduling an operation on a given server at a given start time.
        This includes compute cost and an estimate of future delay penalty for the entire job,
        aligning with the problem's total cost objective.
        It also includes a penalty for scheduling on failing machines.
        """
        # Calculate compute cost for this operation
        op_compute_cost = 0.0
        for t in range(start_time, start_time + op_duration):
            if t >= inst.time_horizon: # Ensure we don't calculate beyond horizon
                op_compute_cost += inst.server_cost[server_id][(inst.time_horizon - 1) % inst.time_horizon] # Use last known cost
            else:
                op_compute_cost += inst.server_cost[server_id][t % inst.time_horizon]

        # Estimate remaining duration of the job after this operation (excluding this op, including future ops)
        remaining_op_durations = sum(o.duration for o in job.operations[current_op_index + 1:])

        # Projected job completion time if this operation finishes at start_time + op_duration,
        # and subsequent operations follow immediately (optimistic assumption for lookahead).
        projected_job_completion_time = (start_time + op_duration) + remaining_op_durations

        # Calculate potential delay penalty for the entire job
        projected_delay = max(0, projected_job_completion_time - job.deadline)
        delay_penalty_contribution = projected_delay * job.delay_weight * inst.weights['delay_penalty']

        # Heuristic for proactive failure avoidance: penalize scheduling in failing machines
        failure_penalty = 0.0
        server_obj = inst.servers[server_id]
        if not server_obj.is_cloud and inst.machine_failures:
            for f in inst.machine_failures:
                if f.machine_id == server_id and f.failure_start < start_time + op_duration and f.failure_end > start_time:
                    # Calculate overlap duration
                    overlap_start = max(start_time, f.failure_start)
                    overlap_end = min(start_time + op_duration, f.failure_end)
                    if overlap_end > overlap_start:
                        # Penalty increases with capacity reduction, overlap duration, and critical operations
                        # Also scaled by the operation's capacity requirement relative to server's base capacity
                        # This aims to represent the risk of the operation being impacted or needing relocation
                        reduction_at_overlap = f.capacity_reduction * (1.0 - math.exp(-f.failure_rate * max(0, overlap_start - f.failure_start)))
                        failure_penalty += reduction_at_overlap * (op.capacity_required / server_obj.capacity) * inst.weights["delay_penalty"] * 1.5 # Adjusted multiplier

        # Total score to minimize for this operation's placement
        return (op_compute_cost * inst.weights['compute_cost']) + delay_penalty_contribution + failure_penalty


    # Prioritize jobs based on a multi-criteria heuristic:
    # 1. Job Priority (CRITICAL, HIGH, MEDIUM, LOW) - lower integer value is higher priority
    # 2. Effective deadline (deadline - total_duration - arrival_time) - tighter deadlines first
    # 3. Delay weight (higher delay weight first) - multiplied by -1 to sort descending
    # 4. Arrival time (earlier is better, for secondary tie-breaking)
    # 5. Job ID (for deterministic tie-breaking)
    sorted_jobs = sorted(inst.jobs.keys(),
                         key=lambda j_id: (
                             inst.jobs[j_id].priority_score,  # Lower priority_score first
                             inst.jobs[j_id].deadline - inst.jobs[j_id].arrival_time - inst.jobs[j_id].total_duration, # Smaller slack first
                             -inst.jobs[j_id].delay_weight, # Higher delay weight first (negate for ascending sort)
                             inst.jobs[j_id].arrival_time,  # Earlier arrival as tie-breaker
                             j_id  # Final tie-breaker
                         ))

    for job_id in sorted_jobs:
        job = inst.jobs[job_id]
        schedule[job_id] = []

        # `current_op_start_time` tracks the earliest an operation of THIS job can start,
        # considering its own arrival_time and preceding operations of the SAME job.
        current_op_start_time = job_available_time[job_id]

        for op_idx, op in enumerate(job.operations):
            best_total_objective_cost = float('inf')
            chosen_server = None
            chosen_start = None

            # 1. Evaluate local machines
            local_servers_for_type = server_types_map.get(op.server_type, [])
            # Filter out cloud servers from this list (they are at the end)
            local_servers_only = [s_id for s_id in local_servers_for_type if not inst.servers[s_id].is_cloud]

            for server_id in local_servers_only:
                # Iterate through possible start times up to the time horizon.
                # Starting from when the job/previous op makes this operation available.
                # Consider a lookahead window for non-critical jobs to find cheaper off-peak slots
                # For critical jobs, prioritize sooner availability, but still check a reasonable window.
                effective_search_window_end = inst.time_horizon
                # Small optimization: for CRITICAL/HIGH priority, if a slot is available
                # soon, we don't need to search too far ahead if it's already good.
                # However, for a complete cost optimization, we should still search.
                # Max 2-hour delay tolerance for a potentially cheaper slot for MEDIUM/LOW priority.
                if job.priority_score > priority_map["HIGH"]: # Medium/Low priority
                    effective_search_window_end = min(inst.time_horizon, current_op_start_time + 2) # Search 2 hours ahead

                for start_time in range(current_op_start_time, effective_search_window_end + 1):
                    if start_time >= inst.time_horizon:
                        break # Cannot schedule beyond time horizon

                    if has_capacity(server_id, start_time, op.duration, op.capacity_required):
                        current_cost_estimate = _get_op_schedule_cost(
                            server_id, start_time, op.duration, op.capacity_required, job, op_idx
                        )
                        if current_cost_estimate < best_total_objective_cost:
                            best_total_objective_cost = current_cost_estimate
                            chosen_server = server_id
                            chosen_start = start_time

            # 2. Evaluate cloud machines (always available immediately if needed)
            cloud_server_id = cloud_servers_by_type.get(op.server_type)
            if cloud_server_id:
                cloud_start_time = current_op_start_time # Cloud has infinite capacity, can always start here

                current_cloud_cost_estimate = _get_op_schedule_cost(
                    cloud_server_id, cloud_start_time, op.duration, op.capacity_required, job, op_idx
                )

                # Decision logic: choose local if it's the best option found so far,
                # or cloud if it's better than the best local found.
                if current_cloud_cost_estimate < best_total_objective_cost:
                    best_total_objective_cost = current_cloud_cost_estimate
                    chosen_server = cloud_server_id
                    chosen_start = cloud_start_time

            # 3. Schedule the chosen option
            if chosen_server is not None:
                # Reserve capacity for the chosen server and start time
                for t in range(chosen_start, chosen_start + op.duration):
                    if t >= inst.time_horizon: # Do not reserve capacity beyond the horizon
                        break
                    if t not in capacity_usage[chosen_server]:
                        capacity_usage[chosen_server][t] = 0
                    capacity_usage[chosen_server][t] += op.capacity_required

                schedule[job_id].append({"server": chosen_server, "start": chosen_start})
                current_op_start_time = chosen_start + op.duration
                job_available_time[job_id] = current_op_start_time # Update for next operations of this job
            else:
                # Fallback: This case should ideally not be reached if cloud is available for every type.
                # If it is reached, it implies a fundamental problem with server availability for a type.
                # For robustness, we will try to force schedule on first compatible cloud server (guaranteed capacity)
                # at the earliest possible time.
                if cloud_server_id:
                    fallback_server = cloud_server_id
                    fallback_start_time = current_op_start_time
                    for t in range(fallback_start_time, fallback_start_time + op.duration):
                        if t >= inst.time_horizon:
                            break
                        if t not in capacity_usage[fallback_server]:
                            capacity_usage[fallback_server][t] = 0
                        capacity_usage[fallback_server][t] += op.capacity_required
                    schedule[job_id].append({"server": fallback_server, "start": fallback_start_time})
                    job_available_time[job_id] = fallback_start_time + op.duration
                else:
                    # If even cloud isn't available (should not happen based on problem setup)
                    raise ValueError(f"No compatible server (local or cloud) found for job {job_id}, operation {op_idx} (type: {op.server_type})")

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