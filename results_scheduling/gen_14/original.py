# EVOLVE-BLOCK-START
"""Constructor-based job-shop scheduling for government data center"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import copy
import os
import math # Make sure math is imported for exponential functions


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
    # math is imported at the top level, so no need to import again

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

class Scheduler:
    def __init__(self, inst: Instance):
        self.inst = inst
        self.capacity_usage: Dict[str, Dict[int, int]] = {s: {} for s in inst.servers}
        self.job_available_time: Dict[str, int] = {j: inst.jobs[j].arrival_time for j in inst.jobs}
        self.schedule: Schedule = {}

        # Precompute server types and cloud servers
        self.server_types_map: Dict[str, List[str]] = {}
        self.cloud_servers_by_type: Dict[str, str] = {}
        for server_id, server in inst.servers.items():
            if server.server_type not in self.server_types_map:
                self.server_types_map[server.server_type] = []
            if server.is_cloud:
                self.cloud_servers_by_type[server.server_type] = server_id
            else:
                self.server_types_map[server.server_type].append(server_id)
        # Add cloud servers to the list of compatible servers, always at the end
        for s_type in self.server_types_map:
            if s_type in self.cloud_servers_by_type:
                self.server_types_map[s_type].append(self.cloud_servers_by_type[s_type])

        # Precompute job total durations and priority values
        self.job_total_durations: Dict[str, int] = {
            job_id: sum(op.duration for op in job.operations)
            for job_id, job in inst.jobs.items()
        }

        self.priority_values = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}

    def _get_current_available_capacity(self, server_id: str, time: int) -> int:
        """
        Calculate available capacity at a specific time, considering machine failures.
        This is a proactive check, assuming the scheduler has knowledge of future failures.
        """
        base_capacity = self.inst.servers[server_id].capacity

        if self.inst.machine_failures is None:
            return base_capacity

        total_reduction = 0
        for failure in self.inst.machine_failures:
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

    def _has_capacity_for_op(self, server_id: str, start: int, duration: int, capacity_req: int) -> bool:
        """Check if server has enough capacity for the entire operation duration."""
        server = self.inst.servers[server_id]
        if server.is_cloud:
            return True # Cloud has effectively infinite capacity for scheduling purposes

        for t in range(start, start + duration):
            # If current time 't' is beyond the instance time horizon, it's invalid
            if t >= self.inst.time_horizon:
                return False
            used = self.capacity_usage[server_id].get(t, 0)
            available = self._get_current_available_capacity(server_id, t)
            if used + capacity_req > available:
                return False
        return True

    def _reserve_capacity(self, server_id: str, start: int, duration: int, capacity_req: int):
        """Reserve capacity for an operation."""
        for t in range(start, start + duration):
            # Ensure not to reserve beyond the time horizon
            if t >= self.inst.time_horizon:
                break
            if t not in self.capacity_usage[server_id]:
                self.capacity_usage[server_id][t] = 0
            self.capacity_usage[server_id][t] += capacity_req

    def _calculate_operation_cost(self, server_id: str, start: int, duration: int) -> float:
        """Calculate the compute cost for an operation."""
        cost = 0.0
        for t in range(start, start + duration):
            # Ensure not to calculate cost beyond the time horizon
            if t >= self.inst.time_horizon:
                break
            cost += self.inst.server_cost[server_id][t % self.inst.time_horizon]
        return cost

    def _calculate_job_urgency(self, job_id: str, current_time: int) -> float:
        job = self.inst.jobs[job_id]
        priority_factor = self.priority_values.get(job.priority, 1) * 100 # Higher value for higher priority

        # Remaining duration for the job from its current operation
        scheduled_ops_count = len(self.schedule.get(job_id, []))
        if scheduled_ops_count >= len(job.operations):
            return -float('inf') # Job is already fully scheduled

        remaining_duration = sum(op.duration for op in job.operations[scheduled_ops_count:])

        # Heuristic for earliest possible completion if scheduled immediately
        earliest_possible_end = max(current_time, job.arrival_time) + remaining_duration

        # Slack: positive means time to spare, negative means overdue
        slack = job.deadline - earliest_possible_end

        # Urgency is higher for lower (more negative) slack.
        # Maximize (priority_factor - slack * delay_weight_factor)
        # Using a higher priority factor will make it more dominant.
        return priority_factor - (slack * job.delay_weight)


    def _find_best_slot_for_operation(self, job: Job, op: Operation, earliest_start: int) -> Dict[str, Any]:
        compatible_servers = self.server_types_map.get(op.server_type, [])

        best_server = None
        best_start_time = -1
        min_total_weighted_score = float('inf')

        cloud_server_id = self.cloud_servers_by_type.get(op.server_type)

        # Define a flexible search window based on job priority.
        # Higher priority jobs get a shorter lookahead to prioritize immediate placement,
        # lower priority jobs get a longer lookahead to find cheaper slots.
        lookahead_window = self.inst.time_horizon # Default to full horizon for maximum cost optimization
        if job.priority == "CRITICAL":
            lookahead_window = 4  # Prioritize immediate placement for CRITICAL
        elif job.priority == "HIGH":
            lookahead_window = 8  # Moderate lookahead for HIGH
        elif job.priority == "MEDIUM":
            lookahead_window = 12 # Longer lookahead for MEDIUM

        search_window_end = min(self.inst.time_horizon, earliest_start + lookahead_window + 1)

        # First, try to find a local slot
        for server_id in compatible_servers:
            server = self.inst.servers[server_id]
            if server.is_cloud:
                continue # Skip cloud in this pass, focus on local

            for start_time in range(earliest_start, search_window_end):
                if not self._has_capacity_for_op(server_id, start_time, op.duration, op.capacity_required):
                    continue

                op_compute_cost = self._calculate_operation_cost(server_id, start_time, op.duration)
                potential_completion_time = start_time + op.duration

                # Estimate remaining duration of the job after this operation (excluding this op duration)
                scheduled_ops_count = len(self.schedule.get(job.id, []))
                remaining_op_durations = sum(o.duration for o in job.operations[scheduled_ops_count + 1:])
                projected_job_completion_time = potential_completion_time + remaining_op_durations

                delay = max(0, projected_job_completion_time - job.deadline)
                delay_penalty = delay * job.delay_weight * self.inst.weights["delay_penalty"]

                # Heuristic for proactive failure avoidance: penalize scheduling in failing machines
                failure_penalty = 0
                if self.inst.machine_failures:
                    for f in self.inst.machine_failures:
                        if f.machine_id == server_id and f.failure_start <= start_time < f.failure_end:
                            # Calculate overlap duration
                            overlap_start = max(start_time, f.failure_start)
                            overlap_end = min(start_time + op.duration, f.failure_end)
                            failure_overlap_duration = max(0, overlap_end - overlap_start)

                            if failure_overlap_duration > 0:
                                # Apply a penalty proportional to the capacity reduction and overlap
                                # Using `op.capacity_required / server.capacity` to gauge the impact on this specific operation
                                # Multiplying by delay_penalty weight ensures it's aligned with the objective's cost structure
                                # A multiplier (e.g., 5 or 10) can make failure avoidance more aggressive.
                                impact_factor = (op.capacity_required / server.capacity) * failure_overlap_duration
                                failure_penalty += f.capacity_reduction * impact_factor * self.inst.weights["delay_penalty"] * 5 # Aggressive multiplier

                total_weighted_score = (op_compute_cost * self.inst.weights["compute_cost"]) + delay_penalty + failure_penalty

                if total_weighted_score < min_total_weighted_score:
                    min_total_weighted_score = total_weighted_score
                    best_server = server_id
                    best_start_time = start_time

        # Calculate cloud cost for immediate start (or earliest_start if already delayed)
        cloud_total_score = float('inf')
        if cloud_server_id:
            cloud_op_compute_cost = self._calculate_operation_cost(cloud_server_id, earliest_start, op.duration)
            cloud_potential_completion_time = earliest_start + op.duration

            # Estimate remaining duration of the job after this operation (excluding this op duration)
            scheduled_ops_count = len(self.schedule.get(job.id, []))
            remaining_op_durations = sum(o.duration for o in job.operations[scheduled_ops_count + 1:])
            cloud_projected_job_completion_time = cloud_potential_completion_time + remaining_op_durations

            cloud_delay = max(0, cloud_projected_job_completion_time - job.deadline)
            cloud_delay_penalty = cloud_delay * job.delay_weight * self.inst.weights["delay_penalty"]

            cloud_total_score = (cloud_op_compute_cost * self.inst.weights["compute_cost"]) + cloud_delay_penalty # No failure penalty for cloud (infinite capacity)

        # Decision logic:
        # If no local option was found or if cloud is significantly better considering urgency

        # Default: cloud must be better (lower score) to be preferred over local.
        cloud_preference_threshold = 1.0 # Cloud must be 'better' than local (lower score)
        if job.priority == "CRITICAL" or job.priority == "HIGH":
            # For high priority jobs, we might accept a slightly worse cloud option (e.g., up to 10% more expensive)
            # if the best local option is considerably worse, or if local is unavailable.
            # This allows cloud to be a viable alternative to avoid critical delays even if local is marginally cheaper.
            # Or, more simply, if current local best score is significantly worse than cloud, prefer cloud.
            if best_server is not None and min_total_weighted_score > cloud_total_score * 1.1: # if local is 10% worse
                 cloud_preference_threshold = 0.9 # Allows cloud to be slightly more expensive, but still good.
            else:
                 cloud_preference_threshold = 1.0 # Cloud must be strictly better or equal.

        # If no local server was found, or if a cloud option is better (lower score) than the best local one,
        # considering the cloud preference threshold.
        # Note: If best_server is None, min_total_weighted_score is float('inf'), so cloud will always be preferred if available.
        if best_server is None or (cloud_server_id and cloud_total_score < min_total_weighted_score * cloud_preference_threshold):
            # Prioritize cloud if it offers a better or comparable score, especially for high-priority jobs.
            # Ensure cloud_server_id is valid before returning.
            if cloud_server_id:
                return {"server": cloud_server_id, "start": earliest_start}
            elif best_server: # If no cloud, but local was found (e.g. cloud_server_id was None)
                return {"server": best_server, "start": best_start_time}
            else: # Absolutely no option found (should ideally not happen if cloud is always present for a type)
                raise ValueError(f"No valid slot found for job {job.id}, operation {op.server_type} (earliest_start: {earliest_start})")
        else:
            return {"server": best_server, "start": best_start_time}


    def schedule_all_jobs(self) -> Schedule:
        unscheduled_jobs = list(self.inst.jobs.keys())

        while unscheduled_jobs:
            # Determine the current scheduling time by finding the earliest 'job_available_time' among unscheduled jobs.
            # This ensures we process events (job arrivals or previous op completions) chronologically.
            current_scheduling_time = self.inst.time_horizon + 1 # Initialize with a time beyond horizon
            if unscheduled_jobs:
                current_scheduling_time = min(
                    self.job_available_time[j_id]
                    for j_id in unscheduled_jobs
                    if len(self.schedule.get(j_id, [])) < len(self.inst.jobs[j_id].operations)
                )

            # Filter jobs that are ready to be scheduled at or before current_scheduling_time
            # and still have unscheduled operations.
            ready_jobs = [
                j_id for j_id in unscheduled_jobs
                if self.job_available_time[j_id] <= current_scheduling_time
                and len(self.schedule.get(j_id, [])) < len(self.inst.jobs[j_id].operations)
            ]

            if not ready_jobs:
                # If no jobs are ready at current_scheduling_time, it means all remaining unscheduled jobs
                # have dependencies or arrival times later than current_scheduling_time.
                # Advance current_scheduling_time to the next earliest job_available_time.
                next_event_time = min(
                    self.job_available_time[j_id]
                    for j_id in unscheduled_jobs
                    if len(self.schedule.get(j_id, [])) < len(self.inst.jobs[j_id].operations)
                ) if unscheduled_jobs else self.inst.time_horizon + 1

                if next_event_time > current_scheduling_time and next_event_time <= self.inst.time_horizon:
                    current_scheduling_time = next_event_time
                    continue # Re-evaluate ready jobs at the new time
                else:
                    break # No more jobs can be scheduled within the horizon

            # Sort ready jobs based on urgency at the current_scheduling_time (highest urgency first)
            sorted_ready_jobs = sorted(
                ready_jobs,
                key=lambda j_id: self._calculate_job_urgency(j_id, current_scheduling_time),
                reverse=True
            )

            job_id_to_schedule = sorted_ready_jobs[0]
            job = self.inst.jobs[job_id_to_schedule]

            self.schedule[job_id_to_schedule] = self.schedule.get(job_id_to_schedule, [])

            # Schedule the next operation for this job
            op_idx = len(self.schedule[job_id_to_schedule])
            current_op = job.operations[op_idx]

            # The earliest this operation can start is its job's arrival time,
            # or the completion time of its previous operation (if any),
            # or the current scheduling time derived from job_available_time.
            earliest_start_for_op = max(job.arrival_time, self.job_available_time[job_id_to_schedule])

            scheduled_op_info = self._find_best_slot_for_operation(job, current_op, earliest_start_for_op)

            server_id = scheduled_op_info["server"]
            start_time = scheduled_op_info["start"]

            self._reserve_capacity(server_id, start_time, current_op.duration, current_op.capacity_required)
            self.schedule[job_id_to_schedule].append({"server": server_id, "start": start_time})
            self.job_available_time[job_id_to_schedule] = start_time + current_op.duration

            if len(self.schedule[job_id_to_schedule]) == len(job.operations):
                unscheduled_jobs.remove(job_id_to_schedule)

        return self.schedule


def construct_schedule(inst: Instance) -> Schedule:
    """
    Constructs a schedule using a Priority-Aware Adaptive Scheduler.

    This scheduler prioritizes jobs based on urgency (deadline, priority, delay weight),
    and makes intelligent local/cloud allocation decisions considering cost,
    potential delays, and machine failure forecasts.
    """
    scheduler = Scheduler(inst)
    return scheduler.schedule_all_jobs()


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