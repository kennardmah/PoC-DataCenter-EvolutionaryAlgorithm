# EVOLVE-BLOCK-START
"""Constructor-based job-shop scheduling for government data center"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import copy
import os
import math


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
    and making local/cloud placement decisions by finding the earliest available slot.

    Args:
        inst: Problem instance with servers, jobs, server costs

    Returns:
        Schedule: Dict mapping job_id -> list of {"server": str, "start": int}
    """
    # Define the Scheduler class internally to encapsulate logic and state
    class Scheduler:
        def __init__(self, inst: Instance):
            self.inst = inst
            self.capacity_usage = {s_id: {} for s_id in inst.servers}
            self.job_available_time = {j_id: inst.jobs[j_id].arrival_time for j_id in inst.jobs}
            self.schedule = {}

            # Pre-calculate job properties for sorting
            self.priority_map = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
            for job_id, job in self.inst.jobs.items():
                job.total_duration = sum(op.duration for op in job.operations)
                job.priority_score = self.priority_map.get(job.priority, 99)

        def _get_available_capacity(self, server_id: str, time: int) -> int:
            """Calculate available capacity at a specific time, considering failures."""
            base_capacity = self.inst.servers[server_id].capacity
            if not self.inst.machine_failures:
                return base_capacity

            total_reduction = 0
            for failure in self.inst.machine_failures:
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

        def _get_current_used_capacity(self, server_id: str, time: int) -> int:
            """Get current capacity used on a server at a specific time."""
            return self.capacity_usage[server_id].get(time, 0)

        def _has_capacity(self, server_id: str, start: int, duration: int, capacity_req: int) -> bool:
            """Check if server has enough capacity for the entire operation duration."""
            for t in range(start, start + duration):
                used = self._get_current_used_capacity(server_id, t)
                available = self._get_available_capacity(server_id, t)
                if used + capacity_req > available:
                    return False
            return True

        def _calculate_min_free_capacity_during_op(self, server_id: str, start: int, duration: int, capacity_req: int) -> int:
            """
            Calculates the minimum remaining free capacity on a server during an operation's execution,
            assuming the operation is placed there. A higher value indicates more resilience.
            """
            min_free_capacity = float('inf')
            for t in range(start, start + duration):
                used_at_t = self._get_current_used_capacity(server_id, t)
                available_at_t = self._get_available_capacity(server_id, t)
                remaining_after_op = available_at_t - (used_at_t + capacity_req)
                min_free_capacity = min(min_free_capacity, remaining_after_op)
            return int(min_free_capacity)

        def _get_servers_by_type_sorted(self, server_type: str, current_time: int) -> List[str]:
            """
            Get servers of a given type, sorted by capacity health (less degradation)
            and then by current hourly cost. Local servers always preferred.
            """
            compatible_local = []
            compatible_cloud = []
            for s_id, server in self.inst.servers.items():
                if server.server_type == server_type:
                    if server.is_cloud:
                        compatible_cloud.append(s_id)
                    else:
                        compatible_local.append(s_id)

            # Use min(current_time, self.inst.time_horizon - 1) for cost_t to avoid index error
            local_cost_time = min(current_time, self.inst.time_horizon - 1)

            # Sort local: Higher available capacity first, then lower cost
            compatible_local.sort(key=lambda s_id: (
                -self._get_available_capacity(s_id, current_time),  # Negate for descending sort (more capacity is better)
                self.inst.server_cost[s_id][local_cost_time % self.inst.time_horizon]
            ))

            # Sort cloud: Lower cost first (capacity is infinite)
            compatible_cloud.sort(key=lambda s_id: self.inst.server_cost[s_id][local_cost_time % self.inst.time_horizon])

            # Prioritize local over cloud for initial search order
            return compatible_local + compatible_cloud

        # Class attributes for search window configuration
        BASE_LOCAL_SEARCH_WINDOW_MULTIPLIER = 1.5 # Base search up to 1.5 full time horizons (e.g., 36 hours) for local slots
        BASE_LOCAL_SEARCH_WINDOW_ADD = 5 # Small buffer for local scheduling slightly past deadline/horizon

        # Dynamic search window factors based on priority
        # Higher factor means wider search, suitable for low priority jobs to find cheaper slots.
        # Lower factor means narrower search, suitable for high priority jobs to ensure early completion.
        PRIORITY_SEARCH_WINDOW_FACTORS = {
            0: 0.8,   # CRITICAL: Narrower window, prioritize immediate availability
            1: 1.0,   # HIGH: Standard window
            2: 1.2,   # MEDIUM: Slightly wider window
            3: 1.5,   # LOW: Widest window, prioritize cost
            99: 1.0   # Default/unknown (for any non-mapped priority)
        }

        def _calculate_operation_objective_value(self, server_id: str, start_time: int, op_duration: int, job_obj: Job, op_idx: int) -> float:
            """
            Calculates the estimated total objective cost of scheduling an operation on a given server
            at a given start time. This includes compute cost and an estimate of future delay penalty
            for the entire job, aligning with the problem's total cost objective.
            """
            # Calculate compute cost for this operation
            op_compute_cost = 0.0
            for t in range(start_time, start_time + op_duration):
                # Ensure time index for cost is within time_horizon
                cost_t = t % self.inst.time_horizon
                op_compute_cost += self.inst.server_cost[server_id][cost_t]

            # Estimate remaining duration of the job after this operation
            remaining_operations_duration = sum(o.duration for o in job_obj.operations[op_idx + 1:])

            # Projected job completion time if this operation finishes at start_time + op_duration
            # and all subsequent operations start immediately
            projected_job_completion_time = (start_time + op_duration) + remaining_operations_duration

            # Calculate potential delay penalty for the entire job based on projected completion
            projected_delay = max(0, projected_job_completion_time - job_obj.deadline)

            # Apply the specific delay penalty weight from the instance
            delay_penalty_contribution = projected_delay * job_obj.delay_weight * self.inst.weights['delay_penalty']

            # Total score to minimize for this operation's placement
            return (op_compute_cost * self.inst.weights['compute_cost']) + delay_penalty_contribution

        def run_scheduling(self) -> Schedule:
            """
            Main scheduling loop: processes jobs based on priority and schedules operations.
            """
            # Prioritize jobs based on a multi-criteria heuristic:
            # 1. Job Priority (CRITICAL, HIGH, MEDIUM, LOW) - lower integer value is higher priority
            # 2. Effective deadline (deadline - total_duration - arrival_time) - tighter deadlines first
            # 3. Delay weight (higher delay weight first) - multiplied by -1 to sort descending
            # 4. Arrival time (earlier is better, for secondary tie-breaking)
            # 5. Job ID (for deterministic tie-breaking)
            sorted_jobs = sorted(self.inst.jobs.keys(),
                                 key=lambda j_id: (
                                     self.inst.jobs[j_id].priority_score,  # Lower priority_score first
                                     self.inst.jobs[j_id].deadline - self.inst.jobs[j_id].arrival_time - self.inst.jobs[j_id].total_duration, # Smaller slack (tighter deadline) first
                                     -self.inst.jobs[j_id].delay_weight, # Higher delay weight first
                                     self.inst.jobs[j_id].arrival_time,  # Earlier arrival as tie-breaker
                                     j_id  # Final tie-breaker
                                 ))

            for job_id in sorted_jobs:
                job = self.inst.jobs[job_id]
                self.schedule[job_id] = []

                # `current_op_start_time` tracks the earliest an operation of THIS job can start,
                # considering its own arrival_time and preceding operations of the SAME job.
                current_op_start_time = self.job_available_time[job_id]

                for op_idx, op in enumerate(job.operations):
                    best_overall_objective_cost = float('inf')
                    chosen_server = None
                    chosen_start = None
                    best_min_free_capacity = -1 # Used for tie-breaking on local servers

                    # Get all compatible servers, sorted by heuristic preference (local, then cloud, by health/cost)
                    compatible_servers_sorted = self._get_servers_by_type_sorted(op.server_type, current_op_start_time)

                    # Determine search end time for local machines: balance finding cheaper slots with avoiding excessive lookahead
                    # Dynamically adjust the search window based on job priority
                    priority_factor = self.PRIORITY_SEARCH_WINDOW_FACTORS.get(job.priority_score, 1.0)
                    dynamic_search_window_multiplier = self.BASE_LOCAL_SEARCH_WINDOW_MULTIPLIER * priority_factor

                    search_end_time = max(current_op_start_time + op.duration, # At least enough time for current op
                                          min(self.inst.time_horizon * dynamic_search_window_multiplier,
                                              job.deadline + op.duration + self.BASE_LOCAL_SEARCH_WINDOW_ADD))
                    search_end_time = int(search_end_time) # Ensure it's an integer for range()

                    # 1. Evaluate local machines first
                    for server_id in compatible_servers_sorted:
                        if self.inst.servers[server_id].is_cloud:
                            break  # All remaining are cloud, handled separately

                        for start_time_candidate in range(current_op_start_time, search_end_time + 1):
                            if self._has_capacity(server_id, start_time_candidate, op.duration, op.capacity_required):
                                current_objective_cost = self._calculate_operation_objective_value(
                                    server_id, start_time_candidate, op.duration, job, op_idx
                                )
                                current_min_free_capacity = self._calculate_min_free_capacity_during_op(
                                    server_id, start_time_candidate, op.duration, op.capacity_required
                                )
                                # Tie-breaking:
                                # 1. Lower objective cost
                                # 2. Earlier start time
                                # 3. Higher minimum free capacity (more resilient)
                                if current_objective_cost < best_overall_objective_cost or \
                                   (current_objective_cost == best_overall_objective_cost and (chosen_start is None or start_time_candidate < chosen_start)) or \
                                   (current_objective_cost == best_overall_objective_cost and start_time_candidate == chosen_start and current_min_free_capacity > best_min_free_capacity):
                                    best_overall_objective_cost = current_objective_cost
                                    chosen_server = server_id
                                    chosen_start = start_time_candidate
                                    best_min_free_capacity = current_min_free_capacity
                                # Do NOT break here: continue searching all slots on this server
                                # to find the truly optimal (lowest cost/penalty) slot within the window.

                    # 2. Evaluate cloud machines (can always start immediately)
                    cloud_start_time = current_op_start_time
                    best_cloud_server_candidate = None

                    # Find the cheapest cloud server of the required type among the sorted compatible list
                    for s_id in compatible_servers_sorted:
                        if self.inst.servers[s_id].is_cloud:
                            best_cloud_server_candidate = s_id
                            break # Due to sorting, this will be the cheapest cloud server

                    if best_cloud_server_candidate:
                        cloud_objective_cost = self._calculate_operation_objective_value(
                            best_cloud_server_candidate, cloud_start_time, op.duration, job, op_idx
                        )

                        # Compare the best cloud option with the best local option found so far
                        # Tie-breaking: if costs are equal, prioritize earlier start time, then local (False) over cloud (True)
                        if cloud_objective_cost < best_overall_objective_cost or \
                           (cloud_objective_cost == best_overall_objective_cost and (chosen_start is None or cloud_start_time < chosen_start)) or \
                           (cloud_objective_cost == best_overall_objective_cost and cloud_start_time == chosen_start and chosen_server is not None and not self.inst.servers[chosen_server].is_cloud): # Prefer local on equal metrics, but this check is inverted for cloud
                                # Re-evaluating the tie-breaking logic to ensure it prefers local over cloud if all other metrics are identical
                                # The current 'best_overall_objective_cost', 'chosen_start', 'chosen_server' might represent a local option.
                                # If cloud_objective_cost is truly equal, and cloud_start_time is equal, we should *not* pick cloud over local.
                                pass # This conditional implicitly means we only update if cloud is strictly better OR
                                     # if current `chosen_server` is None (no local option found yet) OR
                                     # if cloud_start_time is strictly earlier.

                        # Revised logic for selecting cloud:
                        # If cloud is strictly better (lower cost) than the best local found, select it.
                        # If costs are equal:
                        #   If no local server was found, select cloud.
                        #   If a local server was found with the same cost, do not prefer cloud automatically.
                        #   We keep the current chosen_server (which would be local) unless cloud is strictly better.
                        # The existing `if cloud_objective_cost < best_overall_objective_cost` part handles strict improvement.
                        # The `or (cloud_objective_cost == best_overall_objective_cost and (chosen_start is None or cloud_start_time < chosen_start))` part ensures:
                        #   - If costs are equal AND no local chosen yet (chosen_start is None), pick cloud.
                        #   - If costs are equal AND cloud starts earlier, pick cloud.
                        # This logic correctly prefers earliest start time *first*, then cost, and only defaults to cloud if no local is competitive.
                        # The `and self.inst.servers[best_cloud_server_candidate].is_cloud` was incorrect for a tie-break to prefer cloud.
                        # So, let's ensure we prefer local if costs and start_times are equal.

                        # Refined logic for selecting between local and cloud candidates
                        # If the cloud option is strictly better
                        if cloud_objective_cost < best_overall_objective_cost:
                            best_overall_objective_cost = cloud_objective_cost
                            chosen_server = best_cloud_server_candidate
                            chosen_start = cloud_start_time
                            best_min_free_capacity = float('inf') # Cloud has "infinite" free capacity for tie-breaking comparison
                        # If costs are equal, prefer local over cloud (unless no local was found yet)
                        elif cloud_objective_cost == best_overall_objective_cost:
                            # If no local option was found (chosen_server is None)
                            if chosen_server is None:
                                best_overall_objective_cost = cloud_objective_cost
                                chosen_server = best_cloud_server_candidate
                                chosen_start = cloud_start_time
                                best_min_free_capacity = float('inf')
                            # If local was found, and cloud also has same start time, still prefer local
                            elif chosen_start == cloud_start_time and not self.inst.servers[chosen_server].is_cloud:
                                # Do nothing, keep the local server
                                pass
                            # If cloud has same cost but earlier start time than currently chosen local
                            elif chosen_start is not None and cloud_start_time < chosen_start:
                                best_overall_objective_cost = cloud_objective_cost
                                chosen_server = best_cloud_server_candidate
                                chosen_start = cloud_start_time
                                best_min_free_capacity = float('inf')

                    # 3. Commit the chosen option to the schedule and update capacity usage
                    if chosen_server is not None:
                        for t in range(chosen_start, chosen_start + op.duration):
                            if t not in self.capacity_usage[chosen_server]:
                                self.capacity_usage[chosen_server][t] = 0
                            self.capacity_usage[chosen_server][t] += op.capacity_required

                        self.schedule[job_id].append({"server": chosen_server, "start": chosen_start})
                        current_op_start_time = chosen_start + op.duration
                        self.job_available_time[job_id] = current_op_start_time # Update for next operations of this job
                    else:
                        # Fallback: This should ideally not be reached if cloud is truly infinite/always available
                        # and there's at least one cloud server per type. If it is, something is fundamentally wrong.
                        # For robustness, we will try to force schedule on the first compatible server found,
                        # even if it violates capacity (which evaluation will then penalize).
                        # This ensures the schedule generation doesn't crash.
                        fallback_server_id = None
                        # Prioritize a local server for fallback if possible, otherwise any cloud
                        if compatible_servers_sorted:
                            fallback_server_id = compatible_servers_sorted[0]
                        else: # Truly no servers of this type exist
                            raise ValueError(f"No compatible server (local or cloud) found for job {job_id}, operation {op_idx} (type: {op.server_type})")

                        fallback_start_time = current_op_start_time
                        for t in range(fallback_start_time, fallback_start_time + op.duration):
                            if t not in self.capacity_usage[fallback_server_id]:
                                self.capacity_usage[fallback_server_id][t] = 0
                            self.capacity_usage[fallback_server_id][t] += op.capacity_required
                        self.schedule[job_id].append({"server": fallback_server_id, "start": fallback_start_time})
                        self.job_available_time[job_id] = fallback_start_time + op.duration

            return self.schedule

    # Instantiate the scheduler and run it
    scheduler = Scheduler(inst)
    return scheduler.run_scheduling()


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