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
            self.last_scheduled_local_server_per_job = {}
            self.LOW_PRIORITY_LOCAL_SEARCH_WINDOW = 3 # hours. For LOW priority jobs, only search this far for a local slot.

            # Pre-calculate job properties for sorting
            self.priority_map = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
            for job_id, job in self.inst.jobs.items():
                job.total_duration = sum(op.duration for op in job.operations)
                job.priority_score = self.priority_map.get(job.priority, 99) # Lower score for higher priority

        def _get_available_capacity(self, server_id: str, time: int) -> int:
            """Calculate available capacity at a specific time, considering failures.
            This is the raw available capacity before considering current pending usage.
            """
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

        def _calculate_operation_compute_cost(self, server_id: str, start: int, duration: int) -> float:
            """Calculates the compute cost for an operation on a given server at a specific time."""
            cost = 0.0
            for t in range(start, start + duration):
                cost += self.inst.server_cost[server_id][t % self.inst.time_horizon]
            return cost

        def _get_servers_by_type(self, server_type: str, prefer_local: bool = True) -> List[str]:
            """Get servers of a given type, optionally preferring local over cloud."""
            servers = [s.id for s in self.inst.servers.values() if s.server_type == server_type]
            if prefer_local:
                # Sort: local machines first, then cloud
                servers.sort(key=lambda s: self.inst.servers[s].is_cloud)
            return servers

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
                    scheduled = False
                    best_server_choice = None
                    best_start_time_choice = None

                    # Store all valid local placement candidates (start_time, preferred_for_chaining_score, cost_for_op, neg_min_projected_capacity_during_op, server_id)
                    local_candidates = []

                    # 1. Try to schedule on a local machine
                    local_servers_for_type = [s_id for s_id, s in self.inst.servers.items()
                                              if s.server_type == op.server_type and not s.is_cloud]

                    # Prioritize local servers by current max capacity over the potential operation duration
                    # For simplicity, we can sort by current capacity at `current_op_start_time`
                    local_servers_for_type.sort(key=lambda s_id: self._get_available_capacity(s_id, current_op_start_time), reverse=True)

                    # Prioritized Sequential Scheduling for Critical/High Jobs
                    # If this is a CRITICAL/HIGH job and the previous operation was on a local machine of the same type,
                    # try to chain it on the same machine first.
                    preferred_for_chaining_server = None
                    if job.priority in ["CRITICAL", "HIGH"] and op_idx > 0:
                        last_op_info = self.schedule[job_id][op_idx - 1]
                        last_server_id = last_op_info["server"]
                        if not self.inst.servers[last_server_id].is_cloud and \
                           self.inst.servers[last_server_id].server_type == op.server_type:
                            preferred_for_chaining_server = last_server_id
                            # Move preferred server to the front for checking
                            if preferred_for_chaining_server in local_servers_for_type:
                                local_servers_for_type.remove(preferred_for_chaining_server)
                                local_servers_for_type.insert(0, preferred_for_chaining_server)

                    # Determine search window for local machines
                    # For LOW priority jobs, limit the search to a smaller window to allow faster cloud fallback
                    local_search_end_time = self.inst.time_horizon
                    if job.priority_score == self.priority_map["LOW"]: # If it's a LOW priority job
                        local_search_end_time = min(self.inst.time_horizon, current_op_start_time + self.LOW_PRIORITY_LOCAL_SEARCH_WINDOW)


                    for server_id in local_servers_for_type:
                        for start_time_candidate in range(current_op_start_time, local_search_end_time):
                            if self._has_capacity(server_id, start_time_candidate, op.duration, op.capacity_required):
                                cost_candidate = self._calculate_operation_compute_cost(server_id, start_time_candidate, op.duration)

                                # Calculate minimum projected remaining capacity after placement over the operation's duration
                                min_projected_capacity_during_op = float('inf')
                                for t in range(start_time_candidate, start_time_candidate + op.duration):
                                    available = self._get_available_capacity(server_id, t)
                                    used = self._get_current_used_capacity(server_id, t)
                                    min_projected_capacity_during_op = min(min_projected_capacity_during_op, (available - (used + op.capacity_required)))
                                # Use negative for sorting descending (max capacity)
                                neg_min_projected_capacity_during_op = -min_projected_capacity_during_op

                                preferred_for_chaining_score = 0 if server_id == preferred_for_chaining_server else 1
                                local_candidates.append((start_time_candidate, preferred_for_chaining_score, cost_candidate, neg_min_projected_capacity_during_op, server_id))
                                break # Found the earliest slot on *this specific server*, move to next server

                    if local_candidates:
                        # Sort local candidates:
                        # (1) earliest start
                        # (2) preferred for chaining (0 = preferred, 1 = normal)
                        # (3) lowest cost
                        # (4) highest *minimum projected* remaining capacity (neg_min_projected_capacity_during_op for sorting descending)
                        # (5) server_id for deterministic tie-breaking
                        local_candidates.sort()

                        best_local_start, _, _, _, best_local_server = local_candidates[0]
                        best_server_choice = best_local_server
                        best_start_time_choice = best_local_start
                        scheduled = True

                    # 2. If no local machine was chosen (either none available or search window truncated for LOW priority), use cloud immediately (to avoid delay)
                    if not scheduled:
                        cloud_servers_for_type = [s_id for s_id, s in self.inst.servers.items()
                                                  if s.server_type == op.server_type and s.is_cloud]

                        if cloud_servers_for_type:
                            cloud_start_time = current_op_start_time # Cloud has infinite capacity, can always start here

                            best_cloud_server_candidate = None
                            min_cloud_cost = float('inf')

                            # Select the cloud server with the lowest compute cost
                            for cloud_server_id in cloud_servers_for_type:
                                current_cloud_cost = self._calculate_operation_compute_cost(cloud_server_id, cloud_start_time, op.duration)
                                if current_cloud_cost < min_cloud_cost:
                                    min_cloud_cost = current_cloud_cost
                                    best_cloud_server_candidate = cloud_server_id

                            if best_cloud_server_candidate is not None:
                                best_server_choice = best_cloud_server_candidate
                                best_start_time_choice = cloud_start_time
                                scheduled = True
                                # Clear last scheduled local server as we've moved to cloud
                                self.last_scheduled_local_server_per_job.pop(job_id, None)

                    # 3. Commit the chosen option to the schedule and update capacity usage
                    if best_server_choice is not None:
                        for t in range(best_start_time_choice, best_start_time_choice + op.duration):
                            if t not in self.capacity_usage[best_server_choice]:
                                self.capacity_usage[best_server_choice][t] = 0
                            self.capacity_usage[best_server_choice][t] += op.capacity_required

                        self.schedule[job_id].append({"server": best_server_choice, "start": best_start_time_choice})
                        current_op_start_time = best_start_time_choice + op.duration
                        self.job_available_time[job_id] = current_op_start_time # Update for next operations of this job

                        # Update last scheduled local server for chaining if a local server was used
                        if not self.inst.servers[best_server_choice].is_cloud:
                            self.last_scheduled_local_server_per_job[job_id] = best_server_choice
                        else:
                            self.last_scheduled_local_server_per_job.pop(job_id, None) # Clear if cloud was used
                    else:
                        # Fallback: This should ideally not be reached if cloud is truly infinite/always available
                        # and there's at least one cloud server per type. If it is, something is fundamentally wrong.
                        raise ValueError(f"No compatible server (local or cloud) found for job {job_id}, operation {op_idx} (type: {op.server_type})")

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