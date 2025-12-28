# EVOLVE-BLOCK-START
"""Constructor-based job-shop scheduling for government data center"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
import math


# =============================================================================
# PROBLEM DEFINITION (UNCHANGED)
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
# FITNESS EVALUATION (UNCHANGED)
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

class JobShopScheduler:
    """
    Encapsulates the scheduling logic, prioritizing jobs and making intelligent
    local/cloud placement decisions.
    """
    def __init__(self, inst: Instance):
        self.inst = inst
        self.capacity_usage: Dict[str, Dict[int, int]] = {s.id: {} for s in inst.servers.values()}
        self.job_available_time: Dict[str, int] = {j.id: j.arrival_time for j in inst.jobs.values()}
        self.schedule: Schedule = {}
        self.job_current_op_idx: Dict[str, int] = {j.id: 0 for j in inst.jobs.values()} # Tracks current operation for each job

        self.priority_map = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        self.same_server_bonus_cost_reduction = 10.0 # Cost reduction for chaining critical/high jobs on same local server
        
        # Adaptive Lookahead Window parameters
        self.MAX_LOCAL_SEARCH_WINDOW_MULTIPLIER = 2 # Multiplier for duration, for non-critical jobs
        self.MAX_LOCAL_SEARCH_WINDOW_ADD = 6 # Additional hours to search, for non-critical jobs

        self._precompute_job_properties()
        self._categorize_servers()
        self.precomputed_available_capacity: Dict[str, Dict[int, int]] = self._precompute_all_capacities()

    def _precompute_job_properties(self):
        """Pre-calculate total duration and priority scores for each job."""
        for job_id, job in self.inst.jobs.items():
            job.total_duration = sum(op.duration for op in job.operations)
            job.priority_score = self.priority_map.get(job.priority, 99)

    def _categorize_servers(self):
        """Categorize servers by type for faster lookup."""
        self.local_servers_by_type: Dict[str, List[str]] = {}
        self.cloud_servers_by_type: Dict[str, List[str]] = {}

        for server_id, server in self.inst.servers.items():
            if server.is_cloud:
                self.cloud_servers_by_type.setdefault(server.server_type, []).append(server_id)
            else:
                self.local_servers_by_type.setdefault(server.server_type, []).append(server_id)

        # Sort cloud servers by cost per hour (cheapest first) for tie-breaking
        for s_type in self.cloud_servers_by_type:
            self.cloud_servers_by_type[s_type].sort(
                key=lambda s_id: self.inst.server_cost[s_id][0 % self.inst.time_horizon] # Use cost at t=0 as representative
            )

    def _precompute_all_capacities(self) -> Dict[str, Dict[int, int]]:
        """
        Precompute the static available capacity for all local servers across the time horizon,
        considering only machine failures. This avoids recalculating exponential functions repeatedly.
        """
        precomputed_caps = {}
        for server_id, server in self.inst.servers.items():
            if server.is_cloud:
                continue
            precomputed_caps[server_id] = {}
            for t in range(self.inst.time_horizon):
                base_capacity = server.capacity
                total_reduction = 0
                if self.inst.machine_failures:
                    for failure in self.inst.machine_failures:
                        if failure.machine_id != server_id:
                            continue

                        if t < failure.failure_start:
                            continue
                        elif t >= failure.failure_end:
                            time_since_end = t - failure.failure_end
                            recovery_factor = 1.0 - math.exp(-failure.recovery_rate * time_since_end)
                            reduction = failure.capacity_reduction * (1.0 - recovery_factor)
                            total_reduction += reduction
                        else: # During failure window
                            time_since_start = t - failure.failure_start
                            failure_factor = 1.0 - math.exp(-failure.failure_rate * time_since_start)
                            reduction = failure.capacity_reduction * failure_factor
                            total_reduction += reduction
                precomputed_caps[server_id][t] = max(0, base_capacity - int(total_reduction))
        return precomputed_caps

    def _get_available_capacity(self, server_id: str, time: int) -> int:
        """
        Get available capacity at a specific time using precomputed values for local servers,
        or infinite for cloud servers.
        """
        server = self.inst.servers[server_id]
        if server.is_cloud:
            return 999999999 # Effectively infinite

        if 0 <= time < self.inst.time_horizon:
            return self.precomputed_available_capacity[server_id][time]

        # For times beyond precomputed horizon (e.g., if an operation extends past T_H)
        # Assume full capacity if no ongoing failure.
        return server.capacity

    def _has_capacity(self, server_id: str, start: int, duration: int, capacity_req: int) -> bool:
        """Check if server has enough capacity for the entire operation duration."""
        for t in range(start, start + duration):
            used = self.capacity_usage[server_id].get(t, 0)
            available = self._get_available_capacity(server_id, t)
            if used + capacity_req > available:
                return False
        return True

    def _calculate_operation_compute_cost(self, server_id: str, start: int, duration: int) -> float:
        """Calculates the compute cost for an operation on a given server at a specific time."""
        cost = 0.0
        for t in range(start, start + duration):
            if t >= self.inst.time_horizon: # Cap cost calculation at time horizon
                cost += self.inst.server_cost[server_id][(self.inst.time_horizon - 1) % self.inst.time_horizon] # Use last known cost
                continue
            cost += self.inst.server_cost[server_id][t % self.inst.time_horizon]
        return cost

    def _calculate_remaining_capacity_sum(self, server_id: str, start: int, duration: int, capacity_req: int) -> int:
        """Calculates the sum of remaining capacity over the operation's duration AFTER placement."""
        remaining_capacity_sum = 0
        for t in range(start, start + duration):
            used = self.capacity_usage[server_id].get(t, 0)
            available = self._get_available_capacity(server_id, t)
            remaining_capacity_sum += max(0, available - (used + capacity_req))
        return remaining_capacity_sum

    def _calculate_objective_for_placement(self, job: Job, op: Operation, op_idx: int, server_id: str, start_time: int, last_scheduled_server: Optional[str]) -> float:
        """
        Calculates a combined objective score for placing an operation at a given time on a given server.
        Lower score is better.
        """
        # 1. Compute Cost
        compute_cost = self._calculate_operation_compute_cost(server_id, start_time, op.duration)
        
        # Apply same-server chaining bonus for CRITICAL/HIGH priority jobs
        if (job.priority_score <= self.priority_map["HIGH"]) and (server_id == last_scheduled_server) and (not self.inst.servers[server_id].is_cloud):
            compute_cost -= self.same_server_bonus_cost_reduction

        # 2. Projected Delay Penalty
        # Estimate completion time for *this operation*
        op_completion_time = start_time + op.duration

        # Estimate completion time for *the entire job* if this operation finishes at op_completion_time
        remaining_operations_duration = sum(j_op.duration for j_op in job.operations[op_idx + 1:])
        projected_job_completion_time = op_completion_time + remaining_operations_duration
        
        delay = max(0, projected_job_completion_time - job.deadline)
        delay_penalty = delay * job.delay_weight * self.inst.weights["delay_penalty"]

        # The total objective is a weighted sum.
        total_objective = self.inst.weights["compute_cost"] * compute_cost + delay_penalty
        return total_objective

    def _reserve_capacity(self, server_id: str, start: int, duration: int, capacity_req: int):
        """Reserves capacity for an operation on a given server."""
        for t in range(start, start + duration):
            self.capacity_usage[server_id][t] = self.capacity_usage[server_id].get(t, 0) + capacity_req

    def _find_best_slot(self, job: Job, op_idx: int, op: Operation, earliest_start: int, last_scheduled_server: Optional[str]) -> Optional[Tuple[int, str]]:
        """
        Finds the best slot (local or cloud) based on a unified objective function.
        Returns (start_time, server_id) or None.
        """
        local_servers = self.local_servers_by_type.get(op.server_type, [])
        cloud_servers = self.cloud_servers_by_type.get(op.server_type, [])
        
        all_candidates = [] # Stores (objective_score, start_time, -remaining_cap_sum, server_id)

        # Determine the effective search end time for local servers based on job priority
        # Critical/High priority jobs search for the absolute earliest slot with minimal lookahead to avoid delays.
        # Medium/Low priority jobs search further ahead to find cheaper slots (cost-time windows).
        base_search_window = op.duration * self.MAX_LOCAL_SEARCH_WINDOW_MULTIPLIER + self.MAX_LOCAL_SEARCH_WINDOW_ADD
        
        if job.priority_score <= self.priority_map["HIGH"]: # Critical or High priority
            # For critical jobs, search is primarily for earliest possible start, with small window to check local cost
            search_end_time = min(self.inst.time_horizon, earliest_start + op.duration + 2) # small lookahead
        else: # Medium or Low priority
            search_end_time = min(self.inst.time_horizon, earliest_start + base_search_window)

        # --- Evaluate Local Servers ---
        for server_id in local_servers:
            for start_time_candidate in range(earliest_start, search_end_time):
                if self._has_capacity(server_id, start_time_candidate, op.duration, op.capacity_required):
                    objective_score = self._calculate_objective_for_placement(
                        job, op, op_idx, server_id, start_time_candidate, last_scheduled_server
                    )
                    remaining_capacity_sum = self._calculate_remaining_capacity_sum(
                        server_id, start_time_candidate, op.duration, op.capacity_required
                    )
                    all_candidates.append((objective_score, start_time_candidate, -remaining_capacity_sum, server_id))
                    
                    # For local search, once the earliest valid slot is found for this server,
                    # we don't need to search further in time for THIS server in this inner loop.
                    # The overall best will be chosen from all candidate (server, time) pairs.
                    break 

        # --- Evaluate Cloud Servers ---
        # Cloud operations can always start at `earliest_start`
        if cloud_servers:
            cloud_start_time = earliest_start
            # Find the best cloud server (lowest cost as per categorization, then deterministic by ID)
            # The cloud_servers list is already sorted by cost in _categorize_servers.
            best_cloud_server_id = cloud_servers[0] 
            
            objective_score_cloud = self._calculate_objective_for_placement(
                job, op, op_idx, best_cloud_server_id, cloud_start_time, last_scheduled_server
            )
            # Remaining capacity for cloud is effectively infinite, so we use a large negative value
            # to make it appear 'better' in tie-breaking if all else is equal, but less important than cost/delay.
            # Using 0 or negative infinity can also work, but consistency helps.
            remaining_capacity_sum_cloud = 999999999 # Large positive, so -ve is large negative
            all_candidates.append((objective_score_cloud, cloud_start_time, -remaining_capacity_sum_cloud, best_cloud_server_id))
        
        if all_candidates:
            # Sort all collected candidates:
            # 1. By objective score (ascending, lowest is best)
            # 2. By start_time (ascending, earliest is best in case of objective ties)
            # 3. By -remaining_capacity_sum (ascending, meaning highest remaining capacity is best in case of ties)
            # 4. By server_id (ascending, for deterministic tie-breaking)
            best_candidate = sorted(all_candidates)[0]
            return (best_candidate[1], best_candidate[3]) # Return (start_time, server_id)
        return None

    def schedule_all_jobs(self) -> Schedule:
        """
        Main method to orchestrate scheduling all jobs according to priorities
        and resource availability.
        """
        # Prioritize jobs based on a multi-criteria heuristic:
        # 1. Job Priority (CRITICAL, HIGH, MEDIUM, LOW) - lower integer value is higher priority
        # 2. Effective deadline (deadline - total_duration - arrival_time) - tighter deadlines first
        # 3. Delay weight (higher delay weight first) - multiplied by -1 to sort descending
        # 4. Arrival time (earlier is better, for secondary tie-breaking)
        # 5. Job ID (for deterministic tie-breaking)
        sorted_job_ids = sorted(self.inst.jobs.keys(),
                                key=lambda j_id: (
                                    self.inst.jobs[j_id].priority_score,
                                    self.inst.jobs[j_id].deadline - self.inst.jobs[j_id].arrival_time - self.inst.jobs[j_id].total_duration,
                                    -self.inst.jobs[j_id].delay_weight,
                                    self.inst.jobs[j_id].arrival_time,
                                    j_id
                                ))

        for job_id in sorted_job_ids:
            job = self.inst.jobs[job_id]
            self.schedule[job_id] = []

            # `current_op_start_time` tracks the earliest an operation of THIS job can start,
            # considering its own arrival_time and preceding operations of the SAME job.
            current_op_start_time = self.job_available_time[job_id]
            last_scheduled_server: Optional[str] = None 

            for op_idx, op in enumerate(job.operations):
                best_slot = self._find_best_slot(job, op_idx, op, current_op_start_time, last_scheduled_server)

                if best_slot:
                    final_start, final_server = best_slot
                    self._reserve_capacity(final_server, final_start, op.duration, op.capacity_required)
                    self.schedule[job_id].append({"server": final_server, "start": final_start})
                    current_op_start_time = final_start + op.duration
                    last_scheduled_server = final_server
                else:
                    # Fallback for extreme cases (e.g., no compatible server defined)
                    raise ValueError(f"No compatible server (local or cloud) found for job {job_id}, operation {op_idx} (type: {op.server_type})")

                self.job_available_time[job_id] = current_op_start_time # Update for next operations of this job

        return self.schedule


def construct_schedule(inst: Instance) -> Schedule:
    """
    Main entry point for scheduling. Creates a JobShopScheduler and runs it.
    """
    scheduler = JobShopScheduler(inst)
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