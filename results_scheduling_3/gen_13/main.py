# Modified initial.py with ONLINE scheduling support
# Key change: incremental, event-driven scheduling without future job knowledge

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import math
from collections import defaultdict

# =============================================================================
# DATA STRUCTURES (unchanged)
# =============================================================================

TIME_HORIZON = 24

@dataclass
class Server:
    id: str
    server_type: str
    capacity: int
    is_cloud: bool = False


@dataclass
class Operation:
    server_type: str
    duration: int
    capacity_required: int = 1


@dataclass
class Job:
    id: str
    description: str
    priority: str
    deadline: int
    delay_weight: float
    operations: List[Operation]
    arrival_time: int = 0


@dataclass
class MachineFailure:
    machine_id: str
    failure_start: int
    failure_end: int
    capacity_reduction: int
    failure_rate: float = 1.0
    recovery_rate: float = 1.0
    reason: str = ""
    is_random: bool = False


@dataclass
class Instance:
    servers: Dict[str, Server]
    jobs: Dict[str, Job]
    server_cost: Dict[str, Dict[int, float]]
    time_horizon: int
    weights: Dict[str, float]
    machine_failures: Optional[List[MachineFailure]] = None


Schedule = Dict[str, List[Dict[str, Any]]]


# =============================================================================
# INSTANCE CREATION
# =============================================================================
import random
def create_instance(
    scenario: str = "baseline",
    failure_scenario: Optional[str] = None,
    enable_random_failures: bool = True,
    random_failure_rate: float = 6.0,
    random_seed: Optional[int] = random.randint(0, 1_000_000)
) -> Instance:
    """
    Create the problem instance from CSV files.

    Args:
        scenario: Job scenario to load from CSV
        failure_scenario: Machine failure scenario or None for no failures
        enable_random_failures: If True, add stochastic background failures
        random_failure_rate: Mean hours between random failures per machine
        random_seed: Random seed for reproducible failures (None for true randomness)

    Returns:
        Instance object with jobs, servers, and optional machine failures from CSV
    """
    from helpers.scenario_loader import create_instance_from_csv
    return create_instance_from_csv(
        scenario=scenario,
        failure_scenario=failure_scenario,
        enable_random_failures=enable_random_failures,
        random_failure_rate=random_failure_rate,
        random_seed=random_seed
    )


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
    cloud_relocations: List[str] = None

    def __post_init__(self):
        if self.cloud_relocations is None:
            self.cloud_relocations = []

    def is_valid(self) -> bool:
        return len(self.violations) == 0


def evaluate(inst: Instance, schedule: Schedule) -> Fitness:
    """
    Evaluate a schedule with capacity constraints.
    """
    violations = []
    total_compute_cost = 0
    total_delay_penalty = 0
    job_completion_times = {}
    cloud_relocations = []

    # Track capacity usage per machine per timestep
    capacity_usage = {s: {} for s in inst.servers}

    def get_available_capacity_at_time(server_id: str, time: int) -> int:
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

    # Process each job's operations
    for job_id, ops in schedule.items():
        if job_id not in inst.jobs:
            violations.append(f"Unknown job {job_id}")
            continue

        job = inst.jobs[job_id]
        current_time = 0

        for idx, op_sched in enumerate(ops):
            if idx >= len(job.operations):
                violations.append(f"{job_id}: too many operations")
                continue

            operation = job.operations[idx]
            server_id = op_sched["server"]
            start = op_sched["start"]
            duration = operation.duration
            capacity_req = operation.capacity_required

            # Check server exists and type matches
            if server_id not in inst.servers:
                violations.append(f"{job_id} op {idx}: unknown server {server_id}")
                continue

            server = inst.servers[server_id]
            if server.server_type != operation.server_type:
                violations.append(
                    f"{job_id} op {idx}: type mismatch ({server.server_type} != {operation.server_type})"
                )
                continue

            # Check precedence
            if start < current_time:
                violations.append(f"{job_id} op {idx}: starts at {start} before previous ends at {current_time}")

            current_time = start + duration

            # Calculate compute cost
            for t in range(start, start + duration):
                hour = t % inst.time_horizon
                cost_per_hour = inst.server_cost[server_id][hour]
                total_compute_cost += cost_per_hour

            # Track capacity usage
            for t in range(start, start + duration):
                if t not in capacity_usage[server_id]:
                    capacity_usage[server_id][t] = 0
                capacity_usage[server_id][t] += capacity_req

        # Record job completion time
        job_completion_times[job_id] = current_time

        # Calculate delay penalty
        delay = max(0, current_time - job.deadline)
        total_delay_penalty += delay * job.delay_weight

    # Check capacity constraints
    for server_id, usage_by_time in capacity_usage.items():
        server = inst.servers[server_id]
        if server.is_cloud:
            continue  # Cloud has unlimited capacity

        for time, used in usage_by_time.items():
            available = get_available_capacity_at_time(server_id, time)
            if used > available:
                violations.append(
                    f"{server_id}: capacity exceeded at t={time} (used: {used}, available: {available})"
                )

    # Calculate weighted score
    score = (
        total_compute_cost * inst.weights["compute_cost"]
        + total_delay_penalty * inst.weights["delay_penalty"]
    )

    # Add penalty for violations
    if violations:
        score += 1e6 * len(violations)

    return Fitness(
        score=score,
        compute_cost=total_compute_cost,
        delay_penalty=total_delay_penalty,
        violations=violations,
        job_completion_times=job_completion_times,
        cloud_relocations=cloud_relocations,
    )


# =============================================================================
# ONLINE SCHEDULING CORE
# =============================================================================


def get_available_capacity(inst: Instance, server_id: str, time: int) -> int:
    """Get available capacity at a specific time (for scheduling decisions)."""
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


def has_capacity(inst, capacity_usage, server_id, start, duration, req):
    for t in range(start, start + duration):
        used = capacity_usage[server_id].get(t, 0)
        if used + req > get_available_capacity(inst, server_id, t):
            return False
    return True


def operation_objective(inst, server_id, start, op, job, remaining_duration):
    compute = sum(inst.server_cost[server_id][t % inst.time_horizon]
                  for t in range(start, start + op.duration))
    projected_completion = start + op.duration + remaining_duration
    delay = max(0, projected_completion - job.deadline)
    return (
        compute * inst.weights['compute_cost']
        + delay * job.delay_weight * inst.weights['delay_penalty']
    )


# =============================================================================
# =============================================================================
# SCHEDULE CONSTRUCTOR (ONLINE, EVOLVABLE)
# =============================================================================

# EVOLVE-BLOCK-START

# Constants for priority factors and cloud cost
PRIORITY_FACTORS_DELAY = {
    "Emergency": 3.0,  # Higher penalty for emergency jobs
    "High": 2.0,       # High penalty for high priority
    "Medium": 1.0,     # Default penalty
    "Low": 0.5         # Lower penalty for low priority
}
# Priority factors specifically for procrastination penalty (how bad is it to wait)
PRIORITY_FACTORS_PROCRASTINATION = {
    "Emergency": 2.0,
    "High": 1.5,
    "Medium": 1.0,
    "Low": 0.8
}
CLOUD_COST_MULTIPLIER = 1.3 # Make cloud machines moderately more expensive

class SchedulerState:
    def __init__(self, inst: Instance, initial_capacity_usage: Dict[str, Dict[int, int]]):
        self.inst = inst
        self.capacity_usage = initial_capacity_usage
        self.max_scheduled_end_time: int = 0 # Tracks the latest scheduled end time across ALL jobs.

    def get_available_capacity(self, server_id: str, time: int) -> int:
        """Get available capacity at a specific time (for scheduling decisions)."""
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

        return max(0, base_capacity - int(total_reduction))

    def _has_capacity(self, server_id: str, start: int, duration: int, req: int, cached_server_capacity: Dict[int, int]) -> bool:
        """Optimized capacity check using a pre-computed/cached available capacity for the current server."""
        for t in range(start, start + duration):
            used = self.capacity_usage[server_id].get(t, 0)
            available = cached_server_capacity.get(t, 0) # Fallback to 0 if not pre-cached (should be pre-cached though)
            if used + req > available:
                return False
        return True

    def _calculate_placement_score(self, server_id: str, start_time: int, op: Operation, job: Job, remaining_job_duration: int, is_cloud: bool, earliest_possible_start_for_op: int) -> float:
        """
        Calculates the objective cost for a given operation, incorporating priority-based
        delay penalties, an adaptive procrastination penalty, and cloud cost multiplier.
        """
        compute_cost = sum(self.inst.server_cost[server_id][t % self.inst.time_horizon]
                           for t in range(start_time, start_time + op.duration))
        
        if is_cloud:
            compute_cost *= CLOUD_COST_MULTIPLIER

        projected_job_completion = start_time + op.duration + remaining_job_duration
        delay = max(0, projected_job_completion - job.deadline)
        
        # Apply priority factor to delay penalty
        priority_factor_delay = PRIORITY_FACTORS_DELAY.get(job.priority, 1.0)
        delay_penalty_component = delay * job.delay_weight * self.inst.weights['delay_penalty'] * priority_factor_delay

        # Adaptive Procrastination Penalty (Recommendation 4)
        # Penalize waiting more for urgent jobs (low slack)
        
        # Slack for THIS operation if it starts at earliest_possible_start_for_op
        current_op_projected_completion = earliest_possible_start_for_op + op.duration + remaining_job_duration
        current_job_slack = job.deadline - current_op_projected_completion
        
        procrastination_base_rate = 0.02 # Default minimal rate per unit of waiting time
        
        # Increase procrastination penalty if deadline is tight for the entire job
        if current_job_slack < self.inst.time_horizon:
            procrastination_base_rate = 0.08
        if current_job_slack < op.duration: # Very tight
            procrastination_base_rate = 0.15
        if current_job_slack < 0: # Already past deadline
            procrastination_base_rate = 0.30 # Heavily penalize any further waiting
        
        # Apply additional priority factor to procrastination rate
        priority_factor_procrastination = PRIORITY_FACTORS_PROCRASTINATION.get(job.priority, 1.0)
        procrastination_effective_rate = procrastination_base_rate * priority_factor_procrastination
        
        # Penalize the difference between proposed start_time and the earliest possible start
        # Scale by compute_cost_weight to keep it in a similar magnitude
        procrastination_penalty_component = (start_time - earliest_possible_start_for_op) * procrastination_effective_rate * self.inst.weights['compute_cost']

        total_score = (
            compute_cost * self.inst.weights['compute_cost']
            + delay_penalty_component
            + procrastination_penalty_component
        )
        return total_score

    def _get_dynamic_search_horizon(self, job: Job, op: Operation, remaining_job_duration: int, earliest_possible_start_for_op: int) -> int:
        """
        Dynamically adjusts the local search horizon based on job slack and priority. (Recommendation 1)
        Returns the absolute end time (exclusive) for the search.
        """
        # Calculate job slack: how much time is available until the deadline
        # if all remaining operations were scheduled immediately after this one.
        projected_min_completion = earliest_possible_start_for_op + op.duration + remaining_job_duration
        current_slack = job.deadline - projected_min_completion
        
        base_lookahead = self.inst.time_horizon # A full day cycle
        
        # Jobs with low or negative slack need a shorter, more focused search.
        # Jobs with ample slack can afford a wider search for cheaper slots.
        if current_slack <= 0: # Already past deadline or extremely tight
            # Look ahead for a critical period to find *any* slot.
            lookahead_factor = 0.5
        elif current_slack < base_lookahead: # Moderately tight
            lookahead_factor = 1.0
        else: # Ample slack
            lookahead_factor = 2.0 # Wider search window for cost optimization
        
        # The search horizon is relative to the current_earliest_start_for_op + a lookahead buffer.
        # Also cap it by job.deadline + op.duration + a buffer, but prioritize the dynamic calculation.
        dynamic_end_time_candidate = earliest_possible_start_for_op + int(base_lookahead * lookahead_factor)
        
        # Ensure the search doesn't go unreasonably far into the future, but also covers the deadline.
        upper_bound = job.deadline + op.duration + base_lookahead
        
        # Max horizon to prevent excessive computation for very long-running jobs or distant deadlines
        absolute_max_search_horizon = self.max_scheduled_end_time + self.inst.time_horizon * 3 # e.g., current max + 3 days

        return min(dynamic_end_time_candidate, upper_bound, absolute_max_search_horizon)


    def schedule_job_online(self,
                            job: Job,
                            schedule: Schedule,
                            current_arrival_time: int):
        
        schedule[job.id] = []
        # earliest_start_for_current_op tracks the start time of the current operation
        # This is `max(job_arrival, current_time_global, completion_of_prev_job_op)`
        earliest_possible_start_for_op = max(job.arrival_time, current_arrival_time, self.max_scheduled_end_time)

        for idx, op in enumerate(job.operations):
            remaining_job_duration = sum(o.duration for o in job.operations[idx + 1:])

            best_overall_score = float('inf')
            chosen_server = None
            chosen_start = -1

            # --- Step 1: Evaluate the best cloud option immediately (Recommendation 3 & 5) ---
            cloud_server_candidate = None
            for server_id, server in self.inst.servers.items():
                if server.is_cloud and server.server_type == op.server_type:
                    cloud_server_candidate = server_id
                    break
            
            if cloud_server_candidate:
                cloud_score = self._calculate_placement_score(cloud_server_candidate, earliest_possible_start_for_op, op, job, remaining_job_duration, True, earliest_possible_start_for_op)
                best_overall_score = cloud_score
                chosen_server = cloud_server_candidate
                chosen_start = earliest_possible_start_for_op
            else:
                # This scenario indicates no compatible cloud server exists, which is unexpected based on problem assumptions.
                # If truly no cloud, best_overall_score remains inf.
                pass # Will attempt local search, if that also fails, it will raise an error.

            # --- Step 2: Search for the BEST local placement within the dynamic horizon (Recommendation 1 & 2) ---
            
            # Determine dynamic search horizon for local machines
            search_end_time = self._get_dynamic_search_horizon(job, op, remaining_job_duration, earliest_possible_start_for_op)
            
            # Pre-cache available capacity for local servers to speed up the search
            local_server_capacity_cache = defaultdict(dict)
            for server_id, server in self.inst.servers.items():
                if not server.is_cloud and server.server_type == op.server_type:
                    # Pre-calculate available capacity for all relevant times for this local server
                    # The range needs to cover any possible start time + duration
                    for t_check in range(earliest_possible_start_for_op, search_end_time + op.duration): 
                        local_server_capacity_cache[server_id][t_check] = self.get_available_capacity(server_id, t_check)

            for server_id, server in self.inst.servers.items():
                if not server.is_cloud and server.server_type == op.server_type:
                    # Iterate through all possible start times for this local server within the horizon
                    for start_time_candidate in range(earliest_possible_start_for_op, search_end_time):
                        
                        # Check if this slot has capacity using the optimized helper, leveraging the cache
                        if self._has_capacity(server_id, start_time_candidate, op.duration, op.capacity_required, local_server_capacity_cache[server_id]):
                            score = self._calculate_placement_score(server_id, start_time_candidate, op, job, remaining_job_duration, False, earliest_possible_start_for_op)
                            
                            if score < best_overall_score:
                                best_overall_score = score
                                chosen_server = server_id
                                chosen_start = start_time_candidate
                            # CRITICAL: Do NOT break here. We need to find the BEST local slot, not just the first.

            # --- Step 3: Final decision and capacity reservation ---
            if chosen_server is None:
                # This case should ideally not be reached if cloud is truly a fallback,
                # unless no compatible server (local or cloud) was found.
                # If cloud_server_candidate was found, it should have been chosen as initial best.
                # This could only happen if no compatible server of any kind exists.
                raise Exception(f"Failed to schedule operation {idx} for job {job.id}. No compatible server (local or cloud) found.")

            # Reserve capacity for the chosen slot
            for t in range(chosen_start, chosen_start + op.duration):
                self.capacity_usage[chosen_server][t] = self.capacity_usage[chosen_chosen_server].get(t, 0) + op.capacity_required

            schedule[job.id].append({"server": chosen_server, "start": chosen_start})
            # Update earliest_possible_start_for_op for the next operation in the same job
            earliest_possible_start_for_op = chosen_start + op.duration
            # Update the global max scheduled time
            self.max_scheduled_end_time = max(self.max_scheduled_end_time, earliest_possible_start_for_op)


# EVOLVE-BLOCK-END


# =============================================================================
# EVENT-DRIVEN ONLINE EXECUTION LOOP (NOT EVOLVABLE - ENFORCES ONLINE CONSTRAINT)
# =============================================================================


def construct_schedule(inst: Instance) -> Schedule:
    """
    STRICT ONLINE SCHEDULER - NO FUTURE KNOWLEDGE
    
    This function enforces the online constraint and is NOT evolved.
    Only schedule_job_online (above) can be modified during evolution.
    
    CRITICAL ONLINE CONSTRAINT:
    When scheduling a job arriving at time T, you can ONLY access:
    - Jobs that have already been scheduled (arrival_time < T)
    - The current job (arrival_time == T)
    
    You CANNOT access or analyze jobs with arrival_time > T
    
    Implementation: Process jobs in arrival order, one at a time.
    """
    schedule = {}
    capacity_usage = defaultdict(dict)

    # Create list of (arrival_time, job) tuples and sort ONLY by arrival time
    # Note: We must access inst.jobs to know when they arrive, but we
    # process them strictly in order without lookahead
    job_arrivals = [(job.arrival_time, job) for job in inst.jobs.values()]
    job_arrivals.sort(key=lambda x: x[0])  # Sort only by arrival time
    
    # Process each job as it arrives (strict temporal order)
    for arrival_time, job in job_arrivals:
        # At this moment, only this job and earlier jobs exist in our world
        # Jobs arriving later are "in the future" and unknown
        schedule_job_online(inst, job, schedule, capacity_usage, arrival_time)

    return schedule


# =============================================================================
# RUN SCHEDULING (ENTRY POINT FOR EVALUATION)
# =============================================================================

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
            inst = create_instance(scenario=scenario_name, failure_scenario=fail_scenario)
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
        inst = create_instance(scenario=scenario, failure_scenario=failure_scenario)
        schedule = construct_schedule(inst)
        fitness = evaluate(inst, schedule)
        return schedule, fitness, float(fitness.score)
