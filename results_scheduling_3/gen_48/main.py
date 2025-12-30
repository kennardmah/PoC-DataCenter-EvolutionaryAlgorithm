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
PRIORITY_FACTORS = {
    "Emergency": 3.0,  # Higher penalty for emergency jobs
    "High": 2.0,       # High penalty for high priority
    "Medium": 1.0,     # Default penalty
    "Low": 0.5         # Lower penalty for low priority
}
CLOUD_COST_MULTIPLIER = 1.2 # Make cloud machines more expensive to encourage local usage
WAIT_TIME_PENALTY_RATE = 0.1 # Penalty for each unit of time an operation waits to start
LOCAL_MACHINE_PREFERENCE_BONUS = -5.0 # Small bonus for choosing a local machine


def _calculate_placement_score(inst: Instance, server_id: str, start: int, op: Operation, job: Job, remaining_duration: int, current_op_release_time: int) -> float:
    """
    Calculates the objective cost for a given operation placement, incorporating
    priority-based delay penalties, cloud cost multiplier, and a waiting penalty.
    """
    is_cloud = inst.servers[server_id].is_cloud
    
    # Base compute cost
    compute_cost = sum(inst.server_cost[server_id][t % inst.time_horizon]
                       for t in range(start, start + op.duration))
    
    if is_cloud:
        compute_cost *= CLOUD_COST_MULTIPLIER

    # Delay penalty
    projected_completion = start + op.duration + remaining_duration
    delay = max(0, projected_completion - job.deadline)
    priority_factor = PRIORITY_FACTORS.get(job.priority, 1.0)
    delay_penalty = delay * job.delay_weight * priority_factor
    
    # Procrastination penalty: penalize waiting time
    wait_time = max(0, start - current_op_release_time)
    # Scale wait penalty by delay_weight and priority, to reflect urgency
    procrastination_penalty = wait_time * job.delay_weight * priority_factor * WAIT_TIME_PENALTY_RATE

    # Local machine preference bonus
    local_incentive = LOCAL_MACHINE_PREFERENCE_BONUS if not is_cloud else 0.0

    return (
        compute_cost * inst.weights['compute_cost']
        + delay_penalty * inst.weights['delay_penalty']
        + procrastination_penalty
        + local_incentive
    )


def _has_capacity_optimized(capacity_usage: Dict[str, Dict[int, int]], server_id: str, start: int, duration: int, req: int, cached_server_capacity: Dict[int, int]) -> bool:
    """
    Optimized capacity check using a pre-computed/cached available capacity for the current server.
    cached_server_capacity: dict {time: effective_capacity} for the specific server.
    """
    for t in range(start, start + duration):
        used = capacity_usage[server_id].get(t, 0)
        available = cached_server_capacity.get(t, 0) # Use cached available capacity
        if used + req > available:
            return False
    return True


def schedule_job_online(inst: Instance,
                        job: Job,
                        schedule: Schedule,
                        capacity_usage: Dict[str, Dict[int, int]],
                        current_time: int):

    schedule[job.id] = []
    current_op_release_time = max(job.arrival_time, current_time)

    for idx, op in enumerate(job.operations):
        remaining_duration = sum(o.duration for o in job.operations[idx + 1:])

        best_score = float('inf')
        chosen_server = None
        chosen_start = -1

        # Determine a reasonable search window for local machines.
        # Max search horizon to prevent infinite loops and limit computation.
        max_search_horizon_absolute = current_op_release_time + 3 * inst.time_horizon # e.g., next 3 days
        
        # Search up to job.deadline + op_duration + some buffer, but cap it by absolute max.
        # Using TIME_HORIZON as a buffer provides flexibility.
        search_end_time = min(job.deadline + inst.time_horizon, max_search_horizon_absolute)
        
        # Pre-populate capacity cache for all local servers once per operation for the search window
        local_server_capacity_cache = defaultdict(dict)
        for server_id, server in inst.servers.items():
            if not server.is_cloud and server.server_type == op.server_type:
                # Pre-calculate available capacity for all relevant times for this local server
                # The range should cover from current_op_release_time up to search_end_time + op.duration
                for t_check in range(current_op_release_time, search_end_time + op.duration + 1): 
                    local_server_capacity_cache[server_id][t_check] = get_available_capacity(inst, server_id, t_check)

        # 1. Evaluate Cloud options first as a baseline and guaranteed placement
        for server_id, server in inst.servers.items():
            if server.server_type == op.server_type and server.is_cloud:
                score = _calculate_placement_score(inst, server_id, current_op_release_time, op, job, remaining_duration, current_op_release_time)
                if score < best_score:
                    best_score = score
                    chosen_server = server_id
                    chosen_start = current_op_release_time
                # Cloud servers are essentially interchangeable for capacity at any time,
                # so we can stop after finding the first compatible cloud server (or the one with cheapest cost for current_op_release_time)
                # For this setup, we're taking the first one that matches type.
                break 

        # 2. Evaluate Local machines for the best slot within the search window
        for server_id, server in inst.servers.items():
            if server.server_type != op.server_type or server.is_cloud:
                continue
            
            # Iterate through possible start times for this local server
            # Note: search_end_time is exclusive in range, so add 1 to include it.
            for start_time_candidate in range(current_op_release_time, search_end_time + 1): 
                # Check if this slot has capacity using the optimized helper, leveraging the cache
                if _has_capacity_optimized(capacity_usage, server_id, start_time_candidate, op.duration, op.capacity_required, local_server_capacity_cache[server_id]):
                    score = _calculate_placement_score(inst, server_id, start_time_candidate, op, job, remaining_duration, current_op_release_time)
                    if score < best_score:
                        best_score = score
                        chosen_server = server_id
                        chosen_start = start_time_candidate
                    # FIRST-FIT: Take the first available slot on this server and move to the next server.
                    # This heuristic was shown to perform well in comparison to iterating all slots.
                    break 
        
        # If no server was found, this indicates a problem or an empty `compatible_servers` list.
        # With the cloud baseline, `chosen_server` should never be None if cloud is available for that type.
        # This acts as a final failsafe, potentially picking the *initial* cloud option if nothing better was found.
        if chosen_server is None:
            # Re-confirm cloud for guaranteed placement in edge cases
            for server_id, server in inst.servers.items():
                if server.is_cloud and server.server_type == op.server_type:
                    chosen_server = server_id
                    chosen_start = current_op_release_time # Cloud can always start immediately
                    break
            if chosen_server is None: # Should not happen if instance is valid and cloud exists for type
                 raise RuntimeError(f"Failed to schedule operation {idx} for job {job.id}. No compatible server found.")

        # Reserve capacity for the chosen slot
        for t in range(chosen_start, chosen_start + op.duration):
            capacity_usage[chosen_server][t] = capacity_usage[chosen_server].get(t, 0) + op.capacity_required

        schedule[job.id].append({"server": chosen_server, "start": chosen_start})
        current_op_release_time = chosen_start + op.duration

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