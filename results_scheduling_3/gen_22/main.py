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

# INCREMENTAL ONLINE JOB SCHEDULER
# =============================================================================
# YOU ARE EDITING THIS FUNCTION (schedule_job_online)
#
# STRICT ONLINE CONSTRAINT:
# When scheduling a job, you can ONLY use:
#   1. The current job being scheduled
#   2. The current time (when this job arrived)
#   3. Previously scheduled operations (in schedule and capacity_usage)
#
# You CANNOT:
#   - Peek at future jobs (those arriving after current_time)
#   - Reorder or reschedule already-placed operations
#   - Use global knowledge about all jobs
#
# This simulates real-time scheduling where decisions are irrevocable.
# =============================================================================


def schedule_job_online(inst: Instance,
                        job: Job,
                        schedule: Schedule,
                        capacity_usage: Dict[str, Dict[int, int]],
                        current_time: int):
    """
    Schedules a single job online using a slack-aware, best-fit heuristic.
    """
    schedule[job.id] = []
    current_op_release_time = max(job.arrival_time, current_time)

    # Heuristic Parameters
    CLOUD_COST_MULTIPLIER = 1.75 # Increased multiplier to further discourage unnecessary cloud use

    # Procrastination penalty factors based on priority
    # Higher factor means more penalty for waiting
    PRIORITY_PROCR_PENALTY_FACTOR = {
        "Emergency": 0.5, # Very high penalty for Emergency jobs to start late
        "High": 0.2,      # High penalty for High priority jobs
        "Medium": 0.05,   # Moderate penalty for Medium jobs
        "Low": 0.01       # Low penalty for Low priority jobs
    }

    # Slack-based search horizons for local machines
    SLACK_BASED_SEARCH_HORIZONS = {
        "critical": 12, # For jobs already past deadline or very little slack
        "tight": 48,    # For jobs with some slack but still urgent
        "normal": 96    # For jobs with ample slack
    }

    for idx, op in enumerate(job.operations):
        remaining_duration = sum(o.duration for o in job.operations[idx + 1:])

        def _calculate_score(server_id, start_time):
            """Calculates a placement score, including a penalty for waiting."""
            server = inst.servers[server_id]
            compute_cost = sum(inst.server_cost[server_id][t % inst.time_horizon]
                               for t in range(start_time, start_time + op.duration))

            if server.is_cloud:
                compute_cost *= CLOUD_COST_MULTIPLIER

            projected_completion = start_time + op.duration + remaining_duration
            delay = max(0, projected_completion - job.deadline)
            delay_penalty = delay * job.delay_weight

            # Procrastination Penalty: Penalize waiting time, scaled by job importance.
            wait_time = max(0, start_time - current_op_release_time)
            procrastination_factor = PRIORITY_PROCR_PENALTY_FACTOR.get(job.priority, 0.05)
            procrastination_penalty = wait_time * job.delay_weight * procrastination_factor

            return (compute_cost * inst.weights['compute_cost'] +
                    delay_penalty * inst.weights['delay_penalty'] +
                    procrastination_penalty)

        best_score = float('inf')
        chosen_server = None
        chosen_start = -1

        compatible_servers = [s_id for s_id, s in inst.servers.items() if s.server_type == op.server_type]
        local_servers = [s_id for s_id in compatible_servers if not inst.servers[s_id].is_cloud]
        cloud_servers = [s_id for s_id in compatible_servers if inst.servers[s_id].is_cloud]

        # 1. Establish a cloud benchmark if available, as a guaranteed fallback.
        if cloud_servers:
            # Find the cheapest cloud server for this operation type (assuming all cloud servers of same type have similar cost structures, but using min for robustness)
            cheapest_cloud_id = min(cloud_servers, key=lambda s_id: inst.server_cost[s_id][current_op_release_time % inst.time_horizon])
            cloud_score = _calculate_score(cheapest_cloud_id, current_op_release_time)
            best_score = cloud_score
            chosen_server = cheapest_cloud_id
            chosen_start = current_op_release_time

        # 2. Determine dynamic search horizon for local machines based on job slack.
        time_to_deadline = job.deadline - (current_op_release_time + op.duration + remaining_duration)
        if time_to_deadline <= 0:
            search_horizon_value = SLACK_BASED_SEARCH_HORIZONS["critical"]
        elif time_to_deadline <= 24:
            search_horizon_value = SLACK_BASED_SEARCH_HORIZONS["tight"]
        else:
            search_horizon_value = SLACK_BASED_SEARCH_HORIZONS["normal"]

        # Cap the search horizon to a reasonable absolute maximum to prevent excessive computation
        absolute_max_search_time = current_op_release_time + 3 * inst.time_horizon
        search_end_time = min(current_op_release_time + search_horizon_value, absolute_max_search_time)

        # Pre-cache available capacity for local servers to speed up the search.
        local_server_capacity_cache = defaultdict(dict)
        for server_id in local_servers:
            for t_check in range(current_op_release_time, search_end_time + op.duration + 1):
                local_server_capacity_cache[server_id][t_check] = get_available_capacity(inst, server_id, t_check)

        # 3. Best-Fit Search: Find the best slot on local machines within the dynamic horizon.
        for server_id in local_servers:
            for start_t in range(current_op_release_time, search_end_time + 1): # +1 to include search_end_time
                if has_capacity(inst, capacity_usage, server_id, start_t, op.duration, op.capacity_required):
                    score = _calculate_score(server_id, start_t)
                    if score < best_score:
                        best_score = score
                        chosen_server = server_id
                        chosen_start = start_t
        
        # 4. Failsafe: If no satisfactory placement was found, especially if local search failed or cloud was not initially
        # the best option, ensure a placement. This handles cases where local search *within the horizon* might not find a slot,
        # but one exists later, or if the initial cloud benchmark was too pessimistic against currently full local machines.
        if chosen_server is None:
            if cloud_servers:
                # If no local or initial cloud was chosen, force the cheapest cloud at earliest time
                cheapest_cloud_id = min(cloud_servers, key=lambda s_id: inst.server_cost[s_id][current_op_release_time % inst.time_horizon])
                best_score = _calculate_score(cheapest_cloud_id, current_op_release_time)
                chosen_server = cheapest_cloud_id
                chosen_start = current_op_release_time
            elif local_servers:
                # If no cloud is available for this type, and local search within horizon failed,
                # perform an exhaustive search on the cheapest local machine to guarantee placement.
                cheapest_local_id = min(local_servers, key=lambda s_id: sum(inst.server_cost[s_id].values()))
                start_t = current_op_release_time
                while True: # Search indefinitely to guarantee a placement
                    if has_capacity(inst, capacity_usage, cheapest_local_id, start_t, op.duration, op.capacity_required):
                        best_score = _calculate_score(cheapest_local_id, start_t)
                        chosen_server = cheapest_local_id
                        chosen_start = start_t
                        break
                    start_t += 1
            else:
                # This should only happen if there are no compatible servers at all
                raise RuntimeError(f"Could not find any compatible server for job {job.id}, op {idx}")


        # Reserve capacity for the chosen operation.
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