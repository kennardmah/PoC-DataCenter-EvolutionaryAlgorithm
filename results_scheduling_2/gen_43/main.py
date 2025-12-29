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

def create_instance(
    scenario: str = "baseline",
    failure_scenario: Optional[str] = None,
    enable_random_failures: bool = True,
    random_failure_rate: float = 6.0,
    random_seed: Optional[int] = 42
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

    # Define parameters for the scheduling heuristic
    # Tuned to make higher priorities significantly more sensitive to delays.
    PRIORITY_MULTIPLIERS = {
        'emergency': 12.0, # Highest urgency, highest penalty for delay
        'high': 4.0,
        'medium': 1.0,
        'low': 0.5,       # Lowest urgency, more flexible with delay
    }

    # Base cost to deter cloud usage (added to compute cost if cloud is used).
    # This encourages the scheduler to prefer cheaper local machines.
    CLOUD_DETERRENCE_BASE = 120.0 # Increased deterrence to push for local when possible
    # Discount for emergency jobs using cloud: reduces the deterrence,
    # making cloud a more palatable option for urgent tasks.
    CLOUD_EMERGENCY_DISCOUNT = 0.95 # Higher discount for emergency jobs to use cloud

    # Define a threshold for "critical" delay that triggers aggressive cloud fallback.
    CRITICAL_DELAY_THRESHOLD_HOURS = 4 # If job completion > deadline + this threshold, consider aggressive cloud

    # Factor to aggressively discount cloud deterrence when critical delay threshold is met.
    CLOUD_AGGRESSIVE_DISCOUNT_FACTOR = 0.99 # Make cloud almost free in critically delayed scenarios

    # Define a maximum horizon to search for available slots on local machines.
    # This prevents excessively long searches and keeps decisions localized while still
    # allowing for some lookahead to find optimal local slots.
    MAX_LOCAL_SEARCH_WINDOW = inst.time_horizon * 3 + 12 # Extended search window (24*3+12 = 84 hours)

    # Helper function to calculate the score for a proposed operation placement
    def _calculate_placement_score(
        _inst: Instance,
        _job: Job,
        _operation: Operation,
        _proposed_start: int,
        _server_id: str,
        _is_cloud_server: bool,
        _remaining_ops_duration: int,
        _priority_multipliers: Dict[str, float],
        _cloud_deterrence_base: float,
        _cloud_emergency_discount: float,
        _aggressive_discount_active: bool = False # New parameter for aggressive discount
    ) -> (float, int):
        """
        Calculates a weighted score for a proposed operation placement,
        considering compute cost, the impact on the job's overall delay penalty,
        and adding dynamic cloud deterrence. Returns (total_score, projected_job_completion).
        """
        # Compute cost for THIS operation
        op_compute_cost = sum(_inst.server_cost[_server_id][t % _inst.time_horizon]
                              for t in range(_proposed_start, _proposed_start + _operation.duration))

        # Add cloud deterrence if the operation is placed on a cloud server.
        if _is_cloud_server:
            deterrence = _cloud_deterrence_base
            if _aggressive_discount_active:
                # Apply aggressive discount if triggered by critical delay
                deterrence *= (1.0 - CLOUD_AGGRESSIVE_DISCOUNT_FACTOR)
            elif _job.priority == 'emergency':
                # Apply normal emergency discount
                deterrence *= (1.0 - _cloud_emergency_discount)
            op_compute_cost += deterrence

        # Projected completion time for THE ENTIRE JOB if this operation starts at _proposed_start.
        projected_op_end = _proposed_start + _operation.duration
        projected_job_completion = projected_op_end + _remaining_ops_duration

        # Calculate the delay penalty for THE ENTIRE JOB.
        delay = max(0, projected_job_completion - _job.deadline)

        # Apply priority multiplier to the delay penalty to reflect job urgency.
        priority_factor = _priority_multipliers.get(_job.priority, 1.0)
        delay_penalty = delay * _job.delay_weight * priority_factor * _inst.weights['delay_penalty']

        # Total score for this specific operation placement.
        total_score = (
            op_compute_cost * _inst.weights['compute_cost']
            + delay_penalty
        )

        # Tie-breaker: prefer earlier starts slightly if costs are otherwise equal.
        # A small positive coefficient for proposed_start ensures that given two options
        # with identical weighted cost, the one starting earlier is preferred, promoting flow.
        total_score += _proposed_start * 0.0001

        return total_score, projected_job_completion # Return projected completion for critical delay check

    schedule[job.id] = []
    # op_start_time_predecessor tracks the earliest an operation can start based on
    # the completion time of the previous operation in this job, and the job's arrival time.
    op_start_time_predecessor = max(job.arrival_time, current_time)

    for idx, op in enumerate(job.operations):
        # Calculate the remaining duration for subsequent operations in this job.
        # This is crucial for accurately projecting the job's overall completion time
        # and thus its delay penalty.
        remaining_operations_duration = sum(o.duration for o in job.operations[idx + 1:])

        best_score_local = float('inf')
        chosen_server_local = None
        chosen_start_local = None
        projected_job_completion_if_local = float('inf')

        best_score_cloud = float('inf')
        chosen_server_cloud = None
        chosen_start_cloud = None
        projected_job_completion_if_cloud = float('inf')


        # --- Phase 1: Evaluate Local Options (Best-Fit Search) ---
        for server_id, server in inst.servers.items():
            if not server.is_cloud and server.server_type == op.server_type:
                search_end_time = min(job.deadline + inst.time_horizon,
                                      op_start_time_predecessor + MAX_LOCAL_SEARCH_WINDOW)

                for proposed_start in range(op_start_time_predecessor, search_end_time + 1):
                    if has_capacity(inst, capacity_usage, server_id, proposed_start, op.duration, op.capacity_required):
                        score, projected_completion = _calculate_placement_score(
                            inst, job, op, proposed_start, server_id, False, # Not a cloud server
                            remaining_operations_duration, PRIORITY_MULTIPLIERS,
                            CLOUD_DETERRENCE_BASE, CLOUD_EMERGENCY_DISCOUNT
                        )
                        if score < best_score_local:
                            best_score_local = score
                            chosen_server_local = server_id
                            chosen_start_local = proposed_start
                            projected_job_completion_if_local = projected_completion


        # --- Phase 2: Evaluate Cloud Options (Immediate Start with Standard Deterrence) ---
        for server_id, server in inst.servers.items():
            if server.is_cloud and server.server_type == op.server_type:
                proposed_start = op_start_time_predecessor
                score, projected_completion = _calculate_placement_score(
                    inst, job, op, proposed_start, server_id, True, # Is a cloud server
                    remaining_operations_duration, PRIORITY_MULTIPLIERS,
                    CLOUD_DETERRENCE_BASE, CLOUD_EMERGENCY_DISCOUNT
                )
                # We typically assume one 'best' cloud option (e.g., lowest cost).
                # If multiple cloud options existed with different costs, this would need refinement.
                if score < best_score_cloud:
                    best_score_cloud = score
                    chosen_server_cloud = server_id
                    chosen_start_cloud = proposed_start
                    projected_job_completion_if_cloud = projected_completion
                break # Assuming one cloud server per type for simplicity, or taking the first best.


        # --- Phase 3: Conditional Cloud Fallback Logic ---
        final_chosen_server = None
        final_chosen_start = None

        # Determine if a critical delay is projected with the best local option
        is_critical_delay_with_local = (
            chosen_server_local is not None and
            projected_job_completion_if_local > (job.deadline + CRITICAL_DELAY_THRESHOLD_HOURS)
        )

        # If no local option was found at all, we effectively have a critical situation requiring cloud
        no_local_option_found = (chosen_server_local is None)

        if (is_critical_delay_with_local or no_local_option_found) and chosen_server_cloud is not None:
            # Re-calculate cloud score with aggressive discount
            aggressive_cloud_score, _ = _calculate_placement_score(
                inst, job, op, chosen_start_cloud, chosen_server_cloud, True,
                remaining_operations_duration, PRIORITY_MULTIPLIERS,
                CLOUD_DETERRENCE_BASE, CLOUD_EMERGENCY_DISCOUNT,
                _aggressive_discount_active=True # Activate aggressive discount
            )
            
            # If the aggressively discounted cloud is better than the best local (or no local was found)
            if aggressive_cloud_score < best_score_local: # Compares aggressively priced cloud to best local
                final_chosen_server = chosen_server_cloud
                final_chosen_start = chosen_start_cloud
            else:
                # If aggressive cloud still isn't better than local, stick to local if it exists
                if chosen_server_local is not None:
                    final_chosen_server = chosen_server_local
                    final_chosen_start = chosen_start_local
                else: # Fallback to standard cloud if no local was found and aggressive cloud wasn't strictly better
                    final_chosen_server = chosen_server_cloud
                    final_chosen_start = chosen_start_cloud
        else:
            # No critical delay detected, or no cloud option exists.
            # Choose between the best standard local and best standard cloud options.
            if best_score_local <= best_score_cloud:
                final_chosen_server = chosen_server_local
                final_chosen_start = chosen_start_local
            else:
                final_chosen_server = chosen_server_cloud
                final_chosen_start = chosen_start_cloud

        # --- Final Selection Validation ---
        if final_chosen_server is None:
            # This indicates a severe problem, likely no compatible server type exists or infinite costs.
            raise Exception(f"Could not schedule operation {idx} for job {job.id}: No compatible server (local or cloud) found after exhaustive search.")


        # Reserve capacity for the chosen operation in the `capacity_usage` dictionary.
        for t in range(final_chosen_start, final_chosen_start + op.duration):
            capacity_usage[final_chosen_server][t] = capacity_usage[final_chosen_server].get(t, 0) + op.capacity_required

        # Add the scheduled operation to the job's schedule.
        schedule[job.id].append({"server": final_chosen_server, "start": final_chosen_start})

        # Update the start time for the next operation in this job.
        # It must start after the current operation finishes.
        op_start_time_predecessor = final_chosen_start + op.duration

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
