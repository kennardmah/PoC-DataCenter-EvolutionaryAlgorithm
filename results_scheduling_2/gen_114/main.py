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

    class OperationScheduler:
        """
        Encapsulates heuristic logic for evaluating and selecting the best placement
        for a single operation within a job.
        """
        # Tunable Parameters as Class Attributes
        PRIORITY_MULTIPLIERS = {
            'emergency': 10.0, # High impact for emergency
            'high': 4.0,
            'medium': 1.0,
            'low': 0.5,
        }
        # Base cloud deterrence, dynamically adjusted later.
        CLOUD_DETERRENCE_INITIAL_BASE = 80.0
        CLOUD_EMERGENCY_DISCOUNT_FACTOR = 0.7 # Reduce cloud deterrence by this factor for emergency jobs
        CONGESTION_CLOUD_ADJUSTMENT_FACTOR = 0.1 # How much congestion affects cloud cost (0 to 1)
        NEXT_OP_PREVIEW_COST_FACTOR = 0.1 # How much to factor in the next operation's estimated cost
        TIE_BREAKER_EARLY_START_FACTOR = 0.0001
        TIE_BREAKER_CAPACITY_FACTOR = 0.000001 # Small factor to prefer options that leave more capacity

        def __init__(self, inst_ref: Instance, capacity_usage_ref: Dict[str, Dict[int, int]], current_time_ref: int):
            self.inst = inst_ref
            self.capacity_usage = capacity_usage_ref
            self.current_time = current_time_ref

            # Pre-process servers for faster lookup
            self.servers_by_type = defaultdict(lambda: {'local': [], 'cloud': []})
            for s_id, server in inst_ref.servers.items():
                if server.is_cloud:
                    self.servers_by_type[server.server_type]['cloud'].append(server)
                else:
                    self.servers_by_type[server.server_type]['local'].append(server)

        def _get_available_capacity(self, server_id: str, time: int) -> int:
            """Wrapper for generic capacity check, using instance-level method."""
            return _get_available_capacity_generic(self.inst, server_id, time)

        def _has_capacity(self, server_id: str, start: int, duration: int, req: int) -> bool:
            """Wrapper for generic capacity check, using instance-level method."""
            return _has_capacity_generic(self.inst, self.capacity_usage, server_id, start, duration, req)

        def _get_local_congestion(self, server_type: str, time: int) -> float:
            """
            Estimates congestion for a given local server type at a specific time.
            Returns a value between 0 (no congestion) and 1 (full congestion).
            """
            total_type_capacity = 0
            total_type_used = 0
            for server in self.servers_by_type[server_type]['local']:
                total_type_capacity += self._get_available_capacity(server.id, time) # Current available, considering failures
                total_type_used += self.capacity_usage[server.id].get(time, 0)

            if total_type_capacity == 0:
                return 1.0 # Fully congested if no capacity available

            # Congestion is (capacity currently used) / (total available capacity for type)
            # Cap at 1.0 to prevent issues if usage temporarily exceeds initial base capacity due to failures.
            return min(1.0, total_type_used / total_type_capacity if total_type_capacity > 0 else 1.0)

        def _calculate_placement_score(
            self,
            job_ref: Job,
            operation: Operation,
            proposed_start: int,
            server_id: str,
            is_cloud_server: bool,
            remaining_ops_duration: int,
            next_op_estimated_cost: float
        ) -> float:
            """
            Calculates a weighted score for a proposed operation placement,
            considering compute cost, delay penalty, dynamic cloud deterrence,
            and proactive intra-job dependency cost.
            """
            # Compute cost for THIS operation
            op_compute_cost = sum(self.inst.server_cost[server_id][t % self.inst.time_horizon]
                                  for t in range(proposed_start, proposed_start + operation.duration))

            # Dynamic Cloud Deterrence
            if is_cloud_server:
                # Adjust base deterrence based on local congestion
                congestion = self._get_local_congestion(operation.server_type, proposed_start)
                # If local is congested (congestion closer to 1), reduce deterrence
                # If local is free (congestion closer to 0), increase deterrence
                dynamic_deterrence = self.CLOUD_DETERRENCE_INITIAL_BASE * (1 + (1 - congestion) * self.CONGESTION_CLOUD_ADJUSTMENT_FACTOR)

                if job_ref.priority == 'emergency':
                    dynamic_deterrence *= (1.0 - self.CLOUD_EMERGENCY_DISCOUNT_FACTOR)
                op_compute_cost += dynamic_deterrence

            # Projected completion time for THE ENTIRE JOB
            projected_op_end = proposed_start + operation.duration
            projected_job_completion = projected_op_end + remaining_ops_duration

            # Calculate the delay penalty for THE ENTIRE JOB.
            delay = max(0, projected_job_completion - job_ref.deadline)

            # Apply priority multiplier to the delay penalty.
            priority_factor = self.PRIORITY_MULTIPLIERS.get(job_ref.priority, 1.0)
            delay_penalty = delay * job_ref.delay_weight * priority_factor * self.inst.weights['delay_penalty']

            # Total score for this specific operation placement.
            total_score = (
                op_compute_cost * self.inst.weights['compute_cost']
                + delay_penalty
                + next_op_estimated_cost * self.NEXT_OP_PREVIEW_COST_FACTOR # Proactive dependency cost
            )

            # Tie-breakers:
            # 1. Prefer earlier starts
            total_score += proposed_start * self.TIE_BREAKER_EARLY_START_FACTOR
            
            # 2. For local servers, prefer choices that leave more capacity
            if not is_cloud_server:
                remaining_cap_after_op = self._get_available_capacity(server_id, proposed_start) - (self.capacity_usage[server_id].get(proposed_start, 0) + operation.capacity_required)
                total_score -= remaining_cap_after_op * self.TIE_BREAKER_CAPACITY_FACTOR # Subtract to make it smaller (better score)

            return total_score

        def _estimate_next_op_cost(self, job_ref: Job, next_op_idx: int, earliest_start: int) -> float:
            """
            A light simulation to estimate the minimum compute cost of the next operation.
            This is a very rough estimate and doesn't consider capacity, just minimum cost.
            """
            if next_op_idx >= len(job_ref.operations):
                return 0.0

            next_op = job_ref.operations[next_op_idx]
            min_cost = float('inf')

            # Consider cloud first for a quick estimate
            for server in self.servers_by_type[next_op.server_type]['cloud']:
                cost = sum(self.inst.server_cost[server.id][t % self.inst.time_horizon]
                           for t in range(earliest_start, earliest_start + next_op.duration))
                min_cost = min(min_cost, cost)

            # If no cloud or if local is cheaper, consider local options' base cost
            for server in self.servers_by_type[next_op.server_type]['local']:
                cost = sum(self.inst.server_cost[server.id][t % self.inst.time_horizon]
                           for t in range(earliest_start, earliest_start + next_op.duration))
                min_cost = min(min_cost, cost)
            
            return min_cost if min_cost != float('inf') else 0.0


        def find_best_placement(self, job_ref: Job, op_idx: int, op: Operation, op_start_time_predecessor: int) -> Dict[str, Any]:
            """
            Finds the best server and start time for a given operation.
            """
            remaining_ops_duration = sum(o.duration for o in job_ref.operations[op_idx + 1:])
            next_op_estimated_cost = self._estimate_next_op_cost(job_ref, op_idx + 1, op_start_time_predecessor + op.duration)

            best_score = float('inf')
            best_placement = None # (chosen_server_id, chosen_start_time)

            # Determine local search window based on priority
            MAX_LOCAL_SEARCH_WINDOW = self.inst.time_horizon * 2 + 12 # Base window
            if job_ref.priority == 'emergency':
                MAX_LOCAL_SEARCH_WINDOW *= 1.5 # Extend window for emergency jobs
            elif job_ref.priority == 'high':
                MAX_LOCAL_SEARCH_WINDOW *= 1.2
            
            search_end_time_local = min(job_ref.deadline + self.inst.time_horizon,
                                        op_start_time_predecessor + int(MAX_LOCAL_SEARCH_WINDOW))

            candidate_placements = [] # Store (score, server_id, start_time)

            # --- Evaluate Cloud Options ---
            for server in self.servers_by_type[op.server_type]['cloud']:
                proposed_start = op_start_time_predecessor
                score = self._calculate_placement_score(
                    job_ref, op, proposed_start, server.id, True,
                    remaining_ops_duration, next_op_estimated_cost
                )
                candidate_placements.append((score, server.id, proposed_start))

            # --- Evaluate Local Options (Best-Fit within Search Horizon) ---
            for server in self.servers_by_type[op.server_type]['local']:
                local_server_best_score = float('inf')
                local_server_best_start = -1

                for proposed_start in range(op_start_time_predecessor, search_end_time_local + 1):
                    if self._has_capacity(server.id, proposed_start, op.duration, op.capacity_required):
                        score = self._calculate_placement_score(
                            job_ref, op, proposed_start, server.id, False,
                            remaining_ops_duration, next_op_estimated_cost
                        )
                        if score < local_server_best_score:
                            local_server_best_score = score
                            local_server_best_start = proposed_start
                
                if local_server_best_start != -1:
                    candidate_placements.append((local_server_best_score, server.id, local_server_best_start))

            # --- Select the overall best option ---
            if not candidate_placements:
                # This should ideally not happen if cloud is always an option
                raise Exception(f"No feasible placement found for operation {op_idx} of job {job_ref.id} (type {op.server_type}) starting at {op_start_time_predecessor}. Check instance config.")

            best_score, chosen_server_id, chosen_start_time = min(candidate_placements, key=lambda x: x[0])
            
            return {"server": chosen_server_id, "start": chosen_start_time}

    # Instantiate the operation scheduler
    op_scheduler = OperationScheduler(inst, capacity_usage, current_time)

    schedule[job.id] = []
    op_start_time_predecessor = max(job.arrival_time, current_time)

    for idx, op in enumerate(job.operations):
        # Find the best placement for the current operation
        chosen_placement = op_scheduler.find_best_placement(job, idx, op, op_start_time_predecessor)
        chosen_server_id = chosen_placement["server"]
        chosen_start_time = chosen_placement["start"]

        # Reserve capacity for the chosen operation
        for t in range(chosen_start_time, chosen_start_time + op.duration):
            op_scheduler.capacity_usage[chosen_server_id][t] = op_scheduler.capacity_usage[chosen_server_id].get(t, 0) + op.capacity_required

        # Add to job's schedule
        schedule[job.id].append({"server": chosen_server_id, "start": chosen_start_time})

        # Update the start time for the next operation in this job
        op_start_time_predecessor = chosen_start_time + op.duration

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