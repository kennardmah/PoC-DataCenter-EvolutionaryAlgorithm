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

@dataclass
class PlacementCandidate:
    score: float
    server_id: str
    start_time: int
    is_cloud: bool

class PlacementEvaluator:
    """
    Evaluates and selects the best placement for a single operation, considering
    multiple cost factors, job urgency, and resource availability.
    """
    # --- Tunable Heuristic Parameters ---
    BASE_LOCAL_SEARCH_HORIZON = 48 # Base hours to look ahead for local machines
    PROCR_PENALTY_FACTOR = 0.05
    DEADLINE_PROXIMITY_FACTOR = 0.5
    LOAD_BALANCE_PENALTY_FACTOR = 0.1
    LOAD_BALANCE_LOOKAHEAD = 3 # Hours to look ahead for congestion

    # Priority-specific settings: horizon_mult, cloud_aversion, delay_scale, procr_mult, load_penalty_factor
    PRIORITY_SETTINGS = {
        "emergency": {"horizon_mult": 0.5, "cloud_aversion": 0.7, "delay_scale": 5.0, "procr_mult": 2.0, "load_penalty_factor": 0.02},
        "urgent":    {"horizon_mult": 0.8, "cloud_aversion": 0.8, "delay_scale": 2.0, "procr_mult": 1.5, "load_penalty_factor": 0.05},
        "high":      {"horizon_mult": 1.0, "cloud_aversion": 0.9, "delay_scale": 1.0, "procr_mult": 1.0, "load_penalty_factor": 0.1},
        "medium":    {"horizon_mult": 1.2, "cloud_aversion": 1.0, "delay_scale": 0.8, "procr_mult": 0.8, "load_penalty_factor": 0.15},
        "low":       {"horizon_mult": 1.5, "cloud_aversion": 1.2, "delay_scale": 0.5, "procr_mult": 0.5, "load_penalty_factor": 0.2},
    }

    def __init__(self, inst, job, op, capacity_usage, earliest_start, remaining_duration):
        self.inst = inst
        self.job = job
        self.op = op
        self.capacity_usage = capacity_usage
        self.earliest_start = earliest_start
        self.remaining_duration = remaining_duration
        self.priority_config = self.PRIORITY_SETTINGS.get(job.priority.lower(), self.PRIORITY_SETTINGS["medium"])

    def _calculate_score(self, server_id: str, start_time: int, is_cloud: bool) -> float:
        """Calculates a composite score for a potential placement, adjusted by job urgency."""
        # 1. Compute Cost
        compute_cost = sum(self.inst.server_cost[server_id][t % self.inst.time_horizon]
                           for t in range(start_time, start_time + self.op.duration)) * self.inst.weights['compute_cost']

        # 2. Delay Penalty
        projected_completion = start_time + self.op.duration + self.remaining_duration
        delay = max(0, projected_completion - self.job.deadline)
        delay_cost = delay * self.job.delay_weight * self.inst.weights['delay_penalty'] * self.priority_config["delay_scale"]

        # 3. Procrastination Penalty (penalize waiting unnecessarily)
        wait_time = max(0, start_time - self.earliest_start)
        procrastination_penalty = wait_time * self.PROCR_PENALTY_FACTOR * self.priority_config["procr_mult"]

        # 4. Deadline Proximity Penalty (escalates as completion nears/passes deadline)
        deadline_proximity_penalty = 0.0
        remaining_slack_at_completion = self.job.deadline - projected_completion
        if remaining_slack_at_completion < 0: # Already late
            deadline_proximity_penalty = abs(remaining_slack_at_completion) * self.DEADLINE_PROXIMITY_FACTOR * self.priority_config["delay_scale"] * 2 # Heavier penalty for being late
        elif remaining_slack_at_completion < self.op.duration * 2: # Close to deadline (within 2x operation duration)
            deadline_proximity_penalty = (self.op.duration * 2 - remaining_slack_at_completion) * self.DEADLINE_PROXIMITY_FACTOR * self.priority_config["delay_scale"]

        total_score = compute_cost + delay_cost + procrastination_penalty + deadline_proximity_penalty

        # 5. Load-Balancing Penalty (for local machines only)
        if not is_cloud:
            load_penalty = 0.0
            # Look ahead for congestion around the operation's scheduled time
            for t in range(start_time, start_time + self.op.duration + self.LOAD_BALANCE_LOOKAHEAD):
                current_used = self.capacity_usage[server_id].get(t, 0)
                available_cap = get_available_capacity(self.inst, server_id, t)
                if available_cap > 0:
                    utilization = (current_used + self.op.capacity_required) / available_cap
                    # Penalize more for higher utilization, especially above a threshold
                    load_penalty += max(0, utilization - 0.5) * self.LOAD_BALANCE_PENALTY_FACTOR * self.priority_config["load_penalty_factor"]
            total_score += load_penalty

        # 6. Apply Cloud Aversion Factor
        if is_cloud:
            total_score *= self.priority_config["cloud_aversion"]

        return total_score

    def find_best_placement(self) -> PlacementCandidate:
        """Finds the best placement by evaluating local and cloud options."""
        best_candidate = PlacementCandidate(float('inf'), None, -1, False)

        # --- Evaluate Cloud Options ---
        compatible_cloud_servers = [s for s in self.inst.servers.values() if s.server_type == self.op.server_type and s.is_cloud]
        for server in compatible_cloud_servers:
            score = self._calculate_score(server.id, self.earliest_start, is_cloud=True)
            if score < best_candidate.score:
                best_candidate = PlacementCandidate(score, server.id, self.earliest_start, True)

        # --- Evaluate Local Options ---
        compatible_local_servers = [s for s in self.inst.servers.values() if s.server_type == self.op.server_type and not s.is_cloud]

        # Determine dynamic search horizon for local machines
        total_remaining_job_duration = self.op.duration + self.remaining_duration
        current_job_slack = self.job.deadline - (self.earliest_start + total_remaining_job_duration)

        # Adjust base horizon by priority and slack
        horizon_multiplier = self.priority_config["horizon_mult"]
        slack_adjustment_factor = 1.0 + min(1.0, max(-0.5, current_job_slack / (self.inst.time_horizon * 2))) # Limit adjustment range

        effective_local_search_horizon = int(self.BASE_LOCAL_SEARCH_HORIZON * horizon_multiplier * slack_adjustment_factor)
        effective_local_search_horizon = max(self.op.duration, effective_local_search_horizon) # Ensure at least op duration

        search_limit = self.earliest_start + effective_local_search_horizon

        for server in compatible_local_servers:
            # Perform a best-fit search for this local server within its dynamic horizon
            for start_time_candidate in range(self.earliest_start, search_limit):
                if has_capacity(self.inst, self.capacity_usage, server.id, start_time_candidate, self.op.duration, self.op.capacity_required):
                    score = self._calculate_score(server.id, start_time_candidate, is_cloud=False)
                    if score < best_candidate.score:
                        best_candidate = PlacementCandidate(score, server.id, start_time_candidate, False)

        # --- Failsafe: if no placement found within horizon or best score is still infinite ---
        if best_candidate.server_id is None:
            # If no local servers or no local slot ever found, and no cloud was chosen, force cloud if available
            if compatible_cloud_servers:
                # Re-evaluate cloud with the original (unmodified) cloud_aversion for failsafe choice
                cheapest_cloud_server_id = min(compatible_cloud_servers,
                                                key=lambda s: sum(self.inst.server_cost[s.id].values())).id
                score = self._calculate_score(cheapest_cloud_server_id, self.earliest_start, is_cloud=True)
                # Override cloud aversion to ensure it's chosen if no other option
                best_candidate = PlacementCandidate(score / self.priority_config["cloud_aversion"], cheapest_cloud_server_id, self.earliest_start, True)
                return best_candidate

            # As a last resort, unbounded search on the cheapest local machine
            if compatible_local_servers:
                cheapest_local_server_id = min(compatible_local_servers,
                                               key=lambda s: sum(self.inst.server_cost[s.id].values())).id

                failsafe_start_time = self.earliest_start
                while True: # Unbounded search
                    if has_capacity(self.inst, self.capacity_usage, cheapest_local_server_id, failsafe_start_time, self.op.duration, self.op.capacity_required):
                        score = self._calculate_score(cheapest_local_server_id, failsafe_start_time, is_cloud=False)
                        return PlacementCandidate(score, cheapest_local_server_id, failsafe_start_time, False)
                    failsafe_start_time += 1


            raise Exception(f"CRITICAL ERROR: No compatible servers (local or cloud) found for job {self.job.id} operation of type {self.op.server_type}.")

        return best_candidate


def schedule_job_online(inst: Instance,
                        job: Job,
                        schedule: Schedule,
                        capacity_usage: Dict[str, Dict[int, int]],
                        current_time: int):

    schedule[job.id] = []
    current_op_earliest_start = max(job.arrival_time, current_time)

    for idx, op in enumerate(job.operations):
        remaining_duration = sum(o.duration for o in job.operations[idx + 1:])

        placement_evaluator = PlacementEvaluator(
            inst=inst,
            job=job,
            op=op,
            capacity_usage=capacity_usage,
            earliest_start=current_op_earliest_start,
            remaining_duration=remaining_duration
        )

        chosen_placement = placement_evaluator.find_best_placement()

        for t in range(chosen_placement.start_time, chosen_placement.start_time + op.duration):
            capacity_usage[chosen_placement.server_id][t] = capacity_usage[chosen_placement.server_id].get(t, 0) + op.capacity_required

        schedule[job.id].append({"server": chosen_placement.server_id, "start": chosen_placement.start_time})
        current_op_earliest_start = chosen_placement.start_time + op.duration

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