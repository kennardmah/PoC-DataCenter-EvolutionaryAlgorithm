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
class OperationScheduler:
    """
    Schedules a single operation using a contention-aware, urgency-driven heuristic.
    It assesses future resource contention to make robust local vs. cloud decisions.
    """
    PRIORITY_MAP = {"emergency": 6, "critical": 5, "urgent": 4, "high": 3, "medium": 2, "low": 1}
    HOTSPOT_PENALTY_FACTOR = 15.0  # Penalty for creating a future capacity hotspot (>80% utilization)
    PROCRASTINATION_PENALTY_FACTOR = 0.05 # Penalty for waiting, applied per hour
    CRITICAL_UTIL_THRESHOLD = 0.8  # Utilization threshold for hotspot penalty

    def __init__(self, inst, job, op, capacity_usage, earliest_start, remaining_duration):
        self.inst = inst
        self.job = job
        self.op = op
        self.capacity_usage = capacity_usage
        self.earliest_start = earliest_start
        self.remaining_duration = remaining_duration
        self.total_job_duration_left = op.duration + remaining_duration
        self.unified_urgency = self._get_unified_urgency()

    def _get_unified_urgency(self) -> float:
        """Calculates a continuous urgency score based on priority, slack, and workload."""
        priority_val = self.PRIORITY_MAP.get(self.job.priority, 2.0)
        projected_completion = self.earliest_start + self.total_job_duration_left
        slack = self.job.deadline - projected_completion

        # Non-linear penalty for negative slack
        if slack < 0:
            slack_penalty = math.exp(-slack / (TIME_HORIZON / 2.0))
        else:
            slack_penalty = max(0.0, 1.0 - (slack / (TIME_HORIZON * 2)))

        return priority_val * (1.0 + slack_penalty)

    def _calculate_contention_penalty(self, server_id: str, start_time: int) -> float:
        """Penalizes placements that create future resource hotspots."""
        penalty = 0.0
        lookahead_horizon = start_time + self.op.duration + TIME_HORIZON // 4 # Look ahead a bit

        for t in range(start_time, lookahead_horizon):
            available_cap = get_available_capacity(self.inst, server_id, t)
            if available_cap <= 0:
                continue

            used_cap = self.capacity_usage.get(server_id, {}).get(t, 0)
            future_util = (used_cap + self.op.capacity_required) / available_cap

            if future_util > self.CRITICAL_UTIL_THRESHOLD:
                # Quadratic penalty for pushing utilization over the threshold
                penalty += ((future_util - self.CRITICAL_UTIL_THRESHOLD)**2) * self.HOTSPOT_PENALTY_FACTOR

        # Scale by job's importance
        return penalty * self.unified_urgency

    def _calculate_placement_score(self, server_id: str, start_time: int) -> float:
        """Calculates a holistic score for a potential placement."""
        server = self.inst.servers[server_id]

        compute_cost = sum(self.inst.server_cost[server_id][t % TIME_HORIZON]
                           for t in range(start_time, start_time + self.op.duration))
        projected_completion = start_time + self.op.duration + self.remaining_duration
        delay = max(0, projected_completion - self.job.deadline)
        delay_penalty = delay * self.job.delay_weight

        wait_time = start_time - self.earliest_start
        procrastination_penalty = wait_time * self.PROCRASTINATION_PENALTY_FACTOR * self.unified_urgency

        contention_penalty = 0.0
        if not server.is_cloud:
            contention_penalty = self._calculate_contention_penalty(server_id, start_time)

        total_score = (compute_cost * self.inst.weights['compute_cost'] +
                       delay_penalty * self.inst.weights['delay_penalty'] +
                       procrastination_penalty +
                       contention_penalty)
        return total_score

    def find_best_placement(self):
        """Finds the best server and start time using an iterative, urgency-driven search."""
        best_placement = (float('inf'), None, None) # (score, server_id, start_time)

        # 1. Establish cloud benchmark
        cloud_options = [s_id for s_id, s in self.inst.servers.items()
                         if s.server_type == self.op.server_type and s.is_cloud]
        if cloud_options:
            cheapest_cloud = min(cloud_options, key=lambda s_id: self.inst.server_cost[s_id][self.earliest_start % TIME_HORIZON])
            cloud_score = self._calculate_placement_score(cheapest_cloud, self.earliest_start)
            best_placement = (cloud_score, cheapest_cloud, self.earliest_start)

        # 2. Iterative search on local servers
        local_options = [s_id for s_id, s in self.inst.servers.items()
                         if s.server_type == self.op.server_type and not s.is_cloud]

        # Search horizon scales inversely with urgency
        initial_horizon = max(self.op.duration, int(TIME_HORIZON * 3 / max(1, self.unified_urgency)))
        step_horizon = max(self.op.duration, int(TIME_HORIZON / max(1, self.unified_urgency)))
        max_iter = 4

        search_end = self.earliest_start + initial_horizon

        for i in range(max_iter):
            found_in_iter = False
            for server_id in local_options:
                for start_t in range(self.earliest_start, search_end):
                    if has_capacity(self.inst, self.capacity_usage, server_id, start_t, self.op.duration, self.op.capacity_required):
                        score = self._calculate_placement_score(server_id, start_t)
                        if score < best_placement[0]:
                            best_placement = (score, server_id, start_t)
                            found_in_iter = True
                        break # Found first-fit in window for this server, move to next

            # If we found a local placement better than the cloud, we can be more confident and stop early
            if found_in_iter and not self.inst.servers[best_placement[1]].is_cloud:
                break

            search_end += step_horizon

        # 3. Failsafe if no placement found
        if best_placement[1] is None:
            if local_options: # Should only happen if no cloud and extreme congestion
                cheapest_local = min(local_options, key=lambda s: sum(self.inst.server_cost[s].values()))
                start_t = self.earliest_start
                while True:
                    if has_capacity(self.inst, self.capacity_usage, cheapest_local, start_t, self.op.duration, self.op.capacity_required):
                        score = self._calculate_placement_score(cheapest_local, start_t)
                        best_placement = (score, cheapest_local, start_t)
                        break
                    start_t += 1
            else:
                raise RuntimeError(f"No compatible servers for job {self.job.id}")

        return best_placement[1], best_placement[2]

def schedule_job_online(inst: Instance,
                        job: Job,
                        schedule: Schedule,
                        capacity_usage: Dict[str, Dict[int, int]],
                        current_time: int):
    """
    Orchestrates the scheduling of a single job by using the OperationScheduler
    for each of its operations.
    """
    schedule[job.id] = []
    current_op_release_time = max(job.arrival_time, current_time)

    for idx, op in enumerate(job.operations):
        remaining_duration = sum(o.duration for o in job.operations[idx + 1:])

        scheduler = OperationScheduler(
            inst=inst,
            job=job,
            op=op,
            capacity_usage=capacity_usage,
            earliest_start=current_op_release_time,
            remaining_duration=remaining_duration
        )

        chosen_server, chosen_start = scheduler.find_best_placement()

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