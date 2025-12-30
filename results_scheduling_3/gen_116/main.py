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
class UrgencyProfile:
    urgency_score: float
    search_horizon: int
    cloud_aggressiveness_factor: float
    procrastination_multiplier: float


class JobScheduler:
    def __init__(self, inst: Instance, job: Job,
                 schedule: Schedule, capacity_usage: Dict[str, Dict[int, int]],
                 current_system_time: int):
        self.inst = inst
        self.job = job
        self.schedule = schedule
        self.capacity_usage = capacity_usage
        self.current_op_start_time = max(job.arrival_time, current_system_time)
        self.scheduled_job_ops = []

        # --- NEW: Tunable parameters for the urgency and scoring models ---
        self.hotspot_threshold = 0.8
        self.hotspot_penalty_factor = 25.0
        self.base_procrastination_penalty = 0.07

        # Compute the urgency profile for this job once
        self.urgency_profile = self._compute_urgency_profile()

    def _compute_urgency_profile(self) -> UrgencyProfile:
        """
        Computes a unified, continuous urgency profile for the job, which then drives
        all other heuristic parameters.
        """
        priority_map = {"emergency": 10.0, "critical": 6.0, "urgent": 4.0, "high": 2.0, "medium": 1.0, "low": 0.5}
        base_priority_urgency = priority_map.get(self.job.priority.lower(), 1.0)

        total_job_duration = sum(op.duration for op in self.job.operations)
        current_slack = self.job.deadline - (self.current_op_start_time + total_job_duration)

        slack_urgency_multiplier = 1.0
        if current_slack < 0:
            slack_urgency_multiplier = 1.0 + (math.exp(abs(current_slack) / 8.0) - 1.0)
        elif current_slack < total_job_duration * 0.5:
            slack_urgency_multiplier = 1.0 + ((total_job_duration * 0.5 - current_slack) / (total_job_duration * 0.5))**2 * 3.0

        urgency_score = base_priority_urgency * slack_urgency_multiplier
        urgency_score = max(0.5, min(20.0, urgency_score))

        search_horizon = int(72 / (1 + 0.2 * urgency_score))
        search_horizon = max(total_job_duration, search_horizon)

        cloud_aggressiveness_factor = 1.25 - (0.05 * urgency_score)
        cloud_aggressiveness_factor = max(0.85, min(1.4, cloud_aggressiveness_factor))

        procrastination_multiplier = 1.0 + urgency_score / 2.0

        return UrgencyProfile(
            urgency_score=urgency_score,
            search_horizon=search_horizon,
            cloud_aggressiveness_factor=cloud_aggressiveness_factor,
            procrastination_multiplier=procrastination_multiplier
        )

    def _calculate_operation_score(self, server_id: str, start_time: int, op: Operation,
                                   remaining_duration_after_op: int) -> float:
        """Calculates a composite score based on cost, delay, and proactive penalties."""
        server = self.inst.servers[server_id]

        compute_cost = sum(self.inst.server_cost[server_id][t % self.inst.time_horizon]
                           for t in range(start_time, start_time + op.duration))

        projected_completion_time = start_time + op.duration + remaining_duration_after_op
        delay = max(0, projected_completion_time - self.job.deadline)
        delay_cost = delay * self.job.delay_weight * self.inst.weights['delay_penalty']

        wait_time = start_time - self.current_op_start_time
        procrastination_penalty = wait_time * self.base_procrastination_penalty * self.urgency_profile.procrastination_multiplier

        slack_after_this_op = self.job.deadline - projected_completion_time
        critical_slack_buffer = remaining_duration_after_op * 1.2
        if slack_after_this_op < critical_slack_buffer:
            slack_shortfall = critical_slack_buffer - slack_after_this_op
            procrastination_penalty += slack_shortfall * self.job.delay_weight * self.urgency_profile.urgency_score * 0.1

        hotspot_penalty = 0.0
        if not server.is_cloud:
            max_utilization = 0
            for t in range(start_time, start_time + op.duration):
                used = self.capacity_usage.get(server_id, {}).get(t, 0)
                available = get_available_capacity(self.inst, server_id, t)
                if available > 0:
                    utilization = (used + op.capacity_required) / available
                    max_utilization = max(max_utilization, utilization)
            
            if max_utilization > self.hotspot_threshold:
                penalty_magnitude = ((max_utilization - self.hotspot_threshold) / (1.0 - self.hotspot_threshold))**2
                hotspot_penalty = penalty_magnitude * self.hotspot_penalty_factor * op.duration * self.urgency_profile.urgency_score
        
        total_score = (compute_cost * self.inst.weights['compute_cost'] +
                       delay_cost +
                       procrastination_penalty +
                       hotspot_penalty)
        return total_score

    def _find_best_cloud_placement(self, op: Operation, remaining_duration_after_op: int) -> Optional[tuple]:
        best_score, best_server_id = float('inf'), None
        for server_id, server in self.inst.servers.items():
            if server.server_type == op.server_type and server.is_cloud:
                score = self._calculate_operation_score(server_id, self.current_op_start_time, op, remaining_duration_after_op)
                if score < best_score:
                    best_score, best_server_id = score, server_id
        return (best_score, best_server_id, self.current_op_start_time) if best_server_id else None

    def _find_best_local_placement(self, op: Operation, remaining_duration_after_op: int) -> Optional[tuple]:
        best_score, best_placement = float('inf'), None
        search_limit = self.current_op_start_time + self.urgency_profile.search_horizon

        for server_id, server in self.inst.servers.items():
            if server.server_type == op.server_type and not server.is_cloud:
                for start in range(self.current_op_start_time, search_limit + 1):
                    if has_capacity(self.inst, self.capacity_usage, server_id, start, op.duration, op.capacity_required):
                        score = self._calculate_operation_score(server_id, start, op, remaining_duration_after_op)
                        if score < best_score:
                            best_score, best_placement = score, (score, server_id, start)
        return best_placement

    def schedule_all_operations(self):
        for idx, op in enumerate(self.job.operations):
            remaining_duration_after_op = sum(o.duration for o in self.job.operations[idx + 1:])

            cloud_res = self._find_best_cloud_placement(op, remaining_duration_after_op)
            local_res = self._find_best_local_placement(op, remaining_duration_after_op)

            cloud_score = cloud_res[0] if cloud_res else float('inf')
            local_score = local_res[0] if local_res else float('inf')

            chosen_server, chosen_start = None, -1
            cloud_pref_factor = self.urgency_profile.cloud_aggressiveness_factor

            if local_res and local_score < (cloud_score * cloud_pref_factor):
                chosen_server, chosen_start = local_res[1], local_res[2]
            elif cloud_res:
                chosen_server, chosen_start = cloud_res[1], cloud_res[2]
            else:
                compatible_servers = [s for s in self.inst.servers.values() if s.server_type == op.server_type]
                if any(s.is_cloud for s in compatible_servers):
                    cheapest_cloud = min((s for s in compatible_servers if s.is_cloud),
                                         key=lambda s: self.inst.server_cost[s.id][self.current_op_start_time % self.inst.time_horizon])
                    chosen_server, chosen_start = cheapest_cloud.id, self.current_op_start_time
                else:
                    cheapest_local = min((s for s in compatible_servers if not s.is_cloud),
                                         key=lambda s: sum(self.inst.server_cost[s.id].values()))
                    start_t = self.current_op_start_time
                    while True:
                        if has_capacity(self.inst, self.capacity_usage, cheapest_local.id, start_t, op.duration, op.capacity_required):
                            chosen_server, chosen_start = cheapest_local.id, start_t
                            break
                        start_t += 1

            for t in range(chosen_start, chosen_start + op.duration):
                self.capacity_usage[chosen_server][t] = self.capacity_usage[chosen_server].get(t, 0) + op.capacity_required

            self.scheduled_job_ops.append({"server": chosen_server, "start": chosen_start})
            self.current_op_start_time = chosen_start + op.duration

        self.schedule[self.job.id] = self.scheduled_job_ops


def schedule_job_online(inst: Instance,
                        job: Job,
                        schedule: Schedule,
                        capacity_usage: Dict[str, Dict[int, int]],
                        current_time: int):
    """Orchestrates scheduling for a single job using the JobScheduler class."""
    job_scheduler = JobScheduler(inst, job, schedule, capacity_usage, current_time)
    job_scheduler.schedule_all_operations()

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