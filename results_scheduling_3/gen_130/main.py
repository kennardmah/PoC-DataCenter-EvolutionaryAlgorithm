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


class JobUrgencyProfile:
    """
    Calculates and holds a dynamic urgency profile for a job, which in turn
    derives key scheduling parameters.
    """
    def __init__(self, job: Job, current_time: int, total_job_duration: int, system_load: float):
        # Base priority weights (higher is more urgent)
        priority_map = {"emergency": 5.0, "urgent": 3.0, "high": 2.0, "medium": 1.0, "low": 0.5}
        base_priority_urgency = priority_map.get(job.priority.lower(), 1.0)

        current_slack = job.deadline - (current_time + total_job_duration)
        slack_urgency_multiplier = 1.0
        if current_slack < 0:
            slack_urgency_multiplier = 1.0 + abs(current_slack) / 6.0
        elif current_slack < TIME_HORIZON / 3: # If slack is less than 8 hours
            slack_urgency_multiplier = 1.0 + (TIME_HORIZON / 3 - current_slack) / (TIME_HORIZON / 3)

        self.urgency_score = base_priority_urgency * slack_urgency_multiplier
        self.urgency_score = max(0.5, min(10.0, self.urgency_score))

        # 1. Cloud Aggressiveness: Higher urgency -> lower factor -> favors cloud speed
        #    Also reduce factor if system-wide local load is high, making cloud more attractive
        self.cloud_aggressiveness_factor = 1.25 - (0.1 * self.urgency_score)
        system_load_influence = (system_load - 0.5) * 0.2 # E.g., if load is 0.8, influence is 0.06 (reduce factor)
        self.cloud_aggressiveness_factor -= system_load_influence # Reduce factor if system_load > 0.5, increase if < 0.5
        self.cloud_aggressiveness_factor = max(0.8, min(1.3, self.cloud_aggressiveness_factor))

        # 2. Local Search Horizon: Higher urgency -> shorter horizon to find a slot quickly
        self.search_horizon = int(72 / (1 + 0.15 * self.urgency_score))
        self.search_horizon = max(job.operations[0].duration + 1, self.search_horizon)

        # 3. Procrastination Penalty Multiplier: Higher urgency -> greater penalty for waiting
        self.procrastination_penalty_multiplier = 1.0 + self.urgency_score / 2.0


class PlacementDecisionMaker:
    """Finds the best placement for a single operation using an urgency-driven scoring model."""

    def __init__(self, inst, job, op, capacity_usage, earliest_start, remaining_duration, urgency_profile: JobUrgencyProfile):
        self.inst = inst
        self.job = job
        self.op = op
        self.capacity_usage = capacity_usage
        self.earliest_start = earliest_start
        self.remaining_duration = remaining_duration
        self.urgency = urgency_profile
        self.base_procrastination_penalty = 0.05

        # New Load Balancing Parameters
        self.high_utilization_threshold = 0.7
        self.high_utilization_penalty_factor = 250.0 # Multiplier for cubic penalty
        self.critical_capacity_threshold_ratio = 0.20 # 20% of current available capacity
        self.critical_capacity_fixed_penalty = 100.0 # Fixed penalty for hitting critical capacity
        self.hotspot_lookahead_window = 6 # Look 6 hours into the future for load

    def _calculate_operation_score(self, server_id, start_time):
        compute_cost = sum(self.inst.server_cost[server_id][t % self.inst.time_horizon]
                           for t in range(start_time, start_time + self.op.duration))

        projected_completion = start_time + self.op.duration + self.remaining_duration
        delay = max(0, projected_completion - self.job.deadline)
        delay_penalty = delay * self.job.delay_weight

        wait_time = start_time - self.earliest_start
        # Base procrastination penalty scaled by job urgency
        procrastination_penalty = (wait_time * self.base_procrastination_penalty *
                                   self.urgency.procrastination_penalty_multiplier)

        # Add penalty for multi-operation impact: if delaying this op severely compresses slack for subsequent ops
        if self.remaining_duration > 0:
            projected_next_op_start = start_time + self.op.duration
            remaining_job_slack = self.job.deadline - (projected_next_op_start + self.remaining_duration)

            # Penalize if remaining slack is tight relative to remaining operations duration
            if remaining_job_slack < self.remaining_duration * 0.5: # e.g., less than 50% of remaining op duration
                # The tighter the slack, the higher the additional penalty, scaled by job urgency
                procrastination_penalty += (self.remaining_duration * 0.5 - remaining_job_slack) * 0.5 * self.urgency.urgency_score


        load_penalty = 0.0
        server = self.inst.servers[server_id]
        if not server.is_cloud:
            for t in range(start_time, start_time + self.op.duration + self.hotspot_lookahead_window):
                used = self.capacity_usage.get(server_id, {}).get(t, 0)
                available_at_t = get_available_capacity(self.inst, server_id, t)

                if available_at_t > 0:
                    projected_used = used + self.op.capacity_required
                    current_utilization = projected_used / available_at_t

                    # 1. Enhanced Non-Linear High Utilization Penalty (cubic)
                    if current_utilization > self.high_utilization_threshold:
                        over_utilization = current_utilization - self.high_utilization_threshold
                        load_penalty += (over_utilization ** 3) * self.high_utilization_penalty_factor

                    # 2. Distinct Critical Capacity Buffer Penalty
                    remaining_capacity_after_this_op = available_at_t - projected_used
                    if remaining_capacity_after_this_op < available_at_t * self.critical_capacity_threshold_ratio:
                        # Apply a fixed, strong penalty to avoid critical capacity drops
                        load_penalty += self.critical_capacity_fixed_penalty

            # Scale total local load penalty by job urgency
            load_penalty *= self.urgency.urgency_score

        return (compute_cost * self.inst.weights['compute_cost'] +
                delay_penalty * self.inst.weights['delay_penalty'] +
                procrastination_penalty + load_penalty)

    def find_best_placement(self):
        best_local = PlacementCandidate(float('inf'), None, -1)
        cloud_benchmark = PlacementCandidate(float('inf'), None, -1)

        compatible_servers = [s for s in self.inst.servers.values() if s.server_type == self.op.server_type]

        for server in compatible_servers:
            if server.is_cloud:
                score = self._calculate_operation_score(server.id, self.earliest_start)
                if score < cloud_benchmark.score:
                    cloud_benchmark = PlacementCandidate(score, server.id, self.earliest_start)
            else: # Is local
                search_limit = self.earliest_start + self.urgency.search_horizon
                for start_t in range(self.earliest_start, search_limit):
                    if has_capacity(self.inst, self.capacity_usage, server.id, start_t, self.op.duration, self.op.capacity_required):
                        score = self._calculate_operation_score(server.id, start_t)
                        if score < best_local.score:
                            best_local = PlacementCandidate(score, server.id, start_t)
                        break # First-fit for this server

        has_local = best_local.server_id is not None
        has_cloud = cloud_benchmark.server_id is not None

        if has_local and has_cloud:
            if best_local.score <= cloud_benchmark.score * self.urgency.cloud_aggressiveness_factor:
                return best_local.server_id, best_local.start_time
            else:
                return cloud_benchmark.server_id, cloud_benchmark.start_time
        elif has_local:
            return best_local.server_id, best_local.start_time
        elif has_cloud:
            return cloud_benchmark.server_id, cloud_benchmark.start_time

        # Failsafe: Unbounded search on cheapest local machine
        local_servers = [s for s in compatible_servers if not s.is_cloud]
        if local_servers:
            cheapest_local_id = min(local_servers, key=lambda s: sum(self.inst.server_cost[s.id].values())).id
            start_t = self.earliest_start
            while True:
                if has_capacity(self.inst, self.capacity_usage, cheapest_local_id, start_t, self.op.duration, self.op.capacity_required):
                    return cheapest_local_id, start_t
                start_t += 1

        if has_cloud: # Ultimate failsafe if local failsafe fails
            return cloud_benchmark.server_id, cloud_benchmark.start_time

        raise RuntimeError(f"Could not schedule op for job {self.job.id}, no compatible servers found.")


def schedule_job_online(inst: Instance,
                        job: Job,
                        schedule: Schedule,
                        capacity_usage: Dict[str, Dict[int, int]],
                        current_time: int):
    schedule[job.id] = []
    current_op_release_time = max(job.arrival_time, current_time)

    total_job_duration = sum(op.duration for op in job.operations)

    # Calculate system-wide local machine load to inform cloud aggressiveness
    total_local_capacity_potential = 0
    total_local_used_actual = 0
    load_lookahead_window_for_system_load = 6 # Look 6 hours into the future for system load aggregation

    # Ensure capacity_usage has entries for all local servers to prevent KeyError on .get() in below loop
    for server_id, server in inst.servers.items():
        if not server.is_cloud:
            # Ensure an entry exists for the server in capacity_usage
            if server_id not in capacity_usage:
                capacity_usage[server_id] = defaultdict(int) # Initialize with defaultdict if not already

            for t in range(current_time, current_time + load_lookahead_window_for_system_load):
                # We need capacity including failures, but usage for *actual* current demand
                # get_available_capacity correctly accounts for failures
                total_local_capacity_potential += get_available_capacity(inst, server_id, t)
                total_local_used_actual += capacity_usage[server_id].get(t, 0)

    current_system_load = total_local_used_actual / total_local_capacity_potential if total_local_capacity_potential > 0 else 0.0

    urgency_profile = JobUrgencyProfile(job, current_time, total_job_duration, current_system_load)

    for idx, op in enumerate(job.operations):
        remaining_duration = sum(o.duration for o in job.operations[idx + 1:])

        decision_maker = PlacementDecisionMaker(
            inst, job, op, capacity_usage, current_op_release_time, remaining_duration, urgency_profile
        )
        chosen_server, chosen_start = decision_maker.find_best_placement()

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