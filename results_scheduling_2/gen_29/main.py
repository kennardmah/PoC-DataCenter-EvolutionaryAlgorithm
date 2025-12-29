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

# === New Configuration Data Class ===
@dataclass
class SchedulingConfig:
    priority_multipliers: Dict[str, float] = field(default_factory=lambda: {
        'emergency': 10.0, # High impact for emergency
        'high': 4.0,       # Increased from previous iteration
        'medium': 1.0,
        'low': 0.5,
    })
    cloud_deterrence_base: float = 100.0
    cloud_emergency_discount_factor: float = 0.8
    local_search_window_hours: int = 24
    emergency_cloud_trigger_delay_threshold: int = 5 # hours of delay
    emergency_cloud_cost_reduction_factor: float = 0.5 # Cloud deterrence is halved (1-0.5)
    tie_breaker_earlier_start_factor: float = 0.0001


# === New Class to encapsulate job scheduling logic ===
class JobScheduler:
    def __init__(self, inst: Instance, job: Job, schedule: Schedule,
                 capacity_usage: Dict[str, Dict[int, int]], config: SchedulingConfig):
        self.inst = inst
        self.job = job
        self.schedule = schedule
        self.capacity_usage = capacity_usage
        self.config = config
        self.schedule[job.id] = [] # Initialize schedule for this job

    @staticmethod
    def _get_available_capacity(inst: Instance, server_id: str, time: int) -> int:
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

    def _has_capacity(self, server_id: str, start: int, duration: int, req: int) -> bool:
        """Checks if a server has enough capacity for an operation at a given time slot."""
        for t in range(start, start + duration):
            used = self.capacity_usage[server_id].get(t, 0)
            if used + req > self._get_available_capacity(self.inst, server_id, t):
                return False
        return True

    def _calculate_placement_score(
        self,
        operation: Operation,
        proposed_start: int,
        server_id: str,
        is_cloud_server: bool,
        remaining_ops_duration: int,
        cloud_deterrence_override_factor: float = 1.0 # For emergency cloud cost adjustment
    ) -> float:
        """
        Calculates a weighted score for a proposed operation placement,
        considering compute cost, the impact on the job's overall delay penalty,
        and adding dynamic cloud deterrence.
        """
        # Compute cost for THIS operation
        op_compute_cost = sum(self.inst.server_cost[server_id][t % self.inst.time_horizon]
                              for t in range(proposed_start, proposed_start + operation.duration))

        # Add cloud deterrence if the operation is placed on a cloud server.
        if is_cloud_server:
            deterrence = self.config.cloud_deterrence_base
            if self.job.priority == 'emergency':
                # Apply base discount for emergency jobs
                deterrence *= (1.0 - self.config.cloud_emergency_discount_factor)

            # Apply any additional override factor (e.g., from emergency threshold trigger)
            deterrence *= cloud_deterrence_override_factor

            op_compute_cost += deterrence

        # Projected completion time for THE ENTIRE JOB if this operation starts at proposed_start.
        projected_op_end = proposed_start + operation.duration
        projected_job_completion = projected_op_end + remaining_ops_duration

        # Calculate the delay penalty for THE ENTIRE JOB.
        delay = max(0, projected_job_completion - self.job.deadline)

        # Apply priority multiplier to the delay penalty to reflect job urgency.
        priority_factor = self.config.priority_multipliers.get(self.job.priority, 1.0)
        delay_penalty = delay * self.job.delay_weight * priority_factor * self.inst.weights['delay_penalty']

        # Total score for this specific operation placement.
        total_score = (
            op_compute_cost * self.inst.weights['compute_cost']
            + delay_penalty
        )

        # Tie-breaker: prefer earlier starts slightly if costs are otherwise equal.
        total_score += proposed_start * self.config.tie_breaker_earlier_start_factor

        return total_score

    def _find_best_placement_for_operation(
        self,
        op_idx: int,
        operation: Operation,
        op_start_time_for_job: int, # Earliest start time for current operation based on predecessors
        remaining_operations_duration: int
    ) -> Tuple[float, int, str, bool]: # (score, start_time, server_id, is_cloud)
        """Finds the best server and start time for a single operation."""

        best_overall_candidate = (float('inf'), -1, None, None)
        best_local_candidate = (float('inf'), -1, None) # (score, start_time, server_id)

        # --- Evaluate ALL Options (Local and Cloud) ---
        for server_id, server in self.inst.servers.items():
            if server.server_type != operation.server_type:
                continue

            if server.is_cloud:
                proposed_start = op_start_time_for_job
                score = self._calculate_placement_score(
                    operation, proposed_start, server_id, True,
                    remaining_operations_duration
                )
                if score < best_overall_candidate[0]:
                    best_overall_candidate = (score, proposed_start, server_id, True)
            else:
                # Local: search for the BEST available slot within the search window
                search_end_time = op_start_time_for_job + self.config.local_search_window_hours

                for proposed_start in range(op_start_time_for_job, search_end_time + 1):
                    if self._has_capacity(server_id, proposed_start, operation.duration, operation.capacity_required):
                        score = self._calculate_placement_score(
                            operation, proposed_start, server_id, False,
                            remaining_operations_duration
                        )
                        if score < best_overall_candidate[0]:
                            best_overall_candidate = (score, proposed_start, server_id, False)
                        if score < best_local_candidate[0]:
                            best_local_candidate = (score, proposed_start, server_id) # Track best local

        # --- Emergency Cloud Decision Threshold ---
        if self.job.priority == 'emergency' and best_local_candidate[0] != float('inf'):
            local_score, local_start, _ = best_local_candidate

            # Calculate projected delay if the best local option is chosen
            projected_op_end_local = local_start + operation.duration
            projected_job_completion_local = projected_op_end_local + remaining_operations_duration
            delay_if_local = max(0, projected_job_completion_local - self.job.deadline)

            if delay_if_local > self.config.emergency_cloud_trigger_delay_threshold:
                # If local option causes significant delay for an emergency job,
                # re-evaluate cloud options with a further reduced deterrence.
                for server_id_cloud, server_cloud in self.inst.servers.items():
                    if server_cloud.is_cloud and server_cloud.server_type == operation.server_type:
                        proposed_start_cloud = op_start_time_for_job

                        # Calculate cloud score with the emergency-triggered cost reduction
                        score_cloud_discounted = self._calculate_placement_score(
                            operation, proposed_start_cloud, server_id_cloud, True,
                            remaining_operations_duration, _cloud_deterrence_override_factor=self.config.emergency_cloud_cost_reduction_factor
                        )

                        # If this discounted cloud option is better than the current best overall, choose it.
                        if score_cloud_discounted < best_overall_candidate[0]:
                            best_overall_candidate = (score_cloud_discounted, proposed_start_cloud, server_id_cloud, True)
                        break # Assume one cloud server per type is sufficient to check

        return best_overall_candidate

    def plan_operations(self, current_time: int):
        """Plans all operations for the job."""
        op_start_time_for_job = max(self.job.arrival_time, current_time)

        for idx, op in enumerate(self.job.operations):
            remaining_operations_duration = sum(o.duration for o in self.job.operations[idx + 1:])

            _, chosen_start, chosen_server_id, _ = self._find_best_placement_for_operation(
                idx, op, op_start_time_for_job, remaining_operations_duration
            )

            if chosen_server_id is None:
                raise RuntimeError(
                    f"No suitable server found for operation {idx} of job {self.job.id} "
                    f"starting at {op_start_time_for_job}. Check instance configuration "
                    f"or search parameters."
                )

            # Reserve capacity for the chosen slot
            for t in range(chosen_start, chosen_start + op.duration):
                self.capacity_usage[chosen_server_id][t] = self.capacity_usage[chosen_server_id].get(t, 0) + op.capacity_required

            self.schedule[self.job.id].append({"server": chosen_server_id, "start": chosen_start})
            op_start_time_for_job = chosen_start + op.duration


def schedule_job_online(inst: Instance,
                        job: Job,
                        schedule: Schedule,
                        capacity_usage: Dict[str, Dict[int, int]],
                        current_time: int):
    """
    Schedules a single job's operations online.
    This function delegates the actual planning to a JobScheduler instance.
    """
    # Initialize scheduling configuration (can be made dynamic based on scenario if needed)
    config = SchedulingConfig()

    # Create a scheduler for the current job and plan its operations
    job_scheduler = JobScheduler(inst, job, schedule, capacity_usage, config)
    job_scheduler.plan_operations(current_time)

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
