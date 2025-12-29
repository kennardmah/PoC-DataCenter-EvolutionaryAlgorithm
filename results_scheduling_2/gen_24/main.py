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

class JobScheduler:
    def __init__(self, inst: Instance, job: Job, global_capacity_usage: Dict[str, Dict[int, int]], current_time: int):
        self.inst = inst
        self.job = job
        self.global_capacity_usage = global_capacity_usage
        self.current_time = current_time
        self.job_schedule = []
        self.op_start_time_predecessor = max(job.arrival_time, current_time)

        # Configurable parameters for the heuristic
        self.PRIORITY_MULTIPLIERS = {
            'emergency': 8.0,
            'high': 3.0,
            'medium': 1.0,
            'low': 0.5,
        }
        self.CLOUD_DETERRENCE_PENALTY = 50.0 # Cost penalty for choosing cloud
        self.CLOUD_EMERGENCY_DISCOUNT = 0.75 # Discount cloud deterrence for emergency jobs
        self.LOCAL_SEARCH_WINDOW_HOURS = inst.time_horizon * 3 # Search up to 3 cycles for local machines
        self.SIMULATION_LOCAL_SEARCH_WINDOW = inst.time_horizon // 2 # Shorter window for simulation efficiency
        self.SIMULATION_MAX_HORIZON_FACTOR = 5 # How many time_horizons to simulate up to
        self.START_TIME_TIE_BREAKER = 0.0001 # Small cost to prefer earlier start times

    def schedule_job(self):
        for op_idx, op in enumerate(self.job.operations):
            best_overall_job_cost = float('inf')
            chosen_server_id = None
            chosen_start_time = None

            candidate_placements = self._generate_candidate_placements(op_idx, op)

            for cand_server_id, cand_start_time in candidate_placements:
                current_path_cost, job_finish_time = self._evaluate_placement_with_lookahead(
                    op_idx, op, cand_server_id, cand_start_time
                )

                if current_path_cost < best_overall_job_cost:
                    best_overall_job_cost = current_path_cost
                    chosen_server_id = cand_server_id
                    chosen_start_time = cand_start_time

            # Fallback if no valid placement was found (e.g., extremely tight constraints)
            if chosen_server_id is None:
                # This should ideally not be reached if cloud is truly unlimited for compatible types.
                # As a last resort, force to cloud immediately if possible.
                fallback_server = None
                for s_id, server_obj in self.inst.servers.items():
                    if server_obj.is_cloud and server_obj.server_type == op.server_type:
                        fallback_server = s_id
                        break
                if fallback_server:
                    chosen_server_id = fallback_server
                    chosen_start_time = self.op_start_time_predecessor
                else:
                    raise RuntimeError(f"Job {self.job.id} operation {op_idx}: No compatible server found even for fallback!")

            # Reserve capacity in the actual global usage
            for t in range(chosen_start_time, chosen_start_time + op.duration):
                self.global_capacity_usage[chosen_server_id][t] = \
                    self.global_capacity_usage[chosen_server_id].get(t, 0) + op.capacity_required

            self.job_schedule.append({"server": chosen_server_id, "start": chosen_start_time})
            self.op_start_time_predecessor = chosen_start_time + op.duration

        return self.job_schedule

    def _generate_candidate_placements(self, op_idx: int, op: Operation) -> List[tuple]:
        """Generates a list of (server_id, start_time) candidates for the current operation."""
        candidates = []

        # 1. Cloud options (always immediate start after predecessor finishes)
        for server_id, server in self.inst.servers.items():
            if server.is_cloud and server.server_type == op.server_type:
                candidates.append((server_id, self.op_start_time_predecessor))

        # 2. Local options (search within a window)
        for server_id, server in self.inst.servers.items():
            if not server.is_cloud and server.server_type == op.server_type:
                search_end_time = min(self.job.deadline + self.inst.time_horizon,
                                      self.op_start_time_predecessor + self.LOCAL_SEARCH_WINDOW_HOURS)

                # Prioritize a few earliest available slots to explore diverse paths
                found_slots_on_server = []
                current_search_time = self.op_start_time_predecessor
                MAX_LOCAL_SLOT_OPTIONS = 5 # Number of distinct slots to check per local machine

                while len(found_slots_on_server) < MAX_LOCAL_SLOT_OPTIONS and current_search_time <= search_end_time:
                    # Find the next available slot for this duration
                    slot_found_at = -1
                    for t_check in range(current_search_time, search_end_time + 1):
                        if has_capacity(self.inst, self.global_capacity_usage, server_id, t_check, op.duration, op.capacity_required):
                            slot_found_at = t_check
                            break
                    
                    if slot_found_at != -1:
                        found_slots_on_server.append(slot_found_at)
                        current_search_time = slot_found_at + op.duration # Jump past this slot to find the next
                    else:
                        break # No more slots found on this server within the window

                for s_time in found_slots_on_server:
                    candidates.append((server_id, s_time))

        return candidates

    def _evaluate_placement_with_lookahead(self, op_idx: int, op: Operation,
                                           cand_server_id: str, cand_start_time: int) -> tuple[float, int]:
        """
        Evaluates a candidate placement for the current operation by simulating
        the entire job's completion and calculating its total cost.
        Returns (total_job_cost, final_job_completion_time).
        """
        
        # 1. Calculate cost for the current operation
        is_cloud_server = self.inst.servers[cand_server_id].is_cloud
        current_op_compute_cost = simple_operation_cost(self.inst, cand_server_id, cand_start_time, op)

        if is_cloud_server:
            deterrence = self.CLOUD_DETERRENCE_PENALTY
            if self.job.priority == 'emergency':
                deterrence *= (1.0 - self.CLOUD_EMERGENCY_DISCOUNT)
            current_op_compute_cost += deterrence

        # 2. Create a temporary capacity usage for simulation
        sim_capacity_usage = defaultdict(dict)
        for server_id, times in self.global_capacity_usage.items():
            for t, usage in times.items():
                sim_capacity_usage[server_id][t] = usage

        # 3. Place the current operation in the simulation copy
        for t in range(cand_start_time, cand_start_time + op.duration):
            sim_capacity_usage[cand_server_id][t] = sim_capacity_usage[cand_server_id].get(t, 0) + op.capacity_required

        # 4. Simulate remaining operations
        sim_future_compute_cost, sim_job_finish_time_after_current_op = self._simulate_remaining_operations(
            op_idx + 1, cand_start_time + op.duration, sim_capacity_usage
        )

        # 5. Calculate total job cost for this path
        total_compute_cost = current_op_compute_cost + sim_future_compute_cost
        
        delay = max(0, sim_job_finish_time_after_current_op - self.job.deadline)
        priority_factor = self.PRIORITY_MULTIPLIERS.get(self.job.priority, 1.0)
        total_delay_penalty = delay * self.job.delay_weight * priority_factor * self.inst.weights['delay_penalty']

        overall_job_cost = (
            total_compute_cost * self.inst.weights['compute_cost']
            + total_delay_penalty
            + cand_start_time * self.START_TIME_TIE_BREAKER # Tie-breaker: prefer earlier starts slightly
        )

        return overall_job_cost, sim_job_finish_time_after_current_op

    def _simulate_remaining_operations(self, start_op_idx: int,
                                       current_op_predecessor_finish_time: int,
                                       sim_capacity_usage: Dict[str, Dict[int, int]]) -> tuple[float, int]:
        """
        Simulates greedy scheduling of remaining operations to estimate total cost and completion time.
        This uses a simplified objective for speed.
        Returns (simulated_compute_cost_for_remaining_ops, final_simulated_completion_time).
        """
        total_sim_compute_cost = 0.0
        sim_current_start = current_op_predecessor_finish_time
        sim_last_op_finish_time = current_op_predecessor_finish_time

        # Define a horizon for simulation to prevent infinite loops and manage complexity
        simulation_end_time_limit = self.job.deadline + self.inst.time_horizon * self.SIMULATION_MAX_HORIZON_FACTOR
        
        for idx in range(start_op_idx, len(self.job.operations)):
            op = self.job.operations[idx]
            
            best_sim_placement_cost = float('inf')
            best_sim_server_id = None
            best_sim_start_time = None

            # 1. Try Cloud option first (always available immediately)
            cloud_server_id = None
            for s_id, server_obj in self.inst.servers.items():
                if server_obj.is_cloud and server_obj.server_type == op.server_type:
                    cloud_server_id = s_id
                    break
            
            if cloud_server_id:
                cloud_start = sim_current_start
                cloud_op_compute_cost = simple_operation_cost(self.inst, cloud_server_id, cloud_start, op)
                # Add cloud deterrence for simulation too, but perhaps a smaller one or proportional to cost_weights
                sim_deterrence = self.CLOUD_DETERRENCE_PENALTY * 0.5 # Reduced deterrence for simulation
                if self.job.priority == 'emergency':
                    sim_deterrence *= (1.0 - self.CLOUD_EMERGENCY_DISCOUNT)
                cloud_op_compute_cost += sim_deterrence
                
                best_sim_placement_cost = cloud_op_compute_cost
                best_sim_server_id = cloud_server_id
                best_sim_start_time = cloud_start


            # 2. Then try Local servers, searching for the earliest available slot within simulation window
            local_sim_search_end_time = sim_current_start + self.SIMULATION_LOCAL_SEARCH_WINDOW
            local_sim_search_end_time = min(local_sim_search_end_time, simulation_end_time_limit)


            for server_id, server in self.inst.servers.items():
                if server.is_cloud or server.server_type != op.server_type:
                    continue
                
                for start_t in range(sim_current_start, local_sim_search_end_time + 1):
                    if has_capacity(self.inst, sim_capacity_usage, server_id, start_t, op.duration, op.capacity_required):
                        local_op_compute_cost = simple_operation_cost(self.inst, server_id, start_t, op)
                        
                        # In simulation, we mainly care about earliest finish time and a reasonable cost
                        # If this local slot is better, or significantly earlier, take it
                        current_best_finish_time = (best_sim_start_time + op.duration) if best_sim_server_id else float('inf')
                        
                        if (start_t + op.duration) < current_best_finish_time:
                            best_sim_placement_cost = local_op_compute_cost
                            best_sim_server_id = server_id
                            best_sim_start_time = start_t
                            break # Found earliest slot for this local server, move to next server to potentially find an even earlier slot overall
                        elif (start_t + op.duration) == current_best_finish_time and local_op_compute_cost < best_sim_placement_cost:
                            best_sim_placement_cost = local_op_compute_cost
                            best_sim_server_id = server_id
                            best_sim_start_time = start_t
                            break # Found equally early, but cheaper, break and move on.
            
            # If no placement found within simulation limits (shouldn't happen with cloud fallback)
            if best_sim_server_id is None:
                # Force to cloud immediately, if possible, or fail if no compatible cloud server exists.
                # This is a defensive measure for highly constrained simulations.
                for s_id, server_obj in self.inst.servers.items():
                    if server_obj.is_cloud and server_obj.server_type == op.server_type:
                        best_sim_server_id = s_id
                        best_sim_start_time = sim_current_start
                        best_sim_placement_cost = simple_operation_cost(self.inst, best_sim_server_id, best_sim_start_time, op)
                        best_sim_placement_cost += self.CLOUD_DETERRENCE_PENALTY * 0.5
                        break
                if best_sim_server_id is None:
                    # Should be extremely rare if initial instance has compatible cloud servers
                    raise RuntimeError(f"Simulation failed to place operation {idx} for job {self.job.id} even on cloud.")

            # Reserve capacity in the *simulation copy*
            for t in range(best_sim_start_time, best_sim_start_time + op.duration):
                sim_capacity_usage[best_sim_server_id][t] = sim_capacity_usage[best_sim_server_id].get(t, 0) + op.capacity_required

            total_sim_compute_cost += best_sim_placement_cost
            sim_current_start = best_sim_start_time + op.duration
            sim_last_op_finish_time = sim_current_start
        
        return total_sim_compute_cost, sim_last_op_finish_time


def schedule_job_online(inst: Instance,
                        job: Job,
                        schedule: Schedule,
                        capacity_usage: Dict[str, Dict[int, int]],
                        current_time: int):
    """
    Main entry point for scheduling a single job online.
    Delegates to the JobScheduler class for detailed logic.
    """
    scheduler = JobScheduler(inst, job, capacity_usage, current_time)
    schedule[job.id] = scheduler.schedule_job()


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