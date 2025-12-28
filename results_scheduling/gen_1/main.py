# EVOLVE-BLOCK-START
"""Constructor-based job-shop scheduling for government data center"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import copy
import os
import math # Make sure math is imported for exponential functions


# =============================================================================
# PROBLEM DEFINITION (UNCHANGED)
# =============================================================================

TIME_HORIZON = 24
@dataclass
class Server:
    id: str
    server_type: str
    capacity: int  # Available capacity units
    is_cloud: bool = False  # Cloud machines have infinite capacity but 5x cost


@dataclass
class Operation:
    server_type: str
    duration: int
    capacity_required: int = 1  # Capacity units consumed during execution


@dataclass
class Job:
    id: str
    description: str
    priority: str
    deadline: int
    delay_weight: float
    operations: List[Operation]
    arrival_time: int = 0  # Time when job becomes available for scheduling


@dataclass
class MachineFailure:
    machine_id: str
    failure_start: int
    failure_end: int
    capacity_reduction: int  # Amount of capacity lost during failure
    failure_rate: float = 1.0  # Exponential failure rate (1.0 = instant, <1.0 = gradual)
    recovery_rate: float = 1.0  # Exponential recovery rate (1.0 = instant, <1.0 = gradual)
    reason: str = ""  # Description of failure


@dataclass
class Instance:
    servers: Dict[str, Server]
    jobs: Dict[str, Job]
    server_cost: Dict[str, Dict[int, float]]
    time_horizon: int
    weights: Dict[str, float]
    machine_failures: Optional[List[MachineFailure]] = None


def create_instance(
    scenario: str = "baseline",
    failure_scenario: Optional[str] = None
) -> Instance:
    """
    Create the problem instance from CSV files.
    
    Args:
        scenario: Job scenario to load from CSV ('baseline', 'peak_load', 'emergency', 'light_load')
        failure_scenario: Machine failure scenario ('earthquake_moderate', 'tsunami_warning', etc.)
                         or None for no failures
    
    Returns:
        Instance object with jobs, servers, and optional machine failures from CSV
    """
    from helpers.scenario_loader import create_instance_from_csv
    return create_instance_from_csv(scenario, failure_scenario)


# =============================================================================
# FITNESS EVALUATION (UNCHANGED)
# =============================================================================

Schedule = Dict[str, List[Dict[str, Any]]]  # job_id -> [{server, start}, ...]


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
    Evaluate a schedule with capacity constraints and automatic cloud failover.
    
    When capacity violations occur (e.g., due to failures during job execution),
    affected operations are automatically relocated to cloud backup resources.
    This maintains validity but incurs higher costs (5x multiplier for cloud).
    """
    
    violations = []
    total_compute_cost = 0
    total_delay_penalty = 0
    job_completion_times = {}
    cloud_relocations = []  # Track which operations were moved to cloud
    
    # Track capacity usage per machine per timestep
    # capacity_usage[server_id][time] = total capacity used
    capacity_usage = {s: {} for s in inst.servers}
    
    # Helper function to calculate available capacity with exponential failure/recovery
    def get_available_capacity(server_id: str, time: int) -> int:
        """Calculate available capacity considering exponential failure/recovery."""
        base_capacity = inst.servers[server_id].capacity
        
        if not inst.machine_failures:
            return base_capacity
        
        # Apply all failures affecting this machine at this time
        total_reduction = 0
        for failure in inst.machine_failures:
            if failure.machine_id != server_id:
                continue
            
            # Before failure window
            if time < failure.failure_start:
                continue
            # After failure window
            elif time >= failure.failure_end:
                # Exponential recovery
                time_since_end = time - failure.failure_end
                recovery_factor = 1.0 - math.exp(-failure.recovery_rate * time_since_end)
                reduction = failure.capacity_reduction * (1.0 - recovery_factor)
                total_reduction += reduction
            # During failure window
            else:
                # Exponential failure onset
                time_since_start = time - failure.failure_start
                failure_factor = 1.0 - math.exp(-failure.failure_rate * time_since_start)
                reduction = failure.capacity_reduction * failure_factor
                total_reduction += reduction
        
        available = max(0, base_capacity - int(total_reduction))
        return available
    
    def get_cloud_server(server_type: str) -> Optional[str]:
        """Find a cloud server of the given type."""
        for server_id, server in inst.servers.items():
            if server.is_cloud and server.server_type == server_type:
                return server_id
        return None

    # First pass: validate basic structure and track capacity usage
    for job_id, ops in schedule.items():
        job = inst.jobs[job_id]
        if len(ops) != len(job.operations):
            violations.append(f"{job_id}: wrong number of operations")
            continue

        # Check first operation doesn't start before arrival
        if ops and ops[0]["start"] < job.arrival_time:
            violations.append(f"{job_id}: starts at {ops[0]['start']} before arrival at {job.arrival_time}")

        prev_end = None
        for idx, op in enumerate(ops):
            server_id = op["server"]
            start = op["start"]
            duration = job.operations[idx].duration
            capacity_req = job.operations[idx].capacity_required
            end = start + duration

            if prev_end is not None and start < prev_end:
                violations.append(f"{job_id}: op {idx} starts before previous ends")
            prev_end = end

            required_type = job.operations[idx].server_type
            actual_type = inst.servers[server_id].server_type
            if required_type != actual_type:
                violations.append(f"{job_id}: op {idx} requires {required_type}, got {actual_type}")

            # Track capacity usage for each timestep
            for t in range(start, end):
                if t not in capacity_usage[server_id]:
                    capacity_usage[server_id][t] = 0
                capacity_usage[server_id][t] += capacity_req

        completion = prev_end if prev_end else 0
        job_completion_times[job_id] = completion
        delay = max(0, completion - job.deadline)
        total_delay_penalty += delay * job.delay_weight

    # Second pass: detect capacity violations and relocate to cloud
    capacity_violations = {}  # (server_id, time) -> exceeded amount
    
    for server_id, usage_by_time in capacity_usage.items():
        # Skip cloud servers (they have unlimited capacity)
        if inst.servers[server_id].is_cloud:
            continue
            
        for time, used_capacity in usage_by_time.items():
            available_capacity = get_available_capacity(server_id, time)
            if used_capacity > available_capacity:
                excess = used_capacity - available_capacity
                capacity_violations[(server_id, time)] = excess
    
    # If violations detected, relocate affected operations to cloud
    if capacity_violations:
        # Find operations that overlap with violation times
        for job_id, ops in schedule.items():
            job = inst.jobs[job_id]
            for idx, op in enumerate(ops):
                server_id = op["server"]
                start = op["start"]
                duration = job.operations[idx].duration
                end = start + duration
                
                # Skip if already on cloud
                if inst.servers[server_id].is_cloud:
                    continue
                
                # Check if this operation overlaps with a capacity violation
                needs_relocation = False
                for t in range(start, end):
                    if (server_id, t) in capacity_violations:
                        needs_relocation = True
                        break
                
                if needs_relocation:
                    # Find cloud backup of same type
                    cloud_server = get_cloud_server(inst.servers[server_id].server_type)
                    if cloud_server:
                        # Remove from local capacity tracking
                        for t in range(start, end):
                            if t in capacity_usage[server_id]:
                                capacity_usage[server_id][t] -= job.operations[idx].capacity_required
                        
                        # Add to cloud capacity tracking
                        for t in range(start, end):
                            if t not in capacity_usage[cloud_server]:
                                capacity_usage[cloud_server][t] = 0
                            capacity_usage[cloud_server][t] += job.operations[idx].capacity_required
                        
                        # Update the schedule to reflect cloud relocation
                        op["server"] = cloud_server
                        op["relocated_to_cloud"] = True
                        
                        cloud_relocations.append(
                            f"{job_id} op{idx}: {server_id} → {cloud_server} (failure during execution)"
                        )

    # Calculate costs (now including relocated operations on cloud)
    for job_id, ops in schedule.items():
        job = inst.jobs[job_id]
        for idx, op in enumerate(ops):
            server_id = op["server"]
            start = op["start"]
            duration = job.operations[idx].duration
            
            # Calculate cost
            for t in range(start, start + duration):
                total_compute_cost += inst.server_cost[server_id][t % inst.time_horizon]

    # Final validation: check if capacity violations were resolved
    for server_id, usage_by_time in capacity_usage.items():
        if inst.servers[server_id].is_cloud:
            continue  # Cloud has unlimited capacity
            
        for time, used_capacity in usage_by_time.items():
            available_capacity = get_available_capacity(server_id, time)
            if used_capacity > available_capacity:
                violations.append(
                    f"{server_id}: capacity exceeded at t={time} even after cloud relocation "
                    f"(used: {used_capacity}, available: {available_capacity})"
                )

    score = (inst.weights["compute_cost"] * total_compute_cost +
             inst.weights["delay_penalty"] * total_delay_penalty +
             len(violations) * 10000)

    return Fitness(round(score, 2), round(total_compute_cost, 2),
                   round(total_delay_penalty, 2), violations, job_completion_times,
                   cloud_relocations)


# =============================================================================
# SCHEDULE CONSTRUCTOR (TO BE EVOLVED)
# =============================================================================

class Scheduler:
    def __init__(self, inst: Instance):
        self.inst = inst
        self.capacity_usage: Dict[str, Dict[int, int]] = {s: {} for s in inst.servers}
        self.job_available_time: Dict[str, int] = {j: inst.jobs[j].arrival_time for j in inst.jobs}
        self.schedule: Schedule = {}

        # Precompute server types and cloud servers
        self.server_types_map: Dict[str, List[str]] = {}
        self.cloud_servers_by_type: Dict[str, str] = {}
        for server_id, server in inst.servers.items():
            if server.server_type not in self.server_types_map:
                self.server_types_map[server.server_type] = []
            if server.is_cloud:
                self.cloud_servers_by_type[server.server_type] = server_id
            else:
                self.server_types_map[server.server_type].append(server_id)
        # Add cloud servers to the list of compatible servers, always at the end
        for s_type in self.server_types_map:
            if s_type in self.cloud_servers_by_type:
                self.server_types_map[s_type].append(self.cloud_servers_by_type[s_type])

        # Precompute job total durations
        self.job_total_durations: Dict[str, int] = {
            job_id: sum(op.duration for op in job.operations)
            for job_id, job in inst.jobs.items()
        }
        
        # Priority mapping for CRITICAL, HIGH, MEDIUM, LOW
        self.priority_values = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}

    def _get_current_available_capacity(self, server_id: str, time: int) -> int:
        """
        Calculate available capacity at a specific time, considering machine failures.
        This is a proactive check, assuming the scheduler has knowledge of future failures.
        """
        base_capacity = self.inst.servers[server_id].capacity
        
        if self.inst.machine_failures is None:
            return base_capacity
        
        total_reduction = 0
        for failure in self.inst.machine_failures:
            if failure.machine_id != server_id:
                continue
            
            # Before failure window
            if time < failure.failure_start:
                continue
            # After failure window
            elif time >= failure.failure_end:
                time_since_end = time - failure.failure_end
                recovery_factor = 1.0 - math.exp(-failure.recovery_rate * time_since_end)
                reduction = failure.capacity_reduction * (1.0 - recovery_factor)
                total_reduction += reduction
            # During failure window
            else:
                time_since_start = time - failure.failure_start
                failure_factor = 1.0 - math.exp(-failure.failure_rate * time_since_start)
                reduction = failure.capacity_reduction * failure_factor
                total_reduction += reduction
        
        return max(0, base_capacity - int(total_reduction))

    def _has_capacity_for_op(self, server_id: str, start: int, duration: int, capacity_req: int) -> bool:
        """Check if server has enough capacity for the entire operation duration."""
        server = self.inst.servers[server_id]
        if server.is_cloud:
            return True # Cloud has effectively infinite capacity for scheduling purposes

        for t in range(start, start + duration):
            used = self.capacity_usage[server_id].get(t, 0)
            available = self._get_current_available_capacity(server_id, t)
            if used + capacity_req > available:
                return False
        return True

    def _reserve_capacity(self, server_id: str, start: int, duration: int, capacity_req: int):
        """Reserve capacity for an operation."""
        for t in range(start, start + duration):
            if t not in self.capacity_usage[server_id]:
                self.capacity_usage[server_id][t] = 0
            self.capacity_usage[server_id][t] += capacity_req

    def _calculate_operation_cost(self, server_id: str, start: int, duration: int) -> float:
        """Calculate the compute cost for an operation."""
        cost = 0.0
        for t in range(start, start + duration):
            cost += self.inst.server_cost[server_id][t % self.inst.time_horizon]
        return cost

    def _calculate_job_urgency(self, job_id: str, current_time: int) -> float:
        job = self.inst.jobs[job_id]
        priority_factor = self.priority_values.get(job.priority, 1) * 100 # Higher value for higher priority

        # Remaining duration for the job from its current operation
        remaining_duration = sum(op.duration for op in job.operations[len(self.schedule.get(job_id, [])):])
        
        # Time pressure: how much time is left relative to the job's duration
        time_until_deadline = max(0, job.deadline - current_time - remaining_duration)
        
        # Invert time_until_deadline so smaller is more urgent
        time_pressure_score = 1.0 / (time_until_deadline + 1e-6) # Add small epsilon to avoid division by zero

        # A combined urgency score
        # Higher priority_factor means more urgent
        # Higher time_pressure_score means more urgent
        # Higher delay_weight means more critical to avoid delay
        return (priority_factor * time_pressure_score) + (job.delay_weight * 10)


    def _find_best_slot_for_operation(self, job: Job, op: Operation, earliest_start: int) -> Dict[str, Any]:
        compatible_servers = self.server_types_map.get(op.server_type, [])
        
        best_server = None
        best_start_time = -1
        min_total_score = float('inf')

        cloud_server_id = self.cloud_servers_by_type.get(op.server_type)
        
        # Consider a lookahead window, dynamically adjusted based on job priority
        lookahead_window = self.inst.time_horizon 
        if job.priority == "CRITICAL":
            lookahead_window = 6 # Shorter lookahead for critical jobs to prioritize immediate placement
        elif job.priority == "HIGH":
            lookahead_window = 12 # Medium lookahead
        
        # First, try to find a local slot
        for server_id in compatible_servers:
            server = self.inst.servers[server_id]
            if server.is_cloud:
                continue # Skip cloud in this pass

            for start_time in range(earliest_start, earliest_start + lookahead_window + 1):
                if start_time >= self.inst.time_horizon:
                    break # Don't schedule past the time horizon if possible

                if self._has_capacity_for_op(server_id, start_time, op.duration, op.capacity_required):
                    op_cost = self._calculate_operation_cost(server_id, start_time, op.duration)
                    potential_completion_time = start_time + op.duration
                    
                    delay = max(0, potential_completion_time - job.deadline)
                    delay_penalty = delay * job.delay_weight * self.inst.weights["delay_penalty"]
                    
                    # Heuristic for proactive failure avoidance: penalize scheduling in failing machines
                    failure_penalty = 0
                    if self.inst.machine_failures:
                        for f in self.inst.machine_failures:
                            if f.machine_id == server_id and f.failure_start <= start_time < f.failure_end:
                                # Penalize more if capacity reduction is significant
                                failure_penalty += f.capacity_reduction * 0.1 * (f.failure_rate if f.failure_rate < 1.0 else 1.0)
                    
                    total_score = (op_cost * self.inst.weights["compute_cost"]) + delay_penalty + failure_penalty

                    if total_score < min_total_score:
                        min_total_score = total_score
                        best_server = server_id
                        best_start_time = start_time
        
        # If no local slot found or if a cloud option is significantly better for critical/high priority, use cloud
        cloud_cost_penalty_factor = 5 # Cloud operations are 5x more expensive
        
        # Calculate cloud cost for immediate start
        cloud_op_cost_base = self._calculate_operation_cost(cloud_server_id, earliest_start, op.duration) if cloud_server_id else float('inf')
        cloud_op_cost = cloud_op_cost_base * cloud_cost_penalty_factor

        cloud_completion_time = earliest_start + op.duration
        cloud_delay = max(0, cloud_completion_time - job.deadline)
        cloud_delay_penalty = cloud_delay * job.delay_weight * self.inst.weights["delay_penalty"]
        cloud_total_score = (cloud_op_cost * self.inst.weights["compute_cost"]) + cloud_delay_penalty
        
        # Decision logic:
        # If no local option was found
        # OR if local option leads to a significant delay for a critical job
        # OR if cloud is significantly cheaper (e.g., local has very high hourly rates at earliest_start)
        
        # Thresholds can be tuned
        cloud_preference_threshold = 1.0 # factor by which cloud must be better or local must be worse
        if job.priority == "CRITICAL" or job.priority == "HIGH":
            cloud_preference_threshold = 0.8 # More aggressive cloud usage for high priority
            
        if best_server is None or (cloud_server_id and cloud_total_score < min_total_score * cloud_preference_threshold):
            return {"server": cloud_server_id, "start": earliest_start}
        else:
            return {"server": best_server, "start": best_start_time}

    def schedule_all_jobs(self) -> Schedule:
        # Sort jobs by urgency score (descending)
        # We need `current_time` for urgency calculation, which changes.
        # So we recalculate urgency for remaining unscheduled jobs dynamically.
        
        unscheduled_jobs = list(self.inst.jobs.keys())
        
        while unscheduled_jobs:
            current_scheduling_time = min(self.job_available_time[j_id] for j_id in unscheduled_jobs) if unscheduled_jobs else 0
            
            # Sort remaining jobs based on urgency at the current_scheduling_time
            sorted_jobs_for_pass = sorted(
                unscheduled_jobs,
                key=lambda j_id: self._calculate_job_urgency(j_id, current_scheduling_time),
                reverse=True
            )

            job_id_to_schedule = sorted_jobs_for_pass[0]
            job = self.inst.jobs[job_id_to_schedule]
            
            self.schedule[job_id_to_schedule] = self.schedule.get(job_id_to_schedule, [])

            # Schedule the next operation for this job
            op_idx = len(self.schedule[job_id_to_schedule])
            current_op = job.operations[op_idx]
            
            earliest_start_for_op = max(job.arrival_time, self.job_available_time[job_id_to_schedule])
            
            scheduled_op_info = self._find_best_slot_for_operation(job, current_op, earliest_start_for_op)
            
            server_id = scheduled_op_info["server"]
            start_time = scheduled_op_info["start"]

            self._reserve_capacity(server_id, start_time, current_op.duration, current_op.capacity_required)
            self.schedule[job_id_to_schedule].append({"server": server_id, "start": start_time})
            self.job_available_time[job_id_to_schedule] = start_time + current_op.duration

            if len(self.schedule[job_id_to_schedule]) == len(job.operations):
                unscheduled_jobs.remove(job_id_to_schedule)
        
        return self.schedule


def construct_schedule(inst: Instance) -> Schedule:
    """
    Constructs a schedule using a Priority-Aware Adaptive Scheduler.
    
    This scheduler prioritizes jobs based on urgency (deadline, priority, delay weight),
    and makes intelligent local/cloud allocation decisions considering cost,
    potential delays, and machine failure forecasts.
    """
    scheduler = Scheduler(inst)
    return scheduler.schedule_all_jobs()


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
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
    from helpers.scenario_loader import create_instance_from_csv
    
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
            inst = create_instance_from_csv(scenario=scenario_name, failure_scenario=fail_scenario)
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
        inst = create_instance_from_csv(scenario=scenario, failure_scenario=failure_scenario)
        schedule = construct_schedule(inst)
        fitness = evaluate(inst, schedule)
        return schedule, fitness, float(fitness.score)
