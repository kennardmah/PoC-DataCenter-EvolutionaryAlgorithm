# EVOLVE-BLOCK-START
    # import math 

    def get_servers_by_type(server_type: str, prefer_local: bool = True) -> List[str]:
        """Get servers of a given type, optionally preferring local over cloud."""
        servers = [s.id for s in inst.servers.values() if s.server_type == server_type]
        if prefer_local:
            # Sort: local machines first, then cloud
            servers.sort(key=lambda s: inst.servers[s].is_cloud)
        return servers

    def get_available_capacity(server_id: str, time: int) -> int:
        """
        Calculate available capacity at a specific time.

        NOTE: In true online setting, we wouldn't know about future failures.
        However, for evaluation purposes, we include failure information to
        accurately model the system state.
        """
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

    def has_capacity(server_id: str, start: int, duration: int, capacity_req: int) -> bool:
        """Check if server has enough capacity for the entire operation duration."""
        for t in range(start, start + duration):
            # Ensure not to check beyond the time horizon for local machines
            if t >= inst.time_horizon and not inst.servers[server_id].is_cloud:
                return False
            used = capacity_usage[server_id].get(t, 0)
            available = get_available_capacity(server_id, t)
            if used + capacity_req > available:
                return False
        return True

    # Track capacity usage per machine per timestep
    capacity_usage = {s: {} for s in inst.servers}

    # Track when each job can start (max of arrival time and previous operation finish time)
    job_available_time = {j: inst.jobs[j].arrival_time for j in inst.jobs}

    schedule = {}

    # Pre-calculate total duration for each job and priority scores
    priority_map = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    for job_id, job in inst.jobs.items():
        job.total_duration = sum(op.duration for op in job.operations)
        job.priority_score = priority_map.get(job.priority, 99) # Lower score for higher priority

    # Prioritize jobs based on a multi-criteria heuristic:
    # 1. Job Priority (CRITICAL, HIGH, MEDIUM, LOW) - lower integer value is higher priority
    # 2. Effective deadline (deadline - total_duration - arrival_time) - tighter deadlines first
    # 3. Delay weight (higher delay weight first) - multiplied by -1 to sort descending
    # 4. Arrival time (earlier is better, for secondary tie-breaking)
    # 5. Job ID (for deterministic tie-breaking)
    sorted_jobs = sorted(inst.jobs.keys(),
                         key=lambda j_id: (
                             inst.jobs[j_id].priority_score,  # Lower priority_score first
                             inst.jobs[j_id].deadline - inst.jobs[j_id].arrival_time - inst.jobs[j_id].total_duration, # Smaller slack (tighter deadline) first
                             -inst.jobs[j_id].delay_weight, # Higher delay weight first
                             inst.jobs[j_id].arrival_time,  # Earlier arrival as tie-breaker
                             j_id  # Final tie-breaker
                         ))

    def _get_op_schedule_cost(server_id: str, start_time: int, op_duration: int, capacity_req: int, job_id: str, current_op_index: int) -> float:
        """
        Calculates the estimated cost of scheduling an operation on a given server at a given start time.
        This includes compute cost and an estimate of future delay penalty for the entire job,
        aligning with the problem's total cost objective.
        """
        job = inst.jobs[job_id]

        # Calculate compute cost for this operation
        op_compute_cost = 0.0
        for t in range(start_time, start_time + op_duration):
            # Prevent indexing beyond time_horizon for server_cost if an operation extends past it
            cost_time_idx = t % inst.time_horizon
            op_compute_cost += inst.server_cost[server_id][cost_time_idx]

        # Estimate remaining duration of the job after this operation (including this op duration)
        # This is a key part: we consider the *total* job completion to estimate delay,
        # not just the current operation.
        remaining_op_durations = sum(o.duration for o in job.operations[current_op_index + 1:])

        # Projected job completion time if this operation finishes at start_time + op_duration
        projected_job_completion_time = (start_time + op_duration) + remaining_op_durations

        # Calculate potential delay penalty for the entire job
        projected_delay = max(0, projected_job_completion_time - job.deadline)
        delay_penalty_contribution = projected_delay * job.delay_weight * inst.weights['delay_penalty']

        # Total objective to minimize: weighted sum of compute cost and delay penalty.
        return (op_compute_cost * inst.weights['compute_cost']) + delay_penalty_contribution


    for job_id in sorted_jobs:
        job = inst.jobs[job_id]
        schedule[job_id] = []

        # `current_op_start_time` tracks the earliest an operation of THIS job can start,
        # considering its own arrival_time and preceding operations of the SAME job.
        current_op_start_time = job_available_time[job_id]

        for op_idx, op in enumerate(job.operations):
            best_total_objective_cost = float('inf')
            chosen_server = None
            chosen_start = None

            # 1. Evaluate local machines for the best cost slot
            local_servers = [s.id for s in inst.servers.values() if s.server_type == op.server_type and not s.is_cloud]
            for server_id in local_servers:
                # Iterate through possible start times up to the time horizon,
                # starting from when the job/previous op makes this operation available.
                for start_time in range(current_op_start_time, inst.time_horizon + 1): # Check up to and including time_horizon as end_time can be time_horizon+duration
                    if has_capacity(server_id, start_time, op.duration, op.capacity_required):
                        current_cost_estimate = _get_op_schedule_cost(
                            server_id, start_time, op.duration, op.capacity_required, job_id, op_idx
                        )
                        if current_cost_estimate < best_total_objective_cost:
                            best_total_objective_cost = current_cost_estimate
                            chosen_server = server_id
                            chosen_start = start_time
                        # IMPORTANT: DO NOT break here. We need to find the *best* local slot, not just the earliest.
                        # continue searching this server's timeline for a better (cheaper) slot.

            # 2. Evaluate cloud machines (always available immediately if needed, but higher cost)
            cloud_servers = [s.id for s in inst.servers.values() if s.server_type == op.server_type and s.is_cloud]
            if cloud_servers:
                cloud_server_id = cloud_servers[0] # Assuming one primary cloud server per type
                cloud_start_time = current_op_start_time # Cloud has infinite capacity, can always start here

                # Even if cloud can start immediately, check if it pushes job completion beyond deadline,
                # contributing to delay penalty.
                current_cost_estimate = _get_op_schedule_cost(
                    cloud_server_id, cloud_start_time, op.duration, op.capacity_required, job_id, op_idx
                )
                if current_cost_estimate < best_total_objective_cost:
                    best_total_objective_cost = current_cost_estimate
                    chosen_server = cloud_server_id
                    chosen_start = cloud_start_time

            # 3. Schedule the chosen option
            if chosen_server is not None:
                # Reserve capacity for the chosen server and start time
                for t in range(chosen_start, chosen_start + op.duration):
                    # Ensure not to track capacity beyond the given time horizon for simulation.
                    # This primarily applies to local machines. Cloud is effectively infinite.
                    if t >= inst.time_horizon and not inst.servers[chosen_server].is_cloud:
                        break # Stop tracking if beyond horizon for local machine
                    if t not in capacity_usage[chosen_server]:
                        capacity_usage[chosen_server][t] = 0
                    capacity_usage[chosen_server][t] += op.capacity_required

                schedule[job_id].append({"server": chosen_server, "start": chosen_start})
                current_op_start_time = chosen_start + op.duration
                job_available_time[job_id] = current_op_start_time # Update for next operations of this job
            else:
                # Fallback: This case should ideally not be reached if cloud is available for every type.
                # If it is reached, it implies a fundamental problem with server availability for a type.
                # For robustness, we will try to force schedule on any compatible server (local or cloud)
                # at the earliest possible time, favoring cloud to ensure completion.
                fallback_server = None
                if cloud_servers:
                    fallback_server = cloud_servers[0]
                elif local_servers:
                    fallback_server = local_servers[0]
                else:
                    # Should not happen: no compatible server for operation's type at all.
                    raise ValueError(f"No compatible server (local or cloud) found for job {job_id}, operation {op_idx} (type: {op.server_type})")

                fallback_start_time = current_op_start_time
                # For fallback, if chosen_server remained None, it implies no valid slot was found
                # that satisfied has_capacity and/or the optimization function.
                # This should only realistically happen if there's *no* cloud server of the type,
                # and no local server could fit even by extending beyond time_horizon.
                # Since cloud has infinite capacity, this path should rarely be taken for actual scheduling.
                for t in range(fallback_start_time, fallback_start_time + op.duration):
                    if t not in capacity_usage[fallback_server]:
                        capacity_usage[fallback_server][t] = 0
                    capacity_usage[fallback_server][t] += op.capacity_required
                schedule[job_id].append({"server": fallback_server, "start": fallback_start_time})
                job_available_time[job_id] = fallback_start_time + op.duration

    return schedule


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