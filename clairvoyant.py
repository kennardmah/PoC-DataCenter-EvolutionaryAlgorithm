'''
This file looks at the compute resources and expected failures
to estimate the optimal solution for a given scheduling scenario.
Knowing the future = optimal scheduling - some error margin due to
uncertainty in machine states.
'''

from typing import Dict, Tuple, Optional
from helpers.scenario_loader import create_instance_from_csv
from initial import Instance, evaluate, Schedule
import math
import pulp


def calculate_theoretical_lower_bound(inst: Instance) -> float:
    """
    Calculate theoretical best-case cost (lower bound).
    Assumes perfect scheduling: all jobs start at arrival, use cheapest resources,
    no capacity conflicts, all finish before deadline.
    
    Args:
        inst: Problem instance
        
    Returns:
        Theoretical minimum cost (compute cost only, no delays)
    """
    total_min_cost = 0.0
    
    for job_id, job in inst.jobs.items():
        arrival = job.arrival_time
        
        for op in job.operations:
            # Find cheapest local machine of this type during operation window
            compatible_servers = [s for s in inst.servers.values() 
                                if s.server_type == op.server_type and not s.is_cloud]
            
            if not compatible_servers:
                continue
                
            # Calculate minimum cost for this operation
            min_op_cost = float('inf')
            
            for server in compatible_servers:
                # Start at earliest possible time (arrival)
                start_time = arrival
                op_cost = 0
                
                for t in range(start_time, start_time + op.duration):
                    if t < inst.time_horizon:
                        op_cost += inst.server_cost[server.id][t % inst.time_horizon]
                
                min_op_cost = min(min_op_cost, op_cost)
            
            total_min_cost += min_op_cost
            arrival += op.duration  # Next operation starts after this one
    
    return round(total_min_cost, 2)


def calculate_greedy_optimal(inst: Instance) -> Tuple[Schedule, float, Dict]:
    """
    Calculate a greedy optimal solution using perfect knowledge of future.
    Uses earliest-deadline-first (EDF) with cost-aware machine selection.
    
    Args:
        inst: Problem instance
        
    Returns:
        Tuple of (schedule, cost, metrics)
    """
    def get_available_capacity(server_id: str, time: int) -> int:
        """Calculate available capacity considering failures."""
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
    
    def find_earliest_slot(server_id: str, earliest_start: int, duration: int, capacity_req: int) -> Tuple[int, float]:
        """Find earliest available time slot for operation on this server (greedy but deadline-aware)."""
        for start in range(earliest_start, inst.time_horizon):
            if start + duration > inst.time_horizon:
                break
            
            # Check capacity availability
            has_capacity = True
            for t in range(start, start + duration):
                used = capacity_usage[server_id].get(t, 0)
                available = get_available_capacity(server_id, t)
                if used + capacity_req > available:
                    has_capacity = False
                    break
            
            if not has_capacity:
                continue
            
            # Calculate cost for this slot
            cost = sum(inst.server_cost[server_id][t % inst.time_horizon] 
                      for t in range(start, start + duration))
            
            return start, cost
        
        return -1, float('inf')
    
    # Track capacity usage
    capacity_usage = {s: {} for s in inst.servers}
    job_available_time = {j: inst.jobs[j].arrival_time for j in inst.jobs}
    schedule = {}
    
    # Sort jobs by priority first (critical jobs get best slots), then deadline
    priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    sorted_jobs = sorted(inst.jobs.keys(),
                        key=lambda j: (priority_order[inst.jobs[j].priority],
                                      inst.jobs[j].deadline, j))
    
    for job_id in sorted_jobs:
        job = inst.jobs[job_id]
        schedule[job_id] = []
        
        for op in job.operations:
            # Get compatible servers (local first)
            compatible = [s for s in inst.servers.values() 
                         if s.server_type == op.server_type]
            compatible.sort(key=lambda s: s.is_cloud)  # Local first
            
            best_server = None
            best_start = -1
            best_cost = float('inf')
            
            earliest_start = job_available_time[job_id]
            
            # Try local machines first (earliest slot, not cheapest)
            for server in compatible:
                if server.is_cloud:
                    continue
                    
                start, cost = find_earliest_slot(server.id, earliest_start, 
                                                op.duration, op.capacity_required)
                
                if start >= 0 and cost < best_cost:
                    best_server = server
                    best_start = start
                    best_cost = cost
                    break  # Take first available local machine
            
            # If no local machine found, try cloud
            if best_server is None:
                for server in compatible:
                    if not server.is_cloud:
                        continue
                    
                    start, cost = find_earliest_slot(server.id, earliest_start, 
                                                    op.duration, op.capacity_required)
                    
                    if start >= 0:
                        best_server = server
                        best_start = start
                        best_cost = cost
                        break
            
            if best_server is None:
                # Fallback to cloud if no local slot found
                cloud_servers = [s for s in compatible if s.is_cloud]
                if cloud_servers:
                    best_server = cloud_servers[0]
                    best_start = earliest_start
            
            # Schedule the operation
            if best_server:
                for t in range(best_start, best_start + op.duration):
                    if t not in capacity_usage[best_server.id]:
                        capacity_usage[best_server.id][t] = 0
                    capacity_usage[best_server.id][t] += op.capacity_required
                
                schedule[job_id].append({"server": best_server.id, "start": best_start})
                job_available_time[job_id] = best_start + op.duration
    
    # Evaluate the schedule
    fitness = evaluate(inst, schedule)
    
    metrics = {
        "compute_cost": fitness.compute_cost,
        "delay_penalty": fitness.delay_penalty,
        "total_cost": fitness.score,
        "violations": len(fitness.violations),
        "valid": fitness.is_valid()
    }
    
    return schedule, fitness.score, metrics


def calculate_true_optimal(inst: Instance, time_limit: int = 300) -> Tuple[Schedule, float, Dict]:
    """
    Calculate the TRUE optimal solution using Mixed Integer Programming.
    Guarantees global optimality but may be slow for large instances.
    
    Args:
        inst: Problem instance
        time_limit: Maximum solve time in seconds (default 300s = 5 minutes)
        
    Returns:
        Tuple of (schedule, cost, metrics)
    """
    print(f"Setting up MIP model with {len(inst.jobs)} jobs, {len(inst.servers)} servers, {inst.time_horizon} time steps...")
    
    # Create the optimization problem
    prob = pulp.LpProblem("Optimal_Job_Scheduling", pulp.LpMinimize)
    
    def get_available_capacity(server_id: str, time: int) -> int:
        """Calculate available capacity considering failures."""
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
    
    # Decision Variables
    # x[job_id][op_idx][server_id][t] = 1 if operation starts at time t on server
    x = {}
    for job_id, job in inst.jobs.items():
        x[job_id] = {}
        for op_idx, op in enumerate(job.operations):
            x[job_id][op_idx] = {}
            compatible_servers = [s for s in inst.servers.values() 
                                if s.server_type == op.server_type]
            
            for server in compatible_servers:
                x[job_id][op_idx][server.id] = {}
                # Can only start if it can finish within time horizon
                max_start = inst.time_horizon - op.duration
                for t in range(max_start + 1):
                    var_name = f"x_{job_id}_op{op_idx}_{server.id}_t{t}"
                    x[job_id][op_idx][server.id][t] = pulp.LpVariable(var_name, cat='Binary')
    
    print("Decision variables created. Adding constraints...")
    
    # Constraint 1: Each operation must be assigned to exactly one server and start time
    for job_id, job in inst.jobs.items():
        for op_idx in range(len(job.operations)):
            prob += (
                pulp.lpSum(x[job_id][op_idx][server_id][t] 
                          for server_id in x[job_id][op_idx].keys() 
                          for t in x[job_id][op_idx][server_id].keys()) == 1,
                f"assign_job{job_id}_op{op_idx}"
            )
    
    # Constraint 2: Operations within a job must be sequential (precedence)
    for job_id, job in inst.jobs.items():
        for op_idx in range(len(job.operations) - 1):
            op_current = job.operations[op_idx]
            # Start time of current operation
            start_current = pulp.lpSum(
                t * x[job_id][op_idx][s][t]
                for s in x[job_id][op_idx].keys()
                for t in x[job_id][op_idx][s].keys()
            )
            # Start time of next operation
            start_next = pulp.lpSum(
                t * x[job_id][op_idx + 1][s][t]
                for s in x[job_id][op_idx + 1].keys()
                for t in x[job_id][op_idx + 1][s].keys()
            )
            # Next must start after current finishes
            prob += (
                start_next >= start_current + op_current.duration,
                f"precedence_job{job_id}_op{op_idx}"
            )
    
    # Constraint 3: First operation cannot start before arrival time
    for job_id, job in inst.jobs.items():
        start_first = pulp.lpSum(
            t * x[job_id][0][s][t]
            for s in x[job_id][0].keys()
            for t in x[job_id][0][s].keys()
        )
        prob += (
            start_first >= job.arrival_time,
            f"arrival_job{job_id}"
        )
    
    # Constraint 4: Capacity constraints for non-cloud servers
    print("Adding capacity constraints...")
    for server_id, server in inst.servers.items():
        if server.is_cloud:
            continue  # Cloud has unlimited capacity
        
        for t in range(inst.time_horizon):
            capacity = get_available_capacity(server_id, t)
            if capacity == 0:
                continue  # No need to add constraint if no capacity
            
            # Sum all operations running on this server at time t
            running_ops = []
            for job_id, job in inst.jobs.items():
                for op_idx, op in enumerate(job.operations):
                    if server_id not in x[job_id][op_idx]:
                        continue  # Server not compatible with this operation
                    
                    # Operation is running at time t if it starts at any time t' where t' <= t < t' + duration
                    for start_time in x[job_id][op_idx][server_id].keys():
                        if start_time <= t < start_time + op.duration:
                            running_ops.append(
                                op.capacity_required * x[job_id][op_idx][server_id][start_time]
                            )
            
            if running_ops:
                prob += (
                    pulp.lpSum(running_ops) <= capacity,
                    f"capacity_{server_id}_t{t}"
                )
    
    print("Setting up objective function...")
    
    # Objective: Minimize total cost (compute cost + delay penalties)
    compute_costs = []
    delay_penalties = []
    
    # Compute costs
    for job_id, job in inst.jobs.items():
        for op_idx, op in enumerate(job.operations):
            for server_id in x[job_id][op_idx].keys():
                for start_time in x[job_id][op_idx][server_id].keys():
                    # Cost of running this operation on this server starting at start_time
                    op_cost = sum(
                        inst.server_cost[server_id][t % inst.time_horizon]
                        for t in range(start_time, start_time + op.duration)
                    )
                    compute_costs.append(op_cost * x[job_id][op_idx][server_id][start_time])
    
    # Delay penalties
    priority_penalties = {"CRITICAL": 1000.0, "HIGH": 100.0, "MEDIUM": 10.0, "LOW": 1.0}
    
    for job_id, job in inst.jobs.items():
        # Completion time = start of last operation + its duration
        last_op_idx = len(job.operations) - 1
        last_op = job.operations[last_op_idx]
        
        completion_time = pulp.lpSum(
            (t + last_op.duration) * x[job_id][last_op_idx][s][t]
            for s in x[job_id][last_op_idx].keys()
            for t in x[job_id][last_op_idx][s].keys()
        )
        
        # Delay = max(0, completion_time - deadline)
        # Since we can't directly model max in LP, we use auxiliary variable
        delay_var = pulp.LpVariable(f"delay_{job_id}", lowBound=0)
        prob += delay_var >= completion_time - job.deadline
        
        penalty = priority_penalties[job.priority]
        delay_penalties.append(penalty * delay_var)
    
    # Total objective
    prob += pulp.lpSum(compute_costs) + pulp.lpSum(delay_penalties)
    
    print(f"Model setup complete. Solving with time limit of {time_limit}s...")
    print(f"Variables: {len(prob.variables())}, Constraints: {len(prob.constraints)}")
    
    # Solve the problem
    solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=1, gapRel=0.01)  # 1% optimality gap
    prob.solve(solver)
    
    # Check solution status
    status = pulp.LpStatus[prob.status]
    print(f"Solver status: {status}")
    
    if status not in ['Optimal', 'Not Solved']:
        print(f"Warning: Solution may not be optimal. Status: {status}")
    
    # Extract schedule from solution
    schedule = {}
    for job_id, job in inst.jobs.items():
        schedule[job_id] = []
        for op_idx in range(len(job.operations)):
            assigned = False
            for server_id in x[job_id][op_idx].keys():
                for t in x[job_id][op_idx][server_id].keys():
                    if pulp.value(x[job_id][op_idx][server_id][t]) > 0.5:  # Binary variable is 1
                        schedule[job_id].append({"server": server_id, "start": t})
                        assigned = True
                        break
                if assigned:
                    break
    
    # Evaluate the schedule
    fitness = evaluate(inst, schedule)
    
    metrics = {
        "compute_cost": fitness.compute_cost,
        "delay_penalty": fitness.delay_penalty,
        "total_cost": fitness.score,
        "violations": len(fitness.violations),
        "valid": fitness.is_valid(),
        "solver_status": status,
        "objective_value": pulp.value(prob.objective),
        "num_variables": len(prob.variables()),
        "num_constraints": len(prob.constraints)
    }
    
    return schedule, fitness.score, metrics


def compute_optimal_benchmark(scenario: str = "normal", 
                              failure_scenario: str = None,
                              use_mip: bool = True,
                              time_limit: int = 300) -> Dict:
    """
    Compute optimal benchmarks for a scenario.
    
    Args:
        scenario: Scenario name ('normal', 'earthquake')
        failure_scenario: Failure scenario name or None
        use_mip: If True, use MIP solver (slow but optimal). If False, use greedy (fast but suboptimal)
        time_limit: Time limit for MIP solver in seconds
        
    Returns:
        Dictionary with benchmark metrics
    """
    inst = create_instance_from_csv(scenario=scenario, failure_scenario=failure_scenario)
    
    # Calculate theoretical lower bound
    lower_bound = calculate_theoretical_lower_bound(inst)
    
    # Calculate optimal solution
    if use_mip:
        print(f"\n{'='*60}")
        print(f"Using MIP solver (may take up to {time_limit}s)...")
        print(f"{'='*60}")
        optimal_schedule, optimal_cost, optimal_metrics = calculate_true_optimal(inst, time_limit)
        method = "mip"
    else:
        print(f"\n{'='*60}")
        print("Using greedy heuristic (fast but suboptimal)...")
        print(f"{'='*60}")
        optimal_schedule, optimal_cost, optimal_metrics = calculate_greedy_optimal(inst)
        method = "greedy"
    
    # Also calculate greedy for comparison if using MIP
    if use_mip:
        print("\nCalculating greedy solution for comparison...")
        greedy_schedule, greedy_cost, greedy_metrics = calculate_greedy_optimal(inst)
        optimality_gap = ((greedy_cost - optimal_cost) / optimal_cost * 100) if optimal_cost > 0 else 0
    else:
        greedy_cost = optimal_cost
        greedy_metrics = optimal_metrics
        optimality_gap = 0
    
    return {
        "scenario": scenario,
        "failure_scenario": failure_scenario,
        "method": method,
        "theoretical_lower_bound": lower_bound,
        "optimal_cost": optimal_cost,
        "optimal_schedule": optimal_schedule,
        "optimal_metrics": optimal_metrics,
        "greedy_cost": greedy_cost,
        "greedy_metrics": greedy_metrics,
        "optimality_gap_percent": optimality_gap,
        "num_jobs": len(inst.jobs),
        "time_horizon": inst.time_horizon
    }


if __name__ == "__main__":
    import sys
    
    # Check if user wants greedy mode
    use_mip = "--greedy" not in sys.argv
    time_limit = 300  # 5 minutes default
    
    if "--time-limit" in sys.argv:
        idx = sys.argv.index("--time-limit")
        if idx + 1 < len(sys.argv):
            time_limit = int(sys.argv[idx + 1])
    
    print("Computing optimal benchmarks...\n")
    print(f"Mode: {'MIP (True Optimal)' if use_mip else 'Greedy Heuristic'}")
    if use_mip:
        print(f"Time limit: {time_limit}s per scenario\n")
    
    # Normal scenario
    print("=" * 60)
    print("SCENARIO: Normal Day")
    print("=" * 60)
    normal_bench = compute_optimal_benchmark("normal", use_mip=use_mip, time_limit=time_limit)
    print(f"\nJobs: {normal_bench['num_jobs']}")
    print(f"Theoretical Lower Bound: ¥{normal_bench['theoretical_lower_bound']:.2f}")
    print(f"\nOptimal Solution ({normal_bench['method'].upper()}):")
    print(f"  Total Cost: ¥{normal_bench['optimal_cost']:.2f}")
    print(f"  - Compute Cost: ¥{normal_bench['optimal_metrics']['compute_cost']:.2f}")
    print(f"  - Delay Penalty: ¥{normal_bench['optimal_metrics']['delay_penalty']:.2f}")
    print(f"  - Violations: {normal_bench['optimal_metrics']['violations']}")
    print(f"  - Valid: {normal_bench['optimal_metrics']['valid']}")
    
    if use_mip and 'solver_status' in normal_bench['optimal_metrics']:
        print(f"  - Solver Status: {normal_bench['optimal_metrics']['solver_status']}")
        print(f"  - Variables: {normal_bench['optimal_metrics']['num_variables']}")
        print(f"  - Constraints: {normal_bench['optimal_metrics']['num_constraints']}")
        print(f"\nGreedy Solution (for comparison):")
        print(f"  Total Cost: ¥{normal_bench['greedy_cost']:.2f}")
        print(f"  Optimality Gap: {normal_bench['optimality_gap_percent']:.2f}%")
    
    # Earthquake scenario
    print("\n" + "=" * 60)
    print("SCENARIO: Earthquake Emergency")
    print("=" * 60)
    quake_bench = compute_optimal_benchmark("earthquake", "earthquake_moderate", 
                                           use_mip=use_mip, time_limit=time_limit)
    print(f"\nJobs: {quake_bench['num_jobs']}")
    print(f"Theoretical Lower Bound: ¥{quake_bench['theoretical_lower_bound']:.2f}")
    print(f"\nOptimal Solution ({quake_bench['method'].upper()}):")
    print(f"  Total Cost: ¥{quake_bench['optimal_cost']:.2f}")
    print(f"  - Compute Cost: ¥{quake_bench['optimal_metrics']['compute_cost']:.2f}")
    print(f"  - Delay Penalty: ¥{quake_bench['optimal_metrics']['delay_penalty']:.2f}")
    print(f"  - Violations: {quake_bench['optimal_metrics']['violations']}")
    print(f"  - Valid: {quake_bench['optimal_metrics']['valid']}")
    
    if use_mip and 'solver_status' in quake_bench['optimal_metrics']:
        print(f"  - Solver Status: {quake_bench['optimal_metrics']['solver_status']}")
        print(f"  - Variables: {quake_bench['optimal_metrics']['num_variables']}")
        print(f"  - Constraints: {quake_bench['optimal_metrics']['num_constraints']}")
        print(f"\nGreedy Solution (for comparison):")
        print(f"  Total Cost: ¥{quake_bench['greedy_cost']:.2f}")
        print(f"  Optimality Gap: {quake_bench['optimality_gap_percent']:.2f}%")
    
    # Weighted score
    print("\n" + "=" * 60)
    print("WEIGHTED MULTI-SCENARIO BENCHMARK")
    print("=" * 60)
    weighted_optimal = 0.8 * normal_bench['optimal_cost'] + 0.2 * quake_bench['optimal_cost']
    print(f"Weighted Optimal: ¥{weighted_optimal:.2f}")
    print(f"  (0.8 × {normal_bench['optimal_cost']:.2f} + 0.2 × {quake_bench['optimal_cost']:.2f})")
    
    if use_mip:
        weighted_greedy = 0.8 * normal_bench['greedy_cost'] + 0.2 * quake_bench['greedy_cost']
        weighted_gap = ((weighted_greedy - weighted_optimal) / weighted_optimal * 100) if weighted_optimal > 0 else 0
        print(f"\nWeighted Greedy: ¥{weighted_greedy:.2f}")
        print(f"Weighted Gap: {weighted_gap:.2f}%")
    
    print("\n" + "=" * 60)
    print("Usage: python clairvoyant_solution.py [--greedy] [--time-limit SECONDS]")
    print("  --greedy: Use fast greedy heuristic instead of MIP solver")
    print("  --time-limit: Set time limit for MIP solver (default: 300s)")
    print("=" * 60)


