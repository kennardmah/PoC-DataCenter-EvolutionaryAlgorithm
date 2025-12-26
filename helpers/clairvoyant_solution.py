'''
This file looks at the compute resources and expected failures
to estimate the optimal solution for a given scheduling scenario.
Knowing the future = optimal scheduling - some error margin due to
uncertainty in machine states.
'''

from typing import Dict, Tuple
from helpers.scenario_loader import create_instance_from_csv
from initial import Instance, evaluate, Schedule
import math


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


def compute_optimal_benchmark(scenario: str = "normal", 
                              failure_scenario: str = None) -> Dict:
    """
    Compute optimal benchmarks for a scenario.
    
    Args:
        scenario: Scenario name ('normal', 'earthquake')
        failure_scenario: Failure scenario name or None
        
    Returns:
        Dictionary with benchmark metrics
    """
    inst = create_instance_from_csv(scenario=scenario, failure_scenario=failure_scenario)
    
    # Calculate theoretical lower bound
    lower_bound = calculate_theoretical_lower_bound(inst)
    
    # Calculate greedy optimal solution
    optimal_schedule, optimal_cost, optimal_metrics = calculate_greedy_optimal(inst)
    
    return {
        "scenario": scenario,
        "failure_scenario": failure_scenario,
        "theoretical_lower_bound": lower_bound,
        "greedy_optimal_cost": optimal_cost,
        "greedy_optimal_schedule": optimal_schedule,
        "greedy_optimal_metrics": optimal_metrics,
        "num_jobs": len(inst.jobs),
        "time_horizon": inst.time_horizon
    }


if __name__ == "__main__":
    print("Computing optimal benchmarks...\n")
    
    # Normal scenario
    print("=" * 60)
    print("SCENARIO: Normal Day")
    print("=" * 60)
    normal_bench = compute_optimal_benchmark("normal")
    print(f"Jobs: {normal_bench['num_jobs']}")
    print(f"Theoretical Lower Bound: ¥{normal_bench['theoretical_lower_bound']:.2f}")
    print(f"Greedy Optimal Cost: ¥{normal_bench['greedy_optimal_cost']:.2f}")
    print(f"  - Compute Cost: ¥{normal_bench['greedy_optimal_metrics']['compute_cost']:.2f}")
    print(f"  - Delay Penalty: ¥{normal_bench['greedy_optimal_metrics']['delay_penalty']:.2f}")
    print(f"  - Violations: {normal_bench['greedy_optimal_metrics']['violations']}")
    print(f"  - Valid: {normal_bench['greedy_optimal_metrics']['valid']}")
    
    # Earthquake scenario
    print("\n" + "=" * 60)
    print("SCENARIO: Earthquake Emergency")
    print("=" * 60)
    quake_bench = compute_optimal_benchmark("earthquake", "earthquake_moderate")
    print(f"Jobs: {quake_bench['num_jobs']}")
    print(f"Theoretical Lower Bound: ¥{quake_bench['theoretical_lower_bound']:.2f}")
    print(f"Greedy Optimal Cost: ¥{quake_bench['greedy_optimal_cost']:.2f}")
    print(f"  - Compute Cost: ¥{quake_bench['greedy_optimal_metrics']['compute_cost']:.2f}")
    print(f"  - Delay Penalty: ¥{quake_bench['greedy_optimal_metrics']['delay_penalty']:.2f}")
    print(f"  - Violations: {quake_bench['greedy_optimal_metrics']['violations']}")
    print(f"  - Valid: {quake_bench['greedy_optimal_metrics']['valid']}")
    
    # Weighted score
    print("\n" + "=" * 60)
    print("WEIGHTED MULTI-SCENARIO BENCHMARK")
    print("=" * 60)
    weighted_optimal = 0.8 * normal_bench['greedy_optimal_cost'] + 0.2 * quake_bench['greedy_optimal_cost']
    print(f"Weighted Optimal: ¥{weighted_optimal:.2f}")
    print(f"  (0.8 × {normal_bench['greedy_optimal_cost']:.2f} + 0.2 × {quake_bench['greedy_optimal_cost']:.2f})")


