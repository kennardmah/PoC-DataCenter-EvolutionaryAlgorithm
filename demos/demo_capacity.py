#!/usr/bin/env python3
"""
Demo: Capacity-based scheduling with exponential failure/recovery
Shows how machines degrade and recover gradually during disasters
"""

from helpers.scenario_loader import create_instance_from_csv
from initial import construct_schedule, evaluate
import math


def demo_capacity_model():
    """Demonstrate capacity-based scheduling with gradual failures."""
    
    print("="*70)
    print("CAPACITY-BASED SCHEDULING DEMO")
    print("="*70)
    
    # Load baseline instance
    inst = create_instance_from_csv('baseline', None)
    
    print("\n1. Machine Capacities:")
    print("-" * 70)
    for server_id, server in sorted(inst.servers.items()):
        if not server.is_cloud:
            print(f"  {server_id} ({server.server_type:10s}): {server.capacity:6d} units")
    
    print("\n2. Job Capacity Requirements:")
    print("-" * 70)
    for job_id in ['J1', 'J2', 'J3']:
        job = inst.jobs[job_id]
        print(f"  {job_id}: {job.description[:40]}")
        for idx, op in enumerate(job.operations):
            print(f"      Op{idx+1}: {op.server_type:10s} - {op.duration}h × {op.capacity_required} units")
    
    # Schedule baseline
    schedule = construct_schedule(inst)
    fitness_baseline = evaluate(inst, schedule)
    
    print(f"\n3. Baseline Schedule (No Failures):")
    print("-" * 70)
    print(f"  Compute Cost: ¥{fitness_baseline.compute_cost}")
    print(f"  Delay Penalty: {fitness_baseline.delay_penalty}")
    print(f"  Score: {fitness_baseline.score}")
    print(f"  Valid: {len(fitness_baseline.violations) == 0}")
    
    # Load with earthquake failure
    inst_fail = create_instance_from_csv('baseline', 'earthquake_moderate')
    schedule_fail = construct_schedule(inst_fail)
    fitness_fail = evaluate(inst_fail, schedule_fail)
    
    print(f"\n4. With Earthquake Failures:")
    print("-" * 70)
    for failure in inst_fail.machine_failures:
        print(f"  {failure.machine_id} ({inst_fail.servers[failure.machine_id].server_type}):")
        print(f"    Time: t={failure.failure_start}-{failure.failure_end}")
        print(f"    Capacity reduction: {failure.capacity_reduction} units")
        print(f"    Failure rate: {failure.failure_rate} (how fast it degrades)")
        print(f"    Recovery rate: {failure.recovery_rate} (how fast it recovers)")
    
    print(f"\n  Schedule Results:")
    print(f"    Compute Cost: ¥{fitness_fail.compute_cost}")
    print(f"    Delay Penalty: {fitness_fail.delay_penalty}")
    print(f"    Score: {fitness_fail.score}")
    print(f"    Impact: +¥{fitness_fail.score - fitness_baseline.score} cost increase")
    
    # Show capacity over time for M1 (GPU)
    print(f"\n5. M1 (GPU) Capacity Evolution:")
    print("-" * 70)
    m1_failure = [f for f in inst_fail.machine_failures if f.machine_id == 'M1'][0]
    base_capacity = inst_fail.servers['M1'].capacity
    
    print(f"  Base capacity: {base_capacity} units")
    print(f"  Time  │ Available │ Status")
    print(f"  ──────┼───────────┼─────────────────────────")
    
    for t in range(6, 17):
        if t < m1_failure.failure_start:
            avail = base_capacity
            status = "Normal"
        elif t >= m1_failure.failure_end:
            time_since_end = t - m1_failure.failure_end
            recovery_factor = 1.0 - math.exp(-m1_failure.recovery_rate * time_since_end)
            reduction = m1_failure.capacity_reduction * (1.0 - recovery_factor)
            avail = base_capacity - int(reduction)
            status = f"Recovering ({int(recovery_factor*100)}%)"
        else:
            time_since_start = t - m1_failure.failure_start
            failure_factor = 1.0 - math.exp(-m1_failure.failure_rate * time_since_start)
            reduction = m1_failure.capacity_reduction * failure_factor
            avail = base_capacity - int(reduction)
            status = f"Degrading ({int(failure_factor*100)}%)"
        
        bar = "█" * (avail // 2) + "░" * ((base_capacity - avail) // 2)
        print(f"  t={t:2d}  │   {avail:2d}      │ {bar:12s} {status}")
    
    print(f"\n6. Exponential Model Explanation:")
    print("-" * 70)
    print(f"  • Failure_rate = {m1_failure.failure_rate}: Higher = faster degradation")
    print(f"    - 1.0 = instant failure")
    print(f"    - 0.5 = gradual failure over ~3 hours")
    print(f"    - 0.8 = moderate-speed failure")
    print(f"  ")
    print(f"  • Recovery_rate = {m1_failure.recovery_rate}: Higher = faster recovery")
    print(f"    - 1.0 = instant recovery")
    print(f"    - 0.5 = gradual recovery over ~3 hours")
    print(f"    - 0.6 = moderate-speed recovery")
    print(f"  ")
    print(f"  • Formula: reduction(t) = max_reduction × (1 - e^(-rate × time))")
    
    print(f"\n7. Cloud Overflow:")
    print("-" * 70)
    print(f"  Cloud machines have 'infinite' capacity at 5x cost:")
    for server_id in ['C1', 'C2', 'C3']:
        server = inst.servers[server_id]
        print(f"    {server_id} ({server.server_type:10s}): {server.capacity:7d} units (is_cloud={server.is_cloud})")
    
    print("\n" + "="*70)
    print("✓ Capacity model allows realistic resource constraints!")
    print("✓ Exponential failure/recovery models gradual degradation!")
    print("="*70)


if __name__ == "__main__":
    demo_capacity_model()
