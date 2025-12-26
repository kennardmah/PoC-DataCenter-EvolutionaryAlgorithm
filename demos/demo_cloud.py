#!/usr/bin/env python3
"""
Demo: Cloud overflow with machine failures
Shows how cloud machines provide infinite capacity at 5x cost
"""

import sys
import os
from helpers.scenario_loader import create_instance_from_csv
from initial import construct_schedule, evaluate

def demo_cloud_overflow():
    """Demonstrate cloud overflow during machine failures."""
    
    print("="*70)
    print("DEMO: Cloud Overflow with Machine Failures")
    print("="*70)
    
    # Load emergency scenario with earthquake (M1 and M3 offline t=8-14)
    inst = create_instance_from_csv("emergency", "earthquake_moderate")
    
    print(f"\nScenario: Emergency response with earthquake")
    print(f"Jobs: {len(inst.jobs)}")
    print(f"Machines: {len(inst.servers)}")
    print(f"  - Local machines (M1-M5): capacity=1")
    print(f"  - Cloud machines (C1-C5): capacity=infinite, cost=5x")
    
    if inst.machine_failures:
        print(f"\nMachine Failures:")
        for failure in inst.machine_failures:
            print(f"  - {failure.machine_id}: t={failure.failure_start}-{failure.failure_end}")
            print(f"    Reason: {failure.reason}")
    
    # Run scheduling
    print("\nRunning scheduler...")
    schedule = construct_schedule(inst)
    fitness = evaluate(inst, schedule)
    
    print(f"\nResults:")
    print(f"  Total Score: {fitness.score}")
    print(f"  Compute Cost: ¥{fitness.compute_cost}")
    print(f"  Delay Penalty: {fitness.delay_penalty}")
    print(f"  Valid: {fitness.is_valid()}")
    
    # Analyze local vs cloud usage
    local_ops = 0
    cloud_ops = 0
    local_cost = 0.0
    cloud_cost = 0.0
    
    for job_id, ops in schedule.items():
        job = inst.jobs[job_id]
        for idx, op_sched in enumerate(ops):
            server_id = op_sched["server"]
            start = op_sched["start"]
            duration = job.operations[idx].duration
            
            # Calculate cost for this operation
            op_cost = sum(inst.server_cost[server_id][t] for t in range(start, start + duration))
            
            if inst.servers[server_id].is_cloud:
                cloud_ops += 1
                cloud_cost += op_cost
            else:
                local_ops += 1
                local_cost += op_cost
    
    total_ops = local_ops + cloud_ops
    print(f"\nResource Usage:")
    print(f"  Local machine operations: {local_ops}/{total_ops} ({100*local_ops/total_ops:.1f}%)")
    print(f"  Cloud machine operations: {cloud_ops}/{total_ops} ({100*cloud_ops/total_ops:.1f}%)")
    print(f"\nCost Breakdown:")
    print(f"  Local cost: ¥{local_cost:.0f} ({100*local_cost/fitness.compute_cost:.1f}%)")
    print(f"  Cloud cost: ¥{cloud_cost:.0f} ({100*cloud_cost/fitness.compute_cost:.1f}%)")
    
    if cloud_ops > 0:
        print(f"\n💡 Cloud overflow was used!")
        print(f"   - {cloud_ops} operations ran on cloud machines")
        print(f"   - Cloud cost premium: ¥{cloud_cost - (cloud_cost/5):.0f}")
        print(f"   - Benefit: Avoided delays and capacity constraints")
    else:
        print(f"\n✓ All operations ran on local machines (most cost-effective)")
    
    # Show some specific cloud operations
    if cloud_ops > 0:
        print(f"\nSample cloud operations:")
        count = 0
        for job_id, ops in schedule.items():
            job = inst.jobs[job_id]
            for idx, op_sched in enumerate(ops):
                server_id = op_sched["server"]
                if inst.servers[server_id].is_cloud:
                    start = op_sched["start"]
                    duration = job.operations[idx].duration
                    op_cost = sum(inst.server_cost[server_id][t] for t in range(start, start + duration))
                    print(f"  - {job_id} ({job.priority}): {job.operations[idx].server_type} on {server_id}")
                    print(f"    Time: t={start}-{start+duration}, Cost: ¥{op_cost}")
                    count += 1
                    if count >= 5:
                        break
            if count >= 5:
                break
        if cloud_ops > 5:
            print(f"  ... and {cloud_ops - 5} more")
    
    if not fitness.is_valid():
        print(f"\n⚠️ Schedule has {len(fitness.violations)} violations:")
        for v in fitness.violations[:5]:
            print(f"  - {v}")


if __name__ == "__main__":
    demo_cloud_overflow()
