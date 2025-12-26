"""
Compare online (no future knowledge) vs clairvoyant (perfect knowledge) scheduling.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'helpers'))

from helpers.online_wrapper import run_online_scheduling
from initial import construct_schedule, evaluate
from helpers.scenario_loader import create_instance_from_csv


def validate_arrival_times(schedule, inst):
    """Verify that no job starts before its arrival time."""
    violations = []
    for job_id, ops in schedule.items():
        if not ops:
            continue
        job = inst.jobs[job_id]
        first_start = ops[0]["start"]
        if first_start < job.arrival_time:
            violations.append(
                f"{job_id}: starts at t={first_start} before arrival at t={job.arrival_time}"
            )
    return violations


def compare_online_vs_clairvoyant(scenario: str = "normal", failure_scenario: str = None):
    """
    Compare online scheduling (no future knowledge) vs clairvoyant (perfect knowledge).
    
    Args:
        scenario: Job scenario name
        failure_scenario: Failure scenario name or None
    """
    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario.upper()}")
    print(f"{'='*60}\n")
    
    # Run online scheduler (only sees jobs as they arrive)
    print("ONLINE SCHEDULER (No Future Knowledge):")
    print("  - Processes jobs in arrival order")
    print("  - Cannot see jobs that haven't arrived yet")
    print("  - Realistic constraint for production systems\n")
    
    online_schedule, online_fitness, online_score = run_online_scheduling(
        scenario=scenario, 
        failure_scenario=failure_scenario
    )
    
    print(f"  Cost: ¥{online_score:.2f}")
    print(f"    - Compute: ¥{online_fitness.compute_cost:.2f}")
    print(f"    - Delay Penalty: ¥{online_fitness.delay_penalty:.2f}")
    print(f"  Valid: {online_fitness.is_valid()}")
    
    # Validate arrival times
    inst = create_instance_from_csv(scenario=scenario, failure_scenario=failure_scenario)
    arrival_violations = validate_arrival_times(online_schedule, inst)
    if arrival_violations:
        print(f"  ⚠️  ARRIVAL TIME VIOLATIONS: {len(arrival_violations)}")
        for v in arrival_violations[:3]:  # Show first 3
            print(f"      {v}")
    else:
        print(f"  ✓ All jobs respect arrival times")
    
    # Run clairvoyant (sees all jobs upfront)
    print("\nCLAIRVOYANT SCHEDULER (Perfect Future Knowledge):")
    print("  - Sees all jobs and arrival times upfront")
    print("  - Can sort/prioritize knowing all future arrivals")
    print("  - Still respects arrival time constraints\n")
    
    clairvoyant_schedule = construct_schedule(inst)  # Current FIFO sees all jobs
    clairvoyant_fitness = evaluate(inst, clairvoyant_schedule)
    clairvoyant_score = float(clairvoyant_fitness.score)
    
    print(f"  Cost: ¥{clairvoyant_score:.2f}")
    print(f"    - Compute: ¥{clairvoyant_fitness.compute_cost:.2f}")
    print(f"    - Delay Penalty: ¥{clairvoyant_fitness.delay_penalty:.2f}")
    print(f"  Valid: {clairvoyant_fitness.is_valid()}")
    
    # Validate arrival times for clairvoyant
    arrival_violations = validate_arrival_times(clairvoyant_schedule, inst)
    if arrival_violations:
        print(f"  ⚠️  ARRIVAL TIME VIOLATIONS: {len(arrival_violations)}")
        for v in arrival_violations[:3]:  # Show first 3
            print(f"      {v}")
    else:
        print(f"  ✓ All jobs respect arrival times")
    
    # Calculate competitive ratio
    gap = online_score - clairvoyant_score
    ratio = online_score / clairvoyant_score if clairvoyant_score > 0 else float('inf')
    
    print(f"\n{'='*60}")
    print(f"ONLINE vs CLAIRVOYANT GAP:")
    print(f"{'='*60}")
    print(f"  Absolute Gap: ¥{gap:.2f} ({gap/clairvoyant_score*100:.1f}%)")
    print(f"  Competitive Ratio: {ratio:.3f}x")
    print(f"    (Online pays {ratio:.3f}x what clairvoyant pays)")
    print(f"{'='*60}\n")
    
    return {
        'online': {'score': online_score, 'fitness': online_fitness},
        'clairvoyant': {'score': clairvoyant_score, 'fitness': clairvoyant_fitness},
        'gap': gap,
        'ratio': ratio
    }


if __name__ == "__main__":
    import sys
    
    # Compare normal scenario
    results_normal = compare_online_vs_clairvoyant("normal")
    
    # Compare earthquake scenario
    results_earthquake = compare_online_vs_clairvoyant("earthquake", "earthquake_moderate")
    
    # Overall summary
    print(f"\n{'='*60}")
    print("MULTI-SCENARIO WEIGHTED COMPARISON")
    print(f"{'='*60}")
    
    weighted_online = 0.8 * results_normal['online']['score'] + 0.2 * results_earthquake['online']['score']
    weighted_clairvoyant = 0.8 * results_normal['clairvoyant']['score'] + 0.2 * results_earthquake['clairvoyant']['score']
    
    print(f"Weighted Online (80% normal + 20% earthquake): ¥{weighted_online:.2f}")
    print(f"Weighted Clairvoyant (80% normal + 20% earthquake): ¥{weighted_clairvoyant:.2f}")
    print(f"Weighted Gap: ¥{weighted_online - weighted_clairvoyant:.2f}")
    print(f"Weighted Ratio: {weighted_online/weighted_clairvoyant:.3f}x")
    print(f"{'='*60}\n")
