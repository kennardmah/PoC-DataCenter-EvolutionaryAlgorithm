#!/usr/bin/env python3
"""
Test script to run scheduling across multiple scenarios
"""

from scenario_loader import create_instance_from_csv, get_available_scenarios
from initial import construct_schedule, evaluate

def test_scenario(job_scenario: str, failure_scenario: str = None):
    """Test a specific scenario combination."""
    print(f"\n{'='*70}")
    print(f"Testing: {job_scenario}")
    if failure_scenario:
        print(f"With failures: {failure_scenario}")
    print('='*70)
    
    # Load instance
    inst = create_instance_from_csv(job_scenario, failure_scenario)
    
    print(f"Jobs: {len(inst.jobs)}")
    print(f"Machines: {len(inst.servers)}")
    if inst.machine_failures:
        print(f"Machine failures: {len(inst.machine_failures)}")
        for failure in inst.machine_failures:
            print(f"  - {failure.machine_id}: t={failure.failure_start}-{failure.failure_end} ({failure.reason})")
    
    # Run scheduling
    schedule = construct_schedule(inst)
    fitness = evaluate(inst, schedule)
    
    print(f"\nResults:")
    print(f"  Score: {fitness.score}")
    print(f"  Compute Cost: ¥{fitness.compute_cost}")
    print(f"  Delay Penalty: {fitness.delay_penalty}")
    print(f"  Valid: {len(fitness.violations) == 0}")
    
    if len(fitness.violations) > 0:
        print(f"\nViolations ({len(fitness.violations)} total):")
        for v in fitness.violations[:5]:  # Show first 5
            print(f"  - {v}")
        if len(fitness.violations) > 5:
            print(f"  ... and {len(fitness.violations) - 5} more")
    
    # Job completion summary
    late_jobs = []
    for job_id, completion in fitness.job_completion_times.items():
        job = inst.jobs[job_id]
        if completion > job.deadline:
            late_jobs.append((job_id, job.priority, completion - job.deadline))
    
    if late_jobs:
        print(f"\nLate jobs: {len(late_jobs)}")
        for job_id, priority, delay in sorted(late_jobs, key=lambda x: -x[2])[:3]:
            print(f"  - {job_id} ({priority}): {delay}h late")
    else:
        print(f"\n✓ All jobs completed on time!")
    
    return fitness.score


if __name__ == "__main__":
    print("Available scenarios:")
    scenarios = get_available_scenarios()
    print(f"  Job scenarios: {', '.join(scenarios['jobs'])}")
    print(f"  Failure scenarios: {', '.join(scenarios['failures'])}")
    
    # Test all job scenarios without failures
    results = {}
    
    print("\n" + "="*70)
    print("TESTING ALL JOB SCENARIOS (NO FAILURES)")
    print("="*70)
    
    for job_scenario in scenarios['jobs']:
        score = test_scenario(job_scenario)
        results[job_scenario] = score
    
    # Test emergency scenario with different failures
    print("\n" + "="*70)
    print("TESTING EMERGENCY SCENARIO WITH DIFFERENT FAILURES")
    print("="*70)
    
    failure_results = {}
    for failure in ['earthquake_moderate', 'tsunami_warning', 'power_outage']:
        score = test_scenario('emergency', failure)
        failure_results[failure] = score
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nJob Scenarios (no failures):")
    for scenario, score in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {scenario:20s}: {score:8.1f}")
    
    print("\nEmergency + Failures:")
    for scenario, score in sorted(failure_results.items(), key=lambda x: x[1]):
        print(f"  {scenario:20s}: {score:8.1f}")
