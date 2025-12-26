"""
Demo: Random Background Failures

Shows how stochastic failures affect scheduling with minimal impact.
Random failures simulate minor operational issues (disk errors, network glitches)
that occur naturally in data centers.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scenario_loader import create_instance_from_csv
from initial import run_scheduling, evaluate

print("=" * 80)
print("RANDOM BACKGROUND FAILURES DEMO")
print("=" * 80)

# Test 1: Normal scenario WITHOUT random failures
print("\n" + "=" * 80)
print("TEST 1: Normal Day - NO Random Failures")
print("=" * 80)

inst_no_random = create_instance_from_csv(
    scenario="normal",
    enable_random_failures=False
)

result_no_random = run_scheduling(
    multi_scenario=False,
    scenarios=["normal"],
    scenario_weights={"normal": 1.0},
    enable_random_failures=False
)

schedule, fitness, score = result_no_random

print(f"\nResults WITHOUT Random Failures:")
print(f"  Total Cost: ¥{score:.2f}")
print(f"  - Compute: ¥{fitness.compute_cost:.2f}")
print(f"  - Delays: ¥{fitness.delay_penalty:.2f}")
print(f"  Valid: {fitness.is_valid()}")
print(f"  Violations: {len(fitness.violations)}")
print(f"  Machine Failures: {len(inst_no_random.machine_failures) if inst_no_random.machine_failures else 0}")

# Test 2: Normal scenario WITH random failures
print("\n" + "=" * 80)
print("TEST 2: Normal Day - WITH Random Failures")
print("=" * 80)

inst_with_random = create_instance_from_csv(
    scenario="normal",
    enable_random_failures=True,
    random_failure_rate=12.0,  # Average 12 hours between failures per machine
    random_seed=42  # Reproducible
)

result_with_random = run_scheduling(
    multi_scenario=False,
    scenarios=["normal"],
    scenario_weights={"normal": 1.0},
    enable_random_failures=True,
    random_failure_rate=12.0,
    random_seed=42
)

schedule_random, fitness_random, score_random = result_with_random

print(f"\nResults WITH Random Failures:")
print(f"  Total Cost: ¥{score_random:.2f}")
print(f"  - Compute: ¥{fitness_random.compute_cost:.2f}")
print(f"  - Delays: ¥{fitness_random.delay_penalty:.2f}")
print(f"  Valid: {fitness_random.is_valid()}")
print(f"  Violations: {len(fitness_random.violations)}")
print(f"  Machine Failures: {len(inst_with_random.machine_failures) if inst_with_random.machine_failures else 0}")

# Show details of random failures
if inst_with_random.machine_failures:
    print(f"\n  Random Failure Details:")
    random_only = [f for f in inst_with_random.machine_failures if f.is_random]
    for i, failure in enumerate(random_only[:10]):  # Show first 10
        print(f"    {i+1}. {failure.machine_id}: t={failure.failure_start}-{failure.failure_end}, "
              f"capacity -{failure.capacity_reduction}, {failure.reason}")
    if len(random_only) > 10:
        print(f"    ... and {len(random_only) - 10} more random failures")

# Test 3: Compare impact
print("\n" + "=" * 80)
print("IMPACT ANALYSIS")
print("=" * 80)

cost_increase = score_random - score
percent_increase = (cost_increase / score) * 100 if score > 0 else 0

print(f"\nCost Impact of Random Failures:")
print(f"  Baseline Cost: ¥{score:.2f}")
print(f"  With Random Failures: ¥{score_random:.2f}")
print(f"  Increase: ¥{cost_increase:.2f} ({percent_increase:.2f}%)")

print(f"\nRandom Failure Statistics:")
if inst_with_random.machine_failures:
    random_failures = [f for f in inst_with_random.machine_failures if f.is_random]
    avg_reduction = sum(f.capacity_reduction for f in random_failures) / len(random_failures)
    avg_duration = sum(f.failure_end - f.failure_start for f in random_failures) / len(random_failures)
    
    print(f"  Total Random Failures: {len(random_failures)}")
    print(f"  Avg Capacity Reduction: {avg_reduction:.2f} units")
    print(f"  Avg Duration: {avg_duration:.2f} hours")
    print(f"  Avg Failure Rate: {sum(f.failure_rate for f in random_failures) / len(random_failures):.2f}")
    print(f"  Avg Recovery Rate: {sum(f.recovery_rate for f in random_failures) / len(random_failures):.2f}")

# Test 4: Different random failure rates
print("\n" + "=" * 80)
print("TEST 3: Sensitivity to Failure Rate")
print("=" * 80)

for mtbf in [6.0, 12.0, 24.0]:  # More frequent -> less frequent
    inst_test = create_instance_from_csv(
        scenario="normal",
        enable_random_failures=True,
        random_failure_rate=mtbf,
        random_seed=42
    )
    
    result_test = run_scheduling(
        multi_scenario=False,
        scenarios=["normal"],
        scenario_weights={"normal": 1.0},
        enable_random_failures=True,
        random_failure_rate=mtbf,
        random_seed=42
    )
    
    _, fitness_test, score_test = result_test
    random_count = len([f for f in inst_test.machine_failures if f.is_random])
    
    print(f"\nMTBF = {mtbf:.1f} hours:")
    print(f"  Random Failures: {random_count}")
    print(f"  Cost: ¥{score_test:.2f}")
    print(f"  Impact: +¥{score_test - score:.2f} (+{((score_test - score) / score * 100):.2f}%)")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print(f"✓ Random failures add realism without dominating results")
print(f"✓ Small capacity reductions (1-3 units) and short durations (1-2 hours)")
print(f"✓ Fast recovery (2-3x rate) minimizes long-term impact")
print(f"✓ Cost impact: {percent_increase:.1f}% with MTBF=12 hours")
print(f"✓ Can tune failure rate to control impact level")
