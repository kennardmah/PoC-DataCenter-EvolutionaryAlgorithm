"""
Demo: Random Failures Impact on Earthquake Scenario

Shows how random failures combined with major disasters affect scheduling.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from helpers.scenario_loader import create_instance_from_csv
from initial import run_scheduling

print("=" * 80)
print("RANDOM FAILURES + EARTHQUAKE SCENARIO")
print("=" * 80)

# Test 1: Earthquake WITHOUT random failures
print("\n" + "=" * 80)
print("TEST 1: Earthquake - Major Failures Only")
print("=" * 80)

result_no_random = run_scheduling(
    scenario="earthquake",
    failure_scenario="earthquake_moderate",
    enable_random_failures=False
)

schedule, fitness, score = result_no_random

print(f"\nResults WITHOUT Random Failures:")
print(f"  Total Cost: ¥{score:.2f}")
print(f"  - Compute: ¥{fitness.compute_cost:.2f}")
print(f"  - Delays: ¥{fitness.delay_penalty:.2f}")
print(f"  Valid: {fitness.is_valid()}")
print(f"  Violations: {len(fitness.violations)}")

# Test 2: Earthquake WITH random failures
print("\n" + "=" * 80)
print("TEST 2: Earthquake - Major + Random Failures")
print("=" * 80)

result_with_random = run_scheduling(
    scenario="earthquake",
    failure_scenario="earthquake_moderate",
    enable_random_failures=True,
    random_failure_rate=8.0,  # More frequent for stressed system
    random_seed=42
)

schedule_random, fitness_random, score_random = result_with_random

inst_with_random = create_instance_from_csv(
    scenario="earthquake",
    failure_scenario="earthquake_moderate",
    enable_random_failures=True,
    random_failure_rate=8.0,
    random_seed=42
)

print(f"\nResults WITH Random Failures:")
print(f"  Total Cost: ¥{score_random:.2f}")
print(f"  - Compute: ¥{fitness_random.compute_cost:.2f}")
print(f"  - Delays: ¥{fitness_random.delay_penalty:.2f}")
print(f"  Valid: {fitness_random.is_valid()}")
print(f"  Violations: {len(fitness_random.violations)}")

# Count failures by type
major_failures = [f for f in inst_with_random.machine_failures if not f.is_random]
random_failures = [f for f in inst_with_random.machine_failures if f.is_random]

print(f"\n  Failure Breakdown:")
print(f"    - Major (earthquake): {len(major_failures)}")
print(f"    - Random (operational): {len(random_failures)}")

# Test 3: Multi-scenario with random failures
print("\n" + "=" * 80)
print("TEST 3: Multi-Scenario Evaluation (80% Normal + 20% Earthquake)")
print("=" * 80)

print("\n  WITHOUT Random Failures:")
result_multi_no_random = run_scheduling(
    multi_scenario=True,
    scenarios=["normal", "earthquake"],
    scenario_weights={"normal": 0.8, "earthquake": 0.2},
    failure_scenarios={"earthquake": "earthquake_moderate"},
    enable_random_failures=False
)

normal_score_no_rand = result_multi_no_random["scenarios"]["normal"][2]
earthquake_score_no_rand = result_multi_no_random["scenarios"]["earthquake"][2]
weighted_no_rand = result_multi_no_random["weighted_score"]

print(f"    Normal: ¥{normal_score_no_rand:.2f}")
print(f"    Earthquake: ¥{earthquake_score_no_rand:.2f}")
print(f"    Weighted: ¥{weighted_no_rand:.2f}")

print("\n  WITH Random Failures (MTBF=12h):")
result_multi_with_random = run_scheduling(
    multi_scenario=True,
    scenarios=["normal", "earthquake"],
    scenario_weights={"normal": 0.8, "earthquake": 0.2},
    failure_scenarios={"earthquake": "earthquake_moderate"},
    enable_random_failures=True,
    random_failure_rate=12.0,
    random_seed=42
)

normal_score_with_rand = result_multi_with_random["scenarios"]["normal"][2]
earthquake_score_with_rand = result_multi_with_random["scenarios"]["earthquake"][2]
weighted_with_rand = result_multi_with_random["weighted_score"]

print(f"    Normal: ¥{normal_score_with_rand:.2f}")
print(f"    Earthquake: ¥{earthquake_score_with_rand:.2f}")
print(f"    Weighted: ¥{weighted_with_rand:.2f}")

# Impact analysis
print("\n" + "=" * 80)
print("IMPACT ANALYSIS")
print("=" * 80)

normal_impact = ((normal_score_with_rand - normal_score_no_rand) / normal_score_no_rand * 100) if normal_score_no_rand > 0 else 0
earthquake_impact = ((earthquake_score_with_rand - earthquake_score_no_rand) / earthquake_score_no_rand * 100) if earthquake_score_no_rand > 0 else 0
weighted_impact = ((weighted_with_rand - weighted_no_rand) / weighted_no_rand * 100) if weighted_no_rand > 0 else 0

print(f"\nRandom Failures Cost Impact:")
print(f"  Normal Scenario: +{normal_impact:.2f}%")
print(f"  Earthquake Scenario: +{earthquake_impact:.2f}%")
print(f"  Weighted Score: +{weighted_impact:.2f}%")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print(f"✓ Random failures simulate realistic operational issues")
print(f"✓ Small stochastic events (1-3 unit capacity loss, 1-2 hour duration)")
print(f"✓ Exponential inter-arrival times (mean time between failures)")
print(f"✓ Fast recovery (2-3x recovery rate vs failure rate)")
print(f"✓ Can be combined with major failure scenarios")
print(f"✓ Weighted score impact: {weighted_impact:.2f}% (minimal but realistic)")
