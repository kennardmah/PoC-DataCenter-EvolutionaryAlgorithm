#!/usr/bin/env python3
"""
Quick comparison: FIFO Baseline vs Clairvoyant Optimal
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'helpers'))

from helpers.scenario_loader import create_instance_from_csv
from helpers.clairvoyant_solution import calculate_true_optimal
from initial import construct_schedule, evaluate

print("=" * 70)
print("BASELINE FIFO vs CLAIRVOYANT OPTIMAL COMPARISON")
print("=" * 70)

# Load scenarios
normal_inst = create_instance_from_csv(scenario="normal")
earthquake_inst = create_instance_from_csv(scenario="earthquake", failure_scenario="earthquake_moderate")

print("\n" + "=" * 70)
print("1️⃣  FIFO BASELINE (initial.py)")
print("=" * 70)

# FIFO Normal
fifo_normal_sched = construct_schedule(normal_inst)
fifo_normal_fit = evaluate(normal_inst, fifo_normal_sched)
fifo_normal_score = float(fifo_normal_fit.score)

# FIFO Earthquake
fifo_eq_sched = construct_schedule(earthquake_inst)
fifo_eq_fit = evaluate(earthquake_inst, fifo_eq_sched)
fifo_eq_score = float(fifo_eq_fit.score)

# FIFO Weighted
fifo_weighted = 0.8 * fifo_normal_score + 0.2 * fifo_eq_score

print(f"\nNormal Scenario:")
print(f"  Cost: ¥{fifo_normal_score:.2f}")
print(f"    - Compute: ¥{fifo_normal_fit.compute_cost:.2f}")
print(f"    - Delays: ¥{fifo_normal_fit.delay_penalty:.2f}")
print(f"  Valid: {fifo_normal_fit.is_valid()}")

print(f"\nEarthquake Scenario:")
print(f"  Cost: ¥{fifo_eq_score:.2f}")
print(f"    - Compute: ¥{fifo_eq_fit.compute_cost:.2f}")
print(f"    - Delays: ¥{fifo_eq_fit.delay_penalty:.2f}")
print(f"  Valid: {fifo_eq_fit.is_valid()}")

print(f"\n📊 Weighted Score: ¥{fifo_weighted:.2f}")
print(f"   (0.8 × {fifo_normal_score:.2f} + 0.2 × {fifo_eq_score:.2f})")

print("\n" + "=" * 70)
print("2️⃣  CLAIRVOYANT OPTIMAL (MIP Solver)")
print("=" * 70)
print("Computing optimal solutions (may take 1-2 minutes)...\n")

# Optimal Normal
opt_normal_sched, opt_normal_score, opt_normal_metrics = calculate_true_optimal(normal_inst, time_limit=300)
opt_normal_fit = evaluate(normal_inst, opt_normal_sched)

# Optimal Earthquake
opt_eq_sched, opt_eq_score, opt_eq_metrics = calculate_true_optimal(earthquake_inst, time_limit=300)
opt_eq_fit = evaluate(earthquake_inst, opt_eq_sched)

# Optimal Weighted
opt_weighted = 0.8 * opt_normal_score + 0.2 * opt_eq_score

print(f"\nNormal Scenario:")
print(f"  Cost: ¥{opt_normal_score:.2f}")
print(f"    - Compute: ¥{opt_normal_fit.compute_cost:.2f}")
print(f"    - Delays: ¥{opt_normal_fit.delay_penalty:.2f}")
print(f"  Valid: {opt_normal_fit.is_valid()}")
print(f"  Status: {opt_normal_metrics['solver_status']}")

print(f"\nEarthquake Scenario:")
print(f"  Cost: ¥{opt_eq_score:.2f}")
print(f"    - Compute: ¥{opt_eq_fit.compute_cost:.2f}")
print(f"    - Delays: ¥{opt_eq_fit.delay_penalty:.2f}")
print(f"  Valid: {opt_eq_fit.is_valid()}")
print(f"  Status: {opt_eq_metrics['solver_status']}")

print(f"\n📊 Weighted Score: ¥{opt_weighted:.2f}")
print(f"   (0.8 × {opt_normal_score:.2f} + 0.2 × {opt_eq_score:.2f})")

print("\n" + "=" * 70)
print("3️⃣  GAP ANALYSIS")
print("=" * 70)

normal_gap = fifo_normal_score - opt_normal_score
normal_gap_pct = (normal_gap / opt_normal_score * 100) if opt_normal_score > 0 else 0

eq_gap = fifo_eq_score - opt_eq_score
eq_gap_pct = (eq_gap / opt_eq_score * 100) if opt_eq_score > 0 else 0

weighted_gap = fifo_weighted - opt_weighted
weighted_gap_pct = (weighted_gap / opt_weighted * 100) if opt_weighted > 0 else 0

print(f"\nNormal Scenario:")
print(f"  FIFO:     ¥{fifo_normal_score:.2f}")
print(f"  Optimal:  ¥{opt_normal_score:.2f}")
print(f"  Gap:      ¥{normal_gap:.2f} ({normal_gap_pct:.2f}% worse)")

print(f"\nEarthquake Scenario:")
print(f"  FIFO:     ¥{fifo_eq_score:.2f}")
print(f"  Optimal:  ¥{opt_eq_score:.2f}")
print(f"  Gap:      ¥{eq_gap:.2f} ({eq_gap_pct:.2f}% worse)")

print(f"\n📈 WEIGHTED (ShinkaEvolve Target):")
print(f"  FIFO Baseline:  ¥{fifo_weighted:.2f}  → Score: {-fifo_weighted:.2f}")
print(f"  Optimal Goal:   ¥{opt_weighted:.2f}  → Score: {-opt_weighted:.2f}")
print(f"  Gap to Close:   ¥{weighted_gap:.2f} ({weighted_gap_pct:.2f}% improvement possible)")

print("\n" + "=" * 70)
print("🎯 EVOLUTION GOAL")
print("=" * 70)
print(f"\nCurrent Baseline Score:  {-fifo_weighted:.2f}")
print(f"Theoretical Best Score:  {-opt_weighted:.2f}")
print(f"Improvement Potential:   {abs(-opt_weighted - (-fifo_weighted)):.2f} points ({weighted_gap_pct:.1f}%)")
print(f"\n✨ ShinkaEvolve should discover solutions approaching {-opt_weighted:.2f}")

print("\n" + "=" * 70)