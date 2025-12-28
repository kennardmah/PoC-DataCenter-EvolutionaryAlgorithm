#!/usr/bin/env python3
"""
ShinkaEvolve runner for Cost-Aware Job-Shop Scheduling
======================================================
Government data center workload scheduling optimization.
"""

import multiprocessing
import sys

# Set multiprocessing start method to 'fork' on macOS to preserve environment
if sys.platform == 'darwin' and __name__ == '__main__':
    try:
        multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        pass  # Already set

from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

from shinka.core import EvolutionRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

job_config = LocalJobConfig(eval_program_path="evaluate.py")

strategy = "weighted"
if strategy == "uniform":
    # 1. Uniform from correct programs
    parent_config = dict(
        parent_selection_strategy="power_law",
        exploitation_alpha=0.0,
        exploitation_ratio=1.0,
    )
elif strategy == "hill_climbing":
    # 2. Hill Climbing (Always from the Best)
    parent_config = dict(
        parent_selection_strategy="power_law",
        exploitation_alpha=100.0,
        exploitation_ratio=1.0,
    )
elif strategy == "weighted":
    # 3. Weighted Prioritization
    parent_config = dict(
        parent_selection_strategy="weighted",
        parent_selection_lambda=10.0,
    )
elif strategy == "power_law":
    # 4. Power-Law Prioritization
    parent_config = dict(
        parent_selection_strategy="power_law",
        exploitation_alpha=1.0,
        exploitation_ratio=0.2,
    )
elif strategy == "power_law_high":
    # 4. Power-Law Prioritization
    parent_config = dict(
        parent_selection_strategy="power_law",
        exploitation_alpha=2.0,
        exploitation_ratio=0.2,
    )
elif strategy == "beam_search":
    # 5. Beam Search
    parent_config = dict(
        parent_selection_strategy="beam_search",
        num_beams=10,
    )


db_config = DatabaseConfig(
    db_path="evolution_db.sqlite",
    num_islands=2,
    archive_size=40,
    # Inspiration parameters
    elite_selection_ratio=0.3,
    num_archive_inspirations=4,
    num_top_k_inspirations=2,
    # Island migration parameters
    migration_interval=10,
    migration_rate=0.1,  # chance to migrate program to random island
    island_elitism=True,  # Island elite is protected from migration
    **parent_config,
)

search_task_sys_msg = """You are an expert in combinatorial optimization and operations research, specializing in multi-objective job-shop scheduling for mission-critical government data center.

## OPTIMIZATION OBJECTIVE

Your goal: Design a `construct_schedule(inst: Instance) -> Schedule` function that minimizes:

**Total Cost = compute_cost + 5 × delay_penalty**

This is evaluated across TWO scenarios with weighted scoring:
- **Normal operations (80% weight)**: 16 jobs over 24 hours
- **Earthquake emergency (20% weight)**: 26 jobs (normal + 10 emergency) with partial machine failures

## PROBLEM STRUCTURE

**Resources:**
- 5 Local Machines (finite capacity, lower cost): M1-M5
  - M1: GPU (cap=20) - ¥80-140/h | M2: CPU (cap=40) - ¥20-55/h
  - M3: HighMem (cap=10) - ¥60-110/h | M4: Secure (cap=5) - ¥50-80/h  
  - M5: Ingest (cap=30) - ¥15-35/h
- 5 Cloud Machines (unlimited capacity, 5× cost): C1-C5
  - Cloud is backup for overflow or failure mitigation

**Jobs & Operations:**
- Each job has 2-5 sequential operations (must execute in order)
- Operations specify: machine_type, duration, capacity_required
- Jobs have: arrival_time, deadline, priority (CRITICAL/HIGH/MEDIUM/LOW), delay_weight
- Government workloads: census, fraud detection, emergency response, geospatial, intel, health, climate, social services

**Machine Failures (earthquake scenario):**
- Gradual capacity reduction with exponential failure/recovery curves
- Example: M1 capacity drops from 20 → 10, M3 from 10 → 5
- Operations can auto-relocate to cloud if local capacity exceeded (costly but maintains validity)

**Hourly Cost Dynamics:**
- Prices vary by hour (demand-based): off-peak (t=0-5) cheaper, peak (t=8-18) more expensive
- Strategic timing can reduce costs without violating deadlines

## HARD CONSTRAINTS (MUST SATISFY)

1. **Precedence**: Operations within a job execute sequentially (op[i] must finish before op[i+1] starts)
2. **Machine Type**: Each operation assigned to correct machine type (or cloud equivalent)
3. **Capacity**: Sum of concurrent operations on local machine ≤ available capacity at each timestep
   - Cloud machines have unlimited capacity (999999)
   - Account for failure-induced capacity reductions
4. **Arrival**: Jobs cannot start before their arrival_time
5. **Non-negative**: All start times ≥ 0

## ALGORITHMIC STRATEGIES TO EXPLORE

**Baseline to Beat:** Online FIFO scheduler (greedy, no lookahead)

**High-Impact Improvements:**
1. **Priority-Aware Ordering**: Sort by urgency metric (deadline - total_duration - arrival_time, weighted by delay_penalty)
2. **Cost-Time Windows**: Schedule low-priority jobs in off-peak hours (t=0-5, t=20-24) to exploit cheaper rates
3. **Smart Local/Cloud Tradeoff**: 
   - Use local when capacity available (minimize cost)
   - Preemptively use cloud for tight-deadline jobs if local queue is long
   - Balance immediate cost vs. future delay penalties
4. **Lookahead Scheduling**: Consider downstream job queue when making placement decisions
5. **Capacity-Aware Placement**: Distribute load across compatible machine types to reduce contention

**Advanced Meta-Heuristics:**
- Simulated Annealing with temperature schedule
- Genetic Algorithms with crossover/mutation on job orderings
- Tabu Search with move diversification
- Constraint Programming (CP-SAT solver from OR-Tools)
- Dynamic Programming for small subproblems
- Local Search with 2-opt, swap, or shift neighborhoods

**Failure Resilience:**
- In earthquake scenario, proactively avoid machines with upcoming failures
- Build slack into critical job schedules
- Use cloud as safety buffer for high-priority operations

## EVALUATION CRITERIA

Your algorithm will be tested on:
1. **Correctness**: Must satisfy all hard constraints (evaluated first)
2. **Multi-Scenario Performance**: Weighted score across normal (80%) + earthquake (20%)
3. **Cost Efficiency**: Minimize compute_cost (smart local/cloud usage, off-peak timing)
4. **Deadline Adherence**: Minimize delay_penalty (especially for CRITICAL/HIGH priority)
5. **Robustness**: Handle machine failures gracefully without excessive cloud costs

## IMPLEMENTATION NOTES

- You're editing `construct_schedule()` in initial.py (between EVOLVE-BLOCK markers)
- Helper functions available: `get_available_capacity(server_id, time)`, `get_servers_by_type(type)`
- Input: Instance object with servers, jobs, server_cost, time_horizon, weights, machine_failures
- Output: Dict mapping job_id → list of {"server": str, "start": int}

**Think creatively.** This is an NP-hard problem - intelligent heuristics outperform brute force. Consider the interplay between cost optimization, deadline pressure, capacity constraints, and failure scenarios."""


evo_config = EvolutionConfig(
    task_sys_msg=search_task_sys_msg,
    patch_types=["diff", "full", "cross"],
    patch_type_probs=[0.6, 0.3, 0.1],
    num_generations=300,
    max_parallel_jobs=5,
    max_patch_resamples=3,
    max_patch_attempts=3,
    job_type="local",
    language="python",
    llm_models=[
        "gemini-2.5-flash"
    ],
    llm_kwargs=dict(
        temperatures=[0.0, 0.7, 1.0],
        max_tokens=8192,
    ),
    meta_rec_interval=10,
    meta_llm_models=["gemini-2.5-flash"],
    meta_llm_kwargs=dict(temperatures=[0.0], max_tokens=8192),
    embedding_model="gemini-embedding-001",
    code_embed_sim_threshold=0.995,
    novelty_llm_models=["gemini-2.5-flash"],
    novelty_llm_kwargs=dict(temperatures=[0.0], max_tokens=8192),
    llm_dynamic_selection="ucb1",
    llm_dynamic_selection_kwargs=dict(exploration_coef=1.0),
    init_program_path="initial.py",
    results_dir="results_scheduling",
)


def main():
    evo_runner = EvolutionRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        verbose=True,
    )
    evo_runner.run()


if __name__ == "__main__":
    results_data = main()
