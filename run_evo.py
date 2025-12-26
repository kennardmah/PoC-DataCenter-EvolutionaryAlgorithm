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

# Verify Google API key is loaded
if not os.getenv('GOOGLE_API_KEY'):
    raise ValueError("GOOGLE_API_KEY not found in environment. Please add it to your .env file.")

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

search_task_sys_msg = """You are an expert in combinatorial optimization, specifically job-shop scheduling and cost-aware computing. The goal is to minimize the weighted sum of compute cost and delay penalties for a government data center.

## Problem Description
- 8 government jobs with ordered operations must be scheduled on 10 machines (5 local + 5 cloud):
  
  **Local Machines (limited capacity):**
  - M1: GPU Compute (training/inference bottleneck) - ¥80-140/h
  - M2: CPU Compute (control logic, feature engineering) - ¥20-55/h
  - M3: High-Memory Analytics (graphs, large matrices) - ¥60-110/h
  - M4: Secure/Isolated Compute (gov-only workloads) - ¥50-80/h
  - M5: Data Ingest & Preprocessing (IO-heavy, staging) - ¥15-35/h
  
  **Cloud Machines (infinite capacity, 5x cost):**
  - C1: Cloud GPU - ¥400-700/h
  - C2: Cloud CPU - ¥100-275/h
  - C3: Cloud HighMem - ¥300-550/h
  - C4: Cloud Secure - ¥250-400/h
  - C5: Cloud Ingest - ¥75-175/h

- Each operation requires a specific machine type and has a duration
- Machine costs vary by hour (demand-based pricing)
- Jobs include: census processing, ML fraud detection, emergency response, geospatial analysis, classified intel, public health, climate modeling, social services
- Jobs have deadlines and priority-weighted delay penalties (CRITICAL > HIGH > MEDIUM > LOW)
- Local machines can only process one operation at a time (capacity constraint)
- Cloud machines have infinite capacity but cost 5x more (use as overflow)
- Operations within a job must be sequential (precedence constraints)

## Objective
Minimize: compute_cost + 5 × delay_penalty

## Current Best Approaches
The baseline uses priority-based scheduling with local search. Key improvement directions:

1. **Smarter job ordering**: Consider slack time (deadline - processing time), cost opportunity windows
2. **Cost-aware timing**: Shift operations to off-peak hours (0-5) when possible without violating deadlines
3. **Local vs. Cloud tradeoff**: Use local machines when available, cloud overflow only when necessary or to avoid delays
4. **Server load balancing**: Distribute work across compatible servers to reduce waiting
5. **Look-ahead scheduling**: Consider downstream operations when making decisions
6. **Meta-heuristics**: Try simulated annealing, genetic algorithms, or tabu search
7. **Constraint programming**: Model as CP-SAT problem for optimal solutions
8. **Machine failure resilience**: Handle machine unavailability by routing to cloud or rescheduling

## Constraints (MUST satisfy)machine types (or their cloud equivalents)
- Operations within a job must not overlap (precedence)
- No two operations can use the same LOCAL machine at the same time (capacity)
- Cloud machines have infinite capacity (can run multiple operations simultaneously)
- Start times must be non-negative
- Machine failures must be respected (cannot use failed machines during failure windows)

Be creative and try different algorithmic approaches. Consider the local vs. cloud cost tradeoff strategically
Be creative and try different algorithmic approaches. The problem is NP-hard, so heuristics and meta-heuristics are appropriate."""


evo_config = EvolutionConfig(
    task_sys_msg=search_task_sys_msg,
    patch_types=["diff", "full", "cross"],
    patch_type_probs=[0.6, 0.3, 0.1],
    num_generations=400,
    max_parallel_jobs=5,
    max_patch_resamples=3,
    max_patch_attempts=3,
    job_type="local",
    language="python",
    llm_models=[
        "gemini-2.0-flash-exp",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ],
    llm_kwargs=dict(
        temperatures=[0.0, 0.7, 1.0],
        max_tokens=8192,
    ),
    meta_rec_interval=10,
    meta_llm_models=["gemini-1.5-flash"],
    meta_llm_kwargs=dict(temperatures=[0.0], max_tokens=8192),
    embedding_model="text-embedding-004",
    code_embed_sim_threshold=0.995,
    novelty_llm_models=["gemini-1.5-flash"],
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
