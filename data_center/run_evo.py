#!/usr/bin/env python3
"""
ShinkaEvolve runner for Cost-Aware Job-Shop Scheduling
======================================================
Government data center workload scheduling optimization.
"""

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
- 6 jobs with ordered operations must be scheduled on 4 servers (2 CPU, 1 GPU, 1 Storage)
- Each operation requires a specific server type and has a duration
- Server costs vary by hour (demand-based pricing like cloud spot instances):
  - Off-peak (0-5): CPU ¥16, GPU ¥36, Storage ¥10
  - Morning peak (6-9): CPU ¥36, GPU ¥81, Storage ¥22
  - Midday (10-16): CPU ¥24, GPU ¥54, Storage ¥14
  - Evening peak (17-21): CPU ¥44, GPU ¥99, Storage ¥26
  - Late night (22-23): CPU ¥20, GPU ¥45, Storage ¥12
- Jobs have deadlines and priority-weighted delay penalties
- Servers can only process one operation at a time (no overlap)
- Operations within a job must be sequential (precedence constraints)

## Objective
Minimize: compute_cost + 5 × delay_penalty

## Current Best Approaches
The baseline uses priority-based scheduling with local search. Key improvement directions:

1. **Smarter job ordering**: Consider slack time (deadline - processing time), cost opportunity windows
2. **Cost-aware timing**: Shift operations to off-peak hours (0-5) when possible without violating deadlines
3. **Server load balancing**: Distribute work across compatible servers to reduce waiting
4. **Look-ahead scheduling**: Consider downstream operations when making decisions
5. **Meta-heuristics**: Try simulated annealing, genetic algorithms, or tabu search
6. **Constraint programming**: Model as CP-SAT problem for optimal solutions
7. **Dynamic programming**: For small instances, exact methods may be tractable

## Constraints (MUST satisfy)
- All operations must be assigned to compatible server types
- Operations within a job must not overlap (precedence)
- No two operations can use the same server at the same time (capacity)
- Start times must be non-negative

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
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0",
        "o4-mini",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
    ],
    llm_kwargs=dict(
        temperatures=[0.0, 0.5, 1.0],
        reasoning_efforts=["auto", "low", "medium", "high"],
        max_tokens=32768,
    ),
    meta_rec_interval=10,
    meta_llm_models=["gpt-5-nano"],
    meta_llm_kwargs=dict(temperatures=[0.0], max_tokens=16384),
    embedding_model="text-embedding-3-small",
    code_embed_sim_threshold=0.995,
    novelty_llm_models=["gpt-5-nano"],
    novelty_llm_kwargs=dict(temperatures=[0.0], max_tokens=16384),
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
