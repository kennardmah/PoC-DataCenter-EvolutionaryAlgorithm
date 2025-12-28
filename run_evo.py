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

search_task_sys_msg = """You are an expert in online job-shop scheduling and operations research.

Your task is to design a function `construct_schedule(inst: Instance) -> Schedule`
that minimizes total cost:

Total Cost = compute_cost + 5 × delay_penalty

The schedule is evaluated across two scenarios:
• Normal operations (80% weight)
• Earthquake emergency with reduced local capacity and additional urgent jobs (20% weight)

Key problem characteristics:
• Jobs arrive over time (online scheduling); future jobs and failures are unknown
• Each job has sequential operations with fixed machine types
• Local machines have limited, time-varying capacity and lower cost
• Cloud machines have unlimited capacity but higher cost
• Jobs have deadlines, priorities, and delay penalties
• Emergency jobs are high-priority and arrive unexpectedly

Hard constraints:
• Respect job arrival times and operation order
• Do not exceed machine capacity at any time
• Assign operations only to compatible machine types (local or cloud)
• All start times must be non-negative

Focus on discovering effective online dispatch heuristics that balance:
• Priority and deadline urgency
• Local vs. cloud cost tradeoffs
• Capacity contention and failure resilience

Be creative. This is an NP-hard online scheduling problem where strong heuristics
outperform exact optimization.
"""


evo_config = EvolutionConfig(
    task_sys_msg=search_task_sys_msg,
    patch_types=["diff", "full", "cross"],
    patch_type_probs=[0.6, 0.3, 0.1],
    num_generations=140,
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
