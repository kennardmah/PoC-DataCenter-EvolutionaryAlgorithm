"""
Fitness Evaluation for Energy-Aware Job-Shop Scheduling
========================================================
Evaluates a schedule and returns fitness score.
Lower is better (0 = perfect).
"""

from dataclasses import dataclass
from typing import Dict, List, Any
from problem import Instance, Schedule


@dataclass
class Fitness:
    """Evaluation result."""
    score: float
    energy_cost: float
    delay_penalty: float
    violations: List[str]
    job_completion_times: Dict[str, int]
    
    def is_valid(self) -> bool:
        return len(self.violations) == 0


def evaluate(inst: Instance, schedule: Schedule) -> Fitness:
    """
    Evaluate a schedule for the job-shop problem.
    Returns Fitness object (lower score = better).
    """
    server_usage = {s: [] for s in inst.servers}
    job_completion_times = {}
    total_energy_cost = 0.0
    total_delay_penalty = 0.0
    violations = []

    for job_id, ops in schedule.items():
        job = inst.jobs[job_id]
        
        if len(ops) != len(job.operations):
            violations.append(f"{job_id}: wrong number of operations")
            continue

        prev_end = None
        for idx, op in enumerate(ops):
            server_id = op["server"]
            start = op["start"]
            duration = job.operations[idx].duration
            end = start + duration

            # Job order constraint
            if prev_end is not None and start < prev_end:
                violations.append(f"{job_id}: op {idx} starts before previous ends")
            prev_end = end

            # Server type constraint
            required_type = job.operations[idx].server_type
            actual_type = inst.servers[server_id].server_type
            if required_type != actual_type:
                violations.append(f"{job_id}: op {idx} requires {required_type}, got {actual_type}")

            # Record server usage
            server_usage[server_id].append((start, end, job_id, idx))

            # Energy cost
            power = inst.servers[server_id].power_kw
            for t in range(start, end):
                total_energy_cost += power * inst.energy_price[t % inst.time_horizon]

        # Completion time & delay
        completion = prev_end if prev_end else 0
        job_completion_times[job_id] = completion
        delay = max(0, completion - job.deadline)
        total_delay_penalty += delay * job.delay_weight

    # Server capacity constraints (no overlap)
    for server_id, intervals in server_usage.items():
        intervals.sort()
        for i in range(len(intervals) - 1):
            _, end1, job1, _ = intervals[i]
            start2, _, job2, _ = intervals[i + 1]
            if start2 < end1:
                violations.append(f"{server_id}: overlap between {job1} and {job2}")

    # Compute objective
    score = (
        inst.weights["energy_cost"] * total_energy_cost
        + inst.weights["delay_penalty"] * total_delay_penalty
    )
    
    # Add heavy penalty for violations
    score += len(violations) * 10000

    return Fitness(
        score=round(score, 2),
        energy_cost=round(total_energy_cost, 2),
        delay_penalty=round(total_delay_penalty, 2),
        violations=violations,
        job_completion_times=job_completion_times
    )
