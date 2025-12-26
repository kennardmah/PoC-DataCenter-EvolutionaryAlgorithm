"""
Evaluator for cost-aware job-shop scheduling with improved timeout handling
"""

import os
import sys
import json
import argparse
from typing import Tuple, Optional, List, Dict, Any

# Add helpers to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'helpers'))

from shinka.core import run_shinka_eval


def format_schedule_string(schedule: Dict, fitness) -> str:
    """Formats schedule into a multi-line string for display."""
    lines = []
    for job_id in sorted(schedule.keys()):
        ops = schedule[job_id]
        op_str = " → ".join([f"{o['server']}@t={o['start']}" for o in ops])
        completion = fitness.job_completion_times.get(job_id, "?")
        lines.append(f"  {job_id}: {op_str} (done: t={completion})")
    return "\n".join(lines)


def validate_schedule(
    run_output: Tuple[Dict, Any, float],
    atol: float = 1e-6,
) -> Tuple[bool, Optional[str]]:
    """
    Validates schedule results based on the output of 'run_scheduling'.
    
    For multi-scenario runs, run_output is actually a dict with scenario results.
    For single scenario runs, it's the standard tuple format.

    Args:
        run_output: Either Tuple (schedule, fitness, reported_score) from run_scheduling
                    or Dict with scenario results for multi-scenario evaluation
        atol: Absolute tolerance for numerical checks.

    Returns:
        (is_valid: bool, error_message: Optional[str])
    """
    # Check if this is a multi-scenario result (dict) or single scenario (tuple)
    if isinstance(run_output, dict) and "scenarios" in run_output:
        # Multi-scenario validation: all scenarios must be valid
        for scenario_name, result in run_output["scenarios"].items():
            schedule, fitness, reported_score = result
            is_valid, msg = _validate_single_schedule(
                schedule, fitness, reported_score, scenario_name, atol
            )
            if not is_valid:
                return False, f"[{scenario_name}] {msg}"
        return True, "All scenarios valid. All constraints satisfied."
    else:
        # Single scenario validation
        schedule, fitness, reported_score = run_output
        return _validate_single_schedule(
            schedule, fitness, reported_score, "baseline", atol
        )


def _validate_single_schedule(
    schedule: Dict,
    fitness: Any,
    reported_score: float,
    scenario: str,
    atol: float = 1e-6,
) -> Tuple[bool, Optional[str]]:
    """Validates a single scenario schedule."""
    msg = "The schedule is valid. All constraints satisfied."

    # Import instance for validation
    from initial import create_instance
    inst = create_instance(scenario=scenario, failure_scenario=None)

    # Check all jobs are scheduled
    if set(schedule.keys()) != set(inst.jobs.keys()):
        missing = set(inst.jobs.keys()) - set(schedule.keys())
        extra = set(schedule.keys()) - set(inst.jobs.keys())
        msg = f"Job mismatch. Missing: {missing}, Extra: {extra}"
        return False, msg

    # Check operation counts
    for job_id, ops in schedule.items():
        expected = len(inst.jobs[job_id].operations)
        if len(ops) != expected:
            msg = f"{job_id}: expected {expected} ops, got {len(ops)}"
            return False, msg

    # Check server type compatibility
    for job_id, ops in schedule.items():
        job = inst.jobs[job_id]
        for idx, op_sched in enumerate(ops):
            required = job.operations[idx].server_type
            actual = inst.servers[op_sched["server"]].server_type
            if required != actual:
                msg = f"{job_id} op {idx}: requires {required}, assigned {actual}"
                return False, msg

    # Check job precedence constraints
    for job_id, ops in schedule.items():
        job = inst.jobs[job_id]
        prev_end = 0
        for idx, op_sched in enumerate(ops):
            start = op_sched["start"]
            duration = job.operations[idx].duration
            if start < prev_end:
                msg = f"{job_id} op {idx}: starts at {start} before prev ends at {prev_end}"
                return False, msg
            prev_end = start + duration

    # Check server capacity (no overlaps)
    server_intervals = {s: [] for s in inst.servers}
    for job_id, ops in schedule.items():
        job = inst.jobs[job_id]
        for idx, op_sched in enumerate(ops):
            server = op_sched["server"]
            start = op_sched["start"]
            end = start + job.operations[idx].duration
            server_intervals[server].append((start, end, job_id))

    for server, intervals in server_intervals.items():
        intervals.sort()
        for i in range(len(intervals) - 1):
            _, end1, job1 = intervals[i]
            start2, _, job2 = intervals[i + 1]
            if start2 < end1:
                msg = f"{server}: overlap between {job1} (ends {end1}) and {job2} (starts {start2})"
                return False, msg

    # Check non-negative start times
    for job_id, ops in schedule.items():
        for idx, op_sched in enumerate(ops):
            if op_sched["start"] < 0:
                msg = f"{job_id} op {idx}: negative start time {op_sched['start']}"
                return False, msg

    # Check fitness violations match
    if fitness.violations:
        msg = f"Fitness has violations: {fitness.violations}"
        return False, msg

    # Check score consistency
    if abs(fitness.score - reported_score) > atol:
        msg = f"Score mismatch: fitness.score={fitness.score}, reported={reported_score}"
        return False, msg

    return True, msg


def get_scheduling_kwargs(run_index: int) -> Dict[str, Any]:
    """Provides keyword arguments for multi-scenario scheduling runs."""
    return {
        "multi_scenario": True,
        "scenarios": ["normal", "earthquake"],
        "scenario_weights": {"normal": 0.8, "earthquake": 0.2},
        "failure_scenarios": {"earthquake": "earthquake_moderate"}
    }


def aggregate_scheduling_metrics(
    results: List[Tuple[Dict, Any, float]], results_dir: str
) -> Dict[str, Any]:
    """
    Aggregates metrics for scheduling. Handles both single and multi-scenario results.
    Saves extra.json with detailed scheduling information.
    """
    if not results:
        return {"combined_score": float("inf"), "error": "No results to aggregate"}

    result = results[0]
    
    # Check if this is multi-scenario result
    if isinstance(result, dict) and "scenarios" in result:
        # Multi-scenario aggregation
        weighted_score = result["weighted_score"]
        combined_score = -weighted_score  # Negative because lower is better
        
        public_metrics = {
            "multi_scenario": True,
            "weighted_score": float(weighted_score),
            "scenarios": {}
        }
        private_metrics = {
            "scenarios": {}
        }
        
        for scenario_name, (schedule, fitness, score) in result["scenarios"].items():
            public_metrics["scenarios"][scenario_name] = {
                "schedule_str": format_schedule_string(schedule, fitness),
                "num_jobs": len(schedule),
                "is_valid": fitness.is_valid(),
            }
            private_metrics["scenarios"][scenario_name] = {
                "score": float(score),
                "compute_cost": float(fitness.compute_cost),
                "delay_penalty": float(fitness.delay_penalty),
                "num_violations": len(fitness.violations),
            }
        
        extra_data = {
            "multi_scenario": True,
            "weighted_score": weighted_score,
            "scenario_weights": result.get("weights", {}),
            "scenarios": {}
        }
        for scenario_name, (schedule, fitness, score) in result["scenarios"].items():
            extra_data["scenarios"][scenario_name] = {
                "schedule": schedule,
                "score": score,
                "compute_cost": fitness.compute_cost,
                "delay_penalty": fitness.delay_penalty,
                "violations": fitness.violations,
                "job_completion_times": fitness.job_completion_times,
            }
    else:
        # Single scenario aggregation (backward compatibility)
        schedule, fitness, reported_score = result
        combined_score = -reported_score  # Negative because lower is better

        public_metrics = {
            "multi_scenario": False,
            "schedule_str": format_schedule_string(schedule, fitness),
            "num_jobs": len(schedule),
            "is_valid": fitness.is_valid(),
        }
        private_metrics = {
            "score": float(reported_score),
            "compute_cost": float(fitness.compute_cost),
            "delay_penalty": float(fitness.delay_penalty),
            "num_violations": len(fitness.violations),
        }
        extra_data = {
            "schedule": schedule,
            "score": reported_score,
            "compute_cost": fitness.compute_cost,
            "delay_penalty": fitness.delay_penalty,
            "violations": fitness.violations,
            "job_completion_times": fitness.job_completion_times,
        }
    
    metrics = {
        "combined_score": float(combined_score),
        "public": public_metrics,
        "private": private_metrics,
    }

    extra_file = os.path.join(results_dir, "extra.json")
    try:
        with open(extra_file, "w") as f:
            json.dump(extra_data, f, indent=2)
        print(f"Detailed scheduling data saved to {extra_file}")
    except Exception as e:
        print(f"Error saving extra.json: {e}")
        metrics["extra_json_save_error"] = str(e)

    return metrics


def main(program_path: str, results_dir: str):
    """Runs the scheduling evaluation using shinka.eval."""
    print(f"Evaluating program: {program_path}")
    print(f"Saving results to: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)

    num_experiment_runs = 1

    # Define a nested function to pass results_dir to the aggregator
    def _aggregator_with_context(
        r: List[Tuple[Dict, Any, float]],
    ) -> Dict[str, Any]:
        return aggregate_scheduling_metrics(r, results_dir)

    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_scheduling",
        num_runs=num_experiment_runs,
        get_experiment_kwargs=get_scheduling_kwargs,
        validate_fn=validate_schedule,
        aggregate_metrics_fn=_aggregator_with_context,
    )

    if correct:
        print("Evaluation and Validation completed successfully.")
    else:
        print(f"Evaluation or Validation failed: {error_msg}")

    print("Metrics:")
    for key, value in metrics.items():
        if isinstance(value, str) and len(value) > 100:
            print(f"  {key}: <string_too_long_to_display>")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Job-shop scheduling evaluator using shinka.eval"
    )
    parser.add_argument(
        "--program_path",
        type=str,
        default="initial.py",
        help="Path to program to evaluate (must contain 'run_scheduling')",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Dir to save results (metrics.json, correct.json, extra.json)",
    )
    parsed_args = parser.parse_args()
    main(parsed_args.program_path, parsed_args.results_dir)
