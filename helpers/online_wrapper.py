"""
Online scheduling wrapper - enforces no future knowledge constraint.

This wrapper simulates time progression and only reveals jobs to the scheduler
as they arrive, preventing the scheduler from making decisions based on future arrivals.
"""

from typing import Dict, List, Any, Optional
from dataclasses import replace
import copy


def online_construct_schedule(inst, construct_schedule_fn):
    """
    Wrapper that enforces online scheduling constraint - no future knowledge.
    
    Jobs are only revealed to the scheduler as they arrive over time.
    The scheduler cannot see or plan for jobs that haven't arrived yet.
    
    Args:
        inst: Full problem instance (with all jobs and their arrival times)
        construct_schedule_fn: The scheduling function from initial.py
        
    Returns:
        schedule: Complete schedule built incrementally as jobs arrived
    """
    # Track all jobs by arrival time
    jobs_by_arrival = {}
    for job_id, job in inst.jobs.items():
        arrival = job.arrival_time
        if arrival not in jobs_by_arrival:
            jobs_by_arrival[arrival] = []
        jobs_by_arrival[arrival].append(job_id)
    
    # Sort arrival times
    arrival_times = sorted(jobs_by_arrival.keys())
    
    # Build schedule incrementally
    full_schedule = {}
    revealed_jobs = {}  # Jobs that have arrived and are visible to scheduler
    
    for arrival_time in arrival_times:
        # Reveal newly arrived jobs
        newly_arrived = jobs_by_arrival[arrival_time]
        for job_id in newly_arrived:
            revealed_jobs[job_id] = inst.jobs[job_id]
        
        # Create a filtered instance with only revealed jobs
        # This is what the scheduler can see (no future knowledge)
        filtered_inst = copy.deepcopy(inst)
        filtered_inst.jobs = revealed_jobs.copy()
        
        # Let the scheduler make decisions with current information only
        current_schedule = construct_schedule_fn(filtered_inst)
        
        # Merge new scheduling decisions (only for newly arrived jobs)
        # Keep previous decisions unchanged (realistic - can't change past)
        for job_id in newly_arrived:
            if job_id in current_schedule:
                full_schedule[job_id] = current_schedule[job_id]
    
    return full_schedule


def run_online_scheduling(scenario: str = "normal", failure_scenario: Optional[str] = None):
    """
    Run online scheduler and return results.
    
    This enforces the online constraint: the scheduler only sees jobs
    that have already arrived, preventing offline optimization.
    
    Args:
        scenario: Job scenario to load from CSV
        failure_scenario: Machine failure scenario from CSV or None
        
    Returns:
        (schedule, fitness, score)
    """
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    
    from helpers.scenario_loader import create_instance_from_csv
    from initial import construct_schedule, evaluate
    
    # Load full instance (with all jobs)
    inst = create_instance_from_csv(scenario=scenario, failure_scenario=failure_scenario)
    
    # Run online scheduler (jobs revealed incrementally)
    schedule = online_construct_schedule(inst, construct_schedule)
    
    # Evaluate the complete schedule
    fitness = evaluate(inst, schedule)
    
    return schedule, fitness, float(fitness.score)
