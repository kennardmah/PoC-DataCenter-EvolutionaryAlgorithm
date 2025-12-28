"""
CSV-based scenario loader for government data center scheduling
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Server:
    id: str
    server_type: str
    capacity: int  # Available capacity units
    is_cloud: bool = False  # Cloud machines have infinite capacity but 5x cost


@dataclass
class Operation:
    server_type: str
    duration: int
    capacity_required: int = 1  # Capacity units consumed during execution


@dataclass
class Job:
    id: str
    description: str
    priority: str
    deadline: int
    delay_weight: float
    operations: List[Operation]
    arrival_time: int = 0  # Time when job becomes available for scheduling


@dataclass
class MachineFailure:
    machine_id: str
    failure_start: int
    failure_end: int
    capacity_reduction: int  # Amount of capacity lost during failure
    failure_rate: float = 1.0  # Exponential failure rate (1.0 = instant, <1.0 = gradual)
    recovery_rate: float = 1.0  # Exponential recovery rate (1.0 = instant, <1.0 = gradual)
    reason: str = ""  # Description of failure
    is_random: bool = False  # True for stochastic background failures


@dataclass
class Instance:
    servers: Dict[str, Server]
    jobs: Dict[str, Job]
    server_cost: Dict[str, Dict[int, float]]
    time_horizon: int
    weights: Dict[str, float]
    machine_failures: Optional[List[MachineFailure]] = None


def load_machines(csv_path: str = "more_data/machines.csv") -> Tuple[Dict[str, Server], Dict[str, Dict[int, float]]]:
    """
    Load machine configurations from CSV.
    
    Returns:
        (servers_dict, server_cost_dict)
    """
    servers = {}
    server_cost = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            machine_id = row['machine_id']
            machine_type = row['machine_type']
            capacity = int(row['capacity'])
            is_cloud = row.get('is_cloud', 'false').lower() == 'true'
            
            servers[machine_id] = Server(machine_id, machine_type, capacity, is_cloud)
            
            # Build hourly cost structure
            cost_offpeak = int(row['cost_offpeak'])
            cost_morning = int(row['cost_morning'])
            cost_midday = int(row['cost_midday'])
            cost_evening = int(row['cost_evening'])
            cost_late = int(row['cost_late'])
            
            server_cost[machine_id] = {
                # Off-peak (0-5)
                0: cost_offpeak, 1: cost_offpeak, 2: cost_offpeak,
                3: cost_offpeak, 4: cost_offpeak, 5: cost_offpeak,
                # Morning peak (6-9)
                6: cost_morning, 7: cost_morning, 8: cost_morning, 9: cost_morning,
                # Midday (10-16)
                10: cost_midday, 11: cost_midday, 12: cost_midday, 13: cost_midday,
                14: cost_midday, 15: cost_midday, 16: cost_midday,
                # Evening peak (17-21)
                17: cost_evening, 18: cost_evening, 19: cost_evening,
                20: cost_evening, 21: cost_evening,
                # Late night (22-23)
                22: cost_late, 23: cost_late
            }
    
    return servers, server_cost


def load_jobs(csv_path: str) -> Dict[str, Job]:
    """
    Load job scenarios from CSV.
    
    Args:
        csv_path: Path to scenario CSV file
    
    Returns:
        Dictionary of Job objects
    """
    jobs = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            job_id = row['job_id']
            description = row['description']
            priority = row['priority']
            deadline = int(row['deadline'])
            delay_weight = float(row['delay_weight'])
            arrival_time = int(row.get('arrival_time', 0))  # Default to 0 if not specified
            
            # Parse operations: "Ingest:2:5,CPU:3:4,GPU:1:8" (Type:Duration:Capacity)
            operations = []
            for op_str in row['operations'].split(','):
                parts = op_str.split(':')
                op_type = parts[0]
                duration = int(parts[1])
                capacity = int(parts[2]) if len(parts) > 2 else 1  # Default capacity = 1
                operations.append(Operation(op_type, duration, capacity))
            
            jobs[job_id] = Job(job_id, description, priority, deadline, delay_weight, operations, arrival_time)
    
    return jobs


def generate_random_failures(
    machines: Dict[str, Server],
    time_horizon: int,
    mean_time_between_failures: float = 12.0,  # Hours between failures per machine
    seed: Optional[int] = 42
) -> List[MachineFailure]:
    """
    Generate random background failures using exponential distribution.
    
    These simulate minor operational issues (disk errors, network glitches, etc.)
    that don't significantly impact overall performance but add realism.
    
    Args:
        machines: Dictionary of available machines
        time_horizon: Planning horizon in hours
        mean_time_between_failures: Average hours between failures for each machine
        seed: Random seed for reproducibility (None for true randomness)
    
    Returns:
        List of random MachineFailure objects
    """
    import random
    import numpy as np
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    random_failures = []
    
    # Only apply to local (non-cloud) machines
    local_machines = [m for m in machines.values() if not m.is_cloud]
    
    for machine in local_machines:
        current_time = 0
        
        # Generate failures using exponential inter-arrival times
        while current_time < time_horizon:
            # Sample next failure time
            inter_arrival_time = np.random.exponential(mean_time_between_failures)
            current_time += inter_arrival_time
            
            if current_time >= time_horizon:
                break
            
            # Random small capacity reduction (1-3 units, max 15% of capacity)
            max_reduction = max(1, int(machine.capacity * 0.15))
            capacity_reduction = random.randint(1, min(3, max_reduction))
            
            # Short duration (1-2 hours)
            duration = random.randint(1, 2)
            failure_start = int(current_time)
            failure_end = min(failure_start + duration, time_horizon)
            
            # Fast failure onset and recovery
            failure_rate = random.uniform(1.5, 2.5)  # Quick degradation
            recovery_rate = random.uniform(2.0, 3.0)  # Quick recovery
            
            random_failures.append(MachineFailure(
                machine_id=machine.id,
                failure_start=failure_start,
                failure_end=failure_end,
                capacity_reduction=capacity_reduction,
                failure_rate=failure_rate,
                recovery_rate=recovery_rate,
                reason="Random operational issue",
                is_random=True
            ))
    
    return random_failures


def load_machine_failures(csv_path: str = "more_data/machine_failures.csv", 
                          scenario: Optional[str] = None) -> List[MachineFailure]:
    """
    Load machine failure scenarios from CSV.
    
    Args:
        csv_path: Path to machine failures CSV
        scenario: Specific failure scenario to load (e.g., 'earthquake_moderate')
                 If None, no failures are loaded
    
    Returns:
        List of MachineFailure objects with capacity reduction and gradual failure/recovery rates
    """
    if scenario is None:
        return []
    
    failures = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['failure_scenario'] == scenario:
                failures.append(MachineFailure(
                    machine_id=row['machine_id'],
                    failure_start=int(row['failure_start']),
                    failure_end=int(row['failure_end']),
                    capacity_reduction=int(row['capacity_reduction']),
                    failure_rate=float(row.get('failure_rate', 1.0)),
                    recovery_rate=float(row.get('recovery_rate', 1.0)),
                    reason=row.get('reason', ''),
                    is_random=False
                ))
    
    return failures


def create_instance_from_csv(
    scenario: Optional[str] = None,  # New parameter name for consistency
    job_scenario: Optional[str] = None,  # Deprecated, kept for backward compatibility
    failure_scenario: Optional[str] = None,
    time_horizon: int = 24,
    compute_cost_weight: float = 1.0,
    delay_penalty_weight: float = 5.0,
    enable_random_failures: bool = False,  # Enable stochastic background failures
    random_failure_rate: float = 12.0,  # Mean hours between failures per machine
    random_seed: Optional[int] = 42  # Seed for reproducible random failures
) -> Instance:
    """
    Create a problem instance from CSV files.
    
    Args:
        scenario: Name of job scenario (normal, earthquake, baseline, peak_load, etc.)
        job_scenario: [DEPRECATED] Use 'scenario' instead
        failure_scenario: Name of failure scenario (earthquake_moderate, tsunami_warning, etc.)
                         or None for no failures
        time_horizon: Planning horizon in hours
        compute_cost_weight: Weight for compute cost in objective
        delay_penalty_weight: Weight for delay penalty in objective
        enable_random_failures: If True, add stochastic background failures
        random_failure_rate: Mean hours between random failures per machine
        random_seed: Random seed for reproducible failures (None for true randomness)
    
    Returns:
        Instance object
    """
    # Handle backward compatibility: prefer 'scenario' over 'job_scenario'
    effective_scenario = scenario if scenario is not None else (job_scenario if job_scenario is not None else "normal")
    
    # Determine base path - data is in parent directory since we're in helpers/
    base_path = Path(__file__).parent.parent / "more_data"
    
    # Load machines (same for all scenarios)
    servers, server_cost = load_machines(str(base_path / "machines.csv"))
    
    # Load jobs for specified scenario
    job_csv = base_path / f"scenario_{effective_scenario}.csv"
    jobs = load_jobs(str(job_csv))
    
    # Load machine failures if specified
    failures = []
    if failure_scenario:
        failures = load_machine_failures(str(base_path / "machine_failures.csv"), failure_scenario)
    
    # Add random background failures if enabled
    if enable_random_failures:
        random_failures = generate_random_failures(
            servers, 
            time_horizon, 
            mean_time_between_failures=random_failure_rate,
            seed=random_seed
        )
        failures.extend(random_failures)
    
    return Instance(
        servers=servers,
        jobs=jobs,
        server_cost=server_cost,
        time_horizon=time_horizon,
        weights={"compute_cost": compute_cost_weight, "delay_penalty": delay_penalty_weight},
        machine_failures=failures if failures else None
    )


def get_available_scenarios() -> Dict[str, List[str]]:
    """
    List all available scenarios.
    
    Returns:
        Dictionary with 'jobs' and 'failures' keys containing available scenarios
    """
    base_path = Path(__file__).parent.parent / "more_data"
    
    # Find job scenarios
    job_scenarios = []
    for f in base_path.glob("scenario_*.csv"):
        scenario_name = f.stem.replace("scenario_", "")
        job_scenarios.append(scenario_name)
    
    # Find failure scenarios
    failure_scenarios = []
    failures_file = base_path / "machine_failures.csv"
    if failures_file.exists():
        with open(failures_file, 'r') as f:
            reader = csv.DictReader(f)
            failure_scenarios = list(set(row['failure_scenario'] for row in reader))
    
    return {
        "jobs": sorted(job_scenarios),
        "failures": sorted(failure_scenarios)
    }


if __name__ == "__main__":
    # Test the loader
    print("Available scenarios:")
    scenarios = get_available_scenarios()
    print(f"  Job scenarios: {scenarios['jobs']}")
    print(f"  Failure scenarios: {scenarios['failures']}")
    
    print("\n" + "="*60)
    print("Testing baseline scenario:")
    inst = create_instance_from_csv("baseline")
    print(f"  Machines: {len(inst.servers)}")
    print(f"  Jobs: {len(inst.jobs)}")
    print(f"  Machine failures: {len(inst.machine_failures) if inst.machine_failures else 0}")
    
    print("\n" + "="*60)
    print("Testing emergency scenario with earthquake:")
    inst = create_instance_from_csv("emergency", "earthquake_moderate")
    print(f"  Machines: {len(inst.servers)}")
    print(f"  Jobs: {len(inst.jobs)}")
    print(f"  Machine failures: {len(inst.machine_failures) if inst.machine_failures else 0}")
    if inst.machine_failures:
        for failure in inst.machine_failures:
            print(f"    - {failure.machine_id}: t={failure.failure_start}-{failure.failure_end} ({failure.reason})")
