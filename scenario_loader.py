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


@dataclass
class MachineFailure:
    machine_id: str
    failure_start: int
    failure_end: int
    capacity_reduction: int  # Amount of capacity lost during failure
    failure_rate: float = 1.0  # Exponential failure rate (1.0 = instant, <1.0 = gradual)
    recovery_rate: float = 1.0  # Exponential recovery rate (1.0 = instant, <1.0 = gradual)
    reason: str = ""  # Description of failure


@dataclass
class Instance:
    servers: Dict[str, Server]
    jobs: Dict[str, Job]
    server_cost: Dict[str, Dict[int, float]]
    time_horizon: int
    weights: Dict[str, float]
    machine_failures: Optional[List[MachineFailure]] = None


def load_machines(csv_path: str = "data/machines.csv") -> Tuple[Dict[str, Server], Dict[str, Dict[int, float]]]:
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
            is_cloud = row.get('is_cloud', 'false').lower() == 'true'
            
            servers[machine_id] = Server(machine_id, machine_type, is_cloud)
            
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
            
            # Parse operations: "Ingest:2,CPU:3,GPU:1"
            operations = []
            for op_str in row['operations'].split(','):
                op_type, duration = op_str.split(':')
                operations.append(Operation(op_type, int(duration)))
            
            jobs[job_id] = Job(job_id, description, priority, deadline, delay_weight, operations)
    
    return jobs


def load_machine_failures(csv_path: str = "data/machine_failures.csv", 
                          scenario: Optional[str] = None) -> List[MachineFailure]:
    """
    Load machine failure scenarios from CSV.
    
    Args:
        csv_path: Path to machine failures CSV
        scenario: Specific failure scenario to load (e.g., 'earthquake_moderate')
                 If None, no failures are loaded
    
    Returns:
        List of MachineFailure objects with performance degradation
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
                    reason=row['reason']
                ))
    
    return failures


def create_instance_from_csv(
    job_scenario: str = "baseline",
    failure_scenario: Optional[str] = None,
    time_horizon: int = 24,
    compute_cost_weight: float = 1.0,
    delay_penalty_weight: float = 5.0
) -> Instance:
    """
    Create a problem instance from CSV files.
    
    Args:
        job_scenario: Name of job scenario (baseline, peak_load, emergency, light_load)
        failure_scenario: Name of failure scenario (earthquake_moderate, tsunami_warning, etc.)
                         or None for no failures
        time_horizon: Planning horizon in hours
        compute_cost_weight: Weight for compute cost in objective
        delay_penalty_weight: Weight for delay penalty in objective
    
    Returns:
        Instance object
    """
    # Determine base path
    base_path = Path(__file__).parent / "data"
    
    # Load machines (same for all scenarios)
    servers, server_cost = load_machines(str(base_path / "machines.csv"))
    
    # Load jobs for specified scenario
    job_csv = base_path / f"scenario_{job_scenario}.csv"
    jobs = load_jobs(str(job_csv))
    
    # Load machine failures if specified
    failures = []
    if failure_scenario:
        failures = load_machine_failures(str(base_path / "machine_failures.csv"), failure_scenario)
    
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
    base_path = Path(__file__).parent.parent / "data"
    
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
