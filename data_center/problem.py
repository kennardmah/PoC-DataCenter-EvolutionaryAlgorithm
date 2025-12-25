"""
Problem Definition for Energy-Aware Job-Shop Scheduling
========================================================
Data structures and instance creation for the NP-hard scheduling problem.
"""

from dataclasses import dataclass
from typing import Dict, List, Any

# =============================================================================
# CONSTANTS
# =============================================================================

TIME_HORIZON = 24  # hours
TIME_UNIT = 1      # 1 hour per slot

ENERGY_PRICE = {  # Electricity price (¥ / kWh) by hour
    0: 8, 1: 8, 2: 8, 3: 8, 4: 8, 5: 8,          # off-peak
    6: 18, 7: 18, 8: 18, 9: 18,                  # morning peak
    10: 12, 11: 12, 12: 12, 13: 12, 14: 12, 15: 12, 16: 12,
    17: 22, 18: 22, 19: 22, 20: 22, 21: 22,      # evening peak
    22: 10, 23: 10                               # late night
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Server:
    """A server in the data center."""
    id: str
    server_type: str
    power_kw: float


@dataclass
class Operation:
    """An operation within a job."""
    server_type: str
    duration: int


@dataclass
class Job:
    """A job (workload) to be scheduled."""
    id: str
    description: str
    priority: str
    deadline: int
    delay_weight: float
    operations: List[Operation]


@dataclass
class Instance:
    """Complete problem instance."""
    servers: Dict[str, Server]
    jobs: Dict[str, Job]
    energy_price: Dict[int, float]
    time_horizon: int
    weights: Dict[str, float]


# Schedule type: job_id -> [(server_id, start_time), ...]
Schedule = Dict[str, List[Dict[str, Any]]]


# =============================================================================
# INSTANCE CREATION
# =============================================================================

def create_instance() -> Instance:
    """Create the default problem instance."""
    servers = {
        "S1": Server("S1", "CPU", 2.0),
        "S2": Server("S2", "CPU", 2.0),
        "S3": Server("S3", "GPU", 4.5),
        "S4": Server("S4", "Storage", 1.2),
    }
    
    jobs = {
        "J1": Job("J1", "Emergency response analytics", "HIGH", 8, 10.0, [
            Operation("Storage", 1), Operation("CPU", 2), Operation("GPU", 1)
        ]),
        "J2": Job("J2", "Disaster damage estimation", "HIGH", 10, 8.0, [
            Operation("Storage", 1), Operation("GPU", 2), Operation("CPU", 1)
        ]),
        "J3": Job("J3", "Tax data aggregation", "MEDIUM", 18, 4.0, [
            Operation("CPU", 2), Operation("Storage", 1), Operation("CPU", 1)
        ]),
        "J4": Job("J4", "Census preprocessing", "MEDIUM", 20, 3.0, [
            Operation("Storage", 1), Operation("CPU", 2)
        ]),
        "J5": Job("J5", "Open data batch update", "LOW", 22, 1.0, [
            Operation("CPU", 1), Operation("Storage", 1)
        ]),
        "J6": Job("J6", "Archive and backup", "LOW", 24, 0.5, [
            Operation("Storage", 2), Operation("CPU", 2)
        ]),
    }
    
    weights = {"energy_cost": 1.0, "delay_penalty": 5.0}
    
    return Instance(servers, jobs, ENERGY_PRICE, TIME_HORIZON, weights)


def generate(num_jobs: int = 6, num_servers: int = 4, seed: int = None) -> Instance:
    """
    Generate a random problem instance for testing.
    
    Args:
        num_jobs: Number of jobs to generate
        num_servers: Number of servers to generate
        seed: Random seed for reproducibility
    """
    import random
    if seed is not None:
        random.seed(seed)
    
    server_types = ["CPU", "GPU", "Storage"]
    priorities = ["HIGH", "MEDIUM", "LOW"]
    
    # Generate servers
    servers = {}
    for i in range(num_servers):
        sid = f"S{i+1}"
        stype = server_types[i % len(server_types)]
        power = random.uniform(1.0, 5.0)
        servers[sid] = Server(sid, stype, round(power, 1))
    
    # Generate jobs
    jobs = {}
    for i in range(num_jobs):
        jid = f"J{i+1}"
        priority = priorities[i % len(priorities)]
        deadline = random.randint(8, 24)
        delay_weight = {"HIGH": 10.0, "MEDIUM": 4.0, "LOW": 1.0}[priority]
        
        # Generate 2-4 operations per job
        num_ops = random.randint(2, 4)
        operations = []
        for _ in range(num_ops):
            op_type = random.choice(server_types)
            duration = random.randint(1, 3)
            operations.append(Operation(op_type, duration))
        
        jobs[jid] = Job(jid, f"Job {i+1}", priority, deadline, delay_weight, operations)
    
    return Instance(servers, jobs, ENERGY_PRICE, TIME_HORIZON, 
                   {"energy_cost": 1.0, "delay_penalty": 5.0})
