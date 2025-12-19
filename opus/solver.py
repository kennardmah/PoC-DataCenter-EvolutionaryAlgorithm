"""
Solver Interface for ShinkaEvolve
=================================
This module defines the interface that evolved algorithms must implement.

ShinkaEvolve will generate Python functions that follow this interface,
then evaluate them on test instances.
"""

from typing import Dict, List, Callable
from problem import Instance, Train
from fitness import Schedule, evaluate, Fitness


# =============================================================================
# SOLVER INTERFACE
# =============================================================================

# Type alias for solver functions
# A solver takes an Instance and returns a Schedule
Solver = Callable[[Instance], Schedule]


def run_solver(solver: Solver, inst: Instance) -> Fitness:
    """
    Run a solver on an instance and return its fitness.
    
    This is the main entry point for ShinkaEvolve evaluation.
    """
    schedule = solver(inst)
    return evaluate(inst, schedule)


# =============================================================================
# BASELINE SOLVERS (for comparison)
# =============================================================================

def naive(inst: Instance) -> Schedule:
    """
    Baseline: All trains depart at earliest time.
    Fast but usually many conflicts.
    """
    return {t.id: t.earliest for t in inst.trains}


def greedy_fcfs(inst: Instance) -> Schedule:
    """
    First-Come-First-Served: Schedule in order of earliest departure.
    Each train gets first available slot without conflicts.
    """
    from fitness import simulate, find_conflicts
    
    schedule = {}
    sorted_trains = sorted(inst.trains, key=lambda t: t.earliest)
    
    for train in sorted_trains:
        best_time = train.earliest
        
        # Try times until no conflict
        for t in range(train.earliest, inst.horizon):
            test = {**schedule, train.id: t}
            runs = simulate(inst, test)
            if not find_conflicts(runs):
                best_time = t
                break
        
        schedule[train.id] = best_time
    
    return schedule


def greedy_priority(inst: Instance) -> Schedule:
    """
    Priority-first: High priority trains scheduled first.
    """
    from fitness import simulate, find_conflicts
    
    schedule = {}
    sorted_trains = sorted(inst.trains, key=lambda t: (-t.priority, t.earliest))
    
    for train in sorted_trains:
        best_time = train.earliest
        
        for t in range(train.earliest, inst.horizon):
            test = {**schedule, train.id: t}
            runs = simulate(inst, test)
            if not find_conflicts(runs):
                best_time = t
                break
        
        schedule[train.id] = best_time
    
    return schedule


def direction_wave(inst: Instance) -> Schedule:
    """
    Separate by direction: eastbound first, then westbound.
    Reduces head-on conflicts.
    """
    schedule = {}
    
    # Classify by direction (destination > origin = "east")
    east = [t for t in inst.trains if t.destination > t.origin]
    west = [t for t in inst.trains if t.destination < t.origin]
    
    # Schedule eastbound
    time = 0
    for t in sorted(east, key=lambda x: x.earliest):
        schedule[t.id] = max(t.earliest, time)
        time = schedule[t.id] + 2
    
    # Gap between waves
    time += 5
    
    # Schedule westbound
    for t in sorted(west, key=lambda x: x.earliest):
        schedule[t.id] = max(t.earliest, time)
        time = schedule[t.id] + 2
    
    return schedule


# =============================================================================
# SOLVER TEMPLATE FOR SHINKAEVOLVE
# =============================================================================

SOLVER_TEMPLATE = '''
def solve(inst: Instance) -> Schedule:
    """
    Solve the DTDP instance.
    
    Args:
        inst: Problem instance with:
            - inst.graph: RailGraph with vertices, edges
            - inst.trains: List[Train] with id, origin, destination, earliest, priority
            - inst.horizon: Maximum time
    
    Returns:
        Schedule: Dict[train_id, departure_time]
    
    Available helpers:
        - inst.graph.path(origin, dest) -> List[int] or None
        - inst.graph.travel_time(u, v) -> int
        - inst.graph.neighbors(v) -> List[int]
    """
    schedule = {}
    
    # YOUR ALGORITHM HERE
    # Example: schedule each train at its earliest time
    for train in inst.trains:
        schedule[train.id] = train.earliest
    
    return schedule
'''


# =============================================================================
# BENCHMARK
# =============================================================================

def benchmark(inst: Instance, solvers: Dict[str, Solver] = None) -> Dict[str, Fitness]:
    """
    Run multiple solvers on an instance and compare results.
    """
    if solvers is None:
        solvers = {
            "naive": naive,
            "greedy_fcfs": greedy_fcfs,
            "greedy_priority": greedy_priority,
            "direction_wave": direction_wave,
        }
    
    results = {}
    for name, solver in solvers.items():
        results[name] = run_solver(solver, inst)
    
    return results


def print_benchmark(results: Dict[str, Fitness]):
    """Pretty print benchmark results."""
    print(f"{'Solver':<20} {'Score':>10} {'Conflicts':>10} {'Delay':>10}")
    print("-" * 52)
    
    for name, fit in sorted(results.items(), key=lambda x: x[1].score):
        print(f"{name:<20} {fit.score:>10.1f} {fit.num_conflicts:>10} {fit.delay:>10.1f}")


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    from problem import generate
    
    inst = generate(num_stations=5, num_trains=8, seed=42)
    
    print(f"Instance: {inst.num_stations} stations, {inst.num_trains} trains")
    print()
    
    results = benchmark(inst)
    print_benchmark(results)
    
    print()
    print("Best:", min(results.items(), key=lambda x: x[1].score)[0])
