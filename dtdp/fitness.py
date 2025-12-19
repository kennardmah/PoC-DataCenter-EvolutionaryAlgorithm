"""
Fitness Evaluation for DTDP
===========================
Evaluates a schedule (train departures) and returns fitness score.

Solution format: Dict[train_id, departure_time]
Fitness: Lower is better (0 = perfect)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Set
from problem import Instance, Train


# =============================================================================
# SOLUTION TYPE
# =============================================================================

Schedule = Dict[int, int]  # train_id -> departure_time


# =============================================================================
# SIMULATION
# =============================================================================

@dataclass
class TrainRun:
    """Result of simulating one train."""
    train_id: int
    departure: int
    arrival: int
    path: List[int]
    segments: List[Tuple[int, int, int, int]]  # (from, to, start_time, end_time)


def simulate(inst: Instance, schedule: Schedule) -> List[TrainRun]:
    """
    Simulate all trains following the schedule.
    Returns timing info for each train.
    """
    runs = []
    
    for train in inst.trains:
        dep = schedule.get(train.id, train.earliest)
        path = inst.graph.path(train.origin, train.destination)
        
        if not path:
            continue
        
        segments = []
        t = dep
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            travel = inst.graph.travel_time(u, v)
            segments.append((u, v, t, t + travel))
            t += travel
        
        runs.append(TrainRun(train.id, dep, t, path, segments))
    
    return runs


# =============================================================================
# CONFLICT DETECTION
# =============================================================================

def find_conflicts(runs: List[TrainRun]) -> List[dict]:
    """
    Find all segment conflicts.
    
    Conflict types:
    - same_dir: Two trains on same segment, same direction
    - head_on: Two trains on same segment, opposite directions
    """
    # Build segment occupancy: (u, v, t) -> set of train_ids
    occupancy: Dict[Tuple[int, int, int], Set[int]] = {}
    
    for run in runs:
        for (u, v, t_start, t_end) in run.segments:
            for t in range(t_start, t_end):
                key = (u, v, t)
                if key not in occupancy:
                    occupancy[key] = set()
                occupancy[key].add(run.train_id)
    
    conflicts = []
    seen = set()
    
    for (u, v, t), trains in occupancy.items():
        # Same direction conflict
        if len(trains) > 1:
            key = ("same", tuple(sorted(trains)), t)
            if key not in seen:
                seen.add(key)
                conflicts.append({
                    "type": "same_dir",
                    "segment": (u, v),
                    "time": t,
                    "trains": list(trains)
                })
        
        # Head-on conflict (opposite direction at same time)
        opposite = occupancy.get((v, u, t), set())
        if opposite:
            all_trains = trains | opposite
            key = ("head_on", tuple(sorted(all_trains)), t)
            if key not in seen:
                seen.add(key)
                conflicts.append({
                    "type": "head_on",
                    "segment": (u, v),
                    "time": t,
                    "trains": list(all_trains)
                })
    
    return conflicts


# =============================================================================
# FITNESS FUNCTION
# =============================================================================

@dataclass
class Fitness:
    """Fitness evaluation result."""
    score: float           # Total fitness (lower = better)
    delay: float           # Delay penalty
    conflicts: float       # Conflict penalty  
    makespan: float        # Schedule length penalty
    
    num_conflicts: int     # Raw conflict count
    completed: int         # Trains that reached destination
    total_trains: int      # Total trains
    
    details: dict = None   # Optional detailed breakdown


# Default weights - tuned for clear fitness signal
WEIGHTS = {
    "delay": 1.0,       # Per time unit × priority
    "conflict": 50.0,   # Per conflict (heavy penalty)
    "makespan": 0.1,    # Per time unit of total duration
}


def evaluate(inst: Instance, schedule: Schedule, weights: dict = None) -> Fitness:
    """
    Evaluate a schedule's fitness.
    
    Args:
        inst: Problem instance
        schedule: Dict mapping train_id -> departure_time
        weights: Optional custom weights
    
    Returns:
        Fitness object with score breakdown
    """
    w = {**WEIGHTS, **(weights or {})}
    
    # Simulate
    runs = simulate(inst, schedule)
    conflicts = find_conflicts(runs)
    
    # Delay penalty: sum of (departure - earliest) * priority
    delay = 0.0
    train_map = {t.id: t for t in inst.trains}
    for run in runs:
        train = train_map[run.train_id]
        d = max(0, run.departure - train.earliest)
        delay += d * train.priority
    
    # Conflict penalty
    conflict_score = len(conflicts) * w["conflict"]
    
    # Makespan penalty
    if runs:
        makespan = max(r.arrival for r in runs)
    else:
        makespan = 0
    makespan_score = makespan * w["makespan"]
    
    # Total score
    total = delay * w["delay"] + conflict_score + makespan_score
    
    return Fitness(
        score=round(total, 2),
        delay=round(delay * w["delay"], 2),
        conflicts=round(conflict_score, 2),
        makespan=round(makespan_score, 2),
        num_conflicts=len(conflicts),
        completed=len(runs),
        total_trains=len(inst.trains),
        details={"runs": runs, "conflict_list": conflicts}
    )


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    from problem import generate
    
    inst = generate(num_stations=5, num_trains=6, seed=42)
    
    # Naive schedule: all depart at earliest
    schedule = {t.id: t.earliest for t in inst.trains}
    
    result = evaluate(inst, schedule)
    
    print("Fitness Evaluation")
    print("=" * 40)
    print(f"Total Score: {result.score} (lower = better)")
    print(f"  Delay:     {result.delay}")
    print(f"  Conflicts: {result.conflicts} ({result.num_conflicts} conflicts)")
    print(f"  Makespan:  {result.makespan}")
    print(f"Completed: {result.completed}/{result.total_trains}")
