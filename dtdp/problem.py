"""
DTDP - Dynamic Train Dispatching Problem
=========================================
A minimal NP-hard problem for Japan railway scheduling PoC.

Graph Model: G = (V, E)
  V = stations (nodes)
  E = single-track segments with travel times (weighted edges)

Decision: When should each train depart to minimize conflicts and delays?
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import json
import random


# =============================================================================
# GRAPH STRUCTURE
# =============================================================================

@dataclass
class RailGraph:
    """
    Railway network as a weighted directed graph.
    
    G = (V, E) where:
        V = station IDs (integers)
        E = {(u, v): travel_time} for connected stations
    """
    vertices: Set[int]
    edges: Dict[Tuple[int, int], int]
    
    def __post_init__(self):
        self._adj: Dict[int, List[int]] = defaultdict(list)
        for (u, v) in self.edges:
            self._adj[u].append(v)
    
    def neighbors(self, v: int) -> List[int]:
        return self._adj[v]
    
    def travel_time(self, u: int, v: int) -> Optional[int]:
        return self.edges.get((u, v))
    
    def path(self, origin: int, dest: int) -> Optional[List[int]]:
        """BFS shortest path."""
        if origin == dest:
            return [origin]
        visited = {origin}
        queue = [(origin, [origin])]
        while queue:
            curr, p = queue.pop(0)
            for nxt in self.neighbors(curr):
                if nxt == dest:
                    return p + [nxt]
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, p + [nxt]))
        return None
    
    def to_dot(self) -> str:
        """Export to Graphviz DOT format."""
        lines = [
            "graph Rail {",
            "  rankdir=LR;",
            "  node [shape=circle, style=filled, fillcolor=lightblue];"
        ]
        seen = set()
        for (u, v), t in self.edges.items():
            edge = tuple(sorted([u, v]))
            if edge not in seen:
                seen.add(edge)
                lines.append(f'  {u} -- {v} [label="{t}"];')
        lines.append("}")
        return "\n".join(lines)


# =============================================================================
# PROBLEM INSTANCE
# =============================================================================

@dataclass
class Train:
    """A train to be scheduled."""
    id: int
    origin: int
    destination: int
    earliest: int      # Earliest departure time
    priority: float    # Delay penalty weight (1.0 = normal)


@dataclass
class Instance:
    """Complete DTDP problem instance."""
    graph: RailGraph
    trains: List[Train]
    horizon: int       # Maximum time
    
    @property
    def num_stations(self) -> int:
        return len(self.graph.vertices)
    
    @property
    def num_trains(self) -> int:
        return len(self.trains)


# =============================================================================
# INSTANCE GENERATORS
# =============================================================================

def make_linear(n: int, min_t: int = 2, max_t: int = 5, seed: int = None) -> RailGraph:
    """
    Linear corridor: S0 -- S1 -- S2 -- ... -- Sn-1
    
    Simple topology, good for initial testing.
    """
    if seed is not None:
        random.seed(seed)
    
    V = set(range(n))
    E = {}
    for i in range(n - 1):
        t = random.randint(min_t, max_t)
        E[(i, i+1)] = t
        E[(i+1, i)] = t
    return RailGraph(V, E)


def make_branching(n: int, seed: int = None) -> RailGraph:
    """
    Y-shaped network with branches.
    
         S4
         |
    S0--S1--S2--S3
         |
         S5--S6
    
    More complex, tests path-finding.
    """
    if seed is not None:
        random.seed(seed)
    
    V = set(range(n))
    E = {}
    
    # Main line (60% of stations)
    main_len = max(3, int(n * 0.6))
    for i in range(main_len - 1):
        t = random.randint(2, 5)
        E[(i, i+1)] = t
        E[(i+1, i)] = t
    
    # Branches from middle stations
    next_id = main_len
    for branch_from in range(1, main_len - 1):
        if next_id >= n:
            break
        if random.random() < 0.5:
            t = random.randint(2, 4)
            E[(branch_from, next_id)] = t
            E[(next_id, branch_from)] = t
            # Extend branch
            while next_id + 1 < n and random.random() < 0.4:
                t = random.randint(2, 4)
                E[(next_id, next_id + 1)] = t
                E[(next_id + 1, next_id)] = t
                next_id += 1
            next_id += 1
    
    return RailGraph(V, E)


def make_trains(
    graph: RailGraph,
    num_trains: int,
    horizon: int,
    seed: int = None,
    conflict_density: float = 0.7
) -> List[Train]:
    """
    Generate trains with clustered departures to create conflicts.
    
    conflict_density: 0.0 = spread out, 1.0 = highly clustered
    """
    if seed is not None:
        random.seed(seed)
    
    stations = list(graph.vertices)
    trains = []
    
    # Create time clusters for conflicts
    num_clusters = max(1, num_trains // 3)
    clusters = [random.randint(0, horizon // 3) for _ in range(num_clusters)]
    
    for i in range(num_trains):
        # Random OD pair with valid path
        while True:
            o = random.choice(stations)
            d = random.choice(stations)
            if o != d and graph.path(o, d):
                break
        
        # Clustered or spread departure
        if random.random() < conflict_density:
            earliest = max(0, random.choice(clusters) + random.randint(-2, 2))
        else:
            earliest = random.randint(0, horizon // 2)
        
        # Most normal priority, some high priority
        priority = 1.0 if random.random() < 0.8 else round(random.uniform(2, 5), 1)
        
        trains.append(Train(i, o, d, earliest, priority))
    
    return trains


def generate(
    num_stations: int = 5,
    num_trains: int = 8,
    horizon: int = 50,
    topology: str = "linear",
    seed: int = None
) -> Instance:
    """
    Generate a complete DTDP instance.
    
    Args:
        num_stations: Number of stations (|V|)
        num_trains: Number of trains to schedule
        horizon: Time horizon
        topology: "linear" or "branching"
        seed: Random seed for reproducibility
    """
    if topology == "linear":
        graph = make_linear(num_stations, seed=seed)
    else:
        graph = make_branching(num_stations, seed=seed)
    
    trains = make_trains(graph, num_trains, horizon, seed=seed)
    
    return Instance(graph, trains, horizon)


# =============================================================================
# SERIALIZATION
# =============================================================================

def to_dict(inst: Instance) -> dict:
    return {
        "graph": {
            "V": list(inst.graph.vertices),
            "E": {f"{u},{v}": t for (u,v), t in inst.graph.edges.items()}
        },
        "trains": [
            {"id": t.id, "o": t.origin, "d": t.destination, 
             "earliest": t.earliest, "priority": t.priority}
            for t in inst.trains
        ],
        "horizon": inst.horizon
    }


def from_dict(d: dict) -> Instance:
    V = set(d["graph"]["V"])
    E = {tuple(map(int, k.split(","))): v for k, v in d["graph"]["E"].items()}
    graph = RailGraph(V, E)
    trains = [Train(**t) for t in d["trains"]]
    return Instance(graph, trains, d["horizon"])


def save(inst: Instance, path: str):
    with open(path, "w") as f:
        json.dump(to_dict(inst), f, indent=2)


def load(path: str) -> Instance:
    with open(path) as f:
        return from_dict(json.load(f))


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    inst = generate(num_stations=5, num_trains=6, seed=42)
    
    print("DTDP Instance")
    print("=" * 40)
    print(f"|V| = {inst.num_stations}, |E| = {len(inst.graph.edges)//2}")
    print(f"Trains: {inst.num_trains}")
    print()
    
    print("Graph (DOT):")
    print(inst.graph.to_dot())
    print()
    
    print("Trains:")
    for t in inst.trains:
        path = inst.graph.path(t.origin, t.destination)
        path_str = "→".join(map(str, path)) if path else "NO PATH"
        print(f"  T{t.id}: {path_str} | earliest={t.earliest} | priority={t.priority}")
