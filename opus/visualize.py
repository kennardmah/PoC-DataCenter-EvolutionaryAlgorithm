"""
Visualization for DTDP
======================
ASCII and text-based visualizations for easy understanding.
"""

from typing import Dict, List
from problem import Instance, RailGraph
from fitness import Schedule, simulate, find_conflicts, TrainRun


# =============================================================================
# GRAPH VISUALIZATION
# =============================================================================

def show_graph(graph: RailGraph) -> str:
    """ASCII representation of the graph."""
    lines = []
    lines.append(f"Graph: |V|={len(graph.vertices)}, |E|={len(graph.edges)//2}")
    lines.append("")
    
    # Check if linear
    vertices = sorted(graph.vertices)
    is_linear = all(
        abs(u - v) == 1 
        for (u, v) in graph.edges.keys()
    )
    
    if is_linear and len(vertices) <= 10:
        # Linear display: [S0]--2--[S1]--3--[S2]
        parts = []
        for i, v in enumerate(vertices):
            parts.append(f"[{v}]")
            if i < len(vertices) - 1:
                t = graph.travel_time(v, vertices[i+1])
                parts.append(f"--{t}--")
        lines.append("".join(parts))
    else:
        # Adjacency list
        lines.append("Adjacency:")
        for v in vertices:
            neighbors = []
            for u in graph.neighbors(v):
                t = graph.travel_time(v, u)
                neighbors.append(f"{u}({t})")
            lines.append(f"  {v} -> {', '.join(neighbors)}")
    
    return "\n".join(lines)


# =============================================================================
# TRAIN VISUALIZATION
# =============================================================================

def show_trains(inst: Instance) -> str:
    """Show all trains and their routes."""
    lines = []
    lines.append(f"Trains: {inst.num_trains}")
    lines.append("")
    
    for t in inst.trains:
        path = inst.graph.path(t.origin, t.destination)
        path_str = "→".join(map(str, path)) if path else "NO PATH"
        pri = f"★{t.priority}" if t.priority > 1 else ""
        lines.append(f"  T{t.id}: {path_str}  earliest={t.earliest} {pri}")
    
    return "\n".join(lines)


# =============================================================================
# SCHEDULE VISUALIZATION
# =============================================================================

def show_schedule(inst: Instance, schedule: Schedule) -> str:
    """Show schedule as a timeline."""
    runs = simulate(inst, schedule)
    conflicts = find_conflicts(runs)
    
    lines = []
    lines.append("Schedule Timeline:")
    lines.append("")
    
    # Sort by departure
    runs_sorted = sorted(runs, key=lambda r: r.departure)
    
    for run in runs_sorted:
        train = next(t for t in inst.trains if t.id == run.train_id)
        delay = run.departure - train.earliest
        delay_str = f"(+{delay})" if delay > 0 else ""
        path_str = "→".join(map(str, run.path))
        
        lines.append(f"  t={run.departure:3d}: T{run.train_id} {path_str} "
                    f"[arr={run.arrival}] {delay_str}")
    
    if conflicts:
        lines.append("")
        lines.append(f"⚠ Conflicts: {len(conflicts)}")
        for c in conflicts[:5]:  # Show first 5
            lines.append(f"  {c['type']} on {c['segment']} at t={c['time']}: "
                        f"trains {c['trains']}")
        if len(conflicts) > 5:
            lines.append(f"  ... and {len(conflicts)-5} more")
    
    return "\n".join(lines)


# =============================================================================
# TIME-SPACE DIAGRAM (ASCII)
# =============================================================================

def show_timespace(inst: Instance, schedule: Schedule, width: int = 60) -> str:
    """
    ASCII time-space diagram showing train movements.
    
    Rows = stations, Columns = time
    """
    runs = simulate(inst, schedule)
    
    if not runs:
        return "No trains to display"
    
    # Find time range
    max_time = max(r.arrival for r in runs)
    time_scale = max(1, max_time // width)
    
    stations = sorted(inst.graph.vertices)
    
    # Build grid: station -> time -> train_ids
    grid: Dict[int, Dict[int, List[int]]] = {s: {} for s in stations}
    
    for run in runs:
        # Mark departure station
        t_scaled = run.departure // time_scale
        if t_scaled not in grid[run.path[0]]:
            grid[run.path[0]][t_scaled] = []
        grid[run.path[0]][t_scaled].append(run.train_id)
        
        # Mark each segment traversal
        for (u, v, t_start, t_end) in run.segments:
            for t in range(t_start, t_end + 1):
                t_scaled = t // time_scale
                if t_scaled not in grid[v]:
                    grid[v][t_scaled] = []
                if run.train_id not in grid[v].get(t_scaled, []):
                    grid[v][t_scaled] = grid[v].get(t_scaled, []) + [run.train_id]
    
    # Render
    lines = []
    lines.append("Time-Space Diagram (station × time)")
    lines.append(f"Scale: 1 char = {time_scale} time units")
    lines.append("")
    
    # Time axis
    time_axis = "     "
    for t in range(0, width, 10):
        time_axis += f"{t*time_scale:<10}"
    lines.append(time_axis)
    
    # Station rows
    for s in stations:
        row = f"S{s:<3} "
        for t in range(width):
            trains = grid[s].get(t, [])
            if not trains:
                row += "·"
            elif len(trains) == 1:
                row += str(trains[0] % 10)  # Single digit
            else:
                row += "X"  # Conflict
        lines.append(row)
    
    return "\n".join(lines)


# =============================================================================
# FULL REPORT
# =============================================================================

def report(inst: Instance, schedule: Schedule) -> str:
    """Generate full visualization report."""
    from fitness import evaluate
    
    fit = evaluate(inst, schedule)
    
    sections = []
    
    # Header
    sections.append("=" * 60)
    sections.append("DTDP SOLUTION REPORT")
    sections.append("=" * 60)
    
    # Graph
    sections.append("")
    sections.append(show_graph(inst.graph))
    
    # Trains
    sections.append("")
    sections.append(show_trains(inst))
    
    # Fitness
    sections.append("")
    sections.append("Fitness:")
    sections.append(f"  Score: {fit.score} (lower = better)")
    sections.append(f"  Delay: {fit.delay}")
    sections.append(f"  Conflicts: {fit.conflicts} ({fit.num_conflicts})")
    sections.append(f"  Makespan: {fit.makespan}")
    
    # Schedule
    sections.append("")
    sections.append(show_schedule(inst, schedule))
    
    # Time-space diagram
    sections.append("")
    sections.append(show_timespace(inst, schedule))
    
    return "\n".join(sections)


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    from problem import generate
    from solver import greedy_fcfs
    
    inst = generate(num_stations=5, num_trains=6, seed=42)
    schedule = greedy_fcfs(inst)
    
    print(report(inst, schedule))
