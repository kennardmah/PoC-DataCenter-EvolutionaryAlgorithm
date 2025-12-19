"""
ShinkaEvolve Integration
========================
Interface for algorithm evolution using GenAI.

This module provides:
1. Test instances of varying difficulty
2. Evaluation harness for evolved solvers
3. Fitness feedback for the LLM
"""

from typing import Callable, List, Dict, Any
from problem import Instance, generate
from fitness import Schedule, evaluate, Fitness
from solver import Solver, SOLVER_TEMPLATE


# =============================================================================
# TEST SUITE
# =============================================================================

def make_test_suite() -> List[Instance]:
    """
    Create a diverse test suite for evaluating evolved algorithms.
    
    Includes:
    - Easy (few trains, low conflict)
    - Medium (moderate conflicts)
    - Hard (many trains, high conflict density)
    """
    return [
        # Easy
        generate(num_stations=4, num_trains=4, seed=1),
        generate(num_stations=5, num_trains=5, seed=2),
        
        # Medium
        generate(num_stations=5, num_trains=8, seed=10),
        generate(num_stations=6, num_trains=10, seed=11),
        generate(num_stations=7, num_trains=10, seed=12, topology="branching"),
        
        # Hard
        generate(num_stations=8, num_trains=15, seed=20),
        generate(num_stations=10, num_trains=20, seed=21),
        generate(num_stations=10, num_trains=20, seed=22, topology="branching"),
    ]


# =============================================================================
# EVALUATION HARNESS
# =============================================================================

def evaluate_solver(solver: Solver, instances: List[Instance] = None) -> Dict[str, Any]:
    """
    Evaluate a solver on the test suite.
    
    Returns:
        dict with:
        - total_score: Sum of all fitness scores (lower = better)
        - avg_score: Average fitness score
        - num_conflicts: Total conflicts across all instances
        - per_instance: List of per-instance results
    """
    if instances is None:
        instances = make_test_suite()
    
    results = []
    total_score = 0
    total_conflicts = 0
    
    for i, inst in enumerate(instances):
        try:
            schedule = solver(inst)
            fit = evaluate(inst, schedule)
            results.append({
                "instance": i,
                "score": fit.score,
                "conflicts": fit.num_conflicts,
                "delay": fit.delay,
                "success": True
            })
            total_score += fit.score
            total_conflicts += fit.num_conflicts
        except Exception as e:
            # Penalize errors heavily
            results.append({
                "instance": i,
                "score": 10000,
                "conflicts": 999,
                "delay": 999,
                "success": False,
                "error": str(e)
            })
            total_score += 10000
            total_conflicts += 999
    
    return {
        "total_score": round(total_score, 2),
        "avg_score": round(total_score / len(instances), 2),
        "num_conflicts": total_conflicts,
        "num_instances": len(instances),
        "success_rate": sum(1 for r in results if r["success"]) / len(instances),
        "per_instance": results
    }


# =============================================================================
# FEEDBACK GENERATOR
# =============================================================================

def generate_feedback(result: Dict[str, Any], verbose: bool = False) -> str:
    """
    Generate human-readable feedback for the LLM.
    
    This helps ShinkaEvolve understand what to improve.
    """
    lines = []
    
    lines.append(f"EVALUATION RESULTS")
    lines.append(f"==================")
    lines.append(f"Total Score: {result['total_score']} (lower = better)")
    lines.append(f"Average Score: {result['avg_score']}")
    lines.append(f"Total Conflicts: {result['num_conflicts']}")
    lines.append(f"Success Rate: {result['success_rate']*100:.0f}%")
    
    if verbose:
        lines.append("")
        lines.append("Per-Instance Breakdown:")
        for r in result["per_instance"]:
            status = "✓" if r["success"] else "✗"
            lines.append(f"  {status} Instance {r['instance']}: "
                        f"score={r['score']:.1f}, conflicts={r['conflicts']}")
            if "error" in r:
                lines.append(f"      Error: {r['error']}")
    
    # Suggestions based on results
    lines.append("")
    lines.append("Suggestions:")
    
    if result["num_conflicts"] > 10:
        lines.append("  - High conflict count. Try separating trains by direction or time.")
    
    avg_delay = sum(r.get("delay", 0) for r in result["per_instance"]) / len(result["per_instance"])
    if avg_delay > 50:
        lines.append("  - High delay. Consider prioritizing high-priority trains.")
    
    if result["success_rate"] < 1.0:
        lines.append("  - Some instances failed. Check for edge cases (empty paths, etc.)")
    
    return "\n".join(lines)


# =============================================================================
# SHINKAEVOLVE INTERFACE
# =============================================================================

class DTDPProblem:
    """
    Problem interface for ShinkaEvolve.
    
    Usage:
        problem = DTDPProblem()
        score = problem.evaluate(solver_code)
        feedback = problem.get_feedback()
    """
    
    def __init__(self, difficulty: str = "medium"):
        """
        Initialize with difficulty level.
        
        Args:
            difficulty: "easy", "medium", "hard", or "full"
        """
        all_instances = make_test_suite()
        
        if difficulty == "easy":
            self.instances = all_instances[:2]
        elif difficulty == "medium":
            self.instances = all_instances[2:5]
        elif difficulty == "hard":
            self.instances = all_instances[5:]
        else:  # full
            self.instances = all_instances
        
        self.last_result = None
    
    def get_template(self) -> str:
        """Get the solver template code."""
        return SOLVER_TEMPLATE
    
    def evaluate(self, solver: Solver) -> float:
        """
        Evaluate a solver function.
        
        Args:
            solver: Function taking Instance, returning Schedule
        
        Returns:
            Total fitness score (lower = better)
        """
        self.last_result = evaluate_solver(solver, self.instances)
        return self.last_result["total_score"]
    
    def get_feedback(self, verbose: bool = True) -> str:
        """Get feedback from last evaluation."""
        if self.last_result is None:
            return "No evaluation run yet."
        return generate_feedback(self.last_result, verbose)
    
    def get_best_baseline(self) -> float:
        """Get best baseline score for comparison."""
        from solver import greedy_fcfs, greedy_priority, direction_wave
        
        best = float("inf")
        for solver in [greedy_fcfs, greedy_priority, direction_wave]:
            result = evaluate_solver(solver, self.instances)
            best = min(best, result["total_score"])
        return best


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    from solver import naive, greedy_fcfs, greedy_priority, direction_wave
    
    problem = DTDPProblem(difficulty="full")
    
    print("Testing baseline solvers...")
    print()
    
    for name, solver in [
        ("naive", naive),
        ("greedy_fcfs", greedy_fcfs),
        ("greedy_priority", greedy_priority),
        ("direction_wave", direction_wave),
    ]:
        score = problem.evaluate(solver)
        print(f"{name}: {score}")
    
    print()
    print(problem.get_feedback())
