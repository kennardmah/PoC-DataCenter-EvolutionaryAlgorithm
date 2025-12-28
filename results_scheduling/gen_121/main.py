# EVOLVE-BLOCK-START
<NAME>
scheduler_encapsulated_precomputed_contention
</NAME>
<DESCRIPTION>
This version structurally redesigns the `construct_schedule` function by encapsulating the scheduling logic within a `JobShopScheduler` class, enhancing modularity and state management.

Key structural and algorithmic improvements include:
1.  **Class Encapsulation**: All scheduling logic and shared state (capacity usage, job availability, precomputed data) are now contained within the `JobShopScheduler` class. This improves code organization, readability, and makes it easier to extend or modify specific scheduling components.
2.  **Pre-computation of Dynamic Capacities (Recommendation 3)**: A new `_precompute_all_capacities` method is introduced to calculate and cache the effective available capacity for all local servers across the entire `time_horizon`. This significantly reduces redundant and computationally intensive `math.exp` calls within the frequently invoked `_get_available_capacity` and `_has_capacity` methods during the scheduling loop.
3.  **Unified Candidate Evaluation and Multi-Criteria Tie-breaking (Recommendation 4 adapted)**: Instead of separate local and cloud search loops, this version collects all viable operation placements (local and cloud) into a single `candidate_placements` list. Each candidate is a tuple including its objective cost, whether it's a cloud server, its start time, the minimum remaining free capacity on the server during the operation (for local servers), and the server ID. This list is then sorted using a robust multi-criteria heuristic:
    *   Minimize total objective cost (primary).
    *   Prefer local servers over cloud servers (secondary).
    *   Minimize the operation's start time (tertiary).
    *   **Maximize the minimum free capacity during the operation** (quaternary, implemented by negating this value for ascending sort). This directly addresses Recommendation 4 by preferring more robust placements with less contention or more buffer against further unexpected capacity reductions.
    *   Minimize server ID (for deterministic tie-breaking).
4.  **Dedicated Helper Methods**: The class structure promotes breaking down complex logic into smaller, focused methods (e.g., `_calculate_operation_compute_cost`, `_calculate_remaining_capacity_sum`, `_reserve_capacity`), making the code easier to understand and debug.
5.  **Input Data Immutability**: Job properties like `total_duration` and `priority_score` are now computed and stored internally within the `Scheduler` instance as dictionaries (`self.job_total_durations`, `self.job_priority_scores`) rather than modifying the input `Job` dataclass objects, preserving the integrity of the original `Instance`.
</DESCRIPTION>
<CODE>
# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_scheduling(
    scenario: str = "baseline",
    failure_scenario: Optional[str] = None,
    multi_scenario: bool = False,
    scenarios: Optional[List[str]] = None,
    scenario_weights: Optional[Dict[str, float]] = None,
    failure_scenarios: Optional[Dict[str, str]] = None
):
    """
    Run the schedule constructor and return results.

    Args:
        scenario: Job scenario to load from CSV (for single-scenario mode)
        failure_scenario: Machine failure scenario from CSV or None (for single-scenario mode)
        multi_scenario: If True, evaluate across multiple scenarios
        scenarios: List of scenario names to evaluate (for multi-scenario mode)
        scenario_weights: Weight for each scenario in final score (for multi-scenario mode)
        failure_scenarios: Mapping of scenario name to failure scenario (for multi-scenario mode)
    """
    from helpers.scenario_loader import create_instance_from_csv

    if multi_scenario and scenarios:
        # Multi-scenario evaluation
        results = {}
        weighted_score = 0.0

        for scenario_name in scenarios:
            # Get failure scenario for this scenario (if any)
            fail_scenario = None
            if failure_scenarios and scenario_name in failure_scenarios:
                fail_scenario = failure_scenarios[scenario_name]

            # Create instance and run scheduling
            inst = create_instance_from_csv(scenario=scenario_name, failure_scenario=fail_scenario)
            schedule = construct_schedule(inst)
            fitness = evaluate(inst, schedule)
            score = float(fitness.score)

            # Store results
            results[scenario_name] = (schedule, fitness, score)

            # Add weighted contribution
            weight = scenario_weights.get(scenario_name, 1.0) if scenario_weights else 1.0
            weighted_score += weight * score

        # Return multi-scenario results
        return {
            "scenarios": results,
            "weighted_score": weighted_score,
            "weights": scenario_weights or {s: 1.0 for s in scenarios}
        }
    else:
        # Single scenario evaluation (backward compatibility)
        inst = create_instance_from_csv(scenario=scenario, failure_scenario=failure_scenario)
        schedule = construct_schedule(inst)
        fitness = evaluate(inst, schedule)
        return schedule, fitness, float(fitness.score)