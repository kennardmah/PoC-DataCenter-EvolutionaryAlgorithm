# Repository Structure

## Root Level (Main Entry Points)
- `evaluate.py` - Main evaluation script for scheduling algorithms
- `initial.py` - FIFO baseline scheduler implementation
- `run_evo.py` - Evolutionary algorithm runner

## Helpers Directory
Helper modules and utilities:
- `scenario_loader.py` - Load scenarios and instances from CSV
- `online_wrapper.py` - Online scheduling wrapper (no future knowledge)
- `clairvoyant_solution.py` - Clairvoyant scheduler (perfect knowledge)

## Demos Directory
Example scripts and notebooks:
- `demo_pipeline.ipynb` - **Main visualization notebook** with online vs clairvoyant comparison
- `compare_online_offline.py` - Compare online vs clairvoyant performance
- `demo_capacity.py` - Capacity constraint demonstrations
- `demo_cloud.py` - Cloud overflow demonstrations  
- `demo_random_failures.py` - Random failure demonstrations
- `demo_combined_failures.py` - Combined failure scenarios
- `test_scenarios.py` - Scenario testing utilities
- `viz_schedules.ipynb` - Additional schedule visualizations

## Data Directory
- `machines.csv` - Machine specifications and time-varying costs
- `machine_failures.csv` - Pre-defined failure scenarios
- `scenario_normal.csv` - Normal day job workload (16 jobs)
- `scenario_earthquake.csv` - Emergency workload (10 jobs)

## Key Concepts

### Online vs Clairvoyant Scheduling
- **Online** (`online_wrapper.py`): Realistic scheduler that only sees jobs as they arrive
  - No future knowledge
  - Makes decisions incrementally
  - Competitive ratio: ~2.94x on weighted scenarios
  
- **Clairvoyant** (`initial.py` with full instance): Theoretical benchmark
  - Sees all future arrivals upfront
  - Can globally optimize priorities
  - Still respects arrival time constraints
  - Provides lower bound for online performance

### Usage
```python
# Online scheduling (realistic)
from helpers.online_wrapper import run_online_scheduling
schedule, fitness, score = run_online_scheduling("normal")

# Clairvoyant scheduling (benchmark)
from helpers.scenario_loader import create_instance_from_csv
from initial import construct_schedule, evaluate
inst = create_instance_from_csv("normal")
schedule = construct_schedule(inst)
fitness = evaluate(inst, schedule)

# Compare both
python demos/compare_online_offline.py
```
