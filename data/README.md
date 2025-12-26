# Government Data Center Scheduling Scenarios

This directory contains CSV-based scenario definitions for testing the scheduling algorithm across diverse workload and failure conditions.

## File Structure

### Machine Configuration
- **machines.csv**: Defines 10 machines (5 local + 5 cloud) with capacity and cost profiles
  
  **Local Machines (limited capacity):**
  - M1: GPU Compute - capacity=20 (training/inference)
  - M2: CPU Compute - capacity=40 (control logic, feature engineering)
  - M3: High-Memory Analytics - capacity=10 (graphs, large matrices)
  - M4: Secure/Isolated Compute - capacity=5 (gov-only workloads)
  - M5: Data Ingest & Preprocessing - capacity=30 (IO-heavy staging)
  
  **Cloud Machines (infinite capacity, 5x cost):**
  - C1-C5: Cloud equivalents of each type
  - capacity=999999 (effectively unlimited parallel operations)
  - 5x the cost of local machines
  - Use for overflow when local machines are at capacity or during failures

### Job Scenarios

**Primary Scenarios (used in evaluation):**

- **scenario_normal.csv**: Realistic 24-hour workload (16 jobs)
  - Overnight batch: Data ETL, climate training, historical data, performance analytics (t=0-6)
  - Morning: Tax fraud detection, health dashboards (t=6-9)
  - Business hours: Emergency coord, classified intel, traffic monitoring (t=8-14)
  - Afternoon/Evening: Audits, citizen services, border security (t=14-24)
  - Late night: Database backups, log archival (t=20-24)
  - Mix of all priority levels: CRITICAL(2), HIGH(3), MEDIUM(6), LOW(5)
  - Tests realistic daily operations with time-varying workload
  - **80% weight** in multi-scenario evaluation

- **scenario_earthquake.csv**: Normal day + disaster response (26 jobs)
  - **All 16 normal jobs** from scenario_normal.csv (J1-J16)
  - **+ 10 earthquake emergency jobs** at t=10-13 (E1-E10)
  - Emergency jobs: 5 CRITICAL, 4 HIGH, 1 MEDIUM (damage assessment, evacuation, medical)
  - Combined with `earthquake_moderate` failure scenario (M1 -10 capacity, M3 -5 capacity)
  - Tests: "What if earthquake strikes during normal workday?"
  - Heavy resource contention: 26 concurrent jobs + reduced machine capacity
  - **20% weight** in multi-scenario evaluation

### Machine Failure Scenarios

- **machine_failures.csv**: Realistic failure events (Japan context) with **capacity reduction**
  - Machines don't go completely offline - they lose some capacity instead
  - `capacity_reduction`: Amount of capacity lost during failure (e.g., M1 normally has 20, loses 10 → effective capacity of 10)
  - Failures use exponential degradation/recovery model with configurable rates
  - `failure_rate`: Speed of capacity degradation (higher = faster degradation)
  - `recovery_rate`: Speed of capacity restoration (higher = faster recovery)
  
  **Failure scenarios:**
  - **earthquake_moderate**: M1 (capacity -10), M3 (capacity -5) - cooling/memory damage
  - **tsunami_warning**: M4 (capacity -3) - facility access restricted during evacuation
  - **power_outage**: M2 (capacity -20), M5 (capacity -15) - rolling blackout
  - **typhoon_severe**: M2 (capacity -25), M3 (capacity -7) - major cooling failure, water damage
  - **nuclear_alert**: M4 (capacity -2) - security protocol overhead
  
  **Impact:**
  - Machines have reduced capacity during failures (e.g., M1: 20→10 during earthquake)
  - Gradual degradation/recovery using exponential model (not instant on/off)
  - Increases resource contention and potential deadline violations
  - Encourages use of cloud overflow (expensive but reliable)

## Usage

### Python Code
```python
from scenario_loader import create_instance_from_csv, get_available_scenarios

# List available scenarios
scenarios = get_available_scenarios()
print(scenarios)  # {'jobs': [...], 'failures': [...]}

# Load normal day scenario (no failures)
inst = create_instance_from_csv(scenario="normal")

# Load earthquake scenario with failures
inst = create_instance_from_csv(scenario="earthquake", failure_scenario="earthquake_moderate")

# Multi-scenario evaluation (automatic in evaluate.py)
from initial import run_scheduling
result = run_scheduling(
    multi_scenario=True,
    scenarios=["normal", "earthquake"],
    scenario_weights={"normal": 0.8, "earthquake": 0.2},
    failure_scenarios={"earthquake": "earthquake_moderate"}
)
print(f"Weighted score: {result['weighted_score']}")
```

### Evaluation Script
```bash
# Multi-scenario evaluation (default - tests both normal + earthquake)
python evaluate.py --program_path initial.py --results_dir results

# Results include weighted aggregate of both scenarios
# - Normal day: 80% weight
# - Earthquake: 20% weight
```

## Scenario Characteristics

| Scenario      | Jobs | CRITICAL | HIGH | MEDIUM | LOW | Arrival Times | Eval Weight |
|---------------|------|----------|------|--------|-----|---------------|-------------|
| normal        | 16   | 2        | 3    | 6      | 5   | t=0-21        | **80%**     |
| earthquake    | 26   | 7        | 7    | 7      | 5   | t=0-21 + disaster t=10-13 | **20%** |

## Adding New Scenarios

1. Create a new CSV file: `scenario_<name>.csv`
2. Follow the format:
   ```csv
   job_id,description,priority,deadline,delay_weight,arrival_time,operations
   J1,Job description,CRITICAL,6,15.0,0,"Ingest:1:5,GPU:2:8,CPU:1:3"
   ```
3. Operations format: `"<Type>:<Duration>:<Capacity>,<Type>:<Duration>:<Capacity>,..."`
4. Valid types: GPU, CPU, HighMem, Secure, Ingest
5. Valid priorities: CRITICAL, HIGH, MEDIUM, LOW

## Testing Recommendations

1. **Multi-Scenario Evaluation**: Default evaluation tests both scenarios with weighted scoring
2. **Weighted Optimization**: Scheduler must balance cost-efficiency (normal) + disaster response (earthquake)
3. **Individual Testing**: Use scenario_loader to test specific scenarios manually
4. **Stress Testing**: `emergency` + `earthquake_moderate` is the hardest combination
5. **Cost Optimization**: `light_load` is best for testing off-peak scheduling
