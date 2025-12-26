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

- **scenario_baseline.csv**: Standard workload (8 jobs)
  - Mix of CRITICAL, HIGH, MEDIUM, and LOW priority jobs
  - Realistic government data processing tasks
  - Moderate resource utilization

- **scenario_peak_load.csv**: Heavy workload (12 jobs)
  - Additional jobs for traffic monitoring, financial audit, border security
  - Tests algorithm under resource contention
  - More CRITICAL and HIGH priority jobs

- **scenario_emergency.csv**: Disaster response (10 jobs)
  - Japan earthquake/tsunami scenario
  - All CRITICAL/HIGH priority jobs with tight deadlines
  - Heavy use of GPU and HighMem for damage assessment and modeling
  - Tests deadline-driven scheduling under pressure

- **scenario_light_load.csv**: Maintenance period (5 jobs)
  - Low priority batch jobs
  - Relaxed deadlines
  - Opportunity to test off-peak cost optimization

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

# Load baseline scenario (no failures)
inst = create_instance_from_csv("baseline")

# Load emergency scenario with earthquake
inst = create_instance_from_csv("emergency", "earthquake_moderate")

# Load peak load with power outage
inst = create_instance_from_csv("peak_load", "power_outage")
```

### Evaluation Script
```bash
# Test baseline scenario
python evaluate.py --scenario baseline

# Test with failures
python evaluate.py --scenario emergency --failures earthquake_moderate
```

## Scenario Characteristics

| Scenario      | Jobs | CRITICAL | HIGH | MEDIUM | LOW | Avg Deadline |
|---------------|------|----------|------|--------|-----|--------------|
| baseline      | 8    | 2        | 2    | 3      | 1   | 13.0         |
| peak_load     | 12   | 3        | 3    | 5      | 1   | 14.5         |
| emergency     | 10   | 5        | 4    | 1      | 0   | 6.7          |
| light_load    | 5    | 0        | 0    | 1      | 4   | 21.4         |

## Adding New Scenarios

1. Create a new CSV file: `scenario_<name>.csv`
2. Follow the format:
   ```csv
   job_id,description,priority,deadline,delay_weight,operations
   J1,Job description,CRITICAL,6,15.0,"Ingest:1,GPU:2,CPU:1"
   ```
3. Operations format: `"<Type>:<Duration>,<Type>:<Duration>,..."`
4. Valid types: GPU, CPU, HighMem, Secure, Ingest
5. Valid priorities: CRITICAL, HIGH, MEDIUM, LOW

## Testing Recommendations

1. **Algorithm Development**: Start with `baseline` scenario
2. **Robustness Testing**: Test all job scenarios without failures
3. **Failure Resilience**: Add failure scenarios incrementally
4. **Stress Testing**: `emergency` + `earthquake_moderate` is the hardest combination
5. **Cost Optimization**: `light_load` is best for testing off-peak scheduling
