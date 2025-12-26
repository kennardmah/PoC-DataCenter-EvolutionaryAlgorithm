# Cost-Aware Job-Shop Scheduling for Government Data Centers

## Overview

This simulation models a **government data center** that must schedule computational workloads across limited server resources while managing:
- **Time-varying costs** (cheaper at night, expensive during peak hours)
- **Machine failures** (earthquakes, power outages, etc.)
- **Job deadlines** with priority-based penalties
- **Cloud overflow** (expensive backup capacity)

The goal is to minimize total cost = compute cost + delay penalties.

---

## How the Simulation Works: Step-by-Step

### 1. **Scenario Selection**

Two primary scenarios test the scheduler under different conditions:

- **Normal Day** (16 jobs, 24-hour realistic workload)
  - **Overnight batch** (t=0-6): Census ETL, climate training, historical data, performance analytics
  - **Morning** (t=6-9): Tax fraud detection, health dashboards
  - **Business hours** (t=8-14): Emergency coord, classified intel, traffic monitoring
  - **Afternoon/Evening** (t=14-24): Audits, citizen services, border security
  - **Late night** (t=20-24): Database backups, log archival
  - Mix of priorities: 2 CRITICAL, 3 HIGH, 6 MEDIUM, 5 LOW
  - **80% evaluation weight** - optimizes for daily operations
  
- **Earthquake Emergency** (26 jobs = normal + disaster)
  - **All 16 normal day jobs continue running**
  - **Earthquake strikes at t=10** - 10 emergency jobs flood in (t=10-13)
  - Emergency jobs: 5 CRITICAL, 4 HIGH, 1 MEDIUM (damage assessment, evacuation, medical)
  - Machine failures: M1 capacity -10, M3 capacity -5 (earthquake_moderate)
  - Tests: "What if earthquake happens during normal workday?"
  - **20% evaluation weight** - ensures disaster resilience

**Important:** Scenarios are **predetermined**, loaded from CSV files in `data/`. The earthquake doesn't happen randomly—it's a scripted scenario that always occurs at t=10 in the earthquake scenario.

### 2. **Job Arrival Process**

Jobs **do NOT all arrive at once**. Each job has an `arrival_time`:

```
Earthquake Scenario Example:
t=10: Earthquake strikes! → J1, J2, J3 arrive (damage assessment, shelters, evacuation)
t=11: → J4, J5, J9 arrive (infrastructure, medical, power grid)
t=12: → J6, J7 arrive (communications, casualty estimates)
t=13: → J8, J10 arrive (supply chain, inspections)
```

**Queuing dynamics:**
- Jobs can't be scheduled before they arrive
- Early-arriving jobs may block resources needed by later high-priority jobs
- Scheduler must decide: use expensive cloud to free local resources, or wait?

### 3. **Machine Configuration**

**Local Machines (limited capacity):**
- **M1**: 1x GPU machine (capacity=20) - ML training/inference
- **M2**: 1x CPU machine (capacity=40) - Data processing
- **M3**: 1x High-Memory machine (capacity=10) - Analytics
- **M4**: 1x Secure machine (capacity=5) - Classified workloads
- **M5**: 1x Ingest machine (capacity=30) - Data staging

Each local machine can run **multiple operations simultaneously** up to its capacity limit.

**Cloud Machines (infinite capacity, 5x cost):**
- **C1-C5**: Cloud equivalents of each type
- Can run unlimited parallel operations
- Cost 5x more than local machines
- Used for overflow when local machines are busy or failed

### 4. **Time-Varying Costs**

Server costs change by hour to reflect demand-based pricing:

| Time Period | Example: GPU (M1) | Example: CPU (M2) | Reason |
|-------------|-------------------|-------------------|--------|
| 0-5 (night) | ¥80/hr | ¥20/hr | Off-peak |
| 6-9 (morning) | ¥120/hr | ¥45/hr | Morning rush |
| 10-16 (midday) | ¥100/hr | ¥30/hr | Business hours |
| 17-21 (evening) | ¥140/hr | ¥55/hr | **Peak hours** |
| 22-23 (late) | ¥90/hr | ¥25/hr | Winding down |

**Cloud costs are 5x higher** at all times.

**Strategy implications:**
- Delay non-urgent jobs to off-peak hours (if deadline allows)
- Critical jobs may need expensive evening slots
- Cloud overflow is very expensive but sometimes necessary

### 5. **Machine Failures (Optional)**

Failures are **predetermined** from `data/machine_failures.csv`, not random:

**Failure Scenarios:**
- `earthquake_moderate`: M1 (capacity -10), M3 (capacity -5) at t=8-12/14 - cooling/memory damage
- `tsunami_warning`: M4 (capacity -3) at t=5-10 - facility access restricted
- `power_outage`: M2 (capacity -20), M5 (capacity -15) at t=15-18 - rolling blackout
- `typhoon_severe`: M2 (capacity -25), M3 (capacity -7) at t=10-20/22 - major cooling failure
- `nuclear_alert`: M4 (capacity -2) at t=0-8 - security protocol overhead

**Exponential Failure Model:**
Failures don't happen instantly—they follow exponential curves:

1. **Failure onset** (gradual capacity reduction):
   - `failure_rate = 1.0`: Instant capacity reduction at start time
   - `failure_rate = 0.5`: Gradual capacity reduction over ~2 hours
   - Capacity reduces: `max_reduction * (1 - e^(-failure_rate * time))`

2. **Recovery** (gradual restoration):
   - `recovery_rate = 1.0`: Instant recovery at end time
   - `recovery_rate = 0.5`: Gradual restoration over ~2 hours
   - Capacity restores: `max_reduction * e^(-recovery_rate * time_since_end)`

**Example:** Power outage on M2 (capacity=40, reduction=20) at t=15-18:
- t=14: Machine fully operational (capacity=40)
- t=15: **Instant failure** (capacity=20) - lost 20 capacity
- t=15-18: Machine degraded (capacity=20)
- t=18: **Instant recovery** starts (capacity=20)
- t=19: Fully operational again (capacity=40)

**Note:** Most scenarios use instant failure/recovery (rate ≥0.8), but model supports gradual degradation.

### 6. **How Jobs Are Scheduled**

The scheduler (in `initial.py` or evolved versions) determines:
1. **Which machine** to use for each operation (local vs cloud)
2. **When** to start each operation (considering arrival, precedence, capacity)

**Job Structure Example:**
```
J2: Tax fraud detection
├─ Operation 1: Ingest (1 hour, needs M5/C5) - load data
├─ Operation 2: GPU (3 hours, needs M1/C1) - train model
└─ Operation 3: CPU (1 hour, needs M2/C2) - generate report
```

**Constraints the scheduler must satisfy:**
1. ✅ **Precedence**: Op2 can't start before Op1 finishes
2. ✅ **Arrival**: Can't start before job arrives
3. ✅ **Machine type**: Each operation needs specific machine type
4. ✅ **Capacity**: Total concurrent operations can't exceed machine capacity
5. ✅ **No preemption**: Once started, operations run to completion

**Scheduling strategies** (evolved by AI):
- **Priority-based**: Schedule CRITICAL jobs first
- **Cost-aware**: Shift operations to off-peak hours when possible
- **Lookahead**: Reserve local machines for high-priority jobs, use cloud for others
- **Slack-based**: Calculate deadline - remaining work, schedule tightest jobs first

### 7. **How Jobs Are Completed**

Step-by-step execution:

```
t=6: J1 arrives (census processing)
  → Op1 (Ingest, 2hr) scheduled on M5a starting t=6
  → Op1 runs t=6-8, cost = 2 × ¥30 = ¥60

t=8: Op1 completes
  → Op2 (HighMem, 3hr) scheduled on M3a starting t=8
  → Op2 runs t=8-11, cost = 3 × ¥95 = ¥285

t=11: Op2 completes
  → Op3 (CPU, 2hr) scheduled on M2a starting t=11
  → Op3 runs t=11-13, cost = 2 × ¥30 = ¥60

t=13: Op3 completes → J1 FINISHED
  → Completion time: 13
  → Deadline: 18
  → Delay: max(0, 13-18) = 0 → No penalty!
  → Total cost for J1: ¥405
```

**If machine fails during execution:**
```
t=8: Op2 starts on M3a
t=10: EARTHQUAKE! M3a fails (capacity drops to 0)
  → Op2 is stuck, can't make progress
  → Either: wait for recovery (wastes time, may miss deadline)
  → Or: reschedule to C3 (cloud, 5x cost but available)
```

### 8. **Evaluation and Scoring**

Each schedule is evaluated on:

**Cost Components:**
```python
compute_cost = sum of (operation_duration × machine_cost_per_hour)
delay_penalty = sum of (delay × job_priority_weight)
  where delay = max(0, completion_time - deadline)

total_score = 1.0 × compute_cost + 5.0 × delay_penalty
```

**Validation Checks:**
- ✅ All jobs scheduled with correct number of operations
- ✅ Operations use correct machine types
- ✅ No job starts before arrival time
- ✅ Precedence constraints satisfied (ops in order)
- ✅ No machine capacity violations (total concurrent capacity ≤ machine limit)
- ✅ No overlapping operations on same machine

**Penalty for violations:**
- Invalid schedules get +10,000 per violation
- This forces the evolutionary algorithm to find valid solutions

### 9. **Multi-Scenario Weighted Evaluation**

The algorithm is tested on **two scenarios simultaneously** with weighted scoring:

**Scenarios:**
- **Normal Day (80% weight)**: 12 jobs spread across 24 hours, realistic daily operations
- **Earthquake Emergency (20% weight)**: 10 CRITICAL/HIGH priority jobs + earthquake_moderate failures

**Weighted Fitness Function:**
```python
total_score = 0.8 × normal_score + 0.2 × earthquake_score

where each scenario_score = compute_cost × weight + delay_penalty × weight + violations × 10000
```

**Why weighted?**
✅ Ensures good performance on typical workloads (80% focus)  
✅ Still handles disaster scenarios gracefully (20% focus)  
✅ Prevents over-optimization for rare emergencies  
✅ More realistic evaluation than single-scenario testing

The evolutionary algorithm evolves schedulers that balance both scenarios.

---

## Quick Start

### 1. Install Dependencies
```bash
uv sync
```

### 2. Run Evaluation (Single Test)
```bash
uv run python evaluate.py --program_path initial.py --results_dir results
```

### 3. Run Evolution (AI-Driven Optimization)
```bash
uv run python run_evo.py
```

This will:
1. Start with the baseline scheduler (`initial.py`)
2. Use AI to propose improvements (different scheduling strategies)
3. Test each variant on the baseline scenario
4. Keep the best performers
5. Generate new variants from successful schedulers
6. Repeat for 400 generations

Results saved to `results_scheduling/gen_N/`

---

## File Structure

```
PoC-JPNGOV-DataCenter/
├── data/
│   ├── machines.csv              # Machine specs & costs
│   ├── machine_failures.csv      # Failure scenarios
│   ├── scenario_baseline.csv     # 8 jobs, normal workday
│   ├── scenario_peak_load.csv    # 12 jobs, rush hours
│   ├── scenario_emergency.csv    # 10 jobs, earthquake response
│   └── scenario_light_load.csv   # 5 jobs, overnight batch
├── initial.py                    # Baseline scheduler (FIFO + priority)
├── evaluate.py                   # Evaluation harness
├── run_evo.py                    # Evolutionary optimization
├── scenario_loader.py            # CSV data loading
└── results_scheduling/           # Evolution outputs
    └── gen_N/
        ├── main.py              # Evolved scheduler code
        └── results/
            ├── metrics.json     # Performance scores
            └── extra.json       # Detailed schedule
```

---

---

## Technical Details

### Problem Formulation

### Inputs

- **Jobs (Workloads)**  
  A set of jobs \( J = \{1, \dots, n\} \).  
  Each job \( j \in J \) consists of an ordered sequence of operations:
  \[
  O_j = (o_{j1}, o_{j2}, \dots, o_{jk_j})
  \]

- **Operations**
  Each operation \( o_{ji} \) has:
  - a required server type
  - a processing time (duration) \( p_{ji} \)

- **Servers (Machines)**  
  A set of servers \( M = \{1, \dots, m\} \), each with:
  - a resource type (CPU, GPU, Storage)
  - capacity 1 (can process only one operation at a time)
  - time-varying cost per hour \( c_m(t) \)

- **Cost Model**
  - Time horizon \( T \) divided into discrete time slots (hours)
  - Server cost \( c_m(t) \) for each server \( m \) at time \( t \)
  - Cost varies by demand: off-peak (cheap), midday (moderate), peak hours (expensive)

- **Deadlines and Penalties**
  - Each job \( j \) has a deadline \( d_j \)
  - A delay penalty weight \( w_j \)

---

### Constraints

1. **Job precedence constraints**  
   Operations of a job must be executed in the given order.

2. **Machine capacity constraints**  
   Each server can process at most one operation at a time.

3. **Machine compatibility constraints**  
   An operation can only be executed on servers of the required type.

4. **Non-preemption**  
   Once started, an operation must run to completion without interruption.

---

### Objective Function

Minimize a weighted sum of server cost and delay penalties:

\[
\min \left(
\alpha \sum_{j,i} \sum_{t=s_{ji}}^{s_{ji}+p_{ji}-1} c_{m(j,i)}(t)
\;+\;
\beta \sum_j w_j \cdot \max(0, C_j - d_j)
\right)
\]

Where:
- \( s_{ji} \) is the start time of operation \( o_{ji} \)
- \( m(j,i) \) is the server assigned to operation \( o_{ji} \)
- \( c_m(t) \) is the cost of server \( m \) at time \( t \)
- \( C_j \) is the completion time of job \( j \)
- \( \alpha, \beta \) are policy weights

---

## 3. Computational Complexity

This problem is a variant of the **Job-Shop Scheduling Problem (JSP)**:

- If server costs are constant and deadlines are ignored, the problem reduces to minimizing makespan in a classic JSP.
- The classic JSP is **NP-hard**.
- Adding time-varying costs and weighted deadlines strictly increases complexity.

**Therefore, this problem is NP-hard in the general case.**

More precisely, it belongs to the class of:
- **Cost-Aware Job-Shop Scheduling**
- **Multi-objective NP-hard combinatorial optimization problems**
- **Time-dependent resource-constrained scheduling problems**

---

## 4. Optimal Solutions and Tractability

### With Complete Information

If all inputs are known in advance (job specifications, deadlines, server costs), the problem can be solved to **provable optimality** using:

1. **Mixed Integer Linear Programming (MILP)** - formulate constraints and objective, solve with commercial solvers
2. **Constraint Programming (CP-SAT)** - Google OR-Tools CP-SAT solver
3. **Branch and Bound** with intelligent pruning strategies

### Instance Size Matters

- **Small instances** (like this demo: 6 jobs, 4 servers, 24 hours): Exact methods are **tractable** and can find optimal solutions in seconds to minutes
- **Large-scale instances** (100s of jobs, 1000s of operations): Exact methods become **computationally infeasible**

The number of feasible schedules grows combinatorially with:
- the number of jobs,
- the number of operations per job,
- the number of shared servers.

### Why Use Evolutionary Methods?

For this demonstration:
1. **Educational value**: Shows how evolutionary algorithms work on well-defined problems
2. **Scalability**: Evolutionary methods scale better to large instances
3. **Flexibility**: Easy to modify objectives and constraints without reformulating
4. **Benchmarking**: Can compare evolved solutions against optimal (from CP-SAT)

For production systems with larger instances, evolutionary/heuristic methods become necessary.

---

## 5. AlphaEvolve / ShinkaEvolve Approach

### Conceptual Overview

**AlphaEvolve / ShinkaEvolve** is a population-based evolutionary optimization framework tailored for
