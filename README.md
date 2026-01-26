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

### Scenarios

Two primary scenarios test the scheduler under different conditions:

- Normal Day
- Earthquake Emergency

Both are tested with Offline (Future information) and Online (No Future information) Approaches

**Important:** Scenarios are **predetermined**, loaded from CSV files in `data/`. The earthquake doesn't happen randomly—it's a scripted scenario that always occurs at t=10 in the earthquake scenario.

### Job Arrival

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

### Resources

**Local Machines (limited capacity):**
- **M1**: 1x GPU machine (capacity=20) 
- **M2**: 1x CPU machine (capacity=40) 
- **M3**: 1x High-Memory machine (capacity=10) 
- **M4**: 1x Secure machine (capacity=5) 
- **M5**: 1x Ingest machine (capacity=30) 

Each local machine can run multiple operations simultaneously up to its capacity limit.

**Cloud Machines (infinite capacity, 5x cost):**
- **C1-C5**: Cloud equivalents of each type
- Can run unlimited parallel operations
- Cost 5x more than local machines
- Used for overflow when local machines are busy or failed

##### Costs

Server costs change by hour to reflect demand-based pricing:

| Time Period | Example: GPU (M1) | Example: CPU (M2) | Reason |
|-------------|-------------------|-------------------|--------|
| 0-5 (night) | ¥80/hr | ¥20/hr | Off-peak |
| 6-9 (morning) | ¥120/hr | ¥45/hr | Morning rush |
| 10-16 (midday) | ¥100/hr | ¥30/hr | Business hours |
| 17-21 (evening) | ¥140/hr | ¥55/hr | **Peak hours** |
| 22-23 (late) | ¥90/hr | ¥25/hr | Winding down |

**Cloud costs are 5x higher** at all times.

- Delay non-urgent jobs to off-peak hours (if deadline allows)
- Critical jobs may need expensive evening slots
- Cloud overflow is very expensive but sometimes necessary

##### Failures

Failures are **predetermined** from `data/machine_failures.csv` and random:

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

### **How Jobs Are Scheduled**

The scheduler (in `initial.py` or evolved versions) determines:
1. **Which machine** to use for each operation (local vs cloud)
2. **When** to start each operation (considering arrival, precedence, capacity)

**Job Structure Example:**
```
J2: Tax fraud detection
├─ Operation 1: Ingest (1 hour, needs M5/C5)
├─ Operation 2: GPU (3 hours, needs M1/C1)
└─ Operation 3: CPU (1 hour, needs M2/C2)
```

**Constraints the scheduler must satisfy:**
1. Op2 can't start before Op1 finishes
2. Can't start before job arrives
3. Each operation needs specific machine type
4. Total concurrent operations can't exceed machine capacity
5. Once started, operations run to completion

### **Evaluation**: Multi-Scenario Weighted Evaluation

The algorithm is tested on **two scenarios simultaneously** with weighted scoring:

**Scenarios:**
- **Normal Day (80% weight)**: 12 jobs spread across 24 hours, realistic daily operations
- **Earthquake Emergency (20% weight)**: 10 CRITICAL/HIGH priority jobs + earthquake_moderate failures

**Weighted Fitness Function:**
```python
total_score = 0.8 × normal_score + 0.2 × earthquake_score

where each scenario_score = compute_cost × weight + delay_penalty × weight
```

**Why weighted?**
- Ensures good performance on typical workloads (80% focus)  
- Still handles disaster scenarios gracefully (20% focus)
- Prevents over-optimization for rare emergencies
- More realistic evaluation than single-scenario testing

The evolutionary algorithm evolves schedulers that balance both scenarios.

---

## File Structure

```
PoC-JPNGOV-DataCenter/
├── data/ # experiment configs
│   ├── machines.csv              
│   ├── machine_failures.csv      
│   ├── scenario_baseline.csv     
│   ├── scenario_earthquake.csv   
├── initial.py                    # Baseline scheduler (FIFO + priority)
├── evaluate.py                   # Evaluation
├── run_evo.py                    # Evolutionary optimization
├── scenario_loader.py            # Helper
└── results_scheduling/           # Evolution outputs (for each test)
    └── gen_N/
        ├── main.py              # Evolved scheduler code
        └── results/
            ├── metrics.json     # Performance scores
            └── extra.json       # Detailed schedule
```

---

---


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

## AlphaEvolve / ShinkaEvolve Approach

**AlphaEvolve / ShinkaEvolve** is a population-based evolutionary optimization framework tailored for
