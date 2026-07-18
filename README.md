# Cost-Aware Job-Shop Scheduling for Data Centers using Evolutionary Algorithms

## Overview

This proof-of-concept (PoC) demonstrates **LLM-driven evolutionary optimization** to solve dynamic job-shop scheduling in data centers. The system uses **SakanaAI's ShinkaEvolve** to evolve scheduling policies that minimize operational costs while managing:

- **Time-varying costs** (cheaper at night, expensive during peak hours)
- **Stochastic failures** (earthquakes, power outages, hardware degradation)
- **Job deadlines** with priority-based penalties
- **Cloud overflow** (expensive backup capacity)

**Goal:** Minimize total cost = compute cost + delay penalties, while maintaining robustness across diverse operational scenarios.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Scenario Optimization** | Simultaneously optimizes for normal operations (80%) and disaster scenarios (20%) |
| **Exponential Failure Models** | Realistic gradual capacity degradation during failures and recovery |
| **Time-Varying Costs** | Hourly pricing reflects real demand-based infrastructure costs |
| **Evolutionary Learning** | LLM-driven iteration discovers novel scheduling strategies beyond classical baselines |
| **Hybrid Cloud Strategy** | Automatically decides when to use expensive cloud resources vs. waiting |

---

## Problem Formulation

### Input Elements

**Jobs (Workloads):**
- Set of jobs $J = \{1, \dots, n\}$
- Each job $j$ contains an ordered sequence of operations $O_j = (o_{j1}, o_{j2}, \dots, o_{jk_j})$
- Each operation has: required server type, processing duration $p_{ji}$

**Servers (Machines):**
- Local machines: M1 (GPU), M2 (CPU), M3 (High-Memory), M4 (Secure), M5 (Ingest)
- Cloud equivalents: C1–C5 (infinite capacity, 5× cost)
- Each machine has time-varying costs $c_m(t)$

**Cost Model:**
- Time-varying pricing by hour reflecting demand patterns
- Off-peak (night): ¥20–¥80/hr depending on machine type
- Peak (evening): ¥55–¥140/hr
- Cloud costs: 5× higher at all times

**Deadlines & Penalties:**
- Each job has a deadline $d_j$ and delay penalty weight $w_j$

### Constraints

1. **Job precedence:** Operations must execute in order
2. **Machine capacity:** Each machine processes one operation at a time (local); unlimited parallel (cloud)
3. **Machine compatibility:** Operations can only run on compatible machine types
4. **Non-preemption:** Operations run to completion without interruption
5. **Availability:** Operations can't start before job arrival

### Objective Function

$$\min \left(
\alpha \sum_{j,i} \sum_{t=s_{ji}}^{s_{ji}+p_{ji}-1} c_{m(j,i)}(t)
\;+\;
\beta \sum_j w_j \cdot \max(0, C_j - d_j)
\right)$$

Where:
- $s_{ji}$ = start time of operation $o_{ji}$
- $m(j,i)$ = assigned server
- $C_j$ = completion time of job $j$
- $\alpha, \beta$ = policy weights balancing cost vs. delay

---

## Simulation Architecture

### Scenarios

Two complementary scenarios test scheduler robustness:

| Scenario | Weight | Characteristics |
|----------|--------|-----------------|
| **Normal Day** | 80% | 12 jobs spread across 24 hours; realistic workload mix |
| **Earthquake Emergency** | 20% | 10 CRITICAL/HIGH priority jobs + shock failures at t=10 |

The weighted evaluation prevents over-optimization for rare disasters while maintaining strong baseline performance.

### Job Arrival & Queuing Dynamics

Jobs do **not** arrive simultaneously. Each has an `arrival_time`:

```
Earthquake Scenario Timeline:
t=10:  Earthquake strikes → J1, J2, J3 arrive (damage assessment, shelters, evacuation)
t=11:  → J4, J5, J9 arrive (infrastructure, medical, power grid)
t=12:  → J6, J7 arrive (communications, casualty estimates)
t=13:  → J8, J10 arrive (supply chain, inspections)
```

**Queuing effects:**
- Early-arriving jobs may block resources needed by high-priority later jobs
- Scheduler must decide: use expensive cloud to free resources or wait?
- This creates genuine optimization tradeoffs beyond static scheduling

### Resource Configuration

**Local Machines (capacity-limited):**
- M1 (GPU): capacity=20
- M2 (CPU): capacity=40
- M3 (High-Memory): capacity=10
- M4 (Secure): capacity=5
- M5 (Ingest): capacity=30

Each can run multiple operations in parallel up to capacity.

**Cloud Machines (infinite capacity):**
- C1–C5: Cloud equivalents, 5× cost multiplier
- Unlimited parallel operations
- Used for overflow when local machines are congested or failed

### Failure Models

Failures are **predetermined** but follow realistic exponential dynamics:

**Failure Onset (gradual capacity reduction):**
$$\text{reduction} = \text{max\_reduction} \times (1 - e^{-\text{failure\_rate} \times t})$$

- `failure_rate = 1.0`: Instant reduction at start
- `failure_rate = 0.5`: Gradual reduction over ~2 hours

**Recovery (gradual restoration):**
$$\text{capacity} = \text{reduced\_capacity} + (\text{max\_reduction} - \text{reduced\_capacity}) \times (1 - e^{-\text{recovery\_rate} \times t_{\text{since\_end}}})$$

This realistic failure model prevents trivial "just switch to cloud" strategies.

---

## Scheduler Design

### Decision Logic

The scheduler (initial baseline or evolved versions) determines:
1. **Which machine** to use (local vs. cloud) for each operation
2. **When** to start (respecting constraints)

### Example Job Structure

```
J2: Tax fraud detection
├─ Operation 1: Ingest (1 hr, needs M5/C5)
├─ Operation 2: GPU (3 hrs, needs M1/C1)
└─ Operation 3: CPU (1 hr, needs M2/C2)
```

### Scheduling Constraints

1. Op2 can't start before Op1 finishes
2. Can't start before job arrival time
3. Each operation needs specific machine type
4. Total concurrent operations ≤ machine capacity
5. Operations are non-preemptive

---

## Evolutionary Optimization (ShinkaEvolve)

**ShinkaEvolve** is an LLM-driven evolutionary framework that:
- **Generates** candidate scheduling strategies as executable Python code
- **Evaluates** each strategy on both scenarios simultaneously
- **Mutates** high-performing solutions to discover novel improvements
- **Converges** on robust schedulers that handle both normal and emergency workloads

The evolutionary loop discovers scheduling policies that balance cost, latency, and robustness in ways classical heuristics miss.

---

## File Structure

```
PoC-DataCenter-EvolutionaryAlgorithm/
├── data/                              # Experiment configurations
│   ├── machines.csv                   # Machine specs & capacities
│   ├── machine_failures.csv           # Failure scenarios
│   ├── scenario_baseline.csv          # Normal day jobs
│   └── scenario_earthquake.csv        # Emergency scenario
├── initial.py                         # Baseline scheduler (FIFO + priority)
├── evaluate.py                        # Evaluation engine
├── run_evo.py                         # Evolutionary optimization loop
├── scenario_loader.py                 # Configuration & data loading
└── results_scheduling/                # Evolution outputs
    └── gen_N/
        ├── main.py                    # Evolved scheduler code
        └── results/
            ├── metrics.json           # Performance scores
            └── extra.json             # Detailed schedule traces
```

---

## Results & Performance

### Shock Failure Scenarios

![Shock Failure Scenarios](figs/Shock%20Failure%20Scenarios.png)

Visualization of different failure modes and their impact on resource availability and system performance.

### MIP Baseline vs. Evolutionary Solutions

![MIP Solution under Earthquake Scenario](figs/MIP%20Solution%20under%20Earthquake%20Scenario%2C%20Left%3A%20Gantt%20chart%20of%20job%20schedules%2C%20including%20cloud%20usage%20%28C1%2C%20C3%2C%29.png)

Left: Gantt chart showing job schedules and cloud resource utilization (C1, C3, C4) under earthquake scenario  
Right: Resource capacity over time, highlighting failure periods and recovery dynamics

### Evolutionary Algorithm Convergence

![ShinkaEvolve Outperforms MIP Baseline](figs/ShinkaEvolve%20Outperforms%20MIP%20Baseline%20after%20~70%20Generations.png)

ShinkaEvolve discovers solutions outperforming MIP baseline after ~70 generations, demonstrating effectiveness of LLM-driven code generation for scheduling optimization.

### Solution Stability

![Convergence Analysis](figs/ShinkaEvolve%27s%20Discovered%20Solution%20is%20Nearly%20Stagnant%20throughout%20Generations.png)

Solution quality stabilizes and plateaus across generations after reaching near-optimal configuration, indicating convergence to a robust scheduling policy.

---

## Usage

### Running the Baseline

```bash
python evaluate.py --scheduler initial
```

### Running Evolutionary Optimization

```bash
python run_evo.py --generations 100 --population 20
```

### Analyzing Results

```bash
python evaluate.py --scheduler results_scheduling/gen_N/main.py
```

---

## Key Insights

1. **Weighted multi-scenario evaluation** prevents overfitting to rare disasters while maintaining practical performance
2. **Exponential failure models** create realistic constraints that rule out trivial "always use cloud" strategies
3. **Evolutionary discovery** finds scheduling policies that balance competing objectives better than hand-crafted heuristics
4. **Hybrid cloud strategy** emerges naturally through optimization: use local when cheap, cloud when beneficial despite cost

---

## References

- SakanaAI ShinkaEvolve: [GitHub](https://github.com/sakanaai/shinkaevolve)
- Job-shop scheduling: classical NP-hard optimization with modern evolutionary approaches
- Data center cost models: based on real time-of-use pricing patterns

---

## License

MIT

