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
| **Shock Failure Models** | Realistic shock failures (earthquakes, power outages) with exponential recovery dynamics |
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
5. **Availability:** Operations cannot start before job arrival

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

### Shock Failure Model

The PoC emphasizes **realistic shock failures** that severely impact data center operations. Unlike gradual wear, shocks are sudden capacity loss followed by recovery:

**Shock Failure Patterns:**
- **Instant Shock** ($\text{failure rate} = 1.0$): Capacity drops to zero immediately at failure start time
- **Gradual Onset** ($\text{failure rate} = 0.5$): Capacity degrades exponentially over ~2 hours (e.g., earthquake aftershocks, cascading failures)
- **Exponential Recovery** ($\text{recovery rate} = 0.5$): Capacity restores gradually over ~2 hours (system repairs, redundancy activation)

**Example: Earthquake at t=10**
```
Machine M2 (CPU) under Earthquake Scenario:
t=10:  Earthquake strikes. Capacity: 40 → 0 (instant or gradual)
t=10–12: Capacity reduced: 0 + (40 × e^(-0.5 × Δt))  [recovery begins]
t=12:  Capacity ≈ 10 (50% recovered)
t=14:  Capacity ≈ 25 (62% recovered)
t=16:  Capacity ≈ 36 (90% recovered)
```

**Why this matters:**
- Trivial solutions like "always use cloud" become suboptimal once costs dominate
- Scheduler must predict when local capacity will recover and batch jobs strategically
- Forces genuine optimization: balance holding jobs vs. expensive cloud overflow

### Test Scenarios

Two complementary scenarios evaluate scheduler robustness:

| Scenario | Weight | Characteristics |
|----------|--------|-----------------|
| **Normal Day** | 80% | 12 jobs across 24 hours; realistic cost-driven scheduling |
| **Earthquake Emergency** | 20% | 10 CRITICAL/HIGH priority jobs; shock failures at t=10 on all machines |

**Weighted evaluation strategy:**
- Ensures practical performance on typical operations (80% focus)
- Still handles disasters gracefully without over-optimization (20% focus)
- More realistic than single-scenario testing

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
- Scheduler must decide: use expensive cloud to free resources or hold and wait for recovery?
- This creates genuine optimization tradeoffs beyond static scheduling

### Resource Configuration

**Local Machines (capacity-limited):**
- M1 (GPU): capacity = 20
- M2 (CPU): capacity = 40
- M3 (High-Memory): capacity = 10
- M4 (Secure): capacity = 5
- M5 (Ingest): capacity = 30

Each can run multiple operations in parallel up to capacity limit.

**Cloud Machines (infinite capacity):**
- C1–C5: Cloud equivalents, 5× cost multiplier
- Unlimited parallel operations
- Used for overflow when local machines are congested or failed

### Failure Dynamics

**Failure Onset (capacity reduction during shock):**

$$\text{capacity}(t) = \text{original} - \text{reduction} \times \left(1 - e^{-\text{failure\_rate} \times (t - t_{\text{start}})}\right)$$

**Recovery (capacity restoration after shock):**

$$\text{capacity}(t) = \text{original} - \text{reduction} \times e^{-\text{recovery\_rate} \times (t - t_{\text{end}})}$$

**Parameters:**
- `failure_rate = 1.0`: Instant capacity loss (switch to cloud immediately)
- `failure_rate = 0.5`: Gradual degradation (opportunity to batch jobs before critical loss)
- `recovery_rate = 1.0`: Instant recovery (unlikely in practice)
- `recovery_rate = 0.5`: 2-hour recovery window (scheduler can predict when to resume local use)

This realistic model prevents trivial greedy algorithms from solving the problem.

---

## Scheduler Design

### Decision Logic

The scheduler (baseline or evolved versions) determines:
1. **Which machine** to use (local vs. cloud) for each operation
2. **When** to start each operation (respecting precedence, arrival, capacity constraints)

### Example Job Structure

```
J2: Tax Fraud Detection
├─ Operation 1: Ingest (1 hr, requires M5/C5)
├─ Operation 2: GPU (3 hrs, requires M1/C1)
└─ Operation 3: CPU (1 hr, requires M2/C2)
```

### Scheduling Constraints

1. Operation 2 cannot start before Operation 1 finishes
2. No operation can start before job arrival time
3. Each operation requires a specific machine type
4. Total concurrent operations ≤ machine capacity
5. Operations are non-preemptive (cannot be paused/resumed)

---

## Evolutionary Optimization (ShinkaEvolve)

**ShinkaEvolve** is an LLM-driven evolutionary framework that:
- **Generates** candidate scheduling strategies as executable Python code
- **Evaluates** each strategy on both scenarios simultaneously using the weighted fitness function
- **Mutates** high-performing solutions to discover novel improvements
- **Converges** on robust schedulers that balance cost, latency, and resilience

The evolutionary loop discovers scheduling policies that optimize competing objectives (minimize cost while meeting deadlines under failures) in ways classical heuristics struggle to achieve.

---

## File Structure

```
PoC-DataCenter-EvolutionaryAlgorithm/
├── data/                              # Experiment configurations
│   ├── machines.csv                   # Machine specs and capacities
│   ├── machine_failures.csv           # Shock failure scenarios
│   ├── scenario_baseline.csv          # Normal day jobs
│   └── scenario_earthquake.csv        # Emergency scenario
├── initial.py                         # Baseline scheduler (FIFO + priority)
├── evaluate.py                        # Evaluation engine
├── run_evo.py                         # Evolutionary optimization loop
├── scenario_loader.py                 # Configuration and data loading
└── results_scheduling/                # Evolution outputs
    └── gen_N/
        ├── main.py                    # Evolved scheduler code
        └── results/
            ├── metrics.json           # Performance scores
            └── extra.json             # Detailed schedule traces
```

---

## Results & Performance

**See `figs/` directory for detailed visualizations:**

- **Shock Failure Scenarios**: Visualization of different failure modes and their exponential recovery dynamics
- **Baseline Comparison**: Gantt charts showing MIP solutions vs. evolved schedulers under earthquake scenario
- **Convergence**: ShinkaEvolve outperforms MIP baseline after ~70 generations
- **Stability Analysis**: Solution convergence and near-stagnation post-optimization

---

## Usage

### Running the Baseline Scheduler

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

1. **Shock failures dominate design**: Realistic exponential failure/recovery models rule out trivial greedy solutions
2. **Weighted multi-scenario evaluation**: Prevents overfitting to rare disasters while maintaining practical baseline performance
3. **Evolutionary discovery**: Finds scheduling policies balancing cost, latency, and resilience better than hand-crafted heuristics
4. **Hybrid cloud strategy emerges**: Use local when cheap and available, cloud when local capacity is shocked or too expensive
5. **Queuing dynamics matter**: Staggered job arrivals create genuine optimization tradeoffs that static models miss

---

## References

- SakanaAI ShinkaEvolve: [GitHub](https://github.com/sakanaai/shinkaevolve)
- Job-shop scheduling: classical NP-hard optimization with modern evolutionary approaches
- Data center cost models: based on real time-of-use pricing patterns
- Fault-tolerant scheduling: exponential recovery models common in systems literature

---

## License

MIT
