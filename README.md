# Cost-Aware Job-Shop Scheduling for Government Data Centers

## 1. Problem Definition

We consider a **government data center workload scheduling problem** in which a set of computational workloads must be executed on limited server resources over a finite time horizon. Each workload consists of a sequence of processing steps that must be executed in a fixed order. Server costs vary by time of day due to demand-based pricing (similar to cloud spot pricing).

The objective is to determine **when and where** each workload step is executed so as to balance:
- **Server cost** (direct compute cost per hour × time-varying price)
- **Service delay penalties** (missing deadlines for public services)

This problem is motivated by government-operated data centers where:
- workloads are mission-critical,
- compute costs and resource efficiency matter,
- resilience and transparency are required.

---

## 2. Formal Problem Description

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
