# Energy-Aware Job-Shop Scheduling for Government Data Centers

## 1. Problem Definition

We consider a **government data center workload scheduling problem** in which a set of computational workloads must be executed on limited server resources over a finite time horizon. Each workload consists of a sequence of processing steps that must be executed in a fixed order. Servers consume electricity while processing workloads, and electricity prices vary over time.

The objective is to determine **when and where** each workload step is executed so as to balance:
- **Energy cost** (electricity consumption × time-varying price)
- **Service delay penalties** (missing deadlines for public services)

This problem is motivated by government-operated data centers where:
- workloads are mission-critical,
- energy costs and peak power usage matter,
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
  - a processing time \( p_{ji} \)
  - an energy consumption rate \( e_{ji} \)

- **Servers (Machines)**  
  A set of servers \( M = \{1, \dots, m\} \), each with:
  - a resource type
  - capacity 1 (can process only one operation at a time)
  - fixed power consumption rate

- **Energy Model**
  - Time horizon \( T \) divided into discrete time slots
  - Electricity price \( c(t) \) for each time slot \( t \)

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

Minimize a weighted sum of energy cost and delay penalties:

\[
\min \left(
\alpha \sum_{j,i} \int_{s_{ji}}^{s_{ji}+p_{ji}} e_{ji} \cdot c(t)\,dt
\;+\;
\beta \sum_j w_j \cdot \max(0, C_j - d_j)
\right)
\]

Where:
- \( s_{ji} \) is the start time of operation \( o_{ji} \)
- \( C_j \) is the completion time of job \( j \)
- \( \alpha, \beta \) are policy weights

---

## 3. Type of NP-Hard Problem

This problem is a variant of the **Job-Shop Scheduling Problem (JSP)**:

- If electricity prices are constant and deadlines are ignored, the problem reduces to minimizing makespan in a classic JSP.
- The classic JSP is **NP-hard**.
- Adding time-varying energy costs and deadlines strictly increases complexity.

**Therefore, this problem is NP-hard.**

More precisely, it belongs to the class of:
- **Energy-Aware Job-Shop Scheduling**
- **Multi-objective NP-hard combinatorial optimization problems**
- **Dynamic resource-constrained scheduling problems**

---

## 4. Why Exact Optimization Is Infeasible

The number of feasible schedules grows combinatorially with:
- the number of jobs,
- the number of operations per job,
- the number of shared servers.

Evaluating a schedule is polynomial-time, but finding an optimal schedule is computationally intractable for realistic instances.

This motivates the use of **heuristic and evolutionary optimization methods**.

---

## 5. AlphaEvolve / ShinkaEvolve Approach

### Conceptual Overview

**AlphaEvolve / ShinkaEvolve** is a population-based evolutionary optimization framework tailored for
