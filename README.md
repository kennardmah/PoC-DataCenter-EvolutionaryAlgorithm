# ShinkaEvolve to Optimize Transportation Decisions

## TO DO LIST 

### Build Simulation

(1) create ideal data types and fake data (this will be used for testing)
(2) replace fake data with real data

Input Data (from MTA datasets)

**Network structure**
- Stations (nodes)
  - Passenger arrival rates (by time bucket)
  - Platform capacity (e.g., maximum waiting passengers)
- Track connections (edges)
  - Average travel time between stations
  - Reliability or delay penalty (optional)

**Passenger demand**
- Origin–Destination (OD) demand between stations  
  - Aggregated by time bucket (e.g., AM peak, PM peak)

**Train characteristics**
- Average capacity per train
- Fleet size constraints (maximum trains per line)
- Average dwell time assumptions

---

Decision Variables (Strategy)

(1) create simple `initial.py` file
(2) link file with `fitness.py`
(3) test if it evolves over time

**Service-level decisions**
- Train frequency per line per time bucket  
  - Expressed as trains per hour
- High-level fleet allocation  
  - Number of trains assigned to each line

---

Simulation Outputs (Evaluation Metrics)

(1) define `fitness.py` file
(2) define single objective
(3) explore multi-objective

**Passenger performance**
- Average passenger travel time
- Average waiting time at stations
- Crowding level (load factor)

**Operational performance**
- Total operating cost  
  - Proportional to total train-hours operated
- Capacity utilization


