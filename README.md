### Define System

##### Infrastructure Data
Model as directed graph `G = (V, E)`:
Nodes (V):
- stations
- junctions
- entry/exit points

Edges (E):
- length
- direction (single vs double)
- capacity (usually 1)
- travel time distribution

##### Constraints
- each block has (entry time, exit time) and only one train can occupy this
\[
    [t_i^{in}, t_i^{out}) \cap [t_j^{in}, t_j^{out})
\]
- trains must be seperated in time
\[
    t_j^{in} \geq t_i^{out} + h
\]
- train must traverse connected edges, respect directionality, not teleport or skip blocks
- no overtaking (for sake of this simulation)
- arrival time is always greater than departure time


##### Station and Platforms
For each station:
- number of platforms
- platform track comptability
- min/max dwell times