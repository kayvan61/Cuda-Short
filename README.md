# Cuda-Short

This is a term project for the UT multi-core computing class.

It provides a parallel implementation of Dijkstra's shortest path algo as detailed in a paper [2].

Graphs are represented as positive weighted adjacency matricies. a weight of 0 notes that there is no edge. No negative weights are allowed.

To see the difference in compute time vs the CPU:
1. switch to directory src
2. `make timeTest`
3. `./timeTest`

To verify results of random Graphs:
1. switch to directory src
2. `make funcTest`
3. `./funcTest`

To run a demo with an adjacency matrix from a file:
1. switch to directory src
2. `make demo`
3. `./demo <fileName>`

# Adjacency Matrix File Format

The program assumes adjacency matricies will be in the format where `n` is the number of nodes, and `s` is the source node.

```
n
s
<row 1 (data assumed to be n x n)>
<row 2>
<...  >
<row n>
```

# Requirements

1. This make file assumes a cuda GPU compatible with compute_60
2. Nvidia's cuda toolkit
3. g++7

# References

1. https://www.mcs.anl.gov/~itf/dbpp/text/node35.html
2. https://www.researchgate.net/publication/237149132_A_New_GPU-based_Approach_to_the_Shortest_Path_Problem