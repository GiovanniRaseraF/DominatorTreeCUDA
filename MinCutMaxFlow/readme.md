# MinCutMaxFlow algorithm
- Author: [Giovanni Rasera](https://github.com/GiovanniRaseraF)
- Help From: [Professor Andrea Formisano](https://users.dimi.uniud.it/~andrea.formisano/)
- Help From: https://web.stanford.edu/class/archive/cs/cs161/cs161.1172/CS161Lecture16.pdf
- Help From: https://www.tutorialspoint.com/data_structures_algorithms/dsa_kargers_minimum_cut_algorithm.htm
- Help From: https://www.baeldung.com/cs/minimum-cut-graphs
- Help From: https://it.wikipedia.org/wiki/Algoritmo_di_Ford-Fulkerson
- Help From: https://www.nvidia.com/content/GTC/documents/1060_GTC09.pdf
- Help From: https://en.wikipedia.org/wiki/Push%E2%80%93relabel_maximum_flow_algorithm
- Help From: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4563095
- Help From: https://arxiv.org/pdf/2404.00270
- Help From: https://github.com/NTUDDSNLab/WBPR/tree/master/maxflow-cuda
- Help From: https://www.adrian-haarbach.de/idp-graph-algorithms/implementation/maxflow-push-relabel/index_en.html

## Problem Description
- Write a program that, given a node x in V (different from s), determines a subset D of nodes such that: 
    1.  the source s does not belong to D, and the node x does not belong to D 
    2. every directed path from s to x passes through at least one node in D 
    3. D is minimal with respect to cardinality 
      (i.e., there is no other set that satisfies properties 1 and 2 and has fewer elements than D)

## Solution
- MinCutMaxFlow algorithm

## Implementation
### Sequential 
- Ford-Folkerson algorithm
    - Uses bfs and dfs to determinate if the last node can be reached
    - Uses a new Graph called G' generate from G tha allows the algorithm to cut Nodes and not edges
- Karger’s Algorithm
    - It'a a Monte Carlo algorithm that uses super nodes and super adges and random choises to subdivide the graph

### Parallel
- (Goldberg-Tarjan) Push–relabel maximum flow algorithm
- TODO: implement parallel algo


## How to compile
### Sequential
```bash
make # V = 1024 just run once
#
make test --silent # V encreases every run
```
### Parallel
```bash
# TODO: to implement
```
