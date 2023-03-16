# Concept

# Motivation
A lot of apps need an iterative structure
- we use a structure called define-once--run-repeatedly

- we define a graph of execution
- cudaGraphAddKenelNode(&a, graph, NULL, 0, &nodeParams);   NODES
- cudaGraphAddDependencies(graph, &a, &b, 1);               ARCHI

- Initialize and execution
- cudaGraphInstantiate(&instance, graph);
```c++
for(int i = 0; i < 100; i++)
    cudaGraphLaunch(instance, stream)
```

- 