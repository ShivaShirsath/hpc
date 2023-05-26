Experiment 1. 
Parallel Breadth First Search and Depth First Search based on existing algorithms using 
OpenMP.
Aim:
Design and implement Parallel Breadth First Search and Depth First Search based on 
existing algorithms using OpenMP. Use a Tree or an undirected graph for BFS and 
DFS.
Theory: 
 High performance computing (HPC) refers to the use of computer systems with 
significant computational power to solve complex problems. OpenMP (Open Multi-
Processing) is a popular library for shared-memory parallel programming, which 
provides a set of compiler directives and library routines to create parallel regions of 
code.
 Using OpenMP, programmers can specify the number of threads to execute in 
parallel, distribute the work among threads, and synchronize the results. This approach 
can greatly improve the performance of HPC applications by exploiting the parallelism 
inherent in the algorithms. OpenMP is widely used in scientific computing, 
engineering, and other fields that require intensive computation, and it has become a de

Algorithm: 
o Parallel Breadth First Search (BFS)
1. Initialize a queue and push the starting vertex into it
2. While the queue is not empty, dequeue a vertex, mark it as visited, and 
add all its unvisited neighbors to the queue
3. Use OpenMP parallel for to parallelize the loop over the adjacent 
vertices of a vertex
o Parallel Depth First Search (DFS)
1. Mark the starting vertex as visited
2. Recursively visit all the unvisited neighbors of the vertex
3. Use OpenMP parallel for to parallelize the loop over the adjacent 
vertices of a vertex
