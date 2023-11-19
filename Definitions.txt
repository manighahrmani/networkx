# Graph representation
I am working on treewidth and minimum triangulation (chordal supergraphs with the minimum number of added edges) of simple undirected graphs. More specifically grid graphs on 5 rows and 5 or more columns (5xn grids). Here is my graph representation: Vertices have labels (represented in string) and the vertex with label 10203 is on row 2 and column 3. Similarly, 11114 is the vertex on row 11 and column 14 (the left-most character is always 1).

# Hypothesis to prove
I know that 5xn grids have a treewidth of 5. I want to test my hypothesis that all 5xn grids have a minimum triangulation where the treewidth of this triangulation is 5 (all of the largest cliques in this triangulation have 6 vertices).

# Definition of elimination ordering
To do this, I have constructed minimum elimination orderings of these graphs. We can construct a minimum triangulation H from a minimum elimination ordering with the following algorithm:
- Let G_0 = G (the input graph)
- We can construct G_i from G_{i-1} by first adding all the missing edges between the neighbors of the ith vertex in G_{i-1} and then removing the ith vertex from G_{i-1}.
- By this definition, G_n is an empty graph
The triangulation H is constructed from G by the addition of all the edges during the minimum elimination ordering. 

# Definition of madj
Given an elimination ordering, the higher adjacency of a vertex v, written madj(v) (where v is the ith vertex in the elimination ordering of graph G) is the neighborhood of v_i in the graph G_{i-1}.
madj(v) can also be calculated from the input graph G: madj(v) contains every vertex u in G such that there exists a path from u to v (or an edge) that contains vertices that come earlier in the elimination ordering than both u and v.
For example, take a 5x5 grid where we first eliminate 10505 and then 10405. Then madj(10405) = {10305, 10404, 10504} and madj(10505) = {10405, 10504}.

# Existing theorems (where madj of a vertex has only 2 vertices)
I already have a theorem that tells me how to safely eliminate vertices where their madj has 2 vertices so that the resulting elimination ordering still results in a minimum triangulation. So for example in a 5x5 grid, I know that I can safely eliminate 10505 first.

# Existing theorems (where madj of a vertex has 3 vertices with other restrictions)
I also have a theorem that tells me I can eliminate v at step i where madj(v) had n vertices if G_{i_{i-1}} has the vertex connectivity of n and madj(v) is an almost clique in G_{i-{i-1}} (almost clique means that there exists a vertex u in madj(v) such that madj(v)-u is a clique). For example, after 10101, 10105, 10501 and 10505 are eliminated in the 5x5 grid, the resulting graph becomes 3-connected and I know I can eliminate 10405 next because its madj is {10305, 10404, 10504} and {10404, 10504} is a clique in G_{i_{i-1}} so {10305, 10404, 10504} is an almost clique.
