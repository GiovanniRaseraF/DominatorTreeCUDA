// C++ program for finding minimum cut using Ford-Fulkerson
#include <iostream>
#include <limits.h>
#include <string.h>
#include <array>
#include <queue>
using namespace std;
 
// Number of vertices in given graph
#define V 7
 
/* Returns true if there is a path from source 's' to sink 't' in
  residual graph. Also fills parent[] to store the path */
int bfs(std::array<std::array<int, V>, V> &rGraph, int s, int t, int parent[])
{
    // Create a visited array and mark all vertices as not visited
    bool visited[V];
    memset(visited, 0, sizeof(visited));
 
    // Create a queue, enqueue source vertex and mark source vertex
    // as visited
    queue <int> q;
    q.push(s);
    visited[s] = true;
    parent[s] = -1;
 
    // Standard BFS Loop
    while (!q.empty())
    {
        int u = q.front();
        q.pop();
 
        for (int v=0; v<V; v++)
        {
            if (visited[v]==false && rGraph[u][v] > 0)
            {
                q.push(v);
                parent[v] = u;
                visited[v] = true;
            }
        }
    }
 
    // If we reached sink in BFS starting from source, then return
    // true, else false
    return (visited[t] == true);
}
 
// A DFS based function to find all reachable vertices from s.  The function
// marks visited[i] as true if i is reachable from s.  The initial values in
// visited[] must be false. We can also use BFS to find reachable vertices
void dfs(std::array<std::array<int, V>, V> &rGraph, int s, bool visited[])
{
    visited[s] = true;
    for (int i = 0; i < V; i++)
       if (rGraph[s][i] && !visited[i])
           dfs(rGraph, i, visited);
}
 
// Prints the minimum s-t cut
void minCut(std::array<std::array<int, V>, V> &graph, int s, int t, std::array<std::array<int, V>, V> &rGraph)
{
    int u, v;
 
    // Create a residual graph and fill the residual graph with
    // given capacities in the original graph as residual capacities
    // in residual graph
    // rGraph[i][j] indicates residual capacity of edge i-j
    for (u = 0; u < V; u++)
        for (v = 0; v < V; v++)
             rGraph[u][v] = graph[u][v];
 
    int parent[V];  // This array is filled by BFS and to store path
 
    // Augment the flow while there is a path from source to sink
    while (bfs(rGraph, s, t, parent))
    {
        // Find minimum residual capacity of the edges along the
        // path filled by BFS. Or we can say find the maximum flow
        // through the path found.
        int path_flow = INT_MAX;
        for (v=t; v!=s; v=parent[v])
        {
            u = parent[v];
            path_flow = min(path_flow, rGraph[u][v]);
        }
 
        // update residual capacities of the edges and reverse edges
        // along the path
        for (v=t; v != s; v=parent[v])
        {
            u = parent[v];
            rGraph[u][v] -= path_flow;
            rGraph[v][u] += path_flow;
        }
    }
 
    
    
 
    return;
}

void printResidual( std::array<std::array<int, V>, V> &graph, int s, int t, std::array<std::array<int, V>, V> &rGraph ){
    minCut(graph, s, t, rGraph);
    
    // Flow is maximum now, find vertices reachable from s
    bool visited[V];
    memset(visited, false, sizeof(visited));
    dfs(rGraph, s, visited);

    // Print all edges that are from a reachable vertex to
    // non-reachable vertex in the original graph
    for (int i = 0; i < V; i++)
      for (int j = 0; j < V; j++)
         if (visited[i] && !visited[j] && graph[i][j])
              cout << i << " - " << j << endl;
}
 
// Driver program to test above functions
int main()
{
    // Let us create a graph shown in the above example
    //int graph[V][V] = { {0, 16, 13, 0, 0, 0 , 0},
                    //     {0, 0, 10, 12, 0, 0, 0},
                    //     {0, 4, 0, 0, 14, 0, 0},
                    //     {0, 0, 9, 0, 0, 20, 0},
                    //     {0, 0, 0, 7, 0, 4, 0},
                    //     {0, 0, 0, 0, 0, 0, 0},
                    //     {0, 0, 0, 0, 0, 0, 0}
                    //   };
 
    //minCut(graph, 0, 6);

    std::array<std::array<int, V>, V> graph = {{ 
    //   0  1  2  3  4  5  6
        {0, 1, 0, 1, 1, 1, 0}, // 0
        {0, 0, 1, 0, 0, 0, 0}, // 1
        {0, 0, 0, 0, 0, 0, 1}, // 2
        {0, 0, 1, 0, 0, 0, 0}, // 3
        {0, 0, 1, 0, 0, 0, 10}, // 4
        {0, 0, 1, 0, 0, 0, 0}, // 5
        {0, 0, 0, 0, 0, 0, 0}, // 6
    }};

    std::array<std::array<int, V>, V> rgraph = {{ 
    //   0  1  2  3  4  5  6
        {0, 1, 0, 1, 1, 1, 0}, // 0
        {0, 0, 1, 0, 0, 0, 0}, // 1
        {0, 0, 0, 0, 0, 0, 1}, // 2
        {0, 0, 1, 0, 0, 0, 0}, // 3
        {0, 0, 1, 0, 0, 0, 10}, // 4
        {0, 0, 1, 0, 0, 0, 0}, // 5
        {0, 0, 0, 0, 0, 0, 0}, // 6
    }};

    printResidual(graph, 0, 6, rgraph);
 
    return 0;
}