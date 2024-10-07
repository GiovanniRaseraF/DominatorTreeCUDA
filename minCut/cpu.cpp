// C++ program for finding minimum cut using Ford-Fulkerson
#include <iostream>
#include <limits.h>
#include <string.h>
#include <array>
#include <queue>
#include <chrono>
#include <thread>
#include <future>
using namespace std;

// Number of vertices in given graph
//#define V 1024


#ifdef USEARRAY
    typedef std::array<std::array<int, V>, V> Graph; // impossible to use for big data
#endif

#ifdef USEVECTOR
    typedef std::vector<std::vector<int>> Graph;
#endif

std::chrono::time_point<std::chrono::steady_clock> start_bfs = std::chrono::steady_clock::now();
std::chrono::time_point<std::chrono::steady_clock> end_bfs   = std::chrono::steady_clock::now();
int callcount_bfs = 0;
uint64_t countms_bfs = 0;

void resetBfsTimes(){
    start_bfs = std::chrono::steady_clock::now();
    end_bfs   = std::chrono::steady_clock::now();
    callcount_bfs = 0;
    countms_bfs = 0;
}

std::chrono::time_point<std::chrono::steady_clock> start_dfs = std::chrono::steady_clock::now();
std::chrono::time_point<std::chrono::steady_clock> end_dfs   = std::chrono::steady_clock::now();
int callcount_dfs = 0;
uint64_t countms_dfs = 0;

void resetDfsTimes(){
    start_dfs = std::chrono::steady_clock::now();
    end_dfs   = std::chrono::steady_clock::now();
    callcount_dfs = 0;
    countms_dfs = 0;
}

int bfs(Graph &rGraph, int s, int t, int parent[]){
    callcount_bfs +=1;
    start_bfs = std::chrono::steady_clock::now();
    bool visited[V];
    memset(visited, 0, sizeof(visited));
 
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
    
    end_bfs = std::chrono::steady_clock::now();
    auto countMinCutCPU = std::chrono::duration_cast<std::chrono::milliseconds>(end_bfs - start_bfs).count();
    countms_bfs += countMinCutCPU;

    // arrived to end ? 
    return (visited[t] == true);
}
 
void dfs(Graph &rGraph, int s, bool visited[])
{
    visited[s] = true;
    for (int i = 0; i < V; i++)
       if (rGraph[s][i] && !visited[i])
           dfs(rGraph, i, visited);
}

void initResidual(Graph &graph, Graph &rGraph){
    for (int u = 0; u < V; u++)
        for (int v = 0; v < V; v++)
             rGraph[u][v] = graph[u][v];
}

void minCut(Graph &graph, int s, int t, Graph &rGraph){
    initResidual(graph, rGraph); 
 
    int parent[V];  // This array is filled by BFS and to store path
 
    int v, u;
    // Augment the flow while there is a path from source to sink
    while (bfs(rGraph, s, t, parent))
    {
        int path_flow = INT_MAX;
        for (v=t; v!=s; v=parent[v])
        {
            u = parent[v];
            path_flow = min(path_flow, rGraph[u][v]);
        }
 
        for (v=t; v != s; v=parent[v])
        {
            u = parent[v];
            rGraph[u][v] -= path_flow;
            rGraph[v][u] += path_flow;
        }
    }
 
    return;
}

void printResidual( Graph &graph, int s, int t, Graph &rGraph ){
    std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
    std::chrono::time_point<std::chrono::steady_clock> end   = std::chrono::steady_clock::now();
    // Flow is maximum now, find vertices reachable from s
    bool visited[V];
    memset(visited, false, sizeof(visited));

    start = std::chrono::steady_clock::now();
    minCut(graph, s, t, rGraph);
    end = std::chrono::steady_clock::now();
    auto countMinCutCPU= std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "minCut time: " << countMinCutCPU<< " ms";

    start_dfs = std::chrono::steady_clock::now();
    dfs(rGraph, s, visited);
    end_dfs = std::chrono::steady_clock::now();
    auto countMinCutDFSCPU= std::chrono::duration_cast<std::chrono::milliseconds>(end_dfs - start_dfs).count();
    std::cout << std::endl;
    std::cout << "DFS time: " << countMinCutDFSCPU<< " ms";

    // Print all edges that are from a reachable vertex to
    // non-reachable vertex in the original graph
    for (int i = 0; i < V; i++)
      for (int j = 0; j < V; j++)
         if (visited[i] && !visited[j] && graph[i][j])
              //cout << i << " - " << j << endl;
              continue;
}
 
// Driver program to test above functions
int main(){
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

    // Graph graph = {{ 
    // //   0  1  2  3  4  5  6
    //     {0, 1, 0, 1, 1, 1, 0}, // 0
    //     {0, 0, 1, 0, 0, 0, 0}, // 1
    //     {0, 0, 0, 0, 0, 0, 1}, // 2
    //     {0, 0, 1, 0, 0, 0, 0}, // 3
    //     {0, 0, 1, 0, 0, 0, 10}, // 4
    //     {0, 0, 1, 0, 0, 0, 0}, // 5
    //     {0, 0, 0, 0, 0, 0, 0}, // 6
    // }};

    // Graph rgraph = {{ 
    // //   0  1  2  3  4  5  6
    //     {0, 1, 0, 1, 1, 1, 0}, // 0
    //     {0, 0, 1, 0, 0, 0, 0}, // 1
    //     {0, 0, 0, 0, 0, 0, 1}, // 2
    //     {0, 0, 1, 0, 0, 0, 0}, // 3
    //     {0, 0, 1, 0, 0, 0, 10}, // 4
    //     {0, 0, 1, 0, 0, 0, 0}, // 5
    //     {0, 0, 0, 0, 0, 0, 0}, // 6
    // }};

    //printResidual(graph, 0, 6, rgraph);

    std::cout << "MinCutCPU: V=" << V << std::endl;

#ifdef USEARRAY
    Graph rgraph;
    Graph graph;
#endif

#ifdef USEVECTOR
    Graph rgraph(V);
    Graph graph(V);
    for(int i = 0; i < V; i ++){
        for(int j = 0; j < V; j++){
            graph[i].push_back(0);
            rgraph[i].push_back(0);
        }
    }
#endif

    resetBfsTimes();
    for(int i = 0; i < V; i ++){
        for(int j = 0; j < V; j++){
            graph[i][j] = 0;
        }
    }

    // generate a connection form start to finish
    for(int i = 0; i < V-1; i ++){
        graph[i][i+1] = 1;
    }
    std::cout << "min cut will be one of the edges" << std::endl;
    printResidual(graph, 0, V-1, rgraph);
    std::cout << std::endl;


    // Example 2
    resetBfsTimes();
    for(int i = 0; i < V; i ++){
        for(int j = 0; j < V; j++){
            graph[i][j] = 0;
        }
    }

    // generate a connection form start to finish
    for(int i = 0; i < V-1; i ++){
        graph[i][i+1] = 1;
    }
    // add adge 0 -> 300
    graph[0][300] = 100;
    std::cout <<  "min cut will be after 300" << std::endl;
    printResidual(graph, 0, V-1, rgraph);

    std::cout << std::endl;
    std::cout << "bfs total time: " << countms_bfs << " ms" << std::endl;
    std::cout << "bfs calls: " << callcount_bfs << " times called" << std::endl;
    std::cout << "bfs avg time: " << countms_bfs / callcount_bfs << " ms" << std::endl;
    std::cout << std::endl;
 
    // Example 3
    resetBfsTimes();
    for(int i = 0; i < V; i ++){
        for(int j = 0; j < V; j++){
            graph[i][j] = 0;
        }
    }

    // generate a connection form start to finish
    for(int i = 0; i < V-1; i ++){
        graph[i][i+1] = 1;
    }
    // add adge 0 -> 300
    graph[0][300] = 100;
    for(int i = 0; i < 300; i ++){
        graph[0][i+1] = 100;
        graph[i][i+100] = 100;
    }
    std::cout <<  "More lines" << std::endl;
    printResidual(graph, 0, V-1, rgraph);

    std::cout << std::endl;
    std::cout << "bfs total time: " << countms_bfs << " ms" << std::endl;
    std::cout << "bfs calls: " << callcount_bfs << " times called" << std::endl;
    std::cout << "bfs avg time: " << countms_bfs / callcount_bfs << " ms" << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    return 0;
}