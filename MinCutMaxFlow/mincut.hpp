// Author: Giovanni Rasera
// Source: https://web.stanford.edu/class/archive/cs/cs161/cs161.1172/CS161Lecture16.pdf
// Help From: https://www.tutorialspoint.com/data_structures_algorithms/dsa_kargers_minimum_cut_algorithm.htm
// Help From: https://www.baeldung.com/cs/minimum-cut-graphs
// 

#include <deque>
#include <queue>
#include <climints>
#include <list>
#include <array>
#include <vector>
#include <algorithm>
#include <iostream>
#include <random>
#include <unordered_set>
#include <tuple>

// can use in cpu
typedef std::vector<std::vector<int>> Graph;

constexpr int V = 1024;

namespace sequential{
    namespace Default{
        bool bfs(Graph &rGraph, std::vector<int> &parent, int source, int to){
            // Init
            std::vector<bool> visited(V, false);
            std::queue<int> q;
            q.push(source);
            visited[source] = true;
            parent[source] = -1; 

            // bfs loop
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

            // return true if it can reach to form source
            return visited[to];
        }

        void dfs(Graph &rGraph, std::vector<bool> &visited, int source){
            visited[source] = true;
            for (int i = 0; i < V; i++)
            if (rGraph[source][i] && !visited[i])
                dfs(rGraph, visited, i);
        }

        void initialize(Graph &graph, Graph &rGraph){ // G = (V, E)
            for (int u = 0; u < V; u++)
                for (int v = 0; v < V; v++)
                    rGraph[u][v] = graph[u][v];
        }

        void minCutMaxFlow(Graph &graph, Graph &rGraph, int source, int to){
            initialize(graph, rGraph); 
 
            std::vector<int> parent(V, -1);
 
            int v, u;

            while(bfs(rGraph, parent, source, to)){
                std::cout << "Done bfs!" << std::endl;
                int path_flow = INT_MAX;
                for (v=to; v!=source; v=parent[v])
                {
                    u = parent[v];
                    path_flow = std::min(path_flow, rGraph[u][v]);
                }
 
                for (v=to; v != source; v=parent[v])
                {
                    u = parent[v];
                    rGraph[u][v] -= path_flow;
                    rGraph[v][u] += path_flow;
                }
            }

        }
        
    }
};