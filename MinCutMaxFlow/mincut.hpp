// Author: Giovanni Rasera
// Source: https://web.stanford.edu/class/archive/cs/cs161/cs161.1172/CS161Lecture16.pdf
// Help From: https://www.tutorialspoint.com/data_structures_algorithms/dsa_kargers_minimum_cut_algorithm.htm
// Help From: https://www.baeldung.com/cs/minimum-cut-graphs
// 

#include <deque>
#include <queue>
#include <climits>
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

namespace sequential{
    namespace Default{
        bool bfs(Graph &rGraph, std::vector<int> &parent, int source, int to){
            // Init
            std::vector<bool> visited(rGraph.size(), false);
            std::queue<int> q;
            q.push(source);
            visited[source] = true;
            parent[source] = -1; 

            // bfs loop
            while (!q.empty())
            {
                int u = q.front();
                q.pop();
 
                for (int v=0; v<rGraph.size(); v++)
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
            for (int i = 0; i < rGraph.size(); i++)
            if (rGraph[source][i] && !visited[i])
                dfs(rGraph, visited, i);
        }

        void initialize(Graph &graph, Graph &rGraph){ // G = (graph.size(), E)
            for (int u = 0; u < graph.size(); u++)
                for (int v = 0; v < graph.size(); v++)
                    rGraph[u][v] = graph[u][v];
        }

        std::vector<std::tuple<int, int>> minCutMaxFlow(Graph &graph, Graph &rGraph, int source, int to){
            initialize(graph, rGraph); 

            // return structure
            std::vector<std::tuple<int, int>> ret;

            std::vector<int> parent(graph.size(), -1);
            std::vector<bool> visited(graph.size(), false);
 
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

            // run dfs on the residual graph
            dfs(rGraph, visited, source);

            // checking if there is connection in the residual graph
            for(int i = 0; i < graph.size(); i++)
                for(int j = 0; j < graph.size(); j++)
                    if(visited[i] && ! visited[j] && graph[i][j])
                    ret.push_back({i, j});
            
            return ret;
        }
        
    }
};