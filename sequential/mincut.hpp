// Author: Giovanni Rasera
// Help From: Professor Andrea Formisano
// Help From: https://web.stanford.edu/class/archive/cs/cs161/cs161.1172/CS161Lecture16.pdf
// Help From: https://www.tutorialspoint.com/data_structures_algorithms/dsa_kargers_minimum_cut_algorithm.htm
// Help From: https://www.baeldung.com/cs/minimum-cut-graphs
// Help From: https://it.wikipedia.org/wiki/Algoritmo_di_Ford-Fulkerson

#pragma once
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
#include <chrono>

// CPU data
typedef std::vector<std::vector<int>> Graph;
typedef std::vector<std::vector<int>> ResidualFlow;
typedef std::vector<int> ExcessFlow;
typedef std::vector<int> Height;
typedef int Excess_total;

namespace sequential{
    namespace FordFulkerson{
        /*
        As suggested by: Professor Andrea Formisano 
        A node v in G becames v' composed of (v'even -> v'odd).
            - All the original in(v) edges than were coming to v are now 
            connected to v'even with infinite cost.
            - All the original out(v) edges than were originating from v are now 
            originating from v'odd with infinite cost
            - The cost from v'odd to v'even is 1
        example:
        G
        0 -> 1 -> 2 -> 3  /   0 -> 2
        G'
        (0 -> 1) -> (2 -> 3) -> (4 -> 5) -> (6 -> 7)  /    1 -> 4
        */
        void buildGPrimeFromG(const Graph& graph, Graph& graphPrime){
            // in and out edges
            for (int u = 0; u < graph.size(); u++){
                for (int v = 0; v < graph.size(); v++){
                    if(graph[u][v] > 0){
                        int uPodd = u*2 +1;
                        int vPeven = v*2;
                        graphPrime[uPodd][vPeven] = INT_MAX;
                    }
                }
            }

            // all internal to 1
            for (int v = 0; v < graph.size(); v++){
                int vPodd = v*2+1;
                int vPeven = v*2;
                graphPrime[vPeven][vPodd] = 1;
            }
        }

        bool bfs(Graph &rGraph, std::vector<int> &parent, int source, int to){
            // Init
            std::vector<bool> visited(rGraph.size(), false);
            std::queue<int> q;
            q.push(source);
            visited[source] = true;
            parent[source] = -1; 

            // bfs loop
            while (!q.empty()){
                int u = q.front();
                q.pop();
 
                for (int v=0; v<rGraph.size(); v++){
                    // if i did not visit this node and the node is a neighbor then
                    if (visited[v]==false && rGraph[u][v] > 0){
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
            for (int i = 0; i < rGraph.size(); i++){
                if (rGraph[source][i] && !visited[i]){
                    dfs(rGraph, visited, i);
                }
            }
        }

        /*
        Prepare the rGraph
        */ 
        void initialize(Graph &graph, Graph &rGraph){ // G = (graph.size(), E)
            for (int u = 0; u < graph.size(); u++){
                for (int v = 0; v < graph.size(); v++){
                    rGraph[u][v] = graph[u][v];
                }
            }
        }

        std::vector<std::tuple<int, int>> minCutMaxFlow(Graph &graph, Graph &rGraph, int source, int to){
            initialize(graph, rGraph); 

            // return structure
            std::vector<std::tuple<int, int>> ret;

            std::vector<int> parent(graph.size(), -1);
            std::vector<bool> visited(graph.size(), false);
 
            int v, u;

            using namespace std::chrono; 
            auto start = high_resolution_clock::now();
            // Actual Algorithm
            while(bfs(rGraph, parent, source, to)){
                int path_flow = INT_MAX;
                // a path from to -> source
                for (v = to; v != source; v = parent[v]){
                    u = parent[v];
                    path_flow = std::min(path_flow, rGraph[u][v]);
                }
                
                // update the flow in the residual graph
                for (v = to; v != source; v = parent[v]){
                    u = parent[v];
                    rGraph[u][v] -= path_flow;
                    rGraph[v][u] += path_flow;
                }
            }

            // time ends
            auto end = high_resolution_clock::now();
            
            // Run dfs on the residual graph
            dfs(rGraph, visited, source);

            // Checking if there is connection in the residual graph
            for(int i = 0; i < graph.size(); i++)
                for(int j = 0; j < graph.size(); j++)
                    if(visited[i] && ! visited[j] && graph[i][j])
                    ret.push_back({i, j});

            // Info Print
            int V = graph.size();
            int E = V * V;
            auto nanos      = duration_cast<nanoseconds>(end-start).count();
            auto micros     = duration_cast<microseconds>(end-start).count();
            auto millis     = duration_cast<milliseconds>(end-start).count();
            std::cout << "### " 
                << nanos    << ", " << micros   << ", " << millis   << ", " 
                << V << ", " << E << ", " << source << ", " << to << ", " << ret.size() << std::endl;
            
            return ret;
        }
        
    }
};