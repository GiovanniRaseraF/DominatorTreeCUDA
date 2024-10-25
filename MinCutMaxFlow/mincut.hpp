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

// CPU data
typedef std::vector<std::vector<int>> Graph;
typedef std::vector<std::vector<int>> ResidualFlow;
typedef std::vector<int> ExcessFlow;
typedef std::vector<int> Height;
typedef int Excess_total;

namespace sequential{
    namespace GraphCutsGeneric {
        bool active(int x, ExcessFlow &u, Height &h, int HEIGHT_MAX){
            return u[x] > 0 && h[x] < HEIGHT_MAX;
        }
        bool canXpushToY(int x, int y, ResidualFlow &c, Height &h){
            return c[x][y] > 0 && h[y] == h[x] - 1;
        }

        bool canXbeRelabled(int x, const ResidualFlow &c, const Height &h, const Graph &G){
            for(int y = 0; y < c.size(); y++){
                if(G[x][y] > 0){
                    if(c[x][y] > 0 && h[y] >= h[x]){
                    }else{
                        return false;
                    }
                }
            }
            return true;
        }

        void push(int x, Height &h, ExcessFlow &u, ResidualFlow &c, const Graph &G, int HEIGHT_MAX){
            if(active(x, u, h, HEIGHT_MAX)){
                for(int y = 0; y < G.size(); y++){
                    if(G[x][y] > 0){
                        if(h[y] == h[x] - 1){
                            int flow = std::min(c[x][y], u[x]);
                            u[x] -= flow;
                            u[y] += flow;
                            c[x][y] -= flow;
                            c[y][x] -= flow;
                        }
                    }
                }
            }
        }

        void relable(int x, Height &h, ExcessFlow &u, ResidualFlow &c, const Graph &G, int HEIGHT_MAX){
            if(active(x, u, h, HEIGHT_MAX)){
                int my_height = HEIGHT_MAX;
                for(int y = 0; y < G.size(); y++){
                    if(G[x][y] > 0){
                        if(c[x][y] > 0){
                            my_height = std::min(my_height, h[y]+1);
                        }
                    }
                }
                h[x] = my_height;
            }
        }

        void minCutMaxFlow(Graph &G, Graph &Gf, int source, int to){
            std::cout << "sequential::minCutMaxFlow" << std::endl;
            int N = G.size();

            Excess_total etotal = 0;
            ExcessFlow e(N, 0); 
            Height h(N, 0); 
            h[0] = N;

            ResidualFlow cf(N); 
            for(int i = 0; i < N; i++){
                for(int j = 0; j < N; j++){
                    cf[i].push_back(0);
                }
            }

            std::cout << "\n\n\ne: ";
            for(int j = 0; j < N; j++){
                std::cout << e[j] << " ";
            }
            std::cout << "\n";

            std::cout << "h: ";
            for(int j = 0; j < N; j++){
                std::cout << h[j] << " ";
            }
            std::cout << "\n";

            std::cout << "graph:\n";
            for(int i = 0; i < N; i ++){
                for(int j = 0; j < N; j++){
                    printf("%d/%d  ", Gf[i][j], cf[i][j]);
                }
                printf("\n");
            }

            //preflow(G, Gf, cf, e, etotal);

            //while((e[source] + e[to]) < etotal){
                // Step 1: Push-relabel kernel (GPU)
                int cicle = G.size(); // = |V|
                while(cicle > 0){
                    for(int u = 0; u < N; u++){
		                push(u, h, e, cf, G, N);	
                    }

                    for(int u = 0; u < N; u++){
		                relable(u, h, e, cf, G, N);	
                    }

                    std::cout << "\n\n\ne: ";
                    for(int j = 0; j < N; j++){
                        std::cout << e[j] << " ";
                    }
                    std::cout << "\n";

                    std::cout << "h: ";
                    for(int j = 0; j < N; j++){
                        std::cout << h[j] << " ";
                    }
                    std::cout << "\n";

                    std::cout << "graph:\n";
                    for(int i = 0; i < N; i ++){
                        for(int j = 0; j < N; j++){
                            printf("%d/%d  ", Gf[i][j], cf[i][j]);
                        }
                        printf("\n");
                    }

                    std::cin.ignore();
                    cicle--;
                }
            //}
        }


    };

    namespace GoldbergTarjan{
        // Initialize the flow
        void preflow(const Graph &G, Graph &Gf, ResidualFlow &cf, ExcessFlow &excess, Excess_total &excessTotal){
            std::cout << "called Preflow" << std::endl;
            // maybe i can parallelize this
            for(int s = 0; s < G.size(); s++){
                for(int v = 0; v < G.size(); v++){
                    if(G[s][v] > 0){
                        cf[s][v] = 0;
                        Gf[s][v] = 0;
                        cf[v][s] = G[s][v];
                        Gf[v][s] = G[s][v];
                        excess[v] = G[s][v];
                        excessTotal += G[s][v];
                    }
                }
            }
        }

        void pushrelable(const Graph &G, const Graph &Gf, ResidualFlow &cf, int x, ExcessFlow &excess, Height &h, int HEIGHT_MAX){
            int NumberOfNodes = G.size();
            int u = x;

            if(excess[u] > 0 && h[u] < NumberOfNodes){
                int hprime = INT_MAX/2;
                int vprime = INT_MAX/2;
                for(int v = 0; v < NumberOfNodes; v++){
                    if(Gf[u][v] > 0){ // is (u,v) £ Ef ?
                        if(h[v] < hprime){
                            hprime = h[v];
                            vprime = v;
                        }
                    }
                }
                if(h[u] > hprime){
                    int d = std::min(excess[u], cf[u][vprime]);
                    cf[u][vprime]-=d;
                    excess[u]-=d;
                    cf[vprime][u]+=d;
                    excess[vprime]+=d;
                }else{
                    h[u] = hprime + 1;
                }

            }
        }

        void minCutMaxFlow(Graph &G, Graph &Gf, int source, int to){
            std::cout << "sequential::minCutMaxFlow" << std::endl;
            int N = G.size();

            Excess_total etotal = 0;
            ExcessFlow e(N, 0); 
            Height h(N, 0); 
            h[0] = N;

            ResidualFlow cf(N); 
            for(int i = 0; i < N; i++){
                for(int j = 0; j < N; j++){
                    cf[i].push_back(0);
                }
            }

            std::cout << "\n\n\ne: ";
            for(int j = 0; j < N; j++){
                std::cout << e[j] << " ";
            }
            std::cout << "\n";

            std::cout << "h: ";
            for(int j = 0; j < N; j++){
                std::cout << h[j] << " ";
            }
            std::cout << "\n";

            std::cout << "graph:\n";
            for(int i = 0; i < N; i ++){
                for(int j = 0; j < N; j++){
                    printf("%d/%d  ", Gf[i][j], cf[i][j]);
                }
                printf("\n");
            }

            preflow(G, Gf, cf, e, etotal);

            //while((e[source] + e[to]) < etotal){
                // Step 1: Push-relabel kernel (GPU)
                int cicle = G.size(); // = |V|
                while(cicle > 0){
                    for(int u = 0; u < N; u++){
		                pushrelable(G, Gf, cf, u, e, h, N);	
                    }

                    std::cout << "\n\n\ne: ";
                    for(int j = 0; j < N; j++){
                        std::cout << e[j] << " ";
                    }
                    std::cout << "\n";

                    std::cout << "h: ";
                    for(int j = 0; j < N; j++){
                        std::cout << h[j] << " ";
                    }
                    std::cout << "\n";

                    std::cout << "graph:\n";
                    for(int i = 0; i < N; i ++){
                        for(int j = 0; j < N; j++){
                            printf("%d/%d  ", Gf[i][j], cf[i][j]);
                        }
                        printf("\n");
                    }

                    std::cin.ignore();
                    cicle--;
                }
            //}
        }
    };


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