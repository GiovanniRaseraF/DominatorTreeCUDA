#include <limits.h>
#include <stdio.h>
#include <vector>
#include <array>
#include <ctime>
#include <chrono>
#include <iostream>

using namespace std::chrono_literals;
using namespace std::chrono;
 
// Number of vertices in the graph
#define V 512
 
int minDistance(std::array<int, V> &dist, bool sptSet[])
{
    // Initialize min value
    int min = INT_MAX, min_index;
 
    for (int v = 0; v < V; v++)
        if (sptSet[v] == false && dist[v] <= min)
            min = dist[v], min_index = v;
 
    return min_index;
}
 
void printSolution(std::array<int, V> &dist, int n)
{
    printf("Vertex   Distance from Source\n");
    for (int i = 0; i < V; i+=1)
        printf("\t%d \t\t\t\t %d\n", i, dist[i]);
}
 
std::array<int, V> dijkstra(std::array<std::array<int, V>, V> &graph, int src)
{
    std::array<int, V> dist; 
    bool sptSet[V]; 
                    
    for (int i = 0; i < V; i++)
        dist[i] = INT_MAX, sptSet[i] = false;
 
    dist[src] = 0;
 
    for (int count = 0; count < V - 1; count++) {
        int u = minDistance(dist, sptSet);
 
        sptSet[u] = true;
 
        // picked vertex.
        for (int v = 0; v < V; v++)
            if (
                !sptSet[v] && graph[u][v] 
                && dist[u] != INT_MAX 
                && dist[u] + graph[u][v] < dist[v]
                )
                    dist[v] = dist[u] + graph[u][v];
    }
    
    return dist;
}

//
// Can you reach destination from source ? 
//
bool isDestinationReachable(std::array<std::array<int, V>, V> &graph, int source, int destination){
    auto solution = dijkstra(graph, 0);
    //printSolution(solution, V);
    return (solution[destination] != INT_MAX);
}
 
int main()
{
    std::array<std::array<int, V>, V> graph;
    srand(0);

    // populate graph
    for(int i = 0; i < V; i++){
        for(int j = 0; j < V; j++){
            graph[i][j] = rand() % V;
        }
    }

    // now calculate
    std::cout << "Threads: " << THREADSCOUNT << std::endl;
    std::cout << "V: " << V << std::endl;
    std::cout << "V*V: " << V*V << std::endl;
    std::cout << "Sub if you remove: >>>" << std::endl;

    std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
    std::chrono::time_point<std::chrono::steady_clock> end   = std::chrono::steady_clock::now();
 
    start = std::chrono::steady_clock::now();
    // calculate
    for(int i = 0; i < V; i++){
        for(int j = 0; j < V; j++){
            {
                // modifier
                int store = graph[i][j];
                graph[i][j] = 0;
                
                auto canReach = isDestinationReachable(graph, 0, V-1);
                if(canReach == false){
                    //std::cout << "If you remove arch " << i << ", " << j << " you cannot reach destination";
                }

                graph[i][j] = store;
            }
        }
    }
    end = std::chrono::steady_clock::now();
    
    auto countCuda = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "time: " << countCuda << " ms";
    return 0;
}