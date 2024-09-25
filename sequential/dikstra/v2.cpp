#include <limits.h>
#include <stdio.h>
#include <vector>
#include <array>
#include <ctime>
#include <chrono>
#include <iostream>
#include <thread>
#include <future>

using namespace std::chrono_literals;
using namespace std::chrono;
 
// Number of vertices in the graph
#define V 1024
 
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
    for (int i = 0; i < V; i+=32)
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

    return (solution[destination] != INT_MAX);
}

constexpr int THREADSCOUNT = 16;
int main()
{
    std::array<std::array<int, V>, V> graph;
    std::array<std::future<void>, THREADSCOUNT> futures;
    srand(0);

    // populate graph
    for(int i = 0; i < V; i++){
        for(int j = 0; j < V; j++){
            graph[i][j] = rand() % V;
        }
    }

    // run on each thread
    for(int threadIdx = 0; threadIdx < THREADSCOUNT; threadIdx++){
        auto f = std::async(
                    std::launch::async, 

                    [&](int r){
                        std::cout << r;
                        std::this_thread::sleep_for(500ms);
                    }, 

                    43
                );
        futures[threadIdx] = std::move(f);
    }

    // wait
    for(int threadIdx = 0; threadIdx < THREADSCOUNT; threadIdx++){
        futures[threadIdx].wait();
    }

    // calculate
    //std::cout << "Return to Compute>>>"; std::cin.ignore();
    ////auto solution = dijkstra(graph, 0);
    //for(int i = 0; i < V; i++){
        //for(int j = 0; j < V; j++){
            //{
                //// modifier
                //int store = graph[i][j];
                //graph[i][j] = 0;
                
                //auto canReach = isDestinationReachable(graph, 0, 1023);
                //if(canReach == false){
                    //std::cout << "If you remove arch " << i << ", " << j << " you cannot reach destination";
                //}

                //graph[i][j] = store;
            //}
        //}
    //}
    // print solution
    //printSolution(solution, V);
 
    return 0;
}