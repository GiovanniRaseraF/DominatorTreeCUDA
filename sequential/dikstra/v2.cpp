#include <limits.h>
#include <stdio.h>
#include <vector>
#include <array>
#include <ctime>
#include <chrono>
#include <iostream>
#include <thread>
#include <future>
#include <memory>

using namespace std::chrono_literals;
using namespace std::chrono;
 
// Number of vertices in the graph
constexpr int V = 512;
constexpr int THREADSCOUNT = 8;
constexpr int step = V / THREADSCOUNT;
 
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
        printf("\t%d \t\t\t\t %s\n", i, (dist[i] == INT_MAX ? "CANNOT REACH" : "REACHABLE"));
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
    auto solution = dijkstra(graph, source);
    //printSolution(solution, V);
    if (solution[destination] == INT_MAX) return false;
    else return true;
}

void perThreadExecute(std::array<std::array<int, V>, V> graph, int source, int destination, int threadIdx){
    auto canReach = true;//isDestinationReachable(graph, source, destination);
    // need to chech if is reachable
    if(canReach){
        {
            for(int i = (threadIdx-1) * step; i < ((threadIdx-1) * step) + step; i++)
                for(int j = 0; j < V; j++){
                    {
                        // modifier
                        int store = graph[i][j];
                        graph[i][j] = 0;
                
                        auto canReach = isDestinationReachable(graph, source, destination);
                        if(canReach == false){
                            //std::cout << "arch " << i << ", " << j << "\n";
                        }
                        graph[i][j] = store;
                    }
                }
        }
    }
        
}

int main()
{
    srand(0);
    std::array<std::array<int, V>, V> graph0;

    //int graph[V][V] = 
    //{ //  0  1  2  3  4  5  6  7  8 
        //{ 0, 0, 0, 0, 0, 0, 0, 1, 0 }, // 0
        //{ 0, 0, 0, 0, 0, 0, 0, 0, 0 }, // 1
        //{ 0, 0, 0, 0, 0, 0, 0, 0, 0 }, // 2
        //{ 0, 0, 0, 0, 0, 0, 0, 0, 0 }, // 3
        //{ 0, 0, 0, 0, 0, 0, 0, 0, 0 }, // 4
        //{ 0, 0, 0, 0, 0, 0, 0, 0, 0 }, // 5
        //{ 0, 0, 0, 0, 0, 0, 0, 0, 0 }, // 6
        //{ 0, 0, 0, 0, 0, 0, 0, 0, 1 }, // 7
        //{ 0, 0, 0, 0, 0, 0, 0, 0, 0 }  // 7
    //};
    std::array<std::future<void>, THREADSCOUNT> futures;

    // populate graph
    for(int i = 0; i < V; i++){
        for(int j = 0; j < V; j++){
            int temp = rand() % 1;
            graph0[i][j] = temp;
            //graph0[i][j] = graph[i][j];
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
    // run on each thread
    for(int threadIdx = 0; threadIdx < THREADSCOUNT; threadIdx++){

        auto f = std::async(
                    std::launch::async, 
                    [&](int index){
                        // pass every thing
                        perThreadExecute(graph0, 0, V-1, index);
                    },
                    threadIdx
                );

        futures[threadIdx] = std::move(f);
    }

    // wait
    for(int threadIdx = 0; threadIdx < THREADSCOUNT; threadIdx++){
        futures[threadIdx].wait();
    }
    end = std::chrono::steady_clock::now();
    auto countCuda = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "time: " << countCuda << " ms";

    return 0;
}