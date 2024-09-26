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
#include <iomanip>

using namespace std::chrono_literals;
using namespace std::chrono;
 
// Number of vertices in the graph
constexpr int NODES = 512;
constexpr int THREADSCOUNT = 16;
constexpr int step = NODES / THREADSCOUNT;
 
int minDistance(std::array<int, NODES> &dist, bool sptSet[])
{
    // Initialize min value
    int min = INT_MAX, min_index;
 
    for (int v = 0; v < NODES; v++)
        if (sptSet[v] == false && dist[v] <= min)
            min = dist[v], min_index = v;
 
    return min_index;
}
 
void printSolution(std::array<int, NODES> &dist, int n){
    std::cout << ("Vertex   Distance from Source") << std::endl;

    for (int i = 0; i < NODES; i+=1)
        printf("\t%d \t\t\t\t %s\n", i, (dist[i] == INT_MAX ? "CANNOT REACH" : "REACHABLE"));
}
 
std::array<int, NODES> dijkstra(std::array<std::array<int, NODES>, NODES> &graph, int src){
    std::array<int, NODES> dist; 
    bool sptSet[NODES]; 
                    
    for (int i = 0; i < NODES; i++)
        dist[i] = INT_MAX, sptSet[i] = false;
 
    dist[src] = 0;
 
    for (int count = 0; count < NODES - 1; count++) {
        int u = minDistance(dist, sptSet);
 
        sptSet[u] = true;
 
        // picked vertex.
        for (int v = 0; v < NODES; v++)
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
bool isDestinationReachable(std::array<std::array<int, NODES>, NODES> &graph, int source, int destination){
    auto solution = dijkstra(graph, source);
    //printSolution(solution, NODES);
    if (solution[destination] == INT_MAX) return false;
    else return true;
}

void perThreadExecute(std::array<std::array<int, NODES>, NODES> graph, int source, int destination, int threadIdx){
    auto canReach = true;
    //isDestinationReachable(graph, source, destination);
    // need to chech if is reachable
    if(canReach){
        {
            for(int i = (threadIdx-1) * step; i < ((threadIdx-1) * step) + step; i++)
                for(int j = 0; j < NODES; j++){
                    {
                        // modifier
                        int store = graph[i][j];
                        graph[i][j] = 0;
                
                        auto canReach = isDestinationReachable(graph, source, destination);
                        if(canReach == false){
                            // here i can add the shared store
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
    std::array<std::array<int, NODES>, NODES> graph0;

    //int graph[NODES][NODES] = 
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
    for(int i = 0; i < NODES; i++){
        for(int j = 0; j < NODES; j++){
            int temp = rand() % 1;
            graph0[i][j] = temp;
            //graph0[i][j] = graph[i][j];
        }
    }


    // now calculate
    std::cout << std::setw(20) << "Threads: " << THREADSCOUNT << std::endl;
    std::cout << std::setw(20) << "NODES: " << NODES << std::endl;
    std::cout << std::setw(20) << "NODES*NODES: " << NODES*NODES << std::endl;
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
                        perThreadExecute(graph0, 0, NODES-1, index);
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