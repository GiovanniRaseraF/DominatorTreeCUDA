// Author: Giovanni Rasera

#include <iostream>
#include <queue>
#include <vector>
#include <tuple>
#include <iomanip>
// #include "mincut.hpp"
// #include "tests.hpp"
using namespace std;

const int inf = 1000000000;

const int numberOfNodes = 7;
int n = numberOfNodes;
vector<vector<int>> capacity, flow;
vector<int> height{}, excess{}, seen{};
queue<int> excess_vertices{};

void push(int u, int v) {
    int d = min(excess[u], capacity[u][v] - flow[u][v]);
    flow[u][v] += d;
    flow[v][u] -= d;
    excess[u] -= d;
    excess[v] += d;
    if (d && excess[v] == d)
        excess_vertices.push(v);
}

void relabel(int u) {
    int d = inf;
    for (int i = 0; i < n; i++) {
        if (capacity[u][i] - flow[u][i] > 0)
            d = min(d, height[i]);
    }
    if (d < inf)
        height[u] = d + 1;
}

void discharge(int u) {
    while (excess[u] > 0) {
        if (seen[u] < n) {
            int v = seen[u];
            if (capacity[u][v] - flow[u][v] > 0 && height[u] > height[v])
                push(u, v);
            else 
                seen[u]++;
        } else {
            relabel(u);
            seen[u] = 0;
        }
    }
}

std::vector<std::tuple<int, int>>max_flow(int s, int t) {
    height.assign(n, 0);
    height[s] = n;
    flow.assign(n, vector<int>(n, 0));
    excess.assign(n, 0);
    excess[s] = inf;
    for (int i = 0; i < n; i++) {
        if (i != s)
            push(s, i);
    }
    seen.assign(n, 0);

    while (!excess_vertices.empty()) {
        int u = excess_vertices.front();
        excess_vertices.pop();
        if (u != s && u != t)
            discharge(u);
    }

    for (int i = 0; i < n; i++){
        std::cout << i << " e " << excess[i] << " h " << height[i] << std::endl;
    }

    std::cout << "caps/flows:\n";
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            std::cout << std::setw(2) << std::left << capacity[i][j]  << "/" << std::setw(4) << std::left << flow[i][j];
        }
        std::cout << std::endl;
    }
    std::vector<std::tuple<int, int>> ret{};

    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            if(capacity[i][j]!= 0 && flow[i][j] == capacity[i][j]){
                ret.push_back({i, j});
            }
        }
    }
    return ret;

}


int main(){
    capacity.assign(numberOfNodes, vector<int>(numberOfNodes, 0));
    int source = 0;
    int to = numberOfNodes-1;
    //
    capacity[source][1] = 3;
    capacity[source][2] = 9;
    capacity[source][3] = 5;
    capacity[source][4] = 6;
    capacity[source][5] = 2;

    capacity[1][2] = 3;
    capacity[2][3] = 3;
    capacity[2][1] = 3;
    capacity[3][2] = 3;
    capacity[4][3] = 4;
    capacity[5][4] = 1;

    capacity[1][to] = 10;
    capacity[2][to] = 2;
    capacity[3][to] = 1;
    capacity[4][to] = 8;
    capacity[5][to] = 9;

    auto result = max_flow(source, to);
    std::cout << "Edges to remove are: " << std::endl;
    for(auto r : result){
        int from = std::get<0>(r);
        int to = std::get<1>(r);

        std::cout << from << " --> " << to << std::endl;
    }
}