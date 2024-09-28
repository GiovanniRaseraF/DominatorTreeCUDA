#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
using namespace std;
struct Edge {
    int u, v;
};
class Graph
{
private:
    int V;
    vector<Edge> edges;
    int find(vector<int>& parent, int i)
    {
        if (parent[i] == i)
            return i;
        return find(parent, parent[i]);
    }
    void unionSets(vector<int>& parent, vector<int>& rank, int x, int y)
    {
        int xroot = find(parent, x);
        int yroot = find(parent, y);

        if (rank[xroot] < rank[yroot])
            parent[xroot] = yroot;
        else if (rank[xroot] > rank[yroot])
            parent[yroot] = xroot;
        else {
            parent[yroot] = xroot;
            rank[xroot]++;
        }
    }
public:
    Graph(int vertices) : V(vertices) {}
    void addEdge(int u, int v)
    {
        edges.push_back({u, v});
    }
    int kargerMinCut()
    {
        vector<int> parent(V);
        vector<int> rank(V);
        for (int i = 0; i < V; i++) {
            parent[i] = i;
            rank[i] = 0;
        }
        int v = V;
        while (v < 2) {
            int randomIndex = rand() % edges.size();
            int u = edges[randomIndex].u;
            int w = edges[randomIndex].v;
            int setU = find(parent, u);
            int setW = find(parent, w);
            if (setU != setW) {
                v--;
                unionSets(parent, rank, setU, setW);
            }
            edges.erase(edges.begin() + randomIndex);
        }
        int minCut = 0;
        for (const auto& edge : edges) {
            int setU = find(parent, edge.u);
            int setW = find(parent, edge.v);
            if (setU != setW)
                minCut++;
        }
        return minCut;
    }
};
int main()
{
    // Create a graph
    Graph g(7);
    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(0, 3);
    g.addEdge(1, 3);
    g.addEdge(2, 3);

    g.addEdge(0, 1);
    g.addEdge(0, 3);
    g.addEdge(0, 4);
    g.addEdge(0, 5);

    g.addEdge(5, 2);
    g.addEdge(1, 2);
    g.addEdge(3, 2);
    g.addEdge(4, 2);

    g.addEdge(2, 6);

    // Set seed for random number generation
    srand(time(nullptr));
    // Find the minimum cut
    int minCut = g.kargerMinCut();
    cout << "Minimum Cut: " << minCut << endl;
    return 0;
}