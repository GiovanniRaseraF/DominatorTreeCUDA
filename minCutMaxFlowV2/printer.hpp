// Author: Giovanni Rasera
// Use: just for debugging info

#pragma once

#include <iostream>
#include <iomanip>

void print(
            int * offsets,int * Roffsets,
            int *destinations,int *Rdestinations,
            int *capacities,int *Rcapacities,
            int *flowIndex,int * heights,
            int *forwardFlows,int *backwardFlows,
            int *excesses,int *excessTotal,int numNodes,int numEdges,int source,int to
        ){
            std::cout << std::setw(30) << std::left << "int offsets[numNodes+1]{";
            for (int i=0; i < numNodes + 1; i++) {
                printf("%d, ", offsets[i]);
            }
            printf("};\n");

            std::cout << std::setw(30) << std::left << "int rOffsets[numNodes+1]{";
            for (int i=0; i < numNodes + 1; i++) {
                printf("%d, ", Roffsets[i]);
            }
            printf("};\n");

            std::cout << std::setw(30) << std::left << ("int destinations[numEdges]{");
            for (int i=0; i < numEdges; i++) {
                printf("%d, ", destinations[i]);
            }
            printf("};\n");

            std::cout << std::setw(30) << std::left << ("int rDestinations[numEdges]{");
            for (int i=0; i < numEdges; i++) {
                printf("%d, ", Rdestinations[i]);
            }
            printf("};\n");

            std::cout << std::setw(30) << std::left << ("int capacities[numEdges]{");
            for (int i=0; i < numEdges; i++) {
                printf("%d, ", capacities[i]);
            }
            printf("};\n");

            std::cout << std::setw(30) << std::left << ("int rCapacities[numEdges]{");
            for (int i=0; i < numEdges; i++) {
                printf("%d, ", capacities[i]);
            }
            printf("};\n");

            std::cout << std::setw(30) << std::left << ("int flowIndex[numEdges]{");
            for (int i=0; i < numEdges; i++) {
                printf("%d, ", flowIndex[i]);
            }
            printf("};\n");

            std::cout << std::setw(30) << std::left << ("int heights[numNodes]{");
            for (int i=0; i < numNodes; i++) {
                printf("%d, ", heights[i]);
            }
            printf("};\n");

            std::cout << std::setw(30) << std::left << ("int forwardFlow[numEdges]{");
            for (int i=0; i < numEdges; i++) {
                printf("%d, ", forwardFlows[i]);
            }
            printf("};\n");

            std::cout << std::setw(30) << std::left << ("int backwardFlows[numEdges]{");
            for (int i=0; i < numEdges; i++) {
                printf("%d, ", backwardFlows[i]);
            }
            printf("};\n");

            std::cout << std::setw(30) << std::left << ("int excesses[numNodes]{");
            for (int i=0; i < numNodes; i++) {
                printf("%d, ", excesses[i]);
            }
            printf("};\n");

            std::cout << std::setw(30) << std::left << ("int excessTotal[1]{%d};\n", *excessTotal);
        }