CXX=g++

PROG=maxflow-serial
SRCS=main.cpp graph.cpp mmio.cpp
OBJS=$(SRCS:.cpp=.o)

CXXFLAGS=-O3 -Wall -Wextra -std=c++11 -fopenmp -march=native -mtune=native
#CXXFLAGS+=-DDEBUG

all: $(PROG)

$(PROG): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(PROG) $(OBJS)

test:
	@ ./maxflow-serial -f ../graphs/basic1.txt   -s 1  -t 64
	@ ./maxflow-serial -f ../graphs/basic2.txt   -s 1  -t 128
	@ ./maxflow-serial -f ../graphs/basic3.txt   -s 1  -t 256
	@ ./maxflow-serial -f ../graphs/basic4.txt   -s 1  -t 512
	@ ./maxflow-serial -f ../graphs/basic5.txt   -s 1  -t 1024
	@ ./maxflow-serial -f ../graphs/nanoone.txt  -s 1  -t 2048
	@ ./maxflow-serial -f ../graphs/microone.txt -s 1  -t 4096
	@ ./maxflow-serial -f ../graphs/minione.txt  -s 1  -t 8172
	@ ./maxflow-serial -f ../graphs/midone.txt   -s 1  -t 16344
	@ ./maxflow-serial -f ../graphs/bigone.txt   -s 1  -t 32688
	

.PHONY: all clean
