
GCC=g++
CXX_FLAGS=-Wall

all: graph.o main.o
	$(GCC) $(CXX_FLAGS) $^ -o graph

%.o: %.cpp
	$(GCC) -c $< -o $@

clean:
	rm *.o graph
