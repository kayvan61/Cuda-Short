#include "graph.hpp"

WeightedGraph::WeightedGraph(int size) : adjMat(size, std::vector<int>(size, 0)) {}

WeightedGraph::WeightedGraph(int** data, int size) : adjMat(size, std::vector<int>(size, 0)) {
  for(int i = 0; i < size; i++) {
    for(int j = 0; j < size; j++) {
      adjMat[i][j] = data[i][j];
    }
  }
}

void WeightedGraph::randomize() {
  int size  = adjMat.size();
  for(int i = 0; i < size; i++) {
    for(int j = i; j < size; j++) {
      if(i == j) { continue; } // no self pointing nodes
      int val = rand() % 255;
      adjMat[i][j] = val;
      adjMat[j][i] = val;
    }
  }
}

int WeightedGraph::isAdj(int node1, int node2) const {
  return adjMat[node1][node2];
}

int WeightedGraph::getNodeCount() const {
  return adjMat.size();
}

const std::vector<std::vector<int>>& WeightedGraph::getRawAdjMat() const {
  return adjMat;
}

std::ostream& operator<< (std::ostream& out, const WeightedGraph& d) {
  for(int i = 0; i < d.getNodeCount(); i++) {
      for(int j = 0; j < d.getNodeCount(); j++) {
	out << d.getRawAdjMat()[i][j] << ", ";
      }
      out << std::endl;
  }
  return out;
}
