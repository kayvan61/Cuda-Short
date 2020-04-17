/*
 * representation of a graph in memory
 * implemented using an adjacency matrix
 * most for bundling some potentially useful
 * anything marked const safe is thread safe too
 */
#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <iostream>
#include <vector>

class WeightedGraph {
  std::vector<std::vector<int>> adjMat;
  
 public:
  WeightedGraph(int);          // random init with node count decided
  WeightedGraph(int**, int);   // init to graph from elsewhere
  ~WeightedGraph() = default; // no heap data need to free

  void randomize();           // generate random values for adjMat
  int isAdj(int, int) const;  // return weight between two nodes
  int getNodeCount() const;   // return number of nodes

  const std::vector<std::vector<int>>& getRawAdjMat() const; // return raw reference to adj mat

  friend std::ostream& operator<< (std::ostream&, const WeightedGraph&);
};

#endif
