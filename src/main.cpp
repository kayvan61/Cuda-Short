#include "graph.hpp"
#include <iostream>

__global__ void shortestPathCuda(int **adjMat, int graphSize, int startNode, int* shortestOut) {
  int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
}

int** genTestAdjMat();

int main() {
  int** adjMat;
  int* shortestOut;
  int gSize;
  int startingNode;

  int** _d_adjMat;
  int*  _d_outVec;
  int*  _d_checkedNodes;
  
}

int** genTestAdjMat() {
  int** ret = malloc(25 * sizeof(int));

  for(int i = 0; i < 5; i++) {
    for(int j = i; j  < 5; j++)  {
      ret[i][j] = 5;
      ret[j][i] = 5;
    }
  }

  return ret;
}
