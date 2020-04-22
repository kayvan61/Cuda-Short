#include <iostream>
#include "CudaLock.hpp"

__global__ void relax(int*, int*, int*, int, Lock*);
__global__ void min(int*, int*, int*);
__global__ void update(int*, int*, int*, int, int);
__global__ void init(int*, int*, int*, int);
__global__ void findAllMins(int**, int*, int);

int** genTestAdjMat();

int main() {
  
  Lock relaxLock;

  int** adjMat;
  int*  shortestOut;
  int   gSize = 5;
  int   startingNode = 0;

  int** _d_adjMat;
  int*  _d_outVec;
  int*  _d_unvisited;
  int*  _d_frontier;
  int*  _d_estimates;
  int*  _d_delta;
  int*  _d_minOutEdge;

  adjMat = genTestAdjMat();
  shortestOut = (int*)malloc(sizeof(int) * gSize);

  cudaMalloc((void**) &_d_adjMat,     sizeof(int) * gSize * gSize);
  cudaMalloc((void**) &_d_outVec,     sizeof(int) * gSize);
  cudaMalloc((void**) &_d_unvisited,  sizeof(int) * gSize);
  cudaMalloc((void**) &_d_frontier,   sizeof(int) * gSize);
  cudaMalloc((void**) &_d_estimates,  sizeof(int) * gSize);
  cudaMalloc((void**) &_d_minOutEdge, sizeof(int) * gSize);
  cudaMalloc((void**) &_d_delta,      sizeof(int));

  cudaMemcpy(_d_adjMat, adjMat, sizeof(int) * gSize * gSize, cudaMemcpyHostToDevice);
  cudaMemset(_d_outVec,      0, sizeof(int) * gSize);
  cudaMemset(_d_unvisited,   0, sizeof(int) * gSize);
  cudaMemset(_d_frontier,    0, sizeof(int) * gSize);
  cudaMemset(_d_estimates,   0, sizeof(int) * gSize);
  cudaMemset(_d_minOutEdge,  0, sizeof(int) * gSize);

  // O(n) but total algo is larger than O(n) so who cares
  findAllMins<<<1, gSize>>>(_d_adjMat, _d_minOutEdge, gSize);

  cudaMemcpy(shortestOut, _d_minOutEdge, sizeof(int) * gSize, cudaMemcpyDeviceToHost);

  for(int i = 0; i < 5; i++) {
    printf("%d ", shortestOut[i]);
  }

  /*
   * pseudo-code algo
   * d[i]   = shortest path so far from source to i
   * U      = unvisited verts
   * F      = frontier verts
   * del    = biggest d[i] (i from U) that we can add to frontier
   * del[i] = min(d[u] + del[u] for all u in U)  (ith iteration)
   * del[u] = minimum weight of its outgoing edges
   * 
   * init<<<>>>(U, F, d)
   * while(del != -1) 
   *   relax<<<>>>(U, F, d)
   *   del = min<<<>>>(U, d)
   *   update<<<>>>(U, F, d, del)
   */

  
  
}

// find min edge out
__global__ void findAllMins(int** adjMat, int* outVec, int gSize) {
  int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(globalThreadId < gSize) {
    int* curVec = &((int*)adjMat)[globalThreadId];
    int min = -1;

    for(int i = 0; i < gSize; i++) {
      if(curVec[i] > min || min == -1) {
	min = curVec[i];
      }
    }

    outVec[globalThreadId] = min;
  }
}

/*
 * if(F[tid]) 
 *   for all j that are successors of tid
 *     if(U[j])
 *       atomic_start
 *       d[j] = min(d[j], d[tid] + w[tid][j])
 *       atomic_end
 *
 */
__global__ void relax(int* U, int* F, int* d, int gSize, int** adjMat, Lock* lock) {
  int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

  if (globalThreadId < gSize) {
    if (F[globalThreadId]) {
      for (int i = 0; i < gSize; i++) {
	if(adjMat[globalThreadId][i] != -1) {
	  lock->lock();
	  int min   = d[i];
	  int other = d[globalThreadId] + adjMat[globalThreadId][i];
	  d[i] = min < other ? min : other;
	  lock->unlock();
	}
      }
    }
  }
}

__global__ void min(int* U, int* d, int* outDel);

/*
 * F[tid] = false
 * if(U[tid] and d[tid] < del)
 *   U[tid] = false
 *   F[tid] = true
 *
 */
__global__ void update(int* U, int* F, int* d, int del, int gSize) {
  int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

  if (globalThreadId < gSize) {
    F[globalThreadId] = 0;
    if(U[globalThreadId] && d[globalThreadId] < del) {
      U[globalThreadId] = 0;
      F[globalThreadId] = 1;
    }
  } 
}

/*
 * U[tid] = true
 * F[tid] = false
 * d[tid] = -1
 */
__global__ void init(int* U, int* F, int* d, int startNode, int gSize) {
  int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

  if (globalThreadId < gSize) {
    U[globalThreadId] = 1;
    F[globalThreadId] = 0;
    d[globalThreadId] = -1;
  }

  if(globalThreadId == 0) {
    d[globalThreadId] = 0;
  }
}

int** genTestAdjMat() {
  int** ret = (int**)malloc(25 * sizeof(int));

  ret[0][0] = 2;
  ret[0][1] = 1;
  ret[0][2] = 0;
  ret[0][3] = 1;
  ret[0][4] = 2;

  ret[1][0] = 7;
  ret[1][1] = 6;
  ret[1][2] = 5;
  ret[1][3] = 6;
  ret[1][4] = 1;

  ret[2][0] = 10;
  ret[2][1] = 10;
  ret[2][2] = 10;
  ret[2][3] = 10;
  ret[2][4] = 2;

  ret[3][0] = 11;
  ret[3][1] = 3;
  ret[3][2] = 11;
  ret[3][3] = 10;
  ret[3][4] = 5;

  ret[4][0] = 6;
  ret[4][1] = 4;
  ret[4][2] = 6;
  ret[4][3] = 7;
  ret[4][4] = 8;

  return ret;
}
