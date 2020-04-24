#include <iostream>
#include <stdio.h>
#include <climits>

#define BLOCK_SIZE 512

/* 
 * Definitions:
 * d[i]   = shortest path so far from source to i
 * U      = unvisited verts
 * F      = frontier verts
 * del    = biggest d[i] (i from U) that we can add to frontier
 * del[i] = min(d[u] + del[u] for all u in U)  (ith iteration)
 * del[u] = minimum weight of its outgoing edges
 *
 */

// find min edge out
__global__ void findAllMins(int* adjMat, int* outVec, int gSize) {
  int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
  int ind = globalThreadId * gSize;
  int min = INT_MAX;
    
  if(globalThreadId < gSize) {
    for(int i = 0; i < gSize; i++) {
      if(adjMat[ind + i] < min && adjMat[ind + i] > 0) {
	min = adjMat[ind + i];
      }
    }
    outVec[globalThreadId] = min;
  }
}

/*
 * forall i in parallel do 
 *   if (F[i]) 
 *     forall j predecessor of i do
 *       if (U[j])
 *         c[i]= min(c[i],c[j]+w[j,i]);
 *
 */
__global__ void relax(int* U, int* F, int* d, int gSize, int* adjMat) {
  int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

  if (globalThreadId < gSize) {
    if (F[globalThreadId]) {
      for (int i = 0; i < gSize; i++) {
	if(adjMat[globalThreadId*gSize + i] && i != globalThreadId && U[i]) {
	  atomicMin(&d[i], d[globalThreadId] + adjMat[globalThreadId * gSize + i]);
	}
      }
    }
  }
}

__global__ void min(int* U, int* d, int* outDel, int* minOutEdges, int gSize, int useD) {
  int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
  
  int pos1 = 2*globalThreadId;
  int pos2 = 2*globalThreadId + 1;
  int val1, val2;
  if(pos1 < gSize) {
    val1 = minOutEdges[pos1] + (useD ? d[pos1] : 0);
    if(pos2 < gSize) {
      val2 = minOutEdges[pos2] + (useD ? d[pos2] : 0);
      
      val1 = val1 <= 0 ? INT_MAX : val1;
      val2 = val2 <= 0 ? INT_MAX : val2;
      if(useD) {
	val1 = U[pos1] ? val1 : INT_MAX;
	val2 = U[pos2] ? val2 : INT_MAX;
      }
      if(val1 > val2) {
	outDel[globalThreadId] = val2;	 
      }
      else{
	outDel[globalThreadId] = val1;
      }
    }
    else {
      val1 = val1 <= 0 ? INT_MAX : val1;
      if(useD) {
	val1 = U[pos1] ? val1 : INT_MAX;
      }
      outDel[globalThreadId] = val1;
    }
  }
}

/*
 * F[tid] = false
 * if(U[tid] and d[tid] < del)
 *   U[tid] = false
 *   F[tid] = true
 *
 */
__global__ void update(int* U, int* F, int* d, int* del, int gSize) {
  int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (globalThreadId < gSize) {
    F[globalThreadId] = 0;
    if(U[globalThreadId] && d[globalThreadId] < del[0]) {
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
    d[globalThreadId] = INT_MAX;
  }

  if(globalThreadId == 0) {
    d[globalThreadId] = 0;
    U[globalThreadId] = 0;
    F[globalThreadId] = 1;
  }
}

void doShortest(int* adjMat, int* shortestOut, int gSize, int startingNode,
		int*  _d_adjMat,
		int*  _d_outVec,
		int*  _d_unvisited,
		int*  _d_frontier,
		int*  _d_estimates,
		int*  _d_delta,
		int*  _d_minOutEdge) {
  int   del;
  int numBlocks  = (gSize / BLOCK_SIZE) + 1;
  
  // O(n) but total algo is larger than O(n) so who cares
  findAllMins<<<numBlocks, BLOCK_SIZE>>>(_d_adjMat, _d_minOutEdge, gSize);

  /*
   * pseudo-code algo
   *
   * init<<<>>>(U, F, d)
   * while(del != -1) 
   *   relax<<<>>>(U, F, d)
   *   del = min<<<>>>(U, d)
   *   update<<<>>>(U, F, d, del)
   */

  int  curSize = gSize;
  int  dFlag;
  int* _d_minTemp1;
  int* _d_minTemp2;
  int* tempDebug = (int*) malloc(sizeof(int) * gSize);
  
  cudaMalloc((void**) &_d_minTemp1 , sizeof(int) * gSize);
  
  init<<<numBlocks, BLOCK_SIZE>>>(_d_unvisited, _d_frontier, _d_estimates, startingNode, gSize);

  do {
    dFlag = 1;
    curSize = gSize;
    cudaMemcpy(_d_minTemp1,   _d_minOutEdge, sizeof(int) * gSize, cudaMemcpyDeviceToDevice);

    relax<<<numBlocks, BLOCK_SIZE>>>(_d_unvisited, _d_frontier, _d_estimates, gSize, _d_adjMat);

    do {
      min<<<numBlocks, BLOCK_SIZE>>>(_d_unvisited, _d_estimates, _d_delta, _d_minTemp1, curSize, dFlag);
      _d_minTemp2 = _d_minTemp1;
      _d_minTemp1 = _d_delta;
      _d_delta    = _d_minTemp2;

      curSize /= 2;
      dFlag = 0;
    } while (curSize > 0);
    
    _d_minTemp2 = _d_minTemp1;
    _d_minTemp1 = _d_delta;
    _d_delta    = _d_minTemp2;
    
    update<<<numBlocks, BLOCK_SIZE>>>(_d_unvisited, _d_frontier, _d_estimates, _d_delta, gSize);
    
    cudaMemcpy(&del, _d_delta, sizeof(int), cudaMemcpyDeviceToHost);
    
  } while(del != INT_MAX);
  
  cudaMemcpy(shortestOut, _d_estimates, sizeof(int) * gSize, cudaMemcpyDeviceToHost);
  
#ifndef NO_PRINT
  for(int i = 0; i < gSize; i++){
    printf("shotest path from %d to %d is %d long.\n", startingNode, i, shortestOut[i]);
  }
  printf("\n");
#endif

  cudaFree(_d_minTemp1);
}
