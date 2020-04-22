#include <iostream>
#include <stdio.h>
#include "CudaLock.hpp"

__global__ void relax(int* U, int* F, int* d, int gSize, int* adjMat, Lock lock);
__global__ void min(int*, int*, int*, int*, int, int);
__global__ void update(int*, int*, int*, int*, int);
__global__ void init(int*, int*, int*, int, int);
__global__ void findAllMins(int*, int*, int);

int* genTestAdjMat();

int main() {
  
  Lock relaxLock;

  int*  adjMat;
  int*  shortestOut;
  int   gSize = 5;
  int   startingNode = 0;
  int   del;
  
  int*  _d_adjMat;
  int*  _d_outVec;
  int*  _d_unvisited;
  int*  _d_frontier;
  int*  _d_estimates;
  int*  _d_delta;
  int*  _d_minOutEdge;

  adjMat      = genTestAdjMat();
  shortestOut = (int*)malloc(sizeof(int) * gSize);

  cudaMalloc((void**) &_d_adjMat,     sizeof(int) * gSize * gSize);
  cudaMalloc((void**) &_d_outVec,     sizeof(int) * gSize);
  cudaMalloc((void**) &_d_unvisited,  sizeof(int) * gSize);
  cudaMalloc((void**) &_d_frontier,   sizeof(int) * gSize);
  cudaMalloc((void**) &_d_estimates,  sizeof(int) * gSize);
  cudaMalloc((void**) &_d_minOutEdge, sizeof(int) * gSize);
  cudaMalloc((void**) &_d_delta,      sizeof(int) * gSize);
  
  cudaMemcpy((void*)_d_adjMat, (void*)adjMat, sizeof(int) * gSize * gSize, cudaMemcpyHostToDevice);
  cudaMemset((void*)_d_outVec,             0, sizeof(int) * gSize);
  cudaMemset((void*)_d_unvisited,          0, sizeof(int) * gSize);
  cudaMemset((void*)_d_frontier,           0, sizeof(int) * gSize);
  cudaMemset((void*)_d_estimates,          0, sizeof(int) * gSize);
  cudaMemset((void*)_d_minOutEdge,         0, sizeof(int) * gSize);

  // O(n) but total algo is larger than O(n) so who cares
  findAllMins<<<1, gSize>>>(_d_adjMat, _d_minOutEdge, gSize);

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

  int  curSize = gSize;
  int  dFlag;
  int* _d_minTemp1;
  int* _d_minTemp2;
  int* tempDebug = (int*) malloc(sizeof(int) * gSize);
  
  cudaMalloc((void**) &_d_minTemp1 , sizeof(int) * gSize);
  
  init<<<1, gSize>>>(_d_unvisited, _d_frontier, _d_estimates, startingNode, gSize);
  do {
    printf("start.\n");
    fflush(stdout);
    
    dFlag = 1;
    curSize = gSize;
    cudaMemcpy(_d_minTemp1,   _d_minOutEdge, sizeof(int) * gSize, cudaMemcpyDeviceToDevice);
    
    relax<<<gSize, 1>>>(_d_unvisited, _d_frontier, _d_estimates, gSize, _d_adjMat, relaxLock);
    cudaDeviceSynchronize();
    printf("done with relax kernel.\n");
    fflush(stdout);

    do {
      min<<<1, gSize>>>(_d_unvisited, _d_estimates, _d_delta, _d_minTemp1, curSize, dFlag);
      cudaDeviceSynchronize();
      _d_minTemp2 = _d_minTemp1;
      _d_minTemp1 = _d_delta;
      _d_delta    = _d_minTemp2;

      curSize /= 2;
      dFlag = 0;
      printf("in min loop...\n");
      fflush(stdout);
    } while (curSize > 0);
    
    _d_minTemp2 = _d_minTemp1;
    _d_minTemp1 = _d_delta;
    _d_delta    = _d_minTemp2;
    
    update<<<1, gSize>>>(_d_unvisited, _d_frontier, _d_estimates, _d_delta, gSize);
    cudaDeviceSynchronize();
    printf("done with update kernel.\n");
    fflush(stdout);
    
    cudaMemcpy(&del, _d_delta, sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaMemcpy(tempDebug, _d_estimates, sizeof(int) * gSize, cudaMemcpyDeviceToHost);
    for(int i = 0; i < gSize; i++){
      printf("%d ", tempDebug[i]);
    }
    printf("\n");
    fflush(stdout);
  } while(del != 0xFFFFFFFF);

  
  cudaFree(_d_minTemp1);
  cudaFree(_d_adjMat);
  cudaFree(_d_outVec);
  cudaFree(_d_unvisited);
  cudaFree(_d_frontier);
  cudaFree(_d_estimates);
  cudaFree(_d_minOutEdge);
  cudaFree(_d_delta);

  
}

// find min edge out
__global__ void findAllMins(int* adjMat, int* outVec, int gSize) {
  int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
  int ind = globalThreadId * gSize;
  int min = -1;
    
  if(globalThreadId < gSize) {
    for(int i = 0; i < gSize; i++) {
      if(adjMat[ind + i] < min || min == -1) {
	min = adjMat[ind + i];
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
__global__ void relax(int* U, int* F, int* d, int gSize, int* adjMat, Lock lock) {
  int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

  if (globalThreadId < gSize) {
    if (F[globalThreadId]) {
      for (int i = 0; i < gSize; i++) {
	if(adjMat[globalThreadId*gSize + i] != -1) {
	  lock.lock(globalThreadId);
	  
	  printf("gSize: %d, i: %d, globalThreadId: %d LOCK\n", gSize, i, globalThreadId);	 

	  int min   = d[i];
	  int other = d[globalThreadId] + adjMat[globalThreadId * gSize + i];
	  d[i] = min < other ? min : other;
	  
	  printf("i: %d, globalThreadId: %d UNLOCK\n", i, globalThreadId);
	  
	  lock.unlock();
	}
      }
    } else {
      printf("globalThreadId: %d NOT FRONTIER\n", globalThreadId);	 
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
      val2 = minOutEdges[pos2] + (useD ? d[pos2] : 0);;
      if(!useD) {
	if(val1 > val2) {
	  outDel[globalThreadId] = val2;	 
	}
	else {
	  outDel[globalThreadId] = val1;
	}
      }
      else if(U[pos1] && U[pos2]) {
	if(val1 > val2) {
	  outDel[globalThreadId] = val2;
	}
	else {
	  outDel[globalThreadId] = val1;
	}
      }
      else if(U[pos1]) {
	outDel[globalThreadId] = val1;
      }
      else if(U[pos2]) {
	outDel[globalThreadId] = val2;
      }
      else {
	outDel[globalThreadId] = 0xFFFFFFFF;
      }
    }
    else {
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
    if(U[globalThreadId] && d[globalThreadId] <= del[0]) {
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
    d[globalThreadId] = 0x7FFFFFFF;
  }

  if(globalThreadId == 0) {
    d[globalThreadId] = 0;
    U[globalThreadId] = 0;
    F[globalThreadId] = 1;

  }
}

int* genTestAdjMat() {
  int* ret = (int*)malloc(25 * sizeof(int));

  ret[0] = 2;
  ret[1] = 1;
  ret[2] = 0;
  ret[3] = 1;
  ret[4] = 2;

  ret[5] = 7;
  ret[6] = 6;
  ret[7] = 5;
  ret[8] = 6;
  ret[9] = 1;

  ret[10] = 10;
  ret[11] = 10;
  ret[12] = 10;
  ret[13] = 10;
  ret[14] = 2;

  ret[15] = 11;
  ret[16] = 3;
  ret[17] = 11;
  ret[18] = 10;
  ret[19] = 5;

  ret[20] = 6;
  ret[21] = 4;
  ret[22] = 6;
  ret[23] = 7;
  ret[24] = 8;

  return ret;
}
