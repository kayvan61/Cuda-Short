#include <iostream>
#include <stdio.h>
#include "CudaLock.hpp"

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
  int min = -1;
    
  if(globalThreadId < gSize) {
    for(int i = 0; i < gSize; i++) {
      if((adjMat[ind + i] < min && adjMat[ind + i] >= 0) || min <= -1) {
	min = adjMat[ind + i];
      }
    }
    if(min >= 0) {
      outVec[globalThreadId] = min;
    }
    else {
      outVec[globalThreadId] = -1;
    }    
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

	  int min   = d[i];
	  int other = d[globalThreadId] + adjMat[globalThreadId * gSize + i];
	  if(other >= 0 && min >= 0) {
	    d[i] = min < other ? min : other;
	  }
	  else {
	    d[i] = -1;
	  }
	  
	  lock.unlock();
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
      val2 = minOutEdges[pos2] + (useD ? d[pos2] : 0);;
      if(!useD) {
	if(val1 > val2 && val2 >= 0 && minOutEdges[pos2] >= 0) {
	  outDel[globalThreadId] = val2;	 
	}
	else if(val1 >= 0 && minOutEdges[pos1] >= 0){
	  outDel[globalThreadId] = val1;
	}
	else {
	  outDel[globalThreadId] = 0xFFFFFFFF;
	}
      }
      else if(U[pos1] && U[pos2]) {
	if(val1 > val2 && val2 >= 0 && minOutEdges[pos2] >= 0) {
	  outDel[globalThreadId] = val2;
	}
	else if(val1 >= 0 && minOutEdges[pos1] >= 0){
	  outDel[globalThreadId] = val1;
	}
	else {
	  outDel[globalThreadId] = 0xFFFFFFFF;
	}
      }
      else if(U[pos1] && val1 >= 0 && minOutEdges[pos1] >= 0) {
	outDel[globalThreadId] = val1;
      }
      else if(U[pos2] && val2 >= 0 && minOutEdges[pos2] >= 0) {
	outDel[globalThreadId] = val2;
      }
      else {
	outDel[globalThreadId] = 0xFFFFFFFF;
      }
    }
    else {
      if(outDel[globalThreadId] >= 0 && minOutEdges[pos1] >= 0) {
	outDel[globalThreadId] = val1;
      }
      else {
	outDel[globalThreadId] = 0xFFFFFFFF;
      }
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