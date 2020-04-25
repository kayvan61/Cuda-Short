#include <iostream>
#include <stdio.h>
#include <sys/time.h>
#include <curand.h>
#include <curand_kernel.h>
#include "shortestKernels.cu"
#ifdef TIMING
#include "CPU_short.hpp"
#include "timingTests.hpp"
#endif
#ifdef FUNC_TEST
#include "CPU_short.hpp"
#include "funcTests.hpp"
#endif



int* genTestAdjMat(int*);
void runTimingTest();
void runCPUTimingTest();
int* read_input(const char* input);

int main() {

#ifdef TIMING
  runTimingTest();
  runCPUTimingTest();
  return 0;
#endif

#ifdef FUNC_TEST
  runFuncTests();
  return 0;
#endif

  int*  adjMat;
  int*  shortestOut;
  int   gSize;
  int   startingNode = 0;

  int*  _d_adjMat;
  int*  _d_outVec;
  int*  _d_unvisited;
  int*  _d_frontier;
  int*  _d_estimates;
  int*  _d_delta;
  int*  _d_minOutEdge;

  adjMat      = genTestAdjMat(&gSize);
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

  doShortest( adjMat,
	      shortestOut,
	      gSize,
	      startingNode,
	      _d_adjMat,
	      _d_outVec,
	      _d_unvisited,
	      _d_frontier,
	      _d_estimates,
	      _d_delta,
	      _d_minOutEdge);

  cudaFree(_d_adjMat);
  cudaFree(_d_outVec);
  cudaFree(_d_unvisited);
  cudaFree(_d_frontier);
  cudaFree(_d_estimates);
  cudaFree(_d_minOutEdge);
  cudaFree(_d_delta);

  free(adjMat);
  free(shortestOut);
}

int* genTestAdjMat(int* gSize) {
  *gSize = 5;
  int temp[49] = {0, 10, 0, 5, 0,
		  0, 0, 1, 2, 0,
		  0, 0, 0, 0, 4,
		  0, 3, 9, 0, 2,
		  7, 0, 6, 0, 0};
  for(int i = 0; i < 49; i++) {
    if(temp[i] == 0) {
      temp[i] = -1;
    }
  }

  int* ret = (int*)malloc(49 * sizeof(int));

  memcpy(ret, temp, sizeof(int) * 49);

  return ret;
}

int* read_input(const char* input) {
    int arraySize = 0;
    int num;
    int index = 0;
    FILE* inputF = fopen(input, "r");
    while (fscanf(inputF, "%d, ", &num) != EOF) {
        arraySize++;
    }

    int* in = (int*)malloc(arraySize * sizeof(int));
    rewind(inputF);
    while (fscanf(inputF, "%d, ", &in[index]) != EOF) {
        index++;
    }

    for (int i = 0; i < arraySize; i++) {
        printf("%d ", in[i]);
    }

    return in;
}
