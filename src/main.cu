#include <iostream>
#include <stdio.h>
#include <sys/time.h>
#include <curand.h>
#include <curand_kernel.h>
#include "CudaLock.hpp"
#include "shortestKernels.cu"

#define BLOCK_SIZE 512
#define TIMING
#define TIMING_MAX_SIZE  4096 * 1024
#define TIMING_STEP_SIZE 512

int* genTestAdjMat(int*);
void doShortest(int* adjMat,
		int* shortestOut,
		int gSize,
		int startingNode,
		int*  _d_adjMat,
		int*  _d_outVec,
		int*  _d_unvisited,
		int*  _d_frontier,
		int*  _d_estimates,
		int*  _d_delta,
		int*  _d_minOutEdge);
void runTimingTest();

int main() {

#ifdef TIMING
  runTimingTest();
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

void doShortest(int* adjMat, int* shortestOut, int gSize, int startingNode,
		int*  _d_adjMat,
		int*  _d_outVec,
		int*  _d_unvisited,
		int*  _d_frontier,
		int*  _d_estimates,
		int*  _d_delta,
		int*  _d_minOutEdge) {
  int   del;
  Lock relaxLock;
  
  // O(n) but total algo is larger than O(n) so who cares
  findAllMins<<<1, gSize>>>(_d_adjMat, _d_minOutEdge, gSize);

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
  
  init<<<1, gSize>>>(_d_unvisited, _d_frontier, _d_estimates, startingNode, gSize);
  int numBlocks  = (gSize / BLOCK_SIZE) + 1;
  do {
    dFlag = 1;
    curSize = gSize;
    cudaMemcpy(_d_minTemp1,   _d_minOutEdge, sizeof(int) * gSize, cudaMemcpyDeviceToDevice);
    
    relax<<<gSize, 1>>>(_d_unvisited, _d_frontier, _d_estimates, gSize, _d_adjMat, relaxLock);
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
    
    fflush(stdout);
  } while(del != 0xFFFFFFFF);

  cudaMemcpy(shortestOut, _d_estimates, sizeof(int) * gSize, cudaMemcpyDeviceToHost);
  
#ifndef TIMING
  for(int i = 0; i < gSize; i++){
    printf("shotest path from %d to %d is %d long.\n", startingNode, i, shortestOut[i]);
  }
  printf("\n");
#endif

  cudaFree(_d_minTemp1);
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

__global__ void cudaRandArr(int* a, int s) {
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;


  
  if(gtid < s) {
    curandState_t state;

    /* we have to initialize the state */
    curand_init(0, /* the seed controls the sequence of random values that are produced */
		0, /* the sequence number is only important with multiple cores */
		0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
		&state);

    a[gtid] = curand(&state) % 256 - 1;
  }
}

int* genRandAdjMat(int size) {
  int* ret = (int*)malloc(size * size * sizeof(int));

  int* _d_temp;
  
  cudaMalloc((void**) &_d_temp, sizeof(int) * size * size);
  
  cudaRandArr<<<(size / BLOCK_SIZE) + 1, BLOCK_SIZE>>>(_d_temp, size*size);

  cudaMemcpy(ret, _d_temp, sizeof(int) * size * size, cudaMemcpyDeviceToHost);
  cudaFree(_d_temp);
  
  return ret;
}

void handlerError(cudaError c) {
  if(c == cudaSuccess) {
    return;
  }
  printf("ERROR FATAL: %d\n", c);
}

void runTimingTest() {
  srand(1500);
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

  for(gSize = TIMING_STEP_SIZE; gSize <= TIMING_MAX_SIZE; gSize += TIMING_STEP_SIZE) {
    struct timeval cpu_stop, cpu_start;
    gettimeofday(&cpu_start, NULL);
    
    adjMat      = genRandAdjMat(gSize);
    shortestOut = (int*)malloc(sizeof(int) * gSize);
    
    gettimeofday(&cpu_stop, NULL);
    printf("took %lu ms\n", ((cpu_stop.tv_sec - cpu_start.tv_sec) * 1000000 + cpu_stop.tv_usec - cpu_start.tv_usec) / 1000); 

    cudaEvent_t mem_start, mem_stop;
    float mem_gpu_time = 0.0f;
    handlerError(cudaEventCreate(&mem_start));
    handlerError(cudaEventCreate(&mem_stop));
    handlerError(cudaEventRecord(mem_start, 0));
    
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
    
    cudaEvent_t start, stop;
    float gpu_time = 0.0f;
    handlerError(cudaEventCreate(&start));
    handlerError(cudaEventCreate(&stop));
    handlerError(cudaEventRecord(start, 0));
    
    doShortest(adjMat,
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

    handlerError(cudaDeviceSynchronize());
    handlerError(cudaEventRecord(stop, 0));
    handlerError(cudaEventSynchronize(stop));
    handlerError(cudaEventElapsedTime(&gpu_time, start, stop));    
    
    cudaFree(_d_adjMat);
    cudaFree(_d_outVec);
    cudaFree(_d_unvisited);
    cudaFree(_d_frontier);
    cudaFree(_d_estimates);
    cudaFree(_d_minOutEdge);
    cudaFree(_d_delta);
    free(adjMat);
    free(shortestOut);

    handlerError(cudaDeviceSynchronize());
    handlerError(cudaEventRecord(mem_stop, 0));
    handlerError(cudaEventSynchronize(mem_stop));
    handlerError(cudaEventElapsedTime(&mem_gpu_time, mem_start, mem_stop));

    printf("%d size graph took : %f ms running, %f ms memory manage\n", gSize, gpu_time, mem_gpu_time);

    handlerError(cudaEventDestroy(stop));
    handlerError(cudaEventDestroy(start));
    handlerError(cudaEventDestroy(mem_stop));
    handlerError(cudaEventDestroy(mem_start));
  }
}
