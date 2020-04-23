#define TIMING_MAX_SIZE  8192
#define TIMING_STEP_SIZE 512

int* genRandAdjMat(int size) {
  int* ret = (int*)malloc(size * size * sizeof(int));
  srand(12345);
  for(int i = 0; i < size * size; i++) {
    ret[i] = rand()%256 - 1;
  } 

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
  int   gSize = TIMING_MAX_SIZE;
  int   startingNode = 0;
  
  int*  _d_adjMat;
  int*  _d_outVec;
  int*  _d_unvisited;
  int*  _d_frontier;
  int*  _d_estimates;
  int*  _d_delta;
  int*  _d_minOutEdge;

  adjMat = genRandAdjMat(TIMING_MAX_SIZE);
  
  for(gSize = TIMING_STEP_SIZE; gSize <= TIMING_MAX_SIZE; gSize += TIMING_STEP_SIZE) {
    
    shortestOut = (int*)malloc(sizeof(int) * gSize);

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
    cudaMemset((void*)_d_delta,              0, sizeof(int) * gSize);
    
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
    
    free(shortestOut);
    cudaFree(_d_adjMat);
    cudaFree(_d_outVec);
    cudaFree(_d_unvisited);
    cudaFree(_d_frontier);
    cudaFree(_d_estimates);
    cudaFree(_d_minOutEdge);
    cudaFree(_d_delta);

    handlerError(cudaDeviceSynchronize());
    handlerError(cudaEventRecord(mem_stop, 0));
    handlerError(cudaEventSynchronize(mem_stop));
    handlerError(cudaEventElapsedTime(&mem_gpu_time, mem_start, mem_stop));

    printf("%d size graph took : %f ms running, %f ms including memory manage\n", gSize, gpu_time, mem_gpu_time);

    handlerError(cudaEventDestroy(stop));
    handlerError(cudaEventDestroy(start));
    handlerError(cudaEventDestroy(mem_stop));
    handlerError(cudaEventDestroy(mem_start));
  }

  free(adjMat);
}

void runCPUTimingTest() {

  int** adjMat = (int**)malloc(sizeof(int*) * TIMING_MAX_SIZE);
  
  for(int i = 0; i < TIMING_MAX_SIZE; i++) {
    adjMat[i] = (int*)malloc(sizeof(int*) * TIMING_MAX_SIZE);
    for(int j = 0; j < TIMING_MAX_SIZE; j++) {
      adjMat[i][j] = rand()%256 - 1;
    }
  }

  clock_t start, end;
  
  for(int gSize = TIMING_STEP_SIZE; gSize <= TIMING_MAX_SIZE; gSize += TIMING_STEP_SIZE) {
    start = clock();
    
    CPU_Dijkstra(adjMat, 0, gSize);

    end = clock();

    printf("Size: %d, CPU Time: %f ms\n", gSize, (double)(end - start) / CLOCKS_PER_SEC * 1000);
  }

  for(int i = 0; i < TIMING_MAX_SIZE; i++) {
    free(adjMat[i]);
  }
  free(adjMat);
}
