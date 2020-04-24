#include <cassert>

#define TEST_COUNT 100

int assertEqArr(int* arr1, int* arr2, int size) {
  for(int i = 0; i < size; i++) {
    if(arr1[i] != arr2[i]) {
      printf("Node %d differs. GPU: %d, CPU: %d\n", i, arr1[i], arr2[i]);
      return i;
    }
  }
  return -1;
}

void printArr(int *arr, int size) {
  for(int i = 0; i < size; i++) {
    printf("%d ", arr[i]);
  }
  printf("\n");
}

int* genRandAdjMat(int size) {
  int* ret = (int*)malloc(size * size * sizeof(int));
  srand(12345);
  for(int i = 0; i < size * size; i++) {
    ret[i] = rand()%512;
  } 

  return ret;
}

void runFuncTests() {
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
  
  for(int i = 0; i < TEST_COUNT; i++) {

    gSize = rand() % 8196 + 1;
    adjMat = genRandAdjMat(gSize);
    
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
    cudaMemset((void*)_d_delta,              0, sizeof(int) * gSize);
    
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

    cudaDeviceSynchronize();


    int** CPUadjMat      = (int**)malloc(sizeof(int*) * gSize);
    int*  CPUshortestOut = (int*) malloc(sizeof(int*) * gSize);
    
    for(int i = 0; i < gSize; i++) {
      CPUadjMat[i] = (int*)malloc(sizeof(int*) * gSize);
      for(int j = 0; j < gSize; j++) {
	CPUadjMat[i][j] = adjMat[i * gSize + j];
      }
    }

    CPU_Dijkstra(CPUadjMat, 0, gSize, CPUshortestOut);
    int temp;
    if ((temp = assertEqArr(shortestOut, CPUshortestOut, gSize)) == -1) {
      printf("Yay! Correct for random graph of size %d!\n", gSize);
    }
    else {
      printf("GPU: ");
      printArr(shortestOut, 100);
      printf("CPU: ");
      printArr(CPUshortestOut, 100);
      printf("adjMat0: ");
      printArr(adjMat, 100);
      printf("adjMat%d: ", temp);
      printArr(adjMat + (temp*gSize), 100);
      exit(-1);
    }

    for(int i = 0; i < gSize; i++) {
      free(CPUadjMat[i]);
    }
    free(CPUadjMat);

    free(adjMat);
    cudaFree(_d_adjMat);
    cudaFree(_d_outVec);
    cudaFree(_d_unvisited);
    cudaFree(_d_frontier);
    cudaFree(_d_estimates);
    cudaFree(_d_minOutEdge);
    cudaFree(_d_delta);
  
    cudaDeviceSynchronize();
  }
}
