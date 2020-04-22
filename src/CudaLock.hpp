#ifndef LOCK_HPP
#define LOCK_HPP

struct Lock{
  int *mutex;
  Lock(void){
    int state = 0;
    cudaMalloc((void**) &mutex, sizeof(int));
    cudaMemcpy(mutex, &state, sizeof(int), cudaMemcpyHostToDevice);
  }

  Lock(const Lock& other) {
    this->mutex = other.mutex;
  }
  
  ~Lock(void){
    cudaFree(mutex);
  }
  __device__ void lock(int i){
    while(atomicCAS(mutex, 0, 1) != 0);
  }
  __device__ void unlock(void){
    atomicExch(mutex, 0);
  }
};

#endif
