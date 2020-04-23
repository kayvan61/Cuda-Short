
GCC=nvcc
CXX_FLAGS=-arch compute_60
TIMING_FLAGS=-D TIMING -D NO_PRINT
FUNC_FLAGS=-D FUNC_TEST -D NO_PRINT 

demo: main.cu shortestKernels.cu CudaLock.hpp
	$(GCC) $(CXX_FLAGS) $< -o $@

timeTest: main.cu shortestKernels.cu CudaLock.hpp CPU_short.hpp timingTests.hpp
	$(GCC) $(CXX_FLAGS) $(TIMING_FLAGS) $< -o $@

funcTests: main.cu shortestKernels.cu CudaLock.hpp CPU_short.hpp funcTests.hpp
	$(GCC) $(CXX_FLAGS) $(FUNC_FLAGS) $< -o $@

%.o: %.cpp
	$(GCC) -c $< -o $@

clean:
	rm *.o graph
