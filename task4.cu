#include <stdio.h>
#include <iostream>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_reduce.cuh>

#define ID(i ,j, ld) (((i)*(ld))+(j))
using namespace cub;

// Device code
__global__ void Calculation(double* Anew, double* A, int N)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	if (i > 0 && j > 0 && i < N - 1 && j < N - 1){
		Anew[ID(i, j, N)] = 0.25 * (A[ID(i + 1, j, N)] + A[ID(i - 1, j, N)] + A[ID(i, j + 1, N)] + A[ID(i, j - 1, N)]);
	}
}

__global__ void Err(double* Anew, double* A, double* Err, int N)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	if (i < N && j < N){
		Err[ID(i, j, N)] = Anew[ID(i, j, N)] - A[ID(i, j, N)];
    }
}




template <
    int                     BLOCK_THREADS,
    int                     ITEMS_PER_THREAD,
    BlockReduceAlgorithm    ALGORITHM>
__global__ void BlockReduceKernel(
    double         *d_in,          // Tile of input
    double         *d_out)         // Tile aggregate
{
    // Specialize BlockReduce type for our thread block
    typedef BlockReduce<double, BLOCK_THREADS, ALGORITHM> BlockReduceT;
    // Shared memory
    __shared__ typename BlockReduceT::TempStorage temp_storage;
    // Per-thread tile data
    double thread_data[ITEMS_PER_THREAD];
	int block_offset = blockIdx.x * (BLOCK_THREADS * ITEMS_PER_THREAD);
    LoadDirectStriped<BLOCK_THREADS>(threadIdx.x + block_offset, d_in, thread_data);
    // Compute max
    double aggregate = BlockReduceT(temp_storage).Reduce(thread_data, cub::Max());
    // Store aggregate
    if (threadIdx.x == 0)
    {
        d_out[blockIdx.x] = aggregate;
    }
}


//Host code 
int main(int argc, char** argv)
{
	clock_t start = clock();
	int iter_max;
	int n;
	double tol;
	sscanf(argv[1], "%lf", &tol);
	sscanf(argv[2], "%d", &n);
	sscanf(argv[3], "%d", &iter_max);
	size_t size = n * n * sizeof(double);
    int numBlocks = (n*n) / (256*16) + 1;
	if (n*n % (256*16) == 0)
		numBlocks--;
	printf("numBlocks: %d\n", numBlocks);
	//Allocate input vectors A and A_new in host and divice memory 
	double* h_A = (double*)calloc(n*n, sizeof(double));
	double* h_Anew = (double*)calloc(n*n, sizeof(double));
    double* d_A;
    double* d_Anew;
    double* d_Err;
    double* d_out;
    double* res;
	cudaMalloc((void**)&d_A, size);
	cudaMalloc((void**)&d_Anew, size);
	cudaMalloc((void**)&d_Err, size);
	cudaMalloc((void**)&d_out, numBlocks * sizeof(double));
	cudaMalloc((void**)&res, sizeof(double));
	// Initialize input vectors
	h_A[ID(0, 0, n)] = 10;
	h_A[ID(0, n-1, n)] = 20;
	h_A[ID(n-1, 0, n)] = 20;
	h_A[ID(n-1, n-1, n)] = 30;
	double step = (double)10/(n-1);
	for(int i = 1; i < n-1;i++){
		h_A[ID(0, i, n)] = h_A[ID(0, i-1, n)] + step;
		h_A[ID(n-1, i, n)] = h_A[ID(n-1, i-1, n)] + step;
		h_A[ID(i, 0, n)] = h_A[ID(i-1, 0, n)] + step;
		h_A[ID(i, n-1, n)] = h_A[ID(i-1, n-1, n)] + step;
	}
	int iter = 0;
	double err = 1;
	// Copy vectors from host memory to device memory
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Anew, h_A, size, cudaMemcpyHostToDevice);
	// Invoke kernel
	dim3 threadsPerBlock(8, 8);
	dim3 blocksPerGrid(n / threadsPerBlock.x + 1, n / threadsPerBlock.y + 1);
	while(err > tol && iter < iter_max){
		iter = iter + 1;
		Calculation<<<blocksPerGrid, threadsPerBlock >>>(d_Anew, d_A, n);
		Calculation<<<blocksPerGrid, threadsPerBlock >>>(d_A, d_Anew, n);
		if (iter % 100 == 0 || iter == 1){
			Err<<<blocksPerGrid, threadsPerBlock >>>(d_A, d_Anew, d_Err, n);
			BlockReduceKernel<256, 16, BLOCK_REDUCE_WARP_REDUCTIONS><<<numBlocks, 256>>>(d_Err, d_out);
	        BlockReduceKernel<256, 1, BLOCK_REDUCE_WARP_REDUCTIONS><<<1, numBlocks>>>(d_out, res);
	        cudaMemcpy(&err, res, sizeof(double), cudaMemcpyDeviceToHost);
		}
		if (iter % 10000 == 0 || iter == 1) printf("%d %.15lf\n", iter, err);
	}
	

	// Copy result from device memory to host memory
	// cudaMemcpy(h_Anew, d_out, numBlocks*sizeof(double), cudaMemcpyDeviceToHost);
	// h_Anew contains the result in host memory 
	// Free device memory 
	cudaFree(d_A);
	cudaFree(d_Anew);
	cudaFree(d_Err);
	cudaFree(d_out);
	cudaFree(res);
	printf("\n");
	clock_t end = clock();
	printf("Number of iterations: %d\nError: %.15lf\n", iter, err);
	printf("\nExecution time: %lf\n", (double)(end - start) / CLOCKS_PER_SEC);
	// Free host memory
	free(h_A);
	free(h_Anew);
	return 0;
}
