/*
* This program is designed to solve the heat equation (five-point pattern) in a two-dimensional
* domain on uniform grids. Boundary conditions – linear interpolation between the corners of
* the region. The value in the corners is 10, 20, 30, 20.
* The algorithm is as follows:
*   * First we fill in the boundaries of the array.
*   * Then, in a loop, we run through all the elements of the array except the boundary ones
*     and calculate them as the average between the four neighbors, and save the result to a new array.
*     Then we subtract the old array from the new one element by element and find the maximum. That's how we
*     found the error. We repeat this procedure until the error becomes less than we need or until the
*     number of iterations exceeds the limit we have set.
*/

#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_reduce.cuh>

#define ID(i ,j, ld) (((i)*(ld))+(j))
using namespace cub;

// Device code
__global__ void Calculation(double* Anew, double* A, int dimx, int dimy)
{
	// Calculating each element on the device
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	if (i > 0 && j > 0 && i < dimy - 1 && j < dimx - 1){
		Anew[ID(i, j, dimx)] = 0.25 * (A[ID(i + 1, j, dimx)] + A[ID(i - 1, j, dimx)] + A[ID(i, j + 1, dimx)] + A[ID(i, j - 1, dimx)]);
	}
}


// Device code
__global__ void Err(double* Anew, double* A, double* Err, int dimx, int dimy)
{
	// Calculating an array of errors
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	if (i < dimy && j < dimx){
		Err[ID(i, j, dimx)] = Anew[ID(i, j, dimx)] - A[ID(i, j, dimx)];
    }
}

///////////////////////////////////////////////////////////////////////////////
// Calculating the maximum error by block reduce
// using functions from the CUB library
///////////////////////////////////////////////////////////////////////////////
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

	int rank,np;
    int proc = 1;
    /* Initialize the MPI library */
    MPI_Init(&argc,&argv);
    /* Determine the calling process rank and total number of ranks */
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&np);
    /* Call MPI routines like MPI_Send, MPI_Recv, ... */
    cudaSetDevice(rank);    
	
    int dimx = n;
    int dimy = n / np;
    numBlocks /= np;
	if (rank == proc) printf("numBlocks: %d\n", numBlocks);    


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
	cudaMalloc((void**)&d_Err, size/np);
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

	double *h_top_boundary = NULL, *h_bottom_boundary = NULL;
    double *h_top_halo = NULL, *h_bottom_halo = NULL;
	int num_halo_points = dimx - 2;
	int num_halo_bytes = num_halo_points * sizeof(double);

	cudaHostAlloc((void **)&h_top_boundary, num_halo_bytes, cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_bottom_boundary,num_halo_bytes, cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_top_halo, num_halo_bytes, cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_bottom_halo, num_halo_bytes, cudaHostAllocDefault);

	// get the range of stream priorities for this device
	int priority_high, priority_low;
	cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
	// create streams with highest and lowest available priorities
	cudaStream_t stream0, stream1;
	cudaStreamCreateWithPriority(&stream0, cudaStreamNonBlocking, priority_high);
	cudaStreamCreateWithPriority(&stream1, cudaStreamNonBlocking, priority_low);
 
	// Copy vectors from host memory to device memory
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Anew, h_A, size, cudaMemcpyHostToDevice);
	// Invoke kernel
	dim3 threadsPerBlock(8, 8);
	dim3 blocksPerGrid(dimy / threadsPerBlock.x + 1, dimx / threadsPerBlock.y + 1);
    int to_lborder = dimx * ((rank+1) * dimy - 2);
	int to_uborder = dimx * (rank * dimy - 1);
	int offset = rank * dimx * dimy;
	int top_neighbor = (rank > 0) ? (rank - 1) : MPI_PROC_NULL;
    int bottom_neighbor = (rank < np - 1) ? (rank + 1) : MPI_PROC_NULL;
	///////////////////////////////////////////////////////////////////////////
	// We calculate the elements two times in a row by swapping arrays.
	// Every 100 iterations we calculate the maximum error using BlockReduceKernel
	///////////////////////////////////////////////////////////////////////////
	MPI_Status status;
    MPI_Barrier( MPI_COMM_WORLD );
	while(iter < iter_max){
		iter = iter + 2;
		//First of all, we will calculate the boundary nodes needed by the rest
		if (rank < np - 1) Calculation<<<blocksPerGrid, threadsPerBlock, 0, stream0>>>(d_Anew + to_lborder, d_A + to_lborder, dimx, 3);
		if (rank > 0) Calculation<<<blocksPerGrid, threadsPerBlock, 0, stream0>>>(d_Anew +to_uborder, d_A + to_uborder, dimx, 3);
		//Calculation of remaining points
        Calculation<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_Anew + offset, d_A + offset, dimx, dimy);
		//Sending data needed by other nodes to the host
		if (rank < np -1) cudaMemcpyAsync(h_bottom_boundary, d_Anew + to_lborder + dimx + 1, num_halo_bytes, cudaMemcpyDeviceToHost, stream0);
		if (rank > 0) cudaMemcpyAsync(h_top_boundary, d_Anew + to_uborder + dimx + 1, num_halo_bytes, cudaMemcpyDeviceToHost, stream0);
		cudaStreamSynchronize(stream0);
        //Sending data to the bottom, receiving from the top
		MPI_Sendrecv(h_bottom_boundary, num_halo_points, MPI_DOUBLE,
		            bottom_neighbor, iter, h_top_halo, num_halo_points, MPI_DOUBLE,
					top_neighbor, iter, MPI_COMM_WORLD, &status);
		//Sending data to the top, receiving from the bottom
		MPI_Sendrecv(h_top_boundary, num_halo_points, MPI_DOUBLE,
		            top_neighbor, iter, h_bottom_halo, num_halo_points, MPI_DOUBLE,
					bottom_neighbor, iter, MPI_COMM_WORLD, &status);
		
		if (rank < np - 1) cudaMemcpyAsync(d_Anew + to_lborder + dimx*2 + 1, h_bottom_halo, num_halo_bytes, cudaMemcpyHostToDevice, stream0);
		if (rank > 0) cudaMemcpyAsync(d_Anew + to_uborder + 1, h_top_halo, num_halo_bytes, cudaMemcpyHostToDevice, stream0);
		cudaDeviceSynchronize();

        ///////////////////////////////////////////////////////////////////////////////////////////
		//Swapping arrays Anew and A
		///////////////////////////////////////////////////////////////////////////////////////////

		//First of all, we will calculate the boundary nodes needed by the rest
		if (rank < np - 1) Calculation<<<blocksPerGrid, threadsPerBlock, 0, stream0>>>(d_A + to_lborder, d_Anew + to_lborder, dimx, 3);
		if (rank > 0) Calculation<<<blocksPerGrid, threadsPerBlock, 0, stream0>>>(d_A +to_uborder, d_Anew + to_uborder, dimx, 3);
		//Calculation of remaining points
        Calculation<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_A + offset, d_Anew + offset, dimx, dimy);
		//Sending data needed by other nodes to the host
		if (rank < np -1) cudaMemcpyAsync(h_bottom_boundary, d_A + to_lborder + dimx + 1, num_halo_bytes, cudaMemcpyDeviceToHost, stream0);
		if (rank > 0) cudaMemcpyAsync(h_top_boundary, d_A + to_uborder + dimx + 1, num_halo_bytes, cudaMemcpyDeviceToHost, stream0);
		cudaStreamSynchronize(stream0);
        //Sending data to the bottom, receiving from the top
		MPI_Sendrecv(h_bottom_boundary, num_halo_points, MPI_DOUBLE,
		            bottom_neighbor, iter, h_top_halo, num_halo_points, MPI_DOUBLE,
					top_neighbor, iter, MPI_COMM_WORLD, &status);
		//Sending data to the top, receiving from the bottom
		MPI_Sendrecv(h_top_boundary, num_halo_points, MPI_DOUBLE,
		            top_neighbor, iter, h_bottom_halo, num_halo_points, MPI_DOUBLE,
					bottom_neighbor, iter, MPI_COMM_WORLD, &status);
		
		if (rank < np - 1) cudaMemcpyAsync(d_A + to_lborder + dimx*2 + 1, h_bottom_halo, num_halo_bytes, cudaMemcpyHostToDevice, stream0);
		if (rank > 0) cudaMemcpyAsync(d_A + to_uborder + 1, h_top_halo, num_halo_bytes, cudaMemcpyHostToDevice, stream0);
		cudaDeviceSynchronize();
		
        
		if (iter % 100 == 0 || iter == 1){
			Err<<<blocksPerGrid, threadsPerBlock >>>(d_A + offset, d_Anew + offset, d_Err, dimx, dimy);
			BlockReduceKernel<256, 16, BLOCK_REDUCE_WARP_REDUCTIONS><<<numBlocks, 256>>>(d_Err, d_out);
	        BlockReduceKernel<64, 1, BLOCK_REDUCE_WARP_REDUCTIONS><<<1, numBlocks>>>(d_out, res);
			// Copy result from device memory to host memory
	        cudaMemcpy(&err, res, sizeof(double), cudaMemcpyDeviceToHost);
		}
		if (rank == proc) if (iter % 10000 == 0 || iter == 1) printf("%d %.15lf\n", iter, err);
	}
	
	// Free device memory 
	cudaFree(d_A);
	cudaFree(d_Anew);
	cudaFree(d_Err);
	cudaFree(d_out);
	cudaFree(res);
    /* Shutdown MPI library */
    MPI_Finalize();
	clock_t end = clock();
	if (rank == proc) printf("Number of iterations: %d\nError: %.15lf\n", iter, err);
	if (rank == proc) printf("\nExecution time: %lf\n", (double)(end - start) / CLOCKS_PER_SEC);
	// Free host memory
	free(h_A);
	free(h_Anew);
	return 0;
}
