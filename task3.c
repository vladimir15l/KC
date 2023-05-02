/*
* This program is designed to solve the heat equation (five-point pattern) in a two-dimensional
* domain on uniform grids. Boundary conditions – linear interpolation between the corners of
* the region. The value in the corners is 10, 20, 30, 20.
* The algorithm is as follows:
*   * First we fill in the boundaries of the array.
*   * Then, in a loop, we run through all the elements of the array except the boundary ones
*     and calculate them as the average between the four neighbors, and save the result to a new array.
*     Then we subtract the old array from the new one step by step and find the maximum. That's how we
*     found the error. We repeat this procedure until the error becomes less than we need or until the
*     number of iterations exceeds the limit we have set.
*/

#define _CRT_SECURE_NO_WARNINGS
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cublas_v2.h>
#include <openacc.h>
#include <cuda_runtime.h>

#define IDX2C(i ,j, ld) (((j)*(ld))+(i)) 


int main(int argc, char** argv){
	clock_t start = clock();
	int iter_max;
	int n;
	double tol;
	sscanf(argv[1], "%lf", &tol);
	sscanf(argv[2], "%d", &n);
	sscanf(argv[3], "%d", &iter_max);
    
	acc_set_device_num(3, acc_device_default);
	cublasHandle_t handle;
	cublasStatus_t stat;
	cublasCreate(&handle);
	///////////////////////////////////////////////////////////////////////////
	// Creating variables, arrays and initializing them
	///////////////////////////////////////////////////////////////////////////  
	double* A = (double*)calloc(n * n, sizeof(double));
	double* Anew = (double*)calloc(n * n, sizeof(double));
	double* Aerr = (double*)calloc(n * n, sizeof(double));
    int iter = 0;
	double err = 1;
	int result;
	int flag = 0;

	//Filling in borders
	A[IDX2C(0, 0, n)] = 10;
	A[IDX2C(0, n-1, n)] = 20;
	A[IDX2C(n-1, 0, n)] = 20;
	A[IDX2C(n-1, n-1, n)] = 30;
    double step = (double)10/(n-1);
	for(int i = 1; i < n-1;i++){
		A[IDX2C(0, i, n)] = A[IDX2C(0, i-1, n)] + step;
		A[IDX2C(n-1, i, n)] = A[IDX2C(n-1, i-1, n)] + step;
		A[IDX2C(i, 0, n)] = A[IDX2C(i-1, 0, n)] + step;
		A[IDX2C(i, n-1, n)] = A[IDX2C(i-1, n-1, n)] + step;
	}
    
    // Copy the variables we need to the GPU and create on the GPU
    #pragma acc data copy(err) copyin(A[:n*n]) create(Anew[:n*n], Aerr[:n*n])
	{
    // We fill in the boundaries of the created array on the GPU
	#pragma acc parallel
	{
	Anew[IDX2C(0, 0, n)] = 10;
	Anew[IDX2C(0, n-1, n)] = 20;
	Anew[IDX2C(n-1, 0, n)] = 20;
	Anew[IDX2C(n-1, n-1, n)] = 30;
	#pragma acc loop
	for(int i = 1; i < n - 1; i++){
		Anew[IDX2C(0, i, n)] = A[IDX2C(0, i, n)];
		Anew[IDX2C(n-1, i, n)] = A[IDX2C(n-1, i, n)];
		Anew[IDX2C(i, 0, n)] = A[IDX2C(i, 0, n)];
		Anew[IDX2C(i, n-1, n)] = A[IDX2C(i, n-1, n)];
	}
	}
	///////////////////////////////////////////////////////////////////////////
	// The main part of the program. We calculate the elements as the average
	// of the four neighbors. We find the element-by-element difference between
	// the new and old arrays and the maximum error using the cuBLAS library to
	// the error value we need or the maximum number of iterations
	///////////////////////////////////////////////////////////////////////////
	while(err > tol && iter < iter_max){
		iter = iter + 1;
		// Parallelize the calculation of elements using OpenACC 
		#pragma acc data present(A, Anew)
        #pragma acc parallel loop gang num_workers(4) vector_length(128) 
        for(int i = 1; i < n - 1; i++) { 
            #pragma acc loop vector 
            for(int j = 1; j < n - 1; j++)  
                Anew[IDX2C(j, i, n)] = 0.25 * (A[IDX2C(j, i+1, n)] + A[IDX2C(j, i-1, n)] + A[IDX2C(j-1, i, n)] + A[IDX2C(j+1, i, n)]);
        }
 
        // We calculate the error every 100 iterations using the cuBLAS library 
        if (iter % 100 == 0 || iter == 1)
        {  
            double alpha = -1.0; 
                 
            #pragma acc host_data use_device(Anew, A, Aerr) 
            { 
				// copy Anew to Aerr
				stat = cublasDcopy(handle, n * n, Anew, 1, Aerr, 1);
				if (stat != CUBLAS_STATUS_SUCCESS){
			        printf("cublasDcopy failed\n");
			        cublasDestroy(handle);
			        flag = 1;
			        break; 
		        }
                // Aerr(Anew) - A 
                stat = cublasDaxpy(handle, n * n, &alpha, A, 1, Aerr, 1);
				if (stat != CUBLAS_STATUS_SUCCESS){
			        printf("cublasDaxpy failed\n");
			        cublasDestroy(handle);
			        flag = 1;
			        break;
		        } 
                // We find the maximum error in Aerr and return the error index
                stat = cublasIdamax(handle, n * n, Aerr, 1, &result);
				if (stat != CUBLAS_STATUS_SUCCESS){
			        printf("cublasIdamax failed\n");
			        cublasDestroy(handle);
			        flag = 1;
			        break;
		        } 
            } 
			#pragma acc kernels
				err = Aerr[result-1];
			// updating the error value on the host
            #pragma acc update host(err) 
        }
		double* t = A;
		A = Anew;
		Anew = t;
		if (iter % 10000 == 0 || iter == 1) printf("%d %.15lf\n", iter, err); 
	}
	}
	if (flag == 1){
		return EXIT_FAILURE;
	}
	clock_t end = clock();

	cublasDestroy(handle);

	printf("Number of iterations: %d\nError: %.15lf\n", iter, err);
	printf("\nExecution time: %lf\n", (double)(end - start) / CLOCKS_PER_SEC);
	// freeing up memory
	free(A);
	free(Anew);
	return 0;
}