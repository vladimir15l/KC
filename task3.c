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
    
    acc_set_device_num(2, acc_device_default);
	cublasHandle_t handle;
	cublasStatus_t stat;
	cublasCreate(&handle);
	

	double* A = (double*)calloc(n * n, sizeof(double));
	double* Anew = (double*)calloc(n * n, sizeof(double));
	double* Aerr = (double*)calloc(n * n, sizeof(double));
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
	int iter = 0;
	double err = 1;
	int result;
	int flag = 0;
    #pragma acc data copyin(A[:n*n]) create(Anew[:n*n], Aerr[:n*n])
	{
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
	while(err > tol && iter < iter_max){
		iter = iter + 1;
		err = 0;
		#pragma acc data present(A, Anew)
        #pragma acc parallel loop gang worker num_workers(4) vector_length(128)
		for(int j = 1; j < n - 1; j++){
            #pragma acc loop vector
			for(int i = 1; i < n - 1; i++){ 
				Anew[IDX2C(j, i, n)] = 0.25 * (A[IDX2C(j, i+1, n)] + A[IDX2C(j, i-1, n)] + A[IDX2C(j-1, i, n)] + A[IDX2C(j+1, i, n)]);
			}
		}
		const double alpha = -1;
		#pragma acc host_data use_device(A, Anew, Aerr)
		{
		    if ((iter % 100 == 0 || iter == 1) && (iter >= 30000 || iter == 1)){
				stat = cublasDcopy(handle, n*n, Anew, 1, Aerr, 1);
		        if (stat != CUBLAS_STATUS_SUCCESS){
			        printf("cublasDcopy failed\n");
			        cublasDestroy(handle);
			        flag = 1;
			        break; 
		        }
		        stat = cublasDaxpy(handle, n*n, &alpha, A, 1, Aerr, 1);
		        if (stat != CUBLAS_STATUS_SUCCESS){
			        printf("cublasDaxpy failed\n");
			        cublasDestroy(handle);
			        flag = 1;
			        break;
		        }
		        stat = cublasIdamax(handle, n*n, Aerr, 1, &result);
		        if (stat != CUBLAS_STATUS_SUCCESS){
			        printf("cublasIdamax failed\n");
			        cublasDestroy(handle);
			        flag = 1;
			        break;
		        }
			}
		}
		#pragma acc kernels
		{
			err = Aerr[result-1];
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
	free(A);
	free(Anew);
	return 0;
}

