#define _CRT_SECURE_NO_WARNINGS
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#pragma acc routine seq
double max(double a, double b){
	return a > b ? a : b;
}


int main(int argc, char** argv){
	clock_t start = clock();
	int iter_max;
	int n;
	double tol;
	sscanf(argv[1], "%lf", &tol);
	sscanf(argv[2], "%d", &n);
	sscanf(argv[3], "%d", &iter_max);
	double** A = (double**)malloc(n * sizeof(double*));
	double** Anew = (double**)malloc(n * sizeof(double*));
	for(int i = 0; i < n; i++){
		A[i] = (double*)calloc(n, sizeof(double));
		Anew[i] = (double*)calloc(n, sizeof(double));
	}
	A[0][0] = 10;
	A[0][n-1] = 20;
	A[n-1][0] = 20;
	A[n-1][n-1] = 30;
	double step = (double)10/(n-1);
	for(int i = 1; i < n-1;i++){
		A[0][i] = A[0][i-1] + step;
		A[n-1][i] = A[n-1][i-1] + step;
		A[i][0] = A[i-1][0] + step;
		A[i][n-1] = A[i-1][n-1] + step;
	}
	int iter = 0;
	double err = 1;
        #pragma acc data copy(A[:n][:n]) create(Anew[:n][:n])
	{
	#pragma acc parallel
	{
	Anew[0][0] = 10;
	Anew[0][n-1] = 20;
	Anew[n-1][0] = 20;
	Anew[n-1][n-1] = 30;
	#pragma acc loop
	for(int i = 1; i < n - 1; i++){
		Anew[0][i] = A[0][i];
		Anew[n-1][i] = A[n-1][i];
		Anew[i][0] = A[i][0];
		Anew[i][n-1] = A[i][n-1];
	}
	}
	while(err > tol && iter < iter_max){
		iter = iter + 1;
		err = 0;
		#pragma acc data present(A, Anew)
                #pragma acc parallel loop gang worker num_workers(4) vector_length(128) 
		for(int j = 1; j < n - 1; j++){
                        #pragma acc loop vector reduction(max:err)
			for(int i = 1; i < n - 1; i++){ 
				Anew[j][i] = 0.25 * (A[j][i+1] + A[j][i-1] + A[j-1][i] + A[j+1][i]);	
				err = max(err, Anew[j][i] - A[j][i]);
			}
		}
		double** t = A;
		A = Anew;
		Anew = t;
		if (iter % 10000 == 0 || iter == 1) printf("%d %.15lf\n", iter, err);
	}
	}	
	clock_t end = clock();
	printf("%d %.15lf\n", iter, err);
	printf("\n%lf\n", (double)(end - start) / CLOCKS_PER_SEC);
	for(int i = 0; i < n; i++){
		free(A[i]);
		free(Anew[i]);
	}
	free(A);
	free(Anew);
	return 0;
}

