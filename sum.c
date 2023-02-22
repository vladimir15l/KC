#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>



double arr[10000000];

int main(){
	clock_t start = clock();
	int size = 10000000;
	double sum = 0;
#pragma acc data copy(sum) copyin(arr[:size])
	{
#pragma acc kernels
	{
	for (int i = 0; i < size; i++){
		arr[i] = sin((2*M_PI*i)/size);
	}
	for (int i = 0; i < size; i++){
		sum += arr[i];
	}
	}
	}
	clock_t end = clock();
	double result = 0.0;
	result += (double)(end - start)/CLOCKS_PER_SEC;
	printf("%.15lf\n", sum);
	printf("%.15lf", result);
	return 0;
}
