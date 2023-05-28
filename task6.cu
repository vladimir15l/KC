/*implementation of a small neural network*/
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <math.h>
#include <cublas_v2.h>


cublasHandle_t handle;
// Just a sigmoid
__global__ void sigmoid(float* x) {
    int idx = threadIdx.x;
    x[idx] = exp(x[idx]) / (1 + exp(x[idx]));
}

// This class implements a fully connected layer
class Linear {
    float* weight;
    float* bias;
    int in_features;
    int out_features;
public:
    Linear() {
        weight = NULL;
        bias = NULL;
        in_features = 0;
        out_features = 0;
    };
    Linear(int in, int out) {
        weight = NULL;
        bias = NULL;
        in_features = in;
        out_features = out;
    }
    //initializes weights and bias
    void initializer(FILE* weights){
        float* w = (float*)malloc(in_features * out_features * sizeof(float));
        float* b = (float*)malloc(out_features * sizeof(float));
        fread(w, sizeof(float), in_features*out_features, weights);
        fread(b, sizeof(float), out_features, weights);
        cudaMalloc((void**)&weight, in_features * out_features * sizeof(float));
        cudaMalloc((void**)&bias, out_features * sizeof(float));
        cudaMemcpy(weight, w, in_features * out_features * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(bias, b, out_features * sizeof(float), cudaMemcpyHostToDevice);
        free(w);
        free(b);
    }
    // the vector with the input data is multiplied with the weight matrix
    float* operator() (float* x) {
        const float a = 1;
        cublasSgemv(handle, CUBLAS_OP_T, in_features, out_features, &a, weight, in_features, x, 1, &a, bias, 1);
        cublasScopy(handle, out_features, bias, 1, x, 1);  
        return x;
    }
    ~Linear() {
        if (weight)
            cudaFree(weight);
        if (bias)
            cudaFree(bias);
    }
};

// A neural network model with three fully connected layers
class Net {
    Linear fc1;
    Linear fc2;
    Linear fc3;
    // direct dissemination of information
    float forward(float* x) {
        sigmoid<<<1, 256>>>(fc1(x));
        sigmoid<<<1, 16>>>(fc2(x));
        sigmoid<<<1, 1>>>(fc3(x));
        float result;
        cudaMemcpy(&result, x, sizeof(float), cudaMemcpyDeviceToHost);
        return result;
    }
public:
    Net(int in, int middle1, int middle2) {
        cublasCreate(&handle);
        FILE* weight = fopen("weight.npy", "rb");
        if (weight == NULL) {
            printf(" Error writing in weight file\n");
            exit(1);
        }
        fc1 = Linear(in, middle1);
        fc2 = Linear(middle1, middle2);
        fc3 = Linear(middle2, 1);
        fc1.initializer(weight);
        fc2.initializer(weight);
        fc3.initializer(weight);
    }
    // Launching a neural network. Reading input data from a file
    // and starting a direct flow of information
    float operator() (char* file, int size) {        
        FILE* input = fopen(file, "rb");
        if (input == NULL) {
            printf(" Error writing in input file\n");
            exit(1);
        }
        float* input_layer = (float*)malloc(size * sizeof(float));  
    
        if(input_layer) fread(input_layer, sizeof(float), size, input);

        float* d_layer;
        cudaMalloc((void**)&d_layer, size*sizeof(float));
        cudaMemcpy(d_layer, input_layer, size*sizeof(float), cudaMemcpyHostToDevice);
	    free(input_layer);
        return forward(d_layer);
    }
    ~Net(){
        cublasDestroy(handle);
    }
};

int main() {
    int size = 1024;
    Net net = Net(1024, 256, 16);    
    float result = net("input.npy", size);
    printf("%lf\n\n", result);    
    return 0;
}