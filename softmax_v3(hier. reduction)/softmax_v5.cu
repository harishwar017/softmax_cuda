#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cfloat>
#include <cstdlib> 
#include <time.h>
#include <chrono>


__global__ void norm_kernel(const float* input, float* maxi_p, int N){
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int thread = threadIdx.x;
    __shared__ float temp[256];
    if(index < N){
        temp[thread] = input[index];
    }
    else{
        temp[thread] = 0.0f;
    }
     __syncthreads();

    for(int i = blockDim.x/2; i > 0; i/=2){
        if(thread < i){
            temp[thread] = fmaxf(temp[thread], temp[thread + i]);
        }

        __syncthreads();
    }

    if(thread == 0){
        maxi_p[blockIdx.x] = temp[0];
    }
    
}

__global__ void softmax_kernel(const float* input, float* output, float* deno, int N, float global_max) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < N) {
        float denom = *deno;
        denom = fmaxf(denom, 1e-8f);  // ensure nonzero, without changing correct results
        output[index] = expf(input[index] - global_max) / denom;
    }
}

__global__ void addition_kernel(const float* input, float* deno, int N, float global_max) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float temp[256];
    int thread = threadIdx.x;

    if(index < N){
        temp[thread] = expf(input[index] - global_max) ;
    }
    else{
        temp[thread] = 0.0f;
    }
    __syncthreads();
        
    for(int i= blockDim.x/2; i >0; i/=2){
        if(thread < i){
            temp[thread] += temp[thread + i];
        }
        __syncthreads();
    }

    if(thread == 0){
        atomicAdd(deno, temp[0]);
    }
 
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // setup formDenominator and Max value
    float* deno;
    cudaMalloc(&deno, sizeof(float));
    float zero = 0.0f;
    cudaMemcpy(deno, &zero, sizeof(float), cudaMemcpyHostToDevice);

    float* maxi = (float*)malloc(sizeof(float) * blocksPerGrid);
    for (int i = 0; i < blocksPerGrid; i++) maxi[i] = -FLT_MAX;

    float* maxi_p;
    cudaMalloc(&maxi_p, sizeof(float) * blocksPerGrid);
    cudaMemcpy(maxi_p, maxi, sizeof(float) * blocksPerGrid, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // KERNEL 1: NORMALIZE (Find Max in Blocks)
    cudaEventRecord(start, 0);
    norm_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, maxi_p, N);
    cudaEventRecord(stop, 0);

    int cur_len = blocksPerGrid;
    float* d_in = maxi_p;
    float* d_out = nullptr;
    cudaMalloc(&d_out, sizeof(float) * cur_len ); 

    float* allocated_p1 = maxi_p;
    float* allocated_p2 = d_out;

    while (cur_len > 1) {
        int out_blocks = (cur_len + threadsPerBlock - 1) / threadsPerBlock;

        norm_kernel<<<out_blocks, threadsPerBlock>>>(d_in, d_out, cur_len);
        cudaEventSynchronize(stop);;

        // swap pointers & update cur_len
        cur_len = out_blocks;
        float* tmp = d_in;
        d_in = d_out;
        d_out = tmp;
    }

    cudaEventSynchronize(stop);
    
    // cudaFree(maxi_p);
    float global_max;
    cudaMemcpy(&global_max, d_in, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(allocated_p1); 
    cudaFree(allocated_p2);

    // KERNEL 2: REDUCTION (Compute Denominator for Softmax)
    cudaEventRecord(start, 0);
    addition_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, deno, N, global_max);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);

    // KERNEL 3: SOFTMAX (Final Calculation)
    cudaEventRecord(start, 0);
    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, deno, N, global_max);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);

    cudaFree(deno);
    free(maxi);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    const int N = 10000000;
    const size_t bytes = sizeof(float) * N;
    
    float *h_input = NULL; 
    float *h_output = NULL; 
    float *d_input = NULL; 
    float *d_output = NULL; 
    
    // --- 1. SETUP AND ALLOCATION ---
    h_input = (float*)malloc(bytes);
    h_output = (float*)malloc(bytes);
    

    cudaMalloc(&d_input, bytes); 
    cudaMalloc(&d_output, bytes);
    

    // --- 2. DATA INITIALIZATION ---
    srand(static_cast<unsigned int>(time(0))); 
    
    for (int i = 0; i < N; i++) {
        int random_int = rand();
        float min_val = -10.0f;
        float max_val = 10.0f;
        h_input[i] = min_val + (static_cast<float>(random_int) / static_cast<float>(RAND_MAX)) * (max_val - min_val);
    }
    
    printf("\n--- Data Transfer & Execution ---\n");
    // --- 3. HtoD Transfer (Time this if needed, but we focus on Kernel time) ---
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    // --- 4. KERNEL EXECUTION (You need to update your 'solve' function to use the timers) ---
    solve(d_input, d_output, N);

    // --- 5. DtoH Transfer ---
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    
    // --- 6. VERIFICATION AND CLEANUP ---
    float sum_check = 0.0f;

    for(int i = 0; i < N; i++){
        sum_check += h_output[i];
    }
    printf("-------------------------------\n");
    printf("Verification Sum of Outputs: %12.10f (Should be 1.0)\n", sum_check);

    free(h_input);
    free(h_output);

    return 0;
}