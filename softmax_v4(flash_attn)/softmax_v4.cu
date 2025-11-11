#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cfloat>
#include <cstdlib>
#include <time.h>
#include <chrono>

__global__ void fused_kernel(const float* input, int N, float* output_softmax){
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int thread = threadIdx.x;
    __shared__ float s_max[1024], s_sum[1024];

    float running_max = -FLT_MAX;
    float running_sum = 0.0f;

    for(int i=thread; i < N; i+= blockDim.x){

        float current = input[i];
        float max_new = fmaxf(running_max, current);

        running_sum = running_sum * expf(running_max - max_new) + expf(current - max_new);
        running_max = max_new;
    }

    s_max[thread] = running_max;
    s_sum[thread] = running_sum;
    __syncthreads();

    for(int s = blockDim.x/2; s > 0; s/=2){
        if(thread < s){
            float m_i = s_max[thread + s];
            float l_i = s_sum[thread + s];

            float m_current = s_max[thread];
            float l_current = s_sum[thread];

            float max_new = fmaxf(m_current, m_i);

            s_sum[thread] = l_current * expf(m_current - max_new) + l_i * expf(m_i - max_new);
            s_max[thread] = max_new;
        }

        __syncthreads();
    }

    // function below code is useless, I have added it just for clarity that the reduction is over
    if(thread == 0){
        s_max[0] = s_max[0]; // m_final
        s_sum[0] = s_sum[0]; // l_final
    }
    
    __syncthreads();

    if (index < N) {
        float m_final = s_max[0];
        float l_final = s_sum[0]; // The total normalized sum (Sum(exp(x-m_final)))

        // The final Softmax value: P_i = exp(x_i - m_final) / l_final
        output_softmax[index] = expf(input[index] - m_final) / l_final;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 1024;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;


    // --- Timing Events for Kernels ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // KERNEL 1: NORMALIZE (Find Max in Blocks)
    cudaEventRecord(start, 0);
    fused_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, N, output);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);


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
    printf("HtoD Data Transfer Complete.\n");

    // --- 4. KERNEL EXECUTION (You need to update your 'solve' function to use the timers) ---
    solve(d_input, d_output, N);

    // --- 5. DtoH Transfer ---
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    
    // --- 6. VERIFICATION AND CLEANUP ---
    float sum_check = 0.0f;
    printf("--- Softmax Results (N=%d) ---\n", N);
   
    for(int i = 0; i < N; i++){
        sum_check += h_output[i];
    }
    printf("-------------------------------\n");
    printf("Verification Sum of Outputs: %12.10f (Should be 1.0)\n", sum_check);

    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
