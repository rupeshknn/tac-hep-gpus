/*
# Basic cuda implementation

compile: "nvcc -o cuda_code cuda_code.cu"
run check: "./cuda_code -check"
run code with size 512 (default): "./cuda_code"
run code with size 4096: "./cuda_code -size 4096"

nsys comment: Most time speant on memory allocation and movement (about 90%)
*/

#include <cuda.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Stencil kernel
__global__ void stencilKernel(const int* d_A, int* d_Ac, int DSIZE, int radius) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= radius && idx < DSIZE - radius && idy >= radius && idy < DSIZE - radius) {
        int temp = -d_A[idx * DSIZE + idy];
        for (int r = -radius; r < radius+1; r++) {
            temp += d_A[(idx + r) * DSIZE + idy] + d_A[idx * DSIZE + idy + r];
        }
        d_Ac[idx * DSIZE + idy] = temp;
    }
}

// Matrix multiplication kernel
__global__ void matmulKernel(const int* d_Ac, const int* d_Bc, int* d_C, int DSIZE) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < DSIZE && col < DSIZE) {
        int sum = 0;
        for (int k = 0; k < DSIZE; ++k) {
            sum += d_Ac[row * DSIZE + k] * d_Bc[k * DSIZE + col];
        }
        d_C[row * DSIZE + col] = sum;
    }
}

// Host function
int* stencilMatmul(bool isRand, int radius, const int DSIZE) {
    // Allocate host memory
    int *h_A, *h_B, *h_Ac, *h_Bc, *h_C;
    h_A = new int[DSIZE * DSIZE];
    h_B = new int[DSIZE * DSIZE];
    h_Ac = new int[DSIZE * DSIZE];
    h_Bc = new int[DSIZE * DSIZE];
    h_C = new int[DSIZE * DSIZE];

    // Initialize matrices
    for (int i = 0; i < DSIZE; ++i) {
        for (int j = 0; j < DSIZE; ++j) {
            h_A[i * DSIZE + j] = isRand ? rand() % 10 : 1;
            h_B[i * DSIZE + j] = isRand ? rand() % 10 : 1;
            h_Ac[i * DSIZE + j] = h_A[i * DSIZE + j];
            h_Bc[i * DSIZE + j] = h_B[i * DSIZE + j];
            h_C[i * DSIZE + j] = 0;
        }
    }

    // Allocate device memory
    int *d_A, *d_B, *d_Ac, *d_Bc, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, DSIZE * DSIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, DSIZE * DSIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_Ac, DSIZE * DSIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_Bc, DSIZE * DSIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, DSIZE * DSIZE * sizeof(int)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, DSIZE * DSIZE * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, DSIZE * DSIZE * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Ac, h_Ac, DSIZE * DSIZE * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Bc, h_Bc, DSIZE * DSIZE * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, DSIZE * DSIZE * sizeof(int), cudaMemcpyHostToDevice));

    // Kernel configurations
    dim3 blockDim(16, 16);
    dim3 gridDim((DSIZE + blockDim.x - 1) / blockDim.x, (DSIZE + blockDim.y - 1) / blockDim.y);

    // Launch stencil kernels
    stencilKernel<<<gridDim, blockDim>>>(d_A, d_Ac, DSIZE, radius);
    stencilKernel<<<gridDim, blockDim>>>(d_B, d_Bc, DSIZE, radius);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Launch matrix multiplication kernel
    matmulKernel<<<gridDim, blockDim>>>(d_Ac, d_Bc, d_C, DSIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    // CUDA_CHECK(cudaMemcpy(h_C, d_Ac, DSIZE * DSIZE * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, DSIZE * DSIZE * sizeof(int), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_Ac));
    CUDA_CHECK(cudaFree(d_Bc));
    CUDA_CHECK(cudaFree(d_C));

    // Free unused host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_Ac;
    delete[] h_Bc;

    return h_C;
}

int main(int argc, char const *argv[]) {
    bool check = false, dsize_set = false;
    uint DSIZE;

    if ( argc > 1){
        if (strcmp( argv[1], "-check") == 0){
            check = true;
        }
        if (strcmp( argv[1], "-size") == 0){
            DSIZE = std::atoi(argv[2]);
            dsize_set = true;
        }
    }
    int print_num = 10;
    int * C;
    if (check){
        DSIZE = 10;
        C = stencilMatmul(false, 1, DSIZE);
        if (C[0] != 10)
            printf("Mismatch at index [%d,%d], was: %d, should be: %d\n", 0,0, C[0], 10);
        else if (C[1] != 42)
            printf("Mismatch at index [%d,%d], was: %d, should be: %d\n", 0,1, C[1], 42);
        else if (C[11] != 202)
            printf("Mismatch at index [%d,%d], was: %d, should be: %d\n", 2,1, C[11], 202);
        else
            printf("Sucess!\n");
        
        printf("C = [\n");
        for (int i = 0; i < print_num; i++) {
            printf("     [");
            for (int j = 0; j < print_num; j++) {
                printf("%3d, ", C[DSIZE*j + i]);
            }
            printf("\b\b  ]\n");
        }
        printf("    ]\n");
    } else{
        DSIZE = dsize_set ? DSIZE: 512;
        printf("the dsize is %d\n", DSIZE);
        const int radius = 3;

        auto start = std::chrono::steady_clock::now();

        C = stencilMatmul(true, radius, DSIZE);

        auto finish = std::chrono::steady_clock::now();
        double elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(finish - start).count();
        printf("time to run = %.2f\n\n", elapsed_seconds);
    }

    // Free unified memory for result
    delete[] C;

    return 0;
}