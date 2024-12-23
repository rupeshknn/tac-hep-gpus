/*
# Managed memory
compile: "nvcc -o cuda_mm cuda_mm.cu"
run check: "./cuda_mm -check"
run code with size 512 (default): "./cuda_mm"
run code with size 4096: "./cuda_mm -size 4096"

nsys comment: Most time speant on memory allocation and movement (about 87%) slight improvement to cuda_code.
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
__global__ void stencilKernel(const int* A, int* Ac, int DSIZE, int radius) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= radius && idx < DSIZE - radius && idy >= radius && idy < DSIZE - radius) {
        int temp = -A[idx * DSIZE + idy];
        for (int r = -radius; r < radius+1; r++) {
            temp += A[(idx + r) * DSIZE + idy] + A[idx * DSIZE + idy + r];
        }
        Ac[idx * DSIZE + idy] = temp;
    }
}

// Matrix multiplication kernel
__global__ void matmulKernel(const int* Ac, const int* Bc, int* C, int DSIZE) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < DSIZE && col < DSIZE) {
        int sum = 0;
        for (int k = 0; k < DSIZE; ++k) {
            sum += Ac[row * DSIZE + k] * Bc[k * DSIZE + col];
        }
        C[row * DSIZE + col] = sum;
    }
}

// Host function
int* stencilMatmul(bool isRand, int radius, const int DSIZE) {
    // Unified memory allocation
    int *A, *B, *Ac, *Bc, *C;
    CUDA_CHECK(cudaMallocManaged(&A, DSIZE * DSIZE * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&B, DSIZE * DSIZE * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&Ac, DSIZE * DSIZE * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&Bc, DSIZE * DSIZE * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&C, DSIZE * DSIZE * sizeof(int)));

    // Initialize matrices
    for (int i = 0; i < DSIZE; ++i) {
        for (int j = 0; j < DSIZE; ++j) {
            A[i * DSIZE + j] = isRand ? rand() % 10 : 1;
            B[i * DSIZE + j] = isRand ? rand() % 10 : 1;
            Ac[i * DSIZE + j] = A[i * DSIZE + j];
            Bc[i * DSIZE + j] = B[i * DSIZE + j];
            C[i * DSIZE + j] = 0;
        }
    }

    // Kernel configurations
    dim3 blockDim(16, 16);
    dim3 gridDim((DSIZE + blockDim.x - 1) / blockDim.x, (DSIZE + blockDim.y - 1) / blockDim.y);

    // Launch stencil kernels
    stencilKernel<<<gridDim, blockDim>>>(A, Ac, DSIZE, radius);
    stencilKernel<<<gridDim, blockDim>>>(B, Bc, DSIZE, radius);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Launch matrix multiplication kernel
    matmulKernel<<<gridDim, blockDim>>>(Ac, Bc, C, DSIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Free unified memory
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(Ac));
    CUDA_CHECK(cudaFree(Bc));

    return C; // Return result (managed memory pointer)
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
    CUDA_CHECK(cudaFree(C));

    return 0;
}
