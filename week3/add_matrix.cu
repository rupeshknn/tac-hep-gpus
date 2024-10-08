#include <stdio.h>


const int DSIZE_X = 256;
const int DSIZE_Y = 256;
const int block_size = 32;

__global__ void add_matrix(const float *A, const float *B, float *C, int DSIZE_X, int DSIZE_Y)
{
    //FIXME:
    // Express in terms of threads and blocks
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    // Add the two matrices - make sure you are not out of range
    if (idx <  DSIZE_X && idy < DSIZE_Y )
        C[idy*DSIZE_Y + idx] =  A[idy*DSIZE_Y + idx] + B[idy*DSIZE_Y + idx];
}

int main()
{
    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
    int print_num = 3;
    // Create and allocate memory for host and device pointers 
    h_A = new float[DSIZE_X * DSIZE_Y];
    h_B = new float[DSIZE_X * DSIZE_Y];
    h_C = new float[DSIZE_X * DSIZE_Y];

    cudaMalloc(&d_A, DSIZE_X * DSIZE_Y*sizeof(float));
    cudaMalloc(&d_B, DSIZE_X * DSIZE_Y*sizeof(float));
    cudaMalloc(&d_C, DSIZE_X * DSIZE_Y*sizeof(float));

    // Fill in the matrices
    // FIXME
    for (int i = 0; i < DSIZE_X; i++) {
        for (int j = 0; j < DSIZE_Y; j++) {
            h_A[i*DSIZE_X + j] = rand()/(float)RAND_MAX;
            h_B[i*DSIZE_X + j] = rand()/(float)RAND_MAX;
            h_C[i*DSIZE_X + j] = 0;
        }
    }

    // Copy from host to device
     // Copy from host to device
    cudaMemcpy(d_A, h_A, DSIZE_X * DSIZE_Y*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE_X * DSIZE_Y*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, DSIZE_X * DSIZE_Y*sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    // dim3 is a built in CUDA type that allows you to define the block 
    // size and grid size in more than 1 dimentions
    // Syntax : dim3(Nx,Ny,Nz)
    dim3 blockSize(block_size, block_size); 
    dim3 gridSize(DSIZE_X/block_size, DSIZE_Y/block_size); 
    
    add_matrix<<<gridSize, blockSize>>>(d_A, d_B, d_C, DSIZE_X, DSIZE_Y);

    // Copy back to host
    cudaMemcpy(h_C, d_C, DSIZE_X * DSIZE_Y*sizeof(float), cudaMemcpyDeviceToHost);
    // Print and check some elements to make the addition was succesfull
    printf("A = [");
    for (int i = 0; i < print_num; i++) {
        printf("[ ");
        for (int j = 0; j < print_num; j++) {
            printf("%f, ", h_A[DSIZE_Y*j + i]);
        }
        printf("]\n");
    }
    printf("]\n");

    printf("B = [");
    for (int i = 0; i < print_num; i++) {
        printf("[ ");
        for (int j = 0; j < print_num; j++) {
            printf("%f, ", h_B[DSIZE_Y*j + i]);
        }
        printf("]\n");
    }
    printf("]\n");

    printf("A+B = [");
    for (int i = 0; i < print_num; i++) {
        printf("[ ");
        for (int j = 0; j < print_num; j++) {
            printf("%f, ", h_C[DSIZE_Y*j + i]);
        }
        printf("]\n");
    }
    printf("]\n");
    // Free the memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}