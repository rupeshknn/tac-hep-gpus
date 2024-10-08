#include <stdio.h>


const int DSIZE = 40960;
const int block_size = 256;
const int grid_size = DSIZE/block_size;


__global__ void vector_swap(float *A, float *B, float *C, int DSIZE) {

    //FIXME:
    // Express the vector index in terms of threads and blocks
    int idx =  threadIdx.x + blockIdx.x * blockDim.x;
    // Swap the vector elements - make sure you are not out of range
    if (idx < DSIZE){
        C[idx] = A[idx];
        A[idx] = B[idx];
        B[idx] = C[idx];
    }
}


int main() {


    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
    int print_num = 10;
    h_A = new float[DSIZE];
    h_B = new float[DSIZE];
    h_C = new float[DSIZE];


    for (int i = 0; i < DSIZE; i++) {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
        h_C[i] = 0;
    }

    printf("Old A = [");
    for (int i = 0; i<print_num; i++){
        printf("%f, ", h_A[i]);
    }
    printf("]\n");

    printf("Old B = [");
    for (int i = 0; i<print_num; i++){
        printf("%f, ", h_B[i]);
    }
    printf("]\n");

    // Allocate memory for host and device pointers 
    cudaMalloc(&d_A, DSIZE*sizeof(float));
    cudaMalloc(&d_B, DSIZE*sizeof(float));
    cudaMalloc(&d_C, DSIZE*sizeof(float));

    // Copy from host to device
    cudaMemcpy(d_A, h_A, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, DSIZE*sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    vector_swap<<<grid_size, block_size>>>(d_A, d_B, d_C, DSIZE);

    // Copy back to host
    cudaMemcpy(h_A, d_A, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B, d_B, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);

    // Print and check some elements to make sure swapping was successfull
    printf("New A = [");
    for (int i = 0; i<print_num; i++){
        printf("%f, ", h_A[i]);
    }
    printf("]\n");

    printf("New B = [");
    for (int i = 0; i<print_num; i++){
        printf("%f, ", h_B[i]);
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
