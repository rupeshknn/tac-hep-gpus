/*

compile: "c++ -o c_code c_code.cpp"
run check: "./c_code -check"
run code with size 512 (default): "./c_code"
run code with size 2048: "./c_code -size 2048"

*/

#include <stdio.h>
#include <stdlib.h> 
#include <cstring>
#include <chrono>

int* stencil_matmul(bool isrnad, int radius, const int DSIZE)
{
    int *h_A, *h_B, *h_Ac, *h_Bc, *h_C;
    int print_num = 3;
    // Create and allocate memory for host and device pointers 
    h_A = new int[DSIZE * DSIZE];
    h_B = new int[DSIZE * DSIZE];
    h_Ac = new int[DSIZE * DSIZE];
    h_Bc = new int[DSIZE * DSIZE];
    h_C = new int[DSIZE * DSIZE];

    // Fill in the matrices
    for (int i = 0; i < DSIZE; i++) {
        for (int j = 0; j < DSIZE; j++) {
            if (isrnad){
                h_A[i*DSIZE + j] = rand() % 10;
                h_B[i*DSIZE + j] = rand() % 10;
            } else{
                h_A[i*DSIZE + j] = 1;
                h_B[i*DSIZE + j] = 1;
            }
            h_Ac[i*DSIZE + j] = h_A[i*DSIZE + j];
            h_Bc[i*DSIZE + j] = h_B[i*DSIZE + j];
            h_C[i*DSIZE + j] = 0;
        }
    }
    int tempA = 0, tempB = 0;
    for (int idx = radius; idx < DSIZE-radius; idx++ ){
        for (int idy = radius; idy < DSIZE-radius; idy++ ){
            tempA = -h_A[idx*DSIZE + idy];
            tempB = -h_B[idx*DSIZE + idy];
            for (int idr = -radius; idr < radius+1; idr++ ){
                tempA += h_A[(idx+idr)*DSIZE + idy] + h_A[idx*DSIZE + idy+idr];
                tempB += h_B[(idx+idr)*DSIZE + idy] + h_B[idx*DSIZE + idy+idr];
            }
            h_Ac[idx*DSIZE + idy] = tempA;
            h_Bc[idx*DSIZE + idy] = tempB;
        }
    }

    for (int i=0; i<DSIZE; i++){
        for (int j=0; j<DSIZE; j++){
            h_C[i*DSIZE+j] = 0;
            for (int k=0; k<DSIZE; k++){
                h_C[i*DSIZE+j] += h_Ac[i*DSIZE+k]*h_Bc[k*DSIZE+j];
            }
        }
    }

    return h_C;
}

int main(int argc, char const *argv[]){
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
        C = stencil_matmul(false, 1, DSIZE);
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

        C = stencil_matmul(true, radius, DSIZE);

        auto finish = std::chrono::steady_clock::now();
        double elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(finish - start).count();
        printf("time to run = %.2f S\n\n", elapsed_seconds);
    }

}