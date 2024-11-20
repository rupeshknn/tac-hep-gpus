#include <stdio.h>
#include <stdlib.h> 

const int DSIZE_X = 10;
const int DSIZE_Y = 10;
const int radius = 3;
int main()
{
    int *h_A, *h_B, *h_Ac, *h_Bc, *h_C;
    int print_num = 3;
    // Create and allocate memory for host and device pointers 
    h_A = new int[DSIZE_X * DSIZE_Y];
    h_B = new int[DSIZE_X * DSIZE_Y];
    h_Ac = new int[DSIZE_X * DSIZE_Y];
    h_Bc = new int[DSIZE_X * DSIZE_Y];
    h_C = new int[DSIZE_X * DSIZE_Y];

    // Fill in the matrices
    for (int i = 0; i < DSIZE_X; i++) {
        for (int j = 0; j < DSIZE_Y; j++) {
            h_A[i*DSIZE_X + j] = rand() % 10;
            h_B[i*DSIZE_X + j] = rand() % 10;
            h_Ac[i*DSIZE_X + j] = h_A[i*DSIZE_X + j];
            h_Bc[i*DSIZE_X + j] = h_B[i*DSIZE_X + j];
            h_C[i*DSIZE_X + j] = 0;
        }
    }
    int tempA = 0, tempB = 0;
    for (int idx = radius; idx < DSIZE_X-radius; idx++ ){
        for (int idy = radius; idy < DSIZE_Y-radius; idy++ ){
            tempA = -h_A[idx*DSIZE_X + idy];
            tempB = -h_B[idx*DSIZE_X + idy];
            for (int idr = -radius; idr < radius+1; idr++ ){
                tempA += h_A[(idx+idr)*DSIZE_X + idy] + h_A[idx*DSIZE_X + idy+idr];
                tempB += h_B[(idx+idr)*DSIZE_X + idy] + h_B[idx*DSIZE_X + idy+idr];
            }
            h_Ac[idx*DSIZE_X + idy] = tempA;
            h_Bc[idx*DSIZE_X + idy] = tempB;
        }
    }

    for (int i=0; i<DSIZE_X; i++){
        for (int j=0; j<DSIZE_Y; j++){
            h_C[i*DSIZE_X+j] = 0;
            for (int k=0; k<DSIZE_X; k++){
                h_C[i*DSIZE_X+j] += h_Ac[i*DSIZE_X+k]*h_Bc[k*DSIZE_X+j];
            }
        }
    }

    printf("C = [");
    print_num = DSIZE_X;
    for (int i = 0; i < print_num; i++) {
        printf("     [");
        for (int j = 0; j < print_num; j++) {
            printf("%3d, ", h_Ac[DSIZE_Y*j + i]);
        }
        printf("\b\b  ]\n");
    }
    printf("]\n");
}