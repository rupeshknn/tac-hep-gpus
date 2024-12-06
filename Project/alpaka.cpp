/*
# Alpaka implementation.

cuda compile: "nvcc -x cu -std=c++17 -O2 -g --expt-relaxed-constexpr -I$ALPAKA_BASE/include -DALPAKA_ACC_GPU_CUDA_ENABLED alpaka.cpp -o alpaka_cuda"
run check: "./alpaka_cuda -check"
run code with size 512 (default): "./alpaka_cuda"
run code with size 4096: "./alpaka_cuda -size 4096"

cpu compile: "g++ -std=c++17 -O2 -g -I$ALPAKA_BASE/include -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED alpaka.cpp -o alpaka_cpu"
run check: "./alpaka_cpu -check"
run code with size 512 (default): "./alpaka_cpu"
run code with size 2048: "./alpaka_cpu -size 2048"
 */

#include <cassert>
#include <cstdio>
#include <random>

#include <alpaka/alpaka.hpp>

#include "config.h"
#include "WorkDiv.hpp"
#include <chrono>


struct stencil2D {
    template <typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                T const* __restrict__ d_A,
                                T* __restrict__ d_Aout,
                                int radius,
                                Vec2D size) const {
        for (auto ndindex : alpaka::uniformElementsND(acc, size)) {
            auto idx = ndindex[0];
            auto idy = ndindex[1];
            auto DSIZE = size[1];
            // auto index = (ndindex[0] * size[1] + ndindex[1])

            if (idx >= radius && idx < DSIZE - radius && idy >= radius && idy < DSIZE - radius) {
                int temp = -d_A[idx * DSIZE + idy];
                for (int r = -radius; r < radius+1; r++) {
                    temp += d_A[(idx + r) * DSIZE + idy] + d_A[idx * DSIZE + idy + r];
                }
                d_Aout[idx * DSIZE + idy] = temp;
            }
        }
    }
};

struct matrixmul {
    template <typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                T const* __restrict__ d_A,
                                T const* __restrict__ d_B,
                                T* __restrict__ d_ABout,
                                Vec2D size) const {
        for (auto ndindex : alpaka::uniformElementsND(acc, size)) {
            auto idx = ndindex[0];
            auto idy = ndindex[1];
            auto DSIZE = size[1];
            // auto index = (ndindex[0] * size[1] + ndindex[1])
            if (idy < DSIZE && idx < DSIZE) {
                    int sum = 0;
                    for (int k = 0; k < DSIZE; ++k) {
                        sum += d_A[idy * DSIZE + k] * d_B[k * DSIZE + idx];
                    }
                    d_ABout[idy * DSIZE + idx] = sum;
            }
            
        }
    }
};

// Host function
void stencilMatmul(Host host, Platform platform, Device device, bool isRand, int radius, const int DSIZE, int* out) {

    // 3-dimensional and linearised buffer size
    Vec2D ndsize = {DSIZE, DSIZE};
    uint32_t size = ndsize.prod();
    auto h_A = alpaka::allocMappedBuf<int, uint32_t>(host, platform, size);
    auto h_B = alpaka::allocMappedBuf<int, uint32_t>(host, platform, size);
    auto h_As = alpaka::allocMappedBuf<int, uint32_t>(host, platform, size);
    auto h_Bs = alpaka::allocMappedBuf<int, uint32_t>(host, platform, size);
    auto h_C = alpaka::allocMappedBuf<int, uint32_t>(host, platform, size);

    for (uint32_t i = 0; i < size; ++i) {
        h_A[i] = isRand ? rand() % 10 : 1;
        h_B[i] = isRand ? rand() % 10 : 1;
        h_As[i] = h_A[i];
        h_Bs[i] = h_B[i];
    }

    // run the test the given device
    auto queue = Queue{device};

    // allocate input and output buffers on the device
    auto d_A = alpaka::allocAsyncBuf<int, uint32_t>(queue, size);
    auto d_B = alpaka::allocAsyncBuf<int, uint32_t>(queue, size);
    auto d_As = alpaka::allocAsyncBuf<int, uint32_t>(queue, size);
    auto d_Bs = alpaka::allocAsyncBuf<int, uint32_t>(queue, size);
    auto d_C = alpaka::allocAsyncBuf<int, uint32_t>(queue, size);
    
    // copy the input data to the device; the size is known from the buffer objects
    alpaka::memcpy(queue, d_A, h_A);
    alpaka::memcpy(queue, d_B, h_B);
    alpaka::memcpy(queue, d_As, h_As);
    alpaka::memcpy(queue, d_Bs, h_Bs);

    alpaka::memset(queue, d_C, 0x00);

    // launch the 3-dimensional kernel
    auto div = makeWorkDiv<Acc2D>({32, 32}, {16, 16});
    std::cout << "Testing VectorAddKernel3D with vector indices with a grid of "
            << alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(div) << " blocks x "
            << alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(div) << " threads x "
            << alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(div) << " elements...\n";
    alpaka::exec<Acc2D>(
        queue, div, stencil2D{}, d_A.data(), d_As.data(), radius, ndsize);
    alpaka::exec<Acc2D>(
        queue, div, stencil2D{}, d_B.data(), d_Bs.data(), radius, ndsize);
    alpaka::wait(queue);
    alpaka::exec<Acc2D>(
        queue, div, matrixmul{}, d_As.data(), d_Bs.data(), d_C.data(), ndsize);

    // copy the results from the device to the host
    alpaka::memcpy(queue, h_C, d_C);

    // wait for all the operations to complete
    alpaka::wait(queue);
    // alpaka::memcpy(queue, out, h_C);
    for (uint32_t i = 0; i < size; ++i) {
        out[i] = h_C[i];
    }
}

int main(int argc, char const *argv[]) {
    // initialise the accelerator platform
    Platform platform;
    // require at least one device
    std::uint32_t n = alpaka::getDevCount(platform);
    if (n == 0) {
        exit(EXIT_FAILURE);
    }

    // use the single host device
    HostPlatform host_platform;
    Host host = alpaka::getDevByIdx(host_platform, 0u);
    std::cout << "Host:   " << alpaka::getName(host) << '\n';

    // use the first device
    Device device = alpaka::getDevByIdx(platform, 0u);
    std::cout << "Device: " << alpaka::getName(device) << '\n';


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
        printf("Matrix Size - %d\n", DSIZE);
        C = new int[DSIZE * DSIZE];
        stencilMatmul(host, platform, device, false, 1, DSIZE, C);
        if (C[0] != 10)
            printf("Mismatch at index [%d,%d], was: %d, should be: %d\n", 0,0, C[0], 10);
        else if (C[1] != 42)
            printf("Mismatch at index [%d,%d], was: %d, should be: %d\n", 0,1, C[1], 42);
        else if (C[11] != 202)
            printf("Mismatch at index [%d,%d], was: %d, should be: %d\n", 2,1, C[11], 202);
        else
            printf("Sucess!\n");

        // print test result
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
        // set DSIZE from CLI arg.
        DSIZE = dsize_set ? DSIZE: 512;
        printf("Matrix Size - %d\n", DSIZE);

        // Start clock
        auto start = std::chrono::steady_clock::now();
        const int radius = 3;
        C = new int[DSIZE * DSIZE];
        stencilMatmul(host, platform, device, true, radius, DSIZE, C);

        // Stop clock
        auto finish = std::chrono::steady_clock::now();
        double elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(finish - start).count();
        printf("time to run = %.2f S\n\n", elapsed_seconds);
    }
    // Free host memory
    delete[] C;

    return 0;
}