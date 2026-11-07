#include "config.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

const int BLOCK_SIZE = 16;

#define CHECK_CUDA(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", " \
                  << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
}

// CUDA Kernel
__global__ void matrixMulCUDA(const int64_t* A, const int64_t* B, int64_t* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        int64_t sum = 0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

bool compare_results(const std::vector<int64_t>& ref, const std::vector<int64_t>& target, const char* name) {
    for (size_t i = 0; i < ref.size(); ++i) {
        if (ref[i] != target[i]) {
            std::cout << "Mismatch in " << name << " at index " << i << " Expected: " << ref[i] << " Got: " << target[i] << std::endl;
            return false;
        }
    }
    std::cout << "Verification " << name << ": SUCCESS" << std::endl;
    return true;
}

int main() {
    size_t size_bytes = N * N * sizeof(int64_t);
    
    std::cout << "Scalar, AVX2, Cuda - Bench" << std::endl;
    std::cout << "Matrix Size: " << N << " x " << N << std::endl;
    std::cout << std::fixed << std::setprecision(4);

    // выделение памяти
    std::vector<int64_t> h_A(N * N);
    std::vector<int64_t> h_B(N * N);
    std::vector<int64_t> h_C_scalar(N * N);
    std::vector<int64_t> h_C_avx(N * N);
    std::vector<int64_t> h_C_gpu(N * N);

    std::cout << "Generating matrices..." << std::endl;
    generate_matrix(h_A);
    generate_matrix(h_B);

    // скалярное cpu
    std::cout << "\n1. Scalar Computation (CPU)" << std::endl;
    std::cout << "Calculating..." << std::endl;
    auto start_sca = std::chrono::high_resolution_clock::now();
    
    scalar_matrix_multiplication(h_A, h_B, h_C_scalar);
    
    auto end_sca = std::chrono::high_resolution_clock::now();
    double time_sca = std::chrono::duration<double>(end_sca - start_sca).count();
    std::cout << "Scalar Time: " << time_sca << " s" << std::endl;

    // векторное avx2 cpu
    std::cout << "\n2. AVX2 Computation (CPU)" << std::endl;
    auto start_avx = std::chrono::high_resolution_clock::now();
    
    avx2_matrix_multiplication(h_A, h_B, h_C_avx);
    
    auto end_avx = std::chrono::high_resolution_clock::now();
    double time_avx = std::chrono::duration<double>(end_avx - start_avx).count();
    std::cout << "AVX2 Time:   " << time_avx << " s" << std::endl;

    // Проверка Скаляр vs AVX (Пункт 4 задания)
    compare_results(h_C_scalar, h_C_avx, "Scalar vs AVX");

    // cuda gpu
    std::cout << "\n3. CUDA Computation (GPU)" << std::endl;
    int64_t *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size_bytes));
    CHECK_CUDA(cudaMalloc(&d_B, size_bytes));
    CHECK_CUDA(cudaMalloc(&d_C, size_bytes));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), size_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), size_bytes, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrixMulCUDA<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float time_gpu_ms = 0;
    cudaEventElapsedTime(&time_gpu_ms, start, stop);
    double time_gpu = time_gpu_ms / 1000.0;
    
    std::cout << "GPU Time:    " << time_gpu << " s" << std::endl;

    CHECK_CUDA(cudaMemcpy(h_C_gpu.data(), d_C, size_bytes, cudaMemcpyDeviceToHost));
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    // Проверка AVX vs GPU
    compare_results(h_C_avx, h_C_gpu, "AVX vs GPU");

    // результаты
    std::cout << "\nRESULTS" << std::endl;
    std::cout << "Scalar: " << time_sca << " s" << std::endl;
    std::cout << "AVX2:   " << time_avx << " s (Speedup vs Scalar: " << time_sca / time_avx << "x)" << std::endl;
    std::cout << "GPU:    " << time_gpu << " s (Speedup vs Scalar: " << time_sca / time_gpu << "x)" << std::endl;
    std::cout << "GPU Speedup vs AVX2: " << time_avx / time_gpu << "x" << std::endl;

    size_t ram_usage = size_bytes * 5; // A, B, C_sca, C_avx, C_gpu
    std::cout << "\nTotal RAM used: " << ram_usage / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "GPU VRAM used: " << (size_bytes * 3) / (1024.0 * 1024.0) << " MB" << std::endl;

    return 0;
}