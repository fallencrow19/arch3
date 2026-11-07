#include "config.h"
#include <random>
#include <immintrin.h> 
#include <algorithm>
#include <iostream>

// Эмуляция умножения для AVX2
inline __m256i _mm256_mullo_epi64_emu(const __m256i a, const __m256i b) {
    __m256i lo_lo_prod = _mm256_mul_epu32(a, b);
    __m256i a_hi_shuf = _mm256_shuffle_epi32(a, 0xB1); 
    __m256i hi_lo_prod = _mm256_mul_epu32(a_hi_shuf, b);
    __m256i b_hi_shuf = _mm256_shuffle_epi32(b, 0xB1);
    __m256i lo_hi_prod = _mm256_mul_epu32(a, b_hi_shuf);
    __m256i cross_prod_sum = _mm256_add_epi64(hi_lo_prod, lo_hi_prod);
    __m256i cross_prod_shifted = _mm256_slli_epi64(cross_prod_sum, 32);
    return _mm256_add_epi64(lo_lo_prod, cross_prod_shifted);
}

void generate_matrix(std::vector<int64_t>& matrix) {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<int64_t> distrib(0, 10); // Маленькие числа, чтобы избежать переполнения
    for (int i = 0; i < N * N; ++i) {
        matrix[i] = distrib(gen);
    }
}

// скалярная
void scalar_matrix_multiplication(const std::vector<int64_t>& A, const std::vector<int64_t>& B, std::vector<int64_t>& C) {
    std::fill(C.begin(), C.end(), 0);
    
    // строка на столбец
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int64_t sum = 0;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// векторная avx2
void avx2_matrix_multiplication(const std::vector<int64_t>& A, const std::vector<int64_t>& B, std::vector<int64_t>& C) {
    std::fill(C.begin(), C.end(), 0);
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            __m256i a_vec = _mm256_set1_epi64x(A[i * N + k]);
            for (int j = 0; j < N; j += 4) {
                __m256i b_vec = _mm256_loadu_si256((const __m256i*)&B[k * N + j]);
                __m256i c_vec = _mm256_loadu_si256((const __m256i*)&C[i * N + j]);
                __m256i prod = _mm256_mullo_epi64_emu(a_vec, b_vec);
                c_vec = _mm256_add_epi64(c_vec, prod);
                _mm256_storeu_si256((__m256i*)&C[i * N + j], c_vec);
            }
        }
    }
}