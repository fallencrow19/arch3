#ifndef CONFIG_H
#define CONFIG_H

#include <vector>
#include <cstdint>

const int N = 4096;

void generate_matrix(std::vector<int64_t>& matrix);
void scalar_matrix_multiplication(const std::vector<int64_t>& A, const std::vector<int64_t>& B, std::vector<int64_t>& C);
void avx2_matrix_multiplication(const std::vector<int64_t>& A, const std::vector<int64_t>& B, std::vector<int64_t>& C);

#endif