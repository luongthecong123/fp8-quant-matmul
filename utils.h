#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <hip/amd_detail/amd_hip_bf16.h>

void viewMat(const std::vector<__hip_bfloat16>& vec, const size_t M, const size_t N){
    for (size_t i = 0; i < M; i++){
        std::cout << "  [";
        for (size_t j = 0; j < N; j++){
            std::cout << std::setw(8) << std::setprecision(4) << std::fixed << (float)vec[i * N + j];
            if (j != N - 1){
                std::cout << ", ";
            }
        }
        std::cout << "]";
        if (i != M - 1) {
            std::cout << ",\n";
        }
    }
    std::cout << "\n";
}

std::vector<float> genMat(size_t len_1D){
    std::default_random_engine e(7); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> uniform_dist(-5.f, 5.f);
    std::vector<float> vec(len_1D);
    for (size_t i = 0; i < len_1D; i++){
        vec.at(i) = uniform_dist(e);
    }
    return vec;
}

bool allClose(const std::vector<__hip_bfloat16>& matA, const std::vector<__hip_bfloat16>& matB, const float& tolerance){
    if (matA.size() != matB.size()){
        return false;
    }

    for (int i = 0; i < matA.size(); i++){
        if (std::abs((float)(matA.at(i) - matB.at(i))) > tolerance){
            std::cout << i << "---" << (float)matA.at(i) << "---" <<(float)matB.at(i) <<std::endl;
            return false;
        }
    }
    return true;
}

void row_to_col_major(const std::vector<float>& row_major_input, std::vector<float>& col_major_output, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            col_major_output[j*M + i] = row_major_input[i*N + j];
        }
    }
}

void col_to_row_major(const std::vector<float>& col_major_input, std::vector<float>& row_major_output, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            row_major_output[i*N + j] = col_major_input[j*M + i];
        }
    }
}

void gemmCPU(std::vector<float>& matC, const std::vector<float>& matA, const std::vector<float>& matB, 
    size_t M, size_t N, size_t K){
    for (int i = 0; i < M; i++){ // C[i][j] = sigma A[i][t] * B[t][j]
        for (int j = 0; j < N; j++){
            float sum = 0;
            for (int t = 0; t < K; t++){ // Common dimension
                sum += matA[i * K + t] * matB[t * N + j];
            }
            matC[i * N + j] = sum;
        }
    }
}