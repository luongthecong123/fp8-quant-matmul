#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_fp8.h>
#include <hip/amd_detail/amd_hip_bf16.h>
#include <hip/hip_cooperative_groups.h>
#include <rocwmma/rocwmma.hpp>
#include <rocwmma/rocwmma_transforms.hpp>

namespace cg = cooperative_groups;
using floatx16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;

#include "utils.h"

#include "mn_large.hip"

constexpr const int BLOCK = 128;

__global__ void convert_float_to_fp8(const float* input, __hip_fp8_e4m3_fnuz* output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = (__hip_fp8_e4m3_fnuz)input[idx];
    }
}

// __global__ void test_fp8() {
//     auto frag_a = rocwmma::fragment<rocwmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __hip_fp8_e4m3_fnuz, rocwmma::row_major>();

//     printf("Number of elements: %d\n", frag_a.num_elements);
// }

__global__ void ref_kernel(const __hip_fp8_e4m3_fnuz* a, const __hip_fp8_e4m3_fnuz* b, const float* as, const float* bs, 
                   __hip_bfloat16* c, int m, int n, int k) {
    int cx = threadIdx.x + blockDim.x * blockIdx.x;
    int cy = threadIdx.y + blockDim.y * blockIdx.y;
    if(cx >= m || cy >= n) return;
    
    int sn = (n + BLOCK - 1) / BLOCK;
    
    float result = 0;
    for(int i = 0; i < k; i += BLOCK) {
        float block_result = 0;
        for(int ii = 0; ii < BLOCK; ++ii) {
            float av = (float)a[cx + (i + ii) * m];
            float bv = (float)b[cy + (i + ii) * n];
            block_result += av * bv; 
        }        
        result += block_result * as[cx + i/BLOCK * m] * bs[cy/BLOCK + i/BLOCK * sn];
    }
    c[cx * n + cy] = (__hip_bfloat16)result;
}

void gemmGPU(std::vector<__hip_bfloat16>& matC, const std::vector<float>& matA, const std::vector<float>& matB, 
    const std::vector<float>& matAs, const std::vector<float>& matBs, size_t M, size_t N, size_t K, int option){
    
    // Host pointers
    const float* matA_ptr = matA.data();
    const float* matB_ptr = matB.data();
    __hip_bfloat16* matC_ptr = matC.data();
    const float* matAs_ptr = matAs.data();
    const float* matBs_ptr = matBs.data(); 
    
    // Device pointers
    float *d_A, *d_B, *d_As, *d_Bs;
    __hip_bfloat16 *d_C;
    
    // Allocate memory
    (void) hipMalloc(&d_A, sizeof(float) * matA.size());
    (void) hipMalloc(&d_B, sizeof(float) * matB.size());
    (void) hipMalloc(&d_C, sizeof(__hip_bfloat16) * matC.size());
    (void) hipMalloc(&d_As, sizeof(float) * matAs.size());
    (void) hipMalloc(&d_Bs, sizeof(float) * matBs.size());  
    
    // Copy data to device memory
    (void) hipMemcpy(d_A, matA_ptr, sizeof(float) * matA.size(), hipMemcpyHostToDevice);
    (void) hipMemcpy(d_B, matB_ptr, sizeof(float) * matB.size(), hipMemcpyHostToDevice);
    (void) hipMemcpy(d_As, matAs_ptr, sizeof(float) * matAs.size(), hipMemcpyHostToDevice);
    (void) hipMemcpy(d_Bs, matBs_ptr, sizeof(float) * matBs.size(), hipMemcpyHostToDevice);
    
    // Convert float to fp8
    __hip_fp8_e4m3_fnuz *d_A_half, *d_B_half;
    (void) hipMalloc(&d_A_half, sizeof(__hip_fp8_e4m3_fnuz) * matA.size());
    (void) hipMalloc(&d_B_half, sizeof(__hip_fp8_e4m3_fnuz) * matB.size());

    convert_float_to_fp8<<<M * K / 256, 256>>>(d_A, d_A_half, M * K);
    convert_float_to_fp8<<<N * K / 256, 256>>>(d_B, d_B_half, N * K);
    
    switch(option){ // Reference
        case 2: {
        ref_kernel<<<dim3((M+15)/16, (N+15)/16), dim3(16, 16)>>>(d_A_half, d_B_half, d_As, d_Bs, d_C, M, N, K);
        break;
        }

        case 1: { // reverse
            int num_warp_x = NUM_WARP_X;
            int num_warp_y = NUM_WARP_Y;
            dim3 block_size((WARP_SIZE * num_warp_x), num_warp_y);
            // dim3 grid_size(M / (num_warp_x * WMMA_N), N / (num_warp_y * WMMA_M));

            dim3 grid_size((M + num_warp_x * WMMA_N - 1) / (num_warp_x * WMMA_N), 
                           (N + num_warp_y * WMMA_M - 1) / (num_warp_y * WMMA_M));

            in_smem_kernel<<<grid_size, block_size, 0, 0>>> (d_A_half, d_B_half, d_As, d_Bs, (rocwmma::bfloat16_t*)d_C, M, N, K);
            break;                           
        }


        case 0: { // normal
            int num_warp_x = NUM_WARP_X;
            int num_warp_y = NUM_WARP_Y;
            dim3 block_size((WARP_SIZE * num_warp_x), num_warp_y);
            dim3 grid_size((N + num_warp_x * WMMA_N - 1) / (num_warp_x * WMMA_N), 
                           (M + num_warp_y * WMMA_M - 1) / (num_warp_y * WMMA_M));

            in_smem_kernel<<<grid_size, block_size, 0, 0>>> (d_A_half, d_B_half, d_As, d_Bs, (rocwmma::bfloat16_t*)d_C, M, N, K);
            break;
        } 
        
    }

    (void) hipMemcpy(matC_ptr, d_C, sizeof(__hip_bfloat16) * matC.size(), hipMemcpyDeviceToHost);

    // Free memory
    (void) hipFree(d_A);
    (void) hipFree(d_B);
    (void) hipFree(d_C);
    (void) hipFree(d_As);
    (void) hipFree(d_Bs);
    (void) hipFree(d_A_half);
    (void) hipFree(d_B_half);    

}

int main() {
    // size_t M = 64 * 12;
    // size_t N = 64 * 4;
    // size_t K = 384 * 18;

    size_t M = 128;
    size_t N = 128;
    size_t K = 128 * 2; 

    // size_t M = 768;
    // size_t N = 512;
    // size_t K = 512; 



    // size_t M = 128;
    // size_t N = 128;
    // size_t K = 7168;

    // size_t K = 7168; size_t M = 1024; size_t N = 512;
    // size_t K = 7168; size_t M = 1024; size_t N = 1536;
    // size_t K = 2304; size_t M = 1024; size_t N = 7168;// balance 110771 restricted 110571        
    // size_t K = 3584; size_t M = 6144; size_t N = 1536;
    const auto matA = genMat(M * K);
    const auto matB = genMat(K * N);

    std::vector<__hip_bfloat16> matC_my(M * N);
    std::vector<__hip_bfloat16> matC_ref(M * N);

    const auto matAs = genMat(M * K / 128);
    const auto matBs = genMat((K / 128) * ((N + 128 - 1) / 128));

    std::vector<float> matA_colmaj(M * K);
    row_to_col_major(matA, matA_colmaj, M, K);
    std::vector<float> matAs_colmaj(M * K / 128);
    row_to_col_major(matAs, matAs_colmaj, M, K / 128);

    // test_fp8<<<1, dim3 (64, 1)>>>();

    gemmGPU(matC_ref, matA_colmaj, matB, matAs, matBs, M, N, K, 2);
    gemmGPU(matC_my, matA_colmaj, matB, matAs, matBs, M, N, K, OPTION);

    float tolerance = 100;
    if (allClose(matC_ref, matC_my, tolerance)){
        std::cout<<"2 matrices are equal\n";
    }

    // std::cout<<"-------------- Mat ref-----------------"<<std::endl;
    // viewMat(matC_ref, M, N);

    // std::cout<<"-------------- Mat mine-----------------"<<std::endl;
    // viewMat(matC_my, M, N);
}
