#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_fp8.h>
#include <hip/amd_detail/amd_hip_bf16.h>
#include <hip/hip_cooperative_groups.h>
#include <rocwmma/rocwmma.hpp>
// #include <rocwmma/rocwmma_transforms.hpp>

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
        // case 0: {
        // ref_kernel<<<dim3((M+15)/16, (N+15)/16), dim3(16, 16)>>>(d_A_half, d_B_half, d_As, d_Bs, d_C, M, N, K);
        // break;
        // }

        case 0: { // normal
            int num_warp_x = NUM_WARP_X;
            int num_warp_y = NUM_WARP_Y;
            dim3 block_size((WARP_SIZE * num_warp_x), num_warp_y);
            dim3 grid_size((N + num_warp_x * WMMA_N - 1) / (num_warp_x * WMMA_N), 
                           (M + num_warp_y * WMMA_M - 1) / (num_warp_y * WMMA_M));

            in_smem_kernel<<<grid_size, block_size, 0, 0>>> (d_A_half, d_B_half, d_As, d_Bs, (rocwmma::bfloat16_t*)d_C, M, N, K);
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

        case 2: { // normal
            int num_warp_x = NUM_WARP_X;
            int num_warp_y = NUM_WARP_Y;
            dim3 block_size((WARP_SIZE * num_warp_x), num_warp_y);
            dim3 grid_size((N + num_warp_x * WMMA_N - 1) / (num_warp_x * WMMA_N), 
                           (M + num_warp_y * WMMA_M - 1) / (num_warp_y * WMMA_M));

            hipEvent_t start, stop;
            (void) hipEventCreate(&start);
            (void) hipEventCreate(&stop);
            
            float total_time = 0.0f;

            for (int i = 0; i < 10; ++i) {
                int num_warp_x = NUM_WARP_X;
                int num_warp_y = NUM_WARP_Y;
                dim3 block_size((WARP_SIZE * num_warp_x), num_warp_y);
                dim3 grid_size((N + num_warp_x * WMMA_N - 1) / (num_warp_x * WMMA_N * 2), 
                               (M + num_warp_y * WMMA_M - 1) / (num_warp_y * WMMA_M));
            
                (void) hipEventRecord(start);
            
                in_smem_kernel<<<grid_size, block_size, 0, 0>>>(
                    d_A_half, d_B_half, d_As, d_Bs, (rocwmma::bfloat16_t*)d_C, M, N, K
                );
            
                (void) hipEventRecord(stop);
                (void) hipEventSynchronize(stop);
            
                float milliseconds = 0;
                (void) hipEventElapsedTime(&milliseconds, start, stop);
            
                if (i > 0) total_time += milliseconds;  // Skip first run for warm-up
                std::cout << "Run " << i + 1 << ": " << milliseconds << " ms" << std::endl;
            }
            
            float average = 1000 * (total_time / 9.0f);
            std::cout << "Average of last 9 runs: " << average << " us" << std::endl;
            
            (void) hipEventDestroy(start);
            (void) hipEventDestroy(stop);

            break;
        }

        case 3: { // reverse
            int num_warp_x = NUM_WARP_X;
            int num_warp_y = NUM_WARP_Y;
            dim3 block_size((WARP_SIZE * num_warp_x), num_warp_y);
            dim3 grid_size((N + num_warp_x * WMMA_N - 1) / (num_warp_x * WMMA_N), 
                           (M + num_warp_y * WMMA_M - 1) / (num_warp_y * WMMA_M));

            hipEvent_t start, stop;
            (void) hipEventCreate(&start);
            (void) hipEventCreate(&stop);
            
            float total_time = 0.0f;

            for (int i = 0; i < 10; ++i) {
                int num_warp_x = NUM_WARP_X;
                int num_warp_y = NUM_WARP_Y;
                dim3 block_size((WARP_SIZE * num_warp_x), num_warp_y);
                dim3 grid_size((N + num_warp_x * WMMA_N - 1) / (num_warp_x * WMMA_N * 2), 
                               (M + num_warp_y * WMMA_M - 1) / (num_warp_y * WMMA_M));
            
                (void) hipEventRecord(start);
            
                in_smem_kernel<<<grid_size, block_size, 0, 0>>>(
                    d_A_half, d_B_half, d_As, d_Bs, (rocwmma::bfloat16_t*)d_C, M, N, K
                );
            
                (void) hipEventRecord(stop);
                (void) hipEventSynchronize(stop);
            
                float milliseconds = 0;
                (void) hipEventElapsedTime(&milliseconds, start, stop);
            
                if (i > 0) total_time += milliseconds;  // Skip first run for warm-up
                std::cout << "Run " << i + 1 << ": " << milliseconds << " ms" << std::endl;
            }
            
            float average = 1000 * (total_time / 9.0f);
            std::cout << "Average of last 9 runs: " << average << " us" << std::endl;
            
            (void) hipEventDestroy(start);
            (void) hipEventDestroy(stop);

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
    // size_t M = 64 * 4;
    // size_t N = 64 * 10;
    // size_t K = 128 * 56;


    // size_t M = 128 * 56;
    // size_t N = 128 * 56;
    // size_t K = 128 * 4; 


    // size_t M = 1024;
    // size_t N = 1536;
    // size_t K = 7168;
    
    // size_t M = 1024;
    // size_t N = 4608;
    // size_t K = 7168;    

    // size_t K = 7168; size_t M = 1024; size_t N = 1536;
    // size_t K = 7168; size_t M = 1024; size_t N = 576;
    // size_t K = 7168; size_t M = 1024; size_t N = 512;


    // size_t K = 1024; size_t M = 1024; size_t N = 4096;
    // size_t K = 1024; size_t M = 1024; size_t N = 7168;



    // size_t K = 2048; size_t M = 1024; size_t N = 7168;// balance 100026 restricted 98864
    size_t K = 2304; size_t M = 1024; size_t N = 7168;// balance 108771      
    // size_t K = 7168; size_t M = 6144; size_t N = 1536;
    // size_t K = 2304; size_t M = 7168; size_t N = 1024;
    // size_t K = 7168; size_t M = 6144; size_t N = 512;

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

    // gemmGPU(matC_ref, matA_colmaj, matB, matAs, matBs, M, N, K, 0);
    gemmGPU(matC_my, matA_colmaj, matB, matAs, matBs, M, N, K, OPTION);

}
