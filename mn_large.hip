#define K_JUMP 128
#define WMMA_M 32
#define WMMA_N 32
#define WMMA_K 16
#define NUM_WARP_X 4
#define NUM_WARP_Y 4
#define WARP_SIZE 64
#define WMMA_PLACEHOLDER 4
#define OPTION 0

__global__ void __launch_bounds__(1024, 8) in_smem_kernel(const __hip_fp8_e4m3_fnuz* A_ptr, const __hip_fp8_e4m3_fnuz* B_ptr, const float* As_ptr, const float* Bs_ptr, rocwmma::bfloat16_t* C_ptr, int M, int N, int K) {    
    /*
    A: M x K col-maj  --- As (1x128 scaling): M x K/128 col-maj
    B: K x N row-maj  --- Bs (128x128 scaling): K/128 x N/128 row-maj
    C: M x N row-maj
    M, N is divisible by 64 (some are only divisible by 32)
    K is divisible by 128
    18.5 KD of LDS
    */
    
    __shared__ __hip_fp8_e4m3_fnuz tileA[128][72]; // row-major
    __shared__ __hip_fp8_e4m3_fnuz tileB[128][72]; // col-major, = row-major transposed
    int lda = 72;
    int ldb = 72;
    __shared__ float tileAs[128];

    auto frag_acc_master = rocwmma::fragment<rocwmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>();
    rocwmma::fill_fragment(frag_acc_master, 0.f);

    cg::thread_block this_tb = cg::this_thread_block();
    auto group64 = cg::tiled_partition<64>(this_tb); // 1024 threads -> 64 x 16
    int t64x = group64.thread_rank();
    int t64y = group64.meta_group_rank(); 
    
    for (int kbase = 0; kbase < K; kbase += K_JUMP) {



        // ----------------------------- FIRST HALF --------------------------
        #pragma unroll
        for (int kmini_load = 0; kmini_load < 64; kmini_load += 16){  // 16 is num group
            #pragma unroll
            for (int idx = 0; idx < 2; idx++) {
                int off = idx * 64;
                tileA[t64x + off][kmini_load + t64y] = A_ptr[blockIdx.y * WMMA_M * NUM_WARP_Y + t64x + off + M * (kbase + kmini_load + t64y)];
                tileB[t64x + off][kmini_load + t64y] = B_ptr[(kbase + kmini_load + t64y) * N + blockIdx.x * WMMA_N * NUM_WARP_X + t64x + off];
            }
        }

        if (t64y == 0) {
            int sn = (N + 128 - 1) / 128;    
            float tileBs = Bs_ptr[(kbase / 128) * sn + ((blockIdx.x * NUM_WARP_X * WMMA_N) / 128)];
            float2 val = reinterpret_cast<const float2*>(As_ptr + blockIdx.y * WMMA_M * NUM_WARP_Y + M * (kbase / 128))[t64x];
            reinterpret_cast<float2*>(tileAs)[t64x] = make_float2(val.x * tileBs, val.y * tileBs);
        }       

        __syncthreads();


        // MFMA on SMEM
        auto frag_acc_block = rocwmma::fragment<rocwmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>() ;
        rocwmma::fill_fragment(frag_acc_block, 0.f);

        auto frag_a = rocwmma::fragment<rocwmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __hip_fp8_e4m3_fnuz, rocwmma::row_major>();
        auto frag_b = rocwmma::fragment<rocwmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __hip_fp8_e4m3_fnuz, rocwmma::col_major>(); 

        int warp_idx_y = threadIdx.y;
        int warp_idx_x = threadIdx.x / WARP_SIZE;

        #pragma unroll 2
        for (int kmini = 0; kmini < 64; kmini += WMMA_K) {           
            rocwmma::load_matrix_sync(frag_a, &tileA[0][0] + (warp_idx_y * WMMA_M) * lda + kmini, lda);            
            rocwmma::load_matrix_sync(frag_b, &tileB[0][0] + kmini + ldb * (warp_idx_x * WMMA_N), ldb);
            rocwmma::mma_sync(frag_acc_block, frag_a, frag_b, frag_acc_block);
        } // end for kmini

        __syncthreads();

        // ----------------------------- SECOND HALF --------------------------
        
        #pragma unroll
        for (int kmini_load = 0; kmini_load < 64; kmini_load += 16){
            #pragma unroll
            for (int idx = 0; idx < 2; idx++) {
                int off = idx * 64;
                tileA[t64x + off][kmini_load + t64y] = A_ptr[blockIdx.y * WMMA_M * NUM_WARP_Y + t64x + off + M * (kbase + kmini_load + 64 + t64y)];
                tileB[t64x + off][kmini_load + t64y] = B_ptr[(kbase + kmini_load + 64 + t64y) * N + blockIdx.x * WMMA_N * NUM_WARP_X + t64x + off];
            }
        }
        __syncthreads();

        #pragma unroll
        for (int kmini = 0; kmini < 64; kmini += WMMA_K) {           
            rocwmma::load_matrix_sync(frag_a, &tileA[0][0] + (warp_idx_y * WMMA_M) * lda + kmini, lda);            
            rocwmma::load_matrix_sync(frag_b, &tileB[0][0] + kmini + ldb * (warp_idx_x * WMMA_N), ldb);
            rocwmma::mma_sync(frag_acc_block, frag_a, frag_b, frag_acc_block);
        } // end for kmini

        // Apply scaler
        #pragma unroll
        for (int i = 0; i < 4; ++i){
            #pragma unroll
            for (int j = 0; j < 4; ++j){
                frag_acc_master.x[i * 4 + j] += tileAs[warp_idx_y * WMMA_M + 4 * (t64x / 32) + 8 * i + j] * frag_acc_block.x[i * 4 + j];
            }
        }            

        __syncthreads();
    } // end for kbase

    int warp_idx_x = threadIdx.x / WARP_SIZE;    
    int col_base_C = (blockIdx.x * NUM_WARP_X + warp_idx_x) * WMMA_N;
    if (col_base_C >= N) return;    

    int warp_idx_y = threadIdx.y;
    int row_base_C = (blockIdx.y * NUM_WARP_Y + warp_idx_y) * WMMA_M;
    if (row_base_C >= M) return;

    // Convert fp32 to bf16 and store
    auto frag_result = rocwmma::fragment<rocwmma::accumulator, WMMA_M, WMMA_N, WMMA_PLACEHOLDER, rocwmma::bfloat16_t>(); //Placeholder frag
    
    #pragma unroll
    for (int i = 0; i < frag_acc_master.num_elements; ++i) {
        frag_result.x[i] = (rocwmma::bfloat16_t)frag_acc_master.x[i];
    }
      
    rocwmma::store_matrix_sync(C_ptr + row_base_C * N + col_base_C, frag_result, N, rocwmma::mem_row_major);
}