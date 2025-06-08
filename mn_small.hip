#define K_JUMP 256
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 32
#define NUM_WARP_X 4
#define NUM_WARP_Y 4
#define WARP_SIZE 64
#define WMMA_PLACEHOLDER 8

__global__ void __launch_bounds__(1024, 8) k_large_kernel(const __hip_fp8_e4m3_fnuz* A_ptr, const __hip_fp8_e4m3_fnuz* B_ptr, const float* As_ptr, const float* Bs_ptr, rocwmma::bfloat16_t* C_ptr, int M, int N, int K) {
    /*
    Perform B @ A.T = C.T, so we can store scale As in a VGPR of each thread in a warp
    B: NxK col-maj -- Bs (128x128 scaling): N/128 x K/128 col-maj
    A.T: KxM row-maj -- As (1x128 scaling): K/ 128 x M row-maj
    C.T: NxM col-maj
    M, N are divisible by 64, K is divisible by 256
    */
    int sn = (N + 128 - 1) / 128;

    __shared__ __hip_fp8_e4m3_fnuz tileA[64][256]; // col-maj = row-maj transposed 
    __shared__ __hip_fp8_e4m3_fnuz tileB[64][256]; // row-maj

    auto frag_acc_master = rocwmma::fragment<rocwmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>();
    rocwmma::fill_fragment(frag_acc_master, 0.f);

    cg::thread_block this_tb = cg::this_thread_block();

    for (int kbase = 0; kbase < K; kbase += K_JUMP) {
        int warp_idx_y = threadIdx.y;
        int row_base_C = (blockIdx.y * NUM_WARP_Y + warp_idx_y) * WMMA_M;
    
        int warp_idx_x = threadIdx.x / WARP_SIZE;    
        int col_base_C = (blockIdx.x * NUM_WARP_X + warp_idx_x) * WMMA_N;

        int t64x = this_tb.thread_rank() % 64;
        int t64y = this_tb.thread_rank() / 64;

        #pragma unroll 4
        for (int kmini = 0; kmini < 128; kmini += 16){ // 16 is num group
            int K_idx_shifty = (kmini + t64y + 8 * (t64x % 16)) % 128;            
            tileB[t64x][K_idx_shifty] = B_ptr[(kbase + kmini + t64y) * N + blockIdx.y * WMMA_N * NUM_WARP_Y + t64x];   
            tileB[t64x][128 + K_idx_shifty] = B_ptr[(128 + kbase + kmini + t64y) * N + blockIdx.y * WMMA_N * NUM_WARP_Y + t64x];
        }   
        #pragma unroll
        for (int kmini = 0; kmini < 128; kmini += 16){ // 16 is num group
            int K_idx_shifty = (kmini + t64y + 8 * (t64x % 16)) % 128;            
            tileA[t64x][K_idx_shifty] = A_ptr[blockIdx.x * WMMA_M * NUM_WARP_X + t64x + M * (kbase + kmini + t64y)];       
            tileA[t64x][128 + K_idx_shifty] = A_ptr[blockIdx.x * WMMA_M * NUM_WARP_X + t64x + M * (128 + kbase + kmini + t64y)];  
        } 

        float tileAs1 = As_ptr[(kbase / 128) * M + col_base_C + t64x % WMMA_N];
        float tileBs1 = Bs_ptr[(kbase / 128) * sn + (blockIdx.y * NUM_WARP_Y * WMMA_M) / 128];
        float scale1 = tileAs1 * tileBs1;
        float tileAs2 = As_ptr[(1 + kbase / 128) * M + col_base_C + t64x % WMMA_N];
        float tileBs2 = Bs_ptr[(1 + kbase / 128) * sn + (blockIdx.y * NUM_WARP_Y * WMMA_M) / 128];
        float scale2 = tileAs2 * tileBs2;

        __syncthreads(); 
        // MFMA

        // MFMA on SMEM
        auto frag_acc_block = rocwmma::fragment<rocwmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>() ;
        rocwmma::fill_fragment(frag_acc_block, 0.f);

        auto frag_a = rocwmma::fragment<rocwmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __hip_fp8_e4m3_fnuz, rocwmma::row_major>();
        auto frag_b = rocwmma::fragment<rocwmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __hip_fp8_e4m3_fnuz, rocwmma::col_major>();

        #pragma unroll
        for (int kmini = 0; kmini < 128; kmini += WMMA_K){
            int tileB_row = warp_idx_y * WMMA_M + t64x % 16;
            int K_idx_base = kmini + (t64x / 16) * 8;
            int K_idx_shiftyB = (K_idx_base + 8 * tileB_row) % 128;
            // memcpy(&frag_a.x[0], &tileB[tileB_row][K_idx_shiftyB], sizeof(long));
            memcpy(&frag_a.x[0], &tileB[0][0] + tileB_row * 256 + K_idx_shiftyB, sizeof(long)); 

            int tileA_col = warp_idx_x * WMMA_N + t64x % 16;
            // int K_idx_shiftyA = (K_idx_base + 8 * tileA_col) % 128;
            // memcpy(&frag_b.x[0], &tileA[tileA_col][K_idx_shiftyB], sizeof(long));
            memcpy(&frag_b.x[0], &tileA[0][0] + tileA_col * 256 + K_idx_shiftyB, sizeof(long));
            rocwmma::mma_sync(frag_acc_block, frag_a, frag_b, frag_acc_block);
        }


        // Mult scalers
        #pragma unroll
        for (int i = 0; i < frag_acc_master.num_elements; ++i){
            frag_acc_master.x[i] = fmaf(scale1, frag_acc_block.x[i], frag_acc_master.x[i]);            
        }

        
        auto frag_acc_block2 = rocwmma::fragment<rocwmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>() ;
        rocwmma::fill_fragment(frag_acc_block2, 0.f);

        #pragma unroll
        for (int kmini = 0; kmini < 128; kmini += WMMA_K){
            int tileB_row = warp_idx_y * WMMA_M + t64x % 16;
            int K_idx_base = kmini + (t64x / 16) * 8;
            int K_idx_shiftyB = (K_idx_base + 8 * tileB_row) % 128;
            // memcpy(&frag_a.x[0], &tileB[tileB_row][128 + K_idx_shiftyB], sizeof(long));
            memcpy(&frag_a.x[0], &tileB[0][0] + tileB_row * 256 + 128 + K_idx_shiftyB, sizeof(long)); 

            int tileA_col = warp_idx_x * WMMA_N + t64x % 16;
            // int K_idx_shiftyA = (K_idx_base + 8 * tileA_col) % 128;
            // memcpy(&frag_b.x[0], &tileA[tileA_col][128 + K_idx_shiftyB], sizeof(long));
            memcpy(&frag_b.x[0], &tileA[0][0] + tileA_col * 256 + 128 + K_idx_shiftyB, sizeof(long));

            rocwmma::mma_sync(frag_acc_block2, frag_a, frag_b, frag_acc_block2);
        } 


        // Mult scalers
        #pragma unroll
        for (int i = 0; i < frag_acc_master.num_elements; ++i){
            frag_acc_master.x[i] = fmaf(scale2, frag_acc_block2.x[i], frag_acc_master.x[i]);            
        }


    }// end for (int kbase = 0; kbase < K; kbase += 128) 
    int warp_idx_y = threadIdx.y;
    int row_base_C = (blockIdx.y * NUM_WARP_Y + warp_idx_y) * WMMA_M;
    if (row_base_C >= N) return;

    int warp_idx_x = threadIdx.x / WARP_SIZE;    
    int col_base_C = (blockIdx.x * NUM_WARP_X + warp_idx_x) * WMMA_N;
    if (col_base_C >= M) return; 
        
    // Convert fp32 to bf16 and store
    // TODO: change WMMA_PLACEHOLDER from 8 to 16
    auto frag_result = rocwmma::fragment<rocwmma::accumulator, WMMA_M, WMMA_N, WMMA_PLACEHOLDER, rocwmma::bfloat16_t>(); //Placeholder frag
    
    #pragma unroll
    for (int i = 0; i < frag_acc_master.num_elements; ++i) {
        frag_result.x[i] = (rocwmma::bfloat16_t)frag_acc_master.x[i];
    }
       
    rocwmma::store_matrix_sync(C_ptr + row_base_C + N * col_base_C, frag_result, N, rocwmma::mem_col_major);
}