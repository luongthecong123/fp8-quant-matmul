## Table of Contents
1. [Summary](#summary)
2. [FP8 Quantization and Scaling](#fp8-quantization-and-scaling)
3. [Solution](#solution)
   - [Overall](#overall)
   - [Maximize Hardware Utilization](#maximize-hardware-utilization)
   - [Kernel for Large M, N](#kernel-for-large-m-n)
   - [Kernel for Small M, N with Large K](#kernel-for-small-m-n-with-large-k)
4. [Quantization Strategies Comparison](#quantization-strategies-comparison)

## Summary

This repository contains my solution to the AMD FP8 GEMM Challenge, organized by the GPU Mode community. The solution is beginner-friendly HIP kernel with just 100 lines of code but still managed to achieve rank 5/163 by the end of the competition, mean geometric latency: 136 us.

| k    | m    | n    | latency (us) | k    | m    | n    | latency (us) |
|------|------|------|---------|------|------|---------|---------|
| 7168 | 1024 | 1536 | 147     | 7168 | 6144 | 1536    | 341     |
| 1536 | 1024 | 3072 | 50.6    | 1536 | 6144 | 3072    | 160     |
| 7168 | 1024 | 576  | 75.9    | 7168 | 6144 | 576     | 173     |
| 256  | 1024 | 7168 | 30.9    | 256  | 6144 | 7168    | 88.6    |
| 2048 | 1024 | 7168 | 106     | 2048 | 6144 | 7168    | 426     |
| 7168 | 1024 | 4608 | 193     | 7168 | 6144 | 4608    | 948     |
| 2304 | 1024 | 7168 | 120     | 2304 | 6144 | 7168    | 479     |
| 7168 | 1024 | 512  | 75.1    | 7168 | 6144 | 512     | 163     |
| 512  | 1024 | 4096 | 30.9    | 512  | 6144 | 4096    | 97.7    |

Future work for large M, N kernel:
- A separate kernel for matrix transpose to make GMEM --> SMEM faster in gemm kernel, so that we can tolerate lower occupancy
- Persistent kernel to overlap epilogue (row-wise scaling + write to GMEM) and the matmul of next matC's tile
- Reduce occupancy from 100% to 50%:
   - 1 CU manages 2 blocks, each block has 512 threads (50% occupancy), each thread now has 2x the amount of registers (64 --> 128)
   - 512 threads = 4x2 warps. Each warp holds registers for 2 32x32 matC tile. One block calculates 256x128 elements in matC
   - Memory load in 2 stages, each stage load 256x64 FP8 elems, 128x64 FP8 elems, 256 FP32 elems to tileA, tileB, tileAs respectively. Totaling (256x(64+8) + 128x(64+8) + 256x4)/1024 = 28 KB of SMEM per block.
  
In addition to the solution, the provided pytorch code showcases three FP8 scaling strategies:
- **Global scaling**
- **Block scaling** (128×128)
- **Row-wise block scaling** (1×128)

Usage:
To check correctness
```
hipcc -o check check.cu
./check
```
Note: Your rocm version must contain /opt/rocm/include/hip/hip_fp8.h (compatible with CDNA3).
To run benchmark:
```
hipcc -o sgemm sgemm.cu --offload-arch=gfx942 -std=c++20 -Rpass-analysis=kernel-resource-usage --save-temps
rocprof --stats --basenames on ./sgemm
rocprof -i rocprof_counters.txt ./sgemm
```
results.stats.csv will contain kernel duration in ns
rocprof_counters.csv will contain information about matrix core utilization (the higher the better), LDS bank conflicts,...
for multiple run benchmark, modify #define OPTION in .hip files.

## FP8 Quantization and Scaling

To avoid overflow when converting BF16 to FP8, a scaling factor is used to first bring the BF16 data to the representable range of FP8 by dividing the max and min value by a scaling factor. Then we can quantize the scaled values to FP8 and perform low-precision matrix multiplication for lower memory footprint and faster throughput. The result is accumulated in full precision FP32, multiply with scaling factor (in FP32) to bring it back to the correct value, before casting to BF16.
To improve accuracy, local scaling is used where each block of 128x128 elements have 1 distinct scaling factor. DeepSeek V3 improved accuracy further by implementing row-wise block scaling where for each block 1x128 in the weight matrix and normal block scaling for x matrix. In inference, row-wise block scaling factors for weight is calculated once during initialization.

## Solution

The AMD FP8 GEMM challenge asked the competitors to calculate row-wise block scaling for multiplying matrix A (MxK column major) and matrix B (NxK column major). The result can be intepreted as C = A@B.T where A is column major, B.T is row major and C (MxN) is row major.

MI300X provides 2 matrix core (AMD's tensor core) instructions to help calculate small matrix multiplication:
```
32x32x16:
        FLOPs: 32768
        Execution cycles: 32
        FLOPs/CU/cycle: 4096
16x16x32:
        FLOPs: 16384
        Execution cycles: 16
        FLOPs/CU/cycle: 4096
```
FLOPS utilization calculation for MI300X:
- FP8 peak FLOPS calculation = Peak_Engine_Clock * num_CU * FLOPs/CU/cycle = 2100e6 * 304 * 4096 = 2614.9 TFLOPS
- FP8 achieved FLOPS on shape 7168x6144x4608 = FLOPS / Latency = 7168 * 6144 * 4608 * 2 / 948e-6 = 428.1 TFLOPS
- FLOPS utilization = Achieved / Peak = 16.37 %
- Theoretical speed-of-light latency for shape 7168x6144x4608 = 7168 * 6144 * 4608 * 2 / 2614.9e12 = 155.2 us

<div align="center">
  <table>
    <tr>
      <td align="center">
        <h5>32x32x16 matC thread-register mapping</h3>
        <img src="images\thread-register-mapping-matC-32x32x16.png" alt="32x32x16 VGPR Layout" width="600"/>
      </td>
      <td align="center">
        <h5>16x16x32 matC thread-register mapping</h3>
        <img src="images\thread-register-mapping-matC-16x16x32.png" alt="16x16x32 VGPR Layout" width="600"/>
      </td>
    </tr>
  </table>
</div>

### Overall
For my solution, I used 32x32x16 instruction to calculate GEMM on problems with large M, N due to better arithmetic intensity (1 warp of 64 threads can calculate 32x32 elements in matC). And used 16x16x32 instruction for problems where M, N are small but K is large. rocwmma api is used to perform matrix core operations since I found that it led to less register spills.

### Maximize hardware utilization
MI300X has a cache line of 128 bytes, which matched the size of block scaling, it's beneficial that 1 warp loads data from the same cache line. Data will be loaded from global memory in chunk of 128 bytes at a time, retrieve data found in the cache line will result in a cache hit, where it only cost us the memory access latency of L2 or L1 cache instead of the slow global memory. 
MI300X also provided us with 64 KB of shared memory (LDS) with the same speed as L1 cache, for which all threads within a block can share and access at a lower latency compared to L2 cache. We can use this for data reuse in block partitioned matrix multiplication.
Each Compute unit in MI300X can handle 2048 threads, and each block has max thread of 1024. I will launch 2 blocks on each compute units so each block can have 32 KB of LDS to load larger tiles.
Loop unrolling is essential since it helps the compiler to see the future and plans loading of a chunk of data and share it to each iteration or calculate offset before hand. One should make fine tune `#pragma unroll` to unroll fully or not unrolling at all `#pragma unroll 1` to keep 64 VGPRs (vector registers) per thread for maximum theoretical occupancy per Compute Unit, maximum occupancy can help hides latency by switching to run other threads while the current threads are waiting for data to arrive. For viewing register pressure, there is a very good blog post for that https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-register-pressure-readme/ . TLDR, you can generate generated assembly and view VGPRs, SGPRs usage and occupancy using:
```
hipcc -o sgemm sgemm.cu --offload-arch=gfx942 -std=c++20 -Rpass-analysis=kernel-resource-usage --save-temps
```

### Kernel for large M, N
For each block, I used 3 types of shared memory:
```cuda
__shared__ __hip_fp8_e4m3_fnuz tileA[128][72]; // row-major
__shared__ __hip_fp8_e4m3_fnuz tileB[128][72]; // col-major, = row-major transposed
__shared__ float tileAs[128];
```
Here, this kernel used 4x4 warps (4x4x64 = 1024 threads) and 32x32x16 matrix multiplication accumulation (mfma) instruction to calculate an output block of 128x128 elements. Each warp will hold a fraction of the matrix elements:
frag_c += frag_a x frag_b
32x32  += 32x16 x 16x32

frag_a and b need to store 32x16 = 512 elements, divided by 64 threads in a warp, therefore, each thread will hold values of 8 fp8 elements = 64 bit or 2 32-bit VGPRs (vector register).

Looking at the mapping between the thread and the register according to https://github.com/ROCm/amd_matrix_instruction_calculator or an example at https://github.com/amd/amd-lab-notes/blob/release/matrix-cores/src/mfma_fp32_32x32x8fp16.cpp . For frag_a (32x16) the first 32 lanes (threads) in a warp will hold 8 consecutive columns of each row in the first 32 rows in matrix A. And the last 32 lanes will hold the next 8 consecutive columns of the same first 32 rows in matrix A. These hint us that we need to have these 8 consecutive element per row to be contiguous in memory, so that loading them to the 2 VGPRs can be fast. For example, if the LDS for tileA is column major, we would need 8 loads of 8 bits (ds_read_u8) to fill 2 VGPRs per lane, but if we have tileA in row major, we can read all 8 fp8 elements using a single ds_read_b64 instruction which is 8 times faster. Therefore, we will load column-major matrix A into a row-major-tileA stored in shared memory with transpose loading.

The following code load use 64 threads in a warp to load 128 contiguous elements (coalesced read to global memory) in a column of matrix A (col-maj) to 128 strided elements (uncoalesced store to shared memory) in a column in tileA (row-maj):

```cuda
tileA[t64x][kmini_load + t64y] = A_ptr[blockIdx.y * WMMA_M * NUM_WARP_Y + t64x + M * (kbase + kmini_load + t64y)];
tileA[t64x + 64][kmini_load + t64y] = A_ptr[blockIdx.y * WMMA_M * NUM_WARP_Y + t64x + 64 + M * (kbase + kmini_load + t64y)];
```
where:

```
   • kbase and kmini_load: offset along the K-dimension
      • kbase jumps 128 elements for each iteration
      • kmini_load jumps 16 elements for each iteration
   • t64x: thread idx in a warp
   • t64y: warp idx in a thread block
```

TileAs (row wise block scaling factor) is also loaded in the same manner.

Padding of 8 elements from 64 to 72 for the K-dimension to avoid bank conflict. Why not padding of 1 or 4 ? A bank if 32bit, each fp8 element is 1/4 of a bank, so if we padd by 1 fp8 elements, 4 consecutive rows will be in the same bank. Padding of 4 seems good in theory, but upon inspection with amd matrix instruction calculator, GPR alignment requirement: 8 bytes, I think this is the reason behind the performance degradation when padding with 4. As a result, my code still suffers from a 2-way bank conflict. Zero bank conflict can be achieved by loading to shared memory in the exact layout required by the fragments, this approach gained small improvement on the largest shape.

To calculate a full 128x128 block , we need 2 load loops (each load 128x64 elements per matrix) and 2 mfma loops and 1 loop to apply scalers. I found that this is faster than loading a full 128x128 block where we have to wait until 128x128 elements are fully loaded before we can do matmul. Upon inspecting the compiled code, we can see that the loading of scaling factor from SMEM to registers are interleaved with the second mfma which helped hide latency.

After the second matmul, we can apply scaler. Loading the scaler in shared memory provides the benefit of the compiler generating ds_read_b128 instruction with the help of `#pragma unroll`, to fetch data to calculate shared memory on the accumulation fragments which is very fast:
```cuda
#pragma unroll
for (int i = 0; i < 4; ++i){
    #pragma unroll
    for (int j = 0; j < 4; ++j){
        frag_acc_master.x[i * 4 + j] += tileAs[warp_idx_y * WMMA_M + 4 * (t64x / 32) + 8 * i + j] * frag_acc_block.x[i * 4 + j];
    }
}
```
Here, 2 accumulation fragments are used, one for the global accumulation (master), one to store intermediate accumulation per block scaling. Analyzing the 2 links provided above, one can peer into the register-thread mapping to produce the indexing.

### Kernel for small M, N with large K

For these problems, I used 16x16x32 mfma instruction, each block has 16 warps (1024 threads), to calculate an output tile of 64x64 elements in matrix C. Here, we can split the matmul into more blocks to better saturate the 304 CUs of MI300X. For problem such as: K = 7168, M = 1024, N = 512. If we use 32x32x16 to calculate 128x128, we only launch 1024x512/(128x128) = 32 blocks, each CU manages 2 blocks, that equals to using only 16/304 CUs, very low achieved occupancy. But if we use 16x16x32 to calculate 64x64 elements per block, we get 64/304 CUs, that's 4 times the achieved occupancy. For larger M, N since we launch multiple full waves (different from wave front), the poor achieved occupancy of the last wave is neligible.
For this kernel, I perform C.T = B @ A.T to load tileAs into register in a coalesced and straight forward manner. Each block used a full 32 KB LDS:
```cuda
__shared__ __hip_fp8_e4m3_fnuz tileA[64][256]; // col-maj = row-maj transposed 
__shared__ __hip_fp8_e4m3_fnuz tileB[64][256]; // row-maj
```
To load these without bank conflict, rotating is used, where the next row is shifted by 8 elements, elements that overshoot the dimension limit will be wrapped around using modulo "% 128" for 128 elements. Modulo " % 16" is used to reduce computation since the pattern will be repeated after 16 rows. I found that rotating reduced VGPR pressure compared to XOR swizzling, I think that is because this pattern is more predictable for the compiler.
```cuda
#pragma unroll
for (int kmini = 0; kmini < 128; kmini += 16){ // 16 is num group
    int K_idx_shifty = (kmini + t64y + 8 * (t64x % 16)) % 128;            
    tileA[t64x][K_idx_shifty] = A_ptr[blockIdx.x * WMMA_M * NUM_WARP_X + t64x + M * (kbase + kmini + t64y)];       
    tileA[t64x][128 + K_idx_shifty] = A_ptr[blockIdx.x * WMMA_M * NUM_WARP_X + t64x + M * (128 + kbase + kmini + t64y)];  
}
```
Next, calculate the rotated index to load the correct elements to the 2 VGPRs per lane. memcpy will copy a sizeof(long) = 8 bytes data straight from the address at first element of a frag in SMEM to the address at first element of the 2 VGPRs of frag_a and frag_b (using &frag_a.x[0]).
```cuda
#pragma unroll
for (int kmini = 0; kmini < 128; kmini += WMMA_K){
    int tileB_row = warp_idx_y * WMMA_M + t64x % 16;
    int K_idx_base = kmini + (t64x / 16) * 8;
    int K_idx_shiftyB = (K_idx_base + 8 * tileB_row) % 128;
    memcpy(&frag_a.x[0], &tileB[tileB_row][128 + K_idx_shiftyB], sizeof(long));

    int tileA_col = warp_idx_x * WMMA_N + t64x % 16;
    memcpy(&frag_b.x[0], &tileA[tileA_col][128 + K_idx_shiftyB], sizeof(long));

    rocwmma::mma_sync(frag_acc_block2, frag_a, frag_b, frag_acc_block2);
}
```

Here, we load 2 blocks of 128 K-elements, therefore we need to also load 2 scaling factors. The below scaling factors are loaded directly into register without using SMEM since we have used up all available SMEM for tileA and tileB.
```cuda
float tileAs1 = As_ptr[(kbase / 128) * M + col_base_C + t64x % WMMA_N];
float tileBs1 = Bs_ptr[(kbase / 128) * sn + (blockIdx.y * NUM_WARP_Y * WMMA_M) / 128];
float scale1 = tileAs1 * tileBs1;
float tileAs2 = As_ptr[(1 + kbase / 128) * M + col_base_C + t64x % WMMA_N];
float tileBs2 = Bs_ptr[(1 + kbase / 128) * sn + (blockIdx.y * NUM_WARP_Y * WMMA_M) / 128];
float scale2 = tileAs2 * tileBs2;

#pragma unroll
for (int i = 0; i < frag_acc_master.num_elements; ++i){
    frag_acc_master.x[i] = fmaf(scale1, frag_acc_block.x[i], frag_acc_master.x[i]);            
}
```

I found that this kernel executes in lock steps very well with small M, N and the first 128 columns are consumed first by mfma, then it can be reloaded. Hence, it is like a form of double buffering and we can remove a few syncthreads to let data loading and computation overlap. With 32x32x16, it uses 16 VGPRs for frag accum master and another 16 VGPRs for frag acc block, so there aren't enough VGPRs to take advantage of double buffering. But for 16x16x32, only 4 VGPRs per lane are required, so there are excess registers to take advantage of double buffering.

Cheers !
