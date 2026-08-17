#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
#include <cmath>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

#define CU_CHECK(call) do {                                      \
    CUresult _e = (call);                                        \
    if (_e != CUDA_SUCCESS) {                                    \
        const char* err_name = "Unknown";                         \
        cuGetErrorName(_e, &err_name);                           \
        fprintf(stderr, "CUresult error %s at %s:%d\n",           \
                err_name, __FILE__, __LINE__);                   \
        exit(1);                                                  \
    }                                                            \
} while(0)

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void expect_tx(uint64_t* bar, uint32_t bytes, uint32_t& phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %2;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(bytes), "r"(phase));
    
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(bytes) : "memory");
    
    phase ^= 1;
}

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase));
}

__device__ __forceinline__ void tma_load_2d_swizzled(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
}

template<int RowMajorA, int RowMajorB, int Accumulate>
__device__ __forceinline__ void gemm_128x128(
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>& acc,
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 128, __nv_bfloat16, RowMajorA>& a0,
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 128, __nv_bfloat16, RowMajorB>& b0,
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 128, __nv_bfloat16, RowMajorA>& a1,
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 128, __nv_bfloat16, RowMajorB>& b1)
{
    if constexpr (Accumulate == 0) {
        nvcuda::wmma::fill_fragment(acc, 0.0f);
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        nvcuda::wmma::mma_sync(acc, a0, b0);
        nvcuda::wmma::mma_sync(acc, a1, b1);
    }
}

template<bool IsTransposed>
__device__ __forceinline__ void load_swizzled(
    const char* name,
    const __nv_bfloat16* smem,
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 128, __nv_bfloat16, 0>& frag,
    int row_start, int col_start) 
{
    int warp_row_base = (threadIdx.x / 32) * 2;
    int warp_col_base = (threadIdx.x % 32) / 4;
    
    int s_row = row_start + warp_row_base * 16;
    int s_col = col_start + (IsTransposed ? warp_row_base * 16 : warp_col_base * 16);
    
    nvcuda::wmma::load_matrix_sync(frag, smem + s_row * 64 + s_col, 64);
}

template<bool IsTransposed>
__device__ __forceinline__ void load_transposed_swizzled(
    const char* name, 
    const __nv_bfloat16* smem_A, __nv_bfloat16* smem_B_T,
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 128, __nv_bfloat16, 0>& frag,
    int row_start, int col_start) 
{
    transpose_smem_128x64_swizzled(smem_A, smem_B_T);
    load_swizzled<false>(name, smem_B_T, frag, row_start, col_start);
}

template<bool IsTransposed>
__device__ __forceinline__ void store_swizzled(
    const char* name, 
    __nv_bfloat16* smem, 
    int i_tile, int j_tile, 
    const float (&local_array)[16][16]) 
{
    int warp_row_base = (threadIdx.x / 32) * 2;
    int warp_col_base = (threadIdx.x % 32) / 4;
    
    int s_row = (IsTransposed ? j_tile * 16 : i_tile * 16) + (threadIdx.x % 16);
    int s_col = (IsTransposed ? i_tile * 16 : j_tile * 16) + (threadIdx.x % 16 == threadIdx.x % 32 ? 0 : 16); 
    
    int swizzled_chunk = (s_row % 8) ^ (s_col / 8);
    int swizzled_col = swizzled_chunk * 8 + (s_col % 8);
    
    __nv_bfloat16* smem_ptr = smem + s_row * 64 + swizzled_col;
    *reinterpret_cast<float4*>(smem_ptr) = *reinterpret_cast<const float4*>(&local_array[(threadIdx.x % 32) / 4][s_col % 16]);
}

__device__ __forceinline__ void write_swizzled_bf16(__nv_bfloat16* smem, int row, int col, __nv_bfloat16 val) {
    int col_chunk = col / 4;
    int chunk_offset = col % 4;
    int swizzled_col_chunk = (row % 8) ^ col_chunk;
    int swizzled_col = swizzled_col_chunk * 4 + chunk_offset;
    smem[row * 64 + swizzled_col] = val;
}

__device__ __forceinline__ __nv_bfloat16 read_swizzled_bf16(__nv_bfloat16* smem, int row, int col) {
    int col_chunk = col / 4;
    int chunk_offset = col % 4;
    int swizzled_col_chunk = (row % 8) ^ col_chunk;
    int swizzled_col = swizzled_col_chunk * 4 + chunk_offset;
    return smem[row * 64 + swizzled_col];
}

__device__ __forceinline__ void transpose_smem_128x64_swizzled(__nv_bfloat16* smem_A, __nv_bfloat16* smem_B_T) {
    for (int i = threadIdx.x; i < 128 * 64; i += blockDim.x) {
        int r = i / 64;
        int c = i % 64;
        __nv_bfloat16 val = read_swizzled_bf16(smem_A, r, c);
        write_swizzled_bf16(smem_B_T, c, r, val);
    }
    __syncthreads();
}

extern __shared__ __align__(128) uint8_t smem_pool[];

__device__ void setup_smem_ptrs(
    __nv_bfloat16*& smem_Q0, __nv_bfloat16*& smem_Q1,
    __nv_bfloat16*& smem_K0, __nv_bfloat16*& smem_K1,
    __nv_bfloat16*& smem_V0, __nv_bfloat16*& smem_V1,
    __nv_bfloat16*& smem_O0, __nv_bfloat16*& smem_O1,
    __nv_bfloat16*& smem_dO0, __nv_bfloat16*& smem_dO1,
    __nv_bfloat16*& smem_dQ0, __nv_bfloat16*& smem_dQ1,
    __nv_bfloat16*& smem_dK0, __nv_bfloat16*& smem_dK1,
    __nv_bfloat16*& smem_dV0, __nv_bfloat16*& smem_dV1,
    __nv_bfloat16*& smem_D0, __nv_bfloat16*& smem_D1,
    __nv_bfloat16*& smem_S0, __nv_bfloat16*& smem_S1,
    __nv_bfloat16*& smem_P0, __nv_bfloat16*& smem_P1,
    __nv_bfloat16*& smem_PT0, __nv_bfloat16*& smem_PT1,
    __nv_bfloat16*& smem_dPT0, __nv_bfloat16*& smem_dPT1,
    float*& smem_L, float*& smem_Lambda,
    uint64_t*& bar_Q, uint64_t*& bar_K,
    uint64_t*& bar_V, uint64_t*& bar_O, uint64_t*& bar_dO) 
{
    uint8_t* p = smem_pool;
    
    smem_Q0  = reinterpret_cast<__nv_bfloat16*>(p); p += 16384;
    smem_Q1  = reinterpret_cast<__nv_bfloat16*>(p); p += 16384;
    smem_K0  = reinterpret_cast<__nv_bfloat16*>(p); p += 16384;
    smem_K1  = reinterpret_cast<__nv_bfloat16*>(p); p += 16384;
    smem_V0  = reinterpret_cast<__nv_bfloat16*>(p); p += 16384;
    smem_V1  = reinterpret_cast<__nv_bfloat16*>(p); p += 16384;
    smem_O0  = reinterpret_cast<__nv_bfloat16*>(p); p += 16384;
    smem_O1  = reinterpret_cast<__nv_bfloat16*>(p); p += 16384;
    smem_dO0 = reinterpret_cast<__nv_bfloat16*>(p); p += 16384;
    smem_dO1 = reinterpret_cast<__nv_bfloat16*>(p); p += 16384;
    smem_dQ0 = reinterpret_cast<__nv_bfloat16*>(p); p += 16384;
    smem_dQ1 = reinterpret_cast<__nv_bfloat16*>(p); p += 16384;
    smem_dK0 = reinterpret_cast<__nv_bfloat16*>(p); p += 16384;
    smem_dK1 = reinterpret_cast<__nv_bfloat16*>(p); p += 16384;
    smem_dV0 = reinterpret_cast<__nv_bfloat16*>(p); p += 16384;
    smem_dV1 = reinterpret_cast<__nv_bfloat16*>(p); p += 16384;
    smem_D0  = reinterpret_cast<__nv_bfloat16*>(p); p += 16384;
    smem_D1  = reinterpret_cast<__nv_bfloat16*>(p); p += 16384;
    smem_S0  = reinterpret_cast<__nv_bfloat16*>(p); p += 16384;
    smem_S1  = reinterpret_cast<__nv_bfloat16*>(p); p += 16384;
    smem_P0  = reinterpret_cast<__nv_bfloat16*>(p); p += 16384;
    smem_P1  = reinterpret_cast<__nv_bfloat16*>(p); p += 16384;
    smem_PT0 = reinterpret_cast<__nv_bfloat16*>(p); p += 16384;
    smem_PT1 = reinterpret_cast<__nv_bfloat16*>(p); p += 16384;
    smem_dPT0= reinterpret_cast<__nv_bfloat16*>(p); p += 16384;
    smem_dPT1= reinterpret_cast<__nv_bfloat16*>(p); p += 16384;
    
    p = (uint8_t*)(((uintptr_t)p + 255) & ~255);
    
    smem_L = reinterpret_cast<float*>(p); p += 512;
    smem_Lambda = reinterpret_cast<float*>(p); p += 512;
    
    p = (uint8_t*)(((uintptr_t)p + 255) & ~255);
    
    bar_Q = reinterpret_cast<uint64_t*>(p); p += 8;
    bar_K = reinterpret_cast<uint64_t*>(p); p += 8;
    bar_V = reinterpret_cast<uint64_t*>(p); p += 8;
    bar_O = reinterpret_cast<uint64_t*>(p); p += 8;
    bar_dO= reinterpret_cast<uint64_t*>(p); p += 8;
}

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2, 
        globalAddress,
        globalDim,
        globalStrides,
        boxDim, 
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        l2Promotion,
        oobFill
    );
}

__global__ void attn_backward_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* L_ptr,
    __nv_bfloat16* dQ_ptr,
    __nv_bfloat16* dK_ptr,
    __nv_bfloat16* dV_ptr,
    uint32_t S, float scale) 
{
    uint32_t q_tile = blockIdx.x;
    uint32_t b_h_idx = blockIdx.y;
    uint32_t q_start = q_tile * 128;
    uint32_t phase = 0;
    int tid = threadIdx.x;

    if (q_start >= S) return;

    __nv_bfloat16 *smem_Q0, *smem_Q1, *smem_K0, *smem_K1, *smem_V0, *smem_V1, *smem_O0, *smem_O1, *smem_dO0, *smem_dO1;
    __nv_bfloat16 *smem_dQ0, *smem_dQ1, *smem_dK0, *smem_dK1, *smem_dV0, *smem_dV1;
    __nv_bfloat16 *smem_D0, *smem_D1, *smem_S0, *smem_S1, *smem_P0, *smem_P1, *smem_PT0, *smem_PT1, *smem_dPT0, *smem_dPT1;
    float *smem_L, *smem_Lambda;
    uint64_t *bar_Q, *bar_K, *bar_V, *bar_O, *bar_dO;

    setup_smem_ptrs(smem_Q0, smem_Q1, smem_K0, smem_K1, smem_V0, smem_V1, smem_O0, smem_O1, smem_dO0, smem_dO1,
                    smem_dQ0, smem_dQ1, smem_dK0, smem_dK1, smem_dV0, smem_dV1,
                    smem_D0, smem_D1, smem_S0, smem_S1, smem_P0, smem_P1, smem_PT0, smem_PT1, smem_dPT0, smem_dPT1,
                    smem_L, smem_Lambda, bar_Q, bar_K, bar_V, bar_O, bar_dO);

    if (tid == 0) {
        init_smem_barrier_fn(bar_Q, 1);
        init_smem_barrier_fn(bar_K, 1);
        init_smem_barrier_fn(bar_V, 1);
        init_smem_barrier_fn(bar_O, 1);
        init_smem_barrier_fn(bar_dO, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    // === Pass 1: Compute dQ ===
    uint32_t total_tiles = (S + 127) / 128;
    uint32_t bh_offset = b_h_idx * S;

    if (tid == 0) {
        expect_tx(bar_Q, 32768, phase);
        tma_load_2d_swizzled(&tma_Q, bar_Q, smem_Q0, 0, bh_offset + q_start);
        tma_load_2d_swizzled(&tma_Q, bar_Q, smem_Q1, 64, bh_offset + q_start);
    }
    mbarrier_wait_fn(bar_Q, phase);
    if (tid < 128) {
        smem_L[tid] = (q_start + tid < S) ? L_ptr[b_h_idx * S + q_start + tid] : 0.0f;
    }
    __syncthreads();

    // Zero dQ shared memory
    for (int i = tid; i < 128 * 64; i += blockDim.x) {
        int row = i / 64;
        int col = i % 64;
        write_swizzled_bf16(smem_dQ0, row, col, __float2bfloat16(0.0f));
        write_swizzled_bf16(smem_dQ1, row, col, __float2bfloat16(0.0f));
    }
    __syncthreads();

    for (int kv_tile = 0; kv_tile < total_tiles; kv_tile++) {
        uint32_t kv_start = kv_tile * 128;
        
        if (tid == 0) {
            expect_tx(bar_K, 32768, phase);
            tma_load_2d_swizzled(&tma_K, bar_K, smem_K0, 0, bh_offset + kv_start);
            tma_load_2d_swizzled(&tma_K, bar_K, smem_K1, 64, bh_offset + kv_start);

            expect_tx(bar_V, 32768, phase);
            tma_load_2d_swizzled(&tma_V, bar_V, smem_V0, 0, bh_offset + kv_start);
            tma_load_2d_swizzled(&tma_V, bar_V, smem_V1, 64, bh_offset + kv_start);

            expect_tx(bar_O, 32768, phase);
            tma_load_2d_swizzled(&tma_O, bar_O, smem_O0, 0, bh_offset + kv_start);
            tma_load_2d_swizzled(&tma_O, bar_O, smem_O1, 64, bh_offset + kv_start);

            expect_tx(bar_dO, 32768, phase);
            tma_load_2d_swizzled(&tma_dO, bar_dO, smem_dO0, 0, bh_offset + kv_start);
            tma_load_2d_swizzled(&tma_dO, bar_dO, smem_dO1, 64, bh_offset + kv_start);
        }
        mbarrier_wait_fn(bar_K, phase);
        mbarrier_wait_fn(bar_V, phase);
        mbarrier_wait_fn(bar_O, phase);
        mbarrier_wait_fn(bar_dO, phase);

        // Compute row sums of O * dO -> Lambda
        __syncthreads();
        float sum = 0;
        for(int c = 0; c < 64; c += 2) {
            float o0 = __bfloat162float(read_swizzled_bf16(smem_O0, tid, c));
            float do_val0 = __bfloat162float(read_swizzled_bf16(smem_dO0, tid, c));
            sum += o0 * do_val0;
            
            float o1 = __bfloat162float(read_swizzled_bf16(smem_O1, tid, c));
            float do_val1 = __bfloat162float(read_swizzled_bf16(smem_dO1, tid, c));
            sum += o1 * do_val1;
        }
        smem_Lambda[tid] = sum;
        __syncthreads();

        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> S_fragments[64];
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> D_fragments[64];
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> dP_fragments[64];
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> dQ_fragments[64];

        float S_local[4][16][16];
        float D_local[4][16][16];
        float dP_local[4][16][16];
        float dQ_local[4][16][16];

        int frag_idx_base = (tid / 4) * 8 + (tid % 4) * 2;
        for (int k = 0; k < 4; ++k) {
            int frag_idx = frag_idx_base + k;
            int i = frag_idx / 8;
            int j = frag_idx % 8;
            
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 128, __nv_bfloat16, 0> Q0_a, Q1_a;
            load_swizzled<false>("Q0", smem_Q0, Q0_a, 0, 0);
            load_swizzled<false>("Q1", smem_Q1, Q1_a, 0, 0);

            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 128, __nv_bfloat16, 1> K_T0_b, K_T1_b;
            load_transposed_swizzled<true>("K0", smem_K0, smem_K_T0, K_T0_b, 0, 0);
            load_transposed_swizzled<true>("K1", smem_K1, smem_K_T1, K_T1_b, 0, 0);
            
            gemm_128x128<0, 1, (kv_tile == 0) ? 0 : 1>(S_fragments[frag_idx], Q0_a, K_T0_b, Q1_a, K_T1_b);
            nvcuda::wmma::store_matrix_sync(&S_local[k][0][0], S_fragments[frag_idx], 16, nvcuda::wmma::mem_row_major);
        }

        __syncthreads(); 
        for(int j = 0; j < 4; j++) {
            for(int r = 0; r < 16; r++) {
                for(int c = 0; c < 16; c+=2) {
                    float s0 = S_local[j][r][c];
                    float p0 = expf(s0 * scale - smem_L[(tid/32)*32 + j*16 + r]);
                    
                    float s1 = S_local[j][r][c+1];
                    float p1 = expf(s1 * scale - smem_L[(tid/32)*32 + j*16 + r]);
                    
                    write_swizzled_bf16(smem_P0, (tid/32)*32 + j*16 + r, j*16 + c, __float2bfloat16(p0));
                    write_swizzled_bf16(smem_P1, (tid/32)*32 + j*16 + r, j*16 + c+1, __float2bfloat16(p1));
                }
            }
        }
        __syncthreads();

        // Compute D = dO - P * Lambda
        for(int j = 0; j < 4; j++) {
            for(int r = 0; r < 16; r++) {
                int row = (tid/32)*32 + j*16 + r;
                float lambda = smem_Lambda[row];
                
                for(int c = 0; c < 16; c+=2) {
                    float p0 = __bfloat162float(read_swizzled_bf16(smem_P0, tid, c));
                    float do_val0 = __bfloat162float(read_swizzled_bf16(smem_dO0, tid, c));
                    write_swizzled_bf16(smem_D0, tid, c, __float2bfloat16(do_val0 - p0 * lambda));
                    
                    float p1 = __bfloat162float(read_swizzled_bf16(smem_P1, tid, c+1));
                    float do_val1 = __bfloat162float(read_swizzled_bf16(smem_dO1, tid, c+1));
                    write_swizzled_bf16(smem_D1, tid, c+1, __float2bfloat16(do_val1 - p1 * lambda));
                }
            }
        }
        __syncthreads();

        for (int k = 0; k < 4; ++k) {
            int frag_idx = frag_idx_base + k;
            int i = frag_idx / 8;
            int j = frag_idx % 8;
            
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 128, __nv_bfloat16, 0> D0_a, D1_a;
            load_swizzled<false>("D0", smem_D0, D0_a, 0, 0);
            load_swizzled<false>("D1", smem_D1, D1_a, 0, 0);

            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 128, __nv_bfloat16, 1> V_T0_b, V_T1_b;
            load_transposed_swizzled<true>("V0", smem_V0, smem_V_T0, V_T0_b, 0, 0);
            load_transposed_swizzled<true>("V1", smem_V1, smem_V_T1, V_T1_b, 0, 0);
            
            gemm_128x128<0, 1, 1>(dP_fragments[frag_idx], D0_a, V_T0_b, D1_a, V_T1_b);
            nvcuda::wmma::store_matrix_sync(&dP_local[k][0][0], dP_fragments[frag_idx], 16, nvcuda::wmma::mem_row_major);
        }

        __syncthreads(); 
        for(int k = 0; k < 4; k++) {
            int frag_idx = frag_idx_base + k;
            int i = frag_idx / 8;
            int j = frag_idx % 8;
            store_swizzled<false>("dP0", smem_dP0, i, j, dP_local[k]);
            store_swizzled<false>("dP1", smem_dP1, i, j, dP_local[k]);
        }
        __syncthreads();

        for (int k = 0; k < 4; ++k) {
            int frag_idx = frag_idx_base + k;
            int i = frag_idx / 8;
            int j = frag_idx % 8;
            
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 128, __nv_bfloat16, 0> dP0_a, dP1_a;
            load_swizzled<false>("dP0", smem_dP0, dP0_a, 0, 0);
            load_swizzled<false>("dP1", smem_dP1, dP1_a, 0, 0);

            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 128, __nv_bfloat16, 0> K0_b, K1_b;
            load_swizzled<false>("K0", smem_K0, K0_b, 0, 0);
            load_swizzled<false>("K1", smem_K1, K1_b, 0, 0);
            
            gemm_128x128<0, 0, 1>(dQ_fragments[frag_idx], dP0_a, K0_b, dP1_a, K1_b);
            nvcuda::wmma::store_matrix_sync(&dQ_local[k][0][0], dQ_fragments[frag_idx], 16, nvcuda::wmma::mem_row_major);
        }

        __syncthreads(); 
        for(int k = 0; k < 4; k++) {
            int frag_idx = frag_idx_base + k;
            int i = frag_idx / 8;
            int j = frag_idx % 8;
            for(int r = 0; r < 16; r++) {
                for(int c = 0; c < 16; c+=2) {
                    write_swizzled_bf16(smem_dQ0, i*16 + r, j*16 + c, __float2bfloat16(dQ_local[k][r][c]));
                    write_swizzled_bf16(smem_dQ1, i*16 + r, j*16 + c, __float2bfloat16(dQ_local[k][r][c+1]));
                }
            }
        }
        __syncthreads();
    }

    // Epilogue dQ
    for(int i = tid; i < 128 * 64 / 4; i += blockDim.x) {
        int row = (i * 4) / 64;
        int col = (i * 4) % 64;
        if (q_start + row < S) {
            float4 f0 = *reinterpret_cast<float4*>(&smem_dQ0[row * 64 + col]);
            float4 f1 = *reinterpret_cast<float4*>(&smem_dQ1[row * 64 + col]);
            __nv_bfloat16 out[8];
            out[0] = __float2bfloat16(f0.x); out[1] = __float2bfloat16(f0.y); out[2] = __float2bfloat16(f0.z); out[3] = __float2bfloat16(f0.w);
            out[4] = __float2bfloat16(f1.x); out[5] = __float2bfloat16(f1.y); out[6] = __float2bfloat16(f1.z); out[7] = __float2bfloat16(f1.w);
            *reinterpret_cast<float4*>(&dQ_ptr[(b_h_idx * S + q_start + row) * 128 + col]) = *reinterpret_cast<float4*>(out);
            *reinterpret_cast<float4*>(&dQ_ptr[(b_h_idx * S + q_start + row) * 128 + col + 64]) = *reinterpret_cast<float4*>(&out[4]);
        }
    }
    __syncthreads();

    // === Pass 2: Compute dK and dV ===
    uint32_t kv_start = q_start; 
    
    if (tid == 0) {
        expect_tx(bar_K, 32768, phase);
        tma_load_2d_swizzled(&tma_K, bar_K, smem_K0, 0, bh_offset + kv_start);
        tma_load_2d_swizzled(&tma_K, bar_K, smem_K1, 64, bh_offset + kv_start);
        
        expect_tx(bar_V, 32768, phase);
        tma_load_2d_swizzled(&tma_V, bar_V, smem_V0, 0, bh_offset + kv_start);
        tma_load_2d_swizzled(&tma_V, bar_V, smem_V1, 64, bh_offset + kv_start);
    }
    mbarrier_wait_fn(bar_K, phase);
    mbarrier_wait_fn(bar_V, phase);
    if (tid < 128) {
        smem_L[tid] = (kv_start + tid < S) ? L_ptr[b_h_idx * S + kv_start + tid] : 0.0f;
    }
    __syncthreads();

    // Zero dK and dV shared memory
    for (int i = tid; i < 128 * 64; i += blockDim.x) {
        int row = i / 64;
        int col = i % 64;
        write_swizzled_bf16(smem_dK0, row, col, __float2bfloat16(0.0f));
        write_swizzled_bf16(smem_dK1, row, col, __float2bfloat16(0.0f));
        write_swizzled_bf16(smem_dV0, row, col, __float2bfloat16(0.0f));
        write_swizzled_bf16(smem_dV1, row, col, __float2bfloat16(0.0f));
    }
    __syncthreads();

    for (int q_tile_p2 = 0; q_tile_p2 < total_tiles; q_tile_p2++) {
        uint32_t qs = q_tile_p2 * 128;
        
        if (tid == 0) {
            expect_tx(bar_Q, 32768, phase);
            tma_load_2d_swizzled(&tma_Q, bar_Q, smem_Q0, 0, bh_offset + qs);
            tma_load_2d_swizzled(&tma_Q, bar_Q, smem_Q1, 64, bh_offset + qs);

            expect_tx(bar_O, 32768, phase);
            tma_load_2d_swizzled(&tma_O, bar_O, smem_O0, 0, bh_offset + qs);
            tma_load_2d_swizzled(&tma_O, bar_O, smem_O1, 64, bh_offset + qs);

            expect_tx(bar_dO, 32768, phase);
            tma_load_2d_swizzled(&tma_dO, bar_dO, smem_dO0, 0, bh_offset + qs);
            tma_load_2d_swizzled(&tma_dO, bar_dO, smem_dO1, 64, bh_offset + qs);
        }
        mbarrier_wait_fn(bar_Q, phase);
        mbarrier_wait_fn(bar_O, phase);
        mbarrier_wait_fn(bar_dO, phase);

        __syncthreads();
        float sum = 0;
        for(int c = 0; c < 64; c += 2) {
            float o0 = __bfloat162float(read_swizzled_bf16(smem_O0, tid, c));
            float do_val0 = __bfloat162float(read_swizzled_bf16(smem_dO0, tid, c));
            sum += o0 * do_val0;
            
            float o1 = __bfloat162float(read_swizzled_bf16(smem_O1, tid, c));
            float do_val1 = __bfloat162float(read_swizzled_bf16(smem_dO1, tid, c));
            sum += o1 * do_val1;
        }
        smem_Lambda[tid] = sum;
        __syncthreads();

        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> S_fragments[64];
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> D_fragments[64];
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> dP_fragments[64];
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> dK_fragments[64];
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> dV_fragments[64];

        float S_local[4][16][16];
        float D_local[4][16][16];
        float dP_local[4][16][16];
        float dK_local[4][16][16];
        float dV_local[4][16][16];

        int frag_idx_base = (tid / 4) * 8 + (tid % 4) * 2;
        for (int k = 0; k < 4; ++k) {
            int frag_idx = frag_idx_base + k;
            int i = frag_idx / 8;
            int j = frag_idx % 8;
            
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 128, __nv_bfloat16, 0> K0_a, K1_a;
            load_swizzled<false>("K0", smem_K0, K0_a, 0, 0);
            load_swizzled<false>("K1", smem_K1, K1_a, 0, 0);

            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 128, __nv_bfloat16, 1> Q_T0_b, Q_T1_b;
            load_transposed_swizzled<true>("Q0", smem_Q0, smem_Q_T0, Q_T0_b, 0, 0);
            load_transposed_swizzled<true>("Q1", smem_Q1, smem_Q_T1, Q_T1_b, 0, 0);
            
            gemm_128x128<0, 1, (q_tile_p2 == 0) ? 0 : 1>(S_fragments[frag_idx], K0_a, Q_T0_b, K1_a, Q_T1_b);
            nvcuda::wmma::store_matrix_sync(&S_local[k][0][0], S_fragments[frag_idx], 16, nvcuda::wmma::mem_row_major);
        }

        __syncthreads(); 
        for(int j = 0; j < 4; j++) {
            for(int r = 0; r < 16; r++) {
                for(int c = 0; c < 16; c+=2) {
                    float s0 = S_local[j][r][c];
                    float p0 = expf(s0 * scale - smem_L[(tid/32)*32 + j*16 + r]);
                    
                    float s1 = S_local[j][r][c+1];
                    float p1 = expf(s1 * scale - smem_L[(tid/32)*32 + j*16 + r]);
                    
                    write_swizzled_bf16(smem_P0, (tid/32)*32 + j*16 + r, j*16 + c, __float2bfloat16(p0));
                    write_swizzled_bf16(smem_P1, (tid/32)*32 + j*16 + r, j*16 + c+1, __float2bfloat16(p1));
                }
            }
        }
        __syncthreads();

        for(int j = 0; j < 4; j++) {
            for(int r = 0; r < 16; r++) {
                int row = (tid/32)*32 + j*16 + r;
                float lambda = smem_Lambda[row];
                
                for(int c = 0; c < 16; c+=2) {
                    float p0 = __bfloat162float(read_swizzled_bf16(smem_P0, tid, c));
                    float do_val0 = __bfloat162float(read_swizzled_bf16(smem_dO0, tid, c));
                    write_swizzled_bf16(smem_D0, tid, c, __float2bfloat16(do_val0 - p0 * lambda));
                    
                    float p1 = __bfloat162float(read_swizzled_bf16(smem_P1, tid, c+1));
                    float do_val1 = __bfloat162float(read_swizzled_bf16(smem_dO1, tid, c+1));
                    write_swizzled_bf16(smem_D1, tid, c+1, __float2bfloat16(do_val1 - p1 * lambda));
                }
            }
        }
        __syncthreads();

        for (int k = 0; k < 4; ++k) {
            int frag_idx = frag_idx_base + k;
            int i = frag_idx / 8;
            int j = frag_idx % 8;
            
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 128, __nv_bfloat16, 0> D0_a, D1_a;
            load_swizzled<false>("D0", smem_D0, D0_a, 0, 0);
            load_swizzled<false>("D1", smem_D1, D1_a, 0, 0);

            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 128, __nv_bfloat16, 1> V_T0_b, V_T1_b;
            load_transposed_swizzled<true>("V0", smem_V0, smem_V_T0, V_T0_b, 0, 0);
            load_transposed_swizzled<true>("V1", smem_V1, smem_V_T1, V_T1_b, 0, 0);
            
            gemm_128x128<0, 1, 1>(dP_fragments[frag_idx], D0_a, V_T0_b, D1_a, V_T1_b);
            nvcuda::wmma::store_matrix_sync(&dP_local[k][0][0], dP_fragments[frag_idx], 16, nvcuda::wmma::mem_row_major);
        }

        __syncthreads(); 
        for(int k = 0; k < 4; k++) {
            int frag_idx = frag_idx_base + k;
            int i = frag_idx / 8;
            int j = frag_idx % 8;
            store_swizzled<false>("dP0", smem_dP0, i, j, dP_local[k]);
            store_swizzled<false>("dP1", smem_dP1, i, j, dP_local[k]);
        }
        __syncthreads();

        for (int k = 0; k < 4; ++k) {
            int frag_idx = frag_idx_base + k;
            int i = frag_idx / 8;
            int j = frag_idx % 8;
            
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 128, __nv_bfloat16, 1> dPT0_b, dPT1_b;
            load_transposed_swizzled<true>("dP0", smem_dP0, smem_dPT0, dPT0_b, 0, 0);
            load_transposed_swizzled<true>("dP1", smem_dP1, smem_dPT1, dPT1_b, 0, 0);

            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 128, __nv_bfloat16, 0> Q0_a, Q1_a;
            load_swizzled<false>("Q0", smem_Q0, Q0_a, 0, 0);
            load_swizzled<false>("Q1", smem_Q1, Q1_a, 0, 0);
            
            gemm_128x128<0, 1, 1>(dK_fragments[frag_idx], Q0_a, dPT0_b, Q1_a, dPT1_b);
            nvcuda::wmma::store_matrix_sync(&dK_local[k][0][0], dK_fragments[frag_idx], 16, nvcuda::wmma::mem_row_major);
        }

        for (int k = 0; k < 4; ++k) {
            int frag_idx = frag_idx_base + k;
            int i = frag_idx / 8;
            int j = frag_idx % 8;
            
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 128, __nv_bfloat16, 1> PT0_b, PT1_b;
            load_transposed_swizzled<true>("P0", smem_P0, smem_PT0, PT0_b, 0, 0);
            load_transposed_swizzled<true>("P1", smem_P1, smem_PT1, PT1_b, 0, 0);

            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 128, __nv_bfloat16, 0> dO0_a, dO1_a;
            load_swizzled<false>("dO0", smem_dO0, dO0_a, 0, 0);
            load_swizzled<false>("dO1", smem_dO1, dO1_a, 0, 0);
            
            gemm_128x128<0, 1, 1>(dV_fragments[frag_idx], dO0_a, PT0_b, dO1_a, PT1_b);
            nvcuda::wmma::store_matrix_sync(&dV_local[k][0][0], dV_fragments[frag_idx], 16, nvcuda::wmma::mem_row_major);
        }

        __syncthreads(); 
        for(int k = 0; k < 4; k++) {
            int frag_idx = frag_idx_base + k;
            int i = frag_idx / 8;
            int j = frag_idx % 8;
            for(int r = 0; r < 16; r++) {
                for(int c = 0; c < 16; c+=2) {
                    write_swizzled_bf16(smem_dK0, i*16 + r, j*16 + c, __float2bfloat16(dK_local[k][r][c]));
                    write_swizzled_bf16(smem_dK1, i*16 + r, j*16 + c, __float2bfloat16(dK_local[k][r][c+1]));
                    
                    write_swizzled_bf16(smem_dV0, i*16 + r, j*16 + c, __float2bfloat16(dV_local[k][r][c]));
                    write_swizzled_bf16(smem_dV1, i*16 + r, j*16 + c, __float2bfloat16(dV_local[k][r][c+1]));
                }
            }
        }
        __syncthreads();
    }
    
    // Epilogue dK and dV
    for(int i = tid; i < 128 * 64 / 4; i += blockDim.x) {
        int row = (i * 4) / 64;
        int col = (i * 4) % 64;
        if (kv_start + row < S) {
            float4 fk0 = *reinterpret_cast<float4*>(&smem_dK0[row * 64 + col]);
            float4 fk1 = *reinterpret_cast<float4*>(&smem_dK1[row * 64 + col]);
            __nv_bfloat16 out_k[8];
            out_k[0] = __float2bfloat16(fk0.x); out_k[1] = __float2bfloat16(fk0.y); out_k[2] = __float2bfloat16(fk0.z); out_k[3] = __float2bfloat16(fk0.w);
            out_k[4] = __float2bfloat16(fk1.x); out_k[5] = __float2bfloat16(fk1.y); out_k[6] = __float2bfloat16(fk1.z); out_k[7] = __float2bfloat16(fk1.w);
            *reinterpret_cast<float4*>(&dK_ptr[(b_h_idx * S + kv_start + row) * 128 + col]) = *reinterpret_cast<float4*>(out_k);
            *reinterpret_cast<float4*>(&dK_ptr[(b_h_idx * S + kv_start + row) * 128 + col + 64]) = *reinterpret_cast<float4*>(&out_k[4]);

            float4 fv0 = *reinterpret_cast<float4*>(&smem_dV0[row * 64 + col]);
            float4 fv1 = *reinterpret_cast<float4*>(&smem_dV1[row * 64 + col]);
            __nv_bfloat16 out_v[8];
            out_v[0] = __float2bfloat16(fv0.x); out_v[1] = __float2bfloat16(fv0.y); out_v[2] = __float2bfloat16(fv0.z); out_v[3] = __float2bfloat16(fv0.w);
            out_v[4] = __float2bfloat16(fv1.x); out_v[5] = __float2bfloat16(fv1.y); out_v[6] = __float2bfloat16(fv1.z); out_v[7] = __float2bfloat16(fv1.w);
            *reinterpret_cast<float4*>(&dV_ptr[(b_h_idx * S + kv_start + row) * 128 + col]) = *reinterpret_cast<float4*>(out_v);
            *reinterpret_cast<float4*>(&dV_ptr[(b_h_idx * S + kv_start + row) * 128 + col + 64]) = *reinterpret_cast<float4*>(&out_v[4]);
        }
    }
    __syncthreads();
}

namespace tvm_ffi_mha_bwd {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L, 
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
         
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    uint32_t B = 4, H = 48, S = Q.size(0) / (B * H), d = 128;
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    void* Q_ptr = Q.data_ptr();
    void* K_ptr = K.data_ptr();
    void* V_ptr = V.data_ptr();
    void* O_ptr = O.data_ptr();
    void* dO_ptr = dO.data_ptr();
    const float* L_ptr = static_cast<const float*>(L.data_ptr());
    
    __nv_bfloat16* dQ_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O, tma_dO;
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_Q, Q_ptr, 128, S * B * H, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_K, K_ptr, 128, S * B * H, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_V, V_ptr, 128, S * B * H, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_O, O_ptr, 128, S * B * H, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_dO, dO_ptr, 128, S * B * H, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
    dim3 grid((S + 127) / 128, B * H);
    dim3 block(128);
    
    uint32_t smem_size = 17 * 16384 + 1024 + 40; 
    
    CUDA_CHECK(cudaFuncSetAttribute((const void*)attn_backward_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    attn_backward_kernel<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, tma_O, tma_dO,
        L_ptr,
        dQ_ptr, dK_ptr, dV_ptr,
        S, 1.0f / sqrtf((float)d)
    );
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_bwd::run);

} // namespace tvm_ffi_mha_bwd