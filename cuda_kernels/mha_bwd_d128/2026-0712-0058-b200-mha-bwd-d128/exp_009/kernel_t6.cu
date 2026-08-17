#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <mma.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>
#include <algorithm>

using namespace nvcuda;

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_mha_bwd {

__device__ __forceinline__ int warpgroup_id = 0;
__device__ __forceinline__ int lane_id = 0;

__device__ __forceinline__ __nv_bfloat16 atomicAdd_bf16(__nv_bfloat16* address, __nv_bfloat16 val) {
    uint32_t* addr32 = (uint32_t*)(((uintptr_t)address) & ~3);
    int offset = (((uintptr_t)address) & 3) / 2;
    uint32_t old = *addr32;
    uint32_t assumed;
    do {
        assumed = old;
        uint16_t old_bf16 = (offset == 0) ? (assumed & 0xFFFF) : (assumed >> 16);
        float f_val = __bfloat162float(val);
        float f_old = __bfloat162float(*reinterpret_cast<float*>(&old_bf16));
        float f_sum = f_old + f_val;
        __nv_bfloat16 sum_bf16 = __float2bfloat16(f_sum);
        uint16_t sum_u16 = *reinterpret_cast<uint16_t*>(&sum_bf16);
        uint32_t new_val = (offset == 0) ? (sum_u16 | (assumed & 0xFFFF0000)) : ((sum_u16 << 16) | (assumed & 0x0000FFFF));
        old = atomicCAS(addr32, assumed, new_val);
    } while (assumed != old);
    return val;
}

__device__ __forceinline__ void atomic_add_float4(float* address, float4 val) {
    uint32_t* addr32 = (uint32_t*)(((uintptr_t)address) & ~15);
    int offset = (((uintptr_t)address) & 15) / 4;
    uint4 old = *(uint4*)addr32;
    uint4 assumed;
    float* f_old = (float*)&old;
    float* f_assumed = (float*)&assumed;
    do {
        assumed = old;
        float4 to_add = make_float4(0,0,0,0);
        if (offset <= 0) to_add.x = val.x;
        if (offset <= 1) to_add.y = val.y;
        if (offset <= 2) to_add.z = val.z;
        if (offset <= 3) to_add.w = val.w;
        
        f_assumed[offset + 0] += to_add.x;
        f_assumed[offset + 1] += to_add.y;
        f_assumed[offset + 2] += to_add.z;
        f_assumed[offset + 3] += to_add.w;
        old = *(uint4*)atomicCAS((unsigned long long*)addr32, old, assumed);
    } while (assumed != old);
}

__device__ __forceinline__ void cp_async_128_byte(void* smem_dst, const void* smem_src) {
    uint32_t dst = (uint32_t)__cvta_generic_to_shared(smem_dst);
    uint32_t src = (uint32_t)__cvta_generic_to_shared(smem_src);
    asm volatile("cp.async.shared.shared.p128 [%0], [%1];\n" :: "r"(dst), "r"(src));
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

__device__ __forceinline__ bool elect_one_sync_fn() {
    uint32_t pred;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "elect.sync _|p, 0xFFFFFFFF;\n"
        "selp.b32 %0, 1, 0, p;\n"
        "}\n"
        : "=r"(pred));
    return pred != 0;
}

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(phase));
}

__device__ __forceinline__ void tma_load_2d_cg2_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t ba = (uint32_t)__cvta_generic_to_shared(bar) & 0xFEFFFFFF;
    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(sa), "l"((uint64_t)d), "r"(c0), "r"(c1), "r"(ba) : "memory");
}

__device__ __forceinline__ __nv_bfloat16 read_swizzled(const __nv_bfloat16* smem, uint32_t row, uint32_t col) {
    uint32_t s_col = (col / 8) ^ (row % 8);
    return smem[row * 64 + s_col * 8 + (col % 8)];
}

__device__ __forceinline__ __nv_bfloat16 read_swizzled_or_zero(const __nv_bfloat16* smem, uint32_t row, uint32_t col, uint32_t max_rows) {
    if (row >= max_rows) return __float2bfloat16(0.0f);
    uint32_t s_col = (col / 8) ^ (row % 8);
    return smem[row * 64 + s_col * 8 + (col % 8)];
}

__device__ __forceinline__ void write_swizzled(__nv_bfloat16* smem, uint32_t row, uint32_t col, __nv_bfloat16 val, uint32_t max_rows) {
    if (row < max_rows) {
        uint32_t s_col = (col / 8) ^ (row % 8);
        smem[row * 64 + s_col * 8 + (col % 8)] = val;
    }
}

__device__ __forceinline__ void load_matrix(const __nv_bfloat16* smem, uint32_t row_base, uint32_t col_base, 
                                     wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major>& src_frag) {
    uint32_t wg = warpgroup_id;
    uint32_t row_idx = row_base + wg * 16;
    for (int step = 0; step < 4; ++step) {
        uint32_t col = col_base + step * 16;
        uint32_t s_col = (col / 8) ^ (row_idx % 8);
        uint32_t smem_addr = (row_idx * 64) + (s_col * 8) + (col % 8);
        uint4 in_val = *(const uint4*)(smem + smem_addr);
        wmma::fill_fragment(src_frag);
        *reinterpret_cast<uint4*>(&src_frag.x[step * 2]) = in_val;
    }
}

__device__ __forceinline__ void load_matrix_transposed(const __nv_bfloat16* smem, uint32_t row_base, uint32_t col_base, 
                                             wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major>& src_frag) {
    uint32_t wg = warpgroup_id;
    uint32_t col_idx = col_base + wg * 16;
    for (int step = 0; step < 4; ++step) {
        uint32_t row = row_base + step * 16;
        uint32_t s_col = (row / 8) ^ (col_idx % 8);
        uint32_t smem_addr = (col_idx * 64) + (s_col * 8) + (row % 8);
        uint4 in_val = *(const uint4*)(smem + smem_addr);
        wmma::fill_fragment(src_frag);
        *reinterpret_cast<uint4*>(&src_frag.x[step * 2]) = in_val;
    }
}

__device__ __forceinline__ void store_matrix(__nv_bfloat16* gmem, int row_base, int tid, 
                                            const wmma::fragment<wmma::accumulator, 16, 16, 16, float>& out_frag) {
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            float val = wmma::read_register(out_frag, i, j);
            int r = row_base + warpgroup_id * 16 + i;
            int c = tid % 64 + (j / 4) * 16 + (tid / 64) * 64;
            if (r < S_len && c < 128) {
                gmem[(b_h * S_len + r) * 128 + c] = __float2bfloat16(val);
            }
        }
    }
}

__global__ void mha_bwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* L,
    __nv_bfloat16* dQ,
    __nv_bfloat16* dK,
    __nv_bfloat16* dV,
    int S_len)
{
    int b_h = blockIdx.y;
    int n_blk = blockIdx.x;
    int n_base = n_blk * 64;
    int tid = threadIdx.x;
    warpgroup_id = tid / 32;
    lane_id = tid % 32;

    if (n_base >= S_len) return;

    extern __shared__ __align__(128) char smem_raw[];
    __nv_bfloat16* smem_pool = (__nv_bfloat16*)smem_raw;
    
    __nv_bfloat16* smem_Q0 = smem_pool;                
    __nv_bfloat16* smem_Q1 = smem_pool + 4096;         
    __nv_bfloat16* smem_K0 = smem_pool + 8192;         
    __nv_bfloat16* smem_K1 = smem_pool + 12288;        
    __nv_bfloat16* smem_V0 = smem_pool + 16384;        
    __nv_bfloat16* smem_V1 = smem_pool + 20480;        
    __nv_bfloat16* smem_dO = smem_pool + 24576;        
    __nv_bfloat16* smem_O = smem_pool + 32768;         
    __nv_bfloat16* smem_dS = smem_pool + 40960;        
    __nv_bfloat16* smem_P_T = smem_pool + 49152;       
    __nv_bfloat16* smem_dQ_stage = smem_pool + 20480;  

    __shared__ float smem_D[64];
    __shared__ float smem_LSE[64];

    __shared__ alignas(16) uint64_t mbar_load[2];
    __shared__ alignas(16) uint64_t mbar_S[2];
    __shared__ alignas(16) uint64_t mbar_dP[2];
    __shared__ alignas(16) uint64_t mbar_dV[2];
    __shared__ alignas(16) uint64_t mbar_dK_dQ[2];

    uint32_t c0 = cluster_rank_fn();

    if (elect_one_sync_fn()) {
        init_smem_barrier_fn(&mbar_load[c0], 1);
        init_smem_barrier_fn(&mbar_S[c0], 1);
        init_smem_barrier_fn(&mbar_dP[c0], 1);
        init_smem_barrier_fn(&mbar_dV[c0], 1);
        init_smem_barrier_fn(&mbar_dK_dQ[c0], 1);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();
    __syncthreads();

    int n_base_c0 = n_base + (c0 == 0 ? 0 : 64);
    int outer_offset = b_h * S_len + n_base_c0;
    
    mbarrier_arrive_and_expect_tx_fn(&mbar_load[c0], 16384);
    if (elect_one_sync_fn()) {
        tma_load_2d_cg2_fn(&tma_K, &mbar_load[0], smem_K0, 0, outer_offset);
        tma_load_2d_cg2_fn(&tma_K, &mbar_load[0], smem_K1, 64, outer_offset);
        tma_load_2d_cg2_fn(&tma_V, &mbar_load[0], smem_V0, 0, outer_offset);
        tma_load_2d_cg2_fn(&tma_V, &mbar_load[0], smem_V1, 64, outer_offset);
    }
    mbarrier_wait_fn(&mbar_load[c0], 0);

    int q_base = 0;
    outer_offset = b_h * S_len + q_base;
    mbarrier_arrive_and_expect_tx_fn(&mbar_load[c0], 16384);
    if (elect_one_sync_fn()) {
        tma_load_2d_cg2_fn(&tma_Q, &mbar_load[0], smem_Q0, 0, outer_offset);
        tma_load_2d_cg2_fn(&tma_Q, &mbar_load[0], smem_Q1, 64, outer_offset);
        tma_load_2d_cg2_fn(&tma_dO, &mbar_load[0], smem_dO, 0, outer_offset);
        tma_load_2d_cg2_fn(&tma_O, &mbar_load[0], smem_dO + 4096, 0, outer_offset);
        tma_load_2d_cg2_fn(&tma_dO, &mbar_load[0], smem_O, 64, outer_offset);
        tma_load_2d_cg2_fn(&tma_O, &mbar_load[0], smem_O + 4096, 64, outer_offset);
    }

    if (tid < 64) {
        float d = 0;
        int o_row = q_base + tid;
        if (o_row < S_len) {
            for (int c = 0; c < 64; c++) {
                d += __bfloat162float(read_swizzled(smem_dO, tid, c)) * __bfloat162float(read_swizzled(smem_O, tid, c));
                d += __bfloat162float(read_swizzled(smem_dO + 4096, tid, c)) * __bfloat162float(read_swizzled(smem_O + 4096, tid, c));
            }
        }
        smem_D[tid] = d;
        smem_LSE[tid] = (q_base + tid < S_len) ? L[b_h * S_len + q_base + tid] : 0;
    }
    __syncthreads();
    
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_S_C[4];
    wmma::fill_fragment(frag_S_C[0]);
    wmma::fill_fragment(frag_S_C[1]);
    wmma::fill_fragment(frag_S_C[2]);
    wmma::fill_fragment(frag_S_C[3]);
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> frag_Q_A[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> frag_K_B[4];

    for (int k_step = 0; k_step < 8; ++k_step) {
        load_matrix(k_step < 4 ? smem_Q0 : smem_Q1, warpgroup_id * 16, k_step * 16, frag_Q_A[k_step % 4]);
        load_matrix(k_step < 4 ? smem_K0 : smem_K1, warpgroup_id * 16, k_step * 16, frag_K_B[k_step % 4]);
        wmma::scale_dot_reduce_sync(__sync_warp_memory, frag_Q_A[k_step % 4], frag_K_B[k_step % 4], frag_S_C[0]);
    }

    float attn_scale = 1.0f / sqrtf(128.0f);
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            float s = wmma::read_register(frag_S_C[0], i, j) * attn_scale - smem_LSE[tid];
            write_swizzled(smem_P_T, tid, j, __float2bfloat16(expf(s)), S_len);
        }
    }
    __syncthreads();
    
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_dP_T_C[4];
    wmma::fill_fragment(frag_dP_T_C[0]);
    wmma::fill_fragment(frag_dP_T_C[1]);
    wmma::fill_fragment(frag_dP_T_C[2]);
    wmma::fill_fragment(frag_dP_T_C[3]);
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> frag_V_A[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> frag_dO_B[4];

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            wmma::fill_fragment(frag_dP_T_C[i]);
            for (int k_step = 0; k_step < 4; ++k_step) {
                load_matrix(i == 0 ? smem_V0 : smem_V1, tid * 16, k_step * 16, frag_V_A[k_step]);
                load_matrix_transposed(j == 0 ? smem_dO : (j == 1 ? smem_dO + 2048 : (j == 2 ? smem_O : (smem_O + 2048))), k_step * 16, warpgroup_id * 16, frag_dO_B[k_step]);
                wmma::scale_dot_reduce_sync(__sync_warp_memory, frag_V_A[k_step], frag_dO_B[k_step], frag_dP_T_C[i]);
            }
        }
    }

    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            float dp = wmma::read_register(frag_dP_T_C[0], i, j);
            float p = __bfloat162float(read_swizzled_or_zero(smem_P_T, tid, j, S_len));
            float ds = p * (dp - smem_D[tid]) * attn_scale;
            write_swizzled(smem_dS, tid, j, __float2bfloat16(ds), S_len);
        }
    }
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_dV_C[2];
    wmma::fill_fragment(frag_dV_C[0]);
    wmma::fill_fragment(frag_dV_C[1]);
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> frag_PT_A[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> frag_dO_B2[4];

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            wmma::fill_fragment(frag_dV_C[i]);
            for (int k = 0; k < 4; ++k) {
                load_matrix_transposed(smem_P_T, k * 16, j * 16, frag_PT_A[k]);
                load_matrix_transposed(j == 0 ? smem_dO : (j == 1 ? smem_dO + 2048 : (j == 2 ? smem_O : (smem_O + 2048))), k * 16, warpgroup_id * 16, frag_dO_B2[k]);
                wmma::scale_dot_reduce_sync(__sync_warp_memory, frag_PT_A[k], frag_dO_B2[k], frag_dV_C[i]);
            }
        }
    }

    int phase_load = 0;
    mbarrier_wait_fn(&mbar_load[c0], phase_load);
    phase_load ^= 1;
    
    for (int q_blk = 1; q_blk < (S_len + 63) / 64; ++q_blk) {
        int next_q_base = q_blk * 64;
        if (next_q_base >= S_len) break;

        mbarrier_arrive_and_expect_tx_fn(&mbar_load[c0], 16384);
        if (elect_one_sync_fn()) {
            int next_outer = b_h * S_len + next_q_base;
            tma_load_2d_cg2_fn(&tma_Q, &mbar_load[0], smem_Q0, 0, next_outer);
            tma_load_2d_cg2_fn(&tma_Q, &mbar_load[0], smem_Q1, 64, next_outer);
            tma_load_2d_cg2_fn(&tma_dO, &mbar_load[0], smem_dO, 0, next_outer);
            tma_load_2d_cg2_fn(&tma_O, &mbar_load[0], smem_dO + 4096, 0, next_outer);
            tma_load_2d_cg2_fn(&tma_dO, &mbar_load[0], smem_O, 64, next_outer);
            tma_load_2d_cg2_fn(&tma_O, &mbar_load[0], smem_O + 4096, 64, next_outer);
        }
        
        if (tid < 64) {
            float d = 0;
            int o_row = q_base + tid;
            if (o_row < S_len) {
                for (int c = 0; c < 64; c++) {
                    d += __bfloat162float(read_swizzled(smem_dO, tid, c)) * __bfloat162float(read_swizzled(smem_O, tid, c));
                    d += __bfloat162float(read_swizzled(smem_dO + 4096, tid, c)) * __bfloat162float(read_swizzled(smem_O + 4096, tid, c));
                }
            }
            smem_D[tid] = d;
            smem_LSE[tid] = (q_base + tid < S_len) ? L[b_h * S_len + q_base + tid] : 0;
        }
        __syncthreads();
        
        mbarrier_wait_fn(&mbar_load[c0], phase_load);
        phase_load ^= 1;
        
        wmma::fill_fragment(frag_S_C[0]);
        for (int k_step = 0; k_step < 8; ++k_step) {
            load_matrix(k_step < 4 ? smem_Q0 : smem_Q1, warpgroup_id * 16, k_step * 16, frag_Q_A[k_step % 4]);
            load_matrix(k_step < 4 ? smem_K0 : smem_K1, warpgroup_id * 16, k_step * 16, frag_K_B[k_step % 4]);
            wmma::scale_dot_reduce_sync(__sync_warp_memory, frag_Q_A[k_step % 4], frag_K_B[k_step % 4], frag_S_C[0]);
        }

        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++) {
                float s = wmma::read_register(frag_S_C[0], i, j) * attn_scale - smem_LSE[tid];
                write_swizzled(smem_P_T, tid, j, __float2bfloat16(expf(s)), S_len);
            }
        }
        __syncthreads(); 
        
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 4; ++j) {
                wmma::fill_fragment(frag_dP_T_C[i]);
                for (int k_step = 0; k_step < 4; ++k_step) {
                    load_matrix(i == 0 ? smem_V0 : smem_V1, tid * 16, k_step * 16, frag_V_A[k_step]);
                    load_matrix_transposed(j == 0 ? smem_dO : (j == 1 ? smem_dO + 2048 : (j == 2 ? smem_O : (smem_O + 2048))), k_step * 16, warpgroup_id * 16, frag_dO_B[k_step]);
                    wmma::scale_dot_reduce_sync(__sync_warp_memory, frag_V_A[k_step], frag_dO_B[k_step], frag_dP_T_C[i]);
                }
            }
        }

        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++) {
                float dp = wmma::read_register(frag_dP_T_C[0], i, j);
                float p = __bfloat162float(read_swizzled_or_zero(smem_P_T, tid, j, S_len));
                float ds = p * (dp - smem_D[tid]) * attn_scale;
                write_swizzled(smem_dS, tid, j, __float2bfloat16(ds), S_len);
            }
        }
        __syncthreads(); 

        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 4; ++j) {
                wmma::fill_fragment(frag_dV_C[i]);
                for (int k = 0; k < 4; ++k) {
                    load_matrix_transposed(smem_P_T, k * 16, j * 16, frag_PT_A[k]);
                    load_matrix_transposed(j == 0 ? smem_dO : (j == 1 ? smem_dO + 2048 : (j == 2 ? smem_O : (smem_O + 2048))), k * 16, warpgroup_id * 16, frag_dO_B2[k]);
                    wmma::scale_dot_reduce_sync(__sync_warp_memory, frag_PT_A[k], frag_dO_B2[k], frag_dV_C[i]);
                }
            }
        }

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_dK_C[2];
        wmma::fill_fragment(frag_dK_C[0]);
        wmma::fill_fragment(frag_dK_C[1]);
        
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> frag_dS_A[4];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> frag_Q_B[4];

        for (int i = 0; i < 2; ++i) {
            for (int k_step = 0; k_step < 4; ++k_step) {
                load_matrix_transposed(smem_dS, k_step * 16, warpgroup_id * 16, frag_dS_A[k_step % 4]);
                load_matrix_transposed(i == 0 ? smem_Q0 : smem_Q1, k_step * 16, warpgroup_id * 16, frag_Q_B[k_step % 4]);
                wmma::scale_dot_reduce_sync(__sync_warp_memory, frag_dS_A[k_step % 4], frag_Q_B[k_step % 4], frag_dK_C[i]);
            }
        }
        store_matrix(dK, n_base_c0, tid, frag_dK_C);
        
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_dQ_C[2];
        wmma::fill_fragment(frag_dQ_C[0]);
        wmma::fill_fragment(frag_dQ_C[1]);
        
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> frag_dS_A2[4];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> frag_K_B2[4];

        for (int i = 0; i < 2; ++i) {
            for (int k_step = 0; k_step < 4; ++k_step) {
                load_matrix(smem_dS, k_step * 16, warpgroup_id * 16, frag_dS_A2[k_step % 4]);
                load_matrix(k_step < 4 ? smem_K0 : smem_K1, k_step * 16, warpgroup_id * 16, frag_K_B2[k_step % 4]);
                wmma::scale_dot_reduce_sync(__sync_warp_memory, frag_dS_A2[k_step % 4], frag_K_B2[k_step % 4], frag_dQ_C[i]);
            }
        }

        for (int i = 0; i < 2; ++i) {
            cp_async_128_byte((__nv_bfloat16*)(smem_dQ_stage + tid * 16), (__nv_bfloat16*)(&frag_dQ_C[i].x[0]));
            __syncthreads();
            
            float4 dQ_vals = make_float4(0,0,0,0);
            float* f_dQ = (float*)&dQ_vals;
            float* f_frag = (float*)&frag_dQ_C[i].x[0];
            for(int k=0; k<16; ++k) f_dQ[k] = f_frag[k];
            
            int q_row = (q_base - 64) + warpgroup_id * 16;
            uint32_t b_off = (b_h * S_len + q_row) * 128;
            float* dQ_ptr = (float*)&dQ[b_off + ((tid % 64) + (tid / 64) * 64) * 2];
            atomic_add_float4(dQ_ptr, dQ_vals);
        }
        __syncthreads();

        q_base = next_q_base;
    }

    store_matrix(dV, n_base_c0, tid, frag_dV_C);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = Q.size(3);

    if (d != 128) {
        fprintf(stderr, "Expected head dim 128, got %ld\n", d);
        exit(1);
    }

    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_data = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_data = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_data = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_data = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_data = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_data = static_cast<__nv_bfloat16*>(dV.data_ptr());

    CUDA_CHECK(cudaMemsetAsync(dQ_data, 0, B * H * S * 128 * sizeof(__nv_bfloat16)));

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUtensorMap tma_Q, tma_K, tma_V, tma_O, tma_dO;

    auto create_tma = [&](CUtensorMap* tma, const void* ptr) {
        cuuint64_t globalDim[2] = {128, (cuuint64_t)(B * H * S)};
        cuuint64_t globalStrides[1] = {128 * 2};
        cuuint32_t boxDim[2] = {64, 64};
        cuuint32_t elementStrides[2] = {1, 1};
        CUresult res = cuTensorMapEncodeTiled(
            tma, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)ptr, globalDim, globalStrides, boxDim, elementStrides,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        if (res != CUDA_SUCCESS) {
            fprintf(stderr, "cuTensorMapEncodeTiled failed with %d\n", res);
            exit(1);
        }
    };

    create_tma(&tma_Q, Q_data);
    create_tma(&tma_K, K_data);
    create_tma(&tma_V, V_data);
    create_tma(&tma_O, O_data);
    create_tma(&tma_dO, dO_data);

    int num_tiles = (S + 63) / 64;
    dim3 grid(num_tiles, B * H);
    dim3 block(128);
    
    int smem_size = 73728;
    CUDA_CHECK(cudaFuncSetAttribute(tvm_ffi_mha_bwd::mha_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    CUDA_CHECK(cudaLaunchKernelEx(&config, tvm_ffi_mha_bwd::mha_bwd_kernel,
        tma_Q, tma_K, tma_V, tma_O, tma_dO, L_data,
        dQ_data, dK_data, dV_data, S
    ));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_bwd::run);

} // namespace tvm_ffi_mha_bwd