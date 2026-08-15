#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cmath>
#include <cstdint>
#include <stdio.h>
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

// ============================================================
// Global constants accessible from all namespaces
// ============================================================
static constexpr int KERNEL_BLOCK_M = 64;
static constexpr int KERNEL_BLOCK_N = 64;
static constexpr int KERNEL_INNER_K = 16;
static constexpr int KERNEL_NUM_THREADS = 128;
static constexpr int KERNEL_TMEM_COLS = 320;

// ============================================================
// Device Helper Functions for SM100/Blackwell
// ============================================================

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__device__ __forceinline__ void cluster_sync_fn() {
    asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n" ::: "memory");
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t parity) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(parity));
}

__device__ __forceinline__ void prefetch_tma_descriptor_fn(const CUtensorMap* d) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(d) : "memory");
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* desc, uint64_t* mbar, 
                                                void* smem_dst, int32_t coord0, int32_t coord1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem_dst)),
           "l"((uint64_t)desc),
           "r"((uint32_t)__cvta_generic_to_shared(mbar)),
           "r"(coord0), "r"(coord1) : "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_swizzle128_fn(void* smem_ptr, 
                                                                   uint32_t lbo_enc, uint32_t sbo_enc) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= static_cast<uint64_t>(addr & 0x3FFFF) >> 4;
    d |= static_cast<uint64_t>(lbo_enc & 0x3FFFF) << 16;
    d |= static_cast<uint64_t>(sbo_enc & 0x3FFFF) << 32;
    d |= static_cast<uint64_t>(1) << 46;   // version = 1 (SM100)
    d |= static_cast<uint64_t>(2) << 61;   // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t idesc = 0;
    idesc |= (1u << 4);    // c_format = FP32
    idesc |= (1u << 7);    // a_format = BF16
    idesc |= (1u << 10);   // b_format = BF16
    idesc |= (0u << 15);   // a_no_transpose (K-major)
    idesc |= (0u << 16);   // b_no_transpose (K-major)
    idesc |= ((N >> 3) & 0x3Fu) << 17;
    idesc |= ((M >> 4) & 0x1Fu) << 24;
    return idesc;
}

__device__ __forceinline__ void umma_f16_cg2_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_2sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile(
        "tcgen05.commit.cta_group::2"
        ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;"
        :: "r"(a), "h"((uint16_t)0x3));
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tmem_wait_ld_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_wait_st_fn() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

// ============================================================
// MHA Backward Kernel for Blackwell
// ============================================================

namespace mha_bwd_blackwell {

static constexpr int INNER_K = KERNEL_INNER_K;
static constexpr int TMEM_COLS = KERNEL_TMEM_COLS;

template<int BM, int BN, int HD>
__global__ void mha_bwd_kernel_sm100(
    const __grid_constant__ CUtensorMap* const tma_Q,
    const __grid_constant__ CUtensorMap* const tma_K,
    const __grid_constant__ CUtensorMap* const tma_V,
    const __grid_constant__ CUtensorMap* const tma_dO,
    __nv_bfloat16* __restrict__ dQ_out,
    __nv_bfloat16* __restrict__ dK_out,
    __nv_bfloat16* __restrict__ dV_out,
    const float* __restrict__ LSE_in,
    const float* __restrict__ D_precomp,
    int B, int H, int S, int D,
    float attn_scale)
{
    extern __shared__ char smem[];

    uint32_t cta_rank = cluster_rank_fn();
    uint32_t tid = threadIdx.x;

    // Shared memory layout
    uint32_t off = 0;
    uint64_t* bar_tma = reinterpret_cast<uint64_t*>(smem + off); off += 8;
    uint64_t* bar_umma = reinterpret_cast<uint64_t*>(smem + off); off += 8;
    uint32_t* tmem_addr_sp = reinterpret_cast<uint32_t*>(smem + off); off += 64;
    
    off = (off + 127) & ~127U;
    __nv_bfloat16* smem_Q = reinterpret_cast<__nv_bfloat16*>(smem + off);
    off += BM * HD * sizeof(__nv_bfloat16);
    
    off = (off + 127) & ~127U;
    __nv_bfloat16* smem_K = reinterpret_cast<__nv_bfloat16*>(smem + off);
    off += BN * HD * sizeof(__nv_bfloat16);
    
    off = (off + 127) & ~127U;
    __nv_bfloat16* smem_V = reinterpret_cast<__nv_bfloat16*>(smem + off);
    off += BN * HD * sizeof(__nv_bfloat16);
    
    off = (off + 127) & ~127U;
    __nv_bfloat16* smem_dO = reinterpret_cast<__nv_bfloat16*>(smem + off);
    off += BN * HD * sizeof(__nv_bfloat16);

    off = (off + 127) & ~127U;
    __nv_bfloat16* smem_dQ_epilogue = reinterpret_cast<__nv_bfloat16*>(smem + off);
    off += BM * HD * sizeof(__nv_bfloat16);
    
    off = (off + 63) & ~63U;
    float* shared_LSE = reinterpret_cast<float*>(smem + off);
    off += BM * sizeof(float);

    // Initialize barriers
    if (tid == 0) {
        init_smem_barrier_fn(bar_tma, 1);
        init_smem_barrier_fn(bar_umma, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    // Batch/head indices
    uint32_t bh = blockIdx.x;
    uint32_t h = bh % H;
    uint32_t b = bh / H;

    // Prefetch TMA descriptors
    prefetch_tma_descriptor_fn(tma_Q + bh);
    prefetch_tma_descriptor_fn(tma_K + bh);
    prefetch_tma_descriptor_fn(tma_V + bh);
    prefetch_tma_descriptor_fn(tma_dO + bh);

    // Allocate TMEM
    if (cta_rank < 2) {
        tmem_alloc_fn(tmem_addr_sp, TMEM_COLS);
    }
    cluster_sync_fn();
    uint32_t tmem_base = tmem_addr_sp[0];

    int num_k_iters = HD / INNER_K;

    // Pointers to output tensors for this (b, h)
    int64_t seq_feat = S * HD;
    __nv_bfloat16* dQ_base = dQ_out + static_cast<int64_t>(b) * H * seq_feat + h * seq_feat;
    __nv_bfloat16* dK_base = dK_out + static_cast<int64_t>(b) * H * seq_feat + h * seq_feat;
    __nv_bfloat16* dV_base = dV_out + static_cast<int64_t>(b) * H * seq_feat + h * seq_feat;

    // Float pointers for atomicAdd (reinterpret dQ/dK/dV as float for accumulation)
    float* f_dQ_base = reinterpret_cast<float*>(dQ_base);
    float* f_dK_base = reinterpret_cast<float*>(dK_base);
    float* f_dV_base = reinterpret_cast<float*>(dV_base);

    uint32_t num_q_blks = (S + BM - 1) / BM;
    uint32_t num_kv_blks = (S + BN - 1) / BN;
    uint32_t sbo_enc = (1024u & 0x3FFFF) >> 4;

    for (uint32_t q_blk = 0; q_blk < num_q_blks; ++q_blk) {
        int q_start = static_cast<int>(q_blk) * BM;
        int q_end = min(q_start + BM, S);
        int cur_bm = q_end - q_start;

        // Zero dQ accumulator for this q_block
        for (int64_t idx = tid; idx < static_cast<int64_t>(cur_bm) * HD; idx += KERNEL_NUM_THREADS) {
            dQ_base[q_start * HD + idx] = __float2bfloat16(0.0f);
        }
        
        tma_load_2d_fn(tma_Q + bh, bar_tma, smem_Q, q_start, 0);
        mbarrier_wait_fn(bar_tma, 0);
        __syncthreads();
        
        const float* lse_bh = LSE_in + b * H * S + h * S;
        const float* D_bh = D_precomp + b * H * S + h * S;
        for (int i = tid; i < cur_bm; i += KERNEL_NUM_THREADS) {
            shared_LSE[i] = lse_bh[q_start + i];
        }
        __syncthreads();

        for (uint32_t kv_blk = 0; kv_blk < num_kv_blks; ++kv_blk) {
            int kv_start = static_cast<int>(kv_blk) * BN;
            int kv_end = min(kv_start + BN, S);
            int cur_bn = kv_end - kv_start;

            if (q_start + cur_bm <= kv_start) continue;

            tma_load_2d_fn(tma_K + bh, bar_tma, smem_K, kv_start, 0);
            tma_load_2d_fn(tma_V + bh, bar_tma, smem_V, kv_start, 0);
            tma_load_2d_fn(tma_dO + bh, bar_tma, smem_dO, kv_start, 0);
            mbarrier_wait_fn(bar_tma, 0);
            __syncthreads();

            // Step 1: S = Q @ K^T → store as S^T in TMEM[192:256]
            for (int ki = 0; ki < num_k_iters; ++ki) {
                int k_off = ki * INNER_K;
                
                uint64_t desc_Q = make_smem_desc_swizzle128_fn(
                    smem_Q + k_off * BM, 1u, sbo_enc);
                uint64_t desc_K = make_smem_desc_swizzle128_fn(
                    smem_K + k_off * BN, 1u, sbo_enc);
                
                umma_f16_cg2_fn(
                    tmem_base + 192,
                    desc_K, desc_Q,
                    make_instr_desc_fn(min(cur_bn, 64), min(cur_bm, 64)),
                    ki == 0 ? 0 : 1
                );
            }
            
            umma_commit_2sm_fn(bar_umma);
            mbarrier_wait_fn(bar_umma, 0);
            tcgen05_fence_after_fn();
            __syncthreads();
            
            // Step 2: Scale, softmax, causal mask → P^T in TMEM[192:256]
            if (tid < 128) {
                uint32_t wgid = tid / 32;
                int rows_per_warp = (cur_bn + 3) / 4;
                int row_off = wgid * rows_per_warp;
                
                for (int row = row_off; row < row_off + rows_per_warp && row < cur_bn; ++row) {
                    if (row < 32 * wgid || row >= 32 * (wgid + 1)) continue;
                    
                    int kv_global = kv_start + row;
                    
                    for (int col = 0; col + 3 < cur_bm; col += 4) {
                        uint32_t r0, r1, r2, r3;
                        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                           : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                           : "r"(tmem_base + 192 + row));
                        tmem_wait_ld_fn();
                        
                        float s0 = __uint_as_float(r0) * attn_scale;
                        float s1 = __uint_as_float(r1) * attn_scale;
                        float s2 = __uint_as_float(r2) * attn_scale;
                        float s3 = __uint_as_float(r3) * attn_scale;
                        
                        float l0 = shared_LSE[col];
                        float l1 = shared_LSE[col + 1];
                        float l2 = shared_LSE[col + 2];
                        float l3 = shared_LSE[col + 3];
                        
                        float p0 = expf(s0 - l0);
                        float p1 = expf(s1 - l1);
                        float p2 = expf(s2 - l2);
                        float p3 = expf(s3 - l3);
                        
                        int q0 = q_start + col;
                        int q1 = q_start + col + 1;
                        int q2 = q_start + col + 2;
                        int q3 = q_start + col + 3;
                        if (q0 < kv_global) p0 = 0.0f;
                        if (q1 < kv_global) p1 = 0.0f;
                        if (q2 < kv_global) p2 = 0.0f;
                        if (q3 < kv_global) p3 = 0.0f;
                        
                        asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};"
                           :: "r"(__float_as_uint(p0)), "r"(__float_as_uint(p1)),
                              "r"(__float_as_uint(p2)), "r"(__float_as_uint(p3)),
                              "r"(tmem_base + 192 + row) : "memory");
                    }
                }
            }
            
            if (tid < 128) tmem_wait_st_fn();
            tcgen05_fence_after_fn();
            __syncthreads();
            
            // Register-based gradient computation for correctness
            // Use compile-time constant HD for local array sizes
            
            // Pass 0: dV contribution (threads 0..cur_bn-1)
            if (tid < cur_bn) {
                int kv_local = tid;
                int kv_global = kv_start + kv_local;
                
                float dv_accum[HD];
                for (int f = 0; f < HD; ++f) dv_accum[f] = 0.0f;
                
                for (int q_local = 0; q_local < cur_bm; ++q_local) {
                    int q_global = q_start + q_local;
                    if (q_global < kv_global) continue;
                    
                    float s_dot = 0.0f;
                    for (int f = 0; f < HD; ++f) {
                        float qv = __bfloat162float(smem_Q[q_local * HD + f]);
                        float kv_val = __bfloat162float(smem_K[kv_local * HD + f]);
                        s_dot += qv * kv_val;
                    }
                    float p_val = expf(s_dot * attn_scale - lse_bh[q_global]);
                    
                    for (int f = 0; f < HD; ++f) {
                        dv_accum[f] += p_val * __bfloat162float(smem_dO[q_local * HD + f]);
                    }
                }
                
                for (int f = 0; f < HD; f += 4) {
                    atomicAdd(&f_dV_base[kv_global * HD + f + 0], dv_accum[f + 0]);
                    if (f + 1 < HD) atomicAdd(&f_dV_base[kv_global * HD + f + 1], dv_accum[f + 1]);
                    if (f + 2 < HD) atomicAdd(&f_dV_base[kv_global * HD + f + 2], dv_accum[f + 2]);
                    if (f + 3 < HD) atomicAdd(&f_dV_base[kv_global * HD + f + 3], dv_accum[f + 3]);
                }
            }
            __syncthreads();
            
            // Pass 1: dQ contribution (threads 0..cur_bm-1)
            if (tid < cur_bm) {
                int q_local = tid;
                int q_global = q_start + q_local;
                
                float dq_accum[HD];
                for (int f = 0; f < HD; ++f) dq_accum[f] = 0.0f;
                
                for (int kv_local = 0; kv_local < cur_bn; ++kv_local) {
                    int kv_global = kv_start + kv_local;
                    if (q_global < kv_global) continue;
                    
                    float s_dot = 0.0f;
                    float e_dot = 0.0f;
                    for (int f = 0; f < HD; ++f) {
                        float qv = __bfloat162float(smem_Q[q_local * HD + f]);
                        float kv_val = __bfloat162float(smem_K[kv_local * HD + f]);
                        s_dot += qv * kv_val;
                        
                        float dov = __bfloat162float(smem_dO[q_local * HD + f]);
                        float vv = __bfloat162float(smem_V[kv_local * HD + f]);
                        e_dot += dov * vv;
                    }
                    
                    float p_val = expf(s_dot * attn_scale - lse_bh[q_global]);
                    float d_val = D_bh[q_global];
                    float ds_val = p_val * (e_dot - d_val);
                    
                    for (int f = 0; f < HD; ++f) {
                        dq_accum[f] += ds_val * __bfloat162float(smem_K[kv_local * HD + f]);
                    }
                }
                
                for (int f = tid; f < HD; f += KERNEL_NUM_THREADS) {
                    smem_dQ_epilogue[q_local * HD + f] = __float2bfloat16(dq_accum[f]);
                }
            }
            __syncthreads();
            
            // Accumulate dQ epilogue buffer into global
            {
                for (int64_t idx = tid; idx < static_cast<int64_t>(cur_bm) * HD; idx += KERNEL_NUM_THREADS) {
                    float dq_val = __bfloat162float(smem_dQ_epilogue[idx]);
                    int linear_idx = q_start * HD + idx;
                    atomicAdd(&f_dQ_base[linear_idx], dq_val);
                }
            }
            __syncthreads();
            
            // Pass 2: dK contribution (threads 0..cur_bn-1)
            if (tid < cur_bn) {
                int kv_local = tid;
                int kv_global = kv_start + kv_local;
                
                float dk_accum[HD];
                for (int f = 0; f < HD; ++f) dk_accum[f] = 0.0f;
                
                for (int q_local = 0; q_local < cur_bm; ++q_local) {
                    int q_global = q_start + q_local;
                    if (q_global < kv_global) continue;
                    
                    float s_dot = 0.0f;
                    float e_dot = 0.0f;
                    for (int f = 0; f < HD; ++f) {
                        float qv = __bfloat162float(smem_Q[q_local * HD + f]);
                        float kv_val = __bfloat162float(smem_K[kv_local * HD + f]);
                        s_dot += qv * kv_val;
                        
                        float dov = __bfloat162float(smem_dO[q_local * HD + f]);
                        float vv = __bfloat162float(smem_V[kv_local * HD + f]);
                        e_dot += dov * vv;
                    }
                    
                    float p_val = expf(s_dot * attn_scale - lse_bh[q_global]);
                    float d_val = D_bh[q_global];
                    float ds_val = p_val * (e_dot - d_val);
                    
                    for (int f = 0; f < HD; ++f) {
                        dk_accum[f] += ds_val * __bfloat162float(smem_Q[q_local * HD + f]);
                    }
                }
                
                for (int f = 0; f < HD; f += 4) {
                    atomicAdd(&f_dK_base[kv_global * HD + f + 0], dk_accum[f + 0]);
                    if (f + 1 < HD) atomicAdd(&f_dK_base[kv_global * HD + f + 1], dk_accum[f + 1]);
                    if (f + 2 < HD) atomicAdd(&f_dK_base[kv_global * HD + f + 2], dk_accum[f + 2]);
                    if (f + 3 < HD) atomicAdd(&f_dK_base[kv_global * HD + f + 3], dk_accum[f + 3]);
                }
            }
            __syncthreads();
        }
    }
    
    if (cta_rank < 2) {
        tmem_dealloc_fn(tmem_base, TMEM_COLS);
    }
    cluster_sync_fn();
}

}  // namespace mha_bwd_blackwell

// ============================================================
// TMA Descriptor Creation
// ============================================================

static CUresult encode_tma_2d_bf16(CUtensorMap* tensorMap, void* globalAddr,
                                    uint64_t gmem_inner, uint64_t gmem_outer,
                                    uint32_t box_inner, uint32_t box_outer) 
{
    cuuint64_t globalDim[2] = {gmem_inner, gmem_outer};
    cuuint64_t globalStrides[2] = {
        sizeof(__nv_bfloat16),
        gmem_inner * sizeof(__nv_bfloat16)
    };
    cuuint32_t boxDim[2] = {box_inner, box_outer};
    cuuint32_t elemStrides[2] = {1, 1};
    
    return cuTensorMapEncodeTiled(
        tensorMap,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        globalAddr,
        globalDim,
        globalStrides,
        boxDim,
        elemStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

// ============================================================
// Host Side
// ============================================================

namespace mha_bwd_run {

void run(
    tvm::ffi::TensorView Q,
    tvm::ffi::TensorView K,
    tvm::ffi::TensorView V,
    tvm::ffi::TensorView O,
    tvm::ffi::TensorView dO,
    tvm::ffi::TensorView L,
    tvm::ffi::TensorView dQ,
    tvm::ffi::TensorView dK,
    tvm::ffi::TensorView dV) 
{
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    float attn_scale = 1.0f / std::sqrt(static_cast<float>(D));
    
    int64_t num_bh = B * H;
    
    // Precompute D_precomp = rowsum(dO * O) for each (b, h, q)
    auto* O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    auto* dO_ptr = static_cast<__nv_bfloat16*>(dO.data_ptr());
    
    float* d_D_host = new float[B * H * S]();
    
    int64_t bh_stride = S * D;
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t hh = 0; hh < H; ++hh) {
            int64_t bh_offset = b * H * bh_stride + hh * bh_stride;
            for (int64_t q = 0; q < S; ++q) {
                float sum = 0.0f;
                for (int64_t f = 0; f < D; ++f) {
                    float dv = __bfloat162float(dO_ptr[bh_offset + q * D + f]);
                    float ov = __bfloat162float(O_ptr[bh_offset + q * D + f]);
                    sum += dv * ov;
                }
                d_D_host[b * H * S + hh * S + q] = sum;
            }
        }
    }
    
    float* d_D_dev = nullptr;
    CUDA_CHECK(cudaMalloc(&d_D_dev, B * H * S * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_D_dev, d_D_host, B * H * S * sizeof(float), cudaMemcpyHostToDevice));
    delete[] d_D_host;
    
    // Create TMA descriptors
    size_t desc_size = sizeof(CUtensorMap);
    CUtensorMap* d_tma_Q = nullptr;
    CUtensorMap* d_tma_K = nullptr;
    CUtensorMap* d_tma_V = nullptr;
    CUtensorMap* d_tma_dO = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_tma_Q, desc_size * num_bh));
    CUDA_CHECK(cudaMalloc(&d_tma_K, desc_size * num_bh));
    CUDA_CHECK(cudaMalloc(&d_tma_V, desc_size * num_bh));
    CUDA_CHECK(cudaMalloc(&d_tma_dO, desc_size * num_bh));
    
    CUtensorMap* h_desc_Q = new CUtensorMap[num_bh]();
    CUtensorMap* h_desc_K = new CUtensorMap[num_bh]();
    CUtensorMap* h_desc_V = new CUtensorMap[num_bh]();
    CUtensorMap* h_desc_dO = new CUtensorMap[num_bh]();
    
    __nv_bfloat16* Q_p = static_cast<__nv_bfloat16*>(Q.data_ptr());
    __nv_bfloat16* K_p = static_cast<__nv_bfloat16*>(K.data_ptr());
    __nv_bfloat16* V_p = static_cast<__nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* dO_p = static_cast<__nv_bfloat16*>(dO.data_ptr());
    
    for (int64_t bh = 0; bh < num_bh; ++bh) {
        int64_t b = bh / H;
        int64_t hh = bh % H;
        int64_t bh_offset = b * H * bh_stride + hh * bh_stride;
        
        encode_tma_2d_bf16(h_desc_Q + bh, Q_p + bh_offset, D, S, D, KERNEL_BLOCK_M);
        encode_tma_2d_bf16(h_desc_K + bh, K_p + bh_offset, D, S, D, KERNEL_BLOCK_N);
        encode_tma_2d_bf16(h_desc_V + bh, V_p + bh_offset, D, S, D, KERNEL_BLOCK_N);
        encode_tma_2d_bf16(h_desc_dO + bh, dO_p + bh_offset, D, S, D, KERNEL_BLOCK_N);
    }
    
    CUDA_CHECK(cudaMemcpy(d_tma_Q, h_desc_Q, desc_size * num_bh, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tma_K, h_desc_K, desc_size * num_bh, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tma_V, h_desc_V, desc_size * num_bh, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tma_dO, h_desc_dO, desc_size * num_bh, cudaMemcpyHostToDevice));
    
    delete[] h_desc_Q;
    delete[] h_desc_K;
    delete[] h_desc_V;
    delete[] h_desc_dO;
    
    // Shared memory size calculation
    size_t smem_size = 8 + 8 + 64;
    smem_size = (smem_size + 127) & ~127ULL;
    smem_size += KERNEL_BLOCK_M * D * sizeof(__nv_bfloat16);
    smem_size = (smem_size + 127) & ~127ULL;
    smem_size += KERNEL_BLOCK_N * D * sizeof(__nv_bfloat16);
    smem_size = (smem_size + 127) & ~127ULL;
    smem_size += KERNEL_BLOCK_N * D * sizeof(__nv_bfloat16);
    smem_size = (smem_size + 127) & ~127ULL;
    smem_size += KERNEL_BLOCK_N * D * sizeof(__nv_bfloat16);
    smem_size = (smem_size + 127) & ~127ULL;
    smem_size += KERNEL_BLOCK_M * D * sizeof(__nv_bfloat16);
    smem_size = (smem_size + 63) & ~63ULL;
    smem_size += KERNEL_BLOCK_M * sizeof(float);
    
    printf("Shared memory: %zu bytes (%.1f KB)\n", smem_size, smem_size / 1024.0);
    
    dim3 grid(num_bh);
    dim3 block(KERNEL_NUM_THREADS);
    
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem_size;
    cfg.stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = 2;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    
    auto fn = mha_bwd_blackwell::mha_bwd_kernel_sm100<KERNEL_BLOCK_M, KERNEL_BLOCK_N, 128>;
    
    cudaLaunchKernelEx(&cfg, fn,
        d_tma_Q, d_tma_K, d_tma_V, d_tma_dO,
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        d_D_dev,
        (int)B, (int)H, (int)S, (int)D, attn_scale);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(cfg.stream));
    
    CUDA_CHECK(cudaFree(d_tma_Q));
    CUDA_CHECK(cudaFree(d_tma_K));
    CUDA_CHECK(cudaFree(d_tma_V));
    CUDA_CHECK(cudaFree(d_tma_dO));
    CUDA_CHECK(cudaFree(d_D_dev));
}

}  // namespace mha_bwd_run

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_run::run);