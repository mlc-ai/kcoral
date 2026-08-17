#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
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

#define CU_CHECK(call) do {                                        \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        fprintf(stderr, "CU error %d at %s:%d\n",                 \
                (int)_e, __FILE__, __LINE__);                      \
        exit(1);                                                   \
    }                                                              \
} while(0)

// -------------------------------------------------------------------------
// Device Helper Functions
// -------------------------------------------------------------------------

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes) : "memory");
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

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void tma_load_multicast_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, uint16_t mask) {
    uint64_t cache_hint = 0;
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%0], [%1, {%4, %5}], [%2], %3, %6;"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "h"(mask), "r"(c0), "r"(c1), "l"(cache_hint) : "memory");
}

__device__ __forceinline__ void tmem_alloc_fn_cg2(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
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
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
        :: "r"(a), "h"((uint16_t)0x3));
}

__device__ __forceinline__ uint64_t make_smem_desc_sbo_lbo(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61; // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_major(uint32_t M, uint32_t N, uint32_t a_major, uint32_t b_major) {
    uint32_t d = 0;
    d |= (1u << 4);    // c_format = FP32
    d |= (1u << 7);    // a_format = BF16
    d |= (1u << 10);   // b_format = BF16
    d |= (a_major << 15);   // a_major
    d |= (b_major << 16);   // b_major
    d |= ((N / 8) << 17);   // n_dim
    d |= ((M / 16) << 24);  // m_dim
    return d;
}

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

// -------------------------------------------------------------------------
// Main Kernel
// -------------------------------------------------------------------------

__global__ void mha_kernel(
    const __grid_constant__ CUtensorMap tma_q,
    const __grid_constant__ CUtensorMap tma_k,
    const __grid_constant__ CUtensorMap tma_v,
    __nv_bfloat16* out_O,
    float* out_LSE,
    uint32_t S)
{
    uint32_t bh = blockIdx.y;
    uint32_t cluster_idx = blockIdx.x / 2;
    uint32_t cta_rank = cluster_rank_fn();
    uint32_t m_start = cluster_idx * 128 + cta_rank * 64;
    
    if (m_start >= S) return;
    
    extern __shared__ __align__(1024) char smem[];
    __nv_bfloat16* smem_q0 = (__nv_bfloat16*)smem;
    __nv_bfloat16* smem_q1 = smem_q0 + 4096;
    __nv_bfloat16* smem_k0 = smem_q1 + 4096;
    __nv_bfloat16* smem_k1 = smem_k0 + 4096;
    __nv_bfloat16* smem_v0 = smem_k1 + 4096;
    __nv_bfloat16* smem_v1 = smem_v0 + 4096;
    __nv_bfloat16* smem_s  = smem_v1 + 4096;
    
    uint64_t* bar_load = (uint64_t*)(smem_s + 4096);
    uint64_t* bar_umma = bar_load + 1;
    float* row_max = (float*)(bar_umma + 1);
    float* row_sum = row_max + 64;
    uint32_t* tmem_base = (uint32_t*)(row_sum + 64);
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar_load, 1);
        init_smem_barrier_fn(bar_umma, 1);
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        tmem_alloc_fn_cg2(tmem_base, 256);
    }
    __syncthreads();
    
    uint32_t tmem_addr = *tmem_base;
    uint32_t tmem_q0 = tmem_addr + 0;
    uint32_t tmem_q1 = tmem_q0 + 256;
    uint32_t tmem_k0 = tmem_q1 + 256;
    uint32_t tmem_k1 = tmem_k0 + 128;
    uint32_t tmem_v0 = tmem_k1 + 128;
    uint32_t tmem_v1 = tmem_v0 + 128;
    uint32_t tmem_s  = tmem_v1 + 128;
    uint32_t tmem_p  = tmem_s + 64;
    uint32_t tmem_o  = tmem_p + 64;
    
    uint32_t load_phase = 0;
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_load, 32768); 
        tma_load_multicast_2d_fn(&tma_q, bar_load, smem_q0, 0, bh * S + m_start, 0x3);
        tma_load_multicast_2d_fn(&tma_q, bar_load, smem_q1, 64, bh * S + m_start, 0x3);
    }
    __syncthreads();
    mbarrier_wait_fn(bar_load, load_phase % 2);
    load_phase++;
    
    if (threadIdx.x < 64) {
        row_max[threadIdx.x] = -INFINITY;
        row_sum[threadIdx.x] = 0.0f;
    }
    
    float o_reg0[64];
    float o_reg1[64];
    #pragma unroll
    for (uint32_t c = 0; c < 64; ++c) {
        o_reg0[c] = 0.0f;
        o_reg1[c] = 0.0f;
    }
    
    uint32_t umma_phase = 0;
    uint32_t S_total = S;
    
    uint32_t block_idx = m_start / 64;
    
    for (uint32_t j_block = 0; j_block <= block_idx; ++j_block) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_load, 65536);
            tma_load_multicast_2d_fn(&tma_k, bar_load, smem_k0, 0, bh * S + j_block * 64, 0x3);
            tma_load_multicast_2d_fn(&tma_k, bar_load, smem_k1, 64, bh * S + j_block * 64, 0x3);
            tma_load_multicast_2d_fn(&tma_v, bar_load, smem_v0, 0, bh * S + j_block * 64, 0x3);
            tma_load_multicast_2d_fn(&tma_v, bar_load, smem_v1, 64, bh * S + j_block * 64, 0x3);
        }
        __syncthreads();
        mbarrier_wait_fn(bar_load, load_phase % 2);
        load_phase++;
        
        fence_proxy_async_fn();
        
        if (threadIdx.x == 0) {
            #pragma unroll
            for (uint32_t k = 0; k < 4; ++k) {
                uint64_t desc_q0 = make_smem_desc_sbo_lbo(smem_q0 + k * 16, 1, 1024);
                uint64_t desc_k0 = make_smem_desc_sbo_lbo(smem_k0 + k * 16, 1, 1024);
                uint32_t idesc = make_instr_desc_major(128, 64, 0, 0);
                uint32_t acc = (k == 0) ? 0 : 1;
                umma_f16_cg2_fn(tmem_p, desc_q0, desc_k0, idesc, acc);
            }
            #pragma unroll
            for (uint32_t k = 0; k < 4; ++k) {
                uint64_t desc_q1 = make_smem_desc_sbo_lbo(smem_q1 + k * 16, 1, 1024);
                uint64_t desc_k1 = make_smem_desc_sbo_lbo(smem_k1 + k * 16, 1, 1024);
                uint32_t idesc = make_instr_desc_major(128, 64, 0, 0);
                uint32_t acc = (k == 0) ? 0 : 1;
                umma_f16_cg2_fn(tmem_p, desc_q1, desc_k1, idesc, acc);
            }
            umma_commit_2sm_fn(bar_umma);
        }
        mbarrier_wait_fn(bar_umma, umma_phase % 2);
        umma_phase++;
        
        uint32_t tid = threadIdx.x;
        uint32_t local_row = tid % 64;
        uint32_t query_pos = m_start + local_row;
        
        float p_val[64];
        #pragma unroll
        for (uint32_t col = 0; col < 64; col += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_p + col));
            p_val[col] = __uint_as_float(r0);
            p_val[col+1] = __uint_as_float(r1);
            p_val[col+2] = __uint_as_float(r2);
            p_val[col+3] = __uint_as_float(r3);
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float m = -INFINITY;
        #pragma unroll
        for (uint32_t c = 0; c < 64; ++c) {
            float val = p_val[c];
            uint32_t key_pos = j_block * 64 + c;
            if (key_pos > query_pos || key_pos >= S_total) {
                val = -INFINITY;
            } else {
                val *= 0.08838834764f; // 1.0 / sqrt(128)
            }
            p_val[c] = val;
            if (val > m) m = val;
        }
        
        float prev_m = row_max[local_row];
        float m_new = (prev_m > m) ? prev_m : m;
        
        float s_old = __expf(prev_m - m_new);
        row_sum[local_row] *= s_old;
        
        #pragma unroll
        for (uint32_t c = 0; c < 64; ++c) {
            o_reg0[c] *= s_old;
            o_reg1[c] *= s_old;
        }
        
        float s_sum = 0.0f;
        #pragma unroll
        for (uint32_t c = 0; c < 64; ++c) {
            float s = __expf(p_val[c] - m_new);
            s_sum += s;
            
            uint32_t chunk_x = c / 8;
            uint32_t swizzled_chunk_x = chunk_x ^ (tid % 8);
            uint32_t c_swizzled = swizzled_chunk_x * 8 + (c % 8);
            smem_s[tid * 64 + c_swizzled] = __float2bfloat16(s);
        }
        row_sum[local_row] += s_sum;
        row_max[local_row] = m_new;
        
        __syncthreads();
        fence_proxy_async_fn(); 
        
        if (threadIdx.x == 0) {
            #pragma unroll
            for (uint32_t k = 0; k < 4; ++k) {
                uint64_t desc_s = make_smem_desc_sbo_lbo(smem_s + k * 16, 1, 1024);
                uint64_t desc_v0 = make_smem_desc_sbo_lbo(smem_v0 + k * 64, 8192, 1024);
                uint32_t idesc = make_instr_desc_major(128, 128, 0, 1);
                uint32_t acc_o0 = (j_block == 0 && k == 0) ? 0 : 1;
                umma_f16_cg2_fn(tmem_o, desc_s, desc_v0, idesc, acc_o0);
            }
            #pragma unroll
            for (uint32_t k = 0; k < 4; ++k) {
                uint64_t desc_s = make_smem_desc_sbo_lbo(smem_s + k * 16, 1, 1024);
                uint64_t desc_v1 = make_smem_desc_sbo_lbo(smem_v1 + k * 64, 8192, 1024);
                uint32_t idesc = make_instr_desc_major(128, 128, 0, 1);
                uint32_t acc_o1 = (j_block == 0 && k == 0) ? 0 : 1;
                umma_f16_cg2_fn(tmem_o + 64, desc_s, desc_v1, idesc, acc_o1);
            }
            umma_commit_2sm_fn(bar_umma);
        }
        mbarrier_wait_fn(bar_umma, umma_phase % 2);
        umma_phase++;
        
        #pragma unroll
        for (uint32_t col = 0; col < 64; col += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_o + col));
            o_reg0[0 + col] += __uint_as_float(r0);
            o_reg0[1 + col] += __uint_as_float(r1);
            o_reg0[2 + col] += __uint_as_float(r2);
            o_reg0[3 + col] += __uint_as_float(r3);
        }
        #pragma unroll
        for (uint32_t col = 0; col < 64; col += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_o + 64 + col));
            o_reg1[0 + col] += __uint_as_float(r0);
            o_reg1[1 + col] += __uint_as_float(r1);
            o_reg1[2 + col] += __uint_as_float(r2);
            o_reg1[3 + col] += __uint_as_float(r3);
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    }
    
    __syncthreads();
    
    // Direct Coalesced Vectorized Global Writes
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t num_steps = 64 / 4; // 16 steps
    
    __nv_bfloat16* D = out_O + (uint64_t)bh * S * 128;
    uint32_t M = S;
    uint32_t N = 128;
    uint32_t m_block = cluster_idx * 2 + (cta_rank / 2); // approximate block mapping relative to original S tiling
    uint32_t n_block = 0;
    uint32_t BM = 64;
    uint32_t BN = 128;
    
    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 4 + warp_id;
        if (row >= BM) continue;
        uint32_t global_row = m_block * BM + row;
        uint32_t col_start = lane_id * 4;
        
        if (global_row < M) {
            float s_final = row_sum[row];
            
            float f0_0 = o_reg0[step * 4 * 4 + lane_id * 4 + 0];
            float f1_0 = o_reg0[step * 4 * 4 + lane_id * 4 + 1];
            float f2_0 = o_reg0[step * 4 * 4 + lane_id * 4 + 2];
            float f3_0 = o_reg0[step * 4 * 4 + lane_id * 4 + 3];
            __nv_bfloat16 out0 = __float2bfloat16(f0_0 / s_final);
            __nv_bfloat16 out1 = __float2bfloat16(f1_0 / s_final);
            __nv_bfloat16 out2 = __float2bfloat16(f2_0 / s_final);
            __nv_bfloat16 out3 = __float2bfloat16(f3_0 / s_final);
            uint2 out_vec0 = *reinterpret_cast<uint2*>(&out0);
            
            uint32_t global_col0 = n_block * BN + col_start;
            if (global_col0 + 3 < N) {
                *reinterpret_cast<uint2*>(D + (uint64_t)global_row * N + global_col0) = out_vec0;
            }
            
            float f0_1 = o_reg1[step * 4 * 4 + lane_id * 4 + 0];
            float f1_1 = o_reg1[step * 4 * 4 + lane_id * 4 + 1];
            float f2_1 = o_reg1[step * 4 * 4 + lane_id * 4 + 2];
            float f3_1 = o_reg1[step * 4 * 4 + lane_id * 4 + 3];
            __nv_bfloat16 out4 = __float2bfloat16(f0_1 / s_final);
            __nv_bfloat16 out5 = __float2bfloat16(f1_1 / s_final);
            __nv_bfloat16 out6 = __float2bfloat16(f2_1 / s_final);
            __nv_bfloat16 out7 = __float2bfloat16(f3_1 / s_final);
            uint2 out_vec1 = *reinterpret_cast<uint2*>(&out4);
            
            uint32_t global_col1 = n_block * BN + 64 + col_start;
            if (global_col1 + 3 < N) {
                *reinterpret_cast<uint2*>(D + (uint64_t)global_row * N + global_col1) = out_vec1;
            }
        }
    }
    
    if (threadIdx.x < 64) {
        if (m_start + threadIdx.x < S_total) {
            out_LSE[bh * S + m_start + threadIdx.x] = row_max[threadIdx.x] + logf(row_sum[threadIdx.x]);
        }
    }
}

// -------------------------------------------------------------------------
// Host-Side Setup and FFI Binding
// -------------------------------------------------------------------------

namespace tvm_ffi_mha {

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

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    uint32_t B = Q.size(0);
    uint32_t H = Q.size(1);
    uint32_t S = Q.size(2);
    uint32_t D = Q.size(3); 
    
    uint32_t total_S = B * H * S;
    
    CUtensorMap tma_q, tma_k, tma_v;
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_q, Q.data_ptr(), D, total_S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_k, K.data_ptr(), D, total_S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_v, V.data_ptr(), D, total_S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
    uint32_t smem_size = 73728; 
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    dim3 grid((S + 63) / 64, B * H);
    dim3 block(128, 1, 1);
    
    CUDA_CHECK(cudaFuncSetAttribute(mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    CUDA_CHECK(cudaFuncSetAttribute(mha_kernel, cudaFuncAttributeClusterSize, 2));
    
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
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, mha_kernel,
        tma_q, tma_k, tma_v, 
        static_cast<__nv_bfloat16*>(O.data_ptr()), 
        static_cast<float*>(LSE.data_ptr()), 
        S
    ));
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_mha