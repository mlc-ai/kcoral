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

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void umma_f16_cg1_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_1sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64 [%0];"
        :: "r"(a));
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= (0u << 15);
    d |= (0u << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
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
    uint32_t m_start = blockIdx.x * 64;
    
    if (m_start >= S) return;
    
    extern __shared__ __align__(128) char smem[];
    __nv_bfloat16* smem_q = (__nv_bfloat16*)smem;
    __nv_bfloat16* smem_k = smem_q + 64 * 128;
    __nv_bfloat16* smem_v = smem_k + 64 * 128;
    __nv_bfloat16* smem_o = smem_v + 64 * 128;
    __nv_bfloat16* smem_s = smem_k;
    
    uint64_t* bar_q = (uint64_t*)(smem_o + 64 * 128);
    uint64_t* bar_k = bar_q + 1;
    uint64_t* bar_v = bar_k + 1;
    uint64_t* bar_umma = bar_v + 1;
    float* row_max = (float*)(bar_umma + 1);
    float* row_sum = row_max + 64;
    uint32_t* tmem_base = (uint32_t*)(row_sum + 64);
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar_q, 1);
        init_smem_barrier_fn(bar_k, 1);
        init_smem_barrier_fn(bar_v, 1);
        init_smem_barrier_fn(bar_umma, 1);
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        tmem_alloc_fn(tmem_base, 256);
    }
    __syncthreads();
    uint32_t tmem_addr = *tmem_base;
    uint32_t tmem_p = tmem_addr + 0;
    uint32_t tmem_o = tmem_addr + 64;
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_q, 16384);
        tma_load_2d_fn(&tma_q, bar_q, smem_q, 0, bh * S + m_start);
        tma_load_2d_fn(&tma_q, bar_q, smem_q + 4096, 64, bh * S + m_start);
    }
    mbarrier_wait_fn(bar_q, 0);
    
    float o_reg[128];
    for (uint32_t c = 0; c < 128; ++c) o_reg[c] = 0.0f;
    
    if (threadIdx.x < 64) {
        row_max[threadIdx.x] = -INFINITY;
        row_sum[threadIdx.x] = 0.0f;
    }
    
    uint32_t max_phase = 0;
    
    for (uint32_t j_block = 0; j_block <= m_start / 64; ++j_block) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_k, 16384);
            tma_load_2d_fn(&tma_k, bar_k, smem_k, 0, bh * S + j_block * 64);
            tma_load_2d_fn(&tma_k, bar_k, smem_k + 4096, 64, bh * S + j_block * 64);
            
            mbarrier_arrive_and_expect_tx_fn(bar_v, 16384);
            tma_load_2d_fn(&tma_v, bar_v, smem_v, 0, bh * S + j_block * 64);
            tma_load_2d_fn(&tma_v, bar_v, smem_v + 4096, 64, bh * S + j_block * 64);
        }
        mbarrier_wait_fn(bar_k, max_phase % 2);
        mbarrier_wait_fn(bar_v, max_phase % 2);
        max_phase++;
        
        if (threadIdx.x == 0) {
            for (uint32_t k = 0; k < 4; ++k) {
                uint64_t desc_q = make_smem_desc_sm100_fn(smem_q + k * 32, 1, 1024);
                uint64_t desc_k = make_smem_desc_sm100_fn(smem_k + k * 32, 1, 1024);
                uint32_t idesc = make_instr_desc_fn(64, 64);
                uint32_t acc = (k == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_p, desc_q, desc_k, idesc, acc);
            }
            umma_commit_1sm_fn(bar_umma);
        }
        mbarrier_wait_fn(bar_umma, max_phase % 2);
        max_phase++;
        
        float p_regs[4][16];
        if (threadIdx.x < 64) {
            uint32_t i = threadIdx.x;
            #pragma unroll
            for (uint32_t col = 0; col < 64; col += 4) {
                uint32_t r0, r1, r2, r3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_p + col));
                p_regs[0][(col/4)*4 + 0] = __uint_as_float(r0);
                p_regs[0][(col/4)*4 + 1] = __uint_as_float(r1);
                p_regs[0][(col/4)*4 + 2] = __uint_as_float(r2);
                p_regs[0][(col/4)*4 + 3] = __uint_as_float(r3);
            }
            tmem_load_fence_fn();
        }
        
        if (threadIdx.x < 64) {
            uint32_t i = threadIdx.x;
            float m = -INFINITY;
            #pragma unroll
            for (uint32_t c = 0; c < 64; ++c) {
                float val = p_regs[0][c];
                if (j_block * 64 + c > m_start + i) {
                    val = -INFINITY;
                } else {
                    val *= 0.08838834764f; // 1.0 / sqrt(128)
                }
                p_regs[0][c] = val;
                if (val > m) m = val;
            }
            
            float prev_m = row_max[i];
            float m_new = (prev_m > m) ? prev_m : m;
            
            float s_old = __expf(prev_m - m_new);
            row_sum[i] *= s_old;
            
            #pragma unroll
            for (uint32_t c = 0; c < 128; ++c) {
                o_reg[c] *= s_old;
            }
            
            float s_sum = 0.0f;
            #pragma unroll
            for (uint32_t c = 0; c < 64; ++c) {
                float s = __expf(p_regs[0][c] - m_new);
                s_sum += s;
                
                uint32_t c_swizzled = ((c / 8) ^ (i % 8)) * 8 + (c % 8);
                smem_s[i * 128 + c_swizzled] = __float2bfloat16(s);
            }
            row_sum[i] += s_sum;
            row_max[i] = m_new;
        }
        
        __syncthreads();
        fence_proxy_async_fn();
        
        if (threadIdx.x == 0) {
            for (uint32_t k = 0; k < 4; ++k) {
                uint64_t desc_s = make_smem_desc_sm100_fn(smem_s + k * 32, 1, 1024);
                uint64_t desc_v = make_smem_desc_sm100_fn(smem_v + k * 32, 1, 1024);
                uint32_t idesc = make_instr_desc_fn(64, 128);
                uint32_t acc = (k == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_o, desc_s, desc_v, idesc, acc);
            }
            umma_commit_1sm_fn(bar_umma);
        }
        mbarrier_wait_fn(bar_umma, max_phase % 2);
        max_phase++;
        
        if (threadIdx.x < 64) {
            uint32_t i = threadIdx.x;
            #pragma unroll
            for (uint32_t col = 0; col < 64; col += 4) {
                uint32_t r0, r1, r2, r3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_o + col));
                float f0 = __uint_as_float(r0);
                float f1 = __uint_as_float(r1);
                float f2 = __uint_as_float(r2);
                float f3 = __uint_as_float(r3);
                o_reg[0 + col] += f0;
                o_reg[1 + col] += f1;
                o_reg[2 + col] += f2;
                o_reg[3 + col] += f3;
            }
            #pragma unroll
            for (uint32_t col = 64; col < 128; col += 4) {
                uint32_t r0, r1, r2, r3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_o + col));
                float f0 = __uint_as_float(r0);
                float f1 = __uint_as_float(r1);
                float f2 = __uint_as_float(r2);
                float f3 = __uint_as_float(r3);
                o_reg[64 + col - 64] += f0;
                o_reg[65 + col - 64] += f1;
                o_reg[66 + col - 64] += f2;
                o_reg[67 + col - 64] += f3;
            }
            tmem_load_fence_fn();
        }
    }
    
    __syncthreads();
    
    if (threadIdx.x < 64) {
        uint32_t i = threadIdx.x;
        float s_final = row_sum[i];
        
        for (uint32_t c = 0; c < 128; ++c) {
            float o_val = o_reg[c] / s_final;
            uint32_t c_swizzled = ((c / 8) ^ (i % 8)) * 8 + (c % 8);
            smem_o[i * 128 + c_swizzled] = __float2bfloat16(o_val);
        }
        
        if (m_start + i < S) {
            out_LSE[bh * S + m_start + i] = row_max[i] + logf(s_final);
        }
    }
    
    __syncthreads();
    
    for (uint32_t col = 0; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(col));
        tmem_load_fence_fn();
        
        uint32_t tid = threadIdx.x;
        uint32_t base = tid * 128 + col;
        smem_o[base + 0] = __float2bfloat16(__uint_as_float(r0));
        smem_o[base + 1] = __float2bfloat16(__uint_as_float(r1));
        smem_o[base + 2] = __float2bfloat16(__uint_as_float(r2));
        smem_o[base + 3] = __float2bfloat16(__uint_as_float(r3));
    }
    __syncthreads();
    
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t num_steps = (64 + 3) / 4;
    
    __nv_bfloat16* D = out_O + (uint64_t)bh * S * 128;
    uint32_t M = S;
    uint32_t N = 128;
    uint32_t m_block = blockIdx.x;
    uint32_t n_block = 0;
    uint32_t BM = 64;
    uint32_t BN = 128;

    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 4 + warp_id;
        if (row >= BM) continue;
        uint32_t global_row = m_block * BM + row;
        uint32_t col_start = lane_id * 4;
        uint32_t global_col = n_block * BN + col_start;
        
        if (global_row < M && global_col + 3 < N) {
            uint2 data = *reinterpret_cast<uint2*>(&smem_o[row * BN + col_start]);
            *reinterpret_cast<uint2*>(D + (uint64_t)global_row * N + global_col) = data;
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
    mha_kernel<<<grid, block, smem_size, stream>>>(
        tma_q, tma_k, tma_v, 
        static_cast<__nv_bfloat16*>(O.data_ptr()), 
        static_cast<float*>(LSE.data_ptr()), 
        S
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_mha