#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
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
                _e, __FILE__, __LINE__);                           \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_example_cuda {

__device__ __forceinline__ uint32_t pack_bf16_fn(uint32_t fp32_a, uint32_t fp32_b) {
    __nv_bfloat16 a = __float2bfloat16(__uint_as_float(fp32_a));
    __nv_bfloat16 b = __float2bfloat16(__uint_as_float(fp32_b));
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};"
        : "=r"(result)
        : "h"(*reinterpret_cast<uint16_t*>(&a)),
          "h"(*reinterpret_cast<uint16_t*>(&b)));
    return result;
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tcgen05_ld_4x(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t taddr = ((warp_id * 32) << 16) | col;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*r0), "=r"(*r1), "=r"(*r2), "=r"(*r3) : "r"(taddr) : "memory");
}

__device__ __forceinline__ void tcgen05_st_4x(uint32_t col, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t taddr = ((warp_id * 32) << 16) | col;
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
                 :: "r"(taddr), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
}

__device__ __forceinline__ void tcgen05_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_wait_st() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_before_fn() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
                 :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_fn(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
                 :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])) : "memory");
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

__device__ __forceinline__ void fence_proxy_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_none_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16; 
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32; 
    d |= (uint64_t)1 << 46; // version = 1
    d |= (uint64_t)0 << 61; // SWIZZLE_NONE
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    // FP32
    d |= (1u << 7);    // BF16
    d |= (1u << 10);   // BF16
    d |= (0u << 15);   // a_major = 0 (K-Major)
    d |= (0u << 16);   // b_major = 0 (K-Major)
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ void umma_f16_fn(
    uint32_t tmem_d, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_d), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_f16_tmem_a_fn(
    uint32_t tmem_d, uint32_t tmem_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}\n"
        :: "r"(tmem_d), "r"(tmem_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_1sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(a));
}

__device__ __forceinline__ void tmem_epilogue_coalesced_4w_fn(
    __nv_bfloat16* D_global, __nv_bfloat16* smem_out,
    uint32_t S_dim, uint32_t D_dim, uint32_t m_block, uint32_t n_block,
    uint32_t BM, uint32_t BN, uint32_t O_tmem_base) {
    
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_base = warp_id * 32;
    for (uint32_t col = 0; col < BN; col += 4) {
        uint32_t r0, r1, r2, r3;
        uint32_t taddr = (lane_base << 16) | (O_tmem_base + col);
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(taddr));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t base_idx = threadIdx.x * BN + col;
        smem_out[base_idx + 0] = __float2bfloat16(__uint_as_float(r0));
        smem_out[base_idx + 1] = __float2bfloat16(__uint_as_float(r1));
        smem_out[base_idx + 2] = __float2bfloat16(__uint_as_float(r2));
        smem_out[base_idx + 3] = __float2bfloat16(__uint_as_float(r3));
    }
    __syncthreads();
    
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t num_steps = (BM + 3) / 4;
    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 4 + warp_id;
        if (row >= BM) continue;
        uint32_t global_row = m_block * BM + row;
        uint32_t col_start = lane_id * 4;
        uint32_t global_col = n_block * BN + col_start;
        if (global_row < S_dim && global_col + 3 < D_dim) {
            uint2 data = *reinterpret_cast<uint2*>(&smem_out[row * BN + col_start]);
            *reinterpret_cast<uint2*>(D_global + (uint64_t)global_row * D_dim + global_col) = data;
        }
    }
}

struct SharedStorage {
    alignas(16) uint64_t bar_Q;
    alignas(16) uint64_t bar_K;
    alignas(16) uint64_t bar_V;
    alignas(16) uint64_t bar_U;
    alignas(16) uint32_t tmem_base;
    alignas(128) __nv_bfloat16 Q[128 * 128];
    alignas(128) __nv_bfloat16 K[128 * 128];
    alignas(128) __nv_bfloat16 V[128 * 128];
    alignas(128) __nv_bfloat16 O[128 * 128];
};

extern __shared__ alignas(128) char smem_buf[];

__global__ void mha_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* __restrict__ O_global,
    float* __restrict__ LSE_global,
    int S, int D, int B, int H) 
{
    SharedStorage& shared = *reinterpret_cast<SharedStorage*>(smem_buf);
    
    int m = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    int batch_head_idx = b * H + h;
    
    if (threadIdx.x < 32) {
        tmem_alloc_fn(&shared.tmem_base, 512);
    }
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&shared.bar_Q, 128);
        init_smem_barrier_fn(&shared.bar_K, 128);
        init_smem_barrier_fn(&shared.bar_V, 128);
        init_smem_barrier_fn(&shared.bar_U, 129);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    
    uint32_t tmem_base = shared.tmem_base;
    uint32_t O_tmem_base = tmem_base;
    uint32_t S_tmem_base = tmem_base + 128;
    uint32_t P_tmem_base = tmem_base + 256;
    
    uint32_t tx_bytes = 128 * 128 * sizeof(__nv_bfloat16);
    
    uint32_t phase_Q = 0, phase_K = 0, phase_V = 0, phase_U = 0;
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&shared.bar_Q, tx_bytes);
        tma_load_2d_fn(&tma_Q, &shared.bar_Q, shared.Q, 0, batch_head_idx * S + m * 128);
    } else {
        mbarrier_arrive_fn(&shared.bar_Q);
    }
    mbarrier_wait_fn(&shared.bar_Q, phase_Q);
    phase_Q ^= 1;
    fence_proxy_async_shared_fn();
    __syncthreads();
    
    for (int i = 0; i < 128; i += 4) {
        tcgen05_st_4x(O_tmem_base + i, 0, 0, 0, 0);
    }
    tcgen05_wait_st();
    tcgen05_fence_before_fn();
    __syncthreads();
    tcgen05_fence_after_fn();
    
    bool is_valid_q = (m * 128 + threadIdx.x) < S;
    float m_prev = is_valid_q ? -INFINITY : 0.0f;
    float l_prev = 0.0f;
    
    for (int n = 0; n < (S + 127) / 128; ++n) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&shared.bar_K, tx_bytes);
            tma_load_2d_fn(&tma_K, &shared.bar_K, shared.K, 0, batch_head_idx * S + n * 128);
            
            mbarrier_arrive_and_expect_tx_fn(&shared.bar_V, tx_bytes);
            tma_load_2d_fn(&tma_V, &shared.bar_V, shared.V, 0, batch_head_idx * S + n * 128);
        } else {
            mbarrier_arrive_fn(&shared.bar_K);
            mbarrier_arrive_fn(&shared.bar_V);
        }
        mbarrier_wait_fn(&shared.bar_K, phase_K);
        mbarrier_wait_fn(&shared.bar_V, phase_V);
        phase_K ^= 1;
        phase_V ^= 1;
        fence_proxy_async_shared_fn();
        __syncthreads();
        
        uint32_t idesc_S = make_instr_desc_fn(128, 128);
        for (int k_step = 0; k_step < 8; ++k_step) {
            uint64_t desc_q = make_smem_desc_none_fn(shared.Q + k_step * 16, 2048, 128);
            uint64_t desc_k = make_smem_desc_none_fn(shared.K + k_step * 16, 2048, 128);
            uint32_t accum = (k_step == 0) ? 0 : 1;
            if (threadIdx.x == 0) {
                umma_f16_fn(S_tmem_base, desc_q, desc_k, idesc_S, accum);
            }
        }
        
        if (threadIdx.x == 0) umma_commit_1sm_fn(&shared.bar_U);
        mbarrier_arrive_fn(&shared.bar_U);
        mbarrier_wait_fn(&shared.bar_U, phase_U);
        phase_U ^= 1;
        tcgen05_fence_after_fn();
        __syncthreads();
        
        float m_curr = m_prev;
        if (!is_valid_q) m_curr = 0.0f;
        
        float S_row[128];
        float scale = 0.0883883476f; 
        
        for (int chunk = 0; chunk < 4; ++chunk) {
            int base = chunk * 32;
            for (int i = 0; i < 32; i += 4) {
                tcgen05_ld_4x(S_tmem_base + base + i, (uint32_t*)&S_row[base+i], (uint32_t*)&S_row[base+i+1], (uint32_t*)&S_row[base+i+2], (uint32_t*)&S_row[base+i+3]);
            }
            tcgen05_wait_ld();
            for (int i = 0; i < 32; ++i) {
                int key_idx = n * 128 + base + i;
                if (key_idx >= S || !is_valid_q) {
                    S_row[base+i] = -INFINITY;
                } else {
                    S_row[base+i] *= scale;
                }
                m_curr = fmaxf(m_curr, S_row[base+i]);
            }
        }
        
        float diff = m_prev - m_curr;
        float exp_diff = 1.0f;
        if (m_prev != -INFINITY) {
            exp_diff = exp2f(diff * 1.4426950408889634f);
        }
        
        float l_curr = 0.0f;
        uint32_t P_row[64];
        l_prev = l_prev * exp_diff;
        l_curr = l_prev;
        
        for (int i = 0; i < 128; i += 2) {
            float p0 = exp2f((S_row[i] - m_curr) * 1.4426950408889634f);
            float p1 = exp2f((S_row[i+1] - m_curr) * 1.4426950408889634f);
            l_curr += p0 + p1;
            P_row[i/2] = pack_bf16_fn(*(uint32_t*)&p0, *(uint32_t*)&p1);
        }
        
        for (int chunk = 0; chunk < 2; ++chunk) {
            int base = chunk * 32;
            for (int i = 0; i < 32; i += 4) {
                tcgen05_st_4x(P_tmem_base + base + i, P_row[base+i], P_row[base+i+1], P_row[base+i+2], P_row[base+i+3]);
            }
        }
        
        for (int i = 0; i < 128; i += 4) {
            uint32_t c0, c1, c2, c3;
            tcgen05_ld_4x(O_tmem_base + i, &c0, &c1, &c2, &c3);
            tcgen05_wait_ld();
            
            float f0 = *(float*)&c0 * exp_diff;
            float f1 = *(float*)&c1 * exp_diff;
            float f2 = *(float*)&c2 * exp_diff;
            float f3 = *(float*)&c3 * exp_diff;
            
            tcgen05_st_4x(O_tmem_base + i, *(uint32_t*)&f0, *(uint32_t*)&f1, *(uint32_t*)&f2, *(uint32_t*)&f3);
        }
        tcgen05_wait_st();
        tcgen05_fence_before_fn();
        __syncthreads();
        tcgen05_fence_after_fn();
        
        m_prev = m_curr;
        l_prev = l_curr;
        
        uint32_t idesc_O = make_instr_desc_fn(128, 128);
        idesc_O |= (1u << 16); // V is MN-Major
        for (int k_step = 0; k_step < 8; ++k_step) {
            uint32_t tmem_p = P_tmem_base + k_step * 8;
            uint64_t desc_v = make_smem_desc_none_fn(shared.V + k_step * 16 * 128, 128, 2048);
            if (threadIdx.x == 0) {
                umma_f16_tmem_a_fn(O_tmem_base, tmem_p, desc_v, idesc_O, 1);
            }
        }
        
        if (threadIdx.x == 0) umma_commit_1sm_fn(&shared.bar_U);
        mbarrier_arrive_fn(&shared.bar_U);
        mbarrier_wait_fn(&shared.bar_U, phase_U);
        phase_U ^= 1;
        tcgen05_fence_after_fn();
        __syncthreads();
    }
    
    float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
    for (int i = 0; i < 128; i += 4) {
        uint32_t c0, c1, c2, c3;
        tcgen05_ld_4x(O_tmem_base + i, &c0, &c1, &c2, &c3);
        tcgen05_wait_ld();
        
        float f0 = *(float*)&c0 * inv_l;
        float f1 = *(float*)&c1 * inv_l;
        float f2 = *(float*)&c2 * inv_l;
        float f3 = *(float*)&c3 * inv_l;
        
        tcgen05_st_4x(O_tmem_base + i, *(uint32_t*)&f0, *(uint32_t*)&f1, *(uint32_t*)&f2, *(uint32_t*)&f3);
    }
    tcgen05_wait_st();
    tcgen05_fence_before_fn();
    __syncthreads();
    tcgen05_fence_after_fn();
    
    __nv_bfloat16* O_ptr = O_global + batch_head_idx * S * D;
    tmem_epilogue_coalesced_4w_fn(O_ptr, shared.O, S, D, m, 0, 128, 128, O_tmem_base);
    
    if (threadIdx.x < 128) {
        int seq_idx = m * 128 + threadIdx.x;
        if (seq_idx < S && is_valid_q) {
            float lse = m_prev + logf(l_prev);
            LSE_global[batch_head_idx * S + seq_idx] = lse;
        }
    }
    
    __syncthreads();
    if (threadIdx.x < 32) {
        tmem_dealloc_fn(tmem_base, 512);
    }
}

CUresult create_tma_2d_descriptor_none(CUtensorMap* d, void* globalAddress, uint64_t total_S) {
    cuuint64_t globalDim[2] = {128, total_S};
    cuuint64_t globalStrides[1] = {256}; 
    cuuint32_t boxDim[2] = {128, 128}; 
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, globalAddress, globalDim,
        globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);

    CUtensorMap tma_Q, tma_K, tma_V;
    CU_CHECK(create_tma_2d_descriptor_none(&tma_Q, Q.data_ptr(), B * H * S));
    CU_CHECK(create_tma_2d_descriptor_none(&tma_K, K.data_ptr(), B * H * S));
    CU_CHECK(create_tma_2d_descriptor_none(&tma_V, V.data_ptr(), B * H * S));

    dim3 grid((S + 127) / 128, H, B);
    dim3 block(128, 1, 1);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUDA_CHECK(cudaFuncSetAttribute(mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedStorage)));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = sizeof(SharedStorage);
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 1;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    __nv_bfloat16* O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_ptr = static_cast<float*>(LSE.data_ptr());
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, mha_kernel, tma_Q, tma_K, tma_V, O_ptr, LSE_ptr, (int)S, (int)D, (int)B, (int)H));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda