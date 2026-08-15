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

namespace tvm_ffi_example_cuda {

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
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

__device__ __forceinline__ void tma_load_4d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

__device__ __forceinline__ void tmem_alloc_1sm_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_1sm_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
        :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_no_swizzle(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)0 << 61; // SWIZZLE_NONE
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, bool trans_a, bool trans_b) {
    uint32_t d = 0;
    d |= (1u << 4);    // c_format = FP32
    d |= (1u << 7);    // a_format = BF16
    d |= (1u << 10);   // b_format = BF16
    d |= ((trans_a ? 1u : 0u) << 15);
    d |= ((trans_b ? 1u : 0u) << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ void tcgen05_mma_f16(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void tcgen05_mma_f16_tmem(
    uint32_t tmem_c, uint32_t tmem_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}\n"
        :: "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void tcgen05_commit_1sm(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(a));
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

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

struct SharedMemory {
    alignas(1024) uint8_t smem_Q[32768];
    alignas(1024) uint8_t smem_K[32768];
    alignas(1024) uint8_t smem_V[32768];
    alignas(8) uint64_t mbar_k;
    alignas(8) uint64_t mbar_mma;
    alignas(8) uint64_t mbar_q;
    alignas(8) uint32_t tmem_addr;
};

__global__ void mha_fwd_sm100_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* __restrict__ O_ptr,
    float* __restrict__ LSE_ptr,
    int S_len, int D, int H) 
{
    setmaxnreg_inc_sync_fn<256>();

    int S_q = blockIdx.x * 128;
    int head_idx = blockIdx.y % H;
    int batch_idx = blockIdx.y / H;

    extern __shared__ SharedMemory smem;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&smem.mbar_k, 1);
        init_smem_barrier_fn(&smem.mbar_q, 1);
        init_smem_barrier_fn(&smem.mbar_mma, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        tmem_alloc_1sm_fn(&smem.tmem_addr, 256);
    }
    __syncthreads();

    uint32_t tmem_s = smem.tmem_addr;
    uint32_t tmem_o = smem.tmem_addr + 128;
    uint32_t tmem_p = tmem_s;

    uint32_t idesc_qk = make_instr_desc_fn(128, 128, false, false);
    uint32_t idesc_pv = make_instr_desc_fn(128, 128, false, true);
    
    int parity_q = 0;
    int parity_k = 0;
    int parity_mma = 0;

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&smem.mbar_q, 32768);
        tma_load_4d_fn(&tma_Q, &smem.mbar_q, smem.smem_Q, 0, S_q, head_idx, batch_idx);
    }
    mbarrier_wait_fn(&smem.mbar_q, parity_q);
    parity_q ^= 1;

    float m_i = -INFINITY;
    float d_i = 0.0f;
    bool valid_q = (S_q + threadIdx.x < S_len);

    uint8_t* sq = smem.smem_Q;
    uint8_t* sk = smem.smem_K;
    uint8_t* sv = smem.smem_V;

    int num_k_tiles = (S_len + 127) / 128;
    for (int k_idx = 0; k_idx < num_k_tiles; ++k_idx) {
        int S_k = k_idx * 128;
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&smem.mbar_k, 32768 * 2);
            tma_load_4d_fn(&tma_K, &smem.mbar_k, sk, 0, S_k, head_idx, batch_idx);
            tma_load_4d_fn(&tma_V, &smem.mbar_k, sv, 0, S_k, head_idx, batch_idx);
        }
        mbarrier_wait_fn(&smem.mbar_k, parity_k);
        parity_k ^= 1;

        if (threadIdx.x == 0) {
            for(int i = 0; i < 8; ++i) { 
                uint32_t accum = (i == 0) ? 0 : 1;
                uint64_t dq = make_smem_desc_sm100_no_swizzle(sq + i * 32, 16, 2048); 
                uint64_t dk = make_smem_desc_sm100_no_swizzle(sk + i * 32, 16, 2048); 
                tcgen05_mma_f16(tmem_s, dq, dk, idesc_qk, accum);
            }
            tcgen05_commit_1sm(&smem.mbar_mma);
        }
        mbarrier_wait_fn(&smem.mbar_mma, parity_mma);
        parity_mma ^= 1;

        float m_prev = m_i;
        float m_curr = m_prev;
        float s_vals[128];
        
        #pragma unroll
        for (int i = 0; i < 128; i += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_s + i));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float f0 = __uint_as_float(r0) * 0.08838834764f; // 1 / sqrt(128)
            float f1 = __uint_as_float(r1) * 0.08838834764f;
            float f2 = __uint_as_float(r2) * 0.08838834764f;
            float f3 = __uint_as_float(r3) * 0.08838834764f;
            
            if (!valid_q || S_k + i >= S_len) f0 = -INFINITY;
            if (!valid_q || S_k + i + 1 >= S_len) f1 = -INFINITY;
            if (!valid_q || S_k + i + 2 >= S_len) f2 = -INFINITY;
            if (!valid_q || S_k + i + 3 >= S_len) f3 = -INFINITY;
            
            s_vals[i] = f0; s_vals[i+1] = f1; s_vals[i+2] = f2; s_vals[i+3] = f3;
            if (valid_q) {
                m_curr = fmaxf(m_curr, f0);
                m_curr = fmaxf(m_curr, f1);
                m_curr = fmaxf(m_curr, f2);
                m_curr = fmaxf(m_curr, f3);
            }
        }
        
        float rescale = 1.0f;
        if (valid_q && k_idx > 0 && m_prev != m_curr && m_prev != -INFINITY) {
            rescale = fast_exp2f_fn((m_prev - m_curr) * 1.4426950408889634f); // * log2(e)
            for (int i = 0; i < 128; i += 4) {
                uint32_t o0, o1, o2, o3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                             : "=r"(o0),"=r"(o1),"=r"(o2),"=r"(o3) : "r"(tmem_o + i));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float f0 = __uint_as_float(o0) * rescale;
                float f1 = __uint_as_float(o1) * rescale;
                float f2 = __uint_as_float(o2) * rescale;
                float f3 = __uint_as_float(o3) * rescale;
                o0 = __float_as_uint(f0);
                o1 = __float_as_uint(f1);
                o2 = __float_as_uint(f2);
                o3 = __float_as_uint(f3);
                asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
                             :: "r"(tmem_o + i), "r"(o0),"r"(o1),"r"(o2),"r"(o3) : "memory");
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        }
        
        float d_curr = d_i * rescale;
        for (int i = 0; i < 128; ++i) {
            if (!valid_q || S_k + i >= S_len) {
                s_vals[i] = 0.0f;
            } else {
                s_vals[i] = fast_exp2f_fn((s_vals[i] - m_curr) * 1.4426950408889634f);
                d_curr += s_vals[i];
            }
        }
        
        for (int i = 0; i < 128; i += 4) {
            uint32_t p0 = pack_bf16_fn(__float_as_uint(s_vals[i]), __float_as_uint(s_vals[i+1]));
            uint32_t p1 = pack_bf16_fn(__float_as_uint(s_vals[i+2]), __float_as_uint(s_vals[i+3]));
            asm volatile("tcgen05.st.sync.aligned.32x32b.x2.b32 [%0], {%1,%2};"
                         :: "r"(tmem_p + i/2), "r"(p0), "r"(p1) : "memory");
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        
        m_i = m_curr;
        d_i = d_curr;
        
        if (threadIdx.x == 0) {
            for(int i = 0; i < 8; ++i) {
                uint32_t accum = (k_idx == 0 && i == 0) ? 0 : 1;
                uint32_t p_tm = tmem_p + i * 8; 
                uint64_t dv = make_smem_desc_sm100_no_swizzle(sv + i * 4096, 256, 16); 
                tcgen05_mma_f16_tmem(tmem_o, p_tm, dv, idesc_pv, accum);
            }
            tcgen05_commit_1sm(&smem.mbar_mma);
        }
        mbarrier_wait_fn(&smem.mbar_mma, parity_mma);
        parity_mma ^= 1;
    }

    if (valid_q) {
        float lse_val = m_i + __logf(d_i);
        LSE_ptr[batch_idx * H * S_len + head_idx * S_len + S_q + threadIdx.x] = lse_val;
    }

    float inv_d = valid_q ? (1.0f / d_i) : 0.0f;
    __nv_bfloat16* smem_out = (__nv_bfloat16*)sq;
    
    for (uint32_t col = 0; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_o + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t base = threadIdx.x * 128 + col;
        float f0 = __uint_as_float(r0) * inv_d;
        float f1 = __uint_as_float(r1) * inv_d;
        float f2 = __uint_as_float(r2) * inv_d;
        float f3 = __uint_as_float(r3) * inv_d;
        smem_out[base + 0] = __float2bfloat16(f0);
        smem_out[base + 1] = __float2bfloat16(f1);
        smem_out[base + 2] = __float2bfloat16(f2);
        smem_out[base + 3] = __float2bfloat16(f3);
    }
    __syncthreads();

    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t num_steps = 32;
    __nv_bfloat16* o_head_ptr = O_ptr + (batch_idx * H + head_idx) * S_len * D;
    
    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 4 + warp_id;
        if (row >= 128) continue;
        uint32_t global_row = S_q + row;
        uint32_t col_start = lane_id * 4;
        if (global_row < S_len && col_start + 3 < D) {
            uint2 data = *reinterpret_cast<uint2*>(&smem_out[row * 128 + col_start]);
            *reinterpret_cast<uint2*>(o_head_ptr + global_row * D + col_start) = data;
        }
    }

    __syncthreads();
    if (threadIdx.x < 32) {
        tmem_dealloc_1sm_fn(smem.tmem_addr, 256);
    }
}

CUresult create_tma_4d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t d0, uint64_t d1, uint64_t d2, uint64_t d3, uint32_t b0, uint32_t b1, CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[4] = {d0, d1, d2, d3};
    cuuint64_t globalStrides[3] = {d0 * 2, d0 * d1 * 2, d0 * d1 * d2 * 2};
    cuuint32_t boxDim[4] = {b0, b1, 1, 1};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        4,
        globalAddress,
        globalDim,
        globalStrides,
        boxDim, 
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S_len = Q.size(2);
    int64_t D = Q.size(3);
    
    CUtensorMap tma_Q, tma_K, tma_V;
    create_tma_4d_descriptor_2B(&tma_Q, Q.data_ptr(), D, S_len, H, B, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE);
    create_tma_4d_descriptor_2B(&tma_K, K.data_ptr(), D, S_len, H, B, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE);
    create_tma_4d_descriptor_2B(&tma_V, V.data_ptr(), D, S_len, H, B, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE);
    
    int num_blocks = (S_len + 127) / 128;
    dim3 grid(num_blocks, B * H);
    dim3 block(128);
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    size_t smem_size = sizeof(SharedMemory);
    cudaFuncSetAttribute(mha_fwd_sm100_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    mha_fwd_sm100_kernel<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V,
        static_cast<__nv_bfloat16*>(O.data_ptr()),
        static_cast<float*>(LSE.data_ptr()),
        S_len, D, H
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda