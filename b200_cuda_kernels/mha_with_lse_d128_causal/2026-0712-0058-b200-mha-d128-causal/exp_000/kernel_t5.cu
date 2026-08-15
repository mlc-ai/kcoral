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

namespace tvm_ffi_mha {

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

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
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

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
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
        "tcgen05.commit.cta_group::2"
        ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;"
        :: "r"(a), "h"((uint16_t)0x3)); 
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;   // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);     // c_format = FP32
    d |= (1u << 7);     // a_format = BF16
    d |= (1u << 10);    // b_format = BF16
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ uint32_t pack_bf16_fn(float a, float b) {
    __nv_bfloat16 ba = __float2bfloat16(a);
    __nv_bfloat16 bb = __float2bfloat16(b);
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};"
        : "=r"(result)
        : "h"(*reinterpret_cast<uint16_t*>(&ba)),
          "h"(*reinterpret_cast<uint16_t*>(&bb)));
    return result;
}


__global__ __launch_bounds__(128, 1)
void mha_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O, float* LSE, uint32_t S) 
{
    extern __shared__ __align__(1024) uint8_t smem_pool[];
    __nv_bfloat16* q_smem = (__nv_bfloat16*)smem_pool;
    __nv_bfloat16* k_smem = q_smem + 16384;
    __nv_bfloat16* v_smem = k_smem + 16384;
    __nv_bfloat16* p_smem = v_smem + 16384;
    uint64_t* bar_q = (uint64_t*)(p_smem + 16384);
    uint64_t* bar_k = bar_q + 1;
    uint64_t* bar_v = bar_k + 1;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar_q, 1);
        init_smem_barrier_fn(bar_k, 1);
        init_smem_barrier_fn(bar_v, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t smem_s_base, smem_o_base;
    if (threadIdx.x == 0) {
        tmem_alloc_fn(&smem_s_base, 256);
        smem_o_base = smem_s_base + 128;
    }
    __syncthreads();

    uint32_t total_q_blks = (S + 127) / 128;
    uint32_t q_blk_idx = blockIdx.x % total_q_blks;
    uint32_t head_idx = blockIdx.x / total_q_blks;

    int phase[3] = {0, 0, 0};
    float scale_factor = 1.0f / sqrtf(128.0f);

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_q, 32768);
        tma_load_3d_fn(&tma_Q, bar_q, q_smem, 0, q_blk_idx * 128, head_idx);
        tma_load_3d_fn(&tma_Q, bar_q, (uint8_t*)q_smem + 8192, 64, q_blk_idx * 128, head_idx);
    }
    mbarrier_wait_fn(bar_q, phase[0]);
    phase[0] ^= 1;
    __syncthreads();

    uint32_t global_q_idx = q_blk_idx * 128 + threadIdx.x;
    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    uint32_t qk_idesc = make_instr_desc_fn(128, 128);
    uint32_t pv_idesc = make_instr_desc_fn(128, 128) | (1u << 16); // Transpose B

    #pragma unroll 1
    for (uint32_t k_blk_idx = 0; k_blk_idx <= q_blk_idx; ++k_blk_idx) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_k, 32768);
            tma_load_3d_fn(&tma_K, bar_k, k_smem, 0, k_blk_idx * 128, head_idx);
            tma_load_3d_fn(&tma_K, bar_k, (uint8_t*)k_smem + 8192, 64, k_blk_idx * 128, head_idx);
        }
        mbarrier_wait_fn(bar_k, phase[1]);
        phase[1] ^= 1;
        __syncthreads();

        uint64_t desc_a = make_smem_desc_sm100_fn(q_smem, 0, 1024);
        uint64_t desc_b = make_smem_desc_sm100_fn(k_smem, 0, 1024);
        
        int s_base = 0;
        // Loop for the first 64 elements of K
        for (int i = 0; i < 4; ++i) {
            umma_f16_cg2_fn(s_base, desc_a, desc_b, qk_idesc, i > 0);
            desc_a += 16;
            desc_b += 16;
            s_base += 16;
        }

        // Jump to next chunk of Q/K without accumulating S
        desc_a = make_smem_desc_sm100_fn((uint8_t*)q_smem + 8192, 0, 1024);
        desc_b = make_smem_desc_sm100_fn((uint8_t*)k_smem + 8192, 0, 1024);

        // Loop for the second 64 elements of K
        for (int i = 0; i < 4; ++i) {
            umma_f16_cg2_fn(s_base, desc_a, desc_b, qk_idesc, true);
            desc_a += 16;
            desc_b += 16;
            s_base += 16;
        }
        
        umma_commit_2sm_fn(bar_q);
        mbarrier_wait_fn(bar_q, phase[0]);
        phase[0] ^= 1;
        __syncthreads();

        float m_curr = -INFINITY;
        float l_curr = 0.0f;

        for (int c = 0; c < 128; c += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(smem_s_base + c));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float s0 = __uint_as_float(r0);
            float s1 = __uint_as_float(r1);
            float s2 = __uint_as_float(r2);
            float s3 = __uint_as_float(r3);

            uint32_t col_base = k_blk_idx * 128 + c;
            if (global_q_idx >= S || col_base >= S || col_base > global_q_idx) {
                s0 = -INFINITY; s1 = -INFINITY; s2 = -INFINITY; s3 = -INFINITY;
            } else {
                s0 *= scale_factor; s1 *= scale_factor; s2 *= scale_factor; s3 *= scale_factor;
            }
            m_curr = fmaxf(m_curr, fmaxf(fmaxf(s0, s1), fmaxf(s2, s3)));
            
            s0 = fast_exp2f_fn((s0 - m_curr) * 1.4426950f);
            s1 = fast_exp2f_fn((s1 - m_curr) * 1.4426950f);
            s2 = fast_exp2f_fn((s2 - m_curr) * 1.4426950f);
            s3 = fast_exp2f_fn((s3 - m_curr) * 1.4426950f);
            l_curr += s0 + s1 + s2 + s3;
            
            uint32_t p0 = pack_bf16_fn(s0, s1);
            uint32_t p1 = pack_bf16_fn(s2, s3);
            
            int x_chunk = c / 8;
            int x_swizzled = x_chunk ^ (threadIdx.x % 8);
            int swizzled_c = x_swizzled * 8 + (c % 8);
            
            *(uint32_t*)&p_smem[threadIdx.x * 128 + swizzled_c] = p0;
            *(uint32_t*)&p_smem[threadIdx.x * 128 + swizzled_c + 2] = p1;
        }

        m_curr = fmaxf(m_curr, __shfl_xor_sync(0xFFFFFFFF, m_curr, 1));
        m_curr = fmaxf(m_curr, __shfl_xor_sync(0xFFFFFFFF, m_curr, 2));
        m_curr = fmaxf(m_curr, __shfl_xor_sync(0xFFFFFFFF, m_curr, 4));
        m_curr = fmaxf(m_curr, __shfl_xor_sync(0xFFFFFFFF, m_curr, 8));
        m_curr = fmaxf(m_curr, __shfl_xor_sync(0xFFFFFFFF, m_curr, 16));

        l_curr += __shfl_xor_sync(0xFFFFFFFF, l_curr, 1);
        l_curr += __shfl_xor_sync(0xFFFFFFFF, l_curr, 2);
        l_curr += __shfl_xor_sync(0xFFFFFFFF, l_curr, 4);
        l_curr += __shfl_xor_sync(0xFFFFFFFF, l_curr, 8);
        l_curr += __shfl_xor_sync(0xFFFFFFFF, l_curr, 16);
        
        float m_new = fmaxf(m_prev, m_curr);
        float l_new = l_prev * fast_exp2f_fn((m_prev - m_new) * 1.4426950f) + l_curr;
        
        if (l_prev > 0.0f) {
            float scale = fast_exp2f_fn((m_prev - m_new) * 1.4426950f);
            for (int c = 0; c < 128; c += 4) {
                uint32_t r0, r1, r2, r3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(smem_o_base + c));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                
                float f0 = __uint_as_float(r0) * scale;
                float f1 = __uint_as_float(r1) * scale;
                float f2 = __uint_as_float(r2) * scale;
                float f3 = __uint_as_float(r3) * scale;
                
                uint32_t p0 = pack_bf16_fn(f0, f1);
                uint32_t p1 = pack_bf16_fn(f2, f3);
                
                asm volatile("tcgen05.st.sync.aligned.16x128b.x2.pack::16b {%0}, [%1];"
                :: "r"(p0), "r"(smem_o_base + (c << 1)));
                asm volatile("tcgen05.st.sync.aligned.16x128b.x2.pack::16b {%0}, [%1];"
                :: "r"(p1), "r"(smem_o_base + (c << 1) + 2));
            }
        }
        
        __syncthreads();
        fence_proxy_async_fn();
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_v, 32768);
            tma_load_3d_fn(&tma_V, bar_v, v_smem, 0, k_blk_idx * 128, head_idx);
            tma_load_3d_fn(&tma_V, bar_v, (uint8_t*)v_smem + 8192, 64, k_blk_idx * 128, head_idx);
        }
        mbarrier_wait_fn(bar_v, phase[2]);
        phase[2] ^= 1;
        __syncthreads();

        uint64_t desc_p = make_smem_desc_sm100_fn(p_smem, 0, 1024);
        uint64_t desc_v = make_smem_desc_sm100_fn(v_smem, 1024, 1024);

        int o_base = 0;
        // Iterate 4 times over the first 64 items of V
        for (int i = 0; i < 4; ++i) {
            umma_f16_cg2_fn(o_base, desc_p, desc_v, pv_idesc, i > 0);
            desc_p += 16;
            desc_v += 16;
            o_base += 16;
        }

        // Iterate 4 times over the next 64 items of V
        uint64_t desc_p2 = make_smem_desc_sm100_fn((uint8_t*)p_smem + 8192, 0, 1024);
        uint64_t desc_v2 = make_smem_desc_sm100_fn((uint8_t*)v_smem + 8192, 1024, 1024);
        for (int i = 0; i < 4; ++i) {
            umma_f16_cg2_fn(o_base, desc_p2, desc_v2, pv_idesc, true);
            desc_p2 += 16;
            desc_v2 += 16;
            o_base += 16;
        }

        umma_commit_2sm_fn(bar_v);
        mbarrier_wait_fn(bar_v, phase[2]);
        phase[2] ^= 1;
        __syncthreads();
        
        m_prev = m_new;
        l_prev = l_new;
    }

    for (int c = 0; c < 128; c += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(smem_o_base + c));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float f0 = __uint_as_float(r0) / l_prev;
        float f1 = __uint_as_float(r1) / l_prev;
        float f2 = __uint_as_float(r2) / l_prev;
        float f3 = __uint_as_float(r3) / l_prev;

        if (global_q_idx < S) {
            __nv_bfloat16 bf0 = __float2bfloat16(f0);
            __nv_bfloat16 bf1 = __float2bfloat16(f1);
            __nv_bfloat16 bf2 = __float2bfloat16(f2);
            __nv_bfloat16 bf3 = __float2bfloat16(f3);
            
            uint32_t base_idx = head_idx * S * 128 + global_q_idx * 128 + c;
            O[base_idx] = bf0;
            O[base_idx + 1] = bf1;
            O[base_idx + 2] = bf2;
            O[base_idx + 3] = bf3;
        }
    }

    if (threadIdx.x < 128 && global_q_idx < S) {
        LSE[head_idx * S + global_q_idx] = m_prev + logf(l_prev);
    }
    
    fence_proxy_async_fn();

    if (threadIdx.x == 0) {
        tmem_dealloc_fn(smem_s_base, 256);
    }
}

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_dim0, uint64_t gmem_dim1, uint64_t gmem_dim2, uint32_t smem_dim0, uint32_t smem_dim1, uint32_t smem_dim2, CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[3] = {gmem_dim0, gmem_dim1, gmem_dim2};
    cuuint64_t globalStrides[2] = {gmem_dim0 * 2, gmem_dim0 * gmem_dim1 * 2};
    cuuint32_t boxDim[3] = {smem_dim0, smem_dim1, smem_dim2};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        3, 
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

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);

    CUtensorMap tma_Q, tma_K, tma_V;
    create_tma_3d_descriptor_2B(&tma_Q, Q.data_ptr(), D, S, B * H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B);
    create_tma_3d_descriptor_2B(&tma_K, K.data_ptr(), D, S, B * H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B);
    create_tma_3d_descriptor_2B(&tma_V, V.data_ptr(), D, S, B * H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B);

    int64_t total_q_blks = (S + 127) / 128;
    dim3 grid(total_q_blks * B * H, 1, 1);
    dim3 block(128, 1, 1);

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUDA_CHECK(cudaFuncSetAttribute(mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 228 * 1024));
    mha_kernel<<<grid, block, 228 * 1024, stream>>>(tma_Q, tma_K, tma_V, static_cast<__nv_bfloat16*>(O.data_ptr()), static_cast<float*>(LSE.data_ptr()), S);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha::run);

}  // namespace tvm_ffi_mha