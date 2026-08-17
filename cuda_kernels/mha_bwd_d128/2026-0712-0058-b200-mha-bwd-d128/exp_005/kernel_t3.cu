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

namespace tvm_ffi_attention_bwd {

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

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
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

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
        :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void umma_commit_1sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile(
        "tcgen05.commit.cta_group::1"
        ".mbarrier::arrive::one.shared::cta.b64"
        " [%0];"
        :: "r"(a));
}

__device__ __forceinline__ uint64_t make_smem_desc_128b_swizzle(void* smem_ptr, bool is_mn_major, uint32_t k_dim) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint32_t base_offset = (addr >> 7) & 0x7;
    uint32_t sbo = 1024;
    uint32_t lbo = is_mn_major ? (k_dim / 8) * sbo : 0;
    uint64_t d = 0;
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    d |= (uint64_t)base_offset << 49;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, bool a_transpose, bool b_transpose) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= ((a_transpose ? 1 : 0) << 15);   
    d |= ((b_transpose ? 1 : 0) << 16);   
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ uint32_t swizzle_128B_2B(uint32_t row, uint32_t col) {
    uint32_t x = col / 8;
    uint32_t rem = col % 8;
    uint32_t y = row % 8;
    uint32_t swizzled_x = x ^ y;
    return (row * 128) + (swizzled_x * 8) + rem;
}

__global__ void attention_backward_dQ_dK_dV(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    uint32_t S, float scale_factor)
{
    uint32_t batch_head = blockIdx.y;
    uint32_t n_block = blockIdx.x * 128;
    
    extern __shared__ char smem_raw[];
    uintptr_t smem_addr = (uintptr_t)smem_raw;
    uintptr_t aligned_addr = (smem_addr + 1023) & ~1023;
    __nv_bfloat16* smem_buf = (__nv_bfloat16*)aligned_addr;

    __nv_bfloat16* smem_Q = smem_buf;
    __nv_bfloat16* smem_K = smem_Q + 128 * 128;
    __nv_bfloat16* smem_V = smem_K + 128 * 128;
    __nv_bfloat16* smem_dO = smem_V + 128 * 128;
    __nv_bfloat16* smem_O = smem_dO + 128 * 128;
    __nv_bfloat16* smem_P = smem_O + 128 * 128;
    __nv_bfloat16* smem_dS = smem_P + 128 * 128;
    float* smem_D = (float*)(smem_dS + 128 * 128);
    float* smem_L = smem_D + 128;

    const __nv_bfloat16* q_ptr = Q + batch_head * S * 128;
    const __nv_bfloat16* k_ptr = K + batch_head * S * 128;
    const __nv_bfloat16* v_ptr = V + batch_head * S * 128;
    const __nv_bfloat16* o_ptr = O + batch_head * S * 128;
    const __nv_bfloat16* do_ptr = dO + batch_head * S * 128;
    __nv_bfloat16* dq_ptr = dQ + batch_head * S * 128;
    __nv_bfloat16* dk_ptr = dK + batch_head * S * 128;
    __nv_bfloat16* dv_ptr = dV + batch_head * S * 128;
    const float* l_ptr = L + batch_head * S;

    __shared__ alignas(16) uint64_t bar;
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&bar, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        uint32_t tmem_S, tmem_dP, tmem_dV, tmem_dK;
        tmem_alloc_fn(&tmem_S, 128);
        tmem_alloc_fn(&tmem_dP, 128);
        tmem_alloc_fn(&tmem_dV, 128);
        tmem_alloc_fn(&tmem_dK, 128);
    }
    __syncthreads();

    uint32_t phase = 0;

    for(int i=0; i<128; i+=2) {
        __nv_bfloat162 k_packed = *(__nv_bfloat162*)&k_ptr[n_block * 128 + threadIdx.x * 128 + i];
        uint32_t idx = swizzle_128B_2B(threadIdx.x, i);
        smem_K[idx] = __low2float(k_packed);
        smem_K[idx+1] = __high2float(k_packed);
        
        __nv_bfloat162 v_packed = *(__nv_bfloat162*)&v_ptr[n_block * 128 + threadIdx.x * 128 + i];
        smem_V[idx] = __low2float(v_packed);
        smem_V[idx+1] = __high2float(v_packed);
    }
    
    fence_proxy_async_fn();

    uint32_t tmem_dQ;

    for (int m_block = 0; m_block < S; m_block += 128) {
        for(int i=0; i<128; i+=2) {
            __nv_bfloat162 q_packed = *(__nv_bfloat162*)&q_ptr[m_block * 128 + threadIdx.x * 128 + i];
            uint32_t idx = swizzle_128B_2B(threadIdx.x, i);
            smem_Q[idx] = __low2float(q_packed);
            smem_Q[idx+1] = __high2float(q_packed);
            
            __nv_bfloat162 do_packed = *(__nv_bfloat162*)&do_ptr[m_block * 128 + threadIdx.x * 128 + i];
            smem_dO[idx] = __low2float(do_packed);
            smem_dO[idx+1] = __high2float(do_packed);

            __nv_bfloat162 o_packed = *(__nv_bfloat162*)&o_ptr[m_block * 128 + threadIdx.x * 128 + i];
            smem_O[idx] = __low2float(o_packed);
            smem_O[idx+1] = __high2float(o_packed);
        }

        if (threadIdx.x < 128) {
            int m = threadIdx.x;
            float d_val = 0;
            for(int c=0; c<128; c++) {
                d_val += __bfloat162float(smem_O[swizzle_128B_2B(m, c)]) * __bfloat162float(smem_dO[swizzle_128B_2B(m, c)]);
            }
            smem_D[m] = d_val;
            smem_L[m] = (m_block + m < S) ? l_ptr[m_block + m] : 0.0f;
        }
        __syncthreads();
        fence_proxy_async_fn();

        if (threadIdx.x == 0) {
            for(int i=0; i<8; i++) {
                uint64_t desc_a = make_smem_desc_128b_swizzle(smem_Q, false, 128);
                uint64_t desc_b = make_smem_desc_128b_swizzle(smem_K, false, 128);
                uint32_t idesc = make_instr_desc_fn(128, 128, false, false);
                asm volatile("{\n.reg .pred p;\n setp.ne.b32 p, %4, 0;\n tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}"
                    :: "r"(tmem_S), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(i==0 ? 0 : 1));
            }
            umma_commit_1sm_fn(&bar);
        }
        mbarrier_wait_fn(&bar, phase);
        phase ^= 1;
        __syncthreads();

        for(uint32_t col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_S + col, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            
            float s0 = __uint_as_float(r0);
            float s1 = __uint_as_float(r1);
            float s2 = __uint_as_float(r2);
            float s3 = __uint_as_float(r3);
            
            float l_val = smem_L[threadIdx.x];
            float p0 = __expf(s0 * scale_factor - l_val);
            float p1 = __expf(s1 * scale_factor - l_val);
            float p2 = __expf(s2 * scale_factor - l_val);
            float p3 = __expf(s3 * scale_factor - l_val);
            
            uint32_t idx = swizzle_128B_2B(threadIdx.x, col);
            smem_P[idx] = __float2bfloat16(p0);
            smem_P[idx+1] = __float2bfloat16(p1);
            smem_P[idx+2] = __float2bfloat16(p2);
            smem_P[idx+3] = __float2bfloat16(p3);
        }
        __syncthreads();
        fence_proxy_async_fn();

        if (threadIdx.x == 0) {
            for(int i=0; i<8; i++) {
                uint64_t desc_a = make_smem_desc_128b_swizzle(smem_V, false, 128);
                uint64_t desc_b = make_smem_desc_128b_swizzle(smem_dO, false, 128);
                uint32_t idesc = make_instr_desc_fn(128, 128, false, false);
                asm volatile("{\n.reg .pred p;\n setp.ne.b32 p, %4, 0;\n tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}"
                    :: "r"(tmem_dP), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(1));
            }
            umma_commit_1sm_fn(&bar);
        }
        mbarrier_wait_fn(&bar, phase);
        phase ^= 1;
        __syncthreads();

        for(uint32_t col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_dP + col, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            
            float dp0 = __uint_as_float(r0);
            float dp1 = __uint_as_float(r1);
            float dp2 = __uint_as_float(r2);
            float dp3 = __uint_as_float(r3);
            
            float d_val = smem_D[threadIdx.x];
            
            uint32_t idx = swizzle_128B_2B(threadIdx.x, col);
            float p0 = __bfloat162float(smem_P[idx]);
            float p1 = __bfloat162float(smem_P[idx+1]);
            float p2 = __bfloat162float(smem_P[idx+2]);
            float p3 = __bfloat162float(smem_P[idx+3]);
            
            float ds0 = p0 * (dp0 - d_val) * scale_factor;
            float ds1 = p1 * (dp1 - d_val) * scale_factor;
            float ds2 = p2 * (dp2 - d_val) * scale_factor;
            float ds3 = p3 * (dp3 - d_val) * scale_factor;
            
            smem_dS[idx] = __float2bfloat16(ds0);
            smem_dS[idx+1] = __float2bfloat16(ds1);
            smem_dS[idx+2] = __float2bfloat16(ds2);
            smem_dS[idx+3] = __float2bfloat16(ds3);
        }
        __syncthreads();
        fence_proxy_async_fn();

        if (threadIdx.x == 0) {
            for(int i=0; i<8; i++) {
                uint64_t desc_a = make_smem_desc_128b_swizzle(smem_P, false, 128);
                uint64_t desc_b = make_smem_desc_128b_swizzle(smem_dO, true, 128);
                uint32_t idesc = make_instr_desc_fn(128, 128, false, true);
                asm volatile("{\n.reg .pred p;\n setp.ne.b32 p, %4, 0;\n tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}"
                    :: "r"(tmem_dV), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(1));
            }
            for(int i=0; i<8; i++) {
                uint64_t desc_a = make_smem_desc_128b_swizzle(smem_dS, false, 128);
                uint64_t desc_b = make_smem_desc_128b_swizzle(smem_Q, true, 128);
                uint32_t idesc = make_instr_desc_fn(128, 128, false, true);
                asm volatile("{\n.reg .pred p;\n setp.ne.b32 p, %4, 0;\n tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}"
                    :: "r"(tmem_dK), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(1));
            }
            umma_commit_1sm_fn(&bar);
        }
        mbarrier_wait_fn(&bar, phase);
        phase ^= 1;
        __syncthreads();

        if (threadIdx.x < 32) {
            tmem_dealloc_fn(tmem_S, 128);
            tmem_alloc_fn(&tmem_dQ, 128);
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            for(int i=0; i<8; i++) {
                uint64_t desc_a = make_smem_desc_128b_swizzle(smem_dS, false, 128);
                uint64_t desc_b = make_smem_desc_128b_swizzle(smem_K, true, 128);
                uint32_t idesc = make_instr_desc_fn(128, 128, false, true);
                asm volatile("{\n.reg .pred p;\n setp.ne.b32 p, %4, 0;\n tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}"
                    :: "r"(tmem_dQ), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(i==0 ? 0 : 1));
            }
            umma_commit_1sm_fn(&bar);
        }
        mbarrier_wait_fn(&bar, phase);
        phase ^= 1;
        __syncthreads();

        if (threadIdx.x < 128) {
            int m = threadIdx.x;
            if (m_block + m < S) {
                for(uint32_t col = 0; col < 128; col += 2) {
                    uint32_t r0, r1, r2, r3;
                    tmem_load_4x_fn(tmem_dQ + col, &r0, &r1, &r2, &r3);
                    tmem_load_fence_fn();
                    float dq0 = __uint_as_float(r0);
                    float dq1 = __uint_as_float(r1);
                    
                    __nv_bfloat162 dq_packed = __float2bfloat162(dq0, dq1);
                    atomicAdd((__nv_bfloat162*)&dq_ptr[m * 128 + col], dq_packed);
                }
            }
        }

        if (threadIdx.x < 32) {
            tmem_dealloc_fn(tmem_dQ, 128);
            tmem_alloc_fn(&tmem_S, 128);
        }
        __syncthreads();
    }

    if (threadIdx.x < 128) {
        int m = threadIdx.x;
        if (n_block + m < S) {
            for(uint32_t col = 0; col < 128; col += 2) {
                uint32_t r0, r1, r2, r3;
                tmem_load_4x_fn(tmem_dK + col, &r0, &r1, &r2, &r3);
                tmem_load_fence_fn();
                float dk0 = __uint_as_float(r0);
                float dk1 = __uint_as_float(r1);
                __nv_bfloat162 dk_packed = __float2bfloat162(dk0, dk1);
                *(__nv_bfloat162*)&dk_ptr[(n_block + m) * 128 + col] = dk_packed;
                
                tmem_load_4x_fn(tmem_dV + col, &r0, &r1, &r2, &r3);
                tmem_load_fence_fn();
                float dv0 = __uint_as_float(r0);
                float dv1 = __uint_as_float(r1);
                __nv_bfloat162 dv_packed = __float2bfloat162(dv0, dv1);
                *(__nv_bfloat162*)&dv_ptr[(n_block + m) * 128 + col] = dv_packed;
            }
        }
    }

    if (threadIdx.x < 32) {
        tmem_dealloc_fn(tmem_S, 128);
        tmem_dealloc_fn(tmem_dP, 128);
        tmem_dealloc_fn(tmem_dV, 128);
        tmem_dealloc_fn(tmem_dK, 128);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = Q.size(3); 
    
    float scale_factor = 1.0f / sqrtf((float)d);

    dim3 grid((S + 127) / 128, B * H);
    dim3 block(128);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    size_t shmem = 220000;

    CUDA_CHECK(cudaFuncSetAttribute(attention_backward_dQ_dK_dV, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem));

    attention_backward_dQ_dK_dV<<<grid, block, shmem, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(O.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        S, scale_factor);
    
    CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_attention_bwd::run);

} // namespace tvm_ffi_attention_bwd