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

__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
    uint32_t smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(smem_ptr), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
__device__ __forceinline__ void cp_async_wait_0() { asm volatile("cp.async.wait_group 0;"); }

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void tmem_alloc_cg1_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_cg1_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(addr), "r"(ncols));
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

__device__ __forceinline__ void umma_commit_cg1_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(a));
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
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

__device__ __forceinline__ void tmem_load_8x_fn(uint32_t col,
    uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3,
    uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3),
                   "=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7) : "r"(col));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_swizzled(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint32_t base_offset = (addr >> 7) & 0x7;
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)base_offset << 49;
    d |= (uint64_t)2 << 61; // 128B swizzle
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_custom_fn(uint32_t M, uint32_t N, int a_major, int b_major) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= ((uint32_t)a_major << 15);   
    d |= ((uint32_t)b_major << 16); 
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ void load_swizzled_128b_16(void* smem, const void* gmem, int rows, int cols) {
    int total_chunks = (rows * cols) / 8;
    const char* g = (const char*)gmem;
    char* s = (char*)smem;
    for(int i = threadIdx.x; i < total_chunks; i += blockDim.x) {
        int r = i / (cols / 8);
        int c_16b = i % (cols / 8);
        int swizzled_c_16b = (c_16b % 8) ^ (r % 8);
        uint32_t smem_offset = r * (cols * 2) + (c_16b / 8) * 128 + swizzled_c_16b * 16;
        cp_async_16(s + smem_offset, g + i * 16);
    }
}

struct SharedStorage {
    alignas(1024) __nv_bfloat16 Q[128 * 128]; // 32KB
    alignas(1024) __nv_bfloat16 K[64 * 128];  // 16KB
    alignas(1024) __nv_bfloat16 V[64 * 128];  // 16KB
    alignas(1024) __nv_bfloat16 P[128 * 64];  // 16KB
    alignas(8) uint64_t bar_mma[1];
    alignas(16) uint32_t tmem_addr;
};

__global__ void mha_fwd_sm100_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S, int D)
{
    extern __shared__ char smem_buf[];
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(smem_buf);

    int b = blockIdx.z;
    int h = blockIdx.y;
    int m_block = blockIdx.x;
    int tid = threadIdx.x;

    const __nv_bfloat16* q_ptr = Q + b * H * S * D + h * S * D + m_block * 128 * D;
    const __nv_bfloat16* k_base = K + b * H * S * D + h * S * D;
    const __nv_bfloat16* v_base = V + b * H * S * D + h * S * D;
    __nv_bfloat16* o_ptr = O + b * H * S * D + h * S * D + m_block * 128 * D;
    float* lse_ptr = LSE + b * H * S + h * S + m_block * 128;

    if (tid == 0) {
        init_smem_barrier_fn(smem.bar_mma, 1);
        tmem_alloc_cg1_fn(&smem.tmem_addr, 256);
    }
    
    load_swizzled_128b_16(smem.Q, q_ptr, 128, 128);
    cp_async_commit();
    cp_async_wait_0();
    __syncthreads();

    uint32_t tmem_O_new = smem.tmem_addr;
    uint32_t tmem_S = smem.tmem_addr + 128;
    uint32_t mma_phase = 0;

    float m_i = -INFINITY;
    float l_i = 0.0f;
    float O_reg[128] = {0};

    int num_n_blocks = (S + 63) / 64;
    for(int n_block = 0; n_block < num_n_blocks; ++n_block) {
        load_swizzled_128b_16(smem.K, k_base + n_block * 64 * D, 64, 128);
        load_swizzled_128b_16(smem.V, v_base + n_block * 64 * D, 64, 128);
        cp_async_commit();
        cp_async_wait_0();
        __syncthreads();

        if (tid == 0) {
            for(int k = 0; k < 128; k += 16) {
                uint32_t q_col_16b = k / 8;
                uint32_t q_swizzled_c = (q_col_16b % 8) ^ 0;
                uint32_t q_offset = (q_col_16b / 8) * 128 + q_swizzled_c * 16;
                uint64_t d_Q = make_smem_desc_sm100_swizzled((char*)smem.Q + q_offset, 1, 1024);

                uint32_t k_col_16b = k / 8;
                uint32_t k_swizzled_c = (k_col_16b % 8) ^ 0;
                uint32_t k_offset = (k_col_16b / 8) * 128 + k_swizzled_c * 16;
                uint64_t d_K = make_smem_desc_sm100_swizzled((char*)smem.K + k_offset, 1, 1024);

                uint32_t idesc = make_instr_desc_custom_fn(128, 64, 0, 0);
                umma_f16_cg1_fn(tmem_S, d_Q, d_K, idesc, (k == 0) ? 0 : 1);
            }
            umma_commit_cg1_fn(smem.bar_mma);
            mbarrier_wait_fn(smem.bar_mma, mma_phase);
        }
        __syncthreads();
        mma_phase ^= 1;

        float row_max = -INFINITY;
        float P_reg[64];

        for(int c = 0; c < 64; c += 8) {
            uint32_t r[8];
            tmem_load_8x_fn(tmem_S + c, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
            tmem_load_fence_fn(); 
            for(int i = 0; i < 8; ++i) {
                float val = __uint_as_float(r[i]);
                val *= 0.08838834764f;
                P_reg[c + i] = val;
                row_max = max(row_max, val);
            }
        }

        float m_new = max(m_i, row_max);
        float exp_diff = expf(m_i - m_new);
        l_i = l_i * exp_diff;

        float row_sum_new = 0.0f;
        for(int c = 0; c < 64; ++c) {
            float p = fast_exp2f_fn((P_reg[c] - m_new) * 1.4426950408889634f);
            P_reg[c] = p;
            row_sum_new += p;
        }
        l_i += row_sum_new;
        m_i = m_new;

        for(int i = 0; i < 128; ++i) {
            O_reg[i] *= exp_diff;
        }

        for(int c = 0; c < 64; ++c) {
            int c_16b = c / 8;
            int swizzled_c = (c_16b % 8) ^ (tid % 8);
            uint32_t p_offset = tid * 128 + (c_16b / 8) * 128 + swizzled_c * 16 + (c % 8) * 2;
            ((__nv_bfloat16*)((char*)smem.P + p_offset))[0] = __float2bfloat16(P_reg[c]);
        }
        __syncthreads();

        if (tid == 0) {
            for(int k = 0; k < 64; k += 16) {
                uint32_t p_col_16b = k / 8;
                uint32_t p_swizzled_c = (p_col_16b % 8) ^ 0;
                uint32_t p_offset = (p_col_16b / 8) * 128 + p_swizzled_c * 16;
                uint64_t d_P = make_smem_desc_sm100_swizzled((char*)smem.P + p_offset, 1, 1024);

                uint32_t v_row = k;
                uint32_t v_col_16b = 0;
                uint32_t v_swizzled_c = (v_col_16b % 8) ^ (v_row % 8);
                uint32_t v_offset = v_row * 256 + (v_col_16b / 8) * 128 + v_swizzled_c * 16;
                uint64_t d_V = make_smem_desc_sm100_swizzled((char*)smem.V + v_offset, 8192, 1024);

                uint32_t idesc = make_instr_desc_custom_fn(128, 128, 0, 1);
                umma_f16_cg1_fn(tmem_O_new, d_P, d_V, idesc, (k == 0) ? 0 : 1);
            }
            umma_commit_cg1_fn(smem.bar_mma);
            mbarrier_wait_fn(smem.bar_mma, mma_phase);
        }
        __syncthreads();
        mma_phase ^= 1;

        for(int c = 0; c < 128; c += 8) {
            uint32_t r[8];
            tmem_load_8x_fn(tmem_O_new + c, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
            tmem_load_fence_fn();
            for(int i = 0; i < 8; ++i) {
                O_reg[c + i] += __uint_as_float(r[i]);
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        tmem_dealloc_cg1_fn(0, 256);
    }

    float lse = m_i + logf(l_i);
    lse_ptr[tid] = lse;

    for(int i = 0; i < 128; ++i) {
        O_reg[i] /= l_i;
    }

    __nv_bfloat16* smem_Q_linear = (__nv_bfloat16*)smem.Q;
    for(int c = 0; c < 128; ++c) {
        smem_Q_linear[tid * 128 + c] = __float2bfloat16(O_reg[c]);
    }
    __syncthreads();

    for(int i = tid; i < 128 * 128; i += blockDim.x) {
        o_ptr[i] = smem_Q_linear[i];
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int D = Q.size(3);

    const __nv_bfloat16* q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* k_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* v_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_ptr = static_cast<float*>(LSE.data_ptr());

    dim3 grid((S + 127) / 128, H, B);
    dim3 block(128); // 4 warps
    size_t smem_bytes = sizeof(SharedStorage);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUDA_CHECK(cudaFuncSetAttribute(mha_fwd_sm100_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    mha_fwd_sm100_kernel<<<grid, block, smem_bytes, stream>>>(q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr, B, H, S, D);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda