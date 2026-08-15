#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        exit(1);                                                   \
    }                                                              \
} while(0)

__device__ __forceinline__ void tma_copy_1d_g2s_fn(void const* gmem, uint64_t* mbar, void* smem, int32_t bytes) {
    uint32_t smem_mbar = (uint32_t)__cvta_generic_to_shared(mbar);
    uint32_t smem_ptr  = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
        :: "r"(smem_ptr), "l"(gmem), "r"(bytes), "r"(smem_mbar) : "memory");
}

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

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tcgen05_commit_cg1_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64 [%0];"
        :: "r"(a) : "memory"); 
}

__device__ __forceinline__ void umma_f16_cg1_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum) : "memory");
}

__device__ __forceinline__ void tmem_alloc_cg1_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(a), "r"(ncols) : "memory");
}

__device__ __forceinline__ void tmem_dealloc_cg1_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(addr), "r"(ncols) : "memory");
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)0 << 61;   
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, int a_major, int b_major) {
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

extern __shared__ __align__(128) uint8_t smem_base_bytes[];

__global__ void MHA_Kernel(
    const __nv_bfloat16* __restrict__ Q_ptr,
    const __nv_bfloat16* __restrict__ K_ptr,
    const __nv_bfloat16* __restrict__ V_ptr,
    __nv_bfloat16* __restrict__ O_ptr,
    float* __restrict__ LSE_ptr,
    int B, int H, int S) 
{
    __nv_bfloat16* smem_Q = (__nv_bfloat16*)(smem_base_bytes);
    __nv_bfloat16* smem_K = (__nv_bfloat16*)(smem_base_bytes + 32768);
    __nv_bfloat16* smem_V = (__nv_bfloat16*)(smem_base_bytes + 65536);
    __nv_bfloat16* smem_P = (__nv_bfloat16*)(smem_base_bytes + 98304);
    float* smem_O = (float*)(smem_base_bytes + 131072);
    
    uint64_t* mbar_tma = (uint64_t*)(smem_base_bytes + 196608);
    uint64_t* mbar_umma = (uint64_t*)(smem_base_bytes + 196616);
    uint32_t* tmem_qk_addr = (uint32_t*)(smem_base_bytes + 196624);
    uint32_t* tmem_pv_addr = (uint32_t*)(smem_base_bytes + 196628);

    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp = tid / 32;

    int b_idx = blockIdx.z;
    int h_idx = blockIdx.y;
    int s_idx = blockIdx.x;
    
    size_t batch_offset = (size_t)b_idx * H * S * 128 + (size_t)h_idx * S * 128;
    const __nv_bfloat16* Q_gmem = Q_ptr + batch_offset;
    const __nv_bfloat16* K_gmem = K_ptr + batch_offset;
    const __nv_bfloat16* V_gmem = V_ptr + batch_offset;
    __nv_bfloat16* O_gmem = O_ptr + batch_offset;
    float* LSE_gmem = LSE_ptr + (size_t)b_idx * H * S + (size_t)h_idx * S;

    int global_row = s_idx * 128 + tid;

    if (tid == 0) {
        init_smem_barrier_fn(mbar_tma, 1);
        init_smem_barrier_fn(mbar_umma, 1);
    }
    fence_smem_barrier_init_fn();
    
    if (warp == 0) {
        tmem_alloc_cg1_fn(tmem_qk_addr, 128);
        tmem_alloc_cg1_fn(tmem_pv_addr, 128);
    }
    __syncthreads();

    uint32_t tmem_qk = *tmem_qk_addr;
    uint32_t tmem_pv = *tmem_pv_addr;

    int phase_tma = 0;
    int phase_umma = 0;

    int q_rows = (S - s_idx * 128) > 128 ? 128 : (S - s_idx * 128);
    if (q_rows < 0) q_rows = 0;
    int q_bytes = q_rows * 256;
    
    if (warp == 0 && lane == 0) {
        if (q_bytes > 0) {
            tma_copy_1d_g2s_fn(Q_gmem + s_idx * 128 * 128, mbar_tma, smem_Q, q_bytes);
            mbarrier_arrive_and_expect_tx_fn(mbar_tma, q_bytes);
        } else {
            mbarrier_arrive_and_expect_tx_fn(mbar_tma, 0);
        }
    }
    mbarrier_wait_fn(mbar_tma, phase_tma);
    phase_tma ^= 1;
    
    if (q_rows < 128) {
        for (int i = tid; i < 128 * 128; i += 128) {
            if (i >= q_rows * 128) smem_Q[i] = __float2bfloat16(0.0f);
        }
    }
    __syncthreads();

    float m_val = -CUDART_INF_F;
    float l_val = 0.0f;

    for (int i = 0; i < 128; ++i) {
        smem_O[tid * 128 + i] = 0.0f;
    }
    __syncthreads();

    uint32_t idesc_qk = make_instr_desc_fn(128, 128, 1, 1);
    uint32_t idesc_pv = make_instr_desc_fn(128, 128, 1, 0);
    float scale = 1.0f / sqrtf(128.0f);

    int num_chunks = (S + 127) / 128;

    for (int j_chunk = 0; j_chunk < num_chunks; ++j_chunk) {
        int k_rows = (S - j_chunk * 128) > 128 ? 128 : (S - j_chunk * 128);
        if (k_rows < 0) k_rows = 0;
        int k_bytes = k_rows * 256;
        
        if (warp == 0 && lane == 0) {
            if (k_bytes > 0) {
                tma_copy_1d_g2s_fn(K_gmem + j_chunk * 128 * 128, mbar_tma, smem_K, k_bytes);
                tma_copy_1d_g2s_fn(V_gmem + j_chunk * 128 * 128, mbar_tma, smem_V, k_bytes);
                mbarrier_arrive_and_expect_tx_fn(mbar_tma, 2 * k_bytes);
            } else {
                mbarrier_arrive_and_expect_tx_fn(mbar_tma, 0);
            }
        }
        mbarrier_wait_fn(mbar_tma, phase_tma);
        phase_tma ^= 1;
        
        if (k_rows < 128) {
            for (int i = tid; i < 128 * 128; i += 128) {
                if (i >= k_rows * 128) {
                    smem_K[i] = __float2bfloat16(0.0f);
                    smem_V[i] = __float2bfloat16(0.0f);
                }
            }
        }
        __syncthreads();

        if (warp == 0 && lane == 0) {
            for (int k = 0; k < 8; ++k) {
                uint64_t desc_a = make_smem_desc_sm100_fn((char*)smem_Q + k * 32, 2048, 128);
                uint64_t desc_b = make_smem_desc_sm100_fn((char*)smem_K + k * 32, 2048, 128);
                uint32_t accum = (k == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_qk, desc_a, desc_b, idesc_qk, accum);
            }
            tcgen05_commit_cg1_fn(mbar_umma);
        }
        mbarrier_wait_fn(mbar_umma, phase_umma);
        phase_umma ^= 1;

        float row_max = -CUDART_INF_F;
        float row_sum = 0.0f;
        float S_row[128];

        for (int col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_qk + col));
            tmem_load_fence_fn();
            
            float f0 = __uint_as_float(r0) * scale;
            float f1 = __uint_as_float(r1) * scale;
            float f2 = __uint_as_float(r2) * scale;
            float f3 = __uint_as_float(r3) * scale;
            
            int k_idx = j_chunk * 128 + col;
            if (k_idx >= S) f0 = -CUDART_INF_F;
            if (k_idx + 1 >= S) f1 = -CUDART_INF_F;
            if (k_idx + 2 >= S) f2 = -CUDART_INF_F;
            if (k_idx + 3 >= S) f3 = -CUDART_INF_F;
            
            S_row[col] = f0; S_row[col+1] = f1; S_row[col+2] = f2; S_row[col+3] = f3;
            
            row_max = fmaxf(row_max, fmaxf(fmaxf(f0, f1), fmaxf(f2, f3)));
        }

        float m_new = fmaxf(m_val, row_max);
        float exp_diff = 0.0f;
        if (m_new != -CUDART_INF_F) {
            exp_diff = expf(m_val - m_new);
        }
        
        for (int col = 0; col < 128; ++col) {
            float p = 0.0f;
            if (m_new != -CUDART_INF_F) {
                p = expf(S_row[col] - m_new);
            }
            row_sum += p;
            smem_P[tid * 128 + col] = __float2bfloat16(p);
        }

        float l_new = l_val * exp_diff + row_sum;

        __syncthreads();
        fence_async_shared_fn();

        if (warp == 0 && lane == 0) {
            for (int k = 0; k < 8; ++k) {
                uint64_t desc_a = make_smem_desc_sm100_fn((char*)smem_P + k * 32, 2048, 128);
                uint64_t desc_b = make_smem_desc_sm100_fn((char*)smem_V + k * 4096, 16, 2048);
                uint32_t accum = (k == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_pv, desc_a, desc_b, idesc_pv, accum);
            }
            tcgen05_commit_cg1_fn(mbar_umma);
        }
        mbarrier_wait_fn(mbar_umma, phase_umma);
        phase_umma ^= 1;

        for (int col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_pv + col));
            tmem_load_fence_fn();
            
            float f0 = __uint_as_float(r0);
            float f1 = __uint_as_float(r1);
            float f2 = __uint_as_float(r2);
            float f3 = __uint_as_float(r3);
            
            smem_O[tid * 128 + col]     = smem_O[tid * 128 + col]     * exp_diff + f0;
            smem_O[tid * 128 + col + 1] = smem_O[tid * 128 + col + 1] * exp_diff + f1;
            smem_O[tid * 128 + col + 2] = smem_O[tid * 128 + col + 2] * exp_diff + f2;
            smem_O[tid * 128 + col + 3] = smem_O[tid * 128 + col + 3] * exp_diff + f3;
        }

        m_val = m_new;
        l_val = l_new;
        __syncthreads();
    }

    if (global_row < S) {
        for (int col = 0; col < 128; col += 4) {
            float f0 = smem_O[tid * 128 + col]     / l_val;
            float f1 = smem_O[tid * 128 + col + 1] / l_val;
            float f2 = smem_O[tid * 128 + col + 2] / l_val;
            float f3 = smem_O[tid * 128 + col + 3] / l_val;
            
            O_gmem[global_row * 128 + col]     = __float2bfloat16(f0);
            O_gmem[global_row * 128 + col + 1] = __float2bfloat16(f1);
            O_gmem[global_row * 128 + col + 2] = __float2bfloat16(f2);
            O_gmem[global_row * 128 + col + 3] = __float2bfloat16(f3);
        }
        LSE_gmem[global_row] = m_val + logf(l_val);
    }

    if (warp == 0) {
        tmem_dealloc_cg1_fn(tmem_qk, 128);
        tmem_dealloc_cg1_fn(tmem_pv, 128);
    }
}

namespace tvm_ffi_example_cuda {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    
    int blocks_x = (S + 127) / 128;
    dim3 blocks(blocks_x, H, B);
    dim3 threads(128);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int smem_size = 196632;
    CUDA_CHECK(cudaFuncSetAttribute(MHA_Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    MHA_Kernel<<<blocks, threads, smem_size, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<__nv_bfloat16*>(O.data_ptr()),
        static_cast<float*>(LSE.data_ptr()),
        B, H, S
    );
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

} // namespace tvm_ffi_example_cuda