#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
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

__device__ __forceinline__ uint16_t __float2bfloat16_impl(float f) {
    __nv_bfloat16 b = __float2bfloat16(f);
    return *reinterpret_cast<uint16_t*>(&b);
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

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void fence_mbarrier_init_fn() {
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

__device__ __forceinline__ void tma_copy_1d_g2s_fn(void const* gmem, uint64_t* mbar, void* smem, int32_t bytes) {
    uint32_t smem_mbar = (uint32_t)__cvta_generic_to_shared(mbar);
    uint32_t smem_ptr  = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
        :: "r"(smem_ptr), "l"(gmem), "r"(bytes), "r"(smem_mbar) : "memory");
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

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_no_swizzle(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16; 
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32; 
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)0 << 61;   
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

__device__ __forceinline__ void umma_f16_cg1_fn(uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, bool accum) {
    uint32_t acc_val = accum ? 1 : 0;
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(acc_val));
}

__device__ __forceinline__ void umma_commit_1sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        :: "r"(a));
}

extern __shared__ __align__(128) uint8_t dynamic_smem[];

__global__ __launch_bounds__(128)
void CausalAttentionBlackwellKernel(
    const uint16_t* __restrict__ Q,
    const uint16_t* __restrict__ K,
    const uint16_t* __restrict__ V,
    uint16_t* __restrict__ O,
    float* __restrict__ LSE,
    int S, int H)
{
    int bx = blockIdx.x; 
    int by = blockIdx.y; 
    int bz = blockIdx.z; 

    int q_start = bx * 128;
    if (q_start >= S) return;
    int q_len = min(128, S - q_start);

    uint16_t* smem_Q = (uint16_t*)dynamic_smem;
    uint16_t* smem_K = smem_Q + 16384;
    uint16_t* smem_V = smem_K + 16384;
    uint16_t* smem_P = smem_V + 16384;
    
    uint64_t* mbar_Q    = (uint64_t*)(smem_P + 16384);
    uint64_t* mbar_KV   = mbar_Q + 1;
    uint64_t* mbar_umma = mbar_KV + 1;
    uint32_t* tmem_addr_ptr = (uint32_t*)(mbar_umma + 1);

    if (threadIdx.x < 32) {
        if (threadIdx.x == 0) {
            init_smem_barrier_fn(mbar_Q, 1);
            init_smem_barrier_fn(mbar_KV, 1);
            init_smem_barrier_fn(mbar_umma, 1);
            fence_mbarrier_init_fn();
        }
        tmem_alloc_fn(tmem_addr_ptr, 128);
    }
    __syncthreads();

    uint32_t tmem_addr = *tmem_addr_ptr;
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, q_len * 128 * 2);
        uint64_t q_offset = (bz * H * S + by * S + q_start) * 128;
        tma_copy_1d_g2s_fn(Q + q_offset, mbar_Q, smem_Q, q_len * 128 * 2);
    }
    
    if (q_len < 128) {
        for (int i = threadIdx.x; i < (128 - q_len) * 128; i += blockDim.x) {
            smem_Q[q_len * 128 + i] = 0;
        }
    }

    uint32_t w = threadIdx.x / 32;
    uint32_t lane = threadIdx.x % 32;
    uint32_t my_row = w * 32 + lane;
    uint32_t global_row = q_start + my_row;

    float m_i = -INFINITY;
    float l_i = 0.0f;
    float O_acc[2][16][4]; 

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 16; ++j)
            for (int k = 0; k < 4; ++k)
                O_acc[i][j][k] = 0.0f;

    uint32_t idesc_QK = make_instr_desc_fn(128, 128);
    float scale = 1.0f / sqrtf(128.0f);
    
    mbarrier_wait_fn(mbar_Q, 0);
    fence_async_shared_fn();

    int kv_phase = 0;
    for (int k_start = 0; k_start <= q_start; k_start += 128) {
        int k_len = min(128, S - k_start);
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_KV, k_len * 128 * 2 * 2);
            uint64_t kv_offset = (bz * H * S + by * S + k_start) * 128;
            tma_copy_1d_g2s_fn(K + kv_offset, mbar_KV, smem_K, k_len * 128 * 2);
            tma_copy_1d_g2s_fn(V + kv_offset, mbar_KV, smem_V, k_len * 128 * 2);
        }
        
        if (k_len < 128) {
            for (int i = threadIdx.x; i < (128 - k_len) * 128; i += blockDim.x) {
                smem_K[k_len * 128 + i] = 0;
                smem_V[k_len * 128 + i] = 0;
            }
        }
        
        mbarrier_wait_fn(mbar_KV, kv_phase & 1);
        fence_async_shared_fn();
        
        if (threadIdx.x == 0) {
            for (int k_step = 0; k_step < 8; ++k_step) {
                uint32_t q_addr = (uint32_t)__cvta_generic_to_shared(smem_Q) + k_step * 32;
                uint32_t k_addr = (uint32_t)__cvta_generic_to_shared(smem_K) + k_step * 32;
                uint64_t desc_Q = make_smem_desc_sm100_no_swizzle((void*)q_addr, 16, 2048);
                uint64_t desc_K = make_smem_desc_sm100_no_swizzle((void*)k_addr, 16, 2048);
                bool accum = (k_step > 0);
                umma_f16_cg1_fn(tmem_addr, desc_Q, desc_K, idesc_QK, accum);
            }
            umma_commit_1sm_fn(mbar_umma);
        }
        mbarrier_wait_fn(mbar_umma, kv_phase & 1);
        tcgen05_fence_after_fn();
        
        float row_max = -INFINITY;
        for (int c = 0; c < 128; c += 4) {
            uint32_t r[4];
            tmem_load_4x_fn(tmem_addr + c, &r[0], &r[1], &r[2], &r[3]);
            tmem_load_fence_fn();
            for (int i = 0; i < 4; ++i) {
                float val = __uint_as_float(r[i]) * scale;
                int global_col = k_start + c + i;
                if (global_col > global_row || global_row >= S || global_col >= S) val = -INFINITY;
                row_max = fmaxf(row_max, val);
            }
        }
        
        float m_new = fmaxf(m_i, row_max);
        float m_new_safe = (m_new == -INFINITY) ? 0.0f : m_new;
        float exp_diff = (m_i == -INFINITY) ? 0.0f : expf(m_i - m_new_safe);
        
        float exp_diff_0 = __shfl_sync(0xffffffff, exp_diff, lane / 4);
        float exp_diff_1 = __shfl_sync(0xffffffff, exp_diff, lane / 4 + 8);
        float exp_diff_2 = __shfl_sync(0xffffffff, exp_diff, 16 + lane / 4);
        float exp_diff_3 = __shfl_sync(0xffffffff, exp_diff, 16 + lane / 4 + 8);

        for (int j = 0; j < 16; ++j) {
            O_acc[0][j][0] *= exp_diff_0;
            O_acc[0][j][1] *= exp_diff_0;
            O_acc[0][j][2] *= exp_diff_1;
            O_acc[0][j][3] *= exp_diff_1;
            
            O_acc[1][j][0] *= exp_diff_2;
            O_acc[1][j][1] *= exp_diff_2;
            O_acc[1][j][2] *= exp_diff_3;
            O_acc[1][j][3] *= exp_diff_3;
        }

        float row_sum = 0.0f;
        for (int c = 0; c < 128; c += 4) {
            uint32_t r[4];
            tmem_load_4x_fn(tmem_addr + c, &r[0], &r[1], &r[2], &r[3]);
            tmem_load_fence_fn();
            
            float p[4];
            for (int i = 0; i < 4; ++i) {
                float val = __uint_as_float(r[i]) * scale;
                int global_col = k_start + c + i;
                if (global_col > global_row || global_row >= S || global_col >= S) val = -INFINITY;
                p[i] = (val == -INFINITY) ? 0.0f : expf(val - m_new_safe);
                row_sum += p[i];
            }
            
            uint32_t p01 = pack_bf16_fn(__float_as_uint(p[0]), __float_as_uint(p[1]));
            uint32_t p23 = pack_bf16_fn(__float_as_uint(p[2]), __float_as_uint(p[3]));
            
            uint32_t* smem_P_ptr = (uint32_t*)&smem_P[my_row * 128 + c];
            smem_P_ptr[0] = p01;
            smem_P_ptr[1] = p23;
        }
        l_i = l_i * exp_diff + row_sum;
        m_i = m_new;
        
        __syncthreads(); 
        
        for (int k = 0; k < 128; k += 16) {
            uint32_t P_regs[2][4]; 
            for (int i = 0; i < 2; ++i) {
                int p_row = lane % 16;
                int p_col = k + (lane / 16) * 8;
                uint32_t p_addr = (uint32_t)__cvta_generic_to_shared(&smem_P[(w * 32 + i * 16 + p_row) * 128 + p_col]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                             : "=r"(P_regs[i][0]), "=r"(P_regs[i][1]), "=r"(P_regs[i][2]), "=r"(P_regs[i][3])
                             : "r"(p_addr));
            }
        
            uint32_t V_regs[8][4];
            for (int j = 0; j < 8; ++j) {
                int v_col_start = j * 16;
                int v_row = lane % 16;
                int v_col = v_col_start + (lane / 16) * 8;
                uint32_t v_addr = (uint32_t)__cvta_generic_to_shared(&smem_V[(k + v_row) * 128 + v_col]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                             : "=r"(V_regs[j][0]), "=r"(V_regs[j][1]), "=r"(V_regs[j][2]), "=r"(V_regs[j][3])
                             : "r"(v_addr));
            }
        
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 8; ++j) {
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                                 : "+f"(O_acc[i][j*2][0]), "+f"(O_acc[i][j*2][1]), "+f"(O_acc[i][j*2][2]), "+f"(O_acc[i][j*2][3])
                                 : "r"(P_regs[i][0]), "r"(P_regs[i][1]), "r"(P_regs[i][2]), "r"(P_regs[i][3]),
                                   "r"(V_regs[j][0]), "r"(V_regs[j][1]));
                                   
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                                 : "+f"(O_acc[i][j*2+1][0]), "+f"(O_acc[i][j*2+1][1]), "+f"(O_acc[i][j*2+1][2]), "+f"(O_acc[i][j*2+1][3])
                                 : "r"(P_regs[i][0]), "r"(P_regs[i][1]), "r"(P_regs[i][2]), "r"(P_regs[i][3]),
                                   "r"(V_regs[j][2]), "r"(V_regs[j][3]));
                }
            }
        }
        __syncthreads(); 
        kv_phase++;
    }

    float l_i_0 = __shfl_sync(0xffffffff, l_i, lane / 4);
    float l_i_1 = __shfl_sync(0xffffffff, l_i, lane / 4 + 8);
    float l_i_2 = __shfl_sync(0xffffffff, l_i, 16 + lane / 4);
    float l_i_3 = __shfl_sync(0xffffffff, l_i, 16 + lane / 4 + 8);
    
    for (int j = 0; j < 16; ++j) {
        O_acc[0][j][0] /= l_i_0;
        O_acc[0][j][1] /= l_i_0;
        O_acc[0][j][2] /= l_i_1;
        O_acc[0][j][3] /= l_i_1;
        
        O_acc[1][j][0] /= l_i_2;
        O_acc[1][j][1] /= l_i_2;
        O_acc[1][j][2] /= l_i_3;
        O_acc[1][j][3] /= l_i_3;
    }

    __syncthreads();
    for (int i = 0; i < 2; ++i) {
        int smem_r_base = w * 32 + i * 16;
        for (int j = 0; j < 16; ++j) {
            int c_base = j * 8;
            int r0 = lane / 4;
            int r1 = lane / 4 + 8;
            int c0 = (lane % 4) * 2;
            int c1 = (lane % 4) * 2 + 1;
            
            smem_P[(smem_r_base + r0) * 128 + c_base + c0] = __float2bfloat16_impl(O_acc[i][j][0]);
            smem_P[(smem_r_base + r0) * 128 + c_base + c1] = __float2bfloat16_impl(O_acc[i][j][1]);
            smem_P[(smem_r_base + r1) * 128 + c_base + c0] = __float2bfloat16_impl(O_acc[i][j][2]);
            smem_P[(smem_r_base + r1) * 128 + c_base + c1] = __float2bfloat16_impl(O_acc[i][j][3]);
        }
    }
    __syncthreads();

    uint4* smem_vec = (uint4*)smem_P;
    uint4* O_vec = (uint4*)(O + (bz * H * S + by * S + q_start) * 128);
    for (int idx = threadIdx.x; idx < 16384 / 8; idx += 128) {
        int row = idx / 16;
        if (q_start + row < S) {
            O_vec[idx] = smem_vec[idx];
        }
    }

    if (global_row < S) {
        LSE[bz * H * S + by * S + global_row] = m_i + logf(l_i);
    }
    
    __syncthreads();
    if (threadIdx.x < 32) {
        tmem_dealloc_fn(tmem_addr, 128);
    }
}

namespace tvm_ffi_mha {
void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);

    const uint16_t* Q_data = static_cast<const uint16_t*>(Q.data_ptr());
    const uint16_t* K_data = static_cast<const uint16_t*>(K.data_ptr());
    const uint16_t* V_data = static_cast<const uint16_t*>(V.data_ptr());
    uint16_t* O_data = static_cast<uint16_t*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());

    dim3 blocks((S + 127) / 128, H, B);
    dim3 threads(128);

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int smem_size = 131104; 
    CUDA_CHECK(cudaFuncSetAttribute(CausalAttentionBlackwellKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    CausalAttentionBlackwellKernel<<<blocks, threads, smem_size, stream>>>(Q_data, K_data, V_data, O_data, LSE_data, S, H);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha::run);

} // namespace tvm_ffi_mha