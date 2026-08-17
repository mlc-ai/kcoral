#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>
#include <math.h>
#include <stdio.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_fmha {

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
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
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase));
}

__device__ __forceinline__ void fence_proxy_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
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

__device__ __forceinline__ void commit_umma_1sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1"
        ".mbarrier::arrive::one.shared::cta.b64"
        " [%0];"
        :: "r"(a));
}

__device__ __forceinline__ void umma_f16_cg1_acc(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, 1, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc));
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo, uint32_t layout_type) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)layout_type << 61;
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

__device__ __forceinline__ uint32_t make_instr_desc_fn_pv(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= (0u << 15);
    d |= (1u << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__global__ __launch_bounds__(128, 1)
void fmha_4_kernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* LSE,
    int64_t S, int64_t D)
{
    int row_block = blockIdx.x;
    int bh = blockIdx.y;
    int tid = threadIdx.x;

    setmaxnreg_inc_sync_fn<248>();

    __nv_bfloat16* ptr_O_bh = O + bh * S * D;
    float* ptr_LSE_bh = LSE + bh * S;

    int s_start = row_block * 128;

    extern __shared__ char smem_pool[];
    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem_pool);
    uint32_t align_offset = (16 - (smem_addr % 16)) % 16;
    char* aligned_smem = smem_pool + align_offset;

    __nv_bfloat16* Q_smem = (__nv_bfloat16*)aligned_smem;
    __nv_bfloat16* K_smem = Q_smem + 128 * 128;
    __nv_bfloat16* V_smem = K_smem + 128 * 128;
    float* O_smem_f32 = (float*)(V_smem + 128 * 128);
    __nv_bfloat16* O_smem_bf16 = (__nv_bfloat16*)(O_smem_f32 + 128 * 128);
    __nv_bfloat16* P_smem = O_smem_bf16;
    
    uint64_t* mbar = (uint64_t*)(O_smem_bf16 + 128 * 128);

    extern __shared__ __align__(16) uint32_t tmem_pool[];
    uint32_t* tmem_S_ptr = tmem_pool;
    uint32_t* tmem_O_ptr = tmem_pool + 1;
    uint32_t* tmem_P_ptr = tmem_pool + 2;
    
    if (tid == 0) {
        init_smem_barrier_fn(mbar, 1);
        fence_smem_barrier_init_fn();
        tmem_alloc_fn(tmem_S_ptr, 128);
        tmem_alloc_fn(tmem_O_ptr, 128);
        tmem_alloc_fn(tmem_P_ptr, 128);
    }
    __syncthreads();
    
    uint32_t tmem_S = tmem_S_ptr[0];
    uint32_t tmem_O = tmem_O_ptr[0];
    uint32_t tmem_P = tmem_P_ptr[0];

    float m_prev_row[128];
    float l_prev_row[128];
    for (int i = tid; i < 128; i += 128) {
        m_prev_row[i] = -1e20f;
        l_prev_row[i] = 0.0f;
    }
    for (int i = tid; i < 128 * 128; i += 128) {
        O_smem_f32[i] = 0.0f;
    }
    __syncthreads();
    
    uint32_t phase = 0;

    const __nv_bfloat16* ptr_Q_bh = Q + bh * S * D;
    for(int i = tid; i < 128 * 16; i += 128) {
        int row = i / 16;
        int col = (i % 16) * 8;
        float4 val;
        if (s_start + row < S) {
            val = *reinterpret_cast<const float4*>(&ptr_Q_bh[(s_start + row) * D + col]);
        } else {
            val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        *reinterpret_cast<float4*>(&Q_smem[row * 128 + col]) = val;
    }
    
    if (tid == 0) {
        for (int col = 0; col < 128; col += 8) {
            uint32_t r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0, r7 = 0;
            asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%4], {%0,%1,%2,%3,%5,%6,%7};\n"
                :: "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(tmem_O + col),
                   "r"(r4), "r"(r5), "r"(r6), "r"(r7));
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    }
    __syncthreads();

    float scale = 1.0f / sqrtf((float)D);

    const __nv_bfloat16* ptr_K_bh = K + bh * S * D;
    const __nv_bfloat16* ptr_V_bh = V + bh * S * D;

    for (int j = 0; j <= row_block; ++j) {
        int kv_start = j * 128;

        for(int i = tid; i < 128 * 16; i += 128) {
            int row = i / 16;
            int col = (i % 16) * 8;
            float4 k_val, v_val;
            if (kv_start + row < S) {
                k_val = *reinterpret_cast<const float4*>(&ptr_K_bh[(kv_start + row) * D + col]);
                v_val = *reinterpret_cast<const float4*>(&ptr_V_bh[(kv_start + row) * D + col]);
            } else {
                k_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                v_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
            *reinterpret_cast<float4*>(&K_smem[row * 128 + col]) = k_val;
            *reinterpret_cast<float4*>(&V_smem[row * 128 + col]) = v_val;
        }
        __syncthreads();

        uint32_t col_offset = 0;
        for(int k = 0; k < 128; k += 32) { 
            for(int n = 0; n < 128; n += 32) { 
                __nv_bfloat16* q_ptr = Q_smem + k;
                __nv_bfloat16* k_ptr = K_smem + k;
                
                uint64_t desc_Q_k = make_smem_desc_sm100_fn(q_ptr, 2048, 128, 0);
                uint64_t desc_K_k = make_smem_desc_sm100_fn(k_ptr, 2048, 128, 0);
                uint32_t idesc_QK = make_instr_desc_fn(128, 128);
                
                umma_f16_cg1_acc(tmem_S + col_offset, desc_Q_k, desc_K_k, idesc_QK);
                col_offset += 32;
            }
        }
        
        if (tid == 0) {
            commit_umma_1sm_fn(mbar);
        }
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;

        float thread_max = -1e20f;
        float S_reg[4][8]; 
        
        uint32_t col_base = 0;
        do {
            uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),
                 "=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(tmem_S + col_base));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            int global_row = s_start + tid;
            S_reg[0][0] = __uint_as_float(r0) * scale;
            S_reg[0][1] = __uint_as_float(r1) * scale;
            S_reg[0][2] = __uint_as_float(r2) * scale;
            S_reg[0][3] = __uint_as_float(r3) * scale;
            S_reg[0][4] = __uint_as_float(r4) * scale;
            S_reg[0][5] = __uint_as_float(r5) * scale;
            S_reg[0][6] = __uint_as_float(r6) * scale;
            S_reg[0][7] = __uint_as_float(r7) * scale;
            
            col_base += 8;
            int col = col_base - 8;
            int global_col = kv_start + col;
            for(int c = 0; c < 8; ++c) {
                if (global_col + c > global_row || global_col + c >= S) {
                    S_reg[0][c] = -1e20f;
                }
                thread_max = fmaxf(thread_max, S_reg[0][c]);
            }
        } while (col_base < 128);

        float m_new = fmaxf(m_prev_row[tid], thread_max);
        float temp_l = 0.0f;
        
        for (int c = 0; c < 8; ++c) {
            float p = (S_reg[0][c] <= -1e20f) ? 0.0f : __expf(S_reg[0][c] - m_new);
            temp_l += p;
            S_reg[0][c] = p;
        }
        
        float factor = 1.0f;
        if (m_prev_row[tid] <= -1e19f) {
            m_prev_row[tid] = m_new;
            l_prev_row[tid] = temp_l;
        } else {
            factor = __expf(m_prev_row[tid] - m_new);
            m_prev_row[tid] = m_new;
            l_prev_row[tid] = l_prev_row[tid] * factor + temp_l;
        }
        
        for (int c = 0; c < 8; ++c) {
            S_reg[0][c] *= factor;
        }
        
        for (int c = 0; c < 8; ++c) {
            int col = col_base - 8 + c;
            int chunk = col / 8;
            int swizzled_chunk = (tid % 8) ^ chunk;
            int swizzled_col = swizzled_chunk * 8 + (col % 8);
            P_smem[tid * 128 + swizzled_col] = __float2bfloat16(S_reg[0][c]);
        }
        
        if (factor != 1.0f) {
            for(int c = tid; c < 128 * 128; c += 128) {
                O_smem_f32[c] *= factor;
            }
        }
        __syncthreads();
        
        float O_reg[4];
        for(int col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_O + col));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            O_reg[0] = __uint_as_float(r0);
            O_reg[1] = __uint_as_float(r1);
            O_reg[2] = __uint_as_float(r2);
            O_reg[3] = __uint_as_float(r3);
            
            if (factor != 1.0f) {
                O_reg[0] *= factor;
                O_reg[1] *= factor;
                O_reg[2] *= factor;
                O_reg[3] *= factor;
            }
            
            O_smem_f32[tid * 128 + col] = O_reg[0];
            O_smem_f32[tid * 128 + col + 1] = O_reg[1];
            O_smem_f32[tid * 128 + col + 2] = O_reg[2];
            O_smem_f32[tid * 128 + col + 3] = O_reg[3];
        }
        __syncthreads();
        
        fence_proxy_async_shared_fn();
        
        uint32_t row_offset = 0;
        for (int k = 0; k < 128; k += 32) {
            __nv_bfloat16* p_ptr = P_smem + k;
            __nv_bfloat16* v_ptr = V_smem + k * 128; 
            
            uint64_t desc_P_k = make_smem_desc_sm100_fn(p_ptr, 2048, 128, 0);
            uint64_t desc_V_k = make_smem_desc_sm100_fn(v_ptr, 128, 2048, 0);
            uint32_t idesc_PV = make_instr_desc_fn_pv(128, 128);
            
            umma_f16_cg1_acc(tmem_O + row_offset, desc_P_k, desc_V_k, idesc_PV);
            row_offset += 32;
        }
        
        if (tid == 0) {
            commit_umma_1sm_fn(mbar);
        }
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        
        for (int col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_O + col));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float f0 = __uint_as_float(r0);
            float f1 = __uint_as_float(r1);
            float f2 = __uint_as_float(r2);
            float f3 = __uint_as_float(r3);
            
            O_smem_f32[tid * 128 + col] += f0;
            O_smem_f32[tid * 128 + col + 1] += f1;
            O_smem_f32[tid * 128 + col + 2] += f2;
            O_smem_f32[tid * 128 + col + 3] += f3;
        }
        
        __syncthreads();
    }

    __syncthreads();
    
    for (int i = tid; i < 128 * 128; i += 128) {
        int row = i / 128;
        int col = i % 128;
        float val = O_smem_f32[i];
        if (l_prev_row[row] > 0.0f) {
            val /= l_prev_row[row];
        }
        O_smem_bf16[row * 128 + col] = __float2bfloat16(val);
    }
    __syncthreads();
    
    uint32_t warp_id = tid / 32;
    uint32_t lane_id = tid % 32;
    uint32_t num_steps = (128 + 3) / 4;
    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 4 + warp_id;
        if (row >= 128) continue;
        uint32_t global_row = s_start + row;
        uint32_t col_start = lane_id * 4;
        uint32_t global_col = col_start;
        if (global_row < S && global_col + 3 < 128) {
            uint2 data = *reinterpret_cast<uint2*>(&O_smem_bf16[row * 128 + col_start]);
            *reinterpret_cast<uint2*>(ptr_O_bh + (uint64_t)global_row * D + global_col) = data;
        }
    }
    
    if (tid < 128 && s_start + tid < S) {
        ptr_LSE_bh[s_start + tid] = m_prev_row[tid] + logf(l_prev_row[tid]);
    }

    if (tid == 0) {
        tmem_dealloc_fn(tmem_S, 128);
        tmem_dealloc_fn(tmem_O, 128);
        tmem_dealloc_fn(tmem_P, 128);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_ptr = static_cast<float*>(LSE.data_ptr());
    
    int64_t num_row_blocks = (S + 127) / 128;
    dim3 grid(num_row_blocks, B * H);
    dim3 block(128);
    
    int smem_size = 16 + 4 * 128 * 128 * sizeof(__nv_bfloat16) + 128 * 128 * sizeof(float) + 256;
                    
    CUDA_CHECK(cudaFuncSetAttribute(
        fmha_4_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    ));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    fmha_4_kernel<<<grid, block, smem_size, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr, S, D
    );
    CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_fmha