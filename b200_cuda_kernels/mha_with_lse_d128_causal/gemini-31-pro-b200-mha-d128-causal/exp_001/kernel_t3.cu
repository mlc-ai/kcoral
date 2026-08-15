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
        fprintf(stderr, "CUDA Driver error %d at %s:%d\n",         \
                _e, __FILE__, __LINE__);                           \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_example_cuda {

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tmem_alloc_cg1_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a), "r"(ncols) : "memory");
}

__device__ __forceinline__ void tmem_dealloc_cg1_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(addr), "r"(ncols) : "memory");
}

__device__ __forceinline__ void tmem_load_8x_fn(uint32_t col,
    uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3,
    uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3),
     "=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7) : "r"(col));
}

__device__ __forceinline__ void tmem_store_4x_fn(uint32_t col, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
        :: "r"(col), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_store_wait_fn() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
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

__device__ __forceinline__ uint64_t make_smem_desc(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint64_t d = 0;
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46; 
    d |= (uint64_t)2 << 61; // 128-Byte swizzling
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc(uint32_t M, uint32_t N, int transpose_a, int transpose_b) {
    uint32_t d = 0;
    d |= (1u << 4);    // c_format = FP32
    d |= (1u << 7);    // a_format = BF16
    d |= (1u << 10);   // b_format = BF16
    d |= ((uint32_t)transpose_a << 15);
    d |= ((uint32_t)transpose_b << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ uint64_t advance_smem_desc(uint64_t desc, uint32_t bytes) {
    uint32_t current_addr = (desc & 0x3FFF) << 4; 
    current_addr += bytes;
    desc &= ~0x3FFFull; 
    desc |= (current_addr >> 4);
    
    uint32_t base_offset = (current_addr >> 7) & 0x7;
    desc &= ~(0x7ull << 49);
    desc |= ((uint64_t)base_offset << 49);
    
    return desc;
}

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

__global__ void __launch_bounds__(128, 1) mha_causal_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O_ptr,
    float* LSE_ptr,
    int S, int D, int H)
{
    int q_tile_idx = blockIdx.x;
    int h_idx = blockIdx.y;
    int b_idx = blockIdx.z;
    int q_start = q_tile_idx * 128;
    int q_pos = q_start + threadIdx.x;
    uint32_t batch_offset = b_idx * H * S + h_idx * S;

    extern __shared__ __align__(128) uint8_t smem_pool[];
    __nv_bfloat16* Q_smem_0   = (__nv_bfloat16*)smem_pool;           // 16KB
    __nv_bfloat16* Q_smem_1   = Q_smem_0 + 8192;                     // 16KB
    __nv_bfloat16* K_smem_0_0 = Q_smem_1 + 8192;                     // 16KB
    __nv_bfloat16* K_smem_0_1 = K_smem_0_0 + 8192;                   // 16KB
    __nv_bfloat16* K_smem_1_0 = K_smem_0_1 + 8192;                   // 16KB
    __nv_bfloat16* K_smem_1_1 = K_smem_1_0 + 8192;                   // 16KB
    __nv_bfloat16* V_smem_0_0 = K_smem_1_1 + 8192;                   // 16KB
    __nv_bfloat16* V_smem_0_1 = V_smem_0_0 + 8192;                   // 16KB
    __nv_bfloat16* V_smem_1_0 = V_smem_0_1 + 8192;                   // 16KB
    __nv_bfloat16* V_smem_1_1 = V_smem_1_0 + 8192;                   // 16KB
    __nv_bfloat16* P_smem_0   = V_smem_1_1 + 8192;                   // 16KB
    __nv_bfloat16* P_smem_1   = P_smem_0 + 8192;                     // 16KB

    __nv_bfloat16* K_smems[2][2] = {{K_smem_0_0, K_smem_0_1}, {K_smem_1_0, K_smem_1_1}};
    __nv_bfloat16* V_smems[2][2] = {{V_smem_0_0, V_smem_0_1}, {V_smem_1_0, V_smem_1_1}};

    __shared__ uint32_t shared_tmem_s, shared_tmem_o;
    if (threadIdx.x < 32) {
        tmem_alloc_cg1_fn(&shared_tmem_s, 128);
        tmem_alloc_cg1_fn(&shared_tmem_o, 128);
    }
    __syncthreads();
    uint32_t tmem_s_addr = shared_tmem_s;
    uint32_t tmem_o_addr = shared_tmem_o;

    uint32_t Regs[128]; 

    __shared__ uint64_t tma_bar_Q[1];
    __shared__ uint64_t tma_bar_KV[2];
    __shared__ uint64_t umma_bar[1];
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(tma_bar_Q, 1);
        init_smem_barrier_fn(&tma_bar_KV[0], 1);
        init_smem_barrier_fn(&tma_bar_KV[1], 1);
        init_smem_barrier_fn(umma_bar, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(tma_bar_Q, 2 * 64 * 128 * 2);
        tma_load_2d_fn(&tma_Q, tma_bar_Q, Q_smem_0, 0, batch_offset + q_start);
        tma_load_2d_fn(&tma_Q, tma_bar_Q, Q_smem_1, 64, batch_offset + q_start);
    }
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&tma_bar_KV[0], 4 * 64 * 128 * 2);
        tma_load_2d_fn(&tma_K, &tma_bar_KV[0], K_smems[0][0], 0, batch_offset + 0);
        tma_load_2d_fn(&tma_K, &tma_bar_KV[0], K_smems[0][1], 64, batch_offset + 0);
        tma_load_2d_fn(&tma_V, &tma_bar_KV[0], V_smems[0][0], 0, batch_offset + 0);
        tma_load_2d_fn(&tma_V, &tma_bar_KV[0], V_smems[0][1], 64, batch_offset + 0);
    }

    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    float scale = 0.08838834764f; 
    
    int phase_KV[2] = {0, 0};
    int umma_phase = 0;
    
    mbarrier_wait_fn(tma_bar_Q, 0); 
    __syncthreads();
    
    int max_k_idx = q_start / 128;
    for (int k_idx = 0; k_idx <= max_k_idx; k_idx++) {
        int buf_idx = k_idx % 2;
        int next_buf_idx = (k_idx + 1) % 2;
        int k_start = k_idx * 128;
        
        if (k_idx < max_k_idx) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&tma_bar_KV[next_buf_idx], 4 * 64 * 128 * 2);
                tma_load_2d_fn(&tma_K, &tma_bar_KV[next_buf_idx], K_smems[next_buf_idx][0], 0, batch_offset + k_start + 128);
                tma_load_2d_fn(&tma_K, &tma_bar_KV[next_buf_idx], K_smems[next_buf_idx][1], 64, batch_offset + k_start + 128);
                tma_load_2d_fn(&tma_V, &tma_bar_KV[next_buf_idx], V_smems[next_buf_idx][0], 0, batch_offset + k_start + 128);
                tma_load_2d_fn(&tma_V, &tma_bar_KV[next_buf_idx], V_smems[next_buf_idx][1], 64, batch_offset + k_start + 128);
            }
        }
        
        mbarrier_wait_fn(&tma_bar_KV[buf_idx], phase_KV[buf_idx]);
        phase_KV[buf_idx] ^= 1;
        __syncthreads();
        
        fence_async_shared_fn(); 
        
        if (threadIdx.x == 0) {
            uint64_t desc_Q_0 = make_smem_desc(Q_smem_0, 1, 1024);
            uint64_t desc_Q_1 = make_smem_desc(Q_smem_1, 1, 1024);
            uint64_t desc_K_0 = make_smem_desc(K_smems[buf_idx][0], 1, 1024);
            uint64_t desc_K_1 = make_smem_desc(K_smems[buf_idx][1], 1, 1024);
            uint32_t idesc_QK = make_instr_desc(128, 128, 0, 0); 
            
            uint64_t cur_desc_Q = desc_Q_0;
            uint64_t cur_desc_K = desc_K_0;
            for (int k_step = 0; k_step < 128; k_step += 16) {
                if (k_step == 64) {
                    cur_desc_Q = desc_Q_1;
                    cur_desc_K = desc_K_1;
                }
                umma_f16_cg1_fn(tmem_s_addr, cur_desc_Q, cur_desc_K, idesc_QK, (k_step == 0) ? 0 : 1);
                cur_desc_Q = advance_smem_desc(cur_desc_Q, 32);
                cur_desc_K = advance_smem_desc(cur_desc_K, 32);
            }
            uint32_t a = (uint32_t)__cvta_generic_to_shared(&umma_bar[0]);
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(a));
        }
        mbarrier_wait_fn(umma_bar, umma_phase);
        umma_phase ^= 1;
        __syncthreads();
        
        for (int c = 0; c < 128; c += 8) {
            tmem_load_8x_fn(tmem_s_addr + c, 
                &Regs[c], &Regs[c+1], &Regs[c+2], &Regs[c+3],
                &Regs[c+4], &Regs[c+5], &Regs[c+6], &Regs[c+7]);
        }
        tmem_load_fence_fn();
        
        float m_new = m_prev;
        for (int c = 0; c < 128; c++) {
            float val = __uint_as_float(Regs[c]) * scale;
            int k_pos = k_start + c;
            if (k_pos > q_pos || k_pos >= S || q_pos >= S) {
                val = -INFINITY;
            }
            m_new = fmaxf(m_new, val);
        }
        
        float alpha = 0.0f;
        if (m_prev == -INFINITY) {
            alpha = 0.0f;
        } else if (m_new != -INFINITY) {
            alpha = fast_exp2f_fn((m_prev - m_new) * 1.44269504089f);
        }
        
        float row_sum = 0.0f;
        for (int c = 0; c < 128; c++) {
            float val = __uint_as_float(Regs[c]) * scale;
            int k_pos = k_start + c;
            if (k_pos > q_pos || k_pos >= S || q_pos >= S) {
                val = -INFINITY;
            }
            float p = 0.0f;
            if (val != -INFINITY) {
                p = fast_exp2f_fn((val - m_new) * 1.44269504089f);
            }
            row_sum += p;
            
            __nv_bfloat16* P_ptr = (c < 64) ? P_smem_0 : P_smem_1;
            int local_c = c % 64;
            int span_idx = local_c / 8;
            int offset = local_c % 8;
            int swizzled_span = (threadIdx.x % 8) ^ span_idx;
            P_ptr[threadIdx.x * 64 + swizzled_span * 8 + offset] = __float2bfloat16(p);
        }
        
        float l_new = l_prev * alpha + row_sum;
        
        if (k_start > 0) {
            for (int c = 0; c < 128; c += 8) {
                tmem_load_8x_fn(tmem_o_addr + c, 
                    &Regs[c], &Regs[c+1], &Regs[c+2], &Regs[c+3],
                    &Regs[c+4], &Regs[c+5], &Regs[c+6], &Regs[c+7]);
            }
            tmem_load_fence_fn();
            for (int c = 0; c < 128; c += 4) {
                float o0 = __uint_as_float(Regs[c+0]) * alpha;
                float o1 = __uint_as_float(Regs[c+1]) * alpha;
                float o2 = __uint_as_float(Regs[c+2]) * alpha;
                float o3 = __uint_as_float(Regs[c+3]) * alpha;
                tmem_store_4x_fn(tmem_o_addr + c, __float_as_uint(o0), __float_as_uint(o1), __float_as_uint(o2), __float_as_uint(o3));
            }
            tmem_store_wait_fn();
        }
        
        m_prev = m_new;
        l_prev = l_new;
        
        __syncthreads();
        fence_async_shared_fn(); 
        
        if (threadIdx.x == 0) {
            uint64_t desc_P_0 = make_smem_desc(P_smem_0, 1, 1024);
            uint64_t desc_P_1 = make_smem_desc(P_smem_1, 1, 1024);
            uint64_t desc_V_0 = make_smem_desc(V_smems[buf_idx][0], 16384, 1024);
            uint64_t desc_V_1 = make_smem_desc(V_smems[buf_idx][1], 16384, 1024);
            uint32_t idesc_PV = make_instr_desc(128, 64, 0, 1);
            
            int accum_o_base = (k_start > 0) ? 1 : 0;
            
            uint64_t cur_desc_P = desc_P_0;
            uint64_t cur_desc_V = desc_V_0;
            for (int k_step = 0; k_step < 128; k_step += 16) {
                if (k_step == 64) cur_desc_P = desc_P_1;
                umma_f16_cg1_fn(tmem_o_addr, cur_desc_P, cur_desc_V, idesc_PV, (k_step == 0 && accum_o_base == 0) ? 0 : 1);
                cur_desc_P = advance_smem_desc(cur_desc_P, 32);
                cur_desc_V = advance_smem_desc(cur_desc_V, 2048);
            }
            
            cur_desc_P = desc_P_0;
            cur_desc_V = desc_V_1;
            for (int k_step = 0; k_step < 128; k_step += 16) {
                if (k_step == 64) cur_desc_P = desc_P_1;
                umma_f16_cg1_fn(tmem_o_addr + 64, cur_desc_P, cur_desc_V, idesc_PV, (k_step == 0 && accum_o_base == 0) ? 0 : 1);
                cur_desc_P = advance_smem_desc(cur_desc_P, 32);
                cur_desc_V = advance_smem_desc(cur_desc_V, 2048);
            }
            
            uint32_t a = (uint32_t)__cvta_generic_to_shared(&umma_bar[0]);
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(a));
        }
        mbarrier_wait_fn(umma_bar, umma_phase);
        umma_phase ^= 1;
        __syncthreads();
    }
    
    for (int c = 0; c < 128; c += 8) {
        tmem_load_8x_fn(tmem_o_addr + c, 
            &Regs[c], &Regs[c+1], &Regs[c+2], &Regs[c+3],
            &Regs[c+4], &Regs[c+5], &Regs[c+6], &Regs[c+7]);
    }
    tmem_load_fence_fn();
    
    __nv_bfloat16* O_stage = Q_smem_0; 
    
    if (q_pos < S) {
        float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
        for (int c = 0; c < 128; c++) {
            float o = __uint_as_float(Regs[c]) * inv_l;
            O_stage[threadIdx.x * 128 + c] = __float2bfloat16(o);
        }
        float lse = m_prev + logf(l_prev);
        LSE_ptr[b_idx * H * S + h_idx * S + q_pos] = lse;
    }
    __syncthreads();
    
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint64_t base_out = (b_idx * H * S + h_idx * S) * D;
    for (int row = warp_id; row < 128; row += 4) {
        int out_q_pos = q_start + row;
        if (out_q_pos < S) {
            for (int c = lane_id * 4; c < 128; c += 128) {
                uint2 data = *reinterpret_cast<uint2*>(&O_stage[row * 128 + c]);
                *reinterpret_cast<uint2*>(O_ptr + base_out + out_q_pos * D + c) = data;
            }
        }
    }
    
    if (threadIdx.x < 32) {
        tmem_dealloc_cg1_fn(tmem_s_addr, 128);
        tmem_dealloc_cg1_fn(tmem_o_addr, 128);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    uint64_t outer_dim = B * H * S;

    const void* q_ptr = Q.data_ptr();
    const void* k_ptr = K.data_ptr();
    const void* v_ptr = V.data_ptr();
    __nv_bfloat16* o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_ptr = static_cast<float*>(LSE.data_ptr());

    CUtensorMap tma_Q, tma_K, tma_V;
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_Q, const_cast<void*>(q_ptr), 128, outer_dim, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_K, const_cast<void*>(k_ptr), 128, outer_dim, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_V, const_cast<void*>(v_ptr), 128, outer_dim, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    int blocks_s = (S + 127) / 128;
    dim3 grid(blocks_s, H, B);
    dim3 block(128);
    int smem_size = 192 * 1024;
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUDA_CHECK(cudaFuncSetAttribute(mha_causal_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    mha_causal_kernel<<<grid, block, smem_size, stream>>>(tma_Q, tma_K, tma_V, o_ptr, lse_ptr, S, D, H);
    CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda