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

#ifndef M_LOG2E
#define M_LOG2E 1.4426950408889634f
#endif

CUresult create_tma_4d_descriptor_none(CUtensorMap* d, void* globalAddress, 
    uint64_t D, uint64_t S, uint64_t H, uint64_t B, 
    uint32_t smem_D, uint32_t smem_S) {
    cuuint64_t globalDim[4] = {D, S, H, B};
    cuuint64_t globalStrides[3] = {D*2, D*S*2, D*S*H*2};
    cuuint32_t boxDim[4] = {smem_D, smem_S, 1, 1};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, globalAddress, globalDim, globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
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

__device__ __forceinline__ void tmem_alloc_cg1_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_cg1_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tmem_relinquish_cg1_fn() {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_before_fn() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
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

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t offset_bytes, uint32_t lbo, uint32_t sbo, uint32_t swizzle_mode) {
    uint64_t d = 0;
    uint32_t addr = ((uint32_t)__cvta_generic_to_shared(smem_ptr) + offset_bytes);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)swizzle_mode << 61;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, uint32_t a_major, uint32_t b_major) {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= (a_major << 15);
    d |= (b_major << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ void umma_qk_fn(uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc, uint32_t idesc, int accumulate) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(accumulate));
}

__device__ __forceinline__ void umma_pv_fn(uint32_t d_tmem, uint32_t a_tmem, uint64_t b_desc, uint32_t idesc, int accumulate) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}\n"
        :: "r"(d_tmem), "r"(a_tmem), "l"(b_desc), "r"(idesc), "r"(accumulate));
}

__device__ __forceinline__ void umma_commit_cg1_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1"
        ".mbarrier::arrive::one.shared::cluster.b64"
        " [%0];"
        :: "r"(a));
}

struct SharedStorage {
    alignas(1024) __nv_bfloat16 Q[128][128];
    union {
        struct {
            alignas(1024) __nv_bfloat16 K[2][128][128];
            alignas(1024) __nv_bfloat16 V[2][128][128];
        };
        alignas(1024) __nv_bfloat16 smem_out[128][128];
    };
    alignas(8) uint64_t mbar_q[1];
    alignas(8) uint64_t mbar_k[2][1];
    alignas(8) uint64_t mbar_v[2][1];
    alignas(8) uint64_t mbar_mma[1];
};


__global__ __launch_bounds__(128) void attention_kernel(
    __nv_bfloat16* Q_ptr, __nv_bfloat16* K_ptr, __nv_bfloat16* V_ptr,
    __nv_bfloat16* O_ptr, float* LSE_ptr,
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    uint32_t seq_len, uint32_t B, uint32_t H) 
{
    uint32_t m_block = blockIdx.x;
    uint32_t head_idx = blockIdx.y;
    uint32_t batch_idx = blockIdx.z;

    extern __shared__ __align__(1024) uint8_t smem_buf[];
    SharedStorage* smem = (SharedStorage*)smem_buf;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&smem->mbar_q[0], 1);
        init_smem_barrier_fn(&smem->mbar_k[0][0], 1);
        init_smem_barrier_fn(&smem->mbar_k[1][0], 1);
        init_smem_barrier_fn(&smem->mbar_v[0][0], 1);
        init_smem_barrier_fn(&smem->mbar_v[1][0], 1);
        init_smem_barrier_fn(&smem->mbar_mma[0], 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    setmaxnreg_inc_sync_fn<248>();

    __shared__ uint32_t tmem_base;
    if (threadIdx.x < 32) {
        tmem_alloc_cg1_fn(&tmem_base, 512);
    }
    __syncthreads();
    
    uint32_t S_col = tmem_base;
    uint32_t O_col = tmem_base + 128;
    uint32_t P_col = tmem_base + 256;

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&smem->mbar_q[0], 128 * 128 * 2);
        tma_load_4d_fn(&tma_Q, &smem->mbar_q[0], smem->Q, 0, m_block * 128, head_idx, batch_idx);
    }

    uint32_t n_blocks = (seq_len + 127) / 128;
    float m_i = -INFINITY;
    float l_i = 0.0f;
    float scale = 1.0f / sqrtf(128.0f);

    for (int i = 0; i < 128; i += 4) {
        tmem_store_4x_fn(O_col + i, 0, 0, 0, 0);
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");

    mbarrier_wait_fn(&smem->mbar_q[0], 0);

    int pipe_idx = 0;
    if (threadIdx.x == 0 && n_blocks > 0) {
        mbarrier_arrive_and_expect_tx_fn(&smem->mbar_k[0][0], 128 * 128 * 2);
        tma_load_4d_fn(&tma_K, &smem->mbar_k[0][0], smem->K[0], 0, 0, head_idx, batch_idx);
        
        mbarrier_arrive_and_expect_tx_fn(&smem->mbar_v[0][0], 128 * 128 * 2);
        tma_load_4d_fn(&tma_V, &smem->mbar_v[0][0], smem->V[0], 0, 0, head_idx, batch_idx);
    }

    uint32_t idesc_qk = make_instr_desc_fn(128, 128, 0, 0);
    uint32_t idesc_pv = make_instr_desc_fn(128, 128, 0, 1);

    for (uint32_t n = 0; n < n_blocks; ++n) {
        mbarrier_wait_fn(&smem->mbar_k[pipe_idx][0], (n / 2) & 1);
        mbarrier_wait_fn(&smem->mbar_v[pipe_idx][0], (n / 2) & 1);
        __syncthreads();
        tcgen05_fence_after_fn();
        
        if (threadIdx.x == 0 && n + 1 < n_blocks) {
            int next_pipe = 1 - pipe_idx;
            mbarrier_arrive_and_expect_tx_fn(&smem->mbar_k[next_pipe][0], 128 * 128 * 2);
            tma_load_4d_fn(&tma_K, &smem->mbar_k[next_pipe][0], smem->K[next_pipe], 0, (n + 1) * 128, head_idx, batch_idx);
            
            mbarrier_arrive_and_expect_tx_fn(&smem->mbar_v[next_pipe][0], 128 * 128 * 2);
            tma_load_4d_fn(&tma_V, &smem->mbar_v[next_pipe][0], smem->V[next_pipe], 0, (n + 1) * 128, head_idx, batch_idx);
        }
        
        if (threadIdx.x == 0) {
            for (int k = 0; k < 8; ++k) {
                uint64_t desc_q_k = make_smem_desc_sm100_fn(smem->Q, k * 32, 32768, 2048, 0);
                uint64_t desc_k_k = make_smem_desc_sm100_fn(smem->K[pipe_idx], k * 32, 32768, 2048, 0);
                umma_qk_fn(S_col, desc_q_k, desc_k_k, idesc_qk, (k > 0 ? 1 : 0));
            }
            umma_commit_cg1_fn(&smem->mbar_mma[0]);
        }
        mbarrier_wait_fn(&smem->mbar_mma[0], (n * 2) & 1);
        __syncthreads();
        tcgen05_fence_after_fn();
        
        float m_ij = -INFINITY;
        float s[128];
        for (int i = 0; i < 128; i += 8) {
            tmem_load_8x_fn(S_col + i, (uint32_t*)&s[i], (uint32_t*)&s[i+1], (uint32_t*)&s[i+2], (uint32_t*)&s[i+3], 
                                       (uint32_t*)&s[i+4], (uint32_t*)&s[i+5], (uint32_t*)&s[i+6], (uint32_t*)&s[i+7]);
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        uint32_t k_offset = n * 128;
        for (int i = 0; i < 128; ++i) {
            if (k_offset + i >= seq_len) {
                s[i] = -INFINITY;
            } else {
                s[i] *= scale;
            }
            m_ij = fmaxf(m_ij, s[i]);
        }
        
        float m_new = fmaxf(m_i, m_ij);
        float exp_diff = fast_exp2f_fn((m_i - m_new) * M_LOG2E);
        l_i = l_i * exp_diff;
        
        float l_ij = 0.0f;
        for (int i = 0; i < 128; ++i) {
            s[i] = fast_exp2f_fn((s[i] - m_new) * M_LOG2E);
            l_ij += s[i];
        }
        l_i += l_ij;
        
        if (m_new > m_i) {
            for (int i = 0; i < 128; i += 8) {
                uint32_t o_r[8];
                tmem_load_8x_fn(O_col + i, &o_r[0], &o_r[1], &o_r[2], &o_r[3], &o_r[4], &o_r[5], &o_r[6], &o_r[7]);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                for (int k = 0; k < 8; ++k) {
                    float o = __uint_as_float(o_r[k]) * exp_diff;
                    o_r[k] = __float_as_uint(o);
                }
                tmem_store_4x_fn(O_col + i, o_r[0], o_r[1], o_r[2], o_r[3]);
                tmem_store_4x_fn(O_col + i + 4, o_r[4], o_r[5], o_r[6], o_r[7]);
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        }
        m_i = m_new;
        
        for (int i = 0; i < 128; i += 8) {
            uint32_t p0 = pack_bf16_fn(__float_as_uint(s[i+0]), __float_as_uint(s[i+1]));
            uint32_t p1 = pack_bf16_fn(__float_as_uint(s[i+2]), __float_as_uint(s[i+3]));
            uint32_t p2 = pack_bf16_fn(__float_as_uint(s[i+4]), __float_as_uint(s[i+5]));
            uint32_t p3 = pack_bf16_fn(__float_as_uint(s[i+6]), __float_as_uint(s[i+7]));
            tmem_store_4x_fn(P_col + (i / 2), p0, p1, p2, p3);
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        
        tcgen05_fence_before_fn();
        __syncthreads();
        tcgen05_fence_after_fn();
        
        if (threadIdx.x == 0) {
            for (int k = 0; k < 8; ++k) {
                uint32_t a_tmem = P_col + k * 8;
                uint64_t desc_v_k = make_smem_desc_sm100_fn(smem->V[pipe_idx], k * 4096, 2048, 32768, 0);
                umma_pv_fn(O_col, a_tmem, desc_v_k, idesc_pv, 1);
            }
            umma_commit_cg1_fn(&smem->mbar_mma[0]);
        }
        mbarrier_wait_fn(&smem->mbar_mma[0], (n * 2 + 1) & 1);
        __syncthreads();
        tcgen05_fence_after_fn();
        
        pipe_idx = 1 - pipe_idx;
    }

    if (n_blocks > 0) {
        float inv_l = 1.0f / l_i;
        for (int i = 0; i < 128; i += 8) {
            uint32_t o_r[8];
            tmem_load_8x_fn(O_col + i, &o_r[0], &o_r[1], &o_r[2], &o_r[3], &o_r[4], &o_r[5], &o_r[6], &o_r[7]);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            for (int k = 0; k < 8; ++k) {
                float o = __uint_as_float(o_r[k]) * inv_l;
                o_r[k] = __float_as_uint(o);
            }
            tmem_store_4x_fn(O_col + i, o_r[0], o_r[1], o_r[2], o_r[3]);
            tmem_store_4x_fn(O_col + i + 4, o_r[4], o_r[5], o_r[6], o_r[7]);
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    }

    tcgen05_fence_before_fn();
    __syncthreads();
    tcgen05_fence_after_fn();

    for (int col = 0; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
       : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(O_col + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        smem->smem_out[threadIdx.x][col + 0] = __float2bfloat16(__uint_as_float(r0));
        smem->smem_out[threadIdx.x][col + 1] = __float2bfloat16(__uint_as_float(r1));
        smem->smem_out[threadIdx.x][col + 2] = __float2bfloat16(__uint_as_float(r2));
        smem->smem_out[threadIdx.x][col + 3] = __float2bfloat16(__uint_as_float(r3));
    }
    __syncthreads();

    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t num_steps = 128 / 4;
    __nv_bfloat16* out_ptr = O_ptr + (batch_idx * H + head_idx) * seq_len * 128;

    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 4 + warp_id;
        uint32_t global_row = m_block * 128 + row;
        uint32_t col_start = lane_id * 4;
        
        if (global_row < seq_len) {
            uint2 data = *reinterpret_cast<uint2*>(&smem->smem_out[row][col_start]);
            *reinterpret_cast<uint2*>(out_ptr + global_row * 128 + col_start) = data;
        }
    }

    if (threadIdx.x < 128) {
        uint32_t row = m_block * 128 + threadIdx.x;
        if (row < seq_len) {
            LSE_ptr[(batch_idx * H + head_idx) * seq_len + row] = m_i + logf(l_i);
        }
    }

    if (threadIdx.x < 32) {
        tmem_dealloc_cg1_fn(tmem_base, 512);
        tmem_relinquish_cg1_fn();
    }
}

namespace tvm_ffi_cuda {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B_val = Q.size(0);
    int64_t H_val = Q.size(1);
    int64_t S_val = Q.size(2);
    
    __nv_bfloat16* q_ptr = static_cast<__nv_bfloat16*>(Q.data_ptr());
    __nv_bfloat16* k_ptr = static_cast<__nv_bfloat16*>(K.data_ptr());
    __nv_bfloat16* v_ptr = static_cast<__nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_ptr = static_cast<float*>(LSE.data_ptr());
    
    CUtensorMap tma_Q, tma_K, tma_V;
    
    CUresult res;
    res = create_tma_4d_descriptor_none(&tma_Q, q_ptr, 128, S_val, H_val, B_val, 128, 128);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA Q failed\n"); exit(1); }
    
    res = create_tma_4d_descriptor_none(&tma_K, k_ptr, 128, S_val, H_val, B_val, 128, 128);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA K failed\n"); exit(1); }
    
    res = create_tma_4d_descriptor_none(&tma_V, v_ptr, 128, S_val, H_val, B_val, 128, 128);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA V failed\n"); exit(1); }
    
    dim3 grid((S_val + 127) / 128, H_val, B_val);
    dim3 block(128);
    int smem_size = sizeof(SharedStorage);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUDA_CHECK(cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    attention_kernel<<<grid, block, smem_size, stream>>>(q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr, tma_Q, tma_K, tma_V, S_val, B_val, H_val);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_cuda::run);

} // namespace tvm_ffi_cuda