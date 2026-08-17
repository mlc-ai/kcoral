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
        fprintf(stderr, "CU error %d at %s:%d\n",                 \
                _e, __FILE__, __LINE__);                           \
        exit(1);                                                   \
    }                                                              \
} while(0)

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float exp_f(float x) {
    return fast_exp2f_fn(x * 1.44269504f);
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

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
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

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t tmem_addr, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(tmem_addr));
}

__device__ __forceinline__ void tmem_load_8x_fn(uint32_t tmem_addr, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3, uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
        : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3),"=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7) : "r"(tmem_addr));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;" ::: "memory");
}

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
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

__device__ __forceinline__ void umma_commit_1sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1"
        ".mbarrier::arrive::one.shared::cta.b64"
        " [%0];"
        :: "r"(a));
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, bool a_major, bool b_major) {
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

__global__ void mha_kernel_sm100(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O,
    float* LSE,
    int S_len) 
{
    int q_blk = blockIdx.x;
    int bh = blockIdx.y;
    int S = S_len;

    extern __shared__ __align__(1024) __nv_bfloat16 smem_buf[];
    __nv_bfloat16* smem_Q_0 = smem_buf;                   // 16KB
    __nv_bfloat16* smem_Q_1 = smem_buf + 8192;            // 16KB
    __nv_bfloat16* smem_K_0 = smem_buf + 16384;           // 16KB
    __nv_bfloat16* smem_K_1 = smem_buf + 24576;           // 16KB
    __nv_bfloat16* smem_V_0 = smem_buf + 32768;           // 16KB
    __nv_bfloat16* smem_V_1 = smem_buf + 40960;           // 16KB
    __nv_bfloat16* smem_P = smem_buf + 49152;             // 32KB
    
    __shared__ __align__(8) uint64_t mbar_Q[1];
    __shared__ __align__(8) uint64_t mbar_K[2];
    __shared__ __align__(8) uint64_t mbar_V[2];
    __shared__ __align__(8) uint64_t mbar_P[1];
    __shared__ __align__(8) uint64_t mbar_O[1];

    __shared__ float smem_m_prev[128];
    __shared__ float smem_sum_prev[128];

    if (threadIdx.x < 128) {
        smem_m_prev[threadIdx.x] = -1e20f;
        smem_sum_prev[threadIdx.x] = 0.0f;
    }

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&mbar_Q[0], 1);
        init_smem_barrier_fn(&mbar_K[0], 1);
        init_smem_barrier_fn(&mbar_V[0], 1);
        init_smem_barrier_fn(&mbar_K[1], 1);
        init_smem_barrier_fn(&mbar_V[1], 1);
        init_smem_barrier_fn(&mbar_P[0], 1);
        init_smem_barrier_fn(&mbar_O[0], 1);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();

    uint32_t tmem_P, tmem_O_0, tmem_O_1;
    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_P, 128);
        tmem_alloc_fn(&tmem_O_0, 64);
        tmem_alloc_fn(&tmem_O_1, 64);
    }
    __syncthreads();

    uint32_t phase_Q = 0, phase_K[2] = {0, 0}, phase_V[2] = {0, 0}, phase_P = 0, phase_O = 0;

    int num_blocks = (S + 127) / 128;
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&mbar_Q[0], 32768);
        tma_load_2d_fn(&tma_Q, &mbar_Q[0], smem_Q_0, 0, bh * S + q_blk * 128);
        tma_load_2d_fn(&tma_Q, &mbar_Q[0], smem_Q_1, 64, bh * S + q_blk * 128);
    }
    mbarrier_wait_fn(&mbar_Q[0], phase_Q);
    phase_Q ^= 1;

    uint32_t idesc_P = make_instr_desc_fn(128, 128, 0, 0);
    uint32_t idesc_O = make_instr_desc_fn(128, 64, 0, 1);
    float sqrt_D = 11.3137085f; // sqrt(128)
    
    int next_k_blk = 0;
    int next_valid = (next_k_blk <= q_blk) && (next_k_blk < num_blocks);
    
    if (next_valid) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&mbar_K[0], 32768);
            tma_load_2d_fn(&tma_K, &mbar_K[0], smem_K_0, 0, bh * S + next_k_blk * 128);
            tma_load_2d_fn(&tma_K, &mbar_K[0], smem_K_1, 64, bh * S + next_k_blk * 128);
            
            mbarrier_arrive_and_expect_tx_fn(&mbar_V[0], 32768);
            tma_load_2d_fn(&tma_V, &mbar_V[0], smem_V_0, 0, bh * S + next_k_blk * 128);
            tma_load_2d_fn(&tma_V, &mbar_V[0], smem_V_1, 64, bh * S + next_k_blk * 128);
        }
    }

    for (int k_blk = 0; k_blk <= q_blk && k_blk < num_blocks; k_blk++) {
        int buf_idx = k_blk % 2;
        int next_buf = (k_blk + 1) % 2;
        
        mbarrier_wait_fn(&mbar_K[buf_idx], phase_K[buf_idx]);
        phase_K[buf_idx] ^= 1;
        mbarrier_wait_fn(&mbar_V[buf_idx], phase_V[buf_idx]);
        phase_V[buf_idx] ^= 1;
        
        __nv_bfloat16* cur_K_0 = (buf_idx == 0) ? smem_K_0 : smem_K_1;
        __nv_bfloat16* cur_K_1 = (buf_idx == 0) ? smem_K_0 + 8192 : smem_K_1 + 8192;
        __nv_bfloat16* cur_V_0 = (buf_idx == 0) ? smem_V_0 : smem_V_1;
        __nv_bfloat16* cur_V_1 = (buf_idx == 0) ? smem_V_0 + 8192 : smem_V_1 + 8192;
        
        uint64_t desc_Q_0 = make_smem_desc_sm100_fn(smem_Q_0, 1, 1024);
        uint64_t desc_Q_1 = make_smem_desc_sm100_fn(smem_Q_1, 1, 1024);
        uint64_t desc_K_0 = make_smem_desc_sm100_fn(cur_K_0, 1, 1024);
        uint64_t desc_K_1 = make_smem_desc_sm100_fn(cur_K_1, 1, 1024);
        
        fence_proxy_async_fn();
        
        if (threadIdx.x == 0) {
            for (int k = 0; k < 8; ++k) {
                uint64_t d_Q = (k < 4) ? desc_Q_0 : desc_Q_1;
                uint64_t d_K = (k < 4) ? desc_K_0 : desc_K_1;
                uint32_t off = (k < 4) ? (k * 2) : ((k - 4) * 2);
                uint32_t accum = (k == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_P, d_Q + off, d_K + off, idesc_P, accum);
            }
            umma_commit_1sm_fn(&mbar_P[0]);
        }
        mbarrier_wait_fn(&mbar_P[0], phase_P);
        phase_P ^= 1;

        float m_local = -1e20f;
        uint32_t p_quad = threadIdx.x % 4;
        uint32_t warp_id = threadIdx.x / 32;
        uint32_t lane_id = threadIdx.x % 32;
        uint32_t lane_idx = warp_id * 32 + lane_id + (p_quad < 2 ? 0 : 32);
        
        for(int col = 0; col < 32; col += 8) {
            uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
            tmem_load_8x_fn(tmem_P + (lane_idx << 16) + col + p_quad * 32, &r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);
            tmem_load_fence_fn();
            float p_vals[8] = {__uint_as_float(r0), __uint_as_float(r1), __uint_as_float(r2), __uint_as_float(r3), __uint_as_float(r4), __uint_as_float(r5), __uint_as_float(r6), __uint_as_float(r7)};
            for(int i = 0; i < 8; ++i) {
                int q_idx = q_blk * 128 + lane_idx;
                int k_idx = k_blk * 128 + p_quad * 32 + col + i;
                bool valid = (q_idx < S) && (k_idx < S) && (k_idx <= q_idx);
                if (valid) {
                    p_vals[i] /= sqrt_D;
                    m_local = fmaxf(m_local, p_vals[i]);
                }
            }
        }
        
        for (int offset = 1; offset < 4; offset *= 2) {
            m_local = fmaxf(m_local, __shfl_xor_sync(0xffffffff, m_local, offset));
        }
        
        float m_prev = smem_m_prev[lane_idx];
        float m_new = fmaxf(m_prev, m_local);
        
        float sum_scaled = smem_sum_prev[lane_idx] * exp_f(m_prev - m_new);
        
        float sum_local = 0.0f;
        for(int col = 0; col < 32; col += 8) {
            uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
            tmem_load_8x_fn(tmem_P + (lane_idx << 16) + col + p_quad * 32, &r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);
            tmem_load_fence_fn();
            float p_vals[8] = {__uint_as_float(r0), __uint_as_float(r1), __uint_as_float(r2), __uint_as_float(r3), __uint_as_float(r4), __uint_as_float(r5), __uint_as_float(r6), __uint_as_float(r7)};
            for(int i = 0; i < 8; ++i) {
                int q_idx = q_blk * 128 + lane_idx;
                int k_idx = k_blk * 128 + p_quad * 32 + col + i;
                bool valid = (q_idx < S) && (k_idx < S) && (k_idx <= q_idx);
                if (valid) {
                    p_vals[i] /= sqrt_D;
                    p_vals[i] = exp_f(p_vals[i] - m_new);
                    sum_local += p_vals[i];
                } else {
                    p_vals[i] = 0.0f;
                }
                
                int real_i = p_quad * 32 + col + i;
                uint32_t c_x = real_i / 64;
                uint32_t c_rem = real_i % 64;
                uint32_t chunk_y = lane_idx % 8;
                uint32_t swizzled_idx = lane_idx * 128 + c_x * 64 + (chunk_y ^ (c_rem / 8)) * 8 + (c_rem % 8);
                smem_P[swizzled_idx] = __float2bfloat16(p_vals[i]);
            }
        }

        for (int offset = 1; offset < 4; offset *= 2) {
            sum_local += __shfl_xor_sync(0xffffffff, sum_local, offset);
        }
        
        smem_m_prev[lane_idx] = m_new;
        smem_sum_prev[lane_idx] = sum_scaled + sum_local;
        
        __syncthreads(); 
        fence_async_shared_fn();

        if (threadIdx.x == 0) {
            uint64_t desc_P = make_smem_desc_sm100_fn(smem_P, 1, 1024);
            uint64_t desc_V_0 = make_smem_desc_sm100_fn(cur_V_0, 8192, 1024);
            uint64_t desc_V_1 = make_smem_desc_sm100_fn(cur_V_1, 8192, 1024);
            
            for (int j = 0; j < 8; ++j) {
                uint64_t d_V = (j < 4) ? desc_V_0 : desc_V_1;
                uint32_t off_P = (j < 4) ? (j * 2) : ((j - 4) * 2);
                uint32_t off_V = (j < 4) ? (j * 2) : ((j - 4) * 2); 
                uint32_t accum_o = (k_blk == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_O_0, desc_P + off_P, d_V + off_V, idesc_O, accum_o);
                umma_f16_cg1_fn(tmem_O_1, desc_P + off_P, d_V + off_V, idesc_O, accum_o);
            }
            umma_commit_1sm_fn(&mbar_O[0]);
        }
        mbarrier_wait_fn(&mbar_O[0], phase_O);
        phase_O ^= 1;
        
        if (next_valid) {
            if (threadIdx.x == 0) {
                __nv_bfloat16* n_K_0 = (next_buf == 0) ? smem_K_0 : smem_K_1;
                __nv_bfloat16* n_K_1 = (next_buf == 0) ? smem_K_0 + 8192 : smem_K_1 + 8192;
                mbarrier_arrive_and_expect_tx_fn(&mbar_K[next_buf], 32768);
                tma_load_2d_fn(&tma_K, &mbar_K[next_buf], n_K_0, 0, bh * S + next_k_blk * 128);
                tma_load_2d_fn(&tma_K, &mbar_K[next_buf], n_K_1, 64, bh * S + next_k_blk * 128);
                
                __nv_bfloat16* n_V_0 = (next_buf == 0) ? smem_V_0 : smem_V_1;
                __nv_bfloat16* n_V_1 = (next_buf == 0) ? smem_V_0 + 8192 : smem_V_1 + 8192;
                mbarrier_arrive_and_expect_tx_fn(&mbar_V[next_buf], 32768);
                tma_load_2d_fn(&tma_V, &mbar_V[next_buf], n_V_0, 0, bh * S + next_k_blk * 128);
                tma_load_2d_fn(&tma_V, &mbar_V[next_buf], n_V_1, 64, bh * S + next_k_blk * 128);
            }
        }
        
        next_k_blk++;
        next_valid = (next_k_blk <= q_blk) && (next_k_blk < num_blocks);
    }

    int my_row = q_blk * 128 + threadIdx.x;
    for (int col = 0; col < 64; col += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn((tmem_O_0 + (threadIdx.x << 16)) + col, &r0, &r1, &r2, &r3);
        tmem_load_4x_fn((tmem_O_1 + (threadIdx.x << 16)) + col, &r0, &r1, &r2, &r3);
        tmem_load_fence_fn();
        
        if (my_row < S) {
            __nv_bfloat16 o0 = __float2bfloat16(__uint_as_float(r0));
            __nv_bfloat16 o1 = __float2bfloat16(__uint_as_float(r1));
            __nv_bfloat16 o2 = __float2bfloat16(__uint_as_float(r2));
            __nv_bfloat16 o3 = __float2bfloat16(__uint_as_float(r3));
            
            if (col + 3 < 128) {
                O[(bh * S + my_row) * 128 + col] = o0;
                O[(bh * S + my_row) * 128 + col + 1] = o1;
                O[(bh * S + my_row) * 128 + col + 2] = o2;
                O[(bh * S + my_row) * 128 + col + 3] = o3;
                
                O[(bh * S + my_row) * 128 + col + 64] = o0;
                O[(bh * S + my_row) * 128 + col + 65] = o1;
                O[(bh * S + my_row) * 128 + col + 66] = o2;
                O[(bh * S + my_row) * 128 + col + 67] = o3;
            }
        }
    }

    if (threadIdx.x < 128) {
        int row_idx = q_blk * 128 + threadIdx.x;
        if (row_idx < S) {
            LSE[bh * S + row_idx] = smem_m_prev[threadIdx.x] + logf(smem_sum_prev[threadIdx.x]);
        }
    }

    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_P, 128);
        tmem_dealloc_fn(tmem_O_0, 64);
        tmem_dealloc_fn(tmem_O_1, 64);
    }
}

namespace tvm_ffi_mha {

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill
    );
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3); 
    
    CUtensorMap tma_Q, tma_K, tma_V;
    __nv_bfloat16* q_ptr = static_cast<__nv_bfloat16*>(Q.data_ptr());
    __nv_bfloat16* k_ptr = static_cast<__nv_bfloat16*>(K.data_ptr());
    __nv_bfloat16* v_ptr = static_cast<__nv_bfloat16*>(V.data_ptr());

    CU_CHECK(create_tma_2d_descriptor_2B(&tma_Q, q_ptr, D, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_K, k_ptr, D, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_V, v_ptr, D, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    dim3 grid((S + 127) / 128, B * H);
    dim3 block(128);
    int smem_size = 128 * 1024 + 256;
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_size;
    config.stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, mha_kernel_sm100, tma_Q, tma_K, tma_V, static_cast<__nv_bfloat16*>(O.data_ptr()), static_cast<float*>(LSE.data_ptr()), S));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(config.stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_mha