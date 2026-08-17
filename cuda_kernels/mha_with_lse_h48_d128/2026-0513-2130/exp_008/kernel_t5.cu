#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <math.h>
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

// ------------------------------------------------------------------
// SM100 Specific Helpers
// ------------------------------------------------------------------
template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_dec_sync_fn() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

__device__ __forceinline__ float exp2f_hw(float x) {
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

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
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

__device__ __forceinline__ void tmem_alloc_1sm_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_1sm_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void umma_ss_f16_fn(
    uint32_t tmem_d, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_d), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_ts_f16_fn(
    uint32_t tmem_d, uint32_t tmem_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}\n"
        :: "r"(tmem_d), "r"(tmem_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_1sm_fn(uint64_t* bar) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        :: "l"((uint64_t)bar) : "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo, uint32_t swizzle) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr) >> 4;
    d |= (uint64_t)(addr & 0x3FFF);
    d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16; 
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;   // version = 1
    d |= (uint64_t)swizzle << 61;   
    return d;
}

__device__ __forceinline__ uint64_t advance_smem_desc_fn(uint64_t desc, uint32_t byte_offset) {
    uint32_t addr = (desc & 0x3FFF);
    addr += (byte_offset >> 4);
    return (desc & ~0x3FFFull) | (addr & 0x3FFF);
}

__device__ __forceinline__ uint32_t make_idesc_qk(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);           // c_format = FP32
    d |= (1u << 7);           // a_format = BF16
    d |= (1u << 10);          // b_format = BF16
    d |= (1u << 16);          // Transpose B Matrix = 1 (MN-Major for K memory)
    d |= ((N / 8) << 17);     // n_dim
    d |= ((M / 16) << 24);    // m_dim
    return d;
}

__device__ __forceinline__ uint32_t make_idesc_pv(uint32_t M, uint32_t D) {
    uint32_t d = 0;
    d |= (1u << 4);           // c_format = FP32
    d |= (1u << 7);           // a_format = BF16
    d |= (1u << 10);          // b_format = BF16
    d |= (0u << 16);          // Transpose B Matrix = 0 (K-Major for V memory)
    d |= ((D / 8) << 17);     // n_dim
    d |= ((M / 16) << 24);    // m_dim
    return d;
}

// ------------------------------------------------------------------
// Host TMA Helper
// ------------------------------------------------------------------
CUresult create_tma_2d_descriptor_2B(
    CUtensorMap* d, void* globalAddress, 
    uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, 
    uint32_t smem_inner_dim, uint32_t smem_outer_dim, 
    CUtensorMapDataType dataType, 
    CUtensorMapSwizzle swizzle, 
    CUtensorMapL2promotion l2Promotion, 
    CUtensorMapFloatOOBfill oobFill) 
{
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2}; 
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, dataType, 2, globalAddress, globalDim, globalStrides, boxDim, 
        elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill);
}

// ------------------------------------------------------------------
// Kernel Implementation
// ------------------------------------------------------------------
struct SharedStorage {
    __align__(16) uint64_t mbar_q[1];
    __align__(16) uint64_t mbar_k[2];
    __align__(16) uint64_t mbar_v[2];
    __align__(16) uint64_t mbar_mma[2];
    __align__(16) uint32_t tmem_addr;
    __align__(128) uint16_t smem_Q[128 * 128];
    __align__(128) uint16_t smem_K[2][128 * 128];
    __align__(128) uint16_t smem_V[2][128 * 128];
};

__global__ void __launch_bounds__(128, 1) mha_fwd_sm100_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O, float* LSE,
    int S_len, int H, float scale) 
{
    setmaxnreg_inc_sync_fn<248>();

    extern __shared__ char smem_buf[];
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(smem_buf);

    int q_start = blockIdx.x * 128;
    int bh_offset = (blockIdx.z * H + blockIdx.y) * S_len;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&smem.mbar_q[0], 1);
        for(int i = 0; i < 2; ++i) {
            init_smem_barrier_fn(&smem.mbar_k[i], 1);
            init_smem_barrier_fn(&smem.mbar_v[i], 1);
            init_smem_barrier_fn(&smem.mbar_mma[i], 1);
        }
    }
    
    if (threadIdx.x < 32) {
        tmem_alloc_1sm_fn(&smem.tmem_addr, 512);
    }
    __syncthreads();

    uint32_t col_S = smem.tmem_addr + 0;
    uint32_t col_O = smem.tmem_addr + 128;
    uint32_t col_P = smem.tmem_addr + 256;

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&smem.mbar_q[0], 128 * 128 * 2);
        tma_load_2d_fn(&tma_Q, &smem.mbar_q[0], smem.smem_Q, 0, bh_offset + q_start);
        
        mbarrier_arrive_and_expect_tx_fn(&smem.mbar_k[0], 128 * 128 * 2);
        tma_load_2d_fn(&tma_K, &smem.mbar_k[0], smem.smem_K[0], 0, bh_offset + 0);
        
        mbarrier_arrive_and_expect_tx_fn(&smem.mbar_v[0], 128 * 128 * 2);
        tma_load_2d_fn(&tma_V, &smem.mbar_v[0], smem.smem_V[0], 0, bh_offset + 0);
    }
    
    uint64_t desc_q = make_smem_desc_sm100_fn(smem.smem_Q, 2048, 128, 0); // K-Major (Transpose B = 1 expects B as MN-major)
    uint64_t desc_k[2];
    uint64_t desc_v[2];
    for(int i = 0; i < 2; ++i) {
        desc_k[i] = make_smem_desc_sm100_fn(smem.smem_K[i], 128, 2048, 0); // MN-Major matches make_idesc_qk Transpose B = 1
        desc_v[i] = make_smem_desc_sm100_fn(smem.smem_V[i], 2048, 128, 0); // K-Major matches make_idesc_pv Transpose B = 0
    }
    uint32_t idesc_qk = make_idesc_qk(128, 128);
    uint32_t idesc_pv = make_idesc_pv(128, 128);

    uint32_t phase_q = 0;
    uint32_t phase_k[2] = {0, 0};
    uint32_t phase_v[2] = {0, 0};
    uint32_t phase_mma[2] = {0, 0};
    
    mbarrier_wait_fn(&smem.mbar_q[0], phase_q);

    int S_blocks = (S_len + 127) / 128;
    int pipe_idx = 0;

    float m_old = -INFINITY;
    float l_old = 0.0f;
    constexpr float log2e = 1.4426950408889634f;
    
    for(int j = 0; j < S_blocks; ++j) {
        mbarrier_wait_fn(&smem.mbar_k[pipe_idx], phase_k[pipe_idx]);
        mbarrier_wait_fn(&smem.mbar_v[pipe_idx], phase_v[pipe_idx]);

        int next_j = j + 1;
        int next_pipe = pipe_idx ^ 1;
        if (next_j < S_blocks) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&smem.mbar_k[next_pipe], 128 * 128 * 2);
                tma_load_2d_fn(&tma_K, &smem.mbar_k[next_pipe], smem.smem_K[next_pipe], 0, bh_offset + next_j * 128);
                mbarrier_arrive_and_expect_tx_fn(&smem.mbar_v[next_pipe], 128 * 128 * 2);
                tma_load_2d_fn(&tma_V, &smem.mbar_v[next_pipe], smem.smem_V[next_pipe], 0, bh_offset + next_j * 128);
            }
        }

        __syncthreads();

        // 1. Q * K^T -> S
        if (threadIdx.x == 0) {
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            for(int i = 0; i < 8; ++i) {
                uint64_t da = advance_smem_desc_fn(desc_q, i * 32);
                uint64_t db = advance_smem_desc_fn(desc_k[pipe_idx], i * 32);
                uint32_t accum = (i == 0) ? 0 : 1;
                umma_ss_f16_fn(col_S, da, db, idesc_qk, accum);
            }
            umma_commit_1sm_fn(&smem.mbar_mma[pipe_idx]);
        }
        mbarrier_wait_fn(&smem.mbar_mma[pipe_idx], phase_mma[pipe_idx]);
        phase_mma[pipe_idx] ^= 1;

        __syncthreads();

        // 2. Read S and process softmax
        float S_reg[128];
        float max_val = -INFINITY;
        
        #pragma unroll 32
        for(int i = 0; i < 128; i += 4) {
            uint32_t col = col_S + i;
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(col));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

            float f0 = __uint_as_float(r0) * scale;
            float f1 = __uint_as_float(r1) * scale;
            float f2 = __uint_as_float(r2) * scale;
            float f3 = __uint_as_float(r3) * scale;

            if (j == S_blocks - 1) {
                if ((j * 128 + i + 0) >= S_len) f0 = -INFINITY;
                if ((j * 128 + i + 1) >= S_len) f1 = -INFINITY;
                if ((j * 128 + i + 2) >= S_len) f2 = -INFINITY;
                if ((j * 128 + i + 3) >= S_len) f3 = -INFINITY;
            }

            S_reg[i+0] = f0;
            S_reg[i+1] = f1;
            S_reg[i+2] = f2;
            S_reg[i+3] = f3;

            max_val = max(max_val, f0);
            max_val = max(max_val, f1);
            max_val = max(max_val, f2);
            max_val = max(max_val, f3);
        }

        float m_new = max(m_old, max_val);
        float l_scale = (m_old == -INFINITY) ? 0.0f : exp2f_hw((m_old - m_new) * log2e);

        // 3. Rescale O (Inline conditional fast rescale leveraging conditional divergence uniformity per Warp)
        bool need_rescale = (l_scale < 1.0f && j > 0);
        if (__any_sync(0xFFFFFFFF, need_rescale)) {
            #pragma unroll 32
            for(int i = 0; i < 128; i += 4) {
                uint32_t col = col_O + i;
                uint32_t r0, r1, r2, r3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                             : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(col));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

                if (need_rescale) {
                    r0 = __float_as_uint(__uint_as_float(r0) * l_scale);
                    r1 = __float_as_uint(__uint_as_float(r1) * l_scale);
                    r2 = __float_as_uint(__uint_as_float(r2) * l_scale);
                    r3 = __float_as_uint(__uint_as_float(r3) * l_scale);
                }

                asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
                             :: "r"(col), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            }
        }

        // 4. Compute and store P natively as 2x packed BF16 per TMEM cell (64 columns utilized)
        float sum = 0.0f;
        uint32_t P_reg[64];
        #pragma unroll 32
        for(int i = 0; i < 128; i += 2) {
            float p0 = exp2f_hw((S_reg[i+0] - m_new) * log2e);
            float p1 = exp2f_hw((S_reg[i+1] - m_new) * log2e);
            sum += p0 + p1;
            P_reg[i/2] = pack_bf16_fn(__float_as_uint(p0), __float_as_uint(p1));
        }
        l_old = l_old * l_scale + sum;
        m_old = m_new;

        __syncthreads(); 

        #pragma unroll 16
        for(int i = 0; i < 64; i += 4) {
            uint32_t col = col_P + i;
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
                         :: "r"(col), "r"(P_reg[i+0]), "r"(P_reg[i+1]), "r"(P_reg[i+2]), "r"(P_reg[i+3]) : "memory");
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");

        __syncthreads(); 

        // 5. P * V -> O
        if (threadIdx.x == 0) {
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            for(int i = 0; i < 8; ++i) {
                uint32_t tmem_a = col_P + i * 8; // Advance 8 packed columns per iteration
                uint64_t db = advance_smem_desc_fn(desc_v[pipe_idx], i * 4096);
                uint32_t accum = (j == 0 && i == 0) ? 0 : 1;
                umma_ts_f16_fn(col_O, tmem_a, db, idesc_pv, accum);
            }
            umma_commit_1sm_fn(&smem.mbar_mma[pipe_idx]);
        }
        mbarrier_wait_fn(&smem.mbar_mma[pipe_idx], phase_mma[pipe_idx]);
        phase_mma[pipe_idx] ^= 1;

        phase_k[pipe_idx] ^= 1;
        phase_v[pipe_idx] ^= 1;
        pipe_idx = next_pipe;
    }

    // Epilogue: Final scaling and global store
    float out_scale = 1.0f / l_old;

    int row = q_start + threadIdx.x;
    if (row < S_len) {
        int lse_idx = (blockIdx.z * H + blockIdx.y) * S_len + row;
        LSE[lse_idx] = m_old + logf(l_old);
    }

    __syncthreads();

    #pragma unroll 32
    for(int i = 0; i < 128; i += 4) {
        uint32_t col = col_O + i;
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        if (row < S_len) {
            float o0 = __uint_as_float(r0) * out_scale;
            float o1 = __uint_as_float(r1) * out_scale;
            float o2 = __uint_as_float(r2) * out_scale;
            float o3 = __uint_as_float(r3) * out_scale;

            __nv_bfloat16 val0 = __float2bfloat16(o0);
            __nv_bfloat16 val1 = __float2bfloat16(o1);
            __nv_bfloat16 val2 = __float2bfloat16(o2);
            __nv_bfloat16 val3 = __float2bfloat16(o3);

            smem.smem_Q[threadIdx.x * 128 + i + 0] = *reinterpret_cast<uint16_t*>(&val0);
            smem.smem_Q[threadIdx.x * 128 + i + 1] = *reinterpret_cast<uint16_t*>(&val1);
            smem.smem_Q[threadIdx.x * 128 + i + 2] = *reinterpret_cast<uint16_t*>(&val2);
            smem.smem_Q[threadIdx.x * 128 + i + 3] = *reinterpret_cast<uint16_t*>(&val3);
        }
    }

    __syncthreads();

    int num_steps = 128 / 4;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    for(int step = 0; step < num_steps; ++step) {
        int r = step * 4 + warp_id; 
        if (r < 128 && q_start + r < S_len) {
            int out_base = (blockIdx.z * H + blockIdx.y) * S_len * 128 + (q_start + r) * 128;
            int col_start = lane_id * 4;
            uint2 data = *reinterpret_cast<uint2*>(&smem.smem_Q[r * 128 + col_start]);
            *reinterpret_cast<uint2*>(&O[out_base + col_start]) = data;
        }
    }

    __syncthreads();
    if (threadIdx.x < 32) {
        tmem_dealloc_1sm_fn(smem.tmem_addr, 512);
    }
}

namespace tvm_ffi_mha {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);

    CUtensorMap tma_Q, tma_K, tma_V;
    create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), D, B * H * S, D, 128, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_NONE, 
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), D, B * H * S, D, 128, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_NONE, 
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), D, B * H * S, D, 128, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_NONE, 
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    int grid_x = (S + 127) / 128;
    dim3 grid(grid_x, H, B);
    dim3 block(128); 

    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = sizeof(SharedStorage);
    config.stream = stream;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 1;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    cudaFuncSetAttribute((void*)mha_fwd_sm100_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedStorage));

    CUDA_CHECK(cudaLaunchKernelEx(&config, mha_fwd_sm100_kernel, 
        tma_Q, tma_K, tma_V, 
        static_cast<__nv_bfloat16*>(O.data_ptr()), 
        static_cast<float*>(LSE.data_ptr()), 
        S, H, scale));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha::run);

}  // namespace tvm_ffi_mha