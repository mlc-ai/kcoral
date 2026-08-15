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

// ----------------------------------------------------------------------
// PTX Wrappers
// ----------------------------------------------------------------------

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(tx_bytes) : "memory");
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

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(bar)),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
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

__device__ __forceinline__ void read_tmem_to_smem(void* smem_dst, uint32_t tmem_src_base, int col_offset, int row_offset) {
    // Copy 16 rows x 256 bytes (64 floats) from TMEM to SMEM
    asm volatile(
        "tcgen05.cp.cta_group::1.16x256b [%0], [%1];" 
        :: "r"((uint32_t)__cvta_generic_to_shared(smem_dst)),
        "r"(tmem_src_base + (col_offset << 2) + (row_offset << 18)) : "memory");
}

__device__ __forceinline__ void write_smem_to_tmem(void* smem_src, uint32_t tmem_dst_base, int col_offset, int row_offset) {
    // Copy 16 rows x 256 bytes (64 floats) from SMEM to TMEM
    asm volatile(
        "tcgen05.cp.cta_group::1.16x256b [%0], [%1];" 
        :: "r"(tmem_dst_base + (col_offset << 2) + (row_offset << 18)),
        "r"((uint32_t)__cvta_generic_to_shared(smem_src)) : "memory");
}

__device__ __forceinline__ void tmem_commit_cp_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(a));
}

__device__ __forceinline__ void tmem_wait_cp_fn(uint64_t* bar, uint32_t phase) {
    mbarrier_wait_fn(bar, phase);
}

__device__ __forceinline__ void tmem_commit_mma_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(a));
}

__device__ __forceinline__ void tmem_wait_mma_fn(uint64_t* bar, uint32_t phase) {
    mbarrier_wait_fn(bar, phase);
}

__device__ __forceinline__ void umma_f16_cta1_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ uint64_t make_smem_desc_k_major(void* smem_ptr, void* base_ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint32_t base_addr = (uint32_t)__cvta_generic_to_shared(base_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    uint64_t lbo = 16; 
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    uint64_t sbo = 1024;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    uint64_t base = (base_addr >> 7) & 7;
    d |= (base << 49);
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, uint32_t b_transpose) {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    d |= (b_transpose << 16);
    return d;
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ int swizzle_128B_64(int row, int col) {
    int x = col / 8;
    int rem = col % 8;
    int swizzled_x = (row % 8) ^ x;
    return swizzled_x * 8 + rem;
}

// ----------------------------------------------------------------------
// Shared Storage & Kernels
// ----------------------------------------------------------------------

struct __align__(1024) SharedStorage {
    __nv_bfloat16 s_Q0[128 * 64];
    __nv_bfloat16 s_Q1[128 * 64];
    __nv_bfloat16 s_K0[128 * 64];
    __nv_bfloat16 s_K1[128 * 64];
    __nv_bfloat16 s_V0[128 * 64];
    __nv_bfloat16 s_V1[128 * 64];
    __nv_bfloat16 s_dO0[128 * 64];
    __nv_bfloat16 s_dO1[128 * 64];
    
    // dS stored transposed
    __nv_bfloat16 s_dPT[128 * 128]; 
    // P_T stored normally
    __nv_bfloat16 s_PT[128 * 128];  
    
    float s_L[128];
    float s_STEMP[1024]; // Intermediate staging buffer for TMEM reads (16x64 floats)
    uint64_t s_bar;
};

__global__ void convert_fp32_to_bf16(__nv_bfloat16* out, const float* in, uint64_t num_elements) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        out[idx] = __float2bfloat16(in[idx]);
    }
}

__global__ void bwd_dq_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* L, float* dQ_workspace,
    uint32_t S, float scale)
{
    uint32_t m_block = blockIdx.x * 128;
    uint32_t n_block = blockIdx.y * 128;
    uint32_t bh = blockIdx.z;
    uint32_t offset_bh = bh * S;

    if (m_block >= S || n_block >= S) return;

    extern __shared__ __align__(1024) char smem_buf[];
    SharedStorage* s = (SharedStorage*)smem_buf;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&s->s_bar, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t phase = 0;
    uint32_t tid = threadIdx.x;

    if (threadIdx.x == 0) {
        uint32_t tx_bytes = 4 * (128 * 64 * 2);
        mbarrier_arrive_and_expect_tx_fn(&s->s_bar, tx_bytes);

        tma_load_3d_fn(&tma_Q, &s->s_bar, s->s_Q0, 0, offset_bh + m_block, bh);
        tma_load_3d_fn(&tma_Q, &s->s_bar, s->s_Q1, 64, offset_bh + m_block, bh);
        tma_load_3d_fn(&tma_dO, &s->s_bar, s->s_dO0, 0, offset_bh + m_block, bh);
        tma_load_3d_fn(&tma_dO, &s->s_bar, s->s_dO1, 64, offset_bh + m_block, bh);
    }

    if (tid < 128) {
        s->s_L[tid] = (m_block + tid < S) ? L[offset_bh + m_block + tid] : 0.0f;
    }
    mbarrier_wait_fn(&s->s_bar, phase);
    phase ^= 1;
    __syncthreads();
    fence_proxy_async_fn();

    uint32_t tmem_S, tmem_dP, tmem_Q, tmem_K, tmem_V, tmem_dO, tmem_dPT, tmem_dQ0, tmem_dQ1;

    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_S, 128);
        tmem_alloc_fn(&tmem_dP, 128);
        tmem_alloc_fn(&tmem_Q, 128);
        tmem_alloc_fn(&tmem_K, 128);
        tmem_alloc_fn(&tmem_V, 128);
        tmem_alloc_fn(&tmem_dO, 128);
        tmem_alloc_fn(&tmem_dPT, 128);
        tmem_alloc_fn(&tmem_dQ0, 64);
        tmem_alloc_fn(&tmem_dQ1, 64);
        
        mbarrier_arrive_and_expect_tx_fn(&s->s_bar, 4 * (128 * 64 * 2));
        read_tmem_to_smem(s->s_Q0, tmem_Q, 0, 0);
        read_tmem_to_smem(s->s_Q1, tmem_Q, 64, 0);
        read_tmem_to_smem(s->s_K0, tmem_K, 0, 0);
        read_tmem_to_smem(s->s_K1, tmem_K, 64, 0);
    }
    __syncthreads();
    tmem_commit_cp_fn(&s->s_bar);
    tmem_wait_cp_fn(&s->s_bar, phase);
    phase ^= 1;
    __syncthreads();

    // S = Q @ K^T
    uint32_t accum = 0;
    for (uint32_t step = 0; step < 128; step += 16) {
        uint64_t desc_A = make_smem_desc_k_major(&s->s_Q0[(step / 8) * 1024], s->s_Q0);
        uint64_t desc_B = make_smem_desc_k_major(&s->s_K0[(step / 8) * 1024], s->s_K0);
        uint32_t idesc = make_instr_desc_fn(128, 64, 1);
        if (threadIdx.x == 0) {
            umma_f16_cta1_fn(tmem_S, desc_A, desc_B, idesc, accum);
        }
        
        uint64_t desc_A1 = make_smem_desc_k_major(&s->s_Q1[(step / 8) * 1024], s->s_Q1);
        uint64_t desc_B1 = make_smem_desc_k_major(&s->s_K1[(step / 8) * 1024], s->s_K1);
        if (threadIdx.x == 0) {
            umma_f16_cta1_fn(tmem_S, desc_A1, desc_B1, idesc, accum);
        }
        accum = 1;
    }
    
    tmem_commit_mma_fn(&s->s_bar);
    tmem_wait_mma_fn(&s->s_bar, phase);
    phase ^= 1;
    __syncthreads();
    
    float S_local[128] = {0};
    for (uint32_t r = 0; r < 128; r += 16) {
        for (uint32_t c = 0; c < 128; c += 64) {
            read_tmem_to_smem(s->s_STEMP, tmem_S, c, r);
        }
        tmem_commit_cp_fn(&s->s_bar);
        tmem_wait_cp_fn(&s->s_bar, phase);
        phase ^= 1;
        __syncthreads();
        
        uint32_t row = tid / 8;
        uint32_t col = (tid % 8) * 8;
        if (row < 16) {
            S_local[r + row] = s->s_STEMP[row * 64 + col];
        }
    }

    __nv_bfloat16 P_T_local[128];
    for (uint32_t i = 0; i < 128; i++) {
        float s_val = S_local[i] * scale;
        float l_val = s->s_L[tid];
        float p = fast_exp2f_fn((s_val - l_val) * 1.44269504f);
        P_T_local[i] = __float2bfloat16(p);
    }

    for (uint32_t r = 0; r < 128; r += 8) {
        for (uint32_t c = 0; c < 128; c += 8) {
            uint32_t swizzled_c = ((r / 8) ^ (c / 8)) * 8 + (c % 8);
            uint32_t addr = r * 128 + swizzled_c;
            
            uint32_t reg_val = __float_as_uint(__bfloat162float(P_T_local[r * 8 + (c % 8)]));
            uint32_t* ptr = (uint32_t*)&s->s_PT[addr];
            *ptr = reg_val;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&s->s_bar, 4 * (128 * 64 * 2));
        read_tmem_to_smem(s->s_V0, tmem_V, 0, 0);
        read_tmem_to_smem(s->s_V1, tmem_V, 64, 0);
        read_tmem_to_smem(s->s_dO0, tmem_dO, 0, 0);
        read_tmem_to_smem(s->s_dO1, tmem_dO, 64, 0);
    }
    __syncthreads();
    tmem_commit_cp_fn(&s->s_bar);
    tmem_wait_cp_fn(&s->s_bar, phase);
    phase ^= 1;
    __syncthreads();

    // dP = dO * V^T
    accum = 0;
    for (uint32_t step = 0; step < 128; step += 16) {
        uint64_t desc_A = make_smem_desc_k_major(&s->s_dO0[(step / 8) * 1024], s->s_dO0);
        uint64_t desc_B = make_smem_desc_k_major(&s->s_V0[(step / 8) * 1024], s->s_V0);
        uint32_t idesc = make_instr_desc_fn(128, 64, 1);
        if (threadIdx.x == 0) {
            umma_f16_cta1_fn(tmem_dP, desc_A, desc_B, idesc, accum);
        }
        
        uint64_t desc_A1 = make_smem_desc_k_major(&s->s_dO1[(step / 8) * 1024], s->s_dO1);
        uint64_t desc_B1 = make_smem_desc_k_major(&s->s_V1[(step / 8) * 1024], s->s_V1);
        if (threadIdx.x == 0) {
            umma_f16_cta1_fn(tmem_dP, desc_A1, desc_B1, idesc, accum);
        }
        accum = 1;
    }
    
    tmem_commit_mma_fn(&s->s_bar);
    tmem_wait_mma_fn(&s->s_bar, phase);
    phase ^= 1;
    __syncthreads();

    float dP_local[128] = {0};
    for (uint32_t r = 0; r < 128; r += 16) {
        for (uint32_t c = 0; c < 128; c += 64) {
            read_tmem_to_smem(s->s_STEMP, tmem_dP, c, r);
        }
        tmem_commit_cp_fn(&s->s_bar);
        tmem_wait_cp_fn(&s->s_bar, phase);
        phase ^= 1;
        __syncthreads();
        
        uint32_t row = tid / 8;
        uint32_t col = (tid % 8) * 8;
        if (row < 16) {
            dP_local[r + row] = s->s_STEMP[row * 64 + col];
        }
    }

    float D_local[128] = {0};
    for (uint32_t r = 0; r < 128; r += 8) {
        for (uint32_t c = 0; c < 128; c += 8) {
            uint32_t swizzled_c = ((r / 8) ^ (c / 8)) * 8 + (c % 8);
            uint32_t addr = r * 128 + swizzled_c;