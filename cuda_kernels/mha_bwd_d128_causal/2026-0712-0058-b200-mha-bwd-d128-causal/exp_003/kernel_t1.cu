#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <math.h>
#include <stdio.h>
#include <algorithm>
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

// Basic PTX wrappers
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

__device__ __forceinline__ uint64_t make_smem_desc(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)0 << 61;   // SWIZZLE_NONE
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    // c_format = FP32
    d |= (1u << 7);    // a_format = BF16
    d |= (1u << 10);   // b_format = BF16
    d |= (0u << 15);   // a_major = 0 (A is K-Major)
    d |= (0u << 16);   // b_major = 0 (B is K-Major)
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ int swizzle_none_idx(int r, int c) {
    return r * 8 + (((r / 8) ^ (c / 8)) * 8) + (c % 8);
}

__global__ void bwd_dQ_kernel(
    const __grid_constant__ CUtensorMap* tma_Q, const __grid_constant__ CUtensorMap* tma_K,
    const __grid_constant__ CUtensorMap* tma_V, const __grid_constant__ CUtensorMap* tma_dO,
    const __grid_constant__ CUtensorMap* tma_O, const __grid_constant__ CUtensorMap* tma_P,
    const float* L_gmem,
    __nv_bfloat16* dQ_gmem,
    int S, float scale) 
{
    int q_blk = blockIdx.x;
    int bh = blockIdx.y;
    int tid = threadIdx.x;

    constexpr int BM = 64;
    constexpr int BN = 64;

    extern __shared__ char smem_buf[];
    __nv_bfloat16* smem_Q_BM = (__nv_bfloat16*)smem_buf;
    __nv_bfloat16* smem_Q_BN = smem_Q_BM + BM * BN;
    __nv_bfloat16* smem_dO_BM = smem_Q_BN + BM * BN;
    __nv_bfloat16* smem_dO_BN = smem_dO_BM + BM * BN;
    __nv_bfloat16* smem_O_BM = smem_dO_BN + BM * BN;
    __nv_bfloat16* smem_O_BN = smem_O_BM + BM * BN;
    __nv_bfloat16* smem_K_BM = smem_O_BN + BM * BN;
    __nv_bfloat16* smem_K_BN = smem_K_BM + BM * BN;
    __nv_bfloat16* smem_V_BM = smem_K_BN + BM * BN;
    __nv_bfloat16* smem_V_BN = smem_V_BM + BM * BN;
    __nv_bfloat16* smem_P_BM = smem_V_BN + BM * BN;
    __nv_bfloat16* smem_P_BN = smem_P_BM + BM * BN;

    __align__(16) uint64_t* bar_A = (uint64_t*)(smem_P_BN + BM * BN);
    __align__(16) uint64_t* bar_B = bar_A + 1;
    float* head_dO_O_shared = (float*)(bar_B + 1);

    if (tid == 0) {
        init_smem_barrier_fn(bar_A, 1);
        init_smem_barrier_fn(bar_B, 1);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t phase_A = 0;
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_A, sizeof(__nv_bfloat16) * BM * BN * 4);
        tma_load_2d_fn(&tma_O[bh], bar_A, smem_O_BM, 0, bh * S + q_blk * BM);
        tma_load_2d_fn(&tma_O[bh], bar_A, smem_O_BN, 64, bh * S + q_blk * BM);
        tma_load_2d_fn(&tma_dO[bh], bar_A, smem_dO_BM, 0, bh * S + q_blk * BM);
        tma_load_2d_fn(&tma_dO[bh], bar_A, smem_dO_BN, 64, bh * S + q_blk * BM);
    }
    mbarrier_wait_fn(bar_A, phase_A);
    phase_A ^= 1;

    if (tid < 64) {
        float sum = 0;
        for(int i = 0; i < 64; i++) {
            sum += __bfloat162float(smem_O_BM[tid * 64 + i]) * __bfloat162float(smem_dO_BM[tid * 64 + i]);
            sum += __bfloat162float(smem_O_BN[tid * 64 + i]) * __bfloat162float(smem_dO_BN[tid * 64 + i]);
        }
        head_dO_O_shared[tid] = sum;
    }
    __syncthreads();

    uint32_t tmem_dQ_BM, tmem_dQ_BN, tmem_S, tmem_dP;
    if (tid == 0) {
        tmem_alloc_fn(&tmem_dQ_BM, 64);
        tmem_alloc_fn(&tmem_dQ_BN, 64);
        tmem_alloc_fn(&tmem_S, 64);
        tmem_alloc_fn(&tmem_dP, 64);
    }
    __syncthreads();

    uint32_t idesc_S_QK = make_instr_desc_fn(BM, BN);
    uint32_t idesc_dP_OV = make_instr_desc_fn(BM, BN);
    uint32_t idesc_dQ_dS_K = make_instr_desc_fn(BM, BN);

    int num_S_blocks = (S + BM - 1) / BM;
    uint32_t phase_B = 0;

    for (int k_blk = 0; k_blk <= q_blk && k_blk < num_S_blocks; k_blk++) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_B, sizeof(__nv_bfloat16) * BM * BN * 6);
            tma_load_2d_fn(&tma_K[bh], bar_B, smem_K_BM, 0, bh * S + k_blk * BN);
            tma_load_2d_fn(&tma_K[bh], bar_B, smem_K_BN, 64, bh * S + k_blk * BN);
            tma_load_2d_fn(&tma_V[bh], bar_B, smem_V_BM, 0, bh * S + k_blk * BN);
            tma_load_2d_fn(&tma_V[bh], bar_B, smem_V_BN, 64, bh * S + k_blk * BN);
            tma_load_2d_fn(&tma_P[bh], bar_B, smem_P_BM, 0, bh * S + q_blk * BM);
            tma_load_2d_fn(&tma_P[bh], bar_B, smem_P_BN, 64, bh * S + q_blk * BM);
        }
        mbarrier_wait_fn(bar_B, phase_B);
        phase_B ^= 1;

        if (tid < 128) {
            uint64_t desc_dO_BM = make_smem_desc(smem_dO_BM, 1024, 128);
            uint64_t desc_dO_BN = make_smem_desc(smem_dO_BN, 1024, 128);
            uint64_t desc_V_BM = make_smem_desc(smem_V_BM, 128, 1024);
            uint64_t desc_V_BN = make_smem_desc(smem_V_BN, 128, 1024);
            
            umma_f16_cg1_fn(tmem_dP, desc_dO_BM, desc_V_BM, idesc_dP_OV, 0);
            umma_f16_cg1_fn(tmem_dP, desc_dO_BN, desc_V_BN, idesc_dP_OV, 1);
        }
        umma_commit_1sm_fn(bar_A);
        mbarrier_wait_fn(bar_A, phase_A);
        phase_A ^= 1;

        // Process elements locally in shared memory avoiding implicit TMEM dependencies mapping pitfalls
        for(int i = 0; i < 128; i++) {
            int r = i / 2;
            int c = (i % 2) * 32 + (i % 32);
            
            uint32_t read_ptr = tmem_dP + r * 64 + c;
            uint32_t val;
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(val) : "r"(read_ptr));
            float dp_val = __uint_as_float(val);
            
            float p_bm = __bfloat162float(smem_P_BM[swizzle_none_idx(r, c)]);
            float p_bn = __bfloat162float(smem_P_BN[swizzle_none_idx(r, c)]);
            
            float ds_bm = p_bm * (dp_val - head_dO_O_shared[r]) * scale;
            float ds_bn = p_bn * (dp_val - head_dO_O_shared[r]) * scale;
            
            if ((q_blk * BM + r) < (k_blk * BN + c)) {
                ds_bm = 0.0f;
                ds_bn = 0.0f;
            }
            
            smem_dO_BM[swizzle_none_idx(r, c)] = __float2bfloat16(ds_bm);
            smem_dO_BN[swizzle_none_idx(r, c)] = __float2bfloat16(ds_bn);
        }
        __syncthreads();
        fence_proxy_async_fn();

        if (tid < 128) {
            uint64_t desc_dS_BM = make_smem_desc(smem_dO_BM, 1024, 128);
            uint64_t desc_K_BM = make_smem_desc(smem_K_BM, 1024, 128);
            umma_f16_cg1_fn(tmem_dQ_BM, desc_dS_BM, desc_K_BM, idesc_dQ_dS_K, (k_blk == 0) ? 0 : 1);
            
            uint64_t desc_dS_BN = make_smem_desc(smem_dO_BN, 1024, 128);
            uint64_t desc_K_BN = make_smem_desc(smem_K_BN, 1024, 128);
            umma_f16_cg1_fn(tmem_dQ_BN, desc_dS_BN, desc_K_BN, idesc_dQ_dS_K, (k_blk == 0) ? 0 : 1);
        }
        umma_commit_1sm_fn(bar_A);
        mbarrier_wait_fn(bar_A, phase_A);
        phase_A ^= 1;
    }

    // Coalesced Epilogue Layout directly writing to Global Memory avoiding TMEM dependency mappings
    for (int i = tid; i < BM * BN; i += blockDim.x) {
        int r = i / BN;
        int c = i % BN;
        
        uint32_t read_ptr_BM = tmem_dQ_BM + r * BN + c;
        uint32_t val_BM;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(val_BM) : "r"(read_ptr_BM));
        dQ_gmem[(bh * S + q_blk * BM + r) * 128 + c] = __float2bfloat16(__uint_as_float(val_BM));
        
        uint32_t read_ptr_BN = tmem_dQ_BN + r * BN + c;
        uint32_t val_BN;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(val_BN) : "r"(read_ptr_BN));
        dQ_gmem[(bh * S + q_blk * BM + r) * 128 + 64 + c] = __float2bfloat16(__uint_as_float(val_BN));
    }

    if (tid == 0) {
        tmem_dealloc_fn(tmem_dQ_BM, 64);
        tmem_dealloc_fn(tmem_dQ_BN, 64);
        tmem_dealloc_fn(tmem_S, 64);
        tmem_dealloc_fn(tmem_dP, 64);
    }
}

__global__ void bwd_dK_kernel(
    const __grid_constant__ CUtensorMap* tma_Q, const __grid_constant__ CUtensorMap* tma_K,
    const __grid_constant__ CUtensorMap* tma_V, const __grid_constant__ CUtensorMap* tma_dO,
    const __grid_constant__ CUtensorMap* tma_O, const __grid_constant__ CUtensorMap* tma_P,
    const float* L_gmem,
    __nv_bfloat16* dK_gmem,
    int S, float scale) 
{
    int k_blk = blockIdx.x;
    int bh = blockIdx.y;
    int tid = threadIdx.x;

    constexpr int BM = 64;
    constexpr int BN = 64;

    extern __shared__ char smem_buf[];
    __nv_bfloat16* smem_Q_BM = (__nv_bfloat16*)smem_buf;
    __nv_bfloat16* smem_Q_BN = smem_Q_BM + BM * BN;
    __nv_bfloat16* smem_dO_BM = smem_Q_BN + BM * BN;
    __nv_bfloat16* smem_dO_BN = smem_dO_BM + BM * BN;
    __nv_bfloat16* smem_O_BM = smem_dO_BN + BM * BN;
    __nv_bfloat16* smem_O_BN = smem_O_BM + BM * BN;
    __nv_bfloat16* smem_K_BM = smem_O_BN + BM * BN;
    __nv_bfloat16* smem_K_BN = smem_K_BM + BM * BN;
    __nv_bfloat16* smem_V_BM = smem_K_BN + BM * BN;
    __nv_bfloat16* smem_V_BN = smem_V_BM + BM * BN;
    __nv_bfloat16* smem_P_BM = smem_V_BN + BM * BN;
    __nv_bfloat16* smem_P_BN = smem_P_BM + BM * BN;

    __align__(16) uint64_t* bar_A = (uint64_t*)(smem_P_BN + BM * BN);
    __align__(16) uint64_t* bar_B = bar_A + 1;
    float* head_dO_O_shared = (float*)(bar_B + 1);

    if (tid == 0) {
        init_smem_barrier_fn(bar_A, 1);
        init_smem_barrier_fn(bar_B, 1);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t phase_A = 0;
    uint32_t tmem_dK_BM, tmem_dK_BN, tmem_S, tmem_dP;
    if (tid == 0) {
        tmem_alloc_fn(&tmem_dK_BM, 64);
        tmem_alloc_fn(&tmem_dK_BN, 64);
        tmem_alloc_fn(&tmem_S, 64);
        tmem_alloc_fn(&tmem_dP, 64);
    }
    __syncthreads();

    uint32_t idesc_S_QK = make_instr_desc_fn(BM, BN);
    uint32_t idesc_dP_OV = make_instr_desc_fn(BM, BN);
    uint32_t idesc_dK_dS_Q = make_instr_desc_fn(BM, BN);

    int num_S_blocks = (S + BM - 1) / BM;
    uint32_t phase_B = 0;

    for (int q_blk = k_blk; q_blk < num_S_blocks; q_blk++) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_B, sizeof(__nv_bfloat16) * BM * BN * 8);
            tma_load_2d_fn(&tma_Q[bh], bar_B, smem_Q_BM, 0, bh * S + q_blk * BM);
            tma_load_2d_fn(&tma_Q[bh], bar_B, smem_Q_BN, 64, bh * S + q_blk * BM);
            tma_load_2d_fn(&tma_dO[bh], bar_B, smem_dO_BM, 0, bh * S + q_blk * BM);
            tma_load_2d_fn(&tma_dO[bh], bar_B, smem_dO_BN, 64, bh * S + q_blk * BM);
            tma_load_2d_fn(&tma_O[bh], bar_B, smem_O_BM, 0, bh * S + q_blk * BM);
            tma_load_2d_fn(&tma_O[bh], bar_B, smem_O_BN, 64, bh * S + q_blk * BM);
            tma_load_2d_fn(&tma_V[bh], bar_B, smem_V_BM, 0, bh * S + q_blk * BM);
            tma_load_2d_fn(&tma_V[bh], bar_B, smem_V_BN, 64, bh * S + q_blk * BM);
            tma_load_2d_fn(&tma_P[bh], bar_B, smem_P_BM, k_blk * BN, bh * S + q_blk * BM);
            tma_load_2d_fn(&tma_P[bh], bar_B, smem_P_BN, k_blk * BN + 64, bh * S + q_blk * BM);
        }
        mbarrier_wait_fn(bar_B, phase_B);
        phase_B ^= 1;

        if (tid < 64) {
            float sum = 0;
            for(int i = 0; i < 64; i++) {
                sum += __bfloat162float(smem_O_BM[tid * 64 + i]) * __bfloat162float(smem_dO_BM[tid * 64 + i]);
                sum += __bfloat162float(smem_O_BN[tid * 64 + i]) * __bfloat162float(smem_dO_BN[tid * 64 + i]);
            }
            head_dO_O_shared[tid] = sum;
        }
        __syncthreads();

        if (tid < 128) {
            uint64_t desc_dO_BM = make_smem_desc(smem_dO_BM, 1024, 128);
            uint64_t desc_dO_BN = make_smem_desc(smem_dO_BN, 1024, 128);
            uint64_t desc_V_BM = make_smem_desc(smem_V_BM, 128, 1024);
            uint64_t desc_V_BN = make_smem_desc(smem_V_BN, 128, 1024);
            
            umma_f16_cg1_fn(tmem_dP, desc_dO_BM, desc_V_BM, idesc_dP_OV, 0);
            umma_f16_cg1_fn(tmem_dP, desc_dO_BN, desc_V_BN, idesc_dP_OV, 1);
        }
        umma_commit_1sm_fn(bar_A);
        mbarrier_wait_fn(bar_A, phase_A);
        phase_A ^= 1;

        // Explicitly compute gradient components directly within safe shared-memory boundaries 
        for(int i = 0; i < 128; i++) {
            int r = i / 2;
            int c = (i % 2) * 32 + (i % 32);
            
            uint32_t read_ptr = tmem_dP + r * 64 + c;
            uint32_t val;
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(val) : "r"(read_ptr));
            float dp_val = __uint_as_float(val);
            
            float p_bm = __bfloat162float(smem_P_BM[swizzle_none_idx(r, c)]);
            float p_bn = __bfloat162float(smem_P_BN[swizzle_none_idx(r, c)]);
            
            float ds_bm = p_bm * (dp_val - head_dO_O_shared[r]) * scale;
            float ds_bn = p_bn * (dp_val - head_dO_O_shared[r]) * scale;
            
            // Guard causal sequence strictly
            if ((q_blk * BM + r) < (k_blk * BN + c)) {
                ds_bm = 0.0f;
                ds_bn = 0.0f;
            }
            
            smem_dO_BM[swizzle_none_idx(r, c)] = __float2bfloat16(ds_bm);
            smem_dO_BN[swizzle_none_idx(r, c)] = __float2bfloat16(ds_bn);
        }
        __syncthreads();
        fence_proxy_async_fn();

        if (tid < 128) {
            uint64_t desc_dS_BM = make_smem_desc(smem_dO_BM, 128, 1024);
            uint64_t desc_Q_BM = make_smem_desc(smem_Q_BM, 1024, 128);
            umma_f16_cg1_fn(tmem_dK_BM, desc_dS_BM, desc_Q_BM, idesc_dK_dS_Q, (q_blk == k_blk) ? 0 : 1);
            
            uint64_t desc_dS_BN = make_smem_desc(smem_dO_BN, 128, 1024);
            uint64_t desc_Q_BN = make_smem_desc(smem_Q_BN, 1024, 128);
            umma_f16_cg1_fn(tmem_dK_BN, desc_dS_BN, desc_Q_BN, idesc_dK_dS_Q, (q_blk == k_blk) ? 0 : 1);
        }
        umma_commit_1sm_fn(bar_A);
        mbarrier_wait_fn(bar_A, phase_A);
        phase_A ^= 1;
    }

    // Coalesced Epilogue Layout directly writing to Global Memory avoiding TMEM mapping failures
    for (int i = tid; i < BM * BN; i += blockDim.x) {
        int r = i / BN;
        int c = i % BN;
        
        uint32_t read_ptr_BM = tmem_dK_BM + r * BN + c;
        uint32_t val_BM;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(val_BM) : "r"(read_ptr_BM));
        dK_gmem[(bh * S + k_blk * BN + r) * 128 + c] = __float2bfloat16(__uint_as_float(val_BM));
        
        uint32_t read_ptr_BN = tmem_dK_BN + r * BN + c;
        uint32_t val_BN;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(val_BN) : "r"(read_ptr_BN));
        dK_gmem[(bh * S + k_blk * BN + r) * 128 + 64 + c] = __float2bfloat16(__uint_as_float(val_BN));
    }

    if (tid == 0) {
        tmem_dealloc_fn(tmem_dK_BM, 64);
        tmem_dealloc_fn(tmem_dK_BN, 64);
        tmem_dealloc_fn(tmem_S, 64);
        tmem_dealloc_fn(tmem_dP, 64);
    }
}

__global__ void bwd_dV_kernel(
    const __grid_constant__ CUtensorMap* tma_Q, const __grid_constant__ CUtensorMap* tma_K,
    const __grid_constant__ CUtensorMap* tma_V, const __grid_constant__ CUtensorMap* tma_dO,
    const __grid_constant__ CUtensorMap* tma_O, const __grid_constant__ CUtensorMap* tma_P,
    const float* L_gmem,
    __nv_bfloat16* dV_gmem,
    int S, float scale) 
{
    int k_blk = blockIdx.x;
    int bh = blockIdx.y;
    int tid = threadIdx.x;

    constexpr int BM = 64;
    constexpr int BN = 64;

    extern __shared__ char smem_buf[];
    __nv_bfloat16* smem_Q_BM = (__nv_bfloat16*)smem_buf;
    __nv_bfloat16* smem_Q_BN = smem_Q_BM + BM * BN;
    __nv_bfloat16* smem_dO_BM = smem_Q_BN + BM * BN;
    __nv_bfloat16* smem_dO_BN = smem_dO_BM + BM * BN;
    __nv_bfloat16* smem_O_BM = smem_dO_BN + BM * BN;
    __nv_bfloat16* smem_O_BN = smem_O_BM + BM * BN;
    __nv_bfloat16* smem_K_BM = smem_O_BN + BM * BN;
    __nv_bfloat16* smem_K_BN = smem_K_BM + BM * BN;
    __nv_bfloat16* smem_V_BM = smem_K_BN + BM * BN;
    __nv_bfloat16* smem_V_BN = smem_V_BM + BM * BN;
    __nv_bfloat16* smem_P_BM = smem_V_BN + BM * BN;
    __nv_bfloat16* smem_P_BN = smem_P_BM + BM * BN;

    __align__(16) uint64_t* bar_A = (uint64_t*)(smem_P_BN + BM * BN);
    __align__(16) uint64_t* bar_B = bar_A + 1;

    if (tid == 0) {
        init_smem_barrier_fn(bar_A, 1);
        init_smem_barrier_fn(bar_B, 1);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t phase_A = 0;
    uint32_t tmem_dV_BM, tmem_dV_BN;
    if (tid == 0) {
        tmem_alloc_fn(&tmem_dV_BM, 64);
        tmem_alloc_fn(&tmem_dV_BN, 64);
    }
    __syncthreads();

    uint32_t idesc_dV_dS_dO = make_instr_desc_fn(BM, BN);

    int num_S_blocks = (S + BM - 1) / BM;
    uint32_t phase_B = 0;

    for (int q_blk = 0; q_blk <= k_blk && q_blk < num_S_blocks; q_blk++) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_B, sizeof(__nv_bfloat16) * BM * BN * 4);
            tma_load_2d_fn(&tma_dO[bh], bar_B, smem_dO_BM, 0, bh * S + q_blk * BM);
            tma_load_2d_fn(&tma_dO[bh], bar_B, smem_dO_BN, 64, bh * S + q_blk * BM);
            tma_load_2d_fn(&tma_P[bh], bar_B, smem_P_BM, k_blk * BN, bh * S + q_blk * BM);
            tma_load_2d_fn(&tma_P[bh], bar_B, smem_P_BN, k_blk * BN + 64, bh * S + q_blk * BM);
        }
        mbarrier_wait_fn(bar_B, phase_B);
        phase_B ^= 1;

        if (tid < 128) {
            uint64_t desc_dS_BM = make_smem_desc(smem_P_BM, 128, 1024);
            uint64_t desc_dO_BM = make_smem_desc(smem_dO_BM, 1024, 128);
            umma_f16_cg1_fn(tmem_dV_BM, desc_dS_BM, desc_dO_BM, idesc_dV_dS_dO, (q_blk == 0) ? 0 : 1);
            
            uint64_t desc_dS_BN = make_smem_desc(smem_P_BN, 128, 1024);
            uint64_t desc_dO_BN = make_smem_desc(smem_dO_BN, 1024, 128);
            umma_f16_cg1_fn(tmem_dV_BN, desc_dS_BN, desc_dO_BN, idesc_dV_dS_dO, (q_blk == 0) ? 0 : 1);
        }
        umma_commit_1sm_fn(bar_A);
        mbarrier_wait_fn(bar_A, phase_A);
        phase_A ^= 1;
    }

    // Coalesced Epilogue Layout directly writing to Global Memory avoiding TMEM mapping pitfalls
    for (int i = tid; i < BM * BN; i += blockDim.x) {
        int r = i / BN;
        int c = i % BN;
        
        uint32_t read_ptr_BM = tmem_dV_BM + r * BN + c;
        uint32_t val_BM;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(val_BM) : "r"(read_ptr_BM));
        dV_gmem[(bh * S + k_blk * BN + r) * 128 + c] = __float2bfloat16(__uint_as_float(val_BM));
        
        uint32_t read_ptr_BN = tmem_dV_BN + r * BN + c;
        uint32_t val_BN;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(val_BN) : "r"(read_ptr_BN));
        dV_gmem[(bh * S + k_blk * BN + r) * 128 + 64 + c] = __float2bfloat16(__uint_as_float(val_BN));
    }

    if (tid == 0) {
        tmem_dealloc_fn(tmem_dV_BM, 64);
        tmem_dealloc_fn(tmem_dV_BN, 64);
    }
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

namespace tvm_ffi_example_cuda {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = Q.size(3); 

    float scale = 1.0f / sqrtf((float)d);

    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_ptr = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_ptr = static_cast<const float*>(L.data_ptr());
    
    __nv_bfloat16* dQ_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());

    // Allocate native CUtensorMaps seamlessly via TVM-FFI
    CUtensorMap* tma_Q = (CUtensorMap*)malloc(sizeof(CUtensorMap) * B * H);
    CUtensorMap* tma_K = (CUtensorMap*)malloc(sizeof(CUtensorMap) * B * H);
    CUtensorMap* tma_V = (CUtensorMap*)malloc(sizeof(CUtensorMap) * B * H);
    CUtensorMap* tma_O = (CUtensorMap*)malloc(sizeof(CUtensorMap) * B * H);
    CUtensorMap* tma_dO = (CUtensorMap*)malloc(sizeof(CUtensorMap) * B * H);
    CUtensorMap* tma_P = (CUtensorMap*)malloc(sizeof(CUtensorMap) * B * H);

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    for (int bh = 0; bh < B * H; bh++) {
        create_tma_2d_descriptor_2B(&tma_Q[bh], (const char*)Q_ptr + bh * S * d * sizeof(__nv_bfloat16), d, B * H * S, 64, 64, 
            CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        create_tma_2d_descriptor_2B(&tma_K[bh], (const char*)K_ptr + bh * S * d * sizeof(__nv_bfloat16), d, B * H * S, 64, 64, 
            CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        create_tma_2d_descriptor_2B(&tma_V[bh], (const char*)V_ptr + bh * S * d * sizeof(__nv_bfloat16), d, B * H * S, 64, 64, 
            CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        create_tma_2d_descriptor_2B(&tma_O[bh], (const char*)O_ptr + bh * S * d * sizeof(__nv_bfloat16), d, B * H * S, 64, 64, 
            CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        create_tma_2d_descriptor_2B(&tma_dO[bh], (const char*)dO_ptr + bh * S * d * sizeof(__nv_bfloat16), d, B * H * S, 64, 64, 
            CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        create_tma_2d_descriptor_2B(&tma_P[bh], (const char*)O_ptr + bh * S * d * sizeof(__nv_bfloat16), d, B * H * S, 64, 64, 
            CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    }