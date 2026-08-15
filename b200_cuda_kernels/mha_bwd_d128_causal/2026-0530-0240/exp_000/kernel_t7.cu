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

namespace mha_bwd_sm100 {

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

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_before_fn() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void umma_f16_cg1(uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, int accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_and_wait(uint64_t* mbar, int phase) {
    uint32_t mbar_addr = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(mbar_addr));
    mbarrier_wait_fn(mbar, phase);
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    // FP32 output
    d |= (1u << 7);    // BF16 A
    d |= (1u << 10);   // BF16 B
    d |= (0u << 15);   // A is K-Major (row-major)
    d |= (0u << 16);   // B is K-Major (row-major)
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_K_major(void* ptr, uint32_t stride_bytes, uint32_t MN_dim) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    uint32_t sbo = 8 * stride_bytes;
    uint32_t lbo = (MN_dim / 8) * sbo;
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46; // SM100 version
    d |= (uint64_t)((addr >> 7) & 7) << 49;
    return d;
}

__device__ __forceinline__ void load_async_2(
    void const* g1, void* s1, int b1,
    void const* g2, void* s2, int b2,
    uint64_t* mbar) 
{
    uint32_t smem_mbar = (uint32_t)__cvta_generic_to_shared(mbar);
    if (threadIdx.x == 0) {
        int total_bytes = b1 + b2;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_mbar), "r"(total_bytes) : "memory");
        
        uint32_t s1_p = (uint32_t)__cvta_generic_to_shared(s1);
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
            :: "r"(s1_p), "l"(g1), "r"(b1), "r"(smem_mbar) : "memory");
            
        uint32_t s2_p = (uint32_t)__cvta_generic_to_shared(s2);
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
            :: "r"(s2_p), "l"(g2), "r"(b2), "r"(smem_mbar) : "memory");
    } else {
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_mbar) : "memory");
    }
}

__device__ __forceinline__ void load_async_3(
    void const* g1, void* s1, int b1,
    void const* g2, void* s2, int b2,
    void const* g3, void* s3, int b3,
    uint64_t* mbar) 
{
    uint32_t smem_mbar = (uint32_t)__cvta_generic_to_shared(mbar);
    if (threadIdx.x == 0) {
        int total_bytes = b1 + b2 + b3;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_mbar), "r"(total_bytes) : "memory");
        
        uint32_t s1_p = (uint32_t)__cvta_generic_to_shared(s1);
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
            :: "r"(s1_p), "l"(g1), "r"(b1), "r"(smem_mbar) : "memory");
            
        uint32_t s2_p = (uint32_t)__cvta_generic_to_shared(s2);
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
            :: "r"(s2_p), "l"(g2), "r"(b2), "r"(smem_mbar) : "memory");
            
        uint32_t s3_p = (uint32_t)__cvta_generic_to_shared(s3);
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
            :: "r"(s3_p), "l"(g3), "r"(b3), "r"(smem_mbar) : "memory");
    } else {
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_mbar) : "memory");
    }
}

__device__ __forceinline__ void copy_s2g_safe(void* smem, void* gmem, int bytes) {
    uint32_t smem_int = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
        :: "l"(gmem), "r"(smem_int), "r"(bytes) : "memory");
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
    asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");
}

__global__ void kernel_pass1_dQ(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    int B, int H, int S) 
{
    int i_start = blockIdx.x * 64;
    if (i_start >= S) return;
    int rem_rows = (i_start + 64 > S) ? (S - i_start) : 64;
    int bytes_128 = rem_rows * 128 * sizeof(__nv_bfloat16);
    
    int b = blockIdx.y / H;
    int h = blockIdx.y % H;
    long long batch_head_offset = (long long)(b * H + h) * S * 128;
    
    extern __shared__ char smem[];
    __nv_bfloat16* smem_Q = (__nv_bfloat16*)smem;
    __nv_bfloat16* smem_dO = smem_Q + 64 * 128;
    __nv_bfloat16* smem_O = smem_dO + 64 * 128;
    __nv_bfloat16* smem_K = smem_O + 64 * 128;
    __nv_bfloat16* smem_V = smem_K + 64 * 128;
    __nv_bfloat16* smem_K_T = smem_V + 64 * 128;
    __nv_bfloat16* smem_V_T = smem_K_T + 128 * 64;
    __nv_bfloat16* smem_dS = smem_V_T + 128 * 64;
    float* smem_delta = (float*)(smem_dS + 64 * 64);
    uint64_t* mbar_load = (uint64_t*)(smem_delta + 64);
    uint64_t* mbar_mma = mbar_load + 1;
    uint32_t* tmem_addr = (uint32_t*)(mbar_mma + 1);
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_load, 128);
        init_smem_barrier_fn(mbar_mma, 1);
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    if (threadIdx.x < 32) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" 
            :: "r"((uint32_t)__cvta_generic_to_shared(tmem_addr)), "r"(256));
    }
    __syncthreads();
    
    int load_phase = 0;
    int mma_phase = 0;
    
    load_async_3(
        Q + batch_head_offset + i_start * 128, smem_Q, bytes_128,
        dO + batch_head_offset + i_start * 128, smem_dO, bytes_128,
        O + batch_head_offset + i_start * 128, smem_O, bytes_128,
        mbar_load
    );
    mbarrier_wait_fn(mbar_load, load_phase);
    load_phase ^= 1;
    
    if (rem_rows < 64) {
        for (int i = threadIdx.x; i < (64 - rem_rows) * 128; i += blockDim.x) {
            smem_Q[rem_rows * 128 + i] = __float2bfloat16(0.0f);
            smem_dO[rem_rows * 128 + i] = __float2bfloat16(0.0f);
            smem_O[rem_rows * 128 + i] = __float2bfloat16(0.0f);
        }
    }
    __syncthreads();
    
    if (threadIdx.x < 64) {
        float sum = 0.0f;
        for (int c = 0; c < 128; ++c) {
            float do_val = __bfloat162float(smem_dO[threadIdx.x * 128 + c]);
            float o_val  = __bfloat162float(smem_O[threadIdx.x * 128 + c]);
            sum += do_val * o_val;
        }
        smem_delta[threadIdx.x] = sum;
    }
    __syncthreads();
    
    uint32_t idesc_S = make_instr_desc_fn(64, 64); 
    uint32_t idesc_dP = make_instr_desc_fn(64, 64); 
    uint32_t idesc_dQ = make_instr_desc_fn(64, 128); 
    
    for (int j_start = 0; j_start <= i_start; j_start += 64) {
        int rem_rows_j = (j_start + 64 > S) ? (S - j_start) : 64;
        int bytes_128_j = rem_rows_j * 128 * sizeof(__nv_bfloat16);
        
        load_async_2(
            K + batch_head_offset + j_start * 128, smem_K, bytes_128_j,
            V + batch_head_offset + j_start * 128, smem_V, bytes_128_j,
            mbar_load
        );
        mbarrier_wait_fn(mbar_load, load_phase);
        load_phase ^= 1;
        
        if (rem_rows_j < 64) {
            for (int i = threadIdx.x; i < (64 - rem_rows_j) * 128; i += blockDim.x) {
                smem_K[rem_rows_j * 128 + i] = __float2bfloat16(0.0f);
                smem_V[rem_rows_j * 128 + i] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();
        
        for (int i = threadIdx.x; i < 64 * 128; i += blockDim.x) {
            int r = i / 128;
            int c = i % 128;
            smem_K_T[c * 64 + r] = smem_K[i];
            smem_V_T[c * 64 + r] = smem_V[i];
        }
        __syncthreads();
        
        if (threadIdx.x == 0) {
            tcgen05_fence_after_fn();
            for (int k = 0; k < 128; k += 16) {
                uint64_t desc_Q_k = make_smem_desc_K_major(smem_Q + k, 256, 64);
                uint64_t desc_K_T_k = make_smem_desc_K_major(smem_K_T + k * 64, 128, 64);
                uint64_t desc_dO_k = make_smem_desc_K_major(smem_dO + k, 256, 64);
                uint64_t desc_V_T_k = make_smem_desc_K_major(smem_V_T + k * 64, 128, 64);
                int accum = (k > 0) ? 1 : 0;
                umma_f16_cg1(*tmem_addr + 128, desc_Q_k, desc_K_T_k, idesc_S, accum);
                umma_f16_cg1(*tmem_addr + 192, desc_dO_k, desc_V_T_k, idesc_dP, accum);
            }
            tcgen05_fence_before_fn();
            umma_commit_and_wait(mbar_mma, mma_phase);
            mma_phase ^= 1;
        }
        __syncthreads();
        
        tcgen05_fence_after_fn();
        const float* L_ptr = L + (long long)(b * H + h) * S + i_start;
        
        for (int c = 0; c < 64; c += 16) {
            uint32_t rS[16], rdP[16];
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
               : "=r"(rS[0]), "=r"(rS[1]), "=r"(rS[2]), "=r"(rS[3]),
                 "=r"(rS[4]), "=r"(rS[5]), "=r"(rS[6]), "=r"(rS[7]),
                 "=r"(rS[8]), "=r"(rS[9]), "=r"(rS[10]), "=r"(rS[11]),
                 "=r"(rS[12]), "=r"(rS[13]), "=r"(rS[14]), "=r"(rS[15])
               : "r"(*tmem_addr + 128 + c));
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
               : "=r"(rdP[0]), "=r"(rdP[1]), "=r"(rdP[2]), "=r"(rdP[3]),
                 "=r"(rdP[4]), "=r"(rdP[5]), "=r"(rdP[6]), "=r"(rdP[7]),
                 "=r"(rdP[8]), "=r"(rdP[9]), "=r"(rdP[10]), "=r"(rdP[11]),
                 "=r"(rdP[12]), "=r"(rdP[13]), "=r"(rdP[14]), "=r"(rdP[15])
               : "r"(*tmem_addr + 192 + c));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            
            int warp = threadIdx.x / 32;
            int lane = threadIdx.x % 32;
            int row = warp * 32 + lane;
            
            if (row < 64) {
                float l_val = (i_start + row < S) ? L_ptr[row] : 0.0f;
                float delta_val = smem_delta[row];
                for (int i = 0; i < 16; ++i) {
                    float s_val = __uint_as_float(rS[i]);
                    float dp_val = __uint_as_float(rdP[i]);
                    float p_val = 0.0f;
                    int col = c + i;
                    int global_row = i_start + row;
                    int global_col = j_start + col;
                    if (global_col <= global_row && global_row < S && global_col < S) {
                        p_val = expf(s_val * 0.08838834764f - l_val);
                    }
                    float ds_val = p_val * (dp_val - delta_val) * 0.08838834764f;
                    smem_dS[row * 64 + col] = __float2bfloat16(ds_val);
                }
            }
        }
        __syncthreads();
        
        if (threadIdx.x == 0) {
            tcgen05_fence_after_fn();
            for (int k = 0; k < 64; k += 16) {
                uint64_t desc_dS_k = make_smem_desc_K_major(smem_dS + k, 128, 64);
                uint64_t desc_K_k = make_smem_desc_K_major(smem_K + k * 128, 256, 128);
                int accum = (j_start == 0 && k == 0) ? 0 : 1;
                umma_f16_cg1(*tmem_addr + 0, desc_dS_k, desc_K_k, idesc_dQ, accum);
            }
            tcgen05_fence_before_fn();
            umma_commit_and_wait(mbar_mma, mma_phase);
            mma_phase ^= 1;
        }
        __syncthreads();
    }
    
    tcgen05_fence_after_fn();
    for (int c = 0; c < 128; c += 16) {
        uint32_t rdQ[16];
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
           : "=r"(rdQ[0]), "=r"(rdQ[1]), "=r"(rdQ[2]), "=r"(rdQ[3]),
             "=r"(rdQ[4]), "=r"(rdQ[5]), "=r"(rdQ[6]), "=r"(rdQ[7]),
             "=r"(rdQ[8]), "=r"(rdQ[9]), "=r"(rdQ[10]), "=r"(rdQ[11]),
             "=r"(rdQ[12]), "=r"(rdQ[13]), "=r"(rdQ[14]), "=r"(rdQ[15])
           : "r"(*tmem_addr + 0 + c));
        asm volatile("tcgen05.wait::ld.sync.aligned;");
        
        int warp = threadIdx.x / 32;
        int lane = threadIdx.x % 32;
        int row = warp * 32 + lane;
        
        if (row < 64) {
            for (int i = 0; i < 16; ++i) {
                smem_Q[row * 128 + c + i] = __float2bfloat16(__uint_as_float(rdQ[i]));
            }
        }
    }
    __syncthreads();
    
    if (threadIdx.x < 32) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" 
            :: "r"(*tmem_addr), "r"(256));
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        copy_s2g_safe(smem_Q, dQ + batch_head_offset + i_start * 128, bytes_128);
    }
    __syncthreads();
}

__global__ void kernel_pass2_dK_dV(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int B, int H, int S) 
{
    int j_start = blockIdx.x * 64;
    if (j_start >= S) return;
    int rem_rows = (j_start + 64 > S) ? (S - j_start) : 64;
    int bytes_128 = rem_rows * 128 * sizeof(__nv_bfloat16);
    
    int b = blockIdx.y / H;
    int h = blockIdx.y % H;
    long long batch_head_offset = (long long)(b * H + h) * S * 128;
    
    extern __shared__ char smem[];
    __nv_bfloat16* smem_K = (__nv_bfloat16*)smem;
    __nv_bfloat16* smem_V = smem_K + 64 * 128;
    __nv_bfloat16* smem_Q = smem_V + 64 * 128;
    __nv_bfloat16* smem_dO = smem_Q + 64 * 128;
    __nv_bfloat16* smem_O = smem_dO + 64 * 128;
    __nv_bfloat16* smem_K_T = smem_O + 64 * 128;
    __nv_bfloat16* smem_V_T = smem_K_T + 128 * 64;
    __nv_bfloat16* smem_dS = smem_V_T + 128 * 64;
    __nv_bfloat16* smem_P = smem_dS + 64 * 64;
    __nv_bfloat16* smem_dS_T = smem_P + 64 * 64;
    __nv_bfloat16* smem_P_T = smem_dS_T + 64 * 64;
    float* smem_delta = (float*)(smem_P_T + 64 * 64);
    uint64_t* mbar_load = (uint64_t*)(smem_delta + 64);
    uint64_t* mbar_mma = mbar_load + 1;
    uint32_t* tmem_addr = (uint32_t*)(mbar_mma + 1);
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_load, 128);
        init_smem_barrier_fn(mbar_mma, 1);
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    if (threadIdx.x < 32) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" 
            :: "r"((uint32_t)__cvta_generic_to_shared(tmem_addr)), "r"(512));
    }
    __syncthreads();
    
    int load_phase = 0;
    int mma_phase = 0;
    
    load_async_2(
        K + batch_head_offset + j_start * 128, smem_K, bytes_128,
        V + batch_head_offset + j_start * 128, smem_V, bytes_128,
        mbar_load
    );
    mbarrier_wait_fn(mbar_load, load_phase);
    load_phase ^= 1;
    
    if (rem_rows < 64) { 
        for (int i = threadIdx.x; i < (64 - rem_rows) * 128; i += blockDim.x) {
            smem_K[rem_rows * 128 + i] = __float2bfloat16(0.0f);
            smem_V[rem_rows * 128 + i] = __float2bfloat16(0.0f);
        }
    }
    __syncthreads();
    
    for (int i = threadIdx.x; i < 64 * 128; i += blockDim.x) {
        int r = i / 128;
        int c = i % 128;
        smem_K_T[c * 64 + r] = smem_K[i];
        smem_V_T[c * 64 + r] = smem_V[i];
    }
    __syncthreads();
    
    uint32_t idesc_S = make_instr_desc_fn(64, 64); 
    uint32_t idesc_dP = make_instr_desc_fn(64, 64); 
    uint32_t idesc_dK = make_instr_desc_fn(64, 128); 
    uint32_t idesc_dV = make_instr_desc_fn(64, 128); 
    
    for (int i_start = j_start; i_start < S; i_start += 64) {
        int rem_rows_i = (i_start + 64 > S) ? (S - i_start) : 64;
        int bytes_128_i = rem_rows_i * 128 * sizeof(__nv_bfloat16);
        
        load_async_3(
            Q + batch_head_offset + i_start * 128, smem_Q, bytes_128_i,
            dO + batch_head_offset + i_start * 128, smem_dO, bytes_128_i,
            O + batch_head_offset + i_start * 128, smem_O, bytes_128_i,
            mbar_load
        );
        mbarrier_wait_fn(mbar_load, load_phase);
        load_phase ^= 1;
        
        if (rem_rows_i < 64) {
            for (int i = threadIdx.x; i < (64 - rem_rows_i) * 128; i += blockDim.x) {
                smem_Q[rem_rows_i * 128 + i] = __float2bfloat16(0.0f);
                smem_dO[rem_rows_i * 128 + i] = __float2bfloat16(0.0f);
                smem_O[rem_rows_i * 128 + i] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();
        
        if (threadIdx.x < 64) {
            float sum = 0.0f;
            for (int c = 0; c < 128; ++c) {
                float do_val = __bfloat162float(smem_dO[threadIdx.x * 128 + c]);
                float o_val  = __bfloat162float(smem_O[threadIdx.x * 128 + c]);
                sum += do_val * o_val;
            }
            smem_delta[threadIdx.x] = sum;
        }
        __syncthreads();
        
        if (threadIdx.x == 0) {
            tcgen05_fence_after_fn();
            for (int k = 0; k < 128; k += 16) {
                uint64_t desc_Q_k = make_smem_desc_K_major(smem_Q + k, 256, 64);
                uint64_t desc_K_T_k = make_smem_desc_K_major(smem_K_T + k * 64, 128, 64);
                uint64_t desc_dO_k = make_smem_desc_K_major(smem_dO + k, 256, 64);
                uint64_t desc_V_T_k = make_smem_desc_K_major(smem_V_T + k * 64, 128, 64);
                int accum = (k > 0) ? 1 : 0;
                umma_f16_cg1(*tmem_addr + 256, desc_Q_k, desc_K_T_k, idesc_S, accum);
                umma_f16_cg1(*tmem_addr + 320, desc_dO_k, desc_V_T_k, idesc_dP, accum);
            }
            tcgen05_fence_before_fn();
            umma_commit_and_wait(mbar_mma, mma_phase);
            mma_phase ^= 1;
        }
        __syncthreads();
        
        tcgen05_fence_after_fn();
        const float* L_ptr = L + (long long)(b * H + h) * S + i_start;
        
        for (int c = 0; c < 64; c += 16) {
            uint32_t rS[16], rdP[16];
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
               : "=r"(rS[0]), "=r"(rS[1]), "=r"(rS[2]), "=r"(rS[3]),
                 "=r"(rS[4]), "=r"(rS[5]), "=r"(rS[6]), "=r"(rS[7]),
                 "=r"(rS[8]), "=r"(rS[9]), "=r"(rS[10]), "=r"(rS[11]),
                 "=r"(rS[12]), "=r"(rS[13]), "=r"(rS[14]), "=r"(rS[15])
               : "r"(*tmem_addr + 256 + c));
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
               : "=r"(rdP[0]), "=r"(rdP[1]), "=r"(rdP[2]), "=r"(rdP[3]),
                 "=r"(rdP[4]), "=r"(rdP[5]), "=r"(rdP[6]), "=r"(rdP[7]),
                 "=r"(rdP[8]), "=r"(rdP[9]), "=r"(rdP[10]), "=r"(rdP[11]),
                 "=r"(rdP[12]), "=r"(rdP[13]), "=r"(rdP[14]), "=r"(rdP[15])
               : "r"(*tmem_addr + 320 + c));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            
            int warp = threadIdx.x / 32;
            int lane = threadIdx.x % 32;
            int row = warp * 32 + lane;
            
            if (row < 64) {
                float l_val = (i_start + row < S) ? L_ptr[row] : 0.0f;
                float delta_val = smem_delta[row];
                for (int i = 0; i < 16; ++i) {
                    float s_val = __uint_as_float(rS[i]);
                    float dp_val = __uint_as_float(rdP[i]);
                    float p_val = 0.0f;
                    int col = c + i;
                    int global_row = i_start + row;
                    int global_col = j_start + col;
                    if (global_col <= global_row && global_row < S && global_col < S) {
                        p_val = expf(s_val * 0.08838834764f - l_val);
                    }
                    float ds_val = p_val * (dp_val - delta_val) * 0.08838834764f;
                    smem_dS[row * 64 + col] = __float2bfloat16(ds_val);
                    smem_P[row * 64 + col] = __float2bfloat16(p_val);
                }
            }
        }
        __syncthreads();
        
        for (int i = threadIdx.x; i < 64 * 64; i += blockDim.x) {
            int r = i / 64;
            int c = i % 64;
            smem_dS_T[c * 64 + r] = smem_dS[i];
            smem_P_T[c * 64 + r] = smem_P[i];
        }
        __syncthreads();
        
        if (threadIdx.x == 0) {
            tcgen05_fence_after_fn();
            for (int k = 0; k < 64; k += 16) {
                uint64_t desc_dS_T_k = make_smem_desc_K_major(smem_dS_T + k, 128, 64);
                uint64_t desc_Q_k = make_smem_desc_K_major(smem_Q + k * 128, 256, 128);
                uint64_t desc_P_T_k = make_smem_desc_K_major(smem_P_T + k, 128, 64);
                uint64_t desc_dO_k = make_smem_desc_K_major(smem_dO + k * 128, 256, 128);
                int accum = (i_start == j_start && k == 0) ? 0 : 1;
                umma_f16_cg1(*tmem_addr + 0, desc_dS_T_k, desc_Q_k, idesc_dK, accum);
                umma_f16_cg1(*tmem_addr + 128, desc_P_T_k, desc_dO_k, idesc_dV, accum);
            }
            tcgen05_fence_before_fn();
            umma_commit_and_wait(mbar_mma, mma_phase);
            mma_phase ^= 1;
        }
        __syncthreads();
    }
    
    tcgen05_fence_after_fn();
    for (int c = 0; c < 128; c += 16) {
        uint32_t rdK[16];
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
           : "=r"(rdK[0]), "=r"(rdK[1]), "=r"(rdK[2]), "=r"(rdK[3]),
             "=r"(rdK[4]), "=r"(rdK[5]), "=r"(rdK[6]), "=r"(rdK[7]),
             "=r"(rdK[8]), "=r"(rdK[9]), "=r"(rdK[10]), "=r"(rdK[11]),
             "=r"(rdK[12]), "=r"(rdK[13]), "=r"(rdK[14]), "=r"(rdK[15])
           : "r"(*tmem_addr + 0 + c));
        asm volatile("tcgen05.wait::ld.sync.aligned;");
        
        int warp = threadIdx.x / 32;
        int lane = threadIdx.x % 32;
        int row = warp * 32 + lane;
        if (row < 64) {
            for (int i = 0; i < 16; ++i) {
                smem_K[row * 128 + c + i] = __float2bfloat16(__uint_as_float(rdK[i]));
            }
        }
    }
    
    for (int c = 0; c < 128; c += 16) {
        uint32_t rdV[16];
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
           : "=r"(rdV[0]), "=r"(rdV[1]), "=r"(rdV[2]), "=r"(rdV[3]),
             "=r"(rdV[4]), "=r"(rdV[5]), "=r"(rdV[6]), "=r"(rdV[7]),
             "=r"(rdV[8]), "=r"(rdV[9]), "=r"(rdV[10]), "=r"(rdV[11]),
             "=r"(rdV[12]), "=r"(rdV[13]), "=r"(rdV[14]), "=r"(rdV[15])
           : "r"(*tmem_addr + 128 + c));
        asm volatile("tcgen05.wait::ld.sync.aligned;");
        
        int warp = threadIdx.x / 32;
        int lane = threadIdx.x % 32;
        int row = warp * 32 + lane;
        if (row < 64) {
            for (int i = 0; i < 16; ++i) {
                smem_V[row * 128 + c + i] = __float2bfloat16(__uint_as_float(rdV[i]));
            }
        }
    }
    __syncthreads();
    
    if (threadIdx.x < 32) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" 
            :: "r"(*tmem_addr), "r"(512));
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        copy_s2g_safe(smem_K, dK + batch_head_offset + j_start * 128, bytes_128);
        copy_s2g_safe(smem_V, dV + batch_head_offset + j_start * 128, bytes_128);
    }
    __syncthreads();
}

void run(tvm::ffi::TensorView Q_tv, tvm::ffi::TensorView K_tv, tvm::ffi::TensorView V_tv, 
         tvm::ffi::TensorView O_tv, tvm::ffi::TensorView dO_tv, tvm::ffi::TensorView L_tv,
         tvm::ffi::TensorView dQ_tv, tvm::ffi::TensorView dK_tv, tvm::ffi::TensorView dV_tv) {
    CUDA_CHECK(cudaSetDevice(Q_tv.device().device_id));
    
    int64_t B = Q_tv.size(0);
    int64_t H = Q_tv.size(1);
    int64_t S = Q_tv.size(2);
    
    __nv_bfloat16* Q = static_cast<__nv_bfloat16*>(Q_tv.data_ptr());
    __nv_bfloat16* K = static_cast<__nv_bfloat16*>(K_tv.data_ptr());
    __nv_bfloat16* V = static_cast<__nv_bfloat16*>(V_tv.data_ptr());
    __nv_bfloat16* O = static_cast<__nv_bfloat16*>(O_tv.data_ptr());
    __nv_bfloat16* dO = static_cast<__nv_bfloat16*>(dO_tv.data_ptr());
    float* L = static_cast<float*>(L_tv.data_ptr());
    __nv_bfloat16* dQ = static_cast<__nv_bfloat16*>(dQ_tv.data_ptr());
    __nv_bfloat16* dK = static_cast<__nv_bfloat16*>(dK_tv.data_ptr());
    __nv_bfloat16* dV = static_cast<__nv_bfloat16*>(dV_tv.data_ptr());
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q_tv.device().device_type, Q_tv.device().device_id));
    
    int grid_x = (S + 63) / 64;
    int grid_y = B * H;
    dim3 grid(grid_x, grid_y);
    dim3 block(128);
    
    size_t smem_pass1 = 124 * 1024;
    size_t smem_pass2 = 148 * 1024;
    
    CUDA_CHECK(cudaFuncSetAttribute(kernel_pass1_dQ, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_pass1));
    CUDA_CHECK(cudaFuncSetAttribute(kernel_pass2_dK_dV, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_pass2));
    
    kernel_pass1_dQ<<<grid, block, smem_pass1, stream>>>(Q, K, V, O, dO, L, dQ, B, H, S);
    kernel_pass2_dK_dV<<<grid, block, smem_pass2, stream>>>(Q, K, V, O, dO, L, dK, dV, B, H, S);
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_sm100::run);

}