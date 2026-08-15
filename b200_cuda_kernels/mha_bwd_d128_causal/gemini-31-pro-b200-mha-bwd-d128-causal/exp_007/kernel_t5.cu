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

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define CU_CHECK(call) do { \
    CUresult _e = (call); \
    if (_e != CUDA_SUCCESS) { \
        fprintf(stderr, "CU error %d at %s:%d\n", _e, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

__device__ __forceinline__ void tmem_alloc_cg1_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_cg1_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)0 << 61; // SWIZZLE_NONE
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, uint32_t a_major, uint32_t b_major) {
    uint32_t d = 0;
    d |= (1u << 4);    // FP32 output
    d |= (1u << 7);    // BF16 A
    d |= (1u << 10);   // BF16 B
    d |= (a_major << 15);
    d |= (b_major << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ void umma_f16_cg1_fn(uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_1sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(a));
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
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

__device__ __forceinline__ void tma_load_4d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}

__device__ __forceinline__ void fence_proxy_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
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
    asm("mov.b32 %0, {%1, %2};" : "=r"(result) : "h"(*reinterpret_cast<uint16_t*>(&a)), "h"(*reinterpret_cast<uint16_t*>(&b)));
    return result;
}

__device__ __forceinline__ void run_umma(
    uint32_t tmem_dst, void* A_smem, void* B_smem, 
    int M, int N, int K, bool accumulate,
    int a_major, int b_major,
    int lbo_A, int sbo_A, int lbo_B, int sbo_B,
    int step_A, int step_B) 
{
    uint32_t a_ptr = (uint32_t)__cvta_generic_to_shared(A_smem);
    uint32_t b_ptr = (uint32_t)__cvta_generic_to_shared(B_smem);
    uint32_t idesc = make_instr_desc_fn(M, N, a_major, b_major);
    for (int k = 0; k < K; k += 16) {
        uint64_t desc_a = make_smem_desc_sm100_fn((void*)a_ptr, lbo_A, sbo_A);
        uint64_t desc_b = make_smem_desc_sm100_fn((void*)b_ptr, lbo_B, sbo_B);
        uint32_t accum = (k == 0 && !accumulate) ? 0 : 1;
        umma_f16_cg1_fn(tmem_dst, desc_a, desc_b, idesc, accum);
        a_ptr += step_A;
        b_ptr += step_B;
    }
}

__device__ __forceinline__ void load_L_D(const float* L_base, const float* D_base, float* smem_L, float* smem_D, int valid_i, long l_s2) {
    int tid = threadIdx.x;
    if (tid < 64) {
        if (tid < valid_i) {
            smem_L[tid] = L_base[tid * l_s2];
            smem_D[tid] = D_base[tid];
        } else {
            smem_L[tid] = -INFINITY;
            smem_D[tid] = 0.0f;
        }
    }
}

__global__ void PrecomputeDKernel(const __nv_bfloat16* O, const __nv_bfloat16* dO, float* D, int S, int d,
                                  long o_s0, long o_s1, long o_s2, long o_s3,
                                  long do_s0, long do_s1, long do_s2, long do_s3) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int seq = blockIdx.x * blockDim.x + threadIdx.x;
    if (seq < S) {
        float sum = 0;
        const __nv_bfloat16* o_ptr = O + b * o_s0 + h * o_s1 + seq * o_s2;
        const __nv_bfloat16* do_ptr = dO + b * do_s0 + h * do_s1 + seq * do_s2;
        for (int i = 0; i < d; ++i) {
            sum += __bfloat162float(o_ptr[i * o_s3]) * __bfloat162float(do_ptr[i * do_s3]);
        }
        D[b * gridDim.y * S + h * S + seq] = sum;
    }
}

extern __shared__ __align__(128) char smem[];

__global__ void MhaBwdKernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* L, const float* D,
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    int S, int d,
    long l_s0, long l_s1, long l_s2,
    long dq_s0, long dq_s1, long dq_s2, long dq_s3,
    long dk_s0, long dk_s1, long dk_s2, long dk_s3,
    long dv_s0, long dv_s1, long dv_s2, long dv_s3
) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int j = blockIdx.x;
    
    int num_blocks = (S + 63) / 64;
    int valid_j = min(64, S - j * 64);
    if (valid_j <= 0) return;
    
    __nv_bfloat16* smem_K = (__nv_bfloat16*)smem;
    __nv_bfloat16* smem_V = smem_K + 64 * 128;
    __nv_bfloat16* smem_Q0 = smem_V + 64 * 128;
    __nv_bfloat16* smem_Q1 = smem_Q0 + 64 * 128;
    __nv_bfloat16* smem_dO0 = smem_Q1 + 64 * 128;
    __nv_bfloat16* smem_dO1 = smem_dO0 + 64 * 128;
    __nv_bfloat16* smem_P = smem_dO1 + 64 * 128;
    __nv_bfloat16* smem_P_T = smem_P + 64 * 64;
    __nv_bfloat16* smem_dS = smem_P_T + 64 * 64;
    __nv_bfloat16* smem_dS_T = smem_dS + 64 * 64;
    
    float* smem_L0 = (float*)(smem_dS_T + 64 * 64);
    float* smem_L1 = smem_L0 + 64;
    float* smem_D0 = smem_L1 + 64;
    float* smem_D1 = smem_D0 + 64;
    
    uint64_t* mbar_tma_KV = (uint64_t*)(smem_D1 + 64);
    uint64_t* mbar_tma_Q0 = mbar_tma_KV + 1;
    uint64_t* mbar_tma_Q1 = mbar_tma_Q0 + 1;
    uint64_t* mbar_umma = mbar_tma_Q1 + 1;
    uint32_t* tmem_alloc_addr = (uint32_t*)(mbar_umma + 1);

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if (warp_id == 0) {
        init_smem_barrier_fn(mbar_tma_KV, 1);
        init_smem_barrier_fn(mbar_tma_Q0, 1);
        init_smem_barrier_fn(mbar_tma_Q1, 1);
        init_smem_barrier_fn(mbar_umma, 1);
        fence_smem_barrier_init_fn();
        tmem_alloc_cg1_fn(tmem_alloc_addr, 512);
    }
    __syncthreads();
    
    uint32_t tmem_base = *tmem_alloc_addr;
    uint32_t tmem_dK = tmem_base;         
    uint32_t tmem_dV = tmem_base + 128;   
    uint32_t tmem_S  = tmem_base + 256;   
    uint32_t tmem_dP = tmem_base + 320;   
    uint32_t tmem_dQ = tmem_base + 384;   
    
    int phase_tma_KV = 0;
    int phase_tma_Q[2] = {0, 0};
    uint64_t* mbar_tma_Q[2] = {mbar_tma_Q0, mbar_tma_Q1};
    int phase_umma = 0;

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_tma_KV, 2 * 16384);
        tma_load_4d_fn(&tma_K, mbar_tma_KV, smem_K, 0, j * 64, h, b);
        tma_load_4d_fn(&tma_V, mbar_tma_KV, smem_V, 0, j * 64, h, b);
    }
    mbarrier_wait_fn(mbar_tma_KV, phase_tma_KV);
    
    __nv_bfloat16* smem_Q_ptr[2] = {smem_Q0, smem_Q1};
    __nv_bfloat16* smem_dO_ptr[2] = {smem_dO0, smem_dO1};
    float* smem_L_ptr[2] = {smem_L0, smem_L1};
    float* smem_D_ptr[2] = {smem_D0, smem_D1};

    int i = j;
    if (i < num_blocks) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_tma_Q[0], 2 * 16384);
            tma_load_4d_fn(&tma_Q, mbar_tma_Q[0], smem_Q_ptr[0], 0, i * 64, h, b);
            tma_load_4d_fn(&tma_dO, mbar_tma_Q[0], smem_dO_ptr[0], 0, i * 64, h, b);
        }
        const float* L_base = L + b * l_s0 + h * l_s1 + i * 64 * l_s2;
        const float* D_base = D + b * gridDim.y * S + h * S + i * 64;
        load_L_D(L_base, D_base, smem_L_ptr[0], smem_D_ptr[0], min(64, S - i * 64), l_s2);
    }
    
    float scale = 1.0f / sqrtf((float)d);

    for (int i = j; i < num_blocks; ++i) {
        int p = (i - j) % 2;
        int next_p = (p + 1) % 2;
        int next_i = i + 1;
        
        if (next_i < num_blocks) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(mbar_tma_Q[next_p], 2 * 16384);
                tma_load_4d_fn(&tma_Q, mbar_tma_Q[next_p], smem_Q_ptr[next_p], 0, next_i * 64, h, b);
                tma_load_4d_fn(&tma_dO, mbar_tma_Q[next_p], smem_dO_ptr[next_p], 0, next_i * 64, h, b);
            }
            const float* L_base = L + b * l_s0 + h * l_s1 + next_i * 64 * l_s2;
            const float* D_base = D + b * gridDim.y * S + h * S + next_i * 64;
            load_L_D(L_base, D_base, smem_L_ptr[next_p], smem_D_ptr[next_p], min(64, S - next_i * 64), l_s2);
        }
        
        mbarrier_wait_fn(mbar_tma_Q[p], phase_tma_Q[p]);
        phase_tma_Q[p] ^= 1;
        __syncthreads();
        fence_proxy_async_shared_fn();
        
        if (threadIdx.x == 0) {
            run_umma(tmem_S, smem_Q_ptr[p], smem_K, 64, 64, 128, false, 
                     0, 0, 16, 2048, 256, 16, 32, 32);
            run_umma(tmem_dP, smem_dO_ptr[p], smem_V, 64, 64, 128, false, 
                     0, 0, 16, 2048, 256, 16, 32, 32);
            umma_commit_1sm_fn(mbar_umma);
        }
        mbarrier_wait_fn(mbar_umma, phase_umma);
        phase_umma ^= 1;
        
        if (warp_id < 2) {
            int row = warp_id * 32 + lane_id;
            float L_val = smem_L_ptr[p][row];
            float D_val = smem_D_ptr[p][row];
            
            for (int c = 0; c < 64; c += 4) {
                uint32_t sr0, sr1, sr2, sr3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                             : "=r"(sr0),"=r"(sr1),"=r"(sr2),"=r"(sr3) : "r"(tmem_S + c));
                uint32_t dpr0, dpr1, dpr2, dpr3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                             : "=r"(dpr0),"=r"(dpr1),"=r"(dpr2),"=r"(dpr3) : "r"(tmem_dP + c));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                
                float s0 = __uint_as_float(sr0) * scale;
                float s1 = __uint_as_float(sr1) * scale;
                float s2 = __uint_as_float(sr2) * scale;
                float s3 = __uint_as_float(sr3) * scale;
                
                int global_row = i * 64 + row;
                int global_col = j * 64 + c;
                int valid_i = min(64, S - i * 64);
                
                if (global_row >= S || row >= valid_i) {
                    s0 = -INFINITY; s1 = -INFINITY; s2 = -INFINITY; s3 = -INFINITY;
                } else {
                    if (global_col + 0 > global_row || global_col + 0 >= S || c + 0 >= valid_j) s0 = -INFINITY;
                    if (global_col + 1 > global_row || global_col + 1 >= S || c + 1 >= valid_j) s1 = -INFINITY;
                    if (global_col + 2 > global_row || global_col + 2 >= S || c + 2 >= valid_j) s2 = -INFINITY;
                    if (global_col + 3 > global_row || global_col + 3 >= S || c + 3 >= valid_j) s3 = -INFINITY;
                }
                
                float p0 = fast_exp2f_fn((s0 - L_val) * 1.44269504089f);
                float p1 = fast_exp2f_fn((s1 - L_val) * 1.44269504089f);
                float p2 = fast_exp2f_fn((s2 - L_val) * 1.44269504089f);
                float p3 = fast_exp2f_fn((s3 - L_val) * 1.44269504089f);
                
                if (s0 == -INFINITY) p0 = 0.0f;
                if (s1 == -INFINITY) p1 = 0.0f;
                if (s2 == -INFINITY) p2 = 0.0f;
                if (s3 == -INFINITY) p3 = 0.0f;
                
                uint32_t p_packed0 = pack_bf16_fn(__float_as_uint(p0), __float_as_uint(p1));
                uint32_t p_packed1 = pack_bf16_fn(__float_as_uint(p2), __float_as_uint(p3));
                
                uint32_t p_offset = (row * 64 + c) * 2;
                *reinterpret_cast<uint32_t*>((char*)smem_P + p_offset) = p_packed0;
                *reinterpret_cast<uint32_t*>((char*)smem_P + p_offset + 4) = p_packed1;
                
                smem_P_T[(c + 0) * 64 + row] = __float2bfloat16(p0);
                smem_P_T[(c + 1) * 64 + row] = __float2bfloat16(p1);
                smem_P_T[(c + 2) * 64 + row] = __float2bfloat16(p2);
                smem_P_T[(c + 3) * 64 + row] = __float2bfloat16(p3);
                
                float dp0 = __uint_as_float(dpr0);
                float dp1 = __uint_as_float(dpr1);
                float dp2 = __uint_as_float(dpr2);
                float dp3 = __uint_as_float(dpr3);
                
                float ds0 = p0 * (dp0 - D_val) * scale;
                float ds1 = p1 * (dp1 - D_val) * scale;
                float ds2 = p2 * (dp2 - D_val) * scale;
                float ds3 = p3 * (dp3 - D_val) * scale;
                
                uint32_t ds_packed0 = pack_bf16_fn(__float_as_uint(ds0), __float_as_uint(ds1));
                uint32_t ds_packed1 = pack_bf16_fn(__float_as_uint(ds2), __float_as_uint(ds3));
                
                uint32_t ds_offset = (row * 64 + c) * 2;
                *reinterpret_cast<uint32_t*>((char*)smem_dS + ds_offset) = ds_packed0;
                *reinterpret_cast<uint32_t*>((char*)smem_dS + ds_offset + 4) = ds_packed1;
                
                smem_dS_T[(c + 0) * 64 + row] = __float2bfloat16(ds0);
                smem_dS_T[(c + 1) * 64 + row] = __float2bfloat16(ds1);
                smem_dS_T[(c + 2) * 64 + row] = __float2bfloat16(ds2);
                smem_dS_T[(c + 3) * 64 + row] = __float2bfloat16(ds3);
            }
        }
        __syncthreads();
        fence_proxy_async_shared_fn();
        
        if (threadIdx.x == 0) {
            run_umma(tmem_dV, smem_P_T, smem_dO_ptr[p], 64, 128, 64, (i != j), 
                     0, 1, 16, 1024, 16, 256, 32, 4096);
            run_umma(tmem_dK, smem_dS_T, smem_Q_ptr[p], 64, 128, 64, (i != j), 
                     0, 1, 16, 1024, 16, 256, 32, 4096);
            run_umma(tmem_dQ, smem_dS, smem_K, 64, 128, 64, false, 
                     0, 1, 16, 1024, 16, 256, 32, 4096);
            umma_commit_1sm_fn(mbar_umma);
        }
        mbarrier_wait_fn(mbar_umma, phase_umma);
        phase_umma ^= 1;
        
        if (warp_id < 2) {
            int row = warp_id * 32 + lane_id;
            for (int c = 0; c < 128; c += 4) {
                uint32_t dqr0, dqr1, dqr2, dqr3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                             : "=r"(dqr0),"=r"(dqr1),"=r"(dqr2),"=r"(dqr3) : "r"(tmem_dQ + c));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                uint32_t offset = (row * 128 + c) * 2;
                *reinterpret_cast<uint32_t*>((char*)smem_P + offset) = pack_bf16_fn(__float_as_uint(dqr0), __float_as_uint(dqr1));
                *reinterpret_cast<uint32_t*>((char*)smem_P + offset + 4) = pack_bf16_fn(__float_as_uint(dqr2), __float_as_uint(dqr3));
            }
        }
        __syncthreads();
        
        int valid_i = min(64, S - i * 64);
        for (int idx = threadIdx.x; idx < 64 * 16; idx += blockDim.x) {
            int r = idx >> 4;       
            int c = (idx & 15) << 3;
            int global_row = i * 64 + r;
            if (global_row < S && r < valid_i) {
                float4 val = *reinterpret_cast<float4*>((char*)smem_P + (r * 128 + c) * 2);
                __nv_bfloat162* out_ptr = reinterpret_cast<__nv_bfloat162*>(dQ + b * dq_s0 + h * dq_s1 + global_row * dq_s2) + c / 2;
                atomicAdd(out_ptr, *reinterpret_cast<__nv_bfloat162*>(&val.x));
                atomicAdd(out_ptr + 1, *reinterpret_cast<__nv_bfloat162*>(&val.y));
                atomicAdd(out_ptr + 2, *reinterpret_cast<__nv_bfloat162*>(&val.z));
                atomicAdd(out_ptr + 3, *reinterpret_cast<__nv_bfloat162*>(&val.w));
            }
        }
        __syncthreads();
    }
    
    if (warp_id < 2) {
        int row = warp_id * 32 + lane_id;
        for (int c = 0; c < 128; c += 4) {
            uint32_t dkr0, dkr1, dkr2, dkr3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(dkr0),"=r"(dkr1),"=r"(dkr2),"=r"(dkr3) : "r"(tmem_dK + c));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            uint32_t offset = (row * 128 + c) * 2;
            *reinterpret_cast<uint32_t*>((char*)smem_K + offset) = pack_bf16_fn(__float_as_uint(dkr0), __float_as_uint(dkr1));
            *reinterpret_cast<uint32_t*>((char*)smem_K + offset + 4) = pack_bf16_fn(__float_as_uint(dkr2), __float_as_uint(dkr3));
        }
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < 64 * 16; idx += blockDim.x) {
        int r = idx >> 4;
        int c = (idx & 15) << 3; 
        int global_row = j * 64 + r;
        if (global_row < S && r < valid_j) {
            *reinterpret_cast<float4*>(dK + b * dk_s0 + h * dk_s1 + global_row * dk_s2 + c) = *reinterpret_cast<float4*>((char*)smem_K + (r * 128 + c) * 2);
        }
    }
    
    __syncthreads();
    if (warp_id < 2) {
        int row = warp_id * 32 + lane_id;
        for (int c = 0; c < 128; c += 4) {
            uint32_t dvr0, dvr1, dvr2, dvr3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(dvr0),"=r"(dvr1),"=r"(dvr2),"=r"(dvr3) : "r"(tmem_dV + c));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            uint32_t offset = (row * 128 + c) * 2;
            *reinterpret_cast<uint32_t*>((char*)smem_V + offset) = pack_bf16_fn(__float_as_uint(dvr0), __float_as_uint(dvr1));
            *reinterpret_cast<uint32_t*>((char*)smem_V + offset + 4) = pack_bf16_fn(__float_as_uint(dvr2), __float_as_uint(dvr3));
        }
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < 64 * 16; idx += blockDim.x) {
        int r = idx >> 4;
        int c = (idx & 15) << 3; 
        int global_row = j * 64 + r;
        if (global_row < S && r < valid_j) {
            *reinterpret_cast<float4*>(dV + b * dv_s0 + h * dv_s1 + global_row * dv_s2 + c) = *reinterpret_cast<float4*>((char*)smem_V + (r * 128 + c) * 2);
        }
    }
    
    __syncthreads();
    if (warp_id == 0) {
        tmem_dealloc_cg1_fn(tmem_base, 512);
    }
}

CUresult create_tma_4d_descriptor_2B(CUtensorMap* d, void* globalAddress, 
                                     uint64_t dim0, uint64_t dim1, uint64_t dim2, uint64_t dim3,
                                     uint64_t stride1, uint64_t stride2, uint64_t stride3,
                                     uint32_t box0, uint32_t box1, uint32_t box2, uint32_t box3,
                                     CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[4] = {dim0, dim1, dim2, dim3};
    cuuint64_t globalStrides[3] = {stride1, stride2, stride3};
    cuuint32_t boxDim[4] = {box0, box1, box2, box3};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        4, 
        globalAddress,
        globalDim,
        globalStrides,
        boxDim, 
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

namespace tvm_ffi_mha_bwd {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int d = Q.size(3);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    CUDA_CHECK(cudaMemsetAsync(dQ.data_ptr(), 0, B * H * S * d * sizeof(uint16_t), stream));

    float* D_workspace = nullptr;
    CUDA_CHECK(cudaMallocAsync(&D_workspace, B * H * S * sizeof(float), stream));

    dim3 grid_D((S + 255) / 256, H, B);
    PrecomputeDKernel<<<grid_D, 256, 0, stream>>>(
        (const __nv_bfloat16*)O.data_ptr(), (const __nv_bfloat16*)dO.data_ptr(), D_workspace, S, d,
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3)
    );
    CUDA_CHECK(cudaGetLastError());
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_dO;
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_Q, Q.data_ptr(), d, S, H, B, 
             Q.stride(2)*2, Q.stride(1)*2, Q.stride(0)*2, 128, 64, 1, 1, CU_TENSOR_MAP_SWIZZLE_NONE));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_K, K.data_ptr(), d, S, H, B, 
             K.stride(2)*2, K.stride(1)*2, K.stride(0)*2, 128, 64, 1, 1, CU_TENSOR_MAP_SWIZZLE_NONE));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_V, V.data_ptr(), d, S, H, B, 
             V.stride(2)*2, V.stride(1)*2, V.stride(0)*2, 128, 64, 1, 1, CU_TENSOR_MAP_SWIZZLE_NONE));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_dO, dO.data_ptr(), d, S, H, B, 
             dO.stride(2)*2, dO.stride(1)*2, dO.stride(0)*2, 128, 64, 1, 1, CU_TENSOR_MAP_SWIZZLE_NONE));

    dim3 grid_mha((S + 63) / 64, H, B);
    dim3 block_mha(128);
    int smem_size = 136 * 1024;
    CUDA_CHECK(cudaFuncSetAttribute(MhaBwdKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    MhaBwdKernel<<<grid_mha, block_mha, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, tma_dO,
        (const float*)L.data_ptr(), D_workspace,
        (__nv_bfloat16*)dQ.data_ptr(), (__nv_bfloat16*)dK.data_ptr(), (__nv_bfloat16*)dV.data_ptr(),
        S, d,
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3)
    );
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaFreeAsync(D_workspace, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_bwd::run);

} // namespace tvm_ffi_mha_bwd