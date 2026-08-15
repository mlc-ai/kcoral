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

#define CU_CHECK(call) do {                                        \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        const char* err_str;                                       \
        cuGetErrorString(_e, &err_str);                            \
        fprintf(stderr, "CU error %s at %s:%d\n",                 \
                err_str, __FILE__, __LINE__);                      \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_example_cuda {

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
    uint32_t parity = phase & 1;
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(parity));
}

__device__ __forceinline__ void mbarrier_arrive_fn(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])) : "memory");
}

CUresult create_tma_4d_descriptor_2B(CUtensorMap* d, void* globalAddress, 
    uint64_t dim0, uint64_t dim1, uint64_t dim2, uint64_t dim3,
    uint32_t box0, uint32_t box1,
    CUtensorMapSwizzle swizzle) 
{
    cuuint64_t globalDim[4] = {dim0, dim1, dim2, dim3};
    cuuint64_t globalStrides[3] = {dim0 * 2, dim0 * dim1 * 2, dim0 * dim1 * dim2 * 2};
    cuuint32_t boxDim[4] = {box0, box1, 1, 1};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, globalAddress, globalDim,
        globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

__device__ __forceinline__ void tma_load_4d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_non_swizzled(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)0 << 61;   // NONE Swizzle
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn_custom(uint32_t M, uint32_t N, int a_major, int b_major) {
    uint32_t d = 0;
    d |= (1u << 4);    // FP32
    d |= (1u << 7);    // BF16
    d |= (1u << 10);   // BF16
    d |= ((uint32_t)a_major << 15);
    d |= ((uint32_t)b_major << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ void tcgen05_ld_4x(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ void tcgen05_ld_8x(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3, uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3),"=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7) : "r"(col));
}

__device__ __forceinline__ void tmem_store_4x(uint32_t col, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};"
        :: "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(col) : "memory");
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ float fast_exp2f(float x) {
    float y; asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y;
}

__device__ __forceinline__ uint32_t pack_bf16_fn(uint32_t fp32_a, uint32_t fp32_b) {
    __nv_bfloat16 a = __float2bfloat16(__uint_as_float(fp32_a));
    __nv_bfloat16 b = __float2bfloat16(__uint_as_float(fp32_b));
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};" : "=r"(result) : "h"(*reinterpret_cast<uint16_t*>(&a)), "h"(*reinterpret_cast<uint16_t*>(&b)));
    return result;
}

__device__ __forceinline__ void tmem_alloc_cg1_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void umma_f16_cg1_fn(uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_f16_cg1_tmem_a_fn(uint32_t tmem_c, uint32_t tmem_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}\n"
        :: "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_cg1_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(a));
}

__global__ void compute_D_kernel(const __nv_bfloat16* O, const __nv_bfloat16* dO, float* D, int S, int d) {
    int s = blockIdx.x;
    int bh = blockIdx.y;
    int tid = threadIdx.x;
    float sum = 0;
    int base = bh * S * d + s * d;
    for (int i = tid; i < d; i += blockDim.x) {
        sum += __bfloat162float(O[base + i]) * __bfloat162float(dO[base + i]);
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    __shared__ float shared_sum[4];
    if (tid % 32 == 0) shared_sum[tid / 32] = sum;
    __syncthreads();
    if (tid < 32) {
        sum = (tid < blockDim.x / 32) ? shared_sum[tid] : 0;
        for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (tid == 0) D[bh * S + s] = sum;
    }
}

__global__ void mha_bwd_kernel_sm100(
    const __grid_constant__ CUtensorMap tma_Q, 
    const __grid_constant__ CUtensorMap tma_K, 
    const __grid_constant__ CUtensorMap tma_V, 
    const __grid_constant__ CUtensorMap tma_dO,
    const float* LSE, const float* D,
    __nv_bfloat16* dK_global, __nv_bfloat16* dV_global, __nv_bfloat16* dQ_global,
    int S, int num_q_blocks) 
{
    int k_idx = blockIdx.x;
    int bh = blockIdx.y;
    int k_start = k_idx * 128;
    int q_start_block = k_start / 64; 
    
    extern __shared__ __align__(128) uint8_t smem_buf[];
    __nv_bfloat16* smem_K = (__nv_bfloat16*)smem_buf;                   // 32 KB
    __nv_bfloat16* smem_V = smem_K + 128 * 128;                         // 32 KB
    __nv_bfloat16* smem_Q = smem_V + 128 * 128;                         // 32 KB
    __nv_bfloat16* smem_dO = smem_Q + 2 * 64 * 128;                     // 32 KB
    __nv_bfloat16* smem_dS = smem_dO + 2 * 64 * 128;                    // 16 KB
    float* smem_LSE = (float*)(smem_dS + 128 * 64);                     // 256 B
    float* smem_D = smem_LSE + 64;                                      // 256 B
    uint64_t* mbar_K = (uint64_t*)(smem_D + 64);                        // 8 B
    uint64_t* mbar_V = mbar_K + 1;
    uint64_t* mbar_Q = mbar_V + 1;
    uint64_t* mbar_dO = mbar_Q + 2;
    uint64_t* mbar_umma = mbar_dO + 2;
    
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    __shared__ uint32_t tmem_base;
    if (warp_id == 0) tmem_alloc_cg1_fn(&tmem_base, 512);
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_K, 1);
        init_smem_barrier_fn(mbar_V, 1);
        init_smem_barrier_fn(&mbar_Q[0], 1);
        init_smem_barrier_fn(&mbar_Q[1], 1);
        init_smem_barrier_fn(&mbar_dO[0], 1);
        init_smem_barrier_fn(&mbar_dO[1], 1);
        init_smem_barrier_fn(mbar_umma, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    
    uint32_t tmem_dK = tmem_base;
    uint32_t tmem_dV = tmem_base + 128;
    uint32_t tmem_dQ = tmem_base + 256;
    uint32_t tmem_P  = tmem_base + 384;
    uint32_t tmem_dP = tmem_base + 448;
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_K, 128 * 128 * 2);
        tma_load_4d_fn(&tma_K, mbar_K, smem_K, 0, k_start, bh % 48, bh / 48); 
        mbarrier_arrive_and_expect_tx_fn(mbar_V, 128 * 128 * 2);
        tma_load_4d_fn(&tma_V, mbar_V, smem_V, 0, k_start, bh % 48, bh / 48);
    }
    
    if (threadIdx.x == 0 && q_start_block < num_q_blocks) {
        mbarrier_arrive_and_expect_tx_fn(&mbar_Q[0], 64 * 128 * 2);
        tma_load_4d_fn(&tma_Q, &mbar_Q[0], smem_Q, 0, q_start_block * 64, bh % 48, bh / 48);
        mbarrier_arrive_and_expect_tx_fn(&mbar_dO[0], 64 * 128 * 2);
        tma_load_4d_fn(&tma_dO, &mbar_dO[0], smem_dO, 0, q_start_block * 64, bh % 48, bh / 48);
    }
    
    mbarrier_wait_fn(mbar_K, 0);
    mbarrier_wait_fn(mbar_V, 0);
    
    uint32_t idesc_S  = make_instr_desc_fn_custom(128, 64, 0, 0); 
    uint32_t idesc_dV = make_instr_desc_fn_custom(128, 128, 0, 1); 
    uint32_t idesc_dP = make_instr_desc_fn_custom(128, 64, 0, 0); 
    uint32_t idesc_dK = make_instr_desc_fn_custom(128, 128, 0, 1);
    uint32_t idesc_dQ = make_instr_desc_fn_custom(64, 128, 1, 1);
    
    for (int q_idx = q_start_block; q_idx < num_q_blocks; q_idx++) {
        int buf_idx = (q_idx - q_start_block) % 2;
        int next_buf_idx = (buf_idx + 1) % 2;
        int q_start = q_idx * 64;
        __nv_bfloat16* smem_Q_buf = smem_Q + buf_idx * 8192;
        __nv_bfloat16* smem_dO_buf = smem_dO + buf_idx * 8192;
        
        if (threadIdx.x == 0 && q_idx + 1 < num_q_blocks) {
            mbarrier_arrive_and_expect_tx_fn(&mbar_Q[next_buf_idx], 64 * 128 * 2);
            tma_load_4d_fn(&tma_Q, &mbar_Q[next_buf_idx], smem_Q + next_buf_idx * 8192, 0, (q_idx + 1) * 64, bh % 48, bh / 48);
            mbarrier_arrive_and_expect_tx_fn(&mbar_dO[next_buf_idx], 64 * 128 * 2);
            tma_load_4d_fn(&tma_dO, &mbar_dO[next_buf_idx], smem_dO + next_buf_idx * 8192, 0, (q_idx + 1) * 64, bh % 48, bh / 48);
        }
        
        if (threadIdx.x < 64) {
            smem_LSE[threadIdx.x] = LSE[bh * S + q_start + threadIdx.x];
            smem_D[threadIdx.x] = D[bh * S + q_start + threadIdx.x];
        }
        
        mbarrier_wait_fn(&mbar_Q[buf_idx], (q_idx - q_start_block) / 2);
        mbarrier_wait_fn(&mbar_dO[buf_idx], (q_idx - q_start_block) / 2);
        
        __syncthreads(); 
        
        if (threadIdx.x == 0) {
            for (int k = 0; k < 128; k += 16) {
                uint64_t d_K = make_smem_desc_non_swizzled(smem_K + k, 2048, 128);
                uint64_t d_Q = make_smem_desc_non_swizzled(smem_Q_buf + k, 1024, 128);
                int acc = (k == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_P, d_K, d_Q, idesc_S, acc);
            }
            umma_commit_cg1_fn(mbar_umma);
        }
        mbarrier_wait_fn(mbar_umma, ((q_idx - q_start_block) * 3 + 0)); 
        
        float scale = 0.12751515f; 
        for (int c = 0; c < 64; c += 8) {
            uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
            tcgen05_ld_8x(tmem_P + c, &r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);
            tmem_load_fence_fn();
            
            int r = warp_id * 32 + lane_id; 
            int k_idx_global = k_start + r;
            
            float p0 = fast_exp2f((*(float*)&r0 * scale) - smem_LSE[c+0] * 1.44269504f);
            if (q_start + c + 0 < k_idx_global) p0 = 0.0f;
            float p1 = fast_exp2f((*(float*)&r1 * scale) - smem_LSE[c+1] * 1.44269504f);
            if (q_start + c + 1 < k_idx_global) p1 = 0.0f;
            float p2 = fast_exp2f((*(float*)&r2 * scale) - smem_LSE[c+2] * 1.44269504f);
            if (q_start + c + 2 < k_idx_global) p2 = 0.0f;
            float p3 = fast_exp2f((*(float*)&r3 * scale) - smem_LSE[c+3] * 1.44269504f);
            if (q_start + c + 3 < k_idx_global) p3 = 0.0f;
            float p4 = fast_exp2f((*(float*)&r4 * scale) - smem_LSE[c+4] * 1.44269504f);
            if (q_start + c + 4 < k_idx_global) p4 = 0.0f;
            float p5 = fast_exp2f((*(float*)&r5 * scale) - smem_LSE[c+5] * 1.44269504f);
            if (q_start + c + 5 < k_idx_global) p5 = 0.0f;
            float p6 = fast_exp2f((*(float*)&r6 * scale) - smem_LSE[c+6] * 1.44269504f);
            if (q_start + c + 6 < k_idx_global) p6 = 0.0f;
            float p7 = fast_exp2f((*(float*)&r7 * scale) - smem_LSE[c+7] * 1.44269504f);
            if (q_start + c + 7 < k_idx_global) p7 = 0.0f;
            
            tmem_store_4x(tmem_P + c/2, pack_bf16_fn(*(uint32_t*)&p0, *(uint32_t*)&p1),
                                        pack_bf16_fn(*(uint32_t*)&p2, *(uint32_t*)&p3),
                                        pack_bf16_fn(*(uint32_t*)&p4, *(uint32_t*)&p5),
                                        pack_bf16_fn(*(uint32_t*)&p6, *(uint32_t*)&p7));
        }
        tcgen05_fence_after_fn();
        __syncthreads(); 
        
        if (threadIdx.x == 0) {
            for (int k = 0; k < 64; k += 16) {
                uint64_t d_dO = make_smem_desc_non_swizzled(smem_dO_buf + k * 128, 128, 1024);
                int acc = (k == 0 && q_idx == q_start_block) ? 0 : 1;
                umma_f16_cg1_tmem_a_fn(tmem_dV, tmem_P + (k / 2), d_dO, idesc_dV, acc);
            }
            for (int k = 0; k < 128; k += 16) {
                uint64_t d_V = make_smem_desc_non_swizzled(smem_V + k, 2048, 128);
                uint64_t d_dO = make_smem_desc_non_swizzled(smem_dO_buf + k, 1024, 128);
                int acc = (k == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_dP, d_V, d_dO, idesc_dP, acc);
            }
            umma_commit_cg1_fn(mbar_umma);
        }
        mbarrier_wait_fn(mbar_umma, ((q_idx - q_start_block) * 3 + 1)); 
        
        for (int c = 0; c < 64; c += 8) {
            uint32_t p0, p1, p2, p3; tcgen05_ld_4x(tmem_P + c/2, &p0, &p1, &p2, &p3);
            uint32_t dp0, dp1, dp2, dp3, dp4, dp5, dp6, dp7; tcgen05_ld_8x(tmem_dP + c, &dp0, &dp1, &dp2, &dp3, &dp4, &dp5, &dp6, &dp7);
            tmem_load_fence_fn();
            
            float ds0 = __bfloat162float(*(__nv_bfloat16*)&p0) * (*(float*)&dp0 - smem_D[c+0]);
            float ds1 = __bfloat162float(*((__nv_bfloat16*)&p0 + 1)) * (*(float*)&dp1 - smem_D[c+1]);
            float ds2 = __bfloat162float(*(__nv_bfloat16*)&p1) * (*(float*)&dp2 - smem_D[c+2]);
            float ds3 = __bfloat162float(*((__nv_bfloat16*)&p1 + 1)) * (*(float*)&dp3 - smem_D[c+3]);
            float ds4 = __bfloat162float(*(__nv_bfloat16*)&p2) * (*(float*)&dp4 - smem_D[c+4]);
            float ds5 = __bfloat162float(*((__nv_bfloat16*)&p2 + 1)) * (*(float*)&dp5 - smem_D[c+5]);
            float ds6 = __bfloat162float(*(__nv_bfloat16*)&p3) * (*(float*)&dp6 - smem_D[c+6]);
            float ds7 = __bfloat162float(*((__nv_bfloat16*)&p3 + 1)) * (*(float*)&dp7 - smem_D[c+7]);
            
            uint32_t dsp0 = pack_bf16_fn(*(uint32_t*)&ds0, *(uint32_t*)&ds1);
            uint32_t dsp1 = pack_bf16_fn(*(uint32_t*)&ds2, *(uint32_t*)&ds3);
            uint32_t dsp2 = pack_bf16_fn(*(uint32_t*)&ds4, *(uint32_t*)&ds5);
            uint32_t dsp3 = pack_bf16_fn(*(uint32_t*)&ds6, *(uint32_t*)&ds7);
            
            tmem_store_4x(tmem_dP + c/2, dsp0, dsp1, dsp2, dsp3);
            
            int r = warp_id * 32 + lane_id;
            *(uint32_t*)&smem_dS[r * 64 + c] = dsp0;
            *(uint32_t*)&smem_dS[r * 64 + c + 2] = dsp1;
            *(uint32_t*)&smem_dS[r * 64 + c + 4] = dsp2;
            *(uint32_t*)&smem_dS[r * 64 + c + 6] = dsp3;
        }
        tcgen05_fence_after_fn();
        __syncthreads(); 
        
        if (threadIdx.x == 0) {
            for (int k = 0; k < 64; k += 16) {
                uint64_t d_Q = make_smem_desc_non_swizzled(smem_Q_buf + k * 128, 128, 1024);
                int acc = (k == 0 && q_idx == q_start_block) ? 0 : 1;
                umma_f16_cg1_tmem_a_fn(tmem_dK, tmem_dP + (k / 2), d_Q, idesc_dK, acc);
            }
            for (int k = 0; k < 128; k += 16) {
                uint64_t d_dS = make_smem_desc_non_swizzled(smem_dS + k * 64, 128, 2048);
                uint64_t d_K = make_smem_desc_non_swizzled(smem_K + k * 128, 128, 2048);
                int acc = (k == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_dQ, d_dS, d_K, idesc_dQ, acc);
            }
            umma_commit_cg1_fn(mbar_umma);
        }
        mbarrier_wait_fn(mbar_umma, ((q_idx - q_start_block) * 3 + 2)); 
        
        for (int c = 0; c < 128; c += 4) {
            uint32_t r0, r1, r2, r3; tcgen05_ld_4x(tmem_dQ + c, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            int r = warp_id * 32 + lane_id;
            if (r < 64) {
                __nv_bfloat162 b01 = __floats2bfloat162_rn(*(float*)&r0, *(float*)&r1);
                __nv_bfloat162 b23 = __floats2bfloat162_rn(*(float*)&r2, *(float*)&r3);
                uint64_t base_dQ = (uint64_t)bh * S * 128 + (q_idx * 64 + r) * 128 + c;
                atomicAdd((__nv_bfloat162*)&dQ_global[base_dQ], b01);
                atomicAdd((__nv_bfloat162*)&dQ_global[base_dQ + 2], b23);
            }
        }
    }
    
    for (int c = 0; c < 128; c += 4) {
        uint32_t r0, r1, r2, r3; tcgen05_ld_4x(tmem_dK + c, &r0, &r1, &r2, &r3);
        tmem_load_fence_fn();
        int r = warp_id * 32 + lane_id;
        __nv_bfloat162 b01 = __floats2bfloat162_rn(*(float*)&r0, *(float*)&r1);
        __nv_bfloat162 b23 = __floats2bfloat162_rn(*(float*)&r2, *(float*)&r3);
        uint64_t base_dK = (uint64_t)bh * S * 128 + (k_start + r) * 128 + c;
        *( (__nv_bfloat162*)&dK_global[base_dK] ) = b01;
        *( (__nv_bfloat162*)&dK_global[base_dK + 2] ) = b23;
        
        tcgen05_ld_4x(tmem_dV + c, &r0, &r1, &r2, &r3);
        tmem_load_fence_fn();
        b01 = __floats2bfloat162_rn(*(float*)&r0, *(float*)&r1);
        b23 = __floats2bfloat162_rn(*(float*)&r2, *(float*)&r3);
        uint64_t base_dV = (uint64_t)bh * S * 128 + (k_start + r) * 128 + c;
        *( (__nv_bfloat162*)&dV_global[base_dV] ) = b01;
        *( (__nv_bfloat162*)&dV_global[base_dV + 2] ) = b23;
    }
    
    if (warp_id == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(tmem_dK));
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L, 
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {

    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    int B = 4;
    int H = 48;
    int S = Q.size(2); 

    __nv_bfloat16* Q_ptr = static_cast<__nv_bfloat16*>(Q.data_ptr());
    __nv_bfloat16* K_ptr = static_cast<__nv_bfloat16*>(K.data_ptr());
    __nv_bfloat16* V_ptr = static_cast<__nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    __nv_bfloat16* dO_ptr = static_cast<__nv_bfloat16*>(dO.data_ptr());
    float* L_ptr = static_cast<float*>(L.data_ptr());
    __nv_bfloat16* dQ_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());

    CUDA_CHECK(cudaMemsetAsync(dQ_ptr, 0, B * H * S * 128 * sizeof(__nv_bfloat16), stream));

    float* D_ptr;
    CUDA_CHECK(cudaMallocAsync(&D_ptr, B * H * S * sizeof(float), stream));
    compute_D_kernel<<<dim3(S, B * H), 128, 0, stream>>>(O_ptr, dO_ptr, D_ptr, S, 128);

    CUtensorMap tma_K, tma_V, tma_Q, tma_dO;
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_K, K_ptr, 128, S, H, B, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_V, V_ptr, 128, S, H, B, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_Q, Q_ptr, 128, S, H, B, 128, 64, CU_TENSOR_MAP_SWIZZLE_NONE));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_dO, dO_ptr, 128, S, H, B, 128, 64, CU_TENSOR_MAP_SWIZZLE_NONE));

    int num_q_blocks = S / 64;
    dim3 grid(S / 128, B * H);
    dim3 block(128); 
    size_t smem = 150 * 1024;
    cudaFuncSetAttribute(mha_bwd_kernel_sm100, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    
    mha_bwd_kernel_sm100<<<grid, block, smem, stream>>>(
        tma_Q, tma_K, tma_V, tma_dO, L_ptr, D_ptr, dK_ptr, dV_ptr, dQ_ptr, S, num_q_blocks);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFreeAsync(D_ptr, stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}