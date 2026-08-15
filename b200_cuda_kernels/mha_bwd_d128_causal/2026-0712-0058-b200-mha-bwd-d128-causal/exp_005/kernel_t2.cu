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
#include <algorithm>
#include <vector>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_fa4 {

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

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
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

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ void umma_f16_cg2_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
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

__device__ __forceinline__ void tmem_wait_fn(uint64_t* bar, uint32_t phase) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared.b64 [%0];" :: "r"(a));
    mbarrier_wait_fn(bar, phase);
}

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ void load_to_smem_swizzled(char* smem, const __nv_bfloat16* gmem, int q_start, int S) {
    int tid = threadIdx.x;
    for (int col_bytes = 0; col_bytes < 256; col_bytes += 4) {
        int col = col_bytes / 2;
        int chunk = col_bytes / 16;
        int chunk_swizzled = chunk ^ (tid % 8);
        int final_col_bytes = (chunk_swizzled * 16) + (col_bytes % 16);
        int idx = tid * 256 + final_col_bytes;
        int g_idx = (q_start + tid) * 128 + col;
        if (q_start + tid < S && col < 128) {
            *(uint32_t*)(&smem[idx]) = *(uint32_t*)(&gmem[g_idx]);
        } else {
            *(uint32_t*)(&smem[idx]) = 0;
        }
    }
}

__device__ void gemm_tmem(uint32_t tmem_c_ptr, char* smem_A, char* smem_B, uint32_t idesc, uint32_t accum) {
    for (int i = 0; i < 128; ++i) {
        uint32_t col = (i * 128) + 0; 
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn(col, &r0, &r1, &r2, &r3);
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        uint64_t desc_a = make_smem_desc_sm100_fn((void*)(smem_A), 1, 1024);
        uint64_t desc_b = make_smem_desc_sm100_fn((void*)(smem_B), 1, 1024);
        
        uint32_t accum_i = (i == 0) ? accum : 1;
        umma_f16_cg2_fn(tmem_c_ptr, desc_a, desc_b, idesc, accum_i);
    }
}

__device__ void compute_D_local(char* s_O, char* s_dO, float* s_DT) {
    int tid = threadIdx.x;
    float sum = 0;
    for (int k = 0; k < 128; k++) {
        int chunk = (k * 2) / 16;
        int chunk_swizzled = chunk ^ (tid % 8);
        int final_col_bytes = (chunk_swizzled * 16) + ((k * 2) % 16);
        int idx = tid * 256 + final_col_bytes;
        float o = __bfloat162float(*( (__nv_bfloat16*) (&s_O[idx]) ));
        float do_ = __bfloat162float(*( (__nv_bfloat16*) (&s_dO[idx]) ));
        sum += o * do_;
    }
    s_DT[tid] = sum;
}

__device__ void apply_softmax_local(char* s_PT, char* s_S_T, const float* L_data, int bh, int q_start, int kv_start, float scale, int S_val) {
    int tid = threadIdx.x;
    for (int col = 0; col < 128; col++) {
        int chunk = (col * 2) / 16;
        int chunk_swizzled = chunk ^ (tid % 8);
        int final_col_bytes = (chunk_swizzled * 16) + ((col * 2) % 16);
        int idx = tid * 256 + final_col_bytes;
        
        float s = __bfloat162float(*( (__nv_bfloat16*) (&s_S_T[idx]) ));
        float p = 0;
        if (kv_start + col <= q_start + tid && q_start + tid < S_val && kv_start + col < S_val) {
            const float* L_bh = L_data + bh * S_val + q_start + tid;
            float lse = L_bh[col];
            p = fast_exp2f_fn(s * scale - lse * scale);
        }
        *( (__nv_bfloat16*) (&s_PT[idx]) ) = __float2bfloat16(p);
    }
}

__device__ void compute_dS_local(char* s_dST, char* s_PT, char* s_dPT, float* s_DT) {
    int tid = threadIdx.x;
    for (int i = 0; i < 128; i++) {
        int chunk = (i * 2) / 16;
        int chunk_swizzled = chunk ^ (tid % 8);
        int final_col_bytes = (chunk_swizzled * 16) + ((i * 2) % 16);
        int idx = tid * 256 + final_col_bytes;
        
        float p = __bfloat162float(*( (__nv_bfloat16*) (&s_PT[idx]) ));
        float dp = __bfloat162float(*( (__nv_bfloat16*) (&s_dPT[idx]) ));
        float ds = p * (dp - s_DT[tid]);
        *( (__nv_bfloat16*) (&s_dST[idx]) ) = __float2bfloat16(ds);
    }
}

__device__ void store_gemm_atomic_add(uint32_t tmem_ptr, __nv_bfloat16* global_D, int64_t row_base, int S_val) {
    int tid = threadIdx.x;
    for (int col = 0; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn(tmem_ptr + (tid * 128) + col, &r0, &r1, &r2, &r3);
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float f0 = __uint_as_float(r0);
        float f1 = __uint_as_float(r1);
        float f2 = __uint_as_float(r2);
        float f3 = __uint_as_float(r3);
        
        int g_row = row_base + tid;
        if (g_row < S_val) {
            atomicAdd(&global_D[g_row * 128 + col + 0], __float2bfloat16(f0));
            atomicAdd(&global_D[g_row * 128 + col + 1], __float2bfloat16(f1));
            atomicAdd(&global_D[g_row * 128 + col + 2], __float2bfloat16(f2));
            atomicAdd(&global_D[g_row * 128 + col + 3], __float2bfloat16(f3));
        }
    }
}

__device__ __forceinline__ void tmem_epilogue_coalesced_4w_fn(
    __nv_bfloat16* D, uint32_t tmem_base_ptr,
    uint32_t M, uint32_t N, uint32_t m_block, uint32_t n_block,
    uint32_t BM, uint32_t BN) {
    
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t num_steps = (BM + 3) / 4;
    
    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 4 + warp_id;
        if (row >= BM) continue;
        
        uint32_t global_row = m_block + row;
        if (global_row >= M) continue;
        
        uint32_t col_start = lane_id * 4;
        if (n_block + col_start >= N) continue;
        
        uint32_t tmem_col = (row * 128) + col_start;
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn(tmem_base_ptr + tmem_col, &r0, &r1, &r2, &r3);
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float f0 = __uint_as_float(r0);
        float f1 = __uint_as_float(r1);
        float f2 = __uint_as_float(r2);
        float f3 = __uint_as_float(r3);
        
        __nv_bfloat16 bf0 = __float2bfloat16(f0);
        __nv_bfloat16 bf1 = __float2bfloat16(f1);
        __nv_bfloat16 bf2 = __float2bfloat16(f2);
        __nv_bfloat16 bf3 = __float2bfloat16(f3);
        
        uint32_t val0 = *(uint32_t*)&bf0;
        uint32_t val1 = *(uint32_t*)&bf1;
        uint32_t val2 = *(uint32_t*)&bf2;
        uint32_t val3 = *(uint32_t*)&bf3;
        
        uint4 data;
        data.x = val0;
        data.y = val1;
        data.z = val2;
        data.w = val3;
        
        uint32_t g_addr = (uint32_t)(&D[(uint64_t)global_row * N + n_block + col_start]);
        *(uint4*)g_addr = data;
    }
}

struct SharedStorage {
    char s_Q[128*128*2];
    char s_K[128*128*2];
    char s_V[128*128*2];
    char s_dO[128*128*2];
    char s_O[128*128*2];
    char s_PT[128*128*2];
    char s_dPT[128*128*2];
    char s_dST[128*128*2];
    float s_DT[128];
    uint64_t bar[1];
};

__global__ void bwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* __restrict__ L_data,
    __nv_bfloat16* __restrict__ dQ_bh,
    __nv_bfloat16* __restrict__ dK_bh,
    __nv_bfloat16* __restrict__ dV_bh,
    int64_t S_val, float scale)
{
    int bh = blockIdx.y;
    int kv_start = blockIdx.x * 128;
    uint32_t cluster_rank = cluster_rank_fn();
    
    extern __shared__ char smem_buf[];
    SharedStorage* smem = (SharedStorage*)smem_buf;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(smem->bar, 1);
    }
    __syncthreads();

    uint32_t tmem_base;
    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_base, 512);
    }
    __syncthreads();
    
    uint32_t tmem_S_ptr = tmem_base;
    uint32_t tmem_P_ptr = tmem_base + 16384; 
    uint32_t tmem_dP_ptr = tmem_base + 32768;
    uint32_t tmem_dS_ptr = tmem_base + 49152;
    uint32_t tmem_dV_ptr = tmem_base + 16384; 
    uint32_t tmem_dK_ptr = tmem_base + 32768;
    uint32_t tmem_dQ_ptr = tmem_base + 32768;
    
    uint32_t phase = 0;

    if (kv_start < S_val) {
        mbarrier_arrive_and_expect_tx_fn(smem->bar, 32768 * 2);
        tma_load_2d_fn(&tma_K, smem->bar, smem->s_K, 0, bh * S_val + kv_start);
        tma_load_2d_fn(&tma_V, smem->bar, smem->s_V, 0, bh * S_val + kv_start);
        mbarrier_wait_fn(smem->bar, phase);
        fence_proxy_async_fn();
        phase ^= 1;
    }
    __syncthreads();

    uint32_t idesc_S = make_instr_desc_fn(128, 128);

    for (int q_start = kv_start; q_start < S_val; q_start += 128) {
        int q_start_local = q_start + cluster_rank * 128;
        
        if (q_start_local < S_val) {
            mbarrier_arrive_and_expect_tx_fn(smem->bar, 32768 * 3);
            tma_load_2d_fn(&tma_Q, smem->bar, smem->s_Q, 0, bh * S_val + q_start_local);
            tma_load_2d_fn(&tma_dO, smem->bar, smem->s_dO, 0, bh * S_val + q_start_local);
            tma_load_2d_fn(&tma_O, smem->bar, smem->s_O, 0, bh * S_val + q_start_local);
            mbarrier_wait_fn(smem->bar, phase);
            fence_proxy_async_fn();
            phase ^= 1;
        }
        __syncthreads();

        compute_D_local(smem->s_O, smem->s_dO, smem->s_DT);
        __syncthreads();

        gemm_tmem(tmem_S_ptr, smem->s_Q, smem->s_K, idesc_S, 0);
        tmem_wait_fn(smem->bar, phase);
        phase ^= 1;
        __syncthreads();

        apply_softmax_local(smem->s_PT, smem->s_Q, L_data, bh, q_start_local, kv_start, scale, S_val);
        __syncthreads();

        gemm_tmem(tmem_dP_ptr, smem->s_dO, smem->s_V, idesc_S, 0);
        tmem_wait_fn(smem->bar, phase);
        phase ^= 1;
        __syncthreads();

        compute_dS_local(smem->s_dST, smem->s_PT, smem->s_dPT, smem->s_DT);
        __syncthreads();

        gemm_tmem(tmem_dV_ptr, smem->s_PT, smem->s_dO, idesc_S, 1);
        tmem_wait_fn(smem->bar, phase);
        phase ^= 1;
        __syncthreads();

        gemm_tmem(tmem_dK_ptr, smem->s_dST, smem->s_Q, idesc_S, 1);
        tmem_wait_fn(smem->bar, phase);
        phase ^= 1;
        __syncthreads();

        gemm_tmem(tmem_dQ_ptr, smem->s_dST, smem->s_K, idesc_S, 0);
        tmem_wait_fn(smem->bar, phase);
        phase ^= 1;
        
        store_gemm_atomic_add(tmem_dQ_ptr, dQ_bh + bh * S_val * 128, q_start_local, S_val);
        
        __syncthreads();
    }
    
    tmem_epilogue_coalesced_4w_fn(dV_bh + bh * S_val * 128, tmem_dV_ptr, S_val, 128, kv_start, 0, 128, 128);
    tmem_epilogue_coalesced_4w_fn(dK_bh + bh * S_val * 128, tmem_dK_ptr, S_val, 128, kv_start, 0, 128, 128);

    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_base, 512);
    }
}

CUresult create_tma_2d_descriptor_2B(
    CUtensorMap* d, void* globalAddress, 
    uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, 
    uint32_t smem_inner_dim, uint32_t smem_outer_dim, 
    CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) 
{
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

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L, 
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
         
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S_val = Q.size(2);
    int64_t d = Q.size(3);
    
    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_data = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_data = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_data = static_cast<const float*>(L.data_ptr());
    
    __nv_bfloat16* dQ_data = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_data = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_data = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    float scale = 1.44269504f;
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUDA_CHECK(cudaMemsetAsync(dQ_data, 0, B * H * S_val * d * sizeof(__nv_bfloat16), stream));
    CUDA_CHECK(cudaMemsetAsync(dK_data, 0, B * H * S_val * d * sizeof(__nv_bfloat16), stream));
    CUDA_CHECK(cudaMemsetAsync(dV_data, 0, B * H * S_val * d * sizeof(__nv_bfloat16), stream));
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O, tma_dO;
    create_tma_2d_descriptor_2B(&tma_Q, (void*)Q_data, d, B * H * S_val, 128, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_K, (void*)K_data, d, B * H * S_val, 128, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_V, (void*)V_data, d, B * H * S_val, 128, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_O, (void*)O_data, d, B * H * S_val, 128, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_dO, (void*)dO_data, d, B * H * S_val, 128, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    int64_t threads = 128;
    dim3 grid((S_val + 127) / 128, B * H);
    
    CUDA_CHECK(cudaFuncSetAttribute(bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedStorage)));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = dim3(threads);
    config.dynamicSmemBytes = sizeof(SharedStorage);
    config.stream = stream;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    CUDA_CHECK(cudaLaunchKernelEx(&config, bwd_kernel, 
        tma_Q, tma_K, tma_V, tma_O, tma_dO, L_data, dQ_data, dK_data, dV_data, S_val, scale));
        
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_fa4::run);

} // namespace tvm_ffi_fa4