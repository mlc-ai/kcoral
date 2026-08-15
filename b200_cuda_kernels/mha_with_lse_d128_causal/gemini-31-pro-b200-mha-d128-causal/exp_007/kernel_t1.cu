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
        const char* err_str;                                       \
        cuGetErrorString(_e, &err_str);                            \
        fprintf(stderr, "CU error %s at %s:%d\n",                  \
                err_str, __FILE__, __LINE__);                      \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_cuda {

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

__device__ __forceinline__ void fence_proxy_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
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

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ void tmem_store_4x_fn(uint32_t col, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
                 :: "r"(col), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_store_fence_fn() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_commit_cg1_arrive_one(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(a));
}

__device__ __forceinline__ void umma_f16_cg1(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ uint64_t make_smem_desc_k_major(void* ptr) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    uint64_t d = (addr & 0x3FFFF) >> 4;
    d |= (1ull << 16);           // LBO = 1
    d |= (1024ull >> 4) << 32;   // SBO = 1024
    d |= (1ull << 46);           // version = 1
    d |= (2ull << 61);           // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_mn_major(void* ptr) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    uint64_t d = (addr & 0x3FFFF) >> 4;
    d |= (8192ull >> 4) << 16;   // LBO = 8192
    d |= (1024ull >> 4) << 32;   // SBO = 1024
    d |= (1ull << 46);           // version = 1
    d |= (2ull << 61);           // SWIZZLE_128B
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

__device__ __forceinline__ uint32_t make_instr_desc_n_major(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    // c_format = FP32
    d |= (1u << 7);    // a_format = BF16
    d |= (1u << 10);   // b_format = BF16
    d |= (0u << 15);   // a_major = 0 (A is K-Major)
    d |= (1u << 16);   // b_major = 1 (B is N-Major)
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

struct SharedStorage {
    uint8_t Q0[64 * 64 * 2];
    uint8_t Q1[64 * 64 * 2];
    uint8_t K0[2][64 * 64 * 2];
    uint8_t K1[2][64 * 64 * 2];
    uint8_t V0[2][64 * 64 * 2];
    uint8_t V1[2][64 * 64 * 2];
    uint8_t P[64 * 64 * 2];
    uint64_t mbar_Q[1];
    uint64_t mbar_K[2];
    uint64_t mbar_V[2];
    uint64_t mbar_UMMA[1];
    uint32_t tmem_alloc[4];
};

__global__ void causal_mha_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* __restrict__ O_ptr,
    float* __restrict__ LSE_ptr,
    int S
) {
    extern __shared__ __align__(16) uint8_t smem_dynamic[];
    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem_dynamic);
    uint32_t offset = (1024 - (smem_addr & 1023)) & 1023; // Guarantee 1024-byte alignment
    SharedStorage* smem = reinterpret_cast<SharedStorage*>(smem_dynamic + offset);
    
    int m_start = blockIdx.x * 64;
    int b_idx = blockIdx.z;
    int h_idx = blockIdx.y;
    int H = gridDim.y;
    int c1 = b_idx * H * S + h_idx * S + m_start;

    uint32_t tmem_S_addr, tmem_O0_addr, tmem_O1_addr;
    if (threadIdx.x < 32) tmem_alloc_cg1_fn(&smem->tmem_alloc[0], 64);
    __syncthreads();
    tmem_S_addr = smem->tmem_alloc[0];
    __syncthreads();
    if (threadIdx.x < 32) tmem_alloc_cg1_fn(&smem->tmem_alloc[0], 64);
    __syncthreads();
    tmem_O0_addr = smem->tmem_alloc[0];
    __syncthreads();
    if (threadIdx.x < 32) tmem_alloc_cg1_fn(&smem->tmem_alloc[0], 64);
    __syncthreads();
    tmem_O1_addr = smem->tmem_alloc[0];
    
    int tid = threadIdx.x;
    for (int c = 0; c < 64; c += 4) {
        tmem_store_4x_fn(tmem_O0_addr + c, 0, 0, 0, 0);
        tmem_store_4x_fn(tmem_O1_addr + c, 0, 0, 0, 0);
    }
    tmem_store_fence_fn();
    
    if (tid == 0) {
        init_smem_barrier_fn(smem->mbar_Q, 1);
        init_smem_barrier_fn(&smem->mbar_K[0], 1);
        init_smem_barrier_fn(&smem->mbar_K[1], 1);
        init_smem_barrier_fn(&smem->mbar_V[0], 1);
        init_smem_barrier_fn(&smem->mbar_V[1], 1);
        init_smem_barrier_fn(smem->mbar_UMMA, 1);
    }
    __syncthreads();
    if (tid == 0) fence_smem_barrier_init_fn();
    __syncthreads();
    
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(smem->mbar_Q, 16384);
        tma_load_2d_fn(&tma_Q, smem->mbar_Q, smem->Q0, 0, c1);
        tma_load_2d_fn(&tma_Q, smem->mbar_Q, smem->Q1, 64, c1);
    }
    mbarrier_wait_fn(smem->mbar_Q, 0);
    
    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    int phase_k[2] = {0, 0};
    int phase_v[2] = {0, 0};
    int umma_phase = 0;
    
    uint64_t dQ0 = make_smem_desc_k_major(smem->Q0);
    uint64_t dQ1 = make_smem_desc_k_major(smem->Q1);
    uint32_t idesc_S = make_instr_desc_fn(64, 64);
    uint32_t idesc_O = make_instr_desc_n_major(64, 64);
    
    int kv_blocks = (m_start + 64) / 64;
    for (int kv_idx = 0; kv_idx < kv_blocks; ++kv_idx) {
        int buf = kv_idx % 2;
        int n_start = kv_idx * 64;
        int c1_k = b_idx * H * S + h_idx * S + n_start;
        
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(&smem->mbar_K[buf], 16384);
            tma_load_2d_fn(&tma_K, &smem->mbar_K[buf], smem->K0[buf], 0, c1_k);
            tma_load_2d_fn(&tma_K, &smem->mbar_K[buf], smem->K1[buf], 64, c1_k);
            mbarrier_arrive_and_expect_tx_fn(&smem->mbar_V[buf], 16384);
            tma_load_2d_fn(&tma_V, &smem->mbar_V[buf], smem->V0[buf], 0, c1_k);
            tma_load_2d_fn(&tma_V, &smem->mbar_V[buf], smem->V1[buf], 64, c1_k);
        }
        
        mbarrier_wait_fn(&smem->mbar_K[buf], phase_k[buf]);
        
        uint64_t dK0 = make_smem_desc_k_major(smem->K0[buf]);
        uint64_t dK1 = make_smem_desc_k_major(smem->K1[buf]);
        
        tcgen05_fence_after_fn();
        for (int i = 0; i < 4; ++i) {
            uint32_t accum = (i == 0) ? 0 : 1;
            umma_f16_cg1(tmem_S_addr, dQ0 + i*2, dK0 + i*2, idesc_S, accum);
        }
        for (int i = 0; i < 4; ++i) {
            umma_f16_cg1(tmem_S_addr, dQ1 + i*2, dK1 + i*2, idesc_S, 1);
        }
        if (tid == 0) tcgen05_commit_cg1_arrive_one(smem->mbar_UMMA);
        mbarrier_wait_fn(smem->mbar_UMMA, umma_phase);
        umma_phase ^= 1;
        
        float row_S[64];
        float row_m = -INFINITY;
        float scale = 1.0f / sqrtf(128.0f) * 1.44269504f;
        
        if (tid < 64) {
            for (int c = 0; c < 64; c += 4) {
                uint32_t r[4];
                tmem_load_4x_fn(tmem_S_addr + c, &r[0], &r[1], &r[2], &r[3]);
                tmem_load_fence_fn();
                for (int i = 0; i < 4; ++i) {
                    float val = __uint_as_float(r[i]) * scale;
                    int global_q = m_start + tid;
                    int global_k = n_start + c + i;
                    if (global_k > global_q || global_k >= S) val = -INFINITY;
                    row_S[c + i] = val;
                    row_m = fmaxf(row_m, val);
                }
            }
        }
        
        float m_new = fmaxf(m_prev, row_m);
        float rescale_O = 1.0f;
        if (m_new > m_prev && m_prev != -INFINITY && tid < 64) {
            rescale_O = exp2f(m_prev - m_new);
            for (int c = 0; c < 64; c += 4) {
                uint32_t r[4];
                tmem_load_4x_fn(tmem_O0_addr + c, &r[0], &r[1], &r[2], &r[3]);
                tmem_load_fence_fn();
                for(int i=0; i<4; ++i) {
                    float val = __uint_as_float(r[i]) * rescale_O;
                    r[i] = __float_as_uint(val);
                }
                tmem_store_4x_fn(tmem_O0_addr + c, r[0], r[1], r[2], r[3]);
            }
            for (int c = 0; c < 64; c += 4) {
                uint32_t r[4];
                tmem_load_4x_fn(tmem_O1_addr + c, &r[0], &r[1], &r[2], &r[3]);
                tmem_load_fence_fn();
                for(int i=0; i<4; ++i) {
                    float val = __uint_as_float(r[i]) * rescale_O;
                    r[i] = __float_as_uint(val);
                }
                tmem_store_4x_fn(tmem_O1_addr + c, r[0], r[1], r[2], r[3]);
            }
            tmem_store_fence_fn();
        }
        
        float l_new = l_prev * rescale_O;
        float row_sum = 0.0f;
        if (tid < 64) {
            __nv_bfloat16* P_ptr = (__nv_bfloat16*)smem->P;
            for (int c = 0; c < 64; ++c) {
                float p = exp2f(row_S[c] - m_new);
                if (row_S[c] == -INFINITY) p = 0.0f;
                row_sum += p;
                
                uint32_t chunk_idx = c / 8;
                uint32_t swizzled_chunk = chunk_idx ^ (tid % 8);
                uint32_t swizzled_col = swizzled_chunk * 8 + (c % 8);
                P_ptr[tid * 64 + swizzled_col] = __float2bfloat16(p);
            }
        }
        l_new += row_sum;
        m_prev = m_new;
        l_prev = l_new;
        
        __syncthreads();
        fence_proxy_async_shared_fn();
        
        mbarrier_wait_fn(&smem->mbar_V[buf], phase_v[buf]);
        
        uint64_t dP = make_smem_desc_k_major(smem->P);
        uint64_t dV0 = make_smem_desc_mn_major(smem->V0[buf]);
        uint64_t dV1 = make_smem_desc_mn_major(smem->V1[buf]);
        
        tcgen05_fence_after_fn();
        for (int i = 0; i < 4; ++i) {
            umma_f16_cg1(tmem_O0_addr, dP + i*2, dV0 + i*128, idesc_O, 1);
            umma_f16_cg1(tmem_O1_addr, dP + i*2, dV1 + i*128, idesc_O, 1);
        }
        if (tid == 0) tcgen05_commit_cg1_arrive_one(smem->mbar_UMMA);
        mbarrier_wait_fn(smem->mbar_UMMA, umma_phase);
        umma_phase ^= 1;
        
        __syncthreads();
        phase_k[buf] ^= 1;
        phase_v[buf] ^= 1;
    }
    
    float final_scale = 1.0f / l_prev;
    __syncthreads();
    
    __nv_bfloat16* smem_out = (__nv_bfloat16*)smem->Q0; 
    if (tid < 64) {
        for (int c = 0; c < 64; c += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_O0_addr + c, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            smem_out[tid * 128 + c + 0] = __float2bfloat16(__uint_as_float(r0) * final_scale);
            smem_out[tid * 128 + c + 1] = __float2bfloat16(__uint_as_float(r1) * final_scale);
            smem_out[tid * 128 + c + 2] = __float2bfloat16(__uint_as_float(r2) * final_scale);
            smem_out[tid * 128 + c + 3] = __float2bfloat16(__uint_as_float(r3) * final_scale);
        }
        for (int c = 0; c < 64; c += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_O1_addr + c, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            smem_out[tid * 128 + 64 + c + 0] = __float2bfloat16(__uint_as_float(r0) * final_scale);
            smem_out[tid * 128 + 64 + c + 1] = __float2bfloat16(__uint_as_float(r1) * final_scale);
            smem_out[tid * 128 + 64 + c + 2] = __float2bfloat16(__uint_as_float(r2) * final_scale);
            smem_out[tid * 128 + 64 + c + 3] = __float2bfloat16(__uint_as_float(r3) * final_scale);
        }
        
        int global_row = m_start + tid;
        if (global_row < S) {
            float lse = m_prev * 0.6931471805599453f + logf(l_prev);
            LSE_ptr[b_idx * H * S + h_idx * S + global_row] = lse;
        }
    }
    __syncthreads();
    
    uint32_t warp_id = tid / 32;
    uint32_t lane_id = tid % 32;
    for (uint32_t step = 0; step < 16; ++step) {
        uint32_t row = step * 4 + warp_id;
        int global_row = m_start + row;
        if (row < 64 && global_row < S) {
            uint32_t col_start = lane_id * 4;
            uint2 data = *reinterpret_cast<uint2*>(&smem_out[row * 128 + col_start]);
            *reinterpret_cast<uint2*>(O_ptr + b_idx * H * S * 128 + h_idx * S * 128 + global_row * 128 + col_start) = data;
        }
    }
    
    if (threadIdx.x < 32) tmem_dealloc_cg1_fn(tmem_S_addr, 64);
    if (threadIdx.x < 32) tmem_dealloc_cg1_fn(tmem_O0_addr, 64);
    if (threadIdx.x < 32) tmem_dealloc_cg1_fn(tmem_O1_addr, 64);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    
    __nv_bfloat16* q_ptr = static_cast<__nv_bfloat16*>(Q.data_ptr());
    __nv_bfloat16* k_ptr = static_cast<__nv_bfloat16*>(K.data_ptr());
    __nv_bfloat16* v_ptr = static_cast<__nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_ptr = static_cast<float*>(LSE.data_ptr());
    
    cuuint64_t globalDim[2] = {128, static_cast<cuuint64_t>(B * H * S)};
    cuuint64_t globalStrides[1] = {128 * 2};
    cuuint32_t boxDim[2] = {64, 64};
    cuuint32_t elementStrides[2] = {1, 1};

    CUtensorMap tma_Q, tma_K, tma_V;
    CU_CHECK(cuTensorMapEncodeTiled(
        &tma_Q, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, q_ptr,
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        
    CU_CHECK(cuTensorMapEncodeTiled(
        &tma_K, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, k_ptr,
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        
    CU_CHECK(cuTensorMapEncodeTiled(
        &tma_V, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, v_ptr,
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        
    int blocks_x = (S + 63) / 64;
    dim3 grid(blocks_x, H, B);
    dim3 block(128);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int smem_size = sizeof(SharedStorage) + 1024; // Ensure alignment headroom
    CUDA_CHECK(cudaFuncSetAttribute(causal_mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    causal_mha_kernel<<<grid, block, smem_size, stream>>>(tma_Q, tma_K, tma_V, o_ptr, lse_ptr, S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_cuda::run);

}