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

#define DRV_CHECK(call) do {                                       \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        const char* err_str;                                       \
        cuGetErrorString(_e, &err_str);                            \
        fprintf(stderr, "CUDA Driver error %s at %s:%d\n",         \
                err_str, __FILE__, __LINE__);                      \
        exit(1);                                                   \
    }                                                              \
} while(0)

__device__ __forceinline__ bool elect_one_sync_fn() {
    uint32_t pred;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "elect.sync _|p, 0xFFFFFFFF;\n"
        "selp.b32 %0, 1, 0, p;\n"
        "}\n"
        : "=r"(pred));
    return pred != 0;
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

__device__ __forceinline__ void fence_proxy_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
}

__device__ __forceinline__ void wgmma_16x16x16(uint32_t* B, const uint32_t* A, const uint32_t* P, 
                                                float* acc, uint64_t desc_A, uint64_t desc_P) {
    asm volatile("wgmma.mma.sync.16x16x16.f16.f32 [%0], [%1], [%2], [%3], %4, %5;\n"
                 : : "r"(*(uint32_t*)&B), "r"(*(uint32_t*)&A), "r"(*(uint32_t*)&P),
                   "r"(*(uint32_t*)&acc), "l"(desc_A), "l"(desc_P));
}

__device__ __forceinline__ uint64_t make_m16_desc(void* ptr) {
    uint32_t base = (uint32_t)__cvta_generic_to_shared(ptr);
    uint64_t d = (((uint64_t)(base & 0x3FFFF)) >> 4) | 
                 (1ULL << 46) | 
                 (((uint64_t)(16 * 2)) >> 4) << 16 | 
                 (2ULL << 61);
    return d;
}

__device__ __forceinline__ int swizzle_128B(int row, int col) {
    int chunk_x = col / 8;
    int chunk_x_swizzled = chunk_x ^ (row % 8);
    return chunk_x_swizzled * 8 + (col % 8);
}

__device__ __forceinline__ uint32_t ptr_to_swizzled_offset(void* ptr, int row, int col) {
    return (row * 64 + swizzle_128B(row, col)) * sizeof(__nv_bfloat16);
}

__device__ __forceinline__ uint32_t* bf16_ptr(void* ptr) {
    return (uint32_t*)((char*)ptr);
}

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t dim0, uint64_t dim1, uint64_t dim2, uint32_t box0, uint32_t box1, uint32_t box2, CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[3] = {dim0, dim1, dim2};
    cuuint64_t globalStrides[2] = {dim0 * 2, dim0 * dim1 * 2};
    cuuint32_t boxDim[3] = {box0, box1, box2};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, globalAddress, globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

namespace tvm_ffi_example_cuda {

__global__ __launch_bounds__(128) void mha_bwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_dO,
    float* dQ_workspace,
    float* dK_workspace,
    float* dV_workspace,
    const float* L,
    int32_t S) 
{
    int32_t k_blk = blockIdx.x;
    int32_t j_blk = blockIdx.y;
    int32_t bh = blockIdx.z;
    int32_t num_blocks = (S + 63) / 64;
    float scale = 1.0f / sqrtf(128.0f);
    int tid = threadIdx.x;

    extern __shared__ __align__(1024) char smem_buf[];
    char* s_Q_h0  = smem_buf + 0 * 8192;
    char* s_Q_h1  = smem_buf + 1 * 8192;
    char* s_K_h0  = smem_buf + 2 * 8192;
    char* s_K_h1  = smem_buf + 3 * 8192;
    char* s_V_h0  = smem_buf + 4 * 8192;
    char* s_V_h1  = smem_buf + 5 * 8192;
    char* s_dO_h0 = smem_buf + 6 * 8192;
    char* s_dO_h1 = smem_buf + 7 * 8192;
    char* s_S_acc = smem_buf + 8 * 8192;
    char* s_O_acc = smem_buf + 9 * 8192;
    char* s_dP_acc= smem_buf + 10 * 8192;
    char* s_P     = smem_buf + 11 * 8192;
    char* s_D_s   = smem_buf + 12 * 8192;
    
    uint64_t* bar_outer = (uint64_t*)(smem_buf + 13 * 8192);
    uint64_t* bar_inner = (uint64_t*)(smem_buf + 13 * 8192 + sizeof(uint64_t));

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar_outer, 1);
        init_smem_barrier_fn(bar_inner, 1);
    }
    __syncthreads();

    if (elect_one_sync_fn()) {
        mbarrier_arrive_and_expect_tx_fn(bar_outer, 4 * 8192);
        tma_load_3d_fn(&tma_Q, bar_outer, s_Q_h0, 0, k_blk * 64, bh);
        tma_load_3d_fn(&tma_Q, bar_outer, s_Q_h1, 64, k_blk * 64, bh);
        tma_load_3d_fn(&tma_dO, bar_outer, s_dO_h0, 0, k_blk * 64, bh);
        tma_load_3d_fn(&tma_dO, bar_outer, s_dO_h1, 64, k_blk * 64, bh);
    }

    uint32_t phase_outer = 0;
    mbarrier_wait_fn(bar_outer, phase_outer);
    __syncthreads();
    
    // Ensure all threads are synchronized and ready to receive the barrier flip for the inner loop
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_inner, 0);
    }
    mbarrier_wait_fn(bar_inner, 0);

    if (j_blk < num_blocks) {
        if (elect_one_sync_fn()) {
            mbarrier_arrive_and_expect_tx_fn(bar_inner, 4 * 8192);
            tma_load_3d_fn(&tma_K, bar_inner, s_K_h0, 0, j_blk * 64, bh);
            tma_load_3d_fn(&tma_K, bar_inner, s_K_h1, 64, j_blk * 64, bh);
            tma_load_3d_fn(&tma_V, bar_inner, s_V_h0, 0, j_blk * 64, bh);
            tma_load_3d_fn(&tma_V, bar_inner, s_V_h1, 64, j_blk * 64, bh);
        }
        mbarrier_wait_fn(bar_inner, phase_outer); // Phase parity matches bar_outer initially
    }
    __syncthreads();
    fence_proxy_async_shared_fn();

    float dQ_h0_acc[4] = {0}, dQ_h1_acc[4] = {0};
    float dK_h0_acc[4] = {0}, dK_h1_acc[4] = {0};
    float dV_h0_acc[4] = {0}, dV_h1_acc[4] = {0};

    __nv_bfloat16* s_P_bf16 = (__nv_bfloat16*)s_P;
    __nv_bfloat16* s_D_s_bf16 = (__nv_bfloat16*)s_D_s;

    // 1. S = Q @ K^T
    for (int i = tid; i < 4096; i += blockDim.x) *(float*)((char*)s_S_acc + i * 4) = 0;
    for (int k = 0; k < 4; ++k) {
        for (int l = 0; l < 4; ++l) {
            wgmma_16x16x16(bf16_ptr(s_S_acc) + ptr_to_swizzled_offset(s_S_acc, k * 16, l * 16),
                           bf16_ptr(s_Q_h0) + ptr_to_swizzled_offset(s_Q_h0, k * 16, 0),
                           bf16_ptr(s_K_h0) + ptr_to_swizzled_offset(s_K_h0, l * 16, 0),
                           s_S_acc + k * 16 * 64 + l * 16,
                           make_m16_desc(bf16_ptr(s_S_acc)), make_m16_desc(bf16_ptr(s_S_acc)));
        }
        for (int l = 0; l < 4; ++l) {
            wgmma_16x16x16(bf16_ptr(s_S_acc) + ptr_to_swizzled_offset(s_S_acc, k * 16, l * 16),
                           bf16_ptr(s_Q_h1) + ptr_to_swizzled_offset(s_Q_h1, k * 16, 0),
                           bf16_ptr(s_K_h1) + ptr_to_swizzled_offset(s_K_h1, l * 16, 0),
                           s_S_acc + k * 16 * 64 + l * 16,
                           make_m16_desc(bf16_ptr(s_S_acc)), make_m16_desc(bf16_ptr(s_S_acc)));
        }
    }
    __syncthreads();

    // 2. Apply Softmax and extract P
    int row = tid / 8;
    int col = tid % 8;
    float S_acc[64];
    for (int i = 0; i < 8; ++i) {
        int c = col * 8 + i;
        float s = ((float*)s_S_acc)[row * 16 + c] * scale;
        float l_val = L[bh * S + (j_blk * 64) + c];
        float p = __isnanf(l_val) ? 0.0f : expf(s - l_val);
        
        s_P_bf16[row * 64 + swizzle_128B(row, c)] = __float2bfloat16(p);
    }
    __syncthreads();
    fence_proxy_async_shared_fn();

    // 3. dP = dO @ V^T
    for (int i = tid; i < 4096; i += blockDim.x) *(float*)((char*)s_dP_acc + i * 4) = 0;
    
    for (int k = 0; k < 4; ++k) {
        for (int l = 0; l < 4; ++l) {
            wgmma_16x16x16(bf16_ptr(s_dP_acc) + ptr_to_swizzled_offset(s_dP_acc, k * 16, l * 16),
                           bf16_ptr(s_dO_h0) + ptr_to_swizzled_offset(s_dO_h0, k * 16, 0),
                           bf16_ptr(s_V_h0) + ptr_to_swizzled_offset(s_V_h0, l * 16, 0),
                           s_dP_acc + k * 16 * 64 + l * 16,
                           make_m16_desc(bf16_ptr(s_dP_acc)), make_m16_desc(bf16_ptr(s_dP_acc)));
        }
        for (int l = 0; l < 4; ++l) {
            wgmma_16x16x16(bf16_ptr(s_dP_acc) + ptr_to_swizzled_offset(s_dP_acc, k * 16, l * 16),
                           bf16_ptr(s_dO_h1) + ptr_to_swizzled_offset(s_dO_h1, k * 16, 0),
                           bf16_ptr(s_V_h1) + ptr_to_swizzled_offset(s_V_h1, l * 16, 0),
                           s_dP_acc + k * 16 * 64 + l * 16,
                           make_m16_desc(bf16_ptr(s_dP_acc)), make_m16_desc(bf16_ptr(s_dP_acc)));
        }
    }
    __syncthreads();
    
    // 4. D_s = dP * P * scale
    for (int i = 0; i < 8; ++i) {
        int c = col * 8 + i;
        float dp = ((float*)s_dP_acc)[row * 16 + c];
        float p = __bfloat162float(s_P_bf16[row * 64 + swizzle_128B(row, c)]);
        float ds = dp * p * scale;
        
        s_D_s_bf16[row * 64 + swizzle_128B(row, c)] = __float2bfloat16(ds);
    }
    __syncthreads();
    fence_proxy_async_shared_fn();

    // 6. dQ += D_s @ K
    for (int i = tid; i < 4096; i += blockDim.x) *(float*)((char*)s_O_acc + i * 4) = 0; // Reuse O_acc for dQ_h0
    
    for (int k = 0; k < 4; ++k) {
        for (int l = 0; l < 4; ++l) {
            wgmma_16x16x16(bf16_ptr(s_O_acc) + ptr_to_swizzled_offset(s_O_acc, k * 16, l * 16),
                           bf16_ptr(s_D_s) + ptr_to_swizzled_offset(s_D_s, k * 16, 0),
                           bf16_ptr(s_K_h0) + ptr_to_swizzled_offset(s_K_h0, 0, l * 16),
                           s_O_acc + k * 16 * 64 + l * 16,
                           make_m16_desc(bf16_ptr(s_O_acc)), make_m16_desc(bf16_ptr(s_O_acc)));
        }
    }
    
    for (int i = tid; i < 4096; i += blockDim.x) *(float*)((char*)s_S_acc + i * 4) = 0; // Reuse S_acc for dQ_h1
    
    for (int k = 0; k < 4; ++k) {
        for (int l = 0; l < 4; ++l) {
            wgmma_16x16x16(bf16_ptr(s_S_acc) + ptr_to_swizzled_offset(s_S_acc, k * 16, l * 16),
                           bf16_ptr(s_D_s) + ptr_to_swizzled_offset(s_D_s, k * 16, 0),
                           bf16_ptr(s_K_h1) + ptr_to_swizzled_offset(s_K_h1, 0, l * 16),
                           s_S_acc + k * 16 * 64 + l * 16,
                           make_m16_desc(bf16_ptr(s_S_acc)), make_m16_desc(bf16_ptr(s_S_acc)));
        }
    }
    __syncthreads();
    
    // Read dQ_h0 and dQ_h1, and perform atomic adds
    for (int i = 0; i < 4; ++i) {
        int idx = ((tid * 2) + i) % 64;
        int idy = ((tid * 2) + i) / 64;
        
        float dq0 = ((float*)s_O_acc)[idy * 16 + idx];
        dQ_h0_acc[i] = dq0;
        
        float dq1 = ((float*)s_S_acc)[idy * 16 + idx];
        dQ_h1_acc[i] = dq1;
    }

    // 7. dK += D_s^T @ Q
    for (int i = tid; i < 4096; i += blockDim.x) *(float*)((char*)s_dP_acc + i * 4) = 0; // Reuse dP_acc for dK_h0
    
    for (int k = 0; k < 4; ++k) {
        for (int l = 0; l < 4; ++l) {
            wgmma_16x16x16(bf16_ptr(s_dP_acc) + ptr_to_swizzled_offset(s_dP_acc, k * 16, l * 16),
                           bf16_ptr(s_D_s) + ptr_to_swizzled_offset(s_D_s, 0, k * 16),
                           bf16_ptr(s_Q_h0) + ptr_to_swizzled_offset(s_Q_h0, 0, l * 16),
                           s_dP_acc + k * 16 * 64 + l * 16,
                           make_m16_desc(bf16_ptr(s_dP_acc)), make_m16_desc(bf16_ptr(s_dP_acc)));
        }
    }
    
    for (int i = tid; i < 4096; i += blockDim.x) *(float*)((char*)s_O_acc + i * 4) = 0; // Reuse O_acc for dK_h1
    
    for (int k = 0; k < 4; ++k) {
        for (int l = 0; l < 4; ++l) {
            wgmma_16x16x16(bf16_ptr(s_O_acc) + ptr_to_swizzled_offset(s_O_acc, k * 16, l * 16),
                           bf16_ptr(s_D_s) + ptr_to_swizzled_offset(s_D_s, 0, k * 16),
                           bf16_ptr(s_Q_h1) + ptr_to_swizzled_offset(s_Q_h1, 0, l * 16),
                           s_O_acc + k * 16 * 64 + l * 16,
                           make_m16_desc(bf16_ptr(s_O_acc)), make_m16_desc(bf16_ptr(s_O_acc)));
        }
    }
    __syncthreads();
    
    for (int i = 0; i < 4; ++i) {
        int idx = ((tid * 2) + i) % 64;
        int idy = ((tid * 2) + i) / 64;
        
        float dk0 = ((float*)s_dP_acc)[idy * 16 + idx];
        dK_h0_acc[i] = dk0;
        
        float dk1 = ((float*)s_O_acc)[idy * 16 + idx];
        dK_h1_acc[i] = dk1;
    }

    // 8. dV += P^T @ dO
    for (int i = tid; i < 4096; i += blockDim.x) *(float*)((char*)s_dP_acc + i * 4) = 0; // Reuse dP_acc for dV_h0
    
    for (int k = 0; k < 4; ++k) {
        for (int l = 0; l < 4; ++l) {
            wgmma_16x16x16(bf16_ptr(s_dP_acc) + ptr_to_swizzled_offset(s_dP_acc, k * 16, l * 16),
                           bf16_ptr(s_P) + ptr_to_swizzled_offset(s_P, 0, k * 16),
                           bf16_ptr(s_dO_h0) + ptr_to_swizzled_offset(s_dO_h0, 0, l * 16),
                           s_dP_acc + k * 16 * 64 + l * 16,
                           make_m16_desc(bf16_ptr(s_dP_acc)), make_m16_desc(bf16_ptr(s_dP_acc)));
        }
    }
    
    for (int i = tid; i < 4096; i += blockDim.x) *(float*)((char*)s_O_acc + i * 4) = 0; // Reuse O_acc for dV_h1
    
    for (int k = 0; k < 4; ++k) {
        for (int l = 0; l < 4; ++l) {
            wgmma_16x16x16(bf16_ptr(s_O_acc) + ptr_to_swizzled_offset(s_O_acc, k * 16, l * 16),
                           bf16_ptr(s_P) + ptr_to_swizzled_offset(s_P, 0, k * 16),
                           bf16_ptr(s_dO_h1) + ptr_to_swizzled_offset(s_dO_h1, 0, l * 16),
                           s_O_acc + k * 16 * 64 + l * 16,
                           make_m16_desc(bf16_ptr(s_O_acc)), make_m16_desc(bf16_ptr(s_O_acc)));
        }
    }
    __syncthreads();
    
    for (int i = 0; i < 4; ++i) {
        int idx = ((tid * 2) + i) % 64;
        int idy = ((tid * 2) + i) / 64;
        
        float dv0 = ((float*)s_dP_acc)[idy * 16 + idx];
        dV_h0_acc[i] = dv0;
        
        float dv1 = ((float*)s_O_acc)[idy * 16 + idx];
        dV_h1_acc[i] = dv1;
    }

    // Perform atomic adds safely within block bounds
    if (k_blk * 64 + row * 16 < S && j_blk * 64 + col * 8 < S) {
        for (int i = 0; i < 4; ++i) {
            int idx = ((tid * 2) + i) % 64;
            int idy = ((tid * 2) + i) / 64;
            
            if (k_blk * 64 + idy < S && j_blk * 64 + idx < S) {
                atomicAdd(&dQ_workspace[bh * S * 128 + (k_blk * 64 + idy) * 128 + idx], dQ_h0_acc[i]);
                atomicAdd(&dQ_workspace[bh * S * 128 + (k_blk * 64 + idy) * 128 + idx + 64], dQ_h1_acc[i]);
                
                atomicAdd(&dK_workspace[bh * S * 128 + (j_blk * 64 + idy) * 128 + idx], dK_h0_acc[i]);
                atomicAdd(&dK_workspace[bh * S * 128 + (j_blk * 64 + idy) * 128 + idx + 64], dK_h1_acc[i]);
                
                atomicAdd(&dV_workspace[bh * S * 128 + (j_blk * 64 + idy) * 128 + idx], dV_h0_acc[i]);
                atomicAdd(&dV_workspace[bh * S * 128 + (j_blk * 64 + idy) * 128 + idx + 64], dV_h1_acc[i]);
            }
        }
    }
}

__global__ __launch_bounds__(128) void convert_to_bf16(const float* src, __nv_bfloat16* dst, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2bfloat16(src[idx]);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = Q.size(3);
    
    float* L_data = static_cast<float*>(L.data_ptr());
    
    __nv_bfloat16* Q_ptr = static_cast<__nv_bfloat16*>(Q.data_ptr());
    __nv_bfloat16* K_ptr = static_cast<__nv_bfloat16*>(K.data_ptr());
    __nv_bfloat16* V_ptr = static_cast<__nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* dO_ptr = static_cast<__nv_bfloat16*>(dO.data_ptr());
    __nv_bfloat16* dQ_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_dO;
    
    DRV_CHECK(create_tma_3d_descriptor_2B(&tma_Q, Q_ptr, d, S, B * H, 64, 64, 1, CU_TENSOR_MAP_SWIZZLE_128B));
    DRV_CHECK(create_tma_3d_descriptor_2B(&tma_K, K_ptr, d, S, B * H, 64, 64, 1, CU_TENSOR_MAP_SWIZZLE_128B));
    DRV_CHECK(create_tma_3d_descriptor_2B(&tma_V, V_ptr, d, S, B * H, 64, 64, 1, CU_TENSOR_MAP_SWIZZLE_128B));
    DRV_CHECK(create_tma_3d_descriptor_2B(&tma_dO, dO_ptr, d, S, B * H, 64, 64, 1, CU_TENSOR_MAP_SWIZZLE_128B));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    size_t total_elements = B * H * S * d;
    
    float* dQ_workspace = nullptr;
    float* dK_workspace = nullptr;
    float* dV_workspace = nullptr;

    CUDA_CHECK(cudaMallocAsync(&dQ_workspace, total_elements * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&dK_workspace, total_elements * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&dV_workspace, total_elements * sizeof(float), stream));
    
    CUDA_CHECK(cudaMemsetAsync(dQ_workspace, 0, total_elements * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(dK_workspace, 0, total_elements * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(dV_workspace, 0, total_elements * sizeof(float), stream));
    
    int64_t threads = 128;
    int64_t blocks_x = (S + 63) / 64;
    int64_t blocks_y = (S + 63) / 64;
    int64_t blocks_z = B * H;
    dim3 grid(blocks_x, blocks_y, blocks_z);
    
    int smem_size = 14 * 8192 + 128; // 14 * 8KB + padding for mbarriers
    CUDA_CHECK(cudaFuncSetAttribute((const void*)mha_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    mha_bwd_kernel<<<grid, threads, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, tma_dO,
        dQ_workspace, dK_workspace, dV_workspace,
        L_data, S
    );
    CUDA_CHECK(cudaGetLastError());
    
    size_t items_per_block = 256;
    int num_blocks = (total_elements + items_per_block - 1) / items_per_block;
    
    convert_to_bf16<<<num_blocks, items_per_block, 0, stream>>>(dQ_workspace, dQ_ptr, total_elements);
    convert_to_bf16<<<num_blocks, items_per_block, 0, stream>>>(dK_workspace, dK_ptr, total_elements);
    convert_to_bf16<<<num_blocks, items_per_block, 0, stream>>>(dV_workspace, dV_ptr, total_elements);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaFreeAsync(dQ_workspace, stream));
    CUDA_CHECK(cudaFreeAsync(dK_workspace, stream));
    CUDA_CHECK(cudaFreeAsync(dV_workspace, stream));
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

} // namespace tvm_ffi_example_cuda