#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>
#include <math.h>

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

namespace tvm_ffi_mha_lse {

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

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(bar)),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void wgmma_16x16x16(float* acc, uint64_t desc_A, uint64_t desc_B, uint32_t alpha) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %3, 0;\n"
        "wgmma.m16n16k16.sync.f16.f32 {%0-%15}, %1, %2, p;\n}\n"
        :: "f"(acc[0]), "l"(desc_A), "l"(desc_B), "r"(alpha));
}

__device__ __forceinline__ uint64_t make_desc(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;   // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint64_t make_desc_k_major(void* ptr) {
    return make_desc(ptr, 1, 1024);
}

__device__ __forceinline__ uint64_t make_desc_n_major(void* ptr) {
    return make_desc(ptr, 1024, 1024);
}

__device__ __forceinline__ uint64_t advance_desc_k_major(uint64_t d, int step) {
    uint64_t addr_bits = d & 0x3FFF;
    addr_bits += (step * 16 * 2) / 16; // step * 2
    d &= ~0x3FFF;
    d |= addr_bits;
    return d;
}

__device__ __forceinline__ uint64_t advance_desc_n_major(uint64_t d, int step, int stride_dim_elements) {
    uint64_t addr_bits = d & 0x3FFF;
    uint32_t sbo = stride_dim_elements * 2;
    addr_bits += (step * 16 * sbo) / 16;
    d &= ~0x3FFF;
    d |= addr_bits;
    return d;
}

__device__ __forceinline__ void wgmma_load_16x16_k_major(__nv_bfloat16* smem, const __nv_bfloat16* src) {
    for(int r = 0; r < 16; ++r) {
        for(int c = 0; c < 16; ++c) {
            int c_swizzled = (((c >> 3) & 7) ^ (r & 7)) << 3 | (c & 7);
            smem[r * 16 + c] = src[r * 64 + c_swizzled];
        }
    }
}

__device__ __forceinline__ void wgmma_load_16x16_n_major(__nv_bfloat16* smem, const __nv_bfloat16* src, int base_row) {
    for(int r = 0; r < 16; ++r) {
        int r_actual = base_row + r;
        for(int c = 0; c < 16; ++c) {
            int col_chunk = c >> 3;
            int col_rem = c & 7;
            int r_swizzled = r_actual & 7;
            int chunk_swizzled = col_chunk ^ r_swizzled;
            int c_swizzled = (chunk_swizzled << 3) | col_rem;
            smem[r * 16 + c] = src[r_actual * 64 + c_swizzled];
        }
    }
}

__global__ __launch_bounds__(128, 1) void flashattention_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O,
    float* LSE,
    uint32_t S
) {
    int bh_idx = blockIdx.y;
    int s_idx = blockIdx.x;
    int s_base = s_idx * 64;
    
    int phase_q = 0;
    int phase_kv = 0;
    
    extern __shared__ __align__(1024) __nv_bfloat16 smem_pool[];
    __nv_bfloat16* smem_Q = smem_pool;               
    __nv_bfloat16* smem_K = smem_Q + 16384;          
    __nv_bfloat16* smem_V = smem_K + 8192;           
    __nv_bfloat16* smem_P = smem_V + 8192;           
    __nv_bfloat16* smem_tmp = smem_P + 8192;         
    uint64_t* bar_q = (uint64_t*)(smem_tmp + 512);   
    uint64_t* bar_kv = bar_q + 1;                     
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar_q, 1);
        init_smem_barrier_fn(bar_kv, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_q, 2 * 8192); 
        
        tma_load_3d_fn(&tma_Q, bar_q, (void*)smem_Q, 0, s_base, bh_idx);
        tma_load_3d_fn(&tma_Q, bar_q, (void*)(smem_Q + 8192), 64, s_base, bh_idx); 
    }
    tmem_load_fence_fn();
    mbarrier_wait_fn(bar_q, phase_q);
    phase_q ^= 1;
    
    __nv_bfloat16* my_Q0 = smem_Q + warp_id * 1024;
    __nv_bfloat16* my_Q1 = smem_Q + 8192 + warp_id * 1024;
    
    float running_max = -1e20f;
    float running_sum = 0.0f;
    float acc_O0[4][16] = {0};
    float acc_O1[4][16] = {0};
    
    for (int block_start = 0; block_start < S; block_start += 64) {
        __syncthreads();
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_kv, 4 * 8192); 
            
            tma_load_3d_fn(&tma_K, bar_kv, (void*)smem_K, 0, block_start, bh_idx);
            tma_load_3d_fn(&tma_K, bar_kv, (void*)(smem_K + 8192), 64, block_start, bh_idx);
            
            tma_load_3d_fn(&tma_V, bar_kv, (void*)smem_V, 0, block_start, bh_idx);
            tma_load_3d_fn(&tma_V, bar_kv, (void*)(smem_V + 8192), 64, block_start, bh_idx);
        }
        tmem_load_fence_fn();
        mbarrier_wait_fn(bar_kv, phase_kv);
        phase_kv ^= 1;
        
        __nv_bfloat16* my_K0 = smem_K + warp_id * 1024;
        __nv_bfloat16* my_K1 = smem_K + 8192 + warp_id * 1024;
        __nv_bfloat16* my_V0 = smem_V;
        __nv_bfloat16* my_V1 = smem_V + 8192;
        
        float acc_P[4][16];
        for(int i=0; i<4; ++i) {
            for(int j=0; j<16; ++j) {
                acc_P[i][j] = 0.0f;
            }
        }
        
        for (int n_step = 0; n_step < 4; n_step++) {
            wgmma_load_16x16_k_major(smem_tmp, my_Q0 + n_step * 16);
            uint64_t desc_Q0 = make_desc_k_major(smem_tmp);
            
            wgmma_load_16x16_k_major(smem_tmp + 256, my_K0 + n_step * 16);
            uint64_t desc_K0 = make_desc_k_major(smem_tmp + 256);
            
            for (int k_step = 0; k_step < 4; k_step++) {
                uint64_t dq = advance_desc_k_major(desc_Q0, k_step);
                uint64_t dk = advance_desc_k_major(desc_K0, k_step);
                wgmma_16x16x16(acc_P[n_step], dq, dk, 1);
            }
            
            wgmma_load_16x16_k_major(smem_tmp, my_Q1 + n_step * 16);
            uint64_t desc_Q1 = make_desc_k_major(smem_tmp);
            
            wgmma_load_16x16_k_major(smem_tmp + 256, my_K1 + n_step * 16);
            uint64_t desc_K1 = make_desc_k_major(smem_tmp + 256);
            
            for (int k_step = 0; k_step < 4; k_step++) {
                uint64_t dq = advance_desc_k_major(desc_Q1, k_step);
                uint64_t dk = advance_desc_k_major(desc_K1, k_step);
                wgmma_16x16x16(acc_P[n_step], dq, dk, 1);
            }
        }
        
        float local_max = -1e20f;
        for(int n = 0; n < 4; ++n) {
            for(int j = 0; j < 16; ++j) {
                local_max = fmaxf(local_max, acc_P[n][j]);
            }
        }
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, 1));
        
        float new_max = fmaxf(running_max, local_max);
        float factor = (new_max > running_max) ? expf(running_max - new_max) : 1.0f;
        running_sum *= factor;
        running_max = new_max;
        
        float local_sum = 0.0f;
        int row = warp_id * 16 + lane_id / 2;
        for(int n = 0; n < 4; ++n) {
            for(int j = 0; j < 16; ++j) {
                float p_val = acc_P[n][j];
                int global_col = block_start + n * 16 + j;
                
                if (global_col >= S) {
                    p_val = -1e20f;
                } else {
                    p_val *= 0.08838834764f; // 1 / sqrt(128)
                }
                
                p_val = expf(p_val - new_max) * factor;
                local_sum += p_val;
                acc_P[n][j] = p_val;
            }
        }
        
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, 1);
        running_sum += local_sum;
        
        for(int n = 0; n < 4; ++n) {
            for(int j = 0; j < 16; ++j) {
                acc_P[n][j] /= running_sum;
            }
        }
        
        if ((lane_id % 2) == 0) { 
            for(int n = 0; n < 4; ++n) {
                for(int j = 0; j < 16; ++j) {
                    int col = n * 16 + j;
                    int c_swizzled = (((col >> 3) & 7) ^ (row & 7)) << 3 | (col & 7);
                    smem_P[row * 64 + c_swizzled] = __float2bfloat16(acc_P[n][j]);
                }
            }
        }
        __syncthreads();
        
        for (int n_step = 0; n_step < 4; n_step++) {
            for (int k_step = 0; k_step < 4; k_step++) {
                wgmma_load_16x16_k_major(smem_tmp, smem_P + warp_id * 1024 + k_step * 16);
                uint64_t desc_P = make_desc_k_major(smem_tmp);
                
                wgmma_load_16x16_n_major(smem_tmp + 256, my_V0, k_step * 16 + n_step * 1024);
                uint64_t desc_V0 = make_desc_n_major(smem_tmp + 256);
                
                wgmma_16x16x16(acc_O0[n_step], desc_P, desc_V0, 1);
                
                wgmma_load_16x16_n_major(smem_tmp + 256, my_V1, k_step * 16 + n_step * 1024);
                uint64_t desc_V1 = make_desc_n_major(smem_tmp + 256);
                
                wgmma_16x16x16(acc_O1[n_step], desc_P, desc_V1, 1);
            }
        }
        
        __syncthreads();
    }
    
    for(int n = 0; n < 4; ++n) {
        for(int j = 0; j < 16; j += 2) {
            int row = warp_id * 16 + lane_id / 2;
            int col0 = n * 16 + j;
            int col1 = n * 16 + j + 1;
            
            __nv_bfloat16 out0_0 = __float2bfloat16(acc_O0[n][j]);
            __nv_bfloat16 out0_1 = __float2bfloat16(acc_O0[n][j+1]);
            uint32_t idx0 = bh_idx * S * 128 + row * 128 + col0;
            *(uint32_t*)(&O[idx0]) = *(uint32_t*)(&out0_0);
            *(uint32_t*)(&O[idx0+1]) = *(uint32_t*)(&out0_1);
            
            __nv_bfloat16 out1_0 = __float2bfloat16(acc_O1[n][j]);
            __nv_bfloat16 out1_1 = __float2bfloat16(acc_O1[n][j+1]);
            uint32_t idx1 = bh_idx * S * 128 + row * 128 + col0 + 64;
            *(uint32_t*)(&O[idx1]) = *(uint32_t*)(&out1_0);
            *(uint32_t*)(&O[idx1+1]) = *(uint32_t*)(&out1_1);
        }
    }
    
    if ((lane_id % 2) == 0) {
        int row = warp_id * 16 + lane_id / 2;
        if (bh_idx * S + row < bh_idx * S + S) {
            LSE[bh_idx * S + row] = running_max + logf(running_sum);
        }
    }
}

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, 
                                     uint64_t gmem_inner_dim, uint64_t gmem_mid_dim, uint64_t gmem_outer_dim, 
                                     uint32_t smem_inner_dim, uint32_t smem_mid_dim, uint32_t smem_outer_dim, 
                                     CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[3] = {gmem_inner_dim, gmem_mid_dim, gmem_outer_dim};
    cuuint64_t globalStrides[2] = {gmem_inner_dim * 2, gmem_inner_dim * gmem_mid_dim * 2};
    cuuint32_t boxDim[3] = {smem_inner_dim, smem_mid_dim, smem_outer_dim};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill
    );
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    uint32_t B = Q.size(0);
    uint32_t H = Q.size(1);
    uint32_t S = Q.size(2);
    uint32_t D = Q.size(3);
    
    void* q_ptr = Q.data_ptr();
    void* k_ptr = K.data_ptr();
    void* v_ptr = V.data_ptr();
    __nv_bfloat16* o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_ptr = static_cast<float*>(LSE.data_ptr());
    
    CUtensorMap tma_Q, tma_K, tma_V;
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_Q, q_ptr, D, S, B * H, 64, 64, 1, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_K, k_ptr, D, S, B * H, 64, 64, 1, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_V, v_ptr, D, S, B * H, 64, 64, 1, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
    dim3 grid(S / 64, B * H);
    dim3 block(128);
    
    uint32_t smem_size = 65536;
    CUDA_CHECK(cudaFuncSetAttribute(flashattention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    flashattention_kernel<<<grid, block, smem_size, stream>>>(tma_Q, tma_K, tma_V, o_ptr, lse_ptr, S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_lse::run);

}