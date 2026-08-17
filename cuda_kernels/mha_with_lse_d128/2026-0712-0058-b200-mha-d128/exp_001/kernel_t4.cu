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
        fprintf(stderr, "CU error %d at %s:%d\n",                 \
                (int)_e, __FILE__, __LINE__);                      \
        exit(1);                                                   \
    }                                                              \
} while(0)

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, 
                                     uint64_t dim0, uint64_t dim1, uint64_t dim2, 
                                     uint32_t box0, uint32_t box1, uint32_t box2,
                                     CUtensorMapSwizzle swizzle, 
                                     CUtensorMapL2promotion l2Promotion, 
                                     CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[3] = {dim0, dim1, dim2};
    cuuint64_t globalStrides[2] = {dim0 * 2, dim0 * dim1 * 2};
    cuuint32_t boxDim[3] = {box0, box1, box2};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill
    );
}

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

__global__ void attention_forward_cluster(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O,
    float* LSE,
    uint64_t S,
    uint64_t D,
    uint64_t B,
    uint64_t H) 
{
    extern __shared__ __align__(128) uint8_t smem_pool[];
    uintptr_t pool_addr = (uintptr_t)smem_pool;
    
    uint64_t* mbar_Q   = reinterpret_cast<uint64_t*>(pool_addr);
    uint64_t* mbar_K0  = mbar_Q + 1;
    uint64_t* mbar_K1  = mbar_K0 + 1;
    uint64_t* mbar_V0  = mbar_K1 + 1;
    uint64_t* mbar_V1  = mbar_V0 + 1;
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_Q, 1); 
        init_smem_barrier_fn(mbar_K0, 1); 
        init_smem_barrier_fn(mbar_K1, 1); 
        init_smem_barrier_fn(mbar_V0, 1); 
        init_smem_barrier_fn(mbar_V1, 1); 
    }
    fence_smem_barrier_init_fn();
    __syncthreads();
    
    uint32_t row_base = (uint32_t)(blockIdx.x * 64);
    uint32_t head_idx = row_base / S;
    uint32_t myRow = row_base % S;
    
    uint32_t aligned_addr = pool_addr + 1024 - (pool_addr % 1024);
    char* smem_Q = reinterpret_cast<char*>(aligned_addr);
    
    char* s_Q0_0 = smem_Q;                
    char* s_Q0_1 = smem_Q + 8192;         
    char* s_Q1_0 = smem_Q + 16384;        
    char* s_Q1_1 = smem_Q + 24576;        

    char* s_K0 = smem_Q + 32768;          
    char* s_K1 = smem_Q + 40960;          
    char* s_K2 = smem_Q + 49152;          
    char* s_K3 = smem_Q + 57344;          

    char* s_V0 = smem_Q + 65536;          
    char* s_V1 = smem_Q + 73728;          
    char* s_V2 = smem_Q + 81920;          
    char* s_V3 = smem_Q + 90112;          

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 8192 * 2);
        tma_load_3d_fn(&tma_Q, mbar_Q, s_Q0_0, 0, myRow, head_idx);
        tma_load_3d_fn(&tma_Q, mbar_Q, s_Q0_1, 64, myRow, head_idx);
    }
    
    mbarrier_wait_fn(mbar_Q, 0);
    
    int cur_k = 0;
    int cur_v = 0;
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_K0, 8192 * 2);
        tma_load_3d_fn(&tma_K, mbar_K0, s_K0, 0, 0, head_idx);
        tma_load_3d_fn(&tma_K, mbar_K0, s_K1, 64, 0, head_idx);
        
        mbarrier_arrive_and_expect_tx_fn(mbar_V0, 8192 * 2);
        tma_load_3d_fn(&tma_V, mbar_V0, s_V0, 0, 0, head_idx);
        tma_load_3d_fn(&tma_V, mbar_V0, s_V1, 64, 0, head_idx);
    }
    
    float row_global_max = -1e20f;
    float row_global_sum = 0.0f;
    int phase_K[2] = {0, 0};
    int phase_V[2] = {0, 0};
    
    const float SQRT_D = 1.0f / sqrtf((float)D);
    
    int num_iters = (S + 63) / 64;

    for (int j_blk = 0; j_blk < num_iters; j_blk++) {
        if (threadIdx.x < 32) {
            mbarrier_wait_fn(cur_k == 0 ? mbar_K0 : mbar_K1, phase_K[cur_k]);
        }
        __syncthreads();
        
        float local_max = -1e20f;
        float local_sum = 0.0f;
        
        if (threadIdx.x < 64) {
            float S_vals[64];
            
            for (int i = 0; i < 64; i++) {
                float sum = 0;
                for(int d = 0; d < 64; d++) {
                    int swizzled_col = ((threadIdx.x & 7) ^ (d >> 3)) << 3 | (d & 7);
                    int phys_offset = threadIdx.x * 64 + swizzled_col;
                    
                    __nv_bfloat16 q0 = *(__nv_bfloat16*)(s_Q0_0 + phys_offset * 2);
                    __nv_bfloat16 k0 = *(__nv_bfloat16*)(cur_k == 0 ? s_K0 + phys_offset * 2 : s_K2 + phys_offset * 2);
                    sum += __bfloat162float(q0) * __bfloat162float(k0);
                    
                    __nv_bfloat16 q1 = *(__nv_bfloat16*)(s_Q0_1 + phys_offset * 2);
                    __nv_bfloat16 k1 = *(__nv_bfloat16*)(cur_k == 0 ? s_K1 + phys_offset * 2 : s_K3 + phys_offset * 2);
                    sum += __bfloat162float(q1) * __bfloat162float(k1);
                }
                S_vals[i] = sum * SQRT_D;
            }
            
            for(int i = 0; i < 64; i++) {
                if (j_blk * 64 + i < S) {
                    if (S_vals[i] > local_max) local_max = S_vals[i];
                } else {
                    S_vals[i] = -1e20f;
                }
            }
            
            for(int i = 0; i < 64; i++) {
                float p = expf(S_vals[i] - local_max);
                local_sum += p;
                S_vals[i] = p;
            }
            
            float new_max = fmaxf(row_global_max, local_max);
            float corr_O = expf(row_global_max - new_max);
            float corr_S = expf(local_max - new_max);
            row_global_sum *= corr_O;
            row_global_sum += local_sum * corr_S;
            row_global_max = new_max;
            
            for(int i = 0; i < 64; i++) {
                S_vals[i] *= corr_S;
            }
            
            local_sum = 0.0f;
            for(int i = 0; i < 64; i++) {
                local_sum += S_vals[i];
            }
        }
        __syncthreads();
        
        if (threadIdx.x < 32) {
            mbarrier_wait_fn(cur_v == 0 ? mbar_V0 : mbar_V1, phase_V[cur_v]);
        }
        __syncthreads();
        
        if (threadIdx.x < 64) {
            float P_vals[64];
            for (int i = 0; i < 64; i++) {
                int swizzled_col = ((threadIdx.x & 7) ^ (i >> 3)) << 3 | (i & 7);
                int phys_offset = threadIdx.x * 64 + swizzled_col;
                P_vals[i] = *(float*)(cur_k == 0 ? s_K0 + phys_offset * 4 : s_K2 + phys_offset * 4); 
                P_vals[i] = S_vals[i];
                *(float*)(cur_k == 0 ? s_K0 + phys_offset * 4 : s_K2 + phys_offset * 4) = P_vals[i];
            }
            
            float4 o_regs_packed[8];
            for (int i = 0; i < 8; i++) {
                int swizzled_col = ((threadIdx.x & 7) ^ (i >> 3)) << 3 | (i & 7);
                int phys_offset = threadIdx.x * 64 + swizzled_col;
                o_regs_packed[i] = *(float4*)(cur_v == 0 ? s_V0 + phys_offset * 16 : s_V2 + phys_offset * 16);
            }
            
            float corr_O = expf(row_global_max - row_global_max); 
            
            float4 v0_regs[8];
            float4 v1_regs[8];
            for(int i = 0; i < 8; i++) {
                int swizzled_col = ((threadIdx.x & 7) ^ (i >> 3)) << 3 | (i & 7);
                int phys_offset = threadIdx.x * 64 + swizzled_col;
                v0_regs[i] = *(float4*)((cur_v == 0 ? s_V0 : s_V2) + phys_offset * 16);
                v1_regs[i] = *(float4*)((cur_v == 0 ? s_V1 : s_V3) + phys_offset * 16);
            }
            
            float o_regs[128];
            float* o_f = (float*)o_regs_packed;
            for(int i = 0; i < 128; i++) {
                o_regs[i] = o_f[i] * corr_O;
            }
            
            float* v0_f = (float*)v0_regs;
            float* v1_f = (float*)v1_regs;
            
            for (int i = 0; i < 64; i++) {
                o_regs[i] += P_vals[i] * v0_f[i];
                o_regs[i+64] += P_vals[i] * v1_f[i];
            }
            
            for (int i = 0; i < 8; i++) {
                int swizzled_col = ((threadIdx.x & 7) ^ (i >> 3)) << 3 | (i & 7);
                int phys_offset = threadIdx.x * 64 + swizzled_col;
                *(float4*)((cur_v == 0 ? s_V0 : s_V2) + phys_offset * 16) = o_regs_packed[i];
            }
        }
        
        int next_k = cur_k ^ 1;
        int next_v = cur_v ^ 1;
        
        if (threadIdx.x < 32) {
            if (j_blk + 1 < num_iters) {
                mbarrier_arrive_and_expect_tx_fn(next_k == 0 ? mbar_K0 : mbar_K1, 8192 * 2);
                if (next_k == 0) {
                    tma_load_3d_fn(&tma_K, next_k == 0 ? mbar_K0 : mbar_K1, s_K0, 0, (j_blk + 1) * 64, head_idx);
                    tma_load_3d_fn(&tma_K, next_k == 0 ? mbar_K0 : mbar_K1, s_K1, 64, (j_blk + 1) * 64, head_idx);
                } else {
                    tma_load_3d_fn(&tma_K, next_k == 0 ? mbar_K0 : mbar_K1, s_K2, 0, (j_blk + 1) * 64, head_idx);
                    tma_load_3d_fn(&tma_K, next_k == 0 ? mbar_K0 : mbar_K1, s_K3, 64, (j_blk + 1) * 64, head_idx);
                }
                
                mbarrier_arrive_and_expect_tx_fn(next_v == 0 ? mbar_V0 : mbar_V1, 8192 * 2);
                if (next_v == 0) {
                    tma_load_3d_fn(&tma_V, next_v == 0 ? mbar_V0 : mbar_V1, s_V0, 0, (j_blk + 1) * 64, head_idx);
                    tma_load_3d_fn(&tma_V, next_v == 0 ? mbar_V0 : mbar_V1, s_V1, 64, (j_blk + 1) * 64, head_idx);
                } else {
                    tma_load_3d_fn(&tma_V, next_v == 0 ? mbar_V0 : mbar_V1, s_V2, 0, (j_blk + 1) * 64, head_idx);
                    tma_load_3d_fn(&tma_V, next_v == 0 ? mbar_V0 : mbar_V1, s_V3, 64, (j_blk + 1) * 64, head_idx);
                }
            }
        }
        
        cur_k = next_k;
        cur_v = next_v;
        phase_K[cur_k] ^= 1;
        phase_V[cur_v] ^= 1;
    }
    
    if (threadIdx.x < 64) {
        float4 o_regs_packed[8];
        for (int i = 0; i < 8; i++) {
            int swizzled_col = ((threadIdx.x & 7) ^ (i >> 3)) << 3 | (i & 7);
            int phys_offset = threadIdx.x * 64 + swizzled_col;
            o_regs_packed[i] = *(float4*)(cur_v == 0 ? s_V0 + phys_offset * 16 : s_V2 + phys_offset * 16);
        }
        
        float* o_f = (float*)o_regs_packed;
        for(int i = 0; i < 128; i++) {
            float val = o_f[i] / row_global_sum;
            int col = (i / 2) * 8 + (i % 2) * 4 + (threadIdx.x % 8);
            int row_idx = head_idx * S + threadIdx.x;
            if (row_idx < (int)(B*H*S)) {
                O[row_idx * D + col] = __float2bfloat16(val);
            }
        }
    }
    
    if (threadIdx.x < 64) {
        uint32_t myRow = threadIdx.x;
        int row_idx = head_idx * S + myRow;
        
        if (row_idx < (int)(B*H*S)) {
            float lse = row_global_max + logf(row_global_sum);
            LSE[row_idx] = lse;
        }
    }
}

namespace tvm_ffi {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    uint64_t B = Q.size(0);
    uint64_t H = Q.size(1);
    uint64_t S_val = Q.size(2);
    uint64_t D = Q.size(3);

    void* Q_ptr = Q.data_ptr();
    void* K_ptr = K.data_ptr();
    void* V_ptr = V.data_ptr();
    __nv_bfloat16* O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_ptr = static_cast<float*>(LSE.data_ptr());

    CUtensorMap tma_Q, tma_K, tma_V;

    CU_CHECK(create_tma_3d_descriptor_2B(&tma_Q, Q_ptr, 128, S_val, B * H, 64, 64, 1, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_K, K_ptr, 128, S_val, B * H, 64, 64, 1, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_V, V_ptr, 128, S_val, B * H, 64, 64, 1, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    dim3 grid((S_val + 63) / 64, 1, 1);
    dim3 block(128, 1, 1);

    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 128 * 1024; 
    config.stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 1;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    CUDA_CHECK(cudaFuncSetAttribute((const void*)attention_forward_cluster, cudaFuncAttributeMaxDynamicSharedMemorySize, 128 * 1024));
    CUDA_CHECK(cudaLaunchKernelEx(&config, attention_forward_cluster, tma_Q, tma_K, tma_V, O_ptr, LSE_ptr, S_val, D, B, H));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(config.stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi