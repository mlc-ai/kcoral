#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <stdio.h>
#include <cmath>
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

__device__ __forceinline__ void tma_load_4d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}

__device__ __forceinline__ int swizzle_128B_offset(int r, int c) {
    return r * 64 + (((r % 8) ^ (c / 8)) * 8 + (c % 8));
}

void load_128B_swizzled(__nv_bfloat16* smem, const __nv_bfloat16* gmem, int rows, int cols, int stride, int max_rows) {
    uint32_t num_vectors = cols / 8; 
    for (int i = threadIdx.x; i < rows * num_vectors; i += blockDim.x) {
        int r = i / num_vectors;
        int c_vec = i % num_vectors;
        int c = c_vec * 8;
        if (r < max_rows) {
            uint4 val;
            if (c < cols) {
                val = *(const uint4*)&gmem[r * stride + c];
            } else {
                val = {0, 0, 0, 0};
            }
            int c_rem = c / 8;
            int sc_vec = c_rem ^ (r & 7);
            int sc = r * cols + sc_vec * 8 + (c % 8);
            *(uint4*)&smem[sc] = val;
        }
    }
}

CUresult create_tma_4d_descriptor(CUtensorMap* d, void* globalAddress,
                                  uint64_t dim0, uint64_t dim1, uint64_t dim2, uint64_t dim3,
                                  uint32_t box0, uint32_t box1,
                                  CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[4] = {dim0, dim1, dim2, dim3};
    cuuint64_t globalStrides[3] = {dim0 * 2, dim0 * dim1 * 2, dim0 * dim1 * dim2 * 2};
    cuuint32_t boxDim[4] = {box0, box1, 1, 1};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
        globalAddress, globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

__global__ void run_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O_ptr, float* LSE_ptr,
    int S, int num_heads, int batch_size) 
{
    extern __shared__ __align__(128) uint8_t smem_pool[];
    uintptr_t pool_addr = (uintptr_t)smem_pool;
    uint8_t* cur = smem_pool + (1024 - (pool_addr % 1024)) % 1024;
    
    __nv_bfloat16* s_Q0 = (__nv_bfloat16*)cur; cur += 8192;
    __nv_bfloat16* s_Q1 = (__nv_bfloat16*)cur; cur += 8192;
    __nv_bfloat16* s_K0 = (__nv_bfloat16*)cur; cur += 8192;
    __nv_bfloat16* s_K1 = (__nv_bfloat16*)cur; cur += 8192;
    __nv_bfloat16* s_V0 = (__nv_bfloat16*)cur; cur += 8192;
    __nv_bfloat16* s_V1 = (__nv_bfloat16*)cur; cur += 8192;
    __nv_bfloat16* s_P  = (__nv_bfloat16*)cur; cur += 8192;
    uint64_t* mbar = (uint64_t*)cur; cur += 16; 
    
    int head_idx = blockIdx.y;
    int batch_idx = blockIdx.z;
    int q_base = blockIdx.x * 64;
    int flattened_head = head_idx + batch_idx * num_heads;
    int global_q_base = (flattened_head * S) + q_base;
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&mbar[0], 1);
        init_smem_barrier_fn(&mbar[1], 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&mbar[0], 16384);
        
        const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(tma_Q.globalAddress);
        const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(tma_K.globalAddress);
        const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(tma_V.globalAddress);
        
        uint64_t head_offset = (uint64_t)(batch_idx * num_heads + head_idx) * (128 * 64); 
        
        uint8_t* cur_Q0 = (uint8_t*)s_Q0;
        uint8_t* cur_Q1 = (uint8_t*)s_Q1;
        load_128B_swizzled(s_Q0, Q_ptr + head_offset + q_base * 128, 64, 64, 128, S - q_base);
        load_128B_swizzled(s_Q1, (const __nv_bfloat16*)(Q_ptr + head_offset + q_base * 128 + 64), 64, 64, 128, S - q_base);
        
        mbarrier_arrive_and_expect_tx_fn(&mbar[1], 32768);
        load_128B_swizzled(s_K0, K_ptr + head_offset, 64, 64, 128, S);
        load_128B_swizzled(s_K1, (const __nv_bfloat16*)(K_ptr + head_offset + 64), 64, 64, 128, S);
        load_128B_swizzled(s_V0, V_ptr + head_offset, 64, 64, 128, S);
        load_128B_swizzled(s_V1, (const __nv_bfloat16*)(V_ptr + head_offset + 64), 64, 64, 128, S);
    }
    
    float O_scaled_left[4] = {0};
    float O_scaled_right[4] = {0};
    
    float row_max_prev[2] = {-INFINITY, -INFINITY};
    float row_sum_prev[2] = {0, 0};
    
    int next_barrier_phase = 0;
    float scale_factor = 1.0f / sqrtf(128.0f);
    
    int warp_id = threadIdx.x / 32;
    int my_r_base = warp_id * 16;
    
    for (int k_base = 0; k_base <= q_base; k_base += 64) {
        int next_k_base = k_base + 64;
        
        if (next_k_base <= q_base && next_k_base < S) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&mbar[1], 32768);
                
                uint8_t* cur_K0 = (uint8_t*)s_K0;
                uint8_t* cur_K1 = (uint8_t*)s_K1;
                uint8_t* cur_V0 = (uint8_t*)s_V0;
                uint8_t* cur_V1 = (uint8_t*)s_V1;
                
                const __nv_bfloat16* next_K = K_ptr + head_offset + next_k_base * 128;
                const __nv_bfloat16* next_V = V_ptr + head_offset + next_k_base * 128;
                
                load_128B_swizzled(s_K0, next_K, 64, 64, 128, S - next_k_base);
                load_128B_swizzled(s_K1, (const __nv_bfloat16*)(next_K + 64), 64, 64, 128, S - next_k_base);
                load_128B_swizzled(s_V0, next_V, 64, 64, 128, S - next_k_base);
                load_128B_swizzled(s_V1, (const __nv_bfloat16*)(next_V + 64), 64, 64, 128, S - next_k_base);
            }
        }
        
        mbarrier_wait_fn(&mbar[1], next_barrier_phase & 1);
        
        float S_val[2] = {0};
        int my_c_base = (threadIdx.x % 32) * 2;
        int my_c = my_c_base;
        
        for (int k_blk = 0; k_blk < 4; k_blk++) {
            float sum0 = 0, sum1 = 0;
            for(int k = 0; k < 16; k++) {
                int sc = swizzle_128B_offset(my_r, k_blk * 16 + k);
                sum0 += __bfloat162float(s_Q0[sc]) * __bfloat162float(s_K0[my_c + k_blk * 16 + k + (my_c / 64) * 64]) * scale_factor;
                sum1 += __bfloat162float(s_Q1[sc]) * __bfloat162float(s_K1[my_c + k_blk * 16 + k + (my_c / 64) * 64]) * scale_factor;
            }
            S_val[my_c / 32] = sum0;
            S_val[my_c / 32 + 1] = sum1;
            my_c += 16;
        }
        
        float current_row_max[2] = {-INFINITY, -INFINITY};
        for (int i = 0; i < 2; i++) {
            int global_q_idx = q_base + my_r;
            int global_k_idx = k_base + (threadIdx.x % 32) * 2;
            if (global_q_idx < global_k_idx || global_k_idx >= S) {
                S_val[i] = -INFINITY;
            }
            if (S_val[i] != -INFINITY) {
                current_row_max[i] = fmaxf(current_row_max[i], S_val[i]);
            }
        }
        
        float final_max[2];
        for(int i = 0; i < 2; i++) {
            float my_max = current_row_max[i];
            for (int offset = 1; offset < 32; offset *= 2) {
                my_max = fmaxf(my_max, __shfl_xor_sync(0xFFFFFFFF, my_max, offset));
            }
            final_max[i] = my_max;
        }
        
        float scale_l[2] = {1.0f, 1.0f};
        float scale_r[2] = {1.0f, 1.0f};
        
        for(int i = 0; i < 2; i++) {
            if (final_max[i] > row_max_prev[i]) {
                float s = fast_exp2f_fn((row_max_prev[i] - final_max[i]) * 1.4426950f);
                scale_l[i] = s;
                scale_r[i] = s;
                row_sum_prev[i] *= s;
            }
        }
        
        for(int k_blk = 0; k_blk < 4; k_blk++) {
            O_scaled_left[k_blk] *= scale_l[(k_blk * 16) / 32];
            O_scaled_right[k_blk] *= scale_r[(k_blk * 16) / 32];
        }
        
        float current_row_sum[2] = {0.0f, 0.0f};
        for(int k_blk = 0; k_blk < 4; k_blk++) {
            for(int k = 0; k < 16; k++) {
                int idx = k_blk * 16 + k;
                int my_r_local = my_r_base + (threadIdx.x % 32) / 2 * 2;
                int global_q_idx = q_base + my_r_local;
                int global_k_idx = k_base + idx;
                
                float p = 0.0f;
                if (global_q_idx >= global_k_idx && global_k_idx < S) {
                    p = fast_exp2f_fn((S_val[idx / 32] - final_max[idx / 32]) * 1.4426950f);
                }
                current_row_sum[idx / 32] += p;
                
                s_P[swizzle_128B_offset(my_r, k_blk * 16 + k)] = __float2bfloat16(p);
            }
        }
        
        for(int i = 0; i < 2; i++) {
            float my_sum = current_row_sum[i];
            for (int offset = 1; offset < 32; offset *= 2) {
                my_sum += __shfl_xor_sync(0xFFFFFFFF, my_sum, offset);
            }
            row_sum_prev[i] += my_sum;
            row_max_prev[i] = final_max[i];
        }
        
        for (int k_blk_p = 0; k_blk_p < 4; k_blk_p++) {
            float sum0 = 0, sum1 = 0;
            for(int k = 0; k < 16; k++) {
                float p = __bfloat162float(s_P[swizzle_128B_offset(my_r, k_blk_p * 16 + k)]);
                sum0 += p * __bfloat162float(s_V0[my_c + k_blk_p * 16 + k + (my_c / 64) * 64]);
                sum1 += p * __bfloat162float(s_V1[my_c + k_blk_p * 16 + k + (my_c / 64) * 64]);
            }
            O_scaled_left[k_blk_p] = sum0;
            O_scaled_right[k_blk_p] = sum1;
        }
        
        next_barrier_phase++;
        __syncthreads();
    }
    
    if (global_q_idx < S) {
        for (int i = 0; i < 2; i++) {
            float sum = row_sum_prev[i];
            for (int k_blk = i * 2; k_blk < 4; k_blk+=2) {
                float out0 = O_scaled_left[k_blk] / sum;
                float out1 = O_scaled_right[k_blk] / sum;
                
                if (out0 != out0) out0 = 0.0f;
                if (out1 != out1) out1 = 0.0f;
                
                int global_d_idx_0 = my_c + k_blk * 16;
                int global_d_idx_1 = my_c + 1 + k_blk * 16;
                
                if (global_d_idx_0 < 128) {
                    O_ptr[global_q_base * 128 + global_d_idx_0] = __float2bfloat16(out0);
                }
                if (global_d_idx_1 < 128) {
                    O_ptr[global_q_base * 128 + global_d_idx_1] = __float2bfloat16(out1);
                }
            }
        }
        
        if ((threadIdx.x % 32) == 0) {
            LSE_ptr[global_q_base + global_q_idx] = row_max_prev[0] + logf(row_sum_prev[0]);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3); 
    
    if (S == 0) return;
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUtensorMap tma_Q, tma_K, tma_V;
    create_tma_4d_descriptor(&tma_Q, Q.data_ptr(), D, S, H, B, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B);
    create_tma_4d_descriptor(&tma_K, K.data_ptr(), D, S, H, B, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B);
    create_tma_4d_descriptor(&tma_V, V.data_ptr(), D, S, H, B, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B);
    
    uint32_t smem_size = 57344;
    dim3 grid((S + 63) / 64, H, B);
    dim3 block(128);
    
    cudaFuncSetAttribute(run_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, run_kernel, tma_Q, tma_K, tma_V, 
        reinterpret_cast<__nv_bfloat16*>(O.data_ptr()), 
        reinterpret_cast<float*>(LSE.data_ptr()), 
        static_cast<int>(S), static_cast<int>(H), static_cast<int>(B)));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);