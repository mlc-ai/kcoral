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
        fprintf(stderr, "Driver error %d at %s:%d\n",             \
                _e, __FILE__, __LINE__);                           \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_mha {

CUresult create_tma_4d_descriptor_2B(CUtensorMap* d, void* globalAddress, 
                                     uint64_t dim0, uint64_t dim1, uint64_t dim2, uint64_t dim3,
                                     uint32_t box0, uint32_t box1, uint32_t box2, uint32_t box3, 
                                     CUtensorMapDataType dataType,
                                     CUtensorMapSwizzle swizzle, 
                                     CUtensorMapL2promotion l2Promotion, 
                                     CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[4] = {dim0, dim1, dim2, dim3};
    cuuint64_t globalStrides[3] = {dim0 * 2, dim0 * dim1 * 2, dim0 * dim1 * dim2 * 2}; 
    cuuint32_t boxDim[4] = {box0, box1, box2, box3};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, dataType, 4, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        l2Promotion, oobFill
    );
}

__device__ __forceinline__ void tma_load_4d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}

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

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__device__ __forceinline__ void cluster_sync_fn() {
    asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n" ::: "memory");
}

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

__device__ __forceinline__ int swizzle_128B(int row, int col) {
    int x = col / 8;
    int rem = col % 8;
    int swizzled_x = x ^ (row % 8);
    return swizzled_x * 8 + rem;
}

template<typename T>
__device__ __forceinline__ T load_swizzled(T* ptr, int row, int col) {
    return ptr[row * 64 + swizzle_128B(row, col)];
}

#define load_p(row, col) load_swizzled(smem_P, row, col)
#define store_s(half, col, val) smem_S[row * 128 + (half) * 64 + swizzle_128B(row, (half) * 64 + col)] = val;

__global__ __launch_bounds__(128, 1) void mha_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O,
    float* LSE,
    uint32_t S_len)
{
    setmaxnreg_inc_sync_fn<248>();

    uint32_t q_block = blockIdx.x * 128;
    uint32_t bh = blockIdx.y;
    uint32_t b = bh / 48;
    uint32_t h = bh % 48;
    uint32_t cr = cluster_rank_fn();
    
    extern __shared__ char smem_raw[];
    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem_raw);
    uint32_t smem_aligned = (smem_addr + 1023) & ~1023;
    char* smem = smem_raw + (smem_aligned - smem_addr);
    
    __nv_bfloat16* smem_Q_0 = (__nv_bfloat16*)smem;                   // 8192 bytes
    __nv_bfloat16* smem_Q_1 = (__nv_bfloat16*)(smem + 8192);          // 8192 bytes
    __nv_bfloat16* smem_K_0 = (__nv_bfloat16*)(smem + 16384);         // 16384 bytes
    __nv_bfloat16* smem_K_1 = (__nv_bfloat16*)(smem + 32768);         // 16384 bytes
    __nv_bfloat16* smem_V_0 = (__nv_bfloat16*)(smem + 49152);         // 16384 bytes
    __nv_bfloat16* smem_V_1 = (__nv_bfloat16*)(smem + 65536);         // 16384 bytes
    float* smem_S           = (float*)(smem + 81920);                 // 32768 bytes
    __nv_bfloat16* smem_P   = (__nv_bfloat16*)(smem + 114688);        // 8192 bytes 
    
    uint64_t* bar_q   = (uint64_t*)(smem + 122880);
    uint64_t* bar_k0  = (uint64_t*)(smem + 122888);
    uint64_t* bar_k1  = (uint64_t*)(smem + 122896);
    uint64_t* bar_v0  = (uint64_t*)(smem + 122904);
    uint64_t* bar_v1  = (uint64_t*)(smem + 122912);

    __shared__ float prev_max[64];
    __shared__ float sum_exp[64];
    if (threadIdx.x < 64) {
        prev_max[threadIdx.x] = -INFINITY;
        sum_exp[threadIdx.x] = 0;
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar_q, 1);
        init_smem_barrier_fn(bar_k0, 1);
        init_smem_barrier_fn(bar_k1, 1);
        init_smem_barrier_fn(bar_v0, 1);
        init_smem_barrier_fn(bar_v1, 1);
    }
    fence_smem_barrier_init_fn();
    cluster_sync_fn();

    uint32_t lane_offset = (threadIdx.x / 32) * 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t cr_lane_offset = cr * 64;
    
    int c1_q = q_block + cr * 64;
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_q, 16384); 
        tma_load_4d_fn(&tma_Q, bar_q, smem_Q_0, 0, c1_q, h, b);
        tma_load_4d_fn(&tma_Q, bar_q, (char*)smem_Q_0 + 8192, 64, c1_q, h, b);
        
        mbarrier_arrive_and_expect_tx_fn(bar_k0, 16384);
        tma_load_4d_fn(&tma_K, bar_k0, smem_K_0, 0, 0, h, b);
        tma_load_4d_fn(&tma_K, bar_k0, (char*)smem_K_0 + 8192, 64, 0, h, b);
        
        mbarrier_arrive_and_expect_tx_fn(bar_v0, 16384);
        tma_load_4d_fn(&tma_V, bar_v0, smem_V_0, 0, 0, h, b);
        tma_load_4d_fn(&tma_V, bar_v0, (char*)smem_V_0 + 8192, 64, 0, h, b);
    }
    
    for (int i = threadIdx.x; i < 64 * 64; i += blockDim.x) {
        int row = i / 64;
        int col = i % 64;
        smem_S[row * 128 + swizzle_128B(row, col)] = 0;
        smem_S[row * 128 + 64 + swizzle_128B(row, 64 + col)] = 0;
    }
    
    mbarrier_wait_fn(bar_q, 0);

    float scale = 1.0f / sqrtf(128.0f);

    uint32_t phase_k0 = 0;
    uint32_t phase_k1 = 0;
    uint32_t phase_v0 = 0;
    uint32_t phase_v1 = 0;

    uint32_t num_steps = (S_len + 63) / 64;
    
    for (int step = 0; step < num_steps; ++step) {
        uint32_t next_step = step + 1;
        uint32_t kv_block = step * 64;
        uint32_t next_kv_block = next_step * 64;
        
        int buf = step % 2;
        int next_buf = next_step % 2;
        
        uint64_t* current_bar_k = (buf == 0) ? bar_k0 : bar_k1;
        uint64_t* next_bar_k = (next_buf == 0) ? bar_k0 : bar_k1;
        
        uint64_t* current_bar_v = (buf == 0) ? bar_v0 : bar_v1;
        uint64_t* next_bar_v = (next_buf == 0) ? bar_v0 : bar_v1;
        
        __nv_bfloat16* current_K_0 = (buf == 0) ? smem_K_0 : smem_K_1;
        __nv_bfloat16* current_V_0 = (buf == 0) ? smem_V_0 : smem_V_1;
        __nv_bfloat16* current_K_1 = (buf == 0) ? (__nv_bfloat16*)((char*)smem_K_0 + 8192) : (__nv_bfloat16*)((char*)smem_K_1 + 8192);
        __nv_bfloat16* current_V_1 = (buf == 0) ? (__nv_bfloat16*)((char*)smem_V_0 + 8192) : (__nv_bfloat16*)((char*)smem_V_1 + 8192);
        
        __nv_bfloat16* next_K_0 = (next_buf == 0) ? smem_K_0 : smem_K_1;
        __nv_bfloat16* next_V_0 = (next_buf == 0) ? smem_V_0 : smem_V_1;
        __nv_bfloat16* next_K_1 = (next_buf == 0) ? (__nv_bfloat16*)((char*)smem_K_0 + 8192) : (__nv_bfloat16*)((char*)smem_K_1 + 8192);
        __nv_bfloat16* next_V_1 = (next_buf == 0) ? (__nv_bfloat16*)((char*)smem_V_0 + 8192) : (__nv_bfloat16*)((char*)smem_V_1 + 8192);

        if (next_step < num_steps) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(next_bar_k, 16384);
                tma_load_4d_fn(&tma_K, next_bar_k, next_K_0, 0, next_kv_block, h, b);
                tma_load_4d_fn(&tma_K, next_bar_k, next_K_1, 64, next_kv_block, h, b);
                
                mbarrier_arrive_and_expect_tx_fn(next_bar_v, 16384);
                tma_load_4d_fn(&tma_V, next_bar_v, next_V_0, 0, next_kv_block, h, b);
                tma_load_4d_fn(&tma_V, next_bar_v, next_V_1, 64, next_kv_block, h, b);
            }
        }
        
        mbarrier_wait_fn(current_bar_k, (buf == 0) ? phase_k0 : phase_k1);
        if ((buf == 0) ? phase_k0 : phase_k1) {
            if (buf == 0) phase_k0 ^= 1; else phase_k1 ^= 1;
        } else {
            if (buf == 0) phase_k0 ^= 1; else phase_k1 ^= 1;
        }

        mbarrier_wait_fn(current_bar_v, (buf == 0) ? phase_v0 : phase_v1);
        if ((buf == 0) ? phase_v0 : phase_v1) {
            if (buf == 0) phase_v0 ^= 1; else phase_v1 ^= 1;
        } else {
            if (buf == 0) phase_v0 ^= 1; else phase_v1 ^= 1;
        }

        if (threadIdx.x < 64) {
            int row = threadIdx.x;
            float s0 = 0, s1 = 0;
            for (int d = 0; d < 64; ++d) {
                float q0_d = __bfloat162float(load_swizzled<__nv_bfloat16>(smem_Q_0, row, d));
                float q1_d = __bfloat162float(load_swizzled<__nv_bfloat16>(smem_Q_1, row, d));
                
                for (int j = 0; j < 64; ++j) {
                    float k0_d = __bfloat162float(load_swizzled<__nv_bfloat16>(current_K_0, j, d));
                    float k0_1_d = __bfloat162float(load_swizzled<__nv_bfloat16>(current_K_1, j, d));
                    float k1_d = __bfloat162float(load_swizzled<__nv_bfloat16>(current_K_0, j + 64, d));
                    float k1_1_d = __bfloat162float(load_swizzled<__nv_bfloat16>(current_K_1, j + 64, d));
                    
                    s0 += q0_d * k0_d + q1_d * k0_1_d;
                    s1 += q0_d * k1_d + q1_d * k1_1_d;
                }
                store_s(0, d, s0);
                store_s(1, d, s1);
            }
            
            float max_val = -INFINITY;
            for (int c = 0; c < 128; ++c) {
                float val = load_s(row, c) * scale;
                uint32_t global_kv_idx = kv_block + c;
                uint32_t global_row = q_block + cr * 64 + row;
                if (global_row >= S_len || global_kv_idx >= S_len) {
                    val = -INFINITY;
                }
                load_s(row, c) = val; 
                max_val = fmaxf(max_val, val);
            }
            
            float prev_max_val = prev_max[row];
            float curr_max = fmaxf(prev_max_val, max_val);
            float scale_sum = expf(prev_max_val - curr_max);
            sum_exp[row] *= scale_sum;
            
            float sum = 0;
            for (int c = 0; c < 128; ++c) {
                float val = load_s(row, c);
                if (isinf(-val)) {
                    load_s(row, c) = 0;
                } else {
                    float e = expf(val - curr_max);
                    load_s(row, c) = e;
                    sum += e;
                }
            }
            sum_exp[row] += sum;
            
            for (int c = 0; c < 64; ++c) {
                float val = (sum_exp[row] > 0.0f) ? (load_s(row, c) / sum_exp[row]) : 0.0f;
                smem_P[row * 64 + swizzle_128B(row, c)] = __float2bfloat16(val);
            }
        }
        __syncthreads();

        for (int d = 0; d < 64; ++d) {
            float p0 = load_p(row, d);
            float p1 = load_p(row, 64 + d);
            
            float v0_0 = load_swizzled<__nv_bfloat16>(smem_V_0, d, row);
            float v1_0 = load_swizzled<__nv_bfloat16>(smem_V_1, d, row);
            
            smem_S[row * 128 + swizzle_128B(row, d)] += p0 * v0_0;
            smem_S[row * 128 + 64 + swizzle_128B(row, 64 + d)] += p1 * v1_0;
        }
        __syncthreads();
    }

    auto write_O = [&](uint32_t global_row, int row) {
        if (global_row < S_len) {
            for (int half = 0; half < 2; ++half) {
                for (uint32_t col = 0; col < 64; col += 8) {
                    uint32_t global_col = half * 64 + col; 
                    __nv_bfloat164 out_val = *reinterpret_cast<__nv_bfloat164*>(&smem_S[row * 128 + half * 64 + swizzle_128B(row, half * 64 + col)]);
                    *reinterpret_cast<__nv_bfloat164*>(&O[(bh * S_len + global_row) * 128 + global_col]) = out_val;
                }
            }
        }
    };

    for (int i = lane_id; i < 64; i += 32) {
        uint32_t global_row = q_block + cr * 64 + lane_offset + i;
        int row = lane_offset + i;
        write_O(global_row, row);
    }
    
    if (threadIdx.x < 64) {
        int row = threadIdx.x;
        uint32_t global_row = q_block + cr * 64 + row;
        if (global_row < S_len) {
            if (sum_exp[row] > 0.0f) {
                LSE[bh * S_len + global_row] = prev_max[row] + logf(sum_exp[row]);
            } else {
                LSE[bh * S_len + global_row] = -INFINITY;
            }
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    uint32_t B = Q.size(0);
    uint32_t H = Q.size(1);
    uint32_t S_len = Q.size(2);
    uint32_t D = Q.size(3); 
    
    CUtensorMap tma_Q, tma_K, tma_V;
    __nv_bfloat16* q_ptr = static_cast<__nv_bfloat16*>(Q.data_ptr());
    __nv_bfloat16* k_ptr = static_cast<__nv_bfloat16*>(K.data_ptr());
    __nv_bfloat16* v_ptr = static_cast<__nv_bfloat16*>(V.data_ptr());
    
    DRV_CHECK(create_tma_4d_descriptor_2B(&tma_Q, q_ptr, D, S_len, H, B, 64, 64, 1, 1, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    DRV_CHECK(create_tma_4d_descriptor_2B(&tma_K, k_ptr, D, S_len, H, B, 64, 64, 1, 1, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    DRV_CHECK(create_tma_4d_descriptor_2B(&tma_V, v_ptr, D, S_len, H, B, 64, 64, 1, 1, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
    __nv_bfloat16* o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_ptr = static_cast<float*>(LSE.data_ptr());
    
    uint32_t num_blocks_x = (S_len + 127) / 128;
    if (num_blocks_x % 2 != 0) num_blocks_x++;
    uint32_t num_blocks_y = B * H;
    
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(num_blocks_x, num_blocks_y);
    config.blockDim = dim3(128);
    config.dynamicSmemBytes = 128 * 1024; 
    config.stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaFuncSetAttribute(mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 128 * 1024));
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, mha_kernel, tma_Q, tma_K, tma_V, o_ptr, lse_ptr, S_len));
    CUDA_CHECK(cudaGetLastError()); 
    CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id))));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_mha