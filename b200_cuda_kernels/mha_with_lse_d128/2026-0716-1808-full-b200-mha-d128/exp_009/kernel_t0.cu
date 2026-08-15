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
    cuuint64_t globalStrides[3] = {dim0 * 2, dim0 * dim1 * 2, dim0 * dim1 * dim2 * 2}; // bf16 assumes 2 bytes element size
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

__device__ __forceinline__ int swizzle_offset(int row, int col) {
    int chunk = col / 8;
    int in_chunk = col % 8;
    int lane_offset = ((threadIdx.x % 32) ^ (chunk % 4)) * 2;
    int swizzled_col = (chunk ^ ((row % 8))) * 8 + in_chunk + lane_offset;
    return swizzled_col;
}

template<typename T>
struct SwizzledMatrixView {
    T* smem_ptr;
    int stride;
    
    __device__ __forceinline__ T& at(int row, int col) {
        int chunk = col / 8;
        int in_chunk = col % 8;
        int lane_offset = ((threadIdx.x % 32) ^ (chunk % 4)) * 2;
        int swizzled_col = (chunk ^ ((row % 8))) * 8 + in_chunk + lane_offset;
        return smem_ptr[row * stride + swizzled_col];
    }
    
    __device__ __forceinline__ const T& at(int row, int col) const {
        int chunk = col / 8;
        int in_chunk = col % 8;
        int lane_offset = ((threadIdx.x % 32) ^ (chunk % 4)) * 2;
        int swizzled_col = (chunk ^ ((row % 8))) * 8 + in_chunk + lane_offset;
        return smem_ptr[row * stride + swizzled_col];
    }
};

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

__device__ __forceinline__ void named_barrier_sync_fn(int bar_id, int count) {
    asm volatile("barrier.sync.aligned %0, %1;" :: "r"(bar_id), "r"(count));
}

__global__ void mha_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O,
    float* LSE,
    uint32_t S_len)
{
    setmaxnreg_inc_sync_fn<248>();

    uint32_t bn = blockIdx.x;
    uint32_t bh = blockIdx.y;
    uint32_t b = bh / 48;
    uint32_t h = bh % 48;
    
    uint32_t q_block = bn * 128;
    uint32_t k_block = bn * 128 + cluster_rank_fn() * 64;
    
    extern __shared__ char smem[];
    __nv_bfloat16* smem_Q_0 = (__nv_bfloat16*)smem;                  // 16KB
    __nv_bfloat16* smem_Q_1 = (__nv_bfloat16*)(smem + 16384);        // 16KB
    __nv_bfloat16* smem_K_0 = (__nv_bfloat16*)(smem + 32768);        // 8KB
    __nv_bfloat16* smem_K_1 = (__nv_bfloat16*)(smem + 40960);        // 8KB
    __nv_bfloat16* smem_V_0 = (__nv_bfloat16*)(smem + 49152);        // 8KB
    __nv_bfloat16* smem_V_1 = (__nv_bfloat16*)(smem + 57344);        // 8KB
    float* smem_S = (float*)(smem + 65536);                          // 32KB
    __nv_bfloat16* smem_P = (__nv_bfloat16*)(smem + 98304);          // 16KB
    
    uint64_t* bar_q = (uint64_t*)(smem + 114688);
    uint64_t* bar_k = (uint64_t*)(smem + 114696);
    uint64_t* bar_v = (uint64_t*)(smem + 114704);
    uint64_t* bar_out = (uint64_t*)(smem + 114712);
    uint64_t* bar_lse = (uint64_t*)(smem + 114720);

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar_q, 1);
        init_smem_barrier_fn(bar_k, 1);
        init_smem_barrier_fn(bar_v, 1);
        init_smem_barrier_fn(bar_out, 1);
        init_smem_barrier_fn(bar_lse, 1);
    }
    fence_smem_barrier_init_fn();
    cluster_sync_fn();

    SwizzledMatrixView<__nv_bfloat16> Q_0(smem_Q_0, 64);
    SwizzledMatrixView<__nv_bfloat16> Q_1(smem_Q_1, 64);
    SwizzledMatrixView<__nv_bfloat16> K_0(smem_K_0, 64);
    SwizzledMatrixView<__nv_bfloat16> K_1(smem_K_1, 64);
    SwizzledMatrixView<__nv_bfloat16> V_0(smem_V_0, 64);
    SwizzledMatrixView<__nv_bfloat16> V_1(smem_V_1, 64);
    SwizzledMatrixView<__nv_bfloat16> P(smem_P, 64);
    SwizzledMatrixView<__nv_bfloat16> O_view(smem_P, 128); // Reuse P's space temporarily for coalesced output writes

    float running_max_QK[2] = {-INFINITY, -INFINITY};
    float running_sum_exp[2] = {0.0f, 0.0f};
    float scale = 1.0f / sqrtf(128.0f);

    // Zero-initialize accumulated outputs in registers
    float out_0[128] = {0};
    float out_1[128] = {0};

    if (threadIdx.x < 64) { // Loader Warp Group (Warp 0 & 1)
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_q, 32768); 
            tma_load_4d_fn(&tma_Q, bar_q, smem_Q_0, 0, q_block, h, b);
            tma_load_4d_fn(&tma_Q, bar_q, smem_Q_1, 64, q_block, h, b);
        }
    }
    mbarrier_wait_fn(bar_q, 0);

    for (int step = 0; step < 2; ++step) {
        uint32_t phase_k = step % 2;
        if (threadIdx.x < 64) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(bar_k, 16384);
                tma_load_4d_fn(&tma_K, bar_k, smem_K_0, 0, k_block + step * 64, h, b);
                tma_load_4d_fn(&tma_K, bar_k, smem_K_1, 64, k_block + step * 64, h, b);
                
                mbarrier_arrive_and_expect_tx_fn(bar_v, 16384);
                tma_load_4d_fn(&tma_V, bar_v, smem_V_0, 0, k_block + step * 64, h, b);
                tma_load_4d_fn(&tma_V, bar_v, smem_V_1, 64, k_block + step * 64, h, b);
            }
        }
        mbarrier_wait_fn(bar_k, phase_k);
        mbarrier_wait_fn(bar_v, phase_k);

        if (threadIdx.x >= 64 && threadIdx.x < 192) { // Math Warp Group (Warp 2..7)
            int row = threadIdx.x - 64;
            int step_offset = step * 64;
            float local_max[2] = {-INFINITY, -INFINITY};

            for (int half = 0; half < 2; ++half) {
                for (int col = 0; col < 64; ++col) {
                    if (k_block + step_offset + col >= S_len) {
                        smem_S[row * 64 + half * 64 + col] = -INFINITY;
                    } else {
                        float sum = 0;
                        if (half == 0) {
                            for (int d = 0; d < 64; d += 2) {
                                __nv_bfloat162 q = *reinterpret_cast<__nv_bfloat162*>(&Q_0.at(row, d));
                                __nv_bfloat162 k = *reinterpret_cast<__nv_bfloat162*>(&K_0.at(step_offset + col, d));
                                float2 q_f = __bfloat1622float2(q);
                                float2 k_f = __bfloat1622float2(k);
                                sum += q_f.x * k_f.x + q_f.y * k_f.y;
                            }
                        } else {
                            for (int d = 0; d < 64; d += 2) {
                                __nv_bfloat162 q = *reinterpret_cast<__nv_bfloat162*>(&Q_1.at(row, d));
                                __nv_bfloat162 k = *reinterpret_cast<__nv_bfloat162*>(&K_1.at(step_offset + col, d));
                                float2 q_f = __bfloat1622float2(q);
                                float2 k_f = __bfloat1622float2(k);
                                sum += q_f.x * k_f.x + q_f.y * k_f.y;
                            }
                        }
                        smem_S[row * 64 + half * 64 + col] = sum * scale;
                    }
                    local_max[half] = fmaxf(local_max[half], smem_S[row * 64 + half * 64 + col]);
                }
            }

            float curr_max = fmaxf(local_max[0], local_max[1]);
            float prev_max = running_max_QK[step];
            float new_max = fmaxf(prev_max, curr_max);
            float sum_rescale = expf(prev_max - new_max);
            running_sum_exp[step] *= sum_rescale;
            running_max_QK[step] = new_max;

            float local_sum = 0;
            for (int c = 0; c < 64; ++c) {
                float val = expf(smem_S[row * 64 + c] - new_max);
                smem_S[row * 64 + c] = val;
                local_sum += val;
                
                float val1 = expf(smem_S[row * 64 + 64 + c] - new_max);
                smem_S[row * 64 + 64 + c] = val1;
                local_sum += val1;
            }
            running_sum_exp[step] += local_sum;

            // Extract Probabilities safely
            for (int c = 0; c < 64; ++c) {
                P.at(row, c) = __float2bfloat16(smem_S[row * 64 + c] / running_sum_exp[step]);
                P.at(row, c + 64) = __float2bfloat16(smem_S[row * 64 + 64 + c] / running_sum_exp[step]);
            }
        }
        __syncthreads();

        if (threadIdx.x >= 64 && threadIdx.x < 192) { // Math Warp Group
            int row = threadIdx.x - 64;
            for (int k = 0; k < 64; ++k) {
                __nv_bfloat16 p = P.at(row, k);
                if (p != __float2bfloat16(0.0f)) {
                    for (int j = 0; j < 128; j += 2) {
                        __nv_bfloat162 v0 = *reinterpret_cast<__nv_bfloat162*>(&V_0.at(k, j));
                        float2 v0_f = __bfloat1622float2(v0);
                        float2 cur_o0 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&O_view.at(row, j)));
                        *reinterpret_cast<__nv_bfloat162*>(&O_view.at(row, j)) = __float22bfloat162(make_float2(cur_o0.x + __bfloat162float(p) * v0_f.x, cur_o0.y + __bfloat162float(p) * v0_f.y));

                        __nv_bfloat162 v1 = *reinterpret_cast<__nv_bfloat162*>(&V_1.at(k, j));
                        float2 v1_f = __bfloat1622float2(v1);
                        float2 cur_o1 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&O_view.at(row, j + 64)));
                        *reinterpret_cast<__nv_bfloat162*>(&O_view.at(row, j + 64)) = __float22bfloat162(make_float2(cur_o1.x + __bfloat162float(p) * v1_f.x, cur_o1.y + __bfloat162float(p) * v1_f.y));
                    }
                }
            }
        }
        __syncthreads();
    }

    if (threadIdx.x >= 64 && threadIdx.x < 192) { // Math Warp Group
        int row = threadIdx.x - 64;
        if (q_block + row < S_len) {
            // Write Linearly Contiguously for Coalesced Vectorized Epilogue
            for (int j = 0; j < 128; j += 2) {
                *reinterpret_cast<__nv_bfloat162*>(&smem_P[row * 128 + j]) = __float22bfloat162(
                    make_float2(out_0[j], out_0[j+1])
                );
                *reinterpret_cast<__nv_bfloat162*>(&smem_P[row * 128 + j + 64]) = __float22bfloat162(
                    make_float2(out_1[j], out_1[j+1])
                );
            }
        }
    }
    
    named_barrier_sync_fn(1, 128);

    if (threadIdx.x < 128) { // Epilogue Helper (Warp 0..3)
        int row = threadIdx.x;
        if (q_block + row < S_len) {
            *reinterpret_cast<uint4*>(&O[(bh * S_len + q_block + row) * 128]) = *reinterpret_cast<uint4*>(&smem_P[row * 128]);
            *reinterpret_cast<uint4*>(&O[(bh * S_len + q_block + row) * 128 + 32]) = *reinterpret_cast<uint4*>(&smem_P[row * 128 + 32]);
            *reinterpret_cast<uint4*>(&O[(bh * S_len + q_block + row) * 128 + 64]) = *reinterpret_cast<uint4*>(&smem_P[row * 128 + 64]);
            *reinterpret_cast<uint4*>(&O[(bh * S_len + q_block + row) * 128 + 96]) = *reinterpret_cast<uint4*>(&smem_P[row * 128 + 96]);

            if (running_sum_exp[0] > 0.0f) {
                LSE[bh * S_len + q_block + row] = running_max_QK[0] + logf(running_sum_exp[0]);
            } else {
                LSE[bh * S_len + q_block + row] = -INFINITY;
            }
        }
    }
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_out, 1);
        mbarrier_arrive_and_expect_tx_fn(bar_lse, 1);
    }
    mbarrier_wait_fn(bar_out, 0);
    mbarrier_wait_fn(bar_lse, 0);
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
    
    DRV_CHECK(create_tma_4d_descriptor_2B(&tma_Q, q_ptr, D, S_len, H, B, 64, 128, 1, 1, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    DRV_CHECK(create_tma_4d_descriptor_2B(&tma_K, k_ptr, D, S_len, H, B, 64, 64, 1, 1, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    DRV_CHECK(create_tma_4d_descriptor_2B(&tma_V, v_ptr, D, S_len, H, B, 64, 64, 1, 1, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
    __nv_bfloat16* o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_ptr = static_cast<float*>(LSE.data_ptr());
    
    uint32_t num_blocks_x = (S_len + 63) / 64;
    uint32_t num_blocks_y = B * H;
    
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(num_blocks_x, num_blocks_y);
    config.blockDim = dim3(256);
    config.dynamicSmemBytes = 114816; // 112 KB allocated dynamically
    config.stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaFuncSetAttribute(mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 114816));
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, mha_kernel, tma_Q, tma_K, tma_V, o_ptr, lse_ptr, S_len));
    CUDA_CHECK(cudaGetLastError()); 
    CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id))));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_mha