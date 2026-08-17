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
#include <mma.h>

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

// ---------------- PTX Wrappers ----------------

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

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
}

__device__ __forceinline__ cudaMatrixDesc make_matrix_desc(const void* ptr, int rows, int cols, bool transpose = false) {
    cudaMatrixDesc desc;
    desc.ptr = ptr;
    desc.mode = transpose ? cudaMatrixTranspose : cudaMatrixRowMajor;
    desc.m = rows;
    desc.n = cols;
    desc.ldd = cols; // For a standard row-major 2D array, leading dimension is cols
    return desc;
}

__device__ __forceinline__ cudaSwizzleDesc make_swizzle_desc(cudaMemSwizzleMode mode, const void* ptr) {
    cudaSwizzleDesc desc;
    desc.mode = mode;
    desc.ldd = static_cast<int64_t>(ptr);
    return desc;
}

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t dim0, uint64_t dim1, uint64_t dim2, uint32_t box0, uint32_t box1, CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[3] = {dim0, dim1, dim2};
    cuuint64_t globalStrides[2] = {dim0 * 2, dim0 * dim1 * 2};
    cuuint32_t boxDim[3] = {box0, box1, 1};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

// ---------------- Kernel ----------------

struct SwizzledSmem {
    alignas(1024) uint16_t Q[128 * 128];
    alignas(1024) uint16_t K[128 * 128];
    alignas(1024) uint16_t V[128 * 128];
    alignas(1024) uint16_t P[128 * 128];
    alignas(1024) float S[128 * 128]; 
};

__global__ void flash_attention_kernel(
    __nv_bfloat16* O, float* LSE,
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    int32_t S, int32_t D, int32_t B_H)
{
    extern __shared__ __align__(128) uint8_t smem_buf[];
    SwizzledSmem* smem = reinterpret_cast<SwizzledSmem*>(smem_buf);

    extern __shared__ __align__(8) uint8_t smem_barriers[];
    uint64_t* bar_Q = reinterpret_cast<uint64_t*>(smem_barriers);
    uint64_t* bar_K = reinterpret_cast<uint64_t*>(smem_barriers + 8);
    uint64_t* bar_V = reinterpret_cast<uint64_t*>(smem_barriers + 16);

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar_Q, 1);
        init_smem_barrier_fn(bar_K, 1);
        init_smem_barrier_fn(bar_V, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    auto acc_S = warp_alloc::alloc<128 * 128>();
    auto acc_O = warp_alloc::alloc<128 * 128>();

    cudaAccDescriptor acc_S, acc_O;
    wgmma.create_acc_descriptor(&acc_S, acc_S.get(), 128, 128, 0);
    wgmma.create_acc_descriptor(&acc_O, acc_O.get(), 128, 128, 1);

    int32_t blk_q = blockIdx.x;
    int32_t row_start_q = blk_q * 128;
    int32_t bh_idx = blockIdx.y;

    if (row_start_q >= S) return;

    uint32_t phase_Q = 0;
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_Q, 128 * 128 * 2);
        tma_load_3d_fn(&tma_Q, bar_Q, smem->Q, 0, row_start_q, bh_idx);
        tma_load_3d_fn(&tma_Q, bar_Q, smem->Q + 8192, 64, row_start_q, bh_idx);
    }
    mbarrier_wait_fn(bar_Q, phase_Q);

    float global_sum[128];
    for(int i = 0; i < 128; ++i) {
        global_sum[i] = 0.0f;
    }

    cudaMatrixDesc d_Q_0 = make_matrix_desc(smem->Q, 128, 64, false);
    cudaMatrixDesc d_Q_1 = make_matrix_desc(smem->Q, 128, 64, false);
    cudaMatrixDesc d_K_0 = make_matrix_desc(smem->K, 128, 64, true);
    cudaMatrixDesc d_K_1 = make_matrix_desc(smem->K, 128, 64, true);
    cudaMatrixDesc d_V_0 = make_matrix_desc(smem->V, 128, 64, false);
    cudaMatrixDesc d_V_1 = make_matrix_desc(smem->V, 128, 64, false);
    cudaMatrixDesc d_P = make_matrix_desc(smem->P, 128, 128, false);
    
    cudaSwizzleDesc s_Q = make_swizzle_desc(cudaMemSwizzleMode128B, smem->Q);
    cudaSwizzleDesc s_K = make_swizzle_desc(cudaMemSwizzleMode128B, smem->K);
    cudaSwizzleDesc s_V = make_swizzle_desc(cudaMemSwizzleMode128B, smem->V);
    cudaSwizzleDesc s_P = make_swizzle_desc(cudaMemSwizzleMode128B, smem->P);

    float softmax_s_old[128];
    for(int i = 0; i < 128; ++i) softmax_s_old[i] = -1e38f;

    uint32_t phase_K = 0, phase_V = 0;

    for (int blk_kv = 0; blk_kv <= blk_q; ++blk_kv) {
        int row_start_kv = blk_kv * 128;
        if (row_start_kv > row_start_q) break;

        __syncthreads();

        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_K, 128 * 128 * 2);
            mbarrier_arrive_and_expect_tx_fn(bar_V, 128 * 128 * 2);
            tma_load_3d_fn(&tma_K, bar_K, smem->K, 0, row_start_kv, bh_idx);
            tma_load_3d_fn(&tma_K, bar_K, smem->K + 8192, 64, row_start_kv, bh_idx);
            
            tma_load_3d_fn(&tma_V, bar_V, smem->V, 0, row_start_kv, bh_idx);
            tma_load_3d_fn(&tma_V, bar_V, smem->V + 8192, 64, row_start_kv, bh_idx);
        }
        mbarrier_wait_fn(bar_K, phase_K);
        mbarrier_wait_fn(bar_V, phase_V);
        phase_K ^= 1;
        phase_V ^= 1;

        __syncthreads();

        wgmma.create_matrix_descriptor(&d_Q_0, smem->Q, 128, 128, false);
        wgmma.create_matrix_descriptor(&d_K_0, smem->K, 128, 128, true);

        for (int k_step = 0; k_step < 128; k_step += 64) {
            wgmma.mma_async<128, 128, 128, 64>(d_S, d_Q_0, d_K_0, acc_S, 128, 128, 128, 64, 0, false);
            wgmma.mma_async<128, 128, 128, 64>(d_S, d_Q_1, d_K_1, acc_S, 128, 128, 128, 64, 0, true);
        }
        
        wgmma.commit_accumulator(acc_S);
        wgmma.wait_commit();

        int row = threadIdx.x;
        float m_val = -1e38f;
        int max_col = (row_start_q + row >= row_start_kv + 127) ? 127 : (row_start_q + row - row_start_kv);

        for(int col = 0; col < 128; ++col) {
            float s_val = acc_S[row * 128 + col] * 0.08838834764f; 
            if (col > max_col) s_val = -1e38f;
            m_val = max(m_val, s_val);
        }
        
        float prev_max = softmax_s_old[row];
        float curr_max = max(prev_max, m_val);
        if (curr_max == -1e38f) curr_max = 0;

        float sum_val = 0;
        float curr_sum = 0;
        for(int col = 0; col < 128; ++col) {
            float s_val = acc_S[row * 128 + col] * 0.08838834764f;
            if (col > max_col) s_val = -1e38f;
            float p_val = 0;
            if (s_val > -1e37f) {
                p_val = __float_expf(s_val - curr_max);
            }
            curr_sum += p_val;
            
            float scale_curr = __float_expf(curr_max - curr_max); // effectively 1
            float scale_prev = __float_expf(prev_max - curr_max);
            
            float final_p = p_val * scale_curr;
            float s_out = acc_S[row * 128 + col] * scale_prev; 
            
            smem->S[row * 128 + col] = s_out; 
            acc_S[row * 128 + col] = final_p; 
        }

        float safe_prev_max = (prev_max == -1e38f) ? 0 : prev_max;
        float safe_curr_max = (curr_max == -1e38f) ? 0 : curr_max;
        float curr_sum_scaled = global_sum[row] * __float_expf(safe_prev_max - safe_curr_max) + curr_sum * __float_expf(curr_max - safe_curr_max);
        global_sum[row] = curr_sum_scaled;
        softmax_s_old[row] = curr_max;

        __syncthreads();

        // Emit P structurally identical to Q layout (swizzle matching TMA encoding layout)
        for (int col = 0; col < 128; ++col) {
            float p_val = acc_S[row * 128 + col];
            uint16_t p_bf16 = __float2bfloat16(p_val);
            
            int half = col / 64;
            int c = col % 64;
            int x = c / 8;
            int rem = c % 8;
            int swizzled_x = (row % 8) ^ x;
            int swizzled_c = swizzled_x * 8 + rem;
            
            (smem->P + half * 8192)[row * 64 + swizzled_c] = p_bf16;
        }
        
        // Hard-sync threads within block to ensure P is fully resident before issuing WGMMA
        named_barrier_sync_fn(1, 128); 

        wgmma.create_matrix_descriptor(&d_P, smem->P, 128, 128, false);
        wgmma.create_matrix_descriptor(&d_V_0, smem->V, 128, 128, false);

        for (int k_step = 0; k_step < 128; k_step += 64) {
            wgmma.mma_async<128, 128, 128, 64>(d_O, d_P, d_V_0, acc_O, 128, 128, 128, 64, 0, true);
            wgmma.mma_async<128, 128, 128, 64>(d_O, d_P, d_V_1, acc_O, 128, 128, 128, 64, 0, true);
        }
        
        wgmma.commit_accumulator(acc_O);
        wgmma.wait_commit();
        
        __syncthreads();
    }

    // Normalization Step
    for(int i = 0; i < 128; ++i) {
        float o_val = acc_O[row * 128 + i];
        o_val /= global_sum[row];
        
        int half = i / 64;
        int c = i % 64;
        int x = c / 8;
        int rem = c % 8;
        int swizzled_x = (row % 8) ^ x;
        int swizzled_c = swizzled_x * 8 + rem;
        
        *(reinterpret_cast<uint16_t*>(&((smem->O + half * 8192)[row * 64 + swizzled_c]))) = __float2bfloat16(o_val);
    }
    __syncthreads();

    // Coalesced Vectorized Output Store directly from generic proxy memory
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) {
        int row = i / 8;
        int col_vec = (i % 8) * 8;
        int global_row = row_start_q + row;
        int global_col = col_vec;
        if (global_row < S && global_col + 7 < 128) {
            uint4 val = *reinterpret_cast<const uint4*>(&smem->O[row * 128 + global_col]);
            *reinterpret_cast<uint4*>(&O[bh_idx * S * 128 + global_row * 128 + global_col]) = val;
        }
    }

    if (threadIdx.x < 128) {
        int global_row = row_start_q + threadIdx.x;
        if (global_row < S) {
            LSE[bh_idx * S + global_row] = softmax_s_old[threadIdx.x] + __logf(global_sum[threadIdx.x]);
        }
    }
}

namespace tvm_ffi_example_cuda {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);

    if (S == 0) return;

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    __nv_bfloat16* Q_g = static_cast<__nv_bfloat16*>(Q.data_ptr());
    __nv_bfloat16* K_g = static_cast<__nv_bfloat16*>(K.data_ptr());
    __nv_bfloat16* V_g = static_cast<__nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_g = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_g = static_cast<float*>(LSE.data_ptr());

    CUtensorMap tma_Q, tma_K, tma_V;
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_Q, Q_g, 128, S, B * H, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_K, K_g, 128, S, B * H, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_V, V_g, 128, S, B * H, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B));

    dim3 grid((S + 127) / 128, B * H);
    dim3 block(128, 1, 1);

    uint32_t smem_size = sizeof(SwizzledSmem) + 1024 + 32; 
    
    CUDA_CHECK(cudaFuncSetAttribute(flash_attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    flash_attention_kernel<<<grid, block, smem_size, stream>>>(
        O_g, LSE_g, tma_Q, tma_K, tma_V, S, D, B * H);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}  // namespace tvm_ffi_example_cuda