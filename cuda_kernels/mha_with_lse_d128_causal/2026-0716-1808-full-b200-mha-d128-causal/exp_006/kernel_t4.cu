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
#include <cutlass/cutlass.h>
#include <cutlass/wgmma/wgmma.h>

using namespace cutlass;

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_kernel {

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
}

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_dim0, uint64_t gmem_dim1, uint64_t gmem_dim2, uint32_t smem_dim0, uint32_t smem_dim1, uint32_t smem_dim2, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[3] = {gmem_dim0, gmem_dim1, gmem_dim2};
    cuuint64_t globalStrides[2] = {gmem_dim0 * 2, gmem_dim0 * gmem_dim1 * 2};
    cuuint32_t boxDim[3] = {smem_dim0, smem_dim1, smem_dim2};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, globalAddress, globalDim,
        globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle, l2Promotion, oobFill
    );
}

__device__ __forceinline__ float read_smem(__nv_bfloat16* smem, int idx) {
    return __bfloat162float(smem[idx]);
}

__device__ __forceinline__ void write_swizzled_half(__nv_bfloat16* smem, int row, int col, __nv_bfloat16 val) {
    int chunk = col / 8;
    int rem = col % 8;
    int swizzled_chunk = (row % 8) ^ chunk;
    int swizzled_col = swizzled_chunk * 8 + rem;
    smem[row * 64 + swizzled_col] = val;
}

__global__ void mha_with_lse_d128_causal(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O, float* LSE, int32_t S_i) 
{
    setmaxnreg_inc_sync_fn<256>();

    int32_t tid = threadIdx.x;
    int32_t block_idx = blockIdx.x;
    int32_t bh = blockIdx.y;
    
    extern __shared__ __align__(1024) char smem_pool[];
    char* smem = (char*)(((uintptr_t)smem_pool + 1023) & ~1023);
    
    char* smem_Q = smem;
    char* smem_K = smem_Q + 32768;
    char* smem_V = smem_K + 32768;
    char* smem_P = smem_K; // Reuse K's space
    char* smem_temp = smem_V + 32768; 
    
    __shared__ __align__(8) uint64_t bar_Q[1];
    __shared__ __align__(8) uint64_t bar_K[1];
    __shared__ __align__(8) uint64_t bar_V[1];
    
    int32_t warp_id = tid / 32;
    int32_t lane_id = tid % 32;
    int32_t row_group = lane_id / 16; // 0 or 1
    int32_t lane_in_group = lane_id % 16; // 0..15
    
    if (tid == 0) {
        init_smem_barrier_fn(bar_Q, 1);
        init_smem_barrier_fn(bar_K, 1);
        init_smem_barrier_fn(bar_V, 1);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();
    __syncthreads();
    
    int32_t phase_Q = 0;
    int32_t phase_K = 0;
    int32_t phase_V = 0;
    
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_Q, 32768);
        tma_load_3d_fn(&tma_Q, bar_Q, smem_Q, 0, bh * S_i + block_idx * 128, bh);
        tma_load_3d_fn(&tma_Q, bar_Q, smem_Q + 16384, 64, bh * S_i + block_idx * 128, bh);
        
        mbarrier_arrive_and_expect_tx_fn(bar_K, 32768);
        tma_load_3d_fn(&tma_K, bar_K, smem_K, 0, bh * S_i + 0 * 128, bh);
        tma_load_3d_fn(&tma_K, bar_K, smem_K + 16384, 64, bh * S_i + 0 * 128, bh);
        
        mbarrier_arrive_and_expect_tx_fn(bar_V, 32768);
        tma_load_3d_fn(&tma_V, bar_V, smem_V, 0, bh * S_i + 0 * 128, bh);
        tma_load_3d_fn(&tma_V, bar_V, smem_V + 16384, 64, bh * S_i + 0 * 128, bh);
    }
    mbarrier_wait_fn(bar_Q, phase_Q);
    mbarrier_wait_fn(bar_K, phase_K);
    mbarrier_wait_fn(bar_V, phase_V);
    __syncthreads();
    
    float o_acc_flat[128];
    #pragma unroll
    for(int i=0; i<128; i++) o_acc_flat[i] = 0.0f;
    
    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    
    int32_t num_steps = block_idx + 1;
    int32_t total_steps = (S_i + 127) / 128;
    if (num_steps > total_steps) {
        num_steps = total_steps;
    }
    
    int32_t global_i = block_idx * 128 + warp_id * 32 + lane_id;
    
    for (int step = 0; step < num_steps; step++) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_K, 32768);
            tma_load_3d_fn(&tma_K, bar_K, smem_K, 0, bh * S_i + step * 128, bh);
            tma_load_3d_fn(&tma_K, bar_K, smem_K + 16384, 64, bh * S_i + step * 128, bh);
            
            mbarrier_arrive_and_expect_tx_fn(bar_V, 32768);
            tma_load_3d_fn(&tma_V, bar_V, smem_V, 0, bh * S_i + step * 128, bh);
            tma_load_3d_fn(&tma_V, bar_V, smem_V + 16384, 64, bh * S_i + step * 128, bh);
        }
        mbarrier_wait_fn(bar_K, phase_K); phase_K ^= 1;
        mbarrier_wait_fn(bar_V, phase_V); phase_V ^= 1;
        __syncthreads();
        
        // Compute QK^T and store temporarily in swizzled format in Q's space
        for (int kg = 0; kg < 8; kg++) {
            wgmma::fragment<wgmma::accumulator, 16, 16, 16, float> acc_S;
            wgmma::fill_fragment(acc_S, 0.0f);
            
            for (int fb = 0; fb < 8; fb++) {
                wgmma::fragment<wgmma::source_a, 16, 16, 16, __nv_bfloat16, wgmma::col_major> q_frag;
                wgmma::fragment<wgmma::source_b, 16, 16, 16, __nv_bfloat16, wgmma::col_major> k_frag;
                
                int half_q = fb / 4;
                int col_q = (fb % 4) * 16;
                wgmma::load_matrix_sync(q_frag, smem_Q + half_q * 16384 + row_group * 16 * 64 + col_q);
                
                int half_k = fb / 4;
                int col_k = (fb % 4) * 16;
                wgmma::load_matrix_sync(k_frag, smem_K + half_k * 16384 + kg * 16 * 64 + col_k);
                
                wgmma::mma_sync(acc_S, q_frag, k_frag);
            }
            wgmma::commit_async();
            wgmma::wait_async();
            
            // Store in linear layout locally to avoid conflicting swizzle reads later
            wgmma::store_matrix_sync(smem_Q + kg * 256, acc_S);
        }
        
        float s_acc_flat[128];
        for (int kg = 0; kg < 8; kg++) {
            for (int j = 0; j < 16; j++) {
                s_acc_flat[kg * 16 + j] = read_smem((__nv_bfloat16*)(smem_Q + kg * 256), j * 16 + lane_in_group);
            }
        }
        
        float m_curr = -INFINITY;
        for (int j = 0; j < 128; j++) {
            int global_j = step * 128 + j;
            if (global_j > global_i || global_j >= S_i) {
                s_acc_flat[j] = -INFINITY;
            } else {
                s_acc_flat[j] *= 0.08838834764f;
            }
            m_curr = fmaxf(m_curr, s_acc_flat[j]);
        }
        
        bool row_valid = m_curr > -INFINITY;
        float m_new = fmaxf(m_prev, m_curr);
        bool rescale = (m_prev > -INFINITY && row_valid && (m_new > m_prev));
        
        if (rescale) {
            float factor = fast_exp2f_fn((m_prev - m_new) * 1.44269504089f);
            for (int d = 0; d < 128; d++) {
                o_acc_flat[d] *= factor;
            }
            l_prev *= factor;
        }
        if (m_prev == -INFINITY || row_valid) {
            m_prev = m_new;
        }
        
        float l_curr = 0.0f;
        for (int j = 0; j < 128; j++) {
            if (s_acc_flat[j] > -INFINITY) {
                s_acc_flat[j] = fast_exp2f_fn((s_acc_flat[j] - m_prev) * 1.44269504089f);
                l_curr += s_acc_flat[j];
            } else {
                s_acc_flat[j] = 0.0f;
            }
        }
        if (row_valid) {
            l_prev += l_curr;
        }
        
        __syncthreads(); 
        
        for (int j = 0; j < 128; j++) {
            if (j < 64) {
                write_swizzled_half((__nv_bfloat16*)smem_P, (warp_id * 32 + lane_id) * 128 + j, __float2bfloat16(s_acc_flat[j]));
            } else {
                write_swizzled_half((__nv_bfloat16*)(smem_P + 8192), (warp_id * 32 + lane_id) * 128 + j, __float2bfloat16(s_acc_flat[j]));
            }
        }
        __syncthreads(); 
        
        // Compute P * V
        for (int fg = 0; fg < 8; fg++) {
            wgmma::fragment<wgmma::accumulator, 16, 16, 16, float> acc_O;
            wgmma::fill_fragment(acc_O, 0.0f);
            
            for (int kb = 0; kb < 8; kb++) {
                wgmma::fragment<wgmma::source_a, 16, 16, 16, __nv_bfloat16, wgmma::col_major> p_frag;
                wgmma::fragment<wgmma::source_b, 16, 16, 16, __nv_bfloat16, wgmma::col_major> v_frag;
                
                int half_p = kb / 4;
                int col_p = (kb % 4) * 16;
                wgmma::load_matrix_sync(p_frag, smem_P + half_p * 16384 + row_group * 16 * 64 + col_p);
                
                int half_v = fg / 4;
                int col_v = (fg % 4) * 16;
                wgmma::load_matrix_sync(v_frag, smem_V + half_v * 16384 + kb * 16 * 64 + col_v);
                
                wgmma::mma_sync(acc_O, p_frag, v_frag);
            }
            wgmma::commit_async();
            wgmma::wait_async();
            
            // Store to unswizzled temp space
            wgmma::store_matrix_sync(smem_temp, acc_O); 
            
            for (int j = 0; j < 16; j++) {
                o_acc_flat[fg * 16 + j] += read_smem((__nv_bfloat16*)smem_temp, j * 16 + lane_in_group);
            }
        }
        
        __syncthreads(); 
    }
    
    float final_l = l_prev;
    float final_m = m_prev;
    
    if (final_l > 0.0f) {
        float f_final = 1.0f / final_l;
        for (int d = 0; d < 128; d++) {
            o_acc_flat[d] *= f_final;
        }
    } else {
        for (int d = 0; d < 128; d++) {
            o_acc_flat[d] = 0.0f;
        }
    }
    
    if (global_i < S_i) {
        for(int d=0; d<128; d++) {
            O[(uint64_t)bh * S_i * 128 + (uint64_t)global_i * 128 + d] = __float2bfloat16(o_acc_flat[d]);
        }
        LSE[(uint64_t)bh * S_i + (uint64_t)global_i] = final_m + logf(final_l);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B_i = Q.size(0);
    int64_t H_i = Q.size(1);
    int64_t S_i = Q.size(2);
    int64_t D_i = Q.size(3); 
    
    if (D_i != 128) {
        fprintf(stderr, "Expected head dimension 128, got %ld\n", D_i);
        exit(1);
    }
    
    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_ptr = static_cast<float*>(LSE.data_ptr());
    
    CUtensorMap tma_Q, tma_K, tma_V;
    CUresult res_q = create_tma_3d_descriptor_2B(&tma_Q, (void*)Q_ptr, 128, S_i, B_i * H_i, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUresult res_k = create_tma_3d_descriptor_2B(&tma_K, (void*)K_ptr, 128, S_i, B_i * H_i, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUresult res_v = create_tma_3d_descriptor_2B(&tma_V, (void*)V_ptr, 128, S_i, B_i * H_i, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    if (res_q != CUDA_SUCCESS || res_k != CUDA_SUCCESS || res_v != CUDA_SUCCESS) {
        fprintf(stderr, "TMA descriptor creation failed!\n");
        exit(1);
    }
    
    int32_t num_blocks = (S_i + 127) / 128;
    dim3 grid(num_blocks, B_i * H_i);
    dim3 block(128);
    
    // Dynamic Shared Memory Calculation:
    // Q: 32768, K: 32768, V: 32768 -> Total 98304 bytes.
    // Additional temp buffer (512 bytes) + barriers (24 bytes).
    // Plus up to 1023 bytes alignment padding.
    int smem_size = 104000;
    CUDA_CHECK(cudaFuncSetAttribute((const void*)mha_with_lse_d128_causal, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    mha_with_lse_d128_causal<<<grid, block, smem_size, stream>>>(tma_Q, tma_K, tma_V, O_ptr, LSE_ptr, S_i);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_kernel