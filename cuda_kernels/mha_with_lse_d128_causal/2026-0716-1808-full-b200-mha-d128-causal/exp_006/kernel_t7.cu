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

namespace tvm_ffi_kernel {

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

__device__ __forceinline__ __nv_bfloat16* make_swizzled_ptr(char* base, int row, int col) {
    int x = col / 8;
    int rem = col % 8;
    int swizzled_x = (row % 8) ^ x;
    int swizzled_col = swizzled_x * 8 + rem;
    return (__nv_bfloat16*)(base + row * 64 * sizeof(__nv_bfloat16) + swizzled_col * sizeof(__nv_bfloat16));
}

__device__ __forceinline__ __nv_bfloat16* get_swizzled_ptr(__nv_bfloat16* smem_base, int row, int col) {
    char* base = (char*)smem_base;
    int tile = col / 64;
    int tile_col = col % 64;
    if (tile == 1) base += 16384;
    return make_swizzled_ptr(base, row, tile_col);
}

__global__ void mha_with_lse_d128_causal(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O, float* LSE, int32_t S_i) 
{
    int32_t tid = threadIdx.x;
    int32_t block_idx = blockIdx.x;
    int32_t bh = blockIdx.y;
    
    extern __shared__ __align__(1024) char smem_pool[];
    char* smem = (char*)(((uintptr_t)smem_pool + 1023) & ~1023);
    
    char* smem_Q = smem;
    char* smem_K = smem_Q + 32768;
    char* smem_V = smem_K + 32768;
    char* smem_P = smem_K; // Reuse K's space post-QKT
    
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
        if (step < num_steps - 1) {
            if (tid == 0) {
                int next_step = step + 1;
                mbarrier_arrive_and_expect_tx_fn(bar_K, 32768);
                tma_load_3d_fn(&tma_K, bar_K, smem_K, 0, bh * S_i + next_step * 128, bh);
                tma_load_3d_fn(&tma_K, bar_K, smem_K + 16384, 64, bh * S_i + next_step * 128, bh);
                
                mbarrier_arrive_and_expect_tx_fn(bar_V, 32768);
                tma_load_3d_fn(&tma_V, bar_V, smem_V, 0, bh * S_i + next_step * 128, bh);
                tma_load_3d_fn(&tma_V, bar_V, smem_V + 16384, 64, bh * S_i + next_step * 128, bh);
            }
        }
        
        mbarrier_wait_fn(bar_K, phase_K); phase_K ^= 1;
        mbarrier_wait_fn(bar_V, phase_V); phase_V ^= 1;
        __syncthreads();
        
        float s_acc_flat[128];
        #pragma unroll
        for(int i=0; i<128; i++) s_acc_flat[i] = 0.0f;
        
        for (int k = 0; k < 64; k += 8) {
            float4 q_f4 = *(float4*)get_swizzled_ptr((__nv_bfloat16*)smem_Q, tid, k);
            uint16_t* q_h = (uint16_t*)&q_f4;
            for (int j = 0; j < 128; j++) {
                float4 k_f4 = *(float4*)get_swizzled_ptr((__nv_bfloat16*)smem_K, j, k);
                uint16_t* k_h = (uint16_t*)&k_f4;
                #pragma unroll
                for(int i=0; i<8; i++) s_acc_flat[j] += __bfloat162float(q_h[i]) * __bfloat162float(k_h[i]);
            }
        }
        for (int k = 64; k < 128; k += 8) {
            float4 q_f4 = *(float4*)get_swizzled_ptr((__nv_bfloat16*)smem_Q, tid, k);
            uint16_t* q_h = (uint16_t*)&q_f4;
            for (int j = 0; j < 128; j++) {
                float4 k_f4 = *(float4*)get_swizzled_ptr((__nv_bfloat16*)smem_K, j, k);
                uint16_t* k_h = (uint16_t*)&k_f4;
                #pragma unroll
                for(int i=0; i<8; i++) s_acc_flat[j] += __bfloat162float(q_h[i]) * __bfloat162float(k_h[i]);
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
            m_prev = m_new;
        } else if (m_prev == -INFINITY) {
            m_prev = m_curr;
        } else {
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
        
        for (int j = 0; j < 128; j+=4) {
            float4 p_f4;
            uint32_t* p_u32 = (uint32_t*)&p_f4;
            p_u32[0] = __float_as_uint(s_acc_flat[j]);
            p_u32[1] = __float_as_uint(s_acc_flat[j+1]);
            p_u32[2] = __float_as_uint(s_acc_flat[j+2]);
            p_u32[3] = __float_as_uint(s_acc_flat[j+3]);
            
            *(float4*)get_swizzled_ptr((__nv_bfloat16*)smem_P, tid, j) = p_f4;
        }
        __syncthreads(); 
        
        for (int j = 0; j < 128; j++) {
            float4 p_f4_0 = *(float4*)get_swizzled_ptr((__nv_bfloat16*)smem_P, tid, j);
            uint32_t* p_u32_0 = (uint32_t*)&p_f4_0;
            
            if (p_u32_0[0] == 0 && p_u32_0[1] == 0 && p_u32_0[2] == 0 && p_u32_0[3] == 0) {
                continue;
            }
            
            float p_0 = __uint_as_float(p_u32_0[0]);
            float p_1 = __uint_as_float(p_u32_0[1]);
            float p_2 = __uint_as_float(p_u32_0[2]);
            float p_3 = __uint_as_float(p_u32_0[3]);
            
            for (int d = 0; d < 64; d += 8) {
                float4 v_f4 = *(float4*)get_swizzled_ptr((__nv_bfloat16*)smem_V, j, d);
                uint16_t* v_h = (uint16_t*)&v_f4;
                #pragma unroll
                for(int i=0; i<8; i++) o_acc_flat[d+i] += p_0 * __bfloat162float(v_h[i]);
                
                float4 v_f4_1 = *(float4*)get_swizzled_ptr((__nv_bfloat16*)smem_V, j+1, d);
                uint16_t* v_h_1 = (uint16_t*)&v_f4_1;
                #pragma unroll
                for(int i=0; i<8; i++) o_acc_flat[d+i] += p_1 * __bfloat162float(v_h_1[i]);
                
                float4 v_f4_2 = *(float4*)get_swizzled_ptr((__nv_bfloat16*)smem_V, j+2, d);
                uint16_t* v_h_2 = (uint16_t*)&v_f4_2;
                #pragma unroll
                for(int i=0; i<8; i++) o_acc_flat[d+i] += p_2 * __bfloat162float(v_h_2[i]);
                
                float4 v_f4_3 = *(float4*)get_swizzled_ptr((__nv_bfloat16*)smem_V, j+3, d);
                uint16_t* v_h_3 = (uint16_t*)&v_f4_3;
                #pragma unroll
                for(int i=0; i<8; i++) o_acc_flat[d+i] += p_3 * __bfloat162float(v_h_3[i]);
            }
            
            for (int d = 64; d < 128; d += 8) {
                float4 v_f4 = *(float4*)get_swizzled_ptr((__nv_bfloat16*)smem_V, j, d);
                uint16_t* v_h = (uint16_t*)&v_f4;
                #pragma unroll
                for(int i=0; i<8; i++) o_acc_flat[d+i] += p_0 * __bfloat162float(v_h[i]);
                
                float4 v_f4_1 = *(float4*)get_swizzled_ptr((__nv_bfloat16*)smem_V, j+1, d);
                uint16_t* v_h_1 = (uint16_t*)&v_f4_1;
                #pragma unroll
                for(int i=0; i<8; i++) o_acc_flat[d+i] += p_1 * __bfloat162float(v_h_1[i]);
                
                float4 v_f4_2 = *(float4*)get_swizzled_ptr((__nv_bfloat16*)smem_V, j+2, d);
                uint16_t* v_h_2 = (uint16_t*)&v_f4_2;
                #pragma unroll
                for(int i=0; i<8; i++) o_acc_flat[d+i] += p_2 * __bfloat162float(v_h_2[i]);
                
                float4 v_f4_3 = *(float4*)get_swizzled_ptr((__nv_bfloat16*)smem_V, j+3, d);
                uint16_t* v_h_3 = (uint16_t*)&v_f4_3;
                #pragma unroll
                for(int i=0; i<8; i++) o_acc_flat[d+i] += p_3 * __bfloat162float(v_h_3[i]);
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
    
    int smem_size = 104000;
    CUDA_CHECK(cudaFuncSetAttribute((const void*)mha_with_lse_d128_causal, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    mha_with_lse_d128_causal<<<grid, block, smem_size, stream>>>(tma_Q, tma_K, tma_V, O_ptr, LSE_ptr, S_i);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_kernel