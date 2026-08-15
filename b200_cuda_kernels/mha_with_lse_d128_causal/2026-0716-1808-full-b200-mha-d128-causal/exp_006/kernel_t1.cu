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

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, globalAddress, globalDim,
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

__global__ void mha_with_lse_d128_causal(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O, float* LSE, int32_t S_i) 
{
    setmaxnreg_inc_sync_fn<256>();

    int32_t block_idx = blockIdx.x;
    int32_t bh = blockIdx.y;
    
    extern __shared__ char smem_base[];
    char* smem = (char*)smem_base;
    char* smem_Q = (char*)(((uintptr_t)smem + 1023) & ~1023);
    char* smem_K = smem_Q + 32768;
    char* smem_V = smem_K + 32768;
    
    __shared__ __align__(8) uint64_t bar_Q[1];
    __shared__ __align__(8) uint64_t bar_K[1];
    __shared__ __align__(8) uint64_t bar_V[1];
    
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
        mbarrier_arrive_and_expect_tx_fn(bar_Q, 65536);
        tma_load_2d_fn(&tma_Q, bar_Q, smem_Q, {0, bh * S_i + block_idx * 128});
        tma_load_2d_fn(&tma_Q, bar_Q, smem_Q + 16384, {64, bh * S_i + block_idx * 128});
        
        mbarrier_arrive_and_expect_tx_fn(bar_K, 32768);
        tma_load_2d_fn(&tma_K, bar_K, smem_K, {0, bh * S_i + 0 * 128});
        tma_load_2d_fn(&tma_K, bar_K, smem_K + 16384, {64, bh * S_i + 0 * 128});
        
        mbarrier_arrive_and_expect_tx_fn(bar_V, 32768);
        tma_load_2d_fn(&tma_V, bar_V, smem_V, {0, bh * S_i + 0 * 128});
        tma_load_2d_fn(&tma_V, bar_V, smem_V + 16384, {64, bh * S_i + 0 * 128});
    }
    mbarrier_wait_fn(bar_Q, phase_Q);
    mbarrier_wait_fn(bar_K, phase_K);
    mbarrier_wait_fn(bar_V, phase_V);
    __syncthreads();
    
    float o_acc_0[64] = {0};
    float o_acc_1[64] = {0};
    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    
    if (block_idx * 128 + tid >= S_i) return;

    int32_t num_steps = block_idx + 1;
    for (int step = 0; step < num_steps; step++) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_K, 32768);
            tma_load_2d_fn(&tma_K, bar_K, smem_K, {0, bh * S_i + step * 128});
            tma_load_2d_fn(&tma_K, bar_K, smem_K + 16384, {64, bh * S_i + step * 128});
            
            mbarrier_arrive_and_expect_tx_fn(bar_V, 32768);
            tma_load_2d_fn(&tma_V, bar_V, smem_V, {0, bh * S_i + step * 128});
            tma_load_2d_fn(&tma_V, bar_V, smem_V + 16384, {64, bh * S_i + step * 128});
        }
        mbarrier_wait_fn(bar_K, phase_K);
        mbarrier_wait_fn(bar_V, phase_V);
        __syncthreads();
        
        float s_acc[128] = {0};
        
        __nv_bfloat16* ptr_Q_0 = (__nv_bfloat16*)smem_Q;
        __nv_bfloat16* ptr_Q_1 = (__nv_bfloat16*)((char*)smem_Q + 16384);
        __nv_bfloat16* ptr_K_0 = (__nv_bfloat16*)smem_K;
        __nv_bfloat16* ptr_K_1 = (__nv_bfloat16*)((char*)smem_K + 16384);
        
        for (int k = 0; k < 64; k += 8) {
            float4 q_f4 = *(float4*)make_swizzled_ptr((char*)ptr_Q_0, tid, k);
            uint16_t* q_h = (uint16_t*)&q_f4;
            for (int j = 0; j < 128; j++) {
                float4 k_f4 = *(float4*)make_swizzled_ptr((char*)ptr_K_0, step * 128 + j, k);
                uint16_t* k_h = (uint16_t*)&k_f4;
                #pragma unroll
                for(int i = 0; i < 8; i++) s_acc[j] += __bfloat162float(q_h[i]) * __bfloat162float(k_h[i]);
            }
        }
        for (int k = 64; k < 128; k += 8) {
            float4 q_f4 = *(float4*)make_swizzled_ptr((char*)ptr_Q_1, tid, k - 64);
            uint16_t* q_h = (uint16_t*)&q_f4;
            for (int j = 0; j < 128; j++) {
                float4 k_f4 = *(float4*)make_swizzled_ptr((char*)ptr_K_1, step * 128 + j, k - 64);
                uint16_t* k_h = (uint16_t*)&k_f4;
                #pragma unroll
                for(int i = 0; i < 8; i++) s_acc[j] += __bfloat162float(q_h[i]) * __bfloat162float(k_h[i]);
            }
        }
        
        int global_i = block_idx * 128 + tid;
        float m_curr = -INFINITY;
        for (int j = 0; j < 128; j++) {
            int global_j = step * 128 + j;
            if (global_j > global_i || global_j >= S_i) {
                s_acc[j] = -INFINITY;
            } else {
                s_acc[j] *= 0.08838834764f;
            }
            m_curr = fmaxf(m_curr, s_acc[j]);
        }
        
        float m_new = fmaxf(m_prev, m_curr);
        if (m_new != m_prev && m_prev > -INFINITY) {
            float factor = fast_exp2f_fn((m_prev - m_new) * 1.44269504089f);
            for(int d=0; d<64; d+=8) {
                float4 fa0 = *(float4*)&o_acc_0[d];
                uint32_t* fa0_u32 = (uint32_t*)&fa0;
                for(int i=0; i<4; i++) fa0_u32[i] = __float_as_uint(__uint_as_float(fa0_u32[i]) * factor);
                *(float4*)&o_acc_0[d] = fa0;
                
                float4 fa1 = *(float4*)&o_acc_1[d];
                uint32_t* fa1_u32 = (uint32_t*)&fa1;
                for(int i=0; i<4; i++) fa1_u32[i] = __float_as_uint(__uint_as_float(fa1_u32[i]) * factor);
                *(float4*)&o_acc_1[d] = fa1;
            }
            l_prev *= factor;
            m_prev = m_new;
        } else if (m_prev == -INFINITY) {
            m_prev = m_new;
        }
        
        float l_curr = 0.0f;
        for (int j = 0; j < 128; j++) {
            float p = 0.0f;
            if (s_acc[j] > -INFINITY) {
                p = fast_exp2f_fn((s_acc[j] - m_prev) * 1.44269504089f);
            }
            l_curr += p;
            s_acc[j] = p; 
        }
        l_prev += l_curr;
        
        float* p_acc = s_acc; 
        
        __nv_bfloat16* ptr_P_0 = (__nv_bfloat16*)ptr_K_0;
        __nv_bfloat16* ptr_P_1 = (__nv_bfloat16*)ptr_K_1;
        for (int j = 0; j < 128; j+=4) {
            float4 p_f4;
            uint32_t* p_u32 = (uint32_t*)&p_f4;
            p_u32[0] = __float_as_uint(p_acc[j]);
            p_u32[1] = __float_as_uint(p_acc[j+1]);
            p_u32[2] = __float_as_uint(p_acc[j+2]);
            p_u32[3] = __float_as_uint(p_acc[j+3]);
            
            *(float4*)make_swizzled_ptr((char*)ptr_P_0, step * 128 + j, 0) = *(float4*)&p_f4;
            *(float4*)make_swizzled_ptr((char*)ptr_P_1, step * 128 + j, 0) = *(float4*)&p_f4;
        }
        
        __nv_bfloat16* ptr_V_0 = (__nv_bfloat16*)smem_V;
        __nv_bfloat16* ptr_V_1 = (__nv_bfloat16*)((char*)smem_V + 16384);
        
        for (int j = 0; j < 128; j++) {
            float p = p_acc[j];
            if (p == 0.0f) continue;
            
            for (int d = 0; d < 64; d += 8) {
                float4 p_f4 = *(float4*)make_swizzled_ptr((char*)ptr_P_0, step * 128 + j, d);
                uint32_t* p_u32 = (uint32_t*)&p_f4;
                p_u32[0] = __float_as_uint(__uint_as_float(p_u32[0]) * p);
                p_u32[1] = __float_as_uint(__uint_as_float(p_u32[1]) * p);
                p_u32[2] = __float_as_uint(__uint_as_float(p_u32[2]) * p);
                p_u32[3] = __float_as_uint(__uint_as_float(p_u32[3]) * p);
                
                float4 v_f4 = *(float4*)make_swizzled_ptr((char*)ptr_V_0, step * 128 + j, d);
                uint16_t* p_h = (uint16_t*)&p_f4;
                uint16_t* v_h = (uint16_t*)&v_f4;
                for(int i=0; i<8; i++) o_acc_0[d+i] += __bfloat162float(p_h[i]) * __bfloat162float(v_h[i]);
            }
            for (int d = 64; d < 128; d += 8) {
                float4 p_f4 = *(float4*)make_swizzled_ptr((char*)ptr_P_1, step * 128 + j, d - 64);
                uint32_t* p_u32 = (uint32_t*)&p_f4;
                p_u32[0] = __float_as_uint(__uint_as_float(p_u32[0]) * p);
                p_u32[1] = __float_as_uint(__uint_as_float(p_u32[1]) * p);
                p_u32[2] = __float_as_uint(__uint_as_float(p_u32[2]) * p);
                p_u32[3] = __float_as_uint(__uint_as_float(p_u32[3]) * p);
                
                float4 v_f4 = *(float4*)make_swizzled_ptr((char*)ptr_V_1, step * 128 + j, d - 64);
                uint16_t* p_h = (uint16_t*)&p_f4;
                uint16_t* v_h = (uint16_t*)&v_f4;
                for(int i=0; i<8; i++) o_acc_1[d - 64 + i] += __bfloat162float(p_h[i]) * __bfloat162float(v_h[i]);
            }
        }
        
        __syncthreads(); 
        phase_K ^= 1;
        phase_V ^= 1;
    }
    
    float final_l = l_prev;
    float final_m = m_prev;
    
    if (final_l > 0.0f) {
        float f_final = 1.0f / final_l;
        for(int d=0; d<64; d+=8) {
            float4 fa0 = *(float4*)&o_acc_0[d];
            uint32_t* fa0_u32 = (uint32_t*)&fa0;
            for(int i=0; i<4; i++) fa0_u32[i] = __float_as_uint(__uint_as_float(fa0_u32[i]) * f_final);
            *(float4*)&o_acc_0[d] = fa0;
            
            float4 fa1 = *(float4*)&o_acc_1[d];
            uint32_t* fa1_u32 = (uint32_t*)&fa1;
            for(int i=0; i<4; i++) fa1_u32[i] = __float_as_uint(__uint_as_float(fa1_u32[i]) * f_final);
            *(float4*)&o_acc_1[d] = fa1;
        }
    } else {
        for(int d=0; d<64; d++) {
            o_acc_0[d] = 0.0f;
            o_acc_1[d] = 0.0f;
        }
    }
    
    int32_t global_i = block_idx * 128 + tid;
    if (global_i < S_i) {
        for(int d=0; d<64; d+=8) {
            float4 out0 = *(float4*)&o_acc_0[d];
            uint16_t* out0_h = (uint16_t*)&out0;
            for(int i=0; i<8; i++) out0_h[i] = __float2bfloat16(__uint_as_float(out0_h[i]));
            *(float4*)&O[(uint64_t)bh * S_i * 128 + (uint64_t)global_i * 128 + d] = out0;
            
            float4 out1 = *(float4*)&o_acc_1[d];
            uint16_t* out1_h = (uint16_t*)&out1;
            for(int i=0; i<8; i++) out1_h[i] = __float2bfloat16(__uint_as_float(out1_h[i]));
            *(float4*)&O[(uint64_t)bh * S_i * 128 + (uint64_t)global_i * 128 + d + 64] = out1;
        }
        
        lse_ptr[(uint64_t)bh * S_i + (block_idx * 128 + tid)] = final_m + logf(final_l);
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
    CUresult res_q = create_tma_2d_descriptor_2B(&tma_Q, (void*)Q_ptr, 128, B_i * H_i * S_i, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUresult res_k = create_tma_2d_descriptor_2B(&tma_K, (void*)K_ptr, 128, B_i * H_i * S_i, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUresult res_v = create_tma_2d_descriptor_2B(&tma_V, (void*)V_ptr, 128, B_i * H_i * S_i, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    if (res_q != CUDA_SUCCESS || res_k != CUDA_SUCCESS || res_v != CUDA_SUCCESS) {
        fprintf(stderr, "TMA descriptor creation failed!\n");
        exit(1);
    }
    
    int32_t num_blocks = (S_i + 127) / 128;
    dim3 grid(num_blocks, B_i * H_i);
    dim3 block(128);
    
    int smem_size = 100000;
    CUDA_CHECK(cudaFuncSetAttribute((const void*)mha_with_lse_d128_causal, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    mha_with_lse_d128_causal<<<grid, block, smem_size, stream>>>(tma_Q, tma_K, tma_V, O_ptr, LSE_ptr, S_i);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_kernel