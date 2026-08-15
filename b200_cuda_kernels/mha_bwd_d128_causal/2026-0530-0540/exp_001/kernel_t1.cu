#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
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

namespace tvm_ffi_mha_bwd {

__global__ void compute_dQ_kernel(
    const __nv_bfloat16* __restrict__ Q, 
    const __nv_bfloat16* __restrict__ K, 
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O, 
    const __nv_bfloat16* __restrict__ dO, 
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ, 
    int S, int d, float scale
) {
    int bh = blockIdx.y;
    int q_block_idx = blockIdx.x;
    int tx = threadIdx.x;
    
    int base_idx = bh * S * 128;
    int q_idx = q_block_idx * 128 + tx;
    
    extern __shared__ char smem[];
    // Pad s_dO to 130 elements to avoid bank conflicts on individual thread writes
    __nv_bfloat16 (*s_dO)[130] = reinterpret_cast<__nv_bfloat16 (*)[130]>(smem);
    __nv_bfloat16 (*s_K)[128] = reinterpret_cast<__nv_bfloat16 (*)[128]>(smem + 33280);
    __nv_bfloat16 (*s_V)[128] = reinterpret_cast<__nv_bfloat16 (*)[128]>(smem + 49664);
    
    uint32_t reg_Q[64];
    float dQ_acc[128] = {0.0f};
    float D_i = 0.0f;
    float l_i = 0.0f;
    
    if (q_idx < S) {
        const uint32_t* g_Q_row = reinterpret_cast<const uint32_t*>(Q + base_idx + q_idx * 128);
        const uint32_t* g_O_row = reinterpret_cast<const uint32_t*>(O + base_idx + q_idx * 128);
        const uint32_t* g_dO_row = reinterpret_cast<const uint32_t*>(dO + base_idx + q_idx * 128);
        
        for (int c = 0; c < 64; ++c) {
            reg_Q[c] = g_Q_row[c];
            uint32_t do_val = g_dO_row[c];
            *reinterpret_cast<uint32_t*>(&s_dO[tx][c*2]) = do_val;
            
            __nv_bfloat162 o_v = *reinterpret_cast<const __nv_bfloat162*>(&g_O_row[c]);
            __nv_bfloat162 do_v = *reinterpret_cast<const __nv_bfloat162*>(&do_val);
            float2 o_f = __bfloat1622float2(o_v);
            float2 do_f = __bfloat1622float2(do_v);
            D_i += o_f.x * do_f.x + o_f.y * do_f.y;
        }
        l_i = L[bh * S + q_idx];
    }
    
    int block_max_q = (q_block_idx * 128 + 127 < S) ? (q_block_idx * 128 + 127) : (S - 1);
    int block_max_j_block = block_max_q / 64;
    
    for (int j_block = 0; j_block <= block_max_j_block; ++j_block) {
        __syncthreads();
        
        int kv_float4 = (64 * 128) / 8;
        const __nv_bfloat16* g_K = K + base_idx + j_block * 64 * 128;
        const __nv_bfloat16* g_V = V + base_idx + j_block * 64 * 128;
        
        for (int i = tx; i < kv_float4; i += 128) {
            int row = i / 16;
            if (j_block * 64 + row < S) {
                reinterpret_cast<float4*>(s_K)[i] = reinterpret_cast<const float4*>(g_K)[i];
                reinterpret_cast<float4*>(s_V)[i] = reinterpret_cast<const float4*>(g_V)[i];
            } else {
                reinterpret_cast<float4*>(s_K)[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                reinterpret_cast<float4*>(s_V)[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        
        __syncthreads();
        
        if (q_idx >= S) continue;
        
        for (int j = 0; j < 64; ++j) {
            int kv_idx = j_block * 64 + j;
            if (kv_idx > q_idx || kv_idx >= S) continue;
            
            float s_ij = 0.0f;
            float ds_ij = 0.0f;
            
            #pragma unroll 16
            for (int c = 0; c < 64; ++c) {
                __nv_bfloat162 q_v = *reinterpret_cast<const __nv_bfloat162*>(&reg_Q[c]);
                __nv_bfloat162 k_v = *reinterpret_cast<const __nv_bfloat162*>(&s_K[j][c*2]);
                float2 q_f = __bfloat1622float2(q_v);
                float2 k_f = __bfloat1622float2(k_v);
                s_ij += q_f.x * k_f.x + q_f.y * k_f.y;
                
                __nv_bfloat162 do_v = *reinterpret_cast<const __nv_bfloat162*>(&s_dO[tx][c*2]);
                __nv_bfloat162 v_v = *reinterpret_cast<const __nv_bfloat162*>(&s_V[j][c*2]);
                float2 do_f = __bfloat1622float2(do_v);
                float2 v_f = __bfloat1622float2(v_v);
                ds_ij += do_f.x * v_f.x + do_f.y * v_f.y;
            }
            
            s_ij *= scale;
            float p_ij = expf(s_ij - l_i);
            float dp_ij = p_ij * (ds_ij - D_i);
            
            #pragma unroll 16
            for (int c = 0; c < 64; ++c) {
                __nv_bfloat162 k_v = *reinterpret_cast<const __nv_bfloat162*>(&s_K[j][c*2]);
                float2 k_f = __bfloat1622float2(k_v);
                dQ_acc[c*2] += dp_ij * k_f.x;
                dQ_acc[c*2+1] += dp_ij * k_f.y;
            }
        }
    }
    
    if (q_idx < S) {
        uint32_t* g_dQ_row = reinterpret_cast<uint32_t*>(dQ + base_idx + q_idx * 128);
        for (int c = 0; c < 64; ++c) {
            float2 dq_f = {dQ_acc[c*2] * scale, dQ_acc[c*2+1] * scale};
            __nv_bfloat162 dq_v = __float22bfloat162_rn(dq_f);
            g_dQ_row[c] = *reinterpret_cast<uint32_t*>(&dq_v);
        }
    }
}

__global__ void compute_dK_dV_kernel(
    const __nv_bfloat16* __restrict__ Q, 
    const __nv_bfloat16* __restrict__ K, 
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O, 
    const __nv_bfloat16* __restrict__ dO, 
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dK, 
    __nv_bfloat16* __restrict__ dV,
    int S, int d, float scale
) {
    int bh = blockIdx.y;
    int kv_block_idx = blockIdx.x;
    int tx = threadIdx.x;
    
    int base_idx = bh * S * 128;
    int kv_idx = kv_block_idx * 128 + tx;
    
    extern __shared__ char smem[];
    float (*s_dK)[129] = reinterpret_cast<float (*)[129]>(smem);
    float (*s_dV)[129] = reinterpret_cast<float (*)[129]>(smem + 128 * 129 * sizeof(float));
    __nv_bfloat16 (*s_Q)[128] = reinterpret_cast<__nv_bfloat16 (*)[128]>(smem + 2 * 128 * 129 * sizeof(float));
    __nv_bfloat16 (*s_dO)[128] = reinterpret_cast<__nv_bfloat16 (*)[128]>(smem + 2 * 128 * 129 * sizeof(float) + 64 * 128 * sizeof(__nv_bfloat16));
    float* s_D = reinterpret_cast<float*>(smem + 2 * 128 * 129 * sizeof(float) + 2 * 64 * 128 * sizeof(__nv_bfloat16));
    
    for(int c = 0; c < 128; ++c) {
        s_dK[tx][c] = 0.0f;
        s_dV[tx][c] = 0.0f;
    }
    
    uint32_t reg_K[64];
    uint32_t reg_V[64];
    
    if (kv_idx < S) {
        const uint32_t* g_K_row = reinterpret_cast<const uint32_t*>(K + base_idx + kv_idx * 128);
        const uint32_t* g_V_row = reinterpret_cast<const uint32_t*>(V + base_idx + kv_idx * 128);
        for (int c = 0; c < 64; ++c) {
            reg_K[c] = g_K_row[c];
            reg_V[c] = g_V_row[c];
        }
    }
    
    int min_q_block = (kv_block_idx * 128) / 64;
    int max_q_block = (S + 63) / 64;
    
    for (int q_block = min_q_block; q_block < max_q_block; ++q_block) {
        __syncthreads();
        
        if (tx < 64) {
            int q_idx_local = q_block * 64 + tx;
            float d_val = 0.0f;
            if (q_idx_local < S) {
                const uint32_t* g_O_row = reinterpret_cast<const uint32_t*>(O + base_idx + q_idx_local * 128);
                const uint32_t* g_dO_row = reinterpret_cast<const uint32_t*>(dO + base_idx + q_idx_local * 128);
                for(int c=0; c<64; ++c) {
                    __nv_bfloat162 o_v = *reinterpret_cast<const __nv_bfloat162*>(&g_O_row[c]);
                    __nv_bfloat162 do_v = *reinterpret_cast<const __nv_bfloat162*>(&g_dO_row[c]);
                    float2 o_f = __bfloat1622float2(o_v);
                    float2 do_f = __bfloat1622float2(do_v);
                    d_val += o_f.x * do_f.x + o_f.y * do_f.y;
                }
            }
            s_D[tx] = d_val;
        }
        
        int q_float4 = (64 * 128) / 8;
        const __nv_bfloat16* g_Q = Q + base_idx + q_block * 64 * 128;
        const __nv_bfloat16* g_dO = dO + base_idx + q_block * 64 * 128;
        
        for (int i = tx; i < q_float4; i += 128) {
            int row = i / 16;
            if (q_block * 64 + row < S) {
                reinterpret_cast<float4*>(s_Q)[i] = reinterpret_cast<const float4*>(g_Q)[i];
                reinterpret_cast<float4*>(s_dO)[i] = reinterpret_cast<const float4*>(g_dO)[i];
            } else {
                reinterpret_cast<float4*>(s_Q)[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                reinterpret_cast<float4*>(s_dO)[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        
        __syncthreads();
        
        if (kv_idx >= S) continue;
        
        for (int i = 0; i < 64; ++i) {
            int q_idx_global = q_block * 64 + i;
            if (q_idx_global < kv_idx) continue;
            if (q_idx_global >= S) continue;
            
            float s_ij = 0.0f;
            float ds_ij = 0.0f;
            
            #pragma unroll 16
            for (int c = 0; c < 64; ++c) {
                __nv_bfloat162 q_v = *reinterpret_cast<const __nv_bfloat162*>(&s_Q[i][c*2]);
                __nv_bfloat162 k_v = *reinterpret_cast<const __nv_bfloat162*>(&reg_K[c]);
                float2 q_f = __bfloat1622float2(q_v);
                float2 k_f = __bfloat1622float2(k_v);
                s_ij += q_f.x * k_f.x + q_f.y * k_f.y;
                
                __nv_bfloat162 do_v = *reinterpret_cast<const __nv_bfloat162*>(&s_dO[i][c*2]);
                __nv_bfloat162 v_v = *reinterpret_cast<const __nv_bfloat162*>(&reg_V[c]);
                float2 do_f = __bfloat1622float2(do_v);
                float2 v_f = __bfloat1622float2(v_v);
                ds_ij += do_f.x * v_f.x + do_f.y * v_f.y;
            }
            
            s_ij *= scale;
            float l_i = L[bh * S + q_idx_global];
            float p_ij = expf(s_ij - l_i);
            float dp_ij = p_ij * (ds_ij - s_D[i]);
            
            #pragma unroll 16
            for (int c = 0; c < 64; ++c) {
                __nv_bfloat162 q_v = *reinterpret_cast<const __nv_bfloat162*>(&s_Q[i][c*2]);
                float2 q_f = __bfloat1622float2(q_v);
                s_dK[tx][c*2]   += dp_ij * q_f.x;
                s_dK[tx][c*2+1] += dp_ij * q_f.y;
                
                __nv_bfloat162 do_v = *reinterpret_cast<const __nv_bfloat162*>(&s_dO[i][c*2]);
                float2 do_f = __bfloat1622float2(do_v);
                s_dV[tx][c*2]   += p_ij * do_f.x;
                s_dV[tx][c*2+1] += p_ij * do_f.y;
            }
        }
    }
    
    if (kv_idx < S) {
        uint32_t* g_dK_row = reinterpret_cast<uint32_t*>(dK + base_idx + kv_idx * 128);
        uint32_t* g_dV_row = reinterpret_cast<uint32_t*>(dV + base_idx + kv_idx * 128);
        
        for (int c = 0; c < 64; ++c) {
            float2 dk_f = {s_dK[tx][c*2] * scale, s_dK[tx][c*2+1] * scale};
            __nv_bfloat162 dk_v = __float22bfloat162_rn(dk_f);
            g_dK_row[c] = *reinterpret_cast<uint32_t*>(&dk_v);
            
            float2 dv_f = {s_dV[tx][c*2], s_dV[tx][c*2+1]};
            __nv_bfloat162 dv_v = __float22bfloat162_rn(dv_f);
            g_dV_row[c] = *reinterpret_cast<uint32_t*>(&dv_v);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int d = Q.size(3);
    
    float scale = 1.0f / std::sqrt(static_cast<float>(d));
    
    dim3 grid_Q((S + 127) / 128, B * H);
    dim3 block_Q(128);
    
    CUDA_CHECK(cudaFuncSetAttribute(compute_dQ_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 66048));
    
    compute_dQ_kernel<<<grid_Q, block_Q, 66048, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(O.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        S, d, scale
    );
    
    dim3 grid_KV((S + 127) / 128, B * H);
    dim3 block_KV(128);
    
    CUDA_CHECK(cudaFuncSetAttribute(compute_dK_dV_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 165120));
    
    compute_dK_dV_kernel<<<grid_KV, block_KV, 165120, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(O.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        S, d, scale
    );
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_bwd::run);

} // namespace tvm_ffi_mha_bwd