#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <stdio.h>
#include <math.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_example_cuda {

// Pass 1: Compute dQ. Outer loop over Q (queries), inner loop over K/V (keys/values).
__global__ void mha_bwd_dQ(
    const __nv_bfloat16* __restrict__ Q, 
    const __nv_bfloat16* __restrict__ K, 
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O, 
    const __nv_bfloat16* __restrict__ dO, 
    const float* __restrict__ L, 
    __nv_bfloat16* __restrict__ dQ,
    int B, int H, int S, int d, float scale) 
{
    int b = blockIdx.y / H;
    int h = blockIdx.y % H;
    int bx = blockIdx.x;
    
    int ty = threadIdx.y; // 0..31 (row within block)
    int tx = threadIdx.x; // 0..31 (column chunk within block)
    
    int i = bx * 32 + ty;
    int k_start = tx * 4; // Each thread handles 4 elements of d=128
    
    int head_offset = (b * H + h) * S;
    int q_offset = (head_offset + i) * d;
    
    float q[4] = {0}, do_[4] = {0}, o[4] = {0}, dq[4] = {0};
    float L_i = 0.0f;
    
    // Load Q_i, dO_i, O_i and L_i into registers
    if (i < S) {
        L_i = L[head_offset + i];
        uint2 q_val2 = *reinterpret_cast<const uint2*>(&Q[q_offset + k_start]);
        uint2 do_val2 = *reinterpret_cast<const uint2*>(&dO[q_offset + k_start]);
        uint2 o_val2 = *reinterpret_cast<const uint2*>(&O[q_offset + k_start]);
        
        const __nv_bfloat16* q_ptr = reinterpret_cast<const __nv_bfloat16*>(&q_val2);
        const __nv_bfloat16* do_ptr = reinterpret_cast<const __nv_bfloat16*>(&do_val2);
        const __nv_bfloat16* o_ptr = reinterpret_cast<const __nv_bfloat16*>(&o_val2);
        
        for (int c = 0; c < 4; ++c) {
            q[c]  = __bfloat162float(q_ptr[c]);
            do_[c]= __bfloat162float(do_ptr[c]);
            o[c]  = __bfloat162float(o_ptr[c]);
        }
    }
    
    // Compute D_i = sum(dO_i * O_i) across the warp
    float sum_d = 0;
    for (int c = 0; c < 4; ++c) {
        sum_d += do_[c] * o[c];
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_d += __shfl_down_sync(0xffffffff, sum_d, offset);
    }
    float D_i = __shfl_sync(0xffffffff, sum_d, 0);
    
    __shared__ __nv_bfloat16 smem_K[32][128];
    __shared__ __nv_bfloat16 smem_V[32][128];
    
    // Iterate over K and V in chunks of 32
    for (int j_block = 0; j_block < S; j_block += 32) {
        int load_j = j_block + ty;
        int kv_offset = (head_offset + load_j) * d;
        
        // Coalesced load to shared memory
        if (load_j < S) {
            uint2 k_val2 = *reinterpret_cast<const uint2*>(&K[kv_offset + k_start]);
            uint2 v_val2 = *reinterpret_cast<const uint2*>(&V[kv_offset + k_start]);
            *reinterpret_cast<uint2*>(&smem_K[ty][k_start]) = k_val2;
            *reinterpret_cast<uint2*>(&smem_V[ty][k_start]) = v_val2;
        }
        __syncthreads();
        
        if (i < S) {
            for (int j_sub = 0; j_sub < 32; ++j_sub) {
                int j = j_block + j_sub;
                if (j < S) {
                    float sum_qk = 0;
                    float sum_dp = 0;
                    
                    uint2 sk_val2 = *reinterpret_cast<const uint2*>(&smem_K[j_sub][k_start]);
                    uint2 sv_val2 = *reinterpret_cast<const uint2*>(&smem_V[j_sub][k_start]);
                    const __nv_bfloat16* sk_ptr = reinterpret_cast<const __nv_bfloat16*>(&sk_val2);
                    const __nv_bfloat16* sv_ptr = reinterpret_cast<const __nv_bfloat16*>(&sv_val2);
                    
                    for (int c = 0; c < 4; ++c) {
                        sum_qk += q[c] * __bfloat162float(sk_ptr[c]);
                        sum_dp += do_[c] * __bfloat162float(sv_ptr[c]);
                    }
                    
                    for (int offset = 16; offset > 0; offset /= 2) {
                        sum_qk += __shfl_down_sync(0xffffffff, sum_qk, offset);
                        sum_dp += __shfl_down_sync(0xffffffff, sum_dp, offset);
                    }
                    
                    sum_qk = __shfl_sync(0xffffffff, sum_qk, 0);
                    sum_dp = __shfl_sync(0xffffffff, sum_dp, 0);
                    
                    float P_ij = expf(sum_qk * scale - L_i);
                    float dS_ij = P_ij * (sum_dp - D_i);
                    
                    for (int c = 0; c < 4; ++c) {
                        dq[c] += dS_ij * __bfloat162float(sk_ptr[c]) * scale;
                    }
                }
            }
        }
        __syncthreads();
    }
    
    // Write out fully reduced dQ
    if (i < S) {
        __nv_bfloat16 dq_bf[4];
        for (int c = 0; c < 4; ++c) {
            dq_bf[c] = __float2bfloat16(dq[c]);
        }
        *reinterpret_cast<uint2*>(&dQ[q_offset + k_start]) = *reinterpret_cast<uint2*>(&dq_bf);
    }
}

// Pass 2: Compute dK and dV. Outer loop over K/V, inner loop over Q.
__global__ void mha_bwd_dK_dV(
    const __nv_bfloat16* __restrict__ Q, 
    const __nv_bfloat16* __restrict__ K, 
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O, 
    const __nv_bfloat16* __restrict__ dO, 
    const float* __restrict__ L, 
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int B, int H, int S, int d, float scale) 
{
    int b = blockIdx.y / H;
    int h = blockIdx.y % H;
    int bx = blockIdx.x;
    
    int ty = threadIdx.y; 
    int tx = threadIdx.x; 
    
    int j = bx * 32 + ty;
    int k_start = tx * 4;
    
    int head_offset = (b * H + h) * S;
    int kv_offset = (head_offset + j) * d;
    
    float k_val[4] = {0}, v_val[4] = {0};
    float dk[4] = {0}, dv[4] = {0};
    
    if (j < S) {
        uint2 k_val2 = *reinterpret_cast<const uint2*>(&K[kv_offset + k_start]);
        uint2 v_val2 = *reinterpret_cast<const uint2*>(&V[kv_offset + k_start]);
        const __nv_bfloat16* k_ptr = reinterpret_cast<const __nv_bfloat16*>(&k_val2);
        const __nv_bfloat16* v_ptr = reinterpret_cast<const __nv_bfloat16*>(&v_val2);
        for (int c = 0; c < 4; ++c) {
            k_val[c] = __bfloat162float(k_ptr[c]);
            v_val[c] = __bfloat162float(v_ptr[c]);
        }
    }
    
    __shared__ __nv_bfloat16 smem_Q[32][128];
    __shared__ __nv_bfloat16 smem_dO[32][128];
    __shared__ float smem_L[32];
    __shared__ float smem_D[32];
    
    // Iterate over Q and dO in chunks of 32
    for (int i_block = 0; i_block < S; i_block += 32) {
        int load_i = i_block + ty;
        int q_offset = (head_offset + load_i) * d;
        
        float L_i = 0.0f;
        float sum_d = 0.0f;
        
        if (load_i < S) {
            if (tx == 0) L_i = L[head_offset + load_i];
            
            uint2 q_val2 = *reinterpret_cast<const uint2*>(&Q[q_offset + k_start]);
            uint2 do_val2 = *reinterpret_cast<const uint2*>(&dO[q_offset + k_start]);
            uint2 o_val2 = *reinterpret_cast<const uint2*>(&O[q_offset + k_start]);
            
            *reinterpret_cast<uint2*>(&smem_Q[ty][k_start]) = q_val2;
            *reinterpret_cast<uint2*>(&smem_dO[ty][k_start]) = do_val2;
            
            const __nv_bfloat16* do_ptr = reinterpret_cast<const __nv_bfloat16*>(&do_val2);
            const __nv_bfloat16* o_ptr = reinterpret_cast<const __nv_bfloat16*>(&o_val2);
            for (int c = 0; c < 4; ++c) {
                sum_d += __bfloat162float(do_ptr[c]) * __bfloat162float(o_ptr[c]);
            }
        }
        
        for (int offset = 16; offset > 0; offset /= 2) {
            sum_d += __shfl_down_sync(0xffffffff, sum_d, offset);
        }
        
        if (tx == 0) {
            smem_L[ty] = L_i;
            smem_D[ty] = sum_d;
        }
        __syncthreads();
        
        if (j < S) {
            for (int i_sub = 0; i_sub < 32; ++i_sub) {
                int i = i_block + i_sub;
                if (i < S) {
                    float sum_qk = 0;
                    float sum_dp = 0;
                    
                    uint2 sq_val2 = *reinterpret_cast<const uint2*>(&smem_Q[i_sub][k_start]);
                    uint2 sdo_val2 = *reinterpret_cast<const uint2*>(&smem_dO[i_sub][k_start]);
                    const __nv_bfloat16* sq_ptr = reinterpret_cast<const __nv_bfloat16*>(&sq_val2);
                    const __nv_bfloat16* sdo_ptr = reinterpret_cast<const __nv_bfloat16*>(&sdo_val2);
                    
                    for (int c = 0; c < 4; ++c) {
                        sum_qk += __bfloat162float(sq_ptr[c]) * k_val[c];
                        sum_dp += __bfloat162float(sdo_ptr[c]) * v_val[c];
                    }
                    
                    for (int offset = 16; offset > 0; offset /= 2) {
                        sum_qk += __shfl_down_sync(0xffffffff, sum_qk, offset);
                        sum_dp += __shfl_down_sync(0xffffffff, sum_dp, offset);
                    }
                    
                    sum_qk = __shfl_sync(0xffffffff, sum_qk, 0);
                    sum_dp = __shfl_sync(0xffffffff, sum_dp, 0);
                    
                    float L_i_shared = smem_L[i_sub];
                    float D_i_shared = smem_D[i_sub];
                    
                    float P_ij = expf(sum_qk * scale - L_i_shared);
                    float dS_ij = P_ij * (sum_dp - D_i_shared);
                    
                    for (int c = 0; c < 4; ++c) {
                        dk[c] += dS_ij * __bfloat162float(sq_ptr[c]) * scale;
                        dv[c] += P_ij * __bfloat162float(sdo_ptr[c]);
                    }
                }
            }
        }
        __syncthreads();
    }
    
    // Write out fully reduced dK and dV
    if (j < S) {
        __nv_bfloat16 dk_bf[4], dv_bf[4];
        for (int c = 0; c < 4; ++c) {
            dk_bf[c] = __float2bfloat16(dk[c]);
            dv_bf[c] = __float2bfloat16(dv[c]);
        }
        *reinterpret_cast<uint2*>(&dK[kv_offset + k_start]) = *reinterpret_cast<uint2*>(&dk_bf);
        *reinterpret_cast<uint2*>(&dV[kv_offset + k_start]) = *reinterpret_cast<uint2*>(&dv_bf);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int d = Q.size(3); 
    
    float scale = 1.0f / sqrtf((float)d);
    
    const __nv_bfloat16* q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* k_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* v_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* o_ptr = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* do_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* l_ptr = static_cast<const float*>(L.data_ptr());
    
    __nv_bfloat16* dq_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dk_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dv_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    // Each block processes 32 sequence tokens
    dim3 grid((S + 31) / 32, B * H, 1);
    dim3 block(32, 32, 1); // 1024 threads, 32 warps mapping to 32 rows
    
    mha_bwd_dQ<<<grid, block, 0, stream>>>(q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, l_ptr, dq_ptr, B, H, S, d, scale);
    mha_bwd_dK_dV<<<grid, block, 0, stream>>>(q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, l_ptr, dk_ptr, dv_ptr, B, H, S, d, scale);
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda