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

namespace mha_bwd_d128 {

constexpr int TILE_M = 64;
constexpr int TILE_N = 64;
constexpr int BLOCK_D = 128;

__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 f32_to_bf16(float x) {
    return __float2bfloat16(x);
}

// Kernel: Compute dV = sum_m P[m,n] * dO[m,d] for each (n,d)
// Each block handles one (b, h, n_start, d_block) and accumulates over all m
__global__ void mha_bwd_dV_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dV,
    int B, int H, int S, int D,
    float sqrt_d_inv) {

    extern __shared__ char smem[];
    
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sK = sQ + TILE_M * BLOCK_D;
    float* sLSE = reinterpret_cast<float*>(sK + TILE_M);
    float* sP = sLSE + TILE_M;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bwid = threadIdx.z;
    int tid = tx + ty * 32;
    
    int bh_idx = blockIdx.x;
    int b = bh_idx / H;
    int h = bh_idx % H;
    
    int n_tile = blockIdx.y;
    int n_start = n_tile * TILE_N;
    
    int d_offset = bwid * 32;
    
    int d_inner = tx;
    
    float acc = 0.0f;
    
    if (b < B && h < H && n_start < S && d_offset + d_inner < D) {
        int base_addr = (b * H + h) * S * D;
        
        for (int m_tile = 0; m_tile < S; m_tile += TILE_M) {
            int m_start = m_tile;
            
            // Load Q tile
            for (int i = tid; i < TILE_M * BLOCK_D; i += 128) {
                int m_idx = i / BLOCK_D;
                int d_idx = i % BLOCK_D;
                if (m_start + m_idx < S && d_offset + d_inner == d_idx || 
                    (d_idx >= d_offset && d_idx < d_offset + 32)) {
                    int addr = base_addr + (m_start + m_idx) * D + d_idx;
                    sQ[i] = (m_start + m_idx < S) ? Q[addr] : f32_to_bf16(0.0f);
                } else {
                    sQ[i] = f32_to_bf16(0.0f);
                }
            }
            
            // Load K tile and LSE
            for (int i = tid; i < TILE_M * BLOCK_D; i += 128) {
                int k_idx = i / BLOCK_D;
                int d_idx = i % BLOCK_D;
                if (k_idx >= d_offset && k_idx < d_offset + 32) {
                    int addr = base_addr + (n_start + k_idx) * D + d_inner;
                    sK[i] = (n_start + k_idx < S) ? K[addr] : f32_to_bf16(0.0f);
                } else {
                    sK[i] = f32_to_bf16(0.0f);
                }
            }
            
            if (ty == 0 && tx < TILE_M) {
                sLSE[tx] = (m_start + tx < S) ? L[base_addr / D + m_start + tx] : 0.0f;
            }
            
            __syncthreads();
            
            // Compute S = Q @ K^T and P = exp(S - LSE)
            float q_val = bf16_to_f32(sQ[d_inner * TILE_M + ty * 32 + tx]);
            for (int m_local = 0; m_local < TILE_M; m_local++) {
                float s = 0.0f;
                for (int d = 0; d < BLOCK_D; d++) {
                    s += bf16_to_f32(sQ[m_local * BLOCK_D + d]) * 
                         bf16_to_f32(sK[d_inner * TILE_M + d]);
                }
                s *= sqrt_d_inv;
                sP[m_local] = expf(s - sLSE[m_local]);
            }
            
            __syncthreads();
            
            // Accumulate dV
            for (int n_local = 0; n_local < TILE_N; n_local += 128) {
                float p_val = (n_local < TILE_N) ? sP[n_local] : 0.0f;
                if (p_val > 0.0f) {
                    acc += p_val * bf16_to_f32(sQ[d_inner * TILE_M + n_local]);
                }
            }
            
            __syncthreads();
        }
    }
    
    // Write result
    if (b < B && h < H && n_start + tx < S && d_offset + ty * 32 + tx < D) {
        int idx = (b * H + h) * S * D + (n_start + tx) * D + d_offset + ty * 32 + tx;
        dV[idx] = f32_to_bf16(acc);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_ptr = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    size_t out_size = B * H * S * D * sizeof(__nv_bfloat16);
    CUDA_CHECK(cudaMemset(dQ_ptr, 0, out_size));
    CUDA_CHECK(cudaMemset(dK_ptr, 0, out_size));
    CUDA_CHECK(cudaMemset(dV_ptr, 0, out_size));
    
    float sqrt_d_inv = 1.0f / sqrtf(static_cast<float>(D));
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    dim3 block(32, 4, 4);
    dim3 grid(B * H, (S + TILE_N - 1) / TILE_N);
    
    size_t smem_size = TILE_M * BLOCK_D * sizeof(__nv_bfloat16) * 2 + 
                       TILE_M * sizeof(float) * 2;
    
    mha_bwd_dV_kernel<<<grid, block, smem_size, stream>>>(
        Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, dV_ptr,
        B, H, S, D, sqrt_d_inv);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128::run);

}  // namespace mha_bwd_d128