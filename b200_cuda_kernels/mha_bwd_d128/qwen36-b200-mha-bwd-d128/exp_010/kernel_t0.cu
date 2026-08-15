#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
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
constexpr int THREADS_X = 32;
constexpr int THREADS_Y = 4;
constexpr int THREADS_PER_BLOCK = THREADS_X * THREADS_Y;

__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float x) {
    return __float2bfloat16(x);
}

__device__ __forceinline__ void warp_reduce_sum(float& val) {
    val += __shfl_down_sync(0xFFFFFFFF, val, 16);
    val += __shfl_down_sync(0xFFFFFFFF, val, 8);
    val += __shfl_down_sync(0xFFFFFFFF, val, 4);
    val += __shfl_down_sync(0xFFFFFFFF, val, 2);
    val += __shfl_down_sync(0xFFFFFFFF, val, 1);
}

// Kernel 1: Compute dV = P^T @ dO
// Each block computes a TILE_M x TILE_N tile of dV
// dV[s_v, h, d] += sum over s_q: P[s_q, s_v] * dO[s_q, d]
__global__ void mha_bwd_dV_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dV,
    int B, int H, int S, int D,
    float sqrt_d_inv) {

    extern __shared__ char smem_raw[];
    
    __nv_bfloat16* smem_Q = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* smem_K = smem_Q + TILE_M * D;
    float* smem_L = reinterpret_cast<float*>(smem_K + TILE_N * D);
    __nv_bfloat16* smem_dO = reinterpret_cast<__nv_bfloat16*>(smem_L + TILE_M);
    float* smem_P = reinterpret_cast<float*>(smem_dO + TILE_M * D);
    
    int block_id = blockIdx.x;
    int head_block = block_id / (gridDim.x / (B * H));
    int batch_head = head_block;
    int b = batch_head / H;
    int h = batch_head % H;
    
    int tile_q_start = (block_id % (gridDim.x / (B * H))) / ((S + TILE_N - 1) / TILE_N);
    int tile_k_start = (block_id % (gridDim.x / (B * H))) % ((S + TILE_N - 1) / TILE_N);
    
    tile_q_start *= TILE_M;
    tile_k_start *= TILE_N;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = tx + ty * THREADS_X;
    
    float accum[THREADS_PER_BLOCK];
    
    for (int q_tile = tile_q_start; q_tile < S; q_tile += TILE_M) {
        for (int k_tile = tile_k_start; k_tile < S; k_tile += TILE_N) {
            float lse = (q_tile < S && h < H && b < B) ? 
                L[((long long)b * H + h) * S + q_tile] : 0.0f;
            
            // Load Q tile and K tile into shared memory
            for (int i = tid; i < TILE_M * D; i += THREADS_PER_BLOCK) {
                int q_pos = i / D;
                int d_pos = i % D;
                int global_q = (long long)((long long)b * H + h) * S * D + (q_tile + q_pos) * D + d_pos;
                if (q_tile + q_pos < S && d_pos < D) {
                    smem_Q[i] = Q[global_q];
                } else {
                    smem_Q[i] = __float2bfloat16(0.0f);
                }
            }
            
            for (int i = tid; i < TILE_N * D; i += THREADS_PER_BLOCK) {
                int k_pos = i / D;
                int d_pos = i % D;
                int global_k = (long long)((long long)b * H + h) * S * D + (k_tile + k_pos) * D + d_pos;
                if (k_tile + k_pos < S && d_pos < D) {
                    smem_K[i] = K[global_k];
                } else {
                    smem_K[i] = __float2bfloat16(0.0f);
                }
            }
            __syncthreads();
            
            // Store LSE for this Q position
            if (ty == 0) {
                smem_L[tx] = lse;
            }
            __syncthreads();
            
            // Load dO tile
            for (int i = tid; i < TILE_M * D; i += THREADS_PER_BLOCK) {
                int q_pos = i / D;
                int d_pos = i % D;
                int global_do = (long long)((long long)b * H + h) * S * D + (q_tile + q_pos) * D + d_pos;
                if (q_tile + q_pos < S && d_pos < D) {
                    smem_dO[i] = dO[global_do];
                } else {
                    smem_dO[i] = __float2bfloat16(0.0f);
                }
            }
            __syncthreads();
            
            // Compute S = Q @ K^T / sqrt(d) and P = softmax
            for (int m = ty; m < TILE_M; m += THREADS_Y) {
                for (int n = tx; n < TILE_N; n += THREADS_X) {
                    float s = 0.0f;
                    for (int d = 0; d < D; d++) {
                        s += bf16_to_float(smem_Q[m * D + d]) * bf16_to_float(smem_K[n * D + d]);
                    }
                    s *= sqrt_d_inv;
                    smem_P[m * TILE_N + n] = expf(s - smem_L[m]);
                }
            }
            __syncthreads();
            
            // Accumulate dV: dV[k_tile + n, d] += sum_m P[m, k_tile + n] * dO[q_tile + m, d]
            for (int out_n = tx; out_n < TILE_N; out_n += THREADS_X) {
                for (int out_d = ty; out_d < D; out_d += THREADS_Y) {
                    float dv_acc = 0.0f;
                    for (int m = 0; m < TILE_M; m++) {
                        dv_acc += smem_P[m * TILE_N + out_n] * 
                                  bf16_to_float(smem_dO[m * D + out_d]);
                    }
                    
                    int global_dv = (long long)((long long)b * H + h) * S * D + 
                                    (k_tile + out_n) * D + out_d;
                    if (k_tile + out_n < S && out_d < D && b < B && h < H) {
                        atomicAdd(
                            reinterpret_cast<float*>(&dV[global_dv]),
                            dv_acc);
                    }
                }
            }
        }
    }
}

// Kernel 2: Compute dQ and dK
// dQ = dS @ K, dK = dS^T @ Q
// dS = P ⊙ (dO @ V^T - local_mean)
__global__ void mha_bwd_dQ_dK_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    int B, int H, int S, int D,
    float sqrt_d_inv) {

    extern __shared__ char smem_raw[];
    
    __nv_bfloat16* smem_Q = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* smem_K = smem_Q + TILE_M * D;
    __nv_bfloat16* smem_V = smem_K + TILE_N * D;
    float* smem_L = reinterpret_cast<float*>(smem_V + TILE_N * D);
    __nv_bfloat16* smem_dO = reinterpret_cast<__nv_bfloat16*>(smem_L + TILE_M);
    float* smem_P = reinterpret_cast<float*>(smem_dO + TILE_M * D);
    float* smem_dPV = reinterpret_cast<float*>(smem_P + TILE_M * TILE_N);
    float* smem_dS = smem_dPV + TILE_M * TILE_N;
    
    int block_id = blockIdx.x;
    int head_block = block_id / (gridDim.x / (B * H));
    int batch_head = head_block;
    int b = batch_head / H;
    int h = batch_head % H;
    
    int tile_q_start = (block_id % (gridDim.x / (B * H))) / ((S + TILE_N - 1) / TILE_N);
    int tile_k_start = (block_id % (gridDim.x / (B * H))) % ((S + TILE_N - 1) / TILE_N);
    
    tile_q_start *= TILE_M;
    tile_k_start *= TILE_N;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = tx + ty * THREADS_X;
    
    for (int q_tile = tile_q_start; q_tile < S; q_tile += TILE_M) {
        for (int k_tile = tile_k_start; k_tile < S; k_tile += TILE_N) {
            float lse = (q_tile < S && h < H && b < B) ? 
                L[((long long)b * H + h) * S + q_tile] : 0.0f;
            
            // Load Q, K, V tiles
            for (int i = tid; i < TILE_M * D; i += THREADS_PER_BLOCK) {
                int pos = i / D;
                int d = i % D;
                int global_addr = (long long)((long long)b * H + h) * S * D + (q_tile + pos) * D + d;
                if (q_tile + pos < S && d < D) {
                    smem_Q[i] = Q[global_addr];
                } else {
                    smem_Q[i] = __float2bfloat16(0.0f);
                }
                
                global_addr = (long long)((long long)b * H + h) * S * D + (k_tile + pos) * D + d;
                if (k_tile + pos < S && d < D) {
                    smem_K[i] = K[global_addr];
                } else {
                    smem_K[i] = __float2bfloat16(0.0f);
                }
            }
            __syncthreads();
            
            for (int i = tid; i < TILE_N * D; i += THREADS_PER_BLOCK) {
                int pos = i / D;
                int d = i % D;
                int global_addr = (long long)((long long)b * H + h) * S * D + (k_tile + pos) * D + d;
                if (k_tile + pos < S && d < D) {
                    smem_V[i] = V[global_addr];
                } else {
                    smem_V[i] = __float2bfloat16(0.0f);
                }
            }
            __syncthreads();
            
            if (ty == 0) {
                smem_L[tx] = lse;
            }
            __syncthreads();
            
            // Load dO
            for (int i = tid; i < TILE_M * D; i += THREADS_PER_BLOCK) {
                int pos = i / D;
                int d = i % D;
                int global_addr = (long long)((long long)b * H + h) * S * D + (q_tile + pos) * D + d;
                if (q_tile + pos < S && d < D) {
                    smem_dO[i] = dO[global_addr];
                } else {
                    smem_dO[i] = __float2bfloat16(0.0f);
                }
            }
            __syncthreads();
            
            // Compute S and P
            for (int m = ty; m < TILE_M; m += THREADS_Y) {
                for (int n = tx; n < TILE_N; n += THREADS_X) {
                    float s = 0.0f;
                    for (int d = 0; d < D; d++) {
                        s += bf16_to_float(smem_Q[m * D + d]) * bf16_to_float(smem_K[n * D + d]);
                    }
                    s *= sqrt_d_inv;
                    smem_P[m * TILE_N + n] = expf(s - smem_L[m]);
                }
            }
            __syncthreads();
            
            // Compute dPV = dO @ V^T
            for (int m = ty; m < TILE_M; m += THREADS_Y) {
                for (int n = tx; n < TILE_N; n += THREADS_X) {
                    float dpv = 0.0f;
                    for (int d = 0; d < D; d++) {
                        dpv += bf16_to_float(smem_dO[m * D + d]) * bf16_to_float(smem_V[n * D + d]);
                    }
                    smem_dPV[m * TILE_N + n] = dpv;
                }
            }
            __syncthreads();
            
            // Compute local mean: sum over n of P[m,n] * dPV[m,n]
            for (int m = ty; m < TILE_M; m += THREADS_Y) {
                float local_mean = 0.0f;
                for (int n = 0; n < TILE_N; n++) {
                    local_mean += smem_P[m * TILE_N + n] * smem_dPV[m * TILE_N + n];
                }
                if (tx == 0) {
                    smem_dPV[m * TILE_N + TILE_N] = local_mean; // Store in extra column
                }
            }
            __syncthreads();
            
            // Compute dS = P ⊙ (dPV - local_mean)
            for (int m = ty; m < TILE_M; m += THREADS_Y) {
                float lm = smem_dPV[m * TILE_N + TILE_N];
                for (int n = tx; n < TILE_N; n += THREADS_X) {
                    smem_dS[m * TILE_N + n] = smem_P[m * TILE_N + n] * (smem_dPV[m * TILE_N + n] - lm);
                }
            }
            __syncthreads();
            
            // Compute dQ contribution: dQ[q_tile+m, :] += dS[m,:] @ K[:, :]
            for (int m_out = ty; m_out < TILE_M; m_out += THREADS_Y) {
                for (int d_out = tx; d_out < D; d_out += THREADS_X) {
                    float dq_acc = 0.0f;
                    for (int n = 0; n < TILE_N; n++) {
                        dq_acc += smem_dS[m_out * TILE_N + n] * 
                                  bf16_to_float(smem_K[n * D + d_out]);
                    }
                    
                    int global_dq = (long long)((long long)b * H + h) * S * D + 
                                    (q_tile + m_out) * D + d_out;
                    if (q_tile + m_out < S && d_out < D && b < B && h < H) {
                        atomicAdd(
                            reinterpret_cast<float*>(&dQ[global_dq]),
                            dq_acc);
                    }
                }
            }
            __syncthreads();
            
            // Compute dK contribution: dK[k_tile+n, :] += dS^T[:,m] @ Q[m,:]
            for (int n_out = ty; n_out < TILE_N; n_out += THREADS_Y) {
                for (int d_out = tx; d_out < D; d_out += THREADS_X) {
                    float dk_acc = 0.0f;
                    for (int m = 0; m < TILE_M; m++) {
                        dk_acc += smem_dS[m * TILE_N + n_out] * 
                                  bf16_to_float(smem_Q[m * D + d_out]);
                    }
                    
                    int global_dk = (long long)((long long)b * H + h) * S * D + 
                                    (k_tile + n_out) * D + d_out;
                    if (k_tile + n_out < S && d_out < D && b < B && h < H) {
                        atomicAdd(
                            reinterpret_cast<float*>(&dK[global_dk]),
                            dk_acc);
                    }
                }
            }
        }
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
    
    // Initialize outputs to zero
    size_t out_size = B * H * S * D * sizeof(__nv_bfloat16);
    CUDA_CHECK(cudaMemset(dQ_ptr, 0, out_size));
    CUDA_CHECK(cudaMemset(dK_ptr, 0, out_size));
    CUDA_CHECK(cudaMemset(dV_ptr, 0, out_size));
    
    float sqrt_d_inv = 1.0f / sqrtf(static_cast<float>(D));
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    // Calculate grid and block dimensions
    int num_blocks_per_bh = ((S + TILE_M - 1) / TILE_M) * ((S + TILE_N - 1) / TILE_N);
    int total_blocks = B * H * num_blocks_per_bh;
    
    dim3 block(THREADS_X, THREADS_Y);
    dim3 grid(total_blocks);
    
    // Shared memory size calculation
    size_t smem_size = 
        TILE_M * D * sizeof(__nv_bfloat16) +  // smem_Q
        TILE_N * D * sizeof(__nv_bfloat16) +  // smem_K or smem_V
        TILE_N * D * sizeof(__nv_bfloat16) +  // smem_V (for dQ_dK kernel)
        TILE_M * sizeof(float) +              // smem_L
        TILE_M * D * sizeof(__nv_bfloat16) +  // smem_dO
        TILE_M * TILE_N * sizeof(float) +     // smem_P
        TILE_M * TILE_N * sizeof(float) +     // smem_dPV
        TILE_M * TILE_N * sizeof(float);      // smem_dS
    
    // Launch dV kernel
    mha_bwd_dV_kernel<<<grid, block, TILE_M * D * sizeof(__nv_bfloat16) * 2 + 
                              TILE_M * sizeof(float) + 
                              TILE_M * D * sizeof(__nv_bfloat16) +
                              TILE_M * TILE_N * sizeof(float), stream>>>(
        Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, dV_ptr,
        B, H, S, D, sqrt_d_inv);
    CUDA_CHECK(cudaGetLastError());
    
    // Launch dQ and dK kernel
    mha_bwd_dQ_dK_kernel<<<grid, block, static_cast<size_t>(smem_size), stream>>>(
        Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, dQ_ptr, dK_ptr,
        B, H, S, D, sqrt_d_inv);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128::run);

}  // namespace mha_bwd_d128