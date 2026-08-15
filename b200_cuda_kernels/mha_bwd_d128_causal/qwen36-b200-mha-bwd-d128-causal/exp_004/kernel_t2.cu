#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_bwd_impl {

__device__ __forceinline__ static float bf16tof(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ static __nv_bfloat16 f2bf16(float x) {
    return __float2bfloat16(x);
}

constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 64;
constexpr int TIL_K = 128; // Since d=128 exactly

/**
 * Causal Multi-Head Attention Backward Kernel
 * 
 * Computes dQ, dK, dV from Q, K, V, dO, L (logsumexp).
 * 
 * Strategy: Two-pass tiled approach
 * Pass 1: For each (bm, bn) tile, accumulate into:
 *   - dAccD[bh, i] += sum_{j in bn} P[i,j] * delta[i,j]
 *   - dAccdV[bh, j, :] += sum_{i in bm, i>=j} P[i,j] * dO[i,:]
 * 
 * Pass 2: Using accumulated dAccD, compute final dQ and dK
 */
__global__ void mha_bwd_kernel_v2(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const float* __restrict__ L,
    const __nv_bfloat16* __restrict__ dO,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    float* __restrict__ dAccD, // [num_bh, S] partial D accumulator
    int S, int d, int num_heads, float inv_sqrt_d)
{
    int bh = blockIdx.x / gridDim.y;
    int n_tile = blockIdx.x % gridDim.y;
    int b = bh / num_heads;
    int h = bh % num_heads;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // Block position
    int bm_start = blockIdx.z * BLOCK_M;
    int bn_start = n_tile * BLOCK_N;
    bool last_bn_tile = (n_tile == (int)gridDim.y - 1);

    uint64_t off = (uint64_t)b * num_heads + h;

    extern __shared__ char smem_char[];
    __align__(8) __nv_bfloat16* s_Q = (__nv_bfloat16*)smem_char;
    __align__(8) __nv_bfloat16* s_K = s_Q + BLOCK_M * d;
    __align__(8) __nv_bfloat16* s_V = s_K + BLOCK_N * d;
    __align__(8) __nv_bfloat16* s_dO = s_V + BLOCK_N * d;
    float* s_P = reinterpret_cast<float*>(reinterpret_cast<char*>(s_dO) + BLOCK_M * d * sizeof(__nv_bfloat16));
    float* s_delta = reinterpret_cast<float*>(reinterpret_cast<char*>(s_P) + BLOCK_M * BLOCK_N * sizeof(float));

    // Pointers within this (b,h) slice
    const __nv_bfloat16* qb = Q + off * S * d;
    const __nv_bfloat16* kb = K + off * S * d;
    const __nv_bfloat16* vb = V + off * S * d;
    const float* lb = L + off * S;
    const __nv_bfloat16* dob = dO + off * S * d;

    // Initialize dAccD accumulator for valid rows
    if (tid == 0) {
        int acc_off = bh * S;
        for (int i = bm_start; i < bm_start + BLOCK_M && i < S; i++) {
            dAccD[acc_off + i] = 0.0f;
        }
    }

    // Load Q tile into shared memory
    for (int f = tid; f < BLOCK_M * d; f += blockDim.x) {
        int i = f / d;
        int col = f % d;
        int gi = bm_start + i;
        if (gi < S) {
            s_Q[f] = qb[gi * d + col];
        } else {
            s_Q[f] = f2bf16(0.f);
        }
    }

    for (int n_iter = 0; n_iter <= 0; n_iter++) { // Only one iteration since d=TIL_K=128
        // Load K tile into shared memory
        for (int f = tid; f < BLOCK_N * d; f += blockDim.x) {
            int j = f / d;
            int col = f % d;
            int gj = bn_start + j;
            if (gj < S) {
                s_K[f] = kb[gj * d + col];
            } else {
                s_K[f] = f2bf16(0.f);
            }
        }

        // Load V tile into shared memory  
        for (int f = tid; f < BLOCK_N * d; f += blockDim.x) {
            int j = f / d;
            int col = f % d;
            int gj = bn_start + j;
            if (gj < S) {
                s_V[f] = vb[gj * d + col];
            } else {
                s_V[f] = f2bf16(0.f);
            }
        }

        // Load dO tile into shared memory
        for (int f = tid; f < BLOCK_M * d; f += blockDim.x) {
            int i = f / d;
            int col = f % d;
            int gi = bm_start + i;
            if (gi < S) {
                s_dO[f] = dob[gi * d + col];
            } else {
                s_dO[f] = f2bf16(0.f);
            }
        }
        __syncthreads();

        // Compute local S[i,j], P[i,j], delta[i,j]
        float local_di = 0.0f;
        for (int i = warp_id; i < BLOCK_M; i += 4) {
            int gi = bm_start + i;
            if (gi >= S) break;
            float Li = lb[gi];
            for (int j = 0; j < BLOCK_N; j++) {
                int gj = bn_start + j;
                if (gj > gi) { // Causal mask
                    continue;
                }
                if (gj >= S) break;

                // Compute similarity Q[i,:] . K[j,:]
                float sim = 0.0f;
                #pragma unroll
                for (int k = 0; k < d; k += 4) {
                    sim += bf16tof(s_Q[i * d + k]) * bf16tof(s_K[j * d + k]);
                    sim += bf16tof(s_Q[i * d + k + 1]) * bf16tof(s_K[j * d + k + 1]);
                    sim += bf16tof(s_Q[i * d + k + 2]) * bf16tof(s_K[j * d + k + 2]);
                    sim += bf16tof(s_Q[i * d + k + 3]) * bf16tof(s_K[j * d + k + 3]);
                }

                // Compute delta = dO[i,:] . V[j,:]
                float delta = 0.0f;
                #pragma unroll
                for (int k = 0; k < d; k += 4) {
                    delta += bf16tof(s_dO[i * d + k]) * bf16tof(s_V[j * d + k]);
                    delta += bf16tof(s_dO[i * d + k + 1]) * bf16tof(s_V[j * d + k + 1]);
                    delta += bf16tof(s_dO[i * d + k + 2]) * bf16tof(s_V[j * d + k + 2]);
                    delta += bf16tof(s_dO[i * d + k + 3]) * bf16tof(s_V[j * d + k + 3]);
                }

                float Pij = expf(sim * inv_sqrt_d - Li);
                s_P[i * BLOCK_N + j] = Pij;
                s_delta[i * BLOCK_N + j] = delta;
                local_di += Pij * delta;
            }
        }

        // Warp reduce local_di
        for (int i = warp_id; i < BLOCK_M; i += 4) {
            float tmp = local_di;
            // Simple warp sync not needed since each warp reduces its own
            __syncwarp();
            for (int stride = 16; stride > 0; stride >>= 1) {
                tmp += __shfl_down_sync(0xFFFFFFFF, tmp, stride);
            }
            if (lane_id == 0) {
                int gi = bm_start + i;
                if (gi < S) {
                    atomicAdd(&dAccD[bh * S + gi], tmp);
                }
            }
        }
        __syncthreads();

        // Update dV: dV[j,:] += sum_{i in bm, i>=j} P[i,j] * dO[i,:]
        for (int j = tid; j < BLOCK_N; j += blockDim.x) {
            int gj = bn_start + j;
            if (gj >= S) continue;
            for (int col = 0; col < d; col++) {
                float acc = 0.0f;
                for (int i = 0; i < BLOCK_M; i++) {
                    int gi = bm_start + i;
                    if (gi >= S || gj > gi) continue;
                    acc += s_P[i * BLOCK_N + j] * bf16tof(s_dO[i * d + col]);
                }
                atomicAdd(reinterpret_cast<float*>(&dV[(off * S + gj) * d + col]), acc);
            }
        }
    }
}

__global__ void mha_bwd_pass2_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const float* __restrict__ L,
    const __nv_bfloat16* __restrict__ dO,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    const float* __restrict__ D, // Final D values [num_bh, S]
    int S, int d, int num_heads, float inv_sqrt_d)
{
    int bh = blockIdx.x / gridDim.y;
    int n_tile = blockIdx.x % gridDim.y;
    int b = bh / num_heads;
    int h = bh % num_heads;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int bm_start = blockIdx.z * BLOCK_M;
    int bn_start = n_tile * BLOCK_N;

    uint64_t off = (uint64_t)b * num_heads + h;

    extern __shared__ char smem_char[];
    __align__(8) __nv_bfloat16* s_Q = (__nv_bfloat16*)smem_char;
    __align__(8) __nv_bfloat16* s_K = s_Q + BLOCK_M * d;
    __align__(8) __nv_bfloat16* s_V = s_K + BLOCK_N * d;
    __align__(8) __nv_bfloat16* s_dO = s_V + BLOCK_N * d;
    float* s_P = reinterpret_cast<float*>(reinterpret_cast<char*>(s_dO) + BLOCK_M * d * sizeof(__nv_bfloat16));
    float* s_delta = reinterpret_cast<float*>(reinterpret_cast<char*>(s_P) + BLOCK_M * BLOCK_N * sizeof(float));

    const __nv_bfloat16* qb = Q + off * S * d;
    const __nv_bfloat16* kb = K + off * S * d;
    const __nv_bfloat16* vb = V + off * S * d;
    const float* lb = L + off * S;
    const __nv_bfloat16* dob = dO + off * S * d;
    const float* Db = D + bh * S;

    // Load tiles
    for (int f = tid; f < BLOCK_M * d; f += blockDim.x) {
        int i = f / d;
        int col = f % d;
        int gi = bm_start + i;
        s_Q[f] = (gi < S) ? qb[gi * d + col] : f2bf16(0.f);
    }
    for (int f = tid; f < BLOCK_N * d; f += blockDim.x) {
        int j = f / d;
        int col = f % d;
        int gj = bn_start + j;
        s_K[f] = (gj < S) ? kb[gj * d + col] : f2bf16(0.f);
        s_V[f] = (gj < S) ? vb[gj * d + col] : f2bf16(0.f);
    }
    for (int f = tid; f < BLOCK_M * d; f += blockDim.x) {
        int i = f / d;
        int col = f % d;
        int gi = bm_start + i;
        s_dO[f] = (gi < S) ? dob[gi * d + col] : f2bf16(0.f);
    }
    __syncthreads();

    // Compute S, P, delta again then compute dQ and dK
    for (int i = warp_id; i < BLOCK_M; i += 4) {
        int gi = bm_start + i;
        if (gi >= S) break;
        float Di = Db[gi];
        float Li = lb[gi];
        
        float dq_acc[d] = {0};
        
        for (int j = 0; j < BLOCK_N; j++) {
            int gj = bn_start + j;
            if (gj > gi || gj >= S) continue;

            float sim = 0.0f, delta = 0.0f;
            for (int k = 0; k < d; k += 4) {
                sim += bf16tof(s_Q[i * d + k])     * bf16tof(s_K[j * d + k]);
                sim += bf16tof(s_Q[i * d + k + 1]) * bf16tof(s_K[j * d + k + 1]);
                sim += bf16tof(s_Q[i * d + k + 2]) * bf16tof(s_K[j * d + k + 2]);
                sim += bf16tof(s_Q[i * d + k + 3]) * bf16tof(s_K[j * d + k + 3]);
                delta += bf16tof(s_dO[i * d + k])     * bf16tof(s_V[j * d + k]);
                delta += bf16tof(s_dO[i * d + k + 1]) * bf16tof(s_V[j * d + k + 1]);
                delta += bf16tof(s_dO[i * d + k + 2]) * bf16tof(s_V[j * d + k + 2]);
                delta += bf16tof(s_dO[i * d + k + 3]) * bf16tof(s_V[j * d + k + 3]);
            }

            float Pij = expf(sim * inv_sqrt_d - Li);
            float dSij = Pij * (delta - Di);
            
            for (int k = 0; k < d; k += 4) {
                dq_acc[k]     += dSij * bf16tof(s_K[j * d + k]);
                dq_acc[k + 1] += dSij * bf16tof(s_K[j * d + k + 1]);
                dq_acc[k + 2] += dSij * bf16tof(s_K[j * d + k + 2]);
                dq_acc[k + 3] += dSij * bf16tof(s_K[j * d + k + 3]);
            }
        }

        // Store dQ
        __syncwarp();
        if (lane_id == 0) {
            for (int k = 0; k < d; k++) {
                atomicAdd(reinterpret_cast<float*>(&dQ[(off * S + gi) * d + k]), dq_acc[k]);
            }
        }
    }

    // Compute dK
    for (int j = tid; j < BLOCK_N; j += blockDim.x) {
        int gj = bn_start + j;
        if (gj >= S) continue;
        float dk_acc[d] = {0};
        for (int i = 0; i < BLOCK_M; i++) {
            int gi = bm_start + i;
            if (gi >= S || gj > gi) continue;
            float Di = Db[gi];
            float Li = lb[gi];

            float sim = 0.0f, delta = 0.0f;
            for (int k = 0; k < d; k += 4) {
                sim += bf16tof(s_Q[i * d + k])     * bf16tof(s_K[j * d + k]);
                sim += bf16tof(s_Q[i * d + k + 1]) * bf16tof(s_K[j * d + k + 1]);
                sim += bf16tof(s_Q[i * d + k + 2]) * bf16tof(s_K[j * d + k + 2]);
                sim += bf16tof(s_Q[i * d + k + 3]) * bf16tof(s_K[j * d + k + 3]);
                delta += bf16tof(s_dO[i * d + k])     * bf16tof(s_V[j * d + k]);
                delta += bf16tof(s_dO[i * d + k + 1]) * bf16tof(s_V[j * d + k + 1]);
                delta += bf16tof(s_dO[i * d + k + 2]) * bf16tof(s_V[j * d + k + 2]);
                delta += bf16tof(s_dO[i * d + k + 3]) * bf16tof(s_V[j * d + k + 3]);
            }
            float Pij = expf(sim * inv_sqrt_d - Li);
            float dSij = Pij * (delta - Di);
            for (int k = 0; k < d; k++) {
                dk_acc[k] += dSij * bf16tof(s_Q[i * d + k]);
            }
        }
        for (int k = 0; k < d; k++) {
            atomicAdd(reinterpret_cast<float*>(&dK[(off * S + gj) * d + k]), dk_acc[k]);
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
    int64_t ddim = Q.size(3);

    int64_t num_bh = B * H;

    const __nv_bfloat16* ptr_Q  = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* ptr_K  = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* ptr_V  = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const float* ptr_L          = static_cast<const float*>(L.data_ptr());
    const __nv_bfloat16* ptr_dO = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    __nv_bfloat16* ptr_dQ       = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* ptr_dK       = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* ptr_dV       = static_cast<__nv_bfloat16*>(dV.data_ptr());

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    // Zero outputs
    size_t out_bytes = num_bh * S * ddim * sizeof(__nv_bfloat16);
    CUDA_CHECK(cudaMemsetAsync(ptr_dQ, 0, out_bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(ptr_dK, 0, out_bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(ptr_dV, 0, out_bytes, stream));

    // Allocate D accumulator: [num_bh, S] fp32
    float* d_D = nullptr;
    size_t d_bytes = num_bh * S * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_D, d_bytes));
    CUDA_CHECK(cudaMemsetAsync(d_D, 0, d_bytes, stream));

    float inv_sqrt_d = 1.0f / sqrtf((float)ddim);
    int n_tiles_n = (S + BLOCK_N - 1) / BLOCK_N;
    int n_tiles_m = (S + BLOCK_M - 1) / BLOCK_M;
    int threads = 256;

    // Shared memory: Q(BLOCK_M*d), K(BLOCK_N*d), V(BLOCK_N*d), dO(BLOCK_M*d), P(BLOCK_M*BLOCK_N fp32), delta(BLOCK_M*BLOCK_N fp32)
    size_t smem_size = 2 * BLOCK_M * ddim * sizeof(__nv_bfloat16) 
                     + 2 * BLOCK_N * ddim * sizeof(__nv_bfloat16)
                     + 2 * BLOCK_M * BLOCK_N * sizeof(float);

    dim3 grid(num_bh * n_tiles_n, 1, n_tiles_m);

    mha_bwd_kernel_v2<<<grid, threads, smem_size, stream>>>(
        ptr_Q, ptr_K, ptr_V, ptr_L, ptr_dO,
        ptr_dQ, ptr_dK, ptr_dV, d_D,
        (int)S, (int)ddim, (int)H, inv_sqrt_d
    );
    CUDA_CHECK(cudaGetLastError());

    // Pass 2: compute dQ, dK using final D values
    mha_bwd_pass2_kernel<<<grid, threads, smem_size, stream>>>(
        ptr_Q, ptr_K, ptr_V, ptr_L, ptr_dO,
        ptr_dQ, ptr_dK, d_D,
        (int)S, (int)ddim, (int)H, inv_sqrt_d
    );
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_D));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_impl::run);

} // namespace mha_bwd_impl