#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

namespace attn_bwd {

constexpr int D = 128;
constexpr int TILE = 64;
constexpr int THREADS = 128;

__global__ void attn_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    float* __restrict__ dQ_float,
    __nv_bfloat16* __restrict__ dK_out,
    __nv_bfloat16* __restrict__ dV_out,
    int B, int H, int S)
{
    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;

    int64_t base = (int64_t)(b * H + h) * S * D;
    int64_t lse_base = (int64_t)(b * H + h) * S;

    const __nv_bfloat16* Q_ = Q + base;
    const __nv_bfloat16* K_ = K + base;
    const __nv_bfloat16* V_ = V + base;
    const __nv_bfloat16* O_ = O + base;
    const __nv_bfloat16* dO_ = dO + base;
    const float* L_ = L + lse_base;
    float* dQ_f = dQ_float + base;
    __nv_bfloat16* dK_ = dK_out + base;
    __nv_bfloat16* dV_ = dV_out + base;

    int tid = threadIdx.x;
    const float scale = 0.0883883476f; // 1/sqrt(128)

    extern __shared__ char smem_raw[];
    float* dK_acc = (float*)smem_raw;           // TILE * D floats
    float* dV_acc = dK_acc + TILE * D;           // TILE * D floats
    float* sS = dV_acc + TILE * D;               // TILE * TILE floats
    float* sD = sS + TILE * TILE;                // TILE floats
    float* sL = sD + TILE;                       // TILE floats
    __nv_bfloat16* sK = (__nv_bfloat16*)(sL + TILE);  // TILE * D bf16
    __nv_bfloat16* sV = sK + TILE * D;           // TILE * D bf16
    __nv_bfloat16* sQ = sV + TILE * D;           // TILE * D bf16
    __nv_bfloat16* sdO = sQ + TILE * D;          // TILE * D bf16

    // Outer loop: key tiles
    for (int kj = 0; kj < S; kj += TILE) {
        int k_len = min(TILE, S - kj);

        // Init dK_acc, dV_acc
        for (int idx = tid; idx < k_len * D; idx += THREADS) {
            dK_acc[idx] = 0.0f;
            dV_acc[idx] = 0.0f;
        }
        __syncthreads();

        // Load K, V tiles
        for (int idx = tid; idx < k_len * D; idx += THREADS) {
            sK[idx] = K_[(kj + idx / D) * D + idx % D];
            sV[idx] = V_[(kj + idx / D) * D + idx % D];
        }
        __syncthreads();

        // Inner loop: query tiles (causal: qi >= kj)
        for (int qi = kj; qi < S; qi += TILE) {
            int q_len = min(TILE, S - qi);

            // Load Q, dO tiles
            for (int idx = tid; idx < q_len * D; idx += THREADS) {
                sQ[idx] = Q_[(qi + idx / D) * D + idx % D];
                sdO[idx] = dO_[(qi + idx / D) * D + idx % D];
            }
            __syncthreads();

            // Compute D[row] = rowsum(dO[row] * O[row]) and load L
            for (int row = tid; row < q_len; row += THREADS) {
                float d_val = 0.0f;
                for (int col = 0; col < D; col++) {
                    d_val += __bfloat162float(sdO[row * D + col]) *
                             __bfloat162float(O_[(qi + row) * D + col]);
                }
                sD[row] = d_val;
                sL[row] = L_[qi + row];
            }
            __syncthreads();

            // Step 1: S = Q @ K^T * scale (with causal mask)
            for (int idx = tid; idx < q_len * k_len; idx += THREADS) {
                int i = idx / k_len;
                int j = idx % k_len;
                if (qi + i < kj + j) {
                    sS[idx] = -INFINITY;
                } else {
                    float s_val = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < D; k += 2) {
                        float2 qf = __bfloat1622float2(*(__nv_bfloat162*)&sQ[i * D + k]);
                        float2 kf = __bfloat1622float2(*(__nv_bfloat162*)&sK[j * D + k]);
                        s_val += qf.x * kf.x + qf.y * kf.y;
                    }
                    sS[idx] = s_val * scale;
                }
            }
            __syncthreads();

            // Step 2: P = exp(S - L)
            for (int idx = tid; idx < q_len * k_len; idx += THREADS) {
                int i = idx / k_len;
                float s_val = sS[idx];
                sS[idx] = (s_val == -INFINITY) ? 0.0f : expf(s_val - sL[i]);
            }
            __syncthreads();

            // Step 3: dV_acc += P^T @ dO
            // dV[j][k] += sum_i P[i][j] * dO[i][k]
            for (int idx = tid; idx < k_len * D; idx += THREADS) {
                int j = idx / D;
                int k = idx % D;
                float dv_val = 0.0f;
                for (int i = 0; i < q_len; i++) {
                    dv_val += sS[i * k_len + j] * __bfloat162float(sdO[i * D + k]);
                }
                dV_acc[idx] += dv_val;
            }
            __syncthreads();

            // Step 4: dS = P * (dP - D), overwrite sS
            // dP[i][j] = sum_k dO[i][k] * V[j][k]
            for (int idx = tid; idx < q_len * k_len; idx += THREADS) {
                int i = idx / k_len;
                int j = idx % k_len;
                if (qi + i < kj + j) {
                    sS[idx] = 0.0f;
                } else {
                    float p_val = sS[idx];
                    float dp_val = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < D; k += 2) {
                        float2 dof = __bfloat1622float2(*(__nv_bfloat162*)&sdO[i * D + k]);
                        float2 vf = __bfloat1622float2(*(__nv_bfloat162*)&sV[j * D + k]);
                        dp_val += dof.x * vf.x + dof.y * vf.y;
                    }
                    sS[idx] = p_val * (dp_val - sD[i]);
                }
            }
            __syncthreads();

            // Step 5: dQ += dS @ K * scale (atomicAdd to dQ_float)
            // dQ[i][k] += scale * sum_j dS[i][j] * K[j][k]
            for (int idx = tid; idx < q_len * D; idx += THREADS) {
                int i = idx / D;
                int k = idx % D;
                float dq_val = 0.0f;
                for (int j = 0; j < k_len; j++) {
                    dq_val += sS[i * k_len + j] * __bfloat162float(sK[j * D + k]);
                }
                atomicAdd(&dQ_f[(qi + i) * D + k], dq_val * scale);
            }

            // Step 6: dK_acc += dS^T @ Q * scale
            // dK[j][k] += scale * sum_i dS[i][j] * Q[i][k]
            for (int idx = tid; idx < k_len * D; idx += THREADS) {
                int j = idx / D;
                int k = idx % D;
                float dk_val = 0.0f;
                for (int i = 0; i < q_len; i++) {
                    dk_val += sS[i * k_len + j] * __bfloat162float(sQ[i * D + k]);
                }
                dK_acc[idx] += dk_val * scale;
            }
            __syncthreads();
        }

        // Write dK, dV for this key tile
        for (int idx = tid; idx < k_len * D; idx += THREADS) {
            dK_[(kj + idx / D) * D + idx % D] = __float2bfloat16(dK_acc[idx]);
            dV_[(kj + idx / D) * D + idx % D] = __float2bfloat16(dV_acc[idx]);
        }
        __syncthreads();
    }
}

__global__ void convert_dq_kernel(
    const float* __restrict__ dQ_float,
    __nv_bfloat16* __restrict__ dQ,
    int64_t total_elements)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        dQ[idx] = __float2bfloat16(dQ_float[idx]);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV)
{
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int d = Q.size(3);

    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_ptr = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_ptr = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    // Allocate float accumulation buffer for dQ
    int64_t total_dq = (int64_t)B * H * S * d;
    float* dQ_float;
    CUDA_CHECK(cudaMalloc(&dQ_float, total_dq * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(dQ_float, 0, total_dq * sizeof(float), stream));

    // Shared memory size:
    // dK_acc + dV_acc: 2 * TILE * D * 4 = 65536
    // sS: TILE * TILE * 4 = 16384
    // sD + sL: 2 * TILE * 4 = 512
    // sK + sV + sQ + sdO: 4 * TILE * D * 2 = 65536
    // Total: 147968 bytes
    size_t smem_size = (size_t)(TILE * D * 4 * 2 + TILE * TILE * 4 + TILE * 4 * 2 + TILE * D * 2 * 4);

    CUDA_CHECK(cudaFuncSetAttribute(attn_bwd_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    int blocks = B * H;
    attn_bwd_kernel<<<blocks, THREADS, smem_size, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
        dQ_float, dK_ptr, dV_ptr, B, H, S);
    CUDA_CHECK(cudaGetLastError());

    // Convert dQ_float to bf16
    int convert_threads = 256;
    int64_t convert_blocks = (total_dq + convert_threads - 1) / convert_threads;
    convert_dq_kernel<<<(int)convert_blocks, convert_threads, 0, stream>>>(
        dQ_float, dQ_ptr, total_dq);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(dQ_float));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, attn_bwd::run);

}  // namespace attn_bwd