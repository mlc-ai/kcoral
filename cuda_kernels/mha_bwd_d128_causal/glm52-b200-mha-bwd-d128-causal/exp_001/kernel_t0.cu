#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", \
                cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

namespace mha_bwd_d128_causal {

constexpr int BQ = 64;
constexpr int BK = 64;
constexpr int D = 128;
constexpr int THREADS = 128;
constexpr float SCALE = 0.08838834764831843f; // 1/sqrt(128)
constexpr int B = 4;
constexpr int H = 48;

__global__ void attn_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ_out,
    float* __restrict__ dK_ws,
    float* __restrict__ dV_ws,
    int S)
{
    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;
    int q_blk = blockIdx.y;
    int q_start = q_blk * BQ;
    int tid = threadIdx.x;

    int64_t head_offset = (int64_t)b * H * S * D + (int64_t)h * S * D;
    int64_t lse_offset = (int64_t)b * H * S + (int64_t)h * S;

    const __nv_bfloat16* Q_base = Q + head_offset;
    const __nv_bfloat16* K_base = K + head_offset;
    const __nv_bfloat16* V_base = V + head_offset;
    const __nv_bfloat16* O_base = O + head_offset;
    const __nv_bfloat16* dO_base = dO + head_offset;
    const float* L_base = L + lse_offset;
    float* dK_base = dK_ws + head_offset;
    float* dV_base = dV_ws + head_offset;
    __nv_bfloat16* dQ_base = dQ_out + head_offset;

    extern __shared__ char smem[];
    __nv_bfloat16* Q_smem  = (__nv_bfloat16*)smem;                // BQ*D
    __nv_bfloat16* dO_smem = Q_smem + BQ * D;                     // BQ*D
    __nv_bfloat16* K_smem  = dO_smem + BQ * D;                    // BK*D
    __nv_bfloat16* V_smem  = K_smem + BK * D;                     // BK*D
    float* P_smem   = (float*)(V_smem + BK * D);                  // BQ*BK (reused for dS)
    float* dQ_smem  = P_smem + BQ * BK;                           // BQ*D
    float* D_smem   = dQ_smem + BQ * D;                           // BQ

    // Load Q block
    for (int i = tid; i < BQ * D; i += THREADS) {
        int row = i / D, col = i % D;
        int gidx = q_start + row;
        Q_smem[i] = (gidx < S) ? Q_base[(int64_t)gidx * D + col] : __float2bfloat16(0.0f);
    }
    // Load dO block
    for (int i = tid; i < BQ * D; i += THREADS) {
        int row = i / D, col = i % D;
        int gidx = q_start + row;
        dO_smem[i] = (gidx < S) ? dO_base[(int64_t)gidx * D + col] : __float2bfloat16(0.0f);
    }
    __syncthreads();

    // Compute D[i] = rowsum(dO[i] * O[i])
    for (int i = tid; i < BQ; i += THREADS) {
        int gidx = q_start + i;
        if (gidx >= S) {
            D_smem[i] = 0.0f;
        } else {
            float sum = 0.0f;
            for (int d = 0; d < D; d++) {
                float ov = __bfloat162float(O_base[(int64_t)gidx * D + d]);
                float dv = __bfloat162float(dO_smem[i * D + d]);
                sum += ov * dv;
            }
            D_smem[i] = sum;
        }
    }
    // Initialize dQ to 0
    for (int i = tid; i < BQ * D; i += THREADS) {
        dQ_smem[i] = 0.0f;
    }
    __syncthreads();

    // Iterate over K/V blocks (causal: k_blk from 0 to ceil((q_start+BQ)/BK)-1)
    int num_k_blocks = (q_start + BQ + BK - 1) / BK;
    for (int k_blk = 0; k_blk < num_k_blocks; k_blk++) {
        int k_start = k_blk * BK;

        // Load K block
        for (int i = tid; i < BK * D; i += THREADS) {
            int row = i / D, col = i % D;
            int gidx = k_start + row;
            K_smem[i] = (gidx < S) ? K_base[(int64_t)gidx * D + col] : __float2bfloat16(0.0f);
        }
        // Load V block
        for (int i = tid; i < BK * D; i += THREADS) {
            int row = i / D, col = i % D;
            int gidx = k_start + row;
            V_smem[i] = (gidx < S) ? V_base[(int64_t)gidx * D + col] : __float2bfloat16(0.0f);
        }
        __syncthreads();

        // Compute P = exp(S - L) where S = Q @ K^T * scale
        for (int i = tid; i < BQ * BK; i += THREADS) {
            int row = i / BK, col = i % BK;
            int q_idx = q_start + row;
            int k_idx = k_start + col;
            if (q_idx >= S || k_idx >= S || k_idx > q_idx) {
                P_smem[i] = 0.0f;
            } else {
                float sum = 0.0f;
                for (int d = 0; d < D; d++) {
                    sum += __bfloat162float(Q_smem[row * D + d]) *
                           __bfloat162float(K_smem[col * D + d]);
                }
                sum *= SCALE;
                P_smem[i] = expf(sum - L_base[q_idx]);
            }
        }
        __syncthreads();

        // dV += P^T @ dO  (BK x D) -> atomicAdd to dV_ws
        for (int i = tid; i < BK * D; i += THREADS) {
            int row = i / D, col = i % D;
            int k_idx = k_start + row;
            if (k_idx >= S) continue;
            float sum = 0.0f;
            for (int q = 0; q < BQ; q++) {
                sum += P_smem[q * BK + row] * __bfloat162float(dO_smem[q * D + col]);
            }
            if (sum != 0.0f) {
                atomicAdd(&dV_base[(int64_t)k_idx * D + col], sum);
            }
        }

        // Compute dP = dO @ V^T, then dS = P * (dP - D_i)
        // Overwrite P_smem with dS
        for (int i = tid; i < BQ * BK; i += THREADS) {
            int row = i / BK, col = i % BK;
            int q_idx = q_start + row;
            int k_idx = k_start + col;
            if (q_idx >= S || k_idx >= S || k_idx > q_idx) {
                P_smem[i] = 0.0f;
                continue;
            }
            float dp = 0.0f;
            for (int d = 0; d < D; d++) {
                dp += __bfloat162float(dO_smem[row * D + d]) *
                      __bfloat162float(V_smem[col * D + d]);
            }
            float p = P_smem[i];
            P_smem[i] = p * (dp - D_smem[row]);
        }
        __syncthreads();

        // dQ += dS @ K * scale (BQ x D)
        for (int i = tid; i < BQ * D; i += THREADS) {
            int row = i / D, col = i % D;
            int q_idx = q_start + row;
            if (q_idx >= S) continue;
            float sum = 0.0f;
            for (int k = 0; k < BK; k++) {
                sum += P_smem[row * BK + k] * __bfloat162float(K_smem[k * D + col]);
            }
            dQ_smem[i] += sum * SCALE;
        }

        // dK += dS^T @ Q * scale (BK x D) -> atomicAdd to dK_ws
        for (int i = tid; i < BK * D; i += THREADS) {
            int row = i / D, col = i % D;
            int k_idx = k_start + row;
            if (k_idx >= S) continue;
            float sum = 0.0f;
            for (int q = 0; q < BQ; q++) {
                sum += P_smem[q * BK + row] * __bfloat162float(Q_smem[q * D + col]);
            }
            if (sum != 0.0f) {
                atomicAdd(&dK_base[(int64_t)k_idx * D + col], sum * SCALE);
            }
        }
        __syncthreads();
    }

    // Write dQ to global
    for (int i = tid; i < BQ * D; i += THREADS) {
        int row = i / D, col = i % D;
        int q_idx = q_start + row;
        if (q_idx < S) {
            dQ_base[(int64_t)q_idx * D + col] = __float2bfloat16(dQ_smem[i]);
        }
    }
}

__global__ void convert_to_bf16(const float* src, __nv_bfloat16* dst, int64_t n) {
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += stride) {
        dst[idx] = __float2bfloat16(src[idx]);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {

    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t S = Q.size(2);
    int64_t total_elements = (int64_t)B * H * S * D;

    float* dK_ws = nullptr;
    float* dV_ws = nullptr;
    CUDA_CHECK(cudaMalloc(&dK_ws, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dV_ws, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMemset(dK_ws, 0, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMemset(dV_ws, 0, total_elements * sizeof(float)));

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    const __nv_bfloat16* Q_ptr  = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr  = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr  = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_ptr  = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_ptr          = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_ptr       = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr       = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr       = static_cast<__nv_bfloat16*>(dV.data_ptr());

    int num_q_blocks = (int)((S + BQ - 1) / BQ);
    dim3 grid(B * H, num_q_blocks);
    dim3 block(THREADS);

    int smem_size = (int)(
        BQ * D * sizeof(__nv_bfloat16) * 4 +  // Q, dO, K, V
        BQ * BK * sizeof(float) +              // P/dS
        BQ * D * sizeof(float) +               // dQ
        BQ * sizeof(float));                   // D

    cudaFuncSetAttribute(attn_bwd_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    attn_bwd_kernel<<<grid, block, smem_size, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
        dQ_ptr, dK_ws, dV_ws, (int)S);
    CUDA_CHECK(cudaGetLastError());

    int convert_threads = 256;
    int64_t convert_blocks = (total_elements + convert_threads - 1) / convert_threads;
    if (convert_blocks > 2147483647LL) convert_blocks = 2147483647LL;

    convert_to_bf16<<<(int)convert_blocks, convert_threads, 0, stream>>>(
        dK_ws, dK_ptr, total_elements);
    convert_to_bf16<<<(int)convert_blocks, convert_threads, 0, stream>>>(
        dV_ws, dV_ptr, total_elements);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFree(dK_ws));
    CUDA_CHECK(cudaFree(dV_ws));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128_causal::run);

}  // namespace mha_bwd_d128_causal