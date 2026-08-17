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
constexpr int THREADS = 256;
constexpr float SCALE = 0.08838834764831843f;
constexpr int B = 4;
constexpr int H = 48;

__device__ __forceinline__ float dot_bf16(
    const __nv_bfloat16* a, const __nv_bfloat16* b, int n) {
    float sum = 0.0f;
    #pragma unroll 8
    for (int i = 0; i < n; i += 2) {
        __nv_bfloat162 a2 = *reinterpret_cast<const __nv_bfloat162*>(&a[i]);
        __nv_bfloat162 b2 = *reinterpret_cast<const __nv_bfloat162*>(&b[i]);
        float2 af = __bfloat1622float2(a2);
        float2 bf = __bfloat1622float2(b2);
        sum += af.x * bf.x + af.y * bf.y;
    }
    return sum;
}

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
    float* P_smem   = (float*)(V_smem + BK * D);                  // BQ*BK
    float* dS_smem  = P_smem + BQ * BK;                           // BQ*BK (separate buffer)
    float* dQ_smem  = dS_smem + BQ * BK;                          // BQ*D
    float* D_smem   = dQ_smem + BQ * D;                           // BQ

    // Vectorized load of Q block (int4 = 8 bf16 elements)
    int total_v4 = BQ * D / 8;
    for (int i = tid; i < total_v4; i += THREADS) {
        int row = i / (D / 8);
        int col8 = i % (D / 8);
        int gidx = q_start + row;
        int4 val;
        if (gidx < S) {
            val = *reinterpret_cast<const int4*>(&Q_base[(int64_t)gidx * D + col8 * 8]);
        } else {
            val = make_int4(0, 0, 0, 0);
        }
        *reinterpret_cast<int4*>(&Q_smem[row * D + col8 * 8]) = val;
    }

    // Vectorized load of dO block
    for (int i = tid; i < total_v4; i += THREADS) {
        int row = i / (D / 8);
        int col8 = i % (D / 8);
        int gidx = q_start + row;
        int4 val;
        if (gidx < S) {
            val = *reinterpret_cast<const int4*>(&dO_base[(int64_t)gidx * D + col8 * 8]);
        } else {
            val = make_int4(0, 0, 0, 0);
        }
        *reinterpret_cast<int4*>(&dO_smem[row * D + col8 * 8]) = val;
    }
    __syncthreads();

    // Compute D[i] = rowsum(dO[i] * O[i])
    for (int i = tid; i < BQ; i += THREADS) {
        int gidx = q_start + i;
        if (gidx >= S) {
            D_smem[i] = 0.0f;
        } else {
            float sum = 0.0f;
            const __nv_bfloat16* optr = &O_base[(int64_t)gidx * D];
            const __nv_bfloat16* dptr = &dO_smem[i * D];
            #pragma unroll 8
            for (int d = 0; d < D; d += 2) {
                __nv_bfloat162 o2 = *reinterpret_cast<const __nv_bfloat162*>(&optr[d]);
                __nv_bfloat162 d2 = *reinterpret_cast<const __nv_bfloat162*>(&dptr[d]);
                float2 of = __bfloat1622float2(o2);
                float2 df = __bfloat1622float2(d2);
                sum += of.x * df.x + of.y * df.y;
            }
            D_smem[i] = sum;
        }
    }

    // Initialize dQ to 0
    int total_dq = BQ * D;
    for (int i = tid; i < total_dq; i += THREADS) {
        dQ_smem[i] = 0.0f;
    }
    __syncthreads();

    // Iterate over K/V blocks (causal: k_blk from 0 to ceil((q_start+BQ)/BK)-1)
    int num_k_blocks = (q_start + BQ + BK - 1) / BK;
    for (int k_blk = 0; k_blk < num_k_blocks; k_blk++) {
        int k_start = k_blk * BK;

        // Vectorized load of K block
        int total_kv = BK * D / 8;
        for (int i = tid; i < total_kv; i += THREADS) {
            int row = i / (D / 8);
            int col8 = i % (D / 8);
            int gidx = k_start + row;
            int4 val;
            if (gidx < S) {
                val = *reinterpret_cast<const int4*>(&K_base[(int64_t)gidx * D + col8 * 8]);
            } else {
                val = make_int4(0, 0, 0, 0);
            }
            *reinterpret_cast<int4*>(&K_smem[row * D + col8 * 8]) = val;
        }

        // Vectorized load of V block
        for (int i = tid; i < total_kv; i += THREADS) {
            int row = i / (D / 8);
            int col8 = i % (D / 8);
            int gidx = k_start + row;
            int4 val;
            if (gidx < S) {
                val = *reinterpret_cast<const int4*>(&V_base[(int64_t)gidx * D + col8 * 8]);
            } else {
                val = make_int4(0, 0, 0, 0);
            }
            *reinterpret_cast<int4*>(&V_smem[row * D + col8 * 8]) = val;
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
                float sum = dot_bf16(&Q_smem[row * D], &K_smem[col * D], D);
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
        __syncthreads(); // Ensure all reads of P_smem in dV are done before overwriting

        // Compute dP = dO @ V^T, then dS = P * (dP - D_i)
        // Store dS in separate buffer to avoid race with P reads
        for (int i = tid; i < BQ * BK; i += THREADS) {
            int row = i / BK, col = i % BK;
            int q_idx = q_start + row;
            int k_idx = k_start + col;
            if (q_idx >= S || k_idx >= S || k_idx > q_idx) {
                dS_smem[i] = 0.0f;
                continue;
            }
            float dp = dot_bf16(&dO_smem[row * D], &V_smem[col * D], D);
            float p = P_smem[i];
            dS_smem[i] = p * (dp - D_smem[row]);
        }
        __syncthreads();

        // dQ += dS @ K * scale (BQ x D)
        for (int i = tid; i < BQ * D; i += THREADS) {
            int row = i / D, col = i % D;
            int q_idx = q_start + row;
            if (q_idx >= S) continue;
            float sum = 0.0f;
            for (int k = 0; k < BK; k++) {
                sum += dS_smem[row * BK + k] * __bfloat162float(K_smem[k * D + col]);
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
                sum += dS_smem[q * BK + row] * __bfloat162float(Q_smem[q * D + col]);
            }
            if (sum != 0.0f) {
                atomicAdd(&dK_base[(int64_t)k_idx * D + col], sum * SCALE);
            }
        }
        __syncthreads();
    }

    // Write dQ to global (vectorized int4 store)
    for (int i = tid; i < BQ * D / 8; i += THREADS) {
        int row = i / (D / 8);
        int col8 = i % (D / 8);
        int q_idx = q_start + row;
        if (q_idx < S) {
            int4 out;
            float* src = &dQ_smem[row * D + col8 * 8];
            __nv_bfloat16* dst = &dQ_base[(int64_t)q_idx * D + col8 * 8];
            __nv_bfloat162 v0 = __floats2bfloat162_rn(src[0], src[1]);
            __nv_bfloat162 v1 = __floats2bfloat162_rn(src[2], src[3]);
            __nv_bfloat162 v2 = __floats2bfloat162_rn(src[4], src[5]);
            __nv_bfloat162 v3 = __floats2bfloat162_rn(src[6], src[7]);
            out = *reinterpret_cast<int4*>(&v0);
            // Pack 4 bf16x2 into int4
            *reinterpret_cast<__nv_bfloat162*>(&dst[0]) = v0;
            *reinterpret_cast<__nv_bfloat162*>(&dst[2]) = v1;
            *reinterpret_cast<__nv_bfloat162*>(&dst[4]) = v2;
            *reinterpret_cast<__nv_bfloat162*>(&dst[6]) = v3;
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
    CUDA_CHECK(cudaMemsetAsync(dK_ws, 0, total_elements * sizeof(float), 0));
    CUDA_CHECK(cudaMemsetAsync(dV_ws, 0, total_elements * sizeof(float), 0));

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

    // Shared memory: Q + dO + K + V (bf16) + P + dS (float) + dQ (float) + D (float)
    int smem_size = (int)(
        BQ * D * sizeof(__nv_bfloat16) * 4 +  // Q, dO, K, V
        BQ * BK * sizeof(float) * 2 +          // P, dS
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