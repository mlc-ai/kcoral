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
constexpr int B_CONST = 4;
constexpr int H_CONST = 48;

__device__ __forceinline__ float dot128_bf16(
    const __nv_bfloat16* a, const __nv_bfloat16* b) {
    float sum = 0.0f;
    #pragma unroll
    for (int d = 0; d < D; d += 2) {
        __nv_bfloat162 a2 = *reinterpret_cast<const __nv_bfloat162*>(&a[d]);
        __nv_bfloat162 b2 = *reinterpret_cast<const __nv_bfloat162*>(&b[d]);
        float2 af = __bfloat1622float2(a2);
        float2 bf = __bfloat1622float2(b2);
        sum += af.x * bf.x + af.y * bf.y;
    }
    return sum;
}

// dQ kernel: each block computes dQ for one Q tile
__global__ void dQ_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ_out,
    int S)
{
    int bh = blockIdx.x;
    int b = bh / H_CONST;
    int h = bh % H_CONST;
    int q_blk = blockIdx.y;
    int q_start = q_blk * BQ;
    int tid = threadIdx.x;

    int64_t head_offset = (int64_t)b * H_CONST * S * D + (int64_t)h * S * D;
    int64_t lse_offset = (int64_t)b * H_CONST * S + (int64_t)h * S;

    const __nv_bfloat16* Q_base = Q + head_offset;
    const __nv_bfloat16* K_base = K + head_offset;
    const __nv_bfloat16* V_base = V + head_offset;
    const __nv_bfloat16* O_base = O + head_offset;
    const __nv_bfloat16* dO_base = dO + head_offset;
    const float* L_base = L + lse_offset;
    __nv_bfloat16* dQ_base = dQ_out + head_offset;

    extern __shared__ char smem[];
    __nv_bfloat16* Q_smem  = (__nv_bfloat16*)smem;
    __nv_bfloat16* dO_smem = Q_smem + BQ * D;
    __nv_bfloat16* K_smem  = dO_smem + BQ * D;
    __nv_bfloat16* V_smem  = K_smem + BK * D;
    float* P_smem   = (float*)(V_smem + BK * D);
    float* dQ_smem  = P_smem + BQ * BK;
    float* D_smem   = dQ_smem + BQ * D;

    // Load Q and dO
    for (int i = tid; i < BQ * D / 8; i += THREADS) {
        int row = i / (D / 8);
        int col8 = i % (D / 8);
        int gidx = q_start + row;
        if (gidx < S) {
            *reinterpret_cast<int4*>(&Q_smem[row * D + col8 * 8]) =
                *reinterpret_cast<const int4*>(&Q_base[(int64_t)gidx * D + col8 * 8]);
            *reinterpret_cast<int4*>(&dO_smem[row * D + col8 * 8]) =
                *reinterpret_cast<const int4*>(&dO_base[(int64_t)gidx * D + col8 * 8]);
        } else {
            *reinterpret_cast<int4*>(&Q_smem[row * D + col8 * 8]) = make_int4(0,0,0,0);
            *reinterpret_cast<int4*>(&dO_smem[row * D + col8 * 8]) = make_int4(0,0,0,0);
        }
    }
    __syncthreads();

    // Compute D[i] = rowsum(dO[i] * O[i])
    for (int i = tid; i < BQ; i += THREADS) {
        int gidx = q_start + i;
        if (gidx >= S) { D_smem[i] = 0.0f; continue; }
        float sum = 0.0f;
        const __nv_bfloat16* optr = &O_base[(int64_t)gidx * D];
        const __nv_bfloat16* dptr = &dO_smem[i * D];
        #pragma unroll
        for (int d = 0; d < D; d += 2) {
            __nv_bfloat162 o2 = *reinterpret_cast<const __nv_bfloat162*>(&optr[d]);
            __nv_bfloat162 d2 = *reinterpret_cast<const __nv_bfloat162*>(&dptr[d]);
            float2 of = __bfloat1622float2(o2);
            float2 df = __bfloat1622float2(d2);
            sum += of.x * df.x + of.y * df.y;
        }
        D_smem[i] = sum;
    }

    // Initialize dQ to 0
    for (int i = tid; i < BQ * D; i += THREADS) dQ_smem[i] = 0.0f;
    __syncthreads();

    int num_k_blocks = (q_start + BQ + BK - 1) / BK;
    for (int k_blk = 0; k_blk < num_k_blocks; k_blk++) {
        int k_start = k_blk * BK;

        // Load K and V
        for (int i = tid; i < BK * D / 8; i += THREADS) {
            int row = i / (D / 8), col8 = i % (D / 8);
            int gidx = k_start + row;
            if (gidx < S) {
                *reinterpret_cast<int4*>(&K_smem[row * D + col8 * 8]) =
                    *reinterpret_cast<const int4*>(&K_base[(int64_t)gidx * D + col8 * 8]);
                *reinterpret_cast<int4*>(&V_smem[row * D + col8 * 8]) =
                    *reinterpret_cast<const int4*>(&V_base[(int64_t)gidx * D + col8 * 8]);
            } else {
                *reinterpret_cast<int4*>(&K_smem[row * D + col8 * 8]) = make_int4(0,0,0,0);
                *reinterpret_cast<int4*>(&V_smem[row * D + col8 * 8]) = make_int4(0,0,0,0);
            }
        }
        __syncthreads();

        // P = exp(Q @ K^T * scale - L), then dS = P * (dO@V^T - D) in-place
        for (int i = tid; i < BQ * BK; i += THREADS) {
            int row = i / BK, col = i % BK;
            int q_idx = q_start + row, k_idx = k_start + col;
            if (q_idx >= S || k_idx >= S || k_idx > q_idx) {
                P_smem[i] = 0.0f; continue;
            }
            float s = dot128_bf16(&Q_smem[row * D], &K_smem[col * D]) * SCALE;
            float p = __expf(s - L_base[q_idx]);
            float dp = dot128_bf16(&dO_smem[row * D], &V_smem[col * D]);
            P_smem[i] = p * (dp - D_smem[row]);
        }
        __syncthreads();

        // dQ += dS @ K * scale
        for (int i = tid; i < BQ * D; i += THREADS) {
            int row = i / D, col = i % D;
            int q_idx = q_start + row;
            if (q_idx >= S) continue;
            float sum = 0.0f;
            for (int k = 0; k < BK; k++)
                sum += P_smem[row * BK + k] * __bfloat162float(K_smem[k * D + col]);
            dQ_smem[i] += sum * SCALE;
        }
        __syncthreads();
    }

    // Write dQ as bf16
    for (int i = tid; i < BQ * D / 8; i += THREADS) {
        int row = i / (D / 8), col8 = i % (D / 8);
        int q_idx = q_start + row;
        if (q_idx < S) {
            float* src = &dQ_smem[row * D + col8 * 8];
            __nv_bfloat16* dst = &dQ_base[(int64_t)q_idx * D + col8 * 8];
            __nv_bfloat162 v0 = __floats2bfloat162_rn(src[0], src[1]);
            __nv_bfloat162 v1 = __floats2bfloat162_rn(src[2], src[3]);
            __nv_bfloat162 v2 = __floats2bfloat162_rn(src[4], src[5]);
            __nv_bfloat162 v3 = __floats2bfloat162_rn(src[6], src[7]);
            *reinterpret_cast<int4*>(dst) = make_int4(
                *reinterpret_cast<int*>(&v0), *reinterpret_cast<int*>(&v1),
                *reinterpret_cast<int*>(&v2), *reinterpret_cast<int*>(&v3));
        }
    }
}

// dK_dV kernel: each block computes dK, dV for one K tile
__global__ void dK_dV_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dK_out,
    __nv_bfloat16* __restrict__ dV_out,
    int S)
{
    int bh = blockIdx.x;
    int b = bh / H_CONST;
    int h = bh % H_CONST;
    int k_blk = blockIdx.y;
    int k_start = k_blk * BK;
    int tid = threadIdx.x;

    int64_t head_offset = (int64_t)b * H_CONST * S * D + (int64_t)h * S * D;
    int64_t lse_offset = (int64_t)b * H_CONST * S + (int64_t)h * S;

    const __nv_bfloat16* Q_base = Q + head_offset;
    const __nv_bfloat16* K_base = K + head_offset;
    const __nv_bfloat16* V_base = V + head_offset;
    const __nv_bfloat16* O_base = O + head_offset;
    const __nv_bfloat16* dO_base = dO + head_offset;
    const float* L_base = L + lse_offset;
    __nv_bfloat16* dK_base = dK_out + head_offset;
    __nv_bfloat16* dV_base = dV_out + head_offset;

    extern __shared__ char smem[];
    __nv_bfloat16* K_smem  = (__nv_bfloat16*)smem;
    __nv_bfloat16* V_smem  = K_smem + BK * D;
    __nv_bfloat16* Q_smem  = V_smem + BK * D;
    __nv_bfloat16* dO_smem = Q_smem + BQ * D;
    float* P_smem   = (float*)(dO_smem + BQ * D);
    float* dK_smem  = P_smem + BQ * BK;
    float* dV_smem  = dK_smem + BK * D;
    float* D_smem   = dV_smem + BK * D;

    // Load K and V (persistent)
    for (int i = tid; i < BK * D / 8; i += THREADS) {
        int row = i / (D / 8), col8 = i % (D / 8);
        int gidx = k_start + row;
        if (gidx < S) {
            *reinterpret_cast<int4*>(&K_smem[row * D + col8 * 8]) =
                *reinterpret_cast<const int4*>(&K_base[(int64_t)gidx * D + col8 * 8]);
            *reinterpret_cast<int4*>(&V_smem[row * D + col8 * 8]) =
                *reinterpret_cast<const int4*>(&V_base[(int64_t)gidx * D + col8 * 8]);
        } else {
            *reinterpret_cast<int4*>(&K_smem[row * D + col8 * 8]) = make_int4(0,0,0,0);
            *reinterpret_cast<int4*>(&V_smem[row * D + col8 * 8]) = make_int4(0,0,0,0);
        }
    }
    __syncthreads();

    // Initialize dK, dV to 0
    for (int i = tid; i < BK * D; i += THREADS) {
        dK_smem[i] = 0.0f;
        dV_smem[i] = 0.0f;
    }
    __syncthreads();

    int num_q_blocks = (S + BQ - 1) / BQ;
    for (int q_blk = k_blk; q_blk < num_q_blocks; q_blk++) {
        int q_start = q_blk * BQ;

        // Load Q and dO
        for (int i = tid; i < BQ * D / 8; i += THREADS) {
            int row = i / (D / 8), col8 = i % (D / 8);
            int gidx = q_start + row;
            if (gidx < S) {
                *reinterpret_cast<int4*>(&Q_smem[row * D + col8 * 8]) =
                    *reinterpret_cast<const int4*>(&Q_base[(int64_t)gidx * D + col8 * 8]);
                *reinterpret_cast<int4*>(&dO_smem[row * D + col8 * 8]) =
                    *reinterpret_cast<const int4*>(&dO_base[(int64_t)gidx * D + col8 * 8]);
            } else {
                *reinterpret_cast<int4*>(&Q_smem[row * D + col8 * 8]) = make_int4(0,0,0,0);
                *reinterpret_cast<int4*>(&dO_smem[row * D + col8 * 8]) = make_int4(0,0,0,0);
            }
        }
        __syncthreads();

        // Compute D[i] = rowsum(dO[i] * O[i])
        for (int i = tid; i < BQ; i += THREADS) {
            int gidx = q_start + i;
            if (gidx >= S) { D_smem[i] = 0.0f; continue; }
            float sum = 0.0f;
            const __nv_bfloat16* optr = &O_base[(int64_t)gidx * D];
            const __nv_bfloat16* dptr = &dO_smem[i * D];
            #pragma unroll
            for (int d = 0; d < D; d += 2) {
                __nv_bfloat162 o2 = *reinterpret_cast<const __nv_bfloat162*>(&optr[d]);
                __nv_bfloat162 d2 = *reinterpret_cast<const __nv_bfloat162*>(&dptr[d]);
                float2 of = __bfloat1622float2(o2);
                float2 df = __bfloat1622float2(d2);
                sum += of.x * df.x + of.y * df.y;
            }
            D_smem[i] = sum;
        }
        __syncthreads();

        // Compute P = exp(Q @ K^T * scale - L)
        for (int i = tid; i < BQ * BK; i += THREADS) {
            int row = i / BK, col = i % BK;
            int q_idx = q_start + row, k_idx = k_start + col;
            if (q_idx >= S || k_idx >= S || k_idx > q_idx) {
                P_smem[i] = 0.0f; continue;
            }
            float s = dot128_bf16(&Q_smem[row * D], &K_smem[col * D]) * SCALE;
            P_smem[i] = __expf(s - L_base[q_idx]);
        }
        __syncthreads();

        // dV += P^T @ dO (BK x D)
        for (int i = tid; i < BK * D; i += THREADS) {
            int row = i / D, col = i % D;
            int k_idx = k_start + row;
            if (k_idx >= S) continue;
            float sum = 0.0f;
            for (int q = 0; q < BQ; q++)
                sum += P_smem[q * BK + row] * __bfloat162float(dO_smem[q * D + col]);
            dV_smem[i] += sum;
        }
        __syncthreads();

        // Compute dS = P * (dO@V^T - D) in-place
        for (int i = tid; i < BQ * BK; i += THREADS) {
            int row = i / BK, col = i % BK;
            int q_idx = q_start + row, k_idx = k_start + col;
            if (q_idx >= S || k_idx >= S || k_idx > q_idx) {
                P_smem[i] = 0.0f; continue;
            }
            float dp = dot128_bf16(&dO_smem[row * D], &V_smem[col * D]);
            float p = P_smem[i];
            P_smem[i] = p * (dp - D_smem[row]);
        }
        __syncthreads();

        // dK += dS^T @ Q * scale (BK x D)
        for (int i = tid; i < BK * D; i += THREADS) {
            int row = i / D, col = i % D;
            int k_idx = k_start + row;
            if (k_idx >= S) continue;
            float sum = 0.0f;
            for (int q = 0; q < BQ; q++)
                sum += P_smem[q * BK + row] * __bfloat162float(Q_smem[q * D + col]);
            dK_smem[i] += sum * SCALE;
        }
        __syncthreads();
    }

    // Write dK and dV as bf16
    for (int i = tid; i < BK * D / 8; i += THREADS) {
        int row = i / (D / 8), col8 = i % (D / 8);
        int k_idx = k_start + row;
        if (k_idx < S) {
            float* dk_src = &dK_smem[row * D + col8 * 8];
            float* dv_src = &dV_smem[row * D + col8 * 8];
            __nv_bfloat16* dk_dst = &dK_base[(int64_t)k_idx * D + col8 * 8];
            __nv_bfloat16* dv_dst = &dV_base[(int64_t)k_idx * D + col8 * 8];
            __nv_bfloat162 dk0 = __floats2bfloat162_rn(dk_src[0], dk_src[1]);
            __nv_bfloat162 dk1 = __floats2bfloat162_rn(dk_src[2], dk_src[3]);
            __nv_bfloat162 dk2 = __floats2bfloat162_rn(dk_src[4], dk_src[5]);
            __nv_bfloat162 dk3 = __floats2bfloat162_rn(dk_src[6], dk_src[7]);
            __nv_bfloat162 dv0 = __floats2bfloat162_rn(dv_src[0], dv_src[1]);
            __nv_bfloat162 dv1 = __floats2bfloat162_rn(dv_src[2], dv_src[3]);
            __nv_bfloat162 dv2 = __floats2bfloat162_rn(dv_src[4], dv_src[5]);
            __nv_bfloat162 dv3 = __floats2bfloat162_rn(dv_src[6], dv_src[7]);
            *reinterpret_cast<int4*>(dk_dst) = make_int4(
                *reinterpret_cast<int*>(&dk0), *reinterpret_cast<int*>(&dk1),
                *reinterpret_cast<int*>(&dk2), *reinterpret_cast<int*>(&dk3));
            *reinterpret_cast<int4*>(dv_dst) = make_int4(
                *reinterpret_cast<int*>(&dv0), *reinterpret_cast<int*>(&dv1),
                *reinterpret_cast<int*>(&dv2), *reinterpret_cast<int*>(&dv3));
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {

    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int64_t S = Q.size(2);

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
    int num_k_blocks = (int)((S + BK - 1) / BK);

    {
        dim3 grid(B_CONST * H_CONST, num_q_blocks);
        dim3 block(THREADS);
        int smem_size = (int)(
            BQ * D * sizeof(__nv_bfloat16) * 4 +
            BQ * BK * sizeof(float) +
            BQ * D * sizeof(float) +
            BQ * sizeof(float));
        cudaFuncSetAttribute(dQ_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        dQ_kernel<<<grid, block, smem_size, stream>>>(
            Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr, (int)S);
        CUDA_CHECK(cudaGetLastError());
    }

    {
        dim3 grid(B_CONST * H_CONST, num_k_blocks);
        dim3 block(THREADS);
        int smem_size = (int)(
            BK * D * sizeof(__nv_bfloat16) * 2 +
            BQ * D * sizeof(__nv_bfloat16) * 2 +
            BQ * BK * sizeof(float) +
            BK * D * sizeof(float) * 2 +
            BQ * sizeof(float));
        cudaFuncSetAttribute(dK_dV_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        dK_dV_kernel<<<grid, block, smem_size, stream>>>(
            Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr, (int)S);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128_causal::run);

}  // namespace mha_bwd_d128_causal