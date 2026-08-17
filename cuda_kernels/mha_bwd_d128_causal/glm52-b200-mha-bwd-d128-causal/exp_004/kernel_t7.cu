#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda::wmma;

constexpr int D = 128;
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 8; // TF32 uses K=8
constexpr int THREADS = 256;
constexpr float SCALE = 0.08838834764831843f;

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_bwd_d128_causal {

__global__ void convert_f32_to_bf16(const float* __restrict__ src,
                                     __nv_bfloat16* __restrict__ dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __float2bfloat16(src[idx]);
}

__global__ __launch_bounds__(256) void attn_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    float* __restrict__ dQ_float,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int B, int H, int S)
{
    int b = blockIdx.z;
    int h = blockIdx.y;
    int j_block = blockIdx.x;
    int j_start = j_block * BN;

    if (j_start >= S) return;

    extern __shared__ char smem[];
    float* sQ  = (float*)smem;                     // BM*D f32 = 32768B
    float* sK  = sQ + BM * D;                      // BN*D f32 = 32768B
    float* sV  = sK + BN * D;                      // BN*D f32 = 32768B
    float* sdO = sV + BN * D;                      // BM*D f32 = 32768B
    float* sS  = sdO + BM * D;                     // BM*BN f32 = 16384B
    float* sdP = sS + BM * BN;                     // BM*BN f32 = 16384B
    float* sP  = sdP + BM * BN;                    // BM*BN f32 = 16384B
    float* sdS = sP + BM * BN;                     // BM*BN f32 = 16384B
    float* sL  = sdS + BM * BN;                    // BM f32 = 256B
    float* sD  = sL + BM;                          // BM f32 = 256B

    // Reuse buffers
    float* sdQ_buf = sS;       // Reuses sS+sdP (32768B = BM*D float)
    float* sdK_out = (float*)sQ;  // Reuses sQ+sK (65536B = BN*D float)
    float* sdV_out = (float*)sV;  // Reuses sV+sdO (65536B = BN*D float)

    int warp_id = threadIdx.x / 32;
    const int64_t bh = (int64_t)(b * H + h);
    const int64_t qk_stride = bh * (int64_t)S * D;
    const int64_t l_stride  = bh * (int64_t)S;

    // Load K_j, V_j
    for (int i = threadIdx.x; i < BN * D; i += THREADS) {
        int n = i / D, dd = i % D;
        int row = j_start + n;
        if (row < S) {
            sK[i] = __bfloat162float(K[qk_stride + (int64_t)row * D + dd]);
            sV[i] = __bfloat162float(V[qk_stride + (int64_t)row * D + dd]);
        } else {
            sK[i] = 0.f;
            sV[i] = 0.f;
        }
    }
    __syncthreads();

    // dK, dV accumulator fragments: BN/16 x D/16 = 4x8 = 32 tiles, 4 per warp
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> dK_frag[4];
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> dV_frag[4];
    for (int t = 0; t < 4; t++) {
        fill_fragment(dK_frag[t], 0.0f);
        fill_fragment(dV_frag[t], 0.0f);
    }

    for (int i_block = 0; i_block < S; i_block += BM) {
        int i_start = i_block;
        if (i_start + BM <= j_start) continue;

        // Load Q_i, dO_i
        for (int i = threadIdx.x; i < BM * D; i += THREADS) {
            int m = i / D, dd = i % D;
            int row = i_start + m;
            if (row < S) {
                sQ[i]  = __bfloat162float(Q[qk_stride + (int64_t)row * D + dd]);
                sdO[i] = __bfloat162float(dO[qk_stride + (int64_t)row * D + dd]);
            } else {
                sQ[i]  = 0.f;
                sdO[i] = 0.f;
            }
        }

        // Load L_i, compute D_i = rowsum(dO * O)
        if (threadIdx.x < BM) {
            int m = threadIdx.x;
            int row = i_start + m;
            if (row < S) {
                sL[m] = L[l_stride + row];
                float d_val = 0.f;
                const __nv_bfloat16* dO_row = &dO[qk_stride + (int64_t)row * D];
                const __nv_bfloat16* O_row  = &O[qk_stride + (int64_t)row * D];
                for (int dd = 0; dd < D; dd++) {
                    d_val += __bfloat162float(dO_row[dd]) * __bfloat162float(O_row[dd]);
                }
                sD[m] = d_val;
            } else {
                sL[m] = 0.f;
                sD[m] = 0.f;
            }
        }
        __syncthreads();

        // S = Q @ K^T * scale -> sS
        #pragma unroll
        for (int t = 0; t < 2; t++) {
            int tile_id = warp_id * 2 + t;
            int tr = tile_id / 4, tc = tile_id % 4;
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
            fill_fragment(c_frag, 0.0f);
            for (int kk = 0; kk < D; kk += WMMA_K) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, precision::tf32, row_major> a_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, precision::tf32, col_major> b_frag;
                load_matrix_sync(a_frag, &sQ[tr * WMMA_M * D + kk], D);
                load_matrix_sync(b_frag, &sK[tc * WMMA_N * D + kk], D);
                mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            for (int i = 0; i < c_frag.num_elements; i++) c_frag.x[i] *= SCALE;
            store_matrix_sync(&sS[tr * WMMA_M * BN + tc * WMMA_N], c_frag, BN, mem_row_major);
        }
        __syncthreads();

        // dP = dO @ V^T -> sdP
        #pragma unroll
        for (int t = 0; t < 2; t++) {
            int tile_id = warp_id * 2 + t;
            int tr = tile_id / 4, tc = tile_id % 4;
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
            fill_fragment(c_frag, 0.0f);
            for (int kk = 0; kk < D; kk += WMMA_K) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, precision::tf32, row_major> a_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, precision::tf32, col_major> b_frag;
                load_matrix_sync(a_frag, &sdO[tr * WMMA_M * D + kk], D);
                load_matrix_sync(b_frag, &sV[tc * WMMA_N * D + kk], D);
                mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            store_matrix_sync(&sdP[tr * WMMA_M * BN + tc * WMMA_N], c_frag, BN, mem_row_major);
        }
        __syncthreads();

        // Element-wise: P = exp(S - L), dS = P * (dP - D)
        for (int i = threadIdx.x; i < BM * BN; i += THREADS) {
            int m = i / BN, n = i % BN;
            int q_row = i_start + m;
            int kv_row = j_start + n;
            float p_val, ds_val;
            if (q_row >= kv_row && q_row < S && kv_row < S) {
                float s_val  = sS[m * BN + n];
                float dp_val = sdP[m * BN + n];
                float l_val  = sL[m];
                float d_val  = sD[m];
                p_val  = expf(s_val - l_val);
                ds_val = p_val * (dp_val - d_val);
            } else {
                p_val  = 0.f;
                ds_val = 0.f;
            }
            sP[m * BN + n]  = p_val;
            sdS[m * BN + n] = ds_val;
        }
        __syncthreads();

        // dQ = dS @ K * scale -> store to sdQ_buf (float), then atomicAdd to dQ_float
        #pragma unroll
        for (int t = 0; t < 4; t++) {
            int tile_id = warp_id * 4 + t;
            int tr = tile_id / 8, tc = tile_id % 8;
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> dq_frag;
            fill_fragment(dq_frag, 0.0f);
            for (int kk = 0; kk < BN; kk += WMMA_K) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, precision::tf32, row_major> a_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, precision::tf32, row_major> b_frag;
                load_matrix_sync(a_frag, &sdS[tr * WMMA_M * BN + kk], BN);
                load_matrix_sync(b_frag, &sK[kk * D + tc * WMMA_N], D);
                mma_sync(dq_frag, a_frag, b_frag, dq_frag);
            }
            for (int i = 0; i < dq_frag.num_elements; i++) dq_frag.x[i] *= SCALE;
            store_matrix_sync(&sdQ_buf[tr * WMMA_M * D + tc * WMMA_N], dq_frag, D, mem_row_major);
        }
        __syncthreads();

        // AtomicAdd dQ to float workspace
        for (int i = threadIdx.x; i < BM * D; i += THREADS) {
            int m = i / D, dd = i % D;
            int row = i_start + m;
            if (row < S) {
                float val = sdQ_buf[i];
                if (val != 0.f) {
                    atomicAdd(&dQ_float[qk_stride + (int64_t)row * D + dd], val);
                }
            }
        }

        // dK += dS^T @ Q * scale
        #pragma unroll
        for (int t = 0; t < 4; t++) {
            int tile_id = warp_id * 4 + t;
            int tr = tile_id / 8, tc = tile_id % 8;
            for (int kk = 0; kk < BM; kk += WMMA_K) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, precision::tf32, col_major> a_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, precision::tf32, row_major> b_frag;
                load_matrix_sync(a_frag, &sdS[kk * BN + tr * WMMA_M], BN);
                load_matrix_sync(b_frag, &sQ[kk * D + tc * WMMA_N], D);
                mma_sync(dK_frag[t], a_frag, b_frag, dK_frag[t]);
            }
        }

        // dV += P^T @ dO
        #pragma unroll
        for (int t = 0; t < 4; t++) {
            int tile_id = warp_id * 4 + t;
            int tr = tile_id / 8, tc = tile_id % 8;
            for (int kk = 0; kk < BM; kk += WMMA_K) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, precision::tf32, col_major> a_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, precision::tf32, row_major> b_frag;
                load_matrix_sync(a_frag, &sP[kk * BN + tr * WMMA_M], BN);
                load_matrix_sync(b_frag, &sdO[kk * D + tc * WMMA_N], D);
                mma_sync(dV_frag[t], a_frag, b_frag, dV_frag[t]);
            }
        }

        __syncthreads();
    }

    // Scale dK fragments
    for (int t = 0; t < 4; t++)
        for (int i = 0; i < dK_frag[t].num_elements; i++)
            dK_frag[t].x[i] *= SCALE;

    // Store dK, dV fragments to shared (reusing sQ/sV space as float buffers)
    for (int t = 0; t < 4; t++) {
        int tile_id = warp_id * 4 + t;
        int tr = tile_id / 8, tc = tile_id % 8;
        store_matrix_sync(&sdK_out[tr * WMMA_M * D + tc * WMMA_N], dK_frag[t], D, mem_row_major);
        store_matrix_sync(&sdV_out[tr * WMMA_M * D + tc * WMMA_N], dV_frag[t], D, mem_row_major);
    }
    __syncthreads();

    // Copy dK, dV to global (float -> bf16)
    for (int i = threadIdx.x; i < BN * D; i += THREADS) {
        int n = i / D, dd = i % D;
        int row = j_start + n;
        if (row < S) {
            dK[qk_stride + (int64_t)row * D + dd] = __float2bfloat16(sdK_out[i]);
            dV[qk_stride + (int64_t)row * D + dd] = __float2bfloat16(sdV_out[i]);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    const int B = 4;
    const int H = 48;
    int64_t S = Q.size(2);

    const __nv_bfloat16* Q_ptr  = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr  = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr  = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_ptr  = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float*         L_ptr  = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_ptr       = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr       = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr       = static_cast<__nv_bfloat16*>(dV.data_ptr());

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    // Allocate float workspace for dQ (accumulated via float atomicAdd)
    size_t dQ_elem = (size_t)B * H * S * D;
    float* dQ_float = nullptr;
    CUDA_CHECK(cudaMalloc(&dQ_float, dQ_elem * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(dQ_float, 0, dQ_elem * sizeof(float), stream));

    int j_blocks = ((int)S + BN - 1) / BN;
    dim3 grid(j_blocks, H, B);
    dim3 block(THREADS);

    int smem_size = 196608; // 192KB
    CUDA_CHECK(cudaFuncSetAttribute(
        attn_bwd_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    attn_bwd_kernel<<<grid, block, smem_size, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
        dQ_float, dK_ptr, dV_ptr, B, H, (int)S);

    // Convert dQ from float to bf16
    int conv_threads = 256;
    int conv_blocks = (dQ_elem + conv_threads - 1) / conv_threads;
    convert_f32_to_bf16<<<conv_blocks, conv_threads, 0, stream>>>(dQ_float, dQ_ptr, (int)dQ_elem);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(dQ_float));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128_causal::run);

}  // namespace mha_bwd_d128_causal