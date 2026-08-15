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
constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
constexpr int THREADS = 256;
constexpr float SCALE = 0.088388348f;

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_bwd_d128_causal {

__global__ __launch_bounds__(256) void attn_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* dQ,
    __nv_bfloat16* dK,
    __nv_bfloat16* dV,
    int B, int H, int S)
{
    int b = blockIdx.z;
    int h = blockIdx.y;
    int i_block = blockIdx.x;
    int i_start = i_block * BM;

    if (i_start >= S) return;

    extern __shared__ char smem[];
    __nv_bfloat16* sQ  = (__nv_bfloat16*)smem;                      // BM*D bf16 = 16384B
    __nv_bfloat16* sK  = sQ + BM * D;                               // BN*D bf16 = 16384B
    __nv_bfloat16* sV  = sK + BN * D;                               // BN*D bf16 = 16384B
    __nv_bfloat16* sdO = sV + BN * D;                               // BM*D bf16 = 16384B
    float*        sS  = (float*)(sdO + BM * D);                     // BM*BN f32 = 16384B
    float*        sdP = sS + BM * BN;                               // BM*BN f32 = 16384B
    __nv_bfloat16* sP  = (__nv_bfloat16*)(sdP + BM * BN);           // BM*BN bf16 = 8192B
    __nv_bfloat16* sdS = sP + BM * BN;                              // BM*BN bf16 = 8192B
    float*        sL  = (float*)(sdS + BM * BN);                    // BM f32 = 256B
    float*        sD  = sL + BM;                                    // BM f32 = 256B
    // Total: 115200B

    // Reuse after MMAs: sK+sV (32768B) for dK/dQ float buf, sS+sdP (32768B) for dV float buf
    float* sdK_buf = (float*)sK;
    float* sdV_buf = sS;

    int warp_id = threadIdx.x / 32;
    const int64_t bh = (int64_t)(b * H + h);
    const int64_t qk_stride = bh * (int64_t)S * D;
    const int64_t l_stride  = bh * (int64_t)S;

    // Load Q_i, dO_i (once)
    for (int i = threadIdx.x; i < BM * D; i += THREADS) {
        int m = i / D, dd = i % D;
        int row = i_start + m;
        if (row < S) {
            sQ[i]  = Q[qk_stride + (int64_t)row * D + dd];
            sdO[i] = dO[qk_stride + (int64_t)row * D + dd];
        } else {
            sQ[i]  = __float2bfloat16(0.f);
            sdO[i] = __float2bfloat16(0.f);
        }
    }

    // Load L_i, compute D_i = rowsum(dO_i * O_i) (once)
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

    // dQ accumulator: 4x8=32 tiles, 4 per warp, 8 warps — BM/16 x D/16
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> dQ_frag[4];
    for (int t = 0; t < 4; t++) fill_fragment(dQ_frag[t], 0.0f);

    int max_j_block = ((i_start + BM - 1) < (S - 1)) ? ((i_start + BM - 1) / BN) : ((S - 1) / BN);

    for (int j_block = 0; j_block <= max_j_block; j_block++) {
        int j_start = j_block * BN;

        // Load K_j, V_j
        for (int i = threadIdx.x; i < BN * D; i += THREADS) {
            int n = i / D, dd = i % D;
            int row = j_start + n;
            if (row < S) {
                sK[i] = K[qk_stride + (int64_t)row * D + dd];
                sV[i] = V[qk_stride + (int64_t)row * D + dd];
            } else {
                sK[i] = __float2bfloat16(0.f);
                sV[i] = __float2bfloat16(0.f);
            }
        }
        __syncthreads();

        // S = Q @ K^T * scale -> sS (16 tiles: 4x4, 2 per warp)
        #pragma unroll
        for (int t = 0; t < 2; t++) {
            int tile_id = warp_id * 2 + t;
            int tr = tile_id / 4, tc = tile_id % 4;
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
            fill_fragment(c_frag, 0.0f);
            for (int kk = 0; kk < D; kk += WMMA_K) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> a_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, col_major> b_frag;
                load_matrix_sync(a_frag, &sQ[tr * WMMA_M * D + kk], D);
                load_matrix_sync(b_frag, &sK[tc * WMMA_N * D + kk], D);
                mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            for (int i = 0; i < c_frag.num_elements; i++) c_frag.x[i] *= SCALE;
            store_matrix_sync(&sS[tr * WMMA_M * BN + tc * WMMA_N], c_frag, BN, mem_row_major);
        }
        __syncthreads();

        // dP = dO @ V^T -> sdP (16 tiles: 4x4, 2 per warp)
        #pragma unroll
        for (int t = 0; t < 2; t++) {
            int tile_id = warp_id * 2 + t;
            int tr = tile_id / 4, tc = tile_id % 4;
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
            fill_fragment(c_frag, 0.0f);
            for (int kk = 0; kk < D; kk += WMMA_K) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> a_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, col_major> b_frag;
                load_matrix_sync(a_frag, &sdO[tr * WMMA_M * D + kk], D);
                load_matrix_sync(b_frag, &sV[tc * WMMA_N * D + kk], D);
                mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            store_matrix_sync(&sdP[tr * WMMA_M * BN + tc * WMMA_N], c_frag, BN, mem_row_major);
        }
        __syncthreads();

        // Elementwise: P = exp(S - L), dS = P * (dP - D)
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
            sP[m * BN + n]  = __float2bfloat16(p_val);
            sdS[m * BN + n] = __float2bfloat16(ds_val);
        }
        __syncthreads();

        // dQ += dS @ K (no scale yet; 32 tiles: 4x8, 4 per warp)
        #pragma unroll
        for (int t = 0; t < 4; t++) {
            int tile_id = warp_id * 4 + t;
            int tr = tile_id / 8, tc = tile_id % 8;
            for (int kk = 0; kk < BN; kk += WMMA_K) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> a_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> b_frag;
                load_matrix_sync(a_frag, &sdS[tr * WMMA_M * BN + kk], BN);
                load_matrix_sync(b_frag, &sK[kk * D + tc * WMMA_N], D);
                mma_sync(dQ_frag[t], a_frag, b_frag, dQ_frag[t]);
            }
        }

        // dK_contrib = dS^T @ Q (no scale yet; 32 tiles: 4x8, 4 per warp)
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> dK_frag[4];
        for (int t = 0; t < 4; t++) fill_fragment(dK_frag[t], 0.0f);
        #pragma unroll
        for (int t = 0; t < 4; t++) {
            int tile_id = warp_id * 4 + t;
            int tr = tile_id / 8, tc = tile_id % 8;
            for (int kk = 0; kk < BM; kk += WMMA_K) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, col_major> a_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> b_frag;
                load_matrix_sync(a_frag, &sdS[kk * BN + tr * WMMA_M], BN);
                load_matrix_sync(b_frag, &sQ[kk * D + tc * WMMA_N], D);
                mma_sync(dK_frag[t], a_frag, b_frag, dK_frag[t]);
            }
        }

        // dV_contrib = P^T @ dO (32 tiles: 4x8, 4 per warp)
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> dV_frag[4];
        for (int t = 0; t < 4; t++) fill_fragment(dV_frag[t], 0.0f);
        #pragma unroll
        for (int t = 0; t < 4; t++) {
            int tile_id = warp_id * 4 + t;
            int tr = tile_id / 8, tc = tile_id % 8;
            for (int kk = 0; kk < BM; kk += WMMA_K) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, col_major> a_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> b_frag;
                load_matrix_sync(a_frag, &sP[kk * BN + tr * WMMA_M], BN);
                load_matrix_sync(b_frag, &sdO[kk * D + tc * WMMA_N], D);
                mma_sync(dV_frag[t], a_frag, b_frag, dV_frag[t]);
            }
        }

        __syncthreads(); // All MMAs done; safe to reuse sK/sV/sS/sdP

        // Scale dK fragments
        for (int t = 0; t < 4; t++)
            for (int i = 0; i < dK_frag[t].num_elements; i++)
                dK_frag[t].x[i] *= SCALE;

        // Store dK (float) to sK+sV reuse, dV (float) to sS+sdP reuse
        #pragma unroll
        for (int t = 0; t < 4; t++) {
            int tile_id = warp_id * 4 + t;
            int tr = tile_id / 8, tc = tile_id % 8;
            store_matrix_sync(&sdK_buf[tr * WMMA_M * D + tc * WMMA_N], dK_frag[t], D, mem_row_major);
            store_matrix_sync(&sdV_buf[tr * WMMA_M * D + tc * WMMA_N], dV_frag[t], D, mem_row_major);
        }
        __syncthreads();

        // AtomicAdd dK, dV to global (bf16)
        for (int i = threadIdx.x; i < BN * D; i += THREADS) {
            int n = i / D, dd = i % D;
            int row = j_start + n;
            if (row < S) {
                atomicAdd(&dK[qk_stride + (int64_t)row * D + dd], __float2bfloat16(sdK_buf[i]));
                atomicAdd(&dV[qk_stride + (int64_t)row * D + dd], __float2bfloat16(sdV_buf[i]));
            }
        }
        __syncthreads();
    }

    // Scale dQ fragments
    for (int t = 0; t < 4; t++)
        for (int i = 0; i < dQ_frag[t].num_elements; i++)
            dQ_frag[t].x[i] *= SCALE;

    // Store dQ (float) to sK+sV reuse, then write to global as bf16
    float* sdQ_buf = (float*)sK;
    #pragma unroll
    for (int t = 0; t < 4; t++) {
        int tile_id = warp_id * 4 + t;
        int tr = tile_id / 8, tc = tile_id % 8;
        store_matrix_sync(&sdQ_buf[tr * WMMA_M * D + tc * WMMA_N], dQ_frag[t], D, mem_row_major);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < BM * D; i += THREADS) {
        int m = i / D, dd = i % D;
        int row = i_start + m;
        if (row < S) {
            dQ[qk_stride + (int64_t)row * D + dd] = __float2bfloat16(sdQ_buf[i]);
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

    // Zero dK, dV (accumulated via atomicAdd across CTAs)
    size_t dkv_bytes = (size_t)B * H * S * D * sizeof(__nv_bfloat16);
    CUDA_CHECK(cudaMemsetAsync(dK_ptr, 0, dkv_bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(dV_ptr, 0, dkv_bytes, stream));

    int i_blocks = ((int)S + BM - 1) / BM;
    dim3 grid(i_blocks, H, B);
    dim3 block(THREADS);

    int smem_size = 115200;
    CUDA_CHECK(cudaFuncSetAttribute(
        attn_bwd_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    attn_bwd_kernel<<<grid, block, smem_size, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
        dQ_ptr, dK_ptr, dV_ptr, B, H, (int)S);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128_causal::run);

}  // namespace mha_bwd_d128_causal