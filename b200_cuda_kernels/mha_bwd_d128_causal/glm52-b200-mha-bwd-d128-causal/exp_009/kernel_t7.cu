#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
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
constexpr int THREADS = 256;
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

__device__ __forceinline__ __half bf16_to_half(__nv_bfloat16 x) {
    return __float2half(__bfloat162float(x));
}

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
    using namespace nvcuda::wmma;

    int bh = blockIdx.x;
    int kj_block = blockIdx.y;
    int kj = kj_block * TILE;

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
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int wr = warp_id / 2;
    int wcs64 = (warp_id % 2) * 2;
    int wcs128 = (warp_id % 2) * 4;
    const float scale = 1.0f / sqrtf(128.0f);

    extern __shared__ char smem_raw[];
    __half* sP_h = (__half*)smem_raw;
    float* sD = (float*)(sP_h + TILE * TILE);
    float* sL = sD + TILE;
    __half* sQ = (__half*)(sL + TILE);
    __half* sK = sQ + TILE * D;
    __half* sV = sK + TILE * D;
    __half* sdO = sV + TILE * D;
    float* staging = (float*)(sdO + TILE * D);

    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> dv_frag[4];
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> dk_frag[4];
    #pragma unroll
    for (int t = 0; t < 4; t++) {
        fill_fragment(dv_frag[t], 0.0f);
        fill_fragment(dk_frag[t], 0.0f);
    }

    int k_len = min(TILE, S - kj);
    if (k_len <= 0) return;

    for (int idx = tid; idx < TILE * D; idx += THREADS) {
        int row = idx / D, col = idx % D;
        sK[idx] = (row < k_len) ? bf16_to_half(K_[(kj + row) * D + col]) : __float2half(0.0f);
        sV[idx] = (row < k_len) ? bf16_to_half(V_[(kj + row) * D + col]) : __float2half(0.0f);
    }
    __syncthreads();

    for (int qi = kj; qi < S; qi += TILE) {
        int q_len = min(TILE, S - qi);

        for (int idx = tid; idx < TILE * D; idx += THREADS) {
            int row = idx / D, col = idx % D;
            sQ[idx] = (row < q_len) ? bf16_to_half(Q_[(qi + row) * D + col]) : __float2half(0.0f);
            sdO[idx] = (row < q_len) ? bf16_to_half(dO_[(qi + row) * D + col]) : __float2half(0.0f);
        }

        for (int row = tid / 4; row < TILE; row += THREADS / 4) {
            int sub = tid % 4;
            if (row < q_len) {
                float partial = 0.0f;
                for (int col = sub * 32; col < (sub + 1) * 32; col++) {
                    partial += __bfloat162float(dO_[(qi + row) * D + col]) *
                               __bfloat162float(O_[(qi + row) * D + col]);
                }
                partial += __shfl_xor_sync(0xFFFFFFFF, partial, 2);
                partial += __shfl_xor_sync(0xFFFFFFFF, partial, 1);
                if (sub == 0) {
                    sD[row] = partial;
                    sL[row] = L_[qi + row];
                }
            } else {
                if (sub == 0) { sD[row] = 0.0f; sL[row] = 0.0f; }
            }
        }
        __syncthreads();

        // S = Q @ K^T, then P = exp(S*scale - L) -> sP_h (half)
        #pragma unroll
        for (int t = 0; t < 2; t++) {
            int c = wcs64 + t;
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_frag;
            fill_fragment(s_frag, 0.0f);
            #pragma unroll
            for (int kk = 0; kk < D / WMMA_K; kk++) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, col_major> b_frag;
                load_matrix_sync(a_frag, sQ + wr * WMMA_M * D + kk * WMMA_K, D);
                load_matrix_sync(b_frag, sK + kk * WMMA_K + c * WMMA_N * D, D);
                mma_sync(s_frag, a_frag, b_frag, s_frag);
            }
            store_matrix_sync(staging + warp_id * 256, s_frag, 16, mem_row_major);
            __syncwarp();
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int idx = lane_id + i * 32;
                int row = idx / 16, col = idx % 16;
                int gi = wr * 16 + row, gj = c * 16 + col;
                float s_val = staging[warp_id * 256 + idx];
                float p_val = (gi >= q_len || gj >= k_len || qi + gi < kj + gj)
                              ? 0.0f : expf(s_val * scale - sL[gi]);
                sP_h[gi * TILE + gj] = __float2half(p_val);
            }
            __syncwarp();
        }
        __syncthreads();

        // dV += P^T @ dO (accumulate into persistent dv_frag)
        #pragma unroll
        for (int t = 0; t < 4; t++) {
            int c = wcs128 + t;
            #pragma unroll
            for (int kk = 0; kk < TILE / WMMA_K; kk++) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, col_major> a_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, row_major> b_frag;
                load_matrix_sync(a_frag, sP_h + wr * WMMA_M + kk * WMMA_K * TILE, TILE);
                load_matrix_sync(b_frag, sdO + kk * WMMA_K * D + c * WMMA_N, D);
                mma_sync(dv_frag[t], a_frag, b_frag, dv_frag[t]);
            }
        }
        __syncthreads();

        // dP = dO @ V^T, dS = P * (dP - D) -> sP_h (half, overwrite)
        #pragma unroll
        for (int t = 0; t < 2; t++) {
            int c = wcs64 + t;
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> dp_frag;
            fill_fragment(dp_frag, 0.0f);
            #pragma unroll
            for (int kk = 0; kk < D / WMMA_K; kk++) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, col_major> b_frag;
                load_matrix_sync(a_frag, sdO + wr * WMMA_M * D + kk * WMMA_K, D);
                load_matrix_sync(b_frag, sV + kk * WMMA_K + c * WMMA_N * D, D);
                mma_sync(dp_frag, a_frag, b_frag, dp_frag);
            }
            store_matrix_sync(staging + warp_id * 256, dp_frag, 16, mem_row_major);
            __syncwarp();
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int idx = lane_id + i * 32;
                int row = idx / 16, col = idx % 16;
                int gi = wr * 16 + row, gj = c * 16 + col;
                float p_val = __half2float(sP_h[gi * TILE + gj]);
                float dp_val = staging[warp_id * 256 + idx];
                float ds_val = (gi >= q_len || gj >= k_len || qi + gi < kj + gj)
                               ? 0.0f : p_val * (dp_val - sD[gi]);
                sP_h[gi * TILE + gj] = __float2half(ds_val);
            }
            __syncwarp();
        }
        __syncthreads();

        // dQ += dS @ K * scale (temporary, atomicAdd to global)
        #pragma unroll
        for (int t = 0; t < 4; t++) {
            int c = wcs128 + t;
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> dq_frag;
            fill_fragment(dq_frag, 0.0f);
            #pragma unroll
            for (int kk = 0; kk < TILE / WMMA_K; kk++) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, row_major> b_frag;
                load_matrix_sync(a_frag, sP_h + wr * WMMA_M * TILE + kk * WMMA_K, TILE);
                load_matrix_sync(b_frag, sK + kk * WMMA_K * D + c * WMMA_N, D);
                mma_sync(dq_frag, a_frag, b_frag, dq_frag);
            }
            #pragma unroll
            for (int i = 0; i < dq_frag.num_elements; i++) dq_frag.x[i] *= scale;
            store_matrix_sync(staging + warp_id * 256, dq_frag, 16, mem_row_major);
            __syncwarp();
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int idx = lane_id + i * 32;
                int row = idx / 16, col = idx % 16;
                int gr = qi + wr * 16 + row, gc = c * 16 + col;
                if (gr < S && gc < D)
                    atomicAdd(&dQ_f[gr * D + gc], staging[warp_id * 256 + idx]);
            }
            __syncwarp();
        }

        // dK += dS^T @ Q (accumulate into persistent dk_frag)
        #pragma unroll
        for (int t = 0; t < 4; t++) {
            int c = wcs128 + t;
            #pragma unroll
            for (int kk = 0; kk < TILE / WMMA_K; kk++) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, col_major> a_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, row_major> b_frag;
                load_matrix_sync(a_frag, sP_h + wr * WMMA_M + kk * WMMA_K * TILE, TILE);
                load_matrix_sync(b_frag, sQ + kk * WMMA_K * D + c * WMMA_N, D);
                mma_sync(dk_frag[t], a_frag, b_frag, dk_frag[t]);
            }
        }
        __syncthreads();
    }

    // Store dK * scale to global
    #pragma unroll
    for (int t = 0; t < 4; t++) {
        int c = wcs128 + t;
        #pragma unroll
        for (int i = 0; i < dk_frag[t].num_elements; i++) dk_frag[t].x[i] *= scale;
        store_matrix_sync(staging + warp_id * 256, dk_frag[t], 16, mem_row_major);
        __syncwarp();
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int idx = lane_id + i * 32;
            int row = idx / 16, col = idx % 16;
            int gr = kj + wr * 16 + row, gc = c * 16 + col;
            if (gr < S && gc < D)
                dK_[gr * D + gc] = __float2bfloat16(staging[warp_id * 256 + idx]);
        }
        __syncwarp();
    }

    // Store dV to global
    #pragma unroll
    for (int t = 0; t < 4; t++) {
        int c = wcs128 + t;
        store_matrix_sync(staging + warp_id * 256, dv_frag[t], 16, mem_row_major);
        __syncwarp();
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int idx = lane_id + i * 32;
            int row = idx / 16, col = idx % 16;
            int gr = kj + wr * 16 + row, gc = c * 16 + col;
            if (gr < S && gc < D)
                dV_[gr * D + gc] = __float2bfloat16(staging[warp_id * 256 + idx]);
        }
        __syncwarp();
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

    int64_t total_dq = (int64_t)B * H * S * d;
    float* dQ_float;
    CUDA_CHECK(cudaMalloc(&dQ_float, total_dq * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(dQ_float, 0, total_dq * sizeof(float), stream));

    // sP_h: 8192, sD+sL: 512, sQ+sK+sV+sdO: 65536, staging: 8192 = 82432
    size_t smem_size = (size_t)(TILE * TILE * 2 + 2 * TILE * 4 + 4 * TILE * D * 2 + 8 * 16 * 16 * 4);

    CUDA_CHECK(cudaFuncSetAttribute(attn_bwd_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    int num_k_tiles = (S + TILE - 1) / TILE;
    dim3 grid(B * H, num_k_tiles);
    attn_bwd_kernel<<<grid, THREADS, smem_size, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
        dQ_float, dK_ptr, dV_ptr, B, H, S);
    CUDA_CHECK(cudaGetLastError());

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