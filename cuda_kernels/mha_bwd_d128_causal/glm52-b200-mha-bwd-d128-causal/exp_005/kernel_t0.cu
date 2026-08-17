#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cmath>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda;

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", \
                cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

namespace attn_bwd {

constexpr int Br = 32;
constexpr int Bc = 32;
constexpr int D = 128;
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int NUM_WARPS = 4;
constexpr int NUM_THREADS = 128;

constexpr int K_OFF       = 0;
constexpr int V_OFF       = Bc * D * 2;
constexpr int Q_OFF       = V_OFF + Br * D * 2;
constexpr int DO_OFF      = Q_OFF + Br * D * 2;
constexpr int S_OFF       = DO_OFF + Br * D * 2;
constexpr int DP_OFF      = S_OFF + Br * Bc * 4;
constexpr int P_BF16_OFF  = DP_OFF + Br * Bc * 4;
constexpr int DS_BF16_OFF = P_BF16_OFF + Br * Bc * 2;
constexpr int L_OFF       = DS_BF16_OFF + Br * Bc * 2;
constexpr int D_OFF       = L_OFF + Br * 4;
constexpr int SMEM_SIZE   = D_OFF + Br * 4;

__global__ void attn_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    float* __restrict__ dQ_fp32,
    __nv_bfloat16* __restrict__ dK_g,
    __nv_bfloat16* __restrict__ dV_g,
    int S, int H) {

    int bh = blockIdx.x;
    int j = blockIdx.y;
    int b = bh / H;
    int h = bh % H;
    int num_blocks = (S + Bc - 1) / Bc;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* K_smem  = reinterpret_cast<__nv_bfloat16*>(smem_raw + K_OFF);
    __nv_bfloat16* V_smem  = reinterpret_cast<__nv_bfloat16*>(smem_raw + V_OFF);
    __nv_bfloat16* Q_smem  = reinterpret_cast<__nv_bfloat16*>(smem_raw + Q_OFF);
    __nv_bfloat16* dO_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + DO_OFF);
    float* S_smem          = reinterpret_cast<float*>(smem_raw + S_OFF);
    float* dP_smem         = reinterpret_cast<float*>(smem_raw + DP_OFF);
    __nv_bfloat16* P_bf16  = reinterpret_cast<__nv_bfloat16*>(smem_raw + P_BF16_OFF);
    __nv_bfloat16* dS_bf16 = reinterpret_cast<__nv_bfloat16*>(smem_raw + DS_BF16_OFF);
    float* L_smem          = reinterpret_cast<float*>(smem_raw + L_OFF);
    float* D_smem          = reinterpret_cast<float*>(smem_raw + D_OFF);
    float* dQ_stage        = reinterpret_cast<float*>(smem_raw + S_OFF);
    float* out_stage       = reinterpret_cast<float*>(smem_raw + K_OFF);

    const float scale = 0.08838834764831845f; // 1/sqrt(128)
    int64_t bh_off = (int64_t)(b * H + h) * S * D;
    int64_t bh_off_L = (int64_t)(b * H + h) * S;

    // Load K_j, V_j
    for (int idx = tid; idx < Bc * D; idx += NUM_THREADS) {
        int r = idx / D;
        int c = idx % D;
        int k_idx = j * Bc + r;
        if (k_idx < S) {
            K_smem[idx] = K[bh_off + (int64_t)k_idx * D + c];
            V_smem[idx] = V[bh_off + (int64_t)k_idx * D + c];
        } else {
            K_smem[idx] = __float2bfloat16(0.0f);
            V_smem[idx] = __float2bfloat16(0.0f);
        }
    }
    __syncthreads();

    int m_tile_dk = warp_id / 2;
    int n_start_dk = (warp_id % 2) * 4;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> dK_frag[4];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> dV_frag[4];
    for (int n = 0; n < 4; n++) {
        wmma::fill_fragment(dK_frag[n], 0.0f);
        wmma::fill_fragment(dV_frag[n], 0.0f);
    }

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_row;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_col;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_row;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> a_col;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> dQ_frag[4];

    for (int i = j; i < num_blocks; i++) {
        // Load Q_i, dO_i
        for (int idx = tid; idx < Br * D; idx += NUM_THREADS) {
            int r = idx / D;
            int c = idx % D;
            int q_idx = i * Br + r;
            if (q_idx < S) {
                Q_smem[idx] = Q[bh_off + (int64_t)q_idx * D + c];
                dO_smem[idx] = dO[bh_off + (int64_t)q_idx * D + c];
            } else {
                Q_smem[idx] = __float2bfloat16(0.0f);
                dO_smem[idx] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        // Load L_i
        if (tid < Br) {
            int r = tid;
            int q_idx = i * Br + r;
            L_smem[r] = (q_idx < S) ? L[bh_off_L + q_idx] : 0.0f;
        }

        // Compute D_i = rowsum(dO_i * O_i)
        for (int r = warp_id * 8; r < (warp_id + 1) * 8; r++) {
            int q_idx = i * Br + r;
            float dot = 0.0f;
            if (q_idx < S) {
                int base_c = lane * 4;
                const __nv_bfloat16* O_ptr = &O[bh_off + (int64_t)q_idx * D];
                for (int e = 0; e < 4; e++) {
                    float dv = __bfloat162float(dO_smem[r * D + base_c + e]);
                    float ov = __bfloat162float(O_ptr[base_c + e]);
                    dot += dv * ov;
                }
            }
            for (int delta = 16; delta > 0; delta >>= 1)
                dot += __shfl_xor_sync(0xffffffff, dot, delta);
            if (lane == 0) D_smem[r] = dot;
        }
        __syncthreads();

        // S = Q @ K^T
        {
            int m_tile = warp_id / 2;
            int n_tile = warp_id % 2;
            wmma::fill_fragment(c_frag, 0.0f);
            for (int k = 0; k < D / WMMA_K; k++) {
                wmma::load_matrix_sync(a_row, &Q_smem[m_tile * 16 * D + k * 16], D);
                wmma::load_matrix_sync(b_col, &K_smem[n_tile * 16 * D + k * 16], D);
                wmma::mma_sync(c_frag, a_row, b_col, c_frag);
            }
            wmma::store_matrix_sync(&S_smem[m_tile * 16 * Bc + n_tile * 16], c_frag, Bc, wmma::mem_row_major);
        }
        __syncthreads();

        // P = exp(S*scale - L) with mask; convert to bf16
        bool need_causal = (i == j);
        for (int idx = tid; idx < Br * Bc; idx += NUM_THREADS) {
            int r = idx / Bc;
            int c = idx % Bc;
            int k_idx = j * Bc + c;
            float s = S_smem[idx] * scale;
            float p;
            if (k_idx >= S || (need_causal && r < c)) {
                p = 0.0f;
            } else {
                p = expf(s - L_smem[r]);
            }
            S_smem[idx] = p;
            P_bf16[idx] = __float2bfloat16(p);
        }
        __syncthreads();

        // dP = dO @ V^T
        {
            int m_tile = warp_id / 2;
            int n_tile = warp_id % 2;
            wmma::fill_fragment(c_frag, 0.0f);
            for (int k = 0; k < D / WMMA_K; k++) {
                wmma::load_matrix_sync(a_row, &dO_smem[m_tile * 16 * D + k * 16], D);
                wmma::load_matrix_sync(b_col, &V_smem[n_tile * 16 * D + k * 16], D);
                wmma::mma_sync(c_frag, a_row, b_col, c_frag);
            }
            wmma::store_matrix_sync(&dP_smem[m_tile * 16 * Bc + n_tile * 16], c_frag, Bc, wmma::mem_row_major);
        }
        __syncthreads();

        // dS = P * (dP - D) * scale, convert to bf16
        for (int idx = tid; idx < Br * Bc; idx += NUM_THREADS) {
            int r = idx / Bc;
            float p = S_smem[idx];
            float dp = dP_smem[idx];
            float dv = D_smem[r];
            float ds = p * (dp - dv) * scale;
            S_smem[idx] = ds;
            dS_bf16[idx] = __float2bfloat16(ds);
        }
        __syncthreads();

        // dV += P^T @ dO
        for (int n = 0; n < 4; n++) {
            int nt = n_start_dk + n;
            for (int k = 0; k < Br / WMMA_K; k++) {
                wmma::load_matrix_sync(a_col, &P_bf16[k * 16 * Bc + m_tile_dk * 16], Bc);
                wmma::load_matrix_sync(b_row, &dO_smem[k * 16 * D + nt * 16], D);
                wmma::mma_sync(dV_frag[n], a_col, b_row, dV_frag[n]);
            }
        }

        // dK += dS^T @ Q
        for (int n = 0; n < 4; n++) {
            int nt = n_start_dk + n;
            for (int k = 0; k < Br / WMMA_K; k++) {
                wmma::load_matrix_sync(a_col, &dS_bf16[k * 16 * Bc + m_tile_dk * 16], Bc);
                wmma::load_matrix_sync(b_row, &Q_smem[k * 16 * D + nt * 16], D);
                wmma::mma_sync(dK_frag[n], a_col, b_row, dK_frag[n]);
            }
        }

        // dQ = dS @ K
        for (int n = 0; n < 4; n++) wmma::fill_fragment(dQ_frag[n], 0.0f);
        for (int k = 0; k < Bc / WMMA_K; k++) {
            wmma::load_matrix_sync(a_row, &dS_bf16[m_tile_dk * 16 * Bc + k * 16], Bc);
            for (int n = 0; n < 4; n++) {
                int nt = n_start_dk + n;
                wmma::load_matrix_sync(b_row, &K_smem[k * 16 * D + nt * 16], D);
                wmma::mma_sync(dQ_frag[n], a_row, b_row, dQ_frag[n]);
            }
        }

        // Store dQ pass 1 (m_tile 0: warps 0,1)
        if (warp_id < 2) {
            for (int n = 0; n < 4; n++) {
                int nt = n_start_dk + n;
                wmma::store_matrix_sync(&dQ_stage[nt * 16], dQ_frag[n], 128, wmma::mem_row_major);
            }
        }
        __syncthreads();
        for (int idx = tid; idx < 16 * D; idx += NUM_THREADS) {
            int r = idx / D;
            int c = idx % D;
            int q_idx = i * Br + r;
            if (q_idx < S)
                atomicAdd(&dQ_fp32[bh_off + (int64_t)q_idx * D + c], dQ_stage[idx]);
        }
        __syncthreads();

        // Store dQ pass 2 (m_tile 1: warps 2,3)
        if (warp_id >= 2) {
            for (int n = 0; n < 4; n++) {
                int nt = n_start_dk + n;
                wmma::store_matrix_sync(&dQ_stage[nt * 16], dQ_frag[n], 128, wmma::mem_row_major);
            }
        }
        __syncthreads();
        for (int idx = tid; idx < 16 * D; idx += NUM_THREADS) {
            int r = idx / D;
            int c = idx % D;
            int q_idx = i * Br + 16 + r;
            if (q_idx < S)
                atomicAdd(&dQ_fp32[bh_off + (int64_t)q_idx * D + c], dQ_stage[idx]);
        }
        __syncthreads();
    }

    // Store dK to global
    for (int n = 0; n < 4; n++) {
        int nt = n_start_dk + n;
        wmma::store_matrix_sync(&out_stage[m_tile_dk * 16 * D + nt * 16], dK_frag[n], D, wmma::mem_row_major);
    }
    __syncthreads();
    for (int idx = tid; idx < Bc * D; idx += NUM_THREADS) {
        int r = idx / D;
        int c = idx % D;
        int k_idx = j * Bc + r;
        if (k_idx < S)
            dK_g[bh_off + (int64_t)k_idx * D + c] = __float2bfloat16(out_stage[idx]);
    }
    __syncthreads();

    // Store dV to global
    for (int n = 0; n < 4; n++) {
        int nt = n_start_dk + n;
        wmma::store_matrix_sync(&out_stage[m_tile_dk * 16 * D + nt * 16], dV_frag[n], D, wmma::mem_row_major);
    }
    __syncthreads();
    for (int idx = tid; idx < Bc * D; idx += NUM_THREADS) {
        int r = idx / D;
        int c = idx % D;
        int k_idx = j * Bc + r;
        if (k_idx < S)
            dV_g[bh_off + (int64_t)k_idx * D + c] = __float2bfloat16(out_stage[idx]);
    }
}

__global__ void convert_kernel(const float* src, __nv_bfloat16* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __float2bfloat16(src[idx]);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = 4, H = 48, d = 128;
    int S = (int)Q.size(2);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    size_t dQ_size = (size_t)B * H * S * d;
    float* dQ_fp32;
    CUDA_CHECK(cudaMalloc(&dQ_fp32, dQ_size * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(dQ_fp32, 0, dQ_size * sizeof(float), stream));

    int num_kv_blocks = (S + Bc - 1) / Bc;
    dim3 grid(B * H, num_kv_blocks);
    dim3 block(NUM_THREADS);

    cudaFuncSetAttribute(attn_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE);

    attn_bwd_kernel<<<grid, block, SMEM_SIZE, stream>>>(
        (const __nv_bfloat16*)Q.data_ptr(),
        (const __nv_bfloat16*)K.data_ptr(),
        (const __nv_bfloat16*)V.data_ptr(),
        (const __nv_bfloat16*)O.data_ptr(),
        (const __nv_bfloat16*)dO.data_ptr(),
        (const float*)L.data_ptr(),
        dQ_fp32,
        (__nv_bfloat16*)dK.data_ptr(),
        (__nv_bfloat16*)dV.data_ptr(),
        S, H);

    int n = (int)dQ_size;
    convert_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        dQ_fp32, (__nv_bfloat16*)dQ.data_ptr(), n);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(dQ_fp32));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, attn_bwd::run);

}  // namespace attn_bwd