#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <mma.h>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda;

namespace flash_attn_bwd {

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", \
                cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

constexpr int D = 128;
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int WM = 16, WN = 16, WK = 16;

constexpr int S_TILES_PER_WARP = (BM/WM) * (BN/WN) / WARPS;
constexpr int DK_TILES_PER_WARP = (BN/WM) * (D/WN) / WARPS;
constexpr int DQ_TILES_PER_WARP = (BM/WM) * (D/WN) / WARPS;

constexpr int SMEM_SIZE = 4 * BN * D * 2 + BM * BN * 4 + BM * BN * 2 + 2 * BM * 4;

__device__ __forceinline__ void load_tile(const __nv_bfloat16* gmem, __nv_bfloat16* smem, int base_row, int tile_rows, int S) {
    int tid = threadIdx.x;
    int total_uint4 = tile_rows * D / 8;
    for (int i = tid; i < total_uint4; i += THREADS) {
        int row = i / (D / 8);
        int col = (i % (D / 8)) * 8;
        int gr = base_row + row;
        if (gr < S) {
            *reinterpret_cast<uint4*>(smem + row * D + col) =
                *reinterpret_cast<const uint4*>(gmem + (size_t)gr * D + col);
        } else {
            *reinterpret_cast<uint4*>(smem + row * D + col) = make_uint4(0, 0, 0, 0);
        }
    }
}

__global__ void compute_D_kernel(const __nv_bfloat16* dO, const __nv_bfloat16* O, float* D_out, int total_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= total_rows) return;
    const __nv_bfloat162* dO_r = reinterpret_cast<const __nv_bfloat162*>(dO + (size_t)row * D);
    const __nv_bfloat162* O_r = reinterpret_cast<const __nv_bfloat162*>(O + (size_t)row * D);
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < D / 2; i++) {
        float2 d = __bfloat1622float2(dO_r[i]);
        float2 o = __bfloat1622float2(O_r[i]);
        sum += d.x * o.x + d.y * o.y;
    }
    D_out[row] = sum;
}

__global__ __launch_bounds__(THREADS, 2)
void dK_dV_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ D_vals,
    __nv_bfloat16* __restrict__ dK_out,
    __nv_bfloat16* __restrict__ dV_out,
    int B, int H, int S)
{
    int bh = blockIdx.x;
    int kv_tile = blockIdx.y;
    int bn_start = kv_tile * BN;
    int b = bh / H, h = bh % H;
    size_t bh_off = (size_t)(b * H + h) * S * D;

    const __nv_bfloat16* Q_bh = Q + bh_off;
    const __nv_bfloat16* K_bh = K + bh_off;
    const __nv_bfloat16* V_bh = V + bh_off;
    const __nv_bfloat16* dO_bh = dO + bh_off;
    const float* L_bh = L + (size_t)(b * H + h) * S;
    const float* D_bh = D_vals + (size_t)(b * H + h) * S;
    __nv_bfloat16* dK_bh = dK_out + bh_off;
    __nv_bfloat16* dV_bh = dV_out + bh_off;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* K_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* V_smem = K_smem + BN * D;
    __nv_bfloat16* Q_smem = V_smem + BN * D;
    __nv_bfloat16* dO_smem = Q_smem + BM * D;
    float* S_smem = reinterpret_cast<float*>(dO_smem + BM * D);
    __nv_bfloat16* P_smem = reinterpret_cast<__nv_bfloat16*>(S_smem + BM * BN);
    float* L_smem = reinterpret_cast<float*>(P_smem + BM * BN);
    float* D_smem = L_smem + BM;

    int tid = threadIdx.x;
    int warp_id = tid / 32;

    wmma::fragment<wmma::accumulator, WM, WN, WK, float> dK_frag[DK_TILES_PER_WARP];
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> dV_frag[DK_TILES_PER_WARP];
    #pragma unroll
    for (int i = 0; i < DK_TILES_PER_WARP; i++) {
        wmma::fill_fragment(dK_frag[i], 0.0f);
        wmma::fill_fragment(dV_frag[i], 0.0f);
    }

    load_tile(K_bh, K_smem, bn_start, BN, S);
    load_tile(V_bh, V_smem, bn_start, BN, S);
    __syncthreads();

    const float scale = 0.08838834764831845f;
    int num_q_tiles = (S + BM - 1) / BM;

    for (int qt = 0; qt < num_q_tiles; qt++) {
        int bm_start = qt * BM;

        load_tile(Q_bh, Q_smem, bm_start, BM, S);
        load_tile(dO_bh, dO_smem, bm_start, BM, S);
        if (tid < BM) {
            int gr = bm_start + tid;
            L_smem[tid] = (gr < S) ? L_bh[gr] : 0.0f;
            D_smem[tid] = (gr < S) ? D_bh[gr] : 0.0f;
        }
        __syncthreads();

        // S = Q @ K^T
        #pragma unroll
        for (int ti = 0; ti < S_TILES_PER_WARP; ti++) {
            int tidx = warp_id * S_TILES_PER_WARP + ti;
            int tm = tidx / (BN/WN), tn = tidx % (BN/WN);
            wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc;
            wmma::fill_fragment(acc, 0.0f);
            #pragma unroll
            for (int kk = 0; kk < D / WK; kk++) {
                wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::col_major> b_frag;
                wmma::load_matrix_sync(a_frag, Q_smem + tm * WM * D + kk * WK, D);
                wmma::load_matrix_sync(b_frag, K_smem + tn * WN * D + kk * WK, D);
                wmma::mma_sync(acc, a_frag, b_frag, acc);
            }
            wmma::store_matrix_sync(S_smem + tm * WM * BN + tn * WN, acc, BN, wmma::mem_row_major);
        }
        __syncthreads();

        // P = exp(S * scale - L)
        for (int i = tid; i < BM * BN; i += THREADS) {
            int row = i / BN, col = i % BN;
            int gr = bm_start + row, gc = bn_start + col;
            P_smem[i] = (gr < S && gc < S) ? __float2bfloat16(__expf(S_smem[i] * scale - L_smem[row])) : __float2bfloat16(0.0f);
        }
        __syncthreads();

        // dP = dO @ V^T (reuse S_smem)
        #pragma unroll
        for (int ti = 0; ti < S_TILES_PER_WARP; ti++) {
            int tidx = warp_id * S_TILES_PER_WARP + ti;
            int tm = tidx / (BN/WN), tn = tidx % (BN/WN);
            wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc;
            wmma::fill_fragment(acc, 0.0f);
            #pragma unroll
            for (int kk = 0; kk < D / WK; kk++) {
                wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::col_major> b_frag;
                wmma::load_matrix_sync(a_frag, dO_smem + tm * WM * D + kk * WK, D);
                wmma::load_matrix_sync(b_frag, V_smem + tn * WN * D + kk * WK, D);
                wmma::mma_sync(acc, a_frag, b_frag, acc);
            }
            wmma::store_matrix_sync(S_smem + tm * WM * BN + tn * WN, acc, BN, wmma::mem_row_major);
        }
        __syncthreads();

        // dV += P^T @ dO
        #pragma unroll
        for (int ti = 0; ti < DK_TILES_PER_WARP; ti++) {
            int tidx = warp_id * DK_TILES_PER_WARP + ti;
            int tm = tidx / (D/WN), tn = tidx % (D/WN);
            #pragma unroll
            for (int kk = 0; kk < BM / WK; kk++) {
                wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::col_major> a_frag;
                wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::row_major> b_frag;
                wmma::load_matrix_sync(a_frag, P_smem + kk * WK * BN + tm * WM, BN);
                wmma::load_matrix_sync(b_frag, dO_smem + kk * WK * D + tn * WN, D);
                wmma::mma_sync(dV_frag[ti], a_frag, b_frag, dV_frag[ti]);
            }
        }
        __syncthreads(); // Ensure dV done reading P_smem before dS writes it

        // dS = P * (dP - D) * scale (overwrite P_smem)
        for (int i = tid; i < BM * BN; i += THREADS) {
            int row = i / BN, col = i % BN;
            int gr = bm_start + row, gc = bn_start + col;
            float p = __bfloat162float(P_smem[i]);
            float dp = S_smem[i];
            P_smem[i] = (gr < S && gc < S) ? __float2bfloat16(p * (dp - D_smem[row]) * scale) : __float2bfloat16(0.0f);
        }
        __syncthreads();

        // dK += dS^T @ Q
        #pragma unroll
        for (int ti = 0; ti < DK_TILES_PER_WARP; ti++) {
            int tidx = warp_id * DK_TILES_PER_WARP + ti;
            int tm = tidx / (D/WN), tn = tidx % (D/WN);
            #pragma unroll
            for (int kk = 0; kk < BM / WK; kk++) {
                wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::col_major> a_frag;
                wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::row_major> b_frag;
                wmma::load_matrix_sync(a_frag, P_smem + kk * WK * BN + tm * WM, BN);
                wmma::load_matrix_sync(b_frag, Q_smem + kk * WK * D + tn * WN, D);
                wmma::mma_sync(dK_frag[ti], a_frag, b_frag, dK_frag[ti]);
            }
        }
        __syncthreads();
    }

    // Epilogue: store dK then dV
    float* epi_buf = reinterpret_cast<float*>(smem_raw);
    #pragma unroll
    for (int ti = 0; ti < DK_TILES_PER_WARP; ti++) {
        int tidx = warp_id * DK_TILES_PER_WARP + ti;
        int tm = tidx / (D/WN), tn = tidx % (D/WN);
        wmma::store_matrix_sync(epi_buf + tm * WM * D + tn * WN, dK_frag[ti], D, wmma::mem_row_major);
    }
    __syncthreads();
    for (int i = tid; i < BN * D; i += THREADS) {
        int row = i / D, col = i % D;
        int gr = bn_start + row;
        if (gr < S) dK_bh[(size_t)gr * D + col] = __float2bfloat16(epi_buf[i]);
    }
    __syncthreads();

    #pragma unroll
    for (int ti = 0; ti < DK_TILES_PER_WARP; ti++) {
        int tidx = warp_id * DK_TILES_PER_WARP + ti;
        int tm = tidx / (D/WN), tn = tidx % (D/WN);
        wmma::store_matrix_sync(epi_buf + tm * WM * D + tn * WN, dV_frag[ti], D, wmma::mem_row_major);
    }
    __syncthreads();
    for (int i = tid; i < BN * D; i += THREADS) {
        int row = i / D, col = i % D;
        int gr = bn_start + row;
        if (gr < S) dV_bh[(size_t)gr * D + col] = __float2bfloat16(epi_buf[i]);
    }
}

__global__ __launch_bounds__(THREADS, 2)
void dQ_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ D_vals,
    __nv_bfloat16* __restrict__ dQ_out,
    int B, int H, int S)
{
    int bh = blockIdx.x;
    int q_tile = blockIdx.y;
    int bm_start = q_tile * BM;
    int b = bh / H, h = bh % H;
    size_t bh_off = (size_t)(b * H + h) * S * D;

    const __nv_bfloat16* Q_bh = Q + bh_off;
    const __nv_bfloat16* K_bh = K + bh_off;
    const __nv_bfloat16* V_bh = V + bh_off;
    const __nv_bfloat16* dO_bh = dO + bh_off;
    const float* L_bh = L + (size_t)(b * H + h) * S;
    const float* D_bh = D_vals + (size_t)(b * H + h) * S;
    __nv_bfloat16* dQ_bh = dQ_out + bh_off;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* Q_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* dO_smem = Q_smem + BM * D;
    __nv_bfloat16* K_smem = dO_smem + BM * D;
    __nv_bfloat16* V_smem = K_smem + BN * D;
    float* S_smem = reinterpret_cast<float*>(V_smem + BN * D);
    __nv_bfloat16* P_smem = reinterpret_cast<__nv_bfloat16*>(S_smem + BM * BN);
    float* L_smem = reinterpret_cast<float*>(P_smem + BM * BN);
    float* D_smem = L_smem + BM;

    int tid = threadIdx.x;
    int warp_id = tid / 32;

    wmma::fragment<wmma::accumulator, WM, WN, WK, float> dQ_frag[DQ_TILES_PER_WARP];
    #pragma unroll
    for (int i = 0; i < DQ_TILES_PER_WARP; i++) wmma::fill_fragment(dQ_frag[i], 0.0f);

    load_tile(Q_bh, Q_smem, bm_start, BM, S);
    load_tile(dO_bh, dO_smem, bm_start, BM, S);
    if (tid < BM) {
        int gr = bm_start + tid;
        L_smem[tid] = (gr < S) ? L_bh[gr] : 0.0f;
        D_smem[tid] = (gr < S) ? D_bh[gr] : 0.0f;
    }
    __syncthreads();

    const float scale = 0.08838834764831845f;
    int num_kv_tiles = (S + BN - 1) / BN;

    for (int kt = 0; kt < num_kv_tiles; kt++) {
        int bn_start = kt * BN;

        load_tile(K_bh, K_smem, bn_start, BN, S);
        load_tile(V_bh, V_smem, bn_start, BN, S);
        __syncthreads();

        // S = Q @ K^T
        #pragma unroll
        for (int ti = 0; ti < S_TILES_PER_WARP; ti++) {
            int tidx = warp_id * S_TILES_PER_WARP + ti;
            int tm = tidx / (BN/WN), tn = tidx % (BN/WN);
            wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc;
            wmma::fill_fragment(acc, 0.0f);
            #pragma unroll
            for (int kk = 0; kk < D / WK; kk++) {
                wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::col_major> b_frag;
                wmma::load_matrix_sync(a_frag, Q_smem + tm * WM * D + kk * WK, D);
                wmma::load_matrix_sync(b_frag, K_smem + tn * WN * D + kk * WK, D);
                wmma::mma_sync(acc, a_frag, b_frag, acc);
            }
            wmma::store_matrix_sync(S_smem + tm * WM * BN + tn * WN, acc, BN, wmma::mem_row_major);
        }
        __syncthreads();

        // P = exp(S * scale - L)
        for (int i = tid; i < BM * BN; i += THREADS) {
            int row = i / BN, col = i % BN;
            int gr = bm_start + row, gc = bn_start + col;
            P_smem[i] = (gr < S && gc < S) ? __float2bfloat16(__expf(S_smem[i] * scale - L_smem[row])) : __float2bfloat16(0.0f);
        }
        __syncthreads();

        // dP = dO @ V^T (reuse S_smem)
        #pragma unroll
        for (int ti = 0; ti < S_TILES_PER_WARP; ti++) {
            int tidx = warp_id * S_TILES_PER_WARP + ti;
            int tm = tidx / (BN/WN), tn = tidx % (BN/WN);
            wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc;
            wmma::fill_fragment(acc, 0.0f);
            #pragma unroll
            for (int kk = 0; kk < D / WK; kk++) {
                wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::col_major> b_frag;
                wmma::load_matrix_sync(a_frag, dO_smem + tm * WM * D + kk * WK, D);
                wmma::load_matrix_sync(b_frag, V_smem + tn * WN * D + kk * WK, D);
                wmma::mma_sync(acc, a_frag, b_frag, acc);
            }
            wmma::store_matrix_sync(S_smem + tm * WM * BN + tn * WN, acc, BN, wmma::mem_row_major);
        }
        __syncthreads();

        // dS = P * (dP - D) * scale (overwrite P_smem)
        for (int i = tid; i < BM * BN; i += THREADS) {
            int row = i / BN, col = i % BN;
            int gr = bm_start + row, gc = bn_start + col;
            float p = __bfloat162float(P_smem[i]);
            float dp = S_smem[i];
            P_smem[i] = (gr < S && gc < S) ? __float2bfloat16(p * (dp - D_smem[row]) * scale) : __float2bfloat16(0.0f);
        }
        __syncthreads();

        // dQ += dS @ K [BM, D]
        #pragma unroll
        for (int ti = 0; ti < DQ_TILES_PER_WARP; ti++) {
            int tidx = warp_id * DQ_TILES_PER_WARP + ti;
            int tm = tidx / (D/WN), tn = tidx % (D/WN);
            #pragma unroll
            for (int kk = 0; kk < BN / WK; kk++) {
                wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::row_major> b_frag;
                wmma::load_matrix_sync(a_frag, P_smem + tm * WM * BN + kk * WK, BN);
                wmma::load_matrix_sync(b_frag, K_smem + kk * WK * D + tn * WN, D);
                wmma::mma_sync(dQ_frag[ti], a_frag, b_frag, dQ_frag[ti]);
            }
        }
        __syncthreads();
    }

    // Store dQ
    float* epi_buf = reinterpret_cast<float*>(smem_raw);
    #pragma unroll
    for (int ti = 0; ti < DQ_TILES_PER_WARP; ti++) {
        int tidx = warp_id * DQ_TILES_PER_WARP + ti;
        int tm = tidx / (D/WN), tn = tidx % (D/WN);
        wmma::store_matrix_sync(epi_buf + tm * WM * D + tn * WN, dQ_frag[ti], D, wmma::mem_row_major);
    }
    __syncthreads();
    for (int i = tid; i < BM * D; i += THREADS) {
        int row = i / D, col = i % D;
        int gr = bm_start + row;
        if (gr < S) dQ_bh[(size_t)gr * D + col] = __float2bfloat16(epi_buf[i]);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {

    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    int B = (int)Q.size(0), H = (int)Q.size(1), S = (int)Q.size(2);
    int total_rows = B * H * S;

    float* D_temp;
    CUDA_CHECK(cudaMalloc(&D_temp, (size_t)total_rows * sizeof(float)));

    {
        int threads = 256, blocks = (total_rows + threads - 1) / threads;
        compute_D_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(dO.data_ptr()),
            static_cast<const __nv_bfloat16*>(O.data_ptr()),
            D_temp, total_rows);
    }

    {
        dim3 grid(B * H, (S + BN - 1) / BN, 1);
        dim3 block(THREADS, 1, 1);
        CUDA_CHECK(cudaFuncSetAttribute(dK_dV_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));
        dK_dV_kernel<<<grid, block, SMEM_SIZE, stream>>>(
            static_cast<const __nv_bfloat16*>(Q.data_ptr()),
            static_cast<const __nv_bfloat16*>(K.data_ptr()),
            static_cast<const __nv_bfloat16*>(V.data_ptr()),
            static_cast<const __nv_bfloat16*>(dO.data_ptr()),
            static_cast<const float*>(L.data_ptr()),
            D_temp,
            static_cast<__nv_bfloat16*>(dK.data_ptr()),
            static_cast<__nv_bfloat16*>(dV.data_ptr()),
            B, H, S);
    }

    {
        dim3 grid(B * H, (S + BM - 1) / BM, 1);
        dim3 block(THREADS, 1, 1);
        CUDA_CHECK(cudaFuncSetAttribute(dQ_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));
        dQ_kernel<<<grid, block, SMEM_SIZE, stream>>>(
            static_cast<const __nv_bfloat16*>(Q.data_ptr()),
            static_cast<const __nv_bfloat16*>(K.data_ptr()),
            static_cast<const __nv_bfloat16*>(V.data_ptr()),
            static_cast<const __nv_bfloat16*>(dO.data_ptr()),
            static_cast<const float*>(L.data_ptr()),
            D_temp,
            static_cast<__nv_bfloat16*>(dQ.data_ptr()),
            B, H, S);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(D_temp));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flash_attn_bwd::run);

}  // namespace flash_attn_bwd