#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
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

namespace mha_bwd_d128 {

constexpr int BM = 32;
constexpr int BN = 64;
constexpr int D = 128;
constexpr int D_PAD = 136;
constexpr int BN_PAD = 72;
constexpr int THREADS = 128;
constexpr int WARPS = THREADS / 32;
constexpr int B_CONST = 4;
constexpr int H_CONST = 48;
constexpr int WM = 16, WN = 16, WK = 16;

__device__ __forceinline__ float fast_expf(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x * 1.4426950408889634f));
    return y;
}

__global__ void dQ_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    float* __restrict__ dQ_out,
    int S)
{
    const float scale = rsqrtf((float)D);
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int num_q_blocks = (S + BM - 1) / BM;
    int q_block = blockIdx.x;
    int bh = q_block / num_q_blocks;
    int q_tile = q_block % num_q_blocks;
    int batch = bh / H_CONST;
    int head = bh % H_CONST;

    int64_t bh_off = (int64_t)batch * H_CONST * S * D + (int64_t)head * S * D;
    int64_t lse_off = (int64_t)batch * H_CONST * S + (int64_t)head * S;

    const __nv_bfloat16* Qp = Q + bh_off;
    const __nv_bfloat16* Kp = K + bh_off;
    const __nv_bfloat16* Vp = V + bh_off;
    const __nv_bfloat16* Op = O + bh_off;
    const __nv_bfloat16* dOp = dO + bh_off;
    const float* Lp = L + lse_off;
    float* dQp = dQ_out + bh_off;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* sQ  = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sdO = sQ + BM * D_PAD;
    __nv_bfloat16* sK  = sdO + BM * D_PAD;
    __nv_bfloat16* sV  = sK + BN * D_PAD;
    float* sSf  = reinterpret_cast<float*>(sV + BN * D_PAD);
    __nv_bfloat16* sPb = reinterpret_cast<__nv_bfloat16*>(sSf + BM * BN);
    __nv_bfloat16* sSb = sPb + BM * BN_PAD;
    float* sLSE = reinterpret_cast<float*>(sSb + BM * BN_PAD);
    float* sDv  = sLSE + BM;
    float* sTmp = sDv + BM;

    int q_start = q_tile * BM;
    int q_len = min(BM, S - q_start);

    // Load Q, dO
    for (int i = tid; i < (BM * D) / 8; i += THREADS) {
        int m = (i * 8) / D, d_off = (i * 8) % D;
        int gm = q_start + m;
        if (gm < S) {
            *reinterpret_cast<int4*>(sQ + m * D_PAD + d_off) =
                *reinterpret_cast<const int4*>(Qp + gm * D + d_off);
            *reinterpret_cast<int4*>(sdO + m * D_PAD + d_off) =
                *reinterpret_cast<const int4*>(dOp + gm * D + d_off);
        } else {
            *reinterpret_cast<int4*>(sQ + m * D_PAD + d_off) = make_int4(0, 0, 0, 0);
            *reinterpret_cast<int4*>(sdO + m * D_PAD + d_off) = make_int4(0, 0, 0, 0);
        }
    }
    for (int i = tid; i < BM; i += THREADS) {
        *reinterpret_cast<int2*>(sQ + i * D_PAD + D) = make_int2(0, 0);
        *reinterpret_cast<int2*>(sdO + i * D_PAD + D) = make_int2(0, 0);
    }

    for (int m = tid; m < BM; m += THREADS)
        sLSE[m] = (m < q_len) ? Lp[q_start + m] : 0.0f;

    for (int m = warp_id; m < BM; m += WARPS) {
        if (m < q_len) {
            float partial = 0.0f;
            for (int d = lane_id; d < D; d += 32)
                partial += __bfloat162float(dOp[(q_start+m)*D+d]) * __bfloat162float(Op[(q_start+m)*D+d]);
            for (int o = 16; o > 0; o >>= 1)
                partial += __shfl_xor_sync(0xFFFFFFFF, partial, o);
            if (lane_id == 0) sDv[m] = partial;
        } else {
            if (lane_id == 0) sDv[m] = 0.0f;
        }
    }
    __syncthreads();

    // dQ: [BM, D] = [32, 128] = 2x8=16 tiles, 4 per warp = 4 fragments
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> dQf[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) wmma::fill_fragment(dQf[i], 0.0f);

    int num_kv = (S + BN - 1) / BN;
    for (int kv = 0; kv < num_kv; kv++) {
        int kv_start = kv * BN;
        int kv_len = min(BN, S - kv_start);

        for (int i = tid; i < (BN * D) / 8; i += THREADS) {
            int n = (i * 8) / D, d_off = (i * 8) % D;
            if (kv_start + n < S) {
                *reinterpret_cast<int4*>(sK + n * D_PAD + d_off) =
                    *reinterpret_cast<const int4*>(Kp + (kv_start + n) * D + d_off);
                *reinterpret_cast<int4*>(sV + n * D_PAD + d_off) =
                    *reinterpret_cast<const int4*>(Vp + (kv_start + n) * D + d_off);
            } else {
                *reinterpret_cast<int4*>(sK + n * D_PAD + d_off) = make_int4(0, 0, 0, 0);
                *reinterpret_cast<int4*>(sV + n * D_PAD + d_off) = make_int4(0, 0, 0, 0);
            }
        }
        for (int i = tid; i < BN; i += THREADS) {
            *reinterpret_cast<int2*>(sK + i * D_PAD + D) = make_int2(0, 0);
            *reinterpret_cast<int2*>(sV + i * D_PAD + D) = make_int2(0, 0);
        }
        __syncthreads();

        // S = Q @ K^T: [32, 64] = 2x4=8 tiles, 2 per warp, 8 k-iters
        #pragma unroll
        for (int ti = 0; ti < 2; ti++) {
            int t = warp_id * 2 + ti;
            int mt = t / 4, nt = t % 4;
            wmma::fragment<wmma::accumulator, WM, WN, WK, float> cf;
            wmma::fill_fragment(cf, 0.0f);
            #pragma unroll
            for (int kk = 0; kk < D; kk += WK) {
                wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::row_major> af;
                wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::col_major> bf;
                wmma::load_matrix_sync(af, sQ + mt*16*D_PAD + kk, D_PAD);
                wmma::load_matrix_sync(bf, sK + nt*16*D_PAD + kk, D_PAD);
                wmma::mma_sync(cf, af, bf, cf);
            }
            wmma::store_matrix_sync(sSf + mt*16*BN + nt*16, cf, BN, wmma::mem_row_major);
        }
        __syncthreads();

        // P = exp(S*scale - LSE) -> sPb [BM, BN]
        for (int idx = tid; idx < BM * BN; idx += THREADS) {
            int m = idx / BN, n = idx % BN;
            float pv = (m < q_len && n < kv_len) ? fast_expf(sSf[idx] * scale - sLSE[m]) : 0.0f;
            sPb[m * BN_PAD + n] = __float2bfloat16(pv);
        }
        __syncthreads();

        // dP = dO @ V^T -> sSf [BM, BN]
        #pragma unroll
        for (int ti = 0; ti < 2; ti++) {
            int t = warp_id * 2 + ti;
            int mt = t / 4, nt = t % 4;
            wmma::fragment<wmma::accumulator, WM, WN, WK, float> cf;
            wmma::fill_fragment(cf, 0.0f);
            #pragma unroll
            for (int kk = 0; kk < D; kk += WK) {
                wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::row_major> af;
                wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::col_major> bf;
                wmma::load_matrix_sync(af, sdO + mt*16*D_PAD + kk, D_PAD);
                wmma::load_matrix_sync(bf, sV + nt*16*D_PAD + kk, D_PAD);
                wmma::mma_sync(cf, af, bf, cf);
            }
            wmma::store_matrix_sync(sSf + mt*16*BN + nt*16, cf, BN, wmma::mem_row_major);
        }
        __syncthreads();

        // dS = P * (dP - D) -> sSb [BM, BN]
        for (int idx = tid; idx < BM * BN; idx += THREADS) {
            int m = idx / BN, n = idx % BN;
            float p_val = __bfloat162float(sPb[m * BN_PAD + n]);
            float ds = p_val * (sSf[idx] - sDv[m]);
            sSb[m * BN_PAD + n] = __float2bfloat16(ds);
        }
        __syncthreads();

        // dQ += dS @ K: [32, 128] = 2x8=16 tiles, 4 per warp, 4 k-iters
        #pragma unroll
        for (int ti = 0; ti < 4; ti++) {
            int t = warp_id * 4 + ti;
            int mt = t / 8, nt = t % 8;
            #pragma unroll
            for (int kk = 0; kk < BN; kk += WK) {
                wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::row_major> af;
                wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::row_major> bf;
                wmma::load_matrix_sync(af, sSb + mt*16*BN_PAD + kk, BN_PAD);
                wmma::load_matrix_sync(bf, sK + kk*D_PAD + nt*16, D_PAD);
                wmma::mma_sync(dQf[ti], af, bf, dQf[ti]);
            }
        }
        __syncthreads();
    }

    // Store dQ
    #pragma unroll
    for (int ti = 0; ti < 4; ti++) {
        int t = warp_id * 4 + ti;
        int mt = t / 8, nt = t % 8;
        float* my = sTmp + warp_id * 256;
        wmma::store_matrix_sync(my, dQf[ti], 16, wmma::mem_row_major);
        __syncwarp();
        for (int j = lane_id; j < 256; j += 32) {
            int row = j / 16, col = j % 16;
            int m = mt*16 + row, d = nt*16 + col;
            if (q_start + m < S)
                dQp[(q_start+m)*D + d] = my[j] * scale;
        }
        __syncwarp();
    }
}

__global__ void dKdV_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    float* __restrict__ dK_out,
    float* __restrict__ dV_out,
    int S)
{
    const float scale = rsqrtf((float)D);
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int num_kv_blocks = (S + BN - 1) / BN;
    int kv_block = blockIdx.x;
    int bh = kv_block / num_kv_blocks;
    int kv_tile = kv_block % num_kv_blocks;
    int batch = bh / H_CONST;
    int head = bh % H_CONST;

    int64_t bh_off = (int64_t)batch * H_CONST * S * D + (int64_t)head * S * D;
    int64_t lse_off = (int64_t)batch * H_CONST * S + (int64_t)head * S;

    const __nv_bfloat16* Qp = Q + bh_off;
    const __nv_bfloat16* Kp = K + bh_off;
    const __nv_bfloat16* Vp = V + bh_off;
    const __nv_bfloat16* Op = O + bh_off;
    const __nv_bfloat16* dOp = dO + bh_off;
    const float* Lp = L + lse_off;
    float* dKp = dK_out + bh_off;
    float* dVp = dV_out + bh_off;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* sK  = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sV  = sK + BN * D_PAD;
    __nv_bfloat16* sQ  = sV + BN * D_PAD;
    __nv_bfloat16* sdO = sQ + BM * D_PAD;
    float* sSf  = reinterpret_cast<float*>(sdO + BM * D_PAD);
    __nv_bfloat16* sPb = reinterpret_cast<__nv_bfloat16*>(sSf + BM * BN);
    __nv_bfloat16* sSb = sPb + BM * BN_PAD;
    float* sLSE = reinterpret_cast<float*>(sSb + BM * BN_PAD);
    float* sDv  = sLSE + BM;
    float* sTmp = sDv + BM;

    int kv_start = kv_tile * BN;
    int kv_len = min(BN, S - kv_start);

    for (int i = tid; i < (BN * D) / 8; i += THREADS) {
        int n = (i * 8) / D, d_off = (i * 8) % D;
        if (kv_start + n < S) {
            *reinterpret_cast<int4*>(sK + n * D_PAD + d_off) =
                *reinterpret_cast<const int4*>(Kp + (kv_start + n) * D + d_off);
            *reinterpret_cast<int4*>(sV + n * D_PAD + d_off) =
                *reinterpret_cast<const int4*>(Vp + (kv_start + n) * D + d_off);
        } else {
            *reinterpret_cast<int4*>(sK + n * D_PAD + d_off) = make_int4(0, 0, 0, 0);
            *reinterpret_cast<int4*>(sV + n * D_PAD + d_off) = make_int4(0, 0, 0, 0);
        }
    }
    for (int i = tid; i < BN; i += THREADS) {
        *reinterpret_cast<int2*>(sK + i * D_PAD + D) = make_int2(0, 0);
        *reinterpret_cast<int2*>(sV + i * D_PAD + D) = make_int2(0, 0);
    }
    __syncthreads();

    // dK, dV: [BN, D] = [64, 128] = 4x8=32 tiles, 8 per warp = 4+4=8 fragments
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> dKf[4];
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> dVf[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        wmma::fill_fragment(dKf[i], 0.0f);
        wmma::fill_fragment(dVf[i], 0.0f);
    }

    int num_q = (S + BM - 1) / BM;
    for (int q_tile = 0; q_tile < num_q; q_tile++) {
        int q_start = q_tile * BM;
        int q_len = min(BM, S - q_start);

        for (int i = tid; i < (BM * D) / 8; i += THREADS) {
            int m = (i * 8) / D, d_off = (i * 8) % D;
            if (q_start + m < S) {
                *reinterpret_cast<int4*>(sQ + m * D_PAD + d_off) =
                    *reinterpret_cast<const int4*>(Qp + (q_start + m) * D + d_off);
                *reinterpret_cast<int4*>(sdO + m * D_PAD + d_off) =
                    *reinterpret_cast<const int4*>(dOp + (q_start + m) * D + d_off);
            } else {
                *reinterpret_cast<int4*>(sQ + m * D_PAD + d_off) = make_int4(0, 0, 0, 0);
                *reinterpret_cast<int4*>(sdO + m * D_PAD + d_off) = make_int4(0, 0, 0, 0);
            }
        }
        for (int i = tid; i < BM; i += THREADS) {
            *reinterpret_cast<int2*>(sQ + i * D_PAD + D) = make_int2(0, 0);
            *reinterpret_cast<int2*>(sdO + i * D_PAD + D) = make_int2(0, 0);
        }

        for (int m = tid; m < BM; m += THREADS)
            sLSE[m] = (m < q_len) ? Lp[q_start + m] : 0.0f;

        for (int m = warp_id; m < BM; m += WARPS) {
            if (m < q_len) {
                float partial = 0.0f;
                for (int d = lane_id; d < D; d += 32)
                    partial += __bfloat162float(dOp[(q_start+m)*D+d]) * __bfloat162float(Op[(q_start+m)*D+d]);
                for (int o = 16; o > 0; o >>= 1)
                    partial += __shfl_xor_sync(0xFFFFFFFF, partial, o);
                if (lane_id == 0) sDv[m] = partial;
            } else {
                if (lane_id == 0) sDv[m] = 0.0f;
            }
        }
        __syncthreads();

        // S = Q @ K^T -> sSf [BM, BN]
        #pragma unroll
        for (int ti = 0; ti < 2; ti++) {
            int t = warp_id * 2 + ti;
            int mt = t / 4, nt = t % 4;
            wmma::fragment<wmma::accumulator, WM, WN, WK, float> cf;
            wmma::fill_fragment(cf, 0.0f);
            #pragma unroll
            for (int kk = 0; kk < D; kk += WK) {
                wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::row_major> af;
                wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::col_major> bf;
                wmma::load_matrix_sync(af, sQ + mt*16*D_PAD + kk, D_PAD);
                wmma::load_matrix_sync(bf, sK + nt*16*D_PAD + kk, D_PAD);
                wmma::mma_sync(cf, af, bf, cf);
            }
            wmma::store_matrix_sync(sSf + mt*16*BN + nt*16, cf, BN, wmma::mem_row_major);
        }
        __syncthreads();

        // P = exp(S*scale - LSE) -> sPb [BM, BN]
        for (int idx = tid; idx < BM * BN; idx += THREADS) {
            int m = idx / BN, n = idx % BN;
            float pv = (m < q_len && n < kv_len) ? fast_expf(sSf[idx] * scale - sLSE[m]) : 0.0f;
            sPb[m * BN_PAD + n] = __float2bfloat16(pv);
        }
        __syncthreads();

        // dP = dO @ V^T -> sSf [BM, BN]
        #pragma unroll
        for (int ti = 0; ti < 2; ti++) {
            int t = warp_id * 2 + ti;
            int mt = t / 4, nt = t % 4;
            wmma::fragment<wmma::accumulator, WM, WN, WK, float> cf;
            wmma::fill_fragment(cf, 0.0f);
            #pragma unroll
            for (int kk = 0; kk < D; kk += WK) {
                wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::row_major> af;
                wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::col_major> bf;
                wmma::load_matrix_sync(af, sdO + mt*16*D_PAD + kk, D_PAD);
                wmma::load_matrix_sync(bf, sV + nt*16*D_PAD + kk, D_PAD);
                wmma::mma_sync(cf, af, bf, cf);
            }
            wmma::store_matrix_sync(sSf + mt*16*BN + nt*16, cf, BN, wmma::mem_row_major);
        }
        __syncthreads();

        // dS = P * (dP - D) -> sSb [BM, BN]
        for (int idx = tid; idx < BM * BN; idx += THREADS) {
            int m = idx / BN, n = idx % BN;
            float p_val = __bfloat162float(sPb[m * BN_PAD + n]);
            float ds = p_val * (sSf[idx] - sDv[m]);
            sSb[m * BN_PAD + n] = __float2bfloat16(ds);
        }
        __syncthreads();

        // dV += P^T @ dO: [64, 128], P is [BM, BN] col_major, 4x8=32 tiles, 8 per warp, 2 k-iters
        #pragma unroll
        for (int ti = 0; ti < 4; ti++) {
            int t = warp_id * 4 + ti;
            int mt = t / 8, nt = t % 8;
            #pragma unroll
            for (int kk = 0; kk < BM; kk += WK) {
                wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::col_major> af;
                wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::row_major> bf;
                wmma::load_matrix_sync(af, sPb + kk*BN_PAD + mt*16, BN_PAD);
                wmma::load_matrix_sync(bf, sdO + kk*D_PAD + nt*16, D_PAD);
                wmma::mma_sync(dVf[ti], af, bf, dVf[ti]);
            }
        }

        // dK += dS^T @ Q: [64, 128], same structure
        #pragma unroll
        for (int ti = 0; ti < 4; ti++) {
            int t = warp_id * 4 + ti;
            int mt = t / 8, nt = t % 8;
            #pragma unroll
            for (int kk = 0; kk < BM; kk += WK) {
                wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::col_major> af;
                wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::row_major> bf;
                wmma::load_matrix_sync(af, sSb + kk*BN_PAD + mt*16, BN_PAD);
                wmma::load_matrix_sync(bf, sQ + kk*D_PAD + nt*16, D_PAD);
                wmma::mma_sync(dKf[ti], af, bf, dKf[ti]);
            }
        }
        __syncthreads();
    }

    // Store dV
    #pragma unroll
    for (int ti = 0; ti < 4; ti++) {
        int t = warp_id * 4 + ti;
        int mt = t / 8, nt = t % 8;
        float* my = sTmp + warp_id * 256;
        wmma::store_matrix_sync(my, dVf[ti], 16, wmma::mem_row_major);
        __syncwarp();
        for (int j = lane_id; j < 256; j += 32) {
            int row = j / 16, col = j % 16;
            int n = mt*16 + row, d = nt*16 + col;
            if (kv_start + n < S)
                dVp[(kv_start+n)*D + d] = my[j];
        }
        __syncwarp();
    }
    __syncthreads();

    // Store dK
    #pragma unroll
    for (int ti = 0; ti < 4; ti++) {
        int t = warp_id * 4 + ti;
        int mt = t / 8, nt = t % 8;
        float* my = sTmp + warp_id * 256;
        wmma::store_matrix_sync(my, dKf[ti], 16, wmma::mem_row_major);
        __syncwarp();
        for (int j = lane_id; j < 256; j += 32) {
            int row = j / 16, col = j % 16;
            int n = mt*16 + row, d = nt*16 + col;
            if (kv_start + n < S)
                dKp[(kv_start+n)*D + d] = my[j] * scale;
        }
        __syncwarp();
    }
}

__global__ void convert_kernel(const float* src, __nv_bfloat16* dst, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) dst[idx] = __float2bfloat16(src[idx]);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {

    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int64_t S = Q.size(2);

    size_t smem = (size_t)(BM*D_PAD*2 + BM*D_PAD*2 + BN*D_PAD*2 + BN*D_PAD*2 + BM*BN*4 + BM*BN_PAD*2 + BM*BN_PAD*2 + BM*4 + BM*4 + WARPS*256*4);

    CUDA_CHECK(cudaFuncSetAttribute(dQ_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    CUDA_CHECK(cudaFuncSetAttribute(dKdV_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    size_t tmp_sz = (size_t)B_CONST * H_CONST * S * D * sizeof(float);
    float *dQf, *dKf, *dVf;
    CUDA_CHECK(cudaMallocAsync(&dQf, tmp_sz, stream));
    CUDA_CHECK(cudaMallocAsync(&dKf, tmp_sz, stream));
    CUDA_CHECK(cudaMallocAsync(&dVf, tmp_sz, stream));

    int nqb = ((int)S + BM - 1) / BM;
    int grid1 = B_CONST * H_CONST * nqb;
    dQ_kernel<<<grid1, THREADS, smem, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(O.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        dQf, (int)S);
    CUDA_CHECK(cudaGetLastError());

    int nkb = ((int)S + BN - 1) / BN;
    int grid2 = B_CONST * H_CONST * nkb;
    dKdV_kernel<<<grid2, THREADS, smem, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(O.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        dKf, dVf, (int)S);
    CUDA_CHECK(cudaGetLastError());

    int nelem = (int)((size_t)B_CONST * H_CONST * S * D);
    int cb = (nelem + 255) / 256;
    convert_kernel<<<cb, 256, 0, stream>>>(dQf, static_cast<__nv_bfloat16*>(dQ.data_ptr()), nelem);
    convert_kernel<<<cb, 256, 0, stream>>>(dKf, static_cast<__nv_bfloat16*>(dK.data_ptr()), nelem);
    convert_kernel<<<cb, 256, 0, stream>>>(dVf, static_cast<__nv_bfloat16*>(dV.data_ptr()), nelem);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFreeAsync(dQf, stream));
    CUDA_CHECK(cudaFreeAsync(dKf, stream));
    CUDA_CHECK(cudaFreeAsync(dVf, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128::run);

}  // namespace mha_bwd_d128