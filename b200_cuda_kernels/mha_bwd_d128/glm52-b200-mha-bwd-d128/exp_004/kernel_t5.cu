#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
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

#define CU_CHECK(call) do { \
    CUresult _r = (call); \
    if (_r != CUDA_SUCCESS) { \
        const char* _err; \
        cuGetErrorString(_r, &_err); \
        fprintf(stderr, "CU error %s at %s:%d\n", _err, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

constexpr int D = 128;
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int WM = 16, WN = 16, WK = 16;
constexpr int TILES_PER_WARP = (BM / WM) * (BN / WN) / WARPS;
constexpr int DKV_TILES_PER_WARP = (BN / WM) * (D / WN) / WARPS;
constexpr int DQ_TILES_PER_WARP = (BM / WM) * (D / WN) / WARPS;

__device__ __forceinline__ void tma_load(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d), "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void mbarrier_init(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}
__device__ __forceinline__ void mbarrier_arrive_etx(uint64_t* bar, uint32_t tx) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx) : "memory");
}
__device__ __forceinline__ void mbarrier_wait(uint64_t* bar, uint32_t phase) {
    asm volatile("{\n.reg .pred P;\nWAIT_%=:\nmbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n@!P bra WAIT_%=;\n}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase));
}
__device__ __forceinline__ void fence_mbarrier_init() { asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory"); }
__device__ __forceinline__ void fence_async_shared() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }

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

__global__ __launch_bounds__(THREADS, 1)
void dK_dV_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* __restrict__ L,
    const float* __restrict__ D_vals,
    __nv_bfloat16* __restrict__ dK_out,
    __nv_bfloat16* __restrict__ dV_out,
    int B, int H, int S)
{
    int bh = blockIdx.x;
    int kv_tile = blockIdx.y;
    int bn_start = kv_tile * BN;
    if (bn_start >= S) return;
    int b = bh / H, h = bh % H;
    int64_t bh_off = (int64_t)(b * H + h) * S;

    const float* L_bh = L + bh_off;
    const float* D_bh = D_vals + bh_off;
    __nv_bfloat16* dK_bh = dK_out + bh_off * D;
    __nv_bfloat16* dV_bh = dV_out + bh_off * D;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* K_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* V_smem = K_smem + 128 * 128;
    __nv_bfloat16* Q_smem = V_smem + 128 * 128;
    __nv_bfloat16* dO_smem = Q_smem + 128 * 128;
    float* S_smem = reinterpret_cast<float*>(dO_smem + 128 * 128);
    __nv_bfloat16* P_smem = reinterpret_cast<__nv_bfloat16*>(S_smem + 128 * 128);
    float* L_smem = reinterpret_cast<float*>(P_smem + 128 * 128);
    float* D_smem = L_smem + 128;

    __shared__ __align__(8) uint64_t bar;

    int tid = threadIdx.x;
    int warp_id = tid / 32;

    if (tid == 0) { mbarrier_init(&bar, 1); fence_mbarrier_init(); }
    __syncthreads();

    wmma::fragment<wmma::accumulator, WM, WN, WK, float> dK_frag[DKV_TILES_PER_WARP];
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> dV_frag[DKV_TILES_PER_WARP];
    #pragma unroll
    for (int i = 0; i < DKV_TILES_PER_WARP; i++) {
        wmma::fill_fragment(dK_frag[i], 0.0f);
        wmma::fill_fragment(dV_frag[i], 0.0f);
    }

    int kv_coord = (int)(bh_off + bn_start);
    uint32_t tma_phase = 0;

    // Load K, V via TMA
    if (tid == 0) {
        mbarrier_arrive_etx(&bar, 2 * 128 * 128 * 2);
        tma_load(&tma_K, &bar, K_smem, 0, kv_coord);
        tma_load(&tma_V, &bar, V_smem, 0, kv_coord);
    }
    mbarrier_wait(&bar, tma_phase); tma_phase ^= 1;
    fence_async_shared();

    const float scale = 0.08838834764831845f;
    int num_q_tiles = (S + BM - 1) / BM;

    for (int qt = 0; qt < num_q_tiles; qt++) {
        int bm_start = qt * BM;
        int q_coord = (int)(bh_off + bm_start);

        // Load Q, dO via TMA
        if (tid == 0) {
            mbarrier_arrive_etx(&bar, 2 * 128 * 128 * 2);
            tma_load(&tma_Q, &bar, Q_smem, 0, q_coord);
            tma_load(&tma_dO, &bar, dO_smem, 0, q_coord);
        }
        mbarrier_wait(&bar, tma_phase); tma_phase ^= 1;
        fence_async_shared();

        // Load L, D
        if (tid < 128) {
            int gr = bm_start + tid;
            L_smem[tid] = (gr < S) ? L_bh[gr] : 0.0f;
            D_smem[tid] = (gr < S) ? D_bh[gr] : 0.0f;
        }
        __syncthreads();

        // S = Q @ K^T
        #pragma unroll
        for (int ti = 0; ti < TILES_PER_WARP; ti++) {
            int tidx = warp_id * TILES_PER_WARP + ti;
            int tm = tidx / (BN / WN), tn = tidx % (BN / WN);
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
        for (int i = tid; i < 128 * 128; i += THREADS) {
            int row = i / 128, col = i % 128;
            int gr = bm_start + row, gc = bn_start + col;
            float val = (gr < S && gc < S) ? __expf(S_smem[i] * scale - L_smem[row]) : 0.0f;
            P_smem[i] = __float2bfloat16(val);
        }
        __syncthreads();

        // dP = dO @ V^T (reuse S_smem)
        #pragma unroll
        for (int ti = 0; ti < TILES_PER_WARP; ti++) {
            int tidx = warp_id * TILES_PER_WARP + ti;
            int tm = tidx / (BN / WN), tn = tidx % (BN / WN);
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
        for (int ti = 0; ti < DKV_TILES_PER_WARP; ti++) {
            int tidx = warp_id * DKV_TILES_PER_WARP + ti;
            int tm = tidx / (D / WN), tn = tidx % (D / WN);
            #pragma unroll
            for (int kk = 0; kk < BM / WK; kk++) {
                wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::col_major> a_frag;
                wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::row_major> b_frag;
                wmma::load_matrix_sync(a_frag, P_smem + kk * WK * BN + tm * WM, BN);
                wmma::load_matrix_sync(b_frag, dO_smem + kk * WK * D + tn * WN, D);
                wmma::mma_sync(dV_frag[ti], a_frag, b_frag, dV_frag[ti]);
            }
        }

        // dS = P * (dP - D) * scale (overwrite P_smem)
        for (int i = tid; i < 128 * 128; i += THREADS) {
            int row = i / 128, col = i % 128;
            int gr = bm_start + row, gc = bn_start + col;
            float p = __bfloat162float(P_smem[i]);
            float dp = S_smem[i];
            float val = (gr < S && gc < S) ? p * (dp - D_smem[row]) * scale : 0.0f;
            P_smem[i] = __float2bfloat16(val);
        }
        __syncthreads();

        // dK += dS^T @ Q
        #pragma unroll
        for (int ti = 0; ti < DKV_TILES_PER_WARP; ti++) {
            int tidx = warp_id * DKV_TILES_PER_WARP + ti;
            int tm = tidx / (D / WN), tn = tidx % (D / WN);
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

    // Epilogue: store dK
    #pragma unroll
    for (int ti = 0; ti < DKV_TILES_PER_WARP; ti++) {
        int tidx = warp_id * DKV_TILES_PER_WARP + ti;
        int tm = tidx / (D / WN), tn = tidx % (D / WN);
        wmma::store_matrix_sync(S_smem + tm * WM * D + tn * WN, dK_frag[ti], D, wmma::mem_row_major);
    }
    __syncthreads();
    for (int i = tid; i < BN * D; i += THREADS) {
        int row = i / D, col = i % D;
        int gr = bn_start + row;
        if (gr < S) dK_bh[(size_t)gr * D + col] = __float2bfloat16(S_smem[i]);
    }
    __syncthreads();

    // Store dV
    #pragma unroll
    for (int ti = 0; ti < DKV_TILES_PER_WARP; ti++) {
        int tidx = warp_id * DKV_TILES_PER_WARP + ti;
        int tm = tidx / (D / WN), tn = tidx % (D / WN);
        wmma::store_matrix_sync(S_smem + tm * WM * D + tn * WN, dV_frag[ti], D, wmma::mem_row_major);
    }
    __syncthreads();
    for (int i = tid; i < BN * D; i += THREADS) {
        int row = i / D, col = i % D;
        int gr = bn_start + row;
        if (gr < S) dV_bh[(size_t)gr * D + col] = __float2bfloat16(S_smem[i]);
    }
}

__global__ __launch_bounds__(THREADS, 1)
void dQ_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* __restrict__ L,
    const float* __restrict__ D_vals,
    __nv_bfloat16* __restrict__ dQ_out,
    int B, int H, int S)
{
    int bh = blockIdx.x;
    int q_tile = blockIdx.y;
    int bm_start = q_tile * BM;
    if (bm_start >= S) return;
    int b = bh / H, h = bh % H;
    int64_t bh_off = (int64_t)(b * H + h) * S;

    const float* L_bh = L + bh_off;
    const float* D_bh = D_vals + bh_off;
    __nv_bfloat16* dQ_bh = dQ_out + bh_off * D;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* Q_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* dO_smem = Q_smem + 128 * 128;
    __nv_bfloat16* K_smem = dO_smem + 128 * 128;
    __nv_bfloat16* V_smem = K_smem + 128 * 128;
    float* S_smem = reinterpret_cast<float*>(V_smem + 128 * 128);
    __nv_bfloat16* P_smem = reinterpret_cast<__nv_bfloat16*>(S_smem + 128 * 128);
    float* L_smem = reinterpret_cast<float*>(P_smem + 128 * 128);
    float* D_smem = L_smem + 128;

    __shared__ __align__(8) uint64_t bar;

    int tid = threadIdx.x;
    int warp_id = tid / 32;

    if (tid == 0) { mbarrier_init(&bar, 1); fence_mbarrier_init(); }
    __syncthreads();

    wmma::fragment<wmma::accumulator, WM, WN, WK, float> dQ_frag[DQ_TILES_PER_WARP];
    #pragma unroll
    for (int i = 0; i < DQ_TILES_PER_WARP; i++) wmma::fill_fragment(dQ_frag[i], 0.0f);

    int q_coord = (int)(bh_off + bm_start);
    uint32_t tma_phase = 0;

    // Load Q, dO once
    if (tid == 0) {
        mbarrier_arrive_etx(&bar, 2 * 128 * 128 * 2);
        tma_load(&tma_Q, &bar, Q_smem, 0, q_coord);
        tma_load(&tma_dO, &bar, dO_smem, 0, q_coord);
    }
    mbarrier_wait(&bar, tma_phase); tma_phase ^= 1;
    fence_async_shared();

    // Load L, D once
    if (tid < 128) {
        int gr = bm_start + tid;
        L_smem[tid] = (gr < S) ? L_bh[gr] : 0.0f;
        D_smem[tid] = (gr < S) ? D_bh[gr] : 0.0f;
    }
    __syncthreads();

    const float scale = 0.08838834764831845f;
    int num_kv_tiles = (S + BN - 1) / BN;

    for (int kt = 0; kt < num_kv_tiles; kt++) {
        int bn_start = kt * BN;
        int kv_coord = (int)(bh_off + bn_start);

        // Load K, V
        if (tid == 0) {
            mbarrier_arrive_etx(&bar, 2 * 128 * 128 * 2);
            tma_load(&tma_K, &bar, K_smem, 0, kv_coord);
            tma_load(&tma_V, &bar, V_smem, 0, kv_coord);
        }
        mbarrier_wait(&bar, tma_phase); tma_phase ^= 1;
        fence_async_shared();

        // S = Q @ K^T
        #pragma unroll
        for (int ti = 0; ti < TILES_PER_WARP; ti++) {
            int tidx = warp_id * TILES_PER_WARP + ti;
            int tm = tidx / (BN / WN), tn = tidx % (BN / WN);
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
        for (int i = tid; i < 128 * 128; i += THREADS) {
            int row = i / 128, col = i % 128;
            int gr = bm_start + row, gc = bn_start + col;
            float val = (gr < S && gc < S) ? __expf(S_smem[i] * scale - L_smem[row]) : 0.0f;
            P_smem[i] = __float2bfloat16(val);
        }
        __syncthreads();

        // dP = dO @ V^T (reuse S_smem)
        #pragma unroll
        for (int ti = 0; ti < TILES_PER_WARP; ti++) {
            int tidx = warp_id * TILES_PER_WARP + ti;
            int tm = tidx / (BN / WN), tn = tidx % (BN / WN);
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
        for (int i = tid; i < 128 * 128; i += THREADS) {
            int row = i / 128, col = i % 128;
            int gr = bm_start + row, gc = bn_start + col;
            float p = __bfloat162float(P_smem[i]);
            float dp = S_smem[i];
            float val = (gr < S && gc < S) ? p * (dp - D_smem[row]) * scale : 0.0f;
            P_smem[i] = __float2bfloat16(val);
        }
        __syncthreads();

        // dQ += dS @ K [BM][D]
        #pragma unroll
        for (int ti = 0; ti < DQ_TILES_PER_WARP; ti++) {
            int tidx = warp_id * DQ_TILES_PER_WARP + ti;
            int tm = tidx / (D / WN), tn = tidx % (D / WN);
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
    #pragma unroll
    for (int ti = 0; ti < DQ_TILES_PER_WARP; ti++) {
        int tidx = warp_id * DQ_TILES_PER_WARP + ti;
        int tm = tidx / (D / WN), tn = tidx % (D / WN);
        wmma::store_matrix_sync(S_smem + tm * WM * D + tn * WN, dQ_frag[ti], D, wmma::mem_row_major);
    }
    __syncthreads();
    for (int i = tid; i < BM * D; i += THREADS) {
        int row = i / D, col = i % D;
        int gr = bm_start + row;
        if (gr < S) dQ_bh[(size_t)gr * D + col] = __float2bfloat16(S_smem[i]);
    }
}

CUresult create_tma_desc(CUtensorMap* d, void* ptr, uint64_t inner, uint64_t outer, uint32_t box_outer) {
    cuuint64_t globalDim[2] = {inner, outer};
    cuuint64_t globalStrides[1] = {inner * 2};
    cuuint32_t boxDim[2] = {(uint32_t)inner, box_outer};
    cuuint32_t elemStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        ptr, globalDim, globalStrides, boxDim, elemStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
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

    CUtensorMap tma_Q, tma_K, tma_V, tma_dO;
    int64_t total_outer = (int64_t)B * H * S;
    CU_CHECK(create_tma_desc(&tma_Q, Q.data_ptr(), D, total_outer, 128));
    CU_CHECK(create_tma_desc(&tma_K, K.data_ptr(), D, total_outer, 128));
    CU_CHECK(create_tma_desc(&tma_V, V.data_ptr(), D, total_outer, 128));
    CU_CHECK(create_tma_desc(&tma_dO, dO.data_ptr(), D, total_outer, 128));

    int smem_size = 4 * 128 * 128 * 2 + 128 * 128 * 4 + 128 * 128 * 2 + 1024;

    {
        dim3 grid(B * H, (S + BN - 1) / BN, 1);
        dim3 block(THREADS, 1, 1);
        CUDA_CHECK(cudaFuncSetAttribute(dK_dV_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        dK_dV_kernel<<<grid, block, smem_size, stream>>>(
            tma_Q, tma_K, tma_V, tma_dO,
            static_cast<const float*>(L.data_ptr()), D_temp,
            static_cast<__nv_bfloat16*>(dK.data_ptr()),
            static_cast<__nv_bfloat16*>(dV.data_ptr()), B, H, S);
    }

    {
        dim3 grid(B * H, (S + BM - 1) / BM, 1);
        dim3 block(THREADS, 1, 1);
        CUDA_CHECK(cudaFuncSetAttribute(dQ_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        dQ_kernel<<<grid, block, smem_size, stream>>>(
            tma_Q, tma_K, tma_V, tma_dO,
            static_cast<const float*>(L.data_ptr()), D_temp,
            static_cast<__nv_bfloat16*>(dQ.data_ptr()), B, H, S);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(D_temp));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flash_attn_bwd::run);

}  // namespace flash_attn_bwd