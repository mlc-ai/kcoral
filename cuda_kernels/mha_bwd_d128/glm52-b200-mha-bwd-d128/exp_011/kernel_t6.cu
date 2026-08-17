#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda.h>
#include <cstdint>
#include <cmath>
#include <stdio.h>
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

using namespace nvcuda;

namespace attn_bwd {

constexpr int BM = 64;
constexpr int BN = 32;
constexpr int D = 128;
constexpr int WARPS = 8;
constexpr int THREADS = 256;
constexpr float SCALE = 0.08838834764831843f;

// K-major layout: dK/dV in SMEM (no atomics), dQ via atomicAdd
// SMEM: K(8KB) + V/dP(8KB) + dK_acc(16KB) + dV_acc(16KB) + Q(16KB) + dO(16KB) + S(8KB) + P/dS(4KB) + D/L(0.25KB) = ~92KB
constexpr int K_OFF    = 0;
constexpr int V_OFF    = K_OFF  + BN * D * 2;          // 8192
constexpr int dK_OFF   = V_OFF  + BN * D * 2;          // 16384
constexpr int dV_OFF   = dK_OFF + BN * D * 4;          // 32768
constexpr int Q_OFF    = dV_OFF + BN * D * 4;          // 49152
constexpr int dO_OFF   = Q_OFF  + BM * D * 2;          // 65536
constexpr int S_OFF    = dO_OFF + BM * D * 2;          // 81920
constexpr int P_OFF    = S_OFF  + BM * BN * 4;          // 86016
constexpr int D_OFF    = P_OFF  + BM * BN * 2;          // 90112
constexpr int L_OFF    = D_OFF  + BM * 4;                // 90368
constexpr int SMEM_SIZE = L_OFF + BM * 4;                // 90624

__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;\n" ::: "memory"); }

__device__ __forceinline__ void load_tile_async_bf16(
    __nv_bfloat16* smem, const __nv_bfloat16* gmem, int rows, int row_start, int S, uint64_t bh_off)
{
    constexpr int VEC = 8;
    int total = rows * (D / VEC);
    for (int idx = threadIdx.x; idx < total; idx += THREADS) {
        int i = idx / (D / VEC), dd = idx % (D / VEC);
        int gr = row_start + i;
        if (gr < S)
            cp_async_16(&smem[i * D + dd * VEC], &gmem[bh_off + (uint64_t)gr * D + dd * VEC]);
        else {
            int4 z = make_int4(0, 0, 0, 0);
            *reinterpret_cast<int4*>(&smem[i * D + dd * VEC]) = z;
        }
    }
}

// C[M,N] = A[M,K] @ B[K,N]^T * scale (B stored row_major as [N,K])
template<int M, int N, int K>
__device__ __forceinline__ void wmma_gemm_rc(
    const __nv_bfloat16* A_smem, const __nv_bfloat16* B_smem,
    float* C_smem, float scale, int warp_id)
{
    constexpr int M_T = M/16, N_T = N/16, K_T = K/16;
    constexpr int TOT = M_T * N_T;
    for (int t = 0; t < (TOT + WARPS - 1) / WARPS; t++) {
        int ti = warp_id + t * WARPS;
        if (ti >= TOT) break;
        int mt = ti / N_T, nt = ti % N_T;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
        wmma::fill_fragment(c, 0.0f);
        for (int kt = 0; kt < K_T; kt++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
            wmma::load_matrix_sync(a, A_smem + mt*16*K + kt*16, K);
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
            wmma::load_matrix_sync(b, B_smem + nt*16*K + kt*16, K);
            wmma::mma_sync(c, a, b, c);
        }
        if (scale != 1.0f) for (int i = 0; i < c.num_elements; i++) c.x[i] *= scale;
        wmma::store_matrix_sync(C_smem + mt*16*N + nt*16, c, N, wmma::mem_row_major);
    }
}

// dV_acc[BN,D] += P^T[BN,BM] @ dO[BM,D] (accumulate in SMEM)
__device__ __forceinline__ void compute_dV_acc(
    const __nv_bfloat16* P_smem, const __nv_bfloat16* dO_smem,
    float* dV_acc, int warp_id)
{
    constexpr int M = BN, N = D, K = BM;
    constexpr int M_T = M/16, N_T = N/16, K_T = K/16;
    constexpr int TOT = M_T * N_T;
    for (int t = 0; t < (TOT + WARPS - 1) / WARPS; t++) {
        int ti = warp_id + t * WARPS;
        if (ti >= TOT) break;
        int mt = ti / N_T, nt = ti % N_T;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
        wmma::load_matrix_sync(c, dV_acc + mt*16*D + nt*16, D, wmma::mem_row_major);
        for (int kt = 0; kt < K_T; kt++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a;
            wmma::load_matrix_sync(a, P_smem + kt*16*BN + mt*16, BN);
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
            wmma::load_matrix_sync(b, dO_smem + kt*16*D + nt*16, D);
            wmma::mma_sync(c, a, b, c);
        }
        wmma::store_matrix_sync(dV_acc + mt*16*D + nt*16, c, D, wmma::mem_row_major);
    }
}

// dK_acc[BN,D] += dS^T[BN,BM] @ Q[BM,D] (accumulate in SMEM)
__device__ __forceinline__ void compute_dK_acc(
    const __nv_bfloat16* dS_smem, const __nv_bfloat16* Q_smem,
    float* dK_acc, int warp_id)
{
    constexpr int M = BN, N = D, K = BM;
    constexpr int M_T = M/16, N_T = N/16, K_T = K/16;
    constexpr int TOT = M_T * N_T;
    for (int t = 0; t < (TOT + WARPS - 1) / WARPS; t++) {
        int ti = warp_id + t * WARPS;
        if (ti >= TOT) break;
        int mt = ti / N_T, nt = ti % N_T;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
        wmma::load_matrix_sync(c, dK_acc + mt*16*D + nt*16, D, wmma::mem_row_major);
        for (int kt = 0; kt < K_T; kt++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a;
            wmma::load_matrix_sync(a, dS_smem + kt*16*BN + mt*16, BN);
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
            wmma::load_matrix_sync(b, Q_smem + kt*16*D + nt*16, D);
            wmma::mma_sync(c, a, b, c);
        }
        wmma::store_matrix_sync(dK_acc + mt*16*D + nt*16, c, D, wmma::mem_row_major);
    }
}

// dQ[BM,D] += dS[BM,BN] @ K[BN,D] (atomicAdd to global fp32)
__device__ __forceinline__ void compute_dQ_atomic(
    const __nv_bfloat16* dS_smem, const __nv_bfloat16* K_smem,
    float* dQ_gmem, int q_start, uint64_t bh_off, int S,
    float* tmp_smem, int warp_id)
{
    constexpr int M = BM, N = D, K = BN;
    constexpr int M_T = M/16, N_T = N/16, K_T = K/16;
    constexpr int TOT = M_T * N_T;
    int lane_id = threadIdx.x % 32;
    for (int t = 0; t < (TOT + WARPS - 1) / WARPS; t++) {
        int ti = warp_id + t * WARPS;
        if (ti >= TOT) break;
        int mt = ti / N_T, nt = ti % N_T;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
        wmma::fill_fragment(c, 0.0f);
        for (int kt = 0; kt < K_T; kt++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
            wmma::load_matrix_sync(a, dS_smem + mt*16*BN + kt*16, BN);
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
            wmma::load_matrix_sync(b, K_smem + kt*16*D + nt*16, D);
            wmma::mma_sync(c, a, b, c);
        }
        float* tile = tmp_smem + warp_id * 256;
        wmma::store_matrix_sync(tile, c, 16, wmma::mem_row_major);
        __syncwarp();
        int row_base = q_start + mt * 16;
        int col_base = nt * 16;
        for (int idx = lane_id; idx < 256; idx += 32) {
            int lr = idx / 16, lc = idx % 16;
            int gr = row_base + lr;
            if (gr < S)
                atomicAdd(&dQ_gmem[bh_off + (uint64_t)gr * D + col_base + lc], tile[lr * 16 + lc]);
        }
        __syncwarp();
    }
}

__global__ __launch_bounds__(256, 2)
void attn_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    float* __restrict__ dQ_fp32,
    __nv_bfloat16* __restrict__ dK_out,
    __nv_bfloat16* __restrict__ dV_out,
    int S)
{
    int bh = blockIdx.x;
    int k_blk = blockIdx.y;
    int k_start = k_blk * BN;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    if (k_start >= S) return;

    extern __shared__ char smem[];
    __nv_bfloat16* K_s    = reinterpret_cast<__nv_bfloat16*>(smem + K_OFF);
    __nv_bfloat16* V_s    = reinterpret_cast<__nv_bfloat16*>(smem + V_OFF);  // reused for dP
    float* dK_acc         = reinterpret_cast<float*>(smem + dK_OFF);
    float* dV_acc         = reinterpret_cast<float*>(smem + dV_OFF);
    __nv_bfloat16* Q_s    = reinterpret_cast<__nv_bfloat16*>(smem + Q_OFF);
    __nv_bfloat16* dO_s   = reinterpret_cast<__nv_bfloat16*>(smem + dO_OFF);
    float* S_s            = reinterpret_cast<float*>(smem + S_OFF);
    __nv_bfloat16* P_s    = reinterpret_cast<__nv_bfloat16*>(smem + P_OFF);  // reused for dS
    float* D_s            = reinterpret_cast<float*>(smem + D_OFF);
    float* L_s            = reinterpret_cast<float*>(smem + L_OFF);

    uint64_t bh_off = (uint64_t)bh * S * D;

    // Load K (persistent throughout kernel)
    load_tile_async_bf16(K_s, K, BN, k_start, S, bh_off);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    // Zero dK_acc, dV_acc
    for (int idx = tid; idx < BN * D; idx += THREADS) {
        dK_acc[idx] = 0.0f;
        dV_acc[idx] = 0.0f;
    }
    __syncthreads();

    int num_q = (S + BM - 1) / BM;
    for (int qj = 0; qj < num_q; qj++) {
        int q_start = qj * BM;

        // Load Q, dO via cp.async
        load_tile_async_bf16(Q_s, Q, BM, q_start, S, bh_off);
        load_tile_async_bf16(dO_s, dO, BM, q_start, S, bh_off);
        // Load V into V_s (will be reused for dP later)
        load_tile_async_bf16(V_s, V, BN, k_start, S, bh_off);
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();

        // Load L
        for (int i = tid; i < BM; i += THREADS)
            L_s[i] = (q_start + i < S) ? L[bh * S + q_start + i] : 0.0f;

        // Compute D = rowsum(dO * O) - load O into S_s temporarily
        {
            __nv_bfloat16* O_s = reinterpret_cast<__nv_bfloat16*>(S_s);
            load_tile_async_bf16(O_s, O, BM, q_start, S, bh_off);
            cp_async_commit();
            cp_async_wait_all();
        }
        __syncthreads();

        for (int i = warp_id; i < BM; i += WARPS) {
            float sum = 0.0f;
            __nv_bfloat16* O_s = reinterpret_cast<__nv_bfloat16*>(S_s);
            for (int dd = lane_id; dd < D; dd += 32)
                sum += __bfloat162float(dO_s[i*D+dd]) * __bfloat162float(O_s[i*D+dd]);
            for (int offset = 16; offset > 0; offset /= 2)
                sum += __shfl_xor_sync(0xffffffff, sum, offset);
            if (lane_id == 0) D_s[i] = sum;
        }
        __syncthreads();

        // S = Q @ K^T * scale (into S_s, overwriting O)
        wmma_gemm_rc<BM, BN, D>(Q_s, K_s, S_s, SCALE, warp_id);
        __syncthreads();

        // dP = dO @ V^T (into V_s, overwriting V - V no longer needed)
        wmma_gemm_rc<BM, BN, D>(dO_s, V_s, reinterpret_cast<float*>(V_s), 1.0f, warp_id);
        // Wait, this overwrites V_s while reading from it. Need separate dP buffer.
        // Actually V_s is read as B matrix in wmma (col_major), and dP is written as accumulator.
        // The wmma stores dP to a separate location. Let me use S_s area after S is done.
        // But S_s is needed for softmax. Let me think...
        // Actually, dP = dO @ V^T writes to accumulator then stores. If I store to V_s, it
        // overwrites V which is being read. This is a race condition.
        // Fix: store dP to a different location. Use P_s area (4KB) - but dP is 8KB.
        // Use D_s + L_s area? Only 512B. Not enough.
        // Let me use a separate dP area.

        // Actually, the wmma_mma_sync reads from fragments (already loaded from SMEM), 
        // then stores to SMEM. The load happens before the store. But with pipelining...
        // To be safe, let me not reuse V_s for dP. Instead, keep dP in S_s after S is used.
        // But S_s is needed for softmax. We need both S and dP simultaneously.
        // OK let me just use the P_s area for dP. P_s is BM*BN*2 = 4KB bf16. dP is BM*BN*4 = 8KB fp32.
        // Not enough. Let me restructure SMEM.

        // Actually, let me store dP into the dV_acc buffer temporarily? No, dV_acc is persistent.
        // Let me use the area after L_s for dP. That would be L_OFF + BM*4 = 90368 + 256 = 90624.
        // dP needs 8KB. Total SMEM = 90624 + 8192 = 98816. Still fits 2 blocks/SM!

        // Let me redefine: dP goes right after L_s
        // SMEM_SIZE becomes 98816
        // But I defined SMEM_SIZE as 90624. Let me fix this.

        // For now, let me just not reuse V and use a separate dP buffer.
        // I'll redefine the SMEM layout.
    }
    // This kernel won't compile correctly due to the dP buffer issue.
    // Let me rewrite it properly.
}

// Let me rewrite the kernel with proper SMEM layout
// K-major: BM=64, BN=32, D=128
// SMEM: K(8KB) + V(8KB) + dK_acc(16KB) + dV_acc(16KB) + Q(16KB) + dO(16KB) + S(8KB) + dP(8KB) + P/dS(4KB) + D/L(0.5KB) = ~100KB

constexpr int K2_OFF    = 0;
constexpr int V2_OFF    = K2_OFF  + BN * D * 2;          // 8192
constexpr int dK2_OFF   = V2_OFF  + BN * D * 2;          // 16384
constexpr int dV2_OFF   = dK2_OFF + BN * D * 4;          // 32768
constexpr int Q2_OFF    = dV2_OFF + BN * D * 4;          // 49152
constexpr int dO2_OFF   = Q2_OFF  + BM * D * 2;          // 65536
constexpr int S2_OFF    = dO2_OFF + BM * D * 2;          // 81920
constexpr int dP2_OFF   = S2_OFF  + BM * BN * 4;          // 86016
constexpr int P2_OFF    = dP2_OFF + BM * BN * 4;          // 90112
constexpr int D2_OFF    = P2_OFF  + BM * BN * 2;          // 94208
constexpr int L2_OFF    = D2_OFF  + BM * 4;                // 94464
constexpr int TMP2_OFF  = L2_OFF  + BM * 4;                // 94720
constexpr int SMEM2_SIZE = TMP2_OFF + WARPS * 16 * 16 * 4; // 102912

__global__ __launch_bounds__(256, 2)
void attn_bwd_kernel_v2(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    float* __restrict__ dQ_fp32,
    __nv_bfloat16* __restrict__ dK_out,
    __nv_bfloat16* __restrict__ dV_out,
    int S)
{
    int bh = blockIdx.x;
    int k_blk = blockIdx.y;
    int k_start = k_blk * BN;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    if (k_start >= S) return;

    extern __shared__ char smem[];
    __nv_bfloat16* K_s    = reinterpret_cast<__nv_bfloat16*>(smem + K2_OFF);
    __nv_bfloat16* V_s    = reinterpret_cast<__nv_bfloat16*>(smem + V2_OFF);
    float* dK_acc         = reinterpret_cast<float*>(smem + dK2_OFF);
    float* dV_acc         = reinterpret_cast<float*>(smem + dV2_OFF);
    __nv_bfloat16* Q_s    = reinterpret_cast<__nv_bfloat16*>(smem + Q2_OFF);
    __nv_bfloat16* dO_s   = reinterpret_cast<__nv_bfloat16*>(smem + dO2_OFF);
    float* S_s            = reinterpret_cast<float*>(smem + S2_OFF);
    float* dP_s           = reinterpret_cast<float*>(smem + dP2_OFF);
    __nv_bfloat16* P_s    = reinterpret_cast<__nv_bfloat16*>(smem + P2_OFF);
    float* D_s            = reinterpret_cast<float*>(smem + D2_OFF);
    float* L_s            = reinterpret_cast<float*>(smem + L2_OFF);
    float* tmp_s          = reinterpret_cast<float*>(smem + TMP2_OFF);

    uint64_t bh_off = (uint64_t)bh * S * D;

    // Load K, V (persistent)
    load_tile_async_bf16(K_s, K, BN, k_start, S, bh_off);
    load_tile_async_bf16(V_s, V, BN, k_start, S, bh_off);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    // Zero dK_acc, dV_acc
    for (int idx = tid; idx < BN * D; idx += THREADS) {
        dK_acc[idx] = 0.0f;
        dV_acc[idx] = 0.0f;
    }
    __syncthreads();

    int num_q = (S + BM - 1) / BM;
    for (int qj = 0; qj < num_q; qj++) {
        int q_start = qj * BM;

        // Load Q, dO via cp.async
        load_tile_async_bf16(Q_s, Q, BM, q_start, S, bh_off);
        load_tile_async_bf16(dO_s, dO, BM, q_start, S, bh_off);
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();

        // Load L
        for (int i = tid; i < BM; i += THREADS)
            L_s[i] = (q_start + i < S) ? L[bh * S + q_start + i] : 0.0f;

        // Compute D = rowsum(dO * O) - load O into S_s temporarily
        {
            __nv_bfloat16* O_s = reinterpret_cast<__nv_bfloat16*>(S_s);
            load_tile_async_bf16(O_s, O, BM, q_start, S, bh_off);
            cp_async_commit();
            cp_async_wait_all();
        }
        __syncthreads();

        for (int i = warp_id; i < BM; i += WARPS) {
            float sum = 0.0f;
            __nv_bfloat16* O_s = reinterpret_cast<__nv_bfloat16*>(S_s);
            for (int dd = lane_id; dd < D; dd += 32)
                sum += __bfloat162float(dO_s[i*D+dd]) * __bfloat162float(O_s[i*D+dd]);
            for (int offset = 16; offset > 0; offset /= 2)
                sum += __shfl_xor_sync(0xffffffff, sum, offset);
            if (lane_id == 0) D_s[i] = sum;
        }
        __syncthreads();

        // S = Q @ K^T * scale (overwrites O in S_s)
        wmma_gemm_rc<BM, BN, D>(Q_s, K_s, S_s, SCALE, warp_id);
        __syncthreads();

        // dP = dO @ V^T
        wmma_gemm_rc<BM, BN, D>(dO_s, V_s, dP_s, 1.0f, warp_id);
        __syncthreads();

        // P = exp(S - L), store bf16 in P_s, keep fp32 in S_s
        for (int idx = tid; idx < BM * BN; idx += THREADS) {
            int i = idx / BN, j = idx % BN;
            int kr = k_start + j;
            if (kr >= S) {
                S_s[idx] = 0.0f;
                P_s[idx] = __float2bfloat16(0.0f);
            } else {
                float pv = expf(S_s[idx] - L_s[i]);
                S_s[idx] = pv;
                P_s[idx] = __float2bfloat16(pv);
            }
        }
        __syncthreads();

        // dV_acc += P^T @ dO (in SMEM, no atomics!)
        compute_dV_acc(P_s, dO_s, dV_acc, warp_id);
        __syncthreads();

        // dS = P * (dP - D) * scale, store bf16 in P_s (overwrite P)
        for (int idx = tid; idx < BM * BN; idx += THREADS) {
            int i = idx / BN;
            float pv = S_s[idx];
            float dpv = dP_s[idx];
            P_s[idx] = __float2bfloat16(pv * (dpv - D_s[i]) * SCALE);
        }
        __syncthreads();

        // dK_acc += dS^T @ Q (in SMEM, no atomics!)
        compute_dK_acc(P_s, Q_s, dK_acc, warp_id);
        __syncthreads();

        // dQ += dS @ K (atomicAdd to global fp32)
        compute_dQ_atomic(P_s, K_s, dQ_fp32, q_start, bh_off, S, tmp_s, warp_id);
        __syncthreads();
    }

    // Store dK, dV from SMEM to global bf16
    for (int idx = tid; idx < BN * D; idx += THREADS) {
        int i = idx / D, dd = idx % D;
        int kr = k_start + i;
        if (kr < S) {
            dK_out[bh_off + (uint64_t)kr * D + dd] = __float2bfloat16(dK_acc[idx]);
            dV_out[bh_off + (uint64_t)kr * D + dd] = __float2bfloat16(dV_acc[idx]);
        }
    }
}

__global__ void convert_fp32_to_bf16_kernel(
    const float* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __float2bfloat16(src[idx]);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV)
{
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = static_cast<int>(Q.size(0));
    int H = static_cast<int>(Q.size(1));
    int S = static_cast<int>(Q.size(2));
    int BH = B * H;

    const __nv_bfloat16* Q_p  = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_p  = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_p  = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_p  = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_p = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_p          = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_p       = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_p       = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_p       = static_cast<__nv_bfloat16*>(dV.data_ptr());

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    size_t total_elems = (size_t)BH * S * D;
    size_t fp32_bytes = total_elems * sizeof(float);

    float* dQ_fp32 = nullptr;
    CUDA_CHECK(cudaMallocAsync(&dQ_fp32, fp32_bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(dQ_fp32, 0, fp32_bytes, stream));

    int num_k_blocks = (S + BN - 1) / BN;
    dim3 grid(BH, num_k_blocks, 1);
    dim3 block(THREADS, 1, 1);

    CUDA_CHECK(cudaFuncSetAttribute(
        attn_bwd_kernel_v2, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM2_SIZE));

    attn_bwd_kernel_v2<<<grid, block, SMEM2_SIZE, stream>>>(
        Q_p, K_p, V_p, O_p, dO_p, L_p, dQ_fp32, dK_p, dV_p, S);

    int conv_threads = 256;
    int conv_blocks = (total_elems + conv_threads - 1) / conv_threads;
    convert_fp32_to_bf16_kernel<<<conv_blocks, conv_threads, 0, stream>>>(dQ_fp32, dQ_p, total_elems);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFreeAsync(dQ_fp32, stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, attn_bwd::run);

} // namespace attn_bwd