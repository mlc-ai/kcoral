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
constexpr int BN = 64;
constexpr int D = 128;
constexpr int WARPS = 8;
constexpr int THREADS = 256;
constexpr float SCALE = 0.08838834764831843f;

// Shared memory layout (~104KB, allows 2 blocks/SM)
constexpr int Q_OFF  = 0;
constexpr int dO_OFF = Q_OFF  + BM * D * 2;       // 16384
constexpr int K_OFF  = dO_OFF + BM * D * 2;       // 32768
constexpr int V_OFF  = K_OFF  + BN * D * 2;       // 49152
constexpr int S_OFF  = V_OFF  + BN * D * 2;       // 65536
constexpr int dP_OFF = S_OFF  + BM * BN * 4;       // 81920
constexpr int P_OFF  = dP_OFF + BM * BN * 4;       // 98304
constexpr int D_OFF  = P_OFF  + BM * BN * 2;       // 106496
constexpr int L_OFF  = D_OFF  + BM * 4;             // 106752
constexpr int SMEM_SIZE = L_OFF + BM * 4;           // 107008

__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}
__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::: "memory");
}

// C[M,N] = A[M,K] @ B[K,N]^T * scale
// A: row_major [M,K], B: row_major [N,K] (loaded as col_major)
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
        if (scale != 1.0f)
            for (int i = 0; i < c.num_elements; i++) c.x[i] *= scale;
        wmma::store_matrix_sync(C_smem + mt*16*N + nt*16, c, N, wmma::mem_row_major);
    }
}

// dV[BN,D] += P^T[BN,BM] @ dO[BM,D] -- atomicAdd to global fp32
__device__ __forceinline__ void compute_dV_atomic(
    const __nv_bfloat16* P_smem, const __nv_bfloat16* dO_smem,
    float* dV_gmem, int k_start, uint64_t bh_off, int S,
    float* tmp_smem, int warp_id)
{
    constexpr int M = BN, N = D, K = BM;
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
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a;
            wmma::load_matrix_sync(a, P_smem + kt*16*BN + mt*16, BN);
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
            wmma::load_matrix_sync(b, dO_smem + kt*16*D + nt*16, D);
            wmma::mma_sync(c, a, b, c);
        }
        float* tile = tmp_smem + warp_id * 256;
        wmma::store_matrix_sync(tile, c, 16, wmma::mem_row_major);
        __syncwarp();
        int row_base = k_start + mt * 16;
        int col_base = nt * 16;
        for (int idx = lane_id; idx < 256; idx += 32) {
            int lr = idx / 16, lc = idx % 16;
            int gr = row_base + lr;
            if (gr < S)
                atomicAdd(&dV_gmem[bh_off + (uint64_t)gr * D + col_base + lc], tile[lr * 16 + lc]);
        }
        __syncwarp();
    }
}

// dK[BN,D] += dS^T[BN,BM] @ Q[BM,D] -- atomicAdd to global fp32
__device__ __forceinline__ void compute_dK_atomic(
    const __nv_bfloat16* dS_smem, const __nv_bfloat16* Q_smem,
    float* dK_gmem, int k_start, uint64_t bh_off, int S,
    float* tmp_smem, int warp_id)
{
    constexpr int M = BN, N = D, K = BM;
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
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a;
            wmma::load_matrix_sync(a, dS_smem + kt*16*BN + mt*16, BN);
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
            wmma::load_matrix_sync(b, Q_smem + kt*16*D + nt*16, D);
            wmma::mma_sync(c, a, b, c);
        }
        float* tile = tmp_smem + warp_id * 256;
        wmma::store_matrix_sync(tile, c, 16, wmma::mem_row_major);
        __syncwarp();
        int row_base = k_start + mt * 16;
        int col_base = nt * 16;
        for (int idx = lane_id; idx < 256; idx += 32) {
            int lr = idx / 16, lc = idx % 16;
            int gr = row_base + lr;
            if (gr < S)
                atomicAdd(&dK_gmem[bh_off + (uint64_t)gr * D + col_base + lc], tile[lr * 16 + lc]);
        }
        __syncwarp();
    }
}

// dQ[BM,D] += dS[BM,BN] @ K[BN,D] -- accumulate in register fragments
__device__ __forceinline__ void compute_dQ_acc(
    const __nv_bfloat16* dS_smem, const __nv_bfloat16* K_smem,
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dQ_frag[4],
    int warp_id)
{
    constexpr int K_T = BN / 16;
    constexpr int N_T = D / 16;
    for (int i = 0; i < 4; i++) {
        int tile = warp_id + i * WARPS;
        int mt = tile / N_T, nt = tile % N_T;
        for (int kt = 0; kt < K_T; kt++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
            wmma::load_matrix_sync(a, dS_smem + mt*16*BN + kt*16, BN);
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
            wmma::load_matrix_sync(b, K_smem + kt*16*D + nt*16, D);
            wmma::mma_sync(dQ_frag[i], a, b, dQ_frag[i]);
        }
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
    __nv_bfloat16* __restrict__ dQ_out,
    float* __restrict__ dK_fp32,
    float* __restrict__ dV_fp32,
    int S)
{
    int bh = blockIdx.x;
    int q_blk = blockIdx.y;
    int q_start = q_blk * BM;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    if (q_start >= S) return;

    extern __shared__ char smem[];
    __nv_bfloat16* Q_s  = reinterpret_cast<__nv_bfloat16*>(smem + Q_OFF);
    __nv_bfloat16* dO_s = reinterpret_cast<__nv_bfloat16*>(smem + dO_OFF);
    __nv_bfloat16* K_s  = reinterpret_cast<__nv_bfloat16*>(smem + K_OFF);
    __nv_bfloat16* V_s  = reinterpret_cast<__nv_bfloat16*>(smem + V_OFF);
    float* S_s          = reinterpret_cast<float*>(smem + S_OFF);
    float* dP_s         = reinterpret_cast<float*>(smem + dP_OFF);
    __nv_bfloat16* P_s  = reinterpret_cast<__nv_bfloat16*>(smem + P_OFF);
    float* D_s          = reinterpret_cast<float*>(smem + D_OFF);
    float* L_s          = reinterpret_cast<float*>(smem + L_OFF);

    uint64_t bh_off = (uint64_t)bh * S * D;

    // Load Q, dO, O(temp in S_s) via cp.async
    {
        constexpr int VEC = 8;
        int total = BM * (D / VEC);
        for (int idx = tid; idx < total; idx += THREADS) {
            int i = idx / (D / VEC), dd = idx % (D / VEC);
            int qr = q_start + i;
            if (qr < S) {
                cp_async_16(&Q_s[i*D + dd*VEC], &Q[bh_off + (uint64_t)qr*D + dd*VEC]);
                cp_async_16(&dO_s[i*D + dd*VEC], &dO[bh_off + (uint64_t)qr*D + dd*VEC]);
                cp_async_16(&((__nv_bfloat16*)S_s)[i*D + dd*VEC], &O[bh_off + (uint64_t)qr*D + dd*VEC]);
            } else {
                int4 z = make_int4(0, 0, 0, 0);
                *reinterpret_cast<int4*>(&Q_s[i*D + dd*VEC]) = z;
                *reinterpret_cast<int4*>(&dO_s[i*D + dd*VEC]) = z;
                *reinterpret_cast<int4*>(&((__nv_bfloat16*)S_s)[i*D + dd*VEC]) = z;
            }
        }
    }
    for (int i = tid; i < BM; i += THREADS)
        L_s[i] = (q_start + i < S) ? L[bh * S + q_start + i] : 0.0f;

    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    // Compute D = rowsum(dO * O) using warp reduction
    {
        __nv_bfloat16* O_s = reinterpret_cast<__nv_bfloat16*>(S_s);
        for (int i = warp_id; i < BM; i += WARPS) {
            float sum = 0.0f;
            for (int dd = lane_id; dd < D; dd += 32)
                sum += __bfloat162float(dO_s[i*D+dd]) * __bfloat162float(O_s[i*D+dd]);
            for (int offset = 16; offset > 0; offset /= 2)
                sum += __shfl_xor_sync(0xffffffff, sum, offset);
            if (lane_id == 0) D_s[i] = sum;
        }
    }
    __syncthreads();

    // Initialize dQ accumulator fragments (in registers)
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dQ_frag[4];
    for (int i = 0; i < 4; i++) wmma::fill_fragment(dQ_frag[i], 0.0f);

    // Main K loop
    int num_k = (S + BN - 1) / BN;
    for (int kj = 0; kj < num_k; kj++) {
        int k_start = kj * BN;

        // Load K, V via cp.async
        {
            constexpr int VEC = 8;
            int total = BN * (D / VEC);
            for (int idx = tid; idx < total; idx += THREADS) {
                int i = idx / (D / VEC), dd = idx % (D / VEC);
                int kr = k_start + i;
                if (kr < S) {
                    cp_async_16(&K_s[i*D + dd*VEC], &K[bh_off + (uint64_t)kr*D + dd*VEC]);
                    cp_async_16(&V_s[i*D + dd*VEC], &V[bh_off + (uint64_t)kr*D + dd*VEC]);
                } else {
                    int4 z = make_int4(0, 0, 0, 0);
                    *reinterpret_cast<int4*>(&K_s[i*D + dd*VEC]) = z;
                    *reinterpret_cast<int4*>(&V_s[i*D + dd*VEC]) = z;
                }
            }
        }
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();

        // S = Q @ K^T * scale
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
                float pv = __expf(S_s[idx] - L_s[i]);
                S_s[idx] = pv;
                P_s[idx] = __float2bfloat16(pv);
            }
        }
        __syncthreads();

        // dV += P^T @ dO (atomicAdd to global fp32, reuse V_s as tmp)
        compute_dV_atomic(P_s, dO_s, dV_fp32, k_start, bh_off, S,
                          reinterpret_cast<float*>(V_s), warp_id);
        __syncthreads();

        // dS = P * (dP - D) * scale, store bf16 in P_s (overwrite P)
        for (int idx = tid; idx < BM * BN; idx += THREADS) {
            int i = idx / BN;
            float pv = S_s[idx];
            float dpv = dP_s[idx];
            P_s[idx] = __float2bfloat16(pv * (dpv - D_s[i]) * SCALE);
        }
        __syncthreads();

        // dK += dS^T @ Q (atomicAdd to global fp32, reuse V_s as tmp)
        compute_dK_atomic(P_s, Q_s, dK_fp32, k_start, bh_off, S,
                          reinterpret_cast<float*>(V_s), warp_id);
        __syncthreads();

        // dQ += dS @ K (accumulate in register fragments)
        compute_dQ_acc(P_s, K_s, dQ_frag, warp_id);
    }

    // Store dQ from registers to global via shared memory
    // Reuse S_s + dP_s as contiguous BM*D fp32 buffer
    {
        constexpr int N_T = D / 16;
        float* dQ_tmp = S_s; // S_s and dP_s are contiguous, 32KB total
        for (int i = 0; i < 4; i++) {
            int tile = warp_id + i * WARPS;
            int mt = tile / N_T, nt = tile % N_T;
            wmma::store_matrix_sync(dQ_tmp + mt*16*D + nt*16, dQ_frag[i], D, wmma::mem_row_major);
        }
    }
    __syncthreads();

    // Write dQ to global bf16
    for (int idx = tid; idx < BM * D; idx += THREADS) {
        int i = idx / D, dd = idx % D;
        int qr = q_start + i;
        if (qr < S)
            dQ_out[bh_off + (uint64_t)qr * D + dd] = __float2bfloat16(S_s[idx]);
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

    float* dK_fp32 = nullptr;
    float* dV_fp32 = nullptr;
    CUDA_CHECK(cudaMallocAsync(&dK_fp32, fp32_bytes, stream));
    CUDA_CHECK(cudaMallocAsync(&dV_fp32, fp32_bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(dK_fp32, 0, fp32_bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(dV_fp32, 0, fp32_bytes, stream));

    int num_q_blocks = (S + BM - 1) / BM;
    dim3 grid(BH, num_q_blocks, 1);
    dim3 block(THREADS, 1, 1);

    CUDA_CHECK(cudaFuncSetAttribute(
        attn_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));

    attn_bwd_kernel<<<grid, block, SMEM_SIZE, stream>>>(
        Q_p, K_p, V_p, O_p, dO_p, L_p, dQ_p, dK_fp32, dV_fp32, S);

    int conv_threads = 256;
    int conv_blocks = (total_elems + conv_threads - 1) / conv_threads;
    convert_fp32_to_bf16_kernel<<<conv_blocks, conv_threads, 0, stream>>>(dK_fp32, dK_p, total_elems);
    convert_fp32_to_bf16_kernel<<<conv_blocks, conv_threads, 0, stream>>>(dV_fp32, dV_p, total_elems);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFreeAsync(dK_fp32, stream));
    CUDA_CHECK(cudaFreeAsync(dV_fp32, stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, attn_bwd::run);

} // namespace attn_bwd