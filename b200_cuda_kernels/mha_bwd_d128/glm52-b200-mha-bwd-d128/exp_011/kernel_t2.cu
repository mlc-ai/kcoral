#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda.h>
#include <cstdint>
#include <cmath>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

using namespace nvcuda;

namespace attn_bwd {

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int D = 128;
constexpr int WARPS = 8;
constexpr int THREADS = 256;
constexpr float SCALE = 0.08838834764831843f; // 1/sqrt(128)

constexpr int Q_OFF    = 0;
constexpr int dO_OFF   = Q_OFF  + BM * D * 2;
constexpr int K_OFF    = dO_OFF + BM * D * 2;
constexpr int V_OFF    = K_OFF  + BN * D * 2;
constexpr int dQ_OFF   = V_OFF  + BN * D * 2;
constexpr int S_OFF    = dQ_OFF + BM * D * 4;
constexpr int P_OFF    = S_OFF  + BM * BN * 4;
constexpr int dP_OFF   = P_OFF  + BM * BN * 2;
constexpr int DVEC_OFF = dP_OFF + BM * BN * 4;
constexpr int L_OFF    = DVEC_OFF + BM * 4;
constexpr int TMP_OFF  = L_OFF  + BM * 4;
constexpr int TOTAL_SMEM = TMP_OFF + WARPS * 16 * 16 * 4;

template<int M, int N, int K>
__device__ __forceinline__ void wmma_gemm_rc(
    const __nv_bfloat16* __restrict__ A_smem,
    const __nv_bfloat16* __restrict__ B_smem,
    float* __restrict__ C_smem,
    float scale_factor,
    int warp_id)
{
    constexpr int M_T = M / 16;
    constexpr int N_T = N / 16;
    constexpr int K_T = K / 16;
    constexpr int TOT = M_T * N_T;

    for (int t = 0; t < (TOT + WARPS - 1) / WARPS; t++) {
        int ti = warp_id + t * WARPS;
        if (ti >= TOT) break;
        int mt = ti / N_T;
        int nt = ti % N_T;

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        for (int kt = 0; kt < K_T; kt++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, A_smem + mt * 16 * K + kt * 16, K);

            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
            wmma::load_matrix_sync(b_frag, B_smem + nt * 16 * K + kt * 16, K);

            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        if (scale_factor != 1.0f) {
            for (int i = 0; i < c_frag.num_elements; i++) c_frag.x[i] *= scale_factor;
        }
        wmma::store_matrix_sync(C_smem + mt * 16 * N + nt * 16, c_frag, N, wmma::mem_row_major);
    }
}

__device__ __forceinline__ void compute_dV(
    const __nv_bfloat16* __restrict__ P_smem,
    const __nv_bfloat16* __restrict__ dO_smem,
    float* __restrict__ dV_ptr,
    int k_start, uint64_t bh, int S,
    float* __restrict__ tmp_smem)
{
    constexpr int M = BN, N = D, K = BM;
    constexpr int M_T = M/16, N_T = N/16, K_T = K/16;
    constexpr int TOT = M_T * N_T;

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    for (int t = 0; t < (TOT + WARPS - 1) / WARPS; t++) {
        int ti = warp_id + t * WARPS;
        if (ti >= TOT) break;
        int mt = ti / N_T;
        int nt = ti % N_T;

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        for (int kt = 0; kt < K_T; kt++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a_frag;
            wmma::load_matrix_sync(a_frag, P_smem + kt * 16 * BN + mt * 16, BN);

            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_frag;
            wmma::load_matrix_sync(b_frag, dO_smem + kt * 16 * D + nt * 16, D);

            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        float* tile = tmp_smem + warp_id * 256;
        wmma::store_matrix_sync(tile, c_frag, 16, wmma::mem_row_major);
        __syncwarp();

        int row_base = k_start + mt * 16;
        int col_base = nt * 16;
        for (int idx = lane_id; idx < 256; idx += 32) {
            int lr = idx / 16, lc = idx % 16;
            int gr = row_base + lr, gc = col_base + lc;
            if (gr < S) {
                atomicAdd(&dV_ptr[bh * S * D + gr * D + gc], tile[lr * 16 + lc]);
            }
        }
        __syncwarp();
    }
}

__device__ __forceinline__ void compute_dK(
    const __nv_bfloat16* __restrict__ dS_smem,
    const __nv_bfloat16* __restrict__ Q_smem,
    float* __restrict__ dK_ptr,
    int k_start, uint64_t bh, int S,
    float* __restrict__ tmp_smem)
{
    constexpr int M = BN, N = D, K = BM;
    constexpr int M_T = M/16, N_T = N/16, K_T = K/16;
    constexpr int TOT = M_T * N_T;

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    for (int t = 0; t < (TOT + WARPS - 1) / WARPS; t++) {
        int ti = warp_id + t * WARPS;
        if (ti >= TOT) break;
        int mt = ti / N_T;
        int nt = ti % N_T;

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        for (int kt = 0; kt < K_T; kt++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a_frag;
            wmma::load_matrix_sync(a_frag, dS_smem + kt * 16 * BN + mt * 16, BN);

            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_frag;
            wmma::load_matrix_sync(b_frag, Q_smem + kt * 16 * D + nt * 16, D);

            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        float* tile = tmp_smem + warp_id * 256;
        wmma::store_matrix_sync(tile, c_frag, 16, wmma::mem_row_major);
        __syncwarp();

        int row_base = k_start + mt * 16;
        int col_base = nt * 16;
        for (int idx = lane_id; idx < 256; idx += 32) {
            int lr = idx / 16, lc = idx % 16;
            int gr = row_base + lr, gc = col_base + lc;
            if (gr < S) {
                atomicAdd(&dK_ptr[bh * S * D + gr * D + gc], tile[lr * 16 + lc]);
            }
        }
        __syncwarp();
    }
}

__device__ __forceinline__ void compute_dQ(
    const __nv_bfloat16* __restrict__ dS_smem,
    const __nv_bfloat16* __restrict__ K_smem,
    float* __restrict__ dQ_acc,
    int warp_id)
{
    constexpr int M = BM, N = D, K = BN;
    constexpr int M_T = M/16, N_T = N/16, K_T = K/16;
    constexpr int TOT = M_T * N_T;

    for (int t = 0; t < (TOT + WARPS - 1) / WARPS; t++) {
        int ti = warp_id + t * WARPS;
        if (ti >= TOT) break;
        int mt = ti / N_T;
        int nt = ti % N_T;

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::load_matrix_sync(c_frag, dQ_acc + mt * 16 * D + nt * 16, D, wmma::mem_row_major);

        for (int kt = 0; kt < K_T; kt++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, dS_smem + mt * 16 * BN + kt * 16, BN);

            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_frag;
            wmma::load_matrix_sync(b_frag, K_smem + kt * 16 * D + nt * 16, D);

            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        wmma::store_matrix_sync(dQ_acc + mt * 16 * D + nt * 16, c_frag, D, wmma::mem_row_major);
    }
}

__global__ void attn_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    float* __restrict__ dK_fp32,
    float* __restrict__ dV_fp32,
    int S)
{
    int bh = blockIdx.x;
    int q_blk = blockIdx.y;
    int q_start = q_blk * BM;
    int tid = threadIdx.x;
    int warp_id = tid / 32;

    if (q_start >= S) return;

    extern __shared__ char smem[];
    __nv_bfloat16* Q_s  = reinterpret_cast<__nv_bfloat16*>(smem + Q_OFF);
    __nv_bfloat16* dO_s = reinterpret_cast<__nv_bfloat16*>(smem + dO_OFF);
    __nv_bfloat16* K_s  = reinterpret_cast<__nv_bfloat16*>(smem + K_OFF);
    __nv_bfloat16* V_s  = reinterpret_cast<__nv_bfloat16*>(smem + V_OFF);
    float* dQ_a         = reinterpret_cast<float*>(smem + dQ_OFF);
    float* S_s          = reinterpret_cast<float*>(smem + S_OFF);
    __nv_bfloat16* P_s  = reinterpret_cast<__nv_bfloat16*>(smem + P_OFF);
    float* dP_s         = reinterpret_cast<float*>(smem + dP_OFF);
    float* D_s          = reinterpret_cast<float*>(smem + DVEC_OFF);
    float* L_s          = reinterpret_cast<float*>(smem + L_OFF);
    float* tmp_s        = reinterpret_cast<float*>(smem + TMP_OFF);

    uint64_t bh_off = (uint64_t)bh * S * D;

    for (int idx = tid; idx < BM * D; idx += THREADS) {
        int i = idx / D, dd = idx % D;
        int qr = q_start + i;
        if (qr < S) {
            Q_s[idx]  = Q[bh_off + qr * D + dd];
            dO_s[idx] = dO[bh_off + qr * D + dd];
            K_s[idx]  = O[bh_off + qr * D + dd];
        } else {
            Q_s[idx]  = __float2bfloat16(0.0f);
            dO_s[idx] = __float2bfloat16(0.0f);
            K_s[idx]  = __float2bfloat16(0.0f);
        }
    }
    for (int i = tid; i < BM; i += THREADS) {
        L_s[i] = (q_start + i < S) ? L[bh * S + q_start + i] : 0.0f;
    }
    __syncthreads();

    for (int i = tid; i < BM; i += THREADS) {
        float sum = 0.0f;
        for (int dd = 0; dd < D; dd++)
            sum += __bfloat162float(dO_s[i * D + dd]) * __bfloat162float(K_s[i * D + dd]);
        D_s[i] = sum;
    }
    for (int idx = tid; idx < BM * D; idx += THREADS)
        dQ_a[idx] = 0.0f;
    __syncthreads();

    int num_k = (S + BN - 1) / BN;
    for (int kj = 0; kj < num_k; kj++) {
        int k_start = kj * BN;

        for (int idx = tid; idx < BN * D; idx += THREADS) {
            int i = idx / D, dd = idx % D;
            int kr = k_start + i;
            if (kr < S) {
                K_s[idx] = K[bh_off + kr * D + dd];
                V_s[idx] = V[bh_off + kr * D + dd];
            } else {
                K_s[idx] = __float2bfloat16(0.0f);
                V_s[idx] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        wmma_gemm_rc<BM, BN, D>(Q_s, K_s, S_s, SCALE, warp_id);
        __syncthreads();

        wmma_gemm_rc<BM, BN, D>(dO_s, V_s, dP_s, 1.0f, warp_id);
        __syncthreads();

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

        compute_dV(P_s, dO_s, dV_fp32, k_start, (uint64_t)bh, S, tmp_s);
        __syncthreads();

        for (int idx = tid; idx < BM * BN; idx += THREADS) {
            int i = idx / BN;
            float pv = S_s[idx];
            float dpv = dP_s[idx];
            // Scale dS by SCALE here so dK and dQ are automatically scaled
            P_s[idx] = __float2bfloat16(pv * (dpv - D_s[i]) * SCALE);
        }
        __syncthreads();

        compute_dK(P_s, Q_s, dK_fp32, k_start, (uint64_t)bh, S, tmp_s);
        __syncthreads();

        compute_dQ(P_s, K_s, dQ_a, warp_id);
        __syncthreads();
    }

    for (int idx = tid; idx < BM * D; idx += THREADS) {
        int i = idx / D, dd = idx % D;
        int qr = q_start + i;
        if (qr < S) {
            dQ[bh_off + qr * D + dd] = __float2bfloat16(dQ_a[idx]);
        }
    }
}

__global__ void convert_fp32_to_bf16_kernel(
    const float* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2bfloat16(src[idx]);
    }
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
        attn_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, TOTAL_SMEM));

    attn_bwd_kernel<<<grid, block, TOTAL_SMEM, stream>>>(
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