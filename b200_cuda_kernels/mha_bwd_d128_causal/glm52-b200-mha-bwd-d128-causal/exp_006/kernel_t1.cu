#include <cuda_bf16.h>
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
        fprintf(stderr, "CUDA error %s at %s:%d\n", \
                cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

namespace attn_bwd {

using namespace nvcuda;

constexpr int D = 128;
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
constexpr int M_TILES = BM / WMMA_M;
constexpr int N_TILES = BN / WMMA_N;
constexpr int K_TILES = D / WMMA_K;
constexpr int WARPS = 4;
constexpr int THREADS = WARPS * 32;
constexpr float SCALE = 0.088388348f; // 1/sqrt(128)

struct Smem {
    __nv_bfloat16 sQ[BM][D];       // 16KB
    __nv_bfloat16 sK[BN][D];       // 16KB
    __nv_bfloat16 sV[BN][D];       // 16KB
    __nv_bfloat16 sdO[BM][D];      // 16KB
    float sS[BM][BN];              // 16KB
    float sP[BM][BN];              // 16KB
    float sdP[BM][BN];             // 16KB
    float sdS[BM][BN];             // 16KB
    float sL[BM];                  // 256B
    float sD_val[BM];              // 256B
    float sdV_acc[BN][D];          // 32KB
    float sdK_acc[BN][D];          // 32KB
};

__global__ void compute_D_kernel(
    const __nv_bfloat16* __restrict__ dO,
    const __nv_bfloat16* __restrict__ O,
    float* __restrict__ D_out,
    int total_rows)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= total_rows) return;
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < D; i++) {
        sum += __bfloat162float(dO[row * D + i]) * __bfloat162float(O[row * D + i]);
    }
    D_out[row] = sum;
}

__global__ void zero_kernel(float* ptr, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) ptr[idx] = 0.0f;
}

__global__ void convert_f2bf_kernel(
    const float* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) dst[idx] = __float2bfloat16(src[idx]);
}

__global__ void attn_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ D_buf,
    float* __restrict__ dQ_fp32,
    __nv_bfloat16* __restrict__ dK_out,
    __nv_bfloat16* __restrict__ dV_out,
    int S_total)
{
    int bh = blockIdx.x;
    int j_block = blockIdx.y;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int j_base = j_block * BN;

    const uint64_t bh_off = (uint64_t)bh * S_total;
    const __nv_bfloat16* Q_ptr = Q + bh_off * D;
    const __nv_bfloat16* K_ptr = K + bh_off * D;
    const __nv_bfloat16* V_ptr = V + bh_off * D;
    const __nv_bfloat16* dO_ptr = dO + bh_off * D;
    const float* L_ptr = L + bh_off;
    const float* D_ptr = D_buf + bh_off;
    float* dQ_ptr = dQ_fp32 + bh_off * D;
    __nv_bfloat16* dK_ptr = dK_out + bh_off * D;
    __nv_bfloat16* dV_ptr = dV_out + bh_off * D;

    extern __shared__ char __smem[];
    Smem& smem = *reinterpret_cast<Smem*>(__smem);

    // Load K, V for this KV block
    {
        const uint4* K_src = reinterpret_cast<const uint4*>(K_ptr + (uint64_t)j_base * D);
        const uint4* V_src = reinterpret_cast<const uint4*>(V_ptr + (uint64_t)j_base * D);
        uint4* sK_dst = reinterpret_cast<uint4*>(smem.sK);
        uint4* sV_dst = reinterpret_cast<uint4*>(smem.sV);
        int total_vec = BN * (D / 8);
        for (int idx = tid; idx < total_vec; idx += THREADS) {
            int n = idx / (D / 8);
            if (j_base + n < S_total) {
                sK_dst[idx] = K_src[idx];
                sV_dst[idx] = V_src[idx];
            } else {
                sK_dst[idx] = make_uint4(0, 0, 0, 0);
                sV_dst[idx] = make_uint4(0, 0, 0, 0);
            }
        }
    }

    // Zero accumulators
    for (int idx = tid; idx < BN * D; idx += THREADS) {
        smem.sdV_acc[idx / D][idx % D] = 0.0f;
        smem.sdK_acc[idx / D][idx % D] = 0.0f;
    }
    __syncthreads();

    int i_min = j_base / BM;
    int i_max = (S_total - 1) / BM;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_row;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_col;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[N_TILES];

    for (int i_block = i_min; i_block <= i_max; i_block++) {
        int i_base = i_block * BM;

        // Load Q, dO
        {
            const uint4* Q_src = reinterpret_cast<const uint4*>(Q_ptr + (uint64_t)i_base * D);
            const uint4* dO_src = reinterpret_cast<const uint4*>(dO_ptr + (uint64_t)i_base * D);
            uint4* sQ_dst = reinterpret_cast<uint4*>(smem.sQ);
            uint4* sdO_dst = reinterpret_cast<uint4*>(smem.sdO);
            int total_vec = BM * (D / 8);
            for (int idx = tid; idx < total_vec; idx += THREADS) {
                int m = idx / (D / 8);
                if (i_base + m < S_total) {
                    sQ_dst[idx] = Q_src[idx];
                    sdO_dst[idx] = dO_src[idx];
                } else {
                    sQ_dst[idx] = make_uint4(0, 0, 0, 0);
                    sdO_dst[idx] = make_uint4(0, 0, 0, 0);
                }
            }
        }

        // Load L, D
        if (tid < BM) {
            int gm = i_base + tid;
            smem.sL[tid] = (gm < S_total) ? L_ptr[gm] : 0.0f;
            smem.sD_val[tid] = (gm < S_total) ? D_ptr[gm] : 0.0f;
        }
        __syncthreads();

        // S = Q @ K^T * scale (using wmma, bf16 inputs, fp32 accum)
        for (int n = 0; n < N_TILES; n++) wmma::fill_fragment(acc[n], 0.0f);
        for (int k = 0; k < K_TILES; k++) {
            wmma::load_matrix_sync(a_row, &smem.sQ[warp_id * 16][k * 16], D);
            for (int n = 0; n < N_TILES; n++) {
                wmma::load_matrix_sync(b_col, &smem.sK[n * 16][k * 16], D);
                wmma::mma_sync(acc[n], a_row, b_col, acc[n]);
            }
        }
        for (int n = 0; n < N_TILES; n++) {
            for (int i = 0; i < acc[n].num_elements; i++) acc[n].x[i] *= SCALE;
            wmma::store_matrix_sync(&smem.sS[warp_id * 16][n * 16], acc[n], BN, wmma::mem_row_major);
        }
        __syncthreads();

        // dP = dO @ V^T (using wmma, bf16 inputs, fp32 accum)
        for (int n = 0; n < N_TILES; n++) wmma::fill_fragment(acc[n], 0.0f);
        for (int k = 0; k < K_TILES; k++) {
            wmma::load_matrix_sync(a_row, &smem.sdO[warp_id * 16][k * 16], D);
            for (int n = 0; n < N_TILES; n++) {
                wmma::load_matrix_sync(b_col, &smem.sV[n * 16][k * 16], D);
                wmma::mma_sync(acc[n], a_row, b_col, acc[n]);
            }
        }
        for (int n = 0; n < N_TILES; n++) {
            wmma::store_matrix_sync(&smem.sdP[warp_id * 16][n * 16], acc[n], BN, wmma::mem_row_major);
        }
        __syncthreads();

        // Element-wise: P = exp(S-L), dS = P*(dP-D)*scale, with causal mask
        for (int idx = tid; idx < BM * BN; idx += THREADS) {
            int m = idx / BN, n = idx % BN;
            int gm = i_base + m, gn = j_base + n;
            float s_val = smem.sS[m][n];
            float dp_val = smem.sdP[m][n];
            if (gm >= S_total || gn > gm || gn >= S_total) {
                smem.sP[m][n] = 0.0f;
                smem.sdS[m][n] = 0.0f;
            } else {
                float lse = smem.sL[m];
                float p = expf(s_val - lse);
                smem.sP[m][n] = p;
                float d_val = smem.sD_val[m];
                float ds = p * (dp_val - d_val) * SCALE;
                smem.sdS[m][n] = ds;
            }
        }
        __syncthreads();

        // dV += P^T @ dO (fp32 CUDA cores)
        // P is (BM, BN), dO is (BM, D), dV is (BN, D)
        for (int idx = tid; idx < BN * D; idx += THREADS) {
            int n = idx / D, d = idx % D;
            float sum = smem.sdV_acc[n][d];
            for (int m = 0; m < BM; m++) {
                sum += smem.sP[m][n] * __bfloat162float(smem.sdO[m][d]);
            }
            smem.sdV_acc[n][d] = sum;
        }

        // dK += dS^T @ Q (fp32 CUDA cores)
        // dS is (BM, BN), Q is (BM, D), dK is (BN, D)
        for (int idx = tid; idx < BN * D; idx += THREADS) {
            int n = idx / D, d = idx % D;
            float sum = smem.sdK_acc[n][d];
            for (int m = 0; m < BM; m++) {
                sum += smem.sdS[m][n] * __bfloat162float(smem.sQ[m][d]);
            }
            smem.sdK_acc[n][d] = sum;
        }

        // dQ += dS @ K (fp32, atomicAdd to global)
        // dS is (BM, BN), K is (BN, D), dQ is (BM, D)
        for (int idx = tid; idx < BM * D; idx += THREADS) {
            int m = idx / D, d = idx % D;
            int gm = i_base + m;
            if (gm >= S_total) continue;
            float sum = 0.0f;
            for (int n = 0; n < BN; n++) {
                int gn = j_base + n;
                if (gn > gm || gn >= S_total) continue;
                sum += smem.sdS[m][n] * __bfloat162float(smem.sK[n][d]);
            }
            atomicAdd(&dQ_ptr[(uint64_t)gm * D + d], sum);
        }
        __syncthreads();
    }

    // Store dK, dV to global as bf16
    for (int idx = tid; idx < BN * D; idx += THREADS) {
        int n = idx / D, dd = idx % D;
        int gn = j_base + n;
        if (gn < S_total) {
            dK_ptr[(uint64_t)gn * D + dd] = __float2bfloat16(smem.sdK_acc[n][dd]);
            dV_ptr[(uint64_t)gn * D + dd] = __float2bfloat16(smem.sdV_acc[n][dd]);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {

    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    constexpr int B = 4, H = 48, D_val = 128, BH = B * H;
    int S = (int)Q.size(2);
    int total_rows = BH * S;
    int total_elems = total_rows * D_val;

    float *dQ_fp32, *D_buf;
    CUDA_CHECK(cudaMalloc(&dQ_fp32, (uint64_t)total_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&D_buf, (uint64_t)total_rows * sizeof(float)));

    {
        int th = 256, bl = (total_rows + th - 1) / th;
        compute_D_kernel<<<bl, th, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(dO.data_ptr()),
            static_cast<const __nv_bfloat16*>(O.data_ptr()),
            D_buf, total_rows);
    }
    {
        int th = 256, bl = (total_elems + th - 1) / th;
        zero_kernel<<<bl, th, 0, stream>>>(dQ_fp32, total_elems);
    }
    {
        size_t smem_size = sizeof(Smem);
        CUDA_CHECK(cudaFuncSetAttribute(attn_bwd_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        dim3 grid(BH, (S + BN - 1) / BN);
        dim3 block(THREADS);
        attn_bwd_kernel<<<grid, block, smem_size, stream>>>(
            static_cast<const __nv_bfloat16*>(Q.data_ptr()),
            static_cast<const __nv_bfloat16*>(K.data_ptr()),
            static_cast<const __nv_bfloat16*>(V.data_ptr()),
            static_cast<const __nv_bfloat16*>(dO.data_ptr()),
            static_cast<const float*>(L.data_ptr()),
            D_buf, dQ_fp32,
            static_cast<__nv_bfloat16*>(dK.data_ptr()),
            static_cast<__nv_bfloat16*>(dV.data_ptr()), S);
    }
    {
        int th = 256, bl = (total_elems + th - 1) / th;
        convert_f2bf_kernel<<<bl, th, 0, stream>>>(
            dQ_fp32, static_cast<__nv_bfloat16*>(dQ.data_ptr()), total_elems);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(dQ_fp32));
    CUDA_CHECK(cudaFree(D_buf));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, attn_bwd::run);

}  // namespace attn_bwd