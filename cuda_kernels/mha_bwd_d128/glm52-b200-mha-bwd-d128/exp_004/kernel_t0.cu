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

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

constexpr int D = 128;
constexpr int BM = 32;
constexpr int BN = 64;
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int WM = 16, WN = 16, WK = 16;

constexpr int SMEM_SIZE =
    BN * D * 2 +   // K_smem bf16
    BN * D * 2 +   // V_smem bf16
    BN * D * 4 +   // dK_smem float
    BN * D * 4 +   // dV_smem float
    BM * D * 2 +   // Q_smem bf16
    BM * D * 2 +   // dO_smem bf16
    BM * BN * 4 +  // S_temp float
    BM * BN * 2 +  // P_smem bf16
    BM * BN * 2;   // dS_smem bf16

__global__ void compute_D_kernel(const __nv_bfloat16* __restrict__ dO,
                                  const __nv_bfloat16* __restrict__ O,
                                  float* __restrict__ D_out,
                                  int total_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= total_rows) return;
    const __nv_bfloat162* dO_row = reinterpret_cast<const __nv_bfloat162*>(dO + (size_t)row * D);
    const __nv_bfloat162* O_row = reinterpret_cast<const __nv_bfloat162*>(O + (size_t)row * D);
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < D / 2; i++) {
        float2 do_f = __bfloat1622float2(dO_row[i]);
        float2 o_f = __bfloat1622float2(O_row[i]);
        sum += do_f.x * o_f.x + do_f.y * o_f.y;
    }
    D_out[row] = sum;
}

__global__ void zero_buffer_kernel(float* ptr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) ptr[idx] = 0.0f;
}

__global__ void convert_bf16_kernel(const float* __restrict__ src,
                                     __nv_bfloat16* __restrict__ dst,
                                     int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2bfloat16(src[idx]);
    }
}

__global__ void flash_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ D_vals,
    float* __restrict__ dQ_workspace,
    __nv_bfloat16* __restrict__ dK_out,
    __nv_bfloat16* __restrict__ dV_out,
    int B, int H, int S)
{
    int bh = blockIdx.x;
    int kv_tile = blockIdx.y;
    int bn_start = kv_tile * BN;

    int b = bh / H;
    int h = bh % H;
    size_t bh_off = (size_t)(b * H + h) * S * D;

    const __nv_bfloat16* Q_bh = Q + bh_off;
    const __nv_bfloat16* K_bh = K + bh_off;
    const __nv_bfloat16* V_bh = V + bh_off;
    const __nv_bfloat16* dO_bh = dO + bh_off;
    const float* L_bh = L + (size_t)(b * H + h) * S;
    const float* D_bh = D_vals + (size_t)(b * H + h) * S;
    float* dQ_bh = dQ_workspace + bh_off;
    __nv_bfloat16* dK_bh = dK_out + bh_off;
    __nv_bfloat16* dV_bh = dV_out + bh_off;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* K_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* V_smem = K_smem + BN * D;
    float* dK_smem = reinterpret_cast<float*>(V_smem + BN * D);
    float* dV_smem = dK_smem + BN * D;
    __nv_bfloat16* Q_smem = reinterpret_cast<__nv_bfloat16*>(dV_smem + BN * D);
    __nv_bfloat16* dO_smem = Q_smem + BM * D;
    float* S_temp = reinterpret_cast<float*>(dO_smem + BM * D);
    __nv_bfloat16* P_smem = reinterpret_cast<__nv_bfloat16*>(S_temp + BM * BN);
    __nv_bfloat16* dS_smem = P_smem + BM * BN;

    __shared__ float L_smem[BM];
    __shared__ float D_smem[BM];

    int tid = threadIdx.x;
    int warp_id = tid / 32;

    // Load K and V tiles (persistent)
    for (int i = tid; i < BN * D; i += THREADS) {
        int row = i / D, col = i % D;
        int gr = bn_start + row;
        K_smem[i] = (gr < S) ? K_bh[(size_t)gr * D + col] : __float2bfloat16(0.f);
        V_smem[i] = (gr < S) ? V_bh[(size_t)gr * D + col] : __float2bfloat16(0.f);
    }

    // Init dK, dV accumulators to 0
    for (int i = tid; i < BN * D; i += THREADS) {
        dK_smem[i] = 0.0f;
        dV_smem[i] = 0.0f;
    }

    __syncthreads();

    const float scale = 0.08838834764831845f; // 1/sqrt(128)
    int num_q_tiles = (S + BM - 1) / BM;

    for (int qt = 0; qt < num_q_tiles; qt++) {
        int bm_start = qt * BM;

        // Load Q and dO for this query tile
        for (int i = tid; i < BM * D; i += THREADS) {
            int row = i / D, col = i % D;
            int gr = bm_start + row;
            Q_smem[i] = (gr < S) ? Q_bh[(size_t)gr * D + col] : __float2bfloat16(0.f);
            dO_smem[i] = (gr < S) ? dO_bh[(size_t)gr * D + col] : __float2bfloat16(0.f);
        }

        // Load L and D for this query tile
        if (tid < BM) {
            int gr = bm_start + tid;
            L_smem[tid] = (gr < S) ? L_bh[gr] : 0.0f;
            D_smem[tid] = (gr < S) ? D_bh[gr] : 0.0f;
        }

        __syncthreads();

        // GEMM 1: S = Q @ K^T -> S_temp (float), [BM][BN]
        {
            constexpr int tm_n = BM / WM;     // 2
            constexpr int tn_n = BN / WN;     // 4
            constexpr int tpw = (tm_n * tn_n) / WARPS; // 1

            #pragma unroll
            for (int tw = 0; tw < tpw; tw++) {
                int tile_idx = warp_id * tpw + tw;
                int tm = tile_idx / tn_n;
                int tn = tile_idx % tn_n;

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
                wmma::store_matrix_sync(S_temp + tm * WM * BN + tn * WN, acc, BN, wmma::mem_row_major);
            }
        }

        __syncthreads();

        // Element-wise: P = exp(S * scale - L), store as bf16
        for (int i = tid; i < BM * BN; i += THREADS) {
            int row = i / BN;
            int col = i % BN;
            int gr = bm_start + row;
            int gc = bn_start + col;
            if (gr < S && gc < S) {
                float s = S_temp[i] * scale;
                P_smem[i] = __float2bfloat16(__expf(s - L_smem[row]));
            } else {
                P_smem[i] = __float2bfloat16(0.0f);
            }
        }

        __syncthreads();

        // GEMM 2: dP = dO @ V^T -> S_temp (float), [BM][BN]
        {
            constexpr int tm_n = BM / WM;
            constexpr int tn_n = BN / WN;
            constexpr int tpw = (tm_n * tn_n) / WARPS;

            #pragma unroll
            for (int tw = 0; tw < tpw; tw++) {
                int tile_idx = warp_id * tpw + tw;
                int tm = tile_idx / tn_n;
                int tn = tile_idx % tn_n;

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
                wmma::store_matrix_sync(S_temp + tm * WM * BN + tn * WN, acc, BN, wmma::mem_row_major);
            }
        }

        __syncthreads();

        // Element-wise: dS = P * (dP - D_val) * scale, store as bf16
        for (int i = tid; i < BM * BN; i += THREADS) {
            int row = i / BN;
            int col = i % BN;
            int gr = bm_start + row;
            int gc = bn_start + col;
            if (gr < S && gc < S) {
                float p = __bfloat162float(P_smem[i]);
                float dp = S_temp[i];
                dS_smem[i] = __float2bfloat16(p * (dp - D_smem[row]) * scale);
            } else {
                dS_smem[i] = __float2bfloat16(0.0f);
            }
        }

        __syncthreads();

        // GEMM 3: dK += dS^T @ Q -> dK_smem [BN][D]
        // A = dS^T (col_major from dS_smem[BM][BN]), B = Q (row_major)
        {
            constexpr int tm_n = BN / WM;      // 4 (BN tiles)
            constexpr int tn_n = D / WN;       // 8 (D tiles)
            constexpr int tpw = (tm_n * tn_n) / WARPS; // 4

            #pragma unroll
            for (int tw = 0; tw < tpw; tw++) {
                int tile_idx = warp_id * tpw + tw;
                int tm = tile_idx / tn_n;
                int tn = tile_idx % tn_n;

                wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc;
                wmma::load_matrix_sync(acc, dK_smem + tm * WM * D + tn * WN, D, wmma::mem_row_major);

                #pragma unroll
                for (int kk = 0; kk < BM / WK; kk++) { // 2
                    wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::col_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::row_major> b_frag;
                    wmma::load_matrix_sync(a_frag, dS_smem + kk * WK * BN + tm * WM, BN);
                    wmma::load_matrix_sync(b_frag, Q_smem + kk * WK * D + tn * WN, D);
                    wmma::mma_sync(acc, a_frag, b_frag, acc);
                }
                wmma::store_matrix_sync(dK_smem + tm * WM * D + tn * WN, acc, D, wmma::mem_row_major);
            }
        }

        __syncthreads();

        // GEMM 4: dV += P^T @ dO -> dV_smem [BN][D]
        {
            constexpr int tm_n = BN / WM;
            constexpr int tn_n = D / WN;
            constexpr int tpw = (tm_n * tn_n) / WARPS;

            #pragma unroll
            for (int tw = 0; tw < tpw; tw++) {
                int tile_idx = warp_id * tpw + tw;
                int tm = tile_idx / tn_n;
                int tn = tile_idx % tn_n;

                wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc;
                wmma::load_matrix_sync(acc, dV_smem + tm * WM * D + tn * WN, D, wmma::mem_row_major);

                #pragma unroll
                for (int kk = 0; kk < BM / WK; kk++) {
                    wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::col_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::row_major> b_frag;
                    wmma::load_matrix_sync(a_frag, P_smem + kk * WK * BN + tm * WM, BN);
                    wmma::load_matrix_sync(b_frag, dO_smem + kk * WK * D + tn * WN, D);
                    wmma::mma_sync(acc, a_frag, b_frag, acc);
                }
                wmma::store_matrix_sync(dV_smem + tm * WM * D + tn * WN, acc, D, wmma::mem_row_major);
            }
        }

        __syncthreads();

        // GEMM 5: dQ += dS @ K -> atomic add to dQ_workspace [BM][D]
        // Split D into 2 chunks, reuse S_temp [BM][D/2]
        for (int dc = 0; dc < 2; dc++) {
            int d_offset = dc * (D / 2);

            {
                constexpr int tm_n = BM / WM;          // 2
                constexpr int tn_n = (D / 2) / WN;     // 4
                constexpr int tpw = (tm_n * tn_n) / WARPS; // 1

                #pragma unroll
                for (int tw = 0; tw < tpw; tw++) {
                    int tile_idx = warp_id * tpw + tw;
                    int tm = tile_idx / tn_n;
                    int tn = tile_idx % tn_n;

                    wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc;
                    wmma::fill_fragment(acc, 0.0f);

                    #pragma unroll
                    for (int kk = 0; kk < BN / WK; kk++) { // 4
                        wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::row_major> a_frag;
                        wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::row_major> b_frag;
                        wmma::load_matrix_sync(a_frag, dS_smem + tm * WM * BN + kk * WK, BN);
                        wmma::load_matrix_sync(b_frag, K_smem + kk * WK * D + d_offset + tn * WN, D);
                        wmma::mma_sync(acc, a_frag, b_frag, acc);
                    }
                    wmma::store_matrix_sync(S_temp + tm * WM * (D / 2) + tn * WN, acc, D / 2, wmma::mem_row_major);
                }
            }

            __syncthreads();

            // Atomic add to dQ workspace
            for (int i = tid; i < BM * (D / 2); i += THREADS) {
                int row = i / (D / 2);
                int col = i % (D / 2);
                int gr = bm_start + row;
                if (gr < S) {
                    atomicAdd(&dQ_bh[(size_t)gr * D + d_offset + col], S_temp[i]);
                }
            }

            __syncthreads();
        }
    }

    // Store dK and dV to global (convert float -> bf16)
    for (int i = tid; i < BN * D; i += THREADS) {
        int row = i / D, col = i % D;
        int gr = bn_start + row;
        if (gr < S) {
            dK_bh[(size_t)gr * D + col] = __float2bfloat16(dK_smem[i]);
            dV_bh[(size_t)gr * D + col] = __float2bfloat16(dV_smem[i]);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {

    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int S = (int)Q.size(2);

    int total_rows = B * H * S;
    int total_elements = B * H * S * D;

    float* D_temp;
    CUDA_CHECK(cudaMalloc(&D_temp, (size_t)total_rows * sizeof(float)));

    float* dQ_workspace;
    CUDA_CHECK(cudaMalloc(&dQ_workspace, (size_t)total_elements * sizeof(float)));

    // Compute D = rowsum(dO * O)
    {
        int threads = 256;
        int blocks = (total_rows + threads - 1) / threads;
        compute_D_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(dO.data_ptr()),
            static_cast<const __nv_bfloat16*>(O.data_ptr()),
            D_temp, total_rows);
    }

    // Zero dQ workspace
    {
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;
        zero_buffer_kernel<<<blocks, threads, 0, stream>>>(dQ_workspace, total_elements);
    }

    // Launch flash backward kernel
    {
        dim3 grid(B * H, (S + BN - 1) / BN, 1);
        dim3 block(THREADS, 1, 1);

        CUDA_CHECK(cudaFuncSetAttribute(flash_bwd_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));

        flash_bwd_kernel<<<grid, block, SMEM_SIZE, stream>>>(
            static_cast<const __nv_bfloat16*>(Q.data_ptr()),
            static_cast<const __nv_bfloat16*>(K.data_ptr()),
            static_cast<const __nv_bfloat16*>(V.data_ptr()),
            static_cast<const __nv_bfloat16*>(dO.data_ptr()),
            static_cast<const float*>(L.data_ptr()),
            D_temp,
            dQ_workspace,
            static_cast<__nv_bfloat16*>(dK.data_ptr()),
            static_cast<__nv_bfloat16*>(dV.data_ptr()),
            B, H, S);
    }

    CUDA_CHECK(cudaGetLastError());

    // Convert dQ workspace to bf16 output
    {
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;
        convert_bf16_kernel<<<blocks, threads, 0, stream>>>(
            dQ_workspace,
            static_cast<__nv_bfloat16*>(dQ.data_ptr()),
            total_elements);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFree(D_temp));
    CUDA_CHECK(cudaFree(dQ_workspace));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flash_attn_bwd::run);

}  // namespace flash_attn_bwd