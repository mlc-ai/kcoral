#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda;

namespace tvm_ffi_mha_bwd {

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int D  = 128;
constexpr int WM = 16;
constexpr int WN = 16;
constexpr int WK = 16;
constexpr int WARPS = 4;
constexpr int THREADS = WARPS * 32;
constexpr int NK_D = D / WK;   // 8
constexpr int NK_BN = BN / WK; // 4
constexpr int NN_BN = BN / WN; // 4
constexpr int NN_D  = D / WN;  // 8

// Shared memory offsets (bytes)
constexpr int OFF_K    = 0;
constexpr int OFF_V    = OFF_K  + BN*D*2;       // 16384 (16 KB)
constexpr int OFF_Q    = OFF_V  + BN*D*2;       // 32768 (32 KB)
constexpr int OFF_dO   = OFF_Q  + BM*D*2;       // 49152 (48 KB) - Reused for P and dS
constexpr int OFF_O    = OFF_dO + BM*D*2;       // 65536 (64 KB) - Reused for S
constexpr int OFF_DQf  = OFF_O  + BM*D*2;       // 81920 (80 KB) - 32 KB FP32 scratchpad
constexpr int OFF_L    = OFF_DQf+ BM*D*4;       // 114688 (112 KB)
constexpr int OFF_Drow = OFF_L  + BM*4;         // 114944 (112.25 KB)
constexpr int SMEM_BYTES = OFF_Drow + BM*4;     // 115200 (112.5 KB)

__device__ __forceinline__ void load_bf16_tile(
    const __nv_bfloat16* gptr, int gldm,
    __nv_bfloat16* sptr, int sldm,
    int rows, int cols, int row_off, int S_max)
{
    const int nvec = rows * (cols / 8);
    int tid = threadIdx.x;
    for (int v = tid; v < nvec; v += blockDim.x) {
        int r = v / (cols / 8);
        int c8 = (v % (cols / 8)) * 8;
        int gr = row_off + r;
        uint4 data;
        if (gr < S_max && c8 < gldm) {
            data = *reinterpret_cast<const uint4*>(&gptr[gr * gldm + c8]);
        } else {
            data = make_uint4(0, 0, 0, 0);
        }
        *reinterpret_cast<uint4*>(&sptr[r * sldm + c8]) = data;
    }
}

__device__ __forceinline__ void load_lse(const float* gptr, float* sptr, int rows, int row_off, int S_max) {
    int tid = threadIdx.x;
    for (int r = tid; r < rows; r += blockDim.x) {
        sptr[r] = (row_off + r < S_max) ? gptr[row_off + r] : 0.f;
    }
}

__global__ void convert_f32_to_bf16(const float* in, __nv_bfloat16* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = __float2bfloat16(in[idx]);
}

__global__ __launch_bounds__(THREADS, 2) void mha_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float*          __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    float*                __restrict__ dK_fp32,
    float*                __restrict__ dV_fp32,
    int S, float attn_scale)
{
    extern __shared__ char smem_buf[];
    __nv_bfloat16* sK  = reinterpret_cast<__nv_bfloat16*>(smem_buf + OFF_K);
    __nv_bfloat16* sV  = reinterpret_cast<__nv_bfloat16*>(smem_buf + OFF_V);
    __nv_bfloat16* sQ  = reinterpret_cast<__nv_bfloat16*>(smem_buf + OFF_Q);
    __nv_bfloat16* sdO = reinterpret_cast<__nv_bfloat16*>(smem_buf + OFF_dO);
    __nv_bfloat16* sO  = reinterpret_cast<__nv_bfloat16*>(smem_buf + OFF_O);
    float*         sS  = reinterpret_cast<float*>(smem_buf + OFF_O); // Reuse sO
    __nv_bfloat16* sP  = reinterpret_cast<__nv_bfloat16*>(smem_buf + OFF_dO); // Reuse sdO
    __nv_bfloat16* sdS = reinterpret_cast<__nv_bfloat16*>(smem_buf + OFF_dO + BM*BN*2); // Reuse sdO 2nd half
    float*         sDQf= reinterpret_cast<float*>(smem_buf + OFF_DQf);
    float*         sL  = reinterpret_cast<float*>(smem_buf + OFF_L);
    float*         sDrow = reinterpret_cast<float*>(smem_buf + OFF_Drow);

    int b = blockIdx.x;
    int h = blockIdx.y;
    int i = blockIdx.z; // Q-block index
    int q_base = i * BM;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int64_t head_off = (int64_t)(b * gridDim.y + h) * S * D;
    const __nv_bfloat16* Q_h  = Q  + head_off;
    const __nv_bfloat16* K_h  = K  + head_off;
    const __nv_bfloat16* V_h  = V  + head_off;
    const __nv_bfloat16* O_h  = O  + head_off;
    const __nv_bfloat16* dO_h = dO + head_off;
    const float*         L_h  = L  + (int64_t)(b * gridDim.y + h) * S;
    __nv_bfloat16* dQ_h = dQ + head_off;
    float* dK_h = dK_fp32 + head_off;
    float* dV_h = dV_fp32 + head_off;

    // Load Q, dO, O tiles and LSE for this Q-block
    load_bf16_tile(Q_h,  D, sQ,  D, BM, D, q_base, S);
    load_bf16_tile(dO_h, D, sdO, D, BM, D, q_base, S);
    load_bf16_tile(O_h,  D, sO,  D, BM, D, q_base, S);
    load_lse(L_h, sL, BM, q_base, S);
    __syncthreads();

    // Compute D_row = rowsum(dO * O) using warp reduction
    for (int r = warp_id; r < BM; r += WARPS) {
        float sum = 0.0f;
        for (int c = lane_id; c < D; c += 32) {
            sum += __bfloat162float(sdO[r * D + c]) * __bfloat162float(sO[r * D + c]);
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
        }
        if (lane_id == 0) sDrow[r] = sum;
    }
    __syncthreads();

    // Accumulator for dQ (persistent across KV blocks)
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> dQ_frag[NN_D];
    #pragma unroll
    for (int n = 0; n < NN_D; ++n) {
        wmma::fill_fragment(dQ_frag[n], 0.0f);
    }

    // Reusable accumulator
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc[NN_D];

    int num_kv_blocks = (S + BN - 1) / BN;

    for (int j = 0; j < num_kv_blocks; ++j) {
        int kv_base = j * BN;

        // Load K, V tiles
        load_bf16_tile(K_h, D, sK, D, BN, D, kv_base, S);
        load_bf16_tile(V_h, D, sV, D, BN, D, kv_base, S);
        __syncthreads();

        // S = Q @ K^T  (raw, no scale)  -> sS (fp32)
        #pragma unroll
        for (int n = 0; n < NN_BN; ++n) {
            wmma::fill_fragment(acc[n], 0.0f);
            #pragma unroll
            for (int k = 0; k < NK_D; ++k) {
                wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::col_major> b;
                wmma::load_matrix_sync(a, &sQ[warp_id * WM * D + k * WK], D);
                wmma::load_matrix_sync(b, &sK[n * WN * D + k * WK], D);
                wmma::mma_sync(acc[n], a, b, acc[n]);
            }
            wmma::store_matrix_sync(&sS[warp_id * WM * BN + n * WN], acc[n], BN, wmma::mem_row_major);
        }
        __syncthreads();

        // P = exp(S*scale - L), with kv masking
        for (int idx = tid; idx < BM * BN; idx += THREADS) {
            int r = idx / BN;
            int c = idx % BN;
            int kv_row = kv_base + c;
            float s = sS[idx] * attn_scale - sL[r];
            float pv = (kv_row < S) ? __expf(s) : 0.0f;
            sP[idx] = __float2bfloat16(pv);
        }
        __syncthreads();

        // dP = dO @ V^T -> sS (reuse)
        #pragma unroll
        for (int n = 0; n < NN_BN; ++n) {
            wmma::fill_fragment(acc[n], 0.0f);
            #pragma unroll
            for (int k = 0; k < NK_D; ++k) {
                wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::col_major> b;
                wmma::load_matrix_sync(a, &sdO[warp_id * WM * D + k * WK], D);
                wmma::load_matrix_sync(b, &sV[n * WN * D + k * WK], D);
                wmma::mma_sync(acc[n], a, b, acc[n]);
            }
            wmma::store_matrix_sync(&sS[warp_id * WM * BN + n * WN], acc[n], BN, wmma::mem_row_major);
        }
        __syncthreads();

        // dS = P * (dP - D) * scale  -> sdS (bf16), with kv masking
        for (int idx = tid; idx < BM * BN; idx += THREADS) {
            int r = idx / BN;
            int c = idx % BN;
            int kv_row = kv_base + c;
            float p = __bfloat162float(sP[idx]);
            float dp = sS[idx];
            float ds = (kv_row < S) ? (p * (dp - sDrow[r]) * attn_scale) : 0.0f;
            sdS[idx] = __float2bfloat16(ds);
        }
        __syncthreads();

        // dQ += dS @ K  (accumulate in registers)
        #pragma unroll
        for (int n = 0; n < NN_D; ++n) {
            #pragma unroll
            for (int k = 0; k < NK_BN; ++k) {
                wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::row_major> b;
                wmma::load_matrix_sync(a, &sdS[warp_id * WM * BN + k * WK], BN);
                wmma::load_matrix_sync(b, &sK[k * WK * D + n * WN], D);
                wmma::mma_sync(dQ_frag[n], a, b, dQ_frag[n]);
            }
        }

        // dV += P^T @ dO -> sDQf -> atomicAdd to global FP32
        #pragma unroll
        for (int n = 0; n < NN_D; ++n) {
            wmma::fill_fragment(acc[n], 0.0f);
            #pragma unroll
            for (int k = 0; k < NK_BN; ++k) {
                wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::col_major> a;
                wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::row_major> b;
                wmma::load_matrix_sync(a, &sP[k * WK * BN + warp_id * WM], BN);
                wmma::load_matrix_sync(b, &sdO[k * WK * D + n * WN], D);
                wmma::mma_sync(acc[n], a, b, acc[n]);
            }
            wmma::store_matrix_sync(&sDQf[warp_id * WM * D + n * WN], acc[n], D, wmma::mem_row_major);
        }
        __syncthreads();
        for (int idx = tid; idx < BN * D; idx += THREADS) {
            int r = idx / D; int c = idx % D;
            int kv_row = kv_base + r;
            if (kv_row < S) {
                atomicAdd(&dV_h[kv_row * D + c], sDQf[idx]);
            }
        }
        __syncthreads();

        // dK += dS^T @ Q -> sDQf -> atomicAdd to global FP32
        #pragma unroll
        for (int n = 0; n < NN_D; ++n) {
            wmma::fill_fragment(acc[n], 0.0f);
            #pragma unroll
            for (int k = 0; k < NK_BN; ++k) {
                wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::col_major> a;
                wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::row_major> b;
                wmma::load_matrix_sync(a, &sdS[k * WK * BN + warp_id * WM], BN);
                wmma::load_matrix_sync(b, &sQ[k * WK * D + n * WN], D);
                wmma::mma_sync(acc[n], a, b, acc[n]);
            }
            wmma::store_matrix_sync(&sDQf[warp_id * WM * D + n * WN], acc[n], D, wmma::mem_row_major);
        }
        __syncthreads();
        for (int idx = tid; idx < BN * D; idx += THREADS) {
            int r = idx / D; int c = idx % D;
            int kv_row = kv_base + r;
            if (kv_row < S) {
                atomicAdd(&dK_h[kv_row * D + c], sDQf[idx]);
            }
        }
        __syncthreads();
    }

    // Store dQ to global (non-atomic, direct BF16 write)
    #pragma unroll
    for (int n = 0; n < NN_D; ++n) {
        wmma::store_matrix_sync(&sDQf[warp_id * WM * D + n * WN], dQ_frag[n], D, wmma::mem_row_major);
    }
    __syncthreads();
    for (int idx = tid; idx < BM * D; idx += THREADS) {
        int r = idx / D; int c = idx % D;
        int q_row = q_base + r;
        if (q_row < S) {
            dQ_h[q_row * D + c] = __float2bfloat16(sDQf[idx]);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV)
{
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = Q.size(3);
    float attn_scale = 1.0f / sqrtf((float)d);

    const __nv_bfloat16* Q_p  = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_p  = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_p  = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_p  = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_p = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float*         L_p  = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_p = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_p = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_p = static_cast<__nv_bfloat16*>(dV.data_ptr());

    int64_t total_elems = B * H * S * d;
    int num_q_blocks = (int)((S + BM - 1) / BM);
    dim3 grid((int)B, (int)H, num_q_blocks);
    dim3 block(THREADS);

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    // Allocate and zero FP32 workspaces for dK and dV atomic accumulations
    float* dK_fp32;
    float* dV_fp32;
    CUDA_CHECK(cudaMalloc(&dK_fp32, total_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dV_fp32, total_elems * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(dK_fp32, 0, total_elems * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(dV_fp32, 0, total_elems * sizeof(float), stream));

    CUDA_CHECK(cudaFuncSetAttribute(mha_bwd_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));

    mha_bwd_kernel<<<grid, block, SMEM_BYTES, stream>>>(
        Q_p, K_p, V_p, O_p, dO_p, L_p, dQ_p, dK_fp32, dV_fp32, (int)S, attn_scale);
    CUDA_CHECK(cudaGetLastError());

    // Convert dK and dV from FP32 to BF16
    int64_t threads_2d = 256;
    int64_t blocks_2d = (total_elems + threads_2d - 1) / threads_2d;
    convert_f32_to_bf16<<<blocks_2d, threads_2d, 0, stream>>>(dK_fp32, dK_p, total_elems);
    convert_f32_to_bf16<<<blocks_2d, threads_2d, 0, stream>>>(dV_fp32, dV_p, total_elems);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(dK_fp32));
    CUDA_CHECK(cudaFree(dV_fp32));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_bwd::run);

}  // namespace tvm_ffi_mha_bwd