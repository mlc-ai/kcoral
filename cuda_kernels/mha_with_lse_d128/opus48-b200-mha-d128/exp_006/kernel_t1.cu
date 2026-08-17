#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda;

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_kernel {

constexpr int BM = 64, BN = 64, HD = 128;
constexpr int THREADS = 128;
constexpr int NB = BN / 16;  // 4
constexpr int NK = HD / 16;  // 8
constexpr int ND = HD / 16;  // 8

__global__ __launch_bounds__(THREADS)
void attn_kernel(const __nv_bfloat16* __restrict__ Q,
                 const __nv_bfloat16* __restrict__ K,
                 const __nv_bfloat16* __restrict__ V,
                 __nv_bfloat16* __restrict__ O,
                 float* __restrict__ LSE,
                 int S, float scale) {
    int bh = blockIdx.y;
    int q_start = blockIdx.x * BM;
    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int groupID = lane >> 2;
    int tig = lane & 3;
    int warpRow = warp * 16;

    extern __shared__ char smem[];
    __nv_bfloat16* Qs = reinterpret_cast<__nv_bfloat16*>(smem);   // 16384
    __nv_bfloat16* Ks = Qs + BM * HD;                             // 16384
    __nv_bfloat16* Vs = Ks + BN * HD;                             // 16384
    __nv_bfloat16* Ps = Vs + BN * HD;                             // 8192
    float* l_sh = reinterpret_cast<float*>(Ps + BM * BN);         // BM
    float* Osh = reinterpret_cast<float*>(Ks);                    // reuse Ks+Vs (32768B)

    const __nv_bfloat16* Qbase = Q + (size_t)bh * S * HD;
    const __nv_bfloat16* Kbase = K + (size_t)bh * S * HD;
    const __nv_bfloat16* Vbase = V + (size_t)bh * S * HD;

    // Load Q tile
    for (int idx = tid; idx < BM * HD; idx += THREADS) {
        int r = idx / HD, c = idx % HD, gr = q_start + r;
        Qs[idx] = (gr < S) ? Qbase[(size_t)gr * HD + c] : __float2bfloat16(0.0f);
    }

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> Of[ND];
    #pragma unroll
    for (int i = 0; i < ND; i++) wmma::fill_fragment(Of[i], 0.0f);

    float mA = -1e30f, mB = -1e30f, lA = 0.f, lB = 0.f;
    __syncthreads();

    int num_kv = (S + BN - 1) / BN;
    for (int kv = 0; kv < num_kv; kv++) {
        int kv_start = kv * BN;
        for (int idx = tid; idx < BN * HD; idx += THREADS) {
            int r = idx / HD, c = idx % HD, gr = kv_start + r;
            if (gr < S) {
                Ks[idx] = Kbase[(size_t)gr * HD + c];
                Vs[idx] = Vbase[(size_t)gr * HD + c];
            } else {
                Ks[idx] = __float2bfloat16(0.0f);
                Vs[idx] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        // S = Q @ K^T
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> Sf[NB];
        #pragma unroll
        for (int n = 0; n < NB; n++) wmma::fill_fragment(Sf[n], 0.0f);
        #pragma unroll
        for (int k = 0; k < NK; k++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
            wmma::load_matrix_sync(af, &Qs[warpRow * HD + k * 16], HD);
            #pragma unroll
            for (int n = 0; n < NB; n++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf;
                wmma::load_matrix_sync(bf, &Ks[(n * 16) * HD + k * 16], HD);
                wmma::mma_sync(Sf[n], af, bf, Sf[n]);
            }
        }

        // scale + mask + rowmax (per-thread over its held elements)
        float locMaxA = -1e30f, locMaxB = -1e30f;
        #pragma unroll
        for (int n = 0; n < NB; n++) {
            #pragma unroll
            for (int t = 0; t < 8; t++) {
                int col = n * 16 + (t >> 2) * 8 + tig * 2 + (t & 1);
                float v = Sf[n].x[t] * scale;
                if (kv_start + col >= S) v = -1e30f;
                Sf[n].x[t] = v;
                if (((t >> 1) & 1) == 0) locMaxA = fmaxf(locMaxA, v);
                else                     locMaxB = fmaxf(locMaxB, v);
            }
        }
        locMaxA = fmaxf(locMaxA, __shfl_xor_sync(0xffffffff, locMaxA, 1));
        locMaxA = fmaxf(locMaxA, __shfl_xor_sync(0xffffffff, locMaxA, 2));
        locMaxB = fmaxf(locMaxB, __shfl_xor_sync(0xffffffff, locMaxB, 1));
        locMaxB = fmaxf(locMaxB, __shfl_xor_sync(0xffffffff, locMaxB, 2));

        float mA_new = fmaxf(mA, locMaxA), mB_new = fmaxf(mB, locMaxB);
        float corrA = __expf(mA - mA_new), corrB = __expf(mB - mB_new);

        float sumA = 0.f, sumB = 0.f;
        #pragma unroll
        for (int n = 0; n < NB; n++) {
            #pragma unroll
            for (int t = 0; t < 8; t++) {
                int rs = (t >> 1) & 1;
                float p = __expf(Sf[n].x[t] - (rs ? mB_new : mA_new));
                Sf[n].x[t] = p;
                if (rs == 0) sumA += p; else sumB += p;
            }
        }
        sumA += __shfl_xor_sync(0xffffffff, sumA, 1); sumA += __shfl_xor_sync(0xffffffff, sumA, 2);
        sumB += __shfl_xor_sync(0xffffffff, sumB, 1); sumB += __shfl_xor_sync(0xffffffff, sumB, 2);

        lA = lA * corrA + sumA; lB = lB * corrB + sumB;
        mA = mA_new; mB = mB_new;

        // store P (bf16) to shared
        #pragma unroll
        for (int n = 0; n < NB; n++) {
            #pragma unroll
            for (int t = 0; t < 8; t++) {
                int rs = (t >> 1) & 1;
                int row = warpRow + (rs ? groupID + 8 : groupID);
                int col = n * 16 + (t >> 2) * 8 + tig * 2 + (t & 1);
                Ps[row * BN + col] = __float2bfloat16(Sf[n].x[t]);
            }
        }
        __syncthreads();

        // rescale O fragments (per-row correction)
        #pragma unroll
        for (int nd = 0; nd < ND; nd++) {
            #pragma unroll
            for (int t = 0; t < 8; t++)
                Of[nd].x[t] *= (((t >> 1) & 1) ? corrB : corrA);
        }

        // O += P @ V
        #pragma unroll
        for (int k = 0; k < NB; k++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
            wmma::load_matrix_sync(af, &Ps[warpRow * BN + k * 16], BN);
            #pragma unroll
            for (int nd = 0; nd < ND; nd++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bf;
                wmma::load_matrix_sync(bf, &Vs[(k * 16) * HD + nd * 16], HD);
                wmma::mma_sync(Of[nd], af, bf, Of[nd]);
            }
        }
        __syncthreads();
    }

    // write LSE and l
    if (tig == 0) {
        int rA = warpRow + groupID, rB = warpRow + groupID + 8;
        l_sh[rA] = lA; l_sh[rB] = lB;
        int gA = q_start + rA, gB = q_start + rB;
        if (gA < S) LSE[(size_t)bh * S + gA] = mA + logf(lA);
        if (gB < S) LSE[(size_t)bh * S + gB] = mB + logf(lB);
    }

    #pragma unroll
    for (int nd = 0; nd < ND; nd++)
        wmma::store_matrix_sync(&Osh[warpRow * HD + nd * 16], Of[nd], HD, wmma::mem_row_major);
    __syncthreads();

    __nv_bfloat16* Obase = O + (size_t)bh * S * HD;
    for (int idx = tid; idx < BM * HD; idx += THREADS) {
        int r = idx / HD, c = idx % HD, gr = q_start + r;
        if (gr < S) {
            float inv = 1.0f / l_sh[r];
            Obase[(size_t)gr * HD + c] = __float2bfloat16(Osh[idx] * inv);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int B = Q.size(0), H = Q.size(1), S = Q.size(2);
    float scale = 1.0f / sqrtf((float)HD);

    const __nv_bfloat16* Qp = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* Kp = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* Vp = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* Op = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* Lp = static_cast<float*>(LSE.data_ptr());

    int num_q = (S + BM - 1) / BM;
    dim3 grid(num_q, B * H);
    dim3 block(THREADS);
    size_t smem = (size_t)BM * HD * 2 + (size_t)BN * HD * 2 + (size_t)BN * HD * 2
                + (size_t)BM * BN * 2 + (size_t)BM * 4;

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(attn_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    attn_kernel<<<grid, block, smem, stream>>>(Qp, Kp, Vp, Op, Lp, S, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel::run);

}  // namespace mha_kernel