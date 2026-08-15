#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
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

namespace mha_d128 {

constexpr int D  = 128;
constexpr int BM = 64;   // query rows per block
constexpr int BN = 64;   // KV rows per tile
constexpr int NTHREADS = 128;

// Shared memory byte size for the kernel
static inline size_t smem_bytes() {
    size_t s = 0;
    s += (size_t)BM*D*2;   // Qs bf16
    s += (size_t)BN*D*2;   // Ks bf16
    s += (size_t)BN*D*2;   // Vs bf16
    s += (size_t)BM*BN*4;  // Ss fp32
    s += (size_t)BM*BN*2;  // Ps bf16
    s += (size_t)BM*D*4;   // Os fp32
    s += (size_t)BM*4;     // ms
    s += (size_t)BM*4;     // ls
    return s;
}

__global__ void mha_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int H, int S, float scale)
{
    extern __shared__ char smem_raw[];
    __nv_bfloat16* Qs = reinterpret_cast<__nv_bfloat16*>(smem_raw);   // [BM][D]
    __nv_bfloat16* Ks = Qs + BM*D;                                    // [BN][D]
    __nv_bfloat16* Vs = Ks + BN*D;                                    // [BN][D]
    float*         Ss = reinterpret_cast<float*>(Vs + BN*D);          // [BM][BN]
    __nv_bfloat16* Ps = reinterpret_cast<__nv_bfloat16*>(Ss + BM*BN); // [BM][BN]
    float*         Os = reinterpret_cast<float*>(Ps + BM*BN);         // [BM][D]
    float*         ms = Os + BM*D;                                    // [BM]
    float*         ls = ms + BM;                                      // [BM]

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;     // 0..3

    const int bh = blockIdx.y;     // = b*H + h
    const int q_tile = blockIdx.x;
    const int q_start = q_tile * BM;

    const __nv_bfloat16* Qbase = Q + (int64_t)bh * S * D;
    const __nv_bfloat16* Kbase = K + (int64_t)bh * S * D;
    const __nv_bfloat16* Vbase = V + (int64_t)bh * S * D;
    __nv_bfloat16* Obase = O + (int64_t)bh * S * D;
    float* LSEbase = LSE + (int64_t)bh * S;

    const __nv_bfloat16 zero_bf = __float2bfloat16(0.0f);

    // init O, m, l
    for (int i = tid; i < BM*D; i += NTHREADS) Os[i] = 0.0f;
    for (int i = tid; i < BM;   i += NTHREADS) { ms[i] = -1e30f; ls[i] = 0.0f; }

    // load Q tile (zero padded)
    for (int idx = tid; idx < BM*D; idx += NTHREADS) {
        int r = idx / D, c = idx % D;
        int gq = q_start + r;
        Qs[idx] = (gq < S) ? Qbase[(int64_t)gq*D + c] : zero_bf;
    }
    __syncthreads();

    // preload Q fragments (row band = warp*16, over D in 8 k-tiles)
    wmma::fragment<wmma::matrix_a, 16,16,16, __nv_bfloat16, wmma::row_major> a_q[8];
    #pragma unroll
    for (int k = 0; k < D/16; ++k) {
        wmma::load_matrix_sync(a_q[k], Qs + (warp*16)*D + k*16, D);
    }

    const int num_kv = (S + BN - 1) / BN;
    for (int kt = 0; kt < num_kv; ++kt) {
        int kv_start = kt * BN;
        int valid = (S - kv_start < BN) ? (S - kv_start) : BN;

        // load K, V tile (zero padded)
        for (int idx = tid; idx < BN*D; idx += NTHREADS) {
            int r = idx / D, c = idx % D;
            int gk = kv_start + r;
            if (gk < S) {
                Ks[idx] = Kbase[(int64_t)gk*D + c];
                Vs[idx] = Vbase[(int64_t)gk*D + c];
            } else {
                Ks[idx] = zero_bf;
                Vs[idx] = zero_bf;
            }
        }
        __syncthreads();

        // S = Q @ K^T   (warp handles rows [warp*16 .. +16], all BN cols)
        #pragma unroll
        for (int n = 0; n < BN/16; ++n) {
            wmma::fragment<wmma::accumulator, 16,16,16, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);
            #pragma unroll
            for (int k = 0; k < D/16; ++k) {
                wmma::fragment<wmma::matrix_b, 16,16,16, __nv_bfloat16, wmma::col_major> b_frag;
                wmma::load_matrix_sync(b_frag, Ks + (n*16)*D + k*16, D);
                wmma::mma_sync(c_frag, a_q[k], b_frag, c_frag);
            }
            wmma::store_matrix_sync(Ss + (warp*16)*BN + n*16, c_frag, BN, wmma::mem_row_major);
        }
        __syncthreads();

        // Online softmax: thread i handles row i (i < BM)
        if (tid < BM) {
            int i = tid;
            float tmax = -1e30f;
            #pragma unroll 8
            for (int j = 0; j < valid; ++j) {
                float s = Ss[i*BN + j] * scale;
                tmax = fmaxf(tmax, s);
            }
            float m_old = ms[i];
            float m_new = fmaxf(m_old, tmax);
            float alpha = __expf(m_old - m_new);   // 0 on first tile (m_old = -1e30)
            float l_new = ls[i] * alpha;

            for (int j = 0; j < BN; ++j) {
                float p = 0.0f;
                if (j < valid) {
                    float s = Ss[i*BN + j] * scale;
                    p = __expf(s - m_new);
                }
                l_new += p;
                Ps[i*BN + j] = __float2bfloat16(p);
            }
            #pragma unroll
            for (int d = 0; d < D; ++d) Os[i*D + d] *= alpha;

            ms[i] = m_new;
            ls[i] = l_new;
        }
        __syncthreads();

        // O += P @ V   (warp handles rows [warp*16 .. +16], all D cols)
        #pragma unroll
        for (int n = 0; n < D/16; ++n) {
            wmma::fragment<wmma::accumulator, 16,16,16, float> c_frag;
            wmma::load_matrix_sync(c_frag, Os + (warp*16)*D + n*16, D, wmma::mem_row_major);
            #pragma unroll
            for (int k = 0; k < BN/16; ++k) {
                wmma::fragment<wmma::matrix_a, 16,16,16, __nv_bfloat16, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16,16,16, __nv_bfloat16, wmma::row_major> b_frag;
                wmma::load_matrix_sync(a_frag, Ps + (warp*16)*BN + k*16, BN);
                wmma::load_matrix_sync(b_frag, Vs + (k*16)*D + n*16, D);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            wmma::store_matrix_sync(Os + (warp*16)*D + n*16, c_frag, D, wmma::mem_row_major);
        }
        __syncthreads();
    }

    // finalize: normalize and write O, LSE
    if (tid < BM) {
        int i = tid;
        int gq = q_start + i;
        if (gq < S) {
            float l = ls[i];
            float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
            #pragma unroll
            for (int d = 0; d < D; ++d) {
                Obase[(int64_t)gq*D + d] = __float2bfloat16(Os[i*D + d] * inv);
            }
            LSEbase[gq] = ms[i] + logf(l);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int Bd = (int)Q.size(0);
    int Hd = (int)Q.size(1);
    int Sd = (int)Q.size(2);
    int Dd = (int)Q.size(3);

    const __nv_bfloat16* Qp = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* Kp = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* Vp = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* Op = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* Lp = static_cast<float*>(LSE.data_ptr());

    float scale = 1.0f / sqrtf((float)Dd);

    int num_q_tiles = (Sd + BM - 1) / BM;
    dim3 grid(num_q_tiles, Bd * Hd);
    dim3 block(NTHREADS);

    size_t smem = smem_bytes();
    static bool attr_set = false;
    if (!attr_set) {
        CUDA_CHECK(cudaFuncSetAttribute(
            (const void*)mha_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem));
        attr_set = true;
    }

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    mha_kernel<<<grid, block, smem, stream>>>(Qp, Kp, Vp, Op, Lp, Hd, Sd, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_d128::run);

}  // namespace mha_d128