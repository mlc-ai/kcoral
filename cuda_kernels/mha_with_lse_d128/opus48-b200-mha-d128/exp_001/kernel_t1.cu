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
constexpr int BM = 128;   // query rows per block
constexpr int BN = 64;    // KV rows per tile
constexpr int NTHREADS = 256;   // 8 warps
constexpr int NWARPS = NTHREADS / 32;

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

// Vectorized (int4 = 8 bf16) coalesced load of nrows x D tile, zero-padded.
__device__ __forceinline__ void load_tile(const __nv_bfloat16* __restrict__ src,
                                          int rows_valid, int nrows,
                                          __nv_bfloat16* __restrict__ dst,
                                          int tid) {
    const int4* src4 = reinterpret_cast<const int4*>(src);
    int4* dst4 = reinterpret_cast<int4*>(dst);
    const int upr = D / 8;           // int4 units per row = 16
    const int total = nrows * upr;
    int4 z; z.x = z.y = z.z = z.w = 0;
    for (int u = tid; u < total; u += NTHREADS) {
        int r = u / upr;
        if (r < rows_valid) dst4[u] = src4[u];
        else                dst4[u] = z;
    }
}

__global__ __launch_bounds__(NTHREADS, 1) void mha_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S, float scale)
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
    const int warp = tid >> 5;     // 0..7
    const int row0 = warp * 16;    // this warp's 16-row band

    const int bh = blockIdx.y;     // = b*H + h
    const int q_start = blockIdx.x * BM;

    const __nv_bfloat16* Qbase = Q + (int64_t)bh * S * D;
    const __nv_bfloat16* Kbase = K + (int64_t)bh * S * D;
    const __nv_bfloat16* Vbase = V + (int64_t)bh * S * D;
    __nv_bfloat16* Obase = O + (int64_t)bh * S * D;
    float* LSEbase = LSE + (int64_t)bh * S;

    // init O, m, l
    for (int i = tid; i < BM*D; i += NTHREADS) Os[i] = 0.0f;
    for (int i = tid; i < BM;   i += NTHREADS) { ms[i] = -1e30f; ls[i] = 0.0f; }

    int q_valid = S - q_start; if (q_valid > BM) q_valid = BM; if (q_valid < 0) q_valid = 0;
    load_tile(Qbase + (int64_t)q_start*D, q_valid, BM, Qs, tid);
    __syncthreads();

    // preload this warp's Q fragments (row band, over D in 8 k-tiles)
    wmma::fragment<wmma::matrix_a, 16,16,16, __nv_bfloat16, wmma::row_major> a_q[D/16];
    #pragma unroll
    for (int k = 0; k < D/16; ++k)
        wmma::load_matrix_sync(a_q[k], Qs + row0*D + k*16, D);

    const int num_kv = (S + BN - 1) / BN;
    for (int kt = 0; kt < num_kv; ++kt) {
        int kv_start = kt * BN;
        int valid = S - kv_start; if (valid > BN) valid = BN;

        load_tile(Kbase + (int64_t)kv_start*D, valid, BN, Ks, tid);
        load_tile(Vbase + (int64_t)kv_start*D, valid, BN, Vs, tid);
        __syncthreads();

        // S = Q @ K^T  : warp handles rows [row0, row0+16), all BN cols
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
            wmma::store_matrix_sync(Ss + row0*BN + n*16, c_frag, BN, wmma::mem_row_major);
        }
        __syncthreads();

        // Online softmax: thread i handles row i (i < BM)
        if (tid < BM) {
            int i = tid;
            const float* srow = Ss + i*BN;
            __nv_bfloat16* prow = Ps + i*BN;

            float tmax = -1e30f;
            #pragma unroll
            for (int j = 0; j < BN; ++j) {
                if (j < valid) tmax = fmaxf(tmax, srow[j] * scale);
            }
            float m_old = ms[i];
            float m_new = fmaxf(m_old, tmax);
            float alpha = (m_new > m_old) ? __expf(m_old - m_new) : 1.0f;

            float rowsum = 0.0f;
            #pragma unroll
            for (int j = 0; j < BN; ++j) {
                float p = 0.0f;
                if (j < valid) p = __expf(srow[j] * scale - m_new);
                rowsum += p;
                prow[j] = __float2bfloat16(p);
            }
            float l_new = ls[i] * alpha + rowsum;

            if (alpha != 1.0f) {
                float* orow = Os + i*D;
                #pragma unroll
                for (int d = 0; d < D; ++d) orow[d] *= alpha;
            }
            ms[i] = m_new;
            ls[i] = l_new;
        }
        __syncthreads();

        // O += P @ V : warp handles rows [row0, row0+16), all D cols
        #pragma unroll
        for (int n = 0; n < D/16; ++n) {
            wmma::fragment<wmma::accumulator, 16,16,16, float> c_frag;
            wmma::load_matrix_sync(c_frag, Os + row0*D + n*16, D, wmma::mem_row_major);
            #pragma unroll
            for (int k = 0; k < BN/16; ++k) {
                wmma::fragment<wmma::matrix_a, 16,16,16, __nv_bfloat16, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16,16,16, __nv_bfloat16, wmma::row_major> b_frag;
                wmma::load_matrix_sync(a_frag, Ps + row0*BN + k*16, BN);
                wmma::load_matrix_sync(b_frag, Vs + (k*16)*D + n*16, D);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            wmma::store_matrix_sync(Os + row0*D + n*16, c_frag, D, wmma::mem_row_major);
        }
        __syncthreads();
    }

    // finalize
    if (tid < BM) {
        int i = tid;
        int gq = q_start + i;
        if (gq < S) {
            float l = ls[i];
            float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
            const float* orow = Os + i*D;
            __nv_bfloat16* out = Obase + (int64_t)gq*D;
            #pragma unroll
            for (int d = 0; d < D; ++d) out[d] = __float2bfloat16(orow[d] * inv);
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

    mha_kernel<<<grid, block, smem, stream>>>(Qp, Kp, Vp, Op, Lp, Sd, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_d128::run);

}  // namespace mha_d128