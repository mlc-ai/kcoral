#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>
#include <math.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda;

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
    }                                                              \
} while(0)

namespace mha_d128_causal {

constexpr int D = 128;
constexpr int BM = 128;   // query rows per CTA
constexpr int BN = 64;    // key rows per iteration
constexpr int NWARPS = 8;
constexpr int NTHREADS = NWARPS * 32;

__device__ __forceinline__ void load_tile(const __nv_bfloat16* base, int row0,
                                           __nv_bfloat16* dst, int S, int nrows, int tid) {
    const uint4* src = reinterpret_cast<const uint4*>(base);
    uint4* d = reinterpret_cast<uint4*>(dst);
    const int vpr = D / 8; // 16 uint4 per row (128 bf16)
    int total = nrows * vpr;
    uint4 zero = make_uint4(0, 0, 0, 0);
    for (int v = tid; v < total; v += NTHREADS) {
        int r = v / vpr;
        int c = v - r * vpr;
        int s = row0 + r;
        if (s < S) d[r * vpr + c] = src[(int64_t)s * vpr + c];
        else       d[r * vpr + c] = zero;
    }
}

__global__ void __launch_bounds__(NTHREADS, 1) attn_kernel(
        const __nv_bfloat16* __restrict__ Q,
        const __nv_bfloat16* __restrict__ K,
        const __nv_bfloat16* __restrict__ V,
        __nv_bfloat16* __restrict__ O,
        float* __restrict__ LSE,
        int B, int H, int S, float scale) {
    extern __shared__ char smem[];
    __nv_bfloat16* Qsh = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* Ksh = Qsh + BM * D;
    __nv_bfloat16* Vsh = Ksh + BN * D;
    float*         Ssh = reinterpret_cast<float*>(Vsh + BN * D);
    __nv_bfloat16* Psh = reinterpret_cast<__nv_bfloat16*>(Ssh + BM * BN);
    float*         Osh = reinterpret_cast<float*>(Psh + BM * BN);
    float* m_arr = Osh + BM * D;
    float* l_arr = m_arr + BM;

    int tid  = threadIdx.x;
    int warp = tid >> 5;
    int b    = blockIdx.z;
    int h    = blockIdx.y;
    int qt   = blockIdx.x;
    int q0   = qt * BM;

    int64_t head_off = (int64_t)(b * H + h) * S * D;
    const __nv_bfloat16* Qbase = Q + head_off;
    const __nv_bfloat16* Kbase = K + head_off;
    const __nv_bfloat16* Vbase = V + head_off;
    __nv_bfloat16* Obase = O + head_off;
    float* LSEbase = LSE + (int64_t)(b * H + h) * S;

    for (int i = tid; i < BM * D; i += NTHREADS) Osh[i] = 0.0f;
    for (int i = tid; i < BM; i += NTHREADS) { m_arr[i] = -INFINITY; l_arr[i] = 0.0f; }

    load_tile(Qbase, q0, Qsh, S, BM, tid);
    __syncthreads();

    int qg_max = q0 + BM - 1; if (qg_max > S - 1) qg_max = S - 1;
    int num_kt = qg_max / BN + 1;

    int R = warp * 16; // this warp's row base within the tile

    for (int jt = 0; jt < num_kt; ++jt) {
        int k0 = jt * BN;
        load_tile(Kbase, k0, Ksh, S, BN, tid);
        load_tile(Vbase, k0, Vsh, S, BN, tid);
        __syncthreads();

        // ---- S = scale * Q @ K^T ----
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
            for (int ct = 0; ct < BN / 16; ++ct) {
                wmma::fill_fragment(c_frag, 0.0f);
                #pragma unroll
                for (int kt = 0; kt < D / 16; ++kt) {
                    wmma::load_matrix_sync(a_frag, Qsh + R * D + kt * 16, D);
                    wmma::load_matrix_sync(b_frag, Ksh + (ct * 16) * D + kt * 16, D);
                    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                }
                #pragma unroll
                for (int t = 0; t < c_frag.num_elements; t++) c_frag.x[t] *= scale;
                wmma::store_matrix_sync(Ssh + R * BN + ct * 16, c_frag, BN, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // ---- online softmax (one thread per query row) ----
        if (tid < BM) {
            int i  = tid;
            int qg = q0 + i;
            float mi = m_arr[i];
            float tmax = -INFINITY;
            for (int j = 0; j < BN; j++) {
                int kg = k0 + j;
                if (kg <= qg && kg < S) tmax = fmaxf(tmax, Ssh[i * BN + j]);
            }
            if (tmax != -INFINITY) {
                float m_new = fmaxf(mi, tmax);
                float corr  = (mi == -INFINITY) ? 0.0f : __expf(mi - m_new);
                float rowsum = 0.0f;
                for (int j = 0; j < BN; j++) {
                    int kg = k0 + j;
                    float p;
                    if (kg <= qg && kg < S) p = __expf(Ssh[i * BN + j] - m_new);
                    else                    p = 0.0f;
                    rowsum += p;
                    Psh[i * BN + j] = __float2bfloat16(p);
                }
                l_arr[i] = l_arr[i] * corr + rowsum;
                m_arr[i] = m_new;
                if (corr != 1.0f) {
                    float* orow = Osh + i * D;
                    #pragma unroll 8
                    for (int d = 0; d < D; d++) orow[d] *= corr;
                }
            } else {
                for (int j = 0; j < BN; j++) Psh[i * BN + j] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        // ---- O += P @ V ----
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            for (int ct = 0; ct < D / 16; ++ct) {
                wmma::load_matrix_sync(acc, Osh + R * D + ct * 16, D, wmma::mem_row_major);
                #pragma unroll
                for (int kt = 0; kt < BN / 16; ++kt) {
                    wmma::load_matrix_sync(a_frag, Psh + R * BN + kt * 16, BN);
                    wmma::load_matrix_sync(b_frag, Vsh + (kt * 16) * D + ct * 16, D);
                    wmma::mma_sync(acc, a_frag, b_frag, acc);
                }
                wmma::store_matrix_sync(Osh + R * D + ct * 16, acc, D, wmma::mem_row_major);
            }
        }
        __syncthreads();
    }

    // ---- finalize ----
    if (tid < BM) {
        int i = tid;
        int s = q0 + i;
        if (s < S) {
            float li  = l_arr[i];
            float inv = (li > 0.0f) ? (1.0f / li) : 0.0f;
            float* orow = Osh + i * D;
            __nv_bfloat16* out = Obase + (int64_t)s * D;
            for (int d = 0; d < D; d++) out[d] = __float2bfloat16(orow[d] * inv);
            LSEbase[s] = m_arr[i] + logf(li);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int Bb = (int)Q.size(0);
    int Hh = (int)Q.size(1);
    int Ss = (int)Q.size(2);
    int Dd = (int)Q.size(3);
    float scale = 1.0f / sqrtf((float)Dd);

    const __nv_bfloat16* Qp = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* Kp = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* Vp = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* Op = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSEp = static_cast<float*>(LSE.data_ptr());

    int num_q_tiles = (Ss + BM - 1) / BM;
    dim3 grid(num_q_tiles, Hh, Bb);
    dim3 block(NTHREADS);

    size_t smem = (size_t)BM * D * 2            // Qsh
                + (size_t)BN * D * 2            // Ksh
                + (size_t)BN * D * 2            // Vsh
                + (size_t)BM * BN * 4           // Ssh (fp32)
                + (size_t)BM * BN * 2           // Psh (bf16)
                + (size_t)BM * D * 4            // Osh (fp32)
                + (size_t)BM * 4                // m
                + (size_t)BM * 4;               // l

    cudaFuncSetAttribute(attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    attn_kernel<<<grid, block, smem, stream>>>(Qp, Kp, Vp, Op, LSEp, Bb, Hh, Ss, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_d128_causal::run);

}  // namespace mha_d128_causal