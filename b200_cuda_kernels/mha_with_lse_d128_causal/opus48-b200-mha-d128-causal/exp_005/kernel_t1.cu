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

constexpr int D  = 128;
constexpr int BM = 128;
constexpr int BN = 64;
constexpr int NWARPS = 8;
constexpr int NTHREADS = NWARPS * 32;
constexpr int SST = BM;   // col-major stride for S
constexpr int PST = BM;   // col-major stride for P
constexpr int OST = BM;   // col-major stride for O

__device__ __forceinline__ void load_tile(const __nv_bfloat16* base, int row0,
                                           __nv_bfloat16* dst, int S, int nrows, int tid) {
    const uint4* src = reinterpret_cast<const uint4*>(base);
    uint4* d = reinterpret_cast<uint4*>(dst);
    const int vpr = D / 8;
    int total = nrows * vpr;
    uint4 zero = make_uint4(0, 0, 0, 0);
    for (int v = tid; v < total; v += NTHREADS) {
        int r = v / vpr;
        int c = v - r * vpr;
        int s = row0 + r;
        d[r * vpr + c] = (s < S) ? src[(int64_t)s * vpr + c] : zero;
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
    float*         Ssh = reinterpret_cast<float*>(Vsh + BN * D);       // col-major [BN][BM]
    __nv_bfloat16* Psh = reinterpret_cast<__nv_bfloat16*>(Ssh + BN * SST); // col-major [BN][BM]
    float*         Osh = reinterpret_cast<float*>(Psh + BN * PST);     // col-major [D][BM]
    float* m_arr = Osh + D * OST;
    float* l_arr = m_arr + BM;
    float* corr  = l_arr + BM;

    int tid  = threadIdx.x;
    int warp = tid >> 5;
    int b = blockIdx.z, h = blockIdx.y, qt = blockIdx.x, q0 = qt * BM;

    int64_t head = ((int64_t)(b * H + h)) * S * D;
    const __nv_bfloat16* Qb = Q + head;
    const __nv_bfloat16* Kb = K + head;
    const __nv_bfloat16* Vb = V + head;
    __nv_bfloat16* Ob = O + head;
    float* LSEb = LSE + ((int64_t)(b * H + h)) * S;

    for (int i = tid; i < D * OST; i += NTHREADS) Osh[i] = 0.f;
    if (tid < BM) { m_arr[tid] = -INFINITY; l_arr[tid] = 0.f; }
    load_tile(Qb, q0, Qsh, S, BM, tid);
    __syncthreads();

    int qg_max = q0 + BM - 1; if (qg_max > S - 1) qg_max = S - 1;
    int num_kt = qg_max / BN + 1;
    int R = warp * 16;

    for (int jt = 0; jt < num_kt; ++jt) {
        int k0 = jt * BN;
        __syncthreads();                       // safe to overwrite K/V (prev PV done)
        load_tile(Kb, k0, Ksh, S, BN, tid);
        load_tile(Vb, k0, Vsh, S, BN, tid);
        __syncthreads();

        // ---- S = scale * Q @ K^T  (store col-major) ----
        {
            wmma::fragment<wmma::matrix_a, 16,16,16, __nv_bfloat16, wmma::row_major> af;
            wmma::fragment<wmma::matrix_b, 16,16,16, __nv_bfloat16, wmma::col_major> bf;
            wmma::fragment<wmma::accumulator, 16,16,16, float> cf;
            for (int ct = 0; ct < BN/16; ++ct) {
                wmma::fill_fragment(cf, 0.f);
                #pragma unroll
                for (int kt = 0; kt < D/16; ++kt) {
                    wmma::load_matrix_sync(af, Qsh + R*D + kt*16, D);
                    wmma::load_matrix_sync(bf, Ksh + (ct*16)*D + kt*16, D);
                    wmma::mma_sync(cf, af, bf, cf);
                }
                #pragma unroll
                for (int t = 0; t < cf.num_elements; ++t) cf.x[t] *= scale;
                wmma::store_matrix_sync(Ssh + R + (ct*16)*SST, cf, SST, wmma::mem_col_major);
            }
        }
        __syncthreads();

        // ---- online softmax + rescale O + write P (col-major, conflict-free) ----
        if (tid < BM) {
            int i = tid, qg = q0 + i;
            float mi = m_arr[i];
            float tmax = -INFINITY;
            #pragma unroll 4
            for (int j = 0; j < BN; ++j) {
                int kg = k0 + j;
                if (kg <= qg && kg < S) tmax = fmaxf(tmax, Ssh[j*SST + i]);
            }
            if (tmax == -INFINITY) {
                corr[i] = 1.f;
                for (int j = 0; j < BN; ++j) Psh[j*PST + i] = __float2bfloat16(0.f);
            } else {
                float m_new = fmaxf(mi, tmax);
                float c = (mi == -INFINITY) ? 0.f : __expf(mi - m_new);
                float rs = 0.f;
                for (int j = 0; j < BN; ++j) {
                    int kg = k0 + j;
                    float p = 0.f;
                    if (kg <= qg && kg < S) p = __expf(Ssh[j*SST + i] - m_new);
                    rs += p;
                    Psh[j*PST + i] = __float2bfloat16(p);
                }
                l_arr[i] = l_arr[i] * c + rs;
                m_arr[i] = m_new;
                corr[i] = c;
                if (c != 1.f) {
                    #pragma unroll 8
                    for (int d = 0; d < D; ++d) Osh[d*OST + i] *= c;
                }
            }
        }
        __syncthreads();

        // ---- O += P @ V ----
        {
            wmma::fragment<wmma::matrix_a, 16,16,16, __nv_bfloat16, wmma::col_major> af;
            wmma::fragment<wmma::matrix_b, 16,16,16, __nv_bfloat16, wmma::row_major> bf;
            wmma::fragment<wmma::accumulator, 16,16,16, float> acc;
            for (int ct = 0; ct < D/16; ++ct) {
                wmma::load_matrix_sync(acc, Osh + R + (ct*16)*OST, OST, wmma::mem_col_major);
                #pragma unroll
                for (int kt = 0; kt < BN/16; ++kt) {
                    wmma::load_matrix_sync(af, Psh + R + (kt*16)*PST, PST);
                    wmma::load_matrix_sync(bf, Vsh + (kt*16)*D + ct*16, D);
                    wmma::mma_sync(acc, af, bf, acc);
                }
                wmma::store_matrix_sync(Osh + R + (ct*16)*OST, acc, OST, wmma::mem_col_major);
            }
        }
    }

    __syncthreads();
    if (tid < BM) {
        int i = tid, s = q0 + i;
        if (s < S) {
            float li = l_arr[i];
            float inv = (li > 0.f) ? (1.f / li) : 0.f;
            __nv_bfloat16* out = Ob + (int64_t)s * D;
            for (int d = 0; d < D; ++d) out[d] = __float2bfloat16(Osh[d*OST + i] * inv);
            LSEb[s] = m_arr[i] + logf(li);
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

    size_t smem = (size_t)BM * D * 2
                + (size_t)BN * D * 2
                + (size_t)BN * D * 2
                + (size_t)BN * SST * 4
                + (size_t)BN * PST * 2
                + (size_t)D  * OST * 4
                + (size_t)BM * 4 * 3;

    cudaFuncSetAttribute(attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    attn_kernel<<<grid, block, smem, stream>>>(Qp, Kp, Vp, Op, LSEp, Bb, Hh, Ss, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_d128_causal::run);

}  // namespace mha_d128_causal