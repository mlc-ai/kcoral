#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cfloat>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_impl {

static constexpr int D = 128;
static constexpr int BK = 64;
static constexpr int MROWS = 4;
static constexpr int THD = 128;
static constexpr int NW = 4;
static constexpr unsigned FULL = 0xffffffffu;

__global__ void mha_k(const __nv_bfloat16* Q, const __nv_bfloat16* K,
                      const __nv_bfloat16* V, __nv_bfloat16* O,
                      float* LSE, int B, int H, int S, float sc) {
    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* sk = smem;
    __nv_bfloat16* sv = sk + BK * D;
    float* ws = reinterpret_cast<float*>(sv + BK * D);
    float* ss = ws + NW * BK * MROWS;

    int bh = blockIdx.x;
    int qs = blockIdx.y * MROWS;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lid = tid & 31;

    uint64_t boff = (uint64_t)bh * S * D;
    const __nv_bfloat16* Qp = Q + boff;
    const __nv_bfloat16* Kp = K + boff;
    const __nv_bfloat16* Vp = V + boff;
    __nv_bfloat16* Op = O + boff;
    float* Lp = LSE + (uint64_t)bh * S;

    float qv[MROWS], mx[MROWS], dn[MROWS], oa[MROWS];
    #pragma unroll
    for (int i = 0; i < MROWS; ++i) {
        int qi = qs + i;
        qv[i] = (qi < S) ? __bfloat162float(Qp[(uint64_t)qi * D + tid]) : 0.f;
        mx[i] = -FLT_MAX;
        dn[i] = 0.f;
        oa[i] = 0.f;
    }

    #pragma unroll
    for (int ks = 0; ks < S; ks += BK) {
        int kk = min(BK, S - ks);

        // Clear workspace
        for (int i = tid; i < NW * BK * MROWS; i += THD) ws[i] = 0.f;
        __syncthreads();

        // Load K and V tile cooperatively
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            int ki = ks + k;
            if (ki < S) {
                uint64_t ko = (uint64_t)ki * D;
                sk[k * D + tid] = Kp[ko + tid];
                sv[k * D + tid] = Vp[ko + tid];
            }
        }
        __syncthreads();

        // Warp-level dot product reduction
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float kvf = __bfloat162float(sk[k * D + tid]);
            #pragma unroll
            for (int i = 0; i < MROWS; ++i) {
                float p = qv[i] * kvf * sc;
                for (int s = 16; s > 0; s >>= 1)
                    p += __shfl_down_sync(FULL, p, s);
                if (lid == 0) ws[wid * BK * MROWS + k * MROWS + i] = p;
            }
        }
        __syncthreads();

        // Merge warp reductions into full scores
        for (int i = tid; i < BK * MROWS; i += THD) {
            float v = 0.f;
            #pragma unroll
            for (int w = 0; w < NW; ++w) v += ws[w * BK * MROWS + i];
            ss[i] = v;
        }
        __syncthreads();

        // Online softmax + weighted V accumulation
        #pragma unroll
        for (int i = 0; i < MROWS; ++i) {
            int qi = qs + i;
            if (qi >= S) continue;

            float omx = mx[i], odn = dn[i], oacc = oa[i];
            float tmx = -FLT_MAX;
            #pragma unroll
            for (int k = 0; k < kk; ++k) {
                float s = ss[k * MROWS + i];
                if (s > tmx) tmx = s;
            }

            float nm = max(omx, tmx);
            float af = expf(omx - nm);
            odn *= af;
            oacc *= af;

            #pragma unroll
            for (int k = 0; k < kk; ++k) {
                float p = expf(ss[k * MROWS + i] - nm);
                odn += p;
                oacc += p * __bfloat162float(sv[k * D + tid]);
            }
            mx[i] = nm;
            dn[i] = odn;
            oa[i] = oacc;
        }
        __syncthreads();
    }

    // Store output
    #pragma unroll
    for (int i = 0; i < MROWS; ++i) {
        int qi = qs + i;
        if (qi < S)
            Op[(uint64_t)qi * D + tid] = __float2bfloat16(oa[i] / dn[i]);
    }
    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < MROWS; ++i) {
            int qi = qs + i;
            if (qi < S)
                Lp[qi] = mx[i] + logf(dn[i]);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int64_t B = Q.size(0), H = Q.size(1), S = Q.size(2);
    float sc = rsqrtf(static_cast<float>(Q.size(3)));

    dim3 g(B * H, (S + MROWS - 1) / MROWS);
    dim3 b(THD);
    size_t smem_bytes = sizeof(__nv_bfloat16) * BK * D
                       + sizeof(__nv_bfloat16) * BK * D
                       + sizeof(float) * NW * BK * MROWS
                       + sizeof(float) * BK * MROWS;

    cudaStream_t st = (cudaStream_t)TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id);

    mha_k<<<g, b, smem_bytes, st>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<__nv_bfloat16*>(O.data_ptr()),
        static_cast<float*>(LSE.data_ptr()),
        (int)B, (int)H, (int)S, sc);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(st));
}

} // namespace mha_impl

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_impl::run);