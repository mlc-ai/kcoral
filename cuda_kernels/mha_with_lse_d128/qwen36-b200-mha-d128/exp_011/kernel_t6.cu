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
static constexpr int NW = THD / 32;
static constexpr unsigned FULL = 0xffffffffu;

__global__ void mha_k(const __nv_bfloat16* Q, const __nv_bfloat16* K,
                      const __nv_bfloat16* V, __nv_bfloat16* O,
                      float* LSE, int B, int H, int S, float sc) {
    extern __shared__ char smem[];
    
    __nv_bfloat16* sk    = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sv    = sk + BK * D;
    float*    ws         = reinterpret_cast<float*>(sv + BK * D);
    float*    ss         = ws + NW * BK * MROWS;

    int bh     = blockIdx.x;
    int qs     = blockIdx.y * MROWS;
    int tid    = threadIdx.x;
    int wid    = tid >> 5;
    int lid    = tid & 31;

    uint64_t boff_BH = static_cast<uint64_t>(bh) * S * D;
    const __nv_bfloat16* Qp   = Q  + boff_BH;
    const __nv_bfloat16* Kp   = K  + boff_BH;
    const __nv_bfloat16* Vp   = V  + boff_BH;
    __nv_bfloat16*          Op = O  + boff_BH;
    float*                   LP = LSE + static_cast<uint64_t>(bh) * S;

    float qv[MROWS], mx[MROWS], dn[MROWS], oa[MROWS];
    for (int i = 0; i < MROWS; ++i) {
        int qi = qs + i;
        qv[i] = (qi < S) ? __bfloat162float(Qp[static_cast<uint64_t>(qi) * D + tid]) : 0.f;
        mx[i] = -FLT_MAX;
        dn[i] = 0.f;
        oa[i] = 0.f;
    }

    for (int ks = 0; ks < S; ks += BK) {
        int kk = min(BK, S - ks);

        // Clear workspace
        for (int i = tid; i < NW * BK * MROWS; i += THD) ws[i] = 0.f;
        __syncthreads();

        // Cooperative K,V load (each thread loads 1 element per row)
        for (int r = 0; r < BK; ++r) {
            int idx = ks + r;
            if (idx < S) {
                uint64_t off = static_cast<uint64_t>(idx) * D;
                sk[r * D + tid] = Kp[off + tid];
                sv[r * D + tid] = Vp[off + tid];
            }
        }
        __syncthreads();

        // Dot products with warp shuffle reduction -> shared
        for (int r = 0; r < BK; ++r) {
            float kval = __bfloat162float(sk[r * D + tid]);
            for (int m = 0; m < MROWS; ++m) {
                float s = qv[m] * kval * sc;
                for (int d = 16; d > 0; d >>= 1)
                    s += __shfl_down_sync(FULL, s, d);
                if (lid == 0) ws[wid * BK * MROWS + r * MROWS + m] = s;
            }
        }
        __syncthreads();

        // Cross-warp merge
        for (int i = tid; i < BK * MROWS; i += THD) {
            float v = 0.f;
            for (int w = 0; w < NW; ++w) v += ws[w * BK * MROWS + i];
            ss[i] = v;
        }
        __syncthreads();

        // Online softmax + V accumulation
        for (int m = 0; m < MROWS; ++m) {
            int qi = qs + m;
            if (qi >= S) continue;

            float omx = mx[m], odn = dn[m], oacc = oa[m];
            float tmx = -FLT_MAX;
            for (int r = 0; r < kk; ++r) {
                float s = ss[r * MROWS + m];
                if (s > tmx) tmx = s;
            }
            float nm = fmaxf(omx, tmx);
            float af = expf(omx - nm);
            odn *= af;
            oacc *= af;
            for (int r = 0; r < kk; ++r) {
                float p = expf(ss[r * MROWS + m] - nm);
                odn += p;
                oacc += p * __bfloat162float(sv[r * D + tid]);
            }
            mx[m] = nm;
            dn[m] = odn;
            oa[m] = oacc;
        }
        __syncthreads();
    }

    // Store O
    for (int m = 0; m < MROWS; ++m) {
        int qi = qs + m;
        if (qi < S)
            Op[static_cast<uint64_t>(qi) * D + tid] = __float2bfloat16(oa[m] / dn[m]);
    }

    // Store LSE
    if (tid == 0) {
        for (int m = 0; m < MROWS; ++m) {
            int qi = qs + m;
            if (qi < S)
                LP[qi] = mx[m] + logf(dn[m]);
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
    size_t smem = sizeof(__nv_bfloat16) * BK * D
                + sizeof(__nv_bfloat16) * BK * D
                + sizeof(float) * NW * BK * MROWS
                + sizeof(float) * BK * MROWS;

    cudaStream_t st = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    mha_k<<<g, b, smem, st>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<__nv_bfloat16*>(O.data_ptr()),
        static_cast<float*>(LSE.data_ptr()),
        static_cast<int>(B), static_cast<int>(H), static_cast<int>(S), sc);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(st));
}

} // namespace mha_impl

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_impl::run);