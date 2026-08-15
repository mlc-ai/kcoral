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
static constexpr int BM = 4;
static constexpr int BK = 64;
static constexpr int NT = 128;
static constexpr unsigned MASK = 0xffffffffu;

__global__ void mha_kernel(const __nv_bfloat16* Q, const __nv_bfloat16* K,
                           const __nv_bfloat16* V, __nv_bfloat16* O,
                           float* LSE, int B, int H, int S, float scale) {
    
    __shared__ __nv_bfloat16 sK[BK][D];
    __shared__ __nv_bfloat16 sV[BK][D];
    __shared__ float ws[4 * BK * BM];

    int bh = blockIdx.x;
    int qs = blockIdx.y * BM;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lid = tid & 31;

    uint64_t base_BH = static_cast<uint64_t>(bh) * S * D;
    const __nv_bfloat16* Qp = Q + base_BH;
    const __nv_bfloat16* Kp = K + base_BH;
    const __nv_bfloat16* Vp = V + base_BH;
    __nv_bfloat16* Op = O + base_BH;
    float* LP = LSE + static_cast<uint64_t>(bh) * S;

    float q_reg[BM], mx[BM], dn[BM], oa[BM];
    for (int i = 0; i < BM; ++i) {
        int qi = qs + i;
        q_reg[i] = (qi < S) ? __bfloat162float(Qp[static_cast<uint64_t>(qi)*D + tid]) : 0.f;
        mx[i] = -FLT_MAX;
        dn[i] = 0.f;
        oa[i] = 0.f;
    }

    for (int ks = 0; ks < S; ks += BK) {
        int kk = min(BK, S - ks);

        // Clear warp workspace
        for (int i = tid; i < 4 * BK * BM; i += NT) ws[i] = 0.f;
        __syncthreads();

        // Cooperative load K and V
        for (int r = 0; r < BK; ++r) {
            int ki = ks + r;
            if (ki < S) {
                uint64_t koff = static_cast<uint64_t>(ki) * D;
                sK[r][tid] = Kp[koff + tid];
                sV[r][tid] = Vp[koff + tid];
            }
        }
        __syncthreads();

        // Compute partial dots + warp reduction
        for (int r = 0; r < KK; ++r) {
            float kval = __bfloat162float(sK[r][tid]);
            for (int i = 0; i < BM; ++i) {
                float s = q_reg[i] * kval * scale;
                for (int d = 16; d > 0; d >>= 1)
                    s += __shfl_down_sync(MASK, s, d);
                if (lid == 0) ws[wid*BK*BM + r*BM + i] = s;
            }
        }
        __syncthreads();

        // Merge warp partials into complete scores
        float ss[KK][BM];
        for (int i = tid; i < KK * BM; i += NT) {
            float v = 0.f;
            for (int w = 0; w < 4; ++w) v += ws[w * BK * BM + i];
            int r = i / BM;
            int c = i % BM;
            ss[r][c] = v;
        }
        __threadfence_block();

        // Online softmax update + weighted V accumulate
        for (int i = 0; i < BM; ++i) {
            int qi = qs + i;
            if (qi >= S) continue;
            
            float omx = mx[i], odn = dn[i], oacc = oa[i];
            float tmx = -FLT_MAX;
            for (int r = 0; r < kk; ++r) {
                float s = ss[r][i];
                if (s > tmx) tmx = s;
            }
            
            float nm = fmaxf(omx, tmx);
            float af = expf(omx - nm);
            odn *= af;
            oacc *= af;
            
            for (int r = 0; r < kk; ++r) {
                float p = expf(ss[r][i] - nm);
                odn += p;
                oacc += p * __bfloat162float(sV[r][tid]);
            }
            mx[i] = nm;
            dn[i] = odn;
            oa[i] = oacc;
        }
        __syncthreads();
    }

    // Store normalized output
    for (int i = 0; i < BM; ++i) {
        int qi = qs + i;
        if (qi < S) {
            float inv_d = (dn[i] > 1e-30f) ? (1.f / dn[i]) : 0.f;
            Op[static_cast<uint64_t>(qi)*D + tid] = __float2bfloat16(oa[i] * inv_d);
        }
    }

    // Store LSE
    if (tid == 0) {
        for (int i = 0; i < BM; ++i) {
            int qi = qs + i;
            if (qi < S && dn[i] > 1e-30f)
                LP[qi] = mx[i] + logf(dn[i]);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0), H = Q.size(1), S = Q.size(2);
    float scale = rsqrtf(static_cast<float>(Q.size(3)));

    dim3 grid(B * H, (S + BM - 1) / BM);
    dim3 block(NT);

    cudaStream_t st = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    mha_kernel<<<grid, block, 0, st>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<__nv_bfloat16*>(O.data_ptr()),
        static_cast<float*>(LSE.data_ptr()),
        static_cast<int>(B), static_cast<int>(H), static_cast<int>(S), scale);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(st));
}

} // namespace mha_impl

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_impl::run);