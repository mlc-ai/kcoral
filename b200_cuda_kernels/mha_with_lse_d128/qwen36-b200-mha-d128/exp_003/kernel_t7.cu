#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>
#include <cfloat>
#include <cmath>
#include <cassert>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                         \
    cudaError_t _e = (call);                                          \
    if (_e != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA err %s at %s:%d\n",                   \
                cudaGetErrorString(_e), __FILE__, __LINE__);          \
        exit(1);                                                      \
    }                                                                 \
} while(0)

namespace mha_d128 {

constexpr int BM   = 64;       // query rows per block
constexpr int BK   = 16;       // key rows per chunk
constexpr int NT   = 128;      // threads per block (4 warps)
constexpr int TPQR = NT / BM;  // threads per output row = 2
constexpr int DIM  = 128;      // head dimension
constexpr int EPT  = DIM / TPQR; // elements per thread = 64

// Shared memory allocation (bf16 elements)
static constexpr int SZ_Q = BM * DIM;   // 8192
static constexpr int SZ_K = BK * DIM;   // 2048
static constexpr int SZ_V = BK * DIM;   // 2048
static constexpr int SZ_T = SZ_Q + SZ_K + SZ_V;  // 12288

__global__ void attn_kernel(
    const __nv_bfloat16* __restrict__ Qgm,
    const __nv_bfloat16* __restrict__ Kgm,
    const __nv_bfloat16* __restrict__ Vgm,
    __nv_bfloat16* __restrict__ Ogm,
    float* __restrict__ LSEgm,
    int B, int H, int S)
{
    extern __shared__ __nv_bfloat16 shm[];

    __nv_bfloat16* sQ = shm + 0;
    __nv_bfloat16* sK = shm + SZ_Q;
    __nv_bfloat16* sV = shm + SZ_Q + SZ_K;

    int bih  = blockIdx.x;            // batch-head flat index [0..B*H)
    int qi   = blockIdx.y;            // query tile index
    int tid  = threadIdx.x;           // [0..NT)
    int rid  = tid / TPQR;            // local row  [0..BM)
    int lane = tid % TPQR;            // lane       [0..TPQR)

    int qglob = qi * BM + rid;         // global query row
    bool qok  = (qglob < S);           // query row validity

    uint64_t base = static_cast<uint64_t>(bih) * S * DIM;
    float invsd  = rsqrtf(static_cast<float>(DIM));

    // ---- Load Q tile ----
    for (int i = tid; i < SZ_Q; i += NT) {
        int r = i / DIM, c = i % DIM;
        int gr = qi * BM + r;
        sQ[i] = (gr < S)
            ? Qgm[base + static_cast<uint64_t>(gr) * DIM + c]
            : __float2bfloat16(0.0f);
    }
    __syncthreads();

    // Per-thread FP32 accumulator
    float oacc[EPT];
    for (int e = 0; e < EPT; ++e) oacc[e] = 0.0f;

    // Online softmax state — use a safe large-negative sentinel
    float m_val = -1e10f;   // running row max
    float l_val = 0.0f;     // running sum-exp

    int nc = (S + BK - 1) / BK;  // total key chunks
    int cs = lane * EPT;          // column start for this lane

    for (int ki = 0; ki < nc; ++ki) {
        int ks = ki * BK;             // global key start
        int keff = min(BK, S - ks);   // valid keys in chunk

        // ---- Load K tile ----
        for (int i = tid; i < SZ_K; i += NT) {
            int r = i / DIM, c = i % DIM;
            int gr = ks + r;
            sK[i] = (gr < S)
                ? Kgm[base + static_cast<uint64_t>(gr) * DIM + c]
                : __float2bfloat16(0.0f);
        }

        // ---- Load V tile ----
        for (int i = tid; i < SZ_V; i += NT) {
            int r = i / DIM, c = i % DIM;
            int gr = ks + r;
            sV[i] = (gr < S)
                ? Vgm[base + static_cast<uint64_t>(gr) * DIM + c]
                : __float2bfloat16(0.0f);
        }
        __syncthreads();

        // ---- Dot-products: Q[row] @ K[kk]^T ----
        float smax = -1e10f;
        float sc[BK];

        if (qok) {
            const __nv_bfloat162* qp =
                reinterpret_cast<const __nv_bfloat162*>(&sQ[rid * DIM]);
            for (int kk = 0; kk < keff; ++kk) {
                float d = 0.0f;
                const __nv_bfloat162* kp =
                    reinterpret_cast<const __nv_bfloat162*>(&sK[kk * DIM]);
                for (int dd = 0; dd < DIM / 2; ++dd) {
                    d += __low2float(qp[dd])  * __low2float(kp[dd]);
                    d += __high2float(qp[dd]) * __high2float(kp[dd]);
                }
                sc[kk] = d * invsd;
                if (sc[kk] > smax) smax = sc[kk];
            }
        }

        // ---- Softmax update ----
        float m_old = m_val;
        m_val = (m_val > smax) ? m_val : smax;
        float alpha = expf(m_old - m_val);

        for (int e = 0; e < EPT; ++e) oacc[e] *= alpha;
        l_val *= alpha;

        for (int kk = 0; kk < keff; ++kk) {
            float w = expf(sc[kk] - m_val);
            l_val += w;
            const __nv_bfloat16* vp = &sV[kk * DIM + cs];
            for (int e = 0; e < EPT; ++e) {
                oacc[e] += w * __bfloat162float(vp[e]);
            }
        }
    }

    // ---- Normalize & write O (bf16) ----
    float nscale = (l_val > 0.0f) ? (1.0f / l_val) : 0.0f;

    if (qok) {
        for (int e = 0; e < EPT; ++e) {
            int gc = cs + e;
            if (gc < DIM) {
                Ogm[base + static_cast<uint64_t>(qglob) * DIM + gc] =
                    __float2bfloat16(oacc[e] * nscale);
            }
        }
        // ---- Write LSE once per row ----
        if (lane == 0) {
            LSEgm[bih * S + qglob] = (l_val > 0.0f)
                ? (m_val + logf(l_val))
                : (-1e10f);
        }
    }
}

void run(tvm::ffi::TensorView Q_tv, tvm::ffi::TensorView K_tv,
         tvm::ffi::TensorView V_tv, tvm::ffi::TensorView O_tv,
         tvm::ffi::TensorView LSE_tv)
{
    CUDA_CHECK(cudaSetDevice(Q_tv.device().device_id));

    constexpr int64_t B = 4, H = 48;
    int64_t S = Q_tv.size(2);

    const __nv_bfloat16* Q  = static_cast<const __nv_bfloat16*>(Q_tv.data_ptr());
    const __nv_bfloat16* K  = static_cast<const __nv_bfloat16*>(K_tv.data_ptr());
    const __nv_bfloat16* V  = static_cast<const __nv_bfloat16*>(V_tv.data_ptr());
    __nv_bfloat16*         O = static_cast<__nv_bfloat16*>(O_tv.data_ptr());
    float*               LSE = static_cast<float*>(LSE_tv.data_ptr());

    dim3 grid(B * H, (S + BM - 1) / BM);
    dim3 block(NT);
    size_t smem = SZ_T * sizeof(__nv_bfloat16);

    cudaStream_t str = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q_tv.device().device_type, Q_tv.device().device_id));

    attn_kernel<<<grid, block, smem, str>>>(Q, K, V, O, LSE, B, H, S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(str));
}

}  // namespace mha_d128

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_d128::run);