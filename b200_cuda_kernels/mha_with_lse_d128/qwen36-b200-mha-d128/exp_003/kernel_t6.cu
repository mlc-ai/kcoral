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

constexpr int BM = 64;       // query rows per block
constexpr int BK = 16;       // key rows per chunk  
constexpr int NT = 128;      // threads per block
constexpr int TPQR = NT / BM;  // threads per row = 2
constexpr int DIM = 128;     // head dimension
constexpr int EPT = DIM / TPQR; // elements per thread = 64

// Shared memory sizes (in bf16 elements)
constexpr int SZ_Q = BM * DIM;   // 8192
constexpr int SZ_K = BK * DIM;   // 2048
constexpr int SZ_V = BK * DIM;   // 2048
constexpr int SZ_TOTAL = SZ_Q + SZ_K + SZ_V; // 12288 ~= 24KB

__global__ void attn_fwd(
    const __nv_bfloat16* __restrict__ Q_g,
    const __nv_bfloat16* __restrict__ K_g,
    const __nv_bfloat16* __restrict__ V_g,
    __nv_bfloat16* __restrict__ O_g,
    float* __restrict__ LSE_g,
    int S)
{
    extern __shared__ __nv_bfloat16 shm[];

    __nv_bfloat16* sQ = shm;                       // offset 0
    __nv_bfloat16* sK = shm + SZ_Q;               // offset 8192
    __nv_bfloat16* sV = shm + SZ_Q + SZ_K;        // offset 10240

    int bh_idx  = blockIdx.x;           // [0 .. B*H)
    int qt_idx  = blockIdx.y;           // query tile index
    int thr_id  = threadIdx.x;          // [0 .. NT)

    int local_r = thr_id / TPQR;        // local row [0 .. BM)
    int local_l = thr_id % TPQR;        // lane [0 .. TPQR)

    int g_qr = qt_idx * BM + local_r;   // global query row
    bool valid_q = (g_qr < S);

    uint64_t bh_base = static_cast<uint64_t>(bh_idx) * S * DIM;

    float inv_sqrt_D = rsqrtf(static_cast<float>(DIM));

    // ---- Load Q tile cooperatively ----
    for (int i = thr_id; i < SZ_Q; i += NT) {
        int sr = i / DIM;
        int sc = i % DIM;
        int gr = qt_idx * BM + sr;
        if (gr < S)
            sQ[i] = Q_g[bh_base + static_cast<uint64_t>(gr) * DIM + sc];
        else
            sQ[i] = __float2bfloat16(0.0f);
    }
    __syncthreads();

    // Per-thread registers for accumulator (FP32)
    float acc[EPT];
    for (int j = 0; j < EPT; j++) acc[j] = 0.0f;

    float mf = -FLT_MAX;  // running max
    float sf = 0.0f;      // running sum-exp

    int nchunks = (S + BK - 1) / BK;
    int cstart = local_l * EPT;   // starting column index for this thread

    for (int ck = 0; ck < nchunks; ck++) {
        int ks = ck * BK;                 // global key start
        int ke = min(BK, S - ks);         // number of VALID keys in this chunk

        // ---- Load K tile ----
        for (int i = thr_id; i < SZ_K; i += NT) {
            int sr = i / DIM;
            int sc = i % DIM;
            int gr = ks + sr;
            if (gr < S)
                sK[i] = K_g[bh_base + static_cast<uint64_t>(gr) * DIM + sc];
            else
                sK[i] = __float2bfloat16(0.0f);
        }

        // ---- Load V tile ----
        for (int i = thr_id; i < SZ_V; i += NT) {
            int sr = i / DIM;
            int sc = i % DIM;
            int gr = ks + sr;
            if (gr < S)
                sV[i] = V_g[bh_base + static_cast<uint64_t>(gr) * DIM + sc];
            else
                sV[i] = __float2bfloat16(0.0f);
        }
        __syncthreads();

        // ---- Compute scores: Q[row] @ K[key]^T ----
        float max_sc = -FLT_MAX;
        float attn[BK];

        if (valid_q) {
            const __nv_bfloat162* qp =
                reinterpret_cast<const __nv_bfloat162*>(&sQ[local_r * DIM]);

            // ONLY compute scores for VALID keys
            for (int kk = 0; kk < ke; kk++) {
                float val = 0.0f;
                const __nv_bfloat162* kp =
                    reinterpret_cast<const __nv_bfloat162*>(&sK[kk * DIM]);
                for (int dd = 0; dd < DIM / 2; dd++) {
                    val += __low2float(qp[dd])  * __low2float(kp[dd]);
                    val += __high2float(qp[dd]) * __high2float(kp[dd]);
                }
                val *= inv_sqrt_D;
                attn[kk] = val;
                if (val > max_sc) max_sc = val;
            }
        }
        // Mark INVALID key positions as -FLT_MAX so exp() yields 0
        for (int kk = (valid_q ? ke : 0); kk < BK; kk++) {
            attn[kk] = -FLT_MAX;
        }
        if (!valid_q) {
            max_sc = -FLT_MAX;
        }

        // ---- Online softmax update ----
        float old_mf = mf;
        mf = (mf > max_sc) ? mf : max_sc;
        float f = expf(old_mf - mf);

        for (int j = 0; j < EPT; j++) acc[j] *= f;
        sf *= f;

        // Accumulate using VALID keys only — though attn[] already has -FLT_MAX
        // for invalid positions, exp(-FLT_MAX) ≈ 0, so both paths work correctly.
        for (int kk = 0; kk < ke; kk++) {
            float p = expf(attn[kk] - mf);
            sf += p;
            const __nv_bfloat16* ptr_v = &sV[kk * DIM + cstart];
            for (int j = 0; j < EPT; j++) {
                acc[j] += p * __bfloat162float(ptr_v[j]);
            }
        }
    }

    // ---- Normalize and store ----
    float out_scale = (sf > 0.0f) ? (1.0f / sf) : 0.0f;

    if (valid_q) {
        for (int j = 0; j < EPT; j++) {
            int gc = cstart + j;
            if (gc < DIM) {
                float fv = acc[j] * out_scale;
                O_g[bh_base + static_cast<uint64_t>(g_qr) * DIM + gc] =
                    __float2bfloat16(fv);
            }
        }
        if (local_l == 0) {
            LSE_g[bh_idx * S + g_qr] =
                (sf > 0.0f) ? (mf + logf(sf)) : (-FLT_MAX);
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

    const __nv_bfloat16* Q = static_cast<const __nv_bfloat16*>(Q_tv.data_ptr());
    const __nv_bfloat16* K = static_cast<const __nv_bfloat16*>(K_tv.data_ptr());
    const __nv_bfloat16* V = static_cast<const __nv_bfloat16*>(V_tv.data_ptr());
    __nv_bfloat16*        O = static_cast<__nv_bfloat16*>(O_tv.data_ptr());
    float*               LSE = static_cast<float*>(LSE_tv.data_ptr());

    dim3 grid(B * H, (S + BM - 1) / BM);
    dim3 block(NT);
    size_t smem = SZ_TOTAL * sizeof(__nv_bfloat16);

    cudaStream_t str = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q_tv.device().device_type, Q_tv.device().device_id));

    attn_fwd<<<grid, block, smem, str>>>(Q, K, V, O, LSE, S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(str));
}

}  // namespace mha_d128

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_d128::run);