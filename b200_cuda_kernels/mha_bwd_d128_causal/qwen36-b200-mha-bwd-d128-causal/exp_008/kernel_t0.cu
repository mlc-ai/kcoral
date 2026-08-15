#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <cmath>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                                \
    cudaError_t _e = (call);                                                 \
    if (_e != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                        \
                __FILE__, __LINE__, cudaGetErrorString(_e));                 \
        exit(1);                                                             \
    }                                                                        \
} while(0)

namespace mha_bwd_impl {

// Tiling parameters
static constexpr int BM   = 16;        // query tile rows
static constexpr int BN   = 16;        // key   tile rows
static constexpr int NT   = 128;       // threads / block
static constexpr int TPR  = NT / BM;   // threads per query row = 8
static constexpr int KPRT = BN / TPR;  // key-cols per thread   = 2
static constexpr int HD   = 128;       // head dim (matches task d=128)

// ======================================================================
// Kernel: fused tiled MHA backward (causal, bf16, fp32 accumulator)
//
// Thread mapping (128 threads / block):
//   qr  = tid / 8     ->  query row inside tile  (0..15)
//   lane= tid % 8     ->  lane inside row-group   (0..7)
//   kc0 = lane * 2    ->  key-col base  (0,2,4,...,14)
//   Each thread owns 1 query row + 2 key columns per tile-pair.
//
// dQ  ->  accumulated in per-thread register array  dq[HD]
// dK,dV->  accumulated via atomicAdd into shared-mem buf (padded)
//        ->  flushed to global after every key-tile iteration
// ======================================================================
__global__ void mha_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float*         __restrict__ L,       // [B,H,S] log-sum-exp
    __nv_bfloat16*       __restrict__ dQ_out,
    __nv_bfloat16*       __restrict__ dK_out,
    __nv_bfloat16*       __restrict__ dV_out,
    int B, int H, int S, int d_head,
    float attn_scale, bool causal)
{
    // ---- block / thread identity ----
    int bh  = blockIdx.x;
    if (bh >= B * H) return;
    int b   = bh / H;
    int h   = bh % H;
    int tid = threadIdx.x;

    int qr  = tid / TPR;                   // 0..15
    int kc0 = (tid % TPR) * KPRT;          // 0,2,4,..,14

    // ---- global strides ----
    int64_t ss = (int64_t)S * d_head;
    int64_t bs = ss * H;

    // ---- base pointers for this (b,h) ----
    const __nv_bfloat16* Qp  = Q  + b*bs + h*ss;
    const __nv_bfloat16* Kp  = K  + b*bs + h*ss;
    const __nv_bfloat16* Vp  = V  + b*bs + h*ss;
    const __nv_bfloat16* Op  = O  + b*bs + h*ss;
    const __nv_bfloat16* dOp = dO + b*bs + h*ss;
    const float*         Lp  = L  + b*H*S + h*S;
    __nv_bfloat16* dQp       = dQ_out + b*bs + h*ss;
    __nv_bfloat16* dKp       = dK_out + b*bs + h*ss;
    __nv_bfloat16* dVp       = dV_out + b*bs + h*ss;

    // ---- shared memory ---------------------------------------------------
    // sdK / sdV padded by +1 to destroy 128-wide bank aliasing
    __shared__ __nv_bfloat16 sQ  [BM][HD];
    __shared__ __nv_bfloat16 sK  [BN][HD];
    __shared__ __nv_bfloat16 sV  [BN][HD];
    __shared__ __nv_bfloat16 sdO [BM][HD];
    __shared__ float         sD  [BM];
    __shared__ float         sdK [BN][HD+1];   // padded
    __shared__ float         sdV [BN][HD+1];   // padded

    // ======================================================================
    // OUTER LOOP – iterate over query tiles
    // ======================================================================
    for (int qt = 0; qt < S; qt += BM) {
        int qt_end = qt + BM;
        if (qt_end > S) qt_end = S;
        int BM_e = qt_end - qt;

        // ---- Phase 1: load Q / dO tiles, compute D ----
        if (qr < BM_e) {
            // cooperative load of Q and dO rows
            for (int i = tid; i < BM_e * d_head; i += NT) {
                int r = i / d_head;
                int c = i % d_head;
                sQ[r][c]  = Qp  [(qt+r)*d_head + c];
                sdO[r][c] = dOp [(qt+r)*d_head + c];
            }

            // D[qr] = row_sum(dO[qr,:] * O[qr,:])
            int qg   = qt + qr;
            float dsum = 0.0f;
            for (int dv = 0; dv < d_head; ++dv) {
                dsum += __bfloat162float(dOp[qg*d_head + dv])
                      * __bfloat162float(Op [qg*d_head + dv]);
            }
            sD[qr] = dsum;
        }
        __syncthreads();

        // ---- register accumulator for dQ (one row per thread) ----
        float dq[HD];
        for (int i = 0; i < HD; ++i) dq[i] = 0.0f;

        // ---- zero shared dK / dV buffers ----
        for (int i = tid; i < BN * HD; i += NT) {
            sdK[i / HD][i % HD] = 0.0f;
            sdV[i / HD][i % HD] = 0.0f;
        }
        __syncthreads();

        // ==================================================================
        // INNER LOOP – iterate over key tiles
        // ==================================================================
        for (int kt = 0; kt < S; kt += BN) {
            int kt_end = kt + BN;
            if (kt_end > S) kt_end = S;
            int BN_e = kt_end - kt;

            // --- load K / V tiles ---
            for (int i = tid; i < BN_e * d_head; i += NT) {
                int r = i / d_head;
                int c = i % d_head;
                sK[r][c] = Kp[(kt+r)*d_head + c];
                sV[r][c] = Vp[(kt+r)*d_head + c];
            }
            __syncthreads();

            // --- compute S_partial and dP_partial --------------------------
            float sval[KPRT];
            float dpvl[KPRT];

            #pragma unroll
            for (int kc = 0; kc < KPRT; ++kc) {
                int kl = kc0 + kc;
                float sv = 0.0f, dv_ = 0.0f;
                if (qr < BM_e && kl < BN_e) {
                    for (int dv = 0; dv < d_head; ++dv) {
                        float q = __bfloat162float(sQ [qr][dv]);
                        float k = __bfloat162float(sK [kl][dv]);
                        float v = __bfloat162float(sV [kl][dv]);
                        float o = __bfloat162float(sdO[qr][dv]);
                        sv += q * k * attn_scale;
                        dv_ += o * v;
                    }
                } else {
                    sv = -1e20f;
                    dv_ = 0.0f;
                }
                sval[kc] = sv;
                dpvl[kc] = dv_;
            }

            // --- causal mask  ->  P  ->  dS  ->  accumulate dQ/dK/dV ------
            int   qg = qt + qr;
            float lv = Lp[qg];
            float dcorr = sD[qr];        // D correction term

            #pragma unroll
            for (int kc = 0; kc < KPRT; ++kc) {
                int kl = kc0 + kc;
                if (kl >= BN_e || qr >= BM_e) continue;

                int kg = kt + kl;
                if (kg >= S) continue;

                float sv = sval[kc];
                if (causal && kg > qg) sv = -1e20f;

                float pv = expf(sv - lv);
                float ds = pv * (dpvl[kc] - dcorr);

                // dQ  accumulation  (register)
                for (int dv = 0; dv < HD; ++dv) {
                    dq[dv] += ds * __bfloat162float(sK[kl][dv]) * attn_scale;
                }

                // dK accumulation  (shared-mem atomic, chain-rule *attn_scale)
                for (int dv = 0; dv < HD; ++dv) {
                    atomicAdd(&sdK[kl][dv],
                              ds * __bfloat162float(sQ[qr][dv]) * attn_scale);
                }

                // dV accumulation  (shared-mem atomic, no attn_scale)
                for (int dv = 0; dv < HD; ++dv) {
                    atomicAdd(&sdV[kl][dv],
                              pv * __bfloat162float(sdO[qr][dv]));
                }
            }

            __syncthreads();

            // --- flush dK / dV to global memory ---
            for (int i = tid; i < BN_e * d_head; i += NT) {
                int r  = i / d_head;
                int c  = i % d_head;
                int kg = kt + r;
                if (kg < S) {
                    dKp[kg*d_head + c] = __float2bfloat16(sdK[r][c]);
                    dVp[kg*d_head + c] = __float2bfloat16(sdV[r][c]);
                }
            }

            // --- zero shared buffers for next key-tile ---
            for (int i = tid; i < BN * HD; i += NT) {
                sdK[i / HD][i % HD] = 0.0f;
                sdV[i / HD][i % HD] = 0.0f;
            }
            __syncthreads();
        } // key-tile loop

        // ==================================================================
        // REDUCE dQ across 8 threads per row-group, write to global
        // ==================================================================
        // Each warp (32 threads) handles 4 consecutive query rows.
        // Rows within a warp sit in lanes [0..7],[8..15],[16..23],[24..31].
        int  gw  = (tid / 8) % 4;                        // 0..3
        unsigned msk = 0xFFu << (gw * 8);

        for (int dv = 0; dv < HD; ++dv) {
            float v = dq[dv];
            v += __shfl_down_sync(msk, v, 4);
            v += __shfl_down_sync(msk, v, 2);
            v += __shfl_down_sync(msk, v, 1);
            if (tid % TPR == 0 && qr < BM_e) {
                dQp[(qt + qr) * d_head + dv] = __float2bfloat16(v);
            }
        }
    } // query-tile loop
}

// ======================================================================
// Host-side wrapper (TVM-FFI)
// ======================================================================
void run(tvm::ffi::TensorView Q,  tvm::ffi::TensorView K,
         tvm::ffi::TensorView V,  tvm::ffi::TensorView O,
         tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ_out, tvm::ffi::TensorView dK_out,
         tvm::ffi::TensorView dV_out)
{
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = Q.size(3);
    assert(d == HD && "expected head-dim 128");

    const __nv_bfloat16* Q_ptr  = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr  = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr  = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_ptr  = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float*         L_ptr  = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_ptr      = static_cast<__nv_bfloat16*>(dQ_out.data_ptr());
    __nv_bfloat16* dK_ptr      = static_cast<__nv_bfloat16*>(dK_out.data_ptr());
    __nv_bfloat16* dV_ptr      = static_cast<__nv_bfloat16*>(dV_out.data_ptr());

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    // dK and dV are accumulated across query tiles → must be zeroed first
    size_t kv_bytes = (size_t)B * H * S * d * sizeof(__nv_bfloat16);
    CUDA_CHECK(cudaMemsetAsync(dK_ptr, 0, kv_bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(dV_ptr, 0, kv_bytes, stream));

    int nblocks = (int)(B * H);
    float scale = 1.0f / std::sqrtf((float)d);

    mha_bwd_kernel<<<nblocks, NT, 0, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
        dQ_ptr, dK_ptr, dV_ptr,
        (int)B, (int)H, (int)S, (int)d,
        scale, true);                       // causal = true

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

} // namespace mha_bwd_impl

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_impl::run);