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

static constexpr int BM   = 16;
static constexpr int BN   = 16;
static constexpr int NT   = 128;
static constexpr int TPR  = NT / BM;   // 8 threads per query row
static constexpr int KPRT = BN / TPR;  // 2 key-cols per thread
static constexpr int HD   = 128;

__global__ void mha_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float*         __restrict__ L,
    __nv_bfloat16*       __restrict__ dQ_out,
    __nv_bfloat16*       __restrict__ dK_out,
    __nv_bfloat16*       __restrict__ dV_out,
    int B, int H, int S, int d_head,
    float attn_scale, bool causal)
{
    int bh  = blockIdx.x;
    if (bh >= B * H) return;
    int b   = bh / H;
    int h   = bh % H;
    int tid = threadIdx.x;

    int qr  = tid / TPR;
    int kc0 = (tid % TPR) * KPRT;

    int64_t ss = (int64_t)S * d_head;
    int64_t bs = ss * H;

    const __nv_bfloat16* Qp  = Q  + b*bs + h*ss;
    const __nv_bfloat16* Kp  = K  + b*bs + h*ss;
    const __nv_bfloat16* Vp  = V  + b*bs + h*ss;
    const __nv_bfloat16* Op  = O  + b*bs + h*ss;
    const __nv_bfloat16* dOp = dO + b*bs + h*ss;
    const float*         Lp  = L  + b*H*S + h*S;
    __nv_bfloat16* dQp      = dQ_out + b*bs + h*ss;
    __nv_bfloat16* dKp      = dK_out + b*bs + h*ss;
    __nv_bfloat16* dVp      = dV_out + b*bs + h*ss;

    __shared__ __nv_bfloat16 sQ  [BM][HD];
    __shared__ __nv_bfloat16 sK  [BN][HD];
    __shared__ __nv_bfloat16 sV  [BN][HD];
    __shared__ __nv_bfloat16 sdO [BM][HD];
    __shared__ float         sD  [BM];
    __shared__ float         sdK [BN][HD+1];
    __shared__ float         sdV [BN][HD+1];

    for (int qt = 0; qt < S; qt += BM) {
        int qt_end = qt + BM;
        if (qt_end > S) qt_end = S;
        int BM_e = qt_end - qt;

        if (qr < BM_e) {
            for (int i = tid; i < BM_e * d_head; i += NT) {
                int r = i / d_head;
                int c = i % d_head;
                sQ[r][c]  = Qp  [(qt+r)*d_head + c];
                sdO[r][c] = dOp [(qt+r)*d_head + c];
            }

            int qg   = qt + qr;
            float dsum = 0.0f;
            for (int dv = 0; dv < d_head; ++dv) {
                dsum += __bfloat162float(dOp[qg*d_head + dv])
                      * __bfloat162float(Op [qg*d_head + dv]);
            }
            sD[qr] = dsum;
        }
        __syncthreads();

        float dq_reg[HD];
        for (int i = 0; i < HD; ++i) dq_reg[i] = 0.0f;

        for (int i = tid; i < BN * HD; i += NT) {
            sdK[i / HD][i % HD] = 0.0f;
            sdV[i / HD][i % HD] = 0.0f;
        }
        __syncthreads();

        for (int kt = 0; kt < S; kt += BN) {
            int kt_end = kt + BN;
            if (kt_end > S) kt_end = S;
            int BN_e = kt_end - kt;

            for (int i = tid; i < BN_e * d_head; i += NT) {
                int r = i / d_head;
                int c = i % d_head;
                sK[r][c] = Kp[(kt+r)*d_head + c];
                sV[r][c] = Vp[(kt+r)*d_head + c];
            }
            __syncthreads();

            float sval[KPRT];
            float dpvl[KPRT];

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

            int   qg = qt + qr;
            float lv = Lp[qg];
            float dcorr = sD[qr];

            for (int kc = 0; kc < KPRT; ++kc) {
                int kl = kc0 + kc;
                if (kl >= BN_e || qr >= BM_e) continue;
                int kg = kt + kl;
                if (kg >= S) continue;

                float sv = sval[kc];
                if (causal && kg > qg) sv = -1e20f;

                float pv = expf(sv - lv);
                float ds = pv * (dpvl[kc] - dcorr);

                for (int dv = 0; dv < HD; ++dv) {
                    dq_reg[dv] += ds * __bfloat162float(sK[kl][dv]) * attn_scale;
                }

                for (int dv = 0; dv < HD; ++dv) {
                    float contrib = ds * __bfloat162float(sQ[qr][dv]) * attn_scale;
                    sdK[kl][dv] += contrib;
                }

                for (int dv = 0; dv < HD; ++dv) {
                    float contrib = pv * __bfloat162float(sdO[qr][dv]);
                    sdV[kl][dv] += contrib;
                }
            }

            __syncthreads();

            for (int i = tid; i < BN_e * d_head; i += NT) {
                int r  = i / d_head;
                int c  = i % d_head;
                int kg = kt + r;
                if (kg < S) {
                    dKp[kg*d_head + c] = __float2bfloat16(sdK[r][c]);
                    dVp[kg*d_head + c] = __float2bfloat16(sdV[r][c]);
                }
            }

            for (int i = tid; i < BN * HD; i += NT) {
                sdK[i / HD][i % HD] = 0.0f;
                sdV[i / HD][i % HD] = 0.0f;
            }
            __syncthreads();
        }

        // Reduce dQ: 8 threads -> 1 using XOR-based tree reduction within warp
        unsigned msk = 0xFFFFFFFF;
        for (int dv = 0; dv < HD; ++dv) {
            float v = dq_reg[dv];
            v += __shfl_xor_sync(msk, v, 4);
            v += __shfl_xor_sync(msk, v, 2);
            v += __shfl_xor_sync(msk, v, 1);
            if (tid % TPR == 0 && qr < BM_e) {
                dQp[(qt + qr) * d_head + dv] = __float2bfloat16(v);
            }
        }
    }
}

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
    assert(d == HD);

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

    size_t kv_bytes = (size_t)B * H * S * d * sizeof(__nv_bfloat16);
    CUDA_CHECK(cudaMemsetAsync(dK_ptr, 0, kv_bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(dV_ptr, 0, kv_bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(dQ_ptr, 0, kv_bytes, stream));

    int nblocks = (int)(B * H);
    float scale = 1.0f / sqrtf((float)d);

    mha_bwd_kernel<<<nblocks, NT, 0, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
        dQ_ptr, dK_ptr, dV_ptr,
        (int)B, (int)H, (int)S, (int)d,
        scale, true);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

} // namespace mha_bwd_impl

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_impl::run);