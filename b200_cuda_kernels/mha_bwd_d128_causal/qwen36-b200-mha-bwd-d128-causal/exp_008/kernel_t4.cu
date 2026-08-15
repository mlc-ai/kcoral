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

static constexpr int BM   = 64;
static constexpr int BN   = 32;
static constexpr int NT   = 128;
static constexpr int HD   = 128;
static constexpr int HALF_D = HD / 2;
static constexpr int PADD = 4;

// =========================================================================
// KERNEL 1: compute dK and dV
//   Grid: B * H * ceil(S/BN) blocks — one block per key-tile per (b,h)
// =========================================================================
__global__ void mha_bwd_dKV_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float*         __restrict__ L,
    __nv_bfloat16*       __restrict__ dK_out,
    __nv_bfloat16*       __restrict__ dV_out,
    int B, int H, int S, int d_head,
    float attn_scale, bool causal)
{
    int bh  = blockIdx.x;
    if (bh >= B * H * ((S + BN - 1) / BN)) return;

    int kt_idx = bh / (B * H);
    int bhi    = bh % (B * H);
    int b      = bhi / H;
    int h      = bhi % H;

    int kt_base = kt_idx * BN;
    int kt_end  = kt_base + BN;
    if (kt_base >= S) return;
    if (kt_end > S) kt_end = S;
    int BN_e = kt_end - kt_base;

    int tid = threadIdx.x;
    int kl  = tid % BN;
    int hdrank = tid / BN;
    int hd_base = hdrank * HALF_D;
    // Each thread handles HALF_D columns (64)

    int64_t ss = (int64_t)S * d_head;
    int64_t bs = ss * H;

    const __nv_bfloat16* Qp  = Q  + b*bs + h*ss;
    const __nv_bfloat16* Kp  = K  + b*bs + h*ss;
    const __nv_bfloat16* Vp  = V  + b*bs + h*ss;
    const __nv_bfloat16* Op  = O  + b*bs + h*ss;
    const __nv_bfloat16* dOp = dO + b*bs + h*ss;
    const float*         Lp  = L  + b*H*S + h*S;
    __nv_bfloat16* dKp      = dK_out + b*bs + h*ss;
    __nv_bfloat16* dVp      = dV_out + b*bs + h*ss;

    extern __shared__ char smem[];
    __nv_bfloat16* sQ     = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sdO    = sQ + BM * (HD + PADD);
    __nv_bfloat16* sK     = sdO + BM * (HD + PADD);
    __nv_bfloat16* sV     = sK + BN * (HD + PADD);
    __nv_bfloat16* sO     = sV + BN * (HD + PADD);
    float*         sD_arr = reinterpret_cast<float*>(sO + BM * (HD + PADD));

    // Load fixed K, V tiles
    for (int i = tid; i < BN_e * d_head; i += NT) {
        int r = i / d_head;
        int c = i % d_head;
        sK[r * (HD + PADD) + c] = Kp[(kt_base + r) * d_head + c];
        sV[r * (HD + PADD) + c] = Vp[(kt_base + r) * d_head + c];
    }
    __syncthreads();

    // Register accumulators — must be compile-time sized
    float dK_reg[HALF_D];
    float dV_reg[HALF_D];
    for (int i = 0; i < HALF_D; ++i) {
        dK_reg[i] = 0.0f;
        dV_reg[i] = 0.0f;
    }

    for (int qt = 0; qt < S; qt += BM) {
        int qt_end = qt + BM;
        if (qt_end > S) qt_end = S;
        int BM_e = qt_end - qt;

        // Load Q, dO, O tiles
        for (int i = tid; i < BM_e * d_head; i += NT) {
            int r = i / d_head;
            int c = i % d_head;
            sQ  [r * (HD + PADD) + c] = Qp  [(qt + r) * d_head + c];
            sdO [r * (HD + PADD) + c] = dOp [(qt + r) * d_head + c];
            sO  [r * (HD + PADD) + c] = Op  [(qt + r) * d_head + c];
        }

        if (tid < BM_e) {
            float dsum = 0.0f;
            int qg = qt + tid;
            for (int dv = 0; dv < d_head; ++dv) {
                dsum += __bfloat162float(dOp[qg * d_head + dv])
                      * __bfloat162float(Op [qg * d_head + dv]);
            }
            sD_arr[tid] = dsum;
        }
        __syncthreads();

        // Process this (qr=kl_row_mapped?) — no, iterate over all query rows
        // Each thread handles 1 key row (kl), iterates over all query rows
        if (kl < BN_e) {
            int kg = kt_base + kl;
            for (int qr = 0; qr < BM_e; ++qr) {
                int qg = qt + qr;

                float s_val = 0.0f;
                float dp_val = 0.0f;
                for (int dv = 0; dv < d_head; ++dv) {
                    s_val += __bfloat162float(sQ  [qr * (HD + PADD) + dv])
                           * __bfloat162float(sK  [kl * (HD + PADD) + dv])
                           * attn_scale;
                    dp_val += __bfloat162float(sdO [qr * (HD + PADD) + dv])
                            * __bfloat162float(sV  [kl * (HD + PADD) + dv]);
                }

                float p_val = 0.0f;
                float ds_val = 0.0f;
                if (causal && kg > qg) {
                    p_val = 0.0f;
                } else {
                    float lv = Lp[qg];
                    p_val = expf(s_val - lv);
                    float dcorr = sD_arr[qr];
                    ds_val = p_val * (dp_val - dcorr);
                }

                for (int di = 0; di < HALF_D; ++di) {
                    int dv = hd_base + di;
                    dK_reg[di] += ds_val * __bfloat162float(sQ [qr * (HD + PADD) + dv]) * attn_scale;
                    dV_reg[di] += p_val  * __bfloat162float(sdO[qr * (HD + PADD) + dv]);
                }
            }
        }
        __syncthreads();
    }

    // Write results
    for (int i = 0; i < HALF_D; ++i) {
        int dv = hd_base + i;
        if (kl < BN_e) {
            int kg = kt_base + kl;
            dKp[kg * d_head + dv] = __float2bfloat16(dK_reg[i]);
            dVp[kg * d_head + dv] = __float2bfloat16(dV_reg[i]);
        }
    }
}

// =========================================================================
// KERNEL 2: compute dQ
//   Grid: B * H * ceil(S/BM) blocks — one block per query-tile per (b,h)
// =========================================================================
__global__ void mha_bwd_dQ_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float*         __restrict__ L,
    __nv_bfloat16*       __restrict__ dQ_out,
    int B, int H, int S, int d_head,
    float attn_scale, bool causal)
{
    int bh  = blockIdx.x;
    if (bh >= B * H * ((S + BM - 1) / BM)) return;

    int qt_idx = bh / (B * H);
    int bhi    = bh % (B * H);
    int b      = bhi / H;
    int h      = bhi % H;

    int qt_base = qt_idx * BM;
    int qt_end  = qt_base + BM;
    if (qt_base >= S) return;
    if (qt_end > S) qt_end = S;
    int BM_e = qt_end - qt_base;

    int tid = threadIdx.x;
    int qr  = tid % BM;
    int hdrank = tid / BM;
    int hd_base = hdrank * HALF_D;

    int64_t ss = (int64_t)S * d_head;
    int64_t bs = ss * H;

    const __nv_bfloat16* Qp  = Q  + b*bs + h*ss;
    const __nv_bfloat16* Kp  = K  + b*bs + h*ss;
    const __nv_bfloat16* Vp  = V  + b*bs + h*ss;
    const __nv_bfloat16* Op  = O  + b*bs + h*ss;
    const __nv_bfloat16* dOp = dO + b*bs + h*ss;
    const float*         Lp  = L  + b*H*S + h*S;
    __nv_bfloat16* dQp      = dQ_out + b*bs + h*ss;

    extern __shared__ char smem[];
    __nv_bfloat16* sQ     = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sdO    = sQ + BM * (HD + PADD);
    __nv_bfloat16* sK     = sdO + BM * (HD + PADD);
    __nv_bfloat16* sV     = sK + BN * (HD + PADD);
    __nv_bfloat16* sO     = sV + BN * (HD + PADD);
    float*         sD_arr = reinterpret_cast<float*>(sO + BM * (HD + PADD));

    // Load fixed Q, dO, O tiles once
    for (int i = tid; i < BM_e * d_head; i += NT) {
        int r = i / d_head;
        int c = i % d_head;
        sQ  [r * (HD + PADD) + c] = Qp  [(qt_base + r) * d_head + c];
        sdO [r * (HD + PADD) + c] = dOp [(qt_base + r) * d_head + c];
        sO  [r * (HD + PADD) + c] = Op  [(qt_base + r) * d_head + c];
    }

    if (tid < BM_e) {
        float dsum = 0.0f;
        int qg = qt_base + tid;
        for (int dv = 0; dv < d_head; ++dv) {
            dsum += __bfloat162float(dOp[qg * d_head + dv])
                  * __bfloat162float(Op [qg * d_head + dv]);
        }
        sD_arr[tid] = dsum;
    }
    __syncthreads();

    float dQ_reg[HALF_D];
    for (int i = 0; i < HALF_D; ++i) dQ_reg[i] = 0.0f;

    for (int kt = 0; kt < S; kt += BN) {
        int kt_end = kt + BN;
        if (kt_end > S) kt_end = S;
        int BN_e = kt_end - kt;

        for (int i = tid; i < BN_e * d_head; i += NT) {
            int r = i / d_head;
            int c = i % d_head;
            sK[r * (HD + PADD) + c] = Kp[(kt + r) * d_head + c];
            sV[r * (HD + PADD) + c] = Vp[(kt + r) * d_head + c];
        }
        __syncthreads();

        if (qr < BM_e) {
            int qg = qt_base + qr;
            for (int kl = 0; kl < BN_e; ++kl) {
                int kg = kt + kl;

                float s_val = 0.0f;
                float dp_val = 0.0f;
                for (int dv = 0; dv < d_head; ++dv) {
                    s_val += __bfloat162float(sQ  [qr * (HD + PADD) + dv])
                           * __bfloat162float(sK  [kl * (HD + PADD) + dv])
                           * attn_scale;
                    dp_val += __bfloat162float(sdO [qr * (HD + PADD) + dv])
                            * __bfloat162float(sV  [kl * (HD + PADD) + dv]);
                }

                float p_val = 0.0f;
                float ds_val = 0.0f;
                if (causal && kg > qg) {
                    p_val = 0.0f;
                } else {
                    float lv = Lp[qg];
                    p_val = expf(s_val - lv);
                    float dcorr = sD_arr[qr];
                    ds_val = p_val * (dp_val - dcorr);
                }

                for (int di = 0; di < HALF_D; ++di) {
                    int dv = hd_base + di;
                    dQ_reg[di] += ds_val * __bfloat162float(sK[kl * (HD + PADD) + dv]) * attn_scale;
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < HALF_D; ++i) {
        int dv = hd_base + i;
        if (qr < BM_e) {
            dQp[(qt_base + qr) * d_head + dv] = __float2bfloat16(dQ_reg[i]);
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

    float scale = 1.0f / sqrtf((float)d);

    // Shared memory: sQ + sdO + sK + sV + sO + sD_arr
    size_t smem_bytes = ((size_t)BM * 3 + (size_t)BN * 2) * (HD + PADD) * 2 + (size_t)BM * 4;

    // dKV: one block per key-tile per (b,h). No atomics needed.
    int nblocks_dkv = (int)(B * H * ((S + BN - 1) / BN));
    mha_bwd_dKV_kernel<<<nblocks_dkv, NT, smem_bytes, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
        dK_ptr, dV_ptr,
        (int)B, (int)H, (int)S, (int)d,
        scale, true);
    CUDA_CHECK(cudaGetLastError());

    // dQ: one block per query-tile per (b,h). No atomics needed.
    int nblocks_dq = (int)(B * H * ((S + BM - 1) / BM));
    mha_bwd_dQ_kernel<<<nblocks_dq, NT, smem_bytes, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
        dQ_ptr,
        (int)B, (int)H, (int)S, (int)d,
        scale, true);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

} // namespace mha_bwd_impl

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_impl::run);