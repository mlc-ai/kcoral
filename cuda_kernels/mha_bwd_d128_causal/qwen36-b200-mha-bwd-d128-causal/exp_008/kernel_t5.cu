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

static constexpr int HD   = 128;
static constexpr int PADD = 4;

// =========================================================================
// KERNEL 1: compute dK, dV  (one block per (b,h,key_tile))
//     Shared mem: sQ + sdO + sK + sV + sD  (NO sO needed)
// =========================================================================
template <int BM, int BN, int NT>
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
    int bhkt = blockIdx.x;
    int n_bhkt = B * H * ((S + BN - 1) / BN);
    if (bhkt >= n_bhkt) return;

    int kt_idx = bhkt / (B * H);
    int bhi    = bhkt % (B * H);
    int b      = bhi / H;
    int h      = bhi % H;

    int kt_base = kt_idx * BN;
    if (kt_base >= S) return;
    int kt_end  = kt_base < S ? (kt_base + BN < S ? kt_base + BN : S) : S;
    int BN_e = kt_end - kt_base;

    int tid = threadIdx.x;
    int kl  = tid % BN;
    int hdrank = tid / BN;           // 0..3 for NT=128,BN=32 => 4 threads per key-row
    int HD_SUB = HD / (NT / BN);     // 128/4 = 32 cols per thread

    int64_t ss = (int64_t)S * d_head;
    int64_t bs = ss * H;

    const __nv_bfloat16* Qp  = Q  + b*bs + h*ss;
    const __nv_bfloat16* Kp  = K  + b*bs + h*ss;
    const __nv_bfloat16* Vp  = V  + b*bs + h*ss;
    const __nv_bfloat16* Op  = O  + b*bs + h*ss;
    const __nv_bfloat16* dOp = dO + b*bs + h*ss;
    const float*         Lp  = L  + b*H*S + h*S;
    __nv_bfloat16* dKp       = dK_out + b*bs + h*ss;
    __nv_bfloat16* dVp       = dV_out + b*bs + h*ss;

    // Shared memory layout: sQ + sdO + sK + sV + sD
    extern __shared__ char smem[];
    __nv_bfloat16* sQ  = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sdO = sQ + BM * (HD + PADD);
    __nv_bfloat16* sK  = sdO + BM * (HD + PADD);
    __nv_bfloat16* sV  = sK + BN * (HD + PADD);
    float*         sD  = reinterpret_cast<float*>(sV + BN * (HD + PADD));

    // Load K, V tiles
    for (int i = tid; i < BN_e * d_head; i += NT) {
        int r = i / d_head; int c = i % d_head;
        sK[r * (HD + PADD) + c] = Kp[(kt_base + r) * d_head + c];
        sV[r * (HD + PADD) + c] = Vp[(kt_base + r) * d_head + c];
    }
    __syncthreads();

    float dK_reg[HD_SUB];
    float dV_reg[HD_SUB];
    for (int i = 0; i < HD_SUB; ++i) { dK_reg[i] = 0.0f; dV_reg[i] = 0.0f; }

    int hd_base = hdrank * HD_SUB;

    for (int qt = 0; qt < S; qt += BM) {
        int qt_end = qt < S ? (qt + BM < S ? qt + BM : S) : S;
        int BM_e = qt_end - qt;

        for (int i = tid; i < BM_e * d_head; i += NT) {
            int r = i / d_head; int c = i % d_head;
            sQ[r * (HD + PADD) + c] = Qp[(qt + r) * d_head + c];
            sdO[r * (HD + PADD) + c] = dOp[(qt + r) * d_head + c];
        }

        if (tid < BM_e) {
            float ds = 0.0f;
            int qg = qt + tid;
            for (int v = 0; v < d_head; ++v)
                ds += __bfloat162float(dOp[qg*d_head+v]) * __bfloat162float(Op[qg*d_head+v]);
            sD[tid] = ds;
        }
        __syncthreads();

        if (kl < BN_e) {
            int kg = kt_base + kl;
            for (int qr = 0; qr < BM_e; ++qr) {
                int qg = qt + qr;
                float sv = 0.0f, dp = 0.0f;
                for (int v = 0; v < d_head; ++v) {
                    sv += __bfloat162float(sQ[qr*(HD+PADD)+v]) * __bfloat162float(sK[kl*(HD+PADD)+v]) * attn_scale;
                    dp += __bfloat162float(sdO[qr*(HD+PADD)+v]) * __bfloat162float(sV[kl*(HD+PADD)+v]);
                }
                float pv = 0.0f, dsv = 0.0f;
                if (!(causal && kg > qg)) {
                    pv = expf(sv - Lp[qg]);
                    dsv = pv * (dp - sD[qr]);
                }
                for (int di = 0; di < HD_SUB; ++di) {
                    int v = hd_base + di;
                    dK_reg[di] += dsv * __bfloat162float(sQ[qr*(HD+PADD)+v]) * attn_scale;
                    dV_reg[di] += pv  * __bfloat162float(sdO[qr*(HD+PADD)+v]);
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < HD_SUB; ++i) {
        int v = hd_base + i;
        if (kl < BN_e) {
            int kg = kt_base + kl;
            dKp[kg*d_head+v] = __float2bfloat16(dK_reg[i]);
            dVp[kg*d_head+v] = __float2bfloat16(dV_reg[i]);
        }
    }
}

// =========================================================================
// KERNEL 2: compute dQ  (one block per (b,h,query_tile))
//     Shared mem: sQ + sdO + sK + sV + sO + sD
// =========================================================================
template <int BM, int BN, int NT>
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
    int bhtq = blockIdx.x;
    int n_bhtq = B * H * ((S + BM - 1) / BM);
    if (bhtq >= n_bhtq) return;

    int qt_idx = bhtq / (B * H);
    int bhi    = bhtq % (B * H);
    int b      = bhi / H;
    int h      = bhi % H;

    int qt_base = qt_idx * BM;
    if (qt_base >= S) return;
    int qt_end  = qt_base < S ? (qt_base + BM < S ? qt_base + BM : S) : S;
    int BM_e = qt_end - qt_base;

    int tid = threadIdx.x;
    int qr  = tid % BM;
    int hdrank = tid / BM;
    int HD_SUB = HD / (NT / BM);

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
    __nv_bfloat16* sQ  = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sdO = sQ + BM * (HD + PADD);
    __nv_bfloat16* sK  = sdO + BM * (HD + PADD);
    __nv_bfloat16* sV  = sK + BN * (HD + PADD);
    __nv_bfloat16* sO  = sV + BN * (HD + PADD);
    float*         sD  = reinterpret_cast<float*>(sO + BM * (HD + PADD));

    // Load Q, dO, O once
    for (int i = tid; i < BM_e * d_head; i += NT) {
        int r = i / d_head; int c = i % d_head;
        sQ [r*(HD+PADD)+c] = Qp [(qt_base+r)*d_head+c];
        sdO[r*(HD+PADD)+c] = dOp[(qt_base+r)*d_head+c];
        sO [r*(HD+PADD)+c] = Op [(qt_base+r)*d_head+c];
    }
    if (tid < BM_e) {
        float ds = 0.0f;
        int qg = qt_base + tid;
        for (int v = 0; v < d_head; ++v)
            ds += __bfloat162float(dOp[qg*d_head+v]) * __bfloat162float(Op[qg*d_head+v]);
        sD[tid] = ds;
    }
    __syncthreads();

    float dQ_reg[HD_SUB];
    for (int i = 0; i < HD_SUB; ++i) dQ_reg[i] = 0.0f;

    int hd_base = hdrank * HD_SUB;

    for (int kt = 0; kt < S; kt += BN) {
        int kt_end = kt < S ? (kt + BN < S ? kt + BN : S) : S;
        int BN_e = kt_end - kt;

        for (int i = tid; i < BN_e * d_head; i += NT) {
            int r = i / d_head; int c = i % d_head;
            sK[r*(HD+PADD)+c] = Kp[(kt+r)*d_head+c];
            sV[r*(HD+PADD)+c] = Vp[(kt+r)*d_head+c];
        }
        __syncthreads();

        if (qr < BM_e) {
            int qg = qt_base + qr;
            for (int kl = 0; kl < BN_e; ++kl) {
                int kg = kt + kl;
                float sv = 0.0f, dp = 0.0f;
                for (int v = 0; v < d_head; ++v) {
                    sv += __bfloat162float(sQ[qr*(HD+PADD)+v]) * __bfloat162float(sK[kl*(HD+PADD)+v]) * attn_scale;
                    dp += __bfloat162float(sdO[qr*(HD+PADD)+v]) * __bfloat162float(sV[kl*(HD+PADD)+v]);
                }
                float pv = 0.0f, dsv = 0.0f;
                if (!(causal && kg > qg)) {
                    pv = expf(sv - Lp[qg]);
                    dsv = pv * (dp - sD[qr]);
                }
                for (int di = 0; di < HD_SUB; ++di) {
                    int v = hd_base + di;
                    dQ_reg[di] += dsv * __bfloat162float(sK[kl*(HD+PADD)+v]) * attn_scale;
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < HD_SUB; ++i) {
        int v = hd_base + i;
        if (qr < BM_e)
            dQp[(qt_base+qr)*d_head+v] = __float2bfloat16(dQ_reg[i]);
    }
}

void run(tvm::ffi::TensorView Q,  tvm::ffi::TensorView K,
         tvm::ffi::TensorView V,  tvm::ffi::TensorView O,
         tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ_out, tvm::ffi::TensorView dK_out,
         tvm::ffi::TensorView dV_out)
{
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0), H = Q.size(1), S = Q.size(2), d = Q.size(3);
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

    size_t bytes = (size_t)B * H * S * d * sizeof(__nv_bfloat16);
    CUDA_CHECK(cudaMemsetAsync(dK_ptr, 0, bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(dV_ptr, 0, bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(dQ_ptr, 0, bytes, stream));

    float scale = 1.0f / sqrtf((float)d);

    // Tile parameters
    constexpr int BM = 64, BN = 32, NT = 128;

    // dKV: 4 buffers (sQ,sdO,sK,sV) + sD float
    size_t smem_dkv = (size_t)BM*(HD+PADD)*2 * 2   // sQ + sdO
                     + (size_t)BN*(HD+PADD)*2 * 2   // sK + sV
                     + (size_t)BM * 4;               // sD
                                                     // = 50688 + 16896 + 256 = 67,840 ... 

    // Wait, let me recalc:
    // sQ: 64*132*2=16896
    // sdO:16896
    // sK: 32*132*2=8448  
    // sV: 8448
    // sD: 256
    // Total = 16896+16896+8448+8448+256 = 50944
    
    // dQ: 5 buffers + sD 
    // sQ:16896 sdO:16896 sK:8448 sV:8448 sO:16896 sD:256
    // Total = 16896+16896+8448+8448+16896+256 = 67840
    
    size_t smem_dq = smem_dkv + (size_t)BM*(HD+PADD)*2;  // add sO

    int nblocks_dkv = (int)(B * H * ((S + BN - 1) / BN));
    mha_bwd_dKV_kernel<BM, BN, NT><<<nblocks_dkv, NT, (int)smem_dkv, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
        dK_ptr, dV_ptr, (int)B, (int)H, (int)S, (int)d, scale, true);
    CUDA_CHECK(cudaGetLastError());

    int nblocks_dq = (int)(B * H * ((S + BM - 1) / BM));
    mha_bwd_dQ_kernel<BM, BN, NT><<<nblocks_dq, NT, (int)smem_dq, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
        dQ_ptr, (int)B, (int)H, (int)S, (int)d, scale, true);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

} // namespace mha_bwd_impl

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_impl::run);