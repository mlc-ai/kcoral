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
static constexpr int PAD  = 8;

// =========================================================================
// dKV kernel: one block per (b,h,key_tile). Walks ALL query tiles.
// BM=64,BN=64,NT=128 => 2 threads/key-row, 64 cols/thread
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
    static_assert(HD == 128 && BN == 64 && NT == 128 && BM == 64);
    static_assert((NT % BN) == 0);
    constexpr int HDRANKS = NT / BN;  // 2
    constexpr int HD_SUB  = HD / HDRANKS; // 64

    int bhkt = blockIdx.x;
    int tot_bhkt = B * H * ((S + BN - 1) / BN);
    if (bhkt >= tot_bhkt) return;

    int kt_idx = bhkt / (B * H);
    int bhi    = bhkt % (B * H);
    int b = bhi / H, h = bhi % H;

    int kt_base = kt_idx * BN;
    if (kt_base >= S) return;
    int kt_end = (kt_base + BN < S) ? kt_base + BN : S;
    int BN_e = kt_end - kt_base;

    int tid = threadIdx.x;
    int kl = tid % BN;
    int hdrank = tid / BN;
    int hd_base = hdrank * HD_SUB;

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

    extern __shared__ char smem[];
    __nv_bfloat16* sQ  = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sdO = sQ + BM * (HD + PAD);
    __nv_bfloat16* sK  = sdO + BM * (HD + PAD);
    __nv_bfloat16* sV  = sK + BN * (HD + PAD);
    float*         sD  = reinterpret_cast<float*>(sV + BN * (HD + PAD));

    // Load fixed K/V tiles
    for (int i = tid; i < BN_e * d_head; i += NT) {
        int r = i / d_head, c = i % d_head;
        sK[r*(HD+PAD)+c] = Kp[(kt_base+r)*d_head+c];
        sV[r*(HD+PAD)+c] = Vp[(kt_base+r)*d_head+c];
    }
    __syncthreads();

    float dK_reg[HD_SUB], dV_reg[HD_SUB];
    for (int i = 0; i < HD_SUB; ++i) { dK_reg[i] = 0.0f; dV_reg[i] = 0.0f; }

    int n_qtiles = (S + BM - 1) / BM;
    for (int qi = 0; qi < n_qtiles; ++qi) {
        int qt = qi * BM;
        if (qt >= S) break;
        int qt_end = (qt + BM < S) ? qt + BM : S;
        int BM_e = qt_end - qt;

        // Load Q/dO tile
        for (int i = tid; i < BM_e * d_head; i += NT) {
            int r = i / d_head, c = i % d_head;
            sQ [r*(HD+PAD)+c] = Qp [(qt+r)*d_head+c];
            sdO[r*(HD+PAD)+c] = dOp[(qt+r)*d_head+c];
        }

        // Compute D per query row
        if (tid < BM_e) {
            float ds = 0.0f;
            int qg = qt + tid;
            for (int v = 0; v < d_head; ++v)
                ds += __bfloat162float(dOp[qg*d_head+v]) * __bfloat162float(Op[qg*d_head+v]);
            sD[tid] = ds;
        }
        __syncthreads();

        // Accumulate dK / dV
        if (kl < BN_e) {
            int kg = kt_base + kl;
            for (int qr = 0; qr < BM_e; ++qr) {
                int qg = qt + qr;
                float sv = 0.0f, dp = 0.0f;
                #pragma unroll
                for (int v = 0; v < d_head; ++v) {
                    sv += __bfloat162float(sQ [qr*(HD+PAD)+v]) * __bfloat162float(sK[kl*(HD+PAD)+v]) * attn_scale;
                    dp += __bfloat162float(sdO[qr*(HD+PAD)+v]) * __bfloat162float(sV[kl*(HD+PAD)+v]);
                }
                float pv = 0.0f, dsv = 0.0f;
                if (!(causal && kg > qg)) {
                    pv  = expf(sv - Lp[qg]);
                    dsv = pv * (dp - sD[qr]);
                }
                #pragma unroll
                for (int di = 0; di < HD_SUB; ++di) {
                    int v = hd_base + di;
                    dK_reg[di] += dsv * __bfloat162float(sQ [qr*(HD+PAD)+v]) * attn_scale;
                    dV_reg[di] += pv  * __bfloat162float(sdO[qr*(HD+PAD)+v]);
                }
            }
        }
        __syncthreads();
    }

    // Write-back
    for (int di = 0; di < HD_SUB; ++di) {
        int v = hd_base + di;
        if (kl < BN_e) {
            int kg = kt_base + kl;
            dKp[kg*d_head+v] = __float2bfloat16(dK_reg[di]);
            dVp[kg*d_head+v] = __float2bfloat16(dV_reg[di]);
        }
    }
}

// =========================================================================
// dQ kernel: one block per (b,h,query_tile). Walks ALL key tiles.
// BM=64,BN=64,NT=128 => 2 threads/query-row, 64 cols/thread
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
    static_assert(HD == 128 && BM == 64 && NT == 128);
    static_assert((NT % BM) == 0);
    constexpr int HDRANKS = NT / BM; // 2
    constexpr int HD_SUB  = HD / HDRANKS; // 64

    int bhtq = blockIdx.x;
    int tot_bhtq = B * H * ((S + BM - 1) / BM);
    if (bhtq >= tot_bhtq) return;

    int qt_idx = bhtq / (B * H);
    int bhi    = bhtq % (B * H);
    int b = bhi / H, h = bhi % H;

    int qt_base = qt_idx * BM;
    if (qt_base >= S) return;
    int qt_end = (qt_base + BM < S) ? qt_base + BM : S;
    int BM_e = qt_end - qt_base;

    int tid = threadIdx.x;
    int qr = tid % BM;
    int hdrank = tid / BM;
    int hd_base = hdrank * HD_SUB;

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
    __nv_bfloat16* sdO = sQ + BM*(HD+PAD);
    __nv_bfloat16* sK  = sdO + BM*(HD+PAD);
    __nv_bfloat16* sV  = sK  + BN*(HD+PAD);
    __nv_bfloat16* sO  = sV  + BN*(HD+PAD);
    float*         sD  = reinterpret_cast<float*>(sO + BM*(HD+PAD));

    // Load fixed Q/dO/O tile
    for (int i = tid; i < BM_e*d_head; i += NT) {
        int r = i/d_head, c = i%d_head;
        sQ[r*(HD+PAD)+c]  = Qp [(qt_base+r)*d_head+c];
        sdO[r*(HD+PAD)+c] = dOp[(qt_base+r)*d_head+c];
        sO[r*(HD+PAD)+c]  = Op [(qt_base+r)*d_head+c];
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

    int n_ktiles = (S + BN - 1) / BN;
    for (int ki = 0; ki < n_ktiles; ++ki) {
        int kt = ki * BN;
        if (kt >= S) break;
        int kt_end = (kt+BN < S) ? kt+BN : S;
        int BN_e = kt_end - kt;

        // Load K/V tile
        for (int i = tid; i < BN_e*d_head; i += NT) {
            int r = i/d_head, c = i%d_head;
            sK[r*(HD+PAD)+c] = Kp[(kt+r)*d_head+c];
            sV[r*(HD+PAD)+c] = Vp[(kt+r)*d_head+c];
        }
        __syncthreads();

        // Accumulate dQ
        if (qr < BM_e) {
            int qg = qt_base + qr;
            for (int kl = 0; kl < BN_e; ++kl) {
                int kg = kt + kl;
                float sv = 0.0f, dp = 0.0f;
                #pragma unroll
                for (int v = 0; v < d_head; ++v) {
                    sv += __bfloat162float(sQ [qr*(HD+PAD)+v]) * __bfloat162float(sK[kl*(HD+PAD)+v]) * attn_scale;
                    dp += __bfloat162float(sdO[qr*(HD+PAD)+v]) * __bfloat162float(sV[kl*(HD+PAD)+v]);
                }
                float pv = 0.0f, dsv = 0.0f;
                if (!(causal && kg > qg)) {
                    pv  = expf(sv - Lp[qg]);
                    dsv = pv * (dp - sD[qr]);
                }
                #pragma unroll
                for (int di = 0; di < HD_SUB; ++di) {
                    int v = hd_base + di;
                    dQ_reg[di] += dsv * __bfloat162float(sK[kl*(HD+PAD)+v]) * attn_scale;
                }
            }
        }
        __syncthreads();
    }

    // Write-back
    for (int di = 0; di < HD_SUB; ++di) {
        int v = hd_base + di;
        if (qr < BM_e)
            dQp[(qt_base+qr)*d_head+v] = __float2bfloat16(dQ_reg[di]);
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

    constexpr int BM = 64, BN = 64, NT = 128;

    // dKV smem: sQ + sdO + sK + sV + sD = BM*2*(HD+PAD)*2 + BN*(HD+PAD)*2 + BM*4
    size_t smem_dkv = (size_t)(2*BM + BN)*(HD+PAD)*2 + (size_t)BM*4;
    // dQ smem: sQ + sdO + sK + sV + sO + sD = BM*3*(HD+PAD)*2 + BN*(HD+PAD)*2 + BM*4
    size_t smem_dq  = smem_dkv + (size_t)BM*(HD+PAD)*2;

    // Verify sizes fit in shared memory (< 164KB for SM100)
    assert(smem_dkv <= 164*1024);
    assert(smem_dq  <= 164*1024);

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