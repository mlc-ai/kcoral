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

static constexpr int BM   = 64;   // query tile height
static constexpr int BN   = 32;   // key tile height
static constexpr int NT   = 128;  // threads / block
static constexpr int HD   = 128;  // head dimension

// =========================================================================
// KERNEL 1: compute dK and dV
//   Grid: B * H * (S / BN) blocks — one block per key-tile
//   Each block walks ALL query tiles, accumulates dK[kt..], dV[kt..]
//   No atomics needed — each block owns unique key rows.
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
    if (bh >= B * H) return;
    int b = bh / H;
    int h = bh % H;

    // Offset by (key_tile_offset * BN)
    int kt_base = bh * BN;  // global key start position
    int kt_end  = kt_base + BN;
    if (kt_base >= S) return;
    if (kt_end > S) kt_end = S;
    int BN_e = kt_end - kt_base;

    int tid = threadIdx.x;
    int kl  = tid % BN;                    // key row (0..31)
    int hdrank = tid / BN;                 // 0 or 1 (two threads per key row)
    int hd_base = hdrank * (d_head / 2);   // head-column start (0 or 64)
    int hd_step = d_head / 2;              // 64 columns per thread

    int64_t ss = (int64_t)S * d_head;
    int64_t bs = ss * H;

    // Base pointers for this (b,h)
    const __nv_bfloat16* Qp  = Q  + b*bs + h*ss;
    const __nv_bfloat16* Kp  = K  + b*bs + h*ss;
    const __nv_bfloat16* Vp  = V  + b*bs + h*ss;
    const __nv_bfloat16* Op  = O  + b*bs + h*ss;
    const __nv_bfloat16* dOp = dO + b*bs + h*ss;
    const float*         Lp  = L  + b*H*S + h*S;
    __nv_bfloat16* dKp      = dK_out + b*bs + h*ss;
    __nv_bfloat16* dVp      = dV_out + b*bs + h*ss;

    // Shared memory
    extern __shared__ char smem[];
    __nv_bfloat16* sQ     = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sdO    = sQ + BM * (d_head + 4);          // padded
    __nv_bfloat16* sK     = sdO + BM * (d_head + 4);
    __nv_bfloat16* sV     = sK + BN * (d_head + 4);
    __nv_bfloat16* sO     = sV + BN * (d_head + 4);
    float*         sD_arr = reinterpret_cast<float*>(sO + BM * (d_head + 4));

    // ---- load fixed K, V tiles into shared memory ----
    for (int i = tid; i < BN_e * d_head; i += NT) {
        int r = i / d_head;
        int c = i % d_head;
        sK[r * (d_head + 4) + c] = Kp[(kt_base + r) * d_head + c];
        sV[r * (d_head + 4) + c] = Vp[(kt_base + r) * d_head + c];
    }
    __syncthreads();

    // ---- register accumulators for dK / dV ----
    float dK_reg[hd_step];
    float dV_reg[hd_step];
    for (int i = 0; i < hd_step; ++i) {
        dK_reg[i] = 0.0f;
        dV_reg[i] = 0.0f;
    }

    // =====================================================================
    // Walk all query tiles
    // =====================================================================
    for (int qt = 0; qt < S; qt += BM) {
        int qt_end = qt + BM;
        if (qt_end > S) qt_end = S;
        int BM_e = qt_end - qt;

        // --- load Q, dO, O tiles into shared memory ---
        for (int i = tid; i < BM_e * d_head; i += NT) {
            int r = i / d_head;
            int c = i % d_head;
            sQ   [r * (d_head + 4) + c] = Qp  [(qt + r) * d_head + c];
            sdO  [r * (d_head + 4) + c] = dOp [(qt + r) * d_head + c];
            sO   [r * (d_head + 4) + c] = Op  [(qt + r) * d_head + c];
        }

        // compute D = row_sum(dO * O) for each query row
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

        // --- compute S, P, dP, dS for this (ql, kl) pair ---
        int qg = qt;  // global query row base
        int kg = kt_base;  // global key row base

        for (int qr = 0; qr < BM_e; ++qr) {
            if (kl >= BN_e) break;

            // S_val = Q[qr] . K[kl] * scale
            float s_val = 0.0f;
            float dp_val = 0.0f;
            for (int dv = 0; dv < d_head; ++dv) {
                s_val += __bfloat162float(sQ  [qr * (d_head + 4) + dv])
                       * __bfloat162float(sK  [kl * (d_head + 4) + dv])
                       * attn_scale;
                dp_val += __bfloat162float(sdO [qr * (d_head + 4) + dv])
                        * __bfloat162float(sV  [kl * (d_head + 4) + dv]);
            }

            int qpos = qt + qr;
            int kpos = kt_base + kl;

            float p_val = 0.0f;
            float ds_val = 0.0f;
            if (causal && kpos > qpos) {
                p_val = 0.0f;
            } else {
                float lv = Lp[qpos];
                p_val = expf(s_val - lv);
                float dcorr = sD_arr[qr];
                ds_val = p_val * (dp_val - dcorr);
            }

            // Accumulate dK: dK[kl, dv] += ds_val * Q[qr, dv] * scale
            // Accumulate dV: dV[kl, dv] += p_val  * dO[qr, dv]
            for (int di = 0; di < hd_step; ++di) {
                int dv = hd_base + di;
                dK_reg[di] += ds_val * __bfloat162float(sQ [qr * (d_head + 4) + dv]) * attn_scale;
                dV_reg[di] += p_val  * __bfloat162float(sdO[qr * (d_head + 4) + dv]);
            }
        }

        __syncthreads();  // safe guard before re-using shared memory
    }

    // ---- write dK / dV to global memory ----
    // Cooperative write: each thread writes its portion
    for (int i = 0; i < hd_step; ++i) {
        int dv = hd_base + i;
        int kg = kt_base + kl;
        if (kg < S && kl < BN_e) {
            dKp[kg * d_head + dv] = __float2bfloat16(dK_reg[i]);
            dVp[kg * d_head + dv] = __float2bfloat16(dV_reg[i]);
        }
    }
}

// =========================================================================
// KERNEL 2: compute dQ
//   Grid: B * H * (S / BM) blocks — one block per query-tile
//   Each block walks ALL key tiles, accumulates dQ[qt..]
//   No atomics needed — each block owns unique query rows.
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
    if (bh >= B * H) return;
    int b = bh / H;
    int h = bh % H;

    int qt_base = bh * BM;
    int qt_end  = qt_base + BM;
    if (qt_base >= S) return;
    if (qt_end > S) qt_end = S;
    int BM_e = qt_end - qt_base;

    int tid = threadIdx.x;
    int qr  = tid % BM;                     // query row (0..63)
    int hdrank = tid / BM;                  // 2 threads per query row (128/64)
    int hd_base = hdrank * (d_head / 2);    // 0 or 64
    int hd_step = d_head / 2;               // 64 columns per thread

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
    __nv_bfloat16* sdO    = sQ + BM * (d_head + 4);
    __nv_bfloat16* sK     = sdO + BM * (d_head + 4);
    __nv_bfloat16* sV     = sK + BN * (d_head + 4);
    __nv_bfloat16* sO     = sV + BN * (d_head + 4);
    float*         sD_arr = reinterpret_cast<float*>(sO + BM * (d_head + 4));

    // ---- load fixed Q, dO, O tiles once ----
    for (int i = tid; i < BM_e * d_head; i += NT) {
        int r = i / d_head;
        int c = i % d_head;
        sQ  [r * (d_head + 4) + c] = Qp  [(qt_base + r) * d_head + c];
        sdO [r * (d_head + 4) + c] = dOp [(qt_base + r) * d_head + c];
        sO  [r * (d_head + 4) + c] = Op  [(qt_base + r) * d_head + c];
    }

    // compute D = row_sum(dO * O)
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

    // ---- register accumulator for dQ ----
    float dQ_reg[hd_step];
    for (int i = 0; i < hd_step; ++i) dQ_reg[i] = 0.0f;

    // =====================================================================
    // Walk all key tiles
    // =====================================================================
    for (int kt = 0; kt < S; kt += BN) {
        int kt_end = kt + BN;
        if (kt_end > S) kt_end = S;
        int BN_e = kt_end - kt;

        // --- load K, V tiles ---
        for (int i = tid; i < BN_e * d_head; i += NT) {
            int r = i / d_head;
            int c = i % d_head;
            sK[r * (d_head + 4) + c] = Kp[(kt + r) * d_head + c];
            sV[r * (d_head + 4) + c] = Vp[(kt + r) * d_head + c];
        }
        __syncthreads();

        int qpos_base = qt_base + qr;

        for (int kl = 0; kl < BN_e; ++kl) {
            // S_val = Q[qr] . K[kl] * scale
            float s_val = 0.0f;
            float dp_val = 0.0f;
            for (int dv = 0; dv < d_head; ++dv) {
                s_val += __bfloat162float(sQ  [qr * (d_head + 4) + dv])
                       * __bfloat162float(sK  [kl * (d_head + 4) + dv])
                       * attn_scale;
                dp_val += __bfloat162float(sdO [qr * (d_head + 4) + dv])
                        * __bfloat162float(sV  [kl * (d_head + 4) + dv]);
            }

            int qg = qt_base + qr;
            int kg = kt + kl;

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

            // Accumulate dQ: dQ[qr, dv] += ds_val * K[kl, dv] * scale
            for (int di = 0; di < hd_step; ++di) {
                int dv = hd_base + di;
                dQ_reg[di] += ds_val * __bfloat162float(sK[kl * (d_head + 4) + dv]) * attn_scale;
            }
        }
        __syncthreads();
    }

    // ---- write dQ to global memory ----
    for (int i = 0; i < hd_step; ++i) {
        int dv = hd_base + i;
        if (qr < BM_e) {
            dQp[(qt_base + qr) * d_head + dv] = __float2bfloat16(dQ_reg[i]);
        }
    }
}

// =========================================================================
// Host-side wrapper
// =========================================================================
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

    // Zero init outputs
    size_t kv_bytes = (size_t)B * H * S * d * sizeof(__nv_bfloat16);
    CUDA_CHECK(cudaMemsetAsync(dK_ptr, 0, kv_bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(dV_ptr, 0, kv_bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(dQ_ptr, 0, kv_bytes, stream));

    float scale = 1.0f / sqrtf((float)d);

    // Shared memory per block:
    //   sQ:  BM * (d+4) * 2
    //   sdO: BM * (d+4) * 2
    //   sK:  BN * (d+4) * 2
    //   sV:  BN * (d+4) * 2
    //   sO:  BM * (d+4) * 2
    //   sD:  BM * 4
    // Total: (BM*3 + BN*2) * (d+4)*2 + BM*4 bytes
    size_t smem_bytes = ((size_t)BM * 3 + (size_t)BN * 2) * (d + 4) * 2 + (size_t)BM * 4;

    // ---- Launch 1: dKV kernel ----
    int nblocks_dkv = (int)(B * H * ((S + BN - 1) / BN));
    mha_bwd_dKV_kernel<<<nblocks_dkv, NT, smem_bytes, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
        dK_ptr, dV_ptr,
        (int)B, (int)H, (int)S, (int)d,
        scale, true);
    CUDA_CHECK(cudaGetLastError());

    // ---- Launch 2: dQ kernel ----
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