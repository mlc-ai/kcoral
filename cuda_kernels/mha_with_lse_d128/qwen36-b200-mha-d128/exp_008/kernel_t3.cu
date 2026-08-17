#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
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

constexpr int HEAD_DIM = 128;
constexpr int SEGSIZE  = 64;
constexpr int BLOCK_M  = 16;
constexpr int NTHREADS = 256;

__global__ void mha_fwd_kernel(
    const __nv_bfloat16* __restrict__ Q_gm,
    const __nv_bfloat16* __restrict__ K_gm,
    const __nv_bfloat16* __restrict__ V_gm,
    __nv_bfloat16* __restrict__ O_gm,
    float* __restrict__ LSE_gm,
    int B, int H, int S, int D)
{
    int bid = blockIdx.x;
    if (bid >= B * H) return;

    int b_idx = bid / H;
    int h_idx = bid % H;

    unsigned tid = threadIdx.x;

    int row_str = D;
    int head_str = S * D;
    int batch_str = H * S * D;

    const __nv_bfloat16* Q0 = Q_gm + b_idx * batch_str + h_idx * head_str;
    const __nv_bfloat16* K0 = K_gm + b_idx * batch_str + h_idx * head_str;
    const __nv_bfloat16* V0 = V_gm + b_idx * batch_str + h_idx * head_str;
    __nv_bfloat16*        O0 = O_gm + b_idx * batch_str + h_idx * head_str;
    float*              LSE0 = LSE_gm + b_idx * H * S + h_idx * S;

    float scale = rsqrtf((float)D);

    // Shared memory layout:
    //   s_Q: BLOCK_M x HEAD_DIM bf16  (Q tile loaded once)
    //   s_K: SEGSIZE x HEAD_DIM bf16  (reloaded each segment)
    //   s_V: SEGSIZE x HEAD_DIM bf16  (reloaded each segment)
    //   s_O: BLOCK_M x HEAD_DIM fp32  (output accumulator)
    extern __shared__ char smem[];
    __nv_bfloat16* s_Q = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* s_K = s_Q + BLOCK_M * HEAD_DIM;
    __nv_bfloat16* s_V = s_K + SEGSIZE * HEAD_DIM;
    float* s_O         = reinterpret_cast<float*>(s_V + SEGSIZE * HEAD_DIM);

    // ---- Load Q tile into shared memory ----
    {
        int nkq = min(S, BLOCK_M);
        const __nv_bfloat16* Qblk = Q0;
        for (int off = tid; off < nkq * HEAD_DIM; off += NTHREADS) {
            int r = off / HEAD_DIM;
            int c = off % HEAD_DIM;
            s_Q[r * HEAD_DIM + c] = Qblk[r * row_str + c];
        }
    }

    // Initialize s_O (output accumulator) to zero
    for (int idx = tid; idx < BLOCK_M * HEAD_DIM; idx += NTHREADS) {
        s_O[idx] = 0.0f;
    }
    __syncthreads();

    // Softmax state per Q row (kept in registers)
    float row_m[BLOCK_M]; // running max
    float row_l[BLOCK_M]; // running sum of exps (relative to current max)

    for (int r = 0; r < BLOCK_M; ++r) {
        row_m[r] = -1e20f;
        row_l[r] = 0.0f;
    }

    // Thread group per Q row: NTHREADS/BLOCK_M = 16 threads share work on one Q row
    int gsz = NTHREADS / BLOCK_M;
    int qr  = tid / gsz;
    int ln  = tid % gsz;

    int n_seg = (S + SEGSIZE - 1) / SEGSIZE;

    for (int seg = 0; seg < n_seg; ++seg) {
        int k0 = seg * SEGSIZE;
        int nk = min(k0 + SEGSIZE, S) - k0;

        // --- Load K & V tiles cooperatively ---
        const __nv_bfloat16* Kblk = K0 + k0 * row_str;
        const __nv_bfloat16* Vblk = V0 + k0 * row_str;
        for (int off = tid; off < nk * HEAD_DIM; off += NTHREADS) {
            int r = off / HEAD_DIM;
            int c = off % HEAD_DIM;
            s_K[r * HEAD_DIM + c] = Kblk[r * row_str + c];
            s_V[r * HEAD_DIM + c] = Vblk[r * row_str + c];
        }
        __syncthreads();

        // --- Compute dot products for my_qrow ---
        // Read Q from shared memory, compute with K
        float q_vals[HEAD_DIM];
        for (int c = ln; c < HEAD_DIM; c += gsz) {
            q_vals[c] = (float)s_Q[qr * HEAD_DIM + c];
        }

        float pvals[SEGSIZE];
        float seg_max = -1e20f;
        for (int j = 0; j < nk; ++j) {
            float dot = 0.0f;
            for (int c = 0; c < HEAD_DIM; c++) {
                dot += q_vals[c] * (float)s_K[j * HEAD_DIM + c];
            }
            pvals[j] = dot * scale;
            if (pvals[j] > seg_max) seg_max = pvals[j];
        }

        // Broadcast seg_max from lane 0 to all lanes in group
        seg_max = __shfl_sync(0xFFFFFFFF, seg_max, 0);
        __syncthreads();

        // --- Online softmax merge ---
        float cur_m = row_m[qr];
        float cur_l = row_l[qr];

        // Compute rescale factor: exp(old_max - new_max)
        float rc = expf(cur_m - seg_max);

        // Update the running max
        row_m[qr] = seg_max;

        // IMPORTANT: Rescale EXISTING s_O by rc FIRST (before adding new contributions)
        if (rc != 1.0f && cur_l > 0.0f) {
            for (int c = ln; c < HEAD_DIM; c += gsz) {
                s_O[qr * HEAD_DIM + c] *= rc;
            }
        }

        // Scale the previous lse
        cur_l *= rc;

        // --- Accumulate new contributions ---
        for (int j = 0; j < nk; ++j) {
            float e = expf(pvals[j] - seg_max);
            cur_l += e;
            for (int c = ln; c < HEAD_DIM; c += gsz) {
                s_O[qr * HEAD_DIM + c] += e * (float)s_V[j * HEAD_DIM + c];
            }
        }

        // Reduce cur_l within group
        for (int o = gsz >> 1; o > 0; o >>= 1) {
            cur_l += __shfl_down_sync(0xFFFFFFFF, cur_l, o);
        }

        if (ln == 0) {
            row_l[qr] = cur_l;
        }

        __syncthreads();
    }

    // ---- Write results ----
    if (qr < S) {
        // Guard against log(0) - if row_l is 0, set LSE to -inf
        float final_lse;
        if (row_l[qr] > 0.0f) {
            final_lse = row_m[qr] + logf(row_l[qr]);
        } else {
            final_lse = -INFINITY;
        }
        LSE0[qr] = final_lse;

        float inv_denom = 1.0f / fmaxf(row_l[qr], 1e-30f);
        for (int c = ln; c < HEAD_DIM; c += gsz) {
            O0[qr * row_str + c] = __float2bfloat16(s_O[qr * HEAD_DIM + c] * inv_denom);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = static_cast<int>(Q.size(0));
    int H = static_cast<int>(Q.size(1));
    int S = static_cast<int>(Q.size(2));
    int D = static_cast<int>(Q.size(3));

    if (S == 0 || B == 0 || H == 0 || D != HEAD_DIM) return;

    const __nv_bfloat16* Q_d = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_d = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_d = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_d = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_d = static_cast<float*>(LSE.data_ptr());

    dim3 grid(B * H);
    dim3 block(NTHREADS);

    // SMEM: s_Q(BLOCK_M*D*2) + s_K(SEGSIZE*D*2) + s_V(SEGSIZE*D*2) + s_O(BLOCK_M*D*4)
    int smem_bytes = (BLOCK_M * HEAD_DIM + 2 * SEGSIZE * HEAD_DIM) * sizeof(__nv_bfloat16)
                   + BLOCK_M * HEAD_DIM * sizeof(float);

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    mha_fwd_kernel<<<grid, block, smem_bytes, stream>>>(
        Q_d, K_d, V_d, O_d, LSE_d, B, H, S, D);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace mha_impl

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_impl::run);