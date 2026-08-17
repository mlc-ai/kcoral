#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <cfloat>
#include <cmath>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_d128_causal {

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int NT = 64;

__global__ void fa_fwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S, int B, int H)
{
    extern __shared__ char smem[];

    // Shared memory layout:
    // sQ:     [BM][128] bf16  = 64*128*2 = 16KB,  offset 0
    // sK:     [BN][128] bf16  = 64*128*2 = 16KB,  offset 16KB
    // sV:     [BN][128] bf16  = 64*128*2 = 16KB,  offset 32KB
    // o_acc:  [BM][128] fp32  = 64*128*4 = 32KB,  offset 48KB
    // Total: ~80KB

    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sK = sQ + BM * 128;
    __nv_bfloat16* sV = sK + BN * 128;
    float* o_acc = reinterpret_cast<float*>(sV + BN * 128);

    int bid = blockIdx.x;
    int q_blocks_per_bh = (S + BM - 1) / BM;
    int q_block_idx = bid % q_blocks_per_bh;
    int bh_idx = bid / q_blocks_per_bh;
    int batch_idx = bh_idx / H;
    int head_idx = bh_idx % H;

    int q_start = q_block_idx * BM;
    int tid = threadIdx.x;

    size_t stride_bh = (size_t)H * S * 128;
    size_t stride_hs = (size_t)S * 128;

    const __nv_bfloat16* Q_base = Q + (size_t)batch_idx * stride_bh + head_idx * stride_hs;
    const __nv_bfloat16* K_base = K + (size_t)batch_idx * stride_bh + head_idx * stride_hs;
    const __nv_bfloat16* V_base = V + (size_t)batch_idx * stride_bh + head_idx * stride_hs;
    __nv_bfloat16* O_base = O + (size_t)batch_idx * stride_bh + head_idx * stride_hs;
    float* LSE_base = LSE + (size_t)batch_idx * H * S + head_idx * S;

    float inv_sqrt_d = rsqrtf(128.0f);
    int num_k_steps = (S + BN - 1) / BN;

    // Init output accumulators
    for (int i = tid; i < BM * 128; i += NT) {
        o_acc[i] = 0.0f;
    }
    __syncthreads();

    // Load Q tile once
    for (int d = tid; d < 128; d += NT) {
        sQ[tid * 128 + d] = Q_base[(size_t)(q_start + tid) * 128 + d];
    }
    __syncthreads();

    // Per-thread online softmax state for this query row
    float m = -FLT_MAX;
    float l = 0.0f;
    int q_abs = q_start + tid;

    float local_s[BN];

    for (int ks = 0; ks < num_k_steps; ks++) {
        int k_start = ks * BN;

        // Cooperatively load K and V tiles
        for (int d = tid; d < 128; d += NT) {
            int k_abs = k_start + tid;
            if (k_abs < S) {
                sK[tid * 128 + d] = K_base[(size_t)k_abs * 128 + d];
                sV[tid * 128 + d] = V_base[(size_t)k_abs * 128 + d];
            } else {
                sK[tid * 128 + d] = __float2bfloat16(0.0f);
                sV[tid * 128 + d] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        // Compute S[q_abs][k] for all k in this block
        for (int k = 0; k < BN; k++) {
            float s = 0.0f;
            int k_abs_local = k_start + k;

            for (int d = 0; d < 128; d += 4) {
                float q0 = __bfloat162float(sQ[tid * 128 + d]);
                float q1 = __bfloat162float(sQ[tid * 128 + d + 1]);
                float q2 = __bfloat162float(sQ[tid * 128 + d + 2]);
                float q3 = __bfloat162float(sQ[tid * 128 + d + 3]);

                float k0 = __bfloat162float(sK[k * 128 + d]);
                float k1 = __bfloat162float(sK[k * 128 + d + 1]);
                float k2 = __bfloat162float(sK[k * 128 + d + 2]);
                float k3 = __bfloat162float(sK[k * 128 + d + 3]);

                s += q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3;
            }
            s *= inv_sqrt_d;

            // Causal mask
            if (q_abs >= S || k_abs_local > q_abs) {
                local_s[k] = -FLT_MAX;
            } else {
                local_s[k] = s;
            }
        }

        // Find new max for this row
        float new_max = -FLT_MAX;
        for (int k = 0; k < BN; k++) {
            if (local_s[k] > new_max) new_max = local_s[k];
        }

        // Rescale if new max exceeds old max
        if (new_max > m && m > -FLT_MAX) {
            float alpha = expf(m - new_max);
            for (int d = 0; d < 128; d++) {
                o_acc[tid * 128 + d] *= alpha;
            }
            l *= alpha;
            m = new_max;
        } else if (new_max > m) {
            m = new_max;
        }

        // Accumulate weighted V into output
        for (int k = 0; k < BN; k++) {
            float sv = local_s[k];
            if (sv == -FLT_MAX) continue;

            float w = expf(sv - m);
            l += w;

            for (int d = 0; d < 128; d++) {
                o_acc[tid * 128 + d] += w * __bfloat162float(sV[k * 128 + d]);
            }
        }

        __syncthreads();
    }

    // Epilogue: normalize and write
    if (q_abs < S) {
        if (l > 0.0f) {
            float inv_l = 1.0f / l;
            LSE_base[q_abs] = m + logf(l);
            for (int d = 0; d < 128; d++) {
                O_base[(size_t)q_abs * 128 + d] = __float2bfloat16(o_acc[tid * 128 + d] * inv_l);
            }
        } else {
            LSE_base[q_abs] = -FLT_MAX;
            for (int d = 0; d < 128; d++) {
                O_base[(size_t)q_abs * 128 + d] = __float2bfloat16(0.0f);
            }
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    // D is hardcoded as 128

    void* q_ptr = (void*)Q.data_ptr();
    void* k_ptr = (void*)K.data_ptr();
    void* v_ptr = (void*)V.data_ptr();
    void* o_ptr = (void*)O.data_ptr();
    void* lse_ptr = (void*)LSE.data_ptr();

    // Zero LSE output
    size_t lse_bytes = B * H * S * sizeof(float);
    CUDA_CHECK(cudaMemsetAsync(lse_ptr, 0, lse_bytes));

    int q_blocks_per_bh = (S + BM - 1) / BM;
    int total_blocks = (int)B * (int)H * q_blocks_per_bh;

    dim3 grid(total_blocks);
    dim3 block(NT);

    // Shared memory: sQ(BM*128*2) + sK(BN*128*2) + sV(BN*128*2) + o_acc(BM*128*4)
    size_t smem_size = BM * 128 * sizeof(__nv_bfloat16)
                     + BN * 128 * sizeof(__nv_bfloat16)
                     + BN * 128 * sizeof(__nv_bfloat16)
                     + BM * 128 * sizeof(float);

    cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id);

    fa_fwd_kernel<<<grid, block, smem_size, stream>>>(
        static_cast<const __nv_bfloat16*>(q_ptr),
        static_cast<const __nv_bfloat16*>(k_ptr),
        static_cast<const __nv_bfloat16*>(v_ptr),
        static_cast<__nv_bfloat16*>(o_ptr),
        static_cast<float*>(lse_ptr),
        (int)S, (int)B, (int)H);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_d128_causal::run);

}  // namespace mha_d128_causal