#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <math.h>
#include <float.h>
#include <cstdint>
#include <stdio.h>
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

namespace flash_mha_d128 {

static constexpr uint32_t BM = 64;
static constexpr uint32_t BN = 64;
static constexpr uint32_t NT = 64;
static constexpr uint32_t DVAL = 128;

__global__ void flash_mha_kernel(
    const __nv_bfloat16* __restrict__ Q_g,
    const __nv_bfloat16* __restrict__ K_g,
    const __nv_bfloat16* __restrict__ V_g,
    __nv_bfloat16* __restrict__ O_g,
    float* __restrict__ LSE_g,
    int B, int H, int S, int D,
    float inv_sqrt_D,
    int64_t stride_Q_B, int64_t stride_Q_H, int64_t stride_Q_S, int64_t stride_Q_D,
    int64_t stride_K_B, int64_t stride_K_H, int64_t stride_K_S, int64_t stride_K_D,
    int64_t stride_V_B, int64_t stride_V_H, int64_t stride_V_S, int64_t stride_V_D,
    int64_t stride_O_B, int64_t stride_O_H, int64_t stride_O_S, int64_t stride_O_D,
    int64_t stride_LSE_B, int64_t stride_LSE_H, int64_t stride_LSE_S
) {
    // Shared memory: sQ[BM][D], sK[BN][D], sV[BN][D]
    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* sQ  = smem;
    __nv_bfloat16* sK  = smem + BM * DVAL;
    __nv_bfloat16* sV  = smem + (BM + BN) * DVAL;

    // Decode block assignment
    int nqblocks = (S + BM - 1) / BM;
    int bh = blockIdx.x / nqblocks;
    int b = bh / H;
    int h = bh % H;
    int qb = blockIdx.x % nqblocks;
    int q_base = qb * BM;
    int tid = threadIdx.x;
    int my_q = tid;
    bool valid = (my_q < BM && q_base + my_q < S);

    // Offsets for this (b, h) pair
    int64_t bq_off = (int64_t)b * stride_Q_B + h * stride_Q_H + q_base * stride_Q_S;
    int64_t bk_off = (int64_t)b * stride_K_B + h * stride_K_H;
    int64_t bv_off = (int64_t)b * stride_V_B + h * stride_V_H;
    int64_t bo_off = (int64_t)b * stride_O_B + h * stride_O_H + q_base * stride_O_S;
    int64_t bl_off = (int64_t)b * stride_LSE_B + h * stride_LSE_H + q_base * stride_LSE_S;

    // Phase 1: Load Q tile into shared memory
    if (valid) {
        #pragma unroll
        for (int d = 0; d < DVAL / 2; d++) {
            int col = d * 2;
            int64_t idx0 = bq_off + my_q * stride_Q_S + col * stride_Q_D;
            int64_t idx1 = bq_off + my_q * stride_Q_S + (col + 1) * stride_Q_D;
            sQ[my_q * DVAL + col]     = Q_g[idx0];
            sQ[my_q * DVAL + col + 1] = Q_g[idx1];
        }
    }
    __syncthreads();

    // Initialize online softmax state
    float row_max = -FLT_MAX;
    float row_sum = 1.0f;
    float o[DVAL];
    #pragma unroll
    for (int d = 0; d < DVAL; d++) o[d] = 0.0f;

    // Main loop over KV tiles
    int nktiles = (S + BN - 1) / BN;
    for (int kt = 0; kt < nktiles; kt++) {
        int k_base = kt * BN;

        // === Load K tile === (ALL threads participate)
        int64_t kb_off = bk_off + k_base * stride_K_S;
        int kr = tid;
        if (kr < BN) {
            #pragma unroll
            for (int d = 0; d < DVAL / 2; d++) {
                int col = d * 2;
                int64_t idx0 = kb_off + kr * stride_K_S + col * stride_K_D;
                int64_t idx1 = kb_off + kr * stride_K_S + (col + 1) * stride_K_D;
                sK[kr * DVAL + col]     = K_g[idx0];
                sK[kr * DVAL + col + 1] = K_g[idx1];
            }
        }

        // === Load V tile === (ALL threads participate)
        int64_t vb_off = bv_off + k_base * stride_V_S;
        int vr = tid;
        if (vr < BN) {
            #pragma unroll
            for (int d = 0; d < DVAL / 2; d++) {
                int col = d * 2;
                int64_t idx0 = vb_off + vr * stride_V_S + col * stride_V_D;
                int64_t idx1 = vb_off + vr * stride_V_S + (col + 1) * stride_V_D;
                sV[vr * DVAL + col]     = V_g[idx0];
                sV[vr * DVAL + col + 1] = V_g[idx1];
            }
        }
        __syncthreads();

        // Skip computation for invalid threads (still participated in sync above)
        if (!valid) continue;

        // === Pass 1: Compute dot products and find new max ===
        float m_new = -FLT_MAX;
        #pragma unroll
        for (int kr = 0; kr < BN; kr++) {
            float s = 0.0f;
            #pragma unroll
            for (int d = 0; d < DVAL; d++) {
                s += __bfloat162float(sQ[my_q * DVAL + d]) * __bfloat162float(sK[kr * DVAL + d]);
            }
            if (k_base + kr < S) {
                s *= inv_sqrt_D;
                if (s > m_new) m_new = s;
            }
        }

        // === Rescale previous output ===
        float alpha = expf(row_max - m_new);
        #pragma unroll
        for (int d = 0; d < DVAL; d++) {
            o[d] *= alpha;
        }
        float p_old_sum = row_sum * alpha;

        // === Pass 2: Compute softmax weights and accumulate output ===
        float p_new_sum = 0.0f;
        #pragma unroll
        for (int kr = 0; kr < BN; kr++) {
            float s = 0.0f;
            #pragma unroll
            for (int d = 0; d < DVAL; d++) {
                s += __bfloat162float(sQ[my_q * DVAL + d]) * __bfloat162float(sK[kr * DVAL + d]);
            }
            if (k_base + kr < S) {
                s *= inv_sqrt_D;
                float p = expf(s - m_new);
                p_new_sum += p;
                #pragma unroll
                for (int d = 0; d < DVAL; d++) {
                    o[d] += p * __bfloat162float(sV[kr * DVAL + d]);
                }
            }
        }

        row_sum = p_old_sum + p_new_sum;
        row_max = m_new;

        __syncthreads();
    }

    // === Epilogue: Normalize and write output ===
    if (valid) {
        float inv_lse = 1.0f / row_sum;
        float lse_out = row_max + logf(row_sum);

        #pragma unroll
        for (int d = 0; d < DVAL / 2; d++) {
            int col = d * 2;
            float f0 = o[col] * inv_lse;
            float f1 = o[col + 1] * inv_lse;
            int64_t o_idx0 = bo_off + my_q * stride_O_S + col * stride_O_D;
            int64_t o_idx1 = bo_off + my_q * stride_O_S + (col + 1) * stride_O_D;
            O_g[o_idx0] = __float2bfloat16(f0);
            O_g[o_idx1] = __float2bfloat16(f1);
        }

        LSE_g[bl_off + my_q * stride_LSE_S] = lse_out;
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);

    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());

    int64_t sqB = Q.stride(0), sqH = Q.stride(1), sqS = Q.stride(2), sqD = Q.stride(3);
    int64_t skB = K.stride(0), skH = K.stride(1), skS = K.stride(2), skD = K.stride(3);
    int64_t svB = V.stride(0), svH = V.stride(1), svS = V.stride(2), svD = V.stride(3);
    int64_t soB = O.stride(0), soH = O.stride(1), soS = O.stride(2), soD = O.stride(3);
    int64_t slB = LSE.stride(0), slH = LSE.stride(1), slS = LSE.stride(2);

    float inv_sqrt_D = 1.0f / sqrtf((float)D);

    int64_t nqblocks = (S + BM - 1) / BM;
    int64_t total_blocks = B * H * nqblocks;

    // Shared memory: (BM + 2*BN) * D * sizeof(bf16) = (64+128)*128*2 = 49152 bytes
    size_t smem_bytes = (BM + 2 * BN) * D * sizeof(__nv_bfloat16);

    dim3 grid((unsigned int)total_blocks);
    dim3 block(NT);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    flash_mha_kernel<<<grid, block, smem_bytes, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data,
        (int)B, (int)H, (int)S, (int)D,
        inv_sqrt_D,
        sqB, sqH, sqS, sqD,
        skB, skH, skS, skD,
        svB, svH, svS, svD,
        soB, soH, soS, soD,
        slB, slH, slS
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flash_mha_d128::run);

} // namespace flash_mha_d128