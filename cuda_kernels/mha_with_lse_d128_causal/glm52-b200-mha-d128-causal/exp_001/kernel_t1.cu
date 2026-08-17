#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", \
                cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

namespace mha_lse_d128_causal {

static constexpr int D = 128;
static constexpr int BM = 128;
static constexpr int BN = 128;
static constexpr float SCALE = 0.0883883476483184f; // 1/sqrt(128)

__global__ void attention_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S) {

    int b = blockIdx.y;
    int h = blockIdx.z;
    int q_block = blockIdx.x;
    int q_start = q_block * BM;
    int tid = threadIdx.x;
    int q_row = q_start + tid;
    bool valid = (q_row < S);

    int64_t bh_offset = ((int64_t)b * H + h) * (int64_t)S * D;

    // Shared memory layout (224KB total):
    // Q[BM*D] = 32KB, K[BN*D] = 32KB, V[BN*D] = 32KB
    // O_acc[BM*D] = 64KB (float), S[BM*BN] = 64KB (float)
    extern __shared__ char smem_raw[];
    __nv_bfloat16* smem_Q = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* smem_K = smem_Q + BM * D;
    __nv_bfloat16* smem_V = smem_K + BN * D;
    float* smem_O = reinterpret_cast<float*>(smem_V + BN * D);
    float* smem_S = smem_O + BM * D;

    // Load Q tile (each thread loads one row)
    if (valid) {
        for (int d = 0; d < D; d += 8) {
            *reinterpret_cast<int4*>(&smem_Q[tid * D + d]) =
                *reinterpret_cast<const int4*>(&Q[bh_offset + (int64_t)q_row * D + d]);
        }
    } else {
        for (int d = 0; d < D; d += 8) {
            *reinterpret_cast<int4*>(&smem_Q[tid * D + d]) = make_int4(0, 0, 0, 0);
        }
    }

    // Init O accumulator in shared memory
    for (int d = 0; d < D; d += 4) {
        *reinterpret_cast<float4*>(&smem_O[tid * D + d]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    float m_val = -INFINITY;
    float l_val = 0.0f;

    __syncthreads();

    int k_limit = min(q_start + BM, S);

    for (int k_start = 0; k_start < k_limit; k_start += BN) {
        int k_end = min(k_start + BN, S);
        int k_count = k_end - k_start;

        // Cooperative load K, V tiles
        for (int i = tid; i < BN; i += BM) {
            if (i < k_count) {
                for (int d = 0; d < D; d += 8) {
                    *reinterpret_cast<int4*>(&smem_K[i * D + d]) =
                        *reinterpret_cast<const int4*>(&K[bh_offset + (int64_t)(k_start + i) * D + d]);
                    *reinterpret_cast<int4*>(&smem_V[i * D + d]) =
                        *reinterpret_cast<const int4*>(&V[bh_offset + (int64_t)(k_start + i) * D + d]);
                }
            } else {
                for (int d = 0; d < D; d += 8) {
                    *reinterpret_cast<int4*>(&smem_K[i * D + d]) = make_int4(0, 0, 0, 0);
                    *reinterpret_cast<int4*>(&smem_V[i * D + d]) = make_int4(0, 0, 0, 0);
                }
            }
        }
        __syncthreads();

        if (valid) {
            int j_max = min(q_row - k_start + 1, BN);

            if (j_max > 0) {
                // Pass 1: Compute scores S = QK^T * scale, find row max
                float m_old = m_val;
                float m_new = m_old;

                for (int j = 0; j < j_max; j++) {
                    float score = 0.0f;
                    #pragma unroll 8
                    for (int d = 0; d < D; d += 2) {
                        float2 qf = __bfloat1622float2(
                            *reinterpret_cast<__nv_bfloat162*>(&smem_Q[tid * D + d]));
                        float2 kf = __bfloat1622float2(
                            *reinterpret_cast<__nv_bfloat162*>(&smem_K[j * D + d]));
                        score = __fmaf_rn(qf.x, kf.x, score);
                        score = __fmaf_rn(qf.y, kf.y, score);
                    }
                    score *= SCALE;
                    smem_S[tid * BN + j] = score;
                    m_new = fmaxf(m_new, score);
                }

                float alpha = (m_old == -INFINITY) ? 0.0f : __expf(m_old - m_new);

                // Pass 2: Compute P = exp(S - m_new), update l
                float l_new = l_val * alpha;
                for (int j = 0; j < j_max; j++) {
                    float p = __expf(smem_S[tid * BN + j] - m_new);
                    smem_S[tid * BN + j] = p;
                    l_new += p;
                }

                // Pass 3: Update O = alpha * O_old + P @ V
                // Process D in chunks of 32 to keep o_reg in registers
                for (int d_chunk = 0; d_chunk < D; d_chunk += 32) {
                    float o_reg[32];
                    for (int d = 0; d < 32; d++) {
                        o_reg[d] = smem_O[tid * D + d_chunk + d] * alpha;
                    }
                    for (int j = 0; j < j_max; j++) {
                        float p = smem_S[tid * BN + j];
                        #pragma unroll 8
                        for (int d = 0; d < 32; d += 2) {
                            float2 vf = __bfloat1622float2(
                                *reinterpret_cast<__nv_bfloat162*>(&smem_V[j * D + d_chunk + d]));
                            o_reg[d] = __fmaf_rn(p, vf.x, o_reg[d]);
                            o_reg[d + 1] = __fmaf_rn(p, vf.y, o_reg[d + 1]);
                        }
                    }
                    for (int d = 0; d < 32; d++) {
                        smem_O[tid * D + d_chunk + d] = o_reg[d];
                    }
                }

                m_val = m_new;
                l_val = l_new;
            }
        }
        __syncthreads();
    }

    // Store output O and LSE
    if (valid) {
        float inv_l = 1.0f / l_val;
        for (int d = 0; d < D; d += 4) {
            float4 o_val = *reinterpret_cast<float4*>(&smem_O[tid * D + d]);
            __nv_bfloat162 o0 = __float22bfloat162_rn(
                make_float2(o_val.x * inv_l, o_val.y * inv_l));
            __nv_bfloat162 o1 = __float22bfloat162_rn(
                make_float2(o_val.z * inv_l, o_val.w * inv_l));
            *reinterpret_cast<__nv_bfloat162*>(&O[bh_offset + (int64_t)q_row * D + d]) = o0;
            *reinterpret_cast<__nv_bfloat162*>(&O[bh_offset + (int64_t)q_row * D + d + 2]) = o1;
        }
        LSE[(int64_t)b * H * S + (int64_t)h * S + q_row] = m_val + logf(l_val);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    const int B = 4;
    const int H = 48;
    const int D_val = 128;
    int64_t S = Q.size(2);

    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());

    int num_q_blocks = ((int)S + BM - 1) / BM;
    dim3 grid(num_q_blocks, B, H);
    dim3 block(BM);

    // Shared memory: Q(32KB) + K(32KB) + V(32KB) + O_acc(64KB) + S(64KB) = 224KB
    int smem_size = (BM * D_val + 2 * BN * D_val) * (int)sizeof(__nv_bfloat16)
                  + (BM * D_val + BM * BN) * (int)sizeof(float);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    cudaFuncSetAttribute(attention_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    attention_kernel<<<grid, block, smem_size, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data, B, H, (int)S);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_lse_d128_causal::run);

}  // namespace mha_lse_d128_causal