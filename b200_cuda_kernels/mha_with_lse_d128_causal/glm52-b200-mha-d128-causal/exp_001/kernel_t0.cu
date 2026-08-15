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

    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* smem_Q = smem;
    __nv_bfloat16* smem_K = smem + BM * D;
    __nv_bfloat16* smem_V = smem + (BM + BN) * D;

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
    __syncthreads();

    // Online softmax state (per query row)
    float m_val = -INFINITY;
    float l_val = 0.0f;
    float o_acc[D];

    #pragma unroll
    for (int d = 0; d < D; d++) {
        o_acc[d] = 0.0f;
    }

    int k_max = min(q_start + BM, S);

    for (int k_start = 0; k_start < k_max; k_start += BN) {
        int k_end = min(k_start + BN, S);
        int k_count = k_end - k_start;

        // Cooperative load K and V tiles
        if (tid < k_count) {
            for (int d = 0; d < D; d += 8) {
                *reinterpret_cast<int4*>(&smem_K[tid * D + d]) =
                    *reinterpret_cast<const int4*>(&K[bh_offset + (int64_t)(k_start + tid) * D + d]);
                *reinterpret_cast<int4*>(&smem_V[tid * D + d]) =
                    *reinterpret_cast<const int4*>(&V[bh_offset + (int64_t)(k_start + tid) * D + d]);
            }
        } else if (tid < BN) {
            for (int d = 0; d < D; d += 8) {
                *reinterpret_cast<int4*>(&smem_K[tid * D + d]) = make_int4(0, 0, 0, 0);
                *reinterpret_cast<int4*>(&smem_V[tid * D + d]) = make_int4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        // Process keys for this thread's query row (causal mask)
        if (valid) {
            int j_max = min(q_row - k_start + 1, BN);
            if (j_max > 0) {
                for (int j = 0; j < j_max; j++) {
                    // Compute attention score s = Q[q_row] . K[j] * SCALE
                    float s = 0.0f;
                    __nv_bfloat16* q_ptr = &smem_Q[tid * D];
                    __nv_bfloat16* k_ptr = &smem_K[j * D];

                    #pragma unroll 4
                    for (int d = 0; d < D; d += 2) {
                        float2 qf = __bfloat1622float2(
                            *reinterpret_cast<__nv_bfloat162*>(&q_ptr[d]));
                        float2 kf = __bfloat1622float2(
                            *reinterpret_cast<__nv_bfloat162*>(&k_ptr[d]));
                        s = __fmaf_rn(qf.x, kf.x, s);
                        s = __fmaf_rn(qf.y, kf.y, s);
                    }
                    s *= SCALE;

                    // Online softmax update
                    float m_new = fmaxf(m_val, s);
                    float alpha = __expf(m_val - m_new);
                    float p = __expf(s - m_new);
                    l_val = l_val * alpha + p;

                    // Accumulate O += p * V[j]
                    __nv_bfloat16* v_ptr = &smem_V[j * D];
                    #pragma unroll 4
                    for (int d = 0; d < D; d += 2) {
                        float2 vf = __bfloat1622float2(
                            *reinterpret_cast<__nv_bfloat162*>(&v_ptr[d]));
                        o_acc[d]     = o_acc[d]     * alpha + p * vf.x;
                        o_acc[d + 1] = o_acc[d + 1] * alpha + p * vf.y;
                    }
                    m_val = m_new;
                }
            }
        }
        __syncthreads();
    }

    // Store output
    if (valid) {
        float inv_l = 1.0f / l_val;
        __nv_bfloat16* o_ptr = &O[bh_offset + (int64_t)q_row * D];
        for (int d = 0; d < D; d += 2) {
            o_ptr[d]     = __float2bfloat16(o_acc[d] * inv_l);
            o_ptr[d + 1] = __float2bfloat16(o_acc[d + 1] * inv_l);
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
    int smem_size = (BM + 2 * BN) * D_val * (int)sizeof(__nv_bfloat16);

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