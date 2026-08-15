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

namespace mha_kernel {

constexpr int D = 128;
constexpr int BK = 64;
constexpr int BQ = 8;
constexpr int THREADS = 256;  // 8 warps, each warp handles one query row

__global__ void attn_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int H, int S)
{
    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;
    int qb = blockIdx.y;
    int q_start = qb * BQ;

    int64_t off = (int64_t)(b * H + h) * (int64_t)S * D;
    const __nv_bfloat16* Qg = Q + off;
    const __nv_bfloat16* Kg = K + off;
    const __nv_bfloat16* Vg = V + off;
    __nv_bfloat16* Og = O + off;
    float* LSEg = LSE + (int64_t)(b * H + h) * (int64_t)S;

    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;
    int q_global = q_start + warp_id;

    extern __shared__ char smem_buf[];
    __nv_bfloat16* sK = reinterpret_cast<__nv_bfloat16*>(smem_buf);
    __nv_bfloat16* sV = sK + BK * D;

    const float scale = 0.08838834764831845f;  // 1/sqrt(128)

    // Load Q row: each lane handles 4 consecutive D elements
    float q[4];
    if (q_global < S) {
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            float2 qf = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(&Qg[(int64_t)q_global * D + lane * 4 + i * 2]));
            q[i * 2]     = qf.x;
            q[i * 2 + 1] = qf.y;
        }
    } else {
        #pragma unroll
        for (int i = 0; i < 4; i++) q[i] = 0.f;
    }

    float o[4] = {0.f, 0.f, 0.f, 0.f};
    float m_row = -INFINITY;
    float l_row = 0.f;

    int block_max_q = (q_start + BQ - 1 < S - 1) ? (q_start + BQ - 1) : (S - 1);
    if (block_max_q < 0) block_max_q = 0;
    int block_last_kb = block_max_q / BK;

    for (int kb = 0; kb <= block_last_kb; kb++) {
        int k_start = kb * BK;
        int k_end = (k_start + BK < S) ? (k_start + BK) : S;
        int k_len = k_end - k_start;

        // Cooperative K/V tile load (all 256 threads)
        for (int i = tid; i < BK * D; i += THREADS) {
            int r = i / D, d = i % D;
            int gr = k_start + r;
            if (gr < S) {
                sK[i] = Kg[(int64_t)gr * D + d];
                sV[i] = Vg[(int64_t)gr * D + d];
            } else {
                sK[i] = __float2bfloat16(0.f);
                sV[i] = __float2bfloat16(0.f);
            }
        }
        __syncthreads();

        // Process each key in tile (per-warp independent)
        if (q_global < S) {
            for (int k = 0; k < k_len; k++) {
                int k_global = k_start + k;
                if (k_global > q_global) break;  // causal mask

                // Dot product: Q[q] . K[k] via warp reduction
                float acc = 0.f;
                #pragma unroll
                for (int i = 0; i < 2; i++) {
                    float2 kf = __bfloat1622float2(
                        *reinterpret_cast<const __nv_bfloat162*>(&sK[k * D + lane * 4 + i * 2]));
                    acc += q[i * 2] * kf.x + q[i * 2 + 1] * kf.y;
                }
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    acc += __shfl_xor_sync(0xffffffff, acc, offset);

                float score = acc * scale;

                // Online softmax: update running max, rescale, accumulate
                if (score > m_row) {
                    float alpha = (m_row == -INFINITY) ? 0.f : __expf(m_row - score);
                    #pragma unroll
                    for (int i = 0; i < 4; i++) o[i] *= alpha;
                    l_row *= alpha;
                    m_row = score;
                }
                float p = __expf(score - m_row);
                l_row += p;

                // O += p * V[k]
                #pragma unroll
                for (int i = 0; i < 2; i++) {
                    float2 vf = __bfloat1622float2(
                        *reinterpret_cast<const __nv_bfloat162*>(&sV[k * D + lane * 4 + i * 2]));
                    o[i * 2]     += p * vf.x;
                    o[i * 2 + 1] += p * vf.y;
                }
            }
        }
        __syncthreads();
    }

    // Write output
    if (q_global < S) {
        float inv_l = 1.f / l_row;
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            Og[(int64_t)q_global * D + lane * 4 + i * 2]     = __float2bfloat16(o[i * 2] * inv_l);
            Og[(int64_t)q_global * D + lane * 4 + i * 2 + 1] = __float2bfloat16(o[i * 2 + 1] * inv_l);
        }
        if (lane == 0) {
            LSEg[q_global] = m_row + logf(l_row);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    const int B = 4;
    const int H = 48;
    int S = (int)Q.size(2);

    const __nv_bfloat16* Qd = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* Kd = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* Vd = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* Od = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSEd = static_cast<float*>(LSE.data_ptr());

    int smem_bytes = (int)(BK * D * sizeof(__nv_bfloat16) * 2);  // 32KB

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(
        attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    dim3 grid(B * H, (S + BQ - 1) / BQ);
    dim3 block(THREADS);
    attn_kernel<<<grid, block, smem_bytes, stream>>>(Qd, Kd, Vd, Od, LSEd, H, S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel::run);

}  // namespace mha_kernel