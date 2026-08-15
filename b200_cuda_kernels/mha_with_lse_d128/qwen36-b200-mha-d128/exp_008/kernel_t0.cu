#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_impl {

constexpr int HDRM = 128;
constexpr int TILE_S = 64;
constexpr int NUM_Q = 16;
constexpr int THREADS = 256;

__global__ void mha_kernel(
    const __nv_bfloat16* __restrict__ Q_in,
    const __nv_bfloat16* __restrict__ K_in,
    const __nv_bfloat16* __restrict__ V_in,
    __nv_bfloat16* __restrict__ O_out,
    float* __restrict__ LSE_out,
    int B, int H, int S, int D)
{
    int bh = blockIdx.x;
    if (bh >= B * H) return;

    int b_id = bh / H;
    int h_id = bh % H;

    uint32_t tid = threadIdx.x;
    uint32_t nt = blockDim.x;

    int stride_s = D;
    int stride_h = S * D;
    int stride_b = H * S * D;

    const __nv_bfloat16* Q_base = Q_in + b_id * stride_b + h_id * stride_h;
    const __nv_bfloat16* K_base = K_in + b_id * stride_b + h_id * stride_h;
    const __nv_bfloat16* V_base = V_in + b_id * stride_b + h_id * stride_h;
    __nv_bfloat16* O_base = O_out + b_id * stride_b + h_id * stride_h;
    float* LSE_base = LSE_out + b_id * (H * S) + h_id * S;

    float inv_sqrt_D = rsqrtf(static_cast<float>(D));

    __shared__ __nv_bfloat16 s_k[TILE_S][HDRM];
    __shared__ __nv_bfloat16 s_v[TILE_S][HDRM];
    __shared__ float s_scores[NUM_Q][TILE_S];

    __nv_bfloat16 q_reg[NUM_Q][HDRM];

    // Load Q into registers
    #pragma unroll
    for (int qq = 0; qq < NUM_Q; ++qq) {
        const __nv_bfloat16* ptr = Q_base + qq * stride_s;
        #pragma unroll
        for (int d = tid; d < HDRM; d += nt) {
            q_reg[qq][d] = ptr[d];
        }
    }
    __syncthreads();

    // Per-Q-row softmax state
    float max_log[NUM_Q];
    float sum_exp[NUM_Q];
    #pragma unroll
    for (int qq = 0; qq < NUM_Q; ++qq) {
        max_log[qq] = -1e20f;
        sum_exp[qq] = 0.f;
    }

    // Per-Q-row output accumulator: each thread owns DIM_PER_THREAD elements
    constexpr int DIM_PER_THREAD = HDRM / (nt / 4);  // = 4
    constexpr int NUM_DIM_GROUPS = nt / 4;           // = 64 groups
    float out_acc[NUM_Q][DIM_PER_THREAD];
    #pragma unroll
    for (int qq = 0; qq < NUM_Q; ++qq) {
        #pragma unroll
        for (int ii = 0; ii < DIM_PER_THREAD; ++ii) {
            out_acc[qq][ii] = 0.f;
        }
    }

    int num_segs = (S + TILE_S - 1) / TILE_S;

    for (int si = 0; si < num_segs; ++si) {
        int ks_start = si * TILE_S;
        int seg_len = min(ks_start + TILE_S, S) - ks_start;

        // Load K tile
        #pragma unroll
        for (int d = tid; d < HDRM; d += nt) {
            #pragma unroll
            for (int kk = 0; kk < seg_len; ++kk) {
                s_k[kk][d] = K_base[(ks_start + kk) * stride_s + d];
            }
        }

        // Load V tile
        #pragma unroll
        for (int d = tid; d < HDRM; d += nt) {
            #pragma unroll
            for (int kk = 0; kk < seg_len; ++kk) {
                s_v[kk][d] = V_base[(ks_start + kk) * stride_s + d];
            }
        }
        __syncthreads();

        // Q @ K^T => scores
        #pragma unroll
        for (int qq = 0; qq < NUM_Q; ++qq) {
            #pragma unroll
            for (int kk = 0; kk < seg_len; ++kk) {
                float acc = 0.f;
                #pragma unroll
                for (int d = 0; d < HDRM; ++d) {
                    acc += static_cast<float>(q_reg[qq][d]) *
                           static_cast<float>(s_k[kk][d]);
                }
                s_scores[qq][kk] = acc * inv_sqrt_D;
            }
        }
        __syncthreads();

        // Local max per Q row
        float loc_max[NUM_Q];
        #pragma unroll
        for (int qq = 0; qq < NUM_Q; ++qq) {
            float mx = -1e20f;
            #pragma unroll
            for (int kk = 0; kk < seg_len; ++kk) {
                float v = s_scores[qq][kk];
                if (v > mx) mx = v;
            }
            loc_max[qq] = mx;
        }

        // Online softmax merge + accumulate
        #pragma unroll
        for (int qq = 0; qq < NUM_Q; ++qq) {
            float old_max = max_log[qq];
            float new_max = loc_max[qq];
            if (new_max > old_max) {
                sum_exp[qq] *= expf(old_max - new_max);
                max_log[qq] = new_max;
            }
            #pragma unroll
            for (int kk = 0; kk < seg_len; ++kk) {
                sum_exp[qq] += expf(s_scores[qq][kk] - max_log[qq]);
            }

            // Accumulate output: O[qq][d] += softmax_weight * V[kk][d]
            float denom_inv = 1.f / sum_exp[qq];
            #pragma unroll
            for (int kk = 0; kk < seg_len; ++kk) {
                float w = expf(s_scores[qq][kk] - max_log[qq]) * denom_inv;
                #pragma unroll
                for (int dg = 0; dg < NUM_DIM_GROUPS; ++dg) {
                    int d = tid + dg * nt;
                    if (d < HDRM) {
                        // Map d to our local DIM_PER_THREAD slot
                        int slot = (d - tid) / nt;
                        if (slot < DIM_PER_THREAD) {
                            out_acc[qq][slot] += w * static_cast<float>(s_v[kk][d]);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    // Write output and LSE
    #pragma unroll
    for (int qq = 0; qq < NUM_Q; ++qq) {
        int qpos = qq;
        if (qpos < S) {
            LSE_base[qpos] = max_log[qq] + logf(sum_exp[qq]);

            __nv_bfloat16* ptr = O_base + qpos * stride_s;
            #pragma unroll
            for (int dg = 0; dg < NUM_DIM_GROUPS; ++dg) {
                int d = tid + dg * nt;
                if (d < HDRM) {
                    int slot = (d - tid) / nt;
                    if (slot < DIM_PER_THREAD) {
                        ptr[d] = __float2bfloat16(out_acc[qq][slot]);
                    }
                }
            }
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

    if (S == 0 || B == 0 || H == 0 || D != HDRM) {
        return;
    }

    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());

    dim3 grid(B * H);
    dim3 block(THREADS);

    int smem_bytes = 2 * TILE_S * HDRM * sizeof(__nv_bfloat16) +
                     NUM_Q * TILE_S * sizeof(float);

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    mha_kernel<<<grid, block, smem_bytes, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data, B, H, S, D);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace mha_impl

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_impl::run);