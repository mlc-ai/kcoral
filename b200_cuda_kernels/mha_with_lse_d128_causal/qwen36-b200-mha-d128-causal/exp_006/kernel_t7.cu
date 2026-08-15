#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
#include <float.h>
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

namespace mha_causal_impl {

static constexpr int HEAD_DIM = 128;
static constexpr int BM = 64;
static constexpr int BN = 64;
static constexpr float INV_SQRT_D = 0.08838834764831844f;
static constexpr int BLOCK_SIZE = 128;
static constexpr int THREADS_PER_ROW = BLOCK_SIZE / BM;  // 2 threads per query row

__global__ void mha_causal_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S, int B, int H)
{
    constexpr int D = HEAD_DIM;
    static_assert(BLOCK_SIZE == 128);
    static_assert(THREADS_PER_ROW == 2);
    constexpr float NEG_INF = -5e4f;

    // Shared memory layout:
    // sQ[BM][D], sK[BN][D], sV[BN][D]
    // Total: (64+64+64)*128*2 = 49152 bytes = 48 KB
    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* sQ  = smem;
    __nv_bfloat16* sK  = smem + BM * D;
    __nv_bfloat16* sV  = smem + (BM + BN) * D;

    int b = blockIdx.y / H;
    int h = blockIdx.y % H;
    int bm_start = blockIdx.x * BM;
    int tid = threadIdx.x;  // 0..127
    int lane_in_row = tid % THREADS_PER_ROW;  // which helper thread (0 or 1)
    int my_q_row = tid / THREADS_PER_ROW;     // 0..63
    
    int bh_offset = (b * H + h) * S * D;
    int eff_bm = min(BM, S - bm_start);
    bool active = (my_q_row < eff_bm);
    int q_global = bm_start + my_q_row;

    // Per-thread accumulators - much smaller now
    // Each of the 2 threads per row owns half of D dimensions
    int d_start = lane_in_row * (D / THREADS_PER_ROW);
    int d_end   = d_start + (D / THREADS_PER_ROW);
    
    float my_max = NEG_INF;
    float my_sum = 0.0f;
    
    // Small per-thread output accumulators for owned dims
    float acc[D / THREADS_PER_ROW];  // 64 floats = 256 bytes per thread
    for (int i = 0; i < (D / THREADS_PER_ROW); i++) acc[i] = 0.0f;

    // Load Q tile into shared memory once
    for (int idx = tid; idx < BM * D; idx += BLOCK_SIZE) {
        int r = idx / D;
        int c = idx % D;
        if (bm_start + r < S) {
            sQ[r * D + c] = Q[bh_offset + (bm_start + r) * D + c];
        } else {
            sQ[r * D + c] = __float2bfloat16(0.0f);
        }
    }
    __syncthreads();

    if (!active) return;

    // Main loop over KV tiles
    for (int bn_idx = 0; bn_idx < S; bn_idx += BN) {
        int cur_bn = bn_idx;
        int eff_bn = min(BN, S - cur_bn);

        // Cooperative load K and V tiles into shared memory
        for (int idx = tid; idx < eff_bn * D; idx += BLOCK_SIZE) {
            int r = idx / D;
            int c = idx % D;
            sK[r * D + c] = K[bh_offset + (cur_bn + r) * D + c];
            sV[r * D + c] = V[bh_offset + (cur_bn + r) * D + c];
        }
        __syncthreads();

        // First pass: compute raw attention scores and find max
        float local_m = NEG_INF;
        bool any_valid = false;
        
        float scores[BN];
        for (int j = 0; j < eff_bn; j++) {
            int k_pos = cur_bn + j;
            if (k_pos > q_global) {
                scores[j] = NEG_INF;
                continue;
            }
            float sim = 0.0f;
            for (int d = d_start; d < d_end; d++) {
                sim += static_cast<float>(sQ[my_q_row * D + d]) * 
                       static_cast<float>(sK[j * D + d]);
            }
            // Reduce partial sums across the 2 threads per row
            sim += __shfl_down_sync(0xFFFFFFFF, sim, 1);  // sync within pair
            sim *= INV_SQRT_D;
            scores[j] = sim;
            if (sim > local_m) {
                local_m = sim;
                any_valid = true;
            }
        }

        if (!any_valid) {
            __syncthreads();
            continue;
        }

        // Warp-level reduction for local_m across the pair is already done via shfl
        // But we need per-row reduction across both lanes of the pair
        float row_max = local_m;
        row_max = max(row_max, __shfl_down_sync(0xFFFFFFFF, local_m, 1));
        local_m = row_max;

        // Second pass: exponentiate relative to local_m and accumulate PV
        float local_sum = 0.0f;
        for (int j = 0; j < eff_bn; j++) {
            int k_pos = cur_bn + j;
            if (k_pos > q_global) continue;
            
            float p = expf(scores[j] - local_m);
            local_sum += p;
            
            for (int d = d_start; d < d_end; d++) {
                acc[d - d_start] += p * static_cast<float>(sV[j * D + d]);
            }
        }

        // Update running statistics
        if (my_max <= NEG_INF + 1.0f) {
            my_max = local_m;
            my_sum = local_sum;
        } else {
            float ratio = expf(my_max - local_m);
            my_max = local_m;
            for (int i = 0; i < (D / THREADS_PER_ROW); i++) {
                acc[i] *= ratio;
            }
            my_sum = my_sum * ratio + local_sum;
        }

        __syncthreads();
    }

    // Write output: each thread writes its portion of the output vector
    if (my_sum <= 0.0f || isnan(my_sum)) {
        my_sum = 1e-30f;
    }
    float inv_sum = 1.0f / my_sum;
    float lse = my_max + logf(my_sum);

    int out_base = bh_offset + q_global * D;
    for (int i = 0; i < (D / THREADS_PER_ROW); i++) {
        O[out_base + d_start + i] = __float2bfloat16(acc[i] * inv_sum);
    }
    
    // Only one thread per row writes LSE
    if (lane_in_row == 0) {
        LSE[(b * H + h) * S + q_global] = lse;
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = static_cast<int>(Q.size(0));
    int H = static_cast<int>(Q.size(1));
    int S = static_cast<int>(Q.size(2));

    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_ptr = static_cast<float*>(LSE.data_ptr());

    dim3 block(BLOCK_SIZE);
    dim3 grid((S + BM - 1) / BM, B * H);
    int smem_size = (BM + BN + BN) * HEAD_DIM * sizeof(__nv_bfloat16);

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    mha_causal_kernel<<<grid, block, smem_size, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr, S, B, H);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_causal_impl::run);

}  // namespace mha_causal_impl