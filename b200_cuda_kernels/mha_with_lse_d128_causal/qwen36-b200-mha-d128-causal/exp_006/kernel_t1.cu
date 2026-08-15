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
static constexpr int BLOCK_SIZE = 256;
static constexpr float INV_SQRT_D = 0.08838834764831844f;

template <int BM_, int BN_, int BSZ>
__global__ void mha_causal_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S, int B, int H)
{
    static_assert(BSZ == 256 && BM_ == 64 && BN_ == 64);
    constexpr int D = HEAD_DIM;
    constexpr int NEG_INF = -1e20f;

    // Shared memory for K and V staging
    extern __shared__ __nv_bfloat16 shared_mem[];
    __nv_bfloat16* sK = shared_mem;
    __nv_bfloat16* sV = shared_mem + BN_ * D;

    const int b = blockIdx.y / H;
    const int h = blockIdx.y % H;
    const int bm_start = blockIdx.x * BM_;
    const int tid = threadIdx.x;

    const int bh_offset = (b * H + h) * S * D;

    // Per-thread accumulators: each thread handles multiple (q, j) combinations
    // With 256 threads over BM*BN = 4096 combos => 16 combos per thread
    constexpr int COMBOS_PER_THREAD = (BM_ * BN_) / BSZ;  // 16

    // Output accumulators per thread
    float acc_out[COMBOS_PER_THREAD][D];
    float acc_max[COMBOS_PER_THREAD];
    float acc_sum[COMBOS_PER_THREAD];

    // Initialize accumulators
    for (int c = 0; c < COMBOS_PER_THREAD; c++) {
        acc_max[c] = NEG_INF;
        acc_sum[c] = 0.0f;
        for (int d = 0; d < D; d++) {
            acc_out[c][d] = 0.0f;
        }
    }

    // Effective BM for last tile
    int eff_bm = min(BM_, S - bm_start);
    if (eff_bm <= 0) return;

    // Precompute which (q_row, q_col_d) this thread owns in a fixed mapping
    // Thread tid maps to specific Q rows. With 256 threads and BM=64, 
    // each Q row is handled by 4 threads for D=128 (each does 32 dims).
    // But for simplicity, let's do: each thread processes (COMBOS_PER_THREAD) different (q_row, k_row) pairs
    // accumulating D-dimensional dot products serially.

    // Map thread -> list of query row indices it is responsible for scoring
    // Strategy: round-robin assignment of q_rows among threads for each bn tile
    // For each (bn_idx) tile, thread t computes contribution to output for certain q_rows
    
    // Simpler mapping: thread handles stride-based distribution
    // For each bn tile, we'll iterate q_rows assigned to this thread
    int q_rows_per_thread = (BM_ + BSZ - 1) / BSZ;  // ceil(64/256) = 1
    // Most threads won't even have a full row! Only 64 threads needed for 64 query rows.
    
    // Better approach: use fewer threads per block matching BM
    // Re-do: each thread handles 1 or more query rows completely
    // With 128 threads, 64 rows: warp 0 -> rows 0-1, each lane handles 1 row? No...
    
    // Cleanest: BLOCK_SIZE = BM = 64, each thread owns exactly 1 query row
    // D=128 dim accumulation per row, BN-key loops in shared memory
    
    // Let's redesign with BLOCK_SIZE = 64
    
    // Actually, since the template forces BSZ=256, let me work with it:
    // - 256 threads, 64 query rows: each row handled by 4 threads
    // - For softmax (per-row state), only 1 thread per row tracks max/sum
    // - For MMA, all 4 threads contribute partial sums
    
    // Row index owned primarily by thread group: row_group[tid/4]
    int my_q_row = tid % BM_;
    bool active = (my_q_row < eff_bm);

    if (!active) {
        // Even inactive threads need to participate in shared memory loads
        // Fall through to shared memory ops below
    }

    // Local buffers for active threads
    float my_row_max = NEG_INF;
    float my_row_sum = 0.0f;
    float my_row_out[D];
    for (int d = 0; d < D; d++) my_row_out[d] = 0.0f;

    // Loop over BN tiles
    for (int bn_idx = 0; bn_idx < S; bn_idx += BN_) {
        int cur_bn = bn_idx;
        int eff_bn = min(BN_, S - cur_bn);

        // Cooperative load K tile
        for (int idx = tid; idx < eff_bn * D; idx += BSZ) {
            int j = idx / D;
            int d = idx % D;
            sK[idx] = K[bh_offset + (cur_bn + j) * D + d];
        }
        // Cooperative load V tile  
        for (int idx = tid; idx < eff_bn * D; idx += BSZ) {
            int j = idx / D;
            int d = idx % D;
            sV[idx] = V[bh_offset + (cur_bn + j) * D + d];
        }
        __syncthreads();

        if (!active) continue;

        int q_pos_global = bm_start + my_q_row;

        // Compute attention scores for this Q row against loaded K tile
        float local_new_m = NEG_INF;
        
        // First pass: compute scores and find new max
        __attribute__((aligned(16))) float scores[BN_];
        
        for (int j = 0; j < eff_bn; j++) {
            int k_pos = cur_bn + j;
            if (k_pos > q_pos_global) {
                scores[j] = NEG_INF;
                continue;
            }
            float sim = 0.0f;
            for (int d = 0; d < D; d++) {
                float q_val = static_cast<float>(Q[bh_offset + q_pos_global * D + d]);
                float k_val = static_cast<float>(sK[j * D + d]);
                sim += q_val * k_val;
            }
            sim *= INV_SQRT_D;
            scores[j] = sim;
            if (sim > local_new_m) local_new_m = sim;
        }

        // Update global max and scale previous outputs
        float old_m = my_row_max;
        float old_s = my_row_sum;

        // Warp-reduce max (only needed within the 4 threads sharing a row, but using full warp is fine)
        for (int offset = 16; offset > 0; offset /= 2) {
            float other = __shfl_down_sync(0xFFFFFFFF, local_new_m, offset);
            if (other > local_new_m) local_new_m = other;
        }

        float new_m = local_new_m;

        // Compute exp(scores - new_m) and accumulate
        float local_sum = 0.0f;
        for (int j = 0; j < eff_bn; j++) {
            if (scores[j] == NEG_INF) {
                scores[j] = 0.0f;
                continue;
            }
            scores[j] = expf(scores[j] - new_m);
            local_sum += scores[j];
            
            // Accumulate: out += score * V
            for (int d = 0; d < D; d++) {
                my_row_out[d] += scores[j] * static_cast<float>(sV[j * D + d]);
            }
        }

        // Scale previous outputs
        float ratio = expf(old_m - new_m);
        for (int d = 0; d < D; d++) {
            my_row_out[d] *= ratio;
        }

        // Update accumulators
        my_row_max = new_m;
        my_row_sum = old_s * ratio + local_sum;

        __syncthreads();
    }

    // Write output
    if (!active) return;

    int q_pos_global = bm_start + my_q_row;
    float denom = 1.0f / my_row_sum;
    float lse_val = my_row_max + logf(my_row_sum);

    int out_base = bh_offset + q_pos_global * D;
    for (int d = 0; d < D; d++) {
        O[out_base + d] = __float2bfloat16(my_row_out[d] * denom);
    }
    LSE[(b * H + h) * S + q_pos_global] = lse_val;
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
    int smem_size = 2 * BN * HEAD_DIM * sizeof(__nv_bfloat16);

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    mha_causal_kernel<BM, BN, BLOCK_SIZE><<<grid, block, smem_size, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr, S, B, H);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_causal_impl::run);

}  // namespace mha_causal_impl