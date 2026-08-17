#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t _e = (call); \
        if (_e != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
            exit(1); \
        } \
    } while(0)

namespace mha_bwd_ns {

static constexpr int HEAD_DIM = 128;
static constexpr int NTHREADS = 1024;
static constexpr int NWARPS = NTHREADS / 32;   // 32 warps per block
static constexpr int COLS_PER_THREAD = HEAD_DIM / 32; // 4 columns per thread

// Warp-level sum reduction: computes sum across all 32 lanes, broadcasts result to all lanes
__device__ __forceinline__ float warp_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return __shfl_sync(0xFFFFFFFF, val, 0);
}

// ================================================================
// Phase 1: Compute dQ and store m[qi] to global buffer
// ================================================================
// Each block handles NWARPS query rows for one (b,h) pair.
// Each warp handles one query row; 32 lanes split the head dimension.
// For each query row qi, iterate over all key rows kj <= qi (causal).
__global__ void mha_bwd_phase1(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    float* __restrict__ m_buf,
    int B, int H, int S, int D)
{
    int bh = blockIdx.y;
    if (bh >= B * H) return;
    int b = bh / H;
    int h = bh % H;

    int64_t base = (int64_t)(b * H + h) * S * D;
    int64_t l_off = (int64_t)(b * H + h) * S;

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // Each warp handles one query row
    int qi = blockIdx.x * NWARPS + warp_id;
    if (qi >= S) return;

    float scale = rsqrtf(static_cast<float>(D));
    float L_qi = L[l_off + qi];

    // Load Q[qi] and dO[qi] for this thread's columns
    float q_vals[COLS_PER_THREAD];
    float dO_vals[COLS_PER_THREAD];
    for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
        int col = lane_id * COLS_PER_THREAD + cd;
        q_vals[cd] = __bfloat162float(Q[base + qi * D + col]);
        dO_vals[cd] = __bfloat162float(dO[base + qi * D + col]);
    }

    // Per-thread accumulators (initialized to zero)
    float dq_vals[COLS_PER_THREAD];
    float ak_vals[COLS_PER_THREAD];
    for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
        dq_vals[cd] = 0.0f;
        ak_vals[cd] = 0.0f;
    }
    float m_sum = 0.0f;

    // Iterate over all valid key rows (causal: kj <= qi)
    for (int kj = 0; kj <= qi; kj++) {
        // Load K[kj] and V[kj] for this thread's columns
        float k_vals[COLS_PER_THREAD];
        float v_vals[COLS_PER_THREAD];
        for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
            int col = lane_id * COLS_PER_THREAD + cd;
            k_vals[cd] = __bfloat162float(K[base + kj * D + col]);
            v_vals[cd] = __bfloat162float(V[base + kj * D + col]);
        }

        // Per-thread partial dot products
        float score_tid = 0.0f;
        float dP_tid = 0.0f;
        #pragma unroll
        for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
            score_tid += q_vals[cd] * k_vals[cd];
            dP_tid += dO_vals[cd] * v_vals[cd];
        }

        // Reduce across warp to get full dot products
        float score_full = warp_sum(score_tid);
        float dP_full = warp_sum(dP_tid);

        // Compute attention weight and weighted dP
        float attn = expf(score_full * scale - L_qi);
        float adP = attn * dP_full;
        m_sum += adP;

        // Per-thread accumulation for dQ and ak
        #pragma unroll
        for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
            dq_vals[cd] += adP * k_vals[cd];
            ak_vals[cd] += attn * k_vals[cd];
        }
    }

    // Write dQ[qi][col] = dq_vals - m_sum * ak_vals (softmax gradient correction)
    for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
        int col = lane_id * COLS_PER_THREAD + cd;
        dQ[base + qi * D + col] = __float2bfloat16(dq_vals[cd] - m_sum * ak_vals[cd]);
    }

    // Store m[qi] (only one thread per warp needed since all threads agree)
    if (lane_id == 0) {
        m_buf[l_off + qi] = m_sum;
    }
}

// ================================================================
// Phase 2: Compute dK using precomputed m[qi]
// ================================================================
// Each block handles NWARPS key rows for one (b,h) pair.
// For each key row kj, iterate over all query rows qi >= kj (causal).
__global__ void mha_bwd_phase2(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ m_buf,
    __nv_bfloat16* __restrict__ dK,
    int B, int H, int S, int D)
{
    int bh = blockIdx.y;
    if (bh >= B * H) return;
    int b = bh / H;
    int h = bh % H;

    int64_t base = (int64_t)(b * H + h) * S * D;
    int64_t l_off = (int64_t)(b * H + h) * S;

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // Each warp handles one key row
    int kj = blockIdx.x * NWARPS + warp_id;
    if (kj >= S) return;

    float scale = rsqrtf(static_cast<float>(D));

    // Load K[kj] and V[kj] once (constant throughout qi loop)
    float k_vals[COLS_PER_THREAD];
    float v_vals[COLS_PER_THREAD];
    for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
        int col = lane_id * COLS_PER_THREAD + cd;
        k_vals[cd] = __bfloat162float(K[base + kj * D + col]);
        v_vals[cd] = __bfloat162float(V[base + kj * D + col]);
    }

    // Per-thread accumulator for dK (initialized to zero)
    float dk_vals[COLS_PER_THREAD];
    for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
        dk_vals[cd] = 0.0f;
    }

    // Iterate over all valid query rows (causal: qi >= kj)
    for (int qi = kj; qi < S; qi++) {
        // Load Q[qi] and dO[qi] for this thread's columns
        float q_vals[COLS_PER_THREAD];
        float dO_vals[COLS_PER_THREAD];
        for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
            int col = lane_id * COLS_PER_THREAD + cd;
            q_vals[cd] = __bfloat162float(Q[base + qi * D + col]);
            dO_vals[cd] = __bfloat162float(dO[base + qi * D + col]);
        }

        // Per-thread partial dot products
        float score_tid = 0.0f;
        float dP_tid = 0.0f;
        #pragma unroll
        for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
            score_tid += q_vals[cd] * k_vals[cd];
            dP_tid += dO_vals[cd] * v_vals[cd];
        }

        // Reduce across warp
        float score_full = warp_sum(score_tid);
        float dP_full = warp_sum(dP_tid);

        // Recompute attention and dScore
        float attn = expf(score_full * scale - L[l_off + qi]);
        float m_qi = m_buf[l_off + qi];
        float dScore = attn * (dP_full - m_qi);

        // Accumulate dK[kj][col] += dScore * Q[qi][col]
        #pragma unroll
        for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
            dk_vals[cd] += dScore * q_vals[cd];
        }
    }

    // Write dK[kj]
    for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
        int col = lane_id * COLS_PER_THREAD + cd;
        dK[base + kj * D + col] = __float2bfloat16(dk_vals[cd]);
    }
}

// ================================================================
// Phase 3: Compute dV
// ================================================================
// Simpler than dK: dV[kj][dd] = sum_{qi>=kj} attn[qi][kj] * dO[qi][dd]
// No m correction needed.
__global__ void mha_bwd_phase3(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dV,
    int B, int H, int S, int D)
{
    int bh = blockIdx.y;
    if (bh >= B * H) return;
    int b = bh / H;
    int h = bh % H;

    int64_t base = (int64_t)(b * H + h) * S * D;
    int64_t l_off = (int64_t)(b * H + h) * S;

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // Each warp handles one key row
    int kj = blockIdx.x * NWARPS + warp_id;
    if (kj >= S) return;

    float scale = rsqrtf(static_cast<float>(D));

    // Load K[kj] once (needed for score computation only)
    float k_vals[COLS_PER_THREAD];
    for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
        int col = lane_id * COLS_PER_THREAD + cd;
        k_vals[cd] = __bfloat162float(K[base + kj * D + col]);
    }

    // Per-thread accumulator for dV (initialized to zero)
    float dv_vals[COLS_PER_THREAD];
    for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
        dv_vals[cd] = 0.0f;
    }

    // Iterate over all valid query rows (causal: qi >= kj)
    for (int qi = kj; qi < S; qi++) {
        // Load Q[qi] and dO[qi] for this thread's columns
        float q_vals[COLS_PER_THREAD];
        float dO_vals[COLS_PER_THREAD];
        for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
            int col = lane_id * COLS_PER_THREAD + cd;
            q_vals[cd] = __bfloat162float(Q[base + qi * D + col]);
            dO_vals[cd] = __bfloat162float(dO[base + qi * D + col]);
        }

        // Per-thread partial score (only need score for attn, not dP)
        float score_tid = 0.0f;
        #pragma unroll
        for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
            score_tid += q_vals[cd] * k_vals[cd];
        }

        // Reduce across warp
        float score_full = warp_sum(score_tid);

        // Compute attention weight
        float attn = expf(score_full * scale - L[l_off + qi]);

        // Accumulate dV[kj][col] += attn * dO[qi][col]
        #pragma unroll
        for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
            dv_vals[cd] += attn * dO_vals[cd];
        }
    }

    // Write dV[kj]
    for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
        int col = lane_id * COLS_PER_THREAD + cd;
        dV[base + kj * D + col] = __float2bfloat16(dv_vals[cd]);
    }
}

// ================================================================
// Host-side runner function
// ================================================================
void run(tvm::ffi::TensorView Q_tv, tvm::ffi::TensorView K_tv, tvm::ffi::TensorView V_tv,
         tvm::ffi::TensorView O_tv, tvm::ffi::TensorView dO_tv, tvm::ffi::TensorView L_tv,
         tvm::ffi::TensorView dQ_tv, tvm::ffi::TensorView dK_tv, tvm::ffi::TensorView dV_tv) {
    CUDA_CHECK(cudaSetDevice(Q_tv.device().device_id));

    // Suppress unused parameter warning (O is not needed for backward)
    (void)O_tv;

    auto shape = Q_tv.shape();
    int B = static_cast<int>(shape[0]);
    int H = static_cast<int>(shape[1]);
    int S = static_cast<int>(shape[2]);
    int D = static_cast<int>(shape[3]);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q_tv.device().device_type, Q_tv.device().device_id));

    const __nv_bfloat16* Q  = static_cast<const __nv_bfloat16*>(Q_tv.data_ptr());
    const __nv_bfloat16* K  = static_cast<const __nv_bfloat16*>(K_tv.data_ptr());
    const __nv_bfloat16* V  = static_cast<const __nv_bfloat16*>(V_tv.data_ptr());
    const __nv_bfloat16* dO = static_cast<const __nv_bfloat16*>(dO_tv.data_ptr());
    const float* L          = static_cast<const float*>(L_tv.data_ptr());
    __nv_bfloat16* dQ       = static_cast<__nv_bfloat16*>(dQ_tv.data_ptr());
    __nv_bfloat16* dK       = static_cast<__nv_bfloat16*>(dK_tv.data_ptr());
    __nv_bfloat16* dV       = static_cast<__nv_bfloat16*>(dV_tv.data_ptr());

    // Allocate m buffer: stores m[qi] for each (b,h,seq) triple
    size_t m_size = static_cast<size_t>(B) * H * S * sizeof(float);
    float* m_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&m_buf, m_size));

    int n_bh = B * H;
    int n_row_blocks = (S + NWARPS - 1) / NWARPS;
    dim3 grid(n_row_blocks, n_bh);
    dim3 block(NTHREADS);

    // Phase 1: Compute dQ and store m[qi]
    mha_bwd_phase1<<<grid, block, 0, stream>>>(Q, K, V, dO, L, dQ, m_buf, B, H, S, D);
    CUDA_CHECK(cudaGetLastError());

    // Phase 2: Compute dK (uses m from phase 1)
    mha_bwd_phase2<<<grid, block, 0, stream>>>(Q, K, V, dO, L, m_buf, dK, B, H, S, D);
    CUDA_CHECK(cudaGetLastError());

    // Phase 3: Compute dV
    mha_bwd_phase3<<<grid, block, 0, stream>>>(Q, K, dO, L, dV, B, H, S, D);
    CUDA_CHECK(cudaGetLastError());

    // Synchronize and clean up
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(m_buf));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_ns::run);

}  // namespace