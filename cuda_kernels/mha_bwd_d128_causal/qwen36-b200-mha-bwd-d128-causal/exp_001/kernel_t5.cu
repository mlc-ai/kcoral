#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>
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
static constexpr int COLS_PER_THREAD = HEAD_DIM / 32; // 4 cols per thread
static constexpr int BLOCK_K = 32;
static constexpr int BLOCK_Q = 32;

__device__ __forceinline__ float warp_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return __shfl_sync(0xFFFFFFFF, val, 0);
}

extern __shared__ __align__(16) unsigned char smem[];

// ================================================================
// Phase 1: Compute dQ and store m[qi]
// ================================================================
__global__ void mha_bwd_dQ(
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

    // Shared memory pointers for K and V tiles
    __nv_bfloat16* smem_K = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* smem_V = smem_K + BLOCK_K * D;

    int tid = threadIdx.x;
    int lane_id = tid % 32;

    // Block covers BLOCK_Q query rows
    int q_base = blockIdx.x * BLOCK_Q;

    // Thread assigned row within block (thread 0..NTHREADS-1 -> row 0..NTHREADS-1 in block)
    // Only BLOCK_Q threads map to valid rows
    int q_row_in_block = tid / 32; // Each warp handles one row

    if (q_row_in_block >= BLOCK_Q) return;

    int qi = q_base + q_row_in_block;
    if (qi >= S) return;

    float scale = rsqrtf(static_cast<float>(D));
    float L_qi = L[l_off + qi];

    const __nv_bfloat16* Q_row = Q + base + qi * D;
    const __nv_bfloat16* dO_row = dO + base + qi * D;
    __nv_bfloat16* dQ_row = dQ + base + qi * D;

    // Load Q[qi] and dO[qi]
    float q_vals[COLS_PER_THREAD] = {};
    float dO_vals[COLS_PER_THREAD] = {};
    for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
        int col = lane_id * COLS_PER_THREAD + cd;
        q_vals[cd] = __bfloat162float(Q_row[col]);
        dO_vals[cd] = __bfloat162float(dO_row[col]);
    }

    float dq_vals[COLS_PER_THREAD] = {};
    float ak_vals[COLS_PER_THREAD] = {};
    float m_sum = 0.0f;

    // Iterate over K-tiles
    for (int tj = 0; tj * BLOCK_K <= qi && tj * BLOCK_K < S; tj++) {
        int kj_base = tj * BLOCK_K;
        int kj_end = min(kj_base + BLOCK_K, S);
        int k_tile_size = kj_end - kj_base;

        // Cooperative load K tile into shared memory
        if (tid < k_tile_size * D) {
            int row = tid / D;
            int col = tid % D;
            smem_K[row * D + col] = K[base + (kj_base + row) * D + col];
        }

        // Cooperative load V tile into shared memory
        if (tid < k_tile_size * D) {
            int row = tid / D;
            int col = tid % D;
            smem_V[row * D + col] = V[base + (kj_base + row) * D + col];
        }
        __syncthreads();

        // Process each k-row in this tile
        for (int kr = 0; kr < k_tile_size; kr++) {
            int kj_actual = kj_base + kr;
            if (kj_actual > qi) continue; // Causal mask

            float k_vals[COLS_PER_THREAD];
            float v_vals[COLS_PER_THREAD];
            for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
                int col = lane_id * COLS_PER_THREAD + cd;
                k_vals[cd] = __bfloat162float(smem_K[kr * D + col]);
                v_vals[cd] = __bfloat162float(smem_V[kr * D + col]);
            }

            float score_tid = 0.0f;
            float dP_tid = 0.0f;
            #pragma unroll
            for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
                score_tid += q_vals[cd] * k_vals[cd];
                dP_tid += dO_vals[cd] * v_vals[cd];
            }

            float score_full = warp_sum(score_tid);
            float dP_full = warp_sum(dP_tid);

            float attn = expf(score_full * scale - L_qi);
            float adP = attn * dP_full;
            m_sum += adP;

            float scaled_adP = adP * scale;
            #pragma unroll
            for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
                dq_vals[cd] += scaled_adP * k_vals[cd];
                ak_vals[cd] += attn * scale * k_vals[cd];
            }
        }
        __syncthreads();
    }

    // Write final result
    for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
        int col = lane_id * COLS_PER_THREAD + cd;
        dQ_row[col] = __float2bfloat16(dq_vals[cd] - m_sum * ak_vals[cd]);
    }

    if (lane_id == 0) {
        m_buf[l_off + qi] = m_sum;
    }
}

// ================================================================
// Phase 2: Compute dK using precomputed m[qi]
// ================================================================
__global__ void mha_bwd_dK(
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

    __nv_bfloat16* smem_Q = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* smem_dO = smem_Q + BLOCK_K * D;

    int tid = threadIdx.x;
    int lane_id = tid % 32;

    int k_base = blockIdx.x * BLOCK_Q;
    int k_row_in_block = tid / 32;

    if (k_row_in_block >= BLOCK_Q) return;

    int kj = k_base + k_row_in_block;
    if (kj >= S) return;

    float scale = rsqrtf(static_cast<float>(D));

    // Load K[kj] and V[kj] once (constant for entire kernel execution)
    const __nv_bfloat16* Kj = K + base + kj * D;
    const __nv_bfloat16* Vj = V + base + kj * D;

    float k_vals[COLS_PER_THREAD];
    float v_vals[COLS_PER_THREAD];
    for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
        int col = lane_id * COLS_PER_THREAD + cd;
        k_vals[cd] = __bfloat162float(Kj[col]);
        v_vals[cd] = __bfloat162float(Vj[col]);
    }

    float dk_vals[COLS_PER_THREAD] = {};

    // Iterate over Q-tiles
    for (int ti = kj / BLOCK_K; ti * BLOCK_K < S; ti++) {
        int qi_base = ti * BLOCK_K;
        int qi_end = min(qi_base + BLOCK_K, S);
        int q_tile_size = qi_end - qi_base;

        // Skip if no valid query rows (all < kj)
        if (qi_end <= kj) continue;

        // Cooperative load Q tile
        if (tid < q_tile_size * D) {
            int row = tid / D;
            int col = tid % D;
            smem_Q[row * D + col] = Q[base + (qi_base + row) * D + col];
        }
        // Cooperative load dO tile
        if (tid < q_tile_size * D) {
            int row = tid / D;
            int col = tid % D;
            smem_dO[row * D + col] = dO[base + (qi_base + row) * D + col];
        }
        __syncthreads();

        for (int qr = 0; qr < q_tile_size; qr++) {
            int qi_actual = qi_base + qr;
            if (qi_actual < kj) continue; // Causal mask

            float q_vals[COLS_PER_THREAD];
            float dO_vals[COLS_PER_THREAD];
            for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
                int col = lane_id * COLS_PER_THREAD + cd;
                q_vals[cd] = __bfloat162float(smem_Q[qr * D + col]);
                dO_vals[cd] = __bfloat162float(smem_dO[qr * D + col]);
            }

            float score_tid = 0.0f;
            float dP_tid = 0.0f;
            #pragma unroll
            for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
                score_tid += q_vals[cd] * k_vals[cd];
                dP_tid += dO_vals[cd] * v_vals[cd];
            }

            float score_full = warp_sum(score_tid);
            float dP_full = warp_sum(dP_tid);

            float attn = expf(score_full * scale - L[l_off + qi_actual]);
            float m_qi = m_buf[l_off + qi_actual];
            float dScore = attn * (dP_full - m_qi) * scale;

            #pragma unroll
            for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
                dk_vals[cd] += dScore * q_vals[cd];
            }
        }
        __syncthreads();
    }

    __nv_bfloat16* dKj = dK + base + kj * D;
    for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
        int col = lane_id * COLS_PER_THREAD + cd;
        dKj[col] = __float2bfloat16(dk_vals[cd]);
    }
}

// ================================================================
// Phase 3: Compute dV
// ================================================================
__global__ void mha_bwd_dV(
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

    __nv_bfloat16* smem_Q = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* smem_dO = smem_Q + BLOCK_K * D;

    int tid = threadIdx.x;
    int lane_id = tid % 32;

    int k_base = blockIdx.x * BLOCK_Q;
    int k_row_in_block = tid / 32;

    if (k_row_in_block >= BLOCK_Q) return;

    int kj = k_base + k_row_in_block;
    if (kj >= S) return;

    float scale = rsqrtf(static_cast<float>(D));

    const __nv_bfloat16* Kj = K + base + kj * D;
    float k_vals[COLS_PER_THREAD];
    for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
        int col = lane_id * COLS_PER_THREAD + cd;
        k_vals[cd] = __bfloat162float(Kj[col]);
    }

    float dv_vals[COLS_PER_THREAD] = {};

    for (int ti = kj / BLOCK_K; ti * BLOCK_K < S; ti++) {
        int qi_base = ti * BLOCK_K;
        int qi_end = min(qi_base + BLOCK_K, S);
        int q_tile_size = qi_end - qi_base;

        if (qi_end <= kj) continue;

        if (tid < q_tile_size * D) {
            int row = tid / D;
            int col = tid % D;
            smem_Q[row * D + col] = Q[base + (qi_base + row) * D + col];
        }
        if (tid < q_tile_size * D) {
            int row = tid / D;
            int col = tid % D;
            smem_dO[row * D + col] = dO[base + (qi_base + row) * D + col];
        }
        __syncthreads();

        for (int qr = 0; qr < q_tile_size; qr++) {
            int qi_actual = qi_base + qr;
            if (qi_actual < kj) continue;

            float q_vals[COLS_PER_THREAD];
            float dO_vals[COLS_PER_THREAD];
            for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
                int col = lane_id * COLS_PER_THREAD + cd;
                q_vals[cd] = __bfloat162float(smem_Q[qr * D + col]);
                dO_vals[cd] = __bfloat162float(smem_dO[qr * D + col]);
            }

            float score_tid = 0.0f;
            #pragma unroll
            for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
                score_tid += q_vals[cd] * k_vals[cd];
            }

            float score_full = warp_sum(score_tid);
            float attn = expf(score_full * scale - L[l_off + qi_actual]);

            #pragma unroll
            for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
                dv_vals[cd] += attn * dO_vals[cd];
            }
        }
        __syncthreads();
    }

    __nv_bfloat16* dVj = dV + base + kj * D;
    for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
        int col = lane_id * COLS_PER_THREAD + cd;
        dVj[col] = __float2bfloat16(dv_vals[cd]);
    }
}

void run(tvm::ffi::TensorView Q_tv, tvm::ffi::TensorView K_tv, tvm::ffi::TensorView V_tv,
         tvm::ffi::TensorView O_tv, tvm::ffi::TensorView dO_tv, tvm::ffi::TensorView L_tv,
         tvm::ffi::TensorView dQ_tv, tvm::ffi::TensorView dK_tv, tvm::ffi::TensorView dV_tv) {
    CUDA_CHECK(cudaSetDevice(Q_tv.device().device_id));
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

    size_t m_size = static_cast<size_t>(B) * H * S * sizeof(float);
    float* m_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&m_buf, m_size));

    int n_bh = B * H;
    int n_seq_blocks = (S + BLOCK_Q - 1) / BLOCK_Q;
    dim3 grid(n_seq_blocks, n_bh);
    dim3 block(NTHREADS);

    // Shared memory: 2 * BLOCK_K * D * sizeof(bf16) = 2*32*128*2 = 16KB
    size_t smem_bytes = 2ULL * BLOCK_K * D * sizeof(__nv_bfloat16);

    mha_bwd_dQ<<<grid, block, smem_bytes, stream>>>(Q, K, V, dO, L, dQ, m_buf, B, H, S, D);
    CUDA_CHECK(cudaGetLastError());

    mha_bwd_dK<<<grid, block, smem_bytes, stream>>>(Q, K, V, dO, L, m_buf, dK, B, H, S, D);
    CUDA_CHECK(cudaGetLastError());

    mha_bwd_dV<<<grid, block, smem_bytes, stream>>>(Q, K, dO, L, dV, B, H, S, D);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(m_buf));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_ns::run);

}  // namespace