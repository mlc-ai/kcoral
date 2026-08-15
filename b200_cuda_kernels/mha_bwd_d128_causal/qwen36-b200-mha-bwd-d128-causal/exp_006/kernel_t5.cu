#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
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

namespace mha_bwd_tiled {

static constexpr int BK_BLOCK_M = 64;
static constexpr int BK_BLOCK_N = 64;
static constexpr int BK_THREADS = 256;
static constexpr int BK_DIM = 128;
static constexpr int BK_TPR = BK_THREADS / BK_BLOCK_M; // 4 threads per q_row
static constexpr int BK_DPT = BK_DIM / BK_TPR;         // 32 dims per thread

// ============================================================
// dQ kernel: dQ[b,h,sq,dim] = sum_{sk<=sq} dS[sq,sk] * K[sk,dim]
// ============================================================
__global__ void mha_bwd_dQ_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    int B, int H, int S
) {
    extern __shared__ char smem_raw[];

    // Shared memory layout
    __nv_bfloat16* sQ  = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sdO = sQ + BK_BLOCK_M * BK_DIM;
    __nv_bfloat16* sO  = sdO + BK_BLOCK_M * BK_DIM;
    __nv_bfloat16* sK  = sO + BK_BLOCK_M * BK_DIM;
    __nv_bfloat16* sV  = sK + BK_BLOCK_N * BK_DIM;

    float* fdata = reinterpret_cast<float*>(sV + BK_BLOCK_N * BK_DIM);
    float* D_partial = fdata;                                // [BK_BLOCK_M * BK_TPR]
    float* D_final   = D_partial + BK_BLOCK_M * BK_TPR;      // [BK_BLOCK_M]
    float* score_arr = D_final + BK_BLOCK_M;                  // [BK_BLOCK_M * BK_BLOCK_N]
    float* dp_arr    = score_arr + BK_BLOCK_M * BK_BLOCK_N;  // [BK_BLOCK_M * BK_BLOCK_N]

    int bh = blockIdx.x;
    if (bh >= B * H) return;
    int b = bh / H;
    int h = bh % H;
    int tid = threadIdx.x;

    size_t base_bh = (size_t)b * H * S * BK_DIM + (size_t)h * S * BK_DIM;
    size_t base_L  = (size_t)b * H * S + (size_t)h * S;

    float inv_sqrt_d = 1.0f / sqrtf((float)BK_DIM);

    int my_q_row  = tid / BK_TPR;
    int my_lane   = tid % BK_TPR;

    float reg_dQ[BK_DPT];
    #pragma unroll
    for (int di = 0; di < BK_DPT; di++) reg_dQ[di] = 0.0f;

    int num_q_tiles = (S + BK_BLOCK_M - 1) / BK_BLOCK_M;
    int num_k_tiles = (S + BK_BLOCK_N - 1) / BK_BLOCK_N;

    for (int mq = 0; mq < num_q_tiles; mq++) {
        int sq_base = mq * BK_BLOCK_M;
        int cur_bm = min(BK_BLOCK_M, S - sq_base);

        // Reset dQ accumulators
        #pragma unroll
        for (int di = 0; di < BK_DPT; di++) reg_dQ[di] = 0.0f;

        // Load Q tile
        for (int i = tid; i < cur_bm * BK_DIM; i += BK_THREADS) {
            sQ[i] = Q[base_bh + (size_t)(sq_base + i / BK_DIM) * BK_DIM + (i % BK_DIM)];
        }
        // Load dO tile
        for (int i = tid; i < cur_bm * BK_DIM; i += BK_THREADS) {
            sdO[i] = dO[base_bh + (size_t)(sq_base + i / BK_DIM) * BK_DIM + (i % BK_DIM)];
        }
        // Load O tile
        for (int i = tid; i < cur_bm * BK_DIM; i += BK_THREADS) {
            sO[i] = O[base_bh + (size_t)(sq_base + i / BK_DIM) * BK_DIM + (i % BK_DIM)];
        }
        __syncthreads();

        for (int mk = 0; mk < num_k_tiles; mk++) {
            int sk_base = mk * BK_BLOCK_N;
            int cur_bn = min(BK_BLOCK_N, S - sk_base);
            if (sk_base >= sq_base + cur_bm) break; // causal early exit

            // Load K tile
            for (int i = tid; i < cur_bn * BK_DIM; i += BK_THREADS) {
                sK[i] = K[base_bh + (size_t)(sk_base + i / BK_DIM) * BK_DIM + (i % BK_DIM)];
            }
            // Load V tile
            for (int i = tid; i < cur_bn * BK_DIM; i += BK_THREADS) {
                sV[i] = V[base_bh + (size_t)(sk_base + i / BK_DIM) * BK_DIM + (i % BK_DIM)];
            }
            __syncthreads();

            // Compute D[qr] = dot(dO[qr], O[qr]) with per-thread partials
            for (int qr = my_q_row; qr < cur_bm; qr += BK_TPR) {
                float dval = 0.0f;
                for (int j = my_lane * BK_DPT; j < my_lane * BK_DPT + BK_DPT; j++) {
                    dval += __bfloat162float(sdO[qr * BK_DIM + j]) * __bfloat162float(sO[qr * BK_DIM + j]);
                }
                D_partial[qr * BK_TPR + my_lane] = dval;
            }
            __syncthreads();

            // Reduce D_partial -> D_final (lane 0 of each group sums)
            if (my_lane == 0) {
                for (int qr = my_q_row; qr < cur_bm; qr += BK_TPR) {
                    float s = D_partial[qr * BK_TPR + 0] + D_partial[qr * BK_TPR + 1]
                            + D_partial[qr * BK_TPR + 2] + D_partial[qr * BK_TPR + 3];
                    D_final[qr] = s;
                }
            }
            __syncthreads();

            // Compute score[qr][kr] and dp_partial[qr][kr] cooperatively
            int total_pairs = cur_bm * cur_bn;
            for (int p = tid; p < total_pairs; p += BK_THREADS) {
                int qr = p / cur_bn;
                int kr = p % cur_bn;
                int abs_sq = sq_base + qr;
                int abs_sk = sk_base + kr;
                if (abs_sk > abs_sq) {
                    score_arr[p] = 0.0f;
                    dp_arr[p] = 0.0f;
                    continue;
                }
                float sc = 0.0f;
                for (int j = 0; j < BK_DIM; j += 4) {
                    sc += __bfloat162float(sQ[qr * BK_DIM + j])   * __bfloat162float(sK[kr * BK_DIM + j]);
                    sc += __bfloat162float(sQ[qr * BK_DIM + j+1]) * __bfloat162float(sK[kr * BK_DIM + j+1]);
                    sc += __bfloat162float(sQ[qr * BK_DIM + j+2]) * __bfloat162float(sK[kr * BK_DIM + j+2]);
                    sc += __bfloat162float(sQ[qr * BK_DIM + j+3]) * __bfloat162float(sK[kr * BK_DIM + j+3]);
                }
                score_arr[p] = sc * inv_sqrt_d;

                float dp = 0.0f;
                for (int j = 0; j < BK_DIM; j += 4) {
                    dp += __bfloat162float(sV[kr * BK_DIM + j])   * __bfloat162float(sdO[qr * BK_DIM + j]);
                    dp += __bfloat162float(sV[kr * BK_DIM + j+1]) * __bfloat162float(sdO[qr * BK_DIM + j+1]);
                    dp += __bfloat162float(sV[kr * BK_DIM + j+2]) * __bfloat162float(sdO[qr * BK_DIM + j+2]);
                    dp += __bfloat162float(sV[kr * BK_DIM + j+3]) * __bfloat162float(sdO[qr * BK_DIM + j+3]);
                }
                dp_arr[p] = dp;
            }
            __syncthreads();

            // Accumulate dQ: dQ[qr, dim] += sum_kr dS[qr, kr] * K[kr, dim]
            for (int qr = my_q_row; qr < cur_bm; qr += BK_TPR) {
                int abs_sq = sq_base + qr;
                float lse = L[base_L + abs_sq];
                float Dv = D_final[qr];

                for (int di = 0; di < BK_DPT; di++) {
                    int dim = my_lane * BK_DPT + di;
                    if (dim >= BK_DIM) break;
                    float accum = 0.0f;
                    for (int kr = 0; kr < cur_bn; kr++) {
                        int p = qr * cur_bn + kr;
                        float pv = expf(score_arr[p] - lse);
                        float ds = pv * (dp_arr[p] - Dv);
                        accum += ds * __bfloat162float(sK[kr * BK_DIM + dim]);
                    }
                    reg_dQ[di] += accum;
                }
            }
            __syncthreads();
        }

        // Write dQ results for this Q-tile
        for (int qr = my_q_row; qr < cur_bm; qr += BK_TPR) {
            int abs_sq = sq_base + qr;
            for (int di = 0; di < BK_DPT; di++) {
                int dim = my_lane * BK_DPT + di;
                if (dim >= BK_DIM) break;
                dQ[base_bh + (size_t)abs_sq * BK_DIM + dim] = __float2bfloat16(reg_dQ[di]);
            }
        }
    }
}

// ============================================================
// dV kernel: dV[b,h,sk,dim] = sum_{sq>=sk} P[sq,sk] * dO[sq,dim]
// ============================================================
__global__ void mha_bwd_dV_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dV,
    int B, int H, int S
) {
    extern __shared__ char smem_raw[];

    // Shared memory layout for dV
    // sQ: [0, BM*D), sdO: [BM*D, 2*BM*D), sK: [2*BM*D, 2*BM*D+BN*D), sV: unused
    // score: f32[BM*BN], dV_acc: f32[BN*DK_PER_GROUP*TGR] where TGR=threads per kr-group
    // Actually: each thread owns some kr rows and DK_PER_GROUP dims. Simplify: each (kr,dim) cell is owned by exactly one thread for accumulation.
    
    // Simpler smem: sQ, sdO, sK, score_f32, dv_acc_f32
    __nv_bfloat16* sQ  = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sdO = sQ + BK_BLOCK_M * BK_DIM;
    __nv_bfloat16* sK  = sdO + BK_BLOCK_M * BK_DIM;

    float* fdata = reinterpret_cast<float*>(sK + BK_BLOCK_N * BK_DIM);
    float* score_arr = fdata;                                        // [BK_BLOCK_M * BK_BLOCK_N]
    float* dv_acc     = score_arr + BK_BLOCK_M * BK_BLOCK_N;         // [BK_BLOCK_N * BK_DIM]

    int bh = blockIdx.x;
    if (bh >= B * H) return;
    int b = bh / H;
    int h = bh % H;
    int tid = threadIdx.x;

    size_t base_bh = (size_t)b * H * S * BK_DIM + (size_t)h * S * BK_DIM;
    size_t base_L  = (size_t)b * H * S + (size_t)h * S;

    float inv_sqrt_d = 1.0f / sqrtf((float)BK_DIM);

    int my_q_row = tid / BK_TPR;
    int my_lane  = tid % BK_TPR;

    int num_k_tiles = (S + BK_BLOCK_N - 1) / BK_BLOCK_N;
    int num_q_tiles = (S + BK_BLOCK_M - 1) / BK_BLOCK_M;

    // dV accumulator is in smem. Thread ownership:
    // For dv_acc[kr * BK_DIM + dim], owner is thread with:
    // my_q_row = kr * BK_TPR + my_lane ... nah, this mapping is weird.
    // Simpler: use grid-stride within the dv_acc reduction.
    // Actually, let me just have each thread own certain (kr, dims) in the acc.
    // BN*DIM = 64*128 = 8192 cells. 256 threads -> 32 cells per thread.
    // Thread tid owns dv_acc for cells starting at tid, stride 256.
    
    for (int mk = 0; mk < num_k_tiles; mk++) {
        int sk_base = mk * BK_BLOCK_N;
        int cur_bn = min(BK_BLOCK_N, S - sk_base);

        // Zero dV accumulator for this K-tile
        for (int cell = tid; cell < cur_bn * BK_DIM; cell += BK_THREADS) {
            dv_acc[cell] = 0.0f;
        }

        // Iterate Q tiles (causal: sq >= sk_base)
        int mq_start = (sk_base + BK_BLOCK_M - 1) / BK_BLOCK_M;
        for (int mq = mq_start; mq < num_q_tiles; mq++) {
            int sq_base = mq * BK_BLOCK_M;
            int cur_bm = min(BK_BLOCK_M, S - sq_base);

            // Load Q tile
            for (int i = tid; i < cur_bm * BK_DIM; i += BK_THREADS) {
                sQ[i] = Q[base_bh + (size_t)(sq_base + i / BK_DIM) * BK_DIM + (i % BK_DIM)];
            }
            // Load dO tile
            for (int i = tid; i < cur_bm * BK_DIM; i += BK_THREADS) {
                sdO[i] = dO[base_bh + (size_t)(sq_base + i / BK_DIM) * BK_DIM + (i % BK_DIM)];
            }
            __syncthreads();

            // Load K tile (only needed for first Q-tile in K-tile iteration)
            if (mq == mq_start) {
                for (int i = tid; i < cur_bn * BK_DIM; i += BK_THREADS) {
                    sK[i] = K[base_bh + (size_t)(sk_base + i / BK_DIM) * BK_DIM + (i % BK_DIM)];
                }
                __syncthreads();
            }

            // Compute score[qr][kr] cooperatively
            int total_pairs = cur_bm * cur_bn;
            for (int p = tid; p < total_pairs; p += BK_THREADS) {
                int qr = p / cur_bn;
                int kr = p % cur_bn;
                int abs_sq = sq_base + qr;
                int abs_sk = sk_base + kr;
                if (abs_sk > abs_sq) {
                    score_arr[p] = 0.0f; // Won't contribute
                    continue;
                }
                float sc = 0.0f;
                for (int j = 0; j < BK_DIM; j += 4) {
                    sc += __bfloat162float(sQ[qr * BK_DIM + j])   * __bfloat162float(sK[kr * BK_DIM + j]);
                    sc += __bfloat162float(sQ[qr * BK_DIM + j+1]) * __bfloat162float(sK[kr * BK_DIM + j+1]);
                    sc += __bfloat162float(sQ[qr * BK_DIM + j+2]) * __bfloat162float(sK[kr * BK_DIM + j+2]);
                    sc += __bfloat162float(sQ[qr * BK_DIM + j+3]) * __bfloat162float(sK[kr * BK_DIM + j+3]);
                }
                score_arr[p] = sc * inv_sqrt_d;
            }
            __syncthreads();

            // Accumulate dV: for each qr, for each kr with valid causal, add P * dO[qr, :] to dv_acc[kr, :]
            for (int qr = my_q_row; qr < cur_bm; qr += BK_TPR) {
                int abs_sq = sq_base + qr;
                float lse = L[base_L + abs_sq];

                for (int di = 0; di < BK_DPT; di++) {
                    int dim = my_lane * BK_DPT + di;
                    if (dim >= BK_DIM) break;
                    
                    float do_val = __bfloat162float(sdO[qr * BK_DIM + dim]);
                    
                    for (int kr = 0; kr < cur_bn; kr++) {
                        int p = qr * cur_bn + kr;
                        int abs_sk = sk_base + kr;
                        if (abs_sk > abs_sq) continue;
                        float pv = expf(score_arr[p] - lse);
                        float contrib = pv * do_val;
                        dv_acc[kr * BK_DIM + dim] += contrib;
                    }
                }
            }
            __syncthreads();
        }

        // Write dV accumulator to global for this K-tile
        for (int cell = tid; cell < cur_bn * BK_DIM; cell += BK_THREADS) {
            int kr = cell / BK_DIM;
            int dim = cell % BK_DIM;
            int abs_sk = sk_base + kr;
            dV[base_bh + (size_t)abs_sk * BK_DIM + dim] = __float2bfloat16(dv_acc[cell]);
        }
    }
}

// ============================================================
// dK kernel: dK[b,h,sk,dim] = sum_{sq>=sk} dS[sq,sk] * Q[sq,dim]
// ============================================================
__global__ void mha_bwd_dK_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dK,
    int B, int H, int S
) {
    extern __shared__ char smem_raw[];

    __nv_bfloat16* sQ  = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sdO = sQ + BK_BLOCK_M * BK_DIM;
    __nv_bfloat16* sO  = sdO + BK_BLOCK_M * BK_DIM;
    __nv_bfloat16* sK  = sO + BK_BLOCK_M * BK_DIM;
    __nv_bfloat16* sV  = sK + BK_BLOCK_N * BK_DIM;

    float* fdata = reinterpret_cast<float*>(sV + BK_BLOCK_N * BK_DIM);
    float* D_partial = fdata;                                         // [BK_BLOCK_M * BK_TPR]
    float* D_final   = D_partial + BK_BLOCK_M * BK_TPR;               // [BK_BLOCK_M]
    float* score_arr = D_final + BK_BLOCK_M;                           // [BK_BLOCK_M * BK_BLOCK_N]
    float* dp_arr    = score_arr + BK_BLOCK_M * BK_BLOCK_N;            // [BK_BLOCK_M * BK_BLOCK_N]
    float* dk_acc    = dp_arr + BK_BLOCK_M * BK_BLOCK_N;               // [BK_BLOCK_N * BK_DIM]

    int bh = blockIdx.x;
    if (bh >= B * H) return;
    int b = bh / H;
    int h = bh % H;
    int tid = threadIdx.x;

    size_t base_bh = (size_t)b * H * S * BK_DIM + (size_t)h * S * BK_DIM;
    size_t base_L  = (size_t)b * H * S + (size_t)h * S;

    float inv_sqrt_d = 1.0f / sqrtf((float)BK_DIM);

    int my_q_row = tid / BK_TPR;
    int my_lane  = tid % BK_TPR;

    int num_k_tiles = (S + BK_BLOCK_N - 1) / BK_BLOCK_N;
    int num_q_tiles = (S + BK_BLOCK_M - 1) / BK_BLOCK_M;

    for (int mk = 0; mk < num_k_tiles; mk++) {
        int sk_base = mk * BK_BLOCK_N;
        int cur_bn = min(BK_BLOCK_N, S - sk_base);

        // Zero dK accumulator for this K-tile
        for (int cell = tid; cell < cur_bn * BK_DIM; cell += BK_THREADS) {
            dk_acc[cell] = 0.0f;
        }

        int mq_start = (sk_base + BK_BLOCK_M - 1) / BK_BLOCK_M;
        for (int mq = mq_start; mq < num_q_tiles; mq++) {
            int sq_base = mq * BK_BLOCK_M;
            int cur_bm = min(BK_BLOCK_M, S - sq_base);

            // Load Q tile
            for (int i = tid; i < cur_bm * BK_DIM; i += BK_THREADS) {
                sQ[i] = Q[base_bh + (size_t)(sq_base + i / BK_DIM) * BK_DIM + (i % BK_DIM)];
            }
            // Load dO tile
            for (int i = tid; i < cur_bm * BK_DIM; i += BK_THREADS) {
                sdO[i] = dO[base_bh + (size_t)(sq_base + i / BK_DIM) * BK_DIM + (i % BK_DIM)];
            }
            // Load O tile
            for (int i = tid; i < cur_bm * BK_DIM; i += BK_THREADS) {
                sO[i] = O[base_bh + (size_t)(sq_base + i / BK_DIM) * BK_DIM + (i % BK_DIM)];
            }
            __syncthreads();

            // Load K tile (first Q-tile only per K-tile)
            if (mq == mq_start) {
                for (int i = tid; i < cur_bn * BK_DIM; i += BK_THREADS) {
                    sK[i] = K[base_bh + (size_t)(sk_base + i / BK_DIM) * BK_DIM + (i % BK_DIM)];
                }
                // Load V tile
                for (int i = tid; i < cur_bn * BK_DIM; i += BK_THREADS) {
                    sV[i] = V[base_bh + (size_t)(sk_base + i / BK_DIM) * BK_DIM + (i % BK_DIM)];
                }
                __syncthreads();
            }

            // Compute D[qr]
            for (int qr = my_q_row; qr < cur_bm; qr += BK_TPR) {
                float dval = 0.0f;
                for (int j = my_lane * BK_DPT; j < my_lane * BK_DPT + BK_DPT; j++) {
                    dval += __bfloat162float(sdO[qr * BK_DIM + j]) * __bfloat162float(sO[qr * BK_DIM + j]);
                }
                D_partial[qr * BK_TPR + my_lane] = dval;
            }
            __syncthreads();

            if (my_lane == 0) {
                for (int qr = my_q_row; qr < cur_bm; qr += BK_TPR) {
                    float s = D_partial[qr * BK_TPR + 0] + D_partial[qr * BK_TPR + 1]
                            + D_partial[qr * BK_TPR + 2] + D_partial[qr * BK_TPR + 3];
                    D_final[qr] = s;
                }
            }
            __syncthreads();

            // Compute score and dp_partial
            int total_pairs = cur_bm * cur_bn;
            for (int p = tid; p < total_pairs; p += BK_THREADS) {
                int qr = p / cur_bn;
                int kr = p % cur_bn;
                int abs_sq = sq_base + qr;
                int abs_sk = sk_base + kr;
                if (abs_sk > abs_sq) {
                    score_arr[p] = 0.0f;
                    dp_arr[p] = 0.0f;
                    continue;
                }
                float sc = 0.0f;
                for (int j = 0; j < BK_DIM; j += 4) {
                    sc += __bfloat162float(sQ[qr * BK_DIM + j])   * __bfloat162float(sK[kr * BK_DIM + j]);
                    sc += __bfloat162float(sQ[qr * BK_DIM + j+1]) * __bfloat162float(sK[kr * BK_DIM + j+1]);
                    sc += __bfloat162float(sQ[qr * BK_DIM + j+2]) * __bfloat162float(sK[kr * BK_DIM + j+2]);
                    sc += __bfloat162float(sQ[qr * BK_DIM + j+3]) * __bfloat162float(sK[kr * BK_DIM + j+3]);
                }
                score_arr[p] = sc * inv_sqrt_d;

                float dp = 0.0f;
                for (int j = 0; j < BK_DIM; j += 4) {
                    dp += __bfloat162float(sV[kr * BK_DIM + j])   * __bfloat162float(sdO[qr * BK_DIM + j]);
                    dp += __bfloat162float(sV[kr * BK_DIM + j+1]) * __bfloat162float(sdO[qr * BK_DIM + j+1]);
                    dp += __bfloat162float(sV[kr * BK_DIM + j+2]) * __bfloat162float(sdO[qr * BK_DIM + j+2]);
                    dp += __bfloat162float(sV[kr * BK_DIM + j+3]) * __bfloat162float(sdO[qr * BK_DIM + j+3]);
                }
                dp_arr[p] = dp;
            }
            __syncthreads();

            // Accumulate dK: for each valid (qr, kr), dK[kr, dim] += dS[qr, kr] * Q[qr, dim]
            for (int qr = my_q_row; qr < cur_bm; qr += BK_TPR) {
                int abs_sq = sq_base + qr;
                float lse = L[base_L + abs_sq];
                float Dv = D_final[qr];

                for (int di = 0; di < BK_DPT; di++) {
                    int dim = my_lane * BK_DPT + di;
                    if (dim >= BK_DIM) break;

                    float q_val = __bfloat162float(sQ[qr * BK_DIM + dim]);

                    for (int kr = 0; kr < cur_bn; kr++) {
                        int p = qr * cur_bn + kr;
                        int abs_sk = sk_base + kr;
                        if (abs_sk > abs_sq) continue;
                        float pv = expf(score_arr[p] - lse);
                        float ds = pv * (dp_arr[p] - Dv);
                        dk_acc[kr * BK_DIM + dim] += ds * q_val;
                    }
                }
            }
            __syncthreads();
        }

        // Write dK accumulator to global
        for (int cell = tid; cell < cur_bn * BK_DIM; cell += BK_THREADS) {
            int kr = cell / BK_DIM;
            int dim = cell % BK_DIM;
            int abs_sk = sk_base + kr;
            dK[base_bh + (size_t)abs_sk * BK_DIM + dim] = __float2bfloat16(dk_acc[cell]);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int S = (int)Q.size(2);
    int d = (int)Q.size(3);

    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_ptr = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_ptr = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    int num_bh = B * H;
    dim3 grid(num_bh);
    dim3 block(BK_THREADS);

    // Compute shared memory sizes
    // dQ kernel: 3*BM*D + 2*BN*D bf16 + (BM*TPR + BM + 2*BM*BN) * 4 f32
    size_t dQ_smem = ((size_t)3 * BK_BLOCK_M + 2 * BK_BLOCK_N) * BK_DIM * sizeof(__nv_bfloat16)
                   + ((size_t)BK_BLOCK_M * BK_TPR + BK_BLOCK_M + 2UL * BK_BLOCK_M * BK_BLOCK_N) * sizeof(float);

    // dV kernel: 2*BM*D + BN*D bf16 + (BM*BN + BN*DIM) * 4 f32
    size_t dV_smem = ((size_t)2 * BK_BLOCK_M + BK_BLOCK_N) * BK_DIM * sizeof(__nv_bfloat16)
                   + ((size_t)BK_BLOCK_M * BK_BLOCK_N + (size_t)BK_BLOCK_N * BK_DIM) * sizeof(float);

    // dK kernel: 3*BM*D + 2*BN*D bf16 + (BM*TPR + BM + 2*BM*BN + BN*DIM) * 4 f32
    size_t dK_smem = ((size_t)3 * BK_BLOCK_M + 2 * BK_BLOCK_N) * BK_DIM * sizeof(__nv_bfloat16)
                   + ((size_t)BK_BLOCK_M * BK_TPR + BK_BLOCK_M + 2UL * BK_BLOCK_M * BK_BLOCK_N + (size_t)BK_BLOCK_N * BK_DIM) * sizeof(float);

    // Launch dV first (lightest kernel, just score computation)
    mha_bwd_dV_kernel<<<grid, block, dV_smem, stream>>>(
        Q_ptr, K_ptr, dO_ptr, L_ptr, dV_ptr, B, H, S);
    CUDA_CHECK(cudaGetLastError());

    // Launch dQ
    mha_bwd_dQ_kernel<<<grid, block, dQ_smem, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr, B, H, S);
    CUDA_CHECK(cudaGetLastError());

    // Launch dK
    mha_bwd_dK_kernel<<<grid, block, dK_smem, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, B, H, S);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_tiled::run);

}  // namespace mha_bwd_tiled