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

namespace mha_bwd_v3 {

static constexpr int BK_M   = 64;
static constexpr int BK_N   = 64;
static constexpr int BK_D   = 128;
static constexpr int BK_TPR = 4;
static constexpr int BK_TP  = 256;
static constexpr int BK_DP  = 32;

// ============================================================
// Combined kernel: computes dQ, dK, dV in one pass
// Each block handles one (b, h) pair, tiled over S x S
// ============================================================
__global__ void mha_bwd_combined_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int B, int H, int S
) {
    extern __shared__ char smem_raw[];

    // Shared memory layout (in bytes):
    // sQ:     [0 : BM*D*2)           bf16
    // sdO:    [BM*D*2 : 2*BM*D*2)   bf16
    // sO:     [2*BM*D*2 : 3*BM*D*2) bf16
    // sK:     [3*BM*D*2 : 3*BM*D*2+BN*D*2) bf16
    // sV:     [above+BN*D*2 : ...)  bf16
    // After bf16 section, float section aligned to 16 bytes:
    // D_red:  [BM*TPR]               f32 partial sums per thread group
    // D:      [BM]                   f32 reduced D per q_row
    // Pmat:   [BM*BN]                f32 attention probabilities
    // ds_mat: [BM*BN]                f32 dS values
    
    size_t off_bf16 = ((size_t)BK_M * 3 + BK_N * 2) * BK_D * sizeof(__nv_bfloat16);
    
    __nv_bfloat16* sQ   = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sdO  = sQ + BK_M * BK_D;
    __nv_bfloat16* sO   = sdO + BK_M * BK_D;
    __nv_bfloat16* sK   = sO + BK_M * BK_D;
    __nv_bfloat16* sV   = sK + BK_N * BK_D;
    
    // Align to 16 bytes for float section
    off_bf16 = (off_bf16 + 15) & ~15;
    float* flt = reinterpret_cast<float*>(smem_raw + off_bf16);
    
    float* D_red  = flt;                            // [BM*TPR]
    float* D_val  = D_red + BK_M * BK_TPR;          // [BM]
    float* Pmat   = D_val + BK_M;                   // [BM*BN]
    float* ds_mat = Pmat + BK_M * BK_N;             // [BM*BN]

    int bh  = blockIdx.x;
    if (bh >= B * H) return;
    int b = bh / H;
    int h = bh % H;
    int tid = threadIdx.x;

    size_t base_bh = (size_t)b * H * S * BK_D + (size_t)h * S * BK_D;
    size_t base_L  = (size_t)b * H * S + (size_t)h * S;

    float inv_sqrt_d = 1.0f / sqrtf((float)BK_D);

    int my_qr  = tid / BK_TPR;
    int my_ln  = tid % BK_TPR;

    // Register accumulators for dQ: each thread owns BK_DP dimensions
    float reg_dQ[BK_DP];

    int num_mq = (S + BK_M - 1) / BK_M;
    int num_mk = (S + BK_N - 1) / BK_N;

    // Process each K-tile (outer loop over kv positions)
    for (int mk = 0; mk < num_mk; mk++) {
        int sk_base = mk * BK_N;
        int cur_bn = min(BK_N, S - sk_base);

        // We will accumulate dK[kr, dim] and dV[kr, dim] in registers and write out
        // after processing all Q-tiles that contribute to this sk range.
        // For dK/dV, each (kr, dim) pair needs accumulation across all sq >= sk.
        // Since BK_N*BK_D = 8192 cells and BK_TP=256 threads, ~32 cells per thread.
        
        // Accumulate dK/dV in shared memory
        // dk_acc: [BN*DP_f32_per_thread_group * TPR ... nah, simpler:]
        // Use registers: each thread accumulates its share
        
        // Simpler approach: compute dK/dV contributions into shared memory arrays
        // Actually let's just use atomic-friendly approach or separate passes.
        
        // To avoid complex smem sharing, let me split into 3 simple passes:
        // Pass 1: Compute dQ (accumulated across K-tiles for each Q-tile)
        // Pass 2: Compute dV (accumulated across Q-tiles for each K-tile)  
        // Pass 3: Compute dK (same structure as dV)
        
        // For now, compute dK and dV into shared memory accumulators
        // dk_smem: [BN*BD] as f32 -> that's 64*128*4 = 32KB, too much extra smem.
        // Instead, accumulate dK/dV in global memory using atomics.
        // For correctness, let me make each (kr,dim) owned by one thread.
        
        // Thread-to-(kr,dim) mapping for local K-tile:
        int num_kd_cells = cur_bn * BK_D;
        int my_kd_start = 0; // Will set per cell ownership below
        
        // Reset: we'll accumulate in registers for our assigned cells
        // This gets complex. Let me simplify further.
        
        // Actually, let me just do three separate loops over Q-tiles:
        // First compute dQ for ALL Q-tiles
        // Then compute dV for ALL K-tiles
        // Then compute dK for ALL K-tiles
        
        // That means I need restructure. For now, let me just compute dQ first (it's clean).
        break;
    }
}

// ============================================================
// Simple single-gradient kernel: dQ only
// ============================================================
__global__ void mha_bwd_dQ_simple(
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
    
    // Layout:
    // sQ[BM*BD], sdO[BM*BD], sO[BM*BD], sK[BN*BD], sV[BN*BD] all bf16
    // Then float: D_part[BM*TPR], D_full[BM], score[BM*BN], dp[BM*BN]
    
    size_t bf16_bytes = ((size_t)BK_M * 3 + BK_N * 2) * BK_D * sizeof(__nv_bfloat16);
    bf16_bytes = (bf16_bytes + 15) & ~15;
    
    __nv_bfloat16* sQ   = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sdO  = sQ + BK_M * BK_D;
    __nv_bfloat16* sO   = sdO + BK_M * BK_D;
    __nv_bfloat16* sK   = sO + BK_M * BK_D;
    __nv_bfloat16* sV   = sK + BK_N * BK_D;
    
    float* fptr = reinterpret_cast<float*>(smem_raw + bf16_bytes);
    float* D_part = fptr;                                    // [BM*TPR]
    float* D_full = D_part + BK_M * BK_TPR;                  // [BM]
    float* score  = D_full + BK_M;                           // [BM*BN]
    float* dp_arr = score + BK_M * BK_N;                     // [BM*BN]

    int bh  = blockIdx.x;
    if (bh >= B * H) return;
    int b = bh / H;
    int h = bh % H;
    int tid = threadIdx.x;

    size_t base_bh = (size_t)b * H * S * BK_D + (size_t)h * S * BK_D;
    size_t base_L  = (size_t)b * H * S + (size_t)h * S;
    float inv_sqrt_d = 1.0f / sqrtf((float)BK_D);

    int my_qr = tid / BK_TPR;
    int my_ln = tid % BK_TPR;
    
    float reg_dQ[BK_DP];
    for (int di = 0; di < BK_DP; di++) reg_dQ[di] = 0.0f;

    int num_mq = (S + BK_M - 1) / BK_M;
    int num_mk = (S + BK_N - 1) / BK_N;

    for (int mq = 0; mq < num_mq; mq++) {
        int sq_base = mq * BK_M;
        int cur_bm = min(BK_M, S - sq_base);
        
        #pragma unroll
        for (int di = 0; di < BK_DP; di++) reg_dQ[di] = 0.0f;
        
        // Load tiles
        for (int i = tid; i < cur_bm * BK_D; i += BK_TP) {
            int r = i / BK_D, c = i % BK_D;
            sQ[i]   = Q[base_bh + (size_t)(sq_base + r) * BK_D + c];
            sdO[i]  = dO[base_bh + (size_t)(sq_base + r) * BK_D + c];
            sO[i]   = O[base_bh + (size_t)(sq_base + r) * BK_D + c];
        }
        __syncthreads();
        
        for (int mk = 0; mk < num_mk; mk++) {
            int sk_base = mk * BK_N;
            int cur_bn = min(BK_N, S - sk_base);
            if (sk_base >= sq_base + cur_bm) break;
            
            for (int i = tid; i < cur_bn * BK_D; i += BK_TP) {
                int r = i / BK_D, c = i % BK_D;
                sK[i] = K[base_bh + (size_t)(sk_base + r) * BK_D + c];
                sV[i] = V[base_bh + (size_t)(sk_base + r) * BK_D + c];
            }
            __syncthreads();
            
            // Compute D partial per thread, then reduce
            for (int qr = my_qr; qr < cur_bm; qr += BK_TPR) {
                float sum = 0.0f;
                for (int j = my_ln * BK_DP; j < (my_ln + 1) * BK_DP && j < BK_D; j++) {
                    sum += __bfloat162float(sdO[qr * BK_D + j]) * __bfloat162float(sO[qr * BK_D + j]);
                }
                D_part[qr * BK_TPR + my_ln] = sum;
            }
            __syncthreads();
            
            if (my_ln == 0) {
                for (int qr = my_qr; qr < cur_bm; qr += BK_TPR) {
                    D_full[qr] = 0.0f;
                    for (int l = 0; l < BK_TPR; l++) {
                        D_full[qr] += D_part[qr * BK_TPR + l];
                    }
                }
            }
            __syncthreads();
            
            // Compute score and dp
            for (int p = tid; p < cur_bm * cur_bn; p += BK_TP) {
                int qr = p / cur_bn, kr = p % cur_bn;
                int abs_sq = sq_base + qr, abs_sk = sk_base + kr;
                if (abs_sk > abs_sq) { score[p] = 0.0f; dp_arr[p] = 0.0f; continue; }
                float sc = 0.0f, dp = 0.0f;
                for (int j = 0; j < BK_D; j += 4) {
                    sc += __bfloat162float(sQ[qr * BK_D + j])   * __bfloat162float(sK[kr * BK_D + j]);
                    sc += __bfloat162float(sQ[qr * BK_D + j+1]) * __bfloat162float(sK[kr * BK_D + j+1]);
                    sc += __bfloat162float(sQ[qr * BK_D + j+2]) * __bfloat162float(sK[kr * BK_D + j+2]);
                    sc += __bfloat162float(sQ[qr * BK_D + j+3]) * __bfloat162float(sK[kr * BK_D + j+3]);
                    dp += __bfloat162float(sV[kr * BK_D + j])   * __bfloat162float(sdO[qr * BK_D + j]);
                    dp += __bfloat162float(sV[kr * BK_D + j+1]) * __bfloat162float(sdO[qr * BK_D + j+1]);
                    dp += __bfloat162float(sV[kr * BK_D + j+2]) * __bfloat162float(sdO[qr * BK_D + j+2]);
                    dp += __bfloat162float(sV[kr * BK_D + j+3]) * __bfloat162float(sdO[qr * BK_D + j+3]);
                }
                score[p] = sc * inv_sqrt_d;
                dp_arr[p] = dp;
            }
            __syncthreads();
            
            // Accumulate dQ
            for (int qr = my_qr; qr < cur_bm; qr += BK_TPR) {
                int abs_sq = sq_base + qr;
                float lse = L[base_L + abs_sq];
                float Dv = D_full[qr];
                for (int di = 0; di < BK_DP; di++) {
                    int dim = my_ln * BK_DP + di;
                    if (dim >= BK_D) break;
                    float acc = 0.0f;
                    for (int kr = 0; kr < cur_bn; kr++) {
                        int p = qr * cur_bn + kr;
                        if (sk_base + kr > abs_sq) continue;
                        float pv = expf(score[p] - lse);
                        float ds = pv * (dp_arr[p] - Dv);
                        acc += ds * __bfloat162float(sK[kr * BK_D + dim]);
                    }
                    reg_dQ[di] += acc;
                }
            }
            __syncthreads();
        }
        
        // Write dQ
        for (int qr = my_qr; qr < cur_bm; qr += BK_TPR) {
            int abs_sq = sq_base + qr;
            for (int di = 0; di < BK_DP; di++) {
                int dim = my_ln * BK_DP + di;
                if (dim >= BK_D) continue;
                dQ[base_bh + (size_t)abs_sq * BK_D + dim] = __float2bfloat16(reg_dQ[di]);
            }
        }
    }
}

// ============================================================
// dV kernel: dV[b,h,sk,dim] = sum_{sq>=sk} P[sq,sk] * dO[sq,dim]
// ============================================================
__global__ void mha_bwd_dV_simple(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dV,
    int B, int H, int S
) {
    extern __shared__ char smem_raw[];
    
    // Layout: sQ[BM*D], sdO[BM*D], sK[BN*D] all bf16
    // Then float: score[BM*BN], dv_acc[BN*BD/threads_in_kr_groups?]
    // Simpler: dv_acc[BN*BD] f32 = 64*128*4 = 32KB extra, total about 80KB OK
    
    size_t bf16_bytes = ((size_t)BK_M * 2 + BK_N) * BK_D * sizeof(__nv_bfloat16);
    bf16_bytes = (bf16_bytes + 15) & ~15;
    
    __nv_bfloat16* sQ   = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sdO  = sQ + BK_M * BK_D;
    __nv_bfloat16* sK   = sdO + BK_M * BK_D;
    
    float* fptr = reinterpret_cast<float*>(smem_raw + bf16_bytes);
    float* score  = fptr;                                // [BM*BN]
    float* dv_acc = score + BK_M * BK_N;                 // [BN*BD]

    int bh  = blockIdx.x;
    if (bh >= B * H) return;
    int b = bh / H;
    int h = bh % H;
    int tid = threadIdx.x;

    size_t base_bh = (size_t)b * H * S * BK_D + (size_t)h * S * BK_D;
    size_t base_L  = (size_t)b * H * S + (size_t)h * S;
    float inv_sqrt_d = 1.0f / sqrtf((float)BK_D);

    int my_qr = tid / BK_TPR;
    int my_ln = tid % BK_TPR;

    int num_mq = (S + BK_M - 1) / BK_M;
    int num_mk = (S + BK_N - 1) / BK_N;

    for (int mk = 0; mk < num_mk; mk++) {
        int sk_base = mk * BK_N;
        int cur_bn = min(BK_N, S - sk_base);

        // Zero dv_acc
        for (int i = tid; i < cur_bn * BK_D; i += BK_TP) {
            dv_acc[i] = 0.0f;
        }
        __syncthreads();
        
        // Load K tile (constant across Q-tiles)
        for (int i = tid; i < cur_bn * BK_D; i += BK_TP) {
            int r = i / BK_D, c = i % BK_D;
            sK[i] = K[base_bh + (size_t)(sk_base + r) * BK_D + c];
        }
        __syncthreads();

        // Only process Q-tiles where some sq >= sk_base
        int mq_start = (sk_base + BK_M - 1) / BK_M;
        for (int mq = mq_start; mq < num_mq; mq++) {
            int sq_base = mq * BK_M;
            int cur_bm = min(BK_M, S - sq_base);
            
            for (int i = tid; i < cur_bm * BK_D; i += BK_TP) {
                int r = i / BK_D, c = i % BK_D;
                sQ[i]   = Q[base_bh + (size_t)(sq_base + r) * BK_D + c];
                sdO[i]  = dO[base_bh + (size_t)(sq_base + r) * BK_D + c];
            }
            __syncthreads();
            
            // Compute score
            for (int p = tid; p < cur_bm * cur_bn; p += BK_TP) {
                int qr = p / cur_bn, kr = p % cur_bn;
                int abs_sq = sq_base + qr, abs_sk = sk_base + kr;
                if (abs_sk > abs_sq) { score[p] = 0.0f; continue; }
                float sc = 0.0f;
                for (int j = 0; j < BK_D; j += 4) {
                    sc += __bfloat162float(sQ[qr * BK_D + j])   * __bfloat162float(sK[kr * BK_D + j]);
                    sc += __bfloat162float(sQ[qr * BK_D + j+1]) * __bfloat162float(sK[kr * BK_D + j+1]);
                    sc += __bfloat162float(sQ[qr * BK_D + j+2]) * __bfloat162float(sK[kr * BK_D + j+2]);
                    sc += __bfloat162float(sQ[qr * BK_D + j+3]) * __bfloat162float(sK[kr * BK_D + j+3]);
                }
                score[p] = sc * inv_sqrt_d;
            }
            __syncthreads();
            
            // Accumulate dV: for each valid (qr, kr), add P * dO[qr, :] to dv_acc[kr, :]
            for (int qr = my_qr; qr < cur_bm; qr += BK_TPR) {
                int abs_sq = sq_base + qr;
                float lse = L[base_L + abs_sq];
                for (int di = 0; di < BK_DP; di++) {
                    int dim = my_ln * BK_DP + di;
                    if (dim >= BK_D) break;
                    float do_val = __bfloat162float(sdO[qr * BK_D + dim]);
                    for (int kr = 0; kr < cur_bn; kr++) {
                        int p = qr * cur_bn + kr;
                        if (sk_base + kr > abs_sq) continue;
                        float pv = expf(score[p] - lse);
                        dv_acc[kr * BK_D + dim] += pv * do_val;
                    }
                }
            }
            __syncthreads();
        }
        
        // Write dV for this K-tile
        for (int i = tid; i < cur_bn * BK_D; i += BK_TP) {
            int kr = i / BK_D, dim = i % BK_D;
            dV[base_bh + (size_t)(sk_base + kr) * BK_D + dim] = __float2bfloat16(dv_acc[i]);
        }
    }
}

// ============================================================
// dK kernel: dK[b,h,sk,dim] = sum_{sq>=sk} dS[sq,sk] * Q[sq,dim]
// ============================================================
__global__ void mha_bwd_dK_simple(
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
    
    size_t bf16_bytes = ((size_t)BK_M * 3 + BK_N * 2) * BK_D * sizeof(__nv_bfloat16);
    bf16_bytes = (bf16_bytes + 15) & ~15;
    
    __nv_bfloat16* sQ   = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sdO  = sQ + BK_M * BK_D;
    __nv_bfloat16* sO   = sdO + BK_M * BK_D;
    __nv_bfloat16* sK   = sO + BK_M * BK_D;
    __nv_bfloat16* sV   = sK + BK_N * BK_D;
    
    float* fptr = reinterpret_cast<float*>(smem_raw + bf16_bytes);
    float* D_part  = fptr;                                   // [BM*TPR]
    float* D_full  = D_part + BK_M * BK_TPR;                 // [BM]
    float* score   = D_full + BK_M;                          // [BM*BN]
    float* dp_arr  = score + BK_M * BK_N;                    // [BM*BN]
    float* dk_acc  = dp_arr + BK_M * BK_N;                   // [BN*BD]

    int bh  = blockIdx.x;
    if (bh >= B * H) return;
    int b = bh / H;
    int h = bh % H;
    int tid = threadIdx.x;

    size_t base_bh = (size_t)b * H * S * BK_D + (size_t)h * S * BK_D;
    size_t base_L  = (size_t)b * H * S + (size_t)h * S;
    float inv_sqrt_d = 1.0f / sqrtf((float)BK_D);

    int my_qr = tid / BK_TPR;
    int my_ln = tid % BK_TPR;

    int num_mq = (S + BK_M - 1) / BK_M;
    int num_mk = (S + BK_N - 1) / BK_N;

    for (int mk = 0; mk < num_mk; mk++) {
        int sk_base = mk * BK_N;
        int cur_bn = min(BK_N, S - sk_base);

        for (int i = tid; i < cur_bn * BK_D; i += BK_TP) {
            dk_acc[i] = 0.0f;
        }
        __syncthreads();
        
        for (int i = tid; i < cur_bn * BK_D; i += BK_TP) {
            int r = i / BK_D, c = i % BK_D;
            sK[i] = K[base_bh + (size_t)(sk_base + r) * BK_D + c];
            sV[i] = V[base_bh + (size_t)(sk_base + r) * BK_D + c];
        }
        __syncthreads();

        int mq_start = (sk_base + BK_M - 1) / BK_M;
        for (int mq = mq_start; mq < num_mq; mq++) {
            int sq_base = mq * BK_M;
            int cur_bm = min(BK_M, S - sq_base);
            
            for (int i = tid; i < cur_bm * BK_D; i += BK_TP) {
                int r = i / BK_D, c = i % BK_D;
                sQ[i]   = Q[base_bh + (size_t)(sq_base + r) * BK_D + c];
                sdO[i]  = dO[base_bh + (size_t)(sq_base + r) * BK_D + c];
                sO[i]   = O[base_bh + (size_t)(sq_base + r) * BK_D + c];
            }
            __syncthreads();
            
            // D computation
            for (int qr = my_qr; qr < cur_bm; qr += BK_TPR) {
                float sum = 0.0f;
                for (int j = my_ln * BK_DP; j < (my_ln + 1) * BK_DP && j < BK_D; j++) {
                    sum += __bfloat162float(sdO[qr * BK_D + j]) * __bfloat162float(sO[qr * BK_D + j]);
                }
                D_part[qr * BK_TPR + my_ln] = sum;
            }
            __syncthreads();
            if (my_ln == 0) {
                for (int qr = my_qr; qr < cur_bm; qr += BK_TPR) {
                    D_full[qr] = 0.0f;
                    for (int l = 0; l < BK_TPR; l++) D_full[qr] += D_part[qr * BK_TPR + l];
                }
            }
            __syncthreads();
            
            // score and dp
            for (int p = tid; p < cur_bm * cur_bn; p += BK_TP) {
                int qr = p / cur_bn, kr = p % cur_bn;
                int abs_sq = sq_base + qr, abs_sk = sk_base + kr;
                if (abs_sk > abs_sq) { score[p] = 0.0f; dp_arr[p] = 0.0f; continue; }
                float sc = 0.0f, dp = 0.0f;
                for (int j = 0; j < BK_D; j += 4) {
                    sc += __bfloat162float(sQ[qr * BK_D + j])   * __bfloat162float(sK[kr * BK_D + j]);
                    sc += __bfloat162float(sQ[qr * BK_D + j+1]) * __bfloat162float(sK[kr * BK_D + j+1]);
                    sc += __bfloat162float(sQ[qr * BK_D + j+2]) * __bfloat162float(sK[kr * BK_D + j+2]);
                    sc += __bfloat162float(sQ[qr * BK_D + j+3]) * __bfloat162float(sK[kr * BK_D + j+3]);
                    dp += __bfloat162float(sV[kr * BK_D + j])   * __bfloat162float(sdO[qr * BK_D + j]);
                    dp += __bfloat162float(sV[kr * BK_D + j+1]) * __bfloat162float(sdO[qr * BK_D + j+1]);
                    dp += __bfloat162float(sV[kr * BK_D + j+2]) * __bfloat162float(sdO[qr * BK_D + j+2]);
                    dp += __bfloat162float(sV[kr * BK_D + j+3]) * __bfloat162float(sdO[qr * BK_D + j+3]);
                }
                score[p] = sc * inv_sqrt_d;
                dp_arr[p] = dp;
            }
            __syncthreads();
            
            // Accumulate dK
            for (int qr = my_qr; qr < cur_bm; qr += BK_TPR) {
                int abs_sq = sq_base + qr;
                float lse = L[base_L + abs_sq];
                float Dv = D_full[qr];
                for (int di = 0; di < BK_DP; di++) {
                    int dim = my_ln * BK_DP + di;
                    if (dim >= BK_D) break;
                    float q_val = __bfloat162float(sQ[qr * BK_D + dim]);
                    for (int kr = 0; kr < cur_bn; kr++) {
                        int p = qr * cur_bn + kr;
                        if (sk_base + kr > abs_sq) continue;
                        float pv = expf(score[p] - lse);
                        float ds = pv * (dp_arr[p] - Dv);
                        dk_acc[kr * BK_D + dim] += ds * q_val;
                    }
                }
            }
            __syncthreads();
        }
        
        for (int i = tid; i < cur_bn * BK_D; i += BK_TP) {
            int kr = i / BK_D, dim = i % BK_D;
            dK[base_bh + (size_t)(sk_base + kr) * BK_D + dim] = __float2bfloat16(dk_acc[i]);
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
    dim3 block(BK_TP);

    // Shared memory sizes
    size_t dQ_smem = ((size_t)BK_M * 3 + BK_N * 2) * BK_D * 2; // bf16 part
    dQ_smem = (dQ_smem + 15) & ~15;
    dQ_smem += ((size_t)BK_M * BK_TPR + BK_M + 2UL * BK_M * BK_N) * 4; // f32 part

    size_t dV_smem = ((size_t)BK_M * 2 + BK_N) * BK_D * 2;
    dV_smem = (dV_smem + 15) & ~15;
    dV_smem += ((size_t)BK_M * BK_N + (size_t)BK_N * BK_D) * 4;

    size_t dK_smem = ((size_t)BK_M * 3 + BK_N * 2) * BK_D * 2;
    dK_smem = (dK_smem + 15) & ~15;
    dK_smem += ((size_t)BK_M * BK_TPR + BK_M + 2UL * BK_M * BK_N + (size_t)BK_N * BK_D) * 4;

    // Print debug info
    fprintf(stderr, "SMEM dQ=%zu dV=%zu dK=%zu BH=%d S=%d\n", dQ_smem, dV_smem, dK_smem, num_bh, S);

    mha_bwd_dQ_simple<<<grid, block, (size_t)dQ_smem, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr, B, H, S);
    CUDA_CHECK(cudaGetLastError());

    mha_bwd_dV_simple<<<grid, block, (size_t)dV_smem, stream>>>(
        Q_ptr, K_ptr, dO_ptr, L_ptr, dV_ptr, B, H, S);
    CUDA_CHECK(cudaGetLastError());

    mha_bwd_dK_simple<<<grid, block, (size_t)dK_smem, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, B, H, S);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_v3::run);

}  // namespace mha_bwd_v3