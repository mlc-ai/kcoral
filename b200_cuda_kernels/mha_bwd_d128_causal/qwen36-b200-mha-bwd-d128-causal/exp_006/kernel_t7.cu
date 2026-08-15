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

namespace mha_bwd_final {

static constexpr int BK_M   = 64;
static constexpr int BK_N   = 64;
static constexpr int BK_D   = 128;
static constexpr int BK_TPR = 4;
static constexpr int BK_TP  = 256;
static constexpr int BK_DP  = 32;

// ============================================================
// dQ kernel
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

    size_t off = ((size_t)BK_M * 3 + BK_N * 2) * BK_D * sizeof(__nv_bfloat16);
    off = (off + 15) & ~15ULL;

    __nv_bfloat16* sQ   = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sdO  = sQ + BK_M * BK_D;
    __nv_bfloat16* sO   = sdO + BK_M * BK_D;
    __nv_bfloat16* sK   = sO + BK_M * BK_D;
    __nv_bfloat16* sV   = sK + BK_N * BK_D;

    float* fp = reinterpret_cast<float*>(smem_raw + off);
    float* D_part = fp;                                 // [BM*TPR]
    float* D_full = D_part + BK_M * BK_TPR;             // [BM]
    float* score  = D_full + BK_M;                      // [BM*BN]
    float* dp_arr = score + BK_M * BK_N;                // [BM*BN]

    int bh = blockIdx.x;
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
    int num_mq = (S + BK_M - 1) / BK_M;
    int num_mk = (S + BK_N - 1) / BK_N;

    for (int mq = 0; mq < num_mq; mq++) {
        int sq_base = mq * BK_M;
        int cur_bm = min(BK_M, S - sq_base);

        #pragma unroll
        for (int di = 0; di < BK_DP; di++) reg_dQ[di] = 0.0f;

        // Load Q, dO, O tiles
        for (int i = tid; i < cur_bm * BK_D; i += BK_TP) {
            int r = i / BK_D, c = i % BK_D;
            int offset = (int)(sq_base + r) * BK_D + c;
            sQ[i]   = Q[base_bh + (size_t)offset];
            sdO[i]  = dO[base_bh + (size_t)offset];
            sO[i]   = O[base_bh + (size_t)offset];
        }
        __syncthreads();

        for (int mk = 0; mk < num_mk; mk++) {
            int sk_base = mk * BK_N;
            int cur_bn = min(BK_N, S - sk_base);
            if (sk_base >= sq_base + cur_bm) break;

            // Load K, V tiles
            for (int i = tid; i < cur_bn * BK_D; i += BK_TP) {
                int r = i / BK_D, c = i % BK_D;
                int offset = (int)(sk_base + r) * BK_D + c;
                sK[i] = K[base_bh + (size_t)offset];
                sV[i] = V[base_bh + (size_t)offset];
            }
            __syncthreads();

            // Compute D partials
            for (int qr = my_qr; qr < cur_bm; qr += BK_TPR) {
                float s = 0.0f;
                int j0 = my_ln * BK_DP;
                int j1 = j0 + BK_DP;
                for (int j = j0; j < j1 && j < BK_D; j++) {
                    s += __bfloat162float(sdO[qr * BK_D + j]) * __bfloat162float(sO[qr * BK_D + j]);
                }
                D_part[qr * BK_TPR + my_ln] = s;
            }
            __syncthreads();

            // Reduce D
            if (my_ln == 0) {
                for (int qr = my_qr; qr < cur_bm; qr += BK_TPR) {
                    D_full[qr] = D_part[qr * BK_TPR + 0] + D_part[qr * BK_TPR + 1]
                               + D_part[qr * BK_TPR + 2] + D_part[qr * BK_TPR + 3];
                }
            }
            __syncthreads();

            // Compute scores and dp
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
// dV kernel
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

    size_t off = ((size_t)BK_M * 2 + BK_N) * BK_D * sizeof(__nv_bfloat16);
    off = (off + 15) & ~15ULL;

    __nv_bfloat16* sQ   = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sdO  = sQ + BK_M * BK_D;
    __nv_bfloat16* sK   = sdO + BK_M * BK_D;

    float* fp = reinterpret_cast<float*>(smem_raw + off);
    float* score  = fp;                             // [BM*BN]
    float* dv_acc = score + BK_M * BK_N;            // [BN*BD]

    int bh = blockIdx.x;
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

        // Load K tile
        for (int i = tid; i < cur_bn * BK_D; i += BK_TP) {
            int r = i / BK_D, c = i % BK_D;
            int offset = (int)(sk_base + r) * BK_D + c;
            sK[i] = K[base_bh + (size_t)offset];
        }
        __syncthreads();

        int mq_start = (sk_base + BK_M - 1) / BK_M;
        for (int mq = mq_start; mq < num_mq; mq++) {
            int sq_base = mq * BK_M;
            int cur_bm = min(BK_M, S - sq_base);

            // Load Q, dO tiles
            for (int i = tid; i < cur_bm * BK_D; i += BK_TP) {
                int r = i / BK_D, c = i % BK_D;
                int offset = (int)(sq_base + r) * BK_D + c;
                sQ[i]   = Q[base_bh + (size_t)offset];
                sdO[i]  = dO[base_bh + (size_t)offset];
            }
            __syncthreads();

            // Compute scores
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

            // Accumulate dV
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

        // Write dV
        for (int i = tid; i < cur_bn * BK_D; i += BK_TP) {
            int kr = i / BK_D, dim = i % BK_D;
            dV[base_bh + (size_t)(sk_base + kr) * BK_D + dim] = __float2bfloat16(dv_acc[i]);
        }
    }
}

// ============================================================
// dK kernel
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

    size_t off = ((size_t)BK_M * 3 + BK_N * 2) * BK_D * sizeof(__nv_bfloat16);
    off = (off + 15) & ~15ULL;

    __nv_bfloat16* sQ   = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sdO  = sQ + BK_M * BK_D;
    __nv_bfloat16* sO   = sdO + BK_M * BK_D;
    __nv_bfloat16* sK   = sO + BK_M * BK_D;
    __nv_bfloat16* sV   = sK + BK_N * BK_D;

    float* fp = reinterpret_cast<float*>(smem_raw + off);
    float* D_part  = fp;                            // [BM*TPR]
    float* D_full  = D_part + BK_M * BK_TPR;        // [BM]
    float* score   = D_full + BK_M;                 // [BM*BN]
    float* dp_arr  = score + BK_M * BK_N;           // [BM*BN]
    float* dk_acc  = dp_arr + BK_M * BK_N;          // [BN*BD]

    int bh = blockIdx.x;
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

        // Zero dk_acc
        for (int i = tid; i < cur_bn * BK_D; i += BK_TP) {
            dk_acc[i] = 0.0f;
        }
        __syncthreads();

        // Load K, V tiles
        for (int i = tid; i < cur_bn * BK_D; i += BK_TP) {
            int r = i / BK_D, c = i % BK_D;
            int offset = (int)(sk_base + r) * BK_D + c;
            sK[i] = K[base_bh + (size_t)offset];
            sV[i] = V[base_bh + (size_t)offset];
        }
        __syncthreads();

        int mq_start = (sk_base + BK_M - 1) / BK_M;
        for (int mq = mq_start; mq < num_mq; mq++) {
            int sq_base = mq * BK_M;
            int cur_bm = min(BK_M, S - sq_base);

            // Load Q, dO, O tiles
            for (int i = tid; i < cur_bm * BK_D; i += BK_TP) {
                int r = i / BK_D, c = i % BK_D;
                int offset = (int)(sq_base + r) * BK_D + c;
                sQ[i]   = Q[base_bh + (size_t)offset];
                sdO[i]  = dO[base_bh + (size_t)offset];
                sO[i]   = O[base_bh + (size_t)offset];
            }
            __syncthreads();

            // D computation
            for (int qr = my_qr; qr < cur_bm; qr += BK_TPR) {
                float s = 0.0f;
                int j0 = my_ln * BK_DP;
                int j1 = j0 + BK_DP;
                for (int j = j0; j < j1 && j < BK_D; j++) {
                    s += __bfloat162float(sdO[qr * BK_D + j]) * __bfloat162float(sO[qr * BK_D + j]);
                }
                D_part[qr * BK_TPR + my_ln] = s;
            }
            __syncthreads();

            if (my_ln == 0) {
                for (int qr = my_qr; qr < cur_bm; qr += BK_TPR) {
                    D_full[qr] = D_part[qr * BK_TPR + 0] + D_part[qr * BK_TPR + 1]
                               + D_part[qr * BK_TPR + 2] + D_part[qr * BK_TPR + 3];
                }
            }
            __syncthreads();

            // Scores and dp
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

        // Write dK
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

    // Shared memory sizes computed carefully
    // dQ/dK: (3*BM + 2*BN)*D*2 bf16 + (BM*TPR + BM + 2*BM*BN)[+]BN*BD for dK f32
    size_t dQ_off = ((size_t)BK_M * 3 + BK_N * 2) * BK_D * sizeof(__nv_bfloat16);
    dQ_off = (dQ_off + 15) & ~15ULL;
    size_t dQ_f32 = (size_t)BK_M * BK_TPR + BK_M + 2UL * BK_M * BK_N;
    size_t dQ_smem = dQ_off + dQ_f32 * sizeof(float);

    size_t dV_off = ((size_t)BK_M * 2 + BK_N) * BK_D * sizeof(__nv_bfloat16);
    dV_off = (dV_off + 15) & ~15ULL;
    size_t dV_f32 = (size_t)BK_M * BK_N + (size_t)BK_N * BK_D;
    size_t dV_smem = dV_off + dV_f32 * sizeof(float);

    size_t dK_off = dQ_off;
    size_t dK_f32 = dQ_f32 + (size_t)BK_N * BK_D;
    size_t dK_smem = dK_off + dK_f32 * sizeof(float);

    dim3 grid(B * H);
    dim3 block(BK_TP);

    mha_bwd_dQ_kernel<<<grid, block, dQ_smem, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr, B, H, S);
    CUDA_CHECK(cudaGetLastError());

    mha_bwd_dV_kernel<<<grid, block, dV_smem, stream>>>(
        Q_ptr, K_ptr, dO_ptr, L_ptr, dV_ptr, B, H, S);
    CUDA_CHECK(cudaGetLastError());

    mha_bwd_dK_kernel<<<grid, block, dK_smem, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, B, H, S);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_final::run);

}  // namespace mha_bwd_final