#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <math.h>
#include <float.h>
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

namespace flash_mha_d128 {

static constexpr uint32_t BM = 64;
static constexpr uint32_t BN = 64;
static constexpr uint32_t WARP_SIZE = 32;
static constexpr uint32_t NUM_THREADS = 64;
static constexpr uint32_t COLS_PER_THREAD = 2;

__global__ void flash_mha_kernel(
    const __nv_bfloat16* __restrict__ Q_g,
    const __nv_bfloat16* __restrict__ K_g,
    const __nv_bfloat16* __restrict__ V_g,
    __nv_bfloat16* __restrict__ O_g,
    float* __restrict__ LSE_g,
    int B, int H, int S, int D,
    float inv_sqrt_D,
    int64_t stride_Q_B, int64_t stride_Q_H, int64_t stride_Q_S, int64_t stride_Q_D,
    int64_t stride_K_B, int64_t stride_K_H, int64_t stride_K_S, int64_t stride_K_D,
    int64_t stride_V_B, int64_t stride_V_H, int64_t stride_V_S, int64_t stride_V_D,
    int64_t stride_O_B, int64_t stride_O_H, int64_t stride_O_S, int64_t stride_O_D,
    int64_t stride_LSE_B, int64_t stride_LSE_H, int64_t stride_LSE_S
) {
    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* sQ  = &smem[0];
    __nv_bfloat16* sK  = &smem[BM * D];
    __nv_bfloat16* sV  = &smem[(BM + BN) * D];

    int blk = blockIdx.x;
    int num_blocks_per_bh = (S + BM - 1) / BM;
    int bh = blk / num_blocks_per_bh;
    int b = bh / H;
    int h = bh % H;
    int q_tile = blk % num_blocks_per_bh;
    
    int q_base = q_tile * BM;
    int tid = threadIdx.x;
    
    int64_t offQ = (int64_t)b * stride_Q_B + h * stride_Q_H + q_base * stride_Q_S;
    int64_t offO = (int64_t)b * stride_O_B + h * stride_O_H + q_base * stride_O_S;
    int64_t offLSE = (int64_t)b * stride_LSE_B + h * stride_LSE_H + q_base * stride_LSE_S;

    int my_q = tid;
    bool valid_thread = (my_q < BM && q_base + my_q < S);

    // Load my Q row into registers (all threads load to share in shared mem)
    float q_reg[COLS_PER_THREAD * 2] = {};
    if (valid_thread) {
        #pragma unroll
        for (int ci = 0; ci < COLS_PER_THREAD; ci++) {
            int col = tid + ci * NUM_THREADS;
            int64_t q_idx = offQ + my_q * stride_Q_S + col * stride_Q_D;
            __nv_bfloat162 qv = reinterpret_cast<const __nv_bfloat162*>(Q_g)[(size_t)q_idx];
            q_reg[ci * 2]     = __bfloat162float(qv.x);
            q_reg[ci * 2 + 1] = __bfloat162float(qv.y);
        }
    }

    // Load Q into shared memory
    if (valid_thread) {
        #pragma unroll
        for (int ci = 0; ci < COLS_PER_THREAD; ci++) {
            int col = tid + ci * NUM_THREADS;
            int64_t q_idx = offQ + my_q * stride_Q_S + col * stride_Q_D;
            __nv_bfloat162 qv = reinterpret_cast<const __nv_bfloat162*>(Q_g)[(size_t)q_idx];
            sQ[my_q * D + col]     = qv.x;
            sQ[my_q * D + col + 1] = qv.y;
        }
    }
    __syncthreads();

    float row_max_val = -FLT_MAX;
    float row_sum_val = 1.0f;
    float o_reg[COLS_PER_THREAD * 2] = {};

    int num_kv_tiles = (S + BN - 1) / BN;
    
    if (valid_thread) {
        for (int kv = 0; kv < num_kv_tiles; kv++) {
            int k_base = kv * BN;
            
            // Load K tile
            int64_t offK = (int64_t)b * stride_K_B + h * stride_K_H + k_base * stride_K_S;
            #pragma unroll
            for (int ci = 0; ci < COLS_PER_THREAD; ci++) {
                int col = tid + ci * NUM_THREADS;
                for (int kr = tid; kr < BN; kr += NUM_THREADS) {
                    int64_t k_idx = offK + kr * stride_K_S + col * stride_K_D;
                    __nv_bfloat162 kv_data = reinterpret_cast<const __nv_bfloat162*>(K_g)[(size_t)k_idx];
                    sK[kr * D + col]     = kv_data.x;
                    sK[kr * D + col + 1] = kv_data.y;
                }
            }

            // Load V tile
            int64_t offV = (int64_t)b * stride_V_B + h * stride_V_H + k_base * stride_V_S;
            #pragma unroll
            for (int ci = 0; ci < COLS_PER_THREAD; ci++) {
                int col = tid + ci * NUM_THREADS;
                for (int vr = tid; vr < BN; vr += NUM_THREADS) {
                    int64_t v_idx = offV + vr * stride_V_S + col * stride_V_D;
                    __nv_bfloat162 vv_data = reinterpret_cast<const __nv_bfloat162*>(V_g)[(size_t)v_idx];
                    sV[vr * D + col]     = vv_data.x;
                    sV[vr * D + col + 1] = vv_data.y;
                }
            }
            __syncthreads();

            // Compute S = Q @ K^T
            float s_vals[BN];
            #pragma unroll
            for (int kr = 0; kr < BN; kr++) {
                float s = 0.0f;
                #pragma unroll
                for (int ci = 0; ci < COLS_PER_THREAD; ci++) {
                    int cidx = ci * 2;
                    int col = tid + ci * NUM_THREADS;
                    __nv_bfloat162 k2 = *reinterpret_cast<__nv_bfloat162*>(sK + kr * D + col);
                    s += q_reg[cidx]     * __bfloat162float(k2.x);
                    s += q_reg[cidx + 1] * __bfloat162float(k2.y);
                }
                s_vals[kr] = (k_base + kr < S) ? s * inv_sqrt_D : -FLT_MAX;
            }

            // Row max
            float m_local = -FLT_MAX;
            #pragma unroll
            for (int kr = 0; kr < BN; kr++) {
                m_local = max(m_local, s_vals[kr]);
            }

            float m_new = m_local;
            float old_scale = expf(row_max_val - m_new);
            #pragma unroll
            for (int ci = 0; ci < COLS_PER_THREAD * 2; ci++) {
                o_reg[ci] *= old_scale;
            }
            float p_sum_old = row_sum_val * old_scale;

            float p_sum_new = 0.0f;
            float exp_vals[BN];
            #pragma unroll
            for (int kr = 0; kr < BN; kr++) {
                exp_vals[kr] = expf(s_vals[kr] - m_new);
                p_sum_new += exp_vals[kr];
            }

            // Accumulate o_reg += exp(S-m) * V
            #pragma unroll
            for (int kr = 0; kr < BN; kr++) {
                float p_kr = exp_vals[kr];
                #pragma unroll
                for (int ci = 0; ci < COLS_PER_THREAD; ci++) {
                    int cidx = ci * 2;
                    int col = tid + ci * NUM_THREADS;
                    __nv_bfloat162 v2 = *reinterpret_cast<__nv_bfloat162*>(sV + kr * D + col);
                    o_reg[cidx]     += p_kr * __bfloat162float(v2.x);
                    o_reg[cidx + 1] += p_kr * __bfloat162float(v2.y);
                }
            }

            row_sum_val = p_sum_old + p_sum_new;
            row_max_val = m_new;

            __syncthreads();
        }

        // Epilogue
        float inv_lse = 1.0f / row_sum_val;
        float lse_out = row_max_val + logf(row_sum_val);
        
        int64_t o_base_idx = offO;
        #pragma unroll
        for (int ci = 0; ci < COLS_PER_THREAD; ci++) {
            int cidx = ci * 2;
            int col = tid + ci * NUM_THREADS;
            float f0 = o_reg[cidx] * inv_lse;
            float f1 = o_reg[cidx + 1] * inv_lse;
            __nv_bfloat162 ov;
            ov.x = __float2bfloat16(f0);
            ov.y = __float2bfloat16(f1);
            int64_t o_idx = o_base_idx + my_q * stride_O_S + col * stride_O_D;
            reinterpret_cast<__nv_bfloat162*>(O_g)[(size_t)o_idx] = ov;
        }

        int64_t lse_idx = offLSE + my_q * stride_LSE_S;
        LSE_g[(size_t)lse_idx] = lse_out;
    } else {
        // Invalid threads still need to participate in syncs
        for (int kv = 0; kv < num_kv_tiles; kv++) {
            // Load K tile
            int64_t offK = (int64_t)b * stride_K_B + h * stride_K_H + kv * BN * stride_K_S;
            #pragma unroll
            for (int ci = 0; ci < COLS_PER_THREAD; ci++) {
                int col = tid + ci * NUM_THREADS;
                for (int kr = tid; kr < BN; kr += NUM_THREADS) {
                    int64_t k_idx = offK + kr * stride_K_S + col * stride_K_D;
                    __nv_bfloat162 kv_data = reinterpret_cast<const __nv_bfloat162*>(K_g)[(size_t)k_idx];
                    sK[kr * D + col]     = kv_data.x;
                    sK[kr * D + col + 1] = kv_data.y;
                }
            }

            // Load V tile
            int64_t offV = (int64_t)b * stride_V_B + h * stride_V_H + kv * BN * stride_V_S;
            #pragma unroll
            for (int ci = 0; ci < COLS_PER_THREAD; ci++) {
                int col = tid + ci * NUM_THREADS;
                for (int vr = tid; vr < BN; vr += NUM_THREADS) {
                    int64_t v_idx = offV + vr * stride_V_S + col * stride_V_D;
                    __nv_bfloat162 vv_data = reinterpret_cast<const __nv_bfloat162*>(V_g)[(size_t)v_idx];
                    sV[vr * D + col]     = vv_data.x;
                    sV[vr * D + col + 1] = vv_data.y;
                }
            }
            __syncthreads();
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());

    int64_t stride_Q_B = Q.stride(0);
    int64_t stride_Q_H = Q.stride(1);
    int64_t stride_Q_S = Q.stride(2);
    int64_t stride_Q_D = Q.stride(3);
    int64_t stride_K_B = K.stride(0);
    int64_t stride_K_H = K.stride(1);
    int64_t stride_K_S = K.stride(2);
    int64_t stride_K_D = K.stride(3);
    int64_t stride_V_B = V.stride(0);
    int64_t stride_V_H = V.stride(1);
    int64_t stride_V_S = V.stride(2);
    int64_t stride_V_D = V.stride(3);
    int64_t stride_O_B = O.stride(0);
    int64_t stride_O_H = O.stride(1);
    int64_t stride_O_S = O.stride(2);
    int64_t stride_O_D = O.stride(3);
    int64_t stride_LSE_B = LSE.stride(0);
    int64_t stride_LSE_H = LSE.stride(1);
    int64_t stride_LSE_S = LSE.stride(2);

    float inv_sqrt_D = 1.0f / sqrtf((float)D);

    int64_t num_blocks_per_bh = (S + BM - 1) / BM;
    int64_t total_blocks = B * H * num_blocks_per_bh;
    
    size_t smem_bytes = (BM + 2 * BN) * D * sizeof(__nv_bfloat16);

    dim3 grid((unsigned int)total_blocks);
    dim3 block(NUM_THREADS);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    flash_mha_kernel<<<grid, block, (size_t)smem_bytes, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data,
        (int)B, (int)H, (int)S, (int)D,
        inv_sqrt_D,
        stride_Q_B, stride_Q_H, stride_Q_S, stride_Q_D,
        stride_K_B, stride_K_H, stride_K_S, stride_K_D,
        stride_V_B, stride_V_H, stride_V_S, stride_V_D,
        stride_O_B, stride_O_H, stride_O_S, stride_O_D,
        stride_LSE_B, stride_LSE_H, stride_LSE_S
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flash_mha_d128::run);

} // namespace flash_mha_d128