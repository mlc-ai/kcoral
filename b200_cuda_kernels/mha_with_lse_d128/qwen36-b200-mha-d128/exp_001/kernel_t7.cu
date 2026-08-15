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

static constexpr uint32_t BM = 16;
static constexpr uint32_t BN = 64;
static constexpr uint32_t NT = 128;
static constexpr uint32_t DVAL = 128;

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
    extern __shared__ char smem_char[];
    alignas(16) __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem_char);
    alignas(16) __nv_bfloat16* sK = sQ + BM * DVAL;
    alignas(16) __nv_bfloat16* sV = sK + BN * DVAL;

    int nqblocks = (S + BM - 1) / BM;
    int bh = blockIdx.x / nqblocks;
    int b = bh / H;
    int h = bh % H;
    int qb = blockIdx.x % nqblocks;
    int q_base = qb * BM;
    int tid = threadIdx.x;

    int64_t bq_off = (int64_t)b * stride_Q_B + h * stride_Q_H + q_base * stride_Q_S;
    int64_t bk_off = (int64_t)b * stride_K_B + h * stride_K_H;
    int64_t bv_off = (int64_t)b * stride_V_B + h * stride_V_H;
    int64_t bo_off = (int64_t)b * stride_O_B + h * stride_O_H + q_base * stride_O_S;
    int64_t bl_off = (int64_t)b * stride_LSE_B + h * stride_LSE_H + q_base * stride_LSE_S;

    // === Load Q tile: ALL threads ===
    #pragma unroll
    for (int idx = tid; idx < BM * DVAL; idx += NT) {
        int row = idx / DVAL;
        int col = idx % DVAL;
        if (q_base + row < S && col < D) {
            sQ[idx] = Q_g[bq_off + row * stride_Q_S + col * stride_Q_D];
        } else {
            sQ[idx] = __float2bfloat16(0.0f);
        }
    }
    __syncthreads();

    // 1 thread per query row (first BM=16 threads compute softmax for their row)
    int my_q = tid % BM;
    bool valid = (my_q < BM && q_base + my_q < S);

    // Online softmax state
    float m_prev = -FLT_MAX;
    float l_prev = 1.0f;
    
    // Output accumulator stored in shared memory after sV to avoid register pressure
    // Layout: o_smem[BM][D] as float32
    float* o_smem = reinterpret_cast<float*>(sV + BN * DVAL);

    // Initialize output accumulators
    #pragma unroll
    for (int i = tid; i < BM * DVAL; i += NT) {
        o_smem[i] = 0.0f;
    }
    __syncthreads();

    int nktiles = (S + BN - 1) / BN;

    for (int kt = 0; kt < nktiles; kt++) {
        int k_base = kt * BN;

        // === Load K tile: all NT threads participate ===
        // Map threads to K/V rows with some threads loading multiple
        if (tid < BN) {
            int kr = tid;
            int64_t kb_r = bk_off + k_base * stride_K_S + kr * stride_K_S;
            // Vectorized load: 64 bf16 elements (32 x bf162)
            #pragma unroll
            for (int d = 0; d < DVAL; d += 2) {
                __nv_bfloat162 kv = reinterpret_cast<const __nv_bfloat162*>(K_g)[(kb_r + d * stride_K_D) / sizeof(__nv_bfloat16)];
                sK[kr * DVAL + d]     = kv.x;
                sK[kr * DVAL + d + 1] = kv.y;
            }
        } else {
            // Threads BN..NT help load remaining or idle
            int extra_kr = tid - BN;
            if (extra_kr < BN) {
                int64_t kb_r = bk_off + k_base * stride_K_S + extra_kr * stride_K_S;
                #pragma unroll
                for (int d = 0; d < DVAL; d += 2) {
                    __nv_bfloat162 kv = reinterpret_cast<const __nv_bfloat162*>(K_g)[(kb_r + d * stride_K_D) / sizeof(__nv_bfloat16)];
                    sK[extra_kr * DVAL + d]     = kv.x;
                    sK[extra_kr * DVAL + d + 1] = kv.y;
                }
            }
        }

        // === Load V tile: same pattern ===
        if (tid < BN) {
            int vr = tid;
            int64_t vb_r = bv_off + k_base * stride_V_S + vr * stride_V_S;
            #pragma unroll
            for (int d = 0; d < DVAL; d += 2) {
                __nv_bfloat162 vv = reinterpret_cast<const __nv_bfloat162*>(V_g)[(vb_r + d * stride_V_D) / sizeof(__nv_bfloat16)];
                sV[vr * DVAL + d]     = vv.x;
                sV[vr * DVAL + d + 1] = vv.y;
            }
        } else {
            int extra_vr = tid - BN;
            if (extra_vr < BN) {
                int64_t vb_r = bv_off + k_base * stride_V_S + extra_vr * stride_V_S;
                #pragma unroll
                for (int d = 0; d < DVAL; d += 2) {
                    __nv_bfloat162 vv = reinterpret_cast<const __nv_bfloat162*>(V_g)[(vb_r + d * stride_V_D) / sizeof(__nv_bfloat16)];
                    sV[extra_vr * DVAL + d]     = vv.x;
                    sV[extra_vr * DVAL + d + 1] = vv.y;
                }
            }
        }

        __syncthreads();

        // === Softmax + accumulation (only first BM threads compute) ===
        if (!valid) continue;

        const __nv_bfloat16* qrow = sQ + my_q * DVAL;
        float* o_row = o_smem + my_q * DVAL;

        // Pass 1: find m_new (row max of scores)
        float m_new = -FLT_MAX;
        #pragma unroll
        for (int kr = 0; kr < BN; kr++) {
            float s = 0.0f;
            const __nv_bfloat16* krow = sK + kr * DVAL;
            // Manual unrolled dot product - process 4 at a time to limit registers
            for (int d = 0; d < DVAL; d += 4) {
                s += __bfloat162float(qrow[d])     * __bfloat162float(krow[d]);
                s += __bfloat162float(qrow[d + 1]) * __bfloat162float(krow[d + 1]);
                s += __bfloat162float(qrow[d + 2]) * __bfloat162float(krow[d + 2]);
                s += __bfloat162float(qrow[d + 3]) * __bfloat162float(krow[d + 3]);
            }
            if (k_base + kr < S) {
                s *= inv_sqrt_D;
                if (s > m_new) m_new = s;
            }
        }

        // Scale previous output accumulators (in shared mem)
        float alpha = expf(m_prev - m_new);
        float l_old = l_prev * alpha;
        #pragma unroll
        for (int d = 0; d < DVAL; d++) {
            o_row[d] *= alpha;
        }

        // Pass 2: compute P@V contribution  
        float l_new = 0.0f;
        #pragma unroll
        for (int kr = 0; kr < BN; kr++) {
            float s = 0.0f;
            const __nv_bfloat16* krow = sK + kr * DVAL;
            for (int d = 0; d < DVAL; d += 4) {
                s += __bfloat162float(qrow[d])     * __bfloat162float(krow[d]);
                s += __bfloat162float(qrow[d + 1]) * __bfloat162float(krow[d + 1]);
                s += __bfloat162float(qrow[d + 2]) * __bfloat162float(krow[d + 2]);
                s += __bfloat162float(qrow[d + 3]) * __bfloat162float(krow[d + 3]);
            }
            
            if (k_base + kr < S) {
                s *= inv_sqrt_D;
                float p_val = expf(s - m_new);
                l_new += p_val;
                
                // Accumulate p_val * vrow into o_row
                const __nv_bfloat16* vrow = sV + kr * DVAL;
                for (int d = 0; d < DVAL; d += 4) {
                    o_row[d]     += p_val * __bfloat162float(vrow[d]);
                    o_row[d + 1] += p_val * __bfloat162float(vrow[d + 1]);
                    o_row[d + 2] += p_val * __bfloat162float(vrow[d + 2]);
                    o_row[d + 3] += p_val * __bfloat162float(vrow[d + 3]);
                }
            }
        }

        l_prev = l_old + l_new;
        m_prev = m_new;

        __syncthreads();
    }

    // === Epilogue: normalize and write output ===
    // All threads participate in writing output
    #pragma unroll
    for (int idx = tid; idx < BM * DVAL; idx += NT) {
        int row = idx / DVAL;
        int col = idx % DVAL;
        if (q_base + row < S) {
            float val = o_smem[idx] / l_prev_arr[idx / DVAL];  // Need per-row l_prev...
            // This approach doesn't work since l_prev is per-thread
        }
    }
    // Fix: only the computing thread writes its own row's output
    if (valid) {
        float inv_lse = 1.0f / l_prev;
        float lse_val = m_prev + logf(l_prev);
        
        float* o_row = o_smem + my_q * DVAL;
        #pragma unroll
        for (int d = 0; d < DVAL; d++) {
            float val = o_row[d] * inv_lse;
            int64_t oidx = bo_off + my_q * stride_O_S + d * stride_O_D;
            O_g[oidx] = __float2bfloat16(val);
        }
        LSE_g[bl_off + my_q * stride_LSE_S] = lse_val;
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

    int64_t sqB = Q.stride(0), sqH = Q.stride(1), sqS = Q.stride(2), sqD = Q.stride(3);
    int64_t skB = K.stride(0), skH = K.stride(1), skS = K.stride(2), skD = K.stride(3);
    int64_t svB = V.stride(0), svH = V.stride(1), svS = V.stride(2), svD = V.stride(3);
    int64_t soB = O.stride(0), soH = O.stride(1), soS = O.stride(2), soD = O.stride(3);
    int64_t slB = LSE.stride(0), slH = LSE.stride(1), slS = LSE.stride(2);

    float inv_sqrt_D = 1.0f / sqrtf((float)D);

    int64_t nqblocks = (S + BM - 1) / BM;
    int64_t total_blocks = B * H * nqblocks;

    // Shared memory: sQ[BM*D bf16] + sK[BN*D bf16] + sV[BN*D bf16] + o_smem[BM*D float32]
    size_t smem_bytes = (BM + BN) * D * sizeof(__nv_bfloat16) + 
                        BN * D * sizeof(__nv_bfloat16) +
                        BM * D * sizeof(float);

    dim3 grid((unsigned int)total_blocks);
    dim3 block(NT);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    flash_mha_kernel<<<grid, block, smem_bytes, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data,
        (int)B, (int)H, (int)S, (int)D,
        inv_sqrt_D,
        sqB, sqH, sqS, sqD, skB, skH, skS, skD,
        svB, svH, svS, svD, soB, soH, soS, soD,
        slB, slH, slS
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flash_mha_d128::run);

} // namespace flash_mha_d128