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
static constexpr uint32_t BN = 32;
static constexpr uint32_t NT = 128;
static constexpr uint32_t DVAL = 128;
// 128 threads share 16 rows → 8 copies per row, each copy owns 128/8 = 16 columns
static constexpr uint32_t COLS_PER_THREAD = DVAL / (NT / BM); // = 16

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

    // Thread assignment
    int my_q = tid % BM;
    int thread_in_group = tid / BM;  // 0..7
    bool valid = (my_q < BM && q_base + my_q < S);
    
    // Each thread owns 16 columns of the output accumulator for its query row
    int start_col = thread_in_group * COLS_PER_THREAD;
    
    // Online softmax state
    float m_prev = -FLT_MAX;
    float l_prev = 1.0f;
    float o_acc[COLS_PER_THREAD]; // 16 floats
    
    if (valid) {
        #pragma unroll
        for (uint32_t i = 0; i < COLS_PER_THREAD; i++) {
            o_acc[i] = 0.0f;
        }
    }

    int nktiles = (S + BN - 1) / BN;

    for (int kt = 0; kt < nktiles; kt++) {
        int k_base = kt * BN;

        // === Load K and V tiles: ALL threads participate ===
        // First BN threads load one K/V row each
        // Remaining threads help with extra loads
        
        // Load K
        if (tid < BN) {
            int kr = tid;
            int64_t kb_r = bk_off + k_base * stride_K_S + kr * stride_K_S;
            #pragma unroll
            for (int d = 0; d < DVAL; d += 2) {
                __nv_bfloat162 kv = reinterpret_cast<const __nv_bfloat162*>(K_g)[(size_t)((kb_r + d * stride_K_D) / sizeof(__nv_bfloat16))];
                sK[kr * DVAL + d]     = kv.x;
                sK[kr * DVAL + d + 1] = kv.y;
            }
        } else if (tid < BN * 2) {
            int kr = tid - BN;
            int64_t kb_r = bk_off + k_base * stride_K_S + kr * stride_K_S;
            #pragma unroll
            for (int d = 0; d < DVAL; d += 2) {
                __nv_bfloat162 kv = reinterpret_cast<const __nv_bfloat162*>(K_g)[(size_t)((kb_r + d * stride_K_D) / sizeof(__nv_bfloat16))];
                sK[kr * DVAL + d]     = kv.x;
                sK[kr * DVAL + d + 1] = kv.y;
            }
        }

        // Load V
        if (tid < BN) {
            int vr = tid;
            int64_t vb_r = bv_off + k_base * stride_V_S + vr * stride_V_S;
            #pragma unroll
            for (int d = 0; d < DVAL; d += 2) {
                __nv_bfloat162 vv = reinterpret_cast<const __nv_bfloat162*>(V_g)[(size_t)((vb_r + d * stride_V_D) / sizeof(__nv_bfloat16))];
                sV[vr * DVAL + d]     = vv.x;
                sV[vr * DVAL + d + 1] = vv.y;
            }
        } else if (tid < BN * 2) {
            int vr = tid - BN;
            int64_t vb_r = bv_off + k_base * stride_V_S + vr * stride_V_S;
            #pragma unroll
            for (int d = 0; d < DVAL; d += 2) {
                __nv_bfloat162 vv = reinterpret_cast<const __nv_bfloat162*>(V_g)[(size_t)((vb_r + d * stride_V_D) / sizeof(__nv_bfloat16))];
                sV[vr * DVAL + d]     = vv.x;
                sV[vr * DVAL + d + 1] = vv.y;
            }
        }

        __syncthreads();

        // === Softmax computation (only valid threads compute, but all must reach barrier) ===
        if (valid) {
            const __nv_bfloat16* qrow = sQ + my_q * DVAL;
            
            // Pass 1: find m_new
            float m_new = -FLT_MAX;
            #pragma unroll
            for (int kr = 0; kr < BN; kr++) {
                float s = 0.0f;
                const __nv_bfloat16* krow = sK + kr * DVAL;
                #pragma unroll
                for (int d = 0; d < DVAL; d++) {
                    s += __bfloat162float(qrow[d]) * __bfloat162float(krow[d]);
                }
                if (k_base + kr < S) {
                    s *= inv_sqrt_D;
                    if (s > m_new) m_new = s;
                }
            }

            // Scale previous accumulation
            float alpha = expf(m_prev - m_new);
            #pragma unroll
            for (uint32_t i = 0; i < COLS_PER_THREAD; i++) {
                o_acc[i] *= alpha;
            }
            float l_old = l_prev * alpha;

            // Pass 2: accumulate P@V
            float l_new = 0.0f;
            #pragma unroll
            for (int kr = 0; kr < BN; kr++) {
                float s = 0.0f;
                const __nv_bfloat16* krow = sK + kr * DVAL;
                #pragma unroll
                for (int d = 0; d < DVAL; d++) {
                    s += __bfloat162float(qrow[d]) * __bfloat162float(krow[d]);
                }
                
                if (k_base + kr < S) {
                    s *= inv_sqrt_D;
                    float p_val = expf(s - m_new);
                    l_new += p_val;
                    
                    const __nv_bfloat16* vrow = sV + kr * DVAL;
                    #pragma unroll
                    for (uint32_t i = 0; i < COLS_PER_THREAD; i++) {
                        int d = start_col + i;
                        o_acc[i] += p_val * __bfloat162float(vrow[d]);
                    }
                }
            }

            l_prev = l_old + l_new;
            m_prev = m_new;
        }

        __syncthreads();
    }

    // === Epilogue ===
    if (valid) {
        float inv_lse = 1.0f / l_prev;
        float lse_val = m_prev + logf(l_prev);
        
        #pragma unroll
        for (uint32_t i = 0; i < COLS_PER_THREAD; i++) {
            int col = start_col + i;
            float val = o_acc[i] * inv_lse;
            int64_t oidx = bo_off + my_q * stride_O_S + col * stride_O_D;
            O_g[oidx] = __float2bfloat16(val);
        }
        
        // Only one thread per query row writes LSE
        if (thread_in_group == 0) {
            LSE_g[bl_off + my_q * stride_LSE_S] = lse_val;
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

    int64_t sqB = Q.stride(0), sqH = Q.stride(1), sqS = Q.stride(2), sqD = Q.stride(3);
    int64_t skB = K.stride(0), skH = K.stride(1), skS = K.stride(2), skD = K.stride(3);
    int64_t svB = V.stride(0), svH = V.stride(1), svS = V.stride(2), svD = V.stride(3);
    int64_t soB = O.stride(0), soH = O.stride(1), soS = O.stride(2), soD = O.stride(3);
    int64_t slB = LSE.stride(0), slH = LSE.stride(1), slS = LSE.stride(2);

    float inv_sqrt_D = 1.0f / sqrtf((float)D);

    int64_t nqblocks = (S + BM - 1) / BM;
    int64_t total_blocks = B * H * nqblocks;

    size_t smem_bytes = (BM + 2 * BN) * D * sizeof(__nv_bfloat16);

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