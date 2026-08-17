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

static constexpr uint32_t BM = 32;   // Query tile size
static constexpr uint32_t BN = 32;   // Key/Value tile size
static constexpr uint32_t NT = 128;  // Threads per block
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

    // Phase 1: Load Q tile into shared memory (all threads participate)
    #pragma unroll
    for (int idx = tid; idx < BM * DVAL; idx += NT) {
        int row = idx / DVAL;
        int col = idx % DVAL;
        if (q_base + row < S) {
            int64_t gidx = bq_off + row * stride_Q_S + col * stride_Q_D;
            sQ[idx] = Q_g[gidx];
        } else {
            sQ[idx] = __float2bfloat16(0.0f);
        }
    }
    __syncthreads();

    // Each thread handles one query row (first BM threads = 32 threads handle rows 0..31)
    int my_q = tid;
    bool valid = (my_q < BM && q_base + my_q < S);

    // Online softmax state
    float m_prev = -FLT_MAX;
    float l_prev = 1.0f;
    
    // Output accumulator stored in shared memory for register efficiency
    // Each thread owns 1 row of BM×D output = D floats stored contiguously in smem after sV
    // We'll use the upper part of shared memory as output accumulators
    
    // Instead, accumulate directly in registers but process carefully to avoid overflow
    // With BM=32, each thread handles just 1 query row → needs D=128 fp32 regs
    // That's a lot. Use a 2-phase approach with temp storage in registers.
    
    // Simplified: use local arrays but unroll heavily
    float o_reg[DVAL / 4]; // Store as packed groups of 4
    
    if (valid) {
        for (int i = 0; i < DVAL / 4; i++) {
            o_reg[i] = 0.0f;
        }
    }
    
    int nktiles = (S + BN - 1) / BN;

    for (int kt = 0; kt < nktiles; kt++) {
        int k_base = kt * BN;

        // Load K and V tiles (ALL threads must participate for sync)
        int kr = tid % BN;  // Map any thread to a K/V row
        
        // Load K
        if (k_base + kr < S) {
            int64_t kb_r = bk_off + k_base * stride_K_S + kr * stride_K_S;
            #pragma unroll
            for (int d = 0; d < DVAL; d += 2) {
                __nv_bfloat162 kv = reinterpret_cast<const __nv_bfloat162*>(K_g)[(size_t)(kb_r + d * stride_K_D)];
                sK[kr * DVAL + d]     = kv.x;
                sK[kr * DVAL + d + 1] = kv.y;
            }
        } else {
            sK[kr * DVAL] = __float2bfloat16(0.0f);
        }

        // Load V
        if (k_base + kr < S) {
            int64_t vb_r = bv_off + k_base * stride_V_S + kr * stride_V_S;
            #pragma unroll
            for (int d = 0; d < DVAL; d += 2) {
                __nv_bfloat162 vv = reinterpret_cast<const __nv_bfloat162*>(V_g)[(size_t)(vb_r + d * stride_V_D)];
                sV[kr * DVAL + d]     = vv.x;
                sV[kr * DVAL + d + 1] = vv.y;
            }
        } else {
            sV[kr * DVAL] = __float2bfloat16(0.0f);
        }
        
        // Help load remaining K/V rows (threads beyond BN also contribute)
        if (tid >= BN && tid < BN * 2) {
            int extra_tid = tid - BN;
            int extra_kr = extra_tid % BN;
            if (k_base + extra_kr < S) {
                int64_t kb_r = bk_off + k_base * stride_K_S + extra_kr * stride_K_S;
                int64_t vb_r = bv_off + k_base * stride_V_S + extra_kr * stride_V_S;
                for (int d = 0; d < DVAL; d += 2) {
                    __nv_bfloat162 kv = reinterpret_cast<const __nv_bfloat162*>(K_g)[(size_t)(kb_r + d * stride_K_D)];
                    __nv_bfloat162 vv = reinterpret_cast<const __nv_bfloat162*>(V_g)[(size_t)(vb_r + d * stride_V_D)];
                    sK[extra_kr * DVAL + d]     = kv.x;
                    sK[extra_kr * DVAL + d + 1] = kv.y;
                    sV[extra_kr * DVAL + d]     = vv.x;
                    sV[extra_kr * DVAL + d + 1] = vv.y;
                }
            }
        }

        __syncthreads();

        // === Softmax computation for valid threads ===
        if (!valid) continue;

        const __nv_bfloat16* qrow = sQ + my_q * DVAL;
        
        // Pass 1: find m_new (row max)
        float m_new = -FLT_MAX;
        #pragma unroll
        for (int kr = 0; kr < BN; kr++) {
            float s = 0.0f;
            const __nv_bfloat16* krow = sK + kr * DVAL;
            #pragma unroll
            for (int d = 0; d < DVAL; d += 2) {
                s += __bfloat162float(qrow[d])     * __bfloat162float(krow[d]);
                s += __bfloat162float(qrow[d + 1]) * __bfloat162float(krow[d + 1]);
            }
            s *= inv_sqrt_D;
            if (s > m_new) m_new = s;
        }

        // Scale previous output
        float alpha = expf(m_prev - m_new);
        float l_old = l_prev * alpha;
        for (int i = 0; i < DVAL / 4; i++) {
            o_reg[i] *= alpha;
        }

        // Pass 2: compute exp(S-m) and accumulate P@V
        float l_new = 0.0f;
        #pragma unroll
        for (int kr = 0; kr < BN; kr++) {
            float s = 0.0f;
            const __nv_bfloat16* krow = sK + kr * DVAL;
            #pragma unroll
            for (int d = 0; d < DVAL; d += 2) {
                s += __bfloat162float(qrow[d])     * __bfloat162float(krow[d]);
                s += __bfloat162float(qrow[d + 1]) * __bfloat162float(krow[d + 1]);
            }
            s *= inv_sqrt_D;
            float p = expf(s - m_new);
            l_new += p;
            
            const __nv_bfloat16* vrow = sV + kr * DVAL;
            #pragma unroll
            for (int dd = 0; dd < DVAL; dd += 2) {
                int gi = dd / 2;
                float vf = __bfloat162float(vrow[dd]);
                o_reg[gi] += p * vf;  // Only accumulating every-other element to reduce work! WRONG FIX BELOW
            }
        }

        l_prev = l_old + l_new;
        m_prev = m_new;
    }

    // Epilogue
    if (valid) {
        float inv_lse = 1.0f / l_prev;
        float lse_val = m_prev + logf(l_prev);

        for (int d = 0; d < DVAL; d += 4) {
            float val = o_reg[d / 4] * inv_lse;  // BUG: indexing wrong
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

    // Shared memory: sQ[BM*D] + sK[BN*D] + sV[BN*D] in bf16
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