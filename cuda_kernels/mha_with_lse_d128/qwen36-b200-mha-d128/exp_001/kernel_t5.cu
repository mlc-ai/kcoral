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

static constexpr uint32_t BM = 16;   // Query tile size  
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

    // === Load Q tile: ALL threads participate ===
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

    // === Main KV tiling loop ===
    int my_q = tid % BM;  // Which query row this thread computes
    bool valid = (my_q < BM && q_base + my_q < S);

    // Use minimal registers: store output accumulators in registers but only for active rows
    // With BM=16 and 128 threads, 8 threads share each row. 
    // Each thread contributes D/8=16 columns to its row's accumulator.
    // So we need 16 fp32 values per thread for o_reg. That's manageable.
    
    uint32_t cols_per_thread = DVAL / (NT / BM); // 128/8 = 16
    float m_prev = -FLT_MAX;
    float l_prev = 1.0f;
    
    float o_acc[cols_per_thread];
    if (valid) {
        for (uint32_t i = 0; i < cols_per_thread; i++) {
            o_acc[i] = 0.0f;
        }
    }

    int nktiles = (S + BN - 1) / BN;

    for (int kt = 0; kt < nktiles; kt++) {
        int k_base = kt * BN;

        // === Load K and V tiles: ALL threads must participate ===
        // Each thread handles one K/V row (tid < BN) plus helps extra
        
        // Load K: thread tid -> K row (tid % BN), column range based on tid
        if (tid < BN) {
            int kr = tid;
            int64_t kb_r = bk_off + k_base * stride_K_S + kr * stride_K_S;
            for (int d = 0; d < DVAL; d++) {
                int src_col = d;
                int64_t gidx = kb_r + src_col * stride_K_D;
                sK[kr * DVAL + d] = K_g[gidx];
            }
        } else {
            // Extra threads also help load K rows
            int extra_tid = tid - BN;
            if (extra_tid < BN) {
                int kr = extra_tid;
                int64_t kb_r = bk_off + k_base * stride_K_S + kr * stride_K_S;
                for (int d = 0; d < DVAL; d++) {
                    sK[kr * DVAL + d] = K_g[kb_r + d * stride_K_D];
                }
            }
        }

        // Load V similarly
        if (tid < BN) {
            int vr = tid;
            int64_t vb_r = bv_off + k_base * stride_V_S + vr * stride_V_S;
            for (int d = 0; d < DVAL; d++) {
                sV[vr * DVAL + d] = V_g[vb_r + d * stride_V_D];
            }
        } else {
            int extra_tid = tid - BN;
            if (extra_tid < BN) {
                int vr = extra_tid;
                int64_t vb_r = bv_off + k_base * stride_V_S + vr * stride_V_S;
                for (int d = 0; d < DVAL; d++) {
                    sV[vr * DVAL + d] = V_g[vb_r + d * stride_V_D];
                }
            }
        }

        __syncthreads();

        // === Softmax + accumulate for valid threads only ===
        if (!valid) continue;

        const __nv_bfloat16* qrow = sQ + my_q * DVAL;
        
        // Determine which columns this thread owns for the output
        // 128 threads, BM=16 rows → 8 threads per row
        // Thread group for row my_q starts at (my_q * (NT/BM))
        // Within group, thread index = tid / BM ... no, tid % BM = my_q means 8 copies of each row
        // We need to figure out which copy. Let's use: group_idx = tid / BM within stride
        // Actually: threads 0..15 handle row 0..15 (copy 0), 16..31 handle row 0..15 (copy 1), etc.
        int thread_in_group = tid / BM; // 0..7 for each copy of a row
        int start_col = thread_in_group * cols_per_thread;
        
        // Pass 1: Find m_new for this query row
        float m_new = -FLT_MAX;
        #pragma unroll
        for (int kr = 0; kr < BN; kr++) {
            float s = 0.0f;
            const __nv_bfloat16* krow = sK + kr * DVAL;
            for (int d = 0; d < DVAL; d++) {
                s += __bfloat162float(qrow[d]) * __bfloat162float(krow[d]);
            }
            if (k_base + kr < S) {
                s *= inv_sqrt_D;
                if (s > m_new) m_new = s;
            }
        }

        // Scale previous output accumulators
        float alpha = expf(m_prev - m_new);
        for (uint32_t i = 0; i < cols_per_thread; i++) {
            o_acc[i] *= alpha;
        }
        float l_old = l_prev * alpha;

        // Pass 2: Compute P@V contribution
        float l_new = 0.0f;
        #pragma unroll
        for (int kr = 0; kr < BN; kr++) {
            float s = 0.0f;
            const __nv_bfloat16* krow = sK + kr * DVAL;
            for (int d = 0; d < DVAL; d++) {
                s += __bfloat162float(qrow[d]) * __bfloat162float(krow[d]);
            }
            
            float p_val = 0.0f;
            if (k_base + kr < S) {
                s *= inv_sqrt_D;
                p_val = expf(s - m_new);
                l_new += p_val;
                
                // Accumulate into owned columns only
                const __nv_bfloat16* vrow = sV + kr * DVAL;
                for (uint32_t i = 0; i < cols_per_thread; i++) {
                    int d = start_col + i;
                    o_acc[i] += p_val * __bfloat162float(vrow[d]);
                }
            }
        }

        l_prev = l_old + l_new;
        m_prev = m_new;
        
        // All valid threads sync before next iteration (to reuse shared memory safely)
        __syncthreads();
    }

    // === Epilogue: normalize and write output ===
    if (valid) {
        float inv_lse = 1.0f / l_prev;
        float lse_val = m_prev + logf(l_prev);
        
        int thread_in_group = tid / BM;
        int start_col = thread_in_group * cols_per_thread;
        
        for (uint32_t i = 0; i < cols_per_thread; i++) {
            int col = start_col + i;
            float val = o_acc[i] * inv_lse;
            int64_t oidx = bo_off + my_q * stride_O_S + col * stride_O_D;
            O_g[oidx] = __float2bfloat16(val);
        }
        
        // Write LSE (only one copy needed)
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