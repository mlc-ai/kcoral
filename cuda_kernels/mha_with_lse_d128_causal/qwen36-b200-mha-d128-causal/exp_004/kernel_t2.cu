#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_mha_blackwell {

constexpr uint32_t BM = 128;   // queries per CTA
constexpr uint32_t BN = 128;   // keys per block step
constexpr uint32_t BK = 16;    // D-per-step for QK gemm
constexpr uint32_t NUM_THREADS = 128;
constexpr float NEG_INF = -1e10f;

extern __shared__ char smem_raw[];

__global__ void mha_kernel(
    const __nv_bfloat16* __restrict__ Q_g,
    const __nv_bfloat16* __restrict__ K_g,
    const __nv_bfloat16* __restrict__ V_g,
    __nv_bfloat16* __restrict__ O_g,
    float* __restrict__ LSE_g,
    int32_t B, int32_t H, int32_t S, int32_t D,
    float scale_factor)
{
    // Shared memory layout (bytes):
    // smem_q:     BM*BK*2  = 4096
    // smem_k:     BN*BK*2  = 4096
    // smem_v:     D*BN*2   = 32768
    // smem_logit: BM*BN*4  = 65536
    // smem_oacc:  BM*D*4   = 65536
    // Total: ~172032
    
    constexpr uint32_t OFF_Q     = 0;
    constexpr uint32_t OFF_K     = BM * BK * 2;
    constexpr uint32_t OFF_V     = OFF_K + BN * BK * 2;
    constexpr uint32_t OFF_LOGIT = OFF_V + (uint32_t)D * BN * 2;
    constexpr uint32_t OFF_OACC  = OFF_LOGIT + BM * BN * 4;
    
    auto* smem_q = reinterpret_cast<__nv_bfloat16*>(smem_raw + OFF_Q);
    auto* smem_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + OFF_K);
    auto* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + OFF_V);
    auto* smem_logit = reinterpret_cast<float*>(smem_raw + OFF_LOGIT);
    auto* smem_oacc = reinterpret_cast<float*>(smem_raw + OFF_OACC);
    
    uint32_t tid = threadIdx.x;
    uint32_t q_local = tid;
    
    uint32_t bh_idx = blockIdx.y;
    uint32_t batch = bh_idx / (uint32_t)H;
    uint32_t head = bh_idx % (uint32_t)H;
    uint32_t q_base = blockIdx.x * BM;
    
    uint64_t strid_head = (uint64_t)batch * H * S * D + (uint64_t)head * S * D;
    
    uint32_t num_d_steps = (D + BK - 1) / BK;
    uint32_t num_bn_steps = (S + BN - 1) / BN;
    
    // Clear output accumulator
    for (uint32_t idx = tid; idx < BM * (uint32_t)D; idx += NUM_THREADS) {
        smem_oacc[idx] = 0.0f;
    }
    __syncthreads();
    
    float row_max = NEG_INF;
    float row_sum = 0.0f;
    
    for (uint32_t bs = 0; bs < num_bn_steps; ++bs) {
        uint32_t k_base = bs * BN;
        
        // Clear logit buffer
        for (uint32_t idx = tid; idx < BM * BN; idx += NUM_THREADS) {
            smem_logit[idx] = 0.0f;
        }
        __syncthreads();
        
        // Accumulate QK^T over D dimension
        for (uint32_t ds = 0; ds < num_d_steps; ++ds) {
            uint32_t d_off = ds * BK;
            
            // Load Q[BM][BK] cooperatively
            for (uint32_t idx = tid; idx < BM * BK; idx += NUM_THREADS) {
                uint32_t r = idx / BK;
                uint32_t c = idx % BK;
                uint32_t qg = q_base + r;
                uint32_t dg = d_off + c;
                smem_q[idx] = (qg < (uint32_t)S && dg < (uint32_t)D)
                    ? Q_g[strid_head + (uint64_t)qg * D + dg]
                    : __float2bfloat16(0.0f);
            }
            
            // Load K[BN][BK] cooperatively
            for (uint32_t idx = tid; idx < BN * BK; idx += NUM_THREADS) {
                uint32_t r = idx / BK;
                uint32_t c = idx % BK;
                uint32_t kg = k_base + r;
                uint32_t dg = d_off + c;
                smem_k[idx] = (kg < (uint32_t)S && dg < (uint32_t)D)
                    ? K_g[strid_head + (uint64_t)kg * D + dg]
                    : __float2bfloat16(0.0f);
            }
            
            __syncthreads();
            
            // GEMM contribution
            for (uint32_t kn = tid; kn < BN; kn += NUM_THREADS) {
                float acc = 0.0f;
                #pragma unroll
                for (uint32_t dk = 0; dk < BK; ++dk) {
                    acc += __bfloat162float(smem_q[q_local * BK + dk]) 
                         * __bfloat162float(smem_k[kn * BK + dk]);
                }
                smem_logit[q_local * BN + kn] += acc;
            }
            __syncthreads();
        }
        
        // Scale and apply causal mask
        uint32_t q_global = q_base + q_local;
        for (uint32_t kn = 0; kn < BN; ++kn) {
            float val = smem_logit[q_local * BN + kn] * scale_factor;
            if (k_base + kn > q_global || k_base + kn >= (uint32_t)S) {
                val = NEG_INF;
            }
            smem_logit[q_local * BN + kn] = val;
        }
        __syncthreads();
        
        // Find row max
        float cur_block_max = NEG_INF;
        for (uint32_t kn = 0; kn < BN; ++kn) {
            float v = smem_logit[q_local * BN + kn];
            if (v > cur_block_max) cur_block_max = v;
        }
        bool is_valid = (cur_block_max > NEG_INF);
        
        // Online softmax: rescale accumulator if needed
        float rescale = 1.0f;
        if (is_valid && cur_block_max > row_max) {
            rescale = expf(row_max - cur_block_max);
            for (uint32_t di = 0; di < (uint32_t)D; ++di) {
                smem_oacc[q_local * (uint32_t)D + di] *= rescale;
            }
            row_max = cur_block_max;
        }
        __syncthreads();
        
        // Load V[BN][D] -> transposed layout [D][BN]
        for (uint32_t idx = tid; idx < BN * (uint32_t)D; idx += NUM_THREADS) {
            uint32_t ki = idx / (uint32_t)D;
            uint32_t di = idx % (uint32_t)D;
            uint32_t kg = k_base + ki;
            smem_v[di * BN + ki] = (kg < (uint32_t)S)
                ? V_g[strid_head + (uint64_t)kg * D + di]
                : __float2bfloat16(0.0f);
        }
        __syncthreads();
        
        // PV Gemm with softmax
        float block_sum = 0.0f;
        if (is_valid) {
            for (uint32_t di = 0; di < (uint32_t)D; ++di) {
                float accum = 0.0f;
                for (uint32_t kn = 0; kn < BN; ++kn) {
                    float logit = smem_logit[q_local * BN + kn];
                    if (logit > NEG_INF) {
                        float pval = expf(logit - cur_block_max);
                        accum += pval * __bfloat162float(smem_v[di * BN + kn]);
                        block_sum += pval;
                    }
                }
                smem_oacc[q_local * (uint32_t)D + di] += accum * rescale;
            }
        }
        
        row_sum += block_sum * rescale;
    }
    
    // Epilogue
    float final_norm = (row_sum > 0.0f && row_max > NEG_INF) ? (1.0f / row_sum) : 0.0f;
    
    if (q_base + q_local < (uint32_t)S) {
        for (uint32_t di = 0; di < (uint32_t)D; ++di) {
            O_g[strid_head + (uint64_t)(q_base + q_local) * D + di]
                = __float2bfloat16(smem_oacc[q_local * (uint32_t)D + di] * final_norm);
        }
        float lse = (row_max > NEG_INF)
            ? (row_sum > 0.0f ? (row_max + logf(row_sum)) : NEG_INF)
            : NEG_INF;
        uint64_t lse_idx = (uint64_t)batch * H * S + (uint64_t)head * S + (q_base + q_local);
        LSE_g[lse_idx] = lse;
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
    
    float scale = 1.0f / sqrtf(static_cast<float>(D));
    
    uint32_t grid_x = (uint32_t)((S + BM - 1) / BM);
    uint32_t grid_y = (uint32_t)(B * H);
    dim3 grid(grid_x, grid_y, 1);
    dim3 block(NUM_THREADS, 1, 1);
    
    // Shared memory: 4096 + 4096 + D*BN*2 + BM*BN*4 + BM*D*4
    uint32_t smem_bytes = (uint32_t)(BM * BK * 2 + BN * BK * 2
                                    + (uint32_t)D * BN * 2
                                    + BM * BN * 4
                                    + BM * (uint32_t)D * 4 + 16);
    
    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    mha_kernel<<<grid, block, smem_bytes, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data,
        static_cast<int32_t>(B), static_cast<int32_t>(H),
        static_cast<int32_t>(S), static_cast<int32_t>(D),
        scale);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_blackwell::run);

}  // namespace tvm_ffi_mha_blackwell