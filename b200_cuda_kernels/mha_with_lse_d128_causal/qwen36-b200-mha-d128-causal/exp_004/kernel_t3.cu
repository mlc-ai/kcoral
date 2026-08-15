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
constexpr uint32_t BD = 128;   // head dimension (fixed by task)
constexpr uint32_t NUM_THREADS = 128;
constexpr float NEG_INF = -1e10f;

// Shared memory layout (bytes) - all compile-time known
constexpr uint32_t OFF_Q     = 0;
constexpr uint32_t OFF_K     = BM * BK * 2;                          // 4096
constexpr uint32_t OFF_V     = OFF_K + BN * BK * 2;                 // 8192
constexpr uint32_t OFF_LOGIT = OFF_V + BD * BN * 2;                 // 40960
constexpr uint32_t OFF_OACC  = OFF_LOGIT + BM * BN * 4;             // 106496
constexpr uint32_t SMEM_SIZE = OFF_OACC + BM * BD * 4;              // 172032

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
    auto* smem_q = reinterpret_cast<__nv_bfloat16*>(smem_raw + OFF_Q);
    auto* smem_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + OFF_K);
    auto* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + OFF_V);
    float* smem_logit = reinterpret_cast<float*>(smem_raw + OFF_LOGIT);
    float* smem_oacc  = reinterpret_cast<float*>(smem_raw + OFF_OACC);
    
    uint32_t tid = threadIdx.x;
    uint32_t q_local = tid;
    
    uint32_t bh_idx = blockIdx.y;
    uint32_t batch = bh_idx / (uint32_t)H;
    uint32_t head = bh_idx % (uint32_t)H;
    uint32_t q_base = blockIdx.x * BM;
    
    uint64_t strid_head = (uint64_t)batch * (uint64_t)H * (uint64_t)S * (uint64_t)D 
                        + (uint64_t)head * (uint64_t)S * (uint64_t)D;
    
    uint32_t num_d_steps = (D + BK - 1) / BK;
    uint32_t num_bn_steps = (S + BN - 1) / BN;
    
    // Clear output accumulator
    for (uint32_t idx = tid; idx < BM * BD; idx += NUM_THREADS) {
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
                    ? Q_g[strid_head + (uint64_t)qg * (uint64_t)D + dg]
                    : __float2bfloat16(0.0f);
            }
            
            // Load K[BN][BK] cooperatively
            for (uint32_t idx = tid; idx < BN * BK; idx += NUM_THREADS) {
                uint32_t r = idx / BK;
                uint32_t c = idx % BK;
                uint32_t kg = k_base + r;
                uint32_t dg = d_off + c;
                smem_k[idx] = (kg < (uint32_t)S && dg < (uint32_t)D)
                    ? K_g[strid_head + (uint64_t)kg * (uint64_t)D + dg]
                    : __float2bfloat16(0.0f);
            }
            
            __syncthreads();
            
            // GEMM contribution: smem_logit[q_local][kn] += sum_d Q*q[d] * K[kn]*[d]
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
        
        // Scale and apply causal mask (per-thread on its own row)
        uint32_t q_global = q_base + q_local;
        float* lrow = smem_logit + q_local * BN;
        for (uint32_t kn = 0; kn < BN; ++kn) {
            float val = lrow[kn] * scale_factor;
            if (k_base + kn > q_global || k_base + kn >= (uint32_t)S) {
                val = NEG_INF;
            }
            lrow[kn] = val;
        }
        __syncthreads();
        
        // Find row max for this block
        float cur_block_max = NEG_INF;
        for (uint32_t kn = 0; kn < BN; ++kn) {
            float v = lrow[kn];
            if (v > cur_block_max) cur_block_max = v;
        }
        bool is_valid = (cur_block_max > NEG_INF);
        
        // Online softmax: rescale accumulator if new max exceeds old
        float rescale = 1.0f;
        if (is_valid && cur_block_max > row_max) {
            rescale = expf(row_max - cur_block_max);
            float* optr = smem_oacc + q_local * BD;
            for (uint32_t di = 0; di < BD; ++di) {
                optr[di] *= rescale;
            }
            row_max = cur_block_max;
        }
        __syncthreads();
        
        // Load V[BN][BD] into transposed layout [BD][BN]
        for (uint32_t idx = tid; idx < BN * BD; idx += NUM_THREADS) {
            uint32_t ki = idx / BD;
            uint32_t di = idx % BD;
            uint32_t kg = k_base + ki;
            smem_v[di * BN + ki] = (kg < (uint32_t)S)
                ? V_g[strid_head + (uint64_t)kg * (uint64_t)D + di]
                : __float2bfloat16(0.0f);
        }
        __syncthreads();
        
        // PV Gemm with softmax probabilities
        float block_sum = 0.0f;
        if (is_valid) {
            float* optr = smem_oacc + q_local * BD;
            for (uint32_t di = 0; di < BD; ++di) {
                float accum = 0.0f;
                const __nv_bfloat16* vrow = smem_v + di * BN;
                for (uint32_t kn = 0; kn < BN; ++kn) {
                    float logit = lrow[kn];
                    if (logit > NEG_INF) {
                        float pval = expf(logit - cur_block_max);
                        accum += pval * __bfloat162float(vrow[kn]);
                        block_sum += pval;
                    }
                }
                optr[di] += accum * rescale;
            }
        }
        
        row_sum += block_sum * rescale;
    }
    
    // Epilogue: normalize and write output
    float final_norm = (row_sum > 0.0f && row_max > NEG_INF) ? (1.0f / row_sum) : 0.0f;
    uint32_t q_global_final = q_base + q_local;
    
    if (q_global_final < (uint32_t)S) {
        float* optr = smem_oacc + q_local * BD;
        uint64_t o_offset = strid_head + (uint64_t)q_global_final * (uint64_t)D;
        for (uint32_t di = 0; di < BD; ++di) {
            O_g[o_offset + di] = __float2bfloat16(optr[di] * final_norm);
        }
        float lse = (row_max > NEG_INF)
            ? (row_sum > 0.0f ? (row_max + logf(row_sum)) : NEG_INF)
            : NEG_INF;
        uint64_t lse_idx = (uint64_t)batch * (uint64_t)H * (uint64_t)S 
                         + (uint64_t)head * (uint64_t)S + q_global_final;
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
    
    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    mha_kernel<<<grid, block, SMEM_SIZE, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data,
        static_cast<int32_t>(B), static_cast<int32_t>(H),
        static_cast<int32_t>(S), static_cast<int32_t>(D),
        scale);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_blackwell::run);

}  // namespace tvm_ffi_mha_blackwell