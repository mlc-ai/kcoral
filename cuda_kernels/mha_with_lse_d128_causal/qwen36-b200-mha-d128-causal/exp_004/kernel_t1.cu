#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cfloat>
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

// Tile dimensions
constexpr uint32_t BM = 128;   // queries per CTA
constexpr uint32_t BN = 128;   // keys per block step
constexpr uint32_t BK = 16;    // D-per-step for QK gemm
constexpr uint32_t BD = 128;   // head dim (must match D)
constexpr uint32_t WARP_SIZE = 32;
constexpr uint32_t NUM_THREADS = BM; // 128 threads = 4 warps
constexpr float NEG_INF = -1e10f;

extern __shared__ char smem_raw[];

__global__ void mha_sm100_kernel(
    const __nv_bfloat16* Q_g,
    const __nv_bfloat16* K_g,
    const __nv_bfloat16* V_g,
    __nv_bfloat16* O_g,
    float* LSE_g,
    int32_t B, int32_t H, int32_t S, int32_t D,
    float scale_factor)
{
    // Shared memory layout (in bytes):
    // smem_q:     BM*BK*2  = 4096   (Q chunk, K-Major)
    // smem_k:     BN*BK*2  = 4096   (K chunk, K-Major)  
    // smem_v:     BD*BN*2  = 32768  (V chunk, transposed: BD rows x BN cols)
    // smem_logit: BM*BN*4  = 65536  (QK^T, fp32)
    // smem_oacc:  BM*BD*4  = 65536  (Output accumulator, fp32)
    // Total: ~172KB
    
    constexpr uint32_t OFF_Q  = 0;
    constexpr uint32_t OFF_K  = BM * BK * 2;
    constexpr uint32_t OFF_V  = OFF_K + BN * BK * 2;
    constexpr uint32_t OFF_LOGIT = OFF_V + BD * BN * 2;
    constexpr uint32_t OFF_OACC = OFF_LOGIT + BM * BN * 4;
    
    auto* smem_q = reinterpret_cast<__nv_bfloat16*>(smem_raw + OFF_Q);
    auto* smem_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + OFF_K);
    auto* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + OFF_V);
    auto* smem_logit = reinterpret_cast<float*>(smem_raw + OFF_LOGIT);
    auto* smem_oacc = reinterpret_cast<float*>(smem_raw + OFF_OACC);
    
    uint32_t tid = threadIdx.x;
    uint32_t q_local = tid; // Each thread owns exactly one query row
    
    // Decode batch/head from blockIdx.y
    uint32_t bh_idx = blockIdx.y;
    uint32_t batch = bh_idx / H;
    uint32_t head = bh_idx % H;
    
    // Query starting position
    uint32_t q_base = blockIdx.x * BM;
    
    // Precompute global strides
    uint64_t bd_stride = (uint64_t)H * S * D; // stride per batch
    uint64_t hd_stride = (uint64_t)S * D;     // stride per head
    uint64_t strid_head = batch * bd_stride + head * hd_stride;
    
    uint32_t num_d_steps = (D + BK - 1) / BK;
    uint32_t num_bn_steps = (S + BN - 1) / BN;
    
    // ---- Clear output accumulator ----
    {
        uint32_t nelem = BM * BD;
        for (uint32_t idx = tid; idx < nelem; idx += NUM_THREADS) {
            smem_oacc[idx] = 0.0f;
        }
        __syncthreads();
    }
    
    // ---- Online softmax state per thread ----
    float row_max = NEG_INF;
    float row_sum = 0.0f;
    
    // ---- Main loop over key/value blocks ----
    for (uint32_t bs = 0; bs < num_bn_steps; ++bs) {
        uint32_t k_base = bs * BN;
        
        // ---- Sub-loop: QK^T accumulation over D dimension ----
        // First clear logit buffer
        {
            uint32_t nelem = BM * BN;
            for (uint32_t idx = tid; idx < nelem; idx += NUM_THREADS) {
                smem_logit[idx] = 0.0f;
            }
            __syncthreads();
        }
        
        for (uint32_t ds = 0; ds < num_d_steps; ++ds) {
            uint32_t d_off = ds * BK;
            
            // Cooperative load Q[BM][BK] into smem_q
            {
                uint32_t nelem = BM * BK;
                uint32_t per_thread = (nelem + NUM_THREADS - 1) / NUM_THREADS; // 16
                for (uint32_t ii = 0; ii < per_thread; ++ii) {
                    uint32_t eid = tid * per_thread + ii;
                    if (eid >= nelem) break;
                    uint32_t r = eid / BK;
                    uint32_t c = eid % BK;
                    uint32_t q_global = q_base + r;
                    uint32_t d_global = d_off + c;
                    if (q_global < S && d_global < D) {
                        smem_q[eid] = Q_g[strid_head + (uint64_t)q_global * D + d_global];
                    } else {
                        smem_q[eid] = __float2bfloat16(0.0f);
                    }
                }
            }
            
            // Cooperative load K[BN][BK] at k_base into smem_k
            {
                uint32_t nelem = BN * BK;
                uint32_t per_thread = (nelem + NUM_THREADS - 1) / NUM_THREADS; // 16
                for (uint32_t ii = 0; ii < per_thread; ++ii) {
                    uint32_t eid = tid * per_thread + ii;
                    if (eid >= nelem) break;
                    uint32_t r = eid / BK;
                    uint32_t c = eid % BK;
                    uint32_t k_global = k_base + r;
                    uint32_t d_global = d_off + c;
                    if (k_global < S && d_global < D) {
                        smem_k[eid] = K_g[strid_head + (uint64_t)k_global * D + d_global];
                    } else {
                        smem_k[eid] = __float2bfloat16(0.0f);
                    }
                }
            }
            
            __syncthreads();
            
            // Inner product GEMM: smem_logit[q_local][kn] += sum_d Q*q[k] * K[kn]*[k]
            for (uint32_t kn = tid; kn < BN; kn += NUM_THREADS) {
                float acc = 0.0f;
                const __nv_bfloat16* __restrict__ qr = smem_q + q_local * BK;
                const __nv_bfloat16* __restrict__ kr = smem_k + kn * BK;
                #pragma unroll
                for (uint32_t dk = 0; dk < BK; ++dk) {
                    float qq = __bfloat162float(qr[dk]);
                    float kk = __bfloat162float(kr[dk]);
                    acc += qq * kk;
                }
                smem_logit[q_local * BN + kn] += acc;
            }
            __syncthreads();
        }
        
        // ---- Apply scale factor and causal mask ----
        {
            uint32_t q_global = q_base + q_local;
            float* __restrict__ lptr = smem_logit + q_local * BN;
            for (uint32_t kn = 0; kn < BN; ++kn) {
                float val = lptr[kn] * scale_factor;
                uint32_t k_global = k_base + kn;
                // Causal: only attend to positions <= query position
                if (k_global > q_global || k_global >= S) {
                    val = NEG_INF;
                }
                lptr[kn] = val;
            }
            __syncthreads();
        }
        
        // ---- Find row max for this block ----
        float cur_block_max = NEG_INF;
        {
            float* __restrict__ lptr = smem_logit + q_local * BN;
            for (uint32_t kn = 0; kn < BN; ++kn) {
                float v = lptr[kn];
                if (v > cur_block_max) cur_block_max = v;
            }
        }
        
        // Handle all-masked case
        bool is_valid_row = (cur_block_max > NEG_INF);
        
        // ---- Online softmax: update row_max, rescale old accumulator ----
        float rescale = 1.0f;
        if (is_valid_row && cur_block_max > row_max) {
            rescale = expf(row_max - cur_block_max);
            // Rescale accumulated output for this query row
            float* __restrict__ optr = smem_oacc + q_local * BD;
            for (uint32_t di = 0; di < BD; ++di) {
                optr[di] *= rescale;
            }
            row_max = cur_block_max;
        }
        __syncthreads();
        
        // ---- Load V[BN][BD] into smem_v (transposed: BD x BN layout) ----
        {
            uint32_t nelem = BN * BD;
            uint32_t per_thread = (nelem + NUM_THREADS - 1) / NUM_THREADS; // 128
            for (uint32_t ii = 0; ii < per_thread; ++ii) {
                uint32_t eid = tid * per_thread + ii;
                if (eid >= nelem) break;
                uint32_t k_in_block = eid / BD; // key index within BN
                uint32_t d_idx = eid % BD;      // d index
                uint32_t k_global = k_base + k_in_block;
                if (k_global < S && d_idx < D) {
                    smem_v[d_idx * BN + k_in_block] = 
                        V_g[strid_head + (uint64_t)k_global * D + d_idx];
                } else {
                    smem_v[d_idx * BN + k_in_block] = __float2bfloat16(0.0f);
                }
            }
            __syncthreads();
        }
        
        // ---- P*V Gemm: accumulate output contribution ----
        float cur_block_sum = 0.0f;
        if (is_valid_row) {
            float* __restrict__ lptr = smem_logit + q_local * BN;
            float* __restrict__ optr = smem_oacc + q_local * BD;
            
            for (uint32_t di = 0; di < BD; ++di) {
                float accum = 0.0f;
                const __nv_bfloat16* __restrict__ vrow = smem_v + di * BN;
                for (uint32_t kn = 0; kn < BN; ++kn) {
                    float logit = lptr[kn];
                    if (logit > NEG_INF) {
                        float pval = expf(logit - cur_block_max);
                        accum += pval * __bfloat162float(vrow[kn]);
                        cur_block_sum += pval;
                    }
                }
                optr[di] += accum * rescale;
            }
        }
        
        // Update online softmax statistics
        row_sum += cur_block_sum * rescale;
    }
    
    // ---- Epilogue: normalize and write output ----
    float final_norm = (row_sum > 0.0f && row_max > NEG_INF) ? (1.0f / row_sum) : 0.0f;
    
    // Write output O[B,H,S,D] in bf16
    {
        float* __restrict__ optr = smem_oacc + q_local * BD;
        uint64_t o_base = strid_head + (uint64_t)(q_base + q_local) * D;
        
        // Vectorized store using uint4 interpretation
        if (q_base + q_local < S) {
            for (uint32_t di = 0; di < BD; di += 2) {
                float f0 = optr[di] * final_norm;
                float f1 = (di + 1 < BD) ? (optr[di+1] * final_norm) : 0.0f;
                if (di < D) {
                    O_g[o_base + di] = __float2bfloat16(f0);
                }
                if (di + 1 < D) {
                    O_g[o_base + di + 1] = __float2bfloat16(f1);
                }
            }
        }
    }
    
    // Write LSE[B,H,S] in fp32
    {
        float lse_val = 0.0f;
        if (row_max > NEG_INF) {
            if (row_sum > 0.0f) {
                lse_val = row_max + logf(row_sum);
            } else {
                lse_val = NEG_INF;
            }
        }
        uint64_t lse_base = (uint64_t)batch * H * S + head * S;
        uint64_t q_global = q_base + q_local;
        if (q_global < S) {
            LSE_g[lse_base + q_global] = lse_val;
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
    
    float scale = 1.0f / sqrtf(static_cast<float>(D));
    
    // Grid: x=Q-tiles, y=batch*heads
    int64_t grid_x = (S + BM - 1) / BM;
    int64_t grid_y = B * H;
    dim3 grid((uint32_t)grid_x, (uint32_t)grid_y, 1);
    dim3 block(NUM_THREADS, 1, 1);
    
    // Shared memory calculation:
    // Q: 128*16*2 = 4096
    // K: 128*16*2 = 4096
    // V: 128*128*2 = 32768
    // Logit: 128*128*4 = 65536
    // OAcc: 128*128*4 = 65536
    // Total: 172032 bytes
    uint32_t smem_bytes = 172032;
    
    cudaStream_t stream = 
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    mha_sm100_kernel<<<grid, block, smem_bytes, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data,
        static_cast<int32_t>(B), static_cast<int32_t>(H),
        static_cast<int32_t>(S), static_cast<int32_t>(D),
        scale);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_blackwell::run);

}  // namespace tvm_ffi_mha_blackwell