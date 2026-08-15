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

// Tiles: process 64 queries x 64 keys per block, head dim fixed at 128
static const uint32_t BM = 64;   // query tile height
static const uint32_t BN = 64;   // key tile width
static const uint32_t BK = 16;   // D-chunk size for QK accumulation
static const uint32_t BD = 128;  // Head dimension
static const uint32_t NUM_THREADS = 256;
static const float NEG_INF_F = -1e10f;

extern __shared__ char smem_raw[];

__global__ void mha_kernel(
    const __nv_bfloat16* __restrict__ Q_g,
    const __nv_bfloat16* __restrict__ K_g,
    const __nv_bfloat16* __restrict__ V_g,
    __nv_bfloat16* __restrict__ O_g,
    float*            * LSE_g,
    int32_t B, int32_t H, int32_t S, int32_t D,
    float scale_factor)
{
    // Shared memory regions (each in bytes):
    // sm_q:   BM*BK*2  = 64*16*2  = 2048
    // sm_k:   BN*BK*2  = 64*16*2  = 2048
    // sm_v:   BD*BN*2  = 128*64*2 = 16384
    // sm_L:   BM*BN*4  = 64*64*4  = 16384
    // sm_O:   BM*BD*4  = 64*128*4 = 32768
    // Total:  70632 bytes (~69 KB, well within limits)
    
    uint32_t off_q  = 0;
    uint32_t off_k  = BM * BK * 2;
    uint32_t off_v  = off_k + BN * BK * 2;
    uint32_t off_L  = off_v + BD * BN * 2;
    uint32_t off_O  = off_L + BM * BN * 4;
    
    __nv_bfloat16* sm_q = reinterpret_cast<__nv_bfloat16*>(smem_raw + off_q);
    __nv_bfloat16* sm_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + off_k);
    __nv_bfloat16* sm_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + off_v);
    float* sm_L = reinterpret_cast<float*>(smem_raw + off_L);
    float* sm_O = reinterpret_cast<float*>(smem_raw + off_O);
    
    uint32_t tid = threadIdx.x;
    uint32_t q_local = tid % BM;  // Each thread owns one query row
    
    uint32_t bh_idx = blockIdx.y;
    uint32_t batch = bh_idx / (uint32_t)H;
    uint32_t head  = bh_idx % (uint32_t)H;
    uint32_t q_base = blockIdx.x * BM;
    
    uint64_t base_addr = (uint64_t)batch * (uint64_t)H * (uint64_t)S * (uint64_t)D
                       + (uint64_t)head  * (uint64_t)S * (uint64_t)D;
    
    uint32_t n_d_steps  = (D + BK - 1) / BK;
    uint32_t n_kv_steps = (S + BN - 1) / BN;
    
    // Zero output accumulator
    {
        uint32_t tot = BM * BD;
        for (uint32_t i = tid; i < tot; i += NUM_THREADS) {
            sm_O[i] = 0.0f;
        }
    }
    __syncthreads();
    
    float reg_max  = NEG_INF_F;
    float reg_sum  = 0.0f;
    
    for (uint32_t step = 0; step < n_kv_steps; ++step) {
        uint32_t k_start = step * BN;
        
        // Zero logits
        {
            uint32_t tot = BM * BN;
            for (uint32_t i = tid; i < tot; i += NUM_THREADS) {
                sm_L[i] = 0.0f;
            }
        }
        __syncthreads();
        
        // Q @ K^T accumulation
        for (uint32_t ds = 0; ds < n_d_steps; ++ds) {
            uint32_t d_off = ds * BK;
            
            // Load Q chunk [BM][BK]
            {
                uint32_t tot = BM * BK;
                for (uint32_t i = tid; i < tot; i += NUM_THREADS) {
                    uint32_t r = i / BK;
                    uint32_t c = i % BK;
                    uint32_t qr = q_base + r;
                    uint32_t dc = d_off + c;
                    sm_q[i] = (qr < (uint32_t)S && dc < (uint32_t)D)
                              ? Q_g[base_addr + (uint64_t)qr * (uint64_t)D + dc]
                              : __float2bfloat16(0.0f);
                }
            }
            // Load K chunk [BN][BK]
            {
                uint32_t tot = BN * BK;
                for (uint32_t i = tid; i < tot; i += NUM_THREADS) {
                    uint32_t r = i / BK;
                    uint32_t c = i % BK;
                    uint32_t kr = k_start + r;
                    uint32_t dc = d_off + c;
                    sm_k[i] = (kr < (uint32_t)S && dc < (uint32_t)D)
                              ? K_g[base_addr + (uint64_t)kr * (uint64_t)D + dc]
                              : __float2bfloat16(0.0f);
                }
            }
            __syncthreads();
            
            // Accumulate dot products
            for (uint32_t kn = tid; kn < BN; kn += NUM_THREADS) {
                float dot = 0.0f;
                const uint32_t qi = q_local * BK;
                const uint32_t ki = kn * BK;
                for (uint32_t dk = 0; dk < BK; ++dk) {
                    dot += __bfloat162float(sm_q[qi + dk])
                         * __bfloat162float(sm_k[ki + dk]);
                }
                sm_L[q_local * BN + kn] += dot;
            }
            __syncthreads();
        }
        
        // Scale + causal mask (per-thread on its own row)
        {
            uint32_t qg = q_base + q_local;
            uint32_t li = q_local * BN;
            for (uint32_t kn = 0; kn < BN; ++kn) {
                float v = sm_L[li + kn] * scale_factor;
                uint32_t kg = k_start + kn;
                if (kg > qg || kg >= (uint32_t)S) v = NEG_INF_F;
                sm_L[li + kn] = v;
            }
        }
        __syncthreads();
        
        // Reduce row max
        float blk_max = NEG_INF_F;
        {
            uint32_t li = q_local * BN;
            for (uint32_t kn = 0; kn < BN; ++kn) {
                float v = sm_L[li + kn];
                if (v > blk_max) blk_max = v;
            }
        }
        bool valid = (blk_max > NEG_INF_F);
        
        // Rescale old accumulator if needed
        float alpha = 1.0f;
        if (valid && blk_max > reg_max) {
            alpha = expf(reg_max - blk_max);
            uint32_t oi = q_local * BD;
            for (uint32_t di = 0; di < BD; ++di) {
                sm_O[oi + di] *= alpha;
            }
            reg_max = blk_max;
        }
        __syncthreads();
        
        // Load V chunk [BN][BD], transposed to [BD][BN]
        {
            uint32_t tot = BN * BD;
            for (uint32_t i = tid; i < tot; i += NUM_THREADS) {
                uint32_t kr = i / BD;
                uint32_t dc = i % BD;
                uint32_t kg = k_start + kr;
                sm_v[dc * BN + kr] = (kg < (uint32_t)S)
                                     ? V_g[base_addr + (uint64_t)kg * (uint64_t)D + dc]
                                     : __float2bfloat16(0.0f);
            }
        }
        __syncthreads();
        
        // P @ V accumulation with softmax
        float blk_sum = 0.0f;
        if (valid) {
            uint32_t li = q_local * BN;
            uint32_t oi = q_local * BD;
            for (uint32_t di = 0; di < BD; ++di) {
                float acc = 0.0f;
                uint32_t vi = di * BN;
                for (uint32_t kn = 0; kn < BN; ++kn) {
                    float lg = sm_L[li + kn];
                    if (lg > NEG_INF_F) {
                        float p = expf(lg - blk_max);
                        acc += p * __bfloat162float(sm_v[vi + kn]);
                        blk_sum += p;
                    }
                }
                sm_O[oi + di] += acc * alpha;
            }
        }
        reg_sum += blk_sum * alpha;
    }
    
    // Epilogue
    float norm = (reg_sum > 0.0f && reg_max > NEG_INF_F) ? (1.0f / reg_sum) : 0.0f;
    uint32_t qg = q_base + q_local;
    
    if (qg < (uint32_t)S) {
        uint32_t oi = q_local * BD;
        uint64_t obase = base_addr + (uint64_t)qg * (uint64_t)D;
        for (uint32_t di = 0; di < BD; ++di) {
            O_g[obase + di] = __float2bfloat16(sm_O[oi + di] * norm);
        }
        float lse = (reg_max > NEG_INF_F)
                   ? (reg_sum > 0.0f ? (reg_max + logf(reg_sum)) : NEG_INF_F)
                   : NEG_INF_F;
        uint64_t lidx = (uint64_t)batch * (uint64_t)H * (uint64_t)S
                      + (uint64_t)head * (uint64_t)S + qg;
        LSE_g[lidx] = lse;
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
        tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    const __nv_bfloat16* Q_p = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_p = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_p = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_p       = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_p             = static_cast<float*>(LSE.data_ptr());
    
    float sc = 1.0f / sqrtf(static_cast<float>(D));
    
    uint32_t gx = (uint32_t)((S + BM - 1) / BM);
    uint32_t gy = (uint32_t)(B * H);
    
    dim3 block(NUM_THREADS, 1, 1);
    dim3 grid(gx, gy, 1);
    
    // Shared memory: 70632 bytes
    uint32_t smem_sz = BM*BK*2 + BN*BK*2 + BD*BN*2 + BM*BN*4 + BM*BD*4 + 16;
    
    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    mha_kernel<<<grid, block, smem_sz, stream>>>(
        Q_p, K_p, V_p, O_p, LSE_p,
        (int32_t)B, (int32_t)H, (int32_t)S, (int32_t)D, sc);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_blackwell::run);

}  // namespace tvm_ffi_mha_blackwell