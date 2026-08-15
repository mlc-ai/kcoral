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

static const uint32_t BM = 64;
static const uint32_t BN = 64;
static const uint32_t BK = 16;
static const uint32_t BD = 128;
static const uint32_t NT = 256;
static const float NEG_INF_F = -1e10f;

__global__ void mha_kernel(
    const __nv_bfloat16* __restrict__ Q_g,
    const __nv_bfloat16* __restrict__ K_g,
    const __nv_bfloat16* __restrict__ V_g,
    __nv_bfloat16* __restrict__ O_g,
    float*            __restrict__ LSE_g,
    int32_t B, int32_t H, int32_t S, int32_t D,
    float scale_factor)
{
    extern __shared__ __align__(16) char smem_base[];
    
    __nv_bfloat16* sq = reinterpret_cast<__nv_bfloat16*>(smem_base);
    __nv_bfloat16* sk = sq + BM * BK;
    __nv_bfloat16* sv = sk + BN * BK;
    float*         sl = reinterpret_cast<float*>(sv + BD * BN);
    float*         so = sl + BM * BN;
    
    uint32_t tid  = threadIdx.x;
    uint32_t ql   = tid % BM;
    uint32_t bh   = blockIdx.y;
    uint32_t batch = bh / static_cast<uint32_t>(H);
    uint32_t head  = bh % static_cast<uint32_t>(H);
    uint32_t qbase = blockIdx.x * BM;
    
    uint64_t base_addr = (uint64_t)batch * (uint64_t)H * (uint64_t)S * (uint64_t)D
                       + (uint64_t)head  * (uint64_t)S * (uint64_t)D;
    
    uint32_t nds  = (D + BK - 1) / BK;
    uint32_t nkvs = (S + BN - 1) / BN;
    
    // Zero accumulator: BM * BD floats
    for (uint32_t i = tid; i < BM * BD; i += NT) so[i] = 0.0f;
    __syncthreads();
    
    float rmax = NEG_INF_F, rsum = 0.0f;
    
    for (uint32_t st = 0; st < nkvs; ++st) {
        uint32_t ks = st * BN;
        
        // Zero logits
        for (uint32_t i = tid; i < BM * BN; i += NT) sl[i] = 0.0f;
        __syncthreads();
        
        for (uint32_t ds = 0; ds < nds; ++ds) {
            uint32_t doff = ds * BK;
            
            // Load Q chunk cooperatively
            for (uint32_t i = tid; i < BM * BK; i += NT) {
                uint32_t r = i / BK, c = i % BK;
                uint32_t qr = qbase + r, dc = doff + c;
                if (qr < (uint32_t)S && dc < (uint32_t)D)
                    sq[i] = Q_g[base_addr + (uint64_t)qr * (uint64_t)D + dc];
                else
                    sq[i] = __float2bfloat16(0.0f);
            }
            
            // Load K chunk cooperatively
            for (uint32_t i = tid; i < BN * BK; i += NT) {
                uint32_t r = i / BK, c = i % BK;
                uint32_t kr = ks + r, dc = doff + c;
                if (kr < (uint32_t)S && dc < (uint32_t)D)
                    sk[i] = K_g[base_addr + (uint64_t)kr * (uint64_t)D + dc];
                else
                    sk[i] = __float2bfloat16(0.0f);
            }
            
            __syncthreads();
            
            // Accumulate QK^T dot products: each thread does BN/NT keys
            for (uint32_t kn = tid; kn < BN; kn += NT) {
                float dot = 0.0f;
                const uint32_t qi = ql * BK;
                const uint32_t ki = kn * BK;
                #pragma unroll
                for (uint32_t dk = 0; dk < BK; ++dk) {
                    dot += __bfloat162float(sq[qi + dk])
                         * __bfloat162float(sk[ki + dk]);
                }
                sl[ql * BN + kn] += dot;
            }
            __syncthreads();
        }
        
        // Apply scale and causal mask per-thread on its own row
        {
            uint32_t qg = qbase + ql;
            uint32_t li = ql * BN;
            for (uint32_t kn = 0; kn < BN; ++kn) {
                float v = sl[li + kn] * scale_factor;
                uint32_t kg = ks + kn;
                if (kg > qg || kg >= (uint32_t)S) v = NEG_INF_F;
                sl[li + kn] = v;
            }
        }
        __syncthreads();
        
        // Compute row max for this key block
        float cmx = NEG_INF_F;
        {
            uint32_t li = ql * BN;
            for (uint32_t k = 0; k < BN; ++k) {
                float v = sl[li + k];
                if (v > cmx) cmx = v;
            }
        }
        bool val = (cmx > NEG_INF_F);
        
        // Rescale accumulated output if we found a new max
        float al = 1.0f;
        if (val && cmx > rmax) {
            al = expf(rmax - cmx);
            uint32_t oi = ql * BD;
            for (uint32_t di = 0; di < BD; ++di) so[oi + di] *= al;
            rmax = cmx;
        }
        __syncthreads();
        
        // Load V [BN][BD], stored transposed as [BD][BN]
        for (uint32_t i = tid; i < BN * BD; i += NT) {
            uint32_t kr = i / BD, dc = i % BD;
            uint32_t kg = ks + kr;
            if (kg < (uint32_t)S)
                sv[dc * BN + kr] = V_g[base_addr + (uint64_t)kg * (uint64_t)D + dc];
            else
                sv[dc * BN + kr] = __float2bfloat16(0.0f);
        }
        __syncthreads();
        
        // P @ V accumulation with softmax probabilities
        float bsu = 0.0f;
        if (val) {
            uint32_t li = ql * BN;
            uint32_t oi = ql * BD;
            for (uint32_t di = 0; di < BD; ++di) {
                float ac = 0.0f;
                uint32_t vi = di * BN;
                for (uint32_t kn = 0; kn < BN; ++kn) {
                    float lg = sl[li + kn];
                    if (lg > NEG_INF_F) {
                        float p = expf(lg - cmx);
                        ac += p * __bfloat162float(sv[vi + kn]);
                        bsu += p;
                    }
                }
                so[oi + di] += ac * al;
            }
        }
        rsum += bsu * al;
    }
    
    // Epilogue: normalize and store output
    float nm = (rsum > 0.0f && rmax > NEG_INF_F) ? (1.0f / rsum) : 0.0f;
    uint32_t qg = qbase + ql;
    
    if (qg < (uint32_t)S) {
        uint32_t oi = ql * BD;
        uint64_t obase = base_addr + (uint64_t)qg * (uint64_t)D;
        for (uint32_t di = 0; di < BD; ++di) {
            O_g[obase + di] = __float2bfloat16(so[oi + di] * nm);
        }
        float lse_val = (rmax > NEG_INF_F)
                      ? (rsum > 0.0f ? (rmax + logf(rsum)) : NEG_INF_F)
                      : NEG_INF_F;
        uint64_t lx = (uint64_t)batch * (uint64_t)H * (uint64_t)S
                    + (uint64_t)head * (uint64_t)S + qg;
        LSE_g[lx] = lse_val;
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
        tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    const __nv_bfloat16* Qp = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* Kp = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* Vp = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* Op = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* Lp = static_cast<float*>(LSE.data_ptr());
    
    float sc = 1.0f / sqrtf(static_cast<float>(D));
    
    uint32_t gx = static_cast<uint32_t>((S + BM - 1) / BM);
    uint32_t gy = static_cast<uint32_t>(B * H);
    
    dim3 blk(NT, 1, 1);
    dim3 gr(gx, gy, 1);
    
    // Calculate exact shared memory requirement:
    // sq: BM*BK bf16  = 64*16*2  = 2048
    // sk: BN*BK bf16  = 64*16*2  = 2048
    // sv: BD*BN bf16  = 128*64*2 = 16384
    // sl: BM*BN fp32  = 64*64*4  = 16384
    // so: BM*BD fp32  = 64*128*4 = 32768
    // Total: 69632 bytes
    uint32_t smem_bytes = 69632;
    
    cudaStream_t str = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    // Verify launch config
    int max_sm_size;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_sm_size, cudaDevAttrMaxSharedMemoryPerBlock, Q.device().device_id));
    if (smem_bytes > static_cast<uint32_t>(max_sm_size)) {
        smem_bytes = static_cast<uint32_t>(max_sm_size);
    }
    
    mha_kernel<<<gr, blk, smem_bytes, str>>>(
        Qp, Kp, Vp, Op, Lp,
        static_cast<int32_t>(B),
        static_cast<int32_t>(H),
        static_cast<int32_t>(S),
        static_cast<int32_t>(D),
        sc);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(str));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_blackwell::run);

}  // namespace tvm_ffi_mha_blackwell