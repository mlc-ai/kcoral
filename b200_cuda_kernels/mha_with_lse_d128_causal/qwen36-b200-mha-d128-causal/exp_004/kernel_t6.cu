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

static const uint32_t BM = 64;   // query tile height
static const uint32_t BN = 64;   // key tile width
static const uint32_t BK = 16;   // D-chunk size
static const uint32_t BD = 128;  // Head dim
static const uint32_t NT = 256;
static const float NEG_INF_F = -1e10f;

extern __shared__ char smem_raw[];

__global__ void mha_kernel(
    const __nv_bfloat16* __restrict__ Q_g,
    const __nv_bfloat16* __restrict__ K_g,
    const __nv_bfloat16* __restrict__ V_g,
    __nv_bfloat16* __restrict__ O_g,
    float*            __restrict__ LSE_g,
    int32_t B, int32_t H, int32_t S, int32_t D,
    float scale_factor)
{
    uint32_t oq  = 0;
    uint32_t ok  = BM * BK * 2;
    uint32_t ov  = ok + BN * BK * 2;
    uint32_t ol  = ov + BD * BN * 2;
    uint32_t oo  = ol + BM * BN * 4;
    
    __nv_bfloat16* sq = reinterpret_cast<__nv_bfloat16*>(smem_raw + oq);
    __nv_bfloat16* sk = reinterpret_cast<__nv_bfloat16*>(smem_raw + ok);
    __nv_bfloat16* sv = reinterpret_cast<__nv_bfloat16*>(smem_raw + ov);
    float*         sl = reinterpret_cast<float*>(smem_raw + ol);
    float*         so = reinterpret_cast<float*>(smem_raw + oo);
    
    uint32_t tid     = threadIdx.x;
    uint32_t ql      = tid % BM;
    uint32_t bh      = blockIdx.y;
    uint32_t batch   = bh / static_cast<uint32_t>(H);
    uint32_t head    = bh % static_cast<uint32_t>(H);
    uint32_t qbase   = blockIdx.x * BM;
    
    uint64_t base = (uint64_t)batch * (uint64_t)H * (uint64_t)S * (uint64_t)D
                  + (uint64_t)head * (uint64_t)S * (uint64_t)D;
    
    uint32_t nds  = (D + BK - 1) / BK;
    uint32_t nkvs = (S + BN - 1) / BN;
    
    // Zero accumulator
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
            
            // Load Q
            for (uint32_t i = tid; i < BM * BK; i += NT) {
                uint32_t r = i / BK, c = i % BK;
                uint32_t qr = qbase + r, dc = doff + c;
                sq[i] = (qr < (uint32_t)S && dc < (uint32_t)D)
                    ? Q_g[base + (uint64_t)qr * (uint64_t)D + dc]
                    : __float2bfloat16(0.0f);
            }
            // Load K
            for (uint32_t i = tid; i < BN * BK; i += NT) {
                uint32_t r = i / BK, c = i % BK;
                uint32_t kr = ks + r, dc = doff + c;
                sk[i] = (kr < (uint32_t)S && dc < (uint32_t)D)
                    ? K_g[base + (uint64_t)kr * (uint64_t)D + dc]
                    : __float2bfloat16(0.0f);
            }
            __syncthreads();
            
            // Q@K^T accumulation
            for (uint32_t kn = tid; kn < BN; kn += NT) {
                float dot = 0.0f;
                uint32_t qi = ql * BK, ki = kn * BK;
                for (uint32_t dk = 0; dk < BK; ++dk) {
                    dot += __bfloat162float(sq[qi+dk]) * __bfloat162float(sk[ki+dk]);
                }
                sl[ql * BN + kn] += dot;
            }
            __syncthreads();
        }
        
        // Scale + causal mask
        {
            uint32_t qg = qbase + ql;
            uint32_t li = ql * BN;
            for (uint32_t kn = 0; kn < BN; ++kn) {
                float v = sl[li+kn] * scale_factor;
                uint32_t kg = ks + kn;
                if (kg > qg || kg >= (uint32_t)S) v = NEG_INF_F;
                sl[li+kn] = v;
            }
        }
        __syncthreads();
        
        // Row max
        float cmx = NEG_INF_F;
        { uint32_t li = ql * BN; for (uint32_t k = 0; k < BN; ++k) { float v = sl[li+k]; if (v > cmx) cmx = v; } }
        bool val = (cmx > NEG_INF_F);
        
        // Rescale accumulator
        float al = 1.0f;
        if (val && cmx > rmax) {
            al = expf(rmax - cmx);
            uint32_t oi = ql * BD;
            for (uint32_t di = 0; di < BD; ++di) so[oi+di] *= al;
            rmax = cmx;
        }
        __syncthreads();
        
        // Load V transposed [BD][BN]
        for (uint32_t i = tid; i < BN * BD; i += NT) {
            uint32_t kr = i / BD, dc = i % BD, kg = ks + kr;
            sv[dc*BN + kr] = (kg < (uint32_t)S)
                ? V_g[base + (uint64_t)kg*(uint64_t)D + dc]
                : __float2bfloat16(0.0f);
        }
        __syncthreads();
        
        // P@V
        float bsu = 0.0f;
        if (val) {
            uint32_t li = ql*BN, oi = ql*BD;
            for (uint32_t di = 0; di < BD; ++di) {
                float ac = 0.0f; uint32_t vi = di*BN;
                for (uint32_t kn = 0; kn < BN; ++kn) {
                    float lg = sl[li+kn];
                    if (lg > NEG_INF_F) {
                        float p = expf(lg - cmx);
                        ac += p * __bfloat162float(sv[vi+kn]);
                        bsu += p;
                    }
                }
                so[oi+di] += ac * al;
            }
        }
        rsum += bsu * al;
    }
    
    // Epilogue
    float nm = (rsum > 0.0f && rmax > NEG_INF_F) ? (1.0f/rsum) : 0.0f;
    uint32_t qg = qbase + ql;
    if (qg < (uint32_t)S) {
        uint32_t oi = ql*BD;
        uint64_t ob = base + (uint64_t)qg*(uint64_t)D;
        for (uint32_t di = 0; di < BD; ++di)
            O_g[ob+di] = __float2bfloat16(so[oi+di] * nm);
        float lse = (rmax > NEG_INF_F) ? (rsum > 0.0f ? (rmax + logf(rsum)) : NEG_INF_F) : NEG_INF_F;
        uint64_t lx = (uint64_t)batch*(uint64_t)H*(uint64_t)S + (uint64_t)head*(uint64_t)S + qg;
        LSE_g[lx] = lse;
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
        tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0), H = Q.size(1), S = Q.size(2), D = Q.size(3);
    
    const __nv_bfloat16* Qp = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* Kp = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* Vp = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* Op       = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* Lp                = static_cast<float*>(LSE.data_ptr());
    
    float sc = 1.0f / sqrtf(static_cast<float>(D));
    
    uint32_t gx = (uint32_t)((S + BM - 1) / BM);
    uint32_t gy = (uint32_t)(B * H);
    dim3 blk(NT, 1, 1);
    dim3 gr(gx, gy, 1);
    
    // Shared mem: 2048 + 2048 + 16384 + 16384 + 32768 = 69632
    uint32_t sm = 70000;
    
    cudaStream_t str = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    mha_kernel<<<gr, blk, sm, str>>>(Qp, Kp, Vp, Op, Lp,
        static_cast<int32_t>(B), static_cast<int32_t>(H),
        static_cast<int32_t>(S), static_cast<int32_t>(D), sc);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(str));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_blackwell::run);

}  // namespace tvm_ffi_mha_blackwell