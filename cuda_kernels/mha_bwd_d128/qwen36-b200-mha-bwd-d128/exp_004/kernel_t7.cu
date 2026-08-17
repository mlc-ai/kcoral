#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

#define CUDA_CHECK(call) do {                                          \
    cudaError_t _e = (call);                                           \
    if (_e != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                   \
                cudaGetErrorString(_e), __FILE__, __LINE__);           \
        exit(1);                                                       \
    }                                                                  \
} while(0)

namespace mha_bwd_d128 {

static constexpr int NT = 256;
static constexpr int TM = 32;
static constexpr int TN = 32;
static constexpr int HD = 128;  // head dimension

__global__ void mha_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O_fwd,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int B, int H, int S, int d,
    float inv_sqrt_d) {
    
    int bh = blockIdx.x;
    if (bh >= B * H) return;
    
    int b = bh / H;
    int h = bh % H;
    
    uint64_t off_bh = ((uint64_t)b * H + h) * (uint64_t)S * d;
    uint64_t off_lse = ((uint64_t)b * H + h) * (uint64_t)S;
    
    int tid = threadIdx.x;
    
    // Initialize dK and dV to zero
    for (int idx = tid; idx < S * d; idx += NT) {
        dK[off_bh + idx] = __float2bfloat16(0.0f);
        dV[off_bh + idx] = __float2bfloat16(0.0f);
    }
    __syncthreads();
    
    int nqtiles = (S + TM - 1) / TM;
    int nktil es = (S + TN - 1) / TN;
    
    for (int mq = 0; mq < nqtiles; mq++) {
        int q_start = mq * TM;
        
        // Shared memory allocations via extern
        extern __shared__ __nv_bfloat16 smem[];
        __nv_bfloat16* Qsm = smem;                    // [TM*HD]
        __nv_bfloat16* dOsm = Qsm + TM*HD;            // [TM*HD]  
        __nv_bfloat16* Ofsm = dOsm + TM*HD;           // [TM*HD]
        __nv_bfloat16* Ksm = Ofsm + TM*HD;            // [TN*HD]
        __nv_bfloat16* Vsm = Ksm + TN*HD;             // [TN*HD]
        char* cbase = reinterpret_cast<char*>(Vsm + TN*HD);
        float* Stile = reinterpret_cast<float*>(cbase);          // [TM*TN]
        float* dPtile = Stile + TM*TN;               // [TM*TN]  
        float* dQacc = dPtile + TM*TN;               // [TM*HD]
        
        // Load Q, dO, O_fwd tiles
        for (int i = tid; i < TM*d; i += NT) {
            int r = i/d;
            int c = i%d;
            int qr = q_start + r;
            if (qr < S) {
                uint64_t gi = off_bh + (uint64_t)qr*d + c;
                Qsm[i] = Q[gi];
                dOsm[i] = dO[gi];
                Ofsm[i] = O_fwd[gi];
            } else {
                Qsm[i] = __float2bfloat16(0.0f);
                dOsm[i] = __float2bfloat16(0.0f);
                Ofsm[i] = __float2bfloat16(0.0f);
            }
        }
        
        // Compute delta[q] = dot(dO[q], O[q])
        float Lvals[TM];
        float delts[TM];
        for (int i = tid; i < TM; i += NT) {
            int qr = q_start + i;
            Lvals[i] = (qr < S) ? L[off_lse + qr] : 0.0f;
            float s = 0.0f;
            if (qr < S) {
                for (int dd = 0; dd < d; dd++) {
                    s += __bfloat162float(dOsm[i*d+dd]) * __bfloat162float(Ofsm[i*d+dd]);
                }
            }
            delts[i] = s;
        }
        
        // Clear dQ accumulator
        for (int i = tid; i < TM*d; i += NT) dQacc[i] = 0.0f;
        __syncthreads();
        
        for (int mk = 0; mk < nktil es; mk++) {
            int k_start = mk * TN;
            
            // Load K, V tiles
            for (int i = tid; i < TN*d; i += NT) {
                int r = i/d;
                int c = i%d;
                int kr = k_start + r;
                if (kr < S) {
                    uint64_t gi = off_bh + (uint64_t)kr*d + c;
                    Ksm[i] = K[gi];
                    Vsm[i] = V[gi];
                } else {
                    Ksm[i] = __float2bfloat16(0.0f);
                    Vsm[i] = __float2bfloat16(0.0f);
                }
            }
            __syncthreads();
            
            // Pass 1: All 256 threads compute S[q,k] and dP[q,k]
            for (int e = 0; e < 4; e++) {
                int idx = tid*4 + e;
                int iq = idx/TN;
                int ik = idx%TN;
                
                float sv = 0.0f, dpv = 0.0f;
                int qr = q_start + iq;
                int kr = k_start + ik;
                
                if (qr < S && kr < S) {
                    int qb = iq*d, kb = ik*d, vb = ik*d, db = iq*d;
                    for (int dd = 0; dd < d; dd += 4) {
                        sv += __bfloat162float(Qsm[qb+dd])*__bfloat162float(Ksm[kb+dd]);
                        sv += __bfloat162float(Qsm[qb+dd+1])*__bfloat162float(Ksm[kb+dd+1]);
                        sv += __bfloat162float(Qsm[qb+dd+2])*__bfloat162float(Ksm[kb+dd+2]);
                        sv += __bfloat162float(Qsm[qb+dd+3])*__bfloat162float(Ksm[kb+dd+3]);
                        
                        dpv += __bfloat162float(dOsm[db+dd])*__bfloat162float(Vsm[vb+dd]);
                        dpv += __bfloat162float(dOsm[db+dd+1])*__bfloat162float(Vsm[vb+dd+1]);
                        dpv += __bfloat162float(dOsm[db+dd+2])*__bfloat162float(Vsm[vb+dd+2]);
                        dpv += __bfloat162float(dOsm[db+dd+3])*__bfloat162float(Vsm[vb+dd+3]);
                    }
                }
                Stile[idx] = sv * inv_sqrt_d;
                dPtile[idx] = dpv;
            }
            __syncthreads();
            
            // Pass 2: Threads 0..127 accumulate dQ
            if (tid < 128) {
                int iq = tid/8;
                int lane = tid%8;
                int cb = lane*16;
                float lv = Lvals[iq], dv = delts[iq];
                float dq_reg[16] = {0};
                
                for (int ik = 0; ik < TN; ik++) {
                    if (k_start+ik >= S) continue;
                    float p = expf(Stile[iq*TN+ik] - lv);
                    float ds = p*(dPtile[iq*TN+ik] - dv);
                    int kb = ik*d;
                    #pragma unroll
                    for (int c = 0; c < 16; c++) {
                        dq_reg[c] += ds * __bfloat162float(Ksm[kb+cb+c]);
                    }
                }
                
                #pragma unroll
                for (int c = 0; c < 16; c++) {
                    dQacc[iq*d+cb+c] += dq_reg[c];
                }
            }
            
            // Pass 3: Threads 128..255 update dK, dV
            if (tid >= 128) {
                int eff = tid - 128;
                int ik = eff/4;
                int lane = eff%4;
                int cb = lane*32;
                int kr = k_start + ik;
                
                if (kr < S) {
                    float dk_reg[32] = {0}, dv_reg[32] = {0};
                    
                    for (int iq = 0; iq < TM; iq++) {
                        if (q_start+iq >= S) continue;
                        float p = expf(Stile[iq*TN+ik] - Lvals[iq]);
                        float ds = p*(dPtile[iq*TN+ik] - delts[iq]);
                        int qb = iq*d, db = iq*d;
                        #pragma unroll
                        for (int c = 0; c < 32; c++) {
                            dk_reg[c] += ds * __bfloat162float(Qsm[qb+cb+c]);
                            dv_reg[c] += p * __bfloat162float(dOsm[db+cb+c]);
                        }
                    }
                    
                    uint64_t ba = off_bh + (uint64_t)kr*d;
                    #pragma unroll
                    for (int c = 0; c < 32; c++) {
                        uint64_t addr = ba + cb + c;
                        dK[addr] = __float2bfloat16(__bfloat162float(dK[addr]) + dk_reg[c]);
                        dV[addr] = __float2bfloat16(__bfloat162float(dV[addr]) + dv_reg[c]);
                    }
                }
            }
            __syncthreads();
        }
        
        // Write dQ
        for (int i = tid; i < TM*d; i += NT) {
            int r = i/d;
            int c = i%d;
            int qr = q_start + r;
            if (qr < S) {
                dQ[off_bh + (uint64_t)qr*d + c] = __float2bfloat16(dQacc[r*d+c]);
            }
        }
        __syncthreads();
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d_dim = Q.size(3);
    
    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_data = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_data = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_data = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_data = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_data = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_data = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d_dim));
    int total_heads = static_cast<int>(B * H);
    
    dim3 grid(total_heads);
    dim3 block(NT);
    
    // Shared mem: 6 bf16 buffers (5*TM*HD + TN*HD = 6*32*128=24576 bf16=49152 bytes)
    // + 2 float32 buffers (2*32*32*4=8192) + 1 float32 buffer (32*128*4=16384)
    // Total: 49152 + 8192 + 16384 = 73728 bytes (~72KB)
    size_t smem_sz = (size_t)(6*TM*HD*sizeof(__nv_bfloat16) + 2*TM*TN*sizeof(float) + TM*HD*sizeof(float));
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    auto kf = &mha_bwd_kernel;
    kf<<<grid, block, smem_sz, stream>>>(
        Q_data, K_data, V_data, O_data, dO_data, L_data,
        dQ_data, dK_data, dV_data,
        static_cast<int>(B), static_cast<int>(H), static_cast<int>(S), static_cast<int>(d_dim),
        inv_sqrt_d);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128::run);

}  // namespace mha_bwd_d128