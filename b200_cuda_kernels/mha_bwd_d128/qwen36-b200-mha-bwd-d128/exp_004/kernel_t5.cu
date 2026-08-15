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

static constexpr int TM = 32;
static constexpr int TN = 32;
static constexpr int NT = 256;
static constexpr int D = 128;

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

    // All threads cooperate to load tiles from global -> shared mem
    int tid = threadIdx.x;
    int bh = blockIdx.x;
    
    if (bh >= B * H) return;
    
    int b = bh / H;
    int h = bh % H;
    
    uint64_t head_off = ((uint64_t)b * H + h) * (uint64_t)S * d;
    uint64_t head_off_lse = ((uint64_t)b * H + h) * (uint64_t)S;
    
    int num_q_tiles = (S + TM - 1) / TM;
    int num_k_tiles = (S + TN - 1) / TN;
    
    // Initialize dK and dV to zero
    for (int idx = tid; idx < S * d; idx += NT) {
        dK[head_off + idx] = __float2bfloat16(0.0f);
        dV[head_off + idx] = __float2bfloat16(0.0f);
    }
    __syncthreads();
    
    float L_vals[TM];
    float delta_arr[TM];
    
    for (int mq = 0; mq < num_q_tiles; mq++) {
        int q_start = mq * TM;
        
        // Allocate shared memory for query tile on each iteration
        // Use dynamic shared memory via extern
        extern __shared__ __nv_bfloat16 smem[];
        
        // Layout: [q][k][v][do][of_sm] then floats inline
        // We'll use separate flat arrays
        int base = 0;
        __nv_bfloat16* Q_sm = &smem[base];          base += TM*D;
        __nv_bfloat16* K_sm = &smem[base];          base += TN*D;
        __nv_bfloat16* V_sm = &smem[base];          base += TN*D;
        __nv_bfloat16* dO_sm = &smem[base];         base += TM*D;
        __nv_bfloat16* O_fwd_sm = &smem[base];      base += TM*D;
        // After base bytes of bf16 data, remaining space holds floats
        // Need: S_tile[TM*TN], dP_tile[TM*TN], dQ_acc[TM*D] in float32
        // Total bf16: 6*TM*D = 6*4096 = 24576 bf16 = 49152 bytes
        // Total float needed: 2*(TM*TN)*4 + (TM*D)*4 = 8192 + 16384 = 24576 bytes
        // Grand total: ~74KB which fits
        
        char* char_base = reinterpret_cast<char*>(smem);
        float* S_tile = reinterpret_cast<float*>(char_base + base * sizeof(__nv_bfloat16));
        float* dP_tile = S_tile + TM*TN;
        float* dQ_acc = dP_tile + TM*TN;
        
        // Load Q, dO, O_fwd tiles
        for (int i = tid; i < TM * d; i += NT) {
            int r = i / d;
            int c = i % d;
            int qr = q_start + r;
            if (qr < S) {
                uint64_t gidx = head_off + (uint64_t)qr * d + c;
                Q_sm[i] = Q[gidx];
                dO_sm[i] = dO[gidx];
                O_fwd_sm[i] = O_fwd[gidx];
            } else {
                Q_sm[i] = __float2bfloat16(0.0f);
                dO_sm[i] = __float2bfloat16(0.0f);
                O_fwd_sm[i] = __float2bfloat16(0.0f);
            }
        }
        
        // Compute delta[i] = dot(dO[i], O_fwd[i]) and load L
        for (int i = tid; i < TM; i += NT) {
            int qr = q_start + i;
            L_vals[i] = (qr < S) ? L[head_off_lse + qr] : 0.0f;
            float s = 0.0f;
            if (qr < S) {
                for (int dd = 0; dd < d; dd += 4) {
                    int bl = i*d + dd;
                    s += __bfloat162float(dO_sm[bl+0])*__bfloat162float(O_fwd_sm[bl+0]);
                    s += __bfloat162float(dO_sm[bl+1])*__bfloat162float(O_fwd_sm[bl+1]);
                    s += __bfloat162float(dO_sm[bl+2])*__bfloat162float(O_fwd_sm[bl+2]);
                    s += __bfloat162float(dO_sm[bl+3])*__bfloat162float(O_fwd_sm[bl+3]);
                }
            }
            delta_arr[i] = s;
        }
        
        // Clear dQ accumulator  
        for (int i = tid; i < TM*d; i += NT) dQ_acc[i] = 0.0f;
        __syncthreads();
        
        for (int mk = 0; mk < num_k_tiles; mk++) {
            int k_start = mk * TN;
            
            // Load K, V tiles
            for (int i = tid; i < TN*d; i += NT) {
                int r = i/d;
                int c = i%d;
                int kr = k_start + r;
                if (kr < S) {
                    uint64_t gidx = head_off + (uint64_t)kr*d + c;
                    K_sm[i] = K[gidx];
                    V_sm[i] = V[gidx];
                } else {
                    K_sm[i] = __float2bfloat16(0.0f);
                    V_sm[i] = __float2bfloat16(0.0f);
                }
            }
            __syncthreads();
            
            // PASS 1: ALL threads compute S[q,k] and dP[q,k]
            for (int e = 0; e < 4; e++) {
                int idx = tid*4 + e;
                int iq = idx/TN;
                int ik = idx%TN;
                
                float sv = 0.0f, dpv = 0.0f;
                int qr = q_start + iq;
                int kr = k_start + ik;
                
                if (qr < S && kr < S) {
                    int qb = iq*d;
                    int kb = ik*d;
                    int vb = ik*d;
                    int db = iq*d;
                    
                    for (int dd = 0; dd < d; dd += 4) {
                        sv += __bfloat162float(Q_sm[qb+dd])*__bfloat162float(K_sm[kb+dd]);
                        sv += __bfloat162float(Q_sm[qb+dd+1])*__bfloat162float(K_sm[kb+dd+1]);
                        sv += __bfloat162float(Q_sm[qb+dd+2])*__bfloat162float(K_sm[kb+dd+2]);
                        sv += __bfloat162float(Q_sm[qb+dd+3])*__bfloat162float(K_sm[kb+dd+3]);
                        
                        dpv += __bfloat162float(dO_sm[db+dd])*__bfloat162float(V_sm[vb+dd]);
                        dpv += __bfloat162float(dO_sm[db+dd+1])*__bfloat162float(V_sm[vb+dd+1]);
                        dpv += __bfloat162float(dO_sm[db+dd+2])*__bfloat162float(V_sm[vb+dd+2]);
                        dpv += __bfloat162float(dO_sm[db+dd+3])*__bfloat162float(V_sm[vb+dd+3]);
                    }
                }
                S_tile[idx] = sv * inv_sqrt_d;
                dP_tile[idx] = dpv;
            }
            __syncthreads();
            
            // PASS 2: Threads 0..127 compute dQ contribution
            if (tid < 128) {
                int iq = tid/8;          // row 0..31
                int lane = tid%8;        // lane 0..7
                int col_base = lane*16;  // cols 0,16,32,...112
                
                float l_val = L_vals[iq];
                float d_val = delta_arr[iq];
                float dq_reg[16];
                #pragma unroll
                for (int c = 0; c < 16; c++) dq_reg[c] = 0.0f;
                
                for (int ik = 0; ik < TN; ik++) {
                    if (k_start+ik >= S) continue;
                    float p_val = expf(S_tile[iq*TN+ik] - l_val);
                    float ds = p_val * (dP_tile[iq*TN+ik] - d_val);
                    int kb = ik*d;
                    #pragma unroll
                    for (int c = 0; c < 16; c++) {
                        dq_reg[c] += ds * __bfloat162float(K_sm[kb + col_base + c]);
                    }
                }
                
                #pragma unroll
                for (int c = 0; c < 16; c++) {
                    dQ_acc[iq*d + col_base + c] += dq_reg[c];
                }
            }
            
            // PASS 3: Threads 128..255 compute dK, dV accumulation
            if (tid >= 128) {
                int eff = tid - 128;     // 0..127
                int ik = eff/4;          // row 0..31
                int lane = eff%4;        // lane 0..3
                int col_base = lane*32;  // cols 0,32,64,96
                int kr = k_start + ik;
                
                if (kr < S) {
                    float dk_reg[32];
                    float dv_reg[32];
                    #pragma unroll
                    for (int c = 0; c < 32; c++) {
                        dk_reg[c] = 0.0f;
                        dv_reg[c] = 0.0f;
                    }
                    
                    for (int iq = 0; iq < TM; iq++) {
                        if (q_start+iq >= S) continue;
                        float p_val = expf(S_tile[iq*TN+ik] - L_vals[iq]);
                        float ds = p_val * (dP_tile[iq*TN+ik] - delta_arr[iq]);
                        int qb = iq*d;
                        int db = iq*d;
                        #pragma unroll
                        for (int c = 0; c < 32; c++) {
                            dk_reg[c] += ds * __bfloat162float(Q_sm[qb + col_base + c]);
                            dv_reg[c] += p_val * __bfloat162float(dO_sm[db + col_base + c]);
                        }
                    }
                    
                    uint64_t base_addr = head_off + (uint64_t)kr*d;
                    #pragma unroll
                    for (int c = 0; c < 32; c++) {
                        uint64_t addr = base_addr + col_base + c;
                        dK[addr] = __float2bfloat16(__bfloat162float(dK[addr]) + dk_reg[c]);
                        dV[addr] = __float2bfloat16(__bfloat162float(dV[addr]) + dv_reg[c]);
                    }
                }
            }
            __syncthreads();
        }
        
        // Write dQ after processing all key tiles
        for (int i = tid; i < TM*d; i += NT) {
            int r = i/d;
            int c = i%d;
            int qr = q_start + r;
            if (qr < S) {
                dQ[head_off + (uint64_t)qr*d + c] = __float2bfloat16(dQ_acc[r*d+c]);
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
    
    // Shared mem: 6*TM*D bf16 + 2*(TM*TN) float + TM*D float
    // = 6*32*128*2 + 2*32*32*4 + 32*128*4
    // = 49152 + 8192 + 16384 = 73728 bytes (~72KB)
    size_t smem_size = (size_t)(6*TM*D*sizeof(__nv_bfloat16) + 
                                 2*TM*TN*sizeof(float) + 
                                 TM*d_dim*sizeof(float));
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    auto kernel_fn = &mha_bwd_kernel;
    kernel_fn<<<grid, block, smem_size, stream>>>(
        Q_data, K_data, V_data, O_data, dO_data, L_data,
        dQ_data, dK_data, dV_data,
        static_cast<int>(B), static_cast<int>(H), static_cast<int>(S), static_cast<int>(d_dim),
        inv_sqrt_d);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128::run);

}  // namespace mha_bwd_d128