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

static constexpr int TILE_M = 32;
static constexpr int TILE_N = 32;
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
    
    extern __shared__ __nv_bfloat16 smem[];
    
    // Flat shared memory layout (all in bf16 units)
    // Q_sm:       [0, TM*D)          = 0..4095
    // K_sm:       [TM*D, +TN*D)      = 4096..8191
    // V_sm:       [+TN*D, +TN*D)     = 8192..12287
    // dO_sm:      [+TN*D, +TM*D)     = 12288..16383
    // O_fwd_sm:   [+TM*D, +TM*D)     = 16384..20479
    // Total bf16 so far: 20480
    
    int qsm_off = 0;
    int ksm_off = TILE_M * D;
    int vsm_off = ksm_off + TILE_N * D;
    int dosm_off = vsm_off + TILE_N * D;
    int ofsm_off = dosm_off + TILE_M * D;
    // Float buffers start after (ofsm_off + TM*D) bf16 = 20480 bf16 = 40960 bytes
    // We store floats in same char buffer but with separate pointer
    
    char* smem_char = reinterpret_cast<char*>(smem);
    float* S_tile = reinterpret_cast<float*>(smem_char + ofsm_off * sizeof(__nv_bfloat16) + TILE_M * D * sizeof(__nv_bfloat16));
    float* dP_tile = S_tile + TILE_M * TILE_N;
    float* dQ_acc = dP_tile + TILE_M * TILE_N;
    
    int bh = blockIdx.x;
    if (bh >= B * H) return;
    
    int b = bh / H;
    int h = bh % H;
    
    uint64_t head_off = ((uint64_t)b * H + h) * (uint64_t)S * d;
    uint64_t head_off_lse = ((uint64_t)b * H + h) * (uint64_t)S;
    
    int tid = threadIdx.x;
    
    int num_q_tiles = (S + TILE_M - 1) / TILE_M;
    int num_k_tiles = (S + TILE_N - 1) / TILE_N;
    
    // Initialize dK and dV to zero
    for (int idx = tid; idx < S * d; idx += NT) {
        dK[head_off + idx] = __float2bfloat16(0.0f);
        dV[head_off + idx] = __float2bfloat16(0.0f);
    }
    __syncthreads();
    
    float L_vals[TILE_M];
    float delta_arr[TILE_M];
    float dQ_chunk[16];
    float dK_chunk[32];
    float dV_chunk[32];
    
    for (int mq = 0; mq < num_q_tiles; mq++) {
        int q_start = mq * TILE_M;
        
        // Load Q, dO, O_fwd
        for (int i = tid; i < TILE_M * d; i += NT) {
            int r = i / d;
            int c = i % d;
            int qr = q_start + r;
            uint64_t gidx = (qr < S) ? head_off + (uint64_t)qr * d + c : 0;
            
            smem[qsm_off + i] = (qr < S) ? Q[gidx] : __float2bfloat16(0.0f);
            smem[dosm_off + i] = (qr < S) ? dO[gidx] : __float2bfloat16(0.0f);
            smem[ofsm_off + i] = (qr < S) ? O_fwd[gidx] : __float2bfloat16(0.0f);
        }
        
        // Compute delta[i] = dot(dO[i], O_fwd[i]) and load L
        for (int i = tid; i < TILE_M; i += NT) {
            int qr = q_start + i;
            L_vals[i] = (qr < S) ? L[head_off_lse + qr] : 0.0f;
            float s = 0.0f;
            if (qr < S) {
                for (int dd = 0; dd < d; dd += 4) {
                    int base = i * d + dd;
                    s += __bfloat162float(smem[dosm_off + base + 0]) * __bfloat162float(smem[ofsm_off + base + 0]);
                    s += __bfloat162float(smem[dosm_off + base + 1]) * __bfloat162float(smem[ofsm_off + base + 1]);
                    s += __bfloat162float(smem[dosm_off + base + 2]) * __bfloat162float(smem[ofsm_off + base + 2]);
                    s += __bfloat162float(smem[dosm_off + base + 3]) * __bfloat162float(smem[ofsm_off + base + 3]);
                }
            }
            delta_arr[i] = s;
        }
        
        // Clear dQ accumulator
        for (int i = tid; i < TILE_M * d; i += NT) {
            dQ_acc[i] = 0.0f;
        }
        __syncthreads();
        
        for (int mk = 0; mk < num_k_tiles; mk++) {
            int k_start = mk * TILE_N;
            
            // Load K, V
            for (int i = tid; i < TILE_N * d; i += NT) {
                int r = i / d;
                int c = i % d;
                int kr = k_start + r;
                uint64_t gidx = (kr < S) ? head_off + (uint64_t)kr * d + c : 0;
                
                smem[ksm_off + i] = (kr < S) ? K[gidx] : __float2bfloat16(0.0f);
                smem[vsm_off + i] = (kr < S) ? V[gidx] : __float2bfloat16(0.0f);
            }
            __syncthreads();
            
            // PASS 1: compute S_tile[q][k] and dP_tile[q][k]
            for (int e = 0; e < 4; e++) {
                int idx = tid * 4 + e;
                int iq = idx / TILE_N;
                int ik = idx % TILE_N;
                
                float sv = 0.0f, dpv = 0.0f;
                int qr = q_start + iq;
                int kr = k_start + ik;
                
                if (qr < S && kr < S) {
                    int qb = qsm_off + iq * d;
                    int kb = ksm_off + ik * d;
                    int vb = vsm_off + ik * d;
                    int db = dosm_off + iq * d;
                    
                    for (int dd = 0; dd < d; dd += 4) {
                        sv += __bfloat162float(smem[qb+dd+0])*__bfloat162float(smem[kb+dd+0]);
                        sv += __bfloat162float(smem[qb+dd+1])*__bfloat162float(smem[kb+dd+1]);
                        sv += __bfloat162float(smem[qb+dd+2])*__bfloat162float(smem[kb+dd+2]);
                        sv += __bfloat162float(smem[qb+dd+3])*__bfloat162float(smem[kb+dd+3]);
                        
                        dpv += __bfloat162float(smem[db+dd+0])*__bfloat162float(smem[vb+dd+0]);
                        dpv += __bfloat162float(smem[db+dd+1])*__bfloat162float(smem[vb+dd+1]);
                        dpv += __bfloat162float(smem[db+dd+2])*__bfloat162float(smem[vb+dd+2]);
                        dpv += __bfloat162float(smem[db+dd+3])*__bfloat162float(smem[vb+dd+3]);
                    }
                }
                S_tile[idx] = sv * inv_sqrt_d;
                dP_tile[idx] = dpv;
            }
            __syncthreads();
            
            // PASS 2: dQ (threads 0..127)
            if (tid < 128) {
                int iq = tid / 8;
                int lane = tid % 8;
                int col_base = lane * 16;
                float l_val = L_vals[iq];
                float d_val = delta_arr[iq];
                
                #pragma unroll
                for (int c = 0; c < 16; c++) dQ_chunk[c] = 0.0f;
                
                for (int ik = 0; ik < TILE_N; ik++) {
                    if (k_start + ik >= S) continue;
                    float p_val = expf(S_tile[iq*TILE_N + ik] - l_val);
                    float ds_val = p_val * (dP_tile[iq*TILE_N + ik] - d_val);
                    int kb = ksm_off + ik * d;
                    #pragma unroll
                    for (int c = 0; c < 16; c++) {
                        dQ_chunk[c] += ds_val * __bfloat162float(smem[kb + col_base + c]);
                    }
                }
                
                #pragma unroll
                for (int c = 0; c < 16; c++) {
                    dQ_acc[iq * d + col_base + c] += dQ_chunk[c];
                }
            }
            
            // PASS 3: dK/dV (threads 128..255)
            if (tid >= 128) {
                int eff = tid - 128;
                int ik = eff / 4;
                int lane = eff % 4;
                int col_base = lane * 32;
                int kr = k_start + ik;
                
                if (kr < S) {
                    #pragma unroll
                    for (int c = 0; c < 32; c++) {
                        dK_chunk[c] = 0.0f;
                        dV_chunk[c] = 0.0f;
                    }
                    
                    for (int iq = 0; iq < TILE_M; iq++) {
                        if (q_start + iq >= S) continue;
                        float p_val = expf(S_tile[iq*TILE_N + ik] - L_vals[iq]);
                        float ds_val = p_val * (dP_tile[iq*TILE_N + ik] - delta_arr[iq]);
                        int qb = qsm_off + iq * d;
                        int db = dosm_off + iq * d;
                        #pragma unroll
                        for (int c = 0; c < 32; c++) {
                            dK_chunk[c] += ds_val * __bfloat162float(smem[qb + col_base + c]);
                            dV_chunk[c] += p_val * __bfloat162float(smem[db + col_base + c]);
                        }
                    }
                    
                    uint64_t base_addr = head_off + (uint64_t)kr * d;
                    #pragma unroll
                    for (int c = 0; c < 32; c++) {
                        uint64_t addr = base_addr + col_base + c;
                        dK[addr] = __float2bfloat16(__bfloat162float(dK[addr]) + dK_chunk[c]);
                        dV[addr] = __float2bfloat16(__bfloat162float(dV[addr]) + dV_chunk[c]);
                    }
                }
            }
            __syncthreads();
        }
        
        // Write dQ
        for (int i = tid; i < TILE_M * d; i += NT) {
            int r = i / d;
            int c = i % d;
            int qr = q_start + r;
            if (qr < S) {
                dQ[head_off + (uint64_t)qr * d + c] = __float2bfloat16(dQ_acc[r * d + c]);
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
    
    // Shared mem: 6 bf16 buffers of 32x128 = 6*8192 = 49152 bf16 bytes
    // Plus 3 float buffers: 2x(32x32x4) + 1x(32x128x4) = 8192+16384 = 24576 bytes
    // Total bf16 = 49152*2 = 98304 + float portion offset... 
    // Simplified: compute exact size
    size_t bf16_bytes = (TILE_M*D + TILE_N*D + TILE_N*D + TILE_M*D + TILE_M*D + TILE_M*D) * sizeof(__nv_bfloat16);
    size_t smem_size = bf16_bytes + TILE_M*TILE_N*sizeof(float) + TILE_M*TILE_N*sizeof(float) + TILE_M*d_dim*sizeof(float);
    
    dim3 grid(total_heads);
    dim3 block(NT);
    
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