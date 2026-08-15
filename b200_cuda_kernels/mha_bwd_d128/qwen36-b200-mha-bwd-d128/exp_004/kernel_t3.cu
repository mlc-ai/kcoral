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
static constexpr int NUM_THREADS = 256;

template<int D>
struct SharedMemLayout {
    alignas(8) __nv_bfloat16 Q_sm[TILE_M * D];
    alignas(8) __nv_bfloat16 K_sm[TILE_N * D];
    alignas(8) __nv_bfloat16 V_sm[TILE_N * D];
    alignas(8) __nv_bfloat16 dO_sm[TILE_M * D];
    alignas(8) __nv_bfloat16 O_fwd_sm[TILE_M * D];
    alignas(16) float S_tile[TILE_M * TILE_N];
    alignas(16) float dP_tile[TILE_M * TILE_N];
    alignas(16) float dQ_acc[TILE_M * D];
};

template<int D>
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
    
    typedef SharedMemLayout<D> SMem;
    extern __shared__ char smem_raw[];
    SMem& smem = *reinterpret_cast<SMem*>(smem_raw);
    
    int bh = blockIdx.x;
    if (bh >= B * H) return;
    
    int b = bh / H;
    int h = bh % H;
    
    uint64_t head_off = ((uint64_t)b * H + h) * (uint64_t)S * d;
    uint64_t head_off_lse = ((uint64_t)b * H + h) * (uint64_t)S;
    
    int tid = threadIdx.x;
    
    int num_q_tiles = (S + TILE_M - 1) / TILE_M;
    int num_k_tiles = (S + TILE_N - 1) / TILE_N;
    
    // Initialize dK and dV to zero for this (b,h)
    for (int idx = tid; idx < S * d; idx += NUM_THREADS) {
        dK[head_off + idx] = __float2bfloat16(0.0f);
        dV[head_off + idx] = __float2bfloat16(0.0f);
    }
    __syncthreads();
    
    // Register arrays
    float L_vals[TILE_M];
    float delta_arr[TILE_M];
    float dQ_chunk[16];
    float dK_chunk[32];
    float dV_chunk[32];
    
    for (int mq = 0; mq < num_q_tiles; mq++) {
        int q_start = mq * TILE_M;
        
        // Load Q, dO, O_fwd tiles into shared memory
        for (int i = tid; i < TILE_M * d; i += NUM_THREADS) {
            int r = i / d;
            int c = i % d;
            int qr = q_start + r;
            if (qr < S) {
                uint64_t src = (uint64_t)qr * d + c;
                smem.Q_sm[i] = Q[head_off + src];
                smem.dO_sm[i] = dO[head_off + src];
                smem.O_fwd_sm[i] = O_fwd[head_off + src];
            } else {
                smem.Q_sm[i] = __float2bfloat16(0.0f);
                smem.dO_sm[i] = __float2bfloat16(0.0f);
                smem.O_fwd_sm[i] = __float2bfloat16(0.0f);
            }
        }
        
        // Load L values and compute delta[i] = dot(dO[i,:], O[i,:])
        for (int i = tid; i < TILE_M; i += NUM_THREADS) {
            int qr = q_start + i;
            L_vals[i] = (qr < S) ? L[head_off_lse + qr] : 0.0f;
            float s = 0.0f;
            if (qr < S) {
                for (int dd = 0; dd < d; dd += 2) {
                    s += __bfloat162float(smem.dO_sm[i * d + dd]) * __bfloat162float(smem.O_fwd_sm[i * d + dd]);
                    s += __bfloat162float(smem.dO_sm[i * d + dd+1]) * __bfloat162float(smem.O_fwd_sm[i * d + dd+1]);
                }
            }
            delta_arr[i] = s;
        }
        
        // Clear dQ accumulator for this query tile
        for (int idx = tid; idx < TILE_M * d; idx += NUM_THREADS) {
            smem.dQ_acc[idx] = 0.0f;
        }
        __syncthreads();
        
        for (int mk = 0; mk < num_k_tiles; mk++) {
            int k_start = mk * TILE_N;
            
            // Load K, V tiles
            for (int idx = tid; idx < TILE_N * d; idx += NUM_THREADS) {
                int r = idx / d;
                int c = idx % d;
                int kr = k_start + r;
                if (kr < S) {
                    uint64_t src = (uint64_t)kr * d + c;
                    smem.K_sm[idx] = K[head_off + src];
                    smem.V_sm[idx] = V[head_off + src];
                } else {
                    smem.K_sm[idx] = __float2bfloat16(0.0f);
                    smem.V_sm[idx] = __float2bfloat16(0.0f);
                }
            }
            __syncthreads();
            
            // PASS 1: All 256 threads compute S_tile and dP_tile
            int ept = (TILE_M * TILE_N) / NUM_THREADS;  // 4
            for (int e = 0; e < ept; e++) {
                int idx = tid * ept + e;
                int iq = idx / TILE_N;
                int ik = idx % TILE_N;
                
                float s_val = 0.0f;
                float dp_val = 0.0f;
                
                int qr = q_start + iq;
                int kr = k_start + ik;
                
                if (qr < S && kr < S) {
                    for (int dd = 0; dd < d; dd += 4) {
                        s_val += __bfloat162float(smem.Q_sm[iq * d + dd])     * __bfloat162float(smem.K_sm[ik * d + dd]);
                        s_val += __bfloat162float(smem.Q_sm[iq * d + dd+1])   * __bfloat162float(smem.K_sm[ik * d + dd+1]);
                        s_val += __bfloat162float(smem.Q_sm[iq * d + dd+2])   * __bfloat162float(smem.K_sm[ik * d + dd+2]);
                        s_val += __bfloat162float(smem.Q_sm[iq * d + dd+3])   * __bfloat162float(smem.K_sm[ik * d + dd+3]);
                        
                        dp_val += __bfloat162float(smem.dO_sm[iq * d + dd])     * __bfloat162float(smem.V_sm[ik * d + dd]);
                        dp_val += __bfloat162float(smem.dO_sm[iq * d + dd+1])   * __bfloat162float(smem.V_sm[ik * d + dd+1]);
                        dp_val += __bfloat162float(smem.dO_sm[iq * d + dd+2])   * __bfloat162float(smem.V_sm[ik * d + dd+2]);
                        dp_val += __bfloat162float(smem.dO_sm[iq * d + dd+3])   * __bfloat162float(smem.V_sm[ik * d + dd+3]);
                    }
                }
                
                smem.S_tile[idx] = s_val * inv_sqrt_d;
                smem.dP_tile[idx] = dp_val;
            }
            __syncthreads();
            
            // PASS 2: Threads 0..127 accumulate dQ_acc
            if (tid < 128) {
                int iq = tid / 8;           // 0..31
                int lane = tid % 8;         // 0..7
                int col_base = lane * 16;   // 0,16,32,...,112
                
                float l_val = L_vals[iq];
                float d_val = delta_arr[iq];
                
                #pragma unroll
                for (int c = 0; c < 16; c++) dQ_chunk[c] = 0.0f;
                
                for (int ik = 0; ik < TILE_N; ik++) {
                    int kr = k_start + ik;
                    if (kr >= S) continue;
                    
                    float s_val = smem.S_tile[iq * TILE_N + ik];
                    float dp_val = smem.dP_tile[iq * TILE_N + ik];
                    float p_val = expf(s_val - l_val);
                    float ds_val = p_val * (dp_val - d_val);
                    
                    #pragma unroll
                    for (int c = 0; c < 16; c++) {
                        dQ_chunk[c] += ds_val * __bfloat162float(smem.K_sm[ik * d + col_base + c]);
                    }
                }
                
                #pragma unroll
                for (int c = 0; c < 16; c++) {
                    smem.dQ_acc[iq * d + col_base + c] += dQ_chunk[c];
                }
            }
            
            // PASS 3: Threads 128..255 accumulate dK, dV to global memory
            if (tid >= 128) {
                int eff = tid - 128;          // 0..127
                int ik = eff / 4;             // 0..31
                int lane = eff % 4;           // 0..3
                int col_base = lane * 32;     // 0,32,64,96
                int kr = k_start + ik;
                
                if (kr < S) {
                    #pragma unroll
                    for (int c = 0; c < 32; c++) {
                        dK_chunk[c] = 0.0f;
                        dV_chunk[c] = 0.0f;
                    }
                    
                    for (int iq = 0; iq < TILE_M; iq++) {
                        int qr = q_start + iq;
                        if (qr >= S) continue;
                        
                        float s_val = smem.S_tile[iq * TILE_N + ik];
                        float dp_val = smem.dP_tile[iq * TILE_N + ik];
                        float p_val = expf(s_val - L_vals[iq]);
                        float ds_val = p_val * (dp_val - delta_arr[iq]);
                        
                        #pragma unroll
                        for (int c = 0; c < 32; c++) {
                            dK_chunk[c] += ds_val * __bfloat162float(smem.Q_sm[iq * d + col_base + c]);
                            dV_chunk[c] += p_val * __bfloat162float(smem.dO_sm[iq * d + col_base + c]);
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
        
        // Write dQ to global memory after all key tiles for this query tile
        for (int idx = tid; idx < TILE_M * d; idx += NUM_THREADS) {
            int r = idx / d;
            int c = idx % d;
            int qr = q_start + r;
            if (qr < S) {
                dQ[head_off + (uint64_t)qr * d + c] = __float2bfloat16(smem.dQ_acc[r * d + c]);
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
    dim3 block(NUM_THREADS);
    
    size_t smem_size = sizeof(SharedMemLayout<128>);
    printf("Shared memory size: %zu bytes (%.1f KB)\n", smem_size, smem_size / 1024.0);
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    auto kernel_fn = &mha_bwd_kernel<128>;
    kernel_fn<<<grid, block, static_cast<size_t>(smem_size), stream>>>(
        Q_data, K_data, V_data, O_data, dO_data, L_data,
        dQ_data, dK_data, dV_data,
        static_cast<int>(B), static_cast<int>(H), static_cast<int>(S), static_cast<int>(d_dim),
        inv_sqrt_d);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128::run);

}  // namespace mha_bwd_d128