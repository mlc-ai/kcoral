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
static constexpr int EPT = (TILE_M * TILE_N) / NUM_THREADS;  // 4 elements per thread in Pass 1

// Pass 2: 128 threads for dQ accumulation (8 per query row)
static constexpr int TPASS2 = 128;
static constexpr int TPR2 = TPASS2 / TILE_M;       // 8 threads per row
static constexpr int COLS2 = 128 / TPR2;           // 16 cols per thread (d=128)

// Pass 3: 128 threads for dK/dV accumulation (4 per key row)
static constexpr int TPASS3 = 128;
static constexpr int TPR3 = TPASS3 / TILE_N;       // 4 threads per key row
static constexpr int COLS3 = 128 / TPR3;           // 32 cols per thread

template<int TILE_M, int TILE_N, int NUM_THREADS, int EPT>
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
    
    uint64_t head_off = ((uint64_t)b * H + h) * S * d;
    uint64_t head_off_lse = ((uint64_t)b * H + h) * S;
    
    const __nv_bfloat16* Q_h = Q + head_off;
    const __nv_bfloat16* K_h = K + head_off;
    const __nv_bfloat16* V_h = V + head_off;
    const __nv_bfloat16* O_fwd_h = O_fwd + head_off;
    const __nv_bfloat16* dO_h = dO + head_off;
    const float* L_h = L + head_off_lse;
    __nv_bfloat16* dQ_h = dQ + head_off;
    __nv_bfloat16* dK_h = dK + head_off;
    __nv_bfloat16* dV_h = dV + head_off;
    
    int tid = threadIdx.x;
    
    // Shared memory
    extern __shared__ char smem[];
    
    uint64_t off = 0;
    __nv_bfloat16* Q_sm = reinterpret_cast<__nv_bfloat16*>(smem + off);
    off += TILE_M * d * sizeof(__nv_bfloat16);
    __nv_bfloat16* K_sm = reinterpret_cast<__nv_bfloat16*>(smem + off);
    off += TILE_N * d * sizeof(__nv_bfloat16);
    __nv_bfloat16* V_sm = reinterpret_cast<__nv_bfloat16*>(smem + off);
    off += TILE_N * d * sizeof(__nv_bfloat16);
    __nv_bfloat16* dO_sm = reinterpret_cast<__nv_bfloat16*>(smem + off);
    off += TILE_M * d * sizeof(__nv_bfloat16);
    __nv_bfloat16* O_fwd_sm = reinterpret_cast<__nv_bfloat16*>(smem + off);
    off += TILE_M * d * sizeof(__nv_bfloat16);
    float* S_tile = reinterpret_cast<float*>(smem + off);
    off += TILE_M * TILE_N * sizeof(float);
    float* dP_tile = reinterpret_cast<float*>(smem + off);
    off += TILE_M * TILE_N * sizeof(float);
    float* dQ_acc = reinterpret_cast<float*>(smem + off);
    off += TILE_M * d * sizeof(float);
    // Total ~64KB
    
    __syncthreads();
    
    int num_q_tiles = (S + TILE_M - 1) / TILE_M;
    int num_k_tiles = (S + TILE_N - 1) / TILE_N;
    
    // Initialize dK and dV to zero for this (b,h)
    for (int idx = tid; idx < S * d; idx += NUM_THREADS) {
        dK_h[idx] = __float2bfloat16(0.0f);
        dV_h[idx] = __float2bfloat16(0.0f);
    }
    __syncthreads();
    
    // Register arrays reused across all query/key tile iterations
    float L_vals[TILE_M];
    float delta_arr[TILE_M];
    float dQ_chunk[COLS2];
    float dK_chunk[COLS3];
    float dV_chunk[COLS3];
    
    for (int mq = 0; mq < num_q_tiles; mq++) {
        int q_start = mq * TILE_M;
        
        // Load Q, dO, O_fwd tiles into shared memory
        for (int idx = tid; idx < TILE_M * d; idx += NUM_THREADS) {
            int r = idx / d;
            int c = idx % d;
            int qr = q_start + r;
            if (qr < S) {
                Q_sm[idx] = Q_h[qr * d + c];
                dO_sm[idx] = dO_h[qr * d + c];
                O_fwd_sm[idx] = O_fwd_h[qr * d + c];
            } else {
                Q_sm[idx] = __float2bfloat16(0.0f);
                dO_sm[idx] = __float2bfloat16(0.0f);
                O_fwd_sm[idx] = __float2bfloat16(0.0f);
            }
        }
        
        // Load L values and compute delta[i] = dot(dO[i,:], O[i,:])
        for (int i = tid; i < TILE_M; i += NUM_THREADS) {
            int qr = q_start + i;
            L_vals[i] = (qr < S) ? L_h[qr] : 0.0f;
            float s = 0.0f;
            if (qr < S) {
                for (int dd = 0; dd < d; dd++) {
                    s += __bfloat162float(dO_sm[i * d + dd]) * __bfloat162float(O_fwd_sm[i * d + dd]);
                }
            }
            delta_arr[i] = s;
        }
        
        // Clear dQ accumulator for this query tile
        for (int idx = tid; idx < TILE_M * d; idx += NUM_THREADS) {
            dQ_acc[idx] = 0.0f;
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
                    K_sm[idx] = K_h[kr * d + c];
                    V_sm[idx] = V_h[kr * d + c];
                } else {
                    K_sm[idx] = __float2bfloat16(0.0f);
                    V_sm[idx] = __float2bfloat16(0.0f);
                }
            }
            __syncthreads();
            
            // PASS 1: All 256 threads compute S_tile[q,k] and dP_tile[q,k]
            for (int e = 0; e < EPT; e++) {
                int idx = tid * EPT + e;
                if (idx >= TILE_M * TILE_N) break;
                
                int iq = idx / TILE_N;
                int ik = idx % TILE_N;
                int qr = q_start + iq;
                int kr = k_start + ik;
                
                float s_val = 0.0f;
                float dp_val = 0.0f;
                
                if (qr < S && kr < S) {
                    // Vectorized dot products (4 elements at a time)
                    for (int dd = 0; dd < d; dd += 4) {
                        float q0 = __bfloat162float(Q_sm[iq * d + dd + 0]);
                        float q1 = __bfloat162float(Q_sm[iq * d + dd + 1]);
                        float q2 = __bfloat162float(Q_sm[iq * d + dd + 2]);
                        float q3 = __bfloat162float(Q_sm[iq * d + dd + 3]);
                        
                        float k0 = __bfloat162float(K_sm[ik * d + dd + 0]);
                        float k1 = __bfloat162float(K_sm[ik * d + dd + 1]);
                        float k2 = __bfloat162float(K_sm[ik * d + dd + 2]);
                        float k3 = __bfloat162float(K_sm[ik * d + dd + 3]);
                        
                        float do0 = __bfloat162float(dO_sm[iq * d + dd + 0]);
                        float do1 = __bfloat162float(dO_sm[iq * d + dd + 1]);
                        float do2 = __bfloat162float(dO_sm[iq * d + dd + 2]);
                        float do3 = __bfloat162float(dO_sm[iq * d + dd + 3]);
                        
                        float v0 = __bfloat162float(V_sm[ik * d + dd + 0]);
                        float v1 = __bfloat162float(V_sm[ik * d + dd + 1]);
                        float v2 = __bfloat162float(V_sm[ik * d + dd + 2]);
                        float v3 = __bfloat162float(V_sm[ik * d + dd + 3]);
                        
                        s_val += q0*k0 + q1*k1 + q2*k2 + q3*k3;
                        dp_val += do0*v0 + do1*v1 + do2*v2 + do3*v3;
                    }
                }
                
                S_tile[idx] = s_val * inv_sqrt_d;
                dP_tile[idx] = dp_val;
            }
            __syncthreads();
            
            // PASS 2: Threads 0..127 accumulate dQ_acc
            if (tid < TPASS2) {
                int iq = tid / TPR2;           // 0..31
                int lane = tid % TPR2;          // 0..7
                int col_base = lane * COLS2;    // 0,16,32,...,112
                
                float l_val = L_vals[iq];
                float d_val = delta_arr[iq];
                
                for (int c = 0; c < COLS2; c++) dQ_chunk[c] = 0.0f;
                
                for (int ik = 0; ik < TILE_N; ik++) {
                    int kr = k_start + ik;
                    if (kr >= S) continue;
                    int idx = iq * TILE_N + ik;
                    float s_val = S_tile[idx];
                    float dp_val = dP_tile[idx];
                    float p_val = expf(s_val - l_val);
                    float ds_val = p_val * (dp_val - d_val);
                    for (int c = 0; c < COLS2; c++) {
                        dQ_chunk[c] += ds_val * __bfloat162float(K_sm[ik * d + col_base + c]);
                    }
                }
                
                for (int c = 0; c < COLS2; c++) {
                    dQ_acc[iq * d + col_base + c] += dQ_chunk[c];
                }
            }
            
            // PASS 3: Threads 128..255 accumulate dK, dV to global memory
            if (tid >= TPASS2) {
                int eff = tid - TPASS2;          // 0..127
                int ik = eff / TPR3;             // 0..31
                int lane = eff % TPR3;           // 0..3
                int col_base = lane * COLS3;     // 0,32,64,96
                int kr = k_start + ik;
                
                if (kr < S) {
                    for (int c = 0; c < COLS3; c++) {
                        dK_chunk[c] = 0.0f;
                        dV_chunk[c] = 0.0f;
                    }
                    
                    for (int iq = 0; iq < TILE_M; iq++) {
                        int qr = q_start + iq;
                        if (qr >= S) continue;
                        int idx = iq * TILE_N + ik;
                        float s_val = S_tile[idx];
                        float dp_val = dP_tile[idx];
                        float p_val = expf(s_val - L_vals[iq]);
                        float ds_val = p_val * (dp_val - delta_arr[iq]);
                        for (int c = 0; c < COLS3; c++) {
                            dK_chunk[c] += ds_val * __bfloat162float(Q_sm[iq * d + col_base + c]);
                            dV_chunk[c] += p_val * __bfloat162float(dO_sm[iq * d + col_base + c]);
                        }
                    }
                    
                    uint64_t base_addr = (uint64_t)kr * d;
                    for (int c = 0; c < COLS3; c++) {
                        uint64_t addr = base_addr + col_base + c;
                        dK_h[addr] = __float2bfloat16(__bfloat162float(dK_h[addr]) + dK_chunk[c]);
                        dV_h[addr] = __float2bfloat16(__bfloat162float(dV_h[addr]) + dV_chunk[c]);
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
                dQ_h[(uint64_t)qr * d + c] = __float2bfloat16(dQ_acc[idx]);
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
    int64_t d = Q.size(3);
    
    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_data = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_data = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_data = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_data = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_data = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_data = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d));
    
    int total_heads = static_cast<int>(B * H);
    int block_size = NUM_THREADS;
    int grid_size = total_heads;
    
    size_t smem_size = static_cast<size_t>(TILE_M) * d * sizeof(__nv_bfloat16) * 3 +
                       static_cast<size_t>(TILE_N) * d * sizeof(__nv_bfloat16) * 2 +
                       static_cast<size_t>(TILE_M) * TILE_N * sizeof(float) * 2 +
                       static_cast<size_t>(TILE_M) * d * sizeof(float);
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    auto kernel_fn = &mha_bwd_kernel<TILE_M, TILE_N, NUM_THREADS, EPT>;
    kernel_fn<<<grid_size, block_size, smem_size, stream>>>(
        Q_data, K_data, V_data, O_data, dO_data, L_data,
        dQ_data, dK_data, dV_data,
        static_cast<int>(B), static_cast<int>(H), static_cast<int>(S), static_cast<int>(d),
        inv_sqrt_d);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128::run);

}  // namespace mha_bwd_d128