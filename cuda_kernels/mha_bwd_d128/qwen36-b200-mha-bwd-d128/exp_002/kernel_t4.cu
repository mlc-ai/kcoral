#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

#define CUDA_CHECK(call) \
    do { cudaError_t e = call; if (e != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

constexpr int BLOCK_DIM_X = 32;
constexpr int BLOCK_DIM_Y = 8;
constexpr int TILE_S = 64;  // Tile size along sequence dimension
constexpr int TILE_D = 16;  // Tile size along head dimension
constexpr int NUM_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y; // 256

__device__ static inline float bf162f(const __nv_bfloat16& x) { return __bfloat162float(x); }
__device__ static inline __nv_bfloat16 f2bf16(float x) { return __float2bfloat16(x); }

extern __shared__ char smem[];

__global__ void mha_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    float* __restrict__ delta_out,
    int B, int H, int S, int d, float scale)
{
    // Shared memory layout:
    // s_Q:    [TILE_S][d]     - current tile of Q (loaded when processing new i-tile)
    // s_dO:   [TILE_S][d]     - current tile of dO
    // s_K:    [TILE_S][d]     - current tile of K (loaded when processing new j-tile)
    // s_V:    [TILE_S][d]     - current tile of V
    
    // Since d=128, TILE_S=64 => each buffer needs 64*128 = 8192 entries
    // Total shared: 4 * 8192 * sizeof(__nv_bfloat16) = ~512KB - too large!
    
    // Smaller approach: thread-local vectors, partial accumulation
    int bh = blockIdx.x;
    int b = bh / H, h = bh % H;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * BLOCK_DIM_X + tx;
    
    const __nv_bfloat16* Q_bh = &Q[((int64_t)b * H + h) * (int64_t)S * d];
    const __nv_bfloat16* K_bh = &K[((int64_t)b * H + h) * (int64_t)S * d];
    const __nv_bfloat16* V_bh = &V[((int64_t)b * H + h) * (int64_t)S * d];
    const __nv_bfloat16* dO_bh = &dO[((int64_t)b * H + h) * (int64_t)S * d];
    const float* L_bh = &L[((int64_t)b * H + h) * S];
    __nv_bfloat16* dQ_bh = &dQ[((int64_t)b * H + h) * (int64_t)S * d];
    __nv_bfloat16* dK_bh = &dK[((int64_t)b * H + h) * (int64_t)S * d];
    __nv_bfloat16* dV_bh = &dV[((int64_t)b * H + h) * (int64_t)S * d];
    float* d_delta = &delta_out[((int64_t)b * H + h) * S];
    
    // Phase 1: Compute delta[i] for all i (atomic reduction per row)
    // Split among threads: each thread computes some i values
    for (int i = tid; i < S; i += NUM_THREADS) {
        float acc = 0.0f;
        float L_i = L_bh[i];
        // Store Q[i] and dO[i] locally for repeated access
        float qcache[32];  // 32 bf16 elements, strided access
        float docache[32];
        
        for (int stride = 0; stride < 4; ++stride) {
            int base_dd = tx + stride * 32;
            if (base_dd < d) {
                qcache[stride] = bf162f(Q_bh[(int64_t)i * d + base_dd]);
                docache[stride] = bf162f(dO_bh[(int64_t)i * d + base_dd]);
            } else {
                qcache[stride] = 0.0f;
                docache[stride] = 0.0f;
            }
        }
        
        for (int j = 0; j < S; ++j) {
            float qk = 0.0f, dov = 0.0f;
            int jj_base = (int64_t)j * d;
            for (int stride = 0; stride < 4; ++stride) {
                int dd = tx + stride * 32;
                if (dd < d) {
                    qk += qcache[stride] * bf162f(K_bh[jj_base + dd]);
                    dov += docache[stride] * bf162f(V_bh[jj_base + dd]);
                }
            }
            float p = expf(scale * qk - L_i);
            acc += p * dov;
        }
        
        atomicAdd(&d_delta[i], acc);
    }
    __syncthreads();
    
    // Phase 2: Compute dV[j][dd] = sum_i P[i][j] * dO[i][dd]
    // Also computed per-element contribution from atomicAdd style accumulation
    for (int out_idx = tid; out_idx < S * d; out_idx += NUM_THREADS) {
        int j = out_idx / d;
        int dd = out_idx % d;
        float acc = 0.0f;
        for (int i = 0; i < S; ++i) {
            float qk = 0.0f;
            for (int stride = 0; stride < 4; ++stride) {
                int ddd = tx + stride * 32;
                if (ddd < d) {
                    qk += bf162f(Q_bh[(int64_t)i * d + ddd]) * bf162f(K_bh[(int64_t)j * d + ddd]);
                }
            }
            float p = expf(scale * qk - L_bh[i]);
            acc += p * bf162f(dO_bh[(int64_t)i * d + dd]);
        }
        atomicAdd((float*)&dV_bh[out_idx], acc);
    }
    __syncthreads();
    
    // Phase 3: Compute dQ[i][di] = scale * sum_j P*(dOV-delta)*K[j][di]
    for (int out_idx = tid; out_idx < S * d; out_idx += NUM_THREADS) {
        int i = out_idx / d;
        int di = out_idx % d;
        float acc = 0.0f;
        float L_i = L_bh[i];
        float delta_i = d_delta[i];
        
        float qcache[4], docache[4];
        for (int stride = 0; stride < 4; ++stride) {
            int ddd = tx + stride * 32;
            if (ddd < d) {
                qcache[stride] = bf162f(Q_bh[(int64_t)i * d + ddd]);
                docache[stride] = bf162f(dO_bh[(int64_t)i * d + ddd]);
            } else {
                qcache[stride] = 0.0f;
                docache[stride] = 0.0f;
            }
        }
        
        for (int j = 0; j < S; ++j) {
            float qk = 0.0f, dov = 0.0f;
            int jj = (int64_t)j * d;
            for (int stride = 0; stride < 4; ++stride) {
                int ddd = tx + stride * 32;
                if (ddd < d) {
                    qk += qcache[stride] * bf162f(K_bh[jj + ddd]);
                    dov += docache[stride] * bf162f(V_bh[jj + ddd]);
                }
            }
            float p = expf(scale * qk - L_i);
            float ds = p * (dov - delta_i);
            acc += ds * bf162f(K_bh[jj + di]) * scale;
        }
        atomicAdd((float*)&dQ_bh[out_idx], acc);
    }
    __syncthreads();
    
    // Phase 4: Compute dK[j][dj] = scale * sum_i P*(dOV-delta)*Q[i][dj]
    for (int out_idx = tid; out_idx < S * d; out_idx += NUM_THREADS) {
        int j = out_idx / d;
        int dj = out_idx % d;
        float acc = 0.0f;
        int jj = (int64_t)j * d;
        float kcache[4];
        for (int stride = 0; stride < 4; ++stride) {
            int ddd = tx + stride * 32;
            if (ddd < d) {
                kcache[stride] = bf162f(K_bh[jj + ddd]);
            } else {
                kcache[stride] = 0.0f;
            }
        }
        
        for (int i = 0; i < S; ++i) {
            float L_i = L_bh[i];
            float delta_i = d_delta[i];
            float qk = 0.0f, dov = 0.0f;
            for (int stride = 0; stride < 4; ++stride) {
                int ddd = tx + stride * 32;
                if (ddd < d) {
                    qk += bf162f(Q_bh[(int64_t)i * d + ddd]) * kcache[stride];
                    dov += bf162f(dO_bh[(int64_t)i * d + ddd]) * bf162f(V_bh[jj + ddd]);
                }
            }
            float p = expf(scale * qk - L_i);
            float ds = p * (dov - delta_i);
            acc += ds * bf162f(Q_bh[(int64_t)i * d + dj]) * scale;
        }
        atomicAdd((float*)&dK_bh[out_idx], acc);
    }
}

namespace mha_bwd_impl {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = 4, H = 48, d = 128;
    int64_t S = Q.size(2);
    float scale = 1.0f / std::sqrt(static_cast<float>(d));
    
    int64_t bh_total = B * H;
    size_t delta_bytes = bh_total * S * sizeof(float);
    float* d_delta = nullptr;
    CUDA_CHECK(cudaMalloc(&d_delta, delta_bytes));
    CUDA_CHECK(cudaMemsetAsync(d_delta, 0, delta_bytes));
    
    dim3 grid(bh_total);
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    size_t smem_size = 0;  // Using register-based caching instead of shared mem
    
    mha_bwd_kernel<<<grid, block, smem_size, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        d_delta,
        static_cast<int>(B), static_cast<int>(H), static_cast<int>(S),
        static_cast<int>(d), scale
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_delta));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_impl::run);

} // namespace mha_bwd_impl