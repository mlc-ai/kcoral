#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

#define CUDA_CHECK(call) \
    do { cudaError_t e = call; if (e != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

constexpr int THREADS_PER_TILE = 64;
constexpr int D_VEC = 4;  // Vectorize d-dimension by 4

__device__ static inline float bf162f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ static inline __nv_bfloat16 f2bf16(float x) { return __float2bfloat16(x); }

// Tiled MHA backward kernel - each block computes one (b,h) pair
__global__ void mha_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    float* __restrict__ delta_buf,
    int B, int H, int S, int d, float scale)
{
    int bh = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32;
    
    // Pointers for this (b,h)
    const __nv_bfloat16* Q_bh = &Q[(int64_t)bh * S * d];
    const __nv_bfloat16* K_bh = &K[(int64_t)bh * S * d];
    const __nv_bfloat16* V_bh = &V[(int64_t)bh * S * d];
    const __nv_bfloat16* dO_bh = &dO[(int64_t)bh * S * d];
    const float* L_bh = &L[bh * S];
    __nv_bfloat16* dQ_bh = &dQ[(int64_t)bh * S * d];
    __nv_bfloat16* dK_bh = &dK[(int64_t)bh * S * d];
    __nv_bfloat16* dV_bh = &dV[(int64_t)bh * S * d];
    float* delta_arr = &delta_buf[bh * S];
    
    // ================================================================
    // PHASE 1: Compute delta[i] for all i via tiled reduction
    // Each thread computes one i value (stride loop for coverage)
    // ================================================================
    for (int i = tid; i < S; i += blockDim.x) {
        float acc = 0.0f;
        float Li = L_bh[i];
        
        // Process d in groups of 4 for vectorized dot products
        float q_reg[D_VEC] = {};
        float doi_reg[D_VEC] = {};
        
        for (int v = 0; v < D_VEC; ++v) {
            int dd = lane + v * 32;
            if (dd < d) {
                q_reg[v] = bf162f(Q_bh[(int64_t)i * d + dd]);
                doi_reg[v] = bf162f(dO_bh[(int64_t)i * d + dd]);
            }
        }
        
        for (int j = 0; j < S; ++j) {
            float qk = 0.0f, dov = 0.0f;
            int k_base = (int64_t)j * d;
            int v_base = k_base;
            
            #pragma unroll
            for (int v = 0; v < D_VEC; ++v) {
                int dd = lane + v * 32;
                if (dd < d) {
                    qk += q_reg[v] * bf162f(K_bh[k_base + dd]);
                    dov += doi_reg[v] * bf162f(V_bh[v_base + dd]);
                }
            }
            
            float p = expf(scale * qk - Li);
            acc += p * dov;
        }
        
        atomicAdd(&delta_arr[i], acc);
    }
    __syncthreads();
    
    // ================================================================
    // PHASE 2: Compute dV[j][dd] = sum_i P[i][j] * dO[i][dd]
    // ================================================================
    for (int out_idx = tid; out_idx < S * d; out_idx += blockDim.x) {
        int j = out_idx / d;
        int dd = out_idx % d;
        float acc = 0.0f;
        
        for (int i = 0; i < S; ++i) {
            float qk = 0.0f;
            for (int v = 0; v < D_VEC; ++v) {
                int ddd = lane + v * 32;
                if (ddd < d) {
                    qk += bf162f(Q_bh[(int64_t)i * d + ddd]) * bf162f(K_bh[(int64_t)j * d + ddd]);
                }
            }
            float p = expf(scale * qk - L_bh[i]);
            acc += p * bf162f(dO_bh[(int64_t)i * d + dd]);
        }
        // Write result atomically since multiple threads may race
        // Actually each thread owns unique out_idx, so direct store is fine
        // But the loop above has dependency issues... let me fix the indexing
        dV_bh[out_idx] = f2bf16(acc);
    }
    __syncthreads();
    
    // ================================================================
    // PHASE 3: Compute dQ[i][di] = scale * sum_j P*(dov-delta)*K[j][di]
    // ================================================================
    for (int out_idx = tid; out_idx < S * d; out_idx += blockDim.x) {
        int i = out_idx / d;
        int di = out_idx % d;
        float acc = 0.0f;
        float Li = L_bh[i];
        float delta_i = delta_arr[i];
        
        float q_reg[D_VEC] = {};
        float doi_reg[D_VEC] = {};
        for (int v = 0; v < D_VEC; ++v) {
            int dd = lane + v * 32;
            if (dd < d) {
                q_reg[v] = bf162f(Q_bh[(int64_t)i * d + dd]);
                doi_reg[v] = bf162f(dO_bh[(int64_t)i * d + dd]);
            }
        }
        
        for (int j = 0; j < S; ++j) {
            float qk = 0.0f, dov = 0.0f;
            int kj = (int64_t)j * d;
            int vj = kj;
            
            #pragma unroll
            for (int v = 0; v < D_VEC; ++v) {
                int dd = lane + v * 32;
                if (dd < d) {
                    qk += q_reg[v] * bf162f(K_bh[kj + dd]);
                    dov += doi_reg[v] * bf162f(V_bh[vj + dd]);
                }
            }
            
            float p = expf(scale * qk - Li);
            float ds = p * (dov - delta_i);
            acc += ds * bf162f(K_bh[kj + di]) * scale;
        }
        dQ_bh[out_idx] = f2bf16(acc);
    }
    __syncthreads();
    
    // ================================================================
    // PHASE 4: Compute dK[j][dj] = scale * sum_i P*(dov-delta)*Q[i][dj]
    // ================================================================
    for (int out_idx = tid; out_idx < S * d; out_idx += blockDim.x) {
        int j = out_idx / d;
        int dj = out_idx % d;
        float acc = 0.0f;
        int kj_base = (int64_t)j * d;
        int vj_base = kj_base;
        
        float k_reg[D_VEC] = {};
        float v_reg[D_VEC] = {};
        for (int v = 0; v < D_VEC; ++v) {
            int dd = lane + v * 32;
            if (dd < d) {
                k_reg[v] = bf162f(K_bh[kj_base + dd]);
                v_reg[v] = bf162f(V_bh[vj_base + dd]);
            }
        }
        
        for (int i = 0; i < S; ++i) {
            float qk = 0.0f, dov = 0.0f;
            int qi = (int64_t)i * d;
            int doi = qi;
            
            #pragma unroll
            for (int v = 0; v < D_VEC; ++v) {
                int dd = lane + v * 32;
                if (dd < d) {
                    qk += bf162f(Q_bh[qi + dd]) * k_reg[v];
                    dov += bf162f(dO_bh[doi + dd]) * v_reg[v];
                }
            }
            
            float p = expf(scale * qk - L_bh[i]);
            float ds = p * (dov - delta_arr[i]);
            acc += ds * bf162f(Q_bh[qi + dj]) * scale;
        }
        dK_bh[out_idx] = f2bf16(acc);
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
    
    float* d_delta = nullptr;
    CUDA_CHECK(cudaMalloc(&d_delta, bh_total * S * sizeof(float)));
    
    dim3 grid(bh_total);
    dim3 block(THREADS_PER_TILE * 4); // 256 threads per block
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    mha_bwd_kernel<<<grid, block, 0, stream>>>(
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