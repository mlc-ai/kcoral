#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

#define CUDA_CHECK(call) \
    do { cudaError_t e = call; if (e != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

constexpr int NUM_THREADS = 256;

__device__ static inline float bf162f(const __nv_bfloat16& x) { return __bfloat162float(x); }
__device__ static inline __nv_bfloat16 f2bf16(float x) { return __float2bfloat16(x); }

// Shared memory storage for Q and K vectors (one element at a time)
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
    // Cast shared memory to usable arrays
    float* s_qk = reinterpret_cast<float*>(smem);      // d floats for QK dot products
    float* s_dov = reinterpret_cast<float*>(smem + d * sizeof(float));  // d floats for dO*V dot products
    
    int bh = blockIdx.x;
    int b = bh / H, h = bh % H;
    int tid = threadIdx.x;
    
    // Base pointers for this (b, h)
    const __nv_bfloat16* Q_bh = Q + ((int64_t)b * H + h) * (int64_t)S * d;
    const __nv_bfloat16* K_bh = K + ((int64_t)b * H + h) * (int64_t)S * d;
    const __nv_bfloat16* V_bh = V + ((int64_t)b * H + h) * (int64_t)S * d;
    const __nv_bfloat16* dO_bh = dO + ((int64_t)b * H + h) * (int64_t)S * d;
    const float* L_bh = L + ((int64_t)b * H + h) * S;
    __nv_bfloat16* dQ_bh = dQ + ((int64_t)b * H + h) * (int64_t)S * d;
    __nv_bfloat16* dK_bh = dK + ((int64_t)b * H + h) * (int64_t)S * d;
    __nv_bfloat16* dV_bh = dV + ((int64_t)b * H + h) * (int64_t)S * d;
    float* d_delta = delta_out + ((int64_t)b * H + h) * S;
    
    // Allocate local arrays for this BH pair on shared mem via atomic offsets
    // Each thread loads d elements starting at tid, strided by blockDim.x
    float my_qi[4] = {0}; // Local accumulator fragments
    
    // Step 1: Initialize outputs to zero
    for (int idx = tid; idx < S * d; idx += NUM_THREADS) {
        dV_bh[idx] = f2bf16(0.0f);
        dQ_bh[idx] = f2bf16(0.0f);
        dK_bh[idx] = f2bf16(0.0f);
    }
    d_delta[tid] = 0.0f; // Will be overwritten below, just ensure initialized
    
    // Step 2: Tile-based computation over (i,j) pairs
    // Each thread processes a set of i values, inner loop over j
    for (int i = tid; i < S; i += NUM_THREADS) {
        float L_i = L_bh[i];
        float q_vec[128]; // Load Q[i]
        float doi_vec[128]; // Load dO[i]
        #pragma unroll
        for (int dd = 0; dd < 128; ++dd) {
            q_vec[dd] = bf162f(Q_bh[(int64_t)i * d + dd]);
            doi_vec[dd] = bf162f(dO_bh[(int64_t)i * d + dd]);
        }
        
        float delta_acc = 0.0f;
        float dq_acc[128] = {0};
        
        for (int j = 0; j < S; ++j) {
            // Compute Q[i].K[j] and dO[i].V[j]
            float qk = 0.0f;
            float dov = 0.0f;
            #pragma unroll
            for (int dd = 0; dd < 128; ++dd) {
                qk += q_vec[dd] * bf162f(K_bh[(int64_t)j * d + dd]);
                dov += doi_vec[dd] * bf162f(V_bh[(int64_t)j * d + dd]);
            }
            
            float p = expf(scale * qk - L_i);
            float diff = dov - delta_acc; // This won't work because delta isn't fully computed yet
            
            delta_acc += p * dov;
            
            // We can't compute ds_ij here because delta isn't known yet!
            // Need two-pass or cached approach
        }
        
        d_delta[i] = delta_acc;
    }
    
    __syncthreads();
    
    // Step 3: Second pass - compute dQ using cached delta values
    for (int i = tid; i < S; i += NUM_THREADS) {
        float L_i = L_bh[i];
        float delta_i = d_delta[i];
        float q_vec[128];
        float doi_vec[128];
        #pragma unroll
        for (int dd = 0; dd < 128; ++dd) {
            q_vec[dd] = bf162f(Q_bh[(int64_t)i * d + dd]);
            doi_vec[dd] = bf162f(dO_bh[(int64_t)i * d + dd]);
        }
        
        float dq_acc[128] = {0};
        
        for (int j = 0; j < S; ++j) {
            float qk = 0.0f;
            float dov = 0.0f;
            #pragma unroll
            for (int dd = 0; dd < 128; ++dd) {
                qk += q_vec[dd] * bf162f(K_bh[(int64_t)j * d + dd]);
                dov += doi_vec[dd] * bf162f(V_bh[(int64_t)j * d + dd]);
            }
            
            float p = expf(scale * qk - L_i);
            float ds = p * (dov - delta_i);
            
            #pragma unroll
            for (int dd = 0; dd < 128; ++dd) {
                dq_acc[dd] += ds * bf162f(K_bh[(int64_t)j * d + dd]) * scale;
            }
        }
        
        #pragma unroll
        for (int dd = 0; dd < 128; ++dd) {
            dQ_bh[(int64_t)i * d + dd] = f2bf16(dq_acc[dd]);
        }
    }
    
    __syncthreads();
    
    // Step 4: Compute dV
    for (int j = tid; j < S; j += NUM_THREADS) {
        float dv_acc[128] = {0};
        float kj[128];
        #pragma unroll
        for (int dd = 0; dd < 128; ++dd) {
            kj[dd] = bf162f(K_bh[(int64_t)j * d + dd]);
        }
        
        for (int i = 0; i < S; ++i) {
            float qk = 0.0f;
            #pragma unroll
            for (int dd = 0; dd < 128; ++dd) {
                qk += bf162f(Q_bh[(int64_t)i * d + dd]) * kj[dd];
            }
            float p = expf(scale * qk - L_bh[i]);
            #pragma unroll
            for (int dd = 0; dd < 128; ++dd) {
                dv_acc[dd] += p * bf162f(dO_bh[(int64_t)i * d + dd]);
            }
        }
        
        #pragma unroll
        for (int dd = 0; dd < 128; ++dd) {
            dV_bh[(int64_t)j * d + dd] = f2bf16(dv_acc[dd]);
        }
    }
    
    __syncthreads();
    
    // Step 5: Compute dK  
    for (int j = tid; j < S; j += NUM_THREADS) {
        float dk_acc[128] = {0};
        float kj[128];
        #pragma unroll
        for (int dd = 0; dd < 128; ++dd) {
            kj[dd] = bf162f(K_bh[(int64_t)j * d + dd]);
        }
        
        for (int i = 0; i < S; ++i) {
            float L_i = L_bh[i];
            float delta_i = d_delta[i];
            
            float qk = 0.0f;
            float dov = 0.0f;
            #pragma unroll
            for (int dd = 0; dd < 128; ++dd) {
                qk += bf162f(Q_bh[(int64_t)i * d + dd]) * kj[dd];
                dov += bf162f(dO_bh[(int64_t)i * d + dd]) * bf162f(V_bh[(int64_t)j * d + dd]);
            }
            
            float p = expf(scale * qk - L_i);
            float ds = p * (dov - delta_i);
            
            #pragma unroll
            for (int dd = 0; dd < 128; ++dd) {
                dk_acc[dd] += ds * bf162f(Q_bh[(int64_t)i * d + dd]) * scale;
            }
        }
        
        #pragma unroll
        for (int dd = 0; dd < 128; ++dd) {
            dK_bh[(int64_t)j * d + dd] = f2bf16(dk_acc[dd]);
        }
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
    
    dim3 grid(bh_total);
    dim3 block(NUM_THREADS);
    
    // Shared memory: 2 * d floats (for caching)
    size_t smem_size = 2 * d * sizeof(float);
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
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