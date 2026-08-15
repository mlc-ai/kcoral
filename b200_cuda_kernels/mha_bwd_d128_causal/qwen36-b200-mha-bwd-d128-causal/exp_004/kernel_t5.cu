#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
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

namespace mha_bwd_impl {

__device__ __forceinline__ static float bf16tof(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ static __nv_bfloat16 f2bf16(float x) {
    return __float2bfloat16(x);
}

constexpr int D = 128;      // Head dimension (constant from task spec)
constexpr int TM = 64;      // Tile M size (query seq)
constexpr int TN = 32;      // Tile N size (key seq)

/**
 * Pass 1: Accumulate partial D values and dV into global memory
 */
__global__ void mha_bwd_pass1_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const float* __restrict__ L,
    const __nv_bfloat16* __restrict__ dO,
    __nv_bfloat16* __restrict__ dV,
    float* __restrict__ dAccD,
    int S, int num_heads, float inv_sqrt_d)
{
    int bh = blockIdx.x;
    if (bh >= (int)gridDim.x / gridDim.y || S <= 0) return;
    
    int n_tile_idx = blockIdx.y;
    int b = bh / num_heads;
    int h = bh % num_heads;
    
    int tid = threadIdx.x;
    int nt = blockDim.x;
    
    int bm_start = blockIdx.z * TM;
    int bn_start = n_tile_idx * TN;
    
    uint64_t off = (uint64_t)b * num_heads + h;
    
    const __nv_bfloat16* qb = Q + off * S * D;
    const __nv_bfloat16* kb = K + off * S * D;
    const __nv_bfloat16* vb = V + off * S * D;
    const float* lb = L + off * S;
    const __nv_bfloat16* dob = dO + off * S * D;
    
    extern __shared__ char smem[];
    
    __align__(16) __nv_bfloat16* s_Q = (__nv_bfloat16*)smem;
    __align__(16) __nv_bfloat16* s_K = s_Q + TM * D;
    __align__(16) __nv_bfloat16* s_V = s_K + TN * D;
    __align__(16) __nv_bfloat16* s_dO = s_V + TN * D;
    
    // Load Q tile
    for (int idx = tid; idx < TM * D; idx += nt) {
        int i = idx / D;
        int f = idx % D;
        int gi = bm_start + i;
        s_Q[idx] = (gi < S) ? qb[gi * D + f] : f2bf16(0.f);
    }
    
    // Load K tile
    for (int idx = tid; idx < TN * D; idx += nt) {
        int j = idx / D;
        int f = idx % D;
        int gj = bn_start + j;
        s_K[idx] = (gj < S) ? kb[gj * D + f] : f2bf16(0.f);
    }
    
    // Load V tile
    for (int idx = tid; idx < TN * D; idx += nt) {
        int j = idx / D;
        int f = idx % D;
        int gj = bn_start + j;
        s_V[idx] = (gj < S) ? vb[gj * D + f] : f2bf16(0.f);
    }
    
    // Load dO tile
    for (int idx = tid; idx < TM * D; idx += nt) {
        int i = idx / D;
        int f = idx % D;
        int gi = bm_start + i;
        s_dO[idx] = (gi < S) ? dob[gi * D + f] : f2bf16(0.f);
    }
    
    __syncthreads();
    
    // Each thread computes partial_dD for its assigned query rows
    for (int i_local = tid; i_local < TM; i_local += nt) {
        int gi = bm_start + i_local;
        if (gi >= S) continue;
        
        float Li = lb[gi];
        float partial_di = 0.f;
        
        for (int j_local = 0; j_local < TN; j_local++) {
            int gj = bn_start + j_local;
            if (gj > gi || gj >= S) continue;
            
            // Dot products unrolled by 4
            float sim = 0.f, delta = 0.f;
            #pragma unroll
            for (int f = 0; f < D; f += 4) {
                sim   += bf16tof(s_Q[i_local*D+f])     * bf16tof(s_K[j_local*D+f]);
                sim   += bf16tof(s_Q[i_local*D+f+1])   * bf16tof(s_K[j_local*D+f+1]);
                sim   += bf16tof(s_Q[i_local*D+f+2])   * bf16tof(s_K[j_local*D+f+2]);
                sim   += bf16tof(s_Q[i_local*D+f+3])   * bf16tof(s_K[j_local*D+f+3]);
                delta += bf16tof(s_dO[i_local*D+f])     * bf16tof(s_V[j_local*D+f]);
                delta += bf16tof(s_dO[i_local*D+f+1])   * bf16tof(s_V[j_local*D+f+1]);
                delta += bf16tof(s_dO[i_local*D+f+2])   * bf16tof(s_V[j_local*D+f+2]);
                delta += bf16tof(s_dO[i_local*D+f+3])   * bf16tof(s_V[j_local*D+f+3]);
            }
            
            float Pij = expf(sim * inv_sqrt_d - Li);
            partial_di += Pij * delta;
        }
        
        atomicAdd(&dAccD[bh * S + gi], partial_di);
    }
    
    __syncthreads();
    
    // Each thread contributes to dV for its assigned key rows
    for (int j_local = tid; j_local < TN; j_local += nt) {
        int gj = bn_start + j_local;
        if (gj >= S) continue;
        
        // Accumulate dV contribution into global memory via atomics per feature chunk
        for (int i_local = 0; i_local < TM; i_local++) {
            int gi = bm_start + i_local;
            if (gi >= S || gj > gi) continue;
            
            float sim = 0.f;
            #pragma unroll
            for (int f = 0; f < D; f += 4) {
                sim += bf16tof(s_Q[i_local*D+f])     * bf16tof(s_K[j_local*D+f]);
                sim += bf16tof(s_Q[i_local*D+f+1])   * bf16tof(s_K[j_local*D+f+1]);
                sim += bf16tof(s_Q[i_local*D+f+2])   * bf16tof(s_K[j_local*D+f+2]);
                sim += bf16tof(s_Q[i_local*D+f+3])   * bf16tof(s_K[j_local*D+f+3]);
            }
            float Pij = expf(sim * inv_sqrt_d - lb[gi]);
            
            // Atomic add to dV
            int dv_base = (off * S + gj) * D;
            #pragma unroll
            for (int f = 0; f < D; f += 4) {
                atomicAdd(reinterpret_cast<float*>(dV) + dv_base + f,
                          Pij * bf16tof(s_dO[i_local*D+f]));
                atomicAdd(reinterpret_cast<float*>(dV) + dv_base + f + 1,
                          Pij * bf16tof(s_dO[i_local*D+f+1]));
                atomicAdd(reinterpret_cast<float*>(dV) + dv_base + f + 2,
                          Pij * bf16tof(s_dO[i_local*D+f+2]));
                atomicAdd(reinterpret_cast<float*>(dV) + dv_base + f + 3,
                          Pij * bf16tof(s_dO[i_local*D+f+3]));
            }
        }
    }
}

/**
 * Pass 2: Compute dQ and dK using final D values
 */
__global__ void mha_bwd_pass2_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const float* __restrict__ L,
    const __nv_bfloat16* __restrict__ dO,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    const float* __restrict__ D_vals,
    int S, int num_heads, float inv_sqrt_d)
{
    int bh = blockIdx.x;
    if (bh >= (int)gridDim.x / gridDim.y || S <= 0) return;
    
    int n_tile_idx = blockIdx.y;
    int b = bh / num_heads;
    int h = bh % num_heads;
    
    int tid = threadIdx.x;
    int nt = blockDim.x;
    
    int bm_start = blockIdx.z * TM;
    int bn_start = n_tile_idx * TN;
    
    uint64_t off = (uint64_t)b * num_heads + h;
    
    const __nv_bfloat16* qb = Q + off * S * D;
    const __nv_bfloat16* kb = K + off * S * D;
    const __nv_bfloat16* vb = V + off * S * D;
    const float* lb = L + off * S;
    const __nv_bfloat16* dob = dO + off * S * D;
    const float* Db = D_vals + bh * S;
    
    extern __shared__ char smem[];
    
    __align__(16) __nv_bfloat16* s_Q = (__nv_bfloat16*)smem;
    __align__(16) __nv_bfloat16* s_K = s_Q + TM * D;
    __align__(16) __nv_bfloat16* s_V = s_K + TN * D;
    __align__(16) __nv_bfloat16* s_dO = s_V + TN * D;
    
    // Load tiles
    for (int idx = tid; idx < TM * D; idx += nt) {
        int i = idx / D;
        int f = idx % D;
        int gi = bm_start + i;
        s_Q[idx] = (gi < S) ? qb[gi * D + f] : f2bf16(0.f);
    }
    for (int idx = tid; idx < TN * D; idx += nt) {
        int j = idx / D;
        int f = idx % D;
        int gj = bn_start + j;
        s_K[idx] = (gj < S) ? kb[gj * D + f] : f2bf16(0.f);
    }
    for (int idx = tid; idx < TN * D; idx += nt) {
        int j = idx / D;
        int f = idx % D;
        int gj = bn_start + j;
        s_V[idx] = (gj < S) ? vb[gj * D + f] : f2bf16(0.f);
    }
    for (int idx = tid; idx < TM * D; idx += nt) {
        int i = idx / D;
        int f = idx % D;
        int gi = bm_start + i;
        s_dO[idx] = (gi < S) ? dob[gi * D + f] : f2bf16(0.f);
    }
    __syncthreads();
    
    // Compute dQ: each thread handles one or more query rows
    for (int i_local = tid; i_local < TM; i_local += nt) {
        int gi = bm_start + i_local;
        if (gi >= S) continue;
        
        float Di = Db[gi];
        float Li = lb[gi];
        float dq_acc = 0.f;  // we accumulate into globals directly
        
        int dQ_base = (off * S + gi) * D;
        
        for (int j_local = 0; j_local < TN; j_local++) {
            int gj = bn_start + j_local;
            if (gj > gi || gj >= S) continue;
            
            float sim = 0.f, delta = 0.f;
            #pragma unroll
            for (int f = 0; f < D; f += 4) {
                sim   += bf16tof(s_Q[i_local*D+f])     * bf16tof(s_K[j_local*D+f]);
                sim   += bf16tof(s_Q[i_local*D+f+1])   * bf16tof(s_K[j_local*D+f+1]);
                sim   += bf16tof(s_Q[i_local*D+f+2])   * bf16tof(s_K[j_local*D+f+2]);
                sim   += bf16tof(s_Q[i_local*D+f+3])   * bf16tof(s_K[j_local*D+f+3]);
                delta += bf16tof(s_dO[i_local*D+f])     * bf16tof(s_V[j_local*D+f]);
                delta += bf16tof(s_dO[i_local*D+f+1])   * bf16tof(s_V[j_local*D+f+1]);
                delta += bf16tof(s_dO[i_local*D+f+2])   * bf16tof(s_V[j_local*D+f+2]);
                delta += bf16tof(s_dO[i_local*D+f+3])   * bf16tof(s_V[j_local*D+f+3]);
            }
            
            float Pij = expf(sim * inv_sqrt_d - Li);
            float dSij = Pij * (delta - Di);
            
            // Atomic add dQ contributions
            #pragma unroll
            for (int f = 0; f < D; f += 4) {
                atomicAdd(reinterpret_cast<float*>(dQ) + dQ_base + f,
                          dSij * bf16tof(s_K[j_local*D+f]));
                atomicAdd(reinterpret_cast<float*>(dQ) + dQ_base + f + 1,
                          dSij * bf16tof(s_K[j_local*D+f+1]));
                atomicAdd(reinterpret_cast<float*>(dQ) + dQ_base + f + 2,
                          dSij * bf16tof(s_K[j_local*D+f+2]));
                atomicAdd(reinterpret_cast<float*>(dQ) + dQ_base + f + 3,
                          dSij * bf16tof(s_K[j_local*D+f+3]));
            }
        }
    }
    
    // Compute dK: each thread handles one or more key rows
    for (int j_local = tid; j_local < TN; j_local += nt) {
        int gj = bn_start + j_local;
        if (gj >= S) continue;
        
        int dK_base = (off * S + gj) * D;
        
        for (int i_local = 0; i_local < TM; i_local++) {
            int gi = bm_start + i_local;
            if (gi >= S || gj > gi) continue;
            
            float sim = 0.f, delta = 0.f;
            #pragma unroll
            for (int f = 0; f < D; f += 4) {
                sim   += bf16tof(s_Q[i_local*D+f])     * bf16tof(s_K[j_local*D+f]);
                sim   += bf16tof(s_Q[i_local*D+f+1])   * bf16tof(s_K[j_local*D+f+1]);
                sim   += bf16tof(s_Q[i_local*D+f+2])   * bf16tof(s_K[j_local*D+f+2]);
                sim   += bf16tof(s_Q[i_local*D+f+3])   * bf16tof(s_K[j_local*D+f+3]);
                delta += bf16tof(s_dO[i_local*D+f])     * bf16tof(s_V[j_local*D+f]);
                delta += bf16tof(s_dO[i_local*D+f+1])   * bf16tof(s_V[j_local*D+f+1]);
                delta += bf16tof(s_dO[i_local*D+f+2])   * bf16tof(s_V[j_local*D+f+2]);
                delta += bf16tof(s_dO[i_local*D+f+3])   * bf16tof(s_V[j_local*D+f+3]);
            }
            
            float Pij = expf(sim * inv_sqrt_d - lb[gi]);
            float dSij = Pij * (delta - Db[gi]);
            
            #pragma unroll
            for (int f = 0; f < D; f += 4) {
                atomicAdd(reinterpret_cast<float*>(dK) + dK_base + f,
                          dSij * bf16tof(s_Q[i_local*D+f]));
                atomicAdd(reinterpret_cast<float*>(dK) + dK_base + f + 1,
                          dSij * bf16tof(s_Q[i_local*D+f+1]));
                atomicAdd(reinterpret_cast<float*>(dK) + dK_base + f + 2,
                          dSij * bf16tof(s_Q[i_local*D+f+2]));
                atomicAdd(reinterpret_cast<float*>(dK) + dK_base + f + 3,
                          dSij * bf16tof(s_Q[i_local*D+f+3]));
            }
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {

    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t ddim = Q.size(3);

    (void)ddim; // Always 128 per task spec
    
    int64_t num_bh = B * H;

    const __nv_bfloat16* ptr_Q  = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* ptr_K  = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* ptr_V  = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const float* ptr_L          = static_cast<const float*>(L.data_ptr());
    const __nv_bfloat16* ptr_dO = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    __nv_bfloat16* ptr_dQ       = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* ptr_dK       = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* ptr_dV       = static_cast<__nv_bfloat16*>(dV.data_ptr());

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    // Zero outputs
    size_t out_bytes = num_bh * S * D * sizeof(__nv_bfloat16);
    CUDA_CHECK(cudaMemsetAsync(ptr_dQ, 0, out_bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(ptr_dK, 0, out_bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(ptr_dV, 0, out_bytes, stream));

    // Allocate D accumulator [num_bh, S] fp32
    float* d_D = nullptr;
    size_t d_buf_bytes = num_bh * S * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_D, d_buf_bytes));
    CUDA_CHECK(cudaMemsetAsync(d_D, 0, d_buf_bytes, stream));

    float inv_sqrt_d = 1.0f / sqrtf((float)D);
    
    int n_tiles_n = (S + TN - 1) / TN;
    int n_tiles_m = (S + TM - 1) / TM;
    int threads = 128;
    
    // Shared memory: Q(TM*D), K(TN*D), V(TN*D), dO(TM*D) bytes
    size_t smem_size = (2ULL * TM + 2ULL * TN) * D * sizeof(__nv_bfloat16);

    dim3 grid(num_bh, n_tiles_n, n_tiles_m);
    dim3 block(threads);

    mha_bwd_pass1_kernel<<<grid, block, smem_size, stream>>>(
        ptr_Q, ptr_K, ptr_V, ptr_L, ptr_dO,
        ptr_dV, d_D,
        (int)S, (int)H, inv_sqrt_d
    );
    CUDA_CHECK(cudaGetLastError());

    mha_bwd_pass2_kernel<<<grid, block, smem_size, stream>>>(
        ptr_Q, ptr_K, ptr_V, ptr_L, ptr_dO,
        ptr_dQ, ptr_dK, d_D,
        (int)S, (int)H, inv_sqrt_d
    );
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_D));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_impl::run);

} // namespace mha_bwd_impl