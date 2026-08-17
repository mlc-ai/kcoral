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
constexpr int NUM_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y;

__device__ static inline float bf162f(const __nv_bfloat16& x) { return __bfloat162float(x); }
__device__ static inline __nv_bfloat16 f2bf16(float x) { return __float2bfloat16(x); }

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
    int bh = blockIdx.x;
    int tx = threadIdx.x;
    int stride_x = blockDim.x;
    int lane = tx & 31;
    int warp_lane_in_block = tx % NUM_THREADS;
    
    const __nv_bfloat16* Q_bh = &Q[(int64_t)bh * S * d];
    const __nv_bfloat16* K_bh = &K[(int64_t)bh * S * d];
    const __nv_bfloat16* V_bh = &V[(int64_t)bh * S * d];
    const __nv_bfloat16* dO_bh = &dO[(int64_t)bh * S * d];
    const float* L_bh = &L[bh * S];
    __nv_bfloat16* dQ_bh = &dQ[(int64_t)bh * S * d];
    __nv_bfloat16* dK_bh = &dK[(int64_t)bh * S * d];
    __nv_bfloat16* dV_bh = &dV[(int64_t)bh * S * d];
    float* d_delta = &delta_out[bh * S];
    
    // ============================================================
    // Phase 1: Compute delta[i] = sum_j P[i][j] * dot(dO[i], V[j])
    // Each thread processes i values with stride NUM_THREADS
    // ============================================================
    for (int i = warp_lane_in_block; i < S; i += NUM_THREADS) {
        float acc = 0.0f;
        float L_i = L_bh[i];
        
        for (int j = 0; j < S; ++j) {
            float qk = 0.0f;
            float dov = 0.0f;
            
            for (int cc = 0; cc < 4; ++cc) {
                int dd = lane + cc * 32;
                if (dd < d) {
                    qk += bf162f(Q_bh[(int64_t)i * d + dd]) * bf162f(K_bh[(int64_t)j * d + dd]);
                    dov += bf162f(dO_bh[(int64_t)i * d + dd]) * bf162f(V_bh[(int64_t)j * d + dd]);
                }
            }
            
            float p = expf(scale * qk - L_i);
            acc += p * dov;
        }
        
        d_delta[i] = acc;
    }
    __syncthreads();
    
    // ============================================================
    // Phase 2: Compute dV[j*d + di] = sum_i P[i][j] * dO[i][di]
    // Each thread owns specific (j, di) pairs
    // ============================================================
    for (int out_idx = warp_lane_in_block; out_idx < S * d; out_idx += NUM_THREADS) {
        int j = out_idx / d;
        int di = out_idx % d;
        float acc = 0.0f;
        
        for (int i = 0; i < S; ++i) {
            float qk = 0.0f;
            for (int cc = 0; cc < 4; ++cc) {
                int dd = lane + cc * 32;
                if (dd < d) {
                    qk += bf162f(Q_bh[(int64_t)i * d + dd]) * bf162f(K_bh[(int64_t)j * d + dd]);
                }
            }
            float p = expf(scale * qk - L_bh[i]);
            acc += p * bf162f(dO_bh[(int64_t)i * d + di]);
        }
        dV_bh[out_idx] = f2bf16(acc);
    }
    __syncthreads();
    
    // ============================================================
    // Phase 3: Compute dQ[i*d + di] = scale * sum_j P*(dov-delta)*K[j][di]
    // ============================================================
    for (int out_idx = warp_lane_in_block; out_idx < S * d; out_idx += NUM_THREADS) {
        int i = out_idx / d;
        int di = out_idx % d;
        float acc = 0.0f;
        float L_i = L_bh[i];
        float delta_i = d_delta[i];
        
        for (int j = 0; j < S; ++j) {
            float qk = 0.0f;
            float dov = 0.0f;
            
            for (int cc = 0; cc < 4; ++cc) {
                int dd = lane + cc * 32;
                if (dd < d) {
                    qk += bf162f(Q_bh[(int64_t)i * d + dd]) * bf162f(K_bh[(int64_t)j * d + dd]);
                    dov += bf162f(dO_bh[(int64_t)i * d + dd]) * bf162f(V_bh[(int64_t)j * d + dd]);
                }
            }
            
            float p = expf(scale * qk - L_i);
            float ds = p * (dov - delta_i);
            acc += ds * bf162f(K_bh[(int64_t)j * d + di]) * scale;
        }
        dQ_bh[out_idx] = f2bf16(acc);
    }
    __syncthreads();
    
    // ============================================================
    // Phase 4: Compute dK[j*d + dj] = scale * sum_i P*(dov-delta)*Q[i][dj]
    // ============================================================
    for (int out_idx = warp_lane_in_block; out_idx < S * d; out_idx += NUM_THREADS) {
        int j = out_idx / d;
        int dj = out_idx % d;
        float acc = 0.0f;
        
        for (int i = 0; i < S; ++i) {
            float L_i = L_bh[i];
            float delta_i = d_delta[i];
            float qk = 0.0f;
            float dov = 0.0f;
            
            for (int cc = 0; cc < 4; ++cc) {
                int dd = lane + cc * 32;
                if (dd < d) {
                    qk += bf162f(Q_bh[(int64_t)i * d + dd]) * bf162f(K_bh[(int64_t)j * d + dd]);
                    dov += bf162f(dO_bh[(int64_t)i * d + dd]) * bf162f(V_bh[(int64_t)j * d + dd]);
                }
            }
            
            float p = expf(scale * qk - L_i);
            float ds = p * (dov - delta_i);
            acc += ds * bf162f(Q_bh[(int64_t)i * d + dj]) * scale;
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
    
    // Allocate delta workspace: B*H*S floats
    tvm::ffi::TensorView d_delta = tvm::ffi::TensorView::Empty(
        {B, H, S}, tvm::ffi::DLDataType{kDLFloat, 32, 1}, Q.device());
    
    dim3 grid(bh_total);
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    
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
        static_cast<float*>(d_delta.data_ptr()),
        static_cast<int>(B), static_cast<int>(H), static_cast<int>(S),
        static_cast<int>(d), scale
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_impl::run);

} // namespace mha_bwd_impl