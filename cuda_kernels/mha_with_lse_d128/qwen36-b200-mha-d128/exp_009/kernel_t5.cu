#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
#include <cfloat>
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

namespace mha_impl {

// Tiled Multi-Head Attention Kernel
// Uses shared memory tiling for Q, K, V and FP32 accumulators.
// Parallelizes across D dimension using threads per row to avoid register spilling.
template <int D, int TILE_M, int TILE_N, int TPB>
__global__ void mha_forward_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE_out,
    int B, int H, int S,
    float inv_sqrt_d) 
{
    constexpr int TPR = TPB / TILE_M; // threads per query row
    static_assert(TPB % TILE_M == 0 && D % TPR == 0);
    
    extern __shared__ char smem[];
    __nv_bfloat16* Q_s = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* K_s = Q_s + TILE_M * D;
    __nv_bfloat16* V_s = K_s + TILE_N * D;
    float* Acc_s = reinterpret_cast<float*>(V_s + TILE_N * D);

    int b         = blockIdx.z;
    int h         = blockIdx.y;
    int q_start   = blockIdx.x * TILE_M;
    int tid       = threadIdx.x;
    int row       = tid / TPR;
    int lane      = tid % TPR;
    
    if (row >= TILE_M) return;
    
    int base_idx = (b * H + h) * S * D;
    int qs       = q_start + row;
    if (qs >= S) return;

    // Pointers
    const __nv_bfloat16* Q_row = Q + base_idx + qs * D;
    const __nv_bfloat16* K_base = K + base_idx;
    const __nv_bfloat16* V_base = V + base_idx;
    __nv_bfloat16* O_row = O + base_idx + qs * D;

    // Cooperative load of Q tile
    int my_d = lane * (D / TPR);
    const uint4* q_src = reinterpret_cast<const uint4*>(Q_row + my_d);
    uint4* q_dst = reinterpret_cast<uint4*>(Q_s + row * D + my_d);
    int chunks = (D / TPR) / 8;
    #pragma unroll
    for (int i = 0; i < chunks; ++i) q_dst[i] = q_src[i];
    __syncthreads();

    // Per-thread running softmax state & accumulator pointer
    float row_max = -FLT_MAX;
    float row_sum = (lane == 0) ? 0.0f : 0.0f;
    float* acc_ptr = Acc_s + row * D + my_d;
    
    // Zero-initialize accumulator chunk
    #pragma unroll
    for (int d = 0; d < D / TPR; ++d) acc_ptr[d] = 0.0f;
    __syncthreads();

    // Iterate over KV sequence in tiles
    constexpr int k_step = TPB / D; // How many KV rows we can load per sync
    for (int n_start = 0; n_start < S; n_start += TILE_N) {
        int n_actual = min(TILE_N, S - n_start);
        
        // Cooperative load of K & V tiles (pipelined by k_step rows)
        for (int k = 0; k < n_actual; k += k_step) {
            int k_end = min(k + k_step, n_actual);
            for (int kr = k; kr < k_end; ++kr) {
                int src_idx = n_start + kr;
                const __nv_bfloat16* k_src = K_base + src_idx * D;
                const __nv_bfloat16* v_src = V_base + src_idx * D;
                __nv_bfloat16* k_dst = K_s + kr * D;
                __nv_bfloat16* v_dst = V_s + kr * D;
                for (int i = threadIdx.x; i < D; i += TPB) {
                    k_dst[i] = k_src[i];
                    v_dst[i] = v_src[i];
                }
            }
            __syncthreads();
            
            // Compute attention scores and accumulate weighted V
            const __nv_bfloat16* q_local = Q_s + row * D + my_d;
            int d_chunk = D / TPR;
            for (int kr = k; kr < k_end; ++kr) {
                const __nv_bfloat16* k_local = K_s + kr * D + my_d;
                const __nv_bfloat16* v_local = V_s + kr * D + my_d;
                
                float score = 0.0f;
                #pragma unroll
                for (int d = 0; d < d_chunk; ++d) {
                    score += static_cast<float>(q_local[d]) * static_cast<float>(k_local[d]);
                }
                score *= inv_sqrt_d;

                // Online softmax update (lock-free per chunk)
                bool updated = (score > row_max);
                float ratio = updated ? expf(row_max - score) : 1.0f;
                if (updated) row_max = score;
                
                #pragma unroll
                for (int d = 0; d < d_chunk; ++d) acc_ptr[d] *= ratio;
                if (lane == 0) row_sum *= ratio;

                float p = expf(score - row_max);
                if (lane == 0) row_sum += p;

                #pragma unroll
                for (int d = 0; d < d_chunk; ++d) {
                    acc_ptr[d] += p * static_cast<float>(v_local[d]);
                }
            }
            __syncthreads();
        }
    }

    // Final normalization & write-out
    __syncwarp();
    float rs = __shfl_sync(0xffffffff, (lane == 0) ? row_sum : 0.0f, 0);
    float inv_sum = (rs > 0.0f) ? (1.0f / rs) : 0.0f;
    
    int d_chunk = D / TPR;
    #pragma unroll
    for (int d = 0; d < d_chunk; ++d) {
        O_row[my_d + d] = __float2bfloat16(acc_ptr[d] * inv_sum);
    }
    
    if (lane == 0) {
        LSE_out[(b * H + h) * S + qs] = (rs > 0.0f) ? (row_max + logf(rs)) : HUGE_VALF;
    }
}

void run(tvm::ffi::TensorView Q_in, tvm::ffi::TensorView K_in, tvm::ffi::TensorView V_in,
         tvm::ffi::TensorView O_out, tvm::ffi::TensorView LSE_out) {
    CUDA_CHECK(cudaSetDevice(Q_in.device().device_id));
    
    int B = static_cast<int>(Q_in.size(0));
    int H = static_cast<int>(Q_in.size(1));
    int S = static_cast<int>(Q_in.size(2));
    int D = static_cast<int>(Q_in.size(3));
    
    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q_in.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K_in.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V_in.data_ptr());
    __nv_bfloat16* O_ptr       = static_cast<__nv_bfloat16*>(O_out.data_ptr());
    float* LSE_ptr             = static_cast<float*>(LSE_out.data_ptr());
    
    // Shape validation
    auto shape_check = [](auto& T, int eB, int eH, int eS, int eD) {
        if(T.size(0)!=eB||T.size(1)!=eH||T.size(2)!=eS||T.size(3)!=eD){fprintf(stderr,"Shape mismatch\n");exit(1);}
    };
    shape_check(K_in, B,H,S,D);
    shape_check(V_in, B,H,S,D);
    shape_check(O_out,B,H,S,D);
    if(LSE_out.size(0)!=B||LSE_out.size(1)!=H||LSE_out.size(2)!=S){fprintf(stderr,"LSE mismatch\n");exit(1);}
    
    float inv_sqrt_d = 1.0f / sqrtf(static_cast<float>(D));
    
    // Tuned parameters for D=128
    constexpr int TILE_M = 64;
    constexpr int TILE_N = 64;
    constexpr int TPB    = 256;
    
    int smem_bytes = (TILE_M + 2 * TILE_N) * D * sizeof(__nv_bfloat16) + TILE_M * D * sizeof(float);
    
    dim3 grid((S + TILE_M - 1) / TILE_M, H, B);
    dim3 block(TPB);
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q_in.device().device_type, Q_in.device().device_id));
        
    switch(D) {
        case 32:  mha_forward_kernel<32, 64, 64, 256><<<grid, block, smem_bytes, stream>>>(Q_ptr,K_ptr,V_ptr,O_ptr,LSE_ptr,B,H,S,inv_sqrt_d); break;
        case 64:  mha_forward_kernel<64, 64, 64, 256><<<grid, block, smem_bytes, stream>>>(Q_ptr,K_ptr,V_ptr,O_ptr,LSE_ptr,B,H,S,inv_sqrt_d); break;
        case 128: mha_forward_kernel<128,64, 64, 256><<<grid, block, smem_bytes, stream>>>(Q_ptr,K_ptr,V_ptr,O_ptr,LSE_ptr,B,H,S,inv_sqrt_d); break;
        default:  fprintf(stderr,"Unsupported head dim D=%d\n",D); exit(1);
    }
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_impl::run);

}  // namespace mha_impl