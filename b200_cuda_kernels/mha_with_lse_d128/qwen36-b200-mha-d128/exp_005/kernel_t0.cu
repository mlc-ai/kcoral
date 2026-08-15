#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define BLOCK_M 64
#define BLOCK_N 64
#define THREADS 256

namespace mha_cuda {

__global__ void flash_attn_fwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S, int D,
    int stride_b, int stride_h, int stride_s
) {
    int bx = blockIdx.x;
    int num_tiles_m = (S + BLOCK_M - 1) / BLOCK_M;
    int total_tiles = H * num_tiles_m;
    
    int batch_idx = bx / total_tiles;
    int rem = bx % total_tiles;
    int head_idx = rem / num_tiles_m;
    int tile_m = rem % num_tiles_m;

    if (batch_idx >= B || head_idx >= H || tile_m >= num_tiles_m) return;

    const __nv_bfloat16* q_ptr = Q + batch_idx * stride_b + head_idx * stride_h + tile_m * BLOCK_M * stride_s;
    const __nv_bfloat16* k_base = K + batch_idx * stride_b + head_idx * stride_h;
    const __nv_bfloat16* v_base = V + batch_idx * stride_b + head_idx * stride_h;
    __nv_bfloat16* o_ptr = O + batch_idx * stride_b + head_idx * stride_h + tile_m * BLOCK_M * stride_s;
    float* lse_ptr = LSE + batch_idx * (H * S) + head_idx * S + tile_m * BLOCK_M;

    extern __shared__ char smem[];
    __nv_bfloat16* sK = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sV = sK + BLOCK_N * D;

    // Register files: each thread handles 4 rows of Q/O (256 threads / 64 rows = 4)
    float o_regs[4][D];
    #pragma unroll
    for(int r=0; r<4; ++r)
        for(int d=0; d<D; ++d) o_regs[r][d] = 0.0f;

    float row_max[4] = {-CUDART_INF_F};
    float row_sum[4] = {0.0f};
    bool row_mask[4];
    #pragma unroll
    for(int r=0; r<4; ++r) row_mask[r] = (tile_m * BLOCK_M + threadIdx.x * 4 + r) < S;

    // Load Q into registers
    float q_regs[4][D];
    #pragma unroll
    for(int r=0; r<4; ++r) {
        int idx = threadIdx.x * 4 + r;
        if(row_mask[r]) {
            const __nv_bfloat16* p = q_ptr + idx * stride_s;
            #pragma unroll
            for(int d=0; d<D; ++d) q_regs[r][d] = __bfloat162float(p[d]);
        } else {
            #pragma unroll
            for(int d=0; d<D; ++d) q_regs[r][d] = 0.0f;
        }
    }

    float scale = rsqrtf(static_cast<float>(D));

    // Iterate over KV sequence in tiles
    for(int n_start = 0; n_start < S; n_start += BLOCK_N) {
        // Load K and V to shared memory
        int tid = threadIdx.x;
        #pragma unroll
        for(int i=tid; i<BLOCK_N*D; i+=THREADS) {
            int r = i / D;
            int c = i % D;
            bool valid = (n_start + r) < S;
            sK[i] = valid ? k_base[(n_start+r)*stride_s + c] : __float2bfloat16(0.0f);
            sV[i] = valid ? v_base[(n_start+r)*stride_s + c] : __float2bfloat16(0.0f);
        }
        __syncthreads();

        // Compute attention scores S = Q @ K^T for this thread's 4 rows
        float s_tile[4][BLOCK_N];
        #pragma unroll
        for(int r=0; r<4; ++r) {
            if(!row_mask[r]) {
                #pragma unroll
                for(int k_r=0; k_r<BLOCK_N; ++k_r) s_tile[r][k_r] = -CUDART_INF_F;
                continue;
            }
            #pragma unroll
            for(int k_r=0; k_r<BLOCK_N; ++k_r) {
                if(n_start + k_r >= S) { s_tile[r][k_r] = -CUDART_INF_F; continue; }
                float ds = 0.0f;
                #pragma unroll
                for(int d=0; d<D; d+=4) {
                    ds += q_regs[r][d]   * __bfloat162float(sK[k_r*D+d]);
                    ds += q_regs[r][d+1] * __bfloat162float(sK[k_r*D+d+1]);
                    ds += q_regs[r][d+2] * __bfloat162float(sK[k_r*D+d+2]);
                    ds += q_regs[r][d+3] * __bfloat162float(sK[k_r*D+d+3]);
                }
                s_tile[r][k_r] = ds * scale;
            }
        }

        // Online softmax update and O accumulation
        #pragma unroll
        for(int r=0; r<4; ++r) {
            if(!row_mask[r]) continue;
            
            float new_max = -CUDART_INF_F;
            #pragma unroll
            for(int k_r=0; k_r<BLOCK_N; ++k_r)
                if(s_tile[r][k_r] > new_max) new_max = s_tile[r][k_r];
            
            new_max = fmaxf(row_max[r], new_max);
            float alpha = expf(row_max[r] - new_max);
            row_sum[r] *= alpha;

            float p_sum = 0.0f;
            #pragma unroll
            for(int k_r=0; k_r<BLOCK_N; ++k_r) {
                float p = expf(s_tile[r][k_r] - new_max);
                p_sum += p;
                
                // O = alpha * O_old + p * V_row
                #pragma unroll
                for(int d=0; d<D; d+=2) {
                    o_regs[r][d]   = alpha * o_regs[r][d]   + p * __bfloat162float(sV[k_r*D+d]);
                    o_regs[r][d+1] = alpha * o_regs[r][d+1] + p * __bfloat162float(sV[k_r*D+d+1]);
                }
            }
            row_sum[r] += p_sum;
            row_max[r] = new_max;
        }
        __syncthreads();
    }

    // Final normalization and store
    #pragma unroll
    for(int r=0; r<4; ++r) {
        int idx = threadIdx.x * 4 + r;
        if(row_mask[r]) {
            float norm = (row_sum[r] == 0.0f) ? 0.0f : (1.0f / row_sum[r]);
            #pragma unroll
            for(int d=0; d<D; ++d) {
                o_ptr[idx * stride_s + d] = __float2bfloat16(o_regs[r][d] * norm);
            }
            lse_ptr[idx] = row_max[r] + logf(row_sum[r]);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int B = 4, H = 48, D = 128;
    int S = Q.size(2);
    
    // Contiguous strides for [B, H, S, D]
    int stride_d = 1;
    int stride_s = D;
    int stride_h = S * D;
    int stride_b = H * S * D;

    int num_tiles_m = (S + BLOCK_M - 1) / BLOCK_M;
    dim3 grid(B * H * num_tiles_m);
    dim3 block(THREADS);
    size_t smem_size = 2ULL * BLOCK_N * D * sizeof(__nv_bfloat16);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    flash_attn_fwd_kernel<<<grid, block, smem_size, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<__nv_bfloat16*>(O.data_ptr()),
        static_cast<float*>(LSE.data_ptr()),
        B, H, S, D, stride_b, stride_h, stride_s
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_cuda::run);

} // namespace mha_cuda