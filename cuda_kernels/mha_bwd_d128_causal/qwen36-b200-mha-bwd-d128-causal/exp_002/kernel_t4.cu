#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <algorithm>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

#define CUDA_CHECK(call) do {                                         \
    cudaError_t _e = (call);                                          \
    if (_e != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                  \
                cudaGetErrorString(_e), __FILE__, __LINE__);          \
        exit(1);                                                       \
    }                                                                  \
} while(0)

namespace mha_bwd_impl {

// Simple memset-like clear
__global__ void mha_clear_kernel(__nv_bfloat16* out, int64_t total) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total; i += stride) {
        out[i] = __float2bfloat16(0.f);
    }
}

// Kernel: Compute dQ[b,h,q,d].
// Grid strides over both (b,h,q) positions AND the d dimension.
// Each invocation computes a SINGLE float4 scalar -> one bf16 write via atomicAdd or direct store.
template<int D>
__global__ __launch_bounds__(512)
void dQ_kernel_v2(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ_out,
    int B, int H, int S)
{
    // Thread ID represents a unique (bh, q, d_idx) tuple
    int64_t total_elems = (int64_t)B * H * S * D;
    int64_t tid = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    
    float inv_sqrt_d = rsqrtf((float)D);
    
    for (int64_t elem = tid; elem < total_elems; elem += stride) {
        int64_t bhS = elem / D;
        int d = static_cast<int>(elem % D);
        
        int q = static_cast<int>(bhS % S);
        int bh = static_cast<int>(bhS / S);
        int b = bh / H;
        int h = bh % H;
        size_t base = ((size_t)b * H + h) * (size_t)S * D;
        
        float lse_q = L[(size_t)bh * S + q];
        const __nv_bfloat16* q_row = Q + base + (size_t)q * D;
        
        // Load this q's row element d into register
        float qd = __bfloat162float(q_row[d]);
        
        // Accumulate sum_k<=q P[q,k]*K[k,d]
        float sum = 0.f;
        for (int k = 0; k <= q; k++) {
            const __nv_bfloat16* k_row = K + base + (size_t)k * D;
            float kd = __bfloat162float(k_row[d]);
            
            // We need score for (q,k): dot(Q[q], K[k])
            // Since we already have Q[q,d] and K[k,d], we need the full dot product
            // This requires looping over all D... which is expensive per element.
            // Better: precompute scores in registers or shared mem.
            // For now, accept the cost but reduce other overhead.
        }
        
        // Write result (we'd accumulate into sum then write)
        dQ_out[elem] = __float2bfloat16(sum);
    }
}

// Better approach: tile-based kernel where each CTA loads a tile of K
// into shared memory, cooperatively processing multiple q positions.
// 
// Strategy: one CTA = one (b,h) pair. Within CTA:
//   - Threads load consecutive K rows into smem (tile of S tiles)
//   - Each thread owns multiple q values
//   - Process S_TILES chunks along S axis

__global__ __launch_bounds__(256)
void attention_bwd_dq(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const float* __restrict__ L_in,
    __nv_bfloat16* __restrict__ dQ_out,
    int B, int H, int S, int D)
{
    extern __shared__ __nv_bfloat16 smem[];
    
    int bh = blockIdx.x;
    if (bh >= B * H) return;
    int b = bh / H;
    int h = bh % H;
    size_t base = ((size_t)b * H + h) * (size_t)S * D;
    
    // Shared memory layout: smem contains tiled copy of K segment
    // Layout: [smem][D] for S_TILE rows (k dimension)
    constexpr int TILE_K = 128;  // Number of k rows in shared memory
    constexpr int TILE_Q = 32;   // Each thread handles TILE_Q consecutive q positions
    
    // Each block handles all q positions. 
    // num_tiles = ceil(S / TILE_K) 
    // For each tile, load TILE_K x D into smem, then each thread processes its q's against valid k's
    
    __nv_bfloat16* smem_k = smem;  // shape: [TILE_K][D]
    
    // Each thread will accumulate dQ for TILE_Q positions
    // To avoid storing TILE_Q*D = 32*128=4096 floats per thread (impossible!),
    // we write partial results back repeatedly.
    
    // First pass: initialize dQ outputs
    {
        for (int d_off = threadIdx.x; d_off < D; d_off += blockDim.x) {
            for (int q = 0; q < S; q++) {
                dQ_out[base + (size_t)q * D + d_off] = __float2bfloat16(0.f);
            }
        }
    }
    
    // Now iterate over k-tiles
    float inv_sqrt_d = rsqrtf((float)D);
    
    for (int ktile = 0; ktile * TILE_K < S; ktile += 1) {
        int k_start = ktile * TILE_K;
        int k_end = min(k_start + TILE_K, S);
        int k_count = k_end - k_start;
        
        // Load K tile into shared memory
        for (int i = threadIdx.x; i < k_count * D; i += blockDim.x) {
            int k_row = i / D;
            int d_col = i % D;
            smem_k[i] = K[base + (size_t)(k_start + k_row) * D + d_col];
        }
        __syncthreads();
        
        // Each thread processes certain q positions. 
        // q must satisfy: causal => k <= q => q >= k_start (for relevant k's)
        // Distribute q workload across threads
        
        int nthreads = blockDim.x;
        // Assign roughly S/nthreads q positions per thread
        // But we need to iterate over ALL k in the tile for each q
        
        // For efficiency: outer loop over q, inner over k
        // Each thread takes some q values
        for (int q = threadIdx.x; q < S; q += nthreads) {
            // Check if any k in this tile is valid (k <= q)
            int k_valid_start = max(k_start, 0);
            int k_valid_end = min(k_end, q + 1);  // k <= q
            if (k_valid_end <= k_valid_start) continue;
            
            const __nv_bfloat16* q_row = Q + base + (size_t)q * D;
            float lse_q = L_in[(size_t)bh * S + q];
            __nv_bfloat16* dq_dst = dQ_out + base + (size_t)q * D;
            
            // Load Q[q,:] into registers for repeated dot products
            // Use float4 loads to minimize instructions
            for (int k_local = k_valid_start - k_start; k_local < k_valid_end - k_start; k_local++) {
                // Score = Q[q] . K[k]
                float score = 0.f;
                const __nv_bfloat16* k_row_smem = smem_k + (size_t)k_local * D;
                
                #pragma unroll
                for (int di = 0; di < D; di += 4) {
                    float q0 = __bfloat162float(q_row[di]);
                    float q1 = __bfloat162float(q_row[di+1]);
                    float q2 = __bfloat162float(q_row[di+2]);
                    float q3 = __bfloat162float(q_row[di+3]);
                    float k0 = __bfloat162float(k_row_smem[di]);
                    float k1 = __bfloat162float(k_row_smem[di+1]);
                    float k2 = __bfloat162float(k_row_smem[di+2]);
                    float k3 = __bfloat162float(k_row_smem[di+3]);
                    score += q0*k0 + q1*k1 + q2*k2 + q3*k3;
                }
                
                float attn = expf(score * inv_sqrt_d - lse_q);
                
                // dQ[q,d] += attn * K[k,d]
                // Vectorized: write 4 bf16 at a time
                for (int di = 0; di < D; di += 4) {
                    float val0 = __bfloat162float(dq_dst[di]) + attn * __bfloat162float(k_row_smem[di]);
                    float val1 = __bfloat162float(dq_dst[di+1]) + attn * __bfloat162float(k_row_smem[di+1]);
                    float val2 = __bfloat162float(dq_dst[di+2]) + attn * __bfloat162float(k_row_smem[di+2]);
                    float val3 = __bfloat162float(dq_dst[di+3]) + attn * __bfloat162float(k_row_smem[di+3]);
                    dq_dst[di]   = __float2bfloat16(val0);
                    dq_dst[di+1] = __float2bfloat16(val1);
                    dq_dst[di+2] = __float2bfloat16(val2);
                    dq_dst[di+3] = __float2bfloat16(val3);
                }
            }
        }
    }
}


void run(tvm::ffi::TensorView Q,
         tvm::ffi::TensorView K,
         tvm::ffi::TensorView V,
         tvm::ffi::TensorView O,
         tvm::ffi::TensorView dO,
         tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ,
         tvm::ffi::TensorView dK,
         tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D_val = Q.size(3);
    assert(D_val == 128 && "Expected D=128");
    int D = static_cast<int>(D_val);
    int BH = static_cast<int>(B * H);
    
    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_ptr = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    // Clear outputs
    int64_t total_elem = B * H * S * D;
    int ct = 512;
    int cb = std::min(static_cast<int>((total_elem + ct - 1) / ct), 65535);
    mha_clear_kernel<<<cb, ct, 0, stream>>>(dQ_ptr, total_elem);
    CUDA_CHECK(cudaGetLastError());
    mha_clear_kernel<<<cb, ct, 0, stream>>>(dK_ptr, total_elem);
    CUDA_CHECK(cudaGetLastError());
    mha_clear_kernel<<<cb, ct, 0, stream>>>(dV_ptr, total_elem);
    CUDA_CHECK(cudaGetLastError());
    
    // Launch dQ kernel: 1 block per (b,h) pair
    // SMEM needed: TILE_K * D * sizeof(bf16) = 128 * 128 * 2 = 32KB
    constexpr int TILE_K = 128;
    int smem_bytes = TILE_K * D * sizeof(__nv_bfloat16);
    
    dim3 grid_q(BH, 1, 1);
    dim3 blk(256, 1, 1);
    attention_bwd_dq<<<grid_q, blk, smem_bytes, stream>>>(
        Q_ptr, K_ptr, L_ptr, dQ_ptr,
        static_cast<int>(B), static_cast<int>(H), static_cast<int>(S), D);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_impl::run);

}  // namespace mha_bwd_impl