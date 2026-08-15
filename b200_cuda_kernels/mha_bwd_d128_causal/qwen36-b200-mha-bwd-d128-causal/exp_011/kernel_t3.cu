#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_bwd_d128_causal {

constexpr int HEAD_DIM = 128;
constexpr int WARP_SIZE = 32;
constexpr int BLOCK_THREADS = 128;

// Tile sizes: small to reduce register pressure
constexpr int TILE_M = 16;  // Query tile size
constexpr int TILE_N = 16;  // Key tile size
constexpr int THREADS_PER_TILE = TILE_M * TILE_N / WARP_SIZE * WARP_SIZE;  // 8 warps needed for 256-thread cooperation

// Actually let's use 128 threads where each handles one (q_local, n_local) for a small sub-tile
// With TILE_M=32, TILE_N=32, we need 1024 positions. 128 threads => each handles 8 positions.
// But that's complex. Simpler: use 128 threads, each handles 1 query row x partial d-dim work.

constexpr int BLOCK_TILES_M = 128;   // Number of query rows per CTA
constexpr int BLOCK_TILES_D = 128;   // Head dimension processed per CTA (=HEAD_DIM)

constexpr float INV_SQRT_D = 0.08838834764831844f;
constexpr float LOG2E = 1.4426950408889634f;

__device__ __forceinline__ float bf162f(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

__device__ __forceinline__ __nv_bfloat16 f2bf16(float val) {
    return __float2bfloat16(val);
}

__device__ __forceinline__ float fast_exp2(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void mha_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int B, int H, int S, int D
) {
    static_assert(HEAD_DIM == 128 && D == HEAD_DIM);
    
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    
    int bh_idx = blockIdx.x;
    if (bh_idx >= B * H) return;
    
    int b = bh_idx / H;
    int h = bh_idx % H;
    
    const __nv_bfloat16* Q_bh = Q + (uint64_t)(b * H + h) * S * D;
    const __nv_bfloat16* K_bh = K + (uint64_t)(b * H + h) * S * D;
    const __nv_bfloat16* V_bh = V + (uint64_t)(b * H + h) * S * D;
    const __nv_bfloat16* dO_bh = dO + (uint64_t)(b * H + h) * S * D;
    const float* L_bh = L + (b * H + h) * S;
    
    __nv_bfloat16* dQ_bh = dQ + (uint64_t)(b * H + h) * S * D;
    __nv_bfloat16* dK_bh = dK + (uint64_t)(b * H + h) * S * D;
    __nv_bfloat16* dV_bh = dV + (uint64_t)(b * H + h) * S * D;
    
    // Shared memory: one Q block, one K/V block at a time
    // sQ_block: 128 x 128 bf16 = 32KB for full query block
    // sKV_block: 32 x 128 bf16 = 8KB for one KV tile
    extern __shared__ char smem[];
    
    // Layout: sQ (128x128 bf16) = 32KB, sKV (32x128 bf16) = 8KB = 40KB total
    __nv_bfloat16* sQ_all = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sKV = sQ_all + BLOCK_TILES_M * D;
    
    // Load entire Q block into shared memory cooperatively
    // Each thread loads 128 / 128 = 1 element per row, iterates over 128 rows
    // Or: each thread handles one row with vectorized loads
    for (int qi = warp_id; qi < BLOCK_TILES_M; qi += 4) {
        uint64_t off = (uint64_t)qi * D;
        for (int di = lane_id; di < D; di += WARP_SIZE) {
            sQ_all[off + di] = (qi < S) ? Q_bh[off + di] : f2bf16(0.0f);
        }
    }
    __syncthreads();
    
    // Initialize dQ accumulators per query row (shared memory)
    float* dQ_smem = reinterpret_cast<float*>(sKV);  // After sKV: 32x128 floats = 16KB
    // Actually we need 128x128 floats = 64KB which exceeds budget
    // Instead: accumulate dQ in registers for our query rows, write periodically
    
    // Strategy: process N in tiles of size TN=32
    // For each (q_row, n_tile): compute attention contribution
    // Use local registers for small pieces, shared memory for larger accumulations
    
    constexpr int TN = 32;  // N tile size
    int num_n_tiles = (S + TN - 1) / TN;
    
    // Each thread processes one query row throughout
    // 128 threads = 128 query rows (perfect match for BLOCK_TILES_M=128)
    int my_qrow = tid;  // 0..127
    if (my_qrow >= S) return;
    
    float my_lse = L_bh[my_qrow];
    
    // Load dO for this query row into registers (each thread gets its own)
    // 128 floats = manageable if accessed sparsely
    // To save registers, load into shared memory first
    float* dO_smem = dQ_smem + BLOCK_TILES_M;  // 128 more floats per row... too much
    
    // Alternative: reload dO elements as needed from global memory
    // Or use shared memory pool that rotates
    
    // Best approach: pre-load dO for all 128 query rows into shared mem
    // That's 128 x 128 floats = 64KB... combined with sQ (32KB) and sKV (8KB) = 104KB total
    // Exceeds typical 128KB limit when considering other uses
    
    // Compromise: use a double-buffered approach and stream
    // For correctness first, let's just accept higher SMEM usage
    
    // Final layout plan:
    // sQ: 128 x 128 bf16 = 32KB at offset 0
    // dO_smem: 128 x 128 floats would be 64KB - skip, reload from global
    // sKV: 32 x 128 bf16 = 8KB at offset 32*128 = 4096 bf16 words
    
    // Accumulate dQ results per row in registers (just 128 floats -> store 4 at a time, accumulate 32 groups)
    // Store partial dQ in shared memory: 128 rows x 128 cols floats = 64KB
    // Total: 32KB + 8KB + 64KB = 104KB. With 128KB limit (minus 1KB reservation = 127KB), this fits tightly
    
    // Simplification: don't accumulate dQ in shared mem, write directly to global dQ
    // For dK/dV, use shared memory accumulation per tile then atomicAdd to global
    
    // Reload dO elements on-the-fly to save shared memory
    
    // Main loop over N tiles
    for (int nt = 0; nt < num_n_tiles; nt++) {
        int nk_start = nt * TN;
        if (nk_start >= S) break;
        
        // Load K[V tile] into shared memory sKV[ni * D + di]
        for (int ni = 0; ni < TN; ni++) {
            int nk = nk_start + ni;
            for (int di = lane_id; di < D; di += WARP_SIZE) {
                if (nk < S) {
                    sKV[ni * D + di] = K_bh[(uint64_t)nk * D + di];
                } else {
                    sKV[ni * D + di] = f2bf16(0.0f);
                }
            }
        }
        __syncthreads();
        
        // Load corresponding V tile (overwrite sKV after K is consumed)
        // Actually we need both K and V simultaneously, so let's double-size sKV
        // sKV_K = sKV, sKV_V = sKV + TN*D
        __nv_bfloat16* sV_tile = sKV + TN * D;
        
        for (int ni = 0; ni < TN; ni++) {
            int nk = nk_start + ni;
            for (int di = lane_id; di < D; di += WARP_SIZE) {
                if (nk < S) {
                    sV_tile[ni * D + di] = V_bh[(uint64_t)nk * D + di];
                } else {
                    sV_tile[ni * D + di] = f2bf16(0.0f);
                }
            }
        }
        __syncthreads();
        
        // Now compute for my_qrow against this NK tile
        // Compute attention scores: score[i] = Q[my_qrow,:] . K[nk_start+i,:] * inv_sqrt_d
        // Then probs[i] = exp(score[i] - lse) if nk <= my_qrow
        
        // Compute dP_scalar[i] = dO[my_qrow,:] . V[nk_start+i,:]
        // These can be done together in one pass
        
        float ds_vals[TN];       // dS for each n-position
        float dq_partial[D];     // Partial dQ accumulation for this tile
        
        // Initialize dq_partial
        #pragma unroll
        for (int i = 0; i < D; i++) dq_partial[i] = 0.0f;
        
        float dP_dot_sum = 0.0f;  // sum_i(P_i * dP_scalar_i)
        
        #pragma unroll
        for (int ni = 0; ni < TN; ni++) {
            int nk = nk_start + ni;
            
            // Compute dot product Q[my_qrow,:] . K[nk,:]
            float score = 0.0f;
            #pragma unroll
            for (int di = 0; di < D; di++) {
                score += bf162f(sQ_all[(uint64_t)my_qrow * D + di]) * bf162f(sKV[ni * D + di]);
            }
            score *= INV_SQRT_D;
            
            // Compute dot product dO[my_qrow,:] . V[nk,:]
            float dP_scalar = 0.0f;
            for (int di = lane_id; di < D; di += WARP_SIZE) {
                dP_scalar += bf162f(dO_bh[(uint64_t)my_qrow * D + di]) * bf162f(sV_tile[ni * D + di]);
            }
            // Warp reduce
            dP_scalar = warp_reduce_sum(dP_scalar);
            
            // Check causality and compute probability
            float prob = (nk <= my_qrow) ? fast_exp2((score - my_lse) * LOG2E) : 0.0f;
            
            // dS = prob * (dP_scalar - dP_dot_sum) ... but dP_dot_sum not ready yet
            // Store intermediates, compute dP_dot_sum first pass, then dS second pass
            ds_vals[ni] = prob;  // temporarily store prob
            
            if (prob > 0.0f) {
                dP_dot_sum += prob * dP_scalar;
                
                // Accumulate dV: dV[nk, :] += prob * dO[my_qrow, :]
                // Use atomic add to shared memory buffer, then flush later
                // For now, defer dV computation
                
                // Accumulate dQ partial: dQ += ds * K[nk, :]
                // Need ds = prob * (dP_scalar - dP_dot_sum), but we haven't finished dP_dot_sum
                // Defer this too
            }
            
            // Store dP_scalar for second pass
            // Reuse ds_vals to store both prob and dP_scalar
            // Actually allocate separate storage
        }
        
        // Second pass: compute final dS and accumulate dQ, dK, dV
        // Need to store dP_scalar values from first pass
        // Rethink: combine into single pass
        
        // SINGLE PASS REWRITE:
        // First accumulate dP_dot_sum, store scores/dP_scalar, then compute in second pass
        
        float stored_prob[TN];
        float stored_dP_scalar[TN];
        
        dP_dot_sum = 0.0f;
        #pragma unroll
        for (int ni = 0; ni < TN; ni++) {
            int nk = nk_start + ni;
            float score = 0.0f;
            #pragma unroll
            for (int di = 0; di < D; di++) {
                score += bf162f(sQ_all[(uint64_t)my_qrow * D + di]) * bf162f(sKV[ni * D + di]);
            }
            score *= INV_SQRT_D;
            
            float dP_scalar = 0.0f;
            for (int di = lane_id; di < D; di += WARP_SIZE) {
                dP_scalar += bf162f(dO_bh[(uint64_t)my_qrow * D + di]) * bf162f(sV_tile[ni * D + di]);
            }
            dP_scalar = warp_reduce_sum(dP_scalar);
            
            float prob = (nk <= my_qrow) ? fast_exp2((score - my_lse) * LOG2E) : 0.0f;
            stored_prob[ni] = prob;
            stored_dP_scalar[ni] = dP_scalar;
            
            if (prob > 0.0f) {
                dP_dot_sum += prob * dP_scalar;
            }
        }
        
        // Now compute outputs
        #pragma unroll
        for (int ni = 0; ni < TN; ni++) {
            int nk = nk_start + ni;
            float prob = stored_prob[ni];
            if (nk > my_qrow || prob == 0.0f) continue;
            
            float dP_scalar = stored_dP_scalar[ni];
            float ds = prob * (dP_scalar - dP_dot_sum);
            
            // dQ[my_qrow, d] += ds * K[nk, d]
            #pragma unroll
            for (int di = 0; di < D; di++) {
                dq_partial[di] += ds * bf162f(sKV[ni * D + di]);
            }
            
            // dK[nk, d] += ds * Q[my_qrow, d] -- atomic to global
            // dV[nk, d] += prob * dO[my_qrow, d] -- atomic to global
            for (int di = lane_id; di < D; di += WARP_SIZE) {
                float dk_val = ds * bf162f(sQ_all[(uint64_t)my_qrow * D + di]);
                float dv_val = prob * bf162f(dO_bh[(uint64_t)my_qrow * D + di]);
                
                // Use FP32 temporary buffers, convert at end
                // For now use direct writes with atomics on float-reinterpreted pointers
                // This works because BF16 is 2 bytes aligned and atomicAdd<float> needs 4-byte alignment
                // dK_bh and dV_bh are nv_bfloat16*, not aligned for float atomics
                
                // Solution: cast through a separate fp32 workspace or use __int2float_rn trick
                // Simplest: store to shared FP32 buffer, reduce, convert, write once
            }
        }
        
        // Write dQ partials to global
        for (int di = lane_id; di < D; di += WARP_SIZE) {
            dQ_bh[(uint64_t)my_qrow * D + di] = f2bf16(dq_partial[di]);
        }
        
        // For dK/dV, accumulate in shared FP32 memory, then write
        // Need TN * D fp32 entries each = 32*128*4*2 = 32KB extra... too much with existing layout
        
        // Simpler: use global atomics via casting to float* with proper alignment
        // dK_bh and dV_bh ARE aligned to 256B typically (CUDAMalloc default), which satisfies 4B alignment
        // The issue before was pointer arithmetic. Let's fix the cast properly.
        
        // Actually the real fix: cast dK_bh[d_offset] correctly
        // nv_bfloat16* -> reinterpret as float* with half the stride
        // dK_bh[k*D + di] is at byte offset (k*D+di)*2
        // As float*: reinterpret_cast<float*>(dK_bh)[(k*D+di)/2] but only if even index
        // Better: just use the raw byte pointer and add appropriate offsets
        
        float* dK_fptr = reinterpret_cast<float*>(dK_bh);
        float* dV_fptr = reinterpret_cast<float*>(dV_bh);
        
        // dK_bh[k*D + di] -> byte offset (k*D+di)*2 -> as float ptr: base + (k*D+di)*2
        // If we start from aligned base, float reinterpret gives us pairs of bf16
        
        // Easier solution: just reset and write at end. Use shared mem accumulation per block.
        __syncthreads();
    }
    
    // FINAL APPROACH: Two-phase kernel. Phase 1 computes and reduces dK/dV in shared memory.
    // Phase 2 writes everything to global.
    // But this doubles the work. Let me just write a simpler but correct kernel.
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_ptr = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_ptr = static_cast<const float*>(L.data_ptr());
    
    __nv_bfloat16* dQ_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    // Launch one block per (batch, head) pair
    dim3 grid(B * H, 1, 1);
    dim3 block(BLOCK_THREADS);
    
    size_t smem_bytes = (BLOCK_TILES_M * D + 2 * TN * D) * sizeof(__nv_bfloat16);
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    mha_bwd_kernel<<<grid, block, smem_bytes, stream>>>(
        Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr,
        dQ_ptr, dK_ptr, dV_ptr,
        (int)B, (int)H, (int)S, (int)D);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128_causal::run);

}  // namespace mha_bwd_d128_causal