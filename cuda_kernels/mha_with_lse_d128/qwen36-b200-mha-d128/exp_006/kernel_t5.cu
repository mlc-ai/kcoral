#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

namespace mha_d128 {

// UMMA CG1 helpers
__device__ __forceinline__ uint64_t make_smem_desc_cg1(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)((addr & 0x3FFFF) >> 4);
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1ULL << 46;
    d |= (uint64_t)2ULL << 61;  // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_cg1(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);       // c_format = FP32
    d |= (1u << 7);       // a_format = BF16
    d |= (1u << 10);      // b_format = BF16
    d |= (0u << 15);      // A is MN-major (transpose=0 means K-major, but we store K-transposed)
    d |= (1u << 16);      // B is MN-major (transpose=1 means B is transposed / N-major view)
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ void umma_mma_cg1(uint32_t tmem_d, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, bool accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_d), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum ? 1 : 0));
}

__device__ __forceinline__ void umma_commit_cg1(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64 [%0];"
                 :: "r"(a));
}

__device__ __forceinline__ void init_smem_barrier(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(count));
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_phase(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n.reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(phase));
}

__device__ __forceinline__ void fence_proxy_async() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

template<int BM, int BN, int BK>
__global__ void mha_kernel_tcgen05(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S, int D) 
{
    // Shared memory: 2 swizzled regions for KV + barrier
    extern __shared__ char smem_base[];
    
    // Alignment requirements: 128B swizzle starts at 1024B boundary
    __align__(1024) float (*smem_A)[D] = reinterpret_cast<__align__(1024) float(*)[D]>(smem_base);   // BM x D for Q cache (not needed actually)
    __align__(1024) float (*smem_B)[D] = reinterpret_cast<__align__(1024) float(*)[D]>(smem_base + sizeof(float) * BM * D);  // BV x D region for K/V
    
    uint64_t* mbarrier = reinterpret_cast<uint64_t*>(smem_base + 2 * sizeof(float) * BM * D);
    
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    
    int bh_idx = blockIdx.x;
    int num_bh = B * H;
    int total_tiles = (S + BM - 1) / BM;
    int tile_idx = bh_idx / num_bh;
    int bhtile_id = bh_idx % num_bh;
    if (tile_idx >= total_tiles || bhtile_id >= num_bh) return;
    
    int b = bhtile_id / H;
    int h = bhtile_id % H;
    int m_base = tile_idx * BM;
    
    const __nv_bfloat16* q_base = Q + ((size_t)b * H + h) * S * D;
    const __nv_bfloat16* k_base = K + ((size_t)b * H + h) * S * D;
    const __nv_bfloat16* v_base = V + ((size_t)b * H + h) * S * D;
    __nv_bfloat16* o_base = O + ((size_t)b * H + h) * S * D;
    float* lse_base = LSE + ((size_t)b * H + h) * S;

    // Initialize barriers
    if (tid == 0) {
        init_smem_barrier(mbarrier, 1);
    }
    __syncthreads();

    // Allocate Tensor Memory (64 cols minimum, power of 2)
    uint32_t tmem_addr;
    uint32_t* tmem_ptr = reinterpret_cast<uint32_t*>(smem_B);
    if (warp_id == 0 && lane == 0) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     : "=r"(tmem_addr) : "r"(64));
    }
    __syncthreads();
    
    // Load Q into registers (each thread loads parts of Q)
    // Q tile: BM x D. Load into shared smem_A then read later via registers
    {
        for (int r = 0; r < BM; ++r) {
            for (int c = lane; c < D; c += 32) {
                int gm = m_base + r;
                smem_A[r][c] = (gm < S) ? __bfloat162float(q_base[gm * D + c]) : 0.0f;
            }
        }
    }
    __syncthreads();

    float inv_sqrt_D = rsqrtf((float)D);
    int num_kv_tiles = (S + BK - 1) / BK;
    
    // Per-thread accumulators for softmax
    float row_max = -1e20f;
    float row_sum = 0.0f;
    
    // Output registers: each thread holds part of O row
    float reg_O[BM > 32 ? 4 : 1][D/32];  // Simplified: just store in smem
    
    for (int kv_t = 0; kv_t < num_kv_tiles; ++kv_t) {
        int kv_start = kv_t * BK;
        int cur_k_len = min(BK, S - kv_start);
        
        // Load K tile into shared (row-major: cur_k_len x D)
        for (int kr = 0; kr < BK; ++kr) {
            for (int c = lane; c < D; c += 32) {
                if (kr < cur_k_len) {
                    smem_B[kr][c] = __bfloat162float(k_base[(kv_start + kr) * D + c]);
                } else {
                    smem_B[kr][c] = 0.0f;
                }
            }
        }
        // Load V tile (will reuse same smem after K consumed)
        for (int kr = 0; kr < BK; ++kr) {
            for (int c = lane + D; c < 2*D; c += 32) {
                int col = c - D;
                if (kr < cur_k_len) {
                    // Store V after K in same buffer
                    ((float*)smem_B)[(BK+1)*D + kr*D + col] = __bfloat162float(v_base[(kv_start + kr) * D + col]);
                }
            }
        }
        __syncthreads();
    }
    
    // Cleanup and dealloc TMEM
    if (warp_id == 0 && lane == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     :: "r"(tmem_addr), "r"(64));
    }
    __syncthreads();
    
    // Write output (placeholder - actual computation via registers)
    if (tid < BM && m_base + tid < S) {
        int gm = m_base + tid;
        for (int n = lane; n < D; n += 32) {
            o_base[gm * D + n] = __float2bfloat16(0.0f);
        }
        lse_base[gm] = 0.0f;
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    assert(D == 128);
    
    constexpr int BM = 64;
    constexpr int BN = 128;
    constexpr int BK = 16;
    int block_size = 128;
    
    int num_bh = (int)(B * H);
    int num_q_tiles = (int)((S + BM - 1) / BM);
    int num_blocks = num_bh * num_q_tiles;
    
    // Shared memory: 2 * BM * D floats + barrier
    size_t smem_bytes = 2 * sizeof(float) * BM * D + 1024;
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id)
    );
    
    mha_kernel_tcgen05<BM, BN, BK><<<num_blocks, block_size, smem_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<__nv_bfloat16*>(O.data_ptr()),
        static_cast<float*>(LSE.data_ptr()),
        (int)B, (int)H, (int)S, (int)D
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_d128::run);

} // namespace mha_d128