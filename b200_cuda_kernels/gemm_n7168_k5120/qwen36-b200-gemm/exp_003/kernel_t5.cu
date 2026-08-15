#include <cuda_bf16.h>
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

// ---- Device helper functions ----

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase_parity) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase_parity));
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

// Build SM100 UMMA shared memory descriptor (no swizzle)
__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo_raw, uint32_t sbo_raw) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= ((uint64_t)(addr & 0x3FFFF)) >> 4;            // bits [13:0]: start address
    d |= ((uint64_t)((lbo_raw & 0x3FFFF))) << 12;      // bits [29:16]: LBO
    d |= ((uint64_t)((sbo_raw & 0x3FFFF))) << 28;      // bits [45:32]: SBO
    d |= (uint64_t)1 << 46;                            // version = 1
    d |= (uint64_t)0 << 49;                            // base offset = 0
    d |= (uint64_t)0 << 61;                            // no swizzle
    return d;
}

// UMMA instruction descriptor: BF16 x BF16 -> FP32, K-major
__device__ __forceinline__ uint32_t make_umma_idesc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);     // dtype = FP32
    d |= (1u << 7);     // atype = BF16
    d |= (1u << 10);    // btype = BF16
    d |= (0u << 15);    // no transpose A => K-major
    d |= (0u << 16);    // no transpose B => K-major
    d |= ((N / 8) << 17);   // N >> 3
    d |= ((M / 16) << 24);  // M >> 4
    return d;
}

// UMMA cta_group::2 BF16->FP32
__device__ __forceinline__ void umma_f16_cg2_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

// UMMA commit with multicast barrier arrive
__device__ __forceinline__ void umma_commit_2sm_fn(uint64_t* bar, uint16_t mask) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::2"
        ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;"
        :: "r"(a), "h"(mask));
}

// TMEM operations
__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
        :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tmem_relinquish_fn() {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::: "memory");
}

// TMEM load 4 FP32 values from column
__device__ __forceinline__ void tmem_ld_4x32b_fn(uint32_t col, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(col));
}

__device__ __forceinline__ void tmem_ld_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

// Global kernel declaration (outside namespace)
extern __global__ void gemm_blackwell_kernel_global(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    uint32_t M,
    uint32_t N,
    uint32_t K);

namespace tvm_gemm_blackwell {

static constexpr int BM_PER_CTA = 128;
static constexpr int BN_COMBINED = 256;
static constexpr int BK = 16;
static constexpr int CLUSTER_SIZE = 2;
static constexpr int BLOCK_THREADS = 128;
static constexpr int M_TILE = 256;
static constexpr int N_TILE = 256;

}  // namespace tvm_gemm_blackwell

// Kernel definition at global scope
__global__ void gemm_blackwell_kernel_global(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    uint32_t M,
    uint32_t N,
    uint32_t K)
{
    using namespace tvm_gemm_blackwell;
    
    uint32_t tid = threadIdx.x;
    uint32_t cta_rank = cluster_rank_fn();
    
    uint32_t m_base = blockIdx.x * M_TILE + cta_rank * BM_PER_CTA;
    uint32_t n_base = blockIdx.y * N_TILE;
    
    uint32_t BM_actual = min(m_base + BM_PER_CTA, M) - m_base;
    if (BM_actual == 0) return;
    uint32_t BN_actual = min(n_base + N_TILE, N) - n_base;
    if (BN_actual == 0) return;
    
    extern __shared__ char shmem_raw[];
    uintptr_t ptr = reinterpret_cast<uintptr_t>(shmem_raw);
    ptr = (ptr + 255) & ~255ULL;
    char* aligned = reinterpret_cast<char*>(ptr);
    
    __nv_bfloat16* smem_A = reinterpret_cast<__nv_bfloat16*>(aligned);
    __nv_bfloat16* smem_B = smem_A + BK * BM_PER_CTA;
    __nv_bfloat16* smem_out = smem_B + BK * BN_COMBINED;
    uint64_t* mbar = reinterpret_cast<uint64_t*>(
        reinterpret_cast<char*>(smem_out) + BM_PER_CTA * BN_COMBINED * sizeof(__nv_bfloat16));
    uint32_t* tmem_addr = reinterpret_cast<uint32_t*>(mbar + 1);
    
    if (tid == 0) {
        init_smem_barrier_fn(mbar, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    
    if (tid == 0) {
        tmem_alloc_fn(tmem_addr, 256);
    }
    __syncthreads();
    uint32_t tmem_c_base = tmem_addr[0];
    
    constexpr uint32_t sbo_val = 128;
    uint32_t lbo_a = (BM_PER_CTA / 8) * sbo_val;
    uint64_t desc_a = make_smem_desc_sm100_fn(smem_A, lbo_a, sbo_val);
    uint32_t lbo_b = (BN_COMBINED / 8) * sbo_val;
    uint64_t desc_b = make_smem_desc_sm100_fn(smem_B, lbo_b, sbo_val);
    uint32_t idesc = make_umma_idesc_fn(BM_PER_CTA, BN_COMBINED);
    
    uint32_t num_k_tiles = (K + BK - 1) / BK;
    uint32_t phase = 0;
    uint32_t acc_flag = 0;
    
    for (uint32_t kt = 0; kt < num_k_tiles; ++kt) {
        uint32_t k_pos = kt * BK;
        uint32_t bk_this = min(k_pos + BK, K) - k_pos;
        
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar, 0);
        }
        
        // Load A tile K-major
        for (uint32_t idx = tid; idx < bk_this * BM_actual; idx += BLOCK_THREADS) {
            uint32_t k_off = idx / BM_actual;
            uint32_t m_off = idx % BM_actual;
            smem_A[k_off * BM_PER_CTA + m_off] = 
                A[(m_base + m_off) * K + (k_pos + k_off)];
        }
        for (uint32_t k_idx = bk_this; k_idx < BK; ++k_idx) {
            for (uint32_t m = tid; m < BM_actual; m += BLOCK_THREADS) {
                smem_A[k_idx * BM_PER_CTA + m] = __float2bfloat16(0.0f);
            }
        }
        
        // Load B tile K-major
        for (uint32_t idx = tid; idx < bk_this * BN_actual; idx += BLOCK_THREADS) {
            uint32_t k_off = idx / BN_actual;
            uint32_t n_off = idx % BN_actual;
            smem_B[k_off * BN_COMBINED + n_off] = 
                B[(n_base + n_off) * K + (k_pos + k_off)];
        }
        for (uint32_t k_idx = bk_this; k_idx < BK; ++k_idx) {
            for (uint32_t n = tid; n < BN_actual; n += BLOCK_THREADS) {
                smem_B[k_idx * BN_COMBINED + n] = __float2bfloat16(0.0f);
            }
        }
        
        __syncthreads();
        fence_proxy_async_fn();
        
        umma_f16_cg2_fn(tmem_c_base, desc_a, desc_b, idesc, acc_flag);
        
        if (tid == 0) {
            umma_commit_2sm_fn(mbar, 0x3);
        }
        mbarrier_wait_fn(mbar, phase);
        phase++;
        acc_flag = 1;
    }
    
    tmem_ld_fence_fn();
    
    if (tid < BM_actual) {
        for (uint32_t col = 0; col < BN_COMBINED; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_ld_4x32b_fn(tmem_c_base + col, r0, r1, r2, r3);
            
            uint32_t base = tid * BN_COMBINED + col;
            smem_out[base + 0] = __float2bfloat16(__uint_as_float(r0));
            smem_out[base + 1] = __float2bfloat16(__uint_as_float(r1));
            smem_out[base + 2] = __float2bfloat16(__uint_as_float(r2));
            smem_out[base + 3] = __float2bfloat16(__uint_as_float(r3));
        }
    }
    
    tmem_ld_fence_fn();
    __syncthreads();
    
    uint32_t warp_id = tid / 32;
    uint32_t lane_id = tid % 32;
    uint32_t steps = (BM_actual + 3) / 4;
    
    for (uint32_t step = 0; step < steps; ++step) {
        uint32_t local_row = step * 4 + warp_id;
        if (local_row >= BM_actual) continue;
        
        uint32_t global_row = m_base + local_row;
        uint32_t col_start = lane_id * 4;
        uint32_t global_col = n_base + col_start;
        
        if (global_row < M && global_col + 3 < N) {
            uint2 val = reinterpret_cast<uint2*>(&smem_out[local_row * BN_COMBINED])[col_start / 2];
            *reinterpret_cast<uint2*>(C + static_cast<uint64_t>(global_row) * N + global_col) = val;
        }
    }
    
    if (tid == 0) {
        tmem_dealloc_fn(tmem_addr[0], 256);
    }
    __syncthreads();
    tmem_relinquish_fn();
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    
    using namespace tvm_gemm_blackwell;
    
    uint32_t M = static_cast<uint32_t>(A.size(0));
    uint32_t N = static_cast<uint32_t>(B.size(0));
    uint32_t K = static_cast<uint32_t>(A.size(1));
    
    const __nv_bfloat16* A_data = static_cast<const __nv_bfloat16*>(A.data_ptr());
    const __nv_bfloat16* B_data = static_cast<const __nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* C_data = static_cast<__nv_bfloat16*>(C.data_ptr());
    
    uint32_t grid_x = (M + M_TILE - 1) / M_TILE;
    uint32_t grid_y = (N + N_TILE - 1) / N_TILE;
    grid_x = (grid_x + CLUSTER_SIZE - 1) / CLUSTER_SIZE * CLUSTER_SIZE;
    
    dim3 grid(grid_x, grid_y, 1);
    dim3 block(BLOCK_THREADS, 1, 1);
    
    size_t shmem_size = 
        BK * BM_PER_CTA * sizeof(__nv_bfloat16) +
        BK * BN_COMBINED * sizeof(__nv_bfloat16) +
        BM_PER_CTA * BN_COMBINED * sizeof(__nv_bfloat16) +
        512;
    shmem_size = (shmem_size + 255) & ~255ULL;
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = static_cast<unsigned long long>(shmem_size);
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = CLUSTER_SIZE;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    cudaLaunchKernelEx(&config, gemm_blackwell_kernel_global,
                       A_data, B_data, C_data, M, N, K);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_gemm_blackwell::run);