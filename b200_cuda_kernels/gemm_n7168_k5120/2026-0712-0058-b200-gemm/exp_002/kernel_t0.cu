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

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase));
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ uint64_t make_smem_desc_swizzled_128B_raw(void* smem_ptr, uint32_t m_byte_offset, uint32_t k_byte_offset, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr) + m_byte_offset + k_byte_offset;
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   // version = 1 (SM100)
    uint32_t base_offset_val = (addr >> 7) & 0x7;
    d |= (base_offset_val << 49);
    d |= (uint64_t)2 << 61;   // layout_type = SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    // c_format = FP32
    d |= (1u << 7);    // a_format = BF16
    d |= (1u << 10);   // b_format = BF16
    d |= (0u << 15);   // a_major = 0 (A is K-Major), 1 (A is M-Major)
    d |= (0u << 16);   // b_major = 0 (B is K-Major), 1 (B is N-Major)
    d |= ((N / 8) << 17);     // n_dim. For cta_group::2, N should be the combined dimension (BN_per_cta * 2).
    d |= ((M / 16) << 24);    // m_dim. For cta_group::2, M should be the combined dimension (BM_per_cta * 2).
    return d;
}

__device__ __forceinline__ void cta_gemm_issue_cg2_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_2sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::2"
        ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;"
        :: "r"(a), "h"((uint16_t)0x3)); // It hardcodes ctaMask=0x3 (CTAs 0 and 1 only). This works only for cluster_size=2.
}

__global__ void __launch_bounds__(128) multigemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    int m_dim, int n_dim, __nv_bfloat16* C) 
{
    // Grid & Cluster mapping logic
    int global_m = blockIdx.x * m_block;
    int global_n = blockIdx.y * n_block;
    int cta_id = cluster_rank_fn();
    
    // Allocate TMEM for output C (256 columns * 128 rows * 4 bytes = 128KB)
    uint32_t tmem_c0, tmem_c1;
    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_c0, 512);
        tmem_alloc_fn(&tmem_c1, 512);
    }
    __syncthreads();

    extern __shared__ __align__(128) uint8_t smem_pool[];
    __nv_bfloat16* A_smem = (__nv_bfloat16*)smem_pool;
    __nv_bfloat16* B_smem = (__nv_bfloat16*)(smem_pool + 65536);

    // MBarriers and Phase tracking
    __shared__ __align__(8) uint64_t bar_A[2], bar_B[2], bar_umma[2];
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&bar_A[0], 1);
        init_smem_barrier_fn(&bar_A[1], 1);
        init_smem_barrier_fn(&bar_B[0], 1);
        init_smem_barrier_fn(&bar_B[1], 1);
        init_smem_barrier_fn(&bar_umma[0], 1);
        init_smem_barrier_fn(&bar_umma[1], 1);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();

    uint32_t phase_A[2] = {0, 0};
    uint32_t phase_B[2] = {0, 0};
    uint32_t phase_umma[2] = {0, 0};
    uint32_t idesc = make_instr_desc_fn(m_block, n_block);

    // Pipeline loops across K dimension (5120)
    for (int k_outer = 0; k_outer < 5120; k_outer += 512) {
        int k_phase = k_outer / 256; 
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&bar_A[cta_id], 131072);
            mbarrier_arrive_and_expect_tx_fn(&bar_B[cta_id], 131072);
        }
        
        // Hardware Pipelines TMA fetch while math executes
        for (uint32_t k = 0; k < 256; k += 64) {
            if (threadIdx.x == 0) {
                tma_load_2d_cg2_fn(&tma_A, &bar_A[cta_id], A_smem + (k * 128), k_outer + k, m_base + cta_id * m_block);
                tma_load_2d_cg2_fn(&tma_B, &bar_B[cta_id], B_smem0 + (k * 128), k_outer + k, global_n + 0);
                tma_load_2d_cg2_fn(&tma_B, &bar_B[cta_id], B_smem1 + (k * 128), k_outer + k, global_n + 128);
            }
        }
        
        mbarrier_wait_fn(&bar_A[cta_id], phase_A[cta_id]);
        mbarrier_wait_fn(&bar_B[cta_id], phase_B[cta_id]);

        if (is_cta0 && threadIdx.x == 0) {
            uint32_t accum = (k_outer == 0) ? 0 : 1;
            for (uint32_t k = 0; k < 256; k += 64) {
                uint64_t desc_a = make_smem_desc_swizzled_128B_raw(A_smem, m_offset + k * 128, k_offset + k * 2, 16, 1024);
                uint64_t desc_b0 = make_smem_desc_swizzled_128B_raw(B_smem0, n_offset0 + k * 128, k_offset + k * 2, 16, 1024);
                uint64_t desc_b1 = make_smem_desc_swizzled_128B_raw(B_smem1, n_offset1 + k * 128, k_offset + k * 2, 16, 1024);
                
                for (uint32_t i = 0; i < 4; ++i) {
                    cta_gemm_issue_cg2_fn(tmem_c0, desc_a, desc_b0, idesc, accum);
                    cta_gemm_issue_cg2_fn(tmem_c1, desc_a, desc_b1, idesc, accum);
                    desc_a += 32;
                    desc_b0 += 32;
                    desc_b1 += 32;
                    accum = 1;
                }
            }
            umma_commit_2sm_fn(&bar_umma[cta_id]);
        }
        
        mbarrier_wait_fn(&bar_umma[cta_id], phase_umma[cta_id]);
        
        phase_A[cta_id] ^= 1;
        phase_B[cta_id] ^= 1;
        phase_umma[cta_id] ^= 1;
    }

    // Direct Epilogue writing to Coalesced Vectors
    tmem_store_bf16_row_fn(C, threadIdx.x, m_dim, n_dim, m_base, global_n + 0, 128);
    tmem_store_bf16_row_fn(C, threadIdx.x, m_dim, n_dim, m_base, global_n + 128, 128);

    // Free Tensor Memory
    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_c0, 512);
        tmem_dealloc_fn(tmem_c1, 512);
    }
    __syncthreads();
}

namespace tvm_ffi_gemm {

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    int64_t m_dim = A.size(0);
    int64_t n_dim = B.size(0);
    int64_t k_dim = B.size(1);

    const __nv_bfloat16* A_data = static_cast<const __nv_bfloat16*>(A.data_ptr());
    const __nv_bfloat16* B_data = static_cast<const __nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* C_data = static_cast<__nv_bfloat16*>(C.data_ptr());

    CUtensorMap tma_A, tma_B;
    create_tma_2d_descriptor_2B(&tma_A, (void*)A_data, k_dim, m_dim, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_B, (void*)B_data, k_dim, n_dim, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    int num_m_blocks = (m_dim + m_block - 1) / m_block;
    int num_n_blocks = (n_dim + n_block - 1) / n_block;
    dim3 grid(num_m_blocks, num_n_blocks, 1);
    dim3 block(128, 1, 1);

    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 200 * 1024;
    config.stream = stream;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, multigemm_kernel, tma_A, tma_B, m_dim, n_dim, C_data));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_gemm::run);

}  // namespace tvm_ffi_gemm