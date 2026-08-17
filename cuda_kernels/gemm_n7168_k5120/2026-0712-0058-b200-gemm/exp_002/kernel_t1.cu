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

#define CU_CHECK(call) do {                                        \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        const char* err_str;                                       \
        cuGetErrorString(_e, &err_str);                            \
        fprintf(stderr, "CU error %s at %s:%d\n",                 \
                err_str, __FILE__, __LINE__);                      \
        exit(1);                                                   \
    }                                                              \
} while(0)

#define m_block 64
#define n_block 64

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

__device__ __forceinline__ uint64_t make_smem_desc(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   // version = 1 (SM100)
    uint64_t base_offset_val = (addr >> 7) & 0x7;
    d |= (base_offset_val << 49);
    d |= (uint64_t)2 << 61;   // layout_type = SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    // c_format = FP32
    d |= (1u << 7);    // a_format = BF16
    d |= (1u << 10);   // b_format = BF16
    d |= (0u << 15);   // a_major = 0 (A is K-Major)
    d |= (0u << 16);   // b_major = 0 (B is K-Major)
    d |= ((N / 8) << 17);     // n_dim
    d |= ((M / 16) << 24);    // m_dim
    return d;
}

__device__ __forceinline__ void cta_gemm_issue_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void commit_1sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1"
        ".mbarrier::arrive::one.shared::cta.b64"
        " [%0];"
        :: "r"(a));
}

__global__ void __launch_bounds__(128) multigemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    int m_dim, int n_dim, __nv_bfloat16* C) 
{
    int cta_id = cluster_rank_fn();
    int m_start = blockIdx.x * 128 + cta_id * m_block;
    int n_start = blockIdx.y * n_block;
    bool cta_is_0 = (cta_id == 0);
    
    uint32_t tmem_c;
    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_c, 128);
    }
    __syncthreads();

    extern __shared__ __align__(128) uint8_t smem_pool[];
    uint64_t* bar_A = (uint64_t*)smem_pool;
    uint64_t* bar_B = (uint64_t*)(smem_pool + 16);
    uint64_t* bar_umma = (uint64_t*)(smem_pool + 32);
    
    // Advance pointer to 1024 byte alignment boundary dynamically to satisfy TMA requirements safely
    uint64_t pool_addr = (uint64_t)smem_pool;
    uint64_t aligned_addr = (pool_addr + 1023) & ~1023;
    __nv_bfloat16* A_smem_base = (__nv_bfloat16*)aligned_addr;
    __nv_bfloat16* B_smem_base = (__nv_bfloat16*)(aligned_addr + 16384);
    
    __nv_bfloat16* A_smem[2] = {A_smem_base, A_smem_base + 4096};
    __nv_bfloat16* B_smem[2] = {B_smem_base, B_smem_base + 4096};

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&bar_A[0], 1);
        init_smem_barrier_fn(&bar_A[1], 1);
        init_smem_barrier_fn(&bar_B[0], 1);
        init_smem_barrier_fn(&bar_B[1], 1);
        init_smem_barrier_fn(&bar_umma[0], 1);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();

    uint32_t phase_A[2] = {0, 0};
    uint32_t phase_B[2] = {0, 0};
    uint32_t phase_umma = 0;
    uint32_t idesc = make_instr_desc_fn(m_block, n_block);

    // Software pipeline prefetching across K dimension (5120)
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&bar_A[0], 8192);
        tma_load_2d_fn(&tma_A, &bar_A[0], A_smem[0], 0, m_start);
        mbarrier_arrive_and_expect_tx_fn(&bar_B[0], 8192);
        tma_load_2d_fn(&tma_B, &bar_B[0], B_smem[0], 0, n_start);
    }

    for (int k_outer = 0; k_outer < 5120; k_outer += 64) {
        int next_k = k_outer + 64;
        int curr_buf = (k_outer / 64) % 2;
        int next_buf = (curr_buf + 1) % 2;
        
        if (next_k < 5120) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&bar_A[next_buf], 8192);
                tma_load_2d_fn(&tma_A, &bar_A[next_buf], A_smem[next_buf], next_k, m_start);
                mbarrier_arrive_and_expect_tx_fn(&bar_B[next_buf], 8192);
                tma_load_2d_fn(&tma_B, &bar_B[next_buf], B_smem[next_buf], next_k, n_start);
            }
        }
        
        mbarrier_wait_fn(&bar_A[curr_buf], phase_A[curr_buf]);
        mbarrier_wait_fn(&bar_B[curr_buf], phase_B[curr_buf]);

        __nv_bfloat16* A_smem_curr = A_smem[curr_buf];
        __nv_bfloat16* B_smem_curr = B_smem[curr_buf];

        if (cta_is_0 && threadIdx.x == 0) {
            uint32_t tmem_c_curr = tmem_c;
            uint32_t accum = (k_outer == 0) ? 0 : 1;
            
            // Unroll and chain asynchronous tensor core instructions
            uint64_t desc_a = make_smem_desc(A_smem_curr, 16, 1024);
            uint64_t desc_b = make_smem_desc(B_smem_curr, 8192, 1024);
            
            for (uint32_t i = 0; i < 4; ++i) {
                cta_gemm_issue_fn(tmem_c_curr, desc_a, desc_b, idesc, accum);
                tmem_c_curr += 2;
                desc_a += 2;
                desc_b += 128; // Walk linearly along the leading dimension
                accum = 1;
            }
            commit_1sm_fn(&bar_umma[0]);
        }
        
        mbarrier_wait_fn(&bar_umma[0], phase_umma);
        
        phase_A[curr_buf] ^= 1;
        phase_B[curr_buf] ^= 1;
        phase_umma ^= 1;
    }

    // Fast hardware-native Epilogue writing directly to Coalesced Vectors
    int m_idx = threadIdx.x;
    if (m_idx < m_block) {
        for (int col = 0; col < n_block; col += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(tmem_c + col));
            
            float f0 = __uint_as_float(r0);
            float f1 = __uint_as_float(r1);
            float f2 = __uint_as_float(r2);
            float f3 = __uint_as_float(r3);
            
            int global_col = n_start + col;
            __nv_bfloat16* out = C + (uint64_t)m_idx * n_dim + global_col;
            if (global_col     < n_dim) out[0] = __float2bfloat16(f0);
            if (global_col + 1 < n_dim) out[1] = __float2bfloat16(f1);
            if (global_col + 2 < n_dim) out[2] = __float2bfloat16(f2);
            if (global_col + 3 < n_dim) out[3] = __float2bfloat16(f3);
        }
    }

    // Free Tensor Memory allocations properly
    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_c, 128);
    }
    __syncthreads();
}

namespace tvm_ffi_gemm {

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2, // tensorRank
        globalAddress,
        globalDim,
        globalStrides,
        boxDim, 
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        l2Promotion,
        oobFill
    );
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    int64_t m_dim = A.size(0);
    int64_t n_dim = B.size(0);
    int64_t k_dim = B.size(1);

    const __nv_bfloat16* A_data = static_cast<const __nv_bfloat16*>(A.data_ptr());
    const __nv_bfloat16* B_data = static_cast<const __nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* C_data = static_cast<__nv_bfloat16*>(C.data_ptr());

    CUtensorMap tma_A, tma_B;
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_A, (void*)A_data, k_dim, m_dim, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_B, (void*)B_data, k_dim, n_dim, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    int num_m_blocks = (m_dim + 127) / 128;
    int num_n_blocks = (n_dim + n_block - 1) / n_block;
    dim3 grid(num_m_blocks, num_n_blocks, 1);
    dim3 block(128, 1, 1);

    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 40 * 1024;
    config.stream = stream;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaFuncSetAttribute(multigemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 40 * 1024));
    CUDA_CHECK(cudaLaunchKernelEx(&config, multigemm_kernel, tma_A, tma_B, m_dim, n_dim, C_data));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_gemm::run);

}  // namespace tvm_ffi_gemm