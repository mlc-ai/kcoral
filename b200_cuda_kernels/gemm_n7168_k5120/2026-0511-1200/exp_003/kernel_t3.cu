#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
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

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2}; // 2 bytes per bf16 element
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2, 
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

__device__ __forceinline__ void tma_load_2d_cg1_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tmem_alloc_cg1_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_cg1_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr) >> 4;
    d |= (uint64_t)(addr & 0x3FFF);
    d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16;
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)0xb0ull << 53;
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);           // dtype = F32
    d |= (1u << 7);           // atype = BF16
    d |= (1u << 10);          // btype = BF16
    d |= (0u << 15);          // Transpose A = 0
    d |= (0u << 16);          // Transpose B = 0
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ void umma_f16_cg1_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_cg1_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        :: "r"(a));
}

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_before_fn() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void my_tmem_epilogue(
    __nv_bfloat16* D, __nv_bfloat16* smem_out,
    uint32_t M, uint32_t N, uint32_t m_block, uint32_t n_block,
    uint32_t BM, uint32_t BN, uint32_t tmem_addr) {
    
    // TMEM to SMEM (collective instruction, executed by all threads in the warp)
    for (uint32_t col = 0; col < BN; col += 4) {
        uint32_t r0, r1, r2, r3;
        uint32_t addr = tmem_addr + col;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(addr));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t base = threadIdx.x * BN + col;
        smem_out[base + 0] = __float2bfloat16(__uint_as_float(r0));
        smem_out[base + 1] = __float2bfloat16(__uint_as_float(r1));
        smem_out[base + 2] = __float2bfloat16(__uint_as_float(r2));
        smem_out[base + 3] = __float2bfloat16(__uint_as_float(r3));
    }
    __syncthreads();

    // SMEM to Global (coalesced)
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t num_steps = (BM + 3) / 4;
    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 4 + warp_id;
        if (row >= BM) continue;
        uint32_t global_row = m_block * BM + row;
        uint32_t col_start = lane_id * 4;
        uint32_t global_col = n_block * BN + col_start;
        if (global_row < M && global_col + 3 < N) {
            uint2 data = *reinterpret_cast<uint2*>(&smem_out[row * BN + col_start]);
            *reinterpret_cast<uint2*>(D + (uint64_t)global_row * N + global_col) = data;
        } else if (global_row < M) {
            for (int i = 0; i < 4; ++i) {
                if (global_col + i < N) {
                    D[global_row * N + global_col + i] = smem_out[row * BN + col_start + i];
                }
            }
        }
    }
}

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 64;
constexpr int NUM_STAGES = 4;

__global__ void gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* C,
    uint32_t M, uint32_t N, uint32_t K) {
    
    // 1024-byte alignment is MANDATORY for 128B Swizzling
    __shared__ alignas(1024) __nv_bfloat16 smem_A[NUM_STAGES][BM][BK];
    __shared__ alignas(1024) __nv_bfloat16 smem_B[NUM_STAGES][BN][BK];
    __shared__ alignas(8) uint64_t mbar_load[NUM_STAGES];
    __shared__ alignas(8) uint64_t mbar_mma[NUM_STAGES];
    __shared__ uint32_t tmem_addr_smem;
    
    uint32_t m_idx = blockIdx.y;
    uint32_t n_idx = blockIdx.x;
    
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    
    uint32_t tx_bytes = (BM * BK + BN * BK) * sizeof(__nv_bfloat16);
    
    if (threadIdx.x == 0) {
        for (int i = 0; i < NUM_STAGES; ++i) {
            init_smem_barrier_fn(&mbar_load[i], 1);
            init_smem_barrier_fn(&mbar_mma[i], 1);
        }
        fence_smem_barrier_init_fn();
    }
    
    if (warp_id == 0) {
        tmem_alloc_cg1_fn(&tmem_addr_smem, 128);
    }
    __syncthreads();
    
    uint32_t tmem_addr = tmem_addr_smem;
    uint32_t k_steps = (K + BK - 1) / BK;
    
    // Prologue: Fire initial TMA loads
    int prefetch = (NUM_STAGES < k_steps) ? NUM_STAGES : k_steps;
    if (threadIdx.x == 0) {
        for (int i = 0; i < prefetch; ++i) {
            mbarrier_arrive_and_expect_tx_fn(&mbar_load[i], tx_bytes);
            tma_load_2d_cg1_fn(&tma_A, &mbar_load[i], smem_A[i], i * BK, m_idx * BM);
            tma_load_2d_cg1_fn(&tma_B, &mbar_load[i], smem_B[i], i * BK, n_idx * BN);
        }
    }
    __syncthreads();
    
    uint32_t idesc = make_instr_desc_fn(BM, BN);
    
    // Decoupled Main Loop: Warp 0 handles UMMA computation
    if (warp_id == 0) {
        for (int mma_step = 0; mma_step < k_steps; ++mma_step) {
            int mma_stage = mma_step % NUM_STAGES;
            mbarrier_wait_fn(&mbar_load[mma_stage], (mma_step / NUM_STAGES) & 1);
            
            // A is [BM, BK] (rows=BM). SBO for A is offset between 8 M-rows = 8 * 128 bytes = 1024 bytes.
            uint64_t desc_A = make_smem_desc_sm100_fn(smem_A[mma_stage], 0, 1024);
            // B is [BK, BN] (rows=BK). SBO for B is offset between 8 K-rows = 8 * 2 bytes = 16 bytes.
            uint64_t desc_B = make_smem_desc_sm100_fn(smem_B[mma_stage], 0, 16);
            
            if (lane_id == 0) {
                uint32_t accum = (mma_step == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_addr, desc_A, desc_B, idesc, accum);
                umma_commit_cg1_fn(&mbar_mma[mma_stage]);
            }
        }
    } 
    // Decoupled Main Loop: Warp 1 handles issuing consecutive TMA loads
    else if (warp_id == 1) {
        for (int load_step = NUM_STAGES; load_step < k_steps; ++load_step) {
            int load_stage = load_step % NUM_STAGES;
            // Wait for the UMMA that previously used `load_stage` to finish to avoid overwriting
            mbarrier_wait_fn(&mbar_mma[load_stage], ((load_step - NUM_STAGES) / NUM_STAGES) & 1);
            if (lane_id == 0) {
                mbarrier_arrive_and_expect_tx_fn(&mbar_load[load_stage], tx_bytes);
                tma_load_2d_cg1_fn(&tma_A, &mbar_load[load_stage], smem_A[load_stage], load_step * BK, m_idx * BM);
                tma_load_2d_cg1_fn(&tma_B, &mbar_load[load_stage], smem_B[load_stage], load_step * BK, n_idx * BN);
            }
        }
    }
    
    // Wait for all outstanding UMMA instructions to complete across the CTA
    // Producer (Warp 0) enforces execution ordering with consumer (Warps 0-3 reading TMEM)
    if (warp_id == 0 && lane_id == 0) {
        int last_step = k_steps - 1;
        mbarrier_wait_fn(&mbar_mma[last_step % NUM_STAGES], (last_step / NUM_STAGES) & 1);
        tcgen05_fence_before_fn();
    }
    __syncthreads();
    tcgen05_fence_after_fn();
    
    // Epilogue
    my_tmem_epilogue(C, (__nv_bfloat16*)smem_A[0], M, N, m_idx, n_idx, BM, BN, tmem_addr);
    
    if (warp_id == 0) {
        tmem_dealloc_cg1_fn(tmem_addr, 128);
    }
}

namespace tvm_ffi_gemm_n7168_k5120 {

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0);
    
    void* A_ptr = A.data_ptr();
    void* B_ptr = B.data_ptr();
    void* C_ptr = C.data_ptr();
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    
    CUtensorMap tma_A, tma_B;
    
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_A, A_ptr, K, M, BK, BM,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_B, B_ptr, K, N, BK, BN,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        
    dim3 block(128);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 1;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_kernel, tma_A, tma_B, (__nv_bfloat16*)C_ptr, (uint32_t)M, (uint32_t)N, (uint32_t)K));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_gemm_n7168_k5120