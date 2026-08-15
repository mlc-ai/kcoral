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

namespace tvm_ffi_example_cuda {

CUresult my_create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
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

__device__ __forceinline__ void tmem_alloc_cg1_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_cg1_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(addr), "r"(ncols));
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
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_before_fn() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr) >> 4;
    d |= (uint64_t)(addr & 0x3FFF);
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;   // version = 1 (SM100)
    d |= (uint64_t)2 << 61;   // layout_type = SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);           // c_format = FP32
    d |= (1u << 7);           // a_format = BF16
    d |= (1u << 10);          // b_format = BF16
    d |= ((N / 8) << 17);     // n_dim
    d |= ((M / 16) << 24);    // m_dim
    return d;
}

__device__ __forceinline__ void umma_commit_cg1_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1"
        ".mbarrier::arrive::one.b64"
        " [%0];"
        :: "r"(a));
}

__device__ __forceinline__ void tmem_epilogue_coalesced_4w_fn(
    __nv_bfloat16* D, __nv_bfloat16* smem_out,
    uint32_t M, uint32_t N, uint32_t m_block, uint32_t n_block,
    uint32_t BM, uint32_t BN, uint32_t tmem_addr) {
    // Phase 1: TMEM -> SMEM
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
    
    // Phase 2: SMEM -> Global (coalesced vectorized 8-byte writes)
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
        }
    }
}

constexpr int STAGE = 4;
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 64;

__global__ void gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* C,
    int M, int N, int K) 
{
    int m_block = blockIdx.y;
    int n_block = blockIdx.x;
    
    int m_idx = m_block * BM;
    int n_idx = n_block * BN;
    
    extern __shared__ __align__(1024) char smem_buf[];
    uint16_t* smem_A = (uint16_t*)smem_buf;
    uint16_t* smem_B = (uint16_t*)(smem_buf + STAGE * BM * BK * sizeof(uint16_t));
    uint64_t* tma_bar = (uint64_t*)(smem_B + STAGE * BN * BK);
    uint64_t* umma_bar = tma_bar + STAGE;
    uint32_t* tmem_addr_smem = (uint32_t*)(umma_bar + STAGE);
    
    if (threadIdx.x < 32) {
        tmem_alloc_cg1_fn(tmem_addr_smem, BN);
    }
    
    if (threadIdx.x < STAGE) {
        init_smem_barrier_fn(&tma_bar[threadIdx.x], 1);
        init_smem_barrier_fn(&umma_bar[threadIdx.x], 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();
    
    uint32_t tmem_addr = *tmem_addr_smem;
    int num_iters = (K + BK - 1) / BK;
    
    if (threadIdx.x == 0) {
        int phase[STAGE] = {0};
        int umma_phase[STAGE] = {0};

        for (int i = 0; i < STAGE - 1; ++i) {
            if (i < num_iters) {
                int k_idx = i * BK;
                mbarrier_arrive_and_expect_tx_fn(&tma_bar[i], BM * BK * 2 * 2);
                tma_load_2d_fn(&tma_A, &tma_bar[i], smem_A + i * BM * BK, k_idx, m_idx);
                tma_load_2d_fn(&tma_B, &tma_bar[i], smem_B + i * BN * BK, k_idx, n_idx);
            }
        }
        
        uint32_t idesc = make_instr_desc_fn(BM, BN);
        uint32_t sbo = 1024; 
        
        for (int i = 0; i < num_iters; ++i) {
            int s = i % STAGE;
            mbarrier_wait_fn(&tma_bar[s], phase[s]);
            
            uint64_t desc_a = make_smem_desc_sm100_fn(smem_A + s * BM * BK, sbo);
            uint64_t desc_b = make_smem_desc_sm100_fn(smem_B + s * BN * BK, sbo);
            
            uint32_t accum = (i == 0) ? 0 : 1;
            
            fence_proxy_async_fn();
            
            asm volatile(
                "{\n.reg .pred p;\n"
                "setp.ne.b32 p, %4, 0;\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                :: "r"(tmem_addr), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
                
            umma_commit_cg1_fn(&umma_bar[s]);
            
            int next_i = i + STAGE - 1;
            if (next_i < num_iters) {
                int next_s = next_i % STAGE;
                if (next_i >= STAGE) {
                    mbarrier_wait_fn(&umma_bar[next_s], umma_phase[next_s]);
                    umma_phase[next_s] ^= 1;
                }
                
                int k_idx = next_i * BK;
                mbarrier_arrive_and_expect_tx_fn(&tma_bar[next_s], BM * BK * 2 * 2);
                tma_load_2d_fn(&tma_A, &tma_bar[next_s], smem_A + next_s * BM * BK, k_idx, m_idx);
                tma_load_2d_fn(&tma_B, &tma_bar[next_s], smem_B + next_s * BN * BK, k_idx, n_idx);
            }
            
            phase[s] ^= 1;
        }
        
        int last_s = (num_iters - 1) % STAGE;
        mbarrier_wait_fn(&umma_bar[last_s], umma_phase[last_s]);
        
        tcgen05_fence_before_fn();
    }
    
    __syncthreads();
    tcgen05_fence_after_fn();
    
    __nv_bfloat16* smem_out = (__nv_bfloat16*)smem_A;
    tmem_epilogue_coalesced_4w_fn(C, smem_out, M, N, m_block, n_block, BM, BN, tmem_addr);
    
    __syncthreads();
    if (threadIdx.x < 32) {
        tmem_dealloc_cg1_fn(tmem_addr, BN);
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0);
    
    CUtensorMap tma_A, tma_B;
    void* A_ptr = A.data_ptr();
    void* B_ptr = B.data_ptr();
    __nv_bfloat16* C_ptr = static_cast<__nv_bfloat16*>(C.data_ptr());
    
    CU_CHECK(my_create_tma_2d_descriptor_2B(&tma_A, A_ptr, K, M, BK, BM, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        
    CU_CHECK(my_create_tma_2d_descriptor_2B(&tma_B, B_ptr, K, N, BK, BN, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
    dim3 block(128);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    
    int smem_bytes = STAGE * BM * BK * 2 * 2 + STAGE * 8 * 2 + 1024;
    
    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
        
    CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    
    gemm_kernel<<<grid, block, smem_bytes, stream>>>(tma_A, tma_B, C_ptr, M, N, K);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

} // namespace tvm_ffi_example_cuda