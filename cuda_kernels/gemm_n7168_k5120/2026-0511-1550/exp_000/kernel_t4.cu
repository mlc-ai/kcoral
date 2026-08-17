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

namespace gemm_n7168_k5120 {

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
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

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)((addr >> 4) & 0x3FFF);
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;   
    
    uint32_t base_offset = (addr >> 7) & 0x7;
    d |= (uint64_t)(base_offset & 0x7) << 49;
    
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);           // c_format = FP32
    d |= (1u << 7);           // a_format = BF16
    d |= (1u << 10);          // b_format = BF16
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ void my_epilogue(
    uint32_t tmem_addr,
    __nv_bfloat16* D, __nv_bfloat16* smem_out,
    uint32_t M, uint32_t N, uint32_t m_block, uint32_t n_block,
    uint32_t BM, uint32_t BN) {
    
    tcgen05_fence_after_fn();
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

__global__ void __launch_bounds__(128) gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* C,
    uint32_t M, uint32_t N, uint32_t K) {
    
    setmaxnreg_inc_sync_fn<256>();
    
    uint32_t m_idx = blockIdx.y * 128;
    uint32_t n_idx = blockIdx.x * 128;
    
    if (m_idx >= M || n_idx >= N) return;
    
    __shared__ __align__(1024) __nv_bfloat16 smem_A[2][128 * 64];
    __shared__ __align__(1024) __nv_bfloat16 smem_B[2][128 * 64];
    
    __shared__ alignas(8) uint64_t bar_A[2];
    __shared__ alignas(8) uint64_t bar_B[2];
    __shared__ alignas(8) uint64_t bar_umma[2];
    
    if (threadIdx.x == 0) {
        for (int i = 0; i < 2; ++i) {
            init_smem_barrier_fn(&bar_A[i], 1);
            init_smem_barrier_fn(&bar_B[i], 1);
            init_smem_barrier_fn(&bar_umma[i], 1);
        }
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    
    __shared__ uint32_t shared_tmem_addr;
    if (threadIdx.x < 32) {
        uint32_t a = (uint32_t)__cvta_generic_to_shared(&shared_tmem_addr);
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     :: "r"(a), "r"(128));
    }
    __syncthreads();
    
    uint32_t tx_bytes = 16384; 
    
    if (threadIdx.x == 0) {
        for (int i = 0; i < 2 && i * 64 < K; ++i) {
            mbarrier_arrive_and_expect_tx_fn(&bar_A[i], tx_bytes);
            tma_load_2d_fn(&tma_A, &bar_A[i], smem_A[i], i * 64, m_idx);
            
            mbarrier_arrive_and_expect_tx_fn(&bar_B[i], tx_bytes);
            tma_load_2d_fn(&tma_B, &bar_B[i], smem_B[i], i * 64, n_idx);
        }
    }
    
    uint32_t idesc = make_instr_desc(128, 128);
    uint32_t phase_A[2] = {0, 0};
    uint32_t phase_B[2] = {0, 0};
    uint32_t phase_umma[2] = {0, 0};
    
    for (int k = 0; k < K; k += 64) {
        int buf = (k / 64) % 2;
        
        mbarrier_wait_fn(&bar_A[buf], phase_A[buf]);
        mbarrier_wait_fn(&bar_B[buf], phase_B[buf]);
        
        __syncthreads();
        tcgen05_fence_after_fn(); 
        
        if (threadIdx.x == 0) {
            uint32_t tmem_addr_reg;
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(tmem_addr_reg) : "r"((uint32_t)__cvta_generic_to_shared(&shared_tmem_addr)));
            
            uint32_t accum = (k == 0) ? 0 : 1;
            uint64_t s_desc_A = make_smem_desc_sm100_fn(smem_A[buf], 1024);
            uint64_t s_desc_B = make_smem_desc_sm100_fn(smem_B[buf], 1024);
            
            asm volatile(
                "{\n.reg .pred p;\n"
                "setp.ne.b32 p, %4, 0;\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                :: "r"(tmem_addr_reg), "l"(s_desc_A), "l"(s_desc_B), "r"(idesc), "r"(accum));
                
            uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar_umma[buf]);
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                :: "r"(a));
        }
        
        mbarrier_wait_fn(&bar_umma[buf], phase_umma[buf]);
        
        int next_k = k + 2 * 64;
        if (next_k < K) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&bar_A[buf], tx_bytes);
                tma_load_2d_fn(&tma_A, &bar_A[buf], smem_A[buf], next_k, m_idx);
                
                mbarrier_arrive_and_expect_tx_fn(&bar_B[buf], tx_bytes);
                tma_load_2d_fn(&tma_B, &bar_B[buf], smem_B[buf], next_k, n_idx);
            }
        }
        
        phase_A[buf] ^= 1;
        phase_B[buf] ^= 1;
        phase_umma[buf] ^= 1;
    }
    
    __syncthreads();
    
    uint32_t tmem_addr_reg;
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(tmem_addr_reg) : "r"((uint32_t)__cvta_generic_to_shared(&shared_tmem_addr)));
    
    my_epilogue(tmem_addr_reg, C, (__nv_bfloat16*)smem_A, M, N, m_idx / 128, n_idx / 128, 128, 128);
    
    __syncthreads();
    if (threadIdx.x < 32) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(tmem_addr_reg), "r"(128));
    }
}

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
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

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0); 

    __nv_bfloat16* A_ptr = static_cast<__nv_bfloat16*>(A.data_ptr());
    __nv_bfloat16* B_ptr = static_cast<__nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* C_ptr = static_cast<__nv_bfloat16*>(C.data_ptr());

    CUtensorMap tma_A, tma_B;
    
    CUresult resA = create_tma_2d_descriptor_2B(&tma_A, A_ptr, K, M, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (resA != CUDA_SUCCESS) { fprintf(stderr, "TMA A failed\n"); exit(1); }
    
    CUresult resB = create_tma_2d_descriptor_2B(&tma_B, B_ptr, K, N, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (resB != CUDA_SUCCESS) { fprintf(stderr, "TMA B failed\n"); exit(1); }

    int gridX = (N + 127) / 128;
    int gridY = (M + 127) / 128;
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(gridX, gridY, 1);
    config.blockDim = dim3(128, 1, 1);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_kernel, tma_A, tma_B, C_ptr, (uint32_t)M, (uint32_t)N, (uint32_t)K));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace gemm_n7168_k5120