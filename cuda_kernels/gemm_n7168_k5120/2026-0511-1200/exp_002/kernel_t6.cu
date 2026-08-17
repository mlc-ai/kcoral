#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>
#include <algorithm>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace gemm_ns {

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

__device__ __forceinline__ void mbarrier_arrive_fn(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])) : "memory");
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

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
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
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        :: "l"((uint64_t)bar) : "memory");
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "l"((uint64_t)bar),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr) >> 4;
    d |= (uint64_t)(addr & 0x3FFF);
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)0xb0ULL << 53; 
    d |= (uint64_t)2 << 61;   
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
    
    // Phase 2: SMEM -> Global
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t num_steps = (BM + 3) / 4;
    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 4 + warp_id;
        if (row >= BM) continue;
        uint32_t global_row = m_block * BM + row;
        uint32_t col_start = lane_id * 4;
        uint32_t global_col = n_block * BN + col_start;
        if (global_row < M) {
            if (global_col + 3 < N) {
                uint2 data = *reinterpret_cast<uint2*>(&smem_out[row * BN + col_start]);
                *reinterpret_cast<uint2*>(D + (uint64_t)global_row * N + global_col) = data;
            } else {
                for (int c = 0; c < 4; ++c) {
                    if (global_col + c < N) {
                        D[(uint64_t)global_row * N + global_col + c] = smem_out[row * BN + col_start + c];
                    }
                }
            }
        }
    }
}

__global__ void gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* C_ptr,
    uint32_t M, uint32_t N, uint32_t K) 
{
    uint32_t n_block = blockIdx.x;
    uint32_t m_block = blockIdx.y;

    __shared__ __align__(1024) uint8_t smem_A[2][128 * 64 * sizeof(__nv_bfloat16)];
    __shared__ __align__(1024) uint8_t smem_B[2][128 * 64 * sizeof(__nv_bfloat16)];
    __shared__ __align__(8) uint64_t tma_bar[2];
    __shared__ __align__(8) uint64_t umma_bar[2];
    __shared__ uint32_t smem_tmem_addr;

    uint32_t K_tiles = K / 64;
    uint32_t tx_bytes = (128 * 64 * sizeof(__nv_bfloat16)) * 2; 

    if (threadIdx.x == 0) {
        for (int i = 0; i < 2; i++) {
            init_smem_barrier_fn(&tma_bar[i], 128);
            init_smem_barrier_fn(&umma_bar[i], 1);
        }
    }
    __syncthreads();
    fence_smem_barrier_init_fn();
    __syncthreads();

    if (threadIdx.x < 32) {
        tmem_alloc_cg1_fn(&smem_tmem_addr, 128);
    }
    __syncthreads();
    uint32_t tmem_addr = smem_tmem_addr;

    uint64_t desc_A[2];
    uint64_t desc_B[2];
    if (threadIdx.x == 0) {
        desc_A[0] = make_smem_desc_sm100_fn(smem_A[0], 1024);
        desc_A[1] = make_smem_desc_sm100_fn(smem_A[1], 1024);
        desc_B[0] = make_smem_desc_sm100_fn(smem_B[0], 1024);
        desc_B[1] = make_smem_desc_sm100_fn(smem_B[1], 1024);
    }
    uint32_t idesc = make_instr_desc_fn(128, 128); 

    int tma_phase[2] = {0, 0};
    int umma_phase[2] = {0, 0};

    // Prologue
    uint32_t prologue_end = min(2U, K_tiles);
    for (uint32_t k = 0; k < prologue_end; k++) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&tma_bar[k], tx_bytes);
            tma_load_2d_fn(&tma_A, &tma_bar[k], smem_A[k], k * 64, m_block * 128);
            tma_load_2d_fn(&tma_B, &tma_bar[k], smem_B[k], k * 64, n_block * 128);
        } else {
            mbarrier_arrive_fn(&tma_bar[k]);
        }
    }

    for (uint32_t k = 0; k < K_tiles; k++) {
        int read_idx = k % 2;
        
        mbarrier_wait_fn(&tma_bar[read_idx], tma_phase[read_idx]);
        tma_phase[read_idx] ^= 1;
        
        fence_proxy_async_fn(); 
        __syncthreads(); 
        
        uint32_t accum = (k > 0) ? 1 : 0;
        if (threadIdx.x == 0) {
            tcgen05_fence_after_fn();
            umma_f16_cg1_fn(tmem_addr, desc_A[read_idx], desc_B[read_idx], idesc, accum);
            umma_commit_cg1_fn(&umma_bar[read_idx]);
        }
        
        // Wait for UMMA to finish before issuing the next TMA load into the same buffer
        mbarrier_wait_fn(&umma_bar[read_idx], umma_phase[read_idx]);
        umma_phase[read_idx] ^= 1;
        
        __syncthreads();
        
        if (k + 2 < K_tiles) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&tma_bar[read_idx], tx_bytes);
                tma_load_2d_fn(&tma_A, &tma_bar[read_idx], smem_A[read_idx], (k + 2) * 64, m_block * 128);
                tma_load_2d_fn(&tma_B, &tma_bar[read_idx], smem_B[read_idx], (k + 2) * 64, n_block * 128);
            } else {
                mbarrier_arrive_fn(&tma_bar[read_idx]);
            }
        }
    }

    __syncthreads();
    
    __nv_bfloat16* smem_out = (__nv_bfloat16*)smem_A;
    tmem_epilogue_coalesced_4w_fn(C_ptr, smem_out, M, N, m_block, n_block, 128, 128, tmem_addr);

    __syncthreads();
    if (threadIdx.x < 32) {
        tmem_dealloc_cg1_fn(tmem_addr, 128);
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id)); 
    
    uint32_t M = A.size(0);
    uint32_t K = A.size(1);
    uint32_t N = B.size(0); 

    __nv_bfloat16* A_ptr = static_cast<__nv_bfloat16*>(A.data_ptr());
    __nv_bfloat16* B_ptr = static_cast<__nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* C_ptr = static_cast<__nv_bfloat16*>(C.data_ptr());

    CUtensorMap tma_A, tma_B;

    CUresult res_A = create_tma_2d_descriptor_2B(
        &tma_A, A_ptr, K, M, 64, 128,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res_A != CUDA_SUCCESS) {
        fprintf(stderr, "TMA A descriptor creation failed\n");
        exit(1);
    }

    CUresult res_B = create_tma_2d_descriptor_2B(
        &tma_B, B_ptr, K, N, 64, 128,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res_B != CUDA_SUCCESS) {
        fprintf(stderr, "TMA B descriptor creation failed\n");
        exit(1);
    }

    int threads = 128;
    dim3 block(threads);
    dim3 grid((N + 127) / 128, (M + 127) / 128);

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    gemm_kernel<<<grid, block, 0, stream>>>(tma_A, tma_B, C_ptr, M, N, K);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace gemm_ns