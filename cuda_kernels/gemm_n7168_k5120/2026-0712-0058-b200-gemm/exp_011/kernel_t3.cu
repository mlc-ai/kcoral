#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
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
        fprintf(stderr, "CU error %d at %s:%d\n",                 \
                _e, __FILE__, __LINE__);                           \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_gemms {

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

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
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
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ uint64_t make_smem_desc(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint64_t advance_desc(uint64_t desc, uint32_t byte_offset) {
    uint32_t addr_bits = desc & 0x3FFF;
    uint32_t new_addr = (addr_bits << 4) + byte_offset;
    uint64_t new_desc = (desc & ~0x3FFF ull) | (new_addr >> 4);
    return new_desc;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, int transpose_A, int transpose_B) {
    uint32_t d = 0;
    d |= (1u << 4);     
    d |= (1u << 7);     
    d |= (1u << 10);    
    d |= ((transpose_A & 1) << 15);    
    d |= ((transpose_B & 1) << 16);    
    d |= ((N >> 3) << 17);     
    d |= ((M >> 4) << 24);    
    return d;
}

__device__ __forceinline__ void umma_cg1_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 p, [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void commit_mma_1sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1"
        ".mbarrier::arrive::one.shared::cta.b64"
        " [%0];"
        :: "r"(a));
}


__global__ void gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* C, int M, int N, int K) 
{
    extern __shared__ __align__(128) uint8_t smem[];
    __nv_bfloat16* smem_A = (__nv_bfloat16*)smem;
    __nv_bfloat16* smem_B = (__nv_bfloat16*)(smem + 32768);
    uint64_t* bar_A = (uint64_t*)(smem + 49152);
    uint64_t* bar_B = (uint64_t*)(smem + 49160);
    uint64_t* bar_mma = (uint64_t*)(smem + 49168);

    uint32_t tmem_c;

    int m_block = blockIdx.x * 128;
    int n_block = blockIdx.y * 64;
    int tid = threadIdx.x;

    if (tid == 0) {
        init_smem_barrier_fn(bar_A, 1);
        init_smem_barrier_fn(bar_B, 1);
        init_smem_barrier_fn(bar_mma, 1);
        tmem_alloc_fn(&tmem_c, 64);
    }
    __syncwarp();
    __syncthreads();

    // Single static base descriptors 
    uint64_t desc_A = make_smem_desc(smem_A, 0, 1024);
    uint64_t desc_B = make_smem_desc(smem_B, 0, 1024);

    // Prologue loads
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_A, 128 * 64 * sizeof(__nv_bfloat16));
        tma_load_2d_fn(&tma_A, bar_A, smem_A, 0, m_block);
        
        mbarrier_arrive_and_expect_tx_fn(bar_B, 64 * 64 * sizeof(__nv_bfloat16));
        tma_load_2d_fn(&tma_B, bar_B, smem_B, 0, n_block);
    }

    int phase_A = 0, phase_B = 0, phase_mma = 0;
    int curr = 0, next = 1;
    uint32_t idesc = make_instr_desc_fn(128, 64, 0, 1); // Transpose B only!

    for (int k_block = 0; k_block < K; k_block += 64) {
        if (k_block + 64 <= K) {
            if (tid == 0) {
                mbarrier_arrive_and_expect_tx_fn(bar_A, 128 * 64 * sizeof(__nv_bfloat16));
                tma_load_2d_fn(&tma_A, bar_A, (void*)(smem_A + next * 128 * 64), k_block + 64, m_block);

                mbarrier_arrive_and_expect_tx_fn(bar_B, 64 * 64 * sizeof(__nv_bfloat16));
                tma_load_2d_fn(&tma_B, bar_B, (void*)(smem_B + next * 64 * 64), k_block + 64, n_block);
            }
        }

        mbarrier_wait_fn(bar_A, phase_A);
        mbarrier_wait_fn(bar_B, phase_B);
        phase_A ^= 1;
        phase_B ^= 1;

        fence_proxy_async_fn();
        __syncthreads();

        if (tid == 0) {
            uint64_t desc_A_curr = desc_A + (curr * (128 * 64 >> 3));
            uint64_t desc_B_curr = desc_B + (curr * (64 * 64 >> 3));

            for (int i = 0; i < 4; ++i) {
                uint32_t p = (k_block == 0 && i == 0 && curr == 0) ? 0 : 1;
                // A is K-major, advance contiguous direction by 16 elements = 32 bytes
                uint64_t da = advance_desc(desc_A_curr, i * 32);
                // B is N-major, advance strided direction logically
                uint64_t db = advance_desc(desc_B_curr, i * 128); 
                umma_cg1_fn(tmem_c, da, db, idesc, p);
            }
            commit_mma_1sm_fn(bar_mma);
        }
        mbarrier_wait_fn(bar_mma, phase_mma);
        phase_mma ^= 1;

        curr ^= 1;
        next ^= 1;
    }
    
    // Wait for final outstanding asynchronous operations
    mbarrier_wait_fn(bar_mma, phase_mma);
    
    // Direct Epilogue writing from TMEM to Global Mem using `tcgen05`
    __syncthreads();
    uint32_t r0, r1;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int row_base = warp_id * 32;

    for (int step = 0; step < 4; ++step) {
        int col = lane_id + step * 32;
        asm volatile(
            "tcgen05.ld.sync.aligned.16x128b.x1.b32 {%0,%1}, [%2];"
            : "=r"(r0), "=r"(r1) : "r"(tmem_c + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        int row0 = row_base + (lane_id % 16);
        int row1 = row_base + (lane_id % 16) + 16;
        float f0 = __uint_as_float(r0);
        float f1 = __uint_as_float(r1);
        
        if (m_block + row0 < M && n_block + col < N) {
            C[(m_block + row0) * N + (n_block + col)] = __float2bfloat16(f0);
        }
        if (m_block + row1 < M && n_block + col < N) {
            C[(m_block + row1) * N + (n_block + col)] = __float2bfloat16(f1);
        }
    }

    if (tid == 0) {
        tmem_dealloc_fn(tmem_c, 64);
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
  CUDA_CHECK(cudaSetDevice(A.device().device_id));

  int64_t M = A.size(0);
  const int64_t N = 7168;
  const int64_t K = 5120;

  CUtensorMap tma_A, tma_B;
  CU_CHECK(create_tma_2d_descriptor_2B(&tma_A, A.data_ptr(), K, M, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  CU_CHECK(create_tma_2d_descriptor_2B(&tma_B, B.data_ptr(), N, K, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

  int smem_size = 49200;
  cudaFuncSetAttribute(gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  dim3 grid((M + 127) / 128, N / 64); 
  dim3 block(128); 
  cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
  
  gemm_kernel<<<grid, block, smem_size, stream>>>(tma_A, tma_B, static_cast<__nv_bfloat16*>(C.data_ptr()), M, N, K);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_gemms::run);

}  // namespace tvm_ffi_gemms