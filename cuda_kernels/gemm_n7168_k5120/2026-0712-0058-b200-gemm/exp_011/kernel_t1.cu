#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>
#include <mma>

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

#define CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE 0
#define CU_TENSOR_MAP_INTERLEAVE_NONE 0

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

__global__ void gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* C, int M, int N, int K) 
{
    extern __shared__ __align__(128) uint8_t smem[];
    __nv_bfloat16* smem_A = (__nv_bfloat16*)smem;
    __nv_bfloat16* smem_B = (__nv_bfloat16*)(smem + 16384);
    float* smem_out_f32 = (float*)(smem + 32768);
    uint64_t* bar_A = (uint64_t*)(smem + 49152);
    uint64_t* bar_B = (uint64_t*)(smem + 49160);

    int m_block = blockIdx.x * 64;
    int n_block = blockIdx.y * 64;
    int tid = threadIdx.x;

    if (tid == 0) {
        init_smem_barrier_fn(bar_A, 1);
        init_smem_barrier_fn(bar_B, 1);
    }
    __syncthreads();

    // Prologue loads
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_A, 64 * 64 * sizeof(__nv_bfloat16));
        tma_load_2d_fn(&tma_A, bar_A, smem_A, 0, m_block);
        
        mbarrier_arrive_and_expect_tx_fn(bar_B, 64 * 64 * sizeof(__nv_bfloat16));
        tma_load_2d_fn(&tma_B, bar_B, smem_B, 0, n_block);
    }

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 64, 64, 64, float> acc_C;
    nvcuda::wmma::fill_fragment(acc_C, 0.0f);

    int phase_A = 0, phase_B = 0;
    int curr = 0, next = 1;

    for (int k_block = 0; k_block < K; k_block += 64) {
        // Software pipeline issuance
        if (k_block + 64 < K) {
            if (tid == 0) {
                mbarrier_arrive_and_expect_tx_fn(bar_A, 64 * 64 * sizeof(__nv_bfloat16));
                tma_load_2d_fn(&tma_A, bar_A, (void*)(smem_A + next * 64 * 64), k_block + 64, m_block);

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

        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 64, 64, 16, __nv_bfloat16, nvcuda::wmma::row_major> A_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 64, 64, 16, __nv_bfloat16, nvcuda::wmma::mem_col_major> B_frag;

        __nv_bfloat16* pA = smem_A + curr * 64 * 64;
        __nv_bfloat16* pB = smem_B + curr * 64 * 64;

        for (int i = 0; i < 4; ++i) {
            nvcuda::wmma::load_matrix_sync(A_frag, pA + i * 16, 64, 8, 8);
            nvcuda::wmma::load_matrix_sync(B_frag, pB + i * 16, 64, 8, 8);
            nvcuda::wmma::mma_sync(acc_C, A_frag, B_frag, acc_C);
        }

        curr ^= 1;
        next ^= 1;
        __syncthreads();
    }

    nvcuda::wmma::store_matrix_sync(smem_out_f32, acc_C, 64, nvcuda::wmma::mem_row_major);
    __syncthreads();

    for (int i = tid; i < 64 * 64; i += 128) {
        int row = i / 64;
        int col = i % 64;
        if (m_block + row < M) {
            C[(m_block + row) * N + (n_block + col)] = __float2bfloat16(smem_out_f32[row * 64 + col]);
        }
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
  CUDA_CHECK(cudaSetDevice(A.device().device_id));

  int64_t M = A.size(0);
  const int64_t N = 7168;
  const int64_t K = 5120;

  CUtensorMap tma_A, tma_B;
  CU_CHECK(create_tma_2d_descriptor_2B(&tma_A, A.data_ptr(), K, M, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  CU_CHECK(create_tma_2d_descriptor_2B(&tma_B, B.data_ptr(), K, N, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

  int smem_size = 49168;
  cudaFuncSetAttribute(gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  dim3 grid((M + 63) / 64, N / 64); 
  dim3 block(128); 
  cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
  
  gemm_kernel<<<grid, block, smem_size, stream>>>(tma_A, tma_B, static_cast<__nv_bfloat16*>(C.data_ptr()), M, N, K);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_gemms::run);

}  // namespace tvm_ffi_gemms