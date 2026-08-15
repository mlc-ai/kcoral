#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
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

namespace tvm_ffi_gemm_cuda {

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 16;

__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(smem_addr), "l"(gmem) : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void load_tile_async(
    const __nv_bfloat16* A, const __nv_bfloat16* B,
    __nv_bfloat16* A_smem_buf, __nv_bfloat16* B_smem_buf,
    int k_start, int M, int N, int K) {
    
    int tid = threadIdx.x;
    int row = tid / 2;
    int col = (tid % 2) * 8;
    
    // Load A tile [BM][BK]
    {
        int gm_row = blockIdx.x * BM + row;
        int gm_col = k_start + col;
        if (gm_row < M) {
            const __nv_bfloat16* src = A + (uint64_t)gm_row * K + gm_col;
            __nv_bfloat16* dst = A_smem_buf + row * BK + col;
            cp_async_16(dst, src);
        }
    }
    // Load B tile [BN][BK] (B is N x K, we store as [N][K])
    {
        int gm_row = blockIdx.y * BN + row;
        int gm_col = k_start + col;
        if (gm_row < N) {
            const __nv_bfloat16* src = B + (uint64_t)gm_row * K + gm_col;
            __nv_bfloat16* dst = B_smem_buf + row * BK + col;
            cp_async_16(dst, src);
        }
    }
}

__global__ void gemm_kernel(
    const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
    int M, int N, int K) {
    
    extern __shared__ char smem[];
    __nv_bfloat16* A_smem = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* B_smem = A_smem + 2 * BM * BK;
    float* C_smem = reinterpret_cast<float*>(B_smem + 2 * BN * BK);
    
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_m = warp_id % 2;  // 0 or 1
    int warp_n = warp_id / 2;  // 0,1,2,3
    int warp_m_base = warp_m * 64;
    int warp_n_base = warp_n * 32;
    
    // Zero shared memory for A and B (handles out-of-bounds rows)
    for (int i = tid; i < 2 * BM * BK; i += blockDim.x) {
        A_smem[i] = __float2bfloat16(0.0f);
    }
    for (int i = tid; i < 2 * BN * BK; i += blockDim.x) {
        B_smem[i] = __float2bfloat16(0.0f);
    }
    __syncthreads();
    
    // WMMA fragments
    using namespace nvcuda::wmma;
    fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a_frag[4][2];
    fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> b_frag[4][2];
    fragment<accumulator, 16, 16, 16, float> c_frag[4][2];
    
    #pragma unroll
    for (int m = 0; m < 4; m++) {
        #pragma unroll
        for (int n = 0; n < 2; n++) {
            fill_fragment(c_frag[m][n], 0.0f);
        }
    }
    
    // Issue first load
    load_tile_async(A, B, A_smem, B_smem, 0, M, N, K);
    cp_async_commit();
    
    // Main K loop
    for (int k = 0; k < K; k += BK) {
        int buf = (k / BK) % 2;
        
        // Issue next load if not last
        if (k + BK < K) {
            load_tile_async(A, B, A_smem + (1 - buf) * BM * BK, B_smem + (1 - buf) * BN * BK,
                            k + BK, M, N, K);
            cp_async_commit();
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_group<0>();
        }
        __syncthreads();
        
        // Compute
        #pragma unroll
        for (int m = 0; m < 4; m++) {
            #pragma unroll
            for (int n = 0; n < 2; n++) {
                load_matrix_sync(a_frag[m][n],
                    A_smem + buf * BM * BK + (warp_m_base + m * 16) * BK, BK);
                load_matrix_sync(b_frag[m][n],
                    B_smem + buf * BN * BK + (warp_n_base + n * 16) * BK, BK);
                mma_sync(c_frag[m][n], a_frag[m][n], b_frag[m][n], c_frag[m][n]);
            }
        }
    }
    
    // Store accumulators to C_smem (FP32)
    #pragma unroll
    for (int m = 0; m < 4; m++) {
        #pragma unroll
        for (int n = 0; n < 2; n++) {
            int row = warp_m_base + m * 16;
            int col = warp_n_base + n * 16;
            store_matrix_sync(C_smem + row * BN + col, c_frag[m][n], BN, mem_row_major);
        }
    }
    __syncthreads();
    
    // Convert FP32 to BF16 and write to global memory
    for (int i = tid * 4; i < BM * BN; i += blockDim.x * 4) {
        int row = i / BN;
        int col = i % BN;
        int gm_row = blockIdx.x * BM + row;
        int gn_col = blockIdx.y * BN + col;
        if (gm_row < M && gn_col + 3 < N) {
            float f0 = C_smem[row * BN + col];
            float f1 = C_smem[row * BN + col + 1];
            float f2 = C_smem[row * BN + col + 2];
            float f3 = C_smem[row * BN + col + 3];
            __nv_bfloat16 b0 = __float2bfloat16(f0);
            __nv_bfloat16 b1 = __float2bfloat16(f1);
            __nv_bfloat16 b2 = __float2bfloat16(f2);
            __nv_bfloat16 b3 = __float2bfloat16(f3);
            uint16_t h0 = *reinterpret_cast<uint16_t*>(&b0);
            uint16_t h1 = *reinterpret_cast<uint16_t*>(&b1);
            uint16_t h2 = *reinterpret_cast<uint16_t*>(&b2);
            uint16_t h3 = *reinterpret_cast<uint16_t*>(&b3);
            uint32_t p0 = (uint32_t(h1) << 16) | h0;
            uint32_t p1 = (uint32_t(h3) << 16) | h2;
            uint2 out = make_uint2(p0, p1);
            *reinterpret_cast<uint2*>(C + (uint64_t)gm_row * N + gn_col) = out;
        } else {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                if (gm_row < M && gn_col + j < N) {
                    C[(uint64_t)gm_row * N + gn_col + j] =
                        __float2bfloat16(C_smem[row * BN + col + j]);
                }
            }
        }
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0);
    
    const __nv_bfloat16* A_ptr = static_cast<const __nv_bfloat16*>(A.data_ptr());
    const __nv_bfloat16* B_ptr = static_cast<const __nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* C_ptr = static_cast<__nv_bfloat16*>(C.data_ptr());
    
    dim3 block(256);
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    int smem_bytes = 2 * BM * BK * sizeof(__nv_bfloat16) +
                     2 * BN * BK * sizeof(__nv_bfloat16) +
                     BM * BN * sizeof(float);
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    
    gemm_kernel<<<grid, block, smem_bytes, stream>>>(A_ptr, B_ptr, C_ptr, (int)M, (int)N, (int)K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_gemm_cuda::run);

}  // namespace tvm_ffi_gemm_cuda