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

namespace gemm_blackwell {

template <int BM, int BN, int BK>
__global__ void gemm_kernel(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int M, int N, int K) 
{
    constexpr int NTOTAL = 256;
    int tid = threadIdx.x;
    int lane_id = tid % 32;

    // Shared memory: As[BM][BK] + Bs[BN][BK]
    extern __shared__ char smem_bytes[];

    __nv_bfloat16* __restrict__ As = 
        reinterpret_cast<__nv_bfloat16*>(smem_bytes);
    __nv_bfloat16* __restrict__ Bs = 
        reinterpret_cast<__nv_bfloat16*>(smem_bytes + BM * BK * sizeof(__nv_bfloat16));

    int m_start = blockIdx.x * BM;
    int n_start = blockIdx.y * BN;

    int num_k_tiles = K / BK;

    // Each thread accumulates BN outputs
    // Thread-to-output mapping: tid -> local_m, local_n_base
    int local_m = tid / (BN / 2);
    int local_n_base = (tid % (BN / 2)) * 2;

    // Warp-level accumulators: each thread holds BN/NTOTAL output sums
    // Actually let's use warp-level: within a warp all threads share partial sums
    // Simplest: register-level accumulation per-thread
    
    __nv_bfloat16 regs_c[BN / 2];
    #pragma unroll
    for (int i = 0; i < BN / 2; i++) {
        regs_c[i] = __float2bfloat16(0.f);
    }

    for (int ki = 0; ki < num_k_tiles; ki++) {
        int kg = ki * BK;

        // ---- Load A tile [BM][BK] ----
        if (tid < BM) {
            int m = tid;
            if (m_start + m < M) {
                const __nv_bfloat16* row = A + (size_t)(m_start + m) * K + kg;
                #pragma unroll
                for (int k = 0; k < BK; k += 4) {
                    As[m * BK + k]     = row[k];
                    As[m * BK + k + 1] = row[k + 1];
                    As[m * BK + k + 2] = row[k + 2];
                    As[m * BK + k + 3] = row[k + 3];
                }
            }
        }

        // ---- Load B tile [BN][BK] ----
        // Threads 64..255: 192 threads for BN*BK = 128*16 = 2048 elements
        if (tid >= BM) {
            int btid = tid - BM;
            int nbthreads = NTOTAL - BM;
            int nelem = BN * BK;
            #pragma unroll
            for (int e = btid; e < nelem; e += nbthreads) {
                int n = e / BK;
                int k = e % BK;
                if (n_start + n < N) {
                    Bs[n * BK + k] = B[(size_t)(n_start + n) * K + kg + k];
                }
            }
        }
        __syncthreads();

        // ---- Multiply-accumulate ----
        // Each thread handles its local_m row and BN/2 columns
        if (local_m < BM && m_start + local_m < M) {
            #pragma unroll
            for (int kk = 0; kk < BK; kk++) {
                float aval = __float2bfloat16(float)(As[local_m * BK + kk]);
                #pragma unroll
                for (int th = 0; th < 2; th++) {
                    int cn = local_n_base + th;
                    float bval = __float2bfloat16(float)(Bs[cn * BK + kk]);
                    float& reg = __ushort_as_float(*reinterpret_cast<unsigned short*>(&regs_c[cn]));
                    // Use inline fma
                    regs_c[cn] = __float2bfloat16(
                        __float2bfloat16(reg) * __ushort_as_short(0) + aval * bval);
                    // Simpler: accumulate in fp32 register array
                }
            }
        }
        __syncthreads();
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id()));
    
    int64_t M = A.size(0);
    constexpr int64_t N = 7168;
    constexpr int64_t K = 5120;
    
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 16;
    constexpr int NTOTAL = 256;
    
    dim3 block(NTOTAL);
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    
    size_t smem_size = BM * BK * sizeof(__nv_bfloat16) +
                       BN * BK * sizeof(__nv_bfloat16);
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    
    gemm_kernel<BM, BN, BK><<<grid, block, smem_size, stream>>>(
        static_cast<const __nv_bfloat16*>(A.data_ptr()),
        static_cast<const __nv_bfloat16*>(B.data_ptr()),
        static_cast<__nv_bfloat16*>(C.data_ptr()),
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_blackwell::run);

} // namespace gemm_blackwell