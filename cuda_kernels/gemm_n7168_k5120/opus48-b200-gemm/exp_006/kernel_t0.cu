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
        fprintf(stderr, "CUDA error %s at %s:%d\n",                \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace gemm_kernel {

using bf16 = __nv_bfloat16;

#define BM 128
#define BN 128
#define BK 32
#define WARPS_M 2
#define WARPS_N 4
#define NUM_WARPS (WARPS_M*WARPS_N)      // 8
#define NUM_THREADS (NUM_WARPS*32)       // 256
#define WM (BM/WARPS_M)                  // 64
#define WN (BN/WARPS_N)                  // 32
#define WMMA_M (WM/16)                   // 4
#define WMMA_N (WN/8)                    // 4
#define K_STEPS (BK/16)                  // 2

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
    return (uint32_t)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
    uint32_t s = smem_addr(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}
template<int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void ldm_x4(uint32_t &r0,uint32_t &r1,uint32_t &r2,uint32_t &r3,uint32_t a){
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(a));
}
__device__ __forceinline__ void ldm_x2(uint32_t &r0,uint32_t &r1,uint32_t a){
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
      : "=r"(r0),"=r"(r1) : "r"(a));
}
__device__ __forceinline__ void mma_f(float* d, const uint32_t* a, const uint32_t* b, const float* c){
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(d[0]),"=f"(d[1]),"=f"(d[2]),"=f"(d[3])
      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),
        "f"(c[0]),"f"(c[1]),"f"(c[2]),"f"(c[3]));
}

__device__ __forceinline__ void load_tile(
    bf16 (*As_buf)[BK], bf16 (*Bs_buf)[BK],
    const bf16* __restrict__ A, const bf16* __restrict__ B,
    int blockM, int blockN, int kt, int M, int K, int tid)
{
    int kbase = kt*BK;
    #pragma unroll
    for (int i = tid; i < BM*BK/8; i += NUM_THREADS) {
        int m = i / (BK/8);
        int k8 = (i % (BK/8)) * 8;
        int gm = blockM*BM + m;
        if (gm >= M) gm = M-1;
        cp_async_16(&As_buf[m][k8], A + (size_t)gm*K + kbase + k8);
    }
    #pragma unroll
    for (int i = tid; i < BN*BK/8; i += NUM_THREADS) {
        int n = i / (BK/8);
        int k8 = (i % (BK/8)) * 8;
        int gn = blockN*BN + n;
        cp_async_16(&Bs_buf[n][k8], B + (size_t)gn*K + kbase + k8);
    }
}

__global__ void __launch_bounds__(NUM_THREADS) gemm_kernel_fn(
    const bf16* __restrict__ A, const bf16* __restrict__ B, bf16* __restrict__ C,
    int M, int N, int K)
{
    __shared__ __align__(16) bf16 As[2][BM][BK];
    __shared__ __align__(16) bf16 Bs[2][BN][BK];

    int blockM = blockIdx.y;
    int blockN = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;
    int warp_m = warp_id / WARPS_N;
    int warp_n = warp_id % WARPS_N;

    int numKTiles = K / BK;

    float acc[WMMA_M][WMMA_N][4];
    #pragma unroll
    for (int i=0;i<WMMA_M;i++)
      #pragma unroll
      for(int j=0;j<WMMA_N;j++)
        #pragma unroll
        for(int t=0;t<4;t++) acc[i][j][t]=0.f;

    load_tile(As[0], Bs[0], A, B, blockM, blockN, 0, M, K, tid);
    cp_async_commit();

    for (int kt = 0; kt < numKTiles; ++kt) {
        int cur = kt & 1;
        if (kt + 1 < numKTiles) {
            load_tile(As[(kt+1)&1], Bs[(kt+1)&1], A, B, blockM, blockN, kt+1, M, K, tid);
            cp_async_commit();
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();

        #pragma unroll
        for (int ki = 0; ki < K_STEPS; ++ki) {
            uint32_t a[WMMA_M][4];
            #pragma unroll
            for (int mi = 0; mi < WMMA_M; ++mi) {
                int a_row = warp_m*WM + mi*16 + (lane%16);
                int a_col = ki*16 + (lane/16)*8;
                ldm_x4(a[mi][0],a[mi][1],a[mi][2],a[mi][3], smem_addr(&As[cur][a_row][a_col]));
            }
            uint32_t b[WMMA_N][2];
            #pragma unroll
            for (int ni = 0; ni < WMMA_N; ++ni) {
                int b_n = warp_n*WN + ni*8 + (lane%8);
                int b_k = ki*16 + ((lane/8)&1)*8;
                ldm_x2(b[ni][0], b[ni][1], smem_addr(&Bs[cur][b_n][b_k]));
            }
            #pragma unroll
            for (int mi=0; mi<WMMA_M; ++mi)
              #pragma unroll
              for (int ni=0; ni<WMMA_N; ++ni)
                mma_f(acc[mi][ni], a[mi], b[ni], acc[mi][ni]);
        }
        __syncthreads();
    }

    // epilogue
    #pragma unroll
    for (int mi=0; mi<WMMA_M; ++mi) {
        #pragma unroll
        for (int ni=0; ni<WMMA_N; ++ni) {
            int row0 = blockM*BM + warp_m*WM + mi*16 + (lane/4);
            int row1 = row0 + 8;
            int col0 = blockN*BN + warp_n*WN + ni*8 + (lane%4)*2;
            float* d = acc[mi][ni];
            if (row0 < M) {
                __nv_bfloat162 v = __floats2bfloat162_rn(d[0], d[1]);
                *reinterpret_cast<__nv_bfloat162*>(&C[(size_t)row0*N + col0]) = v;
            }
            if (row1 < M) {
                __nv_bfloat162 v = __floats2bfloat162_rn(d[2], d[3]);
                *reinterpret_cast<__nv_bfloat162*>(&C[(size_t)row1*N + col0]) = v;
            }
        }
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    int M = (int)A.size(0);
    int K = (int)A.size(1);
    int N = (int)B.size(0);
    const bf16* a = static_cast<const bf16*>(A.data_ptr());
    const bf16* b = static_cast<const bf16*>(B.data_ptr());
    bf16* c = static_cast<bf16*>(C.data_ptr());
    if (M <= 0) return;
    dim3 block(NUM_THREADS);
    dim3 grid(N/BN, (M+BM-1)/BM);
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    gemm_kernel_fn<<<grid, block, 0, stream>>>(a, b, c, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_kernel::run);

}  // namespace gemm_kernel