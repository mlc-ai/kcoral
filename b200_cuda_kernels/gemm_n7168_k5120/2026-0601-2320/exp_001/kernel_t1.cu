#include <cuda_bf16.h>
#include <cuda_runtime.h>
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

namespace gemm_kernel_ns {

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 4;
constexpr int WM = BM / WARPS_M; // 64
constexpr int WN = BN / WARPS_N; // 32
constexpr int MT = WM / 16;      // 4
constexpr int NT = WN / 8;       // 4
constexpr int STAGES = 3;
constexpr int ATILE = BM * BK;   // 4096 bf16
constexpr int BTILE = BN * BK;   // 4096 bf16

__device__ __forceinline__ uint32_t ldu32(const __nv_bfloat16* p) {
    return *reinterpret_cast<const uint32_t*>(p);
}
__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template<int N> __device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}
__device__ __forceinline__ void mma16816(float* d,
        uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,
        uint32_t b0,uint32_t b1) {
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
      : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}

__global__ void __launch_bounds__(256, 2)
gemm_kernel(const __nv_bfloat16* __restrict__ A,
            const __nv_bfloat16* __restrict__ B,
            __nv_bfloat16* __restrict__ C,
            int M, int N, int K) {
    extern __shared__ __align__(16) __nv_bfloat16 smem[];

    const int tid    = threadIdx.x;
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int warp_m = warp / WARPS_N;
    const int warp_n = warp % WARPS_N;
    const int g      = lane >> 2;        // groupID (0..7)
    const int t2     = (lane & 3) * 2;   // threadID_in_group*2

    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;
    const int NUMK = K / BK;

    float acc[MT][NT][4];
    #pragma unroll
    for (int i=0;i<MT;i++)
      #pragma unroll
      for (int j=0;j<NT;j++)
        #pragma unroll
        for (int r=0;r<4;r++) acc[i][j][r]=0.f;

    auto As = [&](int s)->__nv_bfloat16*{ return smem + s*(ATILE+BTILE); };
    auto Bs = [&](int s)->__nv_bfloat16*{ return smem + s*(ATILE+BTILE) + ATILE; };

    auto load_tile = [&](int kt, int buf){
        __nv_bfloat16* as = As(buf);
        __nv_bfloat16* bs = Bs(buf);
        #pragma unroll
        for (int it=0; it<2; ++it) {
            int ci   = tid + it*256;       // 0..511
            int row  = ci >> 2;            // 0..127
            int cgrp = (ci & 3) * 8;       // 0,8,16,24
            int gmrow = m0 + row;
            if (gmrow >= M) gmrow = M - 1;
            const __nv_bfloat16* gA = A + (size_t)gmrow*K + (size_t)kt*BK + cgrp;
            cp_async_16(as + row*BK + cgrp, gA);
            int gnrow = n0 + row;          // N is multiple of BN
            const __nv_bfloat16* gB = B + (size_t)gnrow*K + (size_t)kt*BK + cgrp;
            cp_async_16(bs + row*BK + cgrp, gB);
        }
    };

    auto compute = [&](int buf){
        __nv_bfloat16* as = As(buf);
        __nv_bfloat16* bs = Bs(buf);
        #pragma unroll
        for (int kk=0; kk<BK; kk+=16) {
            uint32_t af[MT][4];
            uint32_t bf[NT][2];
            #pragma unroll
            for (int mt=0; mt<MT; ++mt) {
                int mrow = warp_m*WM + mt*16;
                af[mt][0] = ldu32(as + (mrow+g)*BK   + kk     + t2);
                af[mt][1] = ldu32(as + (mrow+g+8)*BK + kk     + t2);
                af[mt][2] = ldu32(as + (mrow+g)*BK   + kk + 8 + t2);
                af[mt][3] = ldu32(as + (mrow+g+8)*BK + kk + 8 + t2);
            }
            #pragma unroll
            for (int nt=0; nt<NT; ++nt) {
                int ncol = warp_n*WN + nt*8;
                bf[nt][0] = ldu32(bs + (ncol+g)*BK + kk     + t2);
                bf[nt][1] = ldu32(bs + (ncol+g)*BK + kk + 8 + t2);
            }
            #pragma unroll
            for (int mt=0; mt<MT; ++mt)
              #pragma unroll
              for (int nt=0; nt<NT; ++nt)
                mma16816(acc[mt][nt], af[mt][0],af[mt][1],af[mt][2],af[mt][3],
                         bf[nt][0], bf[nt][1]);
        }
    };

    // Prologue: prime STAGES-1 stages
    #pragma unroll
    for (int s=0; s<STAGES-1; ++s) {
        if (s < NUMK) load_tile(s, s);
        cp_commit();
    }

    int read = 0;
    for (int kt=0; kt<NUMK; ++kt) {
        cp_wait<STAGES-2>();
        __syncthreads();
        compute(read);
        __syncthreads();
        int li = kt + (STAGES-1);
        if (li < NUMK) load_tile(li, li % STAGES);
        cp_commit();
        read = (read + 1) % STAGES;
    }

    // Epilogue: write C
    #pragma unroll
    for (int mt=0; mt<MT; ++mt) {
        #pragma unroll
        for (int nt=0; nt<NT; ++nt) {
            int mrow = m0 + warp_m*WM + mt*16;
            int ncol = n0 + warp_n*WN + nt*8;
            int r0 = mrow + g;
            int r1 = mrow + g + 8;
            int c0 = ncol + t2;
            float* d = acc[mt][nt];
            if (r0 < M) {
                __nv_bfloat162 v = __floats2bfloat162_rn(d[0], d[1]);
                *reinterpret_cast<__nv_bfloat162*>(&C[(size_t)r0*N + c0]) = v;
            }
            if (r1 < M) {
                __nv_bfloat162 v = __floats2bfloat162_rn(d[2], d[3]);
                *reinterpret_cast<__nv_bfloat162*>(&C[(size_t)r1*N + c0]) = v;
            }
        }
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    int M = (int)A.size(0);
    int K = (int)A.size(1);
    int N = (int)B.size(0);

    const __nv_bfloat16* a = static_cast<const __nv_bfloat16*>(A.data_ptr());
    const __nv_bfloat16* b = static_cast<const __nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* c       = static_cast<__nv_bfloat16*>(C.data_ptr());

    dim3 grid(N / BN, (M + BM - 1) / BM);
    dim3 block(256);
    int smem_bytes = STAGES * (ATILE + BTILE) * (int)sizeof(__nv_bfloat16);

    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(gemm_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        attr_set = true;
    }

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    gemm_kernel<<<grid, block, smem_bytes, stream>>>(a, b, c, M, N, K);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_kernel_ns::run);

}  // namespace gemm_kernel_ns