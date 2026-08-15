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

namespace gemm_tn_bf16 {

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int ASTRIDE = BK + 8;   // pad to avoid smem bank conflicts (40 bf16)
constexpr int BSTRIDE = BK + 8;
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 4;
constexpr int WM = BM / WARPS_M;  // 64
constexpr int WN = BN / WARPS_N;  // 32
constexpr int MT = WM / 16;       // 4 m-tiles per warp
constexpr int NT = WN / 8;        // 4 n-tiles per warp
constexpr int NTHREADS = 256;

__device__ __forceinline__ uint32_t ld_u32(const __nv_bfloat16* p) {
    return *reinterpret_cast<const uint32_t*>(p);
}

__device__ __forceinline__ void cp_async_cg(void* smem, const void* gmem) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(s), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}
template<int N> __device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void mma_m16n8k16(float* d, const uint32_t* a, const uint32_t* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void load_tiles(
    __nv_bfloat16* As_buf, __nv_bfloat16* Bs_buf,
    const __nv_bfloat16* A, const __nv_bfloat16* B,
    int block_m, int block_n, int kt, int M, int K) {
    int tid = threadIdx.x;
    // A tile: BM rows x BK cols, K-contiguous
    #pragma unroll
    for (int i = tid; i < BM * BK / 8; i += NTHREADS) {
        int row = i >> 2;      // BK/8 = 4 int4 per row
        int c4  = i & 3;
        int col = c4 * 8;
        int grow = block_m + row;
        int4* dst = (int4*)&As_buf[row * ASTRIDE + col];
        if (grow < M) {
            const int4* src = (const int4*)&A[(size_t)grow * K + kt + col];
            cp_async_cg(dst, src);
        } else {
            *dst = make_int4(0, 0, 0, 0);
        }
    }
    // B tile: BN rows x BK cols, K-contiguous  (N divisible by BN)
    #pragma unroll
    for (int i = tid; i < BN * BK / 8; i += NTHREADS) {
        int row = i >> 2;
        int c4  = i & 3;
        int col = c4 * 8;
        int grow = block_n + row;
        int4* dst = (int4*)&Bs_buf[row * BSTRIDE + col];
        const int4* src = (const int4*)&B[(size_t)grow * K + kt + col];
        cp_async_cg(dst, src);
    }
}

__global__ __launch_bounds__(NTHREADS, 2)
void gemm_kernel(const __nv_bfloat16* __restrict__ A,
                 const __nv_bfloat16* __restrict__ B,
                 __nv_bfloat16* __restrict__ C,
                 int M, int N, int K) {
    __shared__ __nv_bfloat16 As[2][BM * ASTRIDE];
    __shared__ __nv_bfloat16 Bs[2][BN * BSTRIDE];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int groupID = lane >> 2;   // 0..7
    const int tidg = lane & 3;       // 0..3
    const int warp_row_base = (warp_id / WARPS_N) * WM;
    const int warp_col_base = (warp_id % WARPS_N) * WN;
    const int block_m = blockIdx.y * BM;
    const int block_n = blockIdx.x * BN;

    float acc[MT][NT][4];
    #pragma unroll
    for (int m = 0; m < MT; m++)
        #pragma unroll
        for (int n = 0; n < NT; n++)
            #pragma unroll
            for (int e = 0; e < 4; e++) acc[m][n][e] = 0.f;

    const int numK = K / BK;
    int buf = 0;

    // prologue
    load_tiles(As[0], Bs[0], A, B, block_m, block_n, 0, M, K);
    cp_commit();

    for (int k = 0; k < numK; k++) {
        if (k + 1 < numK) {
            load_tiles(As[buf ^ 1], Bs[buf ^ 1], A, B, block_m, block_n, (k + 1) * BK, M, K);
            cp_commit();
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();

        #pragma unroll
        for (int ks = 0; ks < BK / 16; ks++) {
            int base_c = ks * 16;
            uint32_t af[MT][4];
            #pragma unroll
            for (int m = 0; m < MT; m++) {
                int mr0 = warp_row_base + m * 16;
                const __nv_bfloat16* ap  = &As[buf][(mr0 + groupID)     * ASTRIDE + base_c + tidg * 2];
                const __nv_bfloat16* ap8 = &As[buf][(mr0 + groupID + 8) * ASTRIDE + base_c + tidg * 2];
                af[m][0] = ld_u32(ap);
                af[m][1] = ld_u32(ap8);
                af[m][2] = ld_u32(ap + 8);
                af[m][3] = ld_u32(ap8 + 8);
            }
            uint32_t bf[NT][2];
            #pragma unroll
            for (int n = 0; n < NT; n++) {
                int nc0 = warp_col_base + n * 8;
                const __nv_bfloat16* bp = &Bs[buf][(nc0 + groupID) * BSTRIDE + base_c + tidg * 2];
                bf[n][0] = ld_u32(bp);
                bf[n][1] = ld_u32(bp + 8);
            }
            #pragma unroll
            for (int m = 0; m < MT; m++)
                #pragma unroll
                for (int n = 0; n < NT; n++)
                    mma_m16n8k16(acc[m][n], af[m], bf[n]);
        }
        __syncthreads();
        buf ^= 1;
    }

    // epilogue
    #pragma unroll
    for (int m = 0; m < MT; m++) {
        #pragma unroll
        for (int n = 0; n < NT; n++) {
            int r0 = block_m + warp_row_base + m * 16 + groupID;
            int r1 = r0 + 8;
            int col = block_n + warp_col_base + n * 8 + tidg * 2;
            float* c = acc[m][n];
            if (r0 < M) {
                *(__nv_bfloat162*)&C[(size_t)r0 * N + col] = __floats2bfloat162_rn(c[0], c[1]);
            }
            if (r1 < M) {
                *(__nv_bfloat162*)&C[(size_t)r1 * N + col] = __floats2bfloat162_rn(c[2], c[3]);
            }
        }
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0);

    const __nv_bfloat16* Ap = static_cast<const __nv_bfloat16*>(A.data_ptr());
    const __nv_bfloat16* Bp = static_cast<const __nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* Cp = static_cast<__nv_bfloat16*>(C.data_ptr());

    if (M == 0) return;

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(NTHREADS);

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    gemm_kernel<<<grid, block, 0, stream>>>(Ap, Bp, Cp, (int)M, (int)N, (int)K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_tn_bf16::run);

}  // namespace gemm_tn_bf16