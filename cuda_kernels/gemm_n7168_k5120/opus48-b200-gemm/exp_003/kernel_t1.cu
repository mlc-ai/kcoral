#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <cstdio>
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

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int THREADS = 256;
constexpr int WARP_N = 4;   // warps along N
constexpr int WARP_M = 2;   // warps along M
constexpr int ASTRIDE = BK + 8;   // padded shared stride (elements)
constexpr int WT_M = 4;     // 64/16 m-subtiles per warp
constexpr int WT_N = 4;     // 32/8 n-subtiles per warp

__device__ __forceinline__ uint32_t cvta_s(const void* p){
    return (uint32_t)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ void load_tile(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16 (*As)[BM][ASTRIDE],
    __nv_bfloat16 (*Bs)[BN][ASTRIDE],
    int kt, int buf, int m0, int n0, int M, int K, int tid)
{
    // A tile: [BM][BK]
    for (int i = tid; i < BM*(BK/8); i += THREADS){
        int row = i / (BK/8);
        int col = (i % (BK/8)) * 8;
        int grow = m0 + row;
        __nv_bfloat16* dst = &As[buf][row][col];
        if (grow < M){
            const __nv_bfloat16* src = A + (size_t)grow*K + (size_t)kt*BK + col;
            asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"
                :: "r"(cvta_s(dst)), "l"(src));
        } else {
            *reinterpret_cast<float4*>(dst) = make_float4(0.f,0.f,0.f,0.f);
        }
    }
    // B tile: [BN][BK]  (N always in range for this problem)
    for (int i = tid; i < BN*(BK/8); i += THREADS){
        int row = i / (BK/8);
        int col = (i % (BK/8)) * 8;
        int grow = n0 + row;
        __nv_bfloat16* dst = &Bs[buf][row][col];
        const __nv_bfloat16* src = B + (size_t)grow*K + (size_t)kt*BK + col;
        asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"
            :: "r"(cvta_s(dst)), "l"(src));
    }
}

__global__ __launch_bounds__(256) void gemm_kernel_fn(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K)
{
    __shared__ __align__(16) __nv_bfloat16 As[2][BM][ASTRIDE];
    __shared__ __align__(16) __nv_bfloat16 Bs[2][BN][ASTRIDE];

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int warp_m = warp / WARP_N;
    int warp_n = warp % WARP_N;

    int m0 = blockIdx.y * BM;
    int n0 = blockIdx.x * BN;

    float acc[WT_M][WT_N][4];
    #pragma unroll
    for(int i=0;i<WT_M;i++)
        #pragma unroll
        for(int j=0;j<WT_N;j++)
            #pragma unroll
            for(int k=0;k<4;k++) acc[i][j][k]=0.f;

    int nkt = K / BK;

    // prologue
    load_tile(A,B,As,Bs, 0, 0, m0, n0, M, K, tid);
    asm volatile("cp.async.commit_group;\n");

    for (int kt=0; kt<nkt; ++kt){
        int cur = kt & 1;
        if (kt+1 < nkt){
            load_tile(A,B,As,Bs, kt+1, (kt+1)&1, m0, n0, M, K, tid);
            asm volatile("cp.async.commit_group;\n");
            asm volatile("cp.async.wait_group 1;\n");
        } else {
            asm volatile("cp.async.wait_group 0;\n");
        }
        __syncthreads();

        #pragma unroll
        for (int kc=0; kc<BK/16; ++kc){
            int koff = kc*16;
            uint32_t Af[WT_M][4];
            uint32_t Bf[WT_N][2];
            #pragma unroll
            for(int mi=0; mi<WT_M; ++mi){
                int arow = warp_m*64 + mi*16 + (lane & 15);
                int acol = koff + ((lane>>4)&1)*8;
                uint32_t p = cvta_s(&As[cur][arow][acol]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
                    : "=r"(Af[mi][0]),"=r"(Af[mi][1]),"=r"(Af[mi][2]),"=r"(Af[mi][3])
                    : "r"(p));
            }
            #pragma unroll
            for(int ni=0; ni<WT_N; ++ni){
                int brow = warp_n*32 + ni*8 + (lane & 7);
                int bcol = koff + ((lane>>3)&1)*8;
                uint32_t p = cvta_s(&Bs[cur][brow][bcol]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2];"
                    : "=r"(Bf[ni][0]),"=r"(Bf[ni][1])
                    : "r"(p));
            }
            #pragma unroll
            for(int mi=0; mi<WT_M; ++mi){
                #pragma unroll
                for(int ni=0; ni<WT_N; ++ni){
                    float* c = acc[mi][ni];
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                        : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
                        : "r"(Af[mi][0]),"r"(Af[mi][1]),"r"(Af[mi][2]),"r"(Af[mi][3]),
                          "r"(Bf[ni][0]),"r"(Bf[ni][1]));
                }
            }
        }
        __syncthreads();
    }

    // epilogue
    int gid  = lane >> 2;
    int tid4 = lane & 3;
    #pragma unroll
    for(int mi=0; mi<WT_M; ++mi){
        #pragma unroll
        for(int ni=0; ni<WT_N; ++ni){
            float* c = acc[mi][ni];
            int m_base = m0 + warp_m*64 + mi*16;
            int n_base = n0 + warp_n*32 + ni*8;
            int r0 = m_base + gid;
            int r1 = m_base + gid + 8;
            int cc = n_base + tid4*2;
            __nv_bfloat162 v01; v01.x=__float2bfloat16(c[0]); v01.y=__float2bfloat16(c[1]);
            __nv_bfloat162 v23; v23.x=__float2bfloat16(c[2]); v23.y=__float2bfloat16(c[3]);
            if (r0 < M) *reinterpret_cast<__nv_bfloat162*>(&C[(size_t)r0*N + cc]) = v01;
            if (r1 < M) *reinterpret_cast<__nv_bfloat162*>(&C[(size_t)r1*N + cc]) = v23;
        }
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C){
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    int M = (int)A.size(0);
    int K = (int)A.size(1);
    int N = (int)B.size(0);

    const __nv_bfloat16* Aptr = static_cast<const __nv_bfloat16*>(A.data_ptr());
    const __nv_bfloat16* Bptr = static_cast<const __nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* Cptr = static_cast<__nv_bfloat16*>(C.data_ptr());

    dim3 block(THREADS);
    dim3 grid(N / BN, (M + BM - 1) / BM);

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    gemm_kernel_fn<<<grid, block, 0, stream>>>(Aptr, Bptr, Cptr, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_kernel::run);

}  // namespace gemm_kernel