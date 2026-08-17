#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                       \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace gemm_cuda {

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 64;
constexpr int SMEM_K = BK + 8;
constexpr int WM = 64;
constexpr int WN = 64;
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;
constexpr int NUM_M = WM / MMA_M;
constexpr int NUM_N = WN / MMA_N;
constexpr int NUM_K = BK / MMA_K;
constexpr int SMEM_BYTES = 2 * (BM * SMEM_K + BN * SMEM_K) * sizeof(__nv_bfloat16);

__device__ __forceinline__ void cp_async_16B(void* smem, const void* gmem) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(addr), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}
template<int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void ldmatrix_x4(
    uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2_trans(
    uint32_t& b0, uint32_t& b1, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(b0), "=r"(b1) : "r"(addr));
}

__device__ __forceinline__ void mma_m16n8k16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

__device__ __forceinline__ void load_tile(
    __nv_bfloat16 (*As)[BM][SMEM_K],
    __nv_bfloat16 (*Bs)[BN][SMEM_K],
    int buf, int k_base,
    int m_base, int n_base, int M, int N, int K,
    int tid,
    const __nv_bfloat16* A, const __nv_bfloat16* B) {
    #pragma unroll
    for (int rd = 0; rd < 8; rd++) {
        int row = rd * 16 + tid / 8;
        int col = (tid % 8) * 8;
        int gm = m_base + row;
        int gn = n_base + row;
        int gk = k_base + col;

        if (gm < M) {
            cp_async_16B(&As[buf][row][col], &A[(size_t)gm * K + gk]);
        } else {
            *reinterpret_cast<int4*>(&As[buf][row][col]) = make_int4(0, 0, 0, 0);
        }
        if (gn < N) {
            cp_async_16B(&Bs[buf][row][col], &B[(size_t)gn * K + gk]);
        } else {
            *reinterpret_cast<int4*>(&Bs[buf][row][col]) = make_int4(0, 0, 0, 0);
        }
    }
    cp_async_commit();
}

__global__ __launch_bounds__(128, 1)
void gemm_kernel(const __nv_bfloat16* __restrict__ A,
                 const __nv_bfloat16* __restrict__ B,
                 __nv_bfloat16* __restrict__ C,
                 int M, int N, int K) {
    extern __shared__ __align__(16) char smem_raw[];
    __nv_bfloat16 (*As)[BM][SMEM_K] = reinterpret_cast<decltype(As)>(smem_raw);
    __nv_bfloat16 (*Bs)[BN][SMEM_K] = reinterpret_cast<decltype(Bs)>(
        smem_raw + 2 * BM * SMEM_K * sizeof(__nv_bfloat16));

    int bm = blockIdx.x;
    int bn = blockIdx.y;
    int m_base = bm * BM;
    int n_base = bn * BN;

    int tid = threadIdx.x;
    int warp = tid / 32;
    int lane = tid % 32;
    int warp_m = (warp / 2) * WM;
    int warp_n = (warp % 2) * WN;

    // For ldmatrix.x4 on A (row-major):
    // Thread t: g = t/8, r = t%8
    // addr = &As[a_row + r + (g/2)*8][k_off + (g%2)*8]
    int g_lane = lane / 8;
    int r_lane = lane % 8;
    int a_row_off = (g_lane / 2) * 8;
    int a_col_off = (g_lane % 2) * 8;

    float acc[NUM_M][NUM_N][4];
    #pragma unroll
    for (int i = 0; i < NUM_M; i++)
        #pragma unroll
        for (int j = 0; j < NUM_N; j++) {
            acc[i][j][0] = 0.f;
            acc[i][j][1] = 0.f;
            acc[i][j][2] = 0.f;
            acc[i][j][3] = 0.f;
        }

    int num_k = K / BK;

    load_tile(As, Bs, 0, 0, m_base, n_base, M, N, K, tid, A, B);

    for (int kt = 0; kt < num_k; kt++) {
        if (kt + 1 < num_k) {
            load_tile(As, Bs, (kt + 1) % 2, (kt + 1) * BK,
                      m_base, n_base, M, N, K, tid, A, B);
        }

        if (kt + 1 < num_k) {
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_group<0>();
        }
        __syncthreads();

        int buf = kt % 2;

        #pragma unroll
        for (int ks = 0; ks < NUM_K; ks++) {
            int k_off = ks * MMA_K;

            // Load A fragments with ldmatrix.x4
            uint32_t a_frag[NUM_M][4];
            #pragma unroll
            for (int mi = 0; mi < NUM_M; mi++) {
                int a_row = warp_m + mi * MMA_M;
                uint32_t addr = (uint32_t)__cvta_generic_to_shared(
                    &As[buf][a_row + r_lane + a_row_off][k_off + a_col_off]);
                ldmatrix_x4(a_frag[mi][0], a_frag[mi][1],
                            a_frag[mi][2], a_frag[mi][3], addr);
            }

            // Load B fragments with ldmatrix.x2.trans and issue MMA
            #pragma unroll
            for (int nj = 0; nj < NUM_N; nj++) {
                int n_base_w = warp_n + nj * MMA_N;

                // ldmatrix.x2.trans: threads 0-7 provide addr for matrix 0,
                // threads 8-15 provide addr for matrix 1
                uint32_t b_addr;
                if (lane < 8) {
                    b_addr = (uint32_t)__cvta_generic_to_shared(
                        &Bs[buf][n_base_w + lane][k_off]);
                } else if (lane < 16) {
                    b_addr = (uint32_t)__cvta_generic_to_shared(
                        &Bs[buf][n_base_w + (lane - 8)][k_off + 8]);
                } else {
                    b_addr = (uint32_t)__cvta_generic_to_shared(&Bs[buf][0][0]);
                }

                uint32_t b0, b1;
                ldmatrix_x2_trans(b0, b1, b_addr);

                #pragma unroll
                for (int mi = 0; mi < NUM_M; mi++) {
                    mma_m16n8k16(
                        acc[mi][nj][0], acc[mi][nj][1],
                        acc[mi][nj][2], acc[mi][nj][3],
                        a_frag[mi][0], a_frag[mi][1],
                        a_frag[mi][2], a_frag[mi][3],
                        b0, b1,
                        acc[mi][nj][0], acc[mi][nj][1],
                        acc[mi][nj][2], acc[mi][nj][3]);
                }
            }
        }
    }

    // Epilogue: D output layout for mma.m16n8k16
    // d0 = D[t/4][2*(t%4)], d1 = D[t/4][2*(t%4)+1]
    // d2 = D[t/4+8][2*(t%4)], d3 = D[t/4+8][2*(t%4)+1]
    #pragma unroll
    for (int mi = 0; mi < NUM_M; mi++) {
        #pragma unroll
        for (int nj = 0; nj < NUM_N; nj++) {
            int row0 = m_base + warp_m + mi * MMA_M + lane / 4;
            int row1 = row0 + 8;
            int col0 = n_base + warp_n + nj * MMA_N + 2 * (lane % 4);

            __nv_bfloat162 c01 = __floats2bfloat162_rn(acc[mi][nj][0], acc[mi][nj][1]);
            __nv_bfloat162 c23 = __floats2bfloat162_rn(acc[mi][nj][2], acc[mi][nj][3]);

            if (row0 < M) {
                *reinterpret_cast<__nv_bfloat162*>(&C[(size_t)row0 * N + col0]) = c01;
            }
            if (row1 < M) {
                *reinterpret_cast<__nv_bfloat162*>(&C[(size_t)row1 * N + col0]) = c23;
            }
        }
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));

    int M = (int)A.size(0);
    int K = (int)A.size(1);
    int N = (int)B.size(0);

    const __nv_bfloat16* A_ptr = static_cast<const __nv_bfloat16*>(A.data_ptr());
    const __nv_bfloat16* B_ptr = static_cast<const __nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* C_ptr = static_cast<__nv_bfloat16*>(C.data_ptr());

    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    dim3 block(128);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    cudaFuncSetAttribute(gemm_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);

    gemm_kernel<<<grid, block, SMEM_BYTES, stream>>>(A_ptr, B_ptr, C_ptr, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_cuda::run);

}  // namespace gemm_cuda