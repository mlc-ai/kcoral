#include <cuda_bf16.h>
#include <cuda_runtime.h>
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

namespace gemm_cuda {

// Tile configuration
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int NUM_WARPS = 8;
constexpr int NUM_THREADS = NUM_WARPS * 32;  // 256
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;
constexpr int NUM_N_TILES = BN / MMA_N;  // 16

// ldmatrix: load 4 8x8 BF16 matrices from shared memory
__device__ __forceinline__ void ldmatrix_x4_b16(
    uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3, uint32_t smem_addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(*r0), "=r"(*r1), "=r"(*r2), "=r"(*r3)
        : "r"(smem_addr));
}

// ldmatrix: load 2 8x8 BF16 matrices from shared memory
__device__ __forceinline__ void ldmatrix_x2_b16(
    uint32_t* r0, uint32_t* r1, uint32_t smem_addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
        : "=r"(*r0), "=r"(*r1)
        : "r"(smem_addr));
}

// mma.sync: BF16 x BF16 -> FP32, M=16, N=8, K=16
__device__ __forceinline__ void mma_m16n8k16_bf16_f32(
    float* d0, float* d1, float* d2, float* d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
        : "=f"(*d0), "=f"(*d1), "=f"(*d2), "=f"(*d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

// cp.async: 16-byte copy from global to shared memory
__device__ __forceinline__ void cp_async_16(uint32_t smem_addr, const void* gmem_ptr) {
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

__global__ __launch_bounds__(NUM_THREADS)
void gemm_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K) {

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const int bm = blockIdx.x * BM;
    const int bn = blockIdx.y * BN;

    // Shared memory: double-buffered A and B tiles
    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* smemA[2];
    __nv_bfloat16* smemB[2];
    smemA[0] = smem;
    smemA[1] = smemA[0] + BM * BK;
    smemB[0] = smemA[1] + BM * BK;
    smemB[1] = smemB[0] + BN * BK;

    // Each warp handles 16 rows (warp_id * 16) x all 128 cols
    const int warp_row = warp_id * MMA_M;

    // Accumulators: 16 N-tiles x 4 FP32 values each
    float acc[NUM_N_TILES][4];
    #pragma unroll
    for (int i = 0; i < NUM_N_TILES; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const int num_k_tiles = K / BK;
    constexpr int A_TILE_INT4 = BM * BK / 8;  // 512
    constexpr int B_TILE_INT4 = BN * BK / 8;  // 512

    // Load first tile using cp.async
    {
        const int bk_offset = 0;
        // Load A
        #pragma unroll
        for (int i = 0; i < (A_TILE_INT4 + NUM_THREADS - 1) / NUM_THREADS; i++) {
            int idx = tid + i * NUM_THREADS;
            if (idx < A_TILE_INT4) {
                int row = idx / (BK / 8);
                int col = (idx % (BK / 8)) * 8;
                uint32_t smem_dst = __cvta_generic_to_shared(
                    &smemA[0][row * BK + col]);
                int ar = bm + row;
                if (ar < M) {
                    cp_async_16(smem_dst, &A[ar * K + bk_offset + col]);
                } else {
                    // Zero fill for out-of-bounds
                    int4 zero = make_int4(0, 0, 0, 0);
                    *reinterpret_cast<int4*>(&smemA[0][row * BK + col]) = zero;
                }
            }
        }
        // Load B
        #pragma unroll
        for (int i = 0; i < (B_TILE_INT4 + NUM_THREADS - 1) / NUM_THREADS; i++) {
            int idx = tid + i * NUM_THREADS;
            if (idx < B_TILE_INT4) {
                int row = idx / (BK / 8);
                int col = (idx % (BK / 8)) * 8;
                uint32_t smem_dst = __cvta_generic_to_shared(
                    &smemB[0][row * BK + col]);
                int br = bn + row;
                if (br < N) {
                    cp_async_16(smem_dst, &B[br * K + bk_offset + col]);
                } else {
                    int4 zero = make_int4(0, 0, 0, 0);
                    *reinterpret_cast<int4*>(&smemB[0][row * BK + col]) = zero;
                }
            }
        }
        cp_async_commit();
    }

    for (int k_step = 0; k_step < num_k_tiles; k_step++) {
        const int stage = k_step % 2;
        const int next_stage = 1 - stage;

        // Issue next tile load (overlapping with compute)
        if (k_step + 1 < num_k_tiles) {
            int bk_next = (k_step + 1) * BK;
            #pragma unroll
            for (int i = 0; i < (A_TILE_INT4 + NUM_THREADS - 1) / NUM_THREADS; i++) {
                int idx = tid + i * NUM_THREADS;
                if (idx < A_TILE_INT4) {
                    int row = idx / (BK / 8);
                    int col = (idx % (BK / 8)) * 8;
                    uint32_t smem_dst = __cvta_generic_to_shared(
                        &smemA[next_stage][row * BK + col]);
                    int ar = bm + row;
                    if (ar < M) {
                        cp_async_16(smem_dst, &A[ar * K + bk_next + col]);
                    } else {
                        int4 zero = make_int4(0, 0, 0, 0);
                        *reinterpret_cast<int4*>(&smemA[next_stage][row * BK + col]) = zero;
                    }
                }
            }
            #pragma unroll
            for (int i = 0; i < (B_TILE_INT4 + NUM_THREADS - 1) / NUM_THREADS; i++) {
                int idx = tid + i * NUM_THREADS;
                if (idx < B_TILE_INT4) {
                    int row = idx / (BK / 8);
                    int col = (idx % (BK / 8)) * 8;
                    uint32_t smem_dst = __cvta_generic_to_shared(
                        &smemB[next_stage][row * BK + col]);
                    int br = bn + row;
                    if (br < N) {
                        cp_async_16(smem_dst, &B[br * K + bk_next + col]);
                    } else {
                        int4 zero = make_int4(0, 0, 0, 0);
                        *reinterpret_cast<int4*>(&smemB[next_stage][row * BK + col]) = zero;
                    }
                }
            }
            cp_async_commit();
        }

        // Wait for current tile's cp.async to complete
        // We have at most 2 outstanding groups: current (wait) and next (just issued)
        if (k_step + 1 < num_k_tiles) {
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_group<0>();
        }
        __syncthreads();

        // Compute: 2 K-halves (K=16 each), 16 N-tiles (N=8 each)
        #pragma unroll
        for (int k_half = 0; k_half < 2; k_half++) {
            // Load A fragment (16x16 BF16) using ldmatrix.x4
            // 4 matrices of 8x8: mat0=(r0-7,c0-7), mat1=(r8-15,c0-7), mat2=(r0-7,c8-15), mat3=(r8-15,c8-15)
            int mat_idx = lane_id / 8;
            int row_in_mat = lane_id % 8;
            int a_row = warp_row + (mat_idx % 2) * 8 + row_in_mat;
            int a_col = k_half * 16 + (mat_idx / 2) * 8;
            uint32_t a_smem_addr = __cvta_generic_to_shared(
                &smemA[stage][a_row * BK + a_col]);

            uint32_t a_frag[4];
            ldmatrix_x4_b16(&a_frag[0], &a_frag[1], &a_frag[2], &a_frag[3], a_smem_addr);

            // For each N-tile
            #pragma unroll
            for (int n_idx = 0; n_idx < NUM_N_TILES; n_idx++) {
                // Load B fragment (16x8 BF16 = 2 8x8 tiles) using ldmatrix.x2
                // B is (N,K) row-major = (K,N) col-major, which is what MMA .col expects
                // mat0: K=0-7, mat1: K=8-15 (relative to k_half)
                int b_n_base = n_idx * 8;
                uint32_t b_smem_addr;
                if (lane_id < 8) {
                    // Matrix 0, row lane_id: B[n_base+lane_id][k_half*16]
                    b_smem_addr = __cvta_generic_to_shared(
                        &smemB[stage][(b_n_base + lane_id) * BK + k_half * 16]);
                } else if (lane_id < 16) {
                    // Matrix 1, row (lane_id-8): B[n_base+lane_id-8][k_half*16+8]
                    b_smem_addr = __cvta_generic_to_shared(
                        &smemB[stage][(b_n_base + lane_id - 8) * BK + k_half * 16 + 8]);
                } else {
                    // Unused but must be valid
                    b_smem_addr = __cvta_generic_to_shared(
                        &smemB[stage][b_n_base * BK + k_half * 16]);
                }

                uint32_t b_frag[2];
                ldmatrix_x2_b16(&b_frag[0], &b_frag[1], b_smem_addr);

                // MMA: D = A * B + C
                mma_m16n8k16_bf16_f32(
                    &acc[n_idx][0], &acc[n_idx][1], &acc[n_idx][2], &acc[n_idx][3],
                    a_frag[0], a_frag[1], a_frag[2], a_frag[3],
                    b_frag[0], b_frag[1],
                    acc[n_idx][0], acc[n_idx][1], acc[n_idx][2], acc[n_idx][3]);
            }
        }

        __syncthreads();  // Ensure compute done before next iteration overwrites buffer
    }

    // Epilogue: store accumulators to global memory
    // MMA output layout per thread (for 16x8 FP32 tile):
    //   acc[0] = D[row=lane/4][col=lane%4*2]
    //   acc[1] = D[row=lane/4][col=lane%4*2+1]
    //   acc[2] = D[row=lane/4+8][col=lane%4*2]
    //   acc[3] = D[row=lane/4+8][col=lane%4*2+1]
    int row0 = bm + warp_row + lane_id / 4;
    int row1 = row0 + 8;
    int col_base = bn + (lane_id % 4) * 2;

    #pragma unroll
    for (int n_idx = 0; n_idx < NUM_N_TILES; n_idx++) {
        int nc0 = col_base + n_idx * MMA_N;
        int nc1 = nc0 + 1;

        if (row0 < M && nc0 < N)
            C[row0 * N + nc0] = __float2bfloat16(acc[n_idx][0]);
        if (row0 < M && nc1 < N)
            C[row0 * N + nc1] = __float2bfloat16(acc[n_idx][1]);
        if (row1 < M && nc0 < N)
            C[row1 * N + nc0] = __float2bfloat16(acc[n_idx][2]);
        if (row1 < M && nc1 < N)
            C[row1 * N + nc1] = __float2bfloat16(acc[n_idx][3]);
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));

    const int M = (int)A.size(0);
    const int N = 7168;
    const int K = 5120;

    const __nv_bfloat16* A_ptr = static_cast<const __nv_bfloat16*>(A.data_ptr());
    const __nv_bfloat16* B_ptr = static_cast<const __nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* C_ptr = static_cast<__nv_bfloat16*>(C.data_ptr());

    const int m_tiles = (M + BM - 1) / BM;
    const int n_tiles = (N + BN - 1) / BN;
    dim3 grid(m_tiles, n_tiles, 1);
    dim3 block(NUM_THREADS, 1, 1);

    // Shared memory: 2 * (A_tile + B_tile) * sizeof(bf16)
    // = 2 * (128*32 + 128*32) * 2 = 32768 bytes
    const int smem_bytes = 2 * (BM * BK + BN * BK) * sizeof(__nv_bfloat16);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(C.device().device_type, C.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    gemm_kernel<<<grid, block, smem_bytes, stream>>>(A_ptr, B_ptr, C_ptr, M, N, K);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_cuda::run);

}  // namespace gemm_cuda