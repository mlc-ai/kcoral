#include <cuda_bf16.h>
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
    CUresult _r = (call);                                          \
    if (_r != CUDA_SUCCESS) {                                      \
        const char* errStr;                                        \
        cuGetErrorString(_r, &errStr);                             \
        fprintf(stderr, "CU error %s at %s:%d\n",                  \
                errStr ? errStr : "unknown", __FILE__, __LINE__);  \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_gemm_cuda {

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 64;
constexpr int NUM_THREADS = 128;

CUresult create_tma_descriptor(CUtensorMap* d, void* ptr,
    uint64_t gdim0, uint64_t gdim1, uint32_t bdim0, uint32_t bdim1,
    CUtensorMapSwizzle swizzle) {
    CUtensorMapDataType dataType = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    cuuint64_t globalDim[2] = {gdim0, gdim1};
    cuuint64_t globalStrides[1] = {gdim0 * 2};
    cuuint32_t boxDim[2] = {bdim0, bdim1};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(d, dataType, 2, ptr, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

__device__ __forceinline__ void init_barrier(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(count));
}

__device__ __forceinline__ void fence_barrier_init() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void arrive_expect_tx(uint64_t* bar, uint32_t tx) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(tx) : "memory");
}

__device__ __forceinline__ void wait_barrier(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n .reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(phase));
}

__device__ __forceinline__ void tma_load(const CUtensorMap* d, uint64_t* bar,
    void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d), "r"((uint32_t)__cvta_generic_to_shared(bar)),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tmem_alloc(uint32_t* dst, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
        :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void umma(
    uint32_t tmem_d, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_d), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit(uint64_t* bar) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)));
}

__device__ __forceinline__ uint64_t make_desc(void* ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;  // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint32_t make_idesc(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    // D type = FP32
    d |= (1u << 7);    // A type = BF16
    d |= (1u << 10);   // B type = BF16
    d |= (0u << 15);   // A K-major (no transpose)
    d |= (0u << 16);   // B K-major (no transpose)
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ void tmem_load_4(uint32_t col,
    uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ void tmem_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__global__ void gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* __restrict__ C, int M, int N, int K) {

    __align__(1024) __shared__ __nv_bfloat16 A_smem[2][BM][BK];
    __align__(1024) __shared__ __nv_bfloat16 B_smem[2][BN][BK];
    __shared__ uint64_t load_bar[2];
    __shared__ uint64_t mma_bar;
    __shared__ uint32_t tmem_addr_val;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // Initialize barriers
    if (tid == 0) {
        init_barrier(&load_bar[0], 1);
        init_barrier(&load_bar[1], 1);
        init_barrier(&mma_bar, 1);
        fence_barrier_init();
    }
    __syncthreads();

    // Allocate TMEM: 128 columns for 128x128 FP32 accumulator
    if (tid < 32) {
        tmem_alloc(&tmem_addr_val, 128);
    }
    __syncthreads();
    uint32_t tmem_c = tmem_addr_val;

    int bm = blockIdx.x * BM;
    int bn = blockIdx.y * BN;
    int k_tiles = K / BK;
    uint32_t idesc = make_idesc(BM, BN);
    uint32_t load_bytes = (uint32_t)(BM * BK * 2 + BN * BK * 2);

    // Double-buffered pipeline
    int buf = 0;
    uint32_t phase_load[2] = {0, 0};
    uint32_t phase_mma = 0;

    // Prologue: issue first TMA load
    if (tid == 0) {
        arrive_expect_tx(&load_bar[buf], load_bytes);
        tma_load(&tma_A, &load_bar[buf], &A_smem[buf][0][0], 0, bm);
        tma_load(&tma_B, &load_bar[buf], &B_smem[buf][0][0], 0, bn);
    }

    uint32_t accum_flag = 0;

    for (int kt = 0; kt < k_tiles; kt++) {
        // Wait for current buffer's TMA load to complete
        wait_barrier(&load_bar[buf], phase_load[buf]);
        phase_load[buf] ^= 1;

        // Issue next TMA load (overlapped with MMA computation)
        int next_buf = 1 - buf;
        if (kt + 1 < k_tiles) {
            if (tid == 0) {
                arrive_expect_tx(&load_bar[next_buf], load_bytes);
                tma_load(&tma_A, &load_bar[next_buf], &A_smem[next_buf][0][0],
                         (kt + 1) * BK, bm);
                tma_load(&tma_B, &load_bar[next_buf], &B_smem[next_buf][0][0],
                         (kt + 1) * BK, bn);
            }
        }

        // Issue MMAs for this tile (4 K-steps of K=16 each)
        // Single-thread semantics: only thread 0 issues
        if (tid == 0) {
            #pragma unroll
            for (int ks = 0; ks < BK / 16; ks++) {
                uint64_t desc_a = make_desc(&A_smem[buf][0][ks * 16], 1, 1024);
                uint64_t desc_b = make_desc(&B_smem[buf][0][ks * 16], 1, 1024);
                umma(tmem_c, desc_a, desc_b, idesc, accum_flag);
                accum_flag = 1;
            }
            umma_commit(&mma_bar);
        }

        // All threads wait for MMA completion
        wait_barrier(&mma_bar, phase_mma);
        phase_mma ^= 1;

        buf = next_buf;
    }

    // ===== Epilogue: TMEM -> SMEM -> Global =====
    // Reuse A_smem space for C_smem (128*128*2 = 32768 bytes)
    __nv_bfloat16* C_smem = reinterpret_cast<__nv_bfloat16*>(A_smem);

    // Phase 1: Load from TMEM, convert FP32->BF16, write to shared memory
    // Each warp handles 32 rows (warp_id * 32 + lane_id)
    for (int col = 0; col < BN; col += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4(col, &r0, &r1, &r2, &r3);
        tmem_wait_ld();
        int row = warp_id * 32 + lane_id;
        C_smem[row * BN + col + 0] = __float2bfloat16(__uint_as_float(r0));
        C_smem[row * BN + col + 1] = __float2bfloat16(__uint_as_float(r1));
        C_smem[row * BN + col + 2] = __float2bfloat16(__uint_as_float(r2));
        C_smem[row * BN + col + 3] = __float2bfloat16(__uint_as_float(r3));
    }
    __syncthreads();

    // Phase 2: Coalesced global writes from shared memory
    int num_steps = (BM + 3) / 4;
    for (int step = 0; step < num_steps; ++step) {
        int row = step * 4 + warp_id;
        if (row >= BM) continue;
        int gm_row = bm + row;
        if (gm_row >= M) continue;
        int col_start = lane_id * 4;
        int gn_col = bn + col_start;
        if (gn_col + 3 < N) {
            uint2 data = *reinterpret_cast<uint2*>(&C_smem[row * BN + col_start]);
            *reinterpret_cast<uint2*>(C + (uint64_t)gm_row * N + gn_col) = data;
        } else {
            for (int j = 0; j < 4; j++) {
                if (gn_col + j < N) {
                    C[(uint64_t)gm_row * N + gn_col + j] = C_smem[row * BN + col_start + j];
                }
            }
        }
    }

    // Deallocate TMEM
    __syncthreads();
    if (tid < 32) {
        tmem_dealloc(tmem_c, 128);
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

    // Create TMA descriptors
    // A is [M, K] row-major (K contiguous) -> K-major: globalDim=[K, M], boxDim=[BK, BM]
    CUtensorMap tma_A, tma_B;
    CU_CHECK(create_tma_descriptor(&tma_A, (void*)A_ptr, K, M, BK, BM,
        CU_TENSOR_MAP_SWIZZLE_128B));
    // B is [N, K] row-major (K contiguous) -> K-major: globalDim=[K, N], boxDim=[BK, BN]
    CU_CHECK(create_tma_descriptor(&tma_B, (void*)B_ptr, K, N, BK, BN,
        CU_TENSOR_MAP_SWIZZLE_128B));

    dim3 block(NUM_THREADS);
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    gemm_kernel<<<grid, block, 0, stream>>>(tma_A, tma_B, C_ptr, (int)M, (int)N, (int)K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_gemm_cuda::run);

}  // namespace tvm_ffi_gemm_cuda