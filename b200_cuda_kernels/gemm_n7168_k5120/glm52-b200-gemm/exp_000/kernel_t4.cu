#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <cstdio>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using bf16 = __nv_bfloat16;

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define CU_CHECK(call) do { \
    CUresult _r = (call); \
    if (_r != CUDA_SUCCESS) { \
        const char* errStr; \
        cuGetErrorString(_r, &errStr); \
        fprintf(stderr, "CU error %s at %s:%d\n", errStr, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

namespace gemm_cuda {

constexpr int BM = 128;
constexpr int BN = 256;
constexpr int BK = 16;
constexpr int NTHREADS = 128;
constexpr int NBUFFERS = 4;
constexpr int K_ITERS = 5120 / BK;

constexpr int A_TILE_BYTES = BM * BK * sizeof(bf16);
constexpr int B_TILE_BYTES = BN * BK * sizeof(bf16);
constexpr int TILE_BYTES = A_TILE_BYTES + B_TILE_BYTES;

constexpr int EPI_CHUNK = 64;
constexpr int EPI_STRIDE = 72;
constexpr int EPI_BYTES = BM * EPI_STRIDE * sizeof(bf16);

constexpr int SMEM_A_OFF = 0;
constexpr int SMEM_B_OFF = SMEM_A_OFF + NBUFFERS * A_TILE_BYTES;
constexpr int SMEM_MBAR_LOAD_OFF = SMEM_B_OFF + NBUFFERS * B_TILE_BYTES;
constexpr int SMEM_MBAR_UMMA_OFF = SMEM_MBAR_LOAD_OFF + NBUFFERS * 8;
constexpr int SMEM_TMEM_OFF = SMEM_MBAR_UMMA_OFF + 8;
constexpr int SMEM_EPI_OFF = (SMEM_TMEM_OFF + 4 + 255) & ~255;
constexpr int SMEM_PAD = 256;
constexpr int SMEM_TOTAL = SMEM_PAD + SMEM_EPI_OFF + EPI_BYTES;

constexpr int TMEM_COLS = BN;

__device__ __forceinline__ void init_mbar(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(count));
}

__device__ __forceinline__ void fence_mbar_init() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t tx) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(tx) : "memory");
}

__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n .reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(phase));
}

__device__ __forceinline__ void tma_load_2d(const CUtensorMap* d, uint64_t* bar,
                                             void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
           "l"((uint64_t)d),
           "r"((uint32_t)__cvta_generic_to_shared(bar)),
           "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tmem_alloc(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
        :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tcgen05_fence_after() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void umma_cg1(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n .reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_cg1(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
        :: "r"(a));
}

__device__ __forceinline__ void tmem_load_4x(uint32_t col,
    uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ void tmem_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_32b(void* smem_ptr,
                                                        uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;    // version = 1
    d |= (uint64_t)6 << 61;    // 32B swizzle
    return d;
}

__device__ __forceinline__ uint32_t make_idesc(int M, int N) {
    uint32_t d = 0;
    d |= (1u << 4);            // dtype = FP32
    d |= (1u << 7);            // atype = BF16
    d |= (1u << 10);           // btype = BF16
    d |= (0u << 15);           // A K-major
    d |= (0u << 16);           // B K-major
    d |= ((N / 8) << 17);      // N dim
    d |= ((M / 16) << 24);     // M dim
    return d;
}

__global__ __launch_bounds__(NTHREADS)
void gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    bf16* __restrict__ C, int M, int N, int K) {

    extern __shared__ char smem_raw[];
    char* smem = (char*)(((uintptr_t)smem_raw + 255) & ~255);

    bf16* smemA[NBUFFERS];
    bf16* smemB[NBUFFERS];
    #pragma unroll
    for (int i = 0; i < NBUFFERS; i++) {
        smemA[i] = reinterpret_cast<bf16*>(smem + SMEM_A_OFF + i * A_TILE_BYTES);
        smemB[i] = reinterpret_cast<bf16*>(smem + SMEM_B_OFF + i * B_TILE_BYTES);
    }
    uint64_t* mbar_load = reinterpret_cast<uint64_t*>(smem + SMEM_MBAR_LOAD_OFF);
    uint64_t* mbar_umma = reinterpret_cast<uint64_t*>(smem + SMEM_MBAR_UMMA_OFF);
    uint32_t* tmem_addr_smem = reinterpret_cast<uint32_t*>(smem + SMEM_TMEM_OFF);
    bf16* smem_epi = reinterpret_cast<bf16*>(smem + SMEM_EPI_OFF);

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int bm = blockIdx.y * BM;
    const int bn = blockIdx.x * BN;

    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < NBUFFERS; i++)
            init_mbar(&mbar_load[i], 1);
        init_mbar(mbar_umma, 1);
        fence_mbar_init();
    }
    __syncthreads();

    if (warp_id == 0) {
        tmem_alloc(tmem_addr_smem, TMEM_COLS);
    }
    __syncthreads();
    uint32_t tmem_addr = *tmem_addr_smem;

    uint32_t idesc = make_idesc(BM, BN);
    constexpr uint32_t SBO = 256;
    constexpr uint32_t LBO = 1;

    #pragma unroll
    for (int i = 0; i < NBUFFERS && i < K_ITERS; i++) {
        if (tid == 0) {
            mbar_arrive_expect_tx(&mbar_load[i], TILE_BYTES);
            tma_load_2d(&tma_A, &mbar_load[i], smemA[i], i * BK, bm);
            tma_load_2d(&tma_B, &mbar_load[i], smemB[i], i * BK, bn);
        }
    }

    uint32_t phase_load[NBUFFERS] = {};
    uint32_t phase_umma = 0;

    for (int k = 0; k < K_ITERS; k++) {
        int buf = k % NBUFFERS;

        mbar_wait(&mbar_load[buf], phase_load[buf]);
        phase_load[buf] ^= 1;

        if (k >= 1) {
            mbar_wait(mbar_umma, phase_umma);
            phase_umma ^= 1;

            int next_k = k + NBUFFERS - 1;
            int next_buf = (k - 1) % NBUFFERS;
            if (next_k < K_ITERS && tid == 0) {
                mbar_arrive_expect_tx(&mbar_load[next_buf], TILE_BYTES);
                tma_load_2d(&tma_A, &mbar_load[next_buf], smemA[next_buf], next_k * BK, bm);
                tma_load_2d(&tma_B, &mbar_load[next_buf], smemB[next_buf], next_k * BK, bn);
            }
        }

        if (tid == 0) {
            uint64_t desc_a = make_smem_desc_32b(smemA[buf], LBO, SBO);
            uint64_t desc_b = make_smem_desc_32b(smemB[buf], LBO, SBO);
            umma_cg1(tmem_addr, desc_a, desc_b, idesc, k > 0 ? 1 : 0);
            umma_commit_cg1(mbar_umma);
        }
    }

    mbar_wait(mbar_umma, phase_umma);
    tcgen05_fence_after();

    constexpr int N_CHUNKS = BN / EPI_CHUNK;

    for (int chunk = 0; chunk < N_CHUNKS; chunk++) {
        int col_base = chunk * EPI_CHUNK;

        #pragma unroll
        for (int cg = 0; cg < EPI_CHUNK / 4; cg++) {
            uint32_t tmem_col = tmem_addr + col_base + cg * 4;
            uint32_t r0, r1, r2, r3;
            tmem_load_4x(tmem_col, &r0, &r1, &r2, &r3);
            tmem_wait_ld();

            int row = warp_id * 32 + lane_id;
            int scol = cg * 4;
            bf16 b0 = __float2bfloat16(__uint_as_float(r0));
            bf16 b1 = __float2bfloat16(__uint_as_float(r1));
            bf16 b2 = __float2bfloat16(__uint_as_float(r2));
            bf16 b3 = __float2bfloat16(__uint_as_float(r3));
            uint32_t p0 = (uint32_t(*(uint16_t*)&b1) << 16) | uint32_t(*(uint16_t*)&b0);
            uint32_t p1 = (uint32_t(*(uint16_t*)&b3) << 16) | uint32_t(*(uint16_t*)&b2);
            uint32_t* dst = reinterpret_cast<uint32_t*>(&smem_epi[row * EPI_STRIDE + scol]);
            dst[0] = p0;
            dst[1] = p1;
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int s = tid + i * NTHREADS;
            int row = s / 16;
            int col = (s % 16) * 4;
            int grow = bm + row;
            int gcol = bn + col_base + col;
            if (grow < M) {
                uint2 data = *reinterpret_cast<uint2*>(
                    &smem_epi[row * EPI_STRIDE + col]);
                *reinterpret_cast<uint2*>(C + (uint64_t)grow * N + gcol) = data;
            }
        }
        __syncthreads();
    }

    if (warp_id == 0) {
        tmem_dealloc(tmem_addr, TMEM_COLS);
    }
}

CUresult create_tma_desc(CUtensorMap* d, void* ptr,
                          uint64_t gdim0, uint64_t gdim1,
                          uint32_t bdim0, uint32_t bdim1,
                          CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[2] = {gdim0, gdim1};
    cuuint64_t globalStrides[1] = {gdim0 * 2};
    cuuint32_t boxDim[2] = {bdim0, bdim1};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        ptr, globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0);

    const bf16* A_ptr = static_cast<const bf16*>(A.data_ptr());
    const bf16* B_ptr = static_cast<const bf16*>(B.data_ptr());
    bf16*       C_ptr = static_cast<bf16*>(C.data_ptr());

    CUtensorMap tma_A, tma_B;
    CU_CHECK(create_tma_desc(&tma_A, (void*)A_ptr, K, M, BK, BM,
        CU_TENSOR_MAP_SWIZZLE_32B));
    CU_CHECK(create_tma_desc(&tma_B, (void*)B_ptr, K, N, BK, BN,
        CU_TENSOR_MAP_SWIZZLE_32B));

    dim3 block(NTHREADS);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    cudaFuncSetAttribute(gemm_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_TOTAL);

    gemm_kernel<<<grid, block, SMEM_TOTAL, stream>>>(
        tma_A, tma_B, C_ptr, (int)M, (int)N, (int)K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_cuda::run);

}  // namespace gemm_cuda