#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", \
                cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define CU_CHECK(call) do { \
    CUresult _r = (call); \
    if (_r != CUDA_SUCCESS) { \
        const char* errStr; \
        cuGetErrorString(_r, &errStr); \
        fprintf(stderr, "CU error %s at %s:%d\n", \
                errStr ? errStr : "unknown", __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

namespace gemm_impl {

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 16;
constexpr int THREADS = 128;
constexpr int A_TILE_BYTES = BM * BK * 2;
constexpr int B_TILE_BYTES = BN * BK * 2;

__device__ __forceinline__ uint32_t sa(void* p) {
    return (uint32_t)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ void init_bar(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(sa(bar)), "r"(count));
}

__device__ __forceinline__ void fence_bar_init() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void bar_arrive_expect_tx(uint64_t* bar, uint32_t tx) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"(sa(bar)), "r"(tx) : "memory");
}

__device__ __forceinline__ void bar_wait(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n.reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n}\n"
        :: "r"(sa(bar)), "r"(phase));
}

__device__ __forceinline__ void tma_load(
    const CUtensorMap* desc, uint64_t* bar, void* smem,
    int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4}], [%2];"
        :: "r"(sa(smem)), "l"((uint64_t)desc),
        "r"(sa(bar)), "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tmem_alloc(uint32_t* dst_smem, int ncols) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(sa(dst_smem)), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
        :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tmem_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ uint64_t make_desc(void* ptr, uint32_t lbo, uint32_t sbo, int sw) {
    uint64_t d = 0;
    uint32_t addr = sa(ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)(sw & 0x7) << 61;
    return d;
}

__device__ __forceinline__ uint32_t make_idesc(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    // D type = FP32
    d |= (1u << 7);    // A type = BF16
    d |= (1u << 10);   // B type = BF16
    d |= (0u << 15);   // A K-major
    d |= (0u << 16);   // B K-major
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
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
        :: "r"(sa(bar)));
}

__device__ __forceinline__ void tcgen05_fence_after() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__global__ __launch_bounds__(THREADS)
void gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K) {

    int bm = blockIdx.y;
    int bn = blockIdx.x;
    int m_off = bm * BM;
    int n_off = bn * BN;

    __shared__ __align__(256) __nv_bfloat16 A_smem[2][BM * BK];
    __shared__ __align__(256) __nv_bfloat16 B_smem[2][BN * BK];
    __shared__ __align__(8) uint64_t mbar[3];
    __shared__ uint32_t tmem_addr_smem;
    extern __shared__ __nv_bfloat16 C_smem[];

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    if (tid == 0) {
        init_bar(&mbar[0], 1);
        init_bar(&mbar[1], 1);
        init_bar(&mbar[2], 1);
        fence_bar_init();
    }
    __syncthreads();

    if (warp_id == 0) {
        tmem_alloc(&tmem_addr_smem, 128);
    }
    __syncthreads();
    uint32_t tmem_d = tmem_addr_smem;

    int num_k = K / BK;
    uint32_t idesc = make_idesc(BM, BN);

    if (tid == 0) {
        bar_arrive_expect_tx(&mbar[0], A_TILE_BYTES + B_TILE_BYTES);
        tma_load(&tma_A, &mbar[0], A_smem[0], 0, m_off);
        tma_load(&tma_B, &mbar[0], B_smem[0], 0, n_off);
    }

    for (int k = 0; k < num_k; k++) {
        int buf = k % 2;

        bar_wait(&mbar[buf], k % 2);

        if (k + 1 < num_k && tid == 0) {
            int next = 1 - buf;
            bar_arrive_expect_tx(&mbar[next], A_TILE_BYTES + B_TILE_BYTES);
            tma_load(&tma_A, &mbar[next], A_smem[next], (k + 1) * BK, m_off);
            tma_load(&tma_B, &mbar[next], B_smem[next], (k + 1) * BK, n_off);
        }

        uint64_t da = make_desc(A_smem[buf], 1, 256, 6);
        uint64_t db = make_desc(B_smem[buf], 1, 256, 6);
        uint32_t accum = (k == 0) ? 0 : 1;

        if (tid == 0) {
            tcgen05_fence_after();
            umma(tmem_d, da, db, idesc, accum);
            umma_commit(&mbar[2]);
        }

        bar_wait(&mbar[2], k % 2);
    }

    for (int col = 0; col < BN; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
            : "r"(tmem_d + col));
        tmem_wait_ld();

        int row = tid;
        __nv_bfloat16* sptr = &C_smem[row * BN + col];
        sptr[0] = __float2bfloat16(__uint_as_float(r0));
        sptr[1] = __float2bfloat16(__uint_as_float(r1));
        sptr[2] = __float2bfloat16(__uint_as_float(r2));
        sptr[3] = __float2bfloat16(__uint_as_float(r3));
    }
    __syncthreads();

    for (int step = 0; step < (BM + 3) / 4; step++) {
        int row = step * 4 + warp_id;
        if (row >= BM) continue;
        int gr = m_off + row;
        if (gr >= M) continue;
        int col_start = lane_id * 4;
        int gc = n_off + col_start;
        if (gc + 3 < N) {
            uint2 data = *reinterpret_cast<uint2*>(&C_smem[row * BN + col_start]);
            *reinterpret_cast<uint2*>(C + (size_t)gr * N + gc) = data;
        }
    }

    if (warp_id == 0) {
        tmem_dealloc(tmem_d, 128);
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    int64_t M = A.size(0);
    int64_t N = B.size(0);
    int64_t K = A.size(1);

    const __nv_bfloat16* A_ptr = static_cast<const __nv_bfloat16*>(A.data_ptr());
    const __nv_bfloat16* B_ptr = static_cast<const __nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* C_ptr = static_cast<__nv_bfloat16*>(C.data_ptr());

    CUtensorMap desc_A, desc_B;
    CUtensorMapDataType dt = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;

    {
        cuuint64_t gd[2] = {(cuuint64_t)K, (cuuint64_t)M};
        cuuint64_t gs[1] = {(cuuint64_t)K * 2};
        cuuint32_t bd[2] = {(cuuint32_t)BK, (cuuint32_t)BM};
        cuuint32_t es[2] = {1, 1};
        CU_CHECK(cuTensorMapEncodeTiled(
            &desc_A, dt, 2, (void*)A_ptr, gd, gs, bd, es,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_32B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_64B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA));
    }

    {
        cuuint64_t gd[2] = {(cuuint64_t)K, (cuuint64_t)N};
        cuuint64_t gs[1] = {(cuuint64_t)K * 2};
        cuuint32_t bd[2] = {(cuuint32_t)BK, (cuuint32_t)BN};
        cuuint32_t es[2] = {1, 1};
        CU_CHECK(cuTensorMapEncodeTiled(
            &desc_B, dt, 2, (void*)B_ptr, gd, gs, bd, es,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_32B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_64B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA));
    }

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(THREADS);

    int smem_bytes = BM * BN * 2;
    cudaFuncSetAttribute(gemm_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    gemm_kernel<<<grid, block, smem_bytes, stream>>>(
        desc_A, desc_B, C_ptr, (int)M, (int)N, (int)K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_impl::run);

}  // namespace gemm_impl