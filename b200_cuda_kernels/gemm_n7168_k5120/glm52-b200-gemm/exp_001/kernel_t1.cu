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

namespace gemm_cuda {

constexpr int BM = 128;
constexpr int BN = 256;
constexpr int BK = 16;
constexpr int NUM_STAGES = 4;
constexpr int NUM_THREADS = 128;

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase));
}

__device__ __forceinline__ void tcgen05_fence_before_fn() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tma_load_2d_cta_fn(
    const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_32b_fn(void* smem_ptr, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)6 << 61;
    return d;
}

__device__ __forceinline__ void umma_f16_cg1_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_cg1_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        :: "r"(a));
}

__device__ __forceinline__ void tmem_alloc_cg1_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_cg1_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
        :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= (0u << 15);
    d |= (0u << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__global__ __launch_bounds__(NUM_THREADS) void gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* C,
    int M, int N, int K) {

    extern __shared__ __align__(256) char smem_raw[];
    char* base = (char*)(((uintptr_t)smem_raw + 255) & ~255);

    __nv_bfloat16* smem_A[NUM_STAGES];
    __nv_bfloat16* smem_B[NUM_STAGES];
    for (int i = 0; i < NUM_STAGES; i++) {
        smem_A[i] = reinterpret_cast<__nv_bfloat16*>(base + i * BM * BK * 2);
    }
    char* ptr = base + NUM_STAGES * BM * BK * 2;
    for (int i = 0; i < NUM_STAGES; i++) {
        smem_B[i] = reinterpret_cast<__nv_bfloat16*>(ptr + i * BN * BK * 2);
    }
    ptr += NUM_STAGES * BN * BK * 2;

    uint64_t* tma_bar = reinterpret_cast<uint64_t*>(ptr);
    ptr += NUM_STAGES * 8;
    uint64_t* umma_bar = reinterpret_cast<uint64_t*>(ptr);
    ptr += NUM_STAGES * 8;
    uint32_t* tmem_addr_smem = reinterpret_cast<uint32_t*>(ptr);
    ptr += 4;

    ptr = (char*)(((uintptr_t)ptr + 255) & ~255);
    __nv_bfloat16* smem_epi = reinterpret_cast<__nv_bfloat16*>(ptr);

    int m_block = blockIdx.x;
    int n_block = blockIdx.y;
    int m_start = m_block * BM;
    int n_start = n_block * BN;
    int num_k_tiles = K / BK;

    if (threadIdx.x == 0) {
        for (int i = 0; i < NUM_STAGES; i++) {
            init_smem_barrier_fn(&tma_bar[i], 1);
            init_smem_barrier_fn(&umma_bar[i], 1);
        }
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        tmem_alloc_cg1_fn(tmem_addr_smem, 256);
    }
    __syncthreads();
    uint32_t tmem_addr = *tmem_addr_smem;

    uint64_t desc_A[NUM_STAGES], desc_B[NUM_STAGES];
    for (int i = 0; i < NUM_STAGES; i++) {
        desc_A[i] = make_smem_desc_32b_fn(smem_A[i], 256);
        desc_B[i] = make_smem_desc_32b_fn(smem_B[i], 256);
    }
    uint32_t idesc = make_instr_desc_fn(BM, BN);
    uint32_t tma_bytes = BM * BK * 2 + BN * BK * 2;

    uint32_t tma_phase[NUM_STAGES] = {};
    uint32_t umma_phase[NUM_STAGES] = {};

    if (threadIdx.x == 0) {
        for (int i = 0; i < NUM_STAGES && i < num_k_tiles; i++) {
            mbarrier_arrive_and_expect_tx_fn(&tma_bar[i], tma_bytes);
            tma_load_2d_cta_fn(&tma_A, &tma_bar[i], smem_A[i], i * BK, m_start);
            tma_load_2d_cta_fn(&tma_B, &tma_bar[i], smem_B[i], i * BK, n_start);
        }
    }

    for (int k = 0; k < num_k_tiles; k++) {
        int stage = k % NUM_STAGES;

        if (threadIdx.x == 0) {
            if (k >= NUM_STAGES) {
                mbarrier_wait_fn(&umma_bar[stage], umma_phase[stage]);
                umma_phase[stage] ^= 1;
            }

            mbarrier_wait_fn(&tma_bar[stage], tma_phase[stage]);
            tma_phase[stage] ^= 1;

            uint32_t accum = (k == 0) ? 0 : 1;
            umma_f16_cg1_fn(tmem_addr, desc_A[stage], desc_B[stage], idesc, accum);
            umma_commit_cg1_fn(&umma_bar[stage]);

            int next_k = k + NUM_STAGES;
            if (next_k < num_k_tiles) {
                mbarrier_arrive_and_expect_tx_fn(&tma_bar[stage], tma_bytes);
                tma_load_2d_cta_fn(&tma_A, &tma_bar[stage], smem_A[stage], next_k * BK, m_start);
                tma_load_2d_cta_fn(&tma_B, &tma_bar[stage], smem_B[stage], next_k * BK, n_start);
            }
        }
    }

    if (threadIdx.x == 0) {
        for (int stage = 0; stage < NUM_STAGES; stage++) {
            if (stage < num_k_tiles) {
                mbarrier_wait_fn(&umma_bar[stage], umma_phase[stage]);
            }
        }
        tcgen05_fence_before_fn();
    }
    __syncthreads();

    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t tmem_row = warp_id * 32 + lane_id;

    for (uint32_t col = 0; col < (uint32_t)BN; col += 8) {
        uint32_t r[8];
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]),
              "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
            : "r"(tmem_addr + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

        uint32_t off = tmem_row * BN + col;
        smem_epi[off + 0] = __float2bfloat16(__uint_as_float(r[0]));
        smem_epi[off + 1] = __float2bfloat16(__uint_as_float(r[1]));
        smem_epi[off + 2] = __float2bfloat16(__uint_as_float(r[2]));
        smem_epi[off + 3] = __float2bfloat16(__uint_as_float(r[3]));
        smem_epi[off + 4] = __float2bfloat16(__uint_as_float(r[4]));
        smem_epi[off + 5] = __float2bfloat16(__uint_as_float(r[5]));
        smem_epi[off + 6] = __float2bfloat16(__uint_as_float(r[6]));
        smem_epi[off + 7] = __float2bfloat16(__uint_as_float(r[7]));
    }
    __syncthreads();

    for (uint32_t row = warp_id; row < (uint32_t)BM; row += 4) {
        uint32_t global_row = m_start + row;
        if (global_row >= (uint32_t)M) continue;

        uint32_t col_start = lane_id * 8;
        uint32_t global_col = n_start + col_start;
        uint32_t smem_off = row * BN + col_start;

        if (global_col + 7 < (uint32_t)N) {
            uint4 data = *reinterpret_cast<uint4*>(&smem_epi[smem_off]);
            *reinterpret_cast<uint4*>(C + (uint64_t)global_row * N + global_col) = data;
        } else {
            for (int j = 0; j < 8 && global_col + j < (uint32_t)N; j++) {
                C[(uint64_t)global_row * N + global_col + j] = smem_epi[smem_off + j];
            }
        }
    }

    __syncthreads();

    if (threadIdx.x < 32) {
        tmem_dealloc_cg1_fn(tmem_addr, 256);
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0);

    __nv_bfloat16* A_ptr = static_cast<__nv_bfloat16*>(A.data_ptr());
    __nv_bfloat16* B_ptr = static_cast<__nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* C_ptr = static_cast<__nv_bfloat16*>(C.data_ptr());

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    CUtensorMap tma_A, tma_B;

    {
        cuuint64_t globalDim[2] = {(cuuint64_t)K, (cuuint64_t)M};
        cuuint64_t globalStrides[1] = {(cuuint64_t)K * 2};
        cuuint32_t boxDim[2] = {(cuuint32_t)BK, (cuuint32_t)BM};
        cuuint32_t elementStrides[2] = {1, 1};
        CU_CHECK(cuTensorMapEncodeTiled(
            &tma_A, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
            A_ptr, globalDim, globalStrides, boxDim, elementStrides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_32B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    }

    {
        cuuint64_t globalDim[2] = {(cuuint64_t)K, (cuuint64_t)N};
        cuuint64_t globalStrides[1] = {(cuuint64_t)K * 2};
        cuuint32_t boxDim[2] = {(cuuint32_t)BK, (cuuint32_t)BN};
        cuuint32_t elementStrides[2] = {1, 1};
        CU_CHECK(cuTensorMapEncodeTiled(
            &tma_B, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
            B_ptr, globalDim, globalStrides, boxDim, elementStrides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_32B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    }

    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN, 1);
    dim3 block(NUM_THREADS, 1, 1);

    int smem_bytes = 256
                   + NUM_STAGES * BM * BK * 2
                   + NUM_STAGES * BN * BK * 2
                   + NUM_STAGES * 8 * 2
                   + 4
                   + 256
                   + BM * BN * 2;

    CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    gemm_kernel<<<grid, block, smem_bytes, stream>>>(
        tma_A, tma_B, C_ptr, (int)M, (int)N, (int)K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_cuda::run);

}  // namespace gemm_cuda