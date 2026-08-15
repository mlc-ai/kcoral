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
constexpr int BK = 64;
constexpr int NUM_THREADS = 128;
constexpr int NUM_STAGES = 3;

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

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void prefetch_tma_descriptor_fn(const CUtensorMap* d) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(d) : "memory");
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
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        :: "r"(a) : "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)((addr >> 7) & 7) << 49;
    d |= (uint64_t)2 << 61;   // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    // dtype = FP32
    d |= (1u << 7);    // atype = BF16
    d |= (1u << 10);   // btype = BF16
    d |= (0u << 15);   // a_major = 0 (K-Major)
    d |= (0u << 16);   // b_major = 0 (K-Major)
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

CUresult create_tma_2d_descriptor_bf16(CUtensorMap* d, void* globalAddress,
    uint64_t gmem_inner_dim, uint64_t gmem_outer_dim,
    uint32_t smem_inner_dim, uint32_t smem_outer_dim,
    CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion,
    CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        globalAddress, globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill);
}

__global__ __launch_bounds__(NUM_THREADS, 1)
void gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* __restrict__ C, int M, int N, int K) {

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int m_start = blockIdx.x * BM;
    const int n_start = blockIdx.y * BN;

    extern __shared__ __align__(1024) char smem_raw[];
    __nv_bfloat16* A_smem[NUM_STAGES];
    __nv_bfloat16* B_smem[NUM_STAGES];
    A_smem[0] = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    #pragma unroll
    for (int i = 1; i < NUM_STAGES; i++) A_smem[i] = A_smem[i-1] + BM * BK;
    B_smem[0] = A_smem[NUM_STAGES-1] + BM * BK;
    #pragma unroll
    for (int i = 1; i < NUM_STAGES; i++) B_smem[i] = B_smem[i-1] + BN * BK;

    uint64_t* barriers = reinterpret_cast<uint64_t*>(
        (reinterpret_cast<uintptr_t>(B_smem[NUM_STAGES-1] + BN * BK) + 7) & ~7ULL);
    uint64_t* full_bar = barriers;
    uint64_t* umma_bar = barriers + NUM_STAGES;

    __shared__ uint32_t tmem_alloc_result;

    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < NUM_STAGES; i++) {
            init_smem_barrier_fn(&full_bar[i], 1);
            init_smem_barrier_fn(&umma_bar[i], 1);
        }
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    if (warp_id == 0) {
        tmem_alloc_cg1_fn(&tmem_alloc_result, 256);
    }
    __syncthreads();
    const uint32_t tmem_addr = tmem_alloc_result;

    if (tid == 0) {
        prefetch_tma_descriptor_fn(&tma_A);
        prefetch_tma_descriptor_fn(&tma_B);
    }

    const int num_k_tiles = K / BK;
    const uint32_t idesc = make_instr_desc_fn(BM, BN);
    constexpr uint32_t tx_bytes = BM * BK * 2 + BN * BK * 2;

    uint32_t phase_full[NUM_STAGES] = {0, 0, 0};
    uint32_t phase_umma[NUM_STAGES] = {0, 0, 0};

    // Prologue: issue TMA for first NUM_STAGES tiles
    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < NUM_STAGES; s++) {
            if (s < num_k_tiles) {
                tma_load_2d_fn(&tma_A, &full_bar[s], A_smem[s], s * BK, m_start);
                tma_load_2d_fn(&tma_B, &full_bar[s], B_smem[s], s * BK, n_start);
                mbarrier_arrive_and_expect_tx_fn(&full_bar[s], tx_bytes);
            }
        }
    }

    for (int k = 0; k < num_k_tiles; k++) {
        int stage = k % NUM_STAGES;

        // Wait for TMA data
        mbarrier_wait_fn(&full_bar[stage], phase_full[stage]);
        phase_full[stage] ^= 1;

        // Issue UMMA: 4 MMAs (K=16 each, total BK=64)
        #pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            uint64_t desc_a = make_smem_desc_sm100_fn(
                reinterpret_cast<char*>(A_smem[stage]) + kk * 32, 1, 1024);
            uint64_t desc_b = make_smem_desc_sm100_fn(
                reinterpret_cast<char*>(B_smem[stage]) + kk * 32, 1, 1024);
            uint32_t accum = (k == 0 && kk == 0) ? 0 : 1;
            if (tid == 0) {
                umma_f16_cg1_fn(tmem_addr, desc_a, desc_b, idesc, accum);
            }
        }
        if (tid == 0) {
            umma_commit_cg1_fn(&umma_bar[stage]);
        }

        // Wait for UMMA (TMEM accumulator ready for next MMA, SMEM buffer free)
        mbarrier_wait_fn(&umma_bar[stage], phase_umma[stage]);
        phase_umma[stage] ^= 1;

        // Issue TMA for stage reuse (k + NUM_STAGES)
        if (tid == 0 && k + NUM_STAGES < num_k_tiles) {
            tma_load_2d_fn(&tma_A, &full_bar[stage], A_smem[stage],
                           (k + NUM_STAGES) * BK, m_start);
            tma_load_2d_fn(&tma_B, &full_bar[stage], B_smem[stage],
                           (k + NUM_STAGES) * BK, n_start);
            mbarrier_arrive_and_expect_tx_fn(&full_bar[stage], tx_bytes);
        }
    }

    __syncthreads();

    // ======================== Epilogue ========================
    __nv_bfloat16* smem_out = reinterpret_cast<__nv_bfloat16*>(smem_raw);

    // TMEM -> SMEM: use .32x32b.x8 for fewer instructions
    // Each warp handles 32 rows (warp_id * 32). 4 warps cover 128 rows.
    // Issue 4 loads of x8 (32 cols), wait once, process 32 values.
    #pragma unroll 4
    for (uint32_t col = 0; col < (uint32_t)BN; col += 32) {
        uint32_t r[4][8];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint32_t addr = tmem_addr + (warp_id * 32 << 16) + col + i * 8;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                : "=r"(r[i][0]), "=r"(r[i][1]), "=r"(r[i][2]), "=r"(r[i][3]),
                  "=r"(r[i][4]), "=r"(r[i][5]), "=r"(r[i][6]), "=r"(r[i][7])
                : "r"(addr));
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint32_t base = (uint32_t)tid * BN + col + i * 8;
            smem_out[base + 0] = __float2bfloat16(__uint_as_float(r[i][0]));
            smem_out[base + 1] = __float2bfloat16(__uint_as_float(r[i][1]));
            smem_out[base + 2] = __float2bfloat16(__uint_as_float(r[i][2]));
            smem_out[base + 3] = __float2bfloat16(__uint_as_float(r[i][3]));
            smem_out[base + 4] = __float2bfloat16(__uint_as_float(r[i][4]));
            smem_out[base + 5] = __float2bfloat16(__uint_as_float(r[i][5]));
            smem_out[base + 6] = __float2bfloat16(__uint_as_float(r[i][6]));
            smem_out[base + 7] = __float2bfloat16(__uint_as_float(r[i][7]));
        }
    }
    __syncthreads();

    // SMEM -> Global: coalesced 128-bit stores
    // Each warp handles 32 consecutive rows; 32 threads x 8 BF16 = 256 cols per row
    #pragma unroll
    for (uint32_t row = warp_id * 32; row < (uint32_t)(warp_id + 1) * 32; row++) {
        uint32_t global_row = m_start + row;
        if (global_row >= (uint32_t)M) continue;
        uint32_t col_start = lane_id * 8;
        uint32_t global_col = n_start + col_start;
        uint4 data = *reinterpret_cast<uint4*>(&smem_out[row * BN + col_start]);
        *reinterpret_cast<uint4*>(C + (uint64_t)global_row * N + global_col) = data;
    }

    __syncthreads();
    if (warp_id == 0) {
        tmem_dealloc_cg1_fn(tmem_addr, 256);
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));

    const int M = (int)A.size(0);
    const int N = 7168;
    const int K = 5120;

    __nv_bfloat16* A_ptr = static_cast<__nv_bfloat16*>(A.data_ptr());
    __nv_bfloat16* B_ptr = static_cast<__nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* C_ptr = static_cast<__nv_bfloat16*>(C.data_ptr());

    CUtensorMap tma_A_desc, tma_B_desc;

    CU_CHECK(create_tma_2d_descriptor_bf16(&tma_A_desc, A_ptr, K, M, BK, BM,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA));

    CU_CHECK(create_tma_2d_descriptor_bf16(&tma_B_desc, B_ptr, K, N, BK, BN,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA));

    const int m_tiles = (M + BM - 1) / BM;
    const int n_tiles = N / BN;
    dim3 grid(m_tiles, n_tiles, 1);
    dim3 block(NUM_THREADS, 1, 1);

    // 3 stages: 3 * (128*64 + 256*64) * 2 + barriers = 147456 + 64 ≈ 144KB
    const int smem_bytes = NUM_STAGES * (BM * BK + BN * BK) * 2 + 256;

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(C.device().device_type, C.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    gemm_kernel<<<grid, block, smem_bytes, stream>>>(
        tma_A_desc, tma_B_desc, C_ptr, M, N, K);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_cuda::run);

}  // namespace gemm_cuda