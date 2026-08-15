#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__); exit(1);} }while(0)
#define CU_CHECK(call) do { CUresult _e=(call); if(_e!=CUDA_SUCCESS){ const char* s=nullptr; \
  cuGetErrorString(_e,&s); fprintf(stderr,"CU error %s at %s:%d\n",s,__FILE__,__LINE__); exit(1);} }while(0)

namespace gemm_kernel_ns {

// ---------------- device helpers ----------------
__device__ __forceinline__ bool elect_one_sync_fn() {
    uint32_t pred;
    asm volatile("{\n.reg .pred p;\n elect.sync _|p, 0xFFFFFFFF;\n selp.b32 %0, 1, 0, p;\n}\n" : "=r"(pred));
    return pred != 0;
}
__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}
__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes) : "memory");
}
__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n.reg .pred P;\n WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase));
}
__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)), "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void tmem_alloc1(uint32_t* dst, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a), "r"(ncols));
}
__device__ __forceinline__ void tmem_dealloc1(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(addr), "r"(ncols));
}
__device__ __forceinline__ void umma_f16_cg1(uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}
__device__ __forceinline__ void umma_commit_1sm(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(a));
}
__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}
__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ uint64_t make_smem_desc(void* ptr, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    d |= (uint64_t)((addr & 0x3FFFF) >> 4);
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   // version = 1 (SM100)
    d |= (uint64_t)2 << 61;   // SWIZZLE_128B
    return d;
}
__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    // c_format = FP32
    d |= (1u << 7);    // a_format = BF16
    d |= (1u << 10);   // b_format = BF16
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

// ---------------- kernel ----------------
// A: [M,K] bf16 (K contiguous), B: [N,K] bf16 (K contiguous). C = A @ B^T -> [M,N]
// One CTA computes a 128(M) x 256(N) output tile. K accumulated in TMEM.
__global__ void gemm_kernel(const __grid_constant__ CUtensorMap dA,
                            const __grid_constant__ CUtensorMap dB,
                            __nv_bfloat16* __restrict__ C,
                            int M, int N, int K, int num_n_tiles) {
    extern __shared__ char smem_raw[];
    uint32_t base_off = (uint32_t)__cvta_generic_to_shared(smem_raw);
    uint32_t pad = ((base_off + 1023u) & ~1023u) - base_off;
    __nv_bfloat16* A_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + pad);   // [128][64]
    __nv_bfloat16* B_smem = A_smem + 128 * 64;                                  // [256][64]
    uint64_t* bar = reinterpret_cast<uint64_t*>(reinterpret_cast<char*>(B_smem) + 256 * 64 * 2);
    uint64_t* bar_tma = bar;       // bar[0]
    uint64_t* bar_mma = bar + 1;   // bar[1]
    uint32_t* tmem_ptr = reinterpret_cast<uint32_t*>(bar + 2);

    int tid = threadIdx.x;
    int warp = tid >> 5;

    int tile = blockIdx.x;
    int m_tile = tile / num_n_tiles;
    int n_tile = tile % num_n_tiles;
    int m_base = m_tile * 128;
    int n_base = n_tile * 256;

    if (tid == 0) {
        init_smem_barrier_fn(bar_tma, 1);
        init_smem_barrier_fn(bar_mma, 1);
    }
    if (warp == 0) {
        tmem_alloc1(tmem_ptr, 256);
    }
    __syncthreads();

    uint32_t tmem_base = tmem_ptr[0];
    uint32_t idesc = make_instr_desc_fn(128, 256);

    bool leader = false;
    if (warp == 0) leader = elect_one_sync_fn();

    uint32_t tx = (uint32_t)(128 * 64 * 2) + (uint32_t)(256 * 64 * 2);

    int num_k = K / 64;
    uint32_t ph_tma = 0, ph_mma = 0;

    for (int ks = 0; ks < num_k; ks++) {
        int k0 = ks * 64;
        if (leader) {
            mbarrier_arrive_and_expect_tx_fn(bar_tma, tx);
            tma_load_2d_fn(&dA, bar_tma, A_smem, k0, m_base);
            tma_load_2d_fn(&dB, bar_tma, B_smem, k0, n_base);
        }
        mbarrier_wait_fn(bar_tma, ph_tma);
        ph_tma ^= 1;

        if (leader) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                uint64_t da = make_smem_desc(A_smem + j * 16, 1024);
                uint64_t db = make_smem_desc(B_smem + j * 16, 1024);
                uint32_t accum = (ks == 0 && j == 0) ? 0u : 1u;
                umma_f16_cg1(tmem_base, da, db, idesc, accum);
            }
            umma_commit_1sm(bar_mma);
        }
        mbarrier_wait_fn(bar_mma, ph_mma);
        ph_mma ^= 1;
    }

    __syncthreads();

    // Epilogue: read TMEM (128 rows x 256 cols FP32), convert to bf16, store to C.
    int grow = m_base + tid;
    for (int col = 0; col < 256; col += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn(tmem_base + (uint32_t)col, &r0, &r1, &r2, &r3);
        tmem_load_fence_fn();
        if (grow < M) {
            __nv_bfloat16 tmp[4];
            tmp[0] = __float2bfloat16(__uint_as_float(r0));
            tmp[1] = __float2bfloat16(__uint_as_float(r1));
            tmp[2] = __float2bfloat16(__uint_as_float(r2));
            tmp[3] = __float2bfloat16(__uint_as_float(r3));
            __nv_bfloat16* o = C + (size_t)grow * N + n_base + col;
            *reinterpret_cast<uint2*>(o) = *reinterpret_cast<uint2*>(tmp);
        }
    }

    __syncthreads();
    if (warp == 0) tmem_dealloc1(tmem_base, 256);
}

// ---------------- host ----------------
static CUresult make_tma_desc(CUtensorMap* d, void* ptr, uint64_t inner, uint64_t outer,
                              uint32_t box_inner, uint32_t box_outer) {
    cuuint64_t globalDim[2] = {inner, outer};
    cuuint64_t globalStrides[1] = {inner * 2};
    cuuint32_t boxDim[2] = {box_inner, box_outer};
    cuuint32_t elemStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, ptr, globalDim, globalStrides, boxDim, elemStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0);

    __nv_bfloat16* Aptr = static_cast<__nv_bfloat16*>(A.data_ptr());
    __nv_bfloat16* Bptr = static_cast<__nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* Cptr = static_cast<__nv_bfloat16*>(C.data_ptr());

    CUtensorMap dA, dB;
    CU_CHECK(make_tma_desc(&dA, Aptr, (uint64_t)K, (uint64_t)M, 64, 128));
    CU_CHECK(make_tma_desc(&dB, Bptr, (uint64_t)K, (uint64_t)N, 64, 256));

    int num_n_tiles = (int)(N / 256);
    int num_m_tiles = (int)((M + 127) / 128);
    int grid = num_m_tiles * num_n_tiles;

    size_t smem_bytes = 1024 + 16384 + 32768 + 64;

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));

    gemm_kernel<<<grid, 128, smem_bytes, stream>>>(dA, dB, Cptr, (int)M, (int)N, (int)K, num_n_tiles);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_kernel_ns::run);

}  // namespace gemm_kernel_ns