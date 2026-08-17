#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                         \
    cudaError_t err = (call);                                         \
    if (err != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                 \
                __FILE__, __LINE__, cudaGetErrorString(err));         \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)

/* ================================================================
 * Device helper functions
 * ================================================================ */

__device__ __forceinline__ uint32_t get_cluster_rank() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__device__ __forceinline__ void barrier_init(uint64_t *b, uint32_t cnt) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
                 : : "r"((uint32_t)__cvta_generic_to_shared(b)), "r"(cnt));
}

__device__ __forceinline__ void fence_barrier_init() {
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

/** Spin-wait on an mbarrier phase.  parity=0 → even phase, 1 → odd. */
__device__ __forceinline__ void barrier_wait(uint64_t *b, uint32_t parity) {
    asm volatile("{\n"
                 "  .reg .pred P;\n"
                 "wl_%=:\n"
                 "  mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
                 "  @!P bra wl_%=;\n"
                 "}\n"
                 : : "r"((uint32_t)__cvta_generic_to_shared(b)), "r"(parity));
}

__device__ __forceinline__ void tmem_alloc(uint32_t *dst, int cols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                 : : "r"(a), "r"(cols));
}

__device__ __forceinline__ void tmem_dealloc(uint32_t addr, int cols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
                 : : "r"(addr), "r"(cols));
}

__device__ __forceinline__ void tmem_fence_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

/** Build a shared-memory descriptor for tcgen05 UMMA. */
__device__ __forceinline__ uint64_t mk_smem_desc(void *ptr, uint32_t lbo, uint32_t sbo, uint32_t swz) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    d  = (uint64_t)((addr) & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo) & 0x3FFFF) >> 4 << 16;
    d |= (uint64_t)((sbo) & 0x3FFFF) >> 4 << 32;
    d |= 1ULL << 46;                       // version = 1 (SM100)
    d |= (uint64_t)swz << 61;              // swizzle: 0=none, 2=128B
    return d;
}

/** Instruction descriptor: BF16×BF16 → FP32, both K-major, no transpose. */
__device__ __forceinline__ uint32_t mk_idesc(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= 1u << 4;                          // dtype = FP32
    d |= 1u << 7;                          // atype = BF16
    d |= 1u << 10;                         // btype = BF16
    d |= 0u << 15;                         // a_no_transpose (K-major)
    d |= 0u << 16;                         // b_no_transpose (K-major)
    d |= (N >> 3) << 17;                   // n_dim (lower 3 bits not included)
    d |= (M >> 4) << 24;                   // m_dim (lower 4 bits not included)
    return d;
}

/* ================================================================
 * Kernel constants
 * ================================================================ */
static constexpr int BM  = 64;
static constexpr int BN  = 128;
static constexpr int BK  = 16;
static constexpr int CSZ = 2;
static constexpr int THR = 128;
static constexpr int NS  = 2;

static constexpr int CM  = BM * CSZ;     // 128 combined M
static constexpr int CN  = BN * CSZ;     // 256 combined N
static constexpr int AE  = BM * BK;      // 1024 bf16 elements for A
static constexpr int BE  = CN * BK;      // 4096 bf16 elements for B
static constexpr int ASZ = AE * 2;       // 2048 bytes for A
static constexpr int BSZ = BE * 2;       // 8192 bytes for B
static constexpr int TMC = CN;           // 256 TMEM columns for D

/* ================================================================
 * Shared memory layout (all in DYNAMIC shared mem)
 *
 * Offset | Size   | Content
 * --------+--------+-------------------
 * 0       | 8      | mbarrier[0]
 * 8       | 8      | mbarrier[1]
 * 16      | 4      | tmem_base_addr
 * 20..31  | 12     | padding (reached by compiler alignment anyway)
 * 32      | 2048   | A staging tile  [BM][BK] bf16  (K-major)
 * 2080    | 8192   | B staging tile  [CN][BK] bf16  (K-major)
 * --------+--------+-------
 * Total ≈ 10272 bytes
 * ================================================================ */
static constexpr int OFF_TMEM = 16;
static constexpr int OFF_A    = 32;
static constexpr int OFF_B    = OFF_A + ASZ;
static constexpr int TOTAL_SMEM = OFF_B + BSZ;

/* ================================================================
 * Main kernel
 * ================================================================ */
__global__ void __launch_bounds__(THR)
bw_gemm_kernel(const __nv_bfloat16 *__restrict__ A,
               const __nv_bfloat16 *__restrict__ B,
               __nv_bfloat16 *__restrict__ C,
               uint32_t M, uint32_t N, uint32_t K,
               uint32_t nblk_n)
{
    extern __shared__ char sm[];

    uint32_t tid  = threadIdx.x;
    uint32_t lane = tid % 32;
    uint32_t cr   = get_cluster_rank();

    /* ---- typed pointers into dynamic shared memory ----
     * CUDA dynamic SM is guaranteed >= 256-byte aligned, so these
     * offset-based casts are naturally well-aligned.            */
    uint64_t          *bar  = reinterpret_cast<uint64_t *>(sm);
    uint32_t          *tptr = reinterpret_cast<uint32_t *>(sm + OFF_TMEM);
    __nv_bfloat16     *spa  = reinterpret_cast<__nv_bfloat16 *>(sm + OFF_A);
    __nv_bfloat16     *spb  = reinterpret_cast<__nv_bfloat16 *>(sm + OFF_B);

    /* ---------- initialise barriers & Tensor Memory ---------- */
    if (tid == 0) {
        barrier_init(&bar[0], 1);
        barrier_init(&bar[1], 1);
        fence_barrier_init();
        tmem_alloc(tptr, TMC);
    }
    __syncthreads();

    /* ---------- compute per-cluster indices ---------- */
    uint32_t cid   = blockIdx.x / CSZ;
    uint32_t nlk   = (N + CN - 1) / CN;
    uint32_t mrb   = cid / nlk;
    uint32_t nnb   = cid % nlk;
    uint32_t rsa   = mrb * CM + cr * BM;
    uint32_t ns    = nnb * CN;

    uint32_t idesc = mk_idesc(CM, CN);

    /* SMEM descriptors — K-major, no-swizzle.
     * Non-swizzled K-major:
     *   SBO = 8 × 16 = 128                     (span = 16 B)
     *   LBO = (ATOM_MMODE_DIM / 8) × SBO
     * For A: BM=64 ⇒ LBO = 8×128 = 1024
     * For B: CN=256⇒ LBO = 32×128 = 4096
     */
    constexpr uint32_t A_LBO = (BM / 8) * 128;
    constexpr uint32_t A_SBO = 128;
    constexpr uint32_t B_LBO = (CN / 8) * 128;
    constexpr uint32_t B_SBO = 128;

    uint64_t ad = mk_smem_desc(spa, A_LBO, A_SBO, 0);
    uint64_t bd = mk_smem_desc(spb, B_LBO, B_SBO, 0);

    uint32_t nkt = (K + BK - 1) / BK;
    uint32_t ta  = *tptr;

    uint32_t ad0 = (uint32_t)ad,   ad1 = (uint32_t)(ad >> 32);
    uint32_t bd0 = (uint32_t)bd,   bd1 = (uint32_t)(bd >> 32);

    /* ===========================================================
     * Main K-loop  (sequential, 2-barrier ping-pong)
     *
     * Phase schedule per barrier (alternating):
     *   phase 2j   (even, parity 0): cooperative-load arrive
     *   phase 2j+1 (odd,  parity 1): UMMA-commit arrive
     * =========================================================== */
    for (uint32_t k = 0; k < nkt; ++k) {
        int s = k % NS;

        /* ----- cooperative K-major load of A -----
         * spa[e] with e = kc·BM + mr                    */
        for (uint32_t e = tid; e < (uint32_t)AE; e += THR) {
            uint32_t mr  = e % BM;
            uint32_t kc  = e / BM;
            uint32_t gmr = rsa + mr;
            uint32_t gkc = k * BK + kc;
            if (gmr < M && gkc < K)
                spa[e] = A[gmr * K + gkc];
        }

        /* ----- cooperative K-major load of B -----
         * spb[e] with e = kc·CN + nr                    */
        for (uint32_t e = tid; e < (uint32_t)BE; e += THR) {
            uint32_t nr  = e % CN;
            uint32_t kc  = e / CN;
            uint32_t gnr = ns + nr;
            uint32_t gkc = k * BK + kc;
            if (gnr < N && gkc < K)
                spb[e] = B[gnr * K + gkc];
        }

        /* All threads must finish before anyone touches the barrier. */
        __syncthreads();

        /* tid 0 arrives → even phase completes instantly (no TX). */
        if (tid == 0) {
            asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
                         : : "r"((uint32_t)__cvta_generic_to_shared(&bar[s]))
                         : "memory");
        }
        barrier_wait(&bar[s], 0);

        /* ----- UMMA: D = A[k] · B[k]^T (+acc after k=0) -----
         * Inputs (no outputs):
         *   %0 = ta     TMEM base address
         *   %1 = ad0    descriptor A low
         *   %2 = ad1    descriptor A high
         *   %3 = bd0    descriptor B low
         *   %4 = bd1    descriptor B high
         *   %5 = idesc  instruction descriptor
         *   %6 = clr    0=clear acc, 1=accumulate               */
        uint32_t clr = (k == 0) ? 0 : 1;
        if (lane == 0) {
            asm volatile(
                ".reg .b64 da,db;\n"
                ".reg .pred p;\n"
                "mov.b64     da, {%1,%2};\n"
                "mov.b64     db, {%3,%4};\n"
                "setp.ne.b32 p, %6, 0;\n"
                "tcgen05.mma.cta_group::2.kind::f16 [%0], da, db, %5, p;\n"
                : : "r"(ta), "r"(ad0), "r"(ad1),
                  "r"(bd0), "r"(bd1), "r"(idesc), "r"(clr));

            asm volatile(
                "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cta.b64 [%0];"
                : : "r"((uint32_t)__cvta_generic_to_shared(&bar[s])));
        }
        barrier_wait(&bar[s], 1);
    }

    /* ===========================================================
     * Epilogue: TMEM (FP32) → BF16 → global C
     * =========================================================== */
    uint32_t tml   = tid;
    uint32_t lr    = tml - cr * BM;
    bool     mine  = (lr < BM);
    uint32_t gr    = mine ? (rsa + lr) : 0xFFFFFFFFu;

    __nv_bfloat16 *out = (mine && gr < M)
                              ? C + (uint64_t)gr * N + ns
                              : nullptr;

    for (uint32_t c = 0; c < (uint32_t)CN; c += 4) {
        uint32_t v0, v1, v2, v3;
        uint32_t ta_ = (tml << 16) | c;
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3)
            : "r"(ta_));

        if (out) {
            if (ns + c + 0 < N) out[c + 0] = __float2bfloat16(__uint_as_float(v0));
            if (ns + c + 1 < N) out[c + 1] = __float2bfloat16(__uint_as_float(v1));
            if (ns + c + 2 < N) out[c + 2] = __float2bfloat16(__uint_as_float(v2));
            if (ns + c + 3 < N) out[c + 3] = __float2bfloat16(__uint_as_float(v3));
        }
    }
    tmem_fence_ld();

    if (lane == 0)
        tmem_dealloc(ta, TMC);
}

/* ================================================================
 * Host runner
 * ================================================================ */
namespace tvm_ffi_example_cuda {

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));

    uint32_t M = static_cast<uint32_t>(A.size(0));
    uint32_t N = static_cast<uint32_t>(B.size(0));
    uint32_t K = static_cast<uint32_t>(A.size(1));

    const __nv_bfloat16 *Ap = static_cast<const __nv_bfloat16 *>(A.data_ptr());
    const __nv_bfloat16 *Bp = static_cast<const __nv_bfloat16 *>(B.data_ptr());
    __nv_bfloat16       *Cp = static_cast<__nv_bfloat16 *>(C.data_ptr());

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    if (M == 0 || N == 0 || K == 0) return;

    uint32_t nlk = (N + CN - 1) / CN;
    uint32_t mrb = (M + CM - 1) / CM;
    uint32_t gx  = nlk * mrb * CSZ;

    dim3 blk(THR);
    dim3 grd(gx);

    size_t smem = TOTAL_SMEM;
    smem = (smem + 255) & ~255UL;

    cudaLaunchConfig_t cfg{};
    cfg.gridDim          = grd;
    cfg.blockDim         = blk;
    cfg.dynamicSmemBytes = smem;
    cfg.stream           = stream;

    cudaLaunchAttribute att{};
    att.id               = cudaLaunchAttributeClusterDimension;
    att.val.clusterDim.x = CSZ;
    att.val.clusterDim.y = 1;
    att.val.clusterDim.z = 1;
    cfg.attrs            = &att;
    cfg.numAttrs         = 1;

    cudaLaunchKernelEx(&cfg, bw_gemm_kernel, Ap, Bp, Cp, M, N, K, nlk);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace tvm_ffi_example_cuda

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);