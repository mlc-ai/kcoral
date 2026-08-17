#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <cstdio>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                    \
    cudaError_t _e = (call);                                     \
    if (_e != cudaSuccess) {                                     \
        fprintf(stderr, "CUDA error %s at %s:%d\n",              \
                cudaGetErrorString(_e), __FILE__, __LINE__);     \
        exit(1);                                                 \
    }                                                            \
} while(0)

#define CU_CHECK(call) do {                                      \
    CUresult _r = (call);                                        \
    if (_r != CUDA_SUCCESS) {                                    \
        const char* errStr;                                      \
        cuGetErrorString(_r, &errStr);                           \
        fprintf(stderr, "CU error %s at %s:%d\n",                \
                errStr, __FILE__, __LINE__);                     \
        exit(1);                                                 \
    }                                                            \
} while(0)

namespace tvm_ffi_example_cuda {

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int D = 128;
constexpr int THREADS = 128;

// ---- TMEM helpers (cta_group::1) ----
__device__ __forceinline__ void tmem_alloc_cg1(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_cg1(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
        :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tcgen05_fence_before() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_after() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

// ---- mbarrier helpers ----
__device__ __forceinline__ void init_barrier(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(count));
}

__device__ __forceinline__ void fence_barrier_init() {
    asm volatile("fence.mbarrier_init.release.cta;\n" ::: "memory");
}

__device__ __forceinline__ void arrive_expect_tx(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void arrive_barrier(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)) : "memory");
}

__device__ __forceinline__ void wait_barrier(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n.reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(phase));
}

// ---- TMA load (2D) ----
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

// ---- TMA store fence ----
__device__ __forceinline__ void tma_fence_async() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

// ---- tcgen05.mma (cta_group::1, kind::f16) ----
__device__ __forceinline__ void umma_cg1(uint32_t tmem_d, uint64_t desc_a,
    uint64_t desc_b, uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_d), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_cg1(uint64_t* bar) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)));
}

// ---- TMEM load ----
__device__ __forceinline__ void tmem_load_32x32b_x4(uint32_t col,
    uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ void tmem_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

// ---- SMEM descriptor (128B swizzle) ----
__device__ __forceinline__ uint64_t make_smem_desc(void* ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;  // 128B swizzle
    return d;
}

// ---- Instruction descriptor ----
__device__ __forceinline__ uint32_t make_idesc(uint32_t M, uint32_t N,
    bool trans_a, bool trans_b) {
    uint32_t d = 0;
    d |= (1u << 4);    // dtype = FP32
    d |= (1u << 7);    // atype = BF16
    d |= (1u << 10);   // btype = BF16
    d |= ((uint32_t)trans_a << 15);
    d |= ((uint32_t)trans_b << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

// ---- Create TMA descriptor ----
CUtensorMapDataType dataType = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;

CUresult create_tma_desc(CUtensorMap* d, void* ptr, uint64_t inner, uint64_t outer,
    uint32_t box_inner, uint32_t box_outer,
    CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2,
    CUtensorMapFloatOOBfill oob) {
    cuuint64_t globalDim[2] = {inner, outer};
    cuuint64_t globalStrides[1] = {inner * 2};
    cuuint32_t boxDim[2] = {box_inner, box_outer};
    cuuint32_t elemStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(d, dataType, 2, ptr, globalDim,
        globalStrides, boxDim, elemStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2, oob);
}

__global__ void attention_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S) {

    int bh = blockIdx.y;
    int b = bh / H;
    int h = bh % H;
    int q_block = blockIdx.x;
    int q_start = q_block * BM;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    const float scale = 0.08838834764831845f;

    // Shared memory layout
    extern __shared__ char smem_raw[];
    __nv_bfloat16* sQ0 = reinterpret_cast<__nv_bfloat16*>(smem_raw);       // 64*64*2 = 8KB
    __nv_bfloat16* sQ1 = sQ0 + 64 * 64;                                     // 8KB
    __nv_bfloat16* sK0 = sQ1 + 64 * 64;                                     // 8KB
    __nv_bfloat16* sK1 = sK0 + 64 * 64;                                     // 8KB
    __nv_bfloat16* sV0 = sK1 + 64 * 64;                                     // 8KB
    __nv_bfloat16* sV1 = sV0 + 64 * 64;                                     // 8KB
    __nv_bfloat16* sP  = sV1 + 64 * 64;                                     // 64*64*2 = 8KB
    float* s_rowmax = reinterpret_cast<float*>(sP + 64 * 64);               // 256B
    float* s_rowsum = s_rowmax + 64;                                        // 256B
    uint64_t* s_bar = reinterpret_cast<uint64_t*>(s_rowsum + 64);           // 2 * 8B
    uint32_t* s_tmem_addr = reinterpret_cast<uint32_t*>(s_bar + 2);         // 4B

    // Allocate TMEM: 256 columns (64 for S, 128 for O)
    if (warp_id == 0 && lane_id == 0) {
        tmem_alloc_cg1(s_tmem_addr, 256);
    }
    __syncthreads();
    uint32_t tmem_base = *s_tmem_addr;
    uint32_t tmem_S = tmem_base;        // S at col 0, 64 cols
    uint32_t tmem_O = tmem_base + 64;   // O at col 64, 128 cols

    // Init barriers
    if (tid == 0) {
        init_barrier(&s_bar[0], 1);
        init_barrier(&s_bar[1], 1);
        fence_barrier_init();
    }
    __syncthreads();

    // Compute outer coordinate for TMA (bh * S + row)
    int outer_coord_base = bh * S;

    // Load Q: 2 TMA loads for D=128 (each loads [64, 64])
    // Q half 0: D[0:64], Q half 1: D[64:128]
    if (tid == 0) {
        arrive_expect_tx(&s_bar[0], 64 * 64 * 2 * 2);  // 2 * 8KB = 16KB
        tma_load_2d(&tma_Q, &s_bar[0], sQ0, 0, outer_coord_base + q_start);
        tma_load_2d(&tma_Q, &s_bar[0], sQ1, 64, outer_coord_base + q_start);
    }
    wait_barrier(&s_bar[0], 0);
    __syncthreads();

    // Init rowmax, rowsum
    for (int i = tid; i < 64; i += THREADS) {
        s_rowmax[i] = -INFINITY;
        s_rowsum[i] = 0.0f;
    }

    // Instruction descriptors
    // QK^T: M=64, N=64, A=K-major (trans_a=0), B=MN-major (trans_b=1)
    uint32_t idesc_qk = make_idesc(64, 64, false, true);
    // PV: M=64, N=64, A=K-major (trans_a=0), B=K-major (trans_b=0)
    uint32_t idesc_pv = make_idesc(64, 64, false, false);

    // SMEM descriptors for 128B swizzle
    // K-major: SBO=1024, LBO=0
    // MN-major: SBO=1024, LBO=(64/8)*1024=8192
    uint64_t desc_Q0 = make_smem_desc(sQ0, 0, 1024);
    uint64_t desc_Q1 = make_smem_desc(sQ1, 0, 1024);
    uint64_t desc_K0 = make_smem_desc(sK0, 8192, 1024);
    uint64_t desc_K1 = make_smem_desc(sK1, 8192, 1024);
    uint64_t desc_V0 = make_smem_desc(sV0, 0, 1024);
    uint64_t desc_V1 = make_smem_desc(sV1, 0, 1024);

    __syncthreads();

    int max_k = min(S, q_start + BM);
    int num_k_blocks = (max_k + BN - 1) / BN;
    uint32_t phase = 0;

    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        int k_start = k_block * BN;

        // Load K, V via TMA (2 loads each for D=128)
        if (tid == 0) {
            arrive_expect_tx(&s_bar[0], 64 * 64 * 2 * 4);  // 4 * 8KB = 32KB
            tma_load_2d(&tma_K, &s_bar[0], sK0, 0, outer_coord_base + k_start);
            tma_load_2d(&tma_K, &s_bar[0], sK1, 64, outer_coord_base + k_start);
            tma_load_2d(&tma_V, &s_bar[0], sV0, 0, outer_coord_base + k_start);
            tma_load_2d(&tma_V, &s_bar[0], sV1, 64, outer_coord_base + k_start);
        }
        wait_barrier(&s_bar[0], phase);
        phase ^= 1;
        __syncthreads();

        // QK^T using tcgen05.mma
        // 8 MMAs: 2 D-halves * 4 K-steps (K=16 per MMA)
        // For 128B swizzle K-major, each K-step advances base by 16*2=32 bytes
        // For 128B swizzle MN-major, each K-step advances base by 16*2=32 bytes
        // Only thread 0 issues MMA (single-thread semantics)
        bool first_k_block = (k_block == 0);

        for (int half = 0; half < 2; half++) {
            uint64_t dQ = (half == 0) ? desc_Q0 : desc_Q1;
            uint64_t dK = (half == 0) ? desc_K0 : desc_K1;

            for (int kstep = 0; kstep < 4; kstep++) {
                uint32_t koff = kstep * 32;  // 16 BF16 * 2 bytes = 32 bytes
                uint64_t a_desc = make_smem_desc(
                    (__nv_bfloat16*)((char*)sQ0 + half * 64 * 64 * 2 + koff), 0, 1024);
                uint64_t b_desc = make_smem_desc(
                    (__nv_bfloat16*)((char*)sK0 + half * 64 * 64 * 2 + koff), 8192, 1024);

                bool accum = !(first_k_block && half == 0 && kstep == 0);
                if (tid == 0) {
                    umma_cg1(tmem_S, a_desc, b_desc, idesc_qk, accum ? 1 : 0);
                }
            }
        }

        // Wait for QK^T completion
        if (tid == 0) {
            umma_commit_cg1(&s_bar[1]);
        }
        wait_barrier(&s_bar[1], phase & 1);
        __syncthreads();

        // Softmax: read S from TMEM, compute P, write to shared memory
        // Warp 0 reads rows 0-31, warp 1 reads rows 32-63
        // Each thread reads one row, 64 FP32 values
        if (warp_id < 2) {
            int row = warp_id * 32 + lane_id;
            int q_pos = q_start + row;
            float s_vals[64];

            // Load 64 FP32 from TMEM (4 loads of 16)
            for (int c = 0; c < 64; c += 16) {
                uint32_t r0, r1, r2, r3;
                tmem_load_32x32b_x4(tmem_S + c, &r0, &r1, &r2, &r3);
                tmem_wait_ld();
                s_vals[c]   = __uint_as_float(r0);
                s_vals[c+1] = __uint_as_float(r1);
                s_vals[c+2] = __uint_as_float(r2);
                s_vals[c+3] = __uint_as_float(r3);
            }

            if (q_pos < S) {
                // Apply scale and causal mask, find max
                float row_max = -INFINITY;
                for (int j = 0; j < 64; j++) {
                    int k_pos = k_start + j;
                    if (k_pos > q_pos || k_pos >= S) {
                        s_vals[j] = -INFINITY;
                    } else {
                        s_vals[j] *= scale;
                        row_max = fmaxf(row_max, s_vals[j]);
                    }
                }

                // Online softmax update
                float old_max = s_rowmax[row];
                float new_max = fmaxf(old_max, row_max);
                float exp_old = (old_max == -INFINITY) ? 0.0f : __expf(old_max - new_max);

                float row_sum = 0.0f;
                for (int j = 0; j < 64; j++) {
                    if (s_vals[j] == -INFINITY) {
                        s_vals[j] = 0.0f;
                    } else {
                        s_vals[j] = __expf(s_vals[j] - new_max);
                        row_sum += s_vals[j];
                    }
                }

                // Write P to shared memory with 128B swizzle
                // P[row][chunk] -> sP[row * 64 + ((row%8) ^ chunk) * 8]
                for (int chunk = 0; chunk < 8; chunk++) {
                    int phys_chunk = (row % 8) ^ chunk;
                    int base = row * 64 + phys_chunk * 8;
                    for (int e = 0; e < 8; e++) {
                        sP[base + e] = __float2bfloat16(s_vals[chunk * 8 + e]);
                    }
                }

                // Update rowmax, rowsum
                s_rowmax[row] = new_max;
                s_rowsum[row] = s_rowsum[row] * exp_old + row_sum;

                // Rescale O in TMEM: read, multiply, write back
                // O is 64 rows x 128 cols FP32 at tmem_O
                // Each thread reads 128 FP32, multiplies by exp_old, writes back
                if (exp_old != 1.0f) {
                    for (int c = 0; c < 128; c += 4) {
                        uint32_t r0, r1, r2, r3;
                        tmem_load_32x32b_x4(tmem_O + c, &r0, &r1, &r2, &r3);
                        tmem_wait_ld();
                        r0 = __float_as_uint(__uint_as_float(r0) * exp_old);
                        r1 = __float_as_uint(__uint_as_float(r1) * exp_old);
                        r2 = __float_as_uint(__uint_as_float(r2) * exp_old);
                        r3 = __float_as_uint(__uint_as_float(r3) * exp_old);
                        // Store back to TMEM using tcgen05.st
                        asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
                            :: "r"(tmem_O + c), "r"(r0), "r"(r1), "r"(r2), "r"(r3));
                    }
                }
            } else {
                // Write zeros to P
                for (int chunk = 0; chunk < 8; chunk++) {
                    int phys_chunk = (row % 8) ^ chunk;
                    int base = row * 64 + phys_chunk * 8;
                    for (int e = 0; e < 8; e++) {
                        sP[base + e] = __float2bfloat16(0.0f);
                    }
                }
            }
        }
        __syncthreads();

        // PV using tcgen05.mma
        // P is [64, 64] K-major 128B swizzle, V is [64, 64] K-major 128B swizzle
        // O is [64, 128] in TMEM, but we do 2 halves of N=64
        uint64_t desc_P = make_smem_desc(sP, 0, 1024);

        for (int half = 0; half < 2; half++) {
            uint64_t dV = (half == 0) ? desc_V0 : desc_V1;
            uint32_t tmem_O_half = tmem_O + half * 64;

            for (int kstep = 0; kstep < 4; kstep++) {
                uint32_t koff = kstep * 32;
                uint64_t a_desc = make_smem_desc(
                    (__nv_bfloat16*)((char*)sP + koff), 0, 1024);
                uint64_t b_desc = make_smem_desc(
                    (__nv_bfloat16*)((char*)sV0 + half * 64 * 64 * 2 + koff), 0, 1024);

                bool accum = !(first_k_block && half == 0 && kstep == 0);
                if (tid == 0) {
                    umma_cg1(tmem_O_half, a_desc, b_desc, idesc_pv, accum ? 1 : 0);
                }
            }
        }

        // Wait for PV completion
        if (tid == 0) {
            umma_commit_cg1(&s_bar[1]);
        }
        wait_barrier(&s_bar[1], (phase + 1) & 1);
        __syncthreads();
    }

    // Epilogue: read O from TMEM, normalize, store to global
    int64_t out_base = (int64_t)(b * H + h) * S * D;

    // Each of 128 threads handles 1 row of O (64 rows, 2 passes for 128 cols)
    // Actually use 64 threads for 64 rows, 2 warps
    if (warp_id < 2) {
        int row = warp_id * 32 + lane_id;
        int q_pos = q_start + row;
        if (q_pos < S) {
            float sum = s_rowsum[row] + 1e-30f;
            float lse_val = s_rowmax[row] + logf(sum);
            LSE[(int64_t)(b * H + h) * S + q_pos] = lse_val;

            // Read O from TMEM: 128 FP32 values
            for (int c = 0; c < 128; c += 8) {
                uint32_t r0, r1, r2, r3;
                tmem_load_32x32b_x4(tmem_O + c, &r0, &r1, &r2, &r3);
                tmem_wait_ld();
                float f0 = __uint_as_float(r0) / sum;
                float f1 = __uint_as_float(r1) / sum;
                float f2 = __uint_as_float(r2) / sum;
                float f3 = __uint_as_float(r3) / sum;

                tmem_load_32x32b_x4(tmem_O + c + 4, &r0, &r1, &r2, &r3);
                tmem_wait_ld();
                float f4 = __uint_as_float(r0) / sum;
                float f5 = __uint_as_float(r1) / sum;
                float f6 = __uint_as_float(r2) / sum;
                float f7 = __uint_as_float(r3) / sum;

                __nv_bfloat16* out_ptr = O + out_base + (int64_t)q_pos * D + c;
                out_ptr[0] = __float2bfloat16(f0);
                out_ptr[1] = __float2bfloat16(f1);
                out_ptr[2] = __float2bfloat16(f2);
                out_ptr[3] = __float2bfloat16(f3);
                out_ptr[4] = __float2bfloat16(f4);
                out_ptr[5] = __float2bfloat16(f5);
                out_ptr[6] = __float2bfloat16(f6);
                out_ptr[7] = __float2bfloat16(f7);
            }
        }
    }

    __syncthreads();
    // Deallocate TMEM
    if (warp_id == 0 && lane_id == 0) {
        tmem_dealloc_cg1(tmem_base, 256);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K,
         tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    const int B = 4;
    const int H = 48;
    const int S = static_cast<int>(Q.size(2));

    __nv_bfloat16* Q_ptr = static_cast<__nv_bfloat16*>(Q.data_ptr());
    __nv_bfloat16* K_ptr = static_cast<__nv_bfloat16*>(K.data_ptr());
    __nv_bfloat16* V_ptr = static_cast<__nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_ptr = static_cast<float*>(LSE.data_ptr());

    // Create TMA descriptors
    // 2D: inner=D=128, outer=S*B*H
    uint64_t outer = (uint64_t)S * B * H;
    CUtensorMap tma_Q, tma_K, tma_V;

    CU_CHECK(create_tma_desc(&tma_Q, Q_ptr, 128, outer, 64, 64,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_desc(&tma_K, K_ptr, 128, outer, 64, 64,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_desc(&tma_V, V_ptr, 128, outer, 64, 64,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    dim3 grid((S + BM - 1) / BM, B * H);
    dim3 block(THREADS);

    int smem_size = 6 * 64 * 64 * 2   // sQ0,sQ1,sK0,sK1,sV0,sV1 (6 * 8KB = 48KB)
                  + 64 * 64 * 2       // sP (8KB)
                  + 64 * 4 * 2        // s_rowmax, s_rowsum
                  + 2 * 8             // s_bar
                  + 4;                // s_tmem_addr

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    cudaFuncSetAttribute(
        attention_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    attention_kernel<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, O_ptr, LSE_ptr, B, H, S);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda