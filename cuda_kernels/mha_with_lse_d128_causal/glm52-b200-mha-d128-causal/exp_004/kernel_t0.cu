#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

namespace mha_lse_d128_causal {

// ---- Helper functions ----

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
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

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_before_fn() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
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

__device__ __forceinline__ uint64_t make_smem_desc_noswizzle(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   // version = 1
    d |= (uint64_t)0 << 61;   // no swizzle
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
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1"
        ".mbarrier::arrive::one.shared::cluster.b64"
        " [%0];"
        :: "r"(a));
}

__device__ __forceinline__ uint32_t pack_bf16_simple(__nv_bfloat16 a, __nv_bfloat16 b) {
    uint16_t ha = *(uint16_t*)&a;
    uint16_t hb = *(uint16_t*)&b;
    return (uint32_t)ha | ((uint32_t)hb << 16);
}

// ---- Constants ----
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int D = 128;
constexpr int BK = 16;
constexpr int NUM_D_CHUNKS = D / BK; // 8
constexpr int NUM_COL_GROUPS = D / 8; // 16
constexpr int KMAJOR_LBO = 2048;
constexpr int KMAJOR_SBO = 128;
constexpr int MNMAJOR_LBO = 128;
constexpr int MNMAJOR_SBO = 2048;

// ---- Kernel ----
__global__ void attention_kernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* LSE,
    int S, int H) {

    int q_block = blockIdx.x;
    int bh = blockIdx.y;
    int b = bh / H;
    int h = bh % H;
    int s_start = q_block * BM;
    int row = threadIdx.x;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    extern __shared__ char smem[];
    char* smem_Q = smem;
    char* smem_KV = smem + 32768;
    char* smem_P = smem + 65536;
    uint64_t* smem_bar_qkt = (uint64_t*)(smem + 98304);
    uint64_t* smem_bar_pv = (uint64_t*)(smem + 98312);
    uint32_t* smem_tmem_addr = (uint32_t*)(smem + 98320);

    // Init barriers
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(smem_bar_qkt, 1);
        init_smem_barrier_fn(smem_bar_pv, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    // TMEM alloc
    if (warp_id == 0) {
        tmem_alloc_cg1_fn(smem_tmem_addr, 256);
    }
    __syncthreads();
    uint32_t tmem_base = *smem_tmem_addr;
    uint32_t tmem_s = tmem_base;
    uint32_t tmem_o = tmem_base + 128;

    // Instruction descriptors
    // QK^T: A=K-major, B=K-major, M=128, N=128
    uint32_t idesc_qkt = 0;
    idesc_qkt |= (1u << 4);    // D type = FP32
    idesc_qkt |= (1u << 7);    // A type = BF16
    idesc_qkt |= (1u << 10);   // B type = BF16
    idesc_qkt |= (0u << 15);   // A K-major
    idesc_qkt |= (0u << 16);   // B K-major
    idesc_qkt |= ((128 / 8) << 17);  // N >> 3 = 16
    idesc_qkt |= ((128 / 16) << 24); // M >> 4 = 8

    // PV: A=K-major, B=MN-major (transpose)
    uint32_t idesc_pv = idesc_qkt | (1u << 16);

    const float scale = 0.08838834764831845f; // 1/sqrt(128)
    const float log2e = 1.4426950408889634f;

    // Load Q to shared (K-major tiled)
    const __nv_bfloat16* Q_bh = Q + (uint64_t)(bh) * S * D;
    if (s_start + row < S) {
        const uint4* gptr = reinterpret_cast<const uint4*>(Q_bh + (uint64_t)(s_start + row) * D);
        #pragma unroll
        for (int c = 0; c < NUM_COL_GROUPS; c++) {
            uint4 data = gptr[c];
            *reinterpret_cast<uint4*>(smem_Q + c * KMAJOR_LBO + row * 16) = data;
        }
    } else {
        #pragma unroll
        for (int c = 0; c < NUM_COL_GROUPS; c++) {
            *reinterpret_cast<uint4*>(smem_Q + c * KMAJOR_LBO + row * 16) = make_uint4(0, 0, 0, 0);
        }
    }
    __syncthreads();

    // Online softmax state
    float m_old = -INFINITY;
    float l_old = 0.0f;
    int q_pos = s_start + row;

    int num_k_blocks = min(q_block + 1, (S + BM - 1) / BM);

    for (int k_block = 0; k_block <= q_block; k_block++) {
        int k_start = k_block * BM;
        if (k_start >= S) break;

        // Load K to shared (K-major tiled)
        const __nv_bfloat16* K_bh = K + (uint64_t)(bh) * S * D;
        if (k_start + row < S) {
            const uint4* gptr = reinterpret_cast<const uint4*>(K_bh + (uint64_t)(k_start + row) * D);
            #pragma unroll
            for (int c = 0; c < NUM_COL_GROUPS; c++) {
                uint4 data = gptr[c];
                *reinterpret_cast<uint4*>(smem_KV + c * KMAJOR_LBO + row * 16) = data;
            }
        } else {
            #pragma unroll
            for (int c = 0; c < NUM_COL_GROUPS; c++) {
                *reinterpret_cast<uint4*>(smem_KV + c * KMAJOR_LBO + row * 16) = make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();
        fence_async_shared_fn();

        // Compute S = Q @ K^T
        if (threadIdx.x == 0) {
            #pragma unroll
            for (int d = 0; d < NUM_D_CHUNKS; d++) {
                uint64_t desc_a = make_smem_desc_noswizzle(smem_Q + d * 4096, KMAJOR_LBO, KMAJOR_SBO);
                uint64_t desc_b = make_smem_desc_noswizzle(smem_KV + d * 4096, KMAJOR_LBO, KMAJOR_SBO);
                uint32_t accum = (d == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_s, desc_a, desc_b, idesc_qkt, accum);
            }
            umma_commit_cg1_fn(smem_bar_qkt);
        }
        mbarrier_wait_fn(smem_bar_qkt, k_block % 2);

        // === Softmax Phase 1: Compute rowmax ===
        float rowmax = -INFINITY;
        #pragma unroll 8
        for (int col = 0; col < BN; col += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_s + col));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

            float vals[4] = {__uint_as_float(r0), __uint_as_float(r1), __uint_as_float(r2), __uint_as_float(r3)};
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int k_pos = k_start + col + i;
                float v = vals[i];
                if (k_block == q_block && k_pos > q_pos) v = -INFINITY;
                if (k_pos >= S) v = -INFINITY;
                if (q_pos >= S) v = -INFINITY;
                rowmax = fmaxf(rowmax, v);
            }
        }

        float m_new = fmaxf(m_old, rowmax);
        float alpha = (m_old == -INFINITY) ? 0.0f : fast_exp2f_fn((m_old - m_new) * log2e);
        m_old = m_new;

        // === Softmax Phase 2: Compute P, store to smem_P, accumulate rowsum ===
        float rowsum = 0.0f;
        #pragma unroll 8
        for (int col = 0; col < BN; col += 8) {
            uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_s + col));
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(tmem_s + col + 4));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

            float vals[8] = {__uint_as_float(r0), __uint_as_float(r1), __uint_as_float(r2),
                             __uint_as_float(r3), __uint_as_float(r4), __uint_as_float(r5),
                             __uint_as_float(r6), __uint_as_float(r7)};
            __nv_bfloat16 bf16_vals[8];

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int k_pos = k_start + col + i;
                float v = vals[i];
                if (k_block == q_block && k_pos > q_pos) v = -INFINITY;
                if (k_pos >= S) v = -INFINITY;
                if (q_pos >= S) v = -INFINITY;
                float p = (v == -INFINITY) ? 0.0f : fast_exp2f_fn((v * scale - m_new) * log2e);
                rowsum += p;
                bf16_vals[i] = __float2bfloat16(p);
            }

            // Pack and store to smem_P (K-major tiled)
            uint32_t p0 = pack_bf16_simple(bf16_vals[0], bf16_vals[1]);
            uint32_t p1 = pack_bf16_simple(bf16_vals[2], bf16_vals[3]);
            uint32_t p2 = pack_bf16_simple(bf16_vals[4], bf16_vals[5]);
            uint32_t p3 = pack_bf16_simple(bf16_vals[6], bf16_vals[7]);
            int kg = col / 8;
            *reinterpret_cast<uint4*>(smem_P + kg * KMAJOR_LBO + row * 16) = make_uint4(p0, p1, p2, p3);
        }

        l_old = l_old * alpha + rowsum;
        __syncthreads();

        // === Rescale O in TMEM ===
        if (k_block > 0) {
            #pragma unroll 4
            for (int col = 0; col < BN; col += 4) {
                uint32_t r0, r1, r2, r3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_o + col));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

                float f0 = __uint_as_float(r0) * alpha;
                float f1 = __uint_as_float(r1) * alpha;
                float f2 = __uint_as_float(r2) * alpha;
                float f3 = __uint_as_float(r3) * alpha;

                asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
                    :: "r"(tmem_o + col), "r"(__float_as_uint(f0)), "r"(__float_as_uint(f1)),
                       "r"(__float_as_uint(f2)), "r"(__float_as_uint(f3)));
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            tcgen05_fence_before_fn();
            __syncthreads();
        }

        // === Load V to shared (MN-major tiled) ===
        const __nv_bfloat16* V_bh = V + (uint64_t)(bh) * S * D;
        int k_idx = k_start + row;
        int kt = row / 8;
        int kr = row % 8;
        if (k_idx < S) {
            const uint4* gptr = reinterpret_cast<const uint4*>(V_bh + (uint64_t)k_idx * D);
            #pragma unroll
            for (int nt = 0; nt < NUM_COL_GROUPS; nt++) {
                uint4 data = gptr[nt];
                *reinterpret_cast<uint4*>(smem_KV + kt * MNMAJOR_LBO + nt * MNMAJOR_SBO + kr * 16) = data;
            }
        } else {
            #pragma unroll
            for (int nt = 0; nt < NUM_COL_GROUPS; nt++) {
                *reinterpret_cast<uint4*>(smem_KV + kt * MNMAJOR_LBO + nt * MNMAJOR_SBO + kr * 16) = make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();
        fence_async_shared_fn();

        // === Compute O += P @ V ===
        if (threadIdx.x == 0) {
            #pragma unroll
            for (int d = 0; d < NUM_D_CHUNKS; d++) {
                // P descriptor (K-major): base at smem_P + d * 4096
                uint64_t desc_p = make_smem_desc_noswizzle(smem_P + d * 4096, KMAJOR_LBO, KMAJOR_SBO);
                // V descriptor (MN-major): base at smem_KV + d * 256
                uint64_t desc_v = make_smem_desc_noswizzle(smem_KV + d * 256, MNMAJOR_LBO, MNMAJOR_SBO);
                uint32_t accum = (k_block == 0 && d == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_o, desc_p, desc_v, idesc_pv, accum);
            }
            umma_commit_cg1_fn(smem_bar_pv);
        }
        mbarrier_wait_fn(smem_bar_pv, k_block % 2);
    }

    // === Epilogue: Final normalization and store ===
    float inv_l = (l_old > 0.0f) ? (1.0f / l_old) : 0.0f;

    __syncthreads(); // Ensure previous MMA fully done

    #pragma unroll 4
    for (int col = 0; col < D; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_o + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

        if (q_pos < S) {
            float f0 = __uint_as_float(r0) * inv_l;
            float f1 = __uint_as_float(r1) * inv_l;
            float f2 = __uint_as_float(r2) * inv_l;
            float f3 = __uint_as_float(r3) * inv_l;

            __nv_bfloat16* O_ptr = O + (uint64_t)(bh) * S * D + (uint64_t)q_pos * D + col;
            O_ptr[0] = __float2bfloat16(f0);
            O_ptr[1] = __float2bfloat16(f1);
            O_ptr[2] = __float2bfloat16(f2);
            O_ptr[3] = __float2bfloat16(f3);
        }
    }

    if (q_pos < S) {
        float lse = (l_old > 0.0f) ? (m_old + logf(l_old)) : -INFINITY;
        LSE[(uint64_t)(bh) * S + q_pos] = lse;
    }

    // Dealloc TMEM
    __syncthreads();
    if (warp_id == 0) {
        tmem_dealloc_cg1_fn(tmem_base, 256);
    }
}

// ---- Host function ----
void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);

    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_ptr = static_cast<float*>(LSE.data_ptr());

    int num_q_blocks = (S + BM - 1) / BM;
    dim3 grid(num_q_blocks, B * H);
    dim3 block(BM);

    size_t smem_size = 98368; // 3*32KB + barriers + tmem_addr

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    attention_kernel<<<grid, block, smem_size, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr, (int)S, (int)H);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_lse_d128_causal::run);

}  // namespace mha_lse_d128_causal