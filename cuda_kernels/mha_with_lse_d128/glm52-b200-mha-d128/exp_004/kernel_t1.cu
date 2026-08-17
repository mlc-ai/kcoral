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

namespace tvm_ffi_mha_cuda {

constexpr int D_HEAD = 128;
constexpr int BM = 128;
constexpr int BK = 64;
constexpr int NUM_THREADS = 128;
constexpr float SCALE = 0.08838834764f;

constexpr int Q_SMEM_OFF = 0;
constexpr int KV_SMEM_OFF = 32768;
constexpr int P_SMEM_OFF = 49152;
constexpr int ML_SMEM_OFF = 65536;
constexpr int MBAR_OFF = 66560;
constexpr int TMEM_ADDR_OFF = 66624;
constexpr int SMEM_SIZE = 67584 + 1024;

// ---- cta_group::1 UMMA helpers ----
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

__device__ __forceinline__ void tmem_store_4x_fn(uint32_t col, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1, %2, %3, %4};"
   :: "r"(col), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
}

__device__ __forceinline__ void tmem_store_wait_fn() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ uint64_t advance_desc(uint64_t desc, uint32_t byte_offset) {
    uint64_t base = desc & 0x3FFF;
    uint64_t new_base = base + (byte_offset >> 4);
    return (desc & 0xFFFFFFFFFFFFC000ULL) | (new_base & 0x3FFF);
}

// ---- TMA descriptor creation ----
static CUresult create_tma_desc_bf16(
    CUtensorMap* d, void* ptr,
    uint64_t gmem_inner, uint64_t gmem_outer,
    uint32_t smem_inner, uint32_t smem_outer,
    CUtensorMapSwizzle swizzle,
    CUtensorMapFloatOOBfill oob_fill)
{
    cuuint64_t globalDim[2] = {gmem_inner, gmem_outer};
    cuuint64_t globalStrides[1] = {gmem_inner * 2};
    cuuint32_t boxDim[2] = {smem_inner, smem_outer};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        ptr, globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, oob_fill);
}

// ---- Kernel ----
__global__ void attention_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S, int B, int H)
{
    const int q_block = blockIdx.x;
    const int bh = blockIdx.y;
    const int b = bh / H;
    const int h = bh % H;
    const int q_start = q_block * BM;
    const int tid = threadIdx.x;
    const bool row_valid = (q_start + tid < S);

    extern __shared__ char smem_raw[];
    char* smem = (char*)(((uintptr_t)smem_raw + 1023) & ~1023);

    __nv_bfloat16* Q_smem = reinterpret_cast<__nv_bfloat16*>(smem + Q_SMEM_OFF);
    __nv_bfloat16* KV_smem = reinterpret_cast<__nv_bfloat16*>(smem + KV_SMEM_OFF);
    __nv_bfloat16* P_smem = reinterpret_cast<__nv_bfloat16*>(smem + P_SMEM_OFF);
    float* m_smem = reinterpret_cast<float*>(smem + ML_SMEM_OFF);
    float* l_smem = m_smem + BM;
    uint64_t* mbar = reinterpret_cast<uint64_t*>(smem + MBAR_OFF);
    uint32_t* tmem_addr_smem = reinterpret_cast<uint32_t*>(smem + TMEM_ADDR_OFF);

    // ---- TMEM allocation ----
    const int warp_id = tid / 32;
    if (warp_id == 0) {
        tmem_alloc_cg1_fn(tmem_addr_smem, 256);
    }
    __syncthreads();
    uint32_t tmem_base = tmem_addr_smem[0];
    uint32_t tmem_S = tmem_base;
    uint32_t tmem_O = tmem_base + BK; // O at column 64

    // ---- Init mbarriers ----
    if (tid == 0) {
        init_smem_barrier_fn(&mbar[0], 1);
        init_smem_barrier_fn(&mbar[1], 1);
        init_smem_barrier_fn(&mbar[2], 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    // ---- Init m, l ----
    if (tid < BM) {
        m_smem[tid] = -INFINITY;
        l_smem[tid] = 0.0f;
    }

    // ---- Prefetch TMA descriptors ----
    prefetch_tma_descriptor_fn(&tma_Q);
    prefetch_tma_descriptor_fn(&tma_K);
    prefetch_tma_descriptor_fn(&tma_V);

    // ---- Load Q via TMA (2 loads for D=128) ----
    int phase0 = 0, phase1 = 0, phase2 = 0;
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(&mbar[0], 16384);
        tma_load_2d_fn(&tma_Q, &mbar[0], Q_smem, 0, q_start);
        mbarrier_arrive_and_expect_tx_fn(&mbar[1], 16384);
        tma_load_2d_fn(&tma_Q, &mbar[1], Q_smem + 8192, 64, q_start);
    }
    mbarrier_wait_fn(&mbar[0], phase0); phase0 ^= 1;
    mbarrier_wait_fn(&mbar[1], phase1); phase1 ^= 1;

    // ---- Descriptors ----
    uint64_t q_desc = make_smem_desc_sm100_fn(Q_smem, 1, 1024);
    uint64_t k_desc = make_smem_desc_sm100_fn(KV_smem, 1, 1024);
    uint64_t p_desc = make_smem_desc_sm100_fn(P_smem, 1, 1024);
    uint64_t v_desc = make_smem_desc_sm100_fn(KV_smem, 8192, 1024);

    uint32_t idesc_qk = make_instr_desc_fn(BM, BK);
    uint32_t idesc_pv = make_instr_desc_fn(BM, D_HEAD) | (1u << 16); // b_major=1 (N-major for V)

    // K-span offsets
    const uint32_t q_kspan_off = (BM / 8) * 1024;   // 16384
    const uint32_t k_kspan_off = (BK / 8) * 1024;    // 8192

    // ---- Main loop over KV blocks ----
    for (int kv_start = 0; kv_start < S; kv_start += BK) {
        int kv_len = min(kv_start + BK, S) - kv_start;
        bool first_block = (kv_start == 0);

        // ---- Load K via TMA ----
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(&mbar[0], 8192);
            tma_load_2d_fn(&tma_K, &mbar[0], KV_smem, 0, kv_start);
            mbarrier_arrive_and_expect_tx_fn(&mbar[1], 8192);
            tma_load_2d_fn(&tma_K, &mbar[1], KV_smem + 4096, 64, kv_start);
        }
        mbarrier_wait_fn(&mbar[0], phase0); phase0 ^= 1;
        mbarrier_wait_fn(&mbar[1], phase1); phase1 ^= 1;

        // ---- UMMA QK^T (K=128, 8 instructions of K=16) ----
        if (tid == 0) {
            #pragma unroll
            for (int k = 0; k < 128; k += 16) {
                int k_span = k / 64;
                int k_chunk = (k % 64) / 16;
                uint32_t q_off = k_span * q_kspan_off + k_chunk * 32;
                uint32_t k_off = k_span * k_kspan_off + k_chunk * 32;
                uint64_t a_d = advance_desc(q_desc, q_off);
                uint64_t b_d = advance_desc(k_desc, k_off);
                uint32_t accum = (k == 0) ? 0u : 1u;
                umma_f16_cg1_fn(tmem_S, a_d, b_d, idesc_qk, accum);
            }
            umma_commit_cg1_fn(&mbar[2]);
        }
        mbarrier_wait_fn(&mbar[2], phase2); phase2 ^= 1;
        tcgen05_fence_after_fn();

        // ---- Issue V load (overlap with softmax) ----
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(&mbar[0], 8192);
            tma_load_2d_fn(&tma_V, &mbar[0], KV_smem, 0, kv_start);
            mbarrier_arrive_and_expect_tx_fn(&mbar[1], 8192);
            tma_load_2d_fn(&tma_V, &mbar[1], KV_smem + 4096, 64, kv_start);
        }

        // ---- Softmax: read S from TMEM, compute P, write to SMEM ----
        float m_old = row_valid ? m_smem[tid] : -INFINITY;
        float l_old = row_valid ? l_smem[tid] : 0.0f;
        float row_max = -INFINITY;

        // Pass 1: find row max
        for (int c = 0; c < BK; c += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_S + c, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            if (row_valid) {
                float s0 = __uint_as_float(r0) * SCALE;
                float s1 = __uint_as_float(r1) * SCALE;
                float s2 = __uint_as_float(r2) * SCALE;
                float s3 = __uint_as_float(r3) * SCALE;
                if (c + 0 >= kv_len) s0 = -INFINITY;
                if (c + 1 >= kv_len) s1 = -INFINITY;
                if (c + 2 >= kv_len) s2 = -INFINITY;
                if (c + 3 >= kv_len) s3 = -INFINITY;
                row_max = fmaxf(row_max, fmaxf(fmaxf(s0, s1), fmaxf(s2, s3)));
            }
        }

        float m_new = row_valid ? fmaxf(m_old, row_max) : -INFINITY;

        // Pass 2: compute P = exp(S - m_new), write to swizzled SMEM, compute sum
        float row_sum = 0.0f;
        int row = tid;
        int m_group = row / 8;
        int row_in_atom = row % 8;

        for (int c = 0; c < BK; c += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_S + c, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();

            float p[4];
            p[0] = __uint_as_float(r0) * SCALE;
            p[1] = __uint_as_float(r1) * SCALE;
            p[2] = __uint_as_float(r2) * SCALE;
            p[3] = __uint_as_float(r3) * SCALE;

            if (row_valid) {
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    if (c + j >= kv_len) p[j] = -INFINITY;
                    p[j] = __expf(p[j] - m_new);
                    row_sum += p[j];
                }
            }

            // Write to swizzled P_smem
            int chunk = c / 8;
            int swizzled_chunk = row_in_atom ^ chunk;
            int byte_addr = m_group * 1024 + row_in_atom * 128 + swizzled_chunk * 16 + (c % 8) * 2;
            __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(
                reinterpret_cast<char*>(P_smem) + byte_addr);
            if (c % 8 == 0) {
                dst[0] = __float2bfloat16(row_valid ? p[0] : 0.0f);
                dst[1] = __float2bfloat16(row_valid ? p[1] : 0.0f);
                dst[2] = __float2bfloat16(row_valid ? p[2] : 0.0f);
                dst[3] = __float2bfloat16(row_valid ? p[3] : 0.0f);
            }
        }

        if (row_valid && row == 0) {
            // Handle remaining chunks (c=4..7 within first chunk group)
        }

        // Actually, let me redo the P write more carefully
        // Each thread writes 64 BF16 in 8 chunks of 8
        // But I'm loading 4 at a time, so I need to accumulate 2 loads per chunk

        // Let me re-do this properly. I'll load all 64 values, then write in chunks of 8.

        // Actually, let me simplify: load 8 at a time, write 8 at a time

        // Redo pass 2 with 8-element loads
        row_sum = 0.0f;
        for (int c = 0; c < BK; c += 8) {
            uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
            tmem_load_8x_fn(tmem_S + c, &r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);
            tmem_load_fence_fn();

            float p[8];
            #pragma unroll
            for (int j = 0; j < 8; j++) p[j] = 0.0f;

            if (row_valid) {
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    float sv = __uint_as_float((&r0)[j]) * SCALE;
                    if (c + j >= kv_len) sv = -INFINITY;
                    p[j] = __expf(sv - m_new);
                    row_sum += p[j];
                }
            }

            // Write 8 BF16 to swizzled SMEM
            int chunk = c / 8;
            int swizzled_chunk = row_in_atom ^ chunk;
            int byte_addr = m_group * 1024 + row_in_atom * 128 + swizzled_chunk * 16;
            __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(
                reinterpret_cast<char*>(P_smem) + byte_addr);
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                dst[j] = __float2bfloat16(p[j]);
            }
        }

        // Update m, l
        if (row_valid) {
            float rescale_factor = (m_old != -INFINITY) ? __expf(m_old - m_new) : 1.0f;
            l_smem[tid] = l_old * rescale_factor + row_sum;
            m_smem[tid] = m_new;
        }

        // ---- Conditional rescale O ----
        bool need_rescale = row_valid && (m_old != -INFINITY) && (m_new != m_old);
        uint32_t mask = __ballot_sync(0xFFFFFFFF, need_rescale);
        if (mask != 0) {
            float rescale = need_rescale ? __expf(m_old - m_new) : 1.0f;
            for (int c = 0; c < D_HEAD; c += 4) {
                uint32_t r0, r1, r2, r3;
                tmem_load_4x_fn(tmem_O + c, &r0, &r1, &r2, &r3);
                tmem_load_fence_fn();
                r0 = __float_as_uint(__uint_as_float(r0) * rescale);
                r1 = __float_as_uint(__uint_as_float(r1) * rescale);
                r2 = __float_as_uint(__uint_as_float(r2) * rescale);
                r3 = __float_as_uint(__uint_as_float(r3) * rescale);
                tmem_store_4x_fn(tmem_O + c, r0, r1, r2, r3);
            }
            tmem_store_wait_fn();
        }

        // ---- Fence P writes, wait for V ----
        __syncthreads();
        fence_async_shared_fn();
        mbarrier_wait_fn(&mbar[0], phase0); phase0 ^= 1;
        mbarrier_wait_fn(&mbar[1], phase1); phase1 ^= 1;

        // ---- UMMA PV (K=64, 4 instructions of K=16) ----
        if (tid == 0) {
            #pragma unroll
            for (int k = 0; k < BK; k += 16) {
                int k_chunk = k / 16;
                uint32_t p_off = k_chunk * 32;
                uint32_t v_off = (k / 8) * 1024; // K-group offset for N-major
                uint64_t a_d = advance_desc(p_desc, p_off);
                uint64_t b_d = advance_desc(v_desc, v_off);
                uint32_t accum = (k == 0 && first_block) ? 0u : 1u;
                umma_f16_cg1_fn(tmem_O, a_d, b_d, idesc_pv, accum);
            }
            umma_commit_cg1_fn(&mbar[2]);
        }
        mbarrier_wait_fn(&mbar[2], phase2); phase2 ^= 1;
        tcgen05_fence_after_fn();
    }

    // ---- Final normalization and output ----
    __syncthreads();

    float final_l = row_valid ? l_smem[tid] : 0.0f;
    float final_m = row_valid ? m_smem[tid] : -INFINITY;
    float inv_l = (final_l > 0.0f) ? (1.0f / final_l) : 0.0f;

    // Write LSE
    if (row_valid && final_l > 0.0f) {
        size_t lse_offset = (size_t)(b * H + h) * S + q_start + tid;
        LSE[lse_offset] = final_m + logf(final_l);
    }

    // Phase 1: TMEM -> SMEM (normalized BF16, standard row-major, reuse Q_smem)
    for (int c = 0; c < D_HEAD; c += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn(tmem_O + c, &r0, &r1, &r2, &r3);
        tmem_load_fence_fn();
        float f0 = __uint_as_float(r0) * inv_l;
        float f1 = __uint_as_float(r1) * inv_l;
        float f2 = __uint_as_float(r2) * inv_l;
        float f3 = __uint_as_float(r3) * inv_l;
        int base = tid * D_HEAD + c;
        Q_smem[base + 0] = __float2bfloat16(f0);
        Q_smem[base + 1] = __float2bfloat16(f1);
        Q_smem[base + 2] = __float2bfloat16(f2);
        Q_smem[base + 3] = __float2bfloat16(f3);
    }
    __syncthreads();

    // Phase 2: SMEM -> Global (coalesced uint2 writes)
    size_t out_offset = (size_t)(b * H + h) * S * D_HEAD;
    __nv_bfloat16* O_bh = O + out_offset;
    int wid = tid / 32;
    int lid = tid % 32;
    int num_steps = BM / 4;
    for (int step = 0; step < num_steps; ++step) {
        int srow = step * 4 + wid;
        int global_row = q_start + srow;
        if (global_row >= S) continue;
        int col_start = lid * 4;
        uint2 data = *reinterpret_cast<uint2*>(&Q_smem[srow * D_HEAD + col_start]);
        *reinterpret_cast<uint2*>(O_bh + (size_t)global_row * D_HEAD + col_start) = data;
    }

    // ---- TMEM deallocation ----
    __syncthreads();
    if (warp_id == 0) {
        tmem_dealloc_cg1_fn(tmem_base, 256);
    }
}

// ---- Host function ----
void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    const int B = static_cast<int>(Q.size(0));
    const int H = static_cast<int>(Q.size(1));
    const int S = static_cast<int>(Q.size(2));
    const int D = static_cast<int>(Q.size(3));
    (void)D;

    __nv_bfloat16* Q_ptr = static_cast<__nv_bfloat16*>(Q.data_ptr());
    __nv_bfloat16* K_ptr = static_cast<__nv_bfloat16*>(K.data_ptr());
    __nv_bfloat16* V_ptr = static_cast<__nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_ptr = static_cast<float*>(LSE.data_ptr());

    // Create TMA descriptors
    // Q: [S, D] K-major, 128B swizzle, boxDim={64, 128}
    CUtensorMap tma_Q, tma_K, tma_V;
    CUDA_CHECK(create_tma_desc_bf16(&tma_Q, Q_ptr, D_HEAD, S, 64, BM,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CUDA_CHECK(create_tma_desc_bf16(&tma_K, K_ptr, D_HEAD, S, 64, BK,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CUDA_CHECK(create_tma_desc_bf16(&tma_V, V_ptr, D_HEAD, S, 64, BK,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA));

    const int num_q_blocks = (S + BM - 1) / BM;
    dim3 grid(num_q_blocks, B * H);
    dim3 block(NUM_THREADS);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(attention_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));

    attention_kernel<<<grid, block, SMEM_SIZE, stream>>>(
        tma_Q, tma_K, tma_V, O_ptr, LSE_ptr, S, B, H);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_cuda::run);

}  // namespace tvm_ffi_mha_cuda