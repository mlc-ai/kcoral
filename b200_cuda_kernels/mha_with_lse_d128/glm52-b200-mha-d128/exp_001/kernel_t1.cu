#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>
#include <cuda.h>

using bf16 = __nv_bfloat16;

constexpr int D = 128;
constexpr int BM = 128;
constexpr int BN = 128;

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

// ---- Device helpers ----

__device__ __forceinline__ float fast_expf(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x * 1.44269504088896340736f));
    return y;
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

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n.reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase));
}

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
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

__device__ __forceinline__ void tmem_load_8x_fn(uint32_t col,
    uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3,
    uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3),
     "=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7) : "r"(col));
}

__device__ __forceinline__ void tmem_store_8x_fn(uint32_t col,
    uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3,
    uint32_t r4, uint32_t r5, uint32_t r6, uint32_t r7) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7};"
   :: "r"(col), "r"(r0),"r"(r1),"r"(r2),"r"(r3),
     "r"(r4),"r"(r5),"r"(r6),"r"(r7));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_store_fence_fn() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
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

__device__ __forceinline__ void umma_commit_1sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64 [%0];"
        :: "r"(a));
}

__device__ __forceinline__ uint64_t make_smem_desc_128b_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;   // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
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

// ---- Kernel ----

__global__ void attn_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    bf16* __restrict__ O, float* __restrict__ LSE,
    int B, int H, int S)
{
    int total_q = (S + BM - 1) / BM;
    int q_block = blockIdx.x % total_q;
    int bh = blockIdx.x / total_q;
    int h = bh % H;
    int b = bh / H;
    int q_start = q_block * BM;
    int tid = threadIdx.x;

    extern __shared__ char smem_raw[];
    // 1024-byte aligned buffers for 128B swizzling
    bf16* Q_smem  = reinterpret_cast<bf16*>(smem_raw);                    // 32KB
    bf16* KV_smem = reinterpret_cast<bf16*>(smem_raw + 32768);            // 32KB
    bf16* P_smem  = reinterpret_cast<bf16*>(smem_raw + 65536);            // 32KB
    uint64_t* bar_tma = reinterpret_cast<uint64_t*>(smem_raw + 98304);    // 8B
    uint64_t* bar_mma = bar_tma + 1;                                      // 8B
    uint32_t* tmem_addr_smem = reinterpret_cast<uint32_t*>(bar_mma + 1);  // 4B

    // TMEM allocation: 256 columns (128 for S, 128 for O)
    if (tid == 0) {
        tmem_alloc_cg1_fn(tmem_addr_smem, 256);
    }
    __syncthreads();
    uint32_t tmem_base = *tmem_addr_smem;
    uint32_t S_tmem = tmem_base;
    uint32_t O_tmem = tmem_base + 128;

    // Init mbarriers
    if (tid == 0) {
        init_smem_barrier_fn(bar_tma, 1);
        init_smem_barrier_fn(bar_mma, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    const float scale = 0.08838834764f; // 1/sqrt(128)

    // Load Q to SMEM (2 TMA loads for 128B swizzled layout)
    int q_row_offset = b * H * S + h * S + q_start;
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_tma, 2 * 128 * 64 * 2);
        tma_load_2d_cta_fn(&tma_Q, bar_tma, Q_smem, 0, q_row_offset);
        tma_load_2d_cta_fn(&tma_Q, bar_tma, Q_smem, 64, q_row_offset);
    }
    mbarrier_wait_fn(bar_tma, 0);

    uint32_t idesc_qk = make_instr_desc_fn(BM, BN);
    uint32_t idesc_pv = make_instr_desc_fn(BM, D);

    float row_max = -INFINITY;
    float row_sum = 0.0f;
    int num_kv_blocks = (S + BN - 1) / BN;
    uint32_t phase_tma = 1;
    uint32_t phase_mma = 0;

    for (int kv = 0; kv < num_kv_blocks; kv++) {
        int kv_start = kv * BN;
        int kv_row_offset = b * H * S + h * S + kv_start;

        // ---- Load K ----
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_tma, 2 * 128 * 64 * 2);
            tma_load_2d_cta_fn(&tma_K, bar_tma, KV_smem, 0, kv_row_offset);
            tma_load_2d_cta_fn(&tma_K, bar_tma, KV_smem, 64, kv_row_offset);
        }
        mbarrier_wait_fn(bar_tma, phase_tma);
        phase_tma ^= 1;

        // ---- QK^T MMA: S = Q @ K^T ----
        if (tid == 0) {
            for (int k_step = 0; k_step < 8; k_step++) {
                uint32_t k_off = (k_step % 4) * 16 + (k_step / 4) * 64; // bf16 elements
                uint64_t desc_a = make_smem_desc_128b_fn(Q_smem + k_off, 1, 1024);
                uint64_t desc_b = make_smem_desc_128b_fn(KV_smem + k_off, 1, 1024);
                umma_f16_cg1_fn(S_tmem, desc_a, desc_b, idesc_qk, (k_step > 0) ? 1 : 0);
            }
            umma_commit_1sm_fn(bar_mma);
        }
        mbarrier_wait_fn(bar_mma, phase_mma);
        phase_mma ^= 1;

        // ---- Softmax: read S from TMEM, online softmax, write P to SMEM ----
        float m_old = row_max;
        float m_new = m_old;

        // Pass 1: find row max (process 16 values at a time)
        for (int batch = 0; batch < 8; batch++) {
            uint32_t r[16];
            for (int i = 0; i < 2; i++) {
                uint32_t col = S_tmem + (batch * 2 + i) * 8;
                tmem_load_8x_fn(col, &r[i*8], &r[i*8+1], &r[i*8+2], &r[i*8+3],
                                &r[i*8+4], &r[i*8+5], &r[i*8+6], &r[i*8+7]);
            }
            tmem_load_fence_fn();
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int j = batch * 16 + i;
                int kv_idx = kv_start + j;
                float s_val = (kv_idx < S) ? (__uint_as_float(r[i]) * scale) : -INFINITY;
                m_new = fmaxf(m_new, s_val);
            }
        }

        float rescale = fast_expf(m_old - m_new);

        // Rescale O in TMEM (skip for first block)
        if (kv > 0) {
            for (int batch = 0; batch < 4; batch++) {
                uint32_t r[32];
                for (int i = 0; i < 4; i++) {
                    uint32_t col = O_tmem + (batch * 4 + i) * 8;
                    tmem_load_8x_fn(col, &r[i*8], &r[i*8+1], &r[i*8+2], &r[i*8+3],
                                    &r[i*8+4], &r[i*8+5], &r[i*8+6], &r[i*8+7]);
                }
                tmem_load_fence_fn();
                #pragma unroll
                for (int i = 0; i < 32; i++) {
                    r[i] = __float_as_uint(__uint_as_float(r[i]) * rescale);
                }
                for (int i = 0; i < 4; i++) {
                    uint32_t col = O_tmem + (batch * 4 + i) * 8;
                    tmem_store_8x_fn(col, r[i*8], r[i*8+1], r[i*8+2], r[i*8+3],
                                    r[i*8+4], r[i*8+5], r[i*8+6], r[i*8+7]);
                }
                tmem_store_fence_fn();
            }
        }

        // Pass 2: compute P = exp(S*scale - m_new), write to swizzled SMEM
        float block_sum = 0.0f;
        for (int batch = 0; batch < 8; batch++) {
            uint32_t r[16];
            for (int i = 0; i < 2; i++) {
                uint32_t col = S_tmem + (batch * 2 + i) * 8;
                tmem_load_8x_fn(col, &r[i*8], &r[i*8+1], &r[i*8+2], &r[i*8+3],
                                &r[i*8+4], &r[i*8+5], &r[i*8+6], &r[i*8+7]);
            }
            tmem_load_fence_fn();

            for (int i = 0; i < 2; i++) {
                int chunk = batch * 2 + i;
                int span = chunk / 8;
                int within_span = (chunk % 8) * 16; // bytes
                uint32_t addr = (uint32_t)__cvta_generic_to_shared(P_smem) +
                                (tid / 8) * 1024 + ((tid % 8) ^ span) * 128 + within_span;
                bf16 p_vals[8];
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    int col_idx = chunk * 8 + j;
                    int kv_idx = kv_start + col_idx;
                    float s_val = (kv_idx < S) ? (__uint_as_float(r[i*8+j]) * scale) : -INFINITY;
                    float p_val = fast_expf(s_val - m_new);
                    block_sum += p_val;
                    p_vals[j] = __float2bfloat16(p_val);
                }
                *reinterpret_cast<int4*>(addr) = *reinterpret_cast<int4*>(p_vals);
            }
        }

        row_sum = row_sum * rescale + block_sum;
        row_max = m_new;

        __syncthreads();
        fence_async_shared_fn(); // Make P (generic proxy) visible to MMA (async proxy)

        // ---- Load V (reuse KV_smem) ----
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_tma, 2 * 128 * 64 * 2);
            tma_load_2d_cta_fn(&tma_V, bar_tma, KV_smem, 0, kv_row_offset);
            tma_load_2d_cta_fn(&tma_V, bar_tma, KV_smem, 64, kv_row_offset);
        }
        mbarrier_wait_fn(bar_tma, phase_tma);
        phase_tma ^= 1;

        // ---- PV MMA: O += P @ V ----
        if (tid == 0) {
            for (int k_step = 0; k_step < 8; k_step++) {
                uint32_t k_off = (k_step % 4) * 16 + (k_step / 4) * 64;
                uint64_t desc_a = make_smem_desc_128b_fn(P_smem + k_off, 1, 1024);
                uint64_t desc_b = make_smem_desc_128b_fn(KV_smem + k_off, 1, 1024);
                umma_f16_cg1_fn(O_tmem, desc_a, desc_b, idesc_pv,
                                (k_step > 0 || kv > 0) ? 1 : 0);
            }
            umma_commit_1sm_fn(bar_mma);
        }
        mbarrier_wait_fn(bar_mma, phase_mma);
        phase_mma ^= 1;
    }

    // ---- Epilogue: read O from TMEM, normalize, write to global ----
    int q_row = q_start + tid;
    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
    size_t out_base = (size_t)(b * H + h) * S * D + (q_row < S ? q_row * D : 0);

    for (int batch = 0; batch < 4; batch++) {
        uint32_t r[32];
        for (int i = 0; i < 4; i++) {
            uint32_t col = O_tmem + (batch * 4 + i) * 8;
            tmem_load_8x_fn(col, &r[i*8], &r[i*8+1], &r[i*8+2], &r[i*8+3],
                            &r[i*8+4], &r[i*8+5], &r[i*8+6], &r[i*8+7]);
        }
        tmem_load_fence_fn();

        if (q_row < S) {
            bf16* out_ptr = O + out_base + batch * 32;
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                out_ptr[i] = __float2bfloat16(__uint_as_float(r[i]) * inv_sum);
            }
        }
    }

    if (q_row < S) {
        LSE[(size_t)(b * H + h) * S + q_row] =
            (row_sum > 0.0f) ? (row_max + logf(row_sum)) : -INFINITY;
    }

    __syncthreads();
    if (tid == 0) {
        tmem_dealloc_cg1_fn(tmem_base, 256);
    }
}

// ---- Host function ----

namespace attn_impl {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int S = (int)Q.size(2);

    bf16* Q_ptr = static_cast<bf16*>(Q.data_ptr());
    bf16* K_ptr = static_cast<bf16*>(K.data_ptr());
    bf16* V_ptr = static_cast<bf16*>(V.data_ptr());
    bf16* O_ptr = static_cast<bf16*>(O.data_ptr());
    float* LSE_ptr = static_cast<float*>(LSE.data_ptr());

    CUtensorMap tma_Q, tma_K, tma_V;

    cuuint64_t globalDim[2] = {(cuuint64_t)D, (cuuint64_t)(B * H * S)};
    cuuint64_t globalStrides[1] = {(cuuint64_t)D * 2};
    cuuint32_t boxDim[2] = {64, 128};
    cuuint32_t elementStrides[2] = {1, 1};

    CU_CHECK(cuTensorMapEncodeTiled(
        &tma_Q, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        Q_ptr, globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    CU_CHECK(cuTensorMapEncodeTiled(
        &tma_K, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        K_ptr, globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    CU_CHECK(cuTensorMapEncodeTiled(
        &tma_V, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        V_ptr, globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    int total_q = (S + BM - 1) / BM;
    int blocks = B * H * total_q;
    int threads = 128;
    size_t smem_size = 3 * 32 * 1024 + 32;

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(attn_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size));

    attn_kernel<<<blocks, threads, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, O_ptr, LSE_ptr, B, H, S);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, attn_impl::run);

}  // namespace attn_impl