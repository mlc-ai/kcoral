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
    if (_e != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); exit(1); } \
} while(0)

#define CU_CHECK(call) do { \
    CUresult _r = (call); \
    if (_r != CUDA_SUCCESS) { const char* e; cuGetErrorString(_r, &e); fprintf(stderr, "CU error %s at %s:%d\n", e?e:"?", __FILE__, __LINE__); exit(1); } \
} while(0)

// ---- Device helpers ----

__device__ __forceinline__ float fast_expf(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x * 1.44269504088896340736f));
    return y;
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)), "l"((uint64_t)d),
           "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void init_bar_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void fence_bar_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void arrive_expect_tx_fn(uint64_t* bar, uint32_t tx) {
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx) : "memory");
}

__device__ __forceinline__ void wait_bar_fn(uint64_t* bar, uint32_t phase) {
    asm volatile("{\n.reg .pred P;\nWAIT_%=:\nmbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n@!P bra WAIT_%=;\n}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase));
}

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tmem_load_8x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3, uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
        : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3),"=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7) : "r"(col));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_store_8x_fn(uint32_t col, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3, uint32_t r4, uint32_t r5, uint32_t r6, uint32_t r7) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
        :: "r"(col), "r"(r0),"r"(r1),"r"(r2),"r"(r3),"r"(r4),"r"(r5),"r"(r6),"r"(r7));
}

__device__ __forceinline__ void tmem_store_fence_fn() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void umma_f16_cg1_fn(uint32_t tmem_d, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_d), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(a));
}

__device__ __forceinline__ uint64_t make_smem_desc_128b_fn(void* ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint32_t make_idesc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4) | (1u << 7) | (1u << 10);  // FP32 out, BF16 A, BF16 B
    d |= ((N / 8) << 17) | ((M / 16) << 24);
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
    int h = bh % H, b = bh / H;
    int q_start = q_block * BM;
    int tid = threadIdx.x;
    int warp_id = tid / 32;

    extern __shared__ char smem_raw[];
    uintptr_t aligned = ((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023;
    char* smem = reinterpret_cast<char*>(aligned);

    bf16* Q_smem  = reinterpret_cast<bf16*>(smem);                       // 32KB
    bf16* K_smem  = Q_smem + 128 * 128;                                  // 32KB (buf 0)
    bf16* K_smem2 = K_smem + 128 * 128;                                  // 32KB (buf 1)
    bf16* P_smem  = K_smem2 + 128 * 128;                                 // 32KB
    bf16* V_smem  = P_smem + 128 * 128;                                  // 32KB
    uint64_t* bar_K = reinterpret_cast<uint64_t*>(V_smem + 128 * 128);   // 16B
    uint64_t* bar_V = bar_K + 2;                                         // 8B
    uint64_t* bar_mma = bar_V + 1;                                       // 8B
    uint32_t* tmem_addr = reinterpret_cast<uint32_t*>(bar_mma + 1);      // 4B

    // TMEM alloc: 256 cols (128 for S, 128 for O)
    if (warp_id == 0) tmem_alloc_fn(tmem_addr, 256);
    __syncthreads();
    uint32_t S_tmem = *tmem_addr;
    uint32_t O_tmem = S_tmem + 128;

    if (tid == 0) {
        init_bar_fn(&bar_K[0], 1);
        init_bar_fn(&bar_K[1], 1);
        init_bar_fn(bar_V, 1);
        init_bar_fn(bar_mma, 1);
        fence_bar_init_fn();
    }
    __syncthreads();

    const float scale = 0.08838834764f;
    int q_off = b * H * S + h * S + q_start;

    // Load Q using bar_V (phase 0)
    if (tid == 0) {
        arrive_expect_tx_fn(bar_V, 32768);
        tma_load_2d_fn(&tma_Q, bar_V, Q_smem, 0, q_off);
        tma_load_2d_fn(&tma_Q, bar_V, Q_smem + 64 * 128, 64, q_off);
    }
    wait_bar_fn(bar_V, 0);
    uint32_t phase_V = 1;

    // Load K[0] using bar_K[0] (phase 0)
    if (tid == 0) {
        arrive_expect_tx_fn(&bar_K[0], 32768);
        tma_load_2d_fn(&tma_K, &bar_K[0], K_smem, 0, q_off);
        tma_load_2d_fn(&tma_K, &bar_K[0], K_smem + 64 * 128, 64, q_off);
    }
    uint32_t phase_K[2] = {1, 0};

    float row_max = -INFINITY;
    float row_sum = 0.0f;
    int num_kv = (S + BN - 1) / BN;
    uint32_t phase_mma = 0;
    uint32_t idesc = make_idesc_fn(BM, BN);  // Same for QK^T and PV (128x128)

    for (int kv = 0; kv < num_kv; kv++) {
        int kv_start = kv * BN;
        int kv_off = b * H * S + h * S + kv_start;
        int kv_buf = kv % 2;

        // Start V[kv] load (overlap with QK^T and softmax)
        if (tid == 0) {
            arrive_expect_tx_fn(bar_V, 32768);
            tma_load_2d_fn(&tma_V, bar_V, V_smem, 0, kv_off);
            tma_load_2d_fn(&tma_V, bar_V, V_smem + 64 * 128, 64, kv_off);
        }

        // Start K[kv+1] load (double-buffer, overlap)
        if (kv < num_kv - 1 && tid == 0) {
            int next_off = b * H * S + h * S + (kv + 1) * BN;
            int next_buf = (kv + 1) % 2;
            bf16* next_K = (next_buf == 0) ? K_smem : K_smem2;
            arrive_expect_tx_fn(&bar_K[next_buf], 32768);
            tma_load_2d_fn(&tma_K, &bar_K[next_buf], next_K, 0, next_off);
            tma_load_2d_fn(&tma_K, &bar_K[next_buf], next_K + 64 * 128, 64, next_off);
        }

        // Wait for K[kv]
        wait_bar_fn(&bar_K[kv_buf], phase_K[kv_buf] ^ 1);
        phase_K[kv_buf] ^= 1;

        bf16* K_buf = (kv_buf == 0) ? K_smem : K_smem2;

        // QK^T: tcgen05 MMA (8 K-steps of 16)
        if (tid == 0) {
            for (int k = 0; k < 8; k++) {
                int kh = k / 4, kw = k % 4;
                bf16* qt = (kh == 0) ? Q_smem : Q_smem + 64 * 128;
                bf16* kt = (kh == 0) ? K_buf : K_buf + 64 * 128;
                uint64_t da = make_smem_desc_128b_fn(qt + kw * 16, 1, 1024);
                uint64_t db = make_smem_desc_128b_fn(kt + kw * 16, 1, 1024);
                umma_f16_cg1_fn(S_tmem, da, db, idesc, (k > 0) ? 1 : 0);
            }
            umma_commit_fn(bar_mma);
        }
        wait_bar_fn(bar_mma, phase_mma);
        phase_mma ^= 1;

        // ---- Softmax: read S from TMEM, compute P, write P to swizzled SMEM ----
        float m_old = row_max;
        float m_new = m_old;

        // Pass 1: find row max (2 chunks of 64, 1 fence each)
        for (int chunk = 0; chunk < 2; chunk++) {
            uint32_t r[64];
            #pragma unroll
            for (int i = 0; i < 8; i++)
                tmem_load_8x_fn(S_tmem + (chunk * 8 + i) * 8,
                    &r[i*8], &r[i*8+1], &r[i*8+2], &r[i*8+3],
                    &r[i*8+4], &r[i*8+5], &r[i*8+6], &r[i*8+7]);
            tmem_load_fence_fn();
            #pragma unroll
            for (int i = 0; i < 64; i++) {
                int j = chunk * 64 + i;
                float sv = (kv_start + j < S) ? __uint_as_float(r[i]) * scale : -INFINITY;
                m_new = fmaxf(m_new, sv);
            }
        }

        float rescale = fast_expf(m_old - m_new);

        // O rescale in TMEM (skip for first block)
        if (kv > 0) {
            for (int batch = 0; batch < 4; batch++) {
                uint32_t r[32];
                #pragma unroll
                for (int i = 0; i < 4; i++)
                    tmem_load_8x_fn(O_tmem + (batch * 4 + i) * 8,
                        &r[i*8], &r[i*8+1], &r[i*8+2], &r[i*8+3],
                        &r[i*8+4], &r[i*8+5], &r[i*8+6], &r[i*8+7]);
                tmem_load_fence_fn();
                #pragma unroll
                for (int i = 0; i < 32; i++)
                    r[i] = __float_as_uint(__uint_as_float(r[i]) * rescale);
                #pragma unroll
                for (int i = 0; i < 4; i++)
                    tmem_store_8x_fn(O_tmem + (batch * 4 + i) * 8,
                        r[i*8], r[i*8+1], r[i*8+2], r[i*8+3],
                        r[i*8+4], r[i*8+5], r[i*8+6], r[i*8+7]);
            }
            tmem_store_fence_fn();
        }

        // Pass 2: compute P = exp(S*scale - m_new), write to 128B swizzled SMEM
        float block_sum = 0.0f;
        int row = tid;
        int core_m = row / 8, within_m = row % 8;

        for (int chunk = 0; chunk < 2; chunk++) {
            uint32_t r[64];
            #pragma unroll
            for (int i = 0; i < 8; i++)
                tmem_load_8x_fn(S_tmem + (chunk * 8 + i) * 8,
                    &r[i*8], &r[i*8+1], &r[i*8+2], &r[i*8+3],
                    &r[i*8+4], &r[i*8+5], &r[i*8+6], &r[i*8+7]);
            tmem_load_fence_fn();

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int col_base = chunk * 64 + i * 8;
                int k_half = col_base / 64;
                int within_k = col_base % 64;
                int chunk_idx = within_k / 8;
                int swizzled_chunk = within_m ^ chunk_idx;
                int byte_off = core_m * 1024 + k_half * 16384 + within_m * 128 + swizzled_chunk * 16;
                int bf16_off = byte_off / 2;

                bf16 pv[8];
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    int col = col_base + j;
                    float sv = (kv_start + col < S) ? __uint_as_float(r[i*8+j]) * scale : -INFINITY;
                    float p = fast_expf(sv - m_new);
                    block_sum += p;
                    pv[j] = __float2bfloat16(p);
                }
                *reinterpret_cast<int4*>(&P_smem[bf16_off]) = *reinterpret_cast<int4*>(pv);
            }
        }

        row_sum = row_sum * rescale + block_sum;
        row_max = m_new;

        // Wait for V load
        wait_bar_fn(bar_V, phase_V);
        phase_V ^= 1;

        // Zero OOB V rows (only needed for last block)
        if (kv_start + BN > S) {
            int g_row = kv_start + tid;
            if (g_row >= S) {
                int r = tid, cm = r / 8, wm = r % 8;
                for (int kh = 0; kh < 2; kh++) {
                    for (int c = 0; c < 8; c++) {
                        int sc = wm ^ c;
                        int off = cm * 1024 + kh * 16384 + wm * 128 + sc * 16;
                        *reinterpret_cast<int4*>((char*)V_smem + off) = make_int4(0, 0, 0, 0);
                    }
                }
            }
        }

        __syncthreads();
        fence_async_shared_fn();  // P: generic -> async proxy

        // PV: tcgen05 MMA (O += P @ V, 8 K-steps of 16)
        if (tid == 0) {
            for (int k = 0; k < 8; k++) {
                int kh = k / 4, kw = k % 4;
                int k_off = kh * 8192 + kw * 16;  // BF16 offset
                uint64_t da = make_smem_desc_128b_fn(P_smem + k_off, 1, 1024);
                uint64_t db = make_smem_desc_128b_fn(V_smem + k_off, 1, 1024);
                umma_f16_cg1_fn(O_tmem, da, db, idesc, (k > 0 || kv > 0) ? 1 : 0);
            }
            umma_commit_fn(bar_mma);
        }
        wait_bar_fn(bar_mma, phase_mma);
        phase_mma ^= 1;
    }

    // ---- Epilogue: read O from TMEM, normalize, write to global ----
    int q_row = q_start + tid;
    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;

    for (int batch = 0; batch < 4; batch++) {
        uint32_t r[32];
        #pragma unroll
        for (int i = 0; i < 4; i++)
            tmem_load_8x_fn(O_tmem + (batch * 4 + i) * 8,
                &r[i*8], &r[i*8+1], &r[i*8+2], &r[i*8+3],
                &r[i*8+4], &r[i*8+5], &r[i*8+6], &r[i*8+7]);
        tmem_load_fence_fn();

        if (q_row < S) {
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int d = batch * 32 + i;
                O[(size_t)(b * H + h) * S * D + q_row * D + d] =
                    __float2bfloat16(__uint_as_float(r[i]) * inv_sum);
            }
        }
    }

    if (q_row < S) {
        LSE[(size_t)(b * H + h) * S + q_row] =
            (row_sum > 0.0f) ? (row_max + logf(row_sum)) : -INFINITY;
    }

    __syncthreads();
    if (warp_id == 0) tmem_dealloc_fn(S_tmem, 256);
}

// ---- Host function ----

namespace attn_impl {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = (int)Q.size(0), H = (int)Q.size(1), S = (int)Q.size(2);
    bf16* Q_ptr = static_cast<bf16*>(Q.data_ptr());
    bf16* K_ptr = static_cast<bf16*>(K.data_ptr());
    bf16* V_ptr = static_cast<bf16*>(V.data_ptr());
    bf16* O_ptr = static_cast<bf16*>(O.data_ptr());
    float* LSE_ptr = static_cast<float*>(LSE.data_ptr());

    CUtensorMap tma_Q, tma_K, tma_V;
    cuuint64_t gdim[2] = {(cuuint64_t)D, (cuuint64_t)(B * H * S)};
    cuuint64_t gstr[1] = {(cuuint64_t)D * 2};
    cuuint32_t box[2] = {64, 128};
    cuuint32_t estr[2] = {1, 1};

    CU_CHECK(cuTensorMapEncodeTiled(&tma_Q, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        Q_ptr, gdim, gstr, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(cuTensorMapEncodeTiled(&tma_K, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        K_ptr, gdim, gstr, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(cuTensorMapEncodeTiled(&tma_V, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        V_ptr, gdim, gstr, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    int total_q = (S + BM - 1) / BM;
    int blocks = B * H * total_q;
    size_t smem_size = 5 * 32 * 1024 + 64 + 1024;  // 5 buffers + barriers + alignment

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(attn_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size));

    attn_kernel<<<blocks, 128, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, O_ptr, LSE_ptr, B, H, S);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, attn_impl::run);

}  // namespace attn_impl