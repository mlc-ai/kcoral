#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>
#include <cuda.h>

using namespace nvcuda;
using bf16 = __nv_bfloat16;

constexpr int D = 128;
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int WT = 16;

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

__device__ __forceinline__ void tma_load_2d_fn(
    const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
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

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
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
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
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
    d |= (0u << 15);   // A K-major (no transpose)
    d |= (0u << 16);   // B K-major (no transpose)
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
    int warp_id = tid / 32;

    extern __shared__ char smem_raw[];
    // Q_smem:  32KB (128B swizzled, 2 tiles of 64x128)
    // KV_smem: 32KB (swizzled for K, plain for V)
    // P_smem:  32KB (row-major bf16)
    // O_smem:  64KB (row-major fp32)
    // Total: 160KB + ~32B barriers
    bf16* Q_smem  = reinterpret_cast<bf16*>(smem_raw);                        // 32KB
    bf16* KV_smem = Q_smem + 128 * 128;                                      // 32KB
    bf16* P_smem  = KV_smem + 128 * 128;                                     // 32KB
    float* O_smem = reinterpret_cast<float*>(P_smem + 128 * 128);            // 64KB
    uint64_t* bar_tma = reinterpret_cast<uint64_t*>(O_smem + 128 * 128);     // 8B
    uint64_t* bar_mma = bar_tma + 1;                                          // 8B
    uint32_t* tmem_addr_smem = reinterpret_cast<uint32_t*>(bar_mma + 1);     // 4B

    // TMEM allocation (128 columns for S)
    if (tid == 0) {
        tmem_alloc_cg1_fn(tmem_addr_smem, 128);
    }
    __syncthreads();
    uint32_t S_tmem = *tmem_addr_smem;

    // Init barriers
    if (tid == 0) {
        init_smem_barrier_fn(bar_tma, 1);
        init_smem_barrier_fn(bar_mma, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    const float scale = 0.08838834764f; // 1/sqrt(128)

    // Load Q (2 TMA tiles, 128B swizzled)
    int q_row_offset = b * H * S + h * S + q_start;
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_tma, 32768);
        tma_load_2d_fn(&tma_Q, bar_tma, Q_smem, 0, q_row_offset);
        tma_load_2d_fn(&tma_Q, bar_tma, Q_smem + 64 * 128, 64, q_row_offset);
    }
    mbarrier_wait_fn(bar_tma, 0);

    // Init O_smem = 0
    for (int i = tid; i < 128 * 128; i += 128)
        O_smem[i] = 0.0f;

    float row_max = -INFINITY;
    float row_sum = 0.0f;
    int num_kv_blocks = (S + BN - 1) / BN;
    uint32_t phase_tma = 1;
    uint32_t phase_mma = 0;
    uint32_t idesc_qk = make_instr_desc_fn(BM, BN);

    for (int kv = 0; kv < num_kv_blocks; kv++) {
        int kv_start = kv * BN;
        int kv_row_offset = b * H * S + h * S + kv_start;

        // ---- Load K (128B swizzled, 2 tiles) ----
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_tma, 32768);
            tma_load_2d_fn(&tma_K, bar_tma, KV_smem, 0, kv_row_offset);
            tma_load_2d_fn(&tma_K, bar_tma, KV_smem + 64 * 128, 64, kv_row_offset);
        }
        mbarrier_wait_fn(bar_tma, phase_tma);
        phase_tma ^= 1;

        // ---- QK^T: tcgen05 MMA (thread 0 issues all 8 K-steps) ----
        if (tid == 0) {
            for (int k_step = 0; k_step < 8; k_step++) {
                int k_within = k_step % 4;
                int k_half = k_step / 4;
                bf16* q_tile = (k_half == 0) ? Q_smem : Q_smem + 64 * 128;
                bf16* k_tile = (k_half == 0) ? KV_smem : KV_smem + 64 * 128;
                uint64_t desc_a = make_smem_desc_128b_fn(q_tile + k_within * 16, 1, 1024);
                uint64_t desc_b = make_smem_desc_128b_fn(k_tile + k_within * 16, 1, 1024);
                umma_f16_cg1_fn(S_tmem, desc_a, desc_b, idesc_qk, (k_step > 0) ? 1 : 0);
            }
            umma_commit_1sm_fn(bar_mma);
        }
        mbarrier_wait_fn(bar_mma, phase_mma);
        phase_mma ^= 1;

        // ---- Softmax: read S from TMEM, compute P ----
        float m_old = row_max;
        float m_new = m_old;

        // Pass 1: find row max (16 loads of 8 cols each = 128 cols)
        for (int batch = 0; batch < 16; batch++) {
            uint32_t r[8];
            uint32_t col = S_tmem + batch * 8;
            tmem_load_8x_fn(col, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
            tmem_load_fence_fn();
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int j = batch * 8 + i;
                int kv_idx = kv_start + j;
                float s_val = (kv_idx < S) ? (__uint_as_float(r[i]) * scale) : -INFINITY;
                m_new = fmaxf(m_new, s_val);
            }
        }

        float rescale = fast_expf(m_old - m_new);

        // Pass 2: compute P = exp(S*scale - m_new), write to P_smem (row-major)
        float block_sum = 0.0f;
        for (int batch = 0; batch < 16; batch++) {
            uint32_t r[8];
            uint32_t col = S_tmem + batch * 8;
            tmem_load_8x_fn(col, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
            tmem_load_fence_fn();

            bf16 p_vals[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int j = batch * 8 + i;
                int kv_idx = kv_start + j;
                float s_val = (kv_idx < S) ? (__uint_as_float(r[i]) * scale) : -INFINITY;
                float p_val = fast_expf(s_val - m_new);
                block_sum += p_val;
                p_vals[i] = __float2bfloat16(p_val);
            }
            int base = tid * 128 + batch * 8;
            *reinterpret_cast<int4*>(&P_smem[base]) = *reinterpret_cast<int4*>(p_vals);
        }

        row_sum = row_sum * rescale + block_sum;
        row_max = m_new;

        __syncthreads();

        // ---- Load V (no swizzle, single 128x128 tile) ----
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_tma, 32768);
            tma_load_2d_fn(&tma_V, bar_tma, KV_smem, 0, kv_row_offset);
        }
        mbarrier_wait_fn(bar_tma, phase_tma);
        phase_tma ^= 1;

        // ---- PV: wmma (4 warps, each handles 32 rows) ----
        // Rescale O on accumulator fragment, then accumulate P@V
        for (int mi = 0; mi < 2; mi++) {
            for (int ni = 0; ni < 8; ni++) {
                wmma::fragment<wmma::accumulator, WT, WT, WT, float> o_frag;
                if (kv > 0) {
                    wmma::load_matrix_sync(o_frag,
                        O_smem + (warp_id * 32 + mi * WT) * 128 + ni * WT,
                        128, wmma::mem_row_major);
                    #pragma unroll
                    for (int i = 0; i < o_frag.num_elements; i++)
                        o_frag.x[i] *= rescale;
                } else {
                    wmma::fill_fragment(o_frag, 0.0f);
                }

                for (int ki = 0; ki < 8; ki++) {
                    wmma::fragment<wmma::matrix_a, WT, WT, WT, bf16, wmma::row_major> p_frag;
                    wmma::load_matrix_sync(p_frag,
                        P_smem + (warp_id * 32 + mi * WT) * 128 + ki * WT, 128);

                    wmma::fragment<wmma::matrix_b, WT, WT, WT, bf16, wmma::row_major> v_frag;
                    wmma::load_matrix_sync(v_frag,
                        KV_smem + ki * WT * 128 + ni * WT, 128);

                    wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
                }

                wmma::store_matrix_sync(
                    O_smem + (warp_id * 32 + mi * WT) * 128 + ni * WT,
                    o_frag, 128, wmma::mem_row_major);
            }
        }
        __syncthreads();
    }

    // ---- Epilogue: normalize O and write to global ----
    int q_row = q_start + tid;
    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
    if (q_row < S) {
        for (int d = 0; d < 128; d += 8) {
            bf16 out_vals[8];
            #pragma unroll
            for (int i = 0; i < 8; i++)
                out_vals[i] = __float2bfloat16(O_smem[tid * 128 + d + i] * inv_sum);
            *reinterpret_cast<int4*>(
                &O[(size_t)(b * H + h) * S * D + q_row * D + d]) =
                *reinterpret_cast<int4*>(out_vals);
        }
        LSE[(size_t)(b * H + h) * S + q_row] =
            (row_sum > 0.0f) ? (row_max + logf(row_sum)) : -INFINITY;
    }

    __syncthreads();
    if (tid == 0) {
        tmem_dealloc_cg1_fn(S_tmem, 128);
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

    // Q and K: 128B swizzle, box {64, 128}
    cuuint64_t globalDim_qk[2] = {(cuuint64_t)D, (cuuint64_t)(B * H * S)};
    cuuint64_t globalStrides_qk[1] = {(cuuint64_t)D * 2};
    cuuint32_t boxDim_qk[2] = {64, 128};
    cuuint32_t elementStrides[2] = {1, 1};

    CU_CHECK(cuTensorMapEncodeTiled(
        &tma_Q, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        Q_ptr, globalDim_qk, globalStrides_qk, boxDim_qk, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    CU_CHECK(cuTensorMapEncodeTiled(
        &tma_K, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        K_ptr, globalDim_qk, globalStrides_qk, boxDim_qk, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    // V: no swizzle, box {128, 128}
    cuuint32_t boxDim_v[2] = {128, 128};

    CU_CHECK(cuTensorMapEncodeTiled(
        &tma_V, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        V_ptr, globalDim_qk, globalStrides_qk, boxDim_v, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    int total_q = (S + BM - 1) / BM;
    int blocks = B * H * total_q;
    int threads = 128;
    size_t smem_size = 3 * 32 * 1024 + 64 * 1024 + 32;

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