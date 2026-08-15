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
    const bf16* __restrict__ V_ptr,
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

    // Shared memory with 1024-byte alignment for 128B swizzle
    extern __shared__ char smem_raw[];
    uintptr_t aligned = ((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023;
    char* smem = reinterpret_cast<char*>(aligned);

    bf16* Q_smem  = reinterpret_cast<bf16*>(smem);                        // 32KB
    bf16* KV_smem = Q_smem + 128 * 128;                                  // 32KB
    bf16* P_smem  = KV_smem + 128 * 128;                                 // 32KB
    float* O_smem = reinterpret_cast<float*>(P_smem + 128 * 128);        // 64KB
    uint64_t* bar_tma = reinterpret_cast<uint64_t*>(O_smem + 128 * 128); // 8B
    uint64_t* bar_mma = bar_tma + 1;                                      // 8B
    uint32_t* tmem_addr_smem = reinterpret_cast<uint32_t*>(bar_mma + 1); // 4B

    // TMEM alloc: requires entire warp to execute .sync.aligned
    if (warp_id == 0) {
        tmem_alloc_cg1_fn(tmem_addr_smem, 128);
    }
    __syncthreads();
    uint32_t S_tmem = *tmem_addr_smem;

    if (tid == 0) {
        init_smem_barrier_fn(bar_tma, 1);
        init_smem_barrier_fn(bar_mma, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    const float scale = 0.08838834764f;

    // Load Q (2 TMA tiles for 128B swizzle, each 64x128 bf16 = 16384 bytes)
    int q_offset = b * H * S + h * S + q_start;
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_tma, 32768);
        tma_load_2d_cta_fn(&tma_Q, bar_tma, Q_smem, 0, q_offset);
        tma_load_2d_cta_fn(&tma_Q, bar_tma, Q_smem + 64 * 128, 64, q_offset);
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

    const bf16* V_base = V_ptr + (size_t)(b * H + h) * S * D;

    for (int kv = 0; kv < num_kv_blocks; kv++) {
        int kv_start = kv * BN;
        int kv_offset = b * H * S + h * S + kv_start;

        // ---- Load K (TMA, 128B swizzled) ----
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_tma, 32768);
            tma_load_2d_cta_fn(&tma_K, bar_tma, KV_smem, 0, kv_offset);
            tma_load_2d_cta_fn(&tma_K, bar_tma, KV_smem + 64 * 128, 64, kv_offset);
        }
        mbarrier_wait_fn(bar_tma, phase_tma);
        phase_tma ^= 1;

        // ---- QK^T: tcgen05 MMA (8 K-steps of 16) ----
        if (tid == 0) {
            for (int k_step = 0; k_step < 8; k_step++) {
                int kw = k_step % 4;
                int kh = k_step / 4;
                bf16* qt = (kh == 0) ? Q_smem : Q_smem + 64 * 128;
                bf16* kt = (kh == 0) ? KV_smem : KV_smem + 64 * 128;
                uint64_t desc_a = make_smem_desc_128b_fn(qt + kw * 16, 16, 1024);
                uint64_t desc_b = make_smem_desc_128b_fn(kt + kw * 16, 16, 1024);
                umma_f16_cg1_fn(S_tmem, desc_a, desc_b, idesc_qk, (k_step > 0) ? 1 : 0);
            }
            umma_commit_1sm_fn(bar_mma);
        }
        mbarrier_wait_fn(bar_mma, phase_mma);
        phase_mma ^= 1;

        // ---- Softmax: read S from TMEM (2 chunks of 64, 2 passes) ----
        float m_old = row_max;
        float m_new = m_old;

        // Pass 1: find row max (2 chunks, 1 fence each)
        for (int chunk = 0; chunk < 2; chunk++) {
            uint32_t r[64];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                tmem_load_8x_fn(S_tmem + (chunk * 8 + i) * 8,
                    &r[i*8], &r[i*8+1], &r[i*8+2], &r[i*8+3],
                    &r[i*8+4], &r[i*8+5], &r[i*8+6], &r[i*8+7]);
            }
            tmem_load_fence_fn();
            #pragma unroll
            for (int i = 0; i < 64; i++) {
                int j = chunk * 64 + i;
                float sv = (kv_start + j < S) ? __uint_as_float(r[i]) * scale : -INFINITY;
                m_new = fmaxf(m_new, sv);
            }
        }

        float rescale = fast_expf(m_old - m_new);

        // Pass 2: compute P, write to SMEM (2 chunks, 1 fence each)
        float block_sum = 0.0f;
        for (int chunk = 0; chunk < 2; chunk++) {
            uint32_t r[64];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                tmem_load_8x_fn(S_tmem + (chunk * 8 + i) * 8,
                    &r[i*8], &r[i*8+1], &r[i*8+2], &r[i*8+3],
                    &r[i*8+4], &r[i*8+5], &r[i*8+6], &r[i*8+7]);
            }
            tmem_load_fence_fn();

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                bf16 pv[8];
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    int col = chunk * 64 + i * 8 + j;
                    float sv = (kv_start + col < S) ? __uint_as_float(r[i*8+j]) * scale : -INFINITY;
                    float p = fast_expf(sv - m_new);
                    block_sum += p;
                    pv[j] = __float2bfloat16(p);
                }
                int base = tid * 128 + chunk * 64 + i * 8;
                *reinterpret_cast<int4*>(&P_smem[base]) = *reinterpret_cast<int4*>(pv);
            }
        }

        row_sum = row_sum * rescale + block_sum;
        row_max = m_new;
        __syncthreads();

        // ---- Load V (regular vectorized loads, linear layout) ----
        for (int i = tid * 8; i < BN * D; i += 128 * 8) {
            int row = i / D, col = i % D;
            int g_row = kv_start + row;
            if (g_row < S)
                *reinterpret_cast<int4*>(&KV_smem[i]) =
                    *reinterpret_cast<const int4*>(&V_base[(size_t)g_row * D + col]);
            else
                *reinterpret_cast<int4*>(&KV_smem[i]) = make_int4(0, 0, 0, 0);
        }
        __syncthreads();

        // ---- PV: wmma (4 warps, each handles 32 rows) ----
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
            bf16 ov[8];
            #pragma unroll
            for (int i = 0; i < 8; i++)
                ov[i] = __float2bfloat16(O_smem[tid * 128 + d + i] * inv_sum);
            *reinterpret_cast<int4*>(
                &O[(size_t)(b * H + h) * S * D + q_row * D + d]) =
                *reinterpret_cast<int4*>(ov);
        }
        LSE[(size_t)(b * H + h) * S + q_row] =
            (row_sum > 0.0f) ? (row_max + logf(row_sum)) : -INFINITY;
    }

    __syncthreads();
    if (warp_id == 0) {
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

    CUtensorMap tma_Q, tma_K;

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

    int total_q = (S + BM - 1) / BM;
    int blocks = B * H * total_q;
    int threads = 128;
    // 3*32KB + 64KB + 32B barriers + 1023B alignment padding
    size_t smem_size = 3 * 32 * 1024 + 64 * 1024 + 1024;

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(attn_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size));

    attn_kernel<<<blocks, threads, smem_size, stream>>>(
        tma_Q, tma_K, V_ptr, O_ptr, LSE_ptr, B, H, S);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, attn_impl::run);

}  // namespace attn_impl