#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
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

CUresult create_tma_4d_descriptor_2B(CUtensorMap* d, void* globalAddress, 
    uint64_t D, uint64_t S, uint64_t H, uint64_t B,
    uint32_t box_D, uint32_t box_S) {
    cuuint64_t globalDim[4] = {D, S, H, B};
    cuuint64_t globalStrides[3] = {D * 2, S * D * 2, H * S * D * 2};
    cuuint32_t boxDim[4] = {box_D, box_S, 1, 1};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        4, 
        globalAddress,
        globalDim,
        globalStrides,
        boxDim, 
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

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

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tma_load_4d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}

__device__ __forceinline__ void tma_store_4d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group [%0, {%2, %3, %4, %5}], [%1];"
        :: "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}

__device__ __forceinline__ void tma_store_fence_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tma_store_commit_fn() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void tma_store_wait_fn() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo, uint32_t swizzle) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    if (swizzle == 2) d |= (uint64_t)((addr >> 7) & 0x7) << 49; // 128B swizzle base_offset
    d |= (uint64_t)swizzle << 61;   
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc(uint32_t M, uint32_t N, uint32_t a_major, uint32_t b_major) {
    uint32_t d = 0;
    d |= (1u << 4);    // c_format = FP32
    d |= (1u << 7);    // a_format = BF16
    d |= (1u << 10);   // b_format = BF16
    d |= (a_major << 15);
    d |= (b_major << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__global__ void mha_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    float* LSE,
    int S_len, int H, int D) 
{
    __shared__ __align__(128) __nv_bfloat16 Q0_smem[128 * 64];
    __shared__ __align__(128) __nv_bfloat16 Q1_smem[128 * 64];
    __shared__ __align__(128) __nv_bfloat16 K0_smem[128 * 64];
    __shared__ __align__(128) __nv_bfloat16 K1_smem[128 * 64];
    __shared__ __align__(128) __nv_bfloat16 V0_smem[128 * 64];
    __shared__ __align__(128) __nv_bfloat16 V1_smem[128 * 64];
    __shared__ __align__(128) __nv_bfloat16 P0_smem[128 * 64];
    __shared__ __align__(128) __nv_bfloat16 P1_smem[128 * 64];

    __shared__ __align__(8) uint64_t mbar_q;
    __shared__ __align__(8) uint64_t mbar_kv;
    __shared__ __align__(8) uint64_t mbar_umma;

    int q_idx = blockIdx.x;
    int h_idx = blockIdx.y;
    int b_idx = blockIdx.z;
    int q_start = q_idx * 128;
    
    if (q_start >= S_len) return;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&mbar_q, 1);
        init_smem_barrier_fn(&mbar_kv, 1);
        init_smem_barrier_fn(&mbar_umma, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&mbar_q, 128 * 128 * 2);
        tma_load_4d_fn(&tma_Q, &mbar_q, Q0_smem, 0,  q_start, h_idx, b_idx);
        tma_load_4d_fn(&tma_Q, &mbar_q, Q1_smem, 64, q_start, h_idx, b_idx);
    }

    __shared__ uint32_t tmem_S_addr;
    __shared__ uint32_t tmem_O0_addr;
    __shared__ uint32_t tmem_O1_addr;
    if (threadIdx.x < 32) {
        tmem_alloc_fn(&tmem_S_addr, 128);
        tmem_alloc_fn(&tmem_O0_addr, 64);
        tmem_alloc_fn(&tmem_O1_addr, 64);
    }

    float m_max = -INFINITY;
    float sum_exp = 0.0f;
    float O0_reg[64] = {0};
    float O1_reg[64] = {0};

    int num_kv_blocks = (S_len + 127) / 128;
    int phase_kv = 0;
    int phase_umma = 0;

    mbarrier_wait_fn(&mbar_q, 0);

    for (int kv_idx = 0; kv_idx < num_kv_blocks; ++kv_idx) {
        int kv_start = kv_idx * 128;

        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&mbar_kv, 128 * 128 * 2 * 2);
            tma_load_4d_fn(&tma_K, &mbar_kv, K0_smem, 0,  kv_start, h_idx, b_idx);
            tma_load_4d_fn(&tma_K, &mbar_kv, K1_smem, 64, kv_start, h_idx, b_idx);
            tma_load_4d_fn(&tma_V, &mbar_kv, V0_smem, 0,  kv_start, h_idx, b_idx);
            tma_load_4d_fn(&tma_V, &mbar_kv, V1_smem, 64, kv_start, h_idx, b_idx);
        }
        mbarrier_wait_fn(&mbar_kv, phase_kv);
        phase_kv ^= 1;

        uint32_t idesc_S = make_instr_desc(128, 128, 0, 0); // K-Major, K-Major
        if (threadIdx.x == 0) {
            for (int k = 0; k < 4; ++k) { // Q0 @ K0^T
                uint64_t a_desc = make_smem_desc_sm100_fn(Q0_smem + k * 16, 1, 1024, 2);
                uint64_t b_desc = make_smem_desc_sm100_fn(K0_smem + k * 16, 1, 1024, 2);
                uint32_t accum = (k == 0) ? 0 : 1;
                asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                             :: "r"(tmem_S_addr), "l"(a_desc), "l"(b_desc), "r"(idesc_S), "r"(accum));
            }
            for (int k = 0; k < 4; ++k) { // Q1 @ K1^T
                uint64_t a_desc = make_smem_desc_sm100_fn(Q1_smem + k * 16, 1, 1024, 2);
                uint64_t b_desc = make_smem_desc_sm100_fn(K1_smem + k * 16, 1, 1024, 2);
                uint32_t accum = 1;
                asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                             :: "r"(tmem_S_addr), "l"(a_desc), "l"(b_desc), "r"(idesc_S), "r"(accum));
            }
            uint32_t mbar_addr = (uint32_t)__cvta_generic_to_shared(&mbar_umma);
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(mbar_addr));
        }
        mbarrier_wait_fn(&mbar_umma, phase_umma);
        phase_umma ^= 1;

        float row_max = -INFINITY;
        for (int col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_S_addr + col));
            tmem_load_fence_fn();

            float f0 = __uint_as_float(r0) * 0.0883883476f;
            float f1 = __uint_as_float(r1) * 0.0883883476f;
            float f2 = __uint_as_float(r2) * 0.0883883476f;
            float f3 = __uint_as_float(r3) * 0.0883883476f;
            int g_col = kv_start + col;
            if (g_col + 0 >= S_len) f0 = -INFINITY;
            if (g_col + 1 >= S_len) f1 = -INFINITY;
            if (g_col + 2 >= S_len) f2 = -INFINITY;
            if (g_col + 3 >= S_len) f3 = -INFINITY;

            row_max = max(row_max, max(max(f0, f1), max(f2, f3)));
        }

        float m_new = max(m_max, row_max);
        float exp_diff = exp2f((m_max - m_new) * 1.44269504f);
        if (m_max == -INFINITY) exp_diff = 0.0f;
        if (m_new == -INFINITY) exp_diff = 1.0f;

        for (int i = 0; i < 64; ++i) { O0_reg[i] *= exp_diff; O1_reg[i] *= exp_diff; }
        sum_exp *= exp_diff;

        int row = threadIdx.x;
        for (int col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_S_addr + col));
            tmem_load_fence_fn();

            float f0 = __uint_as_float(r0) * 0.0883883476f;
            float f1 = __uint_as_float(r1) * 0.0883883476f;
            float f2 = __uint_as_float(r2) * 0.0883883476f;
            float f3 = __uint_as_float(r3) * 0.0883883476f;
            int g_col = kv_start + col;

            float p0 = (g_col + 0 >= S_len || f0 == -INFINITY) ? 0.0f : exp2f((f0 - m_new) * 1.44269504f);
            float p1 = (g_col + 1 >= S_len || f1 == -INFINITY) ? 0.0f : exp2f((f1 - m_new) * 1.44269504f);
            float p2 = (g_col + 2 >= S_len || f2 == -INFINITY) ? 0.0f : exp2f((f2 - m_new) * 1.44269504f);
            float p3 = (g_col + 3 >= S_len || f3 == -INFINITY) ? 0.0f : exp2f((f3 - m_new) * 1.44269504f);
            sum_exp += p0 + p1 + p2 + p3;

            float p_arr[4] = {p0, p1, p2, p3};
            for(int j=0; j<4; ++j) {
                int c = col + j;
                int chunk_idx = (c % 64) / 8;
                int swizzled_chunk = (row % 8) ^ chunk_idx;
                int swizzled_col = swizzled_chunk * 8 + (c % 8);
                if (c < 64) P0_smem[row * 64 + swizzled_col] = __float2bfloat16(p_arr[j]);
                else P1_smem[row * 64 + swizzled_col] = __float2bfloat16(p_arr[j]);
            }
        }
        m_max = m_new;

        __syncthreads();
        if (threadIdx.x == 0) fence_async_shared_fn();
        __syncthreads();

        uint32_t idesc_P_V0 = make_instr_desc(128, 64, 0, 1); // P0 is K-Major, V0 is N-Major
        if (threadIdx.x == 0) {
            for (int k = 0; k < 4; ++k) { // P0 @ V0_top
                uint64_t a_desc = make_smem_desc_sm100_fn(P0_smem + k * 16, 1, 1024, 2);
                uint64_t b_desc = make_smem_desc_sm100_fn(V0_smem + k * 1024, 16384, 1024, 2);
                uint32_t accum = (k == 0) ? 0 : 1;
                asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                             :: "r"(tmem_O0_addr), "l"(a_desc), "l"(b_desc), "r"(idesc_P_V0), "r"(accum));
            }
            for (int k = 0; k < 4; ++k) { // P1 @ V0_bottom
                uint64_t a_desc = make_smem_desc_sm100_fn(P1_smem + k * 16, 1, 1024, 2);
                uint64_t b_desc = make_smem_desc_sm100_fn(V0_smem + (k + 4) * 1024, 16384, 1024, 2);
                asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                             :: "r"(tmem_O0_addr), "l"(a_desc), "l"(b_desc), "r"(idesc_P_V0), "r"(1)); // accum=1
            }

            for (int k = 0; k < 4; ++k) { // P0 @ V1_top
                uint64_t a_desc = make_smem_desc_sm100_fn(P0_smem + k * 16, 1, 1024, 2);
                uint64_t b_desc = make_smem_desc_sm100_fn(V1_smem + k * 1024, 16384, 1024, 2);
                uint32_t accum = (k == 0) ? 0 : 1;
                asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                             :: "r"(tmem_O1_addr), "l"(a_desc), "l"(b_desc), "r"(idesc_P_V0), "r"(accum));
            }
            for (int k = 0; k < 4; ++k) { // P1 @ V1_bottom
                uint64_t a_desc = make_smem_desc_sm100_fn(P1_smem + k * 16, 1, 1024, 2);
                uint64_t b_desc = make_smem_desc_sm100_fn(V1_smem + (k + 4) * 1024, 16384, 1024, 2);
                asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                             :: "r"(tmem_O1_addr), "l"(a_desc), "l"(b_desc), "r"(idesc_P_V0), "r"(1)); // accum=1
            }

            uint32_t mbar_addr = (uint32_t)__cvta_generic_to_shared(&mbar_umma);
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(mbar_addr));
        }
        mbarrier_wait_fn(&mbar_umma, phase_umma);
        phase_umma ^= 1;

        for (int col = 0; col < 64; col += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_O0_addr + col));
            tmem_load_fence_fn();
            O0_reg[col+0] += __uint_as_float(r0); O0_reg[col+1] += __uint_as_float(r1);
            O0_reg[col+2] += __uint_as_float(r2); O0_reg[col+3] += __uint_as_float(r3);
        }
        for (int col = 0; col < 64; col += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_O1_addr + col));
            tmem_load_fence_fn();
            O1_reg[col+0] += __uint_as_float(r0); O1_reg[col+1] += __uint_as_float(r1);
            O1_reg[col+2] += __uint_as_float(r2); O1_reg[col+3] += __uint_as_float(r3);
        }
    }

    if (threadIdx.x < 32) {
        tmem_dealloc_fn(tmem_S_addr, 128);
        tmem_dealloc_fn(tmem_O0_addr, 64);
        tmem_dealloc_fn(tmem_O1_addr, 64);
    }

    float inv_sum = 1.0f / sum_exp;
    for (int i = 0; i < 64; ++i) {
        O0_reg[i] *= inv_sum;
        O1_reg[i] *= inv_sum;
    }

    __syncthreads();
    int row = threadIdx.x;
    for (int col = 0; col < 64; col++) {
        int chunk_idx = col / 8;
        int swizzled_chunk = (row % 8) ^ chunk_idx;
        int swizzled_col = swizzled_chunk * 8 + (col % 8);
        P0_smem[row * 64 + swizzled_col] = __float2bfloat16(O0_reg[col]);
        P1_smem[row * 64 + swizzled_col] = __float2bfloat16(O1_reg[col]);
    }

    if (threadIdx.x == 0) {
        tma_store_fence_fn();
        tma_store_4d_fn(&tma_O, P0_smem, 0,  q_start, h_idx, b_idx);
        tma_store_4d_fn(&tma_O, P1_smem, 64, q_start, h_idx, b_idx);
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
    }

    if (q_start + row < S_len) {
        LSE[b_idx * H * S_len + h_idx * S_len + q_start + row] = m_max + logf(sum_exp);
    }
}

namespace tvm_ffi_example_cuda {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int64_t B = 4;
    int64_t H = 48;
    int64_t S = Q.size(2);
    int64_t D = 128;
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O;
    CUDA_CHECK(create_tma_4d_descriptor_2B(&tma_Q, Q.data_ptr(), D, S, H, B, 64, 128));
    CUDA_CHECK(create_tma_4d_descriptor_2B(&tma_K, K.data_ptr(), D, S, H, B, 64, 128));
    CUDA_CHECK(create_tma_4d_descriptor_2B(&tma_V, V.data_ptr(), D, S, H, B, 64, 128));
    CUDA_CHECK(create_tma_4d_descriptor_2B(&tma_O, O.data_ptr(), D, S, H, B, 64, 128));

    int blocks_S = (S + 127) / 128;
    dim3 grid(blocks_S, H, B);
    dim3 block(128, 1, 1);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    mha_kernel<<<grid, block, 0, stream>>>(
        tma_Q, tma_K, tma_V, tma_O,
        static_cast<float*>(LSE.data_ptr()),
        S, H, D
    );
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda