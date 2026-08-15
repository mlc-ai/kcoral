#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
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

#define CU_CHECK(call) do {                                        \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        const char* err_str;                                       \
        cuGetErrorString(_e, &err_str);                            \
        fprintf(stderr, "CU error %s at %s:%d\n",                  \
                err_str, __FILE__, __LINE__);                      \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_mha {

CUresult create_tma_4d_descriptor_2B(CUtensorMap* d, void* globalAddress,
    uint64_t dim3, uint64_t dim2, uint64_t dim1, uint64_t dim0,
    uint32_t box3, uint32_t box2, uint32_t box1, uint32_t box0) {
    cuuint64_t globalDim[4] = {dim0, dim1, dim2, dim3};
    cuuint64_t globalStrides[3] = {dim0 * 2, dim0 * dim1 * 2, dim0 * dim1 * dim2 * 2};
    cuuint32_t boxDim[4] = {box0, box1, box2, box3};
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
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
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

__device__ __forceinline__ void tma_store_commit_fn() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void tma_store_wait_fn() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
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

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)0 << 61; // SWIZZLE_NONE
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    // c_format = FP32
    d |= (1u << 7);    // a_format = BF16
    d |= (1u << 10);   // b_format = BF16
    d |= (0u << 15);   // a_major = 0 (K-Major)
    d |= (0u << 16);   // b_major = 0 (K-Major)
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_PV_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    // c_format = FP32
    d |= (1u << 7);    // a_format = BF16
    d |= (1u << 10);   // b_format = BF16
    d |= (0u << 15);   // a_major = 0 (K-Major)
    d |= (1u << 16);   // b_major = 1 (N-Major)
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
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
        ".mbarrier::arrive::one.b64"
        " [%0];"
        :: "r"(a));
}

struct SharedStorage {
    __align__(1024) __nv_bfloat16 Q[128][128];
    __align__(1024) __nv_bfloat16 K[128][128];
    __align__(1024) __nv_bfloat16 V[128][128];
    __align__(1024) __nv_bfloat16 P[128][128];
    __align__(8) uint64_t bar_Q;
    __align__(8) uint64_t bar_KV;
    __align__(8) uint64_t bar_S;
    __align__(8) uint64_t bar_O;
    __align__(8) uint32_t tmem_addr_S;
    __align__(8) uint32_t tmem_addr_O;
};

__global__ void mha_fwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    float* LSE,
    int S,
    int H
) {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 248;" ::: "memory");

    extern __shared__ char smem_buf[];
    uintptr_t smem_ptr = reinterpret_cast<uintptr_t>(smem_buf);
    smem_ptr = (smem_ptr + 1023) & ~1023ULL;
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(smem_ptr);

    int q_idx = blockIdx.x * 128;
    int head_idx = blockIdx.y;
    int batch_idx = blockIdx.z;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&smem.bar_Q, 1);
        init_smem_barrier_fn(&smem.bar_KV, 1);
        init_smem_barrier_fn(&smem.bar_S, 1);
        init_smem_barrier_fn(&smem.bar_O, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    if (threadIdx.x < 32) {
        tmem_alloc_cg1_fn(&smem.tmem_addr_S, 128);
        tmem_alloc_cg1_fn(&smem.tmem_addr_O, 128);
    }
    __syncthreads();
    uint32_t tmem_S = smem.tmem_addr_S;
    uint32_t tmem_O = smem.tmem_addr_O;

    uint32_t phase_Q = 0;
    uint32_t phase_KV = 0;
    uint32_t phase_S = 0;
    uint32_t phase_O = 0;

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&smem.bar_Q, 128 * 128 * 2);
        tma_load_4d_fn(&tma_Q, &smem.bar_Q, smem.Q, 0, q_idx, head_idx, batch_idx);
    }
    mbarrier_wait_fn(&smem.bar_Q, phase_Q);
    phase_Q ^= 1;

    float O_accum_reg[128];
    for (int i = 0; i < 128; ++i) O_accum_reg[i] = 0.0f;
    float m_old = -INFINITY;
    float row_sum_reg = 0.0f;

    float scale = 0.0883883476483f; // 1.0 / sqrt(128)
    uint32_t idesc_QK = make_instr_desc_fn(128, 128);
    uint32_t idesc_PV = make_instr_desc_PV_fn(128, 128);

    for (int kv_idx = 0; kv_idx < S; kv_idx += 128) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&smem.bar_KV, 2 * 128 * 128 * 2);
            tma_load_4d_fn(&tma_K, &smem.bar_KV, smem.K, 0, kv_idx, head_idx, batch_idx);
            tma_load_4d_fn(&tma_V, &smem.bar_KV, smem.V, 0, kv_idx, head_idx, batch_idx);
        }
        mbarrier_wait_fn(&smem.bar_KV, phase_KV);
        phase_KV ^= 1;

        if (threadIdx.x == 0) {
            for (int i = 0; i < 8; ++i) {
                uint8_t* ptr_Q = (uint8_t*)smem.Q + i * 32;
                uint8_t* ptr_K = (uint8_t*)smem.K + i * 32;
                uint64_t desc_Q = make_smem_desc_sm100_fn(ptr_Q, 16, 2048);
                uint64_t desc_K = make_smem_desc_sm100_fn(ptr_K, 16, 2048);
                umma_f16_cg1_fn(tmem_S, desc_Q, desc_K, idesc_QK, (i == 0) ? 0 : 1);
            }
            umma_commit_cg1_fn(&smem.bar_S);
        }
        mbarrier_wait_fn(&smem.bar_S, phase_S);
        phase_S ^= 1;

        float m_new = m_old;
        for (int c = 0; c < 128; c += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_S + c));
            tmem_load_fence_fn();

            float v0 = __uint_as_float(r0);
            float v1 = __uint_as_float(r1);
            float v2 = __uint_as_float(r2);
            float v3 = __uint_as_float(r3);

            if (kv_idx + c + 0 >= S) v0 = -INFINITY; else v0 *= scale;
            if (kv_idx + c + 1 >= S) v1 = -INFINITY; else v1 *= scale;
            if (kv_idx + c + 2 >= S) v2 = -INFINITY; else v2 *= scale;
            if (kv_idx + c + 3 >= S) v3 = -INFINITY; else v3 *= scale;

            if (v0 > m_new) m_new = v0;
            if (v1 > m_new) m_new = v1;
            if (v2 > m_new) m_new = v2;
            if (v3 > m_new) m_new = v3;
        }

        float factor = fast_exp2f_fn((m_old - m_new) * 1.4426950408889634f);
        for (int c = 0; c < 128; ++c) {
            O_accum_reg[c] *= factor;
        }
        row_sum_reg *= factor;

        for (int c = 0; c < 128; c += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_S + c));
            tmem_load_fence_fn();

            float v0 = __uint_as_float(r0);
            float v1 = __uint_as_float(r1);
            float v2 = __uint_as_float(r2);
            float v3 = __uint_as_float(r3);

            if (kv_idx + c + 0 >= S) v0 = -INFINITY; else v0 *= scale;
            if (kv_idx + c + 1 >= S) v1 = -INFINITY; else v1 *= scale;
            if (kv_idx + c + 2 >= S) v2 = -INFINITY; else v2 *= scale;
            if (kv_idx + c + 3 >= S) v3 = -INFINITY; else v3 *= scale;

            float p0 = fast_exp2f_fn((v0 - m_new) * 1.4426950408889634f);
            float p1 = fast_exp2f_fn((v1 - m_new) * 1.4426950408889634f);
            float p2 = fast_exp2f_fn((v2 - m_new) * 1.4426950408889634f);
            float p3 = fast_exp2f_fn((v3 - m_new) * 1.4426950408889634f);

            row_sum_reg += p0 + p1 + p2 + p3;

            smem.P[threadIdx.x][c+0] = __float2bfloat16(p0);
            smem.P[threadIdx.x][c+1] = __float2bfloat16(p1);
            smem.P[threadIdx.x][c+2] = __float2bfloat16(p2);
            smem.P[threadIdx.x][c+3] = __float2bfloat16(p3);
        }
        m_old = m_new;

        fence_proxy_async_fn();
        __syncthreads();

        if (threadIdx.x == 0) {
            for (int i = 0; i < 8; ++i) {
                uint8_t* ptr_P = (uint8_t*)smem.P + i * 32;
                uint8_t* ptr_V = (uint8_t*)smem.V + i * 4096;
                uint64_t desc_P = make_smem_desc_sm100_fn(ptr_P, 16, 2048);
                uint64_t desc_V = make_smem_desc_sm100_fn(ptr_V, 128, 256);
                umma_f16_cg1_fn(tmem_O, desc_P, desc_V, idesc_PV, (i == 0) ? 0 : 1);
            }
            umma_commit_cg1_fn(&smem.bar_O);
        }
        mbarrier_wait_fn(&smem.bar_O, phase_O);
        phase_O ^= 1;

        for (int c = 0; c < 128; c += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_O + c));
            tmem_load_fence_fn();
            O_accum_reg[c+0] += __uint_as_float(r0);
            O_accum_reg[c+1] += __uint_as_float(r1);
            O_accum_reg[c+2] += __uint_as_float(r2);
            O_accum_reg[c+3] += __uint_as_float(r3);
        }
    }

    float inv_sum = (row_sum_reg > 0.0f) ? (1.0f / row_sum_reg) : 0.0f;
    for (int c = 0; c < 128; c += 2) {
        float o0 = O_accum_reg[c] * inv_sum;
        float o1 = O_accum_reg[c+1] * inv_sum;
        smem.P[threadIdx.x][c] = __float2bfloat16(o0);
        smem.P[threadIdx.x][c+1] = __float2bfloat16(o1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        tma_store_4d_fn(&tma_O, smem.P, 0, q_idx, head_idx, batch_idx);
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
    }
    
    if (q_idx + threadIdx.x < S) {
        float lse_val = m_old + logf(row_sum_reg);
        LSE[batch_idx * H * S + head_idx * S + q_idx + threadIdx.x] = lse_val;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        tmem_dealloc_cg1_fn(tmem_S, 128);
        tmem_dealloc_cg1_fn(tmem_O, 128);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);

    CUtensorMap tma_Q, tma_K, tma_V, tma_O;
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_Q, Q.data_ptr(), B, H, S, D, 1, 1, 128, 128));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_K, K.data_ptr(), B, H, S, D, 1, 1, 128, 128));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_V, V.data_ptr(), B, H, S, D, 1, 1, 128, 128));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_O, O.data_ptr(), B, H, S, D, 1, 1, 128, 128));
    
    int num_blocks = (S + 127) / 128;
    dim3 grid(num_blocks, H, B);
    dim3 block(128);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    size_t smem_bytes = sizeof(SharedStorage) + 1024; // Ensure alignment space
    CUDA_CHECK(cudaFuncSetAttribute(mha_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    
    mha_fwd_kernel<<<grid, block, smem_bytes, stream>>>(
        tma_Q, tma_K, tma_V, tma_O,
        static_cast<float*>(LSE.data_ptr()),
        static_cast<int>(S),
        static_cast<int>(H)
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}  // namespace tvm_ffi_mha