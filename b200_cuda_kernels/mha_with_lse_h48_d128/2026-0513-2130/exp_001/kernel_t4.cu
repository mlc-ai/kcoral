#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
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

namespace tvm_ffi_example_cuda {

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_fn(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])) : "memory");
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

__device__ __forceinline__ void tma_store_commit_fn() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void tma_store_wait_fn() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ uint32_t pack_bf16_fn(uint32_t fp32_a, uint32_t fp32_b) {
    __nv_bfloat16 a = __float2bfloat16(__uint_as_float(fp32_a));
    __nv_bfloat16 b = __float2bfloat16(__uint_as_float(fp32_b));
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};" : "=r"(result) : "h"(*reinterpret_cast<uint16_t*>(&a)), "h"(*reinterpret_cast<uint16_t*>(&b)));
    return result;
}

__device__ __forceinline__ void tmem_load_8x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3, uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3),
                   "=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7) : "r"(col));
}

__device__ __forceinline__ void tmem_store_4x_fn(uint32_t col, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
                 :: "r"(col), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
}

__device__ __forceinline__ void tmem_store_8x_fn(uint32_t col, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3, uint32_t r4, uint32_t r5, uint32_t r6, uint32_t r7) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
                 :: "r"(col), "r"(r0), "r"(r1), "r"(r2), "r"(r3),
                    "r"(r4), "r"(r5), "r"(r6), "r"(r7) : "memory");
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_store_fence_fn() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void umma_commit_1sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(a));
}

__device__ __forceinline__ uint64_t make_smem_desc_none_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr) >> 4;
    d |= (uint64_t)(addr & 0x3FFF);
    d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16;
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46; // version = 1 (SM100)
    d |= (uint64_t)0 << 61; // SWIZZLE_NONE
    return d;
}

struct __align__(128) SharedStorage {
    __nv_bfloat16 Q[128 * 128];     // 32KB
    __nv_bfloat16 K[2][128 * 128];  // 64KB
    __nv_bfloat16 V[2][128 * 128];  // 64KB
    uint64_t mbar_Q[1];
    uint64_t mbar_K[2];
    uint64_t mbar_V[2];
    uint64_t umma_mbar[1];
    uint32_t tmem_addr;
};

// Explicit capacity allocation for large structures mapped to dynamic bounds limitations
extern __shared__ __align__(128) uint8_t smem_raw[];

__global__ void mha_forward_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    float* LSE_ptr,
    int B, int H, int S, int D
) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int S_q_start = blockIdx.x * 128;

    if (S_q_start >= S) return;

    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(smem_raw);

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&smem.mbar_Q[0], 1);
        init_smem_barrier_fn(&smem.mbar_K[0], 1);
        init_smem_barrier_fn(&smem.mbar_V[0], 1);
        init_smem_barrier_fn(&smem.mbar_K[1], 1);
        init_smem_barrier_fn(&smem.mbar_V[1], 1);
        init_smem_barrier_fn(&smem.umma_mbar[0], 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    // Allocation is warped-synchronous per issue constraint granularity. Thread 0..31 executes.
    if (threadIdx.x < 32) {
        uint32_t a = (uint32_t)__cvta_generic_to_shared(&smem.tmem_addr);
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a), "r"(512));
    }
    __syncthreads();
    
    uint32_t O_col = smem.tmem_addr;
    uint32_t S_col = smem.tmem_addr + 128;
    uint32_t P_col = smem.tmem_addr + 256;

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&smem.mbar_Q[0], 128 * 128 * 2);
        tma_load_4d_fn(&tma_Q, &smem.mbar_Q[0], smem.Q, 0, S_q_start, h, b);

        mbarrier_arrive_and_expect_tx_fn(&smem.mbar_K[0], 128 * 128 * 2);
        tma_load_4d_fn(&tma_K, &smem.mbar_K[0], smem.K[0], 0, 0, h, b);

        mbarrier_arrive_and_expect_tx_fn(&smem.mbar_V[0], 128 * 128 * 2);
        tma_load_4d_fn(&tma_V, &smem.mbar_V[0], smem.V[0], 0, 0, h, b);
    }

    uint32_t idesc_qk = (1u << 4) | (1u << 7) | (1u << 10) | (0u << 15) | (0u << 16) | (16u << 17) | (8u << 24);
    uint32_t idesc_pv = (1u << 4) | (1u << 7) | (1u << 10) | (0u << 15) | (1u << 16) | (16u << 17) | (8u << 24);

    mbarrier_wait_fn(&smem.mbar_Q[0], 0);

    float m_old = -INFINITY;
    float d_old = 0.0f;
    int phase_K[2] = {0, 0};
    int phase_V[2] = {0, 0};
    int umma_phase = 0;
    int stage = 0;
    int num_steps = (S + 127) / 128;

    for (int step = 0; step < num_steps; ++step) {
        mbarrier_wait_fn(&smem.mbar_K[stage], phase_K[stage]);
        mbarrier_wait_fn(&smem.mbar_V[stage], phase_V[stage]);

        // Sync to enforce TMEM layout completion before next iteration starts TMA load overlaps
        __syncthreads(); 

        int next_step = step + 1;
        if (next_step < num_steps) {
            int next_stage = stage ^ 1;
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&smem.mbar_K[next_stage], 128 * 128 * 2);
                tma_load_4d_fn(&tma_K, &smem.mbar_K[next_stage], smem.K[next_stage], 0, next_step * 128, h, b);

                mbarrier_arrive_and_expect_tx_fn(&smem.mbar_V[next_stage], 128 * 128 * 2);
                tma_load_4d_fn(&tma_V, &smem.mbar_V[next_stage], smem.V[next_stage], 0, next_step * 128, h, b);
            }
        }

        tcgen05_fence_after_fn();
        
        // Single thread initiates the sequence of the TCGen05 operations. 
        if (threadIdx.x == 0) {
            for (int k = 0; k < 8; ++k) {
                // Q is K-Major -> LBO=16, SBO=2048
                uint64_t a_desc = make_smem_desc_none_fn(smem.Q + k * 16, 16, 2048);
                // K is K-Major -> LBO=16, SBO=2048 
                uint64_t b_desc = make_smem_desc_none_fn(smem.K[stage] + k * 16, 16, 2048);
                uint32_t accum = (k > 0) ? 1 : 0;
                asm volatile(
                    "{\n.reg .pred p;\n"
                    "setp.ne.b32 p, %3, 0;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %4, p;\n}\n"
                    :: "r"(S_col), "l"(a_desc), "l"(b_desc), "r"(accum), "r"(idesc_qk));
            }
            umma_commit_1sm_fn(&smem.umma_mbar[0]);
        }
        
        mbarrier_wait_fn(&smem.umma_mbar[0], umma_phase);
        tcgen05_fence_after_fn();
        umma_phase ^= 1;
        
        // Sync ensuring the unblocking of all threads before fetching directly off S_col asynchronously
        __syncthreads();

        float m_new = m_old;
        int K_start = step * 128;
        
        for (int c = 0; c < 128; c += 8) {
            uint32_t r[8];
            tmem_load_8x_fn(S_col + c, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
            tmem_load_fence_fn();
            for (int i = 0; i < 8; ++i) {
                float val;
                if (K_start + c + i >= S) val = -INFINITY;
                else val = __uint_as_float(r[i]) * 0.08838834764f; // S = S * 1/sqrt(D)
                m_new = fmaxf(m_new, val);
            }
        }

        float exp_scale = exp2f((m_old - m_new) * 1.44269504089f);
        float d_new = d_old * exp_scale;

        for (int c = 0; c < 128; c += 8) {
            uint32_t r[8];
            tmem_load_8x_fn(S_col + c, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
            tmem_load_fence_fn();
            uint32_t p_bf16[4];
            for (int i = 0; i < 8; i += 2) {
                float v0, v1;
                if (K_start + c + i >= S) v0 = -INFINITY;
                else v0 = __uint_as_float(r[i]) * 0.08838834764f;
                if (K_start + c + i + 1 >= S) v1 = -INFINITY;
                else v1 = __uint_as_float(r[i+1]) * 0.08838834764f;
                
                float p0 = exp2f((v0 - m_new) * 1.44269504089f);
                float p1 = exp2f((v1 - m_new) * 1.44269504089f);
                if (K_start + c + i >= S) p0 = 0.0f;
                if (K_start + c + i + 1 >= S) p1 = 0.0f;
                
                d_new += p0 + p1;
                p_bf16[i/2] = pack_bf16_fn(__float_as_uint(p0), __float_as_uint(p1));
            }
            tmem_store_4x_fn(P_col + c / 2, p_bf16[0], p_bf16[1], p_bf16[2], p_bf16[3]);
        }

        if (step > 0 && exp_scale != 1.0f) {
            for (int c = 0; c < 128; c += 8) {
                uint32_t r[8];
                tmem_load_8x_fn(O_col + c, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
                tmem_load_fence_fn();
                for (int i = 0; i < 8; ++i) {
                    float val = __uint_as_float(r[i]) * exp_scale;
                    r[i] = __float_as_uint(val);
                }
                tmem_store_8x_fn(O_col + c, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
            }
        }
        tmem_store_fence_fn(); // Barrier ensuring P_col & O_col updates complete

        m_old = m_new;
        d_old = d_new;

        // Ensure stores of P elements and scales of prior outputs complete completely prior to proceeding for PV
        __syncthreads();

        tcgen05_fence_after_fn();
        
        // Single thread limits UMMA execution to exactly single-thread initiator limits on TCGen05.
        if (threadIdx.x == 0) {
            for (int k = 0; k < 8; ++k) {
                uint32_t a_tmem = (0 << 16) | (P_col + k * 8);
                // V is MN-Major -> LBO=2048, SBO=16
                uint64_t b_desc_v = make_smem_desc_none_fn(smem.V[stage] + k * 16 * 128, 2048, 16);
                uint32_t accum = (step > 0 || k > 0) ? 1 : 0;
                asm volatile(
                    "{\n.reg .pred p;\n"
                    "setp.ne.b32 p, %3, 0;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %4, p;\n}\n"
                    :: "r"(O_col), "r"(a_tmem), "l"(b_desc_v), "r"(accum), "r"(idesc_pv));
            }
            umma_commit_1sm_fn(&smem.umma_mbar[0]);
        }
        
        mbarrier_wait_fn(&smem.umma_mbar[0], umma_phase);
        tcgen05_fence_after_fn();
        umma_phase ^= 1;

        // Block trailing operations until all threads synchronize
        __syncthreads();

        phase_K[stage] ^= 1;
        phase_V[stage] ^= 1;
        stage ^= 1;
    }

    __syncthreads();
    
    float d_safe = d_old > 0 ? d_old : 1.0f;
    for (int c = 0; c < 128; c += 8) {
        uint32_t r[8];
        tmem_load_8x_fn(O_col + c, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
        tmem_load_fence_fn();
        uint32_t out_bf16[4];
        for (int i = 0; i < 8; i += 2) {
            float o0 = __uint_as_float(r[i]) / d_safe;
            float o1 = __uint_as_float(r[i+1]) / d_safe;
            out_bf16[i/2] = pack_bf16_fn(__float_as_uint(o0), __float_as_uint(o1));
        }
        uint32_t* smem_row = (uint32_t*)&smem.Q[threadIdx.x * 128 + c];
        smem_row[0] = out_bf16[0];
        smem_row[1] = out_bf16[1];
        smem_row[2] = out_bf16[2];
        smem_row[3] = out_bf16[3];
    }
    __syncthreads();
    fence_async_shared_fn(); // Barrier sync between general access modification to memory prior to usage with TMA pipeline 

    if (threadIdx.x == 0) {
        tma_store_4d_fn(&tma_O, smem.Q, 0, S_q_start, h, b);
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
    }

    if (S_q_start + threadIdx.x < S) {
        LSE_ptr[(b * H + h) * S + S_q_start + threadIdx.x] = m_old + logf(d_safe);
    }

    __syncthreads();
    if (threadIdx.x < 32) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(smem.tmem_addr), "r"(512));
    }
}

CUresult create_tma_4d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t dim0, uint64_t dim1, uint64_t dim2, uint64_t dim3, uint32_t box0, uint32_t box1, uint32_t box2, uint32_t box3) {
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

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3); 
    
    if (S == 0 || B == 0 || H == 0 || D == 0) return;

    CUtensorMap tma_Q, tma_K, tma_V, tma_O;
    create_tma_4d_descriptor_2B(&tma_Q, Q.data_ptr(), D, S, H, B, 128, 128, 1, 1);
    create_tma_4d_descriptor_2B(&tma_K, K.data_ptr(), D, S, H, B, 128, 128, 1, 1);
    create_tma_4d_descriptor_2B(&tma_V, V.data_ptr(), D, S, H, B, 128, 128, 1, 1);
    create_tma_4d_descriptor_2B(&tma_O, O.data_ptr(), D, S, H, B, 128, 128, 1, 1);
    
    int threads = 128;
    int blocks_S = (S + 127) / 128;
    dim3 blocks(blocks_S, H, B);
    
    int smem_size = sizeof(SharedStorage);
    CUDA_CHECK(cudaFuncSetAttribute(mha_forward_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    mha_forward_kernel<<<blocks, threads, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, tma_O, static_cast<float*>(LSE.data_ptr()),
        B, H, S, D
    );
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}