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
        fprintf(stderr, "CU error %d at %s:%d\n",                 \
                (int)_e, __FILE__, __LINE__);                      \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_example_cuda {

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ uint32_t pack_bf16_fn(float a_f, float b_f) {
    __nv_bfloat16 a = __float2bfloat16(a_f);
    __nv_bfloat16 b = __float2bfloat16(b_f);
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};"
        : "=r"(result)
        : "h"(*reinterpret_cast<uint16_t*>(&a)),
          "h"(*reinterpret_cast<uint16_t*>(&b)));
    return result;
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
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

__device__ __forceinline__ void tma_load_2d_cta(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_store_2d_cta(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%2, %3}], [%1];"
        :: "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_store_commit_fn() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void tma_store_wait_fn() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tmem_alloc_cg1_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_cg1_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(addr), "r"(ncols));
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
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(a));
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

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tma_store_fence_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_none(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)0 << 61; // CU_TENSOR_MAP_SWIZZLE_NONE
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_custom_fn(uint32_t M, uint32_t N, int a_major, int b_major) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= ((uint32_t)a_major << 15);   
    d |= ((uint32_t)b_major << 16); 
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    uint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    uint64_t globalStrides[1] = {gmem_inner_dim * 2};
    uint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    uint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill
    );
}

struct SharedStorage {
    alignas(128) __nv_bfloat16 Q[128 * 128]; // 32KB
    alignas(128) __nv_bfloat16 K0[64 * 128]; // 16KB
    alignas(128) __nv_bfloat16 K1[64 * 128]; 
    alignas(128) __nv_bfloat16 V0[64 * 128]; 
    alignas(128) __nv_bfloat16 V1[64 * 128]; 
    alignas(128) __nv_bfloat16 P[128 * 64];  // 16KB
    alignas(8) uint64_t bar_mma[1];
    alignas(8) uint64_t bar_tma[2];
    alignas(16) uint32_t tmem_addr;
};

__global__ void __launch_bounds__(128, 1) mha_fwd_sm100_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    float* __restrict__ LSE,
    int B, int H, int S, int D)
{
    extern __shared__ char smem_buf[];
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(smem_buf);

    int b = blockIdx.z;
    int h = blockIdx.y;
    int m_block = blockIdx.x;
    int tid = threadIdx.x;

    float* lse_ptr = LSE + b * H * S + h * S + m_block * 128;
    int coord1_q = (b * H + h) * S + m_block * 128;

    if (tid == 0) {
        init_smem_barrier_fn(&smem.bar_tma[0], 1);
        init_smem_barrier_fn(&smem.bar_tma[1], 1);
        init_smem_barrier_fn(smem.bar_mma, 1);
        
        mbarrier_arrive_and_expect_tx_fn(&smem.bar_tma[0], 32768); 
        tma_load_2d_cta(&tma_Q, &smem.bar_tma[0], smem.Q, 0, coord1_q);
    }
    
    if (tid < 32) {
        tmem_alloc_cg1_fn(&smem.tmem_addr, 256); 
    }
    __syncthreads();
    
    if (tid == 0) {
        mbarrier_wait_fn(&smem.bar_tma[0], 0); 
    }
    __syncthreads();

    uint32_t tmem_S = smem.tmem_addr;       
    uint32_t tmem_O = smem.tmem_addr + 128; 
    
    uint32_t mma_phase = 0;
    int db = 0;
    int phase_tma[2] = {0, 0};

    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(&smem.bar_tma[0], 32768); 
        int kv_seq = (b * H + h) * S + 0;
        tma_load_2d_cta(&tma_K, &smem.bar_tma[0], smem.K0, 0, kv_seq);
        tma_load_2d_cta(&tma_V, &smem.bar_tma[0], smem.V0, 0, kv_seq);
    }

    float m_i = -INFINITY;
    float l_i = 0.0f;
    float O_reg[128] = {0};

    int num_n_blocks = (S + 63) / 64;
    for(int n_block = 0; n_block < num_n_blocks; ++n_block) {
        int next_db = db ^ 1;
        if (n_block + 1 < num_n_blocks) {
            if (tid == 0) {
                mbarrier_arrive_and_expect_tx_fn(&smem.bar_tma[next_db], 32768);
                int next_kv_seq = (b * H + h) * S + (n_block + 1) * 64;
                void* K_dst = (next_db == 0) ? smem.K0 : smem.K1;
                void* V_dst = (next_db == 0) ? smem.V0 : smem.V1;
                tma_load_2d_cta(&tma_K, &smem.bar_tma[next_db], K_dst, 0, next_kv_seq);
                tma_load_2d_cta(&tma_V, &smem.bar_tma[next_db], V_dst, 0, next_kv_seq);
            }
        }
        
        if (tid == 0) mbarrier_wait_fn(&smem.bar_tma[db], phase_tma[db]);
        __syncthreads();
        phase_tma[db] ^= 1;
        
        void* K_curr = (db == 0) ? smem.K0 : smem.K1;
        void* V_curr = (db == 0) ? smem.V0 : smem.V1;

        if (tid == 0) {
            tcgen05_fence_after_fn();
            for (int k = 0; k < 128; k += 16) {
                uint64_t d_Q = make_smem_desc_sm100_none((char*)smem.Q + k * 2, 2048, 128);
                uint64_t d_K = make_smem_desc_sm100_none((char*)K_curr + k * 2, 1024, 128);
                uint32_t idesc = make_instr_desc_custom_fn(128, 64, 0, 0); 
                umma_f16_cg1_fn(tmem_S, d_Q, d_K, idesc, (k == 0) ? 0 : 1);
            }
            umma_commit_cg1_fn(smem.bar_mma);
        }
        mbarrier_wait_fn(smem.bar_mma, mma_phase);
        __syncthreads();
        mma_phase ^= 1;

        float row_max = -INFINITY;
        float P_reg[64];
        for (int c = 0; c < 64; c += 8) {
            uint32_t r[8];
            tmem_load_8x_fn(tmem_S + c, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
            tmem_load_fence_fn(); 
            for(int i = 0; i < 8; ++i) {
                float val = __uint_as_float(r[i]);
                if (n_block * 64 + c + i >= S) val = -INFINITY;
                val *= 0.08838834764f;
                P_reg[c + i] = val;
                row_max = max(row_max, val);
            }
        }

        float m_new = max(m_i, row_max);
        float exp_diff = expf(m_i - m_new);
        l_i = l_i * exp_diff;
        float row_sum_new = 0.0f;

        for (int c = 0; c < 64; c += 8) {
            float p[8];
            for(int i = 0; i < 8; ++i) {
                if (n_block * 64 + c + i >= S) {
                    p[i] = 0.0f;
                } else {
                    p[i] = fast_exp2f_fn((P_reg[c+i] - m_new) * 1.4426950408889634f);
                }
                row_sum_new += p[i];
            }
            
            uint32_t p01 = pack_bf16_fn(p[0], p[1]);
            uint32_t p23 = pack_bf16_fn(p[2], p[3]);
            uint32_t p45 = pack_bf16_fn(p[4], p[5]);
            uint32_t p67 = pack_bf16_fn(p[6], p[7]);
            
            uint32_t offset = tid * 64 + c;
            uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared((char*)smem.P + offset * 2);
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                         :: "r"(smem_addr), "r"(p01), "r"(p23), "r"(p45), "r"(p67) : "memory");
        }
        l_i += row_sum_new;
        m_i = m_new;

        for (int i = 0; i < 128; ++i) {
            O_reg[i] *= exp_diff;
        }

        fence_async_shared_fn(); 
        __syncthreads();

        if (tid == 0) {
            tcgen05_fence_after_fn();
            for (int k = 0; k < 64; k += 16) {
                uint64_t d_P = make_smem_desc_sm100_none((char*)smem.P + k * 2, 2048, 128); 
                uint64_t d_V = make_smem_desc_sm100_none((char*)V_curr + k * 256, 128, 1024); 
                uint32_t idesc = make_instr_desc_custom_fn(128, 128, 0, 1); 
                umma_f16_cg1_fn(tmem_O, d_P, d_V, idesc, (k == 0) ? 0 : 1);
            }
            umma_commit_cg1_fn(smem.bar_mma);
        }
        mbarrier_wait_fn(smem.bar_mma, mma_phase);
        __syncthreads();

        for (int c = 0; c < 128; c += 8) {
            uint32_t r[8];
            tmem_load_8x_fn(tmem_O + c, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
            tmem_load_fence_fn();
            for (int i = 0; i < 8; ++i) {
                O_reg[c + i] += __uint_as_float(r[i]);
            }
        }

        mma_phase ^= 1;
        db ^= 1;
    }

    if (tid < 32) {
        tmem_dealloc_cg1_fn(smem.tmem_addr, 256);
    }

    if (m_block * 128 + tid < S) {
        lse_ptr[tid] = m_i + logf(l_i);
    }

    for (int i = 0; i < 128; ++i) {
        O_reg[i] /= l_i;
    }

    for (int i = 0; i < 128; i += 8) {
        uint32_t p01 = pack_bf16_fn(O_reg[i], O_reg[i+1]);
        uint32_t p23 = pack_bf16_fn(O_reg[i+2], O_reg[i+3]);
        uint32_t p45 = pack_bf16_fn(O_reg[i+4], O_reg[i+5]);
        uint32_t p67 = pack_bf16_fn(O_reg[i+6], O_reg[i+7]);
        
        uint32_t offset = tid * 128 + i;
        uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared((char*)smem.Q + offset * 2);
        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                     :: "r"(smem_addr), "r"(p01), "r"(p23), "r"(p45), "r"(p67) : "memory");
    }
    __syncthreads();
    tma_store_fence_fn();

    if (tid == 0) {
        if (m_block * 128 < S) {
            tma_store_2d_cta(&tma_O, smem.Q, 0, coord1_q);
            tma_store_commit_fn();
            tma_store_wait_fn<0>();
        }
    }
    __syncthreads();
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int D = Q.size(3);

    const __nv_bfloat16* q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* k_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* v_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_ptr = static_cast<float*>(LSE.data_ptr());

    CUtensorMap tma_Q, tma_K, tma_V, tma_O;
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_Q, (void*)q_ptr, D, B*H*S, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_K, (void*)k_ptr, D, B*H*S, 128, 64, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_V, (void*)v_ptr, D, B*H*S, 128, 64, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_O, (void*)o_ptr, D, B*H*S, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    dim3 grid((S + 127) / 128, H, B);
    dim3 block(128); 
    size_t smem_bytes = sizeof(SharedStorage);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUDA_CHECK(cudaFuncSetAttribute(mha_fwd_sm100_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    mha_fwd_sm100_kernel<<<grid, block, smem_bytes, stream>>>(tma_Q, tma_K, tma_V, tma_O, lse_ptr, B, H, S, D);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda