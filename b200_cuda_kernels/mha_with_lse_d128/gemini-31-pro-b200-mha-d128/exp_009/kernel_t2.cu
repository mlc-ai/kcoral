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

__device__ __forceinline__ void tma_store_commit_fn() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void tma_store_wait_fn() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
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

__device__ __forceinline__ uint64_t make_smem_desc_sm100_swizzled(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint32_t base_offset = (addr >> 7) & 0x7;
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)base_offset << 49;
    d |= (uint64_t)2 << 61;
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
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2, 
        globalAddress,
        globalDim,
        globalStrides,
        boxDim, 
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        l2Promotion,
        oobFill
    );
}

struct SharedStorage {
    alignas(1024) __nv_bfloat16 Q0[128 * 64]; 
    alignas(1024) __nv_bfloat16 Q1[128 * 64]; 
    alignas(1024) __nv_bfloat16 K0[64 * 64];  
    alignas(1024) __nv_bfloat16 K1[64 * 64];  
    alignas(1024) __nv_bfloat16 V0[64 * 64];  
    alignas(1024) __nv_bfloat16 V1[64 * 64];  
    alignas(1024) __nv_bfloat16 P[128 * 64];  
    alignas(8) uint64_t bar_mma[1];
    alignas(8) uint64_t bar_tma[1];
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

    if (tid < 32) {
        if (tid == 0) {
            init_smem_barrier_fn(smem.bar_tma, 1);
            mbarrier_arrive_and_expect_tx_fn(smem.bar_tma, 32768);
            tma_load_2d_cta(&tma_Q, smem.bar_tma, smem.Q0, 0, coord1_q);
            tma_load_2d_cta(&tma_Q, smem.bar_tma, smem.Q1, 64, coord1_q);
            
            init_smem_barrier_fn(smem.bar_mma, 1);
        }
        tmem_alloc_cg1_fn(&smem.tmem_addr, 128); // 128 columns
    }
    
    __syncthreads();
    mbarrier_wait_fn(smem.bar_tma, 0); 

    uint32_t tmem_S = smem.tmem_addr;       
    uint32_t tmem_O0 = smem.tmem_addr;      
    uint32_t tmem_O1 = smem.tmem_addr + 64; 
    
    uint32_t mma_phase = 0;
    int phase_tma = 1;

    float m_i = -INFINITY;
    float l_i = 0.0f;
    float O_reg[128] = {0};
    float P_reg[64];

    int num_n_blocks = (S + 63) / 64;
    for(int n_block = 0; n_block < num_n_blocks; ++n_block) {
        int coord1_kv = (b * H + h) * S + n_block * 64;
        
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(smem.bar_tma, 32768);
            tma_load_2d_cta(&tma_K, smem.bar_tma, smem.K0, 0, coord1_kv);
            tma_load_2d_cta(&tma_K, smem.bar_tma, smem.K1, 64, coord1_kv);
            tma_load_2d_cta(&tma_V, smem.bar_tma, smem.V0, 0, coord1_kv);
            tma_load_2d_cta(&tma_V, smem.bar_tma, smem.V1, 64, coord1_kv);
        }
        __syncthreads();
        mbarrier_wait_fn(smem.bar_tma, phase_tma);
        
        if (tid == 0) {
            for (int k = 0; k < 64; k += 16) {
                uint64_t d_Q0 = make_smem_desc_sm100_swizzled((char*)smem.Q0 + k * 2, 1, 1024);
                uint64_t d_K0 = make_smem_desc_sm100_swizzled((char*)smem.K0 + k * 2, 1, 1024);
                uint32_t idesc0 = make_instr_desc_custom_fn(128, 64, 0, 0);
                umma_f16_cg1_fn(tmem_S, d_Q0, d_K0, idesc0, (k == 0) ? 0 : 1);
            }
            for (int k = 0; k < 64; k += 16) {
                uint64_t d_Q1 = make_smem_desc_sm100_swizzled((char*)smem.Q1 + k * 2, 1, 1024);
                uint64_t d_K1 = make_smem_desc_sm100_swizzled((char*)smem.K1 + k * 2, 1, 1024);
                uint32_t idesc1 = make_instr_desc_custom_fn(128, 64, 0, 0);
                umma_f16_cg1_fn(tmem_S, d_Q1, d_K1, idesc1, 1); 
            }
            umma_commit_cg1_fn(smem.bar_mma);
        }
        mbarrier_wait_fn(smem.bar_mma, mma_phase);
        __syncthreads();
        mma_phase ^= 1;

        float row_max = -INFINITY;
        for(int c = 0; c < 64; c += 8) {
            uint32_t r[8];
            tmem_load_8x_fn(tmem_S + c, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
            tmem_load_fence_fn(); 
            for(int i = 0; i < 8; ++i) {
                float val = __uint_as_float(r[i]);
                val *= 0.08838834764f; 
                int global_col = n_block * 64 + c + i;
                if (global_col >= S) val = -INFINITY;
                P_reg[c + i] = val;
                row_max = max(row_max, val);
            }
        }

        float m_new = max(m_i, row_max);
        float exp_diff = expf(m_i - m_new);
        l_i = l_i * exp_diff;

        float row_sum_new = 0.0f;
        for(int c = 0; c < 64; ++c) {
            int global_col = n_block * 64 + c;
            float p = 0.0f;
            if (global_col < S) {
                p = fast_exp2f_fn((P_reg[c] - m_new) * 1.4426950408889634f); 
            }
            P_reg[c] = p;
            row_sum_new += p;
        }
        l_i += row_sum_new;
        m_i = m_new;

        for(int i = 0; i < 128; ++i) {
            O_reg[i] *= exp_diff;
        }

        for(int c = 0; c < 64; c += 8) {
            uint32_t p01 = pack_bf16_fn(P_reg[c], P_reg[c+1]);
            uint32_t p23 = pack_bf16_fn(P_reg[c+2], P_reg[c+3]);
            uint32_t p45 = pack_bf16_fn(P_reg[c+4], P_reg[c+5]);
            uint32_t p67 = pack_bf16_fn(P_reg[c+6], P_reg[c+7]);
            
            int c_16b = c / 8;
            int swizzled_c = (c_16b % 8) ^ (tid % 8);
            uint32_t offset = tid * 128 + swizzled_c * 16;
            
            uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared((char*)smem.P + offset);
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                         :: "r"(smem_addr), "r"(p01), "r"(p23), "r"(p45), "r"(p67) : "memory");
        }
        __syncthreads();
        fence_async_shared_fn();

        if (tid == 0) {
            for(int k = 0; k < 64; k += 16) {
                uint64_t d_P = make_smem_desc_sm100_swizzled((char*)smem.P + k * 2, 1, 1024);
                
                uint64_t d_V0 = make_smem_desc_sm100_swizzled((char*)smem.V0 + k * 128, 8192, 1024);
                uint32_t idesc0 = make_instr_desc_custom_fn(128, 64, 0, 1);
                umma_f16_cg1_fn(tmem_O0, d_P, d_V0, idesc0, (k == 0) ? 0 : 1);
                
                uint64_t d_V1 = make_smem_desc_sm100_swizzled((char*)smem.V1 + k * 128, 8192, 1024);
                uint32_t idesc1 = make_instr_desc_custom_fn(128, 64, 0, 1);
                umma_f16_cg1_fn(tmem_O1, d_P, d_V1, idesc1, (k == 0) ? 0 : 1);
            }
            umma_commit_cg1_fn(smem.bar_mma);
        }
        mbarrier_wait_fn(smem.bar_mma, mma_phase);
        __syncthreads();
        mma_phase ^= 1;

        for(int c = 0; c < 64; c += 8) {
            uint32_t r[8];
            tmem_load_8x_fn(tmem_O0 + c, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
            tmem_load_fence_fn();
            for(int i = 0; i < 8; ++i) {
                O_reg[c + i] += __uint_as_float(r[i]);
            }
        }
        for(int c = 0; c < 64; c += 8) {
            uint32_t r[8];
            tmem_load_8x_fn(tmem_O1 + c, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
            tmem_load_fence_fn();
            for(int i = 0; i < 8; ++i) {
                O_reg[64 + c + i] += __uint_as_float(r[i]);
            }
        }
        __syncthreads();
        phase_tma ^= 1;
    }

    if (tid < 32) {
        tmem_dealloc_cg1_fn(smem.tmem_addr, 128);
    }

    float lse = m_i + logf(l_i);
    if (m_block * 128 + tid < S) {
        lse_ptr[tid] = lse;
    }

    for(int i = 0; i < 128; ++i) {
        O_reg[i] /= l_i;
    }

    for(int c = 0; c < 64; c += 8) {
        uint32_t p01 = pack_bf16_fn(O_reg[c], O_reg[c+1]);
        uint32_t p23 = pack_bf16_fn(O_reg[c+2], O_reg[c+3]);
        uint32_t p45 = pack_bf16_fn(O_reg[c+4], O_reg[c+5]);
        uint32_t p67 = pack_bf16_fn(O_reg[c+6], O_reg[c+7]);
        
        int c_16b = c / 8;
        int swizzled_c = (c_16b % 8) ^ (tid % 8);
        uint32_t offset = tid * 128 + swizzled_c * 16;
        
        uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared((char*)smem.Q0 + offset);
        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                     :: "r"(smem_addr), "r"(p01), "r"(p23), "r"(p45), "r"(p67) : "memory");
    }
    for(int c = 0; c < 64; c += 8) {
        uint32_t p01 = pack_bf16_fn(O_reg[64+c], O_reg[64+c+1]);
        uint32_t p23 = pack_bf16_fn(O_reg[64+c+2], O_reg[64+c+3]);
        uint32_t p45 = pack_bf16_fn(O_reg[64+c+4], O_reg[64+c+5]);
        uint32_t p67 = pack_bf16_fn(O_reg[64+c+6], O_reg[64+c+7]);
        
        int c_16b = c / 8;
        int swizzled_c = (c_16b % 8) ^ (tid % 8);
        uint32_t offset = tid * 128 + swizzled_c * 16;
        
        uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared((char*)smem.Q1 + offset);
        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                     :: "r"(smem_addr), "r"(p01), "r"(p23), "r"(p45), "r"(p67) : "memory");
    }
    __syncthreads();
    tma_store_fence_fn();

    if (tid == 0) {
        tma_store_2d_cta(&tma_O, smem.Q0, 0, coord1_q);
        tma_store_2d_cta(&tma_O, smem.Q1, 64, coord1_q);
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
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
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_Q, (void*)q_ptr, D, B*H*S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_K, (void*)k_ptr, D, B*H*S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_V, (void*)v_ptr, D, B*H*S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_O, (void*)o_ptr, D, B*H*S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

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