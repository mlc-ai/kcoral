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

namespace mha_with_lse_d128 {

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void st_shared_128_fn(uint32_t addr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "r"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
}

__device__ __forceinline__ uint32_t pack_bf16_fn(uint32_t fp32_a, uint32_t fp32_b) {
    __nv_bfloat16 a = __float2bfloat16(__uint_as_float(fp32_a));
    __nv_bfloat16 b = __float2bfloat16(__float_as_float(fp32_b));
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

__device__ __forceinline__ void named_barrier_sync_fn(int bar_id, int count) {
    asm volatile("barrier.sync.aligned %0, %1;" :: "r"(bar_id), "r"(count));
}

__device__ __forceinline__ void tma_load_4d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
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

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void umma_commit_cg1_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        :: "r"(a));
}

__device__ __forceinline__ void umma_f16_cg1_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, int accum) {
    if (accum) {
        asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 1;\n"
            :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc));
    } else {
        asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 0;\n"
            :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc));
    }
}

__device__ __forceinline__ uint32_t make_idesc(int m_dim, int n_dim, bool a_k_major, bool b_k_major) {
    uint32_t d = 0;
    d |= (1u<<4);  
    d |= (1u<<7);  
    d |= (1u<<10); 
    d |= ((uint32_t)(a_k_major ? 0 : 1) << 15);
    d |= ((uint32_t)(b_k_major ? 0 : 1) << 16);
    d |= ((uint32_t)(n_dim / 8) << 17);
    d |= ((uint32_t)(m_dim / 16) << 24);
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_unswizzled(void* ptr, bool is_k_major, uint32_t stride_bytes) {
    uint32_t lbo = is_k_major ? 16 : 8 * stride_bytes;
    uint32_t sbo = is_k_major ? 8 * stride_bytes : 16;
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16; 
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32; 
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)0 << 61; 
    return d;
}

CUresult create_tma_4d_descriptor_2B(CUtensorMap* d, void* globalAddress, 
    uint64_t dim0, uint64_t dim1, uint64_t dim2, uint64_t dim3,
    uint32_t box0, uint32_t box1, 
    CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[4] = {dim0, dim1, dim2, dim3};
    cuuint64_t globalStrides[3] = {dim0*2, dim0*dim1*2, dim0*dim1*dim2*2};
    cuuint32_t boxDim[4] = {box0, box1, 1, 1};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, globalAddress, globalDim, globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

extern __shared__ __align__(1024) uint8_t smem_dyn[];

__global__ void __launch_bounds__(128, 1) mha_fwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* __restrict__ O_global,
    float* __restrict__ LSE_global,
    int seqlen, int H
) {
    setmaxnreg_inc_sync_fn<256>();

    uint8_t* smem = (uint8_t*)(((uintptr_t)smem_dyn + 1023) & ~1023);

    int q_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int batch_idx = blockIdx.z;

    uint32_t smem_offset = 0;
    
    void* smem_Q = smem + smem_offset; smem_offset += 32768;
    
    void* smem_K[2];
    void* smem_V[2];
    for(int i=0; i<2; i++) {
        smem_K[i] = smem + smem_offset; smem_offset += 32768;
        smem_V[i] = smem + smem_offset; smem_offset += 32768;
    }
    
    void* smem_P = smem + smem_offset; smem_offset += 32768;

    uint64_t* mbar_K = (uint64_t*)(smem + smem_offset); smem_offset += 2 * 8;
    uint64_t* mbar_V = (uint64_t*)(smem + smem_offset); smem_offset += 2 * 8;
    uint64_t* mbar_Q = (uint64_t*)(smem + smem_offset); smem_offset += 8;
    uint64_t* mbar_UMMA = (uint64_t*)(smem + smem_offset); smem_offset += 8;
    uint32_t* smem_tmem_base = (uint32_t*)(smem + smem_offset); smem_offset += 8;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&mbar_K[0], 1);
        init_smem_barrier_fn(&mbar_K[1], 1);
        init_smem_barrier_fn(&mbar_V[0], 1);
        init_smem_barrier_fn(&mbar_V[1], 1);
        init_smem_barrier_fn(&mbar_Q[0], 1);
        init_smem_barrier_fn(&mbar_UMMA[0], 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        tmem_alloc_cg1_fn(smem_tmem_base, 256);
    }
    __syncthreads();
    uint32_t tmem_base = *smem_tmem_base;

    int k_iters = (seqlen + 127) / 128;

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&mbar_Q[0], 32768);
        tma_load_4d_fn(&tma_Q, &mbar_Q[0], smem_Q, 0, q_idx * 128, head_idx, batch_idx);
        
        if (k_iters > 0) {
            mbarrier_arrive_and_expect_tx_fn(&mbar_K[0], 32768);
            tma_load_4d_fn(&tma_K, &mbar_K[0], smem_K[0], 0, 0, head_idx, batch_idx);
            
            mbarrier_arrive_and_expect_tx_fn(&mbar_V[0], 32768);
            tma_load_4d_fn(&tma_V, &mbar_V[0], smem_V[0], 0, 0, head_idx, batch_idx);
        }
    }
    
    mbarrier_wait_fn(&mbar_Q[0], 0);

    uint32_t idesc_S = make_idesc(128, 128, true, true);
    uint32_t idesc_O = make_idesc(128, 128, true, false);

    float O_acc[128];
    for(int i=0; i<128; i++) O_acc[i] = 0.0f;
    float m_val = -50000.0f;
    float d_val = 0.0f;
    float scale = 0.0883883476f;

    int umma_phase = 0;

    for(int k=0; k<k_iters; k++) {
        int buf = k % 2;
        int next_buf = (k + 1) % 2;
        
        if (k + 1 < k_iters) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&mbar_K[next_buf], 32768);
                tma_load_4d_fn(&tma_K, &mbar_K[next_buf], smem_K[next_buf], 0, (k+1)*128, head_idx, batch_idx);
                
                mbarrier_arrive_and_expect_tx_fn(&mbar_V[next_buf], 32768);
                tma_load_4d_fn(&tma_V, &mbar_V[next_buf], smem_V[next_buf], 0, (k+1)*128, head_idx, batch_idx);
            }
        }
        
        mbarrier_wait_fn(&mbar_K[buf], (k / 2) % 2);
        
        if (threadIdx.x == 0) {
            tcgen05_fence_after_fn();
            for (int step = 0; step < 128; step += 16) {
                uint64_t desc_Q = make_smem_desc_unswizzled((uint8_t*)smem_Q + step * 2, true, 256);
                uint64_t desc_K = make_smem_desc_unswizzled((uint8_t*)smem_K[buf] + step * 2, true, 256);
                umma_f16_cg1_fn(tmem_base, desc_Q, desc_K, idesc_S, step == 0 ? 0 : 1);
            }
            umma_commit_cg1_fn(&mbar_UMMA[0]);
        }
        
        mbarrier_wait_fn(&mbar_UMMA[0], umma_phase % 2);
        umma_phase++;
        
        float m_local = -50000.0f;
        for(int i=0; i<16; i++) {
            uint32_t r[8];
            tmem_load_8x_fn(tmem_base + i*8, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
            tmem_load_fence_fn();
            for(int j=0; j<8; j++) {
                int col = i*8+j;
                if (k * 128 + col < seqlen) {
                    float val = __uint_as_float(r[j]) * scale;
                    m_local = fmaxf(m_local, val);
                }
            }
        }
        
        float m_new = fmaxf(m_val, m_local);
        float exp_diff = fast_exp2f_fn((m_val - m_new) * 1.44269504f);
        d_val *= exp_diff;
        for(int i=0; i<128; i++) O_acc[i] *= exp_diff;
        m_val = m_new;
        
        float d_local = 0.0f;
        for(int i=0; i<16; i+=2) {
            uint32_t r0[8], r1[8];
            tmem_load_8x_fn(tmem_base + i*8, &r0[0], &r0[1], &r0[2], &r0[3], &r0[4], &r0[5], &r0[6], &r0[7]);
            tmem_load_8x_fn(tmem_base + i*8 + 8, &r1[0], &r1[1], &r1[2], &r1[3], &r1[4], &r1[5], &r1[6], &r1[7]);
            tmem_load_fence_fn();
            
            uint32_t packed[8];
            for(int j=0; j<4; j++) {
                int col0 = i*8 + j*2;
                float v0 = (k * 128 + col0 < seqlen) ? fast_exp2f_fn((__uint_as_float(r0[j*2]) * scale - m_val) * 1.44269504f) : 0.0f;
                float v1 = (k * 128 + col0 + 1 < seqlen) ? fast_exp2f_fn((__uint_as_float(r0[j*2+1]) * scale - m_val) * 1.44269504f) : 0.0f;
                d_local += v0 + v1;
                packed[j] = pack_bf16_fn(__float_as_uint(v0), __float_as_uint(v1));
                
                int col1 = i*8 + 8 + j*2;
                float v2 = (k * 128 + col1 < seqlen) ? fast_exp2f_fn((__uint_as_float(r1[j*2]) * scale - m_val) * 1.44269504f) : 0.0f;
                float v3 = (k * 128 + col1 + 1 < seqlen) ? fast_exp2f_fn((__uint_as_float(r1[j*2+1]) * scale - m_val) * 1.44269504f) : 0.0f;
                d_local += v2 + v3;
                packed[4+j] = pack_bf16_fn(__float_as_uint(v2), __float_as_uint(v3));
            }
            
            uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_P) + threadIdx.x * 256 + i * 8 * 2;
            st_shared_128_fn(addr, packed[0], packed[1], packed[2], packed[3]);
            st_shared_128_fn(addr + 16, packed[4], packed[5], packed[6], packed[7]);
        }
        d_val += d_local;
        
        fence_async_shared_fn();
        named_barrier_sync_fn(0, 128); 
        
        mbarrier_wait_fn(&mbar_V[buf], (k / 2) % 2);
        
        if (threadIdx.x == 0) {
            tcgen05_fence_after_fn();
            for (int step = 0; step < 128; step += 16) {
                uint64_t desc_P = make_smem_desc_unswizzled((uint8_t*)smem_P + step * 2, true, 256);
                uint64_t desc_V = make_smem_desc_unswizzled((uint8_t*)smem_V[buf] + step * 256, false, 256);
                
                umma_f16_cg1_fn(tmem_base + 128, desc_P, desc_V, idesc_O, step == 0 ? 0 : 1);
            }
            umma_commit_cg1_fn(&mbar_UMMA[0]);
        }
        
        mbarrier_wait_fn(&mbar_UMMA[0], umma_phase % 2);
        umma_phase++;
        
        for(int i=0; i<16; i++) {
            uint32_t r[8];
            tmem_load_8x_fn(tmem_base + 128 + i*8, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
            tmem_load_fence_fn();
            for(int j=0; j<8; j++) {
                O_acc[i*8+j] += __uint_as_float(r[j]);
            }
        }
    }

    int row = q_idx * 128 + threadIdx.x;
    if (row < seqlen) {
        float out_scale = 1.0f / d_val;
        for(int i=0; i<128; i++) O_acc[i] *= out_scale;
        
        LSE_global[((uint64_t)batch_idx * H + head_idx) * seqlen + row] = m_val + logf(d_val);
    }
    
    __syncthreads();
    
    __nv_bfloat16* smem_O = (__nv_bfloat16*)smem_Q;
    for(int i=0; i<128; i+=2) {
        uint32_t packed = pack_bf16_fn(__float_as_uint(O_acc[i]), __float_as_uint(O_acc[i+1]));
        int smem_idx = threadIdx.x * 128 + i;
        *(uint32_t*)((uint8_t*)smem_O + smem_idx * 2) = packed;
    }
    __syncthreads();
    
    uint32_t* smem_O_u32 = (uint32_t*)smem_O;
    for(int i=0; i<64; i++) {
        int smem_idx = i * 128 + threadIdx.x;
        int row_o = smem_idx / 64; 
        int col_o = smem_idx % 64; 
        
        int global_row = q_idx * 128 + row_o;
        if (global_row < seqlen) {
            uint64_t global_idx = ((uint64_t)batch_idx * H * seqlen + (uint64_t)head_idx * seqlen + global_row) * 64 + col_o;
            ((uint32_t*)O_global)[global_idx] = smem_O_u32[smem_idx];
        }
    }
    
    __syncthreads();
    if (threadIdx.x < 32) {
        tmem_dealloc_cg1_fn(tmem_base, 256);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    CUtensorMap tma_Q, tma_K, tma_V;
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_Q, Q.data_ptr(), 128, S, H, B, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_K, K.data_ptr(), 128, S, H, B, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_V, V.data_ptr(), 128, S, H, B, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE));
    
    int q_chunks = (S + 127) / 128;
    dim3 grid(q_chunks, H, B);
    dim3 block(128);
    
    int smem_size = 200 * 1024;
    CUDA_CHECK(cudaFuncSetAttribute(mha_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    mha_fwd_kernel<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, 
        static_cast<__nv_bfloat16*>(O.data_ptr()), 
        static_cast<float*>(LSE.data_ptr()), 
        S, H
    );
    CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}