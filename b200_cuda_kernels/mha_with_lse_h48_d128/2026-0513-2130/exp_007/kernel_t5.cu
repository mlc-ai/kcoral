#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

#define CU_CHECK(call) do {                                        \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        const char* msg;                                           \
        cuGetErrorString(_e, &msg);                                \
        fprintf(stderr, "CU error %s at %s:%d\n",                  \
                msg, __FILE__, __LINE__);                          \
        exit(1);                                                   \
    }                                                              \
} while(0)

__device__ __forceinline__ void tmem_alloc_cg1_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_cg1_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(addr), "r"(ncols));
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

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
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

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_k_major_swizzle(void* smem_ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr) >> 4;
    d |= (uint64_t)(addr & 0x3FFF);
    d |= (uint64_t)(1 << 16);           // LBO = 16
    d |= (uint64_t)(1024 >> 4) << 32;   // SBO = 1024 (8 rows * 128 bytes)
    d |= (uint64_t)1 << 46;             // version = 1
    d |= (uint64_t)2 << 61;             // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_mn_major_swizzle(void* smem_ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr) >> 4;
    d |= (uint64_t)(addr & 0x3FFF);
    d |= (uint64_t)(1024 >> 4) << 16;   // LBO = 1024 (8 rows * 128 bytes)
    d |= (uint64_t)(16 >> 4) << 32;     // SBO = 16 (8 cols * 2 bytes)
    d |= (uint64_t)1 << 46;             // version = 1
    d |= (uint64_t)2 << 61;             // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc(uint32_t M, uint32_t N, int trans_b) {
    uint32_t d = 0;
    d |= (1u << 4);           // c_format = FP32
    d |= (1u << 7);           // a_format = BF16
    d |= (1u << 10);          // b_format = BF16
    if (trans_b) d |= (1u << 16); // Transpose B
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void tmem_store_4x_fn(uint32_t col, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
                 :: "r"(col), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
}

__device__ __forceinline__ uint32_t pack_bf16_fn(uint32_t fp32_a, uint32_t fp32_b) {
    __nv_bfloat16 a = __float2bfloat16(__uint_as_float(fp32_a));
    __nv_bfloat16 b = __float2bfloat16(__uint_as_float(fp32_b));
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};"
        : "=r"(result)
        : "h"(*reinterpret_cast<uint16_t*>(&a)),
          "h"(*reinterpret_cast<uint16_t*>(&b)));
    return result;
}

extern __shared__ __align__(128) char smem_buffer[];

__global__ void __launch_bounds__(128) mha_fwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    void* __restrict__ O_data,
    void* __restrict__ LSE_data,
    int S) 
{
    int b_idx = blockIdx.z;
    int h_idx = blockIdx.y;
    int s_q_start = blockIdx.x * 128;

    if (s_q_start >= S) return;

    // Buffer chunking
    __nv_bfloat16* smem_Q_left  = (__nv_bfloat16*)smem_buffer;                     
    __nv_bfloat16* smem_Q_right = smem_Q_left + 8192; // 128 * 64 elements
    __nv_bfloat16* smem_K_left  = smem_Q_right + 8192;
    __nv_bfloat16* smem_K_right = smem_K_left + 8192;
    __nv_bfloat16* smem_V_left  = smem_K_right + 8192;
    __nv_bfloat16* smem_V_right = smem_V_left + 8192;
    __nv_bfloat16* smem_P_left  = smem_V_right + 8192;
    __nv_bfloat16* smem_P_right = smem_P_left + 8192;
    
    uint64_t* mbar_Q = (uint64_t*)(smem_P_right + 8192);            
    uint64_t* mbar_K = mbar_Q + 1;                                 
    uint64_t* mbar_V = mbar_K + 1;                                 
    uint64_t* mbar_mma = mbar_V + 1;                               
    uint32_t* smem_tmem_ptr = (uint32_t*)(mbar_mma + 1);           

    if (threadIdx.x < 32) {
        tmem_alloc_cg1_fn(smem_tmem_ptr, 256);
    }
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(mbar_K, 1);
        init_smem_barrier_fn(mbar_V, 1);
        init_smem_barrier_fn(mbar_mma, 1);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();

    uint32_t tmem_base = *smem_tmem_ptr;
    uint32_t tmem_S = tmem_base;
    uint32_t tmem_O = tmem_base + 128;

    // Initialize tmem_O directly to 0
    for (uint32_t col = 0; col < 128; col += 4) {
        uint32_t zero = 0;
        asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
                     :: "r"(tmem_O + col), "r"(zero), "r"(zero), "r"(zero), "r"(zero) : "memory");
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    __syncthreads();

    uint32_t tx_bytes = 128 * 64 * 2 * 2; // Two 128x64 blocks total size logic
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, tx_bytes);
        tma_load_4d_fn(&tma_Q, mbar_Q, smem_Q_left,  0,  s_q_start, h_idx, b_idx);
        tma_load_4d_fn(&tma_Q, mbar_Q, smem_Q_right, 64, s_q_start, h_idx, b_idx);
    }
    mbarrier_wait_fn(mbar_Q, 0);

    float m_prev = -1e20f;
    float l_prev = 0.0f;
    int phase = 0;
    int mma_phase = 0;
    
    uint32_t idesc_qk = make_instr_desc(128, 128, 0); // K-major
    uint32_t idesc_pv_64 = make_instr_desc(128, 64, 1); // MN-major
    float scale_qk = 0.08838834764f; // 1 / sqrt(128)

    int tid = threadIdx.x; 
    int xor_mask = (tid % 8) * 16;
    int row_offset = tid * 128; 

    for (int s_k = 0; s_k < S; s_k += 128) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_K, tx_bytes);
            tma_load_4d_fn(&tma_K, mbar_K, smem_K_left,  0,  s_k, h_idx, b_idx);
            tma_load_4d_fn(&tma_K, mbar_K, smem_K_right, 64, s_k, h_idx, b_idx);
            
            mbarrier_arrive_and_expect_tx_fn(mbar_V, tx_bytes);
            tma_load_4d_fn(&tma_V, mbar_V, smem_V_left,  0,  s_k, h_idx, b_idx);
            tma_load_4d_fn(&tma_V, mbar_V, smem_V_right, 64, s_k, h_idx, b_idx);
        }
        mbarrier_wait_fn(mbar_K, phase);
        mbarrier_wait_fn(mbar_V, phase);
        
        tcgen05_fence_after_fn(); 
        
        if (threadIdx.x == 0) {
            for (int k = 0; k < 128; k += 16) {
                void* ptr_q = (k < 64) ? ((char*)smem_Q_left + k * 2) : ((char*)smem_Q_right + (k - 64) * 2);
                void* ptr_k = (k < 64) ? ((char*)smem_K_left + k * 2) : ((char*)smem_K_right + (k - 64) * 2);
                
                uint64_t desc_q = make_smem_desc_k_major_swizzle(ptr_q);
                uint64_t desc_k = make_smem_desc_k_major_swizzle(ptr_k);
                uint32_t accum = (k == 0) ? 0 : 1;
                asm volatile(
                    "{\n.reg .pred p;\n"
                    "setp.ne.b32 p, %4, 0;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                    :: "r"(tmem_S), "l"(desc_q), "l"(desc_k), "r"(idesc_qk), "r"(accum));
            }
            uint32_t a = (uint32_t)__cvta_generic_to_shared(mbar_mma);
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(a));
        }
        
        // Wait correctly for QK MMA using parity tracking
        mbarrier_wait_fn(mbar_mma, mma_phase);
        mma_phase ^= 1;
        
        float row_S[128];
        for (int col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_S + col));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            row_S[col]   = __uint_as_float(r0) * scale_qk;
            row_S[col+1] = __uint_as_float(r1) * scale_qk;
            row_S[col+2] = __uint_as_float(r2) * scale_qk;
            row_S[col+3] = __uint_as_float(r3) * scale_qk;
        }
        
        for (int col = 0; col < 128; ++col) {
            if (s_k + col >= S || s_q_start + tid >= S) {
                row_S[col] = -1e20f;
            }
        }
        
        float m_new = m_prev;
        for (int col = 0; col < 128; ++col) {
            m_new = fmaxf(m_new, row_S[col]);
        }
        
        float scale_O = expf(m_prev - m_new);
        
        int needs_scale = (scale_O < 1.0f) ? 1 : 0;
        int block_needs_scale = __syncthreads_or(needs_scale);
        if (block_needs_scale) {
            for (int col = 0; col < 128; col += 4) {
                uint32_t r0, r1, r2, r3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                             : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_O + col));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float f0 = __uint_as_float(r0) * scale_O;
                float f1 = __uint_as_float(r1) * scale_O;
                float f2 = __uint_as_float(r2) * scale_O;
                float f3 = __uint_as_float(r3) * scale_O;
                tmem_store_4x_fn(tmem_O + col, __float_as_uint(f0), __float_as_uint(f1), __float_as_uint(f2), __float_as_uint(f3));
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        }
        __syncthreads();
        
        float sum_P = 0.0f;
        for (int col = 0; col < 128; ++col) {
            float p = fast_exp2f_fn((row_S[col] - m_new) * 1.4426950408889634f);
            sum_P += p;
            __nv_bfloat16 p_bf16 = __float2bfloat16(p);
            
            int c = (col < 64) ? col : (col - 64);
            __nv_bfloat16* base = (col < 64) ? smem_P_left : smem_P_right;
            
            int linear_byte = row_offset + c * 2;
            int swizzled_byte = (linear_byte & ~0x7f) | ((linear_byte & 0x7f) ^ xor_mask);
            *(__nv_bfloat16*)((char*)base + swizzled_byte) = p_bf16;
        }
        
        l_prev = l_prev * scale_O + sum_P;
        m_prev = m_new;
        
        __syncthreads();
        fence_async_shared_fn();
        tcgen05_fence_after_fn();
        
        if (threadIdx.x == 0) {
            for (int k = 0; k < 128; k += 16) {
                void* ptr_p = (k < 64) ? ((char*)smem_P_left + k * 2) : ((char*)smem_P_right + (k - 64) * 2);
                uint64_t desc_p = make_smem_desc_k_major_swizzle(ptr_p);
                
                void* ptr_v_left = (char*)smem_V_left + k * 128;
                uint64_t desc_v_left = make_smem_desc_mn_major_swizzle(ptr_v_left);
                uint32_t accum = 1;
                asm volatile(
                    "{\n.reg .pred p;\n"
                    "setp.ne.b32 p, %4, 0;\n" 
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                    :: "r"(tmem_O), "l"(desc_p), "l"(desc_v_left), "r"(idesc_pv_64), "r"(accum));
                    
                void* ptr_v_right = (char*)smem_V_right + k * 128;
                uint64_t desc_v_right = make_smem_desc_mn_major_swizzle(ptr_v_right);
                asm volatile(
                    "{\n.reg .pred p;\n"
                    "setp.ne.b32 p, %4, 0;\n" 
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                    :: "r"(tmem_O + 64), "l"(desc_p), "l"(desc_v_right), "r"(idesc_pv_64), "r"(accum));
            }
            uint32_t a = (uint32_t)__cvta_generic_to_shared(mbar_mma);
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(a));
        }
        
        // Wait correctly for PV MMA using parity tracking
        mbarrier_wait_fn(mbar_mma, mma_phase);
        mma_phase ^= 1;
        
        __syncthreads();
        phase ^= 1;
    }

    float scale_final = 1.0f / l_prev;
    
    // Process completely uniform to avoid collective thread-divergence undefined behavior 
    for (int col = 0; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_O + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        if (s_q_start + tid < S) {
            float f0 = __uint_as_float(r0) * scale_final;
            float f1 = __uint_as_float(r1) * scale_final;
            float f2 = __uint_as_float(r2) * scale_final;
            float f3 = __uint_as_float(r3) * scale_final;
            
            uint64_t out_idx = (uint64_t)b_idx * (gridDim.y * S * 128) + 
                               (uint64_t)h_idx * (S * 128) + 
                               (uint64_t)(s_q_start + tid) * 128 + col;
            
            __nv_bfloat16* out_ptr = (__nv_bfloat16*)O_data + out_idx;
            uint32_t p0 = pack_bf16_fn(__float_as_uint(f0), __float_as_uint(f1));
            uint32_t p1 = pack_bf16_fn(__float_as_uint(f2), __float_as_uint(f3));
            uint2 p; p.x = p0; p.y = p1;
            *(uint2*)out_ptr = p;
        }
    }
    
    if (s_q_start + tid < S) {
        float lse = m_prev + logf(l_prev);
        uint64_t lse_idx = (uint64_t)b_idx * (gridDim.y * S) + (uint64_t)h_idx * S + (s_q_start + tid);
        ((float*)LSE_data)[lse_idx] = lse;
    }

    __syncthreads();
    if (threadIdx.x < 32) {
        tmem_dealloc_cg1_fn(tmem_base, 256);
    }
}

namespace tvm_ffi_example_cuda {

CUresult create_tma_4d_descriptor_swizzle(CUtensorMap* d, void* globalAddress, 
    uint64_t dim0, uint64_t dim1, uint64_t dim2, uint64_t dim3) {
    
    cuuint64_t globalDim[4] = {dim0, dim1, dim2, dim3};
    cuuint64_t globalStrides[3] = {
        dim0 * 2,
        dim0 * dim1 * 2,
        dim0 * dim1 * dim2 * 2
    };
    cuuint32_t boxDim[4] = {64, 128, 1, 1}; 
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
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int D = Q.size(3); 
    
    CUtensorMap tma_Q, tma_K, tma_V;
    CU_CHECK(create_tma_4d_descriptor_swizzle(&tma_Q, Q.data_ptr(), D, S, H, B));
    CU_CHECK(create_tma_4d_descriptor_swizzle(&tma_K, K.data_ptr(), D, S, H, B));
    CU_CHECK(create_tma_4d_descriptor_swizzle(&tma_V, V.data_ptr(), D, S, H, B));
    
    dim3 grid((S + 127) / 128, H, B);
    dim3 block(128);
    
    int smem_bytes = 129 * 1024; 
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUDA_CHECK(cudaFuncSetAttribute(mha_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    
    mha_fwd_kernel<<<grid, block, smem_bytes, stream>>>(
        tma_Q, tma_K, tma_V, O.data_ptr(), LSE.data_ptr(), S
    );
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}