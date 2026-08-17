#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>
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
                _e, __FILE__, __LINE__);                           \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_example_cuda {

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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

__device__ __forceinline__ void st_shared_128_fn(uint32_t addr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "r"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
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

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
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

__device__ __forceinline__ void tma_load_2d_cg1_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
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
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        :: "r"(a));
}

__device__ __forceinline__ uint64_t make_smem_desc(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint32_t base_offset = (addr >> 7) & 0x7;
    d |= (uint64_t)(addr >> 4) & 0x3FFF;
    d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16;
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32; 
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)base_offset << 49;
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint32_t make_idesc_QK() {
    uint32_t d = 0;
    d |= (1u << 4);           
    d |= (1u << 7);           
    d |= (1u << 10);          
    d |= (0u << 15);          
    d |= (0u << 16);          
    d |= ((128 / 8) << 17);   
    d |= ((128 / 16) << 24);  
    return d;
}

__device__ __forceinline__ uint32_t make_idesc_PV() {
    uint32_t d = 0;
    d |= (1u << 4);           
    d |= (1u << 7);           
    d |= (1u << 10);          
    d |= (0u << 15);          
    d |= (1u << 16);          
    d |= ((128 / 8) << 17);   
    d |= ((128 / 16) << 24);  
    return d;
}


__global__ void mha_fwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O_global, float* LSE_global,
    int B, int H, int S) 
{
    int b = blockIdx.z;
    int h = blockIdx.y;
    int i = blockIdx.x;
    if (i * 128 >= S) return;

    extern __shared__ char smem_buf[];
    __nv_bfloat16* Q_smem = (__nv_bfloat16*)(smem_buf);
    __nv_bfloat16* K_smem = (__nv_bfloat16*)(smem_buf + 32768);
    __nv_bfloat16* V_smem = (__nv_bfloat16*)(smem_buf + 65536);
    __nv_bfloat16* P_smem = (__nv_bfloat16*)(smem_buf + 98304);

    uint64_t* mbar_Q = (uint64_t*)(smem_buf + 131072);
    uint64_t* mbar_K = (uint64_t*)(smem_buf + 131080);
    uint64_t* mbar_V = (uint64_t*)(smem_buf + 131088);
    uint64_t* mbar_mma = (uint64_t*)(smem_buf + 131096);
    uint32_t* tmem_base_smem = (uint32_t*)(smem_buf + 131104);

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(mbar_K, 1);
        init_smem_barrier_fn(mbar_V, 1);
        init_smem_barrier_fn(mbar_mma, 1);
    }
    __syncthreads();

    uint32_t phase_Q = 0, phase_K = 0, phase_V = 0, phase_mma = 0;

    if (threadIdx.x < 32) {
        tmem_alloc_cg1_fn(tmem_base_smem, 256);
    }
    __syncthreads();
    uint32_t tmem_base = *tmem_base_smem;
    uint32_t S_tmem = tmem_base;
    uint32_t O_tmem = tmem_base + 128;

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 32768);
        tma_load_2d_cg1_fn(&tma_Q, mbar_Q, Q_smem, 0, b * H * S + h * S + i * 128);
    }
    mbarrier_wait_fn(mbar_Q, phase_Q);
    phase_Q ^= 1;

    float O_i[128];
    for(int c=0; c<128; c++) O_i[c] = 0.0f;
    float m_i = -INFINITY;
    float l_i = 0.0f;
    float scale = 0.0883883476f;
    uint32_t idesc_QK = make_idesc_QK();
    uint32_t idesc_PV = make_idesc_PV();

    for (int j = 0; j < S; j += 128) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_K, 32768);
            tma_load_2d_cg1_fn(&tma_K, mbar_K, K_smem, 0, b * H * S + h * S + j);
            mbarrier_arrive_and_expect_tx_fn(mbar_V, 32768);
            tma_load_2d_cg1_fn(&tma_V, mbar_V, V_smem, 0, b * H * S + h * S + j);
        }
        
        mbarrier_wait_fn(mbar_K, phase_K);
        phase_K ^= 1;
        
        __syncthreads();
        if (threadIdx.x == 0) {
            for(int k=0; k<8; k++) {
                uint64_t desc_A = make_smem_desc(Q_smem + k * 32, 0, 2048);
                uint64_t desc_B = make_smem_desc(K_smem + k * 4096, 0, 2048);
                uint32_t accum = (k == 0) ? 0 : 1;
                umma_f16_cg1_fn(S_tmem, desc_A, desc_B, idesc_QK, accum);
            }
            umma_commit_cg1_fn(mbar_mma);
        }
        mbarrier_wait_fn(mbar_mma, phase_mma);
        phase_mma ^= 1;
        
        float m_j = -INFINITY;
        for(int c = 0; c < 128; c += 8) {
            uint32_t r[8];
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                         : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]) : "r"(S_tmem + c));
            tmem_load_fence_fn();
            for(int k=0; k<8; k++) {
                float val = __uint_as_float(r[k]) * scale;
                if (j + c + k >= S) val = -INFINITY;
                m_j = max(m_j, val);
            }
        }
        
        float m_prev = m_i;
        m_i = max(m_prev, m_j);
        float exp_diff = fast_exp2f_fn((m_prev - m_i) * 1.44269504f);
        
        float sum_j = 0;
        for(int c = 0; c < 128; c += 8) {
            uint32_t r[8];
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                         : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]) : "r"(S_tmem + c));
            tmem_load_fence_fn();
            for(int k=0; k<8; k++) {
                float val = __uint_as_float(r[k]) * scale;
                if (j + c + k >= S) val = -INFINITY;
                sum_j += fast_exp2f_fn((val - m_i) * 1.44269504f);
            }
        }
        
        l_i = l_i * exp_diff + sum_j;
        
        for(int c = 0; c < 128; c += 8) {
            uint32_t r[8];
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                         : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]) : "r"(S_tmem + c));
            tmem_load_fence_fn();
            uint32_t packed[4];
            for(int k=0; k<4; k++) {
                float val0 = __uint_as_float(r[2*k]) * scale;
                if (j + c + 2*k >= S) val0 = -INFINITY;
                float p0 = fast_exp2f_fn((val0 - m_i) * 1.44269504f);
                
                float val1 = __uint_as_float(r[2*k+1]) * scale;
                if (j + c + 2*k+1 >= S) val1 = -INFINITY;
                float p1 = fast_exp2f_fn((val1 - m_i) * 1.44269504f);
                
                packed[k] = pack_bf16_fn(__float_as_uint(p0), __float_as_uint(p1));
            }
            
            int y = threadIdx.x;
            int x_16b = c / 8;
            int swizzled_x_16b = (y % 8) ^ x_16b;
            uint32_t base_addr = (uint32_t)__cvta_generic_to_shared(P_smem);
            uint32_t addr = base_addr + y * 256 + swizzled_x_16b * 16;
            st_shared_128_fn(addr, packed[0], packed[1], packed[2], packed[3]);
        }
        
        mbarrier_wait_fn(mbar_V, phase_V);
        phase_V ^= 1;
        
        __syncthreads();
        if (threadIdx.x == 0) {
            fence_proxy_async_fn();
            for(int k=0; k<8; k++) {
                uint64_t desc_A = make_smem_desc(P_smem + k * 32, 0, 2048);
                uint64_t desc_B = make_smem_desc(V_smem + k * 4096, 2048, 1024);
                uint32_t accum = (k == 0) ? 0 : 1;
                umma_f16_cg1_fn(O_tmem, desc_A, desc_B, idesc_PV, accum);
            }
            umma_commit_cg1_fn(mbar_mma);
        }
        mbarrier_wait_fn(mbar_mma, phase_mma);
        phase_mma ^= 1;
        
        for(int c=0; c<128; c++) O_i[c] *= exp_diff;
        
        for(int c = 0; c < 128; c += 8) {
            uint32_t r[8];
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                         : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]) : "r"(O_tmem + c));
            tmem_load_fence_fn();
            for(int k=0; k<8; k++) {
                O_i[c+k] += __uint_as_float(r[k]);
            }
        }
    }

    __nv_bfloat16* O_smem_linear = (__nv_bfloat16*)smem_buf;
    for(int c = 0; c < 128; c++) {
        O_smem_linear[threadIdx.x * 128 + c] = __float2bfloat16(O_i[c] / l_i);
    }
    __syncthreads();

    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    for (uint32_t step = 0; step < 32; ++step) {
        uint32_t row = step * 4 + warp_id;
        uint32_t col_start = lane_id * 4;
        
        if (i * 128 + row < S) {
            uint64_t global_row = b * H * S + h * S + i * 128 + row;
            uint64_t global_col = col_start;
            uint2 data = *reinterpret_cast<uint2*>(&O_smem_linear[row * 128 + col_start]);
            *reinterpret_cast<uint2*>(O_global + global_row * 128 + global_col) = data;
        }
    }

    if (i * 128 + threadIdx.x < S) {
        int global_idx = b * H * S + h * S + i * 128 + threadIdx.x;
        LSE_global[global_idx] = m_i + logf(l_i);
    }

    if (threadIdx.x < 32) {
        tmem_dealloc_cg1_fn(tmem_base, 256);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    uint64_t B_H_S = B * H * S;
    
    CUtensorMap tma_Q, tma_K, tma_V;
    
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), 128, B_H_S, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), 128, B_H_S, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), 128, B_H_S, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA));
        
    int num_blocks_S = (S + 127) / 128;
    dim3 grid(num_blocks_S, H, B);
    dim3 block(128);
    
    size_t smem_bytes = 140000; // >= 131104 + required bytes
    
    CUDA_CHECK(cudaFuncSetAttribute(mha_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    
    mha_fwd_kernel<<<grid, block, smem_bytes, stream>>>(tma_Q, tma_K, tma_V, 
        static_cast<__nv_bfloat16*>(O.data_ptr()), 
        static_cast<float*>(LSE.data_ptr()), 
        B, H, S);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda