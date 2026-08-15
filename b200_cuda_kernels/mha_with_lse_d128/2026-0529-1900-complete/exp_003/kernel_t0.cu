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

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
}

__device__ __forceinline__ void tma_store_3d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [%0, {%2, %3, %4}], [%1];"
        :: "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
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
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}

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

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

extern __shared__ char smem[];

__global__ void mha_forward_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    float* lse_ptr,
    int S
) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int s_q = blockIdx.x * 128;
    int bh = b * gridDim.y + h;
    
    if (s_q >= S) return;

    char* smem_Q_left = smem;
    char* smem_Q_right = smem_Q_left + 16384;
    char* smem_K_left = smem_Q_right + 16384;
    char* smem_K_right = smem_K_left + 16384;
    char* smem_V_left = smem_K_right + 16384;
    char* smem_V_right = smem_V_left + 16384;
    char* smem_P_left = smem_V_right + 16384;
    char* smem_P_right = smem_P_left + 16384;
    
    uint64_t* mbar_Q = (uint64_t*)(smem_P_right + 16384);
    uint64_t* mbar_KV = mbar_Q + 1;
    uint64_t* mbar_mma = mbar_KV + 1;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(mbar_KV, 1);
        init_smem_barrier_fn(mbar_mma, 1);
        
        tma_load_3d_fn(&tma_Q, mbar_Q, smem_Q_left, 0, s_q, bh);
        tma_load_3d_fn(&tma_Q, mbar_Q, smem_Q_right, 64, s_q, bh);
    }
    
    __syncthreads();

    uint32_t tmem_base;
    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_base, 256);
    }
    __syncthreads();
    
    uint32_t tmem_O = tmem_base;
    uint32_t tmem_S = tmem_base + 128;

    float m_i = -1e20f;
    float l_i = 0.0f;
    
    mbarrier_wait_fn(mbar_Q, 0);
    
    uint64_t desc_Q_left = make_smem_desc_sm100_fn(smem_Q_left, 1, 1024);
    uint64_t desc_Q_right = make_smem_desc_sm100_fn(smem_Q_right, 1, 1024);
    uint64_t desc_K_left = make_smem_desc_sm100_fn(smem_K_left, 1, 1024);
    uint64_t desc_K_right = make_smem_desc_sm100_fn(smem_K_right, 1, 1024);
    uint64_t desc_V_left_top = make_smem_desc_sm100_fn(smem_V_left, 8192, 1024);
    uint64_t desc_V_left_bottom = make_smem_desc_sm100_fn(smem_V_left + 8192, 8192, 1024);
    uint64_t desc_V_right_top = make_smem_desc_sm100_fn(smem_V_right, 8192, 1024);
    uint64_t desc_V_right_bottom = make_smem_desc_sm100_fn(smem_V_right + 8192, 8192, 1024);
    uint64_t desc_P_left = make_smem_desc_sm100_fn(smem_P_left, 1, 1024);
    uint64_t desc_P_right = make_smem_desc_sm100_fn(smem_P_right, 1, 1024);

    uint32_t idesc_QK = 0;
    idesc_QK |= (1u << 4) | (1u << 7) | (1u << 10);
    idesc_QK |= (0u << 15) | (0u << 16);
    idesc_QK |= (16u << 17) | (8u << 24);

    uint32_t idesc_PV = 0;
    idesc_PV |= (1u << 4) | (1u << 7) | (1u << 10);
    idesc_PV |= (0u << 15) | (1u << 16);
    idesc_PV |= (8u << 17) | (8u << 24);

    uint32_t phase_KV = 0;
    uint32_t phase_mma = 0;
    
    int tid = threadIdx.x;
    
    for (int j = 0; j < S; j += 128) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_KV, 4 * 16384);
            tma_load_3d_fn(&tma_K, mbar_KV, smem_K_left, 0, j, bh);
            tma_load_3d_fn(&tma_K, mbar_KV, smem_K_right, 64, j, bh);
            tma_load_3d_fn(&tma_V, mbar_KV, smem_V_left, 0, j, bh);
            tma_load_3d_fn(&tma_V, mbar_KV, smem_V_right, 64, j, bh);
        }
        
        mbarrier_wait_fn(mbar_KV, phase_KV);
        phase_KV ^= 1;
        
        if (threadIdx.x == 0) {
            asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 0;\n"
                :: "r"(tmem_S), "l"(desc_Q_left), "l"(desc_K_left), "r"(idesc_QK));
            asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 1;\n"
                :: "r"(tmem_S), "l"(desc_Q_right), "l"(desc_K_right), "r"(idesc_QK));
                
            uint32_t a = (uint32_t)__cvta_generic_to_shared(mbar_mma);
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(a));
        }
        
        mbarrier_wait_fn(mbar_mma, phase_mma);
        phase_mma ^= 1;
        
        float row[128];
        float rowmax = -1e20f;
        #pragma unroll 4
        for (int c = 0; c < 128; c += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_S + c));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float f0 = (s_q + tid >= S || j + c + 0 >= S) ? -1e20f : __uint_as_float(r0) * 0.08838834764f;
            float f1 = (s_q + tid >= S || j + c + 1 >= S) ? -1e20f : __uint_as_float(r1) * 0.08838834764f;
            float f2 = (s_q + tid >= S || j + c + 2 >= S) ? -1e20f : __uint_as_float(r2) * 0.08838834764f;
            float f3 = (s_q + tid >= S || j + c + 3 >= S) ? -1e20f : __uint_as_float(r3) * 0.08838834764f;
            rowmax = max(rowmax, max(max(f0, f1), max(f2, f3)));
            row[c+0] = f0; row[c+1] = f1; row[c+2] = f2; row[c+3] = f3;
        }
        
        float m_prev = m_i;
        float m_new = max(m_prev, rowmax);
        
        float sum = 0.0f;
        #pragma unroll 4
        for (int c = 0; c < 128; ++c) {
            float p = fast_exp2f_fn((row[c] - m_new) * 1.44269504089f);
            row[c] = p;
            sum += p;
        }
        
        float scale = fast_exp2f_fn((m_prev - m_new) * 1.44269504089f);
        l_i = l_i * scale + sum;
        m_i = m_new;
        
        #pragma unroll 2
        for (int c = 0; c < 64; c += 8) {
            uint32_t p0 = pack_bf16_fn(__float_as_uint(row[c+0]), __float_as_uint(row[c+1]));
            uint32_t p1 = pack_bf16_fn(__float_as_uint(row[c+2]), __float_as_uint(row[c+3]));
            uint32_t p2 = pack_bf16_fn(__float_as_uint(row[c+4]), __float_as_uint(row[c+5]));
            uint32_t p3 = pack_bf16_fn(__float_as_uint(row[c+6]), __float_as_uint(row[c+7]));
            st_shared_128_fn((uint32_t)__cvta_generic_to_shared(smem_P_left + tid * 128 + c * 2), p0, p1, p2, p3);
        }
        #pragma unroll 2
        for (int c = 64; c < 128; c += 8) {
            uint32_t p0 = pack_bf16_fn(__float_as_uint(row[c+0]), __float_as_uint(row[c+1]));
            uint32_t p1 = pack_bf16_fn(__float_as_uint(row[c+2]), __float_as_uint(row[c+3]));
            uint32_t p2 = pack_bf16_fn(__float_as_uint(row[c+4]), __float_as_uint(row[c+5]));
            uint32_t p3 = pack_bf16_fn(__float_as_uint(row[c+6]), __float_as_uint(row[c+7]));
            st_shared_128_fn((uint32_t)__cvta_generic_to_shared(smem_P_right + tid * 128 + (c - 64) * 2), p0, p1, p2, p3);
        }
        
        fence_async_shared_fn(); 
        
        if (j > 0 && scale != 1.0f) {
            #pragma unroll 4
            for (int c = 0; c < 128; c += 4) {
                uint32_t r0, r1, r2, r3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_O + c));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float f0 = __uint_as_float(r0) * scale;
                float f1 = __uint_as_float(r1) * scale;
                float f2 = __uint_as_float(r2) * scale;
                float f3 = __uint_as_float(r3) * scale;
                r0 = __float_as_uint(f0); r1 = __float_as_uint(f1); r2 = __float_as_uint(f2); r3 = __float_as_uint(f3);
                asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};" :: "r"(tmem_O + c), "r"(r0), "r"(r1), "r"(r2), "r"(r3));
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        }
        
        __syncthreads();
        
        if (threadIdx.x == 0) {
            int accum = (j == 0) ? 0 : 1;
            asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, %4;\n"
                :: "r"(tmem_O + 0), "l"(desc_P_left), "l"(desc_V_left_top), "r"(idesc_PV), "r"(accum));
            asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 1;\n"
                :: "r"(tmem_O + 0), "l"(desc_P_right), "l"(desc_V_left_bottom), "r"(idesc_PV));
            
            asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, %4;\n"
                :: "r"(tmem_O + 64), "l"(desc_P_left), "l"(desc_V_right_top), "r"(idesc_PV), "r"(accum));
            asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 1;\n"
                :: "r"(tmem_O + 64), "l"(desc_P_right), "l"(desc_V_right_bottom), "r"(idesc_PV));
                
            uint32_t a = (uint32_t)__cvta_generic_to_shared(mbar_mma);
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(a));
        }
        
        mbarrier_wait_fn(mbar_mma, phase_mma);
        phase_mma ^= 1;
    }
    
    float inv_l = 1.0f / l_i;
    #pragma unroll 2
    for (int c = 0; c < 128; c += 8) {
        uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" 
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) 
            : "r"(tmem_O + c));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        float f0 = __uint_as_float(r0) * inv_l;
        float f1 = __uint_as_float(r1) * inv_l;
        float f2 = __uint_as_float(r2) * inv_l;
        float f3 = __uint_as_float(r3) * inv_l;
        float f4 = __uint_as_float(r4) * inv_l;
        float f5 = __uint_as_float(r5) * inv_l;
        float f6 = __uint_as_float(r6) * inv_l;
        float f7 = __uint_as_float(r7) * inv_l;
        uint32_t p0 = pack_bf16_fn(__float_as_uint(f0), __float_as_uint(f1));
        uint32_t p1 = pack_bf16_fn(__float_as_uint(f2), __float_as_uint(f3));
        uint32_t p2 = pack_bf16_fn(__float_as_uint(f4), __float_as_uint(f5));
        uint32_t p3 = pack_bf16_fn(__float_as_uint(f6), __float_as_uint(f7));
        
        if (c < 64) {
            st_shared_128_fn((uint32_t)__cvta_generic_to_shared(smem_P_left + tid * 128 + c * 2), p0, p1, p2, p3);
        } else {
            st_shared_128_fn((uint32_t)__cvta_generic_to_shared(smem_P_right + tid * 128 + (c - 64) * 2), p0, p1, p2, p3);
        }
    }
    
    fence_async_shared_fn(); 
    __syncthreads();
    
    if (threadIdx.x == 0) {
        tma_store_3d_fn(&tma_O, smem_P_left, 0, s_q, bh);
        tma_store_3d_fn(&tma_O, smem_P_right, 64, s_q, bh);
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
    }
    
    if (s_q + tid < S) {
        lse_ptr[bh * S + s_q + tid] = m_i + __logf(l_i);
    }
    
    __syncthreads();
    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_base, 256);
    }
}

CUresult create_tma_3d_descriptor_2B(
    CUtensorMap* d, void* globalAddress, 
    uint64_t dim0, uint64_t dim1, uint64_t dim2, 
    uint32_t box0, uint32_t box1, uint32_t box2, 
    CUtensorMapSwizzle swizzle) 
{
    cuuint64_t globalDim[3] = {dim0, dim1, dim2};
    cuuint64_t globalStrides[2] = {dim0 * 2, dim0 * dim1 * 2};
    cuuint32_t boxDim[3] = {box0, box1, box2};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, globalAddress,
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

namespace tvm_ffi_example_cuda {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int D = Q.size(3);

    void* q_ptr = Q.data_ptr();
    void* k_ptr = K.data_ptr();
    void* v_ptr = V.data_ptr();
    void* o_ptr = O.data_ptr();
    float* lse_ptr = static_cast<float*>(LSE.data_ptr());

    CUtensorMap tma_Q, tma_K, tma_V, tma_O;
    if (create_tma_3d_descriptor_2B(&tma_Q, q_ptr, 128, S, B * H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B) != CUDA_SUCCESS ||
        create_tma_3d_descriptor_2B(&tma_K, k_ptr, 128, S, B * H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B) != CUDA_SUCCESS ||
        create_tma_3d_descriptor_2B(&tma_V, v_ptr, 128, S, B * H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B) != CUDA_SUCCESS ||
        create_tma_3d_descriptor_2B(&tma_O, o_ptr, 128, S, B * H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B) != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to create TMA descriptors\n");
        exit(1);
    }

    int blocks_s = (S + 127) / 128;
    dim3 grid(blocks_s, H, B);
    dim3 block(128); 
    
    int smem_bytes = 128 * 1024 + 256; 
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUDA_CHECK(cudaFuncSetAttribute((void*)mha_forward_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    
    mha_forward_kernel<<<grid, block, smem_bytes, stream>>>(
        tma_Q, tma_K, tma_V, tma_O, lse_ptr, S
    );
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}  // namespace tvm_ffi_example_cuda