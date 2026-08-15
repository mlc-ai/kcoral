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

namespace tvm_ffi_example_cuda {

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2, // tensorRank
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

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void tmem_zero(uint32_t tmem_base) {
    uint32_t tid = threadIdx.x;
    uint32_t col = tid * 4; 
    
    uint32_t r0 = 0, r1 = 0, r2 = 0, r3 = 0;
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" 
                 :: "r"(r0),"r"(r1),"r"(r2),"r"(r3),"r"(tmem_base + col));
}

__device__ __forceinline__ void load_s2_reg(float* s_reg, uint32_t s_tmem) {
    uint32_t tid = threadIdx.x;
    uint32_t col = tid * 4;
    
    uint32_t r0, r1, r2, r3;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(s_tmem + col));
    s_reg[0] = __uint_as_float(r0);
    s_reg[1] = __uint_as_float(r1);
    s_reg[2] = __uint_as_float(r2);
    s_reg[3] = __uint_as_float(r3);
}

__device__ __forceinline__ void load_o2_reg(float* o_reg, uint32_t o_tmem) {
    uint32_t tid = threadIdx.x;
    uint32_t col = tid * 4;
    
    uint32_t r0, r1, r2, r3;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(o_tmem + col));
    o_reg[0] = __uint_as_float(r0);
    o_reg[1] = __uint_as_float(r1);
    o_reg[2] = __uint_as_float(r2);
    o_reg[3] = __uint_as_float(r3);
}

__device__ __forceinline__ void umma_cg1_16x16(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void commit_and_wait(uint64_t* bar, uint32_t phase) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared.b64 [%0];" :: "r"(a) : "memory");
    mbarrier_wait_fn(bar, phase);
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16; 
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32; 
    d |= (uint64_t)1 << 46;   
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (0u << 15);   
    d |= (0u << 16);   
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__global__ __launch_bounds__(128) attention_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* __restrict__ O, 
    float* __restrict__ LSE,
    int S_len)
{
    int batch_head = blockIdx.x; 
    int query_start = blockIdx.y * 64;
    int tid = threadIdx.x;
    
    if (query_start >= S_len) return;
    
    extern __shared__ char smem[];
    char* aligned_smem = smem + 256; 
    
    __nv_bfloat16* smem_Q = (__nv_bfloat16*)aligned_smem;                      // 16KB
    __nv_bfloat16* smem_K = (__nv_bfloat16*)(aligned_smem + 16384);             // 32KB
    __nv_bfloat16* smem_V = (__nv_bfloat16*)(aligned_smem + 49152);             // 32KB
    __nv_bfloat16* smem_P = (__nv_bfloat16*)(aligned_smem + 81920);             // 16KB
    
    uint64_t* bar_Q = (uint64_t*)(aligned_smem + 98304);
    uint64_t* bar_K = (uint64_t*)(aligned_smem + 98312);
    uint64_t* bar_V = (uint64_t*)(aligned_smem + 98324);
    
    if (tid == 0) {
        init_smem_barrier_fn(bar_Q, 1);
        init_smem_barrier_fn(bar_K, 1);
        init_smem_barrier_fn(bar_V, 1);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();
    __syncthreads();
    
    uint32_t s_tmem, o_tmem;
    if (tid == 0) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 128;" 
                     :: "r"((uint32_t)__cvta_generic_to_shared(&s_tmem)));
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 128;" 
                     :: "r"((uint32_t)__cvta_generic_to_shared(&o_tmem)));
    }
    __syncthreads();
    
    for (int i = 0; i < 128; ++i) {
        tmem_zero(i * 4 + o_tmem);
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    
    int phase_Q = 0;
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_Q, 16384);
        tma_load_2d_fn(&tma_Q, bar_Q, smem_Q, 0, batch_head * S_len * 128 + query_start * 128);
    }
    mbarrier_wait_fn(bar_Q, phase_Q);
    phase_Q ^= 1;
    
    float local_max[2] = {-1e20f, -1e20f};
    float local_sum[2] = {0, 0};
    
    float s_reg[4] = {0};
    float o_reg[4] = {0};
    
    uint32_t phase_K = 0, phase_V = 0;
    uint32_t idesc_Q = make_instr_desc_fn(64, 128);
    uint32_t idesc_V = make_instr_desc_fn(64, 128);
    
    for (int j = 0; j < S_len; j += 128) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_K, 32768);
            tma_load_2d_fn(&tma_K, bar_K, smem_K, 0, batch_head * S_len * 128 + j * 128);
            
            mbarrier_arrive_and_expect_tx_fn(bar_V, 32768);
            tma_load_2d_fn(&tma_V, bar_V, smem_V, 0, batch_head * S_len * 128 + j * 128);
        }
        
        mbarrier_wait_fn(bar_K, phase_K);
        mbarrier_wait_fn(bar_V, phase_V);
        phase_K ^= 1;
        phase_V ^= 1;
        
        uint64_t desc_Q[8], desc_K[8], desc_P[8], desc_V[8];
        for(int i=0; i<8; ++i) {
            __nv_bfloat16* ptr_Q = smem_Q + i * 16;
            __nv_bfloat16* ptr_K = smem_K + i * 16;
            __nv_bfloat16* ptr_P = smem_P + i * 16;
            __nv_bfloat16* ptr_V = smem_V + i * 16 * 128;
            
            desc_Q[i] = make_smem_desc_sm100_fn(ptr_Q, 1024, 128);
            desc_K[i] = make_smem_desc_sm100_fn(ptr_K, 2048, 128);
            desc_P[i] = make_smem_desc_sm100_fn(ptr_P, 1024, 128);
            desc_V[i] = make_smem_desc_sm100_fn(ptr_V, 2048, 128);
        }
        
        for (int i = 0; i < 128; ++i) {
            tmem_zero(i * 4 + s_tmem);
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        
        for(int i=0; i<8; ++i) {
            umma_cg1_16x16(s_tmem, desc_Q[i], desc_K[i], idesc_Q, i == 0 ? 0 : 1);
        }
        commit_and_wait(bar_Q, phase_Q);
        phase_Q ^= 1;
        
        float my_max[2] = {-1e20f, -1e20f};
        for(int k=0; k<4; ++k) {
            load_s2_reg(s_reg, s_tmem + (tid * 4 + k * 8));
            my_max[0] = max(my_max[0], max(max(s_reg[0], s_reg[1]), max(s_reg[2], s_reg[3])));
            my_max[1] = max(my_max[1], max(max(s_reg[0], s_reg[1]), max(s_reg[2], s_reg[3])));
        }
        
        float new_max[2] = {max(local_max[0], my_max[0]), max(local_max[1], my_max[1])};
        float factor[2] = {expf(local_max[0] - new_max[0]), expf(local_max[1] - new_max[1])};
        
        local_sum[0] *= factor[0];
        local_sum[1] *= factor[1];
        
        for(int k=0; k<4; ++k) {
            load_o2_reg(o_reg, o_tmem + (tid * 4 + k * 8));
            o_reg[0] *= factor[0];
            o_reg[1] *= factor[0];
            o_reg[2] *= factor[0];
            o_reg[3] *= factor[0];
            
            uint32_t r0 = __float_as_uint(o_reg[0]);
            uint32_t r1 = __float_as_uint(o_reg[1]);
            uint32_t r2 = __float_as_uint(o_reg[2]);
            uint32_t r3 = __float_as_uint(o_reg[3]);
            
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" 
                         :: "r"(r0),"r"(r1),"r"(r2),"r"(r3),"r"(o_tmem + (tid * 4 + k * 8)));
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        
        float my_sum[2] = {0, 0};
        for(int k=0; k<4; ++k) {
            load_s2_reg(s_reg, s_tmem + (tid * 4 + k * 8));
            
            float v0 = (j + tid*2 + k*8 < S_len) ? expf(s_reg[0] * (1.0f / sqrtf(128.0f)) - new_max[0]) : 0;
            float v1 = (j + tid*2 + 1 + k*8 < S_len) ? expf(s_reg[1] * (1.0f / sqrtf(128.0f)) - new_max[0]) : 0;
            
            my_sum[0] += v0 + v1;
            *(smem_P + tid*128 + k*16 + 0) = __float2bfloat16(v0);
            *(smem_P + tid*128 + k*16 + 1) = __float2bfloat16(v1);
            
            float v2 = (j + tid*2 + k*8 < S_len) ? expf(s_reg[2] * (1.0f / sqrtf(128.0f)) - new_max[1]) : 0;
            float v3 = (j + tid*2 + 1 + k*8 < S_len) ? expf(s_reg[3] * (1.0f / sqrtf(128.0f)) - new_max[1]) : 0;
            
            my_sum[1] += v2 + v3;
            *(smem_P + tid*128 + k*16 + 64) = __float2bfloat16(v2);
            *(smem_P + tid*128 + k*16 + 65) = __float2bfloat16(v3);
        }
        
        local_sum[0] += my_sum[0];
        local_sum[1] += my_sum[1];
        local_max[0] = new_max[0];
        local_max[1] = new_max[1];
        
        __syncthreads(); 
        
        for(int i=0; i<8; ++i) {
            umma_cg1_16x16(o_tmem, desc_P[i], desc_V[i], idesc_V, 1);
        }
        commit_and_wait(bar_V, phase_V);
        phase_V ^= 1;
        
        __syncthreads(); 
    }
    
    for(int k=0; k<4; ++k) {
        load_o2_reg(o_reg, o_tmem + (tid * 4 + k * 8));
        
        float inv_sum_0 = 1.0f / local_sum[0];
        float out_0 = o_reg[0] * inv_sum_0;
        float out_1 = o_reg[1] * inv_sum_0;
        
        __nv_bfloat16* out_ptr_0 = &((__nv_bfloat16*)O)[batch_head * S_len * 128 + (query_start + tid) * 128 + k * 16];
        *(reinterpret_cast<uint16_t*>(&out_ptr_0[0])) = __float2bfloat16(out_0);
        *(reinterpret_cast<uint16_t*>(&out_ptr_0[1])) = __float2bfloat16(out_1);
        
        float inv_sum_1 = 1.0f / local_sum[1];
        float out_2 = o_reg[2] * inv_sum_1;
        float out_3 = o_reg[3] * inv_sum_1;
        
        __nv_bfloat16* out_ptr_1 = &((__nv_bfloat16*)O)[batch_head * S_len * 128 + (query_start + tid + 1) * 128 + k * 16];
        *(reinterpret_cast<uint16_t*>(&out_ptr_1[0])) = __float2bfloat16(out_2);
        *(reinterpret_cast<uint16_t*>(&out_ptr_1[1])) = __float2bfloat16(out_3);
    }
    
    if (tid < 64) {
        LSE[batch_head * S_len + query_start + tid] = local_max[tid / 32] + logf(local_sum[tid / 32]);
    }
    
    if (tid == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 128;" 
                     :: "r"(s_tmem));
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 128;" 
                     :: "r"(o_tmem));
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    
    CUtensorMap tma_Q, tma_K, tma_V;
    uint64_t S_len = S; 
    
    CUresult res_q = create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), 128, B * H * S_len, 128, 64, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res_q != CUDA_SUCCESS) { fprintf(stderr, "Failed to create tma_Q\n"); exit(1); }
    
    CUresult res_k = create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), 128, B * H * S_len, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res_k != CUDA_SUCCESS) { fprintf(stderr, "Failed to create tma_K\n"); exit(1); }
    
    CUresult res_v = create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), 128, B * H * S_len, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res_v != CUDA_SUCCESS) { fprintf(stderr, "Failed to create tma_V\n"); exit(1); }
    
    dim3 grid(B * H, (S + 63) / 64); 
    dim3 block(128); 
    
    int smem_size = 90112; 
    CUDA_CHECK(cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    attention_kernel<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V,
        static_cast<__nv_bfloat16*>(O.data_ptr()), 
        static_cast<float*>(LSE.data_ptr()), 
        S
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda