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

inline CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
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

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
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

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
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

__device__ __forceinline__ uint64_t make_smem_desc(void* smem_ptr, uint32_t lbo, uint32_t sbo, int swizzle_type) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    uint32_t base_offset = (addr >> 7) & 0x7;
    d |= (uint64_t)(base_offset) << 49;
    d |= (uint64_t)swizzle_type << 61;
    return d;
}

__device__ __forceinline__ uint64_t make_desc_Q_K(void* ptr, uint32_t offset_bytes) {
    // K-Major 128B Swizzled
    return make_smem_desc((char*)ptr + offset_bytes, 1, 1024, 2);
}

__device__ __forceinline__ uint64_t make_desc_V(void* ptr, uint32_t offset_bytes) {
    // MN-Major 128B Swizzled
    return make_smem_desc((char*)ptr + offset_bytes, 16384, 1024, 2);
}

__device__ __forceinline__ uint32_t make_instr_desc_qk(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);  // FP32 accum
    d |= (1u << 7);  // BF16 A
    d |= (1u << 10); // BF16 B
    d |= (0u << 15); // A is K-Major
    d |= (0u << 16); // B is K-Major
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_pv(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);  // FP32 accum
    d |= (1u << 7);  // BF16 A
    d |= (1u << 10); // BF16 B
    d |= (0u << 15); // A is K-Major
    d |= (1u << 16); // B is MN-Major (Transpose)
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
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

__global__ void mha_fwd_sm100_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S,
    float scale
) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int q_step = blockIdx.x;
    
    int q_valid = S - q_step * 128;
    if (q_valid <= 0) return;
    if (q_valid > 128) q_valid = 128;
    
    int tid = threadIdx.x;
    
    extern __shared__ __align__(1024) char smem[];
    char* smem_Q     = smem;                    // 32KB
    char* smem_K     = smem_Q + 32768;          // 32KB
    char* smem_V     = smem_K + 32768;          // 32KB
    char* smem_P     = smem_V + 32768;          // 32KB
    char* smem_O_acc = smem_P + 32768;          // 64KB
    
    float* s_o_acc = (float*)smem_O_acc;
    
    __shared__ uint32_t smem_tmem_base;
    if (tid < 32) {
        tmem_alloc_cg1_fn(&smem_tmem_base, 256);
    }
    
    __shared__ uint64_t mbar_tma_Q[1];
    __shared__ uint64_t mbar_tma_KV[1];
    __shared__ uint64_t mbar_mma[1];
    
    if (tid == 0) {
        init_smem_barrier_fn(mbar_tma_Q, 1);
        init_smem_barrier_fn(mbar_tma_KV, 1);
        init_smem_barrier_fn(mbar_mma, 1);
    }
    
    for(int c = 0; c < 128; c += 4) {
        float4 zeros = {0.0f, 0.0f, 0.0f, 0.0f};
        ((float4*)s_o_acc)[tid * 32 + c / 4] = zeros;
    }
    
    __syncthreads();
    
    if (tid == 0) {
        fence_smem_barrier_init_fn();
    }
    
    __syncthreads();
    
    uint32_t tmem_base = smem_tmem_base;
    
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_tma_Q, 32768);
        tma_load_2d_fn(&tma_Q, mbar_tma_Q, smem_Q, 0, (b * H + h) * S + q_step * 128);
        tma_load_2d_fn(&tma_Q, mbar_tma_Q, smem_Q + 16384, 64, (b * H + h) * S + q_step * 128);
    }
    uint32_t phase_tma_Q = 0;
    mbarrier_wait_fn(mbar_tma_Q, phase_tma_Q);
    
    float m_i = -INFINITY;
    float l_i = 0.0f;
    
    uint32_t idesc_QK = make_instr_desc_qk(128, 128);
    uint32_t idesc_PV = make_instr_desc_pv(128, 128);
    uint32_t tmem_P = tmem_base;
    uint32_t tmem_O_tmp = tmem_base + 128;
    
    uint32_t phase_mma = 0;
    uint32_t phase_tma_KV = 0;
    
    int num_KV_blocks = (S + 127) / 128;
    for (int step = 0; step < num_KV_blocks; step++) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_tma_KV, 32768 * 2);
            tma_load_2d_fn(&tma_K, mbar_tma_KV, smem_K, 0, (b * H + h) * S + step * 128);
            tma_load_2d_fn(&tma_K, mbar_tma_KV, smem_K + 16384, 64, (b * H + h) * S + step * 128);
            tma_load_2d_fn(&tma_V, mbar_tma_KV, smem_V, 0, (b * H + h) * S + step * 128);
            tma_load_2d_fn(&tma_V, mbar_tma_KV, smem_V + 16384, 64, (b * H + h) * S + step * 128);
        }
        mbarrier_wait_fn(mbar_tma_KV, phase_tma_KV);
        phase_tma_KV ^= 1;
        fence_proxy_async_fn();
        
        if (tid == 0) {
            for (int k = 0; k < 128; k += 16) {
                uint32_t tile_offset = (k >= 64) ? 16384 : 0;
                uint32_t k_in_tile = (k >= 64) ? k - 64 : k;
                uint64_t desc_Q_k = make_desc_Q_K(smem_Q + tile_offset, k_in_tile * 2);
                uint64_t desc_K_k = make_desc_Q_K(smem_K + tile_offset, k_in_tile * 2);
                umma_f16_cg1_fn((0<<16) | tmem_P, desc_Q_k, desc_K_k, idesc_QK, (k == 0) ? 0 : 1);
            }
            umma_commit_cg1_fn(mbar_mma);
        }
        mbarrier_wait_fn(mbar_mma, phase_mma);
        phase_mma ^= 1;
        
        float r_row[128];
        float m_prev = m_i;
        float m_curr = m_prev;
        
        for (int c = 0; c < 128; c += 8) {
            uint32_t* r = (uint32_t*)&r_row[c];
            tmem_load_8x_fn(tmem_P + c, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
        }
        tmem_load_fence_fn();
        
        int k_valid = S - step * 128;
        for(int c = 0; c < 128; c++) {
            if (c >= k_valid) {
                r_row[c] = -INFINITY;
            } else {
                r_row[c] *= scale;
            }
            m_curr = fmaxf(m_curr, r_row[c]);
        }
        
        float l_curr = l_i * fast_exp2f_fn((m_prev - m_curr) * 1.44269504f);
        
        for (int c = 0; c < 128; c += 8) {
            r_row[c]   = fast_exp2f_fn((r_row[c] - m_curr) * 1.44269504f);
            r_row[c+1] = fast_exp2f_fn((r_row[c+1] - m_curr) * 1.44269504f);
            r_row[c+2] = fast_exp2f_fn((r_row[c+2] - m_curr) * 1.44269504f);
            r_row[c+3] = fast_exp2f_fn((r_row[c+3] - m_curr) * 1.44269504f);
            r_row[c+4] = fast_exp2f_fn((r_row[c+4] - m_curr) * 1.44269504f);
            r_row[c+5] = fast_exp2f_fn((r_row[c+5] - m_curr) * 1.44269504f);
            r_row[c+6] = fast_exp2f_fn((r_row[c+6] - m_curr) * 1.44269504f);
            r_row[c+7] = fast_exp2f_fn((r_row[c+7] - m_curr) * 1.44269504f);
            
            l_curr += r_row[c] + r_row[c+1] + r_row[c+2] + r_row[c+3] + 
                      r_row[c+4] + r_row[c+5] + r_row[c+6] + r_row[c+7];
            
            uint32_t bf32_0 = pack_bf16_fn(__float_as_uint(r_row[c]), __float_as_uint(r_row[c+1]));
            uint32_t bf32_1 = pack_bf16_fn(__float_as_uint(r_row[c+2]), __float_as_uint(r_row[c+3]));
            uint32_t bf32_2 = pack_bf16_fn(__float_as_uint(r_row[c+4]), __float_as_uint(r_row[c+5]));
            uint32_t bf32_3 = pack_bf16_fn(__float_as_uint(r_row[c+6]), __float_as_uint(r_row[c+7]));
            
            int tile = c / 64;
            int c_in_tile = c % 64;
            int x = c_in_tile / 8;
            int x_swizzled = (tid % 8) ^ x;
            int offset_elements = tile * 8192 + tid * 64 + x_swizzled * 8;
            
            uint4 val;
            val.x = bf32_0; val.y = bf32_1; val.z = bf32_2; val.w = bf32_3;
            ((uint4*)smem_P)[offset_elements / 8] = val;
        }
        
        __syncthreads();
        fence_async_shared_fn();
        
        if (tid == 0) {
            for (int k = 0; k < 128; k += 16) {
                uint32_t tile_offset = (k >= 64) ? 16384 : 0;
                uint32_t k_in_tile = (k >= 64) ? k - 64 : k;
                uint64_t desc_P_k = make_desc_Q_K(smem_P + tile_offset, k_in_tile * 2);
                uint64_t desc_V_k = make_desc_V(smem_V, k * 128);
                umma_f16_cg1_fn((0<<16) | tmem_O_tmp, desc_P_k, desc_V_k, idesc_PV, (k == 0) ? 0 : 1);
            }
            umma_commit_cg1_fn(mbar_mma);
        }
        mbarrier_wait_fn(mbar_mma, phase_mma);
        phase_mma ^= 1;
        
        for (int c = 0; c < 128; c += 8) {
            uint32_t* r = (uint32_t*)&r_row[c];
            tmem_load_8x_fn(tmem_O_tmp + c, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
        }
        tmem_load_fence_fn();
        
        float rescale = fast_exp2f_fn((m_prev - m_curr) * 1.44269504f);
        
        for (int c = 0; c < 128; c += 4) {
            float4 old_val = ((float4*)s_o_acc)[tid * 32 + c / 4];
            float4 new_val;
            new_val.x = old_val.x * rescale + r_row[c];
            new_val.y = old_val.y * rescale + r_row[c+1];
            new_val.z = old_val.z * rescale + r_row[c+2];
            new_val.w = old_val.w * rescale + r_row[c+3];
            ((float4*)s_o_acc)[tid * 32 + c / 4] = new_val;
        }
        
        m_i = m_curr;
        l_i = l_curr;
        __syncthreads(); 
    }
    
    char* smem_O = smem_P;
    if (tid < q_valid) {
        for (int c = 0; c < 128; c += 8) {
            float4 v0 = ((float4*)s_o_acc)[tid * 32 + c / 4];
            float4 v1 = ((float4*)s_o_acc)[tid * 32 + c / 4 + 1];
            uint32_t bf32_0 = pack_bf16_fn(__float_as_uint(v0.x / l_i), __float_as_uint(v0.y / l_i));
            uint32_t bf32_1 = pack_bf16_fn(__float_as_uint(v0.z / l_i), __float_as_uint(v0.w / l_i));
            uint32_t bf32_2 = pack_bf16_fn(__float_as_uint(v1.x / l_i), __float_as_uint(v1.y / l_i));
            uint32_t bf32_3 = pack_bf16_fn(__float_as_uint(v1.z / l_i), __float_as_uint(v1.w / l_i));
            
            uint4 val;
            val.x = bf32_0; val.y = bf32_1; val.z = bf32_2; val.w = bf32_3;
            ((uint4*)smem_O)[(tid * 128 + c) / 8] = val;
        }
    }
    __syncthreads();
    
    uint4* g_O = (uint4*)(O + (b * H + h) * S * 128 + q_step * 128 * 128);
    uint4* s_O = (uint4*)smem_O;
    int total_uint4 = q_valid * 128 / 8;
    for(int i = tid; i < total_uint4; i += 128) {
        g_O[i] = s_O[i];
    }
    
    if (tid < q_valid) {
        float lse = m_i + logf(l_i);
        LSE[(b * H + h) * S + q_step * 128 + tid] = lse;
    }
    
    __syncthreads();
    if (tid < 32) {
        tmem_dealloc_cg1_fn(smem_tmem_base, 256);
    }
}

namespace tvm_ffi_example_cuda {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    
    CUtensorMap tma_Q, tma_K, tma_V;
    create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), 128, B*H*S, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), 128, B*H*S, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), 128, B*H*S, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    int num_q_blocks = (S + 127) / 128;
    dim3 grid(num_q_blocks, H, B);
    dim3 block(128);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUDA_CHECK(cudaFuncSetAttribute(mha_fwd_sm100_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 192 * 1024));
    
    mha_fwd_sm100_kernel<<<grid, block, 192 * 1024, stream>>>(
        tma_Q, tma_K, tma_V,
        static_cast<__nv_bfloat16*>(O.data_ptr()),
        static_cast<float*>(LSE.data_ptr()),
        B, H, S, 1.0f / sqrtf(128.0f)
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}