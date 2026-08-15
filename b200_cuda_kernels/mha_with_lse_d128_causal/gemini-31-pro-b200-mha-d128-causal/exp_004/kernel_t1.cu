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

namespace tvm_ffi_mha {

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

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tmem_alloc_cg1_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_cg1_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void umma_commit_cg1_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        :: "r"(a)); 
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

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   // version = 1 (SM100)
    d |= (uint64_t)0 << 61;   // layout_type = 0 (NO SWIZZLE)
    return d;
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

__global__ void mha_fwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O_out,
    float* LSE_out,
    int S_seq,
    int H) 
{
    extern __shared__ char smem[];
    __nv_bfloat16* Q_smem = (__nv_bfloat16*)smem;
    __nv_bfloat16* K_smem = Q_smem + 128 * 128;
    __nv_bfloat16* V_smem = K_smem + 128 * 128;
    __nv_bfloat16* P_smem = V_smem + 128 * 128;
    uint64_t* mbar_Q = (uint64_t*)(P_smem + 128 * 128);
    uint64_t* mbar_K = mbar_Q + 1;
    uint64_t* mbar_V = mbar_K + 1;
    uint64_t* mbar_UMMA = mbar_V + 1;
    uint32_t* tmem_base_ptr = (uint32_t*)(mbar_UMMA + 1);

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(mbar_K, 1);
        init_smem_barrier_fn(mbar_V, 1);
        init_smem_barrier_fn(mbar_UMMA, 1);
    }
    if (threadIdx.x < 32) {
        tmem_alloc_cg1_fn(tmem_base_ptr, 256);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t tmem_base = *tmem_base_ptr;
    uint32_t S_tmem = tmem_base;
    uint32_t O_tmem = tmem_base + 128;

    for (int i = 0; i < 128; i += 4) {
        uint32_t r0 = 0, r1 = 0, r2 = 0, r3 = 0;
        asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};"
                     :: "r"(r0),"r"(r1),"r"(r2),"r"(r3), "r"(O_tmem + i) : "memory");
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");

    int batch = blockIdx.z;
    int head = blockIdx.y;
    int m_start = blockIdx.x * 128;
    int coord1_base = batch * H * S_seq + head * S_seq;

    uint32_t phase_Q = 0, phase_K = 0, phase_V = 0, phase_UMMA = 0;

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 128*128*2);
        tma_load_2d_fn(&tma_Q, mbar_Q, Q_smem, 0, coord1_base + m_start);
    }
    mbarrier_wait_fn(mbar_Q, phase_Q);
    phase_Q ^= 1;

    float m_val = -INFINITY;
    float l_val = 0.0f;
    bool is_valid_q = (m_start + threadIdx.x < S_seq);

    uint32_t idesc_QK = 0;
    idesc_QK |= (1u << 4);
    idesc_QK |= (1u << 7);
    idesc_QK |= (1u << 10);
    idesc_QK |= (0u << 15);
    idesc_QK |= (0u << 16);
    idesc_QK |= ((128 / 8) << 17);
    idesc_QK |= ((128 / 16) << 24);

    uint32_t idesc_PV = 0;
    idesc_PV |= (1u << 4);
    idesc_PV |= (1u << 7);
    idesc_PV |= (1u << 10);
    idesc_PV |= (0u << 15);
    idesc_PV |= (1u << 16);
    idesc_PV |= ((128 / 8) << 17);
    idesc_PV |= ((128 / 16) << 24);

    for (int n_start = 0; n_start <= m_start; n_start += 128) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_K, 128*128*2);
            tma_load_2d_fn(&tma_K, mbar_K, K_smem, 0, coord1_base + n_start);
            mbarrier_arrive_and_expect_tx_fn(mbar_V, 128*128*2);
            tma_load_2d_fn(&tma_V, mbar_V, V_smem, 0, coord1_base + n_start);
        }
        mbarrier_wait_fn(mbar_K, phase_K);
        phase_K ^= 1;
        
        if (threadIdx.x == 0) {
            uint32_t addr_Q = (uint32_t)__cvta_generic_to_shared(Q_smem);
            uint32_t addr_K = (uint32_t)__cvta_generic_to_shared(K_smem);
            for (int k_step = 0; k_step < 8; ++k_step) {
                uint64_t desc_Q = make_smem_desc_sm100_fn((void*)(addr_Q + k_step * 32), 2048, 128); 
                uint64_t desc_K = make_smem_desc_sm100_fn((void*)(addr_K + k_step * 32), 2048, 128);
                uint32_t accum = (k_step == 0) ? 0 : 1;
                umma_f16_cg1_fn(S_tmem, desc_Q, desc_K, idesc_QK, accum);
            }
            umma_commit_cg1_fn(mbar_UMMA);
        }
        mbarrier_wait_fn(mbar_UMMA, phase_UMMA);
        phase_UMMA ^= 1;
        
        bool is_causal_block = (n_start == m_start);
        float row_max = -INFINITY;
        float S_vals[128];
        for (int i = 0; i < 128; i += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(S_tmem + i));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float f0 = __uint_as_float(r0) * 0.08838834764f;
            float f1 = __uint_as_float(r1) * 0.08838834764f;
            float f2 = __uint_as_float(r2) * 0.08838834764f;
            float f3 = __uint_as_float(r3) * 0.08838834764f;
            
            if (!is_valid_q || (n_start + i + 0 >= S_seq) || (is_causal_block && n_start + i + 0 > m_start + threadIdx.x)) f0 = -INFINITY;
            if (!is_valid_q || (n_start + i + 1 >= S_seq) || (is_causal_block && n_start + i + 1 > m_start + threadIdx.x)) f1 = -INFINITY;
            if (!is_valid_q || (n_start + i + 2 >= S_seq) || (is_causal_block && n_start + i + 2 > m_start + threadIdx.x)) f2 = -INFINITY;
            if (!is_valid_q || (n_start + i + 3 >= S_seq) || (is_causal_block && n_start + i + 3 > m_start + threadIdx.x)) f3 = -INFINITY;
            
            S_vals[i+0] = f0;
            S_vals[i+1] = f1;
            S_vals[i+2] = f2;
            S_vals[i+3] = f3;
            row_max = fmaxf(row_max, f0);
            row_max = fmaxf(row_max, f1);
            row_max = fmaxf(row_max, f2);
            row_max = fmaxf(row_max, f3);
        }
        
        float m_new = fmaxf(m_val, row_max);
        float scale = 1.0f;
        if (m_val != -INFINITY) {
            scale = exp2f((m_val - m_new) * 1.44269504089f);
        }
        
        if (__any_sync(0xffffffff, scale < 1.0f)) {
            for (int i = 0; i < 128; i += 4) {
                uint32_t r0, r1, r2, r3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                             : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(O_tmem + i));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float o0 = __uint_as_float(r0) * scale;
                float o1 = __uint_as_float(r1) * scale;
                float o2 = __uint_as_float(r2) * scale;
                float o3 = __uint_as_float(r3) * scale;
                r0 = __float_as_uint(o0);
                r1 = __float_as_uint(o1);
                r2 = __float_as_uint(o2);
                r3 = __float_as_uint(o3);
                asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};"
                             :: "r"(r0),"r"(r1),"r"(r2),"r"(r3), "r"(O_tmem + i) : "memory");
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        }
        
        float row_sum = 0.0f;
        for (int i = 0; i < 128; i += 4) {
            float f0 = S_vals[i+0];
            float f1 = S_vals[i+1];
            float f2 = S_vals[i+2];
            float f3 = S_vals[i+3];
            
            f0 = exp2f((f0 - m_new) * 1.44269504089f);
            f1 = exp2f((f1 - m_new) * 1.44269504089f);
            f2 = exp2f((f2 - m_new) * 1.44269504089f);
            f3 = exp2f((f3 - m_new) * 1.44269504089f);
            
            if (!is_valid_q || (n_start + i + 0 >= S_seq) || (is_causal_block && n_start + i + 0 > m_start + threadIdx.x)) f0 = 0.0f;
            if (!is_valid_q || (n_start + i + 1 >= S_seq) || (is_causal_block && n_start + i + 1 > m_start + threadIdx.x)) f1 = 0.0f;
            if (!is_valid_q || (n_start + i + 2 >= S_seq) || (is_causal_block && n_start + i + 2 > m_start + threadIdx.x)) f2 = 0.0f;
            if (!is_valid_q || (n_start + i + 3 >= S_seq) || (is_causal_block && n_start + i + 3 > m_start + threadIdx.x)) f3 = 0.0f;
            
            row_sum += f0 + f1 + f2 + f3;
            
            uint32_t p01 = pack_bf16_fn(__float_as_uint(f0), __float_as_uint(f1));
            uint32_t p23 = pack_bf16_fn(__float_as_uint(f2), __float_as_uint(f3));
            
            uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(P_smem) + threadIdx.x * 256 + i * 2;
            asm volatile("st.shared.v2.b32 [%0], {%1, %2};" :: "r"(smem_addr), "r"(p01), "r"(p23));
        }
        
        if (m_val != -INFINITY) {
            l_val = l_val * scale + row_sum;
        } else {
            l_val = row_sum;
        }
        m_val = m_new;
        
        fence_async_shared_fn();
        __syncthreads();
        
        mbarrier_wait_fn(mbar_V, phase_V);
        phase_V ^= 1;
        
        if (threadIdx.x == 0) {
            uint32_t addr_P = (uint32_t)__cvta_generic_to_shared(P_smem);
            uint32_t addr_V = (uint32_t)__cvta_generic_to_shared(V_smem);
            for (int k_step = 0; k_step < 8; ++k_step) {
                uint64_t desc_P = make_smem_desc_sm100_fn((void*)(addr_P + k_step * 32), 2048, 128); 
                uint64_t desc_V = make_smem_desc_sm100_fn((void*)(addr_V + k_step * 4096), 128, 2048); 
                umma_f16_cg1_fn(O_tmem, desc_P, desc_V, idesc_PV, 1);
            }
            umma_commit_cg1_fn(mbar_UMMA);
        }
        mbarrier_wait_fn(mbar_UMMA, phase_UMMA);
        phase_UMMA ^= 1;
    }
    
    __syncthreads();
    float inv_l = (l_val > 0.0f) ? (1.0f / l_val) : 0.0f;
    for (uint32_t col = 0; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(O_tmem + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float f0 = __uint_as_float(r0) * inv_l;
        float f1 = __uint_as_float(r1) * inv_l;
        float f2 = __uint_as_float(r2) * inv_l;
        float f3 = __uint_as_float(r3) * inv_l;
        
        uint32_t base = threadIdx.x * 128 + col;
        P_smem[base + 0] = __float2bfloat16(f0);
        P_smem[base + 1] = __float2bfloat16(f1);
        P_smem[base + 2] = __float2bfloat16(f2);
        P_smem[base + 3] = __float2bfloat16(f3);
    }
    __syncthreads();
    
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    for (uint32_t step = 0; step < 32; ++step) {
        uint32_t row = step * 4 + warp_id;
        uint32_t global_row = m_start + row;
        uint32_t col_start = lane_id * 4;
        
        if (global_row < S_seq) {
            uint2 data = *reinterpret_cast<uint2*>(&P_smem[row * 128 + col_start]);
            *reinterpret_cast<uint2*>(O_out + coord1_base * 128 + global_row * 128 + col_start) = data;
        }
    }
    
    if (is_valid_q) {
        float lse = (m_val == -INFINITY) ? -INFINITY : (m_val + logf(l_val));
        LSE_out[batch * H * S_seq + head * S_seq + m_start + threadIdx.x] = lse;
    }
    
    if (threadIdx.x < 32) {
        tmem_dealloc_cg1_fn(tmem_base, 256);
    }
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

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int B = Q.size(0);
    int H = Q.size(1);
    int S_seq = Q.size(2);
    int D = Q.size(3);
    
    uint64_t total_S = B * H * S_seq;
    
    CUtensorMap tma_Q, tma_K, tma_V;
    create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), D, total_S, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), D, total_S, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), D, total_S, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    int blocks_x = (S_seq + 127) / 128;
    dim3 grid(blocks_x, H, B);
    dim3 block(128);
    
    int smem_size = 128 * 128 * 2 * 4 + 32 + 8; // ~131 KB
    CUDA_CHECK(cudaFuncSetAttribute(mha_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    mha_fwd_kernel<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, 
        static_cast<__nv_bfloat16*>(O.data_ptr()), 
        static_cast<float*>(LSE.data_ptr()), 
        S_seq, H
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}  // namespace tvm_ffi_mha