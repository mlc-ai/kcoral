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


// ---- Minimal Device Helpers for TMA, MBarriers, and SMEM Descriptors ----
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

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ void gemm_64x64x128(
    uint64_t* bar, float* c_smem,
    __nv_bfloat16* Q_0, __nv_bfloat16* Q_1,
    __nv_bfloat16* K_0, __nv_bfloat16* K_1,
    uint32_t idesc, uint32_t accum) {
    
    for (int k = 0; k < 4; ++k) {
        uint64_t desc_a0 = make_smem_desc_sm100_fn(Q_0 + k * 16 * 16, 1, 1024); 
        uint64_t desc_b0 = make_smem_desc_sm100_fn(K_0 + k * 16 * 16, 1, 1024); 
        
        uint64_t desc_a1 = make_smem_desc_sm100_fn(Q_1 + k * 16 * 16, 1, 1024); 
        uint64_t desc_b1 = make_smem_desc_sm100_fn(K_1 + k * 16 * 16, 1, 1024); 
        
        uint32_t addr_c = (uint32_t)__cvta_generic_to_shared(c_smem);
        
        asm volatile(
            "{\n.reg .pred p;\n"
            "setp.ne.b32 p, %4, 0;\n"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
            :: "r"(addr_c + k * 256), "l"(desc_a0), "l"(desc_b0), "r"(idesc), "r"(accum));
            
        asm volatile(
            "{\n.reg .pred p;\n"
            "setp.ne.b32 p, %4, 0;\n"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
            :: "r"(addr_c + k * 256), "l"(desc_a1), "l"(desc_b1), "r"(idesc), "r"(accum));
    }
    
    asm volatile(
        "tcgen05.commit.cta_group::1"
        ".mbarrier::arrive::one.shared::cta.b64"
        " [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])));
}

__device__ __forceinline__ void gemm_64x128_for_V(
    float* P_local, __nv_bfloat16* V_0, __nv_bfloat16* V_1,
    float (*O_0)[64], float (*O_1)[64]) {
    for (int r = 0; r < 64; ++r) {
        for (int c = 0; c < 64; ++c) {
            float p = P_local[r * 64 + c];
            for (int i = 0; i < 64; ++i) {
                float v0 = __bfloat162float(V_0[i * 64 + c]);
                float v1 = __bfloat162float(V_1[i * 64 + c]);
                O_0[r][i] += p * v0;
                O_1[r][i] += p * v1;
            }
        }
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


namespace tvm_ffi_mha {

__global__ void mha_kernel(const __grid_constant__ CUtensorMap tma_Q,
                          const __grid_constant__ CUtensorMap tma_K,
                          const __grid_constant__ CUtensorMap tma_V,
                          __nv_bfloat16* O, float* LSE, uint32_t S) {
    
    extern __shared__ __align__(1024) uint8_t smem[];
    uint64_t* bar = (uint64_t*)smem; 
    __nv_bfloat16* Q_0 = (__nv_bfloat16*)(smem + 1024);     
    __nv_bfloat16* Q_1 = (__nv_bfloat16*)(smem + 9216);    
    __nv_bfloat16* K_0 = (__nv_bfloat16*)(smem + 17408);   
    __nv_bfloat16* K_1 = (__nv_bfloat16*)(smem + 25600);   
    __nv_bfloat16* V_0 = (__nv_bfloat16*)(smem + 33792);   
    __nv_bfloat16* V_1 = (__nv_bfloat16*)(smem + 41984);   
    float* P_local = (float*)(smem + 50176);                 
    
    uint32_t i = blockIdx.x;
    uint32_t batch_head_idx = blockIdx.y;
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    int phase = 0;

    uint32_t coord0_q = 0;
    uint32_t coord1_q = (i * 64 + 0) + batch_head_idx * S;
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar, 2 * 8192);
        tma_load_2d_fn(&tma_Q, bar, Q_0, coord0_q, coord1_q);
        tma_load_2d_fn(&tma_Q, bar, Q_1, coord0_q + 64, coord1_q);
    }
    mbarrier_wait_fn(bar, phase);
    phase ^= 1;
    
    float max_val[64], sum_exp[64];
    for (int r = 0; r < 64; ++r) {
        max_val[r] = -1e20f;
        sum_exp[r] = 0.0f;
    }
    
    float O_0[64][64], O_1[64][64];
    for (int r = 0; r < 64; ++r) {
        for (int c = 0; c < 64; ++c) {
            O_0[r][c] = 0.0f;
            O_1[r][c] = 0.0f;
        }
    }
    
    uint32_t idesc = make_instr_desc_fn(64, 64);
    uint32_t accum = 0;
    
    for (int j = 0; j <= i && i * 64 + 63 >= j * 64; ++j) {
        uint32_t coord0_k = 0;
        uint32_t coord1_k = (j * 64 + 0) + batch_head_idx * S;
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar, 2 * 8192);
            tma_load_2d_fn(&tma_K, bar, K_0, coord0_k, coord1_k);
            tma_load_2d_fn(&tma_K, bar, K_1, coord0_k + 64, coord1_k);
        }
        mbarrier_wait_fn(bar, phase);
        phase ^= 1;
        
        gemm_64x64x128(bar, P_local, Q_0, Q_1, K_0, K_1, idesc, accum);
        accum = 1;
        
        mbarrier_wait_fn(bar, phase);
        phase ^= 1;
        
        for (int r = 0; r < 64; ++r) {
            for (int c = 0; c < 64; ++c) {
                P_local[r * 64 + c] /= 11.3137f; // sqrt(128) -> scale factor
                
                if (c > r) { 
                    P_local[r * 64 + c] = -1e20f;
                }
            }
        }
        
        for (int r = 0; r < 64; ++r) {
            float m = -1e20f;
            for (int c = 0; c < 64; ++c) {
                if (j * 64 + c <= i * 64 + r) {
                    m = fmaxf(m, P_local[r * 64 + c]);
                }
            }
            row_max[r] = m;
            
            float s = 0.0f;
            for (int c = 0; c < 64; ++c) {
                P_local[r * 64 + c] = expf(P_local[r * 64 + c] - row_max[r]);
                s += P_local[r * 64 + c];
            }
            row_sum[r] = s;
        }
        
        for (int r = 0; r < 64; ++r) {
            float m_prev = max_val[r];
            max_val[r] = fmaxf(m_prev, row_max[r]);
            sum_exp[r] *= expf(m_prev - max_val[r]);
            sum_exp[r] += row_sum[r];
            
            float total_s = sum_exp[r];
            for (int c = 0; c < 64; ++c) {
                P_local[r * 64 + c] /= total_s;
            }
        }
        
        gemm_64x128_for_V(P_local, V_0, V_1, O_0, O_1);
    }
    
    __nv_bfloat16* O_base = static_cast<__nv_bfloat16*>(O.data_ptr());
    for (int r = 0; r < 64; ++r) {
        if ((i * 64 + r) < S) {
            for (int c = 0; c < 64; ++c) {
                O_base[(i * 64 + r) * 128 + c] = __float2bfloat16(O_0[r][c]);
                O_base[(i * 64 + r) * 128 + c + 64] = __float2bfloat16(O_1[r][c]);
            }
        }
    }
    
    if (threadIdx.x == 0) {
        float* LSE = static_cast<float*>(lse_ptr);
        for (int r = 0; r < 64; ++r) {
            if ((i * 64 + r) < S) {
                LSE[i * 64 + r] = logf(sum_exp[r]);
            }
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView lse) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    uint32_t B = 4, H = 48, S = Q.size(2), D = 128;
    
    CUtensorMap tma_Q, tma_K, tma_V;
    __nv_bfloat16* q_ptr = static_cast<__nv_bfloat16*>(Q.data_ptr());
    __nv_bfloat16* k_ptr = static_cast<__nv_bfloat16*>(K.data_ptr());
    __nv_bfloat16* v_ptr = static_cast<__nv_bfloat16*>(V.data_ptr());
    
    create_tma_2d_descriptor_2B(&tma_Q, q_ptr, D, B * H * S, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_K, k_ptr, D, B * H * S, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_V, v_ptr, D, B * H * S, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    __nv_bfloat16* o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_ptr = static_cast<float*>(lse.data_ptr());
    
    dim3 grid((S + 63) / 64, B * H);
    dim3 block(128);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUDA_CHECK(cudaFuncSetAttribute(mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 68 * 1024));
    
    mha_kernel<<<grid, block, 68 * 1024, stream>>>(tma_Q, tma_K, tma_V, o_ptr, lse_ptr, S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_mha