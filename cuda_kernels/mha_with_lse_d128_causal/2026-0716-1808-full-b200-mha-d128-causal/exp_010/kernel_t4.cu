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

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
}

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo, uint32_t swizzle) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)(swizzle & 0x7) << 61;
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

__device__ __forceinline__ void wgmma_cta1(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void commit_cta1(uint64_t* bar) {
    asm volatile(
        "tcgen05.commit.cta_group::1"
        ".mbarrier::arrive::one.shared::cta.b64"
        " [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t addr, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
       : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(addr));
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


CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_dim0, uint64_t gmem_dim1, uint64_t gmem_dim2, uint32_t smem_dim0, uint32_t smem_dim1, uint32_t smem_dim2, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[3] = {gmem_dim0, gmem_dim1, gmem_dim2};
    cuuint64_t globalStrides[2] = {gmem_dim0 * 2, gmem_dim0 * gmem_dim1 * 2};
    cuuint32_t boxDim[3] = {smem_dim0, smem_dim1, smem_dim2};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        3,
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

__global__ __launch_bounds__(128, 2) void mha_kernel(const __grid_constant__ CUtensorMap tma_Q,
                          const __grid_constant__ CUtensorMap tma_K,
                          const __grid_constant__ CUtensorMap tma_V,
                          __nv_bfloat16* O, float* LSE, uint32_t S) {
    
    extern __shared__ __align__(128) uint8_t smem_buf[];
    uint64_t* mbar_Q = (uint64_t*)smem_buf;
    uint64_t* mbar_K0 = (uint64_t*)(smem_buf + 8);
    uint64_t* mbar_K1 = (uint64_t*)(smem_buf + 16);
    uint64_t* mbar_V0 = (uint64_t*)(smem_buf + 24);
    uint64_t* mbar_V1 = (uint64_t*)(smem_buf + 32);
    
    // Dynamically align SMEM base to 1024 bytes limit to satisfy swizzle constraints
    uint32_t smem_base = ((uint32_t)__cvta_generic_to_shared(smem_buf) + 1023) & ~1023;
    __nv_bfloat16* Q_0 = (__nv_bfloat16*)(smem_base);         
    __nv_bfloat16* Q_1 = (__nv_bfloat16*)(smem_base + 8192);    
    __nv_bfloat16* K_0[2] = { (__nv_bfloat16*)(smem_base + 16384), (__nv_bfloat16*)(smem_base + 24576) };
    __nv_bfloat16* K_1[2] = { (__nv_bfloat16*)(smem_base + 32768), (__nv_bfloat16*)(smem_base + 40960) };
    __nv_bfloat16* V_0[2] = { (__nv_bfloat16*)(smem_base + 49152), (__nv_bfloat16*)(smem_base + 57344) };
    __nv_bfloat16* V_1[2] = { (__nv_bfloat16*)(smem_base + 65536), (__nv_bfloat16*)(smem_base + 73728) };
    __nv_bfloat16* P_col = (__nv_bfloat16*)(smem_base + 81920);   
    float* P_fp32 = (float*)(smem_base + 90112);                 
    
    uint32_t i = blockIdx.x;
    uint32_t batch_head_idx = blockIdx.y;
    uint32_t cta_id = cluster_rank_fn() % 2;
    uint32_t global_i = i * 2 + cta_id;
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(mbar_K0, 1);
        init_smem_barrier_fn(mbar_K1, 1);
        init_smem_barrier_fn(mbar_V0, 1);
        init_smem_barrier_fn(mbar_V1, 1);
        fence_smem_barrier_init_fn();
        
        uint32_t* tmem_P = (uint32_t*)(smem_base + 98304);
        tmem_alloc_fn(tmem_P, 128);
    }
    __syncthreads();
    
    int phase_Q = 0;
    uint32_t coord1_q = (global_i * 64);
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 4 * 8192);
        tma_load_3d_fn(&tma_Q, mbar_Q, Q_0, 0, coord1_q, batch_head_idx);
        tma_load_3d_fn(&tma_Q, mbar_Q, Q_1, 64, coord1_q, batch_head_idx);
    }
    
    float max_val[64], sum_exp[64];
    for (int r = threadIdx.x; r < 64; r += blockDim.x) {
        max_val[r] = -1e20f;
        sum_exp[r] = 0.0f;
    }
    
    uint32_t idesc_QK = make_instr_desc_fn(64, 64);
    uint32_t idesc_PV = make_instr_desc_fn(64, 64);
    
    int phase_K[2] = {0, 0};
    int phase_V[2] = {0, 0};
    
    if (0 <= global_i) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_K0, 4 * 8192);
            tma_load_3d_fn(&tma_K, mbar_K0, K_0[0], 0, (0 * 64), batch_head_idx);
            tma_load_3d_fn(&tma_K, mbar_K0, K_1[0], 64, (0 * 64), batch_head_idx);
            
            mbarrier_arrive_and_expect_tx_fn(mbar_V0, 4 * 8192);
            tma_load_3d_fn(&tma_V, mbar_V0, V_0[0], 0, (0 * 64), batch_head_idx);
            tma_load_3d_fn(&tma_V, mbar_V0, V_1[0], 64, (0 * 64), batch_head_idx);
        }
    }
    
    for (int j = 0; j <= global_i && global_i * 64 + 63 >= j * 64; ++j) {
        int buf_idx = j % 2;
        int next_buf_idx = (j + 1) % 2;
        uint64_t* current_mbar_K = (buf_idx == 0) ? mbar_K0 : mbar_K1;
        uint64_t* current_mbar_V = (buf_idx == 0) ? mbar_V0 : mbar_V1;
        uint64_t* next_mbar_K = (next_buf_idx == 0) ? mbar_K0 : mbar_K1;
        uint64_t* next_mbar_V = (next_buf_idx == 0) ? mbar_V0 : mbar_V1;
        
        if (j + 1 <= global_i && global_i * 64 + 63 >= (j + 1) * 64) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(next_mbar_K, 4 * 8192);
                tma_load_3d_fn(&tma_K, next_mbar_K, K_0[next_buf_idx], 0, ((j + 1) * 64), batch_head_idx);
                tma_load_3d_fn(&tma_K, next_mbar_K, K_1[next_buf_idx], 64, ((j + 1) * 64), batch_head_idx);
                
                mbarrier_arrive_and_expect_tx_fn(next_mbar_V, 4 * 8192);
                tma_load_3d_fn(&tma_V, next_mbar_V, V_0[next_buf_idx], 0, ((j + 1) * 64), batch_head_idx);
                tma_load_3d_fn(&tma_V, next_mbar_V, V_1[next_buf_idx], 64, ((j + 1) * 64), batch_head_idx);
            }
        }
        
        mbarrier_wait_fn(current_mbar_K, phase_K[buf_idx]);
        mbarrier_wait_fn(current_mbar_V, phase_V[buf_idx]);
        
        uint32_t p_base = 0;
        uint32_t accum_P = 0;
        
        for (int k = 0; k < 4; ++k) {
            uint64_t desc_a0 = make_smem_desc_sm100_fn(Q_0 + k * 16, 1, 1024, 2);
            uint64_t desc_b0 = make_smem_desc_sm100_fn(K_0[buf_idx] + k * 16, 1, 1024, 2);
            wgmma_cta1(p_base + k * 16, desc_a0, desc_b0, idesc_QK, accum_P);
            accum_P = 1;
        }
        for (int k = 0; k < 4; ++k) {
            uint64_t desc_a1 = make_smem_desc_sm100_fn(Q_1 + k * 16, 1, 1024, 2);
            uint64_t desc_b1 = make_smem_desc_sm100_fn(K_1[buf_idx] + k * 16, 1, 1024, 2);
            wgmma_cta1(p_base + k * 16, desc_a1, desc_b1, idesc_QK, accum_P);
        }
        
        commit_cta1(mbar_Q);
        mbarrier_wait_fn(mbar_Q, phase_Q);
        phase_Q ^= 1;
        
        for (int step = 0; step < 8; ++step) {
            int row = (step / 2) * 32 + (threadIdx.x % 64);
            int col = (step % 2) * 32 + (threadIdx.x / 64) * 4;
            
            uint32_t tmem_addr = (row << 16) | col;
            
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_addr, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            
            P_fp32[row * 64 + col + 0] = __uint_as_float(r0);
            P_fp32[row * 64 + col + 1] = __uint_as_float(r1);
            P_fp32[row * 64 + col + 2] = __uint_as_float(r2);
            P_fp32[row * 64 + col + 3] = __uint_as_float(r3);
        }
        __syncthreads(); 
        
        for (int idx = threadIdx.x; idx < 64 * 64; idx += blockDim.x) {
            int r = idx / 64;
            int c = idx % 64;
            if (j * 64 + c > global_i * 64 + r) {
                P_fp32[idx] = -1e20f;
            } else {
                P_fp32[idx] *= 0.08838834764f; // 1.0 / sqrt(128.0)
            }
        }
        
        float row_max[64], row_sum[64];
        for (int r = threadIdx.x; r < 64; r += blockDim.x) {
            float m = -1e20f;
            for (int c = 0; c < 64; ++c) {
                if (j * 64 + c <= global_i * 64 + r) {
                    m = fmaxf(m, P_fp32[r * 64 + c]);
                }
            }
            row_max[r] = m;
            
            float s = 0.0f;
            for (int c = 0; c < 64; ++c) {
                if (j * 64 + c <= global_i * 64 + r) {
                    float p = P_fp32[r * 64 + c];
                    float exp_p = expf(p - m);
                    s += exp_p;
                    P_fp32[r * 64 + c] = exp_p;
                } else {
                    P_fp32[r * 64 + c] = 0.0f;
                }
            }
            row_sum[r] = s;
        }
        __syncthreads(); 
        
        for (int r = threadIdx.x; r < 64; r += blockDim.x) {
            float m_prev = max_val[r];
            max_val[r] = fmaxf(m_prev, row_max[r]);
            sum_exp[r] *= expf(m_prev - max_val[r]);
            sum_exp[r] += row_sum[r] * expf(row_max[r] - max_val[r]);
            
            float total_s = sum_exp[r];
            for (int c = 0; c < 64; ++c) {
                P_fp32[r * 64 + c] = P_fp32[r * 64 + c] * expf(row_max[r] - max_val[r]) / total_s;
            }
        }
        __syncthreads();
        
        for (int idx = threadIdx.x; idx < 64 * 64; idx += blockDim.x) {
            int r = idx / 64;
            int c = idx % 64;
            // Write to P_col in strictly K-major (no swizzle) layout
            P_col[c * 64 + r] = __float2bfloat16(P_fp32[idx]);
        }
        __syncthreads(); 
        
        fence_async_shared_fn();
        
        uint32_t o_base = 0;
        uint32_t accum_O = 0;
        for (int k = 0; k < 4; ++k) {
            // P_col acts as A, passed in K-major (no-swizzle)
            uint64_t desc_a = make_smem_desc_sm100_fn(P_col + k * 16, 1024, 128, 0); 
            // V_0 acts as B, passed in K-major (128B-swizzle)
            uint64_t desc_b0 = make_smem_desc_sm100_fn(V_0[buf_idx] + k * 16 * 64, 1, 1024, 2); 
            
            wgmma_cta1(o_base + k * 16, desc_a, desc_b0, idesc_PV, accum_O);
            accum_O = 1;
        }
        for (int k = 0; k < 4; ++k) {
            uint64_t desc_a = make_smem_desc_sm100_fn(P_col + k * 16, 1024, 128, 0);
            uint64_t desc_b1 = make_smem_desc_sm100_fn(V_1[buf_idx] + k * 16 * 64, 1, 1024, 2);
            
            wgmma_cta1(o_base + 64 + k * 16, desc_a, desc_b1, idesc_PV, accum_O);
        }
        
        commit_cta1(mbar_Q);
        mbarrier_wait_fn(mbar_Q, phase_Q);
        phase_Q ^= 1;
        
        phase_K[buf_idx] ^= 1;
        phase_V[buf_idx] ^= 1;
    }
    
    float* O_smem = (float*)(smem_base + 98304);
    
    for (int step = 0; step < 16; ++step) {
        int half = step / 2;
        int row = (step % 2) * 32 + (threadIdx.x % 64);
        int col = half * 64 + (threadIdx.x / 64) * 4;
        
        uint32_t tmem_addr = (row << 16) | col;
        
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn(tmem_addr, &r0, &r1, &r2, &r3);
        tmem_load_fence_fn();
        
        O_smem[row * 128 + col + 0] = __uint_as_float(r0);
        O_smem[row * 128 + col + 1] = __uint_as_float(r1);
        O_smem[row * 128 + col + 2] = __uint_as_float(r2);
        O_smem[row * 128 + col + 3] = __uint_as_float(r3);
    }
    __syncthreads();
    
    for (int idx = threadIdx.x; idx < 64 * 128; idx += blockDim.x) {
        int r = idx / 128;
        int c = idx % 128;
        if ((global_i * 64 + r) < S) {
            __nv_bfloat16* O_ptr = O + (uint64_t)(batch_head_idx * S + global_i * 64 + r) * 128;
            if (c < 64) {
                O_ptr[c] = __float2bfloat16(O_smem[idx]);
            } else {
                O_ptr[c] = __float2bfloat16(O_smem[idx + 4096]);
            }
        }
    }
    
    if (threadIdx.x < 64 && (global_i * 64 + threadIdx.x) < S) {
        LSE[(uint64_t)batch_head_idx * S + global_i * 64 + threadIdx.x] = max_val[threadIdx.x] + logf(sum_exp[threadIdx.x]);
    }
    
    if (threadIdx.x == 0) {
        uint32_t tmem_addr = 0;
        tmem_dealloc_fn(tmem_addr, 128);
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
    
    create_tma_3d_descriptor_2B(&tma_Q, q_ptr, D, S, B * H, 64, 64, 1, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_3d_descriptor_2B(&tma_K, k_ptr, D, S, B * H, 64, 64, 1, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_3d_descriptor_2B(&tma_V, v_ptr, D, S, B * H, 64, 64, 1, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    __nv_bfloat16* o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_ptr = static_cast<float*>(lse.data_ptr());
    
    dim3 grid((S + 127) / 128, B * H);
    dim3 block(128);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int smem_bytes = 128 * 1024; 
    
    CUDA_CHECK(cudaFuncSetAttribute(mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_bytes;
    config.stream = stream;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    CUDA_CHECK(cudaLaunchKernelEx(&config, mha_kernel, tma_Q, tma_K, tma_V, o_ptr, lse_ptr, S));
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_mha