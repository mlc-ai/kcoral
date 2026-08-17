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

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_load_multicast_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, uint16_t mask) {
    uint64_t cache_hint = 0;
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%0], [%1, {%4, %5}], [%2], %3, %6;"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "h"(mask), "r"(c0), "r"(c1), "l"(cache_hint) : "memory");
}

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
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

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
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

__global__ __launch_bounds__(128, 2) void mha_kernel(const __grid_constant__ CUtensorMap tma_Q,
                          const __grid_constant__ CUtensorMap tma_K,
                          const __grid_constant__ CUtensorMap tma_V,
                          __nv_bfloat16* O, float* LSE, uint32_t S) {
    
    extern __shared__ __align__(128) uint8_t smem_buf[];
    uint64_t* mbar_Q = (uint64_t*)smem_buf;
    uint64_t* mbar_K = (uint64_t*)(smem_buf + 8);
    uint64_t* mbar_V = (uint64_t*)(smem_buf + 16);
    
    // Dynamically align SMEM base to 1024 bytes limit to satisfy swizzle constraints
    uint32_t smem_base = ((uint32_t)__cvta_generic_to_shared(smem_buf) + 1023) & ~1023;
    __nv_bfloat16* Q_0 = (__nv_bfloat16*)(smem_base);         
    __nv_bfloat16* Q_1 = (__nv_bfloat16*)(smem_base + 8192);    
    __nv_bfloat16* P_col = (__nv_bfloat16*)(smem_base + 16384);  
    __nv_bfloat16* K_0 = (__nv_bfloat16*)(smem_base + 24576);   
    __nv_bfloat16* K_1 = (__nv_bfloat16*)(smem_base + 32768);   
    __nv_bfloat16* V_0 = (__nv_bfloat16*)(smem_base + 40960);   
    
    uint32_t* tmem_P = (uint32_t*)(smem_base + 49152);
    uint32_t* tmem_O = (uint32_t*)(smem_base + 49156);
    
    uint32_t i = blockIdx.x;
    uint32_t batch_head_idx = blockIdx.y;
    uint32_t cta_id = cluster_rank_fn() % 2;
    uint32_t coord0_v = cta_id * 64;
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_Q, 2);
        init_smem_barrier_fn(mbar_K, 2);
        init_smem_barrier_fn(mbar_V, 2);
        fence_smem_barrier_init_fn();
        
        tmem_alloc_fn(tmem_P, 64);
        tmem_alloc_fn(tmem_O, 64);
    }
    tmem_load_fence_fn();
    __syncthreads();
    
    int phase_Q = 0;
    uint32_t coord1_q = (i * 64 + 0) + batch_head_idx * S;
    
    if (threadIdx.x == 0 && cta_id == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 4 * 8192);
        tma_load_multicast_2d_fn(&tma_Q, mbar_Q, Q_0, 0, coord1_q, 0x3);
        tma_load_multicast_2d_fn(&tma_Q, mbar_Q, Q_1, 64, coord1_q, 0x3);
    }
    mbarrier_wait_fn(mbar_Q, phase_Q);
    
    float max_val[64], sum_exp[64];
    for (int r = 0; r < 64; ++r) {
        max_val[r] = -1e20f;
        sum_exp[r] = 0.0f;
    }
    
    uint32_t idesc_QK = make_instr_desc_fn(64, 64);
    uint32_t idesc_PV = make_instr_desc_fn(64, 64) | (1U << 16);
    
    uint32_t accum_P = 0;
    uint32_t accum_O = 0;
    
    int phase_K = 0;
    int phase_V = 0;
    
    for (int j = 0; j <= i && i * 64 + 63 >= j * 64; ++j) {
        uint32_t coord1_k = (j * 64 + 0) + batch_head_idx * S;
        uint32_t coord1_v = (j * 64 + 0) + batch_head_idx * S;
        
        if (threadIdx.x == 0 && cta_id == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_K, 4 * 8192);
            tma_load_multicast_2d_fn(&tma_K, mbar_K, K_0, 0, coord1_k, 0x3);
            tma_load_multicast_2d_fn(&tma_K, mbar_K, K_1, 64, coord1_k, 0x3);
        }
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_V, 2 * 8192);
            tma_load_2d_fn(&tma_V, mbar_V, V_0, coord0_v, coord1_v);
        }
        mbarrier_wait_fn(mbar_K, phase_K);
        mbarrier_wait_fn(mbar_V, phase_V);
        
        uint32_t tmem_P_addr = tmem_P[0];
        for (int k = 0; k < 4; ++k) {
            uint64_t desc_a0 = make_smem_desc_sm100_fn(Q_0 + k * 16, 1, 1024);
            uint64_t desc_b0 = make_smem_desc_sm100_fn(K_0 + k * 16, 1, 1024);
            wgmma_cta1(tmem_P_addr, desc_a0, desc_b0, idesc_QK, accum_P);
            accum_P = 1;
        }
        for (int k = 0; k < 4; ++k) {
            uint64_t desc_a1 = make_smem_desc_sm100_fn(Q_1 + k * 16, 1, 1024);
            uint64_t desc_b1 = make_smem_desc_sm100_fn(K_1 + k * 16, 1, 1024);
            wgmma_cta1(tmem_P_addr, desc_a1, desc_b1, idesc_QK, accum_P);
        }
        
        commit_cta1(mbar_P);
        mbarrier_wait_fn(mbar_P, 0); 
        
        for (int step = 0; step < 8; ++step) {
            int row = step / 4;
            int col_chunk = step % 4;
            int col = col_chunk * 16;
            
            uint32_t tmem_addr = (row * 32 << 16) | col;
            
            asm volatile("tcgen05.ld.sync.aligned.16x32bx2.x2.b32 {r0, r1, r2, r3}, [%0], 16;"
                         : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(tmem_addr));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            for(int r_off = 0; r_off < 2; ++r_off) {
                for(int c_off = 0; c_off < 4; ++c_off) {
                    uint32_t val = (c_off == 0) ? r0 : (c_off == 1) ? r1 : (c_off == 2) ? r2 : r3;
                    float f = __uint_as_float(val);
                    int r = row * 32 + (r_off * 16) + (threadIdx.x % 16);
                    int c = col + c_off * 8 + (threadIdx.x % 16) / 2;
                    if (j * 64 + c > i * 64 + r) {
                        P_col[r * 64 + c] = __float2bfloat16(-1e20f);
                    } else {
                        P_col[r * 64 + c] = __float2bfloat16(f / 11.3137f);
                    }
                }
            }
        }
        __syncthreads();
        
        float row_max[64], row_sum[64];
        for (int r = 0; r < 64; ++r) {
            float m = -1e20f;
            for (int c = 0; c < 64; ++c) {
                if (j * 64 + c <= i * 64 + r) {
                    m = fmaxf(m, __bfloat162float(P_col[r * 64 + c]));
                }
            }
            row_max[r] = m;
            
            float s = 0.0f;
            for (int c = 0; c < 64; ++c) {
                if (j * 64 + c <= i * 64 + r) {
                    float p = __bfloat162float(P_col[r * 64 + c]);
                    float exp_p = expf(p - m);
                    s += exp_p;
                    P_col[r * 64 + c] = __float2bfloat16(exp_p);
                } else {
                    P_col[r * 64 + c] = __float2bfloat16(0.0f);
                }
            }
            row_sum[r] = s;
        }
        __syncthreads();
        
        for (int r = 0; r < 64; ++r) {
            float m_prev = max_val[r];
            max_val[r] = fmaxf(m_prev, row_max[r]);
            sum_exp[r] *= expf(m_prev - max_val[r]);
            sum_exp[r] += row_sum[r] * expf(row_max[r] - max_val[r]);
            
            float total_s = sum_exp[r];
            for (int c = 0; c < 64; ++c) {
                P_col[r * 64 + c] = __float2bfloat16(__bfloat162float(P_col[r * 64 + c]) * expf(row_max[r] - max_val[r]) / total_s);
            }
        }
        __syncthreads();
        
        fence_async_shared_fn();
        uint32_t tmem_O_addr = tmem_O[0];
        for (int k = 0; k < 4; ++k) {
            uint64_t desc_a = make_smem_desc_sm100_fn(P_col + k * 16 * 64, 8192, 1024);
            uint64_t desc_b = make_smem_desc_sm100_fn(V_0 + k * 16 * 64, 8192, 1024); 
            
            wgmma_cta1(tmem_O_addr, desc_a, desc_b, idesc_PV, accum_O);
            accum_O = 1;
        }
        commit_cta1(mbar_O);
        mbarrier_wait_fn(mbar_O, 0);
        
        phase_K ^= 1;
        phase_V ^= 1;
    }
    
    for (int step = 0; step < 8; ++step) {
        int row = step / 4;
        int col_chunk = step % 4;
        int col = col_chunk * 16;
        
        uint32_t tmem_addr = (row * 32 << 16) | col;
        
        asm volatile("tcgen05.ld.sync.aligned.16x32bx2.x2.b32 {r0, r1, r2, r3}, [%0], 16;"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(tmem_addr));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        for(int r_off = 0; r_off < 2; ++r_off) {
            for(int c_off = 0; c_off < 4; ++c_off) {
                uint32_t val = (c_off == 0) ? r0 : (c_off == 1) ? r1 : (c_off == 2) ? r2 : r3;
                float f = __uint_as_float(val);
                
                int r_idx = row * 32 + (r_off * 16) + (threadIdx.x % 16);
                int c_idx = c_off * 8 + (threadIdx.x % 16) / 2;
                
                uint32_t global_r = batch_head_idx * S + i * 64 + r_idx;
                uint32_t global_c = cta_id * 64 + c_idx;
                
                if (global_r < S && global_c < 128) {
                    O[(uint64_t)global_r * 128 + global_c] = __float2bfloat16(f);
                }
            }
        }
    }
    
    if (threadIdx.x < 64 && (i * 64 + threadIdx.x) < S) {
        LSE[(i * 64 + threadIdx.x)] = max_val[threadIdx.x] + logf(sum_exp[threadIdx.x]);
    }
    
    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_P[0], 64);
        tmem_dealloc_fn(tmem_O[0], 64);
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
    
    int smem_bytes = 68 * 1024; 
    
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