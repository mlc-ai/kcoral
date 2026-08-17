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

#define DRV_CHECK(call) do {                                       \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        const char* err_name;                                      \
        cuGetErrorName(_e, &err_name);                             \
        fprintf(stderr, "CUDA driver error %s at %s:%d\n",         \
                err_name, __FILE__, __LINE__);                     \
        exit(1);                                                   \
    }                                                              \
} while(0)


__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(phase));
}

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(bar)),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
}

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_swizzled(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61; // 128B SWIZZLE
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

__device__ __forceinline__ void wgmma_cta2(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void commit_cta2(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile(
        "tcgen05.commit.cta_group::2"
        ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;"
        :: "r"(a), "h"((uint16_t)0x3));
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
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
       :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
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
                          const __grid_constant__ CUtensorMap tma_O,
                          float* LSE, uint32_t S) {
    
    extern __shared__ __align__(16) uint8_t smem_buf[];
    uint64_t* mbar_Q = (uint64_t*)smem_buf;
    uint64_t* mbar_KV = (uint64_t*)(smem_buf + 8);
    
    uint32_t smem_base = ((uint32_t)__cvta_generic_to_shared(smem_buf) + 1023) & ~1023;
    __nv_bfloat16* Q_0 = (__nv_bfloat16*)(smem_base);         
    __nv_bfloat16* Q_1 = (__nv_bfloat16*)(smem_base + 8192);    
    __nv_bfloat16* K_0[2] = { (__nv_bfloat16*)(smem_base + 16384), (__nv_bfloat16*)(smem_base + 49152) };
    __nv_bfloat16* K_1[2] = { (__nv_bfloat16*)(smem_base + 24576), (__nv_bfloat16*)(smem_base + 57344) };
    __nv_bfloat16* V_0[2] = { (__nv_bfloat16*)(smem_base + 32768), (__nv_bfloat16*)(smem_base + 65536) };
    __nv_bfloat16* V_1[2] = { (__nv_bfloat16*)(smem_base + 40960), (__nv_bfloat16*)(smem_base + 73728) };
    __nv_bfloat16* P_col = (__nv_bfloat16*)(smem_base + 81920);   
    
    float* P_fp32 = (float*)(smem_base + 90112); 
    
    uint32_t i = blockIdx.x;
    uint32_t batch_head_idx = blockIdx.y;
    uint32_t global_i = i * 128;
    uint32_t cta_id = cluster_rank_fn() % 2;
    uint32_t local_q_start = global_i + cta_id * 64;
    
    if (local_q_start >= S) return;
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_Q, 2);
        init_smem_barrier_fn(mbar_KV, 2);
        fence_smem_barrier_init_fn();
        
        uint32_t* tmem_P = (uint32_t*)(smem_base + 106496);
        tmem_alloc_fn(tmem_P, 64);
        
        uint32_t* tmem_O = (uint32_t*)(smem_base + 106500);
        tmem_alloc_fn(tmem_O, 128);
    }
    __syncthreads();
    
    int phase_Q = 0;
    int phase_K[2] = {0, 0};
    int phase_V[2] = {0, 0};
    
    uint32_t coord1_q = local_q_start;
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 4 * 8192);
        tma_load_3d_fn(&tma_Q, mbar_Q, Q_0, 0, coord1_q, batch_head_idx);
        tma_load_3d_fn(&tma_Q, mbar_Q, Q_1, 64, coord1_q, batch_head_idx);
    }
    mbarrier_wait_fn(mbar_Q, phase_Q);
    phase_Q ^= 1;
    
    float max_val[64], sum_exp[64];
    for (int r = threadIdx.x; r < 64; r += blockDim.x) {
        max_val[r] = -1e20f;
        sum_exp[r] = 0.0f;
    }
    __syncthreads();
    
    uint32_t p_base = 0;
    uint32_t o_base = 64;
    
    // Initial load of KV block 0
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_KV, 4 * 8192);
        tma_load_3d_fn(&tma_K, mbar_KV, K_0[0], 0, 0, batch_head_idx);
        tma_load_3d_fn(&tma_K, mbar_KV, K_1[0], 64, 0, batch_head_idx);
        mbarrier_arrive_and_expect_tx_fn(mbar_KV, 4 * 8192);
        tma_load_3d_fn(&tma_V, mbar_KV, V_0[0], 0, 0, batch_head_idx);
        tma_load_3d_fn(&tma_V, mbar_KV, V_1[0], 64, 0, batch_head_idx);
    }
    
    int max_j = -1;
    
    for (int j = 0; j <= global_i && global_i * 64 + 63 >= j * 64; ++j) {
        int buf_idx = j % 2;
        int next_buf_idx = (j + 1) % 2;
        uint64_t* next_mbar_KV = (next_buf_idx == 0) ? mbar_KV : mbar_KV;
        
        if (j + 1 <= global_i && global_i * 64 + 63 >= (j + 1) * 64) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(next_mbar_KV, 4 * 8192);
                tma_load_3d_fn(&tma_K, next_mbar_KV, K_0[next_buf_idx], 0, (j + 1) * 64, batch_head_idx);
                tma_load_3d_fn(&tma_K, next_mbar_KV, K_1[next_buf_idx], 64, (j + 1) * 64, batch_head_idx);
                
                mbarrier_arrive_and_expect_tx_fn(next_mbar_KV, 4 * 8192);
                tma_load_3d_fn(&tma_V, next_mbar_KV, V_0[next_buf_idx], 0, (j + 1) * 64, batch_head_idx);
                tma_load_3d_fn(&tma_V, next_mbar_KV, V_1[next_buf_idx], 64, (j + 1) * 64, batch_head_idx);
            }
        }
        
        mbarrier_wait_fn(mbar_KV, phase_K[buf_idx]);
        phase_K[buf_idx] ^= 1;
        mbarrier_wait_fn(mbar_KV, phase_V[buf_idx]);
        phase_V[buf_idx] ^= 1;
        
        __nv_bfloat16* K_0_buf = K_0[buf_idx];
        __nv_bfloat16* K_1_buf = K_1[buf_idx];
        __nv_bfloat16* V_0_buf = V_0[buf_idx];
        __nv_bfloat16* V_1_buf = V_1[buf_idx];
        
        uint32_t accum_P = 0;
        
        for (int k = 0; k < 4; ++k) {
            uint64_t desc_a0 = make_smem_desc_swizzled(Q_0 + k * 16, 1, 1024);
            uint64_t desc_b0 = make_smem_desc_swizzled(K_0_buf + k * 16, 1, 1024);
            wgmma_cta2(p_base + k * 1024, desc_a0, desc_b0, make_instr_desc_fn(64, 64), accum_P);
            accum_P = 1;
        }
        for (int k = 0; k < 4; ++k) {
            uint64_t desc_a1 = make_smem_desc_swizzled(Q_1 + k * 16, 1, 1024);
            uint64_t desc_b1 = make_smem_desc_swizzled(K_1_buf + k * 16, 1, 1024);
            wgmma_cta2(p_base + k * 1024, desc_a1, desc_b1, make_instr_desc_fn(64, 64), accum_P);
        }
        
        commit_cta2(mbar_Q);
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
            if (j * 64 + c > local_q_start + r) {
                P_fp32[idx] = -1e20f;
            } else {
                P_fp32[idx] *= 0.08838834764f; // 1.0 / sqrt(128.0)
            }
        }
        
        float row_max[64], row_sum[64];
        for (int r = threadIdx.x; r < 64; r += blockDim.x) {
            float m = -1e20f;
            for (int c = 0; c < 64; ++c) {
                if (j * 64 + c <= local_q_start + r) {
                    m = fmaxf(m, P_fp32[r * 64 + c]);
                }
            }
            row_max[r] = m;
            
            float s = 0.0f;
            for (int c = 0; c < 64; ++c) {
                if (j * 64 + c <= local_q_start + r) {
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
            int swizzled_x = (r % 8) ^ (c / 8);
            P_col[r * 64 + swizzled_x * 8 + (c % 8)] = __float2bfloat16(P_fp32[idx]);
        }
        __syncthreads(); 
        
        fence_async_shared_fn();
        
        uint32_t accum_O = 1;
        for (int k = 0; k < 4; ++k) {
            uint64_t desc_a = make_smem_desc_swizzled(P_col + k * 16, 1, 1024);
            uint64_t desc_b0 = make_smem_desc_swizzled(V_0_buf + k * 16 * 64, 8192, 1024); 
            
            wgmma_cta2(o_base + k * 1024, desc_a, desc_b0, make_instr_desc_fn(64, 64) | (1U << 16), accum_O);
        }
        for (int k = 0; k < 4; ++k) {
            uint64_t desc_a = make_smem_desc_swizzled(P_col + k * 16, 1, 1024);
            uint64_t desc_b1 = make_smem_desc_swizzled(V_1_buf + k * 16 * 64, 8192, 1024);
            
            wgmma_cta2(o_base + 64 + k * 1024, desc_a, desc_b1, make_instr_desc_fn(64, 64) | (1U << 16), accum_O);
        }
        
        commit_cta2(mbar_Q);
        mbarrier_wait_fn(mbar_Q, phase_Q);
        phase_Q ^= 1;
        
        if (j == global_i) {
            max_j = j;
        }
    }
    
    // Epilogue: Store O directly to GMEM using TMA
    for (int step = 0; step < 16; ++step) {
        int row = (step / 4) * 16 + (threadIdx.x % 64);
        int col = (step % 4) * 16 + (threadIdx.x / 64) * 2;
        
        int swizzled_x = (row % 8) ^ (col / 8);
        
        uint32_t tmem_addr = (row << 16) | (64 + col);
        
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn(tmem_addr, &r0, &r1, &r2, &r3);
        tmem_load_fence_fn();
        
        float f0 = __uint_as_float(r0);
        float f1 = __uint_as_float(r1);
        float f2 = __uint_as_float(r2);
        float f3 = __uint_as_float(r3);
        
        int real_col_0 = swizzled_x * 8 + (col % 8) + 0;
        int real_col_1 = swizzled_x * 8 + (col % 8) + 1;
        int real_col_2 = swizzled_x * 8 + (col % 8) + 2;
        int real_col_3 = swizzled_x * 8 + (col % 8) + 3;
        
        if ((global_i * 64 + row) < S) {
            uint64_t base = (uint64_t)(batch_head_idx * S + global_i * 64 + row) * 128;
            __nv_bfloat16* O_ptr = static_cast<__nv_bfloat16*>(tma_O.globalAddress);
            O_ptr[base + real_col_0] = __float2bfloat16(f0);
            O_ptr[base + real_col_1] = __float2bfloat16(f1);
            O_ptr[base + real_col_2] = __float2bfloat16(f2);
            O_ptr[base + real_col_3] = __float2bfloat16(f3);
            
            if (real_col_0 + 64 < 128) {
                O_ptr[base + real_col_0 + 64] = __float2bfloat16(f0);
                O_ptr[base + real_col_1 + 64] = __float2bfloat16(f1);
                O_ptr[base + real_col_2 + 64] = __float2bfloat16(f2);
                O_ptr[base + real_col_3 + 64] = __float2bfloat16(f3);
            }
        }
    }
    
    if ((threadIdx.x % 32) < 64 && (local_q_start + threadIdx.x % 32) < S) {
        LSE[(uint64_t)batch_head_idx * S + local_q_start + threadIdx.x % 32] = max_val[threadIdx.x % 32] + logf(sum_exp[threadIdx.x % 32]);
    }
    
    if (threadIdx.x == 0) {
        uint32_t tmem_P_addr = 0;
        tmem_dealloc_fn(tmem_P_addr, 64);
        
        uint32_t tmem_O_addr = 64;
        tmem_dealloc_fn(tmem_O_addr, 128);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView lse) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    uint32_t B = Q.size(0), H = Q.size(1), S = Q.size(2), D = Q.size(3);
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O;
    __nv_bfloat16* q_ptr = static_cast<__nv_bfloat16*>(Q.data_ptr());
    __nv_bfloat16* k_ptr = static_cast<__nv_bfloat16*>(K.data_ptr());
    __nv_bfloat16* v_ptr = static_cast<__nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    
    DRV_CHECK(create_tma_3d_descriptor_2B(&tma_Q, q_ptr, D, S, B * H, 64, 64, 1, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    DRV_CHECK(create_tma_3d_descriptor_2B(&tma_K, k_ptr, D, S, B * H, 64, 64, 1, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    DRV_CHECK(create_tma_3d_descriptor_2B(&tma_V, v_ptr, D, S, B * H, 64, 64, 1, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    DRV_CHECK(create_tma_3d_descriptor_2B(&tma_O, o_ptr, D, S, B * H, 64, 64, 1, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
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

    CUDA_CHECK(cudaLaunchKernelEx(&config, mha_kernel, tma_Q, tma_K, tma_V, tma_O, lse_ptr, S));
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_mha