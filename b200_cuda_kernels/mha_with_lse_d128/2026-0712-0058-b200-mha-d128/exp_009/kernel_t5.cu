#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
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

#define CU_CHECK(call) do {                                      \
    CUresult _e = (call);                                        \
    if (_e != CUDA_SUCCESS) {                                    \
        fprintf(stderr, "CU error %d at %s:%d\n",                 \
                _e, __FILE__, __LINE__);                         \
        exit(1);                                                 \
    }                                                            \
} while(0)

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_expect_tx_and_arrive_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes));
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

__device__ __forceinline__ void tmem_alloc_fn_1cta(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn_1cta(uint32_t addr, int ncols) {
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

__device__ __forceinline__ void cp_commit_1sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64 [%0];"
        :: "r"(a));
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_cg1_fn(uint32_t M, uint32_t N) {
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

__device__ __forceinline__ uint32_t make_instr_desc_cg1_transpose_b_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (0u << 15);   
    d |= (1u << 16);   
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__global__ __launch_bounds__(128) void run_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O_gmem, float* LSE_gmem, int S_len) 
{
    extern __shared__ __align__(1024) char smem_pool[];
    char* ptr = smem_pool;
    uint64_t* mbar_Q = (uint64_t*)ptr; ptr += 8;
    uint64_t* mbar_K_0 = (uint64_t*)ptr; ptr += 8;
    uint64_t* mbar_K_1 = (uint64_t*)ptr; ptr += 8;
    uint64_t* mbar_V_0 = (uint64_t*)ptr; ptr += 8;
    uint64_t* mbar_V_1 = (uint64_t*)ptr; ptr += 8;
    uint64_t* mbar_P = (uint64_t*)ptr; ptr += 8;
    uint64_t* mbar_O = (uint64_t*)ptr; ptr += 8;
    
    ptr = (char*)(((uintptr_t)ptr + 1023) & ~1023);
    
    __nv_bfloat16* Q0 = (__nv_bfloat16*)ptr; ptr += 8192;
    __nv_bfloat16* Q1 = (__nv_bfloat16*)ptr; ptr += 8192;
    __nv_bfloat16* K0 = (__nv_bfloat16*)ptr; ptr += 8192;
    __nv_bfloat16* K1 = (__nv_bfloat16*)ptr; ptr += 8192;
    __nv_bfloat16* V0 = (__nv_bfloat16*)ptr; ptr += 8192;
    __nv_bfloat16* V1 = (__nv_bfloat16*)ptr; ptr += 8192;
    __nv_bfloat16* K0_next = (__nv_bfloat16*)ptr; ptr += 8192;
    __nv_bfloat16* K1_next = (__nv_bfloat16*)ptr; ptr += 8192;
    __nv_bfloat16* V0_next = (__nv_bfloat16*)ptr; ptr += 8192;
    __nv_bfloat16* V1_next = (__nv_bfloat16*)ptr; ptr += 8192;
    __nv_bfloat16* S_mem = (__nv_bfloat16*)ptr; ptr += 8192;

    int b_h = blockIdx.y;
    int s_off = blockIdx.x * 64;

    auto coord_q = [&](int s) -> int { return b_h * S_len + s; };
    auto coord_k = [&](int s) -> int { return b_h * S_len + s; };
    auto coord_v = [&](int s) -> int { return b_h * S_len + s; };

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(mbar_K_0, 1);
        init_smem_barrier_fn(mbar_K_1, 1);
        init_smem_barrier_fn(mbar_V_0, 1);
        init_smem_barrier_fn(mbar_V_1, 1);
        init_smem_barrier_fn(mbar_P, 1);
        init_smem_barrier_fn(mbar_O, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t P_tmem_base, O_tmem_base;
    if (threadIdx.x == 0) {
        tmem_alloc_fn_1cta(&P_tmem_base, 64);
        tmem_alloc_fn_1cta(&O_tmem_base, 128);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        mbarrier_expect_tx_and_arrive_fn(mbar_Q, 16384);
        tma_load_2d_fn(&tma_Q, mbar_Q, Q0, 0, coord_q(s_off));
        tma_load_2d_fn(&tma_Q, mbar_Q, Q1, 64, coord_q(s_off));
        
        mbarrier_expect_tx_and_arrive_fn(mbar_K_0, 16384);
        tma_load_2d_fn(&tma_K, mbar_K_0, K0, 0, coord_k(0));
        tma_load_2d_fn(&tma_K, mbar_K_0, K1, 64, coord_k(0));
        
        mbarrier_expect_tx_and_arrive_fn(mbar_V_0, 16384);
        tma_load_2d_fn(&tma_V, mbar_V_0, V0, 0, coord_v(0));
        tma_load_2d_fn(&tma_V, mbar_V_0, V1, 64, coord_v(0));
    }

    uint32_t idesc_P = make_instr_desc_cg1_fn(64, 64);
    uint32_t idesc_O = make_instr_desc_cg1_transpose_b_fn(64, 64);

    float local_max_val = -1e20f;
    float local_sum_val = 0.0f;
    uint32_t phase_kv = 0;
    uint32_t phase_p = 0;
    uint32_t phase_o = 0;

    const float scale_factor = 1.0f / sqrtf(128);
    int tid = threadIdx.x;

    for (int kv_off = 0; kv_off < S_len; kv_off += 64) {
        uint32_t my_phase = phase_kv;
        uint64_t* cur_mbar_K = (phase_kv == 0) ? mbar_K_0 : mbar_K_1;
        uint64_t* cur_mbar_V = (phase_kv == 0) ? mbar_V_0 : mbar_V_1;
        
        mbarrier_wait_fn(cur_mbar_K, my_phase);
        mbarrier_wait_fn(cur_mbar_V, my_phase);

        if (threadIdx.x < 64) {
            if (threadIdx.x == 0) {
                mbarrier_expect_tx_and_arrive_fn(mbar_P, 8192);
            }
            __syncthreads(); 
            
            for (int k = 0; k < 64; k += 16) {
                uint32_t ptr_Q0_k = (uint32_t)__cvta_generic_to_shared(Q0 + k);
                uint32_t ptr_Q1_k = (uint32_t)__cvta_generic_to_shared(Q1 + k);
                uint32_t ptr_K0_k = (uint32_t)__cvta_generic_to_shared(K0 + k);
                uint32_t ptr_K1_k = (uint32_t)__cvta_generic_to_shared(K1 + k);
                
                uint64_t desc_Q0_k = make_smem_desc_sm100_fn((void*)ptr_Q0_k, 1, 1024);
                uint64_t desc_Q1_k = make_smem_desc_sm100_fn((void*)ptr_Q1_k, 1, 1024);
                uint64_t desc_K0_k = make_smem_desc_sm100_fn((void*)ptr_K0_k, 1, 1024);
                uint64_t desc_K1_k = make_smem_desc_sm100_fn((void*)ptr_K1_k, 1, 1024);
                
                if (k == 0) {
                    umma_f16_cg1_fn(P_tmem_base, desc_Q0_k, desc_K0_k, idesc_P, 0);
                } else {
                    umma_f16_cg1_fn(P_tmem_base, desc_Q0_k, desc_K0_k, idesc_P, 1);
                }
                umma_f16_cg1_fn(P_tmem_base, desc_Q1_k, desc_K1_k, idesc_P, 1);
            }
            if (threadIdx.x == 0) {
                cp_commit_1sm_fn(mbar_P);
            }
        }
        
        if (tid < 64) {
            mbarrier_wait_fn(mbar_P, phase_p);

            for (int col = 0; col < 64; col += 4) {
                uint32_t tmem_addr = (P_tmem_base & 0xFFFF) + col | (tid << 16);
                uint32_t r0, r1, r2, r3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_addr));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            }

            float row_max = -1e20f;
            float p_val[64];
            
            for (int col = 0; col < 64; col++) {
                uint32_t tmem_addr = (P_tmem_base & 0xFFFF) + col | (tid << 16);
                uint32_t r0, r1, r2, r3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_addr));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

                float f = __uint_as_float(r0) * scale_factor;
                int abs_col = kv_off + col;
                if (abs_col >= S_len) f = -1e20f;
                
                p_val[col] = f;
                row_max = fmaxf(row_max, f);
            }
            
            float new_max = fmaxf(local_max_val, row_max);
            float scale_prev = expf(local_max_val - new_max);
            local_sum_val *= scale_prev;
            
            for (int col = 0; col < 64; col++) {
                int abs_col = kv_off + col;
                if (abs_col >= S_len) {
                    p_val[col] = 0.0f;
                } else {
                    p_val[col] -= new_max;
                    p_val[col] = expf(p_val[col]);
                    local_sum_val += p_val[col];
                }
                int sc_idx = tid * 64 + (((tid % 8) ^ (col / 8)) * 8 + (col % 8));
                S_mem[sc_idx] = __float2bfloat16(p_val[col]);
            }
            local_max_val = new_max;
        }
        __syncthreads();

        if (threadIdx.x < 64) {
            if (threadIdx.x == 0) {
                mbarrier_expect_tx_and_arrive_fn(mbar_O, 16384);
            }
            
            for (int k = 0; k < 64; k += 16) {
                uint32_t ptr_S_k = (uint32_t)__cvta_generic_to_shared(S_mem + k);
                uint32_t ptr_V0_k = (uint32_t)__cvta_generic_to_shared(V0 + k * 64);
                uint32_t ptr_V1_k = (uint32_t)__cvta_generic_to_shared(V1 + k * 64);
                
                uint64_t desc_S_k = make_smem_desc_sm100_fn((void*)ptr_S_k, 1, 1024);
                uint64_t desc_V0_k = make_smem_desc_sm100_fn((void*)ptr_V0_k, 8192, 1024);
                uint64_t desc_V1_k = make_smem_desc_sm100_fn((void*)ptr_V1_k, 8192, 1024);
                
                if (k == 0) {
                    umma_f16_cg1_fn(O_tmem_base, desc_S_k, desc_V0_k, idesc_O, 0);
                    umma_f16_cg1_fn(O_tmem_base + 64, desc_S_k, desc_V1_k, idesc_O, 0);
                } else {
                    umma_f16_cg1_fn(O_tmem_base, desc_S_k, desc_V0_k, idesc_O, 1);
                    umma_f16_cg1_fn(O_tmem_base + 64, desc_S_k, desc_V1_k, idesc_O, 1);
                }
            }
            if (threadIdx.x == 0) {
                cp_commit_1sm_fn(mbar_O);
            }
        }
        
        if (tid < 64) {
            mbarrier_wait_fn(mbar_O, phase_o);
        }
        
        __syncthreads();
        if (kv_off + 64 < S_len) {
            bool is_next_kv = (phase_kv == 0);
            uint64_t* next_mbar_K = (phase_kv == 0) ? mbar_K_1 : mbar_K_0;
            uint64_t* next_mbar_V = (phase_kv == 0) ? mbar_V_1 : mbar_V_0;
            
            if (threadIdx.x == 0) {
                mbarrier_expect_tx_and_arrive_fn(next_mbar_K, 16384);
                tma_load_2d_fn(&tma_K, next_mbar_K, is_next_kv ? K0_next : K0, 0, coord_k(kv_off + 64));
                tma_load_2d_fn(&tma_K, next_mbar_K, is_next_kv ? K1_next : K1, 64, coord_k(kv_off + 64));
                
                mbarrier_expect_tx_and_arrive_fn(next_mbar_V, 16384);
                tma_load_2d_fn(&tma_V, next_mbar_V, is_next_kv ? V0_next : V0, 0, coord_v(kv_off + 64));
                tma_load_2d_fn(&tma_V, next_mbar_V, is_next_kv ? V1_next : V1, 64, coord_v(kv_off + 64));
            }
        }
        
        if (phase_kv == 0) {
            __nv_bfloat16* tmp_K0 = K0; K0 = K0_next; K0_next = tmp_K0;
            __nv_bfloat16* tmp_K1 = K1; K1 = K1_next; K1_next = tmp_K1;
            __nv_bfloat16* tmp_V0 = V0; V0 = V0_next; V0_next = tmp_V0;
            __nv_bfloat16* tmp_V1 = V1; V1 = V1_next; V1_next = tmp_V1;
        }
        
        phase_kv ^= 1;
        phase_p ^= 1;
        phase_o ^= 1;
    }

    for (int half = 0; half < 2; half++) {
        for (int col = 0; col < 64; col += 4) {
            int swizzled_col = (((tid % 8) ^ (col / 8)) * 8 + (col % 8));
            uint32_t tmem_base = half ? (O_tmem_base + 64) : O_tmem_base;
            uint32_t tmem_addr = (tmem_base & 0xFFFF) + swizzled_col | (tid << 16);
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_addr));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

            if (tid < 64) {
                float f0 = __uint_as_float(r0) / local_sum_val;
                float f1 = __uint_as_float(r1) / local_sum_val;
                float f2 = __uint_as_float(r2) / local_sum_val;
                float f3 = __uint_as_float(r3) / local_sum_val;
                
                int row = s_off + tid;
                int g_col0 = (half ? 64 : 0) + col + 0;
                int g_col1 = (half ? 64 : 0) + col + 1;
                int g_col2 = (half ? 64 : 0) + col + 2;
                int g_col3 = (half ? 64 : 0) + col + 3;
                
                if (row < S_len) {
                    uint64_t g_idx0 = (uint64_t)(b_h * S_len + row) * 128 + g_col0;
                    uint64_t g_idx1 = (uint64_t)(b_h * S_len + row) * 128 + g_col1;
                    uint64_t g_idx2 = (uint64_t)(b_h * S_len + row) * 128 + g_col2;
                    uint64_t g_idx3 = (uint64_t)(b_h * S_len + row) * 128 + g_col3;
                    O_gmem[g_idx0] = __float2bfloat16(f0);
                    O_gmem[g_idx1] = __float2bfloat16(f1);
                    O_gmem[g_idx2] = __float2bfloat16(f2);
                    O_gmem[g_idx3] = __float2bfloat16(f3);
                }
            }
        }
    }

    int lse_row = s_off + tid;
    if (tid < 64 && lse_row < S_len) {
        LSE_gmem[b_h * S_len + lse_row] = local_max_val + logf(local_sum_val);
    }

    if (threadIdx.x == 0) {
        tmem_dealloc_fn_1cta(P_tmem_base, 64);
        tmem_dealloc_fn_1cta(O_tmem_base, 128);
    }
}

namespace tvm_ffi_mha {

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
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3); 
    
    CUtensorMap tma_Q, tma_K, tma_V;
    create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), D, B * H * S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), D, B * H * S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), D, B * H * S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    int64_t threads = 128;
    int64_t blocks_x = (S + 63) / 64;
    dim3 grid(blocks_x, B * H);
    dim3 block(threads);
    
    int smem_size = 65536;
    CUDA_CHECK(cudaFuncSetAttribute(run_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    run_kernel<<<grid, block, smem_size, stream>>>(tma_Q, tma_K, tma_V, static_cast<__nv_bfloat16*>(O.data_ptr()), static_cast<float*>(LSE.data_ptr()), S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha::run);

} // namespace tvm_ffi_mha