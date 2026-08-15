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
    d |= (uint64_t)2 << 61;   
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

__device__ __forceinline__ uint32_t make_instr_desc_transpose_b_fn(uint32_t M, uint32_t N) {
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
    uint64_t* mbar_K = (uint64_t*)ptr; ptr += 8;
    uint64_t* mbar_V = (uint64_t*)ptr; ptr += 8;
    uint64_t* mbar_P = (uint64_t*)ptr; ptr += 8;
    uint64_t* mbar_O = (uint64_t*)ptr; ptr += 8;
    
    ptr = (char*)(((uintptr_t)ptr + 1023) & ~1023);
    
    __nv_bfloat16* Q0 = (__nv_bfloat16*)ptr; ptr += 8192;
    __nv_bfloat16* Q1 = (__nv_bfloat16*)ptr; ptr += 8192;
    __nv_bfloat16* K0 = (__nv_bfloat16*)ptr; ptr += 8192;
    __nv_bfloat16* K1 = (__nv_bfloat16*)ptr; ptr += 8192;
    __nv_bfloat16* V0 = (__nv_bfloat16*)ptr; ptr += 8192;
    __nv_bfloat16* V1 = (__nv_bfloat16*)ptr; ptr += 8192;
    __nv_bfloat16* S_mem = (__nv_bfloat16*)ptr; ptr += 8192;

    int b_h = blockIdx.y;
    int s_off = blockIdx.x * 64;
    int s_len_rem = S_len - s_off;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(mbar_K, 1);
        init_smem_barrier_fn(mbar_V, 1);
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
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 16384);
        tma_load_2d_fn(&tma_Q, mbar_Q, Q0, 0, b_h * S_len + s_off);
        tma_load_2d_fn(&tma_Q, mbar_Q, Q1, 64, b_h * S_len + s_off);
    }
    mbarrier_wait_fn(mbar_Q, 0);

    uint64_t desc_Q0 = make_smem_desc_sm100_fn(Q0, 1, 1024);
    uint64_t desc_Q1 = make_smem_desc_sm100_fn(Q1, 1, 1024);
    uint64_t desc_K0 = make_smem_desc_sm100_fn(K0, 8192, 1024);
    uint64_t desc_K1 = make_smem_desc_sm100_fn(K1, 8192, 1024);
    uint64_t desc_V0 = make_smem_desc_sm100_fn(V0, 8192, 1024);
    uint64_t desc_V1 = make_smem_desc_sm100_fn(V1, 8192, 1024);
    uint64_t desc_S = make_smem_desc_sm100_fn(S_mem, 1, 1024);

    uint32_t idesc_P = make_instr_desc_fn(64, 64);
    uint32_t idesc_O = make_instr_desc_transpose_b_fn(64, 64);

    float local_max_val = -1e20f;
    float local_sum_val = 0.0f;
    uint32_t phase = 0;

    const float scale_factor = 1.0f / sqrtf(128);

    for (int kv_off = 0; kv_off < S_len; kv_off += 64) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_K, 16384);
            tma_load_2d_fn(&tma_K, mbar_K, K0, 0, b_h * S_len + kv_off);
            tma_load_2d_fn(&tma_K, mbar_K, K1, 64, b_h * S_len + kv_off);
        }
        mbarrier_wait_fn(mbar_K, phase);

        if (threadIdx.x < 64) {
            for (int k = 0; k < 64; k += 16) {
                uint32_t ptr_Q0_k = (uint32_t)__cvta_generic_to_shared(Q0 + k);
                uint32_t ptr_Q1_k = (uint32_t)__cvta_generic_to_shared(Q1 + k);
                uint32_t ptr_K0_k = (uint32_t)__cvta_generic_to_shared(K0 + k * 64);
                uint32_t ptr_K1_k = (uint32_t)__cvta_generic_to_shared(K1 + k * 64);
                
                uint64_t desc_Q0_k = make_smem_desc_sm100_fn((void*)ptr_Q0_k, 1, 1024);
                uint64_t desc_Q1_k = make_smem_desc_sm100_fn((void*)ptr_Q1_k, 1, 1024);
                uint64_t desc_K0_k = make_smem_desc_sm100_fn((void*)ptr_K0_k, 8192, 1024);
                uint64_t desc_K1_k = make_smem_desc_sm100_fn((void*)ptr_K1_k, 8192, 1024);
                
                if (k == 0) {
                    umma_f16_cg1_fn(P_tmem_base, desc_Q0_k, desc_K0_k, idesc_P, 0);
                } else {
                    umma_f16_cg1_fn(P_tmem_base, desc_Q0_k, desc_K0_k, idesc_P, 1);
                }
                umma_f16_cg1_fn(P_tmem_base, desc_Q1_k, desc_K1_k, idesc_P, 1);
            }
        }
        if (threadIdx.x == 0) {
            cp_commit_1sm_fn(mbar_P);
        }
        mbarrier_wait_fn(mbar_P, phase);

        float row_max = -1e20f;
        float p_val[64];
        
        int lane_id = threadIdx.x % 32;
        int warp_id = threadIdx.x / 32;
        int row_base = warp_id * 32;
        
        for (int col = 0; col < 64; col += 4) {
            uint32_t tmem_addr = ((P_tmem_base & 0xFFFF) + col) | (row_base << 16);
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_addr));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            if (threadIdx.x < 64) {
                float f0 = __uint_as_float(r0) * scale_factor;
                float f1 = __uint_as_float(r1) * scale_factor;
                float f2 = __uint_as_float(r2) * scale_factor;
                float f3 = __uint_as_float(r3) * scale_factor;
                
                int abs_col0 = kv_off + col + 0;
                int abs_col1 = kv_off + col + 1;
                int abs_col2 = kv_off + col + 2;
                int abs_col3 = kv_off + col + 3;
                
                if (abs_col0 >= S_len) f0 = -1e20f;
                if (abs_col1 >= S_len) f1 = -1e20f;
                if (abs_col2 >= S_len) f2 = -1e20f;
                if (abs_col3 >= S_len) f3 = -1e20f;
                
                p_val[col + 0] = f0;
                p_val[col + 1] = f1;
                p_val[col + 2] = f2;
                p_val[col + 3] = f3;
                
                row_max = fmaxf(row_max, fmaxf(fmaxf(f0, f1), fmaxf(f2, f3)));
            }
        }
        
        if (threadIdx.x < 64) {
            float new_max = fmaxf(local_max_val, row_max);
            float scale_prev = expf(local_max_val - new_max);
            local_sum_val *= scale_prev;
            
            for (int col = 0; col < 64; col++) {
                int abs_col = kv_off + col;
                if (abs_col >= S_len) {
                    p_val[col] = -1e20f;
                } else {
                    p_val[col] -= new_max;
                    p_val[col] = expf(p_val[col]);
                    local_sum_val += p_val[col];
                }
                
                int sc_idx = ((threadIdx.x * 64 + col) / 8 ^ (threadIdx.x % 8)) * 8 + (threadIdx.x * 64 + col) % 8;
                S_mem[sc_idx] = __float2bfloat16(p_val[col]);
            }
            
            local_max_val = new_max;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_V, 16384);
            tma_load_2d_fn(&tma_V, mbar_V, V0, 0, b_h * S_len + kv_off);
            tma_load_2d_fn(&tma_V, mbar_V, V1, 64, b_h * S_len + kv_off);
        }
        mbarrier_wait_fn(mbar_V, phase);

        if (threadIdx.x < 64) {
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
        }
        if (threadIdx.x == 0) {
            cp_commit_1sm_fn(mbar_O);
        }
        mbarrier_wait_fn(mbar_O, phase);
        
        phase ^= 1;
    }

    for (int col = 0; col < 128; col += 4) {
        uint32_t tmem_addr = ((O_tmem_base & 0xFFFF) + col) | (row_base << 16);
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_addr));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

        if (threadIdx.x < 64) {
            float f0 = __uint_as_float(r0) / local_sum_val;
            float f1 = __uint_as_float(r1) / local_sum_val;
            float f2 = __uint_as_float(r2) / local_sum_val;
            float f3 = __uint_as_float(r3) / local_sum_val;
            
            uint32_t g_idx = (b_h * S_len + s_off + threadIdx.x) * 128 + col;
            O_gmem[g_idx + 0] = __float2bfloat16(f0);
            O_gmem[g_idx + 1] = __float2bfloat16(f1);
            O_gmem[g_idx + 2] = __float2bfloat16(f2);
            O_gmem[g_idx + 3] = __float2bfloat16(f3);
        }
    }

    if (threadIdx.x < 64) {
        LSE_gmem[b_h * S_len + s_off + threadIdx.x] = local_max_val + logf(local_sum_val);
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
    
    int smem_size = 60000;
    CUDA_CHECK(cudaFuncSetAttribute(run_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    run_kernel<<<grid, threads, smem_size, stream>>>(tma_Q, tma_K, tma_V, O.data_ptr<__nv_bfloat16>(), LSE.data_ptr<float>(), S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha::run);

} // namespace tvm_ffi_mha