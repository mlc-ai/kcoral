#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <stdio.h>
#include <cmath>
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

#define CU_CHECK(call) do {                                        \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        fprintf(stderr, "CU error %d at %s:%d\n",                 \
                _e, __FILE__, __LINE__);                           \
        exit(1);                                                   \
    }                                                              \
} while(0)

// -------------------------------------------------------------------------
// Device helper functions
// -------------------------------------------------------------------------

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
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

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
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

__device__ __forceinline__ void umma_f16_cg1_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_1sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64 [%0];"
        :: "r"(a));
}

__device__ __forceinline__ uint64_t make_smem_desc_k_major(void* smem_base, void* ptr) {
    uint32_t base = (uint32_t)__cvta_generic_to_shared(smem_base);
    uint32_t p = (uint32_t)__cvta_generic_to_shared(ptr);
    uint32_t offset = p - base;
    uint32_t lbo = 1;
    uint32_t sbo = 1024;
    uint64_t d = 0;
    d |= (uint64_t)(base & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    uint32_t base_offset = (offset >> 7) & 0x7;
    d |= (uint64_t)base_offset << 49;
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_n_major(void* smem_base, void* ptr) {
    uint32_t base = (uint32_t)__cvta_generic_to_shared(smem_base);
    uint32_t p = (uint32_t)__cvta_generic_to_shared(ptr);
    uint32_t offset = p - base;
    uint32_t sbo = 1024;
    uint32_t lbo = 8192;
    uint64_t d = 0;
    d |= (uint64_t)(base & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    uint32_t base_offset = (offset >> 7) & 0x7;
    d |= (uint64_t)base_offset << 49;
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_p(void* smem_base, void* ptr) {
    uint32_t base = (uint32_t)__cvta_generic_to_shared(smem_base);
    uint32_t p = (uint32_t)__cvta_generic_to_shared(ptr);
    uint32_t offset = p - base;
    uint32_t lbo = 1024;
    uint32_t sbo = 128;
    uint64_t d = 0;
    d |= (uint64_t)(base & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)0 << 61; 
    uint32_t base_offset = (offset >> 7) & 0x7;
    d |= (uint64_t)base_offset << 49;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_f16(uint32_t M, uint32_t N, bool a_major, bool b_major) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= ((a_major ? 1 : 0) << 15);
    d |= ((b_major ? 1 : 0) << 16);
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

CUresult create_tma_2d_descriptor_BF16(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim) {
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
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

// -------------------------------------------------------------------------
// Attention Kernel
// -------------------------------------------------------------------------

__global__ void attention_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O,
    float* LSE,
    int S_len) 
{
    setmaxnreg_inc_sync_fn<256>();

    int bh = blockIdx.y; 
    int s_idx = blockIdx.x * 64; 
    int num_S_tiles = (S_len + 63) / 64;
    uint32_t tid = threadIdx.x;

    extern __shared__ __align__(1024) char smem_pool[];
    char* ptr = smem_pool;

    ptr = (char*)(((uintptr_t)ptr + 1023) & ~(uintptr_t)1023);
    __nv_bfloat16* smem_Q0 = (__nv_bfloat16*)ptr; ptr += 8192;
    __nv_bfloat16* smem_Q1 = (__nv_bfloat16*)ptr; ptr += 8192;

    ptr = (char*)(((uintptr_t)ptr + 1023) & ~(uintptr_t)1023);
    __nv_bfloat16* smem_K0 = (__nv_bfloat16*)ptr; ptr += 8192;
    __nv_bfloat16* smem_K1 = (__nv_bfloat16*)ptr; ptr += 8192;

    ptr = (char*)(((uintptr_t)ptr + 1023) & ~(uintptr_t)1023);
    __nv_bfloat16* smem_V0 = (__nv_bfloat16*)ptr; ptr += 8192;
    __nv_bfloat16* smem_V1 = (__nv_bfloat16*)ptr; ptr += 8192;

    ptr = (char*)(((uintptr_t)ptr + 1023) & ~(uintptr_t)1023);
    __nv_bfloat16* smem_P = (__nv_bfloat16*)ptr; ptr += 8192;

    ptr = (char*)(((uintptr_t)ptr + 7) & ~(uintptr_t)7);
    uint64_t* mbar_Q = (uint64_t*)ptr; ptr += 8;
    uint64_t* mbar_K = (uint64_t*)ptr; ptr += 8;
    uint64_t* mbar_V = (uint64_t*)ptr; ptr += 8;
    uint64_t* mbar_S = (uint64_t*)ptr; ptr += 8;
    uint64_t* mbar_O = (uint64_t*)ptr; ptr += 8;

    if (tid == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(mbar_K, 1);
        init_smem_barrier_fn(mbar_V, 1);
        init_smem_barrier_fn(mbar_S, 1);
        init_smem_barrier_fn(mbar_O, 1);
    }
    fence_proxy_async_fn();
    __syncthreads();

    uint32_t s_tmem_addr, p_tmem_addr, o_tmem_addr;
    if (tid == 0) {
        tmem_alloc_fn((uint32_t*)&s_tmem_addr, 64);
        tmem_alloc_fn((uint32_t*)&p_tmem_addr, 64);
        tmem_alloc_fn((uint32_t*)&o_tmem_addr, 128);
    }
    __syncthreads();

    int row_offset = bh * S_len + s_idx;
    uint32_t phase_Q = 0;

    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 16384); 
        tma_load_2d_fn(&tma_Q, mbar_Q, smem_Q0, 0, row_offset);         
        tma_load_2d_fn(&tma_Q, mbar_Q, smem_Q1, 64, row_offset);         
    }
    mbarrier_wait_fn(mbar_Q, phase_Q);

    float O_reg[128];
    for (int i=0; i<128; i++) O_reg[i] = 0.0f;

    uint32_t phase_K = 0, phase_V = 0, phase_S = 0, phase_O = 0;
    uint32_t idesc_qk = make_instr_desc_f16(64, 64, false, false);
    uint32_t idesc_pv = make_instr_desc_f16(64, 64, false, true);

    float m_i = -1e20f;
    float l_i = 0.0f;

    for (int j = 0; j < num_S_tiles; j++) {
        int k_idx = j * 64;
        int k_row_offset = bh * S_len + k_idx;

        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_K, 16384);
            tma_load_2d_fn(&tma_K, mbar_K, smem_K0, 0, k_row_offset);          
            tma_load_2d_fn(&tma_K, mbar_K, smem_K1, 64, k_row_offset);

            mbarrier_arrive_and_expect_tx_fn(mbar_V, 16384);
            tma_load_2d_fn(&tma_V, mbar_V, smem_V0, 0, k_row_offset);          
            tma_load_2d_fn(&tma_V, mbar_V, smem_V1, 64, k_row_offset);
        }
        mbarrier_wait_fn(mbar_K, phase_K);
        mbarrier_wait_fn(mbar_V, phase_V);
        phase_K ^= 1;
        phase_V ^= 1;

        fence_proxy_async_fn();

        if (tid == 0) {
            for (int k = 0; k < 4; ++k) {
                uint64_t desc_a = make_smem_desc_k_major(smem_Q0, (__nv_bfloat16*)smem_Q0 + k * 16);
                uint64_t desc_b = make_smem_desc_k_major(smem_K0, (__nv_bfloat16*)smem_K0 + k * 16);
                umma_f16_cg1_fn(s_tmem_addr, desc_a, desc_b, idesc_qk, k == 0 ? 0 : 1);
            }
            for (int k = 0; k < 4; ++k) {
                uint64_t desc_a = make_smem_desc_k_major(smem_Q1, (__nv_bfloat16*)smem_Q1 + k * 16);
                uint64_t desc_b = make_smem_desc_k_major(smem_K1, (__nv_bfloat16*)smem_K1 + k * 16);
                umma_f16_cg1_fn(s_tmem_addr, desc_a, desc_b, idesc_qk, 1);
            }
        }

        if (tid == 0) {
            umma_commit_1sm_fn(mbar_S); 
        }
        mbarrier_wait_fn(mbar_S, phase_S);
        phase_S ^= 1;

        float my_m = -1e20f;
        float val[64];
        
        for (int col = 0; col < 64; col += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(s_tmem_addr + col));
            val[col+0] = __uint_as_float(r0);
            val[col+1] = __uint_as_float(r1);
            val[col+2] = __uint_as_float(r2);
            val[col+3] = __uint_as_float(r3);
        }
        tmem_load_fence_fn();

        for(int i=0; i<64; ++i) {
            int g_col = k_idx + i;
            if (g_col >= S_len || s_idx + tid >= S_len) {
                val[i] = -1e20f;
            } else {
                val[i] *= sqrtf(1.0f / 128.0f);
            }
            my_m = fmaxf(my_m, val[i]);
        }

        float my_new_m = fmaxf(m_i, my_m);
        float my_alpha = expf(m_i - my_new_m);
        float my_beta = expf(my_m - my_new_m);

        float my_new_l = l_i * my_alpha;
        for(int i=0; i<64; ++i) {
            val[i] = fast_exp2f_fn((val[i] - my_new_m) * 1.4426950408889634f) * my_beta;
            my_new_l += val[i];
        }

        l_i = my_new_l;
        m_i = my_new_m;

        for(int i=0; i<128; ++i) {
            O_reg[i] *= my_alpha;
        }

        float val_p[64];
        for(int i=0; i<64; ++i) {
            val_p[i] = val[i];
        }
        
        for(int i=0; i<64; ++i) {
            int4* smem_P_int4 = (int4*)smem_P;
            int4 p_val;
            ((__nv_bfloat16*)&p_val)[0] = __float2bfloat16(val_p[i]);
            ((__nv_bfloat16*)&p_val)[1] = __float2bfloat16(val_p[i+1]);
            smem_P_int4[(tid * 64 + i) / 8] = p_val;
        }
        __syncthreads();
        fence_proxy_async_fn();

        if (tid == 0) {
            for (int k_step = 0; k_step < 4; ++k_step) {
                uint64_t desc_a = make_smem_desc_p(smem_P, (__nv_bfloat16*)smem_P + k_step * 16);
                uint64_t desc_b0 = make_smem_desc_n_major(smem_V0, (__nv_bfloat16*)smem_V0 + k_step * 1024);
                uint64_t desc_b1 = make_smem_desc_n_major(smem_V1, (__nv_bfloat16*)smem_V1 + k_step * 1024);
                
                umma_f16_cg1_fn(o_tmem_addr, desc_a, desc_b0, idesc_pv, k_step == 0 ? 0 : 1);
                umma_f16_cg1_fn(o_tmem_addr + 64, desc_a, desc_b1, idesc_pv, k_step == 0 ? 0 : 1);
            }
        }

        if (tid == 0) {
            umma_commit_1sm_fn(mbar_O);
        }
        mbarrier_wait_fn(mbar_O, phase_O);
        phase_O ^= 1;

        tmem_load_fence_fn();
        
        for (int col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(o_tmem_addr + col));
            O_reg[col+0] += __uint_as_float(r0);
            O_reg[col+1] += __uint_as_float(r1);
            O_reg[col+2] += __uint_as_float(r2);
            O_reg[col+3] += __uint_as_float(r3);
        }
        tmem_load_fence_fn();
        
        __syncthreads();
    }

    for (int i = 0; i < 128; i++) {
        O_reg[i] /= l_i;
    }

    __shared__ __align__(16) __nv_bfloat16 O_packed[128];
    for (int i = 0; i < 128; i++) {
        O_packed[i] = __float2bfloat16(O_reg[i]);
    }
    __syncthreads();

    int global_row = s_idx + tid;
    if (global_row < S_len) {
        uint4* out_vec = (uint4*)(O + bh * S_len * 128 + global_row * 128);
        uint4* O_vec = (uint4*)O_packed;
        *out_vec = *O_vec;
    }

    if (global_row < S_len) {
        LSE[bh * S_len + global_row] = m_i + logf(l_i);
    }

    if (tid == 0) {
        tmem_dealloc_fn(s_tmem_addr, 64);
        tmem_dealloc_fn(p_tmem_addr, 64);
        tmem_dealloc_fn(o_tmem_addr, 128);
    }
    __syncthreads();
}

// -------------------------------------------------------------------------
// TVM-FFI Binding
// -------------------------------------------------------------------------

namespace tvm_ffi_example_cuda {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);

    if (D != 128) {
        fprintf(stderr, "Expected head dim 128, got %ld\n", D);
        exit(1);
    }

    uint64_t total_rows = B * H * S;
    CUtensorMap tma_Q, tma_K, tma_V;

    CU_CHECK(create_tma_2d_descriptor_BF16(&tma_Q, Q.data_ptr(), D, total_rows, 64, 64));
    CU_CHECK(create_tma_2d_descriptor_BF16(&tma_K, K.data_ptr(), D, total_rows, 64, 64));
    CU_CHECK(create_tma_2d_descriptor_BF16(&tma_V, V.data_ptr(), D, total_rows, 64, 64));

    int64_t threads = 64;
    int64_t num_S_tiles = (S + 63) / 64;
    dim3 grid(num_S_tiles, B * H);

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    int smem_size = 65536; 
    CUDA_CHECK(cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    attention_kernel<<<grid, threads, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, 
        static_cast<__nv_bfloat16*>(O.data_ptr()), 
        static_cast<float*>(LSE.data_ptr()), 
        S
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_example_cuda