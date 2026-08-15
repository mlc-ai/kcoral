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

#define CU_CHECK(call) do {                                        \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        fprintf(stderr, "CU error %d at %s:%d\n",                 \
                _e, __FILE__, __LINE__);                           \
        exit(1);                                                   \
    }                                                              \
} while(0)


namespace tvm_ffi_optimized_cuda {

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

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ void tmem_store_fp32(uint32_t addr, float val) {
    uint32_t val_u32 = __float_as_uint(val);
    asm volatile("tcgen05.st.sync.aligned.32x32b.b32 [%0], %1;" :: "r"(addr), "r"(val_u32));
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_128B(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16; 
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32; 
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_128B_with_lbo(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_mnmaj_128B(void* smem_ptr, uint32_t sbo, uint32_t lbo) {
    return make_smem_desc_128B_with_lbo(smem_ptr, lbo, sbo);
}

__device__ __forceinline__ uint64_t make_smem_desc_kmaj_128B(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    return make_smem_desc_128B(smem_ptr, lbo, sbo);
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    // c_format = FP32
    d |= (1u << 7);    // a_format = BF16
    d |= (1u << 10);   // b_format = BF16
    d |= (0u << 15);   // a_major = 0 (A is K-Major)
    d |= (0u << 16);   // b_major = 0 (B is K-Major)
    d |= ((N >> 3) & 0x3F);     
    d |= ((M >> 4) & 0x1F) << 24;    
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn_trans_B(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (0u << 15);   
    d |= (1u << 16);   
    d |= ((N >> 3) & 0x3F);     
    d |= ((M >> 4) & 0x1F) << 24;    
    return d;
}

__device__ __forceinline__ uint64_t add_base_offset(uint64_t desc, uint32_t offset_bytes) {
    uint32_t base = desc & 0x3FFF;
    base += offset_bytes / 16;
    if ((base >> 14) != 0) {
        base &= 0x3FFF;
    }
    desc &= ~0x3FFF;
    desc |= base;
    return desc;
}

__device__ __forceinline__ void umma_f16_cg2(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_f16_cg2_tmem_a(
    uint32_t tmem_c, uint32_t tmem_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], [%1], %2, %3, p;\n}\n"
        :: "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_2sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::2"
        ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;"
        :: "r"(a), "h"((uint16_t)0x3));
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

__global__ void mha_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* gmem_O,
    float* gmem_LSE,
    uint32_t S)
{
    extern __shared__ __align__(1024) char smem_pool[];
    __nv_bfloat16* smem_Q = (__nv_bfloat16*)smem_pool;                
    __nv_bfloat16* smem_K = (__nv_bfloat16*)(smem_pool + 16384);      
    __nv_bfloat16* smem_V = (__nv_bfloat16*)(smem_pool + 32768);      
    uint64_t* mbar = (uint64_t*)(smem_pool + 49152);                 

    uint32_t tid = threadIdx.x;
    uint32_t s_blk = blockIdx.y;
    uint32_t bh = blockIdx.x;
    uint32_t row = s_blk * 64 + tid;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar, 1);
    }
    __syncthreads();

    uint32_t tmem_base;
    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_base, 128); 
    }
    __syncthreads();
    
    uint32_t tmem_P = tmem_base;
    uint32_t tmem_O = tmem_base + 4096; 
    
    uint32_t load_bytes = 0;
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar, 32768);
        tma_load_2d_fn(&tma_Q, mbar, smem_Q, 0, bh * S + row);
        load_bytes = 16384;
    }
    mbarrier_wait_fn(mbar, 0);
    fence_proxy_async_fn();

    uint64_t desc_Q = make_smem_desc_128B(smem_Q, 1, 1024);
    uint64_t desc_K = make_smem_desc_128B(smem_K, 1, 1024);
    uint64_t desc_V = make_smem_desc_128B(smem_V, 1, 1024);
    
    uint32_t idesc_QK = make_instr_desc_fn(128, 64);
    uint32_t idesc_SV = make_instr_desc_fn_trans_B(128, 128);

    float p_max_local[64];
    float p_sum_local[64];
    for (int i = 0; i < 64; ++i) {
        p_max_local[i] = -INFINITY;
        p_sum_local[i] = 0.0f;
    }

    uint32_t phase = 1;

    for (uint32_t i = 0; i < S/64; ++i) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar, 32768);
            tma_load_2d_fn(&tma_K, mbar, smem_K, 0, bh * S + i * 64);
            tma_load_2d_fn(&tma_V, mbar, smem_V, 0, bh * S + i * 64);
        }
        mbarrier_wait_fn(mbar, phase);
        fence_proxy_async_fn();
        phase ^= 1;

        desc_K = make_smem_desc_128B(smem_K, 1, 1024);
        desc_V = make_smem_desc_128B(smem_V, 1, 1024);

        p_max_local[tid] = -INFINITY; 
        
        for (uint32_t k = 0; k < 128; k += 16) {
            uint32_t off = k * 2;
            uint64_t dk = add_base_offset(desc_K, off);
            uint64_t dq = add_base_offset(desc_Q, off);
            
            if (k == 0) {
                umma_f16_cg2(tmem_P, dq, dk, idesc_QK, 0); 
            } else {
                umma_f16_cg2(tmem_P, dq, dk, idesc_QK, 1); 
            }
        }
        umma_commit_2sm_fn(mbar);
        
        float local_max = -INFINITY;
        for (uint32_t col = 0; col < 64; col += 4) {
            uint32_t addr = tid * 64 + col;
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(addr, &r0, &r1, &r2, &r3);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float f0 = __uint_as_float(r0);
            float f1 = __uint_as_float(r1);
            float f2 = __uint_as_float(r2);
            float f3 = __uint_as_float(r3);
            
            local_max = fmaxf(local_max, fmaxf(fmaxf(f0, f1), fmaxf(f2, f3)));
            
            r0 = __float_as_uint(f0);
            r1 = __float_as_uint(f1);
            r2 = __float_as_uint(f2);
            r3 = __float_as_uint(f3);
            
            uint32_t col_u32[4] = {r0, r1, r2, r3};
            asm volatile("tcgen05.st.sync.aligned.32x32b.b32 [%0], %1;" :: "r"(tmem_P + addr), "r"(col_u32[0]));
            asm volatile("tcgen05.st.sync.aligned.32x32b.b32 [%0], %1;" :: "r"(tmem_P + addr + 1), "r"(col_u32[1]));
            asm volatile("tcgen05.st.sync.aligned.32x32b.b32 [%0], %1;" :: "r"(tmem_P + addr + 2), "r"(col_u32[2]));
            asm volatile("tcgen05.st.sync.aligned.32x32b.b32 [%0], %1;" :: "r"(tmem_P + addr + 3), "r"(col_u32[3]));
        }
        
        p_max_local[tid] = fmaxf(p_max_local[tid], local_max);
        
        float current_max = p_max_local[tid];
        float scale = __expf(local_max - current_max);
        p_sum_local[tid] *= scale;
        
        float local_sum = 0;
        for (uint32_t col = 0; col < 64; col += 4) {
            uint32_t addr = tid * 64 + col;
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(addr, &r0, &r1, &r2, &r3);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float f0 = __uint_as_float(r0);
            float f1 = __uint_as_float(r1);
            float f2 = __uint_as_float(r2);
            float f3 = __uint_as_float(r3);
            
            f0 = __expf(f0 - local_max);
            f1 = __expf(f1 - local_max);
            f2 = __expf(f2 - local_max);
            f3 = __expf(f3 - local_max);
            
            local_sum += (f0 + f1 + f2 + f3);
            
            r0 = __float_as_uint(f0);
            r1 = __float_as_uint(f1);
            r2 = __float_as_uint(f2);
            r3 = __float_as_uint(f3);
            
            uint32_t col_u32[4] = {r0, r1, r2, r3};
            asm volatile("tcgen05.st.sync.aligned.32x32b.b32 [%0], %1;" :: "r"(tmem_P + addr), "r"(col_u32[0]));
            asm volatile("tcgen05.st.sync.aligned.32x32b.b32 [%0], %1;" :: "r"(tmem_P + addr + 1), "r"(col_u32[1]));
            asm volatile("tcgen05.st.sync.aligned.32x32b.b32 [%0], %1;" :: "r"(tmem_P + addr + 2), "r"(col_u32[2]));
            asm volatile("tcgen05.st.sync.aligned.32x32b.b32 [%0], %1;" :: "r"(tmem_P + addr + 3), "r"(col_u32[3]));
        }
        p_sum_local[tid] += local_sum;
        
        for (uint32_t col = 0; col < 64; col += 4) {
            uint32_t addr = tid * 64 + col;
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(addr, &r0, &r1, &r2, &r3);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float f0 = __uint_as_float(r0);
            float f1 = __uint_as_float(r1);
            float f2 = __uint_as_float(r2);
            float f3 = __uint_as_float(r3);
            
            f0 *= scale;
            f1 *= scale;
            f2 *= scale;
            f3 *= scale;
            
            r0 = __float_as_uint(f0);
            r1 = __float_as_uint(f1);
            r2 = __float_as_uint(f2);
            r3 = __float_as_uint(f3);
            
            uint32_t col_u32[4] = {r0, r1, r2, r3};
            asm volatile("tcgen05.st.sync.aligned.32x32b.b32 [%0], %1;" :: "r"(tmem_O + addr), "r"(col_u32[0]));
            asm volatile("tcgen05.st.sync.aligned.32x32b.b32 [%0], %1;" :: "r"(tmem_O + addr + 1), "r"(col_u32[1]));
            asm volatile("tcgen05.st.sync.aligned.32x32b.b32 [%0], %1;" :: "r"(tmem_O + addr + 2), "r"(col_u32[2]));
            asm volatile("tcgen05.st.sync.aligned.32x32b.b32 [%0], %1;" :: "r"(tmem_O + addr + 3), "r"(col_u32[3]));
        }
        
        for (uint32_t col = 0; col < 64; col += 4) {
            uint32_t addr = tid * 64 + col;
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(addr, &r0, &r1, &r2, &r3);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float f0 = __uint_as_float(r0);
            float f1 = __uint_as_float(r1);
            float f2 = __uint_as_float(r2);
            float f3 = __uint_as_float(r3);
            
            f0 = __expf(f0 - current_max);
            f1 = __expf(f1 - current_max);
            f2 = __expf(f2 - current_max);
            f3 = __expf(f3 - current_max);
            
            r0 = __float_as_uint(f0);
            r1 = __float_as_uint(f1);
            r2 = __float_as_uint(f2);
            r3 = __float_as_uint(f3);
            
            uint32_t col_u32[4] = {r0, r1, r2, r3};
            asm volatile("tcgen05.st.sync.aligned.32x32b.b32 [%0], %1;" :: "r"(tmem_P + addr), "r"(col_u32[0]));
            asm volatile("tcgen05.st.sync.aligned.32x32b.b32 [%0], %1;" :: "r"(tmem_P + addr + 1), "r"(col_u32[1]));
            asm volatile("tcgen05.st.sync.aligned.32x32b.b32 [%0], %1;" :: "r"(tmem_P + addr + 2), "r"(col_u32[2]));
            asm volatile("tcgen05.st.sync.aligned.32x32b.b32 [%0], %1;" :: "r"(tmem_P + addr + 3), "r"(col_u32[3]));
        }
        
        uint64_t desc_S = make_smem_desc_128B((void*)tmem_P, 1, 1024);
        uint64_t desc_V_mn = make_smem_desc_mnmaj_128B(smem_V, 1024, 16384);
        
        for (uint32_t k = 0; k < 128; k += 16) {
            uint32_t off = k * 2;
            uint64_t ds = add_base_offset(desc_S, off);
            uint64_t dv = add_base_offset(desc_V_mn, off * 8); 
            
            if (k == 0) {
                umma_f16_cg2(tmem_O, ds, dv, idesc_SV, 0);
            } else {
                umma_f16_cg2(tmem_O, ds, dv, idesc_SV, 1);
            }
        }
        umma_commit_2sm_fn(mbar);
    }
    
    for (uint32_t col = 0; col < 64; col += 4) {
        uint32_t addr = tid * 64 + col;
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn(tmem_O + addr, &r0, &r1, &r2, &r3);
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float f0 = __uint_as_float(r0) / p_sum_local[tid];
        float f1 = __uint_as_float(r1) / p_sum_local[tid];
        float f2 = __uint_as_float(r2) / p_sum_local[tid];
        float f3 = __uint_as_float(r3) / p_sum_local[tid];
        
        uint32_t nc = col;
        __nv_bfloat16* out = gmem_O + bh * S * 128 + row * 128 + nc;
        if (row < S && nc < 128) out[0] = __float2bfloat16(f0);
        if (row < S && nc + 1 < 128) out[1] = __float2bfloat16(f1);
        if (row < S && nc + 2 < 128) out[2] = __float2bfloat16(f2);
        if (row < S && nc + 3 < 128) out[3] = __float2bfloat16(f3);
    }
    
    for (uint32_t col = 0; col < 64; col += 4) {
        uint32_t addr = tid * 64 + col;
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn(tmem_O + 4096 + addr, &r0, &r1, &r2, &r3);
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float f0 = __uint_as_float(r0) / p_sum_local[tid];
        float f1 = __uint_as_float(r1) / p_sum_local[tid];
        float f2 = __uint_as_float(r2) / p_sum_local[tid];
        float f3 = __uint_as_float(r3) / p_sum_local[tid];
        
        uint32_t nc = col + 64;
        __nv_bfloat16* out = gmem_O + bh * S * 128 + row * 128 + nc;
        if (row < S && nc < 128) out[0] = __float2bfloat16(f0);
        if (row < S && nc + 1 < 128) out[1] = __float2bfloat16(f1);
        if (row < S && nc + 2 < 128) out[2] = __float2bfloat16(f2);
        if (row < S && nc + 3 < 128) out[3] = __float2bfloat16(f3);
    }
    
    if (tid < 64) {
        float lse = p_max_local[tid] + __logf(p_sum_local[tid]);
        *(gmem_LSE + bh * S + row) = lse;
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    uint32_t B = Q.size(0);
    uint32_t H = Q.size(1);
    uint32_t S = Q.size(2);
    uint32_t D = Q.size(3); 

    CUtensorMap tma_Q, tma_K, tma_V;
    __nv_bfloat16* q_ptr = static_cast<__nv_bfloat16*>(Q.data_ptr());
    __nv_bfloat16* k_ptr = static_cast<__nv_bfloat16*>(K.data_ptr());
    __nv_bfloat16* v_ptr = static_cast<__nv_bfloat16*>(V.data_ptr());

    CU_CHECK(create_tma_2d_descriptor_2B(&tma_Q, q_ptr, D, B * H * S, 128, 64, 
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    CU_CHECK(create_tma_2d_descriptor_2B(&tma_K, k_ptr, D, B * H * S, 128, 64, 
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    CU_CHECK(create_tma_2d_descriptor_2B(&tma_V, v_ptr, D, B * H * S, 128, 64, 
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    __nv_bfloat16* o_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_data = static_cast<float*>(LSE.data_ptr());

    dim3 grid(B * H, S / 64);
    dim3 block(128);

    uint32_t smem_bytes = 85 * 1024;

    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_bytes;
    config.stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    CUDA_CHECK(cudaLaunchKernelEx(&config, mha_kernel, tma_Q, tma_K, tma_V, o_data, lse_data, S));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id))));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_optimized_cuda::run);

} // namespace tvm_ffi_optimized_cuda