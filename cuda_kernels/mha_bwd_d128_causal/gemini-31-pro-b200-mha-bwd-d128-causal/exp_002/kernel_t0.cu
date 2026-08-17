#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
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
        const char* err_str;                                       \
        cuGetErrorString(_e, &err_str);                            \
        fprintf(stderr, "CU error %s at %s:%d\n",                 \
                err_str, __FILE__, __LINE__);                      \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_example_cuda {

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
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

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
}

__device__ __forceinline__ void tma_store_3d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [%0, {%2, %3, %4}], [%1];"
        :: "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
}

__device__ __forceinline__ void tma_store_commit_fn() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void tma_store_wait_fn() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
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

__device__ __forceinline__ uint32_t make_idesc(uint32_t M, uint32_t N, uint32_t a_maj, uint32_t b_maj) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (a_maj << 15);   
    d |= (b_maj << 16);   
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ uint64_t advance_desc_swizzle(uint64_t desc, uint32_t bytes) {
    uint32_t addr = (desc & 0x3FFF) << 4;
    addr += bytes;
    desc = (desc & ~0x3FFFull) | ((addr >> 4) & 0x3FFF);
    uint64_t base_offset = (addr >> 7) & 0x7;
    desc = (desc & ~(0x7ull << 49)) | (base_offset << 49);
    return desc;
}

__device__ __forceinline__ void umma_f16_cg1_fn(uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void compute_S_ij(uint32_t tmem_c, void* Q, void* K, bool accumulate) {
    uint64_t desc_a = make_smem_desc_sm100_fn(Q, 1, 1024);
    uint64_t desc_b = make_smem_desc_sm100_fn(K, 1, 1024);
    uint32_t idesc = make_idesc(128, 128, 0, 0);
    for (int k = 0; k < 128; k += 16) {
        uint64_t da = advance_desc_swizzle(desc_a, k * 2);
        uint64_t db = advance_desc_swizzle(desc_b, k * 2);
        umma_f16_cg1_fn(tmem_c, da, db, idesc, accumulate ? 1 : (k == 0 ? 0 : 1));
    }
}

__device__ __forceinline__ void compute_dQ(uint32_t tmem_c, void* dS, void* K_j, bool accumulate) {
    uint64_t desc_a = make_smem_desc_sm100_fn(dS, 1, 1024);
    uint64_t desc_b = make_smem_desc_sm100_fn(K_j, 16384, 1024);
    uint32_t idesc = make_idesc(128, 128, 0, 1);
    for (int k = 0; k < 128; k += 16) {
        uint64_t da = advance_desc_swizzle(desc_a, k * 2);
        uint64_t db = advance_desc_swizzle(desc_b, k * 256);
        umma_f16_cg1_fn(tmem_c, da, db, idesc, accumulate ? 1 : (k == 0 ? 0 : 1));
    }
}

__device__ __forceinline__ void compute_dK(uint32_t tmem_c, void* dS, void* Q_i, bool accumulate) {
    uint64_t desc_a = make_smem_desc_sm100_fn(dS, 16384, 1024);
    uint64_t desc_b = make_smem_desc_sm100_fn(Q_i, 16384, 1024);
    uint32_t idesc = make_idesc(128, 128, 1, 1);
    for (int k = 0; k < 128; k += 16) {
        uint64_t da = advance_desc_swizzle(desc_a, k * 256);
        uint64_t db = advance_desc_swizzle(desc_b, k * 256);
        umma_f16_cg1_fn(tmem_c, da, db, idesc, accumulate ? 1 : (k == 0 ? 0 : 1));
    }
}

__device__ __forceinline__ __nv_bfloat16* get_swizzled_ptr(__nv_bfloat16* base, uint32_t row, uint32_t col) {
    uint32_t byte_offset = row * 256 + ((col * 2) & ~0x7F) + (((col * 2) & 0x7F) ^ ((row % 8) * 16));
    return (__nv_bfloat16*)((char*)base + byte_offset);
}

__global__ __launch_bounds__(128, 1)
void bwd_dq_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const __grid_constant__ CUtensorMap tma_dQ,
    const float* L_ptr, int S
) {
    int i = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    int batch_head = b * gridDim.y + h;
    
    __shared__ uint32_t smem_tmem[4];
    if (threadIdx.x == 0) {
        tmem_alloc_cg1_fn(&smem_tmem[0], 128);
        tmem_alloc_cg1_fn(&smem_tmem[1], 128);
        tmem_alloc_cg1_fn(&smem_tmem[2], 128);
    }
    __syncthreads();
    uint32_t tmem_dQ = smem_tmem[0];
    uint32_t tmem_S = smem_tmem[1];
    uint32_t tmem_dP = smem_tmem[2];
    
    extern __shared__ __align__(1024) uint8_t smem_pool[];
    __nv_bfloat16* smem_Q = (__nv_bfloat16*)smem_pool;
    __nv_bfloat16* smem_K = smem_Q + 128*128;
    __nv_bfloat16* smem_V = smem_K + 128*128;
    __nv_bfloat16* smem_dO = smem_V + 128*128;
    __nv_bfloat16* smem_dS = smem_dO + 128*128;
    __nv_bfloat16* smem_O_P = smem_dS + 128*128;
    
    uint64_t* mbar_Q = (uint64_t*)(smem_O_P + 128*128);
    uint64_t* mbar_K = mbar_Q + 1;
    uint64_t* mbar_V = mbar_K + 1;
    uint64_t* mbar_O = mbar_V + 1;
    uint64_t* mbar_dO = mbar_O + 1;
    uint64_t* mbar_mma = mbar_dO + 1;
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(mbar_K, 1);
        init_smem_barrier_fn(mbar_V, 1);
        init_smem_barrier_fn(mbar_O, 1);
        init_smem_barrier_fn(mbar_dO, 1);
        init_smem_barrier_fn(mbar_mma, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();
    
    if (threadIdx.x == 0) {
        tma_load_3d_fn(&tma_Q, mbar_Q, smem_Q, 0, i * 128, batch_head);
        tma_load_3d_fn(&tma_O, mbar_O, smem_O_P, 0, i * 128, batch_head);
        tma_load_3d_fn(&tma_dO, mbar_dO, smem_dO, 0, i * 128, batch_head);
    }
    
    mbarrier_wait_fn(mbar_Q, 0);
    mbarrier_wait_fn(mbar_O, 0);
    mbarrier_wait_fn(mbar_dO, 0);
    
    float L_local = -INFINITY;
    int seq_idx = i * 128 + threadIdx.x;
    if (seq_idx < S) {
        L_local = L_ptr[batch_head * S + seq_idx];
    }
    
    float Delta_local = 0.0f;
    for (int c = 0; c < 128; c += 8) {
        uint4 o_vec = *(uint4*)get_swizzled_ptr(smem_O_P, threadIdx.x, c);
        uint4 do_vec = *(uint4*)get_swizzled_ptr(smem_dO, threadIdx.x, c);
        __nv_bfloat16* o_arr = (__nv_bfloat16*)&o_vec;
        __nv_bfloat16* do_arr = (__nv_bfloat16*)&do_vec;
        for(int k=0; k<8; k++) {
            Delta_local += __bfloat162float(o_arr[k]) * __bfloat162float(do_arr[k]);
        }
    }
    
    int phase_K = 0, phase_V = 0, phase_mma = 0;
    bool first_dq = true;
    for (int j = 0; j <= i; j++) {
        if (threadIdx.x == 0) {
            tma_load_3d_fn(&tma_K, mbar_K, smem_K, 0, j * 128, batch_head);
            tma_load_3d_fn(&tma_V, mbar_V, smem_V, 0, j * 128, batch_head);
        }
        mbarrier_wait_fn(mbar_K, phase_K); phase_K ^= 1;
        mbarrier_wait_fn(mbar_V, phase_V); phase_V ^= 1;
        
        fence_async_shared_fn();
        compute_S_ij(tmem_S, smem_Q, smem_K, false);
        compute_S_ij(tmem_dP, smem_dO, smem_V, false);
        
        uint32_t mbar_mma_addr = (uint32_t)__cvta_generic_to_shared(mbar_mma);
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(mbar_mma_addr));
        mbarrier_wait_fn(mbar_mma, phase_mma); phase_mma ^= 1;
        
        for (int col = 0; col < 128; col += 4) {
            uint32_t s_r0, s_r1, s_r2, s_r3;
            uint32_t dp_r0, dp_r1, dp_r2, dp_r3;
            uint32_t s_col = tmem_S + col;
            uint32_t dp_col = tmem_dP + col;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(s_r0),"=r"(s_r1),"=r"(s_r2),"=r"(s_r3) : "r"(s_col));
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(dp_r0),"=r"(dp_r1),"=r"(dp_r2),"=r"(dp_r3) : "r"(dp_col));
            tmem_load_fence_fn();
            
            float scale = 1.0f / sqrtf(128.0f);
            float s_arr[4] = {__uint_as_float(s_r0), __uint_as_float(s_r1), __uint_as_float(s_r2), __uint_as_float(s_r3)};
            float dp_arr[4] = {__uint_as_float(dp_r0), __uint_as_float(dp_r1), __uint_as_float(dp_r2), __uint_as_float(dp_r3)};
            
            __nv_bfloat16 p_bf16[4], ds_bf16[4];
            for(int k=0; k<4; k++) {
                float s_val = s_arr[k] * scale;
                if (i * 128 + threadIdx.x >= S || j * 128 + col + k >= S) s_val = -INFINITY;
                else if (i == j && threadIdx.x < col + k) s_val = -INFINITY;
                
                float p_val = expf(s_val - L_local);
                float ds_val = p_val * (dp_arr[k] - Delta_local);
                p_bf16[k] = __float2bfloat16(p_val);
                ds_bf16[k] = __float2bfloat16(ds_val);
            }
            
            *(uint2*)get_swizzled_ptr(smem_O_P, threadIdx.x, col) = *(uint2*)p_bf16;
            *(uint2*)get_swizzled_ptr(smem_dS, threadIdx.x, col) = *(uint2*)ds_bf16;
        }
        
        __syncthreads();
        fence_async_shared_fn();
        
        compute_dQ(tmem_dQ, smem_dS, smem_K, !first_dq);
        first_dq = false;
        
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(mbar_mma_addr));
        mbarrier_wait_fn(mbar_mma, phase_mma); phase_mma ^= 1;
    }
    
    for (int col = 0; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_dQ + col));
        tmem_load_fence_fn();
        __nv_bfloat16 dq_bf16[4];
        dq_bf16[0] = __float2bfloat16(__uint_as_float(r0));
        dq_bf16[1] = __float2bfloat16(__uint_as_float(r1));
        dq_bf16[2] = __float2bfloat16(__uint_as_float(r2));
        dq_bf16[3] = __float2bfloat16(__uint_as_float(r3));
        *(uint2*)get_swizzled_ptr(smem_Q, threadIdx.x, col) = *(uint2*)dq_bf16;
    }
    __syncthreads();
    fence_async_shared_fn();
    
    if (threadIdx.x == 0) {
        tma_store_3d_fn(&tma_dQ, smem_Q, 0, i * 128, batch_head);
        tma_store_commit_fn();
    }
    tma_store_wait_fn<0>();
    
    if (threadIdx.x == 0) {
        tmem_dealloc_cg1_fn(tmem_dQ, 128);
        tmem_dealloc_cg1_fn(tmem_S, 128);
        tmem_dealloc_cg1_fn(tmem_dP, 128);
    }
}

__global__ __launch_bounds__(128, 1)
void bwd_dkv_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const __grid_constant__ CUtensorMap tma_dK,
    const __grid_constant__ CUtensorMap tma_dV,
    const float* L_ptr, int S
) {
    int j = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    int batch_head = b * gridDim.y + h;
    
    __shared__ uint32_t smem_tmem[4];
    if (threadIdx.x == 0) {
        tmem_alloc_cg1_fn(&smem_tmem[0], 128);
        tmem_alloc_cg1_fn(&smem_tmem[1], 128);
        tmem_alloc_cg1_fn(&smem_tmem[2], 128);
        tmem_alloc_cg1_fn(&smem_tmem[3], 128);
    }
    __syncthreads();
    uint32_t tmem_dK = smem_tmem[0];
    uint32_t tmem_dV = smem_tmem[1];
    uint32_t tmem_S = smem_tmem[2];
    uint32_t tmem_dP = smem_tmem[3];
    
    extern __shared__ __align__(1024) uint8_t smem_pool[];
    __nv_bfloat16* smem_K = (__nv_bfloat16*)smem_pool;
    __nv_bfloat16* smem_V = smem_K + 128*128;
    __nv_bfloat16* smem_Q = smem_V + 128*128;
    __nv_bfloat16* smem_dO = smem_Q + 128*128;
    __nv_bfloat16* smem_dS = smem_dO + 128*128;
    __nv_bfloat16* smem_O_P = smem_dS + 128*128;
    
    uint64_t* mbar_Q = (uint64_t*)(smem_O_P + 128*128);
    uint64_t* mbar_K = mbar_Q + 1;
    uint64_t* mbar_V = mbar_K + 1;
    uint64_t* mbar_O = mbar_V + 1;
    uint64_t* mbar_dO = mbar_O + 1;
    uint64_t* mbar_mma = mbar_dO + 1;
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(mbar_K, 1);
        init_smem_barrier_fn(mbar_V, 1);
        init_smem_barrier_fn(mbar_O, 1);
        init_smem_barrier_fn(mbar_dO, 1);
        init_smem_barrier_fn(mbar_mma, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();
    
    if (threadIdx.x == 0) {
        tma_load_3d_fn(&tma_K, mbar_K, smem_K, 0, j * 128, batch_head);
        tma_load_3d_fn(&tma_V, mbar_V, smem_V, 0, j * 128, batch_head);
    }
    mbarrier_wait_fn(mbar_K, 0);
    mbarrier_wait_fn(mbar_V, 0);
    
    int phase_Q = 0, phase_O = 0, phase_dO = 0, phase_mma = 0;
    bool first_dkv = true;
    int num_i = (S + 127) / 128;
    
    for (int i = j; i < num_i; i++) {
        if (threadIdx.x == 0) {
            tma_load_3d_fn(&tma_Q, mbar_Q, smem_Q, 0, i * 128, batch_head);
            tma_load_3d_fn(&tma_O, mbar_O, smem_O_P, 0, i * 128, batch_head);
            tma_load_3d_fn(&tma_dO, mbar_dO, smem_dO, 0, i * 128, batch_head);
        }
        mbarrier_wait_fn(mbar_Q, phase_Q); phase_Q ^= 1;
        mbarrier_wait_fn(mbar_O, phase_O); phase_O ^= 1;
        mbarrier_wait_fn(mbar_dO, phase_dO); phase_dO ^= 1;
        
        float L_local = -INFINITY;
        int seq_idx = i * 128 + threadIdx.x;
        if (seq_idx < S) {
            L_local = L_ptr[batch_head * S + seq_idx];
        }
        
        float Delta_local = 0.0f;
        for (int c = 0; c < 128; c += 8) {
            uint4 o_vec = *(uint4*)get_swizzled_ptr(smem_O_P, threadIdx.x, c);
            uint4 do_vec = *(uint4*)get_swizzled_ptr(smem_dO, threadIdx.x, c);
            __nv_bfloat16* o_arr = (__nv_bfloat16*)&o_vec;
            __nv_bfloat16* do_arr = (__nv_bfloat16*)&do_vec;
            for(int k=0; k<8; k++) {
                Delta_local += __bfloat162float(o_arr[k]) * __bfloat162float(do_arr[k]);
            }
        }
        
        fence_async_shared_fn();
        compute_S_ij(tmem_S, smem_Q, smem_K, false);
        compute_S_ij(tmem_dP, smem_dO, smem_V, false);
        
        uint32_t mbar_mma_addr = (uint32_t)__cvta_generic_to_shared(mbar_mma);
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(mbar_mma_addr));
        mbarrier_wait_fn(mbar_mma, phase_mma); phase_mma ^= 1;
        
        for (int col = 0; col < 128; col += 4) {
            uint32_t s_r0, s_r1, s_r2, s_r3;
            uint32_t dp_r0, dp_r1, dp_r2, dp_r3;
            uint32_t s_col = tmem_S + col;
            uint32_t dp_col = tmem_dP + col;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(s_r0),"=r"(s_r1),"=r"(s_r2),"=r"(s_r3) : "r"(s_col));
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(dp_r0),"=r"(dp_r1),"=r"(dp_r2),"=r"(dp_r3) : "r"(dp_col));
            tmem_load_fence_fn();
            
            float scale = 1.0f / sqrtf(128.0f);
            float s_arr[4] = {__uint_as_float(s_r0), __uint_as_float(s_r1), __uint_as_float(s_r2), __uint_as_float(s_r3)};
            float dp_arr[4] = {__uint_as_float(dp_r0), __uint_as_float(dp_r1), __uint_as_float(dp_r2), __uint_as_float(dp_r3)};
            
            __nv_bfloat16 p_bf16[4], ds_bf16[4];
            for(int k=0; k<4; k++) {
                float s_val = s_arr[k] * scale;
                if (i * 128 + threadIdx.x >= S || j * 128 + col + k >= S) s_val = -INFINITY;
                else if (i == j && threadIdx.x < col + k) s_val = -INFINITY;
                
                float p_val = expf(s_val - L_local);
                float ds_val = p_val * (dp_arr[k] - Delta_local);
                p_bf16[k] = __float2bfloat16(p_val);
                ds_bf16[k] = __float2bfloat16(ds_val);
            }
            
            *(uint2*)get_swizzled_ptr(smem_O_P, threadIdx.x, col) = *(uint2*)p_bf16;
            *(uint2*)get_swizzled_ptr(smem_dS, threadIdx.x, col) = *(uint2*)ds_bf16;
        }
        
        __syncthreads();
        fence_async_shared_fn();
        
        compute_dK(tmem_dK, smem_dS, smem_Q, !first_dkv);
        compute_dK(tmem_dV, smem_O_P, smem_dO, !first_dkv);
        first_dkv = false;
        
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(mbar_mma_addr));
        mbarrier_wait_fn(mbar_mma, phase_mma); phase_mma ^= 1;
    }
    
    for (int col = 0; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_dK + col));
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(tmem_dV + col));
        tmem_load_fence_fn();
        __nv_bfloat16 dk_bf16[4], dv_bf16[4];
        dk_bf16[0] = __float2bfloat16(__uint_as_float(r0));
        dk_bf16[1] = __float2bfloat16(__uint_as_float(r1));
        dk_bf16[2] = __float2bfloat16(__uint_as_float(r2));
        dk_bf16[3] = __float2bfloat16(__uint_as_float(r3));
        dv_bf16[0] = __float2bfloat16(__uint_as_float(r4));
        dv_bf16[1] = __float2bfloat16(__uint_as_float(r5));
        dv_bf16[2] = __float2bfloat16(__uint_as_float(r6));
        dv_bf16[3] = __float2bfloat16(__uint_as_float(r7));
        *(uint2*)get_swizzled_ptr(smem_K, threadIdx.x, col) = *(uint2*)dk_bf16;
        *(uint2*)get_swizzled_ptr(smem_V, threadIdx.x, col) = *(uint2*)dv_bf16;
    }
    __syncthreads();
    fence_async_shared_fn();
    
    if (threadIdx.x == 0) {
        tma_store_3d_fn(&tma_dK, smem_K, 0, j * 128, batch_head);
        tma_store_3d_fn(&tma_dV, smem_V, 0, j * 128, batch_head);
        tma_store_commit_fn();
    }
    tma_store_wait_fn<0>();
    
    if (threadIdx.x == 0) {
        tmem_dealloc_cg1_fn(tmem_dK, 128);
        tmem_dealloc_cg1_fn(tmem_dV, 128);
        tmem_dealloc_cg1_fn(tmem_S, 128);
        tmem_dealloc_cg1_fn(tmem_dP, 128);
    }
}

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, 
                                     uint64_t dim0, uint64_t dim1, uint64_t dim2,
                                     uint32_t box0, uint32_t box1, uint32_t box2, 
                                     CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[3] = {dim0, dim1, dim2};
    cuuint64_t globalStrides[2] = {dim0 * 2, dim0 * dim1 * 2};
    cuuint32_t boxDim[3] = {box0, box1, box2};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O, tma_dO, tma_dQ, tma_dK, tma_dV;
    uint64_t M_total = B * H;
    
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_Q, Q.data_ptr(), 128, S, M_total, 128, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_K, K.data_ptr(), 128, S, M_total, 128, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_V, V.data_ptr(), 128, S, M_total, 128, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_O, O.data_ptr(), 128, S, M_total, 128, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_dO, dO.data_ptr(), 128, S, M_total, 128, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_dQ, dQ.data_ptr(), 128, S, M_total, 128, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_dK, dK.data_ptr(), 128, S, M_total, 128, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_dV, dV.data_ptr(), 128, S, M_total, 128, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B));
    
    CUDA_CHECK(cudaFuncSetAttribute(bwd_dq_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 200000));
    CUDA_CHECK(cudaFuncSetAttribute(bwd_dkv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 200000));
    
    dim3 grid((S + 127) / 128, H, B);
    dim3 block(128);
    
    bwd_dq_kernel<<<grid, block, 200000, stream>>>(tma_Q, tma_K, tma_V, tma_O, tma_dO, tma_dQ, (float*)L.data_ptr(), S);
    CUDA_CHECK(cudaGetLastError());
    
    bwd_dkv_kernel<<<grid, block, 200000, stream>>>(tma_Q, tma_K, tma_V, tma_O, tma_dO, tma_dK, tma_dV, (float*)L.data_ptr(), S);
    CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}