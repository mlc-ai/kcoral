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

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes) : "memory");
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

__device__ __forceinline__ void tma_load_4d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}

__device__ __forceinline__ void tma_store_4d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group [%0, {%2, %3, %4, %5}], [%1];"
        :: "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
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

__device__ __forceinline__ __nv_bfloat16* get_swizzled_ptr(__nv_bfloat16* base, uint32_t row, uint32_t col) {
    uint32_t c_bytes = col * 2;
    uint32_t c_swizzled = c_bytes ^ ((row % 8) * 16);
    return (__nv_bfloat16*)((char*)base + row * 128 + c_swizzled);
}

__device__ __forceinline__ void compute_S_ij(uint32_t tmem_s, void* Q0, void* Q1, void* K0, void* K1, bool accumulate) {
    uint32_t idesc = make_idesc(128, 128, 0, 0);
    
    uint64_t desc_a0 = make_smem_desc_sm100_fn(Q0, 1, 1024);
    uint64_t desc_b0 = make_smem_desc_sm100_fn(K0, 1, 1024);
    for (int k = 0; k < 64; k += 16) {
        uint64_t da = advance_desc_swizzle(desc_a0, k * 2);
        uint64_t db = advance_desc_swizzle(desc_b0, k * 2);
        umma_f16_cg1_fn(tmem_s, da, db, idesc, accumulate ? 1 : (k == 0 ? 0 : 1));
    }
    
    uint64_t desc_a1 = make_smem_desc_sm100_fn(Q1, 1, 1024);
    uint64_t desc_b1 = make_smem_desc_sm100_fn(K1, 1, 1024);
    for (int k = 0; k < 64; k += 16) {
        uint64_t da = advance_desc_swizzle(desc_a1, k * 2);
        uint64_t db = advance_desc_swizzle(desc_b1, k * 2);
        umma_f16_cg1_fn(tmem_s, da, db, idesc, 1);
    }
}

__device__ __forceinline__ void compute_dQ(uint32_t tmem_dq, void* dS0, void* dS1, void* K0, void* K1, bool accumulate) {
    uint32_t idesc = make_idesc(128, 64, 0, 1);
    
    uint64_t desc_a0 = make_smem_desc_sm100_fn(dS0, 1, 1024);
    uint64_t desc_b0 = make_smem_desc_sm100_fn(K0, 16384, 1024);
    for (int k = 0; k < 64; k += 16) {
        uint64_t da = advance_desc_swizzle(desc_a0, k * 2);
        uint64_t db = advance_desc_swizzle(desc_b0, k * 128);
        umma_f16_cg1_fn(tmem_dq, da, db, idesc, accumulate ? 1 : (k == 0 ? 0 : 1));
    }
    uint64_t desc_a1 = make_smem_desc_sm100_fn(dS1, 1, 1024);
    uint64_t base_b1 = (uint64_t)K0 + 64 * 128;
    uint64_t desc_b1 = make_smem_desc_sm100_fn((void*)base_b1, 16384, 1024);
    for (int k = 0; k < 64; k += 16) {
        uint64_t da = advance_desc_swizzle(desc_a1, k * 2);
        uint64_t db = advance_desc_swizzle(desc_b1, k * 128);
        umma_f16_cg1_fn(tmem_dq, da, db, idesc, 1);
    }
    
    uint32_t tmem_dq1 = tmem_dq + 64;
    desc_b0 = make_smem_desc_sm100_fn(K1, 16384, 1024);
    for (int k = 0; k < 64; k += 16) {
        uint64_t da = advance_desc_swizzle(desc_a0, k * 2);
        uint64_t db = advance_desc_swizzle(desc_b0, k * 128);
        umma_f16_cg1_fn(tmem_dq1, da, db, idesc, accumulate ? 1 : (k == 0 ? 0 : 1));
    }
    base_b1 = (uint64_t)K1 + 64 * 128;
    desc_b1 = make_smem_desc_sm100_fn((void*)base_b1, 16384, 1024);
    for (int k = 0; k < 64; k += 16) {
        uint64_t da = advance_desc_swizzle(desc_a1, k * 2);
        uint64_t db = advance_desc_swizzle(desc_b1, k * 128);
        umma_f16_cg1_fn(tmem_dq1, da, db, idesc, 1);
    }
}

__device__ __forceinline__ void compute_dK(uint32_t tmem_dk, void* dS0, void* dS1, void* Q0, void* Q1, bool accumulate) {
    uint32_t idesc = make_idesc(64, 64, 1, 1);
    
    uint32_t tmem_dk0_top = tmem_dk;
    uint32_t tmem_dk0_bot = tmem_dk + (64 << 16);
    uint64_t desc_a0 = make_smem_desc_sm100_fn(dS0, 16384, 1024);
    uint64_t desc_b0 = make_smem_desc_sm100_fn(Q0, 16384, 1024);
    for (int k = 0; k < 128; k += 16) {
        uint64_t da = advance_desc_swizzle(desc_a0, k * 128);
        uint64_t db = advance_desc_swizzle(desc_b0, k * 128);
        umma_f16_cg1_fn(tmem_dk0_top, da, db, idesc, accumulate ? 1 : (k == 0 ? 0 : 1));
    }
    uint64_t desc_a1 = make_smem_desc_sm100_fn(dS1, 16384, 1024);
    for (int k = 0; k < 128; k += 16) {
        uint64_t da = advance_desc_swizzle(desc_a1, k * 128);
        uint64_t db = advance_desc_swizzle(desc_b0, k * 128);
        umma_f16_cg1_fn(tmem_dk0_bot, da, db, idesc, accumulate ? 1 : (k == 0 ? 0 : 1));
    }
    
    uint32_t tmem_dk1_top = tmem_dk + 64;
    uint32_t tmem_dk1_bot = tmem_dk + 64 + (64 << 16);
    uint64_t desc_b1 = make_smem_desc_sm100_fn(Q1, 16384, 1024);
    for (int k = 0; k < 128; k += 16) {
        uint64_t da = advance_desc_swizzle(desc_a0, k * 128);
        uint64_t db = advance_desc_swizzle(desc_b1, k * 128);
        umma_f16_cg1_fn(tmem_dk1_top, da, db, idesc, accumulate ? 1 : (k == 0 ? 0 : 1));
    }
    for (int k = 0; k < 128; k += 16) {
        uint64_t da = advance_desc_swizzle(desc_a1, k * 128);
        uint64_t db = advance_desc_swizzle(desc_b1, k * 128);
        umma_f16_cg1_fn(tmem_dk1_bot, da, db, idesc, accumulate ? 1 : (k == 0 ? 0 : 1));
    }
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
    
    __shared__ uint32_t smem_tmem[1];
    if (threadIdx.x < 32) {
        tmem_alloc_cg1_fn(&smem_tmem[0], 512);
    }
    __syncthreads();
    uint32_t tmem_base = smem_tmem[0];
    uint32_t tmem_dQ = tmem_base;
    uint32_t tmem_S  = tmem_base + 128;
    uint32_t tmem_dP = tmem_base + 256;
    
    extern __shared__ __align__(1024) uint8_t smem_pool[];
    __nv_bfloat16* smem_Q0 = (__nv_bfloat16*)smem_pool;