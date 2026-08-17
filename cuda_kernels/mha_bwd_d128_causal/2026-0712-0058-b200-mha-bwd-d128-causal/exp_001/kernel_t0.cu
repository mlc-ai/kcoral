#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <math.h>
#include <stdio.h>
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
        fprintf(stderr, "CU error %d at %s:%d\n",                \
                _e, __FILE__, __LINE__);                         \
        exit(1);                                                 \
    }                                                            \
} while(0)

namespace tvm_ffi_mha_bwd {

constexpr uint32_t BM_Q = 64;
constexpr uint32_t BM_NK = 64;
constexpr uint32_t BN_d = 128;

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
        "{\n.reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase));
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

__device__ __forceinline__ uint64_t make_smem_desc_sm100_k_major(void* smem_ptr, uint32_t LBO, uint32_t SBO) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((LBO & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((SBO & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_mn_major(void* smem_ptr, uint32_t BM, uint32_t BN, uint32_t BK) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    uint32_t SBO = 8 * 128;
    uint32_t LBO = (BK / 8) * SBO;
    d |= (uint64_t)((LBO & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((SBO & 0x3FFFF) >> 4) << 32;
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

__device__ __forceinline__ void umma_f16_cg2_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_2sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::2"
        ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;"
        :: "r"(a), "h"((uint16_t)0x3));
}

__device__ __forceinline__ void desc_k_major_A_B(
    uint32_t tmem_C, void* smem_A, void* smem_B,
    uint32_t M, uint32_t N, uint32_t K, uint64_t* barrier, uint32_t& phase) 
{
    uint32_t SBO_A = 8 * 128, LBO_A = 1;
    uint32_t SBO_B = 8 * 128, LBO_B = 1;
    uint32_t stride_A = 1;
    uint32_t stride_B = 1;

    mbarrier_arrive_and_expect_tx_fn(barrier, 4096); 
    
    for (uint32_t k = 0; k < K; k += 16) {
        uint64_t desc_a = make_smem_desc_sm100_k_major(smem_A + k * stride_A, LBO_A, SBO_A);
        uint64_t desc_b = make_smem_desc_sm100_k_major(smem_B + k * stride_B, LBO_B, SBO_B);
        uint32_t idesc = make_instr_desc_fn(M, N);
        bool accum = (k == 0) ? false : true;
        umma_f16_cg2_fn(tmem_C, desc_a, desc_b, idesc, accum ? 1 : 0);
    }
    umma_commit_2sm_fn(barrier);
    mbarrier_wait_fn(barrier, phase);
    phase ^= 1;
}

__device__ __forceinline__ void desc_mn_major_A_B(
    uint32_t tmem_C, void* smem_A, void* smem_B,
    uint32_t M, uint32_t N, uint32_t K, uint64_t* barrier, uint32_t& phase) 
{
    uint32_t SBO_A = 8 * 128, LBO_A = (N / 8) * SBO_A;
    uint32_t SBO_B = 8 * 128, LBO_B = (N / 8) * SBO_B;

    mbarrier_arrive_and_expect_tx_fn(barrier, 4096); 
    
    for (uint32_t k = 0; k < K; k += 16) {
        uint64_t desc_a = make_smem_desc_sm100_mn_major(smem_A, M, N, K);
        uint64_t desc_b = make_smem_desc_sm100_mn_major(smem_B, M, N, K);
        uint32_t idesc = make_instr_desc_fn(M, N);
        idesc |= (1u << 15);
        idesc |= (1u << 16);
        bool accum = (k == 0) ? false : true;
        umma_f16_cg2_fn(tmem_C, desc_a, desc_b, idesc, accum ? 1 : 0);
    }
    umma_commit_2sm_fn(barrier);
    mbarrier_wait_fn(barrier, phase);
    phase ^= 1;
}

__device__ __forceinline__ void load_g2s_bf16_128B_swizzle(
    __nv_bfloat16* smem, const void* global_ptr,
    uint32_t base_row, uint32_t S) 
{