#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
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

namespace tvm_ffi_flash_attention_bwd {

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

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

__device__ __forceinline__ void named_barrier_sync_fn(int bar_id, int count) {
    asm volatile("barrier.sync.aligned %0, %1;" :: "r"(bar_id), "r"(count));
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
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

__device__ __forceinline__ uint64_t make_smem_desc_k(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_mn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_k(uint32_t N, uint32_t M) {
    uint32_t d = (1u << 4) | (1u << 7) | (1u << 10);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_mn(uint32_t N, uint32_t M) {
    uint32_t d = (1u << 4) | (1u << 7) | (1u << 10);
    d |= (1u << 15);
    d |= (1u << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapDataType dataType, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(d, dataType, 2, globalAddress, globalDim, globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill);
}

#define CAST_GRID(num, clus) (((num) + (clus) - 1) / (clus) * (clus))

__global__ void __launch_bounds__(128) flash_attention_bwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    uint32_t S)
{
    setmaxnreg_inc_sync_fn<248>();
    
    uint32_t k_tile = blockIdx.x;
    uint32_t batch_head = blockIdx.y;
    float scale = 1.0f / sqrtf(128.0f);

    extern __shared__ char smem_raw[];
    __nv_bfloat16* smem_Q0 = (__nv_bfloat16*)smem_raw;
    __nv_bfloat16* smem_Q1 = smem_Q0 + 64*64;
    __nv_bfloat16* smem_K0 = smem_Q1 + 64*64;
    __nv_bfloat16* smem_K1 = smem_K0 + 64*64;
    __nv_bfloat16* smem_V0 = smem_K1 + 64*64;
    __nv_bfloat16* smem_V1 = smem_V0 + 64*64;
    __nv_bfloat16* smem_dO0 = smem_V1 + 64*64;
    __nv_bfloat16* smem_dO1 = smem_dO0 + 64*64;
    __nv_bfloat16* smem_O0 = smem_dO1 + 64*64;
    __nv_bfloat16* smem_O1 = smem_O0 + 64*64;
    __nv_bfloat16* smem_P = smem_O1 + 64*64;
    __nv_bfloat16* smem_dS = smem_P + 64*64;
    float* smem_D = (float*)(smem_dS + 64*64);
    float* smem_LSE = smem_D + 64;
    
    uint64_t* bar_kv = (uint64_t*)(smem_LSE + 64);
    uint64_t* bar_q = bar_kv + 1;
    
    uint32_t tmem_base;
    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_base, 512);
    }
    __syncthreads();
    
    uint32_t tmem_ST = tmem_base;
    uint32_t dP_half0 = tmem_base + 64;
    uint32_t dP_half1 = tmem_base + 128;
    uint32_t P_half0 = tmem_base + 192;
    uint32_t P_half1 = tmem_base + 256;
    uint32_t dV_half0 = tmem_base + 320;
    uint32_t dV_half1 = tmem_base + 384;
    
    uint32_t tmem_dST = tmem_base;
    uint32_t dQ_half0 = tmem_base + 64;
    uint32_t dQ_half1 = tmem_base + 128;
    uint32_t dK_half0 = tmem_base + 192;
    uint32_t dK_half1 = tmem_base + 256;
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar_kv, 1);
        init_smem_barrier_fn(&bar_q[0], 1);
        init_smem_barrier_fn(&bar_q[1], 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t first_q_tile = k_tile;

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_kv, 16384 * 4);
        tma_load_2d_fn(&tma_K, bar_kv, smem_K0, 0, k_tile * 64);
        tma_load_2d_fn(&tma_K, bar_kv, smem_K1, 64, k_tile * 64);
        tma_load_2d_fn(&tma_V, bar_kv, smem_V0, 0, k_tile * 64);
        tma_load_2d_fn(&tma_V, bar_kv, smem_V1, 64, k_tile * 64);
        
        if (first_q_tile * 64 < S) {
            mbarrier_arrive_and_expect_tx_fn(&bar_q[0], 16384 * 6);
            tma_load_2d_fn(&tma_Q, &bar_q[0], smem_Q0, 0, first_q_tile * 64);
            tma_load_2d_fn(&tma_Q, &bar_q[0], smem_Q1, 64, first_q_tile * 64);
            tma_load_2d_fn(&tma_dO, &bar_q[0], smem_dO0, 0, first_q_tile * 64);
            tma_load_2d_fn(&tma_dO, &bar_q[0], smem_dO1, 64, first_q_tile * 64);
            tma_load_2d_fn(&tma_O, &bar_q[0], smem_O0, 0, first_q_tile * 64);
            tma_load_2d_fn(&tma_O, &bar_q[0], smem_O1, 64, first_q_tile * 64);
        }
    }
    
    uint32_t phase[2] = {0, 0};
    uint32_t last_q_tile = (S + 63) / 64 - 1;

    if (threadIdx.x == 0) {
        mbarrier_wait_fn(bar_kv, 0);
        if (first_q_tile * 64 < S) {
            mbarrier_wait_fn(&bar_q[0], phase[0]);
            phase[0] ^= 1;
        }
    }
    __syncthreads();
    
    uint64_t desc_K0 = make_smem_desc_k(smem_K0, 0, 1024);
    uint64_t desc_K1 = make_smem_desc_k(smem_K1, 0, 1024);
    uint64_t desc_V0 = make_smem_desc_k(smem_V0, 0, 1024);
    uint64_t desc_V1 = make_smem_desc_k(smem_V1, 0, 1024);

    for (uint32_t q_tile = first_q_tile; q_tile <= last_q_tile; ++q_tile) {
        uint32_t curr = q_tile & 1;
        uint32_t next = (q_tile + 1) & 1;

        if (q_tile + 1 <= last_q_tile && threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&bar_q[next], 16384 * 6);
            tma_load_2d_fn(&tma_Q, &bar_q[next], smem_Q0, 0, (q_tile + 1) * 64);
            tma_load_2d_fn(&tma_Q, &bar_q[next], smem_Q1, 64, (q_tile + 1) * 64);
            tma_load_2d_fn(&tma_dO, &bar_q[next], smem_dO0, 0, (q_tile + 1) * 64);
            tma_load_2d_fn(&tma_dO, &bar_q[next], smem_dO1, 64, (q_tile + 1) * 64);
            tma_load_2d_fn(&tma_O, &bar_q[next], smem_O0, 0, (q_tile + 1) * 64);
            tma_load_2d_fn(&tma_O, &bar_q[next], smem_O1, 64,