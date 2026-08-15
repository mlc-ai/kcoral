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
        fprintf(stderr, "CU error %d at %s:%d\n",                  \
                _e, __FILE__, __LINE__);                           \
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

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ void tmem_load_packed_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ void tmem_store_32x32b_x1(uint32_t col, uint32_t val) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 %0, [%1];" :: "r"(val), "r"(col));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_clear_128x128_packed(uint32_t tmem_S_T) {
    for (int c = 0; c < 64; c++) {
        tmem_store_32x32b_x1(tmem_S_T + c, 0);
    }
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ uint32_t fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return __float_as_uint(y);
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

__device__ __forceinline__ uint32_t make_instr_desc_fn_mn_b(uint32_t M, uint32_t N) {
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

__device__ __forceinline__ uint64_t tmem_load_store_add_64bit(uint64_t addr, uint32_t offset) {
    return addr + offset;
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

__device__ __forceinline__ int swizzle_128B(int x, int y) {
    return (((y % 8) ^ (x / 8)) * 8) + (x % 8);
}

// ---------------- GLOBAL VARIABLES ----------------
__device__ unsigned int __grid_sync_count = 0;
__device__ volatile int __grid_sync_sense = 0;

__device__ __forceinline__ void grid_sync_fn() {
    __syncthreads();
    __threadfence();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        unsigned int num_blocks = gridDim.x * gridDim.y * gridDim.z;
        unsigned int arrived = atomicAdd(&__grid_sync_count, 1);
        if (arrived == num_blocks - 1) {
            __grid_sync_count = 0;
            __threadfence();
            __grid_sync_sense ^= 1;
        } else {
            int expected = __grid_sync_sense ^ 1;
            while (__grid_sync_sense != expected) {}
        }
    }
    __syncthreads();
}

// ---------------- KERNEL DEFINITION ----------------
__global__ __launch_bounds__(128) void causal_bwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q, 
    const __grid_constant__ CUtensorMap tma_K, 
    const __grid_constant__ CUtensorMap tma_V, 
    const __grid_constant__ CUtensorMap tma_dO, 
    const __grid_constant__ CUtensorMap tma_O, 
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV, 
    const float* L, int num_blocks, int M_global, int N_global) 
{
    extern __shared__ char smem[];
    uintptr_t smem_addr = (uintptr_t)smem;
    if (smem_addr % 1024 != 0) {
        smem = (char*)smem + (1024 - (smem_addr % 1024));
    }

    __nv_bfloat16* smem_Q = (__nv_bfloat16*)smem;
    __nv_bfloat16* smem_K = smem_Q + 128 * 128;
    __nv_bfloat16* smem_V = smem_K + 128 * 128;
    __nv_bfloat16* smem_dO = smem_V + 128 * 128;
    __nv_bfloat16* smem_P_T = smem_dO + 128 * 128;
    __nv_bfloat16* smem_dS_T = smem_P_T + 128 * 128;
    float* D_local = (float*)(smem_dS_T + 128 * 128);
    float* LSE_local = D_local + 128;

    uint64_t* mbar_Q = (uint64_t*)(LSE_local + 128);
    uint64_t* mbar_dO = mbar_Q + 1;
    uint64_t* mbar_K = mbar_dO + 1;
    uint64_t* mbar_V = mbar_K + 1;
    uint64_t* mbar_S = mbar_V + 1;
    uint64_t* mbar_dP = mbar_S + 1;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(mbar_dO, 1);
        init_smem_barrier_fn(mbar_K, 1);
        init_smem_barrier_fn(mbar_V, 1);
        init_smem_barrier_fn(mbar_S, 1);
        init_smem_barrier_fn(mbar_dP, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t tmem_base;
    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_base, 512);
    }
    __syncthreads();
    uint32_t tmem_dQ = tmem_base;
    uint32_t tmem_dK = tmem_base + 64;
    uint32_t tmem_dV = tmem_base + 128;
    uint32_t tmem_S_T = tmem_base + 192;
    uint32_t tmem_dP_T = tmem_base + 256;
    uint32_t tmem_dS_T = tmem_base + 320;
    uint32_t tmem_P_T = tmem_S_T;
    uint32_t tmem_K_T = tmem_base + 384;
    uint32_t tmem_dS_T_tm = tmem_base + 448;

    const __nv_bfloat16* O_ptr = (const __nv_bfloat16*)tma_O; // Unrolled hack to bypass compiler limitation
    int bh = blockIdx.y;
    int b_off = bh * M_global; 

    uint32_t phase_Q = 0, phase_dO = 0, phase_K =