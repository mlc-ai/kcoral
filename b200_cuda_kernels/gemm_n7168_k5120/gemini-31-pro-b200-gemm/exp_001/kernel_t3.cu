#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
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

#define CU_CHECK(call) do {                                        \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        fprintf(stderr, "CU error %d at %s:%d\n",                 \
                (int)_e, __FILE__, __LINE__);                      \
        exit(1);                                                   \
    }                                                              \
} while(0)

// Helper functions for SM100 architecture
template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

__device__ __forceinline__ void cluster_sync_fn() {
    asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n" ::: "memory");
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

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void prefetch_tma_descriptor_fn(const CUtensorMap* d) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(d) : "memory");
}

__device__ __forceinline__ void tma_load_2d_cg2_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t ba = (uint32_t)__cvta_generic_to_shared(&bar[0]) & 0xFEFFFFFF; 
    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(sa), "l"((uint64_t)d), "r"(c0), "r"(c1), "r"(ba) : "memory");
}

__device__ __forceinline__ void tma_store_2d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%2, %3}], [%1];"
        :: "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_store_fence_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tma_store_commit_fn() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void tma_store_wait_fn() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
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

__device__ __forceinline__ void tmem_load_16x_fn(uint32_t col,
    uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3,
    uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7,
    uint32_t* r8, uint32_t* r9, uint32_t* r10, uint32_t* r11,
    uint32_t* r12, uint32_t* r13, uint32_t* r14, uint32_t* r15) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3),
     "=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7),
     "=r"(*r8),"=r"(*r9),"=r"(*r10),"=r"(*r11),
     "=r"(*r12),"=r"(*r13),"=r"(*r14),"=r"(*r15) : "r"(col));
}

__device__ __forceinline__ uint32_t pack_bf16_fast_fn(uint32_t fp32_a, uint32_t fp32_b) {
    __nv_bfloat162 res = __floats2bfloat162_rn(__uint_as_float(fp32_a), __uint_as_float(fp32_b));
    return *reinterpret_cast<uint32_t*>(&res);
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

__device__ __forceinline__ void advance_desc_k(uint64_t& desc, uint32_t k_bytes) {
    uint32_t current_addr_encoded = desc & 0x3FFF;
    uint32_t new_addr_encoded = current_addr_encoded + (k_bytes >> 4);
    desc = (desc & ~0x3FFFull) | (new_addr_encoded & 0x3FFF);
}

CUresult create_tma_2d_descriptor(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapDataType dataType, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d,
        dataType,
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

struct alignas(1024) SharedStorage {
    union {
        struct {
            alignas(1024) __nv_bfloat16 A[3][2][128][64]; // 96 KB
            alignas(1024) __nv_bfloat16 B[3][2][128][64]; // 96 KB
        };
        alignas(1024) __nv_bfloat16 C[4][128][64]; // 64 KB
    };
    alignas(128) uint64_t tma_mbar[3];
    alignas(128) uint64_t umma_mbar[3];
};

__global__ void __launch_bounds__(128, 1) gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    const __grid_constant__ CUtensorMap tma_C,
    uint32_t M, uint32_t N, uint32_t K
) {
    setmaxnreg_inc_sync_fn<240>();
    
    if (threadIdx.x == 0) {
        prefetch_tma_descriptor_fn(&tma_A);
        prefetch_tma_descriptor_fn(&tma_B);
        prefetch_tma_descriptor_fn(&tma_C);
    }
    
    extern __shared__ __align__(1024) char smem_buf[];
    SharedStorage* smem = reinterpret_cast<SharedStorage*>(smem_buf);
    
    uint32_t pair_idx = blockIdx.x / 2;
    uint32_t rank_in_pair = blockIdx.x % 2;
    uint32_t M_TILES = (M + 255) / 256;
    uint32_t N_TILES = (N + 255) / 256;
    
    uint32_t swizzle_width = 8;
    uint32_t pairs_per_full_group = M_TILES * swizzle_width;
    uint32_t group_idx = pair_idx / pairs_per_full_group;
    uint32_t in_group_idx = pair_idx % pairs_per_full_group;
    
    uint32_t pair_m = in_group_idx % M_TILES;
    uint32_t pair_n = group_idx * swizzle_width + (in_group_idx / M_TILES);
    
    uint32_t global_m_start_0 = pair_m * 256;
    uint32_t global_m_start_1 = pair_m * 256 + 128;
    uint32_t global_n_start_0 = pair_n * 256;
    uint32_t global_n_start_1 = pair_n * 256 + 128;
    
    uint32_t tma_m_start = (rank_in_pair == 0) ? global_m_start_0 : global_m_start_1;
    uint32_t tma_n_start = (rank_in_pair == 0) ? global_n_start_0 : global_n_start_1;
    
    uint32_t my_m_start = tma_m_start;
    uint32_t my_n_start = global_n_start_0; 
    
    __shared__ uint32_t smem_tmem_c;
    if (threadIdx.x < 32) {
        tmem_alloc_fn(&smem_tmem_c, 256);
    }
    
    if (rank_in_pair == 0 && threadIdx.x == 0) {
        for (int s = 0; s < 3; s++) {
            init_smem_barrier_fn(&smem->tma_mbar[s], 1);
        }
    }
    if (threadIdx.x == 0) {
        for (int s = 0; s < 3; s++) {
            init_smem_barrier_fn(&smem->umma_mbar[s], 1);
        }
    }
    
    __syncthreads();
    if (threadIdx.x == 0) {
        fence_smem_barrier_init_fn();
    }
    cluster_sync_fn(); 

    uint32_t tmem_c = smem_tmem_c;
    int k_iters = K / 128;
    int tma_phase_0 = 0, tma_phase_1 = 0, tma_phase_2 = 0;
    int umma_phase_0 = 0, umma_phase_1 = 0, umma_phase_2 = 0;
    uint32_t idesc = make_instr_desc_fn(256, 256);
    
    // Prologue: Fill the 3 stages
    for (int s = 0; s < 3 && s < k_iters; s++) {
        if (rank_in_pair == 0 && threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&smem->tma_mbar[s], 131072);
        }
        if (threadIdx.x == 0) {
            tma_load_2d_cg2_fn(&tma_A, &smem->tma_mbar[s], smem->A[s][0], s * 128, tma_m_start);
            tma_load_2d_cg2_fn(&tma_A, &smem->tma_mbar[s], smem->A[s][1], s * 128 + 64, tma_m_start);
            tma_load_2d_cg2_fn(&tma_B, &smem->tma_mbar[s], smem->B[s][0], s * 128, tma_n_start);
            tma_load_2d_cg2_fn(&tma_B, &smem->tma_mbar[s], smem->B[s][1], s * 128 + 64, tma_n_start);
        }
    }
    
    // Main loop with properly staged pipelining
    for (int i = 0; i < k_iters; i++) {
        int s = i % 3;
        int current_tma_phase = (s == 0) ? tma_phase_0 : ((s == 1) ? tma_phase_1 : tma_phase_2);
        
        // Wait for TMA i
        if (rank_in_pair == 0 && threadIdx.x == 0) {
            mbarrier_wait_fn(&smem->tma_mbar[s], current_tma_phase);
            if (s == 0) tma_phase_0 ^= 1;
            else if (s == 1) tma_phase_1 ^= 1;
            else tma_phase_2 ^= 1;
            
            fence_proxy_async_fn();
            
            uint64_t dA0 = make_smem_desc_sm100_fn(smem->A[s][0], 1, 1024);
            uint64_t dA1 = make_smem_desc_sm100_fn(smem->A[s][1], 1, 1024);
            uint64_t dB0 = make_smem_desc_sm100_fn(smem->B[s][0], 1, 1024);
            uint64_t dB1 = make_smem_desc_sm100_fn(smem->B[s][1], 1, 1024);
            
            #pragma unroll
            for (int k_step = 0; k_step < 8; k_step++) {
                uint64_t dA = (k_step < 4) ? dA0 : dA1;
                uint64_t dB = (k_step < 4) ? dB0 : dB1;
                advance_desc_k(dA, (k_step % 4) * 32);
                advance_desc_k(dB, (k_step % 4) * 32);
                
                uint32_t accum = (i == 0 && k_step == 0) ? 0 : 1;
                umma_f16_cg2_fn(tmem_c, dA, dB, idesc, accum);
            }
            
            umma_commit_2sm_fn(&smem->umma_mbar[s]);
        }
        
        // Load for future (load_idx = i + 2)
        int load_idx = i + 2;
        if (load_idx < k_iters && i >= 1) {
            int load_s = load_idx % 3;
            int prev_umma_phase = (load_s == 0) ? umma_phase_0 : ((load_s == 1) ? umma_phase_1 : umma_phase_2);
            
            if (threadIdx.x == 0) {
                // Wait for Compute (i-1) to finish before overwriting its buffer
                mbarrier_wait_fn(&smem->umma_mbar[load_s], prev_umma_phase);
                if (load_s == 0) umma_phase_0 ^= 1;
                else if (load_s == 1) umma_phase_1 ^= 1;
                else umma_phase_2 ^= 1;
            }
            
            if (rank_in_pair == 0 && threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&smem->tma_mbar[load_s], 131072);
            }
            if (threadIdx.x == 0) {
                tma_load_2d_cg2_fn(&tma_A, &smem->tma_mbar[load_s], smem->A[load_s][0], load_idx * 128, tma_m_start);
                tma_load_2d_cg2_fn(&tma_A, &smem->tma_mbar[load_s], smem->A[load_s][1], load_idx * 128 + 64, tma_m_start);
                tma_load_2d_cg2_fn(&tma_B, &smem->tma_mbar[load_s], smem->B[load_s][0], load_idx * 128, tma_n_start);
                tma_load_2d_cg2_fn(&tma_B, &smem->tma_mbar[load_s], smem->B[load_s][1], load_idx * 128 + 64, tma_n_start);
            }
        }
    }
    
    // Epilogue wait for the last two computes
    if (threadIdx.x == 0) {
        for (int i = max(0, k_iters - 2); i < k_iters; i++) {
            int s = i % 3;
            int phase = (s == 0) ? umma_phase_0 : ((s == 1) ? umma_phase_1 : umma_phase_2);
            mbarrier_wait_fn(&smem->umma_mbar[s], phase);
            if (s == 0) umma_phase_0 ^= 1;
            else if (s == 1) umma_phase_1 ^= 1;
            else umma_phase_2 ^= 1;
        }
    }
    __syncthreads();
    
    // Epilogue store: TMEM -> SMEM -> Global via TMA
    for (uint32_t c_step = 0; c_step < 256; c_step += 64) {
        uint32_t r0[16], r1[16], r2[16], r3[16];
        
        tmem_load_16x_fn(tmem_c + c_step + 0, 
            &r0[0], &r0[1], &r0[2], &r0[3], &r0[4], &r0[5], &r0[6], &r0[7], 
            &r0[8], &r0[9], &r0[10], &r0[11], &r0[12], &r0[13], &r0[14], &r0[15]);
        
        tmem_load_16x_fn(tmem_c + c_step + 16, 
            &r1[0], &r1[1], &r1[2], &r1[3], &r1[4], &r1[5], &r1[6], &r1[7], 
            &r1[8], &r1[9], &r1[10], &r1[11], &r1[12], &r1[13], &r1[14], &r1[15]);
            
        tmem_load_16x_fn(tmem_c + c_step + 32, 
            &r2[0], &r2[1], &r2[2], &r2[3], &r2[4], &r2[5], &r2[6], &r2[7], 
            &r2[8], &r2[9], &r2[10], &r2[11], &r2[12], &r2[13], &r2[14], &r2[15]);
            
        tmem_load_16x_fn(tmem_c + c_step + 48, 
            &r3[0], &r3[1], &r3[2], &r3[3], &r3[4], &r3[5], &r3[6], &r3[7], 
            &r3[8], &r3[9], &r3[10], &r3[11], &r3[12], &r3[13], &r3[14], &r3[15]);
            
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        auto store_chunk = [&](uint32_t* r, uint32_t offset) {
            uint32_t p0 = pack_bf16_fast_fn(r[0], r[1]);
            uint32_t p1 = pack_bf16_fast_fn(r[2], r[3]);
            uint32_t p2 = pack_bf16_fast_fn(r[4], r[5]);
            uint32_t p3 = pack_bf16_fast_fn(r[6], r[7]);
            
            uint32_t p4 = pack_bf16_fast_fn(r[8], r[9]);
            uint32_t p5 = pack_bf16_fast_fn(r[10], r[11]);
            uint32_t p6 = pack_bf16_fast_fn(r[12], r[13]);
            uint32_t p7 = pack_bf16_fast_fn(r[14], r[15]);
            
            uint32_t step = c_step + offset;
            uint32_t block_idx = step / 64;
            uint32_t col_in_block0 = step % 64;
            uint32_t col_in_block1 = col_in_block0 + 8;
            uint32_t row = threadIdx.x;
            
            uint32_t x0 = col_in_block0 / 8;
            uint32_t swizzled_x0 = (row % 8) ^ x0;
            *reinterpret_cast<uint4*>(&smem->C[block_idx][row][swizzled_x0 * 8]) = make_uint4(p0, p1, p2, p3);
            
            uint32_t x1 = col_in_block1 / 8;
            uint32_t swizzled_x1 = (row % 8) ^ x1;
            *reinterpret_cast<uint4*>(&smem->C[block_idx][row][swizzled_x1 * 8]) = make_uint4(p4, p5, p6, p7);
        };
        
        store_chunk(r0, 0);
        store_chunk(r1, 16);
        store_chunk(r2, 32);
        store_chunk(r3, 48);
    }
    
    __syncthreads();
    
    tma_store_fence_fn();
    if (threadIdx.x == 0) {
        for (int b = 0; b < 4; b++) {
            tma_store_2d_fn(&tma_C, smem->C[b], my_n_start + b * 64, my_m_start);
        }
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
    }
    
    __syncthreads();
    
    if (threadIdx.x < 32) {
        tmem_dealloc_fn(tmem_c, 256);
    }
}

namespace tvm_ffi_example_cuda {

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0);
    
    CUtensorMap tma_A, tma_B, tma_C;
    CU_CHECK(create_tma_2d_descriptor(&tma_A, A.data_ptr(), K, M, 64, 128, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor(&tma_B, B.data_ptr(), K, N, 64, 128, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor(&tma_C, C.data_ptr(), N, M, 64, 128, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    int M_TILES = (M + 255) / 256;
    int N_TILES = (N + 255) / 256;
    int grid_x = M_TILES * N_TILES * 2;

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute((void*)gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedStorage)));

    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(grid_x, 1, 1);
    config.blockDim = dim3(128, 1, 1);
    config.dynamicSmemBytes = sizeof(SharedStorage);
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_kernel, tma_A, tma_B, tma_C, M, N, K));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda