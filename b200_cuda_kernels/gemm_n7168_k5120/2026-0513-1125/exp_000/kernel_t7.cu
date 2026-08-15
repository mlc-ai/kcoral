#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <stdio.h>
#include <stdint.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

#define CU_CHECK(call) do {                                        \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        const char* err_str;                                       \
        cuGetErrorString(_e, &err_str);                            \
        fprintf(stderr, "CU error %s at %s:%d\n",                  \
                err_str, __FILE__, __LINE__);                      \
        exit(1);                                                   \
    }                                                              \
} while(0)

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
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

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tma_load_2d_cg2_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t ba = (uint32_t)__cvta_generic_to_shared(&bar[0]) & 0xFEFFFFFF;
    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(sa), "l"((uint64_t)d), "r"(c0), "r"(c1), "r"(ba) : "memory");
}

__device__ __forceinline__ void tma_load_multicast_2d_cg2_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, uint16_t mask) {
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t ba = (uint32_t)__cvta_generic_to_shared(&bar[0]) & 0xFEFFFFFF;
    uint64_t cache_hint = 0;
    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
        " [%0], [%1, {%4, %5}], [%2], %3, %6;"
        :: "r"(sa), "l"((uint64_t)d), "r"(ba), "h"(mask), "r"(c0), "r"(c1), "l"(cache_hint) : "memory");
}

__device__ __forceinline__ void prefetch_tma_descriptor_fn(const CUtensorMap* d) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(d) : "memory");
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

__device__ __forceinline__ void umma_f16_cg2_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_2sm_fn(uint64_t* bar, uint16_t mask) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::2"
        ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;"
        :: "r"(a), "h"(mask));
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)((addr >> 4) & 0x3FFF);
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    uint32_t base_offset = (addr >> 7) & 0x7;
    d |= (uint64_t)base_offset << 49;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);           // FP32 output
    d |= (1u << 7);           // BF16 input A
    d |= (1u << 10);          // BF16 input B
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    d |= (3u << 30);          // Max internal hardware shift for optimal B-Matrix register reuse
    return d;
}

__device__ __forceinline__ uint32_t pack_bf16_fast(uint32_t f0, uint32_t f1) {
    uint32_t res;
    asm volatile(
        "cvt.rn.bf16x2.f32x2 %0, %1, %2;" 
        : "=r"(res) : "f"(__uint_as_float(f1)), "f"(__uint_as_float(f0))
    );
    return res;
}

__device__ __forceinline__ void tmem_epilogue_fn(
    __nv_bfloat16* D, __nv_bfloat16* smem_out, uint32_t tmem_base,
    uint32_t M, uint32_t N, uint32_t m_idx, uint32_t n_idx,
    uint32_t BM, uint32_t BN) {
    
    // Bank Conflict mitigated symmetrically through 8 element (16 byte) pitch expansion 
    uint32_t BN_PAD = BN + 8; // 264
    
    // Unrolled Phase 1: Batched TMEM -> SMEM translation targeting lowest instruction latency boundaries 
    for (uint32_t col = 0; col < BN; col += 32) {
        uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
        uint32_t r8, r9, r10, r11, r12, r13, r14, r15;
        uint32_t r16, r17, r18, r19, r20, r21, r22, r23;
        uint32_t r24, r25, r26, r27, r28, r29, r30, r31;
        
        uint32_t src0 = tmem_base + col;
        uint32_t src1 = tmem_base + col + 8;
        uint32_t src2 = tmem_base + col + 16;
        uint32_t src3 = tmem_base + col + 24;
        
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(src0));
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                     : "=r"(r8),"=r"(r9),"=r"(r10),"=r"(r11),"=r"(r12),"=r"(r13),"=r"(r14),"=r"(r15) : "r"(src1));
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                     : "=r"(r16),"=r"(r17),"=r"(r18),"=r"(r19),"=r"(r20),"=r"(r21),"=r"(r22),"=r"(r23) : "r"(src2));
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                     : "=r"(r24),"=r"(r25),"=r"(r26),"=r"(r27),"=r"(r28),"=r"(r29),"=r"(r30),"=r"(r31) : "r"(src3));
        
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        uint32_t base = threadIdx.x * BN_PAD + col;
        *reinterpret_cast<uint4*>(&smem_out[base])      = make_uint4(pack_bf16_fast(r0, r1), pack_bf16_fast(r2, r3), pack_bf16_fast(r4, r5), pack_bf16_fast(r6, r7));
        *reinterpret_cast<uint4*>(&smem_out[base + 8])  = make_uint4(pack_bf16_fast(r8, r9), pack_bf16_fast(r10, r11), pack_bf16_fast(r12, r13), pack_bf16_fast(r14, r15));
        *reinterpret_cast<uint4*>(&smem_out[base + 16]) = make_uint4(pack_bf16_fast(r16, r17), pack_bf16_fast(r18, r19), pack_bf16_fast(r20, r21), pack_bf16_fast(r22, r23));
        *reinterpret_cast<uint4*>(&smem_out[base + 24]) = make_uint4(pack_bf16_fast(r24, r25), pack_bf16_fast(r26, r27), pack_bf16_fast(r28, r29), pack_bf16_fast(r30, r31));
    }
    __syncthreads();
    
    // Phase 2: SMEM -> Global Memory (pure 128x256 uint4 coalescing targeting flat GMEM peak bandwidth constraints)
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t num_steps = (BM + 3) / 4;
    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 4 + warp_id;
        if (row >= BM) continue;
        uint32_t global_row = m_idx + row;
        
        uint32_t col_start = lane_id * 8; // Vectorize 8 items -> 16B aligned contiguous span
        uint32_t global_col = n_idx + col_start;
        
        if (col_start < BN && global_row < M && global_col + 7 < N) {
            uint4 data = *reinterpret_cast<uint4*>(&smem_out[row * BN_PAD + col_start]);
            *reinterpret_cast<uint4*>(D + (uint64_t)global_row * N + global_col) = data;
        } else {
            // Guarded memory edge fallback routing
            for (int i = 0; i < 8; ++i) {
                if (col_start + i < BN && global_row < M && global_col + i < N) {
                    D[global_row * N + global_col + i] = smem_out[row * BN_PAD + col_start + i];
                }
            }
        }
    }
}

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, 
                                     uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, 
                                     uint32_t smem_inner_dim, uint32_t smem_outer_dim, 
                                     CUtensorMapDataType dataType, CUtensorMapSwizzle swizzle, 
                                     CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, dataType, 2, globalAddress, globalDim, globalStrides, boxDim,
        elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill
    );
}

// Memory footprint optimized to strictly 96KB, assuring exact 2 Blocks / SM Density Multipliers natively  
struct SharedStorage {
    alignas(1024) __nv_bfloat16 A[3][128][64]; 
    alignas(1024) __nv_bfloat16 B[3][128][64];  
    alignas(16) uint64_t tma_bar[3];           
    alignas(16) uint64_t umma_bar[3];          
};

__global__ __launch_bounds__(128, 2) void gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* C_ptr,
    int M, int N, int K,
    uint16_t mask_even, uint16_t mask_odd) 
{
    extern __shared__ char smem_buf_raw[];
    uintptr_t raw_ptr = reinterpret_cast<uintptr_t>(smem_buf_raw);
    uintptr_t aligned_ptr = (raw_ptr + 1023) & ~1023;
    SharedStorage* smem = reinterpret_cast<SharedStorage*>(aligned_ptr);

    if (threadIdx.x == 0) {
        prefetch_tma_descriptor_fn(&tma_A);
        prefetch_tma_descriptor_fn(&tma_B);
    }

    int rank = cluster_rank_fn();
    
    int pair_rank = rank % 2;
    int is_leader = (rank % 2 == 0);
    int cluster_x = blockIdx.x / 2;
    int cluster_y = blockIdx.y;

    // Explicit Snake Block Rasterization routing for near 100% L2 Cache persistence of Inner matrices 
    if (cluster_y % 2 == 1) {
        cluster_x = (gridDim.x / 2) - 1 - cluster_x;
    }

    int m_idx = cluster_x * 256 + pair_rank * 128;
    int n_idx = cluster_y * 256 + pair_rank * 128;

    if (threadIdx.x == 0) {
        if (rank == 0) {
            for (int i = 0; i < 3; ++i) {
                init_smem_barrier_fn(&smem->tma_bar[i], 1);
            }
        }
        for (int i = 0; i < 3; ++i) {
            init_smem_barrier_fn(&smem->umma_bar[i], 1);
        }
        fence_smem_barrier_init_fn();
    }
    cluster_sync_fn();

    __shared__ uint32_t tmem_addr;
    if (threadIdx.x < 32) {
        tmem_alloc_fn(&tmem_addr, 256);
    }
    __syncthreads();

    int K_ITERS = K / 64;
    int tma_phase[3] = {0, 0, 0};
    int umma_phase[3] = {0, 0, 0};

    // Precalculate runtime UMMA array descriptors, stripping arithmetic bindings from innermost execution
    uint64_t desc_a_arr[3];
    uint64_t desc_b_arr[3];
    if (threadIdx.x == 0) {
        for (int i = 0; i < 3; ++i) {
            desc_a_arr[i] = make_smem_desc_sm100_fn(&smem->A[i][0][0], 1024);
            desc_b_arr[i] = make_smem_desc_sm100_fn(&smem->B[i][0][0], 1024);
        }
    }
    uint32_t idesc = make_instr_desc_fn(256, 256);

    // Initial pre-load stage (Fill overlapping depth pipeline)
    for (int iter = 0; iter < 2; ++iter) {
        int s = iter % 3;
        if (threadIdx.x == 0) {
            if (is_leader) {
                mbarrier_arrive_and_expect_tx_fn(&smem->tma_bar[s], 65536);
            }
            
            tma_load_2d_cg2_fn(&tma_A, &smem->tma_bar[s], &smem->A[s][0][0], iter * 64, m_idx);
            
            // Multicast loading specifically targeting a 75% memory load reduction
            if (rank == 0 || rank == 1) {
                uint16_t mask = (rank == 0) ? mask_even : mask_odd;
                tma_load_multicast_2d_cg2_fn(&tma_B, &smem->tma_bar[s], &smem->B[s][0][0], iter * 64, n_idx, mask);
            }
        }
    }

    #pragma unroll 1
    for (int iter = 0; iter < K_ITERS; ++iter) {
        int s_load = (iter + 2) % 3;
        int s_comp = iter % 3;

        // Start loading iter+2 (pipelined overlapping staging buffer)
        if (iter + 2 < K_ITERS) {
            if (iter >= 1) {
                if (threadIdx.x == 0) {
                    mbarrier_wait_fn(&smem->umma_bar[s_load], umma_phase[s_load]);
                }
                umma_phase[s_load] ^= 1;
            }
            if (threadIdx.x == 0) {
                if (is_leader) {
                    mbarrier_arrive_and_expect_tx_fn(&smem->tma_bar[s_load], 65536);
                }
                
                tma_load_2d_cg2_fn(&tma_A, &smem->tma_bar[s_load], &smem->A[s_load][0][0], (iter + 2) * 64, m_idx);
                
                if (rank == 0 || rank == 1) {
                    uint16_t mask = (rank == 0) ? mask_even : mask_odd;
                    tma_load_multicast_2d_cg2_fn(&tma_B, &smem->tma_bar[s_load], &smem->B[s_load][0][0], (iter + 2) * 64, n_idx, mask);
                }
            }
        }

        if (is_leader) {
            if (threadIdx.x == 0) {
                mbarrier_wait_fn(&smem->tma_bar[s_comp], tma_phase[s_comp]);
                fence_async_shared_fn();

                uint64_t da = desc_a_arr[s_comp];
                uint64_t db = desc_b_arr[s_comp];

                // Unroll exactly 4 steps addressing K=64 bounds across Native K=16 Matrix blocks natively  
                uint32_t acc0 = (iter == 0) ? 0 : 1;
                umma_f16_cg2_fn(tmem_addr, da, db, idesc, acc0);
                da += 2; db += 2;
                umma_f16_cg2_fn(tmem_addr, da, db, idesc, 1);
                da += 2; db += 2;
                umma_f16_cg2_fn(tmem_addr, da, db, idesc, 1);
                da += 2; db += 2;
                umma_f16_cg2_fn(tmem_addr, da, db, idesc, 1);
                
                uint16_t pair_mask = 3 << (rank & ~1);
                umma_commit_2sm_fn(&smem->umma_bar[s_comp], pair_mask);
            }
            tma_phase[s_comp] ^= 1;
        }
    }

    // Barrier isolation preventing raw memory collisions over the final wrap-up 
    int s_last = (K_ITERS - 1) % 3;
    if (threadIdx.x == 0) {
        mbarrier_wait_fn(&smem->umma_bar[s_last], umma_phase[s_last]);
    }
    umma_phase[s_last] ^= 1;
    __syncthreads();

    // Recycle physically expanded raw SMEM buffers directly for final TMEM parallel coalescing exports
    __nv_bfloat16* smem_out = (__nv_bfloat16*)smem->A;
    tmem_epilogue_fn(C_ptr, smem_out, tmem_addr, M, N, m_idx, n_idx, 128, 256);

    __syncthreads();
    if (threadIdx.x < 32) {
        tmem_dealloc_fn(tmem_addr, 256);
    }
}

namespace tvm_ffi_gemm {

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0);
    
    if (M == 0) return;

    CUtensorMap tma_A, tma_B;
    // Overkill 256B L2 Promotion configurations systematically securing maximal global locality hits 
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_A, A.data_ptr(), K, M, 64, 128, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    CU_CHECK(create_tma_2d_descriptor_2B(&tma_B, B.data_ptr(), K, N, 64, 128, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    int grid_x = (M + 255) / 256 * 2;
    int grid_y = (N + 255) / 256;
    
    int cluster_x_dim = 8;
    if (grid_x % 8 == 0) cluster_x_dim = 8;
    else if (grid_x % 4 == 0) cluster_x_dim = 4;
    else cluster_x_dim = 2;
    
    uint16_t mask_even = 0;
    uint16_t mask_odd = 0;
    for (int i = 0; i < cluster_x_dim; i += 2) {
        mask_even |= (1 << i);
        mask_odd |= (1 << (i + 1));
    }

    dim3 grid(grid_x, grid_y, 1);
    dim3 block(128, 1, 1);

    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    // Guaranteed padded buffer allocations isolating SMEM constraints perfectly 
    config.dynamicSmemBytes = sizeof(SharedStorage) + 1024;
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = cluster_x_dim;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    CUDA_CHECK(cudaFuncSetAttribute((void*)gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedStorage) + 1024));
    CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_kernel, tma_A, tma_B, static_cast<__nv_bfloat16*>(C.data_ptr()), M, N, K, mask_even, mask_odd));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_gemm