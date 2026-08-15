#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <stdio.h>
#include <cstdint>
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

// Helper functions for SM100
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

__device__ __forceinline__ void mbarrier_expect_tx_cg2_fn(uint64_t* bar, uint32_t tx_bytes) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    uint32_t remote_a;
    asm volatile("mapa.shared::cluster.u32 %0, %1, 0;"
                 : "=r"(remote_a) : "r"(a));
    asm volatile("mbarrier.arrive.expect_tx.shared::cluster.b64 _, [%0], %1;"
                 :: "r"(remote_a), "r"(tx_bytes) : "memory");
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

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
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

__device__ __forceinline__ void tma_load_2d_cg2_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t ba = (uint32_t)__cvta_generic_to_shared(&bar[0]) & 0xFEFFFFFF;
    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(sa), "l"((uint64_t)d), "r"(c0), "r"(c1), "r"(ba) : "memory");
}

__device__ __forceinline__ void prefetch_tma_descriptor_fn(const CUtensorMap* d) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(d) : "memory");
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

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr) >> 4;
    d |= (uint64_t)(addr & 0x3FFF);
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;   // version = 1
    d |= (uint64_t)2 << 61;   // layout_type = SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);           // c_format = FP32
    d |= (1u << 7);           // a_format = BF16
    d |= (1u << 10);          // b_format = BF16
    d |= ((N / 8) << 17);     // n_dim
    d |= ((M / 16) << 24);    // m_dim
    return d;
}

__device__ __forceinline__ uint32_t pack_bf16_fn(uint32_t fp32_a, uint32_t fp32_b) {
    float2 f;
    f.x = __uint_as_float(fp32_a);
    f.y = __uint_as_float(fp32_b);
    __nv_bfloat162 b = __float22bfloat162_rn(f);
    return *reinterpret_cast<uint32_t*>(&b);
}

__device__ __forceinline__ void my_tmem_epilogue_coalesced(
    __nv_bfloat16* D, __nv_bfloat16* smem_out,
    uint32_t M, uint32_t N, uint32_t m_block_start, uint32_t n_block_start,
    uint32_t BM, uint32_t BN, uint32_t tmem_c) {
    
    uint32_t BN_stride = BN + 8; // Pad row to avoid bank conflicts

    // Phase 1: TMEM -> SMEM (Use all 256 threads to read full TMEM in 8 iterations)
    if (threadIdx.x < 256) {
        uint32_t wg_id = threadIdx.x / 128;
        uint32_t lane = threadIdx.x % 128;
        
        #pragma unroll 1
        for (uint32_t col = wg_id * 16; col < BN; col += 32) {
            uint32_t r[16];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),
                  "=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
                  "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),
                  "=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15])
                : "r"(tmem_c + col)
            );
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

            #pragma unroll
            for(int i = 0; i < 2; ++i) {
                uint32_t base = lane * BN_stride + col + i*8;
                uint32_t pack0 = pack_bf16_fn(r[i*8 + 0], r[i*8 + 1]);
                uint32_t pack1 = pack_bf16_fn(r[i*8 + 2], r[i*8 + 3]);
                uint32_t pack2 = pack_bf16_fn(r[i*8 + 4], r[i*8 + 5]);
                uint32_t pack3 = pack_bf16_fn(r[i*8 + 6], r[i*8 + 7]);
                *reinterpret_cast<uint4*>(&smem_out[base]) = make_uint4(pack0, pack1, pack2, pack3);
            }
        }
    }
    __syncthreads();
    
    // Phase 2: SMEM -> Global (Use all 256 threads to write in parallel)
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t num_warps = blockDim.x / 32;
    
    for (uint32_t row = warp_id; row < BM; row += num_warps) {
        uint32_t global_row = m_block_start + row;
        if (global_row >= M) continue;
        
        #pragma unroll
        for (uint32_t c_step = 0; c_step < BN; c_step += 256) {
            uint32_t col_start = c_step + lane_id * 8;
            uint32_t global_col = n_block_start + col_start;
            
            if (global_col < N) {
                if (global_col + 7 < N) {
                    uint4 data = *reinterpret_cast<uint4*>(&smem_out[row * BN_stride + col_start]);
                    *reinterpret_cast<uint4*>(D + (uint64_t)global_row * N + global_col) = data;
                } else {
                    for (int i = 0; i < 8; ++i) {
                        if (global_col + i < N) {
                            D[(uint64_t)global_row * N + global_col + i] = smem_out[row * BN_stride + col_start + i];
                        }
                    }
                }
            }
        }
    }
}

__global__ void __launch_bounds__(256, 2) gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* C,
    uint32_t M, uint32_t N, uint32_t K)
{
    // Restrict to 128 registers to allow 2 CTAs per SM (256 * 128 = 32768, 65536 per SM)
    setmaxnreg_inc_sync_fn<128>();

    if (threadIdx.x == 0) {
        prefetch_tma_descriptor_fn(&tma_A);
        prefetch_tma_descriptor_fn(&tma_B);
    }

    uint32_t rank = cluster_rank_fn();
    uint32_t cluster_idx_x = blockIdx.x / 2;
    uint32_t cluster_idx_y = blockIdx.y;
    
    uint32_t n_block_start = cluster_idx_x * 256;
    uint32_t m_block_start = cluster_idx_y * 256;
    
    uint32_t my_m = m_block_start + rank * 128;
    uint32_t my_n_b = n_block_start + rank * 128;

    uint32_t safe_my_m = (my_m < M) ? my_m : (M > 0 ? M - 1 : 0);
    uint32_t safe_my_n_b = (my_n_b < N) ? my_n_b : (N > 0 ? N - 1 : 0);

    __shared__ uint32_t tmem_c;
    if (threadIdx.x < 32) {
        tmem_alloc_fn(&tmem_c, 256);
    }
    __syncthreads();

    constexpr int STAGES = 6;
    union SharedStorage {
        struct {
            uint8_t A[STAGES][16384];
            uint8_t B[STAGES][16384];
        } tma;
        __nv_bfloat16 out[128 * 264]; 
    };
    __shared__ __align__(1024) SharedStorage smem;
    __shared__ __align__(8) uint64_t mbar_tma[STAGES];
    __shared__ __align__(8) uint64_t mbar_umma[STAGES];

    uint32_t idesc = make_instr_desc_fn(256, 256);

    if (threadIdx.x == 0) {
        for (int p = 0; p < STAGES; ++p) {
            init_smem_barrier_fn(&mbar_tma[p], 2);
            init_smem_barrier_fn(&mbar_umma[p], 1); 
        }
        fence_smem_barrier_init_fn();
    }
    cluster_sync_fn();

    uint64_t desc_A[STAGES][4];
    uint64_t desc_B[STAGES][4];
    if (rank == 0 && threadIdx.x == 0) {
        for (int b = 0; b < STAGES; ++b) {
            for (int k = 0; k < 4; ++k) {
                desc_A[b][k] = make_smem_desc_sm100_fn(smem.tma.A[b] + k * 32, 1024);
                desc_B[b][k] = make_smem_desc_sm100_fn(smem.tma.B[b] + k * 32, 1024);
            }
        }
    }

    int num_steps = K / 64;
    int phase_tma[STAGES] = {0};
    int phase_umma[STAGES] = {0};

    // Prologue (TMA worker thread 128)
    if (threadIdx.x == 128) {
        for (int step = 0; step < STAGES - 1 && step < num_steps; ++step) {
            mbarrier_expect_tx_cg2_fn(&mbar_tma[step], 32768);
            tma_load_2d_cg2_fn(&tma_A, &mbar_tma[step], smem.tma.A[step], step * 64, safe_my_m);
            tma_load_2d_cg2_fn(&tma_B, &mbar_tma[step], smem.tma.B[step], step * 64, safe_my_n_b);
        }
    }

    // Main Compute Loop: Split between UMMA worker (0) and TMA worker (128)
    for (int step = 0; step < num_steps; ++step) {
        int buf = step % STAGES;
        int next_fetch_step = step + STAGES - 1;
        int next_fetch_buf = next_fetch_step % STAGES;

        // UMMA worker
        if (threadIdx.x == 0) {
            mbarrier_wait_fn(&mbar_tma[buf], phase_tma[buf]);
            
            if (rank == 0) {
                #pragma unroll
                for (int k_mma = 0; k_mma < 4; ++k_mma) {
                    uint32_t accum = (step == 0 && k_mma == 0) ? 0 : 1;
                    umma_f16_cg2_fn(tmem_c, desc_A[buf][k_mma], desc_B[buf][k_mma], idesc, accum);
                }
                umma_commit_2sm_fn(&mbar_umma[buf]);
            }
        }
        phase_tma[buf] ^= 1;

        // TMA worker
        if (next_fetch_step < num_steps) {
            if (threadIdx.x == 128) {
                mbarrier_wait_fn(&mbar_umma[next_fetch_buf], phase_umma[next_fetch_buf]);
                
                mbarrier_expect_tx_cg2_fn(&mbar_tma[next_fetch_buf], 32768);
                tma_load_2d_cg2_fn(&tma_A, &mbar_tma[next_fetch_buf], smem.tma.A[next_fetch_buf], next_fetch_step * 64, safe_my_m);
                tma_load_2d_cg2_fn(&tma_B, &mbar_tma[next_fetch_buf], smem.tma.B[next_fetch_buf], next_fetch_step * 64, safe_my_n_b);
            }
            phase_umma[next_fetch_buf] ^= 1;
        }
    }

    // Thread 0 ensures final UMMA is completed before all threads read TMEM
    if (num_steps > 0) {
        if (threadIdx.x == 0) {
            int last_buf = (num_steps - 1) % STAGES;
            mbarrier_wait_fn(&mbar_umma[last_buf], phase_umma[last_buf]);
            asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
        }
    }

    __syncthreads();
    
    if (threadIdx.x < 128) {
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    }

    // Epilogue
    my_tmem_epilogue_coalesced(C, smem.out, M, N, my_m, n_block_start, 128, 256, tmem_c);

    cluster_sync_fn(); 
    if (threadIdx.x < 32) {
        tmem_dealloc_fn(tmem_c, 256);
    }
}

namespace tvm_ffi_example_cuda {

CUresult create_tma_2d_descriptor_2B(
    CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, 
    uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, 
    CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
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

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0);
    
    __nv_bfloat16* A_ptr = static_cast<__nv_bfloat16*>(A.data_ptr());
    __nv_bfloat16* B_ptr = static_cast<__nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* C_ptr = static_cast<__nv_bfloat16*>(C.data_ptr());
    
    CUtensorMap tma_A, tma_B;
    
    CUresult resA = create_tma_2d_descriptor_2B(
        &tma_A, A_ptr, K, M, 64, 128,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (resA != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to create TMA A: %d\n", resA); exit(1);
    }
    
    CUresult resB = create_tma_2d_descriptor_2B(
        &tma_B, B_ptr, K, N, 64, 128,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (resB != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to create TMA B: %d\n", resB); exit(1);
    }
    
    int clusters_x = (N + 255) / 256;
    int clusters_y = (M + 255) / 256;
    dim3 grid(clusters_x * 2, clusters_y, 1);
    dim3 block(256, 1, 1); 
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_kernel, tma_A, tma_B, C_ptr, (uint32_t)M, (uint32_t)N, (uint32_t)K));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda