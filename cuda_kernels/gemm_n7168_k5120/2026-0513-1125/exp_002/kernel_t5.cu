#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
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
        const char* errStr;                                        \
        cuGetErrorString(_e, &errStr);                             \
        fprintf(stderr, "CU error %d (%s) at %s:%d\n",             \
                _e, errStr, __FILE__, __LINE__);                   \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_example_cuda {

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

__device__ __forceinline__ void mbarrier_arrive_expect_tx_cluster_fn(uint64_t* bar, uint32_t tx, uint32_t target_cta) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    uint32_t remote_a;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(remote_a) : "r"(a), "r"(target_cta));
    asm volatile("mbarrier.arrive.expect_tx.shared::cluster.b64 _, [%0], %1;"
                 :: "r"(remote_a), "r"(tx));
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

__device__ __forceinline__ void tma_load_2d_cg2_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t ba = (uint32_t)__cvta_generic_to_shared(&bar[0]) & 0xFEFFFFFF; 
    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(sa), "l"((uint64_t)d), "r"(c0), "r"(c1), "r"(ba) : "memory");
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

__device__ __forceinline__ void umma_f16_cg2_fn(uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
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

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ uint32_t pack_bf16_fn(uint32_t fp32_a, uint32_t fp32_b) {
    uint32_t result;
    asm("{\n\t"
        ".reg .b16 ha, hb;\n\t"
        "cvt.rn.bf16.f32 ha, %1;\n\t"
        "cvt.rn.bf16.f32 hb, %2;\n\t"
        "mov.b32 %0, {ha, hb};\n\t"
        "}"
        : "=r"(result) : "f"(__uint_as_float(fp32_a)), "f"(__uint_as_float(fp32_b)));
    return result;
}

__device__ __forceinline__ void prefetch_tma_descriptor_fn(const CUtensorMap* d) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(d) : "memory");
}

__device__ __forceinline__ void tmem_epilogue_coalesced_8w_fn(
    uint32_t tmem_c, __nv_bfloat16* D, void* smem_out,
    uint32_t M, uint32_t N, uint32_t global_row_start, uint32_t global_col_start,
    uint32_t num_rows, uint32_t num_cols) {
    
    // Use u32_stride = 132 for perfect uniform 32-way bank access on st.shared.v4
    uint32_t u32_stride = 132; 
    uint32_t* smem_out_u32 = reinterpret_cast<uint32_t*>(smem_out);
    
    // Phase 1: TMEM -> SMEM (Warpgroup 0: 128 threads load all 128 rows)
    if (threadIdx.x < 128) {
        for (uint32_t col = 0; col < num_cols; col += 16) {
            uint32_t r[16];
            uint32_t addr = tmem_c + col;
            
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
                         : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
                           "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15])
                         : "r"(addr));
                         
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            uint32_t base = threadIdx.x * u32_stride + (col / 2);
            
            uint32_t p0 = pack_bf16_fn(r[0], r[1]);
            uint32_t p1 = pack_bf16_fn(r[2], r[3]);
            uint32_t p2 = pack_bf16_fn(r[4], r[5]);
            uint32_t p3 = pack_bf16_fn(r[6], r[7]);
            uint32_t p4 = pack_bf16_fn(r[8], r[9]);
            uint32_t p5 = pack_bf16_fn(r[10], r[11]);
            uint32_t p6 = pack_bf16_fn(r[12], r[13]);
            uint32_t p7 = pack_bf16_fn(r[14], r[15]);
            
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                         :: "r"((uint32_t)__cvta_generic_to_shared(&smem_out_u32[base])),
                            "r"(p0), "r"(p1), "r"(p2), "r"(p3) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                         :: "r"((uint32_t)__cvta_generic_to_shared(&smem_out_u32[base + 4])),
                            "r"(p4), "r"(p5), "r"(p6), "r"(p7) : "memory");
        }
    }
    __syncthreads();
    
    // Phase 2: SMEM -> Global (256 threads = 8 warps)
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t num_steps = (num_rows + 7) / 8;
    
    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 8 + warp_id;
        if (row >= num_rows) continue;
        uint32_t global_row = global_row_start + row;
        
        for (uint32_t col_chunk = 0; col_chunk < num_cols; col_chunk += 128) {
            uint32_t col_start = col_chunk + lane_id * 4;
            uint32_t global_col = global_col_start + col_start;
            if (global_row < M) {
                if (global_col + 3 < N) {
                    uint2 data = *reinterpret_cast<uint2*>(&smem_out_u32[row * u32_stride + (col_start / 2)]);
                    *reinterpret_cast<uint2*>(D + (uint64_t)global_row * N + global_col) = data;
                } else {
                    for (int i = 0; i < 4; ++i) {
                        if (global_col + i < N) {
                            D[(uint64_t)global_row * N + global_col + i] = 
                                reinterpret_cast<__nv_bfloat16*>(smem_out_u32)[row * (u32_stride * 2) + col_start + i];
                        }
                    }
                }
            }
        }
    }
}

constexpr int STAGES = 3;
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 128;
constexpr int CHUNKS = BK / 64;

struct SharedStorage {
    alignas(1024) __nv_bfloat16 A[STAGES][CHUNKS][BM][64];
    alignas(1024) __nv_bfloat16 B[STAGES][CHUNKS][BN][64];
    alignas(16) uint64_t full_mbar[STAGES];
    alignas(16) uint64_t empty_mbar[STAGES];
    alignas(16) uint32_t tmem_c;
};

__global__ __launch_bounds__(256, 1) void gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* C,
    int M, int N, int K) 
{
    setmaxnreg_inc_sync_fn<240>();
    
    extern __shared__ uint8_t dynamic_smem[];
    uintptr_t smem_addr = reinterpret_cast<uintptr_t>(dynamic_smem);
    smem_addr = (smem_addr + 1023) & ~1023; // Guarantee 1024-byte alignment
    SharedStorage& shared = *reinterpret_cast<SharedStorage*>(smem_addr);

    if (threadIdx.x == 0) {
        prefetch_tma_descriptor_fn(&tma_A);
        prefetch_tma_descriptor_fn(&tma_B);
    }

    if (threadIdx.x < 32) {
        tmem_alloc_fn(&shared.tmem_c, 256);
    }
    __syncthreads();
    uint32_t tmem_c = shared.tmem_c;

    uint32_t cluster_x = blockIdx.x / 2;
    uint32_t cluster_y = blockIdx.y;
    uint32_t m_start = cluster_y * 256;
    uint32_t n_start = cluster_x * 256;

    uint32_t ctaid = cluster_rank_fn();
    uint32_t m_idx = m_start + (ctaid == 1 ? 128 : 0);
    uint32_t n_idx = n_start + (ctaid == 1 ? 128 : 0);

    if (ctaid == 0 && threadIdx.x == 0) {
        for (int i = 0; i < STAGES; ++i) {
            init_smem_barrier_fn(&shared.full_mbar[i], 2);
        }
    }
    if (threadIdx.x == 0) {
        for (int i = 0; i < STAGES; ++i) {
            init_smem_barrier_fn(&shared.empty_mbar[i], 1);
        }
        fence_smem_barrier_init_fn();
    }
    cluster_sync_fn();

    int num_iters = (K + BK - 1) / BK;

    // Decoupled execution: Thread 0 handles TMA loads, Thread 128 handles UMMA issues
    if (threadIdx.x == 0) {
        for (int iter = 0; iter < num_iters; ++iter) {
            int stage = iter % STAGES;
            int prev_use = iter - STAGES;
            if (prev_use >= 0) {
                int empty_phase = (prev_use / STAGES) % 2;
                mbarrier_wait_fn(&shared.empty_mbar[stage], empty_phase);
            }
            
            if (ctaid == 0) {
                mbarrier_arrive_and_expect_tx_fn(&shared.full_mbar[stage], 65536);
            } else {
                mbarrier_arrive_expect_tx_cluster_fn(&shared.full_mbar[stage], 65536, 0);
            }
            
            int k_idx = iter * BK;
            tma_load_2d_cg2_fn(&tma_A, &shared.full_mbar[stage], &shared.A[stage][0][0][0], k_idx, m_idx);
            tma_load_2d_cg2_fn(&tma_A, &shared.full_mbar[stage], &shared.A[stage][1][0][0], k_idx + 64, m_idx);
            tma_load_2d_cg2_fn(&tma_B, &shared.full_mbar[stage], &shared.B[stage][0][0][0], k_idx, n_idx);
            tma_load_2d_cg2_fn(&tma_B, &shared.full_mbar[stage], &shared.B[stage][1][0][0], k_idx + 64, n_idx);
        }
    } else if (threadIdx.x == 128 && ctaid == 0) {
        uint32_t idesc = make_instr_desc_fn(256, 256);
        for (int iter = 0; iter < num_iters; ++iter) {
            int stage = iter % STAGES;
            int full_phase = (iter / STAGES) % 2;
            mbarrier_wait_fn(&shared.full_mbar[stage], full_phase);
            
            uint64_t base_desc_a0 = make_smem_desc_sm100_fn(&shared.A[stage][0][0][0], 1024);
            uint64_t base_desc_b0 = make_smem_desc_sm100_fn(&shared.B[stage][0][0][0], 1024);
            uint64_t base_desc_a1 = make_smem_desc_sm100_fn(&shared.A[stage][1][0][0], 1024);
            uint64_t base_desc_b1 = make_smem_desc_sm100_fn(&shared.B[stage][1][0][0], 1024);
            
            #pragma unroll
            for (int step = 0; step < 8; ++step) {
                int chunk = step / 4;
                int sub_step = step % 4;
                uint64_t desc_a = (chunk == 0 ? base_desc_a0 : base_desc_a1) + (sub_step * 2);
                uint64_t desc_b = (chunk == 0 ? base_desc_b0 : base_desc_b1) + (sub_step * 2);
                uint32_t accum = (iter == 0 && step == 0) ? 0 : 1;
                umma_f16_cg2_fn(tmem_c, desc_a, desc_b, idesc, accum);
            }
            
            umma_commit_2sm_fn(&shared.empty_mbar[stage]);
        }
    }

    // Thread 0 waits for the final UMMA to complete before letting epilogue start
    if (threadIdx.x == 0 && num_iters > 0) {
        int last_iter = num_iters - 1;
        int last_stage = last_iter % STAGES;
        int last_phase = (last_iter / STAGES) % 2;
        mbarrier_wait_fn(&shared.empty_mbar[last_stage], last_phase);
    }
    
    __syncthreads();
    tcgen05_fence_after_fn();

    tmem_epilogue_coalesced_8w_fn(tmem_c, C, &shared, M, N, m_idx, n_start, 128, 256);
    __syncthreads();

    if (threadIdx.x < 32) {
        tmem_dealloc_fn(tmem_c, 256);
    }
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

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0);
    
    CUtensorMap tma_A, tma_B;
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_A, A.data_ptr(), K, M, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_B, B.data_ptr(), K, N, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    int cluster_size = 2;
    int grid_x = ((N + 255) / 256) * cluster_size; 
    int grid_y = (M + 255) / 256;
    dim3 grid(grid_x, grid_y, 1);
    dim3 block(256, 1, 1);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    
    int smem_size = sizeof(SharedStorage) + 1024;
    CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = cluster_size;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    cudaLaunchKernelEx(&config, gemm_kernel, tma_A, tma_B, static_cast<__nv_bfloat16*>(C.data_ptr()), M, N, K);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

} // namespace tvm_ffi_example_cuda