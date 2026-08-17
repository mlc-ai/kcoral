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

__device__ __forceinline__ void tmem_epilogue_coalesced_4w_fn(
    uint32_t tmem_c, __nv_bfloat16* D, __nv_bfloat16* smem_out,
    uint32_t M, uint32_t N, uint32_t global_row_start, uint32_t global_col_start,
    uint32_t BM, uint32_t BN) {
    
    for (uint32_t col = 0; col < BN; col += 4) {
        uint32_t r0, r1, r2, r3;
        uint32_t addr = tmem_c + col;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(addr));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t base = threadIdx.x * BN + col;
        smem_out[base + 0] = __float2bfloat16(__uint_as_float(r0));
        smem_out[base + 1] = __float2bfloat16(__uint_as_float(r1));
        smem_out[base + 2] = __float2bfloat16(__uint_as_float(r2));
        smem_out[base + 3] = __float2bfloat16(__uint_as_float(r3));
    }
    __syncthreads();
    
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t num_steps = (BM + 3) / 4;
    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 4 + warp_id;
        if (row >= BM) continue;
        uint32_t global_row = global_row_start + row;
        
        for (uint32_t col_chunk = 0; col_chunk < BN; col_chunk += 128) {
            uint32_t col_start = col_chunk + lane_id * 4;
            uint32_t global_col = global_col_start + col_start;
            if (global_row < M && global_col + 3 < N) {
                uint2 data = *reinterpret_cast<uint2*>(&smem_out[row * BN + col_start]);
                *reinterpret_cast<uint2*>(D + (uint64_t)global_row * N + global_col) = data;
            }
        }
    }
}

constexpr int STAGES = 3;
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 64;

struct SharedStorage {
    alignas(1024) __nv_bfloat16 A[STAGES][BM][BK];
    alignas(1024) __nv_bfloat16 B[STAGES][BN][BK];
    alignas(16) uint64_t full_mbar[STAGES];
    alignas(16) uint64_t empty_mbar[STAGES];
};

__global__ void gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* C,
    int M, int N, int K) 
{
    setmaxnreg_inc_sync_fn<240>();
    
    __shared__ alignas(1024) SharedStorage shared;
    __shared__ uint32_t shared_tmem_c;

    if (threadIdx.x < 32) {
        tmem_alloc_fn(&shared_tmem_c, 256);
    }
    __syncthreads();
    uint32_t tmem_c = shared_tmem_c;

    uint32_t ctaid = cluster_rank_fn();
    uint32_t m_start = blockIdx.y * 256;
    uint32_t n_start = blockIdx.x * 256;
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

    for (int i = 0; i < STAGES - 1; ++i) {
        int k_idx = i * BK;
        if (k_idx >= K) break;
        if (threadIdx.x == 0) {
            if (ctaid == 0) {
                mbarrier_arrive_and_expect_tx_fn(&shared.full_mbar[i], 32768);
            } else {
                mbarrier_arrive_expect_tx_cluster_fn(&shared.full_mbar[i], 32768, 0);
            }
            tma_load_2d_cg2_fn(&tma_A, &shared.full_mbar[i], &shared.A[i], k_idx, m_idx);
            tma_load_2d_cg2_fn(&tma_B, &shared.full_mbar[i], &shared.B[i], k_idx, n_idx);
        }
    }

    int stage_idx = 0;
    int num_iters = (K + BK - 1) / BK;
    
    for (int iter = 0; iter < num_iters; ++iter) {
        int next_iter = iter + STAGES - 1;
        int next_stage = next_iter % STAGES;
        
        if (next_iter < num_iters) {
            int next_k = next_iter * BK;
            if (threadIdx.x == 0) {
                int prev_use = next_iter - STAGES;
                if (prev_use >= 0) {
                    int empty_phase = (prev_use / STAGES) % 2;
                    mbarrier_wait_fn(&shared.empty_mbar[next_stage], empty_phase);
                }
                
                if (ctaid == 0) {
                    mbarrier_arrive_and_expect_tx_fn(&shared.full_mbar[next_stage], 32768);
                } else {
                    mbarrier_arrive_expect_tx_cluster_fn(&shared.full_mbar[next_stage], 32768, 0);
                }
                
                tma_load_2d_cg2_fn(&tma_A, &shared.full_mbar[next_stage], &shared.A[next_stage], next_k, m_idx);
                tma_load_2d_cg2_fn(&tma_B, &shared.full_mbar[next_stage], &shared.B[next_stage], next_k, n_idx);
            }
        }
        
        int full_phase = (iter / STAGES) % 2;
        if (ctaid == 0 && threadIdx.x == 0) {
            mbarrier_wait_fn(&shared.full_mbar[stage_idx], full_phase);
            
            for (int step = 0; step < 4; ++step) {
                uint64_t desc_a = make_smem_desc_sm100_fn(&shared.A[stage_idx][0][step * 16], 1024);
                uint64_t desc_b = make_smem_desc_sm100_fn(&shared.B[stage_idx][0][step * 16], 1024);
                uint32_t idesc = make_instr_desc_fn(256, 256);
                uint32_t accum = (iter == 0 && step == 0) ? 0 : 1;
                umma_f16_cg2_fn(tmem_c, desc_a, desc_b, idesc, accum);
            }
            
            umma_commit_2sm_fn(&shared.empty_mbar[stage_idx]);
        }
        
        stage_idx = (stage_idx + 1) % STAGES;
    }

    if (threadIdx.x == 0 && num_iters > 0) {
        int last_iter = num_iters - 1;
        int last_stage = last_iter % STAGES;
        int last_phase = (last_iter / STAGES) % 2;
        mbarrier_wait_fn(&shared.empty_mbar[last_stage], last_phase);
    }
    __syncthreads();

    __nv_bfloat16* smem_out = reinterpret_cast<__nv_bfloat16*>(&shared);
    tmem_epilogue_coalesced_4w_fn(tmem_c, C, smem_out, M, N, m_idx, n_start, 128, 256);
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
    int grid_x = N / 256; 
    int grid_y = (M + 255) / 256;
    dim3 grid(grid_x, grid_y, 1);
    dim3 block(128, 1, 1);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 0;
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