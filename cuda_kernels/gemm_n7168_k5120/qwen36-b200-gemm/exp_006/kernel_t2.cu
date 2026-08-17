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
    CUresult _r = (call);                                          \
    if (_r != CUDA_SUCCESS) {                                      \
        const char *_err_str;                                      \
        cuGetErrorString(_r, &_err_str);                           \
        fprintf(stderr, "CU error %s at %s:%d\n",                 \
                _err_str, __FILE__, __LINE__);                     \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace gemm_blackwell {

// ==================== Helper Functions ====================

__device__ __forceinline__ uint32_t get_cluster_ctarank() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_dec_sync_fn() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=: \n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(phase));
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(bar)),
        "r"(c0), "r"(c1) : "memory");
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

__device__ __forceinline__ void umma_commit_2sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
        :: "r"(a), "h"((uint16_t)0x3));
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo, uint32_t swizzle_mode) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)swizzle_mode << 61;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);      // c_format = FP32
    d |= (1u << 7);      // a_format = BF16
    d |= (1u << 10);     // b_format = BF16
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void tmem_wait_ld_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

// TMA descriptor creation
CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, CUtensorMapDataType dataType,
    void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim,
    uint32_t smem_inner_dim, uint32_t smem_outer_dim,
    CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion,
    CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, dataType, 2, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill);
}

// ==================== Constants ====================
constexpr uint32_t BM = 128;         // M-dimension per CTA
constexpr uint32_t BN = 128;         // N-dimension per CTA  
constexpr uint32_t BK = 16;          // K-dimension per iteration
constexpr uint32_t BM_PAIR = BM * 2; // 256 combined M
constexpr uint32_t BN_PAIR = BN * 2; // 256 combined N
constexpr uint32_t BN_HALF = BN / 2; // 64
constexpr uint32_t NUM_STAGES = 2;   
constexpr uint32_t N_CONST = 7168;
constexpr uint32_t K_CONST = 5120;
constexpr uint32_t NUM_K_ITERS = K_CONST / BK; // 320

constexpr uint32_t A_TILE_BYTES = BM * BK * sizeof(__nv_bfloat16);     
constexpr uint32_t B_HALF_BYTES = BN_HALF * BK * sizeof(__nv_bfloat16);
constexpr uint32_t NUM_BARRIERS = 8;

constexpr uint32_t SHARED_MEM_PER_CTA = 
    A_TILE_BYTES * NUM_STAGES +         
    B_HALF_BYTES * NUM_STAGES * 2 +       
    64 * NUM_BARRIERS +                
    64;                               

// ==================== Kernel ====================

extern __shared__ uint8_t smem_dynamic[];

__global__ void gemm_kernel_sm100(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* __restrict__ C_out,
    uint32_t M,
    uint32_t grid_cols)
{
    uint32_t cluster_col = blockIdx.x;  
    uint32_t cta_in_cluster = get_cluster_ctarank();
    uint32_t n_start = cluster_col * BN_PAIR;  
    
    uint32_t m_start = blockIdx.y * BM;  
    
    if (m_start >= M || n_start >= N_CONST) return;
    
    // ---- Setup Shared Memory Pointers ----
    __nv_bfloat16* A_tiles[NUM_STAGES];
    __nv_bfloat16* B_tiles[2][NUM_STAGES];  
    uint64_t* prod_barriers[NUM_STAGES];
    uint64_t* cons_barriers[NUM_STAGES];
    
    uint8_t* smem_base = smem_dynamic;
    uint32_t smem_off = 0;
    
    for (int s = 0; s < NUM_STAGES; ++s) {
        A_tiles[s] = reinterpret_cast<__nv_bfloat16*>(smem_base + smem_off);
        smem_off += A_TILE_BYTES;
    }
    
    for (int s = 0; s < NUM_STAGES; ++s) {
        B_tiles[0][s] = reinterpret_cast<__nv_bfloat16*>(smem_base + smem_off);
        smem_off += B_HALF_BYTES;
    }
    for (int s = 0; s < NUM_STAGES; ++s) {
        B_tiles[1][s] = reinterpret_cast<__nv_bfloat16*>(smem_base + smem_off);
        smem_off += B_HALF_BYTES;
    }
    
    for (int s = 0; s < NUM_STAGES; ++s) {
        prod_barriers[s] = reinterpret_cast<uint64_t*>(smem_base + smem_off);
        smem_off += 64;
    }
    for (int s = 0; s < NUM_STAGES; ++s) {
        cons_barriers[s] = reinterpret_cast<uint64_t*>(smem_base + smem_off);
        smem_off += 64;
    }
    
    uint32_t* tmem_addrs = reinterpret_cast<uint32_t*>(smem_base + smem_off);
    
    // ---- Register Management ----
    if (threadIdx.x < 128) {
        setmaxnreg_inc_sync_fn<128>();
    }
    
    // ---- Initialize Barriers ----
    constexpr uint32_t TX_BYTES_PER_STAGE = A_TILE_BYTES + B_HALF_BYTES;
    if (threadIdx.x == 0) {
        for (int s = 0; s < NUM_STAGES; ++s) {
            init_smem_barrier_fn(prod_barriers[s], TX_BYTES_PER_STAGE);
        }
        for (int s = 0; s < NUM_STAGES; ++s) {
            init_smem_barrier_fn(cons_barriers[s], 1);
        }
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    
    // ---- Allocate TMEM ----
    if (threadIdx.x < 128) {
        tmem_alloc_fn(tmem_addrs, 2560);
    }
    __syncthreads();
    
    uint32_t a_tmem = tmem_addrs[0];
    uint32_t b_tmem = tmem_addrs[1];
    uint32_t c_tmem = tmem_addrs[2];
    
    uint32_t idesc = make_instr_desc_fn(BM_PAIR, BN_PAIR);
    
    // ---- Main Computation Loop ----
    uint32_t prod_parity = 0;
    
    // First iteration (k_iter=0): load and compute without double-buffering wait
    if (threadIdx.x == 0) {
        int stage = 0;
        int32_t k_coord = 0;
        int32_t m_coord = static_cast<int32_t>(m_start);
        
        tma_load_2d_fn(&tma_A, prod_barriers[stage], A_tiles[stage], k_coord, m_coord);
        
        int32_t n_coord = static_cast<int32_t>(n_start + cta_in_cluster * BN_HALF);
        tma_load_2d_fn(&tma_B, prod_barriers[stage], B_tiles[cta_in_cluster][stage], k_coord, n_coord);
        
        mbarrier_arrive_and_expect_tx_fn(prod_barriers[stage], TX_BYTES_PER_STAGE);
        prod_parity ^= 1;
    }
    
    if (threadIdx.x < 128) {
        mbarrier_wait_fn(prod_barriers[0], prod_parity ^ 1);
    }
    __syncthreads();
    
    // Compute first tile
    if (threadIdx.x < 128) {
        uint64_t desc_a = make_smem_desc_sm100_fn(A_tiles[0], 1, 1024, 2);
        uint64_t desc_b = make_smem_desc_sm100_fn(B_tiles[cta_in_cluster][0], 1, 1024, 2);
        
        asm volatile(
            "tcgen05.cp.cta_group::2.128x256b [%0], %1;"
            :: "r"(a_tmem), "l"(desc_a) : "memory");
        
        asm volatile(
            "tcgen05.cp.cta_group::2.128x256b [%0], %1;"
            :: "r"(b_tmem), "l"(desc_b) : "memory");
        
        fence_proxy_async_fn();
        
        umma_f16_cg2_fn(c_tmem, a_tmem, b_tmem, idesc, 0);
        umma_commit_2sm_fn(cons_barriers[0]);
    }
    
    // Remaining iterations with double buffering
    for (int k_iter = 1; k_iter < NUM_K_ITERS; ++k_iter) {
        int stage = k_iter & 1;
        int prev_stage = (k_iter - 1) & 1;
        
        // Producer: Load next tile
        if (threadIdx.x == 0) {
            int32_t k_coord = k_iter * BK;
            int32_t m_coord = static_cast<int32_t>(m_start);
            
            tma_load_2d_fn(&tma_A, prod_barriers[stage], A_tiles[stage], k_coord, m_coord);
            
            int32_t n_coord = static_cast<int32_t>(n_start + cta_in_cluster * BN_HALF);
            tma_load_2d_fn(&tma_B, prod_barriers[stage], B_tiles[cta_in_cluster][stage], k_coord, n_coord);
            
            mbarrier_arrive_and_expect_tx_fn(prod_barriers[stage], TX_BYTES_PER_STAGE);
            prod_parity ^= 1;
        }
        
        // Wait for consumer of previous stage to complete
        if (threadIdx.x < 128) {
            mbarrier_wait_fn(cons_barriers[prev_stage], 0);
        }
        
        // Wait for production of current stage
        if (threadIdx.x < 128) {
            mbarrier_wait_fn(prod_barriers[stage], prod_parity ^ 1);
        }
        __syncthreads();
        
        // Consumer: Compute with current stage data
        if (threadIdx.x < 128) {
            uint64_t desc_a = make_smem_desc_sm100_fn(A_tiles[stage], 1, 1024, 2);
            uint64_t desc_b = make_smem_desc_sm100_fn(B_tiles[cta_in_cluster][stage], 1, 1024, 2);
            
            asm volatile(
                "tcgen05.cp.cta_group::2.128x256b [%0], %1;"
                :: "r"(a_tmem), "l"(desc_a) : "memory");
            
            asm volatile(
                "tcgen05.cp.cta_group::2.128x256b [%0], %1;"
                :: "r"(b_tmem), "l"(desc_b) : "memory");
            
            fence_proxy_async_fn();
            
            umma_f16_cg2_fn(c_tmem, a_tmem, b_tmem, idesc, 1);
            umma_commit_2sm_fn(cons_barriers[stage]);
        }
    }
    
    // Finalize: wait for last computation to finish
    int last_stage = (NUM_K_ITERS - 1) & 1;
    if (threadIdx.x < 128) {
        mbarrier_wait_fn(cons_barriers[last_stage], 0);
    }
    __syncthreads();
    
    // Release registers
    if (threadIdx.x < 128) {
        setmaxnreg_dec_sync_fn<32>();
    }
    
    // Deallocate TMEM
    if (threadIdx.x < 128) {
        tmem_dealloc_fn(c_tmem, 2560);
    }
    
    // ---- Epilogue: Store Result to Global ----
    uint32_t tmem_lane = threadIdx.x;
    if (cta_in_cluster == 1) {
        tmem_lane += 128;
    }
    
    uint32_t global_row = m_start + threadIdx.x;
    if (global_row >= M) return;
    
    int64_t base_ptr = static_cast<int64_t>(global_row) * N_CONST;
    
    for (uint32_t col = 0; col < BN_PAIR; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(col));
        
        tmem_wait_ld_fn();
        
        uint32_t nc = n_start + col;
        __nv_bfloat16* out = C_out + base_ptr + nc;
        
        if (nc < N_CONST) *out = __float2bfloat16(__uint_as_float(r0));
        if (nc + 1 < N_CONST) out[1] = __float2bfloat16(__uint_as_float(r1));
        if (nc + 2 < N_CONST) out[2] = __float2bfloat16(__uint_as_float(r2));
        if (nc + 3 < N_CONST) out[3] = __float2bfloat16(__uint_as_float(r3));
    }
}

// ==================== Host Function ====================

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    
    int64_t M = A.size(0);
    
    __nv_bfloat16* A_ptr = static_cast<__nv_bfloat16*>(A.data_ptr());
    __nv_bfloat16* B_ptr = static_cast<__nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* C_ptr = static_cast<__nv_bfloat16*>(C.data_ptr());
    
    CUtensorMap tma_A, tma_B;
    
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_A, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        A_ptr, K_CONST, static_cast<uint64_t>(M),
        BK, BM,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_B, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        B_ptr, K_CONST, N_CONST,
        BK, BN_HALF,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
    uint32_t grid_x = (N_CONST + BN_PAIR - 1) / BN_PAIR;
    uint32_t grid_y = (static_cast<uint32_t>(M) + BM - 1) / BM;
    
    dim3 grid(grid_x, grid_y);
    dim3 block(128, 1, 1);
    uint32_t smem_bytes = SHARED_MEM_PER_CTA;
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_bytes;
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    cudaLaunchKernelEx(&config, gemm_kernel_sm100,
        tma_A, tma_B, C_ptr, static_cast<uint32_t>(M), grid_x);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_blackwell::run);

}  // namespace gemm_blackwell