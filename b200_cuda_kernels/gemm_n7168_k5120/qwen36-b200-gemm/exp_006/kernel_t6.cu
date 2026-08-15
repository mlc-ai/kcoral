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

// ================================================================ Host helper
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

// ================================================================ Device helpers

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

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t parity) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=: \n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(parity));
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
           "l"((uint64_t)d),
           "r"((uint32_t)__cvta_generic_to_shared(bar)),
           "r"(c0), "r"(c1) : "memory");
}

// ================================================================ Kernel constants
constexpr uint32_t BM_TILE = 128;
constexpr uint32_t BN_TILE = 128;
constexpr uint32_t BK_TILE = 64;
constexpr uint32_t N_CONST = 7168;
constexpr uint32_t K_CONST = 5120;
constexpr uint32_t NUM_K_STEPS = K_CONST / BK_TILE;  // 80
constexpr uint32_t NUM_STAGES = 2;

constexpr uint32_t A_BYTES = BM_TILE * BK_TILE * sizeof(__nv_bfloat16);   // 16384
constexpr uint32_t B_BYTES = BN_TILE * BK_TILE * sizeof(__nv_bfloat16);   // 16384

extern __shared__ uint8_t smem_dynamic[];

__global__ void gemm_kernel_sm100(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* __restrict__ C_out,
    uint32_t M)
{
    uint32_t bx = blockIdx.x;
    uint32_t by = blockIdx.y;
    
    uint32_t m_start = by * BM_TILE;
    uint32_t n_start = bx * BN_TILE;
    
    if (m_start >= M || n_start >= N_CONST) return;
    
    // Shared memory layout
    __nv_bfloat16* sA[NUM_STAGES];
    __nv_bfloat16* sB[NUM_STAGES];
    uint64_t* pbar[NUM_STAGES];
    
    uint8_t* base = smem_dynamic;
    uint32_t off = 0;
    for (int i = 0; i < NUM_STAGES; ++i) { sA[i] = reinterpret_cast<__nv_bfloat16*>(base+off); off += A_BYTES; }
    for (int i = 0; i < NUM_STAGES; ++i) { sB[i] = reinterpret_cast<__nv_bfloat16*>(base+off); off += B_BYTES; }
    for (int i = 0; i < NUM_STAGES; ++i) { pbar[i] = reinterpret_cast<uint64_t*>(base+off); off += 64; }
    
    // Init barriers
    constexpr uint32_t TX_BYTES = A_BYTES + B_BYTES;
    
    if (threadIdx.x == 0) {
        for (int i = 0; i < NUM_STAGES; ++i) {
            init_smem_barrier_fn(pbar[i], TX_BYTES);
        }
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    
    uint32_t tid = threadIdx.x;
    uint32_t prod_par = 0;
    
    // Start first TMA load
    if (tid == 0) {
        tma_load_2d_fn(&tma_A, pbar[0], sA[0], 0, static_cast<int32_t>(m_start));
        tma_load_2d_fn(&tma_B, pbar[0], sB[0], 0, static_cast<int32_t>(n_start));
        mbarrier_arrive_and_expect_tx_fn(pbar[0], TX_BYTES);
        prod_par ^= 1;
    }
    
    // Wait for first load
    if (tid < 128) mbarrier_wait_fn(pbar[0], prod_par ^ 1);
    __syncthreads();
    
    // Each thread owns one row of output tile
    uint32_t my_row = tid;  // 0..127
    
    // Process each K step
    for (uint32_t ks = 0; ks < NUM_K_STEPS; ++ks) {
        int stg = ks & 1;
        int next_stg = ((ks + 1) & 1);
        
        // Launch next TMA load (for next K step)
        if (ks + 1 < NUM_K_STEPS && tid == 0) {
            int32_t kc = static_cast<int32_t>((ks+1)*BK_TILE);
            tma_load_2d_fn(&tma_A, pbar[next_stg], sA[next_stg], kc, static_cast<int32_t>(m_start));
            tma_load_2d_fn(&tma_B, pbar[next_stg], sB[next_stg], kc, static_cast<int32_t>(n_start));
            mbarrier_arrive_and_expect_tx_fn(pbar[next_stg], TX_BYTES);
            prod_par ^= 1;
        }
        
        __syncthreads();
        
        // sA[stg]: [BM x BK] row-major => sA[r*k + k_idx]
        // sB[stg]: [BN x BK] row-major => sB[c*BK + k_idx]
        // Output: C[r,c] += sum_k sA[r,k] * sB[c,k]
        
        if (my_row < BM_TILE) {
            uint32_t out_row = m_start + my_row;
            
            // Read A row once into registers (vectorized: 16 uint2 values for BK=64)
            const __nv_bfloat16* a_row = &sA[stg][my_row * BK_TILE];
            
            // Process BN columns in groups of 4
            for (uint32_t nc = 0; nc < BN_TILE; nc += 4) {
                float sums[4] = {0.f, 0.f, 0.f, 0.f};
                
                // Dot product over K dimension
                for (uint32_t kk = 0; kk < BK_TILE; ++kk) {
                    float a_val = __bfloat162float(a_row[kk]);
                    
                    for (uint32_t ci = 0; ci < 4; ++ci) {
                        float b_val = __bfloat162float(sB[stg][(nc + ci) * BK_TILE + kk]);
                        sums[ci] += a_val * b_val;
                    }
                }
                
                // Write partial sums to global memory
                for (uint32_t ci = 0; ci < 4; ++ci) {
                    uint32_t out_col = n_start + nc + ci;
                    if (out_row < M && out_col < N_CONST) {
                        int64_t idx = static_cast<int64_t>(out_row) * N_CONST + out_col;
                        float existing = __bfloat162float(C_out[idx]);
                        C_out[idx] = __float2bfloat16(existing + sums[ci]);
                    }
                }
            }
        }
        
        // Wait for next stage before consuming it
        if (ks + 1 < NUM_K_STEPS) {
            if (tid < 128) mbarrier_wait_fn(pbar[next_stg], prod_par ^ 1);
            __syncthreads();
        }
    }
}

// ================================================================ Host
void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    
    int64_t M_dyn = A.size(0);
    __nv_bfloat16* A_ptr = static_cast<__nv_bfloat16*>(A.data_ptr());
    __nv_bfloat16* B_ptr = static_cast<__nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* C_ptr = static_cast<__nv_bfloat16*>(C.data_ptr());
    
    CUtensorMap tma_A, tma_B;
    
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_A, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        A_ptr, K_CONST, static_cast<uint64_t>(M_dyn),
        BK_TILE, BM_TILE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_B, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        B_ptr, K_CONST, N_CONST,
        BK_TILE, BN_TILE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
    uint32_t gx = (N_CONST + BN_TILE - 1) / BN_TILE;
    uint32_t gy = (static_cast<uint32_t>(M_dyn) + BM_TILE - 1) / BM_TILE;
    
    dim3 grid(gx, gy);
    dim3 block(128, 1, 1);
    uint32_t smem = A_BYTES * NUM_STAGES + B_BYTES * NUM_STAGES + 64 * NUM_STAGES;
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = 1;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    
    cudaLaunchKernelEx(&cfg, gemm_kernel_sm100,
        tma_A, tma_B, C_ptr, static_cast<uint32_t>(M_dyn));
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_blackwell::run);

}  // namespace gemm_blackwell