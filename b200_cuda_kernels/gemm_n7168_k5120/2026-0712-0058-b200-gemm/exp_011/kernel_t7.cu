#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
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
                _e, __FILE__, __LINE__);                           \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_gemms {

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, globalAddress, globalDim, globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill);
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
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

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    d |= (uint64_t)((addr >> 7) & 0x7) << 49;
    return d;
}

__device__ __forceinline__ uint64_t advance_desc(uint64_t desc, uint32_t byte_offset) {
    uint32_t addr_bits = desc & 0x3FFF;
    uint32_t new_addr = (addr_bits << 4) + byte_offset;
    uint64_t new_desc = (desc & ~(0x3FFFull)) | ((new_addr >> 4) & 0x3FFF);
    return new_desc;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn_cg1(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);     
    d |= (1u << 7);     
    d |= (1u << 10);    
    d |= (0u << 15);    
    d |= (1u << 16);    
    d |= ((N >> 3) << 17);     
    d |= ((M >> 4) << 24);    
    return d;
}

__device__ __forceinline__ void umma_cg1_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void commit_mma_1sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1"
        ".mbarrier::arrive::one.shared::cta.b64"
        " [%0];"
        :: "r"(a));
}


__global__ void gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* C, int M, int N, int K) 
{
    extern __shared__ __align__(128) uint8_t smem[];
    
    // Ensure 1024-byte alignment for all SMEM buffers enabling correct hardware swizzling offsets resolution
    __nv_bfloat16* smem_A0 = (__nv_bfloat16*)(((uintptr_t)smem + 1023) & ~1023);
    __nv_bfloat16* smem_A1 = (__nv_bfloat16*)(smem_A0 + 4096); // 64 * 64 bf16 = 8192 bytes
    __nv_bfloat16* smem_B0 = (__nv_bfloat16*)(smem_A1 + 4096);
    __nv_bfloat16* smem_B1 = (__nv_bfloat16*)(smem_B0 + 4096);
    
    uint64_t* bar_A = (uint64_t*)(smem_B1 + 4096);
    uint64_t* bar_B = (uint64_t*)(bar_A + 1);
    uint64_t* bar_mma = (uint64_t*)(bar_B + 1);
    uint32_t* p_tmem_c = (uint32_t*)(bar_mma + 1);

    int m_block = blockIdx.x * 64;
    int n_block = blockIdx.y * 64;
    int tid = threadIdx.x;

    if (m_block >= M) return;

    if (tid == 0) {
        init_smem_barrier_fn(bar_A, 1);
        init_smem_barrier_fn(bar_B, 1);
        init_smem_barrier_fn(bar_mma, 1);
        
        tmem_alloc_fn(p_tmem_c, 64);
    }
    __syncthreads();

    uint32_t tmem_c = *p_tmem_c;

    // Prime the pump with initial TMA requests
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_A, 8192);
        uint32_t swizzled_byte_offset_A = ((0 * 64 * 2) ^ (0 * 2)) & 127;
        tma_load_2d_fn(&tma_A, bar_A, (void*)(smem_A0 + (swizzled_byte_offset_A / 2)), 0, m_block);
        
        mbarrier_arrive_and_expect_tx_fn(bar_B, 8192);
        uint32_t swizzled_byte_offset_B = ((0 * 64 * 2) ^ (0 * 2)) & 127;
        tma_load_2d_fn(&tma_B, bar_B, (void*)(smem_B0 + (swizzled_byte_offset_B / 2)), 0, n_block);
    }

    int phase_A = 0, phase_B = 0, phase_mma = 0;
    int curr = 0, next = 1;
    uint32_t idesc = make_instr_desc_fn_cg1(64, 64); // Hardcoded local instruction descriptor spanning exactly 64x64 multiply

    for (int k_block = 0; k_block < K; k_block += 64) {
        // Software pipeling issuance - queue the future chunk whilst processing current chunk
        if (k_block + 64 <= K) {
            if (tid == 0) {
                mbarrier_arrive_and_expect_tx_fn(bar_A, 8192);
                uint32_t swizzled_byte_offset_A_next = (((next == 0 ? smem_A0 : smem_A1) - smem_A0) * 64 * 2) ^ ((k_block + 64) * 2)) & 127;
                tma_load_2d_fn(&tma_A, bar_A, (void*)((next == 0 ? smem_A0 : smem_A1) + (swizzled_byte_offset_A_next / 2)), k_block + 64, m_block);

                mbarrier_arrive_and_expect_tx_fn(bar_B, 8192);
                uint32_t swizzled_byte_offset_B_next = (((next == 0 ? smem_B0 : smem_B1) - smem_B0) * 64 * 2) ^ ((k_block + 64) * 2)) & 127;
                tma_load_2d_fn(&tma_B, bar_B, (void*)((next == 0 ? smem_B0 : smem_B1) + (swizzled_byte_offset_B_next / 2)), k_block + 64, n_block);
            }
        }

        mbarrier_wait_fn(bar_A, phase_A);
        mbarrier_wait_fn(bar_B, phase_B);
        phase_A ^= 1;
        phase_B ^= 1;

        fence_proxy_async_fn();
        __syncthreads();

        if (tid == 0) {
            // Generate dynamically aligned descriptors taking into account physical SWIZZLE_128B offsets within each double-buffered chunk
            uint64_t desc_A = make_smem_desc_sm100_fn((curr == 0 ? smem_A0 : smem_A1), 1024, 1024);
            uint64_t desc_B = make_smem_desc_sm100_fn((curr == 0 ? smem_B0 : smem_B1), 8192, 1024);

            for (int step = 0; step < 4; ++step) {
                uint32_t p = (k_block == 0 && step == 0 && curr == 0) ? 0 : 1;
                uint64_t da = advance_desc(desc_A, step * 32); // 32 bytes shifts exactly 16 elements inside contiguous K dimension block
                uint64_t db = advance_desc(desc_B, step * 512); // 512 bytes shifts exactly 16 elements inside strided N dimension block (accounting for underlying 128B chunking pattern layout mapping)
                umma_cg1_fn(tmem_c, da, db, idesc, p);
            }
            commit_mma_1sm_fn(bar_mma);
        }
        mbarrier_wait_fn(bar_mma, phase_mma);
        phase_mma ^= 1;

        curr ^= 1;
        next ^= 1;
        __syncthreads();
    }
    
    // Coalesced Direct Epilogue writing mapped identically to physical TMEM swizzle ordering constraints rules to Global Mem using native `tcgen05`
    __syncthreads();
    for (int col = 0; col < 64; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_c + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        int row_base = (tid / 32) * 32;
        int tmem_lane = row_base + (tid % 32);
        int global_row = m_block + tmem_lane;
        int global_col = n_block + col;
        
        if (global_row < M && global_col + 3 < N) {
            C[(uint64_t)global_row * N + global_col + 0] = __float2bfloat16(__uint_as_float(r0));
            C[(uint64_t)global_row * N + global_col + 1] = __float2bfloat16(__uint_as_float(r1));
            C[(uint64_t)global_row * N + global_col + 2] = __float2bfloat16(__uint_as_float(r2));
            C[(uint64_t)global_row * N + global_col + 3] = __float2bfloat16(__uint_as_float(r3));
        }
    }

    if (tid == 0) {
        tmem_dealloc_fn(tmem_c, 64);
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
  CUDA_CHECK(cudaSetDevice(A.device().device_id));

  int64_t M = A.size(0);
  const int64_t N = 7168;
  const int64_t K = 5120;

  CUtensorMap tma_A, tma_B;
  
  CU_CHECK(create_tma_2d_descriptor_2B(&tma_A, A.data_ptr(), K, M, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  CU_CHECK(create_tma_2d_descriptor_2B(&tma_B, B.data_ptr(), K, N, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

  int smem_size = 32896;
  cudaFuncSetAttribute(gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  uint32_t grid_x = (M + 63) / 64;
  if (grid_x % 2 != 0) grid_x++;
  uint32_t grid_y = N / 64;
  
  dim3 grid(grid_x, grid_y, 1); 
  dim3 block(128); 
  cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
  
  cudaLaunchConfig_t config = {};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = smem_size;
  config.stream = stream;
  
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = 2;
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;
  config.attrs = attrs;
  config.numAttrs = 1;
  
  CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_kernel, tma_A, tma_B, static_cast<__nv_bfloat16*>(C.data_ptr()), M, N, K));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_gemms::run);

}  // namespace tvm_ffi_gemms