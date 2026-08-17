#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
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

// -------------------------------------------------------------------------
// Hardware Helper Functions
// -------------------------------------------------------------------------

__device__ __forceinline__ bool elect_one_sync_fn() {
    uint32_t pred;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "elect.sync _|p, 0xFFFFFFFF;\n"
        "selp.b32 %0, 1, 0, p;\n"
        "}\n"
        : "=r"(pred));
    return pred != 0;
}

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
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
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
       :: "r"(a), "r"(ncols));
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

__device__ __forceinline__ uint64_t make_smem_desc_k_major(void* smem_ptr) {
    return make_smem_desc_sm100_fn(smem_ptr, 1, 1024);
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

__device__ __forceinline__ uint64_t walk_desc_byte(uintptr_t start_addr, uint64_t desc, int byte_offset) {
    uint32_t addr = ((start_addr + byte_offset) & 0x3FFFF) >> 4;
    return (desc & ~0x3FFF) | (addr & 0x3FFF);
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

// -------------------------------------------------------------------------
// GEMM Kernel
// -------------------------------------------------------------------------

struct SharedStorage {
    alignas(32) uint32_t tmem_c;
    alignas(1024) uint64_t bar_a[2];
    alignas(1024) uint64_t bar_b[2];
    alignas(1024) __nv_bfloat16 A_smem[2][64 * 128]; // Double buffer A
    alignas(1024) __nv_bfloat16 B_smem[2][256 * 128]; // Double buffer B
};

__global__ __launch_bounds__(128) void gemm_kernel(
    const __grid_constant__ CUtensorMap dA,
    const __grid_constant__ CUtensorMap dB,
    __nv_bfloat16* C, uint32_t M) 
{
    extern __shared__ char raw[];
    char* ptr = (char*)((((uint64_t)raw) + 1023) & ~1023);
    SharedStorage* smem = reinterpret_cast<SharedStorage*>(ptr);

    int tid = threadIdx.x;
    int cta = cluster_rank_fn();
    uint32_t BM_per_cta = 64;
    uint32_t m_base = blockIdx.x * 128 + cta * BM_per_cta;
    uint32_t n_block = blockIdx.y * 256;

    if (elect_one_sync_fn()) {
        tmem_alloc_fn(&smem->tmem_c, 128);
    }
    __syncthreads();
    uint32_t tmem_c = smem->tmem_c;

    if (tid == 0) {
        init_smem_barrier_fn(&smem->bar_a[0], 1);
        init_smem_barrier_fn(&smem->bar_b[0], 1);
        init_smem_barrier_fn(&smem->bar_a[1], 1);
        init_smem_barrier_fn(&smem->bar_b[1], 1);
        
        mbarrier_arrive_and_expect_tx_fn(&smem->bar_a[0], 16384); 
        tma_load_2d_fn(&dA, &smem->bar_a[0], smem->A_smem[0], 0, m_base);
        tma_load_2d_fn(&dA, &smem->bar_a[0], (char*)smem->A_smem[0] + 8192, 64, m_base);

        mbarrier_arrive_and_expect_tx_fn(&smem->bar_b[0], 65536); 
        tma_load_2d_fn(&dB, &smem->bar_b[0], smem->B_smem[0], 0, n_block);
        tma_load_2d_fn(&dB, &smem->bar_b[0], (char*)smem->B_smem[0] + 32768, 64, n_block);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    int phase_a[2] = {0, 0};
    int phase_b[2] = {0, 0};

    mbarrier_wait_fn(&smem->bar_a[0], phase_a[0]); phase_a[0] ^= 1;
    mbarrier_wait_fn(&smem->bar_b[0], phase_b[0]); phase_b[0] ^= 1;

    int stage_curr = 0;
    int stage_next = 1;
    
    int total_iters = 5120 / 128;
    for (int k_idx = 0; k_idx < total_iters; k_idx++) {
        if (k_idx + 1 < total_iters) {
            if (tid == 0) {
                mbarrier_arrive_and_expect_tx_fn(&smem->bar_a[stage_next], 16384); 
                tma_load_2d_fn(&dA, &smem->bar_a[stage_next], smem->A_smem[stage_next], (k_idx + 1) * 128, m_base);
                tma_load_2d_fn(&dA, &smem->bar_a[stage_next], (char*)smem->A_smem[stage_next] + 8192, (k_idx + 1) * 128 + 64, m_base);
                
                mbarrier_arrive_and_expect_tx_fn(&smem->bar_b[stage_next], 65536); 
                tma_load_2d_fn(&dB, &smem->bar_b[stage_next], smem->B_smem[stage_next], (k_idx + 1) * 128, n_block);
                tma_load_2d_fn(&dB, &smem->bar_b[stage_next], (char*)smem->B_smem[stage_next] + 32768, (k_idx + 1) * 128 + 64, n_block);
            }
        }

        mbarrier_wait_fn(&smem->bar_a[stage_curr], phase_a[stage_curr]);
        phase_a[stage_curr] ^= 1;
        
        mbarrier_wait_fn(&smem->bar_b[stage_curr], phase_b[stage_curr]);
        phase_b[stage_curr] ^= 1;
        
        fence_proxy_async_fn();

        uint32_t* cur_A_ptr0 = (uint32_t*)smem->A_smem[stage_curr];
        uint32_t* cur_A_ptr1 = (uint32_t*)(char*)smem->A_smem[stage_curr] + 8192;
        
        uint32_t* cur_B_ptr0 = (uint32_t*)smem->B_smem[stage_curr];
        uint32_t* cur_B_ptr1 = (uint32_t*)(char*)smem->B_smem[stage_curr] + 32768;

        uint64_t desc_a0[8], desc_b0[8];
        desc_a0[0] = make_smem_desc_k_major(cur_A_ptr0);
        desc_a0[1] = walk_desc_byte((uintptr_t)cur_A_ptr0, desc_a0[0], 128);
        desc_a0[2] = walk_desc_byte((uintptr_t)cur_A_ptr0, desc_a0[0], 256);
        desc_a0[3] = walk_desc_byte((uintptr_t)cur_A_ptr0, desc_a0[0], 384);
        desc_a0[4] = make_smem_desc_k_major(cur_A_ptr1);
        desc_a0[5] = walk_desc_byte((uintptr_t)cur_A_ptr1, desc_a0[4], 128);
        desc_a0[6] = walk_desc_byte((uintptr_t)cur_A_ptr1, desc_a0[4], 256);
        desc_a0[7] = walk_desc_byte((uintptr_t)cur_A_ptr1, desc_a0[4], 384);

        desc_b0[0] = make_smem_desc_k_major(cur_B_ptr0);
        desc_b0[1] = walk_desc_byte((uintptr_t)cur_B_ptr0, desc_b0[0], 128);
        desc_b0[2] = walk_desc_byte((uintptr_t)cur_B_ptr0, desc_b0[0], 256);
        desc_b0[3] = walk_desc_byte((uintptr_t)cur_B_ptr0, desc_b0[0], 384);
        desc_b0[4] = make_smem_desc_k_major(cur_B_ptr1);
        desc_b0[5] = walk_desc_byte((uintptr_t)cur_B_ptr1, desc_b0[4], 128);
        desc_b0[6] = walk_desc_byte((uintptr_t)cur_B_ptr1, desc_b0[4], 256);
        desc_b0[7] = walk_desc_byte((uintptr_t)cur_B_ptr1, desc_b0[4], 384);

        uint32_t accum = (k_idx == 0) ? 0 : 1;

        if (cta == 0 && tid == 0) {
            uint32_t idesc0 = make_instr_desc_fn(128, 256);
            
            for(int i = 0; i < 8; ++i) {
                umma_f16_cg2_fn(tmem_c, desc_a0[i], desc_b0[i], idesc0, accum);
                accum = 1;
            }
            
            umma_commit_2sm_fn(&smem->bar_a[stage_curr]);
        }
        mbarrier_wait_fn(&smem->bar_a[stage_curr], phase_a[stage_curr]);
        phase_a[stage_curr] ^= 1;

        stage_curr = stage_next;
        stage_next ^= 1;
    }
    
    for (uint32_t col_base = 0; col_base < 128; col_base += 64) {
        for (int i = 0; i < 4; i++) {
            uint32_t swizzled_col = ((((col_base + i*32) >> 4) ^ (tid & 7)) << 4) + ((col_base + i*32) & 15);
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(swizzled_col));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float f0 = __uint_as_float(r0);
            float f1 = __uint_as_float(r1);
            float f2 = __uint_as_float(r2);
            float f3 = __uint_as_float(r3);
            
            uint32_t nc = n_block + col_base + i*32;
            uint32_t m_idx = m_base + tid;
            
            __nv_bfloat16* out = C + (uint64_t)m_idx * 7168 + nc;
            if (m_idx < M && nc < 7168) out[0] = __float2bfloat16(f0);
            if (m_idx < M && nc + 1 < 7168) out[1] = __float2bfloat16(f1);
            if (m_idx < M && nc + 2 < 7168) out[2] = __float2bfloat16(f2);
            if (m_idx < M && nc + 3 < 7168) out[3] = __float2bfloat16(f3);
        }
    }
}

// -------------------------------------------------------------------------
// TVM-FFI Binding
// -------------------------------------------------------------------------

namespace tvm_ffi_gemm {

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    
    uint32_t M = A.size(0);
    
    CUtensorMap dA, dB;
    void* a_ptr = A.data_ptr();
    void* b_ptr = B.data_ptr();
    
    CUresult resA = create_tma_2d_descriptor_2B(&dA, a_ptr, 5120, M, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (resA != CUDA_SUCCESS) { fprintf(stderr, "TMA A failed\n"); exit(1); }
    
    CUresult resB = create_tma_2d_descriptor_2B(&dB, b_ptr, 5120, 7168, 64, 256, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (resB != CUDA_SUCCESS) { fprintf(stderr, "TMA B failed\n"); exit(1); }
    
    dim3 grid((M + 127) / 128, 7168 / 256);
    dim3 block(128);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    
    size_t smem_size = sizeof(SharedStorage) + 1024;
    CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
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
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_kernel, dA, dB, static_cast<__nv_bfloat16*>(C.data_ptr()), M));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_gemm