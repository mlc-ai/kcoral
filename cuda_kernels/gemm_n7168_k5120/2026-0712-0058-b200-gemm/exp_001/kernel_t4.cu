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

#define CU_CHECK(call) do {                                      \
    CUresult _e = (call);                                        \
    if (_e != CUDA_SUCCESS) {                                    \
        fprintf(stderr, "CU error %d at %s:%d\n",               \
                (int)_e, __FILE__, __LINE__);                      \
        exit(1);                                                   \
    }                                                              \
} while(0)

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

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, uint32_t a_major, uint32_t b_major) {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= (a_major << 15);
    d |= (b_major << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
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

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__global__ void __launch_bounds__(128) gemm_kernel(const __grid_constant__ CUtensorMap tma_A, const __grid_constant__ CUtensorMap tma_B, __nv_bfloat16* C, uint32_t M) {
    extern __shared__ __align__(1024) char smem_pool[];
    
    __nv_bfloat16* smem_A[2] = { (__nv_bfloat16*)smem_pool, (__nv_bfloat16*)(smem_pool + 8192) };
    __nv_bfloat16* smem_B[2] = { (__nv_bfloat16*)(smem_pool + 16384), (__nv_bfloat16*)(smem_pool + 49152) };

    uint64_t* bar_A[2] = { (uint64_t*)(smem_pool + 81920), (uint64_t*)(smem_pool + 81928) };
    uint64_t* bar_B[2] = { (uint64_t*)(smem_pool + 81936), (uint64_t*)(smem_pool + 81944) };
    uint64_t* bar_WGMMA = (uint64_t*)(smem_pool + 81952);
    uint32_t* smem_tmem_c = (uint32_t*)(smem_pool + 81960);

    if (threadIdx.x == 0) {
        tmem_alloc_fn(smem_tmem_c, 256);
    }
    __syncthreads();
    uint32_t tmem_c = *smem_tmem_c;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar_A[0], 1);
        init_smem_barrier_fn(bar_A[1], 1);
        init_smem_barrier_fn(bar_B[0], 1);
        init_smem_barrier_fn(bar_B[1], 1);
        init_smem_barrier_fn(bar_WGMMA, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    uint32_t BM = 64;
    uint32_t BN = 256;
    uint32_t N = 7168;
    uint32_t m_base = blockIdx.x * 128 + cluster_rank_fn() * 64;

    uint32_t gemm_idesc = make_instr_desc_fn(BM, BN, 0, 0);

    uint32_t phase_A[2] = {0, 0};
    uint32_t phase_B[2] = {0, 0};
    uint32_t tmem_c_phase = 0;

    for (uint32_t n_block = 0; n_block < 14; ++n_block) {
        uint32_t n_base = n_block * 512;
        uint32_t n_base_cta = n_base + (cluster_rank_fn() % 2) * 256;
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_A[0], 8192);
            tma_load_2d_fn(&tma_A, bar_A[0], smem_A[0], 0, m_base);
            
            mbarrier_arrive_and_expect_tx_fn(bar_B[0], 32768);
            tma_load_2d_fn(&tma_B, bar_B[0], smem_B[0], 0, n_base_cta);
        }

        for (uint32_t k_block = 0; k_block < 80; ++k_block) {
            int buf_idx = k_block % 2;
            int next_buf_idx = 1 - buf_idx;

            mbarrier_wait_fn(bar_A[buf_idx], phase_A[buf_idx]);
            mbarrier_wait_fn(bar_B[buf_idx], phase_B[buf_idx]);
            phase_A[buf_idx] ^= 1;
            phase_B[buf_idx] ^= 1;

            if (k_block < 79) {
                if (threadIdx.x == 0) {
                    uint32_t k_next = k_block * 64 + 64;
                    
                    mbarrier_arrive_and_expect_tx_fn(bar_A[next_buf_idx], 8192);
                    tma_load_2d_fn(&tma_A, bar_A[next_buf_idx], smem_A[next_buf_idx], k_next, m_base);
                    
                    mbarrier_arrive_and_expect_tx_fn(bar_B[next_buf_idx], 32768);
                    tma_load_2d_fn(&tma_B, bar_B[next_buf_idx], smem_B[next_buf_idx], k_next, n_base_cta);
                }
            }

            fence_proxy_async_fn();

            if (threadIdx.x == 0) {
                __nv_bfloat16* cur_smem_A = smem_A[buf_idx];
                __nv_bfloat16* cur_smem_B = smem_B[buf_idx];

                for(int i = 0; i < 4; ++i) {
                    uint64_t desc_A = make_smem_desc_sm100_fn(cur_smem_A + i * 16, 1, 1024);
                    uint64_t desc_B = make_smem_desc_sm100_fn(cur_smem_B + k_block * 256 * 16 + i * 16, 1, 1024);
                    uint32_t accum = (k_block == 0 && i == 0) ? 0 : 1;
                    umma_f16_cg2_fn(tmem_c, desc_A, desc_B, gemm_idesc, accum);
                }
                umma_commit_2sm_fn(bar_WGMMA);
            }
            mbarrier_wait_fn(bar_WGMMA, tmem_c_phase);
            tmem_c_phase ^= 1;
            __syncthreads();
        }

        if (threadIdx.x < 64) {
            uint32_t m_idx = m_base + threadIdx.x;
            for(uint32_t col = 0; col < 256; col += 4) {
                uint32_t r0, r1, r2, r3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(col));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                
                if (m_idx < M) {
                    float f0 = __uint_as_float(r0);
                    float f1 = __uint_as_float(r1);
                    float f2 = __uint_as_float(r2);
                    float f3 = __uint_as_float(r3);
                    uint32_t nc = n_base_cta + col;
                    __nv_bfloat16* out = C + (uint64_t)m_idx * N + nc;
                    if (nc     < N) out[0] = __float2bfloat16(f0);
                    if (nc + 1 < N) out[1] = __float2bfloat16(f1);
                    if (nc + 2 < N) out[2] = __float2bfloat16(f2);
                    if (nc + 3 < N) out[3] = __float2bfloat16(f3);
                }
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_c, 256);
    }
}

namespace tvm_ffi {

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    int64_t M = A.size(0);
    const int64_t N = 7168;
    const int64_t K = 5120;

    CUtensorMap tma_A, tma_B;
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_A, static_cast<void*>(const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(A.data_ptr()))), K, M, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_B, static_cast<void*>(const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(B.data_ptr()))), K, N, 64, 256, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    uint32_t BM = 64;
    uint32_t blocks_x = (M + 127) / 128;

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(blocks_x, 1, 1);
    config.blockDim = dim3(128, 1, 1);
    config.dynamicSmemBytes = 82944;
    config.stream = stream;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_kernel, tma_A, tma_B, static_cast<__nv_bfloat16*>(C.data_ptr()), M));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi