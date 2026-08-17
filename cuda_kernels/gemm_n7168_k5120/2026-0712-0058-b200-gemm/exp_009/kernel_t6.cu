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
                (int)_e, __FILE__, __LINE__);                      \
        exit(1);                                                   \
    }                                                              \
} while(0)

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
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

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__device__ __forceinline__ void tmem_alloc(uint32_t* ptrs, uint32_t& tmem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(ptrs);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(a), "r"(ncols));
    tmem = *ptrs;
}

__device__ __forceinline__ void tmem_dealloc(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
        :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tma_load_2d_cg2(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t ba = (uint32_t)__cvta_generic_to_shared(&bar[0]) & 0xFEFFFFFF; 
    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(sa), "l"((uint64_t)d), "r"(c0), "r"(c1), "r"(ba) : "memory");
}

__device__ __forceinline__ void umma_f16_cg2(
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

__device__ __forceinline__ uint64_t make_smem_desc(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc(uint32_t M, uint32_t N) {
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

__global__ void gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* C,
    uint32_t M, uint32_t N)
{
    extern __shared__ __align__(1024) __nv_bfloat16 smem_pool[];
    __nv_bfloat16* A_smem = smem_pool;
    __nv_bfloat16* B_smem = A_smem + 4096; 
    uint64_t* barrier_tma = (uint64_t*)(B_smem + 8192);
    uint64_t* barrier_c0 = barrier_tma + 1;

    uint32_t m_block = blockIdx.x;
    uint32_t n_block = blockIdx.y;
    uint32_t cta_id = cluster_rank_fn() % 2;
    uint32_t m_off = m_block * 128 + cta_id * 64;
    uint32_t n_chunk = n_block * 2 + cta_id;
    uint32_t n_off = n_chunk * 128;
    uint32_t tid = threadIdx.x;

    if (m_off >= M) return;

    uint32_t C0_tmem, C1_tmem;
    if (tid < 32) {
        tmem_alloc(&C0_tmem, C0_tmem, 128);
        tmem_alloc(&C1_tmem, C1_tmem, 128);
    }
    __syncthreads();

    if (tid == 0) {
        init_smem_barrier_fn(barrier_tma, 1);
        init_smem_barrier_fn(barrier_c0, 1);
    }
    __syncthreads();

    uint32_t idesc = make_instr_desc(64, 128);

    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(barrier_tma, 24576);
        tma_load_2d_cg2(&tma_A, barrier_tma, A_smem, 0, m_off);
        tma_load_2d_cg2(&tma_B, barrier_tma, B_smem, 0, n_off);
        tma_load_2d_cg2(&tma_B, barrier_tma, B_smem + 4096, 0, n_off + 64);
    }
    
    int phase_tma = 0;
    int phase_c0 = 0;

    for (int k = 0; k < 80; ++k) {
        mbarrier_wait_fn(barrier_tma, phase_tma);
        phase_tma ^= 1;

        if (k < 79) {
            if (tid == 0) {
                int s_next = (k + 1) % 2;
                mbarrier_arrive_and_expect_tx_fn(barrier_tma, 24576);
                tma_load_2d_cg2(&tma_A, barrier_tma, A_smem + s_next * 4096, (k + 1) * 64, m_off);
                tma_load_2d_cg2(&tma_B, barrier_tma, B_smem + s_next * 8192, (k + 1) * 64, n_off);
                tma_load_2d_cg2(&tma_B, barrier_tma, B_smem + s_next * 8192 + 4096, (k + 1) * 64, n_off + 64);
            }
        }

        int s = k % 2;
        fence_proxy_async_fn();

        if (cta_id == 0) {
            mbarrier_arrive_and_expect_tx_fn(barrier_c0, 0);
            
            for (int i = 0; i < 4; ++i) {
                uint64_t desc_a = make_smem_desc(A_smem + s * 4096, 1, 1024);
                uint64_t desc_b = make_smem_desc(B_smem + s * 8192, 8192, 1024);
                umma_f16_cg2(C0_tmem + (i << 4), desc_a, desc_b, idesc, i == 0 ? 0 : 1);
            }
            
            for (int i = 0; i < 4; ++i) {
                uint64_t desc_a = make_smem_desc(A_smem + s * 4096, 1, 1024);
                uint64_t desc_b1 = make_smem_desc(B_smem + s * 8192 + 4096, 8192, 1024);
                umma_f16_cg2(C1_tmem + (i << 4), desc_a, desc_b1, idesc, i == 0 ? 0 : 1);
            }
            
            umma_commit_2sm_fn(barrier_c0);
        }

        mbarrier_wait_fn(barrier_c0, phase_c0);
        phase_c0 ^= 1;
        
        __syncthreads();
    }

    // Epilogue leveraging logical row mapping across the entire warpgroup
    __syncthreads();
    fence_proxy_async_fn();
    
    for (uint32_t col = 0; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3;
        uint32_t r4, r5, r6, r7;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(C0_tmem + col));
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(C1_tmem + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float f0 = __uint_as_float(r0);
        float f1 = __uint_as_float(r1);
        float f2 = __uint_as_float(r2);
        float f3 = __uint_as_float(r3);
        
        float f4 = __uint_as_float(r4);
        float f5 = __uint_as_float(r5);
        float f6 = __uint_as_float(r6);
        float f7 = __uint_as_float(r7);
        
        if (tid < 64) {
            uint32_t m_idx = m_off + tid;
            if (m_idx < M) {
                uint32_t nc = n_off + col;
                if (nc + 3 < N) {
                    __nv_bfloat16* out0 = C + (uint64_t)m_idx * N + nc;
                    out0[0] = __float2bfloat16(f0);
                    out0[1] = __float2bfloat16(f1);
                    out0[2] = __float2bfloat16(f2);
                    out0[3] = __float2bfloat16(f3);
                }
                uint32_t nc1 = n_off + col + 64;
                if (nc1 + 3 < N) {
                    __nv_bfloat16* out1 = C + (uint64_t)m_idx * N + nc1;
                    out1[0] = __float2bfloat16(f4);
                    out1[1] = __float2bfloat16(f5);
                    out1[2] = __float2bfloat16(f6);
                    out1[3] = __float2bfloat16(f7);
                }
            }
        }
    }

    if (tid < 32) {
        tmem_dealloc(C0_tmem, 128);
        tmem_dealloc(C1_tmem, 128);
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
    uint32_t M = A.size(0);
    const uint32_t N = 7168;
    const uint32_t K = 5120;
    
    CUtensorMap tma_A, tma_B;
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_A, A.data_ptr(), K, M, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_B, B.data_ptr(), K, N, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        
    uint32_t num_m_blocks = (M + 127) / 128;
    dim3 grid(num_m_blocks, 28); 
    dim3 block(128); 
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 50192;
    config.stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_kernel, tma_A, tma_B, static_cast<__nv_bfloat16*>(C.data_ptr()), M, N));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(config.stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);