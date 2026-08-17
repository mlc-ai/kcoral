#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
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

#define DRV_CHECK(call) do {                                       \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        fprintf(stderr, "Driver error %d at %s:%d\n",              \
                (int)_e, __FILE__, __LINE__);                      \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_mha {

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
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

__device__ __forceinline__ uint32_t swizzle_128B(uint32_t row, uint32_t col) {
    uint32_t chunk_x = col / 8;
    uint32_t chunk_y = row % 8;
    uint32_t swizzled_chunk_x = chunk_x ^ chunk_y;
    return swizzled_chunk_x * 8 + (col % 8);
}

__device__ __forceinline__ uint32_t offset_Q(int m_offset, int k_offset) {
    return ((m_offset * 64) + (k_offset / 8) * 8 + (k_offset % 8)) * 2;
}

__device__ __forceinline__ uint32_t offset_K(int n_offset, int k_offset) {
    return ((n_offset * 64) + (k_offset / 8) * 8 + (k_offset % 8)) * 2;
}

__device__ __forceinline__ uint32_t offset_V(int n_offset, int k_offset) {
    return ((n_offset * 64) + (k_offset / 8) * 8 + (k_offset % 8)) * 2;
}

__device__ __forceinline__ uint32_t offset_P(int m_offset, int k_offset) {
    return ((m_offset * 64) + (k_offset / 8) * 8 + (k_offset % 8)) * 2;
}

__device__ __forceinline__ void cp_async_128x64_swizzled(void* tmem_dst, const void* smem_src) {
    uint32_t* dst = (uint32_t*)tmem_dst;
    const uint32_t* src = (const uint32_t*)smem_src;
    #pragma unroll
    for (int i = 0; i < 128 * 64 / 4; ++i) {
        asm volatile("cp.async.cg.shared.shared.z [%0], [%1];\n"
                     :: "r"((uint32_t)__cvta_generic_to_shared(&dst[i])),
                        "r"((uint32_t)__cvta_generic_to_shared(&src[i])) : "memory");
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

extern __shared__ __align__(128) uint8_t smem_pool[];

__global__ void mha_kernel(
    __grid_constant__ CUtensorMap desc_Q,
    __grid_constant__ CUtensorMap desc_K,
    __grid_constant__ CUtensorMap desc_V,
    __nv_bfloat16* O_ptr,
    float* LSE_ptr,
    uint32_t S)
{
    setmaxnreg_inc_sync_fn<256>();

    uint32_t bh = blockIdx.y;
    uint32_t q_start = blockIdx.x * 128; 
    
    __nv_bfloat16* Q_shared = (__nv_bfloat16*)(smem_pool + 0);            
    __nv_bfloat16* K_shared = (__nv_bfloat16*)(smem_pool + 32768);         
    __nv_bfloat16* V_shared = (__nv_bfloat16*)(smem_pool + 65536);         
    
    __shared__ uint32_t tmem_addr[1];
    if (threadIdx.x == 0) {
        tmem_alloc_fn(tmem_addr, 512);
    }
    __syncthreads();
    
    __shared__ uint64_t barrier[1];
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&barrier[0], 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();
    
    uint32_t phase = 0;
    
    if (threadIdx.x == 0) {
        uint32_t tx_bytes = 4 * 16384; 
        mbarrier_arrive_and_expect_tx_fn(&barrier[0], tx_bytes);
        
        int s_coord = bh * S + q_start;
        tma_load_2d_fn(&desc_Q, &barrier[0], Q_shared + offset_Q(0, 0), 0, s_coord);
        tma_load_2d_fn(&desc_Q, &barrier[0], Q_shared + offset_Q(0, 64), 64, s_coord);
    }
    mbarrier_wait_fn(&barrier[0], phase);
    phase ^= 1;

    uint32_t tmem_P_base = tmem_addr[0] + 0;
    uint32_t tmem_V = tmem_addr[0] + (128 * 128 * 2) / 4; 
    uint32_t tmem_O = tmem_addr[0] + (128 * 128 * 4) / 4; 

    __nv_bfloat16* P_shared = (__nv_bfloat16*)(smem_pool + 32768); // Reuse K_shared space temporarily

    // --- Pass 1: Accumulate fully scaled QK^T ---
    for (int kv_start = 0; kv_start < S; kv_start += 128) {
        if (threadIdx.x == 0) {
            uint32_t tx_bytes = 4 * 16384; 
            mbarrier_arrive_and_expect_tx_fn(&barrier[0], tx_bytes);
            
            int kv_coord = bh * S + kv_start;
            tma_load_2d_fn(&desc_K, &barrier[0], K_shared + offset_K(0, 0), 0, kv_coord);
            tma_load_2d_fn(&desc_K, &barrier[0], K_shared + offset_K(0, 64), 64, kv_coord);
            
            tma_load_2d_fn(&desc_V, &barrier[0], V_shared + offset_V(0, 0), 0, kv_coord);
            tma_load_2d_fn(&desc_V, &barrier[0], V_shared + offset_V(0, 64), 64, kv_coord);
        }
        mbarrier_wait_fn(&barrier[0], phase);
        phase ^= 1;
        
        __syncthreads();
        fence_proxy_async_fn();
        
        if (threadIdx.x == 0) {
            uint32_t idesc_QKT = make_instr_desc_fn(256, 128);
            #pragma unroll
            for (int k = 0; k < 128; k += 16) {
                uint64_t desc_a = make_smem_desc_sm100_fn(Q_shared + offset_Q(0, k), 1, 1024);
                uint64_t desc_b = make_smem_desc_sm100_fn(K_shared + offset_K(0, k), 1, 1024);
                umma_f16_cg2_fn(tmem_P_base, desc_a, desc_b, idesc_QKT, 1); // Always accumulate
            }
            umma_commit_2sm_fn(&barrier[0]);
        }
        mbarrier_wait_fn(&barrier[0], phase);
        phase ^= 1;
        __syncthreads();
    }
    
    // --- Softmax Scaling & Layout Conversion ---
    uint32_t P_regs[128][16];
    for (int c = 0; c < 128; c += 8) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_P_base + c));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        P_regs[threadIdx.x][0] = r0;
        P_regs[threadIdx.x][1] = r1;
        P_regs[threadIdx.x][2] = r2;
        P_regs[threadIdx.x][3] = r3;
    }
    
    float max_val = -INFINITY;
    for (int c = 0; c < 128; ++c) {
        max_val = fmaxf(max_val, __uint_as_float(P_regs[threadIdx.x][c / 8]));
    }
    
    float sum = 0.0f;
    float scale = 1.0f / sqrtf(128.0f);
    for (int c = 0; c < 128; ++c) {
        float val = __uint_as_float(P_regs[threadIdx.x][c / 8]) * scale;
        float exp_val = expf(val - max_val);
        sum += exp_val;
        P_regs[threadIdx.x][c / 8] = __float_as_uint(exp_val);
    }
    
    if (threadIdx.x < 128) {
        if (q_start + threadIdx.x < S) {
            LSE_ptr[bh * S + q_start + threadIdx.x] = max_val + logf(sum);
        }
    }
    
    __syncthreads();
    for (int c = 0; c < 128; ++c) {
        int swizzled_c = swizzle_128B(threadIdx.x, c);
        P_shared[threadIdx.x * 128 + swizzled_c] = __float2bfloat16(__uint_as_float(P_regs[threadIdx.x][c / 8]) / sum);
    }
    __syncthreads();

    // --- Pass 2: Accumulate Output O ---
    for (int kv_start = 0; kv_start < S; kv_start += 128) {
        uint32_t tmem_P = tmem_addr[0] + 0; 
        
        __syncthreads();
        cp_async_128x64_swizzled((__nv_bfloat16*)tmem_P, P_shared + offset_P(0, kv_start % 128 == 0 ? 0 : 64));
        cp_async_128x64_swizzled((__nv_bfloat16*)tmem_V, V_shared + offset_V(0, kv_start % 128 == 0 ? 0 : 64));
        __syncthreads();
        
        fence_proxy_async_fn();
        
        if (threadIdx.x == 0) {
            uint32_t idesc_PV = make_instr_desc_fn(256, 128);
            #pragma unroll
            for (int k = 0; k < 128; k += 16) {
                uint64_t desc_a = make_smem_desc_sm100_fn((__nv_bfloat16*)tmem_P + k, 1, 1024); 
                uint64_t desc_b = make_smem_desc_sm100_fn((__nv_bfloat16*)tmem_V + k * 2, 128, 1024); 
                umma_f16_cg2_fn(tmem_O, desc_a, desc_b, idesc_PV, 1); // Accumulate
            }
            umma_commit_2sm_fn(&barrier[0]);
        }
        mbarrier_wait_fn(&barrier[0], phase);
        phase ^= 1;
        __syncthreads();
    }
    
    // --- Epilogue: Store Output O ---
    for (int i = 0; i < 128; ++i) {
        int lane = i % 128;
        int row = i / 128;
        
        uint32_t r0, r1;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x2.b32 {%0,%1}, [%2];"
        : "=r"(r0), "=r"(r1) : "r"(tmem_O + lane));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        uint32_t val = pack_bf16_fn(r0, r1);
        
        if (row == 0) {
            if (global_row + row < S) {
                *(uint2*)&O_ptr[(global_row + row) * 128 + lane] = val;
            }
        } else {
            if (global_row + row < S) {
                *(uint2*)&O_ptr[(global_row + row) * 128 + lane] = val;
            }
        }
    }
    
    tmem_dealloc_fn(tmem_addr[0], 512);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    uint32_t B = Q.size(0);
    uint32_t H = Q.size(1);
    uint32_t S = Q.size(2);
    
    __nv_bfloat16* O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_ptr = static_cast<float*>(LSE.data_ptr());
    
    CUtensorMap tma_Q, tma_K, tma_V;
    DRV_CHECK(create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), 128, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    DRV_CHECK(create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), 128, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    DRV_CHECK(create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), 128, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
    dim3 grid((S + 127) / 128, B * H);
    dim3 block(128); 
    
    int smem_bytes = 98304;
    CUDA_CHECK(cudaFuncSetAttribute(mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
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
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, mha_kernel, tma_Q, tma_K, tma_V, O_ptr, LSE_ptr, S));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha::run);

} // namespace tvm_ffi_mha