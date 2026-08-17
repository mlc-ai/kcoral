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

__device__ __forceinline__ void tma_load_4d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
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

__device__ __forceinline__ uint64_t make_smem_desc(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16; 
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32; 
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    uint32_t base_offset = (addr >> 7) & 0x7;
    d |= ((uint64_t)base_offset) << 49;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn_cg2(uint32_t M, uint32_t N, uint32_t a_major, uint32_t b_major) {
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

__device__ __forceinline__ uint32_t swizzle_128B(uint32_t row, uint32_t col) {
    uint32_t chunk_x = col / 8;
    uint32_t chunk_y = row % 8;
    uint32_t swizzled_chunk_x = chunk_x ^ chunk_y;
    return swizzled_chunk_x * 8 + (col % 8);
}

CUresult create_tma_4d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t dim0, uint64_t dim1, uint64_t dim2, uint64_t dim3, uint32_t box0, uint32_t box1, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[4] = {dim0, dim1, dim2, dim3};
    cuuint64_t globalStrides[3] = {dim0 * 2, dim0 * dim1 * 2, dim0 * dim1 * dim2 * 2};
    cuuint32_t boxDim[4] = {box0, box1, 1, 1};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        4,
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

__device__ __forceinline__ uint32_t pack_bf16_fn(uint32_t fp32_a, uint32_t fp32_b) {
    __nv_bfloat16 a = __float2bfloat16(__uint_as_float(fp32_a));
    __nv_bfloat16 b = __float2bfloat16(__uint_as_float(fp32_b));
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};"
        : "=r"(result)
        : "h"(*reinterpret_cast<uint16_t*>(&a)),
          "h"(*reinterpret_cast<uint16_t*>(&b)));
    return result;
}

__device__ __forceinline__ uint32_t pack_bf16_fn_4(float f0, float f1, float f2, float f3) {
    __nv_bfloat16 bf[4];
    bf[0] = __float2bfloat16(f0);
    bf[1] = __float2bfloat16(f1);
    bf[2] = __float2bfloat16(f2);
    bf[3] = __float2bfloat16(f3);
    uint32_t v0, v1;
    asm("mov.b32 %0, {%1, %2};" : "=r"(v0) : "h"(*reinterpret_cast<uint16_t*>(&bf[0])), "h"(*reinterpret_cast<uint16_t*>(&bf[1])));
    asm("mov.b32 %0, {%1, %2};" : "=r"(v1) : "h"(*reinterpret_cast<uint16_t*>(&bf[2])), "h"(*reinterpret_cast<uint16_t*>(&bf[3])));
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};" : "=r"(result) : "r"(v0), "r"(v1));
    return result;
}

__device__ __forceinline__ void cp_async_128x64_swizzled(void* tmem_dst, const void* smem_src) {
    const uint8_t* src = (const uint8_t*)smem_src;
    uint32_t* tmem_buf = reinterpret_cast<uint32_t*>(__cvta_generic_to_shared((void*)tmem_dst));
    
    #pragma unroll
    for (int i = 0; i < 2048; ++i) {
        int row = (i * 8) / 64;
        int col = (i * 8) % 64;
        
        uint32_t swizzled_col = (swizzle_128B(row, col) / 4) * 4;
        
        float4 val = *(float4*)&((const char*)smem_src)[(row * 64 + swizzled_col) * 2];
        
        uint32_t dst_addr = __cvta_generic_to_shared(&tmem_buf[i * 8]);
        
        asm volatile("cp.async.cg.shared.shared.z [%0], [%1];"
                     :: "r"(dst_addr), "r"(__cvta_generic_to_shared(&((const uint32_t*)smem_src)[(row * 64 + swizzled_col) / 2])) : "memory");
    }
}

__device__ __forceinline__ void named_barrier_sync_fn(int bar_id, int count) {
    asm volatile("barrier.sync.aligned %0, %1;" :: "r"(bar_id), "r"(count));
}

extern __shared__ __align__(1024) uint8_t smem_pool[];

__global__ void mha_kernel(
    const __grid_constant__ CUtensorMap desc_Q,
    const __grid_constant__ CUtensorMap desc_K,
    const __grid_constant__ CUtensorMap desc_V,
    __nv_bfloat16* O_ptr,
    float* LSE_ptr,
    uint32_t S, uint32_t H)
{
    uint32_t cta = cluster_rank_fn();
    uint32_t bh = blockIdx.y;
    uint32_t q_start = blockIdx.x * 128;
    
    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem_pool);
    uint32_t pad = (1024 - (smem_addr % 1024)) % 1024;
    uint8_t* aligned_smem = smem_pool + pad;
    
    __nv_bfloat16* Q_shared_0 = (__nv_bfloat16*)(aligned_smem + 0);                
    __nv_bfloat16* Q_shared_1 = (__nv_bfloat16*)(aligned_smem + 8192);             
    __nv_bfloat16* K_shared_0 = (__nv_bfloat16*)(aligned_smem + 16384);            
    __nv_bfloat16* K_shared_1 = (__nv_bfloat16*)(aligned_smem + 24576);            
    __nv_bfloat16* V_shared_0 = (__nv_bfloat16*)(aligned_smem + 32768);            
    __nv_bfloat16* V_shared_1 = (__nv_bfloat16*)(aligned_smem + 40960);            
    __nv_bfloat16* P_shared_0 = (__nv_bfloat16*)(aligned_smem + 16384);            
    __nv_bfloat16* P_shared_1 = (__nv_bfloat16*)(aligned_smem + 24576);           
    
    __shared__ uint32_t tmem_addr[2];
    if (threadIdx.x == 0) {
        tmem_alloc_fn(tmem_addr + cta, 256);
    }
    __syncthreads();
    
    __shared__ uint64_t barrier_Q[2];
    __shared__ uint64_t barrier_KV[2];
    __shared__ uint64_t barrier_PV[2];
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&barrier_Q[cta], 1);
        init_smem_barrier_fn(&barrier_KV[cta], 1);
        init_smem_barrier_fn(&barrier_PV[cta], 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();
    
    uint32_t phase = 0;
    
    if (threadIdx.x == 0) {
        uint32_t tx_bytes = 2 * 16384; 
        mbarrier_arrive_and_expect_tx_fn(&barrier_Q[cta], tx_bytes);
        
        int batch_idx = bh / H;
        int head_idx = bh % H;
        int s_coord = q_start;
        tma_load_4d_fn(&desc_Q, &barrier_Q[cta], (void*)(Q_shared_0), 0, s_coord, head_idx, batch_idx);
        tma_load_4d_fn(&desc_Q, &barrier_Q[cta], (void*)(Q_shared_1), 64, s_coord, head_idx, batch_idx);
    }
    mbarrier_wait_fn(&barrier_Q[cta], phase);
    phase ^= 1;
    
    uint32_t O_regs[16][4];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        O_regs[i][0] = __float_as_uint(0.0f);
        O_regs[i][1] = __float_as_uint(0.0f);
        O_regs[i][2] = __float_as_uint(0.0f);
        O_regs[i][3] = __float_as_uint(0.0f);
    }
    
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float scale = 1.0f / sqrtf(128.0f);
    
    uint32_t tmem_Q_base = tmem_addr[cta] + 0;
    uint32_t tmem_P_base = tmem_addr[cta] + 8192;
    uint32_t tmem_O_base = tmem_addr[cta] + 0;

    for (int kv_start = 0; kv_start < S; kv_start += 128) {
        if (threadIdx.x == 0) {
            uint32_t tx_bytes = 4 * 16384; 
            mbarrier_arrive_and_expect_tx_fn(&barrier_KV[cta], tx_bytes);
            
            int batch_idx = bh / H;
            int head_idx = bh % H;
            int kv_coord = kv_start;
            tma_load_4d_fn(&desc_K, &barrier_KV[cta], (void*)(K_shared_0), 0, kv_coord, head_idx, batch_idx);
            tma_load_4d_fn(&desc_K, &barrier_KV[cta], (void*)(K_shared_1), 64, kv_coord, head_idx, batch_idx);
            tma_load_4d_fn(&desc_V, &barrier_KV[cta], (void*)(V_shared_0), 0, kv_coord, head_idx, batch_idx);
            tma_load_4d_fn(&desc_V, &barrier_KV[cta], (void*)(V_shared_1), 64, kv_coord, head_idx, batch_idx);
        }
        mbarrier_wait_fn(&barrier_KV[cta], phase);
        phase ^= 1;
        
        __syncthreads();
        fence_proxy_async_fn();
        
        uint32_t accum_P = 0; 
        
        if (threadIdx.x == 0) {
            uint32_t idesc_QKT = make_instr_desc_fn_cg2(64, 64, 0, 0);
            #pragma unroll
            for (int k_step = 0; k_step < 64; k_step += 16) {
                uint64_t desc_a0 = make_smem_desc(Q_shared_0 + k_step, 1, 1024);
                uint64_t desc_b0 = make_smem_desc(K_shared_0 + k_step, 1, 1024);
                umma_f16_cg2_fn(tmem_P_base, desc_a0, desc_b0, idesc_QKT, accum_P);
                
                uint64_t desc_a1 = make_smem_desc(Q_shared_1 + k_step, 1, 1024);
                uint64_t desc_b1 = make_smem_desc(K_shared_1 + k_step, 1, 1024);
                umma_f16_cg2_fn(tmem_P_base, desc_a1, desc_b1, idesc_QKT, accum_P);
            }
            umma_commit_2sm_fn(&barrier_KV[cta]);
        }
        mbarrier_wait_fn(&barrier_KV[cta], phase);
        phase ^= 1;
        __syncthreads();
        
        uint32_t P_regs[8][4]; 
        
        for (uint32_t col = 0; col < 64; col += 8) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_P_base + cta * 4096 + col));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            P_regs[(col / 8)][0] = r0;
            P_regs[(col / 8)][1] = r1;
            P_regs[(col / 8)][2] = r2;
            P_regs[(col / 8)][3] = r3;
        }
        
        float thread_max = -INFINITY;
        for (int c = 0; c < 64; ++c) {
            float val = __uint_as_float(P_regs[c / 4][c % 4]);
            if (kv_start + c >= S) val = -INFINITY;
            thread_max = fmaxf(thread_max, val);
        }
        
        float new_max = fmaxf(running_max, thread_max);
        
        if (new_max != running_max) {
            float factor = expf((running_max - new_max) * scale);
            for (int r = 0; r < 16; ++r) {
                for (int c = 0; c < 4; ++c) {
                    O_regs[r][c] = __float_as_uint(__uint_as_float(O_regs[r][c]) * factor);
                }
            }
            running_sum *= factor;
        }
        
        if (thread_max > -INFINITY) {
            running_sum += expf((thread_max - new_max) * scale);
        }
        running_max = new_max;
        
        for (int c = 0; c < 64; ++c) {
            float exp_val = 0.0f;
            if (kv_start + c < S) {
                float val = __uint_as_float(P_regs[c / 4][c % 4]);
                exp_val = expf((val - thread_max) * scale);
            }
            int swizzled_c = swizzle_128B(threadIdx.x, c);
            P_shared_0[threadIdx.x * 64 + swizzled_c] = __float2bfloat16(exp_val);
        }
        
        __syncthreads();
        fence_proxy_async_fn();
        
        if (threadIdx.x == 0) {
            uint32_t idesc_PV = make_instr_desc_fn_cg2(64, 64, 0, 1);
            #pragma unroll
            for (int k_step = 0; k_step < 64; k_step += 16) {
                uint64_t desc_a = make_smem_desc(P_shared_0 + k_step, 1, 1024); 
                uint64_t desc_b0 = make_smem_desc(V_shared_0 + k_step * 64, 8192, 1024); 
                uint64_t desc_b1 = make_smem_desc(V_shared_1 + k_step * 64, 8192, 1024); 
                
                umma_f16_cg2_fn(tmem_O_base, desc_a, desc_b0, idesc_PV, 1); 
                umma_commit_2sm_fn(&barrier_PV[cta]);
                umma_f16_cg2_fn(tmem_O_base + 2048, desc_a, desc_b1, idesc_PV, 1); 
                umma_commit_2sm_fn(&barrier_PV[cta]);
            }
        }
        mbarrier_wait_fn(&barrier_PV[cta], phase);
        phase ^= 1;
        __syncthreads();
    }
    
    if (threadIdx.x < 128) {
        if (q_start + threadIdx.x < S) {
            LSE_ptr[bh * S + q_start + threadIdx.x] = running_max + logf(running_sum);
        }
    }
    
    __syncthreads();
    
    for (int d = 0; d < 128; d += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(tmem_O_base + d));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float f0 = __uint_as_float(r0) / running_sum;
        float f1 = __uint_as_float(r1) / running_sum;
        float f2 = __uint_as_float(r2) / running_sum;
        float f3 = __uint_as_float(r3) / running_sum;
        
        uint32_t val = pack_bf16_fn_4(f0, f1, f2, f3);
        
        int batch_idx = bh / H;
        int head_idx = bh % H;
        uint32_t global_row = q_start + threadIdx.x;
        if (global_row < S) {
            *(uint32_t*)&O_ptr[(batch_idx * H * S + head_idx * S + global_row) * 128 + d] = val;
        }
    }
    
    tmem_dealloc_fn(tmem_addr[cta], 256);
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
    DRV_CHECK(create_tma_4d_descriptor_2B(&tma_Q, Q.data_ptr(), 128, S, H, B, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    DRV_CHECK(create_tma_4d_descriptor_2B(&tma_K, K.data_ptr(), 128, S, H, B, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    DRV_CHECK(create_tma_4d_descriptor_2B(&tma_V, V.data_ptr(), 128, S, H, B, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
    dim3 grid((S + 127) / 128, B * H);
    dim3 block(128); 
    
    int smem_bytes = 96 * 1024;
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
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, mha_kernel, tma_Q, tma_K, tma_V, O_ptr, LSE_ptr, S, H));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha::run);

} // namespace tvm_ffi_mha