#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <stdio.h>
#include <mma.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CU_CHECK(call) do {                                      \
    CUresult _e = (call);                                        \
    if (_e != CUDA_SUCCESS) {                                    \
        fprintf(stderr, "CU error %d at %s:%d\n",                 \
                (int)_e, __FILE__, __LINE__);                    \
        exit(1);                                                 \
    }                                                            \
} while(0)

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_kernel {

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_dim0, uint64_t gmem_dim1, uint64_t gmem_dim2, uint32_t smem_dim0, uint32_t smem_dim1, uint32_t smem_dim2, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[3] = {gmem_dim0, gmem_dim1, gmem_dim2};
    cuuint64_t globalStrides[2] = {gmem_dim0 * 2, gmem_dim0 * gmem_dim1 * 2};
    cuuint32_t boxDim[3] = {smem_dim0, smem_dim1, smem_dim2};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        3,
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

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
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

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t addr, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(addr));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void umma_f16_cg1_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_1sm(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1"
        ".mbarrier::arrive::one.shared::cta.b64"
        " [%0];"
        :: "r"(a));
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo, bool k_major) {
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

struct SharedStorage {
    alignas(128) __nv_bfloat16 s_Q0[16384];
    alignas(128) __nv_bfloat16 s_K[16384];
    alignas(128) __nv_bfloat16 s_P[16384];
    alignas(128) __nv_bfloat16 s_V0[8192];
    alignas(128) __nv_bfloat16 s_V1[8192];
    alignas(128) __nv_bfloat16 s_O[16384];
    alignas(128) float s_M[128];
    alignas(128) float s_L[128];
};

__global__ __launch_bounds__(128) void run_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O,
    float* LSE,
    uint32_t S)
{
    extern __shared__ char smem_pool[];
    SharedStorage* smem = (SharedStorage*)smem_pool;
    
    __shared__ alignas(8) uint64_t bar_Q, bar_KV;
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&bar_Q, 1);
        init_smem_barrier_fn(&bar_KV, 1);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();
    
    uint32_t row_base = blockIdx.x * 128;
    uint32_t bh = blockIdx.y;
    uint32_t tid = threadIdx.x;
    
    if (tid < 128) {
        smem->s_M[tid] = -INFINITY;
        smem->s_L[tid] = 0.0f;
    }
    __syncthreads();
    
    uint32_t s_tmem, o_tmem;
    if (tid == 0) {
        tmem_alloc_fn(&s_tmem, 128);
        tmem_alloc_fn(&o_tmem, 128);
    }
    __syncthreads();
    
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(&bar_Q, 32768);
        tma_load_3d_fn(&tma_Q, &bar_Q, smem->s_Q0, 0, bh * S + row_base, bh);
        tma_load_3d_fn(&tma_Q, &bar_Q, smem->s_Q0 + 8192, 64, bh * S + row_base, bh);
    }
    mbarrier_wait_fn(&bar_Q, 0);
    
    uint32_t num_iters = (S + 127) / 128;
    uint32_t phase_KV = 0;
    
    float scale = 1.0f / sqrtf(128);
    uint32_t idesc_QKT = make_instr_desc_fn(128, 128, 0, 0);
    uint32_t idesc_PV = make_instr_desc_fn(128, 128, 0, 1);
    
    uint64_t desc_Q0 = make_smem_desc_sm100_fn(smem->s_Q0, 1, 1024, true);
    uint64_t desc_K = make_smem_desc_sm100_fn(smem->s_K, 1, 1024, true);
    uint64_t desc_P = make_smem_desc_sm100_fn(smem->s_P, 1, 1024, true);
    uint64_t desc_V0 = make_smem_desc_sm100_fn(smem->s_V0, 16384, 1024, false);
    uint64_t desc_V1 = make_smem_desc_sm100_fn(smem->s_V1, 16384, 1024, false);
    
    for (uint32_t j = 0; j < num_iters; j++) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(&bar_KV, 65536);
            tma_load_3d_fn(&tma_K, &bar_KV, smem->s_K, 0, bh * S + j * 128, bh);
            tma_load_3d_fn(&tma_K, &bar_KV, smem->s_K + 8192, 64, bh * S + j * 128, bh);
            
            tma_load_3d_fn(&tma_V, &bar_KV, smem->s_V0, 0, bh * S + j * 128, bh);
            tma_load_3d_fn(&tma_V, &bar_KV, smem->s_V1, 64, bh * S + j * 128, bh);
        }
        mbarrier_wait_fn(&bar_KV, phase_KV);
        phase_KV ^= 1;
        
        if (tid == 0) {
            for (int k = 0; k < 8; k++) {
                uint64_t d_Q0_k = desc_Q0 + k * 16;
                uint64_t d_K_k = desc_K + k * 16;
                uint32_t s_addr = s_tmem + k * 64;
                umma_f16_cg1_fn(s_addr, d_Q0_k, d_K_k, idesc_QKT, 1);
            }
            umma_commit_1sm(&bar_KV);
        }
        mbarrier_wait_fn(&bar_KV, phase_KV);
        phase_KV ^= 1;
        
        float max_val = -INFINITY;
        for (int col = 0; col < 64; col += 4) {
            uint32_t r0, r1, r2, r3;
            uint32_t addr_s = s_tmem + tid * 65536 + col;
            tmem_load_4x_fn(addr_s, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            float val0 = __uint_as_float(r0); if (j * 128 + col + 0 >= S) val0 = -INFINITY;
            float val1 = __uint_as_float(r1); if (j * 128 + col + 1 >= S) val1 = -INFINITY;
            float val2 = __uint_as_float(r2); if (j * 128 + col + 2 >= S) val2 = -INFINITY;
            float val3 = __uint_as_float(r3); if (j * 128 + col + 3 >= S) val3 = -INFINITY;
            max_val = fmaxf(max_val, fmaxf(fmaxf(val0, val1), fmaxf(val2, val3)));
        }

        float sum_val = 0;
        for (int col = 0; col < 64; col += 4) {
            uint32_t r0, r1, r2, r3;
            uint32_t addr_s = s_tmem + tid * 65536 + col;
            tmem_load_4x_fn(addr_s, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            float val0 = __uint_as_float(r0); if (j * 128 + col + 0 >= S) val0 = -INFINITY; else val0 = expf(val0 * scale - max_val);
            float val1 = __uint_as_float(r1); if (j * 128 + col + 1 >= S) val1 = -INFINITY; else val1 = expf(val1 * scale - max_val);
            float val2 = __uint_as_float(r2); if (j * 128 + col + 2 >= S) val2 = -INFINITY; else val2 = expf(val2 * scale - max_val);
            float val3 = __uint_as_float(r3); if (j * 128 + col + 3 >= S) val3 = -INFINITY; else val3 = expf(val3 * scale - max_val);
            
            sum_val += val0 + val1 + val2 + val3;
            
            int swizzled_col = (col / 8 ^ (tid % 8)) * 8 + (col % 8);
            int idx = tid * 128 + swizzled_col;
            smem->s_P[idx] = __float2bfloat16(val0);
            smem->s_P[idx+1] = __float2bfloat16(val1);
            smem->s_P[idx+2] = __float2bfloat16(val2);
            smem->s_P[idx+3] = __float2bfloat16(val3);
        }
        
        float global_M = smem->s_M[tid];
        float nm = fmaxf(global_M, max_val);
        float alpha = expf(global_M - nm);
        smem->s_M[tid] = nm;
        smem->s_L[tid] = smem->s_L[tid] * alpha + sum_val;
        
        if (tid == 0) {
            for (int k = 0; k < 8; k++) {
                uint64_t d_P_k = desc_P + k * 16;
                uint64_t d_V0_k = desc_V0 + k * 1024;
                uint64_t d_V1_k = desc_V1 + k * 1024;
                umma_f16_cg1_fn(o_tmem, d_P_k, d_V0_k, idesc_PV, 1);
                umma_f16_cg1_fn(o_tmem + 8192, d_P_k, d_V1_k, idesc_PV, 1);
            }
            umma_commit_1sm(&bar_KV);
        }
        mbarrier_wait_fn(&bar_KV, phase_KV);
        phase_KV ^= 1;
        
        __syncthreads();
    }
    
    for (int col = 0; col < 64; col += 4) {
        uint32_t r0, r1, r2, r3;
        uint32_t addr_o = o_tmem + tid * 65536 + col;
        tmem_load_4x_fn(addr_o, &r0, &r1, &r2, &r3);
        tmem_load_fence_fn();
        
        float o0 = __uint_as_float(r0) / smem->s_L[tid];
        float o1 = __uint_as_float(r1) / smem->s_L[tid];
        float o2 = __uint_as_float(r2) / smem->s_L[tid];
        float o3 = __uint_as_float(r3) / smem->s_L[tid];
        
        int swizzled_col = (col / 8 ^ (tid % 8)) * 8 + (col % 8);
        int idx = tid * 128 + swizzled_col;
        smem->s_O[idx] = __float2bfloat16(o0);
        smem->s_O[idx+1] = __float2bfloat16(o1);
        smem->s_O[idx+2] = __float2bfloat16(o2);
        smem->s_O[idx+3] = __float2bfloat16(o3);
    }
    __syncthreads();
    
    for (int i = tid; i < 128 * 128; i += 128) {
        int row = i / 128;
        int col = i % 128;
        int global_row = row_base + row;
        if (global_row < S && col < 128) {
            int swizzled_col = (col / 8 ^ (row % 8)) * 8 + (col % 8);
            int idx = row * 128 + swizzled_col;
            O[(uint64_t)bh * S * 128 + (uint64_t)global_row * 128 + col] = smem->s_O[idx];
        }
    }
    
    if (tid < 128) {
        int global_row = row_base + tid;
        if (global_row < S) {
            LSE[(uint64_t)bh * S + global_row] = smem->s_M[tid] + logf(smem->s_L[tid]);
        }
    }
    
    if (tid == 0) {
        tmem_dealloc_fn(s_tmem, 128);
        tmem_dealloc_fn(o_tmem, 128);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    uint32_t B = Q.size(0);
    uint32_t H = Q.size(1);
    uint32_t S = Q.size(2);
    uint32_t D = Q.size(3);
    
    CUtensorMap tma_Q, tma_K, tma_V;
    __nv_bfloat16* Q_ptr = static_cast<__nv_bfloat16*>(Q.data_ptr());
    __nv_bfloat16* K_ptr = static_cast<__nv_bfloat16*>(K.data_ptr());
    __nv_bfloat16* V_ptr = static_cast<__nv_bfloat16*>(V.data_ptr());
    
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_Q, Q_ptr, D, S, B * H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_K, K_ptr, D, S, B * H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_V, V_ptr, D, S, B * H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
    dim3 grid((S + 127) / 128, B * H);
    dim3 block(128);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUDA_CHECK(cudaFuncSetAttribute(run_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 140 * 1024));
    run_kernel<<<grid, block, 140 * 1024, stream>>>(tma_Q, tma_K, tma_V, static_cast<__nv_bfloat16*>(O.data_ptr()), static_cast<float*>(LSE.data_ptr()), S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_kernel