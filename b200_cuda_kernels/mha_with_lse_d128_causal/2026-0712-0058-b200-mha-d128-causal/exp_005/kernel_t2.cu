#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
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

namespace tvm_ffi_kernel {

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void named_barrier_sync_fn(int bar_id, int count) {
    asm volatile("barrier.sync.aligned %0, %1;" :: "r"(bar_id), "r"(count));
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(phase));
}

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
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

__device__ __forceinline__ void tmem_store_4x_fn(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
        :: "r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3));
}

__device__ __forceinline__ uint32_t pack_bf16_fn(float a, float b) {
    __nv_bfloat162x2_t res;
    res.x[0] = __float2bfloat16(a);
    res.x[1] = __float2bfloat16(b);
    return res.x[2];
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

__device__ __forceinline__ void write_swizzled_128B_32B_atomic_flip(__nv_bfloat16* smem, int row, int col, __nv_bfloat16 val) {
    int x_chunk = col / 8;
    int x_rem = col % 8;
    int y_chunk = row % 8;
    int swizzled_x_chunk = y_chunk ^ x_chunk;
    int new_col = swizzled_x_chunk * 8 + x_rem;
    
    if (((row / 8) % 2) == 1) {
        if (x_rem < 4) {
            new_col += 4;
        } else {
            new_col -= 4;
        }
    }
    smem[row * 128 + new_col] = val;
}

__device__ __forceinline__ __nv_bfloat16 read_swizzled_128B_32B_atomic_flip(const __nv_bfloat16* smem, int row, int col) {
    int x_chunk = col / 8;
    int x_rem = col % 8;
    int y_chunk = row % 8;
    int swizzled_x_chunk = y_chunk ^ x_chunk;
    int new_col = swizzled_x_chunk * 8 + x_rem;
    
    if (((row / 8) % 2) == 1) {
        if (x_rem < 4) {
            new_col += 4;
        } else {
            new_col -= 4;
        }
    }
    return smem[row * 128 + new_col];
}

__device__ __forceinline__ float __logf(float x) {
    float y;
    asm volatile("ln.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(bar)),
        "r"(c0), "r"(c1) : "memory");
}

struct SharedStorage {
    __align__(1024) __nv_bfloat16 Q[128 * 128];
    __align__(1024) __nv_bfloat16 K[128 * 128];
    __align__(1024) __nv_bfloat16 V[128 * 128];
    __align__(1024) __nv_bfloat16 P[128 * 128];
};

__global__ void __launch_bounds__(128) attention_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* __restrict__ O_gmem,
    float* __restrict__ LSE_gmem,
    int S_len) 
{
    extern __shared__ __align__(1024) uint8_t smem_dynamic[];
    SharedStorage& shared = *reinterpret_cast<SharedStorage*>(smem_dynamic);
    __nv_bfloat16* smem_Q = shared.Q;
    __nv_bfloat16* smem_K = shared.K;
    __nv_bfloat16* smem_V = shared.V;
    __nv_bfloat16* smem_P = shared.P;

    int batch_head_idx = blockIdx.y;
    int block_idx = blockIdx.x;
    int seq_start = block_idx * 128;
    if (seq_start >= S_len) return;

    __shared__ alignas(8) uint64_t smem_mbar_QK;
    __shared__ alignas(8) uint64_t smem_mbar_PV;
    __shared__ alignas(8) uint64_t bar_Q;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&smem_mbar_QK, 1);
        init_smem_barrier_fn(&smem_mbar_PV, 1);
        init_smem_barrier_fn(&bar_Q, 1);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();

    __shared__ uint32_t tmem_S_addr;
    __shared__ uint32_t tmem_O_addr;

    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_S_addr, 128);
        tmem_alloc_fn(&tmem_O_addr, 128);
        
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
            :: "r"((uint32_t)__cvta_generic_to_shared(&bar_Q)), "r"(16384));

        tma_load_2d_fn(&tma_Q, &bar_Q, smem_Q, 0, batch_head_idx * S_len + seq_start);
        tma_load_2d_fn(&tma_Q, &bar_Q, (char*)smem_Q + 8192, 64, batch_head_idx * S_len + seq_start);
    }
    
    int tid = threadIdx.x;
    const float scale = 1.0f / sqrtf(128.0f);
    const float LOG2E = 1.4426950408889634f;

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    uint32_t phase_PV = 0;
    uint32_t step = 0;

    uint32_t qt_base = (uint32_t)__cvta_generic_to_shared(smem_Q);
    uint32_t LBO_Q = 0;
    uint32_t SBO_Q = 1024;

    mbarrier_wait_fn(&bar_Q, 0);

    for (int j = 0; j <= block_idx * 128; j += 128) {
        __shared__ alignas(8) uint64_t bar_K;
        __shared__ alignas(8) uint64_t bar_V;
        
        if (threadIdx.x == 0) {
            init_smem_barrier_fn(&bar_K, 1);
            init_smem_barrier_fn(&bar_V, 1);
            
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                :: "r"((uint32_t)__cvta_generic_to_shared(&bar_K)), "r"(16384));
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                :: "r"((uint32_t)__cvta_generic_to_shared(&bar_V)), "r"(16384));

            tma_load_2d_fn(&tma_K, &bar_K, smem_K, 0, batch_head_idx * S_len + j);
            tma_load_2d_fn(&tma_K, &bar_K, (char*)smem_K + 8192, 64, batch_head_idx * S_len + j);
            
            tma_load_2d_fn(&tma_V, &bar_V, smem_V, 0, batch_head_idx * S_len + j);
            tma_load_2d_fn(&tma_V, &bar_V, (char*)smem_V + 8192, 64, batch_head_idx * S_len + j);
        }
        
        mbarrier_wait_fn(&bar_K, 0);
        mbarrier_wait_fn(&bar_V, 0);
        __syncthreads(); 
        
        if (threadIdx.x == 0) {
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                :: "r"((uint32_t)__cvta_generic_to_shared(&smem_mbar_QK)), "r"(16384));
            
            uint32_t kt_base = (uint32_t)__cvta_generic_to_shared(smem_K);
            uint32_t LBO_K = 0;
            uint32_t SBO_K = 1024;
            
            int accum = 0;
            for (int desc_k_step = 0; desc_k_step < 4; ++desc_k_step) {
                uint64_t desc_Q_curr = make_smem_desc_sm100_fn((void*)(qt_base + desc_k_step * 32), LBO_Q, SBO_Q);
                uint64_t desc_K_curr = make_smem_desc_sm100_fn((void*)(kt_base + desc_k_step * 32), LBO_K, SBO_K);
                
                asm volatile(
                    "{\n.reg .pred p;\n"
                    "setp.ne.b32 p, %4, 0;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                    :: "r"(tmem_S_addr), "l"(desc_Q_curr), "l"(desc_K_curr), "r"(make_instr_desc_fn_128x128()), "r"(accum));
                accum = 1;
            }

            asm volatile(
                "tcgen05.commit.cta_group::1"
                ".mbarrier::arrive::one.shared.b64"
                " [%0];"
                :: "r"((uint32_t)__cvta_generic_to_shared(&smem_mbar_QK)));
        }
        mbarrier_wait_fn(&smem_mbar_QK, step % 2);
        
        float rowmax = -INFINITY;
        for (int c = tid; c < 128; c += 128) {
            uint32_t addr = (tid << 16) + c;
            uint32_t r0;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 %0, [%1];" : "=r"(r0) : "r"(addr));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float s_val = __uint_as_float(r0);
            if (j + c > seq_start + tid) s_val = -INFINITY;
            rowmax = fmaxf(rowmax, s_val);
        }
        
        for (int i = 1; i < 128; i *= 2) rowmax = fmaxf(rowmax, __shfl_xor_sync(0xFFFFFFFF, rowmax, i));
        
        float new_max = fmaxf(running_max, rowmax);
        float alpha = fast_exp2f_fn((running_max - new_max) * LOG2E * scale);
        float beta = fast_exp2f_fn((rowmax - new_max) * LOG2E * scale);
        
        for (int c = tid; c < 128; c += 128) {
            uint32_t addr = (tid << 16) + c;
            uint32_t r0;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 %0, [%1];" : "=r"(r0) : "r"(addr));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float s_val = __uint_as_float(r0);
            
            float o_val = s_val; // dummy load, actual O is in tmem_O_addr
            uint32_t o_addr = (tid << 16) + c;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 %0, [%1];" : "=r"(o_val) : "r"(o_addr));
            
            o_val *= alpha;
            tmem_store_4x_fn(o_addr, o_val, 0, 0, 0);
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        }
        
        float rowsum = 0;
        for (int c = tid; c < 128; c += 128) {
            uint32_t addr = (tid << 16) + c;
            uint32_t r0;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 %0, [%1];" : "=r"(r0) : "r"(addr));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float s_val = __uint_as_float(r0);
            
            float val = 0;
            if (j + c > seq_start + tid) {
                val = 0;
            } else {
                val = fast_exp2f_fn((s_val - rowmax) * LOG2E * scale);
            }
            rowsum += val;
            write_swizzled_128B_32B_atomic_flip(smem_P, tid, c, __float2bfloat16(val));
        }
        
        for (int i = 1; i < 128; i *= 2) rowsum += __shfl_xor_sync(0xFFFFFFFF, rowsum, i);
        
        running_sum = running_sum * alpha + rowsum * beta;
        running_max = new_max;
        
        named_barrier_sync_fn(1, 128); 
        fence_async_shared_fn();
        
        if (threadIdx.x == 0) {
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                :: "r"((uint32_t)__cvta_generic_to_shared(&smem_mbar_PV)), "r"(16384));
            
            uint32_t pt_base = (uint32_t)__cvta_generic_to_shared(smem_P);
            uint32_t vt_base = (uint32_t)__cvta_generic_to_shared(smem_V);
            
            uint32_t SBO_P = 1024;
            uint32_t LBO_P = 16384;
            
            uint32_t SBO_V = 1024;
            uint32_t LBO_V = 16384;
            
            for (int desc_k_step = 0; desc_k_step < 4; ++desc_k_step) {
                uint64_t desc_P_curr = make_smem_desc_sm100_fn((void*)(pt_base + desc_k_step * 32), LBO_P, SBO_P);
                uint64_t desc_V_curr = make_smem_desc_sm100_fn((void*)(vt_base + desc_k_step * 2048), LBO_V, SBO_V);
                
                asm volatile(
                    "{\n.reg .pred p;\n"
                    "setp.ne.b32 p, %4, 0;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                    :: "r"(tmem_O_addr), "l"(desc_P_curr), "l"(desc_V_curr), "r"(make_instr_desc_fn_128x128_pv()), "r"(1));
            }

            asm volatile(
                "tcgen05.commit.cta_group::1"
                ".mbarrier::arrive::one.shared.b64"
                " [%0];"
                :: "r"((uint32_t)__cvta_generic_to_shared(&smem_mbar_PV)));
        }
        mbarrier_wait_fn(&smem_mbar_PV, phase_PV);
        phase_PV ^= 1;
        step++;
    }
    
    if (tid < 128) {
        float m_val = running_max;
        float l_val = running_sum;
        LSE_gmem[batch_head_idx * S_len + seq_start + tid] = m_val + __logf(l_val);
    }
    
    float inv_sum = 1.0f / running_sum;
    
    for (int c = 0; c < 128; c += 4) {
        uint32_t addr = (tid << 16) + c;
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(addr));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float f0 = __uint_as_float(r0);
        float f1 = __uint_as_float(r1);
        float f2 = __uint_as_float(r2);
        float f3 = __uint_as_float(r3);
        
        uint32_t packed0 = pack_bf16_fn(f0 * inv_sum, f1 * inv_sum);
        uint32_t packed1 = pack_bf16_fn(f2 * inv_sum, f3 * inv_sum);
        
        uint32_t* ptr = (uint32_t*)(O_gmem + batch_head_idx * S_len * 128 + (seq_start + tid) * 128 + c);
        *ptr = packed0;
        *(ptr + 2) = packed1;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_S_addr, 128);
        tmem_dealloc_fn(tmem_O_addr, 128);
    }
}

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapDataType dataType, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(d, dataType, 2, globalAddress, globalDim, globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill);
}

__device__ __forceinline__ uint32_t make_instr_desc_fn_128x128() {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= ((128 / 8) << 17);     
    d |= ((128 / 16) << 24);    
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn_128x128_pv() {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= (1u << 16);   
    d |= ((128 / 8) << 17);     
    d |= ((128 / 16) << 24);    
    return d;
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    
    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());
    
    CUtensorMap tma_Q, tma_K, tma_V;
    create_tma_2d_descriptor_2B(&tma_Q, (void*)Q_data, 128, B * H * S, 64, 128, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_NONE, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        
    create_tma_2d_descriptor_2B(&tma_K, (void*)K_data, 128, B * H * S, 64, 128, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_NONE, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        
    create_tma_2d_descriptor_2B(&tma_V, (void*)V_data, 128, B * H * S, 64, 128, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_NONE, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    dim3 grid((S + 127) / 128, B * H);
    int threads = 128;
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 196 * 1024));
    attention_kernel<<<grid, threads, 196 * 1024, stream>>>(tma_Q, tma_K, tma_V, O_data, LSE_data, S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_kernel