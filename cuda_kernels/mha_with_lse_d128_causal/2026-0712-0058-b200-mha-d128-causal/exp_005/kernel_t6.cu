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

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void umma_commit_2sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile(
        "tcgen05.commit.cta_group::2"
        ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;"
        :: "r"(a), "h"((uint16_t)0x3));
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

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t addr, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
       : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(addr));
}

__device__ __forceinline__ void tmem_store_4x_fn(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
        :: "r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_store_fence_fn() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
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

struct SharedStorage {
    __align__(1024) __nv_bfloat16 Q[128 * 128];
    __align__(1024) __nv_bfloat16 K[128 * 128];
    __align__(1024) __nv_bfloat16 V[128 * 128];
    __align__(1024) __nv_bfloat16 P[128 * 128];
};

__device__ __forceinline__ uint32_t make_instr_desc_fn_128x128() {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= ((128 / 8) << 17);     
    d |= ((128 / 16) << 24);    
    return d;
}

__global__ void __launch_bounds__(128) attention_kernel(
    const __nv_bfloat16* __restrict__ Q_gmem,
    const __nv_bfloat16* __restrict__ K_gmem,
    const __nv_bfloat16* __restrict__ V_gmem,
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

    __shared__ alignas(8) uint64_t smem_mbar_QK;
    __shared__ alignas(8) uint64_t smem_mbar_PV;

    __shared__ uint32_t tmem_S_addr;
    __shared__ uint32_t tmem_O_addr;

    // Ensure both CTAs participate in TMEM allocation to prevent hanging on cta_group::2 instructions
    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_S_addr, 128);
        tmem_alloc_fn(&tmem_O_addr, 128);
        
        init_smem_barrier_fn(&smem_mbar_QK, 1);
        init_smem_barrier_fn(&smem_mbar_PV, 1);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();

    if (seq_start >= S_len) return;

    int tid = threadIdx.x;
    const float scale = 1.0f / sqrtf(128.0f);
    const float LOG2E = 1.4426950408889634f;

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    uint32_t phase_PV = 0;
    uint32_t step = 0;

    for (int i = 0; i < 16; ++i) {
        ulonglong2 val = {0, 0};
        if (seq_start + tid < S_len) {
            val = *reinterpret_cast<const ulonglong2*>(&Q_gmem[batch_head_idx * S_len * 128 + (seq_start + tid) * 128 + i * 16]);
        }
        uint32_t offset = tid * 128 + (((i * 16) / 8) ^ (tid % 8)) * 8;
        *reinterpret_cast<ulonglong2*>(&smem_Q[offset]) = val;
    }
    
    uint32_t qt_base = (uint32_t)__cvta_generic_to_shared(smem_Q);
    uint32_t LBO_Q = 1024;
    uint32_t SBO_Q = 1024;

    for (int j = 0; j <= block_idx * 128; j += 128) {
        if (tid < 128) {
            for (int i = 0; i < 16; ++i) {
                ulonglong2 k_val = {0, 0};
                ulonglong2 v_val = {0, 0};
                if (j + tid < S_len) {
                    k_val = *reinterpret_cast<const ulonglong2*>(&K_gmem[batch_head_idx * S_len * 128 + (j + tid) * 128 + i * 16]);
                    v_val = *reinterpret_cast<const ulonglong2*>(&V_gmem[batch_head_idx * S_len * 128 + (j + tid) * 128 + i * 16]);
                }
                uint32_t offset = tid * 128 + (((i * 16) / 8) ^ (tid % 8)) * 8;
                *reinterpret_cast<ulonglong2*>(&smem_K[offset]) = k_val;
                *reinterpret_cast<ulonglong2*>(&smem_V[offset]) = v_val;
            }
        }
        
        __syncthreads(); 
        
        if (threadIdx.x == 0) {
            uint32_t kt_base = (uint32_t)__cvta_generic_to_shared(smem_K);
            uint32_t LBO_KT = 1024;
            uint32_t SBO_KT = 1024;
            
            int accum = 0;
            for (int k = 0; k < 8; ++k) {
                uint64_t desc_Q_curr = make_smem_desc_sm100_fn((void*)(qt_base + k * 32), LBO_Q, SBO_Q);
                uint64_t desc_KT_curr = make_smem_desc_sm100_fn((void*)(kt_base + k * 32), LBO_KT, SBO_KT);
                
                asm volatile(
                    "{\n.reg .pred p;\n"
                    "setp.ne.b32 p, %4, 0;\n"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                    :: "r"(tmem_S_addr), "l"(desc_Q_curr), "l"(desc_KT_curr), "r"(make_instr_desc_fn_128x128()), "r"(accum));
                accum = 1;
            }

            umma_commit_2sm_fn(&smem_mbar_QK);
        }
        mbarrier_wait_fn(&smem_mbar_QK, step % 2);
        
        float rowmax = -INFINITY;
        for (int c = 0; c < 128; c += 4) {
            uint32_t addr = (tid << 16) + c;
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(addr, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            float s0 = __uint_as_float(r0) * scale;
            float s1 = __uint_as_float(r1) * scale;
            float s2 = __uint_as_float(r2) * scale;
            float s3 = __uint_as_float(r3) * scale;
            
            if (j + c > seq_start + tid) s0 = -INFINITY;
            if (j + c + 1 > seq_start + tid) s1 = -INFINITY;
            if (j + c + 2 > seq_start + tid) s2 = -INFINITY;
            if (j + c + 3 > seq_start + tid) s3 = -INFINITY;
            
            rowmax = fmaxf(rowmax, fmaxf(fmaxf(s0, s1), fmaxf(s2, s3)));
        }
        
        float new_max = fmaxf(running_max, rowmax);
        float alpha = fast_exp2f_fn((running_max - new_max) * LOG2E);
        
        for (int c = 0; c < 128; c += 4) {
            uint32_t addr = (tid << 16) + c;
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(addr, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            
            float o0 = __uint_as_float(r0) * alpha;
            float o1 = __uint_as_float(r1) * alpha;
            float o2 = __uint_as_float(r2) * alpha;
            float o3 = __uint_as_float(r3) * alpha;
            
            tmem_store_4x_fn(addr, __float_as_uint(o0), __float_as_uint(o1), __float_as_uint(o2), __float_as_uint(o3));
            tmem_store_fence_fn();
        }
        
        float rowsum = 0;
        for (int c = 0; c < 128; c += 4) {
            uint32_t addr = (tid << 16) + c;
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(addr, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            
            float s0 = __uint_as_float(r0) * scale;
            float s1 = __uint_as_float(r1) * scale;
            float s2 = __uint_as_float(r2) * scale;
            float s3 = __uint_as_float(r3) * scale;
            
            float p0 = (j + c > seq_start + tid) ? 0.0f : fast_exp2f_fn((s0 - new_max) * LOG2E);
            float p1 = (j + c + 1 > seq_start + tid) ? 0.0f : fast_exp2f_fn((s1 - new_max) * LOG2E);
            float p2 = (j + c + 2 > seq_start + tid) ? 0.0f : fast_exp2f_fn((s2 - new_max) * LOG2E);
            float p3 = (j + c + 3 > seq_start + tid) ? 0.0f : fast_exp2f_fn((s3 - new_max) * LOG2E);
            
            rowsum += p0 + p1 + p2 + p3;
            
            uint16_t pb0 = *(uint16_t*)(&__float2bfloat16(p0));
            uint16_t pb1 = *(uint16_t*)(&__float2bfloat16(p1));
            uint16_t pb2 = *(uint16_t*)(&__float2bfloat16(p2));
            uint16_t pb3 = *(uint16_t*)(&__float2bfloat16(p3));
            
            uint4 packed_p;
            ((uint16_t*)&packed_p)[0] = pb0;
            ((uint16_t*)&packed_p)[1] = pb1;
            ((uint16_t*)&packed_p)[2] = pb2;
            ((uint16_t*)&packed_p)[3] = pb3;
            
            uint32_t offset = tid * 128 + (((c / 8) ^ (tid % 8)) * 8);
            *(uint4*)&smem_P[offset] = packed_p;
        }
        
        float beta = fast_exp2f_fn((rowmax - new_max) * LOG2E);
        running_sum = running_sum * alpha + rowsum * beta;
        running_max = new_max;
        
        named_barrier_sync_fn(1, 128); 
        fence_async_shared_fn();
        
        if (threadIdx.x == 0) {
            uint32_t pt_base = (uint32_t)__cvta_generic_to_shared(smem_P);
            uint32_t vt_base = (uint32_t)__cvta_generic_to_shared(smem_V);
            
            uint32_t LBO_P = 1024;
            uint32_t SBO_P = 1024;
            
            uint32_t LBO_V = 1024;
            uint32_t SBO_V = 1024;
            
            for (int k = 0; k < 8; ++k) {
                uint64_t desc_P_curr = make_smem_desc_sm100_fn((void*)(pt_base + k * 32), LBO_P, SBO_P);
                uint64_t desc_V_curr = make_smem_desc_sm100_fn((void*)(vt_base + k * 32), LBO_V, SBO_V);
                
                asm volatile(
                    "{\n.reg .pred p;\n"
                    "setp.ne.b32 p, %4, 0;\n"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                    :: "r"(tmem_O_addr), "l"(desc_P_curr), "l"(desc_V_curr), "r"(make_instr_desc_fn_128x128()), "r"(1));
            }

            umma_commit_2sm_fn(&smem_mbar_PV);
        }
        mbarrier_wait_fn(&smem_mbar_PV, phase_PV);
        phase_PV ^= 1;
        step++;
    }
    
    if (tid < 128) {
        LSE_gmem[batch_head_idx * S_len + seq_start + tid] = running_max + __logf(running_sum);
    }
    
    float inv_sum = 1.0f / running_sum;
    
    for (int c = 0; c < 128; c += 4) {
        uint32_t addr = (tid << 16) + c;
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn(addr, &r0, &r1, &r2, &r3);
        tmem_load_fence_fn();
        
        float f0 = __uint_as_float(r0);
        float f1 = __uint_as_float(r1);
        float f2 = __uint_as_float(r2);
        float f3 = __uint_as_float(r3);
        
        __nv_bfloat16* ptr = O_gmem + batch_head_idx * S_len * 128 + (seq_start + tid) * 128 + c;
        if (seq_start + tid < S_len && c + 3 < 128) {
            ptr[0] = __float2bfloat16(f0 * inv_sum);
            ptr[1] = __float2bfloat16(f1 * inv_sum);
            ptr[2] = __float2bfloat16(f2 * inv_sum);
            ptr[3] = __float2bfloat16(f3 * inv_sum);
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_S_addr, 128);
        tmem_dealloc_fn(tmem_O_addr, 128);
    }
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
    
    uint32_t num_blocks = (S + 127) / 128;
    uint32_t grid_x = (num_blocks + 1) & ~1; 
    dim3 grid(grid_x, B * H);
    int threads = 128;
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = dim3(threads, 1, 1);
    config.dynamicSmemBytes = 196 * 1024;
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, attention_kernel, 
        Q_data, K_data, V_data, O_data, LSE_data, S));
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_kernel