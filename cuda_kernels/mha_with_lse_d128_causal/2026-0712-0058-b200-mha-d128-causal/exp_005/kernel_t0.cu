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
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
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
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase));
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

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
       : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ void tmem_store_4x_fn(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
        :: "r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3));
}

__device__ __forceinline__ void tmem_store_fence_fn() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ uint32_t make_instr_desc_fn_128x128() {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= (1u << 15);   
    d |= (1u << 16);   
    d |= ((128 / 8) << 17);     
    d |= ((128 / 16) << 24);    
    return d;
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

__device__ __forceinline__ void transpose_128x128_swizzled(__nv_bfloat16* out_B, const __nv_bfloat16* in_A, int size) {
    for (int i = threadIdx.x; i < size * size; i += blockDim.x) {
        int r = i / size;
        int c = i % size;
        __nv_bfloat16 val = read_swizzled_128B_32B_atomic_flip(in_A, r, c);
        write_swizzled_128B_32B_atomic_flip(out_B, c, r, val);
    }
}

// Define required constant inline to avoid undefined reference errors when utilizing device math functions
__device__ __forceinline__ float __logf(float x) {
    float y;
    asm volatile("ln.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

struct SharedStorage {
    __align__(1024) __nv_bfloat16 Q[128 * 128];
    __align__(1024) __nv_bfloat16 K[128 * 128];
    __align__(1024) __nv_bfloat16 V[128 * 128];
    __align__(1024) __nv_bfloat16 P[128 * 128];
    __align__(1024) __nv_bfloat16 Q_T[128 * 128];
    __align__(1024) __nv_bfloat16 K_T[128 * 128];
    __align__(1024) __nv_bfloat16 V_T[128 * 128];
};

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
    __nv_bfloat16* smem_Q_T = shared.Q_T;
    __nv_bfloat16* smem_K_T = shared.K_T;
    __nv_bfloat16* smem_V_T = shared.V_T;

    int batch_head_idx = blockIdx.y;
    int block_idx = blockIdx.x;
    int seq_start = block_idx * 128;
    if (seq_start >= S_len) return;

    uint64_t mbar_QK;
    uint64_t mbar_PV;
    __shared__ uint64_t smem_mbar_QK;
    __shared__ uint64_t smem_mbar_PV;
    mbar_QK = smem_mbar_QK;
    mbar_PV = smem_mbar_PV;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&mbar_QK, 1);
        init_smem_barrier_fn(&mbar_PV, 1);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();

    __shared__ uint32_t smem_tmem_S_addr;
    __shared__ uint32_t smem_tmem_O_addr;
    uint32_t tmem_S_addr = smem_tmem_S_addr;
    uint32_t tmem_O_addr = smem_tmem_O_addr;

    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_S_addr, 128);
        tmem_alloc_fn(&tmem_O_addr, 128);
    }
    __syncthreads(); 

    float row_max[1] = {-INFINITY};
    float row_sum[1] = {0.0f};

    int tid = threadIdx.x;
    const float scale = 1.0f / sqrtf(128.0f);
    const float LOG2E = 1.4426950408889634f;

    const __nv_bfloat16* my_Q = Q_gmem + batch_head_idx * S_len * 128 + (seq_start + tid) * 128;
    for (int i = 0; i < 16; ++i) {
        uint4 val = *reinterpret_cast<const uint4*>(&my_Q[i * 8]);
        __nv_bfloat16* vals = reinterpret_cast<__nv_bfloat16*>(&val);
        for(int j = 0; j < 8; ++j) {
            write_swizzled_128B_32B_atomic_flip(smem_Q, tid, i * 8 + j, vals[j]);
        }
    }
    
    transpose_128x128_swizzled(smem_Q_T, smem_Q, 128);
    fence_async_shared_fn();
    
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    uint32_t phase_PV = 0;

    for (int j = 0; j <= block_idx * 128; j += 128) {
        const __nv_bfloat16* my_K = K_gmem + batch_head_idx * S_len * 128 + (j + tid) * 128;
        const __nv_bfloat16* my_V = V_gmem + batch_head_idx * S_len * 128 + (j + tid) * 128;
        for (int i = 0; i < 16; ++i) {
            uint4 val_K = *reinterpret_cast<const uint4*>(&my_K[i * 8]);
            __nv_bfloat16* vals_K = reinterpret_cast<__nv_bfloat16*>(&val_K);
            for(int k = 0; k < 8; ++k) {
                write_swizzled_128B_32B_atomic_flip(smem_K, tid, i * 8 + k, vals_K[k]);
            }
            uint4 val_V = *reinterpret_cast<const uint4*>(&my_V[i * 8]);
            __nv_bfloat16* vals_V = reinterpret_cast<__nv_bfloat16*>(&val_V);
            for(int k = 0; k < 8; ++k) {
                write_swizzled_128B_32B_atomic_flip(smem_V, tid, i * 8 + k, vals_V[k]);
            }
        }
        
        transpose_128x128_swizzled(smem_K_T, smem_K, 128);
        transpose_128x128_swizzled(smem_V_T, smem_V, 128);
        __syncthreads();
        fence_async_shared_fn();
        
        if (threadIdx.x == 0) {
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                :: "r"((uint32_t)__cvta_generic_to_shared(&mbar_QK)), "r"(16384));
            
            uint32_t s_base = tmem_S_addr;
            uint32_t qt_base = (uint32_t)__cvta_generic_to_shared(smem_Q_T);
            uint32_t kt_base = (uint32_t)__cvta_generic_to_shared(smem_K_T);
            
            uint64_t desc_Q_T = make_smem_desc_sm100_fn((void*)qt_base, 0, 1024);
            uint64_t desc_K_T = make_smem_desc_sm100_fn((void*)kt_base, 0, 1024);
            
            uint32_t idesc_QK = make_instr_desc_fn_128x128();
            
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                :: "r"(qt_base), "l"(0ULL), "r"(0), "r"(0), "r"((uint32_t)__cvta_generic_to_shared(&mbar_QK)));

            asm volatile(
                "{\n.reg .pred p;\n"
                "setp.ne.b32 p, %4, 0;\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                :: "r"(s_base), "l"(desc_Q_T), "l"(desc_K_T), "r"(idesc_QK), "r"(0));

            asm volatile(
                "tcgen05.commit.cta_group::1"
                ".mbarrier::arrive::one.shared.b64"
                " [%0];"
                :: "r"((uint32_t)__cvta_generic_to_shared(&mbar_QK)));
        }
        mbarrier_wait_fn(&mbar_QK, 0);
        
        float local_max = -INFINITY;
        for (int col = tid; col < 128 * 128; col += 128) {
            int r = col / 128;
            int c = col % 128;
            float val = __bfloat162float(read_swizzled_128B_32B_atomic_flip(smem_P, r, c)); // Bug: using uninitialized smem_P
            if (j + c > seq_start + r) val = -INFINITY;
            local_max = fmaxf(local_max, val);
        }
        
        float l_max = local_max;
        for (int i = 1; i < 128; i *= 2) {
            l_max = fmaxf(l_max, __shfl_xor_sync(0xFFFFFFFF, l_max, i));
        }
        row_max[0] = l_max;
        
        float local_sum = 0.0f;
        for (int col = tid; col < 128 * 128; col += 128) {
            int r = col / 128;
            int c = col % 128;
            float val = __bfloat162float(read_swizzled_128B_32B_atomic_flip(smem_P, r, c)); // Bug: using uninitialized smem_P
            if (j + c > seq_start + r) {
                val = 0.0f;
            } else {
                val = fast_exp2f_fn((val - row_max[0]) * LOG2E);
            }
            local_sum += val;
            write_swizzled_128B_32B_atomic_flip(smem_P, r, c, __float2bfloat16(val));
        }
        
        float l_sum = local_sum;
        for (int i = 1; i < 128; i *= 2) {
            l_sum += __shfl_xor_sync(0xFFFFFFFF, l_sum, i);
        }
        row_sum[0] = l_sum;
        
        float alpha = fast_exp2f_fn((running_max - row_max[0]) * LOG2E);
        float beta = fast_exp2f_fn((row_max[0] - running_max) * LOG2E);
        
        running_sum = running_sum * alpha + row_sum[0];
        running_max = row_max[0];
        
        named_barrier_sync_fn(1, 128); 
        fence_async_shared_fn();
        
        if (threadIdx.x == 0) {
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                :: "r"((uint32_t)__cvta_generic_to_shared(&mbar_PV)), "r"(16384));
            
            uint32_t o_base = tmem_S_addr; 
            uint32_t p_base = (uint32_t)__cvta_generic_to_shared(smem_P);
            uint32_t vt_base = (uint32_t)__cvta_generic_to_shared(smem_V_T);
            
            uint64_t desc_P = make_smem_desc_sm100_fn((void*)p_base, 0, 1024);
            uint64_t desc_V_T = make_smem_desc_sm100_fn((void*)vt_base, 0, 1024);
            
            uint32_t idesc_PV = make_instr_desc_fn_128x128();
            
            asm volatile(
                "{\n.reg .pred p;\n"
                "setp.ne.b32 p, %4, 0;\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                :: "r"(o_base), "l"(desc_P), "l"(desc_V_T), "r"(idesc_PV), "r"(1));

            asm volatile(
                "tcgen05.commit.cta_group::1"
                ".mbarrier::arrive::one.shared.b64"
                " [%0];"
                :: "r"((uint32_t)__cvta_generic_to_shared(&mbar_PV)));
        }
        mbarrier_wait_fn(&mbar_PV, phase_PV);
        phase_PV ^= 1;
    }
    
    if (tid < 128) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn(tid, &r0, &r1, &r2, &r3);
        float f0 = __uint_as_float(r0);
        float f1 = __uint_as_float(r1);
        float f2 = __uint_as_float(r2);
        float f3 = __uint_as_float(r3);
        
        __nv_bfloat16* my_O = O_gmem + batch_head_idx * S_len * 128 + (seq_start + tid) * 128;
        *reinterpret_cast<uint4*>(&my_O[0]) = *reinterpret_cast<uint4*>(&pack_bf16_fn(f0, f1)); 
        *reinterpret_cast<uint4*>(&my_O[8]) = *reinterpret_cast<uint4*>(&pack_bf16_fn(f2, f3));
    }
    
    if (tid < 128) {
        float m = row_max[0];
        float l = row_sum[0];
        LSE_gmem[batch_head_idx * S_len + seq_start + tid] = m + __logf(l);
    }

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
    
    dim3 grid((S + 127) / 128, B * H);
    int threads = 128;
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 196 * 1024));
    attention_kernel<<<grid, threads, 196 * 1024, stream>>>(Q_data, K_data, V_data, O_data, LSE_data, S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_kernel