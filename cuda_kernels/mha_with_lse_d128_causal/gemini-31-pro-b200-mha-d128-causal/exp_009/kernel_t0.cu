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

namespace tvm_ffi_mha {

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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

__device__ __forceinline__ void tmem_alloc_cg1_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols) : "memory");
}

__device__ __forceinline__ void tmem_dealloc_cg1_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols) : "memory");
}

__device__ __forceinline__ void tmem_load_8x_fn(uint32_t col,
    uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3,
    uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3),
     "=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7) : "r"(col));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
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

__device__ __forceinline__ void tcgen05_commit_cg1_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(a) : "memory");
}

__device__ __forceinline__ uint64_t my_make_smem_desc(void* smem_ptr, uint32_t lbo, uint32_t sbo, int swizzle) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    if (swizzle != 0) {
        uint32_t base_offset = (addr >> 7) & 0x7;
        d |= (uint64_t)base_offset << 49;
    }
    d |= (uint64_t)swizzle << 61;
    return d;
}

__device__ __forceinline__ void load_gmem_to_smem_swizzled(
    const __nv_bfloat16* gmem, uint32_t* smem, 
    int seq_idx, int S, int base_offset) {
    const uint32_t* g_ptr = (const uint32_t*)(gmem + base_offset);
    for(int i = threadIdx.x; i < 8192; i += 128) {
        int r = i / 64;
        int c = i % 64;
        int g_seq = seq_idx + r;
        uint32_t val = 0;
        if (g_seq < S) val = g_ptr[g_seq * 64 + c];
        int x = c / 4;
        int sx = (r % 8) ^ x;
        smem[r * 64 + sx * 4 + (c % 4)] = val;
    }
}


__global__ void mha_fwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S, int H) 
{
    setmaxnreg_inc_sync_fn<256>();
    
    int tid = threadIdx.x;
    int b = blockIdx.y / H;
    int h = blockIdx.y % H;
    int q_idx = blockIdx.x * 128;
    if (q_idx >= S) return;
    
    extern __shared__ __align__(128) uint8_t smem_pool[];
    uint16_t* smem_Q = (uint16_t*)(smem_pool);                   // 32KB
    uint16_t* smem_K = (uint16_t*)(smem_pool + 32768);           // 32KB
    uint16_t* smem_V = (uint16_t*)(smem_pool + 65536);           // 32KB
    uint16_t* smem_P = (uint16_t*)(smem_pool + 98304);           // 32KB
    uint64_t* mbar_umma = (uint64_t*)(smem_pool + 131072);       // 8B
    
    if (tid == 0) {
        init_smem_barrier_fn(mbar_umma, 1);
    }
    fence_smem_barrier_init_fn();
    
    __shared__ uint32_t tmem_addr;
    if (tid < 32) tmem_alloc_cg1_fn(&tmem_addr, 128);
    __syncthreads();
    uint32_t tmem_base = tmem_addr;
    
    int base_offset = (b * H + h) * S * 128;
    load_gmem_to_smem_swizzled(Q, (uint32_t*)smem_Q, q_idx, S, base_offset);
    __syncthreads();
    
    float O_acc[128];
    for (int i = 0; i < 128; i++) O_acc[i] = 0.0f;
    float m_val = -1e20f;
    float l_val = 0.0f;
    int phase_umma = 0;
    
    uint32_t idesc_QK = (1u << 4) | (1u << 7) | (1u << 10) | (0u << 15) | (0u << 16) | (16u << 17) | (8u << 24);
    uint32_t idesc_PV = (1u << 4) | (1u << 7) | (1u << 10) | (0u << 15) | (1u << 16) | (16u << 17) | (8u << 24);

    for (int k_idx = 0; k_idx <= q_idx; k_idx += 128) {
        load_gmem_to_smem_swizzled(K, (uint32_t*)smem_K, k_idx, S, base_offset);
        load_gmem_to_smem_swizzled(V, (uint32_t*)smem_V, k_idx, S, base_offset);
        __syncthreads();
        
        // Q * K^T
        if (tid == 0) {
            for(int k = 0; k < 128; k += 16) {
                uint64_t desc_Q = my_make_smem_desc(smem_Q + k, 1, 1024, 2);
                uint64_t desc_K = my_make_smem_desc(smem_K + k, 1, 1024, 2);
                uint32_t accum = (k > 0) ? 1 : 0;
                umma_f16_cg1_fn(tmem_base, desc_Q, desc_K, idesc_QK, accum);
            }
            tcgen05_commit_cg1_fn(mbar_umma);
        }
        mbarrier_wait_fn(mbar_umma, phase_umma);
        phase_umma ^= 1;
        
        float S_row[128];
        for(int c = 0; c < 128; c += 8) {
            tmem_load_8x_fn(tmem_base + c, 
                (uint32_t*)&S_row[c+0], (uint32_t*)&S_row[c+1], (uint32_t*)&S_row[c+2], (uint32_t*)&S_row[c+3],
                (uint32_t*)&S_row[c+4], (uint32_t*)&S_row[c+5], (uint32_t*)&S_row[c+6], (uint32_t*)&S_row[c+7]);
        }
        tmem_load_fence_fn();
        
        float row_max = -1e20f;
        int global_q = q_idx + tid;
        for(int c = 0; c < 128; c++) {
            int global_k = k_idx + c;
            if (global_q < S && global_k < S) {
                if (global_k > global_q) {
                    S_row[c] = -1e20f;
                } else {
                    S_row[c] *= 0.08838834764f; // 1.0 / sqrt(128.0)
                    if (S_row[c] > row_max) row_max = S_row[c];
                }
            } else {
                S_row[c] = -1e20f;
            }
        }
        
        float m_new = max(m_val, row_max);
        float scale = 0.0f;
        if (m_val != -1e20f) {
            scale = fast_exp2f_fn((m_val - m_new) * 1.44269504089f);
        }
        for(int i = 0; i < 128; i++) O_acc[i] *= scale;
        
        float row_sum = 0.0f;
        for(int c = 0; c < 128; c++) {
            float p = 0.0f;
            if (S_row[c] != -1e20f) {
                p = fast_exp2f_fn((S_row[c] - m_new) * 1.44269504089f);
            }
            S_row[c] = p;
            row_sum += p;
        }
        l_val = l_val * scale + row_sum;
        m_val = m_new;
        
        for(int c = 0; c < 128; c += 8) {
            int x = c / 8;
            int y = tid;
            int sx = (y % 8) ^ x;
            uint32_t* p_chunk = &((uint32_t*)smem_P)[y * 16 + sx * 4];
            
            p_chunk[0] = pack_bf16_fn(*(uint32_t*)&S_row[c+0], *(uint32_t*)&S_row[c+1]);
            p_chunk[1] = pack_bf16_fn(*(uint32_t*)&S_row[c+2], *(uint32_t*)&S_row[c+3]);
            p_chunk[2] = pack_bf16_fn(*(uint32_t*)&S_row[c+4], *(uint32_t*)&S_row[c+5]);
            p_chunk[3] = pack_bf16_fn(*(uint32_t*)&S_row[c+6], *(uint32_t*)&S_row[c+7]);
        }
        
        __syncthreads();
        
        // P * V
        if (tid == 0) {
            for(int k = 0; k < 128; k += 16) {
                uint64_t desc_P = my_make_smem_desc(smem_P + k, 1, 1024, 2);
                uint64_t desc_V = my_make_smem_desc(smem_V + k * 128, 16384, 1024, 2);
                uint32_t accum = (k > 0) ? 1 : 0;
                umma_f16_cg1_fn(tmem_base, desc_P, desc_V, idesc_PV, accum);
            }
            tcgen05_commit_cg1_fn(mbar_umma);
        }
        mbarrier_wait_fn(mbar_umma, phase_umma);
        phase_umma ^= 1;
        
        float pv_vals[128];
        for(int c = 0; c < 128; c += 8) {
            tmem_load_8x_fn(tmem_base + c, 
                (uint32_t*)&pv_vals[c+0], (uint32_t*)&pv_vals[c+1], (uint32_t*)&pv_vals[c+2], (uint32_t*)&pv_vals[c+3],
                (uint32_t*)&pv_vals[c+4], (uint32_t*)&pv_vals[c+5], (uint32_t*)&pv_vals[c+6], (uint32_t*)&pv_vals[c+7]);
        }
        tmem_load_fence_fn();
        for(int c = 0; c < 128; c++) O_acc[c] += pv_vals[c];
        
        __syncthreads();
    }
    
    // Store O and LSE
    for(int c = 0; c < 128; c += 2) {
        float o0 = O_acc[c] / l_val;
        float o1 = O_acc[c+1] / l_val;
        ((uint32_t*)smem_P)[tid * 64 + c/2] = pack_bf16_fn(*(uint32_t*)&o0, *(uint32_t*)&o1);
    }
    __syncthreads();
    
    uint4* smem_P_vec = (uint4*)smem_P;
    uint4* O_vec = (uint4*)O;
    for (int i = tid; i < 2048; i += 128) {
        int r = i / 16;
        int gq = q_idx + r;
        if (gq < S) {
            O_vec[((b * H + h) * S + gq) * 16 + (i % 16)] = smem_P_vec[i];
        }
    }
    
    int global_q = q_idx + tid;
    if (global_q < S) {
        float lse = m_val + logf(l_val);
        LSE[b * H * S + h * S + global_q] = lse;
    }
    
    if (tid < 32) tmem_dealloc_cg1_fn(tmem_base, 128);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);

    const __nv_bfloat16* q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* k_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* v_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_ptr = static_cast<float*>(LSE.data_ptr());

    int64_t threads = 128;
    dim3 blocks((S + 127) / 128, B * H);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    cudaFuncSetAttribute(mha_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 131072 + 1024);

    mha_fwd_kernel<<<blocks, threads, 131072 + 1024, stream>>>(q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr, S, H);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}  // namespace tvm_ffi_mha