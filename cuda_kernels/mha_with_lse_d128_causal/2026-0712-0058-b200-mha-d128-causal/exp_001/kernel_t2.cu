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

namespace causal_attention {

constexpr int H = 48;

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
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tmem_alloc_fn_cg1(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
       :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn_cg1(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
       :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
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

__device__ __forceinline__ void umma_commit_1sm_fn_cg1(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1"
        ".mbarrier::arrive::one.shared.b64"
        " [%0];"
        :: "r"(a));
}

__device__ __forceinline__ uint64_t make_smem_desc_cg1(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)0 << 61;   
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn_cg1(uint32_t M, uint32_t N) {
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

__device__ __forceinline__ uint32_t make_instr_desc_fn_cg1_mnmaj(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (0u << 15);   
    d |= (1u << 16);   
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ void shift_tmem_ptr(uint64_t& tmem_desc, int delta_elements, bool is_k_major) {
    uint32_t addr = (tmem_desc & 0x3FFF) << 4;
    addr += delta_elements * (is_k_major ? 2 : 32);
    tmem_desc &= ~0x3FFF;
    tmem_desc |= (addr & 0x3FFFF) >> 4;
}

__device__ __forceinline__ void load_tile(__nv_bfloat16* smem, const __nv_bfloat16* gmem, int b, int h, int s_offset, int S) {
    int tid = threadIdx.x;
    int idx = s_offset + tid;
    uint4* smem_u4 = (uint4*)smem;
    const uint4* gmem_u4 = (const uint4*)gmem;
    for (int i = 0; i < 16; i++) {
        uint4 val = {0, 0, 0, 0};
        if (idx < S) {
            val = gmem_u4[((b * H + h) * S + idx) * 16 + i];
        }
        smem_u4[idx * 16 + i] = val;
    }
}

__global__ void causal_attention_kernel(
    const __nv_bfloat16* Q_gmem, const __nv_bfloat16* K_gmem, const __nv_bfloat16* V_gmem,
    __nv_bfloat16* O_gmem, float* LSE_gmem, int S)
{
    setmaxnreg_inc_sync_fn<256>();

    extern __shared__ __align__(128) char smem[];
    __nv_bfloat16* smem_q = (__nv_bfloat16*)smem;                   // 32 KB
    __nv_bfloat16* smem_k = (__nv_bfloat16*)(smem + 32768);         // 32 KB
    __nv_bfloat16* smem_v = (__nv_bfloat16*)(smem + 65536);         // 32 KB
    __nv_bfloat16* smem_p = (__nv_bfloat16*)(smem + 98304);         // 32 KB
    uint64_t* bar = (uint64_t*)(smem + 131072);                     // 8 Bytes
    
    uint32_t tmem_S_addr, tmem_P_addr, tmem_O_addr;
    if (threadIdx.x == 0) {
        tmem_alloc_fn_cg1(&tmem_S_addr, 128);
        tmem_alloc_fn_cg1(&tmem_P_addr, 128);
        tmem_alloc_fn_cg1(&tmem_O_addr, 128);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar, 1);
    }
    __syncthreads();

    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float scale = 1.0f / sqrtf(128.0f);
    int phase_bar = 0;
    
    int b = blockIdx.x / H;
    int h = blockIdx.x % H;
    int s_offset_q = threadIdx.x;
    
    load_tile(smem_q, Q_gmem, b, h, s_offset_q, S);
    __syncthreads();

    for (int s_offset_k = 0; s_offset_k < S; s_offset_k += 128) {
        load_tile(smem_k, K_gmem, b, h, s_offset_k, S);
        load_tile(smem_v, V_gmem, b, h, s_offset_k, S);
        __syncthreads();
        
        uint32_t tx_bytes_q = (128 * 64 + 128 * 64) * sizeof(__nv_bfloat16);
        fence_proxy_async_fn();
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar, tx_bytes_q);
            uint64_t desc_S = make_smem_desc_cg1(smem_s, 2048, 128);
            uint32_t idesc_qkt = make_instr_desc_fn_cg1(128, 128);
            
            for (int k = 0; k < 4; ++k) {
                uint64_t desc_q = make_smem_desc_cg1(smem_q + k * 16);
                uint64_t desc_k = make_smem_desc_cg1(smem_k + k * 16);
                umma_f16_cg1_fn(desc_S, desc_q, desc_k, idesc_qkt, k == 0 ? 0 : 1);
            }
            for (int k = 0; k < 4; ++k) {
                uint64_t desc_q = make_smem_desc_cg1(smem_q + 64 + k * 16);
                uint64_t desc_k = make_smem_desc_cg1(smem_k + 64 + k * 16);
                shift_tmem_ptr(desc_S, 8192, false);
                umma_f16_cg1_fn(desc_S, desc_q, desc_k, idesc_qkt, 1);
            }
            umma_commit_1sm_fn_cg1(bar);
        }
        mbarrier_wait_fn(bar, phase_bar);
        phase_bar ^= 1;
        
        float my_max = -INFINITY;
        int q_idx = s_offset_q + threadIdx.x;
        
        for (int col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_S_addr + col, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            
            float f0 = __uint_as_float(r0);
            float f1 = __uint_as_float(r1);
            float f2 = __uint_as_float(r2);
            float f3 = __uint_as_float(r3);
            
            f0 *= scale;
            f1 *= scale;
            f2 *= scale;
            f3 *= scale;
            
            int k_idx0 = s_offset_k + col;
            if (k_idx0 > q_idx || k_idx0 >= S) f0 = -INFINITY;
            if (k_idx0 + 1 > q_idx || k_idx0 + 1 >= S) f1 = -INFINITY;
            if (k_idx0 + 2 > q_idx || k_idx0 + 2 >= S) f2 = -INFINITY;
            if (k_idx0 + 3 > q_idx || k_idx0 + 3 >= S) f3 = -INFINITY;
            
            my_max = fmaxf(my_max, fmaxf(fmaxf(f0, f1), fmaxf(f2, f3)));
        }
        
        float curr_max = fmaxf(row_max, my_max);
        float curr_sum = row_sum * (curr_max > -INFINITY && row_max > -INFINITY ? expf(row_max - curr_max) : 0.0f);
        
        float my_sum = 0;
        for (int col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_S_addr + col, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            
            float f0 = __uint_as_float(r0) * scale;
            float f1 = __uint_as_float(r1) * scale;
            float f2 = __uint_as_float(r2) * scale;
            float f3 = __uint_as_float(r3) * scale;
            
            int k_idx0 = s_offset_k + col;
            if (k_idx0 > q_idx || k_idx0 >= S) f0 = -INFINITY;
            if (k_idx0 + 1 > q_idx || k_idx0 + 1 >= S) f1 = -INFINITY;
            if (k_idx0 + 2 > q_idx || k_idx0 + 2 >= S) f2 = -INFINITY;
            if (k_idx0 + 3 > q_idx || k_idx0 + 3 >= S) f3 = -INFINITY;
            
            float p0 = expf(f0 - curr_max);
            float p1 = expf(f1 - curr_max);
            float p2 = expf(f2 - curr_max);
            float p3 = expf(f3 - curr_max);
            
            my_sum += p0 + p1 + p2 + p3;
            
            smem_p[threadIdx.x * 128 + col + 0] = __float2bfloat16(p0);
            smem_p[threadIdx.x * 128 + col + 1] = __float2bfloat16(p1);
            smem_p[threadIdx.x * 128 + col + 2] = __float2bfloat16(p2);
            smem_p[threadIdx.x * 128 + col + 3] = __float2bfloat16(p3);
        }
        curr_sum += my_sum;
        row_max = curr_max;
        row_sum = curr_sum;
        
        __syncthreads();
        
        uint32_t tx_bytes_p = (128 * 64 + 128 * 64) * sizeof(__nv_bfloat16);
        fence_proxy_async_fn();
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar, tx_bytes_p);
            uint64_t desc_O = make_smem_desc_cg1(smem_o, 2048, 128);
            uint32_t idesc_pv = make_instr_desc_fn_cg1_mnmaj(128, 128);
            
            for (int k = 0; k < 4; ++k) {
                uint64_t desc_p = make_smem_desc_cg1(smem_p + k * 16);
                uint64_t desc_v = make_smem_desc_cg1(smem_v + k * 16 * 128);
                umma_f16_cg1_fn(desc_O, desc_p, desc_v, idesc_pv, k == 0 ? 0 : 1);
            }
            for (int k = 0; k < 4; ++k) {
                uint64_t desc_p = make_smem_desc_cg1(smem_p + 64 + k * 16);
                uint64_t desc_v = make_smem_desc_cg1(smem_v + (64 + k * 16) * 128);
                shift_tmem_ptr(desc_O, 8192, false);
                umma_f16_cg1_fn(desc_O, desc_p, desc_v, idesc_pv, 1);
            }
            umma_commit_1sm_fn_cg1(bar);
        }
        mbarrier_wait_fn(bar, phase_bar);
        phase_bar ^= 1;
    }
    
    __syncthreads();
    
    if (q_idx < S) {
        float lse_val = INFINITY;
        if (curr_sum > 0.0f) {
            lse_val = curr_max + logf(curr_sum);
        }
        LSE_gmem[(b * H + h) * S + q_idx] = lse_val;
        
        for (int col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_O_addr + col, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            
            float f0 = __uint_as_float(r0);
            float f1 = __uint_as_float(r1);
            float f2 = __uint_as_float(r2);
            float f3 = __uint_as_float(r3);
            
            if (curr_sum > 0.0f) {
                f0 /= curr_sum;
                f1 /= curr_sum;
                f2 /= curr_sum;
                f3 /= curr_sum;
            }
            
            O_gmem[(b * H + h) * S * 128 + q_idx * 128 + col + 0] = __float2bfloat16(f0);
            O_gmem[(b * H + h) * S * 128 + q_idx * 128 + col + 1] = __float2bfloat16(f1);
            O_gmem[(b * H + h) * S * 128 + q_idx * 128 + col + 2] = __float2bfloat16(f2);
            O_gmem[(b * H + h) * S * 128 + q_idx * 128 + col + 3] = __float2bfloat16(f3);
        }
    }
    
    if (threadIdx.x == 0) {
        tmem_dealloc_fn_cg1(tmem_S_addr, 128);
        tmem_dealloc_fn_cg1(tmem_P_addr, 128);
        tmem_dealloc_fn_cg1(tmem_O_addr, 128);
    }
    __syncthreads();
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int64_t B = Q.size(0);
    int64_t H_ = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);

    if (H_ != H) {
        fprintf(stderr, "Expected H=%d, got H=%ld\n", H, H_);
        exit(1);
    }
    
    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());

    int64_t blocks = B * H_;
    int64_t threads = 128;
    
    int smem_size = 131100;
    CUDA_CHECK(cudaFuncSetAttribute(
        causal_attention_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size));
        
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    causal_attention_kernel<<<blocks, threads, smem_size, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data, S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace causal_attention