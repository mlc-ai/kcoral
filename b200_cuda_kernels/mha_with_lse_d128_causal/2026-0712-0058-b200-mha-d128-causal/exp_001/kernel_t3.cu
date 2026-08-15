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

__device__ __forceinline__ uint64_t make_smem_desc_cg1(void* smem_ptr, uint32_t lbo, uint32_t sbo, bool is_128b_swizzle) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    if (is_128b_swizzle) {
        d |= (uint64_t)2 << 61;   
    }
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

__device__ __forceinline__ uint32_t make_instr_desc_fn_cg1_b_major_1(uint32_t M, uint32_t N) {
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

struct SwizzledStorage {
    __nv_bfloat16* ptr;
    
    __device__ __forceinline__ __nv_bfloat16& operator()(int row, int col) {
        int chunk_x = col / 64;
        int chunk_y = row % 8;
        int swizzled_chunk = chunk_x ^ chunk_y;
        int swizzled_col = swizzled_chunk * 64 + (col % 64);
        return ptr[row * 128 + swizzled_col];
    }
};

__device__ __forceinline__ void load_tile_swizzled(SwizzledStorage& smem, const __nv_bfloat16* gmem, int b, int h, int s_offset, int S) {
    int tid = threadIdx.x;
    int idx = s_offset + tid;
    float4* smem_f4 = (float4*)smem.ptr;
    const float4* gmem_f4 = (const float4*)gmem;
    for (int i = 0; i < 16; i++) {
        float4 val = {0, 0, 0, 0};
        if (idx < S) {
            val = gmem_f4[((b * H + h) * S + idx) * 16 + i];
        }
        smem_f4[(idx * 16) + i] = val;
    }
}

__device__ __forceinline__ void gemm_128x128x128(uint32_t tmem_S_base, SwizzledStorage& Q_storage, SwizzledStorage& K_storage) {
    for (int k = 0; k < 2; ++k) {
        uint64_t desc_a = make_smem_desc_cg1(Q_storage.ptr + k * 64, 1, 1024, true);
        uint64_t desc_b = make_smem_desc_cg1(K_storage.ptr + k * 64 * 128, 16384, 1024, true);
        uint32_t offset_bytes = k * 64 * 4;
        uint32_t accum = 0;
        for (int i = 0; i < 4; ++i) {
            uint32_t idesc_qkt = make_instr_desc_fn_cg1(128, 128);
            umma_f16_cg1_fn(tmem_S_base + offset_bytes + i * 16, desc_a, desc_b, idesc_qkt, accum);
            accum = 1;
        }
    }
}

__device__ __forceinline__ void gemm_PV_128x128x128(uint32_t tmem_O_base, SwizzledStorage& P_storage, SwizzledStorage& V_storage) {
    for (int k = 0; k < 2; ++k) {
        uint64_t desc_a = make_smem_desc_cg1(P_storage.ptr + k * 64, 1, 1024, true);
        uint64_t desc_b = make_smem_desc_cg1(V_storage.ptr + k * 64 * 128, 16384, 1024, true);
        uint32_t offset_bytes = k * 64 * 4;
        uint32_t accum = 0;
        for (int i = 0; i < 4; ++i) {
            uint32_t idesc_pv = make_instr_desc_fn_cg1_b_major_1(128, 128);
            umma_f16_cg1_fn(tmem_O_base + offset_bytes + i * 16, desc_a, desc_b, idesc_pv, accum);
            accum = 1;
        }
    }
}

__device__ __forceinline__ uint32_t pack_bf16_fn(float a, float b) {
    __nv_bfloat16 ba = __float2bfloat16(a);
    __nv_bfloat16 bb = __float2bfloat16(b);
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};"
        : "=r"(result)
        : "h"(*reinterpret_cast<uint16_t*>(&ba)),
          "h"(*reinterpret_cast<uint16_t*>(&bb)));
    return result;
}

__global__ void causal_attention_kernel(
    const __nv_bfloat16* Q_gmem, const __nv_bfloat16* K_gmem, const __nv_bfloat16* V_gmem,
    __nv_bfloat16* O_gmem, float* LSE_gmem, int S)
{
    setmaxnreg_inc_sync_fn<256>();

    extern __shared__ __align__(128) char smem[];
    SwizzledStorage Q_storage { (__nv_bfloat16*)smem };                   
    SwizzledStorage K_storage { (__nv_bfloat16*)(smem + 32768) };         
    SwizzledStorage V_storage { (__nv_bfloat16*)(smem + 65536) };         
    SwizzledStorage P_storage { (__nv_bfloat16*)(smem + 98304) };         
    uint64_t* bar = (uint64_t*)(smem + 131072);                     
    
    uint32_t tmem_S_addr, tmem_O_addr;
    if (threadIdx.x == 0) {
        tmem_alloc_fn_cg1(&tmem_S_addr, 128);
        tmem_alloc_fn_cg1(&tmem_O_addr, 128);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar, 1);
    }
    __syncthreads();
    
    float global_max = -INFINITY;
    float global_sum = 0.0f;
    float scale = 1.0f / sqrtf(128.0f);
    int phase_bar = 0;
    
    int b = blockIdx.x / H;
    int h = blockIdx.x % H;
    int s_offset_q = (blockIdx.x * blockDim.x + threadIdx.x) / 128 * 128;
    int q_idx = s_offset_q + threadIdx.x;
    
    load_tile_swizzled(Q_storage, Q_gmem, b, h, s_offset_q, S);
    __syncthreads();

    for (int s_offset_k = 0; s_offset_k <= s_offset_q + 127; s_offset_k += 128) {
        load_tile_swizzled(K_storage, K_gmem, b, h, s_offset_k, S);
        load_tile_swizzled(V_storage, V_gmem, b, h, s_offset_k, S);
        __syncthreads();
        
        uint32_t tx_bytes_q = (128 * 64 + 128 * 64) * sizeof(__nv_bfloat16);
        fence_proxy_async_fn();
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar, tx_bytes_q);
            gemm_128x128x128(tmem_S_addr, Q_storage, K_storage);
            umma_commit_1sm_fn_cg1(bar);
        }
        mbarrier_wait_fn(bar, phase_bar);
        phase_bar ^= 1;
        
        float my_max = -INFINITY;
        
        for (int col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_S_addr + (threadIdx.x << 16) + col, &r0, &r1, &r2, &r3);
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
            
            my_max = fmaxf(my_max, fmaxf(fmaxf(f0, f1), fmaxf(f2, f3)));
        }
        
        float row_max = my_max;
        row_max = fmaxf(row_max, __shfl_xor_sync(0xFFFFFFFF, row_max, 1));
        row_max = fmaxf(row_max, __shfl_xor_sync(0xFFFFFFFF, row_max, 2));
        row_max = fmaxf(row_max, __shfl_xor_sync(0xFFFFFFFF, row_max, 4));
        row_max = fmaxf(row_max, __shfl_xor_sync(0xFFFFFFFF, row_max, 8));
        row_max = fmaxf(row_max, __shfl_xor_sync(0xFFFFFFFF, row_max, 16));
        
        float curr_max = fmaxf(global_max, row_max);
        float curr_sum = global_sum * (curr_max > -INFINITY && global_max > -INFINITY ? expf(global_max - curr_max) : 0.0f);
        
        float my_sum = 0;
        for (int col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_S_addr + (threadIdx.x << 16) + col, &r0, &r1, &r2, &r3);
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
            
            uint32_t p01 = pack_bf16_fn(p0, p1);
            uint32_t p23 = pack_bf16_fn(p2, p3);
            *(uint32_t*)&P_storage(threadIdx.x, col + 0) = p01;
            *(uint32_t*)&P_storage(threadIdx.x, col + 2) = p23;
        }
        
        float row_sum = my_sum;
        row_sum += __shfl_xor_sync(0xFFFFFFFF, row_sum, 1);
        row_sum += __shfl_xor_sync(0xFFFFFFFF, row_sum, 2);
        row_sum += __shfl_xor_sync(0xFFFFFFFF, row_sum, 4);
        row_sum += __shfl_xor_sync(0xFFFFFFFF, row_sum, 8);
        row_sum += __shfl_xor_sync(0xFFFFFFFF, row_sum, 16);
        
        curr_sum += row_sum;
        global_sum = curr_sum;
        global_max = curr_max;
        
        __syncthreads();
        
        uint32_t tx_bytes_p = (128 * 64 + 128 * 64) * sizeof(__nv_bfloat16);
        fence_proxy_async_fn();
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar, tx_bytes_p);
            gemm_PV_128x128x128(tmem_O_addr, P_storage, V_storage);
            umma_commit_1sm_fn_cg1(bar);
        }
        mbarrier_wait_fn(bar, phase_bar);
        phase_bar ^= 1;
    }
    
    __syncthreads();
    
    if (q_idx < S) {
        float lse_val = INFINITY;
        if (global_sum > 0.0f) {
            lse_val = global_max + logf(global_sum);
        }
        LSE_gmem[(b * H + h) * S + q_idx] = lse_val;
        
        for (int col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_O_addr + (threadIdx.x << 16) + col, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            
            float f0 = __uint_as_float(r0);
            float f1 = __uint_as_float(r1);
            float f2 = __uint_as_float(r2);
            float f3 = __uint_as_float(r3);
            
            if (global_sum > 0.0f) {
                f0 /= global_sum;
                f1 /= global_sum;
                f2 /= global_sum;
                f3 /= global_sum;
            }
            
            O_gmem[(b * H + h) * S * 128 + q_idx * 128 + col + 0] = __float2bfloat16(f0);
            O_gmem[(b * H + h) * S * 128 + q_idx * 128 + col + 1] = __float2bfloat16(f1);
            O_gmem[(b * H + h) * S * 128 + q_idx * 128 + col + 2] = __float2bfloat16(f2);
            O_gmem[(b * H + h) * S * 128 + q_idx * 128 + col + 3] = __float2bfloat16(f3);
        }
    }
    
    if (threadIdx.x == 0) {
        tmem_dealloc_fn_cg1(tmem_S_addr, 128);
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