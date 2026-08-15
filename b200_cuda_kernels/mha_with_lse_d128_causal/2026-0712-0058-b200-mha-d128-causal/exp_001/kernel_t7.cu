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

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t addr, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(addr));
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
        ".mbarrier::arrive::one.shared::cta.b64"
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

__device__ __forceinline__ void load_tile(__nv_bfloat16* smem, const __nv_bfloat16* gmem, int b, int h, int s_offset, int S, bool transpose) {
    int tid = threadIdx.x;
    int idx = s_offset + tid;
    float4* smem_f4 = (float4*)smem;
    const float4* gmem_f4 = (const float4*)gmem;
    for (int i = 0; i < 16; i++) {
        float4 val = {0, 0, 0, 0};
        if (idx < S) {
            if (transpose) {
                val = gmem_f4[((b * H + h) * S + i) * 16 + tid];
            } else {
                val = gmem_f4[((b * H + h) * S + idx) * 16 + i];
            }
        }
        int sc = (i / 8) ^ (tid % 8);
        smem_f4[tid * 16 + sc] = val;
    }
}

__device__ __forceinline__ void gemm_QK_T_128x128x128(uint32_t tmem_S_base, __nv_bfloat16* smem_q, __nv_bfloat16* smem_k) {
    uint64_t desc_q[8];
    uint64_t desc_k[8];
    
    desc_q[0] = make_smem_desc_cg1(smem_q + 0, 1, 1024, true);
    desc_q[1] = make_smem_desc_cg1(smem_q + 16, 1, 1024, true);
    desc_q[2] = make_smem_desc_cg1(smem_q + 32, 1, 1024, true);
    desc_q[3] = make_smem_desc_cg1(smem_q + 48, 1, 1024, true);
    desc_q[4] = make_smem_desc_cg1(smem_q + 8192, 1, 1024, true);
    desc_q[5] = make_smem_desc_cg1(smem_q + 8208, 1, 1024, true);
    desc_q[6] = make_smem_desc_cg1(smem_q + 8224, 1, 1024, true);
    desc_q[7] = make_smem_desc_cg1(smem_q + 8240, 1, 1024, true);
    
    desc_k[0] = make_smem_desc_cg1(smem_k + 0, 1, 1024, true);
    desc_k[1] = make_smem_desc_cg1(smem_k + 16, 1, 1024, true);
    desc_k[2] = make_smem_desc_cg1(smem_k + 32, 1, 1024, true);
    desc_k[3] = make_smem_desc_cg1(smem_k + 48, 1, 1024, true);
    desc_k[4] = make_smem_desc_cg1(smem_k + 8192, 1, 1024, true);
    desc_k[5] = make_smem_desc_cg1(smem_k + 8208, 1, 1024, true);
    desc_k[6] = make_smem_desc_cg1(smem_k + 8224, 1, 1024, true);
    desc_k[7] = make_smem_desc_cg1(smem_k + 8240, 1, 1024, true);

    uint32_t base_col_S = tmem_S_base & 0xFFFF;

    for (int k = 0; k < 8; ++k) {
        uint32_t accum = (k == 0) ? 0 : 1;
        uint32_t idesc_qkt = make_instr_desc_fn_cg1(128, 128);
        umma_f16_cg1_fn((threadIdx.x << 16) | (base_col_S + (k % 4) * 16), desc_q[k], desc_k[k], idesc_qkt, accum);
    }
}

__device__ __forceinline__ void gemm_PV_128x128x128(uint32_t tmem_O_base, __nv_bfloat16* smem_p, __nv_bfloat16* smem_v) {
    uint64_t desc_p[8];
    uint64_t desc_v[8];
    
    desc_p[0] = make_smem_desc_cg1(smem_p + 0, 1, 1024, true);
    desc_p[1] = make_smem_desc_cg1(smem_p + 16, 1, 1024, true);
    desc_p[2] = make_smem_desc_cg1(smem_p + 32, 1, 1024, true);
    desc_p[3] = make_smem_desc_cg1(smem_p + 48, 1, 1024, true);
    desc_p[4] = make_smem_desc_cg1(smem_p + 8192, 1, 1024, true);
    desc_p[5] = make_smem_desc_cg1(smem_p + 8208, 1, 1024, true);
    desc_p[6] = make_smem_desc_cg1(smem_p + 8224, 1, 1024, true);
    desc_p[7] = make_smem_desc_cg1(smem_p + 8240, 1, 1024, true);
    
    desc_v[0] = make_smem_desc_cg1(smem_v + 0, 1024, 1024, true);
    desc_v[1] = make_smem_desc_cg1(smem_v + 16 * 128, 1024, 1024, true);
    desc_v[2] = make_smem_desc_cg1(smem_v + 32 * 128, 1024, 1024, true);
    desc_v[3] = make_smem_desc_cg1(smem_v + 48 * 128, 1024, 1024, true);
    desc_v[4] = make_smem_desc_cg1(smem_v + 64 * 128, 1024, 1024, true);
    desc_v[5] = make_smem_desc_cg1(smem_v + 80 * 128, 1024, 1024, true);
    desc_v[6] = make_smem_desc_cg1(smem_v + 96 * 128, 1024, 1024, true);
    desc_v[7] = make_smem_desc_cg1(smem_v + 112 * 128, 1024, 1024, true);

    uint32_t base_col_O = tmem_O_base & 0xFFFF;

    for (int k = 0; k < 8; ++k) {
        uint32_t accum = (k == 0) ? 0 : 1;
        uint32_t idesc_pv = make_instr_desc_fn_cg1_b_major_1(128, 128);
        umma_f16_cg1_fn((threadIdx.x << 16) | (base_col_O + (k % 4) * 16), desc_p[k], desc_v[k], idesc_pv, accum);
    }
}

__global__ void causal_attention_kernel(
    const __nv_bfloat16* Q_gmem, const __nv_bfloat16* K_gmem, const __nv_bfloat16* V_gmem,
    __nv_bfloat16* O_gmem, float* LSE_gmem, int S)
{
    extern __shared__ __align__(128) char smem[];
    __nv_bfloat16* smem_q = (__nv_bfloat16*)smem;                   
    __nv_bfloat16* smem_k = (__nv_bfloat16*)(smem + 32768);         
    __nv_bfloat16* smem_v = (__nv_bfloat16*)(smem + 65536);         
    __nv_bfloat16* smem_p = (__nv_bfloat16*)(smem + 98304);         
    __nv_bfloat16* smem_o = (__nv_bfloat16*)(smem + 131072);         
    float* smem_lse = (float*)(smem + 163840);                       
    uint64_t* bar0 = (uint64_t*)(smem + 163848);                     

    uint32_t tmem_S_addr, tmem_O_addr;
    if (threadIdx.x == 0) {
        tmem_alloc_fn_cg1(&tmem_S_addr, 128);
        tmem_alloc_fn_cg1(&tmem_O_addr, 128);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar0, 1);
    }
    __syncthreads();
    
    int total_bh = gridDim.x / ((S + 127) / 128);
    int seq_tile_idx = blockIdx.x / total_bh;
    int bh_idx = blockIdx.x % total_bh;
    int b = bh_idx / H;
    int h = bh_idx % H;

    int s_offset_q = seq_tile_idx * 128;
    
    float global_max = -INFINITY;
    float global_sum = 0.0f;
    float scale = 1.0f / sqrtf(128.0f);
    int phase_bar = 0;
    
    int tid = threadIdx.x;
    int q_idx = s_offset_q + tid;
    
    load_tile(smem_q, Q_gmem, b, h, s_offset_q, S, false);
    __syncthreads();

    for (int s_offset_k = 0; s_offset_k <= s_offset_q; s_offset_k += 128) {
        load_tile(smem_k, K_gmem, b, h, s_offset_k, S, false);
        load_tile(smem_v, V_gmem, b, h, s_offset_k, S, false);
        __syncthreads();
        
        fence_proxy_async_fn();
        if (threadIdx.x == 0) {
            gemm_QK_T_128x128x128(tmem_S_addr, smem_q, smem_k);
            umma_commit_1sm_fn_cg1(bar0);
        }
        mbarrier_wait_fn(bar0, phase_bar);
        phase_bar ^= 1;
        
        float my_max = -INFINITY;
        uint32_t base_col_S = tmem_S_addr & 0xFFFF;
        for (int col = 0; col < 128; col += 2) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn((tid << 16) | (base_col_S + col), &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            
            float f0 = __uint_as_float(r0) * scale;
            float f1 = __uint_as_float(r1) * scale;
            
            int k_idx0 = s_offset_k + col;
            if (k_idx0 > q_idx || k_idx0 >= S) f0 = -INFINITY;
            if (k_idx0 + 1 > q_idx || k_idx0 + 1 >= S) f1 = -INFINITY;
            
            my_max = fmaxf(my_max, fmaxf(f0, f1));
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
        for (int col = 0; col < 128; col += 2) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn((tid << 16) | (base_col_S + col), &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            
            float f0 = __uint_as_float(r0) * scale;
            float f1 = __uint_as_float(r1) * scale;
            
            int k_idx0 = s_offset_k + col;
            if (k_idx0 > q_idx || k_idx0 >= S) f0 = -INFINITY;
            if (k_idx0 + 1 > q_idx || k_idx0 + 1 >= S) f1 = -INFINITY;
            
            float p0 = (f0 > -INFINITY) ? expf(f0 - curr_max) : 0.0f;
            float p1 = (f1 > -INFINITY) ? expf(f1 - curr_max) : 0.0f;
            
            my_sum += p0 + p1;
            
            int sc0 = (((col) / 64) ^ (tid % 8)) * 64 + (col % 64);
            int sc1 = (((col + 1) / 64) ^ (tid % 8)) * 64 + ((col + 1) % 64);
            smem_p[tid * 128 + sc0] = __float2bfloat16(p0);
            smem_p[tid * 128 + sc1] = __float2bfloat16(p1);
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
        fence_proxy_async_fn();
        
        if (threadIdx.x == 0) {
            gemm_PV_128x128x128(tmem_O_addr, smem_p, smem_v);
            umma_commit_1sm_fn_cg1(bar0);
        }
        mbarrier_wait_fn(bar0, phase_bar);
        phase_bar ^= 1;
        __syncthreads();
    }
    
    __syncthreads();
    
    smem_lse[tid] = (global_sum > 0.0f) ? (global_max + logf(global_sum)) : INFINITY;
    
    uint32_t base_col_O = tmem_O_addr & 0xFFFF;
    for (int col = 0; col < 128; col += 2) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn((tid << 16) | (base_col_O + col), &r0, &r1, &r2, &r3);
        tmem_load_fence_fn();
        
        float f0 = __uint_as_float(r0);
        float f1 = __uint_as_float(r1);
        
        if (global_sum > 0.0f) {
            f0 /= global_sum;
            f1 /= global_sum;
        }
        
        int sc0 = (((col) / 64) ^ (tid % 8)) * 64 + (col % 64);
        int sc1 = (((col + 1) / 64) ^ (tid % 8)) * 64 + ((col + 1) % 64);
        smem_o[tid * 128 + sc0] = __float2bfloat16(f0);
        smem_o[tid * 128 + sc1] = __float2bfloat16(f1);
    }
    
    __syncthreads();
    
    for (int i = tid; i < 128; i += 128) {
        int sc = ((i / 64) ^ (tid % 8)) * 64 + (i % 64);
        O_gmem[(b * H + h) * S * 128 + q_idx * 128 + i] = smem_o[tid * 128 + sc];
    }
    
    if (q_idx < S) {
        LSE_gmem[(b * H + h) * S + q_idx] = smem_lse[tid];
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

    int64_t blocks = ((S + 127) / 128) * B * H_;
    int64_t threads = 128;
    
    int smem_size = 164000;
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