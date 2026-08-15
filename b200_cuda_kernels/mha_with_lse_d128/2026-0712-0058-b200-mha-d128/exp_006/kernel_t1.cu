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

#define CU_CHECK(call) do {                                      \
    CUresult _e = (call);                                        \
    if (_e != CUDA_SUCCESS) {                                    \
        fprintf(stderr, "CU error %d at %s:%d\n", (int)_e,         \
                __FILE__, __LINE__);                             \
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

CUresult create_tma_4d_descriptor(CUtensorMap* d, void* globalAddress, 
                                   uint64_t g_D, uint64_t g_S, uint64_t g_H, uint64_t g_B,
                                   uint32_t b_D, uint32_t b_S) {
    cuuint64_t globalDim[4] = {g_D, g_S, g_H, g_B};
    cuuint64_t globalStrides[3] = {g_D * 2, g_D * g_S * 2, g_D * g_S * g_H * 2};
    cuuint32_t boxDim[4] = {b_D, b_S, 1, 1};
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
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

__device__ __forceinline__ void tma_load_4d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
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

__device__ __forceinline__ uint32_t tmem_load_4x_fn(uint32_t col) {
    uint32_t r;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0}, [%1];"
   : "=r"(r) : "r"(col));
    return r;
}

__device__ __forceinline__ void tmem_load_float_row_fn(uint32_t base_addr, uint32_t tid, float* val) {
    val[0]  = __uint_as_float(tmem_load_4x_fn(base_addr + (tid << 16) + 0));
    val[1]  = __uint_as_float(tmem_load_4x_fn(base_addr + (tid << 16) + 16));
    val[2]  = __uint_as_float(tmem_load_4x_fn(base_addr + (tid << 16) + 32));
    val[3]  = __uint_as_float(tmem_load_4x_fn(base_addr + (tid << 16) + 48));
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

template <int transpose_b>
__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (0u << 15);   
    d |= (transpose_b << 16); 
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
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

__device__ __forceinline__ void umma_f16_scaled(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum, float scale) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p, %5;\n"
        "}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum), "f"(scale));
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ uint32_t pack_bf16_fn(float a, float b) {
    __nv_bfloat162 ba = __floats2bfloat162({a, b});
    return *reinterpret_cast<uint32_t*>(&ba);
}

__global__ __launch_bounds__(128) void attention_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S, float scale)
{
    int b_idx = blockIdx.z;
    int h_idx = blockIdx.y;
    int s_offset = blockIdx.x * 64;
    
    extern __shared__ __align__(1024) uint8_t smem_buf[];
    __nv_bfloat16* smem_q = (__nv_bfloat16*)smem_buf;
    __nv_bfloat16* smem_k = (__nv_bfloat16*)(smem_buf + 16384);
    __nv_bfloat16* smem_v = (__nv_bfloat16*)(smem_buf + 32768);
    __nv_bfloat16* smem_p = (__nv_bfloat16*)(smem_buf + 49152);
    uint64_t* bar = (uint64_t*)(smem_buf + 57344);
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    
    uint32_t* O_TMEM_ptr = reinterpret_cast<uint32_t*>(smem_p + 4096);
    uint32_t* S_TMEM_ptr = reinterpret_cast<uint32_t*>(smem_p + 4096 + 4);
    if (threadIdx.x == 0) {
        tmem_alloc_fn(O_TMEM_ptr, 128); 
        tmem_alloc_fn(S_TMEM_ptr, 64);  
    }
    __syncthreads(); 
    
    uint32_t O_TMEM = (uint32_t)__cvta_generic_to_shared(O_TMEM_ptr);
    uint32_t S_TMEM = (uint32_t)__cvta_generic_to_shared(S_TMEM_ptr);
    
    uint32_t phase = 0;
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar, 16384);
        tma_load_4d_fn(&tma_Q, bar, smem_q, 0, s_offset, h_idx, b_idx);
        tma_load_4d_fn(&tma_Q, bar, smem_q + 8192, 64, s_offset, h_idx, b_idx);
    }
    mbarrier_wait_fn(bar, phase);
    phase ^= 1;
    
    float m_prev = -1e20f;
    float l_prev = 0.0f;
    
    float m_prev_h = -1e20f;
    float l_prev_h = 0.0f;
    
    float m_prev_l = -1e20f;
    float l_prev_l = 0.0f;
    
    int tid = threadIdx.x;
    uint64_t desc_q0 = make_smem_desc_sm100_fn(smem_q, 1, 1024);
    uint64_t desc_k0 = make_smem_desc_sm100_fn(smem_k, 1, 1024);
    uint64_t desc_v0 = make_smem_desc_sm100_fn(smem_v, 1, 1024);
    
    uint64_t desc_q1 = make_smem_desc_sm100_fn((char*)smem_q + 8192, 1, 1024);
    uint64_t desc_k1 = make_smem_desc_sm100_fn((char*)smem_k + 8192, 1, 1024);
    uint64_t desc_v1 = make_smem_desc_sm100_fn((char*)smem_v + 8192, 1, 1024);
    
    uint64_t desc_p = make_smem_desc_sm100_fn(smem_p, 1, 1024);
    
    uint32_t idesc_QK = make_instr_desc_fn<0>(64, 64);
    uint32_t idesc_PV0 = make_instr_desc_fn<0>(64, 64);
    uint32_t idesc_PV1 = make_instr_desc_fn<0>(64, 64);
    
    float* LSE_bh = LSE + (b_idx * 48 + h_idx) * S;
    __nv_bfloat16* O_bh = O + (b_idx * 48 + h_idx) * S * 128;

    for (int kv_offset = 0; kv_offset < S; kv_offset += 64) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar, 32768); 
            
            tma_load_4d_fn(&tma_K, bar, smem_k, 0, kv_offset, h_idx, b_idx);
            tma_load_4d_fn(&tma_K, bar, smem_k + 8192, 64, kv_offset, h_idx, b_idx);
            
            tma_load_4d_fn(&tma_V, bar, smem_v, 0, kv_offset, h_idx, b_idx);
            tma_load_4d_fn(&tma_V, bar, smem_v + 8192, 64, kv_offset, h_idx, b_idx);
        }
        mbarrier_wait_fn(bar, phase);
        phase ^= 1;
        
        float row_max_h = -1e20f;
        float row_max_l = -1e20f;
        
        if (tid < 64) {
            float val[64];
            tmem_load_float_row_fn(S_TMEM + (tid << 16), tid, val);
            
            float row_max = -1e20f;
            for(int i=0; i<64; i++) {
                if (kv_offset + i >= S) val[i] = -1e20f;
                val[i] *= scale;
                row_max = max(row_max, val[i]);
            }
            
            if (tid < 32) row_max_h = max(row_max_h, row_max);
            else row_max_l = max(row_max_l, row_max);
            
            float m_new, alpha;
            if (tid < 32) {
                m_new = max(m_prev_h, row_max_h);
                alpha = expf(m_prev_h - m_new);
                l_prev_h *= alpha;
                
                float row_sum = 0;
                for(int i=0; i<64; i++) {
                    float p = expf(val[i] - m_new);
                    row_sum += p;
                    int chunk_idx = i / 8;
                    int swizzled_chunk = chunk_idx ^ (tid % 8);
                    int phys_col = swizzled_chunk * 8 + (i % 8);
                    smem_p[tid * 64 + phys_col] = __float2bfloat16(p);
                }
                l_prev_h += row_sum;
                m_prev_h = m_new;
            } else {
                m_new = max(m_prev_l, row_max_l);
                alpha = expf(m_prev_l - m_new);
                l_prev_l *= alpha;
                
                float row_sum = 0;
                for(int i=0; i<64; i++) {
                    float p = expf(val[i] - m_new);
                    row_sum += p;
                    int chunk_idx = i / 8;
                    int swizzled_chunk = chunk_idx ^ (tid % 8);
                    int phys_col = swizzled_chunk * 8 + (i % 8);
                    smem_p[tid * 64 + phys_col] = __float2bfloat16(p);
                }
                l_prev_l += row_sum;
                m_prev_l = m_new;
            }
        }
        
        float m_new_h = max(m_prev_h, row_max_h);
        float alpha_h = expf(m_prev_h - m_new_h);
        
        float m_new_l = max(m_prev_l, row_max_l);
        float alpha_l = expf(m_prev_l - m_new_l);

        fence_proxy_async_fn();
        
        if (tid == 0) {
            for (int k = 0; k < 4; ++k) {
                float sv0_h = (k == 0) ? alpha_h : 1.0f;
                float sv1_h = 1.0f;
                uint64_t dp = desc_p + k * 2;
                uint64_t dv0 = desc_v0 + k * 2;
                uint64_t dv1 = desc_v1 + k * 2;
                
                umma_f16_scaled(O_TMEM, dp, dv0, idesc_PV0, 1, sv0_h);
                umma_f16_scaled(O_TMEM + 64, dp, dv1, idesc_PV1, 1, sv1_h);
                
                float sv0_l = (k == 0) ? alpha_l : 1.0f;
                float sv1_l = 1.0f;
                uint64_t dp_l = desc_p + k * 2 + (32 << 16);
                uint64_t dv0_l = desc_v0 + k * 2;
                uint64_t dv1_l = desc_v1 + k * 2;
                
                umma_f16_scaled(O_TMEM + (32 << 16), dp_l, dv0_l, idesc_PV0, 1, sv0_l);
                umma_f16_scaled(O_TMEM + (32 << 16) + 64, dp_l, dv1_l, idesc_PV1, 1, sv1_l);
            }
        }
        fence_proxy_async_fn();
        __syncthreads();
    }
    
    if (tid < 64) {
        float l = (tid < 32) ? l_prev_h : l_prev_l;
        float m = (tid < 32) ? m_prev_h : m_prev_l;
        int s_idx = s_offset + tid;
        if (s_idx < S && l > 0.0f) {
            LSE_bh[s_idx] = m + logf(l);
        }
    }
    
    float l_h = l_prev_h;
    float l_l = l_prev_l;

    for (int col = 0; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3;
        r0 = tmem_load_4x_fn(O_TMEM + (tid << 16) + col);
        r1 = tmem_load_4x_fn(O_TMEM + (tid << 16) + col + 16);
        r2 = tmem_load_4x_fn(O_TMEM + (tid << 16) + col + 32);
        r3 = tmem_load_4x_fn(O_TMEM + (tid << 16) + col + 48);
        
        float f0 = __uint_as_float(r0);
        float f1 = __uint_as_float(r1);
        float f2 = __uint_as_float(r2);
        float f3 = __uint_as_float(r3);

        float l = (tid < 32) ? l_h : l_l;
        if (l == 0.0f) { f0 = 0; f1 = 0; f2 = 0; f3 = 0; }
        else           { f0 /= l; f1 /= l; f2 /= l; f3 /= l; }
        
        int s_idx = s_offset + tid;
        if (s_idx < S) {
            int base = s_idx * 128 + col;
            *(uint32_t*)&O_bh[base] = pack_bf16_fn(f0, f1);
            *(uint32_t*)&O_bh[base + 2] = pack_bf16_fn(f2, f3);
        }
    }
    
    if (tid == 0) {
        tmem_dealloc_fn(O_TMEM, 128);
        tmem_dealloc_fn(S_TMEM, 64);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    float scale = 1.0f / sqrtf(128);
    
    dim3 grid((S + 63) / 64, H, B);
    dim3 block(128);
    
    int smem_size = 64 * 1024;
    CUDA_CHECK(cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    CUtensorMap tma_Q, tma_K, tma_V;
    CU_CHECK(create_tma_4d_descriptor(&tma_Q, static_cast<const __nv_bfloat16*>(Q.data_ptr()), 128, S, H, B, 64, 64));
    CU_CHECK(create_tma_4d_descriptor(&tma_K, static_cast<const __nv_bfloat16*>(K.data_ptr()), 128, S, H, B, 64, 64));
    CU_CHECK(create_tma_4d_descriptor(&tma_V, static_cast<const __nv_bfloat16*>(V.data_ptr()), 128, S, H, B, 64, 64));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    attention_kernel<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V,
        static_cast<__nv_bfloat16*>(O.data_ptr()),
        static_cast<float*>(LSE.data_ptr()),
        S, scale);
        
    CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_kernel