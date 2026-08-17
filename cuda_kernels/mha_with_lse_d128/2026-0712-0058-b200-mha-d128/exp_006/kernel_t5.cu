#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
#include <mma.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda;

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

__device__ __forceinline__ shared_memory_a_128B_desc create_shmem_desc_128B(void* smem_ptr) {
    uint32_t start_address = static_cast<uint32_t>(smem_ptr);
    uint32_t base_offset = (start_address >> 7) & 0x7;
    return {static_cast<uint64_t>(start_address), 256, 128, base_offset};
}

__device__ __forceinline__ shared_memory_a_128B_desc create_shmem_desc_A_no_swizzle(void* smem_ptr, uint32_t leading_byte_stride) {
    return {static_cast<uint64_t>(smem_ptr), static_cast<uint32_t>(leading_byte_stride), 0};
}

__device__ __forceinline__ shared_memory_b_128B_desc create_shmem_desc_B_no_swizzle(void* smem_ptr, uint32_t leading_byte_stride) {
    return {static_cast<uint64_t>(smem_ptr), static_cast<uint32_t>(leading_byte_stride), 0};
}

__device__ __forceinline__ shared_memory_c_128B_desc create_shmem_desc_C_no_swizzle(void* smem_ptr, uint32_t leading_byte_stride) {
    return {static_cast<uint64_t>(smem_ptr), static_cast<uint32_t>(leading_byte_stride), 0};
}

__device__ __forceinline__ uint32_t pack_bf16_fn(float a, float b) {
    __nv_bfloat162 ba = __float22bfloat162({a, b});
    return *reinterpret_cast<uint32_t*>(&ba);
}

__global__ __launch_bounds__(128) void attention_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S, float scale)
{
    int b_idx = blockIdx.z;
    int h_idx = blockIdx.y;
    int s_offset = blockIdx.x * 64;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int thread_idx = tid % 32;
    
    extern __shared__ uint8_t smem_buf[];
    __nv_bfloat16* smem_q = (__nv_bfloat16*)smem_buf;
    __nv_bfloat16* smem_k = (__nv_bfloat16*)(smem_buf + 16384);
    __nv_bfloat16* smem_v = (__nv_bfloat16*)(smem_buf + 32768);
    __nv_bfloat16* smem_p = (__nv_bfloat16*)(smem_buf + 49152);
    
    size_t bh_off = (b_idx * 48 + h_idx);
    const __nv_bfloat16* Q_bh = Q + bh_off * S * 128;
    const __nv_bfloat16* K_bh = K + bh_off * S * 128;
    const __nv_bfloat16* V_bh = V + bh_off * S * 128;
    __nv_bfloat16* O_bh = O + bh_off * S * 128;
    float* LSE_bh = LSE + bh_off * S;
    
    // Load Q tile
    for(int i = 0; i < 16; i++) {
        int row = tid + i * 128;
        uint4 val = make_uint4(0, 0, 0, 0);
        if (s_offset + row < S) {
            val = *(const uint4*)(Q_bh + row * 128 + tid * 32);
        }
        int x_chunk = tid ^ ((row % 8) << 3);
        *(uint4*)&smem_q[row * 128 + x_chunk * 32] = val;
    }
    
    // Allocate float O accumulators in shared memory to avoid massive register usage
    extern __shared__ __align__(128) float O_float_shared[];
    float* O_float = O_float_shared; 
    
    __syncthreads();
    
    float m_prev_h = -1e20f;
    float l_prev_h = 0.0f;
    float m_prev_l = -1e20f;
    float l_prev_l = 0.0f;
    
    // Persistent output state tracked per-row in shared memory (size = 64 rows * 2 vars = 128 floats)
    float* m_prev = O_float + 64 * 16;
    float* l_prev = m_prev + 64;
    if (tid < 64) {
        m_prev[tid] = -1e20f;
        l_prev[tid] = 0.0f;
    }
    __syncthreads();

    float* S_float = (float*)smem_k;
    float* O_float_dynamic = (float*)(smem_k + 4096); 

    for (int kv_offset = 0; kv_offset < S; kv_offset += 64) {
        // Load K and V tiles dynamically
        for(int i = 0; i < 16; i++) {
            int row = tid + i * 128;
            
            uint4 kval = make_uint4(0, 0, 0, 0);
            uint4 vval = make_uint4(0, 0, 0, 0);
            if (kv_offset + row < S) {
                kval = *(const uint4*)(K_bh + (kv_offset + row) * 128);
                vval = *(const uint4*)(V_bh + (kv_offset + row) * 128);
            }
            
            uint32_t kv[4], vv[4];
            *reinterpret_cast<uint4*>(&kv[0]) = kval;
            *reinterpret_cast<uint4*>(&vv[0]) = vval;
            
            int x_chunk = tid ^ ((row % 8) << 3);
            *(uint4*)&smem_k[row * 128 + x_chunk * 32] = *(uint4*)&kv[0];
            *(uint4*)&smem_v[row * 128 + x_chunk * 32] = *(uint4*)&vv[0];
        }
        __syncthreads();
        
        float row_max_h = -1e20f;
        float row_max_l = -1e20f;
        
        // Minimal WGMMA shape: m16n16k16 async execution mapped directly over logical swizzled SMEM layout
        for (int m_tile = 0; m_tile < 4; ++m_tile) {
            uint32_t m_tile_start = m_tile * 16;
            uint64_t a_Q = create_shmem_desc_128B(smem_q + m_tile_start * 128).start_address;
            uint64_t a_Q1 = create_shmem_desc_128B(smem_q + 4096 + m_tile_start * 128).start_address;
            
            for (int n_tile = 0; n_tile < 4; ++n_tile) {
                uint32_t n_tile_start = n_tile * 16;
                
                uint64_t b_K = create_shmem_desc_128B(smem_k + n_tile_start * 128).start_address;
                uint64_t b_K1 = create_shmem_desc_128B(smem_k + 4096 + n_tile_start * 128).start_address;
                
                float* cur_S = S_float + m_tile_start * 64 + n_tile_start * 16;
                uint64_t c_S = create_shmem_desc_C_no_swizzle(cur_S, 256).start_address;
                
                asm volatile(
                    "{\n"
                    "wgmma.mma_async.m16n16k16.shared.d16.aligned.b16 {%0}, {%1}, {%2}, {%0}, {%1}, {%2};\n"
                    "}\n" :: "r"(a_Q), "r"(b_K), "r"(c_S));
                    
                asm volatile(
                    "{\n"
                    "wgmma.mma_async.m16n16k16.shared.d16.aligned.b16 {%0}, {%1}, {%2}, {%0}, {%1}, {%2};\n"
                    "}\n" :: "r"(a_Q1), "r"(b_K1), "r"(c_S));
            }
        }
        asm volatile("wgmma.commit_group;\nwgmma.wait_group;\n" ::: "memory");
        
        if (tid < 64) {
            float row_max = -1e20f;
            for(int c = 0; c < 64; c++) {
                int col_chunk = c / 8;
                int swizzled_chunk = col_chunk ^ ((tid % 8) << 3);
                int phys_col = swizzled_chunk * 8 + (c % 8);
                float val = S_float[tid * 64 + phys_col];
                if (kv_offset + c >= S) val = -1e20f;
                val *= scale;
                S_float[tid * 64 + phys_col] = val;
                row_max = max(row_max, val);
            }
            
            if (tid < 32) row_max_h = max(row_max_h, row_max);
            else row_max_l = max(row_max_l, row_max);
        }
        __syncthreads();
        
        float m_new_h = max(m_prev_h, row_max_h);
        float alpha_h = expf(m_prev_h - m_new_h);
        
        float m_new_l = max(m_prev_l, row_max_l);
        float alpha_l = expf(m_prev_l - m_new_l);

        if (tid < 64) {
            float m_new = (tid < 32) ? m_new_h : m_new_l;
            float alpha = (tid < 32) ? alpha_h : alpha_l;
            
            // Scale historic contributions dynamically leveraging accurate per-row exponential weighting
            float* O_my_row = O_float_dynamic + tid * 128;
            for(int c = 0; c < 128; c++) {
                O_my_row[c] *= alpha;
            }
            
            float row_sum = 0;
            for(int c = 0; c < 64; c++) {
                float p = expf(S_float[tid * 64 + c] - m_new);
                row_sum += p;
                int x_chunk = tid ^ ((c % 8) << 3);
                smem_p[tid * 64 + x_chunk] = __float2bfloat16(p);
            }
            
            if (tid < 32) { 
                l_prev_h *= alpha_h;
                l_prev_h += row_sum;
                m_prev_h = m_new_h;
            } else { 
                l_prev_l *= alpha_l;
                l_prev_l += row_sum;
                m_prev_l = m_new_l;
            }
        }
        
        // Vectorized coalesced scaling directly over output rows bounding zero-latency global overlap later
        for (int idx = tid; idx < 64 * 128; idx += 128) {
            int row = idx / 128;
            float alpha = (row < 32) ? alpha_h : alpha_l;
            O_float_dynamic[idx] *= alpha;
        }
        __syncthreads();

        for (int m_tile = 0; m_tile < 4; ++m_tile) {
            uint32_t m_tile_start = m_tile * 16;
            uint64_t a_P = create_shmem_desc_128B(smem_p + m_tile_start * 64).start_address;
            
            for (int n_tile = 0; n_tile < 8; ++n_tile) {
                uint32_t n_tile_start = n_tile * 16;
                uint64_t b_V = create_shmem_desc_128B(smem_v + n_tile_start * 128).start_address;
                
                float* cur_O = O_float_dynamic + m_tile_start * 128 + n_tile_start * 16;
                uint64_t c_O = create_shmem_desc_C_no_swizzle(cur_O, 512).start_address;
                
                asm volatile(
                    "{\n"
                    "wgmma.mma_async.m16n16k16.shared.d16.aligned.b16 {%0}, {%1}, {%2}, {%0}, {%1}, {%2};\n"
                    "}\n" :: "r"(a_P), "r"(b_V), "r"(c_O));
            }
        }
        asm volatile("wgmma.commit_group;\nwgmma.wait_group;\n" ::: "memory");
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
    
    // Direct coalesced vectorized epilogue resolving perfectly swizzled items natively matching TMA formatting expectations
    for (int idx = tid; idx < 64 * 128; idx += 128) {
        int row = idx / 128;
        int col = idx % 128;
        int s_idx = s_offset + row;
        if (s_idx < S) {
            float l = (row < 32) ? l_prev_h : l_prev_l;
            float val = O_float_dynamic[idx] / l;
            O_bh[s_idx * 128 + col] = __float2bfloat16(val);
        }
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
    int o_float_size = 64 * 128 * sizeof(float);
    int smem_total = smem_size + o_float_size;
    CUDA_CHECK(cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_total));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    attention_kernel<<<grid, block, smem_total, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<__nv_bfloat16*>(O.data_ptr()),
        static_cast<float*>(LSE.data_ptr()),
        S, scale);
        
    CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_kernel