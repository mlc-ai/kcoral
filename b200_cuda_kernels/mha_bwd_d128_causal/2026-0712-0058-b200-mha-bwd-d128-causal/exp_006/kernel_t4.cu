#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
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

__device__ __forceinline__ int swizzle_128B(int row, int col) {
    int half = col / 64;
    int chunk = (col % 64) / 8;
    int swizzled_chunk = chunk ^ (row % 8);
    return row * 128 + half * 64 + swizzled_chunk * 8 + (col % 8);
}

__device__ __forceinline__ int swizzle_128B_64(int row, int col) {
    int chunk = col / 8;
    int swizzled_chunk = chunk ^ (row % 8);
    return row * 64 + swizzled_chunk * 8 + (col % 8);
}

__device__ __forceinline__ void load_tile_swizzled(const __nv_bfloat16* gmem, __nv_bfloat16* smem, int S, int global_i) {
    int col_chunk = threadIdx.x / 64;
    int row = threadIdx.x % 64;
    int g_row = global_i + row;
    int col_start = col_chunk * 64;
    
    float4 vals[8];
    if (g_row < S) {
        for (int k = 0; k < 8; k++) {
            vals[k] = *(const float4*)&gmem[g_row * 128 + col_start + k * 8];
        }
    } else {
        for (int k = 0; k < 8; k++) {
            vals[k] = {0, 0, 0, 0};
        }
    }
    
    for (int k = 0; k < 8; k++) {
        int col = col_start + k * 8;
        *(float4*)&smem[swizzle_128B(row, col)] = vals[k];
    }
}

__device__ __forceinline__ void atomicAdd_bf16(__nv_bfloat16* address, __nv_bfloat16 val) {
    float val_f = __bfloat162float(val);
    int32_t* addr_i = (int32_t*)((uintptr_t)address & ~1);
    bool is_odd = ((uintptr_t)address) & 1;
    
    while (true) {
        int32_t old = *addr_i;
        uint16_t old_bf = is_odd ? (old >> 16) : (old & 0xFFFF);
        float old_f = __bfloat162float(*(reinterpret_cast<__nv_bfloat16*>(&old_bf)));
        float new_f = old_f + val_f;
        __nv_bfloat16 new_bf16 = __float2bfloat16(new_f);
        uint16_t new_bf = *(reinterpret_cast<uint16_t*>(&new_bf16));
        
        int32_t new_val = is_odd ? (old & 0xFFFF) | (new_bf << 16) : (old & 0xFFFF0000) | new_bf;
        int32_t replaced = atomicCAS(addr_i, old, new_val);
        if (replaced == old) break;
    }
}

__global__ __launch_bounds__(128)
void bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int S,
    float attn_scale,
    uint64_t batch_head_stride)
{
    int global_j = blockIdx.x * 64;
    int batch_head_idx = blockIdx.y;

    if (global_j >= S) return;

    const __nv_bfloat16* q_ptr = Q + batch_head_idx * S * 128;
    const __nv_bfloat16* k_ptr = K + batch_head_idx * S * 128;
    const __nv_bfloat16* v_ptr = V + batch_head_idx * S * 128;
    const __nv_bfloat16* o_ptr = O + batch_head_idx * S * 128;
    const __nv_bfloat16* do_ptr = dO + batch_head_idx * S * 128;
    
    __nv_bfloat16* dq_ptr = dQ + batch_head_idx * S * 128;
    __nv_bfloat16* dk_ptr = dK + batch_head_idx * S * 128;
    __nv_bfloat16* dv_ptr = dV + batch_head_idx * S * 128;
    
    const float* l_ptr = L + batch_head_idx * batch_head_stride;

    extern __shared__ __align__(1024) char smem_pool[];
    __nv_bfloat16* s_k = (__nv_bfloat16*)(smem_pool);                
    __nv_bfloat16* s_v = (__nv_bfloat16*)(smem_pool + 16384);        
    __nv_bfloat16* s_q = (__nv_bfloat16*)(smem_pool + 32768);        
    __nv_bfloat16* s_o = (__nv_bfloat16*)(smem_pool + 49152);        
    __nv_bfloat16* s_do = (__nv_bfloat16*)(smem_pool + 65536);       
    __nv_bfloat16* s_p = (__nv_bfloat16*)(smem_pool + 81920);        
    __nv_bfloat16* s_dp = (__nv_bfloat16*)(smem_pool + 90112);       
    __nv_bfloat16* s_ds = (__nv_bfloat16*)(smem_pool + 98304);       
    float* s_d = (float*)(smem_pool + 106496);                       
    float* s_l = (float*)(smem_pool + 106752);                       

    load_tile_swizzled(k_ptr, s_k, S, global_j);
    load_tile_swizzled(v_ptr, s_v, S, global_j);
    
    float dv_acc[64];
    float dk_acc[64];
    for (int i = 0; i < 64; i++) {
        dv_acc[i] = 0.0f;
        dk_acc[i] = 0.0f;
    }

    int my_j = threadIdx.x % 64;
    int my_d_start = (threadIdx.x / 64) * 64;

    for (int global_i = global_j; global_i < S; global_i += 64) {
        load_tile_swizzled(q_ptr, s_q, S, global_i);
        load_tile_swizzled(o_ptr, s_o, S, global_i);
        load_tile_swizzled(do_ptr, s_do, S, global_i);
        
        if (global_i + threadIdx.x < S) {
            s_l[threadIdx.x] = l_ptr[global_i + threadIdx.x];
            float d_val = 0;
            for (int c = 0; c < 128; ++c) {
                d_val += __bfloat162float(s_o[swizzle_128B(threadIdx.x, c)]) * 
                         __bfloat162float(s_do[swizzle_128B(threadIdx.x, c)]);
            }
            s_d[threadIdx.x] = d_val;
        }
        __syncthreads(); 
        
        float sp_local[64];
        if (threadIdx.x < 64) {
            float2 q_f2[32]; 
            for (int k = 0; k < 16; k++) {
                float4 q_vec = *(float4*)&s_q[swizzle_128B(threadIdx.x, k * 8)];
                uint32_t* q_u32 = (uint32_t*)&q_vec;
                for (int u = 0; u < 4; u++) {
                    __nv_bfloat162 q_bf2 = *(reinterpret_cast<__nv_bfloat162*>(&q_u32[u]));
                    q_f2[k * 4 + u] = __bfloat1622float(q_bf2);
                }
            }
            
            for (int j = 0; j < 64; j++) {
                float acc = 0;
                float4 k_vec = *(float4*)&s_k[swizzle_128B(j, threadIdx.x * 8)];
                uint32_t* k_u32 = (uint32_t*)&k_vec;
                for (int u = 0; u < 4; u++) {
                    __nv_bfloat162 k_bf2 = *(reinterpret_cast<__nv_bfloat162*>(&k_u32[u]));
                    float2 k_f2 = __bfloat1622float(k_bf2);
                    acc += q_f2[threadIdx.x * 4 + u].x * k_f2.x + q_f2[threadIdx.x * 4 + u].y * k_f2.y;
                }
                sp_local[j] = acc;
            }
        }
        __syncthreads();
        
        for (int j = 0; j < 64; j++) {
            int g_i = global_i + threadIdx.x;
            int g_j = global_j + j;
            float p_val = 0;
            if (g_j <= g_i && g_j < S && g_i < S) {
                p_val = expf(sp_local[j] * attn_scale - s_l[threadIdx.x]);
            }
            s_p[swizzle_128B_64(threadIdx.x, j)] = __float2bfloat16(p_val);
        }
        __syncthreads();
        
        float dp_local[64];
        if (threadIdx.x < 64) {
            float2 do_f2[32];
            for (int k = 0; k < 16; k++) {
                float4 do_vec = *(float4*)&s_do[swizzle_128B(threadIdx.x, k * 8)];
                uint32_t* do_u32 = (uint32_t*)&do_vec;
                for (int u = 0; u < 4; u++) {
                    __nv_bfloat162 do_bf2 = *(reinterpret_cast<__nv_bfloat162*>(&do_u32[u]));
                    do_f2[k * 4 + u] = __bfloat1622float(do_bf2);
                }
            }
            
            for (int j = 0; j < 64; j++) {
                float acc = 0;
                float4 v_vec = *(float4*)&s_v[swizzle_128B(j, threadIdx.x * 8)];
                uint32_t* v_u32 = (uint32_t*)&v_vec;
                for (int u = 0; u < 4; u++) {
                    __nv_bfloat162 v_bf2 = *(reinterpret_cast<__nv_bfloat162*>(&v_u32[u]));
                    float2 v_f2 = __bfloat1622float(v_bf2);
                    acc += do_f2[threadIdx.x * 4 + u].x * v_f2.x + do_f2[threadIdx.x * 4 + u].y * v_f2.y;
                }
                dp_local[j] = acc;
            }
        }
        __syncthreads();
        
        float ds_local[64];
        for (int j = 0; j < 64; j++) {
            ds_local[j] = __bfloat162float(s_p[swizzle_128B_64(threadIdx.x, j)]) * (dp_local[j] - s_d[threadIdx.x]);
            s_ds[swizzle_128B_64(threadIdx.x, j)] = __float2bfloat16(ds_local[j]);
        }
        __syncthreads();
        
        for (int i = 0; i < 64; i++) {
            float p_val = __bfloat162float(s_p[swizzle_128B_64(i, my_j)]);
            for (int k = 0; k < 8; k++) {
                float4 do_vec = *(float4*)&s_do[swizzle_128B(global_i + i, my_d_start + k * 8)];
                uint32_t* do_u32 = (uint32_t*)&do_vec;
                for (int u = 0; u < 4; u++) {
                    __nv_bfloat162 do_bf2 = *(reinterpret_cast<__nv_bfloat162*>(&do_u32[u]));
                    float2 do_f2 = __bfloat1622float(do_bf2);
                    dv_acc[k * 8 + u * 2] += p_val * do_f2.x;
                    dv_acc[k * 8 + u * 2 + 1] += p_val * do_f2.y;
                }
            }
        }
        
        for (int i = 0; i < 64; i++) {
            float ds_val = __bfloat162float(s_ds[swizzle_128B_64(i, my_j)]);
            for (int k = 0; k < 8; k++) {
                float4 q_vec = *(float4*)&s_q[swizzle_128B(global_i + i, my_d_start + k * 8)];
                uint32_t* q_u32 = (uint32_t*)&q_vec;
                for (int u = 0; u < 4; u++) {
                    __nv_bfloat162 q_bf2 = *(reinterpret_cast<__nv_bfloat162*>(&q_u32[u]));
                    float2 q_f2 = __bfloat1622float(q_bf2);
                    dk_acc[k * 8 + u * 2] += ds_val * q_f2.x;
                    dk_acc[k * 8 + u * 2 + 1] += ds_val * q_f2.y;
                }
            }
        }
        
        float dq_acc[64];
        for (int c = 0; c < 64; c++) dq_acc[c] = 0;
        
        for (int j = 0; j < 64; j++) {
            float ds_val = __bfloat162float(s_ds[swizzle_128B_64(threadIdx.x, j)]);
            for (int k = 0; k < 8; k++) {
                float4 k_vec = *(float4*)&s_k[swizzle_128B(global_j + j, my_d_start + k * 8)];
                uint32_t* k_u32 = (uint32_t*)&k_vec;
                for (int u = 0; u < 4; u++) {
                    __nv_bfloat162 k_bf2 = *(reinterpret_cast<__nv_bfloat162*>(&k_u32[u]));
                    float2 k_f2 = __bfloat1622float(k_bf2);
                    dq_acc[k * 8 + u * 2] += ds_val * k_f2.x;
                    dq_acc[k * 8 + u * 2 + 1] += ds_val * k_f2.y;
                }
            }
        }
        
        for (int c = 0; c < 64; c++) {
            int g_i = global_i + threadIdx.x;
            int g_d = my_d_start + c;
            if (g_i < S && g_d < 128) {
                atomicAdd_bf16(&dq_ptr[g_i * 128 + g_d], __float2bfloat16(dq_acc[c]));
            }
        }
        __syncthreads();
    } 
    
    for (int c = 0; c < 64; c++) {
        int g_j = global_j + my_j;
        int g_d = my_d_start + c;
        if (g_j < S && g_d < 128) {
            dk_ptr[g_j * 128 + g_d] = __float2bfloat16(dk_acc[c]);
            dv_ptr[g_j * 128 + g_d] = __float2bfloat16(dv_acc[c]);
        }
    }
}

namespace tvm_ffi_mha_bwd {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = Q.size(3);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUDA_CHECK(cudaMemsetAsync(dQ.data_ptr(), 0, B * H * S * d * sizeof(__nv_bfloat16), stream));
    
    int num_j = (S + 63) / 64;
    dim3 grid(num_j, B * H);
    dim3 block(128);
    
    float attn_scale = 1.0f / sqrtf((float)d);
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    int smem_size = 108 * 1024;
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 1;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, bwd_kernel,
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(O.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        S, attn_scale, L.stride(1)
    ));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_bwd::run);

} // namespace tvm_ffi_mha_bwd