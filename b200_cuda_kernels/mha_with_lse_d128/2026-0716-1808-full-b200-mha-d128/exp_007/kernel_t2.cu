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

// -------------------------------------------------------------------------
// Device Helper Functions
// -------------------------------------------------------------------------

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
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ int apply_swizzle_128B(int col, int row) {
    int x = col / 8;
    int seg = x / 8;
    int chunk = x % 8;
    int swizzled_x = seg * 8 + (chunk ^ (row % 8));
    return swizzled_x * 8 + (col % 8);
}

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, const void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        const_cast<void*>(globalAddress),
        globalDim,
        globalStrides,
        boxDim, 
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        l2Promotion,
        oobFill
    );
}

// -------------------------------------------------------------------------
// Main Kernel
// -------------------------------------------------------------------------

__global__ __launch_bounds__(128, 1) void mha_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O, float* LSE,
    int S, int B_H)
{
    int q_tile = blockIdx.y;
    int bh = blockIdx.z;
    int tid = threadIdx.x;
    int row = tid / 32; // 0..3
    int col_chunk = tid % 32; // 0..31
    int q_base = q_tile * 4 + row;
    
    extern __shared__ __align__(1024) char smem[];
    
    // Memory layout definition and initialization
    __nv_bfloat16* smem_K0 = (__nv_bfloat16*)smem;                
    __nv_bfloat16* smem_K1 = (__nv_bfloat16*)(smem + 32768);      
    __nv_bfloat16* smem_V0 = (__nv_bfloat16*)(smem + 65536);      
    __nv_bfloat16* smem_V1 = (__nv_bfloat16*)(smem + 98304);      
    __nv_bfloat16* smem_P  = (__nv_bfloat16*)(smem + 131072);     
    __nv_bfloat16* smem_Q  = (__nv_bfloat16*)(smem + 132096);     
    
    uint64_t* bar_Q  = (uint64_t*)(smem + 133120);               
    uint64_t* bar_KV0 = (uint64_t*)(smem + 133128);              
    uint64_t* bar_KV1 = (uint64_t*)(smem + 133136);              
    
    float* smem_m = (float*)(smem + 133144);                      
    float* smem_l = (float*)(smem + 133160);                      
    float* smem_m_prev = (float*)(smem + 133176);                
    float* smem_l_prev = (float*)(smem + 133192);                
    float* smem_alpha = (float*)(smem + 133208);                 
    
    if (tid == 0) {
        init_smem_barrier_fn(bar_Q, 1);
        init_smem_barrier_fn(bar_KV0, 1);
        init_smem_barrier_fn(bar_KV1, 1);
    }
    
    float2 my_q[64];
    if (q_base < S) {
        const uint32_t* q_u32_ptr = reinterpret_cast<const uint32_t*>(g_Q + q_base * 128);
        for (int i = 0; i < 64; ++i) {
            my_q[i] = reinterpret_cast<float2*>(&q_u32_ptr[i])[0]; // reinterpret cast trick locally
        }
    } else {
        for (int i = 0; i < 64; ++i) {
            my_q[i] = make_float2(0.0f, 0.0f);
        }
    }
    
    if (tid < 4) {
        smem_m[tid] = -INFINITY;
        smem_l[tid] = 0.0f;
    }
    
    float2 acc_o[4] = {make_float2(0.0f, 0.0f), make_float2(0.0f, 0.0f), make_float2(0.0f, 0.0f), make_float2(0.0f, 0.0f)};
    float scale = 1.0f / __sqrtf(128.0f);
    
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_Q, 32768);
        tma_load_2d_fn(&tma_Q, bar_Q, smem_Q0, 0, bh * S + q_base);
        tma_load_2d_fn(&tma_Q, bar_Q, smem_Q1, 64, bh * S + q_base);
    }
    mbarrier_wait_fn(bar_Q, 0);
    fence_proxy_async_fn();
    
    int num_steps = (S + 127) / 128;
    int phase_kv[2] = {0, 0};
    
    if (num_steps > 0) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_KV0, 65536);
            tma_load_2d_fn(&tma_K, bar_KV0, smem_K0, 0, bh * S + 0 * 128);
            tma_load_2d_fn(&tma_K, bar_KV0, smem_K1, 64, bh * S + 0 * 128);
            tma_load_2d_fn(&tma_V, bar_KV0, smem_V0, 0, bh * S + 0 * 128);
            tma_load_2d_fn(&tma_V, bar_KV0, smem_V1, 64, bh * S + 0 * 128);
        }
    }
    
    for (int step = 0; step < num_steps; ++step) {
        int buf_idx = step % 2;
        int next_buf_idx = (step + 1) % 2;
        __nv_bfloat16* smem_K_buf = (buf_idx == 0) ? smem_K0 : smem_K1;
        __nv_bfloat16* smem_V_buf = (buf_idx == 0) ? smem_V0 : smem_V1;
        uint64_t* cur_bar = (buf_idx == 0) ? bar_KV0 : bar_KV1;
        
        if (step + 1 < num_steps) {
            if (tid == 0) {
                __nv_bfloat16* next_K = (next_buf_idx == 0) ? smem_K0 : smem_K1;
                __nv_bfloat16* next_V = (next_buf_idx == 0) ? smem_V0 : smem_V1;
                uint64_t* next_bar = (next_buf_idx == 0) ? bar_KV0 : bar_KV1;
                
                mbarrier_arrive_and_expect_tx_fn(next_bar, 65536);
                tma_load_2d_fn(&tma_K, next_bar, next_K, 0, bh * S + (step + 1) * 128);
                tma_load_2d_fn(&tma_K, next_bar, next_K + 8192, 64, bh * S + (step + 1) * 128);
                tma_load_2d_fn(&tma_V, next_bar, next_V, 0, bh * S + (step + 1) * 128);
                tma_load_2d_fn(&tma_V, next_bar, next_V + 8192, 64, bh * S + (step + 1) * 128);
            }
        }
        
        mbarrier_wait_fn(cur_bar, phase_kv[buf_idx]);
        phase_kv[buf_idx] ^= 1;
        fence_proxy_async_fn();
        
        float s = 0.0f;
        for (int d = 0; d < 64; ++d) {
            float2 q_d = my_q[d];
            
            uint32_t k_addr0 = (uint32_t)__cvta_generic_to_shared(&smem_K_buf[apply_swizzle_128B(d, tid) ]);
            uint32_t k_val0 = *(uint32_t*)k_addr0;
            __nv_bfloat162 k_d = __ushort_as_bfloat162(k_val0);
            
            s += __low2float(q_d) * __low2float(k_d);
            s += __high2float(q_d) * __high2float(k_d);
        }
        
        float2 s_vec = make_float2(s, 0.0f); 
        
        int key_idx = step * 128 + tid;
        if (key_idx >= S) s_vec = make_float2(-INFINITY, -INFINITY);
        else s_vec = make_float2(s * scale, s * scale);
        
        float m = s_vec.x;
        m = max(m, __shfl_xor_sync(0xFFFFFFFF, m, 1));
        m = max(m, __shfl_xor_sync(0xFFFFFFFF, m, 2));
        m = max(m, __shfl_xor_sync(0xFFFFFFFF, m, 4));
        m = max(m, __shfl_xor_sync(0xFFFFFFFF, m, 8));
        
        if (tid % 32 == 0) smem_m[tid / 32] = m;
        __syncthreads();
        
        float prev_max = smem_m_prev[tid / 32];
        float new_max = max(prev_max, smem_m[tid / 32]);
        
        if (new_max > prev_max) {
            float factor = __expf(prev_max - new_max);
            acc_o[0] = make_float2(acc_o[0].x * factor, acc_o[0].y * factor);
            acc_o[1] = make_float2(acc_o[1].x * factor, acc_o[1].y * factor);
            acc_o[2] = make_float2(acc_o[2].x * factor, acc_o[2].y * factor);
            acc_o[3] = make_float2(acc_o[3].x * factor, acc_o[3].y * factor);
            
            if (tid % 32 == 0) {
                smem_l[tid / 32] *= factor;
            }
        }
        
        float p = (smem_m[tid / 32] == -INFINITY) ? 0.0f : __expf(s_vec.x - smem_m[tid / 32]);
        
        float sum = p;
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, 1);
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, 2);
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, 4);
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, 8);
        
        if (tid % 32 == 0) smem_l_prev[tid / 32] = smem_l[tid / 32];
        if (tid % 32 == 0) smem_l[tid / 32] += sum * __expf(smem_m[tid / 32] - new_max);
        if (tid % 32 == 0) smem_m_prev[tid / 32] = new_max;
        
        p *= __expf(smem_m[tid / 32] - new_max);
        
        *(uint32_t*)&smem_P[tid * 2] = __bfloat1622ushort(__float2bfloat162(p, p));
        
        __syncthreads(); 
        
        for (int k = 0; k < 128; k += 4) {
            float2 p_f2 = __bfloat1622float(*(uint32_t*)&smem_P[tid * 2 + k]);
            
            uint32_t v_addr0 = (uint32_t)__cvta_generic_to_shared(&smem_V_buf[apply_swizzle_128B(k, tid)]);
            uint32_t v_addr1 = (uint32_t)__cvta_generic_to_shared(&smem_V_buf[apply_swizzle_128B(k+1, tid)]);
            uint32_t v_addr2 = (uint32_t)__cvta_generic_to_shared(&smem_V_buf[apply_swizzle_128B(k+2, tid)]);
            uint32_t v_addr3 = (uint32_t)__cvta_generic_to_shared(&smem_V_buf[apply_swizzle_128B(k+3, tid)]);
            
            __nv_bfloat162 v0 = __ushort_as_bfloat162(*(uint32_t*)v_addr0);
            __nv_bfloat162 v1 = __ushort_as_bfloat162(*(uint32_t*)v_addr1);
            __nv_bfloat162 v2 = __ushort_as_bfloat162(*(uint32_t*)v_addr2);
            __nv_bfloat162 v3 = __ushort_as_bfloat162(*(uint32_t*)v_addr3);
            
            acc_o[0] = make_float2(acc_o[0].x + p_f2.x * __low2float(v0), acc_o[0].y + p_f2.y * __low2float(v0));
            acc_o[1] = make_float2(acc_o[1].x + p_f2.x * __high2float(v0), acc_o[1].y + p_f2.y * __high2float(v0));
            acc_o[2] = make_float2(acc_o[2].x + p_f2.x * __low2float(v1), acc_o[2].y + p_f2.y * __low2float(v1));
            acc_o[3] = make_float2(acc_o[3].x + p_f2.x * __high2float(v1), acc_o[3].y + p_f2.y * __high2float(v1));
        }
        
        __syncthreads(); 
    }
    
    float out_val0_x = acc_o[0].x;
    float out_val0_y = acc_o[0].y;
    float out_val1_x = acc_o[1].x;
    float out_val1_y = acc_o[1].y;
    float out_val2_x = acc_o[2].x;
    float out_val2_y = acc_o[2].y;
    float out_val3_x = acc_o[3].x;
    float out_val3_y = acc_o[3].y;
    
    if (smem_l[tid / 32] > 0.0f) {
        out_val0_x /= smem_l[tid / 32];
        out_val0_y /= smem_l[tid / 32];
        out_val1_x /= smem_l[tid / 32];
        out_val1_y /= smem_l[tid / 32];
        out_val2_x /= smem_l[tid / 32];
        out_val2_y /= smem_l[tid / 32];
        out_val3_x /= smem_l[tid / 32];
        out_val3_y /= smem_l[tid / 32];
    }
    
    __nv_bfloat16* out = g_O + q_base * 128;
    if (q_base < S) {
        *(uint32_t*)&out[tid] = __bfloat1622ushort(__float2bfloat162(out_val0_x, out_val0_y));
        *(uint32_t*)&out[tid + 1] = __bfloat1622ushort(__float2bfloat162(out_val1_x, out_val1_y));
        *(uint32_t*)&out[tid + 2] = __bfloat1622ushort(__float2bfloat162(out_val2_x, out_val2_y));
        *(uint32_t*)&out[tid + 3] = __bfloat1622ushort(__float2bfloat162(out_val3_x, out_val3_y));
    }
    
    if (tid == 0) {
        if (q_base < S) {
            g_LSE[q_base] = smem_m_prev[tid / 32] + __logf(smem_l[tid / 32]);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, 
         tvm::ffi::TensorView V, tvm::ffi::TensorView O, 
         tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    if (S == 0) return;
    
    int B_H = B * H;
    int smem_size = 134144; 
    
    CUtensorMap tma_Q, tma_K, tma_V;
    
    CUresult res;
    res = create_tma_2d_descriptor_2B(&tma_Q, static_cast<const void*>(Q.data_ptr()), 
        D, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA Q failed\n"); exit(1); }
    
    res = create_tma_2d_descriptor_2B(&tma_K, static_cast<const void*>(K.data_ptr()), 
        D, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA K failed\n"); exit(1); }
    
    res = create_tma_2d_descriptor_2B(&tma_V, static_cast<const void*>(V.data_ptr()), 
        D, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA V failed\n"); exit(1); }
    
    CUDA_CHECK(cudaFuncSetAttribute(
        mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    dim3 grid(1, (S + 3) / 4, B_H);
    dim3 block(128);
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    mha_kernel<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V,
        static_cast<__nv_bfloat16*>(O.data_ptr()),
        static_cast<float*>(LSE.data_ptr()),
        S, B_H);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha::run);

}  // namespace tvm_ffi_mha