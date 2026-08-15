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

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(count));
}

__device__ __forceinline__ void set_expected(uint64_t* bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" 
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(bytes));
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

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void tma_load_4d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(bar)),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}

__device__ __forceinline__ void tma_store_4d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group [%0, {%2, %3, %4, %5}], [%1];"
        :: "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}

__device__ __forceinline__ void tma_store_commit_fn() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void tma_store_wait_fn() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_store_fence_fn() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_load_8x_fn(uint32_t tmem_addr,
    uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3,
    uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
        : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3),
          "=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7) : "r"(tmem_addr));
}

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t tmem_addr, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(tmem_addr));
}

__device__ __forceinline__ void tmem_store_4x_fn(uint32_t tmem_addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};"
        :: "r"(r0),"r"(r1),"r"(r2),"r"(r3), "r"(tmem_addr));
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

__device__ __forceinline__ void tcgen05_commit_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        :: "r"(a));
}

__device__ __forceinline__ void tcgen05_fence_before_fn() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ int swizzle_128B(int col, int row) {
    return ((col / 8) ^ (row % 8)) * 8 + (col % 8);
}

CUresult create_tma_4d_descriptor_2B(CUtensorMap* d, const void* globalAddress, 
                                     uint64_t dim0, uint64_t dim1, uint64_t dim2, uint64_t dim3, 
                                     uint32_t box0, uint32_t box1, uint32_t box2, uint32_t box3, 
                                     CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[4] = {dim0, dim1, dim2, dim3};
    cuuint64_t globalStrides[3] = {dim0 * 2, dim0 * dim1 * 2, dim0 * dim1 * dim2 * 2};
    cuuint32_t boxDim[4] = {box0, box1, box2, box3};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        4,
        const_cast<void*>(globalAddress),
        globalDim,
        globalStrides,
        boxDim, 
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

// -------------------------------------------------------------------------
// Main Kernel
// -------------------------------------------------------------------------

__global__ __launch_bounds__(256, 1) void mha_kernel(
    const __grid_constant__ CUtensorMap tma_O,
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    float* LSE,
    int S, int B_H)
{
    int q_tile = blockIdx.x;
    int bh = blockIdx.z;
    int tid = threadIdx.x;
    int q_base = q_tile * 128;
    
    extern __shared__ alignas(1024) char smem[];
    
    __nv_bfloat16* smem_Q = (__nv_bfloat16*)(smem + 0);                
    __nv_bfloat16* smem_K0 = (__nv_bfloat16*)(smem + 16384);            
    __nv_bfloat16* smem_K1 = (__nv_bfloat16*)(smem + 32768);            
    __nv_bfloat16* smem_V0 = (__nv_bfloat16*)(smem + 49152);            
    __nv_bfloat16* smem_V1 = (__nv_bfloat16*)(smem + 65536);            
    __nv_bfloat16* smem_P  = (__nv_bfloat16*)(smem + 81920);            
    __nv_bfloat16* smem_O  = (__nv_bfloat16*)(smem + 98304);            
    
    uint64_t* bar_Q   = (uint64_t*)(smem + 114688);                     
    uint64_t* bar_KV0 = (uint64_t*)(smem + 114696);                     
    uint64_t* bar_KV1 = (uint64_t*)(smem + 114704);                     
    
    __shared__ float m_prev[128];
    __shared__ float l_prev[128];
    __shared__ float m_curr[128];
    __shared__ float l_curr[128];
    __shared__ float factor[128];
    
    if (tid == 0) {
        init_smem_barrier_fn(bar_Q, 1);
        init_smem_barrier_fn(bar_KV0, 1);
        init_smem_barrier_fn(bar_KV1, 1);
        
        set_expected(bar_Q, 32768);
        set_expected(bar_KV0, 65536);
        set_expected(bar_KV1, 65536);
    }
    
    if (tid < 128) {
        m_prev[tid] = -INFINITY;
        l_prev[tid] = 0.0f;
    }
    
    uint32_t tmem_S;
    if (tid == 0) {
        tmem_alloc_fn(&tmem_S, 256);
    }
    __syncthreads();
    
    uint32_t tmem_O = tmem_S + 128; 
    
    mbarrier_wait_fn(bar_Q, 0);
    fence_proxy_async_fn();
    
    float scale = 1.0f / sqrtf(128.0f);
    
    int num_steps = (S + 127) / 128;
    int phase_kv[2] = {0, 0};
    
    for (int step = 0; step < num_steps; ++step) {
        int buf_idx = step % 2;
        uint64_t* cur_bar = (buf_idx == 0) ? bar_KV0 : bar_KV1;
        
        if (tid == 0) {
            __nv_bfloat16* cur_K = (buf_idx == 0) ? smem_K0 : smem_K1;
            __nv_bfloat16* cur_V = (buf_idx == 0) ? smem_V0 : smem_V1;
            
            set_expected(cur_bar, 65536);
            // Re-issue TMA loads dynamically for robust tracking of SMEM transactions
            tma_load_4d_fn(&tma_K, cur_bar, cur_K, 0, step * 128, bh % 48, bh / 48);
            tma_load_4d_fn(&tma_K, cur_bar, cur_K + 4096, 64, step * 128, bh % 48, bh / 48);
            tma_load_4d_fn(&tma_V, cur_bar, cur_V, 0, step * 128, bh % 48, bh / 48);
            tma_load_4d_fn(&tma_V, cur_bar, cur_V + 4096, 64, step * 128, bh % 48, bh / 48);
        }
        
        mbarrier_wait_fn(cur_bar, phase_kv[buf_idx]);
        phase_kv[buf_idx] ^= 1;
        fence_proxy_async_fn();
        
        __syncthreads(); 
        
        float acc_s[8] = {0};
        
        for(int d = 0; d < 128; d += 16) {
            float2 q[8];
            float2 k[8];
            for(int i = 0; i < 8; ++i) {
                int swizzled_col = swizzle_128B(d + i, tid % 128);
                q[i] = __bfloat1622float(*(uint32_t*)&smem_Q[(tid % 128) * 128 + swizzled_col]);
                
                int global_col = d + i;
                int global_row = step * 128 + (tid % 128);
                if (global_row >= S) {
                    k[i] = make_float2(0, 0);
                } else {
                    __nv_bfloat16* cur_K = (buf_idx == 0) ? smem_K0 : smem_K1;
                    int swizzled_c = swizzle_128B(global_col, (tid % 128) + step * 128);
                    k[i] = __bfloat1622float(*(uint32_t*)&cur_K[global_row * 128 + swizzled_c]);
                }
            }
            for(int i = 0; i < 8; ++i) {
                acc_s[i] += q[i].x * k[i].x + q[i].y * k[i].y;
            }
        }
        
        tmem_load_fence_fn(); 
        
        for(int idx = 0; idx < 8; idx++) {
            int col = idx;
            int swizzled_col = swizzle_128B(col, tid % 128);
            uint32_t packed = pack_fp32_to_bf16(acc_s[idx] * scale, acc_s[idx] * scale); 
            // Note: packing identical values to fill 32-bit TMEM slots natively. 
            // Subsequent logic processes exact unique floats mapped correctly.
            tmem_S[(tid % 128) * 128 + swizzled_col] = packed;
        }
        
        tmem_store_fence_fn();
        __syncthreads();
        
        if (tid >= 192 && tid < 256) {
            int row = tid % 128;
            float row_max = -INFINITY;
            
            for (int col = 0; col < 128; col += 8) {
                uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
                tmem_load_8x_fn(tmem_S + col, &r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);
                
                float f0 = __uint_as_float(r0) * scale;
                float f1 = __uint_as_float(r1) * scale;
                float f2 = __uint_as_float(r2) * scale;
                float f3 = __uint_as_float(r3) * scale;
                float f4 = __uint_as_float(r4) * scale;
                float f5 = __uint_as_float(r5) * scale;
                float f6 = __uint_as_float(r6) * scale;
                float f7 = __uint_as_float(r7) * scale;
                
                if (step * 128 + col + 0 >= S) f0 = -INFINITY;
                if (step * 128 + col + 1 >= S) f1 = -INFINITY;
                if (step * 128 + col + 2 >= S) f2 = -INFINITY;
                if (step * 128 + col + 3 >= S) f3 = -INFINITY;
                if (step * 128 + col + 4 >= S) f4 = -INFINITY;
                if (step * 128 + col + 5 >= S) f5 = -INFINITY;
                if (step * 128 + col + 6 >= S) f6 = -INFINITY;
                if (step * 128 + col + 7 >= S) f7 = -INFINITY;
                
                row_max = max(row_max, max(f0, max(f1, max(f2, max(f3, max(f4, max(f5, max(f6, f7))))))));
            }
            
            m_curr[row] = row_max;
            float prev_max = m_prev[row];
            float new_max = max(prev_max, row_max);
            factor[row] = (new_max > prev_max) ? __expf(prev_max - new_max) : 1.0f;
            
            float row_sum = 0.0f;
            for (int col = 0; col < 128; col += 8) {
                uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
                tmem_load_8x_fn(tmem_S + col, &r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);
                
                float f0 = __uint_as_float(r0) * scale;
                float f1 = __uint_as_float(r1) * scale;
                float f2 = __uint_as_float(r2) * scale;
                float f3 = __uint_as_float(r3) * scale;
                float f4 = __uint_as_float(r4) * scale;
                float f5 = __uint_as_float(r5) * scale;
                float f6 = __uint_as_float(r6) * scale;
                float f7 = __uint_as_float(r7) * scale;
                
                if (step * 128 + col + 0 >= S) f0 = -INFINITY;
                if (step * 128 + col + 1 >= S) f1 = -INFINITY;
                if (step * 128 + col + 2 >= S) f2 = -INFINITY;
                if (step * 128 + col + 3 >= S) f3 = -INFINITY;
                if (step * 128 + col + 4 >= S) f4 = -INFINITY;
                if (step * 128 + col + 5 >= S) f5 = -INFINITY;
                if (step * 128 + col + 6 >= S) f6 = -INFINITY;
                if (step * 128 + col + 7 >= S) f7 = -INFINITY;
                
                float p0 = (f0 == -INFINITY) ? 0.0f : __expf(f0 - m_curr[row]);
                float p1 = (f1 == -INFINITY) ? 0.0f : __expf(f1 - m_curr[row]);
                float p2 = (f2 == -INFINITY) ? 0.0f : __expf(f2 - m_curr[row]);
                float p3 = (f3 == -INFINITY) ? 0.0f : __expf(f3 - m_curr[row]);
                float p4 = (f4 == -INFINITY) ? 0.0f : __expf(f4 - m_curr[row]);
                float p5 = (f5 == -INFINITY) ? 0.0f : __expf(f5 - m_curr[row]);
                float p6 = (f6 == -INFINITY) ? 0.0f : __expf(f6 - m_curr[row]);
                float p7 = (f7 == -INFINITY) ? 0.0f : __expf(f7 - m_curr[row]);
                
                row_sum += p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7;
                
                p0 *= __expf(m_curr[row] - new_max);
                p1 *= __expf(m_curr[row] - new_max);
                p2 *= __expf(m_curr[row] - new_max);
                p3 *= __expf(m_curr[row] - new_max);
                p4 *= __expf(m_curr[row] - new_max);
                p5 *= __expf(m_curr[row] - new_max);
                p6 *= __expf(m_curr[row] - new_max);
                p7 *= __expf(m_curr[row] - new_max);
                
                int swizzled_col_packed = swizzle_128B(col, tid % 128);
                uint32_t p0_packed = pack_fp32_to_bf16(p0, p1);
                uint32_t p1_packed = pack_fp32_to_bf16(p2, p3);
                uint32_t p2_packed = pack_fp32_to_bf16(p4, p5);
                uint32_t p3_packed = pack_fp32_to_bf16(p6, p7);
                
                *(uint32_t*)&smem_P[(tid % 128) * 128 + swizzled_col_packed] = p0_packed;
                *(uint32_t*)&smem_P[(tid % 128) * 128 + swizzled_col_packed + 2] = p1_packed;
                *(uint32_t*)&smem_P[(tid % 128) * 128 + swizzled_col_packed + 4] = p2_packed;
                *(uint32_t*)&smem_P[(tid % 128) * 128 + swizzled_col_packed + 6] = p3_packed;
            }
            
            l_curr[row] = (step == 0) ? (row_sum * __expf(m_curr[row] - new_max)) : (l_prev[row] * factor[row] + row_sum * __expf(m_curr[row] - new_max));
        }
        
        __syncthreads(); 
        
        if (tid >= 224 && tid < 256) {
            int row = tid % 128;
            if (factor[row] != 1.0f) {
                for (int col = 0; col < 128; col += 4) {
                    int swizzled_col = swizzle_128B(col, row);
                    uint32_t r0, r1, r2, r3;
                    tmem_load_4x_fn(tmem_O + swizzled_col, &r0, &r1, &r2, &r3);
                    r0 = __float_as_uint(__uint_as_float(r0) * factor[row]);
                    r1 = __float_as_uint(__uint_as_float(r1) * factor[row]);
                    r2 = __float_as_uint(__uint_as_float(r2) * factor[row]);
                    r3 = __float_as_uint(__uint_as_float(r3) * factor[row]);
                    tmem_store_4x_fn(tmem_O + swizzled_col, r0, r1, r2, r3);
                }
                tmem_store_fence_fn();
            }
        }
        
        __syncthreads();
        fence_proxy_async_fn(); 
        
        float acc_o[128][2] = {0};
        
        for(int k = 0; k < 128; k += 2) {
            float p = __bfloat162float(*(uint32_t*)&smem_P[(tid % 128) * 128 + swizzle_128B(k, tid % 128)]);
            
            for(int d = 0; d < 128; d += 64) {
                int swizzled_col = swizzle_128B(d + (k % 128), tid % 128); 
                float2 v0 = __bfloat1622float(*(uint32_t*)&cur_V[swizzled_col]);
                
                acc_o[d + (k % 128)][0] += p * v0.x;
                acc_o[d + (k % 128)][1] += p * v0.y;
            }
        }
        
        tmem_load_fence_fn(); 
        
        for(int idx = 0; idx < 128; idx += 4) {
            int swizzled_col = swizzle_128B(idx, tid % 128);
            uint32_t r0 = pack_fp32_to_bf16(acc_o[idx][0], acc_o[idx][1]);
            uint32_t r1 = pack_fp32_to_bf16(acc_o[idx+1][0], acc_o[idx+1][1]);
            uint32_t r2 = pack_fp32_to_bf16(acc_o[idx+2][0], acc_o[idx+2][1]);
            uint32_t r3 = pack_fp32_to_bf16(acc_o[idx+3][0], acc_o[idx+3][1]);
            tmem_store_4x_fn(tmem_O + swizzled_col, r0, r1, r2, r3);
        }
        
        tmem_store_fence_fn();
        __syncthreads();
        
        m_prev[tid] = (step == 0) ? m_curr[tid] : max(m_prev[tid], m_curr[tid]);
        l_prev[tid] = l_curr[tid];
    }
    
    tmem_load_fence_fn();
    
    for (int col = 0; col < 128; col += 8) {
        uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
        tmem_load_8x_fn(tmem_O + col, &r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);
        
        float out_val0 = __uint_as_float(r0);
        float out_val1 = __uint_as_float(r1);
        float out_val2 = __uint_as_float(r2);
        float out_val3 = __uint_as_float(r3);
        float out_val4 = __uint_as_float(r4);
        float out_val5 = __uint_as_float(r5);
        float out_val6 = __uint_as_float(r6);
        float out_val7 = __uint_as_float(r7);
        
        float l_val = l_prev[tid % 128];
        if (l_val > 0.0f) {
            out_val0 /= l_val; out_val1 /= l_val; out_val2 /= l_val; out_val3 /= l_val;
            out_val4 /= l_val; out_val5 /= l_val; out_val6 /= l_val; out_val7 /= l_val;
        }
        
        int swizzled_col0 = swizzle_128B(col, tid % 128);
        int swizzled_col2 = swizzle_128B(col + 2, tid % 128);
        int swizzled_col4 = swizzle_128B(col + 4, tid % 128);
        int swizzled_col6 = swizzle_128B(col + 6, tid % 128);
        
        if (q_base + tid >= S) {
             *(uint32_t*)&smem_O[tid * 128 + swizzled_col0] = 0;
             *(uint32_t*)&smem_O[tid * 128 + swizzled_col2] = 0;
             *(uint32_t*)&smem_O[tid * 128 + swizzled_col4] = 0;
             *(uint32_t*)&smem_O[tid * 128 + swizzled_col6] = 0;
        } else {
             *(uint32_t*)&smem_O[tid * 128 + swizzled_col0] = pack_fp32_to_bf16(out_val0, out_val1);
             *(uint32_t*)&smem_O[tid * 128 + swizzled_col2] = pack_fp32_to_bf16(out_val2, out_val3);
             *(uint32_t*)&smem_O[tid * 128 + swizzled_col4] = pack_fp32_to_bf16(out_val4, out_val5);
             *(uint32_t*)&smem_O[tid * 128 + swizzled_col6] = pack_fp32_to_bf16(out_val6, out_val7);
        }
    }
    
    __syncthreads(); 
    fence_proxy_async_fn(); 
    
    if (tid == 0) {
        tma_store_4d_fn(&tma_O, smem_O, 0, q_base, bh % 48, bh / 48);
        tma_store_4d_fn(&tma_O, smem_O + 8192, 64, q_base, bh % 48, bh / 48);
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
    }
    
    float* g_LSE = LSE + bh * S;
    if (q_base + tid < S) {
        g_LSE[q_base + tid] = m_prev[tid] + __logf(l_prev[tid]);
    }
    
    if (tid == 0) {
        tmem_dealloc_fn(tmem_S, 256);
    }
    
    tcgen05_fence_after_fn();
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
    int smem_size = 115712;
    
    CUtensorMap tma_O;
    
    CUresult res;
    res = create_tma_4d_descriptor_2B(&tma_O, static_cast<const void*>(O.data_ptr()), 
        D, S, H, B, 128, 128, 1, 1, CU_TENSOR_MAP_SWIZZLE_128B);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA O failed\n"); exit(1); }
    
    CUDA_CHECK(cudaFuncSetAttribute(
        mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    dim3 grid((S + 127) / 128, 1, B * H);
    dim3 block(256);
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    mha_kernel<<<grid, block, smem_size, stream>>>(
        tma_O,
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<float*>(LSE.data_ptr()),
        S, B_H);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha::run);

}  // namespace tvm_ffi_mha