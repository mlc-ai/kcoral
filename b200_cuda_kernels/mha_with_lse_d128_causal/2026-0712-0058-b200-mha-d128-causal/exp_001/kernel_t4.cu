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

__device__ __forceinline__ uint32_t get_tmem_addr(uint32_t tmem_base, int lane, int col) {
    uint32_t base_col = tmem_base & 0xFFFF;
    uint32_t base_lane = (tmem_base >> 16) & 0xFFFF;
    return ((base_lane + lane) << 16) | (base_col + col);
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
        int chunk_x = i / 8;
        int chunk_y = idx % 8;
        int swizzled_chunk = chunk_x ^ chunk_y;
        int swizzled_i = swizzled_chunk * 8 + (i % 8);
        smem_f4[idx * 16 + swizzled_i] = val;
    }
}

__device__ __forceinline__ void gemm_QK_T_128x128x128(uint32_t tmem_S_base, SwizzledStorage& Q_storage, SwizzledStorage& K_storage) {
    for (int k = 0; k < 8; ++k) {
        uint64_t desc_q = make_smem_desc_cg1(Q_storage.ptr + k * 16, 1, 1024, true);
        uint64_t desc_k = make_smem_desc_cg1(K_storage.ptr + k * 16, 1, 1024, true);
        uint32_t accum = 0;
        for (int i = 0; i < 4; ++i) {
            uint32_t idesc_qkt = make_instr_desc_fn_cg1(128, 128);
            uint32_t addr_S = get_tmem_addr(tmem_S_base, threadIdx.x, accum ? 0 : (i * 16));
            umma_f16_cg1_fn(addr_S, desc_q, desc_k, idesc_qkt, accum);
            accum = 1;
        }
    }
}

__device__ __forceinline__ void gemm_PV_128x128x128(uint32_t tmem_O_base, SwizzledStorage& P_storage, SwizzledStorage& V_storage) {
    for (int k = 0; k < 8; ++k) {
        uint64_t desc_p = make_smem_desc_cg1(P_storage.ptr + k * 16, 1, 1024, true);
        uint64_t desc_v = make_smem_desc_cg1(V_storage.ptr + k * 16 * 128, 16384, 1024, true);
        uint32_t accum = 0;
        for (int i = 0; i < 4; ++i) {
            uint32_t idesc_pv = make_instr_desc_fn_cg1_b_major_1(128, 128);
            uint32_t addr_O = get_tmem_addr(tmem_O_base, threadIdx.x, accum ? 0 : (i * 16));
            umma_f16_cg1_fn(addr_O, desc_p, desc_v, idesc_pv, accum);
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
    extern __shared__ __align__(128) char smem[];
    SwizzledStorage Q_storage0 { (__nv_bfloat16*)smem };                   
    SwizzledStorage Q_storage1 { (__nv_bfloat16*)(smem + 32768) };         
    SwizzledStorage K_storage   { (__nv_bfloat16*)(smem + 65536) };         
    SwizzledStorage V_storage   { (__nv_bfloat16*)(smem + 98304) };         
    SwizzledStorage P_storage   { (__nv_bfloat16*)(smem + 131072) };        
    uint64_t* bar0 = (uint64_t*)(smem + 163840);                      
    uint64_t* bar1 = (uint64_t*)(smem + 163848);                      

    uint32_t tmem_S_addr, tmem_O_base0, tmem_O_base1;
    if (threadIdx.x == 0) {
        tmem_alloc_fn_cg1(&tmem_S_addr, 128);
        tmem_alloc_fn_cg1(&tmem_O_base0, 128);
        tmem_alloc_fn_cg1(&tmem_O_base1, 128);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar0, 1);
        init_smem_barrier_fn(bar1, 1);
    }
    __syncthreads();
    
    int total_bh = gridDim.x / ((S + 127) / 128); // Safe assumption since gridDim is calculated dynamically
    int seq_tile_idx = blockIdx.x / total_bh;
    int bh_idx = blockIdx.x % total_bh;
    int b = bh_idx / H;
    int h = bh_idx % H;

    int s_offset_q0 = seq_tile_idx * 256;
    int s_offset_q1 = s_offset_q0 + 128;
    
    float global_max0 = -INFINITY, global_max1 = -INFINITY;
    float global_sum0 = 0.0f, global_sum1 = 0.0f;
    float scale = 1.0f / sqrtf(128.0f);
    int phase_bar0 = 0, phase_bar1 = 0;
    
    int q_idx0 = s_offset_q0 + threadIdx.x;
    int q_idx1 = s_offset_q1 + threadIdx.x;
    
    load_tile_swizzled(Q_storage0, Q_gmem, b, h, s_offset_q0, S);
    load_tile_swizzled(Q_storage1, Q_gmem, b, h, s_offset_q1, S);
    __syncthreads();

    int max_k = max(s_offset_q0, s_offset_q1) + 128;
    if (max_k > S) max_k = S;

    for (int s_offset_k = 0; s_offset_k < max_k; s_offset_k += 128) {
        load_tile_swizzled(K_storage, K_gmem, b, h, s_offset_k, S);
        load_tile_swizzled(V_storage, V_gmem, b, h, s_offset_k, S);
        __syncthreads();
        
        fence_proxy_async_fn();
        
        if (threadIdx.x < 128) {
            if (s_offset_k <= s_offset_q0 + 127) {
                if (threadIdx.x == 0) {
                    mbarrier_arrive_and_expect_tx_fn(bar0, 1);
                    gemm_QK_T_128x128x128(tmem_S_addr, Q_storage0, K_storage);
                    umma_commit_1sm_fn_cg1(bar0);
                }
                mbarrier_wait_fn(bar0, phase_bar0);
                phase_bar0 ^= 1;
            }
            
            float my_max = -INFINITY;
            if (s_offset_k <= s_offset_q0 + 127) {
                for (int col = 0; col < 128; col += 4) {
                    uint32_t r0, r1, r2, r3;
                    tmem_load_4x_fn(get_tmem_addr(tmem_S_addr, threadIdx.x, col), &r0, &r1, &r2, &r3);
                    tmem_load_fence_fn();
                    
                    float f0 = __uint_as_float(r0) * scale;
                    float f1 = __uint_as_float(r1) * scale;
                    float f2 = __uint_as_float(r2) * scale;
                    float f3 = __uint_as_float(r3) * scale;
                    
                    int k_idx0 = s_offset_k + col;
                    if (k_idx0 > q_idx0 || k_idx0 >= S) f0 = -INFINITY;
                    if (k_idx0 + 1 > q_idx0 || k_idx0 + 1 >= S) f1 = -INFINITY;
                    if (k_idx0 + 2 > q_idx0 || k_idx0 + 2 >= S) f2 = -INFINITY;
                    if (k_idx0 + 3 > q_idx0 || k_idx0 + 3 >= S) f3 = -INFINITY;
                    
                    my_max = fmaxf(my_max, fmaxf(fmaxf(f0, f1), fmaxf(f2, f3)));
                }
            }
            
            float row_max = my_max;
            row_max = fmaxf(row_max, __shfl_xor_sync(0xFFFFFFFF, row_max, 1));
            row_max = fmaxf(row_max, __shfl_xor_sync(0xFFFFFFFF, row_max, 2));
            row_max = fmaxf(row_max, __shfl_xor_sync(0xFFFFFFFF, row_max, 4));
            row_max = fmaxf(row_max, __shfl_xor_sync(0xFFFFFFFF, row_max, 8));
            row_max = fmaxf(row_max, __shfl_xor_sync(0xFFFFFFFF, row_max, 16));
            
            float curr_max = fmaxf(global_max0, row_max);
            float curr_sum = global_sum0 * (curr_max > -INFINITY && global_max0 > -INFINITY ? expf(global_max0 - curr_max) : 0.0f);
            
            float my_sum = 0;
            for (int col = 0; col < 128; col += 4) {
                uint32_t r0, r1, r2, r3;
                tmem_load_4x_fn(get_tmem_addr(tmem_S_addr, threadIdx.x, col), &r0, &r1, &r2, &r3);
                tmem_load_fence_fn();
                
                float f0 = __uint_as_float(r0) * scale;
                float f1 = __uint_as_float(r1) * scale;
                float f2 = __uint_as_float(r2) * scale;
                float f3 = __uint_as_float(r3) * scale;
                
                int k_idx0 = s_offset_k + col;
                if (k_idx0 > q_idx0 || k_idx0 >= S) f0 = -INFINITY;
                if (k_idx0 + 1 > q_idx0 || k_idx0 + 1 >= S) f1 = -INFINITY;
                if (k_idx0 + 2 > q_idx0 || k_idx0 + 2 >= S) f2 = -INFINITY;
                if (k_idx0 + 3 > q_idx0 || k_idx0 + 3 >= S) f3 = -INFINITY;
                
                float p0 = (f0 > -INFINITY) ? expf(f0 - curr_max) : 0.0f;
                float p1 = (f1 > -INFINITY) ? expf(f1 - curr_max) : 0.0f;
                float p2 = (f2 > -INFINITY) ? expf(f2 - curr_max) : 0.0f;
                float p3 = (f3 > -INFINITY) ? expf(f3 - curr_max) : 0.0f;
                
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
            global_sum0 = curr_sum;
            global_max0 = curr_max;
            
            __syncthreads();
            fence_proxy_async_fn();
            
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(bar0, 1);
                gemm_PV_128x128x128(tmem_O_base0, P_storage, V_storage);
                umma_commit_1sm_fn_cg1(bar0);
            }
            mbarrier_wait_fn(bar0, phase_bar0);
            phase_bar0 ^= 1;
        } 
        
        __syncthreads(); // Critical sync to prevent Q1 from overwriting P_storage before Q0 finishes reading it
        
        if (threadIdx.x >= 128) {
            if (s_offset_k <= s_offset_q1 + 127) {
                if (threadIdx.x == 128) {
                    mbarrier_arrive_and_expect_tx_fn(bar1, 1);
                    gemm_QK_T_128x128x128(tmem_S_addr, Q_storage1, K_storage);
                    umma_commit_1sm_fn_cg1(bar1);
                }
                mbarrier_wait_fn(bar1, phase_bar1);
                phase_bar1 ^= 1;
            }
            
            float my_max = -INFINITY;
            if (s_offset_k <= s_offset_q1 + 127) {
                for (int col = 0; col < 128; col += 4) {
                    uint32_t r0, r1, r2, r3;
                    tmem_load_4x_fn(get_tmem_addr(tmem_S_addr, threadIdx.x, col), &r0, &r1, &r2, &r3);
                    tmem_load_fence_fn();
                    
                    float f0 = __uint_as_float(r0) * scale;
                    float f1 = __uint_as_float(r1) * scale;
                    float f2 = __uint_as_float(r2) * scale;
                    float f3 = __uint_as_float(r3) * scale;
                    
                    int k_idx0 = s_offset_k + col;
                    if (k_idx0 > q_idx1 || k_idx0 >= S) f0 = -INFINITY;
                    if (k_idx0 + 1 > q_idx1 || k_idx0 + 1 >= S) f1 = -INFINITY;
                    if (k_idx0 + 2 > q_idx1 || k_idx0 + 2 >= S) f2 = -INFINITY;
                    if (k_idx0 + 3 > q_idx1 || k_idx0 + 3 >= S) f3 = -INFINITY;
                    
                    my_max = fmaxf(my_max, fmaxf(fmaxf(f0, f1), fmaxf(f2, f3)));
                }
            }
            
            float row_max = my_max;
            row_max = fmaxf(row_max, __shfl_xor_sync(0xFFFFFFFF, row_max, 1));
            row_max = fmaxf(row_max, __shfl_xor_sync(0xFFFFFFFF, row_max, 2));
            row_max = fmaxf(row_max, __shfl_xor_sync(0xFFFFFFFF, row_max, 4));
            row_max = fmaxf(row_max, __shfl_xor_sync(0xFFFFFFFF, row_max, 8));
            row_max = fmaxf(row_max, __shfl_xor_sync(0xFFFFFFFF, row_max, 16));
            
            float curr_max = fmaxf(global_max1, row_max);
            float curr_sum = global_sum1 * (curr_max > -INFINITY && global_max1 > -INFINITY ? expf(global_max1 - curr_max) : 0.0f);
            
            float my_sum = 0;
            for (int col = 0; col < 128; col += 4) {
                uint32_t r0, r1, r2, r3;
                tmem_load_4x_fn(get_tmem_addr(tmem_S_addr, threadIdx.x, col), &r0, &r1, &r2, &r3);
                tmem_load_fence_fn();
                
                float f0 = __uint_as_float(r0) * scale;
                float f1 = __uint_as_float(r1) * scale;
                float f2 = __uint_as_float(r2) * scale;
                float f3 = __uint_as_float(r3) * scale;
                
                int k_idx0 = s_offset_k + col;
                if (k_idx0 > q_idx1 || k_idx0 >= S) f0 = -INFINITY;
                if (k_idx0 + 1 > q_idx1 || k_idx0 + 1 >= S) f1 = -INFINITY;
                if (k_idx0 + 2 > q_idx1 || k_idx0 + 2 >= S) f2 = -INFINITY;
                if (k_idx0 + 3 > q_idx1 || k_idx0 + 3 >= S) f3 = -INFINITY;
                
                float p0 = (f0 > -INFINITY) ? expf(f0 - curr_max) : 0.0f;
                float p1 = (f1 > -INFINITY) ? expf(f1 - curr_max) : 0.0f;
                float p2 = (f2 > -INFINITY) ? expf(f2 - curr_max) : 0.0f;
                float p3 = (f3 > -INFINITY) ? expf(f3 - curr_max) : 0.0f;
                
                my_sum += p0 + p1 + p2 + p3;
                
                uint32_t p01 = pack_bf16_fn(p0, p1);
                uint32_t p23 = pack_bf16_fn(p2, p3);
                *(uint32_t*)&P_storage(threadIdx.x % 128, col + 0) = p01;
                *(uint32_t*)&P_storage(threadIdx.x % 128, col + 2) = p23;
            }
            
            float row_sum = my_sum;
            row_sum += __shfl_xor_sync(0xFFFFFFFF, row_sum, 1);
            row_sum += __shfl_xor_sync(0xFFFFFFFF, row_sum, 2);
            row_sum += __shfl_xor_sync(0xFFFFFFFF, row_sum, 4);
            row_sum += __shfl_xor_sync(0xFFFFFFFF, row_sum, 8);
            row_sum += __shfl_xor_sync(0xFFFFFFFF, row_sum, 16);
            
            curr_sum += row_sum;
            global_sum1 = curr_sum;
            global_max1 = curr_max;
            
            __syncthreads();
            fence_proxy_async_fn();
            
            if (threadIdx.x == 128) {
                mbarrier_arrive_and_expect_tx_fn(bar1, 1);
                gemm_PV_128x128x128(tmem_O_base1, P_storage, V_storage);
                umma_commit_1sm_fn_cg1(bar1);
            }
            mbarrier_wait_fn(bar1, phase_bar1);
            phase_bar1 ^= 1;
        }
        __syncthreads();
    }
    
    __syncthreads();
    
    if (threadIdx.x < 128) {
        if (q_idx0 < S) {
            float lse_val = INFINITY;
            if (global_sum0 > 0.0f) {
                lse_val = global_max0 + logf(global_sum0);
            }
            LSE_gmem[(b * H + h) * S + q_idx0] = lse_val;
            
            for (int col = 0; col < 128; col += 4) {
                uint32_t r0, r1, r2, r3;
                tmem_load_4x_fn(get_tmem_addr(tmem_O_base0, threadIdx.x, col), &r0, &r1, &r2, &r3);
                tmem_load_fence_fn();
                
                float f0 = __uint_as_float(r0);
                float f1 = __uint_as_float(r1);
                float f2 = __uint_as_float(r2);
                float f3 = __uint_as_float(r3);
                
                if (global_sum0 > 0.0f) {
                    f0 /= global_sum0;
                    f1 /= global_sum0;
                    f2 /= global_sum0;
                    f3 /= global_sum0;
                }
                
                O_gmem[(b * H + h) * S * 128 + q_idx0 * 128 + col + 0] = __float2bfloat16(f0);
                O_gmem[(b * H + h) * S * 128 + q_idx0 * 128 + col + 1] = __float2bfloat16(f1);
                O_gmem[(b * H + h) * S * 128 + q_idx0 * 128 + col + 2] = __float2bfloat16(f2);
                O_gmem[(b * H + h) * S * 128 + q_idx0 * 128 + col + 3] = __float2bfloat16(f3);
            }
        }
    } else {
        if (q_idx1 < S) {
            float lse_val = INFINITY;
            if (global_sum1 > 0.0f) {
                lse_val = global_max1 + logf(global_sum1);
            }
            LSE_gmem[(b * H + h) * S + q_idx1] = lse_val;
            
            for (int col = 0; col < 128; col += 4) {
                uint32_t r0, r1, r2, r3;
                tmem_load_4x_fn(get_tmem_addr(tmem_O_base1, threadIdx.x, col), &r0, &r1, &r2, &r3);
                tmem_load_fence_fn();
                
                float f0 = __uint_as_float(r0);
                float f1 = __uint_as_float(r1);
                float f2 = __uint_as_float(r2);
                float f3 = __uint_as_float(r3);
                
                if (global_sum1 > 0.0f) {
                    f0 /= global_sum1;
                    f1 /= global_sum1;
                    f2 /= global_sum1;
                    f3 /= global_sum1;
                }
                
                O_gmem[(b * H + h) * S * 128 + q_idx1 * 128 + col + 0] = __float2bfloat16(f0);
                O_gmem[(b * H + h) * S * 128 + q_idx1 * 128 + col + 1] = __float2bfloat16(f1);
                O_gmem[(b * H + h) * S * 128 + q_idx1 * 128 + col + 2] = __float2bfloat16(f2);
                O_gmem[(b * H + h) * S * 128 + q_idx1 * 128 + col + 3] = __float2bfloat16(f3);
            }
        }
    }
    
    if (threadIdx.x == 0) {
        tmem_dealloc_fn_cg1(tmem_S_addr, 128);
        tmem_dealloc_fn_cg1(tmem_O_base0, 128);
        tmem_dealloc_fn_cg1(tmem_O_base1, 128);
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
    int64_t threads = 256;
    
    int smem_size = 160016;
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