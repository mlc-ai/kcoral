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

#define CU_CHECK(call) do { \
    CUresult _e = (call); \
    if (_e != CUDA_SUCCESS) { \
        fprintf(stderr, "CU error %d at %s:%d\n", (int)_e, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", \
                cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)


// ===================================================================
// PTX Helper Functions
// ===================================================================

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

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_store_2d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%2, %3}], [%1];"
        :: "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_store_fence_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tma_store_commit_fn() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void tma_store_wait_fn() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void umma_f16_cg2_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ uint32_t make_tmem_addr_fn(uint32_t row, uint32_t col) {
    return (row << 16) | (col & 0xFFFF);
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc(void* smem_ptr, bool is_k_major, uint32_t k_dim) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint64_t d = (addr & 0x3FFFF) >> 4;
    
    uint32_t sbo = is_k_major ? 1024 : 1024;
    uint32_t lbo = is_k_major ? 1 : ((k_dim / 8) * 1024); 
    
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint64_t desc_k_major(void* ptr) { return make_smem_desc(ptr, true, 64); }
__device__ __forceinline__ uint64_t desc_mn_major(void* ptr) { return make_smem_desc(ptr, false, 64); }

__device__ __forceinline__ void gemm_128x64x64(uint32_t tmem_c_base, 
    uint64_t desc_a_0, uint64_t desc_b_0, uint32_t idesc_0, uint32_t accum) 
{
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        uint32_t tmem_c = tmem_c_base + (k * 256);
        uint64_t desc_a = desc_a_0 + (k * 64); 
        uint64_t desc_b = desc_b_0 + (k * 64); 
        umma_f16_cg2_fn(tmem_c, desc_a, desc_b, idesc_0, accum);
    }
}

__device__ __forceinline__ uint64_t gemm_128x64x64_nounroll(uint32_t tmem_c_base, 
    uint64_t desc_a_0, uint64_t desc_b_0, uint32_t idesc_0, uint32_t accum) 
{
    uint32_t tmem_c_final = 0;
    for (int k = 0; k < 4; ++k) {
        uint32_t tmem_c = tmem_c_base + (k * 256);
        uint64_t desc_a = desc_a_0 + (k * 64); 
        uint64_t desc_b = desc_b_0 + (k * 64); 
        tmem_c_final = tmem_c;
        umma_f16_cg2_fn(tmem_c, desc_a, desc_b, idesc_0, accum);
    }
    return tmem_c_final;
}

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void cluster_sync_fn() {
    asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n" ::: "memory");
}

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__device__ __forceinline__ __nv_bfloat16 swizzle_read(const __nv_bfloat16* smem, int row, int col) {
    int row_chunk = row % 8;
    int col_byte = col * 2;
    int col_chunk = col_byte / 16;
    int swizzled_col_byte = (row_chunk ^ col_chunk) * 16 + (col_byte % 16);
    int swizzled_col = swizzled_col_byte / 2;
    return smem[row * 64 + swizzled_col];
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ __forceinline__ void load_64x128(
    const CUtensorMap* tma, uint64_t* bar, __nv_bfloat16* smem,
    int32_t global_offset, uint32_t& phase) {
    int tid = threadIdx.x;
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar, 16384);
        tma_load_2d_fn(tma, bar, smem, 0, global_offset);
        tma_load_2d_fn(tma, bar, smem + 4096, 64, global_offset);
    }
    mbarrier_wait_fn(bar, phase);
    __syncthreads();
    phase ^= 1;
}

__device__ __forceinline__ void store_64x128(
    const CUtensorMap* tma, __nv_bfloat16* smem, int32_t global_offset) {
    int tid = threadIdx.x;
    if (tid == 0) {
        tma_store_2d_fn(tma, smem, 0, global_offset);
        tma_store_2d_fn(tma, smem + 4096, 64, global_offset);
        tma_store_commit_fn();
    }
    tma_store_wait_fn<0>();
    __syncthreads();
}

__device__ __forceinline__ void epilogue_64x128(__nv_bfloat16* smem_out) {
    int tid = threadIdx.x;
    uint32_t tmem_c_base = make_tmem_addr_fn(0, 0);
    for (uint32_t col = 0; col < 64; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_c_base + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t base = tid * 64 + col;
        smem_out[base + 0] = __float2bfloat16(__uint_as_float(r0));
        smem_out[base + 1] = __float2bfloat16(__uint_as_float(r1));
        smem_out[base + 2] = __float2bfloat16(__uint_as_float(r2));
        smem_out[base + 3] = __float2bfloat16(__uint_as_float(r3));
    }
    __syncthreads();
}

// ===================================================================
// Kernels
// ===================================================================

__global__ void __launch_bounds__(128, 2) kernel_fused(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const __grid_constant__ CUtensorMap tma_dQ,
    const __grid_constant__ CUtensorMap tma_dK,
    const __grid_constant__ CUtensorMap tma_dV,
    const float* L, int S, float scale) 
{
    extern __shared__ __align__(1024) uint8_t smem_pool[];
    uint32_t cta_offset = cluster_rank_fn() * 147456;

    __nv_bfloat16* q_tile_0 = (__nv_bfloat16*)(smem_pool + cta_offset + 0);
    __nv_bfloat16* q_tile_1 = (__nv_bfloat16*)(smem_pool + cta_offset + 8192);
    __nv_bfloat16* do_tile_0 = (__nv_bfloat16*)(smem_pool + cta_offset + 16384);
    __nv_bfloat16* do_tile_1 = (__nv_bfloat16*)(smem_pool + cta_offset + 24576);
    __nv_bfloat16* o_tile_0 = (__nv_bfloat16*)(smem_pool + cta_offset + 32768);
    __nv_bfloat16* o_tile_1 = (__nv_bfloat16*)(smem_pool + cta_offset + 40960);
    __nv_bfloat16* k_tile_0 = (__nv_bfloat16*)(smem_pool + cta_offset + 49152);
    __nv_bfloat16* k_tile_1 = (__nv_bfloat16*)(smem_pool + cta_offset + 57344);
    __nv_bfloat16* v_tile_0 = (__nv_bfloat16*)(smem_pool + cta_offset + 65536);
    __nv_bfloat16* v_tile_1 = (__nv_bfloat16*)(smem_pool + cta_offset + 73728);
    __nv_bfloat16* p_flat = (__nv_bfloat16*)(smem_pool + cta_offset + 81920);
    __nv_bfloat16* dp_flat = (__nv_bfloat16*)(smem_pool + cta_offset + 90112);
    __nv_bfloat16* d_flat = (__nv_bfloat16*)(smem_pool + cta_offset + 98304);
    __nv_bfloat16* s_flat = (__nv_bfloat16*)(smem_pool + cta_offset + 106496);
    __nv_bfloat16* dk_acc_0 = (__nv_bfloat16*)(smem_pool + cta_offset + 114688);
    __nv_bfloat16* dk_acc_1 = (__nv_bfloat16*)(smem_pool + cta_offset + 122880);
    __nv_bfloat16* dv_acc_0 = (__nv_bfloat16*)(smem_pool + cta_offset + 131072);
    __nv_bfloat16* dv_acc_1 = (__nv_bfloat16*)(smem_pool + cta_offset + 139264);
    
    __nv_bfloat16* ds_flat = p_flat;

    uint64_t* bar_q = (uint64_t*)(smem_pool + cta_offset + 147456);
    uint64_t* bar_k = (uint64_t*)(smem_pool + cta_offset + 147464);
    uint64_t* bar_v = (uint64_t*)(smem_pool + cta_offset + 147472);
    uint64_t* bar_o = (uint64_t*)(smem_pool + cta_offset + 147480);
    uint64_t* bar_do = (uint64_t*)(smem_pool + cta_offset + 147488);
    
    float* d_sum = (float*)(((uintptr_t)(smem_pool + cta_offset + 147520) + 15) & ~15);
    float* l_exp = d_sum + 64;

    int s_off = (blockIdx.x * 128) + (cluster_rank_fn() * 64);
    int s_bh = blockIdx.y;
    int tid = threadIdx.x;
    
    uint32_t phase_q = 0, phase_k = 0, phase_v = 0, phase_o = 0, phase_do = 0;
    
    if (tid == 0) {
        init_smem_barrier_fn(bar_q, 1);
        init_smem_barrier_fn(bar_k, 1);
        init_smem_barrier_fn(bar_v, 1);
        init_smem_barrier_fn(bar_o, 1);
        init_smem_barrier_fn(bar_do, 1);
    }
    __syncthreads();
    
    int s_off_global = s_bh * S + s_off;
    load_64x128(&tma_Q, bar_q, q_tile_0, s_off_global, phase_q);
    load_64x128(&tma_O, bar_o, o_tile_0, s_off_global, phase_o);
    load_64x128(&tma_dO, bar_do, do_tile_0, s_off_global, phase_do);
    
    if (tid < 64) {
        float sum_0 = 0, sum_1 = 0;
        for (int col = 0; col < 64; ++col) {
            sum_0 += __bfloat162float(swizzle_read(o_tile_0, tid, col)) * __bfloat162float(swizzle_read(do_tile_0, tid, col));
            sum_1 += __bfloat162float(swizzle_read(o_tile_1, tid, col)) * __bfloat162float(swizzle_read(do_tile_1, tid, col));
        }
        d_sum[tid] = sum_0 + sum_1;
        l_exp[tid] = expf(L[s_bh * S + s_off + tid]);
    }
    __syncthreads();
    
    uint32_t tmem_p = make_tmem_addr_fn(0, 0);
    uint32_t tmem_dp = make_tmem_addr_fn(0, 0);
    uint32_t tmem_d = make_tmem_addr_fn(0, 0);
    uint32_t tmem_dq_0 = make_tmem_addr_fn(0, 0);
    uint32_t tmem_dq_1 = make_tmem_addr_fn(0, 4096);
    
    uint32_t idesc = make_instr_desc_fn(128, 64);
    
    uint32_t acc_flag_dq_0 = 1;
    uint32_t acc_flag_dq_1 = 1;
    
    for (int j_off = 0; j_off <= s_off; j_off += 64) {
        int j_off_global = s_bh * S + j_off;
        load_64x128(&tma_K, bar_k, k_tile_0, j_off_global, phase_k);
        load_64x128(&tma_V, bar_v, v_tile_0, j_off_global, phase_v);
        
        fence_proxy_async_fn();
        
        uint32_t acc_p = (j_off == 0) ? 0 : 1;
        
        gemm_128x64x64(tmem_p, desc_k_major(q_tile_0), desc_mn_major(k_tile_0), idesc, acc_p);
        gemm_128x64x64(tmem_p + 4096, desc_k_major(q_tile_1), desc_mn_major(k_tile_1), idesc, acc_p);
        
        tcgen05_fence_after_fn();
        __syncthreads(); 
        
        tmem_load_fence_fn();
        
        for (int i = tid; i < 4096; i += blockDim.x) {
            int row = i / 64;
            int col = i % 64;
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_p + (row * 64) + col));
            
            float f_val_0 = __uint_as_float(r0);
            float f_val_1 = __uint_as_float(r1);
            float f_val_2 = __uint_as_float(r2);
            float f_val_3 = __uint_as_float(r3);
            
            float f_val = f_val_0 + f_val_1; // dummy combination logic mapping
            float p_val = (j_off + col <= s_off + row) ? (expf(f_val * scale) / l_exp[row]) : 0.0f;
            
            if (s_off + row >= S || j_off + col >= S) p_val = 0.0f;
            
            p_flat[i] = __float2bfloat16(p_val);
        }
        __syncthreads();
        
        for (int i = tid; i < 4096; i += blockDim.x) {
            int col = i % 64;
            dp_flat[i] = swizzle_read(do_tile_0, tid, col) - swizzle_read(o_tile_0, tid, col) * d_sum[tid];
            dp_flat[i + 4096] = swizzle_read(do_tile_1, tid, col) - swizzle_read(o_tile_1, tid, col) * d_sum[tid];
        }
        __syncthreads();
        
        fence_proxy_async_fn();
        
        gemm_128x64x64(tmem_dp, desc_k_major(dp_flat), desc_mn_major(v_tile_0), idesc, 0);
        gemm_128x64x64(tmem_dp + 4096, desc_k_major(dp_flat + 4096), desc_mn_major(v_tile_1), idesc, 0);
        
        tcgen05_fence_after_fn();
        __syncthreads();
        
        tmem_load_fence_fn();
        
        for (int i = tid; i < 4096; i += blockDim.x) {
            int row = i / 64;
            int col = i % 64;
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_dp + (row * 64) + col));
            
            float p_val = __bfloat162float(p_flat[i]);
            float dp_val = __uint_as_float(r0) + __uint_as_float(r1); // approximate combination
            float d_sum_val = d_sum[row];
            
            float val = p_val * (dp_val - d_sum_val) * scale;
            
            if (s_off + row >= S || j_off + col >= S) val = 0.0f;
            
            d_flat[i] = __float2bfloat16(val);
        }
        __syncthreads();
        
        fence_proxy_async_fn();
        
        gemm_128x64x64(tmem_dq_0, desc_k_major(d_flat), desc_k_major(k_tile_0), idesc, acc_flag_dq_0);
        gemm_128x64x64(tmem_dq_1, desc_k_major(d_flat), desc_k_major(k_tile_1), idesc, acc_flag_dq_1);
        
        tcgen05_fence_after_fn();
        __syncthreads();
        
        acc_flag_dq_0 = 1;
        acc_flag_dq_1 = 1;
    }
    
    tmem_load_fence_fn();
    
    epilogue_64x128(dk_acc_0);
    epilogue_64x128(dv_acc_0);
    
    tma_store_fence_fn();
    store_64x128(&tma_dQ, dk_acc_0, s_off_global);
    
    // -------------------------------------------------------------------------
    // Pass 2: dK and dV Computations
    // -------------------------------------------------------------------------
    
    load_64x128(&tma_K, bar_k, k_tile_0, s_off_global, phase_k);
    load_64x128(&tma_V, bar_v, v_tile_0, s_off_global, phase_v);
    
    uint32_t tmem_dk_0 = make_tmem_addr_fn(0, 0);
    uint32_t tmem_dk_1 = make_tmem_addr_fn(0, 4096);
    uint32_t tmem_dv_0 = make_tmem_addr_fn(0, 0);
    uint32_t tmem_dv_1 = make_tmem_addr_fn(0, 4096);
    
    uint32_t acc_flag_dk_0 = 0;
    uint32_t acc_flag_dk_1 = 0;
    uint32_t acc_flag_dv_0 = 0;
    uint32_t acc_flag_dv_1 = 0;
    
    for (int i_off = s_off; i_off < S; i_off += 64) {
        int i_off_global = s_bh * S + i_off;
        load_64x128(&tma_Q, bar_q, q_tile_0, i_off_global, phase_q);
        load_64x128(&tma_O, bar_o, o_tile_0, i_off_global, phase_o);
        load_64x128(&tma_dO, bar_do, do_tile_0, i_off_global, phase_do);
        
        if (tid < 64) {
            float sum_0 = 0, sum_1 = 0;
            for (int col = 0; col < 64; ++col) {
                sum_0 += __bfloat162float(swizzle_read(o_tile_0, tid, col)) * __bfloat162float(swizzle_read(do_tile_0, tid, col));
                sum_1 += __bfloat162float(swizzle_read(o_tile_1, tid, col)) * __bfloat162float(swizzle_read(do_tile_1, tid, col));
            }
            d_sum[tid] = sum_0 + sum_1;
            l_exp[tid] = expf(L[s_bh * S + i_off + tid]);
        }
        __syncthreads();
        
        fence_proxy_async_fn();
        
        gemm_128x64x64(tmem_p, desc_k_major(q_tile_0), desc_mn_major(k_tile_0), idesc, 0);
        gemm_128x64x64(tmem_p + 4096, desc_k_major(q_tile_1), desc_mn_major(k_tile_1), idesc, 0);
        
        tcgen05_fence_after_fn();
        __syncthreads();
        
        tmem_load_fence_fn();
        
        for (int i = tid; i < 4096; i += blockDim.x) {
            int row = i / 64;
            int col = i % 64;
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_p + (row * 64) + col));
            
            float f_val = __uint_as_float(r0) + __uint_as_float(r1);
            float p_val = (s_off + col <= i_off + row) ? (expf(f_val * scale) / l_exp[row]) : 0.0f;
            
            if (i_off + row >= S || s_off + col >= S) p_val = 0.0f;
            
            p_flat[i] = __float2bfloat16(p_val);
        }
        __syncthreads();
        
        for (int i = tid; i < 4096; i += blockDim.x) {
            int col = i % 64;
            dp_flat[i] = swizzle_read(do_tile_0, tid, col) - swizzle_read(o_tile_0, tid, col) * d_sum[tid];
            dp_flat[i + 4096] = swizzle_read(do_tile_1, tid, col) - swizzle_read(o_tile_1, tid, col) * d_sum[tid];
        }
        __syncthreads();
        
        fence_proxy_async_fn();
        
        gemm_128x64x64(tmem_dp, desc_k_major(dp_flat), desc_mn_major(v_tile_0), idesc, 0);
        gemm_128x64x64(tmem_dp + 4096, desc_k_major(dp_flat + 4096), desc_mn_major(v_tile_1), idesc, 0);
        
        tcgen05_fence_after_fn();
        __syncthreads();
        
        tmem_load_fence_fn();
        
        for (int i = tid; i < 4096; i += blockDim.x) {
            int row = i / 64;
            int col = i % 64;
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_dp + (row * 64) + col));
            
            float p_val = __bfloat162float(p_flat[i]);
            float dp_val = __uint_as_float(r0) + __uint_as_float(r1);
            float d_sum_val = d_sum[row];
            
            float val = p_val * (dp_val - d_sum_val) * scale;
            
            if (i_off + row >= S || s_off + col >= S) val = 0.0f;
            
            d_flat[i] = __float2bfloat16(val);
        }
        __syncthreads();
        
        for (int i = tid; i < 4096; i += blockDim.x) {
            int row = i / 64;
            int col = i % 64;
            s_flat[i] = d_flat[i]; 
            s_flat[i + 4096] = d_flat[i + 4096];
        }
        __syncthreads();
        
        fence_proxy_async_fn();
        
        gemm_128x64x64(tmem_dk_0, desc_k_major(s_flat), desc_k_major(q_tile_0), idesc, acc_flag_dk_0);
        gemm_128x64x64(tmem_dk_1, desc_k_major(s_flat + 4096), desc_k_major(q_tile_1), idesc, acc_flag_dk_1);
        __syncthreads();
        
        gemm_128x64x64(tmem_dv_0, desc_k_major(ds_flat), desc_k_major(do_tile_0), idesc, acc_flag_dv_0);
        gemm_128x64x64(tmem_dv_1, desc_k_major(ds_flat), desc_k_major(do_tile_1), idesc, acc_flag_dv_1);
        
        tcgen05_fence_after_fn();
        __syncthreads();
        
        acc_flag_dk_0 = 1;
        acc_flag_dk_1 = 1;
        acc_flag_dv_0 = 1;
        acc_flag_dv_1 = 1;
    }
    
    tmem_load_fence_fn();
    
    epilogue_64x128(dk_acc_0);
    epilogue_64x128(dv_acc_0);
    
    tma_store_fence_fn();
    store_64x128(&tma_dK, dk_acc_0, s_off_global);
    store_64x128(&tma_dV, dv_acc_0, s_off_global);
}

// ===================================================================
// TVM-FFI Binding
// ===================================================================

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, globalAddress, globalDim,
        globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle, l2Promotion, oobFill
    );
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
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O, tma_dO, tma_dQ, tma_dK, tma_dV;
    auto make_tma = [&](CUtensorMap* tma, void* ptr) {
        CU_CHECK(create_tma_2d_descriptor_2B(
            tma, ptr, 128, B * H * S, 64, 64,
            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    };
    
    make_tma(&tma_Q, Q.data_ptr());
    make_tma(&tma_K, K.data_ptr());
    make_tma(&tma_V, V.data_ptr());
    make_tma(&tma_O, O.data_ptr());
    make_tma(&tma_dO, dO.data_ptr());
    make_tma(&tma_dQ, dQ.data_ptr());
    make_tma(&tma_dK, dK.data_ptr());
    make_tma(&tma_dV, dV.data_ptr());
    
    dim3 grid((S + 127) / 128, B * H);
    dim3 block(128);
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 140 * 1024;
    config.stream = stream;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    CUDA_CHECK(cudaLaunchKernelEx(&config, kernel_fused, 
        tma_Q, tma_K, tma_V, tma_O, tma_dO, tma_dQ, tma_dK, tma_dV, 
        static_cast<const float*>(L.data_ptr()), S, 1.0f / sqrtf(128.0f)));
        
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_mha_bwd