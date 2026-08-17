#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>
#include <math.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_example_cuda {

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.expect_tx.shared.b64 [%0], %1;"
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

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void cluster_sync_fn() {
    asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n" ::: "memory");
}

__device__ __forceinline__ uint32_t cluster_rank() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__device__ __forceinline__ void copy_smem_to_tmem_swizzled(
    __nv_bfloat16* smem_ptr, uint32_t* tmem_ptr, int row_base, int col_base, 
    int row_end, int col_end, int tmem_col_base) 
{
    int byte_step_x = 128;
    int byte_step_y = 2;
    
    for (int i = threadIdx.x; i < (row_end - row_base) * (col_end - col_base); i += blockDim.x) {
        int row = i / (col_end - col_base);
        int col = i % (col_end - col_base);
        
        int col_bytes = col * byte_step_y;
        int chunk_idx = col_bytes / 16;
        int sx = (row % 8) ^ chunk_idx;
        
        int smem_byte_offset = (row_base + row) * 128 + sx * 16 + (col_bytes % 16);
        
        __nv_bfloat16 val = *(__nv_bfloat16*)((char*)smem_ptr + smem_byte_offset);
        
        int tmem_row = row_base + row;
        int tmem_col = tmem_col_base + col;
        int tmem_byte_col = tmem_col * byte_step_y;
        int tmem_chunk_idx = tmem_byte_col / 16;
        int tmem_sx = (tmem_row % 8) ^ tmem_chunk_idx;
        
        int tmem_byte_offset = (tmem_row * 128) + tmem_sx * 16 + (tmem_byte_col % 16);
        *(__nv_bfloat16*)((char*)tmem_ptr + tmem_byte_offset) = val;
    }
}

__device__ __forceinline__ uint64_t make_smem_desc_swizzled(void* smem_ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (addr & 0x3FFFF) >> 4;
    uint32_t sbo = 8 * 128; 
    uint32_t lbo = 1;
    d |= (((uint64_t)(lbo & 0x3FFFF) >> 4) << 16);
    d |= (((uint64_t)(sbo & 0x3FFFF) >> 4) << 32);
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_swizzled_mn_major(void* smem_ptr, int K_dim) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (addr & 0x3FFFF) >> 4;
    uint32_t sbo = 1024;
    uint32_t lbo = (K_dim / 8) * sbo;
    d |= (((uint64_t)(lbo & 0x3FFFF) >> 4) << 16);
    d |= (((uint64_t)(sbo & 0x3FFFF) >> 4) << 32);
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, uint32_t a_major = 0, uint32_t b_major = 0) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (a_major << 15);
    d |= (b_major << 16);
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ void umma_f16_cg1(
    uint32_t* d_tmem, uint32_t* a_tmem, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}\n"
        :: "r"(d_tmem), "r"(a_tmem), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_f16_cg2(
    uint32_t* d_tmem, uint32_t* a_tmem, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], [%1], %2, %3, p;\n}\n"
        :: "r"(d_tmem), "r"(a_tmem), "l"(desc_b), "r"(idesc), "r"(accum));
}

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t dim0, uint64_t dim1, uint64_t dim2, uint32_t box0, uint32_t box1, uint32_t box2, CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[3] = {dim0, dim1, dim2};
    cuuint64_t globalStrides[2] = {dim0 * 2, dim0 * dim1 * 2};
    cuuint32_t boxDim[3] = {box0, box1, box2};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

__global__ void causal_mha_lse_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O,
    float* LSE,
    uint32_t S, uint32_t H, uint32_t B) 
{
    extern __shared__ __align__(1024) char smem_pool[];
    __nv_bfloat16* q0_smem = (__nv_bfloat16*)smem_pool;                   // 16KB
    __nv_bfloat16* q1_smem = (__nv_bfloat16*)(smem_pool + 16384);        // 16KB
    __nv_bfloat16* k0_smem = (__nv_bfloat16*)(smem_pool + 32768);        // 16KB
    __nv_bfloat16* k1_smem = (__nv_bfloat16*)(smem_pool + 49152);        // 16KB
    __nv_bfloat16* v0_smem = (__nv_bfloat16*)(smem_pool + 65536);        // 16KB
    __nv_bfloat16* v1_smem = (__nv_bfloat16*)(smem_pool + 81920);        // 16KB
    uint64_t* bar_load = (uint64_t*)(smem_pool + 98304);                 // 8B
    uint64_t* bar_kv = (uint64_t*)(smem_pool + 98312);                   // 16B

    uint32_t tiles_per_block = (S + 127) / 128;
    uint32_t cluster_r = cluster_rank();
    uint32_t q_r = cluster_r / 2;
    uint32_t c_r = cluster_r % 2;
    uint32_t q_start = (cluster_r < 2) ? 0 : 128; 
    
    uint32_t bh_idx = blockIdx.x / tiles_per_block;
    uint32_t q_tile_idx = blockIdx.x % tiles_per_block;
    uint32_t actual_q_start = q_tile_idx * 128;
    int head_idx = bh_idx % H;
    int batch_idx = bh_idx / H;
    uint32_t q_end = min(actual_q_start + 256, S);

    if ((actual_q_start + q_start) >= S) return;
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&bar_load[0], 1);
        init_smem_barrier_fn(&bar_kv[0], 1);
        init_smem_barrier_fn(&bar_kv[1], 1);
    }
    __syncthreads();
    
    __shared__ alignas(16) uint32_t q0_tmem_addr[1];
    __shared__ alignas(16) uint32_t q1_tmem_addr[1];
    __shared__ alignas(16) uint32_t k0_tmem_addr[1];
    __shared__ alignas(16) uint32_t k1_tmem_addr[1];
    __shared__ alignas(16) uint32_t v0_tmem_addr[1];
    __shared__ alignas(16) uint32_t v1_tmem_addr[1];
    __shared__ alignas(16) uint32_t s_tmem_addr[1];
    __shared__ alignas(16) uint32_t p_tmem_addr[1];
    __shared__ alignas(16) uint32_t o_tmem_addr[1];

    if (threadIdx.x == 0) {
        tmem_alloc_fn((uint32_t*)&q0_tmem_addr, 128);
        tmem_alloc_fn((uint32_t*)&q1_tmem_addr, 128);
        tmem_alloc_fn((uint32_t*)&k0_tmem_addr, 128);
        tmem_alloc_fn((uint32_t*)&k1_tmem_addr, 128);
        tmem_alloc_fn((uint32_t*)&v0_tmem_addr, 128);
        tmem_alloc_fn((uint32_t*)&v1_tmem_addr, 128);
        tmem_alloc_fn((uint32_t*)&s_tmem_addr, 128);
        tmem_alloc_fn((uint32_t*)&p_tmem_addr, 128);
        tmem_alloc_fn((uint32_t*)&o_tmem_addr, 128);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        expect_tx_fn(&bar_load[0], 4 * 16384);
        
        tma_load_3d_fn(&tma_Q, &bar_load[0], q0_smem, 0, q_start, bh_idx);
        tma_load_3d_fn(&tma_Q, &bar_load[0], q1_smem, 64, q_start, bh_idx);
        
        tma_load_3d_fn(&tma_Q, &bar_load[0], q0_smem, 0, q_start + 128, bh_idx);
        tma_load_3d_fn(&tma_Q, &bar_load[0], q1_smem, 64, q_start + 128, bh_idx);
    }

    int phase_kv[2] = {0, 0};
    int k_iter = 0;
    uint32_t max_k = min(q_end - 1, (uint32_t)(actual_q_start + 255));
    
    while (k_iter <= max_k) {
        int next_k_iter = k_iter + 128;
        int next_idx = (next_k_iter / 128) % 2;
        
        if (next_k_iter <= max_k) {
            if (threadIdx.x == 0) {
                expect_tx_fn(&bar_kv[next_idx], 4 * 16384);
                tma_load_3d_fn(&tma_K, &bar_kv[next_idx], (next_idx == 0) ? k0_smem : k1_smem, 0, next_k_iter, bh_idx);
                tma_load_3d_fn(&tma_K, &bar_kv[next_idx], (next_idx == 0) ? k0_smem : k1_smem, 64, next_k_iter, bh_idx);
                tma_load_3d_fn(&tma_V, &bar_kv[next_idx], (next_idx == 0) ? v0_smem : v1_smem, 0, next_k_iter, bh_idx);
                tma_load_3d_fn(&tma_V, &bar_kv[next_idx], (next_idx == 0) ? v0_smem : v1_smem, 64, next_k_iter, bh_idx);
            }
        }
        k_iter = next_k_iter;
    }
    
    uint32_t* q0_tmem = (uint32_t*)q0_tmem_addr[0];
    uint32_t* q1_tmem = (uint32_t*)q1_tmem_addr[0];
    
    mbarrier_wait_fn(&bar_load[0], 0);
    
    float m_old[2] = {-INFINITY, -INFINITY};
    float l_old[2] = {0.0f, 0.0f};

    k_iter = 0;
    while (k_iter <= max_k) {
        int curr_idx = (k_iter / 128) % 2;
        mbarrier_wait_fn(&bar_kv[curr_idx], phase_kv[curr_idx]);
        phase_kv[curr_idx] ^= 1;

        uint32_t* s_tmem = (uint32_t*)s_tmem_addr[0];
        for(int i = 0; i < 128 / 4; ++i) *(uint32_t*)&s_tmem[i] = 0;
        
        float my_S[128];
        for (int i = 0; i < 128; i++) my_S[i] = 0.0f;
        
        __nv_bfloat16* my_K = (curr_idx == 0) ? k0_smem : k1_smem;
        __nv_bfloat16* my_V = (curr_idx == 0) ? v0_smem : v1_smem;
        
        for (int K_step = 0; K_step < 64; K_step += 16) {
            copy_smem_to_tmem_swizzled(q0_smem, q0_tmem, 0, 0, 128, 64, K_step);
            copy_smem_to_tmem_swizzled(my_K, (uint32_t*)k0_tmem_addr[0], 0, 0, 128, 64, K_step);
            uint64_t desc_Q0 = make_smem_desc_swizzled(q0_smem);
            uint64_t desc_K0 = make_smem_desc_swizzled(my_K);
            uint32_t idesc_QK = make_instr_desc_fn(128, 128, 0, 0);
            uint32_t accum_QK = (k_iter == 0 && K_step == 0) ? 0 : 1;
            umma_f16_cg1(s_tmem + K_step, q0_tmem + K_step, desc_K0, idesc_QK, accum_QK);
        }
        
        for (int K_step = 0; K_step < 64; K_step += 16) {
            copy_smem_to_tmem_swizzled(q1_smem, q1_tmem, 0, 0, 128, 64, K_step);
            copy_smem_to_tmem_swizzled(my_K, (uint32_t*)k1_tmem_addr[0], 0, 0, 128, 64, K_step);
            uint64_t desc_Q1 = make_smem_desc_swizzled(q1_smem);
            uint64_t desc_K1 = make_smem_desc_swizzled(my_K);
            uint32_t idesc_QK = make_instr_desc_fn(128, 128, 0, 0);
            umma_f16_cg1(s_tmem + K_step, q1_tmem + K_step, desc_K1, idesc_QK, 1);
        }
        
        for (int k = 0; k < 4; k++) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(s_tmem + k * 16));
            my_S[k * 8 + 0] = __uint_as_float(r0);
            my_S[k * 8 + 1] = __uint_as_float(r1);
            my_S[k * 8 + 2] = __uint_as_float(r2);
            my_S[k * 8 + 3] = __uint_as_float(r3);
        }
        
        for (int k = 0; k < 4; k++) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(((uint32_t*)k0_tmem_addr[0]) + k * 16));
            my_S[k * 8 + 0] += __uint_as_float(r0);
            my_S[k * 8 + 1] += __uint_as_float(r1);
            my_S[k * 8 + 2] += __uint_as_float(r2);
            my_S[k * 8 + 3] += __uint_as_float(r3);
        }
        
        for (int k = 0; k < 4; k++) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(s_tmem + 64 + k * 16));
            my_S[k * 8 + 4] = __uint_as_float(r0);
            my_S[k * 8 + 5] = __uint_as_float(r1);
            my_S[k * 8 + 6] = __uint_as_float(r2);
            my_S[k * 8 + 7] = __uint_as_float(r3);
        }
        
        for (int k = 0; k < 4; k++) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(((uint32_t*)k1_tmem_addr[0]) + k * 16));
            my_S[k * 8 + 4] += __uint_as_float(r0);
            my_S[k * 8 + 5] += __uint_as_float(r1);
            my_S[k * 8 + 6] += __uint_as_float(r2);
            my_S[k * 8 + 7] += __uint_as_float(r3);
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

        __shared__ float m_local[4][32];
        float m_row[2] = {-INFINITY, -INFINITY};
        for(int c = 0; c < 64; c++) {
            int global_q_idx = actual_q_start + q_r * 128 + threadIdx.x;
            int global_k_idx = k_iter + c_r * 64 + c;
            float val = my_S[threadIdx.x * 64 + c] * 0.08838834764f; // 1/sqrt(128)
            if (global_k_idx > global_q_idx || global_k_idx >= S) {
                val = -INFINITY;
            }
            my_S[threadIdx.x * 64 + c] = val;
            m_row[c_r] = fmaxf(m_row[c_r], val);
        }

        int warp_id = threadIdx.x / 32;
        int lane_id = threadIdx.x % 32;
        __shared__ float m_local_wg[4][32];
        if (lane_id == 0) m_local_wg[warp_id][c_r] = m_row[c_r];
        __sync_warp();
        
        m_row[c_r] = m_local_wg[warp_id][lane_id];
        for (int offset = 16; offset > 0; offset /= 2) {
            m_row[c_r] = fmaxf(m_row[c_r], __shfl_xor_sync(0xffffffff, m_row[c_r], offset));
        }

        float l_row[2] = {0.0f, 0.0f};
        for(int c = 0; c < 64; c++) {
            if (m_row[c_r] > -INFINITY) {
                float diff = m_row[c_r] - my_S[threadIdx.x * 64 + c];
                float e = fast_exp2f_fn(diff * 1.4426950408889634f);
                my_S[threadIdx.x * 64 + c] = e;
                l_row[c_r] += e;
            }
        }

        __shared__ float l_local_wg[4][32];
        if (lane_id == 0) l_local_wg[warp_id][c_r] = l_row[c_r];
        __sync_warp();
        l_row[c_r] = l_local_wg[warp_id][lane_id];
        for (int offset = 16; offset > 0; offset /= 2) {
            l_row[c_r] += __shfl_xor_sync(0xffffffff, l_row[c_r], offset);
        }

        float m_old_val = m_old[c_r];
        float m_new_val = fmaxf(m_old_val, m_row[c_r]);
        
        float l_old_val = l_old[c_r];
        float l_new_val = l_old_val * fast_exp2f_fn((m_old_val - m_new_val) * 1.4426950408889634f) + 
                          l_row[c_r] * fast_exp2f_fn((m_row[c_r] - m_new_val) * 1.4426950408889634f);
        
        m_old[c_r] = m_new_val;
        l_old[c_r] = l_new_val;

        for(int c = 0; c < 64; c++) {
            float val = my_S[threadIdx.x * 64 + c];
            float e = 0.0f;
            if (m_row[c_r] > -INFINITY) {
                float diff = m_row[c_r] - val;
                e = fast_exp2f_fn(diff * 1.4426950408889634f);
            }
            float e_scaled = e * fast_exp2f_fn((m_row[c_r] - m_new_val) * 1.4426950408889634f);
            
            int tmem_col = c_r * 64 + c;
            int tmem_byte_col = tmem_col * 2;
            int tmem_chunk_idx = tmem_byte_col / 16;
            int tmem_sx = (threadIdx.x % 8) ^ tmem_chunk_idx;
            
            int tmem_byte_offset = (threadIdx.x * 128) + tmem_sx * 16 + (tmem_byte_col % 16);
            
            __nv_bfloat16 bf_val = __float2bfloat16(e_scaled);
            *(uint16_t*)((char*)p_tmem + tmem_byte_offset) = *(uint16_t*)&bf_val;
        }
        
        __syncthreads();

        uint32_t* o_tmem = (uint32_t*)o_tmem_addr[0];
        uint64_t desc_V_maj = make_smem_desc_swizzled_mn_major(my_V, 128);
        uint32_t idesc_PV = make_instr_desc_fn(128, 128, 0, 1); 
        
        for (int K_step = 0; K_step < 64; K_step += 16) {
            uint32_t accum_PV = (k_iter == 0 && K_step == 0) ? 0 : 1;
            umma_f16_cg2(o_tmem + (K_step / 2) * 2, p_tmem + (K_step / 2), desc_V_maj, idesc_PV, accum_PV);
        }
        
        __syncthreads();
        k_iter += 128;
    }
    
    for (int i = 0; i < 128; i += 4) {
        uint32_t p0 = pack_bf16_fn(__float_as_uint(my_S[i] / l_old[c_r]), __float_as_uint(my_S[i+1] / l_old[c_r]));
        uint32_t p1 = pack_bf16_fn(__float_as_uint(my_S[i+2] / l_old[c_r]), __float_as_uint(my_S[i+3] / l_old[c_r]));
        
        __nv_bfloat16* out_O = O + (batch_idx * H + head_idx) * S * 128 + actual_q_start * 128 + q_r * 128 * 128;
        *(uint32_t*)&out_O[threadIdx.x * 128 + c_r * 64 + i] = p0;
        *(uint32_t*)&out_O[threadIdx.x * 128 + c_r * 64 + i + 2] = p1;
    }

    if (threadIdx.x == 0) {
        float lse_val = m_old[c_r] + logf(l_old[c_r]);
        LSE[(batch_idx * H + head_idx) * S + actual_q_start + q_r * 128] = lse_val;
    }
    
    tmem_dealloc_fn(*(uint32_t*)&q0_tmem_addr, 128);
    tmem_dealloc_fn(*(uint32_t*)&q1_tmem_addr, 128);
    tmem_dealloc_fn(*(uint32_t*)&k0_tmem_addr, 128);
    tmem_dealloc_fn(*(uint32_t*)&k1_tmem_addr, 128);
    tmem_dealloc_fn(*(uint32_t*)&v0_tmem_addr, 128);
    tmem_dealloc_fn(*(uint32_t*)&v1_tmem_addr, 128);
    tmem_dealloc_fn(*(uint32_t*)&s_tmem_addr, 128);
    tmem_dealloc_fn(*(uint32_t*)&p_tmem_addr, 128);
    tmem_dealloc_fn(*(uint32_t*)&o_tmem_addr, 128);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    uint32_t B = Q.size(0);
    uint32_t H = Q.size(1);
    uint32_t S = Q.size(2);
    uint32_t D = Q.size(3);

    CUtensorMap tma_Q, tma_K, tma_V;
    CUresult res;
    res = create_tma_3d_descriptor_2B(&tma_Q, Q.data_ptr(), D, S, B * H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA Q encode error\n"); exit(1); }
    res = create_tma_3d_descriptor_2B(&tma_K, K.data_ptr(), D, S, B * H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B);
    res = create_tma_3d_descriptor_2B(&tma_V, V.data_ptr(), D, S, B * H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B);

    uint32_t* o_mem = (uint32_t*)O.data_ptr();
    cudaMemsetAsync(o_mem, 0, B * H * S * D * sizeof(__nv_bfloat16), static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id)));

    __nv_bfloat16* o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_ptr = static_cast<float*>(LSE.data_ptr());

    uint32_t tiles_per_block = (S + 127) / 128;
    dim3 grid(tiles_per_block * B * H);
    dim3 block(128);

    CUDA_CHECK(cudaFuncSetAttribute(
        causal_mha_lse_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        114704
    ));

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 114704;
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, causal_mha_lse_kernel, tma_Q, tma_K, tma_V, o_ptr, lse_ptr, S, H, B));

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

} // namespace tvm_ffi_example_cuda