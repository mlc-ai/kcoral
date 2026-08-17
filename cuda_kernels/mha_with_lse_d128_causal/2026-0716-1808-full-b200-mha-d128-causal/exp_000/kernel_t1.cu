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

__device__ __forceinline__ bool elect_one_sync_fn() {
    uint32_t pred;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "elect.sync _|p, 0xFFFFFFFF;\n"
        "selp.b32 %0, 1, 0, p;\n"
        "}\n"
        : "=r"(pred));
    return pred != 0;
}

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ uint32_t pack_bf16_fn(uint32_t fp32_a, uint32_t fp32_b) {
    __nv_bfloat16 a = __float2bfloat16(__uint_as_float(fp32_a));
    __nv_bfloat16 b = __float2bfloat16(__uint_as_float(fp32_b));
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};"
        : "=r"(result)
        : "h"(*reinterpret_cast<uint16_t*>(&a)),
          "h"(*reinterpret_cast<uint16_t*>(&b)));
    return result;
}

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

__device__ __forceinline__ void read_swizzled_to_tmem(
    __nv_bfloat16* smem_ptr, uint32_t* tmem_ptr, int K_step) 
{
    uint32_t my_reg[16]; 
    
    for (int k = 0; k < 8; ++k) {
        int col = K_step + k * 2;
        int sx = (threadIdx.x % 8) ^ (col / 8);
        int byte_offset = threadIdx.x * 128 + sx * 16 + (col % 8) * 2;
        uint64_t val = *(uint64_t*)(smem_ptr + byte_offset);
        
        int base = k * 2;
        *reinterpret_cast<uint64_t*>(&my_reg[base]) = val;
    }
    
    for(int i = 0; i < 2; i++) {
        uint32_t packed0 = pack_bf16_fn(__float_as_uint(my_reg[i*4 + 0]), __float_as_uint(my_reg[i*4 + 1]));
        uint32_t packed1 = pack_bf16_fn(__float_as_uint(my_reg[i*4 + 2]), __float_as_uint(my_reg[i*4 + 3]));
        *(uint32_t*)&tmem_ptr[K_step + i*4] = packed0;
        *(uint32_t*)&tmem_ptr[K_step + i*4 + 1] = packed1;
    }
}

__device__ __forceinline__ void read_swizzled_to_regs(
    __nv_bfloat16* smem_ptr, float* my_reg, int K_step) 
{
    for (int k = 0; k < 8; ++k) {
        int col = K_step + k * 2;
        int sx = (threadIdx.x % 8) ^ (col / 8);
        int byte_offset = threadIdx.x * 128 + sx * 16 + (col % 8) * 2;
        uint64_t val = *(uint64_t*)(smem_ptr + byte_offset);
        
        int base = k * 2;
        *reinterpret_cast<uint64_t*>(&my_reg[base]) = val;
    }
}

__device__ __forceinline__ uint64_t make_smem_desc_swizzled(void* smem_ptr, int M_or_K_dim) {
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
    setmaxnreg_inc_sync_fn<256>();

    uint32_t total_tiles = B * H * S;
    uint32_t tiles_per_block = (S + 127) / 128;
    uint32_t bh_idx = blockIdx.x / tiles_per_block;
    uint32_t q_tile_idx = blockIdx.x % tiles_per_block;
    uint32_t q_start = q_tile_idx * 128;

    int head_idx = bh_idx % H;
    int batch_idx = bh_idx / H;
    uint32_t q_end = min(q_start + 128, S);

    if (threadIdx.x == 0) {
        if (q_start >= S) return;
    }
    __syncthreads();
    if (q_start >= S) return;

    int cluster_half = cluster_rank() % 2;

    __shared__ __align__(1024) __nv_bfloat16 smem_Q0[128 * 64];
    __shared__ __align__(1024) __nv_bfloat16 smem_Q1[128 * 64];
    __shared__ __align__(1024) __nv_bfloat16 smem_K0[128 * 64];
    __shared__ __align__(1024) __nv_bfloat16 smem_K1[128 * 64];
    __shared__ __align__(1024) __nv_bfloat16 smem_V0[128 * 64];
    __shared__ __align__(1024) __nv_bfloat16 smem_V1[128 * 64];

    __shared__ alignas(128) float smem_S_partial[2][128][64];
    __shared__ alignas(128) float smem_S_transposed[2][64][128];

    __shared__ alignas(16) uint64_t bar_load[2];
    __shared__ alignas(16) uint64_t bar_kv[2];

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&bar_load[0], 1);
        init_smem_barrier_fn(&bar_load[1], 1);
        init_smem_barrier_fn(&bar_kv[0], 1);
        init_smem_barrier_fn(&bar_kv[1], 1);
    }
    
    __shared__ alignas(16) uint32_t tmem_Q0_addr[1];
    __shared__ alignas(16) uint32_t tmem_Q1_addr[1];
    __shared__ alignas(16) uint32_t tmem_K_addr[2];
    __shared__ alignas(16) uint32_t tmem_V_addr[2];
    __shared__ alignas(16) uint32_t tmem_S_addr[2];
    __shared__ alignas(16) uint32_t tmem_P_addr[2];
    __shared__ alignas(16) uint32_t tmem_O_addr[2];

    if (threadIdx.x == 0) {
        tmem_alloc_fn(tmem_Q0_addr, 128);
        tmem_alloc_fn(tmem_Q1_addr, 128);
    }
    tmem_alloc_fn(&tmem_K_addr[cluster_rank()], 128);
    tmem_alloc_fn(&tmem_V_addr[cluster_rank()], 128);
    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_S_addr[cluster_rank()], 64);
        tmem_alloc_fn(&tmem_P_addr[cluster_rank()], 64);
        tmem_alloc_fn(&tmem_O_addr[cluster_rank()], 64);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        expect_tx_fn(&bar_load[0], 4 * 16384);
        tma_load_3d_fn(&tma_Q, &bar_load[0], smem_Q0, 0, 0, bh_idx);
        tma_load_3d_fn(&tma_Q, &bar_load[0], smem_Q1, 64, 0, bh_idx);
        
        expect_tx_fn_fn(&bar_kv[0], 4 * 16384);
        tma_load_3d_fn(&tma_K, &bar_kv[0], smem_K0, 0, 0, bh_idx);
        tma_load_3d_fn(&tma_K, &bar_kv[0], smem_K1, 64, 0, bh_idx);
        tma_load_3d_fn(&tma_V, &bar_kv[0], smem_V0, 0, 0, bh_idx);
        tma_load_3d_fn(&tma_V, &bar_kv[0], smem_V1, 64, 0, bh_idx);
    }

    int phase_load = 0;
    mbarrier_wait_fn(&bar_load[0], phase_load);
    phase_load ^= 1;
    
    int phase_kv[2] = {0, 0};

    float m_old[2] = {-INFINITY, -INFINITY};
    float l_old[2] = {0.0f, 0.0f};
    float my_O[2][128];
    for(int i = 0; i < 128; ++i) {
        my_O[0][i] = 0.0f;
        my_O[1][i] = 0.0f;
    }

    uint32_t max_k = min(q_end - 1, (uint32_t)(q_start + 127));
    
    for (uint32_t k_start = 0; k_start <= max_k; k_start += 128) {
        uint32_t next_k_start = k_start + 128;
        int next_idx = (k_start / 128) % 2;
        int curr_idx = next_idx ^ 1;

        if (next_k_start <= max_k) {
            if (threadIdx.x == 0) {
                expect_tx_fn(&bar_kv[next_idx], 4 * 16384);
                tma_load_3d_fn(&tma_K, &bar_kv[next_idx], smem_K0, 0, next_k_start, bh_idx);
                tma_load_3d_fn(&tma_K, &bar_kv[next_idx], smem_K1, 64, next_k_start, bh_idx);
                tma_load_3d_fn(&tma_V, &bar_kv[next_idx], smem_V0, 0, next_k_start, bh_idx);
                tma_load_3d_fn(&tma_V, &bar_kv[next_idx], smem_V1, 64, next_k_start, bh_idx);
            }
        }
        
        mbarrier_wait_fn(&bar_kv[curr_idx], phase_kv[curr_idx]);
        phase_kv[curr_idx] ^= 1;

        __syncthreads();

        uint32_t* s_tmem = tmem_S_addr[cluster_rank()] + cluster_rank()*128; 
        
        for(int i = 0; i < 128; ++i) {
            *(uint32_t*)&s_tmem[i] = 0;
        }

        __nv_bfloat16* my_K = (cluster_half == 0) ? smem_K0 : smem_K1;
        
        for (int d_half = 0; d_half < 2; ++d_half) {
            uint32_t* q_tmem = tmem_Q0_addr[0] + cluster_rank()*128;
            uint32_t* k_tmem = tmem_K_addr[cluster_rank()] + cluster_rank()*128;
            
            __nv_bfloat16* my_Q = (d_half == 0) ? smem_Q0 : smem_Q1;
            
            for (int K_step = 0; K_step < 64; K_step += 16) {
                read_swizzled_to_tmem(my_Q, q_tmem, K_step);
                read_swizzled_to_tmem(my_K, k_tmem, K_step);
            }
            uint64_t desc_Q = make_smem_desc_swizzled(my_Q, 128);
            uint64_t desc_K = make_smem_desc_swizzled(my_K, 128);
            uint32_t idesc_QK = make_instr_desc_fn(128, 128, 0, 0);
            uint32_t accum_QK = (d_half == 0 && k_start == 0) ? 0 : 1;
            umma_f16_cg1(s_tmem, q_tmem, desc_K, idesc_QK, accum_QK);
        }

        float my_S[128];
        for (int i = 0; i < 128 / 8; ++i) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(s_tmem + i * 8));
            my_S[i * 8 + 0] = __uint_as_float(r0);
            my_S[i * 8 + 1] = __uint_as_float(r1);
            my_S[i * 8 + 2] = __uint_as_float(r2);
            my_S[i * 8 + 3] = __uint_as_float(r3);
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

        for(int i = threadIdx.x; i < 128 * 64; i += blockDim.x) {
            int row = i / 64;
            int col = i % 64;
            smem_S_partial[cluster_half][row][col] = my_S[row * 64 + col];
        }
        __syncthreads();

        int row = threadIdx.x;
        int global_q_idx = q_start + row;
        
        float m_row[2] = {-INFINITY, -INFINITY};
        for(int c = 0; c < 64; c++) {
            int global_k_idx = k_start + cluster_half * 64 + c;
            float val = smem_S_partial[cluster_half][row][c] * 0.08838834764f; // 1/sqrt(128)
            if (global_k_idx > global_q_idx || global_k_idx >= S) {
                val = -INFINITY;
            }
            smem_S_partial[cluster_half][row][c] = val;
            m_row[cluster_half] = fmaxf(m_row[cluster_half], val);
        }

        float m_local[2];
        m_local[0] = m_row[0];
        m_local[1] = m_row[1];
        
        float m_full[2];
        m_full[0] = fmaxf(m_local[0], m_local[1]);
        m_full[1] = fmaxf(m_local[1], m_local[0]);

        float l_row[2] = {0.0f, 0.0f};
        for(int c = 0; c < 64; c++) {
            float val = smem_S_partial[cluster_half][row][c];
            if (m_full[cluster_half] > -INFINITY) {
                float diff = m_full[cluster_half] - val;
                float e = fast_exp2f_fn(diff * 1.4426950408889634f);
                smem_S_partial[cluster_half][row][c] = e;
                l_row[cluster_half] += e;
            }
        }

        float l_local[2];
        l_local[0] = l_row[0];
        l_local[1] = l_row[1];

        float l_full[2];
        l_full[0] = l_local[0] + l_local[1];
        l_full[1] = l_local[1] + l_local[0];

        float m_old_val = m_old[cluster_half];
        float m_new_val = fmaxf(m_old_val, m_full[cluster_half]);
        
        float l_old_val = l_old[cluster_half];
        float l_new_val = l_old_val * fast_exp2f_fn((m_old_val - m_new_val) * 1.4426950408889634f) + 
                          l_full[cluster_half] * fast_exp2f_fn((m_full[cluster_half] - m_new_val) * 1.4426950408889634f);
        
        m_old[cluster_half] = m_new_val;
        l_old[cluster_half] = l_new_val;

        for(int c = 0; c < 64; c++) {
            float val = smem_S_partial[cluster_half][row][c];
            float e = 0.0f;
            if (m_full[cluster_half] > -INFINITY) {
                float diff = m_full[cluster_half] - val;
                e = fast_exp2f_fn(diff * 1.4426950408889634f);
            }
            float e_scaled = e * fast_exp2f_fn((m_full[cluster_half] - m_new_val) * 1.4426950408889634f);
            
            smem_S_partial[cluster_half][row][c] = e_scaled;
        }

        if (l_new_val > 0.0f) {
            for(int c = 0; c < 64; c++) {
                smem_S_partial[cluster_half][row][c] /= l_new_val;
            }
        }
        
        for (int K_step = 0; K_step < 64; K_step += 16) {
            float* s_ptr = smem_S_partial[cluster_half] + K_step * 128;
            uint32_t* p_tmem = tmem_P_addr[cluster_half] + cluster_rank()*128 + K_step;
            for (int r = 0; r < 128; ++r) {
                int c = r ^ ((K_step / 8) * 8);
                __nv_bfloat16 val = __float2bfloat16(s_ptr[r]);
                *(uint32_t*)&p_tmem[c] = pack_bf16_fn(__float_as_uint(val), __float_as_uint(val));
            }
        }
        __syncthreads();

        for (int K_step = 0; K_step < 64; K_step += 16) {
            uint32_t accum = (K_step == 0) ? 0 : 1;
            umma_cg1_p_v(p_tmem, o_tmem, (__nv_bfloat16*)smem_V0, K_step, accum);
        }
        for (int K_step = 0; K_step < 64; K_step += 16) {
            uint32_t accum = (K_step == 0) ? 0 : 1;
            umma_cg1_p_v(p_tmem, o_tmem, (__nv_bfloat16*)smem_V1, K_step, accum);
        }
        
        __syncthreads();
    }

    float o_reg[2][128];
    for (int d_half = 0; d_half < 2; ++d_half) {
        uint32_t* o_tmem = tmem_O_addr[d_half] + cluster_rank()*128;
        for (int i = 0; i < 128 / 8; ++i) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(o_tmem + i * 8));
            o_reg[d_half][i * 8 + 0] = __uint_as_float(r0);
            o_reg[d_half][i * 8 + 1] = __uint_as_float(r1);
            o_reg[d_half][i * 8 + 2] = __uint_as_float(r2);
            o_reg[d_half][i * 8 + 3] = __uint_as_float(r3);
        }
    }
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

    float o_final[2][128];
    for(int d = 0; d < 2; d++) {
        for(int i = 0; i < 128; i++) {
            o_final[d][i] = o_reg[d][i] / l_old[d];
        }
    }
    
    for(int i = 0; i < 128; i++) {
        float temp0 = o_final[0][i];
        float temp1 = o_final[1][i];
        if (cluster_rank() % 2 == 0) {
            o_final[0][i] = temp0;
            o_final[1][i] = temp1;
        } else {
            o_final[0][i] = temp1;
            o_final[1][i] = temp0;
        }
    }

    __nv_bfloat16* out_O = O + (batch_idx * H + head_idx) * S * 128 + q_start * 128;
    for (int d_half = 0; d_half < 2; ++d_half) {
        for (int col = 0; col < 128; col += 4) {
            uint32_t packed0 = pack_bf16_fn(__float_as_uint(o_final[d_half][col]), __float_as_uint(o_final[d_half][col+1]));
            uint32_t packed1 = pack_bf16_fn(__float_as_uint(o_final[d_half][col+2]), __float_as_uint(o_final[d_half][col+3]));
            uint64_t out = ((uint64_t)packed1 << 32) | packed0;
            *(uint64_t*)&out_O[threadIdx.x * 128 + d_half * 128 + col] = out;
        }
    }

    if (threadIdx.x == 0) {
        for(int d_half = 0; d_half < 2; ++d_half) {
            float lse_val = m_old[d_half] + logf(l_old[d_half]);
            LSE[(batch_idx * H + head_idx) * S + q_start + d_half * 128] = lse_val;
        }
    }
    
    tmem_dealloc_fn(*(uint32_t*)tmem_Q0_addr, 128);
    tmem_dealloc_fn(*(uint32_t*)tmem_Q1_addr, 128);
    tmem_dealloc_fn(*(uint32_t*)&tmem_K_addr[0], 128);
    tmem_dealloc_fn(*(uint32_t*)&tmem_K_addr[1], 128);
    tmem_dealloc_fn(*(uint32_t*)&tmem_V_addr[0], 128);
    tmem_dealloc_fn(*(uint32_t*)&tmem_V_addr[1], 128);
    tmem_dealloc_fn(*(uint32_t*)&tmem_S_addr[0], 64);
    tmem_dealloc_fn(*(uint32_t*)&tmem_S_addr[1], 64);
    tmem_dealloc_fn(*(uint32_t*)&tmem_P_addr[0], 64);
    tmem_dealloc_fn(*(uint32_t*)&tmem_P_addr[1], 64);
    tmem_dealloc_fn(*(uint32_t*)&tmem_O_addr[0], 64);
    tmem_dealloc_fn(*(uint32_t*)&tmem_O_addr[1], 64);
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

    __nv_bfloat16* o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_ptr = static_cast<float*>(LSE.data_ptr());

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