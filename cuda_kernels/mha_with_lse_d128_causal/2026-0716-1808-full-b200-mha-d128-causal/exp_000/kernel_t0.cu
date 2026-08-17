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

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void read_swizzled_to_tmem(
    __nv_bfloat16* smem_Q, uint32_t* tmem_Q, int K_step) 
{
    uint32_t my_Q[16]; 
    
    for (int k = 0; k < 8; ++k) {
        int col = K_step + k * 2;
        int sx = (threadIdx.x % 8) ^ (col / 8);
        int byte_offset = threadIdx.x * 128 + sx * 16 + (col % 8) * 2;
        uint64_t val = *(uint64_t*)(smem_Q + byte_offset);
        
        int base = k * 2;
        *reinterpret_cast<uint64_t*>(&my_Q[base]) = val;
    }
    
    for(int i = 0; i < 2; i++) {
        uint32_t packed0 = pack_bf16_fn(__float_as_uint(my_Q[i*4 + 0]), __float_as_uint(my_Q[i*4 + 1]));
        uint32_t packed1 = pack_bf16_fn(__float_as_uint(my_Q[i*4 + 2]), __float_as_uint(my_Q[i*4 + 3]));
        *(uint32_t*)&tmem_Q[K_step + i*4] = packed0;
        *(uint32_t*)&tmem_Q[K_step + i*4 + 1] = packed1;
    }
}

__device__ __forceinline__ void read_swizzled_to_regs(
    __nv_bfloat16* smem_Q, float* my_Q, int K_step) 
{
    for (int k = 0; k < 8; ++k) {
        int col = K_step + k * 2;
        int sx = (threadIdx.x % 8) ^ (col / 8);
        int byte_offset = threadIdx.x * 128 + sx * 16 + (col % 8) * 2;
        uint64_t val = *(uint64_t*)(smem_Q + byte_offset);
        
        int base = k * 2;
        *reinterpret_cast<uint64_t*>(&my_Q[base]) = val;
    }
}

__device__ __forceinline__ uint64_t make_smem_desc_swizzled(void* smem_ptr, int M_or_K_dim) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (addr & 0x3FFFF) >> 4;
    uint32_t sbo = 8 * 128; 
    uint32_t lbo = (M_or_K_dim / 8) * sbo;
    d |= ((lbo & 0x3FFFF) >> 4) << 16;
    d |= ((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_swizzled_mn_major(void* smem_ptr, int K_dim) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (addr & 0x3FFFF) >> 4;
    uint32_t sbo = 128 * 128; 
    uint32_t lbo = (K_dim / 8) * sbo;
    d |= ((lbo & 0x3FFFF) >> 4) << 16;
    d |= ((sbo & 0x3FFFF) >> 4) << 32;
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

__device__ __forceinline__ void umma_cg1_p_v(uint32_t* p_tmem, uint32_t* o_tmem, __nv_bfloat16* smem_V, int K_step) {
    float p_reg[128];
    for (int i = 0; i < 128 / 8; ++i) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
       : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(p_tmem + K_step + i * 8));
        p_reg[i * 8 + 0] = __uint_as_float(r0);
        p_reg[i * 8 + 1] = __uint_as_float(r1);
        p_reg[i * 8 + 2] = __uint_as_float(r2);
        p_reg[i * 8 + 3] = __uint_as_float(r3);
    }
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    
    uint32_t my_P[16];
    for(int i = 0; i < 2; i++) {
        uint32_t packed0 = pack_bf16_fn(__float_as_uint(my_P[i*4 + 0]), __float_as_uint(my_P[i*4 + 1]));
        uint32_t packed1 = pack_bf16_fn(__float_as_uint(my_P[i*4 + 2]), __float_as_uint(my_P[i*4 + 3]));
        *(uint32_t*)&p_tmem[K_step + i*4] = packed0;
        *(uint32_t*)&p_tmem[K_step + i*4 + 1] = packed1;
    }

    float v_reg[4][16];
    for (int row = 0; row < 4; ++row) {
        read_swizzled_to_regs((__nv_bfloat16*)(smem_V + (K_step + row * 16) * 64), v_reg[row], 0);
    }

    uint32_t accum = 0;
    for (int i = 0; i < 2; ++i) {
        uint32_t packed0 = pack_bf16_fn(__float_as_uint(p_reg[i*4 + 0]), __float_as_uint(p_reg[i*4 + 1]));
        uint32_t packed1 = pack_bf16_fn(__float_as_uint(p_reg[i*4 + 2]), __float_as_uint(p_reg[i*4 + 3]));
        *(uint32_t*)&p_tmem[K_step + i*4] = packed0;
        *(uint32_t*)&p_tmem[K_step + i*4 + 1] = packed1;
    }
    uint64_t desc_P = make_smem_desc_swizzled(p_tmem, 128); 

    for(int i = 0; i < 2; i++) {
        uint32_t packed0 = pack_bf16_fn(__float_as_uint(v_reg[i][0]), __float_as_uint(v_reg[i][1]));
        uint32_t packed1 = pack_bf16_fn(__float_as_uint(v_reg[i][2]), __float_as_uint(v_reg[i][3]));
        *(uint32_t*)&smem_V[K_step + i*4] = packed0;
        *(uint32_t*)&smem_V[K_step + i*4 + 1] = packed1;
    }
    uint64_t desc_V = make_smem_desc_swizzled(smem_V, 128);

    uint32_t idesc = make_instr_desc_fn(128, 64, 0, 1);
    uint32_t o_base = (uint32_t)__cvta_generic_to_shared(o_tmem);
    umma_f16_cg1(&o_base, &p_tmem[K_step], desc_V, idesc, accum);
}

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2, 
        globalAddress,
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

__global__ void causal_mha_lse_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O,
    float* LSE,
    uint32_t S,
    uint32_t H) 
{
    setmaxnreg_inc_sync_fn<256>();

    uint32_t q_start = blockIdx.x * 128;
    int bh_idx = (blockIdx.x * 128 + threadIdx.x) / S;
    int head_idx = bh_idx % H;
    int batch_idx = bh_idx / H;

    if (q_start + threadIdx.x >= S) return;

    __shared__ __align__(128) __nv_bfloat16 smem_Q0[128 * 64];
    __shared__ __align__(128) __nv_bfloat16 smem_Q1[128 * 64];
    __shared__ __align__(128) __nv_bfloat16 smem_K0[128 * 64];
    __shared__ __align__(128) __nv_bfloat16 smem_K1[128 * 64];
    __shared__ __align__(128) __nv_bfloat16 smem_V0[128 * 64];
    __shared__ __align__(128) __nv_bfloat16 smem_V1[128 * 64];
    __shared__ __align__(128) __nv_bfloat16 smem_V_transposed[64 * 128];
    __shared__ __align__(8) uint64_t bar[1];

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar, 1);
    }
    __syncthreads();

    uint32_t* q_tmem = (uint32_t*)smem_Q0;
    uint32_t* k_tmem = (uint32_t*)smem_K0;
    uint32_t* p_tmem = (uint32_t*)smem_V0;
    uint32_t* o_tmem = (uint32_t*)smem_V1;

    if (threadIdx.x == 0) {
        tmem_alloc_fn(q_tmem, 128);
        tmem_alloc_fn(k_tmem, 128);
        tmem_alloc_fn(p_tmem, 128);
        tmem_alloc_fn(o_tmem, 128);
    }
    __syncthreads();

    int phase = 0;
    uint32_t kv_start = bh_idx * S + q_start;
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar, 32768); 
        tma_load_2d_fn(&tma_Q, bar, smem_Q0, 0, kv_start);
        tma_load_2d_fn(&tma_Q, bar, smem_Q1, 64, kv_start);
        tma_load_2d_fn(&tma_K, bar, smem_K0, 0, kv_start);
        tma_load_2d_fn(&tma_K, bar, smem_K1, 64, kv_start);
        tma_load_2d_fn(&tma_V, bar, smem_V0, 0, kv_start);
        tma_load_2d_fn(&tma_V, bar, smem_V1, 64, kv_start);
    }

    if ((q_start + 128) / 128 > 0) {
        mbarrier_wait_fn(bar, phase);
        phase ^= 1;
    }

    float m_val[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    float l_val[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float o_reg[128];
    for(int i=0; i<128; ++i) o_reg[i] = 0.0f;

    uint32_t q_end = min(q_start + 128, S);
    
    for (uint32_t k_start = 0; k_start <= q_end - 128; k_start += 128) {
        uint32_t kv_start_next = bh_idx * S + (k_start + 128);
        if (k_start + 128 < min(q_start + 128, S)) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(bar, 32768); 
                tma_load_2d_fn(&tma_K, bar, smem_K0, 0, kv_start_next);
                tma_load_2d_fn(&tma_K, bar, smem_K1, 64, kv_start_next);
                tma_load_2d_fn(&tma_V, bar, smem_V0, 0, kv_start_next);
                tma_load_2d_fn(&tma_V, bar, smem_V1, 64, kv_start_next);
            }
        }

        mbarrier_wait_fn(bar, phase);
        phase ^= 1;

        for (int K_step = 0; K_step < 64; K_step += 16) {
            read_swizzled_to_tmem(smem_Q0, q_tmem, K_step);
            read_swizzled_to_tmem(smem_K0, k_tmem, K_step);
        }
        uint64_t desc_Q0 = make_smem_desc_swizzled(smem_Q0, 128);
        uint64_t desc_K0 = make_smem_desc_swizzled(smem_K0, 128);
        uint32_t idesc_QK = make_instr_desc_fn(128, 128, 0, 0);
        umma_f16_cg1(q_tmem, q_tmem, desc_K0, idesc_QK, 0);
        
        for (int K_step = 0; K_step < 64; K_step += 16) {
            read_swizzled_to_tmem(smem_Q1, q_tmem, K_step);
            read_swizzled_to_tmem(smem_K1, k_tmem, K_step);
        }
        uint64_t desc_Q1 = make_smem_desc_swizzled(smem_Q1, 128);
        uint64_t desc_K1 = make_smem_desc_swizzled(smem_K1, 128);
        umma_f16_cg1(q_tmem, q_tmem, desc_K1, idesc_QK, 1);

        float s_reg[128];
        for (int i = 0; i < 128 / 8; ++i) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(q_tmem + K_step + i * 8));
            s_reg[i * 8 + 0] = __uint_as_float(r0);
            s_reg[i * 8 + 1] = __uint_as_float(r1);
            s_reg[i * 8 + 2] = __uint_as_float(r2);
            s_reg[i * 8 + 3] = __uint_as_float(r3);
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

        int row = threadIdx.x;
        int global_q_idx = q_start + row;
        
        float m_row = -INFINITY;
        for(int c = 0; c < 128; c++) {
            int global_k_idx = k_start + c;
            float val = s_reg[c] * 0.08838834764f;
            if (global_k_idx > global_q_idx || global_k_idx >= S) {
                val = -INFINITY;
            }
            s_reg[c] = val;
            m_row = fmaxf(m_row, val);
        }

        int warp_id = row / 32;
        int lane_id = row % 32;

        __shared__ float m_local[4][32];
        if (lane_id == 0) m_local[warp_id][0] = m_row;
        __sync_warp();

        m_row = m_local[warp_id][lane_id];
        for (int offset = 16; offset > 0; offset /= 2) {
            m_row = fmaxf(m_row, __shfl_xor_sync(0xffffffff, m_row, offset));
        }

        float l_row = 0.0f;
        for(int c = 0; c < 128; c++) {
            if (m_row > -INFINITY) {
                float diff = m_row - s_reg[c];
                float e = fast_exp2f_fn(diff * 1.4426950408889634f);
                s_reg[c] = e;
                l_row += e;
            }
        }

        if (lane_id == 0) m_local[warp_id][0] = l_row;
        __sync_warp();
        l_row = m_local[warp_id][lane_id];
        for (int offset = 16; offset > 0; offset /= 2) {
            l_row += __shfl_xor_sync(0xffffffff, l_row, offset);
        }

        float m_old = m_val[warp_id];
        float m_new = fmaxf(m_old, m_row);
        float l_old = l_val[warp_id];
        float l_new = l_old * fast_exp2f_fn((m_old - m_new) * 1.4426950408889634f) + l_row * fast_exp2f_fn((m_row - m_new) * 1.4426950408889634f);
        
        m_val[warp_id] = m_new;
        l_val[warp_id] = l_new;

        for(int c = 0; c < 128; c++) {
            float e = fast_exp2f_fn((m_row - s_reg[c]) * 1.4426950408889634f);
            s_reg[c] = e * fast_exp2f_fn((m_row - m_new) * 1.4426950408889634f);
        }

        for(int c = 0; c < 128; c++) {
            if (m_new > -INFINITY) {
                s_reg[c] /= l_new;
            } else {
                s_reg[c] = 0.0f;
            }
        }

        for(int i = 0; i < 2; i++) {
            uint32_t packed0 = pack_bf16_fn(__float_as_uint(s_reg[i*4 + 0]), __float_as_uint(s_reg[i*4 + 1]));
            uint32_t packed1 = pack_bf16_fn(__float_as_uint(s_reg[i*4 + 2]), __float_as_uint(s_reg[i*4 + 3]));
            *(uint32_t*)&p_tmem[K_step + i*4] = packed0;
            *(uint32_t*)&p_tmem[K_step + i*4 + 1] = packed1;
        }
        
        for (int i = threadIdx.x; i < 128 * 64; i += blockDim.x) {
            int row = i / 64;
            int col = i % 64;
            int sx = (row % 8) ^ (col / 8);
            int byte_offset = row * 128 + sx * 16 + (col % 8) * 2;
            __nv_bfloat16 val = *(__nv_bfloat16*)(smem_V0 + byte_offset);
            smem_V_transposed[col * 128 + row] = val;
        }
        __syncthreads();

        for (int K_step = 0; K_step < 64; K_step += 16) {
            uint32_t accum = (K_step == 0) ? 0 : 1;
            umma_cg1_p_v(p_tmem, o_tmem, smem_V_transposed, K_step);
        }
        __syncthreads();
        
        for (int K_step = 0; K_step < 64; K_step += 16) {
            uint32_t accum = (K_step == 0) ? 0 : 1;
            umma_cg1_p_v(p_tmem, o_tmem, smem_V_transposed + 4096, K_step);
        }
        __syncthreads();
    }

    for (int i = 0; i < 128 / 8; ++i) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
           : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(o_tmem + i * 8));
        o_reg[i * 8 + 0] = __uint_as_float(r0);
        o_reg[i * 8 + 1] = __uint_as_float(r1);
        o_reg[i * 8 + 2] = __uint_as_float(r2);
        o_reg[i * 8 + 3] = __uint_as_float(r3);
    }
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

    int warp_id = threadIdx.x / 32;
    float l_final = l_val[warp_id];
    for(int i = 0; i < 128; ++i) {
        o_reg[i] /= l_final;
    }
    
    __nv_bfloat16* out_O = O + (batch_idx * H + head_idx) * S * 128 + q_start * 128;
    for (int col = 0; col < 128; col += 4) {
        uint32_t packed0 = pack_bf16_fn(__float_as_uint(o_reg[col]), __float_as_uint(o_reg[col+1]));
        uint32_t packed1 = pack_bf16_fn(__float_as_uint(o_reg[col+2]), __float_as_uint(o_reg[col+3]));
        uint64_t out = ((uint64_t)packed1 << 32) | packed0;
        *(uint64_t*)&out_O[threadIdx.x * 128 + col] = out;
    }

    if (threadIdx.x == 0) {
        float lse_val = m_val[0] + logf(l_val[0]);
        LSE[(batch_idx * H + head_idx) * S + q_start] = lse_val;
    }
    
    tmem_dealloc_fn(*(uint32_t*)q_tmem, 128);
    tmem_dealloc_fn(*(uint32_t*)k_tmem, 128);
    tmem_dealloc_fn(*(uint32_t*)p_tmem, 128);
    tmem_dealloc_fn(*(uint32_t*)o_tmem, 128);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    uint32_t B = Q.size(0);
    uint32_t H = Q.size(1);
    uint32_t S = Q.size(2);
    uint32_t D = Q.size(3);

    CUtensorMap tma_Q, tma_K, tma_V;
    CUresult res;
    res = create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), D, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA Q encode error\n"); exit(1); }
    res = create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), D, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    res = create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), D, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    uint32_t* o_mem = (uint32_t*)O.data_ptr();
    cudaMemsetAsync(o_mem, 0, O.nbytes(), static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id)));

    __nv_bfloat16* o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_ptr = static_cast<float*>(LSE.data_ptr());

    dim3 grid((S + 127) / 128);
    dim3 block(128);

    CUDA_CHECK(cudaFuncSetAttribute(
        causal_mha_lse_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        114704
    ));

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    causal_mha_lse_kernel<<<grid, block, 114704, stream>>>(tma_Q, tma_K, tma_V, o_ptr, lse_ptr, S, H);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

} // namespace tvm_ffi_example_cuda