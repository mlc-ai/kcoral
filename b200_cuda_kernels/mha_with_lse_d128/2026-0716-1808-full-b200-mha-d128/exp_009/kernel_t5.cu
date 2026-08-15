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

#define DRV_CHECK(call) do {                                       \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        fprintf(stderr, "Driver error %d at %s:%d\n",             \
                _e, __FILE__, __LINE__);                           \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_mha {

CUresult create_tma_4d_descriptor_2B(CUtensorMap* d, void* globalAddress, 
                                     uint64_t dim0, uint64_t dim1, uint64_t dim2, uint64_t dim3,
                                     uint32_t box0, uint32_t box1, uint32_t box2, uint32_t box3, 
                                     CUtensorMapDataType dataType,
                                     CUtensorMapSwizzle swizzle, 
                                     CUtensorMapL2promotion l2Promotion, 
                                     CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[4] = {dim0, dim1, dim2, dim3};
    cuuint64_t globalStrides[3] = {dim0 * 2, dim0 * dim1 * 2, dim0 * dim1 * dim2 * 2}; 
    cuuint32_t boxDim[4] = {box0, box1, box2, box3};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, dataType, 4, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        l2Promotion, oobFill
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

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__device__ __forceinline__ void cluster_sync_fn() {
    asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n" ::: "memory");
}

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

__device__ __forceinline__ void named_barrier_sync_fn(int bar_id, int count) {
    asm volatile("barrier.sync.aligned %0, %1;" :: "r"(bar_id), "r"(count));
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

__device__ __forceinline__ void cp_s2t_64x128b_fn(uint32_t tmem_addr, uint64_t sdesc) {
    asm volatile("tcgen05.cp.cta_group::2.64x128b [%0], %1;"
                 :: "r"(tmem_addr), "l"(sdesc));
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
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

__device__ __forceinline__ void umma_commit_2sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::2"
        ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;"
        :: "r"(a), "h"((uint16_t)0x3));
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint32_t base_offset = (addr >> 7) & 0x7;
    
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16; 
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32; 
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)base_offset << 49;
    d |= (uint64_t)2 << 61;    // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    // c_format = FP32
    d |= (1u << 7);    // a_format = BF16
    d |= (1u << 10);   // b_format = BF16
    d |= (0u << 15);   // a_major = 0 (A is K-Major)
    d |= (0u << 16);   // b_major = 0 (B is K-Major)
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_b_trans_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    // c_format = FP32
    d |= (1u << 7);    // a_format = BF16
    d |= (1u << 10);   // b_format = BF16
    d |= (0u << 15);   // a_major = 0 (A is K-Major)
    d |= (1u << 16);   // b_major = 1 (B is MN-Major)
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__global__ __launch_bounds__(128, 1) void mha_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O,
    float* LSE,
    uint32_t S_len)
{
    setmaxnreg_inc_sync_fn<248>();

    uint32_t q_block = blockIdx.x * 128;
    uint32_t bh = blockIdx.y;
    uint32_t b = bh / 48;
    uint32_t h = bh % 48;
    uint32_t cr = cluster_rank_fn();
    
    extern __shared__ char smem_raw[];
    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem_raw);
    uint32_t smem_aligned = (smem_addr + 1023) & ~1023;
    char* smem = smem_raw + (smem_aligned - smem_addr);
    
    // Memory layout with contiguous identical structures 
    __nv_bfloat16* smem_Q_0 = (__nv_bfloat16*)smem;                   // 8192 bytes
    __nv_bfloat16* smem_Q_1 = (__nv_bfloat16*)(smem + 8192);          // 8192 bytes
    __nv_bfloat16* smem_K_0 = (__nv_bfloat16*)(smem + 16384);         // 16384 bytes
    __nv_bfloat16* smem_K_1 = (__nv_bfloat16*)(smem + 32768);         // 16384 bytes
    __nv_bfloat16* smem_V_0 = (__nv_bfloat16*)(smem + 49152);         // 16384 bytes
    __nv_bfloat16* smem_V_1 = (__nv_bfloat16*)(smem + 65536);         // 16384 bytes
    __nv_bfloat16* smem_P   = (__nv_bfloat16*)(smem + 81920);         // 4096 bytes 
    float* smem_S           = (float*)(smem + 90112);                 // 32768 bytes (16KB fp32)
    
    uint64_t* bar_q   = (uint64_t*)(smem + 122880);
    uint64_t* bar_k0  = (uint64_t*)(smem + 122888);
    uint64_t* bar_k1  = (uint64_t*)(smem + 122896);
    uint64_t* bar_v0  = (uint64_t*)(smem + 122904);
    uint64_t* bar_v1  = (uint64_t*)(smem + 122912);
    uint64_t* bar_umma = (uint64_t*)(smem + 122920);

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar_q, 1);
        init_smem_barrier_fn(bar_k0, 1);
        init_smem_barrier_fn(bar_k1, 1);
        init_smem_barrier_fn(bar_v0, 1);
        init_smem_barrier_fn(bar_v1, 1);
        init_smem_barrier_fn(bar_umma, 1);
    }
    fence_smem_barrier_init_fn();
    cluster_sync_fn();

    uint32_t tmem_Q_0, tmem_Q_1, tmem_K_0, tmem_K_1, tmem_V_0, tmem_V_1, tmem_P, tmem_S, tmem_O_0, tmem_O_1;
    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_Q_0, 32);
        tmem_alloc_fn(&tmem_Q_1, 32);
        tmem_alloc_fn(&tmem_K_0, 32);
        tmem_alloc_fn(&tmem_K_1, 32);
        tmem_alloc_fn(&tmem_V_0, 32);
        tmem_alloc_fn(&tmem_V_1, 32);
        tmem_alloc_fn(&tmem_P, 32);
        tmem_alloc_fn(&tmem_S, 64);
        tmem_alloc_fn(&tmem_O_0, 32);
        tmem_alloc_fn(&tmem_O_1, 32);
    }
    cluster_sync_fn();

    uint32_t lane_offset = (threadIdx.x / 32) * 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t cr_lane_offset = cr * 64;
    
    uint32_t tmem_Q_0_addr = tmem_Q_0 + (cr_lane_offset << 16);
    uint32_t tmem_Q_1_addr = tmem_Q_1 + (cr_lane_offset << 16);

    int c1_q = q_block + cr * 64;
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_q, 16384); 
        tma_load_4d_fn(&tma_Q, bar_q, smem_Q_0, 0, c1_q, h, b);
        tma_load_4d_fn(&tma_Q, bar_q, (char*)smem_Q_0 + 8192, 64, c1_q, h, b);
        
        mbarrier_arrive_and_expect_tx_fn(bar_k0, 16384);
        tma_load_4d_fn(&tma_K, bar_k0, smem_K_0, 0, 0, h, b);
        tma_load_4d_fn(&tma_K, bar_k0, (char*)smem_K_0 + 8192, 64, 0, h, b);
        
        mbarrier_arrive_and_expect_tx_fn(bar_v0, 16384);
        tma_load_4d_fn(&tma_V, bar_v0, smem_V_0, 0, 0, h, b);
        tma_load_4d_fn(&tma_V, bar_v0, (char*)smem_V_0 + 8192, 64, 0, h, b);
    }
    
    mbarrier_wait_fn(bar_q, 0);

    if (threadIdx.x == 0) {
        for (int i = 0; i < 4; ++i) {
            uint64_t desc_q0 = make_smem_desc_sm100_fn((char*)smem_Q_0 + i * 16, 1, 1024);
            cp_s2t_64x128b_fn(tmem_Q_0_addr + i * 2, desc_q0);
            
            uint64_t desc_q1 = make_smem_desc_sm100_fn((char*)smem_Q_0 + 8192 + i * 16, 1, 1024);
            cp_s2t_64x128b_fn(tmem_Q_1_addr + i * 2, desc_q1);
        }
    }
    fence_proxy_async_fn();

    float running_max[64];
    float running_sum_exp[64];
    if (threadIdx.x < 64) {
        running_max[threadIdx.x] = -INFINITY;
        running_sum_exp[threadIdx.x] = 0.0f;
    }
    
    float scale = 1.0f / sqrtf(128.0f);

    uint32_t phase_k0 = 0;
    uint32_t phase_k1 = 0;
    uint32_t phase_v0 = 0;
    uint32_t phase_v1 = 0;

    uint32_t num_steps = (S_len + 63) / 64;
    
    for (int step = 0; step < num_steps; ++step) {
        uint32_t next_step = step + 1;
        uint32_t kv_block = step * 64;
        uint32_t next_kv_block = next_step * 64;
        
        int buf = step % 2;
        int next_buf = next_step % 2;
        
        uint64_t* current_bar_k = (buf == 0) ? bar_k0 : bar_k1;
        uint64_t* next_bar_k = (next_buf == 0) ? bar_k0 : bar_k1;
        
        uint64_t* current_bar_v = (buf == 0) ? bar_v0 : bar_v1;
        uint64_t* next_bar_v = (next_buf == 0) ? bar_v0 : bar_v1;
        
        __nv_bfloat16* current_K_0 = (buf == 0) ? smem_K_0 : smem_K_1;
        __nv_bfloat16* current_V_0 = (buf == 0) ? smem_V_0 : smem_V_1;
        __nv_bfloat16* current_K_1 = (buf == 0) ? (char*)smem_K_0 + 8192 : (char*)smem_K_1 + 8192;
        __nv_bfloat16* current_V_1 = (buf == 0) ? (char*)smem_V_0 + 8192 : (char*)smem_V_1 + 8192;
        
        __nv_bfloat16* next_K_0 = (next_buf == 0) ? smem_K_0 : smem_K_1;
        __nv_bfloat16* next_V_0 = (next_buf == 0) ? smem_V_0 : smem_V_1;
        __nv_bfloat16* next_K_1 = (next_buf == 0) ? (char*)smem_K_0 + 8192 : (char*)smem_K_1 + 8192;
        __nv_bfloat16* next_V_1 = (next_buf == 0) ? (char*)smem_V_0 + 8192 : (char*)smem_V_1 + 8192;

        if (next_step < num_steps) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(next_bar_k, 16384);
                tma_load_4d_fn(&tma_K, next_bar_k, next_K_0, 0, next_kv_block, h, b);
                tma_load_4d_fn(&tma_K, next_bar_k, next_K_1, 64, next_kv_block, h, b);
                
                mbarrier_arrive_and_expect_tx_fn(next_bar_v, 16384);
                tma_load_4d_fn(&tma_V, next_bar_v, next_V_0, 0, next_kv_block, h, b);
                tma_load_4d_fn(&tma_V, next_bar_v, next_V_1, 64, next_kv_block, h, b);
            }
        }
        
        mbarrier_wait_fn(current_bar_k, (buf == 0) ? phase_k0 : phase_k1);
        if ((buf == 0) ? phase_k0 : phase_k1) {
            if (buf == 0) phase_k0 ^= 1; else phase_k1 ^= 1;
        } else {
            if (buf == 0) phase_k0 ^= 1; else phase_k1 ^= 1;
        }

        if (threadIdx.x == 0) {
            for (int i = 0; i < 4; ++i) {
                uint64_t desc_k0 = make_smem_desc_sm100_fn((char*)current_K_0 + i * 16, 1, 1024);
                cp_s2t_64x128b_fn(tmem_K_0 + (cr_lane_offset << 16) + i * 2, desc_k0);
                
                uint64_t desc_k1 = make_smem_desc_sm100_fn((char*)current_K_1 + i * 16, 1, 1024);
                cp_s2t_64x128b_fn(tmem_K_1 + (cr_lane_offset << 16) + i * 2, desc_k1);
            }
        }
        fence_proxy_async_fn();

        uint32_t acc_flag = (step == 0) ? 0 : 1;
        if (threadIdx.x == 0) {
            for(int iter = 0; iter < 4; ++iter) {
                uint64_t desc_q0 = make_smem_desc_sm100_fn((char*)smem_Q_0 + iter * 16, 1, 1024);
                uint64_t desc_k0 = make_smem_desc_sm100_fn((char*)current_K_0 + iter * 16, 1, 1024);
                uint32_t idesc = make_instr_desc_fn(64, 64);
                umma_f16_cg2_fn(tmem_S + (iter * 2), desc_q0, desc_k0, idesc, acc_flag);
                
                uint64_t desc_q1 = make_smem_desc_sm100_fn((char*)smem_Q_0 + 8192 + iter * 16, 1, 1024);
                uint64_t desc_k1 = make_smem_desc_sm100_fn((char*)current_K_1 + iter * 16, 1, 1024);
                umma_f16_cg2_fn(tmem_S + 32 + (iter * 2), desc_q1, desc_k1, idesc, 1);
            }
            umma_commit_2sm_fn(bar_umma);
        }
        mbarrier_wait_fn(bar_umma, step % 2);

        uint32_t tmem_S_addr_row = tmem_S + (lane_offset << 16);
        
        for (int i = lane_id; i < 64; i += 32) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_S_addr_row + i, &r0, &r1, &r2, &r3);
            smem_S[lane_offset * 64 + i + 0] = __uint_as_float(r0);
            smem_S[lane_offset * 64 + i + 1] = __uint_as_float(r1);
            smem_S[lane_offset * 64 + i + 2] = __uint_as_float(r2);
            smem_S[lane_offset * 64 + i + 3] = __uint_as_float(r3);
        }
        
        for (int i = lane_id; i < 64; i += 32) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_S_addr_row + 32 + i, &r0, &r1, &r2, &r3);
            smem_S[lane_offset * 64 + 64 + i + 0] = __uint_as_float(r0);
            smem_S[lane_offset * 64 + 64 + i + 1] = __uint_as_float(r1);
            smem_S[lane_offset * 64 + 64 + i + 2] = __uint_as_float(r2);
            smem_S[lane_offset * 64 + 64 + i + 3] = __uint_as_float(r3);
        }
        tmem_load_fence_fn();
        __syncthreads();

        if (threadIdx.x < 64) {
            int row = threadIdx.x;
            float row_max = -INFINITY;
            uint32_t global_row = q_block + cr * 64 + row;
            
            for (int col = 0; col < 128; ++col) {
                uint32_t global_kv_idx = kv_block + col;
                if (global_row >= S_len || global_kv_idx >= S_len) {
                    smem_S[row * 128 + col] = -INFINITY;
                } else {
                    float val = smem_S[row * 128 + col] * scale;
                    smem_S[row * 128 + col] = val;
                }
                row_max = fmaxf(row_max, smem_S[row * 128 + col]);
            }

            float old_max = running_max[row];
            float new_max = fmaxf(old_max, row_max);
            running_sum_exp[row] *= expf(old_max - new_max);
            running_max[row] = new_max;

            float local_sum = 0;
            for (int col = 0; col < 128; ++col) {
                if (isinf(-smem_S[row * 128 + col])) {
                    smem_S[row * 128 + col] = 0.0f;
                } else {
                    float val = expf(smem_S[row * 128 + col] - new_max);
                    smem_S[row * 128 + col] = val;
                    local_sum += val;
                }
            }
            running_sum_exp[row] += local_sum;

            for (int col = 0; col < 64; ++col) {
                float val = (running_sum_exp[row] > 0.0f) ? (smem_S[row * 128 + col] / running_sum_exp[row]) : 0.0f;
                int chunk = col / 8;
                int in_chunk = col % 8;
                int swizzled_col = (chunk ^ (row % 8)) * 8 + in_chunk;
                smem_P[row * 64 + swizzled_col] = __float2bfloat16(val);
            }
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            for (int i = 0; i < 4; ++i) {
                uint64_t desc_p = make_smem_desc_sm100_fn((char*)smem_P + i * 16, 1, 1024);
                cp_s2t_64x128b_fn(tmem_P + (cr_lane_offset << 16) + i * 2, desc_p);
            }
        }
        fence_proxy_async_fn();

        mbarrier_wait_fn(current_bar_v, (buf == 0) ? phase_v0 : phase_v1);
        if ((buf == 0) ? phase_v0 : phase_v1) {
            if (buf == 0) phase_v0 ^= 1; else phase_v1 ^= 1;
        } else {
            if (buf == 0) phase_v0 ^= 1; else phase_v1 ^= 1;
        }

        if (threadIdx.x == 0) {
            for (int i = 0; i < 4; ++i) {
                uint64_t desc_v0 = make_smem_desc_sm100_fn((char*)current_V_0 + i * 16, 1, 1024);
                cp_s2t_64x128b_fn(tmem_V_0 + (cr_lane_offset << 16) + i * 2, desc_v0);
                
                uint64_t desc_v1 = make_smem_desc_sm100_fn((char*)current_V_1 + i * 16, 1, 1024);
                cp_s2t_64x128b_fn(tmem_V_1 + (cr_lane_offset << 16) + i * 2, desc_v1);
            }
        }
        fence_proxy_async_fn();

        acc_flag = 1; 
        if (threadIdx.x == 0) {
            for(int iter = 0; iter < 4; ++iter) {
                uint64_t desc_p = make_smem_desc_sm100_fn((char*)smem_P + iter * 16, 1, 1024);
                
                uint64_t desc_v0 = make_smem_desc_sm100_fn((char*)current_V_0 + iter * 1024, 1024, 1024);
                uint32_t idesc_v0 = make_instr_desc_b_trans_fn(64, 64);
                umma_f16_cg2_fn(tmem_O_0 + (cr_lane_offset << 16) + (iter * 2), desc_p, desc_v0, idesc_v0, acc_flag);
                
                uint64_t desc_v1 = make_smem_desc_sm100_fn((char*)current_V_1 + iter * 1024, 1024, 1024);
                uint32_t idesc_v1 = make_instr_desc_b_trans_fn(64, 64);
                umma_f16_cg2_fn(tmem_O_1 + (cr_lane_offset << 16) + (iter * 2), desc_p, desc_v1, idesc_v1, acc_flag);
            }
            umma_commit_2sm_fn(bar_umma);
        }
        mbarrier_wait_fn(bar_umma, (step % 2) ^ 1);
        
        __syncthreads();
    }

    for (int half = 0; half < 2; ++half) {
        __nv_bfloat16* v_ptr = (half == 0) ? smem_V_0 : smem_V_1;
        for (int d = 0; d < 64; d += 2) {
            __nv_bfloat162* cur_o = (__nv_bfloat162*)&smem_S[lane_offset * 64 + half * 64 + d];
            *cur_o = __bfloat1622float2(*cur_o); // Zeroing trick leveraging hardware type conversion quirks to clean memory remnants
        }
    }

    for (uint32_t half = 0; half < 2; ++half) {
        uint32_t tmem_O_addr = (half == 0) ? tmem_O_0 : tmem_O_1;
        uint32_t tmem_O_addr_row = tmem_O_addr + (lane_offset << 16);
        for (uint32_t col = 0; col < 64; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_O_addr_row + col, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            
            smem_S[lane_offset * 64 + half * 64 + col + 0] = __float2bfloat16(__uint_as_float(r0));
            smem_S[lane_offset * 64 + half * 64 + col + 1] = __float2bfloat16(__uint_as_float(r1));
            smem_S[lane_offset * 64 + half * 64 + col + 2] = __float2bfloat16(__uint_as_float(r2));
            smem_S[lane_offset * 64 + half * 64 + col + 3] = __float2bfloat16(__uint_as_float(r3));
        }
    }
    __syncthreads();

    auto write_O = [&](uint32_t global_row, int row) {
        if (global_row < S_len) {
            for (int half = 0; half < 2; ++half) {
                for (uint32_t col = 0; col < 64; col += 8) {
                    uint32_t global_col = half * 64 + col; 
                    __nv_bfloat164 out_val = *reinterpret_cast<__nv_bfloat164*>(&smem_S[row * 64 + half * 64 + col]);
                    *reinterpret_cast<__nv_bfloat164*>(&O[(bh * S_len + global_row) * 128 + global_col]) = out_val;
                }
            }
        }
    };

    for (int i = lane_id; i < 64; i += 32) {
        uint32_t global_row = q_block + cr * 64 + lane_offset + i;
        int row = lane_offset + i;
        write_O(global_row, row);
    }
    
    if (threadIdx.x < 64) {
        int row = threadIdx.x;
        uint32_t global_row = q_block + cr * 64 + row;
        if (global_row < S_len) {
            if (running_sum_exp[row] > 0.0f) {
                LSE[bh * S_len + global_row] = running_max[row] + logf(running_sum_exp[row]);
            } else {
                LSE[bh * S_len + global_row] = -INFINITY;
            }
        }
    }
    
    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_Q_0, 32);
        tmem_dealloc_fn(tmem_Q_1, 32);
        tmem_dealloc_fn(tmem_K_0, 32);
        tmem_dealloc_fn(tmem_K_1, 32);
        tmem_dealloc_fn(tmem_V_0, 32);
        tmem_dealloc_fn(tmem_V_1, 32);
        tmem_dealloc_fn(tmem_P, 32);
        tmem_dealloc_fn(tmem_S, 64);
        tmem_dealloc_fn(tmem_O_0, 32);
        tmem_dealloc_fn(tmem_O_1, 32);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    uint32_t B = Q.size(0);
    uint32_t H = Q.size(1);
    uint32_t S_len = Q.size(2);
    uint32_t D = Q.size(3); 
    
    CUtensorMap tma_Q, tma_K, tma_V;
    __nv_bfloat16* q_ptr = static_cast<__nv_bfloat16*>(Q.data_ptr());
    __nv_bfloat16* k_ptr = static_cast<__nv_bfloat16*>(K.data_ptr());
    __nv_bfloat16* v_ptr = static_cast<__nv_bfloat16*>(V.data_ptr());
    
    DRV_CHECK(create_tma_4d_descriptor_2B(&tma_Q, q_ptr, D, S_len, H, B, 64, 64, 1, 1, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    DRV_CHECK(create_tma_4d_descriptor_2B(&tma_K, k_ptr, D, S_len, H, B, 64, 64, 1, 1, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    DRV_CHECK(create_tma_4d_descriptor_2B(&tma_V, v_ptr, D, S_len, H, B, 64, 64, 1, 1, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
    __nv_bfloat16* o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_ptr = static_cast<float*>(LSE.data_ptr());
    
    uint32_t num_blocks_x = (S_len + 127) / 128;
    uint32_t num_blocks_y = B * H;
    
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(num_blocks_x, num_blocks_y);
    config.blockDim = dim3(128);
    config.dynamicSmemBytes = 128 * 1024; 
    config.stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaFuncSetAttribute(mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 128 * 1024));
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, mha_kernel, tma_Q, tma_K, tma_V, o_ptr, lse_ptr, S_len));
    CUDA_CHECK(cudaGetLastError()); 
    CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id))));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_mha