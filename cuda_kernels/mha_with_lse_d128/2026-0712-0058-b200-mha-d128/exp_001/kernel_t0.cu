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

// -------------------------------------------------------------------------------------
// Device intrinsics
// -------------------------------------------------------------------------------------

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_dec_sync_fn() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
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

__device__ __forceinline__ void fence_proxy_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
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

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
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
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint64_t modify_base_offset(uint64_t desc, void* new_smem_ptr) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(new_smem_ptr);
    uint64_t new_d = (desc & ~0x3FFF) | ((addr & 0x3FFFF) >> 4);
    uint32_t base_offset = (addr >> 7) & 0x7;
    new_d &= ~(0x7ULL << 49);
    new_d |= ((uint64_t)base_offset << 49);
    return new_d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
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

__device__ __forceinline__ uint32_t make_instr_desc_mixed_major_fn(uint32_t M, uint32_t N) {
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

__device__ __forceinline__ void named_barrier_sync_fn(int bar_id, int count) {
    asm volatile("barrier.sync.aligned %0, %1;" :: "r"(bar_id), "r"(count));
}

__device__ __forceinline__ void named_barrier_sync_fn_warp() {
    asm volatile("barrier.sync.aligned 2, 32;" ::: "memory");
}

__device__ __forceinline__ void named_barrier_arrive_fn_warp() {
    asm volatile("barrier.arrive.aligned 2, 32;" ::: "memory");
}

// -------------------------------------------------------------------------------------
// Kernel
// -------------------------------------------------------------------------------------

__global__ void attention_forward_cluster(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O,
    float* LSE,
    uint64_t S,
    uint64_t D) 
{
    if (threadIdx.x < 32) {
        setmaxnreg_inc_sync_fn<256>();
    }

    extern __shared__ __align__(128) uint8_t smem_pool[];
    uintptr_t pool_addr = (uintptr_t)smem_pool;
    pool_addr = (pool_addr + 127) & ~127; 
    
    uint64_t* mbar_Q  = reinterpret_cast<uint64_t*>(pool_addr);
    uint64_t* mbar_K  = mbar_Q + 1;
    uint64_t* mbar_V  = mbar_K + 1;
    char* smem_Q      = reinterpret_cast<char*>(pool_addr + 10);
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_Q, 1); 
        init_smem_barrier_fn(mbar_K, 1); 
        init_smem_barrier_fn(mbar_V, 1); 
    }
    fence_smem_barrier_init_fn();
    
    uint32_t* tmem_S = reinterpret_cast<uint32_t*>(pool_addr + 81920);
    uint32_t* tmem_O = reinterpret_cast<uint32_t*>(pool_addr + 81968);
    
    if (threadIdx.x < 32) {
        tmem_alloc_fn(tmem_S, 64);
        tmem_alloc_fn(tmem_O, 128);
    }

    uint64_t row_base = (uint64_t)blockIdx.x * 64;
    bool valid[128];
    if (threadIdx.x < 128) {
        valid[threadIdx.x] = (row_base + threadIdx.x < S);
    }
    __syncthreads();
    
    char* s_Q0 = smem_Q;
    char* s_Q1 = smem_Q + 16384;
    char* s_K0 = smem_Q + 32768;
    char* s_K1 = smem_Q + 32768 + 16384;
    char* s_P  = smem_Q + 65536;
    char* s_V0 = smem_Q + 65536 + 8192;
    char* s_V1 = smem_Q + 65536 + 8192 + 16384;

    uint32_t blockRowIdx = row_base / S;
    uint32_t blkOffset = blockRowIdx * S;
    uint32_t myRow = row_base % S;

    int32_t c1_Q = (int32_t)(blkOffset + myRow);
    int32_t c0_Q0 = 0;
    int32_t c0_Q1 = 64;
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 16384 * 2);
        tma_load_2d_fn(&tma_Q, mbar_Q, s_Q0, c0_Q0, c1_Q);
        tma_load_2d_fn(&tma_Q, mbar_Q, s_Q1, c0_Q1, c1_Q);
    }
    
    mbarrier_wait_fn(mbar_Q, 0);
    
    uint64_t* mbar_K_local = mbar_K;
    uint64_t* mbar_V_local = mbar_V;
    
    int cur_k = 0;
    int cur_v = 0;
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&mbar_K_local[cur_k], 16384 * 2);
        tma_load_2d_fn(&tma_K, &mbar_K_local[cur_k], s_K0, 0, c1_Q);
        tma_load_2d_fn(&tma_K, &mbar_K_local[cur_k], s_K1, 64, c1_Q);
        
        mbarrier_arrive_and_expect_tx_fn(&mbar_V_local[cur_v], 16384 * 2);
        tma_load_2d_fn(&tma_V, &mbar_V_local[cur_v], s_V0, 0, c1_Q);
        tma_load_2d_fn(&tma_V, &mbar_V_local[cur_v], s_V1, 64, c1_Q);
    }

    uint64_t desc_Q = make_smem_desc_sm100_fn(s_Q0, 1, 1024);
    uint64_t desc_K = make_smem_desc_sm100_fn(s_K0, 1, 1024);
    uint64_t desc_P = make_smem_desc_sm100_fn(s_P, 1, 1024);
    uint64_t desc_V0 = make_smem_desc_sm100_fn(s_V0, 8192, 1024);
    
    uint32_t idesc_QK = make_instr_desc_fn(64, 64);
    uint32_t idesc_PV = make_instr_desc_mixed_major_fn(64, 128);
    
    float row_global_max[2] = {-1e20f, -1e20f};
    float row_global_sum[2] = {0.0f, 0.0f};
    float row_global_log2sum[2] = {-1e20f, -1e20f};
    float correction[2] = {0.0f, 0.0f};
    bool first_iter = true;
    int phase_K = 0;
    
    const float SQRT_D = 1.0f / sqrtf((float)D);
    const float LOG2E = 1.4426950408889634f;
    const float LN2 = 0.6931471805599453f;
    
    int num_iters = (S + 63) / 64;

    for (int j_blk = 0; j_blk < num_iters; j_blk++) {
        mbarrier_wait_fn(&mbar_K_local[cur_k], phase_K);
        
        int n_valid[128];
        if (threadIdx.x < 128) {
            n_valid[threadIdx.x] = (j_blk * 64 + threadIdx.x < S);
        }
        __syncthreads();
        
        if (first_iter) {
            uint32_t accum = 0;
            for (int k_idx = 0; k_idx < 64; k_idx += 8) {
                int k_step = k_idx * 8;
                uint64_t desc_Q_curr0 = modify_base_offset(desc_Q, s_Q0 + k_step * 128);
                uint64_t desc_Q_curr1 = modify_base_offset(desc_Q, s_Q1 + k_step * 128);
                
                uint64_t desc_K_curr0 = modify_base_offset(desc_K, (cur_k == 0 ? s_K0 : s_K1) + k_step * 128);
                uint64_t desc_K_curr1 = modify_base_offset(desc_K, (cur_k == 0 ? s_K1 : s_K0) + k_step * 128);
                
                if (threadIdx.x == 0) {
                    if (accum == 0) {
                        umma_f16_cg2_fn(*tmem_S, desc_Q_curr0, desc_K_curr0, idesc_QK, accum);
                        umma_f16_cg2_fn(*tmem_S, desc_Q_curr1, desc_K_curr1, idesc_QK, accum);
                    } else {
                        umma_f16_cg2_fn(*tmem_S, desc_Q_curr0, desc_K_curr0, idesc_QK, accum);
                        umma_f16_cg2_fn(*tmem_S, desc_Q_curr1, desc_K_curr1, idesc_QK, accum);
                    }
                    accum = 1;
                } else {
                    // Force compiler to generate instruction bodies for all branches to prevent dead code elimination
                    if (accum == 0) {
                        asm volatile(""); 
                    } else {
                        asm volatile("");
                    }
                }
            }
            first_iter = false;
        } else {
            uint32_t accum = 1;
            for (int k_idx = 0; k_idx < 64; k_idx += 8) {
                int k_step = k_idx * 8;
                uint64_t desc_Q_curr0 = modify_base_offset(desc_Q, s_Q0 + k_step * 128);
                uint64_t desc_Q_curr1 = modify_base_offset(desc_Q, s_Q1 + k_step * 128);
                
                uint64_t desc_K_curr0 = modify_base_offset(desc_K, (cur_k == 0 ? s_K0 : s_K1) + k_step * 128);
                uint64_t desc_K_curr1 = modify_base_offset(desc_K, (cur_k == 0 ? s_K1 : s_K0) + k_step * 128);
                
                if (threadIdx.x == 0) {
                    umma_f16_cg2_fn(*tmem_S, desc_Q_curr0, desc_K_curr0, idesc_QK, accum);
                    umma_f16_cg2_fn(*tmem_S, desc_Q_curr1, desc_K_curr1, idesc_QK, accum);
                }
            }
        }
        
        if (threadIdx.x == 0) {
            umma_commit_2sm_fn(mbar_Q); 
        }
        mbarrier_wait_fn(mbar_Q, 0); 
        
        float local_max[2] = {-1e20f, -1e20f};
        float local_sum[2] = {0.0f, 0.0f};
        
        if (threadIdx.x < 64) {
            uint32_t r0, r1, r2, r3;
            float vals[64];
            for (int col_idx = 0; col_idx < 8; col_idx++) {
                tmem_load_4x_fn(col_idx * 8 + threadIdx.x, &r0, &r1, &r2, &r3);
                vals[col_idx * 4 + 0] = __uint_as_float(r0);
                vals[col_idx * 4 + 1] = __uint_as_float(r1);
                vals[col_idx * 4 + 2] = __uint_as_float(r2);
                vals[col_idx * 4 + 3] = __uint_as_float(r3);
            }
            tmem_load_fence_fn();
            
            int m_idx = threadIdx.x;
            float my_max = -1e20f;
            for(int i = 0; i < 64; i++) {
                int n_idx = i;
                if (n_idx < 64 && n_valid[n_idx]) {
                    float val = vals[i] * SQRT_D;
                    if (val > my_max) my_max = val;
                }
            }
            
            __shared__ float smem_max[64];
            if (threadIdx.x < 64) smem_max[threadIdx.x] = my_max;
            __syncthreads();
            my_max = smem_max[threadIdx.x];
            for (int offset = 1; offset < 64; offset *= 2) {
                float other = (threadIdx.x >= offset) ? smem_max[threadIdx.x - offset] : -1e20f;
                my_max = fmaxf(my_max, other);
                if (threadIdx.x < 64) smem_max[threadIdx.x] = my_max;
            }
            __syncthreads();
            my_max = smem_max[threadIdx.x];
            
            local_max[m_idx] = my_max;
            
            float my_sum = 0.0f;
            for(int i = 0; i < 64; i++) {
                int n_idx = i;
                if (n_idx < 64 && n_valid[n_idx]) {
                    float val = vals[i] * SQRT_D;
                    float p = exp2f((val - my_max) * LOG2E);
                    my_sum += p;
                    vals[i] = p;
                } else {
                    vals[i] = 0.0f;
                }
            }
            
            __shared__ float smem_sum[64];
            if (threadIdx.x < 64) smem_sum[threadIdx.x] = my_sum;
            __syncthreads();
            my_sum = smem_sum[threadIdx.x];
            for (int offset = 1; offset < 64; offset *= 2) {
                float other = (threadIdx.x >= offset) ? smem_sum[threadIdx.x - offset] : 0.0f;
                my_sum += other;
                if (threadIdx.x < 64) smem_sum[threadIdx.x] = my_sum;
            }
            __syncthreads();
            my_sum = smem_sum[threadIdx.x];
            
            local_sum[m_idx] = my_sum;
            
            float new_max = fmaxf(row_global_max[m_idx], local_max[m_idx]);
            correction[m_idx] = exp2f((row_global_max[m_idx] - new_max) * LOG2E);
            float new_sum = row_global_sum[m_idx] * correction[m_idx] + local_sum[m_idx] * exp2f((local_max[m_idx] - new_max) * LOG2E);
            
            row_global_max[m_idx] = new_max;
            row_global_sum[m_idx] = new_sum;
            row_global_log2sum[m_idx] = log2f(new_sum);
            
            for(int i = 0; i < 64; i++) {
                float p = vals[i] * exp2f((local_max[m_idx] - new_max) * LOG2E);
                int col = i;
                int swizzled_col = ((threadIdx.x & 7) ^ (col >> 3)) << 3 | (col & 7);
                __nv_bfloat16 pb = __float2bfloat16(p);
                *(uint16_t*)((char*)s_P + threadIdx.x * 128 + swizzled_col * 2) = *(uint16_t*)&pb;
            }
        }
        
        __syncthreads();
        fence_proxy_async_shared_fn(); 
        
        if (threadIdx.x < 128) {
            if (threadIdx.x == 0) {
                uint32_t accum = 1;
                for (int k_idx = 0; k_idx < 4; k_idx++) {
                    int k_step = k_idx * 8;
                    uint64_t desc_P_curr = modify_base_offset(desc_P, s_P + k_step * 128);
                    
                    uint64_t desc_V_curr0 = modify_base_offset(desc_V0, (cur_v == 0 ? s_V0 : s_V1) + k_step * 16);
                    uint64_t desc_V_curr1 = modify_base_offset(desc_V0, (cur_v == 0 ? s_V1 : s_V0) + 8192 + k_step * 16);
                    
                    umma_f16_cg2_fn(*tmem_O, desc_P_curr, desc_V_curr0, idesc_PV, accum);
                    umma_f16_cg2_fn(*tmem_O, desc_P_curr, desc_V_curr1, idesc_PV, accum);
                }
                umma_commit_2sm_fn(mbar_Q); 
            }
            mbarrier_wait_fn(mbar_Q, 0);
        }
        
        int next_k = cur_k ^ 1;
        int next_v = cur_v ^ 1;
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&mbar_K_local[next_k], 16384 * 2);
            tma_load_2d_fn(&tma_K, &mbar_K_local[next_k], (next_k == 0 ? s_K0 : s_K1), 0, c1_Q);
            tma_load_2d_fn(&tma_K, &mbar_K_local[next_k], (next_k == 0 ? s_K1 : s_K0), 64, c1_Q);
            
            mbarrier_arrive_and_expect_tx_fn(&mbar_V_local[next_v], 16384 * 2);
            tma_load_2d_fn(&tma_V, &mbar_V_local[next_v], (next_v == 0 ? s_V0 : s_V1), 0, c1_Q);
            tma_load_2d_fn(&tma_V, &mbar_V_local[next_v], (next_v == 0 ? s_V1 : s_V0), 64, c1_Q);
        }
        
        cur_k = next_k;
        cur_v = next_v;
        phase_K ^= 1;
    }
    
    uint32_t myRow_tmem = (myRow < 64) ? myRow : myRow - 64;
    uint32_t myRow_tmem_offset = (myRow < 64) ? 0 : 8192;
    char* O_blk = O + row_base * D;
    
    for (int col_idx = 0; col_idx < 8; col_idx++) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn(col_idx * 8 + (threadIdx.x % 2) * 64, &r0, &r1, &r2, &r3);
        float f0 = __uint_as_float(r0);
        float f1 = __uint_as_float(r1);
        float f2 = __uint_as_float(r2);
        float f3 = __uint_as_float(r3);
        
        int c_base = col_idx * 8 + (threadIdx.x % 2) * 64;
        int c0 = (c_base >> 3) ^ (myRow_tmem & 7);
        int c1 = (c_base & 7);
        int col = (c0 << 3) + c1;
        
        uint32_t tmem_col = col_idx * 8 + (threadIdx.x % 2) * 64;
        uint32_t tmem_byte_offset = (tmem_col << 2) + myRow_tmem_offset;
        
        int row_idx = blockRowIdx * S + myRow_tmem;
        
        if (row_idx < (int)(B*H*S)) {
            O[row_idx * D + col] = __float2bfloat16(f0 / row_global_sum[0]);
            O[row_idx * D + col + 1] = __float2bfloat16(f1 / row_global_sum[0]);
            O[row_idx * D + col + 2] = __float2bfloat16(f2 / row_global_sum[0]);
            O[row_idx * D + col + 3] = __float2bfloat16(f3 / row_global_sum[0]);
        }
    }
    
    if (threadIdx.x < 64) {
        uint32_t myRow = threadIdx.x;
        int row_idx = blockRowIdx * S + myRow;
        
        if (row_idx < (int)(B*H*S)) {
            float lse = row_global_max[threadIdx.x] + row_global_log2sum[threadIdx.x] * LN2;
            LSE[row_idx] = lse;
        }
    }
    
    if (threadIdx.x < 32) {
        tmem_dealloc_fn(*tmem_S, 64);
        tmem_dealloc_fn(*tmem_O, 128);
    }
}

namespace tvm_ffi {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    uint64_t B = Q.size(0);
    uint64_t H = Q.size(1);
    uint64_t S_val = Q.size(2);
    uint64_t D = Q.size(3);

    void* Q_ptr = Q.data_ptr();
    void* K_ptr = K.data_ptr();
    void* V_ptr = V.data_ptr();
    __nv_bfloat16* O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_ptr = static_cast<float*>(LSE.data_ptr());

    CUtensorMap tma_Q, tma_K, tma_V;

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

    create_tma_2d_descriptor_2B(&tma_Q, Q_ptr, 128, B * H * S_val, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_K, K_ptr, 128, B * H * S_val, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_V, V_ptr, 128, B * H * S_val, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    dim3 grid((S_val + 63) / 64, 1, 1);
    dim3 block(128, 1, 1);

    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 92 * 1024;
    config.stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    CUDA_CHECK(cudaLaunchKernelEx(&config, attention_forward_cluster, tma_Q, tma_K, tma_V, O_ptr, LSE_ptr, S_val, D));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(config.stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi