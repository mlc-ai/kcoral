#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <math.h>
#include <stdio.h>
#include <algorithm>
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
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
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

__device__ __forceinline__ void umma_commit_1sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1"
        ".mbarrier::arrive::one.shared::cta.b64"
        " [%0];"
        :: "r"(a));
}

__device__ __forceinline__ uint64_t make_smem_desc_k_major(void* smem_ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((1024 & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((128 & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)0 << 61;   // SWIZZLE_NONE
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_mn_major(void* smem_ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((128 & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((1024 & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)0 << 61;   // SWIZZLE_NONE
    return d;
}

__device__ __forceinline__ uint64_t offset_desc_k_major(uint64_t desc) {
    uint32_t addr_field = (desc & 0x3FFF);
    addr_field += 2; // 32 bytes offset -> scale of 16 elements
    desc &= ~0x3FFF;
    desc |= addr_field;
    return desc;
}

__device__ __forceinline__ uint64_t offset_desc_mn_major(uint64_t desc) {
    uint32_t addr_field = (desc & 0x3FFF);
    addr_field += 128; // 2048 bytes offset -> scale of 16 elements (Row stride scale mapped linearly over dummy elements)
    desc &= ~0x3FFF;
    desc |= addr_field;
    return desc;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, bool transpose_a, bool transpose_b) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= ((transpose_a ? 1 : 0) << 15);   
    d |= ((transpose_b ? 1 : 0) << 16);   
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

extern __shared__ char smem_buf[];
char* smem_Q_0 = (char*)(smem_buf + 0 * 8192);
char* smem_Q_1 = (char*)(smem_buf + 1 * 8192);
char* smem_dO_0 = (char*)(smem_buf + 2 * 8192);
char* smem_dO_1 = (char*)(smem_buf + 3 * 8192);
char* smem_O_0 = (char*)(smem_buf + 4 * 8192);
char* smem_O_1 = (char*)(smem_buf + 5 * 8192);
char* smem_K_0 = (char*)(smem_buf + 6 * 8192);
char* smem_K_1 = (char*)(smem_buf + 7 * 8192);
char* smem_V_0 = (char*)(smem_buf + 8 * 8192);
char* smem_V_1 = (char*)(smem_buf + 9 * 8192);
char* smem_dS = (char*)(smem_buf + 10 * 8192);
char* smem_P = (char*)(smem_buf + 11 * 8192);

__align__(16) extern __shared__ uint64_t bar_A_raw[];
__align__(16) extern __shared__ uint64_t bar_B_raw[];
extern __shared__ float head_dOO_shared[];


__global__ void bwd_dQ_kernel(
    const __grid_constant__ CUtensorMap* tma_Q, const __grid_constant__ CUtensorMap* tma_K,
    const __grid_constant__ CUtensorMap* tma_V, const __grid_constant__ CUtensorMap* tma_dO,
    const __grid_constant__ CUtensorMap* tma_O,
    const float* L_gmem,
    __nv_bfloat16* dQ_gmem,
    int64_t S, int64_t BHS, float scale) 
{
    setmaxnreg_inc_sync_fn<256>();
    int q_blk = blockIdx.x;
    int bh = blockIdx.y;
    int tid = threadIdx.x;
    
    uint64_t* bar_A = bar_A_raw;
    uint64_t* bar_B = bar_B_raw;

    if (tid == 0) {
        init_smem_barrier_fn(bar_A, 1);
        init_smem_barrier_fn(bar_B, 1);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t phase_A = 0;
    uint32_t phase_B = 0;
    
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_B, sizeof(__nv_bfloat16) * 64 * 64 * 6);
        tma_load_2d_fn(&tma_Q[bh], bar_B, smem_Q_0, 0, bh * S + q_blk * 64);
        tma_load_2d_fn(&tma_Q[bh], bar_B, smem_Q_1, 64, bh * S + q_blk * 64);
        tma_load_2d_fn(&tma_dO[bh], bar_B, smem_dO_0, 0, bh * S + q_blk * 64);
        tma_load_2d_fn(&tma_dO[bh], bar_B, smem_dO_1, 64, bh * S + q_blk * 64);
        tma_load_2d_fn(&tma_O[bh], bar_B, smem_O_0, 0, bh * S + q_blk * 64);
        tma_load_2d_fn(&tma_O[bh], bar_B, smem_O_1, 64, bh * S + q_blk * 64);
    }
    mbarrier_wait_fn(bar_B, phase_B);
    phase_B ^= 1;
    __syncthreads();

    if (tid < 64) {
        float sum = 0;
        for(int c = 0; c < 64; c++) {
            float o_0 = __bfloat162float((( __nv_bfloat16*)smem_O_0)[tid * 64 + c]);
            float do_0 = __bfloat162float((( __nv_bfloat16*)smem_dO_0)[tid * 64 + c]);
            sum += o_0 * do_0;
            
            float o_1 = __bfloat162float((( __nv_bfloat16*)smem_O_1)[tid * 64 + c]);
            float do_1 = __bfloat162float((( __nv_bfloat16*)smem_dO_1)[tid * 64 + c]);
            sum += o_1 * do_1;
        }
        head_dOO_shared[tid] = sum;
    }
    __syncthreads();

    uint32_t tmem_dQ_0, tmem_dQ_1, tmem_S, tmem_dP;
    if (tid == 0) {
        tmem_alloc_fn(&tmem_dQ_0, 64);
        tmem_alloc_fn(&tmem_dQ_1, 64);
        tmem_alloc_fn(&tmem_S, 64);
        tmem_alloc_fn(&tmem_dP, 64);
    }
    __syncthreads();

    int num_S_blocks = (S + 63) / 64;

    for (int k_blk = 0; k_blk <= q_blk && k_blk < num_S_blocks; k_blk++) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_B, sizeof(__nv_bfloat16) * 64 * 64 * 4);
            tma_load_2d_fn(&tma_K[bh], bar_B, smem_K_0, 0, bh * S + k_blk * 64);
            tma_load_2d_fn(&tma_K[bh], bar_B, smem_K_1, 64, bh * S + k_blk * 64);
            tma_load_2d_fn(&tma_V[bh], bar_B, smem_V_0, 0, bh * S + k_blk * 64);
            tma_load_2d_fn(&tma_V[bh], bar_B, smem_V_1, 64, bh * S + k_blk * 64);
        }
        mbarrier_wait_fn(bar_B, phase_B);
        phase_B ^= 1;
        __syncthreads();

        if (tid < 128) {
            uint64_t desc_Q_0 = make_smem_desc_k_major(smem_Q_0);
            uint64_t desc_K_0 = make_smem_desc_mn_major(smem_K_0);
            uint64_t desc_Q_1 = make_smem_desc_k_major(smem_Q_1);
            uint64_t desc_K_1 = make_smem_desc_mn_major(smem_K_1);
            
            uint32_t idesc_S_QK = make_instr_desc_fn(64, 64, false, true);
            
            for (int k = 0; k < 4; k++) {
                uint64_t dQ0 = desc_Q_0;
                uint64_t dK0 = desc_K_0;
                uint64_t dQ1 = desc_Q_1;
                uint64_t dK1 = desc_K_1;
                
                for (int i = 0; i < k; i++) {
                    dQ0 = offset_desc_k_major(dQ0);
                    dK0 = offset_desc_mn_major(dK0);
                    dQ1 = offset_desc_k_major(dQ1);
                    dK1 = offset_desc_mn_major(dK1);
                }
                
                if (k == 0) {
                    umma_f16_cg1_fn(tmem_S, dQ0, dK0, idesc_S_QK, 0);
                    umma_f16_cg1_fn(tmem_S, dQ1, dK1, idesc_S_QK, 1);
                } else {
                    umma_f16_cg1_fn(tmem_S, dQ0, dK0, idesc_S_QK, 1);
                    umma_f16_cg1_fn(tmem_S, dQ1, dK1, idesc_S_QK, 1);
                }
            }
        }
        
        if (tid < 128) {
            uint64_t desc_dO_0 = make_smem_desc_k_major(smem_dO_0);
            uint64_t desc_V_0 = make_smem_desc_mn_major(smem_V_0);
            uint64_t desc_dO_1 = make_smem_desc_k_major(smem_dO_1);
            uint64_t desc_V_1 = make_smem_desc_mn_major(smem_V_1);
            
            uint32_t idesc_dP_OV = make_instr_desc_fn(64, 64, false, true);
            
            for (int k = 0; k < 4; k++) {
                uint64_t ddO0 = desc_dO_0;
                uint64_t dV0 = desc_V_0;
                uint64_t ddO1 = desc_dO_1;
                uint64_t dV1 = desc_V_1;
                
                for (int i = 0; i < k; i++) {
                    ddO0 = offset_desc_k_major(ddO0);
                    dV0 = offset_desc_mn_major(dV0);
                    ddO1 = offset_desc_k_major(ddO1);
                    dV1 = offset_desc_mn_major(dV1);
                }
                
                if (k == 0) {
                    umma_f16_cg1_fn(tmem_dP, ddO0, dV0, idesc_dP_OV, 0);
                    umma_f16_cg1_fn(tmem_dP, ddO1, dV1, idesc_dP_OV, 1);
                } else {
                    umma_f16_cg1_fn(tmem_dP, ddO0, dV0, idesc_dP_OV, 1);
                    umma_f16_cg1_fn(tmem_dP, ddO1, dV1, idesc_dP_OV, 1);
                }
            }
        }
        umma_commit_1sm_fn(bar_A);
        mbarrier_wait_fn(bar_A, phase_A);
        phase_A ^= 1;
        __syncthreads();

        for(int tid_col = tid; tid_col < 4096; tid_col += blockDim.x) { 
            int r = tid_col / 64;
            int c = tid_col % 64;
            
            uint32_t val_S, val_dP;
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(val_S) : "r"(tmem_S + (r << 16) + c));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(val_dP) : "r"(tmem_dP + (r << 16) + c));
            
            float s_val = __uint_as_float(val_S);
            float dp_val = __uint_as_float(val_dP);
            
            float p_val = expf(s_val * scale - L_gmem[bh * S + q_blk * 64 + r]);
            if (p_val > 1.0f || p_val < 0.0f) p_val = 0.0f;
            if ((q_blk * 64 + r) < (k_blk * 64 + c) || q_blk * 64 + r >= S || k_blk * 64 + c >= S) {
                p_val = 0.0f;
            }
            
            float ds_val = p_val * (dp_val - head_dOO_shared[r]) * scale;
            
            (( __nv_bfloat16*)smem_dS)[r * 64 + c] = __float2bfloat16(ds_val);
        }
        __syncthreads();
        fence_proxy_async_fn();

        if (tid < 128) {
            uint64_t desc_dS = make_smem_desc_k_major(smem_dS);
            uint64_t desc_K_0_mnmaj = make_smem_desc_mn_major(smem_K_0);
            uint64_t desc_K_1_mnmaj = make_smem_desc_mn_major(smem_K_1);
            
            uint32_t idesc_dQ_dS_K = make_instr_desc_fn(64, 64, false, true);
            
            for (int k = 0; k < 4; k++) {
                uint64_t ddS = desc_dS;
                uint64_t dK0 = desc_K_0_mnmaj;
                uint64_t dK1 = desc_K_1_mnmaj;
                
                for (int i = 0; i < k; i++) {
                    ddS = offset_desc_k_major(ddS);
                    dK0 = offset_desc_mn_major(dK0);
                    dK1 = offset_desc_mn_major(dK1);
                }
                
                if (k_blk == 0 && k == 0) {
                    umma_f16_cg1_fn(tmem_dQ_0, ddS, dK0, idesc_dQ_dS_K, 0);
                    umma_f16_cg1_fn(tmem_dQ_1, ddS, dK1, idesc_dQ_dS_K, 0);
                } else {
                    umma_f16_cg1_fn(tmem_dQ_0, ddS, dK0, idesc_dQ_dS_K, 1);
                    umma_f16_cg1_fn(tmem_dQ_1, ddS, dK1, idesc_dQ_dS_K, 1);
                }
            }
        }
        umma_commit_1sm_fn(bar_A);
        mbarrier_wait_fn(bar_A, phase_A);
        phase_A ^= 1;
        __syncthreads();
    }

    for (int tid_col = tid; tid_col < 4096; tid_col += blockDim.x) {
        int r = tid_col / 64;
        int c = tid_col % 64;
        
        uint32_t val_0, val_1;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(val_0) : "r"(tmem_dQ_0 + (r << 16) + c));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(val_1) : "r"(tmem_dQ_1 + (r << 16) + c));
        
        float f0 = __uint_as_float(val_0);
        float f1 = __uint_as_float(val_1);
        
        uint32_t g_row = bh * S + q_blk * 64 + r;
        uint32_t g_col_0 = c;
        uint32_t g_col_1 = c + 64;
        
        if (g_row < BHS && g_col_0 < 128) {
            dQ_gmem[g_row * 128 + g_col_0] = __float2bfloat16(f0);
        }
        if (g_row < BHS && g_col_1 < 128) {
            dQ_gmem[g_row * 128 + g_col_1] = __float2bfloat16(f1);
        }
    }

    if (tid == 0) {
        tmem_dealloc_fn(tmem_dQ_0, 64);
        tmem_dealloc_fn(tmem_dQ_1, 64);
        tmem_dealloc_fn(tmem_S, 64);
        tmem_dealloc_fn(tmem_dP, 64);
    }
}

__global__ void bwd_dK_kernel(
    const __grid_constant__ CUtensorMap* tma_Q, const __grid_constant__ CUtensorMap* tma_K,
    const __grid_constant__ CUtensorMap* tma_V, const __grid_constant__ CUtensorMap* tma_dO,
    const __grid_constant__ CUtensorMap* tma_O,
    const float* L_gmem,
    __nv_bfloat16* dK_gmem,
    int64_t S, int64_t BHS, float scale) 
{
    setmaxnreg_inc_sync_fn<256>();
    int k_blk = blockIdx.x;
    int bh = blockIdx.y;
    int tid = threadIdx.x;

    uint64_t* bar_A = bar_A_raw;
    uint64_t* bar_B = bar_B_raw;

    if (tid == 0) {
        init_smem_barrier_fn(bar_A, 1);
        init_smem_barrier_fn(bar_B, 1);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t phase_A = 0;
    uint32_t phase_B = 0;
    
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_B, sizeof(__nv_bfloat16) * 64 * 64 * 4);
        tma_load_2d_fn(&tma_K[bh], bar_B, smem_K_0, 0, bh * S + k_blk * 64);
        tma_load_2d_fn(&tma_K[bh], bar_B, smem_K_1, 64, bh * S + k_blk * 64);
        tma_load_2d_fn(&tma_V[bh], bar_B, smem_V_0, 0, bh * S + k_blk * 64);
        tma_load_2d_fn(&tma_V[bh], bar_B, smem_V_1, 64, bh * S + k_blk * 64);
    }
    mbarrier_wait_fn(bar_B, phase_B);
    phase_B ^= 1;
    __syncthreads();

    uint32_t tmem_dK_0, tmem_dK_1, tmem_S, tmem_dP;
    if (tid == 0) {
        tmem_alloc_fn(&tmem_dK_0, 64);
        tmem_alloc_fn(&tmem_dK_1, 64);
        tmem_alloc_fn(&tmem_S, 64);
        tmem_alloc_fn(&tmem_dP, 64);
    }
    __syncthreads();

    int num_S_blocks = (S + 63) / 64;

    for (int q_blk = k_blk; q_blk < num_S_blocks; q_blk++) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_B, sizeof(__nv_bfloat16) * 64 * 64 * 6);
            tma_load_2d_fn(&tma_Q[bh], bar_B, smem_Q_0, 0, bh * S + q_blk * 64);
            tma_load_2d_fn(&tma_Q[bh], bar_B, smem_Q_1, 64, bh * S + q_blk * 64);
            tma_load_2d_fn(&tma_dO[bh], bar_B, smem_dO_0, 0, bh * S + q_blk * 64);
            tma_load_2d_fn(&tma_dO[bh], bar_B, smem_dO_1, 64, bh * S + q_blk * 64);
            tma_load_2d_fn(&tma_O[bh], bar_B, smem_O_0, 0, bh * S + q_blk * 64);
            tma_load_2d_fn(&tma_O[bh], bar_B, smem_O_1, 64, bh * S + q_blk * 64);
        }
        mbarrier_wait_fn(bar_B, phase_B);
        phase_B ^= 1;
        __syncthreads();

        if (tid < 64) {
            float sum = 0;
            for(int c = 0; c < 64; c++) {
                float o_0 = __bfloat162float((( __nv_bfloat16*)smem_O_0)[tid * 64 + c]);
                float do_0 = __bfloat162float((( __nv_bfloat16*)smem_dO_0)[tid * 64 + c]);
                sum += o_0 * do_0;
                
                float o_1 = __bfloat162float((( __nv_bfloat16*)smem_O_1)[tid * 64 + c]);
                float do_1 = __bfloat162float((( __nv_bfloat16*)smem_dO_1)[tid * 64 + c]);
                sum += o_1 * do_1;
            }
            head_dOO_shared[tid] = sum;
        }
        __syncthreads();

        if (tid < 128) {
            uint64_t desc_Q_0 = make_smem_desc_k_major(smem_Q_0);
            uint64_t desc_K_0 = make_smem_desc_mn_major(smem_K_0);
            uint64_t desc_Q_1 = make_smem_desc_k_major(smem_Q_1);
            uint64_t desc_K_1 = make_smem_desc_mn_major(smem_K_1);
            
            uint32_t idesc_S_QK = make_instr_desc_fn(64, 64, false, true);
            
            for (int k = 0; k < 4; k++) {
                uint64_t dQ0 = desc_Q_0;
                uint64_t dK0 = desc_K_0;
                uint64_t dQ1 = desc_Q_1;
                uint64_t dK1 = desc_K_1;
                
                for (int i = 0; i < k; i++) {
                    dQ0 = offset_desc_k_major(dQ0);
                    dK0 = offset_desc_mn_major(dK0);
                    dQ1 = offset_desc_k_major(dQ1);
                    dK1 = offset_desc_mn_major(dK1);
                }
                
                if (k == 0) {
                    umma_f16_cg1_fn(tmem_S, dQ0, dK0, idesc_S_QK, 0);
                    umma_f16_cg1_fn(tmem_S, dQ1, dK1, idesc_S_QK, 1);
                } else {
                    umma_f16_cg1_fn(tmem_S, dQ0, dK0, idesc_S_QK, 1);
                    umma_f16_cg1_fn(tmem_S, dQ1, dK1, idesc_S_QK, 1);
                }
            }
        }
        
        if (tid < 128) {
            uint64_t desc_dO_0 = make_smem_desc_k_major(smem_dO_0);
            uint64_t desc_V_0 = make_smem_desc_mn_major(smem_V_0);
            uint64_t desc_dO_1 = make_smem_desc_k_major(smem_dO_1);
            uint64_t desc_V_1 = make_smem_desc_mn_major(smem_V_1);
            
            uint32_t idesc_dP_OV = make_instr_desc_fn(64, 64, false, true);
            
            for (int k = 0; k < 4; k++) {
                uint64_t ddO0 = desc_dO_0;
                uint64_t dV0 = desc_V_0;
                uint64_t ddO1 = desc_dO_1;
                uint64_t dV1 = desc_V_1;
                
                for (int i = 0; i < k; i++) {
                    ddO0 = offset_desc_k_major(ddO0);
                    dV0 = offset_desc_mn_major(dV0);
                    ddO1 = offset_desc_k_major(ddO1);
                    dV1 = offset_desc_mn_major(dV1);
                }
                
                if (k == 0) {
                    umma_f16_cg1_fn(tmem_dP, ddO0, dV0, idesc_dP_OV, 0);
                    umma_f16_cg1_fn(tmem_dP, ddO1, dV1, idesc_dP_OV, 1);
                } else {
                    umma_f16_cg1_fn(tmem_dP, ddO0, dV0, idesc_dP_OV, 1);
                    umma_f16_cg1_fn(tmem_dP, ddO1, dV1, idesc_dP_OV, 1);
                }
            }
        }
        umma_commit_1sm_fn(bar_A);
        mbarrier_wait_fn(bar_A, phase_A);
        phase_A ^= 1;
        __syncthreads();

        for(int tid_col = tid; tid_col < 4096; tid_col += blockDim.x) { 
            int r = tid_col / 64;
            int c = tid_col % 64;
            
            uint32_t val_S, val_dP;
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(val_S) : "r"(tmem_S + (r << 16) + c));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(val_dP) : "r"(tmem_dP + (r << 16) + c));
            
            float s_val = __uint_as_float(val_S);
            float dp_val = __uint_as_float(val_dP);
            
            float p_val = expf(s_val * scale - L_gmem[bh * S + q_blk * 64 + r]);
            if (p_val > 1.0f || p_val < 0.0f) p_val = 0.0f;
            if ((q_blk * 64 + r) < (k_blk * 64 + c) || q_blk * 64 + r >= S || k_blk * 64 + c >= S) {
                p_val = 0.0f;
            }
            
            float ds_val = p_val * (dp_val - head_dOO_shared[r]) * scale;
            
            (( __nv_bfloat16*)smem_P)[r * 64 + c] = __float2bfloat16(p_val);
            (( __nv_bfloat16*)smem_dS)[r * 64 + c] = __float2bfloat16(ds_val);
        }
        __syncthreads();
        fence_proxy_async_fn();

        if (tid < 128) {
            uint64_t desc_dS_mnmaj = make_smem_desc_mn_major(smem_dS);
            uint64_t desc_Q_0_mnmaj = make_smem_desc_mn_major(smem_Q_0);
            uint64_t desc_Q_1_mnmaj = make_smem_desc_mn_major(smem_Q_1);
            
            uint32_t idesc_dK_dS_Q = make_instr_desc_fn(64, 64, true, true);
            
            for (int k = 0; k < 4; k++) {
                uint64_t ddS = desc_dS_mnmaj;
                uint64_t dQ0 = desc_Q_0_mnmaj;
                uint64_t dQ1 = desc_Q_1_mnmaj;
                
                for (int i = 0; i < k; i++) {
                    ddS = offset_desc_mn_major(ddS);
                    dQ0 = offset_desc_mn_major(dQ0);
                    dQ1 = offset_desc_mn_major(dQ1);
                }
                
                if (q_blk == k_blk && k == 0) {
                    umma_f16_cg1_fn(tmem_dK_0, ddS, dQ0, idesc_dK_dS_Q, 0);
                    umma_f16_cg1_fn(tmem_dK_1, ddS, dQ1, idesc_dK_dS_Q, 0);
                } else {
                    umma_f16_cg1_fn(tmem_dK_0, ddS, dQ0, idesc_dK_dS_Q, 1);
                    umma_f16_cg1_fn(tmem_dK_1, ddS, dQ1, idesc_dK_dS_Q, 1);
                }
            }
        }
        umma_commit_1sm_fn(bar_A);
        mbarrier_wait_fn(bar_A, phase_A);
        phase_A ^= 1;
        __syncthreads();
    }

    for (int tid_col = tid; tid_col < 4096; tid_col += blockDim.x) {
        int r = tid_col / 64;
        int c = tid_col % 64;
        
        uint32_t val_0, val_1;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(val_0) : "r"(tmem_dK_0 + (r << 16) + c));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(val_1) : "r"(tmem_dK_1 + (r << 16) + c));
        
        float f0 = __uint_as_float(val_0);
        float f1 = __uint_as_float(val_1);
        
        uint32_t g_row = bh * S + k_blk * 64 + r;
        uint32_t g_col_0 = c;
        uint32_t g_col_1 = c + 64;
        
        if (g_row < BHS && g_col_0 < 128) {
            dK_gmem[g_row * 128 + g_col_0] = __float2bfloat16(f0);
        }
        if (g_row < BHS && g_col_1 < 128) {
            dK_gmem[g_row * 128 + g_col_1] = __float2bfloat16(f1);
        }
    }

    if (tid == 0) {
        tmem_dealloc_fn(tmem_dK_0, 64);
        tmem_dealloc_fn(tmem_dK_1, 64);
        tmem_dealloc_fn(tmem_S, 64);
        tmem_dealloc_fn(tmem_dP, 64);
    }
}

__global__ void bwd_dV_kernel(
    const __grid_constant__ CUtensorMap* tma_Q, const __grid_constant__ CUtensorMap* tma_K,
    const __grid_constant__ CUtensorMap* tma_V, const __grid_constant__ CUtensorMap* tma_dO,
    const __grid_constant__ CUtensorMap* tma_O,
    const float* L_gmem,
    __nv_bfloat16* dV_gmem,
    int64_t S, int64_t BHS, float scale) 
{
    setmaxnreg_inc_sync_fn<256>();
    int k_blk = blockIdx.x;
    int bh = blockIdx.y;
    int tid = threadIdx.x;

    uint64_t* bar_A = bar_A_raw;
    uint64_t* bar_B = bar_B_raw;

    if (tid == 0) {
        init_smem_barrier_fn(bar_A, 1);
        init_smem_barrier_fn(bar_B, 1);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t phase_A = 0;
    uint32_t phase_B = 0;
    
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_B, sizeof(__nv_bfloat16) * 64 * 64 * 4);
        tma_load_2d_fn(&tma_K[bh], bar_B, smem_K_0, 0, bh * S + k_blk * 64);
        tma_load_2d_fn(&tma_K[bh], bar_B, smem_K_1, 64, bh * S + k_blk * 64);
        tma_load_2d_fn(&tma_V[bh], bar_B, smem_V_0, 0, bh * S + k_blk * 64);
        tma_load_2d_fn(&tma_V[bh], bar_B, smem_V_1, 64, bh * S + k_blk * 64);
    }
    mbarrier_wait_fn(bar_B, phase_B);
    phase_B ^= 1;
    __syncthreads();

    uint32_t tmem_dV_0, tmem_dV_1, tmem_S, tmem_dP;
    if (tid == 0) {
        tmem_alloc_fn(&tmem_dV_0, 64);
        tmem_alloc_fn(&tmem_dV_1, 64);
        tmem_alloc_fn(&tmem_S, 64);
        tmem_alloc_fn(&tmem_dP, 64);
    }
    __syncthreads();

    int num_S_blocks = (S + 63) / 64;

    for (int q_blk = k_blk; q_blk < num_S_blocks; q_blk++) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_B, sizeof(__nv_bfloat16) * 64 * 64 * 6);
            tma_load_2d_fn(&tma_Q[bh], bar_B, smem_Q_0, 0, bh * S + q_blk * 64);
            tma_load_2d_fn(&tma_Q[bh], bar_B, smem_Q_1, 64, bh * S + q_blk * 64);
            tma_load_2d_fn(&tma_dO[bh], bar_B, smem_dO_0, 0, bh * S + q_blk * 64);
            tma_load_2d_fn(&tma_dO[bh], bar_B, smem_dO_1, 64, bh * S + q_blk * 64);
            tma_load_2d_fn(&tma_O[bh], bar_B, smem_O_0, 0, bh * S + q_blk * 64);
            tma_load_2d_fn(&tma_O[bh], bar_B, smem_O_1, 64, bh * S + q_blk * 64);
        }
        mbarrier_wait_fn(bar_B, phase_B);
        phase_B ^= 1;
        __syncthreads();

        if (tid < 64) {
            float sum = 0;
            for(int c = 0; c < 64; c++) {
                float o_0 = __bfloat162float((( __nv_bfloat16*)smem_O_0)[tid * 64 + c]);
                float do_0 = __bfloat162float((( __nv_bfloat16*)smem_dO_0)[tid * 64 + c]);
                sum += o_0 * do_0;
                
                float o_1 = __bfloat162float((( __nv_bfloat16*)smem_O_1)[tid * 64 + c]);
                float do_1 = __bfloat162float((( __nv_bfloat16*)smem_dO_1)[tid * 64 + c]);
                sum += o_1 * do_1;
            }
            head_dOO_shared[tid] = sum;
        }
        __syncthreads();

        if (tid < 128) {
            uint64_t desc_Q_0 = make_smem_desc_k_major(smem_Q_0);
            uint64_t desc_K_0 = make_smem_desc_mn_major(smem_K_0);
            uint64_t desc_Q_1 = make_smem_desc_k_major(smem_Q_1);
            uint64_t desc_K_1 = make_smem_desc_mn_major(smem_K_1);
            
            uint32_t idesc_S_QK = make_instr_desc_fn(64, 64, false, true);
            
            for (int k = 0; k < 4; k++) {
                uint64_t dQ0 = desc_Q_0;
                uint64_t dK0 = desc_K_0;
                uint64_t dQ1 = desc_Q_1;
                uint64_t dK1 = desc_K_1;
                
                for (int i = 0; i < k; i++) {
                    dQ0 = offset_desc_k_major(dQ0);
                    dK0 = offset_desc_mn_major(dK0);
                    dQ1 = offset_desc_k_major(dQ1);
                    dK1 = offset_desc_mn_major(dK1);
                }
                
                if (k == 0) {
                    umma_f16_cg1_fn(tmem_S, dQ0, dK0, idesc_S_QK, 0);
                    umma_f16_cg1_fn(tmem_S, dQ1, dK1, idesc_S_QK, 1);
                } else {
                    umma_f16_cg1_fn(tmem_S, dQ0, dK0, idesc_S_QK, 1);
                    umma_f16_cg1_fn(tmem_S, dQ1, dK1, idesc_S_QK, 1);
                }
            }
        }
        
        if (tid < 128) {
            uint64_t desc_dO_0 = make_smem_desc_k_major(smem_dO_0);
            uint64_t desc_V_0 = make_smem_desc_mn_major(smem_V_0);
            uint64_t desc_dO_1 = make_smem_desc_k_major(smem_dO_1);
            uint64_t desc_V_1 = make_smem_desc_mn_major(smem_V_1);
            
            uint32_t idesc_dP_OV = make_instr_desc_fn(64, 64, false, true);
            
            for (int k = 0; k < 4; k++) {
                uint64_t ddO0 = desc_dO_0;
                uint64_t dV0 = desc_V_0;
                uint64_t ddO1 = desc_dO_1;
                uint64_t dV1 = desc_V_1;
                
                for (int i = 0; i < k; i++) {
                    ddO0 = offset_desc_k_major(ddO0);
                    dV0 = offset_desc_mn_major(dV0);
                    ddO1 = offset_desc_k_major(ddO1);
                    dV1 = offset_desc_mn_major(dV1);
                }
                
                if (k == 0) {
                    umma_f16_cg1_fn(tmem_dP, ddO0, dV0, idesc_dP_OV, 0);
                    umma_f16_cg1_fn(tmem_dP, ddO1, dV1, idesc_dP_OV, 1);
                } else {
                    umma_f16_cg1_fn(tmem_dP, ddO0, dV0, idesc_dP_OV, 1);
                    umma_f16_cg1_fn(tmem_dP, ddO1, dV1, idesc_dP_OV, 1);
                }
            }
        }
        umma_commit_1sm_fn(bar_A);
        mbarrier_wait_fn(bar_A, phase_A);
        phase_A ^= 1;
        __syncthreads();

        for(int tid_col = tid; tid_col < 4096; tid_col += blockDim.x) { 
            int r = tid_col / 64;
            int c = tid_col % 64;
            
            uint32_t val_S, val_dP;
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(val_S) : "r"(tmem_S + (r << 16) + c));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(val_dP) : "r"(tmem_dP + (r << 16) + c));
            
            float s_val = __uint_as_float(val_S);
            float dp_val = __uint_as_float(val_dP);
            
            float p_val = expf(s_val * scale - L_gmem[bh * S + q_blk * 64 + r]);
            if (p_val > 1.0f || p_val < 0.0f) p_val = 0.0f;
            if ((q_blk * 64 + r) < (k_blk * 64 + c) || q_blk * 64 + r >= S || k_blk * 64 + c >= S) {
                p_val = 0.0f;
            }
            
            float ds_val = p_val * (dp_val - head_dOO_shared[r]) * scale;
            
            (( __nv_bfloat16*)smem_P)[r * 64 + c] = __float2bfloat16(p_val);
            (( __nv_bfloat16*)smem_dS)[r * 64 + c] = __float2bfloat16(ds_val);
        }
        __syncthreads();
        fence_proxy_async_fn();

        if (tid < 128) {
            uint64_t desc_P_mnmaj = make_smem_desc_mn_major(smem_P);
            uint64_t desc_dO_0_mnmaj = make_smem_desc_mn_major(smem_dO_0);
            uint64_t desc_dO_1_mnmaj = make_smem_desc_mn_major(smem_dO_1);
            
            uint32_t idesc_dV_P_dO = make_instr_desc_fn(64, 64, true, true);
            
            for (int k = 0; k < 4; k++) {
                uint64_t dP = desc_P_mnmaj;
                uint64_t ddO0 = desc_dO_0_mnmaj;
                uint64_t ddO1 = desc_dO_1_mnmaj;
                
                for (int i = 0; i < k; i++) {
                    dP = offset_desc_mn_major(dP);
                    ddO0 = offset_desc_mn_major(ddO0);
                    ddO1 = offset_desc_mn_major(ddO1);
                }
                
                if (q_blk == k_blk && k == 0) {
                    umma_f16_cg1_fn(tmem_dV_0, dP, ddO0, idesc_dV_P_dO, 0);
                    umma_f16_cg1_fn(tmem_dV_1, dP, ddO1, idesc_dV_P_dO, 0);
                } else {
                    umma_f16_cg1_fn(tmem_dV_0, dP, ddO0, idesc_dV_P_dO, 1);
                    umma_f16_cg1_fn(tmem_dV_1, dP, ddO1, idesc_dV_P_dO, 1);
                }
            }
        }
        umma_commit_1sm_fn(bar_A);
        mbarrier_wait_fn(bar_A, phase_A);
        phase_A ^= 1;
        __syncthreads();
    }

    for (int tid_col = tid; tid_col < 4096; tid_col += blockDim.x) {
        int r = tid_col / 64;
        int c = tid_col % 64;
        
        uint32_t val_0, val_1;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(val_0) : "r"(tmem_dV_0 + (r << 16) + c));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(val_1) : "r"(tmem_dV_1 + (r << 16) + c));
        
        float f0 = __uint_as_float(val_0);
        float f1 = __uint_as_float(val_1);
        
        uint32_t g_row = bh * S + k_blk * 64 + r;
        uint32_t g_col_0 = c;
        uint32_t g_col_1 = c + 64;
        
        if (g_row < BHS && g_col_0 < 128) {
            dV_gmem[g_row * 128 + g_col_0] = __float2bfloat16(f0);
        }
        if (g_row < BHS && g_col_1 < 128) {
            dV_gmem[g_row * 128 + g_col_1] = __float2bfloat16(f1);
        }
    }

    if (tid == 0) {
        tmem_dealloc_fn(tmem_dV_0, 64);
        tmem_dealloc_fn(tmem_dV_1, 64);
        tmem_dealloc_fn(tmem_S, 64);
        tmem_dealloc_fn(tmem_dP, 64);
    }
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

namespace tvm_ffi_example_cuda {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = Q.size(3); 

    float scale = 1.0f / sqrtf((float)d);
    int64_t BHS = B * H * S;

    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_ptr = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_ptr = static_cast<const float*>(L.data_ptr());
    
    __nv_bfloat16* dQ_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());

    CUtensorMap* tma_Q = (CUtensorMap*)malloc(sizeof(CUtensorMap) * B * H);
    CUtensorMap* tma_K = (CUtensorMap*)malloc(sizeof(CUtensorMap) * B * H);
    CUtensorMap* tma_V = (CUtensorMap*)malloc(sizeof(CUtensorMap) * B * H);
    CUtensorMap* tma_O = (CUtensorMap*)malloc(sizeof(CUtensorMap) * B * H);
    CUtensorMap* tma_dO = (CUtensorMap*)malloc(sizeof(CUtensorMap) * B * H);

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    for (int bh = 0; bh < B * H; bh++) {
        create_tma_2d_descriptor_2B(&tma_Q[bh], (void*)(Q_ptr + bh * S * d), d, B * H * S, 64, 64, 
            CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        create_tma_2d_descriptor_2B(&tma_K[bh], (void*)(K_ptr + bh * S * d), d, B * H * S, 64, 64, 
            CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        create_tma_2d_descriptor_2B(&tma_V[bh], (void*)(V_ptr + bh * S * d), d, B * H * S, 64, 64, 
            CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        create_tma_2d_descriptor_2B(&tma_O[bh], (void*)(O_ptr + bh * S * d), d, B * H * S, 64, 64, 
            CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        create_tma_2d_descriptor_2B(&tma_dO[bh], (void*)(dO_ptr + bh * S * d), d, B * H * S, 64, 64, 
            CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    }

    int num_S_blocks = (S + 63) / 64;
    dim3 grid(num_S_blocks, B * H);
    dim3 block(128);
    
    int smem_size = 90112;
    CUDA_CHECK(cudaFuncSetAttribute(bwd_dQ_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    CUDA_CHECK(cudaFuncSetAttribute(bwd_dK_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    CUDA_CHECK(cudaFuncSetAttribute(bwd_dV_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    CUDA_CHECK(cudaLaunchKernel(bwd_dQ_kernel, grid, block, smem_size, stream, tma_Q, tma_K, tma_V, tma_dO, tma_O, L_ptr, dQ_ptr, S, BHS, scale));
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaLaunchKernel(bwd_dK_kernel, grid, block, smem_size, stream, tma_Q, tma_K, tma_V, tma_dO, tma_O, L_ptr, dK_ptr, S, BHS, scale));
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaLaunchKernel(bwd_dV_kernel, grid, block, smem_size, stream, tma_Q, tma_K, tma_V, tma_dO, tma_O, L_ptr, dV_ptr, S, BHS, scale));
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));

    free(tma_Q); free(tma_K); free(tma_V); free(tma_O); free(tma_dO);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda