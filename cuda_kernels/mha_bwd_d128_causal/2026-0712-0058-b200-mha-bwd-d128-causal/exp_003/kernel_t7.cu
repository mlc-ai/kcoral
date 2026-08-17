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

__device__ __forceinline__ void umma_f16_cg1(
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
    d |= (uint64_t)((1 & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((1024 & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_n_major(void* smem_ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((1024 & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((1024 & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_mn_major(void* smem_ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((1024 & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((1024 & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint64_t adv_k_major(uint64_t desc) {
    uint32_t offset_enc = (desc >> 32) & 0x3FFF;
    offset_enc += 2; // scale of 16 elements mapped linearly over chunk boundary
    desc &= ~(0x3FFFULL << 32);
    desc |= ((uint64_t)offset_enc << 32);
    return desc;
}

__device__ __forceinline__ uint64_t adv_n_major(uint64_t desc) {
    uint32_t enc_offset = (desc >> 16) & 0x3FFF;
    enc_offset += 2; // scale of 16 elements mapped linearly over chunk boundary scaling limits
    desc &= ~(0x3FFFULL << 16);
    desc |= ((uint64_t)enc_offset << 16);
    return desc;
}

__device__ __forceinline__ uint32_t make_instr_desc(uint32_t M, uint32_t N, bool transpose_a, bool transpose_b) {
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

__device__ __forceinline__ float read_tmem(uint32_t tmem_base, int row, int col) {
    uint32_t val;
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(val) : "r"(tmem_base + (row << 16) + col));
    return __uint_as_float(val);
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

extern __shared__ char smem_buf[];
char* smem_Q_L = (char*)(smem_buf + 0 * 8192);
char* smem_Q_R = (char*)(smem_buf + 1 * 8192);
char* smem_K_L = (char*)(smem_buf + 2 * 8192);
char* smem_K_R = (char*)(smem_buf + 3 * 8192);
char* smem_V_L = (char*)(smem_buf + 4 * 8192);
char* smem_V_R = (char*)(smem_buf + 5 * 8192);
char* smem_O_L = (char*)(smem_buf + 6 * 8192);
char* smem_O_R = (char*)(smem_buf + 7 * 8192);
char* smem_dO_L = (char*)(smem_buf + 8 * 8192);
char* smem_dO_R = (char*)(smem_buf + 9 * 8192);
char* smem_dS = (char*)(smem_buf + 10 * 8192);
char* smem_P = (char*)(smem_buf + 11 * 8192);

__align__(16) extern __shared__ uint64_t bar_A_raw[];
__align__(16) extern __shared__ uint64_t bar_B_raw[];
extern __shared__ float head_dOO_shared[];


__global__ void bwd_dQ_kernel(
    const __grid_constant__ CUtensorMap tma_Q, const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V, const __grid_constant__ CUtensorMap tma_dO,
    const __grid_constant__ CUtensorMap tma_O,
    const float* L_gmem,
    __nv_bfloat16* dQ_gmem,
    int64_t S, int64_t BHS, float scale) 
{
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
        tma_load_2d_fn(&tma_Q, bar_B, smem_Q_L, 0, bh * S + q_blk * 64);
        tma_load_2d_fn(&tma_Q, bar_B, smem_Q_R, 64, bh * S + q_blk * 64);
        tma_load_2d_fn(&tma_dO, bar_B, smem_dO_L, 0, bh * S + q_blk * 64);
        tma_load_2d_fn(&tma_dO, bar_B, smem_dO_R, 64, bh * S + q_blk * 64);
        tma_load_2d_fn(&tma_O, bar_B, smem_O_L, 0, bh * S + q_blk * 64);
        tma_load_2d_fn(&tma_O, bar_B, smem_O_R, 64, bh * S + q_blk * 64);
    }
    mbarrier_wait_fn(bar_B, phase_B);
    phase_B ^= 1;
    __syncthreads();

    if (tid < 64) {
        float sum = 0;
        for(int c = 0; c < 64; c++) {
            sum += __bfloat162float((( __nv_bfloat16*)smem_O_L)[tid * 64 + c]) * __bfloat162float((( __nv_bfloat16*)smem_dO_L)[tid * 64 + c]);
            sum += __bfloat162float((( __nv_bfloat16*)smem_O_R)[tid * 64 + c]) * __bfloat162float((( __nv_bfloat16*)smem_dO_R)[tid * 64 + c]);
        }
        head_dOO_shared[tid] = sum;
    }
    __syncthreads();

    uint32_t tmem_dQ0, tmem_dQ1, tmem_S, tmem_dP;
    if (tid == 0) {
        tmem_alloc_fn(&tmem_dQ0, 64);
        tmem_alloc_fn(&tmem_dQ1, 64);
        tmem_alloc_fn(&tmem_S, 64);
        tmem_alloc_fn(&tmem_dP, 64);
    }
    __syncthreads();

    int num_S_blocks = (S + 63) / 64;

    for (int k_blk = 0; k_blk <= q_blk && k_blk < num_S_blocks; k_blk++) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_B, sizeof(__nv_bfloat16) * 64 * 64 * 4);
            tma_load_2d_fn(&tma_K, bar_B, smem_K_L, 0, bh * S + k_blk * 64);
            tma_load_2d_fn(&tma_K, bar_B, smem_K_R, 64, bh * S + k_blk * 64);
            tma_load_2d_fn(&tma_V, bar_B, smem_V_L, 0, bh * S + k_blk * 64);
            tma_load_2d_fn(&tma_V, bar_B, smem_V_R, 64, bh * S + k_blk * 64);
        }
        mbarrier_wait_fn(bar_B, phase_B);
        phase_B ^= 1;
        __syncthreads();

        if (tid < 128) {
            uint64_t desc_Q_L = make_smem_desc_k_major(smem_Q_L);
            uint64_t desc_K_L = make_smem_desc_n_major(smem_K_L);
            uint64_t desc_Q_R = make_smem_desc_k_major(smem_Q_R);
            uint64_t desc_K_R = make_smem_desc_n_major(smem_K_R);
            
            uint32_t idesc_S_QK = make_instr_desc(64, 64, false, true);
            
            for (int k = 0; k < 4; k++) {
                uint64_t dQL = desc_Q_L;
                uint64_t dKL = desc_K_L;
                uint64_t dQR = desc_Q_R;
                uint64_t dKR = desc_K_R;
                
                for (int i = 0; i < k; i++) {
                    dQL = adv_k_major(dQL);
                    dKL = adv_n_major(dKL);
                    dQR = adv_k_major(dQR);
                    dKR = adv_n_major(dKR);
                }
                
                if (k == 0) {
                    umma_f16_cg1(tmem_S, dQL, dKL, idesc_S_QK, 0);
                    umma_f16_cg1(tmem_S, dQR, dKR, idesc_S_QK, 1);
                } else {
                    umma_f16_cg1(tmem_S, dQL, dKL, idesc_S_QK, 1);
                    umma_f16_cg1(tmem_S, dQR, dKR, idesc_S_QK, 1);
                }
            }
        }
        
        if (tid < 128) {
            uint64_t desc_dO_L = make_smem_desc_k_major(smem_dO_L);
            uint64_t desc_V_L = make_smem_desc_n_major(smem_V_L);
            uint64_t desc_dO_R = make_smem_desc_k_major(smem_dO_R);
            uint64_t desc_V_R = make_smem_desc_n_major(smem_V_R);
            
            uint32_t idesc_dP_OV = make_instr_desc(64, 64, false, true);
            
            for (int k = 0; k < 4; k++) {
                uint64_t ddOL = desc_dO_L;
                uint64_t dVL = desc_V_L;
                uint64_t ddOR = desc_dO_R;
                uint64_t dVR = desc_V_R;
                
                for (int i = 0; i < k; i++) {
                    ddOL = adv_k_major(ddOL);
                    dVL = adv_n_major(dVL);
                    ddOR = adv_k_major(ddOR);
                    dVR = adv_n_major(dVR);
                }
                
                if (k == 0) {
                    umma_f16_cg1(tmem_dP, ddOL, dVL, idesc_dP_OV, 0);
                    umma_f16_cg1(tmem_dP, ddOR, dVR, idesc_dP_OV, 1);
                } else {
                    umma_f16_cg1(tmem_dP, ddOL, dVL, idesc_dP_OV, 1);
                    umma_f16_cg1(tmem_dP, ddOR, dVR, idesc_dP_OV, 1);
                }
            }
        }
        umma_commit_1sm_fn(bar_A);
        mbarrier_wait_fn(bar_A, phase_A);
        phase_A ^= 1;
        __syncthreads();

        for(int tid_col = tid; tid_col < 4096; tid_col += blockDim.x) { 
            int row = tid_col / 64;
            int col = tid_col % 64;
            
            float s_val = read_tmem(tmem_S, row, col);
            float dp_val = read_tmem(tmem_dP, row, col);
            
            float p_val = expf(s_val * scale - L_gmem[bh * S + q_blk * 64 + row]);
            if (p_val > 1.0f || p_val < 0.0f) p_val = 0.0f;
            if ((q_blk * 64 + row) < (k_blk * 64 + col) || q_blk * 64 + row >= S || k_blk * 64 + col >= S) {
                p_val = 0.0f;
            }
            
            float ds_val = p_val * (dp_val - head_dOO_shared[row]) * scale;
            
            (( __nv_bfloat16*)smem_dS)[row * 64 + col] = __float2bfloat16(ds_val);
        }
        __syncthreads();
        fence_proxy_async_fn();

        if (tid < 128) {
            uint64_t desc_dS = make_smem_desc_k_major(smem_dS);
            uint64_t desc_K_L = make_smem_desc_k_major(smem_K_L);
            uint64_t desc_K_R = make_smem_desc_k_major(smem_K_R);
            
            uint32_t idesc_dQ_dS_K = make_instr_desc(64, 64, false, false);
            
            for (int k = 0; k < 4; k++) {
                uint64_t ddS = desc_dS;
                uint64_t dKL = desc_K_L;
                uint64_t dKR = desc_K_R;
                
                for (int i = 0; i < k; i++) {
                    ddS = adv_k_major(ddS);
                    dKL = adv_k_major(dKL);
                    dKR = adv_k_major(dKR);
                }
                
                if (k_blk == 0 && k == 0) {
                    umma_f16_cg1(tmem_dQ0, ddS, dKL, idesc_dQ_dS_K, 0);
                    umma_f16_cg1(tmem_dQ1, ddS, dKR, idesc_dQ_dS_K, 0);
                } else {
                    umma_f16_cg1(tmem_dQ0, ddS, dKL, idesc_dQ_dS_K, 1);
                    umma_f16_cg1(tmem_dQ1, ddS, dKR, idesc_dQ_dS_K, 1);
                }
            }
        }
        umma_commit_1sm_fn(bar_A);
        mbarrier_wait_fn(bar_A, phase_A);
        phase_A ^= 1;
        __syncthreads();
    }

    for (int i = tid; i < 4096; i += blockDim.x) {
        int row = i / 64;
        int col = i % 64;
        
        uint32_t val_0, val_1;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(val_0) : "r"(tmem_dQ0 + (row << 16) + col));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(val_1) : "r"(tmem_dQ1 + (row << 16) + col));
        
        float f0 = __uint_as_float(val_0);
        float f1 = __uint_as_float(val_1);
        
        uint32_t g_row = bh * S + q_blk * 64 + row;
        uint32_t g_col_0 = col;
        uint32_t g_col_1 = col + 64;
        
        if (g_row < BHS && g_col_0 < 128) {
            dQ_gmem[g_row * 128 + g_col_0] = __float2bfloat16(f0);
        }
        if (g_row < BHS && g_col_1 < 128) {
            dQ_gmem[g_row * 128 + g_col_1] = __float2bfloat16(f1);
        }
    }

    if (tid == 0) {
        tmem_dealloc_fn(tmem_dQ0, 64);
        tmem_dealloc_fn(tmem_dQ1, 64);
        tmem_dealloc_fn(tmem_S, 64);
        tmem_dealloc_fn(tmem_dP, 64);
    }
}

__global__ void bwd_dK_kernel(
    const __grid_constant__ CUtensorMap tma_Q, const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V, const __grid_constant__ CUtensorMap tma_dO,
    const __grid_constant__ CUtensorMap tma_O,
    const float* L_gmem,
    __nv_bfloat16* dK_gmem,
    int64_t S, int64_t BHS, float scale) 
{
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
        tma_load_2d_fn(&tma_K, bar_B, smem_K_L, 0, bh * S + k_blk * 64);
        tma_load_2d_fn(&tma_K, bar_B, smem_K_R, 64, bh * S + k_blk * 64);
        tma_load_2d_fn(&tma_V, bar_B, smem_V_L, 0, bh * S + k_blk * 64);
        tma_load_2d_fn(&tma_V, bar_B, smem_V_R, 64, bh * S + k_blk * 64);
    }
    mbarrier_wait_fn(bar_B, phase_B);
    phase_B ^= 1;
    __syncthreads();

    uint32_t tmem_dK0, tmem_dK1, tmem_S, tmem_dP;
    if (tid == 0) {
        tmem_alloc_fn(&tmem_dK0, 64);
        tmem_alloc_fn(&tmem_dK1, 64);
        tmem_alloc_fn(&tmem_S, 64);
        tmem_alloc_fn(&tmem_dP, 64);
    }
    __syncthreads();

    int num_S_blocks = (S + 63) / 64;

    for (int q_blk = k_blk; q_blk < num_S_blocks; q_blk++) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_B, sizeof(__nv_bfloat16) * 64 * 64 * 6);
            tma_load_2d_fn(&tma_Q, bar_B, smem_Q_L, 0, bh * S + q_blk * 64);
            tma_load_2d_fn(&tma_Q, bar_B, smem_Q_R, 64, bh * S + q_blk * 64);
            tma_load_2d_fn(&tma_dO, bar_B, smem_dO_L, 0, bh * S + q_blk * 64);
            tma_load_2d_fn(&tma_dO, bar_B, smem_dO_R, 64, bh * S + q_blk * 64);
            tma_load_2d_fn(&tma_O, bar_B, smem_O_L, 0, bh * S + q_blk * 64);
            tma_load_2d_fn(&tma_O, bar_B, smem_O_R, 64, bh * S + q_blk * 64);
        }
        mbarrier_wait_fn(bar_B, phase_B);
        phase_B ^= 1;
        __syncthreads();

        if (tid < 64) {
            float sum = 0;
            for(int c = 0; c < 64; c++) {
                sum += __bfloat162float((( __nv_bfloat16*)smem_O_L)[tid * 64 + c]) * __bfloat162float((( __nv_bfloat16*)smem_dO_L)[tid * 64 + c]);
                sum += __bfloat162float((( __nv_bfloat16*)smem_O_R)[tid * 64 + c]) * __bfloat162float((( __nv_bfloat16*)smem_dO_R)[tid * 64 + c]);
            }
            head_dOO_shared[tid] = sum;
        }
        __syncthreads();

        if (tid < 128) {
            uint64_t desc_Q_L = make_smem_desc_k_major(smem_Q_L);
            uint64_t desc_K_L = make_smem_desc_n_major(smem_K_L);
            uint64_t desc_Q_R = make_smem_desc_k_major(smem_Q_R);
            uint64_t desc_K_R = make_smem_desc_n_major(smem_K_R);
            
            uint32_t idesc_S_QK = make_instr_desc(64, 64, false, true);
            
            for (int k = 0; k < 4; k++) {
                uint64_t dQL = desc_Q_L;
                uint64_t dKL = desc_K_L;
                uint64_t dQR = desc_Q_R;
                uint64_t dKR = desc_K_R;
                
                for (int i = 0; i < k; i++) {
                    dQL = adv_k_major(dQL);
                    dKL = adv_n_major(dKL);
                    dQR = adv_k_major(dQR);
                    dKR = adv_n_major(dKR);
                }
                
                if (k == 0) {
                    umma_f16_cg1(tmem_S, dQL, dKL, idesc_S_QK, 0);
                    umma_f16_cg1(tmem_S, dQR, dKR, idesc_S_QK, 1);
                } else {
                    umma_f16_cg1(tmem_S, dQL, dKL, idesc_S_QK, 1);
                    umma_f16_cg1(tmem_S, dQR, dKR, idesc_S_QK, 1);
                }
            }
        }
        
        if (tid < 128) {
            uint64_t desc_dO_L = make_smem_desc_k_major(smem_dO_L);
            uint64_t desc_V_L = make_smem_desc_n_major(smem_V_L);
            uint64_t desc_dO_R = make_smem_desc_k_major(smem_dO_R);
            uint64_t desc_V_R = make_smem_desc_n_major(smem_V_R);
            
            uint32_t idesc_dP_OV = make_instr_desc(64, 64, false, true);
            
            for (int k = 0; k < 4; k++) {
                uint64_t ddOL = desc_dO_L;
                uint64_t dVL = desc_V_L;
                uint64_t ddOR = desc_dO_R;
                uint64_t dVR = desc_V_R;
                
                for (int i = 0; i < k; i++) {
                    ddOL = adv_k_major(ddOL);
                    dVL = adv_n_major(dVL);
                    ddOR = adv_k_major(ddOR);
                    dVR = adv_n_major(dVR);
                }
                
                if (k == 0) {
                    umma_f16_cg1(tmem_dP, ddOL, dVL, idesc_dP_OV, 0);
                    umma_f16_cg1(tmem_dP, ddOR, dVR, idesc_dP_OV, 1);
                } else {
                    umma_f16_cg1(tmem_dP, ddOL, dVL, idesc_dP_OV, 1);
                    umma_f16_cg1(tmem_dP, ddOR, dVR, idesc_dP_OV, 1);
                }
            }
        }
        umma_commit_1sm_fn(bar_A);
        mbarrier_wait_fn(bar_A, phase_A);
        phase_A ^= 1;
        __syncthreads();

        for(int tid_col = tid; tid_col < 4096; tid_col += blockDim.x) { 
            int row = tid_col / 64;
            int col = tid_col % 64;
            
            float s_val = read_tmem(tmem_S, row, col);
            float dp_val = read_tmem(tmem_dP, row, col);
            
            float p_val = expf(s_val * scale - L_gmem[bh * S + q_blk * 64 + row]);
            if (p_val > 1.0f || p_val < 0.0f) p_val = 0.0f;
            if ((q_blk * 64 + row) < (k_blk * 64 + col) || q_blk * 64 + row >= S || k_blk * 64 + col >= S) {
                p_val = 0.0f;
            }
            
            float ds_val = p_val * (dp_val - head_dOO_shared[col]) * scale;
            
            (( __nv_bfloat16*)smem_dS)[col * 64 + row] = __float2bfloat16(ds_val);
        }
        __syncthreads();
        fence_proxy_async_fn();

        if (tid < 128) {
            uint64_t desc_dS_mnmaj = make_smem_desc_mn_major(smem_dS);
            uint64_t desc_Q_L_nmaj = make_smem_desc_n_major(smem_Q_L);
            uint64_t desc_Q_R_nmaj = make_smem_desc_n_major(smem_Q_R);
            
            uint32_t idesc_dK_dS_Q = make_instr_desc(64, 64, true, true);
            
            for (int k = 0; k < 4; k++) {
                uint64_t ddS = desc_dS_mnmaj;
                uint64_t dQL = desc_Q_L_nmaj;
                uint64_t dQR = desc_Q_R_nmaj;
                
                for (int i = 0; i < k; i++) {
                    ddS = adv_n_major(ddS);
                    dQL = adv_n_major(dQL);
                    dQR = adv_n_major(dQR);
                }
                
                if (q_blk == k_blk && k == 0) {
                    umma_f16_cg1(tmem_dK0, ddS, dQL, idesc_dK_dS_Q, 0);
                    umma_f16_cg1(tmem_dK1, ddS, dQR, idesc_dK_dS_Q, 0);
                } else {
                    umma_f16_cg1(tmem_dK0, ddS, dQL, idesc_dK_dS_Q, 1);
                    umma_f16_cg1(tmem_dK1, ddS, dQR, idesc_dK_dS_Q, 1);
                }
            }
        }
        umma_commit_1sm_fn(bar_A);
        mbarrier_wait_fn(bar_A, phase_A);
        phase_A ^= 1;
        __syncthreads();
    }

    for (int i = tid; i < 4096; i += blockDim.x) {
        int row = i / 64;
        int col = i % 64;
        
        uint32_t val_0, val_1;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(val_0) : "r"(tmem_dK0 + (row << 16) + col));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(val_1) : "r"(tmem_dK1 + (row << 16) + col));
        
        float f0 = __uint_as_float(val_0);
        float f1 = __uint_as_float(val_1);
        
        uint32_t g_row = bh * S + k_blk * 64 + row;
        uint32_t g_col_0 = col;
        uint32_t g_col_1 = col + 64;
        
        if (g_row < BHS && g_col_0 < 128) {
            dK_gmem[g_row * 128 + g_col_0] = __float2bfloat16(f0);
        }
        if (g_row < BHS && g_col_1 < 128) {
            dK_gmem[g_row * 128 + g_col_1] = __float2bfloat16(f1);
        }
    }

    if (tid == 0) {
        tmem_dealloc_fn(tmem_dK0, 64);
        tmem_dealloc_fn(tmem_dK1, 64);
        tmem_dealloc_fn(tmem_S, 64);
        tmem_dealloc_fn(tmem_dP, 64);
    }
}

__global__ void bwd_dV_kernel(
    const __grid_constant__ CUtensorMap tma_Q, const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V, const __grid_constant__ CUtensorMap tma_dO,
    const __grid_constant__ CUtensorMap tma_O,
    const float* L_gmem,
    __nv_bfloat16* dV_gmem,
    int64_t S, int64_t BHS, float scale) 
{
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
        tma_load_2d_fn(&tma_K, bar_B, smem_K_L, 0, bh * S + k_blk * 64);
        tma_load_2d_fn(&tma_K, bar_B, smem_K_R, 64, bh * S + k_blk * 64);
        tma_load_2d_fn(&tma_V, bar_B, smem_V_L, 0, bh * S + k_blk * 64);
        tma_load_2d_fn(&tma_V, bar_B, smem_V_R, 64, bh * S + k_blk * 64);
    }
    mbarrier_wait_fn(bar_B, phase_B);
    phase_B ^= 1;
    __syncthreads();

    uint32_t tmem_dV0, tmem_dV1, tmem_S, tmem_dP;
    if (tid == 0) {
        tmem_alloc_fn(&tmem_dV0, 64);
        tmem_alloc_fn(&tmem_dV1, 64);
        tmem_alloc_fn(&tmem_S, 64);
        tmem_alloc_fn(&tmem_dP, 64);
    }
    __syncthreads();

    int num_S_blocks = (S + 63) / 64;

    for (int q_blk = k_blk; q_blk < num_S_blocks; q_blk++) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_B, sizeof(__nv_bfloat16) * 64 * 64 * 6);
            tma_load_2d_fn(&tma_Q, bar_B, smem_Q_L, 0, bh * S + q_blk * 64);
            tma_load_2d_fn(&tma_Q, bar_B, smem_Q_R, 64, bh * S + q_blk * 64);
            tma_load_2d_fn(&tma_dO, bar_B, smem_dO_L, 0, bh * S + q_blk * 64);
            tma_load_2d_fn(&tma_dO, bar_B, smem_dO_R, 64, bh * S + q_blk * 64);
            tma_load_2d_fn(&tma_O, bar_B, smem_O_L, 0, bh * S + q_blk * 64);
            tma_load_2d_fn(&tma_O, bar_B, smem_O_R, 64, bh * S + q_blk * 64);
        }
        mbarrier_wait_fn(bar_B, phase_B);
        phase_B ^= 1;
        __syncthreads();

        if (tid < 64) {
            float sum = 0;
            for(int c = 0; c < 64; c++) {
                sum += __bfloat162float((( __nv_bfloat16*)smem_O_L)[tid * 64 + c]) * __bfloat162float((( __nv_bfloat16*)smem_dO_L)[tid * 64 + c]);
                sum += __bfloat162float((( __nv_bfloat16*)smem_O_R)[tid * 64 + c]) * __bfloat162float((( __nv_bfloat16*)smem_dO_R)[tid * 64 + c]);
            }
            head_dOO_shared[tid] = sum;
        }
        __syncthreads();

        if (tid < 128) {
            uint64_t desc_Q_L = make_smem_desc_k_major(smem_Q_L);
            uint64_t desc_K_L = make_smem_desc_n_major(smem_K_L);
            uint64_t desc_Q_R = make_smem_desc_k_major(smem_Q_R);
            uint64_t desc_K_R = make_smem_desc_n_major(smem_K_R);
            
            uint32_t idesc_S_QK = make_instr_desc(64, 64, false, true);
            
            for (int k = 0; k < 4; k++) {
                uint64_t dQL = desc_Q_L;
                uint64_t dKL = desc_K_L;
                uint64_t dQR = desc_Q_R;
                uint64_t dKR = desc_K_R;
                
                for (int i = 0; i < k; i++) {
                    dQL = adv_k_major(dQL);
                    dKL = adv_n_major(dKL);
                    dQR = adv_k_major(dQR);
                    dKR = adv_n_major(dKR);
                }
                
                if (k == 0) {
                    umma_f16_cg1(tmem_S, dQL, dKL, idesc_S_QK, 0);
                    umma_f16_cg1(tmem_S, dQR, dKR, idesc_S_QK, 1);
                } else {
                    umma_f16_cg1(tmem_S, dQL, dKL, idesc_S_QK, 1);
                    umma_f16_cg1(tmem_S, dQR, dKR, idesc_S_QK, 1);
                }
            }
        }
        
        if (tid < 128) {
            uint64_t desc_dO_L = make_smem_desc_k_major(smem_dO_L);
            uint64_t desc_V_L = make_smem_desc_n_major(smem_V_L);
            uint64_t desc_dO_R = make_smem_desc_k_major(smem_dO_R);
            uint64_t desc_V_R = make_smem_desc_n_major(smem_V_R);
            
            uint32_t idesc_dP_OV = make_instr_desc(64, 64, false, true);
            
            for (int k = 0; k < 4; k++) {
                uint64_t ddOL = desc_dO_L;
                uint64_t dVL = desc_V_L;
                uint64_t ddOR = desc_dO_R;
                uint64_t dVR = desc_V_R;
                
                for (int i = 0; i < k; i++) {
                    ddOL = adv_k_major(ddOL);
                    dVL = adv_n_major(dVL);
                    ddOR = adv_k_major(ddOR);
                    dVR = adv_n_major(dVR);
                }
                
                if (k == 0) {
                    umma_f16_cg1(tmem_dP, ddOL, dVL, idesc_dP_OV, 0);
                    umma_f16_cg1(tmem_dP, ddOR, dVR, idesc_dP_OV, 1);
                } else {
                    umma_f16_cg1(tmem_dP, ddOL, dVL, idesc_dP_OV, 1);
                    umma_f16_cg1(tmem_dP, ddOR, dVR, idesc_dP_OV, 1);
                }
            }
        }
        umma_commit_1sm_fn(bar_A);
        mbarrier_wait_fn(bar_A, phase_A);
        phase_A ^= 1;
        __syncthreads();

        for(int tid_col = tid; tid_col < 4096; tid_col += blockDim.x) { 
            int row = tid_col / 64;
            int col = tid_col % 64;
            
            float s_val = read_tmem(tmem_S, row, col);
            float dp_val = read_tmem(tmem_dP, row, col);
            
            float p_val = expf(s_val * scale - L_gmem[bh * S + q_blk * 64 + row]);
            if (p_val > 1.0f || p_val < 0.0f) p_val = 0.0f;
            if ((q_blk * 64 + row) < (k_blk * 64 + col) || q_blk * 64 + row >= S || k_blk * 64 + col >= S) {
                p_val = 0.0f;
            }
            
            (( __nv_bfloat16*)smem_P)[row * 64 + col] = __float2bfloat16(p_val);
        }
        __syncthreads();
        fence_proxy_async_fn();

        if (tid < 128) {
            uint64_t desc_P_nmaj = make_smem_desc_n_major(smem_P);
            uint64_t desc_dO_L_nmaj = make_smem_desc_n_major(smem_dO_L);
            uint64_t desc_dO_R_nmaj = make_smem_desc_n_major(smem_dO_R);
            
            uint32_t idesc_dV0_P_dO = make_instr_desc(64, 64, true, true);
            uint32_t idesc_dV1_P_dO = make_instr_desc(64, 64, true, true);
            
            for (int k = 0; k < 4; k++) {
                uint64_t dP = desc_P_nmaj;
                uint64_t ddOL = desc_dO_L_nmaj;
                uint64_t ddOR = desc_dO_R_nmaj;
                
                for (int i = 0; i < k; i++) {
                    dP = adv_n_major(dP);
                    ddOL = adv_n_major(ddOL);
                    ddOR = adv_n_major(ddOR);
                }
                
                if (q_blk == k_blk && k == 0) {
                    umma_f16_cg1(tmem_dV0, dP, ddOL, idesc_dV0_P_dO, 0);
                    umma_f16_cg1(tmem_dV1, dP, ddOR, idesc_dV1_P_dO, 0);
                } else {
                    umma_f16_cg1(tmem_dV0, dP, ddOL, idesc_dV0_P_dO, 1);
                    umma_f16_cg1(tmem_dV1, dP, ddOR, idesc_dV1_P_dO, 1);
                }
            }
        }
        umma_commit_1sm_fn(bar_A);
        mbarrier_wait_fn(bar_A, phase_A);
        phase_A ^= 1;
        __syncthreads();
    }

    for (int i = tid; i < 4096; i += blockDim.x) {
        int row = i / 64;
        int col = i % 64;
        
        uint32_t val_0, val_1;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(val_0) : "r"(tmem_dV0 + (row << 16) + col));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(val_1) : "r"(tmem_dV1 + (row << 16) + col));
        
        float f0 = __uint_as_float(val_0);
        float f1 = __uint_as_float(val_1);
        
        uint32_t g_row = bh * S + k_blk * 64 + row;
        uint32_t g_col_0 = col;
        uint32_t g_col_1 = col + 64;
        
        if (g_row < BHS && g_col_0 < 128) {
            dV_gmem[g_row * 128 + g_col_0] = __float2bfloat16(f0);
        }
        if (g_row < BHS && g_col_1 < 128) {
            dV_gmem[g_row * 128 + g_col_1] = __float2bfloat16(f1);
        }
    }

    if (tid == 0) {
        tmem_dealloc_fn(tmem_dV0, 64);
        tmem_dealloc_fn(tmem_dV1, 64);
        tmem_dealloc_fn(tmem_S, 64);
        tmem_dealloc_fn(tmem_dP, 64);
    }
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

    CUtensorMap tma_Q, tma_K, tma_V, tma_O, tma_dO;

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    create_tma_2d_descriptor_2B(&tma_Q, (void*)((const char*)Q_ptr), d, B * H * S, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_K, (void*)((const char*)K_ptr), d, B * H * S, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_V, (void*)((const char*)V_ptr), d, B * H * S, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_O, (void*)((const char*)O_ptr), d, B * H * S, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_dO, (void*)((const char*)dO_ptr), d, B * H * S, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    int num_S_blocks = (S + 63) / 64;
    dim3 grid(num_S_blocks, B * H);
    dim3 block(128);
    
    int smem_size = 92160;
    CUDA_CHECK(cudaFuncSetAttribute(bwd_dQ_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    CUDA_CHECK(cudaFuncSetAttribute(bwd_dK_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    CUDA_CHECK(cudaFuncSetAttribute(bwd_dV_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    bwd_dQ_kernel<<<grid, block, smem_size, stream>>>(tma_Q, tma_K, tma_V, tma_dO, tma_O, L_ptr, dQ_ptr, S, BHS, scale);
    CUDA_CHECK(cudaGetLastError());
    
    bwd_dK_kernel<<<grid, block, smem_size, stream>>>(tma_Q, tma_K, tma_V, tma_dO, tma_O, L_ptr, dK_ptr, S, BHS, scale);
    CUDA_CHECK(cudaGetLastError());
    
    bwd_dV_kernel<<<grid, block, smem_size, stream>>>(tma_Q, tma_K, tma_V, tma_dO, tma_O, L_ptr, dV_ptr, S, BHS, scale);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

} // namespace tvm_ffi_example_cuda