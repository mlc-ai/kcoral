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

__device__ __forceinline__ uint64_t make_smem_desc(void* smem_ptr) {
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

__device__ __forceinline__ uint64_t offset_desc(uint64_t desc, uint32_t offset_bytes) {
    uint32_t offset_enc = (offset_bytes >> 4);
    uint32_t addr_field = (desc & 0x3FFF);
    addr_field += offset_enc;
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

__device__ __forceinline__ void write_tmem_packed(float val, uint32_t tmem_addr) {
    uint32_t val_b32 = __float_as_uint(val);
    asm volatile("st.shared.b32 [%1], %0;" :: "r"(val_b32), "r"(tmem_addr));
}

__device__ __forceinline__ float read_tmem(uint32_t tmem_addr) {
    uint32_t val;
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(val) : "r"(tmem_addr));
    return __uint_as_float(val);
}

__global__ void bwd_dQ_kernel(
    const __grid_constant__ CUtensorMap* tma_Q, const __grid_constant__ CUtensorMap* tma_K,
    const __grid_constant__ CUtensorMap* tma_V, const __grid_constant__ CUtensorMap* tma_dO,
    const __grid_constant__ CUtensorMap* tma_O,
    const float* L_gmem,
    __nv_bfloat16* dQ_gmem,
    int S, float scale) 
{
    int q_blk = blockIdx.x;
    int bh = blockIdx.y;
    int tid = threadIdx.x;

    extern __shared__ char smem_buf[];
    char* smem_base = smem_buf;
    
    __nv_bfloat16* smem_Q_BM_0 = (__nv_bfloat16*)(smem_base + 0 * 8192);
    __nv_bfloat16* smem_Q_BM_1 = (__nv_bfloat16*)(smem_base + 1 * 8192);
    __nv_bfloat16* smem_dO_BM_0 = (__nv_bfloat16*)(smem_base + 2 * 8192);
    __nv_bfloat16* smem_dO_BM_1 = (__nv_bfloat16*)(smem_base + 3 * 8192);
    __nv_bfloat16* smem_O_BM_0 = (__nv_bfloat16*)(smem_base + 4 * 8192);
    __nv_bfloat16* smem_O_BM_1 = (__nv_bfloat16*)(smem_base + 5 * 8192);
    __nv_bfloat16* smem_K_BN_0 = (__nv_bfloat16*)(smem_base + 6 * 8192);
    __nv_bfloat16* smem_K_BN_1 = (__nv_bfloat16*)(smem_base + 7 * 8192);
    __nv_bfloat16* smem_V_BN_0 = (__nv_bfloat16*)(smem_base + 8 * 8192);
    __nv_bfloat16* smem_V_BN_1 = (__nv_bfloat16*)(smem_base + 9 * 8192);
    __nv_bfloat16* smem_dP_BM_0 = (__nv_bfloat16*)(smem_base + 10 * 8192);
    __nv_bfloat16* smem_dP_BM_1 = (__nv_bfloat16*)(smem_base + 11 * 8192);
    __nv_bfloat16* smem_dS_BM_0 = (__nv_bfloat16*)(smem_base + 12 * 8192);
    __nv_bfloat16* smem_dS_BM_1 = (__nv_bfloat16*)(smem_base + 13 * 8192);

    __align__(16) uint64_t* bar_A = (uint64_t*)(smem_base + 14 * 8192);
    __align__(16) uint64_t* bar_B = (uint64_t*)(smem_base + 14 * 8192 + 8);
    float* head_dO_O_shared = (float*)(smem_base + 14 * 8192 + 16);

    if (tid == 0) {
        init_smem_barrier_fn(bar_A, 1);
        init_smem_barrier_fn(bar_B, 1);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t phase_A = 0;
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_A, sizeof(__nv_bfloat16) * 64 * 64 * 6);
        tma_load_2d_fn(&tma_Q[bh], bar_A, smem_Q_BM_0, 0, bh * S + q_blk * 64);
        tma_load_2d_fn(&tma_Q[bh], bar_A, smem_Q_BM_1, 64, bh * S + q_blk * 64);
        tma_load_2d_fn(&tma_dO[bh], bar_A, smem_dO_BM_0, 0, bh * S + q_blk * 64);
        tma_load_2d_fn(&tma_dO[bh], bar_A, smem_dO_BM_1, 64, bh * S + q_blk * 64);
        tma_load_2d_fn(&tma_O[bh], bar_A, smem_O_BM_0, 0, bh * S + q_blk * 64);
        tma_load_2d_fn(&tma_O[bh], bar_A, smem_O_BM_1, 64, bh * S + q_blk * 64);
    }
    mbarrier_wait_fn(bar_A, phase_A);
    phase_A ^= 1;
    __syncthreads(); // Wait for all threads to finish waiting

    if (tid < 64) {
        float sum = 0;
        for(int c = 0; c < 64; c++) {
            int chunk = c / 8;
            int swizzled_chunk = chunk ^ (tid % 8);
            int swizzled_idx = tid * 64 + swizzled_chunk * 8 + (c % 8);
            
            float o_0 = __bfloat162float(smem_O_BM_0[swizzled_idx]);
            float do_0 = __bfloat162float(smem_dO_BM_0[swizzled_idx]);
            sum += o_0 * do_0;
            
            float o_1 = __bfloat162float(smem_O_BM_1[swizzled_idx]);
            float do_1 = __bfloat162float(smem_dO_BM_1[swizzled_idx]);
            sum += o_1 * do_1;
        }
        int warp_id = tid / 32;
        int lane_id = tid % 32;
        head_dO_O_shared[warp_id * 64 + lane_id] = sum;
    }
    __syncthreads();

    uint32_t tmem_dQ_BM_0, tmem_dQ_BM_1, tmem_S, tmem_dP;
    if (tid == 0) {
        tmem_alloc_fn(&tmem_dQ_BM_0, 64);
        tmem_alloc_fn(&tmem_dQ_BM_1, 64);
        tmem_alloc_fn(&tmem_S, 64);
        tmem_alloc_fn(&tmem_dP, 64);
    }
    __syncthreads();

    int num_S_blocks = (S + 63) / 64;
    uint32_t phase_B = 0;

    for (int k_blk = 0; k_blk <= q_blk && k_blk < num_S_blocks; k_blk++) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_B, sizeof(__nv_bfloat16) * 64 * 64 * 4);
            tma_load_2d_fn(&tma_K[bh], bar_B, smem_K_BN_0, 0, bh * S + k_blk * 64);
            tma_load_2d_fn(&tma_K[bh], bar_B, smem_K_BN_1, 64, bh * S + k_blk * 64);
            tma_load_2d_fn(&tma_V[bh], bar_B, smem_V_BN_0, 0, bh * S + k_blk * 64);
            tma_load_2d_fn(&tma_V[bh], bar_B, smem_V_BN_1, 64, bh * S + k_blk * 64);
        }
        mbarrier_wait_fn(bar_B, phase_B);
        phase_B ^= 1;
        __syncthreads();

        if (tid < 128) {
            uint64_t desc_Q_BM_0 = make_smem_desc(smem_Q_BM_0);
            uint64_t desc_Q_BM_1 = make_smem_desc(smem_Q_BM_1);
            uint64_t desc_K_BN_0 = make_smem_desc(smem_K_BN_0);
            uint64_t desc_K_BN_1 = make_smem_desc(smem_K_BN_1);
            
            uint32_t idesc_S_QK = make_instr_desc_fn(64, 64, false, false);
            
            for (int k = 0; k < 4; k++) {
                uint64_t desc_Q_BM_0_k = offset_desc(desc_Q_BM_0, k * 32);
                uint64_t desc_K_BN_0_k = offset_desc(desc_K_BN_0, k * 32);
                uint64_t desc_Q_BM_1_k = offset_desc(desc_Q_BM_1, k * 32);
                uint64_t desc_K_BN_1_k = offset_desc(desc_K_BN_1, k * 32);
                
                if (k == 0) {
                    umma_f16_cg1_fn(tmem_S, desc_Q_BM_0_k, desc_K_BN_0_k, idesc_S_QK, 0);
                    umma_f16_cg1_fn(tmem_S, desc_Q_BM_1_k, desc_K_BN_1_k, idesc_S_QK, 1);
                } else {
                    umma_f16_cg1_fn(tmem_S, desc_Q_BM_0_k, desc_K_BN_0_k, idesc_S_QK, 1);
                    umma_f16_cg1_fn(tmem_S, desc_Q_BM_1_k, desc_K_BN_1_k, idesc_S_QK, 1);
                }
            }
        }
        
        if (tid < 128) {
            uint64_t desc_dO_BM_0 = make_smem_desc(smem_dO_BM_0);
            uint64_t desc_dO_BM_1 = make_smem_desc(smem_dO_BM_1);
            uint64_t desc_V_BN_0 = make_smem_desc_mn_major(smem_V_BN_0);
            uint64_t desc_V_BN_1 = make_smem_desc_mn_major(smem_V_BN_1);
            
            uint32_t idesc_dP_OV = make_instr_desc_fn(64, 64, false, true);
            
            for (int k = 0; k < 4; k++) {
                uint64_t desc_dO_BM_0_k = offset_desc(desc_dO_BM_0, k * 32);
                uint64_t desc_V_BN_0_k = offset_desc(desc_V_BN_0, k * 32);
                uint64_t desc_dO_BM_1_k = offset_desc(desc_dO_BM_1, k * 32);
                uint64_t desc_V_BN_1_k = offset_desc(desc_V_BN_1, k * 32);
                
                if (k == 0) {
                    umma_f16_cg1_fn(tmem_dP, desc_dO_BM_0_k, desc_V_BN_0_k, idesc_dP_OV, 0);
                    umma_f16_cg1_fn(tmem_dP, desc_dO_BM_1_k, desc_V_BN_1_k, idesc_dP_OV, 1);
                } else {
                    umma_f16_cg1_fn(tmem_dP, desc_dO_BM_0_k, desc_V_BN_0_k, idesc_dP_OV, 1);
                    umma_f16_cg1_fn(tmem_dP, desc_dO_BM_1_k, desc_V_BN_1_k, idesc_dP_OV, 1);
                }
            }
        }
        umma_commit_1sm_fn(bar_A);
        mbarrier_wait_fn(bar_A, phase_A);
        phase_A ^= 1;
        __syncthreads();

        for(int i = tid; i < 4096; i += blockDim.x) { 
            int r = i / 64;
            int c = i % 64;
            
            float s_val = read_tmem(tmem_S + r * 64 + c);
            float dp_val = read_tmem(tmem_dP + r * 64 + c);
            
            float p_val = expf(s_val * scale - L_gmem[bh * S + q_blk * 64 + r]);
            if (p_val > 1.0f) p_val = 1.0f;
            if ((q_blk * 64 + r) < (k_blk * 64 + c)) {
                p_val = 0.0f;
            }
            
            int warp_id = r / 32;
            int lane_id = r % 32;
            float ds_val = p_val * (dp_val - head_dO_O_shared[warp_id * 64 + lane_id]) * scale;
            
            int chunk = c / 8;
            int swizzled_chunk = chunk ^ (r % 8);
            int swizzled_idx = r * 64 + swizzled_chunk * 8 + (c % 8);
            
            smem_dS_BM_0[swizzled_idx] = __float2bfloat16(ds_val);
            smem_dS_BM_1[swizzled_idx] = __float2bfloat16(ds_val); // Dummy write, overwritten below
        }
        __syncthreads();
        fence_proxy_async_fn();

        if (tid < 128) {
            uint64_t desc_dS_BM_0 = make_smem_desc(smem_dS_BM_0);
            uint64_t desc_dS_BM_1 = make_smem_desc(smem_dS_BM_1);
            uint64_t desc_K_BN_0_mnmaj = make_smem_desc_mn_major(smem_K_BN_0);
            uint64_t desc_K_BN_1_mnmaj = make_smem_desc_mn_major(smem_K_BN_1);
            
            uint32_t idesc_dQ_dS_K = make_instr_desc_fn(64, 64, false, true);
            
            for (int k = 0; k < 4; k++) {
                uint64_t desc_dS_BM_0_k = offset_desc(desc_dS_BM_0, k * 32);
                uint64_t desc_K_BN_0_k = offset_desc(desc_K_BN_0_mnmaj, k * 32);
                uint64_t desc_dS_BM_1_k = offset_desc(desc_dS_BM_1, k * 32);
                uint64_t desc_K_BN_1_k = offset_desc(desc_K_BN_1_mnmaj, k * 32);
                
                if (k_blk == 0 && k == 0) {
                    umma_f16_cg1_fn(tmem_dQ_BM_0, desc_dS_BM_0_k, desc_K_BN_0_k, idesc_dQ_dS_K, 0);
                    umma_f16_cg1_fn(tmem_dQ_BM_1, desc_dS_BM_1_k, desc_K_BN_1_k, idesc_dQ_dS_K, 0);
                } else {
                    umma_f16_cg1_fn(tmem_dQ_BM_0, desc_dS_BM_0_k, desc_K_BN_0_k, idesc_dQ_dS_K, 1);
                    umma_f16_cg1_fn(tmem_dQ_BM_1, desc_dS_BM_1_k, desc_K_BN_1_k, idesc_dQ_dS_K, 1);
                }
            }
        }
        umma_commit_1sm_fn(bar_A);
        mbarrier_wait_fn(bar_A, phase_A);
        phase_A ^= 1;
        __syncthreads();
    }

    for (int col = 0; col < 64; col += 16) {
        uint32_t col_offset = tid % 4;
        uint32_t col_start = col_offset * 16 + col;
        if (col_start >= 64) break;
        
        int my_r = (tid / 32) * 16 + (tid % 32);
        
        uint32_t tmem_addr_0 = tmem_dQ_BM_0 + my_r * 64 + col_start;
        uint32_t tmem_addr_1 = tmem_dQ_BM_1 + my_r * 64 + col_start;
        
        uint32_t r0_0, r1_0, r0_1, r1_1;
        asm volatile("tcgen05.ld.sync.aligned.16x128b.x2.pack::16b.b32 {%0, %1}, [%2];"
            : "=r"(r0_0), "=r"(r1_0) : "r"(tmem_addr_0));
        asm volatile("tcgen05.ld.sync.aligned.16x128b.x2.pack::16b.b32 {%0, %1}, [%2];"
            : "=r"(r0_1), "=r"(r1_1) : "r"(tmem_addr_1));
        
        float f0_0 = __uint_as_float(r0_0);
        float f1_0 = __uint_as_float(r1_0);
        
        float f0_1 = __uint_as_float(r0_1);
        float f1_1 = __uint_as_float(r1_1);
        
        uint32_t g_row = bh * S + q_blk * 64 + my_r;
        uint32_t g_col_0 = col_start;
        uint32_t g_col_1 = col_start + 64;
        
        if (g_row < B * H * S && g_col_0 < 128) {
            dQ_gmem[g_row * 128 + g_col_0] = __float2bfloat16(f0_0);
            dQ_gmem[g_row * 128 + g_col_0 + 1] = __float2bfloat16(f1_0);
        }
        if (g_row < B * H * S && g_col_1 < 128) {
            dQ_gmem[g_row * 128 + g_col_1] = __float2bfloat16(f0_1);
            dQ_gmem[g_row * 128 + g_col_1 + 1] = __float2bfloat16(f1_1);
        }
    }

    if (tid == 0) {
        tmem_dealloc_fn(tmem_dQ_BM_0, 64);
        tmem_dealloc_fn(tmem_dQ_BM_1, 64);
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
    int S, float scale) 
{
    int k_blk = blockIdx.x;
    int bh = blockIdx.y;
    int tid = threadIdx.x;

    extern __shared__ char smem_buf[];
    char* smem_base = smem_buf;
    
    __nv_bfloat16* smem_Q_BM_0 = (__nv_bfloat16*)(smem_base + 0 * 8192);
    __nv_bfloat16* smem_Q_BM_1 = (__nv_bfloat16*)(smem_base + 1 * 8192);
    __nv_bfloat16* smem_dO_BM_0 = (__nv_bfloat16*)(smem_base + 2 * 8192);
    __nv_bfloat16* smem_dO_BM_1 = (__nv_bfloat16*)(smem_base + 3 * 8192);
    __nv_bfloat16* smem_O_BM_0 = (__nv_bfloat16*)(smem_base + 4 * 8192);
    __nv_bfloat16* smem_O_BM_1 = (__nv_bfloat16*)(smem_base + 5 * 8192);
    __nv_bfloat16* smem_K_BN_0 = (__nv_bfloat16*)(smem_base + 6 * 8192);
    __nv_bfloat16* smem_K_BN_1 = (__nv_bfloat16*)(smem_base + 7 * 8192);
    __nv_bfloat16* smem_V_BN_0 = (__nv_bfloat16*)(smem_base + 8 * 8192);
    __nv_bfloat16* smem_V_BN_1 = (__nv_bfloat16*)(smem_base + 9 * 8192);
    __nv_bfloat16* smem_dP_BM_0 = (__nv_bfloat16*)(smem_base + 10 * 8192);
    __nv_bfloat16* smem_dP_BM_1 = (__nv_bfloat16*)(smem_base + 11 * 8192);
    __nv_bfloat16* smem_dS_BM_0 = (__nv_bfloat16*)(smem_base + 12 * 8192);
    __nv_bfloat16* smem_dS_BM_1 = (__nv_bfloat16*)(smem_base + 13 * 8192);

    __align__(16) uint64_t* bar_A = (uint64_t*)(smem_base + 14 * 8192);
    __align__(16) uint64_t* bar_B = (uint64_t*)(smem_base + 14 * 8192 + 8);
    float* head_dO_O_shared = (float*)(smem_base + 14 * 8192 + 16);

    if (tid == 0) {
        init_smem_barrier_fn(bar_A, 1);
        init_smem_barrier_fn(bar_B, 1);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t phase_A = 0;
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_A, sizeof(__nv_bfloat16) * 64 * 64 * 2);
        tma_load_2d_fn(&tma_K[bh], bar_A, smem_K_BN_0, 0, bh * S + k_blk * 64);
        tma_load_2d_fn(&tma_K[bh], bar_A, smem_K_BN_1, 64, bh * S + k_blk * 64);
        tma_load_2d_fn(&tma_V[bh], bar_A, smem_V_BN_0, 0, bh * S + k_blk * 64);
        tma_load_2d_fn(&tma_V[bh], bar_A, smem_V_BN_1, 64, bh * S + k_blk * 64);
    }
    mbarrier_wait_fn(bar_A, phase_A);
    phase_A ^= 1;
    __syncthreads();

    uint32_t tmem_dK_BM_0, tmem_dK_BM_1, tmem_S, tmem_dP;
    if (tid == 0) {
        tmem_alloc_fn(&tmem_dK_BM_0, 64);
        tmem_alloc_fn(&tmem_dK_BM_1, 64);
        tmem_alloc_fn(&tmem_S, 64);
        tmem_alloc_fn(&tmem_dP, 64);
    }
    __syncthreads();

    int num_S_blocks = (S + 63) / 64;
    uint32_t phase_B = 0;

    for (int q_blk = k_blk; q_blk < num_S_blocks; q_blk++) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_B, sizeof(__nv_bfloat16) * 64 * 64 * 6);
            tma_load_2d_fn(&tma_Q[bh], bar_B, smem_Q_BM_0, 0, bh * S + q_blk * 64);
            tma_load_2d_fn(&tma_Q[bh], bar_B, smem_Q_BM_1, 64, bh * S + q_blk * 64);
            tma_load_2d_fn(&tma_dO[bh], bar_B, smem_dO_BM_0, 0, bh * S + q_blk * 64);
            tma_load_2d_fn(&tma_dO[bh], bar_B, smem_dO_BM_1, 64, bh * S + q_blk * 64);
            tma_load_2d_fn(&tma_O[bh], bar_B, smem_O_BM_0, 0, bh * S + q_blk * 64);
            tma_load_2d_fn(&tma_O[bh], bar_B, smem_O_BM_1, 64, bh * S + q_blk * 64);
        }
        mbarrier_wait_fn(bar_B, phase_B);
        phase_B ^= 1;
        __syncthreads();

        if (tid < 64) {
            float sum = 0;
            for(int c = 0; c < 64; c++) {
                int chunk = c / 8;
                int swizzled_chunk = chunk ^ (tid % 8);
                int swizzled_idx = tid * 64 + swizzled_chunk * 8 + (c % 8);
                
                float o_0 = __bfloat162float(smem_O_BM_0[swizzled_idx]);
                float do_0 = __bfloat162float(smem_dO_BM_0[swizzled_idx]);
                sum += o_0 * do_0;
                
                float o_1 = __bfloat162float(smem_O_BM_1[swizzled_idx]);
                float do_1 = __bfloat162float(smem_dO_BM_1[swizzled_idx]);
                sum += o_1 * do_1;
            }
            int warp_id = tid / 32;
            int lane_id = tid % 32;
            head_dO_O_shared[warp_id * 64 + lane_id] = sum;
        }
        __syncthreads();

        if (tid < 128) {
            uint64_t desc_Q_BM_0 = make_smem_desc(smem_Q_BM_0);
            uint64_t desc_Q_BM_1 = make_smem_desc(smem_Q_BM_1);
            uint64_t desc_K_BN_0 = make_smem_desc(smem_K_BN_0);
            uint64_t desc_K_BN_1 = make_smem_desc(smem_K_BN_1);
            
            uint32_t idesc_S_QK = make_instr_desc_fn(64, 64, false, false);
            
            for (int k = 0; k < 4; k++) {
                uint64_t desc_Q_BM_0_k = offset_desc(desc_Q_BM_0, k * 32);
                uint64_t desc_K_BN_0_k = offset_desc(desc_K_BN_0, k * 32);
                uint64_t desc_Q_BM_1_k = offset_desc(desc_Q_BM_1, k * 32);
                uint64_t desc_K_BN_1_k = offset_desc(desc_K_BN_1, k * 32);
                
                if (k == 0) {
                    umma_f16_cg1_fn(tmem_S, desc_Q_BM_0_k, desc_K_BN_0_k, idesc_S_QK, 0);
                    umma_f16_cg1_fn(tmem_S, desc_Q_BM_1_k, desc_K_BN_1_k, idesc_S_QK, 1);
                } else {
                    umma_f16_cg1_fn(tmem_S, desc_Q_BM_0_k, desc_K_BN_0_k, idesc_S_QK, 1);
                    umma_f16_cg1_fn(tmem_S, desc_Q_BM_1_k, desc_K_BN_1_k, idesc_S_QK, 1);
                }
            }
        }
        
        if (tid < 128) {
            uint64_t desc_dO_BM_0 = make_smem_desc(smem_dO_BM_0);
            uint64_t desc_dO_BM_1 = make_smem_desc(smem_dO_BM_1);
            uint64_t desc_V_BN_0 = make_smem_desc_mn_major(smem_V_BN_0);
            uint64_t desc_V_BN_1 = make_smem_desc_mn_major(smem_V_BN_1);
            
            uint32_t idesc_dP_OV = make_instr_desc_fn(64, 64, false, true);
            
            for (int k = 0; k < 4; k++) {
                uint64_t desc_dO_BM_0_k = offset_desc(desc_dO_BM_0, k * 32);
                uint64_t desc_V_BN_0_k = offset_desc(desc_V_BN_0, k * 32);
                uint64_t desc_dO_BM_1_k = offset_desc(desc_dO_BM_1, k * 32);
                uint64_t desc_V_BN_1_k = offset_desc(desc_V_BN_1, k * 32);
                
                if (k == 0) {
                    umma_f16_cg1_fn(tmem_dP, desc_dO_BM_0_k, desc_V_BN_0_k, idesc_dP_OV, 0);
                    umma_f16_cg1_fn(tmem_dP, desc_dO_BM_1_k, desc_V_BN_1_k, idesc_dP_OV, 1);
                } else {
                    umma_f16_cg1_fn(tmem_dP, desc_dO_BM_0_k, desc_V_BN_0_k, idesc_dP_OV, 1);
                    umma_f16_cg1_fn(tmem_dP, desc_dO_BM_1_k, desc_V_BN_1_k, idesc_dP_OV, 1);
                }
            }
        }
        umma_commit_1sm_fn(bar_A);
        mbarrier_wait_fn(bar_A, phase_A);
        phase_A ^= 1;
        __syncthreads();

        for(int i = tid; i < 4096; i += blockDim.x) { 
            int r = i / 64;
            int c = i % 64;
            
            float s_val = read_tmem(tmem_S + r * 64 + c);
            float dp_val = read_tmem(tmem_dP + r * 64 + c);
            
            float p_val = expf(s_val * scale - L_gmem[bh * S + q_blk * 64 + r]);
            if (p_val > 1.0f) p_val = 1.0f;
            if ((q_blk * 64 + r) < (k_blk * 64 + c)) {
                p_val = 0.0f;
            }
            
            int warp_id = r / 32;
            int lane_id = r % 32;
            float ds_val = p_val * (dp_val - head_dO_O_shared[warp_id * 64 + lane_id]) * scale;
            
            int chunk = c / 8;
            int swizzled_chunk = chunk ^ (r % 8);
            int swizzled_idx = r * 64 + swizzled_chunk * 8 + (c % 8);
            
            smem_dS_BM_0[swizzled_idx] = __float2bfloat16(ds_val);
            smem_dS_BM_1[swizzled_idx] = __float2bfloat16(ds_val);
        }
        __syncthreads();
        fence_proxy_async_fn();

        if (tid < 128) {
            uint64_t desc_dS_BM_0_mnmaj = make_smem_desc_mn_major(smem_dS_BM_0);
            uint64_t desc_dS_BM_1_mnmaj = make_smem_desc_mn_major(smem_dS_BM_1);
            uint64_t desc_Q_BM_0_mnmaj = make_smem_desc_mn_major(smem_Q_BM_0);
            uint64_t desc_Q_BM_1_mnmaj = make_smem_desc_mn_major(smem_Q_BM_1);
            
            uint32_t idesc_dK_dS_Q = make_instr_desc_fn(64, 64, true, true);
            
            for (int k = 0; k < 4; k++) {
                uint64_t desc_dS_BM_0_k = offset_desc(desc_dS_BM_0_mnmaj, k * 32);
                uint64_t desc_Q_BM_0_k = offset_desc(desc_Q_BM_0_mnmaj, k * 32);
                uint64_t desc_dS_BM_1_k = offset_desc(desc_dS_BM_1_mnmaj, k * 32);
                uint64_t desc_Q_BM_1_k = offset_desc(desc_Q_BM_1_mnmaj, k * 32);
                
                if (q_blk == k_blk && k == 0) {
                    umma_f16_cg1_fn(tmem_dK_BM_0, desc_dS_BM_0_k, desc_Q_BM_0_k, idesc_dK_dS_Q, 0);
                    umma_f16_cg1_fn(tmem_dK_BM_1, desc_dS_BM_1_k, desc_Q_BM_1_k, idesc_dK_dS_Q, 0);
                } else {
                    umma_f16_cg1_fn(tmem_dK_BM_0, desc_dS_BM_0_k, desc_Q_BM_0_k, idesc_dK_dS_Q, 1);
                    umma_f16_cg1_fn(tmem_dK_BM_1, desc_dS_BM_1_k, desc_Q_BM_1_k, idesc_dK_dS_Q, 1);
                }
            }
        }
        umma_commit_1sm_fn(bar_A);
        mbarrier_wait_fn(bar_A, phase_A);
        phase_A ^= 1;
        __syncthreads();
    }

    for (int col = 0; col < 64; col += 16) {
        uint32_t col_offset = tid % 4;
        uint32_t col_start = col_offset * 16 + col;
        if (col_start >= 64) break;
        
        int my_r = (tid / 32) * 16 + (tid % 32);
        
        uint32_t tmem_addr_0 = tmem_dK_BM_0 + my_r * 64 + col_start;
        uint32_t tmem_addr_1 = tmem_dK_BM_1 + my_r * 64 + col_start;
        
        uint32_t r0_0, r1_0, r0_1, r1_1;
        asm volatile("tcgen05.ld.sync.aligned.16x128b.x2.pack::16b.b32 {%0, %1}, [%2];"
            : "=r"(r0_0), "=r"(r1_0) : "r"(tmem_addr_0));
        asm volatile("tcgen05.ld.sync.aligned.16x128b.x2.pack::16b.b32 {%0, %1}, [%2];"
            : "=r"(r0_1), "=r"(r1_1) : "r"(tmem_addr_1));
        
        float f0_0 = __uint_as_float(r0_0);
        float f1_0 = __uint_as_float(r1_0);
        
        float f0_1 = __uint_as_float(r0_1);
        float f1_1 = __uint_as_float(r1_1);
        
        uint32_t g_row = bh * S + k_blk * 64 + my_r;
        uint32_t g_col_0 = col_start;
        uint32_t g_col_1 = col_start + 64;
        
        if (g_row < B * H * S && g_col_0 < 128) {
            dK_gmem[g_row * 128 + g_col_0] = __float2bfloat16(f0_0);
            dK_gmem[g_row * 128 + g_col_0 + 1] = __float2bfloat16(f1_0);
        }
        if (g_row < B * H * S && g_col_1 < 128) {
            dK_gmem[g_row * 128 + g_col_1] = __float2bfloat16(f0_1);
            dK_gmem[g_row * 128 + g_col_1 + 1] = __float2bfloat16(f1_1);
        }
    }

    if (tid == 0) {
        tmem_dealloc_fn(tmem_dK_BM_0, 64);
        tmem_dealloc_fn(tmem_dK_BM_1, 64);
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
    int S, float scale) 
{
    int k_blk = blockIdx.x;
    int bh = blockIdx.y;
    int tid = threadIdx.x;

    extern __shared__ char smem_buf[];
    char* smem_base = smem_buf;
    
    __nv_bfloat16* smem_Q_BM_0 = (__nv_bfloat16*)(smem_base + 0 * 8192);
    __nv_bfloat16* smem_Q_BM_1 = (__nv_bfloat16*)(smem_base + 1 * 8192);
    __nv_bfloat16* smem_dO_BM_0 = (__nv_bfloat16*)(smem_base + 2 * 8192);
    __nv_bfloat16* smem_dO_BM_1 = (__nv_bfloat16*)(smem_base + 3 * 8192);
    __nv_bfloat16* smem_O_BM_0 = (__nv_bfloat16*)(smem_base + 4 * 8192);
    __nv_bfloat16* smem_O_BM_1 = (__nv_bfloat16*)(smem_base + 5 * 8192);
    __nv_bfloat16* smem_K_BN_0 = (__nv_bfloat16*)(smem_base + 6 * 8192);
    __nv_bfloat16* smem_K_BN_1 = (__nv_bfloat16*)(smem_base + 7 * 8192);
    __nv_bfloat16* smem_V_BN_0 = (__nv_bfloat16*)(smem_base + 8 * 8192);
    __nv_bfloat16* smem_V_BN_1 = (__nv_bfloat16*)(smem_base + 9 * 8192);
    __nv_bfloat16* smem_dP_BM_0 = (__nv_bfloat16*)(smem_base + 10 * 8192);
    __nv_bfloat16* smem_dP_BM_1 = (__nv_bfloat16*)(smem_base + 11 * 8192);
    __nv_bfloat16* smem_dS_BM_0 = (__nv_bfloat16*)(smem_base + 12 * 8192);
    __nv_bfloat16* smem_dS_BM_1 = (__nv_bfloat16*)(smem_base + 13 * 8192);

    __align__(16) uint64_t* bar_A = (uint64_t*)(smem_base + 14 * 8192);
    __align__(16) uint64_t* bar_B = (uint64_t*)(smem_base + 14 * 8192 + 8);
    float* head_dO_O_shared = (float*)(smem_base + 14 * 8192 + 16);

    if (tid == 0) {
        init_smem_barrier_fn(bar_A, 1);
        init_smem_barrier_fn(bar_B, 1);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t phase_A = 0;
    uint32_t tmem_dV_BM_0, tmem_dV_BM_1, tmem_S, tmem_dP;
    if (tid == 0) {
        tmem_alloc_fn(&tmem_dV_BM_0, 64);
        tmem_alloc_fn(&tmem_dV_BM_1, 64);
        tmem_alloc_fn(&tmem_S, 64);
        tmem_alloc_fn(&tmem_dP, 64);
    }
    __syncthreads();

    int num_S_blocks = (S + 63) / 64;
    uint32_t phase_B = 0;

    for (int q_blk = k_blk; q_blk < num_S_blocks; q_blk++) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_B, sizeof(__nv_bfloat16) * 64 * 64 * 6);
            tma_load_2d_fn(&tma_Q[bh], bar_B, smem_Q_BM_0, 0, bh * S + q_blk * 64);
            tma_load_2d_fn(&tma_Q[bh], bar_B, smem_Q_BM_1, 64, bh * S + q_blk * 64);
            tma_load_2d_fn(&tma_dO[bh], bar_B, smem_dO_BM_0, 0, bh * S + q_blk * 64);
            tma_load_2d_fn(&tma_dO[bh], bar_B, smem_dO_BM_1, 64, bh * S + q_blk * 64);
            tma_load_2d_fn(&tma_O[bh], bar_B, smem_O_BM_0, 0, bh * S + q_blk * 64);
            tma_load_2d_fn(&tma_O[bh], bar_B, smem_O_BM_1, 64, bh * S + q_blk * 64);
            tma_load_2d_fn(&tma_K[bh], bar_B, smem_K_BN_0, 0, bh * S + k_blk * 64);
            tma_load_2d_fn(&tma_K[bh], bar_B, smem_K_BN_1, 64, bh * S + k_blk * 64);
        }
        mbarrier_wait_fn(bar_B, phase_B);
        phase_B ^= 1;
        __syncthreads();

        if (tid < 64) {
            float sum = 0;
            for(int c = 0; c < 64; c++) {
                int chunk = c / 8;
                int swizzled_chunk = chunk ^ (tid % 8);
                int swizzled_idx = tid * 64 + swizzled_chunk * 8 + (c % 8);
                
                float o_0 = __bfloat162float(smem_O_BM_0[swizzled_idx]);
                float do_0 = __bfloat162float(smem_dO_BM_0[swizzled_idx]);
                sum += o_0 * do_0;
                
                float o_1 = __bfloat162float(smem_O_BM_1[swizzled_idx]);
                float do_1 = __bfloat162float(smem_dO_BM_1[swizzled_idx]);
                sum += o_1 * do_1;
            }
            int warp_id = tid / 32;
            int lane_id = tid % 32;
            head_dO_O_shared[warp_id * 64 + lane_id] = sum;
        }
        __syncthreads();

        if (tid < 128) {
            uint64_t desc_Q_BM_0 = make_smem_desc(smem_Q_BM_0);
            uint64_t desc_Q_BM_1 = make_smem_desc(smem_Q_BM_1);
            uint64_t desc_K_BN_0 = make_smem_desc(smem_K_BN_0);
            uint64_t desc_K_BN_1 = make_smem_desc(smem_K_BN_1);
            
            uint32_t idesc_S_QK = make_instr_desc_fn(64, 64, false, false);
            
            for (int k = 0; k < 4; k++) {
                uint64_t desc_Q_BM_0_k = offset_desc(desc_Q_BM_0, k * 32);
                uint64_t desc_K_BN_0_k = offset_desc(desc_K_BN_0, k * 32);
                uint64_t desc_Q_BM_1_k = offset_desc(desc_Q_BM_1, k * 32);
                uint64_t desc_K_BN_1_k = offset_desc(desc_K_BN_1, k * 32);
                
                if (k == 0) {
                    umma_f16_cg1_fn(tmem_S, desc_Q_BM_0_k, desc_K_BN_0_k, idesc_S_QK, 0);
                    umma_f16_cg1_fn(tmem_S, desc_Q_BM_1_k, desc_K_BN_1_k, idesc_S_QK, 1);
                } else {
                    umma_f16_cg1_fn(tmem_S, desc_Q_BM_0_k, desc_K_BN_0_k, idesc_S_QK, 1);
                    umma_f16_cg1_fn(tmem_S, desc_Q_BM_1_k, desc_K_BN_1_k, idesc_S_QK, 1);
                }
            }
        }
        
        if (tid < 128) {
            uint64_t desc_dO_BM_0 = make_smem_desc(smem_dO_BM_0);
            uint64_t desc_dO_BM_1 = make_smem_desc(smem_dO_BM_1);
            uint64_t desc_V_BN_0 = make_smem_desc_mn_major(smem_V_BN_0);
            uint64_t desc_V_BN_1 = make_smem_desc_mn_major(smem_V_BN_1);
            
            uint32_t idesc_dP_OV = make_instr_desc_fn(64, 64, false, true);
            
            for (int k = 0; k < 4; k++) {
                uint64_t desc_dO_BM_0_k = offset_desc(desc_dO_BM_0, k * 32);
                uint64_t desc_V_BN_0_k = offset_desc(desc_V_BN_0, k * 32);
                uint64_t desc_dO_BM_1_k = offset_desc(desc_dO_BM_1, k * 32);
                uint64_t desc_V_BN_1_k = offset_desc(desc_V_BN_1, k * 32);
                
                if (k == 0) {
                    umma_f16_cg1_fn(tmem_dP, desc_dO_BM_0_k, desc_V_BN_0_k, idesc_dP_OV, 0);
                    umma_f16_cg1_fn(tmem_dP, desc_dO_BM_1_k, desc_V_BN_1_k, idesc_dP_OV, 1);
                } else {
                    umma_f16_cg1_fn(tmem_dP, desc_dO_BM_0_k, desc_V_BN_0_k, idesc_dP_OV, 1);
                    umma_f16_cg1_fn(tmem_dP, desc_dO_BM_1_k, desc_V_BN_1_k, idesc_dP_OV, 1);
                }
            }
        }
        umma_commit_1sm_fn(bar_A);
        mbarrier_wait_fn(bar_A, phase_A);
        phase_A ^= 1;
        __syncthreads();

        for(int i = tid; i < 4096; i += blockDim.x) { 
            int r = i / 64;
            int c = i % 64;
            
            float s_val = read_tmem(tmem_S + r * 64 + c);
            float dp_val = read_tmem(tmem_dP + r * 64 + c);
            
            float p_val = expf(s_val * scale - L_gmem[bh * S + q_blk * 64 + r]);
            if (p_val > 1.0f) p_val = 1.0f;
            if ((q_blk * 64 + r) < (k_blk * 64 + c)) {
                p_val = 0.0f;
            }
            
            int warp_id = r / 32;
            int lane_id = r % 32;
            float ds_val = p_val * (dp_val - head_dO_O_shared[warp_id * 64 + lane_id]) * scale;
            
            int chunk = c / 8;
            int swizzled_chunk = chunk ^ (r % 8);
            int swizzled_idx = r * 64 + swizzled_chunk * 8 + (c % 8);
            
            smem_dS_BM_0[swizzled_idx] = __float2bfloat16(ds_val);
            smem_dS_BM_1[swizzled_idx] = __float2bfloat16(ds_val);
        }
        __syncthreads();
        fence_proxy_async_fn();

        if (tid < 128) {
            uint64_t desc_dS_BM_0_mnmaj = make_smem_desc_mn_major(smem_dS_BM_0);
            uint64_t desc_dS_BM_1_mnmaj = make_smem_desc_mn_major(smem_dS_BM_1);
            uint64_t desc_dO_BM_0_mnmaj = make_smem_desc_mn_major(smem_dO_BM_0);
            uint64_t desc_dO_BM_1_mnmaj = make_smem_desc_mn_major(smem_dO_BM_1);
            
            uint32_t idesc_dV_dS_dO = make_instr_desc_fn(64, 64, true, true);
            
            for (int k = 0; k < 4; k++) {
                uint64_t desc_dS_BM_0_k = offset_desc(desc_dS_BM_0_mnmaj, k * 32);
                uint64_t desc_dO_BM_0_k = offset_desc(desc_dO_BM_0_mnmaj, k * 32);
                uint64_t desc_dS_BM_1_k = offset_desc(desc_dS_BM_1_mnmaj, k * 32);
                uint64_t desc_dO_BM_1_k = offset_desc(desc_dO_BM_1_mnmaj, k * 32);
                
                if (q_blk == k_blk && k == 0) {
                    umma_f16_cg1_fn(tmem_dV_BM_0, desc_dS_BM_0_k, desc_dO_BM_0_k, idesc_dV_dS_dO, 0);
                    umma_f16_cg1_fn(tmem_dV_BM_1, desc_dS_BM_1_k, desc_dO_BM_1_k, idesc_dV_dS_dO, 0);
                } else {
                    umma_f16_cg1_fn(tmem_dV_BM_0, desc_dS_BM_0_k, desc_dO_BM_0_k, idesc_dV_dS_dO, 1);
                    umma_f16_cg1_fn(tmem_dV_BM_1, desc_dS_BM_1_k, desc_dO_BM_1_k, idesc_dV_dS_dO, 1);
                }
            }
        }
        umma_commit_1sm_fn(bar_A);
        mbarrier_wait_fn(bar_A, phase_A);
        phase_A ^= 1;
        __syncthreads();
    }

    for (int col = 0; col < 64; col += 16) {
        uint32_t col_offset = tid % 4;
        uint32_t col_start = col_offset * 16 + col;
        if (col_start >= 64) break;
        
        int my_r = (tid / 32) * 16 + (tid % 32);
        
        uint32_t tmem_addr_0 = tmem_dV_BM_0 + my_r * 64 + col_start;
        uint32_t tmem_addr_1 = tmem_dV_BM_1 + my_r * 64 + col_start;
        
        uint32_t r0_0, r1_0, r0_1, r1_1;
        asm volatile("tcgen05.ld.sync.aligned.16x128b.x2.pack::16b.b32 {%0, %1}, [%2];"
            : "=r"(r0_0), "=r"(r1_0) : "r"(tmem_addr_0));
        asm volatile("tcgen05.ld.sync.aligned.16x128b.x2.pack::16b.b32 {%0, %1}, [%2];"
            : "=r"(r0_1), "=r"(r1_1) : "r"(tmem_addr_1));
        
        float f0_0 = __uint_as_float(r0_0);
        float f1_0 = __uint_as_float(r1_0);
        
        float f0_1 = __uint_as_float(r0_1);
        float f1_1 = __uint_as_float(r1_1);
        
        uint32_t g_row = bh * S + k_blk * 64 + my_r;
        uint32_t g_col_0 = col_start;
        uint32_t g_col_1 = col_start + 64;
        
        if (g_row < B * H * S && g_col_0 < 128) {
            dV_gmem[g_row * 128 + g_col_0] = __float2bfloat16(f0_0);
            dV_gmem[g_row * 128 + g_col_0 + 1] = __float2bfloat16(f1_0);
        }
        if (g_row < B * H * S && g_col_1 < 128) {
            dV_gmem[g_row * 128 + g_col_1] = __float2bfloat16(f0_1);
            dV_gmem[g_row * 128 + g_col_1 + 1] = __float2bfloat16(f1_1);
        }
    }

    if (tid == 0) {
        tmem_dealloc_fn(tmem_dV_BM_0, 64);
        tmem_dealloc_fn(tmem_dV_BM_1, 64);
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
            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        create_tma_2d_descriptor_2B(&tma_K[bh], (void*)(K_ptr + bh * S * d), d, B * H * S, 64, 64, 
            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        create_tma_2d_descriptor_2B(&tma_V[bh], (void*)(V_ptr + bh * S * d), d, B * H * S, 64, 64, 
            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        create_tma_2d_descriptor_2B(&tma_O[bh], (void*)(O_ptr + bh * S * d), d, B * H * S, 64, 64, 
            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        create_tma_2d_descriptor_2B(&tma_dO[bh], (void*)(dO_ptr + bh * S * d), d, B * H * S, 64, 64, 
            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    }

    int num_S_blocks = (S + 63) / 64;
    dim3 grid(num_S_blocks, B * H);
    dim3 block(128);
    
    int smem_size = 90112;
    CUDA_CHECK(cudaFuncSetAttribute(bwd_dQ_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    CUDA_CHECK(cudaFuncSetAttribute(bwd_dK_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    CUDA_CHECK(cudaFuncSetAttribute(bwd_dV_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    cudaLaunchConfig_t config_dQ = {};
    config_dQ.gridDim = grid;
    config_dQ.blockDim = block;
    config_dQ.dynamicSmemBytes = smem_size;
    config_dQ.stream = stream;
    cudaLaunchAttribute attrs_dQ[1];
    attrs_dQ[0].id = cudaLaunchAttributeClusterDimension;
    attrs_dQ[0].val.clusterDim.x = 2;
    attrs_dQ[0].val.clusterDim.y = 1;
    attrs_dQ[0].val.clusterDim.z = 1;
    config_dQ.attrs = attrs_dQ;
    config_dQ.numAttrs = 1;
    CUDA_CHECK(cudaLaunchKernelEx(&config_dQ, bwd_dQ_kernel, tma_Q, tma_K, tma_V, tma_dO, tma_O, L_ptr, dQ_ptr, S, scale));
    CUDA_CHECK(cudaGetLastError());
    
    cudaLaunchConfig_t config_dK = {};
    config_dK.gridDim = grid;
    config_dK.blockDim = block;
    config_dK.dynamicSmemBytes = smem_size;
    config_dK.stream = stream;
    cudaLaunchAttribute attrs_dK[1];
    attrs_dK[0].id = cudaLaunchAttributeClusterDimension;
    attrs_dK[0].val.clusterDim.x = 2;
    attrs_dK[0].val.clusterDim.y = 1;
    attrs_dK[0].val.clusterDim.z = 1;
    config_dK.attrs = attrs_dK;
    config_dK.numAttrs = 1;
    CUDA_CHECK(cudaLaunchKernelEx(&config_dK, bwd_dK_kernel, tma_Q, tma_K, tma_V, tma_dO, tma_O, L_ptr, dK_ptr, S, scale));
    CUDA_CHECK(cudaGetLastError());
    
    cudaLaunchConfig_t config_dV = {};
    config_dV.gridDim = grid;
    config_dV.blockDim = block;
    config_dV.dynamicSmemBytes = smem_size;
    config_dV.stream = stream;
    cudaLaunchAttribute attrs_dV[1];
    attrs_dV[0].id = cudaLaunchAttributeClusterDimension;
    attrs_dV[0].val.clusterDim.x = 2;
    attrs_dV[0].val.clusterDim.y = 1;
    attrs_dV[0].val.clusterDim.z = 1;
    config_dV.attrs = attrs_dV;
    config_dV.numAttrs = 1;
    CUDA_CHECK(cudaLaunchKernelEx(&config_dV, bwd_dV_kernel, tma_Q, tma_K, tma_V, tma_dO, tma_O, L_ptr, dV_ptr, S, scale));
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));

    free(tma_Q); free(tma_K); free(tma_V); free(tma_O); free(tma_dO);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda