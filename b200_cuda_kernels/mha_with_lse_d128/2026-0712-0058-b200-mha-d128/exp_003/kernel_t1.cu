#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
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

#define CU_CHECK(call) do {                                      \
    CUresult _e = (call);                                        \
    if (_e != CUDA_SUCCESS) {                                    \
        fprintf(stderr, "CU error %d at %s:%d\n", (int)_e,       \
                __FILE__, __LINE__);                               \
        exit(1);                                                   \
    }                                                              \
} while(0)


__device__ __forceinline__ uint32_t to_bits(float f) {
    uint32_t bits;
    asm volatile("{\n"
                 ".reg .f32 val;\n"
                 "mov.b32 val, %1;\n"
                 "mov.b32 %0, val;\n"
                 "}\n" : "=r"(bits) : "f"(f));
    return bits;
}

__device__ __forceinline__ float tmem_to_float(uint32_t val) {
    float f;
    asm volatile("mov.f32 %0, %1;" : "=f"(f) : "r"(val));
    return f;
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

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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

__device__ __forceinline__ void named_barrier_sync_fn(int bar_id, int count) {
    asm volatile("barrier.sync.aligned %0, %1;" :: "r"(bar_id), "r"(count));
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
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

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
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
        ".mbarrier::arrive::one.shared.b64"
        " [%0];"
        :: "r"(a));
}

template <typename T>
__device__ __forceinline__ uint64_t make_smem_desc_k_major(void* smem_ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint32_t lbo = 8 * sizeof(T); 
    uint32_t sbo = 1024;
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
}

template <typename T>
__device__ __forceinline__ uint64_t make_smem_desc_mn_major(void* smem_ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint32_t sbo = 1024;
    uint32_t lbo = (128 / 8) * sbo;
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
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

__device__ __forceinline__ uint32_t make_instr_desc_fn_mn_major_b(uint32_t M, uint32_t N) {
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

constexpr int FUSED_HEAD = 1;

__global__ __launch_bounds__(128, 2)
void fa_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O, float* LSE,
    uint32_t S, uint32_t scale_factor, uint32_t H, uint32_t B) 
{
    int q_block = blockIdx.x;
    int h_idx = blockIdx.y % H;
    int b_idx = blockIdx.z;
    
    uint32_t m_block = q_block * 128;
    if (m_block >= S) return;

    extern __shared__ char smem_pool[];
    __nv_bfloat16* smem_Q = (__nv_bfloat16*)smem_pool;                                 
    __nv_bfloat16* smem_K_0 = (__nv_bfloat16*)(smem_pool + 32768);                      
    __nv_bfloat16* smem_K_1 = (__nv_bfloat16*)(smem_pool + 65536);                      
    __nv_bfloat16* smem_V_0 = (__nv_bfloat16*)(smem_pool + 98304);                      
    __nv_bfloat16* smem_V_1 = (__nv_bfloat16*)(smem_pool + 131072);                     
    __nv_bfloat16* smem_P = (__nv_bfloat16*)(smem_pool + 163840);                       
    __nv_bfloat16* smem_P1 = (__nv_bfloat16*)(smem_pool + 176128);                      

    uint64_t* mbar_Q = (uint64_t*)(smem_pool + 197120);
    uint64_t* mbar_KV_0 = (uint64_t*)(smem_pool + 197128);
    uint64_t* mbar_KV_1 = (uint64_t*)(smem_pool + 197136);
    uint64_t* mbar_UMA_0 = (uint64_t*)(smem_pool + 197144);
    uint64_t* mbar_UMA_1 = (uint64_t*)(smem_pool + 197152);

    uint64_t* mbar_KV[2] = {mbar_KV_0, mbar_KV_1};
    __nv_bfloat16* smem_K[2] = {smem_K_0, smem_K_1};
    __nv_bfloat16* smem_V[2] = {smem_V_0, smem_V_1};

    uint32_t tmem_S, tmem_P, tmem_O;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(mbar_KV_0, 1);
        init_smem_barrier_fn(mbar_KV_1, 1);
        init_smem_barrier_fn(mbar_UMA_0, 1);
        init_smem_barrier_fn(mbar_UMA_1, 1);

        tmem_alloc_fn(&tmem_S, 128);
        tmem_alloc_fn(&tmem_P, 128);
        tmem_alloc_fn(&tmem_O, 128);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t expect_tx_Q = 32768; 
    mbarrier_arrive_and_expect_tx_fn(mbar_Q, expect_tx_Q);

    if (threadIdx.x == 0) {
        tma_load_2d_fn(&tma_Q, mbar_Q, smem_Q, 0, b_idx * H * S + h_idx * S + m_block);
        tma_load_2d_fn(&tma_Q, mbar_Q, smem_Q + 8192, 64, b_idx * H * S + h_idx * S + m_block);
    }

    int num_steps = (S + 127) / 128;
    uint32_t expect_tx = 32768; 
    
    if (num_steps > 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_KV_0, expect_tx);
        if (threadIdx.x == 0) {
            tma_load_2d_fn(&tma_K, mbar_KV_0, smem_K_0, 0, b_idx * H * S + h_idx * S + 0 * 128);
            tma_load_2d_fn(&tma_K, mbar_KV_0, smem_K_0 + 8192, 64, b_idx * H * S + h_idx * S + 0 * 128);
            
            tma_load_2d_fn(&tma_V, mbar_KV_0, smem_V_0, 0, b_idx * H * S + h_idx * S + 0 * 128);
            tma_load_2d_fn(&tma_V, mbar_KV_0, smem_V_0 + 8192, 64, b_idx * H * S + h_idx * S + 0 * 128);
        }
    }

    mbarrier_wait_fn(mbar_Q, 0);

    uint32_t phase_KV[2] = {0};
    int col_step = 32;
    int row = threadIdx.x;

    float rowmax_prev = -1e20f;
    float rowsum_prev = 0.0f;

    uint32_t idesc_QK = make_instr_desc_fn(128, 128);
    uint32_t idesc_PV0 = make_instr_desc_fn_mn_major_b(128, 64);
    uint32_t idesc_PV1 = make_instr_desc_fn_mn_major_b(128, 64);

    for (int step = 0; step < num_steps; step++) {
        int kv = step % 2;
        int next_kv = (step + 1) % 2;

        if (step < num_steps - 1) {
            uint32_t expect_tx_next = 32768; 
            
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(mbar_KV[next_kv], expect_tx_next);
                
                tma_load_2d_fn(&tma_K, mbar_KV[next_kv], smem_K[next_kv], 0, b_idx * H * S + h_idx * S + (step + 1) * 128);
                tma_load_2d_fn(&tma_K, mbar_KV[next_kv], smem_K[next_kv] + 8192, 64, b_idx * H * S + h_idx * S + (step + 1) * 128);
                
                tma_load_2d_fn(&tma_V, mbar_KV[next_kv], smem_V[next_kv], 0, b_idx * H * S + h_idx * S + (step + 1) * 128);
                tma_load_2d_fn(&tma_V, mbar_KV[next_kv], smem_V[next_kv] + 8192, 64, b_idx * H * S + h_idx * S + (step + 1) * 128);
            }
        }

        mbarrier_wait_fn(mbar_KV[kv], phase_KV[kv]);
        phase_KV[kv] ^= 1;

        __nv_bfloat16* smem_K_curr = smem_K[kv];
        __nv_bfloat16* smem_V_curr = smem_V[kv];

        __syncthreads(); 
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_UMA_0, 0);
        }

        for (int k = 0; k < 8; k++) {
            uint64_t desc_Q_k = make_smem_desc_k_major<__nv_bfloat16>(smem_Q + k * 16 * sizeof(__nv_bfloat16));
            uint64_t desc_K_k = make_smem_desc_k_major<__nv_bfloat16>(smem_K_curr + k * 16 * sizeof(__nv_bfloat16));
            umma_f16_cg1_fn(tmem_S, desc_Q_k, desc_K_k, idesc_QK, k == 0 ? 0 : 1);
        }
        
        if (threadIdx.x == 0) {
            umma_commit_1sm_fn(mbar_UMA_0);
        }
        mbarrier_wait_fn(mbar_UMA_0, step % 2);
        
        float rowmax_th = -1e20f;

        for (int i = 0; i < 4; i++) {
            uint32_t r[4];
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"((row << 16) + col_step * i));
            
            for(int k = 0; k < 4; k++) {
                float val = tmem_to_float(r[k]);
                val *= scale_factor; 
                if (val > rowmax_th) {
                    rowmax_th = val;
                }
            }
        }

        float rowsum_th = 0.0f;
        float p_val[128];
        uint32_t p_val_bits[128];

        for (int i = 0; i < 4; i++) {
            uint32_t r[4];
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"((row << 16) + col_step * i));
            
            for(int k = 0; k < 4; k++) {
                int c = col_step * i + k;
                float val = tmem_to_float(r[k]);
                val *= scale_factor;
                p_val[c] = fast_exp2f_fn((val - rowmax_th) * 1.4426950408889634f);
                rowsum_th += p_val[c];
                p_val_bits[c] = to_bits(p_val[c]);
            }
        }

        float correction = fast_exp2f_fn((rowmax_prev - rowmax_th) * 1.4426950408889634f);

        for (int i = 0; i < 4; i++) {
            uint32_t r[4];
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"((row << 16) + col_step * i));
            
            for(int j = 0; j < 4; j++) {
                float val = tmem_to_float(r[j]);
                val *= correction;
                r[j] = to_bits(val);
            }
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                :: "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]) : "r"((row << 16) + col_step * i));
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");

        __syncthreads(); 
        
        for (int k = 0; k < 64; k+=2) {
            float next_p_val = (k + 1 < 64) ? p_val[k+1] : 0.0f;
            uint32_t x_swizzled = ((row % 8) ^ (k / 8)) * 8 + (k % 8);
            uint32_t packed = pack_bf16_fn(p_val_bits[k], to_bits(next_p_val));
            *reinterpret_cast<uint32_t*>(&smem_P[row * 64 + x_swizzled]) = packed;
        }

        for (int k = 0; k < 64; k+=2) {
            float next_p_val = (k + 1 < 64) ? p_val[k + 64] : 0.0f;
            uint32_t x_swizzled = ((row % 8) ^ (k / 8)) * 8 + (k % 8);
            uint32_t packed = pack_bf16_fn(p_val_bits[k + 64], to_bits(next_p_val));
            *reinterpret_cast<uint32_t*>(&smem_P1[row * 64 + x_swizzled]) = packed;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_UMA_1, 0);
        }
        
        for (int k = 0; k < 4; k++) {
            uint64_t desc_P_k = make_smem_desc_k_major<__nv_bfloat16>(smem_P + k * 16 * sizeof(__nv_bfloat16));
            uint64_t desc_V_k = make_smem_desc_mn_major<__nv_bfloat16>(smem_V_curr + k * 1024); 
            umma_f16_cg1_fn(tmem_O, desc_P_k, desc_V_k, idesc_PV0, 1);
        }
        
        for (int k = 0; k < 4; k++) {
            uint64_t desc_P1_k = make_smem_desc_k_major<__nv_bfloat16>(smem_P1 + k * 16 * sizeof(__nv_bfloat16));
            uint64_t desc_V1_k = make_smem_desc_mn_major<__nv_bfloat16>(smem_V_curr + 4096 + k * 1024); 
            umma_f16_cg1_fn(tmem_O, desc_P1_k, desc_V1_k, idesc_PV1, 1);
        }

        if (threadIdx.x == 0) {
            umma_commit_1sm_fn(mbar_UMA_1);
        }
        mbarrier_wait_fn(mbar_UMA_1, step % 2);

        __syncthreads();

        rowsum_prev *= correction;
        rowsum_prev += rowsum_th;
        rowmax_prev = rowmax_th;
    }

    tmem_load_fence_fn();
    __syncthreads();

    uint32_t r_bits[4];
    for (int i = 0; i < 4; i++) {
        uint32_t r[4];
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"((row << 16) + col_step * i));
        
        for(int j = 0; j < 4; j++) {
            r_bits[j] = to_bits(tmem_to_float(r[j]));
        }
        
        for(int j = 0; j < 4; j += 2) {
            float val = tmem_to_float(r[j]);
            if (rowsum_prev > 0.0f) {
                val /= rowsum_prev;
            }
            float next_val = (j + 1 < 4) ? tmem_to_float(r[j+1]) : 0.0f;
            if (rowsum_prev > 0.0f) {
                next_val /= rowsum_prev;
            }
            uint32_t x_swizzled = ((row % 8) ^ (col_step * i / 8)) * 8 + (col_step * i % 8);
            uint32_t packed = pack_bf16_fn(r_bits[j], to_bits(next_val));
            *reinterpret_cast<uint32_t*>(&smem_P[row * 128 + x_swizzled]) = packed;
        }
    }
    __syncthreads();

    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    for (uint32_t step = 0; step < 4; ++step) {
        uint32_t col_start = step * 32;
        uint32_t row = step * 32 + warp_id;
        if (row >= 128) continue;
        
        uint4 data = *reinterpret_cast<uint4*>(&smem_P[row * 128 + col_start]);
        
        uint32_t global_row = m_block + row;
        uint32_t global_col = col_start;
        if (global_row < S && global_col + 15 < 128) {
            uint32_t outer_offset_flat = b_idx * H * S + h_idx * S;
            *reinterpret_cast<uint4*>(O + (uint64_t)(outer_offset_flat + global_row) * 128 + global_col) = data;
        }
    }

    if (lane_id == 0) {
        for (uint32_t step = 0; step < 4; ++step) {
            uint32_t row = step * 32 + warp_id;
            if (row >= 128) continue;
            uint32_t global_row = m_block + row;
            if (global_row < S) {
                uint32_t outer_offset_flat = b_idx * H * S + h_idx * S;
                LSE[outer_offset_flat + global_row] = rowmax_prev * 1.4426950408889634f + __logf(rowsum_prev);
            }
        }
    }
    
    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_S, 128);
        tmem_dealloc_fn(tmem_P, 128);
        tmem_dealloc_fn(tmem_O, 128);
    }
}

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapDataType dataType, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, dataType, 2, globalAddress, globalDim, globalStrides, boxDim,
        elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill
    );
}

namespace tvm_ffi {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3); 
    
    if (D != 128) {
        fprintf(stderr, "Error: Expected head dimension 128, got %ld\n", D);
        exit(1);
    }

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUtensorMap tma_Q, tma_K, tma_V;
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), 128, B * H * S, 64, 128, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), 128, B * H * S, 64, 128, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), 128, B * H * S, 64, 128, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    dim3 grid((S + 127) / 128, FUSED_HEAD * H, B);
    dim3 block(128, 1, 1);
    int smem_size = 192 * 1024 + 256;

    CUDA_CHECK(cudaFuncSetAttribute(fa_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    float scale_factor = 1.0f / sqrtf((float)D);

    fa_kernel<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V,
        static_cast<__nv_bfloat16*>(O.data_ptr()), 
        static_cast<float*>(LSE.data_ptr()),
        S, scale_factor, H, B
    );
    CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi