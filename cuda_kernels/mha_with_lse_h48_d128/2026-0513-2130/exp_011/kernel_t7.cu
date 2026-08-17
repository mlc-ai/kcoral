#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CU_CHECK(call) do {                                        \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        fprintf(stderr, "CU error %d at %s:%d\n",                 \
                _e, __FILE__, __LINE__);                           \
        exit(1);                                                   \
    }                                                              \
} while(0)

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_fn(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])) : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx_fn(uint64_t* bar, uint32_t tx) {
    uint32_t ba = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :: "r"(ba), "r"(tx) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase) : "memory");
}

__device__ __forceinline__ void tma_load_3d_cg1_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t ba = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(sa), "l"((uint64_t)d), "r"(c0), "r"(c1), "r"(c2), "r"(ba) : "memory");
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(a), "r"(ncols) : "memory");
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(addr), "r"(ncols) : "memory");
}

__device__ __forceinline__ void tmem_load_8x_float_fn(uint32_t col, float* r0, float* r1, float* r2, float* r3, float* r4, float* r5, float* r6, float* r7) {
    uint32_t u0, u1, u2, u3, u4, u5, u6, u7;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=r"(u0),"=r"(u1),"=r"(u2),"=r"(u3),"=r"(u4),"=r"(u5),"=r"(u6),"=r"(u7) : "r"(col));
    *r0 = __uint_as_float(u0); *r1 = __uint_as_float(u1); *r2 = __uint_as_float(u2); *r3 = __uint_as_float(u3);
    *r4 = __uint_as_float(u4); *r5 = __uint_as_float(u5); *r6 = __uint_as_float(u6); *r7 = __uint_as_float(u7);
}

__device__ __forceinline__ void tmem_store_4x_fn(uint32_t col, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};"
                 :: "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(col) : "memory");
}

__device__ __forceinline__ void tmem_store_4x_float_fn(uint32_t col, float r0, float r1, float r2, float r3) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};"
                 :: "r"(__float_as_uint(r0)), "r"(__float_as_uint(r1)), "r"(__float_as_uint(r2)), "r"(__float_as_uint(r3)), "r"(col) : "memory");
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_store_fence_fn() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void umma_f16_cg1_tmem_A_fn(
    uint32_t d_tmem, uint32_t a_tmem, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}\n"
        :: "r"(d_tmem), "r"(a_tmem), "l"(desc_b), "r"(idesc), "r"(accum) : "memory");
}

__device__ __forceinline__ void umma_f16_cg1_smem_A_fn(
    uint32_t d_tmem, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(d_tmem), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum) : "memory");
}

__device__ __forceinline__ void umma_commit_cg1_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64 [%0];"
        :: "r"(a) : "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100(void* smem_ptr, uint32_t sbo, uint32_t lbo = 0) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint32_t base_offset = (addr >> 7) & 0x7;
    d |= (uint64_t)((addr >> 4) & 0x3FFF);
    d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16;
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)base_offset << 49;
    d |= (uint64_t)2 << 61; 
    return d;
}

__device__ __forceinline__ uint32_t make_idesc(bool trans_a, bool trans_b, uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);           
    d |= (1u << 7);           
    d |= (1u << 10);          
    if (trans_a) d |= (1u << 15);
    if (trans_b) d |= (1u << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
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

__device__ __forceinline__ void named_barrier_sync_fn(int bar_id, int count) {
    asm volatile("barrier.sync.aligned %0, %1;" :: "r"(bar_id), "r"(count) : "memory");
}

__global__ void __launch_bounds__(160, 1) mha_fwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S,
    int H
) {
    extern __shared__ char smem[];

    __nv_bfloat16* smem_Q0 = (__nv_bfloat16*)smem;
    __nv_bfloat16* smem_Q1 = (__nv_bfloat16*)(smem + 16384);
    
    __nv_bfloat16* smem_K0_0 = (__nv_bfloat16*)(smem + 32768);
    __nv_bfloat16* smem_K1_0 = (__nv_bfloat16*)(smem + 49152);
    __nv_bfloat16* smem_V0_0 = (__nv_bfloat16*)(smem + 65536);
    __nv_bfloat16* smem_V1_0 = (__nv_bfloat16*)(smem + 81920);
    
    __nv_bfloat16* smem_K0_1 = (__nv_bfloat16*)(smem + 98304);
    __nv_bfloat16* smem_K1_1 = (__nv_bfloat16*)(smem + 114688);
    __nv_bfloat16* smem_V0_1 = (__nv_bfloat16*)(smem + 131072);
    __nv_bfloat16* smem_V1_1 = (__nv_bfloat16*)(smem + 147456);
    
    __nv_bfloat16* smem_K0[2] = { smem_K0_0, smem_K0_1 };
    __nv_bfloat16* smem_K1[2] = { smem_K1_0, smem_K1_1 };
    __nv_bfloat16* smem_V0[2] = { smem_V0_0, smem_V0_1 };
    __nv_bfloat16* smem_V1[2] = { smem_V1_0, smem_V1_1 };
    
    uint64_t* mbar_Q = (uint64_t*)(smem + 163840);
    uint64_t* mbar_KV_full = (uint64_t*)(smem + 163848);
    uint64_t* mbar_KV_empty = (uint64_t*)(smem + 163864);
    uint64_t* mbar_MMA_S0 = (uint64_t*)(smem + 163880);
    uint64_t* mbar_MMA_S1 = (uint64_t*)(smem + 163888);
    uint64_t* mbar_MMA_PV = (uint64_t*)(smem + 163896);
    uint32_t* smem_tmem_addr = (uint32_t*)(smem + 163904);
    
    int b = blockIdx.z;
    int h = blockIdx.y;
    int m_block = blockIdx.x;
    
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int tid = threadIdx.x - 32; 
    
    int num_kv_blocks = (S + 127) / 128;
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(&mbar_KV_full[0], 1);
        init_smem_barrier_fn(&mbar_KV_full[1], 1);
        init_smem_barrier_fn(&mbar_KV_empty[0], 1);
        init_smem_barrier_fn(&mbar_KV_empty[1], 1);
        init_smem_barrier_fn(mbar_MMA_S0, 1);
        init_smem_barrier_fn(mbar_MMA_S1, 1);
        init_smem_barrier_fn(mbar_MMA_PV, 1);
        fence_smem_barrier_init_fn();
        
        mbarrier_arrive_fn(&mbar_KV_empty[0]);
        mbarrier_arrive_fn(&mbar_KV_empty[1]);
    }
    __syncthreads();
    
    if (warp_id == 0) {
        int phase_empty[2] = {0, 0};
        int bh = b * H + h;
        
        if (lane_id == 0) {
            mbarrier_arrive_expect_tx_fn(mbar_Q, 32768);
            tma_load_3d_cg1_fn(&tma_Q, mbar_Q, smem_Q0, 0, m_block * 128, bh);
            tma_load_3d_cg1_fn(&tma_Q, mbar_Q, smem_Q1, 64, m_block * 128, bh);
        }
        
        for (int k_idx = 0; k_idx < num_kv_blocks; k_idx++) {
            int stage = k_idx % 2;
            
            mbarrier_wait_fn(&mbar_KV_empty[stage], phase_empty[stage]);
            phase_empty[stage] ^= 1;
            
            if (lane_id == 0) {
                mbarrier_arrive_expect_tx_fn(&mbar_KV_full[stage], 65536);
                tma_load_3d_cg1_fn(&tma_K, &mbar_KV_full[stage], smem_K0[stage], 0, k_idx * 128, bh);
                tma_load_3d_cg1_fn(&tma_K, &mbar_KV_full[stage], smem_K1[stage], 64, k_idx * 128, bh);
                tma_load_3d_cg1_fn(&tma_V, &mbar_KV_full[stage], smem_V0[stage], 0, k_idx * 128, bh);
                tma_load_3d_cg1_fn(&tma_V, &mbar_KV_full[stage], smem_V1[stage], 64, k_idx * 128, bh);
            }
        }
    }
    
    if (warp_id >= 1 && warp_id <= 4) {
        if (warp_id == 1 && lane_id == 0) {
            tmem_alloc_fn(smem_tmem_addr, 512);
        }
        named_barrier_sync_fn(1, 128);
        
        uint32_t base = *smem_tmem_addr;
        uint32_t S_tmem[2] = { base, base + 128 };
        uint32_t P_tmem[2] = { base + 256, base + 320 };
        uint32_t O0_tmem = base + 384;
        uint32_t O1_tmem = base + 448;
        
        uint32_t idesc_S = make_idesc(false, false, 128, 128);
        uint32_t idesc_O_half = make_idesc(false, true, 128, 64);
        
        mbarrier_wait_fn(mbar_Q, 0);
        
        float m_prev = -1e38f;
        float l_prev = 0.0f;
        
        int phase_full[2] = {0, 0};
        int phase_mma_S[2] = {0, 0};
        int phase_mma_PV = 0;
        
        mbarrier_wait_fn(&mbar_KV_full[0], phase_full[0]);
        phase_full[0] ^= 1;
        
        if (warp_id == 1 && lane_id == 0) {
            #pragma unroll 4
            for(int i=0; i<4; i++) {
                uint64_t desc_A0 = make_smem_desc_sm100((char*)smem_Q0 + i*32, 1024, 0);
                uint64_t desc_B0 = make_smem_desc_sm100((char*)smem_K0[0] + i*32, 1024, 0);
                umma_f16_cg1_smem_A_fn(S_tmem[0], desc_A0, desc_B0, idesc_S, (i > 0) ? 1 : 0);
            }
            #pragma unroll 4
            for(int i=0; i<4; i++) {
                uint64_t desc_A1 = make_smem_desc_sm100((char*)smem_Q1 + i*32, 1024, 0);
                uint64_t desc_B1 = make_smem_desc_sm100((char*)smem_K1[0] + i*32, 1024, 0);
                umma_f16_cg1_smem_A_fn(S_tmem[0], desc_A1, desc_B1, idesc_S, 1);
            }
            umma_commit_cg1_fn(mbar_MMA_S0);
        }
        
        for (int k_idx = 0; k_idx < num_kv_blocks; k_idx++) {
            int s_stage = k_idx % 2;
            int next_s_stage = (k_idx + 1) % 2;
            
            uint64_t* cur_mbar_S = (s_stage == 0) ? mbar_MMA_S0 : mbar_MMA_S1;
            uint64_t* next_mbar_S = (s_stage == 0) ? mbar_MMA_S1 : mbar_MMA_S0;
            
            mbarrier_wait_fn(cur_mbar_S, phase_mma_S[s_stage]);
            phase_mma_S[s_stage] ^= 1;
            
            if (k_idx + 1 < num_kv_blocks) {
                mbarrier_wait_fn(&mbar_KV_full[next_s_stage], phase_full[next_s_stage]);
                phase_full[next_s_stage] ^= 1;
                
                if (warp_id == 1 && lane_id == 0) {
                    #pragma unroll 4
                    for(int i=0; i<4; i++) {
                        uint64_t desc_A0 = make_smem_desc_sm100((char*)smem_Q0 + i*32, 1024, 0);
                        uint64_t desc_B0 = make_smem_desc_sm100((char*)smem_K0[next_s_stage] + i*32, 1024, 0);
                        umma_f16_cg1_smem_A_fn(S_tmem[next_s_stage], desc_A0, desc_B0, idesc_S, (i > 0) ? 1 : 0);
                    }
                    #pragma unroll 4
                    for(int i=0; i<4; i++) {
                        uint64_t desc_A1 = make_smem_desc_sm100((char*)smem_Q1 + i*32, 1024, 0);
                        uint64_t desc_B1 = make_smem_desc_sm100((char*)smem_K1[next_s_stage] + i*32, 1024, 0);
                        umma_f16_cg1_smem_A_fn(S_tmem[next_s_stage], desc_A1, desc_B1, idesc_S, 1);
                    }
                    umma_commit_cg1_fn(next_mbar_S);
                }
            }
            
            float row_buf[128];
            #pragma unroll 16
            for(int c=0; c<128; c+=8) {
                tmem_load_8x_float_fn(S_tmem[s_stage] + c, &row_buf[c+0], &row_buf[c+1], &row_buf[c+2], &row_buf[c+3], &row_buf[c+4], &row_buf[c+5], &row_buf[c+6], &row_buf[c+7]);
            }
            tmem_load_fence_fn();
            
            int k_len = S - k_idx * 128;
            int valid_cols = (k_len > 128) ? 128 : (k_len < 0 ? 0 : k_len);
            
            float m_curr = m_prev;
            #pragma unroll 16
            for(int c=0; c<128; c++) {
                float val = row_buf[c] * 0.088388347648f;
                if (c >= valid_cols) val = -1e38f;
                row_buf[c] = val;
                m_curr = fmaxf(m_curr, val);
            }
            
            float scale_prev = fast_exp2f_fn((m_prev - m_curr) * 1.442695040888f);
            float l_curr = l_prev * scale_prev;
            
            #pragma unroll 16
            for(int c=0; c<128; c++) {
                float p = fast_exp2f_fn((row_buf[c] - m_curr) * 1.442695040888f);
                if (c >= valid_cols) p = 0.0f;
                row_buf[c] = p;
                l_curr += p;
            }
            
            #pragma unroll 16
            for(int c=0; c<16; c++) {
                uint32_t r0 = pack_bf16_fn(__float_as_uint(row_buf[c*8+0]), __float_as_uint(row_buf[c*8+1]));
                uint32_t r1 = pack_bf16_fn(__float_as_uint(row_buf[c*8+2]), __float_as_uint(row_buf[c*8+3]));
                uint32_t r2 = pack_bf16_fn(__float_as_uint(row_buf[c*8+4]), __float_as_uint(row_buf[c*8+5]));
                uint32_t r3 = pack_bf16_fn(__float_as_uint(row_buf[c*8+6]), __float_as_uint(row_buf[c*8+7]));
                tmem_store_4x_fn(P_tmem[s_stage] + c * 4, r0, r1, r2, r3);
            }
            tmem_store_fence_fn();
            
            if (k_idx > 0) {
                mbarrier_wait_fn(mbar_MMA_PV, phase_mma_PV);
                phase_mma_PV ^= 1;
            }
            
            if (k_idx > 0 && scale_prev < 1.0f) {
                float o0_buf[64];
                float o1_buf[64];
                #pragma unroll 8
                for(int c=0; c<64; c+=8) {
                    tmem_load_8x_float_fn(O0_tmem + c, &o0_buf[c+0], &o0_buf[c+1], &o0_buf[c+2], &o0_buf[c+3], &o0_buf[c+4], &o0_buf[c+5], &o0_buf[c+6], &o0_buf[c+7]);
                }
                #pragma unroll 8
                for(int c=0; c<64; c+=8) {
                    tmem_load_8x_float_fn(O1_tmem + c, &o1_buf[c+0], &o1_buf[c+1], &o1_buf[c+2], &o1_buf[c+3], &o1_buf[c+4], &o1_buf[c+5], &o1_buf[c+6], &o1_buf[c+7]);
                }
                tmem_load_fence_fn();
                
                #pragma unroll 8
                for(int c=0; c<64; c+=8) {
                    tmem_store_4x_float_fn(O0_tmem + c, o0_buf[c+0]*scale_prev, o0_buf[c+1]*scale_prev, o0_buf[c+2]*scale_prev, o0_buf[c+3]*scale_prev);
                    tmem_store_4x_float_fn(O0_tmem + c + 4, o0_buf[c+4]*scale_prev, o0_buf[c+5]*scale_prev, o0_buf[c+6]*scale_prev, o0_buf[c+7]*scale_prev);
                }
                #pragma unroll 8
                for(int c=0; c<64; c+=8) {
                    tmem_store_4x_float_fn(O1_tmem + c, o1_buf[c+0]*scale_prev, o1_buf[c+1]*scale_prev, o1_buf[c+2]*scale_prev, o1_buf[c+3]*scale_prev);
                    tmem_store_4x_float_fn(O1_tmem + c + 4, o1_buf[c+4]*scale_prev, o1_buf[c+5]*scale_prev, o1_buf[c+6]*scale_prev, o1_buf[c+7]*scale_prev);
                }
                tmem_store_fence_fn();
            }
            
            m_prev = m_curr;
            l_prev = l_curr;
            
            named_barrier_sync_fn(1, 128);
            
            if (warp_id == 1 && lane_id == 0) {
                #pragma unroll 4
                for(int i=0; i<8; i++) {
                    uint32_t a_tmem = P_tmem[s_stage] + i * 8;
                    uint64_t desc_B0 = make_smem_desc_sm100((char*)smem_V0[s_stage] + i * 2048, 1024, 16384);
                    int accum = (k_idx > 0 || i > 0) ? 1 : 0;
                    umma_f16_cg1_tmem_A_fn(O0_tmem, a_tmem, desc_B0, idesc_O_half, accum);
                    
                    uint64_t desc_B1 = make_smem_desc_sm100((char*)smem_V1[s_stage] + i * 2048, 1024, 16384);
                    umma_f16_cg1_tmem_A_fn(O1_tmem, a_tmem, desc_B1, idesc_O_half, accum);
                }
                umma_commit_cg1_fn(mbar_MMA_PV);
            }
            
            if (warp_id == 1 && lane_id == 0) {
                mbarrier_arrive_fn(&mbar_KV_empty[s_stage]);
            }
        }
        
        mbarrier_wait_fn(mbar_MMA_PV, phase_mma_PV);
        
        float inv_l = 1.0f / l_prev;
        float lse = m_prev + logf(l_prev);
        
        if (tid < 128 && m_block * 128 + tid < S) {
            int lse_idx = b * H * S + h * S + m_block * 128 + tid;
            LSE[lse_idx] = lse;
        }
        
        __nv_bfloat16* smem_O = (__nv_bfloat16*)smem_Q0;
        
        float o0_buf[64];
        float o1_buf[64];
        #pragma unroll 8
        for(int c=0; c<64; c+=8) {
            tmem_load_8x_float_fn(O0_tmem + c, &o0_buf[c+0], &o0_buf[c+1], &o0_buf[c+2], &o0_buf[c+3], &o0_buf[c+4], &o0_buf[c+5], &o0_buf[c+6], &o0_buf[c+7]);
        }
        #pragma unroll 8
        for(int c=0; c<64; c+=8) {
            tmem_load_8x_float_fn(O1_tmem + c, &o1_buf[c+0], &o1_buf[c+1], &o1_buf[c+2], &o1_buf[c+3], &o1_buf[c+4], &o1_buf[c+5], &o1_buf[c+6], &o1_buf[c+7]);
        }
        tmem_load_fence_fn();
        
        #pragma unroll 8
        for(int c=0; c<64; c++) {
            smem_O[tid * 128 + c] = __float2bfloat16(o0_buf[c] * inv_l);
            smem_O[tid * 128 + 64 + c] = __float2bfloat16(o1_buf[c] * inv_l);
        }
        
        named_barrier_sync_fn(1, 128);
        
        #pragma unroll 4
        for(int step = 0; step < 32; step++) {
            int idx = step * 512 + tid * 4;
            int row = idx / 128;
            int col = idx % 128;
            if (m_block * 128 + row < S) {
                uint2 val = *(uint2*)(&smem_O[idx]);
                int g_idx = b * H * S * 128 + h * S * 128 + (m_block * 128 + row) * 128 + col;
                *(uint2*)(&O[g_idx]) = val;
            }
        }
        
        named_barrier_sync_fn(1, 128);
        if (warp_id == 1 && lane_id == 0) {
            tmem_dealloc_fn(*smem_tmem_addr, 512);
        }
    }
}

namespace tvm_ffi_mha {

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t D, uint64_t S, uint64_t BH, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[3] = {D, S, BH};
    cuuint64_t globalStrides[2] = {D * 2, S * D * 2};
    cuuint32_t boxDim[3] = {smem_inner_dim, smem_outer_dim, 1};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        3, 
        globalAddress,
        globalDim,
        globalStrides,
        boxDim, 
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int D = Q.size(3);
    
    CUtensorMap tma_Q, tma_K, tma_V;
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_Q, Q.data_ptr(), D, S, B*H, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_K, K.data_ptr(), D, S, B*H, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_V, V.data_ptr(), D, S, B*H, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B));
    
    int num_m_blocks = (S + 127) / 128;
    dim3 grid(num_m_blocks, H, B);
    dim3 block(160);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUDA_CHECK(cudaFuncSetAttribute(mha_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 164 * 1024));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 164 * 1024;
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 1;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, mha_fwd_kernel, tma_Q, tma_K, tma_V, 
                       static_cast<__nv_bfloat16*>(O.data_ptr()), 
                       static_cast<float*>(LSE.data_ptr()), 
                       S, H));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_mha