#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_mha_sm100 {

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

__device__ __forceinline__ void store_smem_swizzle_128B(uint8_t* smem_base, uint32_t row, uint32_t col, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    uint32_t chunk_x = col / 8;
    uint32_t swizzled_chunk_x = (row % 8) ^ chunk_x;
    uint32_t byte_offset = row * 128 + swizzled_chunk_x * 16;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_base + byte_offset);
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
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

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tma_load_4d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}

__device__ __forceinline__ void tmem_alloc_cg1_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_cg1_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ void tmem_load_8x_fn(uint32_t col,
    uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3,
    uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3),
                   "=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7) : "r"(col));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
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

__device__ __forceinline__ void umma_commit_cg1_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(a));
}

__device__ __forceinline__ uint64_t make_smem_desc_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo, uint32_t swizzle_mode) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)((addr >> 4) & 0x3FFF);
    d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16;
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46; 
    
    uint64_t base_offset = (addr >> 7) & 0x7;
    d |= (base_offset << 49);
    
    d |= (uint64_t)swizzle_mode << 61;
    return d;
}

__device__ __forceinline__ uint64_t get_smem_desc_k_adv(uint64_t base_desc, uint32_t addr_adv) {
    uint32_t orig_addr = (base_desc & 0x3FFF) << 4;
    uint32_t new_addr = orig_addr + addr_adv;
    uint64_t d = base_desc;
    d &= ~((uint64_t)0x3FFF);
    d &= ~((uint64_t)0x7 << 49);
    d |= (uint64_t)((new_addr >> 4) & 0x3FFF);
    uint64_t base_offset = (new_addr >> 7) & 0x7;
    d |= (base_offset << 49);
    return d;
}

__device__ __forceinline__ uint64_t get_smem_desc_k_adv_MN(uint64_t base_desc, uint32_t addr_adv) {
    uint32_t orig_addr = (base_desc & 0x3FFF) << 4;
    uint32_t new_addr = orig_addr + addr_adv;
    uint64_t d = base_desc;
    d &= ~((uint64_t)0x3FFF);
    d |= (uint64_t)((new_addr >> 4) & 0x3FFF);
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, bool transpose_b = false) {
    uint32_t d = 0;
    d |= (1u << 4);           
    d |= (1u << 7);           
    d |= (1u << 10);          
    if (transpose_b) d |= (1u << 16); 
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

struct SharedStorage {
    __align__(128) uint8_t Q[32768];
    __align__(128) uint8_t P[32768];
    __align__(128) uint8_t K[2][32768];
    __align__(128) uint8_t V[2][32768];
    __align__(8) uint64_t mbar_Q;
    __align__(8) uint64_t mbar_K[2];
    __align__(8) uint64_t mbar_V[2];
    __align__(8) uint64_t mbar_mma;
    __align__(4) uint32_t tmem_base;
};

__global__ void __launch_bounds__(128, 1) mha_fwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* __restrict__ O_out, float* __restrict__ LSE_out,
    int B, int H, int S) 
{
    setmaxnreg_inc_sync_fn<240>();

    int b_idx = blockIdx.z;
    int h_idx = blockIdx.y;
    int m_block = blockIdx.x * 128;
    if (m_block >= S) return;
    
    int tid = threadIdx.x;

    extern __shared__ uint8_t smem_dynamic[];
    uintptr_t smem_ptr = (uintptr_t)smem_dynamic;
    smem_ptr = (smem_ptr + 127) & ~127;
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(smem_ptr);
    
    if (tid == 0) {
        init_smem_barrier_fn(&smem.mbar_Q, 1);
        init_smem_barrier_fn(&smem.mbar_K[0], 1);
        init_smem_barrier_fn(&smem.mbar_K[1], 1);
        init_smem_barrier_fn(&smem.mbar_V[0], 1);
        init_smem_barrier_fn(&smem.mbar_V[1], 1);
        init_smem_barrier_fn(&smem.mbar_mma, 1);
    }
    
    // tmem.alloc strictly expects Warp instruction uniform semantics
    if (tid < 32) {
        tmem_alloc_cg1_fn(&smem.tmem_base, 256);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();
    
    uint32_t tmem_base = smem.tmem_base;
    uint32_t tmem_S = tmem_base;
    uint32_t tmem_O0 = tmem_base + 128;
    uint32_t tmem_O1 = tmem_base + 192;

    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(&smem.mbar_Q, 32768);
        tma_load_4d_fn(&tma_Q, &smem.mbar_Q, smem.Q, 0, m_block, h_idx, b_idx);
        tma_load_4d_fn(&tma_Q, &smem.mbar_Q, smem.Q + 16384, 64, m_block, h_idx, b_idx);

        mbarrier_arrive_and_expect_tx_fn(&smem.mbar_K[0], 32768);
        tma_load_4d_fn(&tma_K, &smem.mbar_K[0], smem.K[0], 0, 0, h_idx, b_idx);
        tma_load_4d_fn(&tma_K, &smem.mbar_K[0], smem.K[0] + 16384, 64, 0, h_idx, b_idx);

        mbarrier_arrive_and_expect_tx_fn(&smem.mbar_V[0], 32768);
        tma_load_4d_fn(&tma_V, &smem.mbar_V[0], smem.V[0], 0, 0, h_idx, b_idx);
        tma_load_4d_fn(&tma_V, &smem.mbar_V[0], smem.V[0] + 16384, 64, 0, h_idx, b_idx);
    }

    float m_val = -INFINITY;
    float l_val = 0.0f;
    float O_reg[128];
    #pragma unroll
    for (int i=0; i<128; ++i) O_reg[i] = 0.0f;

    mbarrier_wait_fn(&smem.mbar_Q, 0);

    uint64_t desc_Q0 = make_smem_desc_fn(smem.Q, 0, 1024, 2);
    uint64_t desc_Q1 = make_smem_desc_fn(smem.Q + 16384, 0, 1024, 2);

    int pp = 0;
    int phase_K[2] = {0, 0};
    int phase_V[2] = {0, 0};
    int phase_mma = 0;
    uint32_t idesc_QK = make_instr_desc_fn(128, 128, false);
    uint32_t idesc_PV = make_instr_desc_fn(128, 64, true); 

    for (int n_block = 0; n_block < S; n_block += 128) {
        int next_n = n_block + 128;
        int next_pp = pp ^ 1;

        if (next_n < S) {
            if (tid == 0) {
                uint8_t* pK = (uint8_t*)smem.K[next_pp];
                uint8_t* pV = (uint8_t*)smem.V[next_pp];
                
                mbarrier_arrive_and_expect_tx_fn(&smem.mbar_K[next_pp], 32768);
                tma_load_4d_fn(&tma_K, &smem.mbar_K[next_pp], pK, 0, next_n, h_idx, b_idx);
                tma_load_4d_fn(&tma_K, &smem.mbar_K[next_pp], pK + 16384, 64, next_n, h_idx, b_idx);
                
                mbarrier_arrive_and_expect_tx_fn(&smem.mbar_V[next_pp], 32768);
                tma_load_4d_fn(&tma_V, &smem.mbar_V[next_pp], pV, 0, next_n, h_idx, b_idx);
                tma_load_4d_fn(&tma_V, &smem.mbar_V[next_pp], pV + 16384, 64, next_n, h_idx, b_idx);
            }
        }

        mbarrier_wait_fn(&smem.mbar_K[pp], phase_K[pp]);
        phase_K[pp] ^= 1;
        fence_async_shared_fn();
        
        uint8_t* pK_cur = (uint8_t*)smem.K[pp];
        uint64_t desc_K0 = make_smem_desc_fn(pK_cur, 0, 1024, 2);
        uint64_t desc_K1 = make_smem_desc_fn(pK_cur + 16384, 0, 1024, 2);

        if (tid == 0) {
            for (int k = 0; k < 64; k += 16) {
                uint64_t a = get_smem_desc_k_adv(desc_Q0, k * 2);
                uint64_t b = get_smem_desc_k_adv(desc_K0, k * 2);
                umma_f16_cg1_fn(tmem_S, a, b, idesc_QK, k > 0);
            }
            for (int k = 0; k < 64; k += 16) {
                uint64_t a = get_smem_desc_k_adv(desc_Q1, k * 2);
                uint64_t b = get_smem_desc_k_adv(desc_K1, k * 2);
                umma_f16_cg1_fn(tmem_S, a, b, idesc_QK, 1);
            }
            umma_commit_cg1_fn(&smem.mbar_mma); 
        }
        
        mbarrier_wait_fn(&smem.mbar_mma, phase_mma);
        phase_mma ^= 1;

        int valid_n = S - n_block;
        float m_new = m_val;
        
        for (int i = 0; i < 128; i += 8) {
            uint32_t r[8];
            tmem_load_8x_fn(tmem_S + i, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
            tmem_load_fence_fn();
            for (int j=0; j<8; ++j) {
                // Correct scaling log2(e) / sqrt(128) precise to BF16 emulation
                float val = __uint_as_float(r[j]) * 0.12751732644268685f; 
                if (i + j >= valid_n) val = -INFINITY;
                m_new = max(m_new, val);
            }
        }
        
        float scale_O = fast_exp2f_fn(m_val - m_new);
        if (scale_O < 1.0f) {
            for (int i = 0; i < 128; i++) O_reg[i] *= scale_O;
            l_val *= scale_O;
            m_val = m_new;
        }

        float l_new = 0.0f;
        for (int i = 0; i < 128; i += 8) {
            uint32_t r[8];
            tmem_load_8x_fn(tmem_S + i, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
            tmem_load_fence_fn();
            
            float p[8];
            for (int j = 0; j < 8; ++j) {
                float val = __uint_as_float(r[j]) * 0.12751732644268685f;
                if (i + j >= valid_n) val = -INFINITY;
                p[j] = fast_exp2f_fn(val - m_val);
                l_new += p[j];
            }
            
            uint32_t packed0 = pack_bf16_fn(__float_as_uint(p[0]), __float_as_uint(p[1]));
            uint32_t packed1 = pack_bf16_fn(__float_as_uint(p[2]), __float_as_uint(p[3]));
            uint32_t packed2 = pack_bf16_fn(__float_as_uint(p[4]), __float_as_uint(p[5]));
            uint32_t packed3 = pack_bf16_fn(__float_as_uint(p[6]), __float_as_uint(p[7]));
            
            if (i < 64) {
                store_smem_swizzle_128B(smem.P, tid, i, packed0, packed1, packed2, packed3);
            } else {
                store_smem_swizzle_128B(smem.P + 16384, tid, i - 64, packed0, packed1, packed2, packed3);
            }
        }
        l_val += l_new;

        mbarrier_wait_fn(&smem.mbar_V[pp], phase_V[pp]);
        phase_V[pp] ^= 1;
        
        __syncthreads(); 
        fence_async_shared_fn();

        uint8_t* pV_cur = (uint8_t*)smem.V[pp];
        uint64_t desc_P0 = make_smem_desc_fn(smem.P, 0, 1024, 2);
        uint64_t desc_P1 = make_smem_desc_fn(smem.P + 16384, 0, 1024, 2);
        uint64_t desc_V0 = make_smem_desc_fn(pV_cur, 16384, 1024, 2);
        uint64_t desc_V1 = make_smem_desc_fn(pV_cur + 16384, 16384, 1024, 2);

        if (tid == 0) {
            for (int k = 0; k < 64; k += 16) {
                uint64_t a = get_smem_desc_k_adv(desc_P0, k * 2);
                uint64_t b = get_smem_desc_k_adv_MN(desc_V0, k * 128); 
                umma_f16_cg1_fn(tmem_O0, a, b, idesc_PV, k > 0);
            }
            for (int k = 0; k < 64; k += 16) {
                uint64_t a = get_smem_desc_k_adv(desc_P1, k * 2);
                uint64_t b = get_smem_desc_k_adv_MN(desc_V0, (k + 64) * 128); 
                umma_f16_cg1_fn(tmem_O0, a, b, idesc_PV, 1);
            }
            
            for (int k = 0; k < 64; k += 16) {
                uint64_t a = get_smem_desc_k_adv(desc_P0, k * 2);
                uint64_t b = get_smem_desc_k_adv_MN(desc_V1, k * 128);
                umma_f16_cg1_fn(tmem_O1, a, b, idesc_PV, k > 0); 
            }
            for (int k = 0; k < 64; k += 16) {
                uint64_t a = get_smem_desc_k_adv(desc_P1, k * 2);
                uint64_t b = get_smem_desc_k_adv_MN(desc_V1, (k + 64) * 128);
                umma_f16_cg1_fn(tmem_O1, a, b, idesc_PV, 1);
            }
            umma_commit_cg1_fn(&smem.mbar_mma);
        }

        mbarrier_wait_fn(&smem.mbar_mma, phase_mma);
        phase_mma ^= 1;

        for (int i = 0; i < 64; i += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_O0 + i, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            O_reg[i + 0] += __uint_as_float(r0);
            O_reg[i + 1] += __uint_as_float(r1);
            O_reg[i + 2] += __uint_as_float(r2);
            O_reg[i + 3] += __uint_as_float(r3);
        }
        for (int i = 0; i < 64; i += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_O1 + i, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            O_reg[64 + i + 0] += __uint_as_float(r0);
            O_reg[64 + i + 1] += __uint_as_float(r1);
            O_reg[64 + i + 2] += __uint_as_float(r2);
            O_reg[64 + i + 3] += __uint_as_float(r3);
        }

        // Synchronization point needed ensuring TMEM / Registers finish unloading before Next Iteration MMA 
        // overwrites TMEM bounds identically via `umma_f16_cg1_fn(tmem_S, a, b, idesc_QK, k > 0)`.
        __syncthreads();
        pp = next_pp;
    }

    if (m_block + tid < S) {
        float inv_l = 1.0f / l_val;
        
        __syncthreads();
        __nv_bfloat16* smem_out = (__nv_bfloat16*)smem.Q; 
        for (int i = 0; i < 128; i++) {
            smem_out[tid * 128 + i] = __float2bfloat16(O_reg[i] * inv_l);
        }
        __syncthreads();
        
        uint32_t warp_id = tid / 32;
        uint32_t lane_id = tid % 32;
        uint32_t num_steps = 128 / 4; 
        for (uint32_t step = 0; step < num_steps; ++step) {
            uint32_t row = step * 4 + warp_id;
            if (m_block + row >= S) continue;
            uint32_t col_start = lane_id * 4; 
            uint64_t global_offset = (uint64_t)b_idx * (H * S * 128) + (uint64_t)h_idx * (S * 128) + (uint64_t)(m_block + row) * 128 + col_start;
            uint2 data = *reinterpret_cast<uint2*>(&smem_out[row * 128 + col_start]);
            *reinterpret_cast<uint2*>(&O_out[global_offset]) = data;
        }

        float lse = (m_val + log2f(l_val)) * 0.6931471805599453f; 
        uint64_t lse_offset = (uint64_t)b_idx * (H * S) + (uint64_t)h_idx * S + (m_block + tid);
        LSE_out[lse_offset] = lse;
    }

    __syncthreads();
    if (tid < 32) {
        tmem_dealloc_cg1_fn(tmem_base, 256);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {

    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);

    CUtensorMap tma_Q, tma_K, tma_V;

    cuuint64_t globalDim[4] = { (cuuint64_t)D, (cuuint64_t)S, (cuuint64_t)H, (cuuint64_t)B };
    cuuint64_t globalStrides[3] = { (cuuint64_t)(D * 2), (cuuint64_t)(S * D * 2), (cuuint64_t)(H * S * D * 2) };
    cuuint32_t boxDim[4] = { 64, 128, 1, 1 }; 
    cuuint32_t elementStrides[4] = { 1, 1, 1, 1 };

    cuTensorMapEncodeTiled(&tma_Q, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, Q.data_ptr(),
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    cuTensorMapEncodeTiled(&tma_K, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, K.data_ptr(),
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    cuTensorMapEncodeTiled(&tma_V, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, V.data_ptr(),
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    int blocks_x = (S + 127) / 128;
    dim3 grid(blocks_x, H, B);
    dim3 block(128, 1, 1);

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int smem_size = sizeof(SharedStorage) + 128;
    cudaFuncSetAttribute((void*)mha_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    mha_fwd_kernel<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V,
        static_cast<__nv_bfloat16*>(O.data_ptr()),
        static_cast<float*>(LSE.data_ptr()),
        B, H, S
    );
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_sm100::run);

} // namespace tvm_ffi_mha_sm100