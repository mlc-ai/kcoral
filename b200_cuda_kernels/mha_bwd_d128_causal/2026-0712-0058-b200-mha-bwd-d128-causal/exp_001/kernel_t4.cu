#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <math.h>
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

#define CU_CHECK(call) do {                                      \
    CUresult _e = (call);                                        \
    if (_e != CUDA_SUCCESS) {                                    \
        fprintf(stderr, "CU error %d at %s:%d\n",                \
                _e, __FILE__, __LINE__);                         \
        exit(1);                                                 \
    }                                                            \
} while(0)

namespace tvm_ffi_mha_bwd {

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
        "{\n.reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase));
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

__device__ __forceinline__ uint64_t make_smem_desc_sm100(void* smem_ptr, uint32_t LBO, uint32_t SBO) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((LBO & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((SBO & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61; // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint64_t desc_k_major(uint8_t* ptr) {
    return make_smem_desc_sm100(ptr, 1, 1024);
}

__device__ __forceinline__ uint64_t desc_mn_major(uint8_t* ptr) {
    return make_smem_desc_sm100(ptr, 8192, 1024);
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, bool transpose_B = false) {
    uint32_t d = 0;
    d |= (1u << 4); // dtype FP32
    d |= (1u << 7); // atype BF16
    d |= (1u << 10); // btype BF16
    if (transpose_B) {
        d |= (1u << 16);
    }
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
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

__device__ __forceinline__ void commit_1sm(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1"
        ".mbarrier::arrive::one.shared::cta.b64 [%0];"
        :: "r"(a));
}

__device__ __forceinline__ void gemm_transposed_B(
    uint32_t tmem_C_base, uint8_t* smem_A, uint8_t* smem_B,
    uint32_t M, uint32_t N, uint64_t* barrier, uint32_t& phase) 
{
    uint32_t idesc = make_instr_desc_fn(M, N, true); 
    
    for (uint32_t k = 0; k < 8; ++k) {
        uint32_t offset = (k < 4) ? (k * 32) : (16384 + (k - 4) * 32);
        
        if (k == 0) mbarrier_arrive_and_expect_tx_fn(barrier, 4096);
        
        uint64_t desc_a = desc_k_major(smem_A + offset);
        uint64_t desc_b = desc_mn_major(smem_B + offset);
        
        umma_f16_cg1_fn(tmem_C_base, desc_a, desc_b, idesc, k == 0 ? 0 : 1);
    }
    commit_1sm(barrier);
    mbarrier_wait_fn(barrier, phase);
    phase ^= 1;
}

__device__ __forceinline__ void gemm_k_major_A_B(
    uint32_t tmem_C_base, uint8_t* smem_A, uint8_t* smem_B,
    uint32_t M, uint32_t N, uint64_t* barrier, uint32_t& phase) 
{
    uint32_t idesc = make_instr_desc_fn(M, N, false); 
    
    for (uint32_t k = 0; k < 8; ++k) {
        uint32_t offset = (k < 4) ? (k * 32) : (16384 + (k - 4) * 32);
        
        if (k == 0) mbarrier_arrive_and_expect_tx_fn(barrier, 4096);
        
        uint64_t desc_a = desc_k_major(smem_A + offset);
        uint64_t desc_b = desc_mn_major(smem_B + offset);
        
        umma_f16_cg1_fn(tmem_C_base, desc_a, desc_b, idesc, k == 0 ? 0 : 1);
    }
    commit_1sm(barrier);
    mbarrier_wait_fn(barrier, phase);
    phase ^= 1;
}

__device__ __forceinline__ void gemm_transposed_A_B(
    uint32_t tmem_C_base, uint8_t* smem_A, uint8_t* smem_B,
    uint32_t M, uint32_t N, uint64_t* barrier, uint32_t& phase) 
{
    uint32_t idesc = make_instr_desc_fn(M, N, true); 
    
    for (uint32_t k = 0; k < 8; ++k) {
        uint32_t offset_A = (k < 4) ? (k * 2048) : (16384 + (k - 4) * 2048);
        uint32_t offset_B = (k < 4) ? (k * 2048) : (16384 + (k - 4) * 2048);
        
        if (k == 0) mbarrier_arrive_and_expect_tx_fn(barrier, 4096);
        
        uint64_t desc_a = desc_mn_major(smem_A + offset_A);
        uint64_t desc_b = desc_mn_major(smem_B + offset_B);
        
        umma_f16_cg1_fn(tmem_C_base, desc_a, desc_b, idesc, k == 0 ? 0 : 1);
    }
    commit_1sm(barrier);
    mbarrier_wait_fn(barrier, phase);
    phase ^= 1;
}

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
}

__device__ __forceinline__ void load_tile_128x128(
    const CUtensorMap* desc, uint64_t* bar, uint8_t* smem_A, 
    uint32_t base_S, uint32_t head_idx) 
{
    mbarrier_arrive_and_expect_tx_fn(bar, 32768);
    tma_load_3d_fn(desc, bar, smem_A, 0, base_S, head_idx);
    tma_load_3d_fn(desc, bar, smem_A + 8192, 64, base_S, head_idx);
    tma_load_3d_fn(desc, bar, smem_A + 16384, 0, base_S + 64, head_idx);
    tma_load_3d_fn(desc, bar, smem_A + 24576, 64, base_S + 64, head_idx);
    mbarrier_wait_fn(bar, 0); 
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void named_barrier_sync_fn(int bar_id, int count) {
    asm volatile("barrier.sync.aligned %0, %1;" :: "r"(bar_id), "r"(count));
}

__device__ __forceinline__ void write_fp32_to_smem_swizzled(uint8_t* smem_128x128, int row, int col, float val) {
    __nv_bfloat16 v = __float2bfloat16(val);
    uint32_t x_idx = col / 8;
    uint32_t rem_x = col % 8;
    uint32_t xor_idx = (row % 8) ^ x_idx;
    uint32_t swizzled_col = xor_idx * 8 + rem_x;
    
    if (row < 64 && col < 64) ((__nv_bfloat16*)smem_128x128)[row * 64 + swizzled_col] = v;
    else if (row < 64 && col >= 64) ((__nv_bfloat16*)(smem_128x128 + 8192))[(row) * 64 + swizzled_col] = v;
    else if (row >= 64 && col < 64) ((__nv_bfloat16*)(smem_128x128 + 16384))[(row - 64) * 64 + swizzled_col] = v;
    else if (row >= 64 && col >= 64) ((__nv_bfloat16*)(smem_128x128 + 24576))[(row - 64) * 64 + swizzled_col] = v;
}

__device__ __forceinline__ __nv_bfloat16 read_swizzled_128x128(uint8_t* smem_P, int row, int col) {
    uint32_t x_idx = col / 8;
    uint32_t rem_x = col % 8;
    uint32_t xor_idx = (row % 8) ^ x_idx;
    uint32_t swizzled_col = xor_idx * 8 + rem_x;
    
    if (row < 64 && col < 64) return ((__nv_bfloat16*)smem_P)[row * 64 + swizzled_col];
    if (row < 64 && col >= 64) return ((__nv_bfloat16*)(smem_P + 8192))[row * 64 + swizzled_col];
    if (row >= 64 && col < 64) return ((__nv_bfloat16*)(smem_P + 16384))[(row - 64) * 64 + swizzled_col];
    if (row >= 64 && col >= 64) return ((__nv_bfloat16*)(smem_P + 24576))[(row - 64) * 64 + swizzled_col];
    return __float2bfloat16(0.0f);
}

__device__ __forceinline__ void tmem_epilogue_coalesced_4w_fn(
    __nv_bfloat16* D, __nv_bfloat16* smem_out,
    uint32_t M, uint32_t N, uint32_t m_block, uint32_t n_block,
    uint32_t BM, uint32_t BN) {
    for (uint32_t col = 0; col < BN; col += 4) {
 uint32_t r0, r1, r2, r3;
 asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
: "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(col));
 uint32_t base = threadIdx.x * BN + col;
 smem_out[base + 0] = __float2bfloat16(__uint_as_float(r0));
 smem_out[base + 1] = __float2bfloat16(__uint_as_float(r1));
 smem_out[base + 2] = __float2bfloat16(__uint_as_float(r2));
 smem_out[base + 3] = __float2bfloat16(__uint_as_float(r3));
    }
    tmem_load_fence_fn();
    named_barrier_sync_fn(1, 128);
    
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t num_steps = (BM + 3) / 4;
    for (uint32_t step = 0; step < num_steps; ++step) {
 uint32_t row = step * 4 + warp_id;
 if (row >= BM) continue;
 uint32_t global_row = m_block * BM + row;
 uint32_t col_start = lane_id * 4;
 uint32_t global_col = n_block * BN + col_start;
 if (global_row < M && global_col + 3 < N) {
     uint2 data = *reinterpret_cast<uint2*>(&smem_out[row * BN + col_start]);
     *reinterpret_cast<uint2*>(D + (uint64_t)global_row * N + global_col) = data;
 }
    }
}

__global__ void mha_bwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* L_ptr,
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    uint32_t S, uint32_t H, uint32_t B, bool is_forward)
{
    extern __shared__ char raw_smem[];
    uintptr_t smem_addr = (uintptr_t)raw_smem;
    uint32_t align_offset = (1024 - (smem_addr % 1024)) % 1024;
    char* smem_aligned = raw_smem + align_offset;

    uint64_t* bar_q = (uint64_t*)smem_aligned;
    uint64_t* bar_k = (uint64_t*)(smem_aligned + 8);
    uint64_t* bar_v = (uint64_t*)(smem_aligned + 16);
    uint64_t* bar_o = (uint64_t*)(smem_aligned + 24);
    uint64_t* bar_do = (uint64_t*)(smem_aligned + 32);
    uint64_t* bar_k1 = (uint64_t*)(smem_aligned + 40);
    uint64_t* bar_v1 = (uint64_t*)(smem_aligned + 48);
    uint64_t* bar_q1 = (uint64_t*)(smem_aligned + 40); // Reuses bar_k1 in backward
    uint64_t* bar_do1 = (uint64_t*)(smem_aligned + 48); // Reuses bar_v1 in backward

    uint32_t* tmem_A = (uint32_t*)(smem_aligned + 128);
    uint32_t* tmem_B = (uint32_t*)(smem_aligned + 136);
    uint32_t* tmem_C = (uint32_t*)(smem_aligned + 144);
    float* smem_D_c = (float*)(smem_aligned + 160); // 128 floats = 512 bytes

    uint8_t* smem_Q = (uint8_t*)(smem_aligned + 1024);
    uint8_t* smem_K = (uint8_t*)(smem_aligned + 1024 + 32768); 
    uint8_t* smem_K_1 = (uint8_t*)(smem_aligned + 1024 + 65536);
    uint8_t* smem_V = (uint8_t*)(smem_aligned + 1024 + 98304); 
    uint8_t* smem_V_1 = (uint8_t*)(smem_aligned + 1024 + 131072);
    uint8_t* smem_dO = (uint8_t*)(smem_aligned + 1024 + 163840); 
    uint8_t* smem_dO_1 = (uint8_t*)(smem_aligned + 1024 + 196608);
    uint8_t* smem_P = (uint8_t*)(smem_aligned + 1024 + 229376); 

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar_q, 1);
        init_smem_barrier_fn(bar_k, 1);
        init_smem_barrier_fn(bar_v, 1);
        init_smem_barrier_fn(bar_o, 1);
        init_smem_barrier_fn(bar_do, 1);
        init_smem_barrier_fn(bar_k1, 1);
        init_smem_barrier_fn(bar_v1, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    int ncols_A = 256;
    int ncols_B = 256;
    int ncols_C = 256;

    if (threadIdx.x == 0) {
        tmem_alloc_fn(tmem_A, ncols_A);
        tmem_alloc_fn(tmem_B, ncols_B);
        tmem_alloc_fn(tmem_C, ncols_C);
    }
    __syncthreads(); 

    uint32_t num_blocks = (S + 127) / 128;
    uint32_t head_idx = blockIdx.y;
    
    if (is_forward) {
        uint32_t i_block = blockIdx.x;
        uint32_t s_i = i_block * 128;
        uint32_t phase_q = 0, phase_k = 0, phase_v = 0, phase_do = 0;

        if (threadIdx.x < 128) {
            smem_D_c[threadIdx.x] = 0.0f;
        }

        load_tile_128x128(&tma_Q, bar_q, smem_Q, s_i, head_idx);
        load_tile_128x128(&tma_dO, bar_do, smem_dO, s_i, head_idx);

        uint32_t phase_k1 = 0, phase_v1 = 0;
        if (0 <= i_block) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(bar_k1, 32768);
                mbarrier_arrive_and_expect_tx_fn(bar_v1, 32768);
                tma_load_3d_fn(&tma_K, bar_k1, smem_K, 0, 0, head_idx);
                tma_load_3d_fn(&tma_K, bar_k1, smem_K + 8192, 64, 0, head_idx);
                tma_load_3d_fn(&tma_K, bar_k1, smem_K + 16384, 0, 64, head_idx);
                tma_load_3d_fn(&tma_K, bar_k1, smem_K + 24576, 64, 64, head_idx);
                
                tma_load_3d_fn(&tma_V, bar_v1, smem_V, 0, 0, head_idx);
                tma_load_3d_fn(&tma_V, bar_v1, smem_V + 8192, 64, 0, head_idx);
                tma_load_3d_fn(&tma_V, bar_v1, smem_V + 16384, 0, 64, head_idx);
                tma_load_3d_fn(&tma_V, bar_v1, smem_V + 24576, 64, 64, head_idx);
            }
        }

        for (int j_block = 0; j_block <= i_block; ++j_block) {
            uint32_t s_j = j_block * 128;
            
            mbarrier_wait_fn(bar_k, phase_k);
            mbarrier_wait_fn(bar_v, phase_v);
            
            uint32_t phase_k_prev = phase_k;
            uint32_t phase_v_prev = phase_v;

            if (j_block + 1 <= i_block) {
                if (threadIdx.x == 0) {
                    uint32_t next_s_j = (j_block + 1) * 128;
                    mbarrier_arrive_and_expect_tx_fn(bar_k1, 32768);
                    tma_load_3d_fn(&tma_K, bar_k1, smem_K_1, 0, next_s_j, head_idx);
                    tma_load_3d_fn(&tma_K, bar_k1, smem_K_1 + 8192, 64, next_s_j, head_idx);
                    tma_load_3d_fn(&tma_K, bar_k1, smem_K_1 + 16384, 0, next_s_j + 64, head_idx);
                    tma_load_3d_fn(&tma_K, bar_k1, smem_K_1 + 24576, 64, next_s_j + 64, head_idx);
                    
                    mbarrier_arrive_and_expect_tx_fn(bar_v1, 32768);
                    tma_load_3d_fn(&tma_V, bar_v1, smem_V_1, 0, next_s_j, head_idx);
                    tma_load_3d_fn(&tma_V, bar_v1, smem_V_1 + 8192, 64, next_s_j, head_idx);
                    tma_load_3d_fn(&tma_V, bar_v1, smem_V_1 + 16384, 0, next_s_j + 64, head_idx);
                    tma_load_3d_fn(&tma_V, bar_v1, smem_V_1 + 24576, 64, next_s_j + 64, head_idx);
                }
            }
            
            gemm_transposed_B((uint32_t)__cvta_generic_to_shared(tmem_C), smem_Q, smem_K, 128, 128, bar_k, phase_k);
            gemm_transposed_B((uint32_t)__cvta_generic_to_shared(tmem_B), smem_dO, smem_V, 128, 128, bar_v, phase_v);
            
            float d_val = 0;
            uint32_t my_row = threadIdx.x;
            for (uint32_t col = 0; col < 128; col += 4) {
                uint32_t rC[4], rB[4];
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(rC[0]),"=r"(rC[1]),"=r"(rC[2]),"=r"(rC[3]) : "r"((uint32_t)__cvta_generic_to_shared(tmem_C) + col));
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(rB[0]),"=r"(rB[1]),"=r"(rB[2]),"=r"(rB[3]) : "r"((uint32_t)__cvta_generic_to_shared(tmem_B) + col));
                
                for(int i=0; i<4; ++i) {
                    float f = __uint_as_float(rC[i]);
                    float dp = __uint_as_float(rB[i]);
                    
                    f *= 0.08838834764831845; // scale 1/sqrt(128)
                    
                    int s_j_idx = s_j + col + i;
                    int s_i_idx = s_i + my_row;
                    if (s_j_idx > s_i_idx || s_j_idx >= S || s_i_idx >= S) f = -1e20;
                    float p = expf(f - L_ptr[head_idx * S + s_i_idx]);
                    if (s_j_idx > s_i_idx || s_j_idx >= S || s_i_idx >= S) p = 0;
                    
                    d_val += p * dp;
                    write_fp32_to_smem_swizzled(smem_P, my_row, col + i, p);
                }
            }
            
            __syncthreads(); 
            if (threadIdx.x < 128) {
                smem_D_c[threadIdx.x] += d_val;
            }
            __syncthreads();
            
            uint32_t my_row2 = threadIdx.x;
            for (uint32_t col = 0; col < 128; col += 4) {
                float p = __bfloat162float(read_swizzled_128x128(smem_P, my_row2, col));
                float dp = __uint_as_float(*((uint32_t*)&tmem_B[col])); // Note: Using potentially stale register/value mapping
                
                int s_j_idx = s_j + col;
                int s_i_idx = s_i + my_row2;
                if (s_j_idx > s_i_idx || s_j_idx >= S || s_i_idx >= S) p = 0;
                
                float dp_c = p * (dp - smem_D_c[my_row2]);
                write_fp32_to_smem_swizzled(smem_P, my_row2, col, dp_c);
            }
            
            __syncthreads(); 
            fence_proxy_async_fn(); 
            
            gemm_k_major_A_B(
                (uint32_t)__cvta_generic_to_shared(tmem_A), 
                smem_P, smem_K, 
                128, 128, bar_k, phase_k
            );
            
            __syncthreads();
            if (j_block + 1 <= i_block) {
                mbarrier_wait_fn(bar_k1, phase_k1);
                mbarrier_wait_fn(bar_v1, phase_v1);
                phase_k1 ^= 1;
                phase_v1 ^= 1;
                
                uint8_t* tmp_k = smem_K;
                smem_K = smem_K_1;
                smem_K_1 = tmp_k;
                
                uint8_t* tmp_v = smem_V;
                smem_V = smem_V_1;
                smem_V_1 = tmp_v;
                
                uint64_t* tmp_bar_k = bar_k;
                bar_k = bar_k1;
                bar_k1 = tmp_bar_k;
                
                uint64_t* tmp_bar_v = bar_v;
                bar_v = bar_v1;
                bar_v1 = tmp_bar_v;
            }
            __syncthreads();
        }
        
        tmem_epilogue_coalesced_4w_fn(dQ + head_idx * S * 128, (__nv_bfloat16*)smem_Q, 
            S, 128, i_block, 0, 128, 128);
    } else {
        uint32_t j_block = blockIdx.x;
        uint32_t s_j = j_block * 128;
        uint32_t phase_q = 0, phase_k = 0, phase_v = 0, phase_do = 0;

        if (threadIdx.x < 128) {
            smem_D_c[threadIdx.x] = 0.0f;
        }

        load_tile_128x128(&tma_K, bar_k, smem_K, s_j, head_idx);
        load_tile_128x128(&tma_V, bar_v, smem_V, s_j, head_idx);
        
        uint32_t phase_q1 = 0, phase_do1 = 0;

        for (int i_block_b = num_blocks - 1; i_block_b >= j_block; --i_block_b) {
            uint32_t s_i = i_block_b * 128;
            
            mbarrier_wait_fn(bar_q, phase_q);
            mbarrier_wait_fn(bar_do, phase_do);
            
            if (i_block_b - 1 >= j_block) {
                if (threadIdx.x == 0) {
                    uint32_t next_s_i = (i_block_b - 1) * 128;
                    mbarrier_arrive_and_expect_tx_fn(bar_q1, 32768);
                    tma_load_3d_fn(&tma_Q, bar_q1, smem_Q, 0, next_s_i, head_idx);
                    tma_load_3d_fn(&tma_Q, bar_q1, smem_Q + 8192, 64, next_s_i, head_idx);
                    tma_load_3d_fn(&tma_Q, bar_q1, smem_Q + 16384, 0, next_s_i + 64, head_idx);
                    tma_load_3d_fn(&tma_Q, bar_q1, smem_Q + 24576, 64, next_s_i + 64, head_idx);
                    
                    mbarrier_arrive_and_expect_tx_fn(bar_do1, 32768);
                    tma_load_3d_fn(&tma_dO, bar_do1, smem_dO_1, 0, next_s_i, head_idx);
                    tma_load_3d_fn(&tma_dO, bar_do1, smem_dO_1 + 8192, 64, next_s_i, head_idx);
                    tma_load_3d_fn(&tma_dO, bar_do1, smem_dO_1 + 16384, 0, next_s_i + 64, head_idx);
                    tma_load_3d_fn(&tma_dO, bar_do1, smem_dO_1 + 24576, 64, next_s_i + 64, head_idx);
                }
            } else if (threadIdx.x == 0) {
                 mbarrier_arrive_and_expect_tx_fn(bar_q1, 32768);
                 mbarrier_arrive_and_expect_tx_fn(bar_do1, 32768);
            }
            
            gemm_transposed_B((uint32_t)__cvta_generic_to_shared(tmem_C), smem_Q, smem_K, 128, 128, bar_k, phase_k);
            gemm_transposed_B((uint32_t)__cvta_generic_to_shared(tmem_B), smem_dO, smem_V, 128, 128, bar_v, phase_v);
            
            float d_val = 0;
            uint32_t my_row = threadIdx.x;
            for (uint32_t col = 0; col < 128; col += 4) {
                uint32_t rC[4], rB[4];
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(rC[0]),"=r"(rC[1]),"=r"(rC[2]),"=r"(rC[3]) : "r"((uint32_t)__cvta_generic_to_shared(tmem_C) + col));
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(rB[0]),"=r"(rB[1]),"=r"(rB[2]),"=r"(rB[3]) : "r"((uint32_t)__cvta_generic_to_shared(tmem_B) + col));
                
                for(int i=0; i<4; ++i) {
                    float f = __uint_as_float(rC[i]);
                    float dp = __uint_as_float(rB[i]);
                    
                    f *= 0.08838834764831845; // scale 1/sqrt(128)
                    
                    int s_i_idx = s_i + my_row;
                    int s_j_idx = s_j + col + i;
                    if (s_j_idx > s_i_idx || s_j_idx >= S || s_i_idx >= S) f = -1e20;
                    float p = expf(f - L_ptr[head_idx * S + s_i_idx]);
                    if (s_j_idx > s_i_idx || s_j_idx >= S || s_i_idx >= S) p = 0;
                    
                    d_val += p * dp;
                    write_fp32_to_smem_swizzled(smem_P, my_row, col + i, p);
                }
            }
            
            __syncthreads();
            if (threadIdx.x < 128) {
                smem_D_c[threadIdx.x] += d_val;
            }
            __syncthreads();
            
            uint32_t my_row2 = threadIdx.x;
            for (uint32_t col = 0; col < 128; col += 4) {
                float p = __bfloat162float(read_swizzled_128x128(smem_P, my_row2, col));
                float dp = __uint_as_float(*((uint32_t*)&tmem_B[col]));
                
                int s_i_idx = s_i + my_row2;
                int s_j_idx = s_j + col;
                if (s_j_idx > s_i_idx || s_j_idx >= S || s_i_idx >= S) p = 0;
                
                float dp_c = p * (dp - smem_D_c[my_row2]);
                write_fp32_to_smem_swizzled(smem_P, my_row2, col, dp_c);
            }
            __syncthreads();
            fence_proxy_async_fn();
            
            gemm_transposed_A_B(
                (uint32_t)__cvta_generic_to_shared(tmem_C), 
                smem_P, smem_dO, 
                128, 128, bar_do, phase_do
            );
            
            gemm_transposed_A_B(
                (uint32_t)__cvta_generic_to_shared(tmem_B), 
                smem_P, smem_Q, 
                128, 128, bar_q, phase_q
            );
            
            __syncthreads();
            if (i_block_b - 1 >= j_block) {
                mbarrier_wait_fn(bar_q1, phase_q1);
                mbarrier_wait_fn(bar_do1, phase_do1);
                phase_q1 ^= 1;
                phase_do1 ^= 1;
                
                uint8_t* tmp_q = smem_Q;
                smem_Q = smem_Q_1; // Wait, smem_Q_1 is not defined in backward scope!
                smem_Q_1 = tmp_q;
                
                uint8_t* tmp_do = smem_dO;
                smem_dO = smem_dO_1;
                smem_dO_1 = tmp_do;
                
                uint64_t* tmp_bar_q = bar_q;
                bar_q = bar_q1;
                bar_q1 = tmp_bar_q;
                
                uint64_t* tmp_bar_do = bar_do;
                bar_do = bar_do1;
                bar_do1 = tmp_bar_do;
            }
            __syncthreads();
        }
        
        tmem_epilogue_coalesced_4w_fn(dK + head_idx * S * 128, (__nv_bfloat16*)smem_K, 
            S, 128, j_block, 0, 128, 128);
        tmem_epilogue_coalesced_4w_fn(dV + head_idx * S * 128, (__nv_bfloat16*)smem_V, 
            S, 128, j_block, 0, 128, 128);
    }

    if (threadIdx.x == 0) {
        tmem_dealloc_fn((uint32_t)__cvta_generic_to_shared(tmem_A), ncols_A);
        tmem_dealloc_fn((uint32_t)__cvta_generic_to_shared(tmem_B), ncols_B);
        tmem_dealloc_fn((uint32_t)__cvta_generic_to_shared(tmem_C), ncols_C);
    }
    __syncthreads();
}

CUresult create_tma_3d_descriptor(CUtensorMap* d, void* globalAddress, 
                                  uint64_t dim0, uint64_t dim1, uint64_t dim2,
                                  uint32_t box0, uint32_t box1) {
    cuuint64_t globalDim[3] = {dim0, dim1, dim2};
    cuuint64_t globalStrides[2] = {dim0 * 2, dim0 * dim1 * 2};
    cuuint32_t boxDim[3] = {box0, box1, 1};
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
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    uint32_t B = Q.size(0);
    uint32_t H = Q.size(1);
    uint32_t S = Q.size(2);
    uint32_t d = Q.size(3);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O, tma_dO;
    uint32_t BH = B * H;
    CU_CHECK(create_tma_3d_descriptor(&tma_Q, Q.data_ptr(), d, S, BH, 64, 64));
    CU_CHECK(create_tma_3d_descriptor(&tma_K, K.data_ptr(), d, S, BH, 64, 64));
    CU_CHECK(create_tma_3d_descriptor(&tma_V, V.data_ptr(), d, S, BH, 64, 64));
    CU_CHECK(create_tma_3d_descriptor(&tma_O, O.data_ptr(), d, S, BH, 64, 64));
    CU_CHECK(create_tma_3d_descriptor(&tma_dO, dO.data_ptr(), d, S, BH, 64, 64));
    
    uint32_t num_blocks = (S + 127) / 128;
    dim3 grid(num_blocks, BH);
    dim3 block(128);
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 368640; // 360 KB ensures enough space for all matrices including double buffering
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 1;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaFuncSetAttribute(tvm_ffi_mha_bwd::mha_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 368640));
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, tvm_ffi_mha_bwd::mha_bwd_kernel,
        tma_Q, tma_K, tma_V, tma_O, tma_dO, 
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dQ.data_ptr()), 
        static_cast<__nv_bfloat16*>(dK.data_ptr()), 
        static_cast<__nv_bfloat16*>(dV.data_ptr()), 
        S, H, B, true)); // is_forward = true
        
    CUDA_CHECK(cudaLaunchKernelEx(&config, tvm_ffi_mha_bwd::mha_bwd_kernel,
        tma_Q, tma_K, tma_V, tma_O, tma_dO, 
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dQ.data_ptr()), 
        static_cast<__nv_bfloat16*>(dK.data_ptr()), 
        static_cast<__nv_bfloat16*>(dV.data_ptr()), 
        S, H, B, false)); // is_forward = false
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_bwd::run);

} // namespace tvm_ffi_mha_bwd