#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>
#include <algorithm>
#include <cmath>

#define CU_CHECK(call) do {                                      \
    CUresult _e = (call);                                        \
    if (_e != CUDA_SUCCESS) {                                    \
        const char* err_str;                                     \
        cuGetErrorString(_e, &err_str);                          \
        fprintf(stderr, "CUDA Driver error %s at %s:%d\n",       \
                err_str, __FILE__, __LINE__);                    \
        exit(1);                                                 \
    }                                                            \
} while(0)

namespace mha_kernel {

// ----------------------------------------------------------------------
// PTX Helper Functions
// ----------------------------------------------------------------------

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
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
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

__device__ __forceinline__ void st_shared_128_fn(uint32_t addr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "r"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
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

__device__ __forceinline__ void tmem_epilogue_coalesced_4w_fn(
    __nv_bfloat16* D, __nv_bfloat16* smem_out,
    uint32_t M, uint32_t N, uint32_t m_block, uint32_t n_block,
    uint32_t BM, uint32_t BN) {
    for (uint32_t col = 0; col < BN; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t base = threadIdx.x * BN + col;
        smem_out[base + 0] = __float2bfloat16(__uint_as_float(r0));
        smem_out[base + 1] = __float2bfloat16(__uint_as_float(r1));
        smem_out[base + 2] = __float2bfloat16(__uint_as_float(r2));
        smem_out[base + 3] = __float2bfloat16(__uint_as_float(r3));
    }
    __syncthreads();
    
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t num_steps = (BM + 3) / 4;
    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 4 + warp_id;
        if (row >= BM) continue;
        uint32_t global_row = m_block * BM + row; // BUG: should be m_block + row
        uint32_t col_start = lane_id * 4;
        uint32_t global_col = n_block * BN + col_start;
        if (global_row < M && global_col + 3 < N) {
            uint2 data = *reinterpret_cast<uint2*>(&smem_out[row * BN + col_start]);
            *reinterpret_cast<uint2*>(D + (uint64_t)global_row * N + global_col) = data;
        }
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

// ----------------------------------------------------------------------
// Attention Kernel
// ----------------------------------------------------------------------

__global__ __launch_bounds__(128) void mha_attention_wgmma_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O,
    float* LSE,
    uint32_t output_S, uint32_t output_D,
    uint32_t H) 
{
    // ---------------------- Setup ----------------------
    extern __shared__ __align__(128) char smem_pool[];
    uint64_t* mbar = (uint64_t*)smem_pool;
    uint32_t* tmem_Q = (uint32_t*)(smem_pool + 8);
    uint32_t* tmem_K = (uint32_t*)(smem_pool + 16);
    uint32_t* tmem_P = (uint32_t*)(smem_pool + 24);
    uint32_t* tmem_V = (uint32_t*)(smem_pool + 32);
    uint32_t* tmem_QK = (uint32_t*)(smem_pool + 40);
    uint32_t* tmem_PV = (uint32_t*)(smem_pool + 48);

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        tmem_alloc_fn(tmem_Q, 64);
        tmem_alloc_fn(tmem_K, 64);
        tmem_alloc_fn(tmem_P, 64);
        tmem_alloc_fn(tmem_V, 64);
        tmem_alloc_fn(tmem_QK, 64);
        tmem_alloc_fn(tmem_PV, 64);
    }
    __syncthreads();

    // Grid maps sequentially over (batch, head, sequence blocks)
    uint32_t grid_size = gridDim.x * gridDim.y;
    uint32_t seq_idx = blockIdx.x;
    uint32_t bh_idx = blockIdx.y;
    uint32_t b = bh_idx / H;
    uint32_t h = bh_idx % H;

    uint32_t batch_head_offset = (b * H + h) * output_S;
    uint32_t row_offset = seq_idx * 128 + cluster_rank_fn() * 64; 
    
    // ---------------------- Memory Pools ----------------------
    __nv_bfloat16* smem_Q = (__nv_bfloat16*)(smem_pool + 64); // 8192 bytes
    __nv_bfloat16* smem_K = (__nv_bfloat16*)(smem_pool + 8256); // 8192 bytes
    __nv_bfloat16* smem_V = (__nv_bfloat16*)(smem_pool + 16448); // 8192 bytes
    __nv_bfloat16* smem_P = (__nv_bfloat16*)(smem_pool + 24640); // 8192 bytes
    __nv_bfloat16* smem_O = (__nv_bfloat16*)(smem_pool + 32832); // 8192 bytes

    // ---------------------- Constants ----------------------
    constexpr int BM = 64;
    constexpr int BN = 64;
    uint32_t my_m_base = row_offset;
    float qk_scale = 1.0f / sqrtf((float)output_D);
    
    uint32_t idesc_QK = make_instr_desc_fn(BM, BN * 2);
    uint32_t idesc_PV = make_instr_desc_fn(BM, BN);
    
    uint32_t phase = 0;
    uint32_t tid = threadIdx.x;
    int K_step_size = 4; // 4 iterations of K=16 equals K=64

    // ---------------------- QK Pass ----------------------
    float global_max[2] = {-1e20f, -1e20f};
    float global_sum[2] = {0.0f, 0.0f};

    for (int our_n_offset = 0; our_n_offset < 2; our_n_offset++) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar, 8192 * 2); // Expecting 16KB of TMA transfers
            tma_load_2d_fn(&tma_Q, mbar, smem_Q, 0, batch_head_offset + my_m_base);
            tma_load_2d_fn(&tma_Q, mbar, (char*)smem_Q + 4096, 64, batch_head_offset + my_m_base);
            tma_load_2d_fn(&tma_K, mbar, smem_K, 0, batch_head_offset + our_n_offset * 64);
            tma_load_2d_fn(&tma_K, mbar, (char*)smem_K + 4096, 64, batch_head_offset + our_n_offset * 64);
        }
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;

        uint64_t desc_Q[2], desc_K_T[2];
        for (int i = 0; i < 2; ++i) {
            desc_Q[i] = make_smem_desc_sm100_fn(smem_Q + i * 4096, 1, 1024); 
            desc_K_T[i] = make_smem_desc_sm100_fn(smem_K + i * 4096, 1024, 1024); 
        }

        uint32_t our_tmem_col_base = our_n_offset * 64;
        uint32_t our_n_smem_P_offset = our_n_offset * 4096;
        
        for (int k_idx = 0; k_idx < K_step_size; k_idx++) {
            uint32_t tmem_col = our_tmem_col_base + k_idx * 16;
            uint32_t our_k_offset_Q = k_idx * 16;
            uint32_t our_k_offset_K = k_idx * 16; 
            
            uint64_t desc_A = desc_Q[our_k_offset_Q / 4096] + (((our_k_offset_Q % 4096) / 16)) * 16;
            uint64_t desc_B = desc_K_T[our_k_offset_K / 4096] + (((our_k_offset_K % 4096) / 16)) * 16;
            
            umma_f16_cg2_fn(tmem_col, desc_A, desc_B, idesc_QK, k_idx == 0 ? 0 : 1);
        }
        umma_commit_2sm_fn(mbar);
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;

        float local_max[8];
        for (int i = 0; i < 8; ++i) local_max[i] = -1e20f;
        
        for (int k_idx = 0; k_idx < K_step_size; k_idx++) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_QK + our_tmem_col_base + k_idx * 16));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            int idx = (tid / 32) * 2 + ((tid % 32) / 16) + ((tid % 32) % 16) * 2;
            float val0 = __uint_as_float(r0) * qk_scale;
            float val1 = __uint_as_float(r1) * qk_scale;
            float val2 = __uint_as_float(r2) * qk_scale;
            float val3 = __uint_as_float(r3) * qk_scale;
            local_max[idx] = fmaxf(local_max[idx], val0);
            local_max[idx+1] = fmaxf(local_max[idx+1], val1);
            local_max[idx+2] = fmaxf(local_max[idx+2], val2);
            local_max[idx+3] = fmaxf(local_max[idx+3], val3);
        }

        for (int offset = 2; offset < 128; offset *= 2) {
            local_max[0] = fmaxf(local_max[0], __shfl_xor_sync(0xffffffff, local_max[0], offset));
            local_max[1] = fmaxf(local_max[1], __shfl_xor_sync(0xffffffff, local_max[1], offset));
        }

        float m_new_0 = fmaxf(-1e20f, local_max[0]);
        float m_new_1 = fmaxf(-1e20f, local_max[1]);

        float m_curr_0 = global_max[0];
        float m_curr_1 = global_max[1];

        float P_corr_0 = expf(m_curr_0 - m_new_0);
        float P_corr_1 = expf(m_curr_1 - m_new_1);

        global_sum[0] *= P_corr_0;
        global_sum[1] *= P_corr_1;

        float local_sum[8];
        for(int i=0; i<8; ++i) local_sum[i] = 0.0f;

        for (int k_idx = 0; k_idx < K_step_size; k_idx++) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_QK + our_tmem_col_base + k_idx * 16));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            int idx = (tid / 32) * 2 + ((tid % 32) / 16) + ((tid % 32) % 16) * 2;
            float val0 = __uint_as_float(r0) * qk_scale;
            float val1 = __uint_as_float(r1) * qk_scale;
            float val2 = __uint_as_float(r2) * qk_scale;
            float val3 = __uint_as_float(r3) * qk_scale;

            float p0 = expf(val0 - m_new_0);
            float p1 = expf(val1 - m_new_1);
            float p2 = expf(val2 - m_new_0);
            float p3 = expf(val3 - m_new_1);

            local_sum[idx] += p0;
            local_sum[idx+1] += p1;
            local_sum[idx+2] += p2;
            local_sum[idx+3] += p3;

            uint32_t pr0 = pack_bf16_fn(__float_as_uint(p0), __float_as_uint(p1));
            uint32_t pr1 = pack_bf16_fn(__float_as_uint(p2), __float_as_uint(p3));
            
            uint32_t p_base = (uint32_t)__cvta_generic_to_shared(smem_P + our_n_smem_P_offset);
            st_shared_128_fn(p_base + k_idx * 16, pr0, pr1, 0, 0);
        }

        for (int offset = 2; offset < 128; offset *= 2) {
            local_sum[0] += __shfl_xor_sync(0xffffffff, local_sum[0], offset);
            local_sum[1] += __shfl_xor_sync(0xffffffff, local_sum[1], offset);
        }

        global_sum[0] += local_sum[0] * P_corr_0;
        global_sum[1] += local_sum[1] * P_corr_1;

        global_max[0] = m_new_0;
        global_max[1] = m_new_1;
    }

    for (int k_idx = 0; k_idx < K_step_size; k_idx++) {
        uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(tmem_QK + k_idx * 16));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float val0 = __uint_as_float(r0) * qk_scale;
        float val1 = __uint_as_float(r1) * qk_scale;
        float val2 = __uint_as_float(r2) * qk_scale;
        float val3 = __uint_as_float(r3) * qk_scale;
        float val4 = __uint_as_float(r4) * qk_scale;
        float val5 = __uint_as_float(r5) * qk_scale;
        float val6 = __uint_as_float(r6) * qk_scale;
        float val7 = __uint_as_float(r7) * qk_scale;

        float p0 = expf(val0 - global_max[0]) / global_sum[0];
        float p1 = expf(val1 - global_max[1]) / global_sum[1];
        float p2 = expf(val2 - global_max[0]) / global_sum[0];
        float p3 = expf(val3 - global_max[1]) / global_sum[1];
        float p4 = expf(val4 - global_max[0]) / global_sum[0];
        float p5 = expf(val5 - global_max[1]) / global_sum[1];
        float p6 = expf(val6 - global_max[0]) / global_sum[0];
        float p7 = expf(val7 - global_max[1]) / global_sum[1];

        uint32_t pr0 = pack_bf16_fn(__float_as_uint(p0), __float_as_uint(p1));
        uint32_t pr1 = pack_bf16_fn(__float_as_uint(p2), __float_as_uint(p3));
        uint32_t pr2 = pack_bf16_fn(__float_as_uint(p4), __float_as_uint(p5));
        uint32_t pr3 = pack_bf16_fn(__float_as_uint(p6), __float_as_uint(p7));

        uint32_t p_base = (uint32_t)__cvta_generic_to_shared(smem_P);
        st_shared_128_fn(p_base + k_idx * 64, pr0, pr1, pr2, pr3);
    }
    __syncwarp();

    // ---------------------- PV Pass ----------------------
    for (int our_n_offset = 0; our_n_offset < 2; our_n_offset++) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar, 8192);
            tma_load_2d_fn(&tma_V, mbar, smem_V, 0, batch_head_offset + our_n_offset * 64);
            tma_load_2d_fn(&tma_V, mbar, (char*)smem_V + 4096, 64, batch_head_offset + our_n_offset * 64);
        }
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;

        uint64_t desc_P[2], desc_V[2];
        for (int i = 0; i < 2; ++i) {
            desc_P[i] = make_smem_desc_sm100_fn(smem_P + i * 4096, 1, 1024);
            desc_V[i] = make_smem_desc_sm100_fn(smem_V + i * 4096, 1024, 1024);
        }

        for (int k_idx = 0; k_idx < K_step_size; k_idx++) {
            uint32_t tmem_col = our_n_offset * 64 + k_idx * 16;
            uint32_t our_k_offset_P = k_idx * 16;
            uint32_t our_k_offset_V = k_idx * 16;
            
            uint64_t desc_A = desc_P[our_k_offset_P / 4096] + (((our_k_offset_P % 4096) / 16)) * 16;
            uint64_t desc_B = desc_V[our_k_offset_V / 4096] + (((our_k_offset_V % 4096) / 16)) * 16;
            
            umma_f16_cg2_fn(tmem_col, desc_A, desc_B, idesc_PV, k_idx == 0 ? 0 : 1);
        }
        umma_commit_2sm_fn(mbar);
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
    }

    // ---------------------- Epilogue ----------------------
    tmem_epilogue_coalesced_4w_fn(O + ((b * H + h) * output_S + my_m_base) * output_D, smem_O, 
        output_S, output_D, my_m_base, 0, BM, BN);

    if (tid < 128) {
        uint32_t m_idx = my_m_base + tid;
        if (m_idx < output_S) {
            LSE[((b * H + h) * output_S + m_idx)] = global_max[tid % 2];
        }
    }

    // ---------------------- Deallocate TMEM ----------------------
    if (threadIdx.x < 32) {
        tmem_dealloc_fn(*tmem_Q, 64);
        tmem_dealloc_fn(*tmem_K, 64);
        tmem_dealloc_fn(*tmem_P, 64);
        tmem_dealloc_fn(*tmem_V, 64);
        tmem_dealloc_fn(*tmem_QK, 64);
        tmem_dealloc_fn(*tmem_PV, 64);
    }
}

// ----------------------------------------------------------------------
// TVM-FFI Binding
// ----------------------------------------------------------------------

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
         
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    if (S == 0) return;

    CUtensorMap tma_Q, tma_K, tma_V;
    void* q_ptr = Q.data_ptr();
    void* k_ptr = K.data_ptr();
    void* v_ptr = V.data_ptr();

    CU_CHECK(create_tma_2d_descriptor_2B(&tma_Q, q_ptr, D, B * H * S, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_K, k_ptr, D, B * H * S, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_V, v_ptr, D, B * H * S, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    int blocks_x = (S + 127) / 128;
    dim3 grid(blocks_x, B * H);
    dim3 block(128);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 40960;
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, mha_attention_wgmma_kernel, 
        tma_Q, tma_K, tma_V, 
        static_cast<__nv_bfloat16*>(O.data_ptr()), static_cast<float*>(LSE.data_ptr()), 
        S, D, H));
        
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace mha_kernel