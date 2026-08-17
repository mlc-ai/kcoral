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

#define CU_CHECK(call) do {                                        \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        fprintf(stderr, "CU error %d at %s:%d\n",                 \
                (int)_e, __FILE__, __LINE__);                      \
        exit(1);                                                   \
    }                                                              \
} while(0)

// -------------------------------------------------------------------------
// Device Helper Functions
// -------------------------------------------------------------------------

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
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

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
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
    // Fixed: Explicitly use .shared::cta namespace matching the actual state space of our dynamic smem.
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64 [%0];"
        :: "r"(a));
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61; // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn_major(uint32_t M, uint32_t N, uint32_t a_major, uint32_t b_major) {
    uint32_t d = 0;
    d |= (1u << 4);    // c_format = FP32
    d |= (1u << 7);    // a_format = BF16
    d |= (1u << 10);   // b_format = BF16
    d |= (a_major << 15);   // a_major
    d |= (b_major << 16);   // b_major
    d |= ((N / 8) << 17);   // n_dim
    d |= ((M / 16) << 24);  // m_dim
    return d;
}

__device__ __forceinline__ void store_swizzled_128B_u16(u16* base, int x, u16 val) {
    int chunk_x = x / 8;
    int rem = x % 8;
    int swizzled_chunk_x = chunk_x ^ ((base - smem_s) / 64 % 8);
    int swizzled_x = swizzled_chunk_x * 8 + rem;
    base[swizzled_x] = val;
}

// -------------------------------------------------------------------------
// Main Kernel
// -------------------------------------------------------------------------

extern __shared__ __align__(1024) char smem[];
__nv_bfloat16* smem_s; // For referencing in lambda/store_swizzled_128B_u16 scope context

__global__ void mha_kernel(
    const __grid_constant__ CUtensorMap tma_q,
    const __grid_constant__ CUtensorMap tma_k,
    const __grid_constant__ CUtensorMap tma_v,
    __nv_bfloat16* out_O,
    float* out_LSE,
    uint32_t S)
{
    uint32_t bh = blockIdx.y;
    uint32_t m_start = blockIdx.x * 64;
    
    if (m_start >= S) return;
    
    __nv_bfloat16* smem_q = (__nv_bfloat16*)smem;
    __nv_bfloat16* smem_k = smem_q + 4096;
    __nv_bfloat16* smem_v = smem_k + 4096;
    smem_s = smem_v + 4096;
    __nv_bfloat16* smem_temp = smem_s + 4096;
    
    uint64_t* bar_q = (uint64_t*)(smem_temp + 4096);
    uint64_t* bar_k = bar_q + 1;
    uint64_t* bar_v = bar_k + 1;
    uint64_t* bar_umma = bar_v + 1;
    float* row_max = (float*)(bar_umma + 1);
    float* row_sum = row_max + 64;
    uint32_t* tmem_base = (uint32_t*)(row_sum + 64);
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar_q, 1);
        init_smem_barrier_fn(bar_k, 1);
        init_smem_barrier_fn(bar_v, 1);
        init_smem_barrier_fn(bar_umma, 1);
        
        int ncols = 512 + 192 + 64 + 128;
        tmem_alloc_fn(tmem_base, ncols);
    }
    __syncthreads();
    
    uint32_t tmem_addr = *tmem_base;
    uint32_t tmem_q0 = tmem_addr + 0;
    uint32_t tmem_q1 = tmem_q0 + 256;
    uint32_t tmem_k0 = tmem_q1 + 256;
    uint32_t tmem_k1 = tmem_k0 + 128;
    uint32_t tmem_v0 = tmem_k1 + 128;
    uint32_t tmem_v1 = tmem_v0 + 128;
    uint32_t tmem_s  = tmem_v1 + 128;
    uint32_t tmem_p  = tmem_s + 64;
    uint32_t tmem_o  = tmem_p + 64;
    
    uint32_t load_phase = 0;
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_q, 16384);
        tma_load_2d_fn(&tma_q, bar_q, smem_q, 0, bh * S + m_start);
        tma_load_2d_fn(&tma_q, bar_q, smem_q + 4096, 64, bh * S + m_start);
    }
    __syncthreads();
    mbarrier_wait_fn(bar_q, load_phase % 2);
    load_phase++;
    
    if (threadIdx.x < 64) {
        row_max[threadIdx.x] = -INFINITY;
        row_sum[threadIdx.x] = 0.0f;
    }
    
    float o_reg0[64];
    float o_reg1[64];
    #pragma unroll
    for (uint32_t c = 0; c < 64; ++c) {
        o_reg0[c] = 0.0f;
        o_reg1[c] = 0.0f;
    }
    
    uint32_t umma_phase = 0;
    uint32_t S_total = S;
    __nv_bfloat16* D = out_O + (uint64_t)bh * S * 128;
    
    for (uint32_t j_block = 0; j_block <= m_start / 64; ++j_block) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_k, 32768);
            tma_load_2d_fn(&tma_k, bar_k, smem_k, 0, bh * S + j_block * 64);
            tma_load_2d_fn(&tma_k, bar_k, smem_k + 4096, 64, bh * S + j_block * 64);
            
            mbarrier_arrive_and_expect_tx_fn(bar_v, 32768);
            tma_load_2d_fn(&tma_v, bar_v, smem_v, 0, bh * S + j_block * 64);
            tma_load_2d_fn(&tma_v, bar_v, smem_v + 4096, 64, bh * S + j_block * 64);
        }
        __syncthreads();
        mbarrier_wait_fn(bar_k, load_phase % 2);
        mbarrier_wait_fn(bar_v, load_phase % 2);
        load_phase++;
        
        if (threadIdx.x == 0) {
            #pragma unroll
            for (uint32_t k = 0; k < 4; ++k) {
                uint64_t desc_q0 = make_smem_desc_sm100_fn(smem_q + k * 128, 1, 1024);
                uint64_t desc_k0 = make_smem_desc_sm100_fn(smem_k + k * 128, 1, 1024);
                uint32_t idesc = make_instr_desc_fn_major(64, 64, 0, 0);
                uint32_t acc = (k == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_p, desc_q0, desc_k0, idesc, acc);
            }
            #pragma unroll
            for (uint32_t k = 0; k < 4; ++k) {
                uint64_t desc_q1 = make_smem_desc_sm100_fn(smem_q + 4096 + k * 128, 1, 1024);
                uint64_t desc_k1 = make_smem_desc_sm100_fn(smem_k + 4096 + k * 128, 1, 1024);
                uint32_t idesc = make_instr_desc_fn_major(64, 64, 0, 0);
                uint32_t acc = (k == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_p, desc_q1, desc_k1, idesc, acc);
            }
            umma_commit_1sm_fn(bar_umma);
        }
        mbarrier_wait_fn(bar_umma, umma_phase % 2);
        umma_phase++;
        
        float p_regs[16];
        #pragma unroll
        for (uint32_t col = 0; col < 64; col += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_p + col));
            p_regs[(col/4)*4 + 0] = __uint_as_float(r0);
            p_regs[(col/4)*4 + 1] = __uint_as_float(r1);
            p_regs[(col/4)*4 + 2] = __uint_as_float(r2);
            p_regs[(col/4)*4 + 3] = __uint_as_float(r3);
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        if (threadIdx.x < 64) {
            uint32_t i = threadIdx.x;
            
            float m = -INFINITY;
            #pragma unroll
            for (uint32_t c = 0; c < 64; ++c) {
                float val = p_regs[c];
                if (j_block * 64 + c > m_start + i || bh * S + j_block * 64 + c >= S_total) {
                    val = -INFINITY;
                } else {
                    val *= 0.08838834764f;
                }
                p_regs[c] = val;
                if (val > m) m = val;
            }
            
            float prev_m = row_max[i];
            float m_new = (prev_m > m) ? prev_m : m;
            
            float s_old = __expf(prev_m - m_new);
            row_sum[i] *= s_old;
            
            #pragma unroll
            for (uint32_t c = 0; c < 64; ++c) {
                o_reg0[c] *= s_old;
                o_reg1[c] *= s_old;
            }
            
            float s_sum = 0.0f;
            #pragma unroll
            for (uint32_t c = 0; c < 64; ++c) {
                float s = __expf(p_regs[c] - m_new);
                s_sum += s;
                
                __nv_bfloat16 s_bf = __float2bfloat16(s);
                store_swizzled_128B_u16(smem_s + i * 64, c, *(reinterpret_cast<u16*>(&s_bf)));
            }
            row_sum[i] += s_sum;
            row_max[i] = m_new;
        }
        
        __syncthreads();
        asm volatile("fence.proxy.async;\n" ::: "memory");
        
        if (threadIdx.x == 0) {
            #pragma unroll
            for (uint32_t k = 0; k < 4; ++k) {
                uint64_t desc_s = make_smem_desc_sm100_fn(smem_s + k * 128, 1, 1024);
                uint64_t desc_v0 = make_smem_desc_sm100_fn(smem_v + k * 2048, 8192, 1024);
                uint32_t idesc = make_instr_desc_fn_major(64, 64, 0, 1);
                uint32_t acc_o0 = (j_block == 0 && k == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_o, desc_s, desc_v0, idesc, acc_o0);
            }
            #pragma unroll
            for (uint32_t k = 0; k < 4; ++k) {
                uint64_t desc_s = make_smem_desc_sm100_fn(smem_s + k * 128, 1, 1024);
                uint64_t desc_v1 = make_smem_desc_sm100_fn(smem_v + 4096 + k * 2048, 8192, 1024);
                uint32_t idesc = make_instr_desc_fn_major(64, 64, 0, 1);
                uint32_t acc_o1 = (j_block == 0 && k == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_o + 64, desc_s, desc_v1, idesc, acc_o1);
            }
            umma_commit_1sm_fn(bar_umma);
        }
        mbarrier_wait_fn(bar_umma, umma_phase % 2);
        umma_phase++;
        
        #pragma unroll
        for (uint32_t col = 0; col < 64; col += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_o + col));
            float f0 = __uint_as_float(r0);
            float f1 = __uint_as_float(r1);
            float f2 = __uint_as_float(r2);
            float f3 = __uint_as_float(r3);
            
            o_reg0[0 + col] += f0;
            o_reg0[1 + col] += f1;
            o_reg0[2 + col] += f2;
            o_reg0[3 + col] += f3;
        }
        #pragma unroll
        for (uint32_t col = 0; col < 64; col += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_o + 64 + col));
            float f0 = __uint_as_float(r0);
            float f1 = __uint_as_float(r1);
            float f2 = __uint_as_float(r2);
            float f3 = __uint_as_float(r3);
            
            o_reg1[0 + col] += f0;
            o_reg1[1 + col] += f1;
            o_reg1[2 + col] += f2;
            o_reg1[3 + col] += f3;
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    }
    
    __syncthreads();
    
    for (uint32_t col = 0; col < 64; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_o + col));
        float f0 = __uint_as_float(r0);
        float f1 = __uint_as_float(r1);
        float f2 = __uint_as_float(r2);
        float f3 = __uint_as_float(r3);
        
        uint32_t tid = threadIdx.x;
        float s_final = row_sum[tid];
        __nv_bfloat16 out0 = __float2bfloat16(f0 / s_final);
        __nv_bfloat16 out1 = __float2bfloat16(f1 / s_final);
        __nv_bfloat16 out2 = __float2bfloat16(f2 / s_final);
        __nv_bfloat16 out3 = __float2bfloat16(f3 / s_final);
        
        uint32_t base = tid * 64 + col;
        store_swizzled_128B_u16(smem_temp, base + 0, *reinterpret_cast<u16*>(&out0));
        store_swizzled_128B_u16(smem_temp, base + 1, *reinterpret_cast<u16*>(&out1));
        store_swizzled_128B_u16(smem_temp, base + 2, *reinterpret_cast<u16*>(&out2));
        store_swizzled_128B_u16(smem_temp, base + 3, *reinterpret_cast<u16*>(&out3));
    }
    
    for (uint32_t col = 0; col < 64; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_o + 64 + col));
        float f0 = __uint_as_float(r0);
        float f1 = __uint_as_float(r1);
        float f2 = __uint_as_float(r2);
        float f3 = __uint_as_float(r3);
        
        uint32_t tid = threadIdx.x;
        float s_final = row_sum[tid];
        __nv_bfloat16 out0 = __float2bfloat16(f0 / s_final);
        __nv_bfloat16 out1 = __float2bfloat16(f1 / s_final);
        __nv_bfloat16 out2 = __float2bfloat16(f2 / s_final);
        __nv_bfloat16 out3 = __float2bfloat16(f3 / s_final);
        
        uint32_t base = tid * 64 + col;
        store_swizzled_128B_u16(smem_temp + 4096, base + 0, *reinterpret_cast<u16*>(&out0));
        store_swizzled_128B_u16(smem_temp + 4096, base + 1, *reinterpret_cast<u16*>(&out1));
        store_swizzled_128B_u16(smem_temp + 4096, base + 2, *reinterpret_cast<u16*>(&out2));
        store_swizzled_128B_u16(smem_temp + 4096, base + 3, *reinterpret_cast<u16*>(&out3));
    }
    tmem_load_fence_fn();
    __syncthreads();
    
    if (threadIdx.x < 64) {
        if (m_start + threadIdx.x < S_total) {
            out_LSE[bh * S + m_start + threadIdx.x] = row_max[threadIdx.x] + logf(row_sum[threadIdx.x]);
        }
    }
    
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t num_steps = (64 + 3) / 4;
    
    __nv_bfloat16* temp0 = (__nv_bfloat16*)smem_temp;
    __nv_bfloat16* temp1 = temp0 + 4096;
    
    uint32_t M = S;
    uint32_t N = 128;
    uint32_t m_block = blockIdx.x;
    uint32_t n_block = 0;
    uint32_t BM = 64;
    uint32_t BN = 128;

    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 4 + warp_id;
        if (row >= BM) continue;
        uint32_t global_row = m_block * BM + row;
        uint32_t col_start = lane_id * 4;
        
        if (global_row < M) {
            uint32_t global_col0 = n_block * BN + col_start;
            if (global_col0 + 3 < N) {
                uint2 data = *reinterpret_cast<uint2*>(&temp0[row * 64 + col_start]);
                *reinterpret_cast<uint2*>(D + (uint64_t)global_row * N + global_col0) = data;
            }
            uint32_t global_col1 = n_block * BN + 64 + col_start;
            if (global_col1 + 3 < N) {
                uint2 data = *reinterpret_cast<uint2*>(&temp1[row * 64 + col_start]);
                *reinterpret_cast<uint2*>(D + (uint64_t)global_row * N + global_col1) = data;
            }
        }
    }
}

// -------------------------------------------------------------------------
// Host-Side Setup and FFI Binding
// -------------------------------------------------------------------------

namespace tvm_ffi_mha {

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

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    uint32_t B = Q.size(0);
    uint32_t H = Q.size(1);
    uint32_t S = Q.size(2);
    uint32_t D = Q.size(3); 
    
    uint32_t total_S = B * H * S;
    
    CUtensorMap tma_q, tma_k, tma_v;
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_q, Q.data_ptr(), D, total_S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_k, K.data_ptr(), D, total_S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_v, V.data_ptr(), D, total_S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
    uint32_t smem_size = 73728; 
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    dim3 grid((S + 63) / 64, B * H);
    dim3 block(64, 1, 1);
    
    CUDA_CHECK(cudaFuncSetAttribute(mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    mha_kernel<<<grid, block, smem_size, stream>>>(
        tma_q, tma_k, tma_v, 
        static_cast<__nv_bfloat16*>(O.data_ptr()), 
        static_cast<float*>(LSE.data_ptr()), 
        S
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_mha