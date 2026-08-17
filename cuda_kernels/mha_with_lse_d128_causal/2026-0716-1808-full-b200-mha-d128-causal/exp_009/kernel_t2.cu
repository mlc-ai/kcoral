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

__device__ __forceinline__ void init_expect_tx(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.expect_tx.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes));
}

__device__ __forceinline__ void load_tile_128bit_async(
    __nv_bfloat16* smem, const __nv_bfloat16* gmem,
    uint32_t row, uint32_t col, uint32_t stride,
    uint64_t* bar)
{
    uint32_t smem_idx = ((col / 8) ^ (row % 8)) * 8;
    smem_idx = row * 64 + smem_idx;
    asm volatile("cp.async.mbarrier.arrive.shared::cta [%0], [%1], 16, [%2]);"
                 :: "r"((uint32_t)__cvta_generic_to_shared(&smem[smem_idx])),
                    "l"(gmem + row * stride + col), "r"((uint32_t)__cvta_generic_to_shared(&bar[0])));
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
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
        :: "r"(a));
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

// -------------------------------------------------------------------------
// Main Kernel
// -------------------------------------------------------------------------

__global__ void mha_kernel(
    const __nv_bfloat16* Q_gmem,
    const __nv_bfloat16* K_gmem,
    const __nv_bfloat16* V_gmem,
    __nv_bfloat16* out_O,
    float* out_LSE,
    uint32_t S)
{
    uint32_t bh = blockIdx.y;
    uint32_t m_start = blockIdx.x * 64;
    
    if (m_start >= S) return;
    
    extern __shared__ __align__(128) char smem[];
    __nv_bfloat16* q0 = (__nv_bfloat16*)smem;
    __nv_bfloat16* q1 = q0 + 4096;
    __nv_bfloat16* k0 = q1 + 4096;
    __nv_bfloat16* k1 = k0 + 4096;
    __nv_bfloat16* v0 = k1 + 4096;
    __nv_bfloat16* v1 = v0 + 4096;
    __nv_bfloat16* o0 = v1 + 4096;
    __nv_bfloat16* o1 = o0 + 4096;
    __nv_bfloat16* s_ = k0;
    __nv_bfloat16* temp = k1;
    
    uint64_t* bar_load = (uint64_t*)(o1 + 4096);
    uint64_t* bar_umma = bar_load + 1;
    float* row_max = (float*)(bar_umma + 1);
    float* row_sum = row_max + 64;
    uint32_t* tmem_base = (uint32_t*)(row_sum + 64);
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar_load, 1);
        init_smem_barrier_fn(bar_umma, 1);
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        tmem_alloc_fn(tmem_base, 256);
    }
    __syncthreads();
    
    uint32_t tmem_addr = *tmem_base;
    uint32_t tmem_p = tmem_addr + 0;
    uint32_t tmem_o = tmem_addr + 64;
    
    uint32_t load_phase = 0;
    
    if (threadIdx.x == 0) {
        init_expect_tx(bar_load, 16384);
        load_tile_128bit_async(q0, (const __nv_bfloat16*)Q_gmem, bh * S + m_start, 0, 128, bar_load);
        load_tile_128bit_async(q1, (const __nv_bfloat16*)Q_gmem, bh * S + m_start, 64, 128, bar_load);
    }
    __syncthreads();
    mbarrier_wait_fn(bar_load, load_phase % 2);
    load_phase++;
    
    if (threadIdx.x < 64) {
        row_max[threadIdx.x] = -INFINITY;
        row_sum[threadIdx.x] = 0.0f;
    }
    
    uint32_t umma_phase = 0;
    uint32_t S_total = S;
    
    for (uint32_t j_block = 0; j_block <= m_start / 64; ++j_block) {
        if (threadIdx.x == 0) {
            init_expect_tx(bar_load, 32768);
            load_tile_128bit_async(k0, (const __nv_bfloat16*)K_gmem, bh * S + j_block * 64, 0, 128, bar_load);
            load_tile_128bit_async(k1, (const __nv_bfloat16*)K_gmem, bh * S + j_block * 64, 64, 128, bar_load);
            load_tile_128bit_async(v0, (const __nv_bfloat16*)V_gmem, bh * S + j_block * 64, 0, 128, bar_load);
            load_tile_128bit_async(v1, (const __nv_bfloat16*)V_gmem, bh * S + j_block * 64, 64, 128, bar_load);
        }
        __syncthreads();
        mbarrier_wait_fn(bar_load, load_phase % 2);
        load_phase++;
        
        if (threadIdx.x == 0) {
            #pragma unroll
            for (uint32_t k = 0; k < 4; ++k) {
                uint64_t desc_q0 = make_smem_desc_sm100_fn(q0 + k * 16, 1, 1024);
                uint64_t desc_k0 = make_smem_desc_sm100_fn(k0 + k * 16, 1, 1024);
                uint32_t idesc = make_instr_desc_fn(64, 64);
                uint32_t acc = (k == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_p, desc_q0, desc_k0, idesc, acc);
            }
            #pragma unroll
            for (uint32_t k = 0; k < 4; ++k) {
                uint64_t desc_q1 = make_smem_desc_sm100_fn(q1 + k * 16, 1, 1024);
                uint64_t desc_k1 = make_smem_desc_sm100_fn(k1 + k * 16, 1, 1024);
                uint32_t idesc = make_instr_desc_fn(64, 64);
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
        
        float o_reg0[64];
        float o_reg1[64];
        
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
                
                uint32_t chunk_x = c / 8;
                uint32_t swizzled_chunk_x = chunk_x ^ (i % 8);
                uint32_t c_swizzled = swizzled_chunk_x * 8 + (c % 8);
                s_[i * 64 + c_swizzled] = __float2bfloat16(s);
            }
            row_sum[i] += s_sum;
            row_max[i] = m_new;
        }
        
        __syncthreads();
        asm volatile("fence.proxy.async;\n" ::: "memory");
        
        if (threadIdx.x == 0) {
            #pragma unroll
            for (uint32_t k = 0; k < 4; ++k) {
                uint64_t desc_s = make_smem_desc_sm100_fn(s_ + k * 16, 1, 1024);
                uint64_t desc_v0 = make_smem_desc_sm100_fn(v0 + k * 1024, 8192, 1024);
                uint32_t idesc = make_instr_desc_fn(64, 64);
                uint32_t acc_o0 = (j_block == 0 && k == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_o, desc_s, desc_v0, idesc, acc_o0);
            }
            #pragma unroll
            for (uint32_t k = 0; k < 4; ++k) {
                uint64_t desc_s = make_smem_desc_sm100_fn(s_ + k * 16, 1, 1024);
                uint64_t desc_v1 = make_smem_desc_sm100_fn(v1 + k * 1024, 8192, 1024);
                uint32_t idesc = make_instr_desc_fn(64, 64);
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
    
    if (threadIdx.x < 64) {
        uint32_t i = threadIdx.x;
        float s_final = row_sum[i];
        
        #pragma unroll
        for (uint32_t c = 0; c < 64; ++c) {
            float o_val0 = o_reg0[c] / s_final;
            float o_val1 = o_reg1[c] / s_final;
            uint32_t chunk_x = c / 8;
            uint32_t swizzled_chunk_x = chunk_x ^ (i % 8);
            uint32_t c_swizzled = swizzled_chunk_x * 8 + (c % 8);
            o0[i * 64 + c_swizzled] = __float2bfloat16(o_val0);
            o1[i * 64 + c_swizzled] = __float2bfloat16(o_val1);
        }
        
        if (m_start + i < S_total) {
            out_LSE[bh * S + m_start + i] = row_max[i] + logf(row_sum[i]);
        }
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
        uint32_t base = tid * 64 + col;
        float s_final = row_sum[tid];
        o0[base + 0] = __float2bfloat16(f0 / s_final);
        o0[base + 1] = __float2bfloat16(f1 / s_final);
        o0[base + 2] = __float2bfloat16(f2 / s_final);
        o0[base + 3] = __float2bfloat16(f3 / s_final);
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
        uint32_t base = tid * 64 + col;
        float s_final = row_sum[tid];
        o1[base + 0] = __float2bfloat16(f0 / s_final);
        o1[base + 1] = __float2bfloat16(f1 / s_final);
        o1[base + 2] = __float2bfloat16(f2 / s_final);
        o1[base + 3] = __float2bfloat16(f3 / s_final);
    }
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    __syncthreads();
    
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t num_steps = (64 + 3) / 4;
    
    __nv_bfloat16* D = out_O + (uint64_t)bh * S * 128;
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
                uint2 data = *reinterpret_cast<uint2*>(&o0[row * 64 + col_start]);
                *reinterpret_cast<uint2*>(D + (uint64_t)global_row * N + global_col0) = data;
            }
            uint32_t global_col1 = n_block * BN + 64 + col_start;
            if (global_col1 + 3 < N) {
                uint2 data = *reinterpret_cast<uint2*>(&o1[row * 64 + col_start]);
                *reinterpret_cast<uint2*>(D + (uint64_t)global_row * N + global_col1) = data;
            }
        }
    }
}

// -------------------------------------------------------------------------
// Host-Side Setup and FFI Binding
// -------------------------------------------------------------------------

namespace tvm_ffi_mha {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    uint32_t B = Q.size(0);
    uint32_t H = Q.size(1);
    uint32_t S = Q.size(2);
    uint32_t D = Q.size(3); 
    
    uint32_t smem_size = 72000; 
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    dim3 grid((S + 63) / 64, B * H);
    dim3 block(128, 1, 1);
    
    CUDA_CHECK(cudaFuncSetAttribute(mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    mha_kernel<<<grid, block, smem_size, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<__nv_bfloat16*>(O.data_ptr()), 
        static_cast<float*>(LSE.data_ptr()), 
        S
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_mha