#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <mma.h>
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

using namespace nvcuda;

namespace mha_bwd_d128_causal {

__global__ void precompute_D(const __nv_bfloat16* O, const __nv_bfloat16* dO, float* D, int B, int H, int S, int d) {
    int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int head_idx = blockIdx.y;
    int batch_idx = blockIdx.z;
    if (seq_idx < S) {
        float sum = 0.0f;
        int base = ((batch_idx * H + head_idx) * S + seq_idx) * d;
        for (int i = 0; i < d; i++) {
            float o = __bfloat162float(O[base + i]);
            float do_ = __bfloat162float(dO[base + i]);
            sum += o * do_;
        }
        D[(batch_idx * H + head_idx) * S + seq_idx] = sum;
    }
}

__global__ void convert_dQ(const float* dQ_float, __nv_bfloat16* dQ, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        dQ[idx] = __float2bfloat16(dQ_float[idx]);
    }
}

__device__ __forceinline__ uint64_t make_smem_desc_k_major(void* ptr, int k_offset, int M) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr) + k_offset * 2;
    uint32_t sbo = 8 * 128; 
    uint32_t lbo = (M / 8) * sbo;
    uint64_t d = ((uint64_t)(addr & 0x3FFFF) >> 4);
    d |= ((uint64_t)((lbo & 0x3FFFF) >> 4) << 16);
    d |= ((uint64_t)((sbo & 0x3FFFF) >> 4) << 32);
    d |= (1ull << 46);
    d |= (0ull << 61); // SWIZZLE_NONE
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_mn_major(void* ptr, int k_offset, int K_dim) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr) + k_offset * 128; 
    uint32_t lbo = 128; 
    uint32_t sbo = (K_dim / 8) * lbo;
    uint64_t d = ((uint64_t)(addr & 0x3FFFF) >> 4);
    d |= ((uint64_t)((lbo & 0x3FFFF) >> 4) << 16);
    d |= ((uint64_t)((sbo & 0x3FFFF) >> 4) << 32);
    d |= (1ull << 46);
    d |= (0ull << 61); // SWIZZLE_NONE
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc(int M, int N, int a_trans, int b_trans) {
    uint32_t d = 0;
    d |= (1u << 4);    // FP32 acc
    d |= (1u << 7);    // BF16 A
    d |= (1u << 10);   // BF16 B
    d |= ((uint32_t)a_trans << 15);
    d |= ((uint32_t)b_trans << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ void umma_f16_cg1(uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, int accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_f16_cg1_tmem_A(uint32_t tmem_c, uint32_t tmem_a, uint64_t desc_b, uint32_t idesc, int accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}\n"
        :: "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void wait_mma(uint64_t* mbar, int phase) {
    uint32_t mbar_ptr = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(mbar_ptr) : "memory");
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"(mbar_ptr), "r"(phase) : "memory");
}

__device__ __forceinline__ void load_gmem_to_smem_64x64(const __nv_bfloat16* gmem, __nv_bfloat16* smem, int batch_H_S, int d, int row_start, int col_start, int S) {
    int tid = threadIdx.x; 
    for (int i = tid; i < 64 * 64 / 8; i += 128) {
        int r = (i * 8) / 64;
        int c = (i * 8) % 64;
        int global_r = row_start + r;
        int global_c = col_start + c;
        if (global_r < S) {
            uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(&smem[r * 64 + c]);
            const void* gmem_addr = &gmem[batch_H_S * d + global_r * d + global_c];
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         :: "r"(smem_addr), "l"(gmem_addr) : "memory");
        } else {
            uint4 zero = {0,0,0,0};
            *(uint4*)&smem[r * 64 + c] = zero;
        }
    }
}

__device__ __forceinline__ void load_gmem_to_smem_128x64(const __nv_bfloat16* gmem, __nv_bfloat16* smem, int batch_H_S, int d, int row_start, int col_start, int S) {
    int tid = threadIdx.x; 
    for (int i = tid; i < 128 * 64 / 8; i += 128) {
        int r = (i * 8) / 64;
        int c = (i * 8) % 64;
        int global_r = row_start + r;
        int global_c = col_start + c;
        if (global_r < S) {
            uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(&smem[r * 64 + c]);
            const void* gmem_addr = &gmem[batch_H_S * d + global_r * d + global_c];
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         :: "r"(smem_addr), "l"(gmem_addr) : "memory");
        } else {
            uint4 zero = {0,0,0,0};
            *(uint4*)&smem[r * 64 + c] = zero;
        }
    }
}

__device__ __forceinline__ void store_tmem_to_smem(uint32_t tmem_col, __nv_bfloat16* smem, int rows, int cols_bf16) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int row = warp_id * 32 + lane_id;
    int cols_tmem = cols_bf16 / 2;
    for (int c = 0; c < cols_tmem; c += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_col + c));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        if (row < rows) {
            uint4* out = (uint4*)&smem[row * cols_bf16 + c * 2];
            *out = make_uint4(r0, r1, r2, r3);
        }
    }
}

__device__ __forceinline__ void atomic_add_dQ(uint32_t tmem_col, float* dQ_float, int i_start, int j_start, int S, int d_offset, int batch_H_S, int d) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int row = warp_id * 32 + lane_id;
    for (int c = 0; c < 32; c += 4) {
        uint32_t r[4];
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]) : "r"(tmem_col + c));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        if (row < 64) {
            int global_q = j_start + row;
            if (global_q < S) {
                for (int i = 0; i < 4; i++) {
                    uint16_t raw0 = r[i] & 0xFFFF;
                    uint16_t raw1 = r[i] >> 16;
                    __nv_bfloat16 bf0 = reinterpret_cast<__nv_bfloat16&>(raw0);
                    __nv_bfloat16 bf1 = reinterpret_cast<__nv_bfloat16&>(raw1);
                    float f0 = __bfloat162float(bf0);
                    float f1 = __bfloat162float(bf1);
                    
                    int d_idx0 = d_offset + (c + i) * 2 + 0;
                    int d_idx1 = d_offset + (c + i) * 2 + 1;
                    if (f0 != 0.0f) atomicAdd(&dQ_float[batch_H_S * d + global_q * d + d_idx0], f0);
                    if (f1 != 0.0f) atomicAdd(&dQ_float[batch_H_S * d + global_q * d + d_idx1], f1);
                }
            }
        }
    }
}

__device__ __forceinline__ void store_tmem_to_global(uint32_t tmem_col, __nv_bfloat16* out_ptr, int i_start, int S, int d_offset, int batch_H_S, int d) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int row = warp_id * 32 + lane_id;
    for (int c = 0; c < 32; c += 4) {
        uint32_t r[4];
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]) : "r"(tmem_col + c));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        if (row < 128) {
            int global_k = i_start + row;
            if (global_k < S) {
                for (int i = 0; i < 4; i++) {
                    int d_idx = d_offset + (c + i) * 2;
                    uint32_t val = r[i];
                    *(uint32_t*)&out_ptr[batch_H_S * d + global_k * d + d_idx] = val;
                }
            }
        }
    }
}

__global__ void bwd_kernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* O, const __nv_bfloat16* dO, const float* LSE,
    const float* D,
    float* dQ_float, __nv_bfloat16* dK, __nv_bfloat16* dV,
    int B_size, int H, int S, int d) 
{
    int i_start = blockIdx.x * 128;
    int head_idx = blockIdx.y;
    int batch_idx = blockIdx.z;
    if (i_start >= S) return;
    
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int row_in_tmem = warp_id * 32 + lane_id;
    
    __shared__ uint32_t dummy_tmem_addr;
    if (warp_id == 0) {
        uint32_t a = (uint32_t)__cvta_generic_to_shared(&dummy_tmem_addr);
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"(a) : "memory");
    }
    
    __shared__ alignas(1024) __nv_bfloat16 Q_smem[2][64 * 64];
    __shared__ alignas(1024) __nv_bfloat16 K_smem[2][128 * 64]; 
    __shared__ alignas(1024) __nv_bfloat16 V_smem[2][128 * 64];
    __shared__ alignas(1024) __nv_bfloat16 dO_smem[2][64 * 64];
    __shared__ alignas(1024) __nv_bfloat16 dS_smem[128 * 64];
    
    __shared__ float D_j[64]; 
    __shared__ float LSE_j[64]; 
    
    __shared__ alignas(8) uint64_t mbar[1];
    if (tid == 0) init_smem_barrier_fn(mbar, 1);
    
    const uint32_t TMEM_dV0 = 0, TMEM_dV1 = 32, TMEM_dK0 = 64, TMEM_dK1 = 96;
    const uint32_t TMEM_S   = 128, TMEM_P   = 160, TMEM_dP  = 192, TMEM_dS  = 224;
    const uint32_t TMEM_dQ0 = 256, TMEM_dQ1 = 288;
    
    for (int c = 0; c < 32; c += 4) {
        uint32_t z = 0;
        asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};" :: "r"(z),"r"(z),"r"(z),"r"(z), "r"(TMEM_dV0 + c) : "memory");
        asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};" :: "r"(z),"r"(z),"r"(z),"r"(z), "r"(TMEM_dV1 + c) : "memory");
        asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};" :: "r"(z),"r"(z),"r"(z),"r"(z), "r"(TMEM_dK0 + c) : "memory");
        asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};" :: "r"(z),"r"(z),"r"(z),"r"(z), "r"(TMEM_dK1 + c) : "memory");
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    }
    
    int batch_H_S = (batch_idx * H + head_idx) * S;
    load_gmem_to_smem_128x64(K, K_smem[0], batch_H_S, d, i_start, 0, S);
    load_gmem_to_smem_128x64(K, K_smem[1], batch_H_S, d, i_start, 64, S);
    load_gmem_to_smem_128x64(V, V_smem[0], batch_H_S, d, i_start, 0, S);
    load_gmem_to_smem_128x64(V, V_smem[1], batch_H_S, d, i_start, 64, S);
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    
    int j_start_align = (i_start / 64) * 64;
    int phase = 0;
    
    for (int j_start = j_start_align; j_start < S; j_start += 64) {
        load_gmem_to_smem_64x64(Q, Q_smem[0], batch_H_S, d, j_start, 0, S);
        load_gmem_to_smem_64x64(Q, Q_smem[1], batch_H_S, d, j_start, 64, S);
        load_gmem_to_smem_64x64(dO, dO_smem[0], batch_H_S, d, j_start, 0, S);
        load_gmem_to_smem_64x64(dO, dO_smem[1], batch_H_S, d, j_start, 64, S);
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        
        if (tid < 64) {
            int seq_idx = j_start + tid;
            if (seq_idx < S) {
                D_j[tid] = D[batch_H_S + seq_idx];
                LSE_j[tid] = LSE[batch_H_S + seq_idx];
            } else {
                D_j[tid] = 0.0f; LSE_j[tid] = 0.0f;
            }
        }
        asm volatile("cp.async.wait_all;\n" ::: "memory");
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        
        for (int k = 0; k < 64; k += 16) {
            uint64_t a0 = make_smem_desc_k_major(K_smem[0], k, 128);
            uint64_t b0 = make_smem_desc_k_major(Q_smem[0], k, 64);
            uint32_t id = make_instr_desc(128, 64, 0, 0); 
            umma_f16_cg1(TMEM_S, a0, b0, id, (k > 0));
        }
        for (int k = 0; k < 64; k += 16) {
            uint64_t a1 = make_smem_desc_k_major(K_smem[1], k, 128);
            uint64_t b1 = make_smem_desc_k_major(Q_smem[1], k, 64);
            uint32_t id = make_instr_desc(128, 64, 0, 0); 
            umma_f16_cg1(TMEM_S, a1, b1, id, 1);
        }
        wait_mma(mbar, phase); phase ^= 1;
        
        float attn_scale = 1.0f / sqrtf(128.0f);
        for (int c = 0; c < 32; c += 4) {
            uint32_t r[4];
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]) : "r"(TMEM_S + c));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            uint32_t out[4] = {0};
            for (int i = 0; i < 4; i++) {
                uint16_t raw0 = r[i] & 0xFFFF;
                uint16_t raw1 = r[i] >> 16;
                __nv_bfloat16 in_bf0 = reinterpret_cast<__nv_bfloat16&>(raw0);
                __nv_bfloat16 in_bf1 = reinterpret_cast<__nv_bfloat16&>(raw1);
                float f0 = __bfloat162float(in_bf0);
                float f1 = __bfloat162float(in_bf1);
                
                int q0 = (c + i) * 2 + 0; int q1 = (c + i) * 2 + 1;
                int g_q0 = j_start + q0;  int g_q1 = j_start + q1;
                int g_k = i_start + row_in_tmem;
                float p0 = 0.0f, p1 = 0.0f;
                if (row_in_tmem < 128) {
                    if (g_q0 >= g_k && g_q0 < S && g_k < S) p0 = expf(f0 * attn_scale - LSE_j[q0]);
                    if (g_q1 >= g_k && g_q1 < S && g_k < S) p1 = expf(f1 * attn_scale - LSE_j[q1]);
                }
                __nv_bfloat16 out_bf0 = __float2bfloat16(p0);
                __nv_bfloat16 out_bf1 = __float2bfloat16(p1);
                uint16_t out_raw0 = reinterpret_cast<uint16_t&>(out_bf0);
                uint16_t out_raw1 = reinterpret_cast<uint16_t&>(out_bf1);
                out[i] = ((uint32_t)out_raw1 << 16) | (uint32_t)out_raw0;
            }
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};"
                         :: "r"(out[0]),"r"(out[1]),"r"(out[2]),"r"(out[3]), "r"(TMEM_P + c) : "memory");
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        }
        
        for (int k = 0; k < 64; k += 16) {
            uint64_t a0 = make_smem_desc_k_major(V_smem[0], k, 128);
            uint64_t b0 = make_smem_desc_k_major(dO_smem[0], k, 64);
            uint32_t id = make_instr_desc(128, 64, 0, 0); 
            umma_f16_cg1(TMEM_dP, a0, b0, id, (k > 0));
        }
        for (int k = 0; k < 64; k += 16) {
            uint64_t a1 = make_smem_desc_k_major(V_smem[1], k, 128);
            uint64_t b1 = make_smem_desc_k_major(dO_smem[1], k, 64);
            uint32_t id = make_instr_desc(128, 64, 0, 0); 
            umma_f16_cg1(TMEM_dP, a1, b1, id, 1);
        }
        wait_mma(mbar, phase); phase ^= 1;
        
        for (int c = 0; c < 32; c += 4) {
            uint32_t p_r[4], dp_r[4];
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(p_r[0]),"=r"(p_r[1]),"=r"(p_r[2]),"=r"(p_r[3]) : "r"(TMEM_P + c));
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(dp_r[0]),"=r"(dp_r[1]),"=r"(dp_r[2]),"=r"(dp_r[3]) : "r"(TMEM_dP + c));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            uint32_t out[4] = {0};
            for (int i = 0; i < 4; i++) {
                uint16_t p_raw0 = p_r[i] & 0xFFFF;
                uint16_t p_raw1 = p_r[i] >> 16;
                __nv_bfloat16 in_p0 = reinterpret_cast<__nv_bfloat16&>(p_raw0);
                __nv_bfloat16 in_p1 = reinterpret_cast<__nv_bfloat16&>(p_raw1);
                float p0 = __bfloat162float(in_p0);
                float p1 = __bfloat162float(in_p1);

                uint16_t dp_raw0 = dp_r[i] & 0xFFFF;
                uint16_t dp_raw1 = dp_r[i] >> 16;
                __nv_bfloat16 in_dp0 = reinterpret_cast<__nv_bfloat16&>(dp_raw0);
                __nv_bfloat16 in_dp1 = reinterpret_cast<__nv_bfloat16&>(dp_raw1);
                float dp0 = __bfloat162float(in_dp0);
                float dp1 = __bfloat162float(in_dp1);
                
                int q0 = (c + i) * 2 + 0; int q1 = (c + i) * 2 + 1;
                float ds0 = 0.0f, ds1 = 0.0f;
                if (row_in_tmem < 128) {
                    ds0 = p0 * (dp0 - D_j[q0]) * attn_scale;
                    ds1 = p1 * (dp1 - D_j[q1]) * attn_scale;
                }
                __nv_bfloat16 out_bf0 = __float2bfloat16(ds0);
                __nv_bfloat16 out_bf1 = __float2bfloat16(ds1);
                uint16_t out_raw0 = reinterpret_cast<uint16_t&>(out_bf0);
                uint16_t out_raw1 = reinterpret_cast<uint16_t&>(out_bf1);
                out[i] = ((uint32_t)out_raw1 << 16) | (uint32_t)out_raw0;
            }
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};"
                         :: "r"(out[0]),"r"(out[1]),"r"(out[2]),"r"(out[3]), "r"(TMEM_dS + c) : "memory");
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        }
        
        store_tmem_to_smem(TMEM_dS, dS_smem, 128, 64);
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        
        for (int k = 0; k < 64; k += 16) {
            uint32_t a_tmem = TMEM_P + (k / 2); 
            uint64_t b0 = make_smem_desc_mn_major(dO_smem[0], k, 64);
            uint32_t id = make_instr_desc(128, 64, 0, 1);
            umma_f16_cg1_tmem_A(TMEM_dV0, a_tmem, b0, id, 1);
        }
        for (int k = 0; k < 64; k += 16) {
            uint32_t a_tmem = TMEM_P + (k / 2);
            uint64_t b1 = make_smem_desc_mn_major(dO_smem[1], k, 64);
            uint32_t id = make_instr_desc(128, 64, 0, 1);
            umma_f16_cg1_tmem_A(TMEM_dV1, a_tmem, b1, id, 1);
        }
        
        for (int k = 0; k < 64; k += 16) {
            uint32_t a_tmem = TMEM_dS + (k / 2);
            uint64_t b0 = make_smem_desc_mn_major(Q_smem[0], k, 64);
            uint32_t id = make_instr_desc(128, 64, 0, 1);
            umma_f16_cg1_tmem_A(TMEM_dK0, a_tmem, b0, id, 1);
        }
        for (int k = 0; k < 64; k += 16) {
            uint32_t a_tmem = TMEM_dS + (k / 2);
            uint64_t b1 = make_smem_desc_mn_major(Q_smem[1], k, 64);
            uint32_t id = make_instr_desc(128, 64, 0, 1);
            umma_f16_cg1_tmem_A(TMEM_dK1, a_tmem, b1, id, 1);
        }
        
        for (int k = 0; k < 128; k += 16) {
            uint64_t a0 = make_smem_desc_mn_major(dS_smem, k, 128);
            uint64_t b0 = make_smem_desc_mn_major(K_smem[0], k, 128);
            uint32_t id = make_instr_desc(64, 64, 1, 1);
            umma_f16_cg1(TMEM_dQ0, a0, b0, id, (k > 0));
        }
        for (int k = 0; k < 128; k += 16) {
            uint64_t a0 = make_smem_desc_mn_major(dS_smem, k, 128);
            uint64_t b1 = make_smem_desc_mn_major(K_smem[1], k, 128);
            uint32_t id = make_instr_desc(64, 64, 1, 1);
            umma_f16_cg1(TMEM_dQ1, a0, b1, id, (k > 0));
        }
        
        wait_mma(mbar, phase); phase ^= 1;
        
        atomic_add_dQ(TMEM_dQ0, dQ_float, i_start, j_start, S, 0, batch_H_S, d);
        atomic_add_dQ(TMEM_dQ1, dQ_float, i_start, j_start, S, 64, batch_H_S, d);
    }
    
    store_tmem_to_global(TMEM_dV0, dV, i_start, S, 0, batch_H_S, d);
    store_tmem_to_global(TMEM_dV1, dV, i_start, S, 64, batch_H_S, d);
    store_tmem_to_global(TMEM_dK0, dK, i_start, S, 0, batch_H_S, d);
    store_tmem_to_global(TMEM_dK1, dK, i_start, S, 64, batch_H_S, d);
    
    __syncthreads();
    if (warp_id == 0) {
        uint32_t tmem_ptr = dummy_tmem_addr;
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(tmem_ptr) : "memory");
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) 
{
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int d = Q.size(3);

    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_ptr = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_ptr = static_cast<const float*>(L.data_ptr());

    __nv_bfloat16* dQ_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());

    float* D_ptr;
    CUDA_CHECK(cudaMallocAsync(&D_ptr, B * H * S * sizeof(float), stream));

    float* dQ_float_ptr;
    CUDA_CHECK(cudaMallocAsync(&dQ_float_ptr, B * H * S * d * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(dQ_float_ptr, 0, B * H * S * d * sizeof(float), stream));

    dim3 grid_D((S + 127) / 128, H, B);
    dim3 block_D(128);
    precompute_D<<<grid_D, block_D, 0, stream>>>(O_ptr, dO_ptr, D_ptr, B, H, S, d);

    dim3 grid_Bwd((S + 127) / 128, H, B);
    dim3 block_Bwd(128);
    CUDA_CHECK(cudaFuncSetAttribute(bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 0));
    bwd_kernel<<<grid_Bwd, block_Bwd, 0, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, D_ptr, 
        dQ_float_ptr, dK_ptr, dV_ptr, B, H, S, d
    );

    int count = B * H * S * d;
    dim3 grid_Cvt((count + 255) / 256);
    dim3 block_Cvt(256);
    convert_dQ<<<grid_Cvt, block_Cvt, 0, stream>>>(dQ_float_ptr, dQ_ptr, count);

    CUDA_CHECK(cudaFreeAsync(D_ptr, stream));
    CUDA_CHECK(cudaFreeAsync(dQ_float_ptr, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128_causal::run);

}  // namespace mha_bwd_d128_causal