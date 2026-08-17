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

__device__ __forceinline__ uint64_t make_smem_desc_k_major_swizzled(void* smem_ptr, int k_offset_elements) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr) + k_offset_elements * 2;
    uint32_t base_offset = (addr >> 7) & 0x7;
    uint32_t lbo = 1;
    uint32_t sbo = 1024;
    uint64_t d = (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)base_offset << 49;
    d |= (uint64_t)2 << 61; // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_mn_major_swizzled(void* smem_ptr, int k_offset_rows, int N_dim) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr) + k_offset_rows * N_dim * 2;
    uint32_t base_offset = (addr >> 7) & 0x7;
    uint32_t sbo = 1024;
    uint32_t lbo = (N_dim / 8) * sbo;
    uint64_t d = (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)base_offset << 49;
    d |= (uint64_t)2 << 61; // SWIZZLE_128B
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
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"(mbar_ptr), "r"(phase) : "memory");
}

__device__ __forceinline__ void commit_mma(uint64_t* mbar) {
    uint32_t mbar_ptr = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(mbar_ptr) : "memory");
}

__device__ __forceinline__ void tma_load_2d(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(bar)),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void store_tmem_to_smem_swizzled_bf16(uint32_t tmem_col, void* smem_ptr, int rows, int cols_bf16) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int row = warp_id * 32 + lane_id;
    int cols_tmem = cols_bf16 / 2;
    uint8_t* smem = (uint8_t*)smem_ptr;
    
    for (int c = 0; c < cols_tmem; c += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_col + c));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        if (row < rows) {
            int x_chunk = (c * 4) / 16; 
            int y_mod_8 = row % 8;
            int swizzled_x_chunk = x_chunk ^ y_mod_8;
            int swizzled_byte_offset = swizzled_x_chunk * 16;
            
            int linear_row_offset = row * (cols_bf16 * 2);
            uint32_t* out = (uint32_t*)(smem + linear_row_offset + swizzled_byte_offset);
            
            out[0] = r0;
            out[1] = r1;
            out[2] = r2;
            out[3] = r3;
        }
    }
}

__device__ __forceinline__ void atomic_add_dQ(uint32_t tmem_col, float* dQ_float, int i_start, int j_start, int S, int d_offset, int batch_H_S, int d) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int row = warp_id * 32 + lane_id;
    for (int c = 0; c < 64; c += 4) { 
        uint32_t r[4];
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]) : "r"(tmem_col + c));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        if (row < 64) {
            int global_q = j_start + row;
            if (global_q < S) {
                float f0 = __uint_as_float(r[0]);
                float f1 = __uint_as_float(r[1]);
                float f2 = __uint_as_float(r[2]);
                float f3 = __uint_as_float(r[3]);
                
                if (f0 != 0.0f || f1 != 0.0f || f2 != 0.0f || f3 != 0.0f) {
                    int offset = batch_H_S * d + global_q * d + d_offset + c;
                    asm volatile(
                        "red.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
                        :: "l"(&dQ_float[offset]), "f"(f0), "f"(f1), "f"(f2), "f"(f3)
                        : "memory"
                    );
                }
            }
        }
    }
}

__device__ __forceinline__ void store_tmem_fp32_to_global_bf16(uint32_t tmem_col, __nv_bfloat16* out_ptr, int i_start, int S, int d_offset, int batch_H_S, int d) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int row = warp_id * 32 + lane_id;
    for (int c = 0; c < 64; c += 4) { 
        uint32_t r[4];
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]) : "r"(tmem_col + c));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        if (row < 128) {
            int global_k = i_start + row;
            if (global_k < S) {
                float f0 = __uint_as_float(r[0]);
                float f1 = __uint_as_float(r[1]);
                float f2 = __uint_as_float(r[2]);
                float f3 = __uint_as_float(r[3]);
                
                __nv_bfloat16 bf0 = __float2bfloat16(f0);
                __nv_bfloat16 bf1 = __float2bfloat16(f1);
                __nv_bfloat16 bf2 = __float2bfloat16(f2);
                __nv_bfloat16 bf3 = __float2bfloat16(f3);
                
                uint32_t out0 = ((uint32_t)reinterpret_cast<uint16_t&>(bf1) << 16) | reinterpret_cast<uint16_t&>(bf0);
                uint32_t out1 = ((uint32_t)reinterpret_cast<uint16_t&>(bf3) << 16) | reinterpret_cast<uint16_t&>(bf2);
                
                *(uint32_t*)(&out_ptr[batch_H_S * d + global_k * d + d_offset + c + 0]) = out0;
                *(uint32_t*)(&out_ptr[batch_H_S * d + global_k * d + d_offset + c + 2]) = out1;
            }
        }
    }
}

__global__ void bwd_kernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* O, const __nv_bfloat16* dO, const float* LSE,
    const float* D,
    float* dQ_float, __nv_bfloat16* dK, __nv_bfloat16* dV,
    int B_size, int H, int S, int d,
    const __grid_constant__ CUtensorMap tma_Q0, const __grid_constant__ CUtensorMap tma_Q1,
    const __grid_constant__ CUtensorMap tma_K0, const __grid_constant__ CUtensorMap tma_K1,
    const __grid_constant__ CUtensorMap tma_V0, const __grid_constant__ CUtensorMap tma_V1,
    const __grid_constant__ CUtensorMap tma_dO0, const __grid_constant__ CUtensorMap tma_dO1) 
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
    
    const uint32_t TMEM_dV0 = 0, TMEM_dV1 = 64, TMEM_dK0 = 128, TMEM_dK1 = 192;
    const uint32_t TMEM_S   = 256, TMEM_P   = 320, TMEM_dP  = 352, TMEM_dS  = 416;
    const uint32_t TMEM_dQ0 = 256, TMEM_dQ1 = 352; 
    
    for (int c = 0; c < 64; c += 4) {
        uint32_t z = 0;
        asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};" :: "r"(z),"r"(z),"r"(z),"r"(z), "r"(TMEM_dV0 + c) : "memory");
        asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};" :: "r"(z),"r"(z),"r"(z),"r"(z), "r"(TMEM_dV1 + c) : "memory");
        asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};" :: "r"(z),"r"(z),"r"(z),"r"(z), "r"(TMEM_dK0 + c) : "memory");
        asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};" :: "r"(z),"r"(z),"r"(z),"r"(z), "r"(TMEM_dK1 + c) : "memory");
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    }
    
    int batch_H_S = (batch_idx * H + head_idx) * S;
    int phase = 0;
    
    if (tid == 0) {
        uint32_t tx_bytes = (128 * 64 * 2) * 4; 
        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
            :: "r"((uint32_t)__cvta_generic_to_shared(mbar)), "r"(tx_bytes) : "memory");
        tma_load_2d(&tma_K0, mbar, K_smem[0], 0, batch_H_S + i_start);
        tma_load_2d(&tma_K1, mbar, K_smem[1], 64, batch_H_S + i_start);
        tma_load_2d(&tma_V0, mbar, V_smem[0], 0, batch_H_S + i_start);
        tma_load_2d(&tma_V1, mbar, V_smem[1], 64, batch_H_S + i_start);
    }
    if (tid == 0) wait_mma(mbar, phase & 1);
    __syncthreads();
    phase++;
    
    int j_start_align = (i_start / 64) * 64;
    
    for (int j_start = j_start_align; j_start < S; j_start += 64) {
        if (tid == 0) {
            uint32_t tx_bytes = (64 * 64 * 2) * 4;
            asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
                :: "r"((uint32_t)__cvta_generic_to_shared(mbar)), "r"(tx_bytes) : "memory");
            tma_load_2d(&tma_Q0, mbar, Q_smem[0], 0, batch_H_S + j_start);
            tma_load_2d(&tma_Q1, mbar, Q_smem[1], 64, batch_H_S + j_start);
            tma_load_2d(&tma_dO0, mbar, dO_smem[0], 0, batch_H_S + j_start);
            tma_load_2d(&tma_dO1, mbar, dO_smem[1], 64, batch_H_S + j_start);
        }
        
        if (tid < 64) {
            int seq_idx = j_start + tid;
            if (seq_idx < S) {
                D_j[tid] = D[batch_H_S + seq_idx];
                LSE_j[tid] = LSE[batch_H_S + seq_idx];
            } else {
                D_j[tid] = 0.0f; LSE_j[tid] = 0.0f;
            }
        }
        
        if (tid == 0) wait_mma(mbar, phase & 1);
        __syncthreads();
        phase++;
        
        if (tid == 0) {
            for (int k = 0; k < 64; k += 16) {
                uint64_t a0 = make_smem_desc_k_major_swizzled(K_smem[0], k);
                uint64_t b0 = make_smem_desc_k_major_swizzled(Q_smem[0], k);
                uint32_t id = make_instr_desc(128, 64, 0, 0); 
                umma_f16_cg1(TMEM_S, a0, b0, id, (k > 0)); 
            }
            for (int k = 0; k < 64; k += 16) {
                uint64_t a1 = make_smem_desc_k_major_swizzled(K_smem[1], k);
                uint64_t b1 = make_smem_desc_k_major_swizzled(Q_smem[1], k);
                uint32_t id = make_instr_desc(128, 64, 0, 0); 
                umma_f16_cg1(TMEM_S, a1, b1, id, 1);
            }
            commit_mma(mbar);
        }
        if (tid == 0) wait_mma(mbar, phase & 1);
        __syncthreads();
        phase++;
        
        float attn_scale = 1.0f / sqrtf(128.0f);
        for (int c = 0; c < 64; c += 4) {
            uint32_t r[4];
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]) : "r"(TMEM_S + c));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            uint32_t out[2] = {0, 0}; 
            for (int i = 0; i < 4; i+=2) {
                float f0 = __uint_as_float(r[i]);
                float f1 = __uint_as_float(r[i+1]);
                int q0 = c + i;
                int q1 = c + i + 1;
                int g_q0 = j_start + q0;  
                int g_q1 = j_start + q1;
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
                out[i/2] = ((uint32_t)out_raw1 << 16) | (uint32_t)out_raw0;
            }
            asm volatile("tcgen05.st.sync.aligned.32x32b.x2.b32 [%2], {%0,%1};"
                         :: "r"(out[0]),"r"(out[1]), "r"(TMEM_P + c/2) : "memory");
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        }
        
        if (tid == 0) {
            for (int k = 0; k < 64; k += 16) {
                uint64_t a0 = make_smem_desc_k_major_swizzled(V_smem[0], k);
                uint64_t b0 = make_smem_desc_mn_major_swizzled(dO_smem[0], k, 64);
                uint32_t id = make_instr_desc(128, 64, 0, 0); 
                umma_f16_cg1(TMEM_dP, a0, b0, id, (k > 0)); 
            }
            for (int k = 0; k < 64; k += 16) {
                uint64_t a1 = make_smem_desc_k_major_swizzled(V_smem[1], k);
                uint64_t b1 = make_smem_desc_mn_major_swizzled(dO_smem[1], k, 64);
                uint32_t id = make_instr_desc(128, 64, 0, 0); 
                umma_f16_cg1(TMEM_dP, a1, b1, id, 1);
            }
            commit_mma(mbar);
        }
        if (tid == 0) wait_mma(mbar, phase & 1);
        __syncthreads();
        phase++;
        
        for (int c = 0; c < 64; c += 4) { 
            uint32_t p_r[2], dp_r[4]; 
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x2.b32 {%0,%1}, [%2];"
                         : "=r"(p_r[0]),"=r"(p_r[1]) : "r"(TMEM_P + c/2));
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(dp_r[0]),"=r"(dp_r[1]),"=r"(dp_r[2]),"=r"(dp_r[3]) : "r"(TMEM_dP + c));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            uint32_t out[2] = {0, 0};
            for (int i = 0; i < 4; i+=2) { 
                uint16_t p_raw0 = p_r[i/2] & 0xFFFF;
                uint16_t p_raw1 = p_r[i/2] >> 16;
                float p0 = __bfloat162float(reinterpret_cast<__nv_bfloat16&>(p_raw0));
                float p1 = __bfloat162float(reinterpret_cast<__nv_bfloat16&>(p_raw1));

                float dp0 = __uint_as_float(dp_r[i]);
                float dp1 = __uint_as_float(dp_r[i+1]);
                
                int q0 = c + i; 
                int q1 = c + i + 1;
                float ds0 = 0.0f, ds1 = 0.0f;
                if (row_in_tmem < 128) {
                    ds0 = p0 * (dp0 - D_j[q0]) * attn_scale;
                    ds1 = p1 * (dp1 - D_j[q1]) * attn_scale;
                }
                __nv_bfloat16 out_bf0 = __float2bfloat16(ds0);
                __nv_bfloat16 out_bf1 = __float2bfloat16(ds1);
                uint16_t out_raw0 = reinterpret_cast<uint16_t&>(out_bf0);
                uint16_t out_raw1 = reinterpret_cast<uint16_t&>(out_bf1);
                out[i/2] = ((uint32_t)out_raw1 << 16) | (uint32_t)out_raw0;
            }
            asm volatile("tcgen05.st.sync.aligned.32x32b.x2.b32 [%2], {%0,%1};"
                         :: "r"(out[0]),"r"(out[1]), "r"(TMEM_dS + c/2) : "memory");
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        }
        
        store_tmem_to_smem_swizzled_bf16(TMEM_dS, dS_smem, 128, 64);
        __syncthreads();
        
        if (tid == 0) {
            for (int k = 0; k < 64; k += 16) {
                uint32_t a_tmem = TMEM_P + (k / 2); 
                uint64_t b0 = make_smem_desc_mn_major_swizzled(dO_smem[0], k, 64);
                uint32_t id = make_instr_desc(128, 64, 0, 1);
                umma_f16_cg1_tmem_A(TMEM_dV0, a_tmem, b0, id, 1);
            }
            for (int k = 0; k < 64; k += 16) {
                uint32_t a_tmem = TMEM_P + (k / 2);
                uint64_t b1 = make_smem_desc_mn_major_swizzled(dO_smem[1], k, 64);
                uint32_t id = make_instr_desc(128, 64, 0, 1);
                umma_f16_cg1_tmem_A(TMEM_dV1, a_tmem, b1, id, 1);
            }
            
            for (int k = 0; k < 64; k += 16) {
                uint32_t a_tmem = TMEM_dS + (k / 2);
                uint64_t b0 = make_smem_desc_mn_major_swizzled(Q_smem[0], k, 64);
                uint32_t id = make_instr_desc(128, 64, 0, 1);
                umma_f16_cg1_tmem_A(TMEM_dK0, a_tmem, b0, id, 1);
            }
            for (int k = 0; k < 64; k += 16) {
                uint32_t a_tmem = TMEM_dS + (k / 2);
                uint64_t b1 = make_smem_desc_mn_major_swizzled(Q_smem[1], k, 64);
                uint32_t id = make_instr_desc(128, 64, 0, 1);
                umma_f16_cg1_tmem_A(TMEM_dK1, a_tmem, b1, id, 1);
            }
            
            for (int k = 0; k < 128; k += 16) {
                uint64_t a0 = make_smem_desc_mn_major_swizzled(dS_smem, k, 64);
                uint64_t b0 = make_smem_desc_mn_major_swizzled(K_smem[0], k, 64);
                uint32_t id = make_instr_desc(64, 64, 1, 1);
                umma_f16_cg1(TMEM_dQ0, a0, b0, id, (k > 0));
            }
            for (int k = 0; k < 128; k += 16) {
                uint64_t a0 = make_smem_desc_mn_major_swizzled(dS_smem, k, 64);
                uint64_t b1 = make_smem_desc_mn_major_swizzled(K_smem[1], k, 64);
                uint32_t id = make_instr_desc(64, 64, 1, 1);
                umma_f16_cg1(TMEM_dQ1, a0, b1, id, (k > 0));
            }
            commit_mma(mbar);
        }
        
        if (tid == 0) wait_mma(mbar, phase & 1);
        __syncthreads();
        phase++;
        
        atomic_add_dQ(TMEM_dQ0, dQ_float, i_start, j_start, S, 0, batch_H_S, d);
        atomic_add_dQ(TMEM_dQ1, dQ_float, i_start, j_start, S, 64, batch_H_S, d);
    }
    
    store_tmem_fp32_to_global_bf16(TMEM_dV0, dV, i_start, S, 0, batch_H_S, d);
    store_tmem_fp32_to_global_bf16(TMEM_dV1, dV, i_start, S, 64, batch_H_S, d);
    store_tmem_fp32_to_global_bf16(TMEM_dK0, dK, i_start, S, 0, batch_H_S, d);
    store_tmem_fp32_to_global_bf16(TMEM_dK1, dK, i_start, S, 64, batch_H_S, d);
    
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

    CUtensorMap tma_Q0, tma_Q1, tma_K0, tma_K1, tma_V0, tma_V1, tma_dO0, tma_dO1;
    uint64_t globalDim[2] = {(uint64_t)d, (uint64_t)(B * H * S)};
    uint64_t globalStrides[1] = {(uint64_t)d * 2};
    uint32_t boxDim_64[2] = {64, 64}; 
    uint32_t boxDim_128[2] = {64, 128}; 
    uint32_t elemStrides[2] = {1, 1};

    cuTensorMapEncodeTiled(&tma_Q0, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)Q_ptr, globalDim, globalStrides, boxDim_64, elemStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    cuTensorMapEncodeTiled(&tma_Q1, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)Q_ptr, globalDim, globalStrides, boxDim_64, elemStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    cuTensorMapEncodeTiled(&tma_K0, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)K_ptr, globalDim, globalStrides, boxDim_128, elemStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    cuTensorMapEncodeTiled(&tma_K1, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)K_ptr, globalDim, globalStrides, boxDim_128, elemStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    cuTensorMapEncodeTiled(&tma_V0, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)V_ptr, globalDim, globalStrides, boxDim_128, elemStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    cuTensorMapEncodeTiled(&tma_V1, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)V_ptr, globalDim, globalStrides, boxDim_128, elemStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    cuTensorMapEncodeTiled(&tma_dO0, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)dO_ptr, globalDim, globalStrides, boxDim_64, elemStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    cuTensorMapEncodeTiled(&tma_dO1, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)dO_ptr, globalDim, globalStrides, boxDim_64, elemStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    dim3 grid_Bwd((S + 127) / 128, H, B);
    dim3 block_Bwd(128);
    CUDA_CHECK(cudaFuncSetAttribute(bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 0));
    bwd_kernel<<<grid_Bwd, block_Bwd, 0, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, D_ptr, 
        dQ_float_ptr, dK_ptr, dV_ptr, B, H, S, d,
        tma_Q0, tma_Q1, tma_K0, tma_K1, tma_V0, tma_V1, tma_dO0, tma_dO1
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