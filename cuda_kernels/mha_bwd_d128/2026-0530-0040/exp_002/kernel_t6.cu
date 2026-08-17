#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
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

namespace tvm_ffi_mha_bwd {

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

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
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

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void umma_commit_1sm_fn(uint64_t* bar) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        :: "l"((uint64_t)bar));
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, int trans_a, int trans_b) {
    uint32_t d = 0;
    d |= (1u << 4);    // c_format = FP32
    d |= (1u << 7);    // a_format = BF16
    d |= (1u << 10);   // b_format = BF16
    d |= ((uint32_t)trans_a << 15);   
    d |= ((uint32_t)trans_b << 16);   
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_kmajor_byte_offset(void* smem_ptr, uint32_t byte_offset) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr) + byte_offset;
    uint32_t base_offset = (addr >> 7) & 0x7;
    uint64_t d = 0;
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)(1) << 16;       
    d |= (uint64_t)(1024 >> 4) << 32; 
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)base_offset << 49;
    d |= (uint64_t)2 << 61; // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_mnmajor_byte_offset(void* smem_ptr, uint32_t byte_offset) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr) + byte_offset;
    uint32_t base_offset = (addr >> 7) & 0x7; 
    uint64_t d = 0;
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)(2048 >> 4) << 16;  
    d |= (uint64_t)(1024 >> 4) << 32;  
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)base_offset << 49;
    d |= (uint64_t)2 << 61; // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ void umma_f16_f32_accum(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void load_gmem_to_smem_swizzled(
    const __nv_bfloat16* gmem, void* smem_base, int max_idx_f4) {
    int tid = threadIdx.x;
    const float4* gmem_f4 = (const float4*)gmem;
    float4* smem_f4 = (float4*)smem_base;
    for (int i = 0; i < 16; ++i) { 
        int idx = i * 128 + tid; 
        int row = idx / 16;
        int col = idx % 16; 
        float4 val = (idx < max_idx_f4) ? gmem_f4[idx] : make_float4(0,0,0,0);
        int chunk_y = row % 8;
        int swizzle_x = (col & 8) | ((col & 7) ^ chunk_y);
        smem_f4[row * 16 + swizzle_x] = val;
    }
}

__device__ __forceinline__ void store_smem_swizzled_64b(
    void* smem_base, int row, int col_8bytes, uint32_t val0, uint32_t val1) {
    int col_16bytes = col_8bytes / 2;
    int half_idx = col_8bytes % 2;
    int chunk_y = row % 8;
    int swizzle_16b = (col_16bytes & 8) | ((col_16bytes & 7) ^ chunk_y);
    uint32_t byte_addr = (uint32_t)__cvta_generic_to_shared(smem_base) + (row * 16 + swizzle_16b) * 16 + half_idx * 8;
    asm volatile("st.shared.v2.b32 [%0], {%1, %2};" :: "r"(byte_addr), "r"(val0), "r"(val1) : "memory");
}

__device__ __forceinline__ void load_vector_to_smem(
    const float* gmem, float* smem, int max_idx) {
    int tid = threadIdx.x;
    if (tid < 128) {
        smem[tid] = (tid < max_idx) ? gmem[tid] : 0.0f;
    }
}

__global__ void compute_D_kernel(const __nv_bfloat16* O, const __nv_bfloat16* dO, float* D, int B, int H, int S) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * H * S) {
        float sum = 0;
        for (int i = 0; i < 128; ++i) {
            float o = __bfloat162float(O[idx * 128 + i]);
            float do_ = __bfloat162float(dO[idx * 128 + i]);
            sum += o * do_;
        }
        D[idx] = sum;
    }
}

union FloatBfloat162 {
    float f;
    __nv_bfloat162 b;
};

__global__ void mha_bwd_kernel(
    const __nv_bfloat16* Q,
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    const __nv_bfloat16* dO,
    const float* LSE,
    const float* D,
    __nv_bfloat16* dQ,
    __nv_bfloat16* dK,
    __nv_bfloat16* dV,
    int B, int H, int S) 
{
    int i = blockIdx.x; 
    int h = blockIdx.y;
    int b = blockIdx.z;
    int tid = threadIdx.x;

    extern __shared__ char smem[];
    float4* smem_K = (float4*)smem;
    float4* smem_V = smem_K + 2048;
    float4* smem_Q = smem_V + 2048;
    float4* smem_dO = smem_Q + 2048;
    float4* smem_P  = smem_dO + 2048;
    float4* smem_dS = smem_P + 2048;
    float* smem_LSE = (float*)(smem_dS + 2048);
    float* smem_D = smem_LSE + 128;
    
    uint64_t* mbar_umma = (uint64_t*)(smem_D + 128);
    uint32_t* smem_tmem_addr = (uint32_t*)(mbar_umma + 1);

    if (tid == 0) {
        init_smem_barrier_fn(mbar_umma, 1);
    }
    if (tid < 32) {
        tmem_alloc_fn(smem_tmem_addr, 512);
    }
    __syncthreads();
    
    uint32_t tmem_base = *smem_tmem_addr;
    uint32_t tmem_dV = tmem_base + 0;
    uint32_t tmem_dK = tmem_base + 128;
    uint32_t tmem_S  = tmem_base + 256;
    uint32_t tmem_dQ = tmem_base + 256;
    uint32_t tmem_dP = tmem_base + 384;

    int kv_start = i * 128;
    int kv_valid = (S - kv_start > 128) ? 128 : (S - kv_start);
    int kv_offset = b * H * S * 128 + h * S * 128 + kv_start * 128;
    
    load_gmem_to_smem_swizzled(K + kv_offset, smem_K, kv_valid * 16);
    load_gmem_to_smem_swizzled(V + kv_offset, smem_V, kv_valid * 16);

    __syncthreads();
    
    int phase = 0;
    float scale = 0.08838834764f; 
    
    for (int j = 0; j < (S + 127) / 128; ++j) {
        int q_start = j * 128;
        int q_valid = (S - q_start > 128) ? 128 : (S - q_start);
        int q_offset = b * H * S * 128 + h * S * 128 + q_start * 128;
        
        load_gmem_to_smem_swizzled(Q + q_offset, smem_Q, q_valid * 16);
        load_gmem_to_smem_swizzled(dO + q_offset, smem_dO, q_valid * 16);
        
        int vec_offset = b * H * S + h * S + q_start;
        load_vector_to_smem(LSE + vec_offset, smem_LSE, q_valid);
        load_vector_to_smem(D + vec_offset, smem_D, q_valid);
        
        __syncthreads();
        fence_async_shared_fn();
        
        // 1. S^T = K @ Q^T
        uint32_t idesc_S = make_instr_desc_fn(128, 128, 0, 0); 
        for (int k = 0; k < 128; k += 16) {
            uint64_t desc_a = make_smem_desc_kmajor_byte_offset(smem_K, k * 2);
            uint64_t desc_b = make_smem_desc_kmajor_byte_offset(smem_Q, k * 2);
            uint32_t accum = (k == 0) ? 0 : 1;
            umma_f16_f32_accum(tmem_S, desc_a, desc_b, idesc_S, accum); 
        }
        
        // 2. dP^T = V @ dO^T
        uint32_t idesc_dP = make_instr_desc_fn(128, 128, 0, 0);
        for (int k = 0; k < 128; k += 16) {
            uint64_t desc_a = make_smem_desc_kmajor_byte_offset(smem_V, k * 2);
            uint64_t desc_b = make_smem_desc_kmajor_byte_offset(smem_dO, k * 2);
            uint32_t accum = (k == 0) ? 0 : 1;
            umma_f16_f32_accum(tmem_dP, desc_a, desc_b, idesc_dP, accum); 
        }
        
        if (tid == 0) umma_commit_1sm_fn(mbar_umma);
        mbarrier_wait_fn(mbar_umma, phase);
        phase ^= 1;
        
        for (int c = 0; c < 32; ++c) {
            uint32_t col_S = tmem_S + c * 4;
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(col_S));
            
            uint32_t col_dP = tmem_dP + c * 4;
            uint32_t dp0, dp1, dp2, dp3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(dp0),"=r"(dp1),"=r"(dp2),"=r"(dp3) : "r"(col_dP));
            
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float lse0 = smem_LSE[c * 4 + 0];
            float lse1 = smem_LSE[c * 4 + 1];
            float lse2 = smem_LSE[c * 4 + 2];
            float lse3 = smem_LSE[c * 4 + 3];
            
            float D0 = smem_D[c * 4 + 0];
            float D1 = smem_D[c * 4 + 1];
            float D2 = smem_D[c * 4 + 2];
            float D3 = smem_D[c * 4 + 3];
            
            float s0 = __uint_as_float(r0);
            float s1 = __uint_as_float(r1);
            float s2 = __uint_as_float(r2);
            float s3 = __uint_as_float(r3);
            
            int q_idx0 = c * 4 + 0;
            int q_idx1 = c * 4 + 1;
            int q_idx2 = c * 4 + 2;
            int q_idx3 = c * 4 + 3;

            float p0 = fast_exp2f_fn((s0 * scale - lse0) * 1.44269504089f);
            float p1 = fast_exp2f_fn((s1 * scale - lse1) * 1.44269504089f);
            float p2 = fast_exp2f_fn((s2 * scale - lse2) * 1.44269504089f);
            float p3 = fast_exp2f_fn((s3 * scale - lse3) * 1.44269504089f);
            
            if (q_idx0 >= q_valid || tid >= kv_valid) p0 = 0.0f;
            if (q_idx1 >= q_valid || tid >= kv_valid) p1 = 0.0f;
            if (q_idx2 >= q_valid || tid >= kv_valid) p2 = 0.0f;
            if (q_idx3 >= q_valid || tid >= kv_valid) p3 = 0.0f;
            
            float dP0 = __uint_as_float(dp0);
            float dP1 = __uint_as_float(dp1);
            float dP2 = __uint_as_float(dp2);
            float dP3 = __uint_as_float(dp3);
            
            float ds0_f = p0 * (dP0 - D0) * scale;
            float ds1_f = p1 * (dP1 - D1) * scale;
            float ds2_f = p2 * (dP2 - D2) * scale;
            float ds3_f = p3 * (dP3 - D3) * scale;
            
            uint32_t p01 = pack_bf16_fn(__float_as_uint(p0), __float_as_uint(p1));
            uint32_t p23 = pack_bf16_fn(__float_as_uint(p2), __float_as_uint(p3));
            store_smem_swizzled_64b(smem_P, tid, c, p01, p23);
            
            uint32_t ds01 = pack_bf16_fn(__float_as_uint(ds0_f), __float_as_uint(ds1_f));
            uint32_t ds23 = pack_bf16_fn(__float_as_uint(ds2_f), __float_as_uint(ds3_f));
            store_smem_swizzled_64b(smem_dS, tid, c, ds01, ds23);
        }
        
        __syncthreads();
        fence_async_shared_fn();
        
        // 3. dV += P^T @ dO
        uint32_t idesc_dV = make_instr_desc_fn(128, 128, 0, 1);
        for (int k = 0; k < 128; k += 16) {
            uint64_t desc_a = make_smem_desc_kmajor_byte_offset(smem_P, k * 2);
            uint64_t desc_b = make_smem_desc_mnmajor_byte_offset(smem_dO, k * 256);
            uint32_t accum = (j == 0 && k == 0) ? 0 : 1;
            umma_f16_f32_accum(tmem_dV, desc_a, desc_b, idesc_dV, accum); 
        }
        
        // 4. dK += dS^T @ Q
        uint32_t idesc_dK = make_instr_desc_fn(128, 128, 1, 1);
        for (int k = 0; k < 128; k += 16) {
            uint64_t desc_a = make_smem_desc_mnmajor_byte_offset(smem_dS, k * 256);
            uint64_t desc_b = make_smem_desc_mnmajor_byte_offset(smem_Q, k * 256);
            uint32_t accum = (j == 0 && k == 0) ? 0 : 1;
            umma_f16_f32_accum(tmem_dK, desc_a, desc_b, idesc_dK, accum); 
        }
        
        // 5. dQ = dS @ K
        uint32_t idesc_dQ = make_instr_desc_fn(128, 128, 0, 1);
        for (int k = 0; k < 128; k += 16) {
            uint64_t desc_a = make_smem_desc_kmajor_byte_offset(smem_dS, k * 2);
            uint64_t desc_b = make_smem_desc_mnmajor_byte_offset(smem_K, k * 256);
            uint32_t accum = (k == 0) ? 0 : 1;
            umma_f16_f32_accum(tmem_dQ, desc_a, desc_b, idesc_dQ, accum); 
        }
        
        if (tid == 0) umma_commit_1sm_fn(mbar_umma);
        mbarrier_wait_fn(mbar_umma, phase);
        phase ^= 1;
        
        for (int c = 0; c < 32; ++c) {
            uint32_t col_dQ = tmem_dQ + c * 4;
            uint32_t q0, q1, q2, q3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(q0),"=r"(q1),"=r"(q2),"=r"(q3) : "r"(col_dQ));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            uint32_t q01 = pack_bf16_fn(q0, q1);
            uint32_t q23 = pack_bf16_fn(q2, q3);
            
            store_smem_swizzled_64b(smem_dS, tid, c, q01, q23);
        }
        
        __syncthreads();
        
        for (int i = 0; i < 16; ++i) { 
            int idx = i * 128 + tid; 
            int row = idx / 16;
            int col = idx % 16; 
            if (row < q_valid) {
                int chunk_y = row % 8;
                int swizzle_x = (col & 8) | ((col & 7) ^ chunk_y);
                float4 val = smem_dS[row * 16 + swizzle_x];
                
                FloatBfloat162 u0, u1, u2, u3;
                u0.f = val.x; u1.f = val.y; u2.f = val.z; u3.f = val.w;
                
                __nv_bfloat162* dq_ptr = (__nv_bfloat162*)&dQ[b * H * S * 128 + h * S * 128 + (q_start + row) * 128 + col * 8];
                atomicAdd(dq_ptr + 0, u0.b);
                atomicAdd(dq_ptr + 1, u1.b);
                atomicAdd(dq_ptr + 2, u2.b);
                atomicAdd(dq_ptr + 3, u3.b);
            }
        }
        __syncthreads();
    }
    
    for (int c = 0; c < 32; ++c) {
        uint32_t col_dV = tmem_dV + c * 4;
        uint32_t v0, v1, v2, v3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v0),"=r"(v1),"=r"(v2),"=r"(v3) : "r"(col_dV));
        
        uint32_t col_dK = tmem_dK + c * 4;
        uint32_t k0, k1, k2, k3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(k0),"=r"(k1),"=r"(k2),"=r"(k3) : "r"(col_dK));
        
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        uint32_t v01 = pack_bf16_fn(v0, v1);
        uint32_t v23 = pack_bf16_fn(v2, v3);
        store_smem_swizzled_64b(smem_P, tid, c, v01, v23);
        
        uint32_t k01 = pack_bf16_fn(k0, k1);
        uint32_t k23 = pack_bf16_fn(k2, k3);
        store_smem_swizzled_64b(smem_dS, tid, c, k01, k23);
    }
    
    __syncthreads();
    
    for (int i = 0; i < 16; ++i) { 
        int idx = i * 128 + tid; 
        int row = idx / 16;
        int col = idx % 16; 
        if (row < kv_valid) {
            int chunk_y = row % 8;
            int swizzle_x = (col & 8) | ((col & 7) ^ chunk_y);
            
            float4 val_v = smem_P[row * 16 + swizzle_x];
            float4 val_k = smem_dS[row * 16 + swizzle_x];
            
            int offset = b * H * S * 128 + h * S * 128 + (kv_start + row) * 128 + col * 8;
            
            ((float4*)&dV[0])[offset / 8] = val_v;
            ((float4*)&dK[0])[offset / 8] = val_k;
        }
    }
    
    __syncthreads();
    if (tid < 32) {
        tmem_dealloc_fn(tmem_base, 512);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);

    __nv_bfloat16* Q_ptr = static_cast<__nv_bfloat16*>(Q.data_ptr());
    __nv_bfloat16* K_ptr = static_cast<__nv_bfloat16*>(K.data_ptr());
    __nv_bfloat16* V_ptr = static_cast<__nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    __nv_bfloat16* dO_ptr = static_cast<__nv_bfloat16*>(dO.data_ptr());
    float* L_ptr = static_cast<float*>(L.data_ptr());
    
    __nv_bfloat16* dQ_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());

    CUDA_CHECK(cudaMemsetAsync(dQ_ptr, 0, B * H * S * 128 * sizeof(__nv_bfloat16), stream));

    float* D_ptr = nullptr;
    CUDA_CHECK(cudaMallocAsync(&D_ptr, B * H * S * sizeof(float), stream));

    int threads_D = 256;
    int blocks_D = (B * H * S + threads_D - 1) / threads_D;
    compute_D_kernel<<<blocks_D, threads_D, 0, stream>>>(O_ptr, dO_ptr, D_ptr, B, H, S);

    int num_kv_blocks = (S + 127) / 128;
    dim3 grid(num_kv_blocks, H, B);
    dim3 block(128);
    int smem_size = 198656;
    
    CUDA_CHECK(cudaFuncSetAttribute(mha_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    mha_bwd_kernel<<<grid, block, smem_size, stream>>>(Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_ptr, dQ_ptr, dK_ptr, dV_ptr, B, H, S);

    CUDA_CHECK(cudaFreeAsync(D_ptr, stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_bwd::run);

} // namespace tvm_ffi_mha_bwd