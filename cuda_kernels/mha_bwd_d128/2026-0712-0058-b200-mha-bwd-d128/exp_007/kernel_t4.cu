#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
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

// ---------------- PTX Wrappers ----------------

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

__device__ __forceinline__ uint32_t pack_bf16(uint32_t a, uint32_t b) {
    uint32_t res;
    asm volatile("mov.b32 %0, {%1, %2};" : "=r"(res) : "h"(a), "h"(b));
    return res;
}

__device__ __forceinline__ uint16_t bf16_from_uint32(uint32_t val) {
    __nv_bfloat16 res;
    uint32_t packed;
    asm volatile("mov.b32 %0, {%1, %2};" : "=r"(packed) : "h"(val), "h"(val));
    *reinterpret_cast<uint32_t*>(&res) = packed;
    return *reinterpret_cast<uint16_t*>(&res);
}

__global__ void cast_fp32_to_bf16(__nv_bfloat16* out, const float* in, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(in[idx]);
    }
}

// ---------------- Kernel ----------------

__global__ __launch_bounds__(128) void bwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    float* fp32_dQ, float* fp32_dK, float* fp32_dV,
    const float* L, uint32_t S, uint32_t d, uint32_t B, uint32_t H) 
{
    extern __shared__ __align__(128) uint8_t smem_raw[];
    char* smem_pool = (char*)smem_raw;
    
    uint32_t smem_offset = 0;
    uint32_t base_addr = (uint32_t)__cvta_generic_to_shared(smem_pool);
    uint32_t rem = base_addr % 1024;
    if (rem != 0) {
        smem_offset = 1024 - rem;
    }
    char* smem_aligned = smem_pool + smem_offset;
    
    uint32_t offset = 0;
    #define ALLOC_SMEM_128B(name, size) \
        name = (__nv_bfloat16*)(smem_aligned + offset); \
        offset += size; \
        offset = (offset + 1023) & ~1023;
    
    __nv_bfloat16 *smem_Q_0, *smem_Q_1, *smem_K_0, *smem_K_1, *smem_V_0, *smem_V_1;
    __nv_bfloat16 *smem_dO_0, *smem_dO_1, *smem_O_0, *smem_O_1, *smem_P, *smem_dS_0, *smem_dS_1;
    float* smem_D;
    uint64_t* mbar;
    
    ALLOC_SMEM_128B(smem_Q_0, 128 * 64 * sizeof(__nv_bfloat16));
    ALLOC_SMEM_128B(smem_Q_1, 128 * 64 * sizeof(__nv_bfloat16));
    ALLOC_SMEM_128B(smem_dO_0, 128 * 64 * sizeof(__nv_bfloat16));
    ALLOC_SMEM_128B(smem_dO_1, 128 * 64 * sizeof(__nv_bfloat16));
    ALLOC_SMEM_128B(smem_O_0, 128 * 64 * sizeof(__nv_bfloat16));
    ALLOC_SMEM_128B(smem_O_1, 128 * 64 * sizeof(__nv_bfloat16));
    ALLOC_SMEM_128B(smem_K_0, 128 * 64 * sizeof(__nv_bfloat16));
    ALLOC_SMEM_128B(smem_K_1, 128 * 64 * sizeof(__nv_bfloat16));
    ALLOC_SMEM_128B(smem_V_0, 128 * 64 * sizeof(__nv_bfloat16));
    ALLOC_SMEM_128B(smem_V_1, 128 * 64 * sizeof(__nv_bfloat16));
    ALLOC_SMEM_128B(smem_P, 128 * 128 * sizeof(__nv_bfloat16));
    ALLOC_SMEM_128B(smem_dS_0, 128 * 64 * sizeof(__nv_bfloat16));
    ALLOC_SMEM_128B(smem_dS_1, 128 * 64 * sizeof(__nv_bfloat16));
    
    offset = (offset + 3) & ~3;
    smem_D = (float*)(smem_aligned + offset);
    offset += 512;
    
    offset = (offset + 7) & ~7;
    mbar = (uint64_t*)(smem_aligned + offset);
    offset += 8;
    
    #undef ALLOC_SMEM_128B

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    uint32_t q_outer = gridDim.x * blockIdx.y + blockIdx.x;
    uint32_t q_tile = q_outer / (B * H);
    uint32_t b_idx = (q_outer % (B * H)) / H;
    uint32_t h_idx = q_outer % H;
    uint32_t num_kv_outer = (S + 127) / 128;
    uint32_t phase = 0;

    uint32_t seq_offset_q = b_idx * H * S + h_idx * S + q_tile * 128;
    uint32_t d_offset = b_idx * H * S * 128 + h_idx * S * 128;
    
    float* my_dQ = fp32_dQ + d_offset + q_tile * 128 * 128;
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar, 98304);
        tma_load_2d_fn(&tma_Q, mbar, smem_Q_0, 0, seq_offset_q);
        tma_load_2d_fn(&tma_Q, mbar, smem_Q_1, 64, seq_offset_q);
        tma_load_2d_fn(&tma_dO, mbar, smem_dO_0, 0, seq_offset_q);
        tma_load_2d_fn(&tma_dO, mbar, smem_dO_1, 64, seq_offset_q);
        tma_load_2d_fn(&tma_O, mbar, smem_O_0, 0, seq_offset_q);
        tma_load_2d_fn(&tma_O, mbar, smem_O_1, 64, seq_offset_q);
    }
    mbarrier_wait_fn(mbar, phase);
    phase ^= 1;
    
    float sum = 0;
    for (uint32_t i = 0; i < 8; i++) {
        uint4 do_vec0 = *(uint4*)&smem_dO_0[threadIdx.x * 64 + i * 8];
        uint4 o_vec0 = *(uint4*)&smem_O_0[threadIdx.x * 64 + i * 8];
        
        uint32_t q0 = do_vec0.x; float f0 = __bfloat162float(bf16_from_uint32((q0 >> 16) & 0xFFFF));
        uint32_t q1 = do_vec0.x; float f1 = __bfloat162float(bf16_from_uint32(q0 & 0xFFFF));
        uint32_t q2 = do_vec0.y; float f2 = __bfloat162float(bf16_from_uint32((q2 >> 16) & 0xFFFF));
        uint32_t q3 = do_vec0.y; float f3 = __bfloat162float(bf16_from_uint32(q2 & 0xFFFF));
        uint32_t q4 = do_vec0.z; float f4 = __bfloat162float(bf16_from_uint32((q4 >> 16) & 0xFFFF));
        uint32_t q5 = do_vec0.z; float f5 = __bfloat162float(bf16_from_uint32(q4 & 0xFFFF));
        uint32_t q6 = do_vec0.w; float f6 = __bfloat162float(bf16_from_uint32((q6 >> 16) & 0xFFFF));
        uint32_t q7 = do_vec0.w; float f7 = __bfloat162float(bf16_from_uint32(q6 & 0xFFFF));
        sum += f0 * __bfloat162float(bf16_from_uint32(((o_vec0.x >> 16) & 0xFFFF))) + 
               f1 * __bfloat162float(bf16_from_uint32(o_vec0.x & 0xFFFF)) + 
               f2 * __bfloat162float(bf16_from_uint32((o_vec0.y >> 16) & 0xFFFF)) + 
               f3 * __bfloat162float(bf16_from_uint32(o_vec0.y & 0xFFFF)) + 
               f4 * __bfloat162float(bf16_from_uint32((o_vec0.z >> 16) & 0xFFFF)) + 
               f5 * __bfloat162float(bf16_from_uint32(o_vec0.z & 0xFFFF)) + 
               f6 * __bfloat162float(bf16_from_uint32((o_vec0.w >> 16) & 0xFFFF)) + 
               f7 * __bfloat162float(bf16_from_uint32(o_vec0.w & 0xFFFF));
    }
    for (uint32_t i = 0; i < 8; i++) {
        uint4 do_vec1 = *(uint4*)&smem_dO_1[threadIdx.x * 64 + i * 8];
        uint4 o_vec1 = *(uint4*)&smem_O_1[threadIdx.x * 64 + i * 8];
        
        uint32_t q0 = do_vec1.x; float f0 = __bfloat162float(bf16_from_uint32((q0 >> 16) & 0xFFFF));
        uint32_t q1 = do_vec1.x; float f1 = __bfloat162float(bf16_from_uint32(q0 & 0xFFFF));
        uint32_t q2 = do_vec1.y; float f2 = __bfloat162float(bf16_from_uint32((q2 >> 16) & 0xFFFF));
        uint32_t q3 = do_vec1.y; float f3 = __bfloat162float(bf16_from_uint32(q2 & 0xFFFF));
        uint32_t q4 = do_vec1.z; float f4 = __bfloat162float(bf16_from_uint32((q4 >> 16) & 0xFFFF));
        uint32_t q5 = do_vec1.z; float f5 = __bfloat162float(bf16_from_uint32(q4 & 0xFFFF));
        uint32_t q6 = do_vec1.w; float f6 = __bfloat162float(bf16_from_uint32((q6 >> 16) & 0xFFFF));
        uint32_t q7 = do_vec1.w; float f7 = __bfloat162float(bf16_from_uint32(q6 & 0xFFFF));
        sum += f0 * __bfloat162float(bf16_from_uint32(((o_vec1.x >> 16) & 0xFFFF))) + 
               f1 * __bfloat162float(bf16_from_uint32(o_vec1.x & 0xFFFF)) + 
               f2 * __bfloat162float(bf16_from_uint32((o_vec1.y >> 16) & 0xFFFF)) + 
               f3 * __bfloat162float(bf16_from_uint32(o_vec1.y & 0xFFFF)) + 
               f4 * __bfloat162float(bf16_from_uint32((o_vec1.z >> 16) & 0xFFFF)) + 
               f5 * __bfloat162float(bf16_from_uint32(o_vec1.z & 0xFFFF)) + 
               f6 * __bfloat162float(bf16_from_uint32((o_vec1.w >> 16) & 0xFFFF)) + 
               f7 * __bfloat162float(bf16_from_uint32(o_vec1.w & 0xFFFF));
    }
    
    __shared__ float partial_D[128];
    partial_D[threadIdx.x] = sum;
    __syncthreads();
    if (threadIdx.x < 128) {
        smem_D[threadIdx.x] = partial_D[threadIdx.x];
    }
    __syncthreads();

    float dq_sum[32] = {0};
    bool first_k_block = true;
    const float scale = 0.08838834764831845f;

    for (uint32_t kv_outer = 0; kv_outer < num_kv_outer; kv_outer++) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar, 65536);
            uint32_t seq_offset_kv = b_idx * H * S + h_idx * S + kv_outer * 128;
            tma_load_2d_fn(&tma_K, mbar, smem_K_0, 0, seq_offset_kv);
            tma_load_2d_fn(&tma_K, mbar, smem_K_1, 64, seq_offset_kv);
            tma_load_2d_fn(&tma_V, mbar, smem_V_0, 0, seq_offset_kv);
            tma_load_2d_fn(&tma_V, mbar, smem_V_1, 64, seq_offset_kv);
        }
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        
        // Compute S = Q @ K^T and Softmax -> P
        for (int g = 0; g < 32; g++) {
            float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            
            for (int i = 0; i < 8; i++) {
                uint4 q_vec0 = *(uint4*)&smem_Q_0[(kv_outer * 128 + threadIdx.x) * 64 + i * 8];
                uint4 k_vec0 = *(uint4*)&smem_K_0[(g * 4 + 0) * 64 + i * 8];
                uint4 k_vec1 = *(uint4*)&smem_K_0[(g * 4 + 1) * 64 + i * 8];
                uint4 k_vec2 = *(uint4*)&smem_K_0[(g * 4 + 2) * 64 + i * 8];
                uint4 k_vec3 = *(uint4*)&smem_K_0[(g * 4 + 3) * 64 + i * 8];
                
                float f32_q0 = __bfloat162float(bf16_from_uint32((q_vec0.x >> 16) & 0xFFFF));
                float f32_q1 = __bfloat162float(bf16_from_uint32(q_vec0.x & 0xFFFF));
                float f32_q2 = __bfloat162float(bf16_from_uint32((q_vec0.y >> 16) & 0xFFFF));
                float f32_q3 = __bfloat162float(bf16_from_uint32(q_vec0.y & 0xFFFF));
                float f32_q4 = __bfloat162float(bf16_from_uint32((q_vec0.z >> 16) & 0xFFFF));
                float f32_q5 = __bfloat162float(bf16_from_uint32(q_vec0.z & 0xFFFF));
                float f32_q6 = __bfloat162float(bf16_from_uint32((q_vec0.w >> 16) & 0xFFFF));
                float f32_q7 = __bfloat162float(bf16_from_uint32(q_vec0.w & 0xFFFF));
                
                s0 += f32_q0 * __bfloat162float(bf16_from_uint32((k_vec0.x >> 16) & 0xFFFF)) + 
                      f32_q1 * __bfloat162float(bf16_from_uint32(k_vec0.x & 0xFFFF)) + 
                      f32_q2 * __bfloat162float(bf16_from_uint32((k_vec0.y >> 16) & 0xFFFF)) + 
                      f32_q3 * __bfloat162float(bf16_from_uint32(k_vec0.y & 0xFFFF)) + 
                      f32_q4 * __bfloat162float(bf16_from_uint32((k_vec0.z >> 16) & 0xFFFF)) + 
                      f32_q5 * __bfloat162float(bf16_from_uint32(k_vec0.z & 0xFFFF)) + 
                      f32_q6 * __bfloat162float(bf16_from_uint32((k_vec0.w >> 16) & 0xFFFF)) + 
                      f32_q7 * __bfloat162float(bf16_from_uint32(k_vec0.w & 0xFFFF));
                      
                s1 += f32_q0 * __bfloat162float(bf16_from_uint32((k_vec1.x >> 16) & 0xFFFF)) + 
                      f32_q1 * __bfloat162float(bf16_from_uint32(k_vec1.x & 0xFFFF)) + 
                      f32_q2 * __bfloat162float(bf16_from_uint32((k_vec1.y >> 16) & 0xFFFF)) + 
                      f32_q3 * __bfloat162float(bf16_from_uint32(k_vec1.y & 0xFFFF)) + 
                      f32_q4 * __bfloat162float(bf16_from_uint32((k_vec1.z >> 16) & 0xFFFF)) + 
                      f32_q5 * __bfloat162float(bf16_from_uint32(k_vec1.z & 0xFFFF)) + 
                      f32_q6 * __bfloat162float(bf16_from_uint32((k_vec1.w >> 16) & 0xFFFF)) + 
                      f32_q7 * __bfloat162float(bf16_from_uint32(k_vec1.w & 0xFFFF));
                      
                s2 += f32_q0 * __bfloat162float(bf16_from_uint32((k_vec2.x >> 16) & 0xFFFF)) + 
                      f32_q1 * __bfloat162float(bf16_from_uint32(k_vec2.x & 0xFFFF)) + 
                      f32_q2 * __bfloat162float(bf16_from_uint32((k_vec2.y >> 16) & 0xFFFF)) + 
                      f32_q3 * __bfloat162float(bf16_from_uint32(k_vec2.y & 0xFFFF)) + 
                      f32_q4 * __bfloat162float(bf16_from_uint32((k_vec2.z >> 16) & 0xFFFF)) + 
                      f32_q5 * __bfloat162float(bf16_from_uint32(k_vec2.z & 0xFFFF)) + 
                      f32_q6 * __bfloat162float(bf16_from_uint32((k_vec2.w >> 16) & 0xFFFF)) + 
                      f32_q7 * __bfloat162float(bf16_from_uint32(k_vec2.w & 0xFFFF));
                      
                s3 += f32_q0 * __bfloat162float(bf16_from_uint32((k_vec3.x >> 16) & 0xFFFF)) + 
                      f32_q1 * __bfloat162float(bf16_from_uint32(k_vec3.x & 0xFFFF)) + 
                      f32_q2 * __bfloat162float(bf16_from_uint32((k_vec3.y >> 16) & 0xFFFF)) + 
                      f32_q3 * __bfloat162float(bf16_from_uint32(k_vec3.y & 0xFFFF)) + 
                      f32_q4 * __bfloat162float(bf16_from_uint32((k_vec3.z >> 16) & 0xFFFF)) + 
                      f32_q5 * __bfloat162float(bf16_from_uint32(k_vec3.z & 0xFFFF)) + 
                      f32_q6 * __bfloat162float(bf16_from_uint32((k_vec3.w >> 16) & 0xFFFF)) + 
                      f32_q7 * __bfloat162float(bf16_from_uint32(k_vec3.w & 0xFFFF));
            }
            
            for (int i = 0; i < 8; i++) {
                uint4 q_vec1 = *(uint4*)&smem_Q_1[(kv_outer * 128 + threadIdx.x) * 64 + i * 8];
                uint4 k_vec0 = *(uint4*)&smem_K_1[(g * 4 + 0) * 64 + i * 8];
                uint4 k_vec1 = *(uint4*)&smem_K_1[(g * 4 + 1) * 64 + i * 8];
                uint4 k_vec2 = *(uint4*)&smem_K_1[(g * 4 + 2) * 64 + i * 8];
                uint4 k_vec3 = *(uint4*)&smem_K_1[(g * 4 + 3) * 64 + i * 8];
                
                float f32_q0 = __bfloat162float(bf16_from_uint32((q_vec1.x >> 16) & 0xFFFF));
                float f32_q1 = __bfloat162float(bf16_from_uint32(q_vec1.x & 0xFFFF));
                float f32_q2 = __bfloat162float(bf16_from_uint32((q_vec1.y >> 16) & 0xFFFF));
                float f32_q3 = __bfloat162float(bf16_from_uint32(q_vec1.y & 0xFFFF));
                float f32_q4 = __bfloat162float(bf16_from_uint32((q_vec1.z >> 16) & 0xFFFF));
                float f32_q5 = __bfloat162float(bf16_from_uint32(q_vec1.z & 0xFFFF));
                float f32_q6 = __bfloat162float(bf16_from_uint32((q_vec1.w >> 16) & 0xFFFF));
                float f32_q7 = __bfloat162float(bf16_from_uint32(q_vec1.w & 0xFFFF));
                
                s0 += f32_q0 * __bfloat162float(bf16_from_uint32((k_vec0.x >> 16) & 0xFFFF)) + 
                      f32_q1 * __bfloat162float(bf16_from_uint32(k_vec0.x & 0xFFFF)) + 
                      f32_q2 * __bfloat162float(bf16_from_uint32((k_vec0.y >> 16) & 0xFFFF)) + 
                      f32_q3 * __bfloat162float(bf16_from_uint32(k_vec0.y & 0xFFFF)) + 
                      f32_q4 * __bfloat162float(bf16_from_uint32((k_vec0.z >> 16) & 0xFFFF)) + 
                      f32_q5 * __bfloat162float(bf16_from_uint32(k_vec0.z & 0xFFFF)) + 
                      f32_q6 * __bfloat162float(bf16_from_uint32((k_vec0.w >> 16) & 0xFFFF)) + 
                      f32_q7 * __bfloat162float(bf16_from_uint32(k_vec0.w & 0xFFFF));
                      
                s1 += f32_q0 * __bfloat162float(bf16_from_uint32((k_vec1.x >> 16) & 0xFFFF)) + 
                      f32_q1 * __bfloat162float(bf16_from_uint32(k_vec1.x & 0xFFFF)) + 
                      f32_q2 * __bfloat162float(bf16_from_uint32((k_vec1.y >> 16) & 0xFFFF)) + 
                      f32_q3 * __bfloat162float(bf16_from_uint32(k_vec1.y & 0xFFFF)) + 
                      f32_q4 * __bfloat162float(bf16_from_uint32((k_vec1.z >> 16) & 0xFFFF)) + 
                      f32_q5 * __bfloat162float(bf16_from_uint32(k_vec1.z & 0xFFFF)) + 
                      f32_q6 * __bfloat162float(bf16_from_uint32((k_vec1.w >> 16) & 0xFFFF)) + 
                      f32_q7 * __bfloat162float(bf16_from_uint32(k_vec1.w & 0xFFFF));
                      
                s2 += f32_q0 * __bfloat162float(bf16_from_uint32((k_vec2.x >> 16) & 0xFFFF)) + 
                      f32_q1 * __bfloat162float(bf16_from_uint32(k_vec2.x & 0xFFFF)) + 
                      f32_q2 * __bfloat162float(bf16_from_uint32((k_vec2.y >> 16) & 0xFFFF)) + 
                      f32_q3 * __bfloat162float(bf16_from_uint32(k_vec2.y & 0xFFFF)) + 
                      f32_q4 * __bfloat162float(bf16_from_uint32((k_vec2.z >> 16) & 0xFFFF)) + 
                      f32_q5 * __bfloat162float(bf16_from_uint32(k_vec2.z & 0xFFFF)) + 
                      f32_q6 * __bfloat162float(bf16_from_uint32((k_vec2.w >> 16) & 0xFFFF)) + 
                      f32_q7 * __bfloat162float(bf16_from_uint32(k_vec2.w & 0xFFFF));
                      
                s3 += f32_q0 * __bfloat162float(bf16_from_uint32((k_vec3.x >> 16) & 0xFFFF)) + 
                      f32_q1 * __bfloat162float(bf16_from_uint32(k_vec3.x & 0xFFFF)) + 
                      f32_q2 * __bfloat162float(bf16_from_uint32((k_vec3.y >> 16) & 0xFFFF)) + 
                      f32_q3 * __bfloat162float(bf16_from_uint32(k_vec3.y & 0xFFFF)) + 
                      f32_q4 * __bfloat162float(bf16_from_uint32((k_vec3.z >> 16) & 0xFFFF)) + 
                      f32_q5 * __bfloat162float(bf16_from_uint32(k_vec3.z & 0xFFFF)) + 
                      f32_q6 * __bfloat162float(bf16_from_uint32((k_vec3.w >> 16) & 0xFFFF)) + 
                      f32_q7 * __bfloat162float(bf16_from_uint32(k_vec3.w & 0xFFFF));
            }
            
            float LSE_val = (q_tile * 128 + threadIdx.x < S) ? L[b_idx * H * S + h_idx * S + q_tile * 128 + threadIdx.x] : 0.0f;
            float p0 = expf(s0 * scale - LSE_val);
            float p1 = expf(s1 * scale - LSE_val);
            float p2 = expf(s2 * scale - LSE_val);
            float p3 = expf(s3 * scale - LSE_val);
            
            uint32_t packed0 = pack_bf16(*(uint16_t*)&__float2bfloat16(p0), *(uint16_t*)&__float2bfloat16(p1));
            uint32_t packed1 = pack_bf16(*(uint16_t*)&__float2bfloat16(p2), *(uint16_t*)&__float2bfloat16(p3));
            *(uint32_t*)&smem_P[(kv_outer * 128 + threadIdx.x) * 128 + g * 4] = packed0;
            *(uint32_t*)&smem_P[(kv_outer * 128 + threadIdx.x) * 128 + g * 4 + 2] = packed1;
        }
        
        // Compute dP = V @ dO^T
        for (int g = 0; g < 32; g++) {
            float dp0 = 0, dp1 = 0, dp2 = 0, dp3 = 0;
            
            for (int i = 0; i < 8; i++) {
                uint4 do_vec0 = *(uint4*)&smem_dO_0[(kv_outer * 128 + threadIdx.x) * 64 + i * 8];
                uint4 v_vec0 = *(uint4*)&smem_V_0[(g * 4 + 0) * 64 + i * 8];
                uint4 v_vec1 = *(uint4*)&smem_V_0[(g * 4 + 1) * 64 + i * 8];
                uint4 v_vec2 = *(uint4*)&smem_V_0[(g * 4 + 2) * 64 + i * 8];
                uint4 v_vec3 = *(uint4*)&smem_V_0[(g * 4 + 3) * 64 + i * 8];
                
                float f32_do0 = __bfloat162float(bf16_from_uint32((do_vec0.x >> 16) & 0xFFFF));
                float f32_do1 = __bfloat162float(bf16_from_uint32(do_vec0.x & 0xFFFF));
                float f32_do2 = __bfloat162float(bf16_from_uint32((do_vec0.y >> 16) & 0xFFFF));
                float f32_do3 = __bfloat162float(bf16_from_uint32(do_vec0.y & 0xFFFF));
                float f32_do4 = __bfloat162float(bf16_from_uint32((do_vec0.z >> 16) & 0xFFFF));
                float f32_do5 = __bfloat162float(bf16_from_uint32(do_vec0.z & 0xFFFF));
                float f32_do6 = __bfloat162float(bf16_from_uint32((do_vec0.w >> 16) & 0xFFFF));
                float f32_do7 = __bfloat162float(bf16_from_uint32(do_vec0.w & 0xFFFF));
                
                dp0 += f32_do0 * __bfloat162float(bf16_from_uint32((v_vec0.x >> 16) & 0xFFFF)) + 
                       f32_do1 * __bfloat162float(bf16_from_uint32(v_vec0.x & 0xFFFF)) + 
                       f32_do2 * __bfloat162float(bf16_from_uint32((v_vec0.y >> 16) & 0xFFFF)) + 
                       f32_do3 * __bfloat162float(bf16_from_uint32(v_vec0.y & 0xFFFF)) + 
                       f32_do4 * __bfloat162float(bf16_from_uint32((v_vec0.z >> 16) & 0xFFFF)) + 
                       f32_do5 * __bfloat162float(bf16_from_uint32(v_vec0.z & 0xFFFF)) + 
                       f32_do6 * __bfloat162float(bf16_from_uint32((v_vec0.w >> 16) & 0xFFFF)) + 
                       f32_do7 * __bfloat162float(bf16_from_uint32(v_vec0.w & 0xFFFF));
                       
                dp1 += f32_do0 * __bfloat162float(bf16_from_uint32((v_vec1.x >> 16) & 0xFFFF)) + 
                       f32_do1 * __bfloat162float(bf16_from_uint32(v_vec1.x & 0xFFFF)) + 
                       f32_do2 * __bfloat162float(bf16_from_uint32((v_vec1.y >> 16) & 0xFFFF)) + 
                       f32_do3 * __bfloat162float(bf16_from_uint32(v_vec1.y & 0xFFFF)) + 
                       f32_do4 * __bfloat162float(bf16_from_uint32((v_vec1.z >> 16) & 0xFFFF)) + 
                       f32_do5 * __bfloat162float(bf16_from_uint32(v_vec1.z & 0xFFFF)) + 
                       f32_do6 * __bfloat162float(bf16_from_uint32((v_vec1.w >> 16) & 0xFFFF)) + 
                       f32_do7 * __bfloat162float(bf16_from_uint32(v_vec1.w & 0xFFFF));
                       
                dp2 += f32_do0 * __bfloat162float(bf16_from_uint32((v_vec2.x >> 16) & 0xFFFF)) + 
                       f32_do1 * __bfloat162float(bf16_from_uint32(v_vec2.x & 0xFFFF)) + 
                       f32_do2 * __bfloat162float(bf16_from_uint32((v_vec2.y >> 16) & 0xFFFF)) + 
                       f32_do3 * __bfloat162float(bf16_from_uint32(v_vec2.y & 0xFFFF)) + 
                       f32_do4 * __bfloat162float(bf16_from_uint32((v_vec2.z >> 16) & 0xFFFF)) + 
                       f32_do5 * __bfloat162float(bf16_from_uint32(v_vec2.z & 0xFFFF)) + 
                       f32_do6 * __bfloat162float(bf16_from_uint32((v_vec2.w >> 16) & 0xFFFF)) + 
                       f32_do7 * __bfloat162float(bf16_from_uint32(v_vec2.w & 0xFFFF));
                       
                dp3 += f32_do0 * __bfloat162float(bf16_from_uint32((v_vec3.x >> 16) & 0xFFFF)) + 
                       f32_do1 * __bfloat162float(bf16_from_uint32(v_vec3.x & 0xFFFF)) + 
                       f32_do2 * __bfloat162float(bf16_from_uint32((v_vec3.y >> 16) & 0xFFFF)) + 
                       f32_do3 * __bfloat162float(bf16_from_uint32(v_vec3.y & 0xFFFF)) + 
                       f32_do4 * __bfloat162float(bf16_from_uint32((v_vec3.z >> 16) & 0xFFFF)) + 
                       f32_do5 * __bfloat162float(bf16_from_uint32(v_vec3.z & 0xFFFF)) + 
                       f32_do6 * __bfloat162float(bf16_from_uint32((v_vec3.w >> 16) & 0xFFFF)) + 
                       f32_do7 * __bfloat162float(bf16_from_uint32(v_vec3.w & 0xFFFF));
            }
            
            for (int i = 0; i < 8; i++) {
                uint4 do_vec1 = *(uint4*)&smem_dO_1[(kv_outer * 128 + threadIdx.x) * 64 + i * 8];
                uint4 v_vec0 = *(uint4*)&smem_V_1[(g * 4 + 0) * 64 + i * 8];
                uint4 v_vec1 = *(uint4*)&smem_V_1[(g * 4 + 1) * 64 + i * 8];
                uint4 v_vec2 = *(uint4*)&smem_V_1[(g * 4 + 2) * 64 + i * 8];
                uint4 v_vec3 = *(uint4*)&smem_V_1[(g * 4 + 3) * 64 + i * 8];
                
                float f32_do0 = __bfloat162float(bf16_from_uint32((do_vec1.x >> 16) & 0xFFFF));
                float f32_do1 = __bfloat162float(bf16_from_uint32(do_vec1.x & 0xFFFF));
                float f32_do2 = __bfloat162float(bf16_from_uint32((do_vec1.y >> 16) & 0xFFFF));
                float f32_do3 = __bfloat162float(bf16_from_uint32(do_vec1.y & 0xFFFF));
                float f32_do4 = __bfloat162float(bf16_from_uint32((do_vec1.z >> 16) & 0xFFFF));
                float f32_do5 = __bfloat162float(bf16_from_uint32(do_vec1.z & 0xFFFF));
                float f32_do6 = __bfloat162float(bf16_from_uint32((do_vec1.w >> 16) & 0xFFFF));
                float f32_do7 = __bfloat162float(bf16_from_uint32(do_vec1.w & 0xFFFF));
                
                dp0 += f32_do0 * __bfloat162float(bf16_from_uint32((v_vec0.x >> 16) & 0xFFFF)) + 
                       f32_do1 * __bfloat162float(bf16_from_uint32(v_vec0.x & 0xFFFF)) + 
                       f32_do2 * __bfloat162float(bf16_from_uint32((v_vec0.y >> 16) & 0xFFFF)) + 
                       f32_do3 * __bfloat162float(bf16_from_uint32(v_vec0.y & 0xFFFF)) + 
                       f32_do4 * __bfloat162float(bf16_from_uint32((v_vec0.z >> 16) & 0xFFFF)) + 
                       f32_do5 * __bfloat162float(bf16_from_uint32(v_vec0.z & 0xFFFF)) + 
                       f32_do6 * __bfloat162float(bf16_from_uint32((v_vec0.w >> 16) & 0xFFFF)) + 
                       f32_do7 * __bfloat162float(bf16_from_uint32(v_vec0.w & 0xFFFF));
                       
                dp1 += f32_do0 * __bfloat162float(bf16_from_uint32((v_vec1.x >> 16) & 0xFFFF)) + 
                       f32_do1 * __bfloat162float(bf16_from_uint32(v_vec1.x & 0xFFFF)) + 
                       f32_do2 * __bfloat162float(bf16_from_uint32((v_vec1.y >> 16) & 0xFFFF)) + 
                       f32_do3 * __bfloat162float(bf16_from_uint32(v_vec1.y & 0xFFFF)) + 
                       f32_do4 * __bfloat162float(bf16_from_uint32((v_vec1.z >> 16) & 0xFFFF)) + 
                       f32_do5 * __bfloat162float(bf16_from_uint32(v_vec1.z & 0xFFFF)) + 
                       f32_do6 * __bfloat162float(bf16_from_uint32((v_vec1.w >> 16) & 0xFFFF)) + 
                       f32_do7 * __bfloat162float(bf16_from_uint32(v_vec1.w & 0xFFFF));
                       
                dp2 += f32_do0 * __bfloat162float(bf16_from_uint32((v_vec2.x >> 16) & 0xFFFF)) + 
                       f32_do1 * __bfloat162float(bf16_from_uint32(v_vec2.x & 0xFFFF)) + 
                       f32_do2 * __bfloat162float(bf16_from_uint32((v_vec2.y >> 16) & 0xFFFF)) + 
                       f32_do3 * __bfloat162float(bf16_from_uint32(v_vec2.y & 0xFFFF)) + 
                       f32_do4 * __bfloat162float(bf16_from_uint32((v_vec2.z >> 16) & 0xFFFF)) + 
                       f32_do5 * __bfloat162float(bf16_from_uint32(v_vec2.z & 0xFFFF)) + 
                       f32_do6 * __bfloat162float(bf16_from_uint32((v_vec2.w >> 16) & 0xFFFF)) + 
                       f32_do7 * __bfloat162float(bf16_from_uint32(v_vec2.w & 0xFFFF));
                       
                dp3 += f32_do0 * __bfloat162float(bf16_from_uint32((v_vec3.x >> 16) & 0xFFFF)) + 
                       f32_do1 * __bfloat162float(bf16_from_uint32(v_vec3.x & 0xFFFF)) + 
                       f32_do2 * __bfloat162float(bf16_from_uint32((v_vec3.y >> 16) & 0xFFFF)) + 
                       f32_do3 * __bfloat162float(bf16_from_uint32(v_vec3.y & 0xFFFF)) + 
                       f32_do4 * __bfloat162float(bf16_from_uint32((v_vec3.z >> 16) & 0xFFFF)) + 
                       f32_do5 * __bfloat162float(bf16_from_uint32(v_vec3.z & 0xFFFF)) + 
                       f32_do6 * __bfloat162float(bf16_from_uint32((v_vec3.w >> 16) & 0xFFFF)) + 
                       f32_do7 * __bfloat162float(bf16_from_uint32(v_vec3.w & 0xFFFF));
            }
            
            float p0 = __bfloat162float(*(__nv_bfloat16*)&smem_P[(kv_outer * 128 + threadIdx.x) * 128 + g * 4]);
            float p1 = __bfloat162float(*(__nv_bfloat16*)&smem_P[(kv_outer * 128 + threadIdx.x) * 128 + g * 4 + 1]);
            float p2 = __bfloat162float(*(__nv_bfloat16*)&smem_P[(kv_outer * 128 + threadIdx.x) * 128 + g * 4 + 2]);
            float p3 = __bfloat162float(*(__nv_bfloat16*)&smem_P[(kv_outer * 128 + threadIdx.x) * 128 + g * 4 + 3]);
            
            float ds0 = p0 * (dp0 - smem_D[kv_outer * 128 + threadIdx.x]);
            float ds1 = p1 * (dp1 - smem_D[kv_outer * 128 + threadIdx.x]);
            float ds2 = p2 * (dp2 - smem_D[kv_outer * 128 + threadIdx.x]);
            float ds3 = p3 * (dp3 - smem_D[kv_outer * 128 + threadIdx.x]);
            
            uint32_t packed0 = pack_bf16(*(uint16_t*)&__float2bfloat16(ds0), *(uint16_t*)&__float2bfloat16(ds1));
            uint32_t packed1 = pack_bf16(*(uint16_t*)&__float2bfloat16(ds2), *(uint16_t*)&__float2bfloat16(ds3));
            
            int buf_idx = (g * 4) / 64;
            int elem_idx = (g * 4) % 64;
            *(uint32_t*)&smem_dS_0[(kv_outer * 128 + threadIdx.x) * 64 + elem_idx] = packed0;
            *(uint32_t*)&smem_dS_0[(kv_outer * 128 + threadIdx.x) * 64 + elem_idx + 2] = packed1;
            *(uint32_t*)&smem_dS_1[(kv_outer * 128 + threadIdx.x) * 64 + elem_idx] = packed0;
            *(uint32_t*)&smem_dS_1[(kv_outer * 128 + threadIdx.x) * 64 + elem_idx + 2] = packed1;
        }
        
        // Accumulate dQ += dS @ K locally using highly optimized scalar math
        for (int g = 0; g < 32; g++) {
            float ds0 = 0, ds1 = 0, ds2 = 0, ds3 = 0;
            float ds4 = 0, ds5 = 0, ds6 = 0, ds7 = 0;
            
            if (g < 16) {
                uint4 ds_vec0 = *(uint4*)&smem_dS_0[(kv_outer * 128 + threadIdx.x) * 64 + g * 8];
                uint4 ds_vec1 = *(uint4*)&smem_dS_1[(kv_outer * 128 + threadIdx.x) * 64 + g * 8];
                
                uint32_t q0 = ds_vec0.x; float f32_ds0 = __bfloat162float(bf16_from_uint32((q0 >> 16) & 0xFFFF));
                uint32_t q1 = ds_vec0.x; float f32_ds1 = __bfloat162float(bf16_from_uint32(q0 & 0xFFFF));
                uint32_t q2 = ds_vec0.y; float f32_ds2 = __bfloat162float(bf16_from_uint32((q2 >> 16) & 0xFFFF));
                uint32_t q3 = ds_vec0.y; float f32_ds3 = __bfloat162float(bf16_from_uint32(q2 & 0xFFFF));
                
                uint32_t q4 = ds_vec1.x; float f32_ds4 = __bfloat162float(bf16_from_uint32((q4 >> 16) & 0xFFFF));
                uint32_t q5 = ds_vec1.x; float f32_ds5 = __bfloat162float(bf16_from_uint32(q4 & 0xFFFF));
                uint32_t q6 = ds_vec1.y; float f32_ds6 = __bfloat162float(bf16_from_uint32((q6 >> 16) & 0xFFFF));
                uint32_t q7 = ds_vec1.y; float f32_ds7 = __bfloat162float(bf16_from_uint32(q6 & 0xFFFF));
                
                ds0 = f32_ds0; ds1 = f32_ds1; ds2 = f32_ds2; ds3 = f32_ds3;
                ds4 = f32_ds4; ds5 = f32_ds5; ds6 = f32_ds6; ds7 = f32_ds7;
            } else {
                uint4 ds_vec0 = *(uint4*)&smem_dS_0[(kv_outer * 128 + threadIdx.x) * 64 + (g - 16) * 8];
                uint4 ds_vec1 = *(uint4*)&smem_dS_1[(kv_outer * 128 + threadIdx.x) * 64 + (g - 16) * 8];
                
                uint32_t q0 = ds_vec0.x; float f32_ds0 = __bfloat162float(bf16_from_uint32((q0 >> 16) & 0xFFFF));
                uint32_t q1 = ds_vec0.x; float f32_ds1 = __bfloat162float(bf16_from_uint32(q0 & 0xFFFF));
                uint32_t q2 = ds_vec0.y; float f32_ds2 = __bfloat162float(bf16_from_uint32((q2 >> 16) & 0xFFFF));
                uint32_t q3 = ds_vec0.y; float f32_ds3 = __bfloat162float(bf16_from_uint32(q2 & 0xFFFF));
                
                uint32_t q4 = ds_vec1.x; float f32_ds4 = __bfloat162float(bf16_from_uint32((q4 >> 16) & 0xFFFF));
                uint32_t q5 = ds_vec1.x; float f32_ds5 = __bfloat162float(bf16_from_uint32(q4 & 0xFFFF));
                uint32_t q6 = ds_vec1.y; float f32_ds6 = __bfloat162float(bf16_from_uint32((q6 >> 16) & 0xFFFF));
                uint32_t q7 = ds_vec1.y; float f32_ds7 = __bfloat162float(bf16_from_uint32(q6 & 0xFFFF));
                
                ds0 = f32_ds0; ds1 = f32_ds1; ds2 = f32_ds2; ds3 = f32_ds3;
                ds4 = f32_ds4; ds5 = f32_ds5; ds6 = f32_ds6; ds7 = f32_ds7;
            }
            
            if (kv_outer * 128 + threadIdx.x < 128) {
                uint4 k_vec0 = *(uint4*)&smem_K_0[(kv_outer * 128 + threadIdx.x) * 64 + (g % 16) * 8];
                uint4 k_vec1 = *(uint4*)&smem_K_1[(kv_outer * 128 + threadIdx.x) * 64 + (g % 16) * 8];
                
                float f32_k0 = __bfloat162float(bf16_from_uint32((k_vec0.x >> 16) & 0xFFFF));
                float f32_k1 = __bfloat162float(bf16_from_uint32(k_vec0.x & 0xFFFF));
                float f32_k2 = __bfloat162float(bf16_from_uint32((k_vec0.y >> 16) & 0xFFFF));
                float f32_k3 = __bfloat162float(bf16_from_uint32(k_vec0.y & 0xFFFF));
                float f32_k4 = __bfloat162float(bf16_from_uint32((k_vec0.z >> 16) & 0xFFFF));
                float f32_k5 = __bfloat162float(bf16_from_uint32(k_vec0.z & 0xFFFF));
                float f32_k6 = __bfloat162float(bf16_from_uint32((k_vec0.w >> 16) & 0xFFFF));
                float f32_k7 = __bfloat162float(bf16_from_uint32(k_vec0.w & 0xFFFF));
                
                dq_sum[g] += ds0 * f32_k0 + ds1 * f32_k1 + ds2 * f32_k2 + ds3 * f32_k3 + ds4 * f32_k4 + ds5 * f32_k5 + ds6 * f32_k6 + ds7 * f32_k7;
                
                float f32_k8  = __bfloat162float(bf16_from_uint32((k_vec1.x >> 16) & 0xFFFF));
                float f32_k9  = __bfloat162float(bf16_from_uint32(k_vec1.x & 0xFFFF));
                float f32_k10 = __bfloat162float(bf16_from_uint32((k_vec1.y >> 16) & 0xFFFF));
                float f32_k11 = __bfloat162float(bf16_from_uint32(k_vec1.y & 0xFFFF));
                float f32_k12 = __bfloat162float(bf16_from_uint32((k_vec1.z >> 16) & 0xFFFF));
                float f32_k13 = __bfloat162float(bf16_from_uint32(k_vec1.z & 0xFFFF));
                float f32_k14 = __bfloat162float(bf16_from_uint32((k_vec1.w >> 16) & 0xFFFF));
                float f32_k15 = __bfloat162float(bf16_from_uint32(k_vec1.w & 0xFFFF));
                
                dq_sum[g+32] += ds0 * f32_k8 + ds1 * f32_k9 + ds2 * f32_k10 + ds3 * f32_k11 + ds4 * f32_k12 + ds5 * f32_k13 + ds6 * f32_k14 + ds7 * f32_k15;
            }
        }
        
        for (int g = 0; g < 32; g++) {
            float dk0 = 0, dk1 = 0, dk2 = 0, dk3 = 0;
            float dk4 = 0, dk5 = 0, dk6 = 0, dk7 = 0;
            
            for (int i = 0; i < 16; i++) {
                uint4 ds_vec0 = *(uint4*)&smem_dS_0[(i * 8 + (kv_outer * 128 + threadIdx.x) / 64) * 128 + (kv_outer * 128 + threadIdx.x) % 64];
                uint4 ds_vec1 = *(uint4*)&smem_dS_1[(i * 8 + (kv_outer * 128 + threadIdx.x) / 64) * 128 + (kv_outer * 128 + threadIdx.x) % 64];
                
                uint32_t q0 = ds_vec0.x; float f32_ds0 = __bfloat162float(bf16_from_uint32((q0 >> 16) & 0xFFFF));
                uint32_t q1 = ds_vec0.x; float f32_ds1 = __bfloat162float(bf16_from_uint32(q0 & 0xFFFF));
                uint32_t q2 = ds_vec0.y; float f32_ds2 = __bfloat162float(bf16_from_uint32((q2 >> 16) & 0xFFFF));
                uint32_t q3 = ds_vec0.y; float f32_ds3 = __bfloat162float(bf16_from_uint32(q2 & 0xFFFF));
                
                uint32_t q4 = ds_vec1.x; float f32_ds4 = __bfloat162float(bf16_from_uint32((q4 >> 16) & 0xFFFF));
                uint32_t q5 = ds_vec1.x; float f32_ds5 = __bfloat162float(bf16_from_uint32(q4 & 0xFFFF));
                uint32_t q6 = ds_vec1.y; float f32_ds6 = __bfloat162float(bf16_from_uint32((q6 >> 16) & 0xFFFF));
                uint32_t q7 = ds_vec1.y; float f32_ds7 = __bfloat162float(bf16_from_uint32(q6 & 0xFFFF));
                
                uint4 q_vec0 = *(uint4*)&smem_Q_0[(i * 8 + (kv_outer * 128 + threadIdx.x) / 64) * 64 + (g % 16) * 8];
                uint4 q_vec1 = *(uint4*)&smem_Q_1[(i * 8 + (kv_outer * 128 + threadIdx.x) / 64) * 64 + (g % 16) * 8];
                
                float f32_q0 = __bfloat162float(bf16_from_uint32((q_vec0.x >> 16) & 0xFFFF));
                float f32_q1 = __bfloat162float(bf16_from_uint32(q_vec0.x & 0xFFFF));
                float f32_q2 = __bfloat162float(bf16_from_uint32((q_vec0.y >> 16) & 0xFFFF));
                float f32_q3 = __bfloat162float(bf16_from_uint32(q_vec0.y & 0xFFFF));
                float f32_q4 = __bfloat162float(bf16_from_uint32((q_vec0.z >> 16) & 0xFFFF));
                float f32_q5 = __bfloat162float(bf16_from_uint32(q_vec0.z & 0xFFFF));
                float f32_q6 = __bfloat162float(bf16_from_uint32((q_vec0.w >> 16) & 0xFFFF));
                float f32_q7 = __bfloat162float(bf16_from_uint32(q_vec0.w & 0xFFFF));
                
                dk0 += f32_ds0 * f32_q0 + f32_ds1 * f32_q1 + f32_ds2 * f32_q2 + f32_ds3 * f32_q3;
                dk1 += f32_ds0 * f32_q4 + f32_ds1 * f32_q5 + f32_ds2 * f32_q6 + f32_ds3 * f32_q7;
                dk2 += f32_ds4 * f32_q0 + f32_ds5 * f32_q1 + f32_ds6 * f32_q2 + f32_ds7 * f32_q3;
                dk3 += f32_ds4 * f32_q4 + f32_ds5 * f32_q5 + f32_ds6 * f32_q6 + f32_ds7 * f32_q7;
                
                float f32_q8  = __bfloat162float(bf16_from_uint32((q_vec1.x >> 16) & 0xFFFF));
                float f32_q9  = __bfloat162float(bf16_from_uint32(q_vec1.x & 0xFFFF));
                float f32_q10 = __bfloat162float(bf16_from_uint32((q_vec1.y >> 16) & 0xFFFF));
                float f32_q11 = __bfloat162float(bf16_from_uint32(q_vec1.y & 0xFFFF));
                float f32_q12 = __bfloat162float(bf16_from_uint32((q_vec1.z >> 16) & 0xFFFF));
                float f32_q13 = __bfloat162float(bf16_from_uint32(q_vec1.z & 0xFFFF));
                float f32_q14 = __bfloat162float(bf16_from_uint32((q_vec1.w >> 16) & 0xFFFF));
                float f32_q15 = __bfloat162float(bf16_from_uint32(q_vec1.w & 0xFFFF));
                
                dk4 += f32_ds0 * f32_q8 + f32_ds1 * f32_q9 + f32_ds2 * f32_q10 + f32_ds3 * f32_q11;
                dk5 += f32_ds0 * f32_q12 + f32_ds1 * f32_q13 + f32_ds2 * f32_q14 + f32_ds3 * f32_q15;
                dk6 += f32_ds4 * f32_q8 + f32_ds5 * f32_q9 + f32_ds6 * f32_q10 + f32_ds7 * f32_q11;
                dk7 += f32_ds4 * f32_q12 + f32_ds5 * f32_q13 + f32_ds6 * f32_q14 + f32_ds7 * f32_q15;
            }
            
            uint32_t packed0 = pack_bf16(*(uint16_t*)&__float2bfloat16(dk0), *(uint16_t*)&__float2bfloat16(dk1));
            uint32_t packed1 = pack_bf16(*(uint16_t*)&__float2bfloat16(dk2), *(uint16_t*)&__float2bfloat16(dk3));
            *(uint32_t*)&smem_dS_0[(kv_outer * 128 + threadIdx.x) * 64 + (g % 16) * 8] = packed0;
            *(uint32_t*)&smem_dS_0[(kv_outer * 128 + threadIdx.x) * 64 + (g % 16) * 8 + 2] = packed1;
            
            uint32_t packed2 = pack_bf16(*(uint16_t*)&__float2bfloat16(dk4), *(uint16_t*)&__float2bfloat16(dk5));
            uint32_t packed3 = pack_bf16(*(uint16_t*)&__float2bfloat16(dk6), *(uint16_t*)&__float2bfloat16(dk7));
            *(uint32_t*)&smem_dS_1[(kv_outer * 128 + threadIdx.x) * 64 + (g % 16) * 8] = packed2;
            *(uint32_t*)&smem_dS_1[(kv_outer * 128 + threadIdx.x) * 64 + (g % 16) * 8 + 2] = packed3;
        }
        
        for (uint32_t i = 0; i < 128 * 128; i += 128) {
            uint32_t r = i / 128; 
            uint32_t c = i % 128;
            uint32_t buf_idx = c / 64; 
            uint32_t elem_idx = c % 64; 
            
            smem_dS_0[r * 64 + elem_idx] = (buf_idx == 0) ? smem_dS_0[r * 64 + elem_idx] : smem_dS_1[r * 64 + elem_idx];
        }
        __syncthreads();
        
        uint32_t row_start = (kv_outer * 128 + threadIdx.x);
        uint32_t col_start = 0;
        
        for (int g = 0; g < 32; g++) {
            float4 val0, val1;
            val0.x = 0; val0.y = 0; val0.z = 0; val0.w = 0;
            val1.x = 0; val1.y = 0; val1.z = 0; val1.w = 0;
            
            for (int i = 0; i < 16; i++) {
                uint4 ds_vec0 = *(uint4*)&smem_dS_0[(i * 8 + (row_start / 64)) * 128 + (row_start % 64)];
                uint4 ds_vec1 = *(uint4*)&smem_dS_1[(i * 8 + (row_start / 64)) * 128 + (row_start % 64)];
                
                uint32_t q0 = ds_vec0.x; float f32_ds0 = __bfloat162float(bf16_from_uint32((q0 >> 16) & 0xFFFF));
                uint32_t q1 = ds_vec0.x; float f32_ds1 = __bfloat162float(bf16_from_uint32(q0 & 0xFFFF));
                uint32_t q2 = ds_vec0.y; float f32_ds2 = __bfloat162float(bf16_from_uint32((q2 >> 16) & 0xFFFF));
                uint32_t q3 = ds_vec0.y; float f32_ds3 = __bfloat162float(bf16_from_uint32(q2 & 0xFFFF));
                
                uint32_t q4 = ds_vec1.x; float f32_ds4 = __bfloat162float(bf16_from_uint32((q4 >> 16) & 0xFFFF));
                uint32_t q5 = ds_vec1.x; float f32_ds5 = __bfloat162float(bf16_from_uint32(q4 & 0xFFFF));
                uint32_t q6 = ds_vec1.y; float f32_ds6 = __bfloat162float(bf16_from_uint32((q6 >> 16) & 0xFFFF));
                uint32_t q7 = ds_vec1.y; float f32_ds7 = __bfloat162float(bf16_from_uint32(q6 & 0xFFFF));
                
                uint4 q_vec0 = *(uint4*)&smem_Q_0[(i * 8 + (row_start / 64)) * 64 + (g % 16) * 8];
                uint4 q_vec1 = *(uint4*)&smem_Q_1[(i * 8 + (row_start / 64)) * 64 + (g % 16) * 8];
                
                float f32_q0 = __bfloat162float(bf16_from_uint32((q_vec0.x >> 16) & 0xFFFF));
                float f32_q1 = __bfloat162float(bf16_from_uint32(q_vec0.x & 0xFFFF));
                float f32_q2 = __bfloat162float(bf16_from_uint32((q_vec0.y >> 16) & 0xFFFF));
                float f32_q3 = __bfloat162float(bf16_from_uint32(q_vec0.y & 0xFFFF));
                
                val0.x += f32_ds0 * f32_q0;
                val0.y += f32_ds1 * f32_q1;
                val0.z += f32_ds2 * f32_q2;
                val0.w += f32_ds3 * f32_q3;
                
                float f32_q4 = __bfloat162float(bf16_from_uint32((q_vec0.z >> 16) & 0xFFFF));
                float f32_q5 = __bfloat162float(bf16_from_uint32(q_vec0.z & 0xFFFF));
                float f32_q6 = __bfloat162float(bf16_from_uint32((q_vec0.w >> 16) & 0xFFFF));
                float f32_q7 = __bfloat162float(bf16_from_uint32(q_vec0.w & 0xFFFF));
                
                val1.x += f32_ds4 * f32_q4;
                val1.y += f32_ds5 * f32_q5;
                val1.z += f32_ds6 * f32_q6;
                val1.w += f32_ds7 * f32_q7;
                
                float f32_q8  = __bfloat162float(bf16_from_uint32((q_vec1.x >> 16) & 0xFFFF));
                float f32_q9  = __bfloat162float(bf16_from_uint32(q_vec1.x & 0xFFFF));
                float f32_q10 = __bfloat162float(bf16_from_uint32((q_vec1.y >> 16) & 0xFFFF));
                float f32_q11 = __bfloat162float(bf16_from_uint32(q_vec1.y & 0xFFFF));
                
                val0.x += f32_ds0 * f32_q8;
                val0.y += f32_ds1 * f32_q9;
                val0.z += f32_ds2 * f32_q10;
                val0.w += f32_ds3 * f32_q11;
                
                float f32_q12 = __bfloat162float(bf16_from_uint32((q_vec1.z >> 16) & 0xFFFF));
                float f32_q13 = __bfloat162float(bf16_from_uint32(q_vec1.z & 0xFFFF));
                float f32_q14 = __bfloat162float(bf16_from_uint32((q_vec1.w >> 16) & 0xFFFF));
                float f32_q15 = __bfloat162float(bf16_from_uint32(q_vec1.w & 0xFFFF));
                
                val1.x += f32_ds4 * f32_q12;
                val1.y += f32_ds5 * f32_q13;
                val1.z += f32_ds6 * f32_q14;
                val1.w += f32_ds7 * f32_q15;
            }
            
            uint32_t g_row = row_start;
            uint32_t g_col0 = col_start + (g % 16) * 4;
            uint32_t g_col1 = col_start + (g % 16) * 4 + 1;
            uint32_t g_col2 = col_start + (g % 16) * 4 + 2;
            uint32_t g_col3 = col_start + (g % 16) * 4 + 3;
            
            if (g_row < 128) {
                if (g_col0 < 128) atomicAdd(&fp32_dK[d_offset + kv_outer * 128 * 128 + g_row * 128 + g_col0], val0.x);
                if (g_col1 < 128) atomicAdd(&fp32_dK[d_offset + kv_outer * 128 * 128 + g_row * 128 + g_col1], val0.y);
                if (g_col2 < 128) atomicAdd(&fp32_dK[d_offset + kv_outer * 128 * 128 + g_row * 128 + g_col2], val0.z);
                if (g_col3 < 128) atomicAdd(&fp32_dK[d_offset + kv_outer * 128 * 128 + g_row * 128 + g_col3], val0.w);
                
                if (g_col0 + 64 < 128) atomicAdd(&fp32_dK[d_offset + kv_outer * 128 * 128 + g_row * 128 + g_col0 + 64], val1.x);
                if (g_col1 + 64 < 128) atomicAdd(&fp32_dK[d_offset + kv_outer * 128 * 128 + g_row * 128 + g_col1 + 64], val1.y);
                if (g_col2 + 64 < 128) atomicAdd(&fp32_dK[d_offset + kv_outer * 128 * 128 + g_row * 128 + g_col2 + 64], val1.z);
                if (g_col3 + 64 < 128) atomicAdd(&fp32_dK[d_offset + kv_outer * 128 * 128 + g_row * 128 + g_col3 + 64], val1.w);
            }
        }
        
        for (int g = 0; g < 32; g++) {
            float dv0 = 0, dv1 = 0, dv2 = 0, dv3 = 0;
            float dv4 = 0, dv5 = 0, dv6 = 0, dv7 = 0;
            
            for (int i = 0; i < 16; i++) {
                uint4 p_vec0 = *(uint4*)&smem_P[(i * 8 + (kv_outer * 128 + threadIdx.x) / 64) * 128 + (kv_outer * 128 + threadIdx.x) % 64];
                uint4 p_vec1 = *(uint4*)&smem_P[(i * 8 + (kv_outer * 128 + threadIdx.x) / 64) * 128 + (kv_outer * 128 + threadIdx.x) % 64 + 64];
                
                uint32_t q0 = p_vec0.x; float f32_p0 = __bfloat162float(bf16_from_uint32((q0 >> 16) & 0xFFFF));
                uint32_t q1 = p_vec0.x; float f32_p1 = __bfloat162float(bf16_from_uint32(q0 & 0xFFFF));
                uint32_t q2 = p_vec0.y; float f32_p2 = __bfloat162float(bf16_from_uint32((q2 >> 16) & 0xFFFF));
                uint32_t q3 = p_vec0.y; float f32_p3 = __bfloat162float(bf16_from_uint32(q2 & 0xFFFF));
                
                uint32_t q4 = p_vec1.x; float f32_p4 = __bfloat162float(bf16_from_uint32((q4 >> 16) & 0xFFFF));
                uint32_t q5 = p_vec1.x; float f32_p5 = __bfloat162float(bf16_from_uint32(q4 & 0xFFFF));
                uint32_t q6 = p_vec1.y; float f32_p6 = __bfloat162float(bf16_from_uint32((q6 >> 16) & 0xFFFF));
                uint32_t q7 = p_vec1.y; float f32_p7 = __bfloat162float(bf16_from_uint32(q6 & 0xFFFF));
                
                uint4 do_vec0 = *(uint4*)&smem_dO_0[(i * 8 + (kv_outer * 128 + threadIdx.x) / 64) * 64 + (g % 16) * 8];
                uint4 do_vec1 = *(uint4*)&smem_dO_1[(i * 8 + (kv_outer * 128 + threadIdx.x) / 64) * 64 + (g % 16) * 8];
                
                float f32_do0 = __bfloat162float(bf16_from_uint32((do_vec0.x >> 16) & 0xFFFF));
                float f32_do1 = __bfloat162float(bf16_from_uint32(do_vec0.x & 0xFFFF));
                float f32_do2 = __bfloat162float(bf16_from_uint32((do_vec0.y >> 16) & 0xFFFF));
                float f32_do3 = __bfloat162float(bf16_from_uint32(do_vec0.y & 0xFFFF));
                
                dv0 += f32_p0 * f32_do0;
                dv1 += f32_p1 * f32_do1;
                dv2 += f32_p2 * f32_do2;
                dv3 += f32_p3 * f32_do3;
                
                float f32_do4 = __bfloat162float(bf16_from_uint32((do_vec0.z >> 16) & 0xFFFF));
                float f32_do5 = __bfloat162float(bf16_from_uint32(do_vec0.z & 0xFFFF));
                float f32_do6 = __bfloat162float(bf16_from_uint32((do_vec0.w >> 16) & 0xFFFF));
                float f32_do7 = __bfloat162float(bf16_from_uint32(do_vec0.w & 0xFFFF));
                
                dv4 += f32_p4 * f32_do4;
                dv5 += f32_p5 * f32_do5;
                dv6 += f32_p6 * f32_do6;
                dv7 += f32_p7 * f32_do7;
                
                float f32_do8  = __bfloat162float(bf16_from_uint32((do_vec1.x >> 16) & 0xFFFF));
                float f32_do9  = __bfloat162float(bf16_from_uint32(do_vec1.x & 0xFFFF));
                float f32_do10 = __bfloat162float(bf16_from_uint32((do_vec1.y >> 16) & 0xFFFF));
                float f32_do11 = __bfloat162float(bf16_from_uint32(do_vec1.y & 0xFFFF));
                
                dv0 += f32_p0 * f32_do8;
                dv1 += f32_p1 * f32_do9;
                dv2 += f32_p2 * f32_do10;
                dv3 += f32_p3 * f32_do11;
                
                float f32_do12 = __bfloat162float(bf16_from_uint32((do_vec1.z >> 16) & 0xFFFF));
                float f32_do13 = __bfloat162float(bf16_from_uint32(do_vec1.z & 0xFFFF));
                float f32_do14 = __bfloat162float(bf16_from_uint32((do_vec1.w >> 16) & 0xFFFF));
                float f32_do15 = __bfloat162float(bf16_from_uint32(do_vec1.w & 0xFFFF));
                
                dv4 += f32_p4 * f32_do12;
                dv5 += f32_p5 * f32_do13;
                dv6 += f32_p6 * f32_do14;
                dv7 += f32_p7 * f32_do15;
            }
            
            uint32_t packed0 = pack_bf16(*(uint16_t*)&__float2bfloat16(dv0), *(uint16_t*)&__float2bfloat16(dv1));
            uint32_t packed1 = pack_bf16(*(uint16_t*)&__float2bfloat16(dv2), *(uint16_t*)&__float2bfloat16(dv3));
            *(uint32_t*)&smem_dS_0[(kv_outer * 128 + threadIdx.x) * 64 + (g % 16) * 8] = packed0;
            *(uint32_t*)&smem_dS_0[(kv_outer * 128 + threadIdx.x) * 64 + (g % 16) * 8 + 2] = packed1;
            
            uint32_t packed2 = pack_bf16(*(uint16_t*)&__float2bfloat16(dv4), *(uint16_t*)&__float2bfloat16(dv5));
            uint32_t packed3 = pack_bf16(*(uint16_t*)&__float2bfloat16(dv6), *(uint16_t*)&__float2bfloat16(dv7));
            *(uint32_t*)&smem_dS_1[(kv_outer * 128 + threadIdx.x) * 64 + (g % 16) * 8] = packed2;
            *(uint32_t*)&smem_dS_1[(kv_outer * 128 + threadIdx.x) * 64 + (g % 16) * 8 + 2] = packed3;
        }
        
        for (uint32_t i = 0; i < 128 * 128; i += 128) {
            uint32_t r = i / 128; 
            uint32_t c = i % 128;
            uint32_t buf_idx = c / 64; 
            uint32_t elem_idx = c % 64; 
            
            smem_dS_0[r * 64 + elem_idx] = (buf_idx == 0) ? smem_dS_0[r * 64 + elem_idx] : smem_dS_1[r * 64 + elem_idx];
        }
        __syncthreads();
        
        for (uint32_t i = 0; i < 128 * 128; i += 128) {
            uint32_t r = i / 128; 
            uint32_t c = i % 128;
            uint32_t g_idx = c / 4;
            uint32_t elem_idx = c % 4;
            uint32_t buf_idx = (g_idx * 4) / 64;
            uint32_t real_elem = (g_idx * 4) % 64;
            
            __nv_bfloat16 val0, val1, val2, val3;
            if (buf_idx == 0) {
                val0 = smem_dS_0[r * 64 + real_elem];
                val1 = smem_dS_0[r * 64 + real_elem + 1];
                val2 = smem_dS_0[r * 64 + real_elem + 2];
                val3 = smem_dS_0[r * 64 + real_elem + 3];
            } else {
                val0 = smem_dS_1[r * 64 + real_elem];
                val1 = smem_dS_1[r * 64 + real_elem + 1];
                val2 = smem_dS_1[r * 64 + real_elem + 2];
                val3 = smem_dS_1[r * 64 + real_elem + 3];
            }
            
            if (r < 128 && c < 128) {
                atomicAdd(&fp32_dV[d_offset + kv_outer * 128 * 128 + r * 128 + c], __bfloat162float(val0));
                atomicAdd(&fp32_dV[d_offset + kv_outer * 128 * 128 + r * 128 + c + 1], __bfloat162float(val1));
                atomicAdd(&fp32_dV[d_offset + kv_outer * 128 * 128 + r * 128 + c + 2], __bfloat162float(val2));
                atomicAdd(&fp32_dV[d_offset + kv_outer * 128 * 128 + r * 128 + c + 3], __bfloat162float(val3));
            }
        }
        __syncthreads(); 
        
        first_k_block = false;
    }
    
    for (int g = 0; g < 32; g++) {
        float f0 = dq_sum[g];
        float f1 = dq_sum[g+1];
        float f2 = dq_sum[g+2];
        float f3 = dq_sum[g+3];
        
        uint32_t packed0 = pack_bf16(*(uint16_t*)&__float2bfloat16(f0), *(uint16_t*)&__float2bfloat16(f1));
        uint32_t packed1 = pack_bf16(*(uint16_t*)&__float2bfloat16(f2), *(uint16_t*)&__float2bfloat16(f3));
        
        uint32_t row = threadIdx.x;
        uint32_t col = (g / 16) * 64 + (g % 16) * 4;
        
        if (row < 128 && col < 128) {
            *(uint32_t*)&my_dQ[row * 128 + col] = packed0;
            *(uint32_t*)&my_dQ[row * 128 + col + 2] = packed1;
        }
    }
}

namespace tvm_ffi_attention_bwd {

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

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
         
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    uint32_t B = Q.size(0);
    uint32_t H = Q.size(1);
    uint32_t S = Q.size(2);
    uint32_t d = Q.size(3);
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O, tma_dO;
    CUresult res;
    res = create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), 128, B * H * S, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA Q failed\n"); exit(1); }
    res = create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), 128, B * H * S, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA K failed\n"); exit(1); }
    res = create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), 128, B * H * S, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA V failed\n"); exit(1); }
    res = create_tma_2d_descriptor_2B(&tma_O, O.data_ptr(), 128, B * H * S, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA O failed\n"); exit(1); }
    res = create_tma_2d_descriptor_2B(&tma_dO, dO.data_ptr(), 128, B * H * S, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA dO failed\n"); exit(1); }
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    float *fp32_dQ, *fp32_dK, *fp32_dV;
    CUDA_CHECK(cudaMallocAsync(&fp32_dQ, B * H * S * d * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&fp32_dK, B * H * S * d * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&fp32_dV, B * H * S * d * sizeof(float), stream));
    
    CUDA_CHECK(cudaMemsetAsync(fp32_dQ, 0, B * H * S * d * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(fp32_dK, 0, B * H * S * d * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(fp32_dV, 0, B * H * S * d * sizeof(float), stream));
    
    uint32_t num_q_tiles = (S + 127) / 128;
    dim3 grid(num_q_tiles, B * H);
    dim3 block(128);
    
    uint32_t smem_size = 230400;
    CUDA_CHECK(cudaFuncSetAttribute(
        bwd_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    ));
    
    bwd_kernel<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, tma_O, tma_dO,
        fp32_dQ, fp32_dK, fp32_dV,
        static_cast<const float*>(L.data_ptr()),
        S, d, B, H
    );
    
    CUDA_CHECK(cudaGetLastError());
    
    size_t n = B * H * S * d;
    cast_fp32_to_bf16<<<(n + 255) / 256, 256, 0, stream>>>(
        static_cast<__nv_bfloat16*>(dQ.data_ptr()), fp32_dQ, n);
    cast_fp32_to_bf16<<<(n + 255) / 256, 256, 0, stream>>>(
        static_cast<__nv_bfloat16*>(dK.data_ptr()), fp32_dK, n);
    cast_fp32_to_bf16<<<(n + 255) / 256, 256, 0, stream>>>(
        static_cast<__nv_bfloat16*>(dV.data_ptr()), fp32_dV, n);
    
    CUDA_CHECK(cudaFreeAsync(fp32_dQ, stream));
    CUDA_CHECK(cudaFreeAsync(fp32_dK, stream));
    CUDA_CHECK(cudaFreeAsync(fp32_dV, stream));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_attention_bwd::run);

}  // namespace tvm_ffi_attention_bwd