#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
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

#define CU_CHECK(call) do {                                        \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        fprintf(stderr, "CU error %d at %s:%d\n",                 \
                _e, __FILE__, __LINE__);                           \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_mha_lse {

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(phase));
}

__device__ __forceinline__ void named_barrier_sync_fn(int bar_id, int count) {
    asm volatile("barrier.sync.aligned %0, %1;" :: "r"(bar_id), "r"(count));
}

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(bar)),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
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

__device__ __forceinline__ void tcgen05_commit(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile(
        "tcgen05.commit.cta_group::1"
        ".mbarrier::arrive::one.shared::cta.b64"
        " [%0];"
        :: "r"(a));
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

__device__ __forceinline__ void load_to_tmem(uint64_t* bar, uint32_t tmem_base, __nv_bfloat16 (*smem)[64], uint32_t byte_offset) {
    asm volatile("cp.async.bulk.shared::cta.global::cta.bulk_group [%0], [%1], %2;\n"
                 :: "r"(tmem_base), "r"((uint32_t)__cvta_generic_to_shared(&smem[0][0]) + byte_offset), "n"(4096));
}

__device__ __forceinline__ void umma_cg1(uint32_t tmem_d, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_d), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ uint64_t make_desc(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;   // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint64_t k_major(void* ptr) {
    return make_desc(ptr, 1, 1024);
}

__device__ __forceinline__ uint64_t n_major_64(void* ptr) {
    return make_desc(ptr, 8192, 1024);
}

__device__ __forceinline__ uint64_t n_major_128(void* ptr) {
    return make_desc(ptr, 16384, 1024);
}

__device__ __forceinline__ uint64_t advance_desc_k_major(uint64_t d, int step) {
    uint64_t addr_bits = d & 0x3FFF;
    addr_bits += step * 2;
    d &= ~0x3FFF;
    d |= addr_bits;
    return d;
}

__device__ __forceinline__ uint64_t advance_desc_k_major_128(uint64_t d, int step) {
    uint64_t addr_bits = d & 0x3FFF;
    addr_bits += step * 2;
    d &= ~0x3FFF;
    d |= addr_bits;
    return d;
}

__device__ __forceinline__ uint64_t advance_desc_n_major_64(uint64_t d, int step) {
    uint64_t addr_bits = d & 0x3FFF;
    uint32_t sbo = 1024;
    addr_bits += step * ((16 * sbo) / 16);
    d &= ~0x3FFF;
    d |= addr_bits;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, uint32_t a_major, uint32_t b_major) {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= (a_major << 15);
    d |= (b_major << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__global__ __launch_bounds__(128, 1) void flashattention_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O,
    float* LSE,
    uint32_t S
) {
    int bh_idx = blockIdx.y;
    int s_idx = blockIdx.x;
    int s_base = s_idx * 128;
    int row = threadIdx.x;
    
    extern __shared__ __align__(1024) __nv_bfloat16 smem_pool[];
    __nv_bfloat16 (*smem_Q0)[64] = (__nv_bfloat16 (*)[64])smem_pool;
    __nv_bfloat16 (*smem_Q1)[64] = smem_Q0 + 128;
    __nv_bfloat16 (*smem_K0)[64] = smem_Q1 + 128;
    __nv_bfloat16 (*smem_K1)[64] = smem_K0 + 64;
    __nv_bfloat16 (*smem_V0)[64] = smem_K1 + 64;
    __nv_bfloat16 (*smem_V1)[64] = smem_V0 + 64;
    
    __shared__ alignas(16) __nv_bfloat16 smem_P_linear[128 * 64];
    
    __shared__ alignas(128) uint64_t bar_q;
    __shared__ alignas(128) uint64_t bar_kv;
    __shared__ alignas(128) uint64_t bar_umma;
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&bar_q, 1);
        init_smem_barrier_fn(&bar_kv, 1);
        init_smem_barrier_fn(&bar_umma, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    
    __shared__ uint32_t tmem_Q;
    __shared__ uint32_t tmem_K;
    __shared__ uint32_t tmem_P;
    __shared__ uint32_t tmem_O;
    
    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_Q, 64);
        tmem_alloc_fn(&tmem_K, 64);
        tmem_alloc_fn(&tmem_P, 64);
        tmem_alloc_fn(&tmem_O, 128);
    }
    
    int phase_q = 0;
    int phase_kv = 0;
    int phase_umma = 0;
    int warp_id = row / 32;
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&bar_q, 2 * 16384); 
        
        tma_load_3d_fn(&tma_Q, &bar_q, (void*)smem_Q0, 0, s_base, bh_idx);
        tma_load_3d_fn(&tma_Q, &bar_q, (void*)smem_Q1, 64, s_base, bh_idx);
    }
    mbarrier_wait_fn(&bar_q, phase_q);
    phase_q ^= 1;
    
    float running_max = -1e20f;
    float running_sum = 0.0f;
    
    uint32_t idesc_P = make_instr_desc_fn(64, 8, 0, 0);
    uint32_t idesc_O = make_instr_desc_fn(64, 8, 0, 1);
    
    for (int block_start = 0; block_start < S; block_start += 64) {
        bool first = (block_start == 0);
        
        __syncthreads();
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&bar_kv, 4 * 8192); 
            
            tma_load_3d_fn(&tma_K, &bar_kv, (void*)smem_K0, 0, block_start, bh_idx);
            tma_load_3d_fn(&tma_K, &bar_kv, (void*)smem_K1, 64, block_start, bh_idx);
            
            tma_load_3d_fn(&tma_V, &bar_kv, (void*)smem_V0, 0, block_start, bh_idx);
            tma_load_3d_fn(&tma_V, &bar_kv, (void*)smem_V1, 64, block_start, bh_idx);
        }
        mbarrier_wait_fn(&bar_kv, phase_kv);
        phase_kv ^= 1;
        
        if (threadIdx.x == 0) {
            load_to_tmem(&bar_kv, tmem_Q, smem_Q0, warp_id * 4096);
            load_to_tmem(&bar_kv, tmem_K, smem_K0, warp_id * 4096);
        }
        
        if (threadIdx.x == 0) {
            for(int k_step = 0; k_step < 4; k_step++) {
                uint64_t dq = advance_desc_k_major_128(k_major(&smem_Q0[0][0]), k_step);
                uint64_t dk = advance_desc_k_major_128(k_major(&smem_K0[0][0]), k_step);
                umma_cg1(tmem_P, dq, dk, idesc_P, first && k_step == 0 ? 0 : 1);
            }
            load_to_tmem(&bar_kv, tmem_Q, smem_Q1, warp_id * 4096);
            load_to_tmem(&bar_kv, tmem_K, smem_K1, warp_id * 4096);
            for(int k_step = 0; k_step < 4; k_step++) {
                uint64_t dq = advance_desc_k_major_128(k_major(&smem_Q1[0][0]), k_step);
                uint64_t dk = advance_desc_k_major_128(k_major(&smem_K1[0][0]), k_step);
                umma_cg1(tmem_P, dq, dk, idesc_P, 1);
            }
        }
        tcgen05_commit(&bar_umma);
        mbarrier_wait_fn(&bar_umma, phase_umma);
        phase_umma ^= 1;
        
        float max_val = -1e20f;
        for (int col = 0; col < 64; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_P + col, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn(); 
            
            float f0 = __uint_as_float(r0);
            float f1 = __uint_as_float(r1);
            float f2 = __uint_as_float(r2);
            float f3 = __uint_as_float(r3);
            
            int gc0 = block_start + col + 0;
            int gc1 = block_start + col + 1;
            int gc2 = block_start + col + 2;
            int gc3 = block_start + col + 3;
            if (gc0 >= S) f0 = -1e20f; else f0 *= 0.08838834764f;
            if (gc1 >= S) f1 = -1e20f; else f1 *= 0.08838834764f;
            if (gc2 >= S) f2 = -1e20f; else f2 *= 0.08838834764f;
            if (gc3 >= S) f3 = -1e20f; else f3 *= 0.08838834764f;
            
            max_val = fmaxf(max_val, fmaxf(fmaxf(f0, f1), fmaxf(f2, f3)));
        }
        
        float new_max = fmaxf(running_max, max_val);
        float factor = (new_max > running_max) ? expf(running_max - new_max) : 1.0f;
        running_sum *= factor;
        running_max = new_max;
        
        float sum_val = 0.0f;
        for (int col = 0; col < 64; col += 2) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_P + col, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            
            float f0 = __uint_as_float(r0);
            float f1 = __uint_as_float(r1);
            float f2 = __uint_as_float(r2);
            float f3 = __uint_as_float(r3);
            
            int gc0 = block_start + col + 0;
            int gc1 = block_start + col + 1;
            int gc2 = block_start + col + 2;
            int gc3 = block_start + col + 3;
            if (gc0 >= S) f0 = -1e20f; else f0 *= 0.08838834764f;
            if (gc1 >= S) f1 = -1e20f; else f1 *= 0.08838834764f;
            if (gc2 >= S) f2 = -1e20f; else f2 *= 0.08838834764f;
            if (gc3 >= S) f3 = -1e20f; else f3 *= 0.08838834764f;
            
            float exp0 = expf(f0 - new_max) * factor;
            float exp1 = expf(f1 - new_max) * factor;
            float exp2 = expf(f2 - new_max) * factor;
            float exp3 = expf(f3 - new_max) * factor;
            
            sum_val += exp0 + exp1 + exp2 + exp3;
            
            smem_P_linear[row * 64 + col + 0] = __float2bfloat16(exp0);
            smem_P_linear[row * 64 + col + 1] = __float2bfloat16(exp1);
            smem_P_linear[row * 64 + col + 2] = __float2bfloat16(exp2);
            smem_P_linear[row * 64 + col + 3] = __float2bfloat16(exp3);
        }
        running_sum += sum_val;
        
        __syncthreads();
        
        if (threadIdx.x == 0) {
            load_to_tmem(&bar_kv, tmem_P, (__nv_bfloat16 (*)[64])smem_P_linear, warp_id * 4096);
        }
        
        if (threadIdx.x == 0) {
            load_to_tmem(&bar_kv, tmem_K, smem_V0, warp_id * 4096);
            for (int k = 0; k < 4; k++) {
                uint64_t dp = advance_desc_k_major(k_major(&smem_P_linear[0][0]), k);
                uint64_t dv = advance_desc_n_major_64(n_major_64(&smem_V0[0][0]), k);
                umma_cg1(tmem_O, dp, dv, idesc_O, first && k == 0 ? 0 : 1);
            }
            load_to_tmem(&bar_kv, tmem_K, smem_V1, warp_id * 4096);
            for (int k = 0; k < 4; k++) {
                uint64_t dp = advance_desc_k_major(k_major(&smem_P_linear[0][0]), k);
                uint64_t dv = advance_desc_n_major_64(n_major_64(&smem_V1[0][0]), k);
                umma_cg1(tmem_O + 64, dp, dv, idesc_O, first && k == 0 ? 0 : 1);
            }
        }
        tcgen05_commit(&bar_umma);
        mbarrier_wait_fn(&bar_umma, phase_umma);
        phase_umma ^= 1;
        
        __syncthreads();
    }
    
    for (int col = 0; col < 128; col += 32) {
        uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
        tmem_load_8x_fn(tmem_O + col, &r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);
        tmem_load_fence_fn();
        
        float f0 = __uint_as_float(r0);
        float f1 = __uint_as_float(r1);
        float f2 = __uint_as_float(r2);
        float f3 = __uint_as_float(r3);
        float f4 = __uint_as_float(r4);
        float f5 = __uint_as_float(r5);
        float f6 = __uint_as_float(r6);
        float f7 = __uint_as_float(r7);
        
        int global_row = bh_idx * S + s_base + row;
        
        if (s_base + row < S) {
            if (col + 0 < 128) O[global_row * 128 + col + 0] = __float2bfloat16(f0);
            if (col + 1 < 128) O[global_row * 128 + col + 1] = __float2bfloat16(f1);
            if (col + 2 < 128) O[global_row * 128 + col + 2] = __float2bfloat16(f2);
            if (col + 3 < 128) O[global_row * 128 + col + 3] = __float2bfloat16(f3);
            if (col + 4 < 128) O[global_row * 128 + col + 4] = __float2bfloat16(f4);
            if (col + 5 < 128) O[global_row * 128 + col + 5] = __float2bfloat16(f5);
            if (col + 6 < 128) O[global_row * 128 + col + 6] = __float2bfloat16(f6);
            if (col + 7 < 128) O[global_row * 128 + col + 7] = __float2bfloat16(f7);
        }
    }
    
    if (threadIdx.x < 128) {
        int global_row = bh_idx * S + s_base + threadIdx.x;
        if (global_row < bh_idx * S + S) {
            LSE[global_row] = running_max + logf(running_sum);
        }
    }
    
    tmem_dealloc_fn(tmem_Q, 64);
    tmem_dealloc_fn(tmem_K, 64);
    tmem_dealloc_fn(tmem_P, 64);
    tmem_dealloc_fn(tmem_O, 128);
}

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, 
                                     uint64_t gmem_inner_dim, uint64_t gmem_mid_dim, uint64_t gmem_outer_dim, 
                                     uint32_t smem_inner_dim, uint32_t smem_mid_dim, uint32_t smem_outer_dim, 
                                     CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[3] = {gmem_inner_dim, gmem_mid_dim, gmem_outer_dim};
    cuuint64_t globalStrides[2] = {gmem_inner_dim * 2, gmem_inner_dim * gmem_mid_dim * 2};
    cuuint32_t boxDim[3] = {smem_inner_dim, smem_mid_dim, smem_outer_dim};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill
    );
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    uint32_t B = Q.size(0);
    uint32_t H = Q.size(1);
    uint32_t S = Q.size(2);
    uint32_t D = Q.size(3);
    
    void* q_ptr = Q.data_ptr();
    void* k_ptr = K.data_ptr();
    void* v_ptr = V.data_ptr();
    __nv_bfloat16* o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_ptr = static_cast<float*>(LSE.data_ptr());
    
    CUtensorMap tma_Q, tma_K, tma_V;
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_Q, q_ptr, D, S, B*H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_K, k_ptr, D, S, B*H, 64, 64, 1, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_V, v_ptr, D, S, B*H, 64, 64, 1, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
    dim3 grid(S / 128, B * H);
    dim3 block(128);
    
    uint32_t smem_size = 98304;
    CUDA_CHECK(cudaFuncSetAttribute(flashattention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    flashattention_kernel<<<grid, block, smem_size, stream>>>(tma_Q, tma_K, tma_V, o_ptr, lse_ptr, S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_lse::run);

}