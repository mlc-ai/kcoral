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

#define CU_CHECK(call) do {                                        \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        const char* err_str;                                       \
        cuGetErrorString(_e, &err_str);                            \
        fprintf(stderr, "CU error %s at %s:%d\n",                 \
                err_str, __FILE__, __LINE__);                      \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace flashinfer_bwd_d128 {

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

__device__ __forceinline__ void st_shared_128_fn(uint32_t addr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "r"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
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

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
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

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(uint32_t addr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    uint64_t base_offset = (addr >> 7) & 0x7;
    d |= (base_offset << 49);
    d |= (uint64_t)2 << 61; // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    // FP32 C
    d |= (1u << 7);    // BF16 A
    d |= (1u << 10);   // BF16 B
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ void issue_umma_128x128(
    uint32_t tmem_c, uint32_t accum_init,
    void* smem_A, uint32_t tmem_A, int op_type_A,
    void* smem_B, int op_type_B)
{
    uint32_t idesc = make_instr_desc_fn(128, 128);
    if (op_type_A == 1) idesc |= (1u << 15);
    if (op_type_B == 1) idesc |= (1u << 16);
    
    for (int k = 0; k < 128; k += 16) {
        uint32_t accum = (k == 0) ? accum_init : 1;
        
        uint64_t desc_B = 0;
        uint32_t addr_B = (uint32_t)__cvta_generic_to_shared(smem_B);
        if (op_type_B == 0) { // K-Major
            addr_B += (k / 64) * 16384 + (k % 64) * 2;
            desc_B = make_smem_desc_sm100_fn(addr_B, 1, 1024);
        } else { // MN-Major
            addr_B += k * 128; 
            desc_B = make_smem_desc_sm100_fn(addr_B, 16384, 1024);
        }
        
        if (op_type_A == 2) { // TMEM
            uint32_t cur_tmem_A = tmem_A + (k / 2);
            asm volatile(
                "{\n.reg .pred p;\n"
                "setp.ne.b32 p, %4, 0;\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}\n"
                :: "r"(tmem_c), "r"(cur_tmem_A), "l"(desc_B), "r"(idesc), "r"(accum));
        } else {
            uint64_t desc_A = 0;
            uint32_t addr_A = (uint32_t)__cvta_generic_to_shared(smem_A);
            if (op_type_A == 0) { // K-Major
                addr_A += (k / 64) * 16384 + (k % 64) * 2;
                desc_A = make_smem_desc_sm100_fn(addr_A, 1, 1024);
            } else { // MN-Major
                addr_A += k * 128;
                desc_A = make_smem_desc_sm100_fn(addr_A, 16384, 1024);
            }
            
            asm volatile(
                "{\n.reg .pred p;\n"
                "setp.ne.b32 p, %4, 0;\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                :: "r"(tmem_c), "l"(desc_A), "l"(desc_B), "r"(idesc), "r"(accum));
        }
    }
}

__device__ void umma_commit_and_wait(uint64_t* mbar, int phase) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        :: "r"(a) : "memory");
    mbarrier_wait_fn(mbar, phase);
}

__device__ __forceinline__ void tmem_epilogue_coalesced_4w_fn(
    __nv_bfloat16* D, __nv_bfloat16* smem_out,
    uint32_t M, uint32_t N, uint32_t m_block, uint32_t n_block,
    uint32_t BM, uint32_t BN, uint32_t tmem_col_base) {
    for (uint32_t col = 0; col < BN; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
           : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_col_base + col));
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
        uint32_t global_row = m_block * BM + row;
        uint32_t col_start = lane_id * 4;
        uint32_t global_col = n_block * BN + col_start;
        if (global_row < M && global_col + 3 < N) {
            uint2 data = *reinterpret_cast<uint2*>(&smem_out[row * BN + col_start]);
            *reinterpret_cast<uint2*>(D + (uint64_t)global_row * N + global_col) = data;
        }
    }
    __syncthreads();
}

__global__ void compute_D_kernel(const __nv_bfloat16* O, const __nv_bfloat16* dO, float* D, int B, int H, int S, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * H * S) {
        const __nv_bfloat16* o_ptr = O + idx * d;
        const __nv_bfloat16* do_ptr = dO + idx * d;
        float sum = 0;
        for (int i = 0; i < d; ++i) {
            sum += __bfloat162float(o_ptr[i]) * __bfloat162float(do_ptr[i]);
        }
        D[idx] = sum;
    }
}

extern __shared__ __align__(1024) uint8_t smem[];

__global__ void bwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_dO,
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    const float* LSE, const float* D,
    int B, int H, int S, float scale)
{
    uint8_t* smem_K   = smem + 0 * 32768; 
    uint8_t* smem_V   = smem + 1 * 32768; 
    uint8_t* smem_Q   = smem + 2 * 32768; 
    uint8_t* smem_dO  = smem + 3 * 32768; 
    uint8_t* smem_dST = smem + 4 * 32768; 
    float* smem_LSE   = (float*)(smem + 163840); 
    float* smem_D     = (float*)(smem + 163840 + 512); 
    uint64_t* mbar_K  = (uint64_t*)(smem + 163840 + 1024);
    uint64_t* mbar_V  = (uint64_t*)(smem + 163840 + 1024 + 8);
    uint64_t* mbar_Q  = (uint64_t*)(smem + 163840 + 1024 + 16);
    uint64_t* mbar_dO = (uint64_t*)(smem + 163840 + 1024 + 24);
    uint64_t* mbar_umma = (uint64_t*)(smem + 163840 + 1024 + 32);
    uint32_t* smem_tmem = (uint32_t*)(smem + 163840 + 1024 + 40);

    int bh_idx = blockIdx.x;
    int kv_idx = blockIdx.y;
    int bh_offset = bh_idx * S * 128;
    int num_q = S / 128;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_K, 1);
        init_smem_barrier_fn(mbar_V, 1);
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(mbar_dO, 1);
        init_smem_barrier_fn(mbar_umma, 1);
    }
    if (threadIdx.x < 32) {
        tmem_alloc_fn(smem_tmem, 512);
    }
    __syncthreads();
    
    uint32_t tmem_base = *smem_tmem;
    int phase_Q = 0, phase_dO = 0;
    int umma_phase = 0;

    if (threadIdx.x == 0) {
        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(mbar_K)), "r"(32768));
        tma_load_2d_fn(&tma_K, mbar_K, smem_K, 0, bh_idx * S + kv_idx * 128);
        tma_load_2d_fn(&tma_K, mbar_K, smem_K + 16384, 64, bh_idx * S + kv_idx * 128);

        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(mbar_V)), "r"(32768));
        tma_load_2d_fn(&tma_V, mbar_V, smem_V, 0, bh_idx * S + kv_idx * 128);
        tma_load_2d_fn(&tma_V, mbar_V, smem_V + 16384, 64, bh_idx * S + kv_idx * 128);
    }

    mbarrier_wait_fn(mbar_K, 0);
    mbarrier_wait_fn(mbar_V, 0);

    for (int q_idx = 0; q_idx < num_q; ++q_idx) {
        if (threadIdx.x == 0) {
            asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(mbar_Q)), "r"(32768));
            tma_load_2d_fn(&tma_Q, mbar_Q, smem_Q, 0, bh_idx * S + q_idx * 128);
            tma_load_2d_fn(&tma_Q, mbar_Q, smem_Q + 16384, 64, bh_idx * S + q_idx * 128);

            asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(mbar_dO)), "r"(32768));
            tma_load_2d_fn(&tma_dO, mbar_dO, smem_dO, 0, bh_idx * S + q_idx * 128);
            tma_load_2d_fn(&tma_dO, mbar_dO, smem_dO + 16384, 64, bh_idx * S + q_idx * 128);
        }

        mbarrier_wait_fn(mbar_Q, phase_Q); phase_Q ^= 1;
        mbarrier_wait_fn(mbar_dO, phase_dO); phase_dO ^= 1;

        int global_offset = bh_idx * S + q_idx * 128 + threadIdx.x;
        smem_D[threadIdx.x] = D[global_offset];
        smem_LSE[threadIdx.x] = LSE[global_offset];
        __syncthreads();

        // 1. S^T = K @ Q^T
        if (threadIdx.x == 0) {
            issue_umma_128x128(tmem_base + 256, 0, smem_K, 0, 0, smem_Q, 0);
            umma_commit_and_wait(mbar_umma, umma_phase); umma_phase ^= 1;
        }
        __syncthreads();

        // 2. P^T 
        uint32_t row = threadIdx.x;
        for (uint32_t c = 0; c < 128; c += 8) {
            uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) 
               : "r"(tmem_base + 256 + c));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float f0 = fast_exp2f_fn((__uint_as_float(r0) * scale - smem_LSE[c + 0]) * 1.44269504f);
            float f1 = fast_exp2f_fn((__uint_as_float(r1) * scale - smem_LSE[c + 1]) * 1.44269504f);
            float f2 = fast_exp2f_fn((__uint_as_float(r2) * scale - smem_LSE[c + 2]) * 1.44269504f);
            float f3 = fast_exp2f_fn((__uint_as_float(r3) * scale - smem_LSE[c + 3]) * 1.44269504f);
            float f4 = fast_exp2f_fn((__uint_as_float(r4) * scale - smem_LSE[c + 4]) * 1.44269504f);
            float f5 = fast_exp2f_fn((__uint_as_float(r5) * scale - smem_LSE[c + 5]) * 1.44269504f);
            float f6 = fast_exp2f_fn((__uint_as_float(r6) * scale - smem_LSE[c + 6]) * 1.44269504f);
            float f7 = fast_exp2f_fn((__uint_as_float(r7) * scale - smem_LSE[c + 7]) * 1.44269504f);
            
            uint32_t p0 = pack_bf16_fn(__float_as_uint(f0), __float_as_uint(f1));
            uint32_t p1 = pack_bf16_fn(__float_as_uint(f2), __float_as_uint(f3));
            uint32_t p2 = pack_bf16_fn(__float_as_uint(f4), __float_as_uint(f5));
            uint32_t p3 = pack_bf16_fn(__float_as_uint(f6), __float_as_uint(f7));
            
            uint32_t store_col = tmem_base + 256 + c / 2;
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
               :: "r"(store_col), "r"(p0), "r"(p1), "r"(p2), "r"(p3) : "memory");
        }
        __syncthreads();

        // 3. dP^T = V @ dO^T 
        if (threadIdx.x == 0) {
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            issue_umma_128x128(tmem_base + 384, 0, smem_V, 0, 0, smem_dO, 0);
            umma_commit_and_wait(mbar_umma, umma_phase); umma_phase ^= 1;
        }
        __syncthreads();
        
        // 4. dS^T 
        for (uint32_t c = 0; c < 128; c += 8) {
            uint32_t dp0, dp1, dp2, dp3, dp4, dp5, dp6, dp7;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(dp0),"=r"(dp1),"=r"(dp2),"=r"(dp3),"=r"(dp4),"=r"(dp5),"=r"(dp6),"=r"(dp7) 
               : "r"(tmem_base + 384 + c));
               
            uint32_t p0, p1, p2, p3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(p0),"=r"(p1),"=r"(p2),"=r"(p3) : "r"(tmem_base + 256 + c / 2));
               
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float ds0 = __bfloat162float(*(__nv_bfloat16*)&p0) * (__uint_as_float(dp0) - smem_D[c + 0]) * scale;
            float ds1 = __bfloat162float(*((__nv_bfloat16*)&p0 + 1)) * (__uint_as_float(dp1) - smem_D[c + 1]) * scale;
            float ds2 = __bfloat162float(*(__nv_bfloat16*)&p1) * (__uint_as_float(dp2) - smem_D[c + 2]) * scale;
            float ds3 = __bfloat162float(*((__nv_bfloat16*)&p1 + 1)) * (__uint_as_float(dp3) - smem_D[c + 3]) * scale;
            float ds4 = __bfloat162float(*(__nv_bfloat16*)&p2) * (__uint_as_float(dp4) - smem_D[c + 4]) * scale;
            float ds5 = __bfloat162float(*((__nv_bfloat16*)&p2 + 1)) * (__uint_as_float(dp5) - smem_D[c + 5]) * scale;
            float ds6 = __bfloat162float(*(__nv_bfloat16*)&p3) * (__uint_as_float(dp6) - smem_D[c + 6]) * scale;
            float ds7 = __bfloat162float(*((__nv_bfloat16*)&p3 + 1)) * (__uint_as_float(dp7) - smem_D[c + 7]) * scale;
            
            uint32_t out0 = pack_bf16_fn(__float_as_uint(ds0), __float_as_uint(ds1));
            uint32_t out1 = pack_bf16_fn(__float_as_uint(ds2), __float_as_uint(ds3));
            uint32_t out2 = pack_bf16_fn(__float_as_uint(ds4), __float_as_uint(ds5));
            uint32_t out3 = pack_bf16_fn(__float_as_uint(ds6), __float_as_uint(ds7));
            
            uint32_t chunk = c / 64;
            uint32_t c_in_chunk = c % 64;
            uint32_t x = c_in_chunk / 8;
            uint32_t swizzled_x = (row % 8) ^ x;
            uint32_t byte_offset = chunk * 16384 + row * 128 + swizzled_x * 16;
            uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared((uint8_t*)smem_dST + byte_offset);
            st_shared_128_fn(smem_addr, out0, out1, out2, out3);
        }
        __syncthreads();

        // 5. Compute dV, dK, dQ
        if (threadIdx.x == 0) {
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            uint32_t accum_dV = (q_idx == 0) ? 0 : 1;
            issue_umma_128x128(tmem_base + 0, accum_dV, NULL, tmem_base + 256, 2, smem_dO, 1);
            
            uint32_t accum_dK = (q_idx == 0) ? 0 : 1;
            issue_umma_128x128(tmem_base + 128, accum_dK, smem_dST, 0, 0, smem_Q, 1);
            
            issue_umma_128x128(tmem_base + 384, 0, smem_dST, 0, 1, smem_K, 1);
            
            umma_commit_and_wait(mbar_umma, umma_phase); umma_phase ^= 1;
        }
        __syncthreads();
        
        // 6. Write out dQ
        for (uint32_t c = 0; c < 128; c += 8) {
            uint32_t q0, q1, q2, q3, q4, q5, q6, q7;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(q0),"=r"(q1),"=r"(q2),"=r"(q3),"=r"(q4),"=r"(q5),"=r"(q6),"=r"(q7) 
               : "r"(tmem_base + 384 + c));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            uint32_t out0 = pack_bf16_fn(q0, q1);
            uint32_t out1 = pack_bf16_fn(q2, q3);
            uint32_t out2 = pack_bf16_fn(q4, q5);
            uint32_t out3 = pack_bf16_fn(q6, q7);
            
            uint32_t byte_offset = row * 256 + c * 2;
            uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared((uint8_t*)smem_dST + byte_offset); 
            st_shared_128_fn(smem_addr, out0, out1, out2, out3);
        }
        
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        __syncthreads();
        
        int global_row = q_idx * 128 + row;
        if (global_row < S) {
            __nv_bfloat162* smem_ptr = (__nv_bfloat162*)smem_dST;
            __nv_bfloat162* global_ptr = (__nv_bfloat162*)(dQ + bh_offset + global_row * 128);
            for (int col = 0; col < 64; col++) {
                atomicAdd(&global_ptr[col], smem_ptr[row * 64 + col]);
            }
        }
        __syncthreads();
    }
    
    tmem_epilogue_coalesced_4w_fn(dV + bh_offset, (__nv_bfloat16*)smem_Q, S, 128, kv_idx, 0, 128, 128, tmem_base + 0);
    tmem_epilogue_coalesced_4w_fn(dK + bh_offset, (__nv_bfloat16*)smem_Q, S, 128, kv_idx, 0, 128, 128, tmem_base + 128);
    
    if (threadIdx.x < 32) {
        tmem_dealloc_fn(tmem_base, 512);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ_tv, tvm::ffi::TensorView dK_tv, tvm::ffi::TensorView dV_tv) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int d = Q.size(3);
    float scale = 1.0f / sqrtf(d);

    __nv_bfloat16* q_ptr = static_cast<__nv_bfloat16*>(Q.data_ptr());
    __nv_bfloat16* k_ptr = static_cast<__nv_bfloat16*>(K.data_ptr());
    __nv_bfloat16* v_ptr = static_cast<__nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    __nv_bfloat16* do_ptr = static_cast<__nv_bfloat16*>(dO.data_ptr());
    float* l_ptr = static_cast<float*>(L.data_ptr());
    __nv_bfloat16* dq_ptr = static_cast<__nv_bfloat16*>(dQ_tv.data_ptr());
    __nv_bfloat16* dk_ptr = static_cast<__nv_bfloat16*>(dK_tv.data_ptr());
    __nv_bfloat16* dv_ptr = static_cast<__nv_bfloat16*>(dV_tv.data_ptr());

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    float* d_buf;
    CUDA_CHECK(cudaMallocAsync(&d_buf, B * H * S * sizeof(float), stream));
    int d_blocks = (B * H * S + 255) / 256;
    compute_D_kernel<<<d_blocks, 256, 0, stream>>>(o_ptr, do_ptr, d_buf, B, H, S, d);

    CUDA_CHECK(cudaMemsetAsync(dq_ptr, 0, B * H * S * d * sizeof(__nv_bfloat16), stream));

    CUtensorMap tma_Q, tma_K, tma_V, tma_dO;
    cuuint64_t g_dim[2] = { (cuuint64_t)d, (cuuint64_t)(B * H * S) };
    cuuint64_t g_stride[1] = { (cuuint64_t)(d * 2) };
    cuuint32_t s_dim[2] = { 64, 128 };
    cuuint32_t s_stride[2] = { 1, 1 };

    CU_CHECK(cuTensorMapEncodeTiled(&tma_Q, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, q_ptr, g_dim, g_stride, s_dim, s_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(cuTensorMapEncodeTiled(&tma_K, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, k_ptr, g_dim, g_stride, s_dim, s_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(cuTensorMapEncodeTiled(&tma_V, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, v_ptr, g_dim, g_stride, s_dim, s_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(cuTensorMapEncodeTiled(&tma_dO, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, do_ptr, g_dim, g_stride, s_dim, s_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    int num_kv_tiles = (S + 127) / 128;
    dim3 grid(B * H, num_kv_tiles);
    dim3 block(128);
    int smem_size = 5 * 32768 + 1024 + 44; 

    cudaFuncSetAttribute((const void*)bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    bwd_kernel<<<grid, block, smem_size, stream>>>(tma_Q, tma_K, tma_V, tma_dO, dq_ptr, dk_ptr, dv_ptr, l_ptr, d_buf, B, H, S, scale);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFreeAsync(d_buf, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flashinfer_bwd_d128::run);

}