#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CU_CHECK(call) do {                                        \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        const char* err_str = "";                                  \
        cuGetErrorString(_e, &err_str);                            \
        fprintf(stderr, "CU error %d %s at %s:%d\n",               \
                (int)_e, err_str, __FILE__, __LINE__);             \
        exit(1);                                                   \
    }                                                              \
} while(0)

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace {

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

__device__ __forceinline__ void tma_load_4d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(a), "r"(ncols) : "memory");
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(addr), "r"(ncols) : "memory");
}

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ void tmem_load_8x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3,
                                                uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3),
                   "=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7) : "r"(col));
}

__device__ __forceinline__ void tmem_st_4x_fn(uint32_t col, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};"
                 :: "r"(r0),"r"(r1),"r"(r2),"r"(r3), "r"(col) : "memory");
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_store_fence_fn() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_none(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)0 << 61; // SWIZZLE_NONE
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, bool trans_a, bool trans_b) {
    uint32_t d = 0;
    d |= (1u << 4);    // FP32
    d |= (1u << 7);    // BF16
    d |= (1u << 10);   // BF16
    if (trans_a) d |= (1u << 15);
    if (trans_b) d |= (1u << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ void umma_f16_cg1_fn(uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_f16_tmem_a_cg1_fn(uint32_t tmem_c, uint32_t tmem_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}\n"
        :: "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void tcgen05_commit_cg1_fn(uint64_t* mbar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(a) : "memory");
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ uint32_t pack_bf16_fn(float f_a, float f_b) {
    __nv_bfloat16 a = __float2bfloat16(f_a);
    __nv_bfloat16 b = __float2bfloat16(f_b);
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};" : "=r"(result) : "h"(*reinterpret_cast<uint16_t*>(&a)), "h"(*reinterpret_cast<uint16_t*>(&b)));
    return result;
}

CUresult create_tma_4d_descriptor_none(CUtensorMap* d_map, void* globalAddress, 
    uint64_t dim_inner, uint64_t dim_s, uint64_t dim_h, uint64_t dim_b, 
    uint64_t stride_s, uint64_t stride_h, uint64_t stride_b, 
    uint32_t smem_inner, uint32_t smem_outer) {
    
    cuuint64_t globalDim[4] = {dim_inner, dim_s, dim_h, dim_b}; 
    cuuint64_t globalStrides[3] = {stride_s * 2, stride_h * 2, stride_b * 2}; // bytes
    cuuint32_t boxDim[4] = {smem_inner, smem_outer, 1, 1};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

__global__ void mha_bwd_d128_kernel(
    CUtensorMap tma_Q, CUtensorMap tma_K, CUtensorMap tma_V,
    CUtensorMap tma_dO, CUtensorMap tma_O,
    const float* L_ptr, __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    int B, int H, int S, int d,
    long long L_stride_s, long long L_stride_h, long long L_stride_b,
    long long out_stride_s, long long out_stride_h, long long out_stride_b)
{
    extern __shared__ __align__(1024) char smem_buf[];
    __nv_bfloat16* smem_Q = (__nv_bfloat16*)(smem_buf);                     
    __nv_bfloat16* smem_K = (__nv_bfloat16*)(smem_buf + 32768);             
    __nv_bfloat16* smem_V = (__nv_bfloat16*)(smem_buf + 65536);             
    __nv_bfloat16* smem_dO = (__nv_bfloat16*)(smem_buf + 98304);            
    __nv_bfloat16* smem_O = (__nv_bfloat16*)(smem_buf + 131072);            
    __nv_bfloat16* smem_dS = (__nv_bfloat16*)(smem_buf + 163840);           
    float* smem_L = (float*)(smem_buf + 196608);                            
    float* smem_D = (float*)(smem_buf + 196608 + 512);                      
    uint64_t* mbar = (uint64_t*)(smem_buf + 196608 + 1024);
    
    int tid = threadIdx.x;
    if (tid < 4) {
        init_smem_barrier_fn(&mbar[tid], 1);
    }
    __syncthreads();
    if (tid == 0) fence_smem_barrier_init_fn();
    __syncthreads();

    int b = blockIdx.y / H;
    int h = blockIdx.y % H;
    int j_block = blockIdx.x; 
    int j_base = j_block * 128;
    if (j_base >= S) return;

    uint32_t tmem_base;
    if (tid == 0) tmem_alloc_fn(&tmem_base, 512);
    __syncthreads();

    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(&mbar[0], 2 * 128 * 128 * 2); 
        tma_load_4d_fn(&tma_K, &mbar[0], smem_K, 0, j_base, h, b);
        tma_load_4d_fn(&tma_V, &mbar[0], smem_V, 0, j_base, h, b);
    }
    mbarrier_wait_fn(&mbar[0], 0);
    __syncthreads();
    
    if (j_base + tid >= S) {
        for (int k = 0; k < 128; k++) {
            smem_K[tid * 128 + k] = __float2bfloat16(0.0f);
            smem_V[tid * 128 + k] = __float2bfloat16(0.0f);
        }
    }
    __syncthreads();

    int mma_phase = 0;

    for (int i_block = j_block; i_block < (S + 127) / 128; ++i_block) {
        int i_base = i_block * 128;
        
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(&mbar[1], 3 * 128 * 128 * 2);
            tma_load_4d_fn(&tma_Q, &mbar[1], smem_Q, 0, i_base, h, b);
            tma_load_4d_fn(&tma_dO, &mbar[1], smem_dO, 0, i_base, h, b);
            tma_load_4d_fn(&tma_O, &mbar[1], smem_O, 0, i_base, h, b);
        }
        if (i_base + tid < S) {
            smem_L[tid] = L_ptr[b * L_stride_b + h * L_stride_h + (i_base + tid) * L_stride_s];
        } else {
            smem_L[tid] = -1e20f;
        }
        
        mbarrier_wait_fn(&mbar[1], (i_block - j_block) % 2);
        __syncthreads();

        if (i_base + tid >= S) {
            for (int k = 0; k < 128; k++) {
                smem_Q[tid * 128 + k] = __float2bfloat16(0.0f);
                smem_dO[tid * 128 + k] = __float2bfloat16(0.0f);
                smem_O[tid * 128 + k] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        float D_val = 0;
        for (int k = 0; k < 128; k++) {
            float o_val = __bfloat162float(smem_O[tid * 128 + k]);
            float do_val = __bfloat162float(smem_dO[tid * 128 + k]);
            D_val += o_val * do_val;
        }
        smem_D[tid] = D_val;
        __syncthreads();

        uint32_t idesc_S_T = make_instr_desc_fn(128, 128, false, false); 
        uint32_t a_addr = (uint32_t)__cvta_generic_to_shared(smem_K);
        uint32_t b_addr = (uint32_t)__cvta_generic_to_shared(smem_Q);
        for (int k = 0; k < 128; k += 16) {
            uint64_t desc_a = make_smem_desc_none((void*)a_addr, 2048, 128); 
            uint64_t desc_b = make_smem_desc_none((void*)b_addr, 2048, 128); 
            if (tid == 0) umma_f16_cg1_fn(256, desc_a, desc_b, idesc_S_T, (k == 0) ? 0 : 1);
            a_addr += 32; b_addr += 32;
        }
        
        if (tid == 0) tcgen05_commit_cg1_fn(&mbar[3]);
        mbarrier_wait_fn(&mbar[3], mma_phase); mma_phase ^= 1;

        for (int col = 0; col < 128; col += 8) {
            uint32_t r[8];
            tmem_load_8x_fn(256 + col, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
            tmem_load_fence_fn();
            for (int c = 0; c < 8; c++) {
                float f = __uint_as_float(r[c]);
                float val = f * 0.0883883476f; 
                int g_row_Q = i_base + col + c;
                int g_row_K = j_base + tid;
                float p = 0.0f;
                if (g_row_K <= g_row_Q && g_row_Q < S && g_row_K < S) {
                    p = fast_exp2f_fn((val - smem_L[col + c]) * 1.44269504f);
                }
                r[c] = __float_as_uint(p);
            }
            uint32_t p0 = pack_bf16_fn(__uint_as_float(r[0]), __uint_as_float(r[1]));
            uint32_t p1 = pack_bf16_fn(__uint_as_float(r[2]), __uint_as_float(r[3]));
            uint32_t p2 = pack_bf16_fn(__uint_as_float(r[4]), __uint_as_float(r[5]));
            uint32_t p3 = pack_bf16_fn(__uint_as_float(r[6]), __uint_as_float(r[7]));
            tmem_st_4x_fn(256 + col/2, p0, p1, p2, p3);
        }
        tmem_store_fence_fn();

        uint32_t idesc_dV = make_instr_desc_fn(128, 128, false, true); 
        uint32_t tmem_a = 256;
        b_addr = (uint32_t)__cvta_generic_to_shared(smem_dO);
        for (int k = 0; k < 128; k += 16) {
            uint64_t desc_b = make_smem_desc_none((void*)b_addr, 128, 2048);
            if (tid == 0) umma_f16_tmem_a_cg1_fn(128, tmem_a, desc_b, idesc_dV, (i_block == j_block && k == 0) ? 0 : 1);
            tmem_a += 8; b_addr += 4096;
        }

        uint32_t idesc_dP = make_instr_desc_fn(128, 128, false, false); 
        uint32_t a_addr_V = (uint32_t)__cvta_generic_to_shared(smem_V);
        uint32_t b_addr_dO = (uint32_t)__cvta_generic_to_shared(smem_dO);
        for (int k = 0; k < 128; k += 16) {
            uint64_t desc_a = make_smem_desc_none((void*)a_addr_V, 2048, 128);
            uint64_t desc_b = make_smem_desc_none((void*)b_addr_dO, 2048, 128);
            if (tid == 0) umma_f16_cg1_fn(384, desc_a, desc_b, idesc_dP, (k == 0) ? 0 : 1);
            a_addr_V += 32; b_addr_dO += 32;
        }

        if (tid == 0) tcgen05_commit_cg1_fn(&mbar[3]);
        mbarrier_wait_fn(&mbar[3], mma_phase); mma_phase ^= 1;

        for (int col = 0; col < 128; col += 8) {
            uint32_t p[4], dp[8];
            tmem_load_4x_fn(256 + col/2, &p[0], &p[1], &p[2], &p[3]);
            tmem_load_8x_fn(384 + col, &dp[0], &dp[1], &dp[2], &dp[3], &dp[4], &dp[5], &dp[6], &dp[7]);
            tmem_load_fence_fn();
            for (int c = 0; c < 8; c++) {
                float p_val = (c % 2 == 0) ? __bfloat162float( ((__nv_bfloat16*)&p[c/2])[0] ) : __bfloat162float( ((__nv_bfloat16*)&p[c/2])[1] );
                float dp_val = __uint_as_float(dp[c]);
                int g_row_Q = i_base + col + c;
                int g_row_K = j_base + tid;
                float ds = 0.0f;
                if (g_row_K <= g_row_Q && g_row_Q < S && g_row_K < S) {
                    ds = p_val * (dp_val - smem_D[col + c]) * 0.0883883476f;
                }
                dp[c] = __float_as_uint(ds);
            }
            uint32_t s0 = pack_bf16_fn(__uint_as_float(dp[0]), __uint_as_float(dp[1]));
            uint32_t s1 = pack_bf16_fn(__uint_as_float(dp[2]), __uint_as_float(dp[3]));
            uint32_t s2 = pack_bf16_fn(__uint_as_float(dp[4]), __uint_as_float(dp[5]));
            uint32_t s3 = pack_bf16_fn(__uint_as_float(dp[6]), __uint_as_float(dp[7]));
            tmem_st_4x_fn(384 + col/2, s0, s1, s2, s3);
        }
        tmem_store_fence_fn();

        uint32_t idesc_dK = make_instr_desc_fn(128, 128, false, true); 
        uint32_t tmem_a_dS = 384;
        uint32_t b_addr_Q = (uint32_t)__cvta_generic_to_shared(smem_Q);
        for (int k = 0; k < 128; k += 16) {
            uint64_t desc_b = make_smem_desc_none((void*)b_addr_Q, 128, 2048);
            if (tid == 0) umma_f16_tmem_a_cg1_fn(0, tmem_a_dS, desc_b, idesc_dK, (i_block == j_block && k == 0) ? 0 : 1);
            tmem_a_dS += 8; b_addr_Q += 4096;
        }

        for (uint32_t col = 0; col < 64; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(384 + col, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            uint32_t y = tid;
            uint32_t x_bytes = col * 4;
            uint32_t offset = y * 256 + x_bytes;
            *(uint4*)((char*)smem_dS + offset) = make_uint4(r0, r1, r2, r3);
        }
        __syncthreads();

        uint32_t idesc_dQ = make_instr_desc_fn(128, 128, true, true);
        uint32_t a_addr_dS = (uint32_t)__cvta_generic_to_shared(smem_dS);
        uint32_t b_addr_K = (uint32_t)__cvta_generic_to_shared(smem_K);
        for (int k = 0; k < 128; k += 16) {
            uint64_t desc_a = make_smem_desc_none((void*)a_addr_dS, 128, 2048);
            uint64_t desc_b = make_smem_desc_none((void*)b_addr_K, 128, 2048);
            if (tid == 0) umma_f16_cg1_fn(256, desc_a, desc_b, idesc_dQ, (k == 0) ? 0 : 1);
            a_addr_dS += 4096; b_addr_K += 4096;
        }

        if (tid == 0) tcgen05_commit_cg1_fn(&mbar[3]);
        mbarrier_wait_fn(&mbar[3], mma_phase); mma_phase ^= 1;

        for (uint32_t col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(256 + col, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            uint32_t base = tid * 128 + col;
            smem_Q[base + 0] = __float2bfloat16(__uint_as_float(r0));
            smem_Q[base + 1] = __float2bfloat16(__uint_as_float(r1));
            smem_Q[base + 2] = __float2bfloat16(__uint_as_float(r2));
            smem_Q[base + 3] = __float2bfloat16(__uint_as_float(r3));
        }
        __syncthreads();
        
        uint32_t warp_id = tid / 32;
        uint32_t lane = tid % 32;
        for (uint32_t step = 0; step < 32; ++step) {
            uint32_t row = step * 4 + warp_id;
            uint32_t col_start = lane * 4;
            if (row < 128 && col_start < 128 && (i_base + row) < S) {
                uint64_t data = *(uint64_t*)&smem_Q[row * 128 + col_start];
                __nv_bfloat162 d01 = *(__nv_bfloat162*)&data;
                __nv_bfloat162 d23 = *(__nv_bfloat162*)((char*)&data + 4);
                uint64_t out_idx = b * out_stride_b + h * out_stride_h + (i_base + row) * out_stride_s + col_start;
                __nv_bfloat162* out_ptr = (__nv_bfloat162*)&dQ[out_idx];
                atomicAdd(out_ptr, d01);
                atomicAdd(out_ptr + 1, d23);
            }
        }
        __syncthreads();
    }

    for (uint32_t col = 0; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn(0 + col, &r0, &r1, &r2, &r3);
        tmem_load_fence_fn();
        smem_K[tid * 128 + col + 0] = __float2bfloat16(__uint_as_float(r0));
        smem_K[tid * 128 + col + 1] = __float2bfloat16(__uint_as_float(r1));
        smem_K[tid * 128 + col + 2] = __float2bfloat16(__uint_as_float(r2));
        smem_K[tid * 128 + col + 3] = __float2bfloat16(__uint_as_float(r3));
    }
    __syncthreads();
    uint32_t warp_id = tid / 32;
    uint32_t lane = tid % 32;
    for (uint32_t step = 0; step < 32; ++step) {
        uint32_t row = step * 4 + warp_id;
        uint32_t col_start = lane * 4;
        if (row < 128 && col_start < 128 && (j_base + row) < S) {
            uint64_t data = *(uint64_t*)&smem_K[row * 128 + col_start];
            uint64_t out_idx = b * out_stride_b + h * out_stride_h + (j_base + row) * out_stride_s + col_start;
            *(uint64_t*)&dK[out_idx] = data;
        }
    }
    __syncthreads();

    for (uint32_t col = 0; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn(128 + col, &r0, &r1, &r2, &r3);
        tmem_load_fence_fn();
        smem_V[tid * 128 + col + 0] = __float2bfloat16(__uint_as_float(r0));
        smem_V[tid * 128 + col + 1] = __float2bfloat16(__uint_as_float(r1));
        smem_V[tid * 128 + col + 2] = __float2bfloat16(__uint_as_float(r2));
        smem_V[tid * 128 + col + 3] = __float2bfloat16(__uint_as_float(r3));
    }
    __syncthreads();
    for (uint32_t step = 0; step < 32; ++step) {
        uint32_t row = step * 4 + warp_id;
        uint32_t col_start = lane * 4;
        if (row < 128 && col_start < 128 && (j_base + row) < S) {
            uint64_t data = *(uint64_t*)&smem_V[row * 128 + col_start];
            uint64_t out_idx = b * out_stride_b + h * out_stride_h + (j_base + row) * out_stride_s + col_start;
            *(uint64_t*)&dV[out_idx] = data;
        }
    }
    __syncthreads();

    if (tid == 0) tmem_dealloc_fn(tmem_base, 512);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int d = Q.size(3);

    CUDA_CHECK(cudaMemsetAsync(dQ.data_ptr(), 0, B * H * S * d * sizeof(__nv_bfloat16), stream));

    auto get_strides = [](tvm::ffi::TensorView t, int64_t& s_S, int64_t& s_H, int64_t& s_B) {
        const DLTensor* dl = t.operator->();
        s_S = dl->strides ? dl->strides[2] : dl->shape[3];
        s_H = dl->strides ? dl->strides[1] : dl->shape[2] * dl->shape[3];
        s_B = dl->strides ? dl->strides[0] : dl->shape[1] * dl->shape[2] * dl->shape[3];
    };

    int64_t qs, qh, qb, ks, kh, kb, vs, vh, vb, dos, doh, dob, os, oh, ob;
    get_strides(Q, qs, qh, qb);
    get_strides(K, ks, kh, kb);
    get_strides(V, vs, vh, vb);
    get_strides(dO, dos, doh, dob);
    get_strides(O, os, oh, ob);

    int64_t ls, lh, lb;
    const DLTensor* dl_L = L.operator->();
    ls = dl_L->strides ? dl_L->strides[2] : 1;
    lh = dl_L->strides ? dl_L->strides[1] : dl_L->shape[2];
    lb = dl_L->strides ? dl_L->strides[0] : dl_L->shape[1] * dl_L->shape[2];

    int64_t out_s, out_h, out_b;
    get_strides(dQ, out_s, out_h, out_b);

    CUtensorMap tma_Q, tma_K, tma_V, tma_dO, tma_O;
    CU_CHECK(create_tma_4d_descriptor_none(&tma_Q, Q.data_ptr(), d, S, H, B, qs, qh, qb, 128, 128));
    CU_CHECK(create_tma_4d_descriptor_none(&tma_K, K.data_ptr(), d, S, H, B, ks, kh, kb, 128, 128));
    CU_CHECK(create_tma_4d_descriptor_none(&tma_V, V.data_ptr(), d, S, H, B, vs, vh, vb, 128, 128));
    CU_CHECK(create_tma_4d_descriptor_none(&tma_dO, dO.data_ptr(), d, S, H, B, dos, doh, dob, 128, 128));
    CU_CHECK(create_tma_4d_descriptor_none(&tma_O, O.data_ptr(), d, S, H, B, os, oh, ob, 128, 128));

    dim3 grid((S + 127) / 128, B * H);
    dim3 block(128);
    int smem_size = 200 * 1024;

    CUDA_CHECK(cudaFuncSetAttribute((void*)mha_bwd_d128_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    mha_bwd_d128_kernel<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, tma_dO, tma_O,
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        B, H, S, d,
        ls, lh, lb,
        out_s, out_h, out_b
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}