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
                _e, __FILE__, __LINE__);                           \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_example_cuda {

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

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

CUresult create_tma_4d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t dim0, uint64_t dim1, uint64_t dim2, uint64_t dim3, uint32_t box0, uint32_t box1, CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[4] = {dim0, dim1, dim2, dim3};
    cuuint64_t globalStrides[3] = {dim0 * 2, dim0 * dim1 * 2, dim0 * dim1 * dim2 * 2};
    cuuint32_t boxDim[4] = {box0, box1, 1, 1};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, globalAddress,
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

__device__ __forceinline__ void tma_load_4d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}

__device__ __forceinline__ void tma_store_4d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group [%0, {%2, %3, %4, %5}], [%1];"
        :: "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}

__device__ __forceinline__ void tma_store_fence_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tma_store_commit_fn() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void tma_store_wait_fn() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
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

__device__ __forceinline__ void tmem_load_8x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3, uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
        : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3),"=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7) : "r"(col));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void umma_f16_cg1_fn(uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_cg1_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        :: "r"(a));
}

__device__ __forceinline__ uint64_t make_smem_desc_k_major(void* smem_ptr, uint32_t row_stride_bytes) {
    uint32_t lbo = 16;
    uint32_t sbo = 8 * row_stride_bytes;
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)0 << 61; // SWIZZLE_NONE
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_mn_major(void* smem_ptr, uint32_t row_stride_bytes) {
    uint32_t lbo = 16;
    uint32_t sbo = row_stride_bytes;
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)0 << 61; // SWIZZLE_NONE
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, bool is_mn_major) {
    uint32_t lbo = is_mn_major ? 8192 : 1;
    uint32_t sbo = 1024;
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61; // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, bool trans_a, bool trans_b) {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    if (trans_a) d |= (1u << 15);
    if (trans_b) d |= (1u << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ uint64_t advance_desc(uint64_t desc, uint32_t offset_bytes) {
    uint32_t addr = (desc & 0x3FFFF) << 4;
    addr += offset_bytes;
    desc &= ~0x3FFFFull;
    desc |= (addr >> 4);
    uint32_t base_offset = (addr >> 7) & 0x7;
    desc &= ~(0x7ull << 49);
    desc |= ((uint64_t)base_offset << 49);
    return desc;
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ uint32_t pack_bf16_fn(uint32_t fp32_a, uint32_t fp32_b) {
    __nv_bfloat16 a = __float2bfloat16(__uint_as_float(fp32_a));
    __nv_bfloat16 b = __float2bfloat16(__uint_as_float(fp32_b));
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};" : "=r"(result) : "h"(*reinterpret_cast<uint16_t*>(&a)), "h"(*reinterpret_cast<uint16_t*>(&b)));
    return result;
}

__device__ __forceinline__ void load_chunk_16B(void* smem_base, uint32_t r, uint32_t c, uint4& data) {
    uint32_t phys_c = (r % 8) ^ c;
    uint32_t offset = r * 128 + phys_c * 16;
    data = *reinterpret_cast<uint4*>((char*)smem_base + offset);
}

__device__ __forceinline__ void store_chunk_16B(void* smem_base, uint32_t r, uint32_t c, uint4 data) {
    uint32_t phys_c = (r % 8) ^ c;
    uint32_t offset = r * 128 + phys_c * 16;
    *reinterpret_cast<uint4*>((char*)smem_base + offset) = data;
}

__device__ __forceinline__ void load_chunk_16B_unswizzled(void* smem_base, uint32_t r, uint32_t c, uint4& data, uint32_t row_stride_bytes) {
    uint32_t offset = r * row_stride_bytes + c * 16;
    data = *reinterpret_cast<uint4*>((char*)smem_base + offset);
}

__device__ __forceinline__ void store_chunk_16B_unswizzled(void* smem_base, uint32_t r, uint32_t c, uint4 data, uint32_t row_stride_bytes) {
    uint32_t offset = r * row_stride_bytes + c * 16;
    *reinterpret_cast<uint4*>((char*)smem_base + offset) = data;
}

__global__ __launch_bounds__(128, 1) void mha_bwd_pass1_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_dQ,
    const float* L, float scale, int S, int H)
{
    int s_block = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    int s_idx = s_block * 64;
    
    extern __shared__ char smem_buf_raw[];
    char* smem_buf = (char*)(((uintptr_t)smem_buf_raw + 127) & ~127);
    
    __nv_bfloat16* Q_smem  = (__nv_bfloat16*)(smem_buf + 0 * 16384);
    __nv_bfloat16* dO_smem = (__nv_bfloat16*)(smem_buf + 1 * 16384);
    __nv_bfloat16* O_smem  = (__nv_bfloat16*)(smem_buf + 2 * 16384);
    __nv_bfloat16* K_smem  = (__nv_bfloat16*)(smem_buf + 3 * 16384);
    __nv_bfloat16* V_smem  = (__nv_bfloat16*)(smem_buf + 4 * 16384);
    __nv_bfloat16* dS_smem = (__nv_bfloat16*)(smem_buf + 5 * 16384); 
    float* D_smem = (float*)(smem_buf + 5 * 16384 + 8192); 
    float* L_smem = D_smem + 64;                    
    
    uintptr_t mbar_ptr = (uintptr_t)(L_smem + 64);
    mbar_ptr = (mbar_ptr + 7) & ~7;
    uint64_t* mbar = (uint64_t*)mbar_ptr;
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&mbar[0], 1);
        init_smem_barrier_fn(&mbar[1], 1);
        init_smem_barrier_fn(&mbar[2], 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&mbar[0], 49152);
        tma_load_4d_fn(&tma_Q, &mbar[0], Q_smem, 0, s_idx, h, b);
        tma_load_4d_fn(&tma_dO, &mbar[0], dO_smem, 0, s_idx, h, b);
        tma_load_4d_fn(&tma_O, &mbar[0], O_smem, 0, s_idx, h, b);
    }
    
    if (threadIdx.x < 64) {
        int l_idx = (b * H * S) + (h * S) + s_idx + threadIdx.x;
        L_smem[threadIdx.x] = L[l_idx];
    }
    
    mbarrier_wait_fn(&mbar[0], 0);
    
    float D_val = 0;
    uint32_t r_d = threadIdx.x / 2;
    uint32_t c_start = (threadIdx.x % 2) * 8;
    
    if (r_d < 64) {
        for (int i = 0; i < 8; ++i) {
            uint4 O_chunk, dO_chunk;
            load_chunk_16B_unswizzled(O_smem, r_d, c_start + i, O_chunk, 256);
            load_chunk_16B_unswizzled(dO_smem, r_d, c_start + i, dO_chunk, 256);
            uint32_t* O_regs = (uint32_t*)&O_chunk;
            uint32_t* dO_regs = (uint32_t*)&dO_chunk;
            for (int j = 0; j < 4; ++j) {
                float2 o_f2 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&O_regs[j]));
                float2 do_f2 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&dO_regs[j]));
                D_val += o_f2.x * do_f2.x;
                D_val += o_f2.y * do_f2.y;
            }
        }
        D_val += __shfl_xor_sync(0xffffffff, D_val, 1);
        if (threadIdx.x % 2 == 0) D_smem[r_d] = D_val;
    }
    __syncthreads();
    
    __shared__ uint32_t shared_tmem_base;
    if (threadIdx.x < 32) {
        tmem_alloc_fn(&shared_tmem_base, 256);
    }
    __syncthreads();
    uint32_t tmem_base = shared_tmem_base;
    
    uint32_t S_tmem  = tmem_base;
    uint32_t dP_tmem = tmem_base + 64;
    uint32_t dQ_tmem = tmem_base + 128;
    
    uint32_t idesc_S  = make_instr_desc_fn(64, 64, false, false);
    uint32_t idesc_dQ = make_instr_desc_fn(64, 128, false, true);
    
    int num_n_blocks = S / 64;
    
    for (int n_step = 0; n_step < num_n_blocks; ++n_step) {
        int n_idx = n_step * 64;
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&mbar[1], 32768);
            tma_load_4d_fn(&tma_K, &mbar[1], K_smem, 0, n_idx, h, b);
            tma_load_4d_fn(&tma_V, &mbar[1], V_smem, 0, n_idx, h, b);
        }
        mbarrier_wait_fn(&mbar[1], n_step % 2);
        
        if (threadIdx.x == 0) {
            uint64_t desc_Q_S = make_smem_desc_k_major(Q_smem, 256);
            uint64_t desc_K_S = make_smem_desc_k_major(K_smem, 256);
            uint64_t desc_dO_dP = make_smem_desc_k_major(dO_smem, 256);
            uint64_t desc_V_dP = make_smem_desc_k_major(V_smem, 256);
            
            for (int k = 0; k < 128; k += 16) {
                uint32_t accum = (k == 0) ? 0 : 1;
                umma_f16_cg1_fn(S_tmem, desc_Q_S, desc_K_S, idesc_S, accum);
                umma_f16_cg1_fn(dP_tmem, desc_dO_dP, desc_V_dP, idesc_S, accum);
                desc_Q_S = advance_desc(desc_Q_S, 32);
                desc_K_S = advance_desc(desc_K_S, 32);
                desc_dO_dP = advance_desc(desc_dO_dP, 32);
                desc_V_dP = advance_desc(desc_V_dP, 32);
            }
            umma_commit_cg1_fn(&mbar[2]);
        }
        mbarrier_wait_fn(&mbar[2], (n_step * 2) % 2);
        
        uint32_t warp_id = threadIdx.x / 32;
        uint32_t lane_id = threadIdx.x % 32;
        if (warp_id < 2) {
            uint32_t row = warp_id * 32 + lane_id;
            float l_v = L_smem[row];
            float d_v = D_smem[row];
            
            for (int c = 0; c < 64; c += 8) {
                uint32_t s[8], dp[8];
                tmem_load_8x_fn(S_tmem + c, &s[0], &s[1], &s[2], &s[3], &s[4], &s[5], &s[6], &s[7]);
                tmem_load_8x_fn(dP_tmem + c, &dp[0], &dp[1], &dp[2], &dp[3], &dp[4], &dp[5], &dp[6], &dp[7]);
                tmem_load_fence_fn();
                
                uint32_t ds_bf16[4];
                for (int i = 0; i < 8; i += 2) {
                    float p0 = fast_exp2f_fn((__uint_as_float(s[i]) * scale - l_v) * 1.44269504f);
                    float p1 = fast_exp2f_fn((__uint_as_float(s[i+1]) * scale - l_v) * 1.44269504f);
                    float ds0 = p0 * (__uint_as_float(dp[i]) - d_v) * scale;
                    float ds1 = p1 * (__uint_as_float(dp[i+1]) - d_v) * scale;
                    ds_bf16[i/2] = pack_bf16_fn(__float_as_uint(ds0), __float_as_uint(ds1));
                }
                uint4 ds_vec = make_uint4(ds_bf16[0], ds_bf16[1], ds_bf16[2], ds_bf16[3]);
                store_chunk_16B(dS_smem, row, c / 8, ds_vec);
            }
        }
        __syncthreads();
        fence_async_shared_fn();
        
        if (threadIdx.x == 0) {
            uint64_t desc_dS_dQ = make_smem_desc_sm100_fn(dS_smem, false);
            uint64_t desc_K_MN = make_smem_desc_mn_major(K_smem, 256);
            for (int k = 0; k < 64; k += 16) {
                uint32_t accum = (n_step == 0 && k == 0) ? 0 : 1;
                umma_f16_cg1_fn(dQ_tmem, desc_dS_dQ, desc_K_MN, idesc_dQ, accum);
                desc_dS_dQ = advance_desc(desc_dS_dQ, 32);
                desc_K_MN = advance_desc(desc_K_MN, 4096);
            }
            umma_commit_cg1_fn(&mbar[2]);
        }
        mbarrier_wait_fn(&mbar[2], (n_step * 2 + 1) % 2);
    }
    
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    if (warp_id < 2) {
        uint32_t row = warp_id * 32 + lane_id;
        for (int c = 0; c < 128; c += 8) {
            uint32_t dq[8];
            tmem_load_8x_fn(dQ_tmem + c, &dq[0], &dq[1], &dq[2], &dq[3], &dq[4], &dq[5], &dq[6], &dq[7]);
            tmem_load_fence_fn();
            uint32_t dq_bf16[4];
            for (int i = 0; i < 8; i += 2) {
                dq_bf16[i/2] = pack_bf16_fn(dq[i], dq[i+1]);
            }
            uint4 dq_vec = make_uint4(dq_bf16[0], dq_bf16[1], dq_bf16[2], dq_bf16[3]);
            store_chunk_16B_unswizzled(Q_smem, row, c / 8, dq_vec, 256);
        }
    }
    __syncthreads();
    tma_store_fence_fn();
    
    if (threadIdx.x == 0) {
        tma_store_4d_fn(&tma_dQ, Q_smem, 0, s_idx, h, b);
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
    }
    
    __syncthreads();
    if (threadIdx.x < 32) {
        tmem_dealloc_fn(tmem_base, 256);
    }
}

__global__ __launch_bounds__(128, 1) void mha_bwd_pass2_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_dK,
    const __grid_constant__ CUtensorMap tma_dV,
    const float* L, float scale, int S, int H)
{
    int n_block = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    int n_idx = n_block * 64;
    
    extern __shared__ char smem_buf_raw[];
    char* smem_buf = (char*)(((uintptr_t)smem_buf_raw + 127) & ~127);
    
    __nv_bfloat16* K_smem  = (__nv_bfloat16*)(smem_buf + 0 * 16384);
    __nv_bfloat16* V_smem  = (__nv_bfloat16*)(smem_buf + 1 * 16384);
    __nv_bfloat16* Q_smem  = (__nv_bfloat16*)(smem_buf + 2 * 16384);
    __nv_bfloat16* dO_smem = (__nv_bfloat16*)(smem_buf + 3 * 16384);
    __nv_bfloat16* O_smem  = (__nv_bfloat16*)(smem_buf + 4 * 16384);
    __nv_bfloat16* dS_smem = (__nv_bfloat16*)(smem_buf + 5 * 16384);
    __nv_bfloat16* P_smem  = (__nv_bfloat16*)(smem_buf + 5 * 16384 + 8192);
    
    float* D_smem = (float*)(smem_buf + 6 * 16384); 
    float* L_smem = D_smem + 64;                    
    
    uintptr_t mbar_ptr = (uintptr_t)(L_smem + 64);
    mbar_ptr = (mbar_ptr + 7) & ~7;
    uint64_t* mbar = (uint64_t*)mbar_ptr;
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&mbar[0], 1);
        init_smem_barrier_fn(&mbar[1], 1);
        init_smem_barrier_fn(&mbar[2], 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&mbar[0], 32768);
        tma_load_4d_fn(&tma_K, &mbar[0], K_smem, 0, n_idx, h, b);
        tma_load_4d_fn(&tma_V, &mbar[0], V_smem, 0, n_idx, h, b);
    }
    mbarrier_wait_fn(&mbar[0], 0);
    
    __shared__ uint32_t shared_tmem_base;
    if (threadIdx.x < 32) {
        tmem_alloc_fn(&shared_tmem_base, 384); 
    }
    __syncthreads();
    uint32_t tmem_base = shared_tmem_base;
    
    uint32_t S_tmem  = tmem_base;
    uint32_t dP_tmem = tmem_base + 64;
    uint32_t dK_tmem = tmem_base + 128;
    uint32_t dV_tmem = tmem_base + 256;
    
    uint32_t idesc_S  = make_instr_desc_fn(64, 64, false, false);
    uint32_t idesc_dK = make_instr_desc_fn(64, 128, true, true);
    
    int num_m_blocks = S / 64;
    for (int m_step = 0; m_step < num_m_blocks; ++m_step) {
        int m_idx = m_step * 64;
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&mbar[1], 49152);
            tma_load_4d_fn(&tma_Q, &mbar[1], Q_smem, 0, m_idx, h, b);
            tma_load_4d_fn(&tma_dO, &mbar[1], dO_smem, 0, m_idx, h, b);
            tma_load_4d_fn(&tma_O, &mbar[1], O_smem, 0, m_idx, h, b);
        }
        if (threadIdx.x < 64) {
            int l_idx = (b * H * S) + (h * S) + m_idx + threadIdx.x;
            L_smem[threadIdx.x] = L[l_idx];
        }
        mbarrier_wait_fn(&mbar[1], m_step % 2);
        
        float D_val = 0;
        uint32_t r_d = threadIdx.x / 2;
        uint32_t c_start = (threadIdx.x % 2) * 8;
        
        if (r_d < 64) {
            for (int i = 0; i < 8; ++i) {
                uint4 O_chunk, dO_chunk;
                load_chunk_16B_unswizzled(O_smem, r_d, c_start + i, O_chunk, 256);
                load_chunk_16B_unswizzled(dO_smem, r_d, c_start + i, dO_chunk, 256);
                uint32_t* O_regs = (uint32_t*)&O_chunk;
                uint32_t* dO_regs = (uint32_t*)&dO_chunk;
                for (int j = 0; j < 4; ++j) {
                    float2 o_f2 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&O_regs[j]));
                    float2 do_f2 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&dO_regs[j]));
                    D_val += o_f2.x * do_f2.x;
                    D_val += o_f2.y * do_f2.y;
                }
            }
            D_val += __shfl_xor_sync(0xffffffff, D_val, 1);
            if (threadIdx.x % 2 == 0) D_smem[r_d] = D_val;
        }
        __syncthreads();
        
        if (threadIdx.x == 0) {
            uint64_t desc_Q_S = make_smem_desc_k_major(Q_smem, 256);
            uint64_t desc_K_S = make_smem_desc_k_major(K_smem, 256);
            uint64_t desc_dO_dP = make_smem_desc_k_major(dO_smem, 256);
            uint64_t desc_V_dP = make_smem_desc_k_major(V_smem, 256);
            
            for (int k = 0; k < 128; k += 16) {
                uint32_t accum = (k == 0) ? 0 : 1;
                umma_f16_cg1_fn(S_tmem, desc_Q_S, desc_K_S, idesc_S, accum);
                umma_f16_cg1_fn(dP_tmem, desc_dO_dP, desc_V_dP, idesc_S, accum);
                desc_Q_S = advance_desc(desc_Q_S, 32);
                desc_K_S = advance_desc(desc_K_S, 32);
                desc_dO_dP = advance_desc(desc_dO_dP, 32);
                desc_V_dP = advance_desc(desc_V_dP, 32);
            }
            umma_commit_cg1_fn(&mbar[2]);
        }
        mbarrier_wait_fn(&mbar[2], (m_step * 2) % 2);
        
        uint32_t warp_id = threadIdx.x / 32;
        uint32_t lane_id = threadIdx.x % 32;
        if (warp_id < 2) {
            uint32_t row = warp_id * 32 + lane_id;
            float l_v = L_smem[row];
            float d_v = D_smem[row];
            
            for (int c = 0; c < 64; c += 8) {
                uint32_t s[8], dp[8];
                tmem_load_8x_fn(S_tmem + c, &s[0], &s[1], &s[2], &s[3], &s[4], &s[5], &s[6], &s[7]);
                tmem_load_8x_fn(dP_tmem + c, &dp[0], &dp[1], &dp[2], &dp[3], &dp[4], &dp[5], &dp[6], &dp[7]);
                tmem_load_fence_fn();
                
                uint32_t p_bf16[4], ds_bf16[4];
                for (int i = 0; i < 8; i += 2) {
                    float p0 = fast_exp2f_fn((__uint_as_float(s[i]) * scale - l_v) * 1.44269504f);
                    float p1 = fast_exp2f_fn((__uint_as_float(s[i+1]) * scale - l_v) * 1.44269504f);
                    float ds0 = p0 * (__uint_as_float(dp[i]) - d_v) * scale;
                    float ds1 = p1 * (__uint_as_float(dp[i+1]) - d_v) * scale;
                    p_bf16[i/2] = pack_bf16_fn(__float_as_uint(p0), __float_as_uint(p1));
                    ds_bf16[i/2] = pack_bf16_fn(__float_as_uint(ds0), __float_as_uint(ds1));
                }
                store_chunk_16B(P_smem, row, c / 8, make_uint4(p_bf16[0], p_bf16[1], p_bf16[2], p_bf16[3]));
                store_chunk_16B(dS_smem, row, c / 8, make_uint4(ds_bf16[0], ds_bf16[1], ds_bf16[2], ds_bf16[3]));
            }
        }
        __syncthreads();
        fence_async_shared_fn();
        
        if (threadIdx.x == 0) {
            uint64_t desc_dS_MN = make_smem_desc_sm100_fn(dS_smem, true);
            uint64_t desc_Q_MN = make_smem_desc_mn_major(Q_smem, 256);
            uint64_t desc_P_MN = make_smem_desc_sm100_fn(P_smem, true);
            uint64_t desc_dO_MN = make_smem_desc_mn_major(dO_smem, 256);
            
            for (int k = 0; k < 64; k += 16) {
                uint32_t accum = (m_step == 0 && k == 0) ? 0 : 1;
                umma_f16_cg1_fn(dK_tmem, desc_dS_MN, desc_Q_MN, idesc_dK, accum);
                umma_f16_cg1_fn(dV_tmem, desc_P_MN, desc_dO_MN, idesc_dK, accum);
                desc_dS_MN = advance_desc(desc_dS_MN, 2048);
                desc_Q_MN = advance_desc(desc_Q_MN, 4096);
                desc_P_MN = advance_desc(desc_P_MN, 2048);
                desc_dO_MN = advance_desc(desc_dO_MN, 4096);
            }
            umma_commit_cg1_fn(&mbar[2]);
        }
        mbarrier_wait_fn(&mbar[2], (m_step * 2 + 1) % 2);
    }
    
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    if (warp_id < 2) {
        uint32_t row = warp_id * 32 + lane_id;
        for (int c = 0; c < 128; c += 8) {
            uint32_t dk[8], dv[8];
            tmem_load_8x_fn(dK_tmem + c, &dk[0], &dk[1], &dk[2], &dk[3], &dk[4], &dk[5], &dk[6], &dk[7]);
            tmem_load_8x_fn(dV_tmem + c, &dv[0], &dv[1], &dv[2], &dv[3], &dv[4], &dv[5], &dv[6], &dv[7]);
            tmem_load_fence_fn();
            uint32_t dk_bf16[4], dv_bf16[4];
            for (int i = 0; i < 8; i += 2) {
                dk_bf16[i/2] = pack_bf16_fn(dk[i], dk[i+1]);
                dv_bf16[i/2] = pack_bf16_fn(dv[i], dv[i+1]);
            }
            store_chunk_16B_unswizzled(K_smem, row, c / 8, make_uint4(dk_bf16[0], dk_bf16[1], dk_bf16[2], dk_bf16[3]), 256);
            store_chunk_16B_unswizzled(V_smem, row, c / 8, make_uint4(dv_bf16[0], dv_bf16[1], dv_bf16[2], dv_bf16[3]), 256);
        }
    }
    __syncthreads();
    tma_store_fence_fn();
    
    if (threadIdx.x == 0) {
        tma_store_4d_fn(&tma_dK, K_smem, 0, n_idx, h, b);
        tma_store_4d_fn(&tma_dV, V_smem, 0, n_idx, h, b);
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
    }
    
    __syncthreads();
    if (threadIdx.x < 32) {
        tmem_dealloc_fn(tmem_base, 384);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L, tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = 128;
    
    CUtensorMap tma_Q, tma_O, tma_dO, tma_K, tma_V, tma_dQ, tma_dK, tma_dV;
    
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_Q, Q.data_ptr(), d, S, H, B, 128, 64, CU_TENSOR_MAP_SWIZZLE_NONE));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_O, O.data_ptr(), d, S, H, B, 128, 64, CU_TENSOR_MAP_SWIZZLE_NONE));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_dO, dO.data_ptr(), d, S, H, B, 128, 64, CU_TENSOR_MAP_SWIZZLE_NONE));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_K, K.data_ptr(), d, S, H, B, 128, 64, CU_TENSOR_MAP_SWIZZLE_NONE));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_V, V.data_ptr(), d, S, H, B, 128, 64, CU_TENSOR_MAP_SWIZZLE_NONE));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_dQ, dQ.data_ptr(), d, S, H, B, 128, 64, CU_TENSOR_MAP_SWIZZLE_NONE));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_dK, dK.data_ptr(), d, S, H, B, 128, 64, CU_TENSOR_MAP_SWIZZLE_NONE));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_dV, dV.data_ptr(), d, S, H, B, 128, 64, CU_TENSOR_MAP_SWIZZLE_NONE));
    
    float scale = 1.0f / sqrtf(d);
    
    dim3 grid(S / 64, H, B);
    dim3 block(128);
    
    int smem_size_pass1 = 104 * 1024;
    int smem_size_pass2 = 104 * 1024;
    
    CUDA_CHECK(cudaFuncSetAttribute(mha_bwd_pass1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_pass1));
    mha_bwd_pass1_kernel<<<grid, block, smem_size_pass1, stream>>>(
        tma_Q, tma_O, tma_dO, tma_K, tma_V, tma_dQ,
        static_cast<const float*>(L.data_ptr()), scale, S, H);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaFuncSetAttribute(mha_bwd_pass2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_pass2));
    mha_bwd_pass2_kernel<<<grid, block, smem_size_pass2, stream>>>(
        tma_Q, tma_O, tma_dO, tma_K, tma_V, tma_dK, tma_dV,
        static_cast<const float*>(L.data_ptr()), scale, S, H);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

} // namespace tvm_ffi_example_cuda