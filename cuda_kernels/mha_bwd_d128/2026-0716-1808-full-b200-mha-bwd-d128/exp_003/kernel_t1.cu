#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>
#include <algorithm>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_mha_bwd {

constexpr int H = 48;
constexpr int BM = 64;

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

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
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void umma_f16_cta1(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ uint64_t make_smem_desc(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ void store_smem_swizzled(uint8_t* smem_ptr, int row, int col, __nv_bfloat16 val) {
    int chunk_idx = (row % 8) ^ (col / 8);
    int phys_col = chunk_idx * 8 + (col % 8);
    smem_ptr[(row * 64) + phys_col] = val;
}

__device__ __forceinline__ void load_gmem_to_smem_vec(const uint8_t* smem_ptr, const __nv_bfloat16* gmem_ptr, int32_t valid_rows, int32_t stride, int32_t col_offset, int32_t tid) {
    for (int i = 0; i < 64; i += 2) {
        int32_t global_idx = (i * stride) + col_offset + tid * 2;
        if (i < valid_rows) {
            uint2 tmp = *(const uint2*)&gmem_ptr[global_idx];
            int chunk_idx = (i % 8) ^ ((tid * 2) / 8);
            int phys_col = chunk_idx * 8 + ((tid * 2) % 8);
            *(uint32_t*)&smem_ptr[(i * 64) + phys_col] = tmp.x;
            *(uint32_t*)&smem_ptr[(i * 64) + phys_col + 1] = tmp.y;
        } else {
            int chunk_idx = (i % 8) ^ ((tid * 2) / 8);
            int phys_col = chunk_idx * 8 + ((tid * 2) % 8);
            *(uint32_t*)&smem_ptr[(i * 64) + phys_col] = 0;
            *(uint32_t*)&smem_ptr[(i * 64) + phys_col + 1] = 0;
        }
    }
}

__device__ __forceinline__ void store_smem_to_gmem_vec(const __nv_bfloat16* gmem_ptr, const uint8_t* smem_ptr, int32_t valid_rows, int32_t stride, int32_t col_offset, int32_t tid) {
    for (int i = 0; i < 64; i += 2) {
        if (i < valid_rows) {
            int chunk_idx = (i % 8) ^ ((tid * 2) / 8);
            int phys_col = chunk_idx * 8 + ((tid * 2) % 8);
            uint32_t tmp = *(const uint32_t*)&smem_ptr[(i * 64) + phys_col];
            int32_t global_idx = (i * stride) + col_offset + tid * 2;
            *(uint32_t*)&gmem_ptr[global_idx] = tmp;
        }
    }
}

__device__ __forceinline__ void atomic_add_bf16_matrix(__nv_bfloat16* gmem, uint8_t* smem_ptr, int32_t valid_rows, int32_t tid) {
    for (int i = tid; i < 4096; i += 32) {
        __nv_bfloat16 val = *reinterpret_cast<__nv_bfloat16*>(&smem_ptr[(i / 64) * 64 + ((i / 64) % 8) * 8 + (i % 8)]);
        if (val != 0) {
            *(uint32_t*)&gmem[i] = *(uint32_t*)&val;
        }
    }
}

CUresult create_tma_4d_descriptor_2B(CUtensorMap* d, void* globalAddress, 
    uint64_t dim0, uint64_t dim1, uint64_t dim2, uint64_t dim3,
    uint32_t box0, uint32_t box1) {
    cuuint64_t globalDim[4] = {dim0, dim1, dim2, dim3};
    cuuint64_t globalStrides[3] = {dim0 * 2, dim0 * dim1 * 2, dim0 * dim1 * dim2 * 2};
    cuuint32_t boxDim[4] = {box0, box1, 1, 1};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

__global__ void run_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* L, 
    __nv_bfloat16* dQ, 
    float* dK_f32, 
    float* dV_f32, 
    int32_t S, float alpha)
{
    int32_t head = blockIdx.y % H;
    int32_t batch = blockIdx.y / H;
    int32_t batch_head = batch * H + head;
    
    int32_t ib = blockIdx.x * BM;
    if (ib >= S) return;
    int32_t valid_i = min(S - ib, BM);
    int32_t tid = threadIdx.x;
    
    extern __shared__ __align__(1024) uint8_t smem_pool[];
    
    uint8_t* smem_Q_0 = smem_pool + 0;
    uint8_t* smem_Q_1 = smem_pool + 8192;
    uint8_t* smem_dO_0 = smem_pool + 16384;
    uint8_t* smem_dO_1 = smem_pool + 24576;
    uint8_t* smem_K_0 = smem_pool + 32768;
    uint8_t* smem_K_1 = smem_pool + 40960;
    uint8_t* smem_V_0 = smem_pool + 49152;
    uint8_t* smem_V_1 = smem_pool + 57344;
    uint8_t* smem_dP = smem_pool + 65536;
    uint8_t* smem_P = smem_pool + 73728;
    
    float* smem_L = (float*)(smem_pool + 81920);
    uint8_t* smem_bar = (uint8_t*)(smem_pool + 82176); 
    
    uint8_t* smem_dK_0 = smem_pool + 82208;
    uint8_t* smem_dK_1 = smem_pool + 90400;
    uint8_t* smem_dV_0 = smem_pool + 98592;
    uint8_t* smem_dV_1 = smem_pool + 106784;
    
    uint32_t* smem_tmem_0 = (uint32_t*)(smem_pool + 114976);
    uint32_t* smem_tmem_1 = (uint32_t*)(smem_pool + 115008);
    uint32_t* smem_tmem_dP = (uint32_t*)(smem_pool + 115040);
    uint32_t* smem_tmem_dP_T = (uint32_t*)(smem_pool + 115072);
    uint32_t* smem_tmem_P = (uint32_t*)(smem_pool + 115104);
    
    if (tid == 0) {
        init_smem_barrier_fn((uint64_t*)smem_bar, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();
    
    uint32_t phase = 0;
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn((uint64_t*)smem_bar, 65536); 
        
        tma_load_4d_fn(&tma_Q, (uint64_t*)smem_bar, smem_Q_0, 0, ib, head, batch);
        tma_load_4d_fn(&tma_Q, (uint64_t*)smem_bar, smem_Q_1, 64, ib, head, batch);
        
        tma_load_4d_fn(&tma_dO, (uint64_t*)smem_bar, smem_dO_0, 0, ib, head, batch);
        tma_load_4d_fn(&tma_dO, (uint64_t*)smem_bar, smem_dO_1, 64, ib, head, batch);
        
        tma_load_4d_fn(&tma_O, (uint64_t*)smem_bar, smem_dK_0, 0, ib, head, batch); 
        tma_load_4d_fn(&tma_O, (uint64_t*)smem_bar, smem_dK_1, 64, ib, head, batch); 
    }
    
    if (tid < 64) {
        smem_L[tid] = (ib + tid < S) ? L[batch_head * S + ib + tid] : 0.0f;
    }
    
    mbarrier_wait_fn((uint64_t*)smem_bar, phase);
    phase ^= 1;
    
    if (tid == 0) {
        tmem_alloc_fn(smem_tmem_0, 128); 
        tmem_alloc_fn(smem_tmem_1, 128); 
        tmem_alloc_fn(smem_tmem_dP, 64);
        tmem_alloc_fn(smem_tmem_dP_T, 64);
        tmem_alloc_fn(smem_tmem_P, 64);
    }
    __syncthreads();
    
    uint32_t tmem_0 = smem_tmem_0[0];
    uint32_t tmem_1 = smem_tmem_1[0];
    uint32_t tmem_dP = smem_tmem_dP[0];
    uint32_t tmem_dP_T = smem_tmem_dP_T[0];
    uint32_t tmem_P = smem_tmem_P[0];
    
    uint32_t idesc_normal = (1<<4) | (1<<7) | (1<<10) | ((64/8)<<17) | ((64/16)<<24);
    uint32_t idesc_dQ = idesc_normal | (1<<16);
    uint32_t idesc_dK = idesc_normal | (1<<15) | (1<<16);
    uint32_t idesc_dV = idesc_normal | (1<<15) | (1<<16);
    
    // Clear accumulator in TMEM
    for (int i = tid; i < 2048; i += 128) {
        int row = (i / 32);
        int col = (i % 32);
        int chunk_idx = (row % 8) ^ (col / 8);
        int col_aligned = chunk_idx * 8 + (col % 8);
        
        asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%2], {%0,%1,%3,%4};" 
                     :: "r"(0), "r"(0), "r"(tmem_1 + (col_aligned << 16) + (row << 0)), "r"(0), "r"(0));
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    __syncthreads();
    
    for (int j_block = 0; j_block < S; j_block += 64) {
        int jb = j_block;
        int valid_j = min(S - jb, 64);
        
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn((uint64_t*)smem_bar, 65536); 
            
            tma_load_4d_fn(&tma_K, (uint64_t*)smem_bar, smem_K_0, 0, jb, head, batch);
            tma_load_4d_fn(&tma_K, (uint64_t*)smem_bar, smem_K_1, 64, jb, head, batch);
            
            tma_load_4d_fn(&tma_V, (uint64_t*)smem_bar, smem_V_0, 0, jb, head, batch);
            tma_load_4d_fn(&tma_V, (uint64_t*)smem_bar, smem_V_1, 64, jb, head, batch);
        }
        mbarrier_wait_fn((uint64_t*)smem_bar, phase);
        phase ^= 1;
        
        uint64_t desc_Q_0 = make_smem_desc(smem_Q_0, 8192, 1024);
        uint64_t desc_Q_1 = make_smem_desc(smem_Q_1, 8192, 1024);
        uint64_t desc_K_0 = make_smem_desc(smem_K_0, 8192, 1024);
        uint64_t desc_K_1 = make_smem_desc(smem_K_1, 8192, 1024);
        uint64_t desc_dO_0 = make_smem_desc(smem_dO_0, 8192, 1024);
        uint64_t desc_dO_1 = make_smem_desc(smem_dO_1, 8192, 1024);
        uint64_t desc_V_0 = make_smem_desc(smem_V_0, 8192, 1024);
        uint64_t desc_V_1 = make_smem_desc(smem_V_1, 8192, 1024);
        
        uint32_t accum = 0;
        
        for (int k_off = 0; k_off < 64; k_off += 16) {
            uint64_t a_desc = make_smem_desc(smem_Q_0 + k_off, 8192, 1024);
            uint64_t b_desc = make_smem_desc(smem_K_0 + k_off, 8192, 1024);
            umma_f16_cta1(tmem_0, a_desc, b_desc, idesc_normal, accum);
            accum = 1;
        }
        for (int k_off = 0; k_off < 64; k_off += 16) {
            uint64_t a_desc = make_smem_desc(smem_Q_1 + k_off, 8192, 1024);
            uint64_t b_desc = make_smem_desc(smem_K_1 + k_off, 8192, 1024);
            umma_f16_cta1(tmem_0, a_desc, b_desc, idesc_normal, accum);
        }
        
        accum = 0;
        for (int k_off = 0; k_off < 64; k_off += 16) {
            uint64_t a_desc = make_smem_desc(smem_dO_0 + k_off, 8192, 1024);
            uint64_t b_desc = make_smem_desc(smem_V_0 + k_off, 8192, 1024);
            umma_f16_cta1(tmem_0 + 8192, a_desc, b_desc, idesc_normal, accum);
            accum = 1;
        }
        for (int k_off = 0; k_off < 64; k_off += 16) {
            uint64_t a_desc = make_smem_desc(smem_dO_1 + k_off, 8192, 1024);
            uint64_t b_desc = make_smem_desc(smem_V_1 + k_off, 8192, 1024);
            umma_f16_cta1(tmem_0 + 8192, a_desc, b_desc, idesc_normal, accum);
        }
        
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared.b64 [%0];" 
                     :: "r"((uint32_t)__cvta_generic_to_shared(&((uint64_t*)smem_bar)[0])) : "memory");
        mbarrier_wait_fn((uint64_t*)smem_bar, phase);
        phase ^= 1;
        
        float s_reg[8][4];
        float d_reg[8][4];
        
        for (int i = tid; i < 2048; i += 128) {
            int row = (i / 32);
            int col = (i % 32);
            int chunk_idx = (row % 8) ^ (col / 8);
            int col_aligned = chunk_idx * 8 + (col % 8);
            
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_0 + (col_aligned << 16) + (row << 0)));
            tmem_load_fence_fn();
            
            s_reg[(row / 4)][(col_aligned / 8)] = __uint_as_float(r0);
            s_reg[(row / 4) + 1][(col_aligned / 8)] = __uint_as_float(r1);
            s_reg[(row / 4) + 2][(col_aligned / 8)] = __uint_as_float(r2);
            s_reg[(row / 4) + 3][(col_aligned / 8)] = __uint_as_float(r3);
            
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_0 + 8192 + (col_aligned << 16) + (row << 0)));
            tmem_load_fence_fn();
            
            d_reg[(row / 4)][(col_aligned / 8)] = __uint_as_float(r0);
            d_reg[(row / 4) + 1][(col_aligned / 8)] = __uint_as_float(r1);
            d_reg[(row / 4) + 2][(col_aligned / 8)] = __uint_as_float(r2);
            d_reg[(row / 4) + 3][(col_aligned / 8)] = __uint_as_float(r3);
        }
        
        float row_sum_corr[4] = {0, 0, 0, 0};
        for(int r = 0; r < 8; ++r) {
            for(int c = 0; c < 4; ++c) {
                int row_idx = (tid % 64);
                int col_idx = c * 8 + (r * 2); // roughly mapping back to original column
                
                float p0 = (jb + col_idx < S && ib + row_idx < S) ? fast_exp2f_fn(s_reg[r][0] * alpha - smem_L[row_idx]) * 1.44269504f : 0.0f;
                float p1 = (jb + col_idx + 1 < S && ib + row_idx < S) ? fast_exp2f_fn(s_reg[r][1] * alpha - smem_L[row_idx]) * 1.44269504f : 0.0f;
                float p2 = (jb + col_idx + 2 < S && ib + row_idx < S) ? fast_exp2f_fn(s_reg[r][2] * alpha - smem_L[row_idx]) * 1.44269504f : 0.0f;
                float p3 = (jb + col_idx + 3 < S && ib + row_idx < S) ? fast_exp2f_fn(s_reg[r][3] * alpha - smem_L[row_idx]) * 1.44269504f : 0.0f;
                
                int quarter = r / 2;
                row_sum_corr[quarter] += d_reg[r][0] * p0 + d_reg[r][1] * p1 + d_reg[r][2] * p2 + d_reg[r][3] * p3;
            }
        }
        
        // Reduce row_sum_corr across the 2 threads sharing the same row
        int src_lane = (tid % 32) ^ 1;
        row_sum_corr[0] += __shfl_sync(0xffffffff, row_sum_corr[0], src_lane);
        row_sum_corr[1] += __shfl_sync(0xffffffff, row_sum_corr[1], src_lane);
        row_sum_corr[2] += __shfl_sync(0xffffffff, row_sum_corr[2], src_lane);
        row_sum_corr[3] += __shfl_sync(0xffffffff, row_sum_corr[3], src_lane);
        
        for(int r = 0; r < 8; ++r) {
            int row_idx = (tid % 64);
            int col_idx = (r / 2) * 4 + (tid / 64) * 2;
            
            float p0 = (jb + col_idx < S && ib + row_idx < S) ? fast_exp2f_fn(s_reg[r][0] * alpha - smem_L[row_idx]) * 1.44269504f : 0.0f;
            float p1 = (jb + col_idx + 1 < S && ib + row_idx < S) ? fast_exp2f_fn(s_reg[r][1] * alpha - smem_L[row_idx]) * 1.44269504f : 0.0f;
            float p2 = (jb + col_idx + 2 < S && ib + row_idx < S) ? fast_exp2f_fn(s_reg[r][2] * alpha - smem_L[row_idx]) * 1.44269504f : 0.0f;
            float p3 = (jb + col_idx + 3 < S && ib + row_idx < S) ? fast_exp2f_fn(s_reg[r][3] * alpha - smem_L[row_idx]) * 1.44269504f : 0.0f;
            
            int quarter = r / 2;
            float dp0 = d_reg[r][0] - p0 * row_sum_corr[quarter];
            float dp1 = d_reg[r][1] - p1 * row_sum_corr[quarter];
            float dp2 = d_reg[r][2] - p2 * row_sum_corr[quarter];
            float dp3 = d_reg[r][3] - p3 * row_sum_corr[quarter];
            
            int col_aligned = ((r / 2) * 4 + (tid / 64) * 2);
            store_smem_swizzled(smem_dP, row_idx, col_aligned, __float2bfloat16(dp0));
            store_smem_swizzled(smem_dP, row_idx, col_aligned + 1, __float2bfloat16(dp1));
            store_smem_swizzled(smem_dP, row_idx, col_aligned + 2, __float2bfloat16(dp2));
            store_smem_swizzled(smem_dP, row_idx, col_aligned + 3, __float2bfloat16(dp3));
            
            store_smem_swizzled(smem_P, row_idx, col_aligned, __float2bfloat16(p0));
            store_smem_swizzled(smem_P, row_idx, col_aligned + 1, __float2bfloat16(p1));
            store_smem_swizzled(smem_P, row_idx, col_aligned + 2, __float2bfloat16(p2));
            store_smem_swizzled(smem_P, row_idx, col_aligned + 3, __float2bfloat16(p3));
        }
        __syncthreads();
        
        fence_async_shared_fn();
        
        uint64_t desc_dP = make_smem_desc(smem_dP, 8192, 1024);
        uint64_t desc_dP_T = make_smem_desc(smem_dP, 1024, 8192); // Swapping lbo and sbo maps the contiguous rows into columns effectively transposing the orientation
        uint64_t desc_P = make_smem_desc(smem_P, 8192, 1024);
        uint64_t desc_P_T = make_smem_desc(smem_P, 1024, 8192);
        
        if (tid == 0) {
            asm volatile("tcgen05.cp.cta_group::1.sync.aligned.128x128b [%0], %1;" 
                         :: "r"(tmem_dP), "l"(desc_dP));
            asm volatile("tcgen05.cp.cta_group::1.sync.aligned.128x128b [%0], %1;" 
                         :: "r"(tmem_dP_T), "l"(desc_dP_T));
            asm volatile("tcgen05.cp.cta_group::1.sync.aligned.128x128b [%0], %1;" 
                         :: "r"(tmem_P), "l"(desc_P));
            asm volatile("tcgen05.cp.cta_group::1.sync.aligned.128x128b [%0], %1;" 
                         :: "r"(tmem_P + 8192), "l"(desc_P_T)); // Offset by 8192 for utilizing remaining empty space in TMEM chunk dynamically
        }
        
        fence_async_shared_fn();
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        __syncthreads(); 
        
        for (int k_off = 0; k_off < 64; k_off += 16) {
            uint64_t a_desc = make_smem_desc(smem_dP + k_off, 8192, 1024); 
            uint64_t b_desc0 = make_smem_desc(smem_K_0 + k_off * 4096, 1024, 8192); // Scaling offset to match stride
            uint64_t b_desc1 = make_smem_desc(smem_K_1 + k_off * 4096, 1024, 8192);
            umma_f16_cta1(tmem_1, a_desc, b_desc0, idesc_dQ, accum);
            umma_f16_cta1(tmem_1 + 8192, a_desc, b_desc1, idesc_dQ, accum);
        }
        
        __syncthreads();
        
        // Write dK to gmem through unswizzling into linear SMEM structures
        for (int k_off = 0; k_off < 64; k_off += 16) {
            uint64_t a_desc = make_smem_desc(smem_dP + k_off * 4096, 1024, 8192); 
            uint64_t b_desc0 = make_smem_desc(smem_Q_0 + k_off * 4096, 1024, 8192);
            uint64_t b_desc1 = make_smem_desc(smem_Q_1 + k_off * 4096, 1024, 8192);
            umma_f16_cta1(tmem_0, a_desc, b_desc0, idesc_dK, accum);
            umma_f16_cta1(tmem_0 + 8192, a_desc, b_desc1, idesc_dK, accum);
        }
        
        for(int r = tid; r < 2048; r += 128) {
            int row = (r / 32);
            int col = (r % 32);
            int chunk_idx = (row % 8) ^ (col / 8);
            int col_aligned = chunk_idx * 8 + (col % 8);
            
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_0 + (col_aligned << 16) + (row << 0)));
            tmem_load_fence_fn();
            
            *(uint32_t*)&smem_dK_0[(row * 64) + col_aligned] = pack_bf16_pair(__float2bfloat16(__uint_as_float(r0)), __float2bfloat16(__uint_as_float(r1)));
            *(uint32_t*)&smem_dK_0[(row * 64) + col_aligned + 1] = pack_bf16_pair(__float2bfloat16(__uint_as_float(r2)), __float2bfloat16(__uint_as_float(r3)));
            
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_0 + 8192 + (col_aligned << 16) + (row << 0)));
            tmem_load_fence_fn();
            
            *(uint32_t*)&smem_dK_1[(row * 64) + col_aligned] = pack_bf16_pair(__float2bfloat16(__uint_as_float(r0)), __float2bfloat16(__uint_as_float(r1)));
            *(uint32_t*)&smem_dK_1[(row * 64) + col_aligned + 1] = pack_bf16_pair(__float2bfloat16(__uint_as_float(r2)), __float2bfloat16(__uint_as_float(r3)));
        }
        __syncthreads();
        atomic_add_bf16_matrix(dK_f32 + batch_head * S * 128 + jb * 128, smem_dK_0, valid_j, tid);
        atomic_add_bf16_matrix(dK_f32 + batch_head * S * 128 + jb * 128 + 64, smem_dK_1, valid_j, tid);
        
        // Repeat similarly for extracting dV
        for (int k_off = 0; k_off < 64; k_off += 16) {
            uint64_t a_desc = make_smem_desc(smem_P + k_off * 4096, 1024, 8192); 
            uint64_t b_desc0 = make_smem_desc(smem_dO_0 + k_off * 4096, 1024, 8192);
            uint64_t b_desc1 = make_smem_desc(smem_dO_1 + k_off * 4096, 1024, 8192);
            umma_f16_cta1(tmem_1, a_desc, b_desc0, idesc_dV, accum);
            umma_f16_cta1(tmem_1 + 8192, a_desc, b_desc1, idesc_dV, accum);
        }
        
        for(int r = tid; r < 2048; r += 128) {
            int row = (r / 32);
            int col = (r % 32);
            int chunk_idx = (row % 8) ^ (col / 8);
            int col_aligned = chunk_idx * 8 + (col % 8);
            
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_1 + (col_aligned << 16) + (row << 0)));
            tmem_load_fence_fn();
            
            *(uint32_t*)&smem_dV_0[(row * 64) + col_aligned] = pack_bf16_pair(__float2bfloat16(__uint_as_float(r0)), __float2bfloat16(__uint_as_float(r1)));
            *(uint32_t*)&smem_dV_0[(row * 64) + col_aligned + 1] = pack_bf16_pair(__float2bfloat16(__uint_as_float(r2)), __float2bfloat16(__uint_as_float(r3)));
            
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_1 + 8192 + (col_aligned << 16) + (row << 0)));
            tmem_load_fence_fn();
            
            *(uint32_t*)&smem_dV_1[(row * 64) + col_aligned] = pack_bf16_pair(__float2bfloat16(__uint_as_float(r0)), __float2bfloat16(__uint_as_float(r1)));
            *(uint32_t*)&smem_dV_1[(row * 64) + col_aligned + 1] = pack_bf16_pair(__float2bfloat16(__uint_as_float(r2)), __float2bfloat16(__uint_as_float(r3)));
        }
        __syncthreads();
        atomic_add_bf16_matrix(dV_f32 + batch_head * S * 128 + jb * 128, smem_dV_0, valid_j, tid);
        atomic_add_bf16_matrix(dV_f32 + batch_head * S * 128 + jb * 128 + 64, smem_dV_1, valid_j, tid);
    }
    
    __syncthreads(); 
    
    // Epilogue for cleanly writing coalesced fully accumulated dQ
    for(int r = tid; r < 2048; r += 128) {
        int row = (r / 32);
        int col = (r % 32);
        int chunk_idx = (row % 8) ^ (col / 8);
        int col_aligned = chunk_idx * 8 + (col % 8);
        
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_1 + (col_aligned << 16) + (row << 0)));
        tmem_load_fence_fn();
        
        *(uint32_t*)&smem_Q_0[(row * 64) + col_aligned] = pack_bf16_pair(__float2bfloat16(__uint_as_float(r0)), __float2bfloat16(__uint_as_float(r1)));
        *(uint32_t*)&smem_Q_0[(row * 64) + col_aligned + 1] = pack_bf16_pair(__float2bfloat16(__uint_as_float(r2)), __float2bfloat16(__uint_as_float(r3)));
        
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_1 + 8192 + (col_aligned << 16) + (row << 0)));
        tmem_load_fence_fn();
        
        *(uint32_t*)&smem_Q_1[(row * 64) + col_aligned] = pack_bf16_pair(__float2bfloat16(__uint_as_float(r0)), __float2bfloat16(__uint_as_float(r1)));
        *(uint32_t*)&smem_Q_1[(row * 64) + col_aligned + 1] = pack_bf16_pair(__float2bfloat16(__uint_as_float(r2)), __float2bfloat16(__uint_as_float(r3)));
    }
    __syncthreads();
    
    store_smem_to_gmem_vec(dQ + batch_head * S * 128 + ib * 128, smem_Q_0, valid_i, 128, 0, tid);
    store_smem_to_gmem_vec(dQ + batch_head * S * 128 + ib * 128 + 64, smem_Q_1, valid_i, 128, 0, tid);
    
    if (tid == 0) {
        tmem_dealloc_fn(tmem_0, 128);
        tmem_dealloc_fn(tmem_1, 128);
        tmem_dealloc_fn(tmem_dP, 64);
        tmem_dealloc_fn(tmem_dP_T, 64);
        tmem_dealloc_fn(tmem_P, 64);
    }
    __syncthreads();
}

__global__ void convert_f32_to_bf16(const float* in, __nv_bfloat16* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(in[idx]);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L, 
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
         
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t b = Q.size(0); 
    int64_t h = Q.size(1);
    int64_t s = Q.size(2); 
    int64_t d = Q.size(3); 
    
    const __nv_bfloat16* q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* k_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* v_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* o_ptr = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* do_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* l_ptr = static_cast<const float*>(L.data_ptr());
    
    __nv_bfloat16* dq_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dk_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dv_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    int64_t threads = 128;
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    float* dK_f32;
    float* dV_f32;
    
    CUDA_CHECK(cudaMallocAsync(&dK_f32, b * h * s * d * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&dV_f32, b * h * s * d * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(dK_f32, 0, b * h * s * d * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(dV_f32, 0, b * h * s * d * sizeof(float), stream));
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O, tma_dO;
    
    CUresult res_q = create_tma_4d_descriptor_2B(&tma_Q, (void*)q_ptr, 128, s, h, b, 64, 64);
    CUresult res_k = create_tma_4d_descriptor_2B(&tma_K, (void*)k_ptr, 128, s, h, b, 64, 64);
    CUresult res_v = create_tma_4d_descriptor_2B(&tma_V, (void*)v_ptr, 128, s, h, b, 64, 64);
    CUresult res_o = create_tma_4d_descriptor_2B(&tma_O, (void*)o_ptr, 128, s, h, b, 64, 64);
    CUresult res_do = create_tma_4d_descriptor_2B(&tma_dO, (void*)do_ptr, 128, s, h, b, 64, 64);
    
    if (res_q != CUDA_SUCCESS || res_k != CUDA_SUCCESS || res_v != CUDA_SUCCESS || res_o != CUDA_SUCCESS || res_do != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to create TMA descriptor\n");
        exit(1);
    }
    
    dim3 grid((s + BM - 1) / BM, b * h); 
    
    int smem_size = 115200; 
    
    CUDA_CHECK(cudaFuncSetAttribute(run_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    run_kernel<<<grid, threads, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, tma_O, tma_dO, 
        l_ptr, dq_ptr, dK_f32, dV_f32, s, 1.0f / sqrtf((float)d)
    );
    
    CUDA_CHECK(cudaGetLastError());
    
    int64_t total_elements = b * h * s * d;
    int64_t conv_threads = 256;
    int64_t conv_blocks = (total_elements + conv_threads - 1) / conv_threads;
    
    convert_f32_to_bf16<<<conv_blocks, conv_threads, 0, stream>>>(dK_f32, dk_ptr, total_elements);
    convert_f32_to_bf16<<<conv_blocks, conv_threads, 0, stream>>>(dV_f32, dv_ptr, total_elements);
    
    CUDA_CHECK(cudaFreeAsync(dK_f32, stream));
    CUDA_CHECK(cudaFreeAsync(dV_f32, stream));
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_bwd::run);

} // namespace tvm_ffi_mha_bwd