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

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
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

__device__ __forceinline__ uint64_t make_smem_desc_128B_k_major(void* smem_ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((0 & 0x3FFFF) >> 4) << 16; 
    d |= (uint64_t)((1024 & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_128B_mn_major(void* smem_ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((8192 & 0x3FFFF) >> 4) << 16; 
    d |= (uint64_t)((1024 & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint32_t smem_addr_128B(const uint8_t* smem, int row, int col) {
    int chunk_idx = (row % 8) ^ (col / 8);
    return ((row * 64) + chunk_idx * 8 + (col % 8)) * 2;
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

__global__ __launch_bounds__(128, 1) void run_kernel_dQ(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* L, 
    __nv_bfloat16* dQ, 
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
    uint8_t* smem_D_Si = smem_pool + 65536;
    
    float* smem_L = (float*)(smem_pool + 73728);
    uint8_t* smem_bar = (uint8_t*)(smem_pool + 74000); 
    
    uint32_t* smem_tmem_Q_0 = (uint32_t*)(smem_pool + 74016);
    uint32_t* smem_tmem_Q_1 = (uint32_t*)(smem_pool + 74020);
    uint32_t* smem_tmem_dO_0 = (uint32_t*)(smem_pool + 74024);
    uint32_t* smem_tmem_dO_1 = (uint32_t*)(smem_pool + 74028);
    uint32_t* smem_tmem_K_0 = (uint32_t*)(smem_pool + 74032);
    uint32_t* smem_tmem_K_1 = (uint32_t*)(smem_pool + 74036);
    uint32_t* smem_tmem_V_0 = (uint32_t*)(smem_pool + 74040);
    uint32_t* smem_tmem_V_1 = (uint32_t*)(smem_pool + 74044);
    uint32_t* smem_tmem_S = (uint32_t*)(smem_pool + 74048);
    uint32_t* smem_tmem_dP = (uint32_t*)(smem_pool + 74052);
    uint32_t* smem_tmem_D_Si = (uint32_t*)(smem_pool + 74056);
    
    if (tid == 0) {
        init_smem_barrier_fn((uint64_t*)smem_bar, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();
    
    uint32_t phase = 0;
    
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn((uint64_t*)smem_bar, 4 * 8192); 
        
        tma_load_4d_fn(&tma_Q, (uint64_t*)smem_bar, smem_Q_0, 0, ib, head, batch);
        tma_load_4d_fn(&tma_Q, (uint64_t*)smem_bar, smem_Q_1, 64, ib, head, batch);
        tma_load_4d_fn(&tma_dO, (uint64_t*)smem_bar, smem_dO_0, 0, ib, head, batch);
        tma_load_4d_fn(&tma_dO, (uint64_t*)smem_bar, smem_dO_1, 64, ib, head, batch);
    }
    
    if (tid < 64) {
        smem_L[tid] = (ib + tid < S) ? L[batch_head * S + ib + tid] : 0.0f;
    }
    
    mbarrier_wait_fn((uint64_t*)smem_bar, phase);
    phase ^= 1;
    
    if (tid == 0) {
        tmem_alloc_fn(smem_tmem_Q_0, 64); 
        tmem_alloc_fn(smem_tmem_Q_1, 64); 
        tmem_alloc_fn(smem_tmem_dO_0, 64); 
        tmem_alloc_fn(smem_tmem_dO_1, 64); 
        tmem_alloc_fn(smem_tmem_K_0, 64); 
        tmem_alloc_fn(smem_tmem_K_1, 64); 
        tmem_alloc_fn(smem_tmem_V_0, 64); 
        tmem_alloc_fn(smem_tmem_V_1, 64); 
        tmem_alloc_fn(smem_tmem_S, 64); 
        tmem_alloc_fn(smem_tmem_dP, 64); 
        tmem_alloc_fn(smem_tmem_D_Si, 64); 
    }
    __syncthreads();
    
    uint32_t tmem_Q_0 = smem_tmem_Q_0[0];
    uint32_t tmem_Q_1 = smem_tmem_Q_1[0];
    uint32_t tmem_dO_0 = smem_tmem_dO_0[0];
    uint32_t tmem_dO_1 = smem_tmem_dO_1[0];
    uint32_t tmem_K_0 = smem_tmem_K_0[0];
    uint32_t tmem_K_1 = smem_tmem_K_1[0];
    uint32_t tmem_V_0 = smem_tmem_V_0[0];
    uint32_t tmem_V_1 = smem_tmem_V_1[0];
    uint32_t tmem_S = smem_tmem_S[0];
    uint32_t tmem_dP = smem_tmem_dP[0];
    uint32_t tmem_D_Si = smem_tmem_D_Si[0];
    
    uint32_t idesc_base = (1<<4) | (1<<7) | (1<<10) | ((64>>3)<<17) | ((64>>4)<<24);
    uint32_t idesc_K_K = idesc_base | (0<<15) | (0<<16);
    uint32_t idesc_K_MN = idesc_base | (0<<15) | (1<<16);
    
    for (int i = tid; i < 4096; i += 128) {
        int row = (i >> 6) & 63;
        int col = i & 63;
        
        asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%2], {%0,%1,%3,%4};" 
                     :: "r"(0), "r"(0), "r"(tmem_Q_0 + (row << 16) + (col << 0)), "r"(0), "r"(0));
        asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%2], {%0,%1,%3,%4};" 
                     :: "r"(0), "r"(0), "r"(tmem_Q_1 + (row << 16) + (col << 0)), "r"(0), "r"(0));
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int row_start = warp_id * 16;
    
    if (tid == 0) {
        asm volatile("tcgen05.cp.cta_group::1.sync.aligned.128x128b [%0], %1;" 
                     :: "r"(tmem_Q_0), "l"(make_smem_desc_128B_k_major(smem_Q_0)));
        asm volatile("tcgen05.cp.cta_group::1.sync.aligned.128x128b [%0], %1;" 
                     :: "r"(tmem_Q_1), "l"(make_smem_desc_128B_k_major(smem_Q_1)));
        asm volatile("tcgen05.cp.cta_group::1.sync.aligned.128x128b [%0], %1;" 
                     :: "r"(tmem_dO_0), "l"(make_smem_desc_128B_k_major(smem_dO_0)));
        asm volatile("tcgen05.cp.cta_group::1.sync.aligned.128x128b [%0], %1;" 
                     :: "r"(tmem_dO_1), "l"(make_smem_desc_128B_k_major(smem_dO_1)));
    }
    fence_proxy_async_fn();
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    
    for (int j_block = 0; j_block < S; j_block += 64) {
        int jb = j_block;
        
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn((uint64_t*)smem_bar, 4 * 8192); 
            
            tma_load_4d_fn(&tma_K, (uint64_t*)smem_bar, smem_K_0, 0, jb, head, batch);
            tma_load_4d_fn(&tma_K, (uint64_t*)smem_bar, smem_K_1, 64, jb, head, batch);
            
            tma_load_4d_fn(&tma_V, (uint64_t*)smem_bar, smem_V_0, 0, jb, head, batch);
            tma_load_4d_fn(&tma_V, (uint64_t*)smem_bar, smem_V_1, 64, jb, head, batch);
        }
        mbarrier_wait_fn((uint64_t*)smem_bar, phase);
        phase ^= 1;
        
        if (tid == 0) {
            asm volatile("tcgen05.cp.cta_group::1.sync.aligned.128x128b [%0], %1;" 
                         :: "r"(tmem_K_0), "l"(make_smem_desc_128B_k_major(smem_K_0)));
            asm volatile("tcgen05.cp.cta_group::1.sync.aligned.128x128b [%0], %1;" 
                         :: "r"(tmem_K_1), "l"(make_smem_desc_128B_k_major(smem_K_1)));
        }
        fence_proxy_async_fn();
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        for (int k_off = 0; k_off < 64; k_off += 16) {
            uint64_t a_desc = make_smem_desc_128B_k_major(smem_Q_0 + k_off * 2);
            uint64_t b_desc = make_smem_desc_128B_k_major(smem_K_0 + k_off * 2);
            umma_f16_cta1(tmem_S, a_desc, b_desc, idesc_K_K, k_off == 0 ? 0 : 1);
            
            a_desc = make_smem_desc_128B_k_major(smem_Q_1 + k_off * 2);
            b_desc = make_smem_desc_128B_k_major(smem_K_1 + k_off * 2);
            umma_f16_cta1(tmem_S, a_desc, b_desc, idesc_K_K, 1);
        }
        
        if (tid == 0) {
            asm volatile("tcgen05.cp.cta_group::1.sync.aligned.128x128b [%0], %1;" 
                         :: "r"(tmem_dO_0), "l"(make_smem_desc_128B_k_major(smem_dO_0)));
            asm volatile("tcgen05.cp.cta_group::1.sync.aligned.128x128b [%0], %1;" 
                         :: "r"(tmem_dO_1), "l"(make_smem_desc_128B_k_major(smem_dO_1)));
            asm volatile("tcgen05.cp.cta_group::1.sync.aligned.128x128b [%0], %1;" 
                         :: "r"(tmem_V_0), "l"(make_smem_desc_128B_k_major(smem_V_0)));
            asm volatile("tcgen05.cp.cta_group::1.sync.aligned.128x128b [%0], %1;" 
                         :: "r"(tmem_V_1), "l"(make_smem_desc_128B_k_major(smem_V_1)));
        }
        fence_proxy_async_fn();
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        for (int k_off = 0; k_off < 64; k_off += 16) {
            uint64_t a_desc = make_smem_desc_128B_k_major(smem_dO_0 + k_off * 2);
            uint64_t b_desc = make_smem_desc_128B_k_major(smem_V_0 + k_off * 2);
            umma_f16_cta1(tmem_dP, a_desc, b_desc, idesc_K_K, k_off == 0 ? 0 : 1);
            
            a_desc = make_smem_desc_128B_k_major(smem_dO_1 + k_off * 2);
            b_desc = make_smem_desc_128B_k_major(smem_V_1 + k_off * 2);
            umma_f16_cta1(tmem_dP, a_desc, b_desc, idesc_K_K, 1);
        }
        
        if (tid == 0) {
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared.b64 [%0];" 
                         :: "r"((uint32_t)__cvta_generic_to_shared(&((uint64_t*)smem_bar)[0])) : "memory");
        }
        mbarrier_wait_fn((uint64_t*)smem_bar, phase);
        phase ^= 1;
        
        float s_reg[4][8];
        float dp_reg[4][8];
        
        for(int load = 0; load < 4; ++load) {
            int col_s = load * 16;
            uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
            asm volatile("tcgen05.ld.sync.aligned.16x128b.x4.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                         : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(tmem_S + (row_start << 16) + (col_s << 0)));
            tmem_load_fence_fn();
            
            s_reg[load][0] = __uint_as_float(r0);
            s_reg[load][1] = __uint_as_float(r1);
            s_reg[load][2] = __uint_as_float(r2);
            s_reg[load][3] = __uint_as_float(r3);
            s_reg[load][4] = __uint_as_float(r4);
            s_reg[load][5] = __uint_as_float(r5);
            s_reg[load][6] = __uint_as_float(r6);
            s_reg[load][7] = __uint_as_float(r7);
            
            asm volatile("tcgen05.ld.sync.aligned.16x128b.x4.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                         : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(tmem_dP + (row_start << 16) + (col_s << 0)));
            tmem_load_fence_fn();
            
            dp_reg[load][0] = __uint_as_float(r0);
            dp_reg[load][1] = __uint_as_float(r1);
            dp_reg[load][2] = __uint_as_float(r2);
            dp_reg[load][3] = __uint_as_float(r3);
            dp_reg[load][4] = __uint_as_float(r4);
            dp_reg[load][5] = __uint_as_float(r5);
            dp_reg[load][6] = __uint_as_float(r6);
            dp_reg[load][7] = __uint_as_float(r7);
        }
        
        float row_sum_corr[2] = {0, 0};
        float p_reg[4][8];
        float dp_reg_corrected[4][8];
        
        for(int load = 0; load < 4; ++load) {
            for(int v = 0; v < 8; ++v) {
                int row = row_start + (lane_id % 16);
                int col = load * 16 + (lane_id >= 16 ? 8 : 0) + v;
                int quarter = (lane_id >= 16) ? 1 : 0;
                
                int global_row = ib + row;
                int global_col = jb + col;
                
                float p0 = 0.0f;
                if (global_col < S && global_row < S) {
                    p0 = fast_exp2f_fn((s_reg[load][v] * alpha - smem_L[row]) * 1.44269504f);
                }
                
                p_reg[load][v] = p0;
                row_sum_corr[quarter] += dp_reg[load][v] * p0;
            }
        }
        
        int src_lane = (lane_id % 16) ^ 1;
        row_sum_corr[0] += __shfl_sync(0xffffffff, row_sum_corr[0], src_lane);
        row_sum_corr[1] += __shfl_sync(0xffffffff, row_sum_corr[1], src_lane);
        
        for(int load = 0; load < 4; ++load) {
            for(int v = 0; v < 8; ++v) {
                int row = row_start + (lane_id % 16);
                int col = load * 16 + (lane_id >= 16 ? 8 : 0) + v;
                int quarter = (lane_id >= 16) ? 1 : 0;
                
                float p0 = p_reg[load][v];
                float dp0 = dp_reg[load][v] - p0 * row_sum_corr[quarter];
                
                dp_reg_corrected[load][v] = dp0;
            }
        }
        
        __syncthreads();
        
        for(int load = 0; load < 4; ++load) {
            for(int v = 0; v < 8; ++v) {
                int row = row_start + (lane_id % 16);
                int col = load * 16 + (lane_id >= 16 ? 8 : 0) + v;
                
                *(uint32_t*)&smem_D_Si[smem_addr_128B(smem_D_Si, row, col)] = pack_bf16_pair(__float2bfloat16(dp_reg_corrected[load][v]), __float2bfloat16(0.0f));
            }
        }
        __syncthreads();
        
        if (tid == 0) {
            asm volatile("tcgen05.cp.cta_group::1.sync.aligned.128x128b [%0], %1;" 
                         :: "r"(tmem_D_Si), "l"(make_smem_desc_128B_k_major(smem_D_Si)));
            asm volatile("tcgen05.cp.cta_group::1.sync.aligned.128x128b [%0], %1;" 
                         :: "r"(tmem_K_0), "l"(make_smem_desc_128B_k_major(smem_K_0)));
            asm volatile("tcgen05.cp.cta_group::1.sync.aligned.128x128b [%0], %1;" 
                         :: "r"(tmem_K_1), "l"(make_smem_desc_128B_k_major(smem_K_1)));
        }
        fence_proxy_async_fn();
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        for (int k_off = 0; k_off < 64; k_off += 16) {
            uint64_t a_desc = make_smem_desc_128B_k_major(smem_D_Si + k_off * 2); 
            uint64_t b_desc0 = make_smem_desc_128B_mn_major(smem_K_0 + k_off * 128);
            uint64_t b_desc1 = make_smem_desc_128B_mn_major(smem_K_1 + k_off * 128);
            
            umma_f16_cta1(tmem_Q_0, a_desc, b_desc0, idesc_K_MN, j_block == 0 ? 0 : 1);
            umma_f16_cta1(tmem_Q_1, a_desc, b_desc1, idesc_K_MN, j_block == 0 ? 0 : 1);
        }
        
        if (tid == 0) {
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared.b64 [%0];" 
                         :: "r"((uint32_t)__cvta_generic_to_shared(&((uint64_t*)smem_bar)[0])) : "memory");
        }
        mbarrier_wait_fn((uint64_t*)smem_bar, phase);
        phase ^= 1;
        
        __syncthreads();
    }
    
    for(int load = 0; load < 4; ++load) {
        int col_s = load * 16;
        uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
        
        asm volatile("tcgen05.ld.sync.aligned.16x128b.x4.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(tmem_Q_0 + (row_start << 16) + (col_s << 0)));
        tmem_load_fence_fn();
        
        *(uint32_t*)&smem_Q_0[(row_start + (lane_id % 16)) * 64 + (col_s + (lane_id >= 16 ? 8 : 0))] = pack_bf16_pair(__float2bfloat16(__uint_as_float(r0)), __float2bfloat16(__uint_as_float(r1)));
        *(uint32_t*)&smem_Q_0[(row_start + (lane_id % 16)) * 64 + (col_s + (lane_id >= 16 ? 8 : 0) + 2)] = pack_bf16_pair(__float2bfloat16(__uint_as_float(r2)), __float2bfloat16(__uint_as_float(r3)));
        *(uint32_t*)&smem_Q_0[(row_start + (lane_id % 16)) * 64 + (col_s + (lane_id >= 16 ? 8 : 0) + 4)] = pack_bf16_pair(__float2bfloat16(__uint_as_float(r4)), __float2bfloat16(__uint_as_float(r5)));
        *(uint32_t*)&smem_Q_0[(row_start + (lane_id % 16)) * 64 + (col_s + (lane_id >= 16 ? 8 : 0) + 6)] = pack_bf16_pair(__float2bfloat16(__uint_as_float(r6)), __float2bfloat16(__uint_as_float(r7)));
        
        asm volatile("tcgen05.ld.sync.aligned.16x128b.x4.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(tmem_Q_1 + (row_start << 16) + (col_s << 0)));
        tmem_load_fence_fn();
        
        *(uint32_t*)&smem_Q_1[(row_start + (lane_id % 16)) * 64 + (col_s + (lane_id >= 16 ? 8 : 0))] = pack_bf16_pair(__float2bfloat16(__uint_as_float(r0)), __float2bfloat16(__uint_as_float(r1)));
        *(uint32_t*)&smem_Q_1[(row_start + (lane_id % 16)) * 64 + (col_s + (lane_id >= 16 ? 8 : 0) + 2)] = pack_bf16_pair(__float2bfloat16(__uint_as_float(r2)), __float2bfloat16(__uint_as_float(r3)));
        *(uint32_t*)&smem_Q_1[(row_start + (lane_id % 16)) * 64 + (col_s + (lane_id >= 16 ? 8 : 0) + 4)] = pack_bf16_pair(__float2bfloat16(__uint_as_float(r4)), __float2bfloat16(__uint_as_float(r5)));
        *(uint32_t*)&smem_Q_1[(row_start + (lane_id % 16)) * 64 + (col_s + (lane_id >= 16 ? 8 : 0) + 6)] = pack_bf16_pair(__float2bfloat16(__uint_as_float(r6)), __float2bfloat16(__uint_as_float(r7)));
    }
    __syncthreads();
    
    for (int i = tid; i < 4096; i += 128) {
        int row = i / 64;
        int col = i % 64;
        
        if (ib + row < S) {
            *(uint32_t*)&dQ[batch_head * S * 128 + (ib + row) * 128 + col] = *(uint32_t*)&smem_Q_0[(row * 64) + ((row % 8) ^ (col / 8)) * 8 + (col % 8)];
            *(uint32_t*)&dQ[batch_head * S * 128 + (ib + row) * 128 + 64 + col] = *(uint32_t*)&smem_Q_1[(row * 64) + ((row % 8) ^ (col / 8)) * 8 + (col % 8)];
        }
    }
    
    if (tid == 0) {
        tmem_dealloc_fn(tmem_Q_0, 64);
        tmem_dealloc_fn(tmem_Q_1, 64);
        tmem_dealloc_fn(tmem_dO_0, 64);
        tmem_dealloc_fn(tmem_dO_1, 64);
        tmem_dealloc_fn(tmem_K_0, 64);
        tmem_dealloc_fn(tmem_K_1, 64);
        tmem_dealloc_fn(tmem_V_0, 64);
        tmem_dealloc_fn(tmem_V_1, 64);
        tmem_dealloc_fn(tmem_S, 64);
        tmem_dealloc_fn(tmem_dP, 64);
        tmem_dealloc_fn(tmem_D_Si, 64);
    }
    __syncthreads();
}

__global__ __launch_bounds__(128, 1) void run_kernel_dK_dV(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* L, 
    __nv_bfloat16* dK, 
    __nv_bfloat16* dV, 
    int32_t S, float alpha)
{
    int32_t head = blockIdx.y % H;
    int32_t batch = blockIdx.y / H;
    int32_t batch_head = batch * H + head;
    
    int32_t jb = blockIdx.x * BM;
    if (jb >= S) return;
    int32_t valid_j = min(S - jb, BM);
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
    uint8_t* smem_dS_T = smem_pool + 65536;
    uint8_t* smem_P_T = smem_pool + 73728;
    
    float* smem_L = (float*)(smem_pool + 81920);
    uint8_t* smem_bar = (uint8_t*)(smem_pool + 82176);
    
    uint32_t* smem_tmem_Q_0 = (uint32_t*)(smem_pool + 82184);
    uint32_t* smem_tmem_Q_1 = (uint32_t*)(smem_pool + 82188);
    uint32_t* smem_tmem_dO_0 = (uint32_t*)(smem_pool + 82192);
    uint32_t* smem_tmem_dO_1 = (uint32_t*)(smem_pool + 82196);
    uint32_t* smem_tmem_K_0 = (uint32_t*)(smem_pool + 82200);
    uint32_t* smem_tmem_K_1 = (uint32_t*)(smem_pool + 82204);
    uint32_t* smem_tmem_V_0 = (uint32_t*)(smem_pool + 82208);
    uint32_t* smem_tmem_V_1 = (uint32_t*)(smem_pool + 82212);
    uint32_t* smem_tmem_S = (uint32_t*)(smem_pool + 82216);
    uint32_t* smem_tmem_dP = (uint32_t*)(smem_pool + 82220);
    uint32_t* smem_tmem_dS_T = (uint32_t*)(smem_pool + 82224);
    uint32_t* smem_tmem_P_T = (uint32_t*)(smem_pool + 82228);
    
    if (tid == 0) {
        init_smem_barrier_fn((uint64_t*)smem_bar, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();
    
    uint32_t phase = 0;
    
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn((uint64_t*)smem_bar, 4 * 8192);
        
        tma_load_4d_fn(&tma_K, (uint64_t*)smem_bar, smem_K_0, 0, jb, head, batch);
        tma_load_4d_fn(&tma_K, (uint64_t*)smem_bar, smem_K_1, 64, jb, head, batch);
        
        tma_load_4d_fn(&tma_V, (uint64_t*)smem_bar, smem_V_0, 0, jb, head, batch);
        tma_load_4d_fn(&tma_V, (uint64_t*)smem_bar, smem_V_1, 64, jb, head, batch);
    }
    mbarrier_wait_fn((uint64_t*)smem_bar, phase);
    phase ^= 1;
    
    if (tid == 0) {
        tmem_alloc_fn(smem_tmem_Q_0, 64); 
        tmem_alloc_fn(smem_tmem_Q_1, 64); 
        tmem_alloc_fn(smem_tmem_dO_0, 64); 
        tmem_alloc_fn(smem_tmem_dO_1, 64); 
        tmem_alloc_fn(smem_tmem_K_0, 64); 
        tmem_alloc_fn(smem_tmem_K_1, 64); 
        tmem_alloc_fn(smem_tmem_V_0, 64); 
        tmem_alloc_fn(smem_tmem_V_1, 64); 
        tmem_alloc_fn(smem_tmem_S, 64); 
        tmem_alloc_fn(smem_tmem_dP, 64); 
        tmem_alloc_fn(smem_tmem_dS_T, 64); 
        tmem_alloc_fn(smem_tmem_P_T, 64); 
    }
    __syncthreads();
    
    uint32_t tmem_Q_0 = smem_tmem_Q_0[0];
    uint32_t tmem_Q_1 = smem_tmem_Q_1[0];
    uint32_t tmem_dO_0 = smem_tmem_dO_0[0];
    uint32_t tmem_dO_1 = smem_tmem_dO_1[0];
    uint32_t tmem_K_0 = smem_tmem_K_0[0];
    uint32_t tmem_K_1 = smem_tmem_K_1[0];
    uint32_t tmem_V_0 = smem_tmem_V_0[0];
    uint32_t tmem_V_1 = smem_tmem_V_1[0];
    uint32_t tmem_S = smem_tmem_S[0];
    uint32_t tmem_dP = smem_tmem_dP[0];
    uint32_t tmem_dS_T = smem_tmem_dS_T[0];
    uint32_t tmem_P_T = smem_tmem_P_T[0];
    
    uint32_t idesc_base = (1<<4) | (1<<7) | (1<<10) | ((64>>3)<<17) | ((64>>4)<<24);
    uint32_t idesc_K_K = idesc_base | (0<<15) | (0<<16);
    uint32_t idesc_MN_MN = idesc_base | (1<<15) | (1<<16);
    
    for (int i = tid; i < 4096; i += 128) {
        int row = (i >> 6) & 63;
        int col = i & 63;
        
        asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%2], {%0,%1,%3,%4};" 
                     :: "r"(0), "r"(0), "r"(tmem_K_0 + (row << 16) + (col << 0)), "r"(0), "r"(0));
        asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%2], {%0,%1,%3,%4};" 
                     :: "r"(0), "r"(0), "r"(tmem_K_1 + (row << 16) + (col << 0)), "r"(0), "r"(0));
                     
        asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%2], {%0,%1,%3,%4};" 
                     :: "r"(0), "r"(0), "r"(tmem_V_0 + (row << 16) + (col << 0)), "r"(0), "r"(0));
        asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%2], {%0,%1,%3,%4};" 
                     :: "r"(0), "r"(0), "r"(tmem_V_1 + (row << 16) + (col << 0)), "r"(0), "r"(0));
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int row_start = warp_id * 16;
    
    if (tid == 0) {
        asm volatile("tcgen05.cp.cta_group::1.sync.aligned.128x128b [%0], %1;" 
                     :: "r"(tmem_K_0), "l"(make_smem_desc_128B_k_major(smem_K_0)));
        asm volatile("tcgen05.cp.cta_group::1.sync.aligned.128x128b [%0], %1;" 
                     :: "r"(tmem_K_1), "l"(make_smem_desc_128B_k_major(smem_K_1)));
        asm volatile("tcgen05.cp.cta_group::1.sync.aligned.128x128b [%0], %1;" 
                     :: "r"(tmem_V_0), "l"(make_smem_desc_128B_k_major(smem_V_0)));
        asm volatile("tcgen05.cp.cta_group::1.sync.aligned.128x128b [%0], %1;" 
                     :: "r"(tmem_V_1), "l"(make_smem_desc_128B_k_major(smem_V_1)));
    }
    fence_proxy_async_fn();
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    
    for (int i_block = 0; i_block < S; i_block += 64) {
        int ib = i_block;
        
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn((uint64_t*)smem_bar, 4 * 8192);
            
            tma_load_4d_fn(&tma_Q, (uint64_t*)smem_bar, smem_Q_0, 0, ib, head, batch);
            tma_load_4d_fn(&tma_Q, (uint64_t*)smem_bar, smem_Q_1, 64, ib, head, batch);
            
            tma_load_4d_fn(&tma_dO, (uint64_t*)smem_bar, smem_dO_0, 0, ib, head, batch);
            tma_load_4d_fn(&tma_dO, (uint64_t*)smem_bar, smem_dO_1, 64, ib, head, batch);
        }
        
        if (tid < 64) {
            smem_L[tid] = (ib + tid < S) ? L[batch_head * S + ib + tid] : 0.0f;
        }
        
        mbarrier_wait_fn((uint64_t*)smem_bar, phase);
        phase ^= 1;
        
        if (tid == 0) {
            asm volatile("tcgen05.cp.cta_group::1.sync.aligned.128x128b [%0], %1;" 
                         :: "r"(tmem_Q_0), "l"(make_smem_desc_128B_k_major(smem_Q_0)));
            asm volatile("tcgen05.cp.cta_group::1.sync.aligned.128x128b [%0], %1;" 
                         :: "r"(tmem_Q_1), "l"(make_smem_desc_128B_k_major(smem_Q_1)));
        }
        fence_proxy_async_fn();
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        for (int k_off = 0; k_off < 64; k_off += 16) {
            uint64_t a_desc = make_smem_desc_128B_k_major(smem_Q_0 + k_off * 2);
            uint64_t b_desc = make_smem_desc_128B_k_major(smem_K_0 + k_off * 2);
            umma_f16_cta1(tmem_S, a_desc, b_desc, idesc_K_K, k_off == 0 ? 0 : 1);
            
            a_desc = make_smem_desc_128B_k_major(smem_Q_1 + k_off * 2);
            b_desc = make_smem_desc_128B_k_major(smem_K_1 + k_off * 2);
            umma_f16_cta1(tmem_S, a_desc, b_desc, idesc_K_K, 1);
        }
        
        if (tid == 0) {
            asm volatile("tcgen05.cp.cta_group::1.sync.aligned.128x128b [%0], %1;" 
                         :: "r"(tmem_dO_0), "l"(make_smem_desc_128B_k_major(smem_dO_0)));
            asm volatile("tcgen05.cp.cta_group::1.sync.aligned.128x128b [%0], %1;" 
                         :: "r"(tmem_dO_1), "l"(make_smem_desc_128B_k_major(smem_dO_1)));
        }
        fence_proxy_async_fn();
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        for (int k_off = 0; k_off < 64; k_off += 16) {
            uint64_t a_desc = make_smem_desc_128B_k_major(smem_dO_0 + k_off * 2);
            uint64_t b_desc = make_smem_desc_128B_k_major(smem_V_0 + k_off * 2);
            umma_f16_cta1(tmem_dP, a_desc, b_desc, idesc_K_K, k_off == 0 ? 0 : 1);
            
            a_desc = make_smem_desc_128B_k_major(smem_dO_1 + k_off * 2);
            b_desc = make_smem_desc_128B_k_major(smem_V_1 + k_off * 2);
            umma_f16_cta1(tmem_dP, a_desc, b_desc, idesc_K_K, 1);
        }
        
        if (tid == 0) {
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared.b64 [%0];" 
                         :: "r"((uint32_t)__cvta_generic_to_shared(&((uint64_t*)smem_bar)[0])) : "memory");
        }
        mbarrier_wait_fn((uint64_t*)smem_bar, phase);
        phase ^= 1;
        
        float s_reg[4][8];
        float dp_reg[4][8];
        
        for(int load = 0; load < 4; ++load) {
            int col_s = load * 16;
            uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
            asm volatile("tcgen05.ld.sync.aligned.16x128b.x4.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                         : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(tmem_S + (row_start << 16) + (col_s << 0)));
            tmem_load_fence_fn();
            
            s_reg[load][0] = __uint_as_float(r0);
            s_reg[load][1] = __uint_as_float(r1);
            s_reg[load][2] = __uint_as_float(r2);
            s_reg[load][3] = __uint_as_float(r3);
            s_reg[load][4] = __uint_as_float(r4);
            s_reg[load][5] = __uint_as_float(r5);
            s_reg[load][6] = __uint_as_float(r6);
            s_reg[load][7] = __uint_as_float(r7);
            
            asm volatile("tcgen05.ld.sync.aligned.16x128b.x4.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                         : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(tmem_dP + (row_start << 16) + (col_s << 0)));
            tmem_load_fence_fn();
            
            dp_reg[load][0] = __uint_as_float(r0);
            dp_reg[load][1] = __uint_as_float(r1);
            dp_reg[load][2] = __uint_as_float(r2);
            dp_reg[load][3] = __uint_as_float(r3);
            dp_reg[load][4] = __uint_as_float(r4);
            dp_reg[load][5] = __uint_as_float(r5);
            dp_reg[load][6] = __uint_as_float(r6);
            dp_reg[load][7] = __uint_as_float(r7);
        }
        
        float row_sum_corr[2] = {0, 0};
        float p_reg[4][8];
        float dp_reg_corrected[4][8];
        
        for(int load = 0; load < 4; ++load) {
            for(int v = 0; v < 8; ++v) {
                int row = row_start + (lane_id % 16);
                int col = load * 16 + (lane_id >= 16 ? 8 : 0) + v;
                int quarter = (lane_id >= 16) ? 1 : 0;
                
                int global_row = ib + row;
                int global_col = jb + col;
                
                float p0 = 0.0f;
                if (global_col < S && global_row < S) {
                    p0 = fast_exp2f_fn((s_reg[load][v] * alpha - smem_L[row]) * 1.44269504f);
                }
                
                p_reg[load][v] = p0;
                row_sum_corr[quarter] += dp_reg[load][v] * p0;
            }
        }
        
        int src_lane = (lane_id % 16) ^ 1;
        row_sum_corr[0] += __shfl_sync(0xffffffff, row_sum_corr[0], src_lane);
        row_sum_corr[1] += __shfl_sync(0xffffffff, row_sum_corr[1], src_lane);
        
        for(int load = 0; load < 4; ++load) {
            for(int v = 0; v < 8; ++v) {
                int row = row_start + (lane_id % 16);
                int col = load * 16 + (lane_id >= 16 ? 8 : 0) + v;
                int quarter = (lane_id >= 16) ? 1 : 0;
                
                float p0 = p_reg[load][v];
                float dp0 = dp_reg[load][v] - p0 * row_sum_corr[quarter];
                
                dp_reg_corrected[load][v] = dp0;
            }
        }
        
        __syncthreads();
        
        for(int load = 0; load < 4; ++load) {
            for(int v = 0; v < 8; ++v) {
                int row = row_start + (lane_id % 16);
                int col = load * 16 + (lane_id >= 16 ? 8 : 0) + v;
                
                *(uint32_t*)&smem_dS_T[smem_addr_128B(smem_dS_T, col, row)] = pack_bf16_pair(__float2bfloat16(dp_reg_corrected[load][v]), __float2bfloat16(0.0f));
                *(uint32_t*)&smem_P_T[smem_addr_128B(smem_P_T, col, row)] = pack_bf16_pair(__float2bfloat16(p_reg[load][v]), __float2bfloat16(0.0f));
            }
        }
        __syncthreads();
        
        if (tid == 0) {
            asm volatile("tcgen05.cp.cta_group::1.sync.aligned.128x128b [%0], %1;" 
                         :: "r"(tmem_dS_T), "l"(make_smem_desc_128B_mn_major(smem_dS_T)));
            asm volatile("tcgen05.cp.cta_group::1.sync.aligned.128x128b [%0], %1;" 
                         :: "r"(tmem_P_T), "l"(make_smem_desc_128B_mn_major(smem_P_T)));
            asm volatile("tcgen05.cp.cta_group::1.sync.aligned.128x128b [%0], %1;" 
                         :: "r"(tmem_Q_0), "l"(make_smem_desc_128B_mn_major(smem_Q_0)));
            asm volatile("tcgen05.cp.cta_group::1.sync.aligned.128x128b [%0], %1;" 
                         :: "r"(tmem_Q_1), "l"(make_smem_desc_128B_mn_major(smem_Q_1)));
            
            asm volatile("tcgen05.cp.cta_group::1.sync.aligned.128x128b [%0], %1;" 
                         :: "r"(tmem_dO_0), "l"(make_smem_desc_128B_mn_major(smem_dO_0)));
            asm volatile("tcgen05.cp.cta_group::1.sync.aligned.128x128b [%0], %1;" 
                         :: "r"(tmem_dO_1), "l"(make_smem_desc_128B_mn_major(smem_dO_1)));
        }
        fence_proxy_async_fn();
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        for (int k_off = 0; k_off < 64; k_off += 16) {
            uint64_t a_desc = make_smem_desc_128B_mn_major(smem_dS_T + k_off * 128);
            uint64_t b_desc0 = make_smem_desc_128B_mn_major(smem_Q_0 + k_off * 128);
            uint64_t b_desc1 = make_smem_desc_128B_mn_major(smem_Q_1 + k_off * 128);
            
            umma_f16_cta1(tmem_K_0, a_desc, b_desc0, idesc_MN_MN, i_block == 0 ? 0 : 1);
            umma_f16_cta1(tmem_K_1, a_desc, b_desc1, idesc_MN_MN, i_block == 0 ? 0 : 1);
        }
        
        for (int k_off = 0; k_off < 64; k_off += 16) {
            uint64_t a_desc = make_smem_desc_128B_mn_major(smem_P_T + k_off * 128);
            uint64_t b_desc0 = make_smem_desc_128B_mn_major(smem_dO_0 + k_off * 128);
            uint64_t b_desc1 = make_smem_desc_128B_mn_major(smem_dO_1 + k_off * 128);
            
            umma_f16_cta1(tmem_V_0, a_desc, b_desc0, idesc_MN_MN, i_block == 0 ? 0 : 1);
            umma_f16_cta1(tmem_V_1, a_desc, b_desc1, idesc_MN_MN, i_block == 0 ? 0 : 1);
        }
        
        if (tid == 0) {
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared.b64 [%0];" 
                         :: "r"((uint32_t)__cvta_generic_to_shared(&((uint64_t*)smem_bar)[0])) : "memory");
        }
        mbarrier_wait_fn((uint64_t*)smem_bar, phase);
        phase ^= 1;
        
        __syncthreads();
    }
    
    for(int load = 0; load < 4; ++load) {
        int col_s = load * 16;
        uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
        
        asm volatile("tcgen05.ld.sync.aligned.16x128b.x4.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(tmem_K_0 + (row_start << 16) + (col_s << 0)));
        tmem_load_fence_fn();
        
        *(uint32_t*)&smem_K_0[(row_start + (lane_id % 16)) * 64 + (col_s + (lane_id >= 16 ? 8 : 0))] = pack_bf16_pair(__float2bfloat16(__uint_as_float(r0)), __float2bfloat16(__uint_as_float(r1)));
        *(uint32_t*)&smem_K_0[(row_start + (lane_id % 16)) * 64 + (col_s + (lane_id >= 16 ? 8 : 0) + 2)] = pack_bf16_pair(__float2bfloat16(__uint_as_float(r2)), __float2bfloat16(__uint_as_float(r3)));
        *(uint32_t*)&smem_K_0[(row_start + (lane_id % 16)) * 64 + (col_s + (lane_id >= 16 ? 8 : 0) + 4)] = pack_bf16_pair(__float2bfloat16(__uint_as_float(r4)), __float2bfloat16(__uint_as_float(r5)));
        *(uint32_t*)&smem_K_0[(row_start + (lane_id % 16)) * 64 + (col_s + (lane_id >= 16 ? 8 : 0) + 6)] = pack_bf16_pair(__float2bfloat16(__uint_as_float(r6)), __float2bfloat16(__uint_as_float(r7)));
        
        asm volatile("tcgen05.ld.sync.aligned.16x128b.x4.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(tmem_K_1 + (row_start << 16) + (col_s << 0)));
        tmem_load_fence_fn();
        
        *(uint32_t*)&smem_K_1[(row_start + (lane_id % 16)) * 64 + (col_s + (lane_id >= 16 ? 8 : 0))] = pack_bf16_pair(__float2bfloat16(__uint_as_float(r0)), __float2bfloat16(__uint_as_float(r1)));
        *(uint32_t*)&smem_K_1[(row_start + (lane_id % 16)) * 64 + (col_s + (lane_id >= 16 ? 8 : 0) + 2)] = pack_bf16_pair(__float2bfloat16(__uint_as_float(r2)), __float2bfloat16(__uint_as_float(r3)));
        *(uint32_t*)&smem_K_1[(row_start + (lane_id % 16)) * 64 + (col_s + (lane_id >= 16 ? 8 : 0) + 4)] = pack_bf16_pair(__float2bfloat16(__uint_as_float(r4)), __float2bfloat16(__uint_as_float(r5)));
        *(uint32_t*)&smem_K_1[(row_start + (lane_id % 16)) * 64 + (col_s + (lane_id >= 16 ? 8 : 0) + 6)] = pack_bf16_pair(__float2bfloat16(__uint_as_float(r6)), __float2bfloat16(__uint_as_float(r7)));
        
        asm volatile("tcgen05.ld.sync.aligned.16x128b.x4.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(tmem_V_0 + (row_start << 16) + (col_s << 0)));
        tmem_load_fence_fn();
        
        *(uint32_t*)&smem_V_0[(row_start + (lane_id % 16)) * 64 + (col_s + (lane_id >= 16 ? 8 : 0))] = pack_bf16_pair(__float2bfloat16(__uint_as_float(r0)), __float2bfloat16(__uint_as_float(r1)));
        *(uint32_t*)&smem_V_0[(row_start + (lane_id % 16)) * 64 + (col_s + (lane_id >= 16 ? 8 : 0) + 2)] = pack_bf16_pair(__float2bfloat16(__uint_as_float(r2)), __float2bfloat16(__uint_as_float(r3)));
        *(uint32_t*)&smem_V_0[(row_start + (lane_id % 16)) * 64 + (col_s + (lane_id >= 16 ? 8 : 0) + 4)] = pack_bf16_pair(__float2bfloat16(__uint_as_float(r4)), __float2bfloat16(__uint_as_float(r5)));
        *(uint32_t*)&smem_V_0[(row_start + (lane_id % 16)) * 64 + (col_s + (lane_id >= 16 ? 8 : 0) + 6)] = pack_bf16_pair(__float2bfloat16(__uint_as_float(r6)), __float2bfloat16(__uint_as_float(r7)));
        
        asm volatile("tcgen05.ld.sync.aligned.16x128b.x4.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(tmem_V_1 + (row_start << 16) + (col_s << 0)));
        tmem_load_fence_fn();
        
        *(uint32_t*)&smem_V_1[(row_start + (lane_id % 16)) * 64 + (col_s + (lane_id >= 16 ? 8 : 0))] = pack_bf16_pair(__float2bfloat16(__uint_as_float(r0)), __float2bfloat16(__uint_as_float(r1)));
        *(uint32_t*)&smem_V_1[(row_start + (lane_id % 16)) * 64 + (col_s + (lane_id >= 16 ? 8 : 0) + 2)] = pack_bf16_pair(__float2bfloat16(__uint_as_float(r2)), __float2bfloat16(__uint_as_float(r3)));
        *(uint32_t*)&smem_V_1[(row_start + (lane_id % 16)) * 64 + (col_s + (lane_id >= 16 ? 8 : 0) + 4)] = pack_bf16_pair(__float2bfloat16(__uint_as_float(r4)), __float2bfloat16(__uint_as_float(r5)));
        *(uint32_t*)&smem_V_1[(row_start + (lane_id % 16)) * 64 + (col_s + (lane_id >= 16 ? 8 : 0) + 6)] = pack_bf16_pair(__float2bfloat16(__uint_as_float(r6)), __float2bfloat16(__uint_as_float(r7)));
    }
    __syncthreads();
    
    for (int i = tid; i < 4096; i += 128) {
        int row = i / 64;
        int col = i % 64;
        
        if (jb + row < S) {
            *(uint32_t*)&dK[batch_head * S * 128 + (jb + row) * 128 + col] = *(uint32_t*)&smem_K_0[(row * 64) + ((row % 8) ^ (col / 8)) * 8 + (col % 8)];
            *(uint32_t*)&dK[batch_head * S * 128 + (jb + row) * 128 + 64 + col] = *(uint32_t*)&smem_K_1[(row * 64) + ((row % 8) ^ (col / 8)) * 8 + (col % 8)];
            
            *(uint32_t*)&dV[batch_head * S * 128 + (jb + row) * 128 + col] = *(uint32_t*)&smem_V_0[(row * 64) + ((row % 8) ^ (col / 8)) * 8 + (col % 8)];
            *(uint32_t*)&dV[batch_head * S * 128 + (jb + row) * 128 + 64 + col] = *(uint32_t*)&smem_V_1[(row * 64) + ((row % 8) ^ (col / 8)) * 8 + (col % 8)];
        }
    }
    
    if (tid == 0) {
        tmem_dealloc_fn(tmem_Q_0, 64);
        tmem_dealloc_fn(tmem_Q_1, 64);
        tmem_dealloc_fn(tmem_dO_0, 64);
        tmem_dealloc_fn(tmem_dO_1, 64);
        tmem_dealloc_fn(tmem_K_0, 64);
        tmem_dealloc_fn(tmem_K_1, 64);
        tmem_dealloc_fn(tmem_V_0, 64);
        tmem_dealloc_fn(tmem_V_1, 64);
        tmem_dealloc_fn(tmem_S, 64);
        tmem_dealloc_fn(tmem_dP, 64);
        tmem_dealloc_fn(tmem_dS_T, 64);
        tmem_dealloc_fn(tmem_P_T, 64);
    }
    __syncthreads();
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
    const __nv_bfloat16* do_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* l_ptr = static_cast<const float*>(L.data_ptr());
    
    __nv_bfloat16* dq_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dk_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dv_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    int64_t threads = 128;
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_dO;
    
    CUresult res_q = create_tma_4d_descriptor_2B(&tma_Q, (void*)q_ptr, 128, s, h, b, 64, 64);
    CUresult res_k = create_tma_4d_descriptor_2B(&tma_K, (void*)k_ptr, 128, s, h, b, 64, 64);
    CUresult res_v = create_tma_4d_descriptor_2B(&tma_V, (void*)v_ptr, 128, s, h, b, 64, 64);
    CUresult res_do = create_tma_4d_descriptor_2B(&tma_dO, (void*)do_ptr, 128, s, h, b, 64, 64);
    
    if (res_q != CUDA_SUCCESS || res_k != CUDA_SUCCESS || res_v != CUDA_SUCCESS || res_do != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to create TMA descriptor\n");
        exit(1);
    }
    
    dim3 grid((s + BM - 1) / BM, b * h); 
    
    int smem_size_dQ = 75264;
    int smem_size_dK_dV = 83456;
    
    CUDA_CHECK(cudaFuncSetAttribute(run_kernel_dQ, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_dQ));
    CUDA_CHECK(cudaFuncSetAttribute(run_kernel_dK_dV, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_dK_dV));
    
    run_kernel_dQ<<<grid, threads, smem_size_dQ, stream>>>(
        tma_Q, tma_K, tma_V, tma_dO, 
        l_ptr, dq_ptr, s, 1.0f / sqrtf((float)d)
    );
    
    CUDA_CHECK(cudaGetLastError());
    
    run_kernel_dK_dV<<<grid, threads, smem_size_dK_dV, stream>>>(
        tma_Q, tma_K, tma_V, tma_dO, 
        l_ptr, dk_ptr, dv_ptr, s, 1.0f / sqrtf((float)d)
    );
    
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_bwd::run);

} // namespace tvm_ffi_mha_bwd