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
        const char* err_str = nullptr;                             \
        cuGetErrorString(_e, &err_str);                            \
        fprintf(stderr, "CUDA Driver error %s at %s:%d\n",         \
                err_str ? err_str : "Unknown", __FILE__, __LINE__);\
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

__device__ __forceinline__ void tma_load_4d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}

__device__ __forceinline__ void tmem_alloc_1cta_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_1cta_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo, uint32_t swizzle) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)swizzle << 61; 
    return d;
}

__device__ __forceinline__ uint64_t make_k_major_desc(void* ptr) {
    return make_smem_desc_sm100_fn(ptr, 2048, 128, 0); 
}

__device__ __forceinline__ uint64_t make_mn_major_desc(void* ptr, uint32_t K_dim) {
    return make_smem_desc_sm100_fn(ptr, 128, 2048, 0); 
}

__device__ __forceinline__ uint64_t update_desc_k(uint64_t desc, uint32_t new_addr) {
    desc &= ~0x3FFFuLL; 
    desc |= ((new_addr & 0x3FFFF) >> 4);
    desc &= ~(7uLL << 49); 
    desc |= (uint64_t)((new_addr >> 7) & 0x7) << 49;
    return desc;
}

__device__ __forceinline__ uint32_t make_idesc(uint32_t M, uint32_t N, int a_maj, int b_maj) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (a_maj << 15);
    d |= (b_maj << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ void umma_f16_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_1sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(a));
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

CUresult create_tma_4d_descriptor(CUtensorMap* d, void* globalAddress, 
    uint64_t dim0, uint64_t dim1, uint64_t dim2, uint64_t dim3,
    uint32_t box0, uint32_t box1) {
    cuuint64_t globalDim[4] = {dim0, dim1, dim2, dim3};
    cuuint64_t globalStrides[3] = {dim0*2, dim0*dim1*2, dim0*dim1*dim2*2};
    cuuint32_t boxDim[4] = {box0, box1, 1, 1};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
        globalAddress, globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

__global__ void precompute_D_kernel(const __nv_bfloat16* O, const __nv_bfloat16* dO, float* D, int B, int H, int S, int d) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int m_block = blockIdx.x;
    int tid = threadIdx.x;
    
    int m_idx = m_block * 128 + tid;
    if (m_idx >= S) return;
    
    uint64_t offset = ((uint64_t)b * H * S + h * S + m_idx) * 128;
    const uint4* o_ptr = (const uint4*)(O + offset);
    const uint4* do_ptr = (const uint4*)(dO + offset);
    
    float sum = 0.0f;
    for (int k = 0; k < 128; k += 8) { 
        uint4 o_val = o_ptr[k/8];
        uint4 do_val = do_ptr[k/8];
        __nv_bfloat16* o_bf = (__nv_bfloat16*)&o_val;
        __nv_bfloat16* do_bf = (__nv_bfloat16*)&do_val;
        #pragma unroll
        for (int i=0; i<8; ++i) {
            sum += __bfloat162float(o_bf[i]) * __bfloat162float(do_bf[i]);
        }
    }
    uint64_t d_offset = (uint64_t)b * H * S + h * S + m_idx;
    D[d_offset] = sum;
}

__device__ __forceinline__ void atomic_add_dq(
    __nv_bfloat16* D_global, __nv_bfloat16* smem_out,
    uint32_t S, uint32_t d_dim, uint32_t m_block,
    uint32_t BM, uint32_t BN, uint32_t TMEM_COL, int b, int h) {
    
    for (uint32_t col = 0; col < BN; col += 4) {
        uint32_t r0, r1, r2, r3;
        uint32_t tcol = TMEM_COL + col;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tcol));
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
    uint64_t batch_offset = (uint64_t)b * 48 * S * d_dim + (uint64_t)h * S * d_dim;
    
    for (uint32_t step = 0; step < BM/4; ++step) {
        uint32_t row = step * 4 + warp_id;
        uint32_t global_row = m_block * BM + row;
        uint32_t col_start = lane_id * 4;
        
        if (global_row < S && col_start < d_dim) {
            uint2 data = *reinterpret_cast<uint2*>(&smem_out[row * BN + col_start]);
            __nv_bfloat162* val = (__nv_bfloat162*)&data;
            __nv_bfloat162* ptr = (__nv_bfloat162*)(D_global + batch_offset + global_row * d_dim + col_start);
            atomicAdd(&ptr[0], val[0]);
            atomicAdd(&ptr[1], val[1]);
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void store_dv_dk(
    __nv_bfloat16* D_global, __nv_bfloat16* smem_out,
    uint32_t S, uint32_t d_dim, uint32_t n_block,
    uint32_t BM, uint32_t BN, uint32_t TMEM_COL, int b, int h) {
    
    for (uint32_t col = 0; col < BN; col += 4) {
        uint32_t r0, r1, r2, r3;
        uint32_t tcol = TMEM_COL + col;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tcol));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t base = threadIdx.x * BN + col;
        smem_out[base + 0] = __float2bfloat16(__uint_as_float(r0));
        smem_out[base + 1] = __float2bfloat16(__uint_as_float(r1));
        smem_out[base + 2] = __float2bfloat16(__float2bfloat16(__uint_as_float(r2)));
        smem_out[base + 3] = __float2bfloat16(__float2bfloat16(__uint_as_float(r3)));
    }
    __syncthreads();
    
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint64_t batch_offset = (uint64_t)b * 48 * S * d_dim + (uint64_t)h * S * d_dim;
    
    for (uint32_t step = 0; step < BM/4; ++step) {
        uint32_t row = step * 4 + warp_id;
        uint32_t global_row = n_block * BM + row;
        uint32_t col_start = lane_id * 4;
        
        if (global_row < S && col_start < d_dim) {
            uint2 data = *reinterpret_cast<uint2*>(&smem_out[row * BN + col_start]);
            *reinterpret_cast<uint2*>(D_global + batch_offset + global_row * d_dim + col_start) = data;
        }
    }
    __syncthreads();
}

__global__ void bwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_dO,
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    const float* L, const float* D,
    int B_dim, int H, int S, int d_dim) 
{
    int b = blockIdx.z;
    int h = blockIdx.y;
    int n_block = blockIdx.x; 
    
    extern __shared__ __align__(128) uint8_t smem_pool[];
    uint64_t* mbar_K = (uint64_t*)smem_pool;
    uint64_t* mbar_V = mbar_K + 1;
    uint64_t* mbar_Q = mbar_V + 1;
    uint64_t* mbar_dO = mbar_Q + 1;
    uint64_t* mbar_mma = mbar_dO + 1;
    uint32_t* tmem_base_ptr = (uint32_t*)(mbar_mma + 1); 
    
    __nv_bfloat16* smem_K = (__nv_bfloat16*)(smem_pool + 128); 
    __nv_bfloat16* smem_V = smem_K + 128*128;
    __nv_bfloat16* smem_Q = smem_V + 128*128; 
    __nv_bfloat16* smem_dO = smem_Q + 128*128;
    __nv_bfloat16* smem_dS = smem_dO + 128*128; 
    __nv_bfloat16* smem_P = smem_dS + 128*128;
    float* smem_L = (float*)(smem_P + 128*128); 
    float* smem_D = smem_L + 128;
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_K, 1);
        init_smem_barrier_fn(mbar_V, 1);
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(mbar_dO, 1);
        init_smem_barrier_fn(mbar_mma, 1);
    }
    
    // tmem_alloc must be executed collectively by the whole warp.
    // threadIdx.x < 32 evaluates identical for all threads in warp 0.
    if (threadIdx.x < 32) {
        tmem_alloc_1cta_fn(tmem_base_ptr, 512);
    }
    
    fence_smem_barrier_init_fn();
    __syncthreads();
    
    uint32_t tmem_base = *tmem_base_ptr;
    uint32_t tmem_S = tmem_base + 0;
    uint32_t tmem_dV = tmem_base + 128;
    uint32_t tmem_dQ = tmem_base + 256;
    uint32_t tmem_dP = tmem_base + 256;
    uint32_t tmem_dK = tmem_base + 384;
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_K, 32768);
        tma_load_4d_fn(&tma_K, mbar_K, smem_K, 0, n_block*128, h, b);
        mbarrier_arrive_and_expect_tx_fn(mbar_V, 32768);
        tma_load_4d_fn(&tma_V, mbar_V, smem_V, 0, n_block*128, h, b);
    }
    mbarrier_wait_fn(mbar_K, 0);
    mbarrier_wait_fn(mbar_V, 0);
    
    int phase = 0;
    int mma_phase = 0;
    int tid = threadIdx.x;
    
    for (int m_block = n_block; m_block < (S + 127) / 128; ++m_block, phase ^= 1) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_Q, 32768);
            tma_load_4d_fn(&tma_Q, mbar_Q, smem_Q, 0, m_block*128, h, b);
            mbarrier_arrive_and_expect_tx_fn(mbar_dO, 32768);
            tma_load_4d_fn(&tma_dO, mbar_dO, smem_dO, 0, m_block*128, h, b);
        }
        
        int m_idx = m_block * 128 + threadIdx.x;
        uint64_t ld_offset = (uint64_t)b * H * S + h * S + m_idx;
        if (m_idx < S) {
            smem_L[threadIdx.x] = L[ld_offset];
            smem_D[threadIdx.x] = D[ld_offset];
        }
        
        mbarrier_wait_fn(mbar_Q, phase);
        mbarrier_wait_fn(mbar_dO, phase);
        
        // Ensure smem_L and smem_D are visible to all threads across the CTA before usage.
        __syncthreads();
        
        if (threadIdx.x == 0) {
            uint32_t idesc_S = make_idesc(128, 128, 0, 0);
            for(int k=0; k<8; ++k) {
                uint64_t da = update_desc_k(make_k_major_desc(smem_K), (uint32_t)__cvta_generic_to_shared(smem_K) + k*32);
                uint64_t db = update_desc_k(make_k_major_desc(smem_Q), (uint32_t)__cvta_generic_to_shared(smem_Q) + k*32);
                umma_f16_fn(tmem_S, da, db, idesc_S, k>0);
            }
            umma_commit_1sm_fn(mbar_mma);
        }
        mbarrier_wait_fn(mbar_mma, mma_phase);
        mma_phase ^= 1;
        
        for(int c = 0; c < 32; ++c) {
            uint32_t r0, r1, r2, r3;
            uint32_t col_base = tmem_S + c * 4;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(col_base));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float f[4] = {__uint_as_float(r0), __uint_as_float(r1), __uint_as_float(r2), __uint_as_float(r3)};
            uint32_t n_idx = n_block * 128 + tid;
            uint32_t packed[2];
            for(int i=0; i<2; ++i) {
                __nv_bfloat16 bf[2];
                for(int j=0; j<2; ++j) {
                    int c_local = i*2 + j;
                    int global_m = m_block * 128 + c * 4 + c_local;
                    float val = f[c_local];
                    if (n_idx > global_m || global_m >= S) {
                        val = 0.0f;
                    } else {
                        float l_val = smem_L[c * 4 + c_local];
                        val = fast_exp2f_fn((val - l_val) * 1.4426950408889634f);
                    }
                    bf[j] = __float2bfloat16(val);
                }
                packed[i] = *(uint32_t*)&bf[0];
            }
            uint2 val = make_uint2(packed[0], packed[1]);
            *(uint2*)&smem_P[tid * 128 + c*4] = val; 
        }
        
        if (threadIdx.x == 0) {
            uint32_t idesc_dP = make_idesc(128, 128, 0, 0);
            for(int k=0; k<8; ++k) {
                uint64_t da = update_desc_k(make_k_major_desc(smem_V), (uint32_t)__cvta_generic_to_shared(smem_V) + k*32);
                uint64_t db = update_desc_k(make_k_major_desc(smem_dO), (uint32_t)__cvta_generic_to_shared(smem_dO) + k*32);
                umma_f16_fn(tmem_dP, da, db, idesc_dP, k>0);
            }
            umma_commit_1sm_fn(mbar_mma);
        }
        mbarrier_wait_fn(mbar_mma, mma_phase);
        mma_phase ^= 1;
        
        for(int c = 0; c < 32; ++c) {
            uint2 p_val2 = *(uint2*)&smem_P[tid * 128 + c*4];
            __nv_bfloat16 p_bf[4];
            p_bf[0] = ((__nv_bfloat16*)&p_val2.x)[0];
            p_bf[1] = ((__nv_bfloat16*)&p_val2.x)[1];
            p_bf[2] = ((__nv_bfloat16*)&p_val2.y)[0];
            p_bf[3] = ((__nv_bfloat16*)&p_val2.y)[1];
                
            uint32_t dp0, dp1, dp2, dp3; 
            uint32_t col_dp = tmem_dP + c * 4;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(dp0),"=r"(dp1),"=r"(dp2),"=r"(dp3) : "r"(col_dp));
                
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float dp[4] = {__uint_as_float(dp0), __uint_as_float(dp1), __uint_as_float(dp2), __uint_as_float(dp3)};
            
            uint32_t packed[2];
            for(int i=0; i<2; ++i) {
                __nv_bfloat16 ds[2];
                for(int j=0; j<2; ++j) {
                    int c_local = i*2 + j;
                    int local_m = c * 4 + c_local;
                    float p_val = __bfloat162float(p_bf[c_local]);
                    float d_val = smem_D[local_m];
                    float ds_val = p_val * (dp[c_local] - d_val);
                    ds[j] = __float2bfloat16(ds_val);
                }
                packed[i] = *(uint32_t*)&ds[0];
            }
            uint2 val = make_uint2(packed[0], packed[1]);
            *(uint2*)&smem_dS[tid * 128 + c*4] = val;
        }
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        __syncthreads(); 
        
        if (threadIdx.x == 0) {
            uint32_t idesc_dV = make_idesc(128, 128, 0, 1);
            for(int k=0; k<8; ++k) {
                uint64_t da = update_desc_k(make_k_major_desc(smem_P), (uint32_t)__cvta_generic_to_shared(smem_P) + k*32);
                uint64_t db = update_desc_k(make_mn_major_desc(smem_dO, 128), (uint32_t)__cvta_generic_to_shared(smem_dO) + k*4096);
                umma_f16_fn(tmem_dV, da, db, idesc_dV, m_block > n_block || k > 0);
            }
            
            uint32_t idesc_dK = make_idesc(128, 128, 0, 1);
            for(int k=0; k<8; ++k) {
                uint64_t da = update_desc_k(make_k_major_desc(smem_dS), (uint32_t)__cvta_generic_to_shared(smem_dS) + k*32);
                uint64_t db = update_desc_k(make_mn_major_desc(smem_Q, 128), (uint32_t)__cvta_generic_to_shared(smem_Q) + k*4096);
                umma_f16_fn(tmem_dK, da, db, idesc_dK, m_block > n_block || k > 0);
            }
            
            uint32_t idesc_dQ = make_idesc(128, 128, 1, 1);
            for(int k=0; k<8; ++k) {
                uint64_t da = update_desc_k(make_mn_major_desc(smem_dS, 128), (uint32_t)__cvta_generic_to_shared(smem_dS) + k*4096);
                uint64_t db = update_desc_k(make_mn_major_desc(smem_K, 128), (uint32_t)__cvta_generic_to_shared(smem_K) + k*4096);
                umma_f16_fn(tmem_dQ, da, db, idesc_dQ, k > 0);
            }
            
            umma_commit_1sm_fn(mbar_mma);
        }
        mbarrier_wait_fn(mbar_mma, mma_phase);
        mma_phase ^= 1;
        
        atomic_add_dq(dQ, smem_dS, S, d_dim, m_block, 128, 128, tmem_dQ, b, h);
    }
    
    store_dv_dk(dV, smem_dS, S, d_dim, n_block, 128, 128, tmem_dV, b, h);
    store_dv_dk(dK, smem_dS, S, d_dim, n_block, 128, 128, tmem_dK, b, h);
    
    __syncthreads();
    
    // tmem_dealloc must be executed collectively by the whole warp.
    if (threadIdx.x < 32) {
        tmem_dealloc_1cta_fn(tmem_base, 512);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int B_dim = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int d = Q.size(3);
    
    CUDA_CHECK(cudaMemsetAsync(dQ.data_ptr(), 0, B_dim * H * S * d * sizeof(uint16_t), stream));
    
    float* D_ptr;
    CUDA_CHECK(cudaMallocAsync(&D_ptr, B_dim * H * S * sizeof(float), stream));
    
    dim3 gridD((S + 127) / 128, H, B_dim);
    dim3 blockD(128);
    precompute_D_kernel<<<gridD, blockD, 0, stream>>>(
        (const __nv_bfloat16*)O.data_ptr(),
        (const __nv_bfloat16*)dO.data_ptr(),
        D_ptr, B_dim, H, S, d
    );
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_dO;
    CU_CHECK(create_tma_4d_descriptor(&tma_Q, Q.data_ptr(), d, S, H, B_dim, 128, 128));
    CU_CHECK(create_tma_4d_descriptor(&tma_K, K.data_ptr(), d, S, H, B_dim, 128, 128));
    CU_CHECK(create_tma_4d_descriptor(&tma_V, V.data_ptr(), d, S, H, B_dim, 128, 128));
    CU_CHECK(create_tma_4d_descriptor(&tma_dO, dO.data_ptr(), d, S, H, B_dim, 128, 128));
    
    dim3 grid((S + 127) / 128, H, B_dim);
    dim3 block(128);
    int smem_size = 128 + 6 * 128 * 128 * 2 + 2 * 128 * 4; 
    CUDA_CHECK(cudaFuncSetAttribute(bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    bwd_kernel<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, tma_dO,
        (__nv_bfloat16*)dQ.data_ptr(),
        (__nv_bfloat16*)dK.data_ptr(),
        (__nv_bfloat16*)dV.data_ptr(),
        (const float*)L.data_ptr(),
        D_ptr, B_dim, H, S, d
    );
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaFreeAsync(D_ptr, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda