#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
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
        fprintf(stderr, "CU error %d at %s:%d\n", _e, __FILE__, __LINE__); \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_mha_bwd {

// ---------------------------------------------------------
// Precompute D = sum(dO * O, dim=-1)
// ---------------------------------------------------------
__global__ void precompute_D_kernel(const __nv_bfloat16* dO, const __nv_bfloat16* O, float* D, int B, int H, int S, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * H * S) {
        float sum = 0;
        for (int i = 0; i < d; ++i) {
            float do_val = __bfloat162float(dO[idx * d + i]);
            float o_val  = __bfloat162float(O[idx * d + i]);
            sum += do_val * o_val;
        }
        D[idx] = sum;
    }
}

// ---------------------------------------------------------
// Helper intrinsics for SM100
// ---------------------------------------------------------
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

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tmem_alloc_cg1_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_cg1_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void umma_f16_cg1_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
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

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    // c_format = FP32
    d |= (1u << 7);    // a_format = BF16
    d |= (1u << 10);   // b_format = BF16
    d |= (0u << 15);   // a_major = 0 (K-Major)
    d |= (0u << 16);   // b_major = 0 (K-Major)
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_swizzle128b_kmajor(void* smem_ptr, uint32_t stride_cols) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    uint32_t sbo = 8 * (stride_cols * 2); // 8 rows * row_stride_bytes
    uint32_t lbo = 1; 
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)((addr >> 7) & 0x7) << 49;
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ void transpose_smem_swizzled(
    const __nv_bfloat16* src, int src_rows, int src_cols,
    __nv_bfloat16* dst, int dst_rows, int dst_cols) 
{
    for (int idx = threadIdx.x; idx < src_rows * src_cols; idx += blockDim.x) {
        int r = idx / src_cols;
        int c = idx % src_cols;
        
        int src_chunk = c / 8;
        int src_swizzled_chunk = (r % 8) ^ (src_chunk % 8); 
        int src_swizzled_col = (src_chunk / 8) * 64 + src_swizzled_chunk * 8 + (c % 8);
        __nv_bfloat16 val = src[r * src_cols + src_swizzled_col];
        
        int dr = c;
        int dc = r;
        int dst_chunk = dc / 8;
        int dst_swizzled_chunk = (dr % 8) ^ (dst_chunk % 8);
        int dst_swizzled_col = (dst_chunk / 8) * 64 + dst_swizzled_chunk * 8 + (dc % 8);
        dst[dr * dst_cols + dst_swizzled_col] = val;
    }
    __syncthreads();
}

__device__ __forceinline__ void compute_umma_loop(
    uint32_t tmem_acc,
    const __nv_bfloat16* A, const __nv_bfloat16* B,
    uint32_t M, uint32_t N, uint32_t K_inner,
    uint32_t stride_A_cols, uint32_t stride_B_cols,
    bool accumulate)
{
    if (threadIdx.x == 0) {
        uint32_t instr_desc = make_instr_desc_fn(M, N); 
        for (int k = 0; k < K_inner / 16; ++k) {
            uint32_t off_A = k * 16; 
            uint32_t off_B = k * 16 * stride_B_cols; 
            
            uint64_t desc_A = make_smem_desc_swizzle128b_kmajor((void*)(A + off_A), stride_A_cols);
            uint64_t desc_B = make_smem_desc_swizzle128b_kmajor((void*)(B + off_B), stride_B_cols); 
            
            bool acc = accumulate || (k > 0);
            umma_f16_cg1_fn(tmem_acc, desc_A, desc_B, instr_desc, acc ? 1 : 0);
        }
    }
}

__device__ __forceinline__ void process_and_store(
    uint32_t tmem_S, uint32_t tmem_dP,
    __nv_bfloat16* smem_PT, __nv_bfloat16* smem_dST,
    const float* LSE, const float* D,
    int q_base, int k_base, int S_seq)
{
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int row = warp_id * 32 + lane_id; 
    int k_idx = k_base + row;
    
    for (int col_blk = 0; col_blk < 64; col_blk += 4) {
        uint32_t rS[4], rP[4];
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(rS[0]),"=r"(rS[1]),"=r"(rS[2]),"=r"(rS[3]) : "r"(tmem_S + col_blk));
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(rP[0]),"=r"(rP[1]),"=r"(rP[2]),"=r"(rP[3]) : "r"(tmem_dP + col_blk));
            
        tmem_load_fence_fn();
        uint32_t pt[2], dst[2];
        
        for(int i = 0; i < 4; i+=2) {
            float s0 = __uint_as_float(rS[i]);
            float dp0 = __uint_as_float(rP[i]);
            float s1 = __uint_as_float(rS[i+1]);
            float dp1 = __uint_as_float(rP[i+1]);
            
            int q_idx0 = q_base + col_blk + i;
            int q_idx1 = q_base + col_blk + i + 1;
            
            float lse0 = (q_idx0 < S_seq) ? LSE[q_idx0] : 0.0f;
            float lse1 = (q_idx1 < S_seq) ? LSE[q_idx1] : 0.0f;
            
            float d0 = (q_idx0 < S_seq) ? D[q_idx0] : 0.0f;
            float d1 = (q_idx1 < S_seq) ? D[q_idx1] : 0.0f;
            
            float scale = 1.0f / sqrtf(128.0f);
            s0 *= scale; s1 *= scale;
            
            if (q_idx0 >= S_seq || k_idx >= S_seq || k_idx > q_idx0) s0 = -INFINITY;
            if (q_idx1 >= S_seq || k_idx >= S_seq || k_idx > q_idx1) s1 = -INFINITY;
            
            float p0 = expf(s0 - lse0);
            float p1 = expf(s1 - lse1);
            
            float ds0 = p0 * (dp0 - d0) * scale;
            float ds1 = p1 * (dp1 - d1) * scale;
            
            __nv_bfloat16 val_p0 = __float2bfloat16(p0);
            __nv_bfloat16 val_p1 = __float2bfloat16(p1);
            pt[i/2] = (uint32_t)*(uint16_t*)&val_p0 | ((uint32_t)*(uint16_t*)&val_p1 << 16);
            
            __nv_bfloat16 val_ds0 = __float2bfloat16(ds0);
            __nv_bfloat16 val_ds1 = __float2bfloat16(ds1);
            dst[i/2] = (uint32_t)*(uint16_t*)&val_ds0 | ((uint32_t)*(uint16_t*)&val_ds1 << 16);
        }
        
        int chunk = col_blk / 8;
        int swizzled_chunk = (row % 8) ^ (chunk % 8);
        int swizzled_col = (chunk / 8) * 64 + swizzled_chunk * 8 + (col_blk % 8);
        int smem_idx = row * 64 + swizzled_col;
        
        *(uint2*)&smem_PT[smem_idx] = make_uint2(pt[0], pt[1]);
        *(uint2*)&smem_dST[smem_idx] = make_uint2(dst[0], dst[1]);
    }
    __syncthreads();
}

__device__ __forceinline__ void store_dQ(uint32_t tmem0, uint32_t tmem1, __nv_bfloat16* dQ_out, int B_idx, int H_idx, int H, int S, int q_base) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int row = warp_id * 32 + lane_id;
    
    for (int half = 0; half < 2; ++half) {
        uint32_t tmem_addr = (half == 0) ? tmem0 : tmem1;
        int col_offset = (half == 0) ? 0 : 64;
        
        for (int col_blk = 0; col_blk < 64; col_blk += 4) {
            uint32_t r[4];
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]) : "r"(tmem_addr + col_blk));
            tmem_load_fence_fn();
            
            if (row < 64) {
                int q_idx = q_base + row;
                if (q_idx < S) {
                    __nv_bfloat16 v0 = __float2bfloat16(__uint_as_float(r[0]));
                    __nv_bfloat16 v1 = __float2bfloat16(__uint_as_float(r[1]));
                    __nv_bfloat16 v2 = __float2bfloat16(__uint_as_float(r[2]));
                    __nv_bfloat16 v3 = __float2bfloat16(__uint_as_float(r[3]));
                    
                    uint32_t p0 = (uint32_t)*(uint16_t*)&v0 | ((uint32_t)*(uint16_t*)&v1 << 16);
                    uint32_t p1 = (uint32_t)*(uint16_t*)&v2 | ((uint32_t)*(uint16_t*)&v3 << 16);
                    
                    __nv_bfloat16* dq_ptr = dQ_out + B_idx * H * S * 128 + H_idx * S * 128 + q_idx * 128 + col_offset + col_blk;
                    atomicAdd((__nv_bfloat162*)(dq_ptr + 0), *(__nv_bfloat162*)&p0);
                    atomicAdd((__nv_bfloat162*)(dq_ptr + 2), *(__nv_bfloat162*)&p1);
                }
            }
        }
    }
}

__device__ __forceinline__ void store_tmem_to_gmem_128(uint32_t tmem0, uint32_t tmem1, __nv_bfloat16* gmem_ptr, int base_row, int max_row) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int row = warp_id * 32 + lane_id;
    
    for (int half = 0; half < 2; ++half) {
        uint32_t tmem_addr = (half == 0) ? tmem0 : tmem1;
        int col_offset = (half == 0) ? 0 : 64;
        
        for (int col_blk = 0; col_blk < 64; col_blk += 8) {
            uint32_t r[8];
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]) 
                : "r"(tmem_addr + col_blk));
            tmem_load_fence_fn();
            
            if (row < 128 && (base_row + row) < max_row) {
                uint32_t p[4];
                for (int i = 0; i < 4; ++i) {
                    __nv_bfloat16 v0 = __float2bfloat16(__uint_as_float(r[i*2 + 0]));
                    __nv_bfloat16 v1 = __float2bfloat16(__uint_as_float(r[i*2 + 1]));
                    p[i] = (uint32_t)*(uint16_t*)&v0 | ((uint32_t)*(uint16_t*)&v1 << 16);
                }
                
                __nv_bfloat16* out_ptr = gmem_ptr + row * 128 + col_offset + col_blk; 
                *(uint4*)out_ptr = make_uint4(p[0], p[1], p[2], p[3]);
            }
        }
    }
}

// ---------------------------------------------------------
// Main Backward Kernel 
// ---------------------------------------------------------
__global__ void mha_bwd_d128_causal_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_dO,
    __nv_bfloat16* dQ_out,
    __nv_bfloat16* dK_out,
    __nv_bfloat16* dV_out,
    const float* LSE,
    const float* D,
    int B, int H, int S)
{
    extern __shared__ __align__(16) __nv_bfloat16 smem[];
    __nv_bfloat16* smem_K0   = smem;                             
    __nv_bfloat16* smem_K1   = smem_K0 + 8192;                   
    __nv_bfloat16* smem_V0   = smem_K1 + 8192;                   
    __nv_bfloat16* smem_V1   = smem_V0 + 8192;                   
    __nv_bfloat16* smem_Q0   = smem_V1 + 8192;                   
    __nv_bfloat16* smem_Q1   = smem_Q0 + 4096;                   
    __nv_bfloat16* smem_dO0  = smem_Q1 + 4096;                   
    __nv_bfloat16* smem_dO1  = smem_dO0 + 4096;                  
    
    __nv_bfloat16* smem_Q0T  = smem_dO1 + 4096;                  
    __nv_bfloat16* smem_Q1T  = smem_Q0T + 4096;                  
    __nv_bfloat16* smem_dO0T = smem_Q1T + 4096;                  
    __nv_bfloat16* smem_dO1T = smem_dO0T + 4096;                 
    
    __nv_bfloat16* smem_PT   = smem_dO1T + 4096;                 
    __nv_bfloat16* smem_dST  = smem_PT + 8192;                   
    
    __nv_bfloat16* smem_dS0  = smem_dST + 8192;                  
    __nv_bfloat16* smem_dS1  = smem_dS0 + 4096;                  

    __shared__ uint64_t mbar_K;
    __shared__ uint64_t mbar_V;
    __shared__ uint64_t mbar_Q;
    __shared__ uint64_t mbar_dO;
    __shared__ uint64_t mbar_UMMA;
    __shared__ uint32_t tmem_base;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&mbar_K, 1);
        init_smem_barrier_fn(&mbar_V, 1);
        init_smem_barrier_fn(&mbar_Q, 1);
        init_smem_barrier_fn(&mbar_dO, 1);
        init_smem_barrier_fn(&mbar_UMMA, 1);
        fence_smem_barrier_init_fn();
    }
    
    // Exactly one warp collectively allocates the TMEM
    if (threadIdx.x < 32) {
        tmem_alloc_cg1_fn(&tmem_base, 512);
    }
    __syncthreads();

    uint32_t tmem_dK0    = tmem_base + 0;
    uint32_t tmem_dK1    = tmem_base + 64;
    uint32_t tmem_dV0    = tmem_base + 128;
    uint32_t tmem_dV1    = tmem_base + 192;
    uint32_t tmem_dQ0    = tmem_base + 256;
    uint32_t tmem_dQ1    = tmem_base + 320;
    uint32_t tmem_S_mat  = tmem_base + 384;
    uint32_t tmem_dP     = tmem_base + 448;

    int B_idx = blockIdx.z;
    int H_idx = blockIdx.y;
    int j_blk = blockIdx.x; 

    int k_base = j_blk * 128;
    if (k_base >= S) return;

    if (threadIdx.x == 0) {
        uint32_t bytes_128x64 = 128 * 64 * 2;
        mbarrier_arrive_and_expect_tx_fn(&mbar_K, bytes_128x64 * 2);
        tma_load_2d_fn(&tma_K, &mbar_K, smem_K0, 0,  B_idx * H * S + H_idx * S + k_base);
        tma_load_2d_fn(&tma_K, &mbar_K, smem_K1, 64, B_idx * H * S + H_idx * S + k_base);
        
        mbarrier_arrive_and_expect_tx_fn(&mbar_V, bytes_128x64 * 2);
        tma_load_2d_fn(&tma_V, &mbar_V, smem_V0, 0,  B_idx * H * S + H_idx * S + k_base);
        tma_load_2d_fn(&tma_V, &mbar_V, smem_V1, 64, B_idx * H * S + H_idx * S + k_base);
    }

    int phase_Q = 0;
    int phase_dO = 0;
    int phase_UMMA = 0;

    int start_i_blk = j_blk * 2;
    bool first_q = true;

    for (int i_blk = start_i_blk; i_blk * 64 < S; ++i_blk) {
        int q_base = i_blk * 64;
        
        if (threadIdx.x == 0) {
            uint32_t bytes_64x64 = 64 * 64 * 2;
            mbarrier_arrive_and_expect_tx_fn(&mbar_Q, bytes_64x64 * 2);
            tma_load_2d_fn(&tma_Q, &mbar_Q, smem_Q0, 0,  B_idx * H * S + H_idx * S + q_base);
            tma_load_2d_fn(&tma_Q, &mbar_Q, smem_Q1, 64, B_idx * H * S + H_idx * S + q_base);
            
            mbarrier_arrive_and_expect_tx_fn(&mbar_dO, bytes_64x64 * 2);
            tma_load_2d_fn(&tma_dO, &mbar_dO, smem_dO0, 0,  B_idx * H * S + H_idx * S + q_base);
            tma_load_2d_fn(&tma_dO, &mbar_dO, smem_dO1, 64, B_idx * H * S + H_idx * S + q_base);
        }
        
        if (first_q) {
            mbarrier_wait_fn(&mbar_K, 0);
            mbarrier_wait_fn(&mbar_V, 0);
        }
        mbarrier_wait_fn(&mbar_Q, phase_Q);
        mbarrier_wait_fn(&mbar_dO, phase_dO);
        
        fence_proxy_async_fn();
        
        transpose_smem_swizzled(smem_Q0, 64, 64, smem_Q0T, 64, 64);
        transpose_smem_swizzled(smem_Q1, 64, 64, smem_Q1T, 64, 64);
        transpose_smem_swizzled(smem_dO0, 64, 64, smem_dO0T, 64, 64);
        transpose_smem_swizzled(smem_dO1, 64, 64, smem_dO1T, 64, 64);
        
        // Ensure generic writes are visible to UMMA proxy
        fence_proxy_async_fn(); 
        tcgen05_fence_after_fn();
        
        // S = K @ Q^T
        compute_umma_loop(tmem_S_mat, smem_K0, smem_Q0T, 128, 64, 64, 64, 64, false);
        compute_umma_loop(tmem_S_mat, smem_K1, smem_Q1T, 128, 64, 64, 64, 64, true);
        
        // dP = V @ dOT
        compute_umma_loop(tmem_dP, smem_V0, smem_dO0T, 128, 64, 64, 64, 64, false);
        compute_umma_loop(tmem_dP, smem_V1, smem_dO1T, 128, 64, 64, 64, 64, true);
        
        if (threadIdx.x == 0) umma_commit_cg1_fn(&mbar_UMMA);
        mbarrier_wait_fn(&mbar_UMMA, phase_UMMA);
        phase_UMMA ^= 1;
        
        process_and_store(tmem_S_mat, tmem_dP, smem_PT, smem_dST, LSE + B_idx*H*S + H_idx*S, D + B_idx*H*S + H_idx*S, q_base, k_base, S);
        
        // Split dST (128x64) into dS0 (64x64) and dS1 (64x64) while transposing
        transpose_smem_swizzled(smem_dST, 64, 64, smem_dS0, 64, 64);
        transpose_smem_swizzled(smem_dST + 4096, 64, 64, smem_dS1, 64, 64);
        
        fence_proxy_async_fn();
        tcgen05_fence_after_fn();
        
        // dQ0 = dS0 @ K0_top + dS1 @ K0_bottom
        compute_umma_loop(tmem_dQ0, smem_dS0, smem_K0, 64, 64, 64, 64, 64, false);
        compute_umma_loop(tmem_dQ0, smem_dS1, smem_K0 + 4096, 64, 64, 64, 64, 64, true);
        
        // dQ1 = dS0 @ K1_top + dS1 @ K1_bottom
        compute_umma_loop(tmem_dQ1, smem_dS0, smem_K1, 64, 64, 64, 64, 64, false);
        compute_umma_loop(tmem_dQ1, smem_dS1, smem_K1 + 4096, 64, 64, 64, 64, 64, true);
        
        // dK = dST @ Q
        compute_umma_loop(tmem_dK0, smem_dST, smem_Q0, 128, 64, 64, 64, 64, !first_q);
        compute_umma_loop(tmem_dK1, smem_dST, smem_Q1, 128, 64, 64, 64, 64, !first_q);
        
        // dV = PT @ dO
        compute_umma_loop(tmem_dV0, smem_PT, smem_dO0, 128, 64, 64, 64, 64, !first_q);
        compute_umma_loop(tmem_dV1, smem_PT, smem_dO1, 128, 64, 64, 64, 64, !first_q);
        
        if (threadIdx.x == 0) umma_commit_cg1_fn(&mbar_UMMA);
        mbarrier_wait_fn(&mbar_UMMA, phase_UMMA);
        phase_UMMA ^= 1;
        
        store_dQ(tmem_dQ0, tmem_dQ1, dQ_out, B_idx, H_idx, H, S, q_base);
        
        __syncthreads();
        phase_Q ^= 1;
        phase_dO ^= 1;
        first_q = false;
    }
    
    store_tmem_to_gmem_128(tmem_dK0, tmem_dK1, dK_out + B_idx * H * S * 128 + H_idx * S * 128 + k_base * 128, k_base, S);
    store_tmem_to_gmem_128(tmem_dV0, tmem_dV1, dV_out + B_idx * H * S * 128 + H_idx * S * 128 + k_base * 128, k_base, S);

    __syncthreads();
    if (threadIdx.x < 32) {
        tmem_dealloc_cg1_fn(tmem_base, 512);
    }
}

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

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L, tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = Q.size(3); 
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaMemsetAsync(dQ.data_ptr(), 0, B * H * S * d * sizeof(__nv_bfloat16), stream));

    float* D_ptr;
    CUDA_CHECK(cudaMallocAsync(&D_ptr, B * H * S * sizeof(float), stream));
    
    precompute_D_kernel<<<(B * H * S + 255) / 256, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const __nv_bfloat16*>(O.data_ptr()),
        D_ptr, B, H, S, d);

    CUtensorMap tma_Q, tma_K, tma_V, tma_dO;
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), 128, B*H*S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), 128, B*H*S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), 128, B*H*S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_dO, dO.data_ptr(), 128, B*H*S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    dim3 grid((S + 127) / 128, H, B);
    dim3 block(128);
    int smem_size = 188416; // 184 KB 

    CUDA_CHECK(cudaFuncSetAttribute(mha_bwd_d128_causal_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    mha_bwd_d128_causal_kernel<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, tma_dO,
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        D_ptr,
        B, H, S);
    
    CUDA_CHECK(cudaFreeAsync(D_ptr, stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_mha_bwd