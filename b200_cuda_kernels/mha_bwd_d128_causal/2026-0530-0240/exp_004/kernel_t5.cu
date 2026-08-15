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

__device__ __forceinline__ uint64_t make_smem_desc_none(void* smem_ptr, uint32_t stride_cols, bool is_major_k) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    
    uint32_t row_stride_bytes = stride_cols * 2;
    uint32_t lbo = 16; 
    uint32_t sbo = is_major_k ? (8 * row_stride_bytes) : row_stride_bytes;
    
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)0 << 61;   
    return d;
}

__device__ __forceinline__ void transpose_smem_64x128(const __nv_bfloat16* src, __nv_bfloat16* dst) {
    for (int idx = threadIdx.x; idx < 64 * 128; idx += blockDim.x) {
        int r = idx / 128;
        int c = idx % 128;
        dst[c * 64 + r] = src[idx];
    }
    __syncthreads();
}

__device__ __forceinline__ void transpose_smem_128x64(const __nv_bfloat16* src, __nv_bfloat16* dst) {
    for (int idx = threadIdx.x; idx < 128 * 64; idx += blockDim.x) {
        int r = idx / 64;
        int c = idx % 64;
        dst[c * 128 + r] = src[idx];
    }
    __syncthreads();
}

__device__ __forceinline__ void compute_umma_loop(
    uint32_t tmem_acc,
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    uint32_t M, uint32_t N, uint32_t K_inner,
    uint32_t stride_A_cols, uint32_t stride_B_cols,
    bool accumulate)
{
    if (threadIdx.x == 0) {
        uint32_t instr_desc = 0;
        instr_desc |= (1u << 4);    // c_format = FP32
        instr_desc |= (1u << 7);    // a_format = BF16
        instr_desc |= (1u << 10);   // b_format = BF16
        instr_desc |= (0u << 15);   // a_major = 0 (K-Major)
        instr_desc |= (1u << 16);   // b_major = 1 (N-Major)
        instr_desc |= ((N / 8) << 17);
        instr_desc |= ((M / 16) << 24);
        
        for (int k = 0; k < K_inner / 16; ++k) {
            uint32_t off_A = k * 16;
            uint32_t off_B = k * 16 * stride_B_cols;
            
            // Both are physically row-major 
            uint64_t desc_A = make_smem_desc_none((void*)(A + off_A), stride_A_cols, true);
            uint64_t desc_B = make_smem_desc_none((void*)(B + off_B), stride_B_cols, false); 
            
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
        uint32_t rS0, rS1, rS2, rS3;
        uint32_t rP0, rP1, rP2, rP3;
        
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(rS0),"=r"(rS1),"=r"(rS2),"=r"(rS3) : "r"(tmem_S + col_blk));
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(rP0),"=r"(rP1),"=r"(rP2),"=r"(rP3) : "r"(tmem_dP + col_blk));
            
        tmem_load_fence_fn();
        
        float s0 = __uint_as_float(rS0);
        float s1 = __uint_as_float(rS1);
        float s2 = __uint_as_float(rS2);
        float s3 = __uint_as_float(rS3);
        
        float dp0 = __uint_as_float(rP0);
        float dp1 = __uint_as_float(rP1);
        float dp2 = __uint_as_float(rP2);
        float dp3 = __uint_as_float(rP3);
        
        int q_idx0 = q_base + col_blk + 0;
        int q_idx1 = q_base + col_blk + 1;
        int q_idx2 = q_base + col_blk + 2;
        int q_idx3 = q_base + col_blk + 3;
        
        float lse0 = (q_idx0 < S_seq) ? LSE[q_idx0] : 0.0f;
        float lse1 = (q_idx1 < S_seq) ? LSE[q_idx1] : 0.0f;
        float lse2 = (q_idx2 < S_seq) ? LSE[q_idx2] : 0.0f;
        float lse3 = (q_idx3 < S_seq) ? LSE[q_idx3] : 0.0f;
        
        float d0 = (q_idx0 < S_seq) ? D[q_idx0] : 0.0f;
        float d1 = (q_idx1 < S_seq) ? D[q_idx1] : 0.0f;
        float d2 = (q_idx2 < S_seq) ? D[q_idx2] : 0.0f;
        float d3 = (q_idx3 < S_seq) ? D[q_idx3] : 0.0f;
        
        float scale = 1.0f / sqrtf(128.0f);
        s0 *= scale; s1 *= scale; s2 *= scale; s3 *= scale;
        
        if (q_idx0 >= S_seq || k_idx >= S_seq || k_idx > q_idx0) s0 = -INFINITY;
        if (q_idx1 >= S_seq || k_idx >= S_seq || k_idx > q_idx1) s1 = -INFINITY;
        if (q_idx2 >= S_seq || k_idx >= S_seq || k_idx > q_idx2) s2 = -INFINITY;
        if (q_idx3 >= S_seq || k_idx >= S_seq || k_idx > q_idx3) s3 = -INFINITY;
        
        float p0 = expf(s0 - lse0);
        float p1 = expf(s1 - lse1);
        float p2 = expf(s2 - lse2);
        float p3 = expf(s3 - lse3);
        
        float ds0 = p0 * (dp0 - d0) * scale;
        float ds1 = p1 * (dp1 - d1) * scale;
        float ds2 = p2 * (dp2 - d2) * scale;
        float ds3 = p3 * (dp3 - d3) * scale;
        
        __nv_bfloat16 val_p0 = __float2bfloat16(p0);
        __nv_bfloat16 val_p1 = __float2bfloat16(p1);
        __nv_bfloat16 val_p2 = __float2bfloat16(p2);
        __nv_bfloat16 val_p3 = __float2bfloat16(p3);
        
        uint32_t pt_01 = (uint32_t)*(uint16_t*)&val_p0 | ((uint32_t)*(uint16_t*)&val_p1 << 16);
        uint32_t pt_23 = (uint32_t)*(uint16_t*)&val_p2 | ((uint32_t)*(uint16_t*)&val_p3 << 16);
        
        __nv_bfloat16 val_ds0 = __float2bfloat16(ds0);
        __nv_bfloat16 val_ds1 = __float2bfloat16(ds1);
        __nv_bfloat16 val_ds2 = __float2bfloat16(ds2);
        __nv_bfloat16 val_ds3 = __float2bfloat16(ds3);
        
        uint32_t ds_01 = (uint32_t)*(uint16_t*)&val_ds0 | ((uint32_t)*(uint16_t*)&val_ds1 << 16);
        uint32_t ds_23 = (uint32_t)*(uint16_t*)&val_ds2 | ((uint32_t)*(uint16_t*)&val_ds3 << 16);
        
        int smem_idx = row * 64 + col_blk;
        *(uint2*)&smem_PT[smem_idx] = make_uint2(pt_01, pt_23);
        *(uint2*)&smem_dST[smem_idx] = make_uint2(ds_01, ds_23);
    }
    __syncthreads();
}

__device__ __forceinline__ void store_dQ(uint32_t tmem_dQ, __nv_bfloat16* dQ_out, int B_idx, int H_idx, int H, int S, int q_base) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int row = warp_id * 32 + lane_id;
    
    for (int col_blk = 0; col_blk < 128; col_blk += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_dQ + col_blk));
        tmem_load_fence_fn();
        
        if (row < 64) {
            int q_idx = q_base + row;
            if (q_idx < S) {
                __nv_bfloat16 val0 = __float2bfloat16(__uint_as_float(r0));
                __nv_bfloat16 val1 = __float2bfloat16(__uint_as_float(r1));
                __nv_bfloat16 val2 = __float2bfloat16(__uint_as_float(r2));
                __nv_bfloat16 val3 = __float2bfloat16(__uint_as_float(r3));
                
                uint32_t p0 = (uint32_t)*(uint16_t*)&val0 | ((uint32_t)*(uint16_t*)&val1 << 16);
                uint32_t p1 = (uint32_t)*(uint16_t*)&val2 | ((uint32_t)*(uint16_t*)&val3 << 16);
                
                __nv_bfloat16* dq_ptr = dQ_out + B_idx * H * S * 128 + H_idx * S * 128 + q_idx * 128 + col_blk;
                
                // Safely perform perfectly 4-byte aligned addition against __nv_bfloat162 
                atomicAdd((__nv_bfloat162*)(dq_ptr + 0), *(__nv_bfloat162*)&p0);
                atomicAdd((__nv_bfloat162*)(dq_ptr + 2), *(__nv_bfloat162*)&p1);
            }
        }
    }
}

__device__ __forceinline__ void store_tmem_to_gmem(uint32_t tmem_addr, __nv_bfloat16* gmem_ptr, int base_row, int max_row) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int row = warp_id * 32 + lane_id;
    
    for (int col_blk = 0; col_blk < 128; col_blk += 8) {
        uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) 
            : "r"(tmem_addr + col_blk));
        tmem_load_fence_fn();
        
        if (row < 128 && (base_row + row) < max_row) {
            __nv_bfloat16 val0 = __float2bfloat16(__uint_as_float(r0));
            __nv_bfloat16 val1 = __float2bfloat16(__uint_as_float(r1));
            __nv_bfloat16 val2 = __float2bfloat16(__uint_as_float(r2));
            __nv_bfloat16 val3 = __float2bfloat16(__uint_as_float(r3));
            __nv_bfloat16 val4 = __float2bfloat16(__uint_as_float(r4));
            __nv_bfloat16 val5 = __float2bfloat16(__uint_as_float(r5));
            __nv_bfloat16 val6 = __float2bfloat16(__uint_as_float(r6));
            __nv_bfloat16 val7 = __float2bfloat16(__uint_as_float(r7));
            
            __nv_bfloat16* out_ptr = gmem_ptr + row * 128 + col_blk;
            
            uint32_t p0 = (uint32_t)*(uint16_t*)&val0 | ((uint32_t)*(uint16_t*)&val1 << 16);
            uint32_t p1 = (uint32_t)*(uint16_t*)&val2 | ((uint32_t)*(uint16_t*)&val3 << 16);
            uint32_t p2 = (uint32_t)*(uint16_t*)&val4 | ((uint32_t)*(uint16_t*)&val5 << 16);
            uint32_t p3 = (uint32_t)*(uint16_t*)&val6 | ((uint32_t)*(uint16_t*)&val7 << 16);
            
            *(uint4*)out_ptr = make_uint4(p0, p1, p2, p3);
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
    __nv_bfloat16* smem_K   = smem;                             
    __nv_bfloat16* smem_V   = smem_K + 16384;                   
    __nv_bfloat16* smem_Q   = smem_V + 16384;                   
    __nv_bfloat16* smem_QT  = smem_Q + 8192;                    
    __nv_bfloat16* smem_dO  = smem_QT + 8192;                   
    __nv_bfloat16* smem_dOT = smem_dO + 8192;                   
    __nv_bfloat16* smem_PT  = smem_dOT + 8192;                  
    __nv_bfloat16* smem_dST = smem_PT + 8192;                   
    __nv_bfloat16* smem_dS  = smem_dST + 8192;                  

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

    uint32_t tmem_dK    = tmem_base;
    uint32_t tmem_dV    = tmem_base + 128;
    uint32_t tmem_S_mat = tmem_base + 256;
    uint32_t tmem_dP    = tmem_base + 320;
    uint32_t tmem_dQ    = tmem_base + 384;

    int B_idx = blockIdx.z;
    int H_idx = blockIdx.y;
    int j_blk = blockIdx.x; 

    int k_base = j_blk * 128;
    if (k_base >= S) return;

    if (threadIdx.x == 0) {
        uint32_t bytes_128x128 = 128 * 128 * 2;
        mbarrier_arrive_and_expect_tx_fn(&mbar_K, bytes_128x128);
        tma_load_2d_fn(&tma_K, &mbar_K, smem_K, 0, B_idx * H * S + H_idx * S + k_base);
        
        mbarrier_arrive_and_expect_tx_fn(&mbar_V, bytes_128x128);
        tma_load_2d_fn(&tma_V, &mbar_V, smem_V, 0, B_idx * H * S + H_idx * S + k_base);
    }

    int phase_Q = 0;
    int phase_dO = 0;
    int phase_UMMA = 0;

    int start_i_blk = j_blk * 2;
    bool first_q = true;

    for (int i_blk = start_i_blk; i_blk * 64 < S; ++i_blk) {
        int q_base = i_blk * 64;
        
        if (threadIdx.x == 0) {
            uint32_t bytes_64x128 = 64 * 128 * 2;
            mbarrier_arrive_and_expect_tx_fn(&mbar_Q, bytes_64x128);
            tma_load_2d_fn(&tma_Q, &mbar_Q, smem_Q, 0, B_idx * H * S + H_idx * S + q_base);
            
            mbarrier_arrive_and_expect_tx_fn(&mbar_dO, bytes_64x128);
            tma_load_2d_fn(&tma_dO, &mbar_dO, smem_dO, 0, B_idx * H * S + H_idx * S + q_base);
        }
        
        if (first_q) {
            mbarrier_wait_fn(&mbar_K, 0);
            mbarrier_wait_fn(&mbar_V, 0);
        }
        mbarrier_wait_fn(&mbar_Q, phase_Q);
        mbarrier_wait_fn(&mbar_dO, phase_dO);
        
        fence_proxy_async_fn();
        
        transpose_smem_64x128(smem_Q, smem_QT);
        transpose_smem_64x128(smem_dO, smem_dOT);
        
        // Ensure generic writes are visible to UMMA proxy
        fence_proxy_async_fn(); 
        tcgen05_fence_after_fn();
        
        // S = K @ Q^T
        compute_umma_loop(tmem_S_mat, smem_K, smem_QT, 128, 64, 128, 128, 64, false);
        
        // dP = V @ dOT
        compute_umma_loop(tmem_dP, smem_V, smem_dOT, 128, 64, 128, 128, 64, false);
        
        if (threadIdx.x == 0) umma_commit_cg1_fn(&mbar_UMMA);
        mbarrier_wait_fn(&mbar_UMMA, phase_UMMA);
        phase_UMMA ^= 1;
        
        process_and_store(tmem_S_mat, tmem_dP, smem_PT, smem_dST, LSE + B_idx*H*S + H_idx*S, D + B_idx*H*S + H_idx*S, q_base, k_base, S);
        
        transpose_smem_128x64(smem_dST, smem_dS);
        
        fence_proxy_async_fn();
        tcgen05_fence_after_fn();
        
        // dQ = dS @ K
        compute_umma_loop(tmem_dQ, smem_dS, smem_K, 64, 128, 128, 128, 128, false);
        
        // dK = dST @ Q
        compute_umma_loop(tmem_dK, smem_dST, smem_Q, 128, 128, 64, 64, 128, !first_q);
        
        // dV = PT @ dO
        compute_umma_loop(tmem_dV, smem_PT, smem_dO, 128, 128, 64, 64, 128, !first_q);
        
        if (threadIdx.x == 0) umma_commit_cg1_fn(&mbar_UMMA);
        mbarrier_wait_fn(&mbar_UMMA, phase_UMMA);
        phase_UMMA ^= 1;
        
        store_dQ(tmem_dQ, dQ_out, B_idx, H_idx, H, S, q_base);
        
        __syncthreads();
        phase_Q ^= 1;
        phase_dO ^= 1;
        first_q = false;
    }
    
    store_tmem_to_gmem(tmem_dK, dK_out + B_idx * H * S * 128 + H_idx * S * 128 + k_base * 128, k_base, S);
    store_tmem_to_gmem(tmem_dV, dV_out + B_idx * H * S * 128 + H_idx * S * 128 + k_base * 128, k_base, S);

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
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), 128, B*H*S, 128, 64, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), 128, B*H*S, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), 128, B*H*S, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_dO, dO.data_ptr(), 128, B*H*S, 128, 64, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    dim3 grid((S + 127) / 128, H, B);
    dim3 block(128);
    int smem_size = 180224; 

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