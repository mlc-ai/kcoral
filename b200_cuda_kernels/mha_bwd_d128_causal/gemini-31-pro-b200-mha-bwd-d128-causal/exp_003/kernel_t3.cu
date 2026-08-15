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

namespace tvm_ffi_example_cuda {

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16; 
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32; 
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, int a_major, int b_major) {
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

__device__ __forceinline__ void umma_f16_cg1_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void tcgen05_commit_cg1_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        :: "r"(a) : "memory");
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

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_fn(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])) : "memory");
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

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
}

__device__ __forceinline__ void tma_store_3d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [%0, {%2, %3, %4}], [%1];"
        :: "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
}

__device__ __forceinline__ void tma_store_commit_fn() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void tma_store_wait_fn() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ uint32_t pack_bf16(float f0, float f1) {
    __nv_bfloat162 res = __floats2bfloat162_rn(f0, f1);
    return *reinterpret_cast<uint32_t*>(&res);
}

__device__ __forceinline__ uint32_t swizzle_128B(uint32_t row, uint32_t byte_col) {
    uint32_t chunk_x = byte_col / 16;
    uint32_t offset = byte_col % 16;
    uint32_t swizzled_chunk = (row % 8) ^ chunk_x;
    return row * 128 + swizzled_chunk * 16 + offset;
}

#define UMMA_LOOP_K64_KK(tmem, ptrA, ptrB, idesc, accum_start) \
    for (int k = 0; k < 4; k++) { \
        uint64_t descA = make_smem_desc_sm100_fn((char*)ptrA + k * 32, 1, 1024); \
        uint64_t descB = make_smem_desc_sm100_fn((char*)ptrB + k * 32, 1, 1024); \
        umma_f16_cg1_fn(tmem, descA, descB, idesc, (accum_start == 0 && k == 0) ? 0 : 1); \
    }

#define UMMA_LOOP_K64_KM(tmem, ptrA, ptrB, idesc, accum_start) \
    for (int k = 0; k < 4; k++) { \
        uint64_t descA = make_smem_desc_sm100_fn((char*)ptrA + k * 32, 1, 1024); \
        uint64_t descB = make_smem_desc_sm100_fn((char*)ptrB + k * 16 * 128, 2048, 1024); \
        umma_f16_cg1_fn(tmem, descA, descB, idesc, (accum_start == 0 && k == 0) ? 0 : 1); \
    }

#define UMMA_LOOP_K128_MM(tmem, ptrA, ptrB, idesc, accum_start) \
    for (int k = 0; k < 8; k++) { \
        uint64_t descA = make_smem_desc_sm100_fn((char*)ptrA + k * 2048, 2048, 1024); \
        uint64_t descB = make_smem_desc_sm100_fn((char*)ptrB + k * 2048, 2048, 1024); \
        umma_f16_cg1_fn(tmem, descA, descB, idesc, (accum_start == 0 && k == 0) ? 0 : 1); \
    }

__global__ void precompute_D_kernel(
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    float* __restrict__ D,
    int max_rows)
{
    int row = blockIdx.x * 4 + threadIdx.y;
    int tid = threadIdx.x;
    if (row < max_rows) {
        float sum = 0;
        for (int i = tid * 2; i < 128; i += 64) {
            __nv_bfloat162 o_val = *(__nv_bfloat162*)&O[row * 128 + i];
            __nv_bfloat162 do_val = *(__nv_bfloat162*)&dO[row * 128 + i];
            float2 fo = __bfloat1622float2(o_val);
            float2 fdo = __bfloat1622float2(do_val);
            sum += fo.x * fdo.x + fo.y * fdo.y;
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (tid == 0) D[row] = sum;
    }
}

__global__ void mha_bwd_kernel_1(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_dO,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_dQ,
    const float* __restrict__ L, 
    const float* __restrict__ D,
    int S)
{
    int block_i = blockIdx.x;
    int bh = blockIdx.y;
    int tid = threadIdx.x; 
    int lane_id = tid % 32;
    int warp_id = tid / 32;
    int row = warp_id * 32 + lane_id;

    extern __shared__ __align__(1024) char smem[];
    __nv_bfloat16* s_Q0 = (__nv_bfloat16*)(smem + 0);         
    __nv_bfloat16* s_Q1 = (__nv_bfloat16*)(smem + 16384);     
    __nv_bfloat16* s_dO0 = (__nv_bfloat16*)(smem + 32768);    
    __nv_bfloat16* s_dO1 = (__nv_bfloat16*)(smem + 49152);    
    __nv_bfloat16* s_K0 = (__nv_bfloat16*)(smem + 65536);     
    __nv_bfloat16* s_K1 = (__nv_bfloat16*)(smem + 73728);     
    __nv_bfloat16* s_V0 = (__nv_bfloat16*)(smem + 81920);     
    __nv_bfloat16* s_V1 = (__nv_bfloat16*)(smem + 90112);     
    __nv_bfloat16* s_dS = (__nv_bfloat16*)(smem + 98304);     

    uint32_t tmem_S, tmem_dP, tmem_dQ0, tmem_dQ1;
    if (warp_id == 0) {
        tmem_alloc_cg1_fn(&tmem_S, 64);
        tmem_alloc_cg1_fn(&tmem_dP, 64);
        tmem_alloc_cg1_fn(&tmem_dQ0, 64);
        tmem_alloc_cg1_fn(&tmem_dQ1, 64);
    }
    __syncthreads();
    
    uint64_t* mbar = (uint64_t*)(smem + 132096); 
    uint64_t* mbar_umma = (uint64_t*)(smem + 132104);
    
    if (tid == 0) {
        init_smem_barrier_fn(mbar, 128);
        init_smem_barrier_fn(mbar_umma, 1);
    }
    __syncthreads();

    int phase = 0;
    int phase_umma = 0;

    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar, 65536); 
        tma_load_3d_fn(&tma_Q, mbar, s_Q0, 0, block_i * 128, bh);
        tma_load_3d_fn(&tma_Q, mbar, s_Q1, 64, block_i * 128, bh);
        tma_load_3d_fn(&tma_dO, mbar, s_dO0, 0, block_i * 128, bh);
        tma_load_3d_fn(&tma_dO, mbar, s_dO1, 64, block_i * 128, bh);
    } else {
        mbarrier_arrive_fn(mbar);
    }
    mbarrier_wait_fn(mbar, phase);
    phase ^= 1;
    
    float L_val = 0, D_val = 0;
    int global_row_ = bh * S + block_i * 128 + row;
    if (block_i * 128 + row < S) {
        L_val = L[global_row_];
        D_val = D[global_row_];
    }

    int accum_dQ_start = 0;
    
    int j_max = block_i * 2 + 1;
    for (int j_blk = 0; j_blk <= j_max; j_blk++) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar, 32768); 
            tma_load_3d_fn(&tma_K, mbar, s_K0, 0, j_blk * 64, bh);
            tma_load_3d_fn(&tma_K, mbar, s_K1, 64, j_blk * 64, bh);
            tma_load_3d_fn(&tma_V, mbar, s_V0, 0, j_blk * 64, bh);
            tma_load_3d_fn(&tma_V, mbar, s_V1, 64, j_blk * 64, bh);
        } else {
            mbarrier_arrive_fn(mbar);
        }
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        
        uint32_t idesc_S = make_instr_desc_fn(128, 64, 0, 0);
        UMMA_LOOP_K64_KK(tmem_S, s_Q0, s_K0, idesc_S, 0);
        UMMA_LOOP_K64_KK(tmem_S, s_Q1, s_K1, idesc_S, 1);
        
        UMMA_LOOP_K64_KK(tmem_dP, s_dO0, s_V0, idesc_S, 0);
        UMMA_LOOP_K64_KK(tmem_dP, s_dO1, s_V1, idesc_S, 1);
        
        if (tid == 0) tcgen05_commit_cg1_fn(mbar_umma);
        mbarrier_wait_fn(mbar_umma, phase_umma);
        phase_umma ^= 1;
        
        for (int col = 0; col < 64; col += 4) {
            uint32_t sr0, sr1, sr2, sr3, pr0, pr1, pr2, pr3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(sr0),"=r"(sr1),"=r"(sr2),"=r"(sr3) : "r"(tmem_S + col));
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(pr0),"=r"(pr1),"=r"(pr2),"=r"(pr3) : "r"(tmem_dP + col));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float s[4] = { __uint_as_float(sr0), __uint_as_float(sr1), __uint_as_float(sr2), __uint_as_float(sr3) };
            float dp[4] = { __uint_as_float(pr0), __uint_as_float(pr1), __uint_as_float(pr2), __uint_as_float(pr3) };
            
            float ds[4] = {0,0,0,0};
            for (int c = 0; c < 4; c++) {
                int global_col = j_blk * 64 + col + c;
                int gr = block_i * 128 + row;
                if (global_col <= gr && gr < S && global_col < S) {
                    float p = expf(s[c] * 0.0883883476483f - L_val);
                    ds[c] = p * (dp[c] - D_val) * 0.0883883476483f;
                }
            }
            
            uint32_t packed0 = pack_bf16(ds[0], ds[1]);
            uint32_t packed1 = pack_bf16(ds[2], ds[3]);
            
            *(uint32_t*)((char*)s_dS + swizzle_128B(row, (col + 0) * 2)) = packed0;
            *(uint32_t*)((char*)s_dS + swizzle_128B(row, (col + 2) * 2)) = packed1;
        }
        
        fence_proxy_async_fn();
        __syncthreads();
        
        uint32_t idesc_dQ = make_instr_desc_fn(128, 64, 0, 1);
        UMMA_LOOP_K64_KM(tmem_dQ0, s_dS, s_K0, idesc_dQ, accum_dQ_start);
        UMMA_LOOP_K64_KM(tmem_dQ1, s_dS, s_K1, idesc_dQ, accum_dQ_start);
        accum_dQ_start = 1;
        
        if (tid == 0) tcgen05_commit_cg1_fn(mbar_umma);
        mbarrier_wait_fn(mbar_umma, phase_umma);
        phase_umma ^= 1;
    }
    
    for (int pass = 0; pass < 2; pass++) {
        uint32_t tmem_src = (pass == 0) ? tmem_dQ0 : tmem_dQ1;
        for (int col = 0; col < 64; col += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_src + col));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            uint32_t packed0 = pack_bf16(__uint_as_float(r0), __uint_as_float(r1));
            uint32_t packed1 = pack_bf16(__uint_as_float(r2), __uint_as_float(r3));
            
            *(uint32_t*)((char*)s_dS + swizzle_128B(row, (col + 0) * 2)) = packed0;
            *(uint32_t*)((char*)s_dS + swizzle_128B(row, (col + 2) * 2)) = packed1;
        }
        fence_proxy_async_fn();
        __syncthreads();
        
        if (tid == 0) {
            tma_store_3d_fn(&tma_dQ, s_dS, pass * 64, block_i * 128, bh);
            tma_store_commit_fn();
        }
        tma_store_wait_fn<0>();
        __syncthreads();
    }
    
    if (warp_id == 0) {
        tmem_dealloc_cg1_fn(tmem_S, 64);
        tmem_dealloc_cg1_fn(tmem_dP, 64);
        tmem_dealloc_cg1_fn(tmem_dQ0, 64);
        tmem_dealloc_cg1_fn(tmem_dQ1, 64);
    }
}

__global__ void mha_bwd_kernel_2(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_dO,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_dK,
    const __grid_constant__ CUtensorMap tma_dV,
    const float* __restrict__ L, 
    const float* __restrict__ D,
    int S)
{
    int j_blk = blockIdx.x; 
    int bh = blockIdx.y;
    int tid = threadIdx.x; 
    int lane_id = tid % 32;
    int warp_id = tid / 32;
    int row = warp_id * 32 + lane_id;

    extern __shared__ __align__(1024) char smem[];
    __nv_bfloat16* s_Q0 = (__nv_bfloat16*)(smem + 0);         
    __nv_bfloat16* s_Q1 = (__nv_bfloat16*)(smem + 16384);     
    __nv_bfloat16* s_dO0 = (__nv_bfloat16*)(smem + 32768);    
    __nv_bfloat16* s_dO1 = (__nv_bfloat16*)(smem + 49152);    
    __nv_bfloat16* s_K0 = (__nv_bfloat16*)(smem + 65536);     
    __nv_bfloat16* s_K1 = (__nv_bfloat16*)(smem + 73728);     
    __nv_bfloat16* s_V0 = (__nv_bfloat16*)(smem + 81920);     
    __nv_bfloat16* s_V1 = (__nv_bfloat16*)(smem + 90112);     
    __nv_bfloat16* s_dS = (__nv_bfloat16*)(smem + 98304);     
    __nv_bfloat16* s_P  = (__nv_bfloat16*)(smem + 114688);    

    uint32_t tmem_S, tmem_dP, tmem_dK0, tmem_dK1, tmem_dV0, tmem_dV1;
    if (warp_id == 0) {
        tmem_alloc_cg1_fn(&tmem_S, 64);
        tmem_alloc_cg1_fn(&tmem_dP, 64);
        tmem_alloc_cg1_fn(&tmem_dK0, 64);
        tmem_alloc_cg1_fn(&tmem_dK1, 64);
        tmem_alloc_cg1_fn(&tmem_dV0, 64);
        tmem_alloc_cg1_fn(&tmem_dV1, 64);
    }
    __syncthreads();
    
    uint64_t* mbar = (uint64_t*)(smem + 132096); 
    uint64_t* mbar_umma = (uint64_t*)(smem + 132104);
    
    if (tid == 0) {
        init_smem_barrier_fn(mbar, 128);
        init_smem_barrier_fn(mbar_umma, 1);
    }
    __syncthreads();

    int phase = 0;
    int phase_umma = 0;

    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar, 32768); 
        tma_load_3d_fn(&tma_K, mbar, s_K0, 0, j_blk * 64, bh);
        tma_load_3d_fn(&tma_K, mbar, s_K1, 64, j_blk * 64, bh);
        tma_load_3d_fn(&tma_V, mbar, s_V0, 0, j_blk * 64, bh);
        tma_load_3d_fn(&tma_V, mbar, s_V1, 64, j_blk * 64, bh);
    } else {
        mbarrier_arrive_fn(mbar);
    }
    mbarrier_wait_fn(mbar, phase);
    phase ^= 1;
    
    int accum_dV_start = 0;
    
    int i_start = j_blk / 2;
    int i_end = (S + 127) / 128;
    for (int block_i = i_start; block_i < i_end; block_i++) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar, 65536); 
            tma_load_3d_fn(&tma_Q, mbar, s_Q0, 0, block_i * 128, bh);
            tma_load_3d_fn(&tma_Q, mbar, s_Q1, 64, block_i * 128, bh);
            tma_load_3d_fn(&tma_dO, mbar, s_dO0, 0, block_i * 128, bh);
            tma_load_3d_fn(&tma_dO, mbar, s_dO1, 64, block_i * 128, bh);
        } else {
            mbarrier_arrive_fn(mbar);
        }
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        
        uint32_t idesc_S = make_instr_desc_fn(128, 64, 0, 0);
        UMMA_LOOP_K64_KK(tmem_S, s_Q0, s_K0, idesc_S, 0);
        UMMA_LOOP_K64_KK(tmem_S, s_Q1, s_K1, idesc_S, 1);
        
        UMMA_LOOP_K64_KK(tmem_dP, s_dO0, s_V0, idesc_S, 0);
        UMMA_LOOP_K64_KK(tmem_dP, s_dO1, s_V1, idesc_S, 1);
        
        if (tid == 0) tcgen05_commit_cg1_fn(mbar_umma);
        mbarrier_wait_fn(mbar_umma, phase_umma);
        phase_umma ^= 1;
        
        float L_val = 0, D_val = 0;
        int global_row_ = bh * S + block_i * 128 + row;
        if (block_i * 128 + row < S) {
            L_val = L[global_row_];
            D_val = D[global_row_];
        }
        
        for (int col = 0; col < 64; col += 4) {
            uint32_t sr0, sr1, sr2, sr3, pr0, pr1, pr2, pr3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(sr0),"=r"(sr1),"=r"(sr2),"=r"(sr3) : "r"(tmem_S + col));
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(pr0),"=r"(pr1),"=r"(pr2),"=r"(pr3) : "r"(tmem_dP + col));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float s[4] = { __uint_as_float(sr0), __uint_as_float(sr1), __uint_as_float(sr2), __uint_as_float(sr3) };
            float dp[4] = { __uint_as_float(pr0), __uint_as_float(pr1), __uint_as_float(pr2), __uint_as_float(pr3) };
            
            float p_arr[4] = {0,0,0,0}, ds[4] = {0,0,0,0};
            for (int c = 0; c < 4; c++) {
                int global_col = j_blk * 64 + col + c;
                int gr = block_i * 128 + row;
                if (global_col <= gr && gr < S && global_col < S) {
                    p_arr[c] = expf(s[c] * 0.0883883476483f - L_val);
                    ds[c] = p_arr[c] * (dp[c] - D_val) * 0.0883883476483f;
                }
            }
            
            uint32_t packed_dS0 = pack_bf16(ds[0], ds[1]);
            uint32_t packed_dS1 = pack_bf16(ds[2], ds[3]);
            uint32_t packed_P0 = pack_bf16(p_arr[0], p_arr[1]);
            uint32_t packed_P1 = pack_bf16(p_arr[2], p_arr[3]);
            
            *(uint32_t*)((char*)s_dS + swizzle_128B(row, (col + 0) * 2)) = packed_dS0;
            *(uint32_t*)((char*)s_dS + swizzle_128B(row, (col + 2) * 2)) = packed_dS1;
            *(uint32_t*)((char*)s_P + swizzle_128B(row, (col + 0) * 2)) = packed_P0;
            *(uint32_t*)((char*)s_P + swizzle_128B(row, (col + 2) * 2)) = packed_P1;
        }
        
        fence_proxy_async_fn();
        __syncthreads();
        
        uint32_t idesc_dV = make_instr_desc_fn(64, 64, 1, 1);
        UMMA_LOOP_K128_MM(tmem_dV0, s_P, s_dO0, idesc_dV, accum_dV_start);
        UMMA_LOOP_K128_MM(tmem_dV1, s_P, s_dO1, idesc_dV, accum_dV_start);
        
        UMMA_LOOP_K128_MM(tmem_dK0, s_dS, s_Q0, idesc_dV, accum_dV_start);
        UMMA_LOOP_K128_MM(tmem_dK1, s_dS, s_Q1, idesc_dV, accum_dV_start);
        accum_dV_start = 1;
        
        if (tid == 0) tcgen05_commit_cg1_fn(mbar_umma);
        mbarrier_wait_fn(mbar_umma, phase_umma);
        phase_umma ^= 1;
    }
    
    for (int pass = 0; pass < 2; pass++) {
        uint32_t tmem_src = (pass == 0) ? tmem_dK0 : tmem_dK1;
        for (int col = 0; col < 64; col += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_src + col));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            if (row < 64) {
                uint32_t packed0 = pack_bf16(__uint_as_float(r0), __uint_as_float(r1));
                uint32_t packed1 = pack_bf16(__uint_as_float(r2), __uint_as_float(r3));
                *(uint32_t*)((char*)s_K0 + swizzle_128B(row, (col + 0) * 2)) = packed0;
                *(uint32_t*)((char*)s_K0 + swizzle_128B(row, (col + 2) * 2)) = packed1;
            }
        }
        fence_proxy_async_fn();
        __syncthreads();
        if (tid == 0) {
            tma_store_3d_fn(&tma_dK, s_K0, pass * 64, j_blk * 64, bh);
            tma_store_commit_fn();
        }
        tma_store_wait_fn<0>();
        __syncthreads();
    }
    
    for (int pass = 0; pass < 2; pass++) {
        uint32_t tmem_src = (pass == 0) ? tmem_dV0 : tmem_dV1;
        for (int col = 0; col < 64; col += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_src + col));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            if (row < 64) {
                uint32_t packed0 = pack_bf16(__uint_as_float(r0), __uint_as_float(r1));
                uint32_t packed1 = pack_bf16(__uint_as_float(r2), __uint_as_float(r3));
                *(uint32_t*)((char*)s_V0 + swizzle_128B(row, (col + 0) * 2)) = packed0;
                *(uint32_t*)((char*)s_V0 + swizzle_128B(row, (col + 2) * 2)) = packed1;
            }
        }
        fence_proxy_async_fn();
        __syncthreads();
        if (tid == 0) {
            tma_store_3d_fn(&tma_dV, s_V0, pass * 64, j_blk * 64, bh);
            tma_store_commit_fn();
        }
        tma_store_wait_fn<0>();
        __syncthreads();
    }
    
    if (warp_id == 0) {
        tmem_dealloc_cg1_fn(tmem_S, 64);
        tmem_dealloc_cg1_fn(tmem_dP, 64);
        tmem_dealloc_cg1_fn(tmem_dK0, 64);
        tmem_dealloc_cg1_fn(tmem_dK1, 64);
        tmem_dealloc_cg1_fn(tmem_dV0, 64);
        tmem_dealloc_cg1_fn(tmem_dV1, 64);
    }
}

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, 
    uint64_t d0, uint64_t d1, uint64_t d2,
    uint32_t b0, uint32_t b1, uint32_t b2) 
{
    cuuint64_t globalDim[3] = {d0, d1, d2};
    cuuint64_t globalStrides[2] = {d0 * 2, d0 * d1 * 2}; 
    cuuint32_t boxDim[3] = {b0, b1, b2};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, globalAddress,
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) 
{
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int d = Q.size(3);

    const __nv_bfloat16* q_ptr  = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* k_ptr  = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* v_ptr  = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* o_ptr  = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* do_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* l_ptr = static_cast<const float*>(L.data_ptr());

    __nv_bfloat16* dq_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dk_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dv_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());

    float* d_ptr;
    CUDA_CHECK(cudaMallocAsync(&d_ptr, B * H * S * sizeof(float), stream));

    dim3 grid_D((B * H * S + 3) / 4);
    dim3 block_D(32, 4);
    precompute_D_kernel<<<grid_D, block_D, 0, stream>>>(o_ptr, do_ptr, d_ptr, B * H * S);
    CUDA_CHECK(cudaGetLastError());

    CUtensorMap tma_Q, tma_dO, tma_dQ, tma_K, tma_V, tma_dK, tma_dV;
    create_tma_3d_descriptor_2B(&tma_Q, (void*)q_ptr, d, S, B*H, 64, 128, 1);
    create_tma_3d_descriptor_2B(&tma_dO, (void*)do_ptr, d, S, B*H, 64, 128, 1);
    create_tma_3d_descriptor_2B(&tma_dQ, (void*)dq_ptr, d, S, B*H, 64, 128, 1);
    create_tma_3d_descriptor_2B(&tma_K, (void*)k_ptr, d, S, B*H, 64, 64, 1);
    create_tma_3d_descriptor_2B(&tma_V, (void*)v_ptr, d, S, B*H, 64, 64, 1);
    create_tma_3d_descriptor_2B(&tma_dK, (void*)dk_ptr, d, S, B*H, 64, 64, 1);
    create_tma_3d_descriptor_2B(&tma_dV, (void*)dv_ptr, d, S, B*H, 64, 64, 1);

    dim3 threads(128);
    dim3 blocks_k1((S + 127) / 128, B * H);
    int smem_size = 132112; 
    
    CUDA_CHECK(cudaFuncSetAttribute(mha_bwd_kernel_1, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    mha_bwd_kernel_1<<<blocks_k1, threads, smem_size, stream>>>(
        tma_Q, tma_dO, tma_K, tma_V, tma_dQ, l_ptr, d_ptr, S
    );
    CUDA_CHECK(cudaGetLastError());
    
    dim3 blocks_k2((S + 63) / 64, B * H);
    CUDA_CHECK(cudaFuncSetAttribute(mha_bwd_kernel_2, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    mha_bwd_kernel_2<<<blocks_k2, threads, smem_size, stream>>>(
        tma_Q, tma_dO, tma_K, tma_V, tma_dK, tma_dV, l_ptr, d_ptr, S
    );
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaFreeAsync(d_ptr, stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda