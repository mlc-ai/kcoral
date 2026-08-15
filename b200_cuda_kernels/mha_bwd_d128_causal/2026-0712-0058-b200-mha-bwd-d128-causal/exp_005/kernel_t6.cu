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
#include <algorithm>
#include <vector>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_fa4 {

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void fence_mbarrier_init_fn() {
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

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ uint32_t read_tmem_32bit(uint32_t tmem_ptr) {
    uint32_t val;
    asm volatile("tcgen05.ld.sync.aligned.16x64b.x1.b32 %0, [%1];" : "=r"(val) : "r"(tmem_ptr));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    return val;
}

__device__ __forceinline__ void write_tmem_32bit(uint32_t tmem_ptr, uint32_t val) {
    asm volatile("tcgen05.st.sync.aligned.16x64b.x1.b32 [%0], %1;" :: "r"(tmem_ptr), "r"(val));
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_k_major_128b(void* smem_ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((1 & 0x3FFFF) >> 4) << 16; 
    d |= (uint64_t)((1024 & 0x3FFFF) >> 4) << 32; 
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (0u << 15);   
    d |= (0u << 16);   
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ void tmem_wait_fn(uint64_t* bar, uint32_t phase) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared.b64 [%0];" :: "r"(a));
    mbarrier_wait_fn(bar, phase);
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
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

__device__ void gemm_S(char* s_K0, char* s_K1, char* s_Q0, char* s_Q1, uint32_t tmem_S_ptr) {
    uint32_t idesc = make_instr_desc_fn(128, 128);
    for(int k = 0; k < 4; ++k) {
        uint64_t desc_a = make_smem_desc_k_major_128b(s_K0 + k * 32);
        uint64_t desc_b = make_smem_desc_k_major_128b(s_Q0 + k * 32);
        uint32_t accum_i = (k == 0) ? 0 : 1;
        umma_f16_cg1_fn(tmem_S_ptr, desc_a, desc_b, idesc, accum_i);
    }
    for(int k = 0; k < 4; ++k) {
        uint64_t desc_a = make_smem_desc_k_major_128b(s_K1 + k * 32);
        uint64_t desc_b = make_smem_desc_k_major_128b(s_Q1 + k * 32);
        uint32_t accum_i = 1;
        umma_f16_cg1_fn(tmem_S_ptr, desc_a, desc_b, idesc, accum_i);
    }
}

__device__ void gemm_dP(char* s_V0, char* s_V1, char* s_dOT0, char* s_dOT1, uint32_t tmem_dP_ptr) {
    uint32_t idesc = make_instr_desc_fn(128, 128);
    for(int k = 0; k < 4; ++k) {
        uint64_t desc_a = make_smem_desc_k_major_128b(s_V0 + k * 32);
        uint64_t desc_b = make_smem_desc_k_major_128b(s_dOT0 + k * 32);
        uint32_t accum_i = (k == 0) ? 0 : 1;
        umma_f16_cg1_fn(tmem_dP_ptr, desc_a, desc_b, idesc, accum_i);
    }
    for(int k = 0; k < 4; ++k) {
        uint64_t desc_a = make_smem_desc_k_major_128b(s_V1 + k * 32);
        uint64_t desc_b = make_smem_desc_k_major_128b(s_dOT1 + k * 32);
        uint32_t accum_i = 1;
        umma_f16_cg1_fn(tmem_dP_ptr, desc_a, desc_b, idesc, accum_i);
    }
}

__device__ void read_tmem_to_smem(char* s_dest, uint32_t tmem_ptr) {
    int tid = threadIdx.x;
    for (int col = 0; col < 64; col++) {
        float val = __uint_as_float(read_tmem_32bit(tmem_ptr + (tid << 16) + col));
        write_swizzled(s_dest, tid, col, val);
    }
}

__device__ void apply_softmax_local(char* s_PT0, char* s_PT1, uint32_t tmem_S_ptr, const float* L_data, int bh, int q_start, int kv_start, float scale, int S_val) {
    int tid = threadIdx.x;
    const float* L_bh = L_data + bh * S_val + q_start + tid;
    float lse = L_bh[0];
    
    float s0 = __uint_as_float(read_tmem_32bit(tmem_S_ptr + (tid << 16)));
    float s1 = __uint_as_float(read_tmem_32bit(tmem_S_ptr + (tid << 16) + 64));
    
    float p0 = (kv_start <= q_start + tid && q_start + tid < S_val) ? fast_exp2f_fn((s0 * scale - lse) * 1.44269504f) : 0.0f;
    float p1 = (kv_start + 64 <= q_start + tid && q_start + tid < S_val) ? fast_exp2f_fn((s1 * scale - lse) * 1.44269504f) : 0.0f;
    
    write_swizzled(s_PT0, tid, 0, p0);
    write_swizzled(s_PT1, tid, 0, p1);
}

__device__ void compute_dS_local(char* s_dS0, char* s_PT0, char* s_dPT0, float* s_DT, int q_start, int kv_start, int S_val) {
    int tid = threadIdx.x;
    float p = __bfloat162float(read_swizzled(s_PT0, tid, 0));
    float dp = __bfloat162float(read_swizzled(s_dPT0, tid, 0));
    float ds = p * (dp - s_DT[tid]);
    write_swizzled(s_dS0, tid, 0, ds);
}

__device__ __forceinline__ __nv_bfloat16 read_swizzled(const char* smem, int row, int col) {
    int col_bytes = col * 2;
    int chunk = col_bytes / 16;
    int chunk_swizzled = chunk ^ (row % 8);
    int final_col_bytes = (chunk_swizzled * 16) + (col_bytes % 16);
    int idx = row * 128 + final_col_bytes;
    return *( (__nv_bfloat16*) (&smem[idx]) );
}

__device__ __forceinline__ void write_swizzled(char* smem, int row, int col, float val) {
    int col_bytes = col * 2;
    int chunk = col_bytes / 16;
    int chunk_swizzled = chunk ^ (row % 8);
    int final_col_bytes = (chunk_swizzled * 16) + (col_bytes % 16);
    int idx = row * 128 + final_col_bytes;
    __nv_bfloat16 a = __float2bfloat16(val);
    *( (__nv_bfloat16*) (&smem[idx]) ) = a;
}

__device__ void transpose_64x64_swizzled(const char* src, char* dst) {
    for (int i = threadIdx.x; i < 4096; i += blockDim.x) {
        int row = i / 64;
        int col = i % 64;
        
        int col_bytes = col * 2;
        int chunk = col_bytes / 16;
        int chunk_swizzled = chunk ^ (row % 8);
        int final_col_bytes = (chunk_swizzled * 16) + (col_bytes % 16);
        int src_idx = row * 128 + final_col_bytes;
        
        int row_bytes = row * 2;
        int dst_chunk = row_bytes / 16;
        int dst_chunk_swizzled = dst_chunk ^ (col % 8);
        int dst_final_bytes = (dst_chunk_swizzled * 16) + (row_bytes % 16);
        int dst_idx = col * 128 + dst_final_bytes;
        
        *(uint16_t*)&dst[dst_idx] = *(uint16_t*)&src[src_idx];
    }
}

__device__ void gemm_64x64_x64(char* s_A, char* s_B, char* smem_out, int accum) {
    int tid = threadIdx.x;
    for (int col = 0; col < 64; col++) {
        float sum = accum ? __bfloat162float(read_swizzled(smem_out, tid, col)) : 0;
        for (int k = 0; k < 64; k++) {
            sum += __bfloat162float(read_swizzled(s_A, tid, k)) * __bfloat162float(read_swizzled(s_B, col, k));
        }
        write_swizzled(smem_out, tid, col, sum);
    }
}

__device__ void gemm_atomic_add(__nv_bfloat16* global_D, char* smem_A, char* smem_B, int64_t row_base, int S_val) {
    int tid = threadIdx.x;
    for (int col = 0; col < 64; col++) {
        float sum = 0;
        for (int k = 0; k < 64; k++) {
            sum += __bfloat162float(read_swizzled(smem_A, tid, k)) * __bfloat162float(read_swizzled(smem_B, k, col));
        }
        int g_row = row_base + tid;
        if (g_row < S_val) {
            atomicAdd(&global_D[(uint64_t)g_row * 128 + col], __float2bfloat16(sum));
            atomicAdd(&global_D[(uint64_t)g_row * 128 + 64 + col], __float2bfloat16(sum));
        }
    }
}

__device__ void store_gemm_atomic_add(__nv_bfloat16* global_D, char* smem_A0, char* smem_A1, char* smem_B0, char* smem_B1, int64_t row_base, int S_val) {
    int tid = threadIdx.x;
    for (int col = 0; col < 64; col++) {
        float sum0 = 0;
        for (int k = 0; k < 64; k++) {
            sum0 += __bfloat162float(read_swizzled(smem_A0, tid, k)) * __bfloat162float(read_swizzled(smem_B0, k, col));
        }
        int g_row = row_base + tid;
        if (g_row < S_val) {
            atomicAdd(&global_D[(uint64_t)g_row * 128 + col], __float2bfloat16(sum0));
        }
    }
    for (int col = 0; col < 64; col++) {
        float sum1 = 0;
        for (int k = 0; k < 64; k++) {
            sum1 += __bfloat162float(read_swizzled(smem_A1, tid, k)) * __bfloat162float(read_swizzled(smem_B1, k, col));
        }
        int g_row = row_base + tid;
        if (g_row < S_val) {
            atomicAdd(&global_D[(uint64_t)g_row * 128 + 64 + col], __float2bfloat16(sum1));
        }
    }
}

__device__ void store_gemm(__nv_bfloat16* global_D, char* smem_0, char* smem_1, int64_t row_base, int S_val) {
    int tid = threadIdx.x;
    for (int col = 0; col < 64; col++) {
        int g_row = row_base + tid;
        if (g_row < S_val) {
            global_D[(uint64_t)g_row * 128 + col] = read_swizzled(smem_0, tid, col);
            global_D[(uint64_t)g_row * 128 + 64 + col] = read_swizzled(smem_1, tid, col);
        }
    }
}

__device__ void compute_D_local(char* s_O0, char* s_O1, char* s_dO0, char* s_dO1, float* s_DT) {
    int tid = threadIdx.x;
    float sum = 0;
    for(int k=0; k<64; ++k) {
        sum += __bfloat162float(read_swizzled(s_O0, tid, k)) * __bfloat162float(read_swizzled(s_dO0, tid, k));
        sum += __bfloat162float(read_swizzled(s_O1, tid, k)) * __bfloat162float(read_swizzled(s_dO1, tid, k));
    }
    s_DT[tid] = sum;
}

struct SharedStorage {
    __align__(1024) uint32_t pad_to_1024;
    __align__(1024) char s_K0[8192];
    __align__(1024) char s_K1[8192];
    __align__(1024) char s_V0[8192];
    __align__(1024) char s_V1[8192];
    
    __align__(1024) char s_Q0[8192];
    __align__(1024) char s_Q1[8192];
    __align__(1024) char s_O0[8192];
    __align__(1024) char s_O1[8192];
    __align__(1024) char s_dO0[8192];
    __align__(1024) char s_dO1[8192];
    
    __align__(1024) char s_PT0[8192];
    __align__(1024) char s_PT1[8192];
    __align__(1024) char s_dPT0[8192];
    __align__(1024) char s_dPT1[8192];
    __align__(1024) char s_dS0[8192];
    __align__(1024) char s_dS1[8192];
    
    __align__(1024) char s_dV0[8192];
    __align__(1024) char s_dV1[8192];
    __align__(1024) char s_dP0[8192];
    __align__(1024) char s_dP1[8192];
    
    __align__(1024) char s_dOT0[8192];
    __align__(1024) char s_dOT1[8192];
    __align__(1024) char s_Q0_T[8192];
    __align__(1024) char s_Q1_T[8192];
    __align__(1024) char s_K0_T[8192];
    __align__(1024) char s_K1_T[8192];
    __align__(1024) char s_dST0[8192];
    __align__(1024) char s_dST1[8192];
    
    __align__(1024) char s_dQ0[8192];
    __align__(1024) char s_dQ1[8192];
    
    __align__(1024) float s_DT[128];
    __align__(1024) uint64_t bar[1];
};

__global__ void bwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* __restrict__ L_data,
    __nv_bfloat16* __restrict__ dQ_bh,
    __nv_bfloat16* __restrict__ dK_bh,
    __nv_bfloat16* __restrict__ dV_bh,
    int64_t S_val, float scale)
{
    int bh = blockIdx.y;
    int kv_start = blockIdx.x * 64;
    
    extern __shared__ char smem_buf[];
    SharedStorage* smem = (SharedStorage*)smem_buf;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(smem->bar, 1);
    }
    fence_mbarrier_init_fn();
    __syncthreads();
    
    uint32_t tmem_base_cta0;
    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_base_cta0, 512);
    }
    __syncthreads();
    
    uint32_t tmem_S_ptr = tmem_base_cta0;
    uint32_t tmem_dP_ptr = tmem_base_cta0 + 8192;
    uint32_t tmem_dQ_ptr = tmem_base_cta0 + 16384;
    uint32_t tmem_dK_ptr = tmem_base_cta0 + 24576;
    
    uint32_t phase = 0;
    int tid = threadIdx.x;

    char* s_dKT0 = smem->s_dV0;
    char* s_dKT1 = smem->s_dV1;

    if (kv_start < S_val) {
        mbarrier_arrive_and_expect_tx_fn(smem->bar, 8192 * 4);
        tma_load_2d_fn(&tma_K, smem->bar, smem->s_K0, 0, bh * S_val + kv_start);
        tma_load_2d_fn(&tma_K, smem->bar, smem->s_K1, 64, bh * S_val + kv_start);

        tma_load_2d_fn(&tma_V, smem->bar, smem->s_V0, 0, bh * S_val + kv_start);
        tma_load_2d_fn(&tma_V, smem->bar, smem->s_V1, 64, bh * S_val + kv_start);
        
        mbarrier_wait_fn(smem->bar, phase);
        fence_proxy_async_fn();
        phase ^= 1;
    }
    
    for (int i = tid; i < 8192; i += blockDim.x) {
        ((char*)smem->s_dV0)[i] = 0;
        ((char*)smem->s_dV1)[i] = 0;
    }
    __syncthreads();

    // ==== Phase 1: Accumulate dV ====
    for (int q_start = 0; q_start <= kv_start; q_start += 64) {
        if (q_start < S_val) {
            mbarrier_arrive_and_expect_tx_fn(smem->bar, 8192 * 6);
            tma_load_2d_fn(&tma_Q, smem->bar, smem->s_Q0, 0, bh * S_val + q_start);
            tma_load_2d_fn(&tma_Q, smem->bar, smem->s_Q1, 64, bh * S_val + q_start);

            tma_load_2d_fn(&tma_dO, smem->bar, smem->s_dO0, 0, bh * S_val + q_start);
            tma_load_2d_fn(&tma_dO, smem->bar, smem->s_dO1, 64, bh * S_val + q_start);

            tma_load_2d_fn(&tma_O, smem->bar, smem->s_O0, 0, bh * S_val + q_start);
            tma_load_2d_fn(&tma_O, smem->bar, smem->s_O1, 64, bh * S_val + q_start);
            
            mbarrier_wait_fn(smem->bar, phase);
            fence_proxy_async_fn();
            phase ^= 1;
        }
        __syncthreads();

        if (q_start < S_val) {
            compute_D_local(smem->s_O0, smem->s_O1, smem->s_dO0, smem->s_dO1, smem->s_DT);
            
            gemm_S(smem->s_K0, smem->s_K1, smem->s_Q0, smem->s_Q1, tmem_S_ptr);
            tmem_wait_fn(smem->bar, phase);
            phase ^= 1;
        }
        __syncthreads();

        if (q_start < S_val) {
            apply_softmax_local(smem->s_PT0, smem->s_PT1, tmem_S_ptr, L_data, bh, q_start, kv_start, scale, S_val);
            
            transpose_64x64_swizzled(smem->s_dO0, smem->s_dOT0);
            transpose_64x64_swizzled(smem->s_dO1, smem->s_dOT1);
            
            gemm_dP(smem->s_V0, smem->s_V1, smem->s_dOT0, smem->s_dOT1, tmem_dP_ptr);
            tmem_wait_fn(smem->bar, phase);
            phase ^= 1;
        }
        __syncthreads();

        if (q_start < S_val) {
            read_tmem_to_smem(smem->s_dPT0, tmem_dP_ptr);
            read_tmem_to_smem(smem->s_dPT1, tmem_dP_ptr + 8192);
            
            compute_dS_local(smem->s_dS0, smem->s_PT0, smem->s_dPT0, smem->s_DT, q_start, kv_start, S_val);
            compute_dS_local(smem->s_dS1, smem->s_PT1, smem->s_dPT1, smem->s_DT, q_start, kv_start, S_val);
            
            // Transpose dPT to dP effectively  
            transpose_64x64_swizzled(smem->s_dPT0, smem->s_dP0);
            transpose_64x64_swizzled(smem->s_dPT1, smem->s_dP1);

            gemm_64x64_x64(smem->s_PT0, smem->s_dP0, smem->s_dV0, 1); 
            gemm_64x64_x64(smem->s_PT1, smem->s_dP1, smem->s_dV1, 1);
        }
        __syncthreads();
    }
    
    // ==== Phase 2: Accumulate dK and calculate dQ ====
    for (int i = tid; i < 8192; i += blockDim.x) {
        ((char*)s_dKT0)[i] = 0;
        ((char*)s_dKT1)[i] = 0;
    }
    __syncthreads();

    for (int q_start = kv_start; q_start < S_val; q_start += 64) {
        if (q_start < S_val) {
            mbarrier_arrive_and_expect_tx_fn(smem->bar, 8192 * 6);
            tma_load_2d_fn(&tma_Q, smem->bar, smem->s_Q0, 0, bh * S_val + q_start);
            tma_load_2d_fn(&tma_Q, smem->bar, smem->s_Q1, 64, bh * S_val + q_start);

            tma_load_2d_fn(&tma_dO, smem->bar, smem->s_dO0, 0, bh * S_val + q_start);
            tma_load_2d_fn(&tma_dO, smem->bar, smem->s_dO1, 64, bh * S_val + q_start);

            tma_load_2d_fn(&tma_O, smem->bar, smem->s_O0, 0, bh * S_val + q_start);
            tma_load_2d_fn(&tma_O, smem->bar, smem->s_O1, 64, bh * S_val + q_start);
            
            mbarrier_wait_fn(smem->bar, phase);
            fence_proxy_async_fn();
            phase ^= 1;
        }
        __syncthreads();

        if (q_start < S_val) {
            compute_D_local(smem->s_O0, smem->s_O1, smem->s_dO0, smem->s_dO1, smem->s_DT);
            
            gemm_S(smem->s_K0, smem->s_K1, smem->s_Q0, smem->s_Q1, tmem_S_ptr);
            tmem_wait_fn(smem->bar, phase);
            phase ^= 1;
        }
        __syncthreads();

        if (q_start < S_val) {
            apply_softmax_local(smem->s_PT0, smem->s_PT1, tmem_S_ptr, L_data, bh, q_start, kv_start, scale, S_val);
            
            transpose_64x64_swizzled(smem->s_dO0, smem->s_dOT0);
            transpose_64x64_swizzled(smem->s_dO1, smem->s_dOT1);
            
            gemm_dP(smem->s_V0, smem->s_V1, smem->s_dOT0, smem->s_dOT1, tmem_dP_ptr);
            tmem_wait_fn(smem->bar, phase);
            phase ^= 1;
        }
        __syncthreads();

        if (q_start < S_val) {
            read_tmem_to_smem(smem->s_dPT0, tmem_dP_ptr);
            read_tmem_to_smem(smem->s_dPT1, tmem_dP_ptr + 8192);
            
            compute_dS_local(smem->s_dS0, smem->s_PT0, smem->s_dPT0, smem->s_DT, q_start, kv_start, S_val);
            compute_dS_local(smem->s_dS1, smem->s_PT1, smem->s_dPT1, smem->s_DT, q_start, kv_start, S_val);
            
            // Transpose Q to Q_T effectively
            transpose_64x64_swizzled(smem->s_Q0, smem->s_Q0_T);
            transpose_64x64_swizzled(smem->s_Q1, smem->s_Q1_T);
            
            gemm_64x64_x64(smem->s_dS0, smem->s_Q0_T, s_dKT0, 1);
            gemm_64x64_x64(smem->s_dS1, smem->s_Q1_T, s_dKT1, 1);
            
            // Transpose dST effectively
            transpose_64x64_swizzled(smem->s_dS0, smem->s_dST0);
            transpose_64x64_swizzled(smem->s_dS1, smem->s_dST1);
            
            // Transpose K to K_T effectively
            transpose_64x64_swizzled(smem->s_K0, smem->s_K0_T);
            transpose_64x64_swizzled(smem->s_K1, smem->s_K1_T);
            
            gemm_64x64_x64(smem->s_dST0, smem->s_K0_T, smem->s_dQ0, 0);
            gemm_64x64_x64(smem->s_dST1, smem->s_K1_T, smem->s_dQ1, 0);
            
            store_gemm_atomic_add(dQ_bh + bh * S_val * 128, smem->s_dQ0, smem->s_dQ1, smem->s_K0_T, smem->s_K1_T, q_start, S_val);
        }
        __syncthreads();
    }
    
    store_gemm(dV_bh + bh * S_val * 128, smem->s_dV0, smem->s_dV1, kv_start, S_val);
    store_gemm(dK_bh + bh * S_val * 128, s_dKT0, s_dKT1, kv_start, S_val);

    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_base_cta0, 512);
    }
}

CUresult create_tma_2d_descriptor_2B(
    CUtensorMap* d, void* globalAddress, 
    uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, 
    uint32_t smem_inner_dim, uint32_t smem_outer_dim, 
    CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) 
{
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

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L, 
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
         
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S_val = Q.size(2);
    int64_t d = Q.size(3);
    
    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_data = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_data = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_data = static_cast<const float*>(L.data_ptr());
    
    __nv_bfloat16* dQ_data = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_data = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_data = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    float scale = 1.0f / sqrtf((float)d);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUDA_CHECK(cudaMemsetAsync(dQ_data, 0, B * H * S_val * d * sizeof(__nv_bfloat16), stream));
    CUDA_CHECK(cudaMemsetAsync(dK_data, 0, B * H * S_val * d * sizeof(__nv_bfloat16), stream));
    CUDA_CHECK(cudaMemsetAsync(dV_data, 0, B * H * S_val * d * sizeof(__nv_bfloat16), stream));
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O, tma_dO;
    create_tma_2d_descriptor_2B(&tma_Q, (void*)Q_data, d, B * H * S_val, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_K, (void*)K_data, d, B * H * S_val, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_V, (void*)V_data, d, B * H * S_val, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_O, (void*)O_data, d, B * H * S_val, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_dO, (void*)dO_data, d, B * H * S_val, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    int64_t threads = 128;
    dim3 grid((S_val + 63) / 64, B * H);
    
    CUDA_CHECK(cudaFuncSetAttribute(bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedStorage)));
    
    bwd_kernel<<<grid, threads, sizeof(SharedStorage), stream>>>(
        tma_Q, tma_K, tma_V, tma_O, tma_dO, L_data, dQ_data, dK_data, dV_data, S_val, scale);
        
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_fa4::run);

} // namespace tvm_ffi_fa4