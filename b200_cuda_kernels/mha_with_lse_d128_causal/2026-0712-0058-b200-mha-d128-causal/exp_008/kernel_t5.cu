#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CU_CHECK(call) do { \
    CUresult _e = (call); \
    if (_e != CUDA_SUCCESS) { \
        fprintf(stderr, "CUDA Driver error %d at %s:%d\n", (int)_e, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

__device__ __forceinline__ uint32_t precompute_u32_smem(void* ptr) {
    return (uint32_t)__cvta_generic_to_shared(ptr);
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(precompute_u32_smem(bar)), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(precompute_u32_smem(bar)), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n.reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n}\n" :: "r"(precompute_u32_smem(bar)), "r"(phase));
}

__device__ __forceinline__ void tma_load_4d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
        :: "r"(precompute_u32_smem(smem)), "l"((uint64_t)d), "r"(precompute_u32_smem(bar)), "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}

__device__ __forceinline__ uint32_t tmem_alloc_fn(int ncols) {
    uint32_t addr;
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(precompute_u32_smem(&addr)), "r"(ncols));
    return addr;
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tmem_load_8x_fn(uint32_t tmem_addr,
    uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3,
    uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3), "=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7) : "r"(tmem_addr));
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

__device__ __forceinline__ void umma_commit_1sm_fn(uint64_t* bar) {
    asm volatile(
        "tcgen05.commit.cta_group::1"
        ".mbarrier::arrive::one.shared::cta.b64"
        " [%0];"
        :: "r"(precompute_u32_smem(bar)));
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = precompute_u32_smem(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, uint32_t a_major, uint32_t b_major) {
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

__device__ __forceinline__ uint64_t smem_desc_Q0() {
    extern __shared__ __align__(1024) char smem_pool[];
    __nv_bfloat16* smem_Q0 = (__nv_bfloat16*)smem_pool;
    return make_smem_desc_sm100_fn(smem_Q0, 1, 1024);
}
__device__ __forceinline__ uint64_t smem_desc_Q1() {
    extern __shared__ __align__(1024) char smem_pool[];
    __nv_bfloat16* smem_Q1 = (__nv_bfloat16*)(smem_pool + 8192);
    return make_smem_desc_sm100_fn(smem_Q1, 1, 1024);
}

__device__ __forceinline__ uint64_t smem_desc_K0(int idx) {
    extern __shared__ __align__(1024) char smem_pool[];
    __nv_bfloat16* smem_K0[2] = {(__nv_bfloat16*)(smem_pool + 16384), (__nv_bfloat16*)(smem_pool + 24576)};
    return make_smem_desc_sm100_fn(smem_K0[idx], 1, 1024);
}
__device__ __forceinline__ uint64_t smem_desc_K1(int idx) {
    extern __shared__ __align__(1024) char smem_pool[];
    __nv_bfloat16* smem_K1[2] = {(__nv_bfloat16*)(smem_pool + 32768), (__nv_bfloat16*)(smem_pool + 40960)};
    return make_smem_desc_sm100_fn(smem_K1[idx], 1, 1024);
}

__device__ __forceinline__ uint64_t smem_desc_V0(int idx) {
    extern __shared__ __align__(1024) char smem_pool[];
    __nv_bfloat16* smem_V0[2] = {(__nv_bfloat16*)(smem_pool + 49152), (__nv_bfloat16*)(smem_pool + 57344)};
    return make_smem_desc_sm100_fn(smem_V0[idx], 8192, 1024);
}
__device__ __forceinline__ uint64_t smem_desc_V1(int idx) {
    extern __shared__ __align__(1024) char smem_pool[];
    __nv_bfloat16* smem_V1[2] = {(__nv_bfloat16*)(smem_pool + 65536), (__nv_bfloat16*)(smem_pool + 73728)};
    return make_smem_desc_sm100_fn(smem_V1[idx], 8192, 1024);
}

__device__ __forceinline__ uint64_t smem_desc_P() {
    extern __shared__ __align__(1024) char smem_pool[];
    __nv_bfloat16* smem_P = (__nv_bfloat16*)(smem_pool + 81920);
    return make_smem_desc_sm100_fn(smem_P, 1, 1024);
}

__global__ void attention_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S, float scale) 
{
    int b_h = blockIdx.y;
    int q_block = blockIdx.x;
    int num_q_blocks = (S + 63) / 64;
    
    if (q_block >= num_q_blocks) return;
    
    extern __shared__ __align__(1024) char smem_pool[];
    __nv_bfloat16* smem_Q0 = (__nv_bfloat16*)smem_pool;                
    __nv_bfloat16* smem_Q1 = (__nv_bfloat16*)(smem_pool + 8192);       
    __nv_bfloat16* smem_K0[2] = {(__nv_bfloat16*)(smem_pool + 16384), (__nv_bfloat16*)(smem_pool + 24576)};       
    __nv_bfloat16* smem_K1[2] = {(__nv_bfloat16*)(smem_pool + 32768), (__nv_bfloat16*)(smem_pool + 40960)};
    __nv_bfloat16* smem_V0[2] = {(__nv_bfloat16*)(smem_pool + 49152), (__nv_bfloat16*)(smem_pool + 57344)};     
    __nv_bfloat16* smem_V1[2] = {(__nv_bfloat16*)(smem_pool + 65536), (__nv_bfloat16*)(smem_pool + 73728)};
    __nv_bfloat16* smem_P = (__nv_bfloat16*)(smem_pool + 81920);
    __nv_bfloat16* smem_O0 = (__nv_bfloat16*)(smem_pool + 90112);
    __nv_bfloat16* smem_O1 = (__nv_bfloat16*)(smem_pool + 98304);
    
    __shared__ __align__(128) uint64_t mbar_Q[1];
    __shared__ __align__(128) uint64_t mbar_K[2];
    __shared__ __align__(128) uint64_t mbar_V0[2];
    __shared__ __align__(128) uint64_t mbar_V1[2];
    
    int phase_Q[1] = {0};
    int phase_K[2] = {0, 0};
    int phase_V0[2] = {0, 0};
    int phase_V1[2] = {0, 0};
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&mbar_Q[0], 1);
        init_smem_barrier_fn(&mbar_K[0], 1);
        init_smem_barrier_fn(&mbar_K[1], 1);
        init_smem_barrier_fn(&mbar_V0[0], 1);
        init_smem_barrier_fn(&mbar_V0[1], 1);
        init_smem_barrier_fn(&mbar_V1[0], 1);
        init_smem_barrier_fn(&mbar_V1[1], 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    
    int B_H_S = B * H * S;
    int row_offset = b_h * S + q_block * 64;
    int b_idx = b_h / H;
    int h_idx = b_h % H;
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&mbar_Q[0], 16384);
        tma_load_4d_fn(&tma_Q, &mbar_Q[0], smem_Q0, 0, row_offset, h_idx, b_idx);
        tma_load_4d_fn(&tma_Q, &mbar_Q[0], (char*)smem_Q0 + 8192, 64, row_offset, h_idx, b_idx);
        
        mbarrier_arrive_and_expect_tx_fn(&mbar_K[0], 16384);
        tma_load_4d_fn(&tma_K, &mbar_K[0], smem_K0[0], 0, b_h * S, h_idx, b_idx);
        tma_load_4d_fn(&tma_K, &mbar_K[0], (char*)smem_K0[0] + 8192, 64, b_h * S, h_idx, b_idx);
        tma_load_4d_fn(&tma_K, &mbar_K[0], smem_K1[0], 64, b_h * S, h_idx, b_idx);
        tma_load_4d_fn(&tma_K, &mbar_K[0], (char*)smem_K1[0] + 8192, 128, b_h * S, h_idx, b_idx);
        
        mbarrier_arrive_and_expect_tx_fn(&mbar_V0[0], 16384);
        tma_load_4d_fn(&tma_V, &mbar_V0[0], smem_V0[0], 0, b_h * S, h_idx, b_idx);
        tma_load_4d_fn(&tma_V, &mbar_V0[0], (char*)smem_V0[0] + 8192, 64, b_h * S, h_idx, b_idx);
        
        mbarrier_arrive_and_expect_tx_fn(&mbar_V1[0], 16384);
        tma_load_4d_fn(&tma_V, &mbar_V1[0], smem_V1[0], 64, b_h * S, h_idx, b_idx);
        tma_load_4d_fn(&tma_V, &mbar_V1[0], (char*)smem_V1[0] + 8192, 128, b_h * S, h_idx, b_idx);
    }
    
    int warp_id = threadIdx.x / 32;
    int tid = threadIdx.x % 32;
    uint32_t my_m_base = tid / 2;
    uint32_t my_lane_id = my_m_base;
    uint32_t my_lane_offset = ((my_lane_id + warp_id * 16) << 8);
    
    float my_global_max0 = -INFINITY;
    float my_global_sum0 = 0.0f;
    
    uint32_t tmem_S, tmem_O0, tmem_O1, tmem_P_base;
    uint32_t tmem_Q0_base, tmem_Q1_base;
    uint32_t tmem_K0_base, tmem_K1_base;
    uint32_t tmem_V0_base, tmem_V1_base;
    
    if (threadIdx.x == 0) {
        tmem_S = tmem_alloc_fn(64);
        tmem_O0 = tmem_alloc_fn(64);
        tmem_O1 = tmem_alloc_fn(64);
        tmem_P_base = tmem_S; 
        
        tmem_Q0_base = tmem_O0;
        tmem_Q1_base = tmem_O1;
        tmem_K0_base = tmem_V0_base = tmem_alloc_fn(64);
        tmem_K1_base = tmem_V1_base = tmem_alloc_fn(64);
    }
    __syncthreads();

    mbarrier_wait_fn(&mbar_Q[0], phase_Q[0]);
    phase_Q[0] ^= 1;
    
    if (threadIdx.x == 0) {
        asm volatile("tcgen05.cp.cta_group::1.64x128b [%0], [%1];" :: "r"(tmem_Q0_base), "r"(precompute_u32_smem(smem_Q0)) : "memory");
        asm volatile("tcgen05.cp.cta_group::1.64x128b [%0], [%1];" :: "r"(tmem_Q0_base + 32), "r"(precompute_u32_smem((char*)smem_Q0 + 4096)) : "memory");
        
        asm volatile("tcgen05.cp.cta_group::1.64x128b [%0], [%1];" :: "r"(tmem_Q1_base), "r"(precompute_u32_smem(smem_Q1)) : "memory");
        asm volatile("tcgen05.cp.cta_group::1.64x128b [%0], [%1];" :: "r"(tmem_Q1_base + 32), "r"(precompute_u32_smem((char*)smem_Q1 + 4096)) : "memory");
    }
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    
    int global_row0 = q_block * 64 + my_lane_id + warp_id * 16;
    
    for (int step = 0; step <= q_block && step < num_q_blocks; step++) {
        int buf_idx = step % 2;
        int next_buf_idx = (step + 1) % 2;
        
        if (step + 1 <= q_block && threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&mbar_K[next_buf_idx], 16384);
            tma_load_4d_fn(&tma_K, &mbar_K[next_buf_idx], smem_K0[next_buf_idx], 0, b_h * S + (step + 1) * 64, h_idx, b_idx);
            tma_load_4d_fn(&tma_K, &mbar_K[next_buf_idx], (char*)smem_K0[next_buf_idx] + 8192, 64, b_h * S + (step + 1) * 64, h_idx, b_idx);
            tma_load_4d_fn(&tma_K, &mbar_K[next_buf_idx], smem_K1[next_buf_idx], 64, b_h * S + (step + 1) * 64, h_idx, b_idx);
            tma_load_4d_fn(&tma_K, &mbar_K[next_buf_idx], (char*)smem_K1[next_buf_idx] + 8192, 128, b_h * S + (step + 1) * 64, h_idx, b_idx);
            
            mbarrier_arrive_and_expect_tx_fn(&mbar_V0[next_buf_idx], 16384);
            tma_load_4d_fn(&tma_V, &mbar_V0[next_buf_idx], smem_V0[next_buf_idx], 0, b_h * S + (step + 1) * 64, h_idx, b_idx);
            tma_load_4d_fn(&tma_V, &mbar_V0[next_buf_idx], (char*)smem_V0[next_buf_idx] + 8192, 64, b_h * S + (step + 1) * 64, h_idx, b_idx);
            
            mbarrier_arrive_and_expect_tx_fn(&mbar_V1[next_buf_idx], 16384);
            tma_load_4d_fn(&tma_V, &mbar_V1[next_buf_idx], smem_V1[next_buf_idx], 64, b_h * S + (step + 1) * 64, h_idx, b_idx);
            tma_load_4d_fn(&tma_V, &mbar_V1[next_buf_idx], (char*)smem_V1[next_buf_idx] + 8192, 128, b_h * S + (step + 1) * 64, h_idx, b_idx);
        }
        
        mbarrier_wait_fn(&mbar_K[buf_idx], phase_K[buf_idx]);
        phase_K[buf_idx] ^= 1;
        mbarrier_wait_fn(&mbar_V0[buf_idx], phase_V0[buf_idx]);
        phase_V0[buf_idx] ^= 1;
        mbarrier_wait_fn(&mbar_V1[buf_idx], phase_V1[buf_idx]);
        phase_V1[buf_idx] ^= 1;
        
        __syncthreads();
        
        if (threadIdx.x == 0) {
            asm volatile("tcgen05.cp.cta_group::1.64x128b [%0], [%1];" :: "r"(tmem_K0_base), "r"(precompute_u32_smem(smem_K0[buf_idx])) : "memory");
            asm volatile("tcgen05.cp.cta_group::1.64x128b [%0], [%1];" :: "r"(tmem_K0_base + 32), "r"(precompute_u32_smem((char*)smem_K0[buf_idx] + 4096)) : "memory");
            
            asm volatile("tcgen05.cp.cta_group::1.64x128b [%0], [%1];" :: "r"(tmem_K1_base), "r"(precompute_u32_smem(smem_K1[buf_idx])) : "memory");
            asm volatile("tcgen05.cp.cta_group::1.64x128b [%0], [%1];" :: "r"(tmem_K1_base + 32), "r"(precompute_u32_smem((char*)smem_K1[buf_idx] + 4096)) : "memory");
            
            asm volatile("tcgen05.cp.cta_group::1.64x128b [%0], [%1];" :: "r"(tmem_V0_base), "r"(precompute_u32_smem(smem_V0[buf_idx])) : "memory");
            asm volatile("tcgen05.cp.cta_group::1.64x128b [%0], [%1];" :: "r"(tmem_V0_base + 32), "r"(precompute_u32_smem((char*)smem_V0[buf_idx] + 4096)) : "memory";
            
            asm volatile("tcgen05.cp.cta_group::1.64x128b [%0], [%1];" :: "r"(tmem_V1_base), "r"(precompute_u32_smem(smem_V1[buf_idx])) : "memory");
            asm volatile("tcgen05.cp.cta_group::1.64x128b [%0], [%1];" :: "r"(tmem_V1_base + 32), "r"(precompute_u32_smem((char*)smem_V1[buf_idx] + 4096)) : "memory";
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        uint32_t accum_S = (step == 0) ? 0 : 1;
        if (threadIdx.x == 0) {
            uint64_t desc_a0 = smem_desc_Q0();
            uint64_t desc_b0 = smem_desc_K0(buf_idx);
            uint32_t idesc_S = make_instr_desc_fn(64, 64, 0, 1); 
            umma_f16_cg1_fn(tmem_S, desc_a0, desc_b0, idesc_S, accum_S);
            
            uint64_t desc_a1 = smem_desc_Q1();
            uint64_t desc_b1 = smem_desc_K1(buf_idx);
            uint32_t idesc_S1 = make_instr_desc_fn(64, 64, 0, 1);
            umma_f16_cg1_fn(tmem_S, desc_a1, desc_b1, idesc_S1, 1);
            
            umma_commit_1sm_fn(&mbar_K[buf_idx]);
        }
        
        mbarrier_wait_fn(&mbar_K[buf_idx], phase_K[buf_idx]);
        phase_K[buf_idx] ^= 1;
        
        uint32_t r0[8], r1[8];
        tmem_load_8x_fn(my_lane_offset + 0, &r0[0], &r0[1], &r0[2], &r0[3], &r0[4], &r0[5], &r0[6], &r0[7]);
        tmem_load_8x_fn(my_lane_offset + 8, &r1[0], &r1[1], &r1[2], &r1[3], &r1[4], &r1[5], &r1[6], &r1[7]);
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float thread_max0 = -INFINITY;
        for(int i = 0; i < 8; i++) {
            float val = __uint_as_float(r0[i]);
            int chunk = i / 4;
            int rem = i % 4;
            int col = chunk * 8 + rem * 2;
            int global_col = step * 64 + col;
            if (global_col > global_row0 || global_col >= S || global_row0 >= S) {
                val = -INFINITY;
            } else {
                val *= scale;
            }
            r0[i] = __float_as_uint(val);
            thread_max0 = fmaxf(thread_max0, val);
        }
        for(int i = 0; i < 8; i++) {
            float val = __uint_as_float(r1[i]);
            int chunk = i / 4;
            int rem = i % 4;
            int col = chunk * 8 + rem * 2;
            int global_col = step * 64 + col;
            if (global_col > global_row0 || global_col >= S || global_row0 >= S) {
                val = -INFINITY;
            } else {
                val *= scale;
            }
            r1[i] = __float_as_uint(val);
            thread_max0 = fmaxf(thread_max0, val);
        }
        
        float row_max0 = thread_max0;
        for(int i=1; i<4; i++) row_max0 = fmaxf(row_max0, __shfl_xor_sync(0xffffffff, row_max0, i));
        
        float thread_sum0 = 0.0f;
        for(int i = 0; i < 8; i++) {
            float val = __uint_as_float(i < 4 ? r0[i] : r1[i-4]);
            val = __expf(val - row_max0);
            thread_sum0 += val;
            
            __nv_bfloat16 b_val = __float2bfloat16(val);
            int chunk = (i < 4) ? (i / 2) : 2 + ((i - 4) / 2);
            int rem = (i < 4) ? (i % 2) : ((i - 4) % 2);
            int smem_x_swizzled = (chunk ^ ((my_lane_id + warp_id * 16) % 8)) * 8 + rem * 2;
            *reinterpret_cast<uint16_t*>((char*)smem_P + (my_lane_id + warp_id * 16) * 128 + smem_x_swizzled * 2) = *reinterpret_cast<uint16_t*>(&b_val);
        }
        
        float r_s0 = __expf(my_global_max0 - row_max0);
        float new_global_sum0 = my_global_sum0 * r_s0 + thread_sum0;
        my_global_sum0 = new_global_sum0;
        my_global_max0 = row_max0;
        
        __syncthreads();
        
        if (threadIdx.x == 0) {
            asm volatile("tcgen05.cp.cta_group::1.64x128b [%0], [%1];" :: "r"(tmem_P_base), "r"(precompute_u32_smem(smem_P)) : "memory");
            asm volatile("tcgen05.cp.cta_group::1.64x128b [%0], [%1];" :: "r"(tmem_P_base + 32), "r"(precompute_u32_smem((char*)smem_P + 4096)) : "memory";
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        uint32_t accum_O0 = 1;
        uint32_t accum_O1 = 1;
        if (threadIdx.x == 0) {
            uint64_t desc_p = smem_desc_P();
            uint64_t desc_v0 = smem_desc_V0(buf_idx);
            uint32_t idesc_O0 = make_instr_desc_fn(64, 64, 0, 1);
            umma_f16_cg1_fn(tmem_O0, desc_p, desc_v0, idesc_O0, accum_O0); 
            
            uint64_t desc_v1 = smem_desc_V1(buf_idx);
            uint32_t idesc_O1 = make_instr_desc_fn(64, 64, 0, 1);
            umma_f16_cg1_fn(tmem_O1, desc_p, desc_v1, idesc_O1, accum_O1);
            
            umma_commit_1sm_fn(&mbar_V0[buf_idx]);
        }
        
        __syncthreads();
    }
    
    uint32_t out_r0[8], out_r1[8];
    tmem_load_8x_fn(my_lane_offset + 0, &out_r0[0], &out_r0[1], &out_r0[2], &out_r0[3], &out_r0[4], &out_r0[5], &out_r0[6], &out_r0[7]);
    tmem_load_8x_fn(my_lane_offset + 8, &out_r1[0], &out_r1[1], &out_r1[2], &out_r1[3], &out_r1[4], &out_r1[5], &out_r1[6], &out_r1[7]);
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    
    for(int i = 0; i < 8; i++) {
        float val0 = __uint_as_float(out_r0[i]) / my_global_sum0;
        float val1 = __uint_as_float(out_r1[i]) / my_global_sum0;
        
        int chunk = (i < 4) ? (i / 2) : 2 + ((i - 4) / 2);
        int rem = (i < 4) ? (i % 2) : ((i - 4) % 2);
        int smem_x_swizzled = (chunk ^ ((my_lane_id + warp_id * 16) % 8)) * 8 + rem * 2;
        
        __nv_bfloat16 b0 = __float2bfloat16(val0);
        __nv_bfloat16 b1 = __float2bfloat16(val1);
        
        *reinterpret_cast<uint16_t*>((char*)smem_O0 + (my_lane_id + warp_id * 16) * 128 + smem_x_swizzled * 2) = *reinterpret_cast<uint16_t*>(&b0);
        *reinterpret_cast<uint16_t*>((char*)smem_O1 + (my_lane_id + warp_id * 16) * 128 + smem_x_swizzled * 2) = *reinterpret_cast<uint16_t*>(&b1);
    }
    
    __syncthreads();
    
    uint32_t* O_bf16 = (uint32_t*)O;
    if (global_row0 < S) {
        for(int i = 0; i < 4; i++) {
            int chunk = i / 2;
            int rem = i % 2;
            int col = chunk * 8 + rem * 2;
            int smem_x_swizzled = (chunk ^ ((my_lane_id + warp_id * 16) % 8)) * 8 + rem * 2;
            uint32_t val = *reinterpret_cast<uint32_t*>((char*)smem_O0 + (my_lane_id + warp_id * 16) * 128 + smem_x_swizzled * 2);
            O_bf16[((b_h * S + global_row0) * 128 + col) >> 1] = val;
        }
        for(int i = 0; i < 4; i++) {
            int chunk = 2 + i / 2;
            int rem = i % 2;
            int col = chunk * 8 + rem * 2;
            int smem_x_swizzled = (chunk ^ ((my_lane_id + warp_id * 16) % 8)) * 8 + rem * 2;
            uint32_t val = *reinterpret_cast<uint32_t*>((char*)smem_O1 + (my_lane_id + warp_id * 16) * 128 + smem_x_swizzled * 2);
            O_bf16[((b_h * S + global_row0) * 128 + 64 + col) >> 1] = val;
        }
    }
    
    if (tid == 0) {
        if (global_row0 < S) {
            LSE[(b_h * S + global_row0)] = my_global_max0 + __logf(my_global_sum0);
        }
    }
    
    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_S, 64);
        tmem_dealloc_fn(tmem_O0, 64);
        tmem_dealloc_fn(tmem_O1, 64);
        tmem_dealloc_fn(tmem_K0_base, 64);
        tmem_dealloc_fn(tmem_K1_base, 64);
    }
}

namespace tvm_ffi_example_cuda {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3); 
    
    if (S == 0) return;
    
    float scale = 1.0f / sqrtf((float)D);
    
    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());
    
    CUtensorMap tma_Q, tma_K, tma_V;
    uint64_t gmem_dim[4] = {(uint64_t)D, (uint64_t)S, (uint64_t)H, (uint64_t)B};
    uint64_t gmem_strides[3] = {(uint64_t)(D * 2), (uint64_t)(D * S * 2), (uint64_t)(D * S * H * 2)};
    uint32_t smem_dim[4] = {64, 64, 1, 1};
    const cuuint32_t elemStrides[4] = {1, 1, 1, 1};
    
    CU_CHECK(cuTensorMapEncodeTiled(
        &tma_Q, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, (void*)Q_data, gmem_dim, gmem_strides, smem_dim, elemStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(cuTensorMapEncodeTiled(
        &tma_K, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, (void*)K_data, gmem_dim, gmem_strides, smem_dim, elemStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(cuTensorMapEncodeTiled(
        &tma_V, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, (void*)V_data, gmem_dim, gmem_strides, smem_dim, elemStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        
    int num_q_blocks = (S + 63) / 64;
    dim3 grid(num_q_blocks, B * H);
    dim3 block(128); 
    
    int smem_size = 96 * 1024;
    CUDA_CHECK(cudaFuncSetAttribute(
        attention_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    ));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, attention_kernel,
        tma_Q, tma_K, tma_V,
        O_data, LSE_data, B, H, S, scale
    ));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_example_cuda