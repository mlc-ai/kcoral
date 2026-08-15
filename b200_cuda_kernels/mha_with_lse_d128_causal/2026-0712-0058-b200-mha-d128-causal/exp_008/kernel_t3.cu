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

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n.reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n}\n" :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase));
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)), "l"((uint64_t)d), "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ uint32_t tmem_alloc_fn(int ncols) {
    uint32_t addr;
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&addr);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a), "r"(ncols));
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
    uint32_t idesc, uint32_t accum, float scale_d) {
    asm volatile(
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, %4, %5;\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum), "f"(scale_d));
}

__device__ __forceinline__ void umma_commit_1sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1"
        ".mbarrier::arrive::one.shared::cta.b64"
        " [%0];"
        :: "r"(a));
}

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

__device__ __forceinline__ uint64_t smem_desc_Q0_k(int k) {
    extern __shared__ __align__(1024) char smem_pool[];
    __nv_bfloat16* smem_Q0 = (__nv_bfloat16*)smem_pool;
    uint32_t sbo = 1024;
    uint32_t lbo = 1;
    return make_smem_desc_sm100_fn((char*)smem_Q0 + k * 16 * sizeof(__nv_bfloat16), lbo, sbo);
}
__device__ __forceinline__ uint64_t smem_desc_Q1_k(int k) {
    extern __shared__ __align__(1024) char smem_pool[];
    __nv_bfloat16* smem_Q1 = (__nv_bfloat16*)(smem_pool + 8192);
    uint32_t sbo = 1024;
    uint32_t lbo = 1;
    return make_smem_desc_sm100_fn((char*)smem_Q1 + k * 16 * sizeof(__nv_bfloat16), lbo, sbo);
}

__device__ __forceinline__ uint64_t smem_desc_K0_k(int k) {
    extern __shared__ __align__(1024) char smem_pool[];
    __nv_bfloat16* smem_K0[2] = {(__nv_bfloat16*)(smem_pool + 16384), (__nv_bfloat16*)(smem_pool + 24576)};
    uint32_t sbo = 1024;
    uint32_t lbo = 1;
    return make_smem_desc_sm100_fn((char*)smem_K0[0] + k * 16 * sizeof(__nv_bfloat16), lbo, sbo);
}
__device__ __forceinline__ uint64_t smem_desc_K1_k(int k) {
    extern __shared__ __align__(1024) char smem_pool[];
    __nv_bfloat16* smem_K1[2] = {(__nv_bfloat16*)(smem_pool + 32768), (__nv_bfloat16*)(smem_pool + 40960)};
    uint32_t sbo = 1024;
    uint32_t lbo = 1;
    return make_smem_desc_sm100_fn((char*)smem_K1[0] + k * 16 * sizeof(__nv_bfloat16), lbo, sbo);
}

__device__ __forceinline__ uint64_t smem_desc_V0_k(int k) {
    extern __shared__ __align__(1024) char smem_pool[];
    __nv_bfloat16* smem_V0[2] = {(__nv_bfloat16*)(smem_pool + 49152), (__nv_bfloat16*)(smem_pool + 57344)};
    uint32_t sbo = 1024;
    uint32_t lbo = 8192;
    return make_smem_desc_sm100_fn((char*)smem_V0[0] + k * 16 * 128, lbo, sbo);
}
__device__ __forceinline__ uint64_t smem_desc_V1_k(int k) {
    extern __shared__ __align__(1024) char smem_pool[];
    __nv_bfloat16* smem_V1[2] = {(__nv_bfloat16*)(smem_pool + 65536), (__nv_bfloat16*)(smem_pool + 73728)};
    uint32_t sbo = 1024;
    uint32_t lbo = 8192;
    return make_smem_desc_sm100_fn((char*)smem_V1[0] + k * 16 * 128, lbo, sbo);
}

__device__ __forceinline__ uint64_t smem_desc_P_k(int k) {
    extern __shared__ __align__(1024) char smem_pool[];
    __nv_bfloat16* smem_P = (__nv_bfloat16*)(smem_pool + 81920);
    uint32_t sbo = 1024;
    uint32_t lbo = 1;
    return make_smem_desc_sm100_fn((char*)smem_P + k * 16 * sizeof(__nv_bfloat16), lbo, sbo);
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
        
        mbarrier_arrive_and_expect_tx_fn(&mbar_Q[0], 16384);
        tma_load_2d_fn(&tma_Q, &mbar_Q[0], smem_Q0, 0, b_h * S + q_block * 64);
        tma_load_2d_fn(&tma_Q, &mbar_Q[0], smem_Q1, 64, b_h * S + q_block * 64);
        
        mbarrier_arrive_and_expect_tx_fn(&mbar_K[0], 16384);
        tma_load_2d_fn(&tma_K, &mbar_K[0], smem_K0[0], 0, b_h * S + 0 * 64);
        tma_load_2d_fn(&tma_K, &mbar_K[0], smem_K1[0], 64, b_h * S + 0 * 64);
        
        mbarrier_arrive_and_expect_tx_fn(&mbar_V0[0], 16384);
        tma_load_2d_fn(&tma_V, &mbar_V0[0], smem_V0[0], 0, b_h * S + 0 * 64);
        tma_load_2d_fn(&tma_V, &mbar_V0[0], smem_V1[0], 64, b_h * S + 0 * 64);
    }
    
    uint32_t tmem_S, tmem_O0, tmem_O1;
    
    if (threadIdx.x == 0) {
        tmem_S = tmem_alloc_fn(64);
        tmem_O0 = tmem_alloc_fn(64);
        tmem_O1 = tmem_alloc_fn(64);
    }
    __syncthreads();

    mbarrier_wait_fn(&mbar_Q[0], phase_Q[0]);
    phase_Q[0] ^= 1;
    
    int warp_id = threadIdx.x / 32;
    int tid = threadIdx.x % 32;
    uint32_t my_m_base = tid / 2;
    uint32_t my_lane_offset = (my_m_base + warp_id * 16) << 8;
    
    float global_max = -INFINITY;
    float global_sum = 0;
    float global_scale_o = 1.0f;
    
    float thread_r0[64], thread_r1[64];
    for(int i = 0; i < 64; i++) {
        thread_r0[i] = 0.0f;
        thread_r1[i] = 0.0f;
    }
    
    int q_step = q_block;
    for (int step = 0; step <= q_step && step < num_q_blocks; step++) {
        int buf_idx = step % 2;
        
        if (step + 1 <= q_step && threadIdx.x == 0) {
            int next_buf_idx = (step + 1) % 2;
            mbarrier_arrive_and_expect_tx_fn(&mbar_K[next_buf_idx], 16384);
            tma_load_2d_fn(&tma_K, &mbar_K[next_buf_idx], smem_K0[next_buf_idx], 0, b_h * S + (step + 1) * 64);
            tma_load_2d_fn(&tma_K, &mbar_K[next_buf_idx], smem_K1[next_buf_idx], 64, b_h * S + (step + 1) * 64);
            
            mbarrier_arrive_and_expect_tx_fn(&mbar_V0[next_buf_idx], 16384);
            tma_load_2d_fn(&tma_V, &mbar_V0[next_buf_idx], smem_V0[next_buf_idx], 0, b_h * S + (step + 1) * 64);
            tma_load_2d_fn(&tma_V, &mbar_V0[next_buf_idx], smem_V1[next_buf_idx], 64, b_h * S + (step + 1) * 64);
        }
        
        mbarrier_wait_fn(&mbar_K[buf_idx], phase_K[buf_idx]);
        phase_K[buf_idx] ^= 1;
        mbarrier_wait_fn(&mbar_V0[buf_idx], phase_V0[buf_idx]);
        phase_V0[buf_idx] ^= 1;
        
        __syncthreads();
        
        uint32_t accum_S = (step == 0) ? 0 : 1;
        float scale_S = 1.0f;
        if (threadIdx.x == 0) {
            for (int k = 0; k < 4; k++) {
                uint64_t desc_a0 = smem_desc_Q0_k(k);
                uint64_t desc_b0 = smem_desc_K0_k(k);
                uint32_t idesc0 = make_instr_desc_fn(64, 64, 0, 0); 
                umma_f16_cg1_fn(tmem_S, desc_a0, desc_b0, idesc0, accum_S, scale_S);
                accum_S = 1;
            }
            for (int k = 0; k < 4; k++) {
                uint64_t desc_a1 = smem_desc_Q1_k(k);
                uint64_t desc_b1 = smem_desc_K1_k(k);
                uint32_t idesc1 = make_instr_desc_fn(64, 64, 0, 0);
                umma_f16_cg1_fn(tmem_S, desc_a1, desc_b1, idesc1, accum_S, scale_S);
            }
            umma_commit_1sm_fn(&mbar_K[buf_idx]);
        }
        
        mbarrier_wait_fn(&mbar_K[buf_idx], phase_K[buf_idx]);
        phase_K[buf_idx] ^= 1;
        
        uint32_t r0[8], r1[8];
        tmem_load_8x_fn(my_lane_offset + 0, &r0[0], &r0[1], &r0[2], &r0[3], &r0[4], &r0[5], &r0[6], &r0[7]);
        tmem_load_8x_fn(my_lane_offset + 8, &r1[0], &r1[1], &r1[2], &r1[3], &r1[4], &r1[5], &r1[6], &r1[7]);
        tmem_load_8x_fn(my_lane_offset + 16, &r0[8], &r0[9], &r0[10], &r0[11], &r0[12], &r0[13], &r0[14], &r0[15]);
        tmem_load_8x_fn(my_lane_offset + 24, &r1[8], &r1[9], &r1[10], &r1[11], &r1[12], &r1[13], &r1[14], &r1[15]);
        tmem_load_8x_fn(my_lane_offset + 32, &r0[16], &r0[17], &r0[18], &r0[19], &r0[20], &r0[21], &r0[22], &r0[23]);
        tmem_load_8x_fn(my_lane_offset + 40, &r1[16], &r1[17], &r1[18], &r1[19], &r1[20], &r1[21], &r1[22], &r1[23]);
        tmem_load_8x_fn(my_lane_offset + 48, &r0[24], &r0[25], &r0[26], &r0[27], &r0[28], &r0[29], &r0[30], &r0[31]);
        tmem_load_8x_fn(my_lane_offset + 56, &r1[24], &r1[25], &r1[26], &r1[27], &r1[28], &r1[29], &r1[30], &r1[31]);
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        int global_row0 = q_step * 64 + my_m_base + warp_id * 16;
        int global_row1 = q_step * 64 + my_m_base + warp_id * 16 + 16;
        
        float thread_max0 = -INFINITY;
        for(int i = 0; i < 32; i++) {
            float val = __uint_as_float(r0[i]);
            int col_chunk = i / 4;
            int col = col_chunk * 8 + (i % 4);
            int global_col = step * 64 + col;
            if (global_col > global_row0 || global_col >= S || global_row0 >= S) {
                val = -INFINITY;
            } else {
                val *= scale;
            }
            r0[i] = __float_as_uint(val);
            thread_max0 = fmaxf(thread_max0, val);
        }
        
        float thread_max1 = -INFINITY;
        for(int i = 0; i < 32; i++) {
            float val = __uint_as_float(r1[i]);
            int col_chunk = i / 4;
            int col = col_chunk * 8 + (i % 4);
            int global_col = step * 64 + col;
            if (global_col > global_row1 || global_col >= S || global_row1 >= S) {
                val = -INFINITY;
            } else {
                val *= scale;
            }
            r1[i] = __float_as_uint(val);
            thread_max1 = fmaxf(thread_max1, val);
        }
        
        float row_max0 = thread_max0;
        for(int i=1; i<4; i++) row_max0 = fmaxf(row_max0, __shfl_xor_sync(0xffffffff, row_max0, i));
        
        float row_max1 = thread_max1;
        for(int i=1; i<4; i++) row_max1 = fmaxf(row_max1, __shfl_xor_sync(0xffffffff, row_max1, i));
        
        float row_max[2] = {row_max0, row_max1};
        float thread_sum[2] = {0.0f, 0.0f};
        
        for(int i = 0; i < 32; i++) {
            int row_idx = (i / 4) % 2;
            float val = __uint_as_float(row_idx == 0 ? r0[i] : r1[i]);
            val = __expf(val - row_max[row_idx]);
            thread_sum[row_idx] += val;
            
            __nv_bfloat16 b_val = __float2bfloat16(val);
            int col_chunk = i / 4;
            int col_rem = i % 4;
            int smem_x_swizzled = (col_chunk ^ ((my_m_base + warp_id * 16) % 8)) * 8 + col_rem;
            *reinterpret_cast<uint16_t*>((char*)smem_P + (my_m_base + warp_id * 16) * 128 + smem_x_swizzled * 2) = *reinterpret_cast<uint16_t*>(&b_val);
        }
        
        float r_s[2] = {1.0f, 1.0f};
        float r_o[2] = {1.0f, 1.0f};
        
        float new_max0 = fmaxf(global_max, row_max[0]);
        r_o[0] = __expf(global_max - new_max0);
        r_s[0] = r_o[0];
        
        float new_max1 = fmaxf(global_max, row_max[1]);
        r_o[1] = __expf(global_max - new_max1);
        r_s[1] = r_o[1];
        
        float new_global_sum = global_sum * r_s[0];
        
        float row_sum[2] = {thread_sum[0], thread_sum[1]};
        for(int i=1; i<4; i++) {
            row_sum[0] += __shfl_xor_sync(0xffffffff, row_sum[0], i);
            row_sum[1] += __shfl_xor_sync(0xffffffff, row_sum[1], i);
        }
        
        new_global_sum += row_sum[0] * r_s[0] + row_sum[1] * r_s[1];
        
        global_sum = new_global_sum;
        global_max = fmaxf(new_max0, new_max1);
        
        float final_scale_o = fminf(r_o[0], r_o[1]);
        global_scale_o *= final_scale_o;
        
        __syncthreads();
        
        uint32_t accum_O0 = 1;
        uint32_t accum_O1 = 1;
        if (threadIdx.x == 0) {
            for (int k = 0; k < 4; k++) {
                uint64_t desc_p = smem_desc_P_k(k);
                uint64_t desc_v0 = smem_desc_V0_k(k);
                uint32_t idesc_pv0 = make_instr_desc_fn(64, 64, 0, 1);
                umma_f16_cg1_fn(tmem_O0, desc_p, desc_v0, idesc_pv0, accum_O0, final_scale_o);
                accum_O0 = 1;
                
                uint64_t desc_v1 = smem_desc_V1_k(k);
                uint32_t idesc_pv1 = make_instr_desc_fn(64, 64, 0, 1);
                umma_f16_cg1_fn(tmem_O1, desc_p, desc_v1, idesc_pv1, accum_O1, final_scale_o);
                accum_O1 = 1;
            }
            umma_commit_1sm_fn(&mbar_V0[buf_idx]);
        }
        
        mbarrier_wait_fn(&mbar_V0[buf_idx], phase_V0[buf_idx]);
        phase_V0[buf_idx] ^= 1;
        mbarrier_wait_fn(&mbar_V1[buf_idx], phase_V1[buf_idx]);
        phase_V1[buf_idx] ^= 1;
        
        __syncthreads();
    }
    
    uint32_t out_r0[64], out_r1[64];
    tmem_load_8x_fn(my_lane_offset + 0, &out_r0[0], &out_r0[1], &out_r0[2], &out_r0[3], &out_r0[4], &out_r0[5], &out_r0[6], &out_r0[7]);
    tmem_load_8x_fn(my_lane_offset + 8, &out_r1[0], &out_r1[1], &out_r1[2], &out_r1[3], &out_r1[4], &out_r1[5], &out_r1[6], &out_r1[7]);
    tmem_load_8x_fn(my_lane_offset + 16, &out_r0[8], &out_r0[9], &out_r0[10], &out_r0[11], &out_r0[12], &out_r0[13], &out_r0[14], &out_r0[15]);
    tmem_load_8x_fn(my_lane_offset + 24, &out_r1[8], &out_r1[9], &out_r1[10], &out_r1[11], &out_r1[12], &out_r1[13], &out_r1[14], &out_r1[15]);
    tmem_load_8x_fn(my_lane_offset + 32, &out_r0[16], &out_r0[17], &out_r0[18], &out_r0[19], &out_r0[20], &out_r0[21], &out_r0[22], &out_r0[23]);
    tmem_load_8x_fn(my_lane_offset + 40, &out_r1[16], &out_r1[17], &out_r1[18], &out_r1[19], &out_r1[20], &out_r1[21], &out_r1[22], &out_r1[23]);
    tmem_load_8x_fn(my_lane_offset + 48, &out_r0[24], &out_r0[25], &out_r0[26], &out_r0[27], &out_r0[28], &out_r0[29], &out_r0[30], &out_r0[31]);
    tmem_load_8x_fn(my_lane_offset + 56, &out_r1[24], &out_r1[25], &out_r1[26], &out_r1[27], &out_r1[28], &out_r1[29], &out_r1[30], &out_r1[31]);
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    
    for(int i = 0; i < 64; i++) {
        float val0 = __uint_as_float(out_r0[i]) / global_sum;
        float val1 = __uint_as_float(out_r1[i]) / global_sum;
        
        int col_chunk = i / 4;
        int col_rem = i % 4;
        int smem_x_swizzled = (col_chunk ^ ((my_m_base + warp_id * 16) % 8)) * 8 + col_rem;
        
        __nv_bfloat16 b0 = __float2bfloat16(val0);
        __nv_bfloat16 b1 = __float2bfloat16(val1);
        
        *reinterpret_cast<uint16_t*>((char*)smem_O0 + (my_m_base + warp_id * 16) * 128 + smem_x_swizzled * 2) = *reinterpret_cast<uint16_t*>(&b0);
        *reinterpret_cast<uint16_t*>((char*)smem_O1 + (my_m_base + warp_id * 16) * 128 + smem_x_swizzled * 2) = *reinterpret_cast<uint16_t*>(&b1);
    }
    
    __syncthreads();
    
    int global_row0_out = q_step * 64 + my_m_base + warp_id * 16;
    int global_row1_out = q_step * 64 + my_m_base + warp_id * 16 + 16;
    
    uint32_t* O_bf16 = (uint32_t*)O;
    if (global_row0_out < S) {
        for(int i = 0; i < 64 / 2; i++) {
            int col = i * 2;
            int smem_x_swizzled = (col / 8 ^ ((my_m_base + warp_id * 16) % 8)) * 8 + (col % 8);
            uint32_t val = *reinterpret_cast<uint32_t*>((char*)smem_O0 + (my_m_base + warp_id * 16) * 128 + smem_x_swizzled * 2);
            O_bf16[((global_row0_out * 128 + col) >> 1)] = val;
        }
    }
    
    if (global_row1_out < S) {
        for(int i = 0; i < 64 / 2; i++) {
            int col = i * 2;
            int smem_x_swizzled = (col / 8 ^ ((my_m_base + warp_id * 16) % 8)) * 8 + (col % 8);
            uint32_t val = *reinterpret_cast<uint32_t*>((char*)smem_O1 + (my_m_base + warp_id * 16) * 128 + smem_x_swizzled * 2);
            O_bf16[((global_row1_out * 128 + col) >> 1)] = val;
        }
    }
    
    if (tid == 0) {
        float sum_val0 = __expf(-global_max) + __logf(global_sum);
        if (global_row0_out < S) {
            LSE[(b_h * S + global_row0_out)] = sum_val0;
        }
        float sum_val1 = __expf(-global_max) + __logf(global_sum);
        if (global_row1_out < S) {
            LSE[(b_h * S + global_row1_out)] = sum_val1;
        }
    }
    
    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_S, 64);
        tmem_dealloc_fn(tmem_O0, 64);
        tmem_dealloc_fn(tmem_O1, 64);
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
    uint64_t gmem_dim[2] = {(uint64_t)D, (uint64_t)(B * H * S)};
    uint32_t smem_dim[2] = {64, 64}; 
    
    CU_CHECK(cuTensorMapEncodeTiled(
        &tma_Q, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)Q_data, gmem_dim, (const cuuint64_t[]){{D * 2}}, smem_dim, (const cuuint32_t[]){{1}, {1}}, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(cuTensorMapEncodeTiled(
        &tma_K, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)K_data, gmem_dim, (const cuuint64_t[]){{D * 2}}, smem_dim, (const cuuint32_t[]){{1}, {1}}, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(cuTensorMapEncodeTiled(
        &tma_V, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)V_data, gmem_dim, (const cuuint64_t[]){{D * 2}}, smem_dim, (const cuuint32_t[]){{1}, {1}}, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        
    int num_q_blocks = (S + 63) / 64;
    dim3 grid(num_q_blocks, B * H);
    dim3 block(128); 
    
    int smem_size = 104 * 1024;
    CUDA_CHECK(cudaFuncSetAttribute(
        attention_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    ));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    attention_kernel<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V,
        O_data, LSE_data, B, H, S, scale
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_example_cuda