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

// ----------------------------------------------------------------
// Device helpers
// ----------------------------------------------------------------

__device__ __forceinline__ uint16_t bf16_to_bits(__nv_bfloat16 val) {
    uint16_t bits;
    asm volatile("mov.b32 %0, {%1, %2};" : "=r"(bits) : "h"(val), "h"(0));
    return bits;
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
    uint32_t smem_mbar = (uint32_t)__cvta_generic_to_shared(bar);
    uint32_t smem_ptr  = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];\n"
        :: "r"(smem_ptr), "l"((uint64_t)d), "r"(smem_mbar), "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
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

__device__ __forceinline__ void umma_commit_2sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::2"
        ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;"
        :: "r"(a), "h"((uint16_t)0x3));
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

__device__ __forceinline__ uint64_t make_smem_desc_mn_major_128x64(void* smem_ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)(16384 >> 4) << 16;
    d |= (uint64_t)(1024 >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, int major_a, int major_b) {
    uint32_t d = 0;
    d |= (1u << 4);     
    d |= (1u << 7);     
    d |= (1u << 10);    
    d |= ((major_a) << 15);   
    d |= ((major_b) << 16);    
    d |= ((N / 8) << 17);   
    d |= ((M / 16) << 24);  
    return d;
}

__device__ __forceinline__ uint32_t tmem_add_cols(uint32_t addr, uint32_t cols) { 
    return (addr & ~0xFFFF) | ((addr & 0xFFFF) + cols); 
}

__device__ __forceinline__ uint32_t tmem_add_rows(uint32_t addr, uint32_t rows) { 
    return addr + (rows << 16); 
}

__device__ __forceinline__ void fill_tmem_128x128_fp32(uint32_t tmem_base, float val) {
    if (threadIdx.x < 128) {
        uint32_t r = __float_as_uint(val);
        for (int col = 0; col < 128; col += 4) {
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                :: "r"(r), "r"(r), "r"(r), "r"(r), "r"(tmem_add_cols(tmem_base, col)));
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    }
}

__device__ __forceinline__ int get_swizzle_offset_128x128(int row, int col) {
    int span = col / 64;
    int col_in_span = col % 64;
    int x = col_in_span * 2 / 16;
    int rem = col_in_span * 2 % 16;
    int chunk_idx = (row % 8) ^ x;
    int byte_offset_base = row * 256;
    return byte_offset_base + span * 128 + chunk_idx * 16 + rem;
}

__device__ __forceinline__ int get_swizzle_offset_128x64(int row, int col) {
    int x = col * 2 / 16;
    int rem = col * 2 % 16;
    int chunk_idx = (row % 8) ^ x;
    return row * 128 + chunk_idx * 16 + rem;
}

__device__ __forceinline__ void write_swizzled_128B_128x128(uint8_t* smem, int row, int col, float val) {
    __nv_bfloat16 bval = __float2bfloat16(val);
    int byte_offset = get_swizzle_offset_128x128(row, col);
    asm volatile("st.shared.b16 [%0], %1;" :: "r"(byte_offset), "h"(bf16_to_bits(bval)));
}

__device__ __forceinline__ void write_swizzled_128B_128x64(uint8_t* smem, int row, int col, float val) {
    __nv_bfloat16 bval = __float2bfloat16(val);
    int byte_offset = get_swizzle_offset_128x64(row, col);
    asm volatile("st.shared.b16 [%0], %1;" :: "r"(byte_offset), "h"(bf16_to_bits(bval)));
}

__device__ __forceinline__ float read_swizzled_128B_fp32_128x128(const uint8_t* smem, int row, int col) {
    int byte_offset = get_swizzle_offset_128x128(row, col);
    uint16_t bits;
    asm volatile("ld.shared.b16 %0, [%1];" : "=h"(bits) : "r"(byte_offset));
    __nv_bfloat16 val = *reinterpret_cast<__nv_bfloat16*>(&bits);
    return __bfloat162float(val);
}

__device__ __forceinline__ void atomicAddAddFloatBf16(__nv_bfloat16* address, float value) {
    __nv_bfloat16 initial = *address;
    __nv_bfloat16 assumed;
    do {
        assumed = initial;
        float f_initial = __bfloat162float(assumed);
        __nv_bfloat16 expected = __float2bfloat16(f_initial + value);
        initial = atomicCAS(reinterpret_cast<uint32_t*>(address), 
                            bf16_to_bits(assumed), bf16_to_bits(expected));
    } while (assumed != initial);
}

extern __shared__ __align__(128) uint8_t smem_pool[];
extern __shared__ __align__(16) uint64_t mbar[1];

__global__ void __launch_bounds__(128) bwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_dO,
    const __grid_constant__ CUtensorMap tma_O,
    __nv_bfloat16* dQ,
    __nv_bfloat16* dK,
    __nv_bfloat16* dV,
    const float* L,
    int S, int d_dim, int H)
{
    extern __shared__ __align__(128) uint8_t smem_raw[];
    uint8_t* s_K_0 = smem_raw + 0;          // 16KB
    uint8_t* s_K_1 = smem_raw + 16384;      // 16KB
    uint8_t* s_V_0 = smem_raw + 32768;      // 16KB
    uint8_t* s_V_1 = smem_raw + 49152;      // 16KB
    uint8_t* s_Q_0 = smem_raw + 65536;      // 16KB
    uint8_t* s_Q_1 = smem_raw + 81920;      // 16KB
    uint8_t* s_dO_0 = smem_raw + 98304;     // 16KB
    uint8_t* s_dO_1 = smem_raw + 114688;    // 16KB
    uint8_t* s_O_0 = smem_raw + 131072;     // 16KB
    uint8_t* s_O_1 = smem_raw + 147456;     // 16KB
    
    float* s_D = (float*)(smem_raw + 163840);
    float* s_L = (float*)(smem_raw + 164352);
    
    uint8_t* s_P_T = smem_raw + 164864; // 32KB
    uint8_t* s_dS_T = smem_raw + 197632; // 32KB

    uint32_t* smem_tmem_S    = (uint32_t*)(smem_raw + 230400);
    uint32_t* smem_tmem_dP   = (uint32_t*)(smem_raw + 230408);
    uint32_t* smem_tmem_dV_0 = (uint32_t*)(smem_raw + 230416);
    uint32_t* smem_tmem_dV_1 = (uint32_t*)(smem_raw + 230424);
    uint32_t* smem_tmem_dK_0 = (uint32_t*)(smem_raw + 230432);
    uint32_t* smem_tmem_dK_1 = (uint32_t*)(smem_raw + 230440);
    uint32_t* smem_tmem_dQ_0 = (uint32_t*)(smem_raw + 230448);
    uint32_t* smem_tmem_dQ_1 = (uint32_t*)(smem_raw + 230456);

    int b_head = blockIdx.x; 
    int q_blk = blockIdx.y;
    int q_start = q_blk * 128;
    int b_idx = b_head / H;
    int h_idx = b_head % H;

    if (threadIdx.x == 0) {
        tmem_alloc_fn(smem_tmem_S, 128);
        tmem_alloc_fn(smem_tmem_dP, 128);
        tmem_alloc_fn(smem_tmem_dV_0, 128);
        tmem_alloc_fn(smem_tmem_dV_1, 128);
        tmem_alloc_fn(smem_tmem_dK_0, 128);
        tmem_alloc_fn(smem_tmem_dK_1, 128);
        tmem_alloc_fn(smem_tmem_dQ_0, 128);
        tmem_alloc_fn(smem_tmem_dQ_1, 128);

        init_smem_barrier_fn(mbar, 1); 
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    int phase = 0;
    uint32_t tmem_S = smem_tmem_S[0];
    uint32_t tmem_dP = smem_tmem_dP[0];
    uint32_t tmem_dV_0 = smem_tmem_dV_0[0];
    uint32_t tmem_dV_1 = smem_tmem_dV_1[0];
    uint32_t tmem_dK_0 = smem_tmem_dK_0[0];
    uint32_t tmem_dK_1 = smem_tmem_dK_1[0];
    uint32_t tmem_dQ_0 = smem_tmem_dQ_0[0];
    uint32_t tmem_dQ_1 = smem_tmem_dQ_1[0];

    if (threadIdx.x < 128) {
        fill_tmem_128x128_fp32(tmem_dV_0, 0);
        fill_tmem_128x128_fp32(tmem_dV_1, 0);
        fill_tmem_128x128_fp32(tmem_dK_0, 0);
        fill_tmem_128x128_fp32(tmem_dK_1, 0);
    }
    __syncthreads();

    uint32_t cur_tmem_dV_0 = tmem_dV_0;
    uint32_t cur_tmem_dV_1 = tmem_dV_1;
    uint32_t cur_tmem_dK_0 = tmem_dK_0;
    uint32_t cur_tmem_dK_1 = tmem_dK_1;

    int num_q_blks = (S + 127) / 128;
    float attn_scale = 1.0f / sqrtf((float)d_dim);
    int tid = threadIdx.x;

    for (int k_blk = 0; k_blk < num_q_blks; k_blk++) {
        int k_start = k_blk * 128;
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar, 98304);
            tma_load_4d_fn(&tma_Q, mbar, s_Q_0, 0, q_start, h_idx, b_idx);
            tma_load_4d_fn(&tma_Q, mbar, s_Q_1, 64, q_start, h_idx, b_idx);
            tma_load_4d_fn(&tma_dO, mbar, s_dO_0, 0, q_start, h_idx, b_idx);
            tma_load_4d_fn(&tma_dO, mbar, s_dO_1, 64, q_start, h_idx, b_idx);
            tma_load_4d_fn(&tma_O, mbar, s_O_0, 0, q_start, h_idx, b_idx);
            tma_load_4d_fn(&tma_O, mbar, s_O_1, 64, q_start, h_idx, b_idx);
            
            tma_load_4d_fn(&tma_K, mbar, s_K_0, 0, k_start, h_idx, b_idx);
            tma_load_4d_fn(&tma_K, mbar, s_K_1, 64, k_start, h_idx, b_idx);
            tma_load_4d_fn(&tma_V, mbar, s_V_0, 0, k_start, h_idx, b_idx);
            tma_load_4d_fn(&tma_V, mbar, s_V_1, 64, k_start, h_idx, b_idx);
        }
        
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;

        if (tid < 128) {
            int global_row = q_start + tid;
            s_L[tid] = (global_row < S) ? L[b_idx * H * S + h_idx * S + global_row] : 0.0f;
            
            float sum = 0.0f;
            if (global_row < S) {
                for(int i=0; i<64; i++) {
                    sum += read_swizzled_128B_fp32_128x128(s_O_0, tid, i) * read_swizzled_128B_fp32_128x128(s_dO_0, tid, i);
                    sum += read_swizzled_128B_fp32_128x128(s_O_1, tid, i) * read_swizzled_128B_fp32_128x128(s_dO_1, tid, i);
                }
            }
            s_D[tid] = sum;
        }
        __syncthreads();

        uint32_t accum_S = 0;
        for (int K_walker = 0; K_walker < 8; K_walker++) {
            uint64_t desc_a = make_smem_desc_sm100_fn((uint8_t*)s_K_0 + K_walker * 32, 1, 1024);
            uint64_t desc_b = make_smem_desc_sm100_fn((uint8_t*)s_Q_0 + K_walker * 32, 1, 1024);
            uint32_t idesc_S = make_instr_desc_fn(128, 64, 0, 0);
            umma_f16_cg1_fn(tmem_add_cols(tmem_S, K_walker * 16), desc_a, desc_b, idesc_S, accum_S);
            accum_S = 1;
        }
        for (int K_walker = 0; K_walker < 8; K_walker++) {
            uint64_t desc_a = make_smem_desc_sm100_fn((uint8_t*)s_K_1 + K_walker * 32, 1, 1024);
            uint64_t desc_b = make_smem_desc_sm100_fn((uint8_t*)s_Q_1 + K_walker * 32, 1, 1024);
            uint32_t idesc_S = make_instr_desc_fn(128, 64, 0, 0);
            umma_f16_cg1_fn(tmem_add_cols(tmem_S, 64 + K_walker * 16), desc_a, desc_b, idesc_S, accum_S);
        }

        uint32_t accum_dP = 0;
        for (int K_walker = 0; K_walker < 8; K_walker++) {
            uint64_t desc_a = make_smem_desc_sm100_fn((uint8_t*)s_V_0 + K_walker * 32, 1, 1024);
            uint64_t desc_b = make_smem_desc_sm100_fn((uint8_t*)s_dO_0 + K_walker * 32, 1, 1024);
            uint32_t idesc_dP = make_instr_desc_fn(128, 64, 0, 0);
            umma_f16_cg1_fn(tmem_add_cols(tmem_dP, K_walker * 16), desc_a, desc_b, idesc_dP, accum_dP);
            accum_dP = 1;
        }
        for (int K_walker = 0; K_walker < 8; K_walker++) {
            uint64_t desc_a = make_smem_desc_sm100_fn((uint8_t*)s_V_1 + K_walker * 32, 1, 1024);
            uint64_t desc_b = make_smem_desc_sm100_fn((uint8_t*)s_dO_1 + K_walker * 32, 1, 1024);
            uint32_t idesc_dP = make_instr_desc_fn(128, 64, 0, 0);
            umma_f16_cg1_fn(tmem_add_cols(tmem_dP, 64 + K_walker * 16), desc_a, desc_b, idesc_dP, accum_dP);
        }

        umma_commit_2sm_fn(mbar);
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;

        if (tid < 128) {
            for (int col = 0; col < 128; col += 4) {
                uint32_t r0, r1, r2, r3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_add_rows(tmem_add_cols(tmem_S, col), tid)));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                
                float s0 = __uint_as_float(r0);
                float s1 = __uint_as_float(r1);
                float s2 = __uint_as_float(r2);
                float s3 = __uint_as_float(r3);
                
                float p0 = __expf(s0 * attn_scale - s_L[tid]);
                float p1 = __expf(s1 * attn_scale - s_L[tid]);
                float p2 = __expf(s2 * attn_scale - s_L[tid]);
                float p3 = __expf(s3 * attn_scale - s_L[tid]);
                
                if (k_start + tid >= S || q_start + col >= S) {
                    p0 = 0; p1 = 0; p2 = 0; p3 = 0;
                }
                
                int byte_off0 = get_swizzle_offset_128x128(tid, col + 0);
                int byte_off1 = get_swizzle_offset_128x128(tid, col + 1);
                int byte_off2 = get_swizzle_offset_128x128(tid, col + 2);
                int byte_off3 = get_swizzle_offset_128x128(tid, col + 3);
                
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(byte_off0), "h"(bf16_to_bits(__float2bfloat16(p0))));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(byte_off1), "h"(bf16_to_bits(__float2bfloat16(p1))));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(byte_off2), "h"(bf16_to_bits(__float2bfloat16(p2))));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(byte_off3), "h"(bf16_to_bits(__float2bfloat16(p3))));
            }
        }

        if (tid < 128) {
            for (int col = 0; col < 128; col += 4) {
                uint32_t r0, r1, r2, r3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_add_rows(tmem_add_cols(tmem_dP, col), tid)));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                
                float dp0 = __uint_as_float(r0);
                float dp1 = __uint_as_float(r1);
                float dp2 = __uint_as_float(r2);
                float dp3 = __uint_as_float(r3);
                
                float p0 = read_swizzled_128B_fp32_128x128(s_P_T, tid, col + 0);
                float p1 = read_swizzled_128B_fp32_128x128(s_P_T, tid, col + 1);
                float p2 = read_swizzled_128B_fp32_128x128(s_P_T, tid, col + 2);
                float p3 = read_swizzled_128B_fp32_128x128(s_P_T, tid, col + 3);
                
                float ds0 = p0 * (dp0 - s_D[tid]);
                float ds1 = p1 * (dp1 - s_D[tid]);
                float ds2 = p2 * (dp2 - s_D[tid]);
                float ds3 = p3 * (dp3 - s_D[tid]);
                
                int byte_off0 = get_swizzle_offset_128x128(tid, col + 0);
                int byte_off1 = get_swizzle_offset_128x128(tid, col + 1);
                int byte_off2 = get_swizzle_offset_128x128(tid, col + 2);
                int byte_off3 = get_swizzle_offset_128x128(tid, col + 3);
                
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(byte_off0), "h"(bf16_to_bits(__float2bfloat16(ds0))));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(byte_off1), "h"(bf16_to_bits(__float2bfloat16(ds1))));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(byte_off2), "h"(bf16_to_bits(__float2bfloat16(ds2))));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(byte_off3), "h"(bf16_to_bits(__float2bfloat16(ds3))));
            }
        }
        __syncthreads();

        uint32_t accum_dV = 1;
        for (int K_walker = 0; K_walker < 8; K_walker++) {
            uint64_t desc_a = make_smem_desc_sm100_fn((uint8_t*)s_P_T + K_walker * 32, 1, 1024);
            uint64_t desc_b = make_smem_desc_mn_major_128x64((uint8_t*)s_dO_0 + K_walker * 2048);
            uint32_t idesc_dV = make_instr_desc_fn(128, 64, 0, 1);
            umma_f16_cg1_fn(tmem_add_cols(cur_tmem_dV_0, K_walker * 16), desc_a, desc_b, idesc_dV, accum_dV);
        }
        for (int K_walker = 0; K_walker < 8; K_walker++) {
            uint64_t desc_a = make_smem_desc_sm100_fn((uint8_t*)s_P_T + K_walker * 32 + 8192, 1, 1024);
            uint64_t desc_b = make_smem_desc_mn_major_128x64((uint8_t*)s_dO_1 + K_walker * 2048);
            uint32_t idesc_dV = make_instr_desc_fn(128, 64, 0, 1);
            umma_f16_cg1_fn(tmem_add_cols(cur_tmem_dV_1, K_walker * 16), desc_a, desc_b, idesc_dV, accum_dV);
        }

        uint32_t accum_dK = 1;
        for (int K_walker = 0; K_walker < 8; K_walker++) {
            uint64_t desc_a = make_smem_desc_sm100_fn((uint8_t*)s_dS_T + K_walker * 32, 1, 1024);
            uint64_t desc_b = make_smem_desc_mn_major_128x64((uint8_t*)s_Q_0 + K_walker * 2048);
            uint32_t idesc_dK = make_instr_desc_fn(128, 64, 0, 1);
            umma_f16_cg1_fn(tmem_add_cols(cur_tmem_dK_0, K_walker * 16), desc_a, desc_b, idesc_dK, accum_dK);
        }
        for (int K_walker = 0; K_walker < 8; K_walker++) {
            uint64_t desc_a = make_smem_desc_sm100_fn((uint8_t*)s_dS_T + K_walker * 32 + 8192, 1, 1024);
            uint64_t desc_b = make_smem_desc_mn_major_128x64((uint8_t*)s_Q_1 + K_walker * 2048);
            uint32_t idesc_dK = make_instr_desc_fn(128, 64, 0, 1);
            umma_f16_cg1_fn(tmem_add_cols(cur_tmem_dK_1, K_walker * 16), desc_a, desc_b, idesc_dK, accum_dK);
        }
        
        umma_commit_2sm_fn(mbar);
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;

        if (tid < 128) {
            for (int col = 0; col < 128; col += 4) {
                uint32_t r0, r1, r2, r3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_add_rows(tmem_add_cols(tmem_dV_0, col), tid)));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float f0 = __uint_as_float(r0);
                float f1 = __uint_as_float(r1);
                float f2 = __uint_as_float(r2);
                float f3 = __uint_as_float(r3);
                int k_off = col;
                if (k_start + tid < S) {
                    atomicAddAddFloatBf16(&dV[b_idx * H * S * 128 + (k_start + tid) * 128 + k_off], f0);
                    atomicAddAddFloatBf16(&dV[b_idx * H * S * 128 + (k_start + tid) * 128 + k_off + 1], f1);
                    atomicAddAddFloatBf16(&dV[b_idx * H * S * 128 + (k_start + tid) * 128 + k_off + 2], f2);
                    atomicAddAddFloatBf16(&dV[b_idx * H * S * 128 + (k_start + tid) * 128 + k_off + 3], f3);
                }
            }
            for (int col = 0; col < 128; col += 4) {
                uint32_t r0, r1, r2, r3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_add_rows(tmem_add_cols(tmem_dV_1, col), tid)));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float f0 = __uint_as_float(r0);
                float f1 = __uint_as_float(r1);
                float f2 = __uint_as_float(r2);
                float f3 = __uint_as_float(r3);
                int k_off = col;
                if (k_start + tid < S) {
                    atomicAddAddFloatBf16(&dV[b_idx * H * S * 128 + (k_start + tid) * 128 + k_off + 64], f0);
                    atomicAddAddFloatBf16(&dV[b_idx * H * S * 128 + (k_start + tid) * 128 + k_off + 65], f1);
                    atomicAddAddFloatBf16(&dV[b_idx * H * S * 128 + (k_start + tid) * 128 + k_off + 66], f2);
                    atomicAddAddFloatBf16(&dV[b_idx * H * S * 128 + (k_start + tid) * 128 + k_off + 67], f3);
                }
            }
        }

        if (tid < 128) {
            for (int col = 0; col < 128; col += 4) {
                uint32_t r0, r1, r2, r3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_add_rows(tmem_add_cols(tmem_dK_0, col), tid)));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float f0 = __uint_as_float(r0);
                float f1 = __uint_as_float(r1);
                float f2 = __uint_as_float(r2);
                float f3 = __uint_as_float(r3);
                int k_off = col;
                if (k_start + tid < S) {
                    atomicAddAddFloatBf16(&dK[b_idx * H * S * 128 + (k_start + tid) * 128 + k_off], f0);
                    atomicAddAddFloatBf16(&dK[b_idx * H * S * 128 + (k_start + tid) * 128 + k_off + 1], f1);
                    atomicAddAddFloatBf16(&dK[b_idx * H * S * 128 + (k_start + tid) * 128 + k_off + 2], f2);
                    atomicAddAddFloatBf16(&dK[b_idx * H * S * 128 + (k_start + tid) * 128 + k_off + 3], f3);
                }
            }
            for (int col = 0; col < 128; col += 4) {
                uint32_t r0, r1, r2, r3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_add_rows(tmem_add_cols(tmem_dK_1, col), tid)));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float f0 = __uint_as_float(r0);
                float f1 = __uint_as_float(r1);
                float f2 = __uint_as_float(r2);
                float f3 = __uint_as_float(r3);
                int k_off = col;
                if (k_start + tid < S) {
                    atomicAddAddFloatBf16(&dK[b_idx * H * S * 128 + (k_start + tid) * 128 + k_off + 64], f0);
                    atomicAddAddFloatBf16(&dK[b_idx * H * S * 128 + (k_start + tid) * 128 + k_off + 65], f1);
                    atomicAddAddFloatBf16(&dK[b_idx * H * S * 128 + (k_start + tid) * 128 + k_off + 66], f2);
                    atomicAddAddFloatBf16(&dK[b_idx * H * S * 128 + (k_start + tid) * 128 + k_off + 67], f3);
                }
            }
        }
        
        cur_tmem_dV_0 += 128;
        cur_tmem_dV_1 += 128;
        cur_tmem_dK_0 += 128;
        cur_tmem_dK_1 += 128;
        __syncthreads(); 
    }

    uint32_t accum_dQ_0 = 0;
    uint32_t accum_dQ_1 = 0;
    uint32_t cur_tmem_dQ_0 = tmem_dQ_0;
    uint32_t cur_tmem_dQ_1 = tmem_dQ_1;

    if (threadIdx.x < 128) {
        fill_tmem_128x128_fp32(cur_tmem_dQ_0, 0);
        fill_tmem_128x128_fp32(cur_tmem_dQ_1, 0);
    }
    __syncthreads();

    for (int k_blk = 0; k_blk < num_q_blks; k_blk++) {
        int k_start = k_blk * 128;
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar, 98304);
            tma_load_4d_fn(&tma_Q, mbar, s_Q_0, 0, q_start, h_idx, b_idx);
            tma_load_4d_fn(&tma_Q, mbar, s_Q_1, 64, q_start, h_idx, b_idx);
            tma_load_4d_fn(&tma_dO, mbar, s_dO_0, 0, q_start, h_idx, b_idx);
            tma_load_4d_fn(&tma_dO, mbar, s_dO_1, 64, q_start, h_idx, b_idx);
            tma_load_4d_fn(&tma_O, mbar, s_O_0, 0, q_start, h_idx, b_idx);
            tma_load_4d_fn(&tma_O, mbar, s_O_1, 64, q_start, h_idx, b_idx);
            
            tma_load_4d_fn(&tma_K, mbar, s_K_0, 0, k_start, h_idx, b_idx);
            tma_load_4d_fn(&tma_K, mbar, s_K_1, 64, k_start, h_idx, b_idx);
            tma_load_4d_fn(&tma_V, mbar, s_V_0, 0, k_start, h_idx, b_idx);
            tma_load_4d_fn(&tma_V, mbar, s_V_1, 64, k_start, h_idx, b_idx);
        }
        
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;

        if (tid < 128) {
            int global_row = q_start + tid;
            s_L[tid] = (global_row < S) ? L[b_idx * H * S + h_idx * S + global_row] : 0.0f;
            
            float sum = 0.0f;
            if (global_row < S) {
                for(int i=0; i<64; i++) {
                    sum += read_swizzled_128B_fp32_128x128(s_O_0, tid, i) * read_swizzled_128B_fp32_128x128(s_dO_0, tid, i);
                    sum += read_swizzled_128B_fp32_128x128(s_O_1, tid, i) * read_swizzled_128B_fp32_128x128(s_dO_1, tid, i);
                }
            }
            s_D[tid] = sum;
        }
        __syncthreads();

        uint32_t accum_S = 0;
        for (int K_walker = 0; K_walker < 8; K_walker++) {
            uint64_t desc_a = make_smem_desc_sm100_fn((uint8_t*)s_K_0 + K_walker * 32, 1, 1024);
            uint64_t desc_b = make_smem_desc_sm100_fn((uint8_t*)s_Q_0 + K_walker * 32, 1, 1024);
            uint32_t idesc_S = make_instr_desc_fn(128, 64, 0, 0);
            umma_f16_cg1_fn(tmem_add_cols(tmem_S, K_walker * 16), desc_a, desc_b, idesc_S, accum_S);
            accum_S = 1;
        }
        for (int K_walker = 0; K_walker < 8; K_walker++) {
            uint64_t desc_a = make_smem_desc_sm100_fn((uint8_t*)s_K_1 + K_walker * 32, 1, 1024);
            uint64_t desc_b = make_smem_desc_sm100_fn((uint8_t*)s_Q_1 + K_walker * 32, 1, 1024);
            uint32_t idesc_S = make_instr_desc_fn(128, 64, 0, 0);
            umma_f16_cg1_fn(tmem_add_cols(tmem_S, 64 + K_walker * 16), desc_a, desc_b, idesc_S, accum_S);
        }

        uint32_t accum_dP = 0;
        for (int K_walker = 0; K_walker < 8; K_walker++) {
            uint64_t desc_a = make_smem_desc_sm100_fn((uint8_t*)s_V_0 + K_walker * 32, 1, 1024);
            uint64_t desc_b = make_smem_desc_sm100_fn((uint8_t*)s_dO_0 + K_walker * 32, 1, 1024);
            uint32_t idesc_dP = make_instr_desc_fn(128, 64, 0, 0);
            umma_f16_cg1_fn(tmem_add_cols(tmem_dP, K_walker * 16), desc_a, desc_b, idesc_dP, accum_dP);
            accum_dP = 1;
        }
        for (int K_walker = 0; K_walker < 8; K_walker++) {
            uint64_t desc_a = make_smem_desc_sm100_fn((uint8_t*)s_V_1 + K_walker * 32, 1, 1024);
            uint64_t desc_b = make_smem_desc_sm100_fn((uint8_t*)s_dO_1 + K_walker * 32, 1, 1024);
            uint32_t idesc_dP = make_instr_desc_fn(128, 64, 0, 0);
            umma_f16_cg1_fn(tmem_add_cols(tmem_dP, 64 + K_walker * 16), desc_a, desc_b, idesc_dP, accum_dP);
        }

        umma_commit_2sm_fn(mbar);
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;

        if (tid < 128) {
            for (int col = 0; col < 128; col += 4) {
                uint32_t r0, r1, r2, r3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_add_rows(tmem_add_cols(tmem_S, col), tid)));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                
                float s0 = __uint_as_float(r0);
                float s1 = __uint_as_float(r1);
                float s2 = __uint_as_float(r2);
                float s3 = __uint_as_float(r3);
                
                float p0 = __expf(s0 * attn_scale - s_L[tid]);
                float p1 = __expf(s1 * attn_scale - s_L[tid]);
                float p2 = __expf(s2 * attn_scale - s_L[tid]);
                float p3 = __expf(s3 * attn_scale - s_L[tid]);
                
                if (k_start + tid >= S || q_start + col >= S) {
                    p0 = 0; p1 = 0; p2 = 0; p3 = 0;
                }
                
                float dp0 = 0, dp1 = 0, dp2 = 0, dp3 = 0;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_add_rows(tmem_add_cols(tmem_dP, col), tid)));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                dp0 = __uint_as_float(r0);
                dp1 = __uint_as_float(r1);
                dp2 = __uint_as_float(r2);
                dp3 = __uint_as_float(r3);
                
                float ds0 = p0 * (dp0 - s_D[tid]);
                float ds1 = p1 * (dp1 - s_D[tid]);
                float ds2 = p2 * (dp2 - s_D[tid]);
                float ds3 = p3 * (dp3 - s_D[tid]);
                
                int byte_off0 = get_swizzle_offset_128x128(tid, col + 0);
                int byte_off1 = get_swizzle_offset_128x128(tid, col + 1);
                int byte_off2 = get_swizzle_offset_128x128(tid, col + 2);
                int byte_off3 = get_swizzle_offset_128x128(tid, col + 3);
                
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(byte_off0), "h"(bf16_to_bits(__float2bfloat16(ds0))));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(byte_off1), "h"(bf16_to_bits(__float2bfloat16(ds1))));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(byte_off2), "h"(bf16_to_bits(__float2bfloat16(ds2))));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(byte_off3), "h"(bf16_to_bits(__float2bfloat16(ds3))));
            }
        }
        __syncthreads();

        uint32_t accum_dQ_0_local = accum_dQ_0;
        for (int K_walker = 0; K_walker < 8; K_walker++) {
            uint64_t desc_a = make_smem_desc_sm100_fn((uint8_t*)s_dS_T + K_walker * 32, 1, 1024);
            uint64_t desc_b = make_smem_desc_mn_major_128x64((uint8_t*)s_K_0 + K_walker * 2048);
            uint32_t idesc_dQ = make_instr_desc_fn(128, 64, 0, 1);
            umma_f16_cg1_fn(tmem_add_cols(cur_tmem_dQ_0, K_walker * 16), desc_a, desc_b, idesc_dQ, accum_dQ_0_local);
            accum_dQ_0_local = 1;
        }

        uint32_t accum_dQ_1_local = accum_dQ_1;
        for (int K_walker = 0; K_walker < 8; K_walker++) {
            uint64_t desc_a = make_smem_desc_sm100_fn((uint8_t*)s_dS_T + K_walker * 32, 1, 1024);
            uint64_t desc_b = make_smem_desc_mn_major_128x64((uint8_t*)s_K_1 + K_walker * 2048);
            uint32_t idesc_dQ = make_instr_desc_fn(128, 64, 0, 1);
            umma_f16_cg1_fn(tmem_add_cols(cur_tmem_dQ_1, K_walker * 16), desc_a, desc_b, idesc_dQ, accum_dQ_1_local);
            accum_dQ_1_local = 1;
        }

        umma_commit_2sm_fn(mbar);
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        
        accum_dQ_0 = 1; 
        accum_dQ_1 = 1;
        cur_tmem_dQ_0 += 128;
        cur_tmem_dQ_1 += 128;
    }

    if (threadIdx.x < 128) {
        for (int col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_add_rows(tmem_add_cols(tmem_dQ_0, col), threadIdx.x)));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            write_swizzled_128B_128x64(s_Q_0, threadIdx.x, col, __uint_as_float(r0));
            write_swizzled_128B_128x64(s_Q_1, threadIdx.x, col, __uint_as_float(r1));
            write_swizzled_128B_128x64(s_O_0, threadIdx.x, col, __uint_as_float(r2));
            write_swizzled_128B_128x64(s_O_1, threadIdx.x, col, __uint_as_float(r3));
        }
        
        for (int col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_add_rows(tmem_add_cols(tmem_dQ_1, col), threadIdx.x)));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            write_swizzled_128B_128x64(s_Q_0, threadIdx.x, col + 64, __uint_as_float(r0));
            write_swizzled_128B_128x64(s_Q_1, threadIdx.x, col + 64, __uint_as_float(r1));
            write_swizzled_128B_128x64(s_O_0, threadIdx.x, col + 64, __uint_as_float(r2));
            write_swizzled_128B_128x64(s_O_1, threadIdx.x, col + 64, __uint_as_float(r3));
        }
    }
    
    __syncthreads(); 
    
    if (tid < 128) {
        for (int col = 0; col < 128; col += 4) {
            uint2 dq0 = *reinterpret_cast<uint2*>(&s_Q_0[tid * 128 + col]);
            uint2 dq1 = *reinterpret_cast<uint2*>(&s_Q_1[tid * 128 + col]);
            int global_row = q_start + tid;
            if (global_row < S) {
                *reinterpret_cast<uint2*>(&dQ[b_idx * H * S * 128 + global_row * 128 + col]) = dq0;
                *reinterpret_cast<uint2*>(&dQ[b_idx * H * S * 128 + global_row * 128 + col + 64]) = dq1;
            }
        }
    }
}

CUresult create_tma_4d_descriptor(CUtensorMap* d, void* globalAddress, 
                                  uint64_t dim0, uint64_t dim1, uint64_t dim2, uint64_t dim3,
                                  uint32_t box0, uint32_t box1, uint32_t box2, uint32_t box3) {
    cuuint64_t globalDim[4] = {dim0, dim1, dim2, dim3};
    cuuint64_t globalStrides[3] = {dim0 * 2, dim0 * dim1 * 2, dim0 * dim1 * dim2 * 2};
    cuuint32_t boxDim[4] = {box0, box1, box2, box3};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, globalAddress, globalDim, globalStrides, boxDim,
        elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

namespace tvm_ffi {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = Q.size(3);
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_dO, tma_O;
    create_tma_4d_descriptor(&tma_Q, Q.data_ptr(), d, S, H, B, 64, 128, 1, 1);
    create_tma_4d_descriptor(&tma_K, K.data_ptr(), d, S, H, B, 64, 128, 1, 1);
    create_tma_4d_descriptor(&tma_V, V.data_ptr(), d, S, H, B, 64, 128, 1, 1);
    create_tma_4d_descriptor(&tma_dO, dO.data_ptr(), d, S, H, B, 64, 128, 1, 1);
    create_tma_4d_descriptor(&tma_O, O.data_ptr(), d, S, H, B, 64, 128, 1, 1);
    
    int smem_size = 245760; 
    cudaFuncSetAttribute(bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    
    int num_q_blks = (S + 127) / 128;
    dim3 grid(B * H, num_q_blks, 1);
    dim3 block(128, 1, 1);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, bwd_kernel, 
        tma_Q, tma_K, tma_V, tma_dO, tma_O, 
        static_cast<__nv_bfloat16*>(dQ.data_ptr()), 
        static_cast<__nv_bfloat16*>(dK.data_ptr()), 
        static_cast<__nv_bfloat16*>(dV.data_ptr()), 
        static_cast<const float*>(L.data_ptr()), 
        S, d, H));
        
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi