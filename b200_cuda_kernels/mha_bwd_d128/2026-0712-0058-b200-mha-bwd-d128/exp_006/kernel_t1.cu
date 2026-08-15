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

// ----------------------------------------------------------------
// Device helpers
// ----------------------------------------------------------------

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
                :: "r"(r), "r"(r), "r"(r), "r"(r), "r"(col));
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    }
}

__device__ __forceinline__ void load_gmem_to_tmem_bf16_fp32(__nv_bfloat16* gmem, uint32_t tmem_base, int S, int d_dim, int row_start, int col_start) {
    int tid = threadIdx.x;
    if (tid < 128) {
        for (int col = 0; col < 128; col += 4) {
            float f0 = 0, f1 = 0, f2 = 0, f3 = 0;
            if (row_start + tid < S && col_start + col < d_dim) {
                f0 = __bfloat162float(gmem[(row_start + tid) * d_dim + col_start + col + 0]);
                f1 = __bfloat162float(gmem[(row_start + tid) * d_dim + col_start + col + 1]);
                f2 = __bfloat162float(gmem[(row_start + tid) * d_dim + col_start + col + 2]);
                f3 = __bfloat162float(gmem[(row_start + tid) * d_dim + col_start + col + 3]);
            }
            uint32_t r0 = __float_as_uint(f0);
            uint32_t r1 = __float_as_uint(f1);
            uint32_t r2 = __float_as_uint(f2);
            uint32_t r3 = __float_as_uint(f3);
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                :: "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(tmem_add_cols(tmem_base, col)));
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    }
}

__device__ __forceinline__ void read_S_P_vals(uint32_t tmem_S, float (*out_S)[128]) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int row = (warp_id * 32 + lane_id) % 128;
    
    for (int c = 0; c < 8; ++c) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_add_cols(tmem_S, c * 16)));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        out_S[row][c * 16 + 0] = __uint_as_float(r0);
        out_S[row][c * 16 + 1] = __uint_as_float(r1);
        out_S[row][c * 16 + 2] = __uint_as_float(r2);
        out_S[row][c * 16 + 3] = __uint_as_float(r3);
    }
}

__device__ __forceinline__ void write_swizzled_128B(uint8_t* smem, int row, int col, float val) {
    __nv_bfloat16 bval = __float2bfloat16(val);
    int x = col * 2 / 16;
    int rem = col * 2 % 16;
    int chunk_idx = (row % 8) ^ x;
    int byte_offset = row * 256 + chunk_idx * 16 + rem;
    asm volatile("st.shared.b16 [%0], %1;" :: "r"(byte_offset), "h"(bval));
}

__device__ __forceinline__ float read_swizzled_128B_fp32(const uint8_t* smem, int row, int col) {
    int x = col * 2 / 16;
    int rem = col * 2 % 16;
    int chunk_idx = (row % 8) ^ x;
    int byte_offset = row * 256 + chunk_idx * 16 + rem;
    __nv_bfloat16 val;
    asm volatile("ld.shared.b16 %0, [%1];" : "=h"(val) : "r"(byte_offset));
    return __bfloat162float(val);
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

    uint32_t* smem_tmem_S   = (uint32_t*)(smem_raw + 230400); // size 4
    uint32_t* smem_tmem_dV  = (uint32_t*)(smem_raw + 230408); // size 4
    uint32_t* smem_tmem_dP  = (uint32_t*)(smem_raw + 230416); // size 4
    uint32_t* smem_tmem_dK  = (uint32_t*)(smem_raw + 230424); // size 4
    uint32_t* smem_tmem_dQ  = (uint32_t*)(smem_raw + 230432); // size 4

    int b_head = blockIdx.x; 
    int q_blk = blockIdx.y;
    int q_start = q_blk * 128;
    int b_idx = b_head / H;
    int h_idx = b_head % H;

    if (threadIdx.x == 0) {
        tmem_alloc_fn(smem_tmem_S, 128);
        tmem_alloc_fn(smem_tmem_dV, 128);
        tmem_alloc_fn(smem_tmem_dP, 128);
        tmem_alloc_fn(smem_tmem_dK, 128);
        tmem_alloc_fn(smem_tmem_dQ, 128);

        init_smem_barrier_fn(mbar, 1); 
        fence_smem_barrier_init_fn();
        
        mbarrier_arrive_and_expect_tx_fn(mbar, 65536); 
        
        tma_load_4d_fn(&tma_K, mbar, s_K_0, 0, q_start, h_idx, b_idx);
        tma_load_4d_fn(&tma_K, mbar, s_K_1, 64, q_start, h_idx, b_idx);
        tma_load_4d_fn(&tma_V, mbar, s_V_0, 0, q_start, h_idx, b_idx);
        tma_load_4d_fn(&tma_V, mbar, s_V_1, 64, q_start, h_idx, b_idx);
    }

    int phase = 0;
    mbarrier_wait_fn(mbar, phase);
    phase ^= 1;

    uint32_t tmem_S = smem_tmem_S[0];
    uint32_t tmem_dV = smem_tmem_dV[0];
    uint32_t tmem_dP = smem_tmem_dP[0];
    uint32_t tmem_dK = smem_tmem_dK[0];
    uint32_t tmem_dQ = smem_tmem_dQ[0];

    if (threadIdx.x < 128) {
        fill_tmem_128x128_fp32(tmem_dV, 0);
        fill_tmem_128x128_fp32(tmem_dK, 0);
    }
    __syncthreads();

    uint32_t cur_tmem_dV = tmem_dV;
    uint32_t cur_tmem_dK = tmem_dK;
    uint32_t accum_dV = 0;
    uint32_t accum_dK = 0;

    int num_q_blks = (S + 127) / 128;
    float attn_scale = 1.0f / sqrtf((float)d_dim);

    for (int k_blk = q_blk; k_blk >= 0; --k_blk) {
        int k_start = k_blk * 128;
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar, 98304);
            tma_load_4d_fn(&tma_Q, mbar, s_Q_0, 0, k_start, h_idx, b_idx);
            tma_load_4d_fn(&tma_Q, mbar, s_Q_1, 64, k_start, h_idx, b_idx);
            tma_load_4d_fn(&tma_dO, mbar, s_dO_0, 0, k_start, h_idx, b_idx);
            tma_load_4d_fn(&tma_dO, mbar, s_dO_1, 64, k_start, h_idx, b_idx);
            tma_load_4d_fn(&tma_O, mbar, s_O_0, 0, k_start, h_idx, b_idx);
            tma_load_4d_fn(&tma_O, mbar, s_O_1, 64, k_start, h_idx, b_idx);
        }
        
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;

        if (threadIdx.x < 128) {
            int global_row = k_start + threadIdx.x;
            s_L[threadIdx.x] = (global_row < S) ? L[b_head * S + global_row] : 0.0f;
        }

        if (threadIdx.x < 128) {
            int global_row = k_start + threadIdx.x;
            if (global_row < S) {
                float sum = 0.0f;
                for(int i=0; i<128; i++) {
                    float o  = read_swizzled_128B_fp32(s_O_0, threadIdx.x, i);
                    float do_ = read_swizzled_128B_fp32(s_dO_0, threadIdx.x, i);
                    sum += o * do_;
                }
                s_D[threadIdx.x] = sum;
            } else {
                s_D[threadIdx.x] = 0.0f;
            }
        }
        __syncthreads();

        uint32_t accum = 0;
        uint32_t cur_tmem_S = tmem_S;
        for (int K_walker = 0; K_walker < 8; K_walker++) {
            uint64_t desc_a = make_smem_desc_sm100_fn((uint8_t*)s_K_0 + K_walker * 16, 1, 1024);
            uint64_t desc_b = make_smem_desc_sm100_fn((uint8_t*)s_Q_0 + K_walker * 16, 1, 1024);
            uint32_t idesc_S = make_instr_desc_fn(128, 16);
            umma_f16_cg1_fn(tmem_add_cols(cur_tmem_S, K_walker * 16), desc_a, desc_b, idesc_S, accum);
            accum = 1;
        }
        for (int K_walker = 0; K_walker < 8; K_walker++) {
            uint64_t desc_a = make_smem_desc_sm100_fn((uint8_t*)s_K_1 + K_walker * 16, 1, 1024);
            uint64_t desc_b = make_smem_desc_sm100_fn((uint8_t*)s_Q_1 + K_walker * 16, 1, 1024);
            uint32_t idesc_S = make_instr_desc_fn(128, 16);
            umma_f16_cg1_fn(tmem_add_cols(cur_tmem_S, K_walker * 16 + 64), desc_a, desc_b, idesc_S, accum);
        }

        float local_S[128];
        int tid = threadIdx.x;
        for(int i = 0; i < 128; i++) local_S[i] = 0.0f;
        read_S_P_vals(tmem_S, (float(*)[128])local_S);
        
        for(int i = 0; i < 128; i++) {
            float p = local_S[i];
            if (k_start + i < S && q_start < S) {
                float p_val = __expf(p * attn_scale - s_L[tid]);
                write_swizzled_128B(s_P_T, tid, i, p_val);
            } else {
                write_swizzled_128B(s_P_T, tid, i, 0.0f);
            }
        }
        __syncthreads();

        uint32_t accum_dP = 0;
        uint32_t cur_tmem_dP = tmem_dP;
        for (int K_walker = 0; K_walker < 8; K_walker++) {
            uint64_t desc_a = make_smem_desc_sm100_fn((uint8_t*)s_V_0 + K_walker * 16, 1, 1024);
            uint64_t desc_b = make_smem_desc_sm100_fn((uint8_t*)s_dO_0 + K_walker * 16, 1, 1024);
            uint32_t idesc_dP = make_instr_desc_fn(128, 16);
            umma_f16_cg1_fn(tmem_add_cols(cur_tmem_dP, K_walker * 16), desc_a, desc_b, idesc_dP, accum_dP);
            accum_dP = 1;
        }
        for (int K_walker = 0; K_walker < 8; K_walker++) {
            uint64_t desc_a = make_smem_desc_sm100_fn((uint8_t*)s_V_1 + K_walker * 16, 1, 1024);
            uint64_t desc_b = make_smem_desc_sm100_fn((uint8_t*)s_dO_1 + K_walker * 16, 1, 1024);
            uint32_t idesc_dP = make_instr_desc_fn(128, 16);
            umma_f16_cg1_fn(tmem_add_cols(cur_tmem_dP, K_walker * 16 + 64), desc_a, desc_b, idesc_dP, accum_dP);
        }

        float local_dP[128];
        for(int i = 0; i < 128; i++) local_dP[i] = 0.0f;
        read_S_P_vals(tmem_dP, (float(*)[128])local_dP);
        
        for(int i = 0; i < 128; i++) {
            float dp = local_dP[i];
            float p_val = read_swizzled_128B_fp32(s_P_T, tid, i);
            float ds_val = p_val * (dp - s_D[tid]);
            write_swizzled_128B(s_dS_T, tid, i, ds_val);
        }
        __syncthreads();

        for (int K_walker = 0; K_walker < 8; K_walker++) {
            uint64_t desc_a = make_smem_desc_sm100_fn((uint8_t*)s_P_T + K_walker * 16, 1, 1024);
            uint64_t desc_b = make_smem_desc_sm100_fn((uint8_t*)s_dO_0 + K_walker * 1024, 8192, 128);
            uint32_t idesc_dV = make_instr_desc_fn(128, 64, 0, 1);
            umma_f16_cg1_fn(tmem_add_cols(cur_tmem_dV, K_walker * 16), desc_a, desc_b, idesc_dV, accum_dV);
        }
        for (int K_walker = 0; K_walker < 8; K_walker++) {
            uint64_t desc_a = make_smem_desc_sm100_fn((uint8_t*)s_P_T + K_walker * 16 + 8192, 1, 1024);
            uint64_t desc_b = make_smem_desc_sm100_fn((uint8_t*)s_dO_1 + K_walker * 1024, 8192, 128);
            uint32_t idesc_dV = make_instr_desc_fn(128, 64, 0, 1);
            umma_f16_cg1_fn(tmem_add_cols(cur_tmem_dV, K_walker * 16 + 64), desc_a, desc_b, idesc_dV, accum_dV);
        }

        for (int K_walker = 0; K_walker < 8; K_walker++) {
            uint64_t desc_a = make_smem_desc_sm100_fn((uint8_t*)s_dS_T + K_walker * 16, 1, 1024);
            uint64_t desc_b = make_smem_desc_sm100_fn((uint8_t*)s_Q_0 + K_walker * 1024, 8192, 128);
            uint32_t idesc_dK = make_instr_desc_fn(128, 64, 0, 1);
            umma_f16_cg1_fn(tmem_add_cols(cur_tmem_dK, K_walker * 16), desc_a, desc_b, idesc_dK, accum_dK);
        }
        for (int K_walker = 0; K_walker < 8; K_walker++) {
            uint64_t desc_a = make_smem_desc_sm100_fn((uint8_t*)s_dS_T + K_walker * 16 + 8192, 1, 1024);
            uint64_t desc_b = make_smem_desc_sm100_fn((uint8_t*)s_Q_1 + K_walker * 1024, 8192, 128);
            uint32_t idesc_dK = make_instr_desc_fn(128, 64, 0, 1);
            umma_f16_cg1_fn(tmem_add_cols(cur_tmem_dK, K_walker * 16 + 64), desc_a, desc_b, idesc_dK, accum_dK);
        }
        
        cur_tmem_dV += 128;
        cur_tmem_dK += 128;
        __syncthreads(); 
    }

    float local_dV[128];
    float local_dK[128];
    for(int i = 0; i < 128; i++) { local_dV[i] = 0.0f; local_dK[i] = 0.0f; }
    read_S_P_vals(tmem_dV, (float(*)[128])local_dV);
    read_S_P_vals(tmem_dK, (float(*)[128])local_dK);

    for(int i = 0; i < 128; i++) {
        write_swizzled_128B(s_V_0, tid, i, local_dV[i]);
        write_swizzled_128B(s_K_0, tid, i, local_dK[i]);
    }
    __syncthreads();

    if (tid < 128) {
        for (int col = 0; col < 64; col += 4) {
            uint2 dv0 = *reinterpret_cast<uint2*>(&s_V_0[tid * 64 + col]);
            uint2 dv1 = *reinterpret_cast<uint2*>(&s_V_1[tid * 64 + col]);
            int global_row = q_start + tid;
            if (global_row < S) {
                *reinterpret_cast<uint2*>(&dV[b_head * S * 128 + global_row * 128 + col]) = dv0;
                *reinterpret_cast<uint2*>(&dV[b_head * S * 128 + global_row * 128 + col + 64]) = dv1;
            }
        }
        for (int col = 0; col < 64; col += 4) {
            uint2 dk0 = *reinterpret_cast<uint2*>(&s_K_0[tid * 64 + col]);
            uint2 dk1 = *reinterpret_cast<uint2*>(&s_K_1[tid * 64 + col]);
            int global_row = q_start + tid;
            if (global_row < S) {
                *reinterpret_cast<uint2*>(&dK[b_head * S * 128 + global_row * 128 + col]) = dk0;
                *reinterpret_cast<uint2*>(&dK[b_head * S * 128 + global_row * 128 + col + 64]) = dk1;
            }
        }
    }
    __syncthreads(); 

    uint32_t accum_dQ = 0;
    uint32_t cur_tmem_dQ = tmem_dQ;
    
    load_gmem_to_tmem_bf16_fp32(dQ, cur_tmem_dQ, S, d_dim, q_start, 0);
    accum_dQ = 1;

    for (int k_blk = 0; k_blk <= q_blk; k_blk++) {
        int k_start = k_blk * 128;
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar, 32768);
            tma_load_4d_fn(&tma_K, mbar, s_K_0, 0, k_start, h_idx, b_idx);
            tma_load_4d_fn(&tma_K, mbar, s_K_1, 64, k_start, h_idx, b_idx);
        }
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;

        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar, 65536);
            tma_load_4d_fn(&tma_Q, mbar, s_Q_0, 0, q_start, h_idx, b_idx);
            tma_load_4d_fn(&tma_Q, mbar, s_Q_1, 64, q_start, h_idx, b_idx);
            tma_load_4d_fn(&tma_dO, mbar, s_dO_0, 0, q_start, h_idx, b_idx);
            tma_load_4d_fn(&tma_dO, mbar, s_dO_1, 64, q_start, h_idx, b_idx);
            tma_load_4d_fn(&tma_V, mbar, s_V_0, 0, k_start, h_idx, b_idx);
            tma_load_4d_fn(&tma_V, mbar, s_V_1, 64, k_start, h_idx, b_idx