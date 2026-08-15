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

__device__ __forceinline__ uint16_t bf16_to_bits(__nv_bfloat16 val) {
    return *reinterpret_cast<uint16_t*>(&val);
}

__device__ __forceinline__ uint32_t pack_bf16_fn(float a, float b) {
    __nv_bfloat162 p;
    p.x = __float2bfloat16(a);
    p.y = __float2bfloat16(b);
    return *(reinterpret_cast<uint32_t*>(&p));
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(phase));
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

__device__ __forceinline__ void umma_f16_cg2_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_2sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile(
        "tcgen05.commit.cta_group::2"
        ".mbarrier::arrive::one.shared::cluster"
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

__device__ __forceinline__ uint64_t make_smem_desc_k_major_64(void* smem_ptr) {
    return make_smem_desc_sm100_fn(smem_ptr, 1, 1024);
}

__device__ __forceinline__ uint64_t make_smem_desc_n_major_64(void* smem_ptr) {
    return make_smem_desc_sm100_fn(smem_ptr, 16384, 1024);
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

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void fill_tmem_128x128_fp32(uint32_t tmem_base, float val) {
    if (threadIdx.x < 128) {
        uint32_t r = __float_as_uint(val);
        for (int col_base = 0; col_base < 128; col_base += 16) {
            uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
            r0 = r; r1 = r; r2 = r; r3 = r;
            r4 = r; r5 = r; r6 = r; r7 = r;
            asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                :: "r"(r0), "r"(r1), "r"(r2), "r"(r3),
                   "r"(r4), "r"(r5), "r"(r6), "r"(r7),
                   "r"(tmem_add_cols(tmem_base, col_base)));
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    }
    fence_proxy_async_fn();
}

struct SwizzledStore {
    uint8_t* smem;
    int row;
    int col;
    uint32_t buf[4];
    int idx;

    __device__ SwizzledStore(uint8_t* smem, int row, int start_col) 
        : smem(smem), row(row), col(start_col), idx(0) {}

    __device__ void write(int col, float val) {
        buf[idx] = pack_bf16_fn(buf[idx], val); 
        idx++;
        if (idx == 4) {
            int byte_off = get_swizzle_offset_128x128(row, col - 3); 
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" 
                         :: "r"(byte_off), "r"(buf[0]), "r"(buf[1]), "r"(buf[2]), "r"(buf[3]));
            idx = 0;
        }
    }

    __device__ ~SwizzledStore() {
        if (idx > 0) {
            int byte_off = get_swizzle_offset_128x128(row, col - idx);
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" 
                         :: "r"(byte_off), "r"(buf[0]), "r"(buf[1]), "r"(buf[2]), "r"(buf[3]));
        }
    }
};

__device__ __forceinline__ int get_swizzle_offset_128x128(int row, int col) {
    int span = col / 64;
    int col_in_span = col % 64;
    int x = col_in_span * 2 / 16;
    int rem = col_in_span * 2 % 16;
    int chunk_idx = (row % 8) ^ x;
    int byte_offset_base = row * 256;
    return byte_offset_base + span * 128 + chunk_idx * 16 + rem;
}

__device__ __forceinline__ void write_swizzled_transposed(uint8_t* smem, int col, int tid, float val) {
    __nv_bfloat16 bval = __float2bfloat16(val);
    int byte_offset = get_swizzle_offset_128x128(col, tid);
    asm volatile("st.shared.b16 [%0], %1;" :: "r"(byte_offset), "h"(bf16_to_bits(bval)));
}

__device__ __forceinline__ float read_swizzled_128B_fp32_transposed(const uint8_t* smem, int tid, int col) {
    int byte_offset = get_swizzle_offset_128x128(col, tid);
    uint16_t bits;
    asm volatile("ld.shared.b16 %0, [%1];" : "=h"(bits) : "r"(byte_offset));
    __nv_bfloat16 val = *reinterpret_cast<__nv_bfloat16*>(&bits);
    return __bfloat162float(val);
}

__device__ __forceinline__ uint4 read_swizzled_vec_32(uint8_t* smem, int row, int col) {
    int byte_offset = get_swizzle_offset_128x128(row, col);
    uint4 val;
    asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4]);" 
                 : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w) : "r"(byte_offset));
    return val;
}

__device__ __forceinline__ uint4 read_swizzled_vec_32_oob(uint8_t* smem, int row, int col, int max_col) {
    if (col + 7 > max_col) {
        return {0, 0, 0, 0};
    }
    return read_swizzled_vec_32(smem, row, col);
}

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

extern __shared__ __align__(128) uint8_t smem_pool[];
extern __shared__ __align__(16) uint64_t mbar[1];

__global__ void __launch_bounds__(128) bwd_kernel_dv_dk(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_dO,
    const __grid_constant__ CUtensorMap tma_O,
    __nv_bfloat16* dK,
    __nv_bfloat16* dV,
    const float* L,
    int S, int d_dim, int H)
{
    extern __shared__ __align__(128) uint8_t smem_raw[];
    uint8_t* s_K_0 = smem_raw + 0;          
    uint8_t* s_K_1 = smem_raw + 16384;      
    uint8_t* s_V_0 = smem_raw + 32768;      
    uint8_t* s_V_1 = smem_raw + 49152;      
    uint8_t* s_Q_0 = smem_raw + 65536;      
    uint8_t* s_Q_1 = smem_raw + 81920;      
    uint8_t* s_dO_0 = smem_raw + 98304;     
    uint8_t* s_dO_1 = smem_raw + 114688;    
    uint8_t* s_O_0 = smem_raw + 131072;     
    uint8_t* s_O_1 = smem_raw + 147456;     
    
    float* s_D = (float*)(smem_raw + 163840);
    float* s_L = (float*)(smem_raw + 164352);
    
    uint8_t* s_P_T = smem_raw + 164864; 
    uint8_t* s_dS_T = smem_raw + 197632; 

    uint32_t* smem_tmem_S    = (uint32_t*)(smem_raw + 230400);
    uint32_t* smem_tmem_dP   = (uint32_t*)(smem_raw + 230408);
    uint32_t* smem_tmem_dV_0 = (uint32_t*)(smem_raw + 230416);
    uint32_t* smem_tmem_dV_1 = (uint32_t*)(smem_raw + 230424);
    uint32_t* smem_tmem_dK_0 = (uint32_t*)(smem_raw + 230432);
    uint32_t* smem_tmem_dK_1 = (uint32_t*)(smem_raw + 230440);

    int k_blk = blockIdx.x;
    int b_head = blockIdx.y;
    int b_idx = b_head / H;
    int h_idx = b_head % H;
    int tid = threadIdx.x;

    int cluster_offset = (cluster_rank_fn() % 2) * 64;
    int k_start = k_blk * 128 + cluster_offset;

    if (threadIdx.x == 0) {
        tmem_alloc_fn(smem_tmem_S, 128);
        tmem_alloc_fn(smem_tmem_dP, 128);
        tmem_alloc_fn(smem_tmem_dV_0, 128);
        tmem_alloc_fn(smem_tmem_dV_1, 128);
        tmem_alloc_fn(smem_tmem_dK_0, 128);
        tmem_alloc_fn(smem_tmem_dK_1, 128);

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

    if (threadIdx.x < 128) {
        fill_tmem_128x128_fp32(tmem_dV_0, 0);
        fill_tmem_128x128_fp32(tmem_dV_1, 0);
        fill_tmem_128x128_fp32(tmem_dK_0, 0);
        fill_tmem_128x128_fp32(tmem_dK_1, 0);
    }
    __syncthreads();

    float attn_scale = 1.0f / sqrtf((float)d_dim);

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar, 65536); 
        
        tma_load_4d_fn(&tma_K, mbar, s_K_0, 0, k_start, h_idx, b_idx);
        tma_load_4d_fn(&tma_K, mbar, s_K_1, 64, k_start, h_idx, b_idx);
        tma_load_4d_fn(&tma_V, mbar, s_V_0, 0, k_start, h_idx, b_idx);
        tma_load_4d_fn(&tma_V, mbar, s_V_1, 64, k_start, h_idx, b_idx);
    }

    int num_q_blks = (S + 127) / 128;

    for (int q_blk = k_blk; q_blk < num_q_blks; q_blk++) {
        int q_start = q_blk * 128;
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar, 98304);
            tma_load_4d_fn(&tma_Q, mbar, s_Q_0, 0, q_start, h_idx, b_idx);
            tma_load_4d_fn(&tma_Q, mbar, s_Q_1, 64, q_start, h_idx, b_idx);
            tma_load_4d_fn(&tma_dO, mbar, s_dO_0, 0, q_start, h_idx, b_idx);
            tma_load_4d_fn(&tma_dO, mbar, s_dO_1, 64, q_start, h_idx, b_idx);
            tma_load_4d_fn(&tma_O, mbar, s_O_0, 0, q_start, h_idx, b_idx);
            tma_load_4d_fn(&tma_O, mbar, s_O_1, 64, q_start, h_idx, b_idx);
        }
        
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;

        if (tid < 128) {
            int global_row = q_start + tid;
            s_L[tid] = (global_row < S) ? L[b_idx * H * S + h_idx * S + global_row] : 0.0f;
            
            float sum = 0.0f;
            if (global_row < S) {
                uint4 ov0 = read_swizzled_vec_32_oob(s_O_0, tid, 0, 64);
                uint4 dv0 = read_swizzled_vec_32_oob(s_dO_0, tid, 0, 64);
                uint4 ov1 = read_swizzled_vec_32_oob(s_O_1, tid, 0, 64);
                uint4 dv1 = read_swizzled_vec_32_oob(s_dO_1, tid, 0, 64);

                uint32_t* of0 = (uint32_t*)&ov0;
                uint32_t* dof0 = (uint32_t*)&dv0;
                uint32_t* of1 = (uint32_t*)&ov1;
                uint32_t* dof1 = (uint32_t*)&dv1;

                for (int i = 0; i < 4; i++) {
                    __nv_bfloat162 o_val0 = *(__nv_bfloat162*)&of0[i];
                    __nv_bfloat162 do_val0 = *(__nv_bfloat162*)&dof0[i];
                    sum += __bfloat162float(o_val0.x) * __bfloat162float(do_val0.x);
                    sum += __bfloat162float(o_val0.y) * __bfloat162float(do_val0.y);
                    
                    __nv_bfloat162 o_val1 = *(__nv_bfloat162*)&of1[i];
                    __nv_bfloat162 do_val1 = *(__nv_bfloat162*)&dof1[i];
                    sum += __bfloat162float(o_val1.x) * __bfloat162float(do_val1.x);
                    sum += __bfloat162float(o_val1.y) * __bfloat162float(do_val1.y);
                }
            }
            s_D[tid] = sum;
        }
        __syncthreads();

        uint32_t accum_S = 0;
        for (int K_walker = 0; K_walker < 4; K_walker++) {
            uint64_t desc_a = make_smem_desc_k_major_64((uint8_t*)s_K_0 + K_walker * 32);
            uint64_t desc_b = make_smem_desc_k_major_64((uint8_t*)s_Q_0 + K_walker * 32);
            uint32_t idesc_S = make_instr_desc_fn(256, 128, 0, 0);
            umma_f16_cg2_fn(tmem_add_cols(tmem_S, K_walker * 16), desc_a, desc_b, idesc_S, accum_S);
            accum_S = 1;
        }
        for (int K_walker = 0; K_walker < 4; K_walker++) {
            uint64_t desc_a = make_smem_desc_k_major_64((uint8_t*)s_K_1 + K_walker * 32);
            uint64_t desc_b = make_smem_desc_k_major_64((uint8_t*)s_Q_1 + K_walker * 32);
            uint32_t idesc_S = make_instr_desc_fn(256, 128, 0, 0);
            umma_f16_cg2_fn(tmem_add_cols(tmem_S, 64 + K_walker * 16), desc_a, desc_b, idesc_S, accum_S);
        }

        uint32_t accum_dP = 0;
        for (int K_walker = 0; K_walker < 4; K_walker++) {
            uint64_t desc_a = make_smem_desc_k_major_64((uint8_t*)s_V_0 + K_walker * 32);
            uint64_t desc_b = make_smem_desc_k_major_64((uint8_t*)s_dO_0 + K_walker * 32);
            uint32_t idesc_dP = make_instr_desc_fn(256, 128, 0, 0);
            umma_f16_cg2_fn(tmem_add_cols(tmem_dP, K_walker * 16), desc_a, desc_b, idesc_dP, accum_dP);
            accum_dP = 1;
        }
        for (int K_walker = 0; K_walker < 4; K_walker++) {
            uint64_t desc_a = make_smem_desc_k_major_64((uint8_t*)s_V_1 + K_walker * 32);
            uint64_t desc_b = make_smem_desc_k_major_64((uint8_t*)s_dO_1 + K_walker * 32);
            uint32_t idesc_dP = make_instr_desc_fn(256, 128, 0, 0);
            umma_f16_cg2_fn(tmem_add_cols(tmem_dP, 64 + K_walker * 16), desc_a, desc_b, idesc_dP, accum_dP);
        }

        umma_commit_2sm_fn(mbar);
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;

        if (tid < 128) {
            for (int col_base = 0; col_base < 128; col_base += 16) {
                uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                    : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),
                      "=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(tmem_add_cols(tmem_S, col_base)));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                
                float s0 = __uint_as_float(r0);
                float s1 = __uint_as_float(r1);
                float s2 = __uint_as_float(r2);
                float s3 = __uint_as_float(r3);
                float s4 = __uint_as_float(r4);
                float s5 = __uint_as_float(r5);
                float s6 = __uint_as_float(r6);
                float s7 = __uint_as_float(r7);
                
                float p0 = __expf(s0 * attn_scale - s_L[tid]);
                float p1 = __expf(s1 * attn_scale - s_L[tid]);
                float p2 = __expf(s2 * attn_scale - s_L[tid]);
                float p3 = __expf(s3 * attn_scale - s_L[tid]);
                float p4 = __expf(s4 * attn_scale - s_L[tid]);
                float p5 = __expf(s5 * attn_scale - s_L[tid]);
                float p6 = __expf(s6 * attn_scale - s_L[tid]);
                float p7 = __expf(s7 * attn_scale - s_L[tid]);
                
                if (k_start + tid >= S || q_start + col_base >= S) {
                    p0 = 0; p1 = 0; p2 = 0; p3 = 0;
                    p4 = 0; p5 = 0; p6 = 0; p7 = 0;
                }
                
                SwizzledStore P_store((uint8_t*)s_P_T, tid, col_base);
                P_store.write(col_base + 0, p0);
                P_store.write(col_base + 1, p1);
                P_store.write(col_base + 2, p2);
                P_store.write(col_base + 3, p3);
                P_store.write(col_base + 4, p4);
                P_store.write(col_base + 5, p5);
                P_store.write(col_base + 6, p6);
                P_store.write(col_base + 7, p7);
            }
        }
        
        if (tid < 128) {
            for (int col_base = 0; col_base < 128; col_base += 16) {
                uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                    : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),
                      "=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(tmem_add_cols(tmem_dP, col_base)));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                
                float dp0 = __uint_as_float(r0);
                float dp1 = __uint_as_float(r1);
                float dp2 = __uint_as_float(r2);
                float dp3 = __uint_as_float(r3);
                float dp4 = __uint_as_float(r4);
                float dp5 = __uint_as_float(r5);
                float dp6 = __uint_as_float(r6);
                float dp7 = __uint_as_float(r7);
                
                float p0 = read_swizzled_128B_fp32_transposed(s_P_T, tid, col_base + 0);
                float p1 = read_swizzled_128B_fp32_transposed(s_P_T, tid, col_base + 1);
                float p2 = read_swizzled_128B_fp32_transposed(s_P_T, tid, col_base + 2);
                float p3 = read_swizzled_128B_fp32_transposed(s_P_T, tid, col_base + 3);
                float p4 = read_swizzled_128B_fp32_transposed(s_P_T, tid, col_base + 4);
                float p5 = read_swizzled_128B_fp32_transposed(s_P_T, tid, col_base + 5);
                float p6 = read_swizzled_128B_fp32_transposed(s_P_T, tid, col_base + 6);
                float p7 = read_swizzled_128B_fp32_transposed(s_P_T, tid, col_base + 7);
                
                float ds0 = p0 * (dp0 - s_D[tid]);
                float ds1 = p1 * (dp1 - s_D[tid]);
                float ds2 = p2 * (dp2 - s_D[tid]);
                float ds3 = p3 * (dp3 - s_D[tid]);
                float ds4 = p4 * (dp4 - s_D[tid]);
                float ds5 = p5 * (dp5 - s_D[tid]);
                float ds6 = p6 * (dp6 - s_D[tid]);
                float ds7 = p7 * (dp7 - s_D[tid]);
                
                SwizzledStore dS_store((uint8_t*)s_dS_T, tid, col_base);
                dS_store.write(col_base + 0, ds0);
                dS_store.write(col_base + 1, ds1);
                dS_store.write(col_base + 2, ds2);
                dS_store.write(col_base + 3, ds3);
                dS_store.write(col_base + 4, ds4);
                dS_store.write(col_base + 5, ds5);
                dS_store.write(col_base + 6, ds6);
                dS_store.write(col_base + 7, ds7);
            }
        }
        __syncthreads();

        uint32_t accum_dV = 1;
        for (int K_walker = 0; K_walker < 8; K_walker++) {
            uint64_t desc_a = make_smem_desc_k_major_64((uint8_t*)s_P_T + K_walker * 32);
            uint64_t desc_b = make_smem_desc_n_major_64((uint8_t*)s_dO_0 + K_walker * 2048);
            uint32_t idesc_dV = make_instr_desc_fn(128, 64, 0, 1);
            umma_f16_cg2_fn(tmem_add_cols(tmem_dV_0, K_walker * 16), desc_a, desc_b, idesc_dV, accum_dV);
            accum_dV = 1;
        }
        for (int K_walker = 0; K_walker < 8; K_walker++) {
            uint64_t desc_a = make_smem_desc_k_major_64((uint8_t*)s_P_T + K_walker * 32 + 8192);
            uint64_t desc_b = make_smem_desc_n_major_64((uint8_t*)s_dO_1 + K_walker * 2048);
            uint32_t idesc_dV = make_instr_desc_fn(128, 64, 0, 1);
            umma_f16_cg2_fn(tmem_add_cols(tmem_dV_1, K_walker * 16), desc_a, desc_b, idesc_dV, accum_dV);
            accum_dV = 1;
        }

        uint32_t accum_dK = 1;
        for (int K_walker = 0; K_walker < 8; K_walker++) {
            uint64_t desc_a = make_smem_desc_k_major_64((uint8_t*)s_dS_T + K_walker * 32);
            uint64_t desc_b = make_smem_desc_n_major_64((uint8_t*)s_Q_0 + K_walker * 2048);
            uint32_t idesc_dK = make_instr_desc_fn(128, 64, 0, 1);
            umma_f16_cg2_fn(tmem_add_cols(tmem_dK_0, K_walker * 16), desc_a, desc_b, idesc_dK, accum_dK);
            accum_dK = 1;
        }
        for (int K_walker = 0; K_walker < 8; K_walker++) {
            uint64_t desc_a = make_smem_desc_k_major_64((uint8_t*)s_dS_T + K_walker * 32 + 8192);
            uint64_t desc_b = make_smem_desc_n_major_64((uint8_t*)s_Q_1 + K_walker * 2048);
            uint32_t idesc_dK = make_instr_desc_fn(128, 64, 0, 1);
            umma_f16_cg2_fn(tmem_add_cols(tmem_dK_1, K_walker * 16), desc_a, desc_b, idesc_dK, accum_dK);
            accum_dK = 1;
        }
        
        umma_commit_2sm_fn(mbar);
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;

        if (tid < 128) {
            for (int col = 0; col < 128; col += 16) {
                uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                    : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),
                      "=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(tmem_add_cols(tmem_dV_0, col)));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float f0 = __uint_as_float(r0);
                float f1 = __uint_as_float(r1);
                float f2 = __uint_as_float(r2);
                float f3 = __uint_as_float(r3);
                float f4 = __uint_as_float(r4);
                float f5 = __uint_as_float(r5);
                float f6 = __uint_as_float(r6);
                float f7 = __uint_as_float(r7);
                
                uint32_t u0 = pack_bf16_fn(f0, f1);
                uint32_t u1 = pack_bf16_fn(f2, f3);
                uint32_t u2 = pack_bf16_fn(f4, f5);
                uint32_t u3 = pack_bf16_fn(f6, f7);
                int byte_off = get_swizzle_offset_128x128(col, tid);
                asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" :: "r"(byte_off), "r"(u0), "r"(u1), "r"(u2), "r"(u3));
            }
            
            for (int col = 0; col < 128; col += 16) {
                uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                    : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),
                      "=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(tmem_add_cols(tmem_dV_1, col)));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float f0 = __uint_as_float(r0);
                float f1 = __uint_as_float(r1);
                float f2 = __uint_as_float(r2);
                float f3 = __uint_as_float(r3);
                float f4 = __uint_as_float(r4);
                float f5 = __uint_as_float(r5);
                float f6 = __uint_as_float(r6);
                float f7 = __uint_as_float(r7);
                
                uint32_t u0 = pack_bf16_fn(f0, f1);
                uint32_t u1 = pack_bf16_fn(f2, f3);
                uint32_t u2 = pack_bf16_fn(f4, f5);
                uint32_t u3 = pack_bf16_fn(f6, f7);
                int byte_off = get_swizzle_offset_128x128(col, tid);
                asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" :: "r"(byte_off), "r"(u0), "r"(u1), "r"(u2), "r"(u3));
            }

            for (int col = 0; col < 128; col += 16) {
                uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                    : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),
                      "=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(tmem_add_cols(tmem_dK_0, col)));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float f0 = __uint_as_float(r0);
                float f1 = __uint_as_float(r1);
                float f2 = __uint_as_float(r2);
                float f3 = __uint_as_float(r3);
                float f4 = __uint_as_float(r4);
                float f5 = __uint_as_float(r5);
                float f6 = __uint_as_float(r6);
                float f7 = __uint_as_float(r7);
                
                uint32_t u0 = pack_bf16_fn(f0, f1);
                uint32_t u1 = pack_bf16_fn(f2, f3);
                uint32_t u2 = pack_bf16_fn(f4, f5);
                uint32_t u3 = pack_bf16_fn(f6, f7);
                int byte_off = get_swizzle_offset_128x128(col, tid);
                asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" :: "r"(byte_off), "r"(u0), "r"(u1), "r"(u2), "r"(u3));
            }
            
            for (int col = 0; col < 128; col += 16) {
                uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                    : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),
                      "=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(tmem_add_cols(tmem_dK_1, col)));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float f0 = __uint_as_float(r0);
                float f1 = __uint_as_float(r1);
                float f2 = __uint_as_float(r2);
                float f3 = __uint_as_float(r3);
                float f4 = __uint_as_float(r4);
                float f5 = __uint_as_float(r5);
                float f6 = __uint_as_float(r6);
                float f7 = __uint_as_float(r7);
                
                uint32_t u0 = pack_bf16_fn(f0, f1);
                uint32_t u1 = pack_bf16_fn(f2, f3);
                uint32_t u2 = pack_bf16_fn(f4, f5);
                uint32_t u3 = pack_bf16_fn(f6, f7);
                int byte_off = get_swizzle_offset_128x128(col, tid);
                asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" :: "r"(byte_off), "r"(u0), "r"(u1), "r"(u2), "r"(u3));
            }
        }
        __syncthreads();

        if (tid < 128) {
            for (int col = 0; col < 64; col += 16) {
                uint4 dv0 = read_swizzled_vec_32(s_V_0, tid, col);
                uint4 dv1 = read_swizzled_vec_32(s_V_1, tid, col);
                int global_row = k_start + tid;
                if (global_row < S) {
                    int outer_offset = (b_head * S + global_row) * d_dim;
                    uint4* out0 = reinterpret_cast<uint4*>(&dV[outer_offset + col]);
                    uint4* out1 = reinterpret_cast<uint4*>(&dV[outer_offset + col + 64]);
                    *out0 = dv0;
                    *out1 = dv1;
                }
            }
            for (int col = 0; col < 64; col += 16) {
                uint4 dk0 = read_swizzled_vec_32(s_K_0, tid, col);
                uint4 dk1 = read_swizzled_vec_32(s_K_1, tid, col);
                int global_row = k_start + tid;
                if (global_row < S) {
                    int outer_offset = (b_head * S + global_row) * d_dim;
                    uint4* out0 = reinterpret_cast<uint4*>(&dK[outer_offset + col]);
                    uint4* out1 = reinterpret_cast<uint4*>(&dK[outer_offset + col + 64]);
                    *out0 = dk0;
                    *out1 = dk1;
                }
            }
        }
        
        __syncthreads(); 
    }
}

__global__ void __launch_bounds__(128) bwd_kernel_dq(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_dO,
    const __grid_constant__ CUtensorMap tma_O,
    __nv_bfloat16* dQ,
    const float* L,
    int S, int d_dim, int H)
{
    extern __shared__ __align__(128) uint8_t smem_raw[];
    uint8_t* s_K_0 = smem_raw + 0;          
    uint8_t* s_K_1 = smem_raw + 16384;      
    uint8_t* s_V_0 = smem_raw + 32768;      
    uint8_t* s_V_1 = smem_raw + 49152;      
    uint8_t* s_Q_0 = smem_raw + 65536;      
    uint8_t* s_Q_1 = smem_raw + 81920;      
    uint8_t* s_dO_0 = smem_raw + 98304;     
    uint8_t* s_dO_1 = smem_raw + 114688;    
    uint8_t* s_O_0 = smem_raw + 131072;     
    uint8_t* s_O_1 = smem_raw + 147456;     
    
    float* s_D = (float*)(smem_raw + 163840);
    float* s_L = (float*)(smem_raw + 164352);
    
    uint8_t* s_P_T = smem_raw + 164864; 
    uint8_t* s_dS_T = smem_raw + 197632; 

    uint32_t* smem_tmem_S    = (uint32_t*)(smem_raw + 230400);
    uint32_t* smem_tmem_dP   = (uint32_t*)(smem_raw + 230408);
    uint32_t* smem_tmem_dQ_0 = (uint32_t*)(smem_raw + 230416);
    uint32_t* smem_tmem_dQ_1 = (uint32_t*)(smem_raw + 230424);

    int q_blk = blockIdx.x;
    int b_head = blockIdx.y;
    int b_idx = b_head / H;
    int h_idx = b_head % H;
    int tid = threadIdx.x;

    int q_start = q_blk * 128;

    if (threadIdx.x == 0) {
        tmem_alloc_fn(smem_tmem_S, 128);
        tmem_alloc_fn(smem_tmem_dP, 128);
        tmem_alloc_fn(smem_tmem_dQ_0, 128);
        tmem_alloc_fn(smem_tmem_dQ_1, 128);

        init_smem_barrier_fn(mbar, 1); 
        fence_smem_barrier_init_fn();
        
        mbarrier_arrive_and_expect_tx_fn(mbar, 98304);
        tma_load_4d_fn(&tma_Q, mbar, s_Q_0, 0, q_start, h_idx, b_idx);
        tma_load_4d_fn(&tma_Q, mbar, s_Q_1, 64, q_start, h_idx, b_idx);
        tma_load_4d_fn(&tma_dO, mbar, s_dO_0, 0, q_start, h_idx, b_idx);
        tma_load_4d_fn(&tma_dO, mbar, s_dO_1, 64, q_start, h_idx, b_idx);
        tma_load_4d_fn(&tma_O, mbar, s_O_0, 0, q_start, h_idx, b_idx);
        tma_load_4d_fn(&tma_O, mbar, s_O_1, 64, q_start, h_idx, b_idx);
    }
    
    int phase = 0;
    mbarrier_wait_fn(mbar, phase);
    phase ^= 1;

    uint32_t tmem_S = smem_tmem_S[0];
    uint32_t tmem_dP = smem_tmem_dP[0];
    uint32_t tmem_dQ_0 = smem_tmem_dQ_0[0];
    uint32_t tmem_dQ_1 = smem_tmem_dQ_1[0];

    if (threadIdx.x < 128) {
        fill_tmem_128x128_fp32(tmem_dQ_0, 0);
        fill_tmem_128x128_fp32(tmem_dQ_1, 0);
    }
    __syncthreads();

    if (tid < 128) {
        int global_row = q_start + tid;
        s_L[tid] = (global_row < S) ? L[b_idx * H * S + h_idx * S + global_row] : 0.0f;
        
        float sum = 0.0f;
        if (global_row < S) {
            uint4 ov0 = read_swizzled_vec_32_oob(s_O_0, tid, 0, 64);
            uint4 dv0 = read_swizzled_vec_32_oob(s_dO_0, tid, 0, 64);
            uint4 ov1 = read_swizzled_vec_32_oob(s_O_1, tid, 0, 64);
            uint4 dv1 = read_swizzled_vec_32_oob(s_dO_1, tid, 0, 64);

            uint32_t* of0 = (uint32_t*)&ov0;
            uint32_t* dof0 = (uint32_t*)&dv0;
            uint32_t* of1 = (uint32_t*)&ov1;
            uint32_t* dof1 = (uint32_t*)&dv1;

            for (int i = 0; i < 4; i++) {
                __nv_bfloat162 o_val0 = *(__nv_bfloat162*)&of0[i];
                __nv_bfloat162 do_val0 = *(__nv_bfloat162*)&dof0[i];
                sum += __bfloat162float(o_val0.x) * __bfloat162float(do_val0.x);
                sum += __bfloat162float(o_val0.y) * __bfloat162float(do_val0.y);
                
                __nv_bfloat162 o_val1 = *(__nv_bfloat162*)&of1[i];
                __nv_bfloat162 do_val1 = *(__nv_bfloat162*)&dof1[i];
                sum += __bfloat162float(o_val1.x) * __bfloat162float(do_val1.x);
                sum += __bfloat162float(o_val1.y) * __bfloat162float(do_val1.y);
            }
        }
        s_D[tid] = sum;
    }
    __syncthreads();

    int num_q_blks = (S + 127) / 128;
    float attn_scale = 1.0f / sqrtf((float)d_dim);

    for (int k_blk = 0; k_blk < num_q_blks; k_blk++) {
        int k_start = k_blk * 128;
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar, 65536);
            tma_load_4d_fn(&tma_K, mbar, s_K_0, 0, k_start, h_idx, b_idx);
            tma_load_4d_fn(&tma_K, mbar, s_K_1, 64, k_start, h_idx, b_idx);
            tma_load_4d_fn(&tma_V, mbar, s_V_0, 0, k_start, h_idx, b_idx);
            tma_load_4d_fn(&tma_V, mbar, s_V_1, 64, k_start, h_idx, b_idx);
        }
        
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;

        uint32_t accum_S = 0;
        for (int K_walker = 0; K_walker < 4; K_walker++) {
            uint64_t desc_a = make_smem_desc_k_major_64((uint8_t*)s_K_0 + K_walker * 32);
            uint64_t desc_b = make_smem_desc_k_major_64((uint8_t*)s_Q_0 + K_walker * 32);
            uint32_t idesc_S = make_instr_desc_fn(256, 128, 0, 0);
            umma_f16_cg2_fn(tmem_add_cols(tmem_S, K_walker * 16), desc_a, desc_b, idesc_S, accum_S);
            accum_S = 1;
        }
        for (int K_walker = 0; K_walker < 4; K_walker++) {
            uint64_t desc_a = make_smem_desc_k_major_64((uint8_t*)s_K_1 + K_walker * 32);
            uint64_t desc_b = make_smem_desc_k_major_64((uint8_t*)s_Q_1 + K_walker * 32);
            uint32_t idesc_S = make_instr_desc_fn(256, 128, 0, 0);
            umma_f16_cg2_fn(tmem_add_cols(tmem_S, 64 + K_walker * 16), desc_a, desc_b, idesc_S, accum_S);
        }

        uint32_t accum_dP = 0;
        for (int K_walker = 0; K_walker < 4; K_walker++) {
            uint64_t desc_a = make_smem_desc_k_major_64((uint8_t*)s_V_0 + K_walker * 32);
            uint64_t desc_b = make_smem_desc_k_major_64((uint8_t*)s_dO_0 + K_walker * 32);
            uint32_t idesc_dP = make_instr_desc_fn(256, 128, 0, 0);
            umma_f16_cg2_fn(tmem_add_cols(tmem_dP, K_walker * 16), desc_a, desc_b, idesc_dP, accum_dP);
            accum_dP = 1;
        }
        for (int K_walker = 0; K_walker < 4; K_walker++) {
            uint64_t desc_a = make_smem_desc_k_major_64((uint8_t*)s_V_1 + K_walker * 32);
            uint64_t desc_b = make_smem_desc_k_major_64((uint8_t*)s_dO_1 + K_walker * 32);
            uint32_t idesc_dP = make_instr_desc_fn(256, 128, 0, 0);
            umma_f16_cg2_fn(tmem_add_cols(tmem_dP, 64 + K_walker * 16), desc_a, desc_b, idesc_dP, accum_dP);
        }

        umma_commit_2sm_fn(mbar);
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;

        if (tid < 128) {
            for (int col_base = 0; col_base < 128; col_base += 16) {
                uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                    : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),
                      "=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(tmem_add_cols(tmem_S, col_base)));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                
                float s0 = __uint_as_float(r0);
                float s1 = __uint_as_float(r1);
                float s2 = __uint_as_float(r2);
                float s3 = __uint_as_float(r3);
                float s4 = __uint_as_float(r4);
                float s5 = __uint_as_float(r5);
                float s6 = __uint_as_float(r6);
                float s7 = __uint_as_float(r7);
                
                float p0 = __expf(s0 * attn_scale - s_L[tid]);
                float p1 = __expf(s1 * attn_scale - s_L[tid]);
                float p2 = __expf(s2 * attn_scale - s_L[tid]);
                float p3 = __expf(s3 * attn_scale - s_L[tid]);
                float p4 = __expf(s4 * attn_scale - s_L[tid]);
                float p5 = __expf(s5 * attn_scale - s_L[tid]);
                float p6 = __expf(s6 * attn_scale - s_L[tid]);
                float p7 = __expf(s7 * attn_scale - s_L[tid]);
                
                if (k_start + tid >= S || q_start + col_base >= S) {
                    p0 = 0; p1 = 0; p2 = 0; p3 = 0;
                    p4 = 0; p5 = 0; p6 = 0; p7 = 0;
                }
                
                SwizzledStore P_store((uint8_t*)s_P_T, tid, col_base);
                P_store.write(col_base + 0, p0);
                P_store.write(col_base + 1, p1);
                P_store.write(col_base + 2, p2);
                P_store.write(col_base + 3, p3);
                P_store.write(col_base + 4, p4);
                P_store.write(col_base + 5, p5);
                P_store.write(col_base + 6, p6);
                P_store.write(col_base + 7, p7);
            }
        }
        
        if (tid < 128) {
            for (int col_base = 0; col_base < 128; col_base += 16) {
                uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                    : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),
                      "=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(tmem_add_cols(tmem_dP, col_base)));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                
                float dp0 = __uint_as_float(r0);
                float dp1 = __uint_as_float(r1);
                float dp2 = __uint_as_float(r2);
                float dp3 = __uint_as_float(r3);
                float dp4 = __uint_as_float(r4);
                float dp5 = __uint_as_float(r5);
                float dp6 = __uint_as_float(r6);
                float dp7 = __uint_as_float(r7);
                
                float p0 = read_swizzled_128B_fp32_transposed(s_P_T, tid, col_base + 0);
                float p1 = read_swizzled_128B_fp32_transposed(s_P_T, tid, col_base + 1);
                float p2 = read_swizzled_128B_fp32_transposed(s_P_T, tid, col_base + 2);
                float p3 = read_swizzled_128B_fp32_transposed(s_P_T, tid, col_base + 3);
                float p4 = read_swizzled_128B_fp32_transposed(s_P_T, tid, col_base + 4);
                float p5 = read_swizzled_128B_fp32_transposed(s_P_T, tid, col_base + 5);
                float p6 = read_swizzled_128B_fp32_transposed(s_P_T, tid, col_base + 6);
                float p7 = read_swizzled_128B_fp32_transposed(s_P_T, tid, col_base + 7);
                
                float ds0 = p0 * (dp0 - s_D[tid]);
                float ds1 = p1 * (dp1 - s_D[tid]);
                float ds2 = p2 * (dp2 - s_D[tid]);
                float ds3 = p3 * (dp3 - s_D[tid]);
                float ds4 = p4 * (dp4 - s_D[tid]);
                float ds5 = p5 * (dp5 - s_D[tid]);
                float ds6 = p6 * (dp6 - s_D[tid]);
                float ds7 = p7 * (dp7 - s_D[tid]);
                
                SwizzledStore dS_store((uint8_t*)s_dS_T, tid, col_base);
                dS_store.write(col_base + 0, ds0);
                dS_store.write(col_base + 1, ds1);
                dS_store.write(col_base + 2, ds2);
                dS_store.write(col_base + 3, ds3);
                dS_store.write(col_base + 4, ds4);
                dS_store.write(col_base + 5, ds5);
                dS_store.write(col_base + 6, ds6);
                dS_store.write(col_base + 7, ds7);
            }
        }
        __syncthreads();

        uint32_t accum_dQ_0 = 1;
        for (int K_walker = 0; K_walker < 8; K_walker++) {
            uint64_t desc_a = make_smem_desc_k_major_64((uint8_t*)s_dS_T + K_walker * 32);
            uint64_t desc_b = make_smem_desc_n_major_64((uint8_t*)s_K_0 + K_walker * 2048);
            uint32_t idesc_dQ = make_instr_desc_fn(128, 64, 0, 1);
            umma_f16_cg2_fn(tmem_add_cols(tmem_dQ_0, K_walker * 16), desc_a, desc_b, idesc_dQ, accum_dQ_0);
            accum_dQ_0 = 1;
        }

        uint32_t accum_dQ_1 = 1;
        for (int K_walker = 0; K_walker < 8; K_walker++) {
            uint64_t desc_a = make_smem_desc_k_major_64((uint8_t*)s_dS_T + K_walker * 32 + 8192);
            uint64_t desc_b = make_smem_desc_n_major_64((uint8_t*)s_K_1 + K_walker * 2048);
            uint32_t idesc_dQ = make_instr_desc_fn(128, 64, 0, 1);
            umma_f16_cg2_fn(tmem_add_cols(tmem_dQ_1, K_walker * 16), desc_a, desc_b, idesc_dQ, accum_dQ_1);
            accum_dQ_1 = 1;
        }
        
        umma_commit_2sm_fn(mbar);
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
    }

    if (tid < 128) {
        for (int col = 0; col < 128; col += 16) {
            uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),
                  "=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(tmem_add_cols(tmem_dQ_0, col)));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float f0 = __uint_as_float(r0);
            float f1 = __uint_as_float(r1);
            float f2 = __uint_as_float(r2);
            float f3 = __uint_as_float(r3);
            float f4 = __uint_as_float(r4);
            float f5 = __uint_as_float(r5);
            float f6 = __uint_as_float(r6);
            float f7 = __uint_as_float(r7);
            
            uint32_t u0 = pack_bf16_fn(f0, f1);
            uint32_t u1 = pack_bf16_fn(f2, f3);
            uint32_t u2 = pack_bf16_fn(f4, f5);
            uint32_t u3 = pack_bf16_fn(f6, f7);
            int byte_off = get_swizzle_offset_128x128(col, tid);
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" :: "r"(byte_off), "r"(u0), "r"(u1), "r"(u2), "r"(u3));
        }
        
        for (int col = 0; col < 128; col += 16) {
            uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),
                  "=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(tmem_add_cols(tmem_dQ_1, col)));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float f0 = __uint_as_float(r0);
            float f1 = __uint_as_float(r1);
            float f2 = __uint_as_float(r2);
            float f3 = __uint_as_float(r3);
            float f4 = __uint_as_float(r4);
            float f5 = __uint_as_float(r5);
            float f6 = __uint_as_float(r6);
            float f7 = __uint_as_float(r7);
            
            uint32_t u0 = pack_bf16_fn(f0, f1);
            uint32_t u1 = pack_bf16_fn(f2, f3);
            uint32_t u2 = pack_bf16_fn(f4, f5);
            uint32_t u3 = pack_bf16_fn(f6, f7);
            int byte_off = get_swizzle_offset_128x128(col, tid);
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" :: "r"(byte_off), "r"(u0), "r"(u1), "r"(u2), "r"(u3));
        }
    }
    __syncthreads(); 
    
    if (tid < 128) {
        for (int col = 0; col < 64; col += 16) {
            uint4 dq0 = read_swizzled_vec_32(s_Q_0, tid, col);
            uint4 dq1 = read_swizzled_vec_32(s_Q_1, tid, col);
            int global_row = q_start + tid;
            if (global_row < S) {
                int outer_offset_q = (b_head * S + global_row) * d_dim;
                uint4* out0_q = reinterpret_cast<uint4*>(&dQ[outer_offset_q + col]);
                uint4* out1_q = reinterpret_cast<uint4*>(&dQ[outer_offset_q + col + 64]);
                *out0_q = dq0;
                *out1_q = dq1;
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
    
    int smem_size_dv_dk = 164864 + 32 + 230464 - 164864 + 128;
    cudaFuncSetAttribute(bwd_kernel_dv_dk, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_dv_dk);
    
    int smem_size_dq = 245760;
    cudaFuncSetAttribute(bwd_kernel_dq, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_dq);
    
    int num_q_blks = (S + 127) / 128;
    int num_kv_blks = (S + 127) / 128;
    dim3 grid_dv_dk(num_kv_blks, B * H, 1);
    dim3 grid_dq(num_q_blks, B * H, 1);
    dim3 block(128, 1, 1);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    cudaLaunchConfig_t config_dv_dk = {};
    config_dv_dk.gridDim = grid_dv_dk;
    config_dv_dk.blockDim = block;
    config_dv_dk.dynamicSmemBytes = smem_size_dv_dk;
    config_dv_dk.stream = stream;
    
    cudaLaunchConfig_t config_dq = {};
    config_dq.gridDim = grid_dq;
    config_dq.blockDim = block;
    config_dq.dynamicSmemBytes = smem_size_dq;
    config_dq.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config_dv_dk.attrs = attrs;
    config_dv_dk.numAttrs = 1;
    config_dq.attrs = attrs;
    config_dq.numAttrs = 1;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config_dv_dk, bwd_kernel_dv_dk, 
        tma_Q, tma_K, tma_V, tma_dO, tma_O, 
        static_cast<__nv_bfloat16*>(dK.data_ptr()), 
        static_cast<__nv_bfloat16*>(dV.data_ptr()), 
        static_cast<const float*>(L.data_ptr()), 
        S, d, H));
        
    CUDA_CHECK(cudaLaunchKernelEx(&config_dq, bwd_kernel_dq, 
        tma_Q, tma_K, tma_V, tma_dO, tma_O, 
        static_cast<__nv_bfloat16*>(dQ.data_ptr()), 
        static_cast<const float*>(L.data_ptr()), 
        S, d, H));
        
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi