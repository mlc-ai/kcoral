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

#define CU_CHECK(call) do {                                      \
    CUresult _e = (call);                                        \
    if (_e != CUDA_SUCCESS) {                                    \
        const char* err_name = "Unknown";                         \
        cuGetErrorName(_e, &err_name);                           \
        fprintf(stderr, "CUresult error %s at %s:%d\n",           \
                err_name, __FILE__, __LINE__);                   \
        exit(1);                                                  \
    }                                                            \
} while(0)

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void expect_tx(uint64_t* bar, uint32_t bytes, uint32_t& phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %2;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(bytes), "r"(phase));
    
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(bytes) : "memory");
    
    phase ^= 1;
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

__device__ __forceinline__ void tma_load_2d_swizzled(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tmem_store_commit_fn() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void tmem_store_wait_fn() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ uint64_t make_smem_desc(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_f16_cg1(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ void read_tmem_128x64(int tid, uint32_t* tmem_ptr, float (*out)[4][16][16], int row_start, int col_start) {
    int warp_row_base = (tid / 32) * 2;
    int start_row = row_start + warp_row_base * 16;
    int start_col = col_start;
    for(int j = 0; j < 4; j++) {
        uint32_t tmem_addr = *reinterpret_cast<uint32_t*>(tmem_ptr) + (start_col + j * 16);
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_addr));
        for(int i = 0; i < 2; i++) {
            int row = start_row + i * 16;
            out[i][j][threadIdx.x % 16][0] = __uint_as_float(r0);
            out[i][j][threadIdx.x % 16][1] = __uint_as_float(r1);
            out[i][j][threadIdx.x % 16][2] = __uint_as_float(r2);
            out[i][j][threadIdx.x % 16][3] = __uint_as_float(r3);
            if (r3 != 0) {} 
        }
    }
}

__device__ __forceinline__ void write_swizzled_bf16(__nv_bfloat16* smem, int row, int col, __nv_bfloat16 val) {
    int col_chunk = col / 8;
    int chunk_offset = col % 8;
    int swizzled_col_chunk = (row % 8) ^ col_chunk;
    int swizzled_col = swizzled_col_chunk * 8 + chunk_offset;
    smem[row * 64 + swizzled_col] = val;
}

__device__ __forceinline__ __nv_bfloat16 read_swizzled_bf16(__nv_bfloat16* smem, int row, int col) {
    int col_chunk = col / 8;
    int chunk_offset = col % 8;
    int swizzled_col_chunk = (row % 8) ^ col_chunk;
    int swizzled_col = swizzled_col_chunk * 8 + chunk_offset;
    return smem[row * 64 + swizzled_col];
}

__device__ __forceinline__ void transpose_smem_128x64_swizzled(__nv_bfloat16* smem_A, __nv_bfloat16* smem_B_T) {
    for (int i = threadIdx.x; i < 128 * 64; i += blockDim.x) {
        int r = i / 64;
        int c = i % 64;
        __nv_bfloat16 val = read_swizzled_bf16(smem_A, r, c);
        write_swizzled_bf16(smem_B_T, c, r, val);
    }
    __syncthreads();
}

__device__ __forceinline__ void gemm_128x64x64(uint32_t tmem_c_base, uint32_t idesc, __nv_bfloat16* smem_A, __nv_bfloat16* smem_B) {
    int acc = 1;
    for (int k_tile = 0; k_tile < 4; ++k_tile) {
        for (int j = 0; j < 4; ++j) {
            uint32_t tmem_c = tmem_c_base + k_tile * 2; 
            uint64_t desc_a = make_smem_desc(smem_A + k_tile * 16, 1, 1024);
            uint64_t desc_b = make_smem_desc(smem_B + k_tile * 16 * 64 + j * 16, 1024, 1024);
            
            asm volatile("{\n.reg .pred p;\n"
                         "setp.ne.b32 p, %4, 0;\n"
                         "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                         :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(acc));
        }
    }
}

__device__ __forceinline__ void gemm_128x64x64_accumulate(uint32_t tmem_c_base, uint32_t idesc, __nv_bfloat16* smem_A, __nv_bfloat16* smem_B) {
    int acc = 1;
    for (int k_tile = 0; k_tile < 4; ++k_tile) {
        for (int j = 0; j < 4; ++j) {
            uint32_t tmem_c = tmem_c_base + k_tile * 2; 
            uint64_t desc_a = make_smem_desc(smem_A + k_tile * 16, 1, 1024);
            uint64_t desc_b = make_smem_desc(smem_B + k_tile * 16 * 64 + j * 16, 1024, 1024);
            
            asm volatile("{\n.reg .pred p;\n"
                         "setp.ne.b32 p, %4, 0;\n"
                         "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                         :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(acc));
        }
    }
}

extern __shared__ __align__(128) uint8_t smem_pool[];

__device__ void setup_smem_ptrs(
    __nv_bfloat16*& smem_Q0, __nv_bfloat16*& smem_Q1,
    __nv_bfloat16*& smem_K0, __nv_bfloat16*& smem_K1,
    __nv_bfloat16*& smem_V0, __nv_bfloat16*& smem_V1,
    __nv_bfloat16*& smem_O0, __nv_bfloat16*& smem_O1,
    __nv_bfloat16*& smem_dO0, __nv_bfloat16*& smem_dO1,
    __nv_bfloat16*& smem_dP0, __nv_bfloat16*& smem_dP1,
    __nv_bfloat16*& smem_P0, __nv_bfloat16*& smem_P1,
    __nv_bfloat16*& smem_D0, __nv_bfloat16*& smem_D1,
    float*& smem_L, uint64_t*& bar_Q, uint64_t*& bar_K,
    uint64_t*& bar_V, uint64_t*& bar_O, uint64_t*& bar_dO) 
{
    uint8_t* p = smem_pool;
    
    smem_Q0  = reinterpret_cast<__nv_bfloat16*>(p); p += 16384;
    smem_Q1  = reinterpret_cast<__nv_bfloat16*>(p); p += 16384;
    smem_K0  = reinterpret_cast<__nv_bfloat16*>(p); p += 16384;
    smem_K1  = reinterpret_cast<__nv_bfloat16*>(p); p += 16384;
    smem_V0  = reinterpret_cast<__nv_bfloat16*>(p); p += 16384;
    smem_V1  = reinterpret_cast<__nv_bfloat16*>(p); p += 16384;
    smem_O0  = reinterpret_cast<__nv_bfloat16*>(p); p += 16384;
    smem_O1  = reinterpret_cast<__nv_bfloat16*>(p); p += 16384;
    smem_dO0 = reinterpret_cast<__nv_bfloat16*>(p); p += 16384;
    smem_dO1 = reinterpret_cast<__nv_bfloat16*>(p); p += 16384;
    smem_dP0 = reinterpret_cast<__nv_bfloat16*>(p); p += 16384;
    smem_dP1 = reinterpret_cast<__nv_bfloat16*>(p); p += 16384;
    smem_P0  = reinterpret_cast<__nv_bfloat16*>(p); p += 16384;
    smem_P1  = reinterpret_cast<__nv_bfloat16*>(p); p += 16384;
    smem_D0  = reinterpret_cast<__nv_bfloat16*>(p); p += 16384;
    smem_D1  = reinterpret_cast<__nv_bfloat16*>(p); p += 16384;
    
    p = (uint8_t*)(((uintptr_t)p + 255) & ~255);
    
    smem_L = reinterpret_cast<float*>(p); p += 512;
    
    p = (uint8_t*)(((uintptr_t)p + 255) & ~255);
    
    bar_Q = reinterpret_cast<uint64_t*>(p); p += 8;
    bar_K = reinterpret_cast<uint64_t*>(p); p += 8;
    bar_V = reinterpret_cast<uint64_t*>(p); p += 8;
    bar_O = reinterpret_cast<uint64_t*>(p); p += 8;
    bar_dO= reinterpret_cast<uint64_t*>(p); p += 8;
}

__global__ void attn_backward_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* L_ptr,
    __nv_bfloat16* dQ_ptr,
    __nv_bfloat16* dK_ptr,
    __nv_bfloat16* dV_ptr,
    uint32_t S, float scale) 
{
    extern __shared__ uint32_t tmem_S[];
    extern __shared__ uint32_t tmem_dQ[];
    extern __shared__ uint32_t tmem_dK[];
    extern __shared__ uint32_t tmem_dV[];

    uint32_t q_tile = blockIdx.x;
    uint32_t b_h_idx = blockIdx.y;
    uint32_t q_start = q_tile * 128;
    uint32_t phase = 0;
    int tid = threadIdx.x;

    if (q_start >= S) return;

    __nv_bfloat16 *smem_Q0, *smem_Q1, *smem_K0, *smem_K1, *smem_V0, *smem_V1, *smem_O0, *smem_O1, *smem_dO0, *smem_dO1;
    __nv_bfloat16 *smem_dP0, *smem_dP1, *smem_P0, *smem_P1, *smem_D0, *smem_D1;
    float *smem_L;
    uint64_t *bar_Q, *bar_K, *bar_V, *bar_O, *bar_dO;

    setup_smem_ptrs(smem_Q0, smem_Q1, smem_K0, smem_K1, smem_V0, smem_V1, smem_O0, smem_O1, smem_dO0, smem_dO1,
                    smem_dP0, smem_dP1, smem_P0, smem_P1, smem_D0, smem_D1,
                    smem_L, bar_Q, bar_K, bar_V, bar_O, bar_dO);

    if (tid == 0) {
        init_smem_barrier_fn(bar_Q, 1);
        init_smem_barrier_fn(bar_K, 1);
        init_smem_barrier_fn(bar_V, 1);
        init_smem_barrier_fn(bar_O, 1);
        init_smem_barrier_fn(bar_dO, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    if (tid == 0) {
        tmem_alloc_fn(tmem_S, 64);
        tmem_alloc_fn(tmem_dQ, 128);
    }
    __syncthreads();

    // === Pass 1: Compute dQ ===
    uint32_t total_tiles = (S + 127) / 128;
    uint32_t bh_offset = b_h_idx * S;

    if (tid == 0) {
        expect_tx(bar_Q, 32768, phase);
        tma_load_2d_swizzled(&tma_Q, bar_Q, smem_Q0, 0, bh_offset + q_start);
        tma_load_2d_swizzled(&tma_Q, bar_Q, smem_Q1, 64, bh_offset + q_start);
    }
    mbarrier_wait_fn(bar_Q, phase);
    if (tid < 128) {
        smem_L[tid] = (q_start + tid < S) ? L_ptr[b_h_idx * S + q_start + tid] : 0.0f;
    }
    __syncthreads();

    uint32_t tmem_dQ_base = *reinterpret_cast<uint32_t*>(tmem_dQ);
    uint32_t tmem_S_base = *reinterpret_cast<uint32_t*>(tmem_S);
    uint32_t idesc_S = make_instr_desc_f16_cg1(128, 64);
    idesc_S |= (1u << 16); 

    for (int kv_tile = 0; kv_tile < total_tiles; kv_tile++) {
        uint32_t kv_start = kv_tile * 128;
        if (tid == 0) {
            expect_tx(bar_K, 32768, phase);
            tma_load_2d_swizzled(&tma_K, bar_K, smem_K0, 0, bh_offset + kv_start);
            tma_load_2d_swizzled(&tma_K, bar_K, smem_K1, 64, bh_offset + kv_start);

            expect_tx(bar_V, 32768, phase);
            tma_load_2d_swizzled(&tma_V, bar_V, smem_V0, 0, bh_offset + kv_start);
            tma_load_2d_swizzled(&tma_V, bar_V, smem_V1, 64, bh_offset + kv_start);

            expect_tx(bar_O, 32768, phase);
            tma_load_2d_swizzled(&tma_O, bar_O, smem_O0, 0, bh_offset + kv_start);
            tma_load_2d_swizzled(&tma_O, bar_O, smem_O1, 64, bh_offset + kv_start);

            expect_tx(bar_dO, 32768, phase);
            tma_load_2d_swizzled(&tma_dO, bar_dO, smem_dO0, 0, bh_offset + kv_start);
            tma_load_2d_swizzled(&tma_dO, bar_dO, smem_dO1, 64, bh_offset + kv_start);
        }
        mbarrier_wait_fn(bar_K, phase);
        mbarrier_wait_fn(bar_V, phase);
        mbarrier_wait_fn(bar_O, phase);
        mbarrier_wait_fn(bar_dO, phase);

        __syncthreads();
        
        __nv_bfloat16* smem_K_T0 = smem_K0;
        __nv_bfloat16* smem_K_T1 = smem_K1;
        transpose_smem_128x64_swizzled(smem_K0, smem_K_T0);
        transpose_smem_128x64_swizzled(smem_K1, smem_K_T1);
        
        __syncthreads(); 
        
        int acc_S = (kv_tile == 0) ? 0 : 1;
        asm volatile("{\n.reg .pred p;\n"
                     "setp.ne.b32 p, %4, 0;\n"
                     "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                     :: "r"(tmem_S_base), "l"(make_smem_desc(smem_Q0, 1, 1024)), "l"(make_smem_desc(smem_K_T0, 1024, 1024)), "r"(idesc_S), "r"(acc_S));
        
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64 [%0];"
                     :: "r"((uint32_t)__cvta_generic_to_shared(bar_V)) : "memory");
        
        mbarrier_wait_fn(bar_V, phase);
        phase ^= 1;
        tmem_load_fence_fn();

        float S_local[2][4][16][16];
        read_tmem_128x64(tid, tmem_S, S_local, 0, 0);

        __syncthreads(); 
        
        for(int i = 0; i < 2; i++) {
            for(int j = 0; j < 4; j++) {
                for(int r = 0; r < 16; r++) {
                    int row = (tid / 32) * 32 + i * 16 + r;
                    float l_val = smem_L[row];
                    
                    for(int c = 0; c < 16; c++) {
                        float s0 = S_local[i][j][r][c];
                        float p0 = expf(s0 * scale - l_val);
                        
                        __nv_bfloat16 do_val = read_swizzled_bf16(smem_dO0, row, j * 16 + c);
                        float do_f = __bfloat162float(do_val);
                        
                        write_swizzled_bf16(smem_P0, row, j * 16 + c, __float2bfloat16(p0));
                        write_swizzled_bf16(smem_D0, row, j * 16 + c, __float2bfloat16(do_f - p0));
                        
                        float s1 = S_local[i][j][r][c]; 
                        float p1 = expf(s1 * scale - l_val);
                        
                        __nv_bfloat16 do_val1 = read_swizzled_bf16(smem_dO1, row, j * 16 + c);
                        float do_f1 = __bfloat162float(do_val1);
                        
                        write_swizzled_bf16(smem_P1, row, j * 16 + c, __float2bfloat16(p1));
                        write_swizzled_bf16(smem_D1, row, j * 16 + c, __float2bfloat16(do_f1 - p1));
                    }
                }
            }
        }
        __syncthreads();

        __nv_bfloat16* smem_V_T0 = smem_V0;
        __nv_bfloat16* smem_V_T1 = smem_V1;
        transpose_smem_128x64_swizzled(smem_V0, smem_V_T0);
        transpose_smem_128x64_swizzled(smem_V1, smem_V_T1);
        __syncthreads();

        uint32_t tmem_dP_base = *reinterpret_cast<uint32_t*>(tmem_dQ) + 64;
        uint32_t idesc_dP = make_instr_desc_f16_cg1(128, 64);
        idesc_dP |= (1u << 16); 

        int acc_dP = 0;
        asm volatile("{\n.reg .pred p;\n"
                     "setp.ne.b32 p, %4, 0;\n"
                     "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                     :: "r"(tmem_dP_base), "l"(make_smem_desc(smem_D0, 1, 1024)), "l"(make_smem_desc(smem_V_T0, 1024, 1024)), "r"(idesc_dP), "r"(acc_dP));
        
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64 [%0];"
                     :: "r"((uint32_t)__cvta_generic_to_shared(bar_V)) : "memory");
        
        mbarrier_wait_fn(bar_V, phase);
        phase ^= 1;
        tmem_load_fence_fn();

        float dP_local[2][4][16][16];
        read_tmem_128x64(tid, (uint32_t*)&tmem_dP_base, dP_local, 0, 0);

        __syncthreads(); 
        
        for(int i = 0; i < 2; i++) {
            for(int j = 0; j < 4; j++) {
                for(int r = 0; r < 16; r++) {
                    int row = (tid / 32) * 32 + i * 16 + r;
                    for(int c = 0; c < 16; c++) {
                        write_swizzled_bf16(smem_dP0, row, j * 16 + c, __float2bfloat16(dP_local[i][j][r][c]));
                        write_swizzled_bf16(smem_dP1, row, j * 16 + c, __float2bfloat16(dP_local[i][j][r][c]));
                    }
                }
            }
        }
        __syncthreads();

        uint32_t idesc_dQ = make_instr_desc_f16_cg1(128, 64);
        idesc_dQ |= (1u << 16); 

        asm volatile("{\n.reg .pred p;\n"
                     "setp.ne.b32 p, %4, 0;\n"
                     "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                     :: "r"(tmem_dQ_base), "l"(make_smem_desc(smem_dP0, 1, 1024)), "l"(make_smem_desc(smem_V0, 1024, 1024)), "r"(idesc_dQ), "r"(1));
        
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64 [%0];"
                     :: "r"((uint32_t)__cvta_generic_to_shared(bar_V)) : "memory");
        
        mbarrier_wait_fn(bar_V, phase);
        phase ^= 1;
        tmem_load_fence_fn();
        
        __syncthreads();
    }

    epilogue(tmem_dQ, smem_Q0, S, 128, q_start, 0, dQ_ptr, b_h_idx, bh_offset);
    __syncthreads();

    if (tid == 0) {
        tmem_dealloc_fn(*tmem_S, 64);
        tmem_dealloc_fn(*tmem_dQ, 128);
        tmem_alloc_fn(tmem_S, 64);
        tmem_alloc_fn(tmem_dK, 128);
        tmem_alloc_fn(tmem_dV, 128);
    }
    __syncthreads();

    // === Pass 2: Compute dK and dV ===
    uint32_t kv_start = q_start; 
    
    if (tid == 0) {
        expect_tx(bar_K, 32768, phase);
        tma_load_2d_swizzled(&tma_K, bar_K, smem_K0, 0, bh_offset + kv_start);
        tma_load_2d_swizzled(&tma_K, bar_K, smem_K1, 64, bh_offset + kv_start);
        
        expect_tx(bar_V, 32768, phase);
        tma_load_2d_swizzled(&tma_V, bar_V, smem_V0, 0, bh_offset + kv_start);
        tma_load_2d_swizzled(&tma_V, bar_V, smem_V1, 64, bh_offset + kv_start);
    }
    mbarrier_wait_fn(bar_K, phase);
    mbarrier_wait_fn(bar_V, phase);
    if (tid < 128) {
        smem_L[tid] = (kv_start + tid < S) ? L_ptr[b_h_idx * S + kv_start + tid] : 0.0f;
    }
    __syncthreads();

    uint32_t tmem_dK_base = *reinterpret_cast<uint32_t*>(tmem_dK);
    uint32_t tmem_dV_base = *reinterpret_cast<uint32_t*>(tmem_dV);

    for (int q_tile_p2 = 0; q_tile_p2 < total_tiles; q_tile_p2++) {
        uint32_t qs = q_tile_p2 * 128;
        if (tid == 0) {
            expect_tx(bar_Q, 32768, phase);
            tma_load_2d_swizzled(&tma_Q, bar_Q, smem_Q0, 0, bh_offset + qs);
            tma_load_2d_swizzled(&tma_Q, bar_Q, smem_Q1, 64, bh_offset + qs);

            expect_tx(bar_O, 32768, phase);
            tma_load_2d_swizzled(&tma_O, bar_O, smem_O0, 0, bh_offset + qs);
            tma_load_2d_swizzled(&tma_O, bar_O, smem_O1, 64, bh_offset + qs);

            expect_tx(bar_dO, 32768, phase);
            tma_load_2d_swizzled(&tma_dO, bar_dO, smem_dO0, 0, bh_offset + qs);
            tma_load_2d_swizzled(&tma_dO, bar_dO, smem_dO1, 64, bh_offset + qs);
        }
        mbarrier_wait_fn(bar_Q, phase);
        mbarrier_wait_fn(bar_O, phase);
        mbarrier_wait_fn(bar_dO, phase);

        __syncthreads();
        
        __nv_bfloat16* smem_Q_T0 = smem_Q0;
        __nv_bfloat16* smem_Q_T1 = smem_Q1;
        transpose_smem_128x64_swizzled(smem_Q0, smem_Q_T0);
        transpose_smem_128x64_swizzled(smem_Q1, smem_Q_T1);
        
        __syncthreads(); 
        
        int acc_S = (q_tile_p2 == 0) ? 0 : 1;
        asm volatile("{\n.reg .pred p;\n"
                     "setp.ne.b32 p, %4, 0;\n"
                     "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                     :: "r"(tmem_S_base), "l"(make_smem_desc(smem_K0, 1, 1024)), "l"(make_smem_desc(smem_Q_T0, 1024, 1024)), "r"(idesc_S), "r"(acc_S));
        
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64 [%0];"
                     :: "r"((uint32_t)__cvta_generic_to_shared(bar_V)) : "memory");
        
        mbarrier_wait_fn(bar_V, phase);
        phase ^= 1;
        tmem_load_fence_fn();

        float S_local[2][4][16][16];
        read_tmem_128x64(tid, tmem_S, S_local, 0, 0);

        __syncthreads(); 
        
        for(int i = 0; i < 2; i++) {
            for(int j = 0; j < 4; j++) {
                for(int r = 0; r < 16; r++) {
                    int row = (tid / 32) * 32 + i * 16 + r;
                    float l_val = smem_L[row];
                    
                    for(int c = 0; c < 16; c++) {
                        float s0 = S_local[i][j][r][c];
                        float p0 = expf(s0 * scale - l_val);
                        
                        __nv_bfloat16 do_val = read_swizzled_bf16(smem_dO0, row, j * 16 + c);
                        float do_f = __bfloat162float(do_val);
                        
                        write_swizzled_bf16(smem_P0, row, j * 16 + c, __float2bfloat16(p0));
                        write_swizzled_bf16(smem_D0, row, j * 16 + c, __float2bfloat16(do_f - p0));
                        
                        float s1 = S_local[i][j][r][c];
                        float p1 = expf(s1 * scale - l_val);
                        
                        __nv_bfloat16 do_val1 = read_swizzled_bf16(smem_dO1, row, j * 16 + c);
                        float do_f1 = __bfloat162float(do_val1);
                        
                        write_swizzled_bf16(smem_P1, row, j * 16 + c, __float2bfloat16(p1));
                        write_swizzled_bf16(smem_D1, row, j * 16 + c, __float2bfloat16(do_f1 - p1));
                    }
                }
            }
        }
        __syncthreads();

        __nv_bfloat16* smem_V_T0 = smem_V0;
        __nv_bfloat16* smem_V_T1 = smem_V1;
        transpose_smem_128x64_swizzled(smem_V0, smem_V_T0);
        transpose_smem_128x64_swizzled(smem_V1, smem_V_T1);
        __syncthreads();

        uint32_t tmem_dP_base = *reinterpret_cast<uint32_t*>(tmem_S) + 64;
        uint32_t idesc_dP = make_instr_desc_f16_cg1(128, 64);
        idesc_dP |= (1u << 16);

        int acc_dP = 0; 
        asm volatile("{\n.reg .pred p;\n"
                     "setp.ne.b32 p, %4, 0;\n"
                     "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                     :: "r"(tmem_dP_base), "l"(make_smem_desc(smem_D0, 1, 1024)), "l"(make_smem_desc(smem_V_T0, 1024, 1024)), "r"(idesc_dP), "r"(acc_dP));
        
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64 [%0];"
                     :: "r"((uint32_t)__cvta_generic_to_shared(bar_V)) : "memory");
        
        mbarrier_wait_fn(bar_V, phase);
        phase ^= 1;
        tmem_load_fence_fn();

        float dP_local[2][4][16][16];
        read_tmem_128x64(tid, (uint32_t*)&tmem_dP_base, dP_local, 0, 0);

        __syncthreads(); 
        
        for(int i = 0; i < 2; i++) {
            for(int j = 0; j < 4; j++) {
                for(int r = 0; r < 16; r++) {
                    int row = (tid / 32) * 32 + i * 16 + r;
                    for(int c = 0; c < 16; c++) {
                        write_swizzled_bf16(smem_dP0, row, j * 16 + c, __float2bfloat16(dP_local[i][j][r][c]));
                        write_swizzled_bf16(smem_dP1, row, j * 16 + c, __float2bfloat16(dP_local[i][j][r][c]));
                    }
                }
            }
        }
        __syncthreads();
        
        __nv_bfloat16* smem_dPT0 = smem_dP0;
        __nv_bfloat16* smem_dPT1 = smem_dP1;
        transpose_smem_128x64_swizzled(smem_dP0, smem_dPT0);
        transpose_smem_128x64_swizzled(smem_dP1, smem_dPT1);
        __syncthreads();
        
        __nv_bfloat16* smem_PT0 = smem_P0;
        __nv_bfloat16* smem_PT1 = smem_P1;
        transpose_smem_128x64_swizzled(smem_P0, smem_PT0);
        transpose_smem_128x64_swizzled(smem_P1, smem_PT1);
        __syncthreads();

        uint32_t idesc_dK = make_instr_desc_f16_cg1(128, 64);
        idesc_dK |= (1u << 16); 

        asm volatile("{\n.reg .pred p;\n"
                     "setp.ne.b32 p, %4, 0;\n"
                     "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                     :: "r"(tmem_dK_base), "l"(make_smem_desc(smem_dPT0, 1, 1024)), "l"(make_smem_desc(smem_Q_T0, 1024, 1024)), "r"(idesc_dK), "r"(1));
        
        uint32_t idesc_dV = make_instr_desc_f16_cg1(128, 64);
        idesc_dV |= (1u << 16); 

        asm volatile("{\n.reg .pred p;\n"
                     "setp.ne.b32 p, %4, 0;\n"
                     "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                     :: "r"(tmem_dV_base), "l"(make_smem_desc(smem_PT0, 1, 1024)), "l"(make_smem_desc(smem_dO0, 1024, 1024)), "r"(idesc_dV), "r"(1));

        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64 [%0];"
                     :: "r"((uint32_t)__cvta_generic_to_shared(bar_V)) : "memory");
        
        mbarrier_wait_fn(bar_V, phase);
        phase ^= 1;
        tmem_load_fence_fn();
        
        __syncthreads();
    }
    
    epilogue(tmem_dK, smem_K0, S, 128, kv_start, 0, dK_ptr, b_h_idx, bh_offset);
    epilogue(tmem_dV, smem_V0, S, 128, kv_start, 0, dV_ptr, b_h_idx, bh_offset);
    __syncthreads();
    
    if (tid == 0) {
        tmem_dealloc_fn(*tmem_S, 64);
        tmem_dealloc_fn(*tmem_dK, 128);
        tmem_dealloc_fn(*tmem_dV, 128);
    }
    __syncthreads();
}

__device__ __forceinline__ void epilogue(
    uint32_t* tmem_ptr, __nv_bfloat16* smem_out, uint32_t M, uint32_t N, 
    uint32_t m_block, uint32_t n_block, __nv_bfloat16* D_ptr, uint32_t b_h_idx, uint32_t bh_offset) 
{
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    uint32_t tmem_addr = *reinterpret_cast<uint32_t*>(tmem_ptr);
    for (int col = 0; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_addr + col));
        
        int base = tid * 128 + col;
        smem_out[base + 0] = __float2bfloat16(__uint_as_float(r0));
        smem_out[base + 1] = __float2bfloat16(__uint_as_float(r1));
        smem_out[base + 2] = __float2bfloat16(__uint_as_float(r2));
        smem_out[base + 3] = __float2bfloat16(__uint_as_float(r3));
    }
    __syncthreads();

    uint32_t num_steps = (128 + 3) / 4; 
    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 4 + warp_id;
        if (row >= 128) continue;
        uint32_t global_row = m_block + row;
        uint32_t col_start = lane_id * 4;
        uint32_t global_col = n_block + col_start;
        if (global_row < M && global_col + 3 < N) {
            uint2 data = *reinterpret_cast<uint2*>(&smem_out[row * 128 + col_start]);
            *reinterpret_cast<uint2*>(&D_ptr[(bh_offset + global_row) * 128 + global_col]) = data;
        }
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

namespace tvm_ffi_mha_bwd {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L, 
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
         
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    uint32_t B = 4, H = 48, S = Q.size(0) / (B * H), d = 128;
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    void* Q_ptr = Q.data_ptr();
    void* K_ptr = K.data_ptr();
    void* V_ptr = V.data_ptr();
    void* O_ptr = O.data_ptr();
    void* dO_ptr = dO.data_ptr();
    const float* L_ptr = static_cast<const float*>(L.data_ptr());
    
    __nv_bfloat16* dQ_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O, tma_dO;
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_Q, Q_ptr, 128, S * B * H, 64, 128, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_K, K_ptr, 128, S * B * H, 64, 128, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_V, V_ptr, 128, S * B * H, 64, 128, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_O, O_ptr, 128, S * B * H, 64, 128, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_dO, dO_ptr, 128, S * B * H, 64, 128, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
    dim3 grid((S + 127) / 128, B * H);
    dim3 block(128);
    
    uint32_t smem_size = 16 * 16384 + 512 + 40; 
    
    CUDA_CHECK(cudaFuncSetAttribute((const void*)attn_backward_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    attn_backward_kernel<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, tma_O, tma_dO,
        L_ptr,
        dQ_ptr, dK_ptr, dV_ptr,
        S, 1.0f / sqrtf((float)d)
    );
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_bwd::run);

} // namespace tvm_ffi_mha_bwd