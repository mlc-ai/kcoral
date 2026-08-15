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

__device__ __forceinline__ void tma_store_2d_swizzled(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%2, %3}], [%1];"
        :: "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tmem_store_commit_fn() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void tmem_store_wait_fn() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
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

__device__ __forceinline__ void umma_128x128x64(uint32_t* tmem_c, uint32_t* tmem_a, uint32_t* tmem_b, uint32_t idesc) {
    uint32_t tmem_c_base = *reinterpret_cast<uint32_t*>(tmem_c);
    uint32_t tmem_a_base = *reinterpret_cast<uint32_t*>(tmem_a);
    uint32_t tmem_b_base = *reinterpret_cast<uint32_t*>(tmem_b);
    for (int i = 0; i < 4; ++i) {
        uint32_t tmem_c = tmem_c_base + i * 2;
        uint32_t tmem_a = tmem_a_base + i * 2;
        uint32_t tmem_b = tmem_b_base + i * 2;
        uint64_t desc_a = make_smem_desc(cur_a, (cur_a == smem_Q0 || cur_a == smem_Q1) ? 1 : 1024, 1024);
        uint64_t desc_b = make_smem_desc(cur_b, (cur_b == smem_K_T) ? 1024 : 1, 1024);
        
        asm volatile("{\n.reg .pred p;\n"
                     "setp.ne.b32 p, %4, 0;\n"
                     "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                     :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum) : : "memory");
    }
}

__device__ __forceinline__ void read_tmем_128x128(int tid, uint32_t* tmem_ptr, float* out) {
    uint32_t tmem_addr = *reinterpret_cast<uint32_t*>(tmem_ptr);
    int row = tid % 128;
    int col_start = (tid / 128) * 4;
    for (int col = col_start; col < col_start + 4; ++col) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_addr + col));
        out[col] = __uint_as_float(r0);
        out[col+1] = __uint_as_float(r1);
        out[col+2] = __uint_as_float(r2);
        out[col+3] = __uint_as_float(r3);
    }
}

__device__ __forceinline__ void write_swizzled_bf16(__nv_bfloat16* smem, int row, int col, __nv_bfloat16 val) {
    int col_chunk = col / 4;
    int chunk_offset = col % 4;
    int swizzled_col_chunk = (row % 8) ^ col_chunk;
    int swizzled_col = swizzled_col_chunk * 4 + chunk_offset;
    smem[row * 128 + swizzled_col] = val;
}

__device__ __forceinline__ __nv_bfloat16 read_swizzled_bf16(__nv_bfloat16* smem, int row, int col) {
    int col_chunk = col / 4;
    int chunk_offset = col % 4;
    int swizzled_col_chunk = (row % 8) ^ col_chunk;
    int swizzled_col = swizzled_col_chunk * 4 + chunk_offset;
    return smem[row * 128 + swizzled_col];
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

extern __shared__ __align__(128) uint8_t smem_pool[];

__device__ void setup_smem_ptrs(
    __nv_bfloat16*& smem_Q0, __nv_bfloat16*& smem_Q1,
    __nv_bfloat16*& smem_K0, __nv_bfloat16*& smem_K1,
    __nv_bfloat16*& smem_V0, __nv_bfloat16*& smem_V1,
    __nv_bfloat16*& smem_O0, __nv_bfloat16*& smem_dO0,
    float*& dS_local, uint64_t*& bar_Q, uint64_t*& bar_K,
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
    
    p = (uint8_t*)(((uintptr_t)p + 255) & ~255);
    
    dS_local = reinterpret_cast<float*>(p); p += 65536;
    
    p = (uint8_t*)(((uintptr_t)p + 255) & ~255);
    
    bar_Q = reinterpret_cast<uint64_t*>(p); p += 8;
    bar_K = reinterpret_cast<uint64_t*>(p); p += 8;
    bar_V = reinterpret_cast<uint64_t*>(p); p += 8;
    bar_O = reinterpret_cast<uint64_t*>(p); p += 8;
    bar_dO= reinterpret_cast<uint64_t*>(p); p += 8;
}

__device__ __forceinline__ void process_kv_chunk(
    __nv_bfloat16* smem_K0, __nv_bfloat16* smem_K1,
    __nv_bfloat16* smem_V0, __nv_bfloat16* smem_V1,
    __nv_bfloat16* smem_O0, __nv_bfloat16* smem_dO0,
    uint32_t kv_start, uint32_t q_start, uint32_t S) 
{
    for (int i = threadIdx.x; i < 128 * 128; i += blockDim.x) {
        int r = i / 128;
        int c = i % 128;
        int chunk = c / 64;
        int local_c = c % 64;
        
        if (kv_start + r >= S || q_start + c >= S) {
            write_swizzled_bf16(smem_K0, r, c, __float2bfloat16(0.0f));
            write_swizzled_bf16(smem_K1, r, c, __float2bfloat16(0.0f));
            write_swizzled_bf16(smem_V0, r, c, __float2bfloat16(0.0f));
            write_swizzled_bf16(smem_V1, r, c, __float2bfloat16(0.0f));
            write_swizzled_bf16(smem_O0, r, c, __float2bfloat16(0.0f));
            write_swizzled_bf16(smem_dO0, r, c, __float2bfloat16(0.0f));
        }
    }
}

__device__ __forceinline__ void process_q_chunk(
    __nv_bfloat16* smem_Q0, __nv_bfloat16* smem_Q1,
    __nv_bfloat16* smem_O0, __nv_bfloat16* smem_dO0,
    uint32_t q_start, uint32_t kv_start, uint32_t S) 
{
    for (int i = threadIdx.x; i < 128 * 128; i += blockDim.x) {
        int r = i / 128;
        int c = i % 128;
        if (q_start + r >= S || kv_start + c >= S) {
            write_swizzled_bf16(smem_Q0, r, c, __float2bfloat16(0.0f));
            write_swizzled_bf16(smem_Q1, r, c, __float2bfloat16(0.0f));
            write_swizzled_bf16(smem_O0, r, c, __float2bfloat16(0.0f));
            write_swizzled_bf16(smem_dO0, r, c, __float2bfloat16(0.0f));
        }
    }
}

__device__ __forceinline__ void compute_dP_local(
    float* dS_local, float* dP_local, float* L_ptr, uint32_t q_start, uint32_t kv_start, uint32_t S, float scale) 
{
    int r = threadIdx.x;
    if (q_start + r >= S) return;
    
    float l_val = L_ptr[q_start + r];
    for (int c = 0; c < 128; c++) {
        if (kv_start + c >= S) {
            dS_local[r * 128 + c] = 0.0f;
            continue;
        }
        float s_val = dS_local[r * 128 + c] * scale;
        float p_val = expf(s_val - l_val);
        dP_local[r * 128 + c] = p_val * dS_local[r * 128 + c]; 
    }
}

__device__ __forceinline__ void epilogue(
    uint32_t* tmem_ptr, __nv_bfloat16* smem_out, uint32_t M, uint32_t N, 
    uint32_t m_block, uint32_t n_block) 
{
    uint32_t tmem_addr = *reinterpret_cast<uint32_t*>(tmem_ptr);
    for (int col = 0; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_addr + col));
        
        int base = threadIdx.x * 128 + col;
        smem_out[base + 0] = __float2bfloat16(__uint_as_float(r0));
        smem_out[base + 1] = __float2bfloat16(__uint_as_float(r1));
        smem_out[base + 2] = __float2bfloat16(__uint_as_float(r2));
        smem_out[base + 3] = __float2bfloat16(__uint_as_float(r3));
    }
    __syncthreads();
}

__device__ __forceinline__ void run_pass_1(
    uint32_t q_start, uint32_t b_h_idx, uint32_t S, float scale,
    const __grid_constant__ CUtensorMap* tma_Q, const __grid_constant__ CUtensorMap* tma_K,
    const __grid_constant__ CUtensorMap* tma_V, const __grid_constant__ CUtensorMap* tma_O,
    const __grid_constant__ CUtensorMap* tma_dO, const float* L_ptr, uint32_t* tmem_dQ) 
{
    uint32_t total_tiles = (S + 127) / 128;
    uint32_t bh_offset = b_h_idx * S;
    uint32_t phase = 0;

    if (threadIdx.x == 0) {
        expect_tx(bar_Q, 32768, phase);
        tma_load_2d_swizzled(tma_Q, bar_Q, smem_Q0, 0, bh_offset + q_start);
        tma_load_2d_swizzled(tma_Q, bar_Q, smem_Q1, 64, bh_offset + q_start);
    }
    mbarrier_wait_fn(bar_Q, phase);

    for (int kv_tile = 0; kv_tile < total_tiles; kv_tile++) {
        uint32_t kv_start = kv_tile * 128;
        if (threadIdx.x == 0) {
            expect_tx(bar_K, 32768, phase);
            tma_load_2d_swizzled(tma_K, bar_K, smem_K0, 0, bh_offset + kv_start);
            tma_load_2d_swizzled(tma_K, bar_K, smem_K1, 64, bh_offset + kv_start);

            expect_tx(bar_V, 32768, phase);
            tma_load_2d_swizzled(tma_V, bar_V, smem_V0, 0, bh_offset + kv_start);
            tma_load_2d_swizzled(tma_V, bar_V, smem_V1, 64, bh_offset + kv_start);

            expect_tx(bar_O, 32768, phase);
            tma_load_2d_swizzled(tma_O, bar_O, smem_O0, 0, bh_offset + kv_start);

            expect_tx(bar_dO, 32768, phase);
            tma_load_2d_swizzled(tma_dO, bar_dO, smem_dO0, 0, bh_offset + kv_start);
        }
        mbarrier_wait_fn(bar_K, phase);
        mbarrier_wait_fn(bar_V, phase);
        mbarrier_wait_fn(bar_O, phase);
        mbarrier_wait_fn(bar_dO, phase);

        process_kv_chunk(smem_K0, smem_K1, smem_V0, smem_V1, smem_O0, smem_dO0, kv_start, q_start, S);
        __syncthreads();

        __nv_bfloat16* smem_K_T0 = smem_V0;
        __nv_bfloat16* smem_K_T1 = smem_V1;
        transpose_smem_128x64_swizzled(smem_K0, smem_K_T0);
        transpose_smem_128x64_swizzled(smem_K1, smem_K_T1);

        uint32_t tmem_S_addr = *reinterpret_cast<uint32_t*>(tmem_S);
        for(int i = 0; i < 4; i++) {
            asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n"
                         :: "r"(tmem_S_addr + i * 2), "l"(desc_Q), "l"(desc_K_T), "r"(idesc_S) : : "memory");
        }

        read_tmem_128x128(tid, tmem_S, dS_local[tid]);
        tmem_load_fence_fn();
        
        float* P_local = dS_local + 128;
        compute_dP_local(dS_local, P_local, L_ptr, q_start, kv_start, S, scale);

        __nv_bfloat16* smem_dP0 = smem_K0;
        __nv_bfloat16* smem_dP1 = smem_K1;
        for(int j = 0; j < 128; j++) {
            write_swizzled_bf16(smem_dP0, tid, j, __float2bfloat16(P_local[tid * 128 + j]));
        }

        uint32_t tmem_dQ_addr = *reinterpret_cast<uint32_t*>(tmem_dQ);
        for(int i = 0; i < 4; i++) {
            asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n"
                         :: "r"(tmem_dQ_addr + i * 2), "l"(desc_dP), "l"(desc_V), "r"(idesc_dQ) : : "memory");
        }
        
        __syncthreads();
    }
}

__device__ __forceinline__ void run_pass_2(
    uint32_t kv_start, uint32_t b_h_idx, uint32_t S, float scale,
    const __grid_constant__ CUtensorMap* tma_Q, const __grid_constant__ CUtensorMap* tma_K,
    const __grid_constant__ CUtensorMap* tma_V, const __grid_constant__ CUtensorMap* tma_O,
    const __grid_constant__ CUtensorMap* tma_dO, const float* L_ptr, uint32_t* tmem_dK, uint32_t* tmem_dV) 
{
    uint32_t total_tiles = (S + 127) / 128;
    uint32_t bh_offset = b_h_idx * S;
    uint32_t phase = 0;

    if (threadIdx.x == 0) {
        expect_tx(bar_K, 32768, phase);
        tma_load_2d_swizzled(tma_K, bar_K, smem_K0, 0, bh_offset + kv_start);
        tma_load_2d_swizzled(tma_K, bar_K, smem_K1, 64, bh_offset + kv_start);
    }
    mbarrier_wait_fn(bar_K, phase);

    for (int q_tile = 0; q_tile < total_tiles; q_tile++) {
        uint32_t q_start = q_tile * 128;
        if (threadIdx.x == 0) {
            expect_tx(bar_Q, 32768, phase);
            tma_load_2d_swizzled(tma_Q, bar_Q, smem_Q0, 0, bh_offset + q_start);
            tma_load_2d_swizzled(tma_Q, bar_Q, smem_Q1, 64, bh_offset + q_start);

            expect_tx(bar_O, 32768, phase);
            tma_load_2d_swizzled(tma_O, bar_O, smem_O0, 0, bh_offset + q_start);

            expect_tx(bar_dO, 32768, phase);
            tma_load_2d_swizzled(tma_dO, bar_dO, smem_dO0, 0, bh_offset + q_start);
        }
        mbarrier_wait_fn(bar_Q, phase);
        mbarrier_wait_fn(bar_O, phase);
        mbarrier_wait_fn(bar_dO, phase);

        process_q_chunk(smem_Q0, smem_Q1, smem_O0, smem_dO0, q_start, kv_start, S);
        __syncthreads();

        __nv_bfloat16* smem_K_T0 = smem_V0;
        __nv_bfloat16* smem_K_T1 = smem_V1;
        transpose_smem_128x64_swizzled(smem_K0, smem_K_T0);
        transpose_smem_128x64_swizzled(smem_K1, smem_K_T1);

        uint32_t tmem_S_addr = *reinterpret_cast<uint32_t*>(tmem_S);
        for(int i = 0; i < 4; i++) {
            asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n"
                         :: "r"(tmem_S_addr + i * 2), "l"(desc_Q), "l"(desc_K_T), "r"(idesc_S) : : "memory");
        }

        read_tmem_128x128(tid, tmem_S, dS_local[tid]);
        tmem_load_fence_fn();
        
        float* P_local = dS_local + 128;
        compute_dP_local(dS_local, P_local, L_ptr, q_start, kv_start, S, scale);

        __nv_bfloat16* smem_dP0 = smem_Q0;
        __nv_bfloat16* smem_dP1 = smem_Q1;
        for(int j = 0; j < 128; j++) {
            write_swizzled_bf16(smem_dP0, tid, j, __float2bfloat16(P_local[tid * 128 + j]));
        }
        
        __nv_bfloat16* smem_dP_T0 = smem_V0;
        __nv_bfloat16* smem_dP_T1 = smem_V1;
        transpose_smem_128x64_swizzled(smem_dP0, smem_dP_T0);
        transpose_smem_128x64_swizzled(smem_dP1, smem_dP_T1);

        uint32_t tmem_dK_addr = *reinterpret_cast<uint32_t*>(tmem_dK);
        for(int i = 0; i < 4; i++) {
            asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n"
                         :: "r"(tmem_dK_addr + i * 2), "l"(desc_dP_T), "l"(desc_Q_T), "r"(idesc_dK) : : "memory");
        }

        uint32_t tmem_dV_addr = *reinterpret_cast<uint32_t*>(tmem_dV);
        for(int i = 0; i < 4; i++) {
            asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n"
                         :: "r"(tmem_dV_addr + i * 2), "l"(desc_P_T), "l"(desc_dO_T), "r"(idesc_dV) : : "memory");
        }
        
        __syncthreads();
    }
}

__global__ void attn_backward_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* L_ptr,
    const __grid_constant__ CUtensorMap tma_dQ,
    const __grid_constant__ CUtensorMap tma_dK,
    const __grid_constant__ CUtensorMap tma_dV,
    uint32_t S, float scale) 
{
    uint32_t q_tile = blockIdx.x;
    uint32_t b_h_idx = blockIdx.y;
    uint32_t q_start = q_tile * 128;
    uint32_t phase = 0;

    if (q_start >= S) return;

    __nv_bfloat16 *smem_Q0, *smem_Q1, *smem_K0, *smem_K1, *smem_V0, *smem_V1, *smem_O0, *smem_dO0;
    float *dS_local;
    uint64_t *bar_Q, *bar_K, *bar_V, *bar_O, *bar_dO;

    setup_smem_ptrs(smem_Q0, smem_Q1, smem_K0, smem_K1, smem_V0, smem_V1, smem_O0, smem_dO0,
                    dS_local, bar_Q, bar_K, bar_V, bar_O, bar_dO);

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar_Q, 1);
        init_smem_barrier_fn(bar_K, 1);
        init_smem_barrier_fn(bar_V, 1);
        init_smem_barrier_fn(bar_O, 1);
        init_smem_barrier_fn(bar_dO, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_S, 256);
        tmem_alloc_fn(&tmem_dQ, 256);
    }
    __syncthreads();

    run_pass_1(q_start, b_h_idx, S, scale, &tma_Q, &tma_K, &tma_V, &tma_O, &tma_dO, L_ptr, tmem_dQ);

    if (threadIdx.x == 0) {
        tmem_dealloc_fn(*tmem_S, 256);
        tmem_dealloc_fn(*tmem_dQ, 256);
        tmem_alloc_fn(&tmem_S, 256);
        tmem_alloc_fn(&tmem_dK, 256);
        tmem_alloc_fn(&tmem_dV, 256);
    }
    __syncthreads();

    run_pass_2(q_start, b_h_idx, S, scale, &tma_Q, &tma_K, &tma_V, &tma_O, &tma_dO, L_ptr, tmem_dK, tmem_dV);
    
    __syncthreads();

    if (threadIdx.x == 0) {
        tmem_dealloc_fn(*tmem_S, 256);
        tmem_dealloc_fn(*tmem_dK, 256);
        tmem_dealloc_fn(*tmem_dV, 256);
    }
    __syncthreads();
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
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O, tma_dO, tma_dQ, tma_dK, tma_dV;
    create_tma_2d_descriptor_2B(&tma_Q, Q_ptr, 128, S * B * H, 128, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_K, K_ptr, 128, S * B * H, 128, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_V, V_ptr, 128, S * B * H, 128, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_O, O_ptr, 128, S * B * H, 128, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_dO, dO_ptr, 128, S * B * H, 128, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_dQ, dQ_ptr, 128, S * B * H, 128, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_dK, dK_ptr, 128, S * B * H, 128, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_dV, dV_ptr, 128, S * B * H, 128, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    dim3 grid((S + 127) / 128, B * H);
    dim3 block(128);
    
    uint32_t smem_size = 7 * 16384 + 128 * 128 * sizeof(float) + 5 * sizeof(uint64_t);
    
    CUDA_CHECK(cudaFuncSetAttribute((const void*)attn_backward_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    attn_backward_kernel<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, tma_O, tma_dO,
        L_ptr,
        tma_dQ, tma_dK, tma_dV,
        S, 1.0f / sqrtf((float)d)
    );
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_bwd::run);

} // namespace tvm_ffi_mha_bwd