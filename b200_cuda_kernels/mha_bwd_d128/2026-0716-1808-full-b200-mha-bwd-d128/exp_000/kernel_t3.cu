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

__device__ __forceinline__ void gemm_128x64x64(
    wmma::matrix_acc_frag c_frag[2][4],
    __nv_bfloat16* smem_A, 
    __nv_bfloat16* smem_B) 
{
    int warp_id = threadIdx.x / 32;
    int warp_row_base = warp_id * 2;
    
    for (int k_tile = 0; k_tile < 4; ++k_tile) {
        wmma::matrix_frag b_frag[4];
        for (int j = 0; j < 4; ++j) {
            wmma::load_matrix_sync(b_frag[j], smem_B + k_tile * 16 * 64 + j * 16, 64);
        }
        
        wmma::matrix_frag a_frag[2];
        for (int i = 0; i < 2; ++i) {
            wmma::load_matrix_sync(a_frag[i], smem_A + (warp_row_base + i) * 16 * 64 + k_tile * 16, 64);
        }
        
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 4; ++j) {
                wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
            }
        }
    }
}

__device__ __forceinline__ void gemm_128x64x64_accumulate(
    wmma::matrix_acc_frag c_frag[2][4],
    __nv_bfloat16* smem_A, 
    __nv_bfloat16* smem_B) 
{
    int warp_id = threadIdx.x / 32;
    int warp_row_base = warp_id * 2;
    
    for (int k_tile = 0; k_tile < 4; ++k_tile) {
        wmma::matrix_frag b_frag[4];
        for (int j = 0; j < 4; ++j) {
            wmma::load_matrix_sync(b_frag[j], smem_B + k_tile * 16 * 64 + j * 16, 64);
        }
        
        wmma::matrix_frag a_frag[2];
        for (int i = 0; i < 2; ++i) {
            wmma::load_matrix_sync(a_frag[i], smem_A + (warp_row_base + i) * 16 * 64 + k_tile * 16, 64);
        }
        
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 4; ++j) {
                wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
            }
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
    __nv_bfloat16*& smem_K_T0, __nv_bfloat16*& smem_K_T1,
    __nv_bfloat16*& smem_V_T0, __nv_bfloat16*& smem_V_T1,
    __nv_bfloat16*& smem_dS0, __nv_bfloat16*& smem_dS1,
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
    smem_K_T0= reinterpret_cast<__nv_bfloat16*>(p); p += 8192;
    smem_K_T1= reinterpret_cast<__nv_bfloat16*>(p); p += 8192;
    smem_V_T0= reinterpret_cast<__nv_bfloat16*>(p); p += 8192;
    smem_V_T1= reinterpret_cast<__nv_bfloat16*>(p); p += 8192;
    smem_dS0 = reinterpret_cast<__nv_bfloat16*>(p); p += 16384;
    smem_dS1 = reinterpret_cast<__nv_bfloat16*>(p); p += 16384;
    
    p = (uint8_t*)(((uintptr_t)p + 255) & ~255);
    
    smem_L = reinterpret_cast<float*>(p); p += 512;
    
    p = (uint8_t*)(((uintptr_t)p + 255) & ~255);
    
    bar_Q = reinterpret_cast<uint64_t*>(p); p += 8;
    bar_K = reinterpret_cast<uint64_t*>(p); p += 8;
    bar_V = reinterpret_cast<uint64_t*>(p); p += 8;
    bar_O = reinterpret_cast<uint64_t*>(p); p += 8;
    bar_dO= reinterpret_cast<uint64_t*>(p); p += 8;
}

__device__ __forceinline__ void run_pass_1(
    uint32_t q_start, uint32_t b_h_idx, uint32_t S, float scale,
    const CUtensorMap* tma_Q, const CUtensorMap* tma_K,
    const CUtensorMap* tma_V, const CUtensorMap* tma_O,
    const CUtensorMap* tma_dO, const float* L_ptr) 
{
    uint32_t total_tiles = (S + 127) / 128;
    uint32_t bh_offset = b_h_idx * S;
    uint32_t phase = 0;
    int tid = threadIdx.x;

    if (tid == 0) {
        expect_tx(bar_Q, 32768, phase);
        tma_load_2d_swizzled(tma_Q, bar_Q, smem_Q0, 0, bh_offset + q_start);
        tma_load_2d_swizzled(tma_Q, bar_Q, smem_Q1, 64, bh_offset + q_start);
    }
    mbarrier_wait_fn(bar_Q, phase);
    if (tid < 128) {
        smem_L[tid] = (q_start + tid < S) ? L_ptr[b_h_idx * S + q_start + tid] : 0.0f;
    }
    __syncthreads();

    wmma::matrix_acc_frag dQ_accum0[2][4];
    wmma::matrix_acc_frag dQ_accum1[2][4];
    int warp_id = tid / 32;
    int warp_row_base = warp_id * 2;
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 4; j++) {
            wmma::fill_fragment(dQ_accum0[i][j], 0.0f);
            wmma::fill_fragment(dQ_accum1[i][j], 0.0f);
        }
    }

    for (int kv_tile = 0; kv_tile < total_tiles; kv_tile++) {
        uint32_t kv_start = kv_tile * 128;
        if (tid == 0) {
            expect_tx(bar_K, 32768, phase);
            tma_load_2d_swizzled(tma_K, bar_K, smem_K0, 0, bh_offset + kv_start);
            tma_load_2d_swizzled(tma_K, bar_K, smem_K1, 64, bh_offset + kv_start);

            expect_tx(bar_V, 32768, phase);
            tma_load_2d_swizzled(tma_V, bar_V, smem_V0, 0, bh_offset + kv_start);
            tma_load_2d_swizzled(tma_V, bar_V, smem_V1, 64, bh_offset + kv_start);

            expect_tx(bar_O, 32768, phase);
            tma_load_2d_swizzled(tma_O, bar_O, smem_O0, 0, bh_offset + kv_start);
            tma_load_2d_swizzled(tma_O, bar_O, smem_O1, 64, bh_offset + kv_start);

            expect_tx(bar_dO, 32768, phase);
            tma_load_2d_swizzled(tma_dO, bar_dO, smem_dO0, 0, bh_offset + kv_start);
            tma_load_2d_swizzled(tma_dO, bar_dO, smem_dO1, 64, bh_offset + kv_start);
        }
        mbarrier_wait_fn(bar_K, phase);
        mbarrier_wait_fn(bar_V, phase);
        mbarrier_wait_fn(bar_O, phase);
        mbarrier_wait_fn(bar_dO, phase);

        float rowsum_O[2] = {0, 0};
        for (int c = 0; c < 64; c++) {
            rowsum_O[0] += __bfloat162float(read_swizzled_bf16(smem_O0, tid, c)) * __bfloat162float(read_swizzled_bf16(smem_dO0, tid, c));
            rowsum_O[1] += __bfloat162float(read_swizzled_bf16(smem_O1, tid, c)) * __bfloat162float(read_swizzled_bf16(smem_dO1, tid, c));
        }
        
        __syncthreads();
        transpose_smem_128x64_swizzled(smem_K0, smem_K_T0);
        transpose_smem_128x64_swizzled(smem_K1, smem_K_T1);
        __syncthreads();
        transpose_smem_128x64_swizzled(smem_V0, smem_V_T0);
        transpose_smem_128x64_swizzled(smem_V1, smem_V_T1);
        __syncthreads();

        wmma::matrix_acc_frag c_frag_S0[2][4];
        wmma::matrix_acc_frag c_frag_S1[2][4];
        for(int i = 0; i < 2; i++) {
            for(int j = 0; j < 4; j++) {
                wmma::fill_fragment(c_frag_S0[i][j], 0.0f);
                wmma::fill_fragment(c_frag_S1[i][j], 0.0f);
            }
        }
        
        gemm_128x64x64(c_frag_S0, smem_Q0, smem_K_T0);
        gemm_128x64x64(c_frag_S1, smem_Q1, smem_K_T1);
        
        __syncthreads(); 
        
        for(int i = 0; i < 2; i++) {
            for(int j = 0; j < 4; j++) {
                float P_local[16][16];
                wmma::store_matrix_sync(smem_O0 + (warp_row_base + i) * 16 * 64 + j * 16, c_frag_S0[i][j], 64);
                wmma::store_matrix_sync(smem_O1 + (warp_row_base + i) * 16 * 64 + j * 16, c_frag_S1[i][j], 64);
            }
        }
        __syncthreads();

        for(int r = tid; r < 128 * 64; r += blockDim.x) {
            int row = r / 64;
            int col = r % 64;
            float l_val = smem_L[row];
            
            float s0 = __bfloat162float(read_swizzled_bf16(smem_O0, row, col));
            float p0 = expf(s0 * scale - l_val);
            write_swizzled_bf16(smem_O0, row, col, __float2bfloat16(p0));
            
            float s1 = __bfloat162float(read_swizzled_bf16(smem_O1, row, col));
            float p1 = expf(s1 * scale - l_val);
            write_swizzled_bf16(smem_O1, row, col, __float2bfloat16(p1));
        }
        __syncthreads();
        
        __nv_bfloat16* smem_D0 = smem_K0;
        __nv_bfloat16* smem_D1 = smem_K1;
        for(int r = tid; r < 128 * 64; r += blockDim.x) {
            int row = r / 64;
            int col = r % 64;
            float l_val = smem_L[row];
            
            float o0 = __bfloat162float(read_swizzled_bf16(smem_dO0, row, col));
            write_swizzled_bf16(smem_D0, row, col, __float2bfloat16(o0 - l_val));
            
            float o1 = __bfloat162float(read_swizzled_bf16(smem_dO1, row, col));
            write_swizzled_bf16(smem_D1, row, col, __float2bfloat16(o1 - l_val));
        }
        __syncthreads();

        wmma::matrix_acc_frag c_frag_dP0[2][4];
        wmma::matrix_acc_frag c_frag_dP1[2][4];
        for(int i = 0; i < 2; i++) {
            for(int j = 0; j < 4; j++) {
                wmma::fill_fragment(c_frag_dP0[i][j], 0.0f);
                wmma::fill_fragment(c_frag_dP1[i][j], 0.0f);
            }
        }
        
        gemm_128x64x64(c_frag_dP0, smem_D0, smem_V_T0);
        gemm_128x64x64(c_frag_dP1, smem_D1, smem_V_T1);

        __syncthreads();
        for(int i = 0; i < 2; i++) {
            for(int j = 0; j < 4; j++) {
                wmma::store_matrix_sync(smem_dS0 + (warp_row_base + i) * 16 * 64 + j * 16, c_frag_dP0[i][j], 64);
                wmma::store_matrix_sync(smem_dS1 + (warp_row_base + i) * 16 * 64 + j * 16, c_frag_dP1[i][j], 64);
            }
        }
        __syncthreads();

        __nv_bfloat16* smem_dST0 = smem_K_T0;
        __nv_bfloat16* smem_dST1 = smem_K_T1;
        transpose_smem_128x64_swizzled(smem_dS0, smem_dST0);
        transpose_smem_128x64_swizzled(smem_dS1, smem_dST1);
        __syncthreads();

        gemm_128x64x64_accumulate(dQ_accum0, smem_dST0, smem_Q0);
        gemm_128x64x64_accumulate(dQ_accum1, smem_dST1, smem_Q1);

        __syncthreads();
    }

    __nv_bfloat16* dQ_smem = smem_Q0;
    __syncthreads();
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 4; j++) {
            wmma::store_matrix_sync(dQ_smem + (warp_row_base + i) * 16 * 64 + j * 16, dQ_accum0[i][j], 64);
            wmma::store_matrix_sync(dQ_smem + 8192 + (warp_row_base + i) * 16 * 64 + j * 16, dQ_accum1[i][j], 64);
        }
    }
    __syncthreads();

    uint32_t stride = 128;
    for(int i = tid; i < 128 * 128 / 8; i += blockDim.x) {
        int row = i / 16;
        int col_vec = i % 16;
        int col = col_vec * 8;
        if (q_start + row < S) {
            float4 f0 = *reinterpret_cast<float4*>(&dQ_smem[row * 64 + col]);
            float4 f1 = *reinterpret_cast<float4*>(&dQ_smem[8192 + row * 64 + col]);
            __nv_bfloat16 out[8];
            out[0] = __float2bfloat16(f0.x); out[1] = __float2bfloat16(f0.y); out[2] = __float2bfloat16(f0.z); out[3] = __float2bfloat16(f0.w);
            out[4] = __float2bfloat16(f1.x); out[5] = __float2bfloat16(f1.y); out[6] = __float2bfloat16(f1.z); out[7] = __float2bfloat16(f1.w);
            *reinterpret_cast<float4*>(&dQ_ptr[(b_h_idx * S + q_start + row) * 128 + col]) = *reinterpret_cast<float4*>(out);
        }
    }
}

__device__ __forceinline__ void run_pass_2(
    uint32_t kv_start, uint32_t b_h_idx, uint32_t S, float scale,
    const CUtensorMap* tma_Q, const CUtensorMap* tma_K,
    const CUtensorMap* tma_V, const CUtensorMap* tma_O,
    const CUtensorMap* tma_dO, const float* L_ptr) 
{
    uint32_t total_tiles = (S + 127) / 128;
    uint32_t bh_offset = b_h_idx * S;
    uint32_t phase = 0;
    int tid = threadIdx.x;

    if (tid == 0) {
        expect_tx(bar_K, 32768, phase);
        tma_load_2d_swizzled(tma_K, bar_K, smem_K0, 0, bh_offset + kv_start);
        tma_load_2d_swizzled(tma_K, bar_K, smem_K1, 64, bh_offset + kv_start);
        
        expect_tx(bar_V, 32768, phase);
        tma_load_2d_swizzled(tma_V, bar_V, smem_V0, 0, bh_offset + kv_start);
        tma_load_2d_swizzled(tma_V, bar_V, smem_V1, 64, bh_offset + kv_start);
    }
    mbarrier_wait_fn(bar_K, phase);
    mbarrier_wait_fn(bar_V, phase);
    if (tid < 128) {
        smem_L[tid] = (kv_start + tid < S) ? L_ptr[b_h_idx * S + kv_start + tid] : 0.0f;
    }
    __syncthreads();

    wmma::matrix_acc_frag dK_accum0[2][4];
    wmma::matrix_acc_frag dK_accum1[2][4];
    wmma::matrix_acc_frag dV_accum0[2][4];
    wmma::matrix_acc_frag dV_accum1[2][4];
    int warp_id = tid / 32;
    int warp_row_base = warp_id * 2;
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 4; j++) {
            wmma::fill_fragment(dK_accum0[i][j], 0.0f);
            wmma::fill_fragment(dK_accum1[i][j], 0.0f);
            wmma::fill_fragment(dV_accum0[i][j], 0.0f);
            wmma::fill_fragment(dV_accum1[i][j], 0.0f);
        }
    }

    for (int q_tile = 0; q_tile < total_tiles; q_tile++) {
        uint32_t q_start = q_tile * 128;
        if (tid == 0) {
            expect_tx(bar_Q, 32768, phase);
            tma_load_2d_swizzled(tma_Q, bar_Q, smem_Q0, 0, bh_offset + q_start);
            tma_load_2d_swizzled(tma_Q, bar_Q, smem_Q1, 64, bh_offset + q_start);

            expect_tx(bar_O, 32768, phase);
            tma_load_2d_swizzled(tma_O, bar_O, smem_O0, 0, bh_offset + q_start);
            tma_load_2d_swizzled(tma_O, bar_O, smem_O1, 64, bh_offset + q_start);

            expect_tx(bar_dO, 32768, phase);
            tma_load_2d_swizzled(tma_dO, bar_dO, smem_dO0, 0, bh_offset + q_start);
            tma_load_2d_swizzled(tma_dO, bar_dO, smem_dO1, 64, bh_offset + q_start);
        }
        mbarrier_wait_fn(bar_Q, phase);
        mbarrier_wait_fn(bar_O, phase);
        mbarrier_wait_fn(bar_dO, phase);

        float rowsum_O[2] = {0, 0};
        for (int c = 0; c < 64; c++) {
            rowsum_O[0] += __bfloat162float(read_swizzled_bf16(smem_O0, tid, c)) * __bfloat162float(read_swizzled_bf16(smem_dO0, tid, c));
            rowsum_O[1] += __bfloat162float(read_swizzled_bf16(smem_O1, tid, c)) * __bfloat162float(read_swizzled_bf16(smem_dO1, tid, c));
        }
        
        __syncthreads();
        transpose_smem_128x64_swizzled(smem_Q0, smem_K_T0); 
        transpose_smem_128x64_swizzled(smem_Q1, smem_K_T1); 
        __syncthreads();
        transpose_smem_128x64_swizzled(smem_V0, smem_V_T0);
        transpose_smem_128x64_swizzled(smem_V1, smem_V_T1);
        __syncthreads();

        wmma::matrix_acc_frag c_frag_S0[2][4];
        wmma::matrix_acc_frag c_frag_S1[2][4];
        for(int i = 0; i < 2; i++) {
            for(int j = 0; j < 4; j++) {
                wmma::fill_fragment(c_frag_S0[i][j], 0.0f);
                wmma::fill_fragment(c_frag_S1[i][j], 0.0f);
            }
        }
        
        gemm_128x64x64(c_frag_S0, smem_K0, smem_K_T0); 
        gemm_128x64x64(c_frag_S1, smem_K1, smem_K_T1); 
        __syncthreads(); 
        
        for(int i = 0; i < 2; i++) {
            for(int j = 0; j < 4; j++) {
                wmma::store_matrix_sync(smem_O0 + (warp_row_base + i) * 16 * 64 + j * 16, c_frag_S0[i][j], 64);
                wmma::store_matrix_sync(smem_O1 + (warp_row_base + i) * 16 * 64 + j * 16, c_frag_S1[i][j], 64);
            }
        }
        __syncthreads();

        for(int r = tid; r < 128 * 64; r += blockDim.x) {
            int row = r / 64;
            int col = r % 64;
            float l_val = smem_L[row];
            
            float s0 = __bfloat162float(read_swizzled_bf16(smem_O0, row, col));
            float p0 = expf(s0 * scale - l_val);
            write_swizzled_bf16(smem_O0, row, col, __float2bfloat16(p0));
            
            float s1 = __bfloat162float(read_swizzled_bf16(smem_O1, row, col));
            float p1 = expf(s1 * scale - l_val);
            write_swizzled_bf16(smem_O1, row, col, __float2bfloat16(p1));
        }
        __syncthreads();
        
        __nv_bfloat16* smem_D0 = smem_K0;
        __nv_bfloat16* smem_D1 = smem_K1;
        for(int r = tid; r < 128 * 64; r += blockDim.x) {
            int row = r / 64;
            int col = r % 64;
            float l_val = smem_L[row];
            
            float o0 = __bfloat162float(read_swizzled_bf16(smem_dO0, row, col));
            write_swizzled_bf16(smem_D0, row, col, __float2bfloat16(o0 - l_val));
            
            float o1 = __bfloat162float(read_swizzled_bf16(smem_dO1, row, col));
            write_swizzled_bf16(smem_D1, row, col, __float2bfloat16(o1 - l_val));
        }
        __syncthreads();

        wmma::matrix_acc_frag c_frag_dP0[2][4];
        wmma::matrix_acc_frag c_frag_dP1[2][4];
        for(int i = 0; i < 2; i++) {
            for(int j = 0; j < 4; j++) {
                wmma::fill_fragment(c_frag_dP0[i][j], 0.0f);
                wmma::fill_fragment(c_frag_dP1[i][j], 0.0f);
            }
        }
        
        gemm_128x64x64(c_frag_dP0, smem_D0, smem_V_T0);
        gemm_128x64x64(c_frag_dP1, smem_D1, smem_V_T1);

        __syncthreads();
        for(int i = 0; i < 2; i++) {
            for(int j = 0; j < 4; j++) {
                wmma::store_matrix_sync(smem_dS0 + (warp_row_base + i) * 16 * 64 + j * 16, c_frag_dP0[i][j], 64);
                wmma::store_matrix_sync(smem_dS1 + (warp_row_base + i) * 16 * 64 + j * 16, c_frag_dP1[i][j], 64);
            }
        }
        __syncthreads();

        __nv_bfloat16* smem_dPT0 = smem_K_T0; 
        __nv_bfloat16* smem_dPT1 = smem_K_T1; 
        transpose_smem_128x64_swizzled(smem_dS0, smem_dPT0);
        transpose_smem_128x64_swizzled(smem_dS1, smem_dPT1);
        __syncthreads();

        __nv_bfloat16* smem_PT0 = smem_Q0; 
        __nv_bfloat16* smem_PT1 = smem_Q1; 
        transpose_smem_128x64_swizzled(smem_O0, smem_PT0);
        transpose_smem_128x64_swizzled(smem_O1, smem_PT1);
        __syncthreads();

        gemm_128x64x64_accumulate(dK_accum0, smem_dPT0, smem_V_T0); 
        gemm_128x64x64_accumulate(dK_accum1, smem_dPT1, smem_V_T1); 

        gemm_128x64x64_accumulate(dV_accum0, smem_PT0, smem_K_T0); 
        gemm_128x64x64_accumulate(dV_accum1, smem_PT1, smem_K_T1); 

        __syncthreads();
    }

    __nv_bfloat16* dK_smem = smem_K0;
    __nv_bfloat16* dV_smem = smem_V0;
    __syncthreads();
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 4; j++) {
            wmma::store_matrix_sync(dK_smem + (warp_row_base + i) * 16 * 64 + j * 16, dK_accum0[i][j], 64);
            wmma::store_matrix_sync(dK_smem + 8192 + (warp_row_base + i) * 16 * 64 + j * 16, dK_accum1[i][j], 64);
            
            wmma::store_matrix_sync(dV_smem + (warp_row_base + i) * 16 * 64 + j * 16, dV_accum0[i][j], 64);
            wmma::store_matrix_sync(dV_smem + 8192 + (warp_row_base + i) * 16 * 64 + j * 16, dV_accum1[i][j], 64);
        }
    }
    __syncthreads();

    uint32_t stride = 128;
    for(int i = tid; i < 128 * 128 / 8; i += blockDim.x) {
        int row = i / 16;
        int col_vec = i % 16;
        int col = col_vec * 8;
        if (kv_start + row < S) {
            float4 f0 = *reinterpret_cast<float4*>(&dK_smem[row * 64 + col]);
            float4 f1 = *reinterpret_cast<float4*>(&dK_smem[8192 + row * 64 + col]);
            __nv_bfloat16 out_k[8];
            out_k[0] = __float2bfloat16(f0.x); out_k[1] = __float2bfloat16(f0.y); out_k[2] = __float2bfloat16(f0.z); out_k[3] = __float2bfloat16(f0.w);
            out_k[4] = __float2bfloat16(f1.x); out_k[5] = __float2bfloat16(f1.y); out_k[6] = __float2bfloat16(f1.z); out_k[7] = __float2bfloat16(f1.w);
            *reinterpret_cast<float4*>(&dK_ptr[(b_h_idx * S + kv_start + row) * 128 + col]) = *reinterpret_cast<float4*>(out_k);

            float4 g0 = *reinterpret_cast<float4*>(&dV_smem[row * 64 + col]);
            float4 g1 = *reinterpret_cast<float4*>(&dV_smem[8192 + row * 64 + col]);
            __nv_bfloat16 out_v[8];
            out_v[0] = __float2bfloat16(g0.x); out_v[1] = __float2bfloat16(g0.y); out_v[2] = __float2bfloat16(g0.z); out_v[3] = __float2bfloat16(g0.w);
            out_v[4] = __float2bfloat16(g1.x); out_v[5] = __float2bfloat16(g1.y); out_v[6] = __float2bfloat16(g1.z); out_v[7] = __float2bfloat16(g1.w);
            *reinterpret_cast<float4*>(&dV_ptr[(b_h_idx * S + kv_start + row) * 128 + col]) = *reinterpret_cast<float4*>(out_v);
        }
    }
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
    uint32_t q_tile = blockIdx.x;
    uint32_t b_h_idx = blockIdx.y;
    uint32_t q_start = q_tile * 128;
    uint32_t phase = 0;

    if (q_start >= S) return;

    __nv_bfloat16 *smem_Q0, *smem_Q1, *smem_K0, *smem_K1, *smem_V0, *smem_V1, *smem_O0, *smem_O1, *smem_dO0, *smem_dO1;
    __nv_bfloat16 *smem_K_T0, *smem_K_T1, *smem_V_T0, *smem_V_T1, *smem_dS0, *smem_dS1;
    float *smem_L;
    uint64_t *bar_Q, *bar_K, *bar_V, *bar_O, *bar_dO;

    setup_smem_ptrs(smem_Q0, smem_Q1, smem_K0, smem_K1, smem_V0, smem_V1, smem_O0, smem_O1, smem_dO0, smem_dO1,
                    smem_K_T0, smem_K_T1, smem_V_T0, smem_V_T1, smem_dS0, smem_dS1,
                    smem_L, bar_Q, bar_K, bar_V, bar_O, bar_dO);

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar_Q, 1);
        init_smem_barrier_fn(bar_K, 1);
        init_smem_barrier_fn(bar_V, 1);
        init_smem_barrier_fn(bar_O, 1);
        init_smem_barrier_fn(bar_dO, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    run_pass_1(q_start, b_h_idx, S, scale, &tma_Q, &tma_K, &tma_V, &tma_O, &tma_dO, L_ptr);
    
    __syncthreads();

    uint32_t kv_start = q_start;
    run_pass_2(kv_start, b_h_idx, S, scale, &tma_Q, &tma_K, &tma_V, &tma_O, &tma_dO, L_ptr);
    
    __syncthreads();
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
    
    uint32_t smem_size = 14 * 16384 + 512 + 40; 
    
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