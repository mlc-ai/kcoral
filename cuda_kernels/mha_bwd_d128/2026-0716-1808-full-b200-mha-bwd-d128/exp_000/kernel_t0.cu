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

// ---- Minimal PTX Wrappers ----

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

__device__ __forceinline__ void tma_load_2d_swizzled(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ uint32_t pack_bf16_fn(uint32_t fp32_a, uint32_t fp32_b) {
    __nv_bfloat16 a = __float2bfloat16(__uint_as_float(fp32_a));
    __nv_bfloat16 b = __float2bfloat16(__uint_as_float(fp32_b));
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};"
        : "=r"(result)
        : "h"(*reinterpret_cast<uint16_t*>(&a)),
          "h"(*reinterpret_cast<uint16_t*>(&b)));
    return result;
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

// ---- Kernel Helpers ----

__device__ __forceinline__ float read_swizzled_f32(float* smem, int row, int col) {
    int col_chunk = col / 4;
    int chunk_offset = col % 4;
    int swizzled_col_chunk = (row % 8) ^ col_chunk;
    int swizzled_col = swizzled_col_chunk * 4 + chunk_offset;
    return smem[row * 128 + swizzled_col];
}

__device__ __forceinline__ __nv_bfloat16 read_swizzled_bf16(__nv_bfloat16* smem, int row, int col) {
    int col_chunk = col / 4;
    int chunk_offset = col % 4;
    int swizzled_col_chunk = (row % 8) ^ col_chunk;
    int swizzled_col = swizzled_col_chunk * 4 + chunk_offset;
    return smem[row * 128 + swizzled_col];
}

__device__ __forceinline__ void write_swizzled_bf16(__nv_bfloat16* smem, int row, int col, __nv_bfloat16 val) {
    int col_chunk = col / 4;
    int chunk_offset = col % 4;
    int swizzled_col_chunk = (row % 8) ^ col_chunk;
    int swizzled_col = swizzled_col_chunk * 4 + chunk_offset;
    smem[row * 128 + swizzled_col] = val;
}

__device__ __forceinline__ void transpose_smem_128x128_swizzled(__nv_bfloat16* smem_A, __nv_bfloat16* smem_B_T) {
    for (int i = threadIdx.x; i < 128 * 128; i += blockDim.x) {
        int r = i / 128;
        int c = i % 128;
        __nv_bfloat16 val = read_swizzled_bf16(smem_A, r, c);
        write_swizzled_bf16(smem_B_T, c, r, val);
    }
    __syncthreads();
}

__device__ __forceinline__ void gemm_128x128x128(
    __shared__ __nv_bfloat16* smem_DP,
    __shared__ __nv_bfloat16* smem_A,
    __shared__ __nv_bfloat16* smem_B) 
{
    for (int k_tile = 0; k_tile < 128 / 16; ++k_tile) {
        __nv_bfloat16 (*A_ptr)[128] = reinterpret_cast<__nv_bfloat16 (*)[128]>(smem_A);
        __nv_bfloat16 (*B_ptr)[128] = reinterpret_cast<__nv_bfloat16 (*)[128]>(smem_B);
        
        wmma::matrix_frag a_frag[8][8], b_frag[8][8];
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                wmma::load_matrix_sync(a_frag[i][j], A_ptr[i * 16 + k_tile * 16], 128);
                wmma::load_matrix_sync(b_frag[i][j], B_ptr[j * 16 + k_tile * 16], 128);
            }
        }
        
        wmma::matrix_acc_frag c_frag[8][8];
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                wmma::fill_fragment(c_frag[i][j], 0.0f);
            }
        }
        
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                wmma::mma_sync(c_frag[i][j], a_frag[i][j], b_frag[i][j], c_frag[i][j]);
            }
        }
        
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                wmma::store_matrix_sync(smem_DP + (i * 16) * 128 + j * 16, c_frag[i][j], 128, wmma::mem_row_major);
            }
        }
    }
}

__device__ __forceinline__ void compute_dP(
    __shared__ __nv_bfloat16* smem_dP,
    __shared__ __nv_bfloat16* smem_D,
    __shared__ __nv_bfloat16* smem_O_T) 
{
    gemm_128x128x128(smem_dP, smem_D, smem_O_T);
}

__device__ __forceinline__ void compute_dQ_contribution(
    float* dQ,
    __shared__ __nv_bfloat16* smem_dP,
    __shared__ __nv_bfloat16* smem_V,
    uint32_t q_start, uint32_t b_h_idx, uint32_t S) 
{
    uint64_t base_idx = ((uint64_t)b_h_idx * S + q_start) * 128;
    
    for (int row_idx = threadIdx.x; row_idx < 128; row_idx += blockDim.x) {
        float4 dp_row[32]; 
        for(int c = 0; c < 32; c++) {
            dp_row[c] = *reinterpret_cast<float4*>(&smem_dP[row_idx * 128 + c * 4]);
        }
        
        float4 v_row[32];
        for(int c = 0; c < 32; c++) {
            v_row[c] = *reinterpret_cast<float4*>(&smem_V[row_idx * 128 + c * 4]);
        }
        
        for (int col_idx = 0; col_idx < 128; col_idx += 4) {
            float4 dp_vec = dp_row[col_idx / 4];
            float4 v_vec = v_row[col_idx / 4];
            
            float f0 = __bfloat162float(reinterpret_cast<__nv_bfloat16*>(&dp_vec.x)[0]) * __bfloat162float(reinterpret_cast<__nv_bfloat16*>(&v_vec.x)[0]);
            float f1 = __bfloat162float(reinterpret_cast<__nv_bfloat16*>(&dp_vec.x)[1]) * __bfloat162float(reinterpret_cast<__nv_bfloat16*>(&v_vec.x)[1]);
            float f2 = __bfloat162float(reinterpret_cast<__nv_bfloat16*>(&dp_vec.y)[0]) * __bfloat162float(reinterpret_cast<__nv_bfloat16*>(&v_vec.y)[0]);
            float f3 = __bfloat162float(reinterpret_cast<__nv_bfloat16*>(&dp_vec.y)[1]) * __bfloat162float(reinterpret_cast<__nv_bfloat16*>(&v_vec.y)[1]);
            
            atomicAdd(&dQ[base_idx + col_idx + 0], f0);
            atomicAdd(&dQ[base_idx + col_idx + 1], f1);
            atomicAdd(&dQ[base_idx + col_idx + 2], f2);
            atomicAdd(&dQ[base_idx + col_idx + 3], f3);
        }
    }
}

__device__ __forceinline__ void compute_dK_dV_contribution(
    float* dK, float* dV,
    __shared__ __nv_bfloat16* smem_dP_T,
    __shared__ __nv_bfloat16* smem_P_T,
    __shared__ __nv_bfloat16* smem_Q,
    __shared__ __nv_bfloat16* smem_dO,
    uint32_t kv_start, uint32_t b_h_idx, uint32_t S) 
{
    uint64_t base_idx_k = ((uint64_t)b_h_idx * S + kv_start) * 128;
    uint64_t base_idx_v = ((uint64_t)b_h_idx * S + kv_start) * 128;
    
    for (int col_idx = threadIdx.x; col_idx < 128; col_idx += blockDim.x) {
        float dk_val = 0;
        float dv_val = 0;
        
        for (int row_idx = 0; row_idx < 128; row_idx++) {
            dk_val += __bfloat162float(read_swizzled_bf16(smem_dP_T, row_idx, col_idx)) * 
                      __bfloat162float(read_swizzled_bf16(smem_Q, row_idx, col_idx));
            
            dv_val += __bfloat162float(read_swizzled_bf16(smem_P_T, row_idx, col_idx)) * 
                      __bfloat162float(read_swizzled_bf16(smem_dO, row_idx, col_idx));
        }
        
        atomicAdd(&dK[base_idx_k + col_idx], dk_val);
        atomicAdd(&dV[base_idx_v + col_idx], dv_val);
    }
}

// ---- Shared Storage ----

extern __shared__ __align__(128) uint8_t smem_pool[];

__device__ void setup_smem_ptrs(
    __nv_bfloat16*& smem_Q, __nv_bfloat16*& smem_K, __nv_bfloat16*& smem_V,
    __nv_bfloat16*& smem_O, __nv_bfloat16*& smem_dO, __nv_bfloat16*& smem_dP,
    __nv_bfloat16*& smem_D, __nv_bfloat16*& smem_O_T, __nv_bfloat16*& smem_K_T,
    __nv_bfloat16*& smem_dP_T, __nv_bfloat16*& smem_P_T,
    float*& smem_L, uint64_t*& bar_Q, uint64_t*& bar_K, uint64_t*& bar_V,
    uint64_t*& bar_O, uint64_t*& bar_dO) 
{
    uint8_t* p = smem_pool;
    
    smem_Q  = reinterpret_cast<__nv_bfloat16*>(p); p += 32768;
    smem_K  = reinterpret_cast<__nv_bfloat16*>(p); p += 32768;
    smem_V  = reinterpret_cast<__nv_bfloat16*>(p); p += 32768;
    smem_O  = reinterpret_cast<__nv_bfloat16*>(p); p += 32768;
    smem_dO = reinterpret_cast<__nv_bfloat16*>(p); p += 32768;
    smem_dP = reinterpret_cast<__nv_bfloat16*>(p); p += 32768;
    
    smem_D   = smem_Q;   
    smem_O_T = smem_K;   
    smem_K_T = smem_V;   
    smem_dP_T= smem_O;   
    smem_P_T = smem_dO;  
    
    p = (uint8_t*)(((uintptr_t)p + 255) & ~255);
    
    smem_L = reinterpret_cast<float*>(p); p += 512; 
    
    p = (uint8_t*)(((uintptr_t)p + 255) & ~255);
    
    bar_Q = reinterpret_cast<uint64_t*>(p); p += 8;
    bar_K = reinterpret_cast<uint64_t*>(p); p += 8;
    bar_V = reinterpret_cast<uint64_t*>(p); p += 8;
    bar_O = reinterpret_cast<uint64_t*>(p); p += 8;
    bar_dO= reinterpret_cast<uint64_t*>(p); p += 8;
}

// ---- Main Kernel ----

__global__ void attn_backward_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* L_ptr,
    float* dQ, float* dK, float* dV,
    uint32_t S, float scale) 
{
    uint32_t q_tile = blockIdx.x;
    uint32_t b_h_idx = blockIdx.y;
    uint32_t q_start = q_tile * 128;
    uint32_t phase = 0;

    if (q_start >= S) return;

    __nv_bfloat16 *smem_Q, *smem_K, *smem_V, *smem_O, *smem_dO, *smem_dP;
    __nv_bfloat16 *smem_D, *smem_O_T, *smem_K_T, *smem_dP_T, *smem_P_T;
    float *smem_L;
    uint64_t *bar_Q, *bar_K, *bar_V, *bar_O, *bar_dO;

    setup_smem_ptrs(smem_Q, smem_K, smem_V, smem_O, smem_dO, smem_dP,
                    smem_D, smem_O_T, smem_K_T, smem_dP_T, smem_P_T,
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

    uint32_t total_tiles = (S + 127) / 128;

    // === Pass 1: Compute dQ ===
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_Q, 32768);
        tma_load_2d_swizzled(&tma_Q, bar_Q, smem_Q, 0, b_h_idx * S + q_start);
    }
    mbarrier_wait_fn(bar_Q, phase);
    phase ^= 1;

    for (int kv_tile = 0; kv_tile < total_tiles; ++kv_tile) {
        uint32_t kv_start = kv_tile * 128;

        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_O, 32768);
            tma_load_2d_swizzled(&tma_O, bar_O, smem_O, 0, b_h_idx * S + kv_start);

            mbarrier_arrive_and_expect_tx_fn(bar_dO, 32768);
            tma_load_2d_swizzled(&tma_dO, bar_dO, smem_dO, 0, b_h_idx * S + kv_start);

            mbarrier_arrive_and_expect_tx_fn(bar_V, 32768);
            tma_load_2d_swizzled(&tma_V, bar_V, smem_V, 0, b_h_idx * S + kv_start);
        }
        mbarrier_wait_fn(bar_O, phase);
        mbarrier_wait_fn(bar_dO, phase);
        mbarrier_wait_fn(bar_V, phase);
        phase ^= 1;

        if (threadIdx.x < 128) {
            smem_L[threadIdx.x] = (q_start + threadIdx.x < S) ? L_ptr[b_h_idx * S + q_start + threadIdx.x] : 0.0f;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < 128 * 128; i += blockDim.x) {
            int r = i / 128;
            int c = i % 128;
            __nv_bfloat16 do_val = read_swizzled_bf16(smem_dO, r, c);
            float l_val = smem_L[r];
            __nv_bfloat16 d_val = __float2bfloat16(__bfloat162float(do_val) - l_val);
            write_swizzled_bf16(smem_D, r, c, d_val);
        }
        __syncthreads();

        transpose_smem_128x128_swizzled(smem_O, smem_O_T);

        compute_dP(smem_dP, smem_D, smem_O_T);
        __syncthreads();

        compute_dQ_contribution(dQ, smem_dP, smem_V, q_start, b_h_idx, S);
        __syncthreads();
    }

    // === Pass 2: Compute dK and dV ===
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_K, 32768);
        tma_load_2d_swizzled(&tma_K, bar_K, smem_K, 0, b_h_idx * S + kv_start);
    }
    mbarrier_wait_fn(bar_K, phase);
    phase ^= 1;

    for (int q_tile_p2 = 0; q_tile_p2 < total_tiles; ++q_tile_p2) {
        uint32_t qs = q_tile_p2 * 128;
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_Q, 32768);
            tma_load_2d_swizzled(&tma_Q, bar_Q, smem_Q, 0, b_h_idx * S + qs);

            mbarrier_arrive_and_expect_tx_fn(bar_O, 32768);
            tma_load_2d_swizzled(&tma_O, bar_O, smem_O, 0, b_h_idx * S + qs);

            mbarrier_arrive_and_expect_tx_fn(bar_dO, 32768);
            tma_load_2d_swizzled(&tma_dO, bar_dO, smem_dO, 0, b_h_idx * S + qs);
        }
        mbarrier_wait_fn(bar_Q, phase);
        mbarrier_wait_fn(bar_O, phase);
        mbarrier_wait_fn(bar_dO, phase);
        phase ^= 1;

        if (threadIdx.x < 128) {
            smem_L[threadIdx.x] = (qs + threadIdx.x < S) ? L_ptr[b_h_idx * S + qs + threadIdx.x] : 0.0f;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < 128 * 128; i += blockDim.x) {
            int r = i / 128;
            int c = i % 128;
            __nv_bfloat16 do_val = read_swizzled_bf16(smem_dO, r, c);
            float l_val = smem_L[r];
            __nv_bfloat16 d_val = __float2bfloat16(__bfloat162float(do_val) - l_val);
            write_swizzled_bf16(smem_D, r, c, d_val);
        }
        __syncthreads();

        transpose_smem_128x128_swizzled(smem_O, smem_O_T);
        transpose_smem_128x128_swizzled(smem_K, smem_K_T);

        compute_dP(smem_dP, smem_D, smem_O_T);
        
        __shared__ float smem_S[128 * 128];
        for (int i = threadIdx.x; i < 128 * 128; i += blockDim.x) {
            int r = i / 128;
            int c = i % 128;
            float s_val = 0;
            for(int d = 0; d < 128; d++) {
                s_val += __bfloat162float(read_swizzled_bf16(smem_Q, r, d)) * 
                         __bfloat162float(read_swizzled_bf16(smem_K_T, d, c));
            }
            smem_S[r * 128 + c] = s_val;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < 128 * 128; i += blockDim.x) {
            int r = i / 128;
            int c = i % 128;
            float s_val = smem_S[r * 128 + c] * scale;
            float p_val = expf(s_val - smem_L[r]);
            write_swizzled_bf16(smem_dP, r, c, __float2bfloat16(p_val)); // Reuse smem_dP for P
        }
        __syncthreads();

        transpose_smem_128x128_swizzled(smem_dP, smem_dP_T); // dP_T
        transpose_smem_128x128_swizzled(smem_dP, smem_P_T);  // P_T

        compute_dK_dV_contribution(dK, dV, smem_dP_T, smem_P_T, smem_Q, smem_dO, kv_start, b_h_idx, S);
        __syncthreads();
    }
}

// ---- Cleanup Kernels ----

__global__ void convert_float_to_bf16(const float* src, __nv_bfloat16* dst, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2bfloat16(src[idx]);
    }
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
    create_tma_2d_descriptor_2B(&tma_Q, Q_ptr, 128, S * B * H, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_K, K_ptr, 128, S * B * H, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_V, V_ptr, 128, S * B * H, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_O, O_ptr, 128, S * B * H, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_dO, dO_ptr, 128, S * B * H, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    size_t total_elements = B * H * S * d;
    float* workspace_dQ = nullptr;
    float* workspace_dK = nullptr;
    float* workspace_dV = nullptr;
    
    CUDA_CHECK(cudaMallocAsync(&workspace_dQ, total_elements * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&workspace_dK, total_elements * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&workspace_dV, total_elements * sizeof(float), stream));
    
    CUDA_CHECK(cudaMemsetAsync(workspace_dQ, 0, total_elements * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(workspace_dK, 0, total_elements * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(workspace_dV, 0, total_elements * sizeof(float), stream));
    
    dim3 grid((S + 127) / 128, B * H);
    dim3 block(128);
    
    uint32_t smem_size = 11 * 32768 + 1024; 
    
    CUDA_CHECK(cudaFuncSetAttribute((const void*)attn_backward_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    attn_backward_kernel<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, tma_O, tma_dO,
        L_ptr,
        workspace_dQ, workspace_dK, workspace_dV,
        S, 1.0f / sqrtf((float)d)
    );
    CUDA_CHECK(cudaGetLastError());
    
    size_t threads = 256;
    size_t blocks = (total_elements + threads - 1) / threads;
    
    convert_float_to_bf16<<<blocks, threads, 0, stream>>>(workspace_dQ, dQ_ptr, total_elements);
    convert_float_to_bf16<<<blocks, threads, 0, stream>>>(workspace_dK, dK_ptr, total_elements);
    convert_float_to_bf16<<<blocks, threads, 0, stream>>>(workspace_dV, dV_ptr, total_elements);
    
    CUDA_CHECK(cudaFreeAsync(workspace_dQ, stream));
    CUDA_CHECK(cudaFreeAsync(workspace_dK, stream));
    CUDA_CHECK(cudaFreeAsync(workspace_dV, stream));
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_bwd::run);

} // namespace tvm_ffi_mha_bwd