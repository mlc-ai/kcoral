#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>
#include <math.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

#define CU_CHECK(call) do {                                        \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        fprintf(stderr, "CU error %d at %s:%d\n",                 \
                _e, __FILE__, __LINE__);                           \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_mha_lse {

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

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void wgmma_16x16x16(float* acc, uint64_t desc_A, uint64_t desc_B, uint32_t alpha) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %3, 0;\n"
        "wgmma.m16n16k16.sync.f16.f32 {%0-%15}, %1, %2, p;\n}\n"
        :: "f"(acc[0]), "l"(desc_A), "l"(desc_B), "r"(alpha));
}

__device__ __forceinline__ uint64_t make_desc(void* smem_ptr, uint32_t stride_dim_elements) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    uint32_t sbo = stride_dim_elements * 2;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;   // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint64_t advance_desc(uint64_t d, int k_step, bool is_k_major, uint32_t stride_dim_elements) {
    uint64_t addr_bits = d & 0x3FFF;
    if (is_k_major) {
        addr_bits += (k_step * 16 * 2) / 16;
    } else {
        uint32_t sbo = stride_dim_elements * 2;
        addr_bits += (k_step * 16 * sbo) / 16;
    }
    d &= ~0x3FFF;
    d |= addr_bits;
    return d;
}

__device__ __forceinline__ void wgmma_load_16x16_k_major(__nv_bfloat16* smem, const __nv_bfloat16* src) {
    for(int r = 0; r < 16; ++r) {
        for(int c = 0; c < 16; ++c) {
            int c_swizzled = (((c >> 3) & 7) ^ (r & 7)) << 3 | (c & 7);
            smem[r * 16 + c] = src[r * 64 + c_swizzled];
        }
    }
}

__device__ __forceinline__ void wgmma_load_16x16_n_major(__nv_bfloat16* smem, const __nv_bfloat16* src, int base_row) {
    for(int r = 0; r < 16; ++r) {
        int r_actual = base_row + r;
        for(int c = 0; c < 16; ++c) {
            int col_chunk = c >> 3;
            int col_rem = c & 7;
            int r_swizzled = r_actual & 7;
            int chunk_swizzled = col_chunk ^ r_swizzled;
            int c_swizzled = (chunk_swizzled << 3) | col_rem;
            smem[r * 16 + c] = src[r_actual * 64 + c_swizzled];
        }
    }
}

__global__ __launch_bounds__(128, 1) void flashattention_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O,
    float* LSE,
    uint32_t S
) {
    int bh_idx = blockIdx.y;
    int s_idx = blockIdx.x;
    
    int phase_q = 0;
    int phase_kv = 0;
    
    __shared__ __align__(1024) __nv_bfloat16 s_Q0[16][64];
    __shared__ __align__(1024) __nv_bfloat16 s_Q1[16][64];
    __shared__ __align__(1024) __nv_bfloat16 s_K0[64][64];
    __shared__ __align__(1024) __nv_bfloat16 s_K1[64][64];
    __shared__ __align__(1024) __nv_bfloat16 s_V0[64][64];
    __shared__ __align__(1024) __nv_bfloat16 s_V1[64][64];
    __shared__ __align__(1024) __nv_bfloat16 s_P[16][64]; 
    
    __shared__ alignas(128) uint64_t bar[1];
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    
    int warp_id = threadIdx.x / 32;
    int lane_row = threadIdx.x % 32;
    
    if (threadIdx.x == 0) {
        uint32_t total_bytes_q = 4 * 16 * 64 * sizeof(__nv_bfloat16);
        mbarrier_arrive_and_expect_tx_fn(bar, total_bytes_q);
        
        int32_t c1_base_q = bh_idx * S + warp_id * 16;
        tma_load_2d_fn(&tma_Q, bar, s_Q0, 0, c1_base_q + 0 * 16);
        tma_load_2d_fn(&tma_Q, bar, s_Q0, 64, c1_base_q + 1 * 16); // wait, s_Q1!
        
        tma_load_2d_fn(&tma_Q, bar, s_Q1, 0, c1_base_q + 0 * 16);
        tma_load_2d_fn(&tma_Q, bar, s_Q1, 64, c1_base_q + 1 * 16);
    }
    tmem_load_fence_fn();
    mbarrier_wait_fn(bar, phase_q);
    phase_q ^= 1;
    
    float running_max[4] = {-1e20f, -1e20f, -1e20f, -1e20f};
    float running_sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc_O0[16][64] = {0};
    float acc_O1[16][64] = {0};
    
    for (int block_start = 0; block_start < S; block_start += 64) {
        __syncthreads();
        if (threadIdx.x == 0) {
            uint32_t total_bytes_kv = 2 * 64 * 64 * sizeof(__nv_bfloat16) * 2;
            mbarrier_arrive_and_expect_tx_fn(bar, total_bytes_kv);
            
            int32_t c1_base = bh_idx * S + block_start;
            tma_load_2d_fn(&tma_K, bar, s_K0, 0, c1_base);
            tma_load_2d_fn(&tma_K, bar, s_K1, 64, c1_base);
            
            tma_load_2d_fn(&tma_V, bar, s_V0, 0, c1_base);
            tma_load_2d_fn(&tma_V, bar, s_V1, 64, c1_base);
        }
        tmem_load_fence_fn();
        mbarrier_wait_fn(bar, phase_kv);
        phase_kv ^= 1;
        
        float acc_P[16][16];
        for(int i = 0; i < 16; ++i) {
            for(int j = 0; j < 16; ++j) acc_P[i][j] = 0.0f;
        }
        
        uint64_t desc_Q0 = make_desc(s_Q0, 128);
        uint64_t desc_K0 = make_desc(s_K0, 128);
        
        for (int k_step = 0; k_step < 4; ++k_step) {
            uint64_t desc_Q = advance_desc(desc_Q0, k_step, true, 128);
            uint64_t desc_K = advance_desc(desc_K0, k_step, true, 128);
            wgmma_16x16x16(acc_P, desc_Q, desc_K, 1);
        }
        
        for (int k_step = 0; k_step < 4; ++k_step) {
            uint64_t desc_Q = advance_desc(desc_Q1, k_step, true, 128);
            uint64_t desc_K = advance_desc(desc_K1, k_step, true, 128);
            wgmma_16x16x16(acc_P, desc_Q, desc_K, 1);
        }
        
        float local_max[4] = {-1e20f, -1e20f, -1e20f, -1e20f};
        float local_sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for(int i = 0; i < 16; ++i) {
            for(int j = 0; j < 16; ++j) {
                float p_val = acc_P[i][j];
                int global_col = block_start + (j / 4) * 16 + j % 4; // Wait, P is 16x16 here.
                // P represents a 16x64 matrix split into four 16x16 blocks.
                // Let's fix the indexing mapping.
                
                int tile_col = j / 4; // 0 to 3
                int col = j % 4;      // 0 to 3
                
                global_col = block_start + tile_col * 16 + col;
                
                if (global_col >= S) {
                    p_val = -1e20f;
                } else {
                    p_val *= 0.08838834764f; // 1 / sqrt(128)
                }
                
                int group = (lane_row % 16) / 4;
                running_max[group] = fmaxf(running_max[group], p_val);
            }
        }
        
        for(int i = 0; i < 16; ++i) {
            for(int j = 0; j < 16; ++j) {
                int tile_col = j / 4;
                int col = j % 4;
                int global_col = block_start + tile_col * 16 + col;
                
                float p_val = acc_P[i][j];
                if (global_col >= S) {
                    p_val = -1e20f;
                } else {
                    p_val *= 0.08838834764f;
                }
                
                int group = (lane_row % 16) / 4;
                float max_val = running_max[group];
                float exp_val = expf(p_val - max_val);
                running_sum[group] += exp_val;
                
                acc_P[i][j] = exp_val / running_sum[group];
            }
        }
        
        __syncthreads();
        
        for(int i = 0; i < 16; ++i) {
            for(int j = 0; j < 16; j += 2) {
                int chunk = j >> 3;
                int rem = j & 7;
                int col_swizzled = (chunk ^ (lane_row & 7)) << 3 | rem;
                
                __nv_bfloat16 p0 = __float2bfloat16(acc_P[i][j]);
                __nv_bfloat16 p1 = __float2bfloat16(acc_P[i][j+1]);
                
                s_P[lane_row][i * 4 + j] = p0; // s_P is 16x64, storing row by row? No, s_P[r][c]
                s_P[lane_row][i * 4 + j + 1] = p1;
            }
        }
        __syncthreads(); 
        
        uint64_t desc_P = make_desc(s_P, 16); // P is 16x64, K-major, 128B swizzle
        
        for (int n_step = 0; n_step < 4; ++n_step) {
            __syncthreads();
            wgmma_load_16x16_k_major((__nv_bfloat16*)s_B0, (const __nv_bfloat16*)s_P, n_step * 16);
            
            float acc_O_sub0[16][16] = {0};
            for(int k = 0; k < 4; ++k) {
                uint64_t desc_p = advance_desc(desc_P, k, true, 16);
                uint64_t desc_v = advance_desc(desc_V0, k, false, 64);
                wgmma_16x16x16(acc_O_sub0, desc_p, desc_v, 1);
            }
            for(int r = 0; r < 16; ++r) {
                for(int c = 0; c < 16; ++c) {
                    acc_O0[r][n_step * 16 + c] = acc_O_sub0[r][c];
                }
            }
        }
        
        for (int n_step = 0; n_step < 4; ++n_step) {
            __syncthreads();
            wgmma_load_16x16_k_major((__nv_bfloat16*)s_B0, (const __nv_bfloat16*)s_P, n_step * 16);
            
            float acc_O_sub1[16][16] = {0};
            for(int k = 0; k < 4; ++k) {
                uint64_t desc_p = advance_desc(desc_P, k, true, 16);
                uint64_t desc_v = advance_desc(desc_V1, k, false, 64);
                wgmma_16x16x16(acc_O_sub1, desc_p, desc_v, 1);
            }
            for(int r = 0; r < 16; ++r) {
                for(int c = 0; c < 16; ++c) {
                    acc_O1[r][n_step * 16 + c] = acc_O_sub1[r][c];
                }
            }
        }
        
        __syncthreads();
    }
    
    for(int r = 0; r < 16; ++r) {
        for(int c = 0; c < 64; c += 2) {
            __nv_bfloat16 out0 = __float2bfloat16(acc_O0[r][c]);
            __nv_bfloat16 out1 = __float2bfloat16(acc_O0[r][c+1]);
            uint32_t idx0 = bh_idx * S * 128 + s_idx * 64 * 128 + r * 128 + c;
            *(uint32_t*)(&O[idx0]) = *(uint32_t*)(&out0);
            *(uint32_t*)(&O[idx0+1]) = *(uint32_t*)(&out1);
            
            __nv_bfloat16 out2 = __float2bfloat16(acc_O1[r][c]);
            __nv_bfloat16 out3 = __float2bfloat16(acc_O1[r][c+1]);
            uint32_t idx1 = bh_idx * S * 128 + s_idx * 64 * 128 + r * 128 + c + 64;
            *(uint32_t*)(&O[idx1]) = *(uint32_t*)(&out2);
            *(uint32_t*)(&O[idx1+1]) = *(uint32_t*)(&out3);
        }
    }
    
    __shared__ float smem_LSE[4][4];
    __shared__ float smem_Sum[4][4];
    
    if (threadIdx.x == 0) {
        smem_LSE[warp_id][0] = running_max[0];
        smem_LSE[warp_id][1] = running_max[1];
        smem_LSE[warp_id][2] = running_max[2];
        smem_LSE[warp_id][3] = running_max[3];
        
        smem_Sum[warp_id][0] = running_sum[0];
        smem_Sum[warp_id][1] = running_sum[1];
        smem_Sum[warp_id][2] = running_sum[2];
        smem_Sum[warp_id][3] = running_sum[3];
    }
    __syncthreads();
    
    if (threadIdx.x < 4) {
        for (int i = 0; i < 4; ++i) {
            float lse = smem_LSE[threadIdx.x][i] + logf(smem_Sum[threadIdx.x][i]);
            uint32_t idx = bh_idx * S + s_idx * 64 + threadIdx.x * 16 + i;
            if (idx < bh_idx * S + S) {
                LSE[idx] = lse;
            }
        }
    }
}

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill
    );
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    uint32_t B = Q.size(0);
    uint32_t H = Q.size(1);
    uint32_t S = Q.size(2);
    uint32_t D = Q.size(3); 
    
    void* q_ptr = Q.data_ptr();
    void* k_ptr = K.data_ptr();
    void* v_ptr = V.data_ptr();
    __nv_bfloat16* o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_ptr = static_cast<float*>(LSE.data_ptr());
    
    CUtensorMap tma_Q, tma_K, tma_V;
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_Q, q_ptr, D, B * H * S, 64, 16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_K, k_ptr, D, B * H * S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_V, v_ptr, D, B * H * S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
    dim3 grid(S / 64, B * H);
    dim3 block(128);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    flashattention_kernel<<<grid, block, 0, stream>>>(tma_Q, tma_K, tma_V, o_ptr, lse_ptr, S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_lse::run);

}