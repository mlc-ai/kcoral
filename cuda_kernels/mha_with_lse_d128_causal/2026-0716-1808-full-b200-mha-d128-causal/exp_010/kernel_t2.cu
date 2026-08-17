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

__device__ __forceinline__ void tma_load_multicast_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, uint16_t mask) {
    uint64_t cache_hint = 0;
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%0], [%1, {%4, %5}], [%2], %3, %6;"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "h"(mask), "r"(c0), "r"(c1), "l"(cache_hint) : "memory");
}

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
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


namespace tvm_ffi_mha {

__global__ __launch_bounds__(128, 2) void mha_kernel(const __grid_constant__ CUtensorMap tma_Q,
                          const __grid_constant__ CUtensorMap tma_K,
                          const __grid_constant__ CUtensorMap tma_V,
                          __nv_bfloat16* O, float* LSE, uint32_t S) {
    
    extern __shared__ __align__(128) uint8_t smem_buf[];
    uint64_t* mbar_Q = (uint64_t*)smem_buf;
    uint64_t* mbar_K0 = (uint64_t*)(smem_buf + 8);
    uint64_t* mbar_K1 = (uint64_t*)(smem_buf + 16);
    uint64_t* mbar_V0 = (uint64_t*)(smem_buf + 24);
    uint64_t* mbar_V1 = (uint64_t*)(smem_buf + 32);
    
    // Dynamically align SMEM base to 1024 bytes limit to satisfy swizzle constraints
    uint32_t smem_base = ((uint32_t)__cvta_generic_to_shared(smem_buf) + 1023) & ~1023;
    __nv_bfloat16* Q_0 = (__nv_bfloat16*)(smem_base);         
    __nv_bfloat16* Q_1 = (__nv_bfloat16*)(smem_base + 8192);    
    __nv_bfloat16* P_col = (__nv_bfloat16*)(smem_base + 16384);  
    __nv_bfloat16* K_0[2] = { (__nv_bfloat16*)(smem_base + 24576), (__nv_bfloat16*)(smem_base + 32768) };
    __nv_bfloat16* K_1[2] = { (__nv_bfloat16*)(smem_base + 32768), (__nv_bfloat16*)(smem_base + 40960) };
    __nv_bfloat16* V_0[2] = { (__nv_bfloat16*)(smem_base + 49152), (__nv_bfloat16*)(smem_base + 65536) };
    __nv_bfloat16* V_1[2] = { (__nv_bfloat16*)(smem_base + 57344), (__nv_bfloat16*)(smem_base + 73728) };
    float* P_fp32 = (float*)(smem_base + 4096); 
    
    uint32_t i = blockIdx.x;
    uint32_t batch_head_idx = blockIdx.y;
    uint32_t cta_id = cluster_rank_fn() % 2;
    uint32_t coord0_v = cta_id * 64;
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_Q, 2);
        init_smem_barrier_fn(mbar_K0, 2);
        init_smem_barrier_fn(mbar_K1, 2);
        init_smem_barrier_fn(mbar_V0, 2);
        init_smem_barrier_fn(mbar_V1, 2);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    
    int phase_Q = 0;
    uint32_t coord1_q = (i * 64 + 0) + batch_head_idx * S;
    
    if (threadIdx.x == 0 && cta_id == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 4 * 8192);
        tma_load_multicast_2d_fn(&tma_Q, mbar_Q, Q_0, 0, coord1_q, 0x3);
        tma_load_multicast_2d_fn(&tma_Q, mbar_Q, Q_1, 64, coord1_q, 0x3);
    } else if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 0);
    }
    mbarrier_wait_fn(mbar_Q, phase_Q);
    
    float max_val[16], sum_exp[16];
    for (int r = 0; r < 16; ++r) {
        max_val[r] = -1e20f;
        sum_exp[r] = 0.0f;
    }
    
    int warp_id = threadIdx.x / 32;
    int warp_row = warp_id * 16;
    
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> O_frag[4];
    for(int c = 0; c < 4; ++c) {
        wmma::fill_fragment(O_frag[c], 0.0f);
    }
    
    int phase_K0 = 0, phase_K1 = 0, phase_V0 = 0, phase_V1 = 0;
    
    if (0 <= i) {
        if (threadIdx.x == 0 && cta_id == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_K0, 8192);
            tma_load_multicast_2d_fn(&tma_K, mbar_K0, K_0[0], 0, (0 * 64) + batch_head_idx * S, 0x3);
            mbarrier_arrive_and_expect_tx_fn(mbar_K1, 8192);
            tma_load_multicast_2d_fn(&tma_K, mbar_K1, K_1[0], 64, (0 * 64) + batch_head_idx * S, 0x3);
        } else if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_K0, 0);
            mbarrier_arrive_and_expect_tx_fn(mbar_K1, 0);
        }
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_V0, 8192);
            tma_load_2d_fn(&tma_V, mbar_V0, V_0[0], coord0_v, (0 * 64) + batch_head_idx * S);
            if (cta_id == 1) {
                mbarrier_arrive_and_expect_tx_fn(mbar_V1, 8192);
                tma_load_2d_fn(&tma_V, mbar_V1, V_1[0], coord0_v, (0 * 64) + batch_head_idx * S);
            }
        }
    }
    
    int num_iters = (i * 64 + 64) / 64;
    
    for (int j = 0; j < num_iters; ++j) {
        int buf_idx = j % 2;
        int next_buf_idx = (j + 1) % 2;
        
        if (j + 1 < num_iters) {
            if (threadIdx.x == 0 && cta_id == 0) {
                mbarrier_arrive_and_expect_tx_fn(mbar_K0, 8192);
                tma_load_multicast_2d_fn(&tma_K, mbar_K0, K_0[next_buf_idx], 0, ((j + 1) * 64) + batch_head_idx * S, 0x3);
                mbarrier_arrive_and_expect_tx_fn(mbar_K1, 8192);
                tma_load_multicast_2d_fn(&tma_K, mbar_K1, K_1[next_buf_idx], 64, ((j + 1) * 64) + batch_head_idx * S, 0x3);
            } else if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(mbar_K0, 0);
                mbarrier_arrive_and_expect_tx_fn(mbar_K1, 0);
            }
            
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(mbar_V0, 8192);
                tma_load_2d_fn(&tma_V, mbar_V0, V_0[next_buf_idx], coord0_v, ((j + 1) * 64) + batch_head_idx * S);
                if (cta_id == 1) {
                    mbarrier_arrive_and_expect_tx_fn(mbar_V1, 8192);
                    tma_load_2d_fn(&tma_V, mbar_V1, V_1[next_buf_idx], coord0_v, ((j + 1) * 64) + batch_head_idx * S);
                }
            }
        }
        
        mbarrier_wait_fn(mbar_K0, phase_K0);
        mbarrier_wait_fn(mbar_K1, phase_K1);
        mbarrier_wait_fn(mbar_V0, phase_V0);
        mbarrier_wait_fn(mbar_V1, phase_V1);
        
        phase_K0 ^= 1;
        phase_K1 ^= 1;
        phase_V0 ^= 1;
        phase_V1 ^= 1;
        
        __nv_bfloat16* K_0_buf = K_0[buf_idx];
        __nv_bfloat16* K_1_buf = K_1[buf_idx];
        __nv_bfloat16* V_0_buf = V_0[buf_idx];
        __nv_bfloat16* V_1_buf = V_1[buf_idx];
        
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> P_frag[4];
        for(int c = 0; c < 4; ++c) {
            wmma::fill_fragment(P_frag[c], 0.0f);
        }
        
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a0[4];
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a1[4];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b0_row[4];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b1_row[4];
        
        for (int k = 0; k < 4; ++k) {
            wmma::load_matrix_sync(a0[k], Q_0 + (warp_row * 64 + k * 16), 64);
            wmma::load_matrix_sync(a1[k], Q_1 + (warp_row * 64 + k * 16), 64);
            
            wmma::load_matrix_sync(b0_row[k], K_0_buf + (k * 16 * 64 + warp_row), 64);
            wmma::load_matrix_sync(b1_row[k], K_1_buf + (k * 16 * 64 + warp_row), 64);
        }
        
        for (int r_c = 0; r_c < 4; ++r_c) {
            for (int k = 0; k < 4; ++k) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b0_col[k];
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b1_col[k];
                
                wmma::load_matrix_sync(b0_col[k], K_0_buf + (r_c * 16 * 64 + k * 16), 64);
                wmma::load_matrix_sync(b1_col[k], K_1_buf + (r_c * 16 * 64 + k * 16), 64);
                
                for(int i_a = 0; i_a < 4; ++i_a) {
                    wmma::mma_sync(P_frag[r_c], a0[i_a], b0_col[k], P_frag[r_c]);
                    wmma::mma_sync(P_frag[r_c], a1[i_a], b1_col[k], P_frag[r_c]);
                }
            }
        }
        
        for(int c = 0; c < 4; ++c) {
            wmma::store_matrix_sync(&P_fp32[(warp_row * 64) + (c * 16)], P_frag[c], 64, wmma::mem_row_major);
        }
        __syncthreads(); 
        
        for (int idx = threadIdx.x; idx < 64 * 64; idx += blockDim.x) {
            int r = idx / 64;
            int c = idx % 64;
            if (j * 64 + c > i * 64 + r) {
                P_fp32[idx] = -1e20f;
            } else {
                P_fp32[idx] /= 11.3137f; // sqrt(128)
            }
        }
        
        float row_max[64], row_sum[64];
        for (int r = threadIdx.x; r < 64; r += blockDim.x) {
            float m = -1e20f;
            for (int c = 0; c < 64; ++c) {
                if (j * 64 + c <= i * 64 + r) {
                    m = fmaxf(m, P_fp32[r * 64 + c]);
                }
            }
            row_max[r] = m;
            
            float s = 0.0f;
            for (int c = 0; c < 64; ++c) {
                if (j * 64 + c <= i * 64 + r) {
                    float p = P_fp32[r * 64 + c];
                    float exp_p = expf(p - m);
                    s += exp_p;
                    P_fp32[r * 64 + c] = exp_p;
                } else {
                    P_fp32[r * 64 + c] = 0.0f;
                }
            }
            row_sum[r] = s;
        }
        __syncthreads(); 
        
        if (threadIdx.x < 64) {
            int r = threadIdx.x;
            float m_prev = max_val[r % 16];
            max_val[r % 16] = fmaxf(m_prev, row_max[r]);
            sum_exp[r % 16] *= expf(m_prev - max_val[r % 16]);
            sum_exp[r % 16] += row_sum[r] * expf(row_max[r] - max_val[r % 16]);
            
            float total_s = sum_exp[r % 16];
            for (int c = 0; c < 64; ++c) {
                P_fp32[r * 64 + c] /= total_s;
            }
        }
        __syncthreads();
        
        for (int idx = threadIdx.x; idx < 64 * 64; idx += blockDim.x) {
            P_col[idx] = __float2bfloat16(P_fp32[idx]);
        }
        __syncthreads();
        
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a_p[4];
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_v0[4];
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_v1[4];
        
        for (int k = 0; k < 4; ++k) {
            wmma::load_matrix_sync(a_p[k], P_col + (warp_row * 64 + k * 16), 64);
            wmma::load_matrix_sync(b_v0[k], V_0_buf + (k * 16 * 64 + warp_row), 64);
            wmma::load_matrix_sync(b_v1[k], V_1_buf + (k * 16 * 64 + warp_row), 64);
        }
        
        for (int r_c = 0; r_c < 4; ++r_c) {
            for (int k = 0; k < 4; ++k) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> B_v0_col[k];
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> B_v1_col[k];
                
                wmma::load_matrix_sync(B_v0_col[k], V_0_buf + (r_c * 16 * 64 + k * 16), 64);
                wmma::load_matrix_sync(B_v1_col[k], V_1_buf + (r_c * 16 * 64 + k * 16), 64);
                
                wmma::mma_sync(O_frag[r_c], a_p[k], B_v0_col[k], O_frag[r_c]);
                wmma::mma_sync(O_frag[r_c], a_p[k], B_v1_col[k], O_frag[r_c]);
            }
        }
    } 
    
    __nv_bfloat16* O_smem = (__nv_bfloat16*)(smem_base + 24576); 
    
    for(int c = 0; c < 4; ++c) {
        wmma::store_matrix_sync(&O_smem[(warp_row * 64) + (c * 16)], O_frag[c], 64, wmma::mem_row_major);
    }
    __syncthreads();
    
    for (int idx = threadIdx.x; idx < 64 * 64; idx += blockDim.x) {
        int r = idx / 64;
        int c = idx % 64;
        if ((i * 64 + r) < S) {
            O[(i * 64 + r) * 128 + (cta_id * 64 + c)] = O_smem[idx];
        }
    }
    
    if (threadIdx.x < 64 && (i * 64 + threadIdx.x) < S) {
        LSE[(i * 64 + threadIdx.x)] = max_val[threadIdx.x] + logf(sum_exp[threadIdx.x]);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView lse) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    uint32_t B = 4, H = 48, S = Q.size(2), D = 128;
    
    CUtensorMap tma_Q, tma_K, tma_V;
    __nv_bfloat16* q_ptr = static_cast<__nv_bfloat16*>(Q.data_ptr());
    __nv_bfloat16* k_ptr = static_cast<__nv_bfloat16*>(K.data_ptr());
    __nv_bfloat16* v_ptr = static_cast<__nv_bfloat16*>(V.data_ptr());
    
    create_tma_2d_descriptor_2B(&tma_Q, q_ptr, D, B * H * S, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_K, k_ptr, D, B * H * S, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_V, v_ptr, D, B * H * S, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    __nv_bfloat16* o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_ptr = static_cast<float*>(lse.data_ptr());
    
    dim3 grid((S + 63) / 64, B * H);
    dim3 block(128);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int smem_bytes = 100 * 1024; 
    
    CUDA_CHECK(cudaFuncSetAttribute(mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_bytes;
    config.stream = stream;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    CUDA_CHECK(cudaLaunchKernelEx(&config, mha_kernel, tma_Q, tma_K, tma_V, o_ptr, lse_ptr, S));
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_mha