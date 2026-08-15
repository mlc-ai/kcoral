#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
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

namespace tvm_ffi_flash_attention_bwd {

__device__ unsigned int __grid_sync_count = 0;
__device__ volatile int __grid_sync_sense = 0;

__device__ __forceinline__ void grid_sync_fn() {
    __syncthreads();
    __threadfence();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        unsigned int num_blocks = gridDim.x * gridDim.y * gridDim.z;
        unsigned int arrived = atomicAdd(&__grid_sync_count, 1);
        if (arrived == num_blocks - 1) {
            __grid_sync_count = 0;
            __threadfence();
            __grid_sync_sense ^= 1;
        } else {
            int expected = __grid_sync_sense ^ 1;
            while (__grid_sync_sense != expected) {
                // spin wait
            }
        }
    }
    __syncthreads();
}

__device__ void load_tile(__nv_bfloat16* smem, const __nv_bfloat16* gmem, uint32_t row_start, uint32_t col_start, uint32_t S, uint32_t d) {
    for (uint32_t idx = threadIdx.x; idx < 64 * 64; idx += blockDim.x) {
        uint32_t row = idx / 64;
        uint32_t col = idx % 64;
        if (row_start + row < S && col_start + col < d) {
            smem[idx] = gmem[(row_start + row) * d + (col_start + col)];
        } else {
            smem[idx] = __float2bfloat16(0.0f);
        }
    }
}

__device__ void load_lse(float* smem_LSE, const float* gmem_L, uint32_t row_start, uint32_t S) {
    for (uint32_t idx = threadIdx.x; idx < 64; idx += blockDim.x) {
        if (row_start + idx < S) {
            smem_LSE[idx] = gmem_L[row_start + idx];
        } else {
            smem_LSE[idx] = 0.0f;
        }
    }
}

__device__ void compute_D(__nv_bfloat16* smem_O0, __nv_bfloat16* smem_O1, 
                          __nv_bfloat16* smem_dO0, __nv_bfloat16* smem_dO1, 
                          float* smem_D) {
    if (threadIdx.x < 64) {
        uint32_t row = threadIdx.x;
        float d_val = 0;
        for(uint32_t d = 0; d < 64; ++d) {
            d_val += __bfloat162float(smem_O0[row * 64 + d]) * __bfloat162float(smem_dO0[row * 64 + d]);
            d_val += __bfloat162float(smem_O1[row * 64 + d]) * __bfloat162float(smem_dO1[row * 64 + d]);
        }
        smem_D[row] = d_val;
    }
    __syncthreads();
}

__global__ void __launch_bounds__(128) flash_attention_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    uint32_t S)
{
    extern __shared__ char smem_raw[];
    __nv_bfloat16* smem_Q0 = (__nv_bfloat16*)smem_raw;
    __nv_bfloat16* smem_Q1 = smem_Q0 + 64*64;
    __nv_bfloat16* smem_K0 = smem_Q1 + 64*64;
    __nv_bfloat16* smem_K1 = smem_K0 + 64*64;
    __nv_bfloat16* smem_V0 = smem_K1 + 64*64;
    __nv_bfloat16* smem_V1 = smem_V0 + 64*64;
    __nv_bfloat16* smem_dO0 = smem_V1 + 64*64;
    __nv_bfloat16* smem_dO1 = smem_dO0 + 64*64;
    __nv_bfloat16* smem_O0 = smem_dO1 + 64*64;
    __nv_bfloat16* smem_O1 = smem_O0 + 64*64;
    __nv_bfloat16* smem_P = smem_O1 + 64*64;
    __nv_bfloat16* smem_dS = smem_P + 64*64;
    float* smem_LSE = (float*)(smem_dS + 64*64);
    float* smem_D = smem_LSE + 64;

    uint32_t query_tile = blockIdx.x;
    uint32_t batch_head = blockIdx.y;
    float scale = 1.0f / sqrtf(128.0f); // d=128

    const __nv_bfloat16* my_Q = Q + batch_head * S * 128;
    const __nv_bfloat16* my_K = K + batch_head * S * 128;
    const __nv_bfloat16* my_V = V + batch_head * S * 128;
    const __nv_bfloat16* my_O = O + batch_head * S * 128;
    const __nv_bfloat16* my_dO = dO + batch_head * S * 128;
    const float* my_L = L + batch_head * S;

    __nv_bfloat16* my_dQ = dQ + batch_head * S * 128;
    __nv_bfloat16* my_dK = dK + batch_head * S * 128;
    __nv_bfloat16* my_dV = dV + batch_head * S * 128;

    // ==========================================
    // PASS 1: Compute dQ
    // ==========================================
    if (query_tile * 64 < S) {
        load_tile(smem_Q0, my_Q, query_tile * 64, 0, S, 128);
        load_tile(smem_Q1, my_Q, query_tile * 64, 64, S, 128);
        load_tile(smem_dO0, my_dO, query_tile * 64, 0, S, 128);
        load_tile(smem_dO1, my_dO, query_tile * 64, 64, S, 128);
        load_tile(smem_O0, my_O, query_tile * 64, 0, S, 128);
        load_tile(smem_O1, my_O, query_tile * 64, 64, S, 128);
        load_lse(smem_LSE, my_L, query_tile * 64, S);
        __syncthreads();

        compute_D(smem_O0, smem_O1, smem_dO0, smem_dO1, smem_D);

        if (threadIdx.x < 64) {
            float my_dQ_acc0[64] = {0};
            float my_dQ_acc1[64] = {0};

            for (uint32_t k_tile = 0; k_tile <= query_tile; ++k_tile) {
                load_tile(smem_K0, my_K, k_tile * 64, 0, S, 128);
                load_tile(smem_K1, my_K, k_tile * 64, 64, S, 128);
                load_tile(smem_V0, my_V, k_tile * 64, 0, S, 128);
                load_tile(smem_V1, my_V, k_tile * 64, 64, S, 128);
                
                uint32_t tid = threadIdx.x;
                uint32_t query_start = query_tile * 64;
                uint32_t key_start = k_tile * 64;

                for (uint32_t j = 0; j < 64; ++j) {
                    float s = 0;
                    for (uint32_t d = 0; d < 64; ++d) {
                        s += __bfloat162float(smem_Q0[tid * 64 + d]) * __bfloat162float(smem_K0[j * 64 + d]);
                    }
                    for (uint32_t d = 0; d < 64; ++d) {
                        s += __bfloat162float(smem_Q1[tid * 64 + d]) * __bfloat162float(smem_K1[j * 64 + d]);
                    }
                    s *= scale;
                    
                    if (query_start + tid < key_start + j || query_start + tid >= S || key_start + j >= S) {
                        s = 0;
                    } else {
                        s = expf(s - smem_LSE[tid]);
                    }
                    smem_P[tid * 64 + j] = __float2bfloat16(s);

                    float dp = 0;
                    for (uint32_t d = 0; d < 64; ++d) {
                        dp += __bfloat162float(smem_V0[j * 64 + d]) * __bfloat162float(smem_dO0[tid * 64 + d]);
                    }
                    for (uint32_t d = 0; d < 64; ++d) {
                        dp += __bfloat162float(smem_V1[j * 64 + d]) * __bfloat162float(smem_dO1[tid * 64 + d]);
                    }

                    float ds = s * (dp - smem_D[tid]);
                    if (query_start + tid < key_start + j || query_start + tid >= S || key_start + j >= S) {
                        ds = 0;
                    }
                    smem_dS[tid * 64 + j] = __float2bfloat16(ds);
                }

                for (uint32_t j = 0; j < 64; ++j) {
                    float ds = __bfloat162float(smem_dS[tid * 64 + j]);
                    if (ds != 0) {
                        for (uint32_t d = 0; d < 64; ++d) {
                            my_dQ_acc0[d] += ds * __bfloat162float(smem_K0[j * 64 + d]) * scale;
                        }
                        for (uint32_t d = 0; d < 64; ++d) {
                            my_dQ_acc1[d] += ds * __bfloat162float(smem_K1[j * 64 + d]) * scale;
                        }
                    }
                }
            }
            
            uint32_t row = threadIdx.x;
            if (query_tile * 64 + row < S) {
                uint64_t offset = (uint64_t)batch_head * S * 128 + (uint64_t)query_tile * 64 * 128 + (uint64_t)row * 128;
                for (uint32_t d = 0; d < 64; ++d) {
                    my_dQ[offset + d] = __float2bfloat16(my_dQ_acc0[d]);
                    my_dQ[offset + 64 + d] = __float2bfloat16(my_dQ_acc1[d]);
                }
            }
            __syncthreads();
        }
    }

    grid_sync_fn();

    // ==========================================
    // PASS 2: Compute dK and dV
    // ==========================================
    uint32_t k_tile = blockIdx.x;
    if (k_tile * 64 < S) {
        load_tile(smem_K0, my_K, k_tile * 64, 0, S, 128);
        load_tile(smem_K1, my_K, k_tile * 64, 64, S, 128);
        load_tile(smem_V0, my_V, k_tile * 64, 0, S, 128);
        load_tile(smem_V1, my_V, k_tile * 64, 64, S, 128);
        __syncthreads();

        float my_dK_acc[64] = {0};
        float my_dV_acc[64] = {0};
        uint32_t my_j = threadIdx.x % 64;
        uint32_t d_half = threadIdx.x / 64;

        for (uint32_t q_tile = k_tile; q_tile * 64 < S; ++q_tile) {
            load_tile(smem_Q0, my_Q, q_tile * 64, 0, S, 128);
            load_tile(smem_Q1, my_Q, q_tile * 64, 64, S, 128);
            load_tile(smem_dO0, my_dO, q_tile * 64, 0, S, 128);
            load_tile(smem_dO1, my_dO, q_tile * 64, 64, S, 128);
            load_tile(smem_O0, my_O, q_tile * 64, 0, S, 128);
            load_tile(smem_O1, my_O, q_tile * 64, 64, S, 128);
            load_lse(smem_LSE, my_L, q_tile * 64, S);
            __syncthreads();

            compute_D(smem_O0, smem_O1, smem_dO0, smem_dO1, smem_D);

            if (threadIdx.x < 64) {
                uint32_t tid = threadIdx.x;
                uint32_t query_start = q_tile * 64;
                uint32_t key_start = k_tile * 64;

                for (uint32_t j = 0; j < 64; ++j) {
                    float s = 0;
                    for (uint32_t d = 0; d < 64; ++d) {
                        s += __bfloat162float(smem_Q0[tid * 64 + d]) * __bfloat162float(smem_K0[j * 64 + d]);
                    }
                    for (uint32_t d = 0; d < 64; ++d) {
                        s += __bfloat162float(smem_Q1[tid * 64 + d]) * __bfloat162float(smem_K1[j * 64 + d]);
                    }
                    s *= scale;
                    
                    if (query_start + tid < key_start + j || query_start + tid >= S || key_start + j >= S) {
                        s = 0;
                    } else {
                        s = expf(s - smem_LSE[tid]);
                    }
                    smem_P[tid * 64 + j] = __float2bfloat16(s);

                    float dp = 0;
                    for (uint32_t d = 0; d < 64; ++d) {
                        dp += __bfloat162float(smem_V0[j * 64 + d]) * __bfloat162float(smem_dO0[tid * 64 + d]);
                    }
                    for (uint32_t d = 0; d < 64; ++d) {
                        dp += __bfloat162float(smem_V1[j * 64 + d]) * __bfloat162float(smem_dO1[tid * 64 + d]);
                    }

                    float ds = s * (dp - smem_D[tid]);
                    if (query_start + tid < key_start + j || query_start + tid >= S || key_start + j >= S) {
                        ds = 0;
                    }
                    smem_dS[tid * 64 + j] = __float2bfloat16(ds);
                }
            }
            __syncthreads();

            __nv_bfloat16* Q_ptr = d_half == 0 ? smem_Q0 : smem_Q1;
            __nv_bfloat16* K_ptr = d_half == 0 ? smem_K0 : smem_K1;
            __nv_bfloat16* V_ptr = d_half == 0 ? smem_V0 : smem_V1;
            __nv_bfloat16* dO_ptr = d_half == 0 ? smem_dO0 : smem_dO1;

            for (uint32_t i = 0; i < 64; ++i) {
                float ds = __bfloat162float(smem_dS[i * 64 + my_j]);
                if (ds != 0) {
                    for (uint32_t d = 0; d < 64; ++d) {
                        my_dK_acc[d] += ds * __bfloat162float(Q_ptr[i * 64 + d]) * scale;
                    }
                }
                float p = __bfloat162float(smem_P[i * 64 + my_j]);
                if (p != 0) {
                    for (uint32_t d = 0; d < 64; ++d) {
                        my_dV_acc[d] += p * __bfloat162float(dO_ptr[i * 64 + d]);
                    }
                }
            }
            __syncthreads();
        }

        if (k_tile * 64 + my_j < S) {
            uint64_t offset = (uint64_t)batch_head * S * 128 + (uint64_t)k_tile * 64 * 128 + (uint64_t)my_j * 128 + (uint64_t)d_half * 64;
            for (uint32_t d = 0; d < 64; ++d) {
                my_dK[offset + d] = __float2bfloat16(my_dK_acc[d]);
                my_dV[offset + d] = __float2bfloat16(my_dV_acc[d]);
            }
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id)); // This is necessary to ensure the correct GPU is used, especially in multi-GPU setups.
    
    uint32_t B = Q.size(0);
    uint32_t H = Q.size(1);
    uint32_t S = Q.size(2);
    
    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_ptr = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_ptr = static_cast<const float*>(L.data_ptr());
    
    __nv_bfloat16* dQ_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    uint32_t num_tiles = (S + 63) / 64;
    dim3 grid(num_tiles, B * H);
    dim3 block(128);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    cudaFuncSetAttribute(flash_attention_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 102400);
    
    flash_attention_bwd_kernel<<<grid, block, 102400, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
        dQ_ptr, dK_ptr, dV_ptr, S);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}  // namespace tvm_ffi_flash_attention_bwd