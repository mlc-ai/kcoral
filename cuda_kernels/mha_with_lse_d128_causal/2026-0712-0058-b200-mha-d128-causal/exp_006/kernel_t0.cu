#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <stdio.h>
#include <cmath>
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

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void tma_load_4d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}

CUresult create_tma_4d_descriptor(CUtensorMap* d, void* globalAddress,
                                  uint64_t dim0, uint64_t dim1, uint64_t dim2, uint64_t dim3,
                                  uint32_t box0, uint32_t box1,
                                  CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[4] = {dim0, dim1, dim2, dim3};
    cuuint64_t globalStrides[3] = {dim0 * 2, dim0 * dim1 * 2, dim0 * dim1 * dim2 * 2};
    cuuint32_t boxDim[4] = {box0, box1, 1, 1};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
        globalAddress, globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
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

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (0u << 15);   
    d |= (0u << 16);   
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_b_mn_major_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (0u << 15);   
    d |= (1u << 16);   
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ void umma_f16_cg1_fn(uint32_t* tmem_c_fp, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"((uint32_t)tmem_c_fp), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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

__device__ __forceinline__ int swizzle_128B(int row, int col) {
    return ((row % 8) ^ (col / 8)) * 8 + (col % 8);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3); 
    
    if (S == 0) return;
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUtensorMap tma_Q, tma_K, tma_V;
    create_tma_4d_descriptor(&tma_Q, Q.data_ptr(), D, S, H, B, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B);
    create_tma_4d_descriptor(&tma_K, K.data_ptr(), D, S, H, B, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B);
    create_tma_4d_descriptor(&tma_V, V.data_ptr(), D, S, H, B, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B);
    
    int64_t O_bytes = B * H * S * D * sizeof(__nv_bfloat16);
    cudaMemsetAsync(O.data_ptr(), 0, O_bytes, stream);
    
    uint32_t smem_size = 57344;
    dim3 grid((S + 63) / 64, H, B);
    dim3 block(128);
    
    cudaFuncSetAttribute(run_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    run_kernel<<<grid, block, smem_size, stream>>>(tma_Q, tma_K, tma_V, O.data_ptr(), LSE.data_ptr(), S, H, B);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

__global__ void run_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O_ptr, float* LSE_ptr,
    int S, int num_heads, int batch_size) 
{
    extern __shared__ __align__(128) uint8_t smem_pool[];
    uintptr_t pool_addr = (uintptr_t)smem_pool;
    uint8_t* cur = smem_pool + (1024 - (pool_addr % 1024)) % 1024;
    
    __nv_bfloat16* s_Q0 = (__nv_bfloat16*)cur;
    __nv_bfloat16* s_Q1 = s_Q0 + 4096;
    __nv_bfloat16* s_K0 = s_Q1 + 4096;
    __nv_bfloat16* s_K1 = s_K0 + 4096;
    __nv_bfloat16* s_V0 = s_K1 + 4096;
    __nv_bfloat16* s_V1 = s_V0 + 4096;
    __nv_bfloat16* s_P  = s_V1 + 4096;
    uint64_t* mbar = (uint64_t*)(s_P + 4096);
    
    int head_idx = blockIdx.y;
    int batch_idx = blockIdx.z;
    int q_base = blockIdx.x * 64;
    int global_q_base = batch_idx * num_heads * S + head_idx * S + q_base;
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&mbar[0], 1);
        init_smem_barrier_fn(&mbar[1], 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&mbar[0], 16384);
        tma_load_4d_fn(&tma_Q, &mbar[0], s_Q0, 0, q_base, head_idx, batch_idx);
        tma_load_4d_fn(&tma_Q, &mbar[0], s_Q1, 64, q_base, head_idx, batch_idx);
    }
    mbarrier_wait_fn(&mbar[0], 0);
    __syncthreads();
    
    int num_steps = (q_base / 64);
    if (num_steps > (S - 1) / 64) {
        num_steps = (S - 1) / 64;
    }
    
    float lse_sum = 0;
    float lse_max = -INFINITY;
    float prev_max = -INFINITY;
    float O_scaled[16][2] = {0};
    
    uint32_t stride_q = sizeof(__nv_bfloat16);
    uint32_t stride_v = 64 * sizeof(__nv_bfloat16);
    
    int next_barrier_phase = 0;
    float scale_factor = 1.0f / sqrtf(128.0f);
    
    for (int step = 0; step <= num_steps; ++step) {
        int k_base = step * 64;
        int next_k_base = (step + 1) * 64;
        
        if (step + 1 <= num_steps) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&mbar[1], 32768);
                tma_load_4d_fn(&tma_K, &mbar[1], s_K0, 0, next_k_base, head_idx, batch_idx);
                tma_load_4d_fn(&tma_K, &mbar[1], s_K1, 64, next_k_base, head_idx, batch_idx);
                tma_load_4d_fn(&tma_V, &mbar[1], s_V0, 0, next_k_base, head_idx, batch_idx);
                tma_load_4d_fn(&tma_V, &mbar[1], s_V1, 64, next_k_base, head_idx, batch_idx);
            }
        }
        
        int wait_parity = next_barrier_phase & 1;
        mbarrier_wait_fn(&mbar[1], wait_parity);
        
        float S[16][2] = {0};
        for (int k = 0; k < 64; k += 16) {
            uint64_t desc_a = make_smem_desc(s_Q0 + k * stride_q, 0, 1024);
            uint64_t desc_b = make_smem_desc(s_K0 + k * stride_q, 0, 1024);
            uint32_t idesc = make_instr_desc_fn(64, 64);
            umma_f16_cg1_fn((uint32_t*)S, desc_a, desc_b, idesc, k == 0 ? 0 : 1);
        }
        for (int k = 0; k < 64; k += 16) {
            uint64_t desc_a = make_smem_desc(s_Q1 + k * stride_q, 0, 1024);
            uint64_t desc_b = make_smem_desc(s_K1 + k * stride_q, 0, 1024);
            uint32_t idesc = make_instr_desc_fn(64, 64);
            umma_f16_cg1_fn((uint32_t*)S, desc_a, desc_b, idesc, 1);
        }
        
        int warp_id = threadIdx.x / 32;
        int lane_id = threadIdx.x % 32;
        
        for (int i = 0; i < 16; ++i) {
            for (int j = 0; j < 2; ++j) {
                int row = warp_id * 16 + i;
                int col = lane_id * 2 + j;
                int global_q_idx = q_base + row;
                int global_k_idx = k_base + col;
                if (global_q_idx < global_k_idx || global_k_idx >= S) {
                    S[i][j] = -INFINITY;
                } else {
                    S[i][j] *= scale_factor;
                }
            }
        }
        
        for (int i = 0; i < 16; ++i) {
            float row_max = fmaxf(S[i][0], S[i][1]);
            for (int offset = 1; offset < 32; offset *= 2) {
                row_max = fmaxf(row_max, __shfl_xor_sync(0xFFFFFFFF, row_max, offset));
            }
            
            if (row_max > lse_max) {
                lse_max = row_max;
            }
        }
        
        float scale = 1.0f;
        if (prev_max < lse_max) {
            scale = fast_exp2f_fn((prev_max - lse_max) * 1.4426950f);
            lse_sum *= scale;
        }
        
        for (int i = 0; i < 16; ++i) {
            for (int j = 0; j < 2; ++j) {
                O_scaled[i][j] *= scale;
            }
        }
        
        for (int i = 0; i < 16; ++i) {
            float row_sum = 0;
            for (int j = 0; j < 2; ++j) {
                float p = fast_exp2f_fn((S[i][j] - lse_max) * 1.4426950f);
                row_sum += p;
                S[i][j] = p;
            }
            for (int offset = 1; offset < 32; offset *= 2) {
                row_sum += __shfl_xor_sync(0xFFFFFFFF, row_sum, offset);
            }
            lse_sum += row_sum;
        }
        
        for (int i = 0; i < 16; ++i) {
            for (int j = 0; j < 2; j+=2) {
                uint32_t packed = pack_bf16_fn(__float_as_uint(S[i][j]), __float_as_uint(S[i][j+1]));
                int r = warp_id * 16 + i;
                int c = lane_id * 2;
                int sc = swizzle_128B(r, c);
                *(uint32_t*)&s_P[r * 64 + sc] = packed;
            }
        }
        
        __syncthreads();
        fence_proxy_async_fn();
        
        uint64_t desc_P[2] = {make_smem_desc(s_P, 0, 1024), make_smem_desc(s_P, 0, 1024)};
        uint64_t desc_V0[4], desc_V1[4];
        for (int k = 0; k < 64; k += 16) {
            desc_V0[k/16] = make_smem_desc(s_V0 + k * stride_v, 0, 1024);
            desc_V1[k/16] = make_smem_desc(s_V1 + k * stride_v, 0, 1024);
        }
        uint32_t idesc_PV[4] = {make_instr_desc_b_mn_major_fn(64, 64), make_instr_desc_b_mn_major_fn(64, 64), make_instr_desc_b_mn_major_fn(64, 64), make_instr_desc_b_mn_major_fn(64, 64)};
        
        for (int k = 0; k < 64; k += 16) {
            umma_f16_cg1_fn((uint32_t*)O_scaled, desc_P[k/16], desc_V0[k/16], idesc_PV[k/16], k == 0 ? 0 : 1);
        }
        for (int k = 0; k < 64; k += 16) {
            umma_f16_cg1_fn((uint32_t*)O_scaled, desc_P[k/16], desc_V1[k/16], idesc_PV[k/16], k == 0 ? 0 : 1);
        }
        
        prev_max = lse_max;
        next_barrier_phase++;
    }
    
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 2; ++j) {
            float out_val = O_scaled[i][j] / lse_sum;
            int row = warp_id * 16 + i;
            int col = lane_id * 2 + j;
            int global_q_idx = q_base + row;
            int global_d_idx = col; 
            if (global_q_idx < S && global_d_idx < 128) {
                O_ptr[global_q_base * 128 + global_d_idx] = __float2bfloat16(out_val);
            }
            
            int global_d_idx_1 = col + 64; 
            if (global_q_idx < S && global_d_idx_1 < 128) {
                O_ptr[global_q_base * 128 + global_d_idx_1] = __float2bfloat16(out_val);
            }
        }
    }
    
    if (threadIdx.x < 64) {
        int row = threadIdx.x;
        int global_q_idx = q_base + row;
        if (global_q_idx < S) {
            LSE_ptr[global_q_base + global_q_idx] = lse_sum + lse_max;
        }
    }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);