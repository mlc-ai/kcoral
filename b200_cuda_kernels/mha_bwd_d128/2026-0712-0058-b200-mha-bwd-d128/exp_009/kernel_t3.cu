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

namespace tvm_ffi_mha_bwd {

__device__ __forceinline__ __nv_bfloat16 atomicAdd_bf16(__nv_bfloat16* address, __nv_bfloat16 val) {
    uint32_t* addr32 = (uint32_t*)(((uintptr_t)address) & ~3);
    int offset = (((uintptr_t)address) & 3) / 2;
    uint32_t old = *addr32;
    uint32_t assumed;
    do {
        assumed = old;
        uint16_t old_bf16 = (offset == 0) ? (assumed & 0xFFFF) : (assumed >> 16);
        float f_val = __bfloat162float(val);
        float f_old = __bfloat162float(*reinterpret_cast<float*>(&old_bf16));
        float f_sum = f_old + f_val;
        __nv_bfloat16 sum_bf16 = __float2bfloat16(f_sum);
        uint16_t sum_u16 = *reinterpret_cast<uint16_t*>(&sum_bf16);
        uint32_t new_val = (offset == 0) ? (sum_u16 | (assumed & 0xFFFF0000)) : ((sum_u16 << 16) | (assumed & 0x0000FFFF));
        old = atomicCAS(addr32, assumed, new_val);
    } while (assumed != old);
    return val;
}

__device__ void load_tile_128(const __nv_bfloat16* gmem, __nv_bfloat16* smem, int base_row, int max_rows, int stride) {
    for (int i = threadIdx.x; i < 2048; i += blockDim.x) {
        int row = i / 16;
        int col = (i % 16) * 8;
        if (base_row + row < max_rows) {
            *(float4*)(smem + row * 128 + col) = *(const float4*)(gmem + (base_row + row) * stride + col);
        } else {
            *(float4*)(smem + row * 128 + col) = make_float4(0, 0, 0, 0);
        }
    }
}

__device__ void gemm_64_NT(const __nv_bfloat16* A, const __nv_bfloat16* B, float* C_regs, int row, int col_start) {
    for (int c = 0; c < 64; ++c) {
        int col = col_start + c;
        float sum = 0;
        for (int k = 0; k < 128; k += 4) {
            uint32_t a0 = *(const uint32_t*)(A + row * 128 + k);
            uint32_t a1 = *(const uint32_t*)(A + row * 128 + k + 2);
            uint32_t b0 = *(const uint32_t*)(B + col * 128 + k);
            uint32_t b1 = *(const uint32_t*)(B + col * 128 + k + 2);
            __nv_bfloat16* ap = (__nv_bfloat16*)&a0;
            __nv_bfloat16* bp = (__nv_bfloat16*)&b0;
            sum += __bfloat162float(ap[0]) * __bfloat162float(bp[0]);
            sum += __bfloat162float(ap[1]) * __bfloat162float(bp[1]);
            ap = (__nv_bfloat16*)&a1;
            bp = (__nv_bfloat16*)&b1;
            sum += __bfloat162float(ap[0]) * __bfloat162float(bp[0]);
            sum += __bfloat162float(ap[1]) * __bfloat162float(bp[1]);
        }
        C_regs[c] = sum;
    }
}

__device__ void gemm_64_TN(const __nv_bfloat16* A, const __nv_bfloat16* B, float* C_regs, int row, int col_start) {
    for (int c = 0; c < 64; ++c) {
        int col = col_start + c;
        float sum = 0;
        for (int k = 0; k < 128; k += 4) {
            uint32_t a0 = *(const uint32_t*)(A + k * 128 + row);
            uint32_t a1 = *(const uint32_t*)(A + k * 128 + row + 2);
            uint32_t b0 = *(const uint32_t*)(B + k * 128 + col);
            uint32_t b1 = *(const uint32_t*)(B + k * 128 + col + 2);
            __nv_bfloat16* ap = (__nv_bfloat16*)&a0;
            __nv_bfloat16* bp = (__nv_bfloat16*)&b0;
            sum += __bfloat162float(ap[0]) * __bfloat162float(bp[0]);
            sum += __bfloat162float(ap[1]) * __bfloat162float(bp[1]);
            ap = (__nv_bfloat16*)&a1;
            bp = (__nv_bfloat16*)&b1;
            sum += __bfloat162float(ap[0]) * __bfloat162float(bp[0]);
            sum += __bfloat162float(ap[1]) * __bfloat162float(bp[1]);
        }
        C_regs[c] = sum;
    }
}

__device__ void gemm_64_NN(const __nv_bfloat16* A, const __nv_bfloat16* B, float* C_regs, int row, int col_start) {
    for (int c = 0; c < 64; ++c) {
        int col = col_start + c;
        float sum = 0;
        for (int k = 0; k < 128; k += 4) {
            uint32_t a0 = *(const uint32_t*)(A + row * 128 + k);
            uint32_t a1 = *(const uint32_t*)(A + row * 128 + k + 2);
            uint32_t b0 = *(const uint32_t*)(B + k * 128 + col);
            uint32_t b1 = *(const uint32_t*)(B + k * 128 + col + 2);
            __nv_bfloat16* ap = (__nv_bfloat16*)&a0;
            __nv_bfloat16* bp = (__nv_bfloat16*)&b0;
            sum += __bfloat162float(ap[0]) * __bfloat162float(bp[0]);
            sum += __bfloat162float(ap[1]) * __bfloat162float(bp[1]);
            ap = (__nv_bfloat16*)&a1;
            bp = (__nv_bfloat16*)&b1;
            sum += __bfloat162float(ap[0]) * __bfloat162float(bp[0]);
            sum += __bfloat162float(ap[1]) * __bfloat162float(bp[1]);
        }
        C_regs[c] = sum;
    }
}

__global__ void mha_bwd_kernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* O, const __nv_bfloat16* dO, const float* L,
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    int S_len)
{
    int b_h = blockIdx.y;
    int n_blk = blockIdx.x;
    int n_base = n_blk * 128;

    if (n_base >= S_len) return;

    int row = threadIdx.x % 128;
    int col_chunk = threadIdx.x / 128;
    int col_start = col_chunk * 64;

    float attn_scale = 1.0f / sqrtf(128.0f);

    extern __shared__ __align__(128) __nv_bfloat16 smem[];
    __nv_bfloat16* K_smem = smem;                 // 32 KB
    __nv_bfloat16* V_smem = smem + 16384;         // 32 KB
    __nv_bfloat16* Q_smem = smem + 32768;         // 32 KB
    __nv_bfloat16* O_smem = smem + 49152;         // 32 KB
    __nv_bfloat16* dO_smem = smem + 65536;        // 32 KB
    __nv_bfloat16* dS_smem = smem + 81920;        // 32 KB

    __shared__ float D_shared[128];
    __shared__ float LSE_shared[128];

    load_tile_128(K, K_smem, n_base, S_len, 128);
    load_tile_128(V, V_smem, n_base, S_len, 128);
    __syncthreads();

    float dV_regs[64] = {0};
    float dK_regs[64] = {0};

    for (int q_blk = 0; q_blk < (S_len + 127) / 128; ++q_blk) {
        int q_base = q_blk * 128;
        if (q_base >= S_len) break;

        load_tile_128(Q, Q_smem, q_base, S_len, 128);
        load_tile_128(O, O_smem, q_base, S_len, 128);
        load_tile_128(dO, dO_smem, q_base, S_len, 128);
        __syncthreads();

        if (col_chunk == 0) {
            int i = row;
            float d = 0;
            for (int j = 0; j < 128; j += 4) {
                uint32_t do0 = *(const uint32_t*)(dO_smem + i * 128 + j);
                uint32_t do1 = *(const uint32_t*)(dO_smem + i * 128 + j + 2);
                uint32_t o0  = *(const uint32_t*)(O_smem + i * 128 + j);
                uint32_t o1  = *(const uint32_t*)(O_smem + i * 128 + j + 2);
                __nv_bfloat16* dop = (__nv_bfloat16*)&do0;
                __nv_bfloat16* op  = (__nv_bfloat16*)&o0;
                d += __bfloat162float(dop[0]) * __bfloat162float(op[0]);
                d += __bfloat162float(dop[1]) * __bfloat162float(op[1]);
                dop = (__nv_bfloat16*)&do1;
                op  = (__nv_bfloat16*)&o1;
                d += __bfloat162float(dop[0]) * __bfloat162float(op[0]);
                d += __bfloat162float(dop[1]) * __bfloat162float(op[1]);
            }
            D_shared[i] = d;

            LSE_shared[i] = (q_base + i < S_len) ? L[b_h * S_len + q_base + i] : 0;
        }
        __syncthreads();

        float S_regs[64];
        gemm_64_NT(Q_smem, K_smem, S_regs, row, col_start);

        for (int c = 0; c < 64; ++c) {
            int col = col_start + c;
            float s = S_regs[c] * attn_scale - LSE_shared[row];
            if (q_base + row >= S_len) s = 0;
            O_smem[row * 128 + col] = __float2bfloat16(expf(s));
        }
        __syncthreads();

        float dP_T_regs[64];
        gemm_64_NT(V_smem, dO_smem, dP_T_regs, row, col_start);

        for (int c = 0; c < 64; ++c) {
            int col = col_start + c;
            Q_smem[row * 128 + col] = __float2bfloat16(dP_T_regs[c]);
        }
        __syncthreads();

        for (int c = 0; c < 64; ++c) {
            int col = col_start + c;
            uint32_t p_u32 = *(const uint32_t*)(O_smem + col * 128 + row);
            uint32_t dp_u32 = *(const uint32_t*)(Q_smem + row * 128 + col);
            __nv_bfloat16* p_ptr = (__nv_bfloat16*)&p_u32;
            __nv_bfloat16* dp_ptr = (__nv_bfloat16*)&dp_u32;
            float p0 = __bfloat162float(p_ptr[0]);
            float p1 = __bfloat162float(p_ptr[1]);
            float dp0 = __bfloat162float(dp_ptr[0]);
            float dp1 = __bfloat162float(dp_ptr[1]);
            float ds0 = p0 * (dp0 - D_shared[col]) * attn_scale;
            float ds1 = p1 * (dp1 - D_shared[col]) * attn_scale;
            __nv_bfloat16 ds_bf16[2] = {__float2bfloat16(ds0), __float2bfloat16(ds1)};
            *(uint32_t*)(dS_smem + row * 128 + col) = *(uint32_t*)ds_bf16;
        }
        __syncthreads();

        gemm_64_TN(O_smem, dO_smem, dV_regs, row, col_start);

        load_tile_128(Q, Q_smem, q_base, S_len, 128);
        __syncthreads();

        gemm_64_NN(dS_smem, Q_smem, dK_regs, row, col_start);

        float dQ_regs[64];
        gemm_64_TN(dS_smem, K_smem, dQ_regs, row, col_start);

        for (int c = 0; c < 64; ++c) {
            int col = col_start + c;
            if (q_base + row < S_len) {
                atomicAdd_bf16(&dQ[(b_h * S_len + q_base + row) * 128 + col], __float2bfloat16(dQ_regs[c]));
            }
        }
    }

    for (int c = 0; c < 64; ++c) {
        int col = col_start + c;
        if (n_base + row < S_len) {
            dV[(b_h * S_len + n_base + row) * 128 + col] = __float2bfloat16(dV_regs[c]);
            dK[(b_h * S_len + n_base + row) * 128 + col] = __float2bfloat16(dK_regs[c]);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = Q.size(3);

    if (d != 128) {
        fprintf(stderr, "Expected head dim 128, got %ld\n", d);
        exit(1);
    }

    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_data = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_data = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_data = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_data = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_data = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_data = static_cast<__nv_bfloat16*>(dV.data_ptr());

    CUDA_CHECK(cudaMemsetAsync(dQ_data, 0, B * H * S * 128 * sizeof(__nv_bfloat16)));

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    int num_tiles = (S + 127) / 128;
    dim3 grid(num_tiles, B * H);
    dim3 block(256);
    int smem_size = 196608;

    CUDA_CHECK(cudaFuncSetAttribute(tvm_ffi_mha_bwd::mha_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    mha_bwd_kernel<<<grid, block, smem_size, stream>>>(
        Q_data, K_data, V_data, O_data, dO_data, L_data,
        dQ_data, dK_data, dV_data, S
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_bwd::run);

} // namespace tvm_ffi_mha_bwd