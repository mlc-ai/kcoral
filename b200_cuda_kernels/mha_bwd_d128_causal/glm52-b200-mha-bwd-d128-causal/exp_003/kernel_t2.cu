#include <cuda_bf16.h>
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

namespace tvm_ffi_mha_bwd {

#define BQ 8
#define BK 64
#define D 128

__device__ __forceinline__ float warp_dot(float4 a, float4 b) {
    float sum = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, 1);
    return sum;
}

__global__ void mha_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    float* __restrict__ dK_float,
    float* __restrict__ dV_float,
    int B, int H, int S, int d)
{
    const float scale = 1.0f / sqrtf((float)D);
    int batch_head = blockIdx.x;
    int b = batch_head / H;
    int h = batch_head % H;
    int q_block = blockIdx.y;
    int q_start = q_block * BQ;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int my_q = q_start + warp_id;

    if (q_start >= S) return;

    int max_q = min(q_start + BQ, S) - 1;
    int num_key_blocks = max_q / BK + 1;

    size_t head_offset = (size_t)(b * H + h) * S * D;
    const __nv_bfloat16* Q_h = Q + head_offset;
    const __nv_bfloat16* K_h = K + head_offset;
    const __nv_bfloat16* V_h = V + head_offset;
    const __nv_bfloat16* O_h = O + head_offset;
    const __nv_bfloat16* dO_h = dO + head_offset;
    const float* L_h = L + (size_t)(b * H + h) * S;
    __nv_bfloat16* dQ_h = dQ + head_offset;
    float* dK_h = dK_float + head_offset;
    float* dV_h = dV_float + head_offset;

    float4 q_reg = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 dO_reg = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 O_reg = make_float4(0.f, 0.f, 0.f, 0.f);
    float L_val = 0.f;
    float D_i = 0.f;
    if (my_q < S) {
        const __nv_bfloat16* q_ptr = Q_h + my_q * D + lane_id * 4;
        uint2 q_raw = *reinterpret_cast<const uint2*>(q_ptr);
        float2 q_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&q_raw.x));
        float2 q_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&q_raw.y));
        q_reg = make_float4(q_lo.x, q_lo.y, q_hi.x, q_hi.y);

        const __nv_bfloat16* do_ptr = dO_h + my_q * D + lane_id * 4;
        uint2 do_raw = *reinterpret_cast<const uint2*>(do_ptr);
        float2 do_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&do_raw.x));
        float2 do_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&do_raw.y));
        dO_reg = make_float4(do_lo.x, do_lo.y, do_hi.x, do_hi.y);

        const __nv_bfloat16* o_ptr = O_h + my_q * D + lane_id * 4;
        uint2 o_raw = *reinterpret_cast<const uint2*>(o_ptr);
        float2 o_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&o_raw.x));
        float2 o_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&o_raw.y));
        O_reg = make_float4(o_lo.x, o_lo.y, o_hi.x, o_hi.y);

        L_val = L_h[my_q];
        D_i = warp_dot(dO_reg, O_reg);
    }

    extern __shared__ char smem_buffer[];
    __nv_bfloat16* K_smem = reinterpret_cast<__nv_bfloat16*>(smem_buffer);
    __nv_bfloat16* V_smem = reinterpret_cast<__nv_bfloat16*>(K_smem + BK * D);
    float* grad_smem = reinterpret_cast<float*>(V_smem + BK * D);

    // Pass 1: dV
    for (int kb = 0; kb < num_key_blocks; ++kb) {
        int j_start = kb * BK;

        for (int idx = tid; idx < BK * D / 4; idx += blockDim.x) {
            int row = idx / (D / 4);
            int col4 = idx % (D / 4);
            int j = j_start + row;
            if (j < S) {
                uint2 val = *reinterpret_cast<const uint2*>(&K_h[j * D + col4 * 4]);
                *reinterpret_cast<uint2*>(&K_smem[row * D + col4 * 4]) = val;
            } else {
                *reinterpret_cast<uint2*>(&K_smem[row * D + col4 * 4]) = make_uint2(0, 0);
            }
        }
        for (int idx = tid; idx < BK * D; idx += blockDim.x) {
            grad_smem[idx] = 0.f;
        }
        __syncthreads();

        if (my_q < S) {
            for (int j = 0; j < BK; ++j) {
                int j_global = j_start + j;
                if (j_global > my_q || j_global >= S) continue;
                int k_idx = j * D + lane_id * 4;
                uint2 k_raw = *reinterpret_cast<uint2*>(&K_smem[k_idx]);
                float2 k_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&k_raw.x));
                float2 k_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&k_raw.y));
                float4 k_val = make_float4(k_lo.x, k_lo.y, k_hi.x, k_hi.y);
                float s = warp_dot(q_reg, k_val) * scale;
                float p = expf(s - L_val);
                atomicAdd(&grad_smem[j * D + lane_id * 4 + 0], p * dO_reg.x);
                atomicAdd(&grad_smem[j * D + lane_id * 4 + 1], p * dO_reg.y);
                atomicAdd(&grad_smem[j * D + lane_id * 4 + 2], p * dO_reg.z);
                atomicAdd(&grad_smem[j * D + lane_id * 4 + 3], p * dO_reg.w);
            }
        }
        __syncthreads();
        for (int idx = tid; idx < BK * D; idx += blockDim.x) {
            int row = idx / D;
            int j = j_start + row;
            if (j < S) {
                atomicAdd(&dV_h[j * D + (idx % D)], grad_smem[idx]);
            }
        }
        __syncthreads();
    }

    // Pass 2: dQ and dK
    float4 dQ_acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int kb = 0; kb < num_key_blocks; ++kb) {
        int j_start = kb * BK;

        for (int idx = tid; idx < BK * D / 4; idx += blockDim.x) {
            int row = idx / (D / 4);
            int col4 = idx % (D / 4);
            int j = j_start + row;
            if (j < S) {
                uint2 val = *reinterpret_cast<const uint2*>(&K_h[j * D + col4 * 4]);
                *reinterpret_cast<uint2*>(&K_smem[row * D + col4 * 4]) = val;
            } else {
                *reinterpret_cast<uint2*>(&K_smem[row * D + col4 * 4]) = make_uint2(0, 0);
            }
        }
        for (int idx = tid; idx < BK * D / 4; idx += blockDim.x) {
            int row = idx / (D / 4);
            int col4 = idx % (D / 4);
            int j = j_start + row;
            if (j < S) {
                uint2 val = *reinterpret_cast<const uint2*>(&V_h[j * D + col4 * 4]);
                *reinterpret_cast<uint2*>(&V_smem[row * D + col4 * 4]) = val;
            } else {
                *reinterpret_cast<uint2*>(&V_smem[row * D + col4 * 4]) = make_uint2(0, 0);
            }
        }
        for (int idx = tid; idx < BK * D; idx += blockDim.x) {
            grad_smem[idx] = 0.f;
        }
        __syncthreads();

        if (my_q < S) {
            for (int j = 0; j < BK; ++j) {
                int j_global = j_start + j;
                if (j_global > my_q || j_global >= S) continue;
                int k_idx = j * D + lane_id * 4;
                uint2 k_raw = *reinterpret_cast<uint2*>(&K_smem[k_idx]);
                float2 k_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&k_raw.x));
                float2 k_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&k_raw.y));
                float4 k_val = make_float4(k_lo.x, k_lo.y, k_hi.x, k_hi.y);
                float s = warp_dot(q_reg, k_val) * scale;
                float p = expf(s - L_val);
                int v_idx = j * D + lane_id * 4;
                uint2 v_raw = *reinterpret_cast<uint2*>(&V_smem[v_idx]);
                float2 v_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&v_raw.x));
                float2 v_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&v_raw.y));
                float4 v_val = make_float4(v_lo.x, v_lo.y, v_hi.x, v_hi.y);
                float dp = warp_dot(dO_reg, v_val);
                float dS = p * (dp - D_i);
                dQ_acc.x += dS * k_val.x * scale;
                dQ_acc.y += dS * k_val.y * scale;
                dQ_acc.z += dS * k_val.z * scale;
                dQ_acc.w += dS * k_val.w * scale;
                atomicAdd(&grad_smem[j * D + lane_id * 4 + 0], dS * q_reg.x * scale);
                atomicAdd(&grad_smem[j * D + lane_id * 4 + 1], dS * q_reg.y * scale);
                atomicAdd(&grad_smem[j * D + lane_id * 4 + 2], dS * q_reg.z * scale);
                atomicAdd(&grad_smem[j * D + lane_id * 4 + 3], dS * q_reg.w * scale);
            }
        }
        __syncthreads();
        for (int idx = tid; idx < BK * D; idx += blockDim.x) {
            int row = idx / D;
            int j = j_start + row;
            if (j < S) {
                atomicAdd(&dK_h[j * D + (idx % D)], grad_smem[idx]);
            }
        }
        __syncthreads();
    }

    if (my_q < S) {
        __nv_bfloat16* dQ_ptr = dQ_h + my_q * D + lane_id * 4;
        __nv_bfloat162 b01 = __floats2bfloat162_rn(dQ_acc.x, dQ_acc.y);
        __nv_bfloat162 b23 = __floats2bfloat162_rn(dQ_acc.z, dQ_acc.w);
        uint2 packed;
        packed.x = *reinterpret_cast<uint32_t*>(&b01);
        packed.y = *reinterpret_cast<uint32_t*>(&b23);
        *reinterpret_cast<uint2*>(dQ_ptr) = packed;
    }
}

__global__ void cast_to_bf16(const float* src, __nv_bfloat16* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2bfloat16(src[idx]);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int B = 4, H = 48, d = 128;
    int64_t S = Q.size(2);
    int64_t total = B * H * S * d;

    float* dK_float = nullptr;
    float* dV_float = nullptr;
    CUDA_CHECK(cudaMalloc(&dK_float, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dV_float, total * sizeof(float)));
    
    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaMemsetAsync(dK_float, 0, total * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(dV_float, 0, total * sizeof(float), stream));

    dim3 grid(B * H, (S + BQ - 1) / BQ);
    dim3 block(256);
    int smem_bytes = BK * D * sizeof(__nv_bfloat16) * 2 + BK * D * sizeof(float);
    
    cudaFuncSetAttribute((void*)mha_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    mha_bwd_kernel<<<grid, block, smem_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(O.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        dK_float,
        dV_float,
        B, H, (int)S, d
    );
    CUDA_CHECK(cudaGetLastError());

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    cast_to_bf16<<<blocks, threads, 0, stream>>>(dK_float, static_cast<__nv_bfloat16*>(dK.data_ptr()), (int)total);
    cast_to_bf16<<<blocks, threads, 0, stream>>>(dV_float, static_cast<__nv_bfloat16*>(dV.data_ptr()), (int)total);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(dK_float));
    CUDA_CHECK(cudaFree(dV_float));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_bwd::run);

}  // namespace tvm_ffi_mha_bwd