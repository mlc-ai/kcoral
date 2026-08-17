#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

namespace mha_kernel_ns {

constexpr int D = 128;
constexpr int BM = 128;
constexpr int BN = 64;
constexpr int Q_STRIDE = D + 2;   // padding to avoid 32-way bank conflicts
constexpr int S_STRIDE = BN + 1;  // padding to avoid bank conflicts
constexpr float INV_SQRT_D = 0.0883883476483184f;

__global__ __launch_bounds__(128, 1)
void mha_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S) {

    int b = blockIdx.x;
    int h = blockIdx.y;
    int q_block = blockIdx.z;
    int q_start = q_block * BM;
    int tid = threadIdx.x;
    int row = q_start + tid;

    size_t bh_offset = ((size_t)b * H + h) * S * D;
    const __nv_bfloat16* Q_bh = Q + bh_offset;
    const __nv_bfloat16* K_bh = K + bh_offset;
    const __nv_bfloat16* V_bh = V + bh_offset;
    __nv_bfloat16* O_bh = O + bh_offset;
    float* LSE_bh = LSE + ((size_t)b * H + h) * S;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* smem_q = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* smem_k = smem_q + (size_t)BM * Q_STRIDE;
    __nv_bfloat16* smem_v = smem_k + (size_t)BN * D;
    float* smem_s = reinterpret_cast<float*>(smem_v + (size_t)BN * D);

    // Load Q tile: each thread loads its own row (128 bf16) via int (4B) loads
    if (row < S) {
        const __nv_bfloat16* q_src = Q_bh + (size_t)row * D;
        __nv_bfloat16* q_dst = smem_q + (size_t)tid * Q_STRIDE;
        #pragma unroll
        for (int d = 0; d < D; d += 2) {
            *reinterpret_cast<int*>(q_dst + d) = *reinterpret_cast<const int*>(q_src + d);
        }
    }
    __syncthreads();

    // O accumulator in registers
    float o[D];
    #pragma unroll
    for (int d = 0; d < D; d++) o[d] = 0.0f;

    float rowmax = -INFINITY;
    float rowsum = 0.0f;

    // Iterate over KV tiles
    for (int kv = 0; kv < S; kv += BN) {
        int kv_end = min(kv + BN, S);
        int actual_bn = kv_end - kv;

        // Load K tile (vectorized int4 = 8 bf16 per load)
        {
            const int4* k_src = reinterpret_cast<const int4*>(K_bh + (size_t)kv * D);
            int4* k_dst = reinterpret_cast<int4*>(smem_k);
            int total = BN * D / 8;
            for (int i = tid; i < total; i += blockDim.x) {
                int row_in_tile = i / (D / 8);
                if (kv + row_in_tile < S) {
                    k_dst[i] = k_src[i];
                } else {
                    k_dst[i] = make_int4(0, 0, 0, 0);
                }
            }
        }
        // Load V tile
        {
            const int4* v_src = reinterpret_cast<const int4*>(V_bh + (size_t)kv * D);
            int4* v_dst = reinterpret_cast<int4*>(smem_v);
            int total = BN * D / 8;
            for (int i = tid; i < total; i += blockDim.x) {
                int row_in_tile = i / (D / 8);
                if (kv + row_in_tile < S) {
                    v_dst[i] = v_src[i];
                } else {
                    v_dst[i] = make_int4(0, 0, 0, 0);
                }
            }
        }
        __syncthreads();

        // Compute S = Q @ K^T * scale
        if (row < S) {
            for (int j = 0; j < actual_bn; j++) {
                float dot = 0.0f;
                #pragma unroll
                for (int d = 0; d < D; d += 2) {
                    __nv_bfloat162 q2 = *reinterpret_cast<__nv_bfloat162*>(
                        smem_q + (size_t)tid * Q_STRIDE + d);
                    __nv_bfloat162 k2 = *reinterpret_cast<__nv_bfloat162*>(
                        smem_k + (size_t)j * D + d);
                    float2 qf = __bfloat1622float2(q2);
                    float2 kf = __bfloat1622float2(k2);
                    dot = fmaf(qf.x, kf.x, dot);
                    dot = fmaf(qf.y, kf.y, dot);
                }
                smem_s[tid * S_STRIDE + j] = dot * INV_SQRT_D;
            }
        }
        __syncthreads();

        // Online softmax: update rowmax, rescale rowsum and O
        if (row < S) {
            float old_max = rowmax;
            for (int j = 0; j < actual_bn; j++) {
                rowmax = fmaxf(rowmax, smem_s[tid * S_STRIDE + j]);
            }
            float rescale = expf(old_max - rowmax);
            rowsum *= rescale;
            #pragma unroll
            for (int d = 0; d < D; d++) o[d] *= rescale;

            // Compute P = exp(S - rowmax) and update rowsum
            for (int j = 0; j < actual_bn; j++) {
                float p = expf(smem_s[tid * S_STRIDE + j] - rowmax);
                smem_s[tid * S_STRIDE + j] = p;
                rowsum += p;
            }

            // O += P @ V (broadcast V row across all threads)
            for (int j = 0; j < actual_bn; j++) {
                float p = smem_s[tid * S_STRIDE + j];
                #pragma unroll
                for (int d = 0; d < D; d += 2) {
                    __nv_bfloat162 v2 = *reinterpret_cast<__nv_bfloat162*>(
                        smem_v + (size_t)j * D + d);
                    float2 vf = __bfloat1622float2(v2);
                    o[d]     = fmaf(p, vf.x, o[d]);
                    o[d + 1] = fmaf(p, vf.y, o[d + 1]);
                }
            }
        }
        __syncthreads();
    }

    // Store O and LSE
    if (row < S) {
        float inv_rowsum = 1.0f / rowsum;
        #pragma unroll
        for (int d = 0; d < D; d += 8) {
            __nv_bfloat16 tmp[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                tmp[i] = __float2bfloat16(o[d + i] * inv_rowsum);
            }
            *reinterpret_cast<int4*>(O_bh + (size_t)row * D + d) =
                *reinterpret_cast<int4*>(tmp);
        }
        LSE_bh[row] = rowmax + logf(rowsum);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = static_cast<int>(Q.size(0));
    int H = static_cast<int>(Q.size(1));
    int S = static_cast<int>(Q.size(2));

    if (S == 0) return;

    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());

    dim3 grid(B, H, (S + BM - 1) / BM);
    dim3 block(128);

    size_t smem_size = (size_t)BM * Q_STRIDE * sizeof(__nv_bfloat16)
                     + (size_t)BN * D * sizeof(__nv_bfloat16)
                     + (size_t)BN * D * sizeof(__nv_bfloat16)
                     + (size_t)BM * S_STRIDE * sizeof(float);

    CUDA_CHECK(cudaFuncSetAttribute(mha_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    mha_kernel<<<grid, block, smem_size, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data, B, H, S);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel_ns::run);

}  // namespace mha_kernel_ns