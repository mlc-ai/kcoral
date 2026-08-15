#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
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
} while (0)

namespace mha_bwd_d128 {

constexpr int Bq = 16;
constexpr int Bk = 16;
constexpr int D = 128;
constexpr int THREADS = 256;
constexpr int B = 4;
constexpr int H = 48;

// Kernel 1: Compute D_i = sum_d O[i,d] * dO[i,d] = dO_i · O_i
__global__ void compute_D_kernel(
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    float* __restrict__ D_out,
    int S) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * S;
    if (idx >= total) return;
    int s = idx % S;
    int h = (idx / S) % H;
    int b = idx / (S * H);

    const __nv_bfloat16* O_ptr  = O  + ((size_t)(b * H + h) * S + s) * D;
    const __nv_bfloat16* dO_ptr = dO + ((size_t)(b * H + h) * S + s) * D;

    float d_val = 0.0f;
    #pragma unroll
    for (int d = 0; d < D; d++) {
        d_val += __bfloat162float(O_ptr[d]) * __bfloat162float(dO_ptr[d]);
    }
    D_out[idx] = d_val;
}

// Kernel 2: Main attention backward — computes dQ (local), dK & dV (atomic float32)
__global__ void attention_backward_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ D_in,
    float* __restrict__ dQ_fp32,
    float* __restrict__ dK_fp32,
    float* __restrict__ dV_fp32,
    int S) {

    int bh = blockIdx.x;
    int q_block = blockIdx.y;
    int b = bh / H;
    int h = bh % H;
    int q_start = q_block * Bq;
    int tid = threadIdx.x;

    const float scale = rsqrtf((float)D);

    size_t bh_offset = (size_t)(b * H + h) * S * D;
    const __nv_bfloat16* Q_bh  = Q  + bh_offset;
    const __nv_bfloat16* K_bh  = K  + bh_offset;
    const __nv_bfloat16* V_bh  = V  + bh_offset;
    const __nv_bfloat16* dO_bh = dO + bh_offset;
    const float* L_bh  = L  + (size_t)(b * H + h) * S;
    const float* D_bh  = D_in + (size_t)(b * H + h) * S;

    float* dQ_bh = dQ_fp32 + bh_offset;
    float* dK_bh = dK_fp32 + bh_offset;
    float* dV_bh = dV_fp32 + bh_offset;

    __shared__ __nv_bfloat16 Q_smem[Bq][D];
    __shared__ __nv_bfloat16 dO_smem[Bq][D];
    __shared__ __nv_bfloat16 K_smem[Bk][D];
    __shared__ __nv_bfloat16 V_smem[Bk][D];
    __shared__ float P_smem[Bq][Bk];
    __shared__ float dS_smem[Bq][Bk];
    __shared__ float dQ_smem[Bq][D];
    __shared__ float D_smem[Bq];

    // Load Q and dO tiles, init dQ_smem to 0, load D_smem
    {
        int row = tid / 16;
        int col_start = (tid % 16) * 8;
        #pragma unroll
        for (int c = 0; c < 8; c++) {
            int col = col_start + c;
            int q_idx = q_start + row;
            if (q_idx < S) {
                Q_smem[row][col]  = Q_bh[q_idx * D + col];
                dO_smem[row][col] = dO_bh[q_idx * D + col];
            } else {
                Q_smem[row][col]  = __float2bfloat16(0.0f);
                dO_smem[row][col] = __float2bfloat16(0.0f);
            }
            dQ_smem[row][col] = 0.0f;
        }
    }
    if (tid < Bq) {
        int q_idx = q_start + tid;
        D_smem[tid] = (q_idx < S) ? D_bh[q_idx] : 0.0f;
    }
    __syncthreads();

    // Iterate over key blocks
    for (int k_start = 0; k_start < S; k_start += Bk) {
        // Load K, V tiles
        {
            int row = tid / 16;
            int col_start = (tid % 16) * 8;
            #pragma unroll
            for (int c = 0; c < 8; c++) {
                int col = col_start + c;
                int k_idx = k_start + row;
                if (k_idx < S) {
                    K_smem[row][col] = K_bh[k_idx * D + col];
                    V_smem[row][col] = V_bh[k_idx * D + col];
                } else {
                    K_smem[row][col] = __float2bfloat16(0.0f);
                    V_smem[row][col] = __float2bfloat16(0.0f);
                }
            }
        }
        __syncthreads();

        // Compute S = Q @ K^T * scale, P = exp(S - L), dP = dO @ V^T, dS = P*(dP - D)
        // tid -> (i = tid / Bk, j = tid % Bk), one element per thread
        float p_val = 0.0f;
        float ds_val = 0.0f;
        {
            int i = tid / Bk;
            int j = tid % Bk;
            float s_val = 0.0f;
            float dp_val = 0.0f;
            #pragma unroll
            for (int d = 0; d < D; d++) {
                s_val  += __bfloat162float(Q_smem[i][d])  * __bfloat162float(K_smem[j][d]);
                dp_val += __bfloat162float(dO_smem[i][d]) * __bfloat162float(V_smem[j][d]);
            }
            s_val *= scale;
            int q_idx = q_start + i;
            float lse = (q_idx < S) ? L_bh[q_idx] : 0.0f;
            p_val = __expf(s_val - lse);
            ds_val = p_val * (dp_val - D_smem[i]);
            P_smem[i][j]  = p_val;
            dS_smem[i][j] = ds_val;
        }
        __syncthreads();

        // dV[j, d] += sum_i P[i,j] * dO[i,d]  (atomicAdd)
        // dQ[i, d] += sum_j dS[i,j] * K[j,d] * scale  (local)
        // dK[j, d] += sum_i dS[i,j] * Q[i,d] * scale  (atomicAdd)
        {
            int row = tid / 16;
            int col_start = (tid % 16) * 8;
            int k_idx = k_start + row;
            int q_idx = q_start + row;

            // dV
            if (k_idx < S) {
                #pragma unroll
                for (int c = 0; c < 8; c++) {
                    int col = col_start + c;
                    float dv = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < Bq; i++) {
                        dv += P_smem[i][row] * __bfloat162float(dO_smem[i][col]);
                    }
                    atomicAdd(&dV_bh[k_idx * D + col], dv);
                }
            }

            // dQ
            if (q_idx < S) {
                #pragma unroll
                for (int c = 0; c < 8; c++) {
                    int col = col_start + c;
                    float dq = 0.0f;
                    #pragma unroll
                    for (int j = 0; j < Bk; j++) {
                        dq += dS_smem[row][j] * __bfloat162float(K_smem[j][col]);
                    }
                    dQ_smem[row][col] += dq * scale;
                }
            }

            // dK
            if (k_idx < S) {
                #pragma unroll
                for (int c = 0; c < 8; c++) {
                    int col = col_start + c;
                    float dk = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < Bq; i++) {
                        dk += dS_smem[i][row] * __bfloat162float(Q_smem[i][col]);
                    }
                    atomicAdd(&dK_bh[k_idx * D + col], dk * scale);
                }
            }
        }
        __syncthreads();
    }

    // Write dQ to global float32 buffer
    {
        int row = tid / 16;
        int col_start = (tid % 16) * 8;
        int q_idx = q_start + row;
        if (q_idx < S) {
            #pragma unroll
            for (int c = 0; c < 8; c++) {
                int col = col_start + c;
                dQ_bh[q_idx * D + col] = dQ_smem[row][col];
            }
        }
    }
}

// Kernel: Zero-init a float32 buffer
__global__ void zero_init_kernel(float* ptr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) ptr[idx] = 0.0f;
}

// Kernel 3: Convert float32 gradients to bf16
__global__ void convert_to_bf16_kernel(
    const float* __restrict__ dQ_fp32,
    const float* __restrict__ dK_fp32,
    const float* __restrict__ dV_fp32,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    dQ[idx] = __float2bfloat16(dQ_fp32[idx]);
    dK[idx] = __float2bfloat16(dK_fp32[idx]);
    dV[idx] = __float2bfloat16(dV_fp32[idx]);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {

    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int S = static_cast<int>(Q.size(2));
    int total_bh = B * H;
    int total_elements = B * H * S * D;
    int total_lse = B * H * S;

    const __nv_bfloat16* Q_ptr  = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr  = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr  = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_ptr  = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_ptr          = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_ptr       = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr       = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr       = static_cast<__nv_bfloat16*>(dV.data_ptr());

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    // Allocate float32 working buffers
    float *D_buf, *dQ_fp32, *dK_fp32, *dV_fp32;
    CUDA_CHECK(cudaMalloc(&D_buf,  total_lse * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dQ_fp32, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dK_fp32, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dV_fp32, total_elements * sizeof(float)));

    // Zero-init dK and dV buffers (atomicAdd targets)
    int z_threads = 256;
    int z_blocks = (total_elements + z_threads - 1) / z_threads;
    zero_init_kernel<<<z_blocks, z_threads, 0, stream>>>(dK_fp32, total_elements);
    zero_init_kernel<<<z_blocks, z_threads, 0, stream>>>(dV_fp32, total_elements);

    // Kernel 1: Compute D_i = dO_i · O_i
    int d_blocks = (total_lse + z_threads - 1) / z_threads;
    compute_D_kernel<<<d_blocks, z_threads, 0, stream>>>(O_ptr, dO_ptr, D_buf, S);

    // Kernel 2: Main backward
    int q_blocks = (S + Bq - 1) / Bq;
    dim3 grid2(total_bh, q_blocks);
    dim3 block2(THREADS);
    attention_backward_kernel<<<grid2, block2, 0, stream>>>(
        Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_buf,
        dQ_fp32, dK_fp32, dV_fp32, S);

    // Kernel 3: Convert float32 -> bf16
    convert_to_bf16_kernel<<<z_blocks, z_threads, 0, stream>>>(
        dQ_fp32, dK_fp32, dV_fp32, dQ_ptr, dK_ptr, dV_ptr, total_elements);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaFree(D_buf);
    cudaFree(dQ_fp32);
    cudaFree(dK_fp32);
    cudaFree(dV_fp32);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128::run);

}  // namespace mha_bwd_d128