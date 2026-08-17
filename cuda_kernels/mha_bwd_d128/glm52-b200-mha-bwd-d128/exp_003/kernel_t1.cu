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

constexpr int Bq = 64;
constexpr int Bk = 64;
constexpr int D = 128;
constexpr int Ds = 136;  // D stride with padding to avoid bank conflicts
constexpr int THREADS = 256;
constexpr int B = 4;
constexpr int H = 48;

// Shared memory size for attention_backward_kernel
constexpr int SMEM_SIZE =
    Bk * Ds * 2 +  // K_smem (bf16, padded)
    Bk * Ds * 2 +  // V_smem (bf16, padded)
    Bq * Ds * 2 +  // Q_smem (bf16, padded)
    Bq * Ds * 2 +  // dO_smem (bf16, padded)
    Bk * D  * 4 +  // dV_smem (float)
    Bk * D  * 4 +  // dK_smem (float)
    Bq * Bk * 4 +  // P_smem (float)
    Bq * Bk * 4 +  // dS_smem (float)
    Bq * 4;        // D_smem (float)

// Compute D_i = sum_d O[i,d] * dO[i,d]
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
    for (int d = 0; d < D; d++)
        d_val += __bfloat162float(O_ptr[d]) * __bfloat162float(dO_ptr[d]);
    D_out[idx] = d_val;
}

__global__ void zero_init_kernel(float* ptr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) ptr[idx] = 0.0f;
}

// Main backward kernel: grid = (bh, k_block)
// Each block handles one Bk-sized chunk of K/V, loops over all Q blocks.
// dV and dK are local accumulations (no atomics), dQ uses atomicAdd.
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
    int k_block = blockIdx.y;
    int b = bh / H;
    int h = bh % H;
    int k_start = k_block * Bk;
    int tid = threadIdx.x;

    const float scale = rsqrtf((float)D);

    size_t bh_offset = (size_t)(b * H + h) * S * D;
    const __nv_bfloat16* Q_bh  = Q  + bh_offset;
    const __nv_bfloat16* K_bh  = K  + bh_offset;
    const __nv_bfloat16* V_bh  = V  + bh_offset;
    const __nv_bfloat16* dO_bh = dO + bh_offset;
    const float* L_bh = L + (size_t)(b * H + h) * S;
    const float* D_bh = D_in + (size_t)(b * H + h) * S;

    float* dQ_bh = dQ_fp32 + bh_offset;
    float* dK_bh = dK_fp32 + bh_offset;
    float* dV_bh = dV_fp32 + bh_offset;

    extern __shared__ char smem[];
    __nv_bfloat16* K_smem  = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* V_smem  = K_smem + Bk * Ds;
    __nv_bfloat16* Q_smem  = V_smem + Bk * Ds;
    __nv_bfloat16* dO_smem = Q_smem + Bq * Ds;
    float* dV_smem = reinterpret_cast<float*>(dO_smem + Bq * Ds);
    float* dK_smem = dV_smem + Bk * D;
    float* P_smem  = dK_smem + Bk * D;
    float* dS_smem = P_smem + Bq * Bk;
    float* D_smem  = dS_smem + Bq * Bk;

    // Load K and V tiles (Bk * D bf16, stored with stride Ds)
    // 8192 bf16 = 1024 uint4 loads, 256 threads → 4 per thread
    #pragma unroll
    for (int iter = 0; iter < 4; iter++) {
        int load_idx = tid + iter * 256;
        int row = load_idx / 16;  // 16 uint4 per row (128/8)
        int col = (load_idx % 16) * 8;
        int k = k_start + row;
        if (k < S) {
            *reinterpret_cast<uint4*>(&K_smem[row * Ds + col]) =
                *reinterpret_cast<const uint4*>(&K_bh[k * D + col]);
            *reinterpret_cast<uint4*>(&V_smem[row * Ds + col]) =
                *reinterpret_cast<const uint4*>(&V_bh[k * D + col]);
        } else {
            *reinterpret_cast<uint4*>(&K_smem[row * Ds + col]) = make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(&V_smem[row * Ds + col]) = make_uint4(0, 0, 0, 0);
        }
    }

    // Zero dV_smem and dK_smem (Bk * D floats each)
    // 8192 floats = 2048 float4, 256 threads → 8 per thread
    #pragma unroll
    for (int iter = 0; iter < 8; iter++) {
        int store_idx = tid + iter * 256;
        int row = store_idx / 32;  // D/4 = 32
        int col = (store_idx % 32) * 4;
        *reinterpret_cast<float4*>(&dV_smem[row * D + col]) = make_float4(0.f, 0.f, 0.f, 0.f);
        *reinterpret_cast<float4*>(&dK_smem[row * D + col]) = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    __syncthreads();

    // Loop over Q blocks
    for (int q_start = 0; q_start < S; q_start += Bq) {
        // Load Q and dO tiles (Bq * D bf16, stored with stride Ds)
        #pragma unroll
        for (int iter = 0; iter < 4; iter++) {
            int load_idx = tid + iter * 256;
            int row = load_idx / 16;
            int col = (load_idx % 16) * 8;
            int q = q_start + row;
            if (q < S) {
                *reinterpret_cast<uint4*>(&Q_smem[row * Ds + col]) =
                    *reinterpret_cast<const uint4*>(&Q_bh[q * D + col]);
                *reinterpret_cast<uint4*>(&dO_smem[row * Ds + col]) =
                    *reinterpret_cast<const uint4*>(&dO_bh[q * D + col]);
            } else {
                *reinterpret_cast<uint4*>(&Q_smem[row * Ds + col]) = make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4*>(&dO_smem[row * Ds + col]) = make_uint4(0, 0, 0, 0);
            }
        }

        // Load D values
        if (tid < Bq) {
            int q = q_start + tid;
            D_smem[tid] = (q < S) ? D_bh[q] : 0.0f;
        }
        __syncthreads();

        // Compute S[i,j], P[i,j], dP[i,j], dS[i,j]
        // Bq * Bk = 4096 elements, 256 threads → 16 per thread
        for (int iter = 0; iter < 16; iter++) {
            int elem = tid + iter * 256;
            int i = elem / Bk;
            int j = elem % Bk;

            float s_val = 0.0f, dp_val = 0.0f;
            #pragma unroll
            for (int d = 0; d < D; d++) {
                s_val  += __bfloat162float(Q_smem[i * Ds + d]) * __bfloat162float(K_smem[j * Ds + d]);
                dp_val += __bfloat162float(dO_smem[i * Ds + d]) * __bfloat162float(V_smem[j * Ds + d]);
            }
            s_val *= scale;
            int q = q_start + i;
            float lse = (q < S) ? L_bh[q] : 0.0f;
            float p_val = __expf(s_val - lse);
            P_smem[i * Bk + j] = p_val;
            dS_smem[i * Bk + j] = p_val * (dp_val - D_smem[i]);
        }
        __syncthreads();

        // Accumulate dV and dK together (Bk * D = 8192, 256 threads → 32 per thread)
        for (int iter = 0; iter < 32; iter++) {
            int elem = tid + iter * 256;
            int j = elem / D;
            int d = elem % D;
            float dv = 0.0f, dk = 0.0f;
            for (int i = 0; i < Bq; i++) {
                dv += P_smem[i * Bk + j] * __bfloat162float(dO_smem[i * Ds + d]);
                dk += dS_smem[i * Bk + j] * __bfloat162float(Q_smem[i * Ds + d]);
            }
            dV_smem[j * D + d] += dv;
            dK_smem[j * D + d] += dk * scale;
        }
        __syncthreads();

        // dQ atomicAdd (Bq * D = 8192, 256 threads → 32 per thread)
        for (int iter = 0; iter < 32; iter++) {
            int elem = tid + iter * 256;
            int i = elem / D;
            int d = elem % D;
            float dq = 0.0f;
            for (int j = 0; j < Bk; j++)
                dq += dS_smem[i * Bk + j] * __bfloat162float(K_smem[j * Ds + d]);
            dq *= scale;
            int q = q_start + i;
            if (q < S)
                atomicAdd(&dQ_bh[q * D + d], dq);
        }
        __syncthreads();
    }

    // Store dV and dK to global (direct store, unique K/V rows per block)
    #pragma unroll
    for (int iter = 0; iter < 8; iter++) {
        int store_idx = tid + iter * 256;
        int j = store_idx / 32;
        int d = (store_idx % 32) * 4;
        int k = k_start + j;
        if (k < S) {
            *reinterpret_cast<float4*>(&dV_bh[k * D + d]) =
                *reinterpret_cast<float4*>(&dV_smem[j * D + d]);
            *reinterpret_cast<float4*>(&dK_bh[k * D + d]) =
                *reinterpret_cast<float4*>(&dK_smem[j * D + d]);
        }
    }
}

// Convert float32 gradients to bf16
__global__ void convert_to_bf16_kernel(
    const float* __restrict__ dQ_fp32,
    const float* __restrict__ dK_fp32,
    const float* __restrict__ dV_fp32,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
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

    // Zero dQ_fp32 (atomicAdd target); dK/dV are direct-stored
    int z_threads = 256;
    int z_blocks = (total_elements + z_threads - 1) / z_threads;
    zero_init_kernel<<<z_blocks, z_threads, 0, stream>>>(dQ_fp32, total_elements);

    // Compute D_i = dO_i · O_i
    int d_blocks = (total_lse + z_threads - 1) / z_threads;
    compute_D_kernel<<<d_blocks, z_threads, 0, stream>>>(O_ptr, dO_ptr, D_buf, S);

    // Main backward kernel
    int k_blocks = (S + Bk - 1) / Bk;
    dim3 grid2(total_bh, k_blocks);
    dim3 block2(THREADS);

    CUDA_CHECK(cudaFuncSetAttribute(attention_backward_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));

    attention_backward_kernel<<<grid2, block2, SMEM_SIZE, stream>>>(
        Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_buf,
        dQ_fp32, dK_fp32, dV_fp32, S);

    // Convert float32 -> bf16
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