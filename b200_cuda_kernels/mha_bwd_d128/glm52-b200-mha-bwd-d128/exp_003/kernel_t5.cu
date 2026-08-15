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
constexpr int Bk = 64;
constexpr int D = 128;
constexpr int THREADS = 256;
constexpr int B = 4;
constexpr int H = 48;

constexpr int ELEMS_P_DS = (Bq * Bk) / THREADS;  // 4
constexpr int ELEMS_DK_DV = (Bk * D) / THREADS;  // 32
constexpr int ELEMS_DQ = (Bq * D) / THREADS;     // 8

constexpr int SMEM_SIZE =
    Bk * D * 2 +   // K_smem
    Bk * D * 2 +   // V_smem
    Bq * D * 2 +   // Q_smem
    Bq * D * 2 +   // dO_smem
    Bq * Bk * 4 +  // P_smem
    Bq * Bk * 4 +  // dS_smem
    Bq * 4 +       // L_smem
    Bq * 4;        // Di_smem

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
    const float* L_bh  = L  + (size_t)(b * H + h) * S;
    const float* D_bh  = D_in + (size_t)(b * H + h) * S;
    float* dQ_bh = dQ_fp32 + bh_offset;
    float* dK_bh = dK_fp32 + bh_offset;
    float* dV_bh = dV_fp32 + bh_offset;

    extern __shared__ char smem[];
    __nv_bfloat16* K_smem  = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* V_smem  = K_smem + Bk * D;
    __nv_bfloat16* Q_smem  = V_smem + Bk * D;
    __nv_bfloat16* dO_smem = Q_smem + Bq * D;
    float* P_smem  = reinterpret_cast<float*>(dO_smem + Bq * D);
    float* dS_smem = P_smem + Bq * Bk;
    float* L_smem  = dS_smem + Bq * Bk;
    float* Di_smem = L_smem + Bq;

    // Load K, V tiles: Bk*D = 8192 bf16 = 1024 uint4, 256 threads -> 4 each
    #pragma unroll
    for (int iter = 0; iter < 4; iter++) {
        int idx = tid + iter * 256;
        int row = idx / 16;
        int col = (idx % 16) * 8;
        int k = k_start + row;
        if (k < S) {
            *reinterpret_cast<uint4*>(&K_smem[row * D + col]) =
                *reinterpret_cast<const uint4*>(&K_bh[k * D + col]);
            *reinterpret_cast<uint4*>(&V_smem[row * D + col]) =
                *reinterpret_cast<const uint4*>(&V_bh[k * D + col]);
        } else {
            *reinterpret_cast<uint4*>(&K_smem[row * D + col]) = make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(&V_smem[row * D + col]) = make_uint4(0, 0, 0, 0);
        }
    }
    __syncthreads();

    // Register accumulation for dK, dV
    float dK_reg[ELEMS_DK_DV];
    float dV_reg[ELEMS_DK_DV];
    #pragma unroll
    for (int i = 0; i < ELEMS_DK_DV; i++) {
        dK_reg[i] = 0.0f;
        dV_reg[i] = 0.0f;
    }

    // Loop over Q blocks
    for (int q_start = 0; q_start < S; q_start += Bq) {
        // Load Q, dO tiles: Bq*D = 2048 bf16 = 256 uint4, 256 threads -> 1 each
        {
            int row = tid / 16;
            int col = (tid % 16) * 8;
            int q = q_start + row;
            if (q < S) {
                *reinterpret_cast<uint4*>(&Q_smem[row * D + col]) =
                    *reinterpret_cast<const uint4*>(&Q_bh[q * D + col]);
                *reinterpret_cast<uint4*>(&dO_smem[row * D + col]) =
                    *reinterpret_cast<const uint4*>(&dO_bh[q * D + col]);
            } else {
                *reinterpret_cast<uint4*>(&Q_smem[row * D + col]) = make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4*>(&dO_smem[row * D + col]) = make_uint4(0, 0, 0, 0);
            }
        }

        // Load L, Di
        if (tid < Bq) {
            int q = q_start + tid;
            L_smem[tid] = (q < S) ? L_bh[q] : 0.0f;
            Di_smem[tid] = (q < S) ? D_bh[q] : 0.0f;
        }
        __syncthreads();

        // Compute P[i,j] and dS[i,j]: Bq*Bk=1024, 256 threads -> 4 each
        #pragma unroll
        for (int iter = 0; iter < ELEMS_P_DS; iter++) {
            int elem = tid + iter * THREADS;
            int i = elem / Bk;
            int j = elem % Bk;
            float s_val = 0.0f, dp_val = 0.0f;
            #pragma unroll
            for (int d = 0; d < D; d += 2) {
                __nv_bfloat162 q2 = *reinterpret_cast<__nv_bfloat162*>(&Q_smem[i * D + d]);
                __nv_bfloat162 k2 = *reinterpret_cast<__nv_bfloat162*>(&K_smem[j * D + d]);
                float2 qf = __bfloat1622float2(q2);
                float2 kf = __bfloat1622float2(k2);
                s_val += qf.x * kf.x + qf.y * kf.y;

                __nv_bfloat162 do2 = *reinterpret_cast<__nv_bfloat162*>(&dO_smem[i * D + d]);
                __nv_bfloat162 v2 = *reinterpret_cast<__nv_bfloat162*>(&V_smem[j * D + d]);
                float2 dof = __bfloat1622float2(do2);
                float2 vf = __bfloat1622float2(v2);
                dp_val += dof.x * vf.x + dof.y * vf.y;
            }
            s_val *= scale;
            float p_val = __expf(s_val - L_smem[i]);
            P_smem[i * Bk + j] = p_val;
            dS_smem[i * Bk + j] = p_val * (dp_val - Di_smem[i]);
        }
        __syncthreads();

        // Accumulate dK, dV in registers: Bk*D=8192, 256 threads -> 32 each
        #pragma unroll 1
        for (int iter = 0; iter < ELEMS_DK_DV; iter++) {
            int elem = tid + iter * THREADS;
            int j = elem / D;
            int d = elem % D;
            float dk = 0.0f, dv = 0.0f;
            #pragma unroll
            for (int i = 0; i < Bq; i++) {
                dk += dS_smem[i * Bk + j] * __bfloat162float(Q_smem[i * D + d]);
                dv += P_smem[i * Bk + j] * __bfloat162float(dO_smem[i * D + d]);
            }
            dK_reg[iter] += dk * scale;
            dV_reg[iter] += dv;
        }

        // dQ atomic add: Bq*D=2048, 256 threads -> 8 each
        #pragma unroll 1
        for (int iter = 0; iter < ELEMS_DQ; iter++) {
            int elem = tid + iter * THREADS;
            int i = elem / D;
            int d = elem % D;
            float dq = 0.0f;
            #pragma unroll
            for (int j = 0; j < Bk; j++)
                dq += dS_smem[i * Bk + j] * __bfloat162float(K_smem[j * D + d]);
            dq *= scale;
            int q = q_start + i;
            if (q < S)
                atomicAdd(&dQ_bh[q * D + d], dq);
        }
        __syncthreads();
    }

    // Store dK, dV from registers to global (direct store, unique K/V rows)
    #pragma unroll 1
    for (int iter = 0; iter < ELEMS_DK_DV; iter++) {
        int elem = tid + iter * THREADS;
        int j = elem / D;
        int d = elem % D;
        int k = k_start + j;
        if (k < S) {
            dK_bh[k * D + d] = dK_reg[iter];
            dV_bh[k * D + d] = dV_reg[iter];
        }
    }
}

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

    float* work_buf;
    size_t work_size = (size_t)(total_lse + 3 * total_elements) * sizeof(float);
    CUDA_CHECK(cudaMalloc(&work_buf, work_size));
    float* D_buf   = work_buf;
    float* dQ_fp32 = D_buf + total_lse;
    float* dK_fp32 = dQ_fp32 + total_elements;
    float* dV_fp32 = dK_fp32 + total_elements;

    int z_threads = 256;
    int z_blocks = (total_elements + z_threads - 1) / z_threads;
    zero_init_kernel<<<z_blocks, z_threads, 0, stream>>>(dQ_fp32, total_elements);

    int d_blocks = (total_lse + z_threads - 1) / z_threads;
    compute_D_kernel<<<d_blocks, z_threads, 0, stream>>>(O_ptr, dO_ptr, D_buf, S);

    int k_blocks = (S + Bk - 1) / Bk;
    dim3 grid2(total_bh, k_blocks);
    dim3 block2(THREADS);

    CUDA_CHECK(cudaFuncSetAttribute(attention_backward_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));

    attention_backward_kernel<<<grid2, block2, SMEM_SIZE, stream>>>(
        Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_buf,
        dQ_fp32, dK_fp32, dV_fp32, S);

    convert_to_bf16_kernel<<<z_blocks, z_threads, 0, stream>>>(
        dQ_fp32, dK_fp32, dV_fp32, dQ_ptr, dK_ptr, dV_ptr, total_elements);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaFree(work_buf);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128::run);

}  // namespace mha_bwd_d128