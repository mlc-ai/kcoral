#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <math.h>
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

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 4
#define THREADS_PER_BLOCK 128
#define TILE_K 16
#define HEAD_DIM 128
#define ELEMS_PER_LANE 4

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
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
    int S)
{
    int bh = blockIdx.x;
    int q_start = blockIdx.y * WARPS_PER_BLOCK;

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int qi = q_start + warp_id;
    int max_qi = min(q_start + WARPS_PER_BLOCK - 1, S - 1);
    bool valid = (qi < S);

    __shared__ __nv_bfloat16 smem_K[TILE_K * HEAD_DIM];
    __shared__ __nv_bfloat16 smem_V[TILE_K * HEAD_DIM];

    float q[ELEMS_PER_LANE] = {0.0f};
    float do_[ELEMS_PER_LANE] = {0.0f};
    float D_val = 0.0f;
    float lse = 0.0f;

    // Load Q, dO, O for this query row and compute D_i = dO_i . O_i
    if (valid) {
        int base = bh * S * HEAD_DIM + qi * HEAD_DIM;
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_LANE; e++) {
            int idx = lane_id * ELEMS_PER_LANE + e;
            q[e] = __bfloat162float(Q[base + idx]);
            do_[e] = __bfloat162float(dO[base + idx]);
            float o_val = __bfloat162float(O[base + idx]);
            D_val += do_[e] * o_val;
        }
        lse = L[bh * S + qi];
        D_val = warp_reduce_sum(D_val);
    }

    float scale = rsqrtf((float)HEAD_DIM);
    float dq[ELEMS_PER_LANE] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int kt = 0; kt <= max_qi; kt += TILE_K) {
        int tile_rows = min(TILE_K, max_qi - kt + 1);

        // Cooperative load of K and V tile (128 threads, 128 elements per row)
        for (int row = 0; row < tile_rows; row++) {
            int kj = kt + row;
            smem_K[row * HEAD_DIM + threadIdx.x] = K[bh * S * HEAD_DIM + kj * HEAD_DIM + threadIdx.x];
            smem_V[row * HEAD_DIM + threadIdx.x] = V[bh * S * HEAD_DIM + kj * HEAD_DIM + threadIdx.x];
        }
        __syncthreads();

        if (valid) {
            int loop_end = min(kt + tile_rows, qi + 1);
            for (int kj = kt; kj < loop_end; kj++) {
                int lr = kj - kt;
                float score = 0.0f, dot_dov = 0.0f;
                float k_vals[ELEMS_PER_LANE], v_vals[ELEMS_PER_LANE];
                #pragma unroll
                for (int e = 0; e < ELEMS_PER_LANE; e++) {
                    k_vals[e] = __bfloat162float(smem_K[lr * HEAD_DIM + lane_id * ELEMS_PER_LANE + e]);
                    v_vals[e] = __bfloat162float(smem_V[lr * HEAD_DIM + lane_id * ELEMS_PER_LANE + e]);
                    score += q[e] * k_vals[e];
                    dot_dov += do_[e] * v_vals[e];
                }
                score = warp_reduce_sum(score);
                dot_dov = warp_reduce_sum(dot_dov);

                float P = expf(score * scale - lse);
                float dS = P * (dot_dov - D_val);

                // dQ_i += dS * K_j * scale
                #pragma unroll
                for (int e = 0; e < ELEMS_PER_LANE; e++) {
                    dq[e] += dS * k_vals[e] * scale;
                }

                // dK_j += dS * Q_i * scale (atomic float32)
                int dk_base = bh * S * HEAD_DIM + kj * HEAD_DIM + lane_id * ELEMS_PER_LANE;
                #pragma unroll
                for (int e = 0; e < ELEMS_PER_LANE; e++) {
                    atomicAdd(&dK_float[dk_base + e], dS * q[e] * scale);
                }

                // dV_j += P * dO_i (atomic float32)
                int dv_base = bh * S * HEAD_DIM + kj * HEAD_DIM + lane_id * ELEMS_PER_LANE;
                #pragma unroll
                for (int e = 0; e < ELEMS_PER_LANE; e++) {
                    atomicAdd(&dV_float[dv_base + e], P * do_[e]);
                }
            }
        }
        __syncthreads();
    }

    if (valid) {
        int dq_base = bh * S * HEAD_DIM + qi * HEAD_DIM;
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_LANE; e++) {
            dQ[dq_base + lane_id * ELEMS_PER_LANE + e] = __float2bfloat16(dq[e]);
        }
    }
}

__global__ void convert_f32_to_bf16_kernel(const float* src, __nv_bfloat16* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2bfloat16(src[idx]);
    }
}

namespace mha_bwd_d128_causal {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {

    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = 4, H = 48, D = 128;
    int S = static_cast<int>(Q.size(2));
    int total = B * H * S * D;

    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_ptr = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_ptr = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    float* dK_float = nullptr;
    float* dV_float = nullptr;
    CUDA_CHECK(cudaMalloc(&dK_float, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dV_float, total * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(dK_float, 0, total * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(dV_float, 0, total * sizeof(float), stream));

    dim3 grid(B * H, (S + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    dim3 block(THREADS_PER_BLOCK);

    mha_bwd_kernel<<<grid, block, 0, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
        dQ_ptr, dK_float, dV_float, S);

    CUDA_CHECK(cudaGetLastError());

    int convert_threads = 256;
    int convert_blocks = (total + convert_threads - 1) / convert_threads;
    convert_f32_to_bf16_kernel<<<convert_blocks, convert_threads, 0, stream>>>(dK_float, dK_ptr, total);
    convert_f32_to_bf16_kernel<<<convert_blocks, convert_threads, 0, stream>>>(dV_float, dV_ptr, total);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFree(dK_float));
    CUDA_CHECK(cudaFree(dV_float));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128_causal::run);

}  // namespace mha_bwd_d128_causal