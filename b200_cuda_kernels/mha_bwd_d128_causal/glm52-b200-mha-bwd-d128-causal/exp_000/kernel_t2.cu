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
#define HEAD_DIM 128
#define ELEMS_PER_LANE 4
#define TILE_KV 16
#define BQ_TILE 64

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void compute_D_kernel(
    const __nv_bfloat16* __restrict__ dO,
    const __nv_bfloat16* __restrict__ O,
    float* __restrict__ D,
    int S)
{
    int bh = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int qi = blockIdx.y * WARPS_PER_BLOCK + warp_id;
    if (qi >= S) return;

    int base = bh * S * HEAD_DIM + qi * HEAD_DIM;
    float sum = 0.0f;
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_LANE; e++)
        sum += __bfloat162float(dO[base + lane * ELEMS_PER_LANE + e]) *
               __bfloat162float(O[base + lane * ELEMS_PER_LANE + e]);
    sum = warp_reduce_sum(sum);
    if (lane == 0) D[bh * S + qi] = sum;
}

__global__ void compute_dQ_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ D,
    __nv_bfloat16* __restrict__ dQ,
    int S)
{
    int bh = blockIdx.x;
    int q_start = blockIdx.y * WARPS_PER_BLOCK;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int qi = q_start + warp_id;
    int max_qi = min(q_start + WARPS_PER_BLOCK - 1, S - 1);
    bool valid = (qi < S);

    __shared__ __nv_bfloat16 smem_K[TILE_KV * HEAD_DIM];
    __shared__ __nv_bfloat16 smem_V[TILE_KV * HEAD_DIM];

    float q[ELEMS_PER_LANE] = {0.0f}, do_[ELEMS_PER_LANE] = {0.0f};
    float lse = 0.0f, D_val = 0.0f;

    if (valid) {
        int base = bh * S * HEAD_DIM + qi * HEAD_DIM;
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_LANE; e++) {
            q[e] = __bfloat162float(Q[base + lane * ELEMS_PER_LANE + e]);
            do_[e] = __bfloat162float(dO[base + lane * ELEMS_PER_LANE + e]);
        }
        lse = L[bh * S + qi];
        D_val = D[bh * S + qi];
    }

    float scale = rsqrtf((float)HEAD_DIM);
    float dq[ELEMS_PER_LANE] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int kt = 0; kt <= max_qi; kt += TILE_KV) {
        int tile_rows = min(TILE_KV, max_qi - kt + 1);

        for (int i = 0; i < tile_rows; i += WARPS_PER_BLOCK) {
            int row_in_tile = i + warp_id;
            if (row_in_tile < tile_rows) {
                int kj = kt + row_in_tile;
                int gbase = bh * S * HEAD_DIM + kj * HEAD_DIM;
                int sbase = row_in_tile * HEAD_DIM;
                uint2 kv = *reinterpret_cast<const uint2*>(&K[gbase + lane * ELEMS_PER_LANE]);
                *reinterpret_cast<uint2*>(&smem_K[sbase + lane * ELEMS_PER_LANE]) = kv;
                uint2 vv = *reinterpret_cast<const uint2*>(&V[gbase + lane * ELEMS_PER_LANE]);
                *reinterpret_cast<uint2*>(&smem_V[sbase + lane * ELEMS_PER_LANE]) = vv;
            }
        }
        __syncthreads();

        if (valid && qi >= kt) {
            int loop_end = min(kt + tile_rows, qi + 1);
            for (int kj = kt; kj < loop_end; kj++) {
                int lr = kj - kt;
                float score = 0.0f, dot_dov = 0.0f;
                float k_vals[ELEMS_PER_LANE], v_vals[ELEMS_PER_LANE];
                #pragma unroll
                for (int e = 0; e < ELEMS_PER_LANE; e++) {
                    k_vals[e] = __bfloat162float(smem_K[lr * HEAD_DIM + lane * ELEMS_PER_LANE + e]);
                    v_vals[e] = __bfloat162float(smem_V[lr * HEAD_DIM + lane * ELEMS_PER_LANE + e]);
                    score += q[e] * k_vals[e];
                    dot_dov += do_[e] * v_vals[e];
                }
                score = warp_reduce_sum(score);
                dot_dov = warp_reduce_sum(dot_dov);

                float P = __expf(score * scale - lse);
                float dS = P * (dot_dov - D_val);

                #pragma unroll
                for (int e = 0; e < ELEMS_PER_LANE; e++)
                    dq[e] += dS * k_vals[e] * scale;
            }
        }
        __syncthreads();
    }

    if (valid) {
        int base = bh * S * HEAD_DIM + qi * HEAD_DIM;
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_LANE; e++)
            dQ[base + lane * ELEMS_PER_LANE + e] = __float2bfloat16(dq[e]);
    }
}

__global__ void compute_dKV_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ D,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int S)
{
    int bh = blockIdx.x;
    int k_start = blockIdx.y * WARPS_PER_BLOCK;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int kj = k_start + warp_id;
    bool valid = (kj < S);

    __shared__ __nv_bfloat16 smem_Q[BQ_TILE * HEAD_DIM];
    __shared__ __nv_bfloat16 smem_dO[BQ_TILE * HEAD_DIM];
    __shared__ float smem_L[BQ_TILE];
    __shared__ float smem_D[BQ_TILE];

    float k_val[ELEMS_PER_LANE] = {0.0f}, v_val[ELEMS_PER_LANE] = {0.0f};
    if (valid) {
        int base = bh * S * HEAD_DIM + kj * HEAD_DIM;
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_LANE; e++) {
            k_val[e] = __bfloat162float(K[base + lane * ELEMS_PER_LANE + e]);
            v_val[e] = __bfloat162float(V[base + lane * ELEMS_PER_LANE + e]);
        }
    }

    float scale = rsqrtf((float)HEAD_DIM);
    float dk[ELEMS_PER_LANE] = {0.0f, 0.0f, 0.0f, 0.0f};
    float dv[ELEMS_PER_LANE] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int qt = k_start; qt < S; qt += BQ_TILE) {
        int tile_rows = min(BQ_TILE, S - qt);

        for (int i = 0; i < tile_rows; i += WARPS_PER_BLOCK) {
            int row_in_tile = i + warp_id;
            if (row_in_tile < tile_rows) {
                int qi = qt + row_in_tile;
                int gbase = bh * S * HEAD_DIM + qi * HEAD_DIM;
                int sbase = row_in_tile * HEAD_DIM;
                uint2 qv = *reinterpret_cast<const uint2*>(&Q[gbase + lane * ELEMS_PER_LANE]);
                *reinterpret_cast<uint2*>(&smem_Q[sbase + lane * ELEMS_PER_LANE]) = qv;
                uint2 dv2 = *reinterpret_cast<const uint2*>(&dO[gbase + lane * ELEMS_PER_LANE]);
                *reinterpret_cast<uint2*>(&smem_dO[sbase + lane * ELEMS_PER_LANE]) = dv2;
            }
        }
        for (int i = threadIdx.x; i < tile_rows; i += THREADS_PER_BLOCK) {
            smem_L[i] = L[bh * S + qt + i];
            smem_D[i] = D[bh * S + qt + i];
        }
        __syncthreads();

        for (int qi_local = 0; qi_local < tile_rows; qi_local++) {
            int qi = qt + qi_local;
            if (!valid || qi < kj) continue;

            float score = 0.0f, dot_dov = 0.0f;
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_LANE; e++) {
                float qv = __bfloat162float(smem_Q[qi_local * HEAD_DIM + lane * ELEMS_PER_LANE + e]);
                float dov = __bfloat162float(smem_dO[qi_local * HEAD_DIM + lane * ELEMS_PER_LANE + e]);
                score += qv * k_val[e];
                dot_dov += dov * v_val[e];
            }
            score = warp_reduce_sum(score);
            dot_dov = warp_reduce_sum(dot_dov);

            float P = __expf(score * scale - smem_L[qi_local]);
            float dS = P * (dot_dov - smem_D[qi_local]);

            #pragma unroll
            for (int e = 0; e < ELEMS_PER_LANE; e++) {
                dk[e] += dS * __bfloat162float(smem_Q[qi_local * HEAD_DIM + lane * ELEMS_PER_LANE + e]) * scale;
                dv[e] += P * __bfloat162float(smem_dO[qi_local * HEAD_DIM + lane * ELEMS_PER_LANE + e]);
            }
        }
        __syncthreads();
    }

    if (valid) {
        int base = bh * S * HEAD_DIM + kj * HEAD_DIM;
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_LANE; e++) {
            dK[base + lane * ELEMS_PER_LANE + e] = __float2bfloat16(dk[e]);
            dV[base + lane * ELEMS_PER_LANE + e] = __float2bfloat16(dv[e]);
        }
    }
}

namespace mha_bwd_d128_causal {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {

    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = 4, H = 48, D = 128;
    int S = static_cast<int>(Q.size(2));
    int total_l = B * H * S;

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

    float* D_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&D_buf, total_l * sizeof(float)));

    dim3 grid(B * H, (S + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    dim3 block(THREADS_PER_BLOCK);

    compute_D_kernel<<<grid, block, 0, stream>>>(dO_ptr, O_ptr, D_buf, S);
    compute_dQ_kernel<<<grid, block, 0, stream>>>(
        Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_buf, dQ_ptr, S);
    compute_dKV_kernel<<<grid, block, 0, stream>>>(
        Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_buf, dK_ptr, dV_ptr, S);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(D_buf));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128_causal::run);

}  // namespace mha_bwd_d128_causal