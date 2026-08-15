#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", \
                cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

namespace mha_bwd_d128 {

constexpr int D = 128;
constexpr float SCALE = 0.08838834764831845f;
constexpr float LOG2_E = 1.4426950408889634f;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

__device__ __forceinline__ float fast_expf(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x * LOG2_E));
    return y;
}

__device__ __forceinline__ void load4_bf16(const __nv_bfloat16* ptr, float out[4]) {
    uint2 raw = *reinterpret_cast<const uint2*>(ptr);
    __nv_bfloat16 bf[4];
    *reinterpret_cast<uint2*>(bf) = raw;
    out[0] = __bfloat162float(bf[0]);
    out[1] = __bfloat162float(bf[1]);
    out[2] = __bfloat162float(bf[2]);
    out[3] = __bfloat162float(bf[3]);
}

// Precompute D[bh, i] = sum_d(dO[bh, i, d] * O[bh, i, d])
__global__ void compute_D_kernel(
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    float* __restrict__ D_out,
    int S)
{
    int bh = blockIdx.x;
    int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= S) return;

    const __nv_bfloat16* O_row = O + (int64_t)bh * S * D + i * D;
    const __nv_bfloat16* dO_row = dO + (int64_t)bh * S * D + i * D;

    float d_i = 0.f;
    #pragma unroll
    for (int d = 0; d < D; d += 8) {
        uint4 o_v = *reinterpret_cast<const uint4*>(O_row + d);
        uint4 do_v = *reinterpret_cast<const uint4*>(dO_row + d);
        __nv_bfloat16 o_bf[8], do_bf[8];
        *reinterpret_cast<uint4*>(o_bf) = o_v;
        *reinterpret_cast<uint4*>(do_bf) = do_v;
        #pragma unroll
        for (int dd = 0; dd < 8; dd++)
            d_i += __bfloat162float(o_bf[dd]) * __bfloat162float(do_bf[dd]);
    }
    D_out[bh * S + i] = d_i;
}

// dV/dK kernel: each warp handles one KV row, iterates over all Q rows
// No shared memory - relies on L2 cache for Q/dO reuse across warps
__global__ __launch_bounds__(128)
void dV_dK_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ D_pre,
    __nv_bfloat16* __restrict__ dV,
    __nv_bfloat16* __restrict__ dK,
    int S)
{
    constexpr int WARPS = 4;
    int bh = blockIdx.x;
    int j = blockIdx.y * WARPS + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    int d_base = lane * 4;

    if (j >= S) return;

    const __nv_bfloat16* Q_bh = Q + (int64_t)bh * S * D;
    const __nv_bfloat16* K_bh = K + (int64_t)bh * S * D;
    const __nv_bfloat16* V_bh = V + (int64_t)bh * S * D;
    const __nv_bfloat16* dO_bh = dO + (int64_t)bh * S * D;
    const float* L_bh = L + (int64_t)bh * S;
    const float* D_bh = D_pre + (int64_t)bh * S;

    // Load K[j], V[j] into registers (persistent)
    float k[4], v[4];
    load4_bf16(K_bh + (int64_t)j * D + d_base, k);
    load4_bf16(V_bh + (int64_t)j * D + d_base, v);

    float dV_acc[4] = {0.f, 0.f, 0.f, 0.f};
    float dK_acc[4] = {0.f, 0.f, 0.f, 0.f};

    #pragma unroll 1
    for (int i = 0; i < S; i++) {
        float q[4], do_[4];
        load4_bf16(Q_bh + (int64_t)i * D + d_base, q);
        load4_bf16(dO_bh + (int64_t)i * D + d_base, do_);

        float D_i = D_bh[i];
        float L_i = L_bh[i];

        float S_ij = (q[0]*k[0] + q[1]*k[1] + q[2]*k[2] + q[3]*k[3]) * SCALE;
        S_ij = warp_reduce_sum(S_ij);

        float P_ij = fast_expf(S_ij - L_i);

        float dP_ij = do_[0]*v[0] + do_[1]*v[1] + do_[2]*v[2] + do_[3]*v[3];
        dP_ij = warp_reduce_sum(dP_ij);

        float dS_ij = P_ij * (dP_ij - D_i);

        #pragma unroll
        for (int d = 0; d < 4; d++) {
            dV_acc[d] += P_ij * do_[d];
            dK_acc[d] += dS_ij * q[d] * SCALE;
        }
    }

    // Store dV, dK
    __nv_bfloat16* dV_row = dV + (int64_t)bh * S * D + j * D + d_base;
    __nv_bfloat16* dK_row = dK + (int64_t)bh * S * D + j * D + d_base;
    #pragma unroll
    for (int d = 0; d < 4; d++) {
        dV_row[d] = __float2bfloat16(dV_acc[d]);
        dK_row[d] = __float2bfloat16(dK_acc[d]);
    }
}

// dQ kernel: each warp handles one Q row, iterates over all KV rows
__global__ __launch_bounds__(128)
void dQ_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ D_pre,
    __nv_bfloat16* __restrict__ dQ,
    int S)
{
    constexpr int WARPS = 4;
    int bh = blockIdx.x;
    int i = blockIdx.y * WARPS + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    int d_base = lane * 4;

    if (i >= S) return;

    const __nv_bfloat16* Q_bh = Q + (int64_t)bh * S * D;
    const __nv_bfloat16* K_bh = K + (int64_t)bh * S * D;
    const __nv_bfloat16* V_bh = V + (int64_t)bh * S * D;
    const __nv_bfloat16* dO_bh = dO + (int64_t)bh * S * D;
    const float* L_bh = L + (int64_t)bh * S;
    const float* D_bh = D_pre + (int64_t)bh * S;

    // Load Q[i], dO[i] into registers (persistent)
    float q[4], do_[4];
    load4_bf16(Q_bh + (int64_t)i * D + d_base, q);
    load4_bf16(dO_bh + (int64_t)i * D + d_base, do_);

    float D_i = D_bh[i];
    float L_i = L_bh[i];

    float dQ_acc[4] = {0.f, 0.f, 0.f, 0.f};

    #pragma unroll 1
    for (int j = 0; j < S; j++) {
        float k[4], v[4];
        load4_bf16(K_bh + (int64_t)j * D + d_base, k);
        load4_bf16(V_bh + (int64_t)j * D + d_base, v);

        float S_ij = (q[0]*k[0] + q[1]*k[1] + q[2]*k[2] + q[3]*k[3]) * SCALE;
        S_ij = warp_reduce_sum(S_ij);

        float P_ij = fast_expf(S_ij - L_i);

        float dP_ij = do_[0]*v[0] + do_[1]*v[1] + do_[2]*v[2] + do_[3]*v[3];
        dP_ij = warp_reduce_sum(dP_ij);

        float dS_ij = P_ij * (dP_ij - D_i);

        #pragma unroll
        for (int d = 0; d < 4; d++)
            dQ_acc[d] += dS_ij * k[d] * SCALE;
    }

    // Store dQ
    __nv_bfloat16* dQ_row = dQ + (int64_t)bh * S * D + i * D + d_base;
    #pragma unroll
    for (int d = 0; d < 4; d++)
        dQ_row[d] = __float2bfloat16(dQ_acc[d]);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {

    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t S = Q.size(2);
    const int B = 4, H = 48;
    int BH = B * H;

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

    // Allocate D buffer using cudaMalloc (more portable than cudaMallocAsync)
    float* D_ptr;
    CUDA_CHECK(cudaMalloc(&D_ptr, BH * S * sizeof(float)));

    // Compute D = sum(dO * O) per row
    {
        dim3 grid(BH, (int)((S + 255) / 256));
        dim3 block(256);
        compute_D_kernel<<<grid, block, 0, stream>>>(O_ptr, dO_ptr, D_ptr, (int)S);
    }

    // Compute dV, dK
    {
        int blocks_y = (int)((S + 3) / 4);
        dim3 grid(BH, blocks_y);
        dim3 block(128);
        dV_dK_kernel<<<grid, block, 0, stream>>>(
            Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_ptr, dV_ptr, dK_ptr, (int)S);
    }

    // Compute dQ
    {
        int blocks_y = (int)((S + 3) / 4);
        dim3 grid(BH, blocks_y);
        dim3 block(128);
        dQ_kernel<<<grid, block, 0, stream>>>(
            Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_ptr, dQ_ptr, (int)S);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(D_ptr));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128::run);

}  // namespace mha_bwd_d128