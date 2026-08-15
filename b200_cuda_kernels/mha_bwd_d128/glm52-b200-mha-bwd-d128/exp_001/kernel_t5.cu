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

__device__ __forceinline__ void load4_bf16_to_float(const __nv_bfloat16* ptr, float out[4]) {
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
    int i = blockIdx.y * 4 + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (i >= S) return;

    int d_base = lane * 4;
    const __nv_bfloat16* O_row = O + (int64_t)bh * S * D + i * D + d_base;
    const __nv_bfloat16* dO_row = dO + (int64_t)bh * S * D + i * D + d_base;

    float o[4], do_[4];
    load4_bf16_to_float(O_row, o);
    load4_bf16_to_float(dO_row, do_);

    float val = do_[0]*o[0] + do_[1]*o[1] + do_[2]*o[2] + do_[3]*o[3];
    val = warp_reduce_sum(val);

    if (lane == 0) D_out[bh * S + i] = val;
}

template<int BK, int BQ>
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
    constexpr int ROWS_PER_WARP = BK / WARPS;
    constexpr int U4_PER_ROW = D / 8;  // 16

    int bh = blockIdx.x;
    int j_start = blockIdx.y * BK;
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 31;
    int d_base = lane_id << 2;

    __shared__ __nv_bfloat16 sK[BK][D];
    __shared__ __nv_bfloat16 sV[BK][D];
    __shared__ __nv_bfloat16 sQ[BQ][D];
    __shared__ __nv_bfloat16 sdO[BQ][D];
    __shared__ float sD[BQ];
    __shared__ float sL[BQ];

    const __nv_bfloat16* Q_bh = Q + (int64_t)bh * S * D;
    const __nv_bfloat16* K_bh = K + (int64_t)bh * S * D;
    const __nv_bfloat16* V_bh = V + (int64_t)bh * S * D;
    const __nv_bfloat16* dO_bh = dO + (int64_t)bh * S * D;
    const float* L_bh = L + (int64_t)bh * S;
    const float* D_bh = D_pre + (int64_t)bh * S;

    // Load K, V
    {
        int row = tid / U4_PER_ROW;
        int col = (tid % U4_PER_ROW) * 8;
        if (row < BK) {
            if (j_start + row < S) {
                *reinterpret_cast<uint4*>(&sK[row][col]) =
                    *reinterpret_cast<const uint4*>(K_bh + (int64_t)(j_start + row) * D + col);
                *reinterpret_cast<uint4*>(&sV[row][col]) =
                    *reinterpret_cast<const uint4*>(V_bh + (int64_t)(j_start + row) * D + col);
            } else {
                *reinterpret_cast<uint4*>(&sK[row][col]) = make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4*>(&sV[row][col]) = make_uint4(0, 0, 0, 0);
            }
        }
    }
    __syncthreads();

    float dV_acc[ROWS_PER_WARP][4];
    float dK_acc[ROWS_PER_WARP][4];
    #pragma unroll
    for (int j = 0; j < ROWS_PER_WARP; j++)
        #pragma unroll
        for (int d = 0; d < 4; d++) {
            dV_acc[j][d] = 0.f;
            dK_acc[j][d] = 0.f;
        }

    int j_warp_start = warp_id * ROWS_PER_WARP;

    for (int qi = 0; qi < S; qi += BQ) {
        // Load Q, dO
        {
            int row = tid / U4_PER_ROW;
            int col = (tid % U4_PER_ROW) * 8;
            if (row < BQ) {
                if (qi + row < S) {
                    *reinterpret_cast<uint4*>(&sQ[row][col]) =
                        *reinterpret_cast<const uint4*>(Q_bh + (int64_t)(qi + row) * D + col);
                    *reinterpret_cast<uint4*>(&sdO[row][col]) =
                        *reinterpret_cast<const uint4*>(dO_bh + (int64_t)(qi + row) * D + col);
                } else {
                    *reinterpret_cast<uint4*>(&sQ[row][col]) = make_uint4(0, 0, 0, 0);
                    *reinterpret_cast<uint4*>(&sdO[row][col]) = make_uint4(0, 0, 0, 0);
                }
            }
        }
        __syncthreads();

        // Load D, L
        if (tid < BQ) {
            int ig = qi + tid;
            if (ig < S) {
                sD[tid] = D_bh[ig];
                sL[tid] = L_bh[ig];
            } else {
                sD[tid] = 0.f;
                sL[tid] = 0.f;
            }
        }
        __syncthreads();

        // Main computation
        #pragma unroll
        for (int i = 0; i < BQ; i++) {
            if (qi + i >= S) break;

            float q[4], do_[4];
            load4_bf16_to_float(&sQ[i][d_base], q);
            load4_bf16_to_float(&sdO[i][d_base], do_);
            #pragma unroll
            for (int d = 0; d < 4; d++) q[d] *= SCALE;

            float D_i = sD[i];
            float L_i = sL[i];

            #pragma unroll
            for (int j = 0; j < ROWS_PER_WARP; j++) {
                int jg = j_start + j_warp_start + j;
                if (jg >= S) continue;

                float k[4], v[4];
                load4_bf16_to_float(&sK[j_warp_start + j][d_base], k);
                load4_bf16_to_float(&sV[j_warp_start + j][d_base], v);

                float S_ij = q[0]*k[0] + q[1]*k[1] + q[2]*k[2] + q[3]*k[3];
                S_ij = warp_reduce_sum(S_ij);

                float P_ij = fast_expf(S_ij - L_i);

                float dP_ij = do_[0]*v[0] + do_[1]*v[1] + do_[2]*v[2] + do_[3]*v[3];
                dP_ij = warp_reduce_sum(dP_ij);

                float dS_ij = P_ij * (dP_ij - D_i);

                #pragma unroll
                for (int d = 0; d < 4; d++) {
                    dV_acc[j][d] += P_ij * do_[d];
                    dK_acc[j][d] += dS_ij * q[d];
                }
            }
        }
        __syncthreads();
    }

    // Store dV, dK
    __nv_bfloat16* dV_bh = dV + (int64_t)bh * S * D;
    __nv_bfloat16* dK_bh = dK + (int64_t)bh * S * D;

    #pragma unroll
    for (int j = 0; j < ROWS_PER_WARP; j++) {
        int jg = j_start + j_warp_start + j;
        if (jg >= S) continue;
        #pragma unroll
        for (int d = 0; d < 4; d++) {
            dV_bh[(int64_t)jg * D + d_base + d] = __float2bfloat16(dV_acc[j][d]);
            dK_bh[(int64_t)jg * D + d_base + d] = __float2bfloat16(dK_acc[j][d]);
        }
    }
}

template<int BQ, int BK>
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
    constexpr int ROWS_PER_WARP = BQ / WARPS;
    constexpr int U4_PER_ROW = D / 8;

    int bh = blockIdx.x;
    int i_start = blockIdx.y * BQ;
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 31;
    int d_base = lane_id << 2;

    __shared__ __nv_bfloat16 sQ[BQ][D];
    __shared__ __nv_bfloat16 sdO[BQ][D];
    __shared__ float sD[BQ];
    __shared__ float sL[BQ];
    __shared__ __nv_bfloat16 sK[BK][D];
    __shared__ __nv_bfloat16 sV[BK][D];

    const __nv_bfloat16* Q_bh = Q + (int64_t)bh * S * D;
    const __nv_bfloat16* K_bh = K + (int64_t)bh * S * D;
    const __nv_bfloat16* V_bh = V + (int64_t)bh * S * D;
    const __nv_bfloat16* dO_bh = dO + (int64_t)bh * S * D;
    const float* L_bh = L + (int64_t)bh * S;
    const float* D_bh = D_pre + (int64_t)bh * S;

    // Load Q, dO
    {
        int row = tid / U4_PER_ROW;
        int col = (tid % U4_PER_ROW) * 8;
        if (row < BQ) {
            if (i_start + row < S) {
                *reinterpret_cast<uint4*>(&sQ[row][col]) =
                    *reinterpret_cast<const uint4*>(Q_bh + (int64_t)(i_start + row) * D + col);
                *reinterpret_cast<uint4*>(&sdO[row][col]) =
                    *reinterpret_cast<const uint4*>(dO_bh + (int64_t)(i_start + row) * D + col);
            } else {
                *reinterpret_cast<uint4*>(&sQ[row][col]) = make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4*>(&sdO[row][col]) = make_uint4(0, 0, 0, 0);
            }
        }
    }
    __syncthreads();

    // Load D, L
    if (tid < BQ) {
        int ig = i_start + tid;
        if (ig < S) {
            sD[tid] = D_bh[ig];
            sL[tid] = L_bh[ig];
        } else {
            sD[tid] = 0.f;
            sL[tid] = 0.f;
        }
    }
    __syncthreads();

    int i_warp_start = warp_id * ROWS_PER_WARP;

    // Cache Q, dO in registers
    float q_reg[ROWS_PER_WARP][4], do_reg[ROWS_PER_WARP][4];
    #pragma unroll
    for (int i = 0; i < ROWS_PER_WARP; i++) {
        int row = i_warp_start + i;
        int ig = i_start + row;
        if (ig < S) {
            load4_bf16_to_float(&sQ[row][d_base], q_reg[i]);
            load4_bf16_to_float(&sdO[row][d_base], do_reg[i]);
        } else {
            #pragma unroll
            for (int d = 0; d < 4; d++) {
                q_reg[i][d] = 0.f;
                do_reg[i][d] = 0.f;
            }
        }
    }

    float dQ_acc[ROWS_PER_WARP][4];
    #pragma unroll
    for (int i = 0; i < ROWS_PER_WARP; i++)
        #pragma unroll
        for (int d = 0; d < 4; d++)
            dQ_acc[i][d] = 0.f;

    for (int kj = 0; kj < S; kj += BK) {
        // Load K, V
        {
            int row = tid / U4_PER_ROW;
            int col = (tid % U4_PER_ROW) * 8;
            if (row < BK) {
                if (kj + row < S) {
                    *reinterpret_cast<uint4*>(&sK[row][col]) =
                        *reinterpret_cast<const uint4*>(K_bh + (int64_t)(kj + row) * D + col);
                    *reinterpret_cast<uint4*>(&sV[row][col]) =
                        *reinterpret_cast<const uint4*>(V_bh + (int64_t)(kj + row) * D + col);
                } else {
                    *reinterpret_cast<uint4*>(&sK[row][col]) = make_uint4(0, 0, 0, 0);
                    *reinterpret_cast<uint4*>(&sV[row][col]) = make_uint4(0, 0, 0, 0);
                }
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < ROWS_PER_WARP; i++) {
            int ig = i_start + i_warp_start + i;
            if (ig >= S) continue;

            float D_i = sD[i_warp_start + i];
            float L_i = sL[i_warp_start + i];

            #pragma unroll
            for (int j = 0; j < BK; j++) {
                if (kj + j >= S) break;

                float k[4], v[4];
                load4_bf16_to_float(&sK[j][d_base], k);
                load4_bf16_to_float(&sV[j][d_base], v);

                float S_ij = (q_reg[i][0]*k[0] + q_reg[i][1]*k[1] + q_reg[i][2]*k[2] + q_reg[i][3]*k[3]) * SCALE;
                S_ij = warp_reduce_sum(S_ij);

                float P_ij = fast_expf(S_ij - L_i);

                float dP_ij = do_reg[i][0]*v[0] + do_reg[i][1]*v[1] + do_reg[i][2]*v[2] + do_reg[i][3]*v[3];
                dP_ij = warp_reduce_sum(dP_ij);

                float dS_ij = P_ij * (dP_ij - D_i);

                #pragma unroll
                for (int d = 0; d < 4; d++)
                    dQ_acc[i][d] += dS_ij * k[d] * SCALE;
            }
        }
        __syncthreads();
    }

    // Store dQ
    __nv_bfloat16* dQ_bh = dQ + (int64_t)bh * S * D;
    #pragma unroll
    for (int i = 0; i < ROWS_PER_WARP; i++) {
        int ig = i_start + i_warp_start + i;
        if (ig >= S) continue;
        #pragma unroll
        for (int d = 0; d < 4; d++)
            dQ_bh[(int64_t)ig * D + d_base + d] = __float2bfloat16(dQ_acc[i][d]);
    }
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

    float* D_ptr;
    CUDA_CHECK(cudaMallocAsync(&D_ptr, BH * S * sizeof(float), stream));

    {
        dim3 grid(BH, (int)((S + 3) / 4));
        dim3 block(128);
        compute_D_kernel<<<grid, block, 0, stream>>>(O_ptr, dO_ptr, D_ptr, (int)S);
    }

    constexpr int BK1 = 8;
    constexpr int BQ1 = 8;
    {
        dim3 grid(BH, (int)((S + BK1 - 1) / BK1));
        dim3 block(128);
        dV_dK_kernel<BK1, BQ1><<<grid, block, 0, stream>>>(
            Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_ptr, dV_ptr, dK_ptr, (int)S);
    }

    constexpr int BQ2 = 8;
    constexpr int BK2 = 8;
    {
        dim3 grid(BH, (int)((S + BQ2 - 1) / BQ2));
        dim3 block(128);
        dQ_kernel<BQ2, BK2><<<grid, block, 0, stream>>>(
            Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_ptr, dQ_ptr, (int)S);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFreeAsync(D_ptr, stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128::run);

}  // namespace mha_bwd_d128