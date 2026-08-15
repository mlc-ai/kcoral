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

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

template<int BK>
__global__ __launch_bounds__(128)
void dV_dK_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dV,
    __nv_bfloat16* __restrict__ dK,
    int S)
{
    constexpr int WARPS = 4;
    constexpr int ROWS_PER_WARP = BK / WARPS;

    int bh = blockIdx.x;
    int j_start = blockIdx.y * BK;

    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 31;
    int d_base = lane_id << 2;

    extern __shared__ char smem[];
    __nv_bfloat16* sK = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sV = sK + BK * D;

    #pragma unroll
    for (int j = 0; j < ROWS_PER_WARP; j++) {
        int row = warp_id * ROWS_PER_WARP + j;
        int jg = j_start + row;
        if (jg < S) {
            *reinterpret_cast<uint2*>(&sK[row * D + d_base]) =
                *reinterpret_cast<const uint2*>(K + (int64_t)(bh * S + jg) * D + d_base);
            *reinterpret_cast<uint2*>(&sV[row * D + d_base]) =
                *reinterpret_cast<const uint2*>(V + (int64_t)(bh * S + jg) * D + d_base);
        } else {
            *reinterpret_cast<uint2*>(&sK[row * D + d_base]) = make_uint2(0, 0);
            *reinterpret_cast<uint2*>(&sV[row * D + d_base]) = make_uint2(0, 0);
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
    const __nv_bfloat16* Q_bh = Q + (int64_t)bh * S * D;
    const __nv_bfloat16* dO_bh = dO + (int64_t)bh * S * D;
    const __nv_bfloat16* O_bh = O + (int64_t)bh * S * D;
    const float* L_bh = L + (int64_t)bh * S;

    for (int i = 0; i < S; i++) {
        uint2 q_raw = *reinterpret_cast<const uint2*>(Q_bh + (int64_t)i * D + d_base);
        uint2 do_raw = *reinterpret_cast<const uint2*>(dO_bh + (int64_t)i * D + d_base);
        uint2 o_raw = *reinterpret_cast<const uint2*>(O_bh + (int64_t)i * D + d_base);

        __nv_bfloat16 q_bf[4], do_bf[4], o_bf[4];
        *reinterpret_cast<uint2*>(q_bf) = q_raw;
        *reinterpret_cast<uint2*>(do_bf) = do_raw;
        *reinterpret_cast<uint2*>(o_bf) = o_raw;

        float q[4], do_[4], o_[4];
        #pragma unroll
        for (int d = 0; d < 4; d++) {
            q[d] = __bfloat162float(q_bf[d]);
            do_[d] = __bfloat162float(do_bf[d]);
            o_[d] = __bfloat162float(o_bf[d]);
        }

        float D_i = warp_reduce_sum(do_[0]*o_[0] + do_[1]*o_[1] + do_[2]*o_[2] + do_[3]*o_[3]);
        float L_i = L_bh[i];

        #pragma unroll
        for (int j = 0; j < ROWS_PER_WARP; j++) {
            int jg = j_start + j_warp_start + j;
            if (jg >= S) continue;

            float k[4], v[4];
            #pragma unroll
            for (int d = 0; d < 4; d++) {
                k[d] = __bfloat162float(sK[(j_warp_start + j) * D + d_base + d]);
                v[d] = __bfloat162float(sV[(j_warp_start + j) * D + d_base + d]);
            }

            float S_ij = warp_reduce_sum(q[0]*k[0] + q[1]*k[1] + q[2]*k[2] + q[3]*k[3]) * SCALE;
            float P_ij = expf(S_ij - L_i);
            float dP_ij = warp_reduce_sum(do_[0]*v[0] + do_[1]*v[1] + do_[2]*v[2] + do_[3]*v[3]);
            float dS_ij = P_ij * (dP_ij - D_i);

            #pragma unroll
            for (int d = 0; d < 4; d++) {
                dV_acc[j][d] += P_ij * do_[d];
                dK_acc[j][d] += dS_ij * q[d] * SCALE;
            }
        }
    }

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

template<int BQ, int BK2>
__global__ __launch_bounds__(128)
void dQ_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    int S)
{
    constexpr int WARPS = 4;
    constexpr int Q_ROWS_PER_WARP = BQ / WARPS;
    constexpr int KV_ROWS_PER_WARP = BK2 / WARPS;

    int bh = blockIdx.x;
    int i_start = blockIdx.y * BQ;

    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 31;
    int d_base = lane_id << 2;

    extern __shared__ char smem[];
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sdO = sQ + BQ * D;
    __nv_bfloat16* sO = sdO + BQ * D;
    float* sD = reinterpret_cast<float*>(sO + BQ * D);
    float* sL = sD + BQ;
    __nv_bfloat16* sK = reinterpret_cast<__nv_bfloat16*>(sL + BQ);
    __nv_bfloat16* sV = sK + BK2 * D;

    #pragma unroll
    for (int i = 0; i < Q_ROWS_PER_WARP; i++) {
        int row = warp_id * Q_ROWS_PER_WARP + i;
        int ig = i_start + row;
        if (ig < S) {
            *reinterpret_cast<uint2*>(&sQ[row * D + d_base]) =
                *reinterpret_cast<const uint2*>(Q + (int64_t)(bh * S + ig) * D + d_base);
            *reinterpret_cast<uint2*>(&sdO[row * D + d_base]) =
                *reinterpret_cast<const uint2*>(dO + (int64_t)(bh * S + ig) * D + d_base);
            *reinterpret_cast<uint2*>(&sO[row * D + d_base]) =
                *reinterpret_cast<const uint2*>(O + (int64_t)(bh * S + ig) * D + d_base);
        }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < Q_ROWS_PER_WARP; i++) {
        int row = warp_id * Q_ROWS_PER_WARP + i;
        int ig = i_start + row;
        if (ig >= S) { sD[row] = 0.f; sL[row] = 0.f; continue; }
        float partial = 0.f;
        #pragma unroll
        for (int d = 0; d < 4; d++)
            partial += __bfloat162float(sdO[row * D + d_base + d]) * __bfloat162float(sO[row * D + d_base + d]);
        sD[row] = warp_reduce_sum(partial);
        sL[row] = L[(int64_t)bh * S + ig];
    }
    __syncthreads();

    int i_warp_start = warp_id * Q_ROWS_PER_WARP;

    float q_reg[Q_ROWS_PER_WARP][4], do_reg[Q_ROWS_PER_WARP][4];
    #pragma unroll
    for (int i = 0; i < Q_ROWS_PER_WARP; i++) {
        int row = i_warp_start + i;
        int ig = i_start + row;
        if (ig < S) {
            #pragma unroll
            for (int d = 0; d < 4; d++) {
                q_reg[i][d] = __bfloat162float(sQ[row * D + d_base + d]);
                do_reg[i][d] = __bfloat162float(sdO[row * D + d_base + d]);
            }
        }
    }

    float dQ_acc[Q_ROWS_PER_WARP][4];
    #pragma unroll
    for (int i = 0; i < Q_ROWS_PER_WARP; i++)
        #pragma unroll
        for (int d = 0; d < 4; d++) dQ_acc[i][d] = 0.f;

    for (int j_start = 0; j_start < S; j_start += BK2) {
        #pragma unroll
        for (int j = 0; j < KV_ROWS_PER_WARP; j++) {
            int row = warp_id * KV_ROWS_PER_WARP + j;
            int jg = j_start + row;
            if (jg < S) {
                *reinterpret_cast<uint2*>(&sK[row * D + d_base]) =
                    *reinterpret_cast<const uint2*>(K + (int64_t)(bh * S + jg) * D + d_base);
                *reinterpret_cast<uint2*>(&sV[row * D + d_base]) =
                    *reinterpret_cast<const uint2*>(V + (int64_t)(bh * S + jg) * D + d_base);
            } else {
                *reinterpret_cast<uint2*>(&sK[row * D + d_base]) = make_uint2(0, 0);
                *reinterpret_cast<uint2*>(&sV[row * D + d_base]) = make_uint2(0, 0);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < Q_ROWS_PER_WARP; i++) {
            int ig = i_start + i_warp_start + i;
            if (ig >= S) continue;
            float D_i = sD[i_warp_start + i];
            float L_i = sL[i_warp_start + i];

            for (int j = 0; j < BK2; j++) {
                int jg = j_start + j;
                if (jg >= S) continue;

                float k[4], v[4];
                #pragma unroll
                for (int d = 0; d < 4; d++) {
                    k[d] = __bfloat162float(sK[j * D + d_base + d]);
                    v[d] = __bfloat162float(sV[j * D + d_base + d]);
                }

                float S_ij = warp_reduce_sum(
                    q_reg[i][0]*k[0] + q_reg[i][1]*k[1] + q_reg[i][2]*k[2] + q_reg[i][3]*k[3]) * SCALE;
                float P_ij = expf(S_ij - L_i);
                float dP_ij = warp_reduce_sum(
                    do_reg[i][0]*v[0] + do_reg[i][1]*v[1] + do_reg[i][2]*v[2] + do_reg[i][3]*v[3]);
                float dS_ij = P_ij * (dP_ij - D_i);

                #pragma unroll
                for (int d = 0; d < 4; d++)
                    dQ_acc[i][d] += dS_ij * k[d] * SCALE;
            }
        }
        __syncthreads();
    }

    __nv_bfloat16* dQ_bh = dQ + (int64_t)bh * S * D;
    #pragma unroll
    for (int i = 0; i < Q_ROWS_PER_WARP; i++) {
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

    constexpr int BK = 32;
    constexpr int BQ = 32;
    constexpr int BK2 = 32;

    {
        dim3 grid(BH, (int)((S + BK - 1) / BK));
        dim3 block(128);
        int smem = 2 * BK * D * sizeof(__nv_bfloat16);
        dV_dK_kernel<BK><<<grid, block, smem, stream>>>(
            Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dV_ptr, dK_ptr, (int)S);
    }

    {
        dim3 grid(BH, (int)((S + BQ - 1) / BQ));
        dim3 block(128);
        int smem = 3 * BQ * D * sizeof(__nv_bfloat16)
                   + 2 * BQ * sizeof(float)
                   + 2 * BK2 * D * sizeof(__nv_bfloat16);
        dQ_kernel<BQ, BK2><<<grid, block, smem, stream>>>(
            Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr, (int)S);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128::run);

}  // namespace mha_bwd_d128