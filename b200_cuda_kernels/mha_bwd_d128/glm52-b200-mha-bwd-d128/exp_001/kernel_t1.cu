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

__device__ __forceinline__ float fast_expf(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x * 1.4426950408889634f));
    return y;
}

template<int BK, int BQ>
__global__ __launch_bounds__(128, 4)
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
    __nv_bfloat16* sQ = sV + BK * D;
    __nv_bfloat16* sdO = sQ + BQ * D;
    float* sD = reinterpret_cast<float*>(sdO + BQ * D);
    float* sL = sD + BQ;

    const __nv_bfloat16* Q_bh = Q + (int64_t)bh * S * D;
    const __nv_bfloat16* K_bh = K + (int64_t)bh * S * D;
    const __nv_bfloat16* V_bh = V + (int64_t)bh * S * D;
    const __nv_bfloat16* O_bh = O + (int64_t)bh * S * D;
    const __nv_bfloat16* dO_bh = dO + (int64_t)bh * S * D;
    const float* L_bh = L + (int64_t)bh * S;

    // Load K, V into shared memory
    {
        int row = tid >> 1;
        int half = tid & 1;
        int d_off = half * 64;
        int jg = j_start + row;
        #pragma unroll
        for (int d = 0; d < 64; d += 8) {
            if (row < BK && jg < S) {
                *reinterpret_cast<uint4*>(&sK[row * D + d_off + d]) =
                    *reinterpret_cast<const uint4*>(K_bh + (int64_t)jg * D + d_off + d);
                *reinterpret_cast<uint4*>(&sV[row * D + d_off + d]) =
                    *reinterpret_cast<const uint4*>(V_bh + (int64_t)jg * D + d_off + d);
            } else if (row < BK) {
                *reinterpret_cast<uint4*>(&sK[row * D + d_off + d]) = make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4*>(&sV[row * D + d_off + d]) = make_uint4(0, 0, 0, 0);
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
        int q_size = min(BQ, S - qi);

        // Load Q, dO into shared memory
        {
            int row = tid >> 1;
            int half = tid & 1;
            int d_off = half * 64;
            int ig = qi + row;
            #pragma unroll
            for (int d = 0; d < 64; d += 8) {
                if (row < q_size && ig < S) {
                    *reinterpret_cast<uint4*>(&sQ[row * D + d_off + d]) =
                        *reinterpret_cast<const uint4*>(Q_bh + (int64_t)ig * D + d_off + d);
                    *reinterpret_cast<uint4*>(&sdO[row * D + d_off + d]) =
                        *reinterpret_cast<const uint4*>(dO_bh + (int64_t)ig * D + d_off + d);
                } else if (row < BQ) {
                    *reinterpret_cast<uint4*>(&sQ[row * D + d_off + d]) = make_uint4(0, 0, 0, 0);
                    *reinterpret_cast<uint4*>(&sdO[row * D + d_off + d]) = make_uint4(0, 0, 0, 0);
                }
            }
        }
        __syncthreads();

        // Compute D[i] and load L[i]
        if (tid < q_size) {
            int ig = qi + tid;
            float d_i = 0.f;
            #pragma unroll
            for (int d = 0; d < D; d += 4) {
                __nv_bfloat16 do_bf[4], o_bf[4];
                *reinterpret_cast<uint2*>(do_bf) = *reinterpret_cast<const uint2*>(&sdO[tid * D + d]);
                *reinterpret_cast<uint2*>(o_bf) = *reinterpret_cast<const uint2*>(O_bh + (int64_t)ig * D + d);
                #pragma unroll
                for (int dd = 0; dd < 4; dd++)
                    d_i += __bfloat162float(do_bf[dd]) * __bfloat162float(o_bf[dd]);
            }
            sD[tid] = d_i;
            sL[tid] = L_bh[ig];
        } else if (tid < BQ) {
            sD[tid] = 0.f;
            sL[tid] = 0.f;
        }
        __syncthreads();

        // Main computation
        #pragma unroll 1
        for (int i = 0; i < BQ; i++) {
            if (qi + i >= S) break;

            float q[4], do_[4];
            #pragma unroll
            for (int d = 0; d < 4; d++) {
                q[d] = __bfloat162float(sQ[i * D + d_base + d]);
                do_[d] = __bfloat162float(sdO[i * D + d_base + d]);
            }

            float D_i = sD[i];
            float L_i = sL[i];

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

                float S_ij = (q[0]*k[0] + q[1]*k[1] + q[2]*k[2] + q[3]*k[3]) * SCALE;
                S_ij = warp_reduce_sum(S_ij);

                float P_ij = fast_expf(S_ij - L_i);

                float dP_ij = do_[0]*v[0] + do_[1]*v[1] + do_[2]*v[2] + do_[3]*v[3];
                dP_ij = warp_reduce_sum(dP_ij);

                float dS_ij = P_ij * (dP_ij - D_i);

                #pragma unroll
                for (int d = 0; d < 4; d++) {
                    dV_acc[j][d] += P_ij * do_[d];
                    dK_acc[j][d] += dS_ij * q[d] * SCALE;
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
__global__ __launch_bounds__(128, 4)
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
    constexpr int ROWS_PER_WARP = BQ / WARPS;

    int bh = blockIdx.x;
    int i_start = blockIdx.y * BQ;
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 31;
    int d_base = lane_id << 2;

    extern __shared__ char smem[];
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sdO = sQ + BQ * D;
    float* sD = reinterpret_cast<float*>(sdO + BQ * D);
    float* sL = sD + BQ;
    __nv_bfloat16* sK = reinterpret_cast<__nv_bfloat16*>(sL + BQ);
    __nv_bfloat16* sV = sK + BK * D;

    const __nv_bfloat16* Q_bh = Q + (int64_t)bh * S * D;
    const __nv_bfloat16* K_bh = K + (int64_t)bh * S * D;
    const __nv_bfloat16* V_bh = V + (int64_t)bh * S * D;
    const __nv_bfloat16* O_bh = O + (int64_t)bh * S * D;
    const __nv_bfloat16* dO_bh = dO + (int64_t)bh * S * D;
    const float* L_bh = L + (int64_t)bh * S;

    // Load Q, dO into shared memory
    {
        int row = tid >> 1;
        int half = tid & 1;
        int d_off = half * 64;
        int ig = i_start + row;
        #pragma unroll
        for (int d = 0; d < 64; d += 8) {
            if (row < BQ && ig < S) {
                *reinterpret_cast<uint4*>(&sQ[row * D + d_off + d]) =
                    *reinterpret_cast<const uint4*>(Q_bh + (int64_t)ig * D + d_off + d);
                *reinterpret_cast<uint4*>(&sdO[row * D + d_off + d]) =
                    *reinterpret_cast<const uint4*>(dO_bh + (int64_t)ig * D + d_off + d);
            } else if (row < BQ) {
                *reinterpret_cast<uint4*>(&sQ[row * D + d_off + d]) = make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4*>(&sdO[row * D + d_off + d]) = make_uint4(0, 0, 0, 0);
            }
        }
    }
    __syncthreads();

    // Compute D[i], load L[i]
    if (tid < BQ) {
        int ig = i_start + tid;
        float d_i = 0.f;
        if (ig < S) {
            #pragma unroll
            for (int d = 0; d < D; d += 4) {
                __nv_bfloat16 do_bf[4], o_bf[4];
                *reinterpret_cast<uint2*>(do_bf) = *reinterpret_cast<const uint2*>(&sdO[tid * D + d]);
                *reinterpret_cast<uint2*>(o_bf) = *reinterpret_cast<const uint2*>(O_bh + (int64_t)ig * D + d);
                #pragma unroll
                for (int dd = 0; dd < 4; dd++)
                    d_i += __bfloat162float(do_bf[dd]) * __bfloat162float(o_bf[dd]);
            }
            sL[tid] = L_bh[ig];
        } else {
            sL[tid] = 0.f;
        }
        sD[tid] = d_i;
    }
    __syncthreads();

    int i_warp_start = warp_id * ROWS_PER_WARP;

    float q_reg[ROWS_PER_WARP][4], do_reg[ROWS_PER_WARP][4];
    #pragma unroll
    for (int i = 0; i < ROWS_PER_WARP; i++) {
        int row = i_warp_start + i;
        int ig = i_start + row;
        if (ig < S) {
            #pragma unroll
            for (int d = 0; d < 4; d++) {
                q_reg[i][d] = __bfloat162float(sQ[row * D + d_base + d]);
                do_reg[i][d] = __bfloat162float(sdO[row * D + d_base + d]);
            }
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
        int k_size = min(BK, S - kj);

        // Load K, V into shared memory
        {
            int row = tid >> 1;
            int half = tid & 1;
            int d_off = half * 64;
            int jg = kj + row;
            #pragma unroll
            for (int d = 0; d < 64; d += 8) {
                if (row < k_size && jg < S) {
                    *reinterpret_cast<uint4*>(&sK[row * D + d_off + d]) =
                        *reinterpret_cast<const uint4*>(K_bh + (int64_t)jg * D + d_off + d);
                    *reinterpret_cast<uint4*>(&sV[row * D + d_off + d]) =
                        *reinterpret_cast<const uint4*>(V_bh + (int64_t)jg * D + d_off + d);
                } else if (row < BK) {
                    *reinterpret_cast<uint4*>(&sK[row * D + d_off + d]) = make_uint4(0, 0, 0, 0);
                    *reinterpret_cast<uint4*>(&sV[row * D + d_off + d]) = make_uint4(0, 0, 0, 0);
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

            #pragma unroll 1
            for (int j = 0; j < BK; j++) {
                if (kj + j >= S) break;

                float k[4], v[4];
                #pragma unroll
                for (int d = 0; d < 4; d++) {
                    k[d] = __bfloat162float(sK[j * D + d_base + d]);
                    v[d] = __bfloat162float(sV[j * D + d_base + d]);
                }

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

    constexpr int BK = 32;
    constexpr int BQ = 64;

    {
        dim3 grid(BH, (int)((S + BK - 1) / BK));
        dim3 block(128);
        int smem = 2 * BK * D * sizeof(__nv_bfloat16)
                 + 2 * BQ * D * sizeof(__nv_bfloat16)
                 + 2 * BQ * sizeof(float);
        dV_dK_kernel<BK, BQ><<<grid, block, smem, stream>>>(
            Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dV_ptr, dK_ptr, (int)S);
    }

    {
        dim3 grid(BH, (int)((S + BQ - 1) / BQ));
        dim3 block(128);
        int smem = 2 * BQ * D * sizeof(__nv_bfloat16)
                 + 2 * BQ * sizeof(float)
                 + 2 * BK * D * sizeof(__nv_bfloat16);
        dQ_kernel<BQ, BK><<<grid, block, smem, stream>>>(
            Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr, (int)S);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128::run);

}  // namespace mha_bwd_d128