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

__device__ __forceinline__ void store4_bf16(__nv_bfloat16* ptr, const float in[4]) {
    __nv_bfloat16 bf[4];
    bf[0] = __float2bfloat16(in[0]);
    bf[1] = __float2bfloat16(in[1]);
    bf[2] = __float2bfloat16(in[2]);
    bf[3] = __float2bfloat16(in[3]);
    *reinterpret_cast<uint2*>(ptr) = *reinterpret_cast<uint2*>(bf);
}

// ============================================================
// dV/dK kernel: BK=8 KV rows per CTA, BQ=32 Q rows per tile
// 4 warps, 2 KV rows per warp (NJ=2)
// Q/dO tiled in shared memory, K/V persistent in shared memory
// D computed on-the-fly (no separate kernel, no cudaMalloc)
// ============================================================
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
    constexpr int BK = 8;
    constexpr int BQ = 32;
    constexpr int WARPS = 4;
    constexpr int NJ = BK / WARPS;       // 2 KV rows per warp
    constexpr int DQ = BQ / WARPS;       // 8 Q rows per warp for D computation

    int bh = blockIdx.x;
    int j_start = blockIdx.y * BK;
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;
    int d_base = lane * 4;

    __shared__ __nv_bfloat16 sK[BK][D];
    __shared__ __nv_bfloat16 sV[BK][D];
    __shared__ __nv_bfloat16 sQ[BQ][D];
    __shared__ __nv_bfloat16 sdO[BQ][D];
    __shared__ float sD[BQ];
    __shared__ float sL[BQ];

    const __nv_bfloat16* Q_bh = Q + (int64_t)bh * S * D;
    const __nv_bfloat16* K_bh = K + (int64_t)bh * S * D;
    const __nv_bfloat16* V_bh = V + (int64_t)bh * S * D;
    const __nv_bfloat16* O_bh = O + (int64_t)bh * S * D;
    const __nv_bfloat16* dO_bh = dO + (int64_t)bh * S * D;
    const float* L_bh = L + (int64_t)bh * S;

    // Load K, V into shared memory (8 rows * 128 bf16 = 128 uint4s, 128 threads = 1 per thread)
    {
        constexpr int U4 = D / 8;
        int row = tid / U4;
        int col = (tid % U4) * 8;
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
    __syncthreads();

    // Load K, V into registers for this warp
    float k_reg[NJ][4], v_reg[NJ][4];
    int j_warp = warp_id * NJ;
    #pragma unroll
    for (int nj = 0; nj < NJ; nj++) {
        load4_bf16(&sK[j_warp + nj][d_base], k_reg[nj]);
        load4_bf16(&sV[j_warp + nj][d_base], v_reg[nj]);
    }

    // Accumulators
    float dV_acc[NJ][4], dK_acc[NJ][4];
    #pragma unroll
    for (int nj = 0; nj < NJ; nj++)
        #pragma unroll
        for (int d = 0; d < 4; d++) {
            dV_acc[nj][d] = 0.f;
            dK_acc[nj][d] = 0.f;
        }

    // Iterate over Q tiles
    for (int qi = 0; qi < S; qi += BQ) {
        // Load Q, dO into shared memory (32 rows * 16 uint4s = 512 uint4s, 128 threads = 4 per thread)
        {
            constexpr int U4 = D / 8;
            constexpr int TOTAL = BQ * U4;
            for (int idx = tid; idx < TOTAL; idx += 128) {
                int row = idx / U4;
                int col = (idx % U4) * 8;
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

        // Compute D[i] and load L[i] (each warp handles 8 Q rows)
        {
            int q_warp = warp_id * DQ;
            #pragma unroll
            for (int ni = 0; ni < DQ; ni++) {
                int i_local = q_warp + ni;
                int ig = qi + i_local;
                if (ig < S) {
                    float o[4], do_[4];
                    load4_bf16(O_bh + (int64_t)ig * D + d_base, o);
                    #pragma unroll
                    for (int d = 0; d < 4; d++)
                        do_[d] = __bfloat162float(sdO[i_local][d_base + d]);
                    float d_i = warp_reduce_sum(
                        do_[0]*o[0] + do_[1]*o[1] + do_[2]*o[2] + do_[3]*o[3]);
                    if (lane == 0) {
                        sD[i_local] = d_i;
                        sL[i_local] = L_bh[ig];
                    }
                } else {
                    if (lane == 0) {
                        sD[i_local] = 0.f;
                        sL[i_local] = 0.f;
                    }
                }
            }
        }
        __syncthreads();

        // Main computation
        #pragma unroll 1
        for (int i = 0; i < BQ; i++) {
            if (qi + i >= S) break;

            float q[4], do_[4];
            load4_bf16(&sQ[i][d_base], q);
            load4_bf16(&sdO[i][d_base], do_);
            #pragma unroll
            for (int d = 0; d < 4; d++) q[d] *= SCALE;

            float D_i = sD[i];
            float L_i = sL[i];

            #pragma unroll
            for (int nj = 0; nj < NJ; nj++) {
                float S_ij = warp_reduce_sum(
                    q[0]*k_reg[nj][0] + q[1]*k_reg[nj][1] + q[2]*k_reg[nj][2] + q[3]*k_reg[nj][3]);
                float P_ij = fast_expf(S_ij - L_i);
                float dP_ij = warp_reduce_sum(
                    do_[0]*v_reg[nj][0] + do_[1]*v_reg[nj][1] + do_[2]*v_reg[nj][2] + do_[3]*v_reg[nj][3]);
                float dS_ij = P_ij * (dP_ij - D_i);

                #pragma unroll
                for (int d = 0; d < 4; d++) {
                    dV_acc[nj][d] += P_ij * do_[d];
                    dK_acc[nj][d] += dS_ij * q[d];
                }
            }
        }
        __syncthreads();
    }

    // Store dV, dK
    __nv_bfloat16* dV_bh = dV + (int64_t)bh * S * D;
    __nv_bfloat16* dK_bh = dK + (int64_t)bh * S * D;

    #pragma unroll
    for (int nj = 0; nj < NJ; nj++) {
        int j = j_start + j_warp + nj;
        if (j >= S) continue;
        #pragma unroll
        for (int d = 0; d < 4; d++) {
            dV_bh[(int64_t)j * D + d_base + d] = __float2bfloat16(dV_acc[nj][d]);
            dK_bh[(int64_t)j * D + d_base + d] = __float2bfloat16(dK_acc[nj][d]);
        }
    }
}

// ============================================================
// dQ kernel: BQ=8 Q rows per CTA, BK=32 KV rows per tile
// 4 warps, 2 Q rows per warp (NI=2)
// Q/dO cached in registers, K/V tiled in shared memory
// D computed on-the-fly
// ============================================================
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
    constexpr int BQ = 8;
    constexpr int BK = 32;
    constexpr int WARPS = 4;
    constexpr int NI = BQ / WARPS;     // 2 Q rows per warp

    int bh = blockIdx.x;
    int i_start = blockIdx.y * BQ;
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;
    int d_base = lane * 4;

    __shared__ __nv_bfloat16 sQ[BQ][D];
    __shared__ __nv_bfloat16 sdO[BQ][D];
    __shared__ float sD[BQ];
    __shared__ float sL[BQ];
    __shared__ __nv_bfloat16 sK[BK][D];
    __shared__ __nv_bfloat16 sV[BK][D];

    const __nv_bfloat16* Q_bh = Q + (int64_t)bh * S * D;
    const __nv_bfloat16* K_bh = K + (int64_t)bh * S * D;
    const __nv_bfloat16* V_bh = V + (int64_t)bh * S * D;
    const __nv_bfloat16* O_bh = O + (int64_t)bh * S * D;
    const __nv_bfloat16* dO_bh = dO + (int64_t)bh * S * D;
    const float* L_bh = L + (int64_t)bh * S;

    // Load Q, dO into shared memory (8 rows * 16 uint4s = 128 uint4s, 128 threads = 1 per thread)
    {
        constexpr int U4 = D / 8;
        int row = tid / U4;
        int col = (tid % U4) * 8;
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
    __syncthreads();

    // Compute D[i] and load L[i] (each warp handles 2 Q rows)
    {
        int q_warp = warp_id * NI;
        #pragma unroll
        for (int ni = 0; ni < NI; ni++) {
            int i_local = q_warp + ni;
            int ig = i_start + i_local;
            if (ig < S) {
                float o[4], do_[4];
                load4_bf16(O_bh + (int64_t)ig * D + d_base, o);
                #pragma unroll
                for (int d = 0; d < 4; d++)
                    do_[d] = __bfloat162float(sdO[i_local][d_base + d]);
                float d_i = warp_reduce_sum(
                    do_[0]*o[0] + do_[1]*o[1] + do_[2]*o[2] + do_[3]*o[3]);
                if (lane == 0) {
                    sD[i_local] = d_i;
                    sL[i_local] = L_bh[ig];
                }
            } else {
                if (lane == 0) {
                    sD[i_local] = 0.f;
                    sL[i_local] = 0.f;
                }
            }
        }
    }
    __syncthreads();

    // Cache Q, dO, D, L in registers
    int i_warp = warp_id * NI;
    float q_reg[NI][4], do_reg[NI][4];
    float D_reg[NI], L_reg[NI];

    #pragma unroll
    for (int ni = 0; ni < NI; ni++) {
        int i_local = i_warp + ni;
        int ig = i_start + i_local;
        if (ig < S) {
            load4_bf16(&sQ[i_local][d_base], q_reg[ni]);
            load4_bf16(&sdO[i_local][d_base], do_reg[ni]);
            D_reg[ni] = sD[i_local];
            L_reg[ni] = sL[i_local];
        } else {
            #pragma unroll
            for (int d = 0; d < 4; d++) {
                q_reg[ni][d] = 0.f;
                do_reg[ni][d] = 0.f;
            }
            D_reg[ni] = 0.f;
            L_reg[ni] = 0.f;
        }
    }

    // dQ accumulators
    float dQ_acc[NI][4];
    #pragma unroll
    for (int ni = 0; ni < NI; ni++)
        #pragma unroll
        for (int d = 0; d < 4; d++)
            dQ_acc[ni][d] = 0.f;

    // Iterate over KV tiles
    for (int kj = 0; kj < S; kj += BK) {
        // Load K, V into shared memory (32 rows * 16 uint4s = 512 uint4s, 128 threads = 4 per thread)
        {
            constexpr int U4 = D / 8;
            constexpr int TOTAL = BK * U4;
            for (int idx = tid; idx < TOTAL; idx += 128) {
                int row = idx / U4;
                int col = (idx % U4) * 8;
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

        // Main computation
        #pragma unroll 1
        for (int j = 0; j < BK; j++) {
            if (kj + j >= S) break;

            float k[4], v[4];
            load4_bf16(&sK[j][d_base], k);
            load4_bf16(&sV[j][d_base], v);

            #pragma unroll
            for (int ni = 0; ni < NI; ni++) {
                int ig = i_start + i_warp + ni;
                if (ig >= S) continue;

                float S_ij = warp_reduce_sum(
                    (q_reg[ni][0]*k[0] + q_reg[ni][1]*k[1] + q_reg[ni][2]*k[2] + q_reg[ni][3]*k[3]) * SCALE);
                float P_ij = fast_expf(S_ij - L_reg[ni]);
                float dP_ij = warp_reduce_sum(
                    do_reg[ni][0]*v[0] + do_reg[ni][1]*v[1] + do_reg[ni][2]*v[2] + do_reg[ni][3]*v[3]);
                float dS_ij = P_ij * (dP_ij - D_reg[ni]);

                #pragma unroll
                for (int d = 0; d < 4; d++)
                    dQ_acc[ni][d] += dS_ij * k[d] * SCALE;
            }
        }
        __syncthreads();
    }

    // Store dQ
    __nv_bfloat16* dQ_bh = dQ + (int64_t)bh * S * D;

    #pragma unroll
    for (int ni = 0; ni < NI; ni++) {
        int ig = i_start + i_warp + ni;
        if (ig >= S) continue;
        #pragma unroll
        for (int d = 0; d < 4; d++)
            dQ_bh[(int64_t)ig * D + d_base + d] = __float2bfloat16(dQ_acc[ni][d]);
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

    // dV/dK: BK=8 KV rows per CTA
    {
        dim3 grid(BH, (int)((S + 7) / 8));
        dim3 block(128);
        dV_dK_kernel<<<grid, block, 0, stream>>>(
            Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dV_ptr, dK_ptr, (int)S);
    }

    // dQ: BQ=8 Q rows per CTA
    {
        dim3 grid(BH, (int)((S + 7) / 8));
        dim3 block(128);
        dQ_kernel<<<grid, block, 0, stream>>>(
            Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr, (int)S);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128::run);

}  // namespace mha_bwd_d128