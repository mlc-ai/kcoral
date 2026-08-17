#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
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

namespace mha_lse_d128 {

constexpr int D = 128;
constexpr int BQ = 64;
constexpr int BK = 64;
constexpr int NUM_WARPS = 4;
constexpr int THREADS = NUM_WARPS * 32;

__device__ __forceinline__ uint32_t pack_bf16_fn(float a, float b) {
    __nv_bfloat16 ba = __float2bfloat16(a);
    __nv_bfloat16 bb = __float2bfloat16(b);
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};" : "=r"(result) :
        "h"(*reinterpret_cast<uint16_t*>(&ba)),
        "h"(*reinterpret_cast<uint16_t*>(&bb)));
    return result;
}

__device__ __forceinline__ uint32_t pack_2bf16(__nv_bfloat16 a, __nv_bfloat16 b) {
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};" : "=r"(result) :
        "h"(*reinterpret_cast<uint16_t*>(&a)),
        "h"(*reinterpret_cast<uint16_t*>(&b)));
    return result;
}

__device__ __forceinline__ void load_A_x4(uint32_t a[4], const __nv_bfloat16* smem_ptr) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
        : "r"(addr));
}

__device__ __forceinline__ void load_B_trans_x2(uint32_t b[2], const __nv_bfloat16* smem_ptr) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
        : "=r"(b[0]), "=r"(b[1])
        : "r"(addr));
}

__device__ __forceinline__ void mma_m16n8k16(
    float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// Load V fragment for B matrix in PV computation using regular SMEM loads
// B[k_mma][n_mma] = V[k_tile*16 + k_mma, n_tile*8 + n_mma]
// Thread t needs:
// b0: V[kt*16 + (t%4)*2,     nt*8 + t/4], V[kt*16 + (t%4)*2+1,   nt*8 + t/4]
// b1: V[kt*16 + (t%4)*2+8,   nt*8 + t/4], V[kt*16 + (t%4)*2+9,   nt*8 + t/4]
__device__ __forceinline__ void load_V_frag(uint32_t b[2], const __nv_bfloat16* V_smem,
                                             int kt, int nt, int lane_id) {
    int k0 = kt * 16 + (lane_id % 4) * 2;
    int k1 = k0 + 1;
    int k2 = k0 + 8;
    int k3 = k0 + 9;
    int d  = nt * 8 + lane_id / 4;
    b[0] = pack_2bf16(V_smem[k0 * D + d], V_smem[k1 * D + d]);
    b[1] = pack_2bf16(V_smem[k2 * D + d], V_smem[k3 * D + d]);
}

__global__ void mha_causal_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S)
{
    const int bh_idx = blockIdx.x;
    const int b = bh_idx / H;
    const int h = bh_idx % H;
    const int q_block = blockIdx.y;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    extern __shared__ char smem_buf[];
    __nv_bfloat16* Q_smem = reinterpret_cast<__nv_bfloat16*>(smem_buf);
    __nv_bfloat16* K_smem = Q_smem + BQ * D;
    __nv_bfloat16* V_smem = K_smem + BK * D;
    __nv_bfloat16* P_smem = V_smem + BK * D;

    const int64_t bh_offset = (int64_t)(b * H + h) * S * D;
    const __nv_bfloat16* Q_base = Q + bh_offset;
    const __nv_bfloat16* K_base = K + bh_offset;
    const __nv_bfloat16* V_base = V + bh_offset;
    __nv_bfloat16* O_base = O + bh_offset;
    float* LSE_base = LSE + (int64_t)(b * H + h) * S;

    const int q_start = q_block * BQ;
    const int warp_q = q_start + warp_id * 16;
    constexpr float scale = 0.08838834764831845f;

    // Load Q tile to shared memory
    for (int i = threadIdx.x; i < BQ * D / 8; i += THREADS) {
        int row = i / (D / 8);
        int col8 = i % (D / 8);
        int q_idx = q_start + row;
        if (q_idx < S) {
            reinterpret_cast<uint4*>(Q_smem)[i] =
                reinterpret_cast<const uint4*>(Q_base + (int64_t)q_idx * D)[col8];
        } else {
            reinterpret_cast<uint4*>(Q_smem)[i] = make_uint4(0, 0, 0, 0);
        }
    }
    __syncthreads();

    // Preload Q fragments into registers (8 K-tiles of 16x16)
    uint32_t q_frag[8][4];
    #pragma unroll
    for (int kt = 0; kt < 8; kt++) {
        int row = ((lane_id / 8) % 2) * 8 + (lane_id % 8);
        int col = (lane_id / 16) * 8 + kt * 16;
        load_A_x4(q_frag[kt], Q_smem + row * D + col);
    }

    // Output accumulator: 16 N-tiles of 16x8, each 4 FP32 regs
    float o_acc[16][4];
    #pragma unroll
    for (int n = 0; n < 16; n++) {
        o_acc[n][0] = 0.0f; o_acc[n][1] = 0.0f;
        o_acc[n][2] = 0.0f; o_acc[n][3] = 0.0f;
    }

    float row_max[2] = {-INFINITY, -INFINITY};
    float row_sum[2] = {0.0f, 0.0f};

    int max_q = min(q_start + BQ, S);
    int num_k_blocks = (max_q + BK - 1) / BK;

    for (int kb = 0; kb < num_k_blocks; kb++) {
        int k_start = kb * BK;

        // Load K and V to shared memory
        for (int i = threadIdx.x; i < BK * D / 8; i += THREADS) {
            int row = i / (D / 8);
            int col8 = i % (D / 8);
            int k_idx = k_start + row;
            if (k_idx < S) {
                reinterpret_cast<uint4*>(K_smem)[i] =
                    reinterpret_cast<const uint4*>(K_base + (int64_t)k_idx * D)[col8];
                reinterpret_cast<uint4*>(V_smem)[i] =
                    reinterpret_cast<const uint4*>(V_base + (int64_t)k_idx * D)[col8];
            } else {
                reinterpret_cast<uint4*>(K_smem)[i] = make_uint4(0, 0, 0, 0);
                reinterpret_cast<uint4*>(V_smem)[i] = make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        // Compute QK^T: 8 N-tiles x 8 K-tiles = 64 MMAs
        // B = K^T[kt*16:kt*16+16, nt*8:nt*8+8], loaded with ldmatrix.x2.trans
        float scores[8][4];
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            scores[nt][0] = 0.0f; scores[nt][1] = 0.0f;
            scores[nt][2] = 0.0f; scores[nt][3] = 0.0f;

            #pragma unroll
            for (int kt = 0; kt < 8; kt++) {
                uint32_t b_frag[2];
                // K loading: thread t (0-7) loads K[nt*8+t, kt*16:kt*16+8]
                //            thread t (8-15) loads K[nt*8+(t-8), kt*16+8:kt*16+16]
                int k_row, k_col;
                if (lane_id < 8) {
                    k_row = nt * 8 + lane_id;
                    k_col = kt * 16;
                } else if (lane_id < 16) {
                    k_row = nt * 8 + (lane_id - 8);
                    k_col = kt * 16 + 8;
                } else {
                    k_row = 0;
                    k_col = 0;
                }
                load_B_trans_x2(b_frag, K_smem + k_row * D + k_col);
                mma_m16n8k16(scores[nt], q_frag[kt], b_frag, scores[nt]);
            }
        }

        // Apply scale
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            scores[nt][0] *= scale; scores[nt][1] *= scale;
            scores[nt][2] *= scale; scores[nt][3] *= scale;
        }

        // Apply causal mask
        // Thread t: scores[nt][0/1] -> row t/4, col nt*8+(t%4)*2 / +1
        //           scores[nt][2/3] -> row t/4+8, col nt*8+(t%4)*2 / +1
        {
            int lr0 = lane_id / 4;
            int lr1 = lane_id / 4 + 8;
            int gr0 = warp_q + lr0;
            int gr1 = warp_q + lr1;
            int lc = (lane_id % 4) * 2;

            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                int gc0 = k_start + nt * 8 + lc;
                int gc1 = k_start + nt * 8 + lc + 1;
                if (gc0 > gr0 || gc0 >= S) scores[nt][0] = -INFINITY;
                if (gc1 > gr0 || gc1 >= S) scores[nt][1] = -INFINITY;
                if (gc0 > gr1 || gc0 >= S) scores[nt][2] = -INFINITY;
                if (gc1 > gr1 || gc1 >= S) scores[nt][3] = -INFINITY;
            }
        }

        // Online softmax: block max (reduce across t%4 within same row)
        float block_max[2] = {-INFINITY, -INFINITY};
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            block_max[0] = fmaxf(block_max[0], fmaxf(scores[nt][0], scores[nt][1]));
            block_max[1] = fmaxf(block_max[1], fmaxf(scores[nt][2], scores[nt][3]));
        }
        block_max[0] = fmaxf(block_max[0], __shfl_xor_sync(0xFFFFFFFF, block_max[0], 1));
        block_max[0] = fmaxf(block_max[0], __shfl_xor_sync(0xFFFFFFFF, block_max[0], 2));
        block_max[1] = fmaxf(block_max[1], __shfl_xor_sync(0xFFFFFFFF, block_max[1], 1));
        block_max[1] = fmaxf(block_max[1], __shfl_xor_sync(0xFFFFFFFF, block_max[1], 2));

        float new_max[2], exp_diff[2];
        new_max[0] = fmaxf(row_max[0], block_max[0]);
        new_max[1] = fmaxf(row_max[1], block_max[1]);
        exp_diff[0] = (row_max[0] > -INFINITY) ? __expf(row_max[0] - new_max[0]) : 0.0f;
        exp_diff[1] = (row_max[1] > -INFINITY) ? __expf(row_max[1] - new_max[1]) : 0.0f;

        // Rescale O accumulator
        #pragma unroll
        for (int n = 0; n < 16; n++) {
            o_acc[n][0] *= exp_diff[0]; o_acc[n][1] *= exp_diff[0];
            o_acc[n][2] *= exp_diff[1]; o_acc[n][3] *= exp_diff[1];
        }

        // Compute exp and block sum
        float nm0 = (new_max[0] > -INFINITY) ? new_max[0] : 0.0f;
        float nm1 = (new_max[1] > -INFINITY) ? new_max[1] : 0.0f;
        float block_sum[2] = {0.0f, 0.0f};
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            scores[nt][0] = __expf(scores[nt][0] - nm0);
            scores[nt][1] = __expf(scores[nt][1] - nm0);
            scores[nt][2] = __expf(scores[nt][2] - nm1);
            scores[nt][3] = __expf(scores[nt][3] - nm1);
            block_sum[0] += scores[nt][0] + scores[nt][1];
            block_sum[1] += scores[nt][2] + scores[nt][3];
        }
        block_sum[0] += __shfl_xor_sync(0xFFFFFFFF, block_sum[0], 1);
        block_sum[0] += __shfl_xor_sync(0xFFFFFFFF, block_sum[0], 2);
        block_sum[1] += __shfl_xor_sync(0xFFFFFFFF, block_sum[1], 1);
        block_sum[1] += __shfl_xor_sync(0xFFFFFFFF, block_sum[1], 2);

        row_sum[0] = row_sum[0] * exp_diff[0] + block_sum[0];
        row_sum[1] = row_sum[1] * exp_diff[1] + block_sum[1];
        row_max[0] = new_max[0];
        row_max[1] = new_max[1];

        // Store P to shared memory (warp-local)
        {
            int lr0 = lane_id / 4;
            int lr1 = lane_id / 4 + 8;
            int lc = (lane_id % 4) * 2;
            int pbase = warp_id * 16 * BK;

            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                int col = nt * 8 + lc;
                *reinterpret_cast<uint32_t*>(&P_smem[pbase + lr0 * BK + col]) =
                    pack_bf16_fn(scores[nt][0], scores[nt][1]);
                *reinterpret_cast<uint32_t*>(&P_smem[pbase + lr1 * BK + col]) =
                    pack_bf16_fn(scores[nt][2], scores[nt][3]);
            }
        }
        __syncwarp();

        // Compute PV: 16 N-tiles x 4 K-tiles = 64 MMAs
        // A = P[:, kt*16:kt*16+16] loaded with ldmatrix.x4
        // B = V[kt*16:kt*16+16, nt*8:nt*8+8] loaded with regular SMEM loads
        #pragma unroll
        for (int nt = 0; nt < 16; nt++) {
            #pragma unroll
            for (int kt = 0; kt < 4; kt++) {
                // Load A = P[16, 16] using ldmatrix.x4
                uint32_t a_frag[4];
                int row = ((lane_id / 8) % 2) * 8 + (lane_id % 8);
                int col = (lane_id / 16) * 8 + kt * 16;
                load_A_x4(a_frag, P_smem + warp_id * 16 * BK + row * BK + col);

                // Load B = V[16, 8] using regular loads
                uint32_t b_frag[2];
                load_V_frag(b_frag, V_smem, kt, nt, lane_id);

                mma_m16n8k16(o_acc[nt], a_frag, b_frag, o_acc[nt]);
            }
        }

        __syncthreads();
    }

    // Stage output to shared memory for coalesced stores
    __nv_bfloat16* Out_smem = K_smem;
    {
        int lr0 = lane_id / 4;
        int lr1 = lane_id / 4 + 8;
        int lc = (lane_id % 4) * 2;
        float inv0 = (row_sum[0] > 0.0f) ? (1.0f / row_sum[0]) : 0.0f;
        float inv1 = (row_sum[1] > 0.0f) ? (1.0f / row_sum[1]) : 0.0f;

        #pragma unroll
        for (int nt = 0; nt < 16; nt++) {
            int col = nt * 8 + lc;
            *reinterpret_cast<uint32_t*>(&Out_smem[(warp_id * 16 + lr0) * D + col]) =
                pack_bf16_fn(o_acc[nt][0] * inv0, o_acc[nt][1] * inv0);
            *reinterpret_cast<uint32_t*>(&Out_smem[(warp_id * 16 + lr1) * D + col]) =
                pack_bf16_fn(o_acc[nt][2] * inv1, o_acc[nt][3] * inv1);
        }
    }
    __syncthreads();

    // Coalesced uint4 stores to global memory
    for (int i = threadIdx.x; i < BQ * D / 8; i += THREADS) {
        int row = i / (D / 8);
        int col8 = i % (D / 8);
        int g_row = q_start + row;
        if (g_row < S) {
            *reinterpret_cast<uint4*>(O_base + (int64_t)g_row * D + col8 * 8) =
                reinterpret_cast<uint4*>(Out_smem)[i];
        }
    }

    // Write LSE
    {
        if (lane_id % 4 == 0) {
            int gr0 = warp_q + lane_id / 4;
            int gr1 = warp_q + lane_id / 4 + 8;
            if (gr0 < S) {
                LSE_base[gr0] = (row_sum[0] > 0.0f) ? (row_max[0] + logf(row_sum[0])) : -INFINITY;
            }
            if (gr1 < S) {
                LSE_base[gr1] = (row_sum[1] > 0.0f) ? (row_max[1] + logf(row_sum[1])) : -INFINITY;
            }
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    const int B = 4;
    const int H = 48;
    const int S = static_cast<int>(Q.size(2));

    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());

    dim3 grid(B * H, (S + BQ - 1) / BQ);
    dim3 block(THREADS);

    int smem_size = (BQ * D + BK * D + BK * D + NUM_WARPS * 16 * BK) * sizeof(__nv_bfloat16);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(mha_causal_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    mha_causal_kernel<<<grid, block, smem_size, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data, B, H, S);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_lse_d128::run);

}  // namespace mha_lse_d128