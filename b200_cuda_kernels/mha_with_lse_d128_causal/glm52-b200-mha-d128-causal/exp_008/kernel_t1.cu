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
} while(0)

namespace mha_lse_d128 {

constexpr int D = 128;
constexpr int BQ = 64;
constexpr int BK = 64;
constexpr int NUM_WARPS = 4;
constexpr int THREADS = NUM_WARPS * 32;

// Load A matrix (16x16, row-major) from shared memory using ldmatrix.x4
__device__ __forceinline__ void load_A_x4(uint32_t a[4], const __nv_bfloat16* smem_ptr) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
        : "r"(addr));
}

// Load B matrix (16x8, col-major) from row-major shared memory using ldmatrix.x2.trans
__device__ __forceinline__ void load_B_trans_x2(uint32_t b[2], const __nv_bfloat16* smem_ptr) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
        : "=r"(b[0]), "=r"(b[1])
        : "r"(addr));
}

// mma.sync m16n8k16 BF16->FP32
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
    const int warp_q_start = q_start + warp_id * 16;
    constexpr float scale = 0.08838834764831845f; // 1/sqrt(128)

    // Load Q tile to shared memory (64 rows x 128 cols)
    #pragma unroll
    for (int i = threadIdx.x; i < BQ * D / 8; i += THREADS) {
        int row = i / (D / 8);
        int col = (i % (D / 8)) * 8;
        int q_idx = q_start + row;
        if (q_idx < S) {
            reinterpret_cast<uint4*>(Q_smem)[i] =
                reinterpret_cast<const uint4*>(Q_base + (int64_t)q_idx * D)[col / 8];
        } else {
            reinterpret_cast<uint4*>(Q_smem)[i] = make_uint4(0, 0, 0, 0);
        }
    }
    __syncthreads();

    // Preload Q into registers: 8 K-tiles of 16x16, each 4 uint32 regs
    uint32_t q_regs[8][4];
    #pragma unroll
    for (int kt = 0; kt < 8; kt++) {
        int t = lane_id;
        int row = (t % 8) + (t / 16) * 8;
        int col = ((t / 8) % 2) * 8 + kt * 16;
        load_A_x4(q_regs[kt], Q_smem + (warp_id * 16 + row) * D + col);
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
        #pragma unroll
        for (int i = threadIdx.x; i < BK * D / 8; i += THREADS) {
            int row = i / (D / 8);
            int col = (i % (D / 8)) * 8;
            int k_idx = k_start + row;
            if (k_idx < S) {
                reinterpret_cast<uint4*>(K_smem)[i] =
                    reinterpret_cast<const uint4*>(K_base + (int64_t)k_idx * D)[col / 8];
                reinterpret_cast<uint4*>(V_smem)[i] =
                    reinterpret_cast<const uint4*>(V_base + (int64_t)k_idx * D)[col / 8];
            } else {
                reinterpret_cast<uint4*>(K_smem)[i] = make_uint4(0, 0, 0, 0);
                reinterpret_cast<uint4*>(V_smem)[i] = make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        // Compute QK^T: 16x64 score matrix
        // M=16, N=64, K=128 -> 8 N-tiles x 8 K-tiles = 64 MMAs
        float scores[8][4];

        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            scores[nt][0] = 0.0f; scores[nt][1] = 0.0f;
            scores[nt][2] = 0.0f; scores[nt][3] = 0.0f;

            #pragma unroll
            for (int kt = 0; kt < 8; kt++) {
                // Load B = K^T[16, 8] using ldmatrix.x2.trans
                uint32_t b_regs[2];
                int t = lane_id;
                int k_row = nt * 8 + (t % 8);
                int k_col = kt * 16 + (t < 16 ? (t / 8) * 8 : 0);
                load_B_trans_x2(b_regs, K_smem + k_row * D + k_col);

                // MMA: D = A * B + C
                float tmp[4];
                mma_m16n8k16(tmp, q_regs[kt], b_regs, scores[nt]);
                scores[nt][0] = tmp[0]; scores[nt][1] = tmp[1];
                scores[nt][2] = tmp[2]; scores[nt][3] = tmp[3];
            }
        }

        // Apply scale
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            scores[nt][0] *= scale; scores[nt][1] *= scale;
            scores[nt][2] *= scale; scores[nt][3] *= scale;
        }

        // Apply causal mask
        {
            int t = lane_id;
            int lr0 = t / 4;
            int lr1 = t / 4 + 8;
            int gr0 = warp_q_start + lr0;
            int gr1 = warp_q_start + lr1;

            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                int c0 = nt * 8 + 2 * (t % 4);
                int c1 = c0 + 1;
                int k0 = k_start + c0;
                int k1 = k_start + c1;
                if (k0 > gr0 || k0 >= S) scores[nt][0] = -INFINITY;
                if (k1 > gr0 || k1 >= S) scores[nt][1] = -INFINITY;
                if (k0 > gr1 || k0 >= S) scores[nt][2] = -INFINITY;
                if (k1 > gr1 || k1 >= S) scores[nt][3] = -INFINITY;
            }
        }

        // Online softmax: block max
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
        float block_sum[2] = {0.0f, 0.0f};
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            scores[nt][0] = __expf(scores[nt][0] - new_max[0]);
            scores[nt][1] = __expf(scores[nt][1] - new_max[0]);
            scores[nt][2] = __expf(scores[nt][2] - new_max[1]);
            scores[nt][3] = __expf(scores[nt][3] - new_max[1]);
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

        // Convert scores to BF16 and store to P_smem (warp-local)
        {
            int t = lane_id;
            int lr0 = t / 4;
            int lr1 = t / 4 + 8;
            int cb = 2 * (t % 4);

            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                int col = nt * 8 + cb;
                int pbase = warp_id * 16 * BK;
                *reinterpret_cast<uint32_t*>(&P_smem[pbase + lr0 * BK + col]) =
                    pack_bf16_fn(__float_as_uint(scores[nt][0]), __float_as_uint(scores[nt][1]));
                *reinterpret_cast<uint32_t*>(&P_smem[pbase + lr1 * BK + col]) =
                    pack_bf16_fn(__float_as_uint(scores[nt][2]), __float_as_uint(scores[nt][3]));
            }
        }
        __syncwarp();

        // Compute PV: 16x128 output
        // M=16, N=128, K=64 -> 16 N-tiles x 4 K-tiles = 64 MMAs
        #pragma unroll
        for (int nt = 0; nt < 16; nt++) {
            #pragma unroll
            for (int kt = 0; kt < 4; kt++) {
                // Load A = P[16, 16] from P_smem
                uint32_t a_regs[4];
                int t = lane_id;
                int p_row = (t % 8) + (t / 16) * 8;
                int p_col = ((t / 8) % 2) * 8 + kt * 16;
                load_A_x4(a_regs, P_smem + warp_id * 16 * BK + p_row * BK + p_col);

                // Load B = V^T[16, 8] from V_smem using ldmatrix.x2.trans
                uint32_t b_regs[2];
                int v_row = kt * 16 + (t < 16 ? t : 0);
                int v_col = nt * 8;
                load_B_trans_x2(b_regs, V_smem + v_row * D + v_col);

                // MMA: D = A * B + D
                float tmp[4];
                mma_m16n8k16(tmp, a_regs, b_regs, o_acc[nt]);
                o_acc[nt][0] = tmp[0]; o_acc[nt][1] = tmp[1];
                o_acc[nt][2] = tmp[2]; o_acc[nt][3] = tmp[3];
            }
        }

        __syncthreads();
    }

    // Stage output to shared memory for coalesced stores
    __nv_bfloat16* Out_smem = K_smem; // Reuse K_smem space
    {
        int t = lane_id;
        int lr0 = t / 4;
        int lr1 = t / 4 + 8;
        float inv0 = (row_sum[0] > 0.0f) ? (1.0f / row_sum[0]) : 0.0f;
        float inv1 = (row_sum[1] > 0.0f) ? (1.0f / row_sum[1]) : 0.0f;

        #pragma unroll
        for (int nt = 0; nt < 16; nt++) {
            int col = nt * 8 + 2 * (t % 4);
            *reinterpret_cast<uint32_t*>(&Out_smem[(warp_id * 16 + lr0) * D + col]) =
                pack_bf16_fn(__float_as_uint(o_acc[nt][0] * inv0),
                             __float_as_uint(o_acc[nt][1] * inv0));
            *reinterpret_cast<uint32_t*>(&Out_smem[(warp_id * 16 + lr1) * D + col]) =
                pack_bf16_fn(__float_as_uint(o_acc[nt][2] * inv1),
                             __float_as_uint(o_acc[nt][3] * inv1));
        }
    }
    __syncthreads();

    // Coalesced uint4 stores to global memory
    #pragma unroll
    for (int i = threadIdx.x; i < BQ * D / 8; i += THREADS) {
        int row = i / (D / 8);
        int col = (i % (D / 8)) * 8;
        int g_row = q_start + row;
        if (g_row < S) {
            *reinterpret_cast<uint4*>(O_base + (int64_t)g_row * D + col) =
                reinterpret_cast<uint4*>(Out_smem)[i];
        }
    }

    // Write LSE
    {
        int t = lane_id;
        int lr0 = t / 4;
        int lr1 = t / 4 + 8;
        int gr0 = warp_q_start + lr0;
        int gr1 = warp_q_start + lr1;

        if (t % 4 == 0) {
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