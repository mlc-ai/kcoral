#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <cstdio>
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

namespace mha_lse_d128_causal {

static constexpr int D = 128;
static constexpr int BM = 128;
static constexpr int BN = 128;
static constexpr float SCALE = 0.0883883476483184f;
static constexpr int NUM_WARPS = 8;
static constexpr int THREADS = NUM_WARPS * 32;

__device__ __forceinline__ void ldmatrix_x4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t smem_addr) {
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(smem_addr));
}

__device__ __forceinline__ void ldmatrix_x2_trans(
    uint32_t& r0, uint32_t& r1, uint32_t smem_addr) {
    asm volatile(
        "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(r0), "=r"(r1)
        : "r"(smem_addr));
}

__device__ __forceinline__ void ldmatrix_x2(
    uint32_t& r0, uint32_t& r1, uint32_t smem_addr) {
    asm volatile(
        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(r0), "=r"(r1)
        : "r"(smem_addr));
}

__device__ __forceinline__ void mma_m16n8k16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

__global__ void attention_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S) {

    int b = blockIdx.y;
    int h = blockIdx.z;
    int q_block = blockIdx.x;
    int q_start = q_block * BM;

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int q_row_base = warp_id * 16;

    int64_t bh_offset = ((int64_t)b * H + h) * (int64_t)S * D;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* smem_Q = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* smem_K = smem_Q + BM * D;
    __nv_bfloat16* smem_V = smem_K + BN * D;
    __nv_bfloat16* smem_P = smem_V + BN * D;

    {
        int row = threadIdx.x;
        if (row < BM) {
            int g_row = q_start + row;
            if (g_row < S) {
                for (int d = 0; d < D; d += 8) {
                    *reinterpret_cast<int4*>(&smem_Q[row * D + d]) =
                        *reinterpret_cast<const int4*>(&Q[bh_offset + (int64_t)g_row * D + d]);
                }
            } else {
                for (int d = 0; d < D; d += 8)
                    *reinterpret_cast<int4*>(&smem_Q[row * D + d]) = make_int4(0, 0, 0, 0);
            }
        }
    }
    __syncthreads();

    float o_acc[64];
    #pragma unroll
    for (int i = 0; i < 64; i++) o_acc[i] = 0.0f;

    float m_val[2] = {-INFINITY, -INFINITY};
    float l_val[2] = {0.0f, 0.0f};

    int g_row0 = q_start + q_row_base + lane_id / 4;
    int g_row1 = q_start + q_row_base + lane_id / 4 + 8;
    bool valid0 = (g_row0 < S);
    bool valid1 = (g_row1 < S);

    int k_limit = min(q_start + BM, S);

    for (int k_start = 0; k_start < k_limit; k_start += BN) {
        {
            int row = threadIdx.x;
            if (row < BN) {
                int g_key = k_start + row;
                if (g_key < S) {
                    for (int d = 0; d < D; d += 8) {
                        *reinterpret_cast<int4*>(&smem_K[row * D + d]) =
                            *reinterpret_cast<const int4*>(&K[bh_offset + (int64_t)g_key * D + d]);
                        *reinterpret_cast<int4*>(&smem_V[row * D + d]) =
                            *reinterpret_cast<const int4*>(&V[bh_offset + (int64_t)g_key * D + d]);
                    }
                } else {
                    for (int d = 0; d < D; d += 8) {
                        *reinterpret_cast<int4*>(&smem_K[row * D + d]) = make_int4(0, 0, 0, 0);
                        *reinterpret_cast<int4*>(&smem_V[row * D + d]) = make_int4(0, 0, 0, 0);
                    }
                }
            }
        }
        __syncthreads();

        float s_frag[64];
        #pragma unroll
        for (int i = 0; i < 64; i++) s_frag[i] = 0.0f;

        #pragma unroll
        for (int k_tile = 0; k_tile < 8; k_tile++) {
            int d_base = k_tile * 16;
            uint32_t a[4];
            uint32_t q_addr;
            if (lane_id < 8) {
                q_addr = __cvta_generic_to_shared(&smem_Q[(q_row_base + lane_id) * D + d_base]);
            } else if (lane_id < 16) {
                q_addr = __cvta_generic_to_shared(&smem_Q[(q_row_base + lane_id - 8) * D + d_base + 8]);
            } else if (lane_id < 24) {
                q_addr = __cvta_generic_to_shared(&smem_Q[(q_row_base + lane_id - 8) * D + d_base]);
            } else {
                q_addr = __cvta_generic_to_shared(&smem_Q[(q_row_base + lane_id - 16) * D + d_base + 8]);
            }
            ldmatrix_x4(a[0], a[1], a[2], a[3], q_addr);

            #pragma unroll
            for (int n_tile = 0; n_tile < 16; n_tile++) {
                int n_base = n_tile * 8;
                uint32_t b[2];
                uint32_t k_addr;
                if (lane_id < 8) {
                    k_addr = __cvta_generic_to_shared(&smem_K[(n_base + lane_id) * D + d_base]);
                } else if (lane_id < 16) {
                    k_addr = __cvta_generic_to_shared(&smem_K[(n_base + lane_id - 8) * D + d_base + 8]);
                } else {
                    k_addr = __cvta_generic_to_shared(&smem_K[0]);
                }
                ldmatrix_x2_trans(b[0], b[1], k_addr);

                int idx = n_tile * 4;
                mma_m16n8k16(
                    s_frag[idx], s_frag[idx+1], s_frag[idx+2], s_frag[idx+3],
                    a[0], a[1], a[2], a[3], b[0], b[1],
                    s_frag[idx], s_frag[idx+1], s_frag[idx+2], s_frag[idx+3]);
            }
        }

        #pragma unroll
        for (int i = 0; i < 64; i++) s_frag[i] *= SCALE;

        #pragma unroll
        for (int n_tile = 0; n_tile < 16; n_tile++) {
            int key_base = k_start + n_tile * 8;
            int col0 = key_base + (lane_id % 4) * 2;
            int col1 = col0 + 1;
            int idx = n_tile * 4;
            if (!valid0 || col0 > g_row0) s_frag[idx] = -INFINITY;
            if (!valid0 || col1 > g_row0) s_frag[idx+1] = -INFINITY;
            if (!valid1 || col0 > g_row1) s_frag[idx+2] = -INFINITY;
            if (!valid1 || col1 > g_row1) s_frag[idx+3] = -INFINITY;
        }

        float m_old0 = m_val[0], m_old1 = m_val[1];
        float m_new0 = m_old0, m_new1 = m_old1;
        #pragma unroll
        for (int n_tile = 0; n_tile < 16; n_tile++) {
            int idx = n_tile * 4;
            m_new0 = fmaxf(m_new0, fmaxf(s_frag[idx], s_frag[idx+1]));
            m_new1 = fmaxf(m_new1, fmaxf(s_frag[idx+2], s_frag[idx+3]));
        }
        m_new0 = fmaxf(m_new0, __shfl_xor_sync(0xffffffff, m_new0, 1));
        m_new0 = fmaxf(m_new0, __shfl_xor_sync(0xffffffff, m_new0, 2));
        m_new0 = fmaxf(m_new0, __shfl_xor_sync(0xffffffff, m_new0, 4));
        m_new1 = fmaxf(m_new1, __shfl_xor_sync(0xffffffff, m_new1, 1));
        m_new1 = fmaxf(m_new1, __shfl_xor_sync(0xffffffff, m_new1, 2));
        m_new1 = fmaxf(m_new1, __shfl_xor_sync(0xffffffff, m_new1, 4));

        float alpha0, alpha1;
        if (m_new0 > -INFINITY) {
            alpha0 = (m_old0 == -INFINITY) ? 0.0f : expf(m_old0 - m_new0);
        } else {
            alpha0 = 1.0f;
        }
        if (m_new1 > -INFINITY) {
            alpha1 = (m_old1 == -INFINITY) ? 0.0f : expf(m_old1 - m_new1);
        } else {
            alpha1 = 1.0f;
        }

        float l_new0 = l_val[0] * alpha0;
        float l_new1 = l_val[1] * alpha1;
        #pragma unroll
        for (int n_tile = 0; n_tile < 16; n_tile++) {
            int idx = n_tile * 4;
            if (m_new0 > -INFINITY) {
                s_frag[idx]   = expf(s_frag[idx]   - m_new0);
                s_frag[idx+1] = expf(s_frag[idx+1] - m_new0);
                l_new0 += s_frag[idx] + s_frag[idx+1];
            } else {
                s_frag[idx]   = 0.0f;
                s_frag[idx+1] = 0.0f;
            }
            if (m_new1 > -INFINITY) {
                s_frag[idx+2] = expf(s_frag[idx+2] - m_new1);
                s_frag[idx+3] = expf(s_frag[idx+3] - m_new1);
                l_new1 += s_frag[idx+2] + s_frag[idx+3];
            } else {
                s_frag[idx+2] = 0.0f;
                s_frag[idx+3] = 0.0f;
            }
        }
        l_new0 += __shfl_xor_sync(0xffffffff, l_new0, 1);
        l_new0 += __shfl_xor_sync(0xffffffff, l_new0, 2);
        l_new0 += __shfl_xor_sync(0xffffffff, l_new0, 4);
        l_new1 += __shfl_xor_sync(0xffffffff, l_new1, 1);
        l_new1 += __shfl_xor_sync(0xffffffff, l_new1, 2);
        l_new1 += __shfl_xor_sync(0xffffffff, l_new1, 4);

        #pragma unroll
        for (int n_tile = 0; n_tile < 16; n_tile++) {
            int idx = n_tile * 4;
            o_acc[idx]   *= alpha0;
            o_acc[idx+1] *= alpha0;
            o_acc[idx+2] *= alpha1;
            o_acc[idx+3] *= alpha1;
        }

        int p_row0 = q_row_base + lane_id / 4;
        int p_row1 = q_row_base + lane_id / 4 + 8;
        #pragma unroll
        for (int n_tile = 0; n_tile < 16; n_tile++) {
            int col = n_tile * 8 + (lane_id % 4) * 2;
            int idx = n_tile * 4;
            __nv_bfloat162 p0 = __float22bfloat162_rn(make_float2(s_frag[idx], s_frag[idx+1]));
            __nv_bfloat162 p1 = __float22bfloat162_rn(make_float2(s_frag[idx+2], s_frag[idx+3]));
            *reinterpret_cast<__nv_bfloat162*>(&smem_P[p_row0 * BN + col]) = p0;
            *reinterpret_cast<__nv_bfloat162*>(&smem_P[p_row1 * BN + col]) = p1;
        }
        __syncwarp();

        #pragma unroll
        for (int k_tile = 0; k_tile < 8; k_tile++) {
            int k_base = k_tile * 16;
            uint32_t a[4];
            uint32_t p_addr;
            if (lane_id < 8) {
                p_addr = __cvta_generic_to_shared(&smem_P[(q_row_base + lane_id) * BN + k_base]);
            } else if (lane_id < 16) {
                p_addr = __cvta_generic_to_shared(&smem_P[(q_row_base + lane_id - 8) * BN + k_base + 8]);
            } else if (lane_id < 24) {
                p_addr = __cvta_generic_to_shared(&smem_P[(q_row_base + lane_id - 8) * BN + k_base]);
            } else {
                p_addr = __cvta_generic_to_shared(&smem_P[(q_row_base + lane_id - 16) * BN + k_base + 8]);
            }
            ldmatrix_x4(a[0], a[1], a[2], a[3], p_addr);

            #pragma unroll
            for (int n_tile = 0; n_tile < 16; n_tile++) {
                int n_base = n_tile * 8;
                uint32_t b[2];
                uint32_t v_addr;
                if (lane_id < 16) {
                    v_addr = __cvta_generic_to_shared(&smem_V[(k_base + lane_id) * D + n_base]);
                } else {
                    v_addr = __cvta_generic_to_shared(&smem_V[0]);
                }
                ldmatrix_x2(b[0], b[1], v_addr);

                int idx = n_tile * 4;
                mma_m16n8k16(
                    o_acc[idx], o_acc[idx+1], o_acc[idx+2], o_acc[idx+3],
                    a[0], a[1], a[2], a[3], b[0], b[1],
                    o_acc[idx], o_acc[idx+1], o_acc[idx+2], o_acc[idx+3]);
            }
        }

        m_val[0] = m_new0; m_val[1] = m_new1;
        l_val[0] = l_new0; l_val[1] = l_new1;

        __syncthreads();
    }

    if (valid0 && l_val[0] > 0.0f) {
        float inv_l0 = 1.0f / l_val[0];
        #pragma unroll
        for (int n_tile = 0; n_tile < 16; n_tile++) {
            int col = n_tile * 8 + (lane_id % 4) * 2;
            int idx = n_tile * 4;
            __nv_bfloat162 o0 = __float22bfloat162_rn(
                make_float2(o_acc[idx] * inv_l0, o_acc[idx+1] * inv_l0));
            *reinterpret_cast<__nv_bfloat162*>(&O[bh_offset + (int64_t)g_row0 * D + col]) = o0;
        }
        LSE[(int64_t)b * H * S + (int64_t)h * S + g_row0] = m_val[0] + logf(l_val[0]);
    } else if (valid0) {
        #pragma unroll
        for (int n_tile = 0; n_tile < 16; n_tile++) {
            int col = n_tile * 8 + (lane_id % 4) * 2;
            __nv_bfloat162 o0 = __float22bfloat162_rn(make_float2(0.0f, 0.0f));
            *reinterpret_cast<__nv_bfloat162*>(&O[bh_offset + (int64_t)g_row0 * D + col]) = o0;
        }
        LSE[(int64_t)b * H * S + (int64_t)h * S + g_row0] = -INFINITY;
    }

    if (valid1 && l_val[1] > 0.0f) {
        float inv_l1 = 1.0f / l_val[1];
        #pragma unroll
        for (int n_tile = 0; n_tile < 16; n_tile++) {
            int col = n_tile * 8 + (lane_id % 4) * 2;
            int idx = n_tile * 4;
            __nv_bfloat162 o1 = __float22bfloat162_rn(
                make_float2(o_acc[idx+2] * inv_l1, o_acc[idx+3] * inv_l1));
            *reinterpret_cast<__nv_bfloat162*>(&O[bh_offset + (int64_t)g_row1 * D + col]) = o1;
        }
        LSE[(int64_t)b * H * S + (int64_t)h * S + g_row1] = m_val[1] + logf(l_val[1]);
    } else if (valid1) {
        #pragma unroll
        for (int n_tile = 0; n_tile < 16; n_tile++) {
            int col = n_tile * 8 + (lane_id % 4) * 2;
            __nv_bfloat162 o1 = __float22bfloat162_rn(make_float2(0.0f, 0.0f));
            *reinterpret_cast<__nv_bfloat162*>(&O[bh_offset + (int64_t)g_row1 * D + col]) = o1;
        }
        LSE[(int64_t)b * H * S + (int64_t)h * S + g_row1] = -INFINITY;
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    const int B = 4;
    const int H = 48;
    const int D_val = 128;
    int64_t S = Q.size(2);

    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());

    int num_q_blocks = ((int)S + BM - 1) / BM;
    dim3 grid(num_q_blocks, B, H);
    dim3 block(THREADS);

    int smem_size = (BM * D_val + 2 * BN * D_val + BM * BN) * (int)sizeof(__nv_bfloat16);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    cudaFuncSetAttribute(attention_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    attention_kernel<<<grid, block, smem_size, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data, B, H, (int)S);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_lse_d128_causal::run);

}  // namespace mha_lse_d128_causal