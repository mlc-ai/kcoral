#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda;

#define CUDA_CHECK(call) do {                                       \
    cudaError_t _e = (call);                                        \
    if (_e != cudaSuccess) {                                        \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                 \
                cudaGetErrorString(_e), __FILE__, __LINE__);        \
        exit(1);                                                    \
    }                                                               \
} while (0)

namespace flash_attn {

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int D  = 128;
constexpr int THREADS = 256;
constexpr int WM = 16, WN = 16, WK = 16;

__device__ __forceinline__ void cp_async_16(void* smem_dst, const void* gmem_src) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem_src));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::: "memory");
}

__device__ __forceinline__ uint32_t pack_bf16x2(float a, float b) {
    __nv_bfloat16 ba = __float2bfloat16(a);
    __nv_bfloat16 bb = __float2bfloat16(b);
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};" : "=r"(result) :
        "h"(*reinterpret_cast<uint16_t*>(&ba)),
        "h"(*reinterpret_cast<uint16_t*>(&bb)));
    return result;
}

__global__ void flash_attn_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S)
{
    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;
    int q_block = blockIdx.y;
    int q_start = q_block * BM;

    if (q_start >= S) return;
    int q_end = min(q_start + BM, S);

    const __nv_bfloat16* Q_bh = Q + (size_t)(b * H + h) * S * D;
    const __nv_bfloat16* K_bh = K + (size_t)(b * H + h) * S * D;
    const __nv_bfloat16* V_bh = V + (size_t)(b * H + h) * S * D;
    __nv_bfloat16* O_bh = O + (size_t)(b * H + h) * S * D;
    float* LSE_bh = LSE + (size_t)(b * H + h) * S;

    extern __shared__ char smem[];
    __nv_bfloat16* smem_q  = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* smem_k  = smem_q + BM * D;
    __nv_bfloat16* smem_v  = smem_k + BN * D;
    __nv_bfloat16* smem_p  = smem_v + BN * D;
    float* smem_m          = reinterpret_cast<float*>(smem_p + BM * BN);
    float* smem_l          = smem_m + BM;
    float* smem_rescale    = smem_l + BM;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;
    const float scale = 0.08838834764831845f;

    // Load Q and K0
    for (int i = tid * 8; i < BM * D; i += THREADS * 8) {
        int row = i / D, col = i % D;
        if (q_start + row < S)
            cp_async_16(smem_q + i, Q_bh + (q_start + row) * D + col);
        else
            *reinterpret_cast<uint4*>(smem_q + i) = make_uint4(0, 0, 0, 0);
    }
    int kv0_len = min(BN, S);
    for (int i = tid * 8; i < BN * D; i += THREADS * 8) {
        int row = i / D, col = i % D;
        int load_row = min(row, kv0_len - 1);
        cp_async_16(smem_k + i, K_bh + load_row * D + col);
    }
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    if (tid < BM) {
        smem_m[tid] = -INFINITY;
        smem_l[tid] = 0.0f;
    }

    wmma::fragment<wmma::accumulator, WM, WN, WK, float> o_frag[D / WN];
    #pragma unroll
    for (int ni = 0; ni < D / WN; ni++)
        wmma::fill_fragment(o_frag[ni], 0.0f);

    int num_tiles = (q_end + BN - 1) / BN;

    for (int kv_idx = 0; kv_idx < num_tiles; kv_idx++) {
        int kv_start = kv_idx * BN;
        int kv_len = min(kv_start + BN, S) - kv_start;
        bool need_mask = (kv_start + BN > q_start);

        // Issue V load (overlaps with QK^T + softmax)
        for (int i = tid * 8; i < BN * D; i += THREADS * 8) {
            int row = i / D, col = i % D;
            int load_row = min(row, kv_len - 1);
            cp_async_16(smem_v + i, V_bh + (kv_start + load_row) * D + col);
        }
        cp_async_commit();

        // QK^T matmul
        wmma::fragment<wmma::accumulator, WM, WN, WK, float> s_frag[BN / WN];
        wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::col_major> b_frag;

        #pragma unroll
        for (int ni = 0; ni < BN / WN; ni++)
            wmma::fill_fragment(s_frag[ni], 0.0f);

        {
            int m_row = warp_id * WM;
            #pragma unroll
            for (int k = 0; k < D / WK; k++) {
                wmma::load_matrix_sync(a_frag, smem_q + m_row * D + k * WK, D);
                #pragma unroll
                for (int ni = 0; ni < BN / WN; ni++) {
                    wmma::load_matrix_sync(b_frag, smem_k + ni * WN * D + k * WK, D);
                    wmma::mma_sync(s_frag[ni], a_frag, b_frag, s_frag[ni]);
                }
            }
        }

        // Direct-from-fragment softmax: no smem_s store/load needed
        {
            int m_row = warp_id * WM;
            int r0 = lane / 4;         // 0-7
            int r1 = lane / 4 + 8;     // 8-15
            int q_row0 = q_start + m_row + r0;
            int q_row1 = q_start + m_row + r1;
            int c_base = (lane % 4) * 2;

            float max_r0 = -INFINITY, max_r1 = -INFINITY;

            // Scale, mask, find max
            #pragma unroll
            for (int ni = 0; ni < BN / WN; ni++) {
                int n_base = ni * WN;
                float s0 = s_frag[ni].x[0] * scale;
                float s1 = s_frag[ni].x[1] * scale;
                float s2 = s_frag[ni].x[2] * scale;
                float s3 = s_frag[ni].x[3] * scale;
                float s4 = s_frag[ni].x[4] * scale;
                float s5 = s_frag[ni].x[5] * scale;
                float s6 = s_frag[ni].x[6] * scale;
                float s7 = s_frag[ni].x[7] * scale;

                int k0 = kv_start + n_base + c_base;
                int k1 = k0 + 1;
                int k2 = k0 + 8;
                int k3 = k0 + 9;
                if (need_mask) {
                    if (k0 > q_row0) s0 = -INFINITY;
                    if (k1 > q_row0) s1 = -INFINITY;
                    if (k2 > q_row0) s4 = -INFINITY;
                    if (k3 > q_row0) s5 = -INFINITY;
                    if (k0 > q_row1) s2 = -INFINITY;
                    if (k1 > q_row1) s3 = -INFINITY;
                    if (k2 > q_row1) s6 = -INFINITY;
                    if (k3 > q_row1) s7 = -INFINITY;
                }
                if (k0 >= S) { s0 = -INFINITY; s2 = -INFINITY; }
                if (k1 >= S) { s1 = -INFINITY; s3 = -INFINITY; }
                if (k2 >= S) { s4 = -INFINITY; s6 = -INFINITY; }
                if (k3 >= S) { s5 = -INFINITY; s7 = -INFINITY; }

                max_r0 = fmaxf(max_r0, fmaxf(s0, fmaxf(s1, fmaxf(s4, s5))));
                max_r1 = fmaxf(max_r1, fmaxf(s2, fmaxf(s3, fmaxf(s6, s7))));

                s_frag[ni].x[0] = s0; s_frag[ni].x[1] = s1;
                s_frag[ni].x[2] = s2; s_frag[ni].x[3] = s3;
                s_frag[ni].x[4] = s4; s_frag[ni].x[5] = s5;
                s_frag[ni].x[6] = s6; s_frag[ni].x[7] = s7;
            }

            // Warp reduce max among 4 lanes sharing same rows
            max_r0 = fmaxf(max_r0, __shfl_xor_sync(0xffffffff, max_r0, 1));
            max_r0 = fmaxf(max_r0, __shfl_xor_sync(0xffffffff, max_r0, 2));
            max_r1 = fmaxf(max_r1, __shfl_xor_sync(0xffffffff, max_r1, 1));
            max_r1 = fmaxf(max_r1, __shfl_xor_sync(0xffffffff, max_r1, 2));

            float m_old_r0 = smem_m[m_row + r0];
            float m_old_r1 = smem_m[m_row + r1];
            float m_new_r0 = fmaxf(m_old_r0, max_r0);
            float m_new_r1 = fmaxf(m_old_r1, max_r1);
            float rescale_r0 = (m_old_r0 == -INFINITY) ? 0.0f : __expf(m_old_r0 - m_new_r0);
            float rescale_r1 = (m_old_r1 == -INFINITY) ? 0.0f : __expf(m_old_r1 - m_new_r1);

            // Compute P, store to smem_p, compute row sum
            float sum_r0 = 0.0f, sum_r1 = 0.0f;
            #pragma unroll
            for (int ni = 0; ni < BN / WN; ni++) {
                int n_base = ni * WN;
                int col0 = n_base + c_base;
                int col2 = col0 + 8;

                float p0 = (s_frag[ni].x[0] == -INFINITY) ? 0.0f : __expf(s_frag[ni].x[0] - m_new_r0);
                float p1 = (s_frag[ni].x[1] == -INFINITY) ? 0.0f : __expf(s_frag[ni].x[1] - m_new_r0);
                float p4 = (s_frag[ni].x[4] == -INFINITY) ? 0.0f : __expf(s_frag[ni].x[4] - m_new_r0);
                float p5 = (s_frag[ni].x[5] == -INFINITY) ? 0.0f : __expf(s_frag[ni].x[5] - m_new_r0);
                float p2 = (s_frag[ni].x[2] == -INFINITY) ? 0.0f : __expf(s_frag[ni].x[2] - m_new_r1);
                float p3 = (s_frag[ni].x[3] == -INFINITY) ? 0.0f : __expf(s_frag[ni].x[3] - m_new_r1);
                float p6 = (s_frag[ni].x[6] == -INFINITY) ? 0.0f : __expf(s_frag[ni].x[6] - m_new_r1);
                float p7 = (s_frag[ni].x[7] == -INFINITY) ? 0.0f : __expf(s_frag[ni].x[7] - m_new_r1);

                sum_r0 += p0 + p1 + p4 + p5;
                sum_r1 += p2 + p3 + p6 + p7;

                // Pack and store P as bf16x2
                int row0 = m_row + r0;
                int row1 = m_row + r1;
                *reinterpret_cast<uint32_t*>(&smem_p[row0 * BN + col0]) = pack_bf16x2(p0, p1);
                *reinterpret_cast<uint32_t*>(&smem_p[row0 * BN + col2]) = pack_bf16x2(p4, p5);
                *reinterpret_cast<uint32_t*>(&smem_p[row1 * BN + col0]) = pack_bf16x2(p2, p3);
                *reinterpret_cast<uint32_t*>(&smem_p[row1 * BN + col2]) = pack_bf16x2(p6, p7);
            }

            // Warp reduce sum
            sum_r0 += __shfl_xor_sync(0xffffffff, sum_r0, 1);
            sum_r0 += __shfl_xor_sync(0xffffffff, sum_r0, 2);
            sum_r1 += __shfl_xor_sync(0xffffffff, sum_r1, 1);
            sum_r1 += __shfl_xor_sync(0xffffffff, sum_r1, 2);

            // Update stats
            smem_m[m_row + r0] = m_new_r0;
            smem_m[m_row + r1] = m_new_r1;
            smem_l[m_row + r0] = smem_l[m_row + r0] * rescale_r0 + sum_r0;
            smem_l[m_row + r1] = smem_l[m_row + r1] * rescale_r1 + sum_r1;
            smem_rescale[m_row + r0] = rescale_r0;
            smem_rescale[m_row + r1] = rescale_r1;
        }

        // O rescale directly in registers (no smem roundtrip)
        {
            int row0 = warp_id * WM + lane / 4;
            int row1 = warp_id * WM + lane / 4 + 8;
            float r0 = smem_rescale[row0];
            float r1 = smem_rescale[row1];
            #pragma unroll
            for (int ni = 0; ni < D / WN; ni++) {
                o_frag[ni].x[0] *= r0; o_frag[ni].x[1] *= r0;
                o_frag[ni].x[2] *= r1; o_frag[ni].x[3] *= r1;
                o_frag[ni].x[4] *= r0; o_frag[ni].x[5] *= r0;
                o_frag[ni].x[6] *= r1; o_frag[ni].x[7] *= r1;
            }
        }

        // Wait for V load
        cp_async_wait_all();
        __syncthreads();

        // Issue K_next load (overlaps with PV)
        if (kv_idx < num_tiles - 1) {
            int next_start = (kv_idx + 1) * BN;
            int next_len = min(next_start + BN, S) - next_start;
            for (int i = tid * 8; i < BN * D; i += THREADS * 8) {
                int row = i / D, col = i % D;
                int load_row = min(row, next_len - 1);
                cp_async_16(smem_k + i, K_bh + (next_start + load_row) * D + col);
            }
            cp_async_commit();
        }

        // PV matmul
        {
            wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::row_major> p_frag;
            wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::row_major> v_frag;

            int m_row = warp_id * WM;
            #pragma unroll
            for (int k = 0; k < BN / WK; k++) {
                wmma::load_matrix_sync(p_frag, smem_p + m_row * BN + k * WK, BN);
                #pragma unroll
                for (int ni = 0; ni < D / WN; ni++) {
                    wmma::load_matrix_sync(v_frag, smem_v + k * WK * D + ni * WN, D);
                    wmma::mma_sync(o_frag[ni], p_frag, v_frag, o_frag[ni]);
                }
            }
        }

        if (kv_idx < num_tiles - 1) {
            cp_async_wait_all();
        }
        __syncthreads();
    }

    // Store O to smem (reuse smem_q space as float buffer), normalize, write to global
    {
        float* smem_o = reinterpret_cast<float*>(smem);
        int m_row = warp_id * WM;
        #pragma unroll
        for (int ni = 0; ni < D / WN; ni++)
            wmma::store_matrix_sync(smem_o + m_row * D + ni * WN, o_frag[ni], D, wmma::mem_row_major);
    }
    __syncthreads();

    {
        float* smem_o = reinterpret_cast<float*>(smem);
        for (int i = tid; i < BM * D; i += THREADS) {
            int row = i / D, col = i % D;
            if (q_start + row < S) {
                float l = smem_l[row];
                O_bh[(q_start + row) * D + col] = __float2bfloat16(
                    (l > 0.0f) ? (smem_o[i] / l) : 0.0f);
            }
        }
    }

    if (tid < BM) {
        int row = tid;
        if (q_start + row < S) {
            float l = smem_l[row];
            float m = smem_m[row];
            LSE_bh[q_start + row] = (l > 0.0f) ? (m + logf(l)) : -INFINITY;
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int S = (int)Q.size(2);

    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());

    dim3 grid(B * H, (S + BM - 1) / BM);
    dim3 block(THREADS);
    // smem_q + smem_k + smem_v + smem_p + smem_m/l/rescale
    int smem_bytes = BM * D * 2 + BN * D * 2 + BN * D * 2 + BM * BN * 2 + 3 * BM * 4;

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    cudaFuncSetAttribute(flash_attn_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    flash_attn_kernel<<<grid, block, smem_bytes, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data, B, H, S);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flash_attn::run);

}  // namespace flash_attn