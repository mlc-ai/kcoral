#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda;

namespace flash_attn_impl {

constexpr int D = 128;
constexpr int BR = 64;
constexpr int BC = 64;
constexpr int NUM_THREADS = 128;
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int D_TILES = D / WMMA_K;    // 8
constexpr int BC_TILES = BC / WMMA_N;  // 4

// Double-buffered K and V, separate P buffer
constexpr int SMEM_SIZE =
    BR * D * 2 +        // sQ: 8192
    BC * D * 2 * 2 +    // sKV[2]: 16384 (double-buffered, shared K&V)
    BR * BC * 4 +       // sS: 16384
    BR * D * 4 +        // sO: 16384
    BR * D * 4 +        // sTmp: 16384 (P@V result)
    BR * 4 * 3;         // stats: 768

__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(smem_addr), "l"(gmem));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

__global__ void flash_attn_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S)
{
    int bh = blockIdx.x;
    int q_tile = blockIdx.y;
    int q_start = q_tile * BR;

    if (q_start >= S) return;
    int br = min(BR, S - q_start);

    extern __shared__ char smem[];
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sKV[2];
    sKV[0] = sQ + BR * D;
    sKV[1] = sKV[0] + BC * D;
    float* sS = reinterpret_cast<float*>(sKV[1] + BC * D);
    float* sO = sS + BR * BC;
    float* sTmp = sO + BR * D;
    float* s_row_max = sTmp + BR * D;
    float* s_row_sum = s_row_max + BR;
    float* s_alpha = s_row_sum + BR;

    const __nv_bfloat16* Q_base = Q + ((int64_t)bh * S * D);
    const __nv_bfloat16* K_base = K + ((int64_t)bh * S * D);
    const __nv_bfloat16* V_base = V + ((int64_t)bh * S * D);
    __nv_bfloat16* O_base = O + ((int64_t)bh * S * D);
    float* LSE_base = LSE + ((int64_t)bh * S);

    const float scale = 0.08838834764831845f;
    int warp_id = threadIdx.x / 32;

    // Load Q tile using cp.async
    {
        char* dst = reinterpret_cast<char*>(sQ);
        const char* src = reinterpret_cast<const char*>(Q_base + (int64_t)q_start * D);
        int total_bytes = br * D * 2;
        for (int i = threadIdx.x * 16; i < total_bytes; i += NUM_THREADS * 16) {
            cp_async_16(dst + i, src + i);
        }
        for (int i = threadIdx.x + br * D; i < BR * D; i += NUM_THREADS) {
            sQ[i] = __float2bfloat16(0.0f);
        }
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();
    }

    // Initialize sO and running stats
    for (int i = threadIdx.x; i < BR * D; i += NUM_THREADS) {
        sO[i] = 0.0f;
    }
    for (int i = threadIdx.x; i < BR; i += NUM_THREADS) {
        s_row_max[i] = -INFINITY;
        s_row_sum[i] = 0.0f;
    }

    int num_k_tiles = (S + BC - 1) / BC;

    // Pre-load K[0] into buffer 0
    int kv_buf = 0;
    {
        int bc = min(BC, S);
        char* dst = reinterpret_cast<char*>(sKV[0]);
        const char* src = reinterpret_cast<const char*>(K_base);
        int total_bytes = bc * D * 2;
        for (int i = threadIdx.x * 16; i < total_bytes; i += NUM_THREADS * 16) {
            cp_async_16(dst + i, src + i);
        }
        for (int i = threadIdx.x + bc * D; i < BC * D; i += NUM_THREADS) {
            sKV[0][i] = __float2bfloat16(0.0f);
        }
        cp_async_commit();
    }

    for (int kt = 0; kt < num_k_tiles; kt++) {
        int k_start = kt * BC;
        int bc = min(BC, S - k_start);

        if (k_start > q_start + br - 1) break;

        // Wait for K[kt] in current buffer
        cp_async_wait_all();
        __syncthreads();

        // Issue V[kt] load into the OTHER buffer
        int v_buf = 1 - kv_buf;
        {
            char* dst = reinterpret_cast<char*>(sKV[v_buf]);
            const char* src = reinterpret_cast<const char*>(V_base + (int64_t)k_start * D);
            int total_bytes = bc * D * 2;
            for (int i = threadIdx.x * 16; i < total_bytes; i += NUM_THREADS * 16) {
                cp_async_16(dst + i, src + i);
            }
            for (int i = threadIdx.x + bc * D; i < BC * D; i += NUM_THREADS) {
                sKV[v_buf][i] = __float2bfloat16(0.0f);
            }
            cp_async_commit();
        }

        // === QK^T using sKV[kv_buf] (K[kt]) ===
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_frag[4];
        for (int t = 0; t < 4; t++) wmma::fill_fragment(s_frag[t], 0.0f);

        for (int ki = 0; ki < D_TILES; ki++) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag;

            for (int t = 0; t < 4; t++) {
                int flat = warp_id * 4 + t;
                int m_tile = flat / BC_TILES;
                int n_tile = flat % BC_TILES;

                wmma::load_matrix_sync(a_frag, sQ + m_tile * WMMA_M * D + ki * WMMA_K, D);
                wmma::load_matrix_sync(b_frag, sKV[kv_buf] + n_tile * WMMA_N * D + ki * WMMA_K, D);
                wmma::mma_sync(s_frag[t], a_frag, b_frag, s_frag[t]);
            }
        }

        for (int t = 0; t < 4; t++) {
            int flat = warp_id * 4 + t;
            int m_tile = flat / BC_TILES;
            int n_tile = flat % BC_TILES;
            wmma::store_matrix_sync(sS + m_tile * WMMA_M * BC + n_tile * WMMA_N,
                                    s_frag[t], BC, wmma::mem_row_major);
        }
        __syncthreads();

        // Issue K[kt+1] load into kv_buf (overwriting K[kt], which is no longer needed)
        if (kt < num_k_tiles - 1) {
            int next_k_start = (kt + 1) * BC;
            int next_bc = min(BC, S - next_k_start);
            if (next_k_start <= q_start + br - 1) {
                char* dst = reinterpret_cast<char*>(sKV[kv_buf]);
                const char* src = reinterpret_cast<const char*>(K_base + (int64_t)next_k_start * D);
                int total_bytes = next_bc * D * 2;
                for (int i = threadIdx.x * 16; i < total_bytes; i += NUM_THREADS * 16) {
                    cp_async_16(dst + i, src + i);
                }
                for (int i = threadIdx.x + next_bc * D; i < BC * D; i += NUM_THREADS) {
                    sKV[kv_buf][i] = __float2bfloat16(0.0f);
                }
                cp_async_commit();
            }
        }

        // === Softmax ===
        if (threadIdx.x < BR) {
            int row = threadIdx.x;
            if (row < br) {
                int q_pos = q_start + row;
                float local_max = -INFINITY;
                for (int j = 0; j < BC; j++) {
                    int k_pos = k_start + j;
                    float val = sS[row * BC + j] * scale;
                    if (k_pos > q_pos || k_pos >= S) val = -INFINITY;
                    sS[row * BC + j] = val;
                    if (val > local_max) local_max = val;
                }

                float m_old = s_row_max[row];
                float m_new = fmaxf(m_old, local_max);
                float alpha;
                if (m_new == -INFINITY) {
                    alpha = 1.0f;
                } else if (m_old == -INFINITY) {
                    alpha = 0.0f;
                } else {
                    alpha = __expf(m_old - m_new);
                }

                float local_sum = 0.0f;
                if (m_new > -INFINITY) {
                    for (int j = 0; j < BC; j++) {
                        float val = sS[row * BC + j];
                        float p = (val > -INFINITY) ? __expf(val - m_new) : 0.0f;
                        sS[row * BC + j] = p;
                        local_sum += p;
                    }
                } else {
                    for (int j = 0; j < BC; j++) {
                        sS[row * BC + j] = 0.0f;
                    }
                }

                s_row_max[row] = m_new;
                s_row_sum[row] = s_row_sum[row] * alpha + local_sum;
                s_alpha[row] = alpha;
            }
        }
        __syncthreads();

        // Rescale O
        for (int i = threadIdx.x; i < BR * D; i += NUM_THREADS) {
            int row = i / D;
            if (row < br) {
                sO[i] *= s_alpha[row];
            }
        }

        // Convert P to bf16 in sTmp (separate buffer, no race condition)
        __nv_bfloat16* sP = reinterpret_cast<__nv_bfloat16*>(sTmp);
        for (int i = threadIdx.x; i < BR * BC; i += NUM_THREADS) {
            sP[i] = __float2bfloat16(sS[i]);
        }

        // Wait for V[kt] in v_buf
        cp_async_wait_all();
        __syncthreads();

        // === P @ V using sKV[v_buf] (V[kt]) ===
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> pv_frag[8];
        for (int t = 0; t < 8; t++) wmma::fill_fragment(pv_frag[t], 0.0f);

        for (int ki = 0; ki < BC_TILES; ki++) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;

            for (int t = 0; t < 8; t++) {
                int flat = warp_id * 8 + t;
                int m_tile = flat / D_TILES;
                int n_tile = flat % D_TILES;

                wmma::load_matrix_sync(a_frag, sP + m_tile * WMMA_M * BC + ki * WMMA_K, BC);
                wmma::load_matrix_sync(b_frag, sKV[v_buf] + ki * WMMA_K * D + n_tile * WMMA_N, D);
                wmma::mma_sync(pv_frag[t], a_frag, b_frag, pv_frag[t]);
            }
        }

        for (int t = 0; t < 8; t++) {
            int flat = warp_id * 8 + t;
            int m_tile = flat / D_TILES;
            int n_tile = flat % D_TILES;
            wmma::store_matrix_sync(sTmp + m_tile * WMMA_M * D + n_tile * WMMA_N,
                                    pv_frag[t], D, wmma::mem_row_major);
        }
        __syncthreads();

        // O += sTmp
        for (int i = threadIdx.x; i < BR * D; i += NUM_THREADS) {
            sO[i] += sTmp[i];
        }

        // Swap buffers: next iteration's K is in kv_buf, V will go to the other
        kv_buf = v_buf;
        __syncthreads();
    }

    // Final normalization
    for (int i = threadIdx.x; i < BR * D; i += NUM_THREADS) {
        int row = i / D;
        int col = i % D;
        if (row < br) {
            float sum = s_row_sum[row];
            float val = (sum > 0.0f) ? sO[i] / sum : 0.0f;
            O_base[(int64_t)(q_start + row) * D + col] = __float2bfloat16(val);
        }
    }

    if (threadIdx.x < BR) {
        int row = threadIdx.x;
        if (row < br) {
            float sum = s_row_sum[row];
            float lse = (sum > 0.0f) ? (s_row_max[row] + logf(sum)) : -INFINITY;
            LSE_base[q_start + row] = lse;
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    int device_id = Q.device().device_id;
    cudaSetDevice(device_id);

    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int S = (int)Q.size(2);

    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());

    dim3 grid(B * H, (S + BR - 1) / BR);
    dim3 block(NUM_THREADS);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, device_id));

    cudaFuncSetAttribute(flash_attn_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE);

    flash_attn_kernel<<<grid, block, SMEM_SIZE, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data, S);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaStreamSynchronize(stream);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flash_attn_impl::run);

}  // namespace flash_attn_impl