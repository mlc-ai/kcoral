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

constexpr int BM = 64;
constexpr int BN = 128;
constexpr int D  = 128;
constexpr int THREADS = 128;
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
    float* smem_s          = reinterpret_cast<float*>(smem_q + BM * D);
    __nv_bfloat16* smem_kv = reinterpret_cast<__nv_bfloat16*>(smem_s + BM * BN);
    __nv_bfloat16* smem_p  = smem_kv + BN * D;
    float* smem_m          = reinterpret_cast<float*>(smem_p + BM * BN);
    float* smem_l          = smem_m + BM;
    float* smem_rescale    = smem_l + BM;

    int tid = threadIdx.x;
    int warp_id = tid / 32;

    const float scale = 0.08838834764831845f;

    // Load Q tile
    for (int i = tid * 8; i < BM * D; i += THREADS * 8) {
        int row = i / D, col = i % D;
        if (q_start + row < S) {
            cp_async_16(smem_q + i, Q_bh + (q_start + row) * D + col);
        } else {
            *reinterpret_cast<uint4*>(smem_q + i) = make_uint4(0, 0, 0, 0);
        }
    }
    cp_async_commit();
    cp_async_wait_all();

    if (tid < BM) {
        smem_m[tid] = -INFINITY;
        smem_l[tid] = 0.0f;
    }

    wmma::fragment<wmma::accumulator, WM, WN, WK, float> o_frag[D / WN];
    #pragma unroll
    for (int ni = 0; ni < D / WN; ni++)
        wmma::fill_fragment(o_frag[ni], 0.0f);

    __syncthreads();

    int num_kv_tiles = (q_end + BN - 1) / BN;

    for (int kv_idx = 0; kv_idx < num_kv_tiles; kv_idx++) {
        int kv_start = kv_idx * BN;
        int kv_len = min(kv_start + BN, S) - kv_start;
        bool need_mask = (kv_start + BN > q_start);

        // Load K tile
        for (int i = tid * 8; i < BN * D; i += THREADS * 8) {
            int row = i / D, col = i % D;
            int load_row = min(row, kv_len - 1);
            cp_async_16(smem_kv + i, K_bh + (kv_start + load_row) * D + col);
        }
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();

        // S = Q @ K^T
        // K is (BN, D) row-major. K^T[k][n] = K[n][k] = smem_kv[n*D + k]
        // B row_major: B[k][n] = ptr[n*ld + k], ptr=smem_kv, ld=D -> K[n][k] = K^T[k][n] ✓
        {
            wmma::fragment<wmma::accumulator, WM, WN, WK, float> s_frag[BN / WN];
            wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::row_major> b_frag;

            #pragma unroll
            for (int ni = 0; ni < BN / WN; ni++)
                wmma::fill_fragment(s_frag[ni], 0.0f);

            int m_row = warp_id * WM;

            #pragma unroll
            for (int k = 0; k < D / WK; k++) {
                wmma::load_matrix_sync(a_frag, smem_q + m_row * D + k * WK, D);
                #pragma unroll
                for (int ni = 0; ni < BN / WN; ni++) {
                    wmma::load_matrix_sync(b_frag, smem_kv + ni * WN * D + k * WK, D);
                    wmma::mma_sync(s_frag[ni], a_frag, b_frag, s_frag[ni]);
                }
            }

            #pragma unroll
            for (int ni = 0; ni < BN / WN; ni++) {
                wmma::store_matrix_sync(smem_s + m_row * BN + ni * WN, s_frag[ni], BN, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // Start V load (overlaps with softmax + rescale)
        for (int i = tid * 8; i < BN * D; i += THREADS * 8) {
            int row = i / D, col = i % D;
            int load_row = min(row, kv_len - 1);
            cp_async_16(smem_kv + i, V_bh + (kv_start + load_row) * D + col);
        }
        cp_async_commit();

        // Softmax: one thread per row
        if (tid < BM) {
            int row = tid;
            int q_row = q_start + row;

            float row_max = -INFINITY;
            for (int j = 0; j < BN; j++) {
                int k_col = kv_start + j;
                float s = smem_s[row * BN + j] * scale;
                if (k_col >= S || (need_mask && k_col > q_row)) s = -INFINITY;
                smem_s[row * BN + j] = s;
                row_max = fmaxf(row_max, s);
            }

            float m_old = smem_m[row];
            float m_new = fmaxf(m_old, row_max);
            float rescale = (m_old == -INFINITY) ? 0.0f : __expf(m_old - m_new);

            float row_sum = 0.0f;
            for (int j = 0; j < BN; j++) {
                float s = smem_s[row * BN + j];
                float p = (s == -INFINITY) ? 0.0f : __expf(s - m_new);
                smem_p[row * BN + j] = __float2bfloat16(p);
                row_sum += p;
            }

            smem_m[row] = m_new;
            smem_l[row] = smem_l[row] * rescale + row_sum;
            smem_rescale[row] = rescale;
        }
        __syncthreads();

        // Rescale O: store to smem_s (reused, BM*BN = BM*D since BN=D=128), rescale, load back
        {
            int m_row = warp_id * WM;
            #pragma unroll
            for (int ni = 0; ni < D / WN; ni++) {
                wmma::store_matrix_sync(smem_s + m_row * D + ni * WN, o_frag[ni], D, wmma::mem_row_major);
            }
        }
        __syncthreads();

        for (int i = tid; i < BM * D; i += THREADS) {
            int row = i / D;
            smem_s[i] *= smem_rescale[row];
        }
        __syncthreads();

        {
            int m_row = warp_id * WM;
            #pragma unroll
            for (int ni = 0; ni < D / WN; ni++) {
                wmma::load_matrix_sync(o_frag[ni], smem_s + m_row * D + ni * WN, D, wmma::mem_row_major);
            }
        }

        // Wait for V load
        cp_async_wait_all();
        __syncthreads();

        // O += P @ V
        // V is (BN, D) row-major. V[k][n] = smem_kv[k*D + n]
        // B col_major: B[k][n] = ptr[k*ld + n], ptr=smem_kv, ld=D -> V[k][n] ✓
        {
            wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::row_major> p_frag;
            wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::col_major> v_frag;

            int m_row = warp_id * WM;

            #pragma unroll
            for (int k = 0; k < BN / WK; k++) {
                wmma::load_matrix_sync(p_frag, smem_p + m_row * BN + k * WK, BN);
                #pragma unroll
                for (int ni = 0; ni < D / WN; ni++) {
                    wmma::load_matrix_sync(v_frag, smem_kv + k * WK * D + ni * WN, D);
                    wmma::mma_sync(o_frag[ni], p_frag, v_frag, o_frag[ni]);
                }
            }
        }
        __syncthreads();
    }

    // Store O to smem_s, normalize, write to global
    {
        int m_row = warp_id * WM;
        #pragma unroll
        for (int ni = 0; ni < D / WN; ni++) {
            wmma::store_matrix_sync(smem_s + m_row * D + ni * WN, o_frag[ni], D, wmma::mem_row_major);
        }
    }
    __syncthreads();

    for (int i = tid; i < BM * D; i += THREADS) {
        int row = i / D, col = i % D;
        if (q_start + row < S) {
            float l = smem_l[row];
            float o_val = smem_s[row * D + col];
            O_bh[(q_start + row) * D + col] = __float2bfloat16(
                (l > 0.0f) ? (o_val / l) : 0.0f);
        }
    }

    // Store LSE
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
    int smem_bytes = BM * D * 2 + BM * BN * 4 + BN * D * 2 + BM * BN * 2 + 3 * BM * 4;

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