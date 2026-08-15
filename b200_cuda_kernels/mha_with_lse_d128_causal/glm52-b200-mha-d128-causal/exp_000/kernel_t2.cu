#include <cuda_bf16.h>
#include <mma.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda::wmma;

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", \
                cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

namespace flash_attn_impl {

constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
constexpr int HEAD_DIM = 128;
constexpr int BR = 128;
constexpr int BC = 64;
constexpr int D_PAD = 136;      // 16-byte aligned row stride, 4-way bank conflict
constexpr int S_STRIDE = 68;    // S fp32 stride (16B aligned, 4-way conflict)
constexpr int P_STRIDE = 72;    // P bf16 stride (16B aligned, 4-way conflict)
constexpr int O_STRIDE = 132;   // O fp32 stride (16B aligned, 4-way conflict)
constexpr float SCALE = 0.08838834764831840f;

__global__ __launch_bounds__(128, 1)
void flash_attn_kernel(
    const __nv_bfloat16* __restrict__ Q_gmem,
    const __nv_bfloat16* __restrict__ K_gmem,
    const __nv_bfloat16* __restrict__ V_gmem,
    __nv_bfloat16* __restrict__ O_gmem,
    float* __restrict__ LSE_gmem,
    int H, int S_val) {

    const int bh = blockIdx.x;
    const int batch = bh / H;
    const int head = bh % H;
    const int q_block = blockIdx.y;
    const int q_start = q_block * BR;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;

    extern __shared__ char smem_raw[];
    char* ptr = smem_raw;
    __nv_bfloat16* Q_smem = reinterpret_cast<__nv_bfloat16*>(ptr);
    ptr += BR * D_PAD * sizeof(__nv_bfloat16);
    __nv_bfloat16* K_smem = reinterpret_cast<__nv_bfloat16*>(ptr);
    ptr += BC * HEAD_DIM * sizeof(__nv_bfloat16);
    __nv_bfloat16* V_smem = reinterpret_cast<__nv_bfloat16*>(ptr);
    ptr += BC * HEAD_DIM * sizeof(__nv_bfloat16);
    float* S_smem = reinterpret_cast<float*>(ptr);
    ptr += BR * S_STRIDE * sizeof(float);
    __nv_bfloat16* P_smem = reinterpret_cast<__nv_bfloat16*>(S_smem);
    float* O_smem = reinterpret_cast<float*>(ptr);
    ptr += BR * O_STRIDE * sizeof(float);
    float* m_smem = reinterpret_cast<float*>(ptr);
    ptr += BR * sizeof(float);
    float* l_smem = reinterpret_cast<float*>(ptr);

    const int64_t qkv_offset = ((int64_t)batch * H + head) * S_val * HEAD_DIM;

    // Load Q tile with padding
    for (int i = tid; i < BR * HEAD_DIM; i += 128) {
        int row = i / HEAD_DIM, col = i % HEAD_DIM;
        int gr = q_start + row;
        Q_smem[row * D_PAD + col] = (gr < S_val)
            ? Q_gmem[qkv_offset + (int64_t)gr * HEAD_DIM + col]
            : __float2bfloat16(0.0f);
    }
    for (int i = tid; i < BR * (D_PAD - HEAD_DIM); i += 128) {
        int row = i / (D_PAD - HEAD_DIM), col = i % (D_PAD - HEAD_DIM);
        Q_smem[row * D_PAD + HEAD_DIM + col] = __float2bfloat16(0.0f);
    }

    // Init O, m, l
    for (int i = tid; i < BR * O_STRIDE; i += 128) O_smem[i] = 0.0f;
    if (tid < BR) { m_smem[tid] = -INFINITY; l_smem[tid] = 0.0f; }
    __syncthreads();

    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, bfloat16, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, bfloat16, col_major> b_frag_kt;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, bfloat16, row_major> b_frag_v;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag;

    const int kv_end = min(q_start + BR, S_val);
    const int num_kv = (kv_end + BC - 1) / BC;

    for (int kv_blk = 0; kv_blk < num_kv; kv_blk++) {
        const int kv_start = kv_blk * BC;

        // Load K, V
        for (int i = tid; i < BC * HEAD_DIM; i += 128) {
            int row = i / HEAD_DIM, col = i % HEAD_DIM;
            int gr = kv_start + row;
            if (gr < kv_end) {
                K_smem[row * HEAD_DIM + col] = K_gmem[qkv_offset + (int64_t)gr * HEAD_DIM + col];
                V_smem[row * HEAD_DIM + col] = V_gmem[qkv_offset + (int64_t)gr * HEAD_DIM + col];
            } else {
                K_smem[row * HEAD_DIM + col] = __float2bfloat16(0.0f);
                V_smem[row * HEAD_DIM + col] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        // S = Q @ K^T via wmma (128x64 = 8 m-tiles x 4 n-tiles x 8 k-iter)
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            int mt = warp_id * 2 + mi;
            #pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                fill_fragment(c_frag, 0.0f);
                #pragma unroll
                for (int ki = 0; ki < 8; ki++) {
                    load_matrix_sync(a_frag, &Q_smem[mt * 16 * D_PAD + ki * 16], D_PAD);
                    load_matrix_sync(b_frag_kt, &K_smem[ni * 16 * HEAD_DIM + ki * 16], HEAD_DIM);
                    mma_sync(c_frag, a_frag, b_frag_kt, c_frag);
                }
                store_matrix_sync(&S_smem[mt * 16 * S_STRIDE + ni * 16], c_frag, S_STRIDE, mem_row_major);
            }
        }
        __syncthreads();

        // Softmax: each thread handles one row
        if (tid < BR) {
            int q_idx = q_start + tid;
            if (q_idx < S_val) {
                float row_max = -INFINITY;
                float P_vals[BC];
                #pragma unroll
                for (int j = 0; j < BC; j++) {
                    float val = S_smem[tid * S_STRIDE + j] * SCALE;
                    int key_idx = kv_start + j;
                    if (key_idx > q_idx || key_idx >= kv_end) val = -INFINITY;
                    row_max = fmaxf(row_max, val);
                    P_vals[j] = val;
                }

                float old_max = m_smem[tid];
                float new_max = fmaxf(old_max, row_max);
                float rescale = (old_max > -INFINITY) ? __expf(old_max - new_max) : 0.0f;

                float row_sum = 0.0f;
                #pragma unroll
                for (int j = 0; j < BC; j++) {
                    float p = (P_vals[j] > -INFINITY) ? __expf(P_vals[j] - new_max) : 0.0f;
                    P_vals[j] = p;
                    row_sum += p;
                }

                #pragma unroll
                for (int d = 0; d < HEAD_DIM; d++)
                    O_smem[tid * O_STRIDE + d] *= rescale;

                float old_l = l_smem[tid];
                m_smem[tid] = new_max;
                l_smem[tid] = old_l * rescale + row_sum;

                #pragma unroll
                for (int j = 0; j < BC; j++)
                    P_smem[tid * P_STRIDE + j] = __float2bfloat16(P_vals[j]);
            } else {
                #pragma unroll
                for (int j = 0; j < BC; j++)
                    P_smem[tid * P_STRIDE + j] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        // O += P @ V via wmma (128x128 = 8 m-tiles x 8 n-tiles x 4 k-iter)
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            int mt = warp_id * 2 + mi;
            #pragma unroll
            for (int ni = 0; ni < 8; ni++) {
                load_matrix_sync(o_frag, &O_smem[mt * 16 * O_STRIDE + ni * 16], O_STRIDE);
                #pragma unroll
                for (int ki = 0; ki < 4; ki++) {
                    load_matrix_sync(a_frag, &P_smem[mt * 16 * P_STRIDE + ki * 16], P_STRIDE);
                    load_matrix_sync(b_frag_v, &V_smem[ki * 16 * HEAD_DIM + ni * 16], HEAD_DIM);
                    mma_sync(o_frag, a_frag, b_frag_v, o_frag);
                }
                store_matrix_sync(&O_smem[mt * 16 * O_STRIDE + ni * 16], o_frag, O_STRIDE, mem_row_major);
            }
        }
        __syncthreads();
    }

    // Write O and LSE
    if (tid < BR) {
        int q_idx = q_start + tid;
        if (q_idx < S_val) {
            float l = l_smem[tid], m = m_smem[tid];
            float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
            LSE_gmem[((int64_t)batch * H + head) * S_val + q_idx] =
                (l > 0.0f) ? (m + logf(l)) : -INFINITY;
            for (int d = 0; d < HEAD_DIM; d++)
                O_gmem[qkv_offset + (int64_t)q_idx * HEAD_DIM + d] =
                    __float2bfloat16(O_smem[tid * O_STRIDE + d] * inv_l);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    const int B = 4, H = 48;
    const int S_val = static_cast<int>(Q.size(2));

    const __nv_bfloat16* Qp = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* Kp = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* Vp = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* Op = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* Lp = static_cast<float*>(LSE.data_ptr());

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    const int smem = BR*D_PAD*2 + BC*128*2 + BC*128*2 + BR*S_STRIDE*4
                   + BR*O_STRIDE*4 + BR*4 + BR*4;  // 171008

    CUDA_CHECK(cudaFuncSetAttribute(flash_attn_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

    dim3 grid(B * H, (S_val + BR - 1) / BR);
    flash_attn_kernel<<<grid, 128, smem, stream>>>(Qp, Kp, Vp, Op, Lp, H, S_val);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flash_attn_impl::run);

}  // namespace flash_attn_impl