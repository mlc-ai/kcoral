#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>
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

using namespace nvcuda;

namespace mha_lse {

constexpr int Br = 128;
constexpr int Bc = 64;
constexpr int D = 128;
constexpr int THREADS = 128;
constexpr int Bc_pad = 64;
constexpr int D_pad = 128;

constexpr int SMEM_SIZE = Br * D * 2 + Bc * D * 2 + Br * Bc_pad * 4 + Br * Bc_pad * 2 + Br * D_pad * 4;

__global__ __launch_bounds__(THREADS, 1)
void mha_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S, float scale) {

    int tid = threadIdx.x;
    int warp_id = tid / 32;

    int n_q_blocks = (S + Br - 1) / Br;
    int grid_idx = blockIdx.x;
    int batch = grid_idx / (H * n_q_blocks);
    int rest = grid_idx % (H * n_q_blocks);
    int head = rest / n_q_blocks;
    int q_block = rest % n_q_blocks;
    int q_start = q_block * Br;

    extern __shared__ char smem[];
    __nv_bfloat16* smem_q = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* smem_k = smem_q + Br * D;
    float* smem_s = reinterpret_cast<float*>(smem_k + Bc * D);
    __nv_bfloat16* smem_p = reinterpret_cast<__nv_bfloat16*>(smem_s + Br * Bc_pad);
    float* smem_o = reinterpret_cast<float*>(smem_p + Br * Bc_pad);

    const __nv_bfloat16* Q_gptr = Q + ((batch * H + head) * S + q_start) * D;
    for (int i = tid; i < Br * D; i += THREADS) {
        int row = i / D;
        smem_q[i] = (q_start + row < S) ? Q_gptr[i] : __float2bfloat16(0.0f);
    }

    for (int i = tid; i < Br * D_pad; i += THREADS) {
        smem_o[i] = 0.0f;
    }

    for (int i = tid; i < Br * Bc_pad; i += THREADS) {
        smem_p[i] = __float2bfloat16(0.0f);
    }

    __syncthreads();

    float m = -INFINITY;
    float l = 0.0f;

    for (int kv_start = 0; kv_start < S; kv_start += Bc) {
        int actual_Bc = min(Bc, S - kv_start);

        const __nv_bfloat16* K_gptr = K + ((batch * H + head) * S + kv_start) * D;
        for (int i = tid; i < Bc * D; i += THREADS) {
            int row = i / D;
            smem_k[i] = (row < actual_Bc) ? K_gptr[i] : __float2bfloat16(0.0f);
        }
        __syncthreads();

        // QK^T using wmma: S = Q @ K^T * scale
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            int m_tile = warp_id * 2 + mi;
            #pragma unroll
            for (int n_tile = 0; n_tile < 4; n_tile++) {
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
                wmma::fill_fragment(c_frag, 0.0f);
                #pragma unroll
                for (int k_tile = 0; k_tile < 8; k_tile++) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
                    wmma::load_matrix_sync(a_frag, smem_q + m_tile * 16 * D + k_tile * 16, D);
                    wmma::load_matrix_sync(b_frag, smem_k + n_tile * 16 * D + k_tile * 16, D);
                    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                }
                #pragma unroll
                for (int i = 0; i < c_frag.num_elements; i++) {
                    c_frag.x[i] *= scale;
                }
                wmma::store_matrix_sync(smem_s + m_tile * 16 * Bc_pad + n_tile * 16, c_frag, Bc_pad, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // Online softmax: each thread handles one row
        {
            float* s_row = smem_s + tid * Bc_pad;
            __nv_bfloat16* p_row = smem_p + tid * Bc_pad;
            float* o_row = smem_o + tid * D_pad;
            bool valid = (q_start + tid < S);

            if (valid) {
                float m_block = -INFINITY;
                #pragma unroll
                for (int j = 0; j < Bc; j++) {
                    if (kv_start + j < S) {
                        m_block = fmaxf(m_block, s_row[j]);
                    }
                }
                float m_new = fmaxf(m, m_block);
                float scale_o = expf(m - m_new);

                #pragma unroll
                for (int d = 0; d < D; d += 4) {
                    float4 o4 = *reinterpret_cast<float4*>(o_row + d);
                    o4.x *= scale_o; o4.y *= scale_o; o4.z *= scale_o; o4.w *= scale_o;
                    *reinterpret_cast<float4*>(o_row + d) = o4;
                }

                float l_block = 0.0f;
                #pragma unroll
                for (int j = 0; j < Bc; j++) {
                    if (kv_start + j < S) {
                        float p = expf(s_row[j] - m_new);
                        p_row[j] = __float2bfloat16(p);
                        l_block += p;
                    } else {
                        p_row[j] = __float2bfloat16(0.0f);
                    }
                }
                l = l * scale_o + l_block;
                m = m_new;
            } else {
                #pragma unroll
                for (int j = 0; j < Bc; j++) {
                    p_row[j] = __float2bfloat16(0.0f);
                }
            }
        }
        __syncthreads();

        // Load V [actual_Bc, D] into smem_k (reuse buffer)
        const __nv_bfloat16* V_gptr = V + ((batch * H + head) * S + kv_start) * D;
        for (int i = tid; i < Bc * D; i += THREADS) {
            int row = i / D;
            smem_k[i] = (row < actual_Bc) ? V_gptr[i] : __float2bfloat16(0.0f);
        }
        __syncthreads();

        // PV using wmma: O += P @ V
        __nv_bfloat16* smem_v = smem_k;
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            int m_tile = warp_id * 2 + mi;
            #pragma unroll
            for (int n_tile = 0; n_tile < 8; n_tile++) {
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
                wmma::load_matrix_sync(c_frag, smem_o + m_tile * 16 * D_pad + n_tile * 16, D_pad, wmma::mem_row_major);
                #pragma unroll
                for (int k_tile = 0; k_tile < 4; k_tile++) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_frag;
                    wmma::load_matrix_sync(a_frag, smem_p + m_tile * 16 * Bc_pad + k_tile * 16, Bc_pad);
                    wmma::load_matrix_sync(b_frag, smem_v + k_tile * 16 * D + n_tile * 16, D);
                    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                }
                wmma::store_matrix_sync(smem_o + m_tile * 16 * D_pad + n_tile * 16, c_frag, D_pad, wmma::mem_row_major);
            }
        }
        __syncthreads();
    }

    // Final output: normalize O by l, convert to bf16, store to global
    if (q_start + tid < S) {
        float inv_l = 1.0f / l;
        float* o_row = smem_o + tid * D_pad;
        __nv_bfloat16* O_gptr = O + ((batch * H + head) * S + q_start + tid) * D;

        #pragma unroll
        for (int d = 0; d < D; d += 8) {
            float4 o4_0 = *reinterpret_cast<float4*>(o_row + d);
            float4 o4_1 = *reinterpret_cast<float4*>(o_row + d + 4);

            __nv_bfloat162 p01 = __float22bfloat162_rn(make_float2(o4_0.x * inv_l, o4_0.y * inv_l));
            __nv_bfloat162 p23 = __float22bfloat162_rn(make_float2(o4_0.z * inv_l, o4_0.w * inv_l));
            __nv_bfloat162 p45 = __float22bfloat162_rn(make_float2(o4_1.x * inv_l, o4_1.y * inv_l));
            __nv_bfloat162 p67 = __float22bfloat162_rn(make_float2(o4_1.z * inv_l, o4_1.w * inv_l));

            float4 out;
            *reinterpret_cast<__nv_bfloat162*>(&out.x) = p01;
            *reinterpret_cast<__nv_bfloat162*>(&out.y) = p23;
            *reinterpret_cast<__nv_bfloat162*>(&out.z) = p45;
            *reinterpret_cast<__nv_bfloat162*>(&out.w) = p67;
            *reinterpret_cast<float4*>(O_gptr + d) = out;
        }

        LSE[(batch * H + head) * S + q_start + tid] = m + logf(l);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int B = 4;
    int H = 48;
    int D = 128;
    int64_t S = Q.size(2);
    float scale = 1.0f / sqrtf((float)D);
    int n_q_blocks = (static_cast<int>(S) + Br - 1) / Br;
    int grid = B * H * n_q_blocks;
    int block = THREADS;

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));

    mha_kernel<<<grid, block, SMEM_SIZE, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<__nv_bfloat16*>(O.data_ptr()),
        static_cast<float*>(LSE.data_ptr()),
        B, H, static_cast<int>(S), scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_lse::run);

}  // namespace mha_lse