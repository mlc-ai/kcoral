#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda;

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_kernel {

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int HD = 128;

__global__ void attn_kernel(const __nv_bfloat16* __restrict__ Q,
                            const __nv_bfloat16* __restrict__ K,
                            const __nv_bfloat16* __restrict__ V,
                            __nv_bfloat16* __restrict__ O,
                            float* __restrict__ LSE,
                            int S, float scale) {
    int bh = blockIdx.y;
    int qblock = blockIdx.x;
    int q_start = qblock * BM;
    int tid = threadIdx.x;
    int warp = tid >> 5;

    extern __shared__ char smem[];
    __nv_bfloat16* Qs = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* Ks = Qs + BM*HD;
    __nv_bfloat16* Vs = Ks + BN*HD;
    float* Ss = reinterpret_cast<float*>(Vs + BN*HD);
    __nv_bfloat16* Ps = reinterpret_cast<__nv_bfloat16*>(Ss + BM*BN);
    float* Os = reinterpret_cast<float*>(Ps + BM*BN);
    float* m_s = Os + BM*HD;
    float* l_s = m_s + BM;
    float* corr_s = l_s + BM;

    const __nv_bfloat16* Qbase = Q + (size_t)bh * S * HD;
    const __nv_bfloat16* Kbase = K + (size_t)bh * S * HD;
    const __nv_bfloat16* Vbase = V + (size_t)bh * S * HD;

    // Load Q tile
    for (int idx = tid; idx < BM*HD; idx += blockDim.x) {
        int r = idx / HD, c = idx % HD;
        int gr = q_start + r;
        Qs[idx] = (gr < S) ? Qbase[(size_t)gr*HD + c] : __float2bfloat16(0.0f);
    }
    for (int i = tid; i < BM; i += blockDim.x) { m_s[i] = -1e30f; l_s[i] = 0.0f; }
    for (int idx = tid; idx < BM*HD; idx += blockDim.x) Os[idx] = 0.0f;
    __syncthreads();

    int num_kv = (S + BN - 1) / BN;
    for (int kv = 0; kv < num_kv; kv++) {
        int kv_start = kv * BN;
        for (int idx = tid; idx < BN*HD; idx += blockDim.x) {
            int r = idx / HD, c = idx % HD;
            int gr = kv_start + r;
            if (gr < S) {
                Ks[idx] = Kbase[(size_t)gr*HD + c];
                Vs[idx] = Vbase[(size_t)gr*HD + c];
            } else {
                Ks[idx] = __float2bfloat16(0.0f);
                Vs[idx] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        // QK^T  ->  S = Q @ K^T * scale
        {
            int m0 = warp * 16;
            for (int nt = 0; nt < BN/16; nt++) {
                int n0 = nt*16;
                wmma::fragment<wmma::accumulator,16,16,16,float> cf;
                wmma::fill_fragment(cf, 0.0f);
                for (int kt = 0; kt < HD/16; kt++) {
                    int k0 = kt*16;
                    wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> af;
                    wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> bf;
                    wmma::load_matrix_sync(af, &Qs[m0*HD + k0], HD);
                    wmma::load_matrix_sync(bf, &Ks[n0*HD + k0], HD);
                    wmma::mma_sync(cf, af, bf, cf);
                }
                #pragma unroll
                for (int t = 0; t < cf.num_elements; t++) cf.x[t] *= scale;
                wmma::store_matrix_sync(&Ss[m0*BN + n0], cf, BN, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // Online softmax (one thread per row)
        if (tid < BM) {
            int i = tid;
            int valid = S - kv_start; if (valid > BN) valid = BN;
            float m_old = m_s[i];
            float m_new = m_old;
            for (int j = 0; j < BN; j++) {
                float v = (j < valid) ? Ss[i*BN+j] : -1e30f;
                Ss[i*BN+j] = v;
                m_new = fmaxf(m_new, v);
            }
            float corr = __expf(m_old - m_new);
            float s = 0.0f;
            for (int j = 0; j < BN; j++) {
                float p = __expf(Ss[i*BN+j] - m_new);
                Ps[i*BN+j] = __float2bfloat16(p);
                s += p;
            }
            l_s[i] = l_s[i]*corr + s;
            m_s[i] = m_new;
            corr_s[i] = corr;
        }
        __syncthreads();

        // Rescale O accumulator
        for (int idx = tid; idx < BM*HD; idx += blockDim.x) {
            int r = idx / HD;
            Os[idx] *= corr_s[r];
        }
        __syncthreads();

        // O += P @ V
        {
            int m0 = warp*16;
            for (int nt = 0; nt < HD/16; nt++) {
                int n0 = nt*16;
                wmma::fragment<wmma::accumulator,16,16,16,float> cf;
                wmma::load_matrix_sync(cf, &Os[m0*HD + n0], HD, wmma::mem_row_major);
                for (int kt = 0; kt < BN/16; kt++) {
                    int k0 = kt*16;
                    wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> af;
                    wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> bf;
                    wmma::load_matrix_sync(af, &Ps[m0*BN + k0], BN);
                    wmma::load_matrix_sync(bf, &Vs[k0*HD + n0], HD);
                    wmma::mma_sync(cf, af, bf, cf);
                }
                wmma::store_matrix_sync(&Os[m0*HD + n0], cf, HD, wmma::mem_row_major);
            }
        }
        __syncthreads();
    }

    // Finalize: normalize + write LSE
    if (tid < BM) {
        int i = tid;
        int gr = q_start + i;
        if (gr < S) {
            LSE[(size_t)bh*S + gr] = m_s[i] + logf(l_s[i]);
            corr_s[i] = 1.0f / l_s[i];
        } else {
            corr_s[i] = 0.0f;
        }
    }
    __syncthreads();

    __nv_bfloat16* Obase = O + (size_t)bh * S * HD;
    for (int idx = tid; idx < BM*HD; idx += blockDim.x) {
        int r = idx / HD, c = idx % HD;
        int gr = q_start + r;
        if (gr < S) {
            Obase[(size_t)gr*HD + c] = __float2bfloat16(Os[idx] * corr_s[r]);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int B = Q.size(0), H = Q.size(1), S = Q.size(2);
    float scale = 1.0f / sqrtf((float)HD);

    const __nv_bfloat16* Qp = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* Kp = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* Vp = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* Op = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* Lp = static_cast<float*>(LSE.data_ptr());

    int num_q = (S + BM - 1) / BM;
    dim3 grid(num_q, B*H);
    dim3 block(128);
    size_t smem = (size_t)BM*HD*2 + (size_t)BN*HD*2 + (size_t)BN*HD*2
                + (size_t)BM*BN*4 + (size_t)BM*BN*2 + (size_t)BM*HD*4
                + (size_t)3*BM*4;

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(attn_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    attn_kernel<<<grid, block, smem, stream>>>(Qp, Kp, Vp, Op, Lp, S, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel::run);

}  // namespace mha_kernel