#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>
#include <math.h>
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

constexpr int D = 128;
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int NTHREAD = 128;

__global__ void attn_kernel(const __nv_bfloat16* __restrict__ Q,
                            const __nv_bfloat16* __restrict__ K,
                            const __nv_bfloat16* __restrict__ V,
                            __nv_bfloat16* __restrict__ O,
                            float* __restrict__ LSE,
                            int B, int H, int S, int num_qtiles) {
    extern __shared__ char smem[];
    __nv_bfloat16* Qs = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* Ks = Qs + BM * D;
    __nv_bfloat16* Vs = Ks + BN * D;
    __nv_bfloat16* Ps = Vs + BN * D;
    float* Ss = reinterpret_cast<float*>(Ps + BM * BN);
    float* Os = Ss + BM * BN;
    float* ms = Os + BM * D;
    float* ls = ms + BM;

    int tid = threadIdx.x;
    int warp = tid / 32;

    int qtile = blockIdx.x % num_qtiles;
    int bh = blockIdx.x / num_qtiles;
    int h = bh % H;
    int b = bh / H;

    int q0 = qtile * BM;
    if (q0 >= S) return;

    const float scale = rsqrtf((float)D);
    long bh_base = (long)(b * H + h) * S * D;

    // Load Q tile (vectorized 16B = 8 bf16)
    const int VEC = 8;
    int total_vecQ = BM * D / VEC;
    for (int vi = tid; vi < total_vecQ; vi += NTHREAD) {
        int lin = vi * VEC;
        int row = lin / D;
        int col = lin % D;
        int grow = q0 + row;
        int4 val;
        if (grow < S) {
            val = *reinterpret_cast<const int4*>(&Q[bh_base + (long)grow * D + col]);
        } else {
            val = make_int4(0, 0, 0, 0);
        }
        *reinterpret_cast<int4*>(&Qs[row * D + col]) = val;
    }
    // init Os, ms, ls
    for (int i = tid; i < BM * D; i += NTHREAD) Os[i] = 0.f;
    if (tid < BM) { ms[tid] = -INFINITY; ls[tid] = 0.f; }
    __syncthreads();

    int max_key = min(q0 + BM - 1, S - 1);
    int num_kblocks = max_key / BN + 1;

    for (int kb = 0; kb < num_kblocks; kb++) {
        int k0 = kb * BN;

        // Load K, V tiles
        int total_vecK = BN * D / VEC;
        for (int vi = tid; vi < total_vecK; vi += NTHREAD) {
            int lin = vi * VEC;
            int row = lin / D;
            int col = lin % D;
            int gk = k0 + row;
            int4 valk, valv;
            if (gk < S) {
                valk = *reinterpret_cast<const int4*>(&K[bh_base + (long)gk * D + col]);
                valv = *reinterpret_cast<const int4*>(&V[bh_base + (long)gk * D + col]);
            } else {
                valk = make_int4(0, 0, 0, 0);
                valv = make_int4(0, 0, 0, 0);
            }
            *reinterpret_cast<int4*>(&Ks[row * D + col]) = valk;
            *reinterpret_cast<int4*>(&Vs[row * D + col]) = valv;
        }
        __syncthreads();

        // S = Q @ K^T  -> Ss  (warp handles rows [16w,16w+16), 4 col-tiles)
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
            #pragma unroll
            for (int c = 0; c < 4; c++) wmma::fill_fragment(acc[c], 0.f);
            #pragma unroll
            for (int k = 0; k < D / 16; k++) {
                wmma::load_matrix_sync(a_frag, &Qs[16 * warp * D + 16 * k], D);
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    wmma::load_matrix_sync(b_frag, &Ks[16 * c * D + 16 * k], D);
                    wmma::mma_sync(acc[c], a_frag, b_frag, acc[c]);
                }
            }
            #pragma unroll
            for (int c = 0; c < 4; c++)
                wmma::store_matrix_sync(&Ss[16 * warp * BN + 16 * c], acc[c], BN, wmma::mem_row_major);
        }
        __syncthreads();

        // Softmax (online) per row
        if (tid < BM) {
            int row = tid;
            int grow = q0 + row;
            if (grow < S) {
                float m_old = ms[row];
                float l_old = ls[row];
                float rowmax = -INFINITY;
                #pragma unroll
                for (int j = 0; j < BN; j++) {
                    int gk = k0 + j;
                    float s;
                    if (gk < S && gk <= grow) {
                        s = Ss[row * BN + j] * scale;
                    } else {
                        s = -INFINITY;
                    }
                    Ss[row * BN + j] = s;
                    rowmax = fmaxf(rowmax, s);
                }
                float m_new = fmaxf(m_old, rowmax);
                float correction = __expf(m_old - m_new);
                float rowsum = 0.f;
                #pragma unroll
                for (int j = 0; j < BN; j++) {
                    float p = __expf(Ss[row * BN + j] - m_new);
                    Ps[row * BN + j] = __float2bfloat16(p);
                    rowsum += p;
                }
                ms[row] = m_new;
                ls[row] = l_old * correction + rowsum;
                #pragma unroll
                for (int c = 0; c < D; c++) Os[row * D + c] *= correction;
            } else {
                #pragma unroll
                for (int j = 0; j < BN; j++) Ps[row * BN + j] = __float2bfloat16(0.f);
            }
        }
        __syncthreads();

        // O += P @ V  (warp handles rows [16w,16w+16), 8 col-tiles of D)
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> p_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> v_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_acc[8];
            #pragma unroll
            for (int c = 0; c < 8; c++)
                wmma::load_matrix_sync(o_acc[c], &Os[16 * warp * D + 16 * c], D, wmma::mem_row_major);
            #pragma unroll
            for (int k = 0; k < BN / 16; k++) {
                wmma::load_matrix_sync(p_frag, &Ps[16 * warp * BN + 16 * k], BN);
                #pragma unroll
                for (int c = 0; c < 8; c++) {
                    wmma::load_matrix_sync(v_frag, &Vs[16 * k * D + 16 * c], D);
                    wmma::mma_sync(o_acc[c], p_frag, v_frag, o_acc[c]);
                }
            }
            #pragma unroll
            for (int c = 0; c < 8; c++)
                wmma::store_matrix_sync(&Os[16 * warp * D + 16 * c], o_acc[c], D, wmma::mem_row_major);
        }
        __syncthreads();
    }

    // Finalize: O = Os / l ; LSE = m + log(l)
    for (int i = tid; i < BM * D; i += NTHREAD) {
        int row = i / D;
        int col = i % D;
        int grow = q0 + row;
        if (grow < S) {
            float inv = 1.0f / ls[row];
            O[bh_base + (long)grow * D + col] = __float2bfloat16(Os[i] * inv);
        }
    }
    if (tid < BM) {
        int grow = q0 + tid;
        if (grow < S) {
            LSE[(long)(b * H + h) * S + grow] = ms[tid] + logf(ls[tid]);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int S = (int)Q.size(2);
    // D fixed = 128

    const __nv_bfloat16* Qp = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* Kp = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* Vp = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* Op = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSEp = static_cast<float*>(LSE.data_ptr());

    int num_qtiles = (S + BM - 1) / BM;
    dim3 grid(B * H * num_qtiles);
    dim3 block(NTHREAD);

    size_t smem_bytes = (size_t)BM * D * 2      // Qs
                      + (size_t)BN * D * 2      // Ks
                      + (size_t)BN * D * 2      // Vs
                      + (size_t)BM * BN * 2     // Ps
                      + (size_t)BM * BN * 4     // Ss
                      + (size_t)BM * D * 4      // Os
                      + (size_t)BM * 4 * 2;     // ms, ls

    static bool attr_set = false;
    if (!attr_set) {
        CUDA_CHECK(cudaFuncSetAttribute(attn_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));
        attr_set = true;
    }

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    attn_kernel<<<grid, block, smem_bytes, stream>>>(Qp, Kp, Vp, Op, LSEp, B, H, S, num_qtiles);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel::run);

}  // namespace mha_kernel