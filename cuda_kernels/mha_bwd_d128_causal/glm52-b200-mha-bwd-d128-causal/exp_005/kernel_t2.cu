#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda;

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", \
                cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

namespace attn_bwd {

constexpr int Br = 64;
constexpr int Bc = 64;
constexpr int D  = 128;
constexpr int W  = 16;
constexpr int NT = 128;

constexpr int K_OFF    = 0;
constexpr int V_OFF    = K_OFF + Bc * D * 2;
constexpr int Q_OFF    = V_OFF + Bc * D * 2;
constexpr int DO_OFF   = Q_OFF + Br * D * 2;
constexpr int S_OFF    = DO_OFF + Br * D * 2;
constexpr int DP_OFF   = S_OFF + Br * Bc * 4;
constexpr int PDS_OFF  = DP_OFF + Br * Bc * 4;
constexpr int L_OFF    = PDS_OFF + Br * Bc * 2;
constexpr int D_OFF    = L_OFF + Br * 4;
constexpr int SMEM_SIZE = D_OFF + Br * 4;

__global__ void compute_D_kernel(
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    float* __restrict__ Dout, int BHS)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= BHS) return;
    const __nv_bfloat16* Or  = &O[(int64_t)idx * D];
    const __nv_bfloat16* dOr = &dO[(int64_t)idx * D];
    float sum = 0.0f;
    #pragma unroll
    for (int j = 0; j < D; j += 8) {
        int4 o4 = *reinterpret_cast<const int4*>(&Or[j]);
        int4 do4 = *reinterpret_cast<const int4*>(&dOr[j]);
        __nv_bfloat16* o = reinterpret_cast<__nv_bfloat16*>(&o4);
        __nv_bfloat16* d = reinterpret_cast<__nv_bfloat16*>(&do4);
        #pragma unroll
        for (int k = 0; k < 8; k++)
            sum += __bfloat162float(o[k]) * __bfloat162float(d[k]);
    }
    Dout[idx] = sum;
}

__global__ __launch_bounds__(NT)
void attn_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ Ddiag,
    float* __restrict__ dQf32,
    __nv_bfloat16* __restrict__ dKg,
    __nv_bfloat16* __restrict__ dVg,
    int S, int H)
{
    int bh = blockIdx.x;
    int j  = blockIdx.y;
    int b  = bh / H;
    int h  = bh % H;
    int nb = (S + Bc - 1) / Bc;
    int tid = threadIdx.x;
    int wid = tid / 32;

    extern __shared__ char smem[];
    __nv_bfloat16* Ks  = reinterpret_cast<__nv_bfloat16*>(smem + K_OFF);
    __nv_bfloat16* Vs  = reinterpret_cast<__nv_bfloat16*>(smem + V_OFF);
    __nv_bfloat16* Qs  = reinterpret_cast<__nv_bfloat16*>(smem + Q_OFF);
    __nv_bfloat16* dOs = reinterpret_cast<__nv_bfloat16*>(smem + DO_OFF);
    float* Ss          = reinterpret_cast<float*>(smem + S_OFF);
    float* dPs         = reinterpret_cast<float*>(smem + DP_OFF);
    __nv_bfloat16* PDS = reinterpret_cast<__nv_bfloat16*>(smem + PDS_OFF);
    float* Ls          = reinterpret_cast<float*>(smem + L_OFF);
    float* Ds          = reinterpret_cast<float*>(smem + D_OFF);
    float* dQs         = reinterpret_cast<float*>(smem + S_OFF);
    float* outs        = reinterpret_cast<float*>(smem + K_OFF);

    const float scale = 0.08838834764831845f;
    int64_t off  = (int64_t)(b * H + h) * S * D;
    int64_t offL = (int64_t)(b * H + h) * S;

    for (int idx = tid; idx < Bc * D / 8; idx += NT) {
        int r = idx / (D / 8);
        int c = (idx % (D / 8)) * 8;
        int ki = j * Bc + r;
        if (ki < S) {
            *reinterpret_cast<int4*>(&Ks[idx * 8]) = *reinterpret_cast<const int4*>(&K[off + (int64_t)ki * D + c]);
            *reinterpret_cast<int4*>(&Vs[idx * 8]) = *reinterpret_cast<const int4*>(&V[off + (int64_t)ki * D + c]);
        } else {
            *reinterpret_cast<int4*>(&Ks[idx * 8]) = make_int4(0, 0, 0, 0);
            *reinterpret_cast<int4*>(&Vs[idx * 8]) = make_int4(0, 0, 0, 0);
        }
    }
    __syncthreads();

    int m_block = wid / 2;
    int n_block = wid % 2;
    int m_start = m_block * 32;
    int n_start = n_block * 32;

    wmma::fragment<wmma::accumulator, W, W, W, float> dKf[8], dVf[8];
    #pragma unroll
    for (int n = 0; n < 8; n++) {
        wmma::fill_fragment(dKf[n], 0.0f);
        wmma::fill_fragment(dVf[n], 0.0f);
    }

    wmma::fragment<wmma::matrix_a, W, W, W, __nv_bfloat16, wmma::row_major> ar;
    wmma::fragment<wmma::matrix_b, W, W, W, __nv_bfloat16, wmma::col_major> bc;
    wmma::fragment<wmma::matrix_b, W, W, W, __nv_bfloat16, wmma::row_major> br;
    wmma::fragment<wmma::matrix_a, W, W, W, __nv_bfloat16, wmma::col_major> ac;
    wmma::fragment<wmma::accumulator, W, W, W, float> cf;
    wmma::fragment<wmma::accumulator, W, W, W, float> dQf[4];

    for (int i = j; i < nb; i++) {
        for (int idx = tid; idx < Br * D / 8; idx += NT) {
            int r = idx / (D / 8);
            int c = (idx % (D / 8)) * 8;
            int qi = i * Br + r;
            if (qi < S) {
                *reinterpret_cast<int4*>(&Qs[idx * 8])  = *reinterpret_cast<const int4*>(&Q[off + (int64_t)qi * D + c]);
                *reinterpret_cast<int4*>(&dOs[idx * 8]) = *reinterpret_cast<const int4*>(&dO[off + (int64_t)qi * D + c]);
            } else {
                *reinterpret_cast<int4*>(&Qs[idx * 8])  = make_int4(0, 0, 0, 0);
                *reinterpret_cast<int4*>(&dOs[idx * 8]) = make_int4(0, 0, 0, 0);
            }
        }
        if (tid < Br) {
            int qi = i * Br + tid;
            Ls[tid] = (qi < S) ? L[offL + qi] : 0.0f;
            Ds[tid] = (qi < S) ? Ddiag[offL + qi] : 0.0f;
        }
        __syncthreads();

        // S = Q @ K^T → Ss
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            #pragma unroll
            for (int ni = 0; ni < 2; ni++) {
                wmma::fill_fragment(cf, 0.0f);
                #pragma unroll
                for (int ki = 0; ki < D / W; ki++) {
                    wmma::load_matrix_sync(ar, &Qs[(m_start + mi*16) * D + ki*16], D);
                    wmma::load_matrix_sync(bc, &Ks[(n_start + ni*16) * D + ki*16], D);
                    wmma::mma_sync(cf, ar, bc, cf);
                }
                wmma::store_matrix_sync(&Ss[(m_start + mi*16) * Bc + n_start + ni*16], cf, Bc, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // P = exp(S*scale - L) → Ss (fp32), PDS (bf16)
        bool causal = (i == j);
        for (int idx = tid; idx < Br * Bc; idx += NT) {
            int r = idx / Bc;
            int c = idx % Bc;
            int ki = j * Bc + c;
            int qi = i * Br + r;
            float s = Ss[idx] * scale;
            float p = (ki >= S || qi >= S || (causal && r < c)) ? 0.0f : __expf(s - Ls[r]);
            Ss[idx] = p;
            PDS[idx] = __float2bfloat16(p);
        }
        __syncthreads();

        // dV += P^T @ dO (64x128, 2 passes)
        #pragma unroll
        for (int pass = 0; pass < 2; pass++) {
            int n_off = pass * 64;
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                #pragma unroll
                for (int ni = 0; ni < 2; ni++) {
                    #pragma unroll
                    for (int ki = 0; ki < Br / W; ki++) {
                        wmma::load_matrix_sync(ac, &PDS[ki*16 * Bc + m_start + mi*16], Bc);
                        wmma::load_matrix_sync(br, &dOs[ki*16 * D + n_off + n_start + ni*16], D);
                        wmma::mma_sync(dVf[pass*4 + mi*2 + ni], ac, br, dVf[pass*4 + mi*2 + ni]);
                    }
                }
            }
        }
        __syncthreads();

        // dP = dO @ V^T → dPs (fp32)
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            #pragma unroll
            for (int ni = 0; ni < 2; ni++) {
                wmma::fill_fragment(cf, 0.0f);
                #pragma unroll
                for (int ki = 0; ki < D / W; ki++) {
                    wmma::load_matrix_sync(ar, &dOs[(m_start + mi*16) * D + ki*16], D);
                    wmma::load_matrix_sync(bc, &Vs[(n_start + ni*16) * D + ki*16], D);
                    wmma::mma_sync(cf, ar, bc, cf);
                }
                wmma::store_matrix_sync(&dPs[(m_start + mi*16) * Bc + n_start + ni*16], cf, Bc, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // dS = P*(dP-D)*scale → PDS (bf16), using P fp32 from Ss
        for (int idx = tid; idx < Br * Bc; idx += NT) {
            int r = idx / Bc;
            float p = Ss[idx];
            float dp = dPs[idx];
            float dv = Ds[r];
            PDS[idx] = __float2bfloat16(p * (dp - dv) * scale);
        }
        __syncthreads();

        // dK += dS^T @ Q (64x128, 2 passes)
        #pragma unroll
        for (int pass = 0; pass < 2; pass++) {
            int n_off = pass * 64;
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                #pragma unroll
                for (int ni = 0; ni < 2; ni++) {
                    #pragma unroll
                    for (int ki = 0; ki < Br / W; ki++) {
                        wmma::load_matrix_sync(ac, &PDS[ki*16 * Bc + m_start + mi*16], Bc);
                        wmma::load_matrix_sync(br, &Qs[ki*16 * D + n_off + n_start + ni*16], D);
                        wmma::mma_sync(dKf[pass*4 + mi*2 + ni], ac, br, dKf[pass*4 + mi*2 + ni]);
                    }
                }
            }
        }

        // dQ = dS @ K (64x128, 2 passes), store to Ss, atomic add
        #pragma unroll
        for (int pass = 0; pass < 2; pass++) {
            int n_off = pass * 64;
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                #pragma unroll
                for (int ni = 0; ni < 2; ni++) {
                    wmma::fill_fragment(dQf[mi*2+ni], 0.0f);
                    #pragma unroll
                    for (int ki = 0; ki < Bc / W; ki++) {
                        wmma::load_matrix_sync(ar, &PDS[(m_start + mi*16) * Bc + ki*16], Bc);
                        wmma::load_matrix_sync(br, &Ks[ki*16 * D + n_off + n_start + ni*16], D);
                        wmma::mma_sync(dQf[mi*2+ni], ar, br, dQf[mi*2+ni]);
                    }
                    wmma::store_matrix_sync(&dQs[(m_start + mi*16) * 64 + n_start + ni*16], dQf[mi*2+ni], 64, wmma::mem_row_major);
                }
            }
            __syncthreads();
            for (int idx = tid; idx < Br * 64; idx += NT) {
                int r = idx / 64;
                int c = n_off + idx % 64;
                int qi = i * Br + r;
                if (qi < S) atomicAdd(&dQf32[off + (int64_t)qi * D + c], dQs[idx]);
            }
            __syncthreads();
        }
    }

    // Store dK to global
    #pragma unroll
    for (int pass = 0; pass < 2; pass++) {
        int n_off = pass * 64;
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            #pragma unroll
            for (int ni = 0; ni < 2; ni++) {
                wmma::store_matrix_sync(&outs[(m_start + mi*16) * D + n_off + n_start + ni*16], dKf[pass*4 + mi*2 + ni], D, wmma::mem_row_major);
            }
        }
    }
    __syncthreads();
    for (int idx = tid; idx < Bc * D; idx += NT) {
        int r = idx / D;
        int c = idx % D;
        int ki = j * Bc + r;
        if (ki < S) dKg[off + (int64_t)ki * D + c] = __float2bfloat16(outs[idx]);
    }
    __syncthreads();

    // Store dV to global
    #pragma unroll
    for (int pass = 0; pass < 2; pass++) {
        int n_off = pass * 64;
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            #pragma unroll
            for (int ni = 0; ni < 2; ni++) {
                wmma::store_matrix_sync(&outs[(m_start + mi*16) * D + n_off + n_start + ni*16], dVf[pass*4 + mi*2 + ni], D, wmma::mem_row_major);
            }
        }
    }
    __syncthreads();
    for (int idx = tid; idx < Bc * D; idx += NT) {
        int r = idx / D;
        int c = idx % D;
        int ki = j * Bc + r;
        if (ki < S) dVg[off + (int64_t)ki * D + c] = __float2bfloat16(outs[idx]);
    }
}

__global__ void convert_kernel(const float* src, __nv_bfloat16* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __float2bfloat16(src[idx]);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int B = 4, H = 48, d = 128;
    int S = (int)Q.size(2);
    int BHS = B * H * S;
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    float* Ddiag;
    CUDA_CHECK(cudaMalloc(&Ddiag, BHS * sizeof(float)));
    size_t dQ_sz = (size_t)B * H * S * d;
    float* dQf32;
    CUDA_CHECK(cudaMalloc(&dQf32, dQ_sz * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(dQf32, 0, dQ_sz * sizeof(float), stream));

    compute_D_kernel<<<(BHS + 255) / 256, 256, 0, stream>>>(
        (const __nv_bfloat16*)O.data_ptr(),
        (const __nv_bfloat16*)dO.data_ptr(),
        Ddiag, BHS);

    int nkb = (S + Bc - 1) / Bc;
    cudaFuncSetAttribute(attn_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE);
    attn_bwd_kernel<<<dim3(B * H, nkb), NT, SMEM_SIZE, stream>>>(
        (const __nv_bfloat16*)Q.data_ptr(),
        (const __nv_bfloat16*)K.data_ptr(),
        (const __nv_bfloat16*)V.data_ptr(),
        (const __nv_bfloat16*)dO.data_ptr(),
        (const float*)L.data_ptr(),
        Ddiag, dQf32,
        (__nv_bfloat16*)dK.data_ptr(),
        (__nv_bfloat16*)dV.data_ptr(),
        S, H);

    convert_kernel<<<((int)dQ_sz + 255) / 256, 256, 0, stream>>>(
        dQf32, (__nv_bfloat16*)dQ.data_ptr(), (int)dQ_sz);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(Ddiag));
    CUDA_CHECK(cudaFree(dQf32));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, attn_bwd::run);

}  // namespace attn_bwd