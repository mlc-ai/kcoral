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

constexpr int Br = 128;
constexpr int Bc = 64;
constexpr int D  = 128;
constexpr int W  = 16;
constexpr int NT = 256;

constexpr int K_OFF   = 0;
constexpr int V_OFF   = K_OFF + Bc * D * 2;
constexpr int Q_OFF   = V_OFF + Bc * D * 2;
constexpr int DO_OFF  = Q_OFF + Br * D * 2;
constexpr int SDP_OFF = DO_OFF + Br * D * 2;
constexpr int PDS_OFF = SDP_OFF + Br * Bc * 4;
constexpr int L_OFF   = PDS_OFF + Br * Bc * 2;
constexpr int D_OFF   = L_OFF + Br * 4;
constexpr int SMEM_SIZE = D_OFF + Br * 4;

__device__ __forceinline__ void cp_async_16(void* smem_dst, const void* gmem_src) {
    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem_src));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n");
}

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

__global__ __launch_bounds__(NT, 1)
void dK_dV_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ Ddiag,
    __nv_bfloat16* __restrict__ dK_g,
    __nv_bfloat16* __restrict__ dV_g,
    int S, int H)
{
    int bh = blockIdx.x;
    int j  = blockIdx.y;
    int b  = bh / H;
    int h  = bh % H;
    int nb_q = (S + Br - 1) / Br;
    int tid = threadIdx.x;
    int wid = tid / 32;

    extern __shared__ char smem[];
    __nv_bfloat16* Ks  = reinterpret_cast<__nv_bfloat16*>(smem + K_OFF);
    __nv_bfloat16* Vs  = reinterpret_cast<__nv_bfloat16*>(smem + V_OFF);
    __nv_bfloat16* Qs  = reinterpret_cast<__nv_bfloat16*>(smem + Q_OFF);
    __nv_bfloat16* dOs = reinterpret_cast<__nv_bfloat16*>(smem + DO_OFF);
    float* Ss_dP       = reinterpret_cast<float*>(smem + SDP_OFF);
    __nv_bfloat16* PDS = reinterpret_cast<__nv_bfloat16*>(smem + PDS_OFF);
    float* Ls          = reinterpret_cast<float*>(smem + L_OFF);
    float* Ds          = reinterpret_cast<float*>(smem + D_OFF);
    float* outs        = reinterpret_cast<float*>(smem);

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

    wmma::fragment<wmma::accumulator, W, W, W, float> dKf[4], dVf[4];
    #pragma unroll
    for (int m = 0; m < 4; m++) {
        wmma::fill_fragment(dKf[m], 0.0f);
        wmma::fill_fragment(dVf[m], 0.0f);
    }

    wmma::fragment<wmma::matrix_a, W, W, W, __nv_bfloat16, wmma::row_major> a_row;
    wmma::fragment<wmma::matrix_b, W, W, W, __nv_bfloat16, wmma::col_major> b_col;
    wmma::fragment<wmma::matrix_b, W, W, W, __nv_bfloat16, wmma::row_major> b_row;
    wmma::fragment<wmma::matrix_a, W, W, W, __nv_bfloat16, wmma::col_major> a_col;
    wmma::fragment<wmma::accumulator, W, W, W, float> cf;

    int i_start = j / 2;

    for (int i = i_start; i < nb_q; i++) {
        for (int idx = tid; idx < Br * D / 8; idx += NT) {
            int r = idx / (D / 8);
            int c = (idx % (D / 8)) * 8;
            int qi = i * Br + r;
            if (qi < S) {
                cp_async_16(&Qs[idx * 8], &Q[off + (int64_t)qi * D + c]);
                cp_async_16(&dOs[idx * 8], &dO[off + (int64_t)qi * D + c]);
            } else {
                *reinterpret_cast<int4*>(&Qs[idx * 8]) = make_int4(0, 0, 0, 0);
                *reinterpret_cast<int4*>(&dOs[idx * 8]) = make_int4(0, 0, 0, 0);
            }
        }
        if (tid < Br) {
            int qi = i * Br + tid;
            Ls[tid] = (qi < S) ? L[offL + qi] : 0.0f;
            Ds[tid] = (qi < S) ? Ddiag[offL + qi] : 0.0f;
        }
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();

        // S = Q @ K^T -> Ss_dP
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            wmma::fill_fragment(cf, 0.0f);
            #pragma unroll
            for (int k = 0; k < D / W; k++) {
                wmma::load_matrix_sync(a_row, &Qs[wid * 16 * D + k * 16], D);
                wmma::load_matrix_sync(b_col, &Ks[n * 16 * D + k * 16], D);
                wmma::mma_sync(cf, a_row, b_col, cf);
            }
            wmma::store_matrix_sync(&Ss_dP[wid * 16 * Bc + n * 16], cf, Bc, wmma::mem_row_major);
        }
        __syncthreads();

        // P = exp(S*scale - L) -> PDS (bf16)
        for (int idx = tid; idx < Br * Bc; idx += NT) {
            int r = idx / Bc;
            int c = idx % Bc;
            int qi = i * Br + r;
            int kj = j * Bc + c;
            float s = Ss_dP[idx] * scale;
            float p = (kj >= S || qi >= S || qi < kj) ? 0.0f : __expf(s - Ls[r]);
            PDS[idx] = __float2bfloat16(p);
        }
        __syncthreads();

        // dV += P^T @ dO (BEFORE dS overwrites PDS)
        #pragma unroll
        for (int m = 0; m < 4; m++) {
            #pragma unroll
            for (int k = 0; k < Br / W; k++) {
                wmma::load_matrix_sync(a_col, &PDS[k * 16 * Bc + m * 16], Bc);
                wmma::load_matrix_sync(b_row, &dOs[k * 16 * D + wid * 16], D);
                wmma::mma_sync(dVf[m], a_col, b_row, dVf[m]);
            }
        }

        // dP = dO @ V^T -> Ss_dP (overwrite S)
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            wmma::fill_fragment(cf, 0.0f);
            #pragma unroll
            for (int k = 0; k < D / W; k++) {
                wmma::load_matrix_sync(a_row, &dOs[wid * 16 * D + k * 16], D);
                wmma::load_matrix_sync(b_col, &Vs[n * 16 * D + k * 16], D);
                wmma::mma_sync(cf, a_row, b_col, cf);
            }
            wmma::store_matrix_sync(&Ss_dP[wid * 16 * Bc + n * 16], cf, Bc, wmma::mem_row_major);
        }
        __syncthreads();

        // dS = P*(dP-D)*scale -> PDS (overwrite P, using bf16 P from PDS)
        for (int idx = tid; idx < Br * Bc; idx += NT) {
            int r = idx / Bc;
            float p = __bfloat162float(PDS[idx]);
            float dp = Ss_dP[idx];
            float dv = Ds[r];
            PDS[idx] = __float2bfloat16(p * (dp - dv) * scale);
        }
        __syncthreads();

        // dK += dS^T @ Q
        #pragma unroll
        for (int m = 0; m < 4; m++) {
            #pragma unroll
            for (int k = 0; k < Br / W; k++) {
                wmma::load_matrix_sync(a_col, &PDS[k * 16 * Bc + m * 16], Bc);
                wmma::load_matrix_sync(b_row, &Qs[k * 16 * D + wid * 16], D);
                wmma::mma_sync(dKf[m], a_col, b_row, dKf[m]);
            }
        }
        __syncthreads();
    }

    // Store dK
    #pragma unroll
    for (int m = 0; m < 4; m++)
        wmma::store_matrix_sync(&outs[m * 16 * D + wid * 16], dKf[m], D, wmma::mem_row_major);
    __syncthreads();
    for (int idx = tid; idx < Bc * D; idx += NT) {
        int r = idx / D;
        int c = idx % D;
        int ki = j * Bc + r;
        if (ki < S)
            dK_g[off + (int64_t)ki * D + c] = __float2bfloat16(outs[idx]);
    }
    __syncthreads();

    // Store dV
    #pragma unroll
    for (int m = 0; m < 4; m++)
        wmma::store_matrix_sync(&outs[m * 16 * D + wid * 16], dVf[m], D, wmma::mem_row_major);
    __syncthreads();
    for (int idx = tid; idx < Bc * D; idx += NT) {
        int r = idx / D;
        int c = idx % D;
        int ki = j * Bc + r;
        if (ki < S)
            dV_g[off + (int64_t)ki * D + c] = __float2bfloat16(outs[idx]);
    }
}

__global__ __launch_bounds__(NT, 1)
void dQ_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ Ddiag,
    __nv_bfloat16* __restrict__ dQ_g,
    int S, int H)
{
    int bh = blockIdx.x;
    int i  = blockIdx.y;
    int b  = bh / H;
    int h  = bh % H;
    int nb_kv = (S + Bc - 1) / Bc;
    int tid = threadIdx.x;
    int wid = tid / 32;

    extern __shared__ char smem[];
    __nv_bfloat16* Ks  = reinterpret_cast<__nv_bfloat16*>(smem + K_OFF);
    __nv_bfloat16* Vs  = reinterpret_cast<__nv_bfloat16*>(smem + V_OFF);
    __nv_bfloat16* Qs  = reinterpret_cast<__nv_bfloat16*>(smem + Q_OFF);
    __nv_bfloat16* dOs = reinterpret_cast<__nv_bfloat16*>(smem + DO_OFF);
    float* Ss_dP       = reinterpret_cast<float*>(smem + SDP_OFF);
    __nv_bfloat16* PDS = reinterpret_cast<__nv_bfloat16*>(smem + PDS_OFF);
    float* Ls          = reinterpret_cast<float*>(smem + L_OFF);
    float* Ds          = reinterpret_cast<float*>(smem + D_OFF);
    float* outs        = reinterpret_cast<float*>(smem);

    const float scale = 0.08838834764831845f;
    int64_t off  = (int64_t)(b * H + h) * S * D;
    int64_t offL = (int64_t)(b * H + h) * S;

    for (int idx = tid; idx < Br * D / 8; idx += NT) {
        int r = idx / (D / 8);
        int c = (idx % (D / 8)) * 8;
        int qi = i * Br + r;
        if (qi < S) {
            *reinterpret_cast<int4*>(&Qs[idx * 8]) = *reinterpret_cast<const int4*>(&Q[off + (int64_t)qi * D + c]);
            *reinterpret_cast<int4*>(&dOs[idx * 8]) = *reinterpret_cast<const int4*>(&dO[off + (int64_t)qi * D + c]);
        } else {
            *reinterpret_cast<int4*>(&Qs[idx * 8]) = make_int4(0, 0, 0, 0);
            *reinterpret_cast<int4*>(&dOs[idx * 8]) = make_int4(0, 0, 0, 0);
        }
    }
    if (tid < Br) {
        int qi = i * Br + tid;
        Ls[tid] = (qi < S) ? L[offL + qi] : 0.0f;
        Ds[tid] = (qi < S) ? Ddiag[offL + qi] : 0.0f;
    }
    __syncthreads();

    wmma::fragment<wmma::accumulator, W, W, W, float> dQf[8];
    #pragma unroll
    for (int n = 0; n < 8; n++)
        wmma::fill_fragment(dQf[n], 0.0f);

    wmma::fragment<wmma::matrix_a, W, W, W, __nv_bfloat16, wmma::row_major> a_row;
    wmma::fragment<wmma::matrix_b, W, W, W, __nv_bfloat16, wmma::col_major> b_col;
    wmma::fragment<wmma::matrix_b, W, W, W, __nv_bfloat16, wmma::row_major> b_row;
    wmma::fragment<wmma::accumulator, W, W, W, float> cf;

    int j_end = min(2 * i + 2, nb_kv);
    for (int j = 0; j < j_end; j++) {
        for (int idx = tid; idx < Bc * D / 8; idx += NT) {
            int r = idx / (D / 8);
            int c = (idx % (D / 8)) * 8;
            int ki = j * Bc + r;
            if (ki < S) {
                cp_async_16(&Ks[idx * 8], &K[off + (int64_t)ki * D + c]);
                cp_async_16(&Vs[idx * 8], &V[off + (int64_t)ki * D + c]);
            } else {
                *reinterpret_cast<int4*>(&Ks[idx * 8]) = make_int4(0, 0, 0, 0);
                *reinterpret_cast<int4*>(&Vs[idx * 8]) = make_int4(0, 0, 0, 0);
            }
        }
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();

        // S = Q @ K^T -> Ss_dP
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            wmma::fill_fragment(cf, 0.0f);
            #pragma unroll
            for (int k = 0; k < D / W; k++) {
                wmma::load_matrix_sync(a_row, &Qs[wid * 16 * D + k * 16], D);
                wmma::load_matrix_sync(b_col, &Ks[n * 16 * D + k * 16], D);
                wmma::mma_sync(cf, a_row, b_col, cf);
            }
            wmma::store_matrix_sync(&Ss_dP[wid * 16 * Bc + n * 16], cf, Bc, wmma::mem_row_major);
        }
        __syncthreads();

        // P = exp(S*scale - L) -> PDS (bf16)
        for (int idx = tid; idx < Br * Bc; idx += NT) {
            int r = idx / Bc;
            int c = idx % Bc;
            int qi = i * Br + r;
            int kj = j * Bc + c;
            float s = Ss_dP[idx] * scale;
            float p = (kj >= S || qi >= S || qi < kj) ? 0.0f : __expf(s - Ls[r]);
            PDS[idx] = __float2bfloat16(p);
        }
        __syncthreads();

        // dP = dO @ V^T -> Ss_dP
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            wmma::fill_fragment(cf, 0.0f);
            #pragma unroll
            for (int k = 0; k < D / W; k++) {
                wmma::load_matrix_sync(a_row, &dOs[wid * 16 * D + k * 16], D);
                wmma::load_matrix_sync(b_col, &Vs[n * 16 * D + k * 16], D);
                wmma::mma_sync(cf, a_row, b_col, cf);
            }
            wmma::store_matrix_sync(&Ss_dP[wid * 16 * Bc + n * 16], cf, Bc, wmma::mem_row_major);
        }
        __syncthreads();

        // dS = P*(dP-D)*scale -> PDS
        for (int idx = tid; idx < Br * Bc; idx += NT) {
            int r = idx / Bc;
            float p = __bfloat162float(PDS[idx]);
            float dp = Ss_dP[idx];
            float dv = Ds[r];
            PDS[idx] = __float2bfloat16(p * (dp - dv) * scale);
        }
        __syncthreads();

        // dQ += dS @ K
        #pragma unroll
        for (int n = 0; n < 8; n++) {
            #pragma unroll
            for (int k = 0; k < Bc / W; k++) {
                wmma::load_matrix_sync(a_row, &PDS[wid * 16 * Bc + k * 16], Bc);
                wmma::load_matrix_sync(b_row, &Ks[k * 16 * D + n * 16], D);
                wmma::mma_sync(dQf[n], a_row, b_row, dQf[n]);
            }
        }
        __syncthreads();
    }

    // Store dQ
    #pragma unroll
    for (int n = 0; n < 8; n++)
        wmma::store_matrix_sync(&outs[wid * 16 * D + n * 16], dQf[n], D, wmma::mem_row_major);
    __syncthreads();
    for (int idx = tid; idx < Br * D; idx += NT) {
        int r = idx / D;
        int c = idx % D;
        int qi = i * Br + r;
        if (qi < S)
            dQ_g[off + (int64_t)qi * D + c] = __float2bfloat16(outs[idx]);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int B = 4, H = 48;
    int S = (int)Q.size(2);
    int BHS = B * H * S;
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    float* Ddiag;
    CUDA_CHECK(cudaMalloc(&Ddiag, BHS * sizeof(float)));

    compute_D_kernel<<<(BHS + 255) / 256, 256, 0, stream>>>(
        (const __nv_bfloat16*)O.data_ptr(),
        (const __nv_bfloat16*)dO.data_ptr(),
        Ddiag, BHS);

    int nkb = (S + Bc - 1) / Bc;
    int nqb = (S + Br - 1) / Br;

    cudaFuncSetAttribute(dK_dV_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE);
    dK_dV_kernel<<<dim3(B * H, nkb), NT, SMEM_SIZE, stream>>>(
        (const __nv_bfloat16*)Q.data_ptr(),
        (const __nv_bfloat16*)K.data_ptr(),
        (const __nv_bfloat16*)V.data_ptr(),
        (const __nv_bfloat16*)dO.data_ptr(),
        (const float*)L.data_ptr(),
        Ddiag,
        (__nv_bfloat16*)dK.data_ptr(),
        (__nv_bfloat16*)dV.data_ptr(),
        S, H);

    cudaFuncSetAttribute(dQ_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE);
    dQ_kernel<<<dim3(B * H, nqb), NT, SMEM_SIZE, stream>>>(
        (const __nv_bfloat16*)Q.data_ptr(),
        (const __nv_bfloat16*)K.data_ptr(),
        (const __nv_bfloat16*)V.data_ptr(),
        (const __nv_bfloat16*)dO.data_ptr(),
        (const float*)L.data_ptr(),
        Ddiag,
        (__nv_bfloat16*)dQ.data_ptr(),
        S, H);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(Ddiag));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, attn_bwd::run);

}  // namespace attn_bwd