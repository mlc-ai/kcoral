#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cmath>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda;

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while (0)

namespace attention_bwd {

constexpr int D = 128;
constexpr int BQ = 64;
constexpr int BKV = 64;
constexpr int NW = 4;
constexpr int NT = 128;
constexpr int WMM = 16, WMN = 16, WMK = 16;

constexpr int SQ = BQ*D*2 + BQ*D*2 + BKV*D*2 + BKV*D*2 + BQ*BKV*4 + BQ*BKV*4 + BQ*BKV*2 + BQ*8 + NW*WMM*WMN*4;
constexpr int SKV = BKV*D*2 + BKV*D*2 + BQ*D*2 + BQ*D*2 + BQ*BKV*4 + BQ*BKV*4 + BQ*BKV*2 + BQ*BKV*2 + BQ*8 + NW*WMM*WMN*4;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int o = 16; o > 0; o /= 2) val += __shfl_xor_sync(0xffffffff, val, o);
    return val;
}

__global__ void compute_D_kernel(const __nv_bfloat16* __restrict__ O, const __nv_bfloat16* __restrict__ dO, float* __restrict__ Dd, int tr) {
    int r = blockIdx.x;
    if (r >= tr) return;
    int t = threadIdx.x;
    const __nv_bfloat16 *Op = O + (size_t)r * D, *dOp = dO + (size_t)r * D;
    float s = 0.0f;
    for (int i = t; i < D; i += blockDim.x) s += __bfloat162float(Op[i]) * __bfloat162float(dOp[i]);
    s = warp_reduce_sum(s);
    __shared__ float ws[4];
    int w = t / 32, l = t % 32;
    if (l == 0) ws[w] = s;
    __syncthreads();
    if (t == 0) { float T = 0; for (int i = 0; i < blockDim.x/32; i++) T += ws[i]; Dd[r] = T; }
}

__global__ __launch_bounds__(NT, 3) void dQ_kernel(
    const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V, const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L, const float* __restrict__ Dd,
    __nv_bfloat16* __restrict__ dQ_out, int B, int H, int S) {

    extern __shared__ char sm[];
    __nv_bfloat16 *Qi = reinterpret_cast<__nv_bfloat16*>(sm);
    __nv_bfloat16 *dOi = Qi + BQ * D;
    __nv_bfloat16 *Kj = dOi + BQ * D;
    __nv_bfloat16 *Vj = Kj + BKV * D;
    float *S_sm = reinterpret_cast<float*>(Vj + BKV * D);
    float *P_sm = S_sm + BQ * BKV;
    __nv_bfloat16 *dSbf = reinterpret_cast<__nv_bfloat16*>(P_sm + BQ * BKV);
    float *Li = reinterpret_cast<float*>(dSbf + BQ * BKV);
    float *Di = Li + BQ;
    float *stg = Di + BQ;

    int t = threadIdx.x, w = t / 32, l = t % 32;
    int nqb = (S + BQ - 1) / BQ;
    int bh = blockIdx.x / nqb, qb = blockIdx.x % nqb;
    int b = bh / H, h = bh % H, qs = qb * BQ;
    size_t bho = (size_t)(b * H + h) * S * D;
    const __nv_bfloat16 *Qb = Q + bho, *Kb = K + bho, *Vb = V + bho, *dOb = dO + bho;
    const float *Lb = L + (size_t)(b*H+h)*S, *Db = Dd + (size_t)(b*H+h)*S;
    __nv_bfloat16 *dQb = dQ_out + bho;

    {
        int4 *qv = reinterpret_cast<int4*>(Qi), *dv = reinterpret_cast<int4*>(dOi);
        int tv = BQ * D / 8;
        for (int i = t; i < tv; i += NT) {
            int r = i / (D/8), cv = i % (D/8), gr = qs + r;
            if (gr < S) {
                qv[i] = *reinterpret_cast<const int4*>(&Qb[(size_t)gr*D + cv*8]);
                dv[i] = *reinterpret_cast<const int4*>(&dOb[(size_t)gr*D + cv*8]);
            } else { qv[i] = make_int4(0,0,0,0); dv[i] = make_int4(0,0,0,0); }
        }
    }
    if (t < BQ) { int gr = qs + t; Li[t] = (gr < S) ? Lb[gr] : 0.0f; Di[t] = (gr < S) ? Db[gr] : 0.0f; }
    __syncthreads();

    wmma::fragment<wmma::accumulator, WMM, WMN, WMK, float> dQf[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) wmma::fill_fragment(dQf[i], 0.0f);

    const float scale = 0.08838834764f;

    for (int kvs = 0; kvs < S; kvs += BKV) {
        {
            int4 *kv = reinterpret_cast<int4*>(Kj), *vv = reinterpret_cast<int4*>(Vj);
            int tv = BKV * D / 8;
            for (int i = t; i < tv; i += NT) {
                int r = i / (D/8), cv = i % (D/8), gr = kvs + r;
                if (gr < S) {
                    kv[i] = *reinterpret_cast<const int4*>(&Kb[(size_t)gr*D + cv*8]);
                    vv[i] = *reinterpret_cast<const int4*>(&Vb[(size_t)gr*D + cv*8]);
                } else { kv[i] = make_int4(0,0,0,0); vv[i] = make_int4(0,0,0,0); }
            }
        }
        __syncthreads();

        {
            int wr = w / 2, wcb = (w % 2) * 2;
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                int ct = wcb + j;
                wmma::fragment<wmma::accumulator, WMM, WMN, WMK, float> cf;
                wmma::fill_fragment(cf, 0.0f);
                #pragma unroll
                for (int kk = 0; kk < D/WMK; kk++) {
                    wmma::fragment<wmma::matrix_a, WMM, WMN, WMK, __nv_bfloat16, wmma::row_major> af;
                    wmma::fragment<wmma::matrix_b, WMM, WMN, WMK, __nv_bfloat16, wmma::col_major> bf;
                    wmma::load_matrix_sync(af, Qi + wr*16*D + kk*16, D);
                    wmma::load_matrix_sync(bf, Kj + ct*16*D + kk*16, D);
                    wmma::mma_sync(cf, af, bf, cf);
                }
                wmma::store_matrix_sync(S_sm + wr*16*BKV + ct*16, cf, BKV, wmma::mem_row_major);
            }
        }
        __syncthreads();

        for (int i = t; i < BQ * BKV; i += NT) {
            int r = i / BKV, c = i % BKV;
            if (qs + r < S && kvs + c < S) P_sm[i] = __expf(S_sm[i] * scale - Li[r]);
            else P_sm[i] = 0.0f;
        }
        __syncthreads();

        {
            int wr = w / 2, wcb = (w % 2) * 2;
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                int ct = wcb + j;
                wmma::fragment<wmma::accumulator, WMM, WMN, WMK, float> cf;
                wmma::fill_fragment(cf, 0.0f);
                #pragma unroll
                for (int kk = 0; kk < D/WMK; kk++) {
                    wmma::fragment<wmma::matrix_a, WMM, WMN, WMK, __nv_bfloat16, wmma::row_major> af;
                    wmma::fragment<wmma::matrix_b, WMM, WMN, WMK, __nv_bfloat16, wmma::col_major> bf;
                    wmma::load_matrix_sync(af, dOi + wr*16*D + kk*16, D);
                    wmma::load_matrix_sync(bf, Vj + ct*16*D + kk*16, D);
                    wmma::mma_sync(cf, af, bf, cf);
                }
                wmma::store_matrix_sync(S_sm + wr*16*BKV + ct*16, cf, BKV, wmma::mem_row_major);
            }
        }
        __syncthreads();

        for (int i = t; i < BQ * BKV; i += NT) {
            int r = i / BKV, c = i % BKV;
            if (qs + r < S && kvs + c < S) dSbf[i] = __float2bfloat16(P_sm[i] * (S_sm[i] - Di[r]) * scale);
            else dSbf[i] = __float2bfloat16(0.0f);
        }
        __syncthreads();

        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int ct = j;
            #pragma unroll
            for (int kk = 0; kk < BKV/WMK; kk++) {
                wmma::fragment<wmma::matrix_a, WMM, WMN, WMK, __nv_bfloat16, wmma::row_major> af;
                wmma::fragment<wmma::matrix_b, WMM, WMN, WMK, __nv_bfloat16, wmma::row_major> bf;
                wmma::load_matrix_sync(af, dSbf + w*16*BKV + kk*16, BKV);
                wmma::load_matrix_sync(bf, Kj + kk*16*D + ct*16, D);
                wmma::mma_sync(dQf[j], af, bf, dQf[j]);
            }
        }
        __syncthreads();
    }

    {
        float *st = stg + w * WMM * WMN;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            wmma::store_matrix_sync(st, dQf[j], WMN, wmma::mem_row_major);
            __syncwarp();
            int grb = qs + w * 16, gcb = j * 16;
            for (int i = l; i < WMM * WMN; i += 32) {
                int r = i / WMN, c = i % WMN, gr = grb + r;
                if (gr < S) dQb[(size_t)gr * D + gcb + c] = __float2bfloat16(st[i]);
            }
        }
    }
}

__global__ __launch_bounds__(NT, 3) void dKV_kernel(
    const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V, const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L, const float* __restrict__ Dd,
    __nv_bfloat16* __restrict__ dK_out, __nv_bfloat16* __restrict__ dV_out,
    int B, int H, int S) {

    extern __shared__ char sm[];
    __nv_bfloat16 *Kj = reinterpret_cast<__nv_bfloat16*>(sm);
    __nv_bfloat16 *Vj = Kj + BKV * D;
    __nv_bfloat16 *Qi = Vj + BKV * D;
    __nv_bfloat16 *dOi = Qi + BQ * D;
    float *S_sm = reinterpret_cast<float*>(dOi + BQ * D);
    float *P_sm = S_sm + BQ * BKV;
    __nv_bfloat16 *Pbf = reinterpret_cast<__nv_bfloat16*>(P_sm + BQ * BKV);
    __nv_bfloat16 *dSbf = Pbf + BQ * BKV;
    float *Li = reinterpret_cast<float*>(dSbf + BQ * BKV);
    float *Di = Li + BQ;
    float *stg = Di + BQ;

    int t = threadIdx.x, w = t / 32, l = t % 32;
    int nkb = (S + BKV - 1) / BKV;
    int bh = blockIdx.x / nkb, kb = blockIdx.x % nkb;
    int b = bh / H, h = bh % H, kvs = kb * BKV;
    size_t bho = (size_t)(b * H + h) * S * D;
    const __nv_bfloat16 *Qb = Q + bho, *Kb = K + bho, *Vb = V + bho, *dOb = dO + bho;
    const float *Lb = L + (size_t)(b*H+h)*S, *Db = Dd + (size_t)(b*H+h)*S;
    __nv_bfloat16 *dKb = dK_out + bho, *dVb = dV_out + bho;

    {
        int4 *kv = reinterpret_cast<int4*>(Kj), *vv = reinterpret_cast<int4*>(Vj);
        int tv = BKV * D / 8;
        for (int i = t; i < tv; i += NT) {
            int r = i / (D/8), cv = i % (D/8), gr = kvs + r;
            if (gr < S) {
                kv[i] = *reinterpret_cast<const int4*>(&Kb[(size_t)gr*D + cv*8]);
                vv[i] = *reinterpret_cast<const int4*>(&Vb[(size_t)gr*D + cv*8]);
            } else { kv[i] = make_int4(0,0,0,0); vv[i] = make_int4(0,0,0,0); }
        }
    }
    __syncthreads();

    wmma::fragment<wmma::accumulator, WMM, WMN, WMK, float> dKf[8], dVf[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) { wmma::fill_fragment(dKf[i], 0.0f); wmma::fill_fragment(dVf[i], 0.0f); }

    const float scale = 0.08838834764f;

    for (int qs = 0; qs < S; qs += BQ) {
        {
            int4 *qv = reinterpret_cast<int4*>(Qi), *dv = reinterpret_cast<int4*>(dOi);
            int tv = BQ * D / 8;
            for (int i = t; i < tv; i += NT) {
                int r = i / (D/8), cv = i % (D/8), gr = qs + r;
                if (gr < S) {
                    qv[i] = *reinterpret_cast<const int4*>(&Qb[(size_t)gr*D + cv*8]);
                    dv[i] = *reinterpret_cast<const int4*>(&dOb[(size_t)gr*D + cv*8]);
                } else { qv[i] = make_int4(0,0,0,0); dv[i] = make_int4(0,0,0,0); }
            }
        }
        if (t < BQ) { int gr = qs + t; Li[t] = (gr < S) ? Lb[gr] : 0.0f; Di[t] = (gr < S) ? Db[gr] : 0.0f; }
        __syncthreads();

        // S^T = Kj @ Qi^T (BKV x BQ)
        {
            int wr = w / 2, wcb = (w % 2) * 2;
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                int ct = wcb + j;
                wmma::fragment<wmma::accumulator, WMM, WMN, WMK, float> cf;
                wmma::fill_fragment(cf, 0.0f);
                #pragma unroll
                for (int kk = 0; kk < D/WMK; kk++) {
                    wmma::fragment<wmma::matrix_a, WMM, WMN, WMK, __nv_bfloat16, wmma::row_major> af;
                    wmma::fragment<wmma::matrix_b, WMM, WMN, WMK, __nv_bfloat16, wmma::col_major> bf;
                    wmma::load_matrix_sync(af, Kj + wr*16*D + kk*16, D);
                    wmma::load_matrix_sync(bf, Qi + ct*16*D + kk*16, D);
                    wmma::mma_sync(cf, af, bf, cf);
                }
                wmma::store_matrix_sync(S_sm + wr*16*BQ + ct*16, cf, BQ, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // P^T = exp(S^T * scale - L^T)
        for (int i = t; i < BKV * BQ; i += NT) {
            int r = i / BQ, c = i % BQ;
            if (kvs + r < S && qs + c < S) {
                float pv = __expf(S_sm[i] * scale - Li[c]);
                P_sm[i] = pv;
                Pbf[i] = __float2bfloat16(pv);
            } else { P_sm[i] = 0.0f; Pbf[i] = __float2bfloat16(0.0f); }
        }
        __syncthreads();

        // dP^T = Vj @ dOi^T
        {
            int wr = w / 2, wcb = (w % 2) * 2;
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                int ct = wcb + j;
                wmma::fragment<wmma::accumulator, WMM, WMN, WMK, float> cf;
                wmma::fill_fragment(cf, 0.0f);
                #pragma unroll
                for (int kk = 0; kk < D/WMK; kk++) {
                    wmma::fragment<wmma::matrix_a, WMM, WMN, WMK, __nv_bfloat16, wmma::row_major> af;
                    wmma::fragment<wmma::matrix_b, WMM, WMN, WMK, __nv_bfloat16, wmma::col_major> bf;
                    wmma::load_matrix_sync(af, Vj + wr*16*D + kk*16, D);
                    wmma::load_matrix_sync(bf, dOi + ct*16*D + kk*16, D);
                    wmma::mma_sync(cf, af, bf, cf);
                }
                wmma::store_matrix_sync(S_sm + wr*16*BQ + ct*16, cf, BQ, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // dS^T = P^T * (dP^T - D^T) * scale -> bf16
        for (int i = t; i < BKV * BQ; i += NT) {
            int r = i / BQ, c = i % BQ;
            if (kvs + r < S && qs + c < S) {
                dSbf[i] = __float2bfloat16(P_sm[i] * (S_sm[i] - Di[c]) * scale);
            } else { dSbf[i] = __float2bfloat16(0.0f); }
        }
        __syncthreads();

        // dV += P^T @ dOi (BKV x D)
        // A = P^T row_major (BKV x BQ), B = dOi row_major (BQ x D)
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int ct = j;
            #pragma unroll
            for (int kk = 0; kk < BQ/WMK; kk++) {
                wmma::fragment<wmma::matrix_a, WMM, WMN, WMK, __nv_bfloat16, wmma::row_major> af;
                wmma::fragment<wmma::matrix_b, WMM, WMN, WMK, __nv_bfloat16, wmma::row_major> bf;
                wmma::load_matrix_sync(af, Pbf + w*16*BQ + kk*16, BQ);
                wmma::load_matrix_sync(bf, dOi + kk*16*D + ct*16, D);
                wmma::mma_sync(dVf[j], af, bf, dVf[j]);
            }
        }

        // dK += dS^T @ Qi (BKV x D)
        // A = dS^T row_major (BKV x BQ), B = Qi row_major (BQ x D)
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int ct = j;
            #pragma unroll
            for (int kk = 0; kk < BQ/WMK; kk++) {
                wmma::fragment<wmma::matrix_a, WMM, WMN, WMK, __nv_bfloat16, wmma::row_major> af;
                wmma::fragment<wmma::matrix_b, WMM, WMN, WMK, __nv_bfloat16, wmma::row_major> bf;
                wmma::load_matrix_sync(af, dSbf + w*16*BQ + kk*16, BQ);
                wmma::load_matrix_sync(bf, Qi + kk*16*D + ct*16, D);
                wmma::mma_sync(dKf[j], af, bf, dKf[j]);
            }
        }
        __syncthreads();
    }

    {
        float *st = stg + w * WMM * WMN;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int grb = kvs + w * 16, gcb = j * 16;

            wmma::store_matrix_sync(st, dKf[j], WMN, wmma::mem_row_major);
            __syncwarp();
            for (int i = l; i < WMM * WMN; i += 32) {
                int r = i / WMN, c = i % WMN, gr = grb + r;
                if (gr < S) dKb[(size_t)gr * D + gcb + c] = __float2bfloat16(st[i]);
            }

            wmma::store_matrix_sync(st, dVf[j], WMN, wmma::mem_row_major);
            __syncwarp();
            for (int i = l; i < WMM * WMN; i += 32) {
                int r = i / WMN, c = i % WMN, gr = grb + r;
                if (gr < S) dVb[(size_t)gr * D + gcb + c] = __float2bfloat16(st[i]);
            }
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {

    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    const int B = 4, H = 48, S = (int)Q.size(2);
    int tr = B * H * S;

    const __nv_bfloat16 *Qp = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16 *Kp = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16 *Vp = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16 *Op = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16 *dOp = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float *Lp = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16 *dQp = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16 *dKp = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16 *dVp = static_cast<__nv_bfloat16*>(dV.data_ptr());

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    float *Dbuf;
    CUDA_CHECK(cudaMalloc(&Dbuf, (size_t)tr * sizeof(float)));

    compute_D_kernel<<<tr, 128, 0, stream>>>(Op, dOp, Dbuf, tr);

    {
        int nqb = (S + BQ - 1) / BQ;
        int grid = B * H * nqb;
        CUDA_CHECK(cudaFuncSetAttribute(dQ_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SQ));
        dQ_kernel<<<grid, NT, SQ, stream>>>(Qp, Kp, Vp, dOp, Lp, Dbuf, dQp, B, H, S);
    }

    {
        int nkb = (S + BKV - 1) / BKV;
        int grid = B * H * nkb;
        CUDA_CHECK(cudaFuncSetAttribute(dKV_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SKV));
        dKV_kernel<<<grid, NT, SKV, stream>>>(Qp, Kp, Vp, dOp, Lp, Dbuf, dKp, dVp, B, H, S);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(Dbuf));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, attention_bwd::run);

}  // namespace attention_bwd