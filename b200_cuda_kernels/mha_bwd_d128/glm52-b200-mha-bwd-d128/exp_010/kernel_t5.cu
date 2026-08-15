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

constexpr int D_HEAD = 128;
constexpr int BQ = 64;
constexpr int BKV = 128;
constexpr int NWARPS = 8;
constexpr int NTHREADS = 256;
constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;

// SMEM layout:
// Kj: 128*128*2 = 32KB, Vj: 32KB, Qi: 64*128*2 = 16KB, dOi: 16KB
// S/dP float: 64*128*4 = 32KB, P float: 32KB, P_bf16: 16KB, dS_bf16: 16KB
// Li/Di: 0.5KB, staging: 8KB
// Total: ~184.5KB
constexpr int SMEM_SIZE = 32768*2 + 16384*2 + 32768 + 32768 + 16384 + 16384 + 512 + 8192;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

__global__ void compute_D_kernel(
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    float* __restrict__ D, int total_rows) {
    int row = blockIdx.x;
    if (row >= total_rows) return;
    int tid = threadIdx.x;
    const __nv_bfloat16* Op = O + (size_t)row * D_HEAD;
    const __nv_bfloat16* dOp = dO + (size_t)row * D_HEAD;
    float sum = 0.0f;
    for (int i = tid; i < D_HEAD; i += blockDim.x)
        sum += __bfloat162float(Op[i]) * __bfloat162float(dOp[i]);
    sum = warp_reduce_sum(sum);
    __shared__ float ws[8];
    int wid = tid / 32, lid = tid % 32;
    if (lid == 0) ws[wid] = sum;
    __syncthreads();
    if (tid == 0) {
        float t = 0.0f;
        for (int w = 0; w < blockDim.x/32; w++) t += ws[w];
        D[row] = t;
    }
}

__global__ void zero_float_kernel(float* p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = 0.0f;
}

__global__ void convert_f32_bf16_kernel(const float* s, __nv_bfloat16* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = __float2bfloat16(s[i]);
}

__global__ __launch_bounds__(NTHREADS, 1) void attn_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ D_arr,
    float* __restrict__ dQ_f,
    __nv_bfloat16* __restrict__ dK_out,
    __nv_bfloat16* __restrict__ dV_out,
    int B, int H, int S) {

    extern __shared__ char smem[];
    __nv_bfloat16* Kj = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* Vj = Kj + BKV * D_HEAD;
    __nv_bfloat16* Qi = Vj + BKV * D_HEAD;
    __nv_bfloat16* dOi = Qi + BQ * D_HEAD;
    float* S_sm = reinterpret_cast<float*>(dOi + BQ * D_HEAD);
    float* P_sm = S_sm + BQ * BKV;
    __nv_bfloat16* Pbf = reinterpret_cast<__nv_bfloat16*>(P_sm + BQ * BKV);
    __nv_bfloat16* dSbf = Pbf + BQ * BKV;
    float* Li = reinterpret_cast<float*>(dSbf + BQ * BKV);
    float* Di = Li + BQ;
    float* stage = Di + BQ;

    int tid = threadIdx.x, wid = tid / 32, lid = tid % 32;
    int nkv = (S + BKV - 1) / BKV;
    int bh = blockIdx.x / nkv;
    int kvb = blockIdx.x % nkv;
    int b = bh / H, h = bh % H;
    int kvs = kvb * BKV;

    size_t bho = (size_t)(b * H + h) * S * D_HEAD;
    const __nv_bfloat16 *Qb = Q + bho, *Kb = K + bho, *Vb = V + bho, *dOb = dO + bho;
    const float *Lb = L + (size_t)(b*H+h)*S, *Db = D_arr + (size_t)(b*H+h)*S;
    float* dQb = dQ_f + bho;
    __nv_bfloat16 *dKb = dK_out + bho, *dVb = dV_out + bho;

    // Load K, V with int4 vectorization
    {
        int4* Kv = reinterpret_cast<int4*>(Kj);
        int4* Vv = reinterpret_cast<int4*>(Vj);
        int tv = BKV * D_HEAD / 8;
        for (int i = tid; i < tv; i += NTHREADS) {
            int r = i / (D_HEAD/8), cv = i % (D_HEAD/8);
            int gr = kvs + r;
            if (gr < S) {
                Kv[i] = *reinterpret_cast<const int4*>(&Kb[(size_t)gr*D_HEAD + cv*8]);
                Vv[i] = *reinterpret_cast<const int4*>(&Vb[(size_t)gr*D_HEAD + cv*8]);
            } else { Kv[i] = make_int4(0,0,0,0); Vv[i] = make_int4(0,0,0,0); }
        }
    }
    __syncthreads();

    // dK/dV: BKV x D = 128x128 = 8x8 = 64 tiles, 8 per warp
    // Warp w: row_base=(w/4)*4, col_base=(w%4)*2, fi: row=rb+fi/2, col=cb+fi%2
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> dKf[8], dVf[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) { wmma::fill_fragment(dKf[i], 0.0f); wmma::fill_fragment(dVf[i], 0.0f); }

    const float scale = 0.08838834764f;

    for (int qs = 0; qs < S; qs += BQ) {
        // Load Q, dO, L, D
        {
            int4* Qv = reinterpret_cast<int4*>(Qi);
            int4* dOv = reinterpret_cast<int4*>(dOi);
            int tv = BQ * D_HEAD / 8;
            for (int i = tid; i < tv; i += NTHREADS) {
                int r = i / (D_HEAD/8), cv = i % (D_HEAD/8);
                int gr = qs + r;
                if (gr < S) {
                    Qv[i] = *reinterpret_cast<const int4*>(&Qb[(size_t)gr*D_HEAD + cv*8]);
                    dOv[i] = *reinterpret_cast<const int4*>(&dOb[(size_t)gr*D_HEAD + cv*8]);
                } else { Qv[i] = make_int4(0,0,0,0); dOv[i] = make_int4(0,0,0,0); }
            }
        }
        if (tid < BQ) {
            int gr = qs + tid;
            Li[tid] = (gr < S) ? Lb[gr] : 0.0f;
            Di[tid] = (gr < S) ? Db[gr] : 0.0f;
        }
        __syncthreads();

        // S = Qi @ Kj^T (BQ x BKV = 4x8, 4 tiles/warp, 8 K-steps)
        // Warp w: row=w/2, col_base=(w%2)*4
        {
            int wr = wid / 2, wcb = (wid % 2) * 4;
            #pragma unroll
            for (int wj = 0; wj < 4; wj++) {
                int ct = wcb + wj;
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> cf;
                wmma::fill_fragment(cf, 0.0f);
                #pragma unroll
                for (int kk = 0; kk < D_HEAD/WMMA_K; kk++) {
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> af;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> bf;
                    wmma::load_matrix_sync(af, Qi + wr*16*D_HEAD + kk*16, D_HEAD);
                    wmma::load_matrix_sync(bf, Kj + ct*16*D_HEAD + kk*16, D_HEAD);
                    wmma::mma_sync(cf, af, bf, cf);
                }
                wmma::store_matrix_sync(S_sm + wr*16*BKV + ct*16, cf, BKV, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // P = exp(S * scale - L), store float and bf16
        for (int i = tid; i < BQ * BKV; i += NTHREADS) {
            int r = i / BKV, c = i % BKV;
            if (qs + r < S && kvs + c < S) {
                float pv = __expf(S_sm[i] * scale - Li[r]);
                P_sm[i] = pv;
                Pbf[i] = __float2bfloat16(pv);
            } else { P_sm[i] = 0.0f; Pbf[i] = __float2bfloat16(0.0f); }
        }
        __syncthreads();

        // dP = dOi @ Vj^T (reuse S_sm)
        {
            int wr = wid / 2, wcb = (wid % 2) * 4;
            #pragma unroll
            for (int wj = 0; wj < 4; wj++) {
                int ct = wcb + wj;
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> cf;
                wmma::fill_fragment(cf, 0.0f);
                #pragma unroll
                for (int kk = 0; kk < D_HEAD/WMMA_K; kk++) {
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> af;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> bf;
                    wmma::load_matrix_sync(af, dOi + wr*16*D_HEAD + kk*16, D_HEAD);
                    wmma::load_matrix_sync(bf, Vj + ct*16*D_HEAD + kk*16, D_HEAD);
                    wmma::mma_sync(cf, af, bf, cf);
                }
                wmma::store_matrix_sync(S_sm + wr*16*BKV + ct*16, cf, BKV, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // dS = P * (dP - D) * scale -> bf16
        for (int i = tid; i < BQ * BKV; i += NTHREADS) {
            int r = i / BKV, c = i % BKV;
            if (qs + r < S && kvs + c < S) {
                float dsv = P_sm[i] * (S_sm[i] - Di[r]) * scale;
                dSbf[i] = __float2bfloat16(dsv);
            } else { dSbf[i] = __float2bfloat16(0.0f); }
        }
        __syncthreads();

        // dV += P^T @ dOi (BKV x D = 8x8, 8 tiles/warp, 4 K-steps)
        // A=P^T col_major from Pbf (BQ x BKV), B=dOi row_major
        {
            int rb = (wid/4)*4, cb = (wid%4)*2;
            #pragma unroll
            for (int fi = 0; fi < 8; fi++) {
                int rt = rb + fi/2, ct = cb + fi%2;
                #pragma unroll
                for (int kk = 0; kk < BQ/WMMA_K; kk++) {
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> af;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> bf;
                    wmma::load_matrix_sync(af, Pbf + kk*16*BKV + rt*16, BKV);
                    wmma::load_matrix_sync(bf, dOi + kk*16*D_HEAD + ct*16, D_HEAD);
                    wmma::mma_sync(dVf[fi], af, bf, dVf[fi]);
                }
            }
        }

        // dK += dS^T @ Qi (same structure)
        {
            int rb = (wid/4)*4, cb = (wid%4)*2;
            #pragma unroll
            for (int fi = 0; fi < 8; fi++) {
                int rt = rb + fi/2, ct = cb + fi%2;
                #pragma unroll
                for (int kk = 0; kk < BQ/WMMA_K; kk++) {
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> af;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> bf;
                    wmma::load_matrix_sync(af, dSbf + kk*16*BKV + rt*16, BKV);
                    wmma::load_matrix_sync(bf, Qi + kk*16*D_HEAD + ct*16, D_HEAD);
                    wmma::mma_sync(dKf[fi], af, bf, dKf[fi]);
                }
            }
        }

        // dQ += dS @ Kj (BQ x D = 4x8, 4 tiles/warp, 8 K-steps) -> atomic add
        {
            int wr = wid / 2, wcb = (wid % 2) * 4;
            #pragma unroll
            for (int wj = 0; wj < 4; wj++) {
                int ct = wcb + wj;
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> dqf;
                wmma::fill_fragment(dqf, 0.0f);
                #pragma unroll
                for (int kk = 0; kk < BKV/WMMA_K; kk++) {
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> af;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> bf;
                    wmma::load_matrix_sync(af, dSbf + wr*16*BKV + kk*16, BKV);
                    wmma::load_matrix_sync(bf, Kj + kk*16*D_HEAD + ct*16, D_HEAD);
                    wmma::mma_sync(dqf, af, bf, dqf);
                }
                float* st = stage + wid * WMMA_M * WMMA_N;
                wmma::store_matrix_sync(st, dqf, WMMA_N, wmma::mem_row_major);
                __syncwarp();
                int grb = qs + wr*16, gcb = ct*16;
                for (int i = lid; i < WMMA_M*WMMA_N; i += 32) {
                    int r = i/WMMA_N, c = i%WMMA_N;
                    int gr = grb + r;
                    if (gr < S) atomicAdd(&dQb[(size_t)gr*D_HEAD + gcb + c], st[i]);
                }
            }
        }
        __syncthreads();
    }

    // Store dK, dV
    {
        int rb = (wid/4)*4, cb = (wid%4)*2;
        float* st = stage + wid * WMMA_M * WMMA_N;
        #pragma unroll
        for (int fi = 0; fi < 8; fi++) {
            int rt = rb + fi/2, ct = cb + fi%2;
            int grb = kvs + rt*16, gcb = ct*16;

            wmma::store_matrix_sync(st, dKf[fi], WMMA_N, wmma::mem_row_major);
            __syncwarp();
            for (int i = lid; i < WMMA_M*WMMA_N; i += 32) {
                int r = i/WMMA_N, c = i%WMMA_N;
                int gr = grb + r;
                if (gr < S) dKb[(size_t)gr*D_HEAD + gcb + c] = __float2bfloat16(st[i]);
            }

            wmma::store_matrix_sync(st, dVf[fi], WMMA_N, wmma::mem_row_major);
            __syncwarp();
            for (int i = lid; i < WMMA_M*WMMA_N; i += 32) {
                int r = i/WMMA_N, c = i%WMMA_N;
                int gr = grb + r;
                if (gr < S) dVb[(size_t)gr*D_HEAD + gcb + c] = __float2bfloat16(st[i]);
            }
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {

    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    const int B = 4, H = 48, S = (int)Q.size(2);
    int te = B * H * S * 128, tr = B * H * S;

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

    float *Dbuf, *dQf;
    CUDA_CHECK(cudaMalloc(&Dbuf, (size_t)tr * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dQf, (size_t)te * sizeof(float)));

    { int t=256, bl=(te+t-1)/t; zero_float_kernel<<<bl,t,0,stream>>>(dQf, te); }
    compute_D_kernel<<<tr, 128, 0, stream>>>(Op, dOp, Dbuf, tr);

    { int nkv=(S+BKV-1)/BKV, grid=B*H*nkv;
      CUDA_CHECK(cudaFuncSetAttribute(attn_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));
      attn_bwd_kernel<<<grid, NTHREADS, SMEM_SIZE, stream>>>(Qp,Kp,Vp,dOp,Lp,Dbuf,dQf,dKp,dVp,B,H,S);
    }
    { int t=256, bl=(te+t-1)/t; convert_f32_bf16_kernel<<<bl,t,0,stream>>>(dQf, dQp, te); }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(Dbuf));
    CUDA_CHECK(cudaFree(dQf));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, attention_bwd::run);

}  // namespace attention_bwd