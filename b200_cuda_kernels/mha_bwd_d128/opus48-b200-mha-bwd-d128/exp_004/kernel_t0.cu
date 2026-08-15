#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cmath>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_bwd {

using namespace nvcuda;

// D[b,h,s] = sum_c dO * O
__global__ void compute_D_kernel(const __nv_bfloat16* dO, const __nv_bfloat16* O,
                                 float* D, long total) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        long base = idx * 128;
        float acc = 0.f;
        #pragma unroll
        for (int c = 0; c < 128; c++)
            acc += (float)dO[base + c] * (float)O[base + c];
        D[idx] = acc;
    }
}

__global__ void convert_dQ_kernel(const float* acc, __nv_bfloat16* out, long n) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = __float2bfloat16(acc[idx]);
}

__device__ __forceinline__ void load_tile(__nv_bfloat16* dst, const __nv_bfloat16* src,
                                           int rowbase, int S, int tid) {
    // dst[64][128], vectorized int4 (8 bf16) loads, per-row validity
    for (int i = tid; i < 64 * 16; i += 128) {
        int r = i >> 4;
        int cg = i & 15;
        int grow = rowbase + r;
        int4 v;
        if (grow < S) v = reinterpret_cast<const int4*>(src + (long)grow * 128)[cg];
        else          v = make_int4(0, 0, 0, 0);
        reinterpret_cast<int4*>(dst + r * 128)[cg] = v;
    }
}

// SMEM layout (bytes)
// sK 0, sV 16384, sQ 32768, sdO 49152, sScore 65536(f), sPf 81920(f),
// sP 98304(bf), sdS 106496(bf), sdV 114688(f), sdK 147456(f),
// sL 180224(f), sD 180480(f), sdQ 180736(f) -> total 213504
static const int SMEM_BYTES = 213504;

__global__ __launch_bounds__(128) void bwd_kernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* dO, const float* Lp, const float* Dp,
    float* dQacc, __nv_bfloat16* dKo, __nv_bfloat16* dVo,
    int S, int H, float scale)
{
    extern __shared__ char smem[];
    __nv_bfloat16* sK  = (__nv_bfloat16*)(smem + 0);
    __nv_bfloat16* sV  = (__nv_bfloat16*)(smem + 16384);
    __nv_bfloat16* sQ  = (__nv_bfloat16*)(smem + 32768);
    __nv_bfloat16* sdO = (__nv_bfloat16*)(smem + 49152);
    float* sScore      = (float*)(smem + 65536);
    float* sPf         = (float*)(smem + 81920);
    __nv_bfloat16* sP  = (__nv_bfloat16*)(smem + 98304);
    __nv_bfloat16* sdS = (__nv_bfloat16*)(smem + 106496);
    float* sdV         = (float*)(smem + 114688);
    float* sdK         = (float*)(smem + 147456);
    float* sL          = (float*)(smem + 180224);
    float* sD          = (float*)(smem + 180480);
    float* sdQ         = (float*)(smem + 180736);

    int tid  = threadIdx.x;
    int warp = tid >> 5;

    int j = blockIdx.x;   // kv block
    int h = blockIdx.y;
    int b = blockIdx.z;
    int bh = b * H + h;
    long base = (long)bh * S * 128;

    const __nv_bfloat16* Qbh  = Q + base;
    const __nv_bfloat16* Kbh  = K + base;
    const __nv_bfloat16* Vbh  = V + base;
    const __nv_bfloat16* dObh = dO + base;

    int kbase = j * 64;
    int numQ = (S + 63) / 64;

    // load K, V (fixed for this CTA)
    load_tile(sK, Kbh, kbase, S, tid);
    load_tile(sV, Vbh, kbase, S, tid);

    for (int e = tid; e < 64 * 128; e += 128) { sdV[e] = 0.f; sdK[e] = 0.f; }
    __syncthreads();

    for (int i = 0; i < numQ; i++) {
        int qbase = i * 64;
        load_tile(sQ, Qbh, qbase, S, tid);
        load_tile(sdO, dObh, qbase, S, tid);
        for (int r = tid; r < 64; r += 128) {
            int gq = qbase + r;
            sL[r] = (gq < S) ? Lp[(long)bh * S + gq] : 1e30f;
            sD[r] = (gq < S) ? Dp[(long)bh * S + gq] : 0.f;
        }
        __syncthreads();

        // Phase A: S = Q @ K^T  -> sScore
        for (int nt = 0; nt < 4; nt++) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> cS;
            wmma::fill_fragment(cS, 0.f);
            for (int kt = 0; kt < 8; kt++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fb;
                wmma::load_matrix_sync(fa, sQ + warp * 16 * 128 + kt * 16, 128);
                wmma::load_matrix_sync(fb, sK + nt * 16 * 128 + kt * 16, 128);
                wmma::mma_sync(cS, fa, fb, cS);
            }
            wmma::store_matrix_sync(sScore + warp * 16 * 64 + nt * 16, cS, 64, wmma::mem_row_major);
        }
        __syncthreads();

        // Phase B: P = exp(scale*S - L)
        for (int e = tid; e < 64 * 64; e += 128) {
            int q = e >> 6, k = e & 63;
            int gk = kbase + k;
            float s = sScore[e];
            float p = (gk < S) ? __expf(scale * s - sL[q]) : 0.f;
            sPf[e] = p;
            sP[e] = __float2bfloat16(p);
        }
        __syncthreads();

        // Phase C: dV += P^T @ dO
        for (int nt = 0; nt < 8; nt++) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> cV;
            wmma::load_matrix_sync(cV, sdV + warp * 16 * 128 + nt * 16, 128, wmma::mem_row_major);
            for (int kt = 0; kt < 4; kt++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> fa;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> fb;
                wmma::load_matrix_sync(fa, sP + kt * 16 * 64 + warp * 16, 64);
                wmma::load_matrix_sync(fb, sdO + kt * 16 * 128 + nt * 16, 128);
                wmma::mma_sync(cV, fa, fb, cV);
            }
            wmma::store_matrix_sync(sdV + warp * 16 * 128 + nt * 16, cV, 128, wmma::mem_row_major);
        }
        __syncthreads();

        // Phase D: dP = dO @ V^T -> sScore
        for (int nt = 0; nt < 4; nt++) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> cP;
            wmma::fill_fragment(cP, 0.f);
            for (int kt = 0; kt < 8; kt++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fb;
                wmma::load_matrix_sync(fa, sdO + warp * 16 * 128 + kt * 16, 128);
                wmma::load_matrix_sync(fb, sV + nt * 16 * 128 + kt * 16, 128);
                wmma::mma_sync(cP, fa, fb, cP);
            }
            wmma::store_matrix_sync(sScore + warp * 16 * 64 + nt * 16, cP, 64, wmma::mem_row_major);
        }
        __syncthreads();

        // Phase E: dS = scale * P * (dP - D)
        for (int e = tid; e < 64 * 64; e += 128) {
            int q = e >> 6;
            float dp = sScore[e];
            float p = sPf[e];
            float dd = sD[q];
            float ds = scale * p * (dp - dd);
            sdS[e] = __float2bfloat16(ds);
        }
        __syncthreads();

        // Phase F: dK += dS^T @ Q
        for (int nt = 0; nt < 8; nt++) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> cK;
            wmma::load_matrix_sync(cK, sdK + warp * 16 * 128 + nt * 16, 128, wmma::mem_row_major);
            for (int kt = 0; kt < 4; kt++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> fa;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> fb;
                wmma::load_matrix_sync(fa, sdS + kt * 16 * 64 + warp * 16, 64);
                wmma::load_matrix_sync(fb, sQ + kt * 16 * 128 + nt * 16, 128);
                wmma::mma_sync(cK, fa, fb, cK);
            }
            wmma::store_matrix_sync(sdK + warp * 16 * 128 + nt * 16, cK, 128, wmma::mem_row_major);
        }
        // no sync needed: G reads sdS (already ready), writes sdQ

        // Phase G: dQ = dS @ K -> sdQ
        for (int nt = 0; nt < 8; nt++) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> cQ;
            wmma::fill_fragment(cQ, 0.f);
            for (int kt = 0; kt < 4; kt++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> fb;
                wmma::load_matrix_sync(fa, sdS + warp * 16 * 64 + kt * 16, 64);
                wmma::load_matrix_sync(fb, sK + kt * 16 * 128 + nt * 16, 128);
                wmma::mma_sync(cQ, fa, fb, cQ);
            }
            wmma::store_matrix_sync(sdQ + warp * 16 * 128 + nt * 16, cQ, 128, wmma::mem_row_major);
        }
        __syncthreads();

        for (int e = tid; e < 64 * 128; e += 128) {
            int q = e >> 7, c = e & 127;
            int gq = qbase + q;
            if (gq < S) atomicAdd(&dQacc[base + (long)gq * 128 + c], sdQ[e]);
        }
        __syncthreads();
    }

    // write dK, dV
    for (int e = tid; e < 64 * 128; e += 128) {
        int key = e >> 7, c = e & 127;
        int gk = kbase + key;
        if (gk < S) {
            dVo[base + (long)gk * 128 + c] = __float2bfloat16(sdV[e]);
            dKo[base + (long)gk * 128 + c] = __float2bfloat16(sdK[e]);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = Q.size(3);

    const __nv_bfloat16* Qp  = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* Kp  = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* Vp  = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* Op  = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dOp = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* Lp = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQp = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dKp = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dVp = static_cast<__nv_bfloat16*>(dV.data_ptr());

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    if (S <= 0) return;

    long rows = B * H * S;
    long elems = rows * 128;

    float* dDbuf = nullptr;
    float* dQacc = nullptr;
    CUDA_CHECK(cudaMallocAsync(&dDbuf, rows * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&dQacc, elems * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(dQacc, 0, elems * sizeof(float), stream));

    // compute D
    {
        int threads = 256;
        long blocks = (rows + threads - 1) / threads;
        compute_D_kernel<<<blocks, threads, 0, stream>>>(dOp, Op, dDbuf, rows);
        CUDA_CHECK(cudaGetLastError());
    }

    // main backward kernel
    {
        static bool attr_set = false;
        if (!attr_set) {
            CUDA_CHECK(cudaFuncSetAttribute(bwd_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
            attr_set = true;
        }
        int numKV = (int)((S + 63) / 64);
        dim3 grid(numKV, (unsigned)H, (unsigned)B);
        bwd_kernel<<<grid, 128, SMEM_BYTES, stream>>>(
            Qp, Kp, Vp, dOp, Lp, dDbuf, dQacc, dKp, dVp,
            (int)S, (int)H, 1.0f / sqrtf((float)d));
        CUDA_CHECK(cudaGetLastError());
    }

    // convert dQ float -> bf16
    {
        int threads = 256;
        long blocks = (elems + threads - 1) / threads;
        convert_dQ_kernel<<<blocks, threads, 0, stream>>>(dQacc, dQp, elems);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaFreeAsync(dDbuf, stream));
    CUDA_CHECK(cudaFreeAsync(dQacc, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd