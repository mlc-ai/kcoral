#include <cuda_bf16.h>
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
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_bwd {

// ---------------- Preprocess: Delta = rowsum(O * dO) ----------------
__global__ void compute_delta(const __nv_bfloat16* __restrict__ O,
                              const __nv_bfloat16* __restrict__ dO,
                              float* __restrict__ Delta, int total_rows) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (row >= total_rows) return;
    const __nv_bfloat16* o = O + (size_t)row * 128;
    const __nv_bfloat16* g = dO + (size_t)row * 128;
    float s = 0.f;
    #pragma unroll
    for (int c = lane; c < 128; c += 32) {
        s += __bfloat162float(o[c]) * __bfloat162float(g[c]);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        s += __shfl_down_sync(0xffffffff, s, off);
    if (lane == 0) Delta[row] = s;
}

// out[64][64] = A[64][128] @ B[64][128]^T  (contraction over 128)
__device__ __forceinline__ void gemm_AxBt_64(const __nv_bfloat16* A,
                                             const __nv_bfloat16* B,
                                             float* out, int warp) {
    #pragma unroll
    for (int t = 0; t < 2; t++) {
        int l = warp * 2 + t;
        int mt = l >> 2;   // /4
        int nt = l & 3;    // %4
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.0f);
        #pragma unroll
        for (int kt = 0; kt < 8; kt++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf;
            wmma::load_matrix_sync(af, A + mt * 16 * 128 + kt * 16, 128);
            wmma::load_matrix_sync(bf, B + nt * 16 * 128 + kt * 16, 128);
            wmma::mma_sync(acc, af, bf, acc);
        }
        wmma::store_matrix_sync(out + mt * 16 * 64 + nt * 16, acc, 64, wmma::mem_row_major);
    }
}

// ---------------- dK / dV kernel (parallel over KV blocks) ----------------
__global__ void __launch_bounds__(256) bwd_dkv_kernel(
        const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
        const __nv_bfloat16* __restrict__ V, const __nv_bfloat16* __restrict__ dO,
        const float* __restrict__ L, const float* __restrict__ Delta,
        __nv_bfloat16* __restrict__ dK, __nv_bfloat16* __restrict__ dV,
        int S, int H, float scale) {
    int b = blockIdx.z, h = blockIdx.y, kv_block = blockIdx.x;
    int j0 = kv_block * 64;
    int tid = threadIdx.x, warp = tid >> 5;

    size_t bh = (size_t)b * H + h;
    const __nv_bfloat16* Qbh  = Q  + bh * S * 128;
    const __nv_bfloat16* Kbh  = K  + bh * S * 128;
    const __nv_bfloat16* Vbh  = V  + bh * S * 128;
    const __nv_bfloat16* dObh = dO + bh * S * 128;
    const float* Lbh = L + bh * S;
    const float* Dbh = Delta + bh * S;
    __nv_bfloat16* dKbh = dK + bh * S * 128;
    __nv_bfloat16* dVbh = dV + bh * S * 128;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* sK   = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sV   = sK + 64 * 128;
    __nv_bfloat16* sQ   = sV + 64 * 128;
    __nv_bfloat16* sdO  = sQ + 64 * 128;
    float* sS           = reinterpret_cast<float*>(sdO + 64 * 128);
    float* sPf          = sS + 64 * 64;
    __nv_bfloat16* sPbf = reinterpret_cast<__nv_bfloat16*>(sPf + 64 * 64);
    __nv_bfloat16* sdS  = sPbf + 64 * 64;
    float* sL           = reinterpret_cast<float*>(sdS + 64 * 64);
    float* sD           = sL + 64;
    float* sOut         = reinterpret_cast<float*>(smem_raw);

    // Load K, V block (persistent)
    for (int e = tid; e < 64 * 128; e += 256) {
        int n = e >> 7, k = e & 127, g = j0 + n;
        sK[e] = (g < S) ? Kbh[(size_t)g * 128 + k] : __float2bfloat16(0.0f);
        sV[e] = (g < S) ? Vbh[(size_t)g * 128 + k] : __float2bfloat16(0.0f);
    }

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dV_acc[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dK_acc[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) { wmma::fill_fragment(dV_acc[j], 0.f); wmma::fill_fragment(dK_acc[j], 0.f); }
    int n_tile = warp >> 1;
    int k_base = (warp & 1) * 4;

    int num_q = (S + 63) / 64;
    for (int qb = 0; qb < num_q; qb++) {
        int i0 = qb * 64;
        __syncthreads();
        for (int e = tid; e < 64 * 128; e += 256) {
            int i = e >> 7, k = e & 127, g = i0 + i;
            sQ[e]  = (g < S) ? Qbh[(size_t)g * 128 + k]  : __float2bfloat16(0.0f);
            sdO[e] = (g < S) ? dObh[(size_t)g * 128 + k] : __float2bfloat16(0.0f);
        }
        for (int i = tid; i < 64; i += 256) {
            int g = i0 + i;
            sL[i] = (g < S) ? Lbh[g] : INFINITY;
            sD[i] = (g < S) ? Dbh[g] : 0.f;
        }
        __syncthreads();

        // S = Q @ K^T
        gemm_AxBt_64(sQ, sK, sS, warp);
        __syncthreads();

        // P = exp(scale*S - L)
        for (int e = tid; e < 64 * 64; e += 256) {
            int i = e >> 6, j = e & 63;
            int gi = i0 + i, gj = j0 + j;
            float p = (gi < S && gj < S) ? __expf(scale * sS[e] - sL[i]) : 0.f;
            sPf[e] = p;
            sPbf[e] = __float2bfloat16(p);
        }
        __syncthreads();

        // dP = dO @ V^T  (into sS)
        gemm_AxBt_64(sdO, sV, sS, warp);
        __syncthreads();

        // dS' = scale * P * (dP - D)
        for (int e = tid; e < 64 * 64; e += 256) {
            int i = e >> 6;
            float ds = scale * sPf[e] * (sS[e] - sD[i]);
            sdS[e] = __float2bfloat16(ds);
        }
        __syncthreads();

        // dV += P^T @ dO
        #pragma unroll
        for (int jj = 0; jj < 4; jj++) {
            int ktile = k_base + jj;
            #pragma unroll
            for (int it = 0; it < 4; it++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> af;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bf;
                wmma::load_matrix_sync(af, sPbf + it * 16 * 64 + n_tile * 16, 64);
                wmma::load_matrix_sync(bf, sdO + it * 16 * 128 + ktile * 16, 128);
                wmma::mma_sync(dV_acc[jj], af, bf, dV_acc[jj]);
            }
        }
        // dK += dS'^T @ Q
        #pragma unroll
        for (int jj = 0; jj < 4; jj++) {
            int ktile = k_base + jj;
            #pragma unroll
            for (int it = 0; it < 4; it++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> af;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bf;
                wmma::load_matrix_sync(af, sdS + it * 16 * 64 + n_tile * 16, 64);
                wmma::load_matrix_sync(bf, sQ + it * 16 * 128 + ktile * 16, 128);
                wmma::mma_sync(dK_acc[jj], af, bf, dK_acc[jj]);
            }
        }
    }

    __syncthreads();
    // store dV
    #pragma unroll
    for (int jj = 0; jj < 4; jj++) {
        int ktile = k_base + jj;
        wmma::store_matrix_sync(sOut + n_tile * 16 * 128 + ktile * 16, dV_acc[jj], 128, wmma::mem_row_major);
    }
    __syncthreads();
    for (int e = tid; e < 64 * 128; e += 256) {
        int n = e >> 7, k = e & 127, g = j0 + n;
        if (g < S) dVbh[(size_t)g * 128 + k] = __float2bfloat16(sOut[e]);
    }
    __syncthreads();
    // store dK
    #pragma unroll
    for (int jj = 0; jj < 4; jj++) {
        int ktile = k_base + jj;
        wmma::store_matrix_sync(sOut + n_tile * 16 * 128 + ktile * 16, dK_acc[jj], 128, wmma::mem_row_major);
    }
    __syncthreads();
    for (int e = tid; e < 64 * 128; e += 256) {
        int n = e >> 7, k = e & 127, g = j0 + n;
        if (g < S) dKbh[(size_t)g * 128 + k] = __float2bfloat16(sOut[e]);
    }
}

// ---------------- dQ kernel (parallel over Q blocks) ----------------
__global__ void __launch_bounds__(256) bwd_dq_kernel(
        const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
        const __nv_bfloat16* __restrict__ V, const __nv_bfloat16* __restrict__ dO,
        const float* __restrict__ L, const float* __restrict__ Delta,
        __nv_bfloat16* __restrict__ dQ,
        int S, int H, float scale) {
    int b = blockIdx.z, h = blockIdx.y, q_block = blockIdx.x;
    int i0 = q_block * 64;
    int tid = threadIdx.x, warp = tid >> 5;

    size_t bh = (size_t)b * H + h;
    const __nv_bfloat16* Qbh  = Q  + bh * S * 128;
    const __nv_bfloat16* Kbh  = K  + bh * S * 128;
    const __nv_bfloat16* Vbh  = V  + bh * S * 128;
    const __nv_bfloat16* dObh = dO + bh * S * 128;
    const float* Lbh = L + bh * S;
    const float* Dbh = Delta + bh * S;
    __nv_bfloat16* dQbh = dQ + bh * S * 128;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* sQ   = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sdO  = sQ + 64 * 128;
    __nv_bfloat16* sK   = sdO + 64 * 128;
    __nv_bfloat16* sV   = sK + 64 * 128;
    float* sS           = reinterpret_cast<float*>(sV + 64 * 128);
    float* sPf          = sS + 64 * 64;
    __nv_bfloat16* sdS  = reinterpret_cast<__nv_bfloat16*>(sPf + 64 * 64);
    float* sL           = reinterpret_cast<float*>(sdS + 64 * 64);
    float* sD           = sL + 64;
    float* sOut         = reinterpret_cast<float*>(sK);

    // Load Q, dO, L, D (persistent)
    for (int e = tid; e < 64 * 128; e += 256) {
        int i = e >> 7, k = e & 127, g = i0 + i;
        sQ[e]  = (g < S) ? Qbh[(size_t)g * 128 + k]  : __float2bfloat16(0.0f);
        sdO[e] = (g < S) ? dObh[(size_t)g * 128 + k] : __float2bfloat16(0.0f);
    }
    for (int i = tid; i < 64; i += 256) {
        int g = i0 + i;
        sL[i] = (g < S) ? Lbh[g] : INFINITY;
        sD[i] = (g < S) ? Dbh[g] : 0.f;
    }

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dQ_acc[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) wmma::fill_fragment(dQ_acc[j], 0.f);
    int i_tile = warp >> 1;
    int k_base = (warp & 1) * 4;

    int num_kv = (S + 63) / 64;
    for (int kv = 0; kv < num_kv; kv++) {
        int j0 = kv * 64;
        __syncthreads();
        for (int e = tid; e < 64 * 128; e += 256) {
            int n = e >> 7, k = e & 127, g = j0 + n;
            sK[e] = (g < S) ? Kbh[(size_t)g * 128 + k] : __float2bfloat16(0.0f);
            sV[e] = (g < S) ? Vbh[(size_t)g * 128 + k] : __float2bfloat16(0.0f);
        }
        __syncthreads();

        // S = Q @ K^T
        gemm_AxBt_64(sQ, sK, sS, warp);
        __syncthreads();

        // P = exp(scale*S - L)
        for (int e = tid; e < 64 * 64; e += 256) {
            int i = e >> 6, j = e & 63;
            int gi = i0 + i, gj = j0 + j;
            sPf[e] = (gi < S && gj < S) ? __expf(scale * sS[e] - sL[i]) : 0.f;
        }
        __syncthreads();

        // dP = dO @ V^T  (into sS)
        gemm_AxBt_64(sdO, sV, sS, warp);
        __syncthreads();

        // dS' = scale * P * (dP - D)
        for (int e = tid; e < 64 * 64; e += 256) {
            int i = e >> 6;
            float ds = scale * sPf[e] * (sS[e] - sD[i]);
            sdS[e] = __float2bfloat16(ds);
        }
        __syncthreads();

        // dQ += dS' @ K
        #pragma unroll
        for (int jj = 0; jj < 4; jj++) {
            int ktile = k_base + jj;
            #pragma unroll
            for (int jt = 0; jt < 4; jt++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bf;
                wmma::load_matrix_sync(af, sdS + i_tile * 16 * 64 + jt * 16, 64);
                wmma::load_matrix_sync(bf, sK + jt * 16 * 128 + ktile * 16, 128);
                wmma::mma_sync(dQ_acc[jj], af, bf, dQ_acc[jj]);
            }
        }
    }

    __syncthreads();
    #pragma unroll
    for (int jj = 0; jj < 4; jj++) {
        int ktile = k_base + jj;
        wmma::store_matrix_sync(sOut + i_tile * 16 * 128 + ktile * 16, dQ_acc[jj], 128, wmma::mem_row_major);
    }
    __syncthreads();
    for (int e = tid; e < 64 * 128; e += 256) {
        int i = e >> 7, k = e & 127, g = i0 + i;
        if (g < S) dQbh[(size_t)g * 128 + k] = __float2bfloat16(sOut[e]);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int S = (int)Q.size(2);

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

    float scale = 1.0f / sqrtf(128.0f);

    float* Delta = nullptr;
    size_t total = (size_t)B * H * S;
    CUDA_CHECK(cudaMallocAsync(&Delta, total * sizeof(float), stream));

    int dgrid = (int)((total + 3) / 4);
    compute_delta<<<dgrid, 128, 0, stream>>>(Op, dOp, Delta, (int)total);
    CUDA_CHECK(cudaGetLastError());

    const int DKV_SMEM = 115200;
    const int DQ_SMEM  = 107008;
    static bool attr_set = false;
    if (!attr_set) {
        CUDA_CHECK(cudaFuncSetAttribute((const void*)bwd_dkv_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, DKV_SMEM));
        CUDA_CHECK(cudaFuncSetAttribute((const void*)bwd_dq_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, DQ_SMEM));
        attr_set = true;
    }

    int nblk = (S + 63) / 64;
    dim3 grid(nblk, H, B);

    bwd_dkv_kernel<<<grid, 256, DKV_SMEM, stream>>>(
        Qp, Kp, Vp, dOp, Lp, Delta, dKp, dVp, S, H, scale);
    CUDA_CHECK(cudaGetLastError());

    bwd_dq_kernel<<<grid, 256, DQ_SMEM, stream>>>(
        Qp, Kp, Vp, dOp, Lp, Delta, dQp, S, H, scale);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFreeAsync(Delta, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd