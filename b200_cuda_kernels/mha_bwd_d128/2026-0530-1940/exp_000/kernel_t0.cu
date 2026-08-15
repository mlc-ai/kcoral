#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <mma.h>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

namespace mha_bwd {

namespace wmma = nvcuda::wmma;

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

#define BR 64
#define BN 64
#define HD 128

using FragA_row = wmma::fragment<wmma::matrix_a, 16,16,16, __nv_bfloat16, wmma::row_major>;
using FragA_col = wmma::fragment<wmma::matrix_a, 16,16,16, __nv_bfloat16, wmma::col_major>;
using FragB_row = wmma::fragment<wmma::matrix_b, 16,16,16, __nv_bfloat16, wmma::row_major>;
using FragB_col = wmma::fragment<wmma::matrix_b, 16,16,16, __nv_bfloat16, wmma::col_major>;
using FragC     = wmma::fragment<wmma::accumulator, 16,16,16, float>;

// D[row] = sum_k dO[row,k]*O[row,k]
__global__ void compute_D_kernel(const __nv_bfloat16* dO, const __nv_bfloat16* O,
                                 float* D, int total_rows) {
    int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (gw >= total_rows) return;
    const __nv_bfloat16* dOr = dO + (size_t)gw * HD;
    const __nv_bfloat16* Or  = O  + (size_t)gw * HD;
    float sum = 0.f;
    #pragma unroll
    for (int k = lane; k < HD; k += 32) {
        sum += __bfloat162float(dOr[k]) * __bfloat162float(Or[k]);
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if (lane == 0) D[gw] = sum;
}

__device__ __forceinline__ void load_tile(const __nv_bfloat16* gbase, int row_start,
                                           int S, __nv_bfloat16* smem, int rows) {
    int n = rows * HD;
    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
        int r = idx >> 7;
        int c = idx & 127;
        int gr = row_start + r;
        smem[idx] = (gr < S) ? gbase[(size_t)gr * HD + c] : __float2bfloat16(0.f);
    }
}

// =================== dK / dV kernel =====================
__global__ void __launch_bounds__(128) dkv_kernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* dO, const float* L, const float* Dvec,
    __nv_bfloat16* dK, __nv_bfloat16* dV,
    int S, int num_q, int num_kv, float scale) {

    extern __shared__ __align__(16) char smem[];
    __nv_bfloat16* sQ  = (__nv_bfloat16*)(smem + 0);
    __nv_bfloat16* sdO = (__nv_bfloat16*)(smem + 16384);
    __nv_bfloat16* sK  = (__nv_bfloat16*)(smem + 32768);
    __nv_bfloat16* sV  = (__nv_bfloat16*)(smem + 49152);
    __nv_bfloat16* sP  = (__nv_bfloat16*)(smem + 65536);
    float* sTmp = (float*)(smem + 73728);
    float* sdV  = (float*)(smem + 90112);
    float* sdK  = (float*)(smem + 122880);
    float* sL   = (float*)(smem + 155648);
    float* sD   = (float*)(smem + 155904);

    int blk = blockIdx.x;
    int jb = blk % num_kv;
    int bh = blk / num_kv;
    int kv_start = jb * BN;
    int warp = threadIdx.x >> 5;

    const __nv_bfloat16* Qb  = Q  + (size_t)bh * S * HD;
    const __nv_bfloat16* Kb  = K  + (size_t)bh * S * HD;
    const __nv_bfloat16* Vb  = V  + (size_t)bh * S * HD;
    const __nv_bfloat16* dOb = dO + (size_t)bh * S * HD;
    const float* Lb = L + (size_t)bh * S;
    const float* Db = Dvec + (size_t)bh * S;

    load_tile(Kb, kv_start, S, sK, BN);
    load_tile(Vb, kv_start, S, sV, BN);
    for (int idx = threadIdx.x; idx < BN * HD; idx += blockDim.x) { sdV[idx] = 0.f; sdK[idx] = 0.f; }
    __syncthreads();

    for (int ib = 0; ib < num_q; ib++) {
        int q_start = ib * BR;
        load_tile(Qb, q_start, S, sQ, BR);
        load_tile(dOb, q_start, S, sdO, BR);
        for (int r = threadIdx.x; r < BR; r += blockDim.x) {
            int gr = q_start + r;
            sL[r] = (gr < S) ? Lb[gr] : 0.f;
            sD[r] = (gr < S) ? Db[gr] : 0.f;
        }
        __syncthreads();

        // S = Q @ K^T -> sTmp
        for (int c = 0; c < BN/16; c++) {
            FragC acc; wmma::fill_fragment(acc, 0.f);
            for (int k = 0; k < HD/16; k++) {
                FragA_row a; wmma::load_matrix_sync(a, sQ + (warp*16)*HD + k*16, HD);
                FragB_col b; wmma::load_matrix_sync(b, sK + (c*16)*HD + k*16, HD);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(sTmp + (warp*16)*BN + c*16, acc, BN, wmma::mem_row_major);
        }
        __syncthreads();

        // P = exp(scale*S - L), mask invalid KV cols
        for (int idx = threadIdx.x; idx < BR*BN; idx += blockDim.x) {
            int i = idx / BN, j = idx % BN;
            int gj = kv_start + j;
            float p = (gj < S) ? __expf(scale * sTmp[idx] - sL[i]) : 0.f;
            sP[idx] = __float2bfloat16(p);
        }
        __syncthreads();

        // dV += P^T @ dO
        for (int c = 0; c < HD/16; c++) {
            FragC acc; wmma::load_matrix_sync(acc, sdV + (warp*16)*HD + c*16, HD, wmma::mem_row_major);
            for (int k = 0; k < BR/16; k++) {
                FragA_col a; wmma::load_matrix_sync(a, sP + (k*16)*BN + warp*16, BN);
                FragB_row b; wmma::load_matrix_sync(b, sdO + (k*16)*HD + c*16, HD);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(sdV + (warp*16)*HD + c*16, acc, HD, wmma::mem_row_major);
        }
        __syncthreads();

        // dP = dO @ V^T -> sTmp
        for (int c = 0; c < BN/16; c++) {
            FragC acc; wmma::fill_fragment(acc, 0.f);
            for (int k = 0; k < HD/16; k++) {
                FragA_row a; wmma::load_matrix_sync(a, sdO + (warp*16)*HD + k*16, HD);
                FragB_col b; wmma::load_matrix_sync(b, sV + (c*16)*HD + k*16, HD);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(sTmp + (warp*16)*BN + c*16, acc, BN, wmma::mem_row_major);
        }
        __syncthreads();

        // dS = P*(dP - D)  (store into sP)
        for (int idx = threadIdx.x; idx < BR*BN; idx += blockDim.x) {
            int i = idx / BN;
            float p = __bfloat162float(sP[idx]);
            sP[idx] = __float2bfloat16(p * (sTmp[idx] - sD[i]));
        }
        __syncthreads();

        // dK += dS^T @ Q
        for (int c = 0; c < HD/16; c++) {
            FragC acc; wmma::load_matrix_sync(acc, sdK + (warp*16)*HD + c*16, HD, wmma::mem_row_major);
            for (int k = 0; k < BR/16; k++) {
                FragA_col a; wmma::load_matrix_sync(a, sP + (k*16)*BN + warp*16, BN);
                FragB_row b; wmma::load_matrix_sync(b, sQ + (k*16)*HD + c*16, HD);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(sdK + (warp*16)*HD + c*16, acc, HD, wmma::mem_row_major);
        }
        __syncthreads();
    }

    for (int idx = threadIdx.x; idx < BN*HD; idx += blockDim.x) {
        int r = idx >> 7, c = idx & 127;
        int gr = kv_start + r;
        if (gr < S) {
            dK[(size_t)bh*S*HD + (size_t)gr*HD + c] = __float2bfloat16(sdK[idx] * scale);
            dV[(size_t)bh*S*HD + (size_t)gr*HD + c] = __float2bfloat16(sdV[idx]);
        }
    }
}

// =================== dQ kernel =====================
__global__ void __launch_bounds__(128) dq_kernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* dO, const float* L, const float* Dvec,
    __nv_bfloat16* dQ,
    int S, int num_q, int num_kv, float scale) {

    extern __shared__ __align__(16) char smem[];
    __nv_bfloat16* sQ  = (__nv_bfloat16*)(smem + 0);
    __nv_bfloat16* sdO = (__nv_bfloat16*)(smem + 16384);
    __nv_bfloat16* sK  = (__nv_bfloat16*)(smem + 32768);
    __nv_bfloat16* sV  = (__nv_bfloat16*)(smem + 49152);
    __nv_bfloat16* sP  = (__nv_bfloat16*)(smem + 65536);
    float* sTmp = (float*)(smem + 73728);
    float* sdQ  = (float*)(smem + 90112);
    float* sL   = (float*)(smem + 122880);
    float* sD   = (float*)(smem + 123136);

    int blk = blockIdx.x;
    int ib = blk % num_q;
    int bh = blk / num_q;
    int q_start = ib * BR;
    int warp = threadIdx.x >> 5;

    const __nv_bfloat16* Qb  = Q  + (size_t)bh * S * HD;
    const __nv_bfloat16* Kb  = K  + (size_t)bh * S * HD;
    const __nv_bfloat16* Vb  = V  + (size_t)bh * S * HD;
    const __nv_bfloat16* dOb = dO + (size_t)bh * S * HD;
    const float* Lb = L + (size_t)bh * S;
    const float* Db = Dvec + (size_t)bh * S;

    load_tile(Qb, q_start, S, sQ, BR);
    load_tile(dOb, q_start, S, sdO, BR);
    for (int r = threadIdx.x; r < BR; r += blockDim.x) {
        int gr = q_start + r;
        sL[r] = (gr < S) ? Lb[gr] : 0.f;
        sD[r] = (gr < S) ? Db[gr] : 0.f;
    }
    for (int idx = threadIdx.x; idx < BR*HD; idx += blockDim.x) sdQ[idx] = 0.f;
    __syncthreads();

    for (int jb = 0; jb < num_kv; jb++) {
        int kv_start = jb * BN;
        load_tile(Kb, kv_start, S, sK, BN);
        load_tile(Vb, kv_start, S, sV, BN);
        __syncthreads();

        // S = Q @ K^T -> sTmp
        for (int c = 0; c < BN/16; c++) {
            FragC acc; wmma::fill_fragment(acc, 0.f);
            for (int k = 0; k < HD/16; k++) {
                FragA_row a; wmma::load_matrix_sync(a, sQ + (warp*16)*HD + k*16, HD);
                FragB_col b; wmma::load_matrix_sync(b, sK + (c*16)*HD + k*16, HD);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(sTmp + (warp*16)*BN + c*16, acc, BN, wmma::mem_row_major);
        }
        __syncthreads();

        // P
        for (int idx = threadIdx.x; idx < BR*BN; idx += blockDim.x) {
            int i = idx / BN, j = idx % BN;
            int gj = kv_start + j;
            float p = (gj < S) ? __expf(scale * sTmp[idx] - sL[i]) : 0.f;
            sP[idx] = __float2bfloat16(p);
        }
        __syncthreads();

        // dP = dO @ V^T -> sTmp
        for (int c = 0; c < BN/16; c++) {
            FragC acc; wmma::fill_fragment(acc, 0.f);
            for (int k = 0; k < HD/16; k++) {
                FragA_row a; wmma::load_matrix_sync(a, sdO + (warp*16)*HD + k*16, HD);
                FragB_col b; wmma::load_matrix_sync(b, sV + (c*16)*HD + k*16, HD);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(sTmp + (warp*16)*BN + c*16, acc, BN, wmma::mem_row_major);
        }
        __syncthreads();

        // dS = P*(dP - D)
        for (int idx = threadIdx.x; idx < BR*BN; idx += blockDim.x) {
            int i = idx / BN;
            float p = __bfloat162float(sP[idx]);
            sP[idx] = __float2bfloat16(p * (sTmp[idx] - sD[i]));
        }
        __syncthreads();

        // dQ += dS @ K
        for (int c = 0; c < HD/16; c++) {
            FragC acc; wmma::load_matrix_sync(acc, sdQ + (warp*16)*HD + c*16, HD, wmma::mem_row_major);
            for (int k = 0; k < BN/16; k++) {
                FragA_row a; wmma::load_matrix_sync(a, sP + (warp*16)*BN + k*16, BN);
                FragB_row b; wmma::load_matrix_sync(b, sK + (k*16)*HD + c*16, HD);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(sdQ + (warp*16)*HD + c*16, acc, HD, wmma::mem_row_major);
        }
        __syncthreads();
    }

    for (int idx = threadIdx.x; idx < BR*HD; idx += blockDim.x) {
        int r = idx >> 7, c = idx & 127;
        int gr = q_start + r;
        if (gr < S) {
            dQ[(size_t)bh*S*HD + (size_t)gr*HD + c] = __float2bfloat16(sdQ[idx] * scale);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int S = (int)Q.size(2);
    int d = (int)Q.size(3);
    int BH = B * H;
    (void)d;

    const __nv_bfloat16* Qp  = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* Kp  = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* Vp  = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* Op  = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dOp = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* Lp = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQp = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dKp = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dVp = static_cast<__nv_bfloat16*>(dV.data_ptr());

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    float* Dvec = nullptr;
    CUDA_CHECK(cudaMallocAsync(&Dvec, sizeof(float) * (size_t)BH * S, stream));

    // Precompute D = rowsum(dO * O)
    int total_rows = BH * S;
    int threadsD = 256;
    int warpsPerBlock = threadsD / 32;
    int blocksD = (total_rows + warpsPerBlock - 1) / warpsPerBlock;
    compute_D_kernel<<<blocksD, threadsD, 0, stream>>>(dOp, Op, Dvec, total_rows);

    int num_q = (S + BR - 1) / BR;
    int num_kv = (S + BN - 1) / BN;
    float scale = 1.0f / sqrtf((float)HD);

    size_t smem_kv = 156160;
    size_t smem_q  = 123392;
    CUDA_CHECK(cudaFuncSetAttribute(dkv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_kv));
    CUDA_CHECK(cudaFuncSetAttribute(dq_kernel,  cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_q));

    dim3 gkv(BH * num_kv);
    dkv_kernel<<<gkv, 128, smem_kv, stream>>>(Qp, Kp, Vp, dOp, Lp, Dvec, dKp, dVp,
                                              S, num_q, num_kv, scale);

    dim3 gq(BH * num_q);
    dq_kernel<<<gq, 128, smem_q, stream>>>(Qp, Kp, Vp, dOp, Lp, Dvec, dQp,
                                           S, num_q, num_kv, scale);

    CUDA_CHECK(cudaFreeAsync(Dvec, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd