#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>
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

// ---- tile loader: load [64 rows][128 cols] bf16 from global into shared ----
__device__ __forceinline__ void load_tile(const __nv_bfloat16* Xbh, int S, int row0,
                                           __nv_bfloat16* sTile, int tid) {
    const int4* src = reinterpret_cast<const int4*>(Xbh);
    int4* dst = reinterpret_cast<int4*>(sTile);
    // 64 rows * 16 int4/row = 1024 int4
    #pragma unroll
    for (int idx = tid; idx < 1024; idx += 256) {
        int r = idx >> 4;    // /16
        int cg = idx & 15;   // %16
        int grow = row0 + r;
        if (grow < S) dst[idx] = src[(size_t)grow * 16 + cg];
        else          dst[idx] = make_int4(0, 0, 0, 0);
    }
}

// ---- delta: D[row] = sum_d O[row,d]*dO[row,d] ----
__global__ void delta_kernel(const __nv_bfloat16* O, const __nv_bfloat16* dO,
                             float* Delta, int R, int HD) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= R) return;
    const __nv_bfloat16* Op  = O  + (size_t)warp * HD;
    const __nv_bfloat16* dOp = dO + (size_t)warp * HD;
    float s = 0.f;
    for (int k = lane; k < HD; k += 32)
        s += __bfloat162float(Op[k]) * __bfloat162float(dOp[k]);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        s += __shfl_down_sync(0xffffffff, s, off);
    if (lane == 0) Delta[warp] = s;
}

// ---- dK / dV kernel ----
template<int HD, int BM, int BN>
__global__ __launch_bounds__(256) void bwd_dkv_kernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* dO, const float* L, const float* Delta,
    __nv_bfloat16* dK, __nv_bfloat16* dV,
    int B, int H, int S, float scale) {

    int jb = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
    int numBlk = (S + BN - 1) / BN;
    int j0 = jb * BN;
    size_t bh_off  = (size_t)(b * H + h) * S * HD;
    size_t bh_offL = (size_t)(b * H + h) * S;
    const __nv_bfloat16* Qbh  = Q  + bh_off;
    const __nv_bfloat16* Kbh  = K  + bh_off;
    const __nv_bfloat16* Vbh  = V  + bh_off;
    const __nv_bfloat16* dObh = dO + bh_off;
    const float* Lbh = L + bh_offL;
    const float* Dbh = Delta + bh_offL;

    extern __shared__ char smem[];
    __nv_bfloat16* sQ  = (__nv_bfloat16*)smem;
    __nv_bfloat16* sK  = sQ + BM * HD;
    __nv_bfloat16* sV  = sK + BN * HD;
    __nv_bfloat16* sdO = sV + BN * HD;
    float* sP  = (float*)(sdO + BM * HD);
    float* sdP = sP + BM * BN;
    float* sL  = sdP + BM * BN;
    float* sD  = sL + BM;

    int tid = threadIdx.x;
    load_tile(Kbh, S, j0, sK, tid);
    load_tile(Vbh, S, j0, sV, tid);
    __syncthreads();

    float dV_acc[4][8], dK_acc[4][8];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 8; j++) { dV_acc[i][j] = 0.f; dK_acc[i][j] = 0.f; }

    int mm_ty = tid >> 4, mm_tx = tid & 15;
    int acc_n0 = (tid >> 4) * 4, acc_k0 = (tid & 15) * 8;

    for (int ib = jb; ib < numBlk; ib++) {
        int i0 = ib * BM;
        load_tile(Qbh,  S, i0, sQ,  tid);
        load_tile(dObh, S, i0, sdO, tid);
        for (int t = tid; t < BM; t += 256) {
            int gr = i0 + t;
            sL[t] = (gr < S) ? Lbh[gr] : 0.f;
            sD[t] = (gr < S) ? Dbh[gr] : 0.f;
        }
        __syncthreads();

        // matmul: sc = Q.K^T , dc = dO.V^T
        float sc[4][4], dc[4][4];
        #pragma unroll
        for (int i = 0; i < 4; i++)
            #pragma unroll
            for (int j = 0; j < 4; j++) { sc[i][j] = 0.f; dc[i][j] = 0.f; }
        #pragma unroll 4
        for (int k = 0; k < HD; k++) {
            float qv[4], kv[4], ov[4], vv[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                qv[i] = __bfloat162float(sQ [(mm_ty*4+i)*HD + k]);
                ov[i] = __bfloat162float(sdO[(mm_ty*4+i)*HD + k]);
            }
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                kv[j] = __bfloat162float(sK[(mm_tx*4+j)*HD + k]);
                vv[j] = __bfloat162float(sV[(mm_tx*4+j)*HD + k]);
            }
            #pragma unroll
            for (int i = 0; i < 4; i++)
                #pragma unroll
                for (int j = 0; j < 4; j++) { sc[i][j] += qv[i]*kv[j]; dc[i][j] += ov[i]*vv[j]; }
        }
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int m = mm_ty*4+i; int qi = i0 + m; float Lm = sL[m], Dm = sD[m];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                int n = mm_tx*4+j; int kj = j0 + n;
                float score = sc[i][j] * scale;
                float P = 0.f;
                if (qi < S && kj < S && kj <= qi) P = __expf(score - Lm);
                float dS = P * (dc[i][j] - Dm);
                sP[m*BN + n]  = P;
                sdP[m*BN + n] = dS;
            }
        }
        __syncthreads();

        // accumulate dV[n,k] += sum_m P[m,n]*dO[m,k] ; dK[n,k] += sum_m dS[m,n]*Q[m,k]
        #pragma unroll 4
        for (int m = 0; m < BM; m++) {
            float p4[4], ds4[4], o8[8], q8[8];
            #pragma unroll
            for (int i = 0; i < 4; i++) { p4[i] = sP[m*BN+acc_n0+i]; ds4[i] = sdP[m*BN+acc_n0+i]; }
            #pragma unroll
            for (int j = 0; j < 8; j++) { o8[j] = __bfloat162float(sdO[m*HD+acc_k0+j]);
                                          q8[j] = __bfloat162float(sQ [m*HD+acc_k0+j]); }
            #pragma unroll
            for (int i = 0; i < 4; i++)
                #pragma unroll
                for (int j = 0; j < 8; j++) { dV_acc[i][j] += p4[i]*o8[j]; dK_acc[i][j] += ds4[i]*q8[j]; }
        }
        __syncthreads();
    }

    // write
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int n = acc_n0 + i; int kj = j0 + n; if (kj >= S) continue;
        __nv_bfloat16* dKp = dK + bh_off + (size_t)kj * HD;
        __nv_bfloat16* dVp = dV + bh_off + (size_t)kj * HD;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int k = acc_k0 + j;
            dVp[k] = __float2bfloat16(dV_acc[i][j]);
            dKp[k] = __float2bfloat16(dK_acc[i][j] * scale);
        }
    }
}

// ---- dQ kernel ----
template<int HD, int BM, int BN>
__global__ __launch_bounds__(256) void bwd_dq_kernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* dO, const float* L, const float* Delta,
    __nv_bfloat16* dQ, int B, int H, int S, float scale) {

    int ib = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
    int i0 = ib * BM;
    size_t bh_off  = (size_t)(b * H + h) * S * HD;
    size_t bh_offL = (size_t)(b * H + h) * S;
    const __nv_bfloat16* Qbh  = Q  + bh_off;
    const __nv_bfloat16* Kbh  = K  + bh_off;
    const __nv_bfloat16* Vbh  = V  + bh_off;
    const __nv_bfloat16* dObh = dO + bh_off;
    const float* Lbh = L + bh_offL;
    const float* Dbh = Delta + bh_offL;

    extern __shared__ char smem[];
    __nv_bfloat16* sQ  = (__nv_bfloat16*)smem;
    __nv_bfloat16* sK  = sQ + BM * HD;
    __nv_bfloat16* sV  = sK + BN * HD;
    __nv_bfloat16* sdO = sV + BN * HD;
    float* sP  = (float*)(sdO + BM * HD);
    float* sdP = sP + BM * BN;
    float* sL  = sdP + BM * BN;
    float* sD  = sL + BM;

    int tid = threadIdx.x;
    load_tile(Qbh,  S, i0, sQ,  tid);
    load_tile(dObh, S, i0, sdO, tid);
    for (int t = tid; t < BM; t += 256) {
        int gr = i0 + t;
        sL[t] = (gr < S) ? Lbh[gr] : 0.f;
        sD[t] = (gr < S) ? Dbh[gr] : 0.f;
    }
    __syncthreads();

    float dQ_acc[4][8];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 8; j++) dQ_acc[i][j] = 0.f;

    int mm_ty = tid >> 4, mm_tx = tid & 15;
    int acc_m0 = (tid >> 4) * 4, acc_k0 = (tid & 15) * 8;

    for (int jb = 0; jb <= ib; jb++) {
        int j0 = jb * BN;
        load_tile(Kbh, S, j0, sK, tid);
        load_tile(Vbh, S, j0, sV, tid);
        __syncthreads();

        float sc[4][4], dc[4][4];
        #pragma unroll
        for (int i = 0; i < 4; i++)
            #pragma unroll
            for (int j = 0; j < 4; j++) { sc[i][j] = 0.f; dc[i][j] = 0.f; }
        #pragma unroll 4
        for (int k = 0; k < HD; k++) {
            float qv[4], kv[4], ov[4], vv[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                qv[i] = __bfloat162float(sQ [(mm_ty*4+i)*HD + k]);
                ov[i] = __bfloat162float(sdO[(mm_ty*4+i)*HD + k]);
            }
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                kv[j] = __bfloat162float(sK[(mm_tx*4+j)*HD + k]);
                vv[j] = __bfloat162float(sV[(mm_tx*4+j)*HD + k]);
            }
            #pragma unroll
            for (int i = 0; i < 4; i++)
                #pragma unroll
                for (int j = 0; j < 4; j++) { sc[i][j] += qv[i]*kv[j]; dc[i][j] += ov[i]*vv[j]; }
        }
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int m = mm_ty*4+i; int qi = i0 + m; float Lm = sL[m], Dm = sD[m];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                int n = mm_tx*4+j; int kj = j0 + n;
                float score = sc[i][j] * scale;
                float P = 0.f;
                if (qi < S && kj < S && kj <= qi) P = __expf(score - Lm);
                float dS = P * (dc[i][j] - Dm);
                sdP[m*BN + n] = dS;
            }
        }
        __syncthreads();

        // accumulate dQ[m,k] += sum_n dS[m,n]*K[n,k]
        #pragma unroll 4
        for (int n = 0; n < BN; n++) {
            float ds4[4], k8[8];
            #pragma unroll
            for (int i = 0; i < 4; i++) ds4[i] = sdP[(acc_m0+i)*BN + n];
            #pragma unroll
            for (int j = 0; j < 8; j++) k8[j] = __bfloat162float(sK[n*HD + acc_k0 + j]);
            #pragma unroll
            for (int i = 0; i < 4; i++)
                #pragma unroll
                for (int j = 0; j < 8; j++) dQ_acc[i][j] += ds4[i]*k8[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int m = acc_m0 + i; int qi = i0 + m; if (qi >= S) continue;
        __nv_bfloat16* dQp = dQ + bh_off + (size_t)qi * HD;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int k = acc_k0 + j;
            dQp[k] = __float2bfloat16(dQ_acc[i][j] * scale);
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

    const int HD = 128, BM = 64, BN = 64;
    float scale = 1.0f / sqrtf((float)d);

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    const __nv_bfloat16* Qp  = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* Kp  = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* Vp  = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* Op  = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dOp = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* Lp = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQp = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dKp = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dVp = static_cast<__nv_bfloat16*>(dV.data_ptr());

    // scratch delta
    float* Delta = nullptr;
    size_t dbytes = (size_t)B * H * S * sizeof(float);
    CUDA_CHECK(cudaMallocAsync((void**)&Delta, dbytes, stream));

    int R = B * H * S;
    int dblocks = (R + 7) / 8;
    delta_kernel<<<dblocks, 256, 0, stream>>>(Op, dOp, Delta, R, HD);
    CUDA_CHECK(cudaGetLastError());

    int smem = 4 * BM * HD * (int)sizeof(__nv_bfloat16)
             + 2 * BM * BN * (int)sizeof(float)
             + 2 * BM * (int)sizeof(float);

    static bool attr_set = false;
    if (!attr_set) {
        CUDA_CHECK(cudaFuncSetAttribute(bwd_dkv_kernel<HD, BM, BN>,
                   cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        CUDA_CHECK(cudaFuncSetAttribute(bwd_dq_kernel<HD, BM, BN>,
                   cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        attr_set = true;
    }

    int numBlk = (S + BN - 1) / BN;
    dim3 grid(numBlk, H, B);

    bwd_dkv_kernel<HD, BM, BN><<<grid, 256, smem, stream>>>(
        Qp, Kp, Vp, dOp, Lp, Delta, dKp, dVp, B, H, S, scale);
    CUDA_CHECK(cudaGetLastError());

    bwd_dq_kernel<HD, BM, BN><<<grid, 256, smem, stream>>>(
        Qp, Kp, Vp, dOp, Lp, Delta, dQp, B, H, S, scale);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFreeAsync(Delta, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd