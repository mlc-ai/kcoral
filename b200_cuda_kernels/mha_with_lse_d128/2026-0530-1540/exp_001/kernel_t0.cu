#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
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

namespace mha_d128 {

constexpr int D  = 128;   // head dim (fixed)
constexpr int BM = 128;   // queries per block
constexpr int BN = 32;    // keys per tile

__global__ void __launch_bounds__(128) flash_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S, float scale)
{
    int q_tile = blockIdx.x;
    int h      = blockIdx.y;
    int b      = blockIdx.z;
    int tid    = threadIdx.x;

    long bh = (long)(b * H + h) * S;
    const __nv_bfloat16* Qbh = Q + bh * D;
    const __nv_bfloat16* Kbh = K + bh * D;
    const __nv_bfloat16* Vbh = V + bh * D;
    __nv_bfloat16* Obh = O + bh * D;
    float* LSEbh = LSE + bh;

    extern __shared__ char smem[];
    __nv_bfloat16* Qs = (__nv_bfloat16*)smem;          // [D][BM] transposed
    __nv_bfloat16* Ks = Qs + D * BM;                   // [BN][D]
    __nv_bfloat16* Vs = Ks + BN * D;                   // [BN][D]

    const __nv_bfloat16 ZERO = __float2bfloat16(0.f);

    // Load Q tile transposed: Qs[d*BM + j]
    for (int idx = tid; idx < BM * D; idx += blockDim.x) {
        int j = idx / D;
        int d = idx % D;
        int r = q_tile * BM + j;
        Qs[d * BM + j] = (r < S) ? Qbh[(long)r * D + d] : ZERO;
    }
    __syncthreads();

    int row = q_tile * BM + tid;

    float acc[D];
    #pragma unroll
    for (int d = 0; d < D; d++) acc[d] = 0.f;
    float m = -1e30f;
    float l = 0.f;

    int num_kv = (S + BN - 1) / BN;
    for (int kv = 0; kv < num_kv; kv++) {
        int base = kv * BN;

        __syncthreads();
        for (int idx = tid; idx < BN * D; idx += blockDim.x) {
            int j = idx / D;
            int d = idx % D;
            int kk = base + j;
            bool ok = kk < S;
            Ks[j * D + d] = ok ? Kbh[(long)kk * D + d] : ZERO;
            Vs[j * D + d] = ok ? Vbh[(long)kk * D + d] : ZERO;
        }
        __syncthreads();

        // ---- QK^T : scores for this thread's query row vs BN keys ----
        float s[BN];
        #pragma unroll
        for (int j = 0; j < BN; j++) s[j] = 0.f;

        #pragma unroll
        for (int d = 0; d < D; d++) {
            float qv = __bfloat162float(Qs[d * BM + tid]);
            #pragma unroll
            for (int j = 0; j < BN; j++) {
                s[j] += qv * __bfloat162float(Ks[j * D + d]);
            }
        }

        // ---- scale + mask + new running max ----
        float m_new = m;
        #pragma unroll
        for (int j = 0; j < BN; j++) {
            int kk = base + j;
            s[j] = (kk < S) ? s[j] * scale : -1e30f;
            m_new = fmaxf(m_new, s[j]);
        }

        // ---- rescale accumulator ----
        float corr = __expf(m - m_new);
        #pragma unroll
        for (int d = 0; d < D; d++) acc[d] *= corr;
        l *= corr;

        // ---- P @ V ----
        #pragma unroll
        for (int j = 0; j < BN; j++) {
            float p = __expf(s[j] - m_new);
            l += p;
            #pragma unroll
            for (int d = 0; d < D; d++) {
                acc[d] += p * __bfloat162float(Vs[j * D + d]);
            }
        }
        m = m_new;
    }

    if (row < S) {
        float inv = 1.f / l;
        #pragma unroll
        for (int d = 0; d < D; d++) {
            Obh[(long)row * D + d] = __float2bfloat16(acc[d] * inv);
        }
        LSEbh[row] = m + logf(l);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int S = (int)Q.size(2);
    // D is fixed at 128

    float scale = 1.0f / sqrtf((float)D);

    const __nv_bfloat16* Qp = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* Kp = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* Vp = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* Op = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* Lp = static_cast<float*>(LSE.data_ptr());

    dim3 grid((S + BM - 1) / BM, H, B);
    dim3 block(BM);

    size_t smem = (size_t)(D * BM + 2 * BN * D) * sizeof(__nv_bfloat16);

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    cudaFuncSetAttribute(flash_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);

    flash_kernel<<<grid, block, smem, stream>>>(Qp, Kp, Vp, Op, Lp, B, H, S, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_d128::run);

}  // namespace mha_d128