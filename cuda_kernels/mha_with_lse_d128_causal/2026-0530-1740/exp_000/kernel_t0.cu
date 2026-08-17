#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__); exit(1);} } while(0)

namespace mha_kernel {

constexpr int Dc   = 128;
constexpr int BM   = 64;
constexpr int BN   = 64;
constexpr int PADQ = 130;   // padded row stride for Qs (odd #words -> conflict free)
constexpr float SCALE = 0.08838834764831843f;  // 1/sqrt(128)

__global__ void attn_kernel(const __nv_bfloat16* __restrict__ Q,
                            const __nv_bfloat16* __restrict__ K,
                            const __nv_bfloat16* __restrict__ V,
                            __nv_bfloat16* __restrict__ O,
                            float* __restrict__ LSE,
                            int S) {
    extern __shared__ char smem_raw[];
    __nv_bfloat16* Qs = (__nv_bfloat16*)smem_raw;
    __nv_bfloat16* Ks = Qs + BM * PADQ;
    __nv_bfloat16* Vs = Ks + BN * Dc;

    int qtile = blockIdx.x;
    int bh    = blockIdx.y;
    int r     = threadIdx.x;
    int q0    = qtile * BM;
    int qrow  = q0 + r;

    long bhoff = (long)bh * S * Dc;
    const __nv_bfloat16* Qbase = Q + bhoff;
    const __nv_bfloat16* Kbase = K + bhoff;
    const __nv_bfloat16* Vbase = V + bhoff;
    __nv_bfloat16* Obase = O + bhoff;
    float* LSEbase = LSE + (long)bh * S;

    // Load this thread's Q row into shared memory
    if (qrow < S) {
        const __nv_bfloat16* qp = Qbase + (long)qrow * Dc;
        #pragma unroll
        for (int d = 0; d < Dc; d++) Qs[r * PADQ + d] = qp[d];
    }

    float acc[Dc];
    #pragma unroll
    for (int d = 0; d < Dc; d++) acc[d] = 0.f;
    float m = -INFINITY;
    float l = 0.f;

    int nkb   = (q0 + BM - 1) / BN + 1;
    int nkb_s = (S + BN - 1) / BN;
    if (nkb > nkb_s) nkb = nkb_s;

    for (int kb = 0; kb < nkb; kb++) {
        int k0 = kb * BN;
        __syncthreads();
        // Load K,V tile (thread r loads row r)
        int krow = k0 + r;
        if (krow < S) {
            const __nv_bfloat16* kp = Kbase + (long)krow * Dc;
            const __nv_bfloat16* vp = Vbase + (long)krow * Dc;
            #pragma unroll
            for (int d = 0; d < Dc; d++) { Ks[r * Dc + d] = kp[d]; Vs[r * Dc + d] = vp[d]; }
        } else {
            #pragma unroll
            for (int d = 0; d < Dc; d++) { Ks[r * Dc + d] = (__nv_bfloat16)0; Vs[r * Dc + d] = (__nv_bfloat16)0; }
        }
        __syncthreads();

        // S = Q . K^T  (one row vs BN keys)
        float s[BN];
        #pragma unroll
        for (int k = 0; k < BN; k++) s[k] = 0.f;
        for (int d = 0; d < Dc; d++) {
            float q = __bfloat162float(Qs[r * PADQ + d]);
            #pragma unroll
            for (int k = 0; k < BN; k++)
                s[k] += q * __bfloat162float(Ks[k * Dc + d]);
        }

        // scale + causal mask + block max
        float blockmax = -INFINITY;
        #pragma unroll
        for (int k = 0; k < BN; k++) {
            int kj = k0 + k;
            float sv = s[k] * SCALE;
            if (kj > qrow || kj >= S) sv = -INFINITY;
            s[k] = sv;
            blockmax = fmaxf(blockmax, sv);
        }

        // online softmax update
        float mnew = fmaxf(m, blockmax);
        float corr = __expf(m - mnew);   // 0 when m=-inf, 1 when block fully masked
        #pragma unroll
        for (int d = 0; d < Dc; d++) acc[d] *= corr;
        l *= corr;

        for (int k = 0; k < BN; k++) {
            float p = __expf(s[k] - mnew);   // 0 for masked positions
            l += p;
            #pragma unroll
            for (int d = 0; d < Dc; d++)
                acc[d] += p * __bfloat162float(Vs[k * Dc + d]);
        }
        m = mnew;
    }

    if (qrow < S) {
        float inv = 1.f / l;
        __nv_bfloat16* op = Obase + (long)qrow * Dc;
        #pragma unroll
        for (int d = 0; d < Dc; d++) op[d] = __float2bfloat16(acc[d] * inv);
        LSEbase[qrow] = m + logf(l);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);

    const __nv_bfloat16* Qp = (const __nv_bfloat16*)Q.data_ptr();
    const __nv_bfloat16* Kp = (const __nv_bfloat16*)K.data_ptr();
    const __nv_bfloat16* Vp = (const __nv_bfloat16*)V.data_ptr();
    __nv_bfloat16* Op = (__nv_bfloat16*)O.data_ptr();
    float* LSEp = (float*)LSE.data_ptr();

    dim3 grid((S + BM - 1) / BM, B * H);
    dim3 block(BM);
    size_t shmem = (size_t)(BM * PADQ + 2 * BN * Dc) * sizeof(__nv_bfloat16);

    cudaStream_t stream =
        (cudaStream_t)TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id);

    cudaFuncSetAttribute((const void*)attn_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);

    attn_kernel<<<grid, block, shmem, stream>>>(Qp, Kp, Vp, Op, LSEp, S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel::run);

}  // namespace mha_kernel