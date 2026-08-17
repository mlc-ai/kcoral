#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
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

namespace mha_causal {

constexpr int D  = 128;   // head dim (fixed)
constexpr int WR = 16;    // warps per block == query rows per block
constexpr int BC = 64;    // keys per tile
constexpr int THREADS = 32 * WR;

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float warpReduceSum(float v){
    #pragma unroll
    for(int o=16;o>0;o>>=1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__global__ __launch_bounds__(THREADS) void attn_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S, float scale_log2)
{
    __shared__ __align__(16) __nv_bfloat16 Ks[BC*D];
    __shared__ __align__(16) __nv_bfloat16 Vs[BC*D];

    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;

    const int qblock = blockIdx.x;
    const int h      = blockIdx.y;
    const int b      = blockIdx.z;

    const int qrow = qblock*WR + warp_id;

    const int64_t bh = (int64_t)(b*H + h);
    const __nv_bfloat16* Qbh = Q + bh*S*D;
    const __nv_bfloat16* Kbh = K + bh*S*D;
    const __nv_bfloat16* Vbh = V + bh*S*D;
    __nv_bfloat16* Obh = O + bh*S*D;
    float* LSEbh = LSE + bh*S;

    // Load this lane's 4-element slice of the Q row into registers.
    float qreg[4];
    {
        int c = lane*4;
        if (qrow < S) {
            const __nv_bfloat162* qp =
                reinterpret_cast<const __nv_bfloat162*>(Qbh + (int64_t)qrow*D + c);
            float2 a  = __bfloat1622float2(qp[0]);
            float2 b2 = __bfloat1622float2(qp[1]);
            qreg[0]=a.x; qreg[1]=a.y; qreg[2]=b2.x; qreg[3]=b2.y;
        } else {
            qreg[0]=qreg[1]=qreg[2]=qreg[3]=0.f;
        }
    }

    float acc[4] = {0.f,0.f,0.f,0.f};
    float m = -INFINITY;   // running max (log2 units)
    float l = 0.f;         // running denominator

    const int qmax   = qblock*WR + WR - 1;
    const int kv_end = min(S, qmax+1);
    const int num_kb = (kv_end + BC - 1)/BC;

    int4* Ks4 = reinterpret_cast<int4*>(Ks);
    int4* Vs4 = reinterpret_cast<int4*>(Vs);
    const int totalVec = (BC*D)/8;   // int4 == 8 bf16

    for(int kb=0; kb<num_kb; ++kb){
        const int kv_start = kb*BC;

        __syncthreads();
        for(int iv=threadIdx.x; iv<totalVec; iv+=THREADS){
            int elem = iv*8;
            int r = elem / D;
            int c = elem - r*D;
            int gk = kv_start + r;
            int4 kk, vv;
            if(gk < S){
                kk = *reinterpret_cast<const int4*>(Kbh + (int64_t)gk*D + c);
                vv = *reinterpret_cast<const int4*>(Vbh + (int64_t)gk*D + c);
            } else {
                kk = make_int4(0,0,0,0); vv = kk;
            }
            Ks4[iv]=kk; Vs4[iv]=vv;
        }
        __syncthreads();

        #pragma unroll 1
        for(int jj=0; jj<BC; ++jj){
            int kpos = kv_start + jj;
            if (kpos > qrow) break;   // causal (uniform within warp)

            const __nv_bfloat162* kp =
                reinterpret_cast<const __nv_bfloat162*>(&Ks[jj*D + lane*4]);
            float2 k01 = __bfloat1622float2(kp[0]);
            float2 k23 = __bfloat1622float2(kp[1]);
            float pd = qreg[0]*k01.x + qreg[1]*k01.y + qreg[2]*k23.x + qreg[3]*k23.y;
            pd = warpReduceSum(pd);
            float sj = pd * scale_log2;   // log2 units

            float m_new = fmaxf(m, sj);
            float p;
            if (lane==0) p = fast_exp2f_fn(sj - m_new);
            p = __shfl_sync(0xffffffffu, p, 0);

            if (m_new != m){              // uniform across warp
                float corr;
                if (lane==0) corr = fast_exp2f_fn(m - m_new);
                corr = __shfl_sync(0xffffffffu, corr, 0);
                acc[0]*=corr; acc[1]*=corr; acc[2]*=corr; acc[3]*=corr;
                l *= corr;
                m = m_new;
            }
            l += p;

            const __nv_bfloat162* vp =
                reinterpret_cast<const __nv_bfloat162*>(&Vs[jj*D + lane*4]);
            float2 v01 = __bfloat1622float2(vp[0]);
            float2 v23 = __bfloat1622float2(vp[1]);
            acc[0]+=p*v01.x; acc[1]+=p*v01.y; acc[2]+=p*v23.x; acc[3]+=p*v23.y;
        }
    }

    if(qrow < S){
        float inv = 1.0f / l;
        int c = lane*4;
        __nv_bfloat162 o01 = __floats2bfloat162_rn(acc[0]*inv, acc[1]*inv);
        __nv_bfloat162 o23 = __floats2bfloat162_rn(acc[2]*inv, acc[3]*inv);
        __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(Obh + (int64_t)qrow*D + c);
        op[0]=o01; op[1]=o23;
        if(lane==0){
            LSEbh[qrow] = m * 0.6931471805599453f + logf(l);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int S = (int)Q.size(2);
    int Dd = (int)Q.size(3);

    float scale = 1.0f / sqrtf((float)Dd);
    float scale_log2 = scale * 1.4426950408889634f; // * log2(e)

    dim3 grid((S + WR - 1)/WR, H, B);
    dim3 block(THREADS);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    attn_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<__nv_bfloat16*>(O.data_ptr()),
        static_cast<float*>(LSE.data_ptr()),
        B, H, S, scale_log2);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_causal::run);

}  // namespace mha_causal