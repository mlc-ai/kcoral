#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                       \
    cudaError_t _e = (call);                                        \
    if (_e != cudaSuccess) {                                        \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                 \
                cudaGetErrorString(_e), __FILE__, __LINE__);        \
        exit(1);                                                    \
    }                                                               \
} while(0)

namespace mha_causal {

constexpr int D       = 128;         // head dimension (fixed)
constexpr int BK      = 64;          // keys per iteration
constexpr int WARPS   = 16;          // query rows per block
constexpr int BM      = WARPS;
constexpr int THREADS = WARPS * 32;  // 512
constexpr int KPAD    = 130;         // Ksh row stride (bf16), padded to remove bank conflicts

__global__ void __launch_bounds__(THREADS)
mha_kernel(const __nv_bfloat16* __restrict__ Q,
           const __nv_bfloat16* __restrict__ K,
           const __nv_bfloat16* __restrict__ V,
           __nv_bfloat16* __restrict__ O,
           float* __restrict__ LSE,
           int B, int H, int S, float scale) {
    int b    = blockIdx.z;
    int h    = blockIdx.y;
    int q0   = blockIdx.x * BM;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int tid  = threadIdx.x;
    int qi   = q0 + warp;

    __shared__ __nv_bfloat16 Qsh[WARPS * D];
    __shared__ __nv_bfloat16 Ksh[BK * KPAD];
    __shared__ __nv_bfloat16 Vsh[BK * D];
    __shared__ float          Psh[WARPS * BK];

    long hb = ((long)(b * H + h)) * S; // row base (rows), multiply by D for elements

    // Load Q rows for this block
    for (int idx = tid; idx < WARPS * D; idx += THREADS) {
        int w = idx / D;
        int d = idx % D;
        int row = q0 + w;
        Qsh[idx] = (row < S) ? Q[(hb + row) * D + d] : __float2bfloat16(0.0f);
    }
    __syncthreads();

    float m = -1e30f, l_sum = 0.0f;
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;

    int maxq = q0 + BM - 1;
    if (maxq > S - 1) maxq = S - 1;
    int kb_end = maxq / BK;

    for (int kb = 0; kb <= kb_end; kb++) {
        int kbase = kb * BK;

        // Cooperative load of K, V tiles (bf162 vectorized)
        for (int idx = tid; idx < (BK * D) / 2; idx += THREADS) {
            int j  = idx / (D / 2);
            int d2 = (idx % (D / 2)) * 2;
            int kp = kbase + j;
            __nv_bfloat162 kv2, vv2;
            if (kp < S) {
                kv2 = *reinterpret_cast<const __nv_bfloat162*>(&K[(hb + kp) * D + d2]);
                vv2 = *reinterpret_cast<const __nv_bfloat162*>(&V[(hb + kp) * D + d2]);
            } else {
                kv2 = __floats2bfloat162_rn(0.f, 0.f);
                vv2 = kv2;
            }
            *reinterpret_cast<__nv_bfloat162*>(&Ksh[j * KPAD + d2]) = kv2;
            *reinterpret_cast<__nv_bfloat162*>(&Vsh[j * D + d2])    = vv2;
        }
        __syncthreads();

        if (qi < S) {
            int j0 = lane;
            int j1 = lane + 32;
            float s0 = 0.f, s1 = 0.f;
            #pragma unroll
            for (int d = 0; d < D; d += 2) {
                __nv_bfloat162 q2  = *reinterpret_cast<const __nv_bfloat162*>(&Qsh[warp * D + d]);
                __nv_bfloat162 k02 = *reinterpret_cast<const __nv_bfloat162*>(&Ksh[j0 * KPAD + d]);
                __nv_bfloat162 k12 = *reinterpret_cast<const __nv_bfloat162*>(&Ksh[j1 * KPAD + d]);
                float2 qf  = __bfloat1622float2(q2);
                float2 k0f = __bfloat1622float2(k02);
                float2 k1f = __bfloat1622float2(k12);
                s0 += qf.x * k0f.x + qf.y * k0f.y;
                s1 += qf.x * k1f.x + qf.y * k1f.y;
            }
            s0 *= scale;
            s1 *= scale;
            int pos0 = kbase + j0;
            int pos1 = kbase + j1;
            if (pos0 > qi) s0 = -1e30f;
            if (pos1 > qi) s1 = -1e30f;

            // block max over BK keys
            float lmax = fmaxf(s0, s1);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffff, lmax, off));

            float m_new = fmaxf(m, lmax);
            float corr = 1.0f;
            if (m_new > m) corr = __expf(m - m_new);

            float p0 = __expf(s0 - m_new);
            float p1 = __expf(s1 - m_new);
            float lsum = p0 + p1;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                lsum += __shfl_xor_sync(0xffffffff, lsum, off);

            l_sum = l_sum * corr + lsum;
            acc0 *= corr; acc1 *= corr; acc2 *= corr; acc3 *= corr;

            Psh[warp * BK + j0] = p0;
            Psh[warp * BK + j1] = p1;
            __syncwarp();

            // PV: lane owns output dims [4*lane, 4*lane+4)
            int dcol = lane * 4;
            #pragma unroll 4
            for (int j = 0; j < BK; j++) {
                float p = Psh[warp * BK + j];
                __nv_bfloat162 v01 = *reinterpret_cast<const __nv_bfloat162*>(&Vsh[j * D + dcol]);
                __nv_bfloat162 v23 = *reinterpret_cast<const __nv_bfloat162*>(&Vsh[j * D + dcol + 2]);
                float2 v01f = __bfloat1622float2(v01);
                float2 v23f = __bfloat1622float2(v23);
                acc0 += p * v01f.x;
                acc1 += p * v01f.y;
                acc2 += p * v23f.x;
                acc3 += p * v23f.y;
            }
            __syncwarp();
            m = m_new;
        }
        __syncthreads();
    }

    if (qi < S) {
        float inv = 1.0f / l_sum;
        int dcol = lane * 4;
        long ob = (hb + qi) * D + dcol;
        __nv_bfloat162 o01 = __floats2bfloat162_rn(acc0 * inv, acc1 * inv);
        __nv_bfloat162 o23 = __floats2bfloat162_rn(acc2 * inv, acc3 * inv);
        *reinterpret_cast<__nv_bfloat162*>(&O[ob])     = o01;
        *reinterpret_cast<__nv_bfloat162*>(&O[ob + 2]) = o23;
        if (lane == 0) {
            LSE[hb + qi] = m + logf(l_sum);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int B  = (int)Q.size(0);
    int H  = (int)Q.size(1);
    int S  = (int)Q.size(2);
    int Dd = (int)Q.size(3);

    const __nv_bfloat16* Qp = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* Kp = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* Vp = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* Op = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSEp = static_cast<float*>(LSE.data_ptr());

    float scale = 1.0f / sqrtf((float)Dd);

    dim3 grid((S + BM - 1) / BM, H, B);
    dim3 block(THREADS);
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    mha_kernel<<<grid, block, 0, stream>>>(Qp, Kp, Vp, Op, LSEp, B, H, S, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_causal::run);

}  // namespace mha_causal