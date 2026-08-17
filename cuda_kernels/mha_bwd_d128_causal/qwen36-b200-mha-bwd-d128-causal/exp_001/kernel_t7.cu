#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t _e = (call); \
        if (_e != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
            exit(1); \
        } \
    } while(0)

namespace mha_bwd_ns {

static constexpr int HEAD_DIM = 128;
static constexpr int NTHREADS = 256;
static constexpr int CT = HEAD_DIM / 32;
static constexpr int BS = 64;

__device__ __forceinline__ float wsum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_down_sync(0xFFFFFFFF, v, o);
    return __shfl_sync(0xFFFFFFFF, v, 0);
}

extern __shared__ __align__(16) unsigned char smem[];

__global__ void kern_dQ(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V, const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L, __nv_bfloat16* __restrict__ dQ_out, float* __restrict__ mbuf,
    int B, int H, int S, int D) {

    int bh = blockIdx.y;
    if (bh >= B * H) return;
    int64_t base = (int64_t)bh * S * D;
    int64_t loff = (int64_t)bh * S;

    __nv_bfloat16* sK = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sV = sK + BS * D;

    int tid = threadIdx.x;
    int qi_base = blockIdx.x * BS;
    int lane = tid & 31;

    // Each block processes BS query rows
    // Thread processes cols, iterates over q-offset
    for (int qoff = tid / 32; qoff < BS; qoff += NTHREADS / 32) {
        int qi = qi_base + qoff;
        if (qi >= S) continue;

        float sc = rsqrtf((float)D);
        float Lqi = L[loff + qi];

        const __nv_bfloat16* pQ = Q + base + qi * D;
        const __nv_bfloat16* pdO = dO + base + qi * D;
        __nv_bfloat16* pdQ = dQ_out + base + qi * D;

        float qv[CT], dov[CT];
        for (int c = 0; c < CT; c++) {
            int co = lane * CT + c;
            qv[c] = __bfloat162float(pQ[co]);
            dov[c] = __bfloat162float(pdO[co]);
        }

        float dq[CT] = {}, ak[CT] = {};
        float ms = 0.f;

        for (int tj = 0; tj * BS <= qi && tj * BS < S; tj++) {
            int ks = tj * BS, ke = min(ks + BS, S), sz = ke - ks;

            for (int i = tid; i < sz * D; i += NTHREADS) {
                int r = i / D, c = i % D;
                sK[r * D + c] = K[base + (ks + r) * D + c];
                sV[r * D + c] = V[base + (ks + r) * D + c];
            }
            __syncthreads();

            for (int kr = 0; kr < sz; kr++) {
                int kj = ks + kr;
                if (kj > qi) break;

                float kv[CT], vv[CT];
                for (int c = 0; c < CT; c++) {
                    int co = lane * CT + c;
                    kv[c] = __bfloat162float(sK[kr * D + co]);
                    vv[c] = __bfloat162float(sV[kr * D + co]);
                }

                float st = 0.f, dt = 0.f;
                #pragma unroll
                for (int c = 0; c < CT; c++) {
                    st += qv[c] * kv[c];
                    dt += dov[c] * vv[c];
                }

                float sf = wsum(st), df = wsum(dt);
                float att = expf(sf * sc - Lqi);
                float adp = att * df;
                ms += adp;

                float sadp = adp * sc;
                #pragma unroll
                for (int c = 0; c < CT; c++) {
                    dq[c] += sadp * kv[c];
                    ak[c] += att * sc * kv[c];
                }
            }
            __syncthreads();
        }

        for (int c = 0; c < CT; c++) {
            int co = lane * CT + c;
            pdQ[co] = __float2bfloat16(dq[c] - ms * ak[c]);
        }
        if (lane == 0) mbuf[loff + qi] = ms;
    }
}

__global__ void kern_dK(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V, const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L, const float* __restrict__ mbuf,
    __nv_bfloat16* __restrict__ dK_out, int B, int H, int S, int D) {

    int bh = blockIdx.y;
    if (bh >= B * H) return;
    int64_t base = (int64_t)bh * S * D;
    int64_t loff = (int64_t)bh * S;

    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sdO = sQ + BS * D;

    int tid = threadIdx.x;
    int kj_base = blockIdx.x * BS;
    int lane = tid & 31;

    for (int koff = tid / 32; koff < BS; koff += NTHREADS / 32) {
        int kj = kj_base + koff;
        if (kj >= S) continue;

        float sc = rsqrtf((float)D);

        const __nv_bfloat16* pK = K + base + kj * D;
        const __nv_bfloat16* pV = V + base + kj * D;

        float kv[CT], vv[CT];
        for (int c = 0; c < CT; c++) {
            int co = lane * CT + c;
            kv[c] = __bfloat162float(pK[co]);
            vv[c] = __bfloat162float(pV[co]);
        }

        float dk[CT] = {};

        for (int ti = kj / BS; ti * BS < S; ti++) {
            int qs = ti * BS, qe = min(qs + BS, S), sz = qe - qs;
            if (qs >= S) break;
            if (qe <= kj) continue;

            for (int i = tid; i < sz * D; i += NTHREADS) {
                int r = i / D, c = i % D;
                sQ[r * D + c] = Q[base + (qs + r) * D + c];
                sdO[r * D + c] = dO[base + (qs + r) * D + c];
            }
            __syncthreads();

            for (int qr = 0; qr < sz; qr++) {
                int qi = qs + qr;
                if (qi < kj) continue;

                float qv[CT], dov[CT];
                for (int c = 0; c < CT; c++) {
                    int co = lane * CT + c;
                    qv[c] = __bfloat162float(sQ[qr * D + co]);
                    dov[c] = __bfloat162float(sdO[qr * D + co]);
                }

                float st = 0.f, dt = 0.f;
                #pragma unroll
                for (int c = 0; c < CT; c++) {
                    st += qv[c] * kv[c];
                    dt += dov[c] * vv[c];
                }

                float sf = wsum(st), df = wsum(dt);
                float att = expf(sf * sc - L[loff + qi]);
                float mq = mbuf[loff + qi];
                float ds = att * (df - mq) * sc;

                #pragma unroll
                for (int c = 0; c < CT; c++)
                    dk[c] += ds * qv[c];
            }
            __syncthreads();
        }

        __nv_bfloat16* pdK = dK_out + base + kj * D;
        for (int c = 0; c < CT; c++) {
            int co = lane * CT + c;
            pdK[co] = __float2bfloat16(dk[c]);
        }
    }
}

__global__ void kern_dV(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ dO, const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dV_out, int B, int H, int S, int D) {

    int bh = blockIdx.y;
    if (bh >= B * H) return;
    int64_t base = (int64_t)bh * S * D;
    int64_t loff = (int64_t)bh * S;

    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sdO = sQ + BS * D;

    int tid = threadIdx.x;
    int kj_base = blockIdx.x * BS;
    int lane = tid & 31;

    for (int koff = tid / 32; koff < BS; koff += NTHREADS / 32) {
        int kj = kj_base + koff;
        if (kj >= S) continue;

        float sc = rsqrtf((float)D);

        const __nv_bfloat16* pK = K + base + kj * D;
        float kv[CT];
        for (int c = 0; c < CT; c++) {
            int co = lane * CT + c;
            kv[c] = __bfloat162float(pK[co]);
        }

        float dv[CT] = {};

        for (int ti = kj / BS; ti * BS < S; ti++) {
            int qs = ti * BS, qe = min(qs + BS, S), sz = qe - qs;
            if (qs >= S) break;
            if (qe <= kj) continue;

            for (int i = tid; i < sz * D; i += NTHREADS) {
                int r = i / D, c = i % D;
                sQ[r * D + c] = Q[base + (qs + r) * D + c];
                sdO[r * D + c] = dO[base + (qs + r) * D + c];
            }
            __syncthreads();

            for (int qr = 0; qr < sz; qr++) {
                int qi = qs + qr;
                if (qi < kj) continue;

                float qv[CT], dov[CT];
                for (int c = 0; c < CT; c++) {
                    int co = lane * CT + c;
                    qv[c] = __bfloat162float(sQ[qr * D + co]);
                    dov[c] = __bfloat162float(sdO[qr * D + co]);
                }

                float st = 0.f;
                #pragma unroll
                for (int c = 0; c < CT; c++)
                    st += qv[c] * kv[c];

                float sf = wsum(st);
                float att = expf(sf * sc - L[loff + qi]);

                #pragma unroll
                for (int c = 0; c < CT; c++)
                    dv[c] += att * dov[c];
            }
            __syncthreads();
        }

        __nv_bfloat16* pdV = dV_out + base + kj * D;
        for (int c = 0; c < CT; c++) {
            int co = lane * CT + c;
            pdV[co] = __float2bfloat16(dv[c]);
        }
    }
}

void run(tvm::ffi::TensorView Q_tv, tvm::ffi::TensorView K_tv, tvm::ffi::TensorView V_tv,
         tvm::ffi::TensorView O_tv, tvm::ffi::TensorView dO_tv, tvm::ffi::TensorView L_tv,
         tvm::ffi::TensorView dQ_tv, tvm::ffi::TensorView dK_tv, tvm::ffi::TensorView dV_tv) {
    CUDA_CHECK(cudaSetDevice(Q_tv.device().device_id));
    (void)O_tv;

    auto shape = Q_tv.shape();
    int B = static_cast<int>(shape[0]);
    int H = static_cast<int>(shape[1]);
    int S = static_cast<int>(shape[2]);
    int D = static_cast<int>(shape[3]);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q_tv.device().device_type, Q_tv.device().device_id));

    const __nv_bfloat16* Q  = static_cast<const __nv_bfloat16*>(Q_tv.data_ptr());
    const __nv_bfloat16* K  = static_cast<const __nv_bfloat16*>(K_tv.data_ptr());
    const __nv_bfloat16* V  = static_cast<const __nv_bfloat16*>(V_tv.data_ptr());
    const __nv_bfloat16* dO = static_cast<const __nv_bfloat16*>(dO_tv.data_ptr());
    const float* L          = static_cast<const float*>(L_tv.data_ptr());
    __nv_bfloat16* dQ       = static_cast<__nv_bfloat16*>(dQ_tv.data_ptr());
    __nv_bfloat16* dK       = static_cast<__nv_bfloat16*>(dK_tv.data_ptr());
    __nv_bfloat16* dV       = static_cast<__nv_bfloat16*>(dV_tv.data_ptr());

    size_t m_size = static_cast<size_t>(B) * H * S * sizeof(float);
    float* m_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&m_buf, m_size));

    int n_bh = B * H;
    int n_sblocks = (S + BS - 1) / BS;
    dim3 grid(n_sblocks, n_bh);
    dim3 block(NTHREADS);
    size_t smem_bytes = 2ULL * BS * D * sizeof(__nv_bfloat16);

    kern_dQ<<<grid, block, smem_bytes, stream>>>(Q, K, V, dO, L, dQ, m_buf, B, H, S, D);
    CUDA_CHECK(cudaGetLastError());

    kern_dK<<<grid, block, smem_bytes, stream>>>(Q, K, V, dO, L, m_buf, dK, B, H, S, D);
    CUDA_CHECK(cudaGetLastError());

    kern_dV<<<grid, block, smem_bytes, stream>>>(Q, K, dO, L, dV, B, H, S, D);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(m_buf));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_ns::run);

}  // namespace