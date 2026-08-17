#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

namespace mha_bwd_impl {

static constexpr int DH = 128;
static constexpr int BM = 64;
static constexpr int BN = 64;

__global__ void mha_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    float* __restrict__ dQ_f,
    float* __restrict__ dK_f,
    float* __restrict__ dV_f,
    int S, int H) {

    extern __shared__ char smem_char[];
    __nv_bfloat16* smem_Q   = reinterpret_cast<__nv_bfloat16*>(smem_char);
    __nv_bfloat16* smem_dOq = smem_Q   + BM * DH;
    __nv_bfloat16* smem_K   = smem_dOq + BM * DH;
    __nv_bfloat16* smem_V   = smem_K   + BN * DH;

    int b = blockIdx.y;
    int h = blockIdx.z;
    int bh = b * H + h;
    uint64_t base = (uint64_t)bh * S * DH;

    int t = threadIdx.x;
    int q_base = blockIdx.x * BM;
    int q_global = q_base + t;
    bool active = (q_global < S);

    float inv_sd = rsqrtf((float)DH);

    // Phase 0: compute D_q and load Q, dO rows into smem
    float D_q = 0.0f;
    if (active) {
        uint64_t q_off = base + (uint64_t)q_global * DH;
        const __nv_bfloat16* O_row = &O[q_off];
        const __nv_bfloat16* dO_row = &dO[q_off];
        for (int d = t; d < DH; d += blockDim.x) {
            smem_Q[t * DH + d] = Q[q_off + d];
            smem_dOq[t * DH + d] = dO_row[d];
            D_q += __bfloat162float(O_row[d]) * __bfloat162float(dO_row[d]);
        }
    }
    __syncthreads();

    if (!active) return;

    float dq[DH];
    for (int j = 0; j < DH; j++) dq[j] = 0.0f;

    float lse_q = L[bh * S + q_global];
    int ql = t;

    int q_tile_end = min(q_base + BM, S);

    for (int n_base = 0; n_base < q_tile_end; n_base += BN) {
        int n_count = min(n_base + BN, q_tile_end) - n_base;

        // Cooperatively load K and V tiles
        for (int kl = t; kl < n_count; kl += blockDim.x) {
            uint64_t kg_off = base + (uint64_t)(n_base + kl) * DH;
            __nv_bfloat16* krow = &smem_K[kl * DH];
            __nv_bfloat16* vrow = &smem_V[kl * DH];
            const __nv_bfloat16* k_src = &K[kg_off];
            const __nv_bfloat16* v_src = &V[kg_off];
            for (int d = 0; d < DH; d++) {
                krow[d] = k_src[d];
                vrow[d] = v_src[d];
            }
        }
        __syncthreads();

        // Process each K-row for this thread's Q-row
        for (int kl = 0; kl < n_count; kl++) {
            int kg = n_base + kl;
            if (kg > q_global) break;

            float score = 0.0f;
            const __nv_bfloat16* qptr = &smem_Q[ql * DH];
            const __nv_bfloat16* kptr = &smem_K[kl * DH];
            for (int d = 0; d < DH; d++) {
                score += __bfloat162float(qptr[d]) * __bfloat162float(kptr[d]);
            }
            float P = expf(score * inv_sd - lse_q);

            float dp = 0.0f;
            const __nv_bfloat16* doptr = &smem_dOq[ql * DH];
            const __nv_bfloat16* vptr = &smem_V[kl * DH];
            for (int d = 0; d < DH; d++) {
                dp += __bfloat162float(doptr[d]) * __bfloat162float(vptr[d]);
            }

            float ds = P * (dp - D_q);

            uint64_t kd_idx = base + (uint64_t)kg * DH;
            for (int d = 0; d < DH; d++) {
                float Kv = __bfloat162float(kptr[d]);
                float Qv = __bfloat162float(qptr[d]);
                float DOv = __bfloat162float(doptr[d]);

                dq[d] += ds * Kv;
                atomicAdd(&dK_f[kd_idx + d], ds * Qv);
                atomicAdd(&dV_f[kd_idx + d], P * DOv);
            }
        }
    }

    // Write dQ result
    uint64_t dq_out = base + (uint64_t)q_global * DH;
    for (int d = 0; d < DH; d++) {
        dQ_f[dq_out + d] = dq[d];
    }
}

__global__ void convert_float_to_bf16_kernel(
    const float* src, __nv_bfloat16* dst, int nelem) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < nelem; i += stride) {
        dst[i] = __float2bfloat16(src[i]);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t total = B * H * S * DH;

    const __nv_bfloat16* Q_p  = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_p  = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_p  = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_p  = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_p = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_p          = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_p       = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_p       = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_p       = static_cast<__nv_bfloat16*>(dV.data_ptr());

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    int nelem = (int)total;
    float* dQ_f; float* dK_f; float* dV_f;
    CUDA_CHECK(cudaMallocAsync(&dQ_f, nelem * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&dK_f, nelem * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&dV_f, nelem * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(dK_f, 0, nelem * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(dV_f, 0, nelem * sizeof(float), stream));

    size_t smem_bytes = (2ULL * BM + 2ULL * BN) * DH * sizeof(__nv_bfloat16);
    int num_qtiles = (int)((S + BM - 1) / BM);
    dim3 grid(num_qtiles, (int)B, (int)H);
    dim3 block(BM);

    mha_bwd_kernel<<<grid, block, (unsigned int)smem_bytes, stream>>>(
        Q_p, K_p, V_p, O_p, dO_p, L_p, dQ_f, dK_f, dV_f, (int)S, (int)H);
    CUDA_CHECK(cudaGetLastError());

    int ct = 256;
    int cb = (nelem + ct - 1) / ct;
    convert_float_to_bf16_kernel<<<cb, ct, 0, stream>>>(dQ_f, dQ_p, nelem);
    CUDA_CHECK(cudaGetLastError());
    convert_float_to_bf16_kernel<<<cb, ct, 0, stream>>>(dK_f, dK_p, nelem);
    CUDA_CHECK(cudaGetLastError());
    convert_float_to_bf16_kernel<<<cb, ct, 0, stream>>>(dV_f, dV_p, nelem);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFreeAsync(dQ_f, stream));
    CUDA_CHECK(cudaFreeAsync(dK_f, stream));
    CUDA_CHECK(cudaFreeAsync(dV_f, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_impl::run);

} // namespace mha_bwd_impl