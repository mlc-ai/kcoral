#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
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

namespace mha_bwd_impl {

constexpr int TILE_M = 16;
constexpr int TILE_N = 16;
constexpr int TPB = 128;

__device__ __forceinline__ float bf162f32(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

__device__ __forceinline__ __nv_bfloat16 f322bf16(float v) {
    return __float2bfloat16(v);
}

__global__ void mha_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q_g,
    const __nv_bfloat16* __restrict__ K_g,
    const __nv_bfloat16* __restrict__ V_g,
    const __nv_bfloat16* __restrict__ O_g,
    const __nv_bfloat16* __restrict__ dO_g,
    const float* __restrict__ LSE_g,
    __nv_bfloat16* __restrict__ dQ_out,
    __nv_bfloat16* __restrict__ dK_out,
    __nv_bfloat16* __restrict__ dV_out,
    int B, int H, int S, int d,
    float scale) {

    int bid = blockIdx.x;
    int b = bid / H;
    int h = bid % H;
    if (b >= B || h >= H) return;

    int tid = threadIdx.x;

    uint64_t bh_off = ((uint64_t)b * H + h) * (uint64_t)S * d;
    const __nv_bfloat16* Q_bh = Q_g + bh_off;
    const __nv_bfloat16* K_bh = K_g + bh_off;
    const __nv_bfloat16* V_bh = V_g + bh_off;
    const __nv_bfloat16* O_bh = O_g + bh_off;
    const __nv_bfloat16* dO_bh = dO_g + bh_off;
    const float* LSE_bh = LSE_g + (b * H + h) * S;

    int n_qtiles = (S + TILE_M - 1) / TILE_M;
    int n_kvtiles = (S + TILE_N - 1) / TILE_N;

    // Allocate output buffers in shared mem for accumulation, then atomic-add to global
    extern __shared__ char smem[];

    __nv_bfloat16* sQ     = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sK     = sQ + TILE_M * d;
    __nv_bfloat16* sV     = sK + TILE_N * d;
    __nv_bfloat16* sdO    = sV + TILE_N * d;
    float* sLSE           = reinterpret_cast<float*>(sdO + TILE_M * d);
    float* sD             = sLSE + TILE_M;
    float* sP             = sD + TILE_M;
    float* sdS            = sP + TILE_M * TILE_N;
    float* sdOV           = sdS + TILE_M * TILE_N;
    float* sAccDQ         = sdOV + TILE_M * TILE_N;
    float* sAccDK         = sAccDQ + TILE_M * d;
    float* sAccDV         = sAccDK + TILE_N * d;

    // Initialize global outputs to zero
    for (int idx = tid; idx < S * d; idx += TPB) {
        dQ_out[bh_off + idx] = f322bf16(0.0f);
        dK_out[bh_off + idx] = f322bf16(0.0f);
        dV_out[bh_off + idx] = f322bf16(0.0f);
    }
    __syncthreads();

    // Unified loop: for each (qtile, kvtil e) pair compute attn scores and all grads
    for (int tq = 0; tq < n_qtiles; tq++) {
        int qs = tq * TILE_M;
        int qe = min(qs + TILE_M, S);
        int qh = qe - qs;

        for (int tk = 0; tk < n_kvtiles; tk++) {
            int ks = tk * TILE_N;
            int ke = min(ks + TILE_N, S);
            int kh = ke - ks;

            // --- Load Q tile ---
            for (int idx = tid; idx < TILE_M * d; idx += TPB) {
                int r = idx / d;
                int c = idx % d;
                sQ[idx] = (r < qh) ? Q_bh[(uint64_t)(qs + r) * d + c] : f322bf16(0.0f);
            }

            // --- Load K tile ---
            for (int idx = tid; idx < TILE_N * d; idx += TPB) {
                int r = idx / d;
                int c = idx % d;
                sK[idx] = (r < kh) ? K_bh[(uint64_t)(ks + r) * d + c] : f322bf16(0.0f);
            }

            // --- Load V tile ---
            for (int idx = tid; idx < TILE_N * d; idx += TPB) {
                int r = idx / d;
                int c = idx % d;
                sV[idx] = (r < kh) ? V_bh[(uint64_t)(ks + r) * d + c] : f322bf16(0.0f);
            }

            // --- Load dO tile ---
            for (int idx = tid; idx < TILE_M * d; idx += TPB) {
                int r = idx / d;
                int c = idx % d;
                sdO[idx] = (r < qh) ? dO_bh[(uint64_t)(qs + r) * d + c] : f322bf16(0.0f);
            }
            __syncthreads();

            // --- Load LSE and compute D[i] = dO[i] . O[i] ---
            for (int i = tid; i < TILE_M; i += TPB) {
                if (i < qh) {
                    sLSE[i] = LSE_bh[qs + i];
                    float D_val = 0.0f;
                    for (int c = 0; c < d; c++) {
                        D_val += bf162f32(sdO[i * d + c]) * bf162f32(O_bh[(uint64_t)(qs + i) * d + c]);
                    }
                    sD[i] = D_val;
                } else {
                    sLSE[i] = 0.0f;
                    sD[i]   = 0.0f;
                }
            }
            __syncthreads();

            // --- Compute P[i][j] = exp(Q[i]*K[j]*scale - LSE[i]) ---
            for (int ij = tid; ij < TILE_M * TILE_N; ij += TPB) {
                int i = ij / TILE_N;
                int j = ij % TILE_N;
                float s = 0.0f;
                for (int c = 0; c < d; c++) {
                    s += bf162f32(sQ[i * d + c]) * bf162f32(sK[j * d + c]);
                }
                sP[ij] = expf(s * scale - sLSE[i]);
            }
            __syncthreads();

            // --- Compute dOV[i][j] = dO[i] * V[j] ---
            for (int ij = tid; ij < TILE_M * TILE_N; ij += TPB) {
                int i = ij / TILE_N;
                int j = ij % TILE_N;
                float dv = 0.0f;
                for (int c = 0; c < d; c++) {
                    dv += bf162f32(sdO[i * d + c]) * bf162f32(sV[j * d + c]);
                }
                sdOV[ij] = dv;
            }
            __syncthreads();

            // --- Compute dS[i][j] = P[i][j] * (dOV[i][j] - D[i]) ---
            for (int ij = tid; ij < TILE_M * TILE_N; ij += TPB) {
                int i = ij / TILE_N;
                sdS[ij] = sP[ij] * (sdOV[ij] - sD[i]);
            }
            __syncthreads();

            // --- Accumulate dQ[i][c] += sum_j dS[i][j] * K[j][c] ---
            for (int ic = tid; ic < TILE_M * d; ic += TPB) {
                int i = ic / d;
                int c = ic % d;
                float val = 0.0f;
                for (int j = 0; j < kh; j++) {
                    val += sdS[i * TILE_N + j] * bf162f32(sK[j * d + c]);
                }
                sAccDQ[ic] = val;
            }
            __syncthreads();

            // --- Write dQ contribution via atomics ---
            for (int ic = tid; ic < qh * d; ic += TPB) {
                int i = ic / d;
                int c = ic % d;
                int gidx = (qs + i) * d + c;
                float fv = sAccDQ[ic];
                __nv_bfloat16 fval = f322bf16(fv);
                uint32_t* addr_u32 = reinterpret_cast<uint32_t*>(&dQ_out[bh_off + gidx]);
                uint32_t cur = atomicCAS(addr_u32, 0, *(uint32_t*)&fval);
                for (int retry = 0; retry < 16; retry++) {
                    float cur_f = bf162f32(*reinterpret_cast<__nv_bfloat16*>(&cur));
                    float new_f = cur_f + fv;
                    __nv_bfloat16 new_v = f322bf16(new_f);
                    uint32_t new_val = *(uint32_t*)&new_v;
                    uint32_t old = atomicCAS(addr_u32, cur, new_val);
                    if (old == cur) break;
                    cur = old;
                }
            }
            __syncthreads();

            // --- Accumulate dK[j][c] += sum_i dS[i][j] * Q[i][c] ---
            for (int jc = tid; jc < TILE_N * d; jc += TPB) {
                int j = jc / d;
                int c = jc % d;
                float val = 0.0f;
                for (int i = 0; i < qh; i++) {
                    val += sdS[i * TILE_N + j] * bf162f32(sQ[i * d + c]);
                }
                sAccDK[jc] = val;
            }
            __syncthreads();

            // --- Write dK contribution via atomics ---
            for (int jc = tid; jc < kh * d; jc += TPB) {
                int j = jc / d;
                int c = jc % d;
                int gidx = (ks + j) * d + c;
                float fv = sAccDK[jc];
                __nv_bfloat16 fval = f322bf16(fv);
                uint32_t* addr_u32 = reinterpret_cast<uint32_t*>(&dK_out[bh_off + gidx]);
                uint32_t cur = atomicCAS(addr_u32, 0, *(uint32_t*)&fval);
                for (int retry = 0; retry < 16; retry++) {
                    float cur_f = bf162f32(*reinterpret_cast<__nv_bfloat16*>(&cur));
                    float new_f = cur_f + fv;
                    __nv_bfloat16 new_v = f322bf16(new_f);
                    uint32_t new_val = *(uint32_t*)&new_v;
                    uint32_t old = atomicCAS(addr_u32, cur, new_val);
                    if (old == cur) break;
                    cur = old;
                }
            }
            __syncthreads();

            // --- Accumulate dV[j][c] += sum_i P[i][j] * dO[i][c] ---
            for (int jc = tid; jc < TILE_N * d; jc += TPB) {
                int j = jc / d;
                int c = jc % d;
                float val = 0.0f;
                for (int i = 0; i < qh; i++) {
                    val += sP[i * TILE_N + j] * bf162f32(sdO[i * d + c]);
                }
                sAccDV[jc] = val;
            }
            __syncthreads();

            // --- Write dV contribution via atomics ---
            for (int jc = tid; jc < kh * d; jc += TPB) {
                int j = jc / d;
                int c = jc % d;
                int gidx = (ks + j) * d + c;
                float fv = sAccDV[jc];
                __nv_bfloat16 fval = f322bf16(fv);
                uint32_t* addr_u32 = reinterpret_cast<uint32_t*>(&dV_out[bh_off + gidx]);
                uint32_t cur = atomicCAS(addr_u32, 0, *(uint32_t*)&fval);
                for (int retry = 0; retry < 16; retry++) {
                    float cur_f = bf162f32(*reinterpret_cast<__nv_bfloat16*>(&cur));
                    float new_f = cur_f + fv;
                    __nv_bfloat16 new_v = f322bf16(new_f);
                    uint32_t new_val = *(uint32_t*)&new_v;
                    uint32_t old = atomicCAS(addr_u32, cur, new_val);
                    if (old == cur) break;
                    cur = old;
                }
            }
            __syncthreads();
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S_val = Q.size(2);
    int64_t d = Q.size(3);

    int num_bh = static_cast<int>(B * H);
    dim3 grid(num_bh);
    dim3 block(TPB);

    // Compute dynamic shared memory size
    size_t smem_bytes = 0;
    smem_bytes += (size_t)TILE_M * d * sizeof(__nv_bfloat16);  // sQ
    smem_bytes += (size_t)TILE_N * d * sizeof(__nv_bfloat16);  // sK
    smem_bytes += (size_t)TILE_N * d * sizeof(__nv_bfloat16);  // sV
    smem_bytes += (size_t)TILE_M * d * sizeof(__nv_bfloat16);  // sdO
    smem_bytes += (size_t)TILE_M * sizeof(float);              // sLSE
    smem_bytes += (size_t)TILE_M * sizeof(float);              // sD
    smem_bytes += (size_t)TILE_M * TILE_N * sizeof(float);     // sP
    smem_bytes += (size_t)TILE_M * TILE_N * sizeof(float);     // sdS
    smem_bytes += (size_t)TILE_M * TILE_N * sizeof(float);     // sdOV
    smem_bytes += (size_t)TILE_M * d * sizeof(float);          // sAccDQ
    smem_bytes += (size_t)TILE_N * d * sizeof(float);          // sAccDK
    smem_bytes += (size_t)TILE_N * d * sizeof(float);          // sAccDV

    float attn_scale = 1.0f / std::sqrt(static_cast<float>(d));

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    int smem_avail = 0, smem_optin = 0;
    cudaDeviceGetAttribute(&smem_avail, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    if (smem_bytes > static_cast<size_t>(smem_avail) && smem_bytes <= static_cast<size_t>(smem_optin)) {
        CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<cudaFunction_t>(&mha_bwd_kernel),
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(smem_bytes)));
    }

    mha_bwd_kernel<<<grid, block, smem_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(O.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        static_cast<int>(B), static_cast<int>(H), static_cast<int>(S_val), static_cast<int>(d),
        attn_scale
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_impl::run);

}  // namespace mha_bwd_impl