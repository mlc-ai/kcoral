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
        fprintf(stderr, "CUDA error %s at %s:%d\n",                \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_kernel {

constexpr int BM = 64, BN = 64, HD = 128, THREADS = 128;
constexpr int NB = BN / 16, NK = HD / 16, ND = HD / 16;

__device__ __forceinline__ void cp_async16(void* s, const void* g) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n" :: "r"(a), "l"(g) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template<int N> __device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void load_tile(__nv_bfloat16* dst, const __nv_bfloat16* base,
                                          int kv_start, int S, int tid) {
    #pragma unroll
    for (int idx = tid; idx < BN * HD / 8; idx += THREADS) {
        int r = idx / (HD / 8);
        int c = (idx % (HD / 8)) * 8;
        int gr = kv_start + r;
        if (gr < S) cp_async16(&dst[r * HD + c], &base[(size_t)gr * HD + c]);
    }
}

__global__ __launch_bounds__(THREADS)
void attn_kernel(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
                 const __nv_bfloat16* __restrict__ V, __nv_bfloat16* __restrict__ O,
                 float* __restrict__ LSE, int S, float scale) {
    int bh = blockIdx.y, q_start = blockIdx.x * BM;
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int groupID = lane >> 2, tig = lane & 3, warpRow = warp * 16;

    extern __shared__ char smem[];
    __nv_bfloat16* Qs = (__nv_bfloat16*)smem;      // BM*HD
    __nv_bfloat16* Ks = Qs + BM * HD;              // [2][BN*HD]
    __nv_bfloat16* Vs = Ks + 2 * BN * HD;          // [2][BN*HD]
    __nv_bfloat16* Ps = Vs + 2 * BN * HD;          // BM*BN
    float* l_sh = (float*)(Ps + BM * BN);          // BM
    float* Osh = (float*)Ks;                       // reuse (32KB)

    const __nv_bfloat16* Qb = Q + (size_t)bh * S * HD;
    const __nv_bfloat16* Kb = K + (size_t)bh * S * HD;
    const __nv_bfloat16* Vb = V + (size_t)bh * S * HD;

    for (int idx = tid; idx < BM * HD; idx += THREADS) {
        int r = idx / HD, c = idx % HD, gr = q_start + r;
        Qs[idx] = (gr < S) ? Qb[(size_t)gr * HD + c] : (__nv_bfloat16)0;
    }

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> Of[ND];
    #pragma unroll
    for (int i = 0; i < ND; i++) wmma::fill_fragment(Of[i], 0.f);
    float mA = -1e30f, mB = -1e30f, lA = 0, lB = 0;

    int num_kv = (S + BN - 1) / BN;
    load_tile(Ks + 0 * BN * HD, Kb, 0, S, tid);
    load_tile(Vs + 0 * BN * HD, Vb, 0, S, tid);
    cp_commit();
    __syncthreads();

    for (int kv = 0; kv < num_kv; kv++) {
        int cur = kv & 1;
        __syncthreads(); // (A) prev iter done with buffers
        bool pf = (kv + 1 < num_kv);
        if (pf) {
            int nb = (kv + 1) & 1;
            load_tile(Ks + nb * BN * HD, Kb, (kv + 1) * BN, S, tid);
            load_tile(Vs + nb * BN * HD, Vb, (kv + 1) * BN, S, tid);
            cp_commit();
        }
        if (pf) cp_wait<1>(); else cp_wait<0>();
        __syncthreads(); // (B) cur data ready

        __nv_bfloat16* Kc = Ks + cur * BN * HD;
        __nv_bfloat16* Vc = Vs + cur * BN * HD;
        int kv_start = kv * BN;

        // S = Q @ K^T
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> Sf[NB];
        #pragma unroll
        for (int n = 0; n < NB; n++) wmma::fill_fragment(Sf[n], 0.f);
        #pragma unroll
        for (int k = 0; k < NK; k++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
            wmma::load_matrix_sync(af, &Qs[warpRow * HD + k * 16], HD);
            #pragma unroll
            for (int n = 0; n < NB; n++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf;
                wmma::load_matrix_sync(bf, &Kc[(n * 16) * HD + k * 16], HD);
                wmma::mma_sync(Sf[n], af, bf, Sf[n]);
            }
        }

        // online softmax
        float locMaxA = -1e30f, locMaxB = -1e30f;
        #pragma unroll
        for (int n = 0; n < NB; n++)
            #pragma unroll
            for (int t = 0; t < 8; t++) {
                int col = n * 16 + (t >> 2) * 8 + tig * 2 + (t & 1);
                float v = Sf[n].x[t] * scale;
                if (kv_start + col >= S) v = -1e30f;
                Sf[n].x[t] = v;
                if (((t >> 1) & 1) == 0) locMaxA = fmaxf(locMaxA, v); else locMaxB = fmaxf(locMaxB, v);
            }
        locMaxA = fmaxf(locMaxA, __shfl_xor_sync(0xffffffffu, locMaxA, 1));
        locMaxA = fmaxf(locMaxA, __shfl_xor_sync(0xffffffffu, locMaxA, 2));
        locMaxB = fmaxf(locMaxB, __shfl_xor_sync(0xffffffffu, locMaxB, 1));
        locMaxB = fmaxf(locMaxB, __shfl_xor_sync(0xffffffffu, locMaxB, 2));
        float mAn = fmaxf(mA, locMaxA), mBn = fmaxf(mB, locMaxB);
        float cA = __expf(mA - mAn), cB = __expf(mB - mBn);
        float sA = 0, sB = 0;
        #pragma unroll
        for (int n = 0; n < NB; n++)
            #pragma unroll
            for (int t = 0; t < 8; t++) {
                int rs = (t >> 1) & 1;
                float p = __expf(Sf[n].x[t] - (rs ? mBn : mAn));
                Sf[n].x[t] = p;
                if (rs == 0) sA += p; else sB += p;
            }
        sA += __shfl_xor_sync(0xffffffffu, sA, 1); sA += __shfl_xor_sync(0xffffffffu, sA, 2);
        sB += __shfl_xor_sync(0xffffffffu, sB, 1); sB += __shfl_xor_sync(0xffffffffu, sB, 2);
        lA = lA * cA + sA; lB = lB * cB + sB; mA = mAn; mB = mBn;

        #pragma unroll
        for (int n = 0; n < NB; n++)
            #pragma unroll
            for (int t = 0; t < 8; t++) {
                int rs = (t >> 1) & 1;
                int row = warpRow + (rs ? groupID + 8 : groupID);
                int col = n * 16 + (t >> 2) * 8 + tig * 2 + (t & 1);
                Ps[row * BN + col] = __float2bfloat16(Sf[n].x[t]);
            }
        #pragma unroll
        for (int nd = 0; nd < ND; nd++)
            #pragma unroll
            for (int t = 0; t < 8; t++) Of[nd].x[t] *= (((t >> 1) & 1) ? cB : cA);
        __syncthreads(); // (C) P ready

        // O += P @ V
        #pragma unroll
        for (int k = 0; k < NB; k++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
            wmma::load_matrix_sync(af, &Ps[warpRow * BN + k * 16], BN);
            #pragma unroll
            for (int nd = 0; nd < ND; nd++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bf;
                wmma::load_matrix_sync(bf, &Vc[(k * 16) * HD + nd * 16], HD);
                wmma::mma_sync(Of[nd], af, bf, Of[nd]);
            }
        }
    }

    if (tig == 0) {
        int rA = warpRow + groupID, rB = warpRow + groupID + 8;
        l_sh[rA] = lA; l_sh[rB] = lB;
        int gA = q_start + rA, gB = q_start + rB;
        if (gA < S) LSE[(size_t)bh * S + gA] = mA + logf(lA);
        if (gB < S) LSE[(size_t)bh * S + gB] = mB + logf(lB);
    }
    __syncthreads();
    #pragma unroll
    for (int nd = 0; nd < ND; nd++)
        wmma::store_matrix_sync(&Osh[warpRow * HD + nd * 16], Of[nd], HD, wmma::mem_row_major);
    __syncthreads();
    __nv_bfloat16* Ob = O + (size_t)bh * S * HD;
    for (int idx = tid; idx < BM * HD; idx += THREADS) {
        int r = idx / HD, c = idx % HD, gr = q_start + r;
        if (gr < S) { float inv = 1.f / l_sh[r]; Ob[(size_t)gr * HD + c] = __float2bfloat16(Osh[idx] * inv); }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int B = Q.size(0), H = Q.size(1), S = Q.size(2);
    float scale = 1.0f / sqrtf((float)HD);

    const __nv_bfloat16* Qp = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* Kp = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* Vp = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* Op = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* Lp = static_cast<float*>(LSE.data_ptr());

    int num_q = (S + BM - 1) / BM;
    dim3 grid(num_q, B * H);
    dim3 block(THREADS);
    size_t smem = (size_t)BM * HD * 2 + (size_t)2 * BN * HD * 2 + (size_t)2 * BN * HD * 2
                + (size_t)BM * BN * 2 + (size_t)BM * 4;

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(attn_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    attn_kernel<<<grid, block, smem, stream>>>(Qp, Kp, Vp, Op, Lp, S, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel::run);

}  // namespace mha_kernel