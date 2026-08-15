#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>
#include <math.h>
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

using namespace nvcuda;
typedef __nv_bfloat16 bf16;

using FragA    = wmma::fragment<wmma::matrix_a, 16,16,16, bf16, wmma::row_major>;
using FragBcol = wmma::fragment<wmma::matrix_b, 16,16,16, bf16, wmma::col_major>;
using FragBrow = wmma::fragment<wmma::matrix_b, 16,16,16, bf16, wmma::row_major>;
using FragC    = wmma::fragment<wmma::accumulator, 16,16,16, float>;

static constexpr int D = 128;
static constexpr int NT = 256;

__device__ __forceinline__ float fexp2(float x) {
    float y; asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y;
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool ok) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    unsigned bytes = ok ? 16u : 0u;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(s), "l"(gmem), "r"(bytes) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template<int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory"); }

__device__ __forceinline__ void load_async(bf16* dst, const bf16* src_bh, int row0, int nrows, int S) {
    int tid = threadIdx.x;
    int nvec = nrows * 16;
    for (int cc = tid; cc < nvec; cc += NT) {
        int row = cc >> 4;
        int off = (cc & 15) * 8;
        int rg = row0 + row;
        bool ok = rg < S;
        const bf16* src = ok ? (src_bh + (size_t)rg * D + off) : src_bh;
        cp_async16(dst + row * D + off, src, ok);
    }
}

__global__ void compute_delta_kernel(const bf16* dO, const bf16* O, float* Dbuf, int R) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= R) return;
    const bf16* dop = dO + (size_t)warp * D;
    const bf16* op  = O  + (size_t)warp * D;
    float s = 0.f;
    #pragma unroll
    for (int k = lane; k < D; k += 32)
        s += __bfloat162float(dop[k]) * __bfloat162float(op[k]);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);
    if (lane == 0) Dbuf[warp] = s;
}

// scale_l2e = scale * log2(e), so exp(scale*S - L) = exp2(scale_l2e*S - L*log2e)
// ---------------- dK / dV kernel ----------------
// grid (numKVblocks, BH). 256 threads (8 warps). BK=64 keys, iterate BQ=32 queries. 3 blk/SM.
__launch_bounds__(NT, 3)
__global__ void bwd_dkdv_kernel(const bf16* Q, const bf16* K, const bf16* V, const bf16* dO,
                                const float* L, const float* Dbuf,
                                bf16* dK, bf16* dV, int S, float scale) {
    int bh = blockIdx.y;
    int kv0 = blockIdx.x * 64;
    if (kv0 >= S) return;
    int tid = threadIdx.x;
    int warp = tid >> 5;
    float l2e = 1.4426950408889634f;

    extern __shared__ char sm[];
    bf16*  K_s   = (bf16*)(sm + 0);
    bf16*  V_s   = (bf16*)(sm + 16384);
    bf16*  Qb2[2]= { (bf16*)(sm + 32768), (bf16*)(sm + 40960) };
    bf16*  Ob2[2]= { (bf16*)(sm + 49152), (bf16*)(sm + 57344) };
    float* Sf    = (float*)(sm + 65536);
    float* dPf   = (float*)(sm + 73728);
    bf16*  Pbf   = (bf16*)(sm + 81920);
    bf16*  dSbf  = (bf16*)(sm + 85504);
    float* L_s   = (float*)(sm + 89600);
    float* D_s   = (float*)(sm + 89728);
    float* OutF  = (float*)(sm + 0);

    size_t bh_off = (size_t)bh * S * D;
    const bf16* Qbp = Q  + bh_off;
    const bf16* Kbp = K  + bh_off;
    const bf16* Vbp = V  + bh_off;
    const bf16* Obp = dO + bh_off;
    const float* Lb = L    + (size_t)bh * S;
    const float* Db = Dbuf + (size_t)bh * S;
    bf16* dKb = dK + bh_off;
    bf16* dVb = dV + bh_off;

    FragC dV_frag[4], dK_frag[4];
    #pragma unroll
    for (int n = 0; n < 4; n++) { wmma::fill_fragment(dV_frag[n], 0.f); wmma::fill_fragment(dK_frag[n], 0.f); }
    int keybaseW = (warp >> 1) * 16;
    int dc0 = (warp & 1) * 4;

    int nIter = (S - kv0 + 31) / 32;

    load_async(K_s, Kbp, kv0, 64, S);
    load_async(V_s, Vbp, kv0, 64, S);
    load_async(Qb2[0], Qbp, kv0, 32, S);
    load_async(Ob2[0], Obp, kv0, 32, S);
    cp_commit();

    int buf = 0;
    for (int it = 0; it < nIter; it++) {
        int i0 = kv0 + it * 32;
        if (it + 1 < nIter) {
            int in = i0 + 32;
            load_async(Qb2[buf ^ 1], Qbp, in, 32, S);
            load_async(Ob2[buf ^ 1], Obp, in, 32, S);
            cp_commit();
        }
        if (tid < 32) { int ig = i0 + tid; L_s[tid] = ig < S ? Lb[ig] : 0.f; D_s[tid] = ig < S ? Db[ig] : 0.f; }
        cp_wait<1>();
        if (it + 1 >= nIter) cp_wait<0>();
        __syncthreads();

        bf16* Q_s  = Qb2[buf];
        bf16* dO_s = Ob2[buf];

        // 8 warps: warp -> one (keyTile16 x qTile16) of S^T (4 warps) or dP^T (4 warps)
        {
            int keyb = (warp >> 1) * 16;
            int qb   = (warp & 1) * 16;
            const bf16* Aptr = (warp < 4) ? (K_s + keyb * 128) : (V_s + keyb * 128);
            const bf16* Bptr = (warp < 4) ? (Q_s + qb * 128)   : (dO_s + qb * 128);
            FragC acc; wmma::fill_fragment(acc, 0.f);
            #pragma unroll
            for (int kt = 0; kt < 8; kt++) {
                FragA a; FragBcol b;
                wmma::load_matrix_sync(a, Aptr + kt * 16, 128);
                wmma::load_matrix_sync(b, Bptr + kt * 16, 128);
                wmma::mma_sync(acc, a, b, acc);
            }
            float* dst = (warp < 4) ? (Sf + keyb * 32 + qb) : (dPf + keyb * 32 + qb);
            wmma::store_matrix_sync(dst, acc, 32, wmma::mem_row_major);
        }
        __syncthreads();

        // P^T and dS'^T fused
        for (int i = tid; i < 64 * 32; i += NT) {
            int key = i >> 5, q = i & 31;
            int kg = kv0 + key, ig = i0 + q;
            float p = 0.f;
            if (ig < S && kg < S && kg <= ig) p = fexp2(Sf[i] * (scale * l2e) - L_s[q] * l2e);
            Pbf[i]  = __float2bfloat16(p);
            dSbf[i] = __float2bfloat16(scale * p * (dPf[i] - D_s[q]));
        }
        __syncthreads();

        // dV += P^T @ dO ; dK += dS'^T @ Q  (contract 32 queries)
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int dbase = (dc0 + j) * 16;
            #pragma unroll
            for (int kt = 0; kt < 2; kt++) {
                FragA pa; wmma::load_matrix_sync(pa, Pbf + keybaseW * 32 + kt * 16, 32);
                FragBrow ob; wmma::load_matrix_sync(ob, dO_s + kt * 16 * 128 + dbase, 128);
                wmma::mma_sync(dV_frag[j], pa, ob, dV_frag[j]);
                FragA sa; wmma::load_matrix_sync(sa, dSbf + keybaseW * 32 + kt * 16, 32);
                FragBrow qb; wmma::load_matrix_sync(qb, Q_s + kt * 16 * 128 + dbase, 128);
                wmma::mma_sync(dK_frag[j], sa, qb, dK_frag[j]);
            }
        }
        __syncthreads();
        buf ^= 1;
    }

    // write dV
    #pragma unroll
    for (int j = 0; j < 4; j++)
        wmma::store_matrix_sync(OutF + keybaseW * 128 + (dc0 + j) * 16, dV_frag[j], 128, wmma::mem_row_major);
    __syncthreads();
    for (int i = tid; i < 64 * 128; i += NT) {
        int key = i >> 7, dd = i & 127, kg = kv0 + key;
        if (kg < S) dVb[(size_t)kg * D + dd] = __float2bfloat16(OutF[i]);
    }
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < 4; j++)
        wmma::store_matrix_sync(OutF + keybaseW * 128 + (dc0 + j) * 16, dK_frag[j], 128, wmma::mem_row_major);
    __syncthreads();
    for (int i = tid; i < 64 * 128; i += NT) {
        int key = i >> 7, dd = i & 127, kg = kv0 + key;
        if (kg < S) dKb[(size_t)kg * D + dd] = __float2bfloat16(OutF[i]);
    }
}

// ---------------- dQ kernel ----------------
// grid (numQblocks, BH). 256 threads. BQ=64 queries, iterate BK=32 keys. 3 blk/SM.
__launch_bounds__(NT, 3)
__global__ void bwd_dq_kernel(const bf16* Q, const bf16* K, const bf16* V, const bf16* dO,
                              const float* L, const float* Dbuf,
                              bf16* dQ, int S, float scale) {
    int bh = blockIdx.y;
    int q0 = blockIdx.x * 64;
    if (q0 >= S) return;
    int tid = threadIdx.x;
    int warp = tid >> 5;
    float l2e = 1.4426950408889634f;

    extern __shared__ char sm[];
    bf16*  Q_s   = (bf16*)(sm + 0);
    bf16*  dO_s  = (bf16*)(sm + 16384);
    bf16*  Kb2[2]= { (bf16*)(sm + 32768), (bf16*)(sm + 40960) };
    bf16*  Vb2[2]= { (bf16*)(sm + 49152), (bf16*)(sm + 57344) };
    float* Sf    = (float*)(sm + 65536);
    float* dPf   = (float*)(sm + 73728);
    bf16*  dSbf  = (bf16*)(sm + 81920);
    float* L_s   = (float*)(sm + 86016);
    float* D_s   = (float*)(sm + 86272);
    float* OutF  = (float*)(sm + 0);

    size_t bh_off = (size_t)bh * S * D;
    const bf16* Qbp = Q  + bh_off;
    const bf16* Kbp = K  + bh_off;
    const bf16* Vbp = V  + bh_off;
    const bf16* Obp = dO + bh_off;
    const float* Lb = L    + (size_t)bh * S;
    const float* Db = Dbuf + (size_t)bh * S;
    bf16* dQb = dQ + bh_off;

    load_async(Q_s, Qbp, q0, 64, S);
    load_async(dO_s, Obp, q0, 64, S);
    if (tid < 64) { int qg = q0 + tid; L_s[tid] = qg < S ? Lb[qg] : 0.f; D_s[tid] = qg < S ? Db[qg] : 0.f; }

    FragC dQ_frag[4];
    #pragma unroll
    for (int n = 0; n < 4; n++) wmma::fill_fragment(dQ_frag[n], 0.f);
    int qbaseW = (warp >> 1) * 16;
    int dc0 = (warp & 1) * 4;

    int jmax = q0 + 64; if (jmax > S) jmax = S;
    int nIter = (jmax + 31) / 32;

    load_async(Kb2[0], Kbp, 0, 32, S);
    load_async(Vb2[0], Vbp, 0, 32, S);
    cp_commit();

    int buf = 0;
    for (int it = 0; it < nIter; it++) {
        int j0 = it * 32;
        if (it + 1 < nIter) {
            int jn = j0 + 32;
            load_async(Kb2[buf ^ 1], Kbp, jn, 32, S);
            load_async(Vb2[buf ^ 1], Vbp, jn, 32, S);
            cp_commit();
        }
        cp_wait<1>();
        if (it + 1 >= nIter) cp_wait<0>();
        __syncthreads();

        bf16* K_s = Kb2[buf];
        bf16* V_s = Vb2[buf];

        // 8 warps -> S (warps 0-3) & dP (warps 4-7), each one (q16 x key16) tile.
        {
            int qb   = (warp >> 1) * 16;
            int keyb = (warp & 1) * 16;
            const bf16* Aptr = (warp < 4) ? (Q_s + qb * 128) : (dO_s + qb * 128);
            const bf16* Bptr = (warp < 4) ? (K_s + keyb * 128) : (V_s + keyb * 128);
            FragC acc; wmma::fill_fragment(acc, 0.f);
            #pragma unroll
            for (int kt = 0; kt < 8; kt++) {
                FragA a; FragBcol b;
                wmma::load_matrix_sync(a, Aptr + kt * 16, 128);
                wmma::load_matrix_sync(b, Bptr + kt * 16, 128);
                wmma::mma_sync(acc, a, b, acc);
            }
            float* dst = (warp < 4) ? (Sf + qb * 32 + keyb) : (dPf + qb * 32 + keyb);
            wmma::store_matrix_sync(dst, acc, 32, wmma::mem_row_major);
        }
        __syncthreads();

        // dS'
        for (int i = tid; i < 64 * 32; i += NT) {
            int q = i >> 5, key = i & 31;
            int qg = q0 + q, kg = j0 + key;
            float p = 0.f;
            if (kg < S && qg < S && kg <= qg) p = fexp2(Sf[i] * (scale * l2e) - L_s[q] * l2e);
            dSbf[i] = __float2bfloat16(scale * p * (dPf[i] - D_s[q]));
        }
        __syncthreads();

        // dQ += dS' @ K
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int dbase = (dc0 + j) * 16;
            #pragma unroll
            for (int kt = 0; kt < 2; kt++) {
                FragA a; wmma::load_matrix_sync(a, dSbf + qbaseW * 32 + kt * 16, 32);
                FragBrow b; wmma::load_matrix_sync(b, K_s + kt * 16 * 128 + dbase, 128);
                wmma::mma_sync(dQ_frag[j], a, b, dQ_frag[j]);
            }
        }
        __syncthreads();
        buf ^= 1;
    }

    #pragma unroll
    for (int j = 0; j < 4; j++)
        wmma::store_matrix_sync(OutF + qbaseW * 128 + (dc0 + j) * 16, dQ_frag[j], 128, wmma::mem_row_major);
    __syncthreads();
    for (int i = tid; i < 64 * 128; i += NT) {
        int q = i >> 7, dd = i & 127, qg = q0 + q;
        if (qg < S) dQb[(size_t)qg * D + dd] = __float2bfloat16(OutF[i]);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int S = (int)Q.size(2);

    const bf16* Qp  = static_cast<const bf16*>(Q.data_ptr());
    const bf16* Kp  = static_cast<const bf16*>(K.data_ptr());
    const bf16* Vp  = static_cast<const bf16*>(V.data_ptr());
    const bf16* Op  = static_cast<const bf16*>(O.data_ptr());
    const bf16* dOp = static_cast<const bf16*>(dO.data_ptr());
    const float* Lp = static_cast<const float*>(L.data_ptr());
    bf16* dQp = static_cast<bf16*>(dQ.data_ptr());
    bf16* dKp = static_cast<bf16*>(dK.data_ptr());
    bf16* dVp = static_cast<bf16*>(dV.data_ptr());

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    int BH = B * H;
    int R = BH * S;
    float scale = 1.0f / sqrtf((float)D);

    float* Dbuf = nullptr;
    CUDA_CHECK(cudaMallocAsync((void**)&Dbuf, sizeof(float) * (size_t)R, stream));

    {
        int threads = 128, wpb = 4;
        int blocks = (R + wpb - 1) / wpb;
        compute_delta_kernel<<<blocks, threads, 0, stream>>>(dOp, Op, Dbuf, R);
        CUDA_CHECK(cudaGetLastError());
    }

    int numBlocks = (S + 63) / 64;
    size_t smem_dkdv = 90112;
    size_t smem_dq   = 86528;

    static bool attr_set = false;
    if (!attr_set) {
        CUDA_CHECK(cudaFuncSetAttribute((const void*)bwd_dkdv_kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_dkdv));
        CUDA_CHECK(cudaFuncSetAttribute((const void*)bwd_dq_kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_dq));
        attr_set = true;
    }

    { dim3 grid(numBlocks, BH);
      bwd_dkdv_kernel<<<grid, NT, smem_dkdv, stream>>>(Qp, Kp, Vp, dOp, Lp, Dbuf, dKp, dVp, S, scale);
      CUDA_CHECK(cudaGetLastError()); }
    { dim3 grid(numBlocks, BH);
      bwd_dq_kernel<<<grid, NT, smem_dq, stream>>>(Qp, Kp, Vp, dOp, Lp, Dbuf, dQp, S, scale);
      CUDA_CHECK(cudaGetLastError()); }

    CUDA_CHECK(cudaFreeAsync(Dbuf, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd