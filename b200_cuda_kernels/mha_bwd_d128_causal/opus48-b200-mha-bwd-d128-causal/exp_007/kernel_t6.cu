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

// ---------------- dK / dV kernel ----------------
// grid (numKVblocks, BH). 8 warps. BK=64 keys, loop BQ=64 queries.
__launch_bounds__(NT, 2)
__global__ void bwd_dkdv_kernel(const bf16* Q, const bf16* K, const bf16* V, const bf16* dO,
                                const float* L, const float* Dbuf,
                                bf16* dK, bf16* dV, int S, float scale) {
    int bh = blockIdx.y;
    int kv0 = blockIdx.x * 64;
    if (kv0 >= S) return;
    int tid = threadIdx.x;
    int warp = tid >> 5;

    extern __shared__ char sm[];
    bf16*  K_s   = (bf16*)(sm + 0);
    bf16*  V_s   = (bf16*)(sm + 16384);
    bf16*  Q_s   = (bf16*)(sm + 32768);
    bf16*  dO_s  = (bf16*)(sm + 49152);
    float* Sf    = (float*)(sm + 65536);
    float* dPf   = (float*)(sm + 81920);
    bf16*  Pbf   = (bf16*)(sm + 98304);
    float* L_s   = (float*)(sm + 106496);
    float* D_s   = (float*)(sm + 106752);
    float* OutF  = (float*)(sm + 65536);

    size_t bh_off = (size_t)bh * S * D;
    const bf16* Qb  = Q  + bh_off;
    const bf16* Kb  = K  + bh_off;
    const bf16* Vb  = V  + bh_off;
    const bf16* dOb = dO + bh_off;
    const float* Lb = L    + (size_t)bh * S;
    const float* Db = Dbuf + (size_t)bh * S;
    bf16* dKb = dK + bh_off;
    bf16* dVb = dV + bh_off;

    FragC dV_frag[4], dK_frag[4];
    #pragma unroll
    for (int n = 0; n < 4; n++) { wmma::fill_fragment(dV_frag[n], 0.f); wmma::fill_fragment(dK_frag[n], 0.f); }
    int wkey = (warp & 3) * 16;
    int wdh  = (warp >> 2) * 64;

    int nIter = (S - kv0 + 63) / 64;

    load_async(K_s, Kb, kv0, 64, S);
    load_async(V_s, Vb, kv0, 64, S);
    cp_commit();

    for (int it = 0; it < nIter; it++) {
        int i0 = kv0 + it * 64;
        load_async(Q_s, Qb, i0, 64, S);
        load_async(dO_s, dOb, i0, 64, S);
        cp_commit();
        if (tid < 64) { int ig = i0 + tid; L_s[tid] = ig < S ? Lb[ig] : 0.f; D_s[tid] = ig < S ? Db[ig] : 0.f; }
        cp_wait<0>();
        __syncthreads();

        // phase A: warps 0-3 -> S^T ; warps 4-7 -> dP^T. Each warp: 4 indep N-tiles.
        if (warp < 4) {
            int keyb = warp * 16;
            FragC acc[4];
            #pragma unroll
            for (int n = 0; n < 4; n++) wmma::fill_fragment(acc[n], 0.f);
            #pragma unroll
            for (int kt = 0; kt < 8; kt++) {
                FragA a; wmma::load_matrix_sync(a, K_s + keyb * 128 + kt * 16, 128);
                #pragma unroll
                for (int n = 0; n < 4; n++) {
                    FragBcol b; wmma::load_matrix_sync(b, Q_s + (n * 16) * 128 + kt * 16, 128);
                    wmma::mma_sync(acc[n], a, b, acc[n]);
                }
            }
            #pragma unroll
            for (int n = 0; n < 4; n++) wmma::store_matrix_sync(Sf + keyb * 64 + n * 16, acc[n], 64, wmma::mem_row_major);
        } else {
            int keyb = (warp - 4) * 16;
            FragC acc[4];
            #pragma unroll
            for (int n = 0; n < 4; n++) wmma::fill_fragment(acc[n], 0.f);
            #pragma unroll
            for (int kt = 0; kt < 8; kt++) {
                FragA a; wmma::load_matrix_sync(a, V_s + keyb * 128 + kt * 16, 128);
                #pragma unroll
                for (int n = 0; n < 4; n++) {
                    FragBcol b; wmma::load_matrix_sync(b, dO_s + (n * 16) * 128 + kt * 16, 128);
                    wmma::mma_sync(acc[n], a, b, acc[n]);
                }
            }
            #pragma unroll
            for (int n = 0; n < 4; n++) wmma::store_matrix_sync(dPf + keyb * 64 + n * 16, acc[n], 64, wmma::mem_row_major);
        }
        __syncthreads();

        // P^T
        for (int i = tid; i < 64 * 64; i += NT) {
            int key = i >> 6, q = i & 63;
            int kg = kv0 + key, ig = i0 + q;
            float p = 0.f;
            if (ig < S && kg < S && kg <= ig) p = __expf(Sf[i] * scale - L_s[q]);
            Sf[i] = p;
            Pbf[i] = __float2bfloat16(p);
        }
        __syncthreads();

        // dV += P^T @ dO  (4 indep N-tiles per warp)
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            int dbase = wdh + n * 16;
            #pragma unroll
            for (int kt = 0; kt < 4; kt++) {
                FragA a; wmma::load_matrix_sync(a, Pbf + wkey * 64 + kt * 16, 64);
                FragBrow b; wmma::load_matrix_sync(b, dO_s + kt * 16 * 128 + dbase, 128);
                wmma::mma_sync(dV_frag[n], a, b, dV_frag[n]);
            }
        }
        __syncthreads();

        // dS'^T
        for (int i = tid; i < 64 * 64; i += NT) {
            int q = i & 63;
            Pbf[i] = __float2bfloat16(scale * Sf[i] * (dPf[i] - D_s[q]));
        }
        __syncthreads();

        // dK += dS'^T @ Q
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            int dbase = wdh + n * 16;
            #pragma unroll
            for (int kt = 0; kt < 4; kt++) {
                FragA a; wmma::load_matrix_sync(a, Pbf + wkey * 64 + kt * 16, 64);
                FragBrow b; wmma::load_matrix_sync(b, Q_s + kt * 16 * 128 + dbase, 128);
                wmma::mma_sync(dK_frag[n], a, b, dK_frag[n]);
            }
        }
        __syncthreads();
    }

    // write dV
    #pragma unroll
    for (int n = 0; n < 4; n++)
        wmma::store_matrix_sync(OutF + wkey * 128 + wdh + n * 16, dV_frag[n], 128, wmma::mem_row_major);
    __syncthreads();
    for (int i = tid; i < 64 * 128; i += NT) {
        int key = i >> 7, d = i & 127, kg = kv0 + key;
        if (kg < S) dVb[(size_t)kg * D + d] = __float2bfloat16(OutF[i]);
    }
    __syncthreads();
    #pragma unroll
    for (int n = 0; n < 4; n++)
        wmma::store_matrix_sync(OutF + wkey * 128 + wdh + n * 16, dK_frag[n], 128, wmma::mem_row_major);
    __syncthreads();
    for (int i = tid; i < 64 * 128; i += NT) {
        int key = i >> 7, d = i & 127, kg = kv0 + key;
        if (kg < S) dKb[(size_t)kg * D + d] = __float2bfloat16(OutF[i]);
    }
}

// ---------------- dQ kernel ----------------
// grid (numQblocks, BH). 8 warps. BQ=64 queries, loop BK=64 keys.
__launch_bounds__(NT, 2)
__global__ void bwd_dq_kernel(const bf16* Q, const bf16* K, const bf16* V, const bf16* dO,
                              const float* L, const float* Dbuf,
                              bf16* dQ, int S, float scale) {
    int bh = blockIdx.y;
    int q0 = blockIdx.x * 64;
    if (q0 >= S) return;
    int tid = threadIdx.x;
    int warp = tid >> 5;

    extern __shared__ char sm[];
    bf16*  Q_s   = (bf16*)(sm + 0);
    bf16*  dO_s  = (bf16*)(sm + 16384);
    bf16*  K_s   = (bf16*)(sm + 32768);
    bf16*  V_s   = (bf16*)(sm + 49152);
    float* Sf    = (float*)(sm + 65536);
    float* dPf   = (float*)(sm + 81920);
    bf16*  dSbf  = (bf16*)(sm + 98304);
    float* L_s   = (float*)(sm + 106496);
    float* D_s   = (float*)(sm + 106752);
    float* OutF  = (float*)(sm + 65536);

    size_t bh_off = (size_t)bh * S * D;
    const bf16* Qb  = Q  + bh_off;
    const bf16* Kb  = K  + bh_off;
    const bf16* Vb  = V  + bh_off;
    const bf16* dOb = dO + bh_off;
    const float* Lb = L    + (size_t)bh * S;
    const float* Db = Dbuf + (size_t)bh * S;
    bf16* dQb = dQ + bh_off;

    load_async(Q_s, Qb, q0, 64, S);
    load_async(dO_s, dOb, q0, 64, S);
    cp_commit();
    if (tid < 64) { int qg = q0 + tid; L_s[tid] = qg < S ? Lb[qg] : 0.f; D_s[tid] = qg < S ? Db[qg] : 0.f; }

    FragC dQ_frag[4];
    #pragma unroll
    for (int n = 0; n < 4; n++) wmma::fill_fragment(dQ_frag[n], 0.f);
    int wq  = (warp & 3) * 16;
    int wdh = (warp >> 2) * 64;

    int jmax = q0 + 64; if (jmax > S) jmax = S;
    int nIter = (jmax + 63) / 64;

    for (int it = 0; it < nIter; it++) {
        int j0 = it * 64;
        load_async(K_s, Kb, j0, 64, S);
        load_async(V_s, Vb, j0, 64, S);
        cp_commit();
        cp_wait<0>();
        __syncthreads();

        // phase A: warps0-3 -> S ; warps4-7 -> dP
        if (warp < 4) {
            int qb = warp * 16;
            FragC acc[4];
            #pragma unroll
            for (int n = 0; n < 4; n++) wmma::fill_fragment(acc[n], 0.f);
            #pragma unroll
            for (int kt = 0; kt < 8; kt++) {
                FragA a; wmma::load_matrix_sync(a, Q_s + qb * 128 + kt * 16, 128);
                #pragma unroll
                for (int n = 0; n < 4; n++) {
                    FragBcol b; wmma::load_matrix_sync(b, K_s + (n * 16) * 128 + kt * 16, 128);
                    wmma::mma_sync(acc[n], a, b, acc[n]);
                }
            }
            #pragma unroll
            for (int n = 0; n < 4; n++) wmma::store_matrix_sync(Sf + qb * 64 + n * 16, acc[n], 64, wmma::mem_row_major);
        } else {
            int qb = (warp - 4) * 16;
            FragC acc[4];
            #pragma unroll
            for (int n = 0; n < 4; n++) wmma::fill_fragment(acc[n], 0.f);
            #pragma unroll
            for (int kt = 0; kt < 8; kt++) {
                FragA a; wmma::load_matrix_sync(a, dO_s + qb * 128 + kt * 16, 128);
                #pragma unroll
                for (int n = 0; n < 4; n++) {
                    FragBcol b; wmma::load_matrix_sync(b, V_s + (n * 16) * 128 + kt * 16, 128);
                    wmma::mma_sync(acc[n], a, b, acc[n]);
                }
            }
            #pragma unroll
            for (int n = 0; n < 4; n++) wmma::store_matrix_sync(dPf + qb * 64 + n * 16, acc[n], 64, wmma::mem_row_major);
        }
        __syncthreads();

        // dS'
        for (int i = tid; i < 64 * 64; i += NT) {
            int q = i >> 6, key = i & 63;
            int qg = q0 + q, kg = j0 + key;
            float p = 0.f;
            if (kg < S && qg < S && kg <= qg) p = __expf(Sf[i] * scale - L_s[q]);
            dSbf[i] = __float2bfloat16(scale * p * (dPf[i] - D_s[q]));
        }
        __syncthreads();

        // dQ += dS' @ K
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            int dbase = wdh + n * 16;
            #pragma unroll
            for (int kt = 0; kt < 4; kt++) {
                FragA a; wmma::load_matrix_sync(a, dSbf + wq * 64 + kt * 16, 64);
                FragBrow b; wmma::load_matrix_sync(b, K_s + kt * 16 * 128 + dbase, 128);
                wmma::mma_sync(dQ_frag[n], a, b, dQ_frag[n]);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int n = 0; n < 4; n++)
        wmma::store_matrix_sync(OutF + wq * 128 + wdh + n * 16, dQ_frag[n], 128, wmma::mem_row_major);
    __syncthreads();
    for (int i = tid; i < 64 * 128; i += NT) {
        int q = i >> 7, d = i & 127, qg = q0 + q;
        if (qg < S) dQb[(size_t)qg * D + d] = __float2bfloat16(OutF[i]);
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
    size_t smem = 107008;

    static bool attr_set = false;
    if (!attr_set) {
        CUDA_CHECK(cudaFuncSetAttribute((const void*)bwd_dkdv_kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        CUDA_CHECK(cudaFuncSetAttribute((const void*)bwd_dq_kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        attr_set = true;
    }

    { dim3 grid(numBlocks, BH);
      bwd_dkdv_kernel<<<grid, NT, smem, stream>>>(Qp, Kp, Vp, dOp, Lp, Dbuf, dKp, dVp, S, scale);
      CUDA_CHECK(cudaGetLastError()); }
    { dim3 grid(numBlocks, BH);
      bwd_dq_kernel<<<grid, NT, smem, stream>>>(Qp, Kp, Vp, dOp, Lp, Dbuf, dQp, S, scale);
      CUDA_CHECK(cudaGetLastError()); }

    CUDA_CHECK(cudaFreeAsync(Dbuf, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd