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
    bf16*  PdS   = (bf16*)(sm + 98304);
    float* L_s   = (float*)(sm + 106496);
    float* D_s   = (float*)(sm + 106752);
    float* OutF  = (float*)(sm + 0);

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
    int kr = warp >> 1;
    int dc0 = (warp & 1) * 4;
    int keybaseW = kr * 16;

    int nIter = (S - kv0 + 63) / 64;

    load_async(K_s, Kb, kv0, 64, S);
    load_async(V_s, Vb, kv0, 64, S);
    cp_commit();

    for (int idx = 0; idx < nIter; idx++) {
        int i0 = kv0 + idx * 64;
        load_async(Q_s, Qb, i0, 64, S);
        load_async(dO_s, dOb, i0, 64, S);
        cp_commit();
        if (tid < 64) { int ig = i0 + tid; L_s[tid] = ig < S ? Lb[ig] : 0.f; D_s[tid] = ig < S ? Db[ig] : 0.f; }
        cp_wait<0>();
        __syncthreads();

        // S^T and dP^T
        #pragma unroll
        for (int ti = 0; ti < 2; ti++) {
            int t = warp * 2 + ti;
            int keyb = (t >> 2) * 16, qb = (t & 3) * 16;
            FragC sa; wmma::fill_fragment(sa, 0.f);
            FragC da; wmma::fill_fragment(da, 0.f);
            #pragma unroll
            for (int kt = 0; kt < 8; kt++) {
                FragA a1; FragBcol b1;
                wmma::load_matrix_sync(a1, K_s + keyb * 128 + kt * 16, 128);
                wmma::load_matrix_sync(b1, Q_s + qb * 128 + kt * 16, 128);
                wmma::mma_sync(sa, a1, b1, sa);
                FragA a2; FragBcol b2;
                wmma::load_matrix_sync(a2, V_s + keyb * 128 + kt * 16, 128);
                wmma::load_matrix_sync(b2, dO_s + qb * 128 + kt * 16, 128);
                wmma::mma_sync(da, a2, b2, da);
            }
            wmma::store_matrix_sync(Sf + keyb * 64 + qb, sa, 64, wmma::mem_row_major);
            wmma::store_matrix_sync(dPf + keyb * 64 + qb, da, 64, wmma::mem_row_major);
        }
        __syncthreads();

        // P^T = exp(scale*S^T - L); keep float P in Sf, bf16 P in PdS
        for (int i = tid; i < 64 * 64; i += NT) {
            int key = i >> 6, q = i & 63;
            int kg = kv0 + key, ig = i0 + q;
            float p = 0.f;
            if (ig < S && kg < S && kg <= ig) p = __expf(Sf[i] * scale - L_s[q]);
            Sf[i] = p;
            PdS[i] = __float2bfloat16(p);
        }
        __syncthreads();

        // dV += P^T @ dO
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int dbase = (dc0 + j) * 16;
            #pragma unroll
            for (int kt = 0; kt < 4; kt++) {
                FragA a; FragBrow b;
                wmma::load_matrix_sync(a, PdS + keybaseW * 64 + kt * 16, 64);
                wmma::load_matrix_sync(b, dO_s + kt * 16 * 128 + dbase, 128);
                wmma::mma_sync(dV_frag[j], a, b, dV_frag[j]);
            }
        }
        __syncthreads();  // dV MMA done reading PdS

        // dS'^T = scale * P * (dP^T - D_q) -> PdS
        for (int i = tid; i < 64 * 64; i += NT) {
            int q = i & 63;
            float ds = scale * Sf[i] * (dPf[i] - D_s[q]);
            PdS[i] = __float2bfloat16(ds);
        }
        __syncthreads();

        // dK += dS'^T @ Q
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int dbase = (dc0 + j) * 16;
            #pragma unroll
            for (int kt = 0; kt < 4; kt++) {
                FragA a; FragBrow b;
                wmma::load_matrix_sync(a, PdS + keybaseW * 64 + kt * 16, 64);
                wmma::load_matrix_sync(b, Q_s + kt * 16 * 128 + dbase, 128);
                wmma::mma_sync(dK_frag[j], a, b, dK_frag[j]);
            }
        }
        __syncthreads();  // before next iter overwrites Q_s/dO_s
    }

    // write dV
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        int dbase = (dc0 + j) * 16;
        wmma::store_matrix_sync(OutF + keybaseW * 128 + dbase, dV_frag[j], 128, wmma::mem_row_major);
    }
    __syncthreads();
    for (int i = tid; i < 64 * 128; i += NT) {
        int key = i >> 7, d = i & 127, kg = kv0 + key;
        if (kg < S) dVb[(size_t)kg * D + d] = __float2bfloat16(OutF[i]);
    }
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        int dbase = (dc0 + j) * 16;
        wmma::store_matrix_sync(OutF + keybaseW * 128 + dbase, dK_frag[j], 128, wmma::mem_row_major);
    }
    __syncthreads();
    for (int i = tid; i < 64 * 128; i += NT) {
        int key = i >> 7, d = i & 127, kg = kv0 + key;
        if (kg < S) dKb[(size_t)kg * D + d] = __float2bfloat16(OutF[i]);
    }
}

// ---------------- dQ kernel ----------------
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
    float* OutF  = (float*)(sm + 0);

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
    int qr = warp >> 1;
    int dc0 = (warp & 1) * 4;
    int qbaseW = qr * 16;

    int jmax = q0 + 64; if (jmax > S) jmax = S;
    int nIter = (jmax + 63) / 64;

    for (int idx = 0; idx < nIter; idx++) {
        int j0 = idx * 64;
        load_async(K_s, Kb, j0, 64, S);
        load_async(V_s, Vb, j0, 64, S);
        cp_commit();
        cp_wait<0>();
        __syncthreads();

        // S and dP
        #pragma unroll
        for (int ti = 0; ti < 2; ti++) {
            int t = warp * 2 + ti;
            int qb = (t >> 2) * 16, keyb = (t & 3) * 16;
            FragC sa; wmma::fill_fragment(sa, 0.f);
            FragC da; wmma::fill_fragment(da, 0.f);
            #pragma unroll
            for (int kt = 0; kt < 8; kt++) {
                FragA a1; FragBcol b1;
                wmma::load_matrix_sync(a1, Q_s + qb * 128 + kt * 16, 128);
                wmma::load_matrix_sync(b1, K_s + keyb * 128 + kt * 16, 128);
                wmma::mma_sync(sa, a1, b1, sa);
                FragA a2; FragBcol b2;
                wmma::load_matrix_sync(a2, dO_s + qb * 128 + kt * 16, 128);
                wmma::load_matrix_sync(b2, V_s + keyb * 128 + kt * 16, 128);
                wmma::mma_sync(da, a2, b2, da);
            }
            wmma::store_matrix_sync(Sf + qb * 64 + keyb, sa, 64, wmma::mem_row_major);
            wmma::store_matrix_sync(dPf + qb * 64 + keyb, da, 64, wmma::mem_row_major);
        }
        __syncthreads();

        // dS' = scale * P * (dP - D_q)
        for (int i = tid; i < 64 * 64; i += NT) {
            int q = i >> 6, key = i & 63;
            int qg = q0 + q, kg = j0 + key;
            float p = 0.f;
            if (kg < S && qg < S && kg <= qg) p = __expf(Sf[i] * scale - L_s[q]);
            float ds = scale * p * (dPf[i] - D_s[q]);
            dSbf[i] = __float2bfloat16(ds);
        }
        __syncthreads();

        // dQ += dS' @ K
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int dbase = (dc0 + j) * 16;
            #pragma unroll
            for (int kt = 0; kt < 4; kt++) {
                FragA a; FragBrow b;
                wmma::load_matrix_sync(a, dSbf + qbaseW * 64 + kt * 16, 64);
                wmma::load_matrix_sync(b, K_s + kt * 16 * 128 + dbase, 128);
                wmma::mma_sync(dQ_frag[j], a, b, dQ_frag[j]);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        int dbase = (dc0 + j) * 16;
        wmma::store_matrix_sync(OutF + qbaseW * 128 + dbase, dQ_frag[j], 128, wmma::mem_row_major);
    }
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
    size_t smem_dkdv = 107008;
    size_t smem_dq   = 107008;

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