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

// ---------------- delta kernel: D_i = sum_d dO_i[d]*O_i[d] ----------------
__global__ void compute_delta_kernel(const bf16* dO, const bf16* O, float* Dbuf, int R) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= R) return;
    const bf16* dop = dO + (size_t)warp * D;
    const bf16* op  = O  + (size_t)warp * D;
    float s = 0.f;
    #pragma unroll
    for (int k = lane; k < D; k += 32) {
        s += __bfloat162float(dop[k]) * __bfloat162float(op[k]);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);
    if (lane == 0) Dbuf[warp] = s;
}

// load nrows x 128 tile from global (row0..) into shared, zero fill OOB
__device__ __forceinline__ void load_tile(bf16* dst, const bf16* src_bh, int row0, int nrows, int S) {
    int tid = threadIdx.x;
    int nvec = nrows * 16; // 16 int4 per row
    for (int cc = tid; cc < nvec; cc += blockDim.x) {
        int row = cc >> 4;
        int off = (cc & 15) * 8;
        int rg = row0 + row;
        int4 v;
        if (rg < S) v = *reinterpret_cast<const int4*>(src_bh + (size_t)rg * D + off);
        else        v = make_int4(0,0,0,0);
        *reinterpret_cast<int4*>(dst + row * D + off) = v;
    }
}

// ---------------- dK / dV kernel ----------------
// grid: (numKVblocks, B*H), block 128 threads (4 warps). each warp owns 16 keys.
__global__ void bwd_dkdv_kernel(const bf16* Q, const bf16* K, const bf16* V, const bf16* dO,
                                const float* L, const float* Dbuf,
                                bf16* dK, bf16* dV, int S, float scale) {
    int bh = blockIdx.y;
    int kv0 = blockIdx.x * 64;
    if (kv0 >= S) return;

    int tid = threadIdx.x;
    int warp = tid >> 5;

    extern __shared__ char smem[];
    float* Ssh  = (float*)(smem + 0);        // 64*16
    float* Pf   = (float*)(smem + 4096);     // 64*16
    float* dPsh = (float*)(smem + 8192);     // 64*16
    float* L_s  = (float*)(smem + 12288);    // 16
    float* D_s  = (float*)(smem + 12352);    // 16
    bf16*  K_s  = (bf16*)(smem + 12416);     // 64*128
    bf16*  V_s  = (bf16*)(smem + 12416 + 16384);
    bf16*  Q_s  = (bf16*)(smem + 45184);     // 16*128
    bf16*  dO_s = (bf16*)(smem + 49280);     // 16*128
    bf16*  Psh  = (bf16*)(smem + 53376);     // 64*16
    bf16*  dSsh = (bf16*)(smem + 55424);     // 64*16

    size_t bh_off = (size_t)bh * S * D;
    const bf16* Qb  = Q  + bh_off;
    const bf16* Kb  = K  + bh_off;
    const bf16* Vb  = V  + bh_off;
    const bf16* dOb = dO + bh_off;
    const float* Lb = L    + (size_t)bh * S;
    const float* Db = Dbuf + (size_t)bh * S;
    bf16* dKb = dK + bh_off;
    bf16* dVb = dV + bh_off;

    // load K, V for this block once
    load_tile(K_s, Kb, kv0, 64, S);
    load_tile(V_s, Vb, kv0, 64, S);

    FragC dV_frag[8];
    FragC dK_frag[8];
    #pragma unroll
    for (int n = 0; n < 8; n++) { wmma::fill_fragment(dV_frag[n], 0.f); wmma::fill_fragment(dK_frag[n], 0.f); }
    __syncthreads();

    for (int i0 = kv0; i0 < S; i0 += 16) {
        load_tile(Q_s,  Qb,  i0, 16, S);
        load_tile(dO_s, dOb, i0, 16, S);
        if (tid < 16) {
            int ig = i0 + tid;
            L_s[tid] = ig < S ? Lb[ig] : 0.f;
            D_s[tid] = ig < S ? Db[ig] : 0.f;
        }
        __syncthreads();

        // S^T[key,q] = sum_k K[key,k]*Q[q,k]
        {
            FragC s_acc; wmma::fill_fragment(s_acc, 0.f);
            #pragma unroll
            for (int kt = 0; kt < 8; kt++) {
                FragA a; FragBcol b;
                wmma::load_matrix_sync(a, K_s + warp * 2048 + kt * 16, 128);
                wmma::load_matrix_sync(b, Q_s + kt * 16, 128);
                wmma::mma_sync(s_acc, a, b, s_acc);
            }
            wmma::store_matrix_sync(Ssh + warp * 256, s_acc, 16, wmma::mem_row_major);
        }
        __syncthreads();

        // P^T
        for (int idx = tid; idx < 64 * 16; idx += 128) {
            int key = idx >> 4;
            int q   = idx & 15;
            int kg = kv0 + key;
            int ig = i0 + q;
            float p = 0.f;
            if (ig < S && kg < S && kg <= ig) {
                p = __expf(Ssh[idx] * scale - L_s[q]);
            }
            Pf[idx]  = p;
            Psh[idx] = __float2bfloat16(p);
        }
        __syncthreads();

        // dV += P^T @ dO
        {
            FragA a;
            wmma::load_matrix_sync(a, Psh + warp * 256, 16);
            #pragma unroll
            for (int n = 0; n < 8; n++) {
                FragBrow b;
                wmma::load_matrix_sync(b, dO_s + n * 16, 128);
                wmma::mma_sync(dV_frag[n], a, b, dV_frag[n]);
            }
        }
        // dP^T[key,q] = sum_k V[key,k]*dO[q,k]
        {
            FragC dp_acc; wmma::fill_fragment(dp_acc, 0.f);
            #pragma unroll
            for (int kt = 0; kt < 8; kt++) {
                FragA a; FragBcol b;
                wmma::load_matrix_sync(a, V_s + warp * 2048 + kt * 16, 128);
                wmma::load_matrix_sync(b, dO_s + kt * 16, 128);
                wmma::mma_sync(dp_acc, a, b, dp_acc);
            }
            wmma::store_matrix_sync(dPsh + warp * 256, dp_acc, 16, wmma::mem_row_major);
        }
        __syncthreads();

        // dS'[key,q] = scale * P * (dP - D_q)
        for (int idx = tid; idx < 64 * 16; idx += 128) {
            int q = idx & 15;
            float ds = scale * Pf[idx] * (dPsh[idx] - D_s[q]);
            dSsh[idx] = __float2bfloat16(ds);
        }
        __syncthreads();

        // dK += dS'^T @ Q
        {
            FragA a;
            wmma::load_matrix_sync(a, dSsh + warp * 256, 16);
            #pragma unroll
            for (int n = 0; n < 8; n++) {
                FragBrow b;
                wmma::load_matrix_sync(b, Q_s + n * 16, 128);
                wmma::mma_sync(dK_frag[n], a, b, dK_frag[n]);
            }
        }
        __syncthreads();
    }

    // write dV
    #pragma unroll
    for (int n = 0; n < 8; n++) {
        __syncthreads();
        wmma::store_matrix_sync(Ssh + warp * 256, dV_frag[n], 16, wmma::mem_row_major);
        __syncthreads();
        for (int idx = tid; idx < 64 * 16; idx += 128) {
            int key = idx >> 4;
            int c   = idx & 15;
            int kg = kv0 + key;
            if (kg < S) dVb[(size_t)kg * D + n * 16 + c] = __float2bfloat16(Ssh[idx]);
        }
    }
    // write dK
    #pragma unroll
    for (int n = 0; n < 8; n++) {
        __syncthreads();
        wmma::store_matrix_sync(Ssh + warp * 256, dK_frag[n], 16, wmma::mem_row_major);
        __syncthreads();
        for (int idx = tid; idx < 64 * 16; idx += 128) {
            int key = idx >> 4;
            int c   = idx & 15;
            int kg = kv0 + key;
            if (kg < S) dKb[(size_t)kg * D + n * 16 + c] = __float2bfloat16(Ssh[idx]);
        }
    }
}

// ---------------- dQ kernel ----------------
// grid: (numQblocks, B*H), block 128 threads (4 warps). each warp owns 16 queries.
__global__ void bwd_dq_kernel(const bf16* Q, const bf16* K, const bf16* V, const bf16* dO,
                              const float* L, const float* Dbuf,
                              bf16* dQ, int S, float scale) {
    int bh = blockIdx.y;
    int q0 = blockIdx.x * 64;
    if (q0 >= S) return;

    int tid = threadIdx.x;
    int warp = tid >> 5;

    extern __shared__ char smem[];
    float* Ssh  = (float*)(smem + 0);        // 64*16
    float* Pf   = (float*)(smem + 4096);     // 64*16
    float* dPsh = (float*)(smem + 8192);     // 64*16
    float* L_s  = (float*)(smem + 12288);    // 64
    float* D_s  = (float*)(smem + 12544);    // 64
    bf16*  Q_s  = (bf16*)(smem + 12800);     // 64*128
    bf16*  dO_s = (bf16*)(smem + 12800 + 16384);
    bf16*  K_s  = (bf16*)(smem + 45568);     // 16*128
    bf16*  V_s  = (bf16*)(smem + 49664);     // 16*128
    bf16*  dSsh = (bf16*)(smem + 53760);     // 64*16

    size_t bh_off = (size_t)bh * S * D;
    const bf16* Qb  = Q  + bh_off;
    const bf16* Kb  = K  + bh_off;
    const bf16* Vb  = V  + bh_off;
    const bf16* dOb = dO + bh_off;
    const float* Lb = L    + (size_t)bh * S;
    const float* Db = Dbuf + (size_t)bh * S;
    bf16* dQb = dQ + bh_off;

    load_tile(Q_s,  Qb,  q0, 64, S);
    load_tile(dO_s, dOb, q0, 64, S);
    if (tid < 64) {
        int qg = q0 + tid;
        L_s[tid] = qg < S ? Lb[qg] : 0.f;
        D_s[tid] = qg < S ? Db[qg] : 0.f;
    }

    FragC dQ_frag[8];
    #pragma unroll
    for (int n = 0; n < 8; n++) wmma::fill_fragment(dQ_frag[n], 0.f);
    __syncthreads();

    for (int j0 = 0; j0 < S && j0 < q0 + 64; j0 += 16) {
        load_tile(K_s, Kb, j0, 16, S);
        load_tile(V_s, Vb, j0, 16, S);
        __syncthreads();

        // S[q,key] = sum_k Q[q,k]*K[key,k]
        {
            FragC s_acc; wmma::fill_fragment(s_acc, 0.f);
            #pragma unroll
            for (int kt = 0; kt < 8; kt++) {
                FragA a; FragBcol b;
                wmma::load_matrix_sync(a, Q_s + warp * 2048 + kt * 16, 128);
                wmma::load_matrix_sync(b, K_s + kt * 16, 128);
                wmma::mma_sync(s_acc, a, b, s_acc);
            }
            wmma::store_matrix_sync(Ssh + warp * 256, s_acc, 16, wmma::mem_row_major);
        }
        __syncthreads();

        // P[q,key]
        for (int idx = tid; idx < 64 * 16; idx += 128) {
            int q   = idx >> 4;
            int key = idx & 15;
            int qg = q0 + q;
            int kg = j0 + key;
            float p = 0.f;
            if (kg < S && kg <= qg) {
                p = __expf(Ssh[idx] * scale - L_s[q]);
            }
            Pf[idx] = p;
        }
        __syncthreads();

        // dP[q,key] = sum_k dO[q,k]*V[key,k]
        {
            FragC dp_acc; wmma::fill_fragment(dp_acc, 0.f);
            #pragma unroll
            for (int kt = 0; kt < 8; kt++) {
                FragA a; FragBcol b;
                wmma::load_matrix_sync(a, dO_s + warp * 2048 + kt * 16, 128);
                wmma::load_matrix_sync(b, V_s + kt * 16, 128);
                wmma::mma_sync(dp_acc, a, b, dp_acc);
            }
            wmma::store_matrix_sync(dPsh + warp * 256, dp_acc, 16, wmma::mem_row_major);
        }
        __syncthreads();

        // dS'[q,key] = scale * P * (dP - D_q)
        for (int idx = tid; idx < 64 * 16; idx += 128) {
            int q = idx >> 4;
            float ds = scale * Pf[idx] * (dPsh[idx] - D_s[q]);
            dSsh[idx] = __float2bfloat16(ds);
        }
        __syncthreads();

        // dQ += dS' @ K
        {
            FragA a;
            wmma::load_matrix_sync(a, dSsh + warp * 256, 16);
            #pragma unroll
            for (int n = 0; n < 8; n++) {
                FragBrow b;
                wmma::load_matrix_sync(b, K_s + n * 16, 128);
                wmma::mma_sync(dQ_frag[n], a, b, dQ_frag[n]);
            }
        }
        __syncthreads();
    }

    // write dQ
    #pragma unroll
    for (int n = 0; n < 8; n++) {
        __syncthreads();
        wmma::store_matrix_sync(Ssh + warp * 256, dQ_frag[n], 16, wmma::mem_row_major);
        __syncthreads();
        for (int idx = tid; idx < 64 * 16; idx += 128) {
            int q = idx >> 4;
            int c = idx & 15;
            int qg = q0 + q;
            if (qg < S) dQb[(size_t)qg * D + n * 16 + c] = __float2bfloat16(Ssh[idx]);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int S = (int)Q.size(2);
    int d = (int)Q.size(3);
    (void)d;

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

    // delta
    {
        int threads = 128;
        int warpsPerBlock = threads / 32;
        int blocks = (R + warpsPerBlock - 1) / warpsPerBlock;
        compute_delta_kernel<<<blocks, threads, 0, stream>>>(dOp, Op, Dbuf, R);
        CUDA_CHECK(cudaGetLastError());
    }

    int numBlocks = (S + 63) / 64;
    size_t smem_dkdv = 57472;
    size_t smem_dq   = 55808;

    static bool attr_set = false;
    if (!attr_set) {
        CUDA_CHECK(cudaFuncSetAttribute((const void*)bwd_dkdv_kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_dkdv));
        CUDA_CHECK(cudaFuncSetAttribute((const void*)bwd_dq_kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_dq));
        attr_set = true;
    }

    {
        dim3 grid(numBlocks, BH);
        bwd_dkdv_kernel<<<grid, 128, smem_dkdv, stream>>>(Qp, Kp, Vp, dOp, Lp, Dbuf,
                                                          dKp, dVp, S, scale);
        CUDA_CHECK(cudaGetLastError());
    }
    {
        dim3 grid(numBlocks, BH);
        bwd_dq_kernel<<<grid, 128, smem_dq, stream>>>(Qp, Kp, Vp, dOp, Lp, Dbuf,
                                                      dQp, S, scale);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaFreeAsync(Dbuf, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd