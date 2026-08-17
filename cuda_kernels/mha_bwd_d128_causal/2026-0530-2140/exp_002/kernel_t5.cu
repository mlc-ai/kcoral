#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <mma.h>
#include <cmath>
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

namespace mha_bwd_d128_causal {

using namespace nvcuda;
using bf16 = __nv_bfloat16;

constexpr int Hc = 48;
constexpr int Dc = 128;
constexpr int BN = 64;
constexpr int BM = 64;
constexpr int NWARP = 8;
constexpr int KD = Dc / 16; // 8
constexpr int KM = BM / 16; // 4
constexpr int KN = BN / 16; // 4
constexpr int LDD = Dc + 8; // 136
constexpr int LDM = BM + 8; // 72
constexpr int VEC = 8;      // bf16 per int4

__device__ __forceinline__ void split_bf16(float v, bf16& hi, bf16& lo) {
    hi = __float2bfloat16(v);
    lo = __float2bfloat16(v - __bfloat162float(hi));
}

__device__ __forceinline__ void load_tile(bf16* dst, const bf16* base, int start, int rows, int S, int tid) {
    int nvec = rows * Dc / VEC;
    for (int i = tid; i < nvec; i += 256) {
        int r = i / (Dc / VEC);
        int cv = (i % (Dc / VEC)) * VEC;
        int g = start + r;
        int4 v;
        if (g < S) v = *reinterpret_cast<const int4*>(&base[(long)g * Dc + cv]);
        else v = make_int4(0,0,0,0);
        *reinterpret_cast<int4*>(&dst[r * LDD + cv]) = v;
    }
}

// ---------------- D = rowsum(dO ⊙ O) ----------------
__global__ void compute_D_kernel(const bf16* __restrict__ O,
                                 const bf16* __restrict__ dO,
                                 float* __restrict__ D) {
    long row = (long)blockIdx.x;
    int tid = threadIdx.x;
    const bf16* o = O + row * Dc;
    const bf16* g = dO + row * Dc;
    float v = __bfloat162float(o[tid]) * __bfloat162float(g[tid]);
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    __shared__ float ws[4];
    int warp = tid >> 5, lane = tid & 31;
    if (lane == 0) ws[warp] = v;
    __syncthreads();
    if (tid == 0) D[row] = ws[0] + ws[1] + ws[2] + ws[3];
}

// ---------------- dK / dV kernel ----------------
__launch_bounds__(256, 2)
__global__ void dkdv_kernel(const bf16* __restrict__ Q, const bf16* __restrict__ K,
                            const bf16* __restrict__ V, const bf16* __restrict__ dO,
                            const float* __restrict__ L, const float* __restrict__ D,
                            bf16* __restrict__ dK, bf16* __restrict__ dV,
                            int S, float scale) {
    extern __shared__ char smem[];
    bf16* Ksh  = (bf16*)smem;
    bf16* Vsh  = Ksh + BN * LDD;
    bf16* Qsh  = Vsh + BN * LDD;
    bf16* dOsh = Qsh + BM * LDD;
    bf16* Ph   = dOsh + BM * LDD;   // also reused as dSh
    bf16* Pl   = Ph + BN * LDM;     // also reused as dSl
    float* Ssh = (float*)(Pl + BN * LDM);
    float* Lsh = Ssh + BN * LDM;
    float* Dsh = Lsh + BM;
    float* OScr = Ssh;
    bf16* dSh = Ph, *dSl = Pl;

    int kv_tile = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
    int kv_start = kv_tile * BN;
    int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;

    long bh = (long)(b * Hc + h);
    const bf16* Qb  = Q  + bh * S * Dc;
    const bf16* Kb  = K  + bh * S * Dc;
    const bf16* Vb  = V  + bh * S * Dc;
    const bf16* dOb = dO + bh * S * Dc;
    const float* Lb = L  + bh * S;
    const float* Db = D  + bh * S;
    bf16* dKb = dK + bh * S * Dc;
    bf16* dVb = dV + bh * S * Dc;

    load_tile(Ksh, Kb, kv_start, BN, S, tid);
    load_tile(Vsh, Vb, kv_start, BN, S, tid);

    wmma::fragment<wmma::accumulator,16,16,16,float> dVacc[KN];
    wmma::fragment<wmma::accumulator,16,16,16,float> dKacc[KN];
    #pragma unroll
    for (int mt = 0; mt < KN; mt++) { wmma::fill_fragment(dVacc[mt],0.f); wmma::fill_fragment(dKacc[mt],0.f); }

    wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::col_major> fbc;
    wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::row_major> fbr;
    wmma::fragment<wmma::accumulator,16,16,16,float> sacc;

    int numQ = (S + BM - 1) / BM;
    for (int qt = kv_tile; qt < numQ; qt++) {
        int q_start = qt * BM;
        __syncthreads();
        load_tile(Qsh,  Qb,  q_start, BM, S, tid);
        load_tile(dOsh, dOb, q_start, BM, S, tid);
        if (tid < BM) { int qg = q_start + tid; Lsh[tid] = (qg < S) ? Lb[qg] : 0.f; Dsh[tid] = (qg < S) ? Db[qg] : 0.f; }
        __syncthreads();

        // S^T = K @ Q^T -> Ssh
        #pragma unroll
        for (int q2 = 0; q2 < 2; q2++) {
            int ti = warp + q2 * NWARP; int mt = ti / KM; int nt = ti % KM;
            wmma::fill_fragment(sacc, 0.f);
            #pragma unroll
            for (int kt = 0; kt < KD; kt++) {
                wmma::load_matrix_sync(fa, Ksh + (mt*16)*LDD + kt*16, LDD);
                wmma::load_matrix_sync(fbc, Qsh + (nt*16)*LDD + kt*16, LDD);
                wmma::mma_sync(sacc, fa, fbc, sacc);
            }
            wmma::store_matrix_sync(Ssh + (mt*16)*LDM + nt*16, sacc, LDM, wmma::mem_row_major);
        }
        __syncthreads();

        // P^T = exp(scale*S^T - L), split
        for (int idx = tid; idx < BN * BM; idx += 256) {
            int n = idx / BM, m = idx % BM;
            int kg = kv_start + n, qg = q_start + m;
            float p = (kg < S && qg < S && kg <= qg) ? __expf(scale * Ssh[n*LDM+m] - Lsh[m]) : 0.f;
            split_bf16(p, Ph[n*LDM+m], Pl[n*LDM+m]);
        }
        __syncthreads();

        // dV += P^T @ dO  (split)
        #pragma unroll
        for (int mt = 0; mt < KN; mt++) {
            #pragma unroll
            for (int kt = 0; kt < KM; kt++) {
                wmma::load_matrix_sync(fbr, dOsh + (kt*16)*LDD + warp*16, LDD);
                wmma::load_matrix_sync(fa, Ph + (mt*16)*LDM + kt*16, LDM);
                wmma::mma_sync(dVacc[mt], fa, fbr, dVacc[mt]);
                wmma::load_matrix_sync(fa, Pl + (mt*16)*LDM + kt*16, LDM);
                wmma::mma_sync(dVacc[mt], fa, fbr, dVacc[mt]);
            }
        }

        // dP^T = V @ dO^T -> Ssh
        #pragma unroll
        for (int q2 = 0; q2 < 2; q2++) {
            int ti = warp + q2 * NWARP; int mt = ti / KM; int nt = ti % KM;
            wmma::fill_fragment(sacc, 0.f);
            #pragma unroll
            for (int kt = 0; kt < KD; kt++) {
                wmma::load_matrix_sync(fa, Vsh + (mt*16)*LDD + kt*16, LDD);
                wmma::load_matrix_sync(fbc, dOsh + (nt*16)*LDD + kt*16, LDD);
                wmma::mma_sync(sacc, fa, fbc, sacc);
            }
            wmma::store_matrix_sync(Ssh + (mt*16)*LDM + nt*16, sacc, LDM, wmma::mem_row_major);
        }
        __syncthreads();

        // dS^T = P^T*(dP^T - D), split (overwrites Ph/Pl)
        for (int idx = tid; idx < BN * BM; idx += 256) {
            int n = idx / BM, m = idx % BM;
            float pf = __bfloat162float(Ph[n*LDM+m]) + __bfloat162float(Pl[n*LDM+m]);
            float dval = pf * (Ssh[n*LDM+m] - Dsh[m]);
            split_bf16(dval, dSh[n*LDM+m], dSl[n*LDM+m]);
        }
        __syncthreads();

        // dK += dS^T @ Q (split)
        #pragma unroll
        for (int mt = 0; mt < KN; mt++) {
            #pragma unroll
            for (int kt = 0; kt < KM; kt++) {
                wmma::load_matrix_sync(fbr, Qsh + (kt*16)*LDD + warp*16, LDD);
                wmma::load_matrix_sync(fa, dSh + (mt*16)*LDM + kt*16, LDM);
                wmma::mma_sync(dKacc[mt], fa, fbr, dKacc[mt]);
                wmma::load_matrix_sync(fa, dSl + (mt*16)*LDM + kt*16, LDM);
                wmma::mma_sync(dKacc[mt], fa, fbr, dKacc[mt]);
            }
        }
    }

    __syncthreads();
    #pragma unroll
    for (int mt = 0; mt < KN; mt++) {
        wmma::store_matrix_sync(OScr + warp*256, dVacc[mt], 16, wmma::mem_row_major);
        __syncwarp();
        for (int i = lane; i < 256; i += 32) {
            int r = i / 16, c = i % 16;
            int kg = kv_start + mt*16 + r; int dc2 = warp*16 + c;
            if (kg < S) dVb[(long)kg*Dc + dc2] = __float2bfloat16(OScr[warp*256 + i]);
        }
        __syncwarp();
        wmma::store_matrix_sync(OScr + warp*256, dKacc[mt], 16, wmma::mem_row_major);
        __syncwarp();
        for (int i = lane; i < 256; i += 32) {
            int r = i / 16, c = i % 16;
            int kg = kv_start + mt*16 + r; int dc2 = warp*16 + c;
            if (kg < S) dKb[(long)kg*Dc + dc2] = __float2bfloat16(scale * OScr[warp*256 + i]);
        }
        __syncwarp();
    }
}

// ---------------- dQ kernel ----------------
__launch_bounds__(256, 2)
__global__ void dq_kernel(const bf16* __restrict__ Q, const bf16* __restrict__ K,
                          const bf16* __restrict__ V, const bf16* __restrict__ dO,
                          const float* __restrict__ L, const float* __restrict__ D,
                          bf16* __restrict__ dQ, int S, float scale) {
    extern __shared__ char smem[];
    bf16* Ksh  = (bf16*)smem;
    bf16* Vsh  = Ksh + BN * LDD;
    bf16* Qsh  = Vsh + BN * LDD;
    bf16* dOsh = Qsh + BM * LDD;
    bf16* Ph   = dOsh + BM * LDD;   // reused as dSh
    bf16* Pl   = Ph + BN * LDM;     // reused as dSl
    float* Ssh = (float*)(Pl + BN * LDM);
    float* Lsh = Ssh + BN * LDM;
    float* Dsh = Lsh + BM;
    float* OScr = Ssh;
    bf16* dSh = Ph, *dSl = Pl;

    int q_tile = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
    int q_start = q_tile * BM;
    int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;

    long bh = (long)(b * Hc + h);
    const bf16* Qb  = Q  + bh * S * Dc;
    const bf16* Kb  = K  + bh * S * Dc;
    const bf16* Vb  = V  + bh * S * Dc;
    const bf16* dOb = dO + bh * S * Dc;
    const float* Lb = L  + bh * S;
    const float* Db = D  + bh * S;
    bf16* dQb = dQ + bh * S * Dc;

    load_tile(Qsh,  Qb,  q_start, BM, S, tid);
    load_tile(dOsh, dOb, q_start, BM, S, tid);
    if (tid < BM) { int qg = q_start + tid; Lsh[tid] = (qg < S) ? Lb[qg] : 0.f; Dsh[tid] = (qg < S) ? Db[qg] : 0.f; }

    wmma::fragment<wmma::accumulator,16,16,16,float> dQacc[KM];
    #pragma unroll
    for (int mt = 0; mt < KM; mt++) wmma::fill_fragment(dQacc[mt], 0.f);

    wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> fa;
    wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::col_major> fac;
    wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::col_major> fbc;
    wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::row_major> fbr;
    wmma::fragment<wmma::accumulator,16,16,16,float> sacc;

    __syncthreads();

    for (int kt_tile = 0; kt_tile <= q_tile; kt_tile++) {
        int kv_start = kt_tile * BN;
        load_tile(Ksh, Kb, kv_start, BN, S, tid);
        load_tile(Vsh, Vb, kv_start, BN, S, tid);
        __syncthreads();

        // S^T = K @ Q^T -> Ssh
        #pragma unroll
        for (int q2 = 0; q2 < 2; q2++) {
            int ti = warp + q2 * NWARP; int mt = ti / KM; int nt = ti % KM;
            wmma::fill_fragment(sacc, 0.f);
            #pragma unroll
            for (int kt = 0; kt < KD; kt++) {
                wmma::load_matrix_sync(fa, Ksh + (mt*16)*LDD + kt*16, LDD);
                wmma::load_matrix_sync(fbc, Qsh + (nt*16)*LDD + kt*16, LDD);
                wmma::mma_sync(sacc, fa, fbc, sacc);
            }
            wmma::store_matrix_sync(Ssh + (mt*16)*LDM + nt*16, sacc, LDM, wmma::mem_row_major);
        }
        __syncthreads();

        // P^T -> Ph/Pl (split)
        for (int idx = tid; idx < BN * BM; idx += 256) {
            int n = idx / BM, m = idx % BM;
            int kg = kv_start + n, qg = q_start + m;
            float p = (kg < S && qg < S && kg <= qg) ? __expf(scale * Ssh[n*LDM+m] - Lsh[m]) : 0.f;
            split_bf16(p, Ph[n*LDM+m], Pl[n*LDM+m]);
        }
        __syncthreads();

        // dP^T = V @ dO^T -> Ssh
        #pragma unroll
        for (int q2 = 0; q2 < 2; q2++) {
            int ti = warp + q2 * NWARP; int mt = ti / KM; int nt = ti % KM;
            wmma::fill_fragment(sacc, 0.f);
            #pragma unroll
            for (int kt = 0; kt < KD; kt++) {
                wmma::load_matrix_sync(fa, Vsh + (mt*16)*LDD + kt*16, LDD);
                wmma::load_matrix_sync(fbc, dOsh + (nt*16)*LDD + kt*16, LDD);
                wmma::mma_sync(sacc, fa, fbc, sacc);
            }
            wmma::store_matrix_sync(Ssh + (mt*16)*LDM + nt*16, sacc, LDM, wmma::mem_row_major);
        }
        __syncthreads();

        // dS^T = P^T*(dP^T - D), split (overwrites Ph/Pl)
        for (int idx = tid; idx < BN * BM; idx += 256) {
            int n = idx / BM, m = idx % BM;
            float pf = __bfloat162float(Ph[n*LDM+m]) + __bfloat162float(Pl[n*LDM+m]);
            float dval = pf * (Ssh[n*LDM+m] - Dsh[m]);
            split_bf16(dval, dSh[n*LDM+m], dSl[n*LDM+m]);
        }
        __syncthreads();

        // dQ += dS @ K (split, dS via col_major matrix_a)
        #pragma unroll
        for (int mt = 0; mt < KM; mt++) {
            #pragma unroll
            for (int kt2 = 0; kt2 < KN; kt2++) {
                wmma::load_matrix_sync(fbr, Ksh + (kt2*16)*LDD + warp*16, LDD);
                wmma::load_matrix_sync(fac, dSh + (kt2*16)*LDM + mt*16, LDM);
                wmma::mma_sync(dQacc[mt], fac, fbr, dQacc[mt]);
                wmma::load_matrix_sync(fac, dSl + (kt2*16)*LDM + mt*16, LDM);
                wmma::mma_sync(dQacc[mt], fac, fbr, dQacc[mt]);
            }
        }
        __syncthreads();
    }

    __syncthreads();
    #pragma unroll
    for (int mt = 0; mt < KM; mt++) {
        wmma::store_matrix_sync(OScr + warp*256, dQacc[mt], 16, wmma::mem_row_major);
        __syncwarp();
        for (int i = lane; i < 256; i += 32) {
            int r = i / 16, c = i % 16;
            int qg = q_start + mt*16 + r; int dc2 = warp*16 + c;
            if (qg < S) dQb[(long)qg*Dc + dc2] = __float2bfloat16(scale * OScr[warp*256 + i]);
        }
        __syncwarp();
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

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    float scale = 1.0f / sqrtf((float)Dc);

    long rows = (long)B * H * S;
    float* Dp = nullptr;
    CUDA_CHECK(cudaMallocAsync(&Dp, rows * sizeof(float), stream));

    compute_D_kernel<<<(unsigned)rows, 128, 0, stream>>>(Op, dOp, Dp);

    size_t smem = (size_t)(4 * BN * LDD) * sizeof(bf16)   // K,V,Q,dO (BN==BM)
                + (size_t)(2 * BN * LDM) * sizeof(bf16)   // Ph,Pl
                + (size_t)(BN * LDM + 2 * BM) * sizeof(float); // Ssh,L,D

    CUDA_CHECK(cudaFuncSetAttribute(dkdv_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    CUDA_CHECK(cudaFuncSetAttribute(dq_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));

    int numKV = (S + BN - 1) / BN;
    int numQ  = (S + BM - 1) / BM;

    dim3 g1((unsigned)numKV, (unsigned)H, (unsigned)B);
    dkdv_kernel<<<g1, 256, smem, stream>>>(Qp, Kp, Vp, dOp, Lp, Dp, dKp, dVp, S, scale);

    dim3 g2((unsigned)numQ, (unsigned)H, (unsigned)B);
    dq_kernel<<<g2, 256, smem, stream>>>(Qp, Kp, Vp, dOp, Lp, Dp, dQp, S, scale);

    CUDA_CHECK(cudaFreeAsync(Dp, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128_causal::run);

}  // namespace mha_bwd_d128_causal