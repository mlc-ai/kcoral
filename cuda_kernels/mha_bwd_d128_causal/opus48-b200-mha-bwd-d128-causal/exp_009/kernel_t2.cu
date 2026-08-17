#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <mma.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
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

namespace mha_bwd {

typedef __nv_bfloat16 bf16;
constexpr int HD  = 128;
constexpr int BM  = 128;   // large tile (kv for dkdv, q for dq)
constexpr int BN  = 64;    // small tile (q for dkdv, kv for dq)
constexpr int NT  = 256;   // threads

constexpr int SMEM = 197120;

template<int ROWS>
__device__ __forceinline__ void load_tile(const bf16* gbase, int row0, int S, bf16* dst) {
    int tid = threadIdx.x;
    #pragma unroll
    for (int i = tid; i < ROWS*HD/8; i += NT) {
        int e = i*8; int r = e / HD; int c = e % HD;
        int gr = row0 + r;
        if (gr < S) {
            *reinterpret_cast<int4*>(dst + r*HD + c) =
                *reinterpret_cast<const int4*>(gbase + (long)gr*HD + c);
        } else {
            *reinterpret_cast<int4*>(dst + r*HD + c) = make_int4(0,0,0,0);
        }
    }
}

template<int ROWS>
__device__ __forceinline__ void store_out(const float* buf, bf16* gbase, int row0, int S) {
    int tid = threadIdx.x;
    #pragma unroll
    for (int i = tid; i < ROWS*HD/8; i += NT) {
        int e = i*8; int r = e / HD; int c = e % HD;
        int gr = row0 + r;
        if (gr < S) {
            bf16 tmp[8];
            #pragma unroll
            for (int k=0;k<8;k++) tmp[k] = __float2bfloat16(buf[r*HD + c + k]);
            *reinterpret_cast<int4*>(gbase + (long)gr*HD + c) = *reinterpret_cast<int4*>(tmp);
        }
    }
}

__global__ void compute_D(const bf16* __restrict__ O, const bf16* __restrict__ dO,
                          float* __restrict__ Dout, long total_rows) {
    long row = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= total_rows) return;
    const bf16* o = O + row*HD;
    const bf16* g = dO + row*HD;
    float acc = 0.f;
    #pragma unroll
    for (int c=0;c<HD;c++) acc += __bfloat162float(o[c]) * __bfloat162float(g[c]);
    Dout[row] = acc;
}

__device__ __forceinline__ void split_bf16(float v, bf16& hi, bf16& lo){
    hi = __float2bfloat16(v);
    lo = __float2bfloat16(v - __bfloat162float(hi));
}

// ---------------- dK / dV kernel ----------------  (no hi/lo needed)
__launch_bounds__(NT)
__global__ void bwd_dkdv_kernel(
    const bf16* __restrict__ Qg, const bf16* __restrict__ Kg, const bf16* __restrict__ Vg,
    const bf16* __restrict__ dOg, const float* __restrict__ Lg, const float* __restrict__ Dg,
    bf16* __restrict__ dKg, bf16* __restrict__ dVg, int S, float scale)
{
    const int bh = blockIdx.y;
    const int kvb = blockIdx.x;
    const int kv_base = kvb * BM;
    if (kv_base >= S) return;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;  // 0..7

    extern __shared__ char smem[];
    bf16* Ks  = reinterpret_cast<bf16*>(smem);   // [128][128]
    bf16* Vs  = Ks + BM*HD;                       // [128][128]
    bf16* Qs  = Vs + BM*HD;                       // [64][128]
    bf16* dOs = Qs + BN*HD;                       // [64][128]
    bf16* Ps  = dOs + BN*HD;                       // [128][64]
    bf16* dSs = Ps + BM*BN;                        // [128][64]
    float* sS  = reinterpret_cast<float*>(dSs + BM*BN); // [128][64]
    float* sdP = sS + BM*BN;                            // [128][64]
    float* sL  = sdP + BM*BN;                            // [64]
    float* sD  = sL + BN;                                // [64]
    float* outbuf = sS;                                  // [128][128]

    const bf16* Kbase  = Kg  + (long)bh*S*HD;
    const bf16* Vbase  = Vg  + (long)bh*S*HD;
    const bf16* Qbase  = Qg  + (long)bh*S*HD;
    const bf16* dObase = dOg + (long)bh*S*HD;
    const float* Lbase = Lg + (long)bh*S;
    const float* Dbase = Dg + (long)bh*S;

    load_tile<BM>(Kbase, kv_base, S, Ks);
    load_tile<BM>(Vbase, kv_base, S, Vs);

    wmma::fragment<wmma::accumulator,16,16,16,float> dV_frag[8];
    wmma::fragment<wmma::accumulator,16,16,16,float> dK_frag[8];
    #pragma unroll
    for(int n=0;n<8;n++){ wmma::fill_fragment(dV_frag[n],0.f); wmma::fill_fragment(dK_frag[n],0.f);}
    __syncthreads();

    int num_q = (S + BN - 1)/BN;
    for(int qb = kvb*2; qb < num_q; ++qb){
        int q_base = qb*BN;
        load_tile<BN>(Qbase, q_base, S, Qs);
        load_tile<BN>(dObase, q_base, S, dOs);
        for(int i=tid;i<BN;i+=NT){ int gq=q_base+i; sL[i]=(gq<S)?Lbase[gq]:0.f; sD[i]=(gq<S)?Dbase[gq]:0.f; }
        __syncthreads();

        wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::col_major> b_col;
        // S^T = K @ Q^T  [128 kv][64 q]
        #pragma unroll
        for(int n=0;n<4;n++){
            wmma::fragment<wmma::accumulator,16,16,16,float> acc; wmma::fill_fragment(acc,0.f);
            #pragma unroll
            for(int kt=0;kt<8;kt++){
                wmma::load_matrix_sync(a_frag, Ks + (16*warp)*HD + 16*kt, HD);
                wmma::load_matrix_sync(b_col, Qs + (16*n)*HD + 16*kt, HD);
                wmma::mma_sync(acc,a_frag,b_col,acc);
            }
            wmma::store_matrix_sync(sS + (16*warp)*BN + 16*n, acc, BN, wmma::mem_row_major);
        }
        // dP^T = V @ dO^T
        #pragma unroll
        for(int n=0;n<4;n++){
            wmma::fragment<wmma::accumulator,16,16,16,float> acc; wmma::fill_fragment(acc,0.f);
            #pragma unroll
            for(int kt=0;kt<8;kt++){
                wmma::load_matrix_sync(a_frag, Vs + (16*warp)*HD + 16*kt, HD);
                wmma::load_matrix_sync(b_col, dOs + (16*n)*HD + 16*kt, HD);
                wmma::mma_sync(acc,a_frag,b_col,acc);
            }
            wmma::store_matrix_sync(sdP + (16*warp)*BN + 16*n, acc, BN, wmma::mem_row_major);
        }
        __syncthreads();

        for(int idx=tid; idx<BM*BN; idx+=NT){
            int k = idx / BN;
            int q = idx % BN;
            int kg = kv_base + k;
            int qg = q_base + q;
            bool valid = (qg < S) && (qg >= kg);
            float s = scale * sS[idx];
            float p = valid ? __expf(s - sL[q]) : 0.f;
            float ds = p * (sdP[idx] - sD[q]);
            Ps[idx]  = __float2bfloat16(p);
            dSs[idx] = __float2bfloat16(ds);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> aP;
        wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::row_major> bRow;
        // dV += P^T @ dO
        #pragma unroll
        for(int n=0;n<8;n++){
            #pragma unroll
            for(int kt=0;kt<4;kt++){
                wmma::load_matrix_sync(bRow, dOs + (16*kt)*HD + 16*n, HD);
                wmma::load_matrix_sync(aP, Ps + (16*warp)*BN + 16*kt, BN);
                wmma::mma_sync(dV_frag[n], aP, bRow, dV_frag[n]);
            }
        }
        // dK += dS^T @ Q
        #pragma unroll
        for(int n=0;n<8;n++){
            #pragma unroll
            for(int kt=0;kt<4;kt++){
                wmma::load_matrix_sync(bRow, Qs + (16*kt)*HD + 16*n, HD);
                wmma::load_matrix_sync(aP, dSs + (16*warp)*BN + 16*kt, BN);
                wmma::mma_sync(dK_frag[n], aP, bRow, dK_frag[n]);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for(int n=0;n<8;n++)
        wmma::store_matrix_sync(outbuf + (16*warp)*HD + 16*n, dV_frag[n], HD, wmma::mem_row_major);
    __syncthreads();
    store_out<BM>(outbuf, dVg + (long)bh*S*HD, kv_base, S);
    __syncthreads();
    #pragma unroll
    for(int n=0;n<8;n++){
        #pragma unroll
        for(int i=0;i<dK_frag[n].num_elements;i++) dK_frag[n].x[i]*=scale;
        wmma::store_matrix_sync(outbuf + (16*warp)*HD + 16*n, dK_frag[n], HD, wmma::mem_row_major);
    }
    __syncthreads();
    store_out<BM>(outbuf, dKg + (long)bh*S*HD, kv_base, S);
}

// ---------------- dQ kernel ---------------- (hi/lo for dS)
__launch_bounds__(NT)
__global__ void bwd_dq_kernel(
    const bf16* __restrict__ Qg, const bf16* __restrict__ Kg, const bf16* __restrict__ Vg,
    const bf16* __restrict__ dOg, const float* __restrict__ Lg, const float* __restrict__ Dg,
    bf16* __restrict__ dQg, int S, float scale)
{
    const int bh = blockIdx.y;
    const int qb = blockIdx.x;
    const int q_base = qb*BM;
    if (q_base >= S) return;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;

    extern __shared__ char smem[];
    bf16* Qs  = reinterpret_cast<bf16*>(smem);   // [128][128]
    bf16* dOs = Qs + BM*HD;                        // [128][128]
    bf16* Ks  = dOs + BM*HD;                       // [64][128]
    bf16* Vs  = Ks + BN*HD;                        // [64][128]
    bf16* dShi= Vs + BN*HD;                        // [128][64]
    bf16* dSlo= dShi + BM*BN;                      // [128][64]
    float* sS  = reinterpret_cast<float*>(dSlo + BM*BN); // [128][64]
    float* sdP = sS + BM*BN;                             // [128][64]
    float* sL  = sdP + BM*BN;                             // [128]
    float* sD  = sL + BM;                                 // [128]
    float* outbuf = sS;

    const bf16* Qbase  = Qg  + (long)bh*S*HD;
    const bf16* Kbase  = Kg  + (long)bh*S*HD;
    const bf16* Vbase  = Vg  + (long)bh*S*HD;
    const bf16* dObase = dOg + (long)bh*S*HD;
    const float* Lbase = Lg + (long)bh*S;
    const float* Dbase = Dg + (long)bh*S;

    load_tile<BM>(Qbase, q_base, S, Qs);
    load_tile<BM>(dObase, q_base, S, dOs);
    for(int i=tid;i<BM;i+=NT){ int gq=q_base+i; sL[i]=(gq<S)?Lbase[gq]:0.f; sD[i]=(gq<S)?Dbase[gq]:0.f; }

    wmma::fragment<wmma::accumulator,16,16,16,float> dQ_frag[8];
    #pragma unroll
    for(int n=0;n<8;n++) wmma::fill_fragment(dQ_frag[n],0.f);
    __syncthreads();

    int num_kv = (S + BN - 1)/BN;
    int kv_hi = qb*2+1; if (kv_hi > num_kv-1) kv_hi = num_kv-1;
    for(int kvb=0; kvb<=kv_hi; ++kvb){
        int kv_base = kvb*BN;
        load_tile<BN>(Kbase, kv_base, S, Ks);
        load_tile<BN>(Vbase, kv_base, S, Vs);
        __syncthreads();

        wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::col_major> b_col;
        // S = Q @ K^T  [128 q][64 kv]
        #pragma unroll
        for(int n=0;n<4;n++){
            wmma::fragment<wmma::accumulator,16,16,16,float> acc; wmma::fill_fragment(acc,0.f);
            #pragma unroll
            for(int kt=0;kt<8;kt++){
                wmma::load_matrix_sync(a_frag, Qs + (16*warp)*HD + 16*kt, HD);
                wmma::load_matrix_sync(b_col, Ks + (16*n)*HD + 16*kt, HD);
                wmma::mma_sync(acc,a_frag,b_col,acc);
            }
            wmma::store_matrix_sync(sS + (16*warp)*BN + 16*n, acc, BN, wmma::mem_row_major);
        }
        // dP = dO @ V^T
        #pragma unroll
        for(int n=0;n<4;n++){
            wmma::fragment<wmma::accumulator,16,16,16,float> acc; wmma::fill_fragment(acc,0.f);
            #pragma unroll
            for(int kt=0;kt<8;kt++){
                wmma::load_matrix_sync(a_frag, dOs + (16*warp)*HD + 16*kt, HD);
                wmma::load_matrix_sync(b_col, Vs + (16*n)*HD + 16*kt, HD);
                wmma::mma_sync(acc,a_frag,b_col,acc);
            }
            wmma::store_matrix_sync(sdP + (16*warp)*BN + 16*n, acc, BN, wmma::mem_row_major);
        }
        __syncthreads();

        for(int idx=tid; idx<BM*BN; idx+=NT){
            int q = idx / BN;
            int k = idx % BN;
            int qg = q_base + q;
            int kg = kv_base + k;
            bool valid = (qg < S) && (kg <= qg) && (kg < S);
            float s = scale * sS[idx];
            float p = valid ? __expf(s - sL[q]) : 0.f;
            float ds = p * (sdP[idx] - sD[q]);
            bf16 hi,lo; split_bf16(ds, hi, lo);
            dShi[idx] = hi; dSlo[idx] = lo;
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> aS;
        wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::row_major> bK;
        // dQ += dS @ K   (hi + lo)
        #pragma unroll
        for(int n=0;n<8;n++){
            #pragma unroll
            for(int kt=0;kt<4;kt++){
                wmma::load_matrix_sync(bK, Ks + (16*kt)*HD + 16*n, HD);
                wmma::load_matrix_sync(aS, dShi + (16*warp)*BN + 16*kt, BN);
                wmma::mma_sync(dQ_frag[n], aS, bK, dQ_frag[n]);
                wmma::load_matrix_sync(aS, dSlo + (16*warp)*BN + 16*kt, BN);
                wmma::mma_sync(dQ_frag[n], aS, bK, dQ_frag[n]);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for(int n=0;n<8;n++){
        #pragma unroll
        for(int i=0;i<dQ_frag[n].num_elements;i++) dQ_frag[n].x[i]*=scale;
        wmma::store_matrix_sync(outbuf + (16*warp)*HD + 16*n, dQ_frag[n], HD, wmma::mem_row_major);
    }
    __syncthreads();
    store_out<BM>(outbuf, dQg + (long)bh*S*HD, q_base, S);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int S = (int)Q.size(2);
    int BH = B*H;
    float scale = 1.0f / sqrtf((float)HD);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    const bf16* Qp = static_cast<const bf16*>(Q.data_ptr());
    const bf16* Kp = static_cast<const bf16*>(K.data_ptr());
    const bf16* Vp = static_cast<const bf16*>(V.data_ptr());
    const bf16* Op = static_cast<const bf16*>(O.data_ptr());
    const bf16* dOp = static_cast<const bf16*>(dO.data_ptr());
    const float* Lp = static_cast<const float*>(L.data_ptr());
    bf16* dQp = static_cast<bf16*>(dQ.data_ptr());
    bf16* dKp = static_cast<bf16*>(dK.data_ptr());
    bf16* dVp = static_cast<bf16*>(dV.data_ptr());

    long total = (long)BH * S;
    float* Dscratch = nullptr;
    CUDA_CHECK(cudaMallocAsync(&Dscratch, (size_t)total*sizeof(float), stream));

    compute_D<<<(unsigned)((total+255)/256), 256, 0, stream>>>(Op, dOp, Dscratch, total);
    CUDA_CHECK(cudaGetLastError());

    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(bwd_dkdv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        cudaFuncSetAttribute(bwd_dq_kernel,   cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        attr_set = true;
    }

    int nblk = (S + BM - 1) / BM;
    dim3 g1(nblk, BH);
    bwd_dkdv_kernel<<<g1, NT, SMEM, stream>>>(Qp, Kp, Vp, dOp, Lp, Dscratch, dKp, dVp, S, scale);
    CUDA_CHECK(cudaGetLastError());

    dim3 g2(nblk, BH);
    bwd_dq_kernel<<<g2, NT, SMEM, stream>>>(Qp, Kp, Vp, dOp, Lp, Dscratch, dQp, S, scale);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFreeAsync(Dscratch, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd