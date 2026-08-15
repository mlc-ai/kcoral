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

constexpr int BLK = 64;
constexpr int HD  = 128;

// dkdv smem
constexpr int DKDV_SMEM = 4*BLK*HD*2 + 4*BLK*BLK*2 + 2*BLK*BLK*4 + 2*BLK*4; // 131584
// dq smem
constexpr int DQ_SMEM   = 4*BLK*HD*2 + 2*BLK*BLK*2 + 2*BLK*BLK*4 + 2*BLK*4; // 115200

__device__ __forceinline__ void load_tile(const __nv_bfloat16* gbase, int row0, int S,
                                           __nv_bfloat16* dst) {
    int tid = threadIdx.x;
    #pragma unroll
    for (int i = tid; i < BLK*HD/8; i += 128) {
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

__device__ __forceinline__ void store_out(const float* buf, __nv_bfloat16* gbase, int row0, int S) {
    int tid = threadIdx.x;
    #pragma unroll
    for (int i = tid; i < BLK*HD/8; i += 128) {
        int e = i*8; int r = e / HD; int c = e % HD;
        int gr = row0 + r;
        if (gr < S) {
            __nv_bfloat16 tmp[8];
            #pragma unroll
            for (int k=0;k<8;k++) tmp[k] = __float2bfloat16(buf[r*HD + c + k]);
            *reinterpret_cast<int4*>(gbase + (long)gr*HD + c) = *reinterpret_cast<int4*>(tmp);
        }
    }
}

__global__ void compute_D(const __nv_bfloat16* __restrict__ O,
                          const __nv_bfloat16* __restrict__ dO,
                          float* __restrict__ Dout, long total_rows) {
    long row = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= total_rows) return;
    const __nv_bfloat16* o = O + row*HD;
    const __nv_bfloat16* g = dO + row*HD;
    float acc = 0.f;
    #pragma unroll
    for (int c=0;c<HD;c++) acc += __bfloat162float(o[c]) * __bfloat162float(g[c]);
    Dout[row] = acc;
}

__device__ __forceinline__ void split_bf16(float v, __nv_bfloat16& hi, __nv_bfloat16& lo){
    hi = __float2bfloat16(v);
    float hf = __bfloat162float(hi);
    lo = __float2bfloat16(v - hf);
}

// ---------------- dK / dV kernel ----------------
__launch_bounds__(128)
__global__ void bwd_dkdv_kernel(
    const __nv_bfloat16* __restrict__ Qg,
    const __nv_bfloat16* __restrict__ Kg,
    const __nv_bfloat16* __restrict__ Vg,
    const __nv_bfloat16* __restrict__ dOg,
    const float* __restrict__ Lg,
    const float* __restrict__ Dg,
    __nv_bfloat16* __restrict__ dKg,
    __nv_bfloat16* __restrict__ dVg,
    int S, float scale)
{
    const int bh = blockIdx.y;
    const int kvb = blockIdx.x;
    const int kv_base = kvb * BLK;
    if (kv_base >= S) return;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;

    extern __shared__ char smem[];
    __nv_bfloat16* Ks  = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* Vs  = Ks + BLK*HD;
    __nv_bfloat16* Qs  = Vs + BLK*HD;
    __nv_bfloat16* dOs = Qs + BLK*HD;
    __nv_bfloat16* Phi = dOs + BLK*HD;
    __nv_bfloat16* Plo = Phi + BLK*BLK;
    __nv_bfloat16* dShi= Plo + BLK*BLK;
    __nv_bfloat16* dSlo= dShi+ BLK*BLK;
    float* sSt  = reinterpret_cast<float*>(dSlo + BLK*BLK);
    float* sdPt = sSt + BLK*BLK;
    float* sL   = sdPt + BLK*BLK;
    float* sD   = sL + BLK;
    float* outbuf = sSt;

    const __nv_bfloat16* Kbase  = Kg  + (long)bh*S*HD;
    const __nv_bfloat16* Vbase  = Vg  + (long)bh*S*HD;
    const __nv_bfloat16* Qbase  = Qg  + (long)bh*S*HD;
    const __nv_bfloat16* dObase = dOg + (long)bh*S*HD;
    const float* Lbase = Lg + (long)bh*S;
    const float* Dbase = Dg + (long)bh*S;

    load_tile(Kbase, kv_base, S, Ks);
    load_tile(Vbase, kv_base, S, Vs);

    wmma::fragment<wmma::accumulator,16,16,16,float> dV_frag[8];
    wmma::fragment<wmma::accumulator,16,16,16,float> dK_frag[8];
    #pragma unroll
    for(int n=0;n<8;n++){ wmma::fill_fragment(dV_frag[n],0.f); wmma::fill_fragment(dK_frag[n],0.f);}
    __syncthreads();

    int num_q_blocks = (S + BLK - 1)/BLK;
    for(int qb = kvb; qb < num_q_blocks; ++qb){
        int q_base = qb*BLK;
        load_tile(Qbase, q_base, S, Qs);
        load_tile(dObase, q_base, S, dOs);
        for(int i=tid;i<BLK;i+=128){ int gq=q_base+i; sL[i]=(gq<S)?Lbase[gq]:0.f; sD[i]=(gq<S)?Dbase[gq]:0.f; }
        __syncthreads();

        wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> b_col;
        // S^T = K @ Q^T
        #pragma unroll
        for(int n=0;n<4;n++){
            wmma::fragment<wmma::accumulator,16,16,16,float> acc; wmma::fill_fragment(acc,0.f);
            #pragma unroll
            for(int kt=0;kt<8;kt++){
                wmma::load_matrix_sync(a_frag, Ks + (16*warp)*HD + 16*kt, HD);
                wmma::load_matrix_sync(b_col, Qs + (16*n)*HD + 16*kt, HD);
                wmma::mma_sync(acc,a_frag,b_col,acc);
            }
            wmma::store_matrix_sync(sSt + (16*warp)*BLK + 16*n, acc, BLK, wmma::mem_row_major);
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
            wmma::store_matrix_sync(sdPt + (16*warp)*BLK + 16*n, acc, BLK, wmma::mem_row_major);
        }
        __syncthreads();

        for(int idx=tid; idx<BLK*BLK; idx+=128){
            int k = idx / BLK;
            int q = idx % BLK;
            int kg = kv_base + k;
            int qg = q_base + q;
            bool valid = (qg < S) && (qg >= kg);
            float s = scale * sSt[idx];
            float p = valid ? __expf(s - sL[q]) : 0.f;
            float dp = sdPt[idx];
            float ds = p * (dp - sD[q]);
            __nv_bfloat16 hi,lo;
            split_bf16(p, hi, lo);   Phi[idx]=hi; Plo[idx]=lo;
            split_bf16(ds, hi, lo);  dShi[idx]=hi; dSlo[idx]=lo;
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> aP;
        wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> bRow;
        // dV += P^T @ dO  (hi + lo)
        #pragma unroll
        for(int n=0;n<8;n++){
            #pragma unroll
            for(int kt=0;kt<4;kt++){
                wmma::load_matrix_sync(bRow, dOs + (16*kt)*HD + 16*n, HD);
                wmma::load_matrix_sync(aP, Phi + (16*warp)*BLK + 16*kt, BLK);
                wmma::mma_sync(dV_frag[n], aP, bRow, dV_frag[n]);
                wmma::load_matrix_sync(aP, Plo + (16*warp)*BLK + 16*kt, BLK);
                wmma::mma_sync(dV_frag[n], aP, bRow, dV_frag[n]);
            }
        }
        // dK += dS^T @ Q  (hi + lo)
        #pragma unroll
        for(int n=0;n<8;n++){
            #pragma unroll
            for(int kt=0;kt<4;kt++){
                wmma::load_matrix_sync(bRow, Qs + (16*kt)*HD + 16*n, HD);
                wmma::load_matrix_sync(aP, dShi + (16*warp)*BLK + 16*kt, BLK);
                wmma::mma_sync(dK_frag[n], aP, bRow, dK_frag[n]);
                wmma::load_matrix_sync(aP, dSlo + (16*warp)*BLK + 16*kt, BLK);
                wmma::mma_sync(dK_frag[n], aP, bRow, dK_frag[n]);
            }
        }
        __syncthreads();
    }

    // write dV
    #pragma unroll
    for(int n=0;n<8;n++)
        wmma::store_matrix_sync(outbuf + (16*warp)*HD + 16*n, dV_frag[n], HD, wmma::mem_row_major);
    __syncthreads();
    store_out(outbuf, dVg + (long)bh*S*HD, kv_base, S);
    __syncthreads();
    // write dK (scaled)
    #pragma unroll
    for(int n=0;n<8;n++){
        #pragma unroll
        for(int i=0;i<dK_frag[n].num_elements;i++) dK_frag[n].x[i]*=scale;
        wmma::store_matrix_sync(outbuf + (16*warp)*HD + 16*n, dK_frag[n], HD, wmma::mem_row_major);
    }
    __syncthreads();
    store_out(outbuf, dKg + (long)bh*S*HD, kv_base, S);
}

// ---------------- dQ kernel ----------------
__launch_bounds__(128)
__global__ void bwd_dq_kernel(
    const __nv_bfloat16* __restrict__ Qg,
    const __nv_bfloat16* __restrict__ Kg,
    const __nv_bfloat16* __restrict__ Vg,
    const __nv_bfloat16* __restrict__ dOg,
    const float* __restrict__ Lg,
    const float* __restrict__ Dg,
    __nv_bfloat16* __restrict__ dQg,
    int S, float scale)
{
    const int bh = blockIdx.y;
    const int qb = blockIdx.x;
    const int q_base = qb*BLK;
    if (q_base >= S) return;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;

    extern __shared__ char smem[];
    __nv_bfloat16* Qs  = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* dOs = Qs + BLK*HD;
    __nv_bfloat16* Ks  = dOs + BLK*HD;
    __nv_bfloat16* Vs  = Ks + BLK*HD;
    __nv_bfloat16* dShi= Vs + BLK*HD;
    __nv_bfloat16* dSlo= dShi + BLK*BLK;
    float* sS  = reinterpret_cast<float*>(dSlo + BLK*BLK);
    float* sdP = sS + BLK*BLK;
    float* sL  = sdP + BLK*BLK;
    float* sD  = sL + BLK;
    float* outbuf = sS;

    const __nv_bfloat16* Qbase  = Qg  + (long)bh*S*HD;
    const __nv_bfloat16* Kbase  = Kg  + (long)bh*S*HD;
    const __nv_bfloat16* Vbase  = Vg  + (long)bh*S*HD;
    const __nv_bfloat16* dObase = dOg + (long)bh*S*HD;
    const float* Lbase = Lg + (long)bh*S;
    const float* Dbase = Dg + (long)bh*S;

    load_tile(Qbase, q_base, S, Qs);
    load_tile(dObase, q_base, S, dOs);
    for(int i=tid;i<BLK;i+=128){ int gq=q_base+i; sL[i]=(gq<S)?Lbase[gq]:0.f; sD[i]=(gq<S)?Dbase[gq]:0.f; }

    wmma::fragment<wmma::accumulator,16,16,16,float> dQ_frag[8];
    #pragma unroll
    for(int n=0;n<8;n++) wmma::fill_fragment(dQ_frag[n],0.f);
    __syncthreads();

    for(int kvb=0; kvb<=qb; ++kvb){
        int kv_base = kvb*BLK;
        load_tile(Kbase, kv_base, S, Ks);
        load_tile(Vbase, kv_base, S, Vs);
        __syncthreads();

        wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> b_col;
        // S = Q @ K^T
        #pragma unroll
        for(int n=0;n<4;n++){
            wmma::fragment<wmma::accumulator,16,16,16,float> acc; wmma::fill_fragment(acc,0.f);
            #pragma unroll
            for(int kt=0;kt<8;kt++){
                wmma::load_matrix_sync(a_frag, Qs + (16*warp)*HD + 16*kt, HD);
                wmma::load_matrix_sync(b_col, Ks + (16*n)*HD + 16*kt, HD);
                wmma::mma_sync(acc,a_frag,b_col,acc);
            }
            wmma::store_matrix_sync(sS + (16*warp)*BLK + 16*n, acc, BLK, wmma::mem_row_major);
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
            wmma::store_matrix_sync(sdP + (16*warp)*BLK + 16*n, acc, BLK, wmma::mem_row_major);
        }
        __syncthreads();

        for(int idx=tid; idx<BLK*BLK; idx+=128){
            int q = idx / BLK;
            int k = idx % BLK;
            int qg = q_base + q;
            int kg = kv_base + k;
            bool valid = (qg < S) && (kg <= qg) && (kg < S);
            float s = scale * sS[idx];
            float p = valid ? __expf(s - sL[q]) : 0.f;
            float dp = sdP[idx];
            float ds = p * (dp - sD[q]);
            __nv_bfloat16 hi,lo;
            split_bf16(ds, hi, lo);
            dShi[idx] = hi; dSlo[idx] = lo;
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> aS;
        wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> bK;
        // dQ += dS @ K  (hi + lo)
        #pragma unroll
        for(int n=0;n<8;n++){
            #pragma unroll
            for(int kt=0;kt<4;kt++){
                wmma::load_matrix_sync(bK, Ks + (16*kt)*HD + 16*n, HD);
                wmma::load_matrix_sync(aS, dShi + (16*warp)*BLK + 16*kt, BLK);
                wmma::mma_sync(dQ_frag[n], aS, bK, dQ_frag[n]);
                wmma::load_matrix_sync(aS, dSlo + (16*warp)*BLK + 16*kt, BLK);
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
    store_out(outbuf, dQg + (long)bh*S*HD, q_base, S);
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

    const __nv_bfloat16* Qp = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* Kp = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* Vp = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* Op = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dOp = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* Lp = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQp = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dKp = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dVp = static_cast<__nv_bfloat16*>(dV.data_ptr());

    long total = (long)BH * S;
    float* Dscratch = nullptr;
    CUDA_CHECK(cudaMallocAsync(&Dscratch, (size_t)total*sizeof(float), stream));

    compute_D<<<(unsigned)((total+255)/256), 256, 0, stream>>>(Op, dOp, Dscratch, total);
    CUDA_CHECK(cudaGetLastError());

    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(bwd_dkdv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, DKDV_SMEM);
        cudaFuncSetAttribute(bwd_dq_kernel,   cudaFuncAttributeMaxDynamicSharedMemorySize, DQ_SMEM);
        attr_set = true;
    }

    int nkv = (S + BLK - 1) / BLK;
    int nq  = nkv;

    dim3 g1(nkv, BH);
    bwd_dkdv_kernel<<<g1, 128, DKDV_SMEM, stream>>>(Qp, Kp, Vp, dOp, Lp, Dscratch, dKp, dVp, S, scale);
    CUDA_CHECK(cudaGetLastError());

    dim3 g2(nq, BH);
    bwd_dq_kernel<<<g2, 128, DQ_SMEM, stream>>>(Qp, Kp, Vp, dOp, Lp, Dscratch, dQp, S, scale);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFreeAsync(Dscratch, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd