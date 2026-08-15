#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cmath>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

namespace mha_bwd {

using namespace nvcuda;
typedef __nv_bfloat16 bf16;

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__);} } while(0)

constexpr int DH=128, THREADS=256, LDB=136, LDE=136;

template<int ROWS>
__device__ __forceinline__ void load_tile(bf16* dst, const bf16* __restrict__ src, int row_base, int S){
  int tid=threadIdx.x;
  #pragma unroll
  for (int idx=tid; idx<ROWS*16; idx+=THREADS){
     int row = idx >> 4;
     int c8  = (idx & 15) << 3;
     int r = row_base + row;
     int4 v;
     if (r < S) v = *reinterpret_cast<const int4*>(src + (size_t)r*DH + c8);
     else { v.x=v.y=v.z=v.w=0; }
     *reinterpret_cast<int4*>(dst + row*LDB + c8) = v;
  }
}

__global__ void compute_D_kernel(const bf16* O, const bf16* dO, float* D, int nrows){
  int gw = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
  int lane = threadIdx.x & 31;
  if (gw >= nrows) return;
  const bf16* o = O + (size_t)gw*DH;
  const bf16* g = dO + (size_t)gw*DH;
  float s=0.f;
  #pragma unroll
  for (int c=lane;c<DH;c+=32) s += __bfloat162float(o[c])*__bfloat162float(g[c]);
  #pragma unroll
  for (int off=16; off>0; off>>=1) s += __shfl_down_sync(0xffffffffu, s, off);
  if (lane==0) D[gw]=s;
}

// ---------------- dK / dV kernel : QM=64 (q), KN=32 (k) ----------------
__global__ void __launch_bounds__(THREADS,3) bwd_dkdv_kernel(
   const bf16* __restrict__ Q,const bf16* __restrict__ K,const bf16* __restrict__ V,const bf16* __restrict__ dO,
   const float* __restrict__ L,const float* __restrict__ Dbuf, bf16* __restrict__ dK, bf16* __restrict__ dV,
   int S, int num_q_blocks, float scale){
  constexpr int QM=64, KN=32, LDS=40, LDP=40;

  int bh=blockIdx.x; int kvb=blockIdx.y;
  size_t mat_off=(size_t)bh*S*DH;
  size_t lse_off=(size_t)bh*S;
  int kv_base=kvb*KN;

  extern __shared__ char smem[];
  bf16*  K_sh=(bf16*)(smem + 0);          // [32][136]
  bf16*  V_sh=(bf16*)(smem + 8704);       // [32][136]
  bf16*  Q_sh=(bf16*)(smem + 17408);      // [64][136]
  bf16*  dO_sh=(bf16*)(smem + 34816);     // [64][136]
  float* F_sh=(float*)(smem + 52224);     // [64][40]
  bf16*  P_sh=(bf16*)(smem + 62464);      // [64][40]
  bf16*  dS_sh=(bf16*)(smem + 67584);     // [64][40]
  float* L_sh=(float*)(smem + 72704);     // [64]
  float* D_sh=(float*)(smem + 72960);     // [64]
  float* E=(float*)(smem + 0);            // [32][136] epilogue, overlaps K/V

  int tid=threadIdx.x; int w=tid>>5;
  int smt=w>>1, snt=w&1;               // S gemm: q-tile 0..3, k-tile 0..1
  int kt=w>>2, dbase=(w&3)*2;          // accum: k-tile 0..1, d-tiles dbase,dbase+1

  load_tile<KN>(K_sh, K+mat_off, kv_base, S);
  load_tile<KN>(V_sh, V+mat_off, kv_base, S);

  wmma::fragment<wmma::accumulator,16,16,16,float> accV0,accV1,accK0,accK1;
  wmma::fill_fragment(accV0,0.f); wmma::fill_fragment(accV1,0.f);
  wmma::fill_fragment(accK0,0.f); wmma::fill_fragment(accK1,0.f);

  for(int qb=0; qb<num_q_blocks; ++qb){
     int q_base=qb*QM;
     __syncthreads();
     load_tile<QM>(Q_sh, Q+mat_off, q_base, S);
     load_tile<QM>(dO_sh, dO+mat_off, q_base, S);
     for(int i=tid;i<QM;i+=THREADS){int r=q_base+i; L_sh[i]=(r<S)?L[lse_off+r]:0.f; D_sh[i]=(r<S)?Dbuf[lse_off+r]:0.f;}
     __syncthreads();

     // S[q,k] = Q @ K^T
     {
       wmma::fragment<wmma::accumulator,16,16,16,float> c; wmma::fill_fragment(c,0.f);
       wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> a;
       wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::col_major> b;
       #pragma unroll
       for(int kk=0;kk<DH/16;kk++){
         wmma::load_matrix_sync(a, Q_sh + smt*16*LDB + kk*16, LDB);
         wmma::load_matrix_sync(b, K_sh + snt*16*LDB + kk*16, LDB);
         wmma::mma_sync(c,a,b,c);
       }
       wmma::store_matrix_sync(F_sh + smt*16*LDS + snt*16, c, LDS, wmma::mem_row_major);
     }
     __syncthreads();
     for(int idx=tid; idx<QM*KN; idx+=THREADS){
        int q=idx>>5; int k=idx&31;
        int qi=q_base+q; int kj=kv_base+k;
        float p=__expf(scale*F_sh[q*LDS+k]-L_sh[q]);
        if(qi>=S||kj>=S) p=0.f;
        P_sh[q*LDP+k]=__float2bfloat16(p);
     }
     __syncthreads();
     // dP[q,k] = dO @ V^T
     {
       wmma::fragment<wmma::accumulator,16,16,16,float> c; wmma::fill_fragment(c,0.f);
       wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> a;
       wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::col_major> b;
       #pragma unroll
       for(int kk=0;kk<DH/16;kk++){
         wmma::load_matrix_sync(a, dO_sh + smt*16*LDB + kk*16, LDB);
         wmma::load_matrix_sync(b, V_sh + snt*16*LDB + kk*16, LDB);
         wmma::mma_sync(c,a,b,c);
       }
       wmma::store_matrix_sync(F_sh + smt*16*LDS + snt*16, c, LDS, wmma::mem_row_major);
     }
     __syncthreads();
     for(int idx=tid; idx<QM*KN; idx+=THREADS){
        int q=idx>>5; int k=idx&31;
        float p=__bfloat162float(P_sh[q*LDP+k]);
        float ds=p*(F_sh[q*LDS+k]-D_sh[q]);
        dS_sh[q*LDP+k]=__float2bfloat16(ds);
     }
     __syncthreads();

     // dV[k,d] += P^T @ dO ; dK[k,d] += dS^T @ Q   (contract q)
     #pragma unroll
     for(int kk=0;kk<QM/16;kk++){
        wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::col_major> aP,aS;
        wmma::load_matrix_sync(aP, P_sh  + kk*16*LDP + kt*16, LDP);
        wmma::load_matrix_sync(aS, dS_sh + kk*16*LDP + kt*16, LDP);
        wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::row_major> bO0,bO1,bQ0,bQ1;
        wmma::load_matrix_sync(bO0, dO_sh + kk*16*LDB + dbase*16, LDB);
        wmma::load_matrix_sync(bO1, dO_sh + kk*16*LDB + (dbase+1)*16, LDB);
        wmma::load_matrix_sync(bQ0, Q_sh  + kk*16*LDB + dbase*16, LDB);
        wmma::load_matrix_sync(bQ1, Q_sh  + kk*16*LDB + (dbase+1)*16, LDB);
        wmma::mma_sync(accV0, aP, bO0, accV0);
        wmma::mma_sync(accV1, aP, bO1, accV1);
        wmma::mma_sync(accK0, aS, bQ0, accK0);
        wmma::mma_sync(accK1, aS, bQ1, accK1);
     }
  }
  __syncthreads();
  wmma::store_matrix_sync(E + kt*16*LDE + dbase*16, accV0, LDE, wmma::mem_row_major);
  wmma::store_matrix_sync(E + kt*16*LDE + (dbase+1)*16, accV1, LDE, wmma::mem_row_major);
  __syncthreads();
  for(int idx=tid; idx<KN*DH; idx+=THREADS){int row=idx>>7; int col=idx&127; int r=kv_base+row; if(r<S) dV[mat_off+(size_t)r*DH+col]=__float2bfloat16(E[row*LDE+col]);}
  __syncthreads();
  wmma::store_matrix_sync(E + kt*16*LDE + dbase*16, accK0, LDE, wmma::mem_row_major);
  wmma::store_matrix_sync(E + kt*16*LDE + (dbase+1)*16, accK1, LDE, wmma::mem_row_major);
  __syncthreads();
  for(int idx=tid; idx<KN*DH; idx+=THREADS){int row=idx>>7; int col=idx&127; int r=kv_base+row; if(r<S) dK[mat_off+(size_t)r*DH+col]=__float2bfloat16(scale*E[row*LDE+col]);}
}

// ---------------- dQ kernel : QM=32 (q), KN=64 (k) ----------------
__global__ void __launch_bounds__(THREADS,3) bwd_dq_kernel(
   const bf16* __restrict__ Q,const bf16* __restrict__ K,const bf16* __restrict__ V,const bf16* __restrict__ dO,
   const float* __restrict__ L,const float* __restrict__ Dbuf, bf16* __restrict__ dQ,
   int S, int num_kv_blocks, float scale){
  constexpr int QM=32, KN=64, LDS=72, LDP=72;

  int bh=blockIdx.x; int qb=blockIdx.y;
  size_t mat_off=(size_t)bh*S*DH;
  size_t lse_off=(size_t)bh*S;
  int q_base=qb*QM;

  extern __shared__ char smem[];
  bf16*  Q_sh=(bf16*)(smem + 0);          // [32][136]
  bf16*  dO_sh=(bf16*)(smem + 8704);      // [32][136]
  bf16*  K_sh=(bf16*)(smem + 17408);      // [64][136]
  bf16*  V_sh=(bf16*)(smem + 34816);      // [64][136]
  float* F_sh=(float*)(smem + 52224);     // [32][72]
  bf16*  dS_sh=(bf16*)(smem + 61440);     // [32][72] (P then dS)
  float* L_sh=(float*)(smem + 66048);     // [32]
  float* D_sh=(float*)(smem + 66176);     // [32]
  float* E=(float*)(smem + 17408);        // [32][136], overlaps K/V

  int tid=threadIdx.x; int w=tid>>5;
  int smt=w>>2, snt=w&3;               // S gemm: q-tile 0..1, k-tile 0..3
  int qt=w>>2, dbase=(w&3)*2;          // accum: q-tile 0..1, d-tiles dbase,dbase+1

  load_tile<QM>(Q_sh, Q+mat_off, q_base, S);
  load_tile<QM>(dO_sh, dO+mat_off, q_base, S);
  for(int i=tid;i<QM;i+=THREADS){int r=q_base+i; L_sh[i]=(r<S)?L[lse_off+r]:0.f; D_sh[i]=(r<S)?Dbuf[lse_off+r]:0.f;}

  wmma::fragment<wmma::accumulator,16,16,16,float> accQ0,accQ1;
  wmma::fill_fragment(accQ0,0.f); wmma::fill_fragment(accQ1,0.f);
  __syncthreads();

  for(int kvb=0; kvb<num_kv_blocks; ++kvb){
     int kv_base=kvb*KN;
     __syncthreads();
     load_tile<KN>(K_sh, K+mat_off, kv_base, S);
     load_tile<KN>(V_sh, V+mat_off, kv_base, S);
     __syncthreads();
     // S[q,k] = Q @ K^T
     {
       wmma::fragment<wmma::accumulator,16,16,16,float> c; wmma::fill_fragment(c,0.f);
       wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> a;
       wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::col_major> b;
       #pragma unroll
       for(int kk=0;kk<DH/16;kk++){
         wmma::load_matrix_sync(a, Q_sh + smt*16*LDB + kk*16, LDB);
         wmma::load_matrix_sync(b, K_sh + snt*16*LDB + kk*16, LDB);
         wmma::mma_sync(c,a,b,c);
       }
       wmma::store_matrix_sync(F_sh + smt*16*LDS + snt*16, c, LDS, wmma::mem_row_major);
     }
     __syncthreads();
     for(int idx=tid; idx<QM*KN; idx+=THREADS){
        int q=idx>>6; int k=idx&63;
        int qi=q_base+q; int kj=kv_base+k;
        float p=__expf(scale*F_sh[q*LDS+k]-L_sh[q]);
        if(qi>=S||kj>=S) p=0.f;
        dS_sh[q*LDP+k]=__float2bfloat16(p);
     }
     __syncthreads();
     // dP[q,k] = dO @ V^T
     {
       wmma::fragment<wmma::accumulator,16,16,16,float> c; wmma::fill_fragment(c,0.f);
       wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> a;
       wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::col_major> b;
       #pragma unroll
       for(int kk=0;kk<DH/16;kk++){
         wmma::load_matrix_sync(a, dO_sh + smt*16*LDB + kk*16, LDB);
         wmma::load_matrix_sync(b, V_sh + snt*16*LDB + kk*16, LDB);
         wmma::mma_sync(c,a,b,c);
       }
       wmma::store_matrix_sync(F_sh + smt*16*LDS + snt*16, c, LDS, wmma::mem_row_major);
     }
     __syncthreads();
     for(int idx=tid; idx<QM*KN; idx+=THREADS){
        int q=idx>>6; int k=idx&63;
        float p=__bfloat162float(dS_sh[q*LDP+k]);
        float ds=p*(F_sh[q*LDS+k]-D_sh[q]);
        dS_sh[q*LDP+k]=__float2bfloat16(ds);
     }
     __syncthreads();
     // dQ[q,d] += dS @ K   (contract k)
     #pragma unroll
     for(int kk=0;kk<KN/16;kk++){
        wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> a;
        wmma::load_matrix_sync(a, dS_sh + qt*16*LDP + kk*16, LDP);
        wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::row_major> b0,b1;
        wmma::load_matrix_sync(b0, K_sh + kk*16*LDB + dbase*16, LDB);
        wmma::load_matrix_sync(b1, K_sh + kk*16*LDB + (dbase+1)*16, LDB);
        wmma::mma_sync(accQ0, a, b0, accQ0);
        wmma::mma_sync(accQ1, a, b1, accQ1);
     }
  }
  __syncthreads();
  wmma::store_matrix_sync(E + qt*16*LDE + dbase*16, accQ0, LDE, wmma::mem_row_major);
  wmma::store_matrix_sync(E + qt*16*LDE + (dbase+1)*16, accQ1, LDE, wmma::mem_row_major);
  __syncthreads();
  for(int idx=tid; idx<QM*DH; idx+=THREADS){int row=idx>>7; int col=idx&127; int r=q_base+row; if(r<S) dQ[mat_off+(size_t)r*DH+col]=__float2bfloat16(scale*E[row*LDE+col]);}
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int Bsz=(int)Q.size(0), Hsz=(int)Q.size(1), S=(int)Q.size(2);
  int BH=Bsz*Hsz;

  const bf16* Qp=static_cast<const bf16*>(Q.data_ptr());
  const bf16* Kp=static_cast<const bf16*>(K.data_ptr());
  const bf16* Vp=static_cast<const bf16*>(V.data_ptr());
  const bf16* Op=static_cast<const bf16*>(O.data_ptr());
  const bf16* dOp=static_cast<const bf16*>(dO.data_ptr());
  const float* Lp=static_cast<const float*>(L.data_ptr());
  bf16* dQp=static_cast<bf16*>(dQ.data_ptr());
  bf16* dKp=static_cast<bf16*>(dK.data_ptr());
  bf16* dVp=static_cast<bf16*>(dV.data_ptr());

  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

  float* Dbuf=nullptr;
  CUDA_CHECK(cudaMallocAsync(&Dbuf, (size_t)BH*S*sizeof(float), stream));

  int nrows=BH*S;
  int dblocks=(int)(((long long)nrows*32 + 255)/256);
  if (dblocks < 1) dblocks = 1;
  compute_D_kernel<<<dblocks, 256, 0, stream>>>(Op, dOp, Dbuf, nrows);
  CUDA_CHECK(cudaGetLastError());

  float scale=1.0f/std::sqrt((float)DH);

  int smem_dkdv=73216;
  int smem_dq  =66304;
  CUDA_CHECK(cudaFuncSetAttribute(bwd_dkdv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_dkdv));
  CUDA_CHECK(cudaFuncSetAttribute(bwd_dq_kernel,   cudaFuncAttributeMaxDynamicSharedMemorySize, smem_dq));

  int dkdv_kv = (S+31)/32;   int dkdv_q = (S+63)/64;
  int dq_q    = (S+31)/32;   int dq_kv  = (S+63)/64;

  dim3 grid_dkdv(BH, dkdv_kv);
  bwd_dkdv_kernel<<<grid_dkdv, THREADS, smem_dkdv, stream>>>(Qp,Kp,Vp,dOp,Lp,Dbuf,dKp,dVp,S,dkdv_q,scale);
  CUDA_CHECK(cudaGetLastError());
  dim3 grid_dq(BH, dq_q);
  bwd_dq_kernel<<<grid_dq, THREADS, smem_dq, stream>>>(Qp,Kp,Vp,dOp,Lp,Dbuf,dQp,S,dq_kv,scale);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaFreeAsync(Dbuf, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd