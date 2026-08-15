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

constexpr int DH=128, THREADS=512, LDB=136, LDE=136, LDP=72, LDS=72;

__device__ __forceinline__ void load_tile(bf16* dst, const bf16* __restrict__ src, int row_base, int rows, int S){
  int tid=threadIdx.x;
  int total=rows*16;
  for (int idx=tid; idx<total; idx+=THREADS){
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

// Dual GEMM: F0 = A0@B0^T and F1 = A1@B1^T (both 64x64, contract 128).
// 16 warps split into two groups of 8; each warp computes 16x32 (2 col-tiles, shared a).
__device__ __forceinline__ void dual_gemm(
    const bf16* __restrict__ A0, const bf16* __restrict__ B0, float* __restrict__ F0,
    const bf16* __restrict__ A1, const bf16* __restrict__ B1, float* __restrict__ F1){
  int w=threadIdx.x>>5;
  int grp=w>>3;
  int lw=w&7;
  int mt=lw&3;
  int ch=lw>>2;
  const bf16* A = grp? A1:A0;
  const bf16* B = grp? B1:B0;
  float* F = grp? F1:F0;
  int nt0=ch*2, nt1=nt0+1;
  wmma::fragment<wmma::accumulator,16,16,16,float> c0,c1;
  wmma::fill_fragment(c0,0.f); wmma::fill_fragment(c1,0.f);
  wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> a;
  wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::col_major> b0,b1;
  #pragma unroll
  for (int kk=0;kk<DH/16;kk++){
     wmma::load_matrix_sync(a,  A + mt*16*LDB + kk*16, LDB);
     wmma::load_matrix_sync(b0, B + nt0*16*LDB + kk*16, LDB);
     wmma::load_matrix_sync(b1, B + nt1*16*LDB + kk*16, LDB);
     wmma::mma_sync(c0,a,b0,c0);
     wmma::mma_sync(c1,a,b1,c1);
  }
  wmma::store_matrix_sync(F + mt*16*LDS + nt0*16, c0, LDS, wmma::mem_row_major);
  wmma::store_matrix_sync(F + mt*16*LDS + nt1*16, c1, LDS, wmma::mem_row_major);
}

// ---------------- dK / dV kernel ----------------
__global__ void __launch_bounds__(THREADS,2) bwd_dkdv_kernel(
   const bf16* __restrict__ Q,const bf16* __restrict__ K,const bf16* __restrict__ V,const bf16* __restrict__ dO,
   const float* __restrict__ L,const float* __restrict__ Dbuf, bf16* __restrict__ dK, bf16* __restrict__ dV,
   int S, int num_q_blocks, float scale){
  constexpr int QM=64, KN=64;
  int bh=blockIdx.x; int kvb=blockIdx.y;
  size_t mat_off=(size_t)bh*S*DH;
  size_t lse_off=(size_t)bh*S;
  int kv_base=kvb*KN;

  extern __shared__ char smem[];
  bf16*  K_sh=(bf16*)(smem + 0);          // [64][136]
  bf16*  V_sh=(bf16*)(smem + 17408);      // [64][136]
  bf16*  Q_sh=(bf16*)(smem + 34816);      // [64][136]
  bf16*  dO_sh=(bf16*)(smem + 52224);     // [64][136]  end 69632
  float* F1_sh=(float*)(smem + 69632);    // [64][72]   end 88064   (S; then P bf16 in-place)
  float* F2_sh=(float*)(smem + 88064);    // [64][72]   end 106496  (dP; then dS bf16 in-place)
  float* L_sh=(float*)(smem + 106496);
  float* D_sh=(float*)(smem + 106752);    // end 107008
  bf16*  P_sh=(bf16*)F1_sh;
  bf16*  dS_sh=(bf16*)F2_sh;
  float* E=(float*)(smem + 0);            // [64][136] fp32 = 34816, overlaps K/V

  int tid=threadIdx.x; int w=tid>>5;
  int k_out=w&3, dpair=w>>2;
  int dt0=dpair*2, dt1=dt0+1;

  load_tile(K_sh, K+mat_off, kv_base, KN, S);
  load_tile(V_sh, V+mat_off, kv_base, KN, S);

  wmma::fragment<wmma::accumulator,16,16,16,float> accV0,accV1,accK0,accK1;
  wmma::fill_fragment(accV0,0.f); wmma::fill_fragment(accV1,0.f);
  wmma::fill_fragment(accK0,0.f); wmma::fill_fragment(accK1,0.f);

  for(int qb=0; qb<num_q_blocks; ++qb){
     int q_base=qb*QM;
     __syncthreads();
     load_tile(Q_sh, Q+mat_off, q_base, QM, S);
     load_tile(dO_sh, dO+mat_off, q_base, QM, S);
     for(int i=tid;i<QM;i+=THREADS){int r=q_base+i; L_sh[i]=(r<S)?L[lse_off+r]:0.f; D_sh[i]=(r<S)?Dbuf[lse_off+r]:0.f;}
     __syncthreads();

     dual_gemm(Q_sh, K_sh, F1_sh, dO_sh, V_sh, F2_sh);  // S -> F1, dP -> F2
     __syncthreads();

     float sbuf[8], dbuf[8];
     #pragma unroll
     for(int n=0;n<8;n++){int idx=tid+n*512;int q=idx>>6,k=idx&63; sbuf[n]=F1_sh[q*LDS+k]; dbuf[n]=F2_sh[q*LDS+k];}
     __syncthreads();
     #pragma unroll
     for(int n=0;n<8;n++){
        int idx=tid+n*512;int q=idx>>6,k=idx&63;int qi=q_base+q,kj=kv_base+k;
        float p=__expf(scale*sbuf[n]-L_sh[q]);
        if(qi>=S||kj>=S) p=0.f;
        float ds=p*(dbuf[n]-D_sh[q]);
        if(qi>=S||kj>=S) ds=0.f;
        P_sh[q*LDP+k]=__float2bfloat16(p);
        dS_sh[q*LDP+k]=__float2bfloat16(ds);
     }
     __syncthreads();

     // dV[k,d] += P^T@dO ; dK[k,d] += dS^T@Q  (contract q)
     #pragma unroll
     for(int kk=0;kk<QM/16;kk++){
        wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::col_major> aP,aS;
        wmma::load_matrix_sync(aP, P_sh  + kk*16*LDP + k_out*16, LDP);
        wmma::load_matrix_sync(aS, dS_sh + kk*16*LDP + k_out*16, LDP);
        wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::row_major> bO0,bO1,bQ0,bQ1;
        wmma::load_matrix_sync(bO0, dO_sh + kk*16*LDB + dt0*16, LDB);
        wmma::load_matrix_sync(bO1, dO_sh + kk*16*LDB + dt1*16, LDB);
        wmma::load_matrix_sync(bQ0, Q_sh  + kk*16*LDB + dt0*16, LDB);
        wmma::load_matrix_sync(bQ1, Q_sh  + kk*16*LDB + dt1*16, LDB);
        wmma::mma_sync(accV0, aP, bO0, accV0);
        wmma::mma_sync(accV1, aP, bO1, accV1);
        wmma::mma_sync(accK0, aS, bQ0, accK0);
        wmma::mma_sync(accK1, aS, bQ1, accK1);
     }
  }
  __syncthreads();
  wmma::store_matrix_sync(E + k_out*16*LDE + dt0*16, accV0, LDE, wmma::mem_row_major);
  wmma::store_matrix_sync(E + k_out*16*LDE + dt1*16, accV1, LDE, wmma::mem_row_major);
  __syncthreads();
  for(int idx=tid; idx<KN*DH; idx+=THREADS){int row=idx>>7; int col=idx&127; int r=kv_base+row; if(r<S) dV[mat_off+(size_t)r*DH+col]=__float2bfloat16(E[row*LDE+col]);}
  __syncthreads();
  wmma::store_matrix_sync(E + k_out*16*LDE + dt0*16, accK0, LDE, wmma::mem_row_major);
  wmma::store_matrix_sync(E + k_out*16*LDE + dt1*16, accK1, LDE, wmma::mem_row_major);
  __syncthreads();
  for(int idx=tid; idx<KN*DH; idx+=THREADS){int row=idx>>7; int col=idx&127; int r=kv_base+row; if(r<S) dK[mat_off+(size_t)r*DH+col]=__float2bfloat16(scale*E[row*LDE+col]);}
}

// ---------------- dQ kernel ----------------
__global__ void __launch_bounds__(THREADS,2) bwd_dq_kernel(
   const bf16* __restrict__ Q,const bf16* __restrict__ K,const bf16* __restrict__ V,const bf16* __restrict__ dO,
   const float* __restrict__ L,const float* __restrict__ Dbuf, bf16* __restrict__ dQ,
   int S, int num_kv_blocks, float scale){
  constexpr int QM=64, KN=64;
  int bh=blockIdx.x; int qb=blockIdx.y;
  size_t mat_off=(size_t)bh*S*DH;
  size_t lse_off=(size_t)bh*S;
  int q_base=qb*QM;

  extern __shared__ char smem[];
  bf16*  Q_sh=(bf16*)(smem + 0);          // [64][136]
  bf16*  dO_sh=(bf16*)(smem + 17408);     // [64][136]
  bf16*  K_sh=(bf16*)(smem + 34816);      // [64][136]
  bf16*  V_sh=(bf16*)(smem + 52224);      // [64][136]  end 69632
  float* F1_sh=(float*)(smem + 69632);    // [64][72]  (S; then dS bf16 in-place)
  float* F2_sh=(float*)(smem + 88064);    // [64][72]  (dP)
  float* L_sh=(float*)(smem + 106496);
  float* D_sh=(float*)(smem + 106752);    // end 107008
  bf16*  dS_sh=(bf16*)F1_sh;
  float* E=(float*)(smem + 0);            // [64][136] fp32, overlaps Q/dO/K

  int tid=threadIdx.x; int w=tid>>5;
  int q_out=w&3, dpair=w>>2;
  int dt0=dpair*2, dt1=dt0+1;

  load_tile(Q_sh, Q+mat_off, q_base, QM, S);
  load_tile(dO_sh, dO+mat_off, q_base, QM, S);
  for(int i=tid;i<QM;i+=THREADS){int r=q_base+i; L_sh[i]=(r<S)?L[lse_off+r]:0.f; D_sh[i]=(r<S)?Dbuf[lse_off+r]:0.f;}

  wmma::fragment<wmma::accumulator,16,16,16,float> accQ0,accQ1;
  wmma::fill_fragment(accQ0,0.f); wmma::fill_fragment(accQ1,0.f);
  __syncthreads();

  for(int kvb=0; kvb<num_kv_blocks; ++kvb){
     int kv_base=kvb*KN;
     __syncthreads();
     load_tile(K_sh, K+mat_off, kv_base, KN, S);
     load_tile(V_sh, V+mat_off, kv_base, KN, S);
     __syncthreads();

     dual_gemm(Q_sh, K_sh, F1_sh, dO_sh, V_sh, F2_sh);  // S -> F1, dP -> F2
     __syncthreads();

     float sbuf[8], dbuf[8];
     #pragma unroll
     for(int n=0;n<8;n++){int idx=tid+n*512;int q=idx>>6,k=idx&63; sbuf[n]=F1_sh[q*LDS+k]; dbuf[n]=F2_sh[q*LDS+k];}
     __syncthreads();
     #pragma unroll
     for(int n=0;n<8;n++){
        int idx=tid+n*512;int q=idx>>6,k=idx&63;int qi=q_base+q,kj=kv_base+k;
        float p=__expf(scale*sbuf[n]-L_sh[q]);
        if(qi>=S||kj>=S) p=0.f;
        float ds=p*(dbuf[n]-D_sh[q]);
        dS_sh[q*LDP+k]=__float2bfloat16(ds);
     }
     __syncthreads();

     // dQ[q,d] += dS@K  (contract k)
     #pragma unroll
     for(int kk=0;kk<KN/16;kk++){
        wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> a;
        wmma::load_matrix_sync(a, dS_sh + q_out*16*LDP + kk*16, LDP);
        wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::row_major> b0,b1;
        wmma::load_matrix_sync(b0, K_sh + kk*16*LDB + dt0*16, LDB);
        wmma::load_matrix_sync(b1, K_sh + kk*16*LDB + dt1*16, LDB);
        wmma::mma_sync(accQ0, a, b0, accQ0);
        wmma::mma_sync(accQ1, a, b1, accQ1);
     }
  }
  __syncthreads();
  wmma::store_matrix_sync(E + q_out*16*LDE + dt0*16, accQ0, LDE, wmma::mem_row_major);
  wmma::store_matrix_sync(E + q_out*16*LDE + dt1*16, accQ1, LDE, wmma::mem_row_major);
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

  int num_blocks=(S+63)/64;
  float scale=1.0f/std::sqrt((float)DH);

  int smem_bytes=107008;
  CUDA_CHECK(cudaFuncSetAttribute(bwd_dkdv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
  CUDA_CHECK(cudaFuncSetAttribute(bwd_dq_kernel,   cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

  dim3 grid(BH, num_blocks);
  bwd_dkdv_kernel<<<grid, THREADS, smem_bytes, stream>>>(Qp,Kp,Vp,dOp,Lp,Dbuf,dKp,dVp,S,num_blocks,scale);
  CUDA_CHECK(cudaGetLastError());
  bwd_dq_kernel<<<grid, THREADS, smem_bytes, stream>>>(Qp,Kp,Vp,dOp,Lp,Dbuf,dQp,S,num_blocks,scale);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaFreeAsync(Dbuf, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd