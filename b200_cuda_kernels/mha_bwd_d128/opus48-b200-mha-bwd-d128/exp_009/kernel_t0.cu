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

constexpr int BM=64, BN=64, DH=128, THREADS=256;

__device__ __forceinline__ void load_tile(bf16* dst, const bf16* src_base, int row_base, int S){
  int tid=threadIdx.x;
  #pragma unroll
  for (int idx=tid; idx<BM*DH/8; idx+=THREADS){
     int row = idx >> 4;
     int c8  = (idx & 15) << 3;
     int r = row_base + row;
     int4 v;
     if (r < S) v = *reinterpret_cast<const int4*>(src_base + (size_t)r*DH + c8);
     else { v.x=v.y=v.z=v.w=0; }
     reinterpret_cast<int4*>(dst)[idx] = v;
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

// C[64][64] = A[64][128] @ B[64][128]^T   (C[i,j] = sum_k A[i,k]*B[j,k])
__device__ __forceinline__ void gemm_ABt(const bf16* A_sh, const bf16* B_sh, float* C_sh){
  int warp = threadIdx.x>>5;
  #pragma unroll
  for (int e=0;e<2;e++){
     int t=warp*2+e; int mt=t>>2; int nt=t&3;
     wmma::fragment<wmma::accumulator,16,16,16,float> c;
     wmma::fill_fragment(c,0.f);
     wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> a;
     wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::col_major> b;
     #pragma unroll
     for (int kk=0;kk<8;kk++){
        wmma::load_matrix_sync(a, A_sh + mt*16*DH + kk*16, DH);
        wmma::load_matrix_sync(b, B_sh + nt*16*DH + kk*16, DH);
        wmma::mma_sync(c,a,b,c);
     }
     wmma::store_matrix_sync(C_sh + mt*16*64 + nt*16, c, 64, wmma::mem_row_major);
  }
}

__global__ void __launch_bounds__(256) bwd_dkdv_kernel(
   const bf16* Q,const bf16* K,const bf16* V,const bf16* dO,
   const float* L,const float* Dbuf, bf16* dK, bf16* dV,
   int S, int num_q_blocks, float scale){

  int bh=blockIdx.x; int kvb=blockIdx.y;
  size_t mat_off=(size_t)bh*S*DH;
  size_t lse_off=(size_t)bh*S;
  int kv_base=kvb*BN;

  extern __shared__ char smem[];
  float* S_sh=(float*)smem;
  float* dP_sh=S_sh + 64*64;
  bf16* K_sh=(bf16*)(dP_sh + 64*64);
  bf16* V_sh=K_sh + 64*DH;
  bf16* Q_sh=V_sh + 64*DH;
  bf16* dO_sh=Q_sh + 64*DH;
  bf16* P_sh=dO_sh + 64*DH;
  bf16* dS_sh=P_sh + 64*64;
  float* L_sh=(float*)(dS_sh + 64*64);
  float* D_sh=L_sh + 64;
  float* sh_f=S_sh;

  int tid=threadIdx.x; int warp=tid>>5;
  int mt_out=warp&3; int cg=warp>>2;

  load_tile(K_sh, K+mat_off, kv_base, S);
  load_tile(V_sh, V+mat_off, kv_base, S);

  wmma::fragment<wmma::accumulator,16,16,16,float> accV[4], accK[4];
  #pragma unroll
  for(int i=0;i<4;i++){wmma::fill_fragment(accV[i],0.f); wmma::fill_fragment(accK[i],0.f);}

  for(int qb=0; qb<num_q_blocks; ++qb){
     int q_base=qb*BM;
     __syncthreads();
     load_tile(Q_sh, Q+mat_off, q_base, S);
     load_tile(dO_sh, dO+mat_off, q_base, S);
     for(int i=tid;i<64;i+=THREADS){int r=q_base+i; L_sh[i]=(r<S)?L[lse_off+r]:0.f; D_sh[i]=(r<S)?Dbuf[lse_off+r]:0.f;}
     __syncthreads();

     gemm_ABt(Q_sh, K_sh, S_sh);    // S  = Q@K^T
     gemm_ABt(dO_sh, V_sh, dP_sh);  // dP = dO@V^T
     __syncthreads();

     for(int idx=tid; idx<64*64; idx+=THREADS){
        int i=idx>>6; int j=idx&63;
        int qi=q_base+i; int kj=kv_base+j;
        float p=__expf(scale*S_sh[idx]-L_sh[i]);
        float ds=p*(dP_sh[idx]-D_sh[i]);
        if(qi>=S||kj>=S){p=0.f; ds=0.f;}
        P_sh[idx]=__float2bfloat16(p);
        dS_sh[idx]=__float2bfloat16(ds);
     }
     __syncthreads();

     // dV += P^T@dO ; dK(pre-scale) += dS^T@Q
     #pragma unroll
     for(int nl=0;nl<4;nl++){
        int nt=cg*4+nl;
        wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::col_major> aP,aS;
        wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::row_major> bO,bQ;
        #pragma unroll
        for(int kk=0;kk<4;kk++){
           wmma::load_matrix_sync(aP, P_sh + kk*16*64 + mt_out*16, 64);
           wmma::load_matrix_sync(bO, dO_sh + kk*16*DH + nt*16, DH);
           wmma::mma_sync(accV[nl], aP, bO, accV[nl]);
           wmma::load_matrix_sync(aS, dS_sh + kk*16*64 + mt_out*16, 64);
           wmma::load_matrix_sync(bQ, Q_sh + kk*16*DH + nt*16, DH);
           wmma::mma_sync(accK[nl], aS, bQ, accK[nl]);
        }
     }
  }
  __syncthreads();
  // epilogue dV
  #pragma unroll
  for(int nl=0;nl<4;nl++){int nt=cg*4+nl; wmma::store_matrix_sync(sh_f + mt_out*16*DH + nt*16, accV[nl], DH, wmma::mem_row_major);}
  __syncthreads();
  for(int idx=tid; idx<64*DH; idx+=THREADS){int row=idx/DH; int col=idx%DH; int r=kv_base+row; if(r<S) dV[mat_off+(size_t)r*DH+col]=__float2bfloat16(sh_f[idx]);}
  __syncthreads();
  // epilogue dK (apply scale)
  #pragma unroll
  for(int nl=0;nl<4;nl++){int nt=cg*4+nl; wmma::store_matrix_sync(sh_f + mt_out*16*DH + nt*16, accK[nl], DH, wmma::mem_row_major);}
  __syncthreads();
  for(int idx=tid; idx<64*DH; idx+=THREADS){int row=idx/DH; int col=idx%DH; int r=kv_base+row; if(r<S) dK[mat_off+(size_t)r*DH+col]=__float2bfloat16(scale*sh_f[idx]);}
}

__global__ void __launch_bounds__(256) bwd_dq_kernel(
   const bf16* Q,const bf16* K,const bf16* V,const bf16* dO,
   const float* L,const float* Dbuf, bf16* dQ,
   int S, int num_kv_blocks, float scale){

  int bh=blockIdx.x; int qb=blockIdx.y;
  size_t mat_off=(size_t)bh*S*DH;
  size_t lse_off=(size_t)bh*S;
  int q_base=qb*BM;

  extern __shared__ char smem[];
  float* S_sh=(float*)smem;
  float* dP_sh=S_sh + 64*64;
  bf16* K_sh=(bf16*)(dP_sh + 64*64);
  bf16* V_sh=K_sh + 64*DH;
  bf16* Q_sh=V_sh + 64*DH;
  bf16* dO_sh=Q_sh + 64*DH;
  bf16* dS_sh=dO_sh + 64*DH;
  float* L_sh=(float*)(dS_sh + 64*64);
  float* D_sh=L_sh + 64;
  float* sh_f=S_sh;

  int tid=threadIdx.x; int warp=tid>>5;
  int mt_out=warp&3; int cg=warp>>2;

  load_tile(Q_sh, Q+mat_off, q_base, S);
  load_tile(dO_sh, dO+mat_off, q_base, S);
  for(int i=tid;i<64;i+=THREADS){int r=q_base+i; L_sh[i]=(r<S)?L[lse_off+r]:0.f; D_sh[i]=(r<S)?Dbuf[lse_off+r]:0.f;}

  wmma::fragment<wmma::accumulator,16,16,16,float> accQ[4];
  #pragma unroll
  for(int i=0;i<4;i++) wmma::fill_fragment(accQ[i],0.f);
  __syncthreads();

  for(int kvb=0; kvb<num_kv_blocks; ++kvb){
     int kv_base=kvb*BN;
     __syncthreads();
     load_tile(K_sh, K+mat_off, kv_base, S);
     load_tile(V_sh, V+mat_off, kv_base, S);
     __syncthreads();
     gemm_ABt(Q_sh, K_sh, S_sh);
     gemm_ABt(dO_sh, V_sh, dP_sh);
     __syncthreads();
     for(int idx=tid; idx<64*64; idx+=THREADS){
        int i=idx>>6; int j=idx&63;
        int qi=q_base+i; int kj=kv_base+j;
        float p=__expf(scale*S_sh[idx]-L_sh[i]);
        float ds=p*(dP_sh[idx]-D_sh[i]);
        if(qi>=S||kj>=S) ds=0.f;
        dS_sh[idx]=__float2bfloat16(ds);
     }
     __syncthreads();
     // dQ(pre-scale) += dS@K
     #pragma unroll
     for(int nl=0;nl<4;nl++){
        int nt=cg*4+nl;
        wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> a;
        wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::row_major> b;
        #pragma unroll
        for(int kk=0;kk<4;kk++){
           wmma::load_matrix_sync(a, dS_sh + mt_out*16*64 + kk*16, 64);
           wmma::load_matrix_sync(b, K_sh + kk*16*DH + nt*16, DH);
           wmma::mma_sync(accQ[nl], a, b, accQ[nl]);
        }
     }
  }
  __syncthreads();
  #pragma unroll
  for(int nl=0;nl<4;nl++){int nt=cg*4+nl; wmma::store_matrix_sync(sh_f + mt_out*16*DH + nt*16, accQ[nl], DH, wmma::mem_row_major);}
  __syncthreads();
  for(int idx=tid; idx<64*DH; idx+=THREADS){int row=idx/DH; int col=idx%DH; int r=q_base+row; if(r<S) dQ[mat_off+(size_t)r*DH+col]=__float2bfloat16(scale*sh_f[idx]);}
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
  int dblocks=(int)(((long long)nrows*32 + THREADS-1)/THREADS);
  if (dblocks < 1) dblocks = 1;
  compute_D_kernel<<<dblocks, THREADS, 0, stream>>>(Op, dOp, Dbuf, nrows);
  CUDA_CHECK(cudaGetLastError());

  int num_blocks=(S+BN-1)/BN;
  float scale=1.0f/std::sqrt((float)DH);

  size_t smem_dkdv=115200;
  size_t smem_dq  =107008;
  CUDA_CHECK(cudaFuncSetAttribute(bwd_dkdv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_dkdv));
  CUDA_CHECK(cudaFuncSetAttribute(bwd_dq_kernel,   cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_dq));

  dim3 grid(BH, num_blocks);
  bwd_dkdv_kernel<<<grid, THREADS, smem_dkdv, stream>>>(Qp,Kp,Vp,dOp,Lp,Dbuf,dKp,dVp,S,num_blocks,scale);
  CUDA_CHECK(cudaGetLastError());
  bwd_dq_kernel<<<grid, THREADS, smem_dq, stream>>>(Qp,Kp,Vp,dOp,Lp,Dbuf,dQp,S,num_blocks,scale);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaFreeAsync(Dbuf, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd