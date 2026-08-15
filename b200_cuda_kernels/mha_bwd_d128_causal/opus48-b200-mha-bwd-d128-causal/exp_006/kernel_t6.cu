#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e=(call); \
    if(_e!=cudaSuccess){fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__);exit(1);} \
}while(0)

namespace mha_bwd {

using namespace nvcuda;
using bf16 = __nv_bfloat16;
__device__ __forceinline__ float b2f(bf16 x){ return __bfloat162float(x); }

constexpr int DD = 128;
constexpr int BN = 64;
constexpr int BM = 64;
constexpr int NT = 256;

using AR = wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major>;
using BC = wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::col_major>;
using BR = wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::row_major>;
using ACC = wmma::fragment<wmma::accumulator,16,16,16,float>;

__device__ __forceinline__ void split_bf16(float x, bf16& hi, bf16& lo){
  hi = __float2bfloat16(x);
  lo = __float2bfloat16(x - __bfloat162float(hi));
}

__global__ void delta_kernel(const bf16* __restrict__ O, const bf16* __restrict__ dO,
                             float* __restrict__ Delta, long rows){
  long warp = ((long)blockIdx.x*blockDim.x + threadIdx.x)/32;
  int lane = threadIdx.x % 32;
  if (warp >= rows) return;
  const bf16* o = O + warp*DD;
  const bf16* g = dO + warp*DD;
  float acc=0.f;
  #pragma unroll
  for (int k=lane;k<DD;k+=32) acc += b2f(o[k])*b2f(g[k]);
  #pragma unroll
  for (int off=16; off>0; off>>=1) acc += __shfl_down_sync(0xffffffffu, acc, off);
  if (lane==0) Delta[warp]=acc;
}

__global__ void dkdv_kernel(
    const bf16* __restrict__ Q, const bf16* __restrict__ K, const bf16* __restrict__ V,
    const bf16* __restrict__ dO, const float* __restrict__ L, const float* __restrict__ Delta,
    bf16* __restrict__ dK, bf16* __restrict__ dV, int S, float scale)
{
  int bh = blockIdx.y;
  int kv0 = blockIdx.x * BN;
  if (kv0 >= S) return;

  const bf16* Kb = K + (size_t)bh*S*DD;
  const bf16* Vb = V + (size_t)bh*S*DD;
  const bf16* Qb = Q + (size_t)bh*S*DD;
  const bf16* dOb= dO+ (size_t)bh*S*DD;
  const float* Lb = L + (size_t)bh*S;
  const float* Db = Delta + (size_t)bh*S;
  bf16* dKb = dK + (size_t)bh*S*DD;
  bf16* dVb = dV + (size_t)bh*S*DD;

  extern __shared__ char smem[];
  bf16* sK   = (bf16*)smem;
  bf16* sV   = sK + BN*DD;
  bf16* sQ   = sV + BN*DD;
  bf16* sdO  = sQ + BM*DD;
  bf16* sPhi = sdO + BM*DD;
  bf16* sPlo = sPhi + BN*BM;
  bf16* sShi = sPlo + BN*BM;
  bf16* sSlo = sShi + BN*BM;
  float* sS  = (float*)(sSlo + BN*BM);
  float* sdP = sS + BN*BM;
  float* sL  = sdP + BN*BM;
  float* sD  = sL + BM;

  int tid=threadIdx.x, warp=tid/32, lane=tid%32;

  for (int idx=tid; idx<BN*16; idx+=NT){
     int r=idx/16, c=idx%16; int gr=kv0+r;
     int4 vk, vv;
     if(gr<S){ vk=((const int4*)(Kb+(size_t)gr*DD))[c]; vv=((const int4*)(Vb+(size_t)gr*DD))[c]; }
     else { vk=make_int4(0,0,0,0); vv=vk; }
     ((int4*)(sK+r*DD))[c]=vk;
     ((int4*)(sV+r*DD))[c]=vv;
  }

  int nt = warp%4, kbase=(warp/4)*4;
  ACC dVf[4], dKf[4];
  #pragma unroll
  for(int i=0;i<4;i++){ wmma::fill_fragment(dVf[i],0.f); wmma::fill_fragment(dKf[i],0.f);}

  __syncthreads();

  for (int q0=kv0; q0<S; q0+=BM){
     for (int idx=tid; idx<BM*16; idx+=NT){
        int r=idx/16, c=idx%16; int gr=q0+r;
        int4 vq,vo;
        if(gr<S){ vq=((const int4*)(Qb+(size_t)gr*DD))[c]; vo=((const int4*)(dOb+(size_t)gr*DD))[c]; }
        else { vq=make_int4(0,0,0,0); vo=vq; }
        ((int4*)(sQ+r*DD))[c]=vq;
        ((int4*)(sdO+r*DD))[c]=vo;
     }
     for (int m=tid;m<BM;m+=NT){ int qi=q0+m; sL[m]=(qi<S)?Lb[qi]:0.f; sD[m]=(qi<S)?Db[qi]:0.f; }
     __syncthreads();

     for (int t=warp; t<16; t+=8){
        int ntile=t/4, mtile=t%4;
        ACC cs, cp; wmma::fill_fragment(cs,0.f); wmma::fill_fragment(cp,0.f);
        #pragma unroll
        for (int ks=0;ks<8;ks++){
           AR aK, aV; BC bQ, bO;
           wmma::load_matrix_sync(aK, sK+(ntile*16)*DD+ks*16, DD);
           wmma::load_matrix_sync(bQ, sQ+(mtile*16)*DD+ks*16, DD);
           wmma::mma_sync(cs,aK,bQ,cs);
           wmma::load_matrix_sync(aV, sV+(ntile*16)*DD+ks*16, DD);
           wmma::load_matrix_sync(bO, sdO+(mtile*16)*DD+ks*16, DD);
           wmma::mma_sync(cp,aV,bO,cp);
        }
        wmma::store_matrix_sync(sS+(ntile*16)*BM+mtile*16, cs, BM, wmma::mem_row_major);
        wmma::store_matrix_sync(sdP+(ntile*16)*BM+mtile*16, cp, BM, wmma::mem_row_major);
     }
     __syncthreads();

     bool full = (q0 > kv0) && (kv0+BN<=S) && (q0+BM<=S);
     if (full){
        for (int i=tid;i<BN*BM;i+=NT){
           int m=i%BM;
           float p=__expf(scale*sS[i]-sL[m]);
           float ds=p*(sdP[i]-sD[m]);
           bf16 phi,plo,dhi,dlo;
           split_bf16(p,phi,plo); split_bf16(ds,dhi,dlo);
           sPhi[i]=phi; sPlo[i]=plo; sShi[i]=dhi; sSlo[i]=dlo;
        }
     } else {
        for (int i=tid;i<BN*BM;i+=NT){
           int n=i/BM, m=i%BM;
           int kj=kv0+n, qi=q0+m;
           float p=0.f, ds=0.f;
           if(kj<S && qi<S && kj<=qi){
              p=__expf(scale*sS[i]-sL[m]);
              ds=p*(sdP[i]-sD[m]);
           }
           bf16 phi,plo,dhi,dlo;
           split_bf16(p,phi,plo); split_bf16(ds,dhi,dlo);
           sPhi[i]=phi; sPlo[i]=plo; sShi[i]=dhi; sSlo[i]=dlo;
        }
     }
     __syncthreads();

     #pragma unroll
     for (int ms=0;ms<4;ms++){
        AR aPhi,aPlo,aShi,aSlo;
        wmma::load_matrix_sync(aPhi, sPhi+(nt*16)*BM+ms*16, BM);
        wmma::load_matrix_sync(aPlo, sPlo+(nt*16)*BM+ms*16, BM);
        wmma::load_matrix_sync(aShi, sShi+(nt*16)*BM+ms*16, BM);
        wmma::load_matrix_sync(aSlo, sSlo+(nt*16)*BM+ms*16, BM);
        #pragma unroll
        for (int kk=0;kk<4;kk++){
           int kt=kbase+kk;
           BR bO, bQ;
           wmma::load_matrix_sync(bO, sdO+(ms*16)*DD+kt*16, DD);
           wmma::mma_sync(dVf[kk],aPhi,bO,dVf[kk]);
           wmma::mma_sync(dVf[kk],aPlo,bO,dVf[kk]);
           wmma::load_matrix_sync(bQ, sQ+(ms*16)*DD+kt*16, DD);
           wmma::mma_sync(dKf[kk],aShi,bQ,dKf[kk]);
           wmma::mma_sync(dKf[kk],aSlo,bQ,dKf[kk]);
        }
     }
     __syncthreads();
  }

  bf16* stageV = sK;
  bf16* stageK = sV;
  float* scr = sS;
  __syncthreads();
  #pragma unroll
  for (int kk=0;kk<4;kk++){
     int kt=kbase+kk;
     wmma::store_matrix_sync(scr+warp*256, dVf[kk], 16, wmma::mem_row_major);
     __syncwarp();
     for (int e=lane;e<256;e+=32){
        int r=e/16,c=e%16;
        stageV[(nt*16+r)*DD + kt*16 + c] = __float2bfloat16(scr[warp*256+e]);
     }
     __syncwarp();
     wmma::store_matrix_sync(scr+warp*256, dKf[kk], 16, wmma::mem_row_major);
     __syncwarp();
     for (int e=lane;e<256;e+=32){
        int r=e/16,c=e%16;
        stageK[(nt*16+r)*DD + kt*16 + c] = __float2bfloat16(scale*scr[warp*256+e]);
     }
     __syncwarp();
  }
  __syncthreads();
  for (int idx=tid; idx<BN*16; idx+=NT){
     int r=idx/16, c=idx%16; int kj=kv0+r;
     if (kj<S){
        ((int4*)(dVb+(size_t)kj*DD))[c] = ((int4*)(stageV+r*DD))[c];
        ((int4*)(dKb+(size_t)kj*DD))[c] = ((int4*)(stageK+r*DD))[c];
     }
  }
}

__global__ void dq_kernel(
    const bf16* __restrict__ Q, const bf16* __restrict__ K, const bf16* __restrict__ V,
    const bf16* __restrict__ dO, const float* __restrict__ L, const float* __restrict__ Delta,
    bf16* __restrict__ dQ, int S, float scale)
{
  int bh = blockIdx.y;
  int q0b = blockIdx.x * BM;
  if (q0b >= S) return;

  const bf16* Qb = Q + (size_t)bh*S*DD;
  const bf16* Kb = K + (size_t)bh*S*DD;
  const bf16* Vb = V + (size_t)bh*S*DD;
  const bf16* dOb= dO+ (size_t)bh*S*DD;
  const float* Lb = L + (size_t)bh*S;
  const float* Db = Delta + (size_t)bh*S;
  bf16* dQb = dQ + (size_t)bh*S*DD;

  extern __shared__ char smem[];
  bf16* sQ  = (bf16*)smem;
  bf16* sdO = sQ + BM*DD;
  bf16* sK  = sdO + BM*DD;
  bf16* sV  = sK + BN*DD;
  bf16* sShi = sV + BN*DD;
  bf16* sSlo = sShi + BM*BN;
  float* sS  = (float*)(sSlo + BM*BN);
  float* sdP = sS + BM*BN;
  float* sL  = sdP + BM*BN;
  float* sD  = sL + BM;

  int tid=threadIdx.x, warp=tid/32, lane=tid%32;

  for (int idx=tid; idx<BM*16; idx+=NT){
     int r=idx/16, c=idx%16; int gr=q0b+r;
     int4 vq,vo;
     if(gr<S){ vq=((const int4*)(Qb+(size_t)gr*DD))[c]; vo=((const int4*)(dOb+(size_t)gr*DD))[c]; }
     else { vq=make_int4(0,0,0,0); vo=vq; }
     ((int4*)(sQ+r*DD))[c]=vq;
     ((int4*)(sdO+r*DD))[c]=vo;
  }
  for (int m=tid;m<BM;m+=NT){ int qi=q0b+m; sL[m]=(qi<S)?Lb[qi]:0.f; sD[m]=(qi<S)?Db[qi]:0.f; }

  int mt = warp%4, kbase=(warp/4)*4;
  ACC dQf[4];
  #pragma unroll
  for(int i=0;i<4;i++) wmma::fill_fragment(dQf[i],0.f);

  __syncthreads();

  int qend = q0b + BM - 1;
  for (int k0=0; k0<S && k0<=qend; k0+=BN){
     for (int idx=tid; idx<BN*16; idx+=NT){
        int r=idx/16, c=idx%16; int gr=k0+r;
        int4 vk,vv;
        if(gr<S){ vk=((const int4*)(Kb+(size_t)gr*DD))[c]; vv=((const int4*)(Vb+(size_t)gr*DD))[c]; }
        else { vk=make_int4(0,0,0,0); vv=vk; }
        ((int4*)(sK+r*DD))[c]=vk;
        ((int4*)(sV+r*DD))[c]=vv;
     }
     __syncthreads();

     for (int t=warp; t<16; t+=8){
        int mtile=t/4, ntile=t%4;
        ACC cs, cp; wmma::fill_fragment(cs,0.f); wmma::fill_fragment(cp,0.f);
        #pragma unroll
        for (int ks=0;ks<8;ks++){
           AR aQ, aO; BC bK, bV;
           wmma::load_matrix_sync(aQ, sQ+(mtile*16)*DD+ks*16, DD);
           wmma::load_matrix_sync(bK, sK+(ntile*16)*DD+ks*16, DD);
           wmma::mma_sync(cs,aQ,bK,cs);
           wmma::load_matrix_sync(aO, sdO+(mtile*16)*DD+ks*16, DD);
           wmma::load_matrix_sync(bV, sV+(ntile*16)*DD+ks*16, DD);
           wmma::mma_sync(cp,aO,bV,cp);
        }
        wmma::store_matrix_sync(sS+(mtile*16)*BN+ntile*16, cs, BN, wmma::mem_row_major);
        wmma::store_matrix_sync(sdP+(mtile*16)*BN+ntile*16, cp, BN, wmma::mem_row_major);
     }
     __syncthreads();

     bool full = (k0+BN<=q0b) && (k0+BN<=S) && (q0b+BM<=S);
     if (full){
        for (int i=tid;i<BM*BN;i+=NT){
           int m=i/BN;
           float p=__expf(scale*sS[i]-sL[m]);
           float ds=p*(sdP[i]-sD[m]);
           bf16 dhi,dlo; split_bf16(ds,dhi,dlo);
           sShi[i]=dhi; sSlo[i]=dlo;
        }
     } else {
        for (int i=tid;i<BM*BN;i+=NT){
           int m=i/BN, n=i%BN;
           int qi=q0b+m, kj=k0+n;
           float ds=0.f;
           if(kj<S && qi<S && kj<=qi){
              float p=__expf(scale*sS[i]-sL[m]);
              ds=p*(sdP[i]-sD[m]);
           }
           bf16 dhi,dlo; split_bf16(ds,dhi,dlo);
           sShi[i]=dhi; sSlo[i]=dlo;
        }
     }
     __syncthreads();

     #pragma unroll
     for (int ns=0;ns<4;ns++){
        AR aShi,aSlo;
        wmma::load_matrix_sync(aShi, sShi+(mt*16)*BN+ns*16, BN);
        wmma::load_matrix_sync(aSlo, sSlo+(mt*16)*BN+ns*16, BN);
        #pragma unroll
        for (int kk=0;kk<4;kk++){
           int kt=kbase+kk;
           BR bK;
           wmma::load_matrix_sync(bK, sK+(ns*16)*DD+kt*16, DD);
           wmma::mma_sync(dQf[kk],aShi,bK,dQf[kk]);
           wmma::mma_sync(dQf[kk],aSlo,bK,dQf[kk]);
        }
     }
     __syncthreads();
  }

  bf16* stageQ = sK;
  float* scr = sS;
  __syncthreads();
  #pragma unroll
  for (int kk=0;kk<4;kk++){
     int kt=kbase+kk;
     wmma::store_matrix_sync(scr+warp*256, dQf[kk], 16, wmma::mem_row_major);
     __syncwarp();
     for (int e=lane;e<256;e+=32){
        int r=e/16,c=e%16;
        stageQ[(mt*16+r)*DD + kt*16 + c] = __float2bfloat16(scale*scr[warp*256+e]);
     }
     __syncwarp();
  }
  __syncthreads();
  for (int idx=tid; idx<BM*16; idx+=NT){
     int r=idx/16, c=idx%16; int qi=q0b+r;
     if (qi<S) ((int4*)(dQb+(size_t)qi*DD))[c] = ((int4*)(stageQ+r*DD))[c];
  }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV)
{
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int64_t B = Q.size(0), H = Q.size(1), S = Q.size(2);
  int64_t BH = B*H;
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

  const bf16* Qp=(const bf16*)Q.data_ptr();
  const bf16* Kp=(const bf16*)K.data_ptr();
  const bf16* Vp=(const bf16*)V.data_ptr();
  const bf16* Op=(const bf16*)O.data_ptr();
  const bf16* dOp=(const bf16*)dO.data_ptr();
  const float* Lp=(const float*)L.data_ptr();
  bf16* dQp=(bf16*)dQ.data_ptr();
  bf16* dKp=(bf16*)dK.data_ptr();
  bf16* dVp=(bf16*)dV.data_ptr();

  float scale = 1.0f/sqrtf((float)DD);

  float* Delta=nullptr;
  CUDA_CHECK(cudaMalloc(&Delta, sizeof(float)*(size_t)BH*S));

  long rows = (long)BH*S;
  long dblocks = (rows + (NT/32) - 1)/(NT/32);
  delta_kernel<<<(unsigned)dblocks, NT, 0, stream>>>(Op, dOp, Delta, rows);
  CUDA_CHECK(cudaGetLastError());

  size_t smem_kv = (size_t)(4*BN*DD + 4*BN*BM)*sizeof(bf16)
                 + (size_t)(2*BN*BM + 2*BM)*sizeof(float);
  size_t smem_q  = (size_t)(4*BM*DD + 2*BM*BN)*sizeof(bf16)
                 + (size_t)(2*BM*BN + 2*BM)*sizeof(float);

  CUDA_CHECK(cudaFuncSetAttribute(dkdv_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_kv));
  CUDA_CHECK(cudaFuncSetAttribute(dq_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_q));

  dim3 grid_kv((unsigned)((S+BN-1)/BN), (unsigned)BH);
  dim3 grid_q ((unsigned)((S+BM-1)/BM), (unsigned)BH);

  dkdv_kernel<<<grid_kv, NT, smem_kv, stream>>>(Qp,Kp,Vp,dOp,Lp,Delta,dKp,dVp,(int)S,scale);
  CUDA_CHECK(cudaGetLastError());
  dq_kernel<<<grid_q, NT, smem_q, stream>>>(Qp,Kp,Vp,dOp,Lp,Delta,dQp,(int)S,scale);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaFree(Delta));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd