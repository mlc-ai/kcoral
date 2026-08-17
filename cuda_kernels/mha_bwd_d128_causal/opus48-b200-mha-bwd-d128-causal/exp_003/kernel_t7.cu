#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
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

namespace mha_bwd {

#define LDA 136
#define LDP 72

__device__ __forceinline__ void cpasync16(void* s, const void* g){
  unsigned sa=(unsigned)__cvta_generic_to_shared(s);
  asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n" :: "r"(sa),"l"(g));
}
__device__ __forceinline__ void cp_commit(){ asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_wait_all(){ asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }

__device__ __forceinline__ void mma_acc(float acc[4], const uint32_t a[4], uint32_t b0, uint32_t b1){
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
    :"+f"(acc[0]),"+f"(acc[1]),"+f"(acc[2]),"+f"(acc[3])
    :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
}

__device__ __forceinline__ void ld_nt(uint32_t r[4], const __nv_bfloat16* buf, int stride, int rb, int cb, int lane){
  int mtx=lane>>3, rr=lane&7;
  int ro=(mtx&1)?8:0, co=(mtx>=2)?8:0;
  uint32_t a=(uint32_t)__cvta_generic_to_shared(buf+(rb+ro+rr)*stride+(cb+co));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
    :"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]):"r"(a));
}
__device__ __forceinline__ void ld_t(uint32_t r[4], const __nv_bfloat16* buf, int stride, int cb2, int fb, int lane){
  int mtx=lane>>3, rr=lane&7;
  int co=(mtx>=2)?8:0, fo=(mtx&1)?8:0;
  uint32_t a=(uint32_t)__cvta_generic_to_shared(buf+(cb2+co+rr)*stride+(fb+fo));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];"
    :"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]):"r"(a));
}

__device__ __forceinline__ void load_tile_async(__nv_bfloat16* sTile, const __nv_bfloat16* Xbh, int S, int row0, int tid){
  const int4* src = reinterpret_cast<const int4*>(Xbh);
  #pragma unroll
  for(int idx=tid; idx<1024; idx+=256){
    int r=idx>>4, cg=idx&15;
    int grow=row0+r;
    __nv_bfloat16* dstp = sTile + r*LDA + cg*8;
    if(grow<S) cpasync16(dstp, &src[(size_t)grow*16+cg]);
    else *reinterpret_cast<int4*>(dstp)=make_int4(0,0,0,0);
  }
}

__global__ void delta_kernel(const __nv_bfloat16* O, const __nv_bfloat16* dO, float* Delta, int R, int HD){
  int warp = (blockIdx.x*blockDim.x + threadIdx.x)>>5;
  int lane = threadIdx.x&31;
  if(warp>=R) return;
  const __nv_bfloat16* Op = O + (size_t)warp*HD;
  const __nv_bfloat16* dOp = dO + (size_t)warp*HD;
  float s=0.f;
  for(int k=lane;k<HD;k+=32) s += __bfloat162float(Op[k])*__bfloat162float(dOp[k]);
  #pragma unroll
  for(int off=16;off>0;off>>=1) s += __shfl_down_sync(0xffffffff,s,off);
  if(lane==0) Delta[warp]=s;
}

__device__ __forceinline__ void stage_scores(
    const __nv_bfloat16* sQ, const __nv_bfloat16* sK, const __nv_bfloat16* sV, const __nv_bfloat16* sdO,
    __nv_bfloat16* sP, __nv_bfloat16* sdS_hi, __nv_bfloat16* sdS_lo,
    const float* sL, const float* sD,
    int i, int j, int S, float scale, bool storeP, int warp, int lane){
  int mt = warp>>1, nhalf = warp&1;
  int m_base = mt*16;
  float acc_s[4][4], acc_dp[4][4];
  #pragma unroll
  for(int t=0;t<4;t++)
    #pragma unroll
    for(int u=0;u<4;u++){acc_s[t][u]=0.f;acc_dp[t][u]=0.f;}
  #pragma unroll
  for(int kt=0;kt<8;kt++){
    uint32_t aq[4], ado[4];
    ld_nt(aq, sQ, LDA, m_base, kt*16, lane);
    ld_nt(ado, sdO, LDA, m_base, kt*16, lane);
    #pragma unroll
    for(int nb2=0;nb2<2;nb2++){
      int n_base = nhalf*32 + nb2*16;
      uint32_t bk[4], bv[4];
      ld_nt(bk, sK, LDA, n_base, kt*16, lane);
      ld_nt(bv, sV, LDA, n_base, kt*16, lane);
      mma_acc(acc_s[nb2*2+0], aq, bk[0], bk[2]);
      mma_acc(acc_s[nb2*2+1], aq, bk[1], bk[3]);
      mma_acc(acc_dp[nb2*2+0], ado, bv[0], bv[2]);
      mma_acc(acc_dp[nb2*2+1], ado, bv[1], bv[3]);
    }
  }
  int grp = lane>>2, tid = lane&3;
  #pragma unroll
  for(int ntl=0;ntl<4;ntl++){
    int ncol = nhalf*32 + ntl*8;
    int mrow[4] = {m_base+grp, m_base+grp, m_base+grp+8, m_base+grp+8};
    int nn[4]   = {ncol+tid*2, ncol+tid*2+1, ncol+tid*2, ncol+tid*2+1};
    #pragma unroll
    for(int c=0;c<4;c++){
      int m = mrow[c], n = nn[c];
      int qi = i*64+m, kj = j*64+n;
      float sc = acc_s[ntl][c]*scale;
      float P = 0.f;
      if(qi<S && kj<S && kj<=qi) P = __expf(sc - sL[m]);
      float dS = P*(acc_dp[ntl][c] - sD[m]);
      int idx = m*LDP+n;
      if(storeP) sP[idx]=__float2bfloat16(P);
      __nv_bfloat16 dhi=__float2bfloat16(dS);
      __nv_bfloat16 dlo=__float2bfloat16(dS - __bfloat162float(dhi));
      sdS_hi[idx]=dhi; sdS_lo[idx]=dlo;
    }
  }
}

__global__ __launch_bounds__(256) void bwd_dkv_kernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* dO, const float* L, const float* Delta,
    __nv_bfloat16* dK, __nv_bfloat16* dV, int B, int H, int S, float scale){
  int j = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  int numBlk = (S+63)/64;
  size_t bh = (size_t)(b*H+h)*S*128;
  size_t bhL = (size_t)(b*H+h)*S;
  const __nv_bfloat16* Qbh=Q+bh; const __nv_bfloat16* Kbh=K+bh;
  const __nv_bfloat16* Vbh=V+bh; const __nv_bfloat16* dObh=dO+bh;
  const float* Lbh=L+bhL; const float* Dbh=Delta+bhL;
  __nv_bfloat16* dKbh=dK+bh; __nv_bfloat16* dVbh=dV+bh;

  extern __shared__ char smem[];
  __nv_bfloat16* sK    = (__nv_bfloat16*)(smem+0);
  __nv_bfloat16* sV    = (__nv_bfloat16*)(smem+17408);
  __nv_bfloat16* sQ0   = (__nv_bfloat16*)(smem+34816);
  __nv_bfloat16* sdO0  = (__nv_bfloat16*)(smem+52224);
  __nv_bfloat16* sQ1   = (__nv_bfloat16*)(smem+69632);
  __nv_bfloat16* sdO1  = (__nv_bfloat16*)(smem+87040);
  __nv_bfloat16* sP    = (__nv_bfloat16*)(smem+104448);
  __nv_bfloat16* sdS_hi= (__nv_bfloat16*)(smem+113664);
  __nv_bfloat16* sdS_lo= (__nv_bfloat16*)(smem+122880);
  float* sL = (float*)(smem+132096);
  float* sD = (float*)(smem+132352);
  __nv_bfloat16* Qbuf[2] = {sQ0, sQ1};
  __nv_bfloat16* Obuf[2] = {sdO0, sdO1};

  int tidx = threadIdx.x, warp = tidx>>5, lane = tidx&31;

  load_tile_async(sK, Kbh, S, j*64, tidx);
  load_tile_async(sV, Vbh, S, j*64, tidx);
  cp_commit();
  load_tile_async(Qbuf[0], Qbh, S, j*64, tidx);
  load_tile_async(Obuf[0], dObh, S, j*64, tidx);
  cp_commit();

  float dVa[8][4], dKa[8][4];
  #pragma unroll
  for(int e=0;e<8;e++)
    #pragma unroll
    for(int c=0;c<4;c++){dVa[e][c]=0.f;dKa[e][c]=0.f;}
  int nb = warp&3, eh = warp>>2;

  for(int i=j;i<numBlk;i++){
    int bb=(i-j)&1;
    __nv_bfloat16* sQb=Qbuf[bb]; __nv_bfloat16* sOb=Obuf[bb];
    cp_wait_all(); __syncthreads();
    if(i+1<numBlk){
      load_tile_async(Qbuf[bb^1], Qbh, S, (i+1)*64, tidx);
      load_tile_async(Obuf[bb^1], dObh, S, (i+1)*64, tidx);
      cp_commit();
    }
    for(int t=tidx;t<64;t+=256){int gr=i*64+t; sL[t]=gr<S?Lbh[gr]:0.f; sD[t]=gr<S?Dbh[gr]:0.f;}
    __syncthreads();

    stage_scores(sQb,sK,sV,sOb,sP,sdS_hi,sdS_lo,sL,sD,i,j,S,scale,true,warp,lane);
    __syncthreads();

    #pragma unroll
    for(int kt=0;kt<4;kt++){
      uint32_t apt[4], adsh[4], adsl[4];
      ld_t(apt, sP, LDP, kt*16, nb*16, lane);
      ld_t(adsh, sdS_hi, LDP, kt*16, nb*16, lane);
      ld_t(adsl, sdS_lo, LDP, kt*16, nb*16, lane);
      #pragma unroll
      for(int ep=0;ep<4;ep++){
        uint32_t bdo[4], bq[4];
        ld_t(bdo, sOb, LDA, kt*16, eh*64+ep*16, lane);
        ld_t(bq,  sQb, LDA, kt*16, eh*64+ep*16, lane);
        mma_acc(dVa[ep*2+0], apt, bdo[0], bdo[2]);
        mma_acc(dVa[ep*2+1], apt, bdo[1], bdo[3]);
        mma_acc(dKa[ep*2+0], adsh, bq[0], bq[2]);
        mma_acc(dKa[ep*2+0], adsl, bq[0], bq[2]);
        mma_acc(dKa[ep*2+1], adsh, bq[1], bq[3]);
        mma_acc(dKa[ep*2+1], adsl, bq[1], bq[3]);
      }
    }
    __syncthreads();
  }

  int grp = lane>>2, tid = lane&3;
  #pragma unroll
  for(int et=0;et<8;et++){
    int ebase = eh*64 + et*8;
    int n0 = nb*16+grp, n1 = nb*16+grp+8;
    int e0 = ebase+tid*2, e1 = ebase+tid*2+1;
    int kj0 = j*64+n0, kj1 = j*64+n1;
    if(kj0<S){
      dVbh[(size_t)kj0*128+e0]=__float2bfloat16(dVa[et][0]);
      dVbh[(size_t)kj0*128+e1]=__float2bfloat16(dVa[et][1]);
      dKbh[(size_t)kj0*128+e0]=__float2bfloat16(dKa[et][0]*scale);
      dKbh[(size_t)kj0*128+e1]=__float2bfloat16(dKa[et][1]*scale);
    }
    if(kj1<S){
      dVbh[(size_t)kj1*128+e0]=__float2bfloat16(dVa[et][2]);
      dVbh[(size_t)kj1*128+e1]=__float2bfloat16(dVa[et][3]);
      dKbh[(size_t)kj1*128+e0]=__float2bfloat16(dKa[et][2]*scale);
      dKbh[(size_t)kj1*128+e1]=__float2bfloat16(dKa[et][3]*scale);
    }
  }
}

__global__ __launch_bounds__(256) void bwd_dq_kernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* dO, const float* L, const float* Delta,
    __nv_bfloat16* dQ, int B, int H, int S, float scale){
  int i = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  size_t bh = (size_t)(b*H+h)*S*128;
  size_t bhL = (size_t)(b*H+h)*S;
  const __nv_bfloat16* Qbh=Q+bh; const __nv_bfloat16* Kbh=K+bh;
  const __nv_bfloat16* Vbh=V+bh; const __nv_bfloat16* dObh=dO+bh;
  const float* Lbh=L+bhL; const float* Dbh=Delta+bhL;
  __nv_bfloat16* dQbh=dQ+bh;

  extern __shared__ char smem[];
  __nv_bfloat16* sQ    = (__nv_bfloat16*)(smem+0);
  __nv_bfloat16* sdO   = (__nv_bfloat16*)(smem+17408);
  __nv_bfloat16* sK0   = (__nv_bfloat16*)(smem+34816);
  __nv_bfloat16* sV0   = (__nv_bfloat16*)(smem+52224);
  __nv_bfloat16* sK1   = (__nv_bfloat16*)(smem+69632);
  __nv_bfloat16* sV1   = (__nv_bfloat16*)(smem+87040);
  __nv_bfloat16* sdS_hi= (__nv_bfloat16*)(smem+104448);
  __nv_bfloat16* sdS_lo= (__nv_bfloat16*)(smem+113664);
  float* sL = (float*)(smem+122880);
  float* sD = (float*)(smem+123136);
  __nv_bfloat16* Kbuf[2] = {sK0, sK1};
  __nv_bfloat16* Vbuf[2] = {sV0, sV1};

  int tidx = threadIdx.x, warp = tidx>>5, lane = tidx&31;

  load_tile_async(sQ, Qbh, S, i*64, tidx);
  load_tile_async(sdO, dObh, S, i*64, tidx);
  cp_commit();
  load_tile_async(Kbuf[0], Kbh, S, 0, tidx);
  load_tile_async(Vbuf[0], Vbh, S, 0, tidx);
  cp_commit();
  for(int t=tidx;t<64;t+=256){int gr=i*64+t; sL[t]=gr<S?Lbh[gr]:0.f; sD[t]=gr<S?Dbh[gr]:0.f;}

  float dQa[8][4];
  #pragma unroll
  for(int e=0;e<8;e++)
    #pragma unroll
    for(int c=0;c<4;c++) dQa[e][c]=0.f;
  int mb = warp&3, eh = warp>>2;

  for(int j=0;j<=i;j++){
    int bb=j&1;
    __nv_bfloat16* sKb=Kbuf[bb]; __nv_bfloat16* sVb=Vbuf[bb];
    cp_wait_all(); __syncthreads();
    if(j+1<=i){
      load_tile_async(Kbuf[bb^1], Kbh, S, (j+1)*64, tidx);
      load_tile_async(Vbuf[bb^1], Vbh, S, (j+1)*64, tidx);
      cp_commit();
    }

    stage_scores(sQ,sKb,sVb,sdO,nullptr,sdS_hi,sdS_lo,sL,sD,i,j,S,scale,false,warp,lane);
    __syncthreads();

    #pragma unroll
    for(int kt=0;kt<4;kt++){
      uint32_t adsh[4], adsl[4];
      ld_nt(adsh, sdS_hi, LDP, mb*16, kt*16, lane);
      ld_nt(adsl, sdS_lo, LDP, mb*16, kt*16, lane);
      #pragma unroll
      for(int ep=0;ep<4;ep++){
        uint32_t bk[4];
        ld_t(bk, sKb, LDA, kt*16, eh*64+ep*16, lane);
        mma_acc(dQa[ep*2+0], adsh, bk[0], bk[2]);
        mma_acc(dQa[ep*2+0], adsl, bk[0], bk[2]);
        mma_acc(dQa[ep*2+1], adsh, bk[1], bk[3]);
        mma_acc(dQa[ep*2+1], adsl, bk[1], bk[3]);
      }
    }
    __syncthreads();
  }

  int grp = lane>>2, tid = lane&3;
  #pragma unroll
  for(int et=0;et<8;et++){
    int ebase = eh*64 + et*8;
    int m0 = mb*16+grp, m1 = mb*16+grp+8;
    int e0 = ebase+tid*2, e1 = ebase+tid*2+1;
    int qi0 = i*64+m0, qi1 = i*64+m1;
    if(qi0<S){
      dQbh[(size_t)qi0*128+e0]=__float2bfloat16(dQa[et][0]*scale);
      dQbh[(size_t)qi0*128+e1]=__float2bfloat16(dQa[et][1]*scale);
    }
    if(qi1<S){
      dQbh[(size_t)qi1*128+e0]=__float2bfloat16(dQa[et][2]*scale);
      dQbh[(size_t)qi1*128+e1]=__float2bfloat16(dQa[et][3]*scale);
    }
  }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int B=(int)Q.size(0), H=(int)Q.size(1), S=(int)Q.size(2), d=(int)Q.size(3);
  float scale = 1.0f/sqrtf((float)d);

  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

  const __nv_bfloat16* Qp=static_cast<const __nv_bfloat16*>(Q.data_ptr());
  const __nv_bfloat16* Kp=static_cast<const __nv_bfloat16*>(K.data_ptr());
  const __nv_bfloat16* Vp=static_cast<const __nv_bfloat16*>(V.data_ptr());
  const __nv_bfloat16* Op=static_cast<const __nv_bfloat16*>(O.data_ptr());
  const __nv_bfloat16* dOp=static_cast<const __nv_bfloat16*>(dO.data_ptr());
  const float* Lp=static_cast<const float*>(L.data_ptr());
  __nv_bfloat16* dQp=static_cast<__nv_bfloat16*>(dQ.data_ptr());
  __nv_bfloat16* dKp=static_cast<__nv_bfloat16*>(dK.data_ptr());
  __nv_bfloat16* dVp=static_cast<__nv_bfloat16*>(dV.data_ptr());

  float* Delta=nullptr;
  size_t dbytes=(size_t)B*H*S*sizeof(float);
  CUDA_CHECK(cudaMallocAsync((void**)&Delta, dbytes, stream));

  int R=B*H*S;
  int dblocks=(R+7)/8;
  delta_kernel<<<dblocks,256,0,stream>>>(Op,dOp,Delta,R,128);
  CUDA_CHECK(cudaGetLastError());

  int smem_dkv = 132608;
  int smem_dq  = 123392;
  static bool init=false;
  if(!init){
    CUDA_CHECK(cudaFuncSetAttribute(bwd_dkv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_dkv));
    CUDA_CHECK(cudaFuncSetAttribute(bwd_dq_kernel,  cudaFuncAttributeMaxDynamicSharedMemorySize, smem_dq));
    init=true;
  }

  int numBlk=(S+63)/64;
  dim3 grid(numBlk,H,B);

  bwd_dkv_kernel<<<grid,256,smem_dkv,stream>>>(Qp,Kp,Vp,dOp,Lp,Delta,dKp,dVp,B,H,S,scale);
  CUDA_CHECK(cudaGetLastError());

  bwd_dq_kernel<<<grid,256,smem_dq,stream>>>(Qp,Kp,Vp,dOp,Lp,Delta,dQp,B,H,S,scale);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaFreeAsync(Delta,stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd