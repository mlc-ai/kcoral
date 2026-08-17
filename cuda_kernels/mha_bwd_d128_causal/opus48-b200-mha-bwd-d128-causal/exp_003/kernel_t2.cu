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

__device__ __forceinline__ void mma_acc(float acc[4], const uint32_t a[4], uint32_t b0, uint32_t b1){
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
    :"+f"(acc[0]),"+f"(acc[1]),"+f"(acc[2]),"+f"(acc[3])
    :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
}

// Non-transposed fragment load : A-style, contract along COLS.
// Returns r0=TL, r1=BL, r2=TR, r3=BR ; each reg = 2 consecutive-col bf16.
__device__ __forceinline__ void ld_nt(uint32_t r[4], const __nv_bfloat16* buf, int stride, int rb, int cb, int lane){
  int row = lane>>2, col = (lane&3)*2;
  const __nv_bfloat16* p = buf + (rb+row)*stride + cb + col;
  r[0] = *reinterpret_cast<const uint32_t*>(p);
  r[1] = *reinterpret_cast<const uint32_t*>(p + 8*stride);
  r[2] = *reinterpret_cast<const uint32_t*>(p + 8);
  r[3] = *reinterpret_cast<const uint32_t*>(p + 8*stride + 8);
}

// Transposed fragment load : contract along ROWS (cb2), free along cols (fb).
// Returns r0=TL, r1=TR, r2=BL, r3=BR ; each reg = 2 consecutive-row bf16.
__device__ __forceinline__ void ld_t(uint32_t r[4], const __nv_bfloat16* buf, int stride, int cb2, int fb, int lane){
  int rr=(lane&3)*2, cc=lane>>2;
  const __nv_bfloat16* p = buf + (cb2+rr)*stride + fb + cc;
  uint16_t a,b;
  a=*reinterpret_cast<const uint16_t*>(p);              b=*reinterpret_cast<const uint16_t*>(p+stride);        r[0]=(uint32_t)a|((uint32_t)b<<16);
  a=*reinterpret_cast<const uint16_t*>(p+8);            b=*reinterpret_cast<const uint16_t*>(p+stride+8);      r[1]=(uint32_t)a|((uint32_t)b<<16);
  a=*reinterpret_cast<const uint16_t*>(p+8*stride);     b=*reinterpret_cast<const uint16_t*>(p+9*stride);      r[2]=(uint32_t)a|((uint32_t)b<<16);
  a=*reinterpret_cast<const uint16_t*>(p+8*stride+8);   b=*reinterpret_cast<const uint16_t*>(p+9*stride+8);    r[3]=(uint32_t)a|((uint32_t)b<<16);
}

__device__ __forceinline__ void load_tile(__nv_bfloat16* sTile, const __nv_bfloat16* Xbh, int S, int row0, int tid){
  const int4* src = reinterpret_cast<const int4*>(Xbh);
  int4* dst = reinterpret_cast<int4*>(sTile);
  #pragma unroll
  for (int idx = tid; idx < 1024; idx += 256) {
    int r = idx >> 4; int cg = idx & 15;
    int grow = row0 + r;
    dst[idx] = (grow < S) ? src[(size_t)grow*16 + cg] : make_int4(0,0,0,0);
  }
}

// ---- delta : D[row]=sum_e O*dO ----
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

// compute P (=softmax) and dS for a 64x64 block; store to shared as bf16.
__device__ __forceinline__ void stage_scores(
    const __nv_bfloat16* sQ, const __nv_bfloat16* sK, const __nv_bfloat16* sV, const __nv_bfloat16* sdO,
    __nv_bfloat16* sP, __nv_bfloat16* sdS, const float* sL, const float* sD,
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
    ld_nt(aq, sQ, 128, m_base, kt*16, lane);
    ld_nt(ado, sdO, 128, m_base, kt*16, lane);
    #pragma unroll
    for(int nb2=0;nb2<2;nb2++){
      int n_base = nhalf*32 + nb2*16;
      uint32_t bk[4], bv[4];
      ld_nt(bk, sK, 128, n_base, kt*16, lane);
      ld_nt(bv, sV, 128, n_base, kt*16, lane);
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
      if(storeP) sP[m*64+n] = __float2bfloat16(P);
      sdS[m*64+n] = __float2bfloat16(dS);
    }
  }
}

// ---- dK / dV kernel ----
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
  __nv_bfloat16* sQ = (__nv_bfloat16*)(smem+0);
  __nv_bfloat16* sK = (__nv_bfloat16*)(smem+16384);
  __nv_bfloat16* sV = (__nv_bfloat16*)(smem+32768);
  __nv_bfloat16* sdO= (__nv_bfloat16*)(smem+49152);
  __nv_bfloat16* sP = (__nv_bfloat16*)(smem+65536);
  __nv_bfloat16* sdS= (__nv_bfloat16*)(smem+73728);
  float* sL = (float*)(smem+81920);
  float* sD = (float*)(smem+82176);

  int tidx = threadIdx.x, warp = tidx>>5, lane = tidx&31;
  load_tile(sK, Kbh, S, j*64, tidx);
  load_tile(sV, Vbh, S, j*64, tidx);

  float dVa[8][4], dKa[8][4];
  #pragma unroll
  for(int e=0;e<8;e++)
    #pragma unroll
    for(int c=0;c<4;c++){dVa[e][c]=0.f;dKa[e][c]=0.f;}
  int nb = warp&3, eh = warp>>2;
  __syncthreads();

  for(int i=j;i<numBlk;i++){
    load_tile(sQ, Qbh, S, i*64, tidx);
    load_tile(sdO, dObh, S, i*64, tidx);
    for(int t=tidx;t<64;t+=256){int gr=i*64+t; sL[t]=gr<S?Lbh[gr]:0.f; sD[t]=gr<S?Dbh[gr]:0.f;}
    __syncthreads();

    stage_scores(sQ,sK,sV,sdO,sP,sdS,sL,sD,i,j,S,scale,true,warp,lane);
    __syncthreads();

    #pragma unroll
    for(int kt=0;kt<4;kt++){
      uint32_t apt[4], ads[4];
      ld_t(apt, sP, 64, kt*16, nb*16, lane);
      ld_t(ads, sdS, 64, kt*16, nb*16, lane);
      #pragma unroll
      for(int ep=0;ep<4;ep++){
        uint32_t bdo[4], bq[4];
        ld_t(bdo, sdO, 128, kt*16, eh*64+ep*16, lane);
        ld_t(bq,  sQ,  128, kt*16, eh*64+ep*16, lane);
        mma_acc(dVa[ep*2+0], apt, bdo[0], bdo[2]);
        mma_acc(dVa[ep*2+1], apt, bdo[1], bdo[3]);
        mma_acc(dKa[ep*2+0], ads, bq[0], bq[2]);
        mma_acc(dKa[ep*2+1], ads, bq[1], bq[3]);
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

// ---- dQ kernel ----
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
  __nv_bfloat16* sQ = (__nv_bfloat16*)(smem+0);
  __nv_bfloat16* sK = (__nv_bfloat16*)(smem+16384);
  __nv_bfloat16* sV = (__nv_bfloat16*)(smem+32768);
  __nv_bfloat16* sdO= (__nv_bfloat16*)(smem+49152);
  __nv_bfloat16* sdS= (__nv_bfloat16*)(smem+65536);
  float* sL = (float*)(smem+73728);
  float* sD = (float*)(smem+73984);

  int tidx = threadIdx.x, warp = tidx>>5, lane = tidx&31;
  load_tile(sQ, Qbh, S, i*64, tidx);
  load_tile(sdO, dObh, S, i*64, tidx);
  for(int t=tidx;t<64;t+=256){int gr=i*64+t; sL[t]=gr<S?Lbh[gr]:0.f; sD[t]=gr<S?Dbh[gr]:0.f;}

  float dQa[8][4];
  #pragma unroll
  for(int e=0;e<8;e++)
    #pragma unroll
    for(int c=0;c<4;c++) dQa[e][c]=0.f;
  int mb = warp&3, eh = warp>>2;
  __syncthreads();

  for(int j=0;j<=i;j++){
    load_tile(sK, Kbh, S, j*64, tidx);
    load_tile(sV, Vbh, S, j*64, tidx);
    __syncthreads();

    stage_scores(sQ,sK,sV,sdO,nullptr,sdS,sL,sD,i,j,S,scale,false,warp,lane);
    __syncthreads();

    #pragma unroll
    for(int kt=0;kt<4;kt++){
      uint32_t ads[4];
      ld_nt(ads, sdS, 64, mb*16, kt*16, lane);
      #pragma unroll
      for(int ep=0;ep<4;ep++){
        uint32_t bk[4];
        ld_t(bk, sK, 128, kt*16, eh*64+ep*16, lane);
        mma_acc(dQa[ep*2+0], ads, bk[0], bk[2]);
        mma_acc(dQa[ep*2+1], ads, bk[1], bk[3]);
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

  int smem_dkv = 82432;
  int smem_dq  = 74240;
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