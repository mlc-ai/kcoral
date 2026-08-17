#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda;
using bf16 = __nv_bfloat16;

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__);} } while(0)

namespace mha_kernel {

constexpr int BM=128, BN=64, D=128, THREADS=256;

constexpr int OFF_Q =0;
constexpr int OFF_K0=OFF_Q +BM*D*2;   // 32768
constexpr int OFF_K1=OFF_K0+BN*D*2;   // 49152
constexpr int OFF_V0=OFF_K1+BN*D*2;   // 65536
constexpr int OFF_V1=OFF_V0+BN*D*2;   // 81920
constexpr int OFF_S =OFF_V1+BN*D*2;   // 98304
constexpr int OFF_P =OFF_S +BM*BN*4;  // 131072
constexpr int OFF_M =OFF_P +BM*BN*2;  // 147456
constexpr int OFF_L =OFF_M +BM*4;     // 147968
constexpr int OFF_C =OFF_L +BM*4;     // 148480
constexpr int SHMEM =OFF_C +BM*4;     // 148992

__device__ __forceinline__ void cp_async16(void* dst, const void* src){
  uint32_t s=(uint32_t)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"::"r"(s),"l"(src));
}
__device__ __forceinline__ void cp_commit(){ asm volatile("cp.async.commit_group;\n"); }
template<int N> __device__ __forceinline__ void cp_wait(){ asm volatile("cp.async.wait_group %0;\n"::"n"(N):"memory"); }

__device__ __forceinline__ void load_kv(bf16* Kdst, bf16* Vdst,
    const bf16* __restrict__ Kbase, const bf16* __restrict__ Vbase,
    int kv_start, int S, int tid){
  // BN*(D/8) = 64*16 = 1024 chunks of 16 bytes
  #pragma unroll
  for(int vi=tid; vi<BN*(D/8); vi+=THREADS){
    int row=vi/(D/8); int col=(vi%(D/8))*8;
    int gk=kv_start+row;
    if(gk<S){
      cp_async16(Kdst+row*D+col, Kbase+(long)gk*D+col);
      cp_async16(Vdst+row*D+col, Vbase+(long)gk*D+col);
    } else {
      *reinterpret_cast<float4*>(Kdst+row*D+col)=make_float4(0,0,0,0);
      *reinterpret_cast<float4*>(Vdst+row*D+col)=make_float4(0,0,0,0);
    }
  }
}

__global__ __launch_bounds__(256) void mha_kernel_fn(
    const bf16* __restrict__ Q, const bf16* __restrict__ K,
    const bf16* __restrict__ V, bf16* __restrict__ O,
    float* __restrict__ LSE, int B, int H, int S){

  extern __shared__ __align__(16) char smem[];
  bf16*  Qsh =reinterpret_cast<bf16*>(smem+OFF_Q);
  bf16*  Kb[2]={reinterpret_cast<bf16*>(smem+OFF_K0),reinterpret_cast<bf16*>(smem+OFF_K1)};
  bf16*  Vb[2]={reinterpret_cast<bf16*>(smem+OFF_V0),reinterpret_cast<bf16*>(smem+OFF_V1)};
  float* Ssh =reinterpret_cast<float*>(smem+OFF_S);
  bf16*  Psh =reinterpret_cast<bf16*>(smem+OFF_P);
  float* m_run=reinterpret_cast<float*>(smem+OFF_M);
  float* l_run=reinterpret_cast<float*>(smem+OFF_L);
  float* corr =reinterpret_cast<float*>(smem+OFF_C);

  int tid=threadIdx.x;
  int warp=tid>>5;
  int lane=tid&31;
  int groupID=lane>>2;
  int tig=lane&3;
  int qb=blockIdx.x, h=blockIdx.y, b=blockIdx.z;
  int qs=qb*BM;

  const float scale=0.08838834764831843f; // 1/sqrt(128)

  long bh=(long)(b*H+h);
  const bf16* Qbase=Q+bh*(long)S*D;
  const bf16* Kbase=K+bh*(long)S*D;
  const bf16* Vbase=V+bh*(long)S*D;
  bf16* Obase=O+bh*(long)S*D;
  float* LSEbase=LSE+bh*(long)S;

  // O accumulator in registers (8 column-tiles of 16, per warp's 16 rows)
  wmma::fragment<wmma::accumulator,16,16,16,float> Of[8];
  #pragma unroll
  for(int j=0;j<8;j++) wmma::fill_fragment(Of[j],0.0f);

  if(tid<BM){ m_run[tid]=-1e30f; l_run[tid]=0.0f; }

  // Load Q tile (persistent)
  #pragma unroll
  for(int vi=tid; vi<BM*(D/8); vi+=THREADS){
    int row=vi/(D/8); int col=(vi%(D/8))*8;
    int gq=qs+row;
    float4 v;
    if(gq<S) v=*reinterpret_cast<const float4*>(Qbase+(long)gq*D+col);
    else     v=make_float4(0,0,0,0);
    *reinterpret_cast<float4*>(Qsh+row*D+col)=v;
  }

  int q_max_row=(qs+BM-1<S-1)?(qs+BM-1):(S-1);
  int kvb_max=q_max_row/BN;

  // prologue: load block 0
  load_kv(Kb[0],Vb[0],Kbase,Vbase,0,S,tid);
  cp_commit();

  for(int kvb=0; kvb<=kvb_max; ++kvb){
    int cur=kvb&1;
    int kv_start=kvb*BN;
    bool has_next=(kvb<kvb_max);
    if(has_next){
      load_kv(Kb[(kvb+1)&1],Vb[(kvb+1)&1],Kbase,Vbase,(kvb+1)*BN,S,tid);
      cp_commit();
    }
    if(has_next) cp_wait<1>(); else cp_wait<0>();
    __syncthreads();

    bf16* Kcur=Kb[cur];
    bf16* Vcur=Vb[cur];

    // ---- QK^T  (warp handles rows [warp*16,+16), all BN cols) ----
    {
      wmma::fragment<wmma::accumulator,16,16,16,float> Sf[4];
      #pragma unroll
      for(int n=0;n<4;n++) wmma::fill_fragment(Sf[n],0.0f);
      #pragma unroll
      for(int kt=0;kt<8;kt++){
        wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> a;
        wmma::load_matrix_sync(a, Qsh+warp*16*D+kt*16, D);
        #pragma unroll
        for(int n=0;n<4;n++){
          wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::col_major> bb;
          wmma::load_matrix_sync(bb, Kcur+n*16*D+kt*16, D);
          wmma::mma_sync(Sf[n], a, bb, Sf[n]);
        }
      }
      #pragma unroll
      for(int n=0;n<4;n++)
        wmma::store_matrix_sync(Ssh+warp*16*BN+n*16, Sf[n], BN, wmma::mem_row_major);
    }
    __syncthreads();

    // ---- online softmax (one thread per row) ----
    if(tid<BM){
      int row=tid;
      int gq=qs+row;
      float* Srow=Ssh+row*BN;
      bf16* Prow=Psh+row*BN;
      float m_prev=m_run[row];
      float l_prev=l_run[row];
      float bmax=-1e30f;
      #pragma unroll
      for(int c=0;c<BN;c++){
        int gk=kv_start+c;
        float s=(gk<=gq && gk<S)? Srow[c]*scale : -1e30f;
        Srow[c]=s;
        bmax=fmaxf(bmax,s);
      }
      float m_new=fmaxf(m_prev,bmax);
      float cv=__expf(m_prev-m_new);
      float bl=0.0f;
      #pragma unroll
      for(int c=0;c<BN;c++){
        float p=__expf(Srow[c]-m_new);
        Prow[c]=__float2bfloat16(p);
        bl+=p;
      }
      l_run[row]=l_prev*cv+bl;
      m_run[row]=m_new;
      corr[row]=cv;
    }
    __syncthreads();

    // ---- rescale O accumulators by corr (per row, register) ----
    {
      int rb=warp*16;
      float c0=corr[rb+groupID];
      float c1=corr[rb+groupID+8];
      #pragma unroll
      for(int j=0;j<8;j++){
        Of[j].x[0]*=c0; Of[j].x[1]*=c0; Of[j].x[4]*=c0; Of[j].x[5]*=c0;
        Of[j].x[2]*=c1; Of[j].x[3]*=c1; Of[j].x[6]*=c1; Of[j].x[7]*=c1;
      }
    }

    // ---- P @ V (accumulate into Of) ----
    #pragma unroll
    for(int j=0;j<8;j++){
      #pragma unroll
      for(int kt=0;kt<4;kt++){
        wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> a;
        wmma::load_matrix_sync(a, Psh+warp*16*BN+kt*16, BN);
        wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::row_major> bb;
        wmma::load_matrix_sync(bb, Vcur+kt*16*D+j*16, D);
        wmma::mma_sync(Of[j], a, bb, Of[j]);
      }
    }
    __syncthreads();
  }

  // ---- write O (normalized) using accumulator layout ----
  int rb=warp*16;
  int r0=rb+groupID, r1=r0+8;
  int gq0=qs+r0, gq1=qs+r1;
  float linv0=(gq0<S)?(1.0f/l_run[r0]):0.0f;
  float linv1=(gq1<S)?(1.0f/l_run[r1]):0.0f;
  #pragma unroll
  for(int j=0;j<8;j++){
    int jc=j*16;
    int ca=jc+tig*2;
    int cb=jc+tig*2+8;
    if(gq0<S){
      Obase[(long)gq0*D+ca  ]=__float2bfloat16(Of[j].x[0]*linv0);
      Obase[(long)gq0*D+ca+1]=__float2bfloat16(Of[j].x[1]*linv0);
      Obase[(long)gq0*D+cb  ]=__float2bfloat16(Of[j].x[4]*linv0);
      Obase[(long)gq0*D+cb+1]=__float2bfloat16(Of[j].x[5]*linv0);
    }
    if(gq1<S){
      Obase[(long)gq1*D+ca  ]=__float2bfloat16(Of[j].x[2]*linv1);
      Obase[(long)gq1*D+ca+1]=__float2bfloat16(Of[j].x[3]*linv1);
      Obase[(long)gq1*D+cb  ]=__float2bfloat16(Of[j].x[6]*linv1);
      Obase[(long)gq1*D+cb+1]=__float2bfloat16(Of[j].x[7]*linv1);
    }
  }

  // ---- write LSE ----
  if(tid<BM){
    int gq=qs+tid;
    if(gq<S) LSEbase[gq]=m_run[tid]+logf(l_run[tid]);
  }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int Bv=Q.size(0), Hv=Q.size(1), Sv=Q.size(2);

  const bf16* Qp=static_cast<const bf16*>(Q.data_ptr());
  const bf16* Kp=static_cast<const bf16*>(K.data_ptr());
  const bf16* Vp=static_cast<const bf16*>(V.data_ptr());
  bf16* Op=static_cast<bf16*>(O.data_ptr());
  float* Lp=static_cast<float*>(LSE.data_ptr());

  int num_qb=(Sv+BM-1)/BM;
  dim3 grid(num_qb, Hv, Bv);
  dim3 block(THREADS);

  cudaStream_t stream=static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

  CUDA_CHECK(cudaFuncSetAttribute(mha_kernel_fn,
      cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM));

  mha_kernel_fn<<<grid, block, SHMEM, stream>>>(Qp,Kp,Vp,Op,Lp,Bv,Hv,Sv);
  CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel::run);

}  // namespace mha_kernel