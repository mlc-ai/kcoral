#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__); exit(1);} }while(0)

namespace mha_kernel_ns {

#define HD 128
#define BM 128
#define BN 64
#define SD 136
#define I4ROW 17

__device__ __forceinline__ uint32_t saddr(const void* p){ return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int sz){
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"::"r"(dst),"l"(src),"r"(sz):"memory");
}
__device__ __forceinline__ void cp_commit(){ asm volatile("cp.async.commit_group;\n":::"memory"); }
template<int N> __device__ __forceinline__ void cp_wait(){ asm volatile("cp.async.wait_group %0;\n"::"n"(N):"memory"); }

__device__ __forceinline__ void ldm_x4(uint32_t&r0,uint32_t&r1,uint32_t&r2,uint32_t&r3,uint32_t a){
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
    :"=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3):"r"(a));
}
__device__ __forceinline__ void ldm_x2(uint32_t&r0,uint32_t&r1,uint32_t a){
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n":"=r"(r0),"=r"(r1):"r"(a));
}
__device__ __forceinline__ void ldm_x2t(uint32_t&r0,uint32_t&r1,uint32_t a){
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n":"=r"(r0),"=r"(r1):"r"(a));
}
__device__ __forceinline__ void mma_m16n8k16(float&d0,float&d1,float&d2,float&d3,
   uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1){
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
   :"+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
__device__ __forceinline__ uint32_t pack2(float x,float y){
  __nv_bfloat16 a=__float2bfloat16(x),b=__float2bfloat16(y);
  return (uint32_t)*reinterpret_cast<uint16_t*>(&a) | ((uint32_t)*reinterpret_cast<uint16_t*>(&b)<<16);
}
__device__ __forceinline__ float gmax(float v){ v=fmaxf(v,__shfl_xor_sync(0xffffffffu,v,1)); v=fmaxf(v,__shfl_xor_sync(0xffffffffu,v,2)); return v;}
__device__ __forceinline__ float gsum(float v){ v+=__shfl_xor_sync(0xffffffffu,v,1); v+=__shfl_xor_sync(0xffffffffu,v,2); return v;}

__device__ __forceinline__ void qk_gemm(float S[8][4], const __nv_bfloat16* Ksbuf,
        const __nv_bfloat16* Qs, int warp_row, int lane){
  uint32_t qf[8][4];
  int rr=(lane&7)+((lane>>3)&1)*8;
  #pragma unroll
  for(int kk=0;kk<8;kk++){
    int cc=16*kk+(((lane>>3)>>1))*8;
    ldm_x4(qf[kk][0],qf[kk][1],qf[kk][2],qf[kk][3], saddr(&Qs[(warp_row+rr)*SD+cc]));
  }
  int g8=lane&7, hi=(lane>>3)&1;
  #pragma unroll
  for(int nt=0;nt<8;nt++){
    float s0=0,s1=0,s2=0,s3=0;
    int key=8*nt+g8;
    uint32_t kb0[8],kb1[8];
    #pragma unroll
    for(int kk=0;kk<8;kk++){
      int dcol=16*kk+hi*8;
      ldm_x2(kb0[kk],kb1[kk], saddr(&Ksbuf[key*SD+dcol]));
    }
    #pragma unroll
    for(int kk=0;kk<8;kk++)
      mma_m16n8k16(s0,s1,s2,s3, qf[kk][0],qf[kk][1],qf[kk][2],qf[kk][3], kb0[kk],kb1[kk]);
    S[nt][0]=s0;S[nt][1]=s1;S[nt][2]=s2;S[nt][3]=s3;
  }
}

__device__ __forceinline__ void pv_gemm(float O[16][4], const __nv_bfloat16* Vsbuf,
        const uint32_t Af[4][4], int lane){
  int g8=lane&7, hi=(lane>>3)&1;
  #pragma unroll
  for(int nt=0;nt<16;nt++){
    float o0=O[nt][0],o1=O[nt][1],o2=O[nt][2],o3=O[nt][3];
    uint32_t vb0[4],vb1[4];
    #pragma unroll
    for(int kk=0;kk<4;kk++){
      int key=16*kk+hi*8+g8;
      ldm_x2t(vb0[kk],vb1[kk], saddr(&Vsbuf[key*SD + 8*nt]));
    }
    #pragma unroll
    for(int kk=0;kk<4;kk++)
      mma_m16n8k16(o0,o1,o2,o3, Af[kk][0],Af[kk][1],Af[kk][2],Af[kk][3], vb0[kk],vb1[kk]);
    O[nt][0]=o0;O[nt][1]=o1;O[nt][2]=o2;O[nt][3]=o3;
  }
}

extern __shared__ __align__(16) char smem_raw[];

__global__ __launch_bounds__(256,2) void mha_kernel(
    const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V, __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE, int S, float scale){
  int bh=blockIdx.y;
  int qbase=blockIdx.x*BM;
  const __nv_bfloat16* Qp=Q+(size_t)bh*S*HD;
  const __nv_bfloat16* Kp=K+(size_t)bh*S*HD;
  const __nv_bfloat16* Vp=V+(size_t)bh*S*HD;
  __nv_bfloat16* Op=O+(size_t)bh*S*HD;
  float* LSEp=LSE+(size_t)bh*S;

  __nv_bfloat16* Qs=(__nv_bfloat16*)smem_raw;
  __nv_bfloat16* Ks=Qs+BM*SD;
  __nv_bfloat16* Vs=Ks+2*BN*SD;

  int tid=threadIdx.x, warp=tid>>5, lane=tid&31;
  int group=lane>>2, thr=lane&3;
  int warp_row=warp*16;
  int row_a=warp_row+group, row_b=warp_row+group+8;

  const int4* Qp4=(const int4*)Qp;
  int4* Qs4=(int4*)Qs;
  for(int idx=tid; idx<BM*16; idx+=256){
    int r=idx>>4,q=idx&15; int gr=qbase+r;
    int4 v = gr<S ? Qp4[(size_t)gr*16+q] : make_int4(0,0,0,0);
    Qs4[r*I4ROW+q]=v;
  }
  __syncthreads();

  float m_a=-INFINITY,m_b=-INFINITY,l_a=0,l_b=0;
  float Ofrag[16][4];
  #pragma unroll
  for(int i=0;i<16;i++){Ofrag[i][0]=Ofrag[i][1]=Ofrag[i][2]=Ofrag[i][3]=0;}

  int T=(S+BN-1)/BN;
  int4* Ks4=(int4*)Ks; int4* Vs4=(int4*)Vs;
  const int4* Kp4=(const int4*)Kp; const int4* Vp4=(const int4*)Vp;

  #define LOAD_TILE(kv,buf) do{ \
    for(int idx=tid; idx<BN*16; idx+=256){ \
      int r=idx>>4,q=idx&15; int gr=(kv)*BN+r; int sz=gr<S?16:0; \
      cp_async16(saddr(&Ks4[(buf)*BN*I4ROW + r*I4ROW+q]), &Kp4[(size_t)gr*16+q], sz); \
      cp_async16(saddr(&Vs4[(buf)*BN*I4ROW + r*I4ROW+q]), &Vp4[(size_t)gr*16+q], sz); \
    } }while(0)

  float Scur[8][4], Snext[8][4];

  // prologue
  LOAD_TILE(0,0); cp_commit();
  cp_wait<0>(); __syncthreads();
  qk_gemm(Scur, &Ks[0], Qs, warp_row, lane);
  if(T>1){ LOAD_TILE(1,1); cp_commit(); }

  for(int t=0;t<T;t++){
    int b=t&1;
    int kvbase=t*BN;

    // overlap: issue QK for next tile before softmax
    if(t+1<T){
      cp_wait<0>(); __syncthreads();
      qk_gemm(Snext, &Ks[((t+1)&1)*BN*SD], Qs, warp_row, lane);
    }

    // softmax on Scur (tile t)
    bool need_mask = (kvbase+BN > S);
    if(need_mask){
      #pragma unroll
      for(int nt=0;nt<8;nt++){
        int keycol=kvbase+nt*8+thr*2;
        #pragma unroll
        for(int e=0;e<4;e++){
          float val=Scur[nt][e]*scale;
          if(keycol+(e&1)>=S) val=-INFINITY;
          Scur[nt][e]=val;
        }
      }
    } else {
      #pragma unroll
      for(int nt=0;nt<8;nt++){
        Scur[nt][0]*=scale;Scur[nt][1]*=scale;Scur[nt][2]*=scale;Scur[nt][3]*=scale;
      }
    }
    float mt_a=-INFINITY,mt_b=-INFINITY;
    #pragma unroll
    for(int nt=0;nt<8;nt++){ mt_a=fmaxf(mt_a,fmaxf(Scur[nt][0],Scur[nt][1])); mt_b=fmaxf(mt_b,fmaxf(Scur[nt][2],Scur[nt][3]));}
    mt_a=gmax(mt_a);mt_b=gmax(mt_b);
    float nm_a=fmaxf(m_a,mt_a),nm_b=fmaxf(m_b,mt_b);
    float ca=__expf(m_a-nm_a),cb=__expf(m_b-nm_b);
    float sa=0,sb=0;
    #pragma unroll
    for(int nt=0;nt<8;nt++){
      float p0=__expf(Scur[nt][0]-nm_a),p1=__expf(Scur[nt][1]-nm_a);
      float p2=__expf(Scur[nt][2]-nm_b),p3=__expf(Scur[nt][3]-nm_b);
      Scur[nt][0]=p0;Scur[nt][1]=p1;Scur[nt][2]=p2;Scur[nt][3]=p3;
      sa+=p0+p1;sb+=p2+p3;
    }
    sa=gsum(sa);sb=gsum(sb);
    l_a=l_a*ca+sa;l_b=l_b*cb+sb;m_a=nm_a;m_b=nm_b;
    #pragma unroll
    for(int nt=0;nt<16;nt++){Ofrag[nt][0]*=ca;Ofrag[nt][1]*=ca;Ofrag[nt][2]*=cb;Ofrag[nt][3]*=cb;}

    uint32_t Af[4][4];
    #pragma unroll
    for(int kk=0;kk<4;kk++){
      Af[kk][0]=pack2(Scur[2*kk][0],Scur[2*kk][1]);
      Af[kk][1]=pack2(Scur[2*kk][2],Scur[2*kk][3]);
      Af[kk][2]=pack2(Scur[2*kk+1][0],Scur[2*kk+1][1]);
      Af[kk][3]=pack2(Scur[2*kk+1][2],Scur[2*kk+1][3]);
    }

    pv_gemm(Ofrag, &Vs[b*BN*SD], Af, lane);

    __syncthreads();
    if(t+2<T){ LOAD_TILE(t+2,b); cp_commit(); }

    // Snext -> Scur
    if(t+1<T){
      #pragma unroll
      for(int nt=0;nt<8;nt++){ Scur[nt][0]=Snext[nt][0];Scur[nt][1]=Snext[nt][1];Scur[nt][2]=Snext[nt][2];Scur[nt][3]=Snext[nt][3];}
    }
  }

  float ia=1.f/l_a, ib=1.f/l_b;
  int gr_a=qbase+row_a, gr_b=qbase+row_b;
  #pragma unroll
  for(int nt=0;nt<16;nt++){
    int col=8*nt+thr*2;
    if(gr_a<S){ Op[(size_t)gr_a*HD+col]=__float2bfloat16(Ofrag[nt][0]*ia); Op[(size_t)gr_a*HD+col+1]=__float2bfloat16(Ofrag[nt][1]*ia);}
    if(gr_b<S){ Op[(size_t)gr_b*HD+col]=__float2bfloat16(Ofrag[nt][2]*ib); Op[(size_t)gr_b*HD+col+1]=__float2bfloat16(Ofrag[nt][3]*ib);}
  }
  if(thr==0){
    if(gr_a<S) LSEp[gr_a]=m_a+logf(l_a);
    if(gr_b<S) LSEp[gr_b]=m_b+logf(l_b);
  }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int64_t B=Q.size(0),H=Q.size(1),S=Q.size(2);
  const __nv_bfloat16* Qp=(const __nv_bfloat16*)Q.data_ptr();
  const __nv_bfloat16* Kp=(const __nv_bfloat16*)K.data_ptr();
  const __nv_bfloat16* Vp=(const __nv_bfloat16*)V.data_ptr();
  __nv_bfloat16* Op=(__nv_bfloat16*)O.data_ptr();
  float* LSEp=(float*)LSE.data_ptr();

  int num_qtiles=(int)((S+BM-1)/BM);
  dim3 grid(num_qtiles, (unsigned)(B*H));
  dim3 block(256);
  size_t smem=(size_t)(BM + 2*BN + 2*BN)*SD*sizeof(__nv_bfloat16);
  cudaFuncSetAttribute(mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  float scale=1.0f/sqrtf((float)HD);
  cudaStream_t stream=(cudaStream_t)TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id);
  mha_kernel<<<grid,block,smem,stream>>>(Qp,Kp,Vp,Op,LSEp,(int)S,scale);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel_ns::run);

} // namespace mha_kernel_ns