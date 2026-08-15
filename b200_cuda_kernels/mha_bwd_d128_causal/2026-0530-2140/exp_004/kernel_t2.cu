#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); exit(1);} } while(0)

namespace mha_bwd {
using tvm::ffi::TensorView;

constexpr int D=128;
constexpr int BLK=64;
constexpr int NT=128; // 4 warps

__device__ __forceinline__ uint32_t ldu32(const __nv_bfloat16* p){ return *reinterpret_cast<const uint32_t*>(p); }
__device__ __forceinline__ uint32_t pk2(const __nv_bfloat16* p0,const __nv_bfloat16* p1){
  uint16_t a=*reinterpret_cast<const uint16_t*>(p0);
  uint16_t b=*reinterpret_cast<const uint16_t*>(p1);
  return (uint32_t)a | ((uint32_t)b<<16);
}
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, const uint32_t* b){
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
    : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}

__global__ void compute_D_kernel(const __nv_bfloat16* __restrict__ dO,
                                 const __nv_bfloat16* __restrict__ O,
                                 float* __restrict__ Dout, size_t nrows){
    size_t row=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    size_t stride=(size_t)gridDim.x*blockDim.x;
    for(; row<nrows; row+=stride){
        const __nv_bfloat16* dop=dO+row*D; const __nv_bfloat16* op=O+row*D;
        float acc=0.f;
        #pragma unroll
        for(int e=0;e<D;e+=8){
            uint4 a=*reinterpret_cast<const uint4*>(dop+e);
            uint4 c=*reinterpret_cast<const uint4*>(op+e);
            __nv_bfloat16* pa=reinterpret_cast<__nv_bfloat16*>(&a);
            __nv_bfloat16* pc=reinterpret_cast<__nv_bfloat16*>(&c);
            #pragma unroll
            for(int k=0;k<8;k++) acc+=__bfloat162float(pa[k])*__bfloat162float(pc[k]);
        }
        Dout[row]=acc;
    }
}

__device__ __forceinline__ void load_tile(const __nv_bfloat16* gbh,int row0,int S,__nv_bfloat16* sm,int tid){
    #pragma unroll
    for(int v=tid; v<BLK*16; v+=NT){
        int r=v>>4; int c=(v&15)<<3;
        int gr=row0+r;
        uint4 val=make_uint4(0,0,0,0);
        if(gr<S) val=*reinterpret_cast<const uint4*>(gbh+(size_t)gr*D + c);
        *reinterpret_cast<uint4*>(sm + r*D + c)=val;
    }
}

// ---------------- PASS 1: dK, dV ----------------
__global__ __launch_bounds__(NT) void pass1(
    const __nv_bfloat16* __restrict__ Q,const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,const float* __restrict__ Dv,
    __nv_bfloat16* __restrict__ dK,__nv_bfloat16* __restrict__ dV,
    int B,int H,int S,float scale)
{
    int kb=blockIdx.x,h=blockIdx.y,b=blockIdx.z;
    int nq=(S+BLK-1)/BLK;
    extern __shared__ char smem[];
    __nv_bfloat16* Ks=(__nv_bfloat16*)smem;
    __nv_bfloat16* Vs=Ks+BLK*D;
    __nv_bfloat16* Qs=Vs+BLK*D;
    __nv_bfloat16* Os=Qs+BLK*D;
    __nv_bfloat16* PT=Os+BLK*D;       // [key][q] q-contig, LD=64
    __nv_bfloat16* dST=PT+BLK*BLK;    // [key][q]
    float* Lq=(float*)(dST+BLK*BLK);
    float* Dq=Lq+BLK;

    int tid=threadIdx.x,lane=tid&31,warp=tid>>5,g=lane>>2,tl=lane&3;
    size_t bh=((size_t)(b*H+h))*S;
    const __nv_bfloat16* Kb=K+bh*D;
    const __nv_bfloat16* Vb=V+bh*D;
    const __nv_bfloat16* Qb=Q+bh*D;
    const __nv_bfloat16* Ob=dO+bh*D;
    const float* Lb=L+bh; const float* Db=Dv+bh;

    load_tile(Kb,kb*BLK,S,Ks,tid);
    load_tile(Vb,kb*BLK,S,Vs,tid);

    float dVacc[16][4], dKacc[16][4];
    #pragma unroll
    for(int i=0;i<16;i++)
        #pragma unroll
        for(int j=0;j<4;j++){ dVacc[i][j]=0.f; dKacc[i][j]=0.f; }
    __syncthreads();

    for(int qb=kb; qb<nq; qb++){
        load_tile(Qb,qb*BLK,S,Qs,tid);
        load_tile(Ob,qb*BLK,S,Os,tid);
        if(tid<BLK){ int qg=qb*BLK+tid; Lq[tid]=qg<S?Lb[qg]:0.f; Dq[tid]=qg<S?Db[qg]:0.f; }
        __syncthreads();

        // Phase A: S^T[key,q] = K@Q^T -> P^T
        float cS[8][4];
        #pragma unroll
        for(int j=0;j<8;j++){cS[j][0]=cS[j][1]=cS[j][2]=cS[j][3]=0.f;}
        #pragma unroll
        for(int ks=0;ks<8;ks++){
            uint32_t aK[4];
            aK[0]=ldu32(&Ks[(warp*16+g)*D + ks*16+tl*2]);
            aK[1]=ldu32(&Ks[(warp*16+g+8)*D + ks*16+tl*2]);
            aK[2]=ldu32(&Ks[(warp*16+g)*D + ks*16+tl*2+8]);
            aK[3]=ldu32(&Ks[(warp*16+g+8)*D + ks*16+tl*2+8]);
            #pragma unroll
            for(int j=0;j<8;j++){
                uint32_t bQ[2];
                bQ[0]=ldu32(&Qs[(j*8+g)*D + ks*16+tl*2]);
                bQ[1]=ldu32(&Qs[(j*8+g)*D + ks*16+tl*2+8]);
                mma16816(cS[j],aK,bQ);
            }
        }
        #pragma unroll
        for(int j=0;j<8;j++){
            int qa=j*8+tl*2, qb2=j*8+tl*2+1;
            int kA=warp*16+g, kB=warp*16+g+8;
            int kAg=kb*BLK+kA, kBg=kb*BLK+kB;
            int qag=qb*BLK+qa, qbg=qb*BLK+qb2;
            float p0=(kAg<S&&qag<S&&kAg<=qag)?__expf(scale*cS[j][0]-Lq[qa]):0.f;
            float p1=(kAg<S&&qbg<S&&kAg<=qbg)?__expf(scale*cS[j][1]-Lq[qb2]):0.f;
            float p2=(kBg<S&&qag<S&&kBg<=qag)?__expf(scale*cS[j][2]-Lq[qa]):0.f;
            float p3=(kBg<S&&qbg<S&&kBg<=qbg)?__expf(scale*cS[j][3]-Lq[qb2]):0.f;
            PT[kA*BLK+qa]=__float2bfloat16(p0);
            PT[kA*BLK+qb2]=__float2bfloat16(p1);
            PT[kB*BLK+qa]=__float2bfloat16(p2);
            PT[kB*BLK+qb2]=__float2bfloat16(p3);
        }
        // Phase B: dP^T[key,q] = V@dO^T -> dS^T = P*(dP - D)
        float cP[8][4];
        #pragma unroll
        for(int j=0;j<8;j++){cP[j][0]=cP[j][1]=cP[j][2]=cP[j][3]=0.f;}
        #pragma unroll
        for(int ks=0;ks<8;ks++){
            uint32_t aV[4];
            aV[0]=ldu32(&Vs[(warp*16+g)*D + ks*16+tl*2]);
            aV[1]=ldu32(&Vs[(warp*16+g+8)*D + ks*16+tl*2]);
            aV[2]=ldu32(&Vs[(warp*16+g)*D + ks*16+tl*2+8]);
            aV[3]=ldu32(&Vs[(warp*16+g+8)*D + ks*16+tl*2+8]);
            #pragma unroll
            for(int j=0;j<8;j++){
                uint32_t bO[2];
                bO[0]=ldu32(&Os[(j*8+g)*D + ks*16+tl*2]);
                bO[1]=ldu32(&Os[(j*8+g)*D + ks*16+tl*2+8]);
                mma16816(cP[j],aV,bO);
            }
        }
        #pragma unroll
        for(int j=0;j<8;j++){
            int qa=j*8+tl*2, qb2=j*8+tl*2+1;
            int kA=warp*16+g, kB=warp*16+g+8;
            float p0=__bfloat162float(PT[kA*BLK+qa]);
            float p1=__bfloat162float(PT[kA*BLK+qb2]);
            float p2=__bfloat162float(PT[kB*BLK+qa]);
            float p3=__bfloat162float(PT[kB*BLK+qb2]);
            dST[kA*BLK+qa]=__float2bfloat16(p0*(cP[j][0]-Dq[qa]));
            dST[kA*BLK+qb2]=__float2bfloat16(p1*(cP[j][1]-Dq[qb2]));
            dST[kB*BLK+qa]=__float2bfloat16(p2*(cP[j][2]-Dq[qa]));
            dST[kB*BLK+qb2]=__float2bfloat16(p3*(cP[j][3]-Dq[qb2]));
        }
        __syncthreads();

        // Phase C: dV[key,e] += P^T @ dO   (K=q=64)
        #pragma unroll
        for(int ks=0;ks<4;ks++){
            uint32_t aP[4];
            aP[0]=ldu32(&PT[(warp*16+g)*BLK + ks*16+tl*2]);
            aP[1]=ldu32(&PT[(warp*16+g+8)*BLK + ks*16+tl*2]);
            aP[2]=ldu32(&PT[(warp*16+g)*BLK + ks*16+tl*2+8]);
            aP[3]=ldu32(&PT[(warp*16+g+8)*BLK + ks*16+tl*2+8]);
            #pragma unroll
            for(int jj=0;jj<16;jj++){
                int ec=jj*8+g;
                uint32_t bO[2];
                bO[0]=pk2(&Os[(ks*16+tl*2)*D+ec], &Os[(ks*16+tl*2+1)*D+ec]);
                bO[1]=pk2(&Os[(ks*16+tl*2+8)*D+ec], &Os[(ks*16+tl*2+9)*D+ec]);
                mma16816(dVacc[jj],aP,bO);
            }
        }
        // Phase D: dK[key,e] += dS^T @ Q
        #pragma unroll
        for(int ks=0;ks<4;ks++){
            uint32_t aD[4];
            aD[0]=ldu32(&dST[(warp*16+g)*BLK + ks*16+tl*2]);
            aD[1]=ldu32(&dST[(warp*16+g+8)*BLK + ks*16+tl*2]);
            aD[2]=ldu32(&dST[(warp*16+g)*BLK + ks*16+tl*2+8]);
            aD[3]=ldu32(&dST[(warp*16+g+8)*BLK + ks*16+tl*2+8]);
            #pragma unroll
            for(int jj=0;jj<16;jj++){
                int ec=jj*8+g;
                uint32_t bQ[2];
                bQ[0]=pk2(&Qs[(ks*16+tl*2)*D+ec], &Qs[(ks*16+tl*2+1)*D+ec]);
                bQ[1]=pk2(&Qs[(ks*16+tl*2+8)*D+ec], &Qs[(ks*16+tl*2+9)*D+ec]);
                mma16816(dKacc[jj],aD,bQ);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for(int jj=0;jj<16;jj++){
        int e0=jj*8+tl*2, e1=e0+1;
        int kA=kb*BLK+warp*16+g, kB=kb*BLK+warp*16+g+8;
        if(kA<S){
            dV[(bh+kA)*D+e0]=__float2bfloat16(dVacc[jj][0]);
            dV[(bh+kA)*D+e1]=__float2bfloat16(dVacc[jj][1]);
            dK[(bh+kA)*D+e0]=__float2bfloat16(scale*dKacc[jj][0]);
            dK[(bh+kA)*D+e1]=__float2bfloat16(scale*dKacc[jj][1]);
        }
        if(kB<S){
            dV[(bh+kB)*D+e0]=__float2bfloat16(dVacc[jj][2]);
            dV[(bh+kB)*D+e1]=__float2bfloat16(dVacc[jj][3]);
            dK[(bh+kB)*D+e0]=__float2bfloat16(scale*dKacc[jj][2]);
            dK[(bh+kB)*D+e1]=__float2bfloat16(scale*dKacc[jj][3]);
        }
    }
}

// ---------------- PASS 2: dQ ----------------
__global__ __launch_bounds__(NT) void pass2(
    const __nv_bfloat16* __restrict__ Q,const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,const float* __restrict__ Dv,
    __nv_bfloat16* __restrict__ dQ,
    int B,int H,int S,float scale)
{
    int qb=blockIdx.x,h=blockIdx.y,b=blockIdx.z;
    extern __shared__ char smem[];
    __nv_bfloat16* Qs=(__nv_bfloat16*)smem;
    __nv_bfloat16* Os=Qs+BLK*D;
    __nv_bfloat16* Ks=Os+BLK*D;
    __nv_bfloat16* Vs=Ks+BLK*D;
    __nv_bfloat16* Ps=Vs+BLK*D;       // [q][k] k-contig
    __nv_bfloat16* dSs=Ps+BLK*BLK;
    float* Lq=(float*)(dSs+BLK*BLK);
    float* Dq=Lq+BLK;

    int tid=threadIdx.x,lane=tid&31,warp=tid>>5,g=lane>>2,tl=lane&3;
    size_t bh=((size_t)(b*H+h))*S;
    const __nv_bfloat16* Qb=Q+bh*D;
    const __nv_bfloat16* Ob=dO+bh*D;
    const __nv_bfloat16* Kb=K+bh*D;
    const __nv_bfloat16* Vb=V+bh*D;
    const float* Lb=L+bh; const float* Db=Dv+bh;

    load_tile(Qb,qb*BLK,S,Qs,tid);
    load_tile(Ob,qb*BLK,S,Os,tid);
    if(tid<BLK){ int qg=qb*BLK+tid; Lq[tid]=qg<S?Lb[qg]:0.f; Dq[tid]=qg<S?Db[qg]:0.f; }

    float dQacc[16][4];
    #pragma unroll
    for(int i=0;i<16;i++)
        #pragma unroll
        for(int j=0;j<4;j++) dQacc[i][j]=0.f;
    __syncthreads();

    for(int kb=0; kb<=qb; kb++){
        load_tile(Kb,kb*BLK,S,Ks,tid);
        load_tile(Vb,kb*BLK,S,Vs,tid);
        __syncthreads();

        // Phase A: S[q,k]=Q@K^T -> P
        float cS[8][4];
        #pragma unroll
        for(int j=0;j<8;j++){cS[j][0]=cS[j][1]=cS[j][2]=cS[j][3]=0.f;}
        #pragma unroll
        for(int ks=0;ks<8;ks++){
            uint32_t aQ[4];
            aQ[0]=ldu32(&Qs[(warp*16+g)*D + ks*16+tl*2]);
            aQ[1]=ldu32(&Qs[(warp*16+g+8)*D + ks*16+tl*2]);
            aQ[2]=ldu32(&Qs[(warp*16+g)*D + ks*16+tl*2+8]);
            aQ[3]=ldu32(&Qs[(warp*16+g+8)*D + ks*16+tl*2+8]);
            #pragma unroll
            for(int j=0;j<8;j++){
                uint32_t bK[2];
                bK[0]=ldu32(&Ks[(j*8+g)*D + ks*16+tl*2]);
                bK[1]=ldu32(&Ks[(j*8+g)*D + ks*16+tl*2+8]);
                mma16816(cS[j],aQ,bK);
            }
        }
        #pragma unroll
        for(int j=0;j<8;j++){
            int ka=j*8+tl*2, kb2=j*8+tl*2+1;
            int qA=warp*16+g, qB=warp*16+g+8;
            int qAg=qb*BLK+qA, qBg=qb*BLK+qB;
            int kag=kb*BLK+ka, kbg=kb*BLK+kb2;
            float p0=(qAg<S&&kag<S&&kag<=qAg)?__expf(scale*cS[j][0]-Lq[qA]):0.f;
            float p1=(qAg<S&&kbg<S&&kbg<=qAg)?__expf(scale*cS[j][1]-Lq[qA]):0.f;
            float p2=(qBg<S&&kag<S&&kag<=qBg)?__expf(scale*cS[j][2]-Lq[qB]):0.f;
            float p3=(qBg<S&&kbg<S&&kbg<=qBg)?__expf(scale*cS[j][3]-Lq[qB]):0.f;
            Ps[qA*BLK+ka]=__float2bfloat16(p0);
            Ps[qA*BLK+kb2]=__float2bfloat16(p1);
            Ps[qB*BLK+ka]=__float2bfloat16(p2);
            Ps[qB*BLK+kb2]=__float2bfloat16(p3);
        }
        // Phase B: dP[q,k]=dO@V^T -> dS=P*(dP-D)
        float cP[8][4];
        #pragma unroll
        for(int j=0;j<8;j++){cP[j][0]=cP[j][1]=cP[j][2]=cP[j][3]=0.f;}
        #pragma unroll
        for(int ks=0;ks<8;ks++){
            uint32_t aO[4];
            aO[0]=ldu32(&Os[(warp*16+g)*D + ks*16+tl*2]);
            aO[1]=ldu32(&Os[(warp*16+g+8)*D + ks*16+tl*2]);
            aO[2]=ldu32(&Os[(warp*16+g)*D + ks*16+tl*2+8]);
            aO[3]=ldu32(&Os[(warp*16+g+8)*D + ks*16+tl*2+8]);
            #pragma unroll
            for(int j=0;j<8;j++){
                uint32_t bV[2];
                bV[0]=ldu32(&Vs[(j*8+g)*D + ks*16+tl*2]);
                bV[1]=ldu32(&Vs[(j*8+g)*D + ks*16+tl*2+8]);
                mma16816(cP[j],aO,bV);
            }
        }
        #pragma unroll
        for(int j=0;j<8;j++){
            int ka=j*8+tl*2, kb2=j*8+tl*2+1;
            int qA=warp*16+g, qB=warp*16+g+8;
            float p0=__bfloat162float(Ps[qA*BLK+ka]);
            float p1=__bfloat162float(Ps[qA*BLK+kb2]);
            float p2=__bfloat162float(Ps[qB*BLK+ka]);
            float p3=__bfloat162float(Ps[qB*BLK+kb2]);
            dSs[qA*BLK+ka]=__float2bfloat16(p0*(cP[j][0]-Dq[qA]));
            dSs[qA*BLK+kb2]=__float2bfloat16(p1*(cP[j][1]-Dq[qA]));
            dSs[qB*BLK+ka]=__float2bfloat16(p2*(cP[j][2]-Dq[qB]));
            dSs[qB*BLK+kb2]=__float2bfloat16(p3*(cP[j][3]-Dq[qB]));
        }
        __syncthreads();

        // Phase C: dQ[q,e] += dS @ K   (K=k=64)
        #pragma unroll
        for(int ks=0;ks<4;ks++){
            uint32_t aS[4];
            aS[0]=ldu32(&dSs[(warp*16+g)*BLK + ks*16+tl*2]);
            aS[1]=ldu32(&dSs[(warp*16+g+8)*BLK + ks*16+tl*2]);
            aS[2]=ldu32(&dSs[(warp*16+g)*BLK + ks*16+tl*2+8]);
            aS[3]=ldu32(&dSs[(warp*16+g+8)*BLK + ks*16+tl*2+8]);
            #pragma unroll
            for(int jj=0;jj<16;jj++){
                int ec=jj*8+g;
                uint32_t bK[2];
                bK[0]=pk2(&Ks[(ks*16+tl*2)*D+ec], &Ks[(ks*16+tl*2+1)*D+ec]);
                bK[1]=pk2(&Ks[(ks*16+tl*2+8)*D+ec], &Ks[(ks*16+tl*2+9)*D+ec]);
                mma16816(dQacc[jj],aS,bK);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for(int jj=0;jj<16;jj++){
        int e0=jj*8+tl*2, e1=e0+1;
        int qA=qb*BLK+warp*16+g, qB=qb*BLK+warp*16+g+8;
        if(qA<S){
            dQ[(bh+qA)*D+e0]=__float2bfloat16(scale*dQacc[jj][0]);
            dQ[(bh+qA)*D+e1]=__float2bfloat16(scale*dQacc[jj][1]);
        }
        if(qB<S){
            dQ[(bh+qB)*D+e0]=__float2bfloat16(scale*dQacc[jj][2]);
            dQ[(bh+qB)*D+e1]=__float2bfloat16(scale*dQacc[jj][3]);
        }
    }
}

void run(TensorView Q, TensorView K, TensorView V, TensorView O, TensorView dO, TensorView L,
         TensorView dQ, TensorView dK, TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int B=(int)Q.size(0),H=(int)Q.size(1),S=(int)Q.size(2),d=(int)Q.size(3);
    cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id));

    const __nv_bfloat16* Qp=static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* Kp=static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* Vp=static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* Op=static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dOp=static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* Lp=static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQp=static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dKp=static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dVp=static_cast<__nv_bfloat16*>(dV.data_ptr());

    float scale=1.0f/sqrtf((float)d);
    size_t nrows=(size_t)B*H*S;

    float* Dvec=nullptr; CUDA_CHECK(cudaMalloc(&Dvec,nrows*sizeof(float)));
    {
        int thr=256; size_t need=(nrows+thr-1)/thr;
        unsigned blk=(unsigned)std::min(need,(size_t)65535u); if(blk==0) blk=1;
        compute_D_kernel<<<blk,thr,0,stream>>>(dOp,Op,Dvec,nrows);
    }

    int nkv=(S+BLK-1)/BLK, nq=(S+BLK-1)/BLK;
    int smem=(int)(4*BLK*D*sizeof(__nv_bfloat16)+2*BLK*BLK*sizeof(__nv_bfloat16)+2*BLK*sizeof(float));
    CUDA_CHECK(cudaFuncSetAttribute(pass1,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
    CUDA_CHECK(cudaFuncSetAttribute(pass2,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));

    dim3 g1(nkv,H,B),g2(nq,H,B);
    pass1<<<g1,NT,smem,stream>>>(Qp,Kp,Vp,dOp,Lp,Dvec,dKp,dVp,B,H,S,scale);
    pass2<<<g2,NT,smem,stream>>>(Qp,Kp,Vp,dOp,Lp,Dvec,dQp,B,H,S,scale);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaFree(Dvec);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);
}  // namespace mha_bwd