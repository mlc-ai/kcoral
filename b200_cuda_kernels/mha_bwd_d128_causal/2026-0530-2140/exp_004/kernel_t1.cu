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

constexpr int BLK = 64;     // tile size along seq
constexpr int Dh  = 128;    // head dim
constexpr int LD  = 136;    // padded leading dim (reduces bank conflicts; mult of 4 for uint2 loads)
constexpr int NT  = 512;    // threads per block

__device__ __forceinline__ float bf2f(__nv_bfloat16 x){ return __bfloat162float(x); }

// D = rowsum(dO * O)
__global__ void compute_D_kernel(const __nv_bfloat16* __restrict__ dO,
                                 const __nv_bfloat16* __restrict__ O,
                                 float* __restrict__ Dout, size_t nrows){
    size_t row = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x*blockDim.x;
    for(; row<nrows; row+=stride){
        const __nv_bfloat16* dop = dO + row*Dh;
        const __nv_bfloat16* op  = O  + row*Dh;
        float acc=0.f;
        #pragma unroll
        for(int e=0;e<Dh;e+=8){
            uint4 a=*reinterpret_cast<const uint4*>(dop+e);
            uint4 c=*reinterpret_cast<const uint4*>(op+e);
            __nv_bfloat16* pa=reinterpret_cast<__nv_bfloat16*>(&a);
            __nv_bfloat16* pc=reinterpret_cast<__nv_bfloat16*>(&c);
            #pragma unroll
            for(int k=0;k<8;k++) acc += bf2f(pa[k])*bf2f(pc[k]);
        }
        Dout[row]=acc;
    }
}

// load tile [BLK][Dh] from global (row0..) into smem (LD-padded), zero-pad OOB rows
__device__ __forceinline__ void load_tile(const __nv_bfloat16* gbh, int row0, int S,
                                          __nv_bfloat16* sm, int tid){
    #pragma unroll
    for(int v=tid; v<BLK*32; v+=NT){
        int r=v>>5; int c=(v&31)<<2;
        int gr=row0+r;
        uint2 val=make_uint2(0,0);
        if(gr<S) val=*reinterpret_cast<const uint2*>(gbh+(size_t)gr*Dh+c);
        *reinterpret_cast<uint2*>(sm+r*LD+c)=val;
    }
}

// Pass 1: per (kv-block, h, b) compute dK, dV.  Loop query blocks qb=kb..nq-1.
__global__ __launch_bounds__(NT) void pass1_kernel(
    const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V, const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L, const float* __restrict__ Dv,
    __nv_bfloat16* __restrict__ dK, __nv_bfloat16* __restrict__ dV,
    int B,int H,int S,float scale)
{
    int kb=blockIdx.x, h=blockIdx.y, b=blockIdx.z;
    int k0=kb*BLK; int nq=(S+BLK-1)/BLK;

    extern __shared__ char smem[];
    __nv_bfloat16* Ks=(__nv_bfloat16*)smem;
    __nv_bfloat16* Vs=Ks+BLK*LD;
    __nv_bfloat16* Qs=Vs+BLK*LD;
    __nv_bfloat16* Os=Qs+BLK*LD;
    float* PT=(float*)(Os+BLK*LD);
    float* DT=PT+BLK*BLK;
    float* Lq=DT+BLK*BLK;
    float* Dq=Lq+BLK;

    int tid=threadIdx.x; int ty=tid>>5; int tx=tid&31;
    size_t bh=((size_t)(b*H+h))*S;
    const __nv_bfloat16* Kb=K+bh*Dh;
    const __nv_bfloat16* Vb=V+bh*Dh;
    const __nv_bfloat16* Qb=Q+bh*Dh;
    const __nv_bfloat16* Ob=dO+bh*Dh;
    const float* Lb=L+bh;
    const float* Db=Dv+bh;

    load_tile(Kb,k0,S,Ks,tid);
    load_tile(Vb,k0,S,Vs,tid);

    float aDV[4][4], aDK[4][4];
    #pragma unroll
    for(int i=0;i<4;i++)
        #pragma unroll
        for(int j=0;j<4;j++){ aDV[i][j]=0.f; aDK[i][j]=0.f; }
    __syncthreads();

    for(int qb=kb; qb<nq; qb++){
        int q0=qb*BLK;
        load_tile(Qb,q0,S,Qs,tid);
        load_tile(Ob,q0,S,Os,tid);
        if(tid<BLK){ int g=q0+tid; Lq[tid]=g<S?Lb[g]:0.f; Dq[tid]=g<S?Db[g]:0.f; }
        __syncthreads();

        // S^T[k][q] = K(k)·Q(q) ;  k=ty+16ki, q=tx+32qi
        float aS[4][2];
        #pragma unroll
        for(int ki=0;ki<4;ki++){ aS[ki][0]=0.f; aS[ki][1]=0.f; }
        #pragma unroll 4
        for(int e=0;e<Dh;e++){
            float kk0=bf2f(Ks[(ty+0)*LD+e]);
            float kk1=bf2f(Ks[(ty+16)*LD+e]);
            float kk2=bf2f(Ks[(ty+32)*LD+e]);
            float kk3=bf2f(Ks[(ty+48)*LD+e]);
            float qq0=bf2f(Qs[(tx+0)*LD+e]);
            float qq1=bf2f(Qs[(tx+32)*LD+e]);
            aS[0][0]+=kk0*qq0; aS[0][1]+=kk0*qq1;
            aS[1][0]+=kk1*qq0; aS[1][1]+=kk1*qq1;
            aS[2][0]+=kk2*qq0; aS[2][1]+=kk2*qq1;
            aS[3][0]+=kk3*qq0; aS[3][1]+=kk3*qq1;
        }
        float P[4][2];
        #pragma unroll
        for(int ki=0;ki<4;ki++)
            #pragma unroll
            for(int qi=0;qi<2;qi++){
                int k=ty+16*ki, q=tx+32*qi;
                int kg=k0+k, qg=q0+q;
                float p=(kg<S && qg<S && kg<=qg)? __expf(scale*aS[ki][qi]-Lq[q]) : 0.f;
                P[ki][qi]=p;
                PT[k*BLK+q]=p;
            }

        // dP^T[k][q] = V(k)·dO(q)
        float aP[4][2];
        #pragma unroll
        for(int ki=0;ki<4;ki++){ aP[ki][0]=0.f; aP[ki][1]=0.f; }
        #pragma unroll 4
        for(int e=0;e<Dh;e++){
            float vv0=bf2f(Vs[(ty+0)*LD+e]);
            float vv1=bf2f(Vs[(ty+16)*LD+e]);
            float vv2=bf2f(Vs[(ty+32)*LD+e]);
            float vv3=bf2f(Vs[(ty+48)*LD+e]);
            float oo0=bf2f(Os[(tx+0)*LD+e]);
            float oo1=bf2f(Os[(tx+32)*LD+e]);
            aP[0][0]+=vv0*oo0; aP[0][1]+=vv0*oo1;
            aP[1][0]+=vv1*oo0; aP[1][1]+=vv1*oo1;
            aP[2][0]+=vv2*oo0; aP[2][1]+=vv2*oo1;
            aP[3][0]+=vv3*oo0; aP[3][1]+=vv3*oo1;
        }
        #pragma unroll
        for(int ki=0;ki<4;ki++)
            #pragma unroll
            for(int qi=0;qi<2;qi++){
                int k=ty+16*ki, q=tx+32*qi;
                DT[k*BLK+q]=P[ki][qi]*(aP[ki][qi]-Dq[q]);
            }
        __syncthreads();

        // accumulate dV[k][e]+=sum_q P^T[k][q] dO[q][e]; dK[k][e]+=sum_q dS^T[k][q] Q[q][e]
        // k=ty+16ki, e=tx+32ei
        #pragma unroll 4
        for(int q=0;q<BLK;q++){
            float pt0=PT[(ty+0)*BLK+q], pt1=PT[(ty+16)*BLK+q], pt2=PT[(ty+32)*BLK+q], pt3=PT[(ty+48)*BLK+q];
            float dt0=DT[(ty+0)*BLK+q], dt1=DT[(ty+16)*BLK+q], dt2=DT[(ty+32)*BLK+q], dt3=DT[(ty+48)*BLK+q];
            #pragma unroll
            for(int ei=0;ei<4;ei++){
                int e=tx+32*ei;
                float ov=bf2f(Os[q*LD+e]);
                float qv=bf2f(Qs[q*LD+e]);
                aDV[0][ei]+=pt0*ov; aDV[1][ei]+=pt1*ov; aDV[2][ei]+=pt2*ov; aDV[3][ei]+=pt3*ov;
                aDK[0][ei]+=dt0*qv; aDK[1][ei]+=dt1*qv; aDK[2][ei]+=dt2*qv; aDK[3][ei]+=dt3*qv;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for(int ki=0;ki<4;ki++){
        int kg=k0+ty+16*ki;
        if(kg<S){
            #pragma unroll
            for(int ei=0;ei<4;ei++){
                int e=tx+32*ei;
                dV[(bh+kg)*Dh+e]=__float2bfloat16(aDV[ki][ei]);
                dK[(bh+kg)*Dh+e]=__float2bfloat16(scale*aDK[ki][ei]);
            }
        }
    }
}

// Pass 2: per (q-block, h, b) compute dQ.  Loop kv blocks kb=0..qb.
__global__ __launch_bounds__(NT) void pass2_kernel(
    const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V, const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L, const float* __restrict__ Dv,
    __nv_bfloat16* __restrict__ dQ,
    int B,int H,int S,float scale)
{
    int qb=blockIdx.x, h=blockIdx.y, b=blockIdx.z;
    int q0=qb*BLK;

    extern __shared__ char smem[];
    __nv_bfloat16* Qs=(__nv_bfloat16*)smem;
    __nv_bfloat16* Os=Qs+BLK*LD;
    __nv_bfloat16* Ks=Os+BLK*LD;
    __nv_bfloat16* Vs=Ks+BLK*LD;
    float* dS=(float*)(Vs+BLK*LD);
    float* Lq=dS+BLK*BLK;
    float* Dq=Lq+BLK;

    int tid=threadIdx.x; int ty=tid>>5; int tx=tid&31;
    size_t bh=((size_t)(b*H+h))*S;
    const __nv_bfloat16* Qb=Q+bh*Dh;
    const __nv_bfloat16* Ob=dO+bh*Dh;
    const __nv_bfloat16* Kb=K+bh*Dh;
    const __nv_bfloat16* Vb=V+bh*Dh;
    const float* Lb=L+bh;
    const float* Db=Dv+bh;

    load_tile(Qb,q0,S,Qs,tid);
    load_tile(Ob,q0,S,Os,tid);
    if(tid<BLK){ int g=q0+tid; Lq[tid]=g<S?Lb[g]:0.f; Dq[tid]=g<S?Db[g]:0.f; }

    float aDQ[4][4];
    #pragma unroll
    for(int i=0;i<4;i++)
        #pragma unroll
        for(int j=0;j<4;j++) aDQ[i][j]=0.f;
    __syncthreads();

    for(int kb=0; kb<=qb; kb++){
        int k0=kb*BLK;
        load_tile(Kb,k0,S,Ks,tid);
        load_tile(Vb,k0,S,Vs,tid);
        __syncthreads();

        // S[q][k] = Q(q)·K(k); q=ty+16qi, k=tx+32ki
        float aS[4][2];
        #pragma unroll
        for(int qi=0;qi<4;qi++){ aS[qi][0]=0.f; aS[qi][1]=0.f; }
        #pragma unroll 4
        for(int e=0;e<Dh;e++){
            float qq0=bf2f(Qs[(ty+0)*LD+e]);
            float qq1=bf2f(Qs[(ty+16)*LD+e]);
            float qq2=bf2f(Qs[(ty+32)*LD+e]);
            float qq3=bf2f(Qs[(ty+48)*LD+e]);
            float kk0=bf2f(Ks[(tx+0)*LD+e]);
            float kk1=bf2f(Ks[(tx+32)*LD+e]);
            aS[0][0]+=qq0*kk0; aS[0][1]+=qq0*kk1;
            aS[1][0]+=qq1*kk0; aS[1][1]+=qq1*kk1;
            aS[2][0]+=qq2*kk0; aS[2][1]+=qq2*kk1;
            aS[3][0]+=qq3*kk0; aS[3][1]+=qq3*kk1;
        }
        float P[4][2];
        #pragma unroll
        for(int qi=0;qi<4;qi++)
            #pragma unroll
            for(int ki=0;ki<2;ki++){
                int q=ty+16*qi, k=tx+32*ki;
                int qg=q0+q, kg=k0+k;
                float p=(qg<S && kg<S && kg<=qg)? __expf(scale*aS[qi][ki]-Lq[q]) : 0.f;
                P[qi][ki]=p;
            }
        // dP[q][k] = dO(q)·V(k)
        float aP[4][2];
        #pragma unroll
        for(int qi=0;qi<4;qi++){ aP[qi][0]=0.f; aP[qi][1]=0.f; }
        #pragma unroll 4
        for(int e=0;e<Dh;e++){
            float oo0=bf2f(Os[(ty+0)*LD+e]);
            float oo1=bf2f(Os[(ty+16)*LD+e]);
            float oo2=bf2f(Os[(ty+32)*LD+e]);
            float oo3=bf2f(Os[(ty+48)*LD+e]);
            float vv0=bf2f(Vs[(tx+0)*LD+e]);
            float vv1=bf2f(Vs[(tx+32)*LD+e]);
            aP[0][0]+=oo0*vv0; aP[0][1]+=oo0*vv1;
            aP[1][0]+=oo1*vv0; aP[1][1]+=oo1*vv1;
            aP[2][0]+=oo2*vv0; aP[2][1]+=oo2*vv1;
            aP[3][0]+=oo3*vv0; aP[3][1]+=oo3*vv1;
        }
        #pragma unroll
        for(int qi=0;qi<4;qi++)
            #pragma unroll
            for(int ki=0;ki<2;ki++){
                int q=ty+16*qi, k=tx+32*ki;
                dS[q*BLK+k]=P[qi][ki]*(aP[qi][ki]-Dq[q]);
            }
        __syncthreads();

        // dQ[q][e] += sum_k dS[q][k] K[k][e]; q=ty+16qi, e=tx+32ei
        #pragma unroll 4
        for(int k=0;k<BLK;k++){
            float ds0=dS[(ty+0)*BLK+k], ds1=dS[(ty+16)*BLK+k], ds2=dS[(ty+32)*BLK+k], ds3=dS[(ty+48)*BLK+k];
            #pragma unroll
            for(int ei=0;ei<4;ei++){
                int e=tx+32*ei;
                float kv=bf2f(Ks[k*LD+e]);
                aDQ[0][ei]+=ds0*kv; aDQ[1][ei]+=ds1*kv; aDQ[2][ei]+=ds2*kv; aDQ[3][ei]+=ds3*kv;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for(int qi=0;qi<4;qi++){
        int qg=q0+ty+16*qi;
        if(qg<S){
            #pragma unroll
            for(int ei=0;ei<4;ei++){
                int e=tx+32*ei;
                dQ[(bh+qg)*Dh+e]=__float2bfloat16(scale*aDQ[qi][ei]);
            }
        }
    }
}

void run(TensorView Q, TensorView K, TensorView V, TensorView O, TensorView dO, TensorView L,
         TensorView dQ, TensorView dK, TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int B=(int)Q.size(0), H=(int)Q.size(1), S=(int)Q.size(2), d=(int)Q.size(3);

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

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

    float* Dvec=nullptr; CUDA_CHECK(cudaMalloc(&Dvec, nrows*sizeof(float)));

    {
        int thr=256;
        size_t need=(nrows+thr-1)/thr;
        unsigned blk=(unsigned)std::min(need,(size_t)65535u);
        if(blk==0) blk=1;
        compute_D_kernel<<<blk,thr,0,stream>>>(dOp,Op,Dvec,nrows);
    }

    int nkv=(S+BLK-1)/BLK;
    int nq =(S+BLK-1)/BLK;

    int smem1=(int)(4*BLK*LD*sizeof(__nv_bfloat16) + 2*BLK*BLK*sizeof(float) + 2*BLK*sizeof(float));
    int smem2=(int)(4*BLK*LD*sizeof(__nv_bfloat16) + 1*BLK*BLK*sizeof(float) + 2*BLK*sizeof(float));

    CUDA_CHECK(cudaFuncSetAttribute(pass1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem1));
    CUDA_CHECK(cudaFuncSetAttribute(pass2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem2));

    dim3 g1(nkv,H,B), g2(nq,H,B);
    pass1_kernel<<<g1,NT,smem1,stream>>>(Qp,Kp,Vp,dOp,Lp,Dvec,dKp,dVp,B,H,S,scale);
    pass2_kernel<<<g2,NT,smem2,stream>>>(Qp,Kp,Vp,dOp,Lp,Dvec,dQp,B,H,S,scale);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaFree(Dvec);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd