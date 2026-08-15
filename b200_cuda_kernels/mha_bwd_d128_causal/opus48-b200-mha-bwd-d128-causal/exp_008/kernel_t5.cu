#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__);} }while(0)

namespace mha_bwd {
using bf16 = __nv_bfloat16;
using half_t = __half;

#define HD 128
#define NTHREADS 256

__device__ __forceinline__ half_t bf2h(bf16 x){ return __float2half(__bfloat162float(x)); }

__device__ __forceinline__ void ldm_x4(const half_t* p,uint32_t&a,uint32_t&b,uint32_t&c,uint32_t&d){
    uint32_t s=(uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];":"=r"(a),"=r"(b),"=r"(c),"=r"(d):"r"(s));
}
__device__ __forceinline__ void ldm_x2(const half_t* p,uint32_t&a,uint32_t&b){
    uint32_t s=(uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0,%1},[%2];":"=r"(a),"=r"(b):"r"(s));
}

// C[16,N] (+)= A[16,K] * B[N,K], per-warp (16 rows at r0). fp16->f32 mma.
template<int NT8,int KK,int LDA,int LDB>
__device__ __forceinline__ void wmma(const half_t* A,const half_t* B,int r0,int lane,float c[NT8][4],bool accum){
    if(!accum){ 
        #pragma unroll
        for(int i=0;i<NT8;i++){c[i][0]=c[i][1]=c[i][2]=c[i][3]=0.f;} 
    }
    int aR=lane&15, aC=(lane>>4)*8;
    int bR=lane&7, bC=((lane>>3)&1)*8;
    #pragma unroll
    for(int k=0;k<KK;k+=16){
        uint32_t a0,a1,a2,a3;
        ldm_x4(&A[(r0+aR)*LDA + k + aC], a0,a1,a2,a3);
        #pragma unroll
        for(int ni=0;ni<NT8;ni++){
            int nn=ni*8;
            uint32_t b0,b1;
            ldm_x2(&B[(nn+bR)*LDB + k + bC], b0,b1);
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
              "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
              :"+f"(c[ni][0]),"+f"(c[ni][1]),"+f"(c[ni][2]),"+f"(c[ni][3])
              :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
        }
    }
}

__global__ void compute_D_kernel(const bf16* __restrict__ O,const bf16* __restrict__ dO,
                                 float* __restrict__ Dout,int total_rows){
    int row=blockIdx.x; if(row>=total_rows) return;
    int tid=threadIdx.x;
    float v=__bfloat162float(O[(size_t)row*HD+tid])*__bfloat162float(dO[(size_t)row*HD+tid]);
    __shared__ float sm[HD];
    sm[tid]=v; __syncthreads();
    for(int s=HD/2;s>0;s>>=1){ if(tid<s) sm[tid]+=sm[tid+s]; __syncthreads(); }
    if(tid==0) Dout[row]=sm[0];
}

// ---------------- dK, dV : parallel over key blocks (BK=128) --------------
#define BK 128
#define BQ 64
__global__ void __launch_bounds__(NTHREADS) bwd_dkv_kernel(
    const bf16* __restrict__ Q,const bf16* __restrict__ K,const bf16* __restrict__ V,
    const bf16* __restrict__ dO,const float* __restrict__ Lv,const float* __restrict__ Dv,
    bf16* __restrict__ dK,bf16* __restrict__ dV,int S,float scale)
{
    int bh=blockIdx.y, kb=blockIdx.x, kj0=kb*BK;
    if(kj0>=S) return;
    int tid=threadIdx.x, warp_id=tid>>5, lane=tid&31;
    int group=lane>>2, tig=lane&3, t2=tig*2;
    int r0=warp_id*16;

    const bf16* Kp=K+(size_t)bh*S*HD; const bf16* Vp=V+(size_t)bh*S*HD;
    const bf16* Qp=Q+(size_t)bh*S*HD; const bf16* dOp=dO+(size_t)bh*S*HD;
    const float* Lp=Lv+(size_t)bh*S; const float* Dp=Dv+(size_t)bh*S;
    bf16* dKp=dK+(size_t)bh*S*HD; bf16* dVp=dV+(size_t)bh*S*HD;

    extern __shared__ __align__(16) char smem[];
    half_t* Ksm=(half_t*)smem;            // [BK,HD]
    half_t* Vsm=Ksm+BK*HD;                // [BK,HD]
    half_t* Qsm=Vsm+BK*HD;                // [BQ,HD]
    half_t* QTsm=Qsm+BQ*HD;               // [HD,BQ]
    half_t* dOsm=QTsm+HD*BQ;              // [BQ,HD]
    half_t* dOTsm=dOsm+BQ*HD;             // [HD,BQ]
    half_t* Ph=dOTsm+HD*BQ;               // [BK,BQ]
    half_t* dSh=Ph+BK*BQ;                 // [BK,BQ]
    float* Lsm=(float*)(dSh+BK*BQ);       // [BQ]
    float* Dsm=Lsm+BQ;                    // [BQ]
    half_t h0=__float2half(0.f);

    float dVr[16][4], dKr[16][4];
    #pragma unroll
    for(int i=0;i<16;i++){dVr[i][0]=dVr[i][1]=dVr[i][2]=dVr[i][3]=0.f;dKr[i][0]=dKr[i][1]=dKr[i][2]=dKr[i][3]=0.f;}

    for(int idx=tid;idx<BK*HD;idx+=NTHREADS){
        int j=idx/HD,k=idx%HD,gj=kj0+j;
        Ksm[idx]=(gj<S)?bf2h(Kp[(size_t)gj*HD+k]):h0;
        Vsm[idx]=(gj<S)?bf2h(Vp[(size_t)gj*HD+k]):h0;
    }
    __syncthreads();

    int numQB=(S+BQ-1)/BQ;
    for(int qb=kb*(BK/BQ); qb<numQB; qb++){
        int qi0=qb*BQ;
        __syncthreads();
        for(int idx=tid;idx<BQ*HD;idx+=NTHREADS){
            int i=idx/HD,k=idx%HD,gi=qi0+i;
            half_t qv=(gi<S)?bf2h(Qp[(size_t)gi*HD+k]):h0;
            half_t dv=(gi<S)?bf2h(dOp[(size_t)gi*HD+k]):h0;
            Qsm[i*HD+k]=qv; QTsm[k*BQ+i]=qv;
            dOsm[i*HD+k]=dv; dOTsm[k*BQ+i]=dv;
        }
        for(int idx=tid;idx<BQ;idx+=NTHREADS){int gi=qi0+idx; Lsm[idx]=(gi<S)?Lp[gi]:0.f; Dsm[idx]=(gi<S)?Dp[gi]:0.f;}
        __syncthreads();

        float accS[8][4], accP[8][4];
        // S^T = K * Q^T
        wmma<8,HD,HD,HD>(Ksm,Qsm,r0,lane,accS,false);
        #pragma unroll
        for(int ni=0;ni<8;ni++){
            int nn=ni*8, kr0=r0+group, kr1=r0+group+8, qc0=nn+t2, qc1=nn+t2+1;
            { int kg=kj0+kr0,qg=qi0+qc0; bool v=(kg<S)&&(qg<S)&&(kg<=qg); float p=v?__expf(scale*accS[ni][0]-Lsm[qc0]):0.f; Ph[kr0*BQ+qc0]=__float2half(p);}
            { int kg=kj0+kr0,qg=qi0+qc1; bool v=(kg<S)&&(qg<S)&&(kg<=qg); float p=v?__expf(scale*accS[ni][1]-Lsm[qc1]):0.f; Ph[kr0*BQ+qc1]=__float2half(p);}
            { int kg=kj0+kr1,qg=qi0+qc0; bool v=(kg<S)&&(qg<S)&&(kg<=qg); float p=v?__expf(scale*accS[ni][2]-Lsm[qc0]):0.f; Ph[kr1*BQ+qc0]=__float2half(p);}
            { int kg=kj0+kr1,qg=qi0+qc1; bool v=(kg<S)&&(qg<S)&&(kg<=qg); float p=v?__expf(scale*accS[ni][3]-Lsm[qc1]):0.f; Ph[kr1*BQ+qc1]=__float2half(p);}
        }
        __syncwarp();
        // dP^T = V * dO^T
        wmma<8,HD,HD,HD>(Vsm,dOsm,r0,lane,accP,false);
        #pragma unroll
        for(int ni=0;ni<8;ni++){
            int nn=ni*8, kr0=r0+group, kr1=r0+group+8, qc0=nn+t2, qc1=nn+t2+1;
            float p0=__half2float(Ph[kr0*BQ+qc0]); dSh[kr0*BQ+qc0]=__float2half(p0*(accP[ni][0]-Dsm[qc0]));
            float p1=__half2float(Ph[kr0*BQ+qc1]); dSh[kr0*BQ+qc1]=__float2half(p1*(accP[ni][1]-Dsm[qc1]));
            float p2=__half2float(Ph[kr1*BQ+qc0]); dSh[kr1*BQ+qc0]=__float2half(p2*(accP[ni][2]-Dsm[qc0]));
            float p3=__half2float(Ph[kr1*BQ+qc1]); dSh[kr1*BQ+qc1]=__float2half(p3*(accP[ni][3]-Dsm[qc1]));
        }
        __syncwarp();
        // dV += P^T * dO ; dK += dS^T * Q
        wmma<16,BQ,BQ,BQ>(Ph, dOTsm, r0, lane, dVr, true);
        wmma<16,BQ,BQ,BQ>(dSh, QTsm, r0, lane, dKr, true);
    }

    #pragma unroll
    for(int ni=0;ni<16;ni++){
        int col0=ni*8+t2, kr0=r0+group, kr1=r0+group+8;
        int kg0=kj0+kr0, kg1=kj0+kr1;
        if(kg0<S){
            dVp[(size_t)kg0*HD+col0]=__float2bfloat16(dVr[ni][0]);
            dVp[(size_t)kg0*HD+col0+1]=__float2bfloat16(dVr[ni][1]);
            dKp[(size_t)kg0*HD+col0]=__float2bfloat16(scale*dKr[ni][0]);
            dKp[(size_t)kg0*HD+col0+1]=__float2bfloat16(scale*dKr[ni][1]);
        }
        if(kg1<S){
            dVp[(size_t)kg1*HD+col0]=__float2bfloat16(dVr[ni][2]);
            dVp[(size_t)kg1*HD+col0+1]=__float2bfloat16(dVr[ni][3]);
            dKp[(size_t)kg1*HD+col0]=__float2bfloat16(scale*dKr[ni][2]);
            dKp[(size_t)kg1*HD+col0+1]=__float2bfloat16(scale*dKr[ni][3]);
        }
    }
}

// ---------------- dQ : parallel over query blocks (BQD=128) --------------
#define BQD 128
#define BKD 64
__global__ void __launch_bounds__(NTHREADS) bwd_dq_kernel(
    const bf16* __restrict__ Q,const bf16* __restrict__ K,const bf16* __restrict__ V,
    const bf16* __restrict__ dO,const float* __restrict__ Lv,const float* __restrict__ Dv,
    bf16* __restrict__ dQ,int S,float scale)
{
    int bh=blockIdx.y, qb=blockIdx.x, qi0=qb*BQD;
    if(qi0>=S) return;
    int tid=threadIdx.x, warp_id=tid>>5, lane=tid&31;
    int group=lane>>2, tig=lane&3, t2=tig*2;
    int r0=warp_id*16;

    const bf16* Kp=K+(size_t)bh*S*HD; const bf16* Vp=V+(size_t)bh*S*HD;
    const bf16* Qp=Q+(size_t)bh*S*HD; const bf16* dOp=dO+(size_t)bh*S*HD;
    const float* Lp=Lv+(size_t)bh*S; const float* Dp=Dv+(size_t)bh*S;
    bf16* dQp=dQ+(size_t)bh*S*HD;

    extern __shared__ __align__(16) char smem[];
    half_t* Qsm=(half_t*)smem;         // [BQD,HD]
    half_t* dOsm=Qsm+BQD*HD;           // [BQD,HD]
    half_t* Ksm=dOsm+BQD*HD;           // [BKD,HD]
    half_t* KTsm=Ksm+BKD*HD;           // [HD,BKD]
    half_t* Vsm=KTsm+HD*BKD;           // [BKD,HD]
    half_t* Ph=Vsm+BKD*HD;             // [BQD,BKD]
    half_t* dSh=Ph+BQD*BKD;            // [BQD,BKD]
    float* Lsm=(float*)(dSh+BQD*BKD);  // [BQD]
    float* Dsm=Lsm+BQD;                // [BQD]
    half_t h0=__float2half(0.f);

    float dQr[16][4];
    #pragma unroll
    for(int i=0;i<16;i++){dQr[i][0]=dQr[i][1]=dQr[i][2]=dQr[i][3]=0.f;}

    for(int idx=tid;idx<BQD*HD;idx+=NTHREADS){
        int i=idx/HD,k=idx%HD,gi=qi0+i;
        Qsm[idx]=(gi<S)?bf2h(Qp[(size_t)gi*HD+k]):h0;
        dOsm[idx]=(gi<S)?bf2h(dOp[(size_t)gi*HD+k]):h0;
    }
    for(int idx=tid;idx<BQD;idx+=NTHREADS){int gi=qi0+idx; Lsm[idx]=(gi<S)?Lp[gi]:0.f; Dsm[idx]=(gi<S)?Dp[gi]:0.f;}
    __syncthreads();

    int numKB=(S+BKD-1)/BKD;
    int kb_max=(qi0+BQD-1)/BKD;
    for(int kb=0; kb<=kb_max && kb<numKB; kb++){
        int kj0=kb*BKD;
        __syncthreads();
        for(int idx=tid;idx<BKD*HD;idx+=NTHREADS){
            int j=idx/HD,k=idx%HD,gj=kj0+j;
            half_t kv=(gj<S)?bf2h(Kp[(size_t)gj*HD+k]):h0;
            half_t vv=(gj<S)?bf2h(Vp[(size_t)gj*HD+k]):h0;
            Ksm[j*HD+k]=kv; KTsm[k*BKD+j]=kv; Vsm[idx]=vv;
        }
        __syncthreads();

        float accS[8][4], accP[8][4];
        // S = Q * K^T  (N=BKD=64)
        wmma<8,HD,HD,HD>(Qsm,Ksm,r0,lane,accS,false);
        #pragma unroll
        for(int ni=0;ni<8;ni++){
            int nn=ni*8, qr0=r0+group, qr1=r0+group+8, kc0=nn+t2, kc1=nn+t2+1;
            { int qg=qi0+qr0,kg=kj0+kc0; bool v=(qg<S)&&(kg<S)&&(kg<=qg); float p=v?__expf(scale*accS[ni][0]-Lsm[qr0]):0.f; Ph[qr0*BKD+kc0]=__float2half(p);}
            { int qg=qi0+qr0,kg=kj0+kc1; bool v=(qg<S)&&(kg<S)&&(kg<=qg); float p=v?__expf(scale*accS[ni][1]-Lsm[qr0]):0.f; Ph[qr0*BKD+kc1]=__float2half(p);}
            { int qg=qi0+qr1,kg=kj0+kc0; bool v=(qg<S)&&(kg<S)&&(kg<=qg); float p=v?__expf(scale*accS[ni][2]-Lsm[qr1]):0.f; Ph[qr1*BKD+kc0]=__float2half(p);}
            { int qg=qi0+qr1,kg=kj0+kc1; bool v=(qg<S)&&(kg<S)&&(kg<=qg); float p=v?__expf(scale*accS[ni][3]-Lsm[qr1]):0.f; Ph[qr1*BKD+kc1]=__float2half(p);}
        }
        __syncwarp();
        // dP = dO * V^T
        wmma<8,HD,HD,HD>(dOsm,Vsm,r0,lane,accP,false);
        #pragma unroll
        for(int ni=0;ni<8;ni++){
            int nn=ni*8, qr0=r0+group, qr1=r0+group+8, kc0=nn+t2, kc1=nn+t2+1;
            float p0=__half2float(Ph[qr0*BKD+kc0]); dSh[qr0*BKD+kc0]=__float2half(p0*(accP[ni][0]-Dsm[qr0]));
            float p1=__half2float(Ph[qr0*BKD+kc1]); dSh[qr0*BKD+kc1]=__float2half(p1*(accP[ni][1]-Dsm[qr0]));
            float p2=__half2float(Ph[qr1*BKD+kc0]); dSh[qr1*BKD+kc0]=__float2half(p2*(accP[ni][2]-Dsm[qr1]));
            float p3=__half2float(Ph[qr1*BKD+kc1]); dSh[qr1*BKD+kc1]=__float2half(p3*(accP[ni][3]-Dsm[qr1]));
        }
        __syncwarp();
        // dQ += dS * K  (N=HD=128, K=BKD=64)
        wmma<16,BKD,BKD,BKD>(dSh, KTsm, r0, lane, dQr, true);
    }

    #pragma unroll
    for(int ni=0;ni<16;ni++){
        int col0=ni*8+t2, qr0=r0+group, qr1=r0+group+8;
        int qg0=qi0+qr0, qg1=qi0+qr1;
        if(qg0<S){
            dQp[(size_t)qg0*HD+col0]=__float2bfloat16(scale*dQr[ni][0]);
            dQp[(size_t)qg0*HD+col0+1]=__float2bfloat16(scale*dQr[ni][1]);
        }
        if(qg1<S){
            dQp[(size_t)qg1*HD+col0]=__float2bfloat16(scale*dQr[ni][2]);
            dQp[(size_t)qg1*HD+col0+1]=__float2bfloat16(scale*dQr[ni][3]);
        }
    }
}

void run(tvm::ffi::TensorView Q,tvm::ffi::TensorView K,tvm::ffi::TensorView V,
         tvm::ffi::TensorView O,tvm::ffi::TensorView dO,tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ,tvm::ffi::TensorView dK,tvm::ffi::TensorView dV){
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int B=Q.size(0),H=Q.size(1),S=Q.size(2),d=Q.size(3);
    float scale=1.0f/sqrtf((float)d);

    const bf16* Qp=(const bf16*)Q.data_ptr(); const bf16* Kp=(const bf16*)K.data_ptr();
    const bf16* Vp=(const bf16*)V.data_ptr(); const bf16* Op=(const bf16*)O.data_ptr();
    const bf16* dOp=(const bf16*)dO.data_ptr(); const float* Lp=(const float*)L.data_ptr();
    bf16* dQp=(bf16*)dQ.data_ptr(); bf16* dKp=(bf16*)dK.data_ptr(); bf16* dVp=(bf16*)dV.data_ptr();

    cudaStream_t stream=(cudaStream_t)TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id);

    float* Dscr=nullptr;
    CUDA_CHECK(cudaMallocAsync((void**)&Dscr,(size_t)B*H*S*sizeof(float),stream));

    int total_rows=B*H*S;
    compute_D_kernel<<<total_rows,HD,0,stream>>>(Op,dOp,Dscr,total_rows);
    CUDA_CHECK(cudaGetLastError());

    size_t sm_dkv=(size_t)(2*BK*HD + 4*BQ*HD + 2*BK*BQ)*sizeof(half_t) + (size_t)(2*BQ)*sizeof(float);
    size_t sm_dq =(size_t)(2*BQD*HD + 3*BKD*HD + 2*BQD*BKD)*sizeof(half_t) + (size_t)(2*BQD)*sizeof(float);

    CUDA_CHECK(cudaFuncSetAttribute(bwd_dkv_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sm_dkv));
    CUDA_CHECK(cudaFuncSetAttribute(bwd_dq_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sm_dq));

    int numKB=(S+BK-1)/BK;
    dim3 g1(numKB,B*H);
    bwd_dkv_kernel<<<g1,NTHREADS,sm_dkv,stream>>>(Qp,Kp,Vp,dOp,Lp,Dscr,dKp,dVp,S,scale);
    CUDA_CHECK(cudaGetLastError());

    int numQB=(S+BQD-1)/BQD;
    dim3 g2(numQB,B*H);
    bwd_dq_kernel<<<g2,NTHREADS,sm_dq,stream>>>(Qp,Kp,Vp,dOp,Lp,Dscr,dQp,S,scale);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFreeAsync(Dscr,stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd