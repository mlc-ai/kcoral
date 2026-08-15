#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
    }                                                              \
} while(0)

namespace mha {

constexpr int BM=64, BN=64, HD=128, THREADS=128;

__device__ __forceinline__ float ex2(float x){ float y; asm volatile("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x)); return y; }
__device__ __forceinline__ uint32_t cvta(const void*p){ return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp_async16(uint32_t dst,const void* src,bool pred){
    int sz=pred?16:0;
    asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;\n"::"r"(dst),"l"(src),"r"(sz));
}
__device__ __forceinline__ void cp_commit(){ asm volatile("cp.async.commit_group;\n"); }
template<int N> __device__ __forceinline__ void cp_wait(){ asm volatile("cp.async.wait_group %0;\n"::"n"(N):"memory"); }

__device__ __forceinline__ void ldm_x4(unsigned&a,unsigned&b,unsigned&c,unsigned&d,uint32_t s){
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];":"=r"(a),"=r"(b),"=r"(c),"=r"(d):"r"(s)); }
__device__ __forceinline__ void ldm_x2(unsigned&a,unsigned&b,uint32_t s){
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2];":"=r"(a),"=r"(b):"r"(s)); }
__device__ __forceinline__ void ldm_x2t(unsigned&a,unsigned&b,uint32_t s){
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];":"=r"(a),"=r"(b):"r"(s)); }
__device__ __forceinline__ void mma16816(float&d0,float&d1,float&d2,float&d3,
        unsigned a0,unsigned a1,unsigned a2,unsigned a3,unsigned b0,unsigned b1,
        float c0,float c1,float c2,float c3){
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
        :"=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
         "f"(c0),"f"(c1),"f"(c2),"f"(c3)); }

__global__ void __launch_bounds__(128)
mha_kernel(const __nv_bfloat16* __restrict__ Q,const __nv_bfloat16* __restrict__ K,
           const __nv_bfloat16* __restrict__ V,__nv_bfloat16* __restrict__ O,
           float* __restrict__ LSE,int B,int H,int S,float scale){
    extern __shared__ char smem[];
    __nv_bfloat16* Qs=(__nv_bfloat16*)smem;
    __nv_bfloat16* Ks=Qs+BM*HD;
    __nv_bfloat16* Vs=Ks+BN*HD;
    __nv_bfloat16* Ps=Vs+BN*HD;

    int tid=threadIdx.x;
    int warp=tid>>5, lane=tid&31, groupID=lane>>2, tidg=lane&3;
    int q0=blockIdx.x*BM;
    int headoff=blockIdx.z*H+blockIdx.y;
    const __nv_bfloat16* Qbase=Q+(int64_t)headoff*S*HD;
    const __nv_bfloat16* Kbase=K+(int64_t)headoff*S*HD;
    const __nv_bfloat16* Vbase=V+(int64_t)headoff*S*HD;
    __nv_bfloat16* Obase=O+(int64_t)headoff*S*HD;
    float scale2=scale*1.4426950408889634f;

    // load Q (resident)
    {
        const int vpr=HD/8;
        for(int v=tid; v<BM*vpr; v+=THREADS){
            int r=v/vpr,c=(v%vpr)*8; int grow=q0+r;
            cp_async16(cvta(&Qs[r*HD+c]), Qbase+(int64_t)grow*HD+c, grow<S);
        }
    }
    cp_commit();

    float oacc[16][4];
    #pragma unroll
    for(int i=0;i<16;i++){oacc[i][0]=oacc[i][1]=oacc[i][2]=oacc[i][3]=0.f;}
    float mrun0=-1e30f,mrun1=-1e30f,lrun0=0.f,lrun1=0.f;

    cp_wait<0>(); __syncthreads();

    int numKB=(S+BN-1)/BN;
    for(int kbi=0;kbi<numKB;kbi++){
        int kb=kbi*BN;
        // async load K,V
        {
            const int vpr=HD/8;
            for(int v=tid; v<BN*vpr; v+=THREADS){
                int r=v/vpr,c=(v%vpr)*8; int grow=kb+r; bool inr=grow<S;
                cp_async16(cvta(&Ks[r*HD+c]), Kbase+(int64_t)grow*HD+c, inr);
                cp_async16(cvta(&Vs[r*HD+c]), Vbase+(int64_t)grow*HD+c, inr);
            }
        }
        cp_commit(); cp_wait<0>(); __syncthreads();

        // ---- S = Q K^T ----
        float sacc[8][4];
        #pragma unroll
        for(int nt=0;nt<8;nt++){sacc[nt][0]=sacc[nt][1]=sacc[nt][2]=sacc[nt][3]=0.f;}
        #pragma unroll
        for(int kt=0;kt<8;kt++){
            unsigned a0,a1,a2,a3;
            { int m4=lane>>3,r=lane&7; int row=(m4&1)*8+r,col=(m4>>1)*8;
              ldm_x4(a0,a1,a2,a3, cvta(&Qs[(warp*16+row)*HD + kt*16 + col])); }
            #pragma unroll
            for(int nt=0;nt<8;nt++){
                unsigned b0,b1;
                { int q=lane>>3,r=lane&7,blk=q&1; int kvr=nt*8+r,dcol=kt*16+blk*8;
                  ldm_x2(b0,b1, cvta(&Ks[kvr*HD + dcol])); }
                mma16816(sacc[nt][0],sacc[nt][1],sacc[nt][2],sacc[nt][3],
                         a0,a1,a2,a3,b0,b1,
                         sacc[nt][0],sacc[nt][1],sacc[nt][2],sacc[nt][3]);
            }
        }

        // scale + mask
        #pragma unroll
        for(int nt=0;nt<8;nt++){
            int col0=nt*8+tidg*2, col1=col0+1;
            int kc0=kb+col0, kc1=kb+col1;
            sacc[nt][0]=(kc0<S)?sacc[nt][0]*scale2:-1e30f;
            sacc[nt][1]=(kc1<S)?sacc[nt][1]*scale2:-1e30f;
            sacc[nt][2]=(kc0<S)?sacc[nt][2]*scale2:-1e30f;
            sacc[nt][3]=(kc1<S)?sacc[nt][3]*scale2:-1e30f;
        }
        float bm0=-1e30f,bm1=-1e30f;
        #pragma unroll
        for(int nt=0;nt<8;nt++){bm0=fmaxf(bm0,fmaxf(sacc[nt][0],sacc[nt][1])); bm1=fmaxf(bm1,fmaxf(sacc[nt][2],sacc[nt][3]));}
        bm0=fmaxf(bm0,__shfl_xor_sync(0xffffffff,bm0,1)); bm0=fmaxf(bm0,__shfl_xor_sync(0xffffffff,bm0,2));
        bm1=fmaxf(bm1,__shfl_xor_sync(0xffffffff,bm1,1)); bm1=fmaxf(bm1,__shfl_xor_sync(0xffffffff,bm1,2));
        float mnew0=fmaxf(mrun0,bm0), mnew1=fmaxf(mrun1,bm1);
        float corr0=ex2(mrun0-mnew0), corr1=ex2(mrun1-mnew1);
        float sp0=0.f,sp1=0.f;
        #pragma unroll
        for(int nt=0;nt<8;nt++){
            float p0=ex2(sacc[nt][0]-mnew0);
            float p1=ex2(sacc[nt][1]-mnew0);
            float p2=ex2(sacc[nt][2]-mnew1);
            float p3=ex2(sacc[nt][3]-mnew1);
            sp0+=p0+p1; sp1+=p2+p3;
            int col0=nt*8+tidg*2;
            Ps[(warp*16+groupID)*BN + col0]     = __float2bfloat16(p0);
            Ps[(warp*16+groupID)*BN + col0+1]   = __float2bfloat16(p1);
            Ps[(warp*16+groupID+8)*BN + col0]   = __float2bfloat16(p2);
            Ps[(warp*16+groupID+8)*BN + col0+1] = __float2bfloat16(p3);
        }
        sp0+=__shfl_xor_sync(0xffffffff,sp0,1); sp0+=__shfl_xor_sync(0xffffffff,sp0,2);
        sp1+=__shfl_xor_sync(0xffffffff,sp1,1); sp1+=__shfl_xor_sync(0xffffffff,sp1,2);
        lrun0=lrun0*corr0+sp0; lrun1=lrun1*corr1+sp1;
        mrun0=mnew0; mrun1=mnew1;
        #pragma unroll
        for(int nt=0;nt<16;nt++){ oacc[nt][0]*=corr0; oacc[nt][1]*=corr0; oacc[nt][2]*=corr1; oacc[nt][3]*=corr1; }

        __syncthreads();

        // ---- O += P @ V ----
        #pragma unroll
        for(int kt=0;kt<BN/16;kt++){
            unsigned p0,p1,p2,p3;
            { int m4=lane>>3,r=lane&7; int row=(m4&1)*8+r,col=(m4>>1)*8;
              ldm_x4(p0,p1,p2,p3, cvta(&Ps[(warp*16+row)*BN + kt*16 + col])); }
            #pragma unroll
            for(int nt=0;nt<16;nt++){
                unsigned v0,v1;
                { int q=lane>>3,r=lane&7,blk=q&1; int kvr=kt*16+blk*8+r; int dcol=nt*8;
                  ldm_x2t(v0,v1, cvta(&Vs[kvr*HD + dcol])); }
                mma16816(oacc[nt][0],oacc[nt][1],oacc[nt][2],oacc[nt][3],
                         p0,p1,p2,p3, v0,v1,
                         oacc[nt][0],oacc[nt][1],oacc[nt][2],oacc[nt][3]);
            }
        }
        __syncthreads();
    }

    // ---- write out ----
    float inv0=(lrun0>0.f)?1.f/lrun0:0.f, inv1=(lrun1>0.f)?1.f/lrun1:0.f;
    int grow0=q0+warp*16+groupID, grow1=q0+warp*16+groupID+8;
    #pragma unroll
    for(int nt=0;nt<16;nt++){
        int d0=nt*8+tidg*2, d1=d0+1;
        if(grow0<S){ Obase[(int64_t)grow0*HD+d0]=__float2bfloat16(oacc[nt][0]*inv0);
                     Obase[(int64_t)grow0*HD+d1]=__float2bfloat16(oacc[nt][1]*inv0); }
        if(grow1<S){ Obase[(int64_t)grow1*HD+d0]=__float2bfloat16(oacc[nt][2]*inv1);
                     Obase[(int64_t)grow1*HD+d1]=__float2bfloat16(oacc[nt][3]*inv1); }
    }
    if(tidg==0){
        if(grow0<S) LSE[(int64_t)headoff*S+grow0]=0.6931471805599453f*(mrun0+__log2f(lrun0));
        if(grow1<S) LSE[(int64_t)headoff*S+grow1]=0.6931471805599453f*(mrun1+__log2f(lrun1));
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE){
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int Bn=(int)Q.size(0), Hn=(int)Q.size(1), Sn=(int)Q.size(2), Dn=(int)Q.size(3);
    const __nv_bfloat16* Qp=static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* Kp=static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* Vp=static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* Op=static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSEp=static_cast<float*>(LSE.data_ptr());
    float scale=1.0f/sqrtf((float)Dn);

    int numQB=(Sn+BM-1)/BM;
    dim3 grid(numQB, Hn, Bn);
    dim3 block(THREADS);
    size_t smem = (size_t)BM*HD*2 + (size_t)BN*HD*2*2 + (size_t)BM*BN*2;

    cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id));
    cudaFuncSetAttribute(mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem);
    mha_kernel<<<grid,block,smem,stream>>>(Qp,Kp,Vp,Op,LSEp,Bn,Hn,Sn,scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

} // namespace mha

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha::run);