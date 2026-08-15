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
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_causal {

constexpr int D=128, BM=64, BN=64, WARPS=4, THREADS=32*WARPS;
constexpr int NKC=D/16;   // 8  QK k-chunks
constexpr int NKG=BN/16;  // 4  key-groups (16 keys each)
constexpr int NNT=BN/8;   // 8  n-tiles
constexpr int KK=BN/16;   // 4  PV key-chunks
constexpr int DBK=D/16;   // 8  d-blocks (16 d each)
constexpr int NDT=D/8;    // 16 d-tiles
constexpr unsigned FULL=0xffffffffu;

__device__ __forceinline__ float ex2(float x){ float y; asm volatile("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x)); return y; }
__device__ __forceinline__ uint32_t su32(const void* p){ return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void ldm4(uint32_t&r0,uint32_t&r1,uint32_t&r2,uint32_t&r3,uint32_t a){
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
        :"=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3):"r"(a));
}
__device__ __forceinline__ void ldm4t(uint32_t&r0,uint32_t&r1,uint32_t&r2,uint32_t&r3,uint32_t a){
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];"
        :"=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3):"r"(a));
}
__device__ __forceinline__ void mma(float&c0,float&c1,float&c2,float&c3,
        uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1){
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
      :"+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
      :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
__device__ __forceinline__ void cpa16(uint32_t dst,const void* src,int bytes){
    asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;\n"::"r"(dst),"l"(src),"r"(bytes):"memory");
}

__global__ __launch_bounds__(THREADS) void attn_kernel(
    const __nv_bfloat16* __restrict__ Q,const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,__nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,int B,int H,int S,float scale_log2)
{
    extern __shared__ __align__(16) unsigned char sraw[];
    __nv_bfloat16* Qs=(__nv_bfloat16*)sraw;
    __nv_bfloat16* Ks=Qs+BM*D;
    __nv_bfloat16* Vs=Ks+2*BN*D;
    __nv_bfloat16* Ps=Vs+2*BN*D;

    const int warp=threadIdx.x>>5, lane=threadIdx.x&31;
    const int gid=lane>>2, tig=lane&3;
    const int h=blockIdx.y, b=blockIdx.z, q0=blockIdx.x*BM;

    const int64_t bh=(int64_t)(b*H+h);
    const __nv_bfloat16* Qg=Q+bh*S*D;
    const __nv_bfloat16* Kg=K+bh*S*D;
    const __nv_bfloat16* Vg=V+bh*S*D;
    __nv_bfloat16* Og=O+bh*S*D;
    float* LSEg=LSE+bh*S;

    // load Q tile
    for(int i=threadIdx.x;i<(BM*D)/8;i+=THREADS){
        int e=i*8,r=e>>7,c=e&127,gr=q0+r;
        int4 v=(gr<S)?*reinterpret_cast<const int4*>(Qg+(int64_t)gr*D+c):make_int4(0,0,0,0);
        *reinterpret_cast<int4*>(&Qs[r*D+c])=v;
    }

    const int last_key=min(S-1,q0+BM-1);
    const int ntiles=last_key/BN+1;

    auto load_tile=[&](int buf,int kv_start){
        __nv_bfloat16* kb=Ks+buf*BN*D;
        __nv_bfloat16* vb=Vs+buf*BN*D;
        for(int i=threadIdx.x;i<(BN*D)/8;i+=THREADS){
            int e=i*8,key=e>>7,col=e&127,gk=kv_start+key;
            int bytes=(gk<S)?16:0, sk=(gk<S)?gk:(S-1);
            cpa16(su32(&kb[key*D+col]),Kg+(int64_t)sk*D+col,bytes);
            cpa16(su32(&vb[key*D+col]),Vg+(int64_t)sk*D+col,bytes);
        }
    };

    // prologue
    load_tile(0,0);
    asm volatile("cp.async.commit_group;\n":::"memory");
    __syncthreads();

    float Oacc[NDT][4];
    #pragma unroll
    for(int dt=0;dt<NDT;dt++){Oacc[dt][0]=0;Oacc[dt][1]=0;Oacc[dt][2]=0;Oacc[dt][3]=0;}
    float mA=-INFINITY,mB=-INFINITY,lA=0.f,lB=0.f;

    const bool wact=(q0+warp*16)<S;
    const int wmax=q0+warp*16+15;
    const int grA=q0+warp*16+gid, grB=grA+8;

    for(int t=0;t<ntiles;t++){
        int cur=t&1, kv_start=t*BN;
        if(t+1<ntiles){
            load_tile((t+1)&1,(t+1)*BN);
            asm volatile("cp.async.commit_group;\n":::"memory");
            asm volatile("cp.async.wait_group 1;\n":::"memory");
        } else {
            asm volatile("cp.async.wait_group 0;\n":::"memory");
        }
        __syncthreads();

        if(wact && kv_start<=wmax){
            __nv_bfloat16* Kb=Ks+cur*BN*D;
            __nv_bfloat16* Vb=Vs+cur*BN*D;

            // ---- QK^T ----
            float acc[NNT][4];
            #pragma unroll
            for(int nt=0;nt<NNT;nt++){acc[nt][0]=0;acc[nt][1]=0;acc[nt][2]=0;acc[nt][3]=0;}
            #pragma unroll
            for(int kc=0;kc<NKC;kc++){
                uint32_t qa0,qa1,qa2,qa3;
                ldm4(qa0,qa1,qa2,qa3, su32(&Qs[(warp*16+lane%16)*D + kc*16 + (lane/16)*8]));
                #pragma unroll
                for(int kg=0;kg<NKG;kg++){
                    uint32_t k0,k1,k2,k3;
                    ldm4(k0,k1,k2,k3, su32(&Kb[(kg*16+lane%16)*D + kc*16 + (lane/16)*8]));
                    mma(acc[2*kg][0],acc[2*kg][1],acc[2*kg][2],acc[2*kg][3], qa0,qa1,qa2,qa3,k0,k2);
                    mma(acc[2*kg+1][0],acc[2*kg+1][1],acc[2*kg+1][2],acc[2*kg+1][3], qa0,qa1,qa2,qa3,k1,k3);
                }
            }
            #pragma unroll
            for(int nt=0;nt<NNT;nt++){acc[nt][0]*=scale_log2;acc[nt][1]*=scale_log2;acc[nt][2]*=scale_log2;acc[nt][3]*=scale_log2;}

            // ---- causal mask ----
            #pragma unroll
            for(int nt=0;nt<NNT;nt++){
                int k0=kv_start+nt*8+tig*2, k1=k0+1;
                if(k0>grA||k0>=S) acc[nt][0]=-INFINITY;
                if(k1>grA||k1>=S) acc[nt][1]=-INFINITY;
                if(k0>grB||k0>=S) acc[nt][2]=-INFINITY;
                if(k1>grB||k1>=S) acc[nt][3]=-INFINITY;
            }

            // ---- online softmax ----
            float lmA=-INFINITY,lmB=-INFINITY;
            #pragma unroll
            for(int nt=0;nt<NNT;nt++){
                lmA=fmaxf(lmA,fmaxf(acc[nt][0],acc[nt][1]));
                lmB=fmaxf(lmB,fmaxf(acc[nt][2],acc[nt][3]));
            }
            lmA=fmaxf(lmA,__shfl_xor_sync(FULL,lmA,1)); lmA=fmaxf(lmA,__shfl_xor_sync(FULL,lmA,2));
            lmB=fmaxf(lmB,__shfl_xor_sync(FULL,lmB,1)); lmB=fmaxf(lmB,__shfl_xor_sync(FULL,lmB,2));
            float mnA=fmaxf(mA,lmA), mnB=fmaxf(mB,lmB);
            float cA=ex2(mA-mnA), cB=ex2(mB-mnB);
            #pragma unroll
            for(int dt=0;dt<NDT;dt++){Oacc[dt][0]*=cA;Oacc[dt][1]*=cA;Oacc[dt][2]*=cB;Oacc[dt][3]*=cB;}
            lA*=cA; lB*=cB;

            float psA=0,psB=0;
            #pragma unroll
            for(int nt=0;nt<NNT;nt++){
                float p0=ex2(acc[nt][0]-mnA),p1=ex2(acc[nt][1]-mnA);
                float p2=ex2(acc[nt][2]-mnB),p3=ex2(acc[nt][3]-mnB);
                psA+=p0+p1; psB+=p2+p3;
                int key=nt*8+tig*2;
                Ps[(warp*16+gid)*BN+key  ]=__float2bfloat16(p0);
                Ps[(warp*16+gid)*BN+key+1]=__float2bfloat16(p1);
                Ps[(warp*16+gid+8)*BN+key  ]=__float2bfloat16(p2);
                Ps[(warp*16+gid+8)*BN+key+1]=__float2bfloat16(p3);
            }
            psA+=__shfl_xor_sync(FULL,psA,1); psA+=__shfl_xor_sync(FULL,psA,2);
            psB+=__shfl_xor_sync(FULL,psB,1); psB+=__shfl_xor_sync(FULL,psB,2);
            lA+=psA; lB+=psB; mA=mnA; mB=mnB;
            __syncwarp();

            // ---- P @ V ----
            #pragma unroll
            for(int kk=0;kk<KK;kk++){
                uint32_t pa0,pa1,pa2,pa3;
                ldm4(pa0,pa1,pa2,pa3, su32(&Ps[(warp*16+lane%16)*BN + kk*16 + (lane/16)*8]));
                #pragma unroll
                for(int db=0;db<DBK;db++){
                    uint32_t v0,v1,v2,v3;
                    ldm4t(v0,v1,v2,v3, su32(&Vb[(kk*16+lane%16)*D + db*16 + (lane/16)*8]));
                    mma(Oacc[2*db][0],Oacc[2*db][1],Oacc[2*db][2],Oacc[2*db][3], pa0,pa1,pa2,pa3,v0,v1);
                    mma(Oacc[2*db+1][0],Oacc[2*db+1][1],Oacc[2*db+1][2],Oacc[2*db+1][3], pa0,pa1,pa2,pa3,v2,v3);
                }
            }
            __syncwarp();
        }
        __syncthreads();
    }

    if(!wact) return;
    float invA=1.0f/lA, invB=1.0f/lB;
    #pragma unroll
    for(int dt=0;dt<NDT;dt++){
        int col=dt*8+tig*2;
        if(grA<S){
            Og[(int64_t)grA*D+col  ]=__float2bfloat16(Oacc[dt][0]*invA);
            Og[(int64_t)grA*D+col+1]=__float2bfloat16(Oacc[dt][1]*invA);
        }
        if(grB<S){
            Og[(int64_t)grB*D+col  ]=__float2bfloat16(Oacc[dt][2]*invB);
            Og[(int64_t)grB*D+col+1]=__float2bfloat16(Oacc[dt][3]*invB);
        }
    }
    if(tig==0){
        const float LN2=0.6931471805599453f;
        if(grA<S) LSEg[grA]=mA*LN2+logf(lA);
        if(grB<S) LSEg[grB]=mB*LN2+logf(lB);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int Bv=(int)Q.size(0),Hv=(int)Q.size(1),Sv=(int)Q.size(2),Dv=(int)Q.size(3);
    float scale=1.0f/sqrtf((float)Dv);
    float scale_log2=scale*1.4426950408889634f;

    dim3 grid((Sv+BM-1)/BM,Hv,Bv);
    dim3 block(THREADS);
    size_t shmem=(size_t)(BM*D + 2*BN*D + 2*BN*D + BM*BN)*sizeof(__nv_bfloat16);
    CUDA_CHECK(cudaFuncSetAttribute(attn_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)shmem));

    cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id));
    attn_kernel<<<grid,block,shmem,stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<__nv_bfloat16*>(O.data_ptr()),
        static_cast<float*>(LSE.data_ptr()),
        Bv,Hv,Sv,scale_log2);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_causal::run);

}  // namespace mha_causal