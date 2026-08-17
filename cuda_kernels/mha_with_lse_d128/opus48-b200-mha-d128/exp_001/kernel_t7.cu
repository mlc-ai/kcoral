#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s @%s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__); exit(1);} }while(0)
#define CU_CHECK(call) do { CUresult _e=(call); if(_e!=CUDA_SUCCESS){ const char*s; cuGetErrorString(_e,&s); \
  fprintf(stderr,"CU %s @%s:%d\n",s,__FILE__,__LINE__); exit(1);} }while(0)

namespace mha_d128 {
typedef __nv_bfloat16 bf;
constexpr int D=128, BM=128, BN=64, NT=256;

__device__ __forceinline__ uint32_t cvts(const void*p){return (uint32_t)__cvta_generic_to_shared(p);}
__device__ __forceinline__ uint32_t f2u(float f){return __float_as_uint(f);}
__device__ __forceinline__ float ex2(float x){float y; asm volatile("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x)); return y;}

__device__ __forceinline__ void init_bar(uint64_t*b,uint32_t c){asm volatile("mbarrier.init.shared.b64 [%0],%1;"::"r"(cvts(b)),"r"(c));}
__device__ __forceinline__ void fence_bar_init(){asm volatile("fence.mbarrier_init.release.cluster;":::"memory");}
__device__ __forceinline__ void arrive_expect(uint64_t*b,uint32_t tx){asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(cvts(b)),"r"(tx):"memory");}
__device__ __forceinline__ void bar_wait(uint64_t*b,uint32_t ph){
  asm volatile("{\n.reg .pred P;\nW%=:\nmbarrier.try_wait.parity.shared.b64 P,[%0],%1;\n@!P bra W%=;\n}\n"::"r"(cvts(b)),"r"(ph));}
__device__ __forceinline__ void fence_async(){asm volatile("fence.proxy.async;":::"memory");}

__device__ __forceinline__ void tma2d(const CUtensorMap*d,uint64_t*b,void*s,int c0,int c1){
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1,{%3,%4}],[%2];"
    ::"r"(cvts(s)),"l"((uint64_t)d),"r"(cvts(b)),"r"(c0),"r"(c1):"memory");}

__device__ __forceinline__ void tmem_alloc(uint32_t*dst,int nc){asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0],%1;"::"r"(cvts(dst)),"r"(nc));}
__device__ __forceinline__ void tmem_dealloc(uint32_t a,int nc){asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,%1;"::"r"(a),"r"(nc));}
__device__ __forceinline__ void tmem_ld4(uint32_t a,float*r0,float*r1,float*r2,float*r3){
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3},[%4];":"=f"(*r0),"=f"(*r1),"=f"(*r2),"=f"(*r3):"r"(a));}
__device__ __forceinline__ void tmem_st4(uint32_t a,uint32_t r0,uint32_t r1,uint32_t r2,uint32_t r3){
  asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0],{%1,%2,%3,%4};"::"r"(a),"r"(r0),"r"(r1),"r"(r2),"r"(r3):"memory");}
__device__ __forceinline__ void wait_ld(){asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory");}
__device__ __forceinline__ void wait_st(){asm volatile("tcgen05.wait::st.sync.aligned;":::"memory");}
__device__ __forceinline__ void fence_before(){asm volatile("tcgen05.fence::before_thread_sync;":::"memory");}
__device__ __forceinline__ void fence_after(){asm volatile("tcgen05.fence::after_thread_sync;":::"memory");}

__device__ __forceinline__ void umma(uint32_t td,uint64_t da,uint64_t db,uint32_t id,uint32_t ac){
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p,%4,0;\ntcgen05.mma.cta_group::1.kind::f16 [%0],%1,%2,%3,p;\n}\n"
    ::"r"(td),"l"(da),"l"(db),"r"(id),"r"(ac));}
__device__ __forceinline__ void commit(uint64_t*b){asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"::"r"(cvts(b)));}

__device__ __forceinline__ uint64_t mkdesc(const void*p,uint32_t lbo,uint32_t sbo){
  uint32_t a=cvts(p); uint64_t d=0;
  d|=(uint64_t)((a&0x3FFFFu)>>4);
  d|=((uint64_t)((lbo>>4)&0x3FFFu))<<16;
  d|=((uint64_t)((sbo>>4)&0x3FFFu))<<32;
  d|=(uint64_t)1<<46;
  d|=(uint64_t)((a>>7)&0x7u)<<49;
  d|=(uint64_t)2<<61;
  return d;
}
__device__ __forceinline__ uint32_t mkinstr(uint32_t M,uint32_t N,uint32_t am,uint32_t bm){
  uint32_t d=0; d|=(1u<<4); d|=(1u<<7); d|=(1u<<10); d|=(am<<15); d|=(bm<<16);
  d|=((N/8)<<17); d|=((M/16)<<24); return d;
}
__device__ __forceinline__ uint32_t pkbf(float a,float b){
  bf x=__float2bfloat16(a),y=__float2bfloat16(b); uint32_t r;
  asm("mov.b32 %0,{%1,%2};":"=r"(r):"h"(*(uint16_t*)&x),"h"(*(uint16_t*)&y)); return r;
}

__device__ __forceinline__ void issue_QK(uint32_t dst, bf* Q0,bf* Q1,bf* K0,bf* K1,uint32_t id){
  #pragma unroll
  for(int k=0;k<8;k++){
    bf* qp=(k<4)?Q0:Q1; bf* kp=(k<4)?K0:K1;
    uint64_t da=mkdesc(qp+(k&3)*16,1,1024);
    uint64_t db=mkdesc(kp+(k&3)*16,1,1024);
    umma(dst,da,db,id,k>0);
  }
}
__device__ __forceinline__ void issue_PV(uint32_t Obase, bf* P, bf* V0,bf* V1,uint32_t id,int kt){
  #pragma unroll
  for(int dt=0;dt<2;dt++){
    uint32_t Ot=Obase+dt*64; bf* vp=(dt==0)?V0:V1;
    #pragma unroll
    for(int k=0;k<4;k++){
      uint64_t da=mkdesc(P+k*16,1,1024);
      uint64_t db=mkdesc(vp+k*16*64,8192,1024);
      uint32_t ac=(k>0)?1:(kt>0?1:0);
      umma(Ot,da,db,id,ac);
    }
  }
}

__global__ __launch_bounds__(NT,1) void kern(
    const __grid_constant__ CUtensorMap dQ,
    const __grid_constant__ CUtensorMap dK,
    const __grid_constant__ CUtensorMap dV,
    bf* __restrict__ O, float* __restrict__ LSE, int S)
{
    extern __shared__ char smem[];
    uint32_t s0=cvts(smem);
    uint32_t off=((s0+1023u)&~1023u)-s0;
    char* Bp=smem+off;
    bf* QA0=(bf*)(Bp+0);     bf* QA1=(bf*)(Bp+16384);
    bf* QB0=(bf*)(Bp+32768); bf* QB1=(bf*)(Bp+49152);
    bf *K0[3],*K1[3],*V0[3],*V1[3];
    #pragma unroll
    for(int s=0;s<3;s++){ int b=65536+s*32768;
        K0[s]=(bf*)(Bp+b); K1[s]=(bf*)(Bp+b+8192);
        V0[s]=(bf*)(Bp+b+16384); V1[s]=(bf*)(Bp+b+24576); }
    bf* PA=(bf*)(Bp+161792); bf* PB=(bf*)(Bp+178176);   // after 65536+3*32768=161792
    uint64_t* bKV=(uint64_t*)(Bp+194560);   // [3]
    uint64_t* bQK=(uint64_t*)(Bp+194584);   // [2]
    uint64_t* bPV=(uint64_t*)(Bp+194600);
    uint64_t* barQ=(uint64_t*)(Bp+194608);
    uint32_t* tptr=(uint32_t*)(Bp+194616);

    int tid=threadIdx.x, warp=tid>>5;
    bool leader=(tid==0);
    int bh=blockIdx.y, q_start=blockIdx.x*256;

    if(leader){ init_bar(&bKV[0],1); init_bar(&bKV[1],1); init_bar(&bKV[2],1);
        init_bar(&bQK[0],1); init_bar(&bQK[1],1); init_bar(bPV,1); init_bar(barQ,1);
        fence_bar_init(); }
    __syncthreads();
    if(warp==0) tmem_alloc(tptr,512);
    __syncthreads();
    uint32_t tb=*tptr;
    __syncthreads();

    const float scl = rsqrtf((float)D)*1.4426950408889634f;
    const float TAU = 8.0f;
    uint32_t idQK=mkinstr(BM,BN,0,0);
    uint32_t idPV=mkinstr(BM,64,0,1);

    int num=(S+BN-1)/BN;

    int wg=(tid>=128)?1:0;
    int local=tid&127;
    // TMEM columns
    auto SA=[&](int par){return tb+(par?64:0);};
    auto SB=[&](int par){return tb+128+(par?64:0);};
    uint32_t OA=tb+256, OB=tb+384;
    uint32_t Ocol = wg?OB:OA;
    bf* Pbuf = wg?PB:PA;

    uint32_t pKV[3]={0,0,0}, pQK[2]={0,0}, pPV=0, pQ=0;

    // ---- prologue ----
    if(leader){
        arrive_expect(barQ,4*16384);
        tma2d(&dQ,barQ,QA0,0,bh*S+q_start);
        tma2d(&dQ,barQ,QA1,64,bh*S+q_start);
        tma2d(&dQ,barQ,QB0,0,bh*S+q_start+128);
        tma2d(&dQ,barQ,QB1,64,bh*S+q_start+128);
        arrive_expect(&bKV[0],4*8192);
        tma2d(&dK,&bKV[0],K0[0],0,bh*S+0); tma2d(&dK,&bKV[0],K1[0],64,bh*S+0);
        tma2d(&dV,&bKV[0],V0[0],0,bh*S+0); tma2d(&dV,&bKV[0],V1[0],64,bh*S+0);
        if(num>1){ arrive_expect(&bKV[1],4*8192);
            tma2d(&dK,&bKV[1],K0[1],0,bh*S+BN); tma2d(&dK,&bKV[1],K1[1],64,bh*S+BN);
            tma2d(&dV,&bKV[1],V0[1],0,bh*S+BN); tma2d(&dV,&bKV[1],V1[1],64,bh*S+BN); }
    }
    bar_wait(barQ,pQ&1); pQ^=1;
    bar_wait(&bKV[0],pKV[0]&1); pKV[0]^=1;
    if(leader){
        issue_QK(SA(0), QA0,QA1, K0[0],K1[0], idQK);
        issue_QK(SB(0), QB0,QB1, K0[0],K1[0], idQK);
        commit(&bQK[0]);
    }

    float m_ref=-1e30f, l_ref=0.0f;

    for(int kt=0;kt<num;kt++){
        int p=kt&1; int sp=kt%3;

        // wait S(kt) ready
        bar_wait(&bQK[p],pQK[p]&1); pQK[p]^=1;

        // issue QK(kt+1) -> overlaps softmax exp
        if(kt+1<num){
            int np=(kt+1)&1, nsp=(kt+1)%3;
            bar_wait(&bKV[nsp],pKV[nsp]&1); pKV[nsp]^=1;
            if(leader){
                issue_QK(SA(np), QA0,QA1, K0[nsp],K1[nsp], idQK);
                issue_QK(SB(np), QB0,QB1, K0[nsp],K1[nsp], idQK);
                commit(&bQK[np]);
            }
        }

        // ---- softmax(kt) ----
        fence_after();
        uint32_t Scol = wg?SB(p):SA(p);
        float sv[BN];
        #pragma unroll
        for(int c=0;c<BN;c+=4){ float a,b,cc,d; tmem_ld4(Scol+c,&a,&b,&cc,&d);
            sv[c]=a; sv[c+1]=b; sv[c+2]=cc; sv[c+3]=d; }
        wait_ld();

        int valid=S-kt*BN; if(valid>BN) valid=BN;
        float mx=-1e30f;
        #pragma unroll
        for(int j=0;j<BN;j++){ sv[j]=(j<valid)?sv[j]*scl:-1e30f; mx=fmaxf(mx,sv[j]); }
        float new_ref=fmaxf(m_ref,mx);
        bool need=(new_ref-m_ref)>TAU;
        bool do_r=__any_sync(0xffffffffu,need);
        float alpha=1.0f;
        if(do_r){ alpha=ex2(m_ref-new_ref); m_ref=new_ref; }
        float rs=0.0f;
        #pragma unroll
        for(int j=0;j<BN;j++){ float pp=ex2(sv[j]-m_ref); rs+=pp; sv[j]=pp; }
        l_ref=l_ref*alpha+rs;

        // wait PV(kt-1): O free, P consumed
        if(kt>0){ bar_wait(bPV,pPV&1); pPV^=1; }

        // prefetch KV(kt+2) into freed stage
        if(kt+2<num && leader){
            int psp=(kt+2)%3; uint64_t* b=&bKV[psp];
            int coord=bh*S+(kt+2)*BN;
            arrive_expect(b,4*8192);
            tma2d(&dK,b,K0[psp],0,coord); tma2d(&dK,b,K1[psp],64,coord);
            tma2d(&dV,b,V0[psp],0,coord); tma2d(&dV,b,V1[psp],64,coord);
        }

        // conditional rescale of O (warp-collective)
        if(kt>0 && do_r){
            #pragma unroll
            for(int c=0;c<128;c+=4){ float o0,o1,o2,o3; tmem_ld4(Ocol+c,&o0,&o1,&o2,&o3); wait_ld();
                tmem_st4(Ocol+c,f2u(o0*alpha),f2u(o1*alpha),f2u(o2*alpha),f2u(o3*alpha)); }
            wait_st();
        }

        // store P (128B swizzle, K-major)
        #pragma unroll
        for(int g=0;g<8;g++){
            int phys=((local&7)^g)*8;
            bf* dst=Pbuf+local*64+phys;
            *(int4*)dst=make_int4(pkbf(sv[g*8],sv[g*8+1]),pkbf(sv[g*8+2],sv[g*8+3]),
                                  pkbf(sv[g*8+4],sv[g*8+5]),pkbf(sv[g*8+6],sv[g*8+7]));
        }
        fence_before();
        __syncthreads();
        fence_async();

        // PV(kt) accumulate into O
        if(leader){
            fence_after();
            issue_PV(OA, PA, V0[sp],V1[sp], idPV, kt);
            issue_PV(OB, PB, V0[sp],V1[sp], idPV, kt);
            commit(bPV);
        }
    }
    bar_wait(bPV,pPV&1); pPV^=1;

    // ---- finalize ----
    fence_after();
    int gq=q_start + (wg?128:0) + local;
    float inv=(l_ref>0.0f)?1.0f/l_ref:0.0f;
    #pragma unroll
    for(int base=0;base<128;base+=32){
        float t[32];
        #pragma unroll
        for(int i=0;i<32;i+=4) tmem_ld4(Ocol+base+i,&t[i],&t[i+1],&t[i+2],&t[i+3]);
        wait_ld();
        if(gq<S){
            int4* o4=(int4*)(O+((int64_t)(bh*S+gq))*D+base);
            #pragma unroll
            for(int i=0;i<32;i+=8){
                o4[i/8]=make_int4(pkbf(t[i]*inv,t[i+1]*inv),pkbf(t[i+2]*inv,t[i+3]*inv),
                                  pkbf(t[i+4]*inv,t[i+5]*inv),pkbf(t[i+6]*inv,t[i+7]*inv));
            }
        }
    }
    if(gq<S) LSE[(int64_t)bh*S+gq]=m_ref*0.69314718056f+logf(l_ref);

    __syncthreads();
    if(warp==0) tmem_dealloc(tb,512);
}

static CUresult mktma(CUtensorMap*d,void*ptr,uint64_t BHS,uint32_t bin,uint32_t bout){
    cuuint64_t gd[2]={(cuuint64_t)D,(cuuint64_t)BHS};
    cuuint64_t gs[1]={(cuuint64_t)D*2};
    cuuint32_t bd[2]={bin,bout};
    cuuint32_t es[2]={1,1};
    return cuTensorMapEncodeTiled(d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,ptr,gd,gs,bd,es,
        CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE){
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int Bd=(int)Q.size(0), Hd=(int)Q.size(1), Sd=(int)Q.size(2);
    bf* Qp=(bf*)Q.data_ptr(); bf* Kp=(bf*)K.data_ptr(); bf* Vp=(bf*)V.data_ptr();
    bf* Op=(bf*)O.data_ptr(); float* Lp=(float*)LSE.data_ptr();
    uint64_t BHS=(uint64_t)Bd*Hd*Sd;

    CUtensorMap dQ,dK,dV;
    CU_CHECK(mktma(&dQ,Qp,BHS,64,128));
    CU_CHECK(mktma(&dK,Kp,BHS,64,64));
    CU_CHECK(mktma(&dV,Vp,BHS,64,64));

    dim3 grid((Sd+255)/256, Bd*Hd);
    size_t smem=200704;
    static bool set=false;
    if(!set){ CUDA_CHECK(cudaFuncSetAttribute((const void*)kern,
        cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem)); set=true; }
    cudaStream_t stream=(cudaStream_t)TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id);
    kern<<<grid,NT,smem,stream>>>(dQ,dK,dV,Op,Lp,Sd);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_d128::run);

}  // namespace mha_d128