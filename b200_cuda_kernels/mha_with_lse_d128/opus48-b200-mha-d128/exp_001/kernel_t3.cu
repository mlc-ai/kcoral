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

constexpr int D=128, BM=128, BN=64, NT=128;

__device__ __forceinline__ uint32_t cvts(const void*p){return (uint32_t)__cvta_generic_to_shared(p);}
__device__ __forceinline__ uint32_t f2u(float f){return __float_as_uint(f);}

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
  __nv_bfloat16 x=__float2bfloat16(a),y=__float2bfloat16(b); uint32_t r;
  asm("mov.b32 %0,{%1,%2};":"=r"(r):"h"(*(uint16_t*)&x),"h"(*(uint16_t*)&y)); return r;
}

__global__ __launch_bounds__(NT,1) void kern(
    const __grid_constant__ CUtensorMap dQ,
    const __grid_constant__ CUtensorMap dK,
    const __grid_constant__ CUtensorMap dV,
    __nv_bfloat16* __restrict__ O, float* __restrict__ LSE, int S)
{
    extern __shared__ char smem[];
    uint32_t s0=cvts(smem);
    uint32_t off=((s0+1023u)&~1023u)-s0;
    char* base=smem+off;
    __nv_bfloat16* Qs[2]={(__nv_bfloat16*)(base+0),(__nv_bfloat16*)(base+16384)};
    __nv_bfloat16* Ks[2]={(__nv_bfloat16*)(base+32768),(__nv_bfloat16*)(base+40960)};
    __nv_bfloat16* Vs[2]={(__nv_bfloat16*)(base+49152),(__nv_bfloat16*)(base+57344)};
    __nv_bfloat16* Ps=(__nv_bfloat16*)(base+65536);
    uint64_t* barT=(uint64_t*)(base+81920);
    uint64_t* barM=(uint64_t*)(base+81928);
    uint32_t* tptr=(uint32_t*)(base+81936);

    int tid=threadIdx.x, warp=tid>>5;
    bool leader=(tid==0);
    int bh=blockIdx.y, q_start=blockIdx.x*BM;

    if(leader){ init_bar(barT,1); init_bar(barM,1); fence_bar_init(); }
    __syncthreads();
    if(warp==0) tmem_alloc(tptr,256);
    __syncthreads();
    uint32_t tb=*tptr;          // TMEM base
    uint32_t Ocol=tb+64;        // O at col 64
    __syncthreads();

    float scale=rsqrtf((float)D);
    uint32_t idQK=mkinstr(BM,BN,0,0);
    uint32_t idPV=mkinstr(BM,64,0,1);
    uint32_t phT=0, phM=0;

    // load Q
    if(leader){ arrive_expect(barT,2*16384);
        tma2d(&dQ,barT,Qs[0],0,bh*S+q_start);
        tma2d(&dQ,barT,Qs[1],64,bh*S+q_start); }
    bar_wait(barT,phT&1); phT++;
    __syncthreads();

    float m_old=-1e30f, l_old=0.0f;
    int num_kv=(S+BN-1)/BN;

    for(int kt=0;kt<num_kv;kt++){
        int kv_start=kt*BN;
        int valid=min(BN,S-kv_start);
        if(leader){ arrive_expect(barT,4*8192);
            tma2d(&dK,barT,Ks[0],0,bh*S+kv_start);
            tma2d(&dK,barT,Ks[1],64,bh*S+kv_start);
            tma2d(&dV,barT,Vs[0],0,bh*S+kv_start);
            tma2d(&dV,barT,Vs[1],64,bh*S+kv_start); }
        bar_wait(barT,phT&1); phT++;
        __syncthreads();

        // QK^T -> S (TMEM cols 0..63)
        if(leader){
            #pragma unroll
            for(int k=0;k<8;k++){
                uint64_t da=mkdesc(Qs[k>>2]+(k&3)*16,1,1024);
                uint64_t db=mkdesc(Ks[k>>2]+(k&3)*16,1,1024);
                umma(tb,da,db,idQK,k>0);
            }
            commit(barM);
        }
        bar_wait(barM,phM&1); phM++;
        __syncthreads();

        // softmax: thread=row
        fence_after();
        float sv[BN];
        #pragma unroll
        for(int c=0;c<BN;c+=4){ float r0,r1,r2,r3; tmem_ld4(tb+c,&r0,&r1,&r2,&r3);
            sv[c]=r0; sv[c+1]=r1; sv[c+2]=r2; sv[c+3]=r3; }
        wait_ld();
        float mx=-1e30f;
        #pragma unroll
        for(int j=0;j<BN;j++){ sv[j]=(j<valid)?sv[j]*scale:-1e30f; mx=fmaxf(mx,sv[j]); }
        float m_new=fmaxf(m_old,mx);
        float alpha=__expf(m_old-m_new);
        float rs=0.0f;
        #pragma unroll
        for(int j=0;j<BN;j++){ float p=__expf(sv[j]-m_new); rs+=p; sv[j]=p; }
        float l_new=l_old*alpha+rs;

        // rescale O
        if(kt>0){
            for(int c=0;c<128;c+=4){ float o0,o1,o2,o3; tmem_ld4(Ocol+c,&o0,&o1,&o2,&o3); wait_ld();
                tmem_st4(Ocol+c,f2u(o0*alpha),f2u(o1*alpha),f2u(o2*alpha),f2u(o3*alpha)); }
            wait_st();
        }
        // write P (128B swizzle, K-major)
        #pragma unroll
        for(int g=0;g<8;g++){
            int phys=((tid&7)^g)*8;
            __nv_bfloat16* dst=Ps+tid*64+phys;
            *(int4*)dst=make_int4(pkbf(sv[g*8],sv[g*8+1]),pkbf(sv[g*8+2],sv[g*8+3]),
                                  pkbf(sv[g*8+4],sv[g*8+5]),pkbf(sv[g*8+6],sv[g*8+7]));
        }
        m_old=m_new; l_old=l_new;
        fence_before();
        __syncthreads();
        fence_async();

        // O += P@V
        if(leader){
            fence_after();
            #pragma unroll
            for(int dt=0;dt<2;dt++){
                uint32_t Ot=Ocol+dt*64;
                #pragma unroll
                for(int k=0;k<4;k++){
                    uint64_t da=mkdesc(Ps+k*16,1,1024);
                    uint64_t db=mkdesc(Vs[dt]+k*16*64,8192,1024);
                    uint32_t ac=(k>0)?1:(kt>0?1:0);
                    umma(Ot,da,db,idPV,ac);
                }
            }
            commit(barM);
        }
        bar_wait(barM,phM&1); phM++;
        __syncthreads();
    }

    // finalize
    fence_after();
    int gq=q_start+tid;
    float inv=(l_old>0.0f)?1.0f/l_old:0.0f;
    for(int c=0;c<128;c+=4){ float o0,o1,o2,o3; tmem_ld4(Ocol+c,&o0,&o1,&o2,&o3); wait_ld();
        if(gq<S){ __nv_bfloat16* out=O+((int64_t)(bh*S+gq))*D+c;
            out[0]=__float2bfloat16(o0*inv); out[1]=__float2bfloat16(o1*inv);
            out[2]=__float2bfloat16(o2*inv); out[3]=__float2bfloat16(o3*inv); } }
    if(gq<S) LSE[(int64_t)bh*S+gq]=m_old+logf(l_old);

    __syncthreads();
    if(warp==0) tmem_dealloc(tb,256);
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
    int B=(int)Q.size(0), H=(int)Q.size(1), S=(int)Q.size(2);
    __nv_bfloat16* Qp=(__nv_bfloat16*)Q.data_ptr();
    __nv_bfloat16* Kp=(__nv_bfloat16*)K.data_ptr();
    __nv_bfloat16* Vp=(__nv_bfloat16*)V.data_ptr();
    __nv_bfloat16* Op=(__nv_bfloat16*)O.data_ptr();
    float* Lp=(float*)LSE.data_ptr();
    uint64_t BHS=(uint64_t)B*H*S;

    CUtensorMap dQ,dK,dV;
    CU_CHECK(mktma(&dQ,Qp,BHS,64,128));
    CU_CHECK(mktma(&dK,Kp,BHS,64,64));
    CU_CHECK(mktma(&dV,Vp,BHS,64,64));

    dim3 grid((S+BM-1)/BM, B*H);
    size_t smem=90*1024;
    static bool set=false;
    if(!set){ CUDA_CHECK(cudaFuncSetAttribute((const void*)kern,
        cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem)); set=true; }
    cudaStream_t stream=(cudaStream_t)TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id);
    kern<<<grid,NT,smem,stream>>>(dQ,dK,dV,Op,Lp,S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_d128::run);

}