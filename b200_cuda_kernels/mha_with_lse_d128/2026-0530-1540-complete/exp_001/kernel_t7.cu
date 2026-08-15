#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__); exit(1);} }while(0)
#define CU_CHECK(call) do { CUresult _e=(call); if(_e!=CUDA_SUCCESS){ const char* s; cuGetErrorString(_e,&s); \
  fprintf(stderr,"CU error %s at %s:%d\n",s,__FILE__,__LINE__); exit(1);} }while(0)

namespace mha_d128 {

constexpr int D=128, BM=128, BN=128, THREADS=256;
constexpr int CHUNK = 128*64;
constexpr int NPOLY = 64;
#define NEG (-1e30f)
#define TAU2 (8.0f)
#define LOG2E (1.4426950408889634f)
#define LN2   (0.6931471805599453f)

__device__ __forceinline__ int swz_off(int row,int col){
    int xi=col>>3, e=col&7; int pxi=(row&7)^xi;
    return row*64 + pxi*8 + e;
}
__device__ __forceinline__ uint64_t make_desc(const void* ptr,uint32_t lbo,uint32_t sbo){
    uint64_t d=0; uint32_t addr=(uint32_t)__cvta_generic_to_shared(ptr);
    d |= (uint64_t)((addr&0x3FFFF)>>4);
    d |= (uint64_t)((lbo&0x3FFFF)>>4)<<16;
    d |= (uint64_t)((sbo&0x3FFFF)>>4)<<32;
    d |= (uint64_t)1<<46; d |= (uint64_t)2<<61;
    return d;
}
__device__ __forceinline__ uint32_t make_idesc(int M,int N,int aT,int bT){
    uint32_t d=0; d|=(1u<<4); d|=(1u<<7); d|=(1u<<10);
    d|=((uint32_t)aT<<15); d|=((uint32_t)bT<<16);
    d|=(((uint32_t)N>>3)<<17); d|=(((uint32_t)M>>4)<<24);
    return d;
}
__device__ __forceinline__ void tmem_alloc_cg1(uint32_t* dst,int n){
    uint32_t a=(uint32_t)__cvta_generic_to_shared(dst);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"::"r"(a),"r"(n));
}
__device__ __forceinline__ void tmem_relinquish_cg1(){ asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"); }
__device__ __forceinline__ void tmem_dealloc_cg1(uint32_t addr,int n){
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"::"r"(addr),"r"(n));
}
__device__ __forceinline__ void umma_cg1(uint32_t td,uint64_t da,uint64_t db,uint32_t id,uint32_t ac){
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p,%4,0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        ::"r"(td),"l"(da),"l"(db),"r"(id),"r"(ac));
}
__device__ __forceinline__ void umma_commit_cg1(uint64_t* bar){
    uint32_t a=(uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"::"r"(a));
}
__device__ __forceinline__ void tmem_ld4(uint32_t a,uint32_t&r0,uint32_t&r1,uint32_t&r2,uint32_t&r3){
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        :"=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3):"r"(a));
}
__device__ __forceinline__ void tmem_st4(uint32_t a,uint32_t r0,uint32_t r1,uint32_t r2,uint32_t r3){
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
        ::"r"(a),"r"(r0),"r"(r1),"r"(r2),"r"(r3));
}
__device__ __forceinline__ void wait_ld(){ asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory"); }
__device__ __forceinline__ void wait_st(){ asm volatile("tcgen05.wait::st.sync.aligned;":::"memory"); }
__device__ __forceinline__ void fence_after(){ asm volatile("tcgen05.fence::after_thread_sync;":::"memory"); }
__device__ __forceinline__ void fence_before(){ asm volatile("tcgen05.fence::before_thread_sync;":::"memory"); }
__device__ __forceinline__ void bar_init(uint64_t* b,uint32_t c){
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(c));
}
__device__ __forceinline__ void bar_fence_init(){ asm volatile("fence.mbarrier_init.release.cluster;":::"memory"); }
__device__ __forceinline__ void bar_arrive_tx(uint64_t* b,uint32_t tx){
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        ::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(tx):"memory");
}
__device__ __forceinline__ void bar_wait(uint64_t* b,uint32_t ph){
    asm volatile("{\n.reg .pred P;\nW%=:\n mbarrier.try_wait.parity.shared.b64 P,[%0],%1;\n"
        "@!P bra W%=;\n}\n"::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(ph));
}
__device__ __forceinline__ void fence_async(){ asm volatile("fence.proxy.async.shared::cta;":::"memory"); }
__device__ __forceinline__ void nbar(int id,int cnt){ asm volatile("barrier.sync.aligned %0,%1;"::"r"(id),"r"(cnt)); }
__device__ __forceinline__ void tma_2d(const CUtensorMap* d,uint64_t* bar,void* smem,int c0,int c1){
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3,%4}], [%2];"
        ::"r"((uint32_t)__cvta_generic_to_shared(smem)),"l"((uint64_t)d),
          "r"((uint32_t)__cvta_generic_to_shared(bar)),"r"(c0),"r"(c1):"memory");
}
__device__ __forceinline__ void load_blk(const CUtensorMap* d,uint64_t* bar,__nv_bfloat16* buf,int c1){
    tma_2d(d,bar,&buf[0],0,c1); tma_2d(d,bar,&buf[CHUNK],64,c1);
}
__device__ __forceinline__ float ex2_mufu(float x){ float y; asm volatile("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x)); return y; }
__device__ __forceinline__ float ex2_poly(float x){
    x=fmaxf(x,-120.f); float n=rintf(x); float f=x-n;
    float p=0.0096181f; p=fmaf(p,f,0.0555041f); p=fmaf(p,f,0.2402265f);
    p=fmaf(p,f,0.6931472f); p=fmaf(p,f,1.0f);
    int ni=(int)n; ni=ni<-127?-127:ni;
    return p*__uint_as_float((uint32_t)(ni+127)<<23);
}

__global__ __launch_bounds__(THREADS) void attn_kernel(
    const __grid_constant__ CUtensorMap descQ,
    const __grid_constant__ CUtensorMap descK,
    const __grid_constant__ CUtensorMap descV,
    __nv_bfloat16* __restrict__ O, float* __restrict__ LSE,
    int B,int H,int S,int num_kv,float scale)
{
    int bx=blockIdx.x, h=blockIdx.y, b=blockIdx.z;
    int tid=threadIdx.x;
    int wg=tid>>7, wtid=tid&127, lwarp=wtid>>5, warpAll=tid>>5;
    uint32_t lane_off=(uint32_t)(lwarp*32)<<16;
    long rowbase=(long)(b*H+h)*S;
    float scale2=scale*LOG2E;

    int qtileA=bx*2, qtileB=bx*2+1;
    int qtile_wg = wg? qtileB:qtileA;

    __shared__ __align__(8) uint64_t s_bar[7];
    __shared__ uint32_t s_tmem[4];
    uint64_t* bar_q=&s_bar[0]; uint64_t* bar_kL=&s_bar[1]; uint64_t* bar_vL=&s_bar[2];
    uint64_t* bar_qkA=&s_bar[3]; uint64_t* bar_qkB=&s_bar[4];
    uint64_t* bar_pvA=&s_bar[5]; uint64_t* bar_pvB=&s_bar[6];
    uint64_t* bar_qk = wg? bar_qkB:bar_qkA;
    uint64_t* bar_pv = wg? bar_pvB:bar_pvA;

    extern __shared__ char smem_raw[];
    uint32_t raw=(uint32_t)__cvta_generic_to_shared(smem_raw);
    uint32_t pad=((-raw)&1023u);
    __nv_bfloat16* base=(__nv_bfloat16*)(smem_raw+pad);
    __nv_bfloat16* Qs_A=base;
    __nv_bfloat16* Qs_B=base+2*CHUNK;
    __nv_bfloat16* Ks  =base+4*CHUNK;
    __nv_bfloat16* Vs  =base+6*CHUNK;
    __nv_bfloat16* Ps_A=base+8*CHUNK;
    __nv_bfloat16* Ps_B=base+10*CHUNK;
    __nv_bfloat16* Ps_wg = wg? Ps_B:Ps_A;

    if(tid==0){
        bar_init(bar_q,1); bar_init(bar_kL,1); bar_init(bar_vL,1);
        bar_init(bar_qkA,1); bar_init(bar_qkB,1);
        bar_init(bar_pvA,1); bar_init(bar_pvB,1);
        bar_fence_init();
    }
    if(warpAll==0){ tmem_alloc_cg1(s_tmem,512); tmem_relinquish_cg1(); }
    __syncthreads();
    uint32_t tb=s_tmem[0];
    uint32_t Sbase_wg = wg? tb+128 : tb;
    uint32_t Obase_wg = wg? tb+384 : tb+256;

    uint32_t idesc_qk=make_idesc(128,128,0,0);
    uint32_t idesc_pv=make_idesc(128,64,0,1);

    // prologue: Q tiles, K[0], V[0]
    if(tid==0){
        bar_arrive_tx(bar_q, 4*CHUNK*2);
        load_blk(&descQ,bar_q,Qs_A,(int)(rowbase+qtileA*128));
        load_blk(&descQ,bar_q,Qs_B,(int)(rowbase+qtileB*128));
        bar_arrive_tx(bar_kL, 2*CHUNK*2);
        load_blk(&descK,bar_kL,Ks,(int)(rowbase+0*128));
        bar_arrive_tx(bar_vL, 2*CHUNK*2);
        load_blk(&descV,bar_vL,Vs,(int)(rowbase+0*128));
    }
    bar_wait(bar_q,0);
    __syncthreads();

    float m_ref=NEG, l=0.f;
    int ph_k=0, ph_v=0, ph_qk=0, ph_pv=0;

    for(int kv=0; kv<num_kv; kv++){
        // ---- QK (tid0) : reads K[kv] from Ks ----
        if(tid==0){
            bar_wait(bar_kL, ph_k);
            #pragma unroll
            for(int kb=0;kb<8;kb++){
                int ch=kb>>2, sl=kb&3;
                uint64_t da=make_desc(&Qs_A[ch*CHUNK+sl*16],0,1024);
                uint64_t db=make_desc(&Ks[ch*CHUNK+sl*16],0,1024);
                umma_cg1(tb,da,db,idesc_qk, kb==0?0:1);
            }
            umma_commit_cg1(bar_qkA);
            #pragma unroll
            for(int kb=0;kb<8;kb++){
                int ch=kb>>2, sl=kb&3;
                uint64_t da=make_desc(&Qs_B[ch*CHUNK+sl*16],0,1024);
                uint64_t db=make_desc(&Ks[ch*CHUNK+sl*16],0,1024);
                umma_cg1(tb+128,da,db,idesc_qk, kb==0?0:1);
            }
            umma_commit_cg1(bar_qkB);
        }

        // ---- softmax (per warpgroup) ----
        bar_wait(bar_qk,ph_qk);
        fence_after();

        float s[128];
        #pragma unroll
        for(int blk=0;blk<4;blk++){
            uint32_t U[32];
            #pragma unroll
            for(int j=0;j<32;j+=4) tmem_ld4(Sbase_wg+lane_off+blk*32+j,U[j],U[j+1],U[j+2],U[j+3]);
            wait_ld();
            #pragma unroll
            for(int j=0;j<32;j++){ int c=blk*32+j; int k0=kv*128+c;
                s[c]=(k0<S)?__uint_as_float(U[j])*scale2:NEG; }
        }
        __syncthreads();   // sync1: both QK outputs read -> K[kv] free
        if(tid==0 && kv+1<num_kv){
            bar_arrive_tx(bar_kL, 2*CHUNK*2);
            load_blk(&descK,bar_kL,Ks,(int)(rowbase+(kv+1)*128));   // prefetch K[kv+1]
        }

        float tmax=NEG;
        #pragma unroll
        for(int c=0;c<128;c++) tmax=fmaxf(tmax,s[c]);

        float corr,m_new; bool do_resc;
        if(kv==0){ m_new=tmax; corr=1.f; do_resc=false; }
        else{
            bool jump=(tmax-m_ref)>TAU2;
            bool any=__any_sync(0xffffffffu,jump);
            if(any){ m_new=fmaxf(m_ref,tmax); corr=ex2_mufu(m_ref-m_new); do_resc=true; }
            else   { m_new=m_ref; corr=1.f; do_resc=false; }
        }

        float lsum=0.f;
        #pragma unroll
        for(int c=0;c<128;c++){
            float x=s[c]-m_new;
            float p=(c<NPOLY)?ex2_poly(x):ex2_mufu(fmaxf(x,-120.f));
            lsum+=p;
            int ch=c>>6, lc=c&63;
            Ps_wg[ch*CHUNK + swz_off(wtid,lc)] = __float2bfloat16(p);
        }
        l=l*corr+lsum; m_ref=m_new;

        if(do_resc){
            #pragma unroll
            for(int blk=0;blk<4;blk++){
                uint32_t U[32];
                #pragma unroll
                for(int j=0;j<32;j+=4) tmem_ld4(Obase_wg+lane_off+blk*32+j,U[j],U[j+1],U[j+2],U[j+3]);
                wait_ld();
                #pragma unroll
                for(int j=0;j<32;j++) U[j]=__float_as_uint(__uint_as_float(U[j])*corr);
                #pragma unroll
                for(int j=0;j<32;j+=4) tmem_st4(Obase_wg+lane_off+blk*32+j,U[j],U[j+1],U[j+2],U[j+3]);
            }
            wait_st();
        }
        fence_before();
        fence_async();
        nbar(wg?2:1,128);

        // ---- PV (per warpgroup leader) : reads V[kv] from Vs ----
        if(wtid==0){
            bar_wait(bar_vL, ph_v);
            fence_after();
            #pragma unroll
            for(int g=0;g<2;g++){
                #pragma unroll
                for(int kb=0;kb<8;kb++){
                    int pch=kb>>2, psl=kb&3;
                    uint64_t da=make_desc(&Ps_wg[pch*CHUNK+psl*16],0,1024);
                    uint64_t db=make_desc(&Vs[g*CHUNK+kb*1024],16384,1024);
                    umma_cg1(Obase_wg+g*64,da,db,idesc_pv,(kv==0&&kb==0)?0:1);
                }
            }
            umma_commit_cg1(bar_pv);
        }
        bar_wait(bar_pv,ph_pv);
        fence_after();
        __syncthreads();   // sync2: both PV done -> V[kv] free
        if(tid==0 && kv+1<num_kv){
            bar_arrive_tx(bar_vL, 2*CHUNK*2);
            load_blk(&descV,bar_vL,Vs,(int)(rowbase+(kv+1)*128));   // prefetch V[kv+1]
        }

        ph_k^=1; ph_v^=1; ph_qk^=1; ph_pv^=1;
    }

    // epilogue
    int q=qtile_wg*128+wtid;
    float invl=(l>0.f)?1.f/l:0.f;
    __nv_bfloat16* optr=O+(long)(rowbase+q)*D;
    #pragma unroll
    for(int blk=0;blk<4;blk++){
        uint32_t U[32];
        #pragma unroll
        for(int j=0;j<32;j+=4) tmem_ld4(Obase_wg+lane_off+blk*32+j,U[j],U[j+1],U[j+2],U[j+3]);
        wait_ld();
        if(q<S){
            __nv_bfloat16 tmp[32];
            #pragma unroll
            for(int j=0;j<32;j++) tmp[j]=__float2bfloat16(__uint_as_float(U[j])*invl);
            #pragma unroll
            for(int j=0;j<32;j+=8)
                *reinterpret_cast<int4*>(optr+blk*32+j)=*reinterpret_cast<int4*>(&tmp[j]);
        }
    }
    if(q<S) LSE[rowbase+q]=m_ref*LN2 + logf(l);

    __syncthreads();
    if(warpAll==0) tmem_dealloc_cg1(tb,512);
}

CUresult make_tma(CUtensorMap* d,void* ptr,uint64_t outer){
    uint64_t gd[2]={128,outer}; uint64_t gs[1]={128*2};
    uint32_t bd[2]={64,128}; uint32_t es[2]={1,1};
    return cuTensorMapEncodeTiled(d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,ptr,gd,gs,bd,es,
        CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int B=(int)Q.size(0), H=(int)Q.size(1), S=(int)Q.size(2);
    float scale=1.0f/sqrtf((float)D);
    int num_kv=(S+BN-1)/BN;
    int num_qt=(S+BM-1)/BM;
    uint64_t outer=(uint64_t)B*H*S;

    CUtensorMap dQ,dK,dV;
    CU_CHECK(make_tma(&dQ,(void*)Q.data_ptr(),outer));
    CU_CHECK(make_tma(&dK,(void*)K.data_ptr(),outer));
    CU_CHECK(make_tma(&dV,(void*)V.data_ptr(),outer));

    __nv_bfloat16* Op=static_cast<__nv_bfloat16*>(O.data_ptr());
    float* Lp=static_cast<float*>(LSE.data_ptr());

    dim3 grid((num_qt+1)/2, H, B);
    dim3 block(THREADS);
    size_t smem=(size_t)12*CHUNK*2 + 1024;

    cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id));
    CUDA_CHECK(cudaFuncSetAttribute(attn_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
    attn_kernel<<<grid,block,smem,stream>>>(dQ,dK,dV,Op,Lp,B,H,S,num_kv,scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_d128::run);

}  // namespace mha_d128