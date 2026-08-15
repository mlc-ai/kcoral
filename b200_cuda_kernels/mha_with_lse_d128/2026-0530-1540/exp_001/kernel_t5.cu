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

constexpr int D=128, BM=128, BN=128, THREADS=128;
constexpr int CHUNK = 128*64;
constexpr int NPOLY = 64;          // columns [0,NPOLY) use FMA poly, rest MUFU
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
    d |= (uint64_t)1<<46;
    d |= (uint64_t)2<<61;
    return d;
}
__device__ __forceinline__ uint32_t make_idesc(int M,int N,int aT,int bT){
    uint32_t d=0;
    d|=(1u<<4); d|=(1u<<7); d|=(1u<<10);
    d|=((uint32_t)aT<<15); d|=((uint32_t)bT<<16);
    d|=(((uint32_t)N>>3)<<17); d|=(((uint32_t)M>>4)<<24);
    return d;
}
__device__ __forceinline__ void tmem_alloc_cg1(uint32_t* dst,int n){
    uint32_t a=(uint32_t)__cvta_generic_to_shared(dst);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"::"r"(a),"r"(n));
}
__device__ __forceinline__ void tmem_relinquish_cg1(){
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}
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
__device__ __forceinline__ void tma_2d(const CUtensorMap* d,uint64_t* bar,void* smem,int c0,int c1){
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3,%4}], [%2];"
        ::"r"((uint32_t)__cvta_generic_to_shared(smem)),"l"((uint64_t)d),
          "r"((uint32_t)__cvta_generic_to_shared(bar)),"r"(c0),"r"(c1):"memory");
}
__device__ __forceinline__ void load_blk(const CUtensorMap* d,uint64_t* bar,__nv_bfloat16* buf,int c1){
    bar_arrive_tx(bar, 2*CHUNK*2);
    tma_2d(d,bar,&buf[0],     0, c1);
    tma_2d(d,bar,&buf[CHUNK], 64,c1);
}
__device__ __forceinline__ float ex2_mufu(float x){
    float y; asm volatile("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x)); return y;
}
__device__ __forceinline__ float ex2_poly(float x){
    x=fmaxf(x,-120.f);
    float n=rintf(x);
    float f=x-n;
    float p=0.0096181f;
    p=fmaf(p,f,0.0555041f);
    p=fmaf(p,f,0.2402265f);
    p=fmaf(p,f,0.6931472f);
    p=fmaf(p,f,1.0f);
    int ni=(int)n;
    ni = ni<-127?-127:ni;
    float pw=__uint_as_float((uint32_t)(ni+127)<<23);
    return p*pw;
}

__global__ __launch_bounds__(THREADS) void attn_kernel(
    const __grid_constant__ CUtensorMap descQ,
    const __grid_constant__ CUtensorMap descK,
    const __grid_constant__ CUtensorMap descV,
    __nv_bfloat16* __restrict__ O, float* __restrict__ LSE,
    int B,int H,int S,int num_kv,float scale)
{
    int qtile=blockIdx.x, h=blockIdx.y, b=blockIdx.z;
    int tid=threadIdx.x, warp=tid>>5;
    long rowbase=(long)(b*H+h)*S;
    uint32_t lane_off = (uint32_t)(warp*32) << 16;
    float scale2 = scale*LOG2E;

    __shared__ __align__(8) uint64_t s_bar[8];
    __shared__ uint32_t s_tmem[4];
    uint64_t* bar_q   = &s_bar[0];
    uint64_t* bar_k[2]= {&s_bar[1],&s_bar[2]};
    uint64_t* bar_v[2]= {&s_bar[3],&s_bar[4]};
    uint64_t* bar_qk[2]={&s_bar[5],&s_bar[6]};
    uint64_t* bar_pv  = &s_bar[7];

    extern __shared__ char smem_raw[];
    uint32_t raw=(uint32_t)__cvta_generic_to_shared(smem_raw);
    uint32_t pad=((-raw)&1023u);
    __nv_bfloat16* base=(__nv_bfloat16*)(smem_raw+pad);
    __nv_bfloat16* Qs=base;
    __nv_bfloat16* Kbuf[2]={base+2*CHUNK, base+4*CHUNK};
    __nv_bfloat16* Vbuf[2]={base+6*CHUNK, base+8*CHUNK};
    __nv_bfloat16* Pbuf[2]={base+10*CHUNK, base+12*CHUNK};

    if(tid==0){
        bar_init(bar_q,1);
        bar_init(bar_k[0],1); bar_init(bar_k[1],1);
        bar_init(bar_v[0],1); bar_init(bar_v[1],1);
        bar_init(bar_qk[0],1); bar_init(bar_qk[1],1);
        bar_init(bar_pv,1);
        bar_fence_init();
    }
    if(warp==0){ tmem_alloc_cg1(s_tmem,512); tmem_relinquish_cg1(); }
    __syncthreads();
    uint32_t tmem_base=s_tmem[0];
    uint32_t Sbase[2]={tmem_base, tmem_base+128};
    uint32_t Obase=tmem_base+256;

    uint32_t idesc_qk=make_idesc(128,128,0,0);
    uint32_t idesc_pv=make_idesc(128,64,0,1);

    int ph_k[2]={0,0}, ph_v[2]={0,0}, ph_qk[2]={0,0}, ph_pv=0;

    // prologue loads
    if(tid==0){
        load_blk(&descQ,bar_q,Qs,(int)(rowbase+qtile*128));
        load_blk(&descK,bar_k[0],Kbuf[0],(int)(rowbase+0*128));
        if(num_kv>1) load_blk(&descK,bar_k[1],Kbuf[1],(int)(rowbase+1*128));
        load_blk(&descV,bar_v[0],Vbuf[0],(int)(rowbase+0*128));
    }
    bar_wait(bar_q,0);
    __syncthreads();

    // issue QK[0] -> S[0]
    if(tid==0){
        bar_wait(bar_k[0],ph_k[0]); ph_k[0]^=1;
        #pragma unroll
        for(int kb=0;kb<8;kb++){
            int ch=kb>>2, sl=kb&3;
            uint64_t da=make_desc(&Qs[ch*CHUNK+sl*16],0,1024);
            uint64_t db=make_desc(&Kbuf[0][ch*CHUNK+sl*16],0,1024);
            umma_cg1(Sbase[0],da,db,idesc_qk, kb==0?0:1);
        }
        umma_commit_cg1(bar_qk[0]);
    }

    float m_ref=NEG, l=0.f;
    bool pv_pending=false;

    for(int kv=0; kv<num_kv; kv++){
        int cur=kv&1, nxt=(kv+1)&1;
        int kvbase=kv*128;

        bar_wait(bar_qk[cur], ph_qk[cur]); ph_qk[cur]^=1;
        fence_after();

        if(tid==0){
            if(kv+2<num_kv) load_blk(&descK,bar_k[cur],Kbuf[cur],(int)(rowbase+(kv+2)*128));
            if(kv+1<num_kv){
                bar_wait(bar_k[nxt],ph_k[nxt]); ph_k[nxt]^=1;
                #pragma unroll
                for(int kb=0;kb<8;kb++){
                    int ch=kb>>2, sl=kb&3;
                    uint64_t da=make_desc(&Qs[ch*CHUNK+sl*16],0,1024);
                    uint64_t db=make_desc(&Kbuf[nxt][ch*CHUNK+sl*16],0,1024);
                    umma_cg1(Sbase[nxt],da,db,idesc_qk, kb==0?0:1);
                }
                umma_commit_cg1(bar_qk[nxt]);
            }
        }

        // ---- read S[cur] ----
        float s[128];
        #pragma unroll
        for(int blk=0;blk<4;blk++){
            uint32_t U[32];
            #pragma unroll
            for(int j=0;j<32;j+=4) tmem_ld4(Sbase[cur]+lane_off+blk*32+j,U[j],U[j+1],U[j+2],U[j+3]);
            wait_ld();
            #pragma unroll
            for(int j=0;j<32;j++){ int c=blk*32+j; int k0=kvbase+c;
                s[c]=(k0<S)?__uint_as_float(U[j])*scale2:NEG; }
        }
        float tmax=NEG;
        #pragma unroll
        for(int c=0;c<128;c++) tmax=fmaxf(tmax,s[c]);

        float corr, m_new; bool do_resc;
        if(kv==0){ m_new=tmax; corr=1.f; do_resc=false; }
        else{
            bool jump = (tmax - m_ref) > TAU2;
            bool any = __any_sync(0xffffffffu, jump);
            if(any){ m_new=fmaxf(m_ref,tmax); corr=ex2_mufu(m_ref-m_new); do_resc=true; }
            else   { m_new=m_ref; corr=1.f; do_resc=false; }
        }

        // ---- P + lsum ----
        float lsum=0.f;
        #pragma unroll
        for(int c=0;c<128;c++){
            float x=s[c]-m_new;
            float p=(c<NPOLY)?ex2_poly(x):ex2_mufu(fmaxf(x,-120.f));
            lsum+=p;
            int ch=c>>6, lc=c&63;
            Pbuf[cur][ch*CHUNK + swz_off(tid,lc)] = __float2bfloat16(p);
        }
        l = l*corr + lsum;
        m_ref = m_new;

        // wait prev PV
        if(pv_pending){ bar_wait(bar_pv,ph_pv); ph_pv^=1; pv_pending=false; }
        fence_after();

        if(tid==0 && kv+1<num_kv) load_blk(&descV,bar_v[nxt],Vbuf[nxt],(int)(rowbase+(kv+1)*128));

        // ---- conditional O rescale ----
        if(do_resc){
            #pragma unroll
            for(int blk=0;blk<4;blk++){
                uint32_t U[32];
                #pragma unroll
                for(int j=0;j<32;j+=4) tmem_ld4(Obase+lane_off+blk*32+j,U[j],U[j+1],U[j+2],U[j+3]);
                wait_ld();
                #pragma unroll
                for(int j=0;j<32;j++) U[j]=__float_as_uint(__uint_as_float(U[j])*corr);
                #pragma unroll
                for(int j=0;j<32;j+=4) tmem_st4(Obase+lane_off+blk*32+j,U[j],U[j+1],U[j+2],U[j+3]);
            }
            wait_st();
        }

        fence_before();
        fence_async();
        __syncthreads();

        // ---- PV[kv] ----
        if(tid==0){
            bar_wait(bar_v[cur],ph_v[cur]); ph_v[cur]^=1;
            fence_after();
            #pragma unroll
            for(int g=0; g<2; g++){
                #pragma unroll
                for(int kb=0;kb<8;kb++){
                    int pch=kb>>2, psl=kb&3;
                    uint64_t da=make_desc(&Pbuf[cur][pch*CHUNK+psl*16],0,1024);
                    uint64_t db=make_desc(&Vbuf[cur][g*CHUNK+kb*1024],16384,1024);
                    umma_cg1(Obase+g*64,da,db,idesc_pv,(kv==0&&kb==0)?0:1);
                }
            }
            umma_commit_cg1(bar_pv);
        }
        pv_pending=true;
    }

    if(pv_pending){ bar_wait(bar_pv,ph_pv); ph_pv^=1; }
    fence_after();

    int q=qtile*128+tid;
    float invl=(l>0.f)?1.f/l:0.f;
    __nv_bfloat16* optr=O+(long)(rowbase+q)*D;
    #pragma unroll
    for(int blk=0;blk<4;blk++){
        uint32_t U[32];
        #pragma unroll
        for(int j=0;j<32;j+=4) tmem_ld4(Obase+lane_off+blk*32+j,U[j],U[j+1],U[j+2],U[j+3]);
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
    if(warp==0) tmem_dealloc_cg1(tmem_base,512);
}

CUresult make_tma(CUtensorMap* d,void* ptr,uint64_t outer){
    uint64_t gd[2]={128,outer};
    uint64_t gs[1]={128*2};
    uint32_t bd[2]={64,128};
    uint32_t es[2]={1,1};
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
    uint64_t outer=(uint64_t)B*H*S;

    CUtensorMap dQ,dK,dV;
    CU_CHECK(make_tma(&dQ,(void*)Q.data_ptr(),outer));
    CU_CHECK(make_tma(&dK,(void*)K.data_ptr(),outer));
    CU_CHECK(make_tma(&dV,(void*)V.data_ptr(),outer));

    __nv_bfloat16* Op=static_cast<__nv_bfloat16*>(O.data_ptr());
    float* Lp=static_cast<float*>(LSE.data_ptr());

    dim3 grid((S+BM-1)/BM, H, B);
    dim3 block(THREADS);
    size_t smem=(size_t)14*CHUNK*2 + 1024;

    cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id));
    CUDA_CHECK(cudaFuncSetAttribute(attn_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
    attn_kernel<<<grid,block,smem,stream>>>(dQ,dK,dV,Op,Lp,B,H,S,num_kv,scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_d128::run);

}  // namespace mha_d128