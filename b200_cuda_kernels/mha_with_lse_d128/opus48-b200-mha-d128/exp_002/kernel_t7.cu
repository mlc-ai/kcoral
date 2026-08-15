#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__);} }while(0)

namespace mha {
constexpr int HD=128;

__device__ __forceinline__ float ex2(float x){ float y; asm volatile("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x)); return y; }
__device__ __forceinline__ uint32_t cvta(const void*p){ return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp_async16(uint32_t dst,const void* src,bool pred){
    int sz=pred?16:0; asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;\n"::"r"(dst),"l"(src),"r"(sz)); }
__device__ __forceinline__ void cp_commit(){ asm volatile("cp.async.commit_group;\n"); }
template<int N> __device__ __forceinline__ void cp_wait(){ asm volatile("cp.async.wait_group %0;\n"::"n"(N):"memory"); }
__device__ __forceinline__ void init_bar(uint64_t* b,uint32_t c){ asm volatile("mbarrier.init.shared.b64 [%0], %1;"::"r"(cvta(b)),"r"(c)); }
__device__ __forceinline__ void bar_wait(uint64_t* b,uint32_t ph){
    asm volatile("{\n.reg .pred P;\nW_%=:\nmbarrier.try_wait.parity.shared.b64 P,[%0],%1;\n@!P bra W_%=;\n}\n"::"r"(cvta(b)),"r"(ph)); }
__device__ __forceinline__ void tmem_alloc1(uint32_t* d,int n){ asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"::"r"(cvta(d)),"r"(n)); }
__device__ __forceinline__ void tmem_dealloc1(uint32_t a,int n){ asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"::"r"(a),"r"(n)); }
__device__ __forceinline__ void tmem_relinquish1(){ asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"); }
__device__ __forceinline__ void umma1(uint32_t td,uint64_t da,uint64_t db,uint32_t id,uint32_t ac){
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p,%4,0;\ntcgen05.mma.cta_group::1.kind::f16 [%0],%1,%2,%3,p;\n}\n"::"r"(td),"l"(da),"l"(db),"r"(id),"r"(ac)); }
__device__ __forceinline__ void umma_commit1(uint64_t* b){ asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"::"r"(cvta(b)):"memory"); }
__device__ __forceinline__ void tmem_wait_ld(){ asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory"); }
__device__ __forceinline__ void tmem_wait_st(){ asm volatile("tcgen05.wait::st.sync.aligned;":::"memory"); }
__device__ __forceinline__ void t5_before(){ asm volatile("tcgen05.fence::before_thread_sync;":::"memory"); }
__device__ __forceinline__ void t5_after(){ asm volatile("tcgen05.fence::after_thread_sync;":::"memory"); }
__device__ __forceinline__ void fence_async(){ asm volatile("fence.proxy.async;":::"memory"); }
__device__ __forceinline__ void tmem_ld_x32(uint32_t t, float* d){
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
    :"=f"(d[0]),"=f"(d[1]),"=f"(d[2]),"=f"(d[3]),"=f"(d[4]),"=f"(d[5]),"=f"(d[6]),"=f"(d[7]),"=f"(d[8]),"=f"(d[9]),"=f"(d[10]),"=f"(d[11]),"=f"(d[12]),"=f"(d[13]),"=f"(d[14]),"=f"(d[15]),"=f"(d[16]),"=f"(d[17]),"=f"(d[18]),"=f"(d[19]),"=f"(d[20]),"=f"(d[21]),"=f"(d[22]),"=f"(d[23]),"=f"(d[24]),"=f"(d[25]),"=f"(d[26]),"=f"(d[27]),"=f"(d[28]),"=f"(d[29]),"=f"(d[30]),"=f"(d[31]):"r"(t)); }
__device__ __forceinline__ void tmem_st_x32(uint32_t t, const float* d){
    asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%32], {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31};"
    ::"f"(d[0]),"f"(d[1]),"f"(d[2]),"f"(d[3]),"f"(d[4]),"f"(d[5]),"f"(d[6]),"f"(d[7]),"f"(d[8]),"f"(d[9]),"f"(d[10]),"f"(d[11]),"f"(d[12]),"f"(d[13]),"f"(d[14]),"f"(d[15]),"f"(d[16]),"f"(d[17]),"f"(d[18]),"f"(d[19]),"f"(d[20]),"f"(d[21]),"f"(d[22]),"f"(d[23]),"f"(d[24]),"f"(d[25]),"f"(d[26]),"f"(d[27]),"f"(d[28]),"f"(d[29]),"f"(d[30]),"f"(d[31]),"r"(t):"memory"); }
__device__ __forceinline__ uint64_t make_desc(const void* p,uint32_t lbo,uint32_t sbo){
    uint64_t d=0; uint32_t a=cvta(p);
    d|=(uint64_t)((a&0x3FFFF)>>4); d|=(uint64_t)((lbo&0x3FFFF)>>4)<<16; d|=(uint64_t)((sbo&0x3FFFF)>>4)<<32; d|=(uint64_t)1<<46; return d; }
__device__ __forceinline__ uint32_t make_idesc(uint32_t M,uint32_t N,uint32_t am,uint32_t bm){
    uint32_t d=0; d|=1u<<4; d|=1u<<7; d|=1u<<10; d|=am<<15; d|=bm<<16; d|=(N>>3)<<17; d|=(M>>4)<<24; return d; }

__global__ void __launch_bounds__(256)
mha_kernel(const __nv_bfloat16* __restrict__ Q,const __nv_bfloat16* __restrict__ K,
           const __nv_bfloat16* __restrict__ V,__nv_bfloat16* __restrict__ O,
           float* __restrict__ LSE,int B,int H,int S,float scale){
    extern __shared__ char smem[];
    __nv_bfloat16* Q0s=(__nv_bfloat16*)smem;
    __nv_bfloat16* Q1s=Q0s+128*128;
    __nv_bfloat16* Ksb=Q1s+128*128;
    __nv_bfloat16* Vsb=Ksb+2*64*128;
    __nv_bfloat16* P0s=Vsb+2*64*128;
    __nv_bfloat16* P1s=P0s+128*64;
    __shared__ __align__(8) uint64_t barS0,barS1,barO0,barO1;
    __shared__ uint32_t tmem_ptr;

    int tid=threadIdx.x, warp=tid>>5;
    int group=tid>>7, localwarp=warp&3, row=tid&127;
    int q0=blockIdx.x*256;
    int headoff=blockIdx.z*H+blockIdx.y;
    const __nv_bfloat16* Qb=Q+(int64_t)headoff*S*HD;
    const __nv_bfloat16* Kb=K+(int64_t)headoff*S*HD;
    const __nv_bfloat16* Vb=V+(int64_t)headoff*S*HD;
    __nv_bfloat16* Ob=O+(int64_t)headoff*S*HD;
    float scale2=scale*1.4426950408889634f;

    if(tid==0){ init_bar(&barS0,1); init_bar(&barS1,1); init_bar(&barO0,1); init_bar(&barO1,1); }
    if(warp==0) tmem_alloc1(&tmem_ptr,512);
    __syncthreads();
    uint32_t base=tmem_ptr;
    uint32_t S_tm=base+group*64;
    uint32_t O_tm=base+128+group*128;
    uint32_t laneoff=(uint32_t)(localwarp*32)<<16;
    uint64_t* myBarS=(group==0)?&barS0:&barS1;
    uint64_t* myBarO=(group==0)?&barO0:&barO1;
    __nv_bfloat16* Pg=(group==0)?P0s:P1s;
    uint32_t idesc_qk=make_idesc(128,64,0,0);
    uint32_t idesc_pv=make_idesc(128,128,0,1);

    // load Q0 (rows q0..), Q1 (rows q0+128..)
    for(int idx=tid; idx<128*16; idx+=256){
        int r=idx>>4,c=idx&15; int g0=q0+r; int g1=q0+128+r;
        cp_async16(cvta(&Q0s[c*1024+r*8]), Qb+(int64_t)g0*128+c*8, g0<S);
        cp_async16(cvta(&Q1s[c*1024+r*8]), Qb+(int64_t)g1*128+c*8, g1<S);
    }
    int numKB=(S+63)>>6;
    auto loadKV=[&](int blk,int buf){
        __nv_bfloat16* Ks=Ksb+buf*64*128; __nv_bfloat16* Vs=Vsb+buf*64*128;
        for(int idx=tid;idx<64*16;idx+=256){int r=idx>>4,c=idx&15;int gr=blk*64+r;bool in=gr<S;
            cp_async16(cvta(&Ks[c*512+r*8]),Kb+(int64_t)gr*128+c*8,in);
            cp_async16(cvta(&Vs[c*512+r*8]),Vb+(int64_t)gr*128+c*8,in);}
    };
    loadKV(0,0); cp_commit();

    float Mref=-1e30f, Lref=0.f;
    const float TAU=8.0f;

    for(int i=0;i<numKB;i++){
        int buf=i&1;
        cp_wait<0>(); __syncthreads(); fence_async();
        if(i+1<numKB){ loadKV(i+1,(i+1)&1); cp_commit(); }
        __nv_bfloat16* Ks=Ksb+buf*64*128; __nv_bfloat16* Vs=Vsb+buf*64*128;

        // ---- QK both tiles ----
        if(tid==0){
            t5_before();
            #pragma unroll
            for(int kt=0;kt<8;kt++){
                uint64_t da=make_desc(&Q0s[kt*2048],2048,128);
                uint64_t db=make_desc(&Ks[kt*1024],1024,128);
                umma1(base+0,da,db,idesc_qk,kt==0?0:1);
            }
            umma_commit1(&barS0);
            #pragma unroll
            for(int kt=0;kt<8;kt++){
                uint64_t da=make_desc(&Q1s[kt*2048],2048,128);
                uint64_t db=make_desc(&Ks[kt*1024],1024,128);
                umma1(base+64,da,db,idesc_qk,kt==0?0:1);
            }
            umma_commit1(&barS1);
        }

        // ---- softmax (per group) ----
        bar_wait(myBarS, i&1);
        t5_after();
        float s[64];
        tmem_ld_x32(S_tm+laneoff+0,&s[0]);
        tmem_ld_x32(S_tm+laneoff+32,&s[32]);
        tmem_wait_ld();
        int kb=i*64;
        float bmax=-1e30f;
        #pragma unroll
        for(int j=0;j<64;j++){int kv=kb+j; float v=(kv<S)?s[j]*scale2:-1e30f; s[j]=v; bmax=fmaxf(bmax,v);}
        float Mcand=fmaxf(Mref,bmax);
        float corr; bool do_rescale;
        if(i==0){ corr=1.f; Mref=Mcand; do_rescale=false; }
        else{
            float jump=Mcand-Mref;
            #pragma unroll
            for(int o=16;o>0;o>>=1) jump=fmaxf(jump,__shfl_xor_sync(0xffffffff,jump,o));
            if(jump>TAU){ corr=ex2(Mref-Mcand); Mref=Mcand; do_rescale=true; }
            else { corr=1.f; do_rescale=false; }
        }
        float sump=0.f;
        #pragma unroll
        for(int j=0;j<64;j++){ float p=ex2(s[j]-Mref); sump+=p; s[j]=p; }
        Lref=Lref*corr+sump;

        // wait prev PV, rescale O
        if(i>0){
            bar_wait(myBarO,(i-1)&1);
            t5_after();
            if(do_rescale){
                #pragma unroll
                for(int c=0;c<4;c++){
                    float o[32]; tmem_ld_x32(O_tm+laneoff+c*32,o); tmem_wait_ld();
                    #pragma unroll
                    for(int k=0;k<32;k++) o[k]*=corr;
                    tmem_st_x32(O_tm+laneoff+c*32,o);
                }
                tmem_wait_st();
            }
        }
        // write P
        #pragma unroll
        for(int c=0;c<8;c++){
            __nv_bfloat16 tmp[8];
            #pragma unroll
            for(int k=0;k<8;k++) tmp[k]=__float2bfloat16(s[c*8+k]);
            *(uint4*)&Pg[c*1024+row*8]=*(uint4*)tmp;
        }
        t5_before();
        __syncthreads();
        fence_async();

        // ---- PV both tiles ----
        if(tid==0){
            t5_after();
            #pragma unroll
            for(int kt=0;kt<4;kt++){
                uint64_t da=make_desc(&P0s[kt*2048],2048,128);
                uint64_t db=make_desc(&Vs[kt*128],128,1024);
                umma1(base+128,da,db,idesc_pv,(i==0&&kt==0)?0:1);
            }
            umma_commit1(&barO0);
            #pragma unroll
            for(int kt=0;kt<4;kt++){
                uint64_t da=make_desc(&P1s[kt*2048],2048,128);
                uint64_t db=make_desc(&Vs[kt*128],128,1024);
                umma1(base+256,da,db,idesc_pv,(i==0&&kt==0)?0:1);
            }
            umma_commit1(&barO1);
        }
    }
    bar_wait(myBarO,(numKB-1)&1);
    t5_after();

    // epilogue
    float inv=(Lref>0.f)?1.f/Lref:0.f;
    int grow=q0+group*128+row;
    #pragma unroll
    for(int c=0;c<4;c++){
        float o[32]; tmem_ld_x32(O_tm+laneoff+c*32,o); tmem_wait_ld();
        if(grow<S){
            #pragma unroll
            for(int i2=0;i2<32;i2+=8){
                __nv_bfloat16 tmp[8];
                #pragma unroll
                for(int k=0;k<8;k++) tmp[k]=__float2bfloat16(o[i2+k]*inv);
                *(uint4*)&Ob[(int64_t)grow*128+c*32+i2]=*(uint4*)tmp;
            }
        }
    }
    if(grow<S) LSE[(int64_t)headoff*S+grow]=0.6931471805599453f*(Mref+__log2f(Lref));

    __syncthreads();
    if(warp==0){ tmem_dealloc1(tmem_ptr,512); tmem_relinquish1(); }
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
    int numQB=(Sn+255)/256;
    dim3 grid(numQB, Hn, Bn); dim3 block(256);
    size_t smem=(size_t)(128*128 + 128*128 + 2*64*128 + 2*64*128 + 128*64 + 128*64)*2;
    cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id));
    cudaFuncSetAttribute(mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem);
    mha_kernel<<<grid,block,smem,stream>>>(Qp,Kp,Vp,Op,LSEp,Bn,Hn,Sn,scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}
} // namespace mha
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha::run);