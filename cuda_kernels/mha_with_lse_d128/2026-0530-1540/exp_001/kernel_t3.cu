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
constexpr int CHUNK = 128*64;   // bf16 elems per swizzled [128][64] chunk
#define NEG (-1e30f)

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

    __shared__ __align__(8) uint64_t s_bar[3];
    __shared__ uint32_t s_tmem[4];
    uint64_t* bar_q=&s_bar[0]; uint64_t* bar_kv=&s_bar[1]; uint64_t* bar_mma=&s_bar[2];

    extern __shared__ char smem_raw[];
    uint32_t raw=(uint32_t)__cvta_generic_to_shared(smem_raw);
    uint32_t pad=((-raw)&1023u);
    __nv_bfloat16* Qs=(__nv_bfloat16*)(smem_raw+pad);
    __nv_bfloat16* Ks=Qs+2*CHUNK;
    __nv_bfloat16* Vs=Ks+2*CHUNK;
    __nv_bfloat16* Ps=Vs+2*CHUNK;

    if(tid==0){ bar_init(bar_q,1); bar_init(bar_kv,1); bar_init(bar_mma,1); bar_fence_init(); }
    if(warp==0){ tmem_alloc_cg1(s_tmem,256); tmem_relinquish_cg1(); }
    __syncthreads();
    uint32_t tmem_base=s_tmem[0];
    uint32_t Sbase=tmem_base, Obase=tmem_base+128;

    uint32_t idesc_qk=make_idesc(128,128,0,0);
    uint32_t idesc_pv=make_idesc(128,64,0,1);

    // load Q
    if(tid==0){
        bar_arrive_tx(bar_q, 2*CHUNK*2);
        int c1=(int)(rowbase+qtile*128);
        tma_2d(&descQ,bar_q,&Qs[0],     0, c1);
        tma_2d(&descQ,bar_q,&Qs[CHUNK], 64,c1);
    }
    bar_wait(bar_q,0);
    __syncthreads();

    float m=NEG, l=0.f;
    uint32_t p_kv=0, p_mma=0;

    for(int kv=0; kv<num_kv; kv++){
        if(tid==0){
            bar_arrive_tx(bar_kv, 4*CHUNK*2);
            int c1=(int)(rowbase+kv*128);
            tma_2d(&descK,bar_kv,&Ks[0],     0, c1);
            tma_2d(&descK,bar_kv,&Ks[CHUNK], 64,c1);
            tma_2d(&descV,bar_kv,&Vs[0],     0, c1);
            tma_2d(&descV,bar_kv,&Vs[CHUNK], 64,c1);
        }
        bar_wait(bar_kv,p_kv); p_kv^=1;
        __syncthreads();

        // QK
        if(tid==0){
            #pragma unroll
            for(int kb=0;kb<8;kb++){
                int ch=kb>>2, sl=kb&3;
                uint64_t da=make_desc(&Qs[ch*CHUNK+sl*16],0,1024);
                uint64_t db=make_desc(&Ks[ch*CHUNK+sl*16],0,1024);
                umma_cg1(Sbase,da,db,idesc_qk, kb==0?0:1);
            }
            umma_commit_cg1(bar_mma);
        }
        bar_wait(bar_mma,p_mma); p_mma^=1;
        fence_after();

        // read S into registers
        float s[128];
        #pragma unroll
        for(int c=0;c<128;c+=4){
            uint32_t r0,r1,r2,r3; tmem_ld4(Sbase+lane_off+c,r0,r1,r2,r3); wait_ld();
            int k0=kv*128+c;
            s[c+0]=(k0+0<S)?__uint_as_float(r0)*scale:NEG;
            s[c+1]=(k0+1<S)?__uint_as_float(r1)*scale:NEG;
            s[c+2]=(k0+2<S)?__uint_as_float(r2)*scale:NEG;
            s[c+3]=(k0+3<S)?__uint_as_float(r3)*scale:NEG;
        }
        float tmax=NEG;
        #pragma unroll
        for(int c=0;c<128;c++) tmax=fmaxf(tmax,s[c]);
        float mnew=fmaxf(m,tmax);
        float corr=__expf(m-mnew);

        // rescale O
        if(kv>0){
            #pragma unroll
            for(int c=0;c<128;c+=4){
                uint32_t o0,o1,o2,o3; tmem_ld4(Obase+lane_off+c,o0,o1,o2,o3); wait_ld();
                tmem_st4(Obase+lane_off+c,
                    __float_as_uint(__uint_as_float(o0)*corr),
                    __float_as_uint(__uint_as_float(o1)*corr),
                    __float_as_uint(__uint_as_float(o2)*corr),
                    __float_as_uint(__uint_as_float(o3)*corr));
            }
        }

        // P and lsum
        float lsum=0.f;
        #pragma unroll
        for(int c=0;c<128;c++){
            float p=__expf(s[c]-mnew);
            lsum+=p;
            int ch=c>>6, lc=c&63;
            Ps[ch*CHUNK+swz_off(tid,lc)]=__float2bfloat16(p);
        }
        l=l*corr+lsum; m=mnew;

        wait_st();
        fence_before();
        fence_async();
        __syncthreads();

        // PV
        if(tid==0){
            fence_after();
            #pragma unroll
            for(int g=0; g<2; g++){
                #pragma unroll
                for(int kb=0;kb<8;kb++){
                    int pch=kb>>2, psl=kb&3;
                    uint64_t da=make_desc(&Ps[pch*CHUNK+psl*16],0,1024);
                    uint64_t db=make_desc(&Vs[g*CHUNK+kb*1024],16384,1024);
                    umma_cg1(Obase+g*64,da,db,idesc_pv,(kv==0&&kb==0)?0:1);
                }
            }
            umma_commit_cg1(bar_mma);
        }
        bar_wait(bar_mma,p_mma); p_mma^=1;
        fence_after();
        __syncthreads();
    }

    // epilogue
    int q=qtile*128+tid;
    float invl=(l>0.f)?1.f/l:0.f;
    #pragma unroll
    for(int c=0;c<128;c+=4){
        uint32_t o0,o1,o2,o3; tmem_ld4(Obase+lane_off+c,o0,o1,o2,o3); wait_ld();
        if(q<S){
            __nv_bfloat16* op=O+rowbase*D+(long)q*D+c;
            op[0]=__float2bfloat16(__uint_as_float(o0)*invl);
            op[1]=__float2bfloat16(__uint_as_float(o1)*invl);
            op[2]=__float2bfloat16(__uint_as_float(o2)*invl);
            op[3]=__float2bfloat16(__uint_as_float(o3)*invl);
        }
    }
    if(q<S) LSE[rowbase+q]=m+logf(l);

    __syncthreads();
    if(warp==0) tmem_dealloc_cg1(tmem_base,256);
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
    size_t smem=(size_t)8*CHUNK*2 + 1024;

    cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id));
    CUDA_CHECK(cudaFuncSetAttribute(attn_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
    attn_kernel<<<grid,block,smem,stream>>>(dQ,dK,dV,Op,Lp,B,H,S,num_kv,scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_d128::run);

}  // namespace mha_d128