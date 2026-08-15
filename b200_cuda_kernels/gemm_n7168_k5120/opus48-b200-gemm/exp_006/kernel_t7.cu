#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <string.h>
#include <stdio.h>
#include <algorithm>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){fprintf(stderr,"CUDA %s @%s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__);exit(1);} }while(0)
#define CU_CHECK(call) do { CUresult _e=(call); if(_e!=CUDA_SUCCESS){const char* s; cuGetErrorString(_e,&s); fprintf(stderr,"CU %s @%s:%d\n",s,__FILE__,__LINE__);exit(1);} }while(0)

namespace gemm100 {
using bf16 = __nv_bfloat16;

#define NS 5
#define BK 64
#define A_TILE (128*64)
#define B_TILE (128*64)
#define TX_BYTES (4*128*64*2)
#define NUM_THREADS 192
#define GROUP 8
#define OUT_BYTES (128*256*2)

__device__ __forceinline__ bool elect_one_sync_fn(){uint32_t p;asm volatile("{\n.reg .pred q;\nelect.sync _|q,0xFFFFFFFF;\nselp.b32 %0,1,0,q;\n}\n":"=r"(p));return p!=0;}
__device__ __forceinline__ uint32_t cluster_rank_fn(){uint32_t r;asm volatile("mov.u32 %0,%%cluster_ctarank;":"=r"(r));return r;}
__device__ __forceinline__ void cluster_sync_fn(){asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n":::"memory");}
__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* b,uint32_t c){asm volatile("mbarrier.init.shared.b64 [%0],%1;"::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(c));}
__device__ __forceinline__ void fence_smem_barrier_init_fn(){asm volatile("fence.mbarrier_init.release.cluster;\n":::"memory");}
__device__ __forceinline__ void mbarrier_arrive_fn(uint64_t* b){asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];"::"r"((uint32_t)__cvta_generic_to_shared(b)):"memory");}
__device__ __forceinline__ void mbarrier_arrive_cluster_fn(uint64_t* bar,uint32_t tgt){uint32_t a=(uint32_t)__cvta_generic_to_shared(bar);uint32_t ra;asm volatile("mapa.shared::cluster.u32 %0,%1,%2;":"=r"(ra):"r"(a),"r"(tgt));asm volatile("mbarrier.arrive.shared::cluster.b64 _,[%0];"::"r"(ra):"memory");}
__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* b,uint32_t tx){asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(tx):"memory");}
__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* b,uint32_t ph){asm volatile("{\n.reg .pred P;\nW%=:\nmbarrier.try_wait.parity.shared.b64 P,[%0],%1;\n@!P bra W%=;\n}\n"::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(ph));}
__device__ __forceinline__ void named_barrier_sync_fn(int id,int n){asm volatile("barrier.sync.aligned %0,%1;"::"r"(id),"r"(n));}
__device__ __forceinline__ void tcgen05_fence_after_fn(){asm volatile("tcgen05.fence::after_thread_sync;":::"memory");}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst,int nc){uint32_t a=(uint32_t)__cvta_generic_to_shared(dst);asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0],%1;"::"r"(a),"r"(nc));}
__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr,int nc){asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0,%1;"::"r"(addr),"r"(nc));}

__device__ __forceinline__ void tma_load_2d_cg2_fn(const CUtensorMap* d,uint64_t* bar,void* smem,int32_t c0,int32_t c1){
    uint32_t sa=(uint32_t)__cvta_generic_to_shared(smem);
    uint32_t ba=(uint32_t)__cvta_generic_to_shared(bar)&0xFEFFFFFF;
    asm volatile("cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];"
        ::"r"(sa),"l"((uint64_t)d),"r"(c0),"r"(c1),"r"(ba):"memory");
}
__device__ __forceinline__ void umma_f16_cg2_fn(uint32_t tc,uint64_t da,uint64_t db,uint32_t id,uint32_t acc){
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p,%4,0;\ntcgen05.mma.cta_group::2.kind::f16 [%0],%1,%2,%3,p;\n}\n"
        ::"r"(tc),"l"(da),"l"(db),"r"(id),"r"(acc));
}
__device__ __forceinline__ void umma_commit_2sm_fn(uint64_t* bar){
    uint32_t a=(uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0],%1;"::"r"(a),"h"((uint16_t)0x3));
}
__device__ __forceinline__ uint64_t make_smem_desc_fn(void* p,uint32_t lbo,uint32_t sbo){
    uint64_t d=0;uint32_t addr=(uint32_t)__cvta_generic_to_shared(p);
    d|=(uint64_t)(addr&0x3FFFF)>>4;
    d|=(uint64_t)((lbo&0x3FFFF)>>4)<<16;
    d|=(uint64_t)((sbo&0x3FFFF)>>4)<<32;
    d|=(uint64_t)1<<46;
    d|=(uint64_t)2<<61;
    return d;
}
__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M,uint32_t N){
    uint32_t d=0; d|=(1u<<4); d|=(1u<<7); d|=(1u<<10); d|=((N/8)<<17); d|=((M/16)<<24); return d;
}
__device__ __forceinline__ void tmem_ld32(uint32_t addr,uint32_t* r){
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {"
     "%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
     "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
     :"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
      "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]),
      "=r"(r[16]),"=r"(r[17]),"=r"(r[18]),"=r"(r[19]),"=r"(r[20]),"=r"(r[21]),"=r"(r[22]),"=r"(r[23]),
      "=r"(r[24]),"=r"(r[25]),"=r"(r[26]),"=r"(r[27]),"=r"(r[28]),"=r"(r[29]),"=r"(r[30]),"=r"(r[31])
     :"r"(addr));
}

__global__ void __launch_bounds__(NUM_THREADS,1) gemm_kernel(
    const __grid_constant__ CUtensorMap dA, const __grid_constant__ CUtensorMap dB,
    bf16* __restrict__ C, int M, int N, int K, int num_clusters)
{
    extern __shared__ __align__(1024) uint8_t dsmem[];
    bf16* out_smem=(bf16*)dsmem;
    bf16* A_smem=(bf16*)(dsmem+OUT_BYTES);
    bf16* B_smem=(bf16*)(dsmem+OUT_BYTES+NS*A_TILE*2);
    uint64_t* full_bar=(uint64_t*)(dsmem+OUT_BYTES+2*NS*A_TILE*2);
    uint64_t* empty_bar=full_bar+NS;
    uint64_t* tmem_full=empty_bar+NS;
    uint64_t* tmem_empty=tmem_full+2;
    uint32_t* tmem_ptr=(uint32_t*)(tmem_empty+2);

    int cta_rank=cluster_rank_fn();
    int cluster_id=blockIdx.y;
    int tid=threadIdx.x;
    int warp_id=tid>>5;
    int lane=tid&31;
    int num_k=K/BK;

    int tiles_n=N/256;
    int tiles_m=(M+255)/256;
    int total_tiles=tiles_m*tiles_n;

    if(warp_id==0){
        if(lane==0){
            for(int s=0;s<NS;s++){init_smem_barrier_fn(&full_bar[s],1);init_smem_barrier_fn(&empty_bar[s],1);}
            for(int bfr=0;bfr<2;bfr++){init_smem_barrier_fn(&tmem_full[bfr],1);init_smem_barrier_fn(&tmem_empty[bfr],2);}
            for(int s=0;s<NS;s++) mbarrier_arrive_fn(&empty_bar[s]);
            if(cta_rank==0){for(int bfr=0;bfr<2;bfr++){mbarrier_arrive_fn(&tmem_empty[bfr]);mbarrier_arrive_fn(&tmem_empty[bfr]);}}
        }
        fence_smem_barrier_init_fn();
        tmem_alloc_fn(tmem_ptr,512);
    }
    __syncthreads();
    cluster_sync_fn();
    uint32_t tbase=*tmem_ptr;
    uint32_t idesc=make_instr_desc_fn(256,256);

    int tc=0;
    for(int linear=cluster_id; linear<total_tiles; linear+=num_clusters, tc++){
        int tpg=GROUP*tiles_n;
        int grp=linear/tpg;
        int first_m=grp*GROUP;
        int rows=min(GROUP, tiles_m-first_m);
        int r=linear%tpg;
        int m_block=first_m + (r%rows);
        int n_block=r/rows;
        int b=tc&1;
        uint32_t rphase=(tc>>1)&1;

        if(warp_id==4){
            bool el=elect_one_sync_fn();
            int m_start=m_block*256+cta_rank*128;
            int n_start=n_block*256+cta_rank*128;
            for(int kt=0;kt<num_k;kt++){
                int g=tc*num_k+kt;
                int s=g%NS; uint32_t ph=(g/NS)&1;
                mbarrier_wait_fn(&empty_bar[s],ph);
                if(el){
                    if(cta_rank==0) mbarrier_arrive_and_expect_tx_fn(&full_bar[s],TX_BYTES);
                    int kc=kt*BK;
                    tma_load_2d_cg2_fn(&dA,&full_bar[s],(void*)(A_smem+s*A_TILE),kc,m_start);
                    tma_load_2d_cg2_fn(&dB,&full_bar[s],(void*)(B_smem+s*B_TILE),kc,n_start);
                }
            }
        } else if(warp_id==5){
            if(cta_rank==0){
                bool el=elect_one_sync_fn();
                if(el){
                    mbarrier_wait_fn(&tmem_empty[b],rphase);
                    for(int kt=0;kt<num_k;kt++){
                        int g=tc*num_k+kt;
                        int s=g%NS; uint32_t ph=(g/NS)&1;
                        mbarrier_wait_fn(&full_bar[s],ph);
                        uint64_t da=make_smem_desc_fn((void*)(A_smem+s*A_TILE),1,1024);
                        uint64_t db=make_smem_desc_fn((void*)(B_smem+s*B_TILE),1,1024);
                        #pragma unroll
                        for(int kb=0;kb<4;kb++){
                            uint32_t acc=(kt==0&&kb==0)?0:1;
                            umma_f16_cg2_fn(tbase+b*256,da+(uint64_t)(kb*2),db+(uint64_t)(kb*2),idesc,acc);
                        }
                        umma_commit_2sm_fn(&empty_bar[s]);
                    }
                    umma_commit_2sm_fn(&tmem_full[b]);
                }
            }
        } else {
            mbarrier_wait_fn(&tmem_full[b],rphase);
            tcgen05_fence_after_fn();
            #pragma unroll
            for(int col=0;col<256;col+=32){
                uint32_t rr[32];
                tmem_ld32(tbase+b*256+col,rr);
                asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory");
                uint32_t base=tid*256+col;
                #pragma unroll
                for(int j=0;j<32;j+=2){
                    __nv_bfloat162 v=__floats2bfloat162_rn(__uint_as_float(rr[j]),__uint_as_float(rr[j+1]));
                    *reinterpret_cast<__nv_bfloat162*>(&out_smem[base+j])=v;
                }
            }
            named_barrier_sync_fn(2,128);
            if(tid==0){
                if(cta_rank==0) mbarrier_arrive_fn(&tmem_empty[b]);
                else mbarrier_arrive_cluster_fn(&tmem_empty[b],0);
            }
            int base_row=m_block*256+cta_rank*128;
            int base_col=n_block*256;
            #pragma unroll
            for(int i=tid;i<128*256/8;i+=128){
                int elem=i*8;
                int row=elem>>8;
                int col=elem&255;
                int gr=base_row+row;
                if(gr<M){
                    uint4 v=*reinterpret_cast<uint4*>(&out_smem[row*256+col]);
                    *reinterpret_cast<uint4*>(&C[(size_t)gr*N+base_col+col])=v;
                }
            }
            named_barrier_sync_fn(3,128);
        }
    }
    cluster_sync_fn();
    if(warp_id==0) tmem_dealloc_fn(tbase,512);
}

CUresult make_tma_desc(CUtensorMap* d,void* g,uint64_t inner,uint64_t outer,uint32_t bi,uint32_t bo){
    uint64_t gdim[2]={inner,outer};
    uint64_t gstr[1]={inner*2};
    uint32_t bdim[2]={bi,bo};
    uint32_t estr[2]={1,1};
    return cuTensorMapEncodeTiled(d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,g,gdim,gstr,bdim,estr,
        CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C){
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    int dev=A.device().device_id;
    int M=(int)A.size(0), K=(int)A.size(1), N=(int)B.size(0);
    bf16* a=static_cast<bf16*>(A.data_ptr());
    bf16* b=static_cast<bf16*>(B.data_ptr());
    bf16* c=static_cast<bf16*>(C.data_ptr());
    if(M<=0) return;

    CUtensorMap dA,dB;
    CU_CHECK(make_tma_desc(&dA,a,(uint64_t)K,(uint64_t)M,64,128));
    CU_CHECK(make_tma_desc(&dB,b,(uint64_t)K,(uint64_t)N,64,128));

    size_t smem=OUT_BYTES + 2*NS*A_TILE*2 + (2*NS+4)*8 + 16;
    CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));

    int tiles_m=(M+255)/256;
    int tiles_n=N/256;
    int total_tiles=tiles_m*tiles_n;
    int num_clusters=total_tiles<74?total_tiles:74;
    dim3 grid(2,num_clusters,1);
    dim3 block(NUM_THREADS);
    cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type,dev));

    // ---- L2 persistence: pin operand A (long reuse distance) ----
    int maxwin=0;
    cudaDeviceGetAttribute(&maxwin, cudaDevAttrMaxPersistingL2CacheSize, dev);
    size_t a_bytes=(size_t)M*K*2;
    size_t persist=(size_t)maxwin;
    if(a_bytes<persist) persist=a_bytes;
    bool use_persist = (maxwin>0 && persist>0);
    if(use_persist){
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, persist);
        cudaStreamAttrValue av; memset(&av,0,sizeof(av));
        av.accessPolicyWindow.base_ptr=(void*)a;
        av.accessPolicyWindow.num_bytes=persist;
        av.accessPolicyWindow.hitRatio=1.0f;
        av.accessPolicyWindow.hitProp=cudaAccessPropertyPersisting;
        av.accessPolicyWindow.missProp=cudaAccessPropertyStreaming;
        cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &av);
    }

    cudaLaunchConfig_t config={};
    config.gridDim=grid; config.blockDim=block; config.dynamicSmemBytes=smem; config.stream=stream;
    cudaLaunchAttribute attr[1];
    attr[0].id=cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x=2; attr[0].val.clusterDim.y=1; attr[0].val.clusterDim.z=1;
    config.attrs=attr; config.numAttrs=1;
    CUDA_CHECK(cudaLaunchKernelEx(&config,gemm_kernel,dA,dB,c,M,N,K,num_clusters));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if(use_persist){
        cudaStreamAttrValue av; memset(&av,0,sizeof(av));
        av.accessPolicyWindow.base_ptr=(void*)a;
        av.accessPolicyWindow.num_bytes=0;
        av.accessPolicyWindow.hitRatio=0.0f;
        av.accessPolicyWindow.hitProp=cudaAccessPropertyNormal;
        av.accessPolicyWindow.missProp=cudaAccessPropertyNormal;
        cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &av);
        cudaCtxResetPersistingL2Cache();
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0);
    }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm100::run);

}  // namespace gemm100