#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__); exit(1);} }while(0)
#define CU_CHECK(call) do { CUresult _e=(call); if(_e!=CUDA_SUCCESS){ const char* s; cuGetErrorString(_e,&s); \
    fprintf(stderr,"CU error %s at %s:%d\n",s,__FILE__,__LINE__); exit(1);} }while(0)

namespace gemm_sm100 {

constexpr int BM=128, BN=128, BK=64;
constexpr int NCOMB=256, MCOMB=256;
constexpr int NSTAGES=6;
constexpr int A_ELEMS=BM*BK, B_ELEMS=BN*BK;
constexpr int A_BYTES=A_ELEMS*2, B_BYTES=B_ELEMS*2;
constexpr int TOTAL_TX=2*(A_BYTES+B_BYTES);
constexpr int NCLUST=74;

__device__ __forceinline__ void init_barrier(uint64_t* b,int c){
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(c));}
__device__ __forceinline__ void fence_bar_init(){ asm volatile("fence.mbarrier_init.release.cluster;\n":::"memory");}
__device__ __forceinline__ void arrive_expect_tx(uint64_t* b,uint32_t bytes){
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        ::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(bytes):"memory");}
__device__ __forceinline__ void bar_wait(uint64_t* b,uint32_t ph){
    asm volatile("{\n.reg .pred P;\nWT_%=:\nmbarrier.try_wait.parity.shared.b64 P,[%0],%1;\n@!P bra WT_%=;\n}\n"
        ::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(ph));}
__device__ __forceinline__ void arrive_cta(uint64_t* b){
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"::"r"((uint32_t)__cvta_generic_to_shared(b)):"memory");}
__device__ __forceinline__ void arrive_remote0(uint64_t* b){
    uint32_t a=(uint32_t)__cvta_generic_to_shared(b),r,z=0;
    asm volatile("mapa.shared::cluster.u32 %0,%1,%2;":"=r"(r):"r"(a),"r"(z));
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];"::"r"(r):"memory");}
__device__ __forceinline__ void cluster_sync(){ asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n":::"memory");}
__device__ __forceinline__ uint32_t cluster_rank(){ uint32_t r; asm volatile("mov.u32 %0,%%cluster_ctarank;":"=r"(r)); return r;}
__device__ __forceinline__ void named_bar(int id,int cnt){ asm volatile("barrier.sync.aligned %0, %1;"::"r"(id),"r"(cnt));}

__device__ __forceinline__ void tma_load_cg2_leader(const CUtensorMap* d,uint64_t* bar,void* smem,int c0,int c1){
    uint32_t sa=(uint32_t)__cvta_generic_to_shared(smem);
    uint32_t ba=(uint32_t)__cvta_generic_to_shared(bar),rba,zero=0;
    asm volatile("mapa.shared::cluster.u32 %0,%1,%2;":"=r"(rba):"r"(ba),"r"(zero));
    asm volatile("cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        ::"r"(sa),"l"((uint64_t)d),"r"(c0),"r"(c1),"r"(rba):"memory");}

__device__ __forceinline__ void tmem_alloc_cg2(uint32_t* dst,int nc){
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
        ::"r"((uint32_t)__cvta_generic_to_shared(dst)),"r"(nc));}
__device__ __forceinline__ void tmem_dealloc_cg2(uint32_t a,int nc){
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"::"r"(a),"r"(nc));}
__device__ __forceinline__ void umma_cg2(uint32_t tc,uint64_t da,uint64_t db,uint32_t id,int acc){
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p,%4,0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        ::"r"(tc),"l"(da),"l"(db),"r"(id),"r"(acc));}
__device__ __forceinline__ void umma_commit_2sm(uint64_t* b){
    uint32_t a=(uint32_t)__cvta_generic_to_shared(b);
    asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
        ::"r"(a),"h"((uint16_t)0x3));}
__device__ __forceinline__ void tc_fence_after(){ asm volatile("tcgen05.fence::after_thread_sync;":::"memory");}
__device__ __forceinline__ void tmem_wait_ld(){ asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory");}

__device__ __forceinline__ uint64_t make_desc(void* ptr,uint32_t sbo){
    uint64_t d=0; uint32_t a=(uint32_t)__cvta_generic_to_shared(ptr);
    d|=(uint64_t)((a&0x3FFFF)>>4);
    d|=(uint64_t)((sbo&0x3FFFF)>>4)<<32;
    d|=(uint64_t)1<<46; d|=(uint64_t)2<<61;
    return d;}
__device__ __forceinline__ uint32_t make_instr_desc(uint32_t M,uint32_t N){
    uint32_t d=0; d|=(1u<<4); d|=(1u<<7); d|=(1u<<10);
    d|=((N/8)<<17); d|=((M/16)<<24); return d;}
__device__ __forceinline__ void ld16(uint32_t addr,uint32_t* r){
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
      :"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
        "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15])
      :"r"(addr));}
__device__ __forceinline__ uint32_t pk(uint32_t a,uint32_t b){
    __nv_bfloat16 x=__float2bfloat16(__uint_as_float(a));
    __nv_bfloat16 y=__float2bfloat16(__uint_as_float(b));
    uint32_t r; asm("mov.b32 %0, {%1,%2};":"=r"(r):"h"(*(uint16_t*)&x),"h"(*(uint16_t*)&y)); return r;}

__global__ __launch_bounds__(256,1)
void kernel(const __grid_constant__ CUtensorMap tma_A,
            const __grid_constant__ CUtensorMap tma_B,
            __nv_bfloat16* __restrict__ C,int M,int N,int K){
    extern __shared__ __align__(1024) char smem_raw[];
    __nv_bfloat16* As=(__nv_bfloat16*)smem_raw;
    __nv_bfloat16* Bs=(__nv_bfloat16*)(smem_raw+NSTAGES*A_BYTES);
    uint64_t* full =(uint64_t*)(smem_raw+NSTAGES*A_BYTES+NSTAGES*B_BYTES);
    uint64_t* empty=full+NSTAGES;
    uint64_t* mma_done=empty+NSTAGES;
    uint64_t* tmem_free=mma_done+2;
    uint32_t* tmem_ptr=(uint32_t*)(tmem_free+2);

    int tid=threadIdx.x, warp=tid>>5;
    int numK=K/BK;
    int rank=cluster_rank();
    bool leader=(rank==0);
    int cluster_id=blockIdx.x/2;
    int num_n=N/NCOMB;
    int num_m=(M+MCOMB-1)/MCOMB;
    int numTiles=num_m*num_n;
    int L = (numTiles>cluster_id) ? ((numTiles-cluster_id-1)/NCLUST+1) : 0;

    if(tid==0){
        for(int s=0;s<NSTAGES;s++){ init_barrier(&full[s],1); init_barrier(&empty[s],1);}
        init_barrier(&mma_done[0],1); init_barrier(&mma_done[1],1);
        init_barrier(&tmem_free[0],2); init_barrier(&tmem_free[1],2);
        fence_bar_init();
    }
    __syncthreads();
    if(warp==0) tmem_alloc_cg2(tmem_ptr,512);
    __syncthreads();
    cluster_sync();
    uint32_t tmem_base=tmem_ptr[0];
    uint32_t idesc=make_instr_desc(MCOMB,NCOMB);
    int G=L*numK;

    if(tid==0){
        // producer (both CTAs)
        for(int g=0; g<G; g++){
            int s=g%NSTAGES, r=g/NSTAGES;
            if(g>=NSTAGES) bar_wait(&empty[s], (r-1)&1);
            if(leader) arrive_expect_tx(&full[s], TOTAL_TX);
            int li=g/numK, k=g%numK;
            int T=cluster_id+li*NCLUST;
            int nt=T%num_n, mp=T/num_n;
            int m_base=mp*MCOMB+rank*BM;
            int n_base=nt*NCOMB+rank*BN;
            int kt=k*BK;
            tma_load_cg2_leader(&tma_A,&full[s],&As[s*A_ELEMS],kt,m_base);
            tma_load_cg2_leader(&tma_B,&full[s],&Bs[s*B_ELEMS],kt,n_base);
        }
    } else if(leader && tid==32){
        // consumer (leader only)
        for(int g=0; g<G; g++){
            int s=g%NSTAGES, r=g/NSTAGES;
            int li=g/numK, k=g%numK;
            int buf=li&1;
            if(k==0 && li>=2) bar_wait(&tmem_free[buf], ((li/2)-1)&1);
            bar_wait(&full[s], r&1);
            uint32_t tc=tmem_base+buf*256;
            #pragma unroll
            for(int j=0;j<4;j++){
                uint64_t da=make_desc(&As[s*A_ELEMS+j*16],1024);
                uint64_t db=make_desc(&Bs[s*B_ELEMS+j*16],1024);
                umma_cg2(tc,da,db,idesc,(k==0&&j==0)?0:1);
            }
            umma_commit_2sm(&empty[s]);
            if(k==numK-1) umma_commit_2sm(&mma_done[buf]);
        }
    } else if(tid>=128){
        // epilogue warpgroup
        int row=tid-128;
        for(int li=0; li<L; li++){
            int buf=li&1;
            bar_wait(&mma_done[buf], (li/2)&1);
            tc_fence_after();
            int T=cluster_id+li*NCLUST;
            int nt=T%num_n, mp=T/num_n;
            int grow=mp*MCOMB+rank*BM+row;
            int gcol0=nt*NCOMB;
            uint32_t tc=tmem_base+buf*256;
            #pragma unroll
            for(int col=0; col<256; col+=16){
                uint32_t r[16];
                ld16(tc+col,r);
                tmem_wait_ld();
                if(grow<M){
                    uint32_t w[8];
                    #pragma unroll
                    for(int i=0;i<8;i++) w[i]=pk(r[2*i],r[2*i+1]);
                    int gc=gcol0+col;
                    *(int4*)&C[(size_t)grow*N+gc]   = *(int4*)&w[0];
                    *(int4*)&C[(size_t)grow*N+gc+8] = *(int4*)&w[4];
                }
            }
            named_bar(1,128);
            if(tid==128){ if(rank==0) arrive_cta(&tmem_free[buf]); else arrive_remote0(&tmem_free[buf]); }
        }
    }

    __syncthreads();
    cluster_sync();
    if(warp==0) tmem_dealloc_cg2(tmem_base,512);
}

static CUresult make_tma(CUtensorMap* d,void* addr,uint64_t inner,uint64_t outer,
                         uint32_t bi,uint32_t bo){
    uint64_t gdim[2]={inner,outer}; uint64_t gstr[1]={inner*2};
    uint32_t bdim[2]={bi,bo}; uint32_t estr[2]={1,1};
    return cuTensorMapEncodeTiled(d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,addr,gdim,gstr,
        bdim,estr,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView A,tvm::ffi::TensorView B,tvm::ffi::TensorView C){
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    int64_t M=A.size(0), K=A.size(1), N=B.size(0);
    if(M==0) return;

    __nv_bfloat16* Ap=static_cast<__nv_bfloat16*>(A.data_ptr());
    __nv_bfloat16* Bp=static_cast<__nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* Cp=static_cast<__nv_bfloat16*>(C.data_ptr());

    CUtensorMap tA,tB;
    CU_CHECK(make_tma(&tA,Ap,(uint64_t)K,(uint64_t)M,64,BM));
    CU_CHECK(make_tma(&tB,Bp,(uint64_t)K,(uint64_t)N,64,BN));

    int smem_bytes=NSTAGES*A_BYTES+NSTAGES*B_BYTES+(2*NSTAGES+4)*8+16;
    CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem_bytes));

    dim3 grid(2*NCLUST,1,1);
    dim3 block(256);
    cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type,A.device().device_id));

    cudaLaunchConfig_t cfg={};
    cfg.gridDim=grid; cfg.blockDim=block; cfg.dynamicSmemBytes=smem_bytes; cfg.stream=stream;
    cudaLaunchAttribute attr[1];
    attr[0].id=cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x=2; attr[0].val.clusterDim.y=1; attr[0].val.clusterDim.z=1;
    cfg.attrs=attr; cfg.numAttrs=1;

    CUDA_CHECK(cudaLaunchKernelEx(&cfg,kernel,tA,tB,Cp,(int)M,(int)N,(int)K));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_sm100::run);

}  // namespace gemm_sm100