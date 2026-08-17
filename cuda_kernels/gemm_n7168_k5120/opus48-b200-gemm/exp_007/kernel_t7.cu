#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <algorithm>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

namespace gemm_n7168_k5120 {

#define CUDA_CHECK(call) do{cudaError_t _e=(call);if(_e!=cudaSuccess){fprintf(stderr,"CUDA %s %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__);exit(1);}}while(0)
#define CU_CHECK(call) do{CUresult _e=(call);if(_e!=CUDA_SUCCESS){const char*s=0;cuGetErrorString(_e,&s);fprintf(stderr,"CU %s %s:%d\n",s?s:"?",__FILE__,__LINE__);exit(1);}}while(0)

__device__ __forceinline__ void init_smem_barrier(uint64_t* b, uint32_t c){ asm volatile("mbarrier.init.shared.b64 [%0], %1;"::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(c)); }
__device__ __forceinline__ void fence_barrier_init(){ asm volatile("fence.mbarrier_init.release.cluster;\n":::"memory"); }
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* b, uint32_t tx){ asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(tx):"memory"); }
__device__ __forceinline__ void mbar_arrive(uint64_t* b){ asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"::"r"((uint32_t)__cvta_generic_to_shared(b)):"memory"); }
__device__ __forceinline__ void mbar_arrive_remote(uint64_t* b, uint32_t cta){ uint32_t a=(uint32_t)__cvta_generic_to_shared(b), r; asm volatile("mapa.shared::cluster.u32 %0,%1,%2;":"=r"(r):"r"(a),"r"(cta)); asm volatile("mbarrier.arrive.shared::cluster.b64 _,[%0];"::"r"(r)); }
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t ph){ asm volatile("{\n.reg .pred P;\nW%=:\nmbarrier.try_wait.parity.shared.b64 P,[%0],%1;\n@!P bra W%=;\n}\n"::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(ph)); }
__device__ __forceinline__ uint32_t cluster_rank(){ uint32_t r; asm volatile("mov.u32 %0,%%cluster_ctarank;":"=r"(r)); return r; }
__device__ __forceinline__ void cluster_sync(){ asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n":::"memory"); }
__device__ __forceinline__ void nb_sync(int id,int n){ asm volatile("barrier.sync.aligned %0, %1;"::"r"(id),"r"(n)); }
__device__ __forceinline__ void tma_load_2d_cg2(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1){
    uint32_t sa=(uint32_t)__cvta_generic_to_shared(smem); uint32_t ba=(uint32_t)__cvta_generic_to_shared(bar)&0xFEFFFFFF;
    asm volatile("cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];"::"r"(sa),"l"((uint64_t)d),"r"(c0),"r"(c1),"r"(ba):"memory");
}
__device__ __forceinline__ uint64_t make_smem_desc(void* p, uint32_t lbo, uint32_t sbo){
    uint64_t d=0; uint32_t a=(uint32_t)__cvta_generic_to_shared(p);
    d|=((uint64_t)(a&0x3FFFFu))>>4; d|=(uint64_t)((lbo&0x3FFFFu)>>4)<<16; d|=(uint64_t)((sbo&0x3FFFFu)>>4)<<32; d|=(uint64_t)1<<46; d|=(uint64_t)2<<61; return d;
}
__device__ __forceinline__ uint32_t make_idesc(uint32_t M, uint32_t N){ uint32_t d=0; d|=(1u<<4); d|=(1u<<7); d|=(1u<<10); d|=((N/8)<<17); d|=((M/16)<<24); return d; }
__device__ __forceinline__ void tmem_alloc(uint32_t* dst, int nc){ asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"::"r"((uint32_t)__cvta_generic_to_shared(dst)),"r"(nc)); }
__device__ __forceinline__ void tmem_dealloc(uint32_t a, int nc){ asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"::"r"(a),"r"(nc)); }
__device__ __forceinline__ void umma_f16_cg2(uint32_t tc, uint64_t da, uint64_t db, uint32_t id, uint32_t acc){ asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p,%4,0;\ntcgen05.mma.cta_group::2.kind::f16 [%0],%1,%2,%3,p;\n}\n"::"r"(tc),"l"(da),"l"(db),"r"(id),"r"(acc)); }
__device__ __forceinline__ void umma_commit_2sm(uint64_t* bar){ uint32_t a=(uint32_t)__cvta_generic_to_shared(bar); asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0],%1;"::"r"(a),"h"((uint16_t)0x3)); }
__device__ __forceinline__ void tcgen05_fence_after(){ asm volatile("tcgen05.fence::after_thread_sync;":::"memory"); }

__device__ __forceinline__ void tile_to_mn(int T,int m_tiles,int n_tiles,int GROUP_M,int& mc,int& nt){
    int nin = GROUP_M * n_tiles;
    int gid = T / nin;
    int first_m = gid * GROUP_M;
    int gsize = m_tiles - first_m; if(gsize>GROUP_M) gsize=GROUP_M;
    int loc = T - gid*nin;
    mc = first_m + (loc % gsize);
    nt = loc / gsize;
}

constexpr int BM=128, BN=128, BK=64, STAGES=4;
constexpr int OUT_N=256;
constexpr int A_TILE=BM*BK;
constexpr int B_TILE=BN*BK;
constexpr uint32_t TX_BYTES = 2u*(A_TILE+B_TILE)*2u;

__global__ __launch_bounds__(256,1) void gemm_kernel(int M, int N, int K,
        int m_tiles, int n_tiles, int total_tiles, int num_clusters, int GROUP_M,
        __nv_bfloat16* __restrict__ C,
        const __grid_constant__ CUtensorMap dA,
        const __grid_constant__ CUtensorMap dB){
    extern __shared__ char raw[];
    const int tid=threadIdx.x, warp=tid>>5, lane=tid&31;
    const int cta=cluster_rank();
    const int cid=blockIdx.y;

    uint32_t base=(uint32_t)__cvta_generic_to_shared(raw);
    uint32_t aln=(base+1023u)&~1023u;
    char* smem=raw+(aln-base);

    __nv_bfloat16* A_smem=(__nv_bfloat16*)smem;
    __nv_bfloat16* B_smem=(__nv_bfloat16*)(smem + STAGES*A_TILE*2);
    __nv_bfloat16* smem_out=(__nv_bfloat16*)(smem + STAGES*(A_TILE+B_TILE)*2);
    uint64_t* full=(uint64_t*)(smem + STAGES*(A_TILE+B_TILE)*2 + BM*OUT_N*2);
    uint64_t* empty=full+STAGES;
    uint64_t* mma_ready=empty+STAGES;   // [2]
    uint64_t* tmem_free=mma_ready+2;    // [2]
    uint32_t* tmem_addr=(uint32_t*)(tmem_free+2);

    const int NUM_K=K/BK;
    uint32_t idesc=make_idesc(256,256);

    if(tid==0){
        #pragma unroll
        for(int s=0;s<STAGES;s++){ init_smem_barrier(&full[s],1); init_smem_barrier(&empty[s],1); }
        init_smem_barrier(&mma_ready[0],1); init_smem_barrier(&mma_ready[1],1);
        init_smem_barrier(&tmem_free[0],2); init_smem_barrier(&tmem_free[1],2);
        fence_barrier_init();
    }
    if(warp==0){ tmem_alloc(tmem_addr,512); }
    __syncthreads();
    cluster_sync();
    uint32_t tmem_base=*tmem_addr;

    if(warp==0){
        // ---- producer (both CTAs) ----
        if(lane==0){
            uint32_t ep[STAGES];
            #pragma unroll
            for(int s=0;s<STAGES;s++) ep[s]=0;
            int gp=0;
            for(int T=cid; T<total_tiles; T+=num_clusters){
                int mc,nt; tile_to_mn(T,m_tiles,n_tiles,GROUP_M,mc,nt);
                int mb=mc*256, nb=nt*256;
                for(int k=0;k<NUM_K;k++){
                    int s=gp%STAGES;
                    if(gp>=STAGES){ mbar_wait(&empty[s], ep[s]); ep[s]^=1; }
                    if(cta==0) mbar_arrive_expect_tx(&full[s], TX_BYTES);
                    tma_load_2d_cg2(&dA,&full[s], A_smem+s*A_TILE, k*BK, mb+cta*128);
                    tma_load_2d_cg2(&dB,&full[s], B_smem+s*B_TILE, k*BK, nb+cta*128);
                    gp++;
                }
            }
        }
    } else if(warp==1 && cta==0){
        // ---- MMA consumer (leader only) ----
        if(lane==0){
            uint32_t fp[STAGES];
            #pragma unroll
            for(int s=0;s<STAGES;s++) fp[s]=0;
            uint32_t tf[2]={0,0};
            int gc=0, j=0;
            for(int T=cid; T<total_tiles; T+=num_clusters, ++j){
                int buf=j&1;
                if(j>=2){ mbar_wait(&tmem_free[buf], tf[buf]); tf[buf]^=1; }
                uint32_t tc=tmem_base + buf*256;
                for(int k=0;k<NUM_K;k++){
                    int s=gc%STAGES;
                    mbar_wait(&full[s], fp[s]); fp[s]^=1;
                    char* ap=(char*)(A_smem+s*A_TILE);
                    char* bp=(char*)(B_smem+s*B_TILE);
                    #pragma unroll
                    for(int kk=0;kk<BK/16;kk++){
                        uint64_t da=make_smem_desc(ap+kk*32,0,1024);
                        uint64_t db=make_smem_desc(bp+kk*32,0,1024);
                        umma_f16_cg2(tc,da,db,idesc,(k==0&&kk==0)?0u:1u);
                    }
                    umma_commit_2sm(&empty[s]);
                    gc++;
                }
                umma_commit_2sm(&mma_ready[buf]);
            }
        }
    }

    if(warp>=4){
        // ---- epilogue warpgroup (both CTAs) ----
        int elt=tid-128;
        uint32_t mr[2]={0,0};
        int j=0;
        for(int T=cid; T<total_tiles; T+=num_clusters, ++j){
            int buf=j&1;
            int mc,nt; tile_to_mn(T,m_tiles,n_tiles,GROUP_M,mc,nt);
            int mb=mc*256, nb=nt*256;
            mbar_wait(&mma_ready[buf], mr[buf]); mr[buf]^=1;
            tcgen05_fence_after();
            uint32_t tc=tmem_base + buf*256;
            #pragma unroll
            for(int col=0;col<OUT_N;col+=8){
                uint32_t r0,r1,r2,r3,r4,r5,r6,r7;
                uint32_t a=tc+col;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                    :"=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7):"r"(a));
                asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory");
                int bo=elt*OUT_N+col;
                smem_out[bo+0]=__float2bfloat16(__uint_as_float(r0));
                smem_out[bo+1]=__float2bfloat16(__uint_as_float(r1));
                smem_out[bo+2]=__float2bfloat16(__uint_as_float(r2));
                smem_out[bo+3]=__float2bfloat16(__uint_as_float(r3));
                smem_out[bo+4]=__float2bfloat16(__uint_as_float(r4));
                smem_out[bo+5]=__float2bfloat16(__uint_as_float(r5));
                smem_out[bo+6]=__float2bfloat16(__uint_as_float(r6));
                smem_out[bo+7]=__float2bfloat16(__uint_as_float(r7));
            }
            nb_sync(1,128);
            if(elt==0){
                if(cta==0) mbar_arrive(&tmem_free[buf]);
                else       mbar_arrive_remote(&tmem_free[buf], 0);
            }
            const int VPR=OUT_N/8;
            const int total=BM*VPR;
            for(int i=elt;i<total;i+=128){
                int row=i/VPR, cvec=i%VPR, col=cvec*8;
                int grow=mb+cta*128+row;
                if(grow<M){
                    uint4 d=*reinterpret_cast<uint4*>(&smem_out[row*OUT_N+col]);
                    *reinterpret_cast<uint4*>(&C[(int64_t)grow*N + nb + col])=d;
                }
            }
            nb_sync(1,128);
        }
    }

    __syncthreads();
    cluster_sync();
    if(warp==0){ tmem_dealloc(tmem_base,512); }
}

static CUresult make_tma(CUtensorMap* d, void* p, uint64_t inner, uint64_t outer, uint32_t bi, uint32_t bo){
    uint64_t gd[2]={inner,outer}; uint64_t gs[1]={inner*2}; uint32_t bd[2]={bi,bo}; uint32_t es[2]={1,1};
    return cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, p, gd, gs, bd, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C){
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    int64_t M=A.size(0), K=A.size(1), N=B.size(0);
    __nv_bfloat16* Aptr=(__nv_bfloat16*)A.data_ptr();
    __nv_bfloat16* Bptr=(__nv_bfloat16*)B.data_ptr();
    __nv_bfloat16* Cptr=(__nv_bfloat16*)C.data_ptr();

    CUtensorMap dA,dB;
    CU_CHECK(make_tma(&dA,Aptr,(uint64_t)K,(uint64_t)M,BK,BM));
    CU_CHECK(make_tma(&dB,Bptr,(uint64_t)K,(uint64_t)N,BK,BN));

    int dev=A.device().device_id;
    int sm=148; cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev);

    int m_tiles=(int)((M+255)/256);
    int n_tiles=(int)(N/256);
    int total_tiles=m_tiles*n_tiles;
    int num_clusters=std::min(total_tiles, sm/2);
    if(num_clusters<1) num_clusters=1;
    int GROUP_M=std::min(8, m_tiles);
    if(GROUP_M<1) GROUP_M=1;

    dim3 grid(2, num_clusters, 1);
    dim3 block(256,1,1);

    size_t smem_bytes = (size_t)STAGES*(A_TILE+B_TILE)*2 + (size_t)BM*OUT_N*2 + (2*STAGES+4)*8 + 8 + 1024;

    CUDA_CHECK(cudaFuncSetAttribute((const void*)gemm_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));

    cudaStream_t stream=(cudaStream_t)TVMFFIEnvGetStream(A.device().device_type, A.device().device_id);

    cudaLaunchConfig_t config={};
    config.gridDim=grid; config.blockDim=block; config.dynamicSmemBytes=smem_bytes; config.stream=stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id=cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x=2; attrs[0].val.clusterDim.y=1; attrs[0].val.clusterDim.z=1;
    config.attrs=attrs; config.numAttrs=1;

    CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_kernel, (int)M,(int)N,(int)K,
        m_tiles, n_tiles, total_tiles, num_clusters, GROUP_M, Cptr, dA, dB));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_n7168_k5120::run);

}  // namespace gemm_n7168_k5120