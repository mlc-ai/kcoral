#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

namespace gemm_n7168_k5120 {

#define CUDA_CHECK(call) do{cudaError_t _e=(call);if(_e!=cudaSuccess){fprintf(stderr,"CUDA %s %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__);exit(1);}}while(0)
#define CU_CHECK(call) do{CUresult _e=(call);if(_e!=CUDA_SUCCESS){const char*s=0;cuGetErrorString(_e,&s);fprintf(stderr,"CU %s %s:%d\n",s?s:"?",__FILE__,__LINE__);exit(1);}}while(0)

__device__ __forceinline__ void init_smem_barrier(uint64_t* b, uint32_t c){
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(c));
}
__device__ __forceinline__ void fence_barrier_init(){ asm volatile("fence.mbarrier_init.release.cluster;\n":::"memory"); }
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* b, uint32_t tx){
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(tx):"memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t ph){
    asm volatile("{\n.reg .pred P;\nW%=:\nmbarrier.try_wait.parity.shared.b64 P,[%0],%1;\n@!P bra W%=;\n}\n"::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(ph));
}
__device__ __forceinline__ uint32_t cluster_rank(){ uint32_t r; asm volatile("mov.u32 %0,%%cluster_ctarank;":"=r"(r)); return r; }
__device__ __forceinline__ void cluster_sync(){ asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n":::"memory"); }
__device__ __forceinline__ void tma_load_2d_cg2(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1){
    uint32_t sa=(uint32_t)__cvta_generic_to_shared(smem);
    uint32_t ba=(uint32_t)__cvta_generic_to_shared(bar)&0xFEFFFFFF;
    asm volatile("cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];"
        ::"r"(sa),"l"((uint64_t)d),"r"(c0),"r"(c1),"r"(ba):"memory");
}
__device__ __forceinline__ uint64_t make_smem_desc(void* p, uint32_t lbo, uint32_t sbo){
    uint64_t d=0; uint32_t a=(uint32_t)__cvta_generic_to_shared(p);
    d|=((uint64_t)(a&0x3FFFFu))>>4;
    d|=(uint64_t)((lbo&0x3FFFFu)>>4)<<16;
    d|=(uint64_t)((sbo&0x3FFFFu)>>4)<<32;
    d|=(uint64_t)1<<46;
    d|=(uint64_t)2<<61;
    return d;
}
__device__ __forceinline__ uint32_t make_idesc(uint32_t M, uint32_t N){
    uint32_t d=0;
    d|=(1u<<4); d|=(1u<<7); d|=(1u<<10);
    d|=((N/8)<<17); d|=((M/16)<<24);
    return d;
}
__device__ __forceinline__ void tmem_alloc(uint32_t* dst, int nc){
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"::"r"((uint32_t)__cvta_generic_to_shared(dst)),"r"(nc));
}
__device__ __forceinline__ void tmem_dealloc(uint32_t a, int nc){
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"::"r"(a),"r"(nc));
}
__device__ __forceinline__ void umma_f16_cg2(uint32_t tc, uint64_t da, uint64_t db, uint32_t id, uint32_t acc){
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p,%4,0;\ntcgen05.mma.cta_group::2.kind::f16 [%0],%1,%2,%3,p;\n}\n"
        ::"r"(tc),"l"(da),"l"(db),"r"(id),"r"(acc));
}
__device__ __forceinline__ void umma_commit_2sm(uint64_t* bar){
    uint32_t a=(uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0],%1;"
        ::"r"(a),"h"((uint16_t)0x3));
}
__device__ __forceinline__ void tcgen05_fence_after(){ asm volatile("tcgen05.fence::after_thread_sync;":::"memory"); }

constexpr int BM=128, BN=128, BK=64, STAGES=6;
constexpr int OUT_N=256;
constexpr int A_TILE=BM*BK; // 8192 elems
constexpr int B_TILE=BN*BK; // 8192 elems
constexpr uint32_t TX_BYTES = 2u*(A_TILE+B_TILE)*2u; // 65536

__global__ __launch_bounds__(128) void gemm_kernel(int M, int N, int K,
        __nv_bfloat16* __restrict__ C,
        const __grid_constant__ CUtensorMap dA,
        const __grid_constant__ CUtensorMap dB){
    extern __shared__ char raw[];
    const int tid=threadIdx.x, warp=tid>>5, lane=tid&31;
    const int cta=cluster_rank();
    const int n_tile=blockIdx.y, m_cluster=blockIdx.z;
    const int mb=m_cluster*256, nb=n_tile*256;
    const int m_base=mb+cta*128;   // A rows / output row base
    const int n_base=nb+cta*128;   // B N-cols (split)

    uint32_t base=(uint32_t)__cvta_generic_to_shared(raw);
    uint32_t aln=(base+1023u)&~1023u;
    char* smem=raw+(aln-base);

    __nv_bfloat16* A_smem=(__nv_bfloat16*)smem;
    __nv_bfloat16* B_smem=(__nv_bfloat16*)(smem + STAGES*A_TILE*2);
    uint64_t* full=(uint64_t*)(smem + STAGES*(A_TILE+B_TILE)*2);
    uint64_t* empty=full+STAGES;
    uint64_t* mma_done=empty+STAGES;
    uint32_t* tmem_addr=(uint32_t*)(mma_done+1);
    __nv_bfloat16* smem_out=(__nv_bfloat16*)smem;

    const int NUM_K=K/BK;
    uint32_t idesc=make_idesc(256,256);

    if(tid==0){
        #pragma unroll
        for(int s=0;s<STAGES;s++){ init_smem_barrier(&full[s],1); init_smem_barrier(&empty[s],1); }
        init_smem_barrier(mma_done,1);
        fence_barrier_init();
    }
    if(warp==0){ tmem_alloc(tmem_addr,256); }
    __syncthreads();
    cluster_sync();
    uint32_t tmem_base=*tmem_addr;

    if(warp==0 && lane==0){
        // producer (both CTAs)
        uint32_t ep[STAGES];
        #pragma unroll
        for(int s=0;s<STAGES;s++) ep[s]=0;
        for(int k=0;k<NUM_K;k++){
            int s=k%STAGES;
            if(k>=STAGES){ mbar_wait(&empty[s], ep[s]); ep[s]^=1; }
            if(cta==0){ mbar_arrive_expect_tx(&full[s], TX_BYTES); }
            tma_load_2d_cg2(&dA,&full[s], A_smem+s*A_TILE, k*BK, m_base);
            tma_load_2d_cg2(&dB,&full[s], B_smem+s*B_TILE, k*BK, n_base);
        }
    } else if(cta==0 && warp==1 && lane==0){
        // consumer (leader only)
        uint32_t fp[STAGES];
        #pragma unroll
        for(int s=0;s<STAGES;s++) fp[s]=0;
        bool first=true;
        for(int k=0;k<NUM_K;k++){
            int s=k%STAGES;
            mbar_wait(&full[s], fp[s]); fp[s]^=1;
            char* ap=(char*)(A_smem+s*A_TILE);
            char* bp=(char*)(B_smem+s*B_TILE);
            #pragma unroll
            for(int kk=0;kk<BK/16;kk++){
                uint64_t da=make_smem_desc(ap+kk*32,0,1024);
                uint64_t db=make_smem_desc(bp+kk*32,0,1024);
                umma_f16_cg2(tmem_base,da,db,idesc, first?0u:1u);
                first=false;
            }
            umma_commit_2sm(&empty[s]);
        }
        umma_commit_2sm(mma_done);
    }

    mbar_wait(mma_done, 0);
    tcgen05_fence_after();

    // Phase 1: TMEM -> smem_out (each thread owns row=tid)
    #pragma unroll
    for(int col=0;col<OUT_N;col+=8){
        uint32_t r0,r1,r2,r3,r4,r5,r6,r7;
        uint32_t a=tmem_base+col;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            :"=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7):"r"(a));
        asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory");
        int bo=tid*OUT_N+col;
        smem_out[bo+0]=__float2bfloat16(__uint_as_float(r0));
        smem_out[bo+1]=__float2bfloat16(__uint_as_float(r1));
        smem_out[bo+2]=__float2bfloat16(__uint_as_float(r2));
        smem_out[bo+3]=__float2bfloat16(__uint_as_float(r3));
        smem_out[bo+4]=__float2bfloat16(__uint_as_float(r4));
        smem_out[bo+5]=__float2bfloat16(__uint_as_float(r5));
        smem_out[bo+6]=__float2bfloat16(__uint_as_float(r6));
        smem_out[bo+7]=__float2bfloat16(__uint_as_float(r7));
    }
    __syncthreads();

    // Phase 2: coalesced uint4 (8 bf16) global writes
    const int VPR=OUT_N/8; // 32
    const int total=BM*VPR; // 4096
    for(int idx=tid; idx<total; idx+=blockDim.x){
        int row=idx/VPR, cvec=idx%VPR, col=cvec*8;
        int grow=m_base+row;
        if(grow<M){
            uint4 d=*reinterpret_cast<uint4*>(&smem_out[row*OUT_N+col]);
            *reinterpret_cast<uint4*>(&C[(int64_t)grow*N + nb + col])=d;
        }
    }

    cluster_sync();
    if(warp==0){ tmem_dealloc(tmem_base,256); }
}

static CUresult make_tma(CUtensorMap* d, void* p, uint64_t inner, uint64_t outer, uint32_t bi, uint32_t bo){
    uint64_t gd[2]={inner,outer}; uint64_t gs[1]={inner*2};
    uint32_t bd[2]={bi,bo}; uint32_t es[2]={1,1};
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

    int m_clusters=(int)((M+255)/256);
    int n_tiles=(int)(N/256);
    dim3 grid(2, n_tiles, m_clusters);
    dim3 block(128,1,1);

    size_t smem_bytes = (size_t)STAGES*(A_TILE+B_TILE)*2 + (2*STAGES+1)*8 + 4 + 1024;

    CUDA_CHECK(cudaFuncSetAttribute((const void*)gemm_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));

    cudaStream_t stream=(cudaStream_t)TVMFFIEnvGetStream(A.device().device_type, A.device().device_id);

    cudaLaunchConfig_t config={};
    config.gridDim=grid;
    config.blockDim=block;
    config.dynamicSmemBytes=smem_bytes;
    config.stream=stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id=cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x=2;
    attrs[0].val.clusterDim.y=1;
    attrs[0].val.clusterDim.z=1;
    config.attrs=attrs;
    config.numAttrs=1;

    CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_kernel, (int)M,(int)N,(int)K, Cptr, dA, dB));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_n7168_k5120::run);

}  // namespace gemm_n7168_k5120