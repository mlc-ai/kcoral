#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e));exit(1);} }while(0)
#define CU_CHECK(call) do { CUresult _e=(call); if(_e!=CUDA_SUCCESS){const char*s; cuGetErrorString(_e,&s); fprintf(stderr,"CU %s:%d %s\n",__FILE__,__LINE__,s);exit(1);} }while(0)

namespace gemm_blackwell {

constexpr int KW     = 64;
constexpr int BK     = 64;
constexpr int STAGES = 7;
constexpr int TILE_ELEMS = 128*KW;
constexpr int TILE_BYTES = 128*KW*2;
constexpr uint32_t TX_BYTES = 4u*TILE_BYTES;

__device__ __forceinline__ bool elect_one_sync_fn(){
    uint32_t p; asm volatile("{\n.reg .pred p;\n elect.sync _|p,0xFFFFFFFF;\n selp.b32 %0,1,0,p;\n}\n":"=r"(p)); return p!=0;
}
__device__ __forceinline__ void cluster_sync_fn(){ asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n":::"memory"); }
__device__ __forceinline__ void init_bar(uint64_t* b, uint32_t c){ asm volatile("mbarrier.init.shared.b64 [%0], %1;"::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(c)); }
__device__ __forceinline__ void fence_bar_init(){ asm volatile("fence.mbarrier_init.release.cluster;\n":::"memory"); }
__device__ __forceinline__ void arrive_expect_tx(uint64_t* b, uint32_t tx){ asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(tx):"memory"); }
__device__ __forceinline__ void bar_wait(uint64_t* b, uint32_t ph){
    asm volatile("{\n.reg .pred P;\nW_%=:\nmbarrier.try_wait.parity.shared.b64 P,[%0],%1;\n@!P bra W_%=;\n}\n"::"r"((uint32_t)__cvta_generic_to_shared(b)),"r"(ph));
}
__device__ __forceinline__ void bar_arrive(uint64_t* b){ asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"::"r"((uint32_t)__cvta_generic_to_shared(b)):"memory"); }
__device__ __forceinline__ void bar_arrive_remote(uint64_t* b, uint32_t cta){
    uint32_t a=(uint32_t)__cvta_generic_to_shared(b),r; asm volatile("mapa.shared::cluster.u32 %0,%1,%2;":"=r"(r):"r"(a),"r"(cta));
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];"::"r"(r));
}
__device__ __forceinline__ void tma_cg2(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1){
    uint32_t sa=(uint32_t)__cvta_generic_to_shared(smem), ba=(uint32_t)__cvta_generic_to_shared(bar)&0xFEFFFFFF;
    asm volatile("cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];"
        ::"r"(sa),"l"((uint64_t)d),"r"(c0),"r"(c1),"r"(ba):"memory");
}
__device__ __forceinline__ uint64_t smem_desc(void* p, uint32_t lbo, uint32_t sbo){
    uint64_t d=0; uint32_t a=(uint32_t)__cvta_generic_to_shared(p);
    d|=(uint64_t)(a&0x3FFFF)>>4; d|=(uint64_t)((lbo&0x3FFFF)>>4)<<16; d|=(uint64_t)((sbo&0x3FFFF)>>4)<<32;
    d|=(uint64_t)1<<46; d|=(uint64_t)2<<61; return d;
}
__device__ __forceinline__ uint32_t instr_desc(uint32_t M, uint32_t N){ uint32_t d=0; d|=(1u<<4);d|=(1u<<7);d|=(1u<<10);d|=((N/8)<<17);d|=((M/16)<<24); return d; }
__device__ __forceinline__ void umma(uint32_t tc, uint64_t da, uint64_t db, uint32_t id, uint32_t acc){
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p,%4,0;\n tcgen05.mma.cta_group::2.kind::f16 [%0],%1,%2,%3,p;\n}\n"::"r"(tc),"l"(da),"l"(db),"r"(id),"r"(acc));
}
__device__ __forceinline__ void umma_commit(uint64_t* bar){
    uint32_t a=(uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0],%1;"::"r"(a),"h"((uint16_t)0x3));
}
__device__ __forceinline__ void tmem_alloc(uint32_t* d, int nc){ uint32_t a=(uint32_t)__cvta_generic_to_shared(d); asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0],%1;"::"r"(a),"r"(nc)); }
__device__ __forceinline__ void tmem_dealloc(uint32_t a, int nc){ asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0,%1;"::"r"(a),"r"(nc)); }
__device__ __forceinline__ void nbar(int id,int c){ asm volatile("barrier.sync.aligned %0,%1;"::"r"(id),"r"(c)); }

struct __align__(16) Bf8 { __nv_bfloat162 a,b,c,d; };

__global__ __launch_bounds__(256) void gemm_kernel(
    const __grid_constant__ CUtensorMap descA,
    const __grid_constant__ CUtensorMap descB,
    __nv_bfloat16* __restrict__ C, int M, int N, int K, int T, int num_m, int num_n)
{
    extern __shared__ __align__(1024) unsigned char smem[];
    __nv_bfloat16* A_sm = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* B_sm = A_sm + STAGES*TILE_ELEMS;
    uint64_t* full_bar  = reinterpret_cast<uint64_t*>(B_sm + STAGES*TILE_ELEMS);
    uint64_t* empty_bar = full_bar + STAGES;
    uint64_t* tmem_full = empty_bar + STAGES;
    uint64_t* tmem_empty= tmem_full + 2;
    uint32_t* tmem_ptr  = reinterpret_cast<uint32_t*>(tmem_empty + 2);

    int tid  = threadIdx.x;
    int warp = tid >> 5;
    int rank = blockIdx.x;
    bool leader = (rank==0);
    int cid  = blockIdx.y;
    int NC   = gridDim.y;
    int num_kt = K / BK;

    // contiguous, balanced tile range (n-fastest -> same-m runs => A stays hot in L2)
    int t_start = (int)(((long long)cid * T) / NC);
    int t_end   = (int)(((long long)(cid+1) * T) / NC);

    if (tid==0){
        for(int s=0;s<STAGES;s++){ init_bar(&full_bar[s],1); init_bar(&empty_bar[s],1); }
        init_bar(&tmem_full[0],1); init_bar(&tmem_full[1],1);
        init_bar(&tmem_empty[0],2); init_bar(&tmem_empty[1],2);
    }
    if (warp==0) tmem_alloc(tmem_ptr,512);
    __syncthreads();
    fence_bar_init();
    cluster_sync_fn();

    uint32_t tmem_base = tmem_ptr[0];
    uint32_t idesc = instr_desc(256,256);
    const uint32_t SBO = 8u*KW*2u;

    if (warp==0){
        // producer (both CTAs)
        bool el = elect_one_sync_fn();
        int stage=0;
        for (int t=t_start; t<t_end; t++){
            int m_tile=t/num_n, n_tile=t%num_n;
            int my_m=m_tile*256+rank*128, my_n=n_tile*256+rank*128;
            for (int kt=0; kt<num_kt; kt++){
                int buf=stage%STAGES;
                if (stage>=STAGES){ int ph=((stage/STAGES)-1)&1; if(el) bar_wait(&empty_bar[buf],ph); }
                if (el){
                    if(leader) arrive_expect_tx(&full_bar[buf],TX_BYTES);
                    tma_cg2(&descA,&full_bar[buf],A_sm+buf*TILE_ELEMS,kt*BK,my_m);
                    tma_cg2(&descB,&full_bar[buf],B_sm+buf*TILE_ELEMS,kt*BK,my_n);
                }
                stage++;
            }
        }
    } else if (warp==1 && leader){
        // MMA consumer (leader)
        bool el = elect_one_sync_fn();
        if (el){
            uint64_t da_base[STAGES], db_base[STAGES];
            #pragma unroll
            for(int s=0;s<STAGES;s++){ da_base[s]=smem_desc(A_sm+s*TILE_ELEMS,1,SBO); db_base[s]=smem_desc(B_sm+s*TILE_ELEMS,1,SBO); }
            int stage=0, iter=0;
            for (int t=t_start; t<t_end; t++, iter++){
                int bt=iter&1;
                if (iter>=2) bar_wait(&tmem_empty[bt], ((iter>>1)-1)&1);
                for (int kt=0; kt<num_kt; kt++){
                    int buf=stage%STAGES, ph=(stage/STAGES)&1;
                    bar_wait(&full_bar[buf],ph);
                    uint64_t da=da_base[buf], db=db_base[buf];
                    #pragma unroll
                    for (int ks=0; ks<4; ks++){
                        umma(tmem_base+bt*256, da + (uint64_t)ks*2, db + (uint64_t)ks*2, idesc, (kt==0&&ks==0)?0u:1u);
                    }
                    umma_commit(&empty_bar[buf]);
                    stage++;
                }
                umma_commit(&tmem_full[bt]);
            }
        }
    } else if (warp>=4){
        // epilogue warpgroup (both CTAs)
        int row = tid-128;
        int iter=0;
        for (int t=t_start; t<t_end; t++, iter++){
            int bt=iter&1;
            int m_tile=t/num_n, n_tile=t%num_n;
            int gm = m_tile*256 + rank*128 + row;
            int on = n_tile*256;
            bar_wait(&tmem_full[bt], (iter>>1)&1);
            #pragma unroll
            for (int col=0; col<256; col+=8){
                uint32_t r0,r1,r2,r3,r4,r5,r6,r7;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8];"
                    :"=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7):"r"(tmem_base+bt*256+col));
                asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory");
                if (gm < M){
                    Bf8 o;
                    o.a=__floats2bfloat162_rn(__uint_as_float(r0),__uint_as_float(r1));
                    o.b=__floats2bfloat162_rn(__uint_as_float(r2),__uint_as_float(r3));
                    o.c=__floats2bfloat162_rn(__uint_as_float(r4),__uint_as_float(r5));
                    o.d=__floats2bfloat162_rn(__uint_as_float(r6),__uint_as_float(r7));
                    *reinterpret_cast<Bf8*>(&C[(size_t)gm*N + on + col]) = o;
                }
            }
            nbar(1,128);
            if (tid==128){
                if (leader) bar_arrive(&tmem_empty[bt]);
                else        bar_arrive_remote(&tmem_empty[bt],0);
            }
        }
    }

    cluster_sync_fn();
    if (warp==0) tmem_dealloc(tmem_base,512);
}

static CUresult make_tma(CUtensorMap* d, void* addr, uint64_t inner, uint64_t outer, uint32_t bi, uint32_t bo){
    uint64_t gdim[2]={inner,outer}; uint64_t gstr[1]={inner*2}; uint32_t bdim[2]={bi,bo}; uint32_t estr[2]={1,1};
    return cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, addr, gdim, gstr, bdim, estr,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C){
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    int M=(int)A.size(0), K=(int)A.size(1), N=(int)B.size(0);
    __nv_bfloat16* Aptr=static_cast<__nv_bfloat16*>(A.data_ptr());
    __nv_bfloat16* Bptr=static_cast<__nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* Cptr=static_cast<__nv_bfloat16*>(C.data_ptr());

    CUtensorMap dA,dB;
    CU_CHECK(make_tma(&dA,Aptr,(uint64_t)K,(uint64_t)M,KW,128));
    CU_CHECK(make_tma(&dB,Bptr,(uint64_t)K,(uint64_t)N,KW,128));

    size_t smem_bytes=(size_t)2*STAGES*TILE_BYTES + (size_t)(2*STAGES+4)*8 + 16;
    CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem_bytes));

    int num_m=(M+255)/256, num_n=N/256, T=num_m*num_n;

    int dev; CUDA_CHECK(cudaGetDevice(&dev));
    int smCount; CUDA_CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, dev));
    int NC = smCount/2; if(NC<1)NC=1; if(NC>T)NC=T;

    dim3 grid(2, NC, 1); dim3 block(256);
    cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type,A.device().device_id));

    cudaLaunchConfig_t cfg={}; cfg.gridDim=grid; cfg.blockDim=block; cfg.dynamicSmemBytes=smem_bytes; cfg.stream=stream;
    cudaLaunchAttribute attrs[1]; attrs[0].id=cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x=2; attrs[0].val.clusterDim.y=1; attrs[0].val.clusterDim.z=1;
    cfg.attrs=attrs; cfg.numAttrs=1;

    CUDA_CHECK(cudaLaunchKernelEx(&cfg, gemm_kernel, dA, dB, Cptr, M, N, K, T, num_m, num_n));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_blackwell::run);

}  // namespace gemm_blackwell