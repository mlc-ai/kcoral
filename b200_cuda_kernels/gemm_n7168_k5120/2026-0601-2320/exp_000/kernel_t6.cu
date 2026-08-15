#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__);} } while(0)
#define CU_CHECK(call) do { CUresult _e=(call); if(_e!=CUDA_SUCCESS){ const char* s; cuGetErrorString(_e,&s); \
    fprintf(stderr,"CU error %s at %s:%d\n", s, __FILE__, __LINE__);} } while(0)

namespace gemm_n7168_k5120 {

#define BM_CTA 128
#define KSUB 16
#define NSTAGE 12
#define COMB_M 256
#define COMB_N 512
#define NGROUP 2          // 2 N-halves of 256
#define NHALF 256
#define BNH 128           // per-CTA N cols per group
#define A_SUB (BM_CTA*KSUB)   // 2048
#define BG_SUB (BNH*KSUB)     // 2048 per group per CTA
#define A_STAGE (A_SUB)
#define B_STAGE (NGROUP*BG_SUB)  // 4096

__device__ __forceinline__ void init_bar(uint64_t* bar, uint32_t c){
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(c));
}
__device__ __forceinline__ void fence_bar_init(){
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}
__device__ __forceinline__ void arrive_expect_tx(uint64_t* bar, uint32_t tx){
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(tx) : "memory");
}
__device__ __forceinline__ void bar_wait(uint64_t* bar, uint32_t ph){
    asm volatile("{\n.reg .pred P;\nWT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WT_%=;\n}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(ph));
}
__device__ __forceinline__ void tma_load_cg2(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1){
    uint32_t sa=(uint32_t)__cvta_generic_to_shared(smem);
    uint32_t ba=(uint32_t)__cvta_generic_to_shared(bar);
    uint32_t bar0; uint32_t zero=0;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(bar0) : "r"(ba), "r"(zero));
    asm volatile(
      "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
      "[%0], [%1, {%2, %3}], [%4];"
      :: "r"(sa), "l"((uint64_t)d), "r"(c0), "r"(c1), "r"(bar0) : "memory");
}

__device__ __forceinline__ void tmem_alloc_cg2(uint32_t* dst, int ncols){
    uint32_t a=(uint32_t)__cvta_generic_to_shared(dst);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a), "r"(ncols));
}
__device__ __forceinline__ void tmem_relinquish_cg2(){
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
}
__device__ __forceinline__ void tmem_dealloc_cg2(uint32_t addr,int ncols){
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(addr), "r"(ncols));
}
__device__ __forceinline__ void umma_cg2(uint32_t tc, uint64_t da, uint64_t db, uint32_t idesc, uint32_t accum){
    asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
      "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
      :: "r"(tc), "l"(da), "l"(db), "r"(idesc), "r"(accum));
}
__device__ __forceinline__ void umma_commit_2sm(uint64_t* bar){
    uint32_t a=(uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
        :: "r"(a), "h"((uint16_t)0x3));
}
__device__ __forceinline__ void fence_after(){ asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void cluster_sync(){ asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n" ::: "memory"); }
__device__ __forceinline__ uint32_t crank(){ uint32_t r; asm volatile("mov.u32 %0,%%cluster_ctarank;":"=r"(r)); return r; }

__device__ __forceinline__ uint64_t make_desc32(void* p){
    uint64_t d=0; uint32_t a=(uint32_t)__cvta_generic_to_shared(p);
    d |= (uint64_t)((a&0x3FFFFu)>>4);
    d |= ((uint64_t)1)<<16;
    d |= ((uint64_t)((256u&0x3FFFFu)>>4))<<32;
    d |= ((uint64_t)1)<<46;
    d |= ((uint64_t)6)<<61;
    return d;
}

__global__ void __launch_bounds__(128) gemm_kernel(
    const __grid_constant__ CUtensorMap descA,
    const __grid_constant__ CUtensorMap descB,
    __nv_bfloat16* __restrict__ C, int M, int N, int K)
{
    extern __shared__ char smem_raw[];
    uint32_t sbase=(uint32_t)__cvta_generic_to_shared(smem_raw);
    uint32_t pad=(1024u-(sbase&1023u))&1023u;
    char* sm=smem_raw+pad;
    __nv_bfloat16* As=reinterpret_cast<__nv_bfloat16*>(sm);
    __nv_bfloat16* Bs=As + NSTAGE*A_STAGE;
    uint64_t* full=reinterpret_cast<uint64_t*>(Bs + NSTAGE*B_STAGE);
    uint64_t* empty=full+NSTAGE;
    uint64_t* mma_done=empty+NSTAGE;
    uint32_t* tmem_ptr=reinterpret_cast<uint32_t*>(mma_done+1);
    __nv_bfloat16* smem_out=As;

    int tid=threadIdx.x;
    int warp=tid>>5;
    uint32_t rank=crank();

    int n_tile = blockIdx.x>>1;
    int m_cluster = blockIdx.y;
    int a_base = m_cluster*COMB_M + (int)rank*BM_CTA;   // this CTA's A rows
    int m_out_base = a_base;
    int n_out_base = n_tile*COMB_N;

    int KT = K/KSUB;

    if (tid==0){
        #pragma unroll
        for(int s=0;s<NSTAGE;s++){ init_bar(&full[s],1); init_bar(&empty[s],1); }
        init_bar(&mma_done[0],1);
        fence_bar_init();
    }
    if (warp==0){ tmem_alloc_cg2(tmem_ptr,COMB_N); tmem_relinquish_cg2(); }
    cluster_sync();
    uint32_t tmem_base=tmem_ptr[0];

    if (tid==32){
        // producer (both CTAs)
        for(int st=0; st<KT; ++st){
            int s=st%NSTAGE;
            if(st>=NSTAGE){ int ph=((st/NSTAGE)-1)&1; bar_wait(&empty[s], ph); }
            // total bytes arriving at full[s]: 2 CTA * (A + 2*Bgroup)
            if(rank==0) arrive_expect_tx(&full[s], 2u*(A_SUB + NGROUP*BG_SUB)*2u);
            int c0=st*KSUB;
            tma_load_cg2(&descA, &full[s], As + s*A_STAGE, c0, a_base);
            #pragma unroll
            for(int g=0; g<NGROUP; ++g){
                int n_base = n_tile*COMB_N + g*NHALF + (int)rank*BNH;
                tma_load_cg2(&descB, &full[s], Bs + s*B_STAGE + g*BG_SUB, c0, n_base);
            }
        }
    } else if (tid==0 && rank==0){
        // consumer (CTA0 only) issues shared 2-SM UMMAs
        uint32_t idesc=0;
        idesc |= (1u<<4); idesc |= (1u<<7); idesc |= (1u<<10);
        idesc |= ((NHALF/8)<<17);
        idesc |= ((COMB_M/16)<<24);
        for(int st=0; st<KT; ++st){
            int s=st%NSTAGE; int ph=(st/NSTAGE)&1;
            bar_wait(&full[s], ph);
            uint64_t da=make_desc32(As + s*A_STAGE);
            #pragma unroll
            for(int g=0; g<NGROUP; ++g){
                uint64_t db=make_desc32(Bs + s*B_STAGE + g*BG_SUB);
                umma_cg2(tmem_base + g*NHALF, da, db, idesc, (st==0)?0u:1u);
            }
            umma_commit_2sm(&empty[s]);
        }
        umma_commit_2sm(&mma_done[0]);
    }

    bar_wait(&mma_done[0], 0);
    __syncthreads();
    fence_after();

    // epilogue per N-group (limits smem usage to 64KB)
    const int VPR=NHALF/4; // 64
    #pragma unroll
    for(int g=0; g<NGROUP; ++g){
        #pragma unroll
        for(int col=0; col<NHALF; col+=4){
            uint32_t r0,r1,r2,r3;
            uint32_t ta=tmem_base + (uint32_t)(g*NHALF + col);
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(ta));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            int base=tid*NHALF+col;
            smem_out[base+0]=__float2bfloat16(__uint_as_float(r0));
            smem_out[base+1]=__float2bfloat16(__uint_as_float(r1));
            smem_out[base+2]=__float2bfloat16(__uint_as_float(r2));
            smem_out[base+3]=__float2bfloat16(__uint_as_float(r3));
        }
        __syncthreads();
        for(int v=tid; v<BM_CTA*VPR; v+=128){
            int r=v/VPR; int cv=v%VPR; int col=cv*4;
            int gr=m_out_base+r; int gc=n_out_base + g*NHALF + col;
            if(gr<M){
                uint2 d=*reinterpret_cast<uint2*>(&smem_out[r*NHALF+col]);
                *reinterpret_cast<uint2*>(&C[(size_t)gr*N+gc])=d;
            }
        }
        __syncthreads();
    }
    cluster_sync();
    if(warp==0) tmem_dealloc_cg2(tmem_base, COMB_N);
}

static CUresult make_tma(CUtensorMap* d, void* g, uint64_t inner, uint64_t outer, uint32_t bi, uint32_t bo){
    uint64_t gd[2]={inner,outer}; uint64_t gs[1]={inner*2};
    uint32_t bd[2]={bi,bo}; uint32_t es[2]={1,1};
    return cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, g, gd, gs, bd, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_32B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C){
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    int M=(int)A.size(0), K=(int)A.size(1), N=(int)B.size(0);
    __nv_bfloat16* Ap=static_cast<__nv_bfloat16*>(A.data_ptr());
    __nv_bfloat16* Bp=static_cast<__nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* Cp=static_cast<__nv_bfloat16*>(C.data_ptr());

    CUtensorMap dA,dB;
    CU_CHECK(make_tma(&dA, Ap, (uint64_t)K, (uint64_t)M, KSUB, BM_CTA));
    CU_CHECK(make_tma(&dB, Bp, (uint64_t)K, (uint64_t)N, KSUB, BNH));

    int num_n = N/COMB_N;            // 14
    int num_mc = (M+COMB_M-1)/COMB_M;// 32
    size_t smem = 1024 + (size_t)(NSTAGE*A_STAGE + NSTAGE*B_STAGE)*2 + (size_t)(NSTAGE*2+1)*8 + 16;

    CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));

    cudaStream_t stream=static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    cudaLaunchConfig_t cfg={};
    cfg.gridDim=dim3(num_n*2, num_mc, 1);
    cfg.blockDim=dim3(128,1,1);
    cfg.dynamicSmemBytes=smem;
    cfg.stream=stream;
    cudaLaunchAttribute attr[1];
    attr[0].id=cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x=2; attr[0].val.clusterDim.y=1; attr[0].val.clusterDim.z=1;
    cfg.attrs=attr; cfg.numAttrs=1;
    CUDA_CHECK(cudaLaunchKernelEx(&cfg, gemm_kernel, dA, dB, Cp, M, N, K));
    CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_n7168_k5120::run);

}  // namespace gemm_n7168_k5120