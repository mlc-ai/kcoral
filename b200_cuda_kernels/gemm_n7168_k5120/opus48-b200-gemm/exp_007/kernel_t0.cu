#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

namespace gemm_n7168_k5120 {

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__); exit(1);} } while(0)
#define CU_CHECK(call) do { CUresult _e=(call); if(_e!=CUDA_SUCCESS){ const char* s=nullptr; cuGetErrorString(_e,&s); fprintf(stderr,"CU error %s at %s:%d\n", s?s:"?", __FILE__,__LINE__); exit(1);} } while(0)

__device__ __forceinline__ void init_smem_barrier(uint64_t* bar, uint32_t count){
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(bar)),"r"(count));
}
__device__ __forceinline__ void fence_barrier_init(){ asm volatile("fence.mbarrier_init.release.cluster;\n":::"memory"); }
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t tx){
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(bar)),"r"(tx):"memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase){
    asm volatile("{\n.reg .pred P;\nWAIT_%=:\nmbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n@!P bra WAIT_%=;\n}\n" :: "r"((uint32_t)__cvta_generic_to_shared(bar)),"r"(phase));
}
__device__ __forceinline__ void tma_load_2d(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1){
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),"l"((uint64_t)d),"r"((uint32_t)__cvta_generic_to_shared(bar)),"r"(c0),"r"(c1):"memory");
}
__device__ __forceinline__ uint64_t make_smem_desc(void* ptr, uint32_t lbo, uint32_t sbo){
    uint64_t d=0; uint32_t a=(uint32_t)__cvta_generic_to_shared(ptr);
    d |= ((uint64_t)(a & 0x3FFFFu)) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFFu)>>4)<<16;
    d |= (uint64_t)((sbo & 0x3FFFFu)>>4)<<32;
    d |= (uint64_t)1<<46;
    d |= (uint64_t)2<<61; // SWIZZLE_128B
    return d;
}
__device__ __forceinline__ uint32_t make_idesc(uint32_t M, uint32_t N){
    uint32_t d=0;
    d |= (1u<<4);   // c FP32
    d |= (1u<<7);   // a BF16
    d |= (1u<<10);  // b BF16
    d |= ((N/8)<<17);
    d |= ((M/16)<<24);
    return d;
}
__device__ __forceinline__ void tmem_alloc(uint32_t* dst, int ncols){
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(dst)),"r"(ncols));
}
__device__ __forceinline__ void tmem_dealloc(uint32_t addr, int ncols){
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(addr),"r"(ncols));
}
__device__ __forceinline__ void umma_f16(uint32_t tmem_c, uint64_t da, uint64_t db, uint32_t idesc, uint32_t accum){
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c),"l"(da),"l"(db),"r"(idesc),"r"(accum));
}
__device__ __forceinline__ void umma_commit(uint64_t* bar){
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"((uint32_t)__cvta_generic_to_shared(bar)));
}
__device__ __forceinline__ void tcgen05_fence_after(){ asm volatile("tcgen05.fence::after_thread_sync;":::"memory"); }

constexpr int BM=128, BN=256, BK=64, STAGES=4;
constexpr int A_TILE=BM*BK; // 8192 elems
constexpr int B_TILE=BN*BK; // 16384 elems

__global__ __launch_bounds__(128) void gemm_kernel(int M, int N, int K,
        __nv_bfloat16* __restrict__ C,
        const __grid_constant__ CUtensorMap dA,
        const __grid_constant__ CUtensorMap dB){
    extern __shared__ char smem_raw[];
    const int tid=threadIdx.x;
    const int warp=tid>>5;
    const int lane=tid&31;
    const int n_block=blockIdx.x;
    const int m_block=blockIdx.y;

    uint32_t base_sh=(uint32_t)__cvta_generic_to_shared(smem_raw);
    uint32_t aln=(base_sh+1023u)&~1023u;
    char* smem=smem_raw+(aln-base_sh);

    __nv_bfloat16* A_smem=(__nv_bfloat16*)(smem);
    __nv_bfloat16* B_smem=(__nv_bfloat16*)(smem + STAGES*A_TILE*2);
    uint64_t* full=(uint64_t*)(smem + STAGES*A_TILE*2 + STAGES*B_TILE*2);
    uint64_t* empty=full+STAGES;
    uint64_t* mma_done=empty+STAGES;
    uint32_t* tmem_addr=(uint32_t*)(mma_done+1);
    __nv_bfloat16* smem_out=(__nv_bfloat16*)(smem);

    const int NUM_K=K/BK;

    if(tid==0){
        #pragma unroll
        for(int s=0;s<STAGES;s++){ init_smem_barrier(&full[s],1); init_smem_barrier(&empty[s],1);}
        init_smem_barrier(mma_done,1);
        fence_barrier_init();
    }
    if(warp==0){ tmem_alloc(tmem_addr, BN); }
    __syncthreads();
    uint32_t tmem_base=*tmem_addr;
    uint32_t idesc=make_idesc(BM,BN);

    if(warp==0 && lane==0){
        // producer
        uint32_t ep[STAGES];
        #pragma unroll
        for(int s=0;s<STAGES;s++) ep[s]=0;
        for(int k=0;k<NUM_K;k++){
            int s=k%STAGES;
            if(k>=STAGES){ mbar_wait(&empty[s], ep[s]); ep[s]^=1; }
            mbar_arrive_expect_tx(&full[s], (uint32_t)((A_TILE+B_TILE)*2));
            tma_load_2d(&dA,&full[s], A_smem+s*A_TILE, k*BK, m_block*BM);
            tma_load_2d(&dB,&full[s], B_smem+s*B_TILE, k*BK, n_block*BN);
        }
    } else if(warp==1 && lane==0){
        // consumer
        uint32_t fp[STAGES];
        #pragma unroll
        for(int s=0;s<STAGES;s++) fp[s]=0;
        bool first=true;
        for(int k=0;k<NUM_K;k++){
            int s=k%STAGES;
            mbar_wait(&full[s], fp[s]); fp[s]^=1;
            char* aptr=(char*)(A_smem+s*A_TILE);
            char* bptr=(char*)(B_smem+s*B_TILE);
            #pragma unroll
            for(int kk=0;kk<BK/16;kk++){
                uint64_t da=make_smem_desc(aptr+kk*32,0,1024);
                uint64_t db=make_smem_desc(bptr+kk*32,0,1024);
                umma_f16(tmem_base,da,db,idesc,first?0u:1u);
                first=false;
            }
            umma_commit(&empty[s]);
        }
        umma_commit(mma_done);
    }

    mbar_wait(mma_done, 0);
    tcgen05_fence_after();
    __syncthreads();

    // epilogue phase 1: TMEM -> smem_out
    for(int col=0;col<BN;col+=8){
        uint32_t r0,r1,r2,r3,r4,r5,r6,r7;
        uint32_t addr=tmem_base+col;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7):"r"(addr));
        asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory");
        int b=tid*BN+col;
        smem_out[b+0]=__float2bfloat16(__uint_as_float(r0));
        smem_out[b+1]=__float2bfloat16(__uint_as_float(r1));
        smem_out[b+2]=__float2bfloat16(__uint_as_float(r2));
        smem_out[b+3]=__float2bfloat16(__uint_as_float(r3));
        smem_out[b+4]=__float2bfloat16(__uint_as_float(r4));
        smem_out[b+5]=__float2bfloat16(__uint_as_float(r5));
        smem_out[b+6]=__float2bfloat16(__uint_as_float(r6));
        smem_out[b+7]=__float2bfloat16(__uint_as_float(r7));
    }
    __syncthreads();

    // epilogue phase 2: coalesced store (uint4 = 8 bf16)
    const int VPR=BN/8; // 32
    const int total=BM*VPR;
    for(int idx=tid; idx<total; idx+=blockDim.x){
        int row=idx/VPR;
        int cvec=idx%VPR;
        int col=cvec*8;
        int grow=m_block*BM+row;
        if(grow<M){
            uint4 data=*reinterpret_cast<uint4*>(&smem_out[row*BN+col]);
            *reinterpret_cast<uint4*>(&C[(int64_t)grow*N + n_block*BN + col])=data;
        }
    }
    __syncthreads();
    if(warp==0){ tmem_dealloc(tmem_base, BN); }
}

static CUresult make_tma(CUtensorMap* d, void* ptr, uint64_t inner, uint64_t outer, uint32_t bi, uint32_t bo){
    uint64_t gd[2]={inner,outer};
    uint64_t gs[1]={inner*2};
    uint32_t bd[2]={bi,bo};
    uint32_t es[2]={1,1};
    return cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, ptr, gd, gs, bd, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C){
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    int64_t M=A.size(0);
    int64_t K=A.size(1);
    int64_t N=B.size(0);
    __nv_bfloat16* Aptr=(__nv_bfloat16*)A.data_ptr();
    __nv_bfloat16* Bptr=(__nv_bfloat16*)B.data_ptr();
    __nv_bfloat16* Cptr=(__nv_bfloat16*)C.data_ptr();

    CUtensorMap dA, dB;
    CU_CHECK(make_tma(&dA, Aptr, (uint64_t)K, (uint64_t)M, BK, BM));
    CU_CHECK(make_tma(&dB, Bptr, (uint64_t)K, (uint64_t)N, BK, BN));

    int mblocks=(int)((M+BM-1)/BM);
    int nblocks=(int)(N/BN);
    dim3 grid(nblocks, mblocks, 1);
    dim3 block(128,1,1);

    size_t core = (size_t)STAGES*A_TILE*2 + (size_t)STAGES*B_TILE*2 + (2*STAGES+1)*8 + 4;
    size_t smem_bytes = core + 1024;

    CUDA_CHECK(cudaFuncSetAttribute((const void*)gemm_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));

    cudaStream_t stream=(cudaStream_t)TVMFFIEnvGetStream(A.device().device_type, A.device().device_id);
    gemm_kernel<<<grid, block, smem_bytes, stream>>>((int)M,(int)N,(int)K, Cptr, dA, dB);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_n7168_k5120::run);

}  // namespace gemm_n7168_k5120