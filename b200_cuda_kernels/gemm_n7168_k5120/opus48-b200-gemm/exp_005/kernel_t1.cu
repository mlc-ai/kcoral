#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__);exit(1);} } while(0)
#define CU_CHECK(call) do { CUresult _e=(call); if(_e!=CUDA_SUCCESS){const char* s=nullptr; cuGetErrorString(_e,&s); fprintf(stderr,"CU error %s at %s:%d\n",s?s:"?",__FILE__,__LINE__);exit(1);} } while(0)

namespace gemm_sm100 {

constexpr int BM = 128;   // per-CTA M rows
constexpr int BN = 128;   // per-CTA N cols loaded (combined N = 256)
constexpr int BK = 64;
constexpr int NUM_STAGES = 6;
constexpr int KDIM = 5120;
constexpr int NDIM = 7168;
constexpr int NUM_K_TILES = KDIM / BK;           // 80
constexpr int N_COMB = 256;                      // combined N
constexpr int M_COMB = 256;                      // combined M
constexpr uint32_t TOTAL_BYTES = 4u * (BM*BK*2); // A0+A1+B0+B1

__device__ __forceinline__ void init_bar(uint64_t* bar, int count){
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"::"r"((uint32_t)__cvta_generic_to_shared(bar)),"r"(count));
}
__device__ __forceinline__ void bar_arrive_expect(uint64_t* bar, uint32_t tx){
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"::"r"((uint32_t)__cvta_generic_to_shared(bar)),"r"(tx):"memory");
}
__device__ __forceinline__ void bar_wait(uint64_t* bar, uint32_t phase){
    asm volatile("{\n.reg .pred P;\nLAB_%=:\nmbarrier.try_wait.parity.shared.b64 P,[%0],%1;\n@!P bra LAB_%=;\n}\n"::"r"((uint32_t)__cvta_generic_to_shared(bar)),"r"(phase));
}
__device__ __forceinline__ void fence_bar_init(){ asm volatile("fence.mbarrier_init.release.cluster;\n":::"memory"); }
__device__ __forceinline__ void cluster_sync(){ asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n":::"memory"); }
__device__ __forceinline__ uint32_t cluster_rank(){ uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;":"=r"(r)); return r; }

// cta_group::2 TMA load (barrier forced to leader/peer-0)
__device__ __forceinline__ void tma_load_cg2(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1){
    uint32_t sa=(uint32_t)__cvta_generic_to_shared(smem);
    uint32_t ba=((uint32_t)__cvta_generic_to_shared(bar))&0xFEFFFFFF;
    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0],[%1,{%2,%3}],[%4];"
        ::"r"(sa),"l"((uint64_t)d),"r"(c0),"r"(c1),"r"(ba):"memory");
}
__device__ __forceinline__ uint64_t make_desc(void* smem, uint32_t lbo, uint32_t sbo){
    uint64_t dd=0; uint32_t addr=(uint32_t)__cvta_generic_to_shared(smem);
    dd |= (uint64_t)((addr&0x3FFFF)>>4);
    dd |= (uint64_t)((lbo&0x3FFFF)>>4)<<16;
    dd |= (uint64_t)((sbo&0x3FFFF)>>4)<<32;
    dd |= (uint64_t)1<<46;
    dd |= (uint64_t)2<<61;   // 128B swizzle
    return dd;
}
__device__ __forceinline__ uint32_t make_idesc(uint32_t M, uint32_t N){
    uint32_t dd=0;
    dd |= (1u<<4);    // D FP32
    dd |= (1u<<7);    // A BF16
    dd |= (1u<<10);   // B BF16
    dd |= ((N/8)<<17);
    dd |= ((M/16)<<24);
    return dd;
}
__device__ __forceinline__ void umma_cg2(uint32_t tmem, uint64_t da, uint64_t db, uint32_t idesc, uint32_t accum){
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p,%4,0;\ntcgen05.mma.cta_group::2.kind::f16 [%0],%1,%2,%3,p;\n}\n"
        ::"r"(tmem),"l"(da),"l"(db),"r"(idesc),"r"(accum));
}
__device__ __forceinline__ void umma_commit_2sm(uint64_t* bar){
    uint32_t a=(uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
        ::"r"(a),"h"((uint16_t)0x3));
}
__device__ __forceinline__ void tmem_alloc(uint32_t* dst,int ncols){
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0],%1;"::"r"((uint32_t)__cvta_generic_to_shared(dst)),"r"(ncols):"memory");
}
__device__ __forceinline__ void tmem_dealloc(uint32_t addr,int ncols){
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0,%1;"::"r"(addr),"r"(ncols):"memory");
}

__global__ __launch_bounds__(128) void gemm_kernel(const __grid_constant__ CUtensorMap tma_A,
                            const __grid_constant__ CUtensorMap tma_B,
                            __nv_bfloat16* __restrict__ C, int M){
    extern __shared__ char smem_raw[];
    uintptr_t b0 = ((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023;
    __nv_bfloat16* A_smem = (__nv_bfloat16*)b0;
    __nv_bfloat16* B_smem = A_smem + NUM_STAGES*BM*BK;
    uint64_t* full_bar  = (uint64_t*)(B_smem + NUM_STAGES*BN*BK);
    uint64_t* empty_bar = full_bar + NUM_STAGES;
    uint64_t* final_bar = empty_bar + NUM_STAGES;
    uint32_t* tmem_ptr  = (uint32_t*)(final_bar + 1);
    __nv_bfloat16* smem_out = A_smem; // reuse in epilogue

    int tid = threadIdx.x;
    int warp = tid>>5;
    uint32_t rank = cluster_rank();

    int n_pair_tiles = NDIM / N_COMB;              // 28
    int cluster_id = blockIdx.x / 2;
    int tile_m = cluster_id / n_pair_tiles;
    int tile_n = cluster_id % n_pair_tiles;
    int m_pair_base = tile_m * M_COMB;
    int n_base = tile_n * N_COMB;

    if (tid==0){
        for(int s=0;s<NUM_STAGES;s++){ init_bar(&full_bar[s],1); init_bar(&empty_bar[s],1);}
        init_bar(final_bar,1);
    }
    __syncthreads();
    if (warp==0) tmem_alloc(tmem_ptr, N_COMB);
    fence_bar_init();
    cluster_sync();
    uint32_t tmem_base = *tmem_ptr;
    uint32_t idesc = make_idesc(M_COMB, N_COMB);

    bool is_producer = (tid==0);
    bool is_consumer = (tid==32) && (rank==0);

    // A rows for this CTA, B cols for this CTA
    int m_row = m_pair_base + rank*BM;
    int n_col = n_base + rank*BN;

    if (is_producer){
        for(int kt=0; kt<NUM_K_TILES; kt++){
            int stage = kt % NUM_STAGES;
            int r = kt / NUM_STAGES;
            if (kt >= NUM_STAGES){
                bar_wait(&empty_bar[stage], (uint32_t)((r-1)&1));
            }
            if (rank==0){
                bar_arrive_expect(&full_bar[stage], TOTAL_BYTES);
            }
            int kc = kt*BK;
            tma_load_cg2(&tma_A, &full_bar[stage], A_smem + stage*BM*BK, kc, m_row);
            tma_load_cg2(&tma_B, &full_bar[stage], B_smem + stage*BN*BK, kc, n_col);
        }
    }
    if (is_consumer){
        for(int kt=0; kt<NUM_K_TILES; kt++){
            int stage = kt % NUM_STAGES;
            int r = kt / NUM_STAGES;
            bar_wait(&full_bar[stage], (uint32_t)(r&1));
            __nv_bfloat16* Ap = A_smem + stage*BM*BK;
            __nv_bfloat16* Bp = B_smem + stage*BN*BK;
            #pragma unroll
            for(int j=0;j<4;j++){
                uint64_t da = make_desc((char*)Ap + 32*j, 1, 1024);
                uint64_t db = make_desc((char*)Bp + 32*j, 1, 1024);
                uint32_t accum = (kt==0 && j==0) ? 0u : 1u;
                umma_cg2(tmem_base, da, db, idesc, accum);
            }
            if (kt == NUM_K_TILES-1) umma_commit_2sm(final_bar);
            else umma_commit_2sm(&empty_bar[stage]);
        }
    }

    bar_wait(final_bar, 0);
    __syncthreads();

    // Epilogue Phase 1: TMEM -> SMEM (each thread owns a row of its CTA's 128x256 tile)
    for(int col=0; col<N_COMB; col+=4){
        uint32_t r0,r1,r2,r3;
        uint32_t addr = tmem_base + col;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3},[%4];":"=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3):"r"(addr));
        asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory");
        uint32_t base = tid*N_COMB + col;
        smem_out[base+0]=__float2bfloat16(__uint_as_float(r0));
        smem_out[base+1]=__float2bfloat16(__uint_as_float(r1));
        smem_out[base+2]=__float2bfloat16(__uint_as_float(r2));
        smem_out[base+3]=__float2bfloat16(__uint_as_float(r3));
    }
    __syncthreads();
    // Phase 2: SMEM -> global (coalesced uint4 stores)
    int warp_id = tid>>5;
    int lane_id = tid&31;
    int m_out_base = m_pair_base + rank*BM;
    #pragma unroll
    for(int step=0; step<BM/4; step++){
        int row = step*4 + warp_id;
        int gr = m_out_base + row;
        if (gr < M){
            int col_start = lane_id*8;
            int gc = n_base + col_start;
            uint4 data = *reinterpret_cast<uint4*>(&smem_out[row*N_COMB + col_start]);
            *reinterpret_cast<uint4*>(C + (uint64_t)gr*NDIM + gc) = data;
        }
    }
    cluster_sync();
    if (warp==0) tmem_dealloc(tmem_base, N_COMB);
}

static CUresult make_tma(CUtensorMap* d, void* ptr, uint64_t inner, uint64_t outer, uint32_t box_inner, uint32_t box_outer){
    uint64_t gdim[2]={inner,outer};
    uint64_t gstr[1]={inner*2};
    uint32_t bdim[2]={box_inner,box_outer};
    uint32_t estr[2]={1,1};
    return cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, ptr, gdim, gstr, bdim, estr,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C){
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    int M = (int)A.size(0);
    __nv_bfloat16* a=(__nv_bfloat16*)A.data_ptr();
    __nv_bfloat16* b=(__nv_bfloat16*)B.data_ptr();
    __nv_bfloat16* c=(__nv_bfloat16*)C.data_ptr();

    CUtensorMap tA, tB;
    CU_CHECK(make_tma(&tA, a, (uint64_t)KDIM, (uint64_t)M, BK, BM));
    CU_CHECK(make_tma(&tB, b, (uint64_t)KDIM, (uint64_t)NDIM, BK, BN));

    size_t pipe = (size_t)NUM_STAGES*(BM*BK + BN*BK)*sizeof(__nv_bfloat16);
    size_t smem_bytes = pipe + (2*NUM_STAGES+1)*sizeof(uint64_t) + sizeof(uint32_t) + 1024;
    CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));

    int n_pair_tiles = NDIM / N_COMB;              // 28
    int m_pair_tiles = (M + M_COMB - 1)/M_COMB;
    int grid_x = m_pair_tiles * n_pair_tiles * 2;  // 2 CTAs per pair
    cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(A.device().device_type, A.device().device_id);

    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(grid_x,1,1);
    config.blockDim = dim3(128,1,1);
    config.dynamicSmemBytes = smem_bytes;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_kernel, tA, tB, c, M));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_sm100::run);

}  // namespace gemm_sm100