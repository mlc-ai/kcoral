#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <algorithm>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__);exit(1);} } while(0)
#define CU_CHECK(call) do { CUresult _e=(call); if(_e!=CUDA_SUCCESS){const char* s=nullptr; cuGetErrorString(_e,&s); fprintf(stderr,"CU error %s at %s:%d\n",s?s:"?",__FILE__,__LINE__);exit(1);} } while(0)

namespace gemm_sm100 {

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 64;
constexpr int NSUB = 2;
constexpr int NUM_STAGES = 4;
constexpr int KDIM = 5120;
constexpr int NDIM = 7168;
constexpr int NUM_K_TILES = KDIM / BK;
constexpr int N_COMB = 512;
constexpr int M_COMB = 256;
constexpr int GROUP_M = 8;
constexpr uint32_t TOTAL_BYTES = 6u * (BM*BK*2);

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
    dd |= (uint64_t)2<<61;
    return dd;
}
__device__ __forceinline__ uint32_t make_idesc(uint32_t M, uint32_t N){
    uint32_t dd=0;
    dd |= (1u<<4); dd |= (1u<<7); dd |= (1u<<10);
    dd |= ((N/8)<<17); dd |= ((M/16)<<24);
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
                            __nv_bfloat16* __restrict__ C, int M,
                            int num_pid_m, int num_pid_n, int num_clusters, int total_tiles){
    extern __shared__ char smem_raw[];
    uintptr_t b0 = ((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023;
    __nv_bfloat16* A_smem = (__nv_bfloat16*)b0;
    __nv_bfloat16* B_smem = A_smem + NUM_STAGES*BM*BK;
    uint64_t* full_bar  = (uint64_t*)(B_smem + NUM_STAGES*NSUB*BN*BK);
    uint64_t* empty_bar = full_bar + NUM_STAGES;
    uint64_t* final_bar = empty_bar + NUM_STAGES;
    uint64_t* desc_a    = final_bar + 1;
    uint64_t* desc_b    = desc_a + NUM_STAGES;
    uint32_t* tmem_ptr  = (uint32_t*)(desc_b + NUM_STAGES*NSUB);
    __nv_bfloat16* smem_out = A_smem;

    int tid = threadIdx.x;
    int warp = tid>>5;
    uint32_t rank = cluster_rank();

    if (warp==0) tmem_alloc(tmem_ptr, 512);
    __syncthreads();
    uint32_t tmem_base = *tmem_ptr;
    uint32_t idesc = make_idesc(M_COMB, 256);

    bool is_producer = (tid==0);
    bool is_consumer = (tid==32) && (rank==0);

    if (is_consumer){
        #pragma unroll
        for(int s=0;s<NUM_STAGES;s++){
            desc_a[s] = make_desc(A_smem + s*BM*BK, 1, 1024);
            #pragma unroll
            for(int sub=0;sub<NSUB;sub++)
                desc_b[s*NSUB+sub] = make_desc(B_smem + s*NSUB*BN*BK + sub*BN*BK, 1, 1024);
        }
    }

    int cluster_id = blockIdx.x / 2;
    int warp_id = tid>>5;
    int lane_id = tid&31;

    for (int tile = cluster_id; tile < total_tiles; tile += num_clusters){
        // (Re)initialize barriers fresh each tile so phase parities always start at 0
        if (tid==0){
            for(int s=0;s<NUM_STAGES;s++){ init_bar(&full_bar[s],1); init_bar(&empty_bar[s],1);}
            init_bar(final_bar,1);
        }
        __syncthreads();
        fence_bar_init();
        cluster_sync();

        // GROUP_M swizzle for L2 reuse
        int num_pid_in_group = GROUP_M * num_pid_n;
        int group_id = tile / num_pid_in_group;
        int first_pid_m = group_id * GROUP_M;
        int gsize = num_pid_m - first_pid_m; if (gsize > GROUP_M) gsize = GROUP_M;
        int tile_m = first_pid_m + (tile % gsize);
        int tile_n = (tile % num_pid_in_group) / gsize;

        int m_pair_base = tile_m * M_COMB;
        int n_base = tile_n * N_COMB;

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
                tma_load_cg2(&tma_A, &full_bar[stage], A_smem + stage*BM*BK, kc, m_pair_base + rank*BM);
                #pragma unroll
                for(int sub=0; sub<NSUB; sub++){
                    __nv_bfloat16* dst = B_smem + stage*NSUB*BN*BK + sub*BN*BK;
                    int ncol = n_base + sub*256 + rank*BN;
                    tma_load_cg2(&tma_B, &full_bar[stage], dst, kc, ncol);
                }
            }
        }
        if (is_consumer){
            for(int kt=0; kt<NUM_K_TILES; kt++){
                int stage = kt % NUM_STAGES;
                int r = kt / NUM_STAGES;
                bar_wait(&full_bar[stage], (uint32_t)(r&1));
                uint64_t da0 = desc_a[stage];
                #pragma unroll
                for(int sub=0; sub<NSUB; sub++){
                    uint64_t db0 = desc_b[stage*NSUB+sub];
                    uint32_t tmem_sub = tmem_base + sub*256;
                    uint32_t accum = (kt==0) ? 0u : 1u;
                    #pragma unroll
                    for(int j=0;j<4;j++){
                        umma_cg2(tmem_sub, da0 + (uint64_t)(2*j), db0 + (uint64_t)(2*j), idesc, accum);
                        accum = 1u;
                    }
                }
                if (kt == NUM_K_TILES-1) umma_commit_2sm(final_bar);
                else umma_commit_2sm(&empty_bar[stage]);
            }
        }

        bar_wait(final_bar, 0);
        __syncthreads();

        // Epilogue Phase 1: TMEM -> SMEM (batched waits)
        for(int base=0; base<512; base+=32){
            uint32_t rr[32];
            #pragma unroll
            for(int c=0;c<8;c++){
                uint32_t addr = tmem_base + base + c*4;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3},[%4];"
                    :"=r"(rr[c*4]),"=r"(rr[c*4+1]),"=r"(rr[c*4+2]),"=r"(rr[c*4+3]):"r"(addr));
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory");
            uint32_t obase = tid*512 + base;
            #pragma unroll
            for(int c=0;c<32;c++){
                smem_out[obase + c] = __float2bfloat16(__uint_as_float(rr[c]));
            }
        }
        __syncthreads();
        // Phase 2: SMEM -> global (coalesced uint4)
        int m_out_base = m_pair_base + rank*BM;
        #pragma unroll
        for(int step=0; step<BM/4; step++){
            int row = step*4 + warp_id;
            int gr = m_out_base + row;
            if (gr < M){
                #pragma unroll
                for(int u=0; u<2; u++){
                    int col = lane_id*8 + u*256;
                    int gc = n_base + col;
                    uint4 data = *reinterpret_cast<uint4*>(&smem_out[row*512 + col]);
                    *reinterpret_cast<uint4*>(C + (uint64_t)gr*NDIM + gc) = data;
                }
            }
        }
        __syncthreads();
    }

    cluster_sync();
    if (warp==0) tmem_dealloc(tmem_base, 512);
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

    size_t pipe = (size_t)NUM_STAGES*(BM*BK + NSUB*BN*BK)*sizeof(__nv_bfloat16);
    size_t smem_bytes = pipe + (2*NUM_STAGES+1)*sizeof(uint64_t)
                        + (NUM_STAGES + NUM_STAGES*NSUB)*sizeof(uint64_t)
                        + sizeof(uint32_t) + 1024;
    CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));

    int num_pid_n = NDIM / N_COMB;
    int num_pid_m = (M + M_COMB - 1)/M_COMB;
    int total_tiles = num_pid_m * num_pid_n;

    int dev; CUDA_CHECK(cudaGetDevice(&dev));
    int sm_count; CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev));
    int max_clusters = sm_count / 2;
    int num_clusters = std::min(max_clusters, total_tiles);
    if (num_clusters < 1) num_clusters = 1;

    int grid_x = num_clusters * 2;
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
    CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_kernel, tA, tB, c, M, num_pid_m, num_pid_n, num_clusters, total_tiles));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_sm100::run);

}  // namespace gemm_sm100