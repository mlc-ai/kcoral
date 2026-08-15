#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <cstdio>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e));exit(1);} }while(0)
#define CU_CHECK(call) do { CUresult _e=(call); if(_e!=CUDA_SUCCESS){const char*s; cuGetErrorString(_e,&s); fprintf(stderr,"CU %s:%d %s\n",__FILE__,__LINE__,s);exit(1);} }while(0)

namespace gemm_blackwell {

constexpr int ROWS   = 128;
constexpr int KW     = 64;
constexpr int BK     = 64;
constexpr int STAGES = 6;
constexpr int TILE_ELEMS = ROWS*KW;
constexpr int TILE_BYTES = ROWS*KW*2;
constexpr uint32_t TX_BYTES = 4u*TILE_BYTES;
constexpr int GROUP_M = 8;

__device__ __forceinline__ bool elect_one_sync_fn() {
    uint32_t pred;
    asm volatile("{\n.reg .pred p;\n elect.sync _|p, 0xFFFFFFFF;\n selp.b32 %0, 1, 0, p;\n}\n":"=r"(pred));
    return pred!=0;
}
__device__ __forceinline__ void cluster_sync_fn(){ asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n":::"memory"); }
__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count){
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"::"r"((uint32_t)__cvta_generic_to_shared(bar)),"r"(count));
}
__device__ __forceinline__ void fence_smem_barrier_init_fn(){ asm volatile("fence.mbarrier_init.release.cluster;\n":::"memory"); }
__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx){
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"::"r"((uint32_t)__cvta_generic_to_shared(bar)),"r"(tx):"memory");
}
__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase){
    asm volatile("{\n.reg .pred P;\nWT_%=:\nmbarrier.try_wait.parity.shared.b64 P,[%0],%1;\n@!P bra WT_%=;\n}\n"
        ::"r"((uint32_t)__cvta_generic_to_shared(bar)),"r"(phase));
}
__device__ __forceinline__ void tma_load_2d_cg2_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1){
    uint32_t sa=(uint32_t)__cvta_generic_to_shared(smem);
    uint32_t ba=(uint32_t)__cvta_generic_to_shared(bar) & 0xFEFFFFFF;
    asm volatile("cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];"
        ::"r"(sa),"l"((uint64_t)d),"r"(c0),"r"(c1),"r"(ba):"memory");
}
__device__ __forceinline__ uint64_t make_smem_desc_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo){
    uint64_t dsc=0; uint32_t addr=(uint32_t)__cvta_generic_to_shared(smem_ptr);
    dsc |= (uint64_t)(addr & 0x3FFFF) >> 4;
    dsc |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    dsc |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    dsc |= (uint64_t)1 << 46;
    dsc |= (uint64_t)2 << 61;
    return dsc;
}
__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N){
    uint32_t d=0; d|=(1u<<4); d|=(1u<<7); d|=(1u<<10); d|=((N/8)<<17); d|=((M/16)<<24); return d;
}
__device__ __forceinline__ void umma_f16_cg2_fn(uint32_t tmem_c, uint64_t da, uint64_t db, uint32_t idesc, uint32_t accum){
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p,%4,0;\n tcgen05.mma.cta_group::2.kind::f16 [%0],%1,%2,%3,p;\n}\n"
        ::"r"(tmem_c),"l"(da),"l"(db),"r"(idesc),"r"(accum));
}
__device__ __forceinline__ void umma_commit_2sm_fn(uint64_t* bar){
    uint32_t a=(uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0],%1;"
        ::"r"(a),"h"((uint16_t)0x3));
}
__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols){
    uint32_t a=(uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0],%1;"::"r"(a),"r"(ncols));
}
__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols){
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0,%1;"::"r"(addr),"r"(ncols));
}

__global__ __launch_bounds__(128) void gemm_kernel(
    const __grid_constant__ CUtensorMap descA,
    const __grid_constant__ CUtensorMap descB,
    __nv_bfloat16* __restrict__ C, int M, int N, int K)
{
    extern __shared__ __align__(1024) unsigned char smem[];
    __nv_bfloat16* A_sm = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* B_sm = A_sm + STAGES*TILE_ELEMS;
    uint64_t* full_bar  = reinterpret_cast<uint64_t*>(B_sm + STAGES*TILE_ELEMS);
    uint64_t* empty_bar = full_bar + STAGES;
    uint64_t* mma_done  = empty_bar + STAGES;
    uint32_t* tmem_ptr  = reinterpret_cast<uint32_t*>(mma_done + 1);
    __nv_bfloat16* smem_out = A_sm;

    int tid  = threadIdx.x;
    int warp = tid >> 5;
    int rank = blockIdx.x;
    bool leader = (rank == 0);

    // grouped rasterization for L2 locality
    int num_m = (M + 255)/256;
    int num_n = N/256;
    int s = blockIdx.y;
    int tiles_in_group = GROUP_M * num_n;
    int group = s / tiles_in_group;
    int first_m = group * GROUP_M;
    int rows = min(GROUP_M, num_m - first_m);
    int local = s % tiles_in_group;
    int m_tile = first_m + (local % rows);
    int n_tile = local / rows;

    int my_m_start = m_tile*256 + rank*128;
    int my_n_start = n_tile*256 + rank*128;
    int out_n_start = n_tile*256;

    int num_kt = K / BK;

    if (tid == 0) {
        for (int s2=0;s2<STAGES;s2++){ init_smem_barrier_fn(&full_bar[s2],1); init_smem_barrier_fn(&empty_bar[s2],1); }
        init_smem_barrier_fn(mma_done,1);
    }
    if (warp == 0) tmem_alloc_fn(tmem_ptr, 256);
    __syncthreads();
    fence_smem_barrier_init_fn();
    cluster_sync_fn();

    uint32_t tmem_base = tmem_ptr[0];
    uint32_t idesc = make_instr_desc_fn(256,256);
    const uint32_t SBO = 8u*KW*2u;

    if (warp == 0) {
        bool el = elect_one_sync_fn();
        for (int kt=0; kt<num_kt; kt++){
            int buf = kt % STAGES;
            if (kt >= STAGES){
                int ph = ((kt/STAGES)-1) & 1;
                if (el) mbarrier_wait_fn(&empty_bar[buf], ph);
            }
            if (el){
                if (leader) mbarrier_arrive_and_expect_tx_fn(&full_bar[buf], TX_BYTES);
                tma_load_2d_cg2_fn(&descA, &full_bar[buf], A_sm + buf*TILE_ELEMS, kt*BK, my_m_start);
                tma_load_2d_cg2_fn(&descB, &full_bar[buf], B_sm + buf*TILE_ELEMS, kt*BK, my_n_start);
            }
        }
    } else if (warp == 1 && leader) {
        bool el = elect_one_sync_fn();
        if (el){
            for (int kt=0; kt<num_kt; kt++){
                int buf = kt % STAGES;
                int ph = (kt/STAGES) & 1;
                mbarrier_wait_fn(&full_bar[buf], ph);
                #pragma unroll
                for (int ks=0; ks<4; ks++){
                    uint64_t da = make_smem_desc_fn(A_sm + buf*TILE_ELEMS + ks*16, 1, SBO);
                    uint64_t db = make_smem_desc_fn(B_sm + buf*TILE_ELEMS + ks*16, 1, SBO);
                    uint32_t accum = (kt==0 && ks==0) ? 0u : 1u;
                    umma_f16_cg2_fn(tmem_base, da, db, idesc, accum);
                }
                umma_commit_2sm_fn(&empty_bar[buf]);
            }
            umma_commit_2sm_fn(mma_done);
        }
    }

    mbarrier_wait_fn(mma_done, 0);
    __syncthreads();

    // Epilogue: read this CTA's TMEM (128 rows x 256 cols)
    #pragma unroll
    for (int col=0; col<256; col+=8){
        uint32_t r0,r1,r2,r3,r4,r5,r6,r7;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8];"
            :"=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7):"r"(tmem_base+col));
        asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory");
        int base = tid*256 + col;
        smem_out[base+0]=__float2bfloat16(__uint_as_float(r0));
        smem_out[base+1]=__float2bfloat16(__uint_as_float(r1));
        smem_out[base+2]=__float2bfloat16(__uint_as_float(r2));
        smem_out[base+3]=__float2bfloat16(__uint_as_float(r3));
        smem_out[base+4]=__float2bfloat16(__uint_as_float(r4));
        smem_out[base+5]=__float2bfloat16(__uint_as_float(r5));
        smem_out[base+6]=__float2bfloat16(__uint_as_float(r6));
        smem_out[base+7]=__float2bfloat16(__uint_as_float(r7));
    }
    __syncthreads();

    int lane = tid & 31;
    #pragma unroll
    for (int i=0;i<32;i++){
        int row = warp*32 + i;
        int gm = my_m_start + row;
        if (gm >= M) continue;
        int col = lane*8;
        float4 v = *reinterpret_cast<float4*>(&smem_out[row*256 + col]);
        *reinterpret_cast<float4*>(&C[(size_t)gm*N + out_n_start + col]) = v;
    }

    cluster_sync_fn();
    if (warp==0) tmem_dealloc_fn(tmem_base, 256);
}

static CUresult make_tma(CUtensorMap* d, void* addr, uint64_t inner, uint64_t outer,
                         uint32_t box_inner, uint32_t box_outer){
    uint64_t gdim[2] = {inner, outer};
    uint64_t gstride[1] = {inner*2};
    uint32_t bdim[2] = {box_inner, box_outer};
    uint32_t estr[2] = {1,1};
    return cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, addr, gdim, gstride,
        bdim, estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C){
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    int M = (int)A.size(0);
    int K = (int)A.size(1);
    int N = (int)B.size(0);

    __nv_bfloat16* Aptr = static_cast<__nv_bfloat16*>(A.data_ptr());
    __nv_bfloat16* Bptr = static_cast<__nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* Cptr = static_cast<__nv_bfloat16*>(C.data_ptr());

    CUtensorMap descA, descB;
    CU_CHECK(make_tma(&descA, Aptr, (uint64_t)K, (uint64_t)M, KW, ROWS));
    CU_CHECK(make_tma(&descB, Bptr, (uint64_t)K, (uint64_t)N, KW, ROWS));

    size_t smem_bytes = (size_t)2*STAGES*TILE_BYTES + (size_t)(2*STAGES+1)*8 + 16;

    CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));

    int num_m = (M + 255)/256;
    int num_n = N/256;
    int T = num_m * num_n;

    dim3 grid(2, T, 1);
    dim3 block(128);

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_bytes;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_kernel, descA, descB, Cptr, M, N, K));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_blackwell::run);

}  // namespace gemm_blackwell