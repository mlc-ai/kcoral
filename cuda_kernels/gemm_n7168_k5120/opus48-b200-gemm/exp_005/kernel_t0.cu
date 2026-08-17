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

constexpr int BM = 128;
constexpr int BN = 256;
constexpr int BK = 64;
constexpr int NUM_STAGES = 4;
constexpr int KDIM = 5120;
constexpr int NDIM = 7168;
constexpr int NUM_K_TILES = KDIM / BK;

__device__ __forceinline__ void init_bar(uint64_t* bar, int count){
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"::"r"((uint32_t)__cvta_generic_to_shared(bar)),"r"(count));
}
__device__ __forceinline__ void bar_arrive_expect(uint64_t* bar, uint32_t tx){
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"::"r"((uint32_t)__cvta_generic_to_shared(bar)),"r"(tx):"memory");
}
__device__ __forceinline__ void bar_wait(uint64_t* bar, uint32_t phase){
    asm volatile("{\n.reg .pred P;\nLAB_%=:\nmbarrier.try_wait.parity.shared.b64 P,[%0],%1;\n@!P bra LAB_%=;\n}\n"::"r"((uint32_t)__cvta_generic_to_shared(bar)),"r"(phase));
}
__device__ __forceinline__ void tma_load_2d(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1){
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1,{%3,%4}],[%2];"
        ::"r"((uint32_t)__cvta_generic_to_shared(smem)),"l"((uint64_t)d),"r"((uint32_t)__cvta_generic_to_shared(bar)),"r"(c0),"r"(c1):"memory");
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
__device__ __forceinline__ void umma(uint32_t tmem, uint64_t da, uint64_t db, uint32_t idesc, uint32_t accum){
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p,%4,0;\ntcgen05.mma.cta_group::1.kind::f16 [%0],%1,%2,%3,p;\n}\n"
        ::"r"(tmem),"l"(da),"l"(db),"r"(idesc),"r"(accum));
}
__device__ __forceinline__ void umma_commit(uint64_t* bar){
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"::"r"((uint32_t)__cvta_generic_to_shared(bar)):"memory");
}
__device__ __forceinline__ void tmem_alloc(uint32_t* dst,int ncols){
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0],%1;"::"r"((uint32_t)__cvta_generic_to_shared(dst)),"r"(ncols):"memory");
}
__device__ __forceinline__ void tmem_dealloc(uint32_t addr,int ncols){
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,%1;"::"r"(addr),"r"(ncols):"memory");
}

__global__ __launch_bounds__(128) void gemm_kernel(const __grid_constant__ CUtensorMap tma_A,
                            const __grid_constant__ CUtensorMap tma_B,
                            __nv_bfloat16* __restrict__ C, int M){
    extern __shared__ char smem_raw[];
    uintptr_t b0 = ((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023;
    __nv_bfloat16* A_smem = (__nv_bfloat16*)b0;
    __nv_bfloat16* B_smem = A_smem + NUM_STAGES*BM*BK;
    uint64_t* full_bar = (uint64_t*)(B_smem + NUM_STAGES*BN*BK);
    uint64_t* empty_bar = full_bar + NUM_STAGES;
    uint64_t* final_bar = empty_bar + NUM_STAGES;
    uint32_t* tmem_ptr = (uint32_t*)(final_bar + 1);
    __nv_bfloat16* smem_out = A_smem; // reuse during epilogue

    int tid = threadIdx.x;
    int warp = tid>>5;
    int n_tiles_N = NDIM / BN;
    int m_block = blockIdx.x / n_tiles_N;
    int n_block = blockIdx.x % n_tiles_N;

    if (tid==0){
        for(int s=0;s<NUM_STAGES;s++){ init_bar(&full_bar[s],1); init_bar(&empty_bar[s],1);}
        init_bar(final_bar,1);
    }
    if (warp==0){
        tmem_alloc(tmem_ptr, BN);
    }
    __syncthreads();
    uint32_t tmem_base = *tmem_ptr;
    uint32_t idesc = make_idesc(BM, BN);
    const uint32_t AB_bytes = (BM*BK + BN*BK)*2;

    bool is_producer = (tid==0);
    bool is_consumer = (tid==32);

    if (is_producer){
        for(int kt=0; kt<NUM_K_TILES; kt++){
            int stage = kt % NUM_STAGES;
            int r = kt / NUM_STAGES;
            if (kt >= NUM_STAGES){
                bar_wait(&empty_bar[stage], (uint32_t)((r-1)&1));
            }
            bar_arrive_expect(&full_bar[stage], AB_bytes);
            int kc = kt*BK;
            tma_load_2d(&tma_A, &full_bar[stage], A_smem + stage*BM*BK, kc, m_block*BM);
            tma_load_2d(&tma_B, &full_bar[stage], B_smem + stage*BN*BK, kc, n_block*BN);
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
                umma(tmem_base, da, db, idesc, accum);
            }
            if (kt == NUM_K_TILES-1) umma_commit(final_bar);
            else umma_commit(&empty_bar[stage]);
        }
    }

    bar_wait(final_bar, 0);
    __syncthreads();

    // Epilogue Phase 1: TMEM -> SMEM
    for(int col=0; col<BN; col+=4){
        uint32_t r0,r1,r2,r3;
        uint32_t addr = tmem_base + col;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3},[%4];":"=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3):"r"(addr));
        asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory");
        uint32_t base = tid*BN + col;
        smem_out[base+0]=__float2bfloat16(__uint_as_float(r0));
        smem_out[base+1]=__float2bfloat16(__uint_as_float(r1));
        smem_out[base+2]=__float2bfloat16(__uint_as_float(r2));
        smem_out[base+3]=__float2bfloat16(__uint_as_float(r3));
    }
    __syncthreads();
    // Phase 2: SMEM -> global (coalesced uint4 = 8 bf16 per lane)
    int warp_id = tid>>5;
    int lane_id = tid&31;
    #pragma unroll
    for(int step=0; step<BM/4; step++){
        int row = step*4 + warp_id;
        int gr = m_block*BM + row;
        if (gr < M){
            int col_start = lane_id*8;
            int gc = n_block*BN + col_start;
            uint4 data = *reinterpret_cast<uint4*>(&smem_out[row*BN + col_start]);
            *reinterpret_cast<uint4*>(C + (uint64_t)gr*NDIM + gc) = data;
        }
    }
    __syncthreads();
    if (warp==0) tmem_dealloc(tmem_base, BN);
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

    int n_tiles_N = NDIM / BN;
    int m_tiles = (M + BM - 1)/BM;
    dim3 grid(m_tiles*n_tiles_N);
    dim3 block(128);
    cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(A.device().device_type, A.device().device_id);
    gemm_kernel<<<grid, block, smem_bytes, stream>>>(tA, tB, c, M);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_sm100::run);

}  // namespace gemm_sm100