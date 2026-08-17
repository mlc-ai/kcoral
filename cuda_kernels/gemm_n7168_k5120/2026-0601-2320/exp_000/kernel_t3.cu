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

#define BM 128
#define BN 256
#define BK 16
#define NSTAGE 6
#define A_ELEMS (BM*BK)   // 2048
#define B_ELEMS (BN*BK)   // 4096
#define A_BYTES (A_ELEMS*2)
#define B_BYTES (B_ELEMS*2)
#define SBO 256u

// ---------- provided-style device helpers ----------
__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count){
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}
__device__ __forceinline__ void fence_smem_barrier_init_fn(){
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}
__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx){
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx) : "memory");
}
__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase){
    asm volatile(
        "{\n.reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase));
}
__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1){
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

// ---------- cta_group::1 tcgen05 helpers ----------
__device__ __forceinline__ void tmem_alloc_cg1(uint32_t* dst_smem, int ncols){
    uint32_t a=(uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a), "r"(ncols));
}
__device__ __forceinline__ void tmem_relinquish_cg1(){
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}
__device__ __forceinline__ void tmem_dealloc_cg1(uint32_t addr, int ncols){
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(addr), "r"(ncols));
}
__device__ __forceinline__ void umma_cg1(uint32_t tmem_c, uint64_t da, uint64_t db, uint32_t idesc, uint32_t accum){
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(da), "l"(db), "r"(idesc), "r"(accum));
}
__device__ __forceinline__ void umma_commit_cg1(uint64_t* bar){
    uint32_t a=(uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(a) : "memory");
}
__device__ __forceinline__ void tcgen05_fence_before_fn(){ asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tcgen05_fence_after_fn(){ asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }

__device__ __forceinline__ uint64_t make_desc(void* p, uint32_t sbo){
    uint64_t d=0;
    uint32_t a=(uint32_t)__cvta_generic_to_shared(p);
    d |= (uint64_t)((a & 0x3FFFFu) >> 4);
    d |= ((uint64_t)1) << 16;                       // LBO = 1 (unused for K-major swizzle)
    d |= ((uint64_t)((sbo & 0x3FFFFu) >> 4)) << 32; // SBO
    d |= ((uint64_t)1) << 46;                        // SM100 version
    d |= ((uint64_t)6) << 61;                        // 32B swizzle
    return d;
}

__global__ void __launch_bounds__(128,2) gemm_kernel(
    const __grid_constant__ CUtensorMap descA,
    const __grid_constant__ CUtensorMap descB,
    __nv_bfloat16* __restrict__ C, int M, int N, int K)
{
    extern __shared__ char smem_raw[];
    uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem_raw);
    uint32_t pad = (1024u - (sbase & 1023u)) & 1023u;
    char* sm = smem_raw + pad;

    __nv_bfloat16* As = reinterpret_cast<__nv_bfloat16*>(sm);
    __nv_bfloat16* Bs = As + NSTAGE*A_ELEMS;
    uint64_t* full = reinterpret_cast<uint64_t*>(sm + (size_t)NSTAGE*(A_ELEMS+B_ELEMS)*2);
    uint64_t* empty = full + NSTAGE;
    uint64_t* mma_done = empty + NSTAGE;
    uint32_t* tmem_ptr_smem = reinterpret_cast<uint32_t*>(mma_done + 1);
    __nv_bfloat16* smem_out = As;

    int block_n = blockIdx.x;
    int block_m = blockIdx.y;
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int KT = K / BK;

    if (tid == 0){
        #pragma unroll
        for (int s=0;s<NSTAGE;s++){ init_smem_barrier_fn(&full[s],1); init_smem_barrier_fn(&empty[s],1); }
        init_smem_barrier_fn(&mma_done[0],1);
        fence_smem_barrier_init_fn();
    }
    if (warp_id == 0){
        __syncwarp();
        tmem_alloc_cg1(tmem_ptr_smem, BN);
        tmem_relinquish_cg1();
    }
    __syncthreads();

    uint32_t tmem_base = tmem_ptr_smem[0];

    if (tid == 32){
        // producer
        for (int k=0;k<KT;++k){
            int s = k % NSTAGE;
            if (k >= NSTAGE){
                int ph = ((k/NSTAGE) - 1) & 1;
                mbarrier_wait_fn(&empty[s], ph);
            }
            mbarrier_arrive_and_expect_tx_fn(&full[s], A_BYTES + B_BYTES);
            tma_load_2d_fn(&descA, &full[s], As + s*A_ELEMS, k*BK, block_m*BM);
            tma_load_2d_fn(&descB, &full[s], Bs + s*B_ELEMS, k*BK, block_n*BN);
        }
    } else if (tid == 0){
        // consumer (MMA issuer)
        uint32_t idesc = (1u<<4)|(1u<<7)|(1u<<10)|((BN/8)<<17)|((BM/16)<<24);
        for (int k=0;k<KT;++k){
            int s = k % NSTAGE;
            int ph = (k/NSTAGE) & 1;
            mbarrier_wait_fn(&full[s], ph);
            uint64_t da = make_desc(As + s*A_ELEMS, SBO);
            uint64_t db = make_desc(Bs + s*B_ELEMS, SBO);
            umma_cg1(tmem_base, da, db, idesc, (k==0)?0u:1u);
            umma_commit_cg1(&empty[s]);
        }
        umma_commit_cg1(&mma_done[0]);
        mbarrier_wait_fn(&mma_done[0], 0);
        tcgen05_fence_before_fn();
    }

    __syncthreads();
    tcgen05_fence_after_fn();

    // Epilogue: read TMEM -> smem (bf16) -> coalesced global store
    int row = tid; // 0..127
    #pragma unroll
    for (int col=0; col<BN; col+=4){
        uint32_t r0,r1,r2,r3;
        uint32_t taddr = tmem_base + (uint32_t)col;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(taddr));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        int base = row*BN + col;
        smem_out[base+0]=__float2bfloat16(__uint_as_float(r0));
        smem_out[base+1]=__float2bfloat16(__uint_as_float(r1));
        smem_out[base+2]=__float2bfloat16(__uint_as_float(r2));
        smem_out[base+3]=__float2bfloat16(__uint_as_float(r3));
    }
    __syncthreads();

    for (int v=tid; v < BM*(BN/4); v += 128){
        int r = v / (BN/4);
        int cv = v % (BN/4);
        int c = cv*4;
        int gr = block_m*BM + r;
        int gc = block_n*BN + c;
        if (gr < M){
            uint2 d = *reinterpret_cast<uint2*>(&smem_out[r*BN + c]);
            *reinterpret_cast<uint2*>(&C[(size_t)gr*N + gc]) = d;
        }
    }
    __syncthreads();
    if (warp_id == 0){
        tmem_dealloc_cg1(tmem_base, BN);
    }
}

static CUresult make_tma(CUtensorMap* d, void* gptr, uint64_t inner, uint64_t outer,
                         uint32_t binner, uint32_t bouter){
    uint64_t gd[2] = {inner, outer};
    uint64_t gs[1] = {inner*2};
    uint32_t bd[2] = {binner, bouter};
    uint32_t es[2] = {1,1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, gptr, gd, gs, bd, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_32B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C){
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    int M = (int)A.size(0);
    int K = (int)A.size(1);
    int N = (int)B.size(0);

    __nv_bfloat16* Ap = static_cast<__nv_bfloat16*>(A.data_ptr());
    __nv_bfloat16* Bp = static_cast<__nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* Cp = static_cast<__nv_bfloat16*>(C.data_ptr());

    CUtensorMap dA, dB;
    CU_CHECK(make_tma(&dA, Ap, (uint64_t)K, (uint64_t)M, BK, BM));
    CU_CHECK(make_tma(&dB, Bp, (uint64_t)K, (uint64_t)N, BK, BN));

    dim3 grid(N/BN, (M+BM-1)/BM);
    dim3 block(128);

    size_t smem = 1024 + (size_t)NSTAGE*(A_ELEMS+B_ELEMS)*2 + 1024;
    CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    gemm_kernel<<<grid, block, smem, stream>>>(dA, dB, Cp, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_n7168_k5120::run);

}  // namespace gemm_n7168_k5120