#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__); exit(1);} }while(0)
#define CU_CHECK(call) do { CUresult _e=(call); if(_e!=CUDA_SUCCESS){ const char* s; cuGetErrorString(_e,&s); \
    fprintf(stderr,"CU error %s at %s:%d\n",s,__FILE__,__LINE__); exit(1);} }while(0)

namespace gemm_sm100 {

constexpr int BM=128, BN=256, BK=64;
constexpr int NSTAGES=4;
constexpr int A_ELEMS = BM*BK;      // 8192
constexpr int B_ELEMS = BN*BK;      // 16384
constexpr int A_BYTES = A_ELEMS*2;  // 16384
constexpr int B_BYTES = B_ELEMS*2;  // 32768
constexpr int TOT_BYTES = A_BYTES + B_BYTES;

__device__ __forceinline__ void init_barrier(uint64_t* b, int c){
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(b)),"r"(c));
}
__device__ __forceinline__ void fence_bar_init(){ asm volatile("fence.mbarrier_init.release.cluster;\n":::"memory"); }
__device__ __forceinline__ void arrive_expect_tx(uint64_t* b, uint32_t bytes){
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(b)),"r"(bytes):"memory");
}
__device__ __forceinline__ void bar_wait(uint64_t* b, uint32_t ph){
    asm volatile("{\n.reg .pred P;\nWT_%=:\nmbarrier.try_wait.parity.shared.b64 P,[%0],%1;\n@!P bra WT_%=;\n}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(b)),"r"(ph));
}
__device__ __forceinline__ void tma_load(const CUtensorMap* d, uint64_t* bar, void* smem, int c0, int c1){
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),"l"((uint64_t)d),
           "r"((uint32_t)__cvta_generic_to_shared(bar)),"r"(c0),"r"(c1):"memory");
}
__device__ __forceinline__ void tmem_alloc_cg1(uint32_t* dst, int nc){
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(dst)),"r"(nc));
}
__device__ __forceinline__ void tmem_dealloc_cg1(uint32_t a, int nc){
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(a),"r"(nc));
}
__device__ __forceinline__ void umma_cg1(uint32_t tc, uint64_t da, uint64_t db, uint32_t id, int acc){
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p,%4,0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tc),"l"(da),"l"(db),"r"(id),"r"(acc));
}
__device__ __forceinline__ void umma_commit_cg1(uint64_t* b){
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(b)));
}
__device__ __forceinline__ void tc_fence_before(){ asm volatile("tcgen05.fence::before_thread_sync;":::"memory"); }
__device__ __forceinline__ void tc_fence_after(){ asm volatile("tcgen05.fence::after_thread_sync;":::"memory"); }
__device__ __forceinline__ void tmem_wait_ld(){ asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory"); }

__device__ __forceinline__ uint64_t make_desc(void* ptr, uint32_t sbo){
    uint64_t d=0; uint32_t a=(uint32_t)__cvta_generic_to_shared(ptr);
    d |= (uint64_t)((a & 0x3FFFF) >> 4);
    d |= (uint64_t)0 << 16;                        // LBO (unused for K-major swizzle)
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;   // SBO
    d |= (uint64_t)1 << 46;                        // version SM100
    d |= (uint64_t)2 << 61;                        // 128B swizzle
    return d;
}
__device__ __forceinline__ uint32_t make_instr_desc(uint32_t M, uint32_t N){
    uint32_t d=0;
    d |= (1u<<4);   d |= (1u<<7);  d |= (1u<<10);
    d |= (0u<<15);  d |= (0u<<16);
    d |= ((N/8)<<17); d |= ((M/16)<<24);
    return d;
}
__device__ __forceinline__ void ld16(uint32_t addr, uint32_t* r){
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
      : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
        "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15])
      : "r"(addr));
}

__global__ __launch_bounds__(128,1)
void kernel(const __grid_constant__ CUtensorMap tma_A,
            const __grid_constant__ CUtensorMap tma_B,
            __nv_bfloat16* __restrict__ C, int M, int N, int K){
    extern __shared__ __align__(1024) char smem_raw[];
    __nv_bfloat16* As = (__nv_bfloat16*)smem_raw;
    __nv_bfloat16* Bs = (__nv_bfloat16*)(smem_raw + NSTAGES*A_BYTES);
    uint64_t* full  = (uint64_t*)(smem_raw + NSTAGES*A_BYTES + NSTAGES*B_BYTES);
    uint64_t* empty = full + NSTAGES;
    uint32_t* tmem_ptr = (uint32_t*)(empty + NSTAGES);
    __nv_bfloat16* smem_out = (__nv_bfloat16*)smem_raw;

    int tid = threadIdx.x;
    int warp = tid>>5;
    int numK = K / BK;
    int block_m = blockIdx.y * BM;
    int block_n = blockIdx.x * BN;

    if (tid==0){
        for(int s=0;s<NSTAGES;s++){ init_barrier(&full[s],1); init_barrier(&empty[s],1); }
        fence_bar_init();
    }
    __syncthreads();
    if (warp==0) tmem_alloc_cg1(tmem_ptr, BN);
    __syncthreads();
    uint32_t tmem_base = tmem_ptr[0];
    uint32_t idesc = make_instr_desc(BM, BN);

    if (tid==0){
        int pre = numK < NSTAGES ? numK : NSTAGES;
        for(int s=0;s<pre;s++){
            arrive_expect_tx(&full[s], TOT_BYTES);
            tma_load(&tma_A, &full[s], &As[s*A_ELEMS], s*BK, block_m);
            tma_load(&tma_B, &full[s], &Bs[s*B_ELEMS], s*BK, block_n);
        }
        for(int k=0;k<numK;k++){
            int s = k % NSTAGES;
            int r = k / NSTAGES;
            bar_wait(&full[s], r&1);
            #pragma unroll
            for(int j=0;j<4;j++){
                uint64_t da = make_desc(&As[s*A_ELEMS + j*16], 1024);
                uint64_t db = make_desc(&Bs[s*B_ELEMS + j*16], 1024);
                int acc = (k==0 && j==0)?0:1;
                umma_cg1(tmem_base, da, db, idesc, acc);
            }
            umma_commit_cg1(&empty[s]);
            int kp = k + NSTAGES;
            if (kp < numK){
                int sp = kp % NSTAGES;
                int rp = kp / NSTAGES;
                bar_wait(&empty[sp], (rp-1)&1);
                arrive_expect_tx(&full[sp], TOT_BYTES);
                tma_load(&tma_A, &full[sp], &As[sp*A_ELEMS], kp*BK, block_m);
                tma_load(&tma_B, &full[sp], &Bs[sp*B_ELEMS], kp*BK, block_n);
            }
        }
        for(int s=0;s<NSTAGES;s++){
            int uses = numK/NSTAGES + ((s < (numK%NSTAGES))?1:0);
            if (uses>0) bar_wait(&empty[s], (uses-1)&1);
        }
        tc_fence_before();
    }
    __syncthreads();
    tc_fence_after();

    // Epilogue Phase 1: TMEM -> smem_out (row = tid)
    #pragma unroll
    for(int col=0; col<BN; col+=16){
        uint32_t r[16];
        ld16(tmem_base + col, r);
        tmem_wait_ld();
        int base = tid*BN + col;
        #pragma unroll
        for(int i=0;i<16;i++) smem_out[base+i] = __float2bfloat16(__uint_as_float(r[i]));
    }
    __syncthreads();

    // Phase 2: coalesced smem_out -> C
    int total_vec = BM * (BN/8);
    for(int idx=tid; idx<total_vec; idx+=128){
        int row = idx / (BN/8);
        int col = (idx % (BN/8))*8;
        int gr = block_m + row;
        int gc = block_n + col;
        if (gr < M){
            int4 v = *(int4*)&smem_out[row*BN + col];
            *(int4*)&C[(size_t)gr*N + gc] = v;
        }
    }
    __syncthreads();
    if (warp==0) tmem_dealloc_cg1(tmem_base, BN);
}

static CUresult make_tma(CUtensorMap* d, void* addr, uint64_t inner, uint64_t outer,
                         uint32_t box_inner, uint32_t box_outer){
    uint64_t gdim[2] = {inner, outer};
    uint64_t gstr[1] = {inner*2};
    uint32_t bdim[2] = {box_inner, box_outer};
    uint32_t estr[2] = {1,1};
    return cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, addr, gdim, gstr,
        bdim, estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C){
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0);
    if (M==0) return;

    __nv_bfloat16* Ap = static_cast<__nv_bfloat16*>(A.data_ptr());
    __nv_bfloat16* Bp = static_cast<__nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* Cp = static_cast<__nv_bfloat16*>(C.data_ptr());

    CUtensorMap tA, tB;
    CU_CHECK(make_tma(&tA, Ap, (uint64_t)K, (uint64_t)M, 64, BM));
    CU_CHECK(make_tma(&tB, Bp, (uint64_t)K, (uint64_t)N, 64, BN));

    int smem_bytes = NSTAGES*A_BYTES + NSTAGES*B_BYTES + 2*NSTAGES*8 + 64;
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    dim3 grid((N + BN - 1)/BN, (M + BM - 1)/BM);
    dim3 block(128);
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    kernel<<<grid, block, smem_bytes, stream>>>(tA, tB, Cp, (int)M, (int)N, (int)K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_sm100::run);

}  // namespace gemm_sm100