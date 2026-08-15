#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); } } while(0)

namespace gemm_n7168_k5120 {

#define BM 128
#define BN 128
#define BK 32
#define BK_PAD 40
#define NSTAGE 3

__device__ __forceinline__ void load_block(
    const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* As, __nv_bfloat16* Bs, int s, int kb,
    int block_m, int block_n, int M, int K, int tid)
{
    int kk = kb*BK;
    #pragma unroll
    for(int it=0; it<2; ++it){
        int c = tid + it*256;       // 0..511
        int row = c >> 2;           // 0..127
        int col = (c & 3) * 8;      // 0,8,16,24
        int gm = block_m*BM + row;
        bool valid = gm < M;
        const __nv_bfloat16* src = valid ? (A + (size_t)gm*K + kk + col) : A;
        int ssz = valid ? 16 : 0;
        unsigned d = (unsigned)__cvta_generic_to_shared(&As[s*BM*BK_PAD + row*BK_PAD + col]);
        asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;\n" :: "r"(d), "l"(src), "r"(ssz));
    }
    #pragma unroll
    for(int it=0; it<2; ++it){
        int c = tid + it*256;
        int row = c >> 2;
        int col = (c & 3) * 8;
        int gn = block_n*BN + row;
        const __nv_bfloat16* src = B + (size_t)gn*K + kk + col;
        unsigned d = (unsigned)__cvta_generic_to_shared(&Bs[s*BN*BK_PAD + row*BK_PAD + col]);
        asm volatile("cp.async.cg.shared.global [%0],[%1],16,16;\n" :: "r"(d), "l"(src));
    }
}

__global__ void __launch_bounds__(256,2) gemm_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K)
{
    extern __shared__ __align__(16) char smem_raw[];
    __nv_bfloat16* As = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* Bs = As + NSTAGE*BM*BK_PAD;

    int block_m = blockIdx.y;
    int block_n = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;
    int warp_m = warp_id & 3;   // 0..3  -> rows of 32
    int warp_n = warp_id >> 2;  // 0..1  -> cols of 64

    int NK = K / BK;

    float acc[2][8][4];
    #pragma unroll
    for(int i=0;i<2;i++)
      #pragma unroll
      for(int j=0;j<8;j++)
        #pragma unroll
        for(int l=0;l<4;l++) acc[i][j][l]=0.f;

    // prologue
    #pragma unroll
    for(int s=0;s<NSTAGE;s++){
        load_block(A,B,As,Bs,s,s,block_m,block_n,M,K,tid);
        asm volatile("cp.async.commit_group;\n");
    }

    for(int it=0; it<NK; ++it){
        asm volatile("cp.async.wait_group %0;\n" :: "n"(NSTAGE-1));
        __syncthreads();
        int s = it % NSTAGE;

        #pragma unroll
        for(int k16=0;k16<2;k16++){
            int Coff = k16*16;
            uint32_t a[2][4];
            #pragma unroll
            for(int mi=0;mi<2;mi++){
                int R = warp_m*32 + mi*16;
                unsigned addr = (unsigned)__cvta_generic_to_shared(
                    &As[s*BM*BK_PAD + (R + (lane&15))*BK_PAD + Coff + ((lane>>4)*8)]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
                    : "=r"(a[mi][0]),"=r"(a[mi][1]),"=r"(a[mi][2]),"=r"(a[mi][3]) : "r"(addr));
            }
            uint32_t b[8][2];
            #pragma unroll
            for(int nj=0;nj<8;nj++){
                int Rn = warp_n*64 + nj*8;
                unsigned addr = (unsigned)__cvta_generic_to_shared(
                    &Bs[s*BN*BK_PAD + (Rn + (lane&7))*BK_PAD + Coff + ((lane>>3)*8)]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2];\n"
                    : "=r"(b[nj][0]),"=r"(b[nj][1]) : "r"(addr));
            }
            #pragma unroll
            for(int mi=0;mi<2;mi++)
              #pragma unroll
              for(int nj=0;nj<8;nj++){
                asm volatile(
                  "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                  "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                  : "+f"(acc[mi][nj][0]),"+f"(acc[mi][nj][1]),"+f"(acc[mi][nj][2]),"+f"(acc[mi][nj][3])
                  : "r"(a[mi][0]),"r"(a[mi][1]),"r"(a[mi][2]),"r"(a[mi][3]),
                    "r"(b[nj][0]),"r"(b[nj][1]));
              }
        }
        __syncthreads();
        int nb = it + NSTAGE;
        if(nb < NK){
            load_block(A,B,As,Bs,it%NSTAGE,nb,block_m,block_n,M,K,tid);
        }
        asm volatile("cp.async.commit_group;\n");
    }

    // store
    int groupID = lane >> 2;
    int tig = lane & 3;
    #pragma unroll
    for(int mi=0;mi<2;mi++){
        int rowA = block_m*BM + warp_m*32 + mi*16 + groupID;
        int rowB = rowA + 8;
        #pragma unroll
        for(int nj=0;nj<8;nj++){
            int ncol = block_n*BN + warp_n*64 + nj*8 + tig*2;
            if(rowA < M){
                __nv_bfloat162 v = __floats2bfloat162_rn(acc[mi][nj][0], acc[mi][nj][1]);
                *reinterpret_cast<__nv_bfloat162*>(&C[(size_t)rowA*N + ncol]) = v;
            }
            if(rowB < M){
                __nv_bfloat162 v = __floats2bfloat162_rn(acc[mi][nj][2], acc[mi][nj][3]);
                *reinterpret_cast<__nv_bfloat162*>(&C[(size_t)rowB*N + ncol]) = v;
            }
        }
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C){
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    int M = (int)A.size(0);
    int K = (int)A.size(1);
    int N = (int)B.size(0);
    const __nv_bfloat16* Ap = static_cast<const __nv_bfloat16*>(A.data_ptr());
    const __nv_bfloat16* Bp = static_cast<const __nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* Cp = static_cast<__nv_bfloat16*>(C.data_ptr());

    dim3 grid(N/BN, (M+BM-1)/BM);
    dim3 block(256);
    size_t smem = (size_t)NSTAGE*(BM+BN)*BK_PAD*sizeof(__nv_bfloat16);
    cudaFuncSetAttribute(gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    gemm_kernel<<<grid, block, smem, stream>>>(Ap, Bp, Cp, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_n7168_k5120::run);

}  // namespace gemm_n7168_k5120