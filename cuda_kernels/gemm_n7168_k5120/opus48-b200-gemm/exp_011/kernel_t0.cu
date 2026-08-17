#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
    } \
} while(0)

namespace gemm_kernel_ns {

using bf16 = __nv_bfloat16;

constexpr int NN = 7168;
constexpr int KK = 5120;
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 64;
constexpr int PAD = 8;
constexpr int SA = BK + PAD; // 72
constexpr int STAGES = 4;
constexpr int THREADS = 256;
constexpr int WARPS_N = 2;
constexpr int WARPS_M = 4;
constexpr int WM = BM / WARPS_M; // 32
constexpr int WN = BN / WARPS_N; // 64
constexpr int MT = WM / 16; // 2
constexpr int NT16 = WN / 16; // 4
constexpr int NT8 = WN / 8;  // 8
constexpr int K16 = BK / 16; // 4

__device__ __forceinline__ uint32_t cvta_shared(const void* p){
    return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void ldm_x4(uint32_t a, uint32_t&r0,uint32_t&r1,uint32_t&r2,uint32_t&r3){
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
      :"=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3):"r"(a));
}
__device__ __forceinline__ void mma16816(float&d0,float&d1,float&d2,float&d3,
    uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1){
    asm volatile(
     "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
     :"+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
     :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
__device__ __forceinline__ void cp_commit(){ asm volatile("cp.async.commit_group;\n"); }
template<int N> __device__ __forceinline__ void cp_wait(){ asm volatile("cp.async.wait_group %0;\n"::"n"(N)); }

__device__ __forceinline__ void load_tiles(const bf16* A, const bf16* B, bf16* As, bf16* Bs,
    int stage, int k0, int bm0, int bn0, int M, int tid){
   bf16* Ad = As + stage*BM*SA;
   bf16* Bd = Bs + stage*BN*SA;
   #pragma unroll
   for(int i=0;i<4;i++){
      int c = tid + i*256;
      int row = c>>3, ch = c&7, kk = ch*8;
      {
        int m = bm0+row;
        const bf16* g = A + (int64_t)m*KK + (k0+kk);
        uint32_t d = cvta_shared(Ad + row*SA + kk);
        int cpsize = (m<M)?16:0;
        asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;\n"::"r"(d),"l"(g),"r"(cpsize));
      }
      {
        int n = bn0+row;
        const bf16* g = B + (int64_t)n*KK + (k0+kk);
        uint32_t d = cvta_shared(Bd + row*SA + kk);
        asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"::"r"(d),"l"(g));
      }
   }
}

__global__ __launch_bounds__(THREADS) void gemm_kernel(const bf16* __restrict__ A,
    const bf16* __restrict__ B, bf16* __restrict__ C, int M){
    extern __shared__ __align__(16) char smem_raw[];
    bf16* As = reinterpret_cast<bf16*>(smem_raw);
    bf16* Bs = As + STAGES*BM*SA;

    int bn0 = blockIdx.x * BN;
    int bm0 = blockIdx.y * BM;
    int tid = threadIdx.x;
    int warp = tid>>5;
    int lane = tid&31;
    int warp_m = warp / WARPS_N;
    int warp_n = warp % WARPS_N;
    int wm0 = warp_m*WM;
    int wn0 = warp_n*WN;

    float acc[MT][NT8][4];
    #pragma unroll
    for(int i=0;i<MT;i++)
      #pragma unroll
      for(int j=0;j<NT8;j++)
        #pragma unroll
        for(int k=0;k<4;k++) acc[i][j][k]=0.f;

    const int num_k = KK/BK; // 80

    #pragma unroll
    for(int s=0;s<STAGES;s++){
        if(s<num_k) load_tiles(A,B,As,Bs,s,s*BK,bm0,bn0,M,tid);
        cp_commit();
    }

    for(int kt=0; kt<num_k; kt++){
        cp_wait<STAGES-1>();
        __syncthreads();
        int stage = kt % STAGES;
        bf16* As_s = As + stage*BM*SA;
        bf16* Bs_s = Bs + stage*BN*SA;
        #pragma unroll
        for(int ks=0; ks<K16; ks++){
            int kk = ks*16;
            uint32_t af[MT][4];
            #pragma unroll
            for(int mt=0; mt<MT; mt++){
                int m = wm0 + mt*16 + (lane&15);
                int kcol = kk + ((lane>>4)<<3);
                uint32_t addr = cvta_shared(As_s + m*SA + kcol);
                ldm_x4(addr, af[mt][0],af[mt][1],af[mt][2],af[mt][3]);
            }
            uint32_t bf[NT16][4];
            #pragma unroll
            for(int nt=0; nt<NT16; nt++){
                int n = wn0 + nt*16 + (lane&15);
                int kcol = kk + ((lane>>4)<<3);
                uint32_t addr = cvta_shared(Bs_s + n*SA + kcol);
                ldm_x4(addr, bf[nt][0],bf[nt][1],bf[nt][2],bf[nt][3]);
            }
            #pragma unroll
            for(int mt=0; mt<MT; mt++){
                #pragma unroll
                for(int n8=0;n8<NT8;n8++){
                    int n16 = n8>>1, half = n8&1;
                    uint32_t b0 = bf[n16][half];
                    uint32_t b1 = bf[n16][2+half];
                    mma16816(acc[mt][n8][0],acc[mt][n8][1],acc[mt][n8][2],acc[mt][n8][3],
                             af[mt][0],af[mt][1],af[mt][2],af[mt][3], b0, b1);
                }
            }
        }
        __syncthreads();
        int nxt = kt+STAGES;
        if(nxt<num_k){
            load_tiles(A,B,As,Bs,stage,nxt*BK,bm0,bn0,M,tid);
        }
        cp_commit();
    }

    int groupID = lane>>2;
    int tig = lane&3;
    #pragma unroll
    for(int mt=0; mt<MT; mt++){
        int mr0 = bm0 + wm0 + mt*16 + groupID;
        int mr1 = mr0 + 8;
        #pragma unroll
        for(int n8=0;n8<NT8;n8++){
            int ncol = bn0 + wn0 + n8*8 + tig*2;
            float c0=acc[mt][n8][0], c1=acc[mt][n8][1], c2=acc[mt][n8][2], c3=acc[mt][n8][3];
            if(mr0 < M){
                *reinterpret_cast<__nv_bfloat162*>(&C[(int64_t)mr0*NN + ncol]) = __floats2bfloat162_rn(c0,c1);
            }
            if(mr1 < M){
                *reinterpret_cast<__nv_bfloat162*>(&C[(int64_t)mr1*NN + ncol]) = __floats2bfloat162_rn(c2,c3);
            }
        }
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C){
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    int M = (int)A.size(0);
    const bf16* Ap = static_cast<const bf16*>(A.data_ptr());
    const bf16* Bp = static_cast<const bf16*>(B.data_ptr());
    bf16* Cp = static_cast<bf16*>(C.data_ptr());
    dim3 grid(NN/BN, (M+BM-1)/BM);
    dim3 block(THREADS);
    size_t smem = (size_t)STAGES*(BM+BN)*SA*sizeof(bf16);
    cudaFuncSetAttribute(gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    gemm_kernel<<<grid, block, smem, stream>>>(Ap,Bp,Cp,M);
    CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_kernel_ns::run);

} // namespace gemm_kernel_ns