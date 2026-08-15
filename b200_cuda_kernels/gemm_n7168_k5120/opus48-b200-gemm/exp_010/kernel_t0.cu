#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); exit(1);} } while(0)

namespace gemm_n7168_k5120 {

using bf16 = __nv_bfloat16;

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 64;
constexpr int PAD = 8;
constexpr int AS = BK + PAD;      // 72 row stride (avoid bank conflicts)
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 4;
constexpr int THREADS = 256;
constexpr int WM = BM / WARPS_M;  // 64
constexpr int WN = BN / WARPS_N;  // 32
constexpr int MT = WM / 16;       // 4
constexpr int NT = WN / 8;        // 4

__device__ __forceinline__ uint32_t saddr(const void* p){
  return (uint32_t)__cvta_generic_to_shared(p);
}

// Load a BM(or BN) x BK tile into shared memory with cp.async (vectorized 16B).
__device__ __forceinline__ void load_tile(const bf16* __restrict__ G, int grow_base,
                                          int gcol_base, int G_rows, int Kstride,
                                          bf16* dst, int tid){
  #pragma unroll
  for (int i=0;i<4;i++){
    int vec = tid + i*THREADS;    // 0..1023
    int r = vec >> 3;             // 0..127
    int c = (vec & 7) * 8;        // 0,8,...,56
    int grow = grow_base + r;
    bool ok = grow < G_rows;
    int gr = ok ? grow : 0;
    const bf16* src = G + (size_t)gr * Kstride + gcol_base + c;
    uint32_t d = saddr(dst + r*AS + c);
    int ssz = ok ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0],[%1],%2,%3;\n"
      :: "r"(d), "l"(src), "n"(16), "r"(ssz) : "memory");
  }
}

__global__ __launch_bounds__(THREADS) void gemm_kernel(
    const bf16* __restrict__ A, const bf16* __restrict__ B, bf16* __restrict__ C,
    int M, int N, int K){
  extern __shared__ char smem_raw[];
  bf16* sA = reinterpret_cast<bf16*>(smem_raw);
  bf16* sB = sA + 2*BM*AS;

  int tid = threadIdx.x;
  int warp = tid >> 5;
  int lane = tid & 31;
  int warp_m = warp / WARPS_N;
  int warp_n = warp % WARPS_N;
  int wm_base = warp_m * WM;
  int wn_base = warp_n * WN;

  int m_block = blockIdx.y * BM;
  int n_block = blockIdx.x * BN;

  int NKB = K / BK; // 80

  float acc[MT][NT][4];
  #pragma unroll
  for (int i=0;i<MT;i++)
    #pragma unroll
    for (int j=0;j<NT;j++)
      #pragma unroll
      for (int r=0;r<4;r++) acc[i][j][r]=0.f;

  // stage 0
  load_tile(A, m_block, 0, M, K, sA, tid);
  load_tile(B, n_block, 0, N, K, sB, tid);
  asm volatile("cp.async.commit_group;\n" ::: "memory");

  for (int kb=0; kb<NKB; kb++){
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();
    int bi = kb & 1;
    bf16* curA = sA + bi*BM*AS;
    bf16* curB = sB + bi*BM*AS;

    if (kb+1 < NKB){
      int nb = (kb+1)&1;
      load_tile(A, m_block, (kb+1)*BK, M, K, sA + nb*BM*AS, tid);
      load_tile(B, n_block, (kb+1)*BK, N, K, sB + nb*BM*AS, tid);
      asm volatile("cp.async.commit_group;\n" ::: "memory");
    }

    #pragma unroll
    for (int ks=0; ks<BK/16; ks++){
      int k0 = ks*16;
      uint32_t RA[MT][4];
      #pragma unroll
      for (int mt=0; mt<MT; mt++){
        int m0 = wm_base + mt*16;
        int row = lane & 7;
        int mext = ((lane>>3)&1)*8;
        int kext = ((lane>>4)&1)*8;
        uint32_t a = saddr(curA + (m0 + row + mext)*AS + (k0 + kext));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];\n"
          : "=r"(RA[mt][0]),"=r"(RA[mt][1]),"=r"(RA[mt][2]),"=r"(RA[mt][3]) : "r"(a));
      }
      uint32_t RB[NT][2];
      #pragma unroll
      for (int nt=0; nt<NT; nt++){
        int n0 = wn_base + nt*8;
        int row = lane & 7;
        int kext = ((lane>>3)&1)*8;
        uint32_t b = saddr(curB + (n0 + row)*AS + (k0 + kext));
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0,%1},[%2];\n"
          : "=r"(RB[nt][0]),"=r"(RB[nt][1]) : "r"(b));
      }
      #pragma unroll
      for (int mt=0; mt<MT; mt++)
        #pragma unroll
        for (int nt=0; nt<NT; nt++){
          asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
            : "+f"(acc[mt][nt][0]),"+f"(acc[mt][nt][1]),"+f"(acc[mt][nt][2]),"+f"(acc[mt][nt][3])
            : "r"(RA[mt][0]),"r"(RA[mt][1]),"r"(RA[mt][2]),"r"(RA[mt][3]),
              "r"(RB[nt][0]),"r"(RB[nt][1]));
        }
    }
  }

  // epilogue
  int groupID = lane >> 2;
  int tidg = lane & 3;
  #pragma unroll
  for (int mt=0; mt<MT; mt++){
    #pragma unroll
    for (int nt=0; nt<NT; nt++){
      int m_base = m_block + wm_base + mt*16;
      int n_base = n_block + wn_base + nt*8;
      int mm0 = m_base + groupID;
      int mm1 = m_base + groupID + 8;
      int nn = n_base + tidg*2;
      if (mm0 < M){
        __nv_bfloat162 p; p.x=__float2bfloat16(acc[mt][nt][0]); p.y=__float2bfloat16(acc[mt][nt][1]);
        *reinterpret_cast<__nv_bfloat162*>(&C[(size_t)mm0*N + nn]) = p;
      }
      if (mm1 < M){
        __nv_bfloat162 p; p.x=__float2bfloat16(acc[mt][nt][2]); p.y=__float2bfloat16(acc[mt][nt][3]);
        *reinterpret_cast<__nv_bfloat162*>(&C[(size_t)mm1*N + nn]) = p;
      }
    }
  }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C){
  CUDA_CHECK(cudaSetDevice(A.device().device_id));
  int64_t M = A.size(0);
  int64_t K = A.size(1);
  int64_t N = B.size(0);
  const bf16* Ap = static_cast<const bf16*>(A.data_ptr());
  const bf16* Bp = static_cast<const bf16*>(B.data_ptr());
  bf16* Cp = static_cast<bf16*>(C.data_ptr());
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

  int smem_bytes = (2*BM*AS + 2*BN*AS)*(int)sizeof(bf16);
  CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

  dim3 block(THREADS);
  dim3 grid((int)((N+BN-1)/BN), (int)((M+BM-1)/BM));
  gemm_kernel<<<grid, block, smem_bytes, stream>>>(Ap, Bp, Cp, (int)M, (int)N, (int)K);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_n7168_k5120::run);

}  // namespace gemm_n7168_k5120