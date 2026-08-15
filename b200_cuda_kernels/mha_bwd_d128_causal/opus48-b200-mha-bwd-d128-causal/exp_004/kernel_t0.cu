#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cmath>
#include <type_traits>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_bwd_cuda {

using namespace nvcuda;
using row_major = wmma::row_major;
using col_major = wmma::col_major;

constexpr int BN = 64;   // key tile
constexpr int BM = 64;   // query tile
constexpr int DIM = 128; // head dim
constexpr int THREADS = 256;
constexpr int NW = THREADS / 32; // 8 warps

// SMEM layout offsets (bytes)
constexpr int OFF_K    = 0;        // bf16 [BN][DIM] 16384
constexpr int OFF_V    = 16384;    // bf16 16384
constexpr int OFF_Q    = 32768;    // bf16 16384
constexpr int OFF_dO   = 49152;    // bf16 16384
constexpr int OFF_SCR  = 65536;    // f32  [BN][BM] 16384
constexpr int OFF_dPf  = 81920;    // f32  16384
constexpr int OFF_Pbf  = 98304;    // bf16 [BN][BM] 8192
constexpr int OFF_dSbf = 106496;   // bf16 8192
constexpr int OFF_dVa  = 114688;   // f32  [BN][DIM] 32768
constexpr int OFF_dKa  = 147456;   // f32  32768
constexpr int OFF_dQs  = 180224;   // f32  [BM][DIM] 32768
constexpr int OFF_L    = 212992;   // f32  [BM]
constexpr int OFF_D    = 213248;   // f32  [BM]
constexpr int SMEM_BYTES = 213504;

template<typename LayA>
__device__ __forceinline__ const __nv_bfloat16* aptr(const __nv_bfloat16* A, int lda, int mi, int k){
  if constexpr (std::is_same<LayA, row_major>::value)
    return A + (mi*16)*lda + (k*16);
  else
    return A + (mi*16) + (k*16)*lda;
}
template<typename LayB>
__device__ __forceinline__ const __nv_bfloat16* bptr(const __nv_bfloat16* B, int ldb, int k, int ni){
  if constexpr (std::is_same<LayB, row_major>::value)
    return B + (k*16)*ldb + (ni*16);
  else
    return B + (k*16) + (ni*16)*ldb;
}

template<int M,int N,int K,typename LayA,typename LayB, bool ACC>
__device__ __forceinline__ void gemm(const __nv_bfloat16* A,int lda,const __nv_bfloat16* B,int ldb,
                                      float* C,int ldc,int warp_id,int num_warps){
  constexpr int MT=M/16, NT=N/16, KT=K/16;
  constexpr int TOT=MT*NT;
  for(int t=warp_id;t<TOT;t+=num_warps){
    int mi=t/NT, ni=t%NT;
    wmma::fragment<wmma::accumulator,16,16,16,float> c;
    wmma::fill_fragment(c,0.f);
    #pragma unroll
    for(int k=0;k<KT;k++){
      wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,LayA> af;
      wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,LayB> bf;
      wmma::load_matrix_sync(af, aptr<LayA>(A,lda,mi,k), lda);
      wmma::load_matrix_sync(bf, bptr<LayB>(B,ldb,k,ni), ldb);
      wmma::mma_sync(c,af,bf,c);
    }
    float* cp = C + (mi*16)*ldc + (ni*16);
    if constexpr (ACC){
      wmma::fragment<wmma::accumulator,16,16,16,float> cold;
      wmma::load_matrix_sync(cold, cp, ldc, wmma::mem_row_major);
      #pragma unroll
      for(int i=0;i<c.num_elements;i++) c.x[i]+=cold.x[i];
    }
    wmma::store_matrix_sync(cp, c, ldc, wmma::mem_row_major);
  }
}

__global__ void compute_D_kernel(const __nv_bfloat16* dO, const __nv_bfloat16* O,
                                 float* Dg, long total_rows){
  int row = blockIdx.x*(blockDim.x/32) + (threadIdx.x/32);
  int lane = threadIdx.x%32;
  if(row >= total_rows) return;
  const __nv_bfloat16* dOr = dO + (size_t)row*DIM;
  const __nv_bfloat16* Or  = O  + (size_t)row*DIM;
  float sum=0.f;
  for(int e=lane;e<DIM;e+=32){
    sum += __bfloat162float(dOr[e])*__bfloat162float(Or[e]);
  }
  #pragma unroll
  for(int o=16;o>0;o>>=1) sum += __shfl_down_sync(0xffffffff,sum,o);
  if(lane==0) Dg[row]=sum;
}

__global__ __launch_bounds__(THREADS) void mha_bwd_kernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* dO, const float* L, const float* Dg,
    float* dQscratch, __nv_bfloat16* dKout, __nv_bfloat16* dVout,
    int S, float scale){

  extern __shared__ char smem[];
  __nv_bfloat16* Ksh = reinterpret_cast<__nv_bfloat16*>(smem + OFF_K);
  __nv_bfloat16* Vsh = reinterpret_cast<__nv_bfloat16*>(smem + OFF_V);
  __nv_bfloat16* Qsh = reinterpret_cast<__nv_bfloat16*>(smem + OFF_Q);
  __nv_bfloat16* dOsh= reinterpret_cast<__nv_bfloat16*>(smem + OFF_dO);
  float* Scr = reinterpret_cast<float*>(smem + OFF_SCR);
  float* dPf = reinterpret_cast<float*>(smem + OFF_dPf);
  __nv_bfloat16* Pbf = reinterpret_cast<__nv_bfloat16*>(smem + OFF_Pbf);
  __nv_bfloat16* dSbf= reinterpret_cast<__nv_bfloat16*>(smem + OFF_dSbf);
  float* dVacc = reinterpret_cast<float*>(smem + OFF_dVa);
  float* dKacc = reinterpret_cast<float*>(smem + OFF_dKa);
  float* dQsh  = reinterpret_cast<float*>(smem + OFF_dQs);
  float* Lsh = reinterpret_cast<float*>(smem + OFF_L);
  float* Dsh = reinterpret_cast<float*>(smem + OFF_D);

  int tid = threadIdx.x;
  int nthreads = blockDim.x;
  int warp_id = tid/32;

  int bh = blockIdx.y;
  int kb = blockIdx.x;
  int k0 = kb*BN;
  if(k0>=S) return;

  const __nv_bfloat16* Kbase = K + (size_t)bh*S*DIM;
  const __nv_bfloat16* Vbase = V + (size_t)bh*S*DIM;
  const __nv_bfloat16* Qbase = Q + (size_t)bh*S*DIM;
  const __nv_bfloat16* dObase= dO+ (size_t)bh*S*DIM;
  const float* Lbase = L + (size_t)bh*S;
  const float* Dgbase= Dg+ (size_t)bh*S;
  __nv_bfloat16* dKoutbase = dKout + (size_t)bh*S*DIM;
  __nv_bfloat16* dVoutbase = dVout + (size_t)bh*S*DIM;

  const __nv_bfloat16 zero = __float2bfloat16(0.0f);

  // load K,V tile
  for(int idx=tid; idx<BN*DIM; idx+=nthreads){
    int r=idx/DIM, e=idx%DIM;
    int gr=k0+r;
    Ksh[idx] = (gr<S)?Kbase[(size_t)gr*DIM+e]:zero;
    Vsh[idx] = (gr<S)?Vbase[(size_t)gr*DIM+e]:zero;
  }
  for(int idx=tid; idx<BN*DIM; idx+=nthreads){ dVacc[idx]=0.f; dKacc[idx]=0.f; }
  __syncthreads();

  int numQ = (S+BM-1)/BM;
  for(int qb=kb; qb<numQ; qb++){
    int q0=qb*BM;
    // load Q,dO,L,D
    for(int idx=tid; idx<BM*DIM; idx+=nthreads){
      int r=idx/DIM, e=idx%DIM;
      int gr=q0+r;
      Qsh[idx]=(gr<S)?Qbase[(size_t)gr*DIM+e]:zero;
      dOsh[idx]=(gr<S)?dObase[(size_t)gr*DIM+e]:zero;
    }
    for(int i=tid;i<BM;i+=nthreads){
      int gi=q0+i;
      Lsh[i]=(gi<S)?Lbase[gi]:0.f;
      Dsh[i]=(gi<S)?Dgbase[gi]:0.f;
    }
    __syncthreads();

    // scores^T = K * Q^T  -> Scr[j][i]
    gemm<BN,BM,DIM, row_major, col_major, false>(Ksh, DIM, Qsh, DIM, Scr, BM, warp_id, NW);
    __syncthreads();

    // P^T = exp(scale*S^T - L_i)  (masked -> 0)
    for(int idx=tid; idx<BN*BM; idx+=nthreads){
      int j=idx/BM, i=idx%BM;
      int gj=k0+j, gi=q0+i;
      float s=Scr[idx];
      float p=0.f;
      if(gj<S && gi<S && gi>=gj) p=__expf(scale*s - Lsh[i]);
      Scr[idx]=p;
      Pbf[idx]=__float2bfloat16(p);
    }
    __syncthreads();

    // dP^T = V * dO^T
    gemm<BN,BM,DIM, row_major, col_major, false>(Vsh, DIM, dOsh, DIM, dPf, BM, warp_id, NW);
    __syncthreads();

    // dS^T = scale * P^T * (dP^T - D_i)
    for(int idx=tid; idx<BN*BM; idx+=nthreads){
      int i=idx%BM;
      float p=Scr[idx];
      float dp=dPf[idx];
      float ds=scale*p*(dp - Dsh[i]);
      dSbf[idx]=__float2bfloat16(ds);
    }
    __syncthreads();

    // dV += P^T * dO
    gemm<BN,DIM,BM, row_major, row_major, true>(Pbf, BM, dOsh, DIM, dVacc, DIM, warp_id, NW);
    // dK += dS^T * Q
    gemm<BN,DIM,BM, row_major, row_major, true>(dSbf, BM, Qsh, DIM, dKacc, DIM, warp_id, NW);
    // dQ = (dS^T)^T * K
    gemm<BM,DIM,BN, col_major, row_major, false>(dSbf, BM, Ksh, DIM, dQsh, DIM, warp_id, NW);
    __syncthreads();

    // atomic accumulate dQ
    for(int idx=tid; idx<BM*DIM; idx+=nthreads){
      int r=idx/DIM, e=idx%DIM;
      int gi=q0+r;
      if(gi<S) atomicAdd(&dQscratch[((size_t)bh*S+gi)*DIM+e], dQsh[idx]);
    }
    __syncthreads();
  }

  // write dK, dV
  for(int idx=tid; idx<BN*DIM; idx+=nthreads){
    int r=idx/DIM, e=idx%DIM;
    int gr=k0+r;
    if(gr<S){
      dVoutbase[(size_t)gr*DIM+e] = __float2bfloat16(dVacc[idx]);
      dKoutbase[(size_t)gr*DIM+e] = __float2bfloat16(dKacc[idx]);
    }
  }
}

__global__ void convert_dQ_kernel(const float* scratch, __nv_bfloat16* dQ, size_t total){
  size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
  if(i<total) dQ[i]=__float2bfloat16(scratch[i]);
}

// -------- persistent scratch caches --------
static float* s_D = nullptr;  static size_t s_Dsz = 0;
static float* s_dQ = nullptr; static size_t s_dQsz = 0;
static bool   s_attr_set = false;

static void ensure_buf(float** p, size_t* cur, size_t need){
  if(*cur < need){
    if(*p) cudaFree(*p);
    CUDA_CHECK(cudaMalloc(p, need));
    *cur = need;
  }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int64_t B = Q.size(0), H = Q.size(1), S = Q.size(2), d = Q.size(3);
  int64_t BH = B*H;
  (void)d;

  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

  const __nv_bfloat16* Qp = static_cast<const __nv_bfloat16*>(Q.data_ptr());
  const __nv_bfloat16* Kp = static_cast<const __nv_bfloat16*>(K.data_ptr());
  const __nv_bfloat16* Vp = static_cast<const __nv_bfloat16*>(V.data_ptr());
  const __nv_bfloat16* Op = static_cast<const __nv_bfloat16*>(O.data_ptr());
  const __nv_bfloat16* dOp= static_cast<const __nv_bfloat16*>(dO.data_ptr());
  const float* Lp = static_cast<const float*>(L.data_ptr());
  __nv_bfloat16* dQp = static_cast<__nv_bfloat16*>(dQ.data_ptr());
  __nv_bfloat16* dKp = static_cast<__nv_bfloat16*>(dK.data_ptr());
  __nv_bfloat16* dVp = static_cast<__nv_bfloat16*>(dV.data_ptr());

  if(S<=0) return;

  size_t Dbytes  = (size_t)BH*S*sizeof(float);
  size_t dQbytes = (size_t)BH*S*DIM*sizeof(float);
  ensure_buf(&s_D,  &s_Dsz,  Dbytes);
  ensure_buf(&s_dQ, &s_dQsz, dQbytes);

  CUDA_CHECK(cudaMemsetAsync(s_dQ, 0, dQbytes, stream));

  // compute D = rowsum(dO * O)
  long total_rows = (long)BH*S;
  int rows_per_block = THREADS/32;
  dim3 gridD((total_rows + rows_per_block - 1)/rows_per_block);
  compute_D_kernel<<<gridD, THREADS, 0, stream>>>(dOp, Op, s_D, total_rows);
  CUDA_CHECK(cudaGetLastError());

  if(!s_attr_set){
    CUDA_CHECK(cudaFuncSetAttribute((const void*)mha_bwd_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    s_attr_set = true;
  }

  float scale = 1.0f / sqrtf((float)DIM);
  int numKB = (int)((S + BN - 1)/BN);
  dim3 grid(numKB, (unsigned)BH);
  mha_bwd_kernel<<<grid, THREADS, SMEM_BYTES, stream>>>(
      Qp, Kp, Vp, dOp, Lp, s_D, s_dQ, dKp, dVp, (int)S, scale);
  CUDA_CHECK(cudaGetLastError());

  size_t total = (size_t)BH*S*DIM;
  int cthreads = 256;
  size_t cblocks = (total + cthreads - 1)/cthreads;
  convert_dQ_kernel<<<(unsigned)cblocks, cthreads, 0, stream>>>(s_dQ, dQp, total);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_cuda::run);

}  // namespace mha_bwd_cuda