#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_kernel {

constexpr int D     = 128;
constexpr int BM    = 64;
constexpr int BN    = 64;
constexpr int LDS   = D + 8;     // 136, padded to avoid bank conflicts
constexpr int NT_QK = BN / 8;    // 8  N-tiles for QK
constexpr int KS_QK = D / 16;    // 8  K-steps for QK
constexpr int DT_PV = D / 8;     // 16 D-tiles (N of PV)
constexpr int KS_PV = BN / 16;   // 4  K-steps for PV

__device__ __forceinline__ void mma_m16n8k16(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1) {
  asm volatile(
   "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
   "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
   : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
   : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}

__device__ __forceinline__ uint32_t pack_bf16x2(__nv_bfloat16 lo, __nv_bfloat16 hi){
   uint16_t l = *reinterpret_cast<uint16_t*>(&lo);
   uint16_t h = *reinterpret_cast<uint16_t*>(&hi);
   return (uint32_t)l | ((uint32_t)h << 16);
}
__device__ __forceinline__ uint32_t pack_f2bf16x2(float lo, float hi){
   __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
   return *reinterpret_cast<uint32_t*>(&v);
}

__global__ __launch_bounds__(128,2)
void mha_kernel_fn(const __nv_bfloat16* __restrict__ Q,
                   const __nv_bfloat16* __restrict__ K,
                   const __nv_bfloat16* __restrict__ V,
                   __nv_bfloat16* __restrict__ O,
                   float* __restrict__ LSE,
                   int B, int H, int S){
   extern __shared__ char smem[];
   __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem);
   __nv_bfloat16* sK = sQ + BM*LDS;
   __nv_bfloat16* sV = sK + BN*LDS;

   const int qblock = blockIdx.x;
   const int h      = blockIdx.y;
   const int b      = blockIdx.z;
   const int qbase  = qblock*BM;

   const int tIdx  = threadIdx.x;
   const int warp  = tIdx / 32;
   const int lane  = tIdx % 32;
   const int group = lane / 4;   // 0..7
   const int tid   = lane % 4;   // 0..3

   const float scale = rsqrtf((float)D);
   const float NEG   = -1e30f;

   const long qkv_bh = ((long)(b*H + h))*S;  // row offset; *D for element

   // ---- Load Q into sQ (zero-fill OOB rows) ----
   for (int i = tIdx; i < BM*(D/8); i += blockDim.x){
      int row  = i / (D/8);
      int colv = i % (D/8);
      int qrow = qbase + row;
      float4 val;
      if (qrow < S) val = *reinterpret_cast<const float4*>(&Q[(qkv_bh + qrow)*D + colv*8]);
      else          val = make_float4(0,0,0,0);
      *reinterpret_cast<float4*>(&sQ[row*LDS + colv*8]) = val;
   }

   // ---- Accumulators ----
   float Oc[DT_PV][4];
   #pragma unroll
   for (int dt=0; dt<DT_PV; dt++){ Oc[dt][0]=0; Oc[dt][1]=0; Oc[dt][2]=0; Oc[dt][3]=0; }
   float m_a = NEG, m_b = NEG, l_a = 0.f, l_b = 0.f;

   __syncthreads();

   const int nblocks = (S + BN - 1)/BN;
   for (int blk=0; blk<nblocks; blk++){
      const int kvbase = blk*BN;

      // ---- Load K,V ----
      for (int i = tIdx; i < BN*(D/8); i += blockDim.x){
         int row   = i / (D/8);
         int colv  = i % (D/8);
         int kvrow = kvbase + row;
         float4 vk, vv;
         if (kvrow < S){
            vk = *reinterpret_cast<const float4*>(&K[(qkv_bh + kvrow)*D + colv*8]);
            vv = *reinterpret_cast<const float4*>(&V[(qkv_bh + kvrow)*D + colv*8]);
         } else { vk = make_float4(0,0,0,0); vv = make_float4(0,0,0,0); }
         *reinterpret_cast<float4*>(&sK[row*LDS + colv*8]) = vk;
         *reinterpret_cast<float4*>(&sV[row*LDS + colv*8]) = vv;
      }
      __syncthreads();

      // ---- QK^T ----
      float Sc[NT_QK][4];
      #pragma unroll
      for (int nt=0; nt<NT_QK; nt++){ Sc[nt][0]=0;Sc[nt][1]=0;Sc[nt][2]=0;Sc[nt][3]=0; }
      #pragma unroll
      for (int ks=0; ks<KS_QK; ks++){
         int kbase = ks*16;
         const __nv_bfloat16* qa = &sQ[(warp*16+group)*LDS   + kbase + 2*tid];
         const __nv_bfloat16* qb = &sQ[(warp*16+group+8)*LDS + kbase + 2*tid];
         uint32_t a0 = *reinterpret_cast<const uint32_t*>(qa);
         uint32_t a1 = *reinterpret_cast<const uint32_t*>(qb);
         uint32_t a2 = *reinterpret_cast<const uint32_t*>(qa+8);
         uint32_t a3 = *reinterpret_cast<const uint32_t*>(qb+8);
         #pragma unroll
         for (int nt=0; nt<NT_QK; nt++){
            int nbase = nt*8;
            const __nv_bfloat16* kb = &sK[(nbase+group)*LDS + kbase + 2*tid];
            uint32_t b0 = *reinterpret_cast<const uint32_t*>(kb);
            uint32_t b1 = *reinterpret_cast<const uint32_t*>(kb+8);
            mma_m16n8k16(Sc[nt][0],Sc[nt][1],Sc[nt][2],Sc[nt][3], a0,a1,a2,a3, b0,b1);
         }
      }

      // ---- scale + mask ----
      #pragma unroll
      for (int nt=0; nt<NT_QK; nt++){
         int kv0 = kvbase + nt*8 + 2*tid;
         int kv1 = kv0 + 1;
         Sc[nt][0] = Sc[nt][0]*scale; if (kv0>=S) Sc[nt][0]=NEG;
         Sc[nt][1] = Sc[nt][1]*scale; if (kv1>=S) Sc[nt][1]=NEG;
         Sc[nt][2] = Sc[nt][2]*scale; if (kv0>=S) Sc[nt][2]=NEG;
         Sc[nt][3] = Sc[nt][3]*scale; if (kv1>=S) Sc[nt][3]=NEG;
      }

      // ---- row max (quad reduction) ----
      float cm_a = NEG, cm_b = NEG;
      #pragma unroll
      for (int nt=0; nt<NT_QK; nt++){
         cm_a = fmaxf(cm_a, fmaxf(Sc[nt][0], Sc[nt][1]));
         cm_b = fmaxf(cm_b, fmaxf(Sc[nt][2], Sc[nt][3]));
      }
      cm_a = fmaxf(cm_a, __shfl_xor_sync(0xffffffff, cm_a, 1));
      cm_a = fmaxf(cm_a, __shfl_xor_sync(0xffffffff, cm_a, 2));
      cm_b = fmaxf(cm_b, __shfl_xor_sync(0xffffffff, cm_b, 1));
      cm_b = fmaxf(cm_b, __shfl_xor_sync(0xffffffff, cm_b, 2));

      float new_m_a = fmaxf(m_a, cm_a);
      float new_m_b = fmaxf(m_b, cm_b);
      float corr_a  = __expf(m_a - new_m_a);
      float corr_b  = __expf(m_b - new_m_b);

      float ls_a = 0.f, ls_b = 0.f;
      #pragma unroll
      for (int nt=0; nt<NT_QK; nt++){
         float e0 = __expf(Sc[nt][0]-new_m_a); Sc[nt][0]=e0; ls_a+=e0;
         float e1 = __expf(Sc[nt][1]-new_m_a); Sc[nt][1]=e1; ls_a+=e1;
         float e2 = __expf(Sc[nt][2]-new_m_b); Sc[nt][2]=e2; ls_b+=e2;
         float e3 = __expf(Sc[nt][3]-new_m_b); Sc[nt][3]=e3; ls_b+=e3;
      }
      ls_a += __shfl_xor_sync(0xffffffff, ls_a, 1);
      ls_a += __shfl_xor_sync(0xffffffff, ls_a, 2);
      ls_b += __shfl_xor_sync(0xffffffff, ls_b, 1);
      ls_b += __shfl_xor_sync(0xffffffff, ls_b, 2);

      l_a = l_a*corr_a + ls_a;
      l_b = l_b*corr_b + ls_b;

      // ---- correct existing O ----
      #pragma unroll
      for (int dt=0; dt<DT_PV; dt++){
         Oc[dt][0]*=corr_a; Oc[dt][1]*=corr_a;
         Oc[dt][2]*=corr_b; Oc[dt][3]*=corr_b;
      }

      // ---- P @ V ----
      #pragma unroll
      for (int ks=0; ks<KS_PV; ks++){
         uint32_t pa0 = pack_f2bf16x2(Sc[2*ks][0],   Sc[2*ks][1]);
         uint32_t pa1 = pack_f2bf16x2(Sc[2*ks][2],   Sc[2*ks][3]);
         uint32_t pa2 = pack_f2bf16x2(Sc[2*ks+1][0], Sc[2*ks+1][1]);
         uint32_t pa3 = pack_f2bf16x2(Sc[2*ks+1][2], Sc[2*ks+1][3]);
         int kk = ks*16;
         #pragma unroll
         for (int dt=0; dt<DT_PV; dt++){
            int ncol = dt*8 + group;  // d index
            __nv_bfloat16 v00 = sV[(kk+2*tid)*LDS   + ncol];
            __nv_bfloat16 v01 = sV[(kk+2*tid+1)*LDS + ncol];
            __nv_bfloat16 v10 = sV[(kk+2*tid+8)*LDS + ncol];
            __nv_bfloat16 v11 = sV[(kk+2*tid+9)*LDS + ncol];
            uint32_t b0 = pack_bf16x2(v00, v01);
            uint32_t b1 = pack_bf16x2(v10, v11);
            mma_m16n8k16(Oc[dt][0],Oc[dt][1],Oc[dt][2],Oc[dt][3], pa0,pa1,pa2,pa3, b0,b1);
         }
      }

      m_a = new_m_a; m_b = new_m_b;
      __syncthreads();
   }

   // ---- finalize ----
   float inv_a = 1.f/l_a;
   float inv_b = 1.f/l_b;
   int row_a = qbase + warp*16 + group;
   int row_b = qbase + warp*16 + group + 8;
   #pragma unroll
   for (int dt=0; dt<DT_PV; dt++){
      int d0 = 8*dt + 2*tid;
      if (row_a < S){
         __nv_bfloat162 ov = __floats2bfloat162_rn(Oc[dt][0]*inv_a, Oc[dt][1]*inv_a);
         *reinterpret_cast<__nv_bfloat162*>(&O[(qkv_bh + row_a)*D + d0]) = ov;
      }
      if (row_b < S){
         __nv_bfloat162 ov = __floats2bfloat162_rn(Oc[dt][2]*inv_b, Oc[dt][3]*inv_b);
         *reinterpret_cast<__nv_bfloat162*>(&O[(qkv_bh + row_b)*D + d0]) = ov;
      }
   }
   if (tid==0){
      if (row_a < S) LSE[qkv_bh + row_a] = m_a + logf(l_a);
      if (row_b < S) LSE[qkv_bh + row_b] = m_b + logf(l_b);
   }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int B = (int)Q.size(0);
  int H = (int)Q.size(1);
  int S = (int)Q.size(2);

  const __nv_bfloat16* Qp = static_cast<const __nv_bfloat16*>(Q.data_ptr());
  const __nv_bfloat16* Kp = static_cast<const __nv_bfloat16*>(K.data_ptr());
  const __nv_bfloat16* Vp = static_cast<const __nv_bfloat16*>(V.data_ptr());
  __nv_bfloat16* Op = static_cast<__nv_bfloat16*>(O.data_ptr());
  float* LSEp = static_cast<float*>(LSE.data_ptr());

  int shmem = (BM + 2*BN)*LDS*2;  // bytes
  cudaFuncSetAttribute(mha_kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);

  dim3 grid((S+BM-1)/BM, H, B);
  dim3 block(128);
  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
  mha_kernel_fn<<<grid, block, shmem, stream>>>(Qp,Kp,Vp,Op,LSEp,B,H,S);
  CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel::run);

}  // namespace mha_kernel