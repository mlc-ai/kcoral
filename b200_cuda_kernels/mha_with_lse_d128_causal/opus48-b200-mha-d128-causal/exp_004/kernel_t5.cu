#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <math.h>
#include <stdio.h>
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

__device__ __forceinline__ uint32_t smem_u32(const void* p){
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void ldm_x4(uint32_t addr, uint32_t &r0,uint32_t&r1,uint32_t&r2,uint32_t&r3){
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
    :"=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3):"r"(addr));
}
__device__ __forceinline__ void ldm_x2(uint32_t addr, uint32_t &r0,uint32_t&r1){
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2];\n"
    :"=r"(r0),"=r"(r1):"r"(addr));
}
__device__ __forceinline__ void ldm_x2_trans(uint32_t addr, uint32_t &r0,uint32_t&r1){
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];\n"
    :"=r"(r0),"=r"(r1):"r"(addr));
}
__device__ __forceinline__ void mma16816(float* acc,
    uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3, uint32_t b0,uint32_t b1){
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(acc[0]),"+f"(acc[1]),"+f"(acc[2]),"+f"(acc[3])
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}

__device__ __forceinline__ void load_tile_sync(const __nv_bfloat16* src, __nv_bfloat16* dst, int row0, int S){
  int tid = threadIdx.x;
  #pragma unroll
  for(int k=0;k<8;k++){
    int idx = tid + k*128;
    int r  = idx >> 4;
    int c8 = (idx & 15) << 3;
    int grow = row0 + r;
    int4 v;
    if(grow < S){ v = *reinterpret_cast<const int4*>(&src[(long)grow*128 + c8]); }
    else { v.x=v.y=v.z=v.w=0; }
    *reinterpret_cast<int4*>(&dst[r*128 + c8]) = v;
  }
}

__global__ void __launch_bounds__(128,3) attn_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE, int S)
{
  const int Hn = gridDim.y;
  const int b  = blockIdx.z;
  const int h  = blockIdx.y;
  const int qb = blockIdx.x;
  const long bh = (long)b*Hn + h;

  const __nv_bfloat16* Qp = Q + bh*(long)S*128;
  const __nv_bfloat16* Kp = K + bh*(long)S*128;
  const __nv_bfloat16* Vp = V + bh*(long)S*128;
  __nv_bfloat16* Op = O + bh*(long)S*128;
  float* LSEp = LSE + bh*(long)S;

  extern __shared__ char smem_raw[];
  __nv_bfloat16* Qs = (__nv_bfloat16*)smem_raw;   // 8192
  __nv_bfloat16* Ks = Qs + 8192;                  // 8192
  __nv_bfloat16* Vs = Ks + 8192;                  // 8192
  __nv_bfloat16* Ps = Vs + 8192;                  // 4096

  const int tid  = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int gid  = lane >> 2;
  const int tig  = lane & 3;

  const int q_row0 = qb*64;

  load_tile_sync(Qp, Qs, q_row0, S);
  __syncthreads();

  const float scale = rsqrtf(128.0f);

  float m0 = -INFINITY, m1 = -INFINITY;
  float l0 = 0.f, l1 = 0.f;
  float Oreg[16][4];
  #pragma unroll
  for(int nb=0;nb<16;nb++){ Oreg[nb][0]=0.f;Oreg[nb][1]=0.f;Oreg[nb][2]=0.f;Oreg[nb][3]=0.f; }

  for(int kv=0; kv<=qb; kv++){
    load_tile_sync(Kp, Ks, kv*64, S);
    load_tile_sync(Vp, Vs, kv*64, S);
    __syncthreads();

    float Sreg[8][4];
    #pragma unroll
    for(int nb=0;nb<8;nb++){ Sreg[nb][0]=0.f;Sreg[nb][1]=0.f;Sreg[nb][2]=0.f;Sreg[nb][3]=0.f; }

    #pragma unroll
    for(int ks=0; ks<8; ks++){
      uint32_t a0,a1,a2,a3;
      uint32_t aAddr = smem_u32(&Qs[(warp*16 + (lane&15))*128 + ks*16 + ((lane>>4)&1)*8]);
      ldm_x4(aAddr, a0,a1,a2,a3);
      uint32_t bf[8][2];
      #pragma unroll
      for(int nb=0; nb<8; nb++){
        uint32_t bAddr = smem_u32(&Ks[(8*nb + (lane&7))*128 + ks*16 + ((lane>>3)&1)*8]);
        ldm_x2(bAddr, bf[nb][0], bf[nb][1]);
      }
      #pragma unroll
      for(int nb=0; nb<8; nb++){
        mma16816(Sreg[nb], a0,a1,a2,a3, bf[nb][0], bf[nb][1]);
      }
    }

    const int gq0 = q_row0 + warp*16 + gid;
    const int gq1 = q_row0 + warp*16 + gid + 8;
    const bool diag = (kv==qb);
    #pragma unroll
    for(int nb=0; nb<8; nb++){
      int c_a = nb*8 + tig*2;
      int c_b = c_a + 1;
      int gk_a = kv*64 + c_a;
      int gk_b = kv*64 + c_b;
      float s0 = Sreg[nb][0]*scale;
      float s1 = Sreg[nb][1]*scale;
      float s2 = Sreg[nb][2]*scale;
      float s3 = Sreg[nb][3]*scale;
      if(diag){
        if(!(gk_a<=gq0 && gk_a<S)) s0 = -INFINITY;
        if(!(gk_b<=gq0 && gk_b<S)) s1 = -INFINITY;
        if(!(gk_a<=gq1 && gk_a<S)) s2 = -INFINITY;
        if(!(gk_b<=gq1 && gk_b<S)) s3 = -INFINITY;
      }
      Sreg[nb][0]=s0; Sreg[nb][1]=s1; Sreg[nb][2]=s2; Sreg[nb][3]=s3;
    }

    float bm0 = -INFINITY, bm1 = -INFINITY;
    #pragma unroll
    for(int nb=0;nb<8;nb++){
      bm0 = fmaxf(bm0, fmaxf(Sreg[nb][0], Sreg[nb][1]));
      bm1 = fmaxf(bm1, fmaxf(Sreg[nb][2], Sreg[nb][3]));
    }
    bm0 = fmaxf(bm0, __shfl_xor_sync(0xffffffff, bm0, 1));
    bm0 = fmaxf(bm0, __shfl_xor_sync(0xffffffff, bm0, 2));
    bm1 = fmaxf(bm1, __shfl_xor_sync(0xffffffff, bm1, 1));
    bm1 = fmaxf(bm1, __shfl_xor_sync(0xffffffff, bm1, 2));

    float mn0 = fmaxf(m0, bm0);
    float mn1 = fmaxf(m1, bm1);

    float alpha0, alpha1;
    if(mn0==-INFINITY) alpha0=1.f; else if(m0==-INFINITY) alpha0=0.f; else alpha0=__expf(m0-mn0);
    if(mn1==-INFINITY) alpha1=1.f; else if(m1==-INFINITY) alpha1=0.f; else alpha1=__expf(m1-mn1);

    l0 *= alpha0; l1 *= alpha1;
    #pragma unroll
    for(int nb=0;nb<16;nb++){
      Oreg[nb][0]*=alpha0; Oreg[nb][1]*=alpha0;
      Oreg[nb][2]*=alpha1; Oreg[nb][3]*=alpha1;
    }

    float ps0=0.f, ps1=0.f;
    #pragma unroll
    for(int nb=0;nb<8;nb++){
      float p0 = (mn0==-INFINITY)?0.f:__expf(Sreg[nb][0]-mn0);
      float p1 = (mn0==-INFINITY)?0.f:__expf(Sreg[nb][1]-mn0);
      float p2 = (mn1==-INFINITY)?0.f:__expf(Sreg[nb][2]-mn1);
      float p3 = (mn1==-INFINITY)?0.f:__expf(Sreg[nb][3]-mn1);
      ps0 += p0+p1; ps1 += p2+p3;
      int row0s = warp*16 + gid;
      int row1s = warp*16 + gid + 8;
      int col = nb*8 + tig*2;
      Ps[row0s*64 + col]     = __float2bfloat16(p0);
      Ps[row0s*64 + col + 1] = __float2bfloat16(p1);
      Ps[row1s*64 + col]     = __float2bfloat16(p2);
      Ps[row1s*64 + col + 1] = __float2bfloat16(p3);
    }
    ps0 += __shfl_xor_sync(0xffffffff, ps0, 1);
    ps0 += __shfl_xor_sync(0xffffffff, ps0, 2);
    ps1 += __shfl_xor_sync(0xffffffff, ps1, 1);
    ps1 += __shfl_xor_sync(0xffffffff, ps1, 2);
    l0 += ps0; l1 += ps1;
    m0 = mn0; m1 = mn1;

    __syncwarp();

    #pragma unroll
    for(int kt=0; kt<4; kt++){
      uint32_t pa0,pa1,pa2,pa3;
      uint32_t pAddr = smem_u32(&Ps[(warp*16 + (lane&15))*64 + kt*16 + ((lane>>4)&1)*8]);
      ldm_x4(pAddr, pa0,pa1,pa2,pa3);
      #pragma unroll
      for(int g=0; g<2; g++){
        uint32_t vf[8][2];
        #pragma unroll
        for(int j=0;j<8;j++){
          int nb = g*8 + j;
          uint32_t vAddr = smem_u32(&Vs[(kt*16 + (lane&15))*128 + nb*8]);
          ldm_x2_trans(vAddr, vf[j][0], vf[j][1]);
        }
        #pragma unroll
        for(int j=0;j<8;j++){
          int nb = g*8 + j;
          mma16816(Oreg[nb], pa0,pa1,pa2,pa3, vf[j][0], vf[j][1]);
        }
      }
    }
    __syncthreads();
  }

  const int gq0 = q_row0 + warp*16 + gid;
  const int gq1 = q_row0 + warp*16 + gid + 8;
  float inv0 = (l0>0.f)? 1.f/l0 : 0.f;
  float inv1 = (l1>0.f)? 1.f/l1 : 0.f;

  #pragma unroll
  for(int nb=0;nb<16;nb++){
    int d0 = nb*8 + tig*2;
    int d1 = d0 + 1;
    if(gq0 < S){
      Op[(long)gq0*128 + d0] = __float2bfloat16(Oreg[nb][0]*inv0);
      Op[(long)gq0*128 + d1] = __float2bfloat16(Oreg[nb][1]*inv0);
    }
    if(gq1 < S){
      Op[(long)gq1*128 + d0] = __float2bfloat16(Oreg[nb][2]*inv1);
      Op[(long)gq1*128 + d1] = __float2bfloat16(Oreg[nb][3]*inv1);
    }
  }
  if(tig==0){
    if(gq0 < S) LSEp[gq0] = m0 + logf(l0);
    if(gq1 < S) LSEp[gq1] = m1 + logf(l1);
  }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int S = (int)Q.size(2);

  const __nv_bfloat16* Qp = static_cast<const __nv_bfloat16*>(Q.data_ptr());
  const __nv_bfloat16* Kp = static_cast<const __nv_bfloat16*>(K.data_ptr());
  const __nv_bfloat16* Vp = static_cast<const __nv_bfloat16*>(V.data_ptr());
  __nv_bfloat16* Op = static_cast<__nv_bfloat16*>(O.data_ptr());
  float* LSEp = static_cast<float*>(LSE.data_ptr());

  int B  = (int)Q.size(0);
  int Hh = (int)Q.size(1);
  int num_q = (S + 63) / 64;
  dim3 grid(num_q, Hh, B);
  dim3 block(128);
  size_t smem_bytes = (size_t)(8192*3 + 4096) * sizeof(__nv_bfloat16);

  static bool attr_set = false;
  if(!attr_set){
    CUDA_CHECK(cudaFuncSetAttribute(attn_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));
    attr_set = true;
  }

  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

  attn_kernel<<<grid, block, smem_bytes, stream>>>(Qp, Kp, Vp, Op, LSEp, S);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel::run);

}  // namespace mha_kernel