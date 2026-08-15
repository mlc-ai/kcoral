#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ fprintf(stderr,"CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); exit(1);} } while(0)

namespace mha_ns {

__device__ __forceinline__ void init_barrier(uint64_t* bar, int count){
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(count));
}
__device__ __forceinline__ void fence_barrier_init(){ asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase){
  asm volatile("{\n.reg .pred P;\nWL_%=:\n"
    "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
    "@!P bra WL_%=;\n}\n" :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(phase));
}
__device__ __forceinline__ void fence_proxy_async(){ asm volatile("fence.proxy.async;" ::: "memory"); }
__device__ __forceinline__ void tmem_alloc1(uint32_t* dst, int ncols){
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
    :: "r"((uint32_t)__cvta_generic_to_shared(dst)), "r"(ncols));
}
__device__ __forceinline__ void tmem_dealloc1(uint32_t addr, int ncols){
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(addr), "r"(ncols));
}
__device__ __forceinline__ void umma1(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, uint32_t accum){
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
    :: "r"(d), "l"(a), "l"(b), "r"(idesc), "r"(accum));
}
__device__ __forceinline__ void umma_commit1(uint64_t* bar){
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
    :: "r"((uint32_t)__cvta_generic_to_shared(bar)));
}
__device__ __forceinline__ void tc_fence_after(){ asm volatile("tcgen05.fence::after_thread_sync;":::"memory"); }
__device__ __forceinline__ void tc_fence_before(){ asm volatile("tcgen05.fence::before_thread_sync;":::"memory"); }
__device__ __forceinline__ void tc_wait_ld(){ asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory"); }
__device__ __forceinline__ void tc_wait_st(){ asm volatile("tcgen05.wait::st.sync.aligned;":::"memory"); }

__device__ __forceinline__ void tmem_ld32(uint32_t a, float* d){
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 "
    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
    "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
    : "=f"(d[0]),"=f"(d[1]),"=f"(d[2]),"=f"(d[3]),"=f"(d[4]),"=f"(d[5]),"=f"(d[6]),"=f"(d[7]),
      "=f"(d[8]),"=f"(d[9]),"=f"(d[10]),"=f"(d[11]),"=f"(d[12]),"=f"(d[13]),"=f"(d[14]),"=f"(d[15]),
      "=f"(d[16]),"=f"(d[17]),"=f"(d[18]),"=f"(d[19]),"=f"(d[20]),"=f"(d[21]),"=f"(d[22]),"=f"(d[23]),
      "=f"(d[24]),"=f"(d[25]),"=f"(d[26]),"=f"(d[27]),"=f"(d[28]),"=f"(d[29]),"=f"(d[30]),"=f"(d[31])
    : "r"(a));
}
__device__ __forceinline__ void tmem_st32(uint32_t a, const float* s){
  asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], "
    "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,"
    "%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};"
    :: "r"(a),
      "f"(s[0]),"f"(s[1]),"f"(s[2]),"f"(s[3]),"f"(s[4]),"f"(s[5]),"f"(s[6]),"f"(s[7]),
      "f"(s[8]),"f"(s[9]),"f"(s[10]),"f"(s[11]),"f"(s[12]),"f"(s[13]),"f"(s[14]),"f"(s[15]),
      "f"(s[16]),"f"(s[17]),"f"(s[18]),"f"(s[19]),"f"(s[20]),"f"(s[21]),"f"(s[22]),"f"(s[23]),
      "f"(s[24]),"f"(s[25]),"f"(s[26]),"f"(s[27]),"f"(s[28]),"f"(s[29]),"f"(s[30]),"f"(s[31])
    : "memory");
}

__device__ __forceinline__ uint64_t make_desc(uint32_t saddr){
  // no-swizzle, core-matrix-tiled: LBO=128, SBO=2048
  uint64_t d=0;
  d |= (uint64_t)((saddr & 0x3FFFFu) >> 4);
  d |= ((uint64_t)((128u & 0x3FFFFu) >> 4)) << 16;
  d |= ((uint64_t)((2048u & 0x3FFFFu) >> 4)) << 32;
  d |= ((uint64_t)1) << 46;
  return d;
}
__device__ __forceinline__ uint32_t make_idesc(){
  uint32_t d=0;
  d |= (1u<<4);   // dtype fp32
  d |= (1u<<7);   // atype bf16
  d |= (1u<<10);  // btype bf16
  d |= ((128u>>3)<<17); // N=128
  d |= ((128u>>4)<<24); // M=128
  return d;
}

constexpr int D=128, BM=128, BN=128;

__global__ void __launch_bounds__(128,1) mha_kernel(
  const __nv_bfloat16* __restrict__ Q,
  const __nv_bfloat16* __restrict__ K,
  const __nv_bfloat16* __restrict__ V,
  __nv_bfloat16* __restrict__ O,
  float* __restrict__ LSE,
  int S, int H)
{
  extern __shared__ __align__(128) char smem_raw[];
  __nv_bfloat16* Qs = (__nv_bfloat16*)(smem_raw + 0);
  __nv_bfloat16* Ks = (__nv_bfloat16*)(smem_raw + 32768);
  __nv_bfloat16* Vt = (__nv_bfloat16*)(smem_raw + 65536);
  __nv_bfloat16* Ps = (__nv_bfloat16*)(smem_raw + 98304);
  __shared__ __align__(8) uint64_t bar[2];
  __shared__ uint32_t tmem_base_s;

  int tid = threadIdx.x;
  int warp = tid>>5;
  int b = blockIdx.z;
  int h = blockIdx.y;
  int q_start = blockIdx.x * BM;

  long hb = (long)b*H + h;
  long head_base = hb*(long)S*D;
  const __nv_bfloat16* Qh = Q + head_base;
  const __nv_bfloat16* Kh = K + head_base;
  const __nv_bfloat16* Vh = V + head_base;
  __nv_bfloat16* Oh = O + head_base;
  float* LSEh = LSE + hb*(long)S;

  const float scale = 0.08838834764831845f;
  const __nv_bfloat16 zero_bf = __float2bfloat16(0.0f);

  if (tid==0){ init_barrier(&bar[0],1); init_barrier(&bar[1],1); }
  fence_barrier_init();
  __syncthreads();
  if (warp==0) tmem_alloc1(&tmem_base_s, 256);
  __syncthreads();
  uint32_t tmem_base = tmem_base_s;
  uint32_t tmem_S = tmem_base;
  uint32_t tmem_O = tmem_base + 128;

  uint32_t qs_base = (uint32_t)__cvta_generic_to_shared(Qs);
  uint32_t ks_base = (uint32_t)__cvta_generic_to_shared(Ks);
  uint32_t vt_base = (uint32_t)__cvta_generic_to_shared(Vt);
  uint32_t ps_base = (uint32_t)__cvta_generic_to_shared(Ps);
  uint32_t idesc = make_idesc();

  // load Q once (core-tiled)
  for (int u=tid; u<BM*(D/8); u+=128){
    int row=u>>4, k8=u&15;
    int query=q_start+row;
    int4 val=make_int4(0,0,0,0);
    if (query<S) val=*(const int4*)(Qh + (long)query*D + k8*8);
    int off=(row/8)*1024 + k8*64 + (row%8)*8;
    *(int4*)(Qs + off)=val;
  }

  int query = q_start + tid;
  float m_i = -INFINITY, l_i = 0.0f;
  uint32_t phase0=0, phase1=0;
  int tile=0;

  for (int kv_start=0; kv_start<=q_start; kv_start+=BN, tile++){
    // load K (core-tiled)
    for (int u=tid; u<BN*(D/8); u+=128){
      int row=u>>4, k8=u&15;
      int key=kv_start+row;
      int4 val=make_int4(0,0,0,0);
      if (key<S) val=*(const int4*)(Kh + (long)key*D + k8*8);
      int off=(row/8)*1024 + k8*64 + (row%8)*8;
      *(int4*)(Ks + off)=val;
    }
    // load V transposed + core-tiled : Vt[d][key]
    {
      int d=tid;
      for (int kl=0; kl<BN; kl++){
        int key=kv_start+kl;
        __nv_bfloat16 v=(key<S)?Vh[(long)key*D + d]:zero_bf;
        int off=(d/8)*1024 + (kl/8)*64 + (d%8)*8 + (kl%8);
        Vt[off]=v;
      }
    }
    __syncthreads();
    fence_proxy_async();

    // S = Q @ K^T
    if (tid==0){
      #pragma unroll
      for (int t=0;t<8;t++){
        uint64_t da=make_desc(qs_base + t*256);
        uint64_t db=make_desc(ks_base + t*256);
        umma1(tmem_S, da, db, idesc, t==0?0:1);
      }
      umma_commit1(&bar[0]);
    }
    mbar_wait(&bar[0], phase0); phase0^=1;
    tc_fence_after();

    // softmax pass1: tile rowmax
    float tilemax=-INFINITY;
    #pragma unroll
    for (int chunk=0; chunk<4; chunk++){
      float t[32];
      tmem_ld32(tmem_S + chunk*32, t);
      tc_wait_ld();
      #pragma unroll
      for (int i=0;i<32;i++){
        int c=chunk*32+i; int key=kv_start+c;
        float v=t[i]*scale;
        if (key>query || key>=S) v=-INFINITY;
        tilemax=fmaxf(tilemax,v);
      }
    }
    float m_new=fmaxf(m_i,tilemax);
    float corr=__expf(m_i-m_new);
    // pass2: exp -> Ps (core-tiled), rowsum
    float rowsum=0.0f;
    #pragma unroll
    for (int chunk=0; chunk<4; chunk++){
      float t[32];
      tmem_ld32(tmem_S + chunk*32, t);
      tc_wait_ld();
      #pragma unroll
      for (int i=0;i<32;i++){
        int c=chunk*32+i; int key=kv_start+c;
        float v=t[i]*scale;
        if (key>query || key>=S) v=-INFINITY;
        float p=__expf(v-m_new);
        int off=(tid/8)*1024 + (c/8)*64 + (tid%8)*8 + (c%8);
        Ps[off]=__float2bfloat16(p);
        rowsum+=p;
      }
    }
    l_i = l_i*corr + rowsum;
    m_i = m_new;

    __syncthreads();
    fence_proxy_async();

    // rescale O by corr (tile>0)
    if (tile>0){
      tc_fence_after();
      #pragma unroll
      for (int chunk=0; chunk<4; chunk++){
        float o[32];
        tmem_ld32(tmem_O + chunk*32, o);
        tc_wait_ld();
        #pragma unroll
        for (int i=0;i<32;i++) o[i]*=corr;
        tmem_st32(tmem_O + chunk*32, o);
      }
      tc_wait_st();
      tc_fence_before();
    }
    __syncthreads();

    // O += P @ V
    if (tid==0){
      tc_fence_after();
      #pragma unroll
      for (int t=0;t<8;t++){
        uint64_t da=make_desc(ps_base + t*256);
        uint64_t db=make_desc(vt_base + t*256);
        umma1(tmem_O, da, db, idesc, tile==0?0:1);
      }
      umma_commit1(&bar[1]);
    }
    mbar_wait(&bar[1], phase1); phase1^=1;
    __syncthreads();
  }

  // epilogue
  tc_fence_after();
  float inv = (l_i>0.0f)? (1.0f/l_i) : 0.0f;
  #pragma unroll
  for (int chunk=0; chunk<4; chunk++){
    float o[32];
    tmem_ld32(tmem_O + chunk*32, o);
    tc_wait_ld();
    if (query<S){
      #pragma unroll
      for (int j=0;j<4;j++){
        __nv_bfloat16 out[8];
        #pragma unroll
        for (int k=0;k<8;k++) out[k]=__float2bfloat16(o[j*8+k]*inv);
        *(int4*)(Oh + (long)query*D + chunk*32 + j*8) = *(const int4*)out;
      }
    }
  }
  if (query<S){
    LSEh[query] = m_i + __logf(l_i);
  }

  __syncthreads();
  if (warp==0) tmem_dealloc1(tmem_base, 256);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE){
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  int B=(int)Q.size(0), H=(int)Q.size(1), S=(int)Q.size(2);
  if (S==0) return;

  const __nv_bfloat16* Qp=static_cast<const __nv_bfloat16*>(Q.data_ptr());
  const __nv_bfloat16* Kp=static_cast<const __nv_bfloat16*>(K.data_ptr());
  const __nv_bfloat16* Vp=static_cast<const __nv_bfloat16*>(V.data_ptr());
  __nv_bfloat16* Op=static_cast<__nv_bfloat16*>(O.data_ptr());
  float* LSEp=static_cast<float*>(LSE.data_ptr());

  int smem = 131072;
  CUDA_CHECK(cudaFuncSetAttribute(mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

  dim3 grid((S+BM-1)/BM, H, B);
  cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
  mha_kernel<<<grid,128,smem,stream>>>(Qp,Kp,Vp,Op,LSEp,S,H);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_ns::run);

}  // namespace mha_ns