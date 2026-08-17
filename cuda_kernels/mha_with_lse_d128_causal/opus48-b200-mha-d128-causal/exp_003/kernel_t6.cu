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

__device__ __forceinline__ uint64_t make_desc(uint32_t saddr, uint32_t lbo, uint32_t sbo){
  uint64_t d=0;
  d |= (uint64_t)((saddr & 0x3FFFFu) >> 4);
  d |= ((uint64_t)((lbo & 0x3FFFFu) >> 4)) << 16;
  d |= ((uint64_t)((sbo & 0x3FFFFu) >> 4)) << 32;
  d |= ((uint64_t)1) << 46;
  return d;
}
__device__ __forceinline__ uint32_t make_idesc(int transposeB){
  uint32_t d=0;
  d |= (1u<<4); d |= (1u<<7); d |= (1u<<10);
  d |= ((uint32_t)transposeB<<16);
  d |= ((128u>>3)<<17); d |= ((128u>>4)<<24);
  return d;
}
__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem){
  uint32_t s=(uint32_t)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"::"r"(s),"l"(gmem):"memory");
}
__device__ __forceinline__ void cp_async_commit(){ asm volatile("cp.async.commit_group;\n":::"memory"); }
template<int N> __device__ __forceinline__ void cp_async_wait(){ asm volatile("cp.async.wait_group %0;\n"::"n"(N):"memory"); }

__device__ __forceinline__ float warp_reduce_max(float v){
  v=fmaxf(v,__shfl_xor_sync(0xffffffffu,v,16));
  v=fmaxf(v,__shfl_xor_sync(0xffffffffu,v,8));
  v=fmaxf(v,__shfl_xor_sync(0xffffffffu,v,4));
  v=fmaxf(v,__shfl_xor_sync(0xffffffffu,v,2));
  v=fmaxf(v,__shfl_xor_sync(0xffffffffu,v,1));
  return v;
}

constexpr int D=128, BM=128, BN=128;
constexpr float TAU=5.0f;

__device__ __forceinline__ void load_tile(__nv_bfloat16* dst, const __nv_bfloat16* src, int start, int S, int tid){
  int4 z=make_int4(0,0,0,0);
  #pragma unroll
  for(int u=tid;u<128*16;u+=128){
    int row=u>>4, k8=u&15; int key=start+row;
    int off=k8*1024+(row>>3)*64+(row&7)*8;
    if(key<S) cp_async_16(dst+off, src+(long)key*128+k8*8);
    else *(int4*)(dst+off)=z;
  }
}

__global__ void __launch_bounds__(128,1) mha_kernel(
  const __nv_bfloat16* __restrict__ Q,
  const __nv_bfloat16* __restrict__ K,
  const __nv_bfloat16* __restrict__ V,
  __nv_bfloat16* __restrict__ O,
  float* __restrict__ LSE,
  int S, int H)
{
  extern __shared__ __align__(128) char smem_raw[];
  __nv_bfloat16* Qs    = (__nv_bfloat16*)(smem_raw + 0);
  __nv_bfloat16* Kb[2] = { (__nv_bfloat16*)(smem_raw+32768),  (__nv_bfloat16*)(smem_raw+65536) };
  __nv_bfloat16* Vb[2] = { (__nv_bfloat16*)(smem_raw+98304),  (__nv_bfloat16*)(smem_raw+131072) };
  __nv_bfloat16* Pb[2] = { (__nv_bfloat16*)(smem_raw+163840), (__nv_bfloat16*)(smem_raw+196608) };
  __shared__ __align__(8) uint64_t bar_s[2];
  __shared__ __align__(8) uint64_t bar_o;
  __shared__ uint32_t tmem_base_s;

  int tid = threadIdx.x;
  int warp = tid>>5;
  int b = blockIdx.z, h = blockIdx.y;
  int q_start = blockIdx.x * BM;

  long hb = (long)b*H + h;
  long head_base = hb*(long)S*D;
  const __nv_bfloat16* Qh = Q + head_base;
  const __nv_bfloat16* Kh = K + head_base;
  const __nv_bfloat16* Vh = V + head_base;
  __nv_bfloat16* Oh = O + head_base;
  float* LSEh = LSE + hb*(long)S;

  const float scale = 0.08838834764831845f;

  if (tid==0){ init_barrier(&bar_s[0],1); init_barrier(&bar_s[1],1); init_barrier(&bar_o,1); }
  fence_barrier_init();
  __syncthreads();
  if (warp==0) tmem_alloc1(&tmem_base_s, 512);
  __syncthreads();
  uint32_t tmem_base = tmem_base_s;
  uint32_t tmem_S[2] = { tmem_base, tmem_base+128 };
  uint32_t tmem_O = tmem_base + 256;

  uint32_t qs_base = (uint32_t)__cvta_generic_to_shared(Qs);
  uint32_t ks_base[2] = { (uint32_t)__cvta_generic_to_shared(Kb[0]), (uint32_t)__cvta_generic_to_shared(Kb[1]) };
  uint32_t vs_base[2] = { (uint32_t)__cvta_generic_to_shared(Vb[0]), (uint32_t)__cvta_generic_to_shared(Vb[1]) };
  uint32_t ps_base[2] = { (uint32_t)__cvta_generic_to_shared(Pb[0]), (uint32_t)__cvta_generic_to_shared(Pb[1]) };
  uint32_t idesc_s = make_idesc(0);
  uint32_t idesc_o = make_idesc(1);

  int query = q_start + tid;
  int ntiles = q_start / BN + 1;

  // prologue: load Q, K0, V0; issue QK0
  {
    int4 z=make_int4(0,0,0,0);
    for(int u=tid;u<128*16;u+=128){
      int row=u>>4,k8=u&15; int q=q_start+row;
      int off=k8*1024+(row>>3)*64+(row&7)*8;
      if(q<S) cp_async_16(Qs+off, Qh+(long)q*128+k8*8);
      else *(int4*)(Qs+off)=z;
    }
    load_tile(Kb[0],Kh,0,S,tid);
    load_tile(Vb[0],Vh,0,S,tid);
    cp_async_commit(); cp_async_wait<0>();
    __syncthreads(); fence_proxy_async();
    if(tid==0){
      #pragma unroll
      for(int t=0;t<8;t++){
        uint64_t da=make_desc(qs_base+t*4096,2048,128);
        uint64_t db=make_desc(ks_base[0]+t*4096,2048,128);
        umma1(tmem_S[0], da, db, idesc_s, t==0?0:1);
      }
      umma_commit1(&bar_s[0]);
    }
  }

  float m_i=-INFINITY, l_i=0.0f, r_off=-INFINITY;
  uint32_t ph_s[2]={0,0}, ph_o=0;
  float sreg[4][32];

  for(int i=0;i<ntiles;i++){
    int cur=i&1, nxt=(i+1)&1;
    int kv_start=i*BN;
    bool have_next=(i+1)<ntiles;
    bool is_diag=(kv_start==q_start);
    int keylim=S-kv_start;
    bool need_mask = is_diag || (kv_start+BN > S);

    // wait QK(i)
    mbar_wait(&bar_s[cur], ph_s[cur]); ph_s[cur]^=1; tc_fence_after();

    // load K(i+1) into nxt (free: QK(i-1) done), issue QK(i+1) EARLY to overlap softmax
    if(have_next){
      load_tile(Kb[nxt],Kh,(i+1)*BN,S,tid);
      cp_async_commit(); cp_async_wait<0>();
      __syncthreads(); fence_proxy_async();
      if(tid==0){
        #pragma unroll
        for(int t=0;t<8;t++){
          uint64_t da=make_desc(qs_base+t*4096,2048,128);
          uint64_t db=make_desc(ks_base[nxt]+t*4096,2048,128);
          umma1(tmem_S[nxt], da, db, idesc_s, t==0?0:1);
        }
        umma_commit1(&bar_s[nxt]);   // runs concurrently with softmax(i) + PV(i-1)
      }
    }

    // softmax(i): single load pass, compute tilemax
    #pragma unroll
    for(int c=0;c<4;c++) tmem_ld32(tmem_S[cur]+c*32, sreg[c]);
    tc_wait_ld();
    float tilemax=-INFINITY;
    #pragma unroll
    for(int c=0;c<4;c++){
      #pragma unroll
      for(int ii=0;ii<32;ii++){
        int kc=c*32+ii;
        float v=sreg[c][ii]*scale;
        bool valid = need_mask ? ((kc<keylim)&&(!is_diag||kc<=tid)) : true;
        sreg[c][ii] = valid ? v : -INFINITY;
        if(valid) tilemax=fmaxf(tilemax,v);
      }
    }
    float m_new=fmaxf(m_i,tilemax);
    float corr=1.0f; bool do_rescale=false;
    if(i==0){ r_off=m_new; }
    else {
      float wgap=warp_reduce_max(m_new-r_off);
      if(wgap>TAU){ do_rescale=true; corr=__expf(r_off-m_new); r_off=m_new; }
    }
    m_i=m_new;

    float rowsum=0.0f;
    #pragma unroll
    for(int c=0;c<4;c++){
      __nv_bfloat16 pb[32];
      #pragma unroll
      for(int ii=0;ii<32;ii++){
        float p=__expf(sreg[c][ii]-r_off);
        rowsum+=p;
        pb[ii]=__float2bfloat16(p);
      }
      #pragma unroll
      for(int g=0;g<4;g++){
        int kb=c*4+g;
        int off=kb*1024+(tid>>3)*64+(tid&7)*8;
        *(int4*)(Pb[cur]+off) = *(int4*)(pb+g*8);
      }
    }
    if(do_rescale) l_i*=corr;
    l_i+=rowsum;

    // wait PV(i-1) -> Vb[nxt] free, O stable
    if(i>0){ mbar_wait(&bar_o, ph_o); ph_o^=1; tc_fence_after(); }

    // load V(i+1) into nxt (now free)
    if(have_next){ load_tile(Vb[nxt],Vh,(i+1)*BN,S,tid); cp_async_commit(); }

    // rescale O (rare)
    if(do_rescale){
      #pragma unroll
      for(int c=0;c<4;c++){
        float o[32]; tmem_ld32(tmem_O+c*32,o); tc_wait_ld();
        #pragma unroll
        for(int ii=0;ii<32;ii++) o[ii]*=corr;
        tmem_st32(tmem_O+c*32,o);
      }
      tc_wait_st(); tc_fence_before();
    }

    if(have_next) cp_async_wait<0>();   // V(i+1) done
    __syncthreads(); fence_proxy_async();

    // issue PV(i) -> overlaps softmax(i+1)
    if(tid==0){
      #pragma unroll
      for(int t=0;t<8;t++){
        uint64_t da=make_desc(ps_base[cur]+t*4096,2048,128);
        uint64_t db=make_desc(vs_base[cur]+t*256,128,2048);
        uint32_t acc=(i==0 && t==0)?0:1;
        umma1(tmem_O, da, db, idesc_o, acc);
      }
      umma_commit1(&bar_o);
    }
  }

  mbar_wait(&bar_o, ph_o); ph_o^=1; tc_fence_after();

  float inv = (l_i>0.0f)? (1.0f/l_i) : 0.0f;
  float oreg[4][32];
  #pragma unroll
  for(int c=0;c<4;c++) tmem_ld32(tmem_O+c*32,oreg[c]);
  tc_wait_ld();
  if(query<S){
    #pragma unroll
    for(int c=0;c<4;c++){
      #pragma unroll
      for(int j=0;j<4;j++){
        __nv_bfloat16 out[8];
        #pragma unroll
        for(int k=0;k<8;k++) out[k]=__float2bfloat16(oreg[c][j*8+k]*inv);
        *(int4*)(Oh+(long)query*D + c*32 + j*8) = *(const int4*)out;
      }
    }
    LSEh[query] = r_off + __logf(l_i);
  }

  __syncthreads();
  if(warp==0) tmem_dealloc1(tmem_base, 512);
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

  int smem = 229376;
  CUDA_CHECK(cudaFuncSetAttribute(mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

  dim3 grid((S+BM-1)/BM, H, B);
  cudaStream_t stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
  mha_kernel<<<grid,128,smem,stream>>>(Qp,Kp,Vp,Op,LSEp,S,H);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_ns::run);

}  // namespace mha_ns