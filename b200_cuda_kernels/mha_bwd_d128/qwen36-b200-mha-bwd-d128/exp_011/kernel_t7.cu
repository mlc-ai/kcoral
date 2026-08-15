#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdio.h>
#include <cstdint>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call)                                           \
    do {                                                           \
        cudaError_t _e = (call);                                   \
        if (_e != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error %s at %s:%d\n",           \
                    cudaGetErrorString(_e), __FILE__, __LINE__);   \
            exit(1);                                               \
        }                                                          \
    } while (0)

static constexpr int TILE_Q = 32;
static constexpr int TILE_K = 32;
static constexpr int NT     = 128;
static constexpr int NPAIRS = (TILE_Q * TILE_K) / NT;

__global__ void mha_bwd_dq(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O_fwd,
    const __nv_bfloat16* __restrict__ dO_in,
    const float* __restrict__ LSE,
    __nv_bfloat16* __restrict__ dQ_out,
    int64_t B, int64_t H, int64_t S, int64_t D,
    float inv_sqrt_d)
{
    // Shared mem: sh_Q[32x128], sh_O[32x128], sh_dO[32x128], sh_K[32x128], sh_V[32x128] = 40KB bf16
    //             sh_acc[32x128] fp32 = 16KB. Total ~56KB.
    extern __shared__ __align__(16) unsigned char mem[];
    __nv_bfloat16* sq   = reinterpret_cast<__nv_bfloat16*>(mem);
    __nv_bfloat16* so   = sq + TILE_Q*D;
    __nv_bfloat16* sd   = so + TILE_Q*D;
    __nv_bfloat16* sk   = sd + TILE_Q*D;
    __nv_bfloat16* sv   = sk + TILE_K*D;
    float* acc          = reinterpret_cast<float*>(sv + TILE_K*D);

    int tid    = threadIdx.x;
    int64_t bh = blockIdx.z;
    int64_t qt = blockIdx.x;
    int64_t qs = qt * TILE_Q;
    int64_t qe = min(qs+TILE_Q, S);
    int64_t nq = qe - qs;
    int64_t boff = bh * S * D;
    int64_t nktil = (S + TILE_K - 1) / TILE_K;

    // Load Q, O_fwd, dO once (stay in SMEM across k-sweep)
    for (int64_t i=tid; i<nq*D; i+=NT) {
        int64_t r=i/D, c=i%D;
        int64_t g=boff+(qs+r)*D+c;
        sq[i]=Q[g]; so[i]=O_fwd[g]; sd[i]=dO_in[g];
    }
    for (int64_t i=tid; i<nq*D; i+=NT) acc[i]=0.0f;
    __syncthreads();

    float dQ_reg[TILE_Q];
    for (int64_t s=tid; s<nq; s+=NT) {
        float v=0.f;
        for (int d=0;d<D;d++) v += __bfloat162float(sd[s*D+d])*__bfloat162float(so[s*D+d]);
        dQ_reg[s]=v;
    }
    __syncthreads();

    for (int64_t kt=0; kt<nktil; kt++) {
        int64_t ks=kt*TILE_K; int64_t ke=min(ks+TILE_K,S); int64_t nk=ke-ks;
        for (int64_t i=tid;i<nk*D;i+=NT) {
            int64_t r=i/D,c=i%D; int64_t g=boff+(ks+r)*D+c;
            sk[i]=K[g]; sv[i]=V[g];
        }
        __syncthreads();
        #pragma unroll 4
        for (int p=0;p<NPAIRS;p++) {
            int64_t gp=tid*NPAIRS+p; if(gp>=nq*nk) break;
            int64_t sg=gp/TILE_K, gk=gp%TILE_K;
            float sc=0.f;
            for(int dd=0;dd<D;dd++) sc+=__bfloat162float(sq[sg*D+dd])*__bfloat162float(sk[gk*D+dd]);
            sc*=inv_sqrt_d;
            float pv=expf(sc-LSE[bh*S+qs+sg]);
            float da=0.f;
            for(int dd=0;dd<D;dd++) da+=__bfloat162float(sd[sg*D+dd])*__bfloat162float(sv[gk*D+dd]);
            float ds=pv*(da-dQ_reg[sg])*inv_sqrt_d;
            for(int dd=0;dd<D;dd++) acc[sg*D+dd]+=ds*__bfloat162float(sk[gk*D+dd]);
        }
        __syncthreads();
    }

    for (int64_t i=tid;i<nq*D;i+=NT) {
        int64_t r=i/D,c=i%D;
        dQ_out[boff+(qs+r)*D+c]=__float2bfloat16(acc[i]);
    }
}

__global__ void mha_bwd_dk(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O_fwd,
    const __nv_bfloat16* __restrict__ dO_in,
    const float* __restrict__ LSE,
    __nv_bfloat16* __restrict__ dK_out,
    int64_t B, int64_t H, int64_t S, int64_t D,
    float inv_sqrt_d)
{
    extern __shared__ __align__(16) unsigned char mem[];
    __nv_bfloat16* sq   = reinterpret_cast<__nv_bfloat16*>(mem);
    __nv_bfloat16* so   = sq + TILE_Q*D;
    __nv_bfloat16* sd   = so + TILE_Q*D;
    __nv_bfloat16* sk   = sd + TILE_Q*D;
    __nv_bfloat16* sv   = sk + TILE_K*D;
    float* acc          = reinterpret_cast<float*>(sv + TILE_K*D);

    int tid    = threadIdx.x;
    int64_t bh = blockIdx.z;
    int64_t kt = blockIdx.x;
    int64_t ks = kt * TILE_K;
    int64_t ke = min(ks+TILE_K, S);
    int64_t nk = ke - ks;
    int64_t boff = bh * S * D;
    int64_t nqtil = (S + TILE_Q - 1) / TILE_Q;

    for (int64_t i=tid;i<nk*D;i+=NT) {
        int64_t r=i/D,c=i%D; int64_t g=boff+(ks+r)*D+c;
        sk[i]=K[g];
    }
    for (int64_t i=tid;i<nk*D;i+=NT) acc[i]=0.0f;
    __syncthreads();

    for (int64_t qt=0;qt<nqtil;qt++) {
        int64_t qs_q=qt*TILE_Q; int64_t qe_q=min(qs_q+TILE_Q,S); int64_t nqq=qe_q-q;
        if(nqq==0) continue;
        for (int64_t i=tid;i<nqq*D;i+=NT) {
            int64_t r=i/D,c=i%D; int64_t g=boff+(qs_q+r)*D+c;
            sq[i]=Q[g]; so[i]=O_fwd[g]; sd[i]=dO_in[g];
        }
        __syncthreads();
        for (int64_t i=tid;i<nk*D;i+=NT) {
            int64_t r=i/D,c=i%D; int64_t g=boff+(ks+r)*D+c;
            sv[i]=V[g];
        }
        __syncthreads();
        float dQ_r[TILE_Q];
        for (int64_t s=tid;s<nqq;s+=NT) {
            float v=0.f;
            for(int d=0;d<D;d++) v+=__bfloat162float(sd[s*D+d])*__bfloat162float(so[s*D+d]);
            dQ_r[s]=v;
        }
        __syncthreads();
        #pragma unroll 4
        for(int p=0;p<NPAIRS;p++){
            int64_t gp=tid*NPAIRS+p;if(gp>=nqq*nk)break;
            int64_t sg=gp/TILE_K,gk=gp%TILE_K;
            float sc=0.f;
            for(int dd=0;dd<D;dd++) sc+=__bfloat162float(sq[sg*D+dd])*__bfloat162float(sk[gk*D+dd]);
            sc*=inv_sqrt_d;
            float pv=expf(sc-LSE[bh*S+qs_q+sg]);
            float da=0.f;
            for(int dd=0;dd<D;dd++) da+=__bfloat162float(sd[sg*D+dd])*__bfloat162float(sv[gk*D+dd]);
            float ds=pv*(da-dQ_r[sg])*inv_sqrt_d;
            for(int dd=0;dd<D;dd++) acc[gk*D+dd]+=ds*__bfloat162float(sq[sg*D+dd]);
        }
        __syncthreads();
    }

    for (int64_t i=tid;i<nk*D;i+=NT) {
        int64_t r=i/D,c=i%D;
        dK_out[boff+(ks+r)*D+c]=__float2bfloat16(acc[i]);
    }
}

__global__ void mha_bwd_dv(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O_fwd,
    const __nv_bfloat16* __restrict__ dO_in,
    const float* __restrict__ LSE,
    __nv_bfloat16* __restrict__ dV_out,
    int64_t B, int64_t H, int64_t S, int64_t D,
    float inv_sqrt_d)
{
    extern __shared__ __align__(16) unsigned char mem[];
    __nv_bfloat16* sq   = reinterpret_cast<__nv_bfloat16*>(mem);
    __nv_bfloat16* so   = sq + TILE_Q*D;
    __nv_bfloat16* sd   = so + TILE_Q*D;
    __nv_bfloat16* sk   = sd + TILE_Q*D;
    __nv_bfloat16* sv   = sk + TILE_K*D;
    float* acc          = reinterpret_cast<float*>(sv + TILE_K*D);

    int tid    = threadIdx.x;
    int64_t bh = blockIdx.z;
    int64_t kt = blockIdx.x;
    int64_t ks = kt * TILE_K;
    int64_t ke = min(ks+TILE_K, S);
    int64_t nk = ke - ks;
    int64_t boff = bh * S * D;
    int64_t nqtil = (S + TILE_Q - 1) / TILE_Q;

    for (int64_t i=tid;i<nk*D;i+=NT) acc[i]=0.0f;
    __syncthreads();

    for (int64_t qt=0;qt<nqtil;qt++) {
        int64_t qs_q=qt*TILE_Q; int64_t qe_q=min(qs_q+TILE_Q,S); int64_t nqq=qe_q-qs_q;
        if(nqq==0) continue;
        for (int64_t i=tid;i<nqq*D;i+=NT) {
            int64_t r=i/D,c=i%D; int64_t g=boff+(qs_q+r)*D+c;
            sq[i]=Q[g]; so[i]=O_fwd[g]; sd[i]=dO_in[g];
        }
        __syncthreads();
        for (int64_t i=tid;i<nk*D;i+=NT) {
            int64_t r=i/D,c=i%D; int64_t g=boff+(ks+r)*D+c;
            sk[i]=K[g]; sv[i]=V[g];
        }
        __syncthreads();
        float dQ_r[TILE_Q];
        for (int64_t s=tid;s<nqq;s+=NT) {
            float v=0.f;
            for(int d=0;d<D;d++) v+=__bfloat162float(sd[s*D+d])*__bfloat162float(so[s*D+d]);
            dQ_r[s]=v;
        }
        __syncthreads();
        #pragma unroll 4
        for(int p=0;p<NPAIRS;p++){
            int64_t gp=tid*NPAIRS+p;if(gp>=nqq*nk)break;
            int64_t sg=gp/TILE_K,gk=gp%TILE_K;
            float sc=0.f;
            for(int dd=0;dd<D;dd++) sc+=__bfloat162float(sq[sg*D+dd])*__bfloat162float(sk[gk*D+dd]);
            sc*=inv_sqrt_d;
            float pv=expf(sc-LSE[bh*S+qs_q+sg]);
            float da=0.f;
            for(int dd=0;dd<D;dd++) da+=__bfloat162float(sd[sg*D+dd])*__bfloat162float(sv[gk*D+dd]);
            float ds=pv*(da-dQ_r[sg])*inv_sqrt_d;
            (void)ds;
            for(int dd=0;dd<D;dd++) acc[gk*D+dd]+=pv*__bfloat162float(sd[sg*D+dd]);
        }
        __syncthreads();
    }

    for (int64_t i=tid;i<nk*D;i+=NT) {
        int64_t r=i/D,c=i%D;
        dV_out[boff+(ks+r)*D+c]=__float2bfloat16(acc[i]);
    }
}

namespace mha_bwd_impl {

void run(
    tvm::ffi::TensorView Q,
    tvm::ffi::TensorView K,
    tvm::ffi::TensorView V,
    tvm::ffi::TensorView O_fwd,
    tvm::ffi::TensorView dO_in,
    tvm::ffi::TensorView L,
    tvm::ffi::TensorView dQ_out,
    tvm::ffi::TensorView dK_out,
    tvm::ffi::TensorView dV_out)
{
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int64_t B=Q.size(0),H=Q.size(1),S=Q.size(2),D=Q.size(3);
    cudaStream_t stream=static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id));

    size_t out_bytes=B*H*S*D*sizeof(__nv_bfloat16);
    CUDA_CHECK(cudaMemsetAsync(dQ_out.data_ptr(),0,out_bytes,stream));
    CUDA_CHECK(cudaMemsetAsync(dK_out.data_ptr(),0,out_bytes,stream));
    CUDA_CHECK(cudaMemsetAsync(dV_out.data_ptr(),0,out_bytes,stream));

    float invsd=1.0f/sqrtf(static_cast<float>(D));
    dim3 blk(NT);
    size_t smem=(3LL*TILE_Q+2LL*TILE_K)*D*sizeof(__nv_bfloat16)+TILE_K*D*sizeof(float);

    {
        dim3 grd((S+TILE_Q-1)/TILE_Q,1,B*H);
        mha_bwd_dq<<<grd,blk,smem,stream>>>(
            static_cast<const __nv_bfloat16*>(Q.data_ptr()),
            static_cast<const __nv_bfloat16*>(K.data_ptr()),
            static_cast<const __nv_bfloat16*>(V.data_ptr()),
            static_cast<const __nv_bfloat16*>(O_fwd.data_ptr()),
            static_cast<const __nv_bfloat16*>(dO_in.data_ptr()),
            static_cast<const float*>(L.data_ptr()),
            static_cast<__nv_bfloat16*>(dQ_out.data_ptr()),
            B,H,S,D,invsd);
        CUDA_CHECK(cudaGetLastError());
    }
    {
        dim3 grd((S+TILE_K-1)/TILE_K,1,B*H);
        mha_bwd_dk<<<grd,blk,smem,stream>>>(
            static_cast<const __nv_bfloat16*>(Q.data_ptr()),
            static_cast<const __nv_bfloat16*>(K.data_ptr()),
            static_cast<const __nv_bfloat16*>(V.data_ptr()),
            static_cast<const __nv_bfloat16*>(O_fwd.data_ptr()),
            static_cast<const __nv_bfloat16*>(dO_in.data_ptr()),
            static_cast<const float*>(L.data_ptr()),
            static_cast<__nv_bfloat16*>(dK_out.data_ptr()),
            B,H,S,D,invsd);
        CUDA_CHECK(cudaGetLastError());
    }
    {
        dim3 grd((S+TILE_K-1)/TILE_K,1,B*H);
        mha_bwd_dv<<<grd,blk,smem,stream>>>(
            static_cast<const __nv_bfloat16*>(Q.data_ptr()),
            static_cast<const __nv_bfloat16*>(K.data_ptr()),
            static_cast<const __nv_bfloat16*>(V.data_ptr()),
            static_cast<const __nv_bfloat16*>(O_fwd.data_ptr()),
            static_cast<const __nv_bfloat16*>(dO_in.data_ptr()),
            static_cast<const float*>(L.data_ptr()),
            static_cast<__nv_bfloat16*>(dV_out.data_ptr()),
            B,H,S,D,invsd);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_impl::run);

}