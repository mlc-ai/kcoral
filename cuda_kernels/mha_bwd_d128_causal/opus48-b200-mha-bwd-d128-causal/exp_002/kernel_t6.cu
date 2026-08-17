#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
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

namespace mha_bwd {
using bf16 = __nv_bfloat16;
namespace w = nvcuda::wmma;

static constexpr int BM = 64;
static constexpr int BN = 64;
static constexpr int HD = 128;
static constexpr int NTH = 256; // 8 warps

using FragAr = w::fragment<w::matrix_a,16,16,16,bf16,w::row_major>;
using FragAc = w::fragment<w::matrix_a,16,16,16,bf16,w::col_major>;
using FragBr = w::fragment<w::matrix_b,16,16,16,bf16,w::row_major>;
using FragBc = w::fragment<w::matrix_b,16,16,16,bf16,w::col_major>;
using FragC  = w::fragment<w::accumulator,16,16,16,float>;

template<int ROWS>
__device__ __forceinline__ void load_bf16(bf16* s, const bf16* g, int valid, int tid){
    const int perrow=HD/8; int total=ROWS*perrow;
    for(int v=tid; v<total; v+=NTH){
        int r=v/perrow, c=v%perrow;
        int4 val;
        if(r<valid) val=*reinterpret_cast<const int4*>(g+(size_t)r*HD+c*8);
        else val=make_int4(0,0,0,0);
        *reinterpret_cast<int4*>(s+(size_t)r*HD+c*8)=val;
    }
}

// D_i = sum_k dO_ik * O_ik (fp32, exact cuDNN-style delta)
__global__ void compute_D(const bf16* O, const bf16* dO, float* Dout, int total_rows){
    int gwarp=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    int lane=threadIdx.x&31;
    if(gwarp>=total_rows) return;
    const bf16* o=O+(size_t)gwarp*HD;
    const bf16* g=dO+(size_t)gwarp*HD;
    float acc=0.f;
    for(int k=lane;k<HD;k+=32) acc+=__bfloat162float(o[k])*__bfloat162float(g[k]);
    #pragma unroll
    for(int off=16;off>0;off>>=1) acc+=__shfl_down_sync(0xffffffff,acc,off);
    if(lane==0) Dout[gwarp]=acc;
}

// -------- dK/dV: block = one KV tile, loops over Q tiles --------
__launch_bounds__(NTH,1)
__global__ void bwd_dkdv(const bf16* Q,const bf16* K,const bf16* V,const bf16* dO,
                         const float* Lg,const float* Dg,
                         bf16* dKout,bf16* dVout,int S,float scale){
    int bh=blockIdx.y;
    int kv_start=blockIdx.x*BN;
    if(kv_start>=S) return;
    const int tid=threadIdx.x, warp=tid>>5;

    extern __shared__ char sm[];
    bf16* Ks =(bf16*)sm;
    bf16* Vs =Ks+BN*HD;
    bf16* Qs =Vs+BN*HD;
    bf16* dOs=Qs+BM*HD;
    float* Ls=(float*)(dOs+BM*HD);
    float* Ds=Ls+BM;
    float* Sf=Ds+BM;         // 64*64
    float* dPf=Sf+BM*BN;     // 64*64
    bf16* Pbf=(bf16*)(dPf+BM*BN); //64*64
    bf16* dSbf=Pbf+BM*BN;    //64*64
    float* outbuf=Sf;        // reuse (Sf+dPf = 8192 floats = 64*128)

    size_t bh_off=(size_t)bh*S*HD, bh_l=(size_t)bh*S;

    load_bf16<BN>(Ks, K+bh_off+(size_t)kv_start*HD, min(BN,S-kv_start), tid);
    load_bf16<BN>(Vs, V+bh_off+(size_t)kv_start*HD, min(BN,S-kv_start), tid);

    FragC accV[4], accK[4];
    #pragma unroll
    for(int m=0;m<4;m++){ w::fill_fragment(accV[m],0.f); w::fill_fragment(accK[m],0.f); }
    __syncthreads();

    for(int q_start=kv_start; q_start<S; q_start+=BM){
        int qrows=min(BM,S-q_start);
        load_bf16<BM>(Qs,  Q +bh_off+(size_t)q_start*HD, qrows, tid);
        load_bf16<BM>(dOs, dO+bh_off+(size_t)q_start*HD, qrows, tid);
        for(int i=tid;i<BM;i+=NTH){ int qg=q_start+i; Ls[i]=(qg<S)?Lg[bh_l+qg]:0.f; Ds[i]=(qg<S)?Dg[bh_l+qg]:0.f; }
        __syncthreads();

        // Phase A: S=Q@K^T, dP=dO@V^T  (16 tiles, 2 per warp)
        #pragma unroll
        for(int loc=0;loc<2;loc++){
            int t=warp*2+loc; int mt=t/4, nt=t%4;
            FragC accS, accP; w::fill_fragment(accS,0.f); w::fill_fragment(accP,0.f);
            #pragma unroll
            for(int kt=0;kt<HD/16;kt++){
                FragAr qf, of; FragBc kf, vf;
                w::load_matrix_sync(qf, Qs +mt*16*HD+kt*16, HD);
                w::load_matrix_sync(of, dOs+mt*16*HD+kt*16, HD);
                w::load_matrix_sync(kf, Ks +nt*16*HD+kt*16, HD);
                w::load_matrix_sync(vf, Vs +nt*16*HD+kt*16, HD);
                w::mma_sync(accS, qf, kf, accS);
                w::mma_sync(accP, of, vf, accP);
            }
            w::store_matrix_sync(Sf +mt*16*BN+nt*16, accS, BN, w::mem_row_major);
            w::store_matrix_sync(dPf+mt*16*BN+nt*16, accP, BN, w::mem_row_major);
        }
        __syncthreads();

        // elementwise: P, dS
        for(int idx=tid; idx<BM*BN; idx+=NTH){
            int i=idx/BN, j=idx%BN;
            int qg=q_start+i, kg=kv_start+j;
            float P=0.f, dS=0.f;
            if(qg<S && kg<S && qg>=kg){
                P=__expf(scale*Sf[idx]-Ls[i]);
                dS=P*(dPf[idx]-Ds[i]);
            }
            Pbf[idx]=__float2bfloat16(P);
            dSbf[idx]=__float2bfloat16(dS);
        }
        __syncthreads();

        // Phase B: dV += P^T@dO ; dK += dS^T@Q   (warp w -> Ntile w, 4 Mtiles)
        #pragma unroll
        for(int m=0;m<4;m++){
            #pragma unroll
            for(int kt=0;kt<BM/16;kt++){
                FragAc pf, sf; FragBr obf, qbf;
                w::load_matrix_sync(pf,  Pbf +kt*16*BN+m*16, BN);
                w::load_matrix_sync(sf,  dSbf+kt*16*BN+m*16, BN);
                w::load_matrix_sync(obf, dOs +kt*16*HD+warp*16, HD);
                w::load_matrix_sync(qbf, Qs  +kt*16*HD+warp*16, HD);
                w::mma_sync(accV[m], pf, obf, accV[m]);
                w::mma_sync(accK[m], sf, qbf, accK[m]);
            }
        }
        __syncthreads();
    }

    // write dV
    #pragma unroll
    for(int m=0;m<4;m++) w::store_matrix_sync(outbuf+m*16*HD+warp*16, accV[m], HD, w::mem_row_major);
    __syncthreads();
    for(int v=tid; v<BN*HD; v+=NTH){ int key=v/HD,e=v%HD; int kg=kv_start+key; if(kg<S) dVout[bh_off+(size_t)kg*HD+e]=__float2bfloat16(outbuf[v]); }
    __syncthreads();
    #pragma unroll
    for(int m=0;m<4;m++) w::store_matrix_sync(outbuf+m*16*HD+warp*16, accK[m], HD, w::mem_row_major);
    __syncthreads();
    for(int v=tid; v<BN*HD; v+=NTH){ int key=v/HD,e=v%HD; int kg=kv_start+key; if(kg<S) dKout[bh_off+(size_t)kg*HD+e]=__float2bfloat16(scale*outbuf[v]); }
}

// -------- dQ: block = one Q tile, loops over KV tiles --------
__launch_bounds__(NTH,1)
__global__ void bwd_dq(const bf16* Q,const bf16* K,const bf16* V,const bf16* dO,
                       const float* Lg,const float* Dg,
                       bf16* dQout,int S,float scale){
    int bh=blockIdx.y;
    int q_start=blockIdx.x*BM;
    if(q_start>=S) return;
    const int tid=threadIdx.x, warp=tid>>5;

    extern __shared__ char sm[];
    bf16* Qs =(bf16*)sm;
    bf16* dOs=Qs+BM*HD;
    bf16* Ks =dOs+BM*HD;
    bf16* Vs =Ks+BN*HD;
    float* Ls=(float*)(Vs+BN*HD);
    float* Ds=Ls+BM;
    float* Sf=Ds+BM;         // 64*64
    float* dPf=Sf+BM*BN;     // 64*64
    bf16* dSbf=(bf16*)(dPf+BM*BN); // 64*64
    float* outbuf=Sf;        // reuse Sf+dPf = 64*128

    size_t bh_off=(size_t)bh*S*HD, bh_l=(size_t)bh*S;

    int qrows=min(BM,S-q_start);
    load_bf16<BM>(Qs,  Q +bh_off+(size_t)q_start*HD, qrows, tid);
    load_bf16<BM>(dOs, dO+bh_off+(size_t)q_start*HD, qrows, tid);
    for(int i=tid;i<BM;i+=NTH){ int qg=q_start+i; Ls[i]=(qg<S)?Lg[bh_l+qg]:0.f; Ds[i]=(qg<S)?Dg[bh_l+qg]:0.f; }

    FragC accQ[4];
    #pragma unroll
    for(int m=0;m<4;m++) w::fill_fragment(accQ[m],0.f);
    __syncthreads();

    for(int kv_start=0; kv_start<=q_start && kv_start<S; kv_start+=BN){
        int krows=min(BN,S-kv_start);
        load_bf16<BN>(Ks, K+bh_off+(size_t)kv_start*HD, krows, tid);
        load_bf16<BN>(Vs, V+bh_off+(size_t)kv_start*HD, krows, tid);
        __syncthreads();

        // Phase A
        #pragma unroll
        for(int loc=0;loc<2;loc++){
            int t=warp*2+loc; int mt=t/4, nt=t%4;
            FragC accS, accP; w::fill_fragment(accS,0.f); w::fill_fragment(accP,0.f);
            #pragma unroll
            for(int kt=0;kt<HD/16;kt++){
                FragAr qf, of; FragBc kf, vf;
                w::load_matrix_sync(qf, Qs +mt*16*HD+kt*16, HD);
                w::load_matrix_sync(of, dOs+mt*16*HD+kt*16, HD);
                w::load_matrix_sync(kf, Ks +nt*16*HD+kt*16, HD);
                w::load_matrix_sync(vf, Vs +nt*16*HD+kt*16, HD);
                w::mma_sync(accS, qf, kf, accS);
                w::mma_sync(accP, of, vf, accP);
            }
            w::store_matrix_sync(Sf +mt*16*BN+nt*16, accS, BN, w::mem_row_major);
            w::store_matrix_sync(dPf+mt*16*BN+nt*16, accP, BN, w::mem_row_major);
        }
        __syncthreads();

        for(int idx=tid; idx<BM*BN; idx+=NTH){
            int i=idx/BN, j=idx%BN;
            int qg=q_start+i, kg=kv_start+j;
            float dS=0.f;
            if(qg<S && kg<S && qg>=kg){
                float P=__expf(scale*Sf[idx]-Ls[i]);
                dS=P*(dPf[idx]-Ds[i]);
            }
            dSbf[idx]=__float2bfloat16(dS);
        }
        __syncthreads();

        // Phase B: dQ += dS@K  (warp w -> Ntile w, 4 Mtiles; contract j over BN)
        #pragma unroll
        for(int m=0;m<4;m++){
            #pragma unroll
            for(int kt=0;kt<BN/16;kt++){
                FragAr sf; FragBr kf;
                w::load_matrix_sync(sf, dSbf+m*16*BN+kt*16, BN);
                w::load_matrix_sync(kf, Ks  +kt*16*HD+warp*16, HD);
                w::mma_sync(accQ[m], sf, kf, accQ[m]);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for(int m=0;m<4;m++) w::store_matrix_sync(outbuf+m*16*HD+warp*16, accQ[m], HD, w::mem_row_major);
    __syncthreads();
    for(int v=tid; v<BM*HD; v+=NTH){ int i=v/HD,e=v%HD; int qg=q_start+i; if(qg<S) dQout[bh_off+(size_t)qg*HD+e]=__float2bfloat16(scale*outbuf[v]); }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV){
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int64_t B=Q.size(0), H=Q.size(1), S=Q.size(2), d=Q.size(3);

    const bf16* Qp=static_cast<const bf16*>(Q.data_ptr());
    const bf16* Kp=static_cast<const bf16*>(K.data_ptr());
    const bf16* Vp=static_cast<const bf16*>(V.data_ptr());
    const bf16* Op=static_cast<const bf16*>(O.data_ptr());
    const bf16* dOp=static_cast<const bf16*>(dO.data_ptr());
    const float* Lp=static_cast<const float*>(L.data_ptr());
    bf16* dQp=static_cast<bf16*>(dQ.data_ptr());
    bf16* dKp=static_cast<bf16*>(dK.data_ptr());
    bf16* dVp=static_cast<bf16*>(dV.data_ptr());

    float scale=1.0f/sqrtf((float)d);
    cudaStream_t stream=static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    int BH=(int)(B*H), Si=(int)S;
    float* Dscratch=nullptr;
    size_t total_rows=(size_t)BH*Si;
    CUDA_CHECK(cudaMallocAsync(&Dscratch, sizeof(float)*total_rows, stream));

    size_t sh_dkdv=(size_t)(4*BM*HD + 2*BM*BN)*sizeof(bf16) + (size_t)(2*BM + 2*BM*BN)*sizeof(float);
    size_t sh_dq  =(size_t)(4*BM*HD + 1*BM*BN)*sizeof(bf16) + (size_t)(2*BM + 2*BM*BN)*sizeof(float);

    CUDA_CHECK(cudaFuncSetAttribute((const void*)bwd_dkdv,
               cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sh_dkdv));
    CUDA_CHECK(cudaFuncSetAttribute((const void*)bwd_dq,
               cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sh_dq));

    {
        int threads=256, wpb=threads/32;
        int blocks=(int)((total_rows+wpb-1)/wpb);
        compute_D<<<blocks,threads,0,stream>>>(Op,dOp,Dscratch,(int)total_rows);
        CUDA_CHECK(cudaGetLastError());
    }
    {
        int nkv=(Si+BN-1)/BN;
        dim3 grid(nkv,BH);
        bwd_dkdv<<<grid,NTH,sh_dkdv,stream>>>(Qp,Kp,Vp,dOp,Lp,Dscratch,dKp,dVp,Si,scale);
        CUDA_CHECK(cudaGetLastError());
    }
    {
        int nq=(Si+BM-1)/BM;
        dim3 grid(nq,BH);
        bwd_dq<<<grid,NTH,sh_dq,stream>>>(Qp,Kp,Vp,dOp,Lp,Dscratch,dQp,Si,scale);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaFreeAsync(Dscratch,stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

} // namespace mha_bwd