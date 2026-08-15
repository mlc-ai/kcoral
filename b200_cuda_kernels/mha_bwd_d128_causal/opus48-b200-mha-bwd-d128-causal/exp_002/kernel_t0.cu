#include <cuda_bf16.h>
#include <cuda_runtime.h>
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

template<int ROWS,int HD>
__device__ __forceinline__ void load_tile(bf16* smem, const bf16* g, int valid_rows, int tid, int nthreads){
    const int VEC=8; // 8 bf16 per int4
    const int per_row = HD/VEC;
    int total = ROWS*per_row;
    for(int vi=tid; vi<total; vi+=nthreads){
        int row = vi/per_row;
        int cg  = vi%per_row;
        int4 val;
        if(row<valid_rows) val = *reinterpret_cast<const int4*>(g + (size_t)row*HD + cg*VEC);
        else val = make_int4(0,0,0,0);
        *reinterpret_cast<int4*>(smem + (size_t)row*HD + cg*VEC) = val;
    }
}

// D_i = sum_k dO_ik * O_ik  (one warp per row)
__global__ void compute_D(const bf16* O, const bf16* dO, float* Dout, int total_rows, int HD){
    int gwarp = (blockIdx.x*blockDim.x + threadIdx.x)/32;
    int lane = threadIdx.x & 31;
    if(gwarp>=total_rows) return;
    const bf16* o = O + (size_t)gwarp*HD;
    const bf16* g = dO + (size_t)gwarp*HD;
    float acc=0.f;
    for(int k=lane;k<HD;k+=32) acc += __bfloat162float(o[k])*__bfloat162float(g[k]);
    #pragma unroll
    for(int off=16;off>0;off>>=1) acc += __shfl_down_sync(0xffffffff,acc,off);
    if(lane==0) Dout[gwarp]=acc;
}

// dK/dV kernel: block handles one KV tile, loops over Q tiles
template<int BQ,int BN,int HD>
__global__ void bwd_dkdv(const bf16* Q,const bf16* K,const bf16* V,const bf16* dO,
                         const float* Lg,const float* Dg,
                         bf16* dKout,bf16* dVout,int S,float scale){
    int bh = blockIdx.y;
    int kv_start = blockIdx.x*BN;
    if(kv_start>=S) return;
    int krows = min(BN, S-kv_start);
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    extern __shared__ char smem[];
    bf16* Qs = (bf16*)smem;
    bf16* dOs = Qs + BQ*HD;
    bf16* Ks = dOs + BQ*HD;
    bf16* Vs = Ks + BN*HD;
    float* Ps = (float*)(Vs + BN*HD);
    float* dSs = Ps + BQ*BN;
    float* dVacc = dSs + BQ*BN;
    float* dKacc = dVacc + BN*HD;
    float* Ls = dKacc + BN*HD;
    float* Ds = Ls + BQ;

    size_t bh_off = (size_t)bh*S*HD;
    size_t bh_l   = (size_t)bh*S;
    load_tile<BN,HD>(Ks, K + bh_off + (size_t)kv_start*HD, krows, tid, nthreads);
    load_tile<BN,HD>(Vs, V + bh_off + (size_t)kv_start*HD, krows, tid, nthreads);
    for(int f=tid; f<BN*HD; f+=nthreads){ dVacc[f]=0.f; dKacc[f]=0.f; }
    __syncthreads();

    for(int q_start=kv_start; q_start<S; q_start+=BQ){
        int qrows = min(BQ, S-q_start);
        load_tile<BQ,HD>(Qs, Q + bh_off + (size_t)q_start*HD, qrows, tid, nthreads);
        load_tile<BQ,HD>(dOs, dO + bh_off + (size_t)q_start*HD, qrows, tid, nthreads);
        for(int i=tid;i<BQ;i+=nthreads){
            int gr=q_start+i;
            Ls[i] = (i<qrows)?Lg[bh_l+gr]:0.f;
            Ds[i] = (i<qrows)?Dg[bh_l+gr]:0.f;
        }
        __syncthreads();

        // Phase A: P and dS
        for(int e=tid; e<BQ*BN; e+=nthreads){
            int i=e/BN, j=e%BN;
            float p=0.f, ds=0.f;
            int gq=q_start+i, gk=kv_start+j;
            if(i<qrows && j<krows && gq>=gk){
                const __nv_bfloat162* Q2=(const __nv_bfloat162*)(Qs+i*HD);
                const __nv_bfloat162* K2=(const __nv_bfloat162*)(Ks+j*HD);
                const __nv_bfloat162* O2=(const __nv_bfloat162*)(dOs+i*HD);
                const __nv_bfloat162* V2=(const __nv_bfloat162*)(Vs+j*HD);
                float s=0.f, dp=0.f;
                #pragma unroll
                for(int t=0;t<HD/2;t++){
                    float2 qf=__bfloat1622float2(Q2[t]);
                    float2 kf=__bfloat1622float2(K2[t]);
                    float2 of=__bfloat1622float2(O2[t]);
                    float2 vf=__bfloat1622float2(V2[t]);
                    s += qf.x*kf.x + qf.y*kf.y;
                    dp += of.x*vf.x + of.y*vf.y;
                }
                p = expf(scale*s - Ls[i]);
                ds = p*(dp - Ds[i]);
            }
            Ps[i*BN+j]=p; dSs[i*BN+j]=ds;
        }
        __syncthreads();

        // Phase B: accumulate dV, dK
        for(int f=tid; f<BN*HD; f+=nthreads){
            int j=f/HD, k=f%HD;
            float accV=0.f, accK=0.f;
            #pragma unroll 4
            for(int i=0;i<BQ;i++){
                float pv=Ps[i*BN+j];
                float dsv=dSs[i*BN+j];
                accV += pv*__bfloat162float(dOs[i*HD+k]);
                accK += dsv*__bfloat162float(Qs[i*HD+k]);
            }
            dVacc[f]+=accV; dKacc[f]+=accK;
        }
        __syncthreads();
    }

    for(int f=tid; f<BN*HD; f+=nthreads){
        int j=f/HD, k=f%HD;
        if(j<krows){
            size_t idx = bh_off + (size_t)(kv_start+j)*HD + k;
            dVout[idx]=__float2bfloat16(dVacc[f]);
            dKout[idx]=__float2bfloat16(scale*dKacc[f]);
        }
    }
}

// dQ kernel: block handles one Q tile, loops over KV tiles
template<int BQ,int BN,int HD>
__global__ void bwd_dq(const bf16* Q,const bf16* K,const bf16* V,const bf16* dO,
                       const float* Lg,const float* Dg,
                       bf16* dQout,int S,float scale){
    int bh = blockIdx.y;
    int q_start = blockIdx.x*BQ;
    if(q_start>=S) return;
    int qrows = min(BQ, S-q_start);
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    extern __shared__ char smem[];
    bf16* Qs = (bf16*)smem;
    bf16* dOs = Qs + BQ*HD;
    bf16* Ks = dOs + BQ*HD;
    bf16* Vs = Ks + BN*HD;
    float* dSs = (float*)(Vs + BN*HD);
    float* dQacc = dSs + BQ*BN;
    float* Ls = dQacc + BQ*HD;
    float* Ds = Ls + BQ;

    size_t bh_off = (size_t)bh*S*HD;
    size_t bh_l   = (size_t)bh*S;
    load_tile<BQ,HD>(Qs, Q + bh_off + (size_t)q_start*HD, qrows, tid, nthreads);
    load_tile<BQ,HD>(dOs, dO + bh_off + (size_t)q_start*HD, qrows, tid, nthreads);
    for(int i=tid;i<BQ;i+=nthreads){
        int gr=q_start+i;
        Ls[i] = (i<qrows)?Lg[bh_l+gr]:0.f;
        Ds[i] = (i<qrows)?Dg[bh_l+gr]:0.f;
    }
    for(int f=tid; f<BQ*HD; f+=nthreads) dQacc[f]=0.f;
    __syncthreads();

    int kv_end = min(S, q_start+BQ);
    for(int kv_start=0; kv_start<kv_end; kv_start+=BN){
        int krows = min(BN, S-kv_start);
        load_tile<BN,HD>(Ks, K + bh_off + (size_t)kv_start*HD, krows, tid, nthreads);
        load_tile<BN,HD>(Vs, V + bh_off + (size_t)kv_start*HD, krows, tid, nthreads);
        __syncthreads();

        // Phase A: dS
        for(int e=tid; e<BQ*BN; e+=nthreads){
            int i=e/BN, j=e%BN;
            float ds=0.f;
            int gq=q_start+i, gk=kv_start+j;
            if(i<qrows && j<krows && gq>=gk){
                const __nv_bfloat162* Q2=(const __nv_bfloat162*)(Qs+i*HD);
                const __nv_bfloat162* K2=(const __nv_bfloat162*)(Ks+j*HD);
                const __nv_bfloat162* O2=(const __nv_bfloat162*)(dOs+i*HD);
                const __nv_bfloat162* V2=(const __nv_bfloat162*)(Vs+j*HD);
                float s=0.f, dp=0.f;
                #pragma unroll
                for(int t=0;t<HD/2;t++){
                    float2 qf=__bfloat1622float2(Q2[t]);
                    float2 kf=__bfloat1622float2(K2[t]);
                    float2 of=__bfloat1622float2(O2[t]);
                    float2 vf=__bfloat1622float2(V2[t]);
                    s += qf.x*kf.x + qf.y*kf.y;
                    dp += of.x*vf.x + of.y*vf.y;
                }
                float p = expf(scale*s - Ls[i]);
                ds = p*(dp - Ds[i]);
            }
            dSs[i*BN+j]=ds;
        }
        __syncthreads();

        // Phase B: dQ_i += sum_j dS_ij K_j
        for(int f=tid; f<BQ*HD; f+=nthreads){
            int i=f/HD, k=f%HD;
            float acc=0.f;
            #pragma unroll 4
            for(int j=0;j<BN;j++){
                acc += dSs[i*BN+j]*__bfloat162float(Ks[j*HD+k]);
            }
            dQacc[f]+=acc;
        }
        __syncthreads();
    }

    for(int f=tid; f<BQ*HD; f+=nthreads){
        int i=f/HD, k=f%HD;
        if(i<qrows){
            size_t idx = bh_off + (size_t)(q_start+i)*HD + k;
            dQout[idx]=__float2bfloat16(scale*dQacc[f]);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV){
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = Q.size(3);

    const bf16* Qp = static_cast<const bf16*>(Q.data_ptr());
    const bf16* Kp = static_cast<const bf16*>(K.data_ptr());
    const bf16* Vp = static_cast<const bf16*>(V.data_ptr());
    const bf16* Op = static_cast<const bf16*>(O.data_ptr());
    const bf16* dOp = static_cast<const bf16*>(dO.data_ptr());
    const float* Lp = static_cast<const float*>(L.data_ptr());
    bf16* dQp = static_cast<bf16*>(dQ.data_ptr());
    bf16* dKp = static_cast<bf16*>(dK.data_ptr());
    bf16* dVp = static_cast<bf16*>(dV.data_ptr());

    float scale = 1.0f / sqrtf((float)d);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    int BH = (int)(B*H);
    int Si = (int)S;
    int HD = (int)d; // 128

    // Precompute D
    float* Dscratch = nullptr;
    size_t total_rows = (size_t)BH*Si;
    CUDA_CHECK(cudaMallocAsync(&Dscratch, sizeof(float)*total_rows, stream));
    {
        int threads=256;
        int warps_per_block = threads/32;
        int blocks = (int)((total_rows + warps_per_block - 1)/warps_per_block);
        compute_D<<<blocks, threads, 0, stream>>>(Op, dOp, Dscratch, (int)total_rows, HD);
        CUDA_CHECK(cudaGetLastError());
    }

    const int BQ=64, BN=64, HDc=128;

    // dK/dV kernel
    {
        size_t sh2 = (size_t)( (2*BQ*HDc + 2*BN*HDc) )*sizeof(bf16)
                   + (size_t)( (2*BQ*BN + 2*BN*HDc + 2*BQ) )*sizeof(float);
        auto k2 = bwd_dkdv<BQ,BN,HDc>;
        CUDA_CHECK(cudaFuncSetAttribute((const void*)k2,
                   cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh2));
        int nkv = (Si + BN - 1)/BN;
        dim3 grid(nkv, BH);
        k2<<<grid, 256, sh2, stream>>>(Qp,Kp,Vp,dOp,Lp,Dscratch,dKp,dVp,Si,scale);
        CUDA_CHECK(cudaGetLastError());
    }

    // dQ kernel
    {
        size_t sh3 = (size_t)( (2*BQ*HDc + 2*BN*HDc) )*sizeof(bf16)
                   + (size_t)( (BQ*BN + BQ*HDc + 2*BQ) )*sizeof(float);
        auto k3 = bwd_dq<BQ,BN,HDc>;
        CUDA_CHECK(cudaFuncSetAttribute((const void*)k3,
                   cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh3));
        int nq = (Si + BQ - 1)/BQ;
        dim3 grid(nq, BH);
        k3<<<grid, 256, sh3, stream>>>(Qp,Kp,Vp,dOp,Lp,Dscratch,dQp,Si,scale);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaFreeAsync(Dscratch, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

} // namespace mha_bwd