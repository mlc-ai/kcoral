#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__); exit(1);} } while(0)

namespace mha_bwd {

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int Dh = 128;
constexpr int THREADS = 256;
constexpr int REGS = (BM*Dh)/THREADS; // = 32

// ---- Delta = rowsum(dO * O) ----
__global__ void compute_delta_kernel(const __nv_bfloat16* __restrict__ dO,
                                     const __nv_bfloat16* __restrict__ O,
                                     float* __restrict__ Delta, long total_rows){
    int wpb = blockDim.x/32;
    long row = (long)blockIdx.x * wpb + threadIdx.x/32;
    int lane = threadIdx.x & 31;
    if(row >= total_rows) return;
    const __nv_bfloat16* dop = dO + row*Dh;
    const __nv_bfloat16* op  = O  + row*Dh;
    float sum=0.f;
    for(int k=lane;k<Dh;k+=32)
        sum += __bfloat162float(dop[k]) * __bfloat162float(op[k]);
    for(int off=16;off>0;off>>=1) sum += __shfl_down_sync(0xffffffffu,sum,off);
    if(lane==0) Delta[row]=sum;
}

// ---- dK, dV : grid(num_kv, BH) ----
__global__ void dkdv_kernel(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
                            const __nv_bfloat16* __restrict__ V, const __nv_bfloat16* __restrict__ dO,
                            const float* __restrict__ L, const float* __restrict__ Delta,
                            __nv_bfloat16* __restrict__ dK, __nv_bfloat16* __restrict__ dV,
                            int S, float scale){
    extern __shared__ char smem[];
    __nv_bfloat16* sQ  = (__nv_bfloat16*)smem;
    __nv_bfloat16* sdO = sQ  + BM*Dh;
    __nv_bfloat16* sK  = sdO + BM*Dh;
    __nv_bfloat16* sV  = sK  + BN*Dh;
    float* sP  = (float*)(sV + BN*Dh);
    float* sdS = sP  + BM*BN;
    float* sL  = sdS + BM*BN;
    float* sD  = sL  + BM;

    int bh = blockIdx.y;
    int kv_start = blockIdx.x * BN;
    if(kv_start >= S) return;
    long base = (long)bh * S * Dh;
    int tid = threadIdx.x;

    for(int e=tid; e<BN*Dh; e+=THREADS){
        int n=e/Dh, k=e%Dh; int krow=kv_start+n;
        if(krow<S){ sK[e]=K[base+(long)krow*Dh+k]; sV[e]=V[base+(long)krow*Dh+k]; }
        else { sK[e]=__float2bfloat16(0.f); sV[e]=__float2bfloat16(0.f); }
    }

    float dVacc[REGS], dKacc[REGS];
    #pragma unroll
    for(int r=0;r<REGS;r++){ dVacc[r]=0.f; dKacc[r]=0.f; }
    __syncthreads();

    int q_start0 = (kv_start / BM) * BM;
    for(int q_start=q_start0; q_start<S; q_start+=BM){
        for(int e=tid;e<BM*Dh;e+=THREADS){
            int m=e/Dh,k=e%Dh; int qrow=q_start+m;
            if(qrow<S){ sQ[e]=Q[base+(long)qrow*Dh+k]; sdO[e]=dO[base+(long)qrow*Dh+k]; }
            else { sQ[e]=__float2bfloat16(0.f); sdO[e]=__float2bfloat16(0.f); }
        }
        for(int m=tid;m<BM;m+=THREADS){
            int qrow=q_start+m;
            if(qrow<S){ sL[m]=L[(long)bh*S+qrow]; sD[m]=Delta[(long)bh*S+qrow]; }
            else { sL[m]=0.f; sD[m]=0.f; }
        }
        __syncthreads();

        // Phase A: P = exp(scale*Q@K^T - L)
        for(int e=tid;e<BM*BN;e+=THREADS){
            int m=e/BN,n=e%BN; int qrow=q_start+m,krow=kv_start+n;
            float p=0.f;
            if(qrow<S && krow<S && krow<=qrow){
                float dot=0.f;
                for(int k=0;k<Dh;k++) dot += __bfloat162float(sQ[m*Dh+k])*__bfloat162float(sK[n*Dh+k]);
                p = __expf(dot*scale - sL[m]);
            }
            sP[e]=p;
        }
        __syncthreads();

        // Phase B: dS = scale * P*(dP - D)
        for(int e=tid;e<BM*BN;e+=THREADS){
            int m=e/BN,n=e%BN; int qrow=q_start+m,krow=kv_start+n;
            float ds=0.f;
            if(qrow<S && krow<S && krow<=qrow){
                float dp=0.f;
                for(int k=0;k<Dh;k++) dp += __bfloat162float(sdO[m*Dh+k])*__bfloat162float(sV[n*Dh+k]);
                ds = sP[e]*(dp - sD[m])*scale;
            }
            sdS[e]=ds;
        }
        __syncthreads();

        // Phase C/D: dV += P^T@dO ; dK += dS^T@Q
        #pragma unroll
        for(int r=0;r<REGS;r++){
            int idx = tid + THREADS*r;
            int n=idx/Dh, k=idx%Dh;
            float av=0.f, ak=0.f;
            for(int m=0;m<BM;m++){
                av += sP[m*BN+n]  * __bfloat162float(sdO[m*Dh+k]);
                ak += sdS[m*BN+n] * __bfloat162float(sQ[m*Dh+k]);
            }
            dVacc[r]+=av; dKacc[r]+=ak;
        }
        __syncthreads();
    }

    #pragma unroll
    for(int r=0;r<REGS;r++){
        int idx = tid + THREADS*r;
        int n=idx/Dh, k=idx%Dh; int krow=kv_start+n;
        if(krow<S){
            dV[base+(long)krow*Dh+k]=__float2bfloat16(dVacc[r]);
            dK[base+(long)krow*Dh+k]=__float2bfloat16(dKacc[r]);
        }
    }
}

// ---- dQ : grid(num_q, BH) ----
__global__ void dq_kernel(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
                          const __nv_bfloat16* __restrict__ V, const __nv_bfloat16* __restrict__ dO,
                          const float* __restrict__ L, const float* __restrict__ Delta,
                          __nv_bfloat16* __restrict__ dQ, int S, float scale){
    extern __shared__ char smem[];
    __nv_bfloat16* sQ  = (__nv_bfloat16*)smem;
    __nv_bfloat16* sdO = sQ  + BM*Dh;
    __nv_bfloat16* sK  = sdO + BM*Dh;
    __nv_bfloat16* sV  = sK  + BN*Dh;
    float* sP  = (float*)(sV + BN*Dh);
    float* sdS = sP  + BM*BN;
    float* sL  = sdS + BM*BN;
    float* sD  = sL  + BM;

    int bh = blockIdx.y;
    int q_start = blockIdx.x * BM;
    if(q_start >= S) return;
    long base = (long)bh * S * Dh;
    int tid = threadIdx.x;

    for(int e=tid;e<BM*Dh;e+=THREADS){
        int m=e/Dh,k=e%Dh; int qrow=q_start+m;
        if(qrow<S){ sQ[e]=Q[base+(long)qrow*Dh+k]; sdO[e]=dO[base+(long)qrow*Dh+k]; }
        else { sQ[e]=__float2bfloat16(0.f); sdO[e]=__float2bfloat16(0.f); }
    }
    for(int m=tid;m<BM;m+=THREADS){
        int qrow=q_start+m;
        if(qrow<S){ sL[m]=L[(long)bh*S+qrow]; sD[m]=Delta[(long)bh*S+qrow]; }
        else { sL[m]=0.f; sD[m]=0.f; }
    }

    float dQacc[REGS];
    #pragma unroll
    for(int r=0;r<REGS;r++) dQacc[r]=0.f;
    __syncthreads();

    int q_end = q_start + BM - 1;
    for(int kv_start=0; kv_start<=q_end && kv_start<S; kv_start+=BN){
        for(int e=tid;e<BN*Dh;e+=THREADS){
            int n=e/Dh,k=e%Dh; int krow=kv_start+n;
            if(krow<S){ sK[e]=K[base+(long)krow*Dh+k]; sV[e]=V[base+(long)krow*Dh+k]; }
            else { sK[e]=__float2bfloat16(0.f); sV[e]=__float2bfloat16(0.f); }
        }
        __syncthreads();

        for(int e=tid;e<BM*BN;e+=THREADS){
            int m=e/BN,n=e%BN; int qrow=q_start+m,krow=kv_start+n;
            float p=0.f;
            if(qrow<S && krow<S && krow<=qrow){
                float dot=0.f;
                for(int k=0;k<Dh;k++) dot += __bfloat162float(sQ[m*Dh+k])*__bfloat162float(sK[n*Dh+k]);
                p = __expf(dot*scale - sL[m]);
            }
            sP[e]=p;
        }
        __syncthreads();

        for(int e=tid;e<BM*BN;e+=THREADS){
            int m=e/BN,n=e%BN; int qrow=q_start+m,krow=kv_start+n;
            float ds=0.f;
            if(qrow<S && krow<S && krow<=qrow){
                float dp=0.f;
                for(int k=0;k<Dh;k++) dp += __bfloat162float(sdO[m*Dh+k])*__bfloat162float(sV[n*Dh+k]);
                ds = sP[e]*(dp - sD[m])*scale;
            }
            sdS[e]=ds;
        }
        __syncthreads();

        // dQ += dS@K
        #pragma unroll
        for(int r=0;r<REGS;r++){
            int idx = tid + THREADS*r;
            int m=idx/Dh, k=idx%Dh;
            float a=0.f;
            for(int n=0;n<BN;n++)
                a += sdS[m*BN+n]*__bfloat162float(sK[n*Dh+k]);
            dQacc[r]+=a;
        }
        __syncthreads();
    }

    #pragma unroll
    for(int r=0;r<REGS;r++){
        int idx = tid + THREADS*r;
        int m=idx/Dh, k=idx%Dh; int qrow=q_start+m;
        if(qrow<S) dQ[base+(long)qrow*Dh+k]=__float2bfloat16(dQacc[r]);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV){
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int Bsz = Q.size(0), H = Q.size(1), S = Q.size(2), d = Q.size(3);
    int BH = Bsz*H;
    float scale = 1.0f/sqrtf((float)d);

    const __nv_bfloat16* Qp =(const __nv_bfloat16*)Q.data_ptr();
    const __nv_bfloat16* Kp =(const __nv_bfloat16*)K.data_ptr();
    const __nv_bfloat16* Vp =(const __nv_bfloat16*)V.data_ptr();
    const __nv_bfloat16* Op =(const __nv_bfloat16*)O.data_ptr();
    const __nv_bfloat16* dOp=(const __nv_bfloat16*)dO.data_ptr();
    const float* Lp =(const float*)L.data_ptr();
    __nv_bfloat16* dQp=(__nv_bfloat16*)dQ.data_ptr();
    __nv_bfloat16* dKp=(__nv_bfloat16*)dK.data_ptr();
    __nv_bfloat16* dVp=(__nv_bfloat16*)dV.data_ptr();

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    float* Delta=nullptr;
    CUDA_CHECK(cudaMalloc(&Delta, (size_t)BH*S*sizeof(float)));

    long total_rows = (long)BH*S;
    int wpb = 8;
    long blocks_delta = (total_rows + wpb - 1)/wpb;
    compute_delta_kernel<<<(unsigned)blocks_delta, wpb*32, 0, stream>>>(dOp, Op, Delta, total_rows);

    size_t smem = (size_t)(2*BM*Dh + 2*BN*Dh)*sizeof(__nv_bfloat16)
                + (size_t)(2*BM*BN)*sizeof(float)
                + (size_t)(2*BM)*sizeof(float);

    CUDA_CHECK(cudaFuncSetAttribute(dkdv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    CUDA_CHECK(cudaFuncSetAttribute(dq_kernel,   cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));

    int num_kv = (S+BN-1)/BN;
    int num_q  = (S+BM-1)/BM;
    dim3 grid_kv(num_kv, BH);
    dim3 grid_q (num_q,  BH);

    dkdv_kernel<<<grid_kv, THREADS, smem, stream>>>(Qp,Kp,Vp,dOp,Lp,Delta,dKp,dVp,S,scale);
    dq_kernel  <<<grid_q,  THREADS, smem, stream>>>(Qp,Kp,Vp,dOp,Lp,Delta,dQp,S,scale);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(Delta));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd