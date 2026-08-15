#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); exit(1);} } while(0)

namespace mha_bwd {

using tvm::ffi::TensorView;

constexpr int BQ = 64;
constexpr int BK = 64;
constexpr int Dh = 128;

__global__ void compute_D_kernel(const __nv_bfloat16* __restrict__ dO,
                                 const __nv_bfloat16* __restrict__ O,
                                 float* __restrict__ Dout, size_t nrows) {
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; row < nrows; row += stride) {
        const __nv_bfloat16* dop = dO + row*Dh;
        const __nv_bfloat16* op  = O  + row*Dh;
        float acc = 0.f;
        #pragma unroll
        for (int e=0;e<Dh;e++) acc += __bfloat162float(dop[e])*__bfloat162float(op[e]);
        Dout[row] = acc;
    }
}

__global__ void convert_kernel(const float* __restrict__ src, __nv_bfloat16* __restrict__ dst, size_t n) {
    size_t idx = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x*blockDim.x;
    for (; idx<n; idx+=stride) dst[idx] = __float2bfloat16(src[idx]);
}

__global__ void bwd_kernel(
    const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V, const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L, const float* __restrict__ Dvec,
    __nv_bfloat16* __restrict__ dK, __nv_bfloat16* __restrict__ dV, float* __restrict__ dQf,
    int B, int H, int S, float scale)
{
    int kb = blockIdx.x;
    int h  = blockIdx.y;
    int b  = blockIdx.z;
    int k0 = kb*BK;
    int nq = (S + BQ - 1)/BQ;

    extern __shared__ char smem[];
    __nv_bfloat16* Ks  = (__nv_bfloat16*)smem;
    __nv_bfloat16* Vs  = Ks + BK*Dh;
    __nv_bfloat16* Qs  = Vs + BK*Dh;
    __nv_bfloat16* dOs = Qs + BQ*Dh;
    float* Pbuf = (float*)(dOs + BQ*Dh);
    float* Sbuf = Pbuf + BQ*BK;
    float* Ls   = Sbuf + BQ*BK;
    float* Ds   = Ls + BQ;

    int tid = threadIdx.x;
    int tr = tid & 15;      // 0..15
    int tc = tid >> 4;      // 0..15
    int r0 = tr*4;          // row base (4 rows)
    int c0 = tc*4;          // 64-col base (4 cols)
    int c8 = tc*8;          // 128-col base (8 cols)

    size_t base_bh = ((size_t)(b*H + h))*S;
    const __nv_bfloat16* Kbh  = K  + base_bh*Dh;
    const __nv_bfloat16* Vbh  = V  + base_bh*Dh;
    const __nv_bfloat16* Qbh  = Q  + base_bh*Dh;
    const __nv_bfloat16* dObh = dO + base_bh*Dh;
    const float* Lbh = L    + base_bh;
    const float* Dbh = Dvec + base_bh;

    __nv_bfloat16 zero = __float2bfloat16(0.0f);

    // load K, V tile
    #pragma unroll
    for (int n=0;n<32;n++){
        int idx = tid + n*256;
        int r = idx >> 7;       // /128
        int c = idx & 127;
        int gk = k0 + r;
        Ks[r*Dh+c] = (gk<S)? Kbh[(size_t)gk*Dh + c] : zero;
        Vs[r*Dh+c] = (gk<S)? Vbh[(size_t)gk*Dh + c] : zero;
    }

    float accDV[4][8]; float accDK[4][8];
    #pragma unroll
    for(int i=0;i<4;i++)
        #pragma unroll
        for(int j=0;j<8;j++){ accDV[i][j]=0.f; accDK[i][j]=0.f; }

    __syncthreads();

    for (int qb=kb; qb<nq; qb++) {
        int q0 = qb*BQ;

        // load Q, dO
        #pragma unroll
        for (int n=0;n<32;n++){
            int idx = tid + n*256;
            int r = idx >> 7;
            int c = idx & 127;
            int gq = q0 + r;
            Qs[r*Dh+c]  = (gq<S)? Qbh[(size_t)gq*Dh+c]  : zero;
            dOs[r*Dh+c] = (gq<S)? dObh[(size_t)gq*Dh+c] : zero;
        }
        if (tid < BQ) {
            int gq = q0 + tid;
            Ls[tid] = (gq<S)? Lbh[gq] : 0.f;
            Ds[tid] = (gq<S)? Dbh[gq] : 0.f;
        }
        __syncthreads();

        // GEMM1: S = Q*K^T  -> P = exp(scale*S - L), causal masked
        float accS[4][4];
        #pragma unroll
        for(int i=0;i<4;i++)
            #pragma unroll
            for(int j=0;j<4;j++) accS[i][j]=0.f;
        #pragma unroll 4
        for (int e=0;e<Dh;e++){
            float a0=__bfloat162float(Qs[(r0+0)*Dh+e]);
            float a1=__bfloat162float(Qs[(r0+1)*Dh+e]);
            float a2=__bfloat162float(Qs[(r0+2)*Dh+e]);
            float a3=__bfloat162float(Qs[(r0+3)*Dh+e]);
            float b0=__bfloat162float(Ks[(c0+0)*Dh+e]);
            float b1=__bfloat162float(Ks[(c0+1)*Dh+e]);
            float b2=__bfloat162float(Ks[(c0+2)*Dh+e]);
            float b3=__bfloat162float(Ks[(c0+3)*Dh+e]);
            accS[0][0]+=a0*b0;accS[0][1]+=a0*b1;accS[0][2]+=a0*b2;accS[0][3]+=a0*b3;
            accS[1][0]+=a1*b0;accS[1][1]+=a1*b1;accS[1][2]+=a1*b2;accS[1][3]+=a1*b3;
            accS[2][0]+=a2*b0;accS[2][1]+=a2*b1;accS[2][2]+=a2*b2;accS[2][3]+=a2*b3;
            accS[3][0]+=a3*b0;accS[3][1]+=a3*b1;accS[3][2]+=a3*b2;accS[3][3]+=a3*b3;
        }
        #pragma unroll
        for(int i=0;i<4;i++){
            int gq=q0+r0+i;
            float li=Ls[r0+i];
            #pragma unroll
            for(int j=0;j<4;j++){
                int gk=k0+c0+j;
                float p = (gq<S && gk<S && gk<=gq) ? __expf(scale*accS[i][j]-li) : 0.f;
                Pbuf[(r0+i)*BK + (c0+j)] = p;
            }
        }
        __syncthreads();

        // GEMM2: dP = dO*V^T -> dS = P*(dP - D)
        float accDP[4][4];
        #pragma unroll
        for(int i=0;i<4;i++)
            #pragma unroll
            for(int j=0;j<4;j++) accDP[i][j]=0.f;
        #pragma unroll 4
        for (int e=0;e<Dh;e++){
            float a0=__bfloat162float(dOs[(r0+0)*Dh+e]);
            float a1=__bfloat162float(dOs[(r0+1)*Dh+e]);
            float a2=__bfloat162float(dOs[(r0+2)*Dh+e]);
            float a3=__bfloat162float(dOs[(r0+3)*Dh+e]);
            float b0=__bfloat162float(Vs[(c0+0)*Dh+e]);
            float b1=__bfloat162float(Vs[(c0+1)*Dh+e]);
            float b2=__bfloat162float(Vs[(c0+2)*Dh+e]);
            float b3=__bfloat162float(Vs[(c0+3)*Dh+e]);
            accDP[0][0]+=a0*b0;accDP[0][1]+=a0*b1;accDP[0][2]+=a0*b2;accDP[0][3]+=a0*b3;
            accDP[1][0]+=a1*b0;accDP[1][1]+=a1*b1;accDP[1][2]+=a1*b2;accDP[1][3]+=a1*b3;
            accDP[2][0]+=a2*b0;accDP[2][1]+=a2*b1;accDP[2][2]+=a2*b2;accDP[2][3]+=a2*b3;
            accDP[3][0]+=a3*b0;accDP[3][1]+=a3*b1;accDP[3][2]+=a3*b2;accDP[3][3]+=a3*b3;
        }
        #pragma unroll
        for(int i=0;i<4;i++){
            float di=Ds[r0+i];
            #pragma unroll
            for(int j=0;j<4;j++){
                float p = Pbuf[(r0+i)*BK+(c0+j)];
                Sbuf[(r0+i)*BK+(c0+j)] = p*(accDP[i][j]-di);
            }
        }
        __syncthreads();

        // GEMM3: dV += P^T @ dO ; GEMM4: dK += dS^T @ Q  (rows=key, cols=e)
        #pragma unroll 4
        for (int q=0;q<BQ;q++){
            float p0=Pbuf[q*BK + (r0+0)];
            float p1=Pbuf[q*BK + (r0+1)];
            float p2=Pbuf[q*BK + (r0+2)];
            float p3=Pbuf[q*BK + (r0+3)];
            float s0=Sbuf[q*BK + (r0+0)];
            float s1=Sbuf[q*BK + (r0+1)];
            float s2=Sbuf[q*BK + (r0+2)];
            float s3=Sbuf[q*BK + (r0+3)];
            #pragma unroll
            for(int j=0;j<8;j++){
                float dovj=__bfloat162float(dOs[q*Dh + (c8+j)]);
                float qvj =__bfloat162float(Qs[q*Dh + (c8+j)]);
                accDV[0][j]+=p0*dovj; accDV[1][j]+=p1*dovj; accDV[2][j]+=p2*dovj; accDV[3][j]+=p3*dovj;
                accDK[0][j]+=s0*qvj;  accDK[1][j]+=s1*qvj;  accDK[2][j]+=s2*qvj;  accDK[3][j]+=s3*qvj;
            }
        }

        // GEMM5: dQ = scale * dS @ K  (rows=query, cols=e)
        float accDQ[4][8];
        #pragma unroll
        for(int i=0;i<4;i++)
            #pragma unroll
            for(int j=0;j<8;j++) accDQ[i][j]=0.f;
        #pragma unroll 4
        for (int k=0;k<BK;k++){
            float s0=Sbuf[(r0+0)*BK + k];
            float s1=Sbuf[(r0+1)*BK + k];
            float s2=Sbuf[(r0+2)*BK + k];
            float s3=Sbuf[(r0+3)*BK + k];
            #pragma unroll
            for(int j=0;j<8;j++){
                float kvj=__bfloat162float(Ks[k*Dh + (c8+j)]);
                accDQ[0][j]+=s0*kvj; accDQ[1][j]+=s1*kvj; accDQ[2][j]+=s2*kvj; accDQ[3][j]+=s3*kvj;
            }
        }
        #pragma unroll
        for(int i=0;i<4;i++){
            int gq=q0+r0+i;
            if(gq<S){
                #pragma unroll
                for(int j=0;j<8;j++){
                    int e=c8+j;
                    atomicAdd(&dQf[(base_bh+gq)*Dh + e], scale*accDQ[i][j]);
                }
            }
        }
        __syncthreads();
    }

    // write dV, dK
    #pragma unroll
    for(int i=0;i<4;i++){
        int gk=k0+r0+i;
        if(gk<S){
            #pragma unroll
            for(int j=0;j<8;j++){
                int e=c8+j;
                dV[(base_bh+gk)*Dh+e]=__float2bfloat16(accDV[i][j]);
                dK[(base_bh+gk)*Dh+e]=__float2bfloat16(scale*accDK[i][j]);
            }
        }
    }
}

void run(TensorView Q, TensorView K, TensorView V, TensorView O, TensorView dO, TensorView L,
         TensorView dQ, TensorView dK, TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int S = (int)Q.size(2);
    int d = (int)Q.size(3);

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    const __nv_bfloat16* Qp  = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* Kp  = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* Vp  = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* Op  = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dOp = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* Lp = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQp = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dKp = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dVp = static_cast<__nv_bfloat16*>(dV.data_ptr());

    float scale = 1.0f/sqrtf((float)d);

    size_t nrows = (size_t)B*H*S;
    size_t nelem = nrows*(size_t)d;

    float* Dvec=nullptr; CUDA_CHECK(cudaMalloc(&Dvec, nrows*sizeof(float)));
    float* dQf=nullptr;  CUDA_CHECK(cudaMalloc(&dQf, nelem*sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(dQf, 0, nelem*sizeof(float), stream));

    int thr=256;
    {
        size_t need=(nrows+thr-1)/thr;
        unsigned blk=(unsigned)std::min(need,(size_t)65535u*32u);
        if(blk==0) blk=1;
        compute_D_kernel<<<blk,thr,0,stream>>>(dOp,Op,Dvec,nrows);
    }

    int nkv = (S + BK - 1)/BK;
    dim3 grid(nkv, H, B);
    int smem = (int)(BK*Dh*sizeof(__nv_bfloat16)*2 + BQ*Dh*sizeof(__nv_bfloat16)*2
                     + BQ*BK*sizeof(float)*2 + BQ*sizeof(float)*2);
    CUDA_CHECK(cudaFuncSetAttribute(bwd_kernel,
                 cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    bwd_kernel<<<grid, 256, smem, stream>>>(Qp,Kp,Vp,dOp,Lp,Dvec,dKp,dVp,dQf,B,H,S,scale);

    {
        size_t need=(nelem+thr-1)/thr;
        unsigned blk=(unsigned)std::min(need,(size_t)65535u*32u);
        if(blk==0) blk=1;
        convert_kernel<<<blk,thr,0,stream>>>(dQf,dQp,nelem);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaFree(Dvec);
    cudaFree(dQf);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd