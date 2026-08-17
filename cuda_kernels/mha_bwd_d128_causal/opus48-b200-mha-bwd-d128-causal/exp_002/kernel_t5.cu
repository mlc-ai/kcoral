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

static constexpr int BM = 64;
static constexpr int BN = 64;
static constexpr int HD = 128;
static constexpr int NTH = 256;

template<int ROWS>
__device__ __forceinline__ void load_tile_f32(float* smem, const bf16* g, int valid_rows,
                                               int tid){
    const int per_row = HD/8; // 16
    int total = ROWS*per_row;
    for(int vi=tid; vi<total; vi+=NTH){
        int row = vi/per_row;
        int cg  = vi%per_row;
        if(row<valid_rows){
            int4 raw = *reinterpret_cast<const int4*>(g + (size_t)row*HD + cg*8);
            const bf16* b = reinterpret_cast<const bf16*>(&raw);
            #pragma unroll
            for(int e=0;e<8;e++) smem[row*HD + cg*8 + e] = __bfloat162float(b[e]);
        } else {
            #pragma unroll
            for(int e=0;e<8;e++) smem[row*HD + cg*8 + e] = 0.f;
        }
    }
}

// D_i = sum_k dO_ik * O_ik  (fp32, exact cuDNN delta), warp per row
__global__ void compute_D(const bf16* O, const bf16* dO, float* Dout, int total_rows){
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

// -------- dK / dV kernel: block = one KV tile, loops over Q tiles (fp32) --------
__global__ void bwd_dkdv(const bf16* Q,const bf16* K,const bf16* V,const bf16* dO,
                         const float* Lg,const float* Dg,
                         bf16* dKout,bf16* dVout,int S,float scale){
    int bh = blockIdx.y;
    int kv_start = blockIdx.x*BN;
    if(kv_start>=S) return;
    const int tid = threadIdx.x;

    extern __shared__ float smem[];
    float* Ks = smem;
    float* Vs = Ks + BN*HD;
    float* Qs = Vs + BN*HD;
    float* dOs= Qs + BM*HD;
    float* Ps = dOs+ BM*HD;
    float* dSs= Ps + BM*BN;
    float* Ls = dSs+ BM*BN;
    float* Ds = Ls + BM;

    size_t bh_off = (size_t)bh*S*HD;
    size_t bh_l   = (size_t)bh*S;

    load_tile_f32<BN>(Ks, K + bh_off + (size_t)kv_start*HD, min(BN,S-kv_start), tid);
    load_tile_f32<BN>(Vs, V + bh_off + (size_t)kv_start*HD, min(BN,S-kv_start), tid);

    // phase3 mapping (i,j) 4x4
    int i3 = (tid/16)*4;
    int j3 = (tid%16)*4;
    // phase4 mapping (j,k) 4x8
    int j4 = (tid/16)*4;
    int k4 = (tid%16)*8;

    float dVr[4][8], dKr[4][8];
    #pragma unroll
    for(int a=0;a<4;a++)
        #pragma unroll
        for(int b=0;b<8;b++){ dVr[a][b]=0.f; dKr[a][b]=0.f; }
    __syncthreads();

    for(int q_start=kv_start; q_start<S; q_start+=BM){
        int qrows = min(BM,S-q_start);
        load_tile_f32<BM>(Qs,  Q  + bh_off + (size_t)q_start*HD, qrows, tid);
        load_tile_f32<BM>(dOs, dO + bh_off + (size_t)q_start*HD, qrows, tid);
        for(int i=tid;i<BM;i+=NTH){
            int qg=q_start+i;
            Ls[i]=(qg<S)?Lg[bh_l+qg]:0.f;
            Ds[i]=(qg<S)?Dg[bh_l+qg]:0.f;
        }
        __syncthreads();

        // phase3: S, dP -> P, dS
        float Sr[4][4], dPr[4][4];
        #pragma unroll
        for(int a=0;a<4;a++)
            #pragma unroll
            for(int b=0;b<4;b++){ Sr[a][b]=0.f; dPr[a][b]=0.f; }
        #pragma unroll 4
        for(int k=0;k<HD;k++){
            float q[4],o[4],kk[4],v[4];
            #pragma unroll
            for(int a=0;a<4;a++){ q[a]=Qs[(i3+a)*HD+k]; o[a]=dOs[(i3+a)*HD+k]; }
            #pragma unroll
            for(int b=0;b<4;b++){ kk[b]=Ks[(j3+b)*HD+k]; v[b]=Vs[(j3+b)*HD+k]; }
            #pragma unroll
            for(int a=0;a<4;a++)
                #pragma unroll
                for(int b=0;b<4;b++){ Sr[a][b]+=q[a]*kk[b]; dPr[a][b]+=o[a]*v[b]; }
        }
        #pragma unroll
        for(int a=0;a<4;a++){
            #pragma unroll
            for(int b=0;b<4;b++){
                int i=i3+a, j=j3+b;
                int qg=q_start+i, kg=kv_start+j;
                float P=0.f, dS=0.f;
                if(qg<S && kg<S && qg>=kg){
                    float L=Ls[i], D=Ds[i];
                    P = __expf(scale*Sr[a][b]-L);
                    dS = P*(dPr[a][b]-D);
                }
                Ps[i*BN+j]=P;
                dSs[i*BN+j]=dS;
            }
        }
        __syncthreads();

        // phase4: dV += P^T dO ; dK += dS^T Q
        for(int i=0;i<BM;i++){
            float pcol[4], scol[4], ocol[8], qcol[8];
            #pragma unroll
            for(int a=0;a<4;a++){ pcol[a]=Ps[i*BN+(j4+a)]; scol[a]=dSs[i*BN+(j4+a)]; }
            #pragma unroll
            for(int b=0;b<8;b++){ ocol[b]=dOs[i*HD+(k4+b)]; qcol[b]=Qs[i*HD+(k4+b)]; }
            #pragma unroll
            for(int a=0;a<4;a++)
                #pragma unroll
                for(int b=0;b<8;b++){ dVr[a][b]+=pcol[a]*ocol[b]; dKr[a][b]+=scol[a]*qcol[b]; }
        }
        __syncthreads();
    }

    #pragma unroll
    for(int a=0;a<4;a++){
        int j=j4+a; int kg=kv_start+j;
        if(kg<S){
            #pragma unroll
            for(int b=0;b<8;b++){
                int k=k4+b;
                size_t idx=bh_off+(size_t)kg*HD+k;
                dVout[idx]=__float2bfloat16(dVr[a][b]);
                dKout[idx]=__float2bfloat16(scale*dKr[a][b]);
            }
        }
    }
}

// -------- dQ kernel: block = one Q tile, loops over KV tiles (fp32) --------
__global__ void bwd_dq(const bf16* Q,const bf16* K,const bf16* V,const bf16* dO,
                       const float* Lg,const float* Dg,
                       bf16* dQout,int S,float scale){
    int bh = blockIdx.y;
    int q_start = blockIdx.x*BM;
    if(q_start>=S) return;
    const int tid = threadIdx.x;

    extern __shared__ float smem[];
    float* Qs = smem;
    float* dOs= Qs + BM*HD;
    float* Ks = dOs+ BM*HD;
    float* Vs = Ks + BN*HD;
    float* dSs= Vs + BN*HD;
    float* Ls = dSs+ BM*BN;
    float* Ds = Ls + BM;

    size_t bh_off = (size_t)bh*S*HD;
    size_t bh_l   = (size_t)bh*S;

    int qrows = min(BM,S-q_start);
    load_tile_f32<BM>(Qs,  Q  + bh_off + (size_t)q_start*HD, qrows, tid);
    load_tile_f32<BM>(dOs, dO + bh_off + (size_t)q_start*HD, qrows, tid);
    for(int i=tid;i<BM;i+=NTH){
        int qg=q_start+i;
        Ls[i]=(qg<S)?Lg[bh_l+qg]:0.f;
        Ds[i]=(qg<S)?Dg[bh_l+qg]:0.f;
    }

    int i3 = (tid/16)*4;   // phase3 (i,j)
    int j3 = (tid%16)*4;
    int i4 = (tid/16)*4;   // phase4 (i,k)
    int k4 = (tid%16)*8;

    float dQr[4][8];
    #pragma unroll
    for(int a=0;a<4;a++)
        #pragma unroll
        for(int b=0;b<8;b++) dQr[a][b]=0.f;
    __syncthreads();

    for(int kv_start=0; kv_start<=q_start && kv_start<S; kv_start+=BN){
        int krows = min(BN,S-kv_start);
        load_tile_f32<BN>(Ks, K + bh_off + (size_t)kv_start*HD, krows, tid);
        load_tile_f32<BN>(Vs, V + bh_off + (size_t)kv_start*HD, krows, tid);
        __syncthreads();

        float Sr[4][4], dPr[4][4];
        #pragma unroll
        for(int a=0;a<4;a++)
            #pragma unroll
            for(int b=0;b<4;b++){ Sr[a][b]=0.f; dPr[a][b]=0.f; }
        #pragma unroll 4
        for(int k=0;k<HD;k++){
            float q[4],o[4],kk[4],v[4];
            #pragma unroll
            for(int a=0;a<4;a++){ q[a]=Qs[(i3+a)*HD+k]; o[a]=dOs[(i3+a)*HD+k]; }
            #pragma unroll
            for(int b=0;b<4;b++){ kk[b]=Ks[(j3+b)*HD+k]; v[b]=Vs[(j3+b)*HD+k]; }
            #pragma unroll
            for(int a=0;a<4;a++)
                #pragma unroll
                for(int b=0;b<4;b++){ Sr[a][b]+=q[a]*kk[b]; dPr[a][b]+=o[a]*v[b]; }
        }
        #pragma unroll
        for(int a=0;a<4;a++){
            #pragma unroll
            for(int b=0;b<4;b++){
                int i=i3+a, j=j3+b;
                int qg=q_start+i, kg=kv_start+j;
                float dS=0.f;
                if(qg<S && kg<S && qg>=kg){
                    float L=Ls[i], D=Ds[i];
                    float P = __expf(scale*Sr[a][b]-L);
                    dS = P*(dPr[a][b]-D);
                }
                dSs[i*BN+j]=dS;
            }
        }
        __syncthreads();

        // phase4: dQ += dS @ K
        for(int j=0;j<BN;j++){
            float scol[4], kcol[8];
            #pragma unroll
            for(int a=0;a<4;a++) scol[a]=dSs[(i4+a)*BN+j];
            #pragma unroll
            for(int b=0;b<8;b++) kcol[b]=Ks[j*HD+(k4+b)];
            #pragma unroll
            for(int a=0;a<4;a++)
                #pragma unroll
                for(int b=0;b<8;b++) dQr[a][b]+=scol[a]*kcol[b];
        }
        __syncthreads();
    }

    #pragma unroll
    for(int a=0;a<4;a++){
        int i=i4+a; int qg=q_start+i;
        if(qg<S){
            #pragma unroll
            for(int b=0;b<8;b++){
                int k=k4+b;
                size_t idx=bh_off+(size_t)qg*HD+k;
                dQout[idx]=__float2bfloat16(scale*dQr[a][b]);
            }
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

    float* Dscratch = nullptr;
    size_t total_rows = (size_t)BH*Si;
    CUDA_CHECK(cudaMallocAsync(&Dscratch, sizeof(float)*total_rows, stream));

    size_t sh_dkdv = (size_t)(4*BM*HD + 2*BM*BN + 2*BM)*sizeof(float);
    size_t sh_dq   = (size_t)(4*BM*HD + 1*BM*BN + 2*BM)*sizeof(float);

    CUDA_CHECK(cudaFuncSetAttribute((const void*)bwd_dkdv,
               cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh_dkdv));
    CUDA_CHECK(cudaFuncSetAttribute((const void*)bwd_dq,
               cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh_dq));

    {
        int threads=256;
        int wpb = threads/32;
        int blocks = (int)((total_rows + wpb - 1)/wpb);
        compute_D<<<blocks, threads, 0, stream>>>(Op, dOp, Dscratch, (int)total_rows);
        CUDA_CHECK(cudaGetLastError());
    }
    {
        int nkv = (Si + BN - 1)/BN;
        dim3 grid(nkv, BH);
        bwd_dkdv<<<grid, NTH, sh_dkdv, stream>>>(Qp,Kp,Vp,dOp,Lp,Dscratch,dKp,dVp,Si,scale);
        CUDA_CHECK(cudaGetLastError());
    }
    {
        int nq = (Si + BM - 1)/BM;
        dim3 grid(nq, BH);
        bwd_dq<<<grid, NTH, sh_dq, stream>>>(Qp,Kp,Vp,dOp,Lp,Dscratch,dQp,Si,scale);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaFreeAsync(Dscratch, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

} // namespace mha_bwd