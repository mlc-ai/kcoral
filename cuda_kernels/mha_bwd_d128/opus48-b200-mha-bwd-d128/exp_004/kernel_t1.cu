#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cmath>
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

namespace mha_bwd {

using namespace nvcuda;
typedef __nv_bfloat16 bf16;

static const int SMEM_BYTES = 230400;

__global__ void compute_D_kernel(const bf16* dO, const bf16* O, float* D, long total) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        long base = idx * 128;
        float acc = 0.f;
        #pragma unroll
        for (int c = 0; c < 128; c++) acc += (float)dO[base + c] * (float)O[base + c];
        D[idx] = acc;
    }
}

__device__ __forceinline__ void load_tile128(bf16* dst, const bf16* src, int rowbase, int S, int tid){
    #pragma unroll
    for(int i=tid;i<128*16;i+=256){
        int r=i>>4; int cg=i&15; int gr=rowbase+r;
        int4 v; if(gr<S) v=((const int4*)(src+(long)gr*128))[cg]; else v=make_int4(0,0,0,0);
        ((int4*)(dst+r*128))[cg]=v;
    }
}

// C[q,kv] = A[q,d] @ B[kv,d]^T   (A row-major, B row-major used as col_major -> B^T)
__device__ __forceinline__ void gemm_ABt(float* sC, const bf16* sA, const bf16* sB, int warp){
    #pragma unroll
    for(int nt=0;nt<8;nt++){
        wmma::fragment<wmma::accumulator,16,16,16,float> c;
        wmma::fill_fragment(c,0.f);
        #pragma unroll
        for(int kt=0;kt<8;kt++){
            wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> a;
            wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::col_major> b;
            wmma::load_matrix_sync(a, sA + warp*16*128 + kt*16, 128);
            wmma::load_matrix_sync(b, sB + nt*16*128 + kt*16, 128);
            wmma::mma_sync(c,a,b,c);
        }
        wmma::store_matrix_sync(sC + warp*16*128 + nt*16, c, 128, wmma::mem_row_major);
    }
}

__global__ __launch_bounds__(256,1) void bwd_dkdv_kernel(
    const bf16* Q,const bf16* K,const bf16* V,const bf16* dO,
    const float* Lp,const float* Dp, bf16* dKo, bf16* dVo,
    int S,int H,float scale)
{
    extern __shared__ char smem[];
    bf16* sK=(bf16*)(smem+0);
    bf16* sV=(bf16*)(smem+32768);
    bf16* sQ=(bf16*)(smem+65536);
    bf16* sdO=(bf16*)(smem+98304);
    float* sScore=(float*)(smem+131072);
    bf16* sP=(bf16*)(smem+196608);
    float* sL=(float*)(smem+229376);
    float* sD=(float*)(smem+229888);

    int tid=threadIdx.x; int warp=tid>>5;
    int j=blockIdx.x, h=blockIdx.y, b=blockIdx.z;
    int bh=b*H+h; long base=(long)bh*S*128;
    const bf16 *Qb=Q+base,*Kb=K+base,*Vb=V+base,*dOb=dO+base;
    int kvbase=j*128; int numQ=(S+127)/128;

    load_tile128(sK,Kb,kvbase,S,tid);
    load_tile128(sV,Vb,kvbase,S,tid);

    wmma::fragment<wmma::accumulator,16,16,16,float> accV[8],accK[8];
    #pragma unroll
    for(int nt=0;nt<8;nt++){wmma::fill_fragment(accV[nt],0.f);wmma::fill_fragment(accK[nt],0.f);}
    __syncthreads();

    for(int i=0;i<numQ;i++){
        int qbase=i*128;
        load_tile128(sQ,Qb,qbase,S,tid);
        load_tile128(sdO,dOb,qbase,S,tid);
        for(int r=tid;r<128;r+=256){int gq=qbase+r; sL[r]=(gq<S)?Lp[(long)bh*S+gq]:1e30f; sD[r]=(gq<S)?Dp[(long)bh*S+gq]:0.f;}
        __syncthreads();

        gemm_ABt(sScore,sQ,sK,warp);      // S = Q@K^T
        __syncthreads();
        for(int idx=tid;idx<16384;idx+=256){int q=idx>>7,kv=idx&127; float s=sScore[idx];
            float p=(qbase+q<S&&kvbase+kv<S)?__expf(scale*s-sL[q]):0.f; sP[idx]=__float2bfloat16(p);}
        __syncthreads();
        gemm_ABt(sScore,sdO,sV,warp);     // dP = dO@V^T  (overwrite S)
        __syncthreads();

        // dV += P^T @ dO
        #pragma unroll
        for(int nt=0;nt<8;nt++){
            #pragma unroll
            for(int kt=0;kt<8;kt++){
                wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::col_major> a;
                wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::row_major> bb;
                wmma::load_matrix_sync(a, sP + kt*16*128 + warp*16, 128);
                wmma::load_matrix_sync(bb, sdO + kt*16*128 + nt*16, 128);
                wmma::mma_sync(accV[nt],a,bb,accV[nt]);
            }
        }
        __syncthreads();
        // dS = scale*P*(dP - D) -> sP (overwrite P)
        for(int idx=tid;idx<16384;idx+=256){int q=idx>>7; float p=(float)sP[idx]; float dp=sScore[idx];
            float ds=scale*p*(dp-sD[q]); sP[idx]=__float2bfloat16(ds);}
        __syncthreads();
        // dK += dS^T @ Q
        #pragma unroll
        for(int nt=0;nt<8;nt++){
            #pragma unroll
            for(int kt=0;kt<8;kt++){
                wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::col_major> a;
                wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::row_major> bb;
                wmma::load_matrix_sync(a, sP + kt*16*128 + warp*16, 128);
                wmma::load_matrix_sync(bb, sQ + kt*16*128 + nt*16, 128);
                wmma::mma_sync(accK[nt],a,bb,accK[nt]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int nt=0;nt<8;nt++) wmma::store_matrix_sync(sScore+warp*16*128+nt*16,accV[nt],128,wmma::mem_row_major);
    __syncthreads();
    for(int e=tid;e<16384;e+=256){int kvl=e>>7,d=e&127;int gk=kvbase+kvl; if(gk<S) dVo[base+(long)gk*128+d]=__float2bfloat16(sScore[e]);}
    __syncthreads();
    #pragma unroll
    for(int nt=0;nt<8;nt++) wmma::store_matrix_sync(sScore+warp*16*128+nt*16,accK[nt],128,wmma::mem_row_major);
    __syncthreads();
    for(int e=tid;e<16384;e+=256){int kvl=e>>7,d=e&127;int gk=kvbase+kvl; if(gk<S) dKo[base+(long)gk*128+d]=__float2bfloat16(sScore[e]);}
}

__global__ __launch_bounds__(256,1) void bwd_dq_kernel(
    const bf16* Q,const bf16* K,const bf16* V,const bf16* dO,
    const float* Lp,const float* Dp, bf16* dQo, int S,int H,float scale)
{
    extern __shared__ char smem[];
    bf16* sQ=(bf16*)(smem+0);
    bf16* sdO=(bf16*)(smem+32768);
    bf16* sK=(bf16*)(smem+65536);
    bf16* sV=(bf16*)(smem+98304);
    float* sScore=(float*)(smem+131072);
    bf16* sP=(bf16*)(smem+196608);
    float* sL=(float*)(smem+229376);
    float* sD=(float*)(smem+229888);

    int tid=threadIdx.x, warp=tid>>5;
    int iq=blockIdx.x, h=blockIdx.y, b=blockIdx.z;
    int bh=b*H+h; long base=(long)bh*S*128;
    const bf16 *Qb=Q+base,*Kb=K+base,*Vb=V+base,*dOb=dO+base;
    int qbase=iq*128; int numKV=(S+127)/128;

    load_tile128(sQ,Qb,qbase,S,tid);
    load_tile128(sdO,dOb,qbase,S,tid);
    for(int r=tid;r<128;r+=256){int gq=qbase+r; sL[r]=(gq<S)?Lp[(long)bh*S+gq]:1e30f; sD[r]=(gq<S)?Dp[(long)bh*S+gq]:0.f;}

    wmma::fragment<wmma::accumulator,16,16,16,float> accQ[8];
    #pragma unroll
    for(int nt=0;nt<8;nt++) wmma::fill_fragment(accQ[nt],0.f);
    __syncthreads();

    for(int j=0;j<numKV;j++){
        int kvbase=j*128;
        load_tile128(sK,Kb,kvbase,S,tid);
        load_tile128(sV,Vb,kvbase,S,tid);
        __syncthreads();
        gemm_ABt(sScore,sQ,sK,warp);      // S
        __syncthreads();
        for(int idx=tid;idx<16384;idx+=256){int q=idx>>7,kv=idx&127; float s=sScore[idx];
            float p=(qbase+q<S&&kvbase+kv<S)?__expf(scale*s-sL[q]):0.f; sP[idx]=__float2bfloat16(p);}
        __syncthreads();
        gemm_ABt(sScore,sdO,sV,warp);     // dP
        __syncthreads();
        for(int idx=tid;idx<16384;idx+=256){int q=idx>>7; float p=(float)sP[idx]; float dp=sScore[idx];
            float ds=scale*p*(dp-sD[q]); sP[idx]=__float2bfloat16(ds);}
        __syncthreads();
        // dQ += dS @ K
        #pragma unroll
        for(int nt=0;nt<8;nt++){
            #pragma unroll
            for(int kt=0;kt<8;kt++){
                wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> a;
                wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::row_major> bb;
                wmma::load_matrix_sync(a, sP + warp*16*128 + kt*16, 128);
                wmma::load_matrix_sync(bb, sK + kt*16*128 + nt*16, 128);
                wmma::mma_sync(accQ[nt],a,bb,accQ[nt]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int nt=0;nt<8;nt++) wmma::store_matrix_sync(sScore+warp*16*128+nt*16,accQ[nt],128,wmma::mem_row_major);
    __syncthreads();
    for(int e=tid;e<16384;e+=256){int ql=e>>7,d=e&127;int gq=qbase+ql; if(gq<S) dQo[base+(long)gq*128+d]=__float2bfloat16(sScore[e]);}
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = Q.size(3);

    const bf16* Qp  = static_cast<const bf16*>(Q.data_ptr());
    const bf16* Kp  = static_cast<const bf16*>(K.data_ptr());
    const bf16* Vp  = static_cast<const bf16*>(V.data_ptr());
    const bf16* Op  = static_cast<const bf16*>(O.data_ptr());
    const bf16* dOp = static_cast<const bf16*>(dO.data_ptr());
    const float* Lp = static_cast<const float*>(L.data_ptr());
    bf16* dQp = static_cast<bf16*>(dQ.data_ptr());
    bf16* dKp = static_cast<bf16*>(dK.data_ptr());
    bf16* dVp = static_cast<bf16*>(dV.data_ptr());

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    if (S <= 0) return;

    long rows = B * H * S;
    float* Dbuf = nullptr;
    CUDA_CHECK(cudaMallocAsync(&Dbuf, rows * sizeof(float), stream));

    {
        int threads = 256;
        long blocks = (rows + threads - 1) / threads;
        compute_D_kernel<<<blocks, threads, 0, stream>>>(dOp, Op, Dbuf, rows);
        CUDA_CHECK(cudaGetLastError());
    }

    static bool attr_set = false;
    if (!attr_set) {
        CUDA_CHECK(cudaFuncSetAttribute(bwd_dkdv_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
        CUDA_CHECK(cudaFuncSetAttribute(bwd_dq_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
        attr_set = true;
    }

    int nblk = (int)((S + 127) / 128);
    dim3 grid(nblk, (unsigned)H, (unsigned)B);
    float scale = 1.0f / sqrtf((float)d);

    bwd_dkdv_kernel<<<grid, 256, SMEM_BYTES, stream>>>(
        Qp, Kp, Vp, dOp, Lp, Dbuf, dKp, dVp, (int)S, (int)H, scale);
    CUDA_CHECK(cudaGetLastError());

    bwd_dq_kernel<<<grid, 256, SMEM_BYTES, stream>>>(
        Qp, Kp, Vp, dOp, Lp, Dbuf, dQp, (int)S, (int)H, scale);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFreeAsync(Dbuf, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd