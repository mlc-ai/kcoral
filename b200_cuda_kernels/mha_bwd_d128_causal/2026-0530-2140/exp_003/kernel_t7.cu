#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <mma.h>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__); exit(1);} } while(0)

using namespace nvcuda;

namespace mha_bwd {

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int Dh = 128;
constexpr int THREADS = 256;
constexpr int NWARPS = THREADS/32;
constexpr int LDO = Dh + 8;   // 136
constexpr int LDS = BN + 8;   // 72

__device__ __forceinline__ void cp_async16(void* s, const void* g){
    uint32_t sa=(uint32_t)__cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"::"r"(sa),"l"(g));
}
__device__ __forceinline__ void cp_commit(){ asm volatile("cp.async.commit_group;\n"); }
template<int N> __device__ __forceinline__ void cp_wait(){ asm volatile("cp.async.wait_group %0;\n"::"n"(N)); }

__device__ __forceinline__ void load_tile_async(const __nv_bfloat16* __restrict__ g, long base,
        int row0, int rows, int S, __nv_bfloat16* __restrict__ s, int tid){
    int total=rows*16;
    if(row0+rows<=S){
        for(int i=tid;i<total;i+=THREADS){
            int r=i>>4,c=i&15;
            cp_async16(s+r*LDO+c*8, g+base+(long)(row0+r)*Dh+c*8);
        }
    } else {
        for(int i=tid;i<total;i+=THREADS){
            int r=i>>4,c=i&15; int gr=row0+r;
            int4 v; if(gr<S) v=*reinterpret_cast<const int4*>(g+base+(long)gr*Dh+c*8);
            else {v.x=v.y=v.z=v.w=0;}
            *reinterpret_cast<int4*>(s+r*LDO+c*8)=v;
        }
    }
}

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
    int2 dv = *reinterpret_cast<const int2*>(dop + lane*4);
    int2 ov = *reinterpret_cast<const int2*>(op  + lane*4);
    __nv_bfloat16* dp=(__nv_bfloat16*)&dv; __nv_bfloat16* opp=(__nv_bfloat16*)&ov;
    #pragma unroll
    for(int j=0;j<4;j++) sum += __bfloat162float(dp[j])*__bfloat162float(opp[j]);
    for(int off=16;off>0;off>>=1) sum += __shfl_down_sync(0xffffffffu,sum,off);
    if(lane==0) Delta[row]=sum;
}

// Out[BM][LDS] = A[BM][LDO] @ B[BN][LDO]^T  (contract Dh)
__device__ __forceinline__ void gemm_ABt(const __nv_bfloat16* A, const __nv_bfloat16* B,
                                          float* Out, int warp_id){
    constexpr int nti=BM/16, ntj=BN/16, total=nti*ntj;
    for(int t=warp_id; t<total; t+=NWARPS){
        int ti=t/ntj, tj=t%ntj;
        wmma::fragment<wmma::accumulator,16,16,16,float> c;
        wmma::fill_fragment(c,0.f);
        #pragma unroll
        for(int kk=0; kk<Dh; kk+=16){
            wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a;
            wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> b;
            wmma::load_matrix_sync(a, A + ti*16*LDO + kk, LDO);
            wmma::load_matrix_sync(b, B + kk + tj*16*LDO, LDO);
            wmma::mma_sync(c,a,b,c);
        }
        wmma::store_matrix_sync(Out + ti*16*LDS + tj*16, c, LDS, wmma::mem_row_major);
    }
}

// fused smem offsets (double-buffered sQ,sdO)
constexpr int F_sQ=0;
constexpr int F_sdO=F_sQ + 2*BM*LDO*2;
constexpr int F_sK =F_sdO+ 2*BM*LDO*2;
constexpr int F_sV =F_sK + BN*LDO*2;
constexpr int F_sScore =F_sV + BN*LDO*2;
constexpr int F_sScore2=F_sScore + BM*LDS*4;
constexpr int F_sPbf =F_sScore2 + BM*LDS*4;
constexpr int F_sdSbf=F_sPbf + BM*LDS*2;
constexpr int F_sL =F_sdSbf + BM*LDS*2;
constexpr int F_sD =F_sL + BM*4;
constexpr int SMEM_F = F_sD + BM*4;

__global__ void bwd_fused_kernel(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
                 const __nv_bfloat16* __restrict__ V, const __nv_bfloat16* __restrict__ dO,
                 const float* __restrict__ L, const float* __restrict__ Delta,
                 __nv_bfloat16* __restrict__ dK, __nv_bfloat16* __restrict__ dV,
                 float* __restrict__ dQf, int S, float scale){
    extern __shared__ char smem[];
    __nv_bfloat16* sQ =(__nv_bfloat16*)(smem+F_sQ);
    __nv_bfloat16* sdO=(__nv_bfloat16*)(smem+F_sdO);
    __nv_bfloat16* sK =(__nv_bfloat16*)(smem+F_sK);
    __nv_bfloat16* sV =(__nv_bfloat16*)(smem+F_sV);
    float* sScore =(float*)(smem+F_sScore);
    float* sScore2=(float*)(smem+F_sScore2);
    __nv_bfloat16* sPbf =(__nv_bfloat16*)(smem+F_sPbf);
    __nv_bfloat16* sdSbf=(__nv_bfloat16*)(smem+F_sdSbf);
    float* sL=(float*)(smem+F_sL);
    float* sD=(float*)(smem+F_sD);
    float* stage=(float*)(smem+F_sScore); // 36864 >= 32768

    int bh = blockIdx.y;
    int kv_start = blockIdx.x * BN;
    if(kv_start >= S) return;
    long base = (long)bh * S * Dh;
    int tid = threadIdx.x;
    int warp_id = tid/32;
    int rt = warp_id % 4;
    int cg = warp_id / 4;

    load_tile_async(K, base, kv_start, BN, S, sK, tid);
    load_tile_async(V, base, kv_start, BN, S, sV, tid);
    cp_commit();

    wmma::fragment<wmma::accumulator,16,16,16,float> accV[4], accK[4];
    #pragma unroll
    for(int c=0;c<4;c++){ wmma::fill_fragment(accV[c],0.f); wmma::fill_fragment(accK[c],0.f); }

    int niter = (S - kv_start + BM - 1)/BM;
    int cur=0;
    load_tile_async(Q,  base, kv_start, BM, S, sQ,  tid);
    load_tile_async(dO, base, kv_start, BM, S, sdO, tid);
    cp_commit();

    for(int it=0; it<niter; it++){
        int q_start = kv_start + it*BM;
        int qbuf = cur*(BM*LDO);
        int nbuf = (cur^1)*(BM*LDO);
        if(it+1<niter){
            int qn = kv_start + (it+1)*BM;
            load_tile_async(Q,  base, qn, BM, S, sQ + nbuf,  tid);
            load_tile_async(dO, base, qn, BM, S, sdO + nbuf, tid);
            cp_commit();
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();

        for(int m=tid;m<BM;m+=THREADS){
            int qr=q_start+m;
            if(qr<S){ sL[m]=L[(long)bh*S+qr]; sD[m]=Delta[(long)bh*S+qr]; }
            else { sL[m]=0.f; sD[m]=0.f; }
        }
        __syncthreads();

        __nv_bfloat16* sQc=sQ+qbuf;
        __nv_bfloat16* sdOc=sdO+qbuf;

        gemm_ABt(sQc, sK, sScore, warp_id);    // S = Q@K^T
        gemm_ABt(sdOc, sV, sScore2, warp_id);  // dP = dO@V^T
        __syncthreads();

        for(int e=tid;e<BM*BN;e+=THREADS){
            int m=e/BN,n=e%BN; int qr=q_start+m,kr=kv_start+n;
            float p=0.f, ds=0.f;
            if(qr<S && kr<S && kr<=qr){
                p=__expf(sScore[m*LDS+n]*scale - sL[m]);
                ds=scale*p*(sScore2[m*LDS+n]-sD[m]);
            }
            sPbf[m*LDS+n]=__float2bfloat16(p);
            sdSbf[m*LDS+n]=__float2bfloat16(ds);
        }
        __syncthreads();

        // dV += P^T@dO ; dK += dS^T@Q   (contract over q = BM)
        #pragma unroll
        for(int ct=0;ct<4;ct++){
            int col0 = cg*64 + ct*16;
            for(int kk=0;kk<BM;kk+=16){
                wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::col_major> aP, aS;
                wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> bO, bQ;
                wmma::load_matrix_sync(aP, sPbf  + rt*16 + kk*LDS, LDS);
                wmma::load_matrix_sync(bO, sdOc  + kk*LDO + col0, LDO);
                wmma::mma_sync(accV[ct], aP, bO, accV[ct]);
                wmma::load_matrix_sync(aS, sdSbf + rt*16 + kk*LDS, LDS);
                wmma::load_matrix_sync(bQ, sQc   + kk*LDO + col0, LDO);
                wmma::mma_sync(accK[ct], aS, bQ, accK[ct]);
            }
        }
        __syncthreads();   // scores free now -> use stage for dQ

        // dQ = dS @ K  (contract over n = BN), output [BM,Dh] -> stage -> atomicAdd
        #pragma unroll
        for(int ct=0;ct<4;ct++){
            int col0 = cg*64 + ct*16;
            wmma::fragment<wmma::accumulator,16,16,16,float> cq;
            wmma::fill_fragment(cq,0.f);
            for(int nn=0;nn<BN;nn+=16){
                wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a;
                wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> b;
                wmma::load_matrix_sync(a, sdSbf + rt*16*LDS + nn, LDS);
                wmma::load_matrix_sync(b, sK + nn*LDO + col0, LDO);
                wmma::mma_sync(cq, a, b, cq);
            }
            wmma::store_matrix_sync(stage + rt*16*Dh + col0, cq, Dh, wmma::mem_row_major);
        }
        __syncthreads();
        for(int e=tid;e<BM*Dh;e+=THREADS){
            int m=e/Dh,k=e%Dh; int qr=q_start+m;
            if(qr<S) atomicAdd(&dQf[base+(long)qr*Dh+k], stage[e]);
        }
        __syncthreads();
        cur^=1;
    }

    #pragma unroll
    for(int ct=0;ct<4;ct++)
        wmma::store_matrix_sync(stage + rt*16*Dh + cg*64 + ct*16, accV[ct], Dh, wmma::mem_row_major);
    __syncthreads();
    for(int e=tid;e<BN*Dh;e+=THREADS){
        int n=e/Dh,k=e%Dh; int kr=kv_start+n;
        if(kr<S) dV[base+(long)kr*Dh+k]=__float2bfloat16(stage[n*Dh+k]);
    }
    __syncthreads();
    #pragma unroll
    for(int ct=0;ct<4;ct++)
        wmma::store_matrix_sync(stage + rt*16*Dh + cg*64 + ct*16, accK[ct], Dh, wmma::mem_row_major);
    __syncthreads();
    for(int e=tid;e<BN*Dh;e+=THREADS){
        int n=e/Dh,k=e%Dh; int kr=kv_start+n;
        if(kr<S) dK[base+(long)kr*Dh+k]=__float2bfloat16(stage[n*Dh+k]);
    }
}

__global__ void convert_dq_kernel(const float* __restrict__ dQf, __nv_bfloat16* __restrict__ dQ, long n){
    long i = (long)blockIdx.x*blockDim.x + threadIdx.x;
    if(i<n) dQ[i]=__float2bfloat16(dQf[i]);
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

    long elems = (long)BH*S*Dh;
    float* Delta=nullptr; float* dQf=nullptr;
    CUDA_CHECK(cudaMalloc(&Delta, (size_t)BH*S*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dQf, (size_t)elems*sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(dQf, 0, (size_t)elems*sizeof(float), stream));

    long total_rows = (long)BH*S;
    int wpb = 8;
    long blocks_delta = (total_rows + wpb - 1)/wpb;
    compute_delta_kernel<<<(unsigned)blocks_delta, wpb*32, 0, stream>>>(dOp, Op, Delta, total_rows);

    CUDA_CHECK(cudaFuncSetAttribute(bwd_fused_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_F));

    int num_kv = (S+BN-1)/BN;
    dim3 grid_kv(num_kv, BH);
    bwd_fused_kernel<<<grid_kv, THREADS, SMEM_F, stream>>>(Qp,Kp,Vp,dOp,Lp,Delta,dKp,dVp,dQf,S,scale);

    long convblocks = (elems + 255)/256;
    convert_dq_kernel<<<(unsigned)convblocks, 256, 0, stream>>>(dQf, dQp, elems);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(Delta));
    CUDA_CHECK(cudaFree(dQf));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd