#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ fprintf(stderr,"CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__);} } while(0)

namespace mha_bwd {

constexpr int BM=64, BN=64, HD=128, WARP_N=2, NTHREAD=256;
constexpr int LD=HD+8;
constexpr int PBM=BM+8;
constexpr int PBN=BN+8;

__device__ __forceinline__ void cp_async_cg16(void* smem, const void* gmem){
    unsigned s=(unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"::"r"(s),"l"(gmem):"memory");
}
__device__ __forceinline__ void cp_async_commit(){ asm volatile("cp.async.commit_group;\n":::"memory"); }
template<int N> __device__ __forceinline__ void cp_async_wait(){ asm volatile("cp.async.wait_group %0;\n"::"n"(N):"memory"); }

__device__ __forceinline__ void mma16816(const uint32_t a[4], const uint32_t b[2], float* c){
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}

__device__ __forceinline__ void loadA(const __nv_bfloat16* As, int lda, int rowbase, int koff, int lane, uint32_t a[4]){
    int gid=lane>>2, tid=lane&3;
    const __nv_bfloat16* p0 = As + (rowbase+gid)*lda + koff + tid*2;
    const __nv_bfloat16* p1 = As + (rowbase+gid+8)*lda + koff + tid*2;
    a[0]=*reinterpret_cast<const uint32_t*>(p0);
    a[1]=*reinterpret_cast<const uint32_t*>(p1);
    a[2]=*reinterpret_cast<const uint32_t*>(p0+8);
    a[3]=*reinterpret_cast<const uint32_t*>(p1+8);
}
// Bs[n][k] row-major : provides B[k][n]
__device__ __forceinline__ void loadB_NT(const __nv_bfloat16* Bs, int ldb, int colbase, int koff, int lane, uint32_t b[2]){
    int gid=lane>>2, tid=lane&3;
    const __nv_bfloat16* p = Bs + (colbase+gid)*ldb + koff + tid*2;
    b[0]=*reinterpret_cast<const uint32_t*>(p);
    b[1]=*reinterpret_cast<const uint32_t*>(p+8);
}
// Source src[k][n] row-major (k=rows, n=cols). ldmatrix.trans -> mma B operand B[k][n].
__device__ __forceinline__ void ldm_trans_B(const __nv_bfloat16* src, int ld, int koff, int ncol, int lane, uint32_t b[2]){
    const __nv_bfloat16* p = src + (koff + (lane&15))*ld + ncol;
    uint32_t a=(uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];":"=r"(b[0]),"=r"(b[1]):"r"(a));
}

__device__ __forceinline__ void load_tile_async(const __nv_bfloat16* g,int row_base,int S,int d,__nv_bfloat16* smem,int sld,int rows,int tid,int n){
    int total=rows*d;
    for(int i=tid*8;i<total;i+=n*8){
        int row=i/d, col=i-row*d;
        int grow=row_base+row;
        if(grow<S) cp_async_cg16(smem+row*sld+col, g+(long)grow*d+col);
        else *reinterpret_cast<float4*>(smem+row*sld+col)=make_float4(0.f,0.f,0.f,0.f);
    }
}

__global__ void compute_D_kernel(const __nv_bfloat16* dO, const __nv_bfloat16* O, float* Dout, long total_rows, int d){
    long wg = ((long)blockIdx.x*blockDim.x + threadIdx.x)/32;
    int lane = threadIdx.x & 31;
    if(wg>=total_rows) return;
    const __nv_bfloat16* dop=dO+wg*d; const __nv_bfloat16* op=O+wg*d;
    float sum=0.f;
    for(int e=lane;e<d;e+=32) sum += __bfloat162float(dop[e])*__bfloat162float(op[e]);
    for(int off=16;off>0;off>>=1) sum += __shfl_down_sync(0xffffffffu,sum,off);
    if(lane==0) Dout[wg]=sum;
}

__global__ void __launch_bounds__(NTHREAD) bwd_dkdv_kernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* dO, const float* L, const float* Dbuf,
    __nv_bfloat16* dK, __nv_bfloat16* dV, int S, int d, float scale)
{
    extern __shared__ __align__(16) char sm[];
    __nv_bfloat16* K_s=(__nv_bfloat16*)sm;
    __nv_bfloat16* V_s=K_s+BN*LD;
    __nv_bfloat16* Q_s=V_s+BN*LD;          // 2 buffers
    __nv_bfloat16* dO_s=Q_s+2*BM*LD;       // 2 buffers
    __nv_bfloat16* PT_s=dO_s+2*BM*LD;      // [BN][PBM]
    __nv_bfloat16* dST_s=PT_s+BN*PBM;      // [BN][PBM]
    float* L_s=(float*)(dST_s+BN*PBM);
    float* D_s=L_s+BM;

    int tid=threadIdx.x, lane=tid&31, warp_id=tid>>5;
    int warp_m=warp_id/WARP_N, warp_n=warp_id%WARP_N;
    int gid=lane>>2, tlane=lane&3;

    int key_base=blockIdx.x*BN, bh=blockIdx.y;
    const __nv_bfloat16* Qh=Q+(long)bh*S*d;
    const __nv_bfloat16* Kh=K+(long)bh*S*d;
    const __nv_bfloat16* Vh=V+(long)bh*S*d;
    const __nv_bfloat16* dOh=dO+(long)bh*S*d;
    const float* Lh=L+(long)bh*S;
    const float* Dh=Dbuf+(long)bh*S;
    __nv_bfloat16* dKh=dK+(long)bh*S*d;
    __nv_bfloat16* dVh=dV+(long)bh*S*d;

    int rowbase=16*warp_m, colbase=32*warp_n;
    int rb=16*warp_m, cb=64*warp_n;
    int num_q=(S+BM-1)/BM;

    float dV_acc[8][4], dK_acc[8][4];
    #pragma unroll
    for(int i=0;i<8;i++) for(int j=0;j<4;j++){dV_acc[i][j]=0.f;dK_acc[i][j]=0.f;}

    load_tile_async(Kh,key_base,S,d,K_s,LD,BN,tid,blockDim.x);
    load_tile_async(Vh,key_base,S,d,V_s,LD,BN,tid,blockDim.x);
    load_tile_async(Qh,0,S,d,Q_s,LD,BM,tid,blockDim.x);
    load_tile_async(dOh,0,S,d,dO_s,LD,BM,tid,blockDim.x);
    cp_async_commit(); cp_async_wait<0>(); __syncthreads();

    for(int qt=0; qt<num_q; qt++){
        int qb=qt*BM, cur=qt&1, nxt=(qt+1)&1;
        if(qt+1<num_q){
            load_tile_async(Qh,(qt+1)*BM,S,d,Q_s+nxt*BM*LD,LD,BM,tid,blockDim.x);
            load_tile_async(dOh,(qt+1)*BM,S,d,dO_s+nxt*BM*LD,LD,BM,tid,blockDim.x);
            cp_async_commit();
        }
        for(int i=tid;i<BM;i+=blockDim.x){int g=qb+i; L_s[i]=(g<S)?Lh[g]:0.f; D_s[i]=(g<S)?Dh[g]:0.f;}
        __syncthreads();
        __nv_bfloat16* Qc=Q_s+cur*BM*LD;
        __nv_bfloat16* dOc=dO_s+cur*BM*LD;

        // S^T = K @ Q^T
        float pt[4][4];
        #pragma unroll
        for(int nj=0;nj<4;nj++) for(int t=0;t<4;t++) pt[nj][t]=0.f;
        #pragma unroll
        for(int koff=0;koff<HD;koff+=16){
            uint32_t a[4]; loadA(K_s,LD,rowbase,koff,lane,a);
            #pragma unroll
            for(int nj=0;nj<4;nj++){ uint32_t b[2]; loadB_NT(Qc,LD,colbase+nj*8,koff,lane,b); mma16816(a,b,pt[nj]); }
        }
        #pragma unroll
        for(int nj=0;nj<4;nj++){
            int km0=rowbase+gid, km1=rowbase+gid+8;
            int qn0=colbase+nj*8+tlane*2, qn1=qn0+1;
            bool kv0=key_base+km0<S, kv1=key_base+km1<S;
            bool qv0=qb+qn0<S, qv1=qb+qn1<S;
            float p;
            p=(kv0&&qv0)?__expf(scale*pt[nj][0]-L_s[qn0]):0.f; pt[nj][0]=p; PT_s[km0*PBM+qn0]=__float2bfloat16(p);
            p=(kv0&&qv1)?__expf(scale*pt[nj][1]-L_s[qn1]):0.f; pt[nj][1]=p; PT_s[km0*PBM+qn1]=__float2bfloat16(p);
            p=(kv1&&qv0)?__expf(scale*pt[nj][2]-L_s[qn0]):0.f; pt[nj][2]=p; PT_s[km1*PBM+qn0]=__float2bfloat16(p);
            p=(kv1&&qv1)?__expf(scale*pt[nj][3]-L_s[qn1]):0.f; pt[nj][3]=p; PT_s[km1*PBM+qn1]=__float2bfloat16(p);
        }
        // dP^T = V @ dO^T
        float dpt[4][4];
        #pragma unroll
        for(int nj=0;nj<4;nj++) for(int t=0;t<4;t++) dpt[nj][t]=0.f;
        #pragma unroll
        for(int koff=0;koff<HD;koff+=16){
            uint32_t a[4]; loadA(V_s,LD,rowbase,koff,lane,a);
            #pragma unroll
            for(int nj=0;nj<4;nj++){ uint32_t b[2]; loadB_NT(dOc,LD,colbase+nj*8,koff,lane,b); mma16816(a,b,dpt[nj]); }
        }
        // dS^T = P^T*(dP^T - D)
        #pragma unroll
        for(int nj=0;nj<4;nj++){
            int km0=rowbase+gid, km1=rowbase+gid+8;
            int qn0=colbase+nj*8+tlane*2, qn1=qn0+1;
            float Dq0=D_s[qn0], Dq1=D_s[qn1];
            dST_s[km0*PBM+qn0]=__float2bfloat16(pt[nj][0]*(dpt[nj][0]-Dq0));
            dST_s[km0*PBM+qn1]=__float2bfloat16(pt[nj][1]*(dpt[nj][1]-Dq1));
            dST_s[km1*PBM+qn0]=__float2bfloat16(pt[nj][2]*(dpt[nj][2]-Dq0));
            dST_s[km1*PBM+qn1]=__float2bfloat16(pt[nj][3]*(dpt[nj][3]-Dq1));
        }
        __syncthreads();
        // dV += P^T @ dO   (B via ldmatrix.trans from dOc)
        #pragma unroll
        for(int koff=0;koff<BM;koff+=16){
            uint32_t a[4]; loadA(PT_s,PBM,rb,koff,lane,a);
            #pragma unroll
            for(int nj=0;nj<8;nj++){ uint32_t b[2]; ldm_trans_B(dOc,LD,koff,cb+nj*8,lane,b); mma16816(a,b,dV_acc[nj]); }
        }
        // dK += dS^T @ Q   (B via ldmatrix.trans from Qc)
        #pragma unroll
        for(int koff=0;koff<BM;koff+=16){
            uint32_t a[4]; loadA(dST_s,PBM,rb,koff,lane,a);
            #pragma unroll
            for(int nj=0;nj<8;nj++){ uint32_t b[2]; ldm_trans_B(Qc,LD,koff,cb+nj*8,lane,b); mma16816(a,b,dK_acc[nj]); }
        }
        if(qt+1<num_q){ cp_async_wait<0>(); __syncthreads(); }
    }
    #pragma unroll
    for(int nj=0;nj<8;nj++){
        int km0=rb+gid, km1=rb+gid+8;
        int e0=cb+nj*8+tlane*2, e1=e0+1;
        int gk0=key_base+km0, gk1=key_base+km1;
        if(gk0<S){
            dVh[(long)gk0*d+e0]=__float2bfloat16(dV_acc[nj][0]);
            dVh[(long)gk0*d+e1]=__float2bfloat16(dV_acc[nj][1]);
            dKh[(long)gk0*d+e0]=__float2bfloat16(scale*dK_acc[nj][0]);
            dKh[(long)gk0*d+e1]=__float2bfloat16(scale*dK_acc[nj][1]);
        }
        if(gk1<S){
            dVh[(long)gk1*d+e0]=__float2bfloat16(dV_acc[nj][2]);
            dVh[(long)gk1*d+e1]=__float2bfloat16(dV_acc[nj][3]);
            dKh[(long)gk1*d+e0]=__float2bfloat16(scale*dK_acc[nj][2]);
            dKh[(long)gk1*d+e1]=__float2bfloat16(scale*dK_acc[nj][3]);
        }
    }
}

__global__ void __launch_bounds__(NTHREAD) bwd_dq_kernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* dO, const float* L, const float* Dbuf,
    __nv_bfloat16* dQ, int S, int d, float scale)
{
    extern __shared__ __align__(16) char sm[];
    __nv_bfloat16* Q_s=(__nv_bfloat16*)sm;
    __nv_bfloat16* dO_s=Q_s+BM*LD;
    __nv_bfloat16* K_s=dO_s+BM*LD;     // 2 buffers
    __nv_bfloat16* V_s=K_s+2*BN*LD;    // 2 buffers
    __nv_bfloat16* dS_s=V_s+2*BN*LD;   // [BM][PBN]
    float* L_s=(float*)(dS_s+BM*PBN);
    float* D_s=L_s+BM;

    int tid=threadIdx.x, lane=tid&31, warp_id=tid>>5;
    int warp_m=warp_id/WARP_N, warp_n=warp_id%WARP_N;
    int gid=lane>>2, tlane=lane&3;

    int qb=blockIdx.x*BM, bh=blockIdx.y;
    const __nv_bfloat16* Qh=Q+(long)bh*S*d;
    const __nv_bfloat16* Kh=K+(long)bh*S*d;
    const __nv_bfloat16* Vh=V+(long)bh*S*d;
    const __nv_bfloat16* dOh=dO+(long)bh*S*d;
    const float* Lh=L+(long)bh*S;
    const float* Dh=Dbuf+(long)bh*S;
    __nv_bfloat16* dQh=dQ+(long)bh*S*d;

    int rowbase=16*warp_m, colbase=32*warp_n;
    int rb=16*warp_m, cb=64*warp_n;
    int num_k=(S+BN-1)/BN;

    float dQ_acc[8][4];
    #pragma unroll
    for(int i=0;i<8;i++) for(int j=0;j<4;j++) dQ_acc[i][j]=0.f;

    load_tile_async(Qh,qb,S,d,Q_s,LD,BM,tid,blockDim.x);
    load_tile_async(dOh,qb,S,d,dO_s,LD,BM,tid,blockDim.x);
    load_tile_async(Kh,0,S,d,K_s,LD,BN,tid,blockDim.x);
    load_tile_async(Vh,0,S,d,V_s,LD,BN,tid,blockDim.x);
    cp_async_commit(); cp_async_wait<0>(); __syncthreads();
    for(int i=tid;i<BM;i+=blockDim.x){int g=qb+i; L_s[i]=(g<S)?Lh[g]:0.f; D_s[i]=(g<S)?Dh[g]:0.f;}
    __syncthreads();

    for(int kt=0; kt<num_k; kt++){
        int kb=kt*BN, cur=kt&1, nxt=(kt+1)&1;
        if(kt+1<num_k){
            load_tile_async(Kh,(kt+1)*BN,S,d,K_s+nxt*BN*LD,LD,BN,tid,blockDim.x);
            load_tile_async(Vh,(kt+1)*BN,S,d,V_s+nxt*BN*LD,LD,BN,tid,blockDim.x);
            cp_async_commit();
        }
        __nv_bfloat16* Kc=K_s+cur*BN*LD;
        __nv_bfloat16* Vc=V_s+cur*BN*LD;

        // S = Q @ K^T
        float p[4][4];
        #pragma unroll
        for(int nj=0;nj<4;nj++) for(int t=0;t<4;t++) p[nj][t]=0.f;
        #pragma unroll
        for(int koff=0;koff<HD;koff+=16){
            uint32_t a[4]; loadA(Q_s,LD,rowbase,koff,lane,a);
            #pragma unroll
            for(int nj=0;nj<4;nj++){ uint32_t b[2]; loadB_NT(Kc,LD,colbase+nj*8,koff,lane,b); mma16816(a,b,p[nj]); }
        }
        #pragma unroll
        for(int nj=0;nj<4;nj++){
            int m0=rowbase+gid, m1=rowbase+gid+8;
            int n0=colbase+nj*8+tlane*2, n1=n0+1;
            bool qv0=qb+m0<S, qv1=qb+m1<S;
            bool kv0=kb+n0<S, kv1=kb+n1<S;
            float L0=L_s[m0], L1=L_s[m1];
            p[nj][0]=(qv0&&kv0)?__expf(scale*p[nj][0]-L0):0.f;
            p[nj][1]=(qv0&&kv1)?__expf(scale*p[nj][1]-L0):0.f;
            p[nj][2]=(qv1&&kv0)?__expf(scale*p[nj][2]-L1):0.f;
            p[nj][3]=(qv1&&kv1)?__expf(scale*p[nj][3]-L1):0.f;
        }
        // dP = dO @ V^T
        float dp[4][4];
        #pragma unroll
        for(int nj=0;nj<4;nj++) for(int t=0;t<4;t++) dp[nj][t]=0.f;
        #pragma unroll
        for(int koff=0;koff<HD;koff+=16){
            uint32_t a[4]; loadA(dO_s,LD,rowbase,koff,lane,a);
            #pragma unroll
            for(int nj=0;nj<4;nj++){ uint32_t b[2]; loadB_NT(Vc,LD,colbase+nj*8,koff,lane,b); mma16816(a,b,dp[nj]); }
        }
        // dS = P*(dP - D)
        #pragma unroll
        for(int nj=0;nj<4;nj++){
            int m0=rowbase+gid, m1=rowbase+gid+8;
            int n0=colbase+nj*8+tlane*2, n1=n0+1;
            float D0=D_s[m0], D1=D_s[m1];
            dS_s[m0*PBN+n0]=__float2bfloat16(p[nj][0]*(dp[nj][0]-D0));
            dS_s[m0*PBN+n1]=__float2bfloat16(p[nj][1]*(dp[nj][1]-D0));
            dS_s[m1*PBN+n0]=__float2bfloat16(p[nj][2]*(dp[nj][2]-D1));
            dS_s[m1*PBN+n1]=__float2bfloat16(p[nj][3]*(dp[nj][3]-D1));
        }
        __syncthreads();
        // dQ += dS @ K   (B via ldmatrix.trans from Kc)
        #pragma unroll
        for(int koff=0;koff<BN;koff+=16){
            uint32_t a[4]; loadA(dS_s,PBN,rb,koff,lane,a);
            #pragma unroll
            for(int nj=0;nj<8;nj++){ uint32_t b[2]; ldm_trans_B(Kc,LD,koff,cb+nj*8,lane,b); mma16816(a,b,dQ_acc[nj]); }
        }
        if(kt+1<num_k){ cp_async_wait<0>(); __syncthreads(); }
    }
    #pragma unroll
    for(int nj=0;nj<8;nj++){
        int m0=rb+gid, m1=rb+gid+8;
        int e0=cb+nj*8+tlane*2, e1=e0+1;
        int gq0=qb+m0, gq1=qb+m1;
        if(gq0<S){
            dQh[(long)gq0*d+e0]=__float2bfloat16(scale*dQ_acc[nj][0]);
            dQh[(long)gq0*d+e1]=__float2bfloat16(scale*dQ_acc[nj][1]);
        }
        if(gq1<S){
            dQh[(long)gq1*d+e0]=__float2bfloat16(scale*dQ_acc[nj][2]);
            dQh[(long)gq1*d+e1]=__float2bfloat16(scale*dQ_acc[nj][3]);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV){
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int B=(int)Q.size(0), H=(int)Q.size(1), S=(int)Q.size(2), d=(int)Q.size(3);
    if(S<=0) return;

    const __nv_bfloat16* Qp=(const __nv_bfloat16*)Q.data_ptr();
    const __nv_bfloat16* Kp=(const __nv_bfloat16*)K.data_ptr();
    const __nv_bfloat16* Vp=(const __nv_bfloat16*)V.data_ptr();
    const __nv_bfloat16* Op=(const __nv_bfloat16*)O.data_ptr();
    const __nv_bfloat16* dOp=(const __nv_bfloat16*)dO.data_ptr();
    const float* Lp=(const float*)L.data_ptr();
    __nv_bfloat16* dQp=(__nv_bfloat16*)dQ.data_ptr();
    __nv_bfloat16* dKp=(__nv_bfloat16*)dK.data_ptr();
    __nv_bfloat16* dVp=(__nv_bfloat16*)dV.data_ptr();

    int BH=B*H;
    long total_rows=(long)BH*S;
    float* Dbuf=nullptr;
    CUDA_CHECK(cudaMalloc(&Dbuf,(size_t)total_rows*sizeof(float)));

    cudaStream_t stream=(cudaStream_t)TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id);

    long dgrid=(total_rows+7)/8;
    compute_D_kernel<<<(unsigned)dgrid,256,0,stream>>>(dOp,Op,Dbuf,total_rows,d);

    size_t sm2=(size_t)(2*BN*LD + 4*BM*LD + 2*BN*PBM)*sizeof(__nv_bfloat16) + (size_t)(2*BM)*sizeof(float);
    size_t sm1=(size_t)(2*BM*LD + 4*BN*LD + BM*PBN)*sizeof(__nv_bfloat16) + (size_t)(2*BM)*sizeof(float);
    cudaFuncSetAttribute(bwd_dkdv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm2);
    cudaFuncSetAttribute(bwd_dq_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm1);

    float scale=1.0f/sqrtf((float)d);

    dim3 g2((S+BN-1)/BN, BH);
    bwd_dkdv_kernel<<<g2,NTHREAD,sm2,stream>>>(Qp,Kp,Vp,dOp,Lp,Dbuf,dKp,dVp,S,d,scale);

    dim3 g1((S+BM-1)/BM, BH);
    bwd_dq_kernel<<<g1,NTHREAD,sm1,stream>>>(Qp,Kp,Vp,dOp,Lp,Dbuf,dQp,S,d,scale);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaFree(Dbuf);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd