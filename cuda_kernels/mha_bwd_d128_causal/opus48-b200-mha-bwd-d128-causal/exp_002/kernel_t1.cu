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
static constexpr int NW = 4;      // warps
static constexpr int NTH = 128;   // threads

__device__ __forceinline__ uint32_t pk(bf16 a, bf16 b){
    __nv_bfloat162 v; v.x=a; v.y=b;
    return *reinterpret_cast<uint32_t*>(&v);
}

__device__ __forceinline__ void mma16816(
    float&c0,float&c1,float&c2,float&c3,
    uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1){
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        :"+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}

// A fragment: logical A(m,k). tr=false -> T[m*stride+k]; tr=true -> T[k*stride+m]
__device__ __forceinline__ void ldA(uint32_t r[4], const bf16* T, int stride,
                                     int m0, int k0, int lane, bool tr){
    int g=lane>>2; int t=(lane&3)*2;
    int mr[8]={m0+g,m0+g,m0+g+8,m0+g+8,m0+g,m0+g,m0+g+8,m0+g+8};
    int kc[8]={k0+t,k0+t+1,k0+t,k0+t+1,k0+t+8,k0+t+9,k0+t+8,k0+t+9};
    bf16 a[8];
    #pragma unroll
    for(int i=0;i<8;i++){
        int R=mr[i],C=kc[i];
        a[i]= tr ? T[C*stride+R] : T[R*stride+C];
    }
    r[0]=pk(a[0],a[1]); r[1]=pk(a[2],a[3]); r[2]=pk(a[4],a[5]); r[3]=pk(a[6],a[7]);
}

// B fragment: logical B(k,n). tr=false -> T[k*stride+n]; tr=true -> T[n*stride+k]
__device__ __forceinline__ void ldB(uint32_t r[2], const bf16* T, int stride,
                                     int k0, int n0, int lane, bool tr){
    int g=lane>>2; int t=(lane&3)*2;
    int kk[4]={k0+t,k0+t+1,k0+t+8,k0+t+9};
    int n=n0+g;
    bf16 b[4];
    #pragma unroll
    for(int i=0;i<4;i++){
        int K=kk[i];
        b[i]= tr ? T[n*stride+K] : T[K*stride+n];
    }
    r[0]=pk(b[0],b[1]); r[1]=pk(b[2],b[3]);
}

template<int ROWS>
__device__ __forceinline__ void load_tile(bf16* smem, const bf16* g, int valid_rows,
                                           int tid, int nthreads){
    const int VEC=8;
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

// D_i = sum_e dO_ie * O_ie  (warp per row)
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

// -------- dK / dV kernel: block = one KV tile, loops over Q tiles --------
__launch_bounds__(NTH,2)
__global__ void bwd_dkdv(const bf16* Q,const bf16* K,const bf16* V,const bf16* dO,
                         const float* Lg,const float* Dg,
                         bf16* dKout,bf16* dVout,int S,float scale){
    int bh = blockIdx.y;
    int kv_start = blockIdx.x*BN;
    if(kv_start>=S) return;
    int krows = min(BN, S-kv_start);
    const int tid = threadIdx.x;
    const int lane = tid&31;
    const int warp = tid>>5;

    extern __shared__ char smem[];
    bf16* Qs  = (bf16*)smem;
    bf16* dOs = Qs  + BM*HD;
    bf16* Ks  = dOs + BM*HD;
    bf16* Vs  = Ks  + BN*HD;
    bf16* Ps  = Vs  + BN*HD;
    bf16* dSs = Ps  + BM*BN;
    float* Ls = (float*)(dSs + BM*BN);
    float* Ds = Ls + BM;

    size_t bh_off = (size_t)bh*S*HD;
    size_t bh_l   = (size_t)bh*S;

    load_tile<BN>(Ks, K + bh_off + (size_t)kv_start*HD, krows, tid, NTH);
    load_tile<BN>(Vs, V + bh_off + (size_t)kv_start*HD, krows, tid, NTH);

    // persistent accumulators for this warp's KV band (16 rows) x 128 d
    float dVacc[16][4];
    float dKacc[16][4];
    #pragma unroll
    for(int i=0;i<16;i++)
        #pragma unroll
        for(int e=0;e<4;e++){ dVacc[i][e]=0.f; dKacc[i][e]=0.f; }
    __syncthreads();

    int m0q = warp*16; // query band for score phase
    int m0k = warp*16; // kv band for dV/dK phase

    for(int q_start=kv_start; q_start<S; q_start+=BM){
        int qrows = min(BM, S-q_start);
        load_tile<BM>(Qs,  Q  + bh_off + (size_t)q_start*HD, qrows, tid, NTH);
        load_tile<BM>(dOs, dO + bh_off + (size_t)q_start*HD, qrows, tid, NTH);
        for(int i=tid;i<BM;i+=NTH){
            int qg=q_start+i;
            Ls[i]=(qg<S)?Lg[bh_l+qg]:0.f;
            Ds[i]=(qg<S)?Dg[bh_l+qg]:0.f;
        }
        __syncthreads();

        // ---- score S = Q@K^T, dP = dO@V^T ----
        float Sacc[8][4], dPacc[8][4];
        #pragma unroll
        for(int nt=0;nt<8;nt++)
            #pragma unroll
            for(int e=0;e<4;e++){ Sacc[nt][e]=0.f; dPacc[nt][e]=0.f; }

        #pragma unroll
        for(int ks=0;ks<HD/16;ks++){
            int k0=ks*16;
            uint32_t qa[4], da[4];
            ldA(qa,Qs, HD,m0q,k0,lane,false);
            ldA(da,dOs,HD,m0q,k0,lane,false);
            #pragma unroll
            for(int nt=0;nt<8;nt++){
                int n0=nt*8;
                uint32_t kb[2], vb[2];
                ldB(kb,Ks,HD,k0,n0,lane,true);
                ldB(vb,Vs,HD,k0,n0,lane,true);
                mma16816(Sacc[nt][0],Sacc[nt][1],Sacc[nt][2],Sacc[nt][3],
                         qa[0],qa[1],qa[2],qa[3],kb[0],kb[1]);
                mma16816(dPacc[nt][0],dPacc[nt][1],dPacc[nt][2],dPacc[nt][3],
                         da[0],da[1],da[2],da[3],vb[0],vb[1]);
            }
        }

        // ---- P and dS ----
        #pragma unroll
        for(int nt=0;nt<8;nt++){
            #pragma unroll
            for(int e=0;e<4;e++){
                int g=lane>>2, t=(lane&3);
                int rl=(e<2)?g:g+8;
                int cl=nt*8 + t*2 + (e&1);
                int qit=m0q+rl;
                int kit=cl;
                int qg=q_start+qit, kg=kv_start+kit;
                float p=0.f, ds=0.f;
                if(qg<S && kg<S && qg>=kg){
                    float L=Ls[qit], D=Ds[qit];
                    p = __expf(scale*Sacc[nt][e]-L);
                    ds = p*(dPacc[nt][e]-D);
                }
                Ps[qit*BN+kit]  = __float2bfloat16(p);
                dSs[qit*BN+kit] = __float2bfloat16(ds);
            }
        }
        __syncthreads();

        // ---- dV += P^T@dO , dK += dS^T@Q ----
        #pragma unroll
        for(int ks=0;ks<BM/16;ks++){
            int k0=ks*16;
            #pragma unroll
            for(int nt=0;nt<16;nt++){
                int n0=nt*8;
                uint32_t pa[4], sa[4], ob[2], qb[2];
                ldA(pa,Ps, BN,m0k,k0,lane,true);
                ldA(sa,dSs,BN,m0k,k0,lane,true);
                ldB(ob,dOs,HD,k0,n0,lane,false);
                ldB(qb,Qs, HD,k0,n0,lane,false);
                mma16816(dVacc[nt][0],dVacc[nt][1],dVacc[nt][2],dVacc[nt][3],
                         pa[0],pa[1],pa[2],pa[3],ob[0],ob[1]);
                mma16816(dKacc[nt][0],dKacc[nt][1],dKacc[nt][2],dKacc[nt][3],
                         sa[0],sa[1],sa[2],sa[3],qb[0],qb[1]);
            }
        }
        __syncthreads();
    }

    // write dV, dK
    #pragma unroll
    for(int nt=0;nt<16;nt++){
        #pragma unroll
        for(int e=0;e<4;e++){
            int g=lane>>2, t=(lane&3);
            int rl=(e<2)?g:g+8;
            int cl=nt*8 + t*2 + (e&1);
            int kit=m0k+rl;
            int didx=cl;
            int kg=kv_start+kit;
            if(kg<S){
                size_t idx=bh_off+(size_t)kg*HD+didx;
                dVout[idx]=__float2bfloat16(dVacc[nt][e]);
                dKout[idx]=__float2bfloat16(scale*dKacc[nt][e]);
            }
        }
    }
}

// -------- dQ kernel: block = one Q tile, loops over KV tiles --------
__launch_bounds__(NTH,3)
__global__ void bwd_dq(const bf16* Q,const bf16* K,const bf16* V,const bf16* dO,
                       const float* Lg,const float* Dg,
                       bf16* dQout,int S,float scale){
    int bh = blockIdx.y;
    int q_start = blockIdx.x*BM;
    if(q_start>=S) return;
    int qrows = min(BM, S-q_start);
    const int tid = threadIdx.x;
    const int lane = tid&31;
    const int warp = tid>>5;

    extern __shared__ char smem[];
    bf16* Qs  = (bf16*)smem;
    bf16* dOs = Qs  + BM*HD;
    bf16* Ks  = dOs + BM*HD;
    bf16* Vs  = Ks  + BN*HD;
    bf16* dSs = Vs  + BN*HD;
    float* Ls = (float*)(dSs + BM*BN);
    float* Ds = Ls + BM;

    size_t bh_off = (size_t)bh*S*HD;
    size_t bh_l   = (size_t)bh*S;

    load_tile<BM>(Qs,  Q  + bh_off + (size_t)q_start*HD, qrows, tid, NTH);
    load_tile<BM>(dOs, dO + bh_off + (size_t)q_start*HD, qrows, tid, NTH);
    for(int i=tid;i<BM;i+=NTH){
        int qg=q_start+i;
        Ls[i]=(qg<S)?Lg[bh_l+qg]:0.f;
        Ds[i]=(qg<S)?Dg[bh_l+qg]:0.f;
    }

    float dQacc[16][4];
    #pragma unroll
    for(int i=0;i<16;i++)
        #pragma unroll
        for(int e=0;e<4;e++) dQacc[i][e]=0.f;
    __syncthreads();

    int m0q = warp*16; // query band

    for(int kv_start=0; kv_start<=q_start && kv_start<S; kv_start+=BN){
        int krows = min(BN, S-kv_start);
        load_tile<BN>(Ks, K + bh_off + (size_t)kv_start*HD, krows, tid, NTH);
        load_tile<BN>(Vs, V + bh_off + (size_t)kv_start*HD, krows, tid, NTH);
        __syncthreads();

        float Sacc[8][4], dPacc[8][4];
        #pragma unroll
        for(int nt=0;nt<8;nt++)
            #pragma unroll
            for(int e=0;e<4;e++){ Sacc[nt][e]=0.f; dPacc[nt][e]=0.f; }

        #pragma unroll
        for(int ks=0;ks<HD/16;ks++){
            int k0=ks*16;
            uint32_t qa[4], da[4];
            ldA(qa,Qs, HD,m0q,k0,lane,false);
            ldA(da,dOs,HD,m0q,k0,lane,false);
            #pragma unroll
            for(int nt=0;nt<8;nt++){
                int n0=nt*8;
                uint32_t kb[2], vb[2];
                ldB(kb,Ks,HD,k0,n0,lane,true);
                ldB(vb,Vs,HD,k0,n0,lane,true);
                mma16816(Sacc[nt][0],Sacc[nt][1],Sacc[nt][2],Sacc[nt][3],
                         qa[0],qa[1],qa[2],qa[3],kb[0],kb[1]);
                mma16816(dPacc[nt][0],dPacc[nt][1],dPacc[nt][2],dPacc[nt][3],
                         da[0],da[1],da[2],da[3],vb[0],vb[1]);
            }
        }

        // dS
        #pragma unroll
        for(int nt=0;nt<8;nt++){
            #pragma unroll
            for(int e=0;e<4;e++){
                int g=lane>>2, t=(lane&3);
                int rl=(e<2)?g:g+8;
                int cl=nt*8 + t*2 + (e&1);
                int qit=m0q+rl;
                int kit=cl;
                int qg=q_start+qit, kg=kv_start+kit;
                float ds=0.f;
                if(qg<S && kg<S && qg>=kg){
                    float L=Ls[qit], D=Ds[qit];
                    float p = __expf(scale*Sacc[nt][e]-L);
                    ds = p*(dPacc[nt][e]-D);
                }
                dSs[qit*BN+kit] = __float2bfloat16(ds);
            }
        }
        __syncthreads();

        // dQ += dS @ K
        #pragma unroll
        for(int ks=0;ks<BN/16;ks++){
            int k0=ks*16;
            #pragma unroll
            for(int nt=0;nt<16;nt++){
                int n0=nt*8;
                uint32_t sa[4], kb[2];
                ldA(sa,dSs,BN,m0q,k0,lane,false);
                ldB(kb,Ks, HD,k0,n0,lane,false);
                mma16816(dQacc[nt][0],dQacc[nt][1],dQacc[nt][2],dQacc[nt][3],
                         sa[0],sa[1],sa[2],sa[3],kb[0],kb[1]);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for(int nt=0;nt<16;nt++){
        #pragma unroll
        for(int e=0;e<4;e++){
            int g=lane>>2, t=(lane&3);
            int rl=(e<2)?g:g+8;
            int cl=nt*8 + t*2 + (e&1);
            int qit=m0q+rl;
            int didx=cl;
            int qg=q_start+qit;
            if(qg<S){
                size_t idx=bh_off+(size_t)qg*HD+didx;
                dQout[idx]=__float2bfloat16(scale*dQacc[nt][e]);
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
    {
        int threads=256;
        int warps_per_block = threads/32;
        int blocks = (int)((total_rows + warps_per_block - 1)/warps_per_block);
        compute_D<<<blocks, threads, 0, stream>>>(Op, dOp, Dscratch, (int)total_rows);
        CUDA_CHECK(cudaGetLastError());
    }

    size_t sh_dkdv = (size_t)(4*BM*HD + 2*BM*BN)*sizeof(bf16) + (size_t)(2*BM)*sizeof(float);
    size_t sh_dq   = (size_t)(4*BM*HD + 1*BM*BN)*sizeof(bf16) + (size_t)(2*BM)*sizeof(float);

    CUDA_CHECK(cudaFuncSetAttribute((const void*)bwd_dkdv,
               cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh_dkdv));
    CUDA_CHECK(cudaFuncSetAttribute((const void*)bwd_dq,
               cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh_dq));

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