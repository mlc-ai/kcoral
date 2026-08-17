#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <math.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
    }                                                              \
} while(0)

namespace mha_d128_causal {

constexpr int D = 128, BM = 128, BN = 64, NTHREADS = 256;
constexpr int NB8  = BN / 8;    // 8
constexpr int KBQK = D / 16;    // 8
constexpr int DBLK = D / 8;     // 16
constexpr int KBPV = BN / 16;   // 4
constexpr int QST = D + 8;
constexpr int KST = D + 8;
constexpr int VST = D + 8;
constexpr int PST = BN + 8;

__device__ __forceinline__ uint32_t smem_u32(const void* p){ return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void ldm_x4(uint32_t&r0,uint32_t&r1,uint32_t&r2,uint32_t&r3,uint32_t a){
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        :"=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3):"r"(a));
}
__device__ __forceinline__ void ldm_x2(uint32_t&r0,uint32_t&r1,uint32_t a){
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n":"=r"(r0),"=r"(r1):"r"(a));
}
__device__ __forceinline__ void ldm_x2_t(uint32_t&r0,uint32_t&r1,uint32_t a){
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n":"=r"(r0),"=r"(r1):"r"(a));
}
__device__ __forceinline__ void mma16816(float&d0,float&d1,float&d2,float&d3,
    uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1){
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        :"+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
__device__ __forceinline__ float redmax4(float v){
    v=fmaxf(v,__shfl_xor_sync(0xffffffffu,v,1)); v=fmaxf(v,__shfl_xor_sync(0xffffffffu,v,2)); return v;
}
__device__ __forceinline__ float redsum4(float v){
    v+=__shfl_xor_sync(0xffffffffu,v,1); v+=__shfl_xor_sync(0xffffffffu,v,2); return v;
}
__device__ __forceinline__ uint32_t pack2(float a,float b){
    __nv_bfloat162 v=__floats2bfloat162_rn(a,b); uint32_t r; __builtin_memcpy(&r,&v,4); return r;
}
__device__ __forceinline__ void cp_commit(){ asm volatile("cp.async.commit_group;\n"); }

__device__ __forceinline__ void prefetch(const __nv_bfloat16* base,int row0,__nv_bfloat16* dst,
                                          int S,int nrows,int tid,int dst_stride){
    const uint4* src=reinterpret_cast<const uint4*>(base);
    uint4* d=reinterpret_cast<uint4*>(dst);
    int su4=dst_stride/8;
    int total=nrows*16;
    for(int v=tid;v<total;v+=NTHREADS){
        int r=v>>4, c=v&15, s=row0+r;
        uint32_t da=smem_u32(&d[r*su4+c]);
        const uint4* sp=src+(int64_t)s*16+c;
        int sz=(s<S)?16:0;
        asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;\n"::"r"(da),"l"(sp),"r"(sz));
    }
}

__global__ void __launch_bounds__(NTHREADS,1) attn_kernel(
        const __nv_bfloat16* __restrict__ Q,const __nv_bfloat16* __restrict__ K,
        const __nv_bfloat16* __restrict__ V,__nv_bfloat16* __restrict__ O,
        float* __restrict__ LSE,int B,int H,int S,float scale){
    extern __shared__ char smem[];
    __nv_bfloat16* Qsh=reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* Ksh=Qsh+BM*QST;              // 2 stages
    __nv_bfloat16* Vsh=Ksh+2*BN*KST;            // 2 stages
    __nv_bfloat16* Psh=Vsh+2*BN*VST;

    int tid=threadIdx.x, warp=tid>>5, lane=tid&31;
    int gid=lane>>2, tg=lane&3, R=warp*16;
    int b=blockIdx.z, h=blockIdx.y, qt=blockIdx.x, q0=qt*BM;

    int64_t head=((int64_t)(b*H+h))*S*D;
    const __nv_bfloat16 *Qb=Q+head,*Kb=K+head,*Vb=V+head;
    __nv_bfloat16* Ob=O+head;
    float* LSEb=LSE+((int64_t)(b*H+h))*S;

    float o[DBLK][4];
    #pragma unroll
    for(int db=0;db<DBLK;db++){o[db][0]=o[db][1]=o[db][2]=o[db][3]=0.f;}
    float m_a=-INFINITY,m_b=-INFINITY,l_a=0.f,l_b=0.f;

    // load Q (its own group)
    prefetch(Qb,q0,Qsh,S,BM,tid,QST); cp_commit();
    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();

    int qmax=q0+BM-1; if(qmax>S-1)qmax=S-1;
    int num_kt=qmax/BN+1;

    // prime tile 0
    prefetch(Kb,0,Ksh,S,BN,tid,KST);
    prefetch(Vb,0,Vsh,S,BN,tid,VST);
    cp_commit();

    for(int jt=0;jt<num_kt;jt++){
        int cur=jt&1, nxt=(jt+1)&1;
        __nv_bfloat16* Kbuf=Ksh+cur*BN*KST;
        __nv_bfloat16* Vbuf=Vsh+cur*BN*VST;

        bool has_next=(jt+1<num_kt);
        if(has_next){
            int k1=(jt+1)*BN;
            prefetch(Kb,k1,Ksh+nxt*BN*KST,S,BN,tid,KST);
            prefetch(Vb,k1,Vsh+nxt*BN*VST,S,BN,tid,VST);
            cp_commit();
            asm volatile("cp.async.wait_group 1;\n");
        } else {
            asm volatile("cp.async.wait_group 0;\n");
        }
        __syncthreads();

        int k0=jt*BN;

        // ---- S = Q@K^T ----
        float s[NB8][4];
        #pragma unroll
        for(int n=0;n<NB8;n++){s[n][0]=s[n][1]=s[n][2]=s[n][3]=0.f;}
        #pragma unroll
        for(int kb=0;kb<KBQK;kb++){
            uint32_t qa0,qa1,qa2,qa3;
            {int off=(R+(lane&15))*QST + kb*16 + (lane>>4)*8;
             ldm_x4(qa0,qa1,qa2,qa3,smem_u32(&Qsh[off]));}
            #pragma unroll
            for(int n=0;n<NB8;n++){
                uint32_t kb0,kb1;
                {int off=(n*8+(lane&7))*KST + kb*16 + ((lane&8)?8:0);
                 ldm_x2(kb0,kb1,smem_u32(&Kbuf[off]));}
                mma16816(s[n][0],s[n][1],s[n][2],s[n][3],qa0,qa1,qa2,qa3,kb0,kb1);
            }
        }

        int qA=q0+R+gid, qB=qA+8;
        #pragma unroll
        for(int n=0;n<NB8;n++){
            s[n][0]*=scale;s[n][1]*=scale;s[n][2]*=scale;s[n][3]*=scale;
            int c0=k0+n*8+2*tg, c1=c0+1;
            if(!(c0<=qA&&c0<S))s[n][0]=-INFINITY;
            if(!(c1<=qA&&c1<S))s[n][1]=-INFINITY;
            if(!(c0<=qB&&c0<S))s[n][2]=-INFINITY;
            if(!(c1<=qB&&c1<S))s[n][3]=-INFINITY;
        }

        float lma=-INFINITY,lmb=-INFINITY;
        #pragma unroll
        for(int n=0;n<NB8;n++){
            lma=fmaxf(lma,fmaxf(s[n][0],s[n][1]));
            lmb=fmaxf(lmb,fmaxf(s[n][2],s[n][3]));
        }
        lma=redmax4(lma); lmb=redmax4(lmb);

        float mnew_a=fmaxf(m_a,lma), mnew_b=fmaxf(m_b,lmb);
        float corr_a,corr_b;
        if(mnew_a==-INFINITY)corr_a=1.f; else if(m_a==-INFINITY)corr_a=0.f; else corr_a=__expf(m_a-mnew_a);
        if(mnew_b==-INFINITY)corr_b=1.f; else if(m_b==-INFINITY)corr_b=0.f; else corr_b=__expf(m_b-mnew_b);

        int rowa=R+gid, rowb=R+gid+8;
        float rsa=0.f,rsb=0.f;
        #pragma unroll
        for(int n=0;n<NB8;n++){
            float p0a=(s[n][0]==-INFINITY)?0.f:__expf(s[n][0]-mnew_a);
            float p1a=(s[n][1]==-INFINITY)?0.f:__expf(s[n][1]-mnew_a);
            float p0b=(s[n][2]==-INFINITY)?0.f:__expf(s[n][2]-mnew_b);
            float p1b=(s[n][3]==-INFINITY)?0.f:__expf(s[n][3]-mnew_b);
            rsa+=p0a+p1a; rsb+=p0b+p1b;
            int col=n*8+2*tg;
            Psh[rowa*PST+col]  =__float2bfloat16(p0a);
            Psh[rowa*PST+col+1]=__float2bfloat16(p1a);
            Psh[rowb*PST+col]  =__float2bfloat16(p0b);
            Psh[rowb*PST+col+1]=__float2bfloat16(p1b);
        }
        rsa=redsum4(rsa); rsb=redsum4(rsb);
        l_a=l_a*corr_a+rsa; l_b=l_b*corr_b+rsb;
        m_a=mnew_a; m_b=mnew_b;

        #pragma unroll
        for(int db=0;db<DBLK;db++){o[db][0]*=corr_a;o[db][1]*=corr_a;o[db][2]*=corr_b;o[db][3]*=corr_b;}

        __syncwarp();

        // ---- O += P@V ----
        #pragma unroll
        for(int kb=0;kb<KBPV;kb++){
            uint32_t pa0,pa1,pa2,pa3;
            {int off=(R+(lane&15))*PST + kb*16 + (lane>>4)*8;
             ldm_x4(pa0,pa1,pa2,pa3,smem_u32(&Psh[off]));}
            #pragma unroll
            for(int db=0;db<DBLK;db++){
                uint32_t vb0,vb1;
                {int off=(kb*16+(lane&15))*VST + db*8;
                 ldm_x2_t(vb0,vb1,smem_u32(&Vbuf[off]));}
                mma16816(o[db][0],o[db][1],o[db][2],o[db][3],pa0,pa1,pa2,pa3,vb0,vb1);
            }
        }
    }

    float inv_a=(l_a>0.f)?(1.f/l_a):0.f;
    float inv_b=(l_b>0.f)?(1.f/l_b):0.f;
    int qA=q0+R+gid, qB=qA+8;
    if(qA<S){
        #pragma unroll
        for(int db=0;db<DBLK;db++){int d=db*8+2*tg;
            *reinterpret_cast<uint32_t*>(&Ob[(int64_t)qA*D+d])=pack2(o[db][0]*inv_a,o[db][1]*inv_a);}
        if(tg==0) LSEb[qA]=m_a+logf(l_a);
    }
    if(qB<S){
        #pragma unroll
        for(int db=0;db<DBLK;db++){int d=db*8+2*tg;
            *reinterpret_cast<uint32_t*>(&Ob[(int64_t)qB*D+d])=pack2(o[db][2]*inv_b,o[db][3]*inv_b);}
        if(tg==0) LSEb[qB]=m_b+logf(l_b);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int Bb=(int)Q.size(0), Hh=(int)Q.size(1), Ss=(int)Q.size(2), Dd=(int)Q.size(3);
    float scale=1.0f/sqrtf((float)Dd);

    const __nv_bfloat16* Qp=static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* Kp=static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* Vp=static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* Op=static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSEp=static_cast<float*>(LSE.data_ptr());

    int nqt=(Ss+BM-1)/BM;
    dim3 grid(nqt,Hh,Bb), block(NTHREADS);

    size_t smem=(size_t)BM*QST*2 + (size_t)2*BN*KST*2 + (size_t)2*BN*VST*2 + (size_t)BM*PST*2;
    cudaFuncSetAttribute(attn_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem);

    cudaStream_t stream=
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type,Q.device().device_id));
    attn_kernel<<<grid,block,smem,stream>>>(Qp,Kp,Vp,Op,LSEp,Bb,Hh,Ss,scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_d128_causal::run);

}  // namespace mha_d128_causal