#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <math.h>
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

namespace mha_causal {

constexpr int D   = 128;
constexpr int BM  = 64;    // query rows per block
constexpr int BN  = 64;    // keys per tile
constexpr int WARPS = 4;   // 4 warps -> 64 rows (16 each)
constexpr int THREADS = 32*WARPS;
constexpr int NNT = BN/8;    // QK n-tiles          = 8
constexpr int NKC = D/16;    // QK k-chunks over D  = 8
constexpr int NKT = BN/16;   // PV k-chunks (keys)  = 4
constexpr int NDT = D/8;     // PV n-tiles over D   = 16
constexpr unsigned FULL = 0xffffffffu;

__device__ __forceinline__ float ex2(float x){
    float y; asm volatile("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x)); return y;
}

__device__ __forceinline__ void mma16816(
    float &c0,float &c1,float &c2,float &c3,
    uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,
    uint32_t b0,uint32_t b1){
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
      :"+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
      :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}

__global__ __launch_bounds__(THREADS) void attn_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B,int H,int S,float scale_log2)
{
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __nv_bfloat16* Qs = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* Ks = Qs + BM*D;       // 8192
    __nv_bfloat16* Vst= Ks + BN*D;       // 8192 (transposed: [d][key])
    __nv_bfloat16* Ps = Vst + D*BN;      // 8192 -> Ps size 4096

    const int warp = threadIdx.x>>5;
    const int lane = threadIdx.x&31;
    const int gid  = lane>>2;   // 0..7
    const int tig  = lane&3;    // 0..3

    const int h = blockIdx.y;
    const int b = blockIdx.z;
    const int q0 = blockIdx.x*BM;

    const int64_t bh = (int64_t)(b*H + h);
    const __nv_bfloat16* Qg = Q + bh*S*D;
    const __nv_bfloat16* Kg = K + bh*S*D;
    const __nv_bfloat16* Vg = V + bh*S*D;
    __nv_bfloat16* Og = O + bh*S*D;
    float* LSEg = LSE + bh*S;

    // ---- load Q tile once ----
    for(int i=threadIdx.x;i<(BM*D)/8;i+=THREADS){
        int e=i*8; int row=e>>7; int col=e&127;
        int grow=q0+row;
        int4 v = (grow<S) ? *reinterpret_cast<const int4*>(Qg+(int64_t)grow*D+col)
                          : make_int4(0,0,0,0);
        *reinterpret_cast<int4*>(&Qs[row*D+col]) = v;
    }

    // persistent accumulators
    float Oacc[NDT][4];
    #pragma unroll
    for(int dt=0;dt<NDT;dt++){Oacc[dt][0]=0;Oacc[dt][1]=0;Oacc[dt][2]=0;Oacc[dt][3]=0;}
    float m0=-INFINITY,m1=-INFINITY,l0=0.f,l1=0.f;

    const bool warp_active = (q0 + warp*16) < S;
    const int warp_max_row = q0 + warp*16 + 15;
    const int global_row0 = q0 + warp*16 + gid;
    const int global_row1 = global_row0 + 8;

    int last_key = q0+BM-1; if(last_key>S-1) last_key=S-1;
    int num_tiles = last_key/BN + 1;

    for(int t=0;t<num_tiles;t++){
        int kv_start = t*BN;
        __syncthreads();
        // load K tile [key][d]
        for(int i=threadIdx.x;i<(BN*D)/8;i+=THREADS){
            int e=i*8; int key=e>>7; int col=e&127;
            int gk=kv_start+key;
            int4 v = (gk<S) ? *reinterpret_cast<const int4*>(Kg+(int64_t)gk*D+col)
                            : make_int4(0,0,0,0);
            *reinterpret_cast<int4*>(&Ks[key*D+col]) = v;
        }
        // load V tile transposed -> Vst[d][key]
        for(int i=threadIdx.x;i<(BN*D)/8;i+=THREADS){
            int e=i*8; int key=e>>7; int col=e&127;
            int gk=kv_start+key;
            int4 v = (gk<S) ? *reinterpret_cast<const int4*>(Vg+(int64_t)gk*D+col)
                            : make_int4(0,0,0,0);
            const __nv_bfloat16* vb = reinterpret_cast<const __nv_bfloat16*>(&v);
            #pragma unroll
            for(int j=0;j<8;j++) Vst[(col+j)*BN+key]=vb[j];
        }
        __syncthreads();

        if(!warp_active) continue;
        if(kv_start > warp_max_row) continue;

        // ---- QK^T ----
        float s[NNT][4];
        int qrow0 = warp*16+gid;
        int qrow1 = qrow0+8;
        #pragma unroll
        for(int nt=0;nt<NNT;nt++){
            float c0=0,c1=0,c2=0,c3=0;
            #pragma unroll
            for(int kc=0;kc<NKC;kc++){
                int cbase = kc*16 + tig*2;
                uint32_t a0=*reinterpret_cast<uint32_t*>(&Qs[qrow0*D+cbase]);
                uint32_t a2=*reinterpret_cast<uint32_t*>(&Qs[qrow0*D+cbase+8]);
                uint32_t a1=*reinterpret_cast<uint32_t*>(&Qs[qrow1*D+cbase]);
                uint32_t a3=*reinterpret_cast<uint32_t*>(&Qs[qrow1*D+cbase+8]);
                int key = nt*8+gid;
                uint32_t b0=*reinterpret_cast<uint32_t*>(&Ks[key*D+cbase]);
                uint32_t b1=*reinterpret_cast<uint32_t*>(&Ks[key*D+cbase+8]);
                mma16816(c0,c1,c2,c3,a0,a1,a2,a3,b0,b1);
            }
            s[nt][0]=c0*scale_log2; s[nt][1]=c1*scale_log2;
            s[nt][2]=c2*scale_log2; s[nt][3]=c3*scale_log2;
        }

        // ---- causal + range mask ----
        #pragma unroll
        for(int nt=0;nt<NNT;nt++){
            int kc0 = kv_start + nt*8 + tig*2;
            int kc1 = kc0+1;
            if(kc0>global_row0 || kc0>=S) s[nt][0]=-INFINITY;
            if(kc1>global_row0 || kc1>=S) s[nt][1]=-INFINITY;
            if(kc0>global_row1 || kc0>=S) s[nt][2]=-INFINITY;
            if(kc1>global_row1 || kc1>=S) s[nt][3]=-INFINITY;
        }

        // ---- online softmax ----
        float lm0=-INFINITY,lm1=-INFINITY;
        #pragma unroll
        for(int nt=0;nt<NNT;nt++){
            lm0=fmaxf(lm0,fmaxf(s[nt][0],s[nt][1]));
            lm1=fmaxf(lm1,fmaxf(s[nt][2],s[nt][3]));
        }
        lm0=fmaxf(lm0,__shfl_xor_sync(FULL,lm0,1)); lm0=fmaxf(lm0,__shfl_xor_sync(FULL,lm0,2));
        lm1=fmaxf(lm1,__shfl_xor_sync(FULL,lm1,1)); lm1=fmaxf(lm1,__shfl_xor_sync(FULL,lm1,2));

        float mn0=fmaxf(m0,lm0), mn1=fmaxf(m1,lm1);
        float corr0=ex2(m0-mn0), corr1=ex2(m1-mn1);

        #pragma unroll
        for(int dt=0;dt<NDT;dt++){
            Oacc[dt][0]*=corr0; Oacc[dt][1]*=corr0;
            Oacc[dt][2]*=corr1; Oacc[dt][3]*=corr1;
        }
        l0*=corr0; l1*=corr1;

        float ps0=0,ps1=0;
        #pragma unroll
        for(int nt=0;nt<NNT;nt++){
            float p0=ex2(s[nt][0]-mn0);
            float p1=ex2(s[nt][1]-mn0);
            float p2=ex2(s[nt][2]-mn1);
            float p3=ex2(s[nt][3]-mn1);
            ps0+=p0+p1; ps1+=p2+p3;
            int col = nt*8+tig*2;
            Ps[qrow0*BN+col  ]=__float2bfloat16(p0);
            Ps[qrow0*BN+col+1]=__float2bfloat16(p1);
            Ps[qrow1*BN+col  ]=__float2bfloat16(p2);
            Ps[qrow1*BN+col+1]=__float2bfloat16(p3);
        }
        ps0+=__shfl_xor_sync(FULL,ps0,1); ps0+=__shfl_xor_sync(FULL,ps0,2);
        ps1+=__shfl_xor_sync(FULL,ps1,1); ps1+=__shfl_xor_sync(FULL,ps1,2);
        l0+=ps0; l1+=ps1; m0=mn0; m1=mn1;

        __syncwarp();

        // ---- P @ V ----
        int prow0=qrow0, prow1=qrow1;
        #pragma unroll
        for(int dt=0;dt<NDT;dt++){
            float o0=Oacc[dt][0],o1=Oacc[dt][1],o2=Oacc[dt][2],o3=Oacc[dt][3];
            int dcol=dt*8+gid;
            #pragma unroll
            for(int kk=0;kk<NKT;kk++){
                int kbase=kk*16+tig*2;
                uint32_t a0=*reinterpret_cast<uint32_t*>(&Ps[prow0*BN+kbase]);
                uint32_t a2=*reinterpret_cast<uint32_t*>(&Ps[prow0*BN+kbase+8]);
                uint32_t a1=*reinterpret_cast<uint32_t*>(&Ps[prow1*BN+kbase]);
                uint32_t a3=*reinterpret_cast<uint32_t*>(&Ps[prow1*BN+kbase+8]);
                uint32_t b0=*reinterpret_cast<uint32_t*>(&Vst[dcol*BN+kbase]);
                uint32_t b1=*reinterpret_cast<uint32_t*>(&Vst[dcol*BN+kbase+8]);
                mma16816(o0,o1,o2,o3,a0,a1,a2,a3,b0,b1);
            }
            Oacc[dt][0]=o0;Oacc[dt][1]=o1;Oacc[dt][2]=o2;Oacc[dt][3]=o3;
        }
    }

    if(!warp_active) return;

    // ---- write O and LSE ----
    float inv0 = 1.0f/l0, inv1 = 1.0f/l1;
    #pragma unroll
    for(int dt=0;dt<NDT;dt++){
        int col = dt*8 + tig*2;
        if(global_row0<S){
            Og[(int64_t)global_row0*D+col  ]=__float2bfloat16(Oacc[dt][0]*inv0);
            Og[(int64_t)global_row0*D+col+1]=__float2bfloat16(Oacc[dt][1]*inv0);
        }
        if(global_row1<S){
            Og[(int64_t)global_row1*D+col  ]=__float2bfloat16(Oacc[dt][2]*inv1);
            Og[(int64_t)global_row1*D+col+1]=__float2bfloat16(Oacc[dt][3]*inv1);
        }
    }
    if(tig==0){
        const float LN2=0.6931471805599453f;
        if(global_row0<S) LSEg[global_row0]=m0*LN2+logf(l0);
        if(global_row1<S) LSEg[global_row1]=m1*LN2+logf(l1);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int Bv=(int)Q.size(0), Hv=(int)Q.size(1), Sv=(int)Q.size(2), Dv=(int)Q.size(3);
    float scale = 1.0f/sqrtf((float)Dv);
    float scale_log2 = scale*1.4426950408889634f;

    dim3 grid((Sv+BM-1)/BM, Hv, Bv);
    dim3 block(THREADS);
    size_t shmem = (size_t)(BM*D + BN*D + D*BN + BM*BN)*sizeof(__nv_bfloat16);

    CUDA_CHECK(cudaFuncSetAttribute(attn_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem));

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    attn_kernel<<<grid, block, shmem, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<__nv_bfloat16*>(O.data_ptr()),
        static_cast<float*>(LSE.data_ptr()),
        Bv,Hv,Sv,scale_log2);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_causal::run);

}  // namespace mha_causal