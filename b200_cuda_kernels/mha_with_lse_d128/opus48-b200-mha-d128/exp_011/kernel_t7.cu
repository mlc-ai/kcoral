#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
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

namespace mha_lse {

constexpr int D   = 128;
constexpr int BM  = 128;
constexpr int BN  = 64;
constexpr int LDS = 136;      // D + 8 padding
constexpr int NWARP = 8;
constexpr int NTHREAD = NWARP*32;

__device__ __forceinline__ float fexp2(float x){
    float y; asm volatile("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x)); return y;
}
__device__ __forceinline__ uint32_t pack_bf16_f(float x, float y){
    __nv_bfloat16 a = __float2bfloat16(x);
    __nv_bfloat16 b = __float2bfloat16(y);
    uint32_t r;
    asm("mov.b32 %0, {%1,%2};" : "=r"(r)
        : "h"(*reinterpret_cast<uint16_t*>(&a)),
          "h"(*reinterpret_cast<uint16_t*>(&b)));
    return r;
}
__device__ __forceinline__ void ldm_x4(uint32_t&r0,uint32_t&r1,uint32_t&r2,uint32_t&r3,const void* p){
    uint32_t a=(uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(a));
}
__device__ __forceinline__ void ldm_x2(uint32_t&r0,uint32_t&r1,const void* p){
    uint32_t a=(uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
        : "=r"(r0),"=r"(r1) : "r"(a));
}
__device__ __forceinline__ void ldm_x2_trans(uint32_t&r0,uint32_t&r1,const void* p){
    uint32_t a=(uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
        : "=r"(r0),"=r"(r1) : "r"(a));
}
__device__ __forceinline__ void mma_acc(
    float &d0,float &d1,float &d2,float &d3,
    uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,
    uint32_t b0,uint32_t b1){
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
      : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, int bytes){
    uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;\n"
        :: "r"(s),"l"(gmem),"r"(bytes) : "memory");
}
template<int N> __device__ __forceinline__ void cp_wait(){
    asm volatile("cp.async.wait_group %0;\n"::"n"(N):"memory");
}
__device__ __forceinline__ void cp_commit(){
    asm volatile("cp.async.commit_group;\n":::"memory");
}

__global__ __launch_bounds__(NTHREAD,1) void mha_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S)
{
    extern __shared__ __align__(16) __nv_bfloat16 smem[];
    __nv_bfloat16* sQ  = smem;
    __nv_bfloat16* sK0 = sQ  + BM*LDS;
    __nv_bfloat16* sK1 = sK0 + BN*LDS;
    __nv_bfloat16* sV0 = sK1 + BN*LDS;
    __nv_bfloat16* sV1 = sV0 + BN*LDS;
    __nv_bfloat16* sKb[2] = {sK0, sK1};
    __nv_bfloat16* sVb[2] = {sV0, sV1};

    const int b = blockIdx.z, h = blockIdx.y;
    const int q_start = blockIdx.x * BM;

    const int tid    = threadIdx.x;
    const int warpId = tid >> 5;
    const int lane   = tid & 31;
    const int group  = lane >> 2;
    const int tig    = lane & 3;
    const int base   = warpId * 16;

    const float SCALE = 0.08838834764831845f;
    const float LOG2E = 1.4426950408889634f;
    const float LN2   = 0.6931471805599453f;
    const float SC    = SCALE * LOG2E;

    const int64_t bh = (int64_t)(b*H + h);
    const __nv_bfloat16* Qp = Q + (bh*S + q_start)*D;
    const __nv_bfloat16* Kb = K + bh*S*D;
    const __nv_bfloat16* Vb = V + bh*S*D;

    #pragma unroll
    for (int i=0;i<(BM*D/8)/NTHREAD;i++){
        int e = tid + i*NTHREAD;
        int row = (e*8)/D;
        int col = (e*8)%D;
        int grow = q_start + row;
        int4 v = make_int4(0,0,0,0);
        if (grow < S) v = *(const int4*)(Qp + (int64_t)row*D + col);
        *(int4*)(&sQ[row*LDS + col]) = v;
    }
    __syncthreads();

    uint32_t qf[8][4];
    #pragma unroll
    for (int kk=0;kk<8;kk++){
        int r = base + (lane % 16);
        int c = kk*16 + (lane/16)*8;
        ldm_x4(qf[kk][0],qf[kk][1],qf[kk][2],qf[kk][3], &sQ[r*LDS + c]);
    }

    float o[16][4];
    #pragma unroll
    for (int j=0;j<16;j++){ o[j][0]=o[j][1]=o[j][2]=o[j][3]=0.f; }
    float m0=-1e30f, m1=-1e30f, l0=0.f, l1=0.f;

    int num_kv = (S + BN - 1)/BN;

    {
        #pragma unroll
        for (int i=0;i<(BN*D/8)/NTHREAD;i++){
            int e = tid + i*NTHREAD;
            int row = e/16;
            int col = (e%16)*8;
            int bytes = (row<S)?16:0;
            cp_async16(&sKb[0][row*LDS+col], Kb + (int64_t)row*D + col, bytes);
            cp_async16(&sVb[0][row*LDS+col], Vb + (int64_t)row*D + col, bytes);
        }
        cp_commit();
    }

    for (int kt=0; kt<num_kv; kt++){
        int cur = kt & 1;
        int kv_start = kt*BN;
        bool need_mask = (kv_start + BN) > S;

        if (kt+1 < num_kv){
            int nxt = (kt+1)&1;
            int ns  = (kt+1)*BN;
            #pragma unroll
            for (int i=0;i<(BN*D/8)/NTHREAD;i++){
                int e = tid + i*NTHREAD;
                int row = e/16;
                int col = (e%16)*8;
                int grow = ns + row;
                int bytes = (grow<S)?16:0;
                cp_async16(&sKb[nxt][row*LDS+col], Kb + (int64_t)grow*D + col, bytes);
                cp_async16(&sVb[nxt][row*LDS+col], Vb + (int64_t)grow*D + col, bytes);
            }
            cp_commit();
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();

        __nv_bfloat16* sK = sKb[cur];
        __nv_bfloat16* sV = sVb[cur];

        float s[8][4];
        #pragma unroll
        for(int j=0;j<8;j++){ s[j][0]=s[j][1]=s[j][2]=s[j][3]=0.f; }
        #pragma unroll
        for(int kk=0;kk<8;kk++){
            uint32_t kf[8][2];
            #pragma unroll
            for(int nt=0;nt<8;nt++){
                int rr, cc;
                if (lane < 8)      { rr = 8*nt + lane;     cc = kk*16;   }
                else if (lane < 16){ rr = 8*nt + lane - 8; cc = kk*16+8; }
                else               { rr = 8*nt;            cc = kk*16;   }
                ldm_x2(kf[nt][0],kf[nt][1], &sK[rr*LDS + cc]);
            }
            #pragma unroll
            for(int nt=0;nt<8;nt++){
                mma_acc(s[nt][0],s[nt][1],s[nt][2],s[nt][3],
                        qf[kk][0],qf[kk][1],qf[kk][2],qf[kk][3], kf[nt][0],kf[nt][1]);
            }
        }

        if (need_mask){
            #pragma unroll
            for(int j=0;j<8;j++){
                int c0col = kv_start + 8*j + 2*tig;
                s[j][0]*=SC; s[j][1]*=SC; s[j][2]*=SC; s[j][3]*=SC;
                if (c0col     >= S){ s[j][0]=-1e30f; s[j][2]=-1e30f; }
                if (c0col + 1 >= S){ s[j][1]=-1e30f; s[j][3]=-1e30f; }
            }
        } else {
            #pragma unroll
            for(int j=0;j<8;j++){
                s[j][0]*=SC; s[j][1]*=SC; s[j][2]*=SC; s[j][3]*=SC;
            }
        }

        float lm0=-1e30f, lm1=-1e30f;
        #pragma unroll
        for(int j=0;j<8;j++){
            lm0 = fmaxf(lm0, fmaxf(s[j][0], s[j][1]));
            lm1 = fmaxf(lm1, fmaxf(s[j][2], s[j][3]));
        }
        lm0 = fmaxf(lm0, __shfl_xor_sync(0xffffffff, lm0, 1));
        lm0 = fmaxf(lm0, __shfl_xor_sync(0xffffffff, lm0, 2));
        lm1 = fmaxf(lm1, __shfl_xor_sync(0xffffffff, lm1, 1));
        lm1 = fmaxf(lm1, __shfl_xor_sync(0xffffffff, lm1, 2));

        float mnew0 = fmaxf(m0, lm0);
        float mnew1 = fmaxf(m1, lm1);
        float corr0 = fexp2(m0 - mnew0);
        float corr1 = fexp2(m1 - mnew1);

        float ps0=0.f, ps1=0.f;
        #pragma unroll
        for(int j=0;j<8;j++){
            s[j][0]=fexp2(s[j][0]-mnew0); ps0+=s[j][0];
            s[j][1]=fexp2(s[j][1]-mnew0); ps0+=s[j][1];
            s[j][2]=fexp2(s[j][2]-mnew1); ps1+=s[j][2];
            s[j][3]=fexp2(s[j][3]-mnew1); ps1+=s[j][3];
        }
        ps0 += __shfl_xor_sync(0xffffffff, ps0, 1);
        ps0 += __shfl_xor_sync(0xffffffff, ps0, 2);
        ps1 += __shfl_xor_sync(0xffffffff, ps1, 1);
        ps1 += __shfl_xor_sync(0xffffffff, ps1, 2);

        l0 = l0*corr0 + ps0;
        l1 = l1*corr1 + ps1;
        m0 = mnew0; m1 = mnew1;

        #pragma unroll
        for(int j=0;j<16;j++){
            o[j][0]*=corr0; o[j][1]*=corr0; o[j][2]*=corr1; o[j][3]*=corr1;
        }

        #pragma unroll
        for(int kk=0;kk<4;kk++){
            int j0=2*kk, j1=2*kk+1;
            uint32_t A0 = pack_bf16_f(s[j0][0], s[j0][1]);
            uint32_t A1 = pack_bf16_f(s[j0][2], s[j0][3]);
            uint32_t A2 = pack_bf16_f(s[j1][0], s[j1][1]);
            uint32_t A3 = pack_bf16_f(s[j1][2], s[j1][3]);
            #pragma unroll
            for(int half=0; half<2; half++){
                uint32_t vf[8][2];
                #pragma unroll
                for(int t=0;t<8;t++){
                    int nt = half*8 + t;
                    int krow, ncol = 8*nt;
                    if (lane < 8)       krow = 16*kk + lane;
                    else if (lane < 16) krow = 16*kk + 8 + (lane-8);
                    else                krow = 16*kk;
                    ldm_x2_trans(vf[t][0],vf[t][1], &sV[krow*LDS + ncol]);
                }
                #pragma unroll
                for(int t=0;t<8;t++){
                    int nt = half*8 + t;
                    mma_acc(o[nt][0],o[nt][1],o[nt][2],o[nt][3], A0,A1,A2,A3, vf[t][0],vf[t][1]);
                }
            }
        }
        __syncthreads();
    }

    float inv0 = 1.f/l0;
    float inv1 = 1.f/l1;
    int grow0 = q_start + base + group;
    int grow1 = q_start + base + group + 8;
    __nv_bfloat16* Ob = O + bh*S*D;
    #pragma unroll
    for(int out=0; out<16; out++){
        int col0 = 8*out + 2*tig;
        int col1 = col0 + 1;
        if (grow0 < S){
            Ob[(int64_t)grow0*D + col0] = __float2bfloat16(o[out][0]*inv0);
            Ob[(int64_t)grow0*D + col1] = __float2bfloat16(o[out][1]*inv0);
        }
        if (grow1 < S){
            Ob[(int64_t)grow1*D + col0] = __float2bfloat16(o[out][2]*inv1);
            Ob[(int64_t)grow1*D + col1] = __float2bfloat16(o[out][3]*inv1);
        }
    }
    if (tig==0){
        float* LSEb = LSE + bh*S;
        if (grow0 < S) LSEb[grow0] = m0*LN2 + logf(l0);
        if (grow1 < S) LSEb[grow1] = m1*LN2 + logf(l1);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int S = (int)Q.size(2);

    const __nv_bfloat16* Qp = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* Kp = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* Vp = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* Op = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSEp = static_cast<float*>(LSE.data_ptr());

    int num_q = (S + BM - 1)/BM;
    dim3 grid(num_q, H, B);
    dim3 block(NTHREAD);

    size_t smem_bytes = (size_t)(BM + 4*BN) * LDS * sizeof(__nv_bfloat16);
    static bool attr_set = false;
    if (!attr_set){
        CUDA_CHECK(cudaFuncSetAttribute(mha_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));
        attr_set = true;
    }

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    mha_kernel<<<grid, block, smem_bytes, stream>>>(Qp,Kp,Vp,Op,LSEp,B,H,S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_lse::run);

}  // namespace mha_lse