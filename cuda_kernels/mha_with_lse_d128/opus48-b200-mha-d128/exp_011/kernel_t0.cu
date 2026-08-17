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

constexpr int D  = 128;
constexpr int BM = 64;
constexpr int BN = 64;

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

__device__ __forceinline__ uint32_t pack_bf16_r(__nv_bfloat16 a, __nv_bfloat16 b){
    uint32_t r;
    asm("mov.b32 %0, {%1,%2};" : "=r"(r)
        : "h"(*reinterpret_cast<uint16_t*>(&a)),
          "h"(*reinterpret_cast<uint16_t*>(&b)));
    return r;
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

__global__ void mha_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S)
{
    __shared__ __align__(16) __nv_bfloat16 sQ[BM*D];
    __shared__ __align__(16) __nv_bfloat16 sK[BN*D];
    __shared__ __align__(16) __nv_bfloat16 sV[BN*D];

    const int b = blockIdx.z, h = blockIdx.y;
    const int q_start = blockIdx.x * BM;

    const int tid    = threadIdx.x;
    const int warpId = tid >> 5;
    const int lane   = tid & 31;
    const int group  = lane >> 2;   // 0..7
    const int tig    = lane & 3;     // 0..3
    const int base   = warpId * 16;  // query row base within tile

    const float SCALE = 0.08838834764831845f;      // 1/sqrt(128)
    const float LOG2E = 1.4426950408889634f;
    const float LN2   = 0.6931471805599453f;
    const float SC    = SCALE * LOG2E;

    const int64_t bh = (int64_t)(b*H + h);
    const __nv_bfloat16* Qp = Q + (bh*S + q_start)*D;
    const __nv_bfloat16* Kb = K + bh*S*D;
    const __nv_bfloat16* Vb = V + bh*S*D;

    // load Q tile
    for (int e = tid; e < (BM*D)/8; e += blockDim.x){
        int row = (e*8) / D;
        int col = (e*8) % D;
        int grow = q_start + row;
        int4 v = make_int4(0,0,0,0);
        if (grow < S) v = *(const int4*)(Qp + (int64_t)row*D + col);
        *(int4*)(&sQ[row*D + col]) = v;
    }

    float o[16][4];
    #pragma unroll
    for (int j=0;j<16;j++){ o[j][0]=o[j][1]=o[j][2]=o[j][3]=0.f; }
    float m0=-1e30f, m1=-1e30f, l0=0.f, l1=0.f;

    int num_kv = (S + BN - 1)/BN;
    __syncthreads();

    for (int kt=0; kt<num_kv; kt++){
        int kv_start = kt*BN;

        // load K,V tiles
        for (int e = tid; e < (BN*D)/8; e += blockDim.x){
            int row = (e*8)/D;
            int col = (e*8)%D;
            int grow = kv_start + row;
            int4 vk = make_int4(0,0,0,0);
            int4 vv = make_int4(0,0,0,0);
            if (grow < S){
                vk = *(const int4*)(Kb + (int64_t)grow*D + col);
                vv = *(const int4*)(Vb + (int64_t)grow*D + col);
            }
            *(int4*)(&sK[row*D+col]) = vk;
            *(int4*)(&sV[row*D+col]) = vv;
        }
        __syncthreads();

        // ---- matmul1: S = Q @ K^T ----
        float s[8][4];
        #pragma unroll
        for(int j=0;j<8;j++){ s[j][0]=s[j][1]=s[j][2]=s[j][3]=0.f; }
        #pragma unroll
        for(int kstep=0;kstep<8;kstep++){
            int kcol = kstep*16;
            uint32_t a0 = *(const uint32_t*)(&sQ[(base+group)*D   + kcol + 2*tig]);
            uint32_t a1 = *(const uint32_t*)(&sQ[(base+group+8)*D + kcol + 2*tig]);
            uint32_t a2 = *(const uint32_t*)(&sQ[(base+group)*D   + kcol + 2*tig + 8]);
            uint32_t a3 = *(const uint32_t*)(&sQ[(base+group+8)*D + kcol + 2*tig + 8]);
            #pragma unroll
            for(int j=0;j<8;j++){
                int n = 8*j + group;
                uint32_t b0 = *(const uint32_t*)(&sK[n*D + kcol + 2*tig]);
                uint32_t b1 = *(const uint32_t*)(&sK[n*D + kcol + 2*tig + 8]);
                mma_acc(s[j][0],s[j][1],s[j][2],s[j][3], a0,a1,a2,a3, b0,b1);
            }
        }

        // scale + mask invalid key columns
        #pragma unroll
        for(int j=0;j<8;j++){
            int c0col = kv_start + 8*j + 2*tig;
            s[j][0]*=SC; s[j][1]*=SC; s[j][2]*=SC; s[j][3]*=SC;
            if (c0col     >= S){ s[j][0]=-1e30f; s[j][2]=-1e30f; }
            if (c0col + 1 >= S){ s[j][1]=-1e30f; s[j][3]=-1e30f; }
        }

        // row max (over 64 cols)
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

        // rescale O accumulator
        #pragma unroll
        for(int j=0;j<16;j++){
            o[j][0]*=corr0; o[j][1]*=corr0; o[j][2]*=corr1; o[j][3]*=corr1;
        }

        // ---- matmul2: O += P @ V ----
        #pragma unroll
        for(int kstep=0;kstep<4;kstep++){
            int j0=2*kstep, j1=2*kstep+1;
            uint32_t a0 = pack_bf16_f(s[j0][0], s[j0][1]);
            uint32_t a1 = pack_bf16_f(s[j0][2], s[j0][3]);
            uint32_t a2 = pack_bf16_f(s[j1][0], s[j1][1]);
            uint32_t a3 = pack_bf16_f(s[j1][2], s[j1][3]);
            int kbase = kstep*16;
            #pragma unroll
            for(int out=0; out<16; out++){
                int n = 8*out + group;
                __nv_bfloat16 v0 = sV[(kbase+2*tig)*D   + n];
                __nv_bfloat16 v1 = sV[(kbase+2*tig+1)*D + n];
                __nv_bfloat16 v2 = sV[(kbase+2*tig+8)*D + n];
                __nv_bfloat16 v3 = sV[(kbase+2*tig+9)*D + n];
                uint32_t b0 = pack_bf16_r(v0,v1);
                uint32_t b1 = pack_bf16_r(v2,v3);
                mma_acc(o[out][0],o[out][1],o[out][2],o[out][3], a0,a1,a2,a3, b0,b1);
            }
        }
        __syncthreads();
    }

    // finalize + write O
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
    dim3 block(128);

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    mha_kernel<<<grid, block, 0, stream>>>(Qp,Kp,Vp,Op,LSEp,B,H,S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_lse::run);

}  // namespace mha_lse