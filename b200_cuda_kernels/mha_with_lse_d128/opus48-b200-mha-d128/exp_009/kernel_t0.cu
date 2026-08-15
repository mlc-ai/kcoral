#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>
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

namespace mha_d128 {

constexpr int D   = 128;
constexpr int BM  = 64;
constexpr int BN  = 64;
constexpr int KD  = D / 16;   // 8  QK k-chunks
constexpr int NT  = BN / 8;   // 8  QK n-tiles
constexpr int KC  = BN / 16;  // 4  PV k-chunks
constexpr int ND  = D / 8;    // 16 PV n-tiles

__device__ __forceinline__ void mma_m16n8k16(float* d, const uint32_t* a, const uint32_t* b){
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),
          "r"(b[0]),"r"(b[1]));
}

__device__ __forceinline__ uint32_t pack2bf16(float a, float b){
    __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
    uint32_t out;
    memcpy(&out, &v, 4);
    return out;
}

__device__ __forceinline__ float group_max4(float v){
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 1));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 2));
    return v;
}
__device__ __forceinline__ float group_sum4(float v){
    v += __shfl_xor_sync(0xffffffff, v, 1);
    v += __shfl_xor_sync(0xffffffff, v, 2);
    return v;
}

__global__ __launch_bounds__(128) void attn_kernel(
        const __nv_bfloat16* __restrict__ Q,
        const __nv_bfloat16* __restrict__ K,
        const __nv_bfloat16* __restrict__ V,
        __nv_bfloat16* __restrict__ O,
        float* __restrict__ LSE,
        int S){
    extern __shared__ char smem[];
    __nv_bfloat16* Qs   = reinterpret_cast<__nv_bfloat16*>(smem);   // BM*D
    __nv_bfloat16* Ks   = Qs + BM*D;                                // BN*D
    __nv_bfloat16* Vst  = Ks + BN*D;                                // D*BN (transposed)

    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int groupID = lane >> 2;
    int tg = lane & 3;

    int q_tile = blockIdx.x;
    int bh = blockIdx.y;
    int q_start = q_tile * BM;

    int64_t base = (int64_t)bh * S * D;
    const __nv_bfloat16* Qg = Q + base;
    const __nv_bfloat16* Kg = K + base;
    const __nv_bfloat16* Vg = V + base;
    __nv_bfloat16* Og = O + base;
    float* LSEg = LSE + (int64_t)bh * S;

    // ---- Load Q tile ----
    for (int i = tid; i < BM*(D/8); i += blockDim.x){
        int r = i / (D/8);
        int dblk = i % (D/8);
        int s = q_start + r;
        uint4 val;
        if (s < S) val = *reinterpret_cast<const uint4*>(&Qg[(int64_t)s*D + dblk*8]);
        else       val = make_uint4(0,0,0,0);
        *reinterpret_cast<uint4*>(&Qs[r*D + dblk*8]) = val;
    }
    __syncthreads();

    // ---- Cache Q fragments (persistent) ----
    uint32_t qA[KD][4];
    int r0 = warp*16 + groupID;
    int r1 = warp*16 + groupID + 8;
    #pragma unroll
    for (int kc=0; kc<KD; kc++){
        int c = kc*16 + tg*2;
        qA[kc][0] = *reinterpret_cast<uint32_t*>(&Qs[r0*D + c]);
        qA[kc][1] = *reinterpret_cast<uint32_t*>(&Qs[r1*D + c]);
        qA[kc][2] = *reinterpret_cast<uint32_t*>(&Qs[r0*D + c + 8]);
        qA[kc][3] = *reinterpret_cast<uint32_t*>(&Qs[r1*D + c + 8]);
    }

    float acc[ND][4];
    #pragma unroll
    for (int nd=0; nd<ND; nd++){ acc[nd][0]=acc[nd][1]=acc[nd][2]=acc[nd][3]=0.f; }
    float m0=-1e30f, m1=-1e30f, l0=0.f, l1=0.f;

    const float scale_l2e = 0.08838834764831845f * 1.4426950408889634f; // (1/sqrt(128))*log2(e)

    int num_kv = (S + BN - 1) / BN;
    for (int kv=0; kv<num_kv; kv++){
        int kv_start = kv*BN;
        __syncthreads();
        // load K tile
        for (int i = tid; i < BN*(D/8); i += blockDim.x){
            int r = i/(D/8);
            int dblk = i%(D/8);
            int s = kv_start + r;
            uint4 val;
            if (s < S) val = *reinterpret_cast<const uint4*>(&Kg[(int64_t)s*D + dblk*8]);
            else       val = make_uint4(0,0,0,0);
            *reinterpret_cast<uint4*>(&Ks[r*D + dblk*8]) = val;
        }
        // load V tile transposed -> Vst[d*BN + r]
        for (int i = tid; i < BN*(D/8); i += blockDim.x){
            int r = i/(D/8);
            int dblk = i%(D/8);
            int s = kv_start + r;
            uint4 val;
            if (s < S) val = *reinterpret_cast<const uint4*>(&Vg[(int64_t)s*D + dblk*8]);
            else       val = make_uint4(0,0,0,0);
            __nv_bfloat16* vb = reinterpret_cast<__nv_bfloat16*>(&val);
            #pragma unroll
            for (int j=0;j<8;j++){
                int d = dblk*8 + j;
                Vst[d*BN + r] = vb[j];
            }
        }
        __syncthreads();

        // ---- QK^T ----
        float S_reg[NT][4];
        #pragma unroll
        for (int nt=0; nt<NT; nt++){
            float c[4] = {0.f,0.f,0.f,0.f};
            int bn = nt*8 + groupID;
            #pragma unroll
            for (int kc=0; kc<KD; kc++){
                uint32_t rB[2];
                rB[0] = *reinterpret_cast<uint32_t*>(&Ks[bn*D + kc*16 + tg*2]);
                rB[1] = *reinterpret_cast<uint32_t*>(&Ks[bn*D + kc*16 + tg*2 + 8]);
                mma_m16n8k16(c, qA[kc], rB);
            }
            S_reg[nt][0] = c[0]*scale_l2e;
            S_reg[nt][1] = c[1]*scale_l2e;
            S_reg[nt][2] = c[2]*scale_l2e;
            S_reg[nt][3] = c[3]*scale_l2e;
            int col0 = kv_start + nt*8 + tg*2;
            int col1 = col0 + 1;
            if (col0 >= S){ S_reg[nt][0] = -1e30f; S_reg[nt][2] = -1e30f; }
            if (col1 >= S){ S_reg[nt][1] = -1e30f; S_reg[nt][3] = -1e30f; }
        }

        // ---- row max ----
        float rmax0=-1e30f, rmax1=-1e30f;
        #pragma unroll
        for (int nt=0; nt<NT; nt++){
            rmax0 = fmaxf(rmax0, fmaxf(S_reg[nt][0], S_reg[nt][1]));
            rmax1 = fmaxf(rmax1, fmaxf(S_reg[nt][2], S_reg[nt][3]));
        }
        rmax0 = group_max4(rmax0);
        rmax1 = group_max4(rmax1);

        float mnew0 = fmaxf(m0, rmax0);
        float mnew1 = fmaxf(m1, rmax1);
        float corr0 = exp2f(m0 - mnew0);
        float corr1 = exp2f(m1 - mnew1);
        l0 *= corr0; l1 *= corr1;
        #pragma unroll
        for (int nd=0; nd<ND; nd++){
            acc[nd][0]*=corr0; acc[nd][1]*=corr0;
            acc[nd][2]*=corr1; acc[nd][3]*=corr1;
        }
        m0 = mnew0; m1 = mnew1;

        // ---- P and rowsum ----
        float sum0=0.f, sum1=0.f;
        #pragma unroll
        for (int nt=0; nt<NT; nt++){
            float p0 = exp2f(S_reg[nt][0]-m0);
            float p1 = exp2f(S_reg[nt][1]-m0);
            float p2 = exp2f(S_reg[nt][2]-m1);
            float p3 = exp2f(S_reg[nt][3]-m1);
            S_reg[nt][0]=p0; S_reg[nt][1]=p1; S_reg[nt][2]=p2; S_reg[nt][3]=p3;
            sum0 += p0+p1; sum1 += p2+p3;
        }
        sum0 = group_sum4(sum0);
        sum1 = group_sum4(sum1);
        l0 += sum0; l1 += sum1;

        // ---- PV ----
        uint32_t pA[KC][4];
        #pragma unroll
        for (int kc=0; kc<KC; kc++){
            pA[kc][0] = pack2bf16(S_reg[2*kc][0],   S_reg[2*kc][1]);
            pA[kc][1] = pack2bf16(S_reg[2*kc][2],   S_reg[2*kc][3]);
            pA[kc][2] = pack2bf16(S_reg[2*kc+1][0], S_reg[2*kc+1][1]);
            pA[kc][3] = pack2bf16(S_reg[2*kc+1][2], S_reg[2*kc+1][3]);
        }
        #pragma unroll
        for (int nd=0; nd<ND; nd++){
            int d = nd*8 + groupID;
            #pragma unroll
            for (int kc=0; kc<KC; kc++){
                uint32_t rB[2];
                rB[0] = *reinterpret_cast<uint32_t*>(&Vst[d*BN + kc*16 + tg*2]);
                rB[1] = *reinterpret_cast<uint32_t*>(&Vst[d*BN + kc*16 + tg*2 + 8]);
                mma_m16n8k16(acc[nd], pA[kc], rB);
            }
        }
    }

    // ---- Epilogue ----
    float inv_l0 = 1.f / l0;
    float inv_l1 = 1.f / l1;
    int s0 = q_start + r0;
    int s1 = q_start + r1;
    #pragma unroll
    for (int nd=0; nd<ND; nd++){
        int d = nd*8 + tg*2;
        if (s0 < S){
            __nv_bfloat162 v = __floats2bfloat162_rn(acc[nd][0]*inv_l0, acc[nd][1]*inv_l0);
            uint32_t out; memcpy(&out,&v,4);
            *reinterpret_cast<uint32_t*>(&Og[(int64_t)s0*D + d]) = out;
        }
        if (s1 < S){
            __nv_bfloat162 v = __floats2bfloat162_rn(acc[nd][2]*inv_l1, acc[nd][3]*inv_l1);
            uint32_t out; memcpy(&out,&v,4);
            *reinterpret_cast<uint32_t*>(&Og[(int64_t)s1*D + d]) = out;
        }
    }
    const float LN2 = 0.6931471805599453f;
    if (tg == 0){
        if (s0 < S) LSEg[s0] = m0*LN2 + logf(l0);
        if (s1 < S) LSEg[s1] = m1*LN2 + logf(l1);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE){
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int Bsz = (int)Q.size(0);
    int Hsz = (int)Q.size(1);
    int S   = (int)Q.size(2);
    // int Dd = (int)Q.size(3); // == 128

    const __nv_bfloat16* Qp = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* Kp = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* Vp = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* Op = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSEp = static_cast<float*>(LSE.data_ptr());

    int num_q_tiles = (S + BM - 1) / BM;
    dim3 grid(num_q_tiles, Bsz*Hsz);
    int smem = (BM*D + BN*D + D*BN) * (int)sizeof(__nv_bfloat16);

    static bool attr_set = false;
    if (!attr_set){
        CUDA_CHECK(cudaFuncSetAttribute(attn_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        attr_set = true;
    }

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    attn_kernel<<<grid, 128, smem, stream>>>(Qp, Kp, Vp, Op, LSEp, S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_d128::run);

}  // namespace mha_d128