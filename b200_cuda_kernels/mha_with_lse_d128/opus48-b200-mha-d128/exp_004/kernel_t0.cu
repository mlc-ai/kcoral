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
    }                                                              \
} while(0)

namespace mha_kernel {

#define DHEAD 128
#define BM 64
#define BN 64

__device__ __forceinline__ void mma_m16n8k16(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

__device__ __forceinline__ uint32_t pack2bf16(float lo, float hi){
    __nv_bfloat16 a = __float2bfloat16(lo);
    __nv_bfloat16 b = __float2bfloat16(hi);
    uint16_t ai = *reinterpret_cast<uint16_t*>(&a);
    uint16_t bi = *reinterpret_cast<uint16_t*>(&b);
    return (uint32_t)ai | ((uint32_t)bi << 16);
}

__device__ __forceinline__ uint32_t pack_strided(const __nv_bfloat16* base, int i0, int i1){
    uint16_t a = *reinterpret_cast<const uint16_t*>(base + i0);
    uint16_t b = *reinterpret_cast<const uint16_t*>(base + i1);
    return (uint32_t)a | ((uint32_t)b << 16);
}

__global__ void attn_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int Bsz, int Hn, int S)
{
    extern __shared__ char smem[];
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sK = sQ + BM*DHEAD;
    __nv_bfloat16* sV = sK + BN*DHEAD;

    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int groupID = lane >> 2;   // 0..7  -> row group
    int tig = lane & 3;        // 0..3  -> thread-in-group

    int q_block = blockIdx.x;
    int bh = blockIdx.y;
    int q_start = q_block * BM;

    const float scale = rsqrtf((float)DHEAD);

    const __nv_bfloat16* Qh = Q + ((int64_t)bh) * S * DHEAD;
    const __nv_bfloat16* Kh = K + ((int64_t)bh) * S * DHEAD;
    const __nv_bfloat16* Vh = V + ((int64_t)bh) * S * DHEAD;
    __nv_bfloat16* Oh = O + ((int64_t)bh) * S * DHEAD;
    float* LSEh = LSE + ((int64_t)bh) * S;

    // Load Q block (persistent)
    for (int idx = tid; idx < BM*DHEAD; idx += blockDim.x) {
        int r = idx / DHEAD;
        int d = idx % DHEAD;
        int qg = q_start + r;
        sQ[idx] = (qg < S) ? Qh[(int64_t)qg*DHEAD + d] : __float2bfloat16(0.f);
    }
    __syncthreads();

    float o_acc[16][4];
    #pragma unroll
    for (int i=0;i<16;i++){ o_acc[i][0]=o_acc[i][1]=o_acc[i][2]=o_acc[i][3]=0.f; }
    float m_top = -1e30f, m_bot = -1e30f, l_top = 0.f, l_bot = 0.f;

    int arow0 = warp*16 + groupID;
    int arow1 = warp*16 + groupID + 8;

    int num_kv = (S + BN - 1) / BN;
    for (int kv = 0; kv < num_kv; kv++){
        int k_start = kv * BN;
        for (int idx = tid; idx < BN*DHEAD; idx += blockDim.x){
            int r = idx / DHEAD; int d = idx % DHEAD;
            int kg = k_start + r;
            sK[idx] = (kg < S) ? Kh[(int64_t)kg*DHEAD + d] : __float2bfloat16(0.f);
        }
        for (int idx = tid; idx < BN*DHEAD; idx += blockDim.x){
            int r = idx / DHEAD; int d = idx % DHEAD;
            int kg = k_start + r;
            sV[idx] = (kg < S) ? Vh[(int64_t)kg*DHEAD + d] : __float2bfloat16(0.f);
        }
        __syncthreads();

        // ===== S = Q @ K^T  (per warp: 16 x 64) =====
        float s_frag[8][4];
        #pragma unroll
        for (int nt=0;nt<8;nt++){ s_frag[nt][0]=s_frag[nt][1]=s_frag[nt][2]=s_frag[nt][3]=0.f; }
        #pragma unroll
        for (int kt=0; kt<8; kt++){
            int acol = kt*16 + tig*2;
            uint32_t a0 = *reinterpret_cast<uint32_t*>(&sQ[arow0*DHEAD + acol]);
            uint32_t a1 = *reinterpret_cast<uint32_t*>(&sQ[arow1*DHEAD + acol]);
            uint32_t a2 = *reinterpret_cast<uint32_t*>(&sQ[arow0*DHEAD + acol + 8]);
            uint32_t a3 = *reinterpret_cast<uint32_t*>(&sQ[arow1*DHEAD + acol + 8]);
            #pragma unroll
            for (int nt=0; nt<8; nt++){
                int key = nt*8 + groupID;
                int bcol = kt*16 + tig*2;
                uint32_t b0 = *reinterpret_cast<uint32_t*>(&sK[key*DHEAD + bcol]);
                uint32_t b1 = *reinterpret_cast<uint32_t*>(&sK[key*DHEAD + bcol + 8]);
                mma_m16n8k16(s_frag[nt][0],s_frag[nt][1],s_frag[nt][2],s_frag[nt][3],
                             a0,a1,a2,a3,b0,b1,
                             s_frag[nt][0],s_frag[nt][1],s_frag[nt][2],s_frag[nt][3]);
            }
        }

        // scale + mask out-of-range keys
        #pragma unroll
        for (int nt=0; nt<8; nt++){
            int keyA = k_start + nt*8 + tig*2;
            int keyB = keyA + 1;
            s_frag[nt][0] = (keyA < S) ? s_frag[nt][0]*scale : -1e30f;
            s_frag[nt][1] = (keyB < S) ? s_frag[nt][1]*scale : -1e30f;
            s_frag[nt][2] = (keyA < S) ? s_frag[nt][2]*scale : -1e30f;
            s_frag[nt][3] = (keyB < S) ? s_frag[nt][3]*scale : -1e30f;
        }

        // block row max
        float mt = -1e30f, mb = -1e30f;
        #pragma unroll
        for (int nt=0;nt<8;nt++){
            mt = fmaxf(mt, fmaxf(s_frag[nt][0], s_frag[nt][1]));
            mb = fmaxf(mb, fmaxf(s_frag[nt][2], s_frag[nt][3]));
        }
        mt = fmaxf(mt, __shfl_xor_sync(0xffffffff, mt, 1));
        mt = fmaxf(mt, __shfl_xor_sync(0xffffffff, mt, 2));
        mb = fmaxf(mb, __shfl_xor_sync(0xffffffff, mb, 1));
        mb = fmaxf(mb, __shfl_xor_sync(0xffffffff, mb, 2));

        float mt_new = fmaxf(m_top, mt);
        float mb_new = fmaxf(m_bot, mb);
        float corr_t = __expf(m_top - mt_new);
        float corr_b = __expf(m_bot - mb_new);

        float lt=0.f, lb=0.f;
        #pragma unroll
        for (int nt=0;nt<8;nt++){
            float p0=__expf(s_frag[nt][0]-mt_new); lt+=p0; s_frag[nt][0]=p0;
            float p1=__expf(s_frag[nt][1]-mt_new); lt+=p1; s_frag[nt][1]=p1;
            float p2=__expf(s_frag[nt][2]-mb_new); lb+=p2; s_frag[nt][2]=p2;
            float p3=__expf(s_frag[nt][3]-mb_new); lb+=p3; s_frag[nt][3]=p3;
        }
        lt += __shfl_xor_sync(0xffffffff, lt, 1); lt += __shfl_xor_sync(0xffffffff, lt, 2);
        lb += __shfl_xor_sync(0xffffffff, lb, 1); lb += __shfl_xor_sync(0xffffffff, lb, 2);

        l_top = l_top*corr_t + lt;
        l_bot = l_bot*corr_b + lb;
        m_top = mt_new; m_bot = mb_new;

        #pragma unroll
        for (int ntd=0; ntd<16; ntd++){
            o_acc[ntd][0]*=corr_t; o_acc[ntd][1]*=corr_t;
            o_acc[ntd][2]*=corr_b; o_acc[ntd][3]*=corr_b;
        }

        // ===== O += P @ V  (remap C-frag of P directly to A-frag) =====
        #pragma unroll
        for (int kt=0; kt<4; kt++){
            uint32_t a0 = pack2bf16(s_frag[2*kt][0],   s_frag[2*kt][1]);
            uint32_t a1 = pack2bf16(s_frag[2*kt][2],   s_frag[2*kt][3]);
            uint32_t a2 = pack2bf16(s_frag[2*kt+1][0], s_frag[2*kt+1][1]);
            uint32_t a3 = pack2bf16(s_frag[2*kt+1][2], s_frag[2*kt+1][3]);
            int key0 = kt*16 + tig*2;
            #pragma unroll
            for (int nt=0; nt<16; nt++){
                int d = nt*8 + groupID;
                uint32_t b0 = pack_strided(sV, key0*DHEAD + d,     (key0+1)*DHEAD + d);
                uint32_t b1 = pack_strided(sV, (key0+8)*DHEAD + d, (key0+9)*DHEAD + d);
                mma_m16n8k16(o_acc[nt][0],o_acc[nt][1],o_acc[nt][2],o_acc[nt][3],
                             a0,a1,a2,a3,b0,b1,
                             o_acc[nt][0],o_acc[nt][1],o_acc[nt][2],o_acc[nt][3]);
            }
        }
        __syncthreads();
    }

    // epilogue
    float inv_lt = (l_top > 0.f) ? 1.f/l_top : 0.f;
    float inv_lb = (l_bot > 0.f) ? 1.f/l_bot : 0.f;
    int qT = q_start + warp*16 + groupID;
    int qB = qT + 8;
    #pragma unroll
    for (int ntd=0; ntd<16; ntd++){
        int d0 = ntd*8 + tig*2;
        if (qT < S){
            Oh[(int64_t)qT*DHEAD + d0]   = __float2bfloat16(o_acc[ntd][0]*inv_lt);
            Oh[(int64_t)qT*DHEAD + d0+1] = __float2bfloat16(o_acc[ntd][1]*inv_lt);
        }
        if (qB < S){
            Oh[(int64_t)qB*DHEAD + d0]   = __float2bfloat16(o_acc[ntd][2]*inv_lb);
            Oh[(int64_t)qB*DHEAD + d0+1] = __float2bfloat16(o_acc[ntd][3]*inv_lb);
        }
    }
    if (tig == 0){
        if (qT < S) LSEh[qT] = m_top + logf(l_top);
        if (qB < S) LSEh[qB] = m_bot + logf(l_bot);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE){
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int Bsz = (int)Q.size(0);
    int Hn  = (int)Q.size(1);
    int S   = (int)Q.size(2);

    const __nv_bfloat16* Qp = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* Kp = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* Vp = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* Op = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSEp = static_cast<float*>(LSE.data_ptr());

    int num_q = (S + BM - 1)/BM;
    dim3 grid(num_q, Bsz*Hn);
    dim3 block(128);
    size_t smem = (size_t)(BM*DHEAD + BN*DHEAD + BN*DHEAD)*sizeof(__nv_bfloat16);
    cudaFuncSetAttribute(attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    attn_kernel<<<grid, block, smem, stream>>>(Qp,Kp,Vp,Op,LSEp, Bsz,Hn,S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel::run);

}  // namespace mha_kernel