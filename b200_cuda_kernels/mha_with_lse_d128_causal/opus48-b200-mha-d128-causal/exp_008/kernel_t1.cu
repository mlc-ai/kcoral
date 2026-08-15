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
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_kernel {

constexpr int Hc = 48;
constexpr int Dc = 128;
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int THREADS = 128;
constexpr int ST = Dc + 8;   // padded row stride (elements) to avoid bank conflicts

__device__ __forceinline__ uint32_t pack_f(float a, float b){
    __nv_bfloat16 x = __float2bfloat16(a);
    __nv_bfloat16 y = __float2bfloat16(b);
    uint16_t xr = *reinterpret_cast<uint16_t*>(&x);
    uint16_t yr = *reinterpret_cast<uint16_t*>(&y);
    return (uint32_t)xr | ((uint32_t)yr << 16);
}
__device__ __forceinline__ void ldm_x4(uint32_t &r0,uint32_t &r1,uint32_t &r2,uint32_t &r3, const void* p){
    uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
      : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(a));
}
__device__ __forceinline__ void ldm_x4_trans(uint32_t &r0,uint32_t &r1,uint32_t &r2,uint32_t &r3, const void* p){
    uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
      : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(a));
}
__device__ __forceinline__ void mma16816(
    float &d0,float &d1,float &d2,float &d3,
    uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,
    uint32_t b0,uint32_t b1){
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
      : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}

__global__ __launch_bounds__(THREADS) void mha_fwd(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S){

    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* Qs = smem;                // BM*ST
    __nv_bfloat16* Ks = Qs + BM*ST;          // BN*ST
    __nv_bfloat16* Vs = Ks + BN*ST;          // BN*ST

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid  = lane >> 2;   // 0..7
    const int tig  = lane & 3;    // 0..3
    const int lrow = lane & 15;   // ldmatrix row
    const int lcol = (lane >> 4) * 8; // ldmatrix col (0 or 8)

    const int b = blockIdx.z;
    const int h = blockIdx.y;
    const int q0 = blockIdx.x * BM;
    if (q0 >= S) return;

    const size_t bh = (size_t)(b*Hc + h) * (size_t)S;
    const __nv_bfloat16* Qg = Q + bh*Dc;
    const __nv_bfloat16* Kg = K + bh*Dc;
    const __nv_bfloat16* Vg = V + bh*Dc;
    __nv_bfloat16* Og = O + bh*Dc;
    float* LSEg = LSE + bh;

    const float scale = 0.08838834764831845f; // 1/sqrt(128)
    const int nvec = Dc/8; // 16 uint4 per row

    // ---- load Q tile ----
    for (int idx = tid; idx < BM*nvec; idx += THREADS){
        int r = idx / nvec, cv = idx % nvec;
        int gq = q0 + r;
        uint4 val;
        if (gq < S) val = *reinterpret_cast<const uint4*>(&Qg[(size_t)gq*Dc + cv*8]);
        else        val = make_uint4(0,0,0,0);
        *reinterpret_cast<uint4*>(&Qs[r*ST + cv*8]) = val;
    }
    __syncthreads();

    // ---- preload Q fragments (this warp's 16 rows) via ldmatrix ----
    const int wr = warp*16;
    uint32_t Qf[8][4];
    #pragma unroll
    for (int kt=0; kt<8; kt++){
        ldm_x4(Qf[kt][0], Qf[kt][1], Qf[kt][2], Qf[kt][3],
               &Qs[(wr+lrow)*ST + kt*16 + lcol]);
    }

    float Oacc[16][4];
    #pragma unroll
    for (int i=0;i<16;i++){ Oacc[i][0]=0.f;Oacc[i][1]=0.f;Oacc[i][2]=0.f;Oacc[i][3]=0.f; }
    float m_a = -INFINITY, m_b = -INFINITY;
    float l_a = 0.f, l_b = 0.f;

    const int row_a = q0 + wr + gid;
    const int row_b = q0 + wr + gid + 8;

    int qmax = q0 + BM - 1; if (qmax >= S) qmax = S-1;
    const int num_kt = qmax / BN + 1;

    for (int ktile=0; ktile<num_kt; ktile++){
        int kbase = ktile*BN;

        __syncthreads();
        for (int idx = tid; idx < BN*nvec; idx += THREADS){
            int r = idx / nvec, cv = idx % nvec;
            int gk = kbase + r;
            uint4 vk, vv;
            if (gk < S){
                vk = *reinterpret_cast<const uint4*>(&Kg[(size_t)gk*Dc + cv*8]);
                vv = *reinterpret_cast<const uint4*>(&Vg[(size_t)gk*Dc + cv*8]);
            } else { vk = make_uint4(0,0,0,0); vv = vk; }
            *reinterpret_cast<uint4*>(&Ks[r*ST + cv*8]) = vk;
            *reinterpret_cast<uint4*>(&Vs[r*ST + cv*8]) = vv;
        }
        __syncthreads();

        // ---- S = Q @ K^T ----
        float Sacc[8][4];
        #pragma unroll
        for (int nt=0;nt<8;nt++){Sacc[nt][0]=0.f;Sacc[nt][1]=0.f;Sacc[nt][2]=0.f;Sacc[nt][3]=0.f;}
        #pragma unroll
        for (int kt=0; kt<8; kt++){
            int k0 = kt*16;
            #pragma unroll
            for (int np=0; np<4; np++){
                int n0 = np*16;
                uint32_t kb0,kb1,kb2,kb3;
                ldm_x4(kb0,kb1,kb2,kb3, &Ks[(n0+lrow)*ST + k0 + lcol]);
                // kb0=b0(n0), kb1=b0(n0+8), kb2=b1(n0), kb3=b1(n0+8)
                mma16816(Sacc[2*np][0],Sacc[2*np][1],Sacc[2*np][2],Sacc[2*np][3],
                         Qf[kt][0],Qf[kt][1],Qf[kt][2],Qf[kt][3], kb0, kb2);
                mma16816(Sacc[2*np+1][0],Sacc[2*np+1][1],Sacc[2*np+1][2],Sacc[2*np+1][3],
                         Qf[kt][0],Qf[kt][1],Qf[kt][2],Qf[kt][3], kb1, kb3);
            }
        }

        // ---- scale + causal mask ----
        bool need_mask = (kbase + BN - 1) > (q0 + wr);
        #pragma unroll
        for (int nt=0; nt<8; nt++){
            int col0 = nt*8 + tig*2;
            int kp0 = kbase + col0;
            int kp1 = kp0 + 1;
            Sacc[nt][0]*=scale; Sacc[nt][1]*=scale; Sacc[nt][2]*=scale; Sacc[nt][3]*=scale;
            if (need_mask){
                if (kp0 > row_a) Sacc[nt][0] = -INFINITY;
                if (kp1 > row_a) Sacc[nt][1] = -INFINITY;
                if (kp0 > row_b) Sacc[nt][2] = -INFINITY;
                if (kp1 > row_b) Sacc[nt][3] = -INFINITY;
            }
        }

        // ---- row max ----
        float tmax_a = -INFINITY, tmax_b = -INFINITY;
        #pragma unroll
        for (int nt=0; nt<8; nt++){
            tmax_a = fmaxf(tmax_a, fmaxf(Sacc[nt][0], Sacc[nt][1]));
            tmax_b = fmaxf(tmax_b, fmaxf(Sacc[nt][2], Sacc[nt][3]));
        }
        tmax_a = fmaxf(tmax_a, __shfl_xor_sync(0xffffffff, tmax_a, 1));
        tmax_a = fmaxf(tmax_a, __shfl_xor_sync(0xffffffff, tmax_a, 2));
        tmax_b = fmaxf(tmax_b, __shfl_xor_sync(0xffffffff, tmax_b, 1));
        tmax_b = fmaxf(tmax_b, __shfl_xor_sync(0xffffffff, tmax_b, 2));

        float m_new_a = fmaxf(m_a, tmax_a);
        float m_new_b = fmaxf(m_b, tmax_b);
        float alpha_a = __expf(m_a - m_new_a);
        float alpha_b = __expf(m_b - m_new_b);

        l_a *= alpha_a; l_b *= alpha_b;
        #pragma unroll
        for (int dt=0; dt<16; dt++){
            Oacc[dt][0]*=alpha_a; Oacc[dt][1]*=alpha_a;
            Oacc[dt][2]*=alpha_b; Oacc[dt][3]*=alpha_b;
        }

        // ---- P = exp(S - m_new) ----
        float rsum_a=0.f, rsum_b=0.f;
        #pragma unroll
        for (int nt=0; nt<8; nt++){
            float p0 = __expf(Sacc[nt][0] - m_new_a);
            float p1 = __expf(Sacc[nt][1] - m_new_a);
            float p2 = __expf(Sacc[nt][2] - m_new_b);
            float p3 = __expf(Sacc[nt][3] - m_new_b);
            Sacc[nt][0]=p0;Sacc[nt][1]=p1;Sacc[nt][2]=p2;Sacc[nt][3]=p3;
            rsum_a += p0+p1; rsum_b += p2+p3;
        }
        rsum_a += __shfl_xor_sync(0xffffffff, rsum_a, 1);
        rsum_a += __shfl_xor_sync(0xffffffff, rsum_a, 2);
        rsum_b += __shfl_xor_sync(0xffffffff, rsum_b, 1);
        rsum_b += __shfl_xor_sync(0xffffffff, rsum_b, 2);
        l_a += rsum_a; l_b += rsum_b;
        m_a = m_new_a; m_b = m_new_b;

        // ---- O += P @ V ----
        #pragma unroll
        for (int kk=0; kk<4; kk++){
            uint32_t a0 = pack_f(Sacc[2*kk][0],   Sacc[2*kk][1]);
            uint32_t a1 = pack_f(Sacc[2*kk][2],   Sacc[2*kk][3]);
            uint32_t a2 = pack_f(Sacc[2*kk+1][0], Sacc[2*kk+1][1]);
            uint32_t a3 = pack_f(Sacc[2*kk+1][2], Sacc[2*kk+1][3]);
            #pragma unroll
            for (int dp=0; dp<8; dp++){
                uint32_t vb0,vb1,vb2,vb3;
                ldm_x4_trans(vb0,vb1,vb2,vb3, &Vs[(kk*16+lrow)*ST + dp*16 + lcol]);
                // vb0=b0/vb1=b1 for dt=2dp ; vb2=b0/vb3=b1 for dt=2dp+1
                mma16816(Oacc[2*dp][0],Oacc[2*dp][1],Oacc[2*dp][2],Oacc[2*dp][3],
                         a0,a1,a2,a3, vb0, vb1);
                mma16816(Oacc[2*dp+1][0],Oacc[2*dp+1][1],Oacc[2*dp+1][2],Oacc[2*dp+1][3],
                         a0,a1,a2,a3, vb2, vb3);
            }
        }
    }

    // ---- epilogue ----
    float inv_a = (l_a > 0.f) ? 1.f/l_a : 0.f;
    float inv_b = (l_b > 0.f) ? 1.f/l_b : 0.f;

    if (row_a < S){
        #pragma unroll
        for (int dt=0; dt<16; dt++){
            int col0 = dt*8 + tig*2;
            Og[(size_t)row_a*Dc + col0]   = __float2bfloat16(Oacc[dt][0]*inv_a);
            Og[(size_t)row_a*Dc + col0+1] = __float2bfloat16(Oacc[dt][1]*inv_a);
        }
    }
    if (row_b < S){
        #pragma unroll
        for (int dt=0; dt<16; dt++){
            int col0 = dt*8 + tig*2;
            Og[(size_t)row_b*Dc + col0]   = __float2bfloat16(Oacc[dt][2]*inv_b);
            Og[(size_t)row_b*Dc + col0+1] = __float2bfloat16(Oacc[dt][3]*inv_b);
        }
    }
    if (tig == 0){
        if (row_a < S) LSEg[row_a] = m_a + logf(l_a);
        if (row_b < S) LSEg[row_b] = m_b + logf(l_b);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE){
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int Bsz = (int)Q.size(0);
    int H   = (int)Q.size(1);
    int S   = (int)Q.size(2);

    const __nv_bfloat16* Qp = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* Kp = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* Vp = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* Op = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* Lp = static_cast<float*>(LSE.data_ptr());

    int num_qblocks = (S + BM - 1) / BM;
    dim3 grid(num_qblocks, H, Bsz);
    dim3 block(THREADS);
    size_t smem_bytes = (size_t)(BM + 2*BN) * ST * sizeof(__nv_bfloat16);

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    static bool attr_set = false;
    if (!attr_set){
        cudaFuncSetAttribute(mha_fwd, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
        attr_set = true;
    }

    mha_fwd<<<grid, block, smem_bytes, stream>>>(Qp, Kp, Vp, Op, Lp, S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel::run);

}  // namespace mha_kernel