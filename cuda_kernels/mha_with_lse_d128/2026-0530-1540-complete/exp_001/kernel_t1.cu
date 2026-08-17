#include <cuda_bf16.h>
#include <cuda_runtime.h>
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

namespace mha_d128 {

constexpr int D = 128, BM = 128, BN = 64, WARPS = 8, THREADS = 256;
#define FULLMASK 0xffffffffu
#define NEG_INF (-1e30f)

__device__ __forceinline__ uint32_t cvta(const void* p){
    return (uint32_t)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ void ldm_x4(uint32_t addr, uint32_t&r0,uint32_t&r1,uint32_t&r2,uint32_t&r3){
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
        : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void ldm_x4_trans(uint32_t addr, uint32_t&r0,uint32_t&r1,uint32_t&r2,uint32_t&r3){
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];\n"
        : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma16816(float* acc,
        uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3, uint32_t b0, uint32_t b1){
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(acc[0]),"+f"(acc[1]),"+f"(acc[2]),"+f"(acc[3])
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
__device__ __forceinline__ uint32_t pack2(float a,float b){
    __nv_bfloat162 v=__floats2bfloat162_rn(a,b);
    return *reinterpret_cast<uint32_t*>(&v);
}

__global__ void __launch_bounds__(THREADS) flash_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S, float scale)
{
    int qtile = blockIdx.x;
    int h     = blockIdx.y;
    int b     = blockIdx.z;
    int tid   = threadIdx.x;
    int lane  = tid & 31;
    int warp  = tid >> 5;
    int warp_base = warp * 16;
    int gID = lane >> 2;
    int tig = lane & 3;

    long bh = (long)(b * H + h) * S;
    const __nv_bfloat16* Qbh = Q + bh * D;
    const __nv_bfloat16* Kbh = K + bh * D;
    const __nv_bfloat16* Vbh = V + bh * D;
    __nv_bfloat16* Obh = O + bh * D;
    float* LSEbh = LSE + bh;

    extern __shared__ char smem[];
    __nv_bfloat16* Qs = (__nv_bfloat16*)smem;     // [BM][D]
    __nv_bfloat16* Ks = Qs + BM * D;              // [BN][D]
    __nv_bfloat16* Vs = Ks + BN * D;              // [BN][D]

    // Load Q tile
    #pragma unroll
    for (int v = tid; v < (BM * D) / 8; v += THREADS) {
        int row = (v * 8) / D;
        int dc  = (v * 8) % D;
        int gr  = qtile * BM + row;
        int4 q;
        if (gr < S) q = *reinterpret_cast<const int4*>(Qbh + (long)gr * D + dc);
        else        q = make_int4(0,0,0,0);
        *reinterpret_cast<int4*>(Qs + row * D + dc) = q;
    }
    __syncthreads();

    float Oacc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++) Oacc[i][j] = 0.f;
    float m0 = NEG_INF, m1 = NEG_INF, l0 = 0.f, l1 = 0.f;

    int num_kv = (S + BN - 1) / BN;
    for (int kv = 0; kv < num_kv; kv++) {
        int kt_base = kv * BN;

        __syncthreads();
        #pragma unroll
        for (int v = tid; v < (BN * D) / 8; v += THREADS) {
            int row = (v * 8) / D;
            int dc  = (v * 8) % D;
            int bn  = kt_base + row;
            int4 kk, vv;
            if (bn < S) {
                kk = *reinterpret_cast<const int4*>(Kbh + (long)bn * D + dc);
                vv = *reinterpret_cast<const int4*>(Vbh + (long)bn * D + dc);
            } else { kk = make_int4(0,0,0,0); vv = kk; }
            *reinterpret_cast<int4*>(Ks + row * D + dc) = kk;
            *reinterpret_cast<int4*>(Vs + row * D + dc) = vv;
        }
        __syncthreads();

        // ---- QK^T : Sacc[t] = [16 x 8] for t=0..7 (BN=64 keys) ----
        float Sacc[8][4];
        #pragma unroll
        for (int t = 0; t < 8; t++)
            #pragma unroll
            for (int j = 0; j < 4; j++) Sacc[t][j] = 0.f;

        int quad = lane >> 3, rr = lane & 7;
        int ro = rr + ((quad & 1) << 3);
        int co = (quad >= 2) ? 8 : 0;

        #pragma unroll
        for (int s = 0; s < 8; s++) {
            uint32_t qa0,qa1,qa2,qa3;
            ldm_x4(cvta(&Qs[(warp_base + ro) * D + 16 * s + co]), qa0,qa1,qa2,qa3);
            #pragma unroll
            for (int g = 0; g < 4; g++) {
                uint32_t R0,R1,R2,R3;
                ldm_x4(cvta(&Ks[(16 * g + ro) * D + 16 * s + co]), R0,R1,R2,R3);
                mma16816(Sacc[2*g],   qa0,qa1,qa2,qa3, R0, R2);
                mma16816(Sacc[2*g+1], qa0,qa1,qa2,qa3, R1, R3);
            }
        }

        // ---- softmax (online) ----
        #pragma unroll
        for (int t = 0; t < 8; t++) {
            int col = t * 8 + tig * 2;
            int k0 = kt_base + col;
            int k1 = k0 + 1;
            Sacc[t][0] = (k0 < S) ? Sacc[t][0]*scale : NEG_INF;
            Sacc[t][1] = (k1 < S) ? Sacc[t][1]*scale : NEG_INF;
            Sacc[t][2] = (k0 < S) ? Sacc[t][2]*scale : NEG_INF;
            Sacc[t][3] = (k1 < S) ? Sacc[t][3]*scale : NEG_INF;
        }
        float tm0 = NEG_INF, tm1 = NEG_INF;
        #pragma unroll
        for (int t = 0; t < 8; t++) {
            tm0 = fmaxf(tm0, fmaxf(Sacc[t][0], Sacc[t][1]));
            tm1 = fmaxf(tm1, fmaxf(Sacc[t][2], Sacc[t][3]));
        }
        tm0 = fmaxf(tm0, __shfl_xor_sync(FULLMASK, tm0, 1));
        tm0 = fmaxf(tm0, __shfl_xor_sync(FULLMASK, tm0, 2));
        tm1 = fmaxf(tm1, __shfl_xor_sync(FULLMASK, tm1, 1));
        tm1 = fmaxf(tm1, __shfl_xor_sync(FULLMASK, tm1, 2));

        float m0n = fmaxf(m0, tm0), m1n = fmaxf(m1, tm1);
        float corr0 = __expf(m0 - m0n), corr1 = __expf(m1 - m1n);

        #pragma unroll
        for (int nt = 0; nt < 16; nt++) {
            Oacc[nt][0] *= corr0; Oacc[nt][1] *= corr0;
            Oacc[nt][2] *= corr1; Oacc[nt][3] *= corr1;
        }
        l0 *= corr0; l1 *= corr1;

        uint32_t packlow[8], packhigh[8];
        float ps0 = 0.f, ps1 = 0.f;
        #pragma unroll
        for (int t = 0; t < 8; t++) {
            float p0 = __expf(Sacc[t][0] - m0n);
            float p1 = __expf(Sacc[t][1] - m0n);
            float p2 = __expf(Sacc[t][2] - m1n);
            float p3 = __expf(Sacc[t][3] - m1n);
            ps0 += p0 + p1; ps1 += p2 + p3;
            packlow[t]  = pack2(p0, p1);
            packhigh[t] = pack2(p2, p3);
        }
        ps0 += __shfl_xor_sync(FULLMASK, ps0, 1);
        ps0 += __shfl_xor_sync(FULLMASK, ps0, 2);
        ps1 += __shfl_xor_sync(FULLMASK, ps1, 1);
        ps1 += __shfl_xor_sync(FULLMASK, ps1, 2);
        l0 += ps0; l1 += ps1;
        m0 = m0n; m1 = m1n;

        // ---- PV : Oacc[nt] += P @ V ----
        #pragma unroll
        for (int ps = 0; ps < 4; ps++) {
            uint32_t a0 = packlow[2*ps], a1 = packhigh[2*ps];
            uint32_t a2 = packlow[2*ps+1], a3 = packhigh[2*ps+1];
            #pragma unroll
            for (int dt = 0; dt < 8; dt++) {
                uint32_t T0,T1,T2,T3;
                ldm_x4_trans(cvta(&Vs[(16 * ps + ro) * D + 16 * dt + co]), T0,T1,T2,T3);
                mma16816(Oacc[2*dt],   a0,a1,a2,a3, T0, T1);
                mma16816(Oacc[2*dt+1], a0,a1,a2,a3, T2, T3);
            }
        }
    }

    // ---- write O and LSE ----
    int gr0 = qtile * BM + warp_base + gID;
    int gr1 = gr0 + 8;
    float inv0 = 1.f / l0, inv1 = 1.f / l1;
    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        int d = nt * 8 + tig * 2;
        if (gr0 < S) {
            Obh[(long)gr0 * D + d]     = __float2bfloat16(Oacc[nt][0] * inv0);
            Obh[(long)gr0 * D + d + 1] = __float2bfloat16(Oacc[nt][1] * inv0);
        }
        if (gr1 < S) {
            Obh[(long)gr1 * D + d]     = __float2bfloat16(Oacc[nt][2] * inv1);
            Obh[(long)gr1 * D + d + 1] = __float2bfloat16(Oacc[nt][3] * inv1);
        }
    }
    if (tig == 0) {
        if (gr0 < S) LSEbh[gr0] = m0 + logf(l0);
        if (gr1 < S) LSEbh[gr1] = m1 + logf(l1);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int Bsz = (int)Q.size(0);
    int Hn  = (int)Q.size(1);
    int S   = (int)Q.size(2);

    float scale = 1.0f / sqrtf((float)D);

    const __nv_bfloat16* Qp = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* Kp = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* Vp = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* Op = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* Lp = static_cast<float*>(LSE.data_ptr());

    dim3 grid((S + BM - 1) / BM, Hn, Bsz);
    dim3 block(THREADS);

    size_t smem = (size_t)(BM * D + 2 * BN * D) * sizeof(__nv_bfloat16);

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    cudaFuncSetAttribute(flash_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);

    flash_kernel<<<grid, block, smem, stream>>>(Qp, Kp, Vp, Op, Lp, Bsz, Hn, S, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_d128::run);

}  // namespace mha_d128