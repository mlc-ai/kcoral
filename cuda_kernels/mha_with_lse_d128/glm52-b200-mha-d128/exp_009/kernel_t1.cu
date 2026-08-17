#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", \
                cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

namespace mha_cuda {

constexpr int D = 128;
constexpr int D_PAD = 136;  // Bank-conflict-free padding
constexpr int BN_PAD = 68;  // Padding for transposed V layout
constexpr int BM = 64;      // Query tile size
constexpr int BN = 64;      // Key tile size
constexpr int NUM_THREADS = 128;
constexpr float SCALE = 0.08838834764831845f;  // 1/sqrt(128)
constexpr float LOG2E = 1.4426950408889634f;

__device__ __forceinline__ float fast_expf(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x * LOG2E));
    return y;
}

__device__ __forceinline__ uint32_t pack_bf16x2(float a, float b) {
    uint32_t result;
    __nv_bfloat16 ha = __float2bfloat16(a);
    __nv_bfloat16 hb = __float2bfloat16(b);
    asm("mov.b32 %0, {%1, %2};" : "=r"(result) : "h"(*(uint16_t*)&ha), "h"(*(uint16_t*)&hb));
    return result;
}

__device__ __forceinline__ void mma_m16n8k16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

__global__ __launch_bounds__(NUM_THREADS, 2)
void attention_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int H, int S
) {
    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* Q_smem = smem;
    __nv_bfloat16* KV_smem = Q_smem + BM * D_PAD;
    __nv_bfloat16* O_smem = KV_smem + BN * D_PAD;

    int q_start = blockIdx.x * BM;
    int bh = blockIdx.y;
    int b = bh / H;
    int h = bh % H;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int col_off = 2 * (lane_id % 4);
    int r0 = warp_id * 16 + lane_id / 4;
    int r1 = r0 + 8;

    int64_t base = (int64_t)(b * H + h) * S * D;
    const __nv_bfloat16* Q_bh = Q + base;
    const __nv_bfloat16* K_bh = K + base;
    const __nv_bfloat16* V_bh = V + base;

    // Load Q to smem [BM, D_PAD] row-major
    {
        int total = BM * (D / 8);
        for (int i = tid; i < total; i += NUM_THREADS) {
            int row = i / (D / 8), col = i % (D / 8);
            int q_row = q_start + row;
            int4* dst = reinterpret_cast<int4*>(Q_smem + row * D_PAD + col * 8);
            if (q_row < S) {
                *dst = *reinterpret_cast<const int4*>(Q_bh + (int64_t)q_row * D + col * 8);
            } else { *dst = make_int4(0,0,0,0); }
        }
    }
    __syncthreads();

    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    float o_acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) o_acc[i][0]=o_acc[i][1]=o_acc[i][2]=o_acc[i][3]=0.0f;

    for (int k_start = 0; k_start < S; k_start += BN) {
        // Load K to KV_smem [BN, D_PAD] row-major
        {
            int total = BN * (D / 8);
            for (int i = tid; i < total; i += NUM_THREADS) {
                int row = i / (D / 8), col = i % (D / 8);
                int k_row = k_start + row;
                int4* dst = reinterpret_cast<int4*>(KV_smem + row * D_PAD + col * 8);
                if (k_row < S) {
                    *dst = *reinterpret_cast<const int4*>(K_bh + (int64_t)k_row * D + col * 8);
                } else { *dst = make_int4(0,0,0,0); }
            }
        }
        __syncthreads();

        // S = Q @ K^T via mma.m16n8k16
        float scores[8][4];
        #pragma unroll
        for (int i = 0; i < 8; i++) scores[i][0]=scores[i][1]=scores[i][2]=scores[i][3]=0.0f;

        #pragma unroll
        for (int k_iter = 0; k_iter < 8; k_iter++) {
            uint32_t qa0, qa1, qa2, qa3;
            {
                int row0 = warp_id*16 + lane_id/4, row1 = row0 + 8;
                int d = k_iter*16 + col_off;
                qa0 = *reinterpret_cast<uint32_t*>(Q_smem + row0*D_PAD + d);
                qa1 = *reinterpret_cast<uint32_t*>(Q_smem + row0*D_PAD + d + 8);
                qa2 = *reinterpret_cast<uint32_t*>(Q_smem + row1*D_PAD + d);
                qa3 = *reinterpret_cast<uint32_t*>(Q_smem + row1*D_PAD + d + 8);
            }
            #pragma unroll
            for (int n_tile = 0; n_tile < 8; n_tile++) {
                uint32_t kb0, kb1;
                {
                    int key = n_tile*8 + lane_id/4;
                    int d = k_iter*16 + col_off;
                    kb0 = *reinterpret_cast<uint32_t*>(KV_smem + key*D_PAD + d);
                    kb1 = *reinterpret_cast<uint32_t*>(KV_smem + key*D_PAD + d + 8);
                }
                mma_m16n8k16(scores[n_tile][0], scores[n_tile][1],
                    scores[n_tile][2], scores[n_tile][3],
                    qa0, qa1, qa2, qa3, kb0, kb1,
                    scores[n_tile][0], scores[n_tile][1],
                    scores[n_tile][2], scores[n_tile][3]);
            }
        }

        // Scale and mask
        #pragma unroll
        for (int n_tile = 0; n_tile < 8; n_tile++) {
            int key0 = k_start + n_tile*8 + col_off;
            scores[n_tile][0] *= SCALE; scores[n_tile][1] *= SCALE;
            scores[n_tile][2] *= SCALE; scores[n_tile][3] *= SCALE;
            if (key0 >= S) { scores[n_tile][0]=-INFINITY; scores[n_tile][2]=-INFINITY; }
            if (key0+1 >= S) { scores[n_tile][1]=-INFINITY; scores[n_tile][3]=-INFINITY; }
        }

        __syncthreads();

        // Load V transposed: VT_smem[d][k] = V[k][d], stride BN_PAD
        {
            int total = (D/8) * BN;
            for (int i = tid; i < total; i += NUM_THREADS) {
                int k = i / (D/8), d_tile = i % (D/8);
                int k_row = k_start + k;
                if (k_row < S) {
                    int4 val = *reinterpret_cast<const int4*>(V_bh + (int64_t)k_row*D + d_tile*8);
                    __nv_bfloat16* vp = reinterpret_cast<__nv_bfloat16*>(&val);
                    #pragma unroll
                    for (int j = 0; j < 8; j++)
                        KV_smem[(d_tile*8+j)*BN_PAD + k] = vp[j];
                } else {
                    #pragma unroll
                    for (int j = 0; j < 8; j++)
                        KV_smem[(d_tile*8+j)*BN_PAD + k] = __float2bfloat16(0.0f);
                }
            }
        }
        __syncthreads();

        // Online softmax
        float m0n = m0, m1n = m1;
        #pragma unroll
        for (int n_tile = 0; n_tile < 8; n_tile++) {
            m0n = fmaxf(m0n, scores[n_tile][0]); m0n = fmaxf(m0n, scores[n_tile][1]);
            m1n = fmaxf(m1n, scores[n_tile][2]); m1n = fmaxf(m1n, scores[n_tile][3]);
        }
        m0n = fmaxf(m0n, __shfl_xor_sync(0xFFFFFFFF, m0n, 1));
        m0n = fmaxf(m0n, __shfl_xor_sync(0xFFFFFFFF, m0n, 2));
        m1n = fmaxf(m1n, __shfl_xor_sync(0xFFFFFFFF, m1n, 1));
        m1n = fmaxf(m1n, __shfl_xor_sync(0xFFFFFFFF, m1n, 2));

        float sc0 = (m0 > -INFINITY) ? fast_expf(m0 - m0n) : 0.0f;
        float sc1 = (m1 > -INFINITY) ? fast_expf(m1 - m1n) : 0.0f;
        l0 *= sc0; l1 *= sc1;
        #pragma unroll
        for (int n_tile = 0; n_tile < 16; n_tile++) {
            o_acc[n_tile][0] *= sc0; o_acc[n_tile][1] *= sc0;
            o_acc[n_tile][2] *= sc1; o_acc[n_tile][3] *= sc1;
        }
        #pragma unroll
        for (int n_tile = 0; n_tile < 8; n_tile++) {
            float p0=fast_expf(scores[n_tile][0]-m0n), p1=fast_expf(scores[n_tile][1]-m0n);
            float p2=fast_expf(scores[n_tile][2]-m1n), p3=fast_expf(scores[n_tile][3]-m1n);
            scores[n_tile][0]=p0; scores[n_tile][1]=p1; scores[n_tile][2]=p2; scores[n_tile][3]=p3;
            l0 += p0+p1; l1 += p2+p3;
        }
        m0 = m0n; m1 = m1n;

        // O += P @ V via mma
        #pragma unroll
        for (int k_iter = 0; k_iter < 4; k_iter++) {
            uint32_t pa0 = pack_bf16x2(scores[k_iter*2][0], scores[k_iter*2][1]);
            uint32_t pa1 = pack_bf16x2(scores[k_iter*2+1][0], scores[k_iter*2+1][1]);
            uint32_t pa2 = pack_bf16x2(scores[k_iter*2][2], scores[k_iter*2][3]);
            uint32_t pa3 = pack_bf16x2(scores[k_iter*2+1][2], scores[k_iter*2+1][3]);
            #pragma unroll
            for (int n_tile = 0; n_tile < 16; n_tile++) {
                uint32_t vb0, vb1;
                {
                    int d_head = n_tile*8 + lane_id/4;
                    int key_base = k_iter*16 + col_off;
                    vb0 = *reinterpret_cast<uint32_t*>(KV_smem + d_head*BN_PAD + key_base);
                    vb1 = *reinterpret_cast<uint32_t*>(KV_smem + d_head*BN_PAD + key_base + 8);
                }
                mma_m16n8k16(o_acc[n_tile][0], o_acc[n_tile][1],
                    o_acc[n_tile][2], o_acc[n_tile][3],
                    pa0, pa1, pa2, pa3, vb0, vb1,
                    o_acc[n_tile][0], o_acc[n_tile][1],
                    o_acc[n_tile][2], o_acc[n_tile][3]);
            }
        }
        __syncthreads();
    }

    // Normalize and write output
    float inv_l0 = (l0 > 0.0f) ? (1.0f/l0) : 0.0f;
    float inv_l1 = (l1 > 0.0f) ? (1.0f/l1) : 0.0f;

    #pragma unroll
    for (int n_tile = 0; n_tile < 16; n_tile++) {
        int c = n_tile*8 + col_off;
        *reinterpret_cast<uint32_t*>(O_smem + r0*D_PAD + c) = pack_bf16x2(o_acc[n_tile][0]*inv_l0, o_acc[n_tile][1]*inv_l0);
        *reinterpret_cast<uint32_t*>(O_smem + r1*D_PAD + c) = pack_bf16x2(o_acc[n_tile][2]*inv_l1, o_acc[n_tile][3]*inv_l1);
    }

    if (q_start + r0 < S) LSE[(int64_t)(b*H+h)*S + q_start + r0] = m0 + logf(l0);
    if (q_start + r1 < S) LSE[(int64_t)(b*H+h)*S + q_start + r1] = m1 + logf(l1);

    __syncthreads();

    // Coalesced O_smem -> global
    {
        int total = BM * (D / 8);
        for (int i = tid; i < total; i += NUM_THREADS) {
            int row = i / (D/8), col = i % (D/8);
            int q_row = q_start + row;
            if (q_row < S) {
                int4 val = *reinterpret_cast<int4*>(O_smem + row*D_PAD + col*8);
                *reinterpret_cast<int4*>(O + base + (int64_t)q_row*D + col*8) = val;
            }
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    const int B = 4, H = 48;
    int S = static_cast<int>(Q.size(2));

    const __nv_bfloat16* Qd = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* Kd = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* Vd = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* Od = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* Ld = static_cast<float*>(LSE.data_ptr());

    dim3 grid((S + BM - 1) / BM, B * H);
    dim3 block(NUM_THREADS);
    size_t smem = (BM * D_PAD + BN * D_PAD + BM * D_PAD) * sizeof(__nv_bfloat16);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(attention_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));

    attention_kernel<<<grid, block, smem, stream>>>(Qd, Kd, Vd, Od, Ld, H, S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_cuda::run);

}  // namespace mha_cuda