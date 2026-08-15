#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                   \
    cudaError_t _e = (call);                                     \
    if (_e != cudaSuccess) {                                     \
        fprintf(stderr, "CUDA error %s at %s:%d\n",              \
                cudaGetErrorString(_e), __FILE__, __LINE__);      \
        exit(1);                                                  \
    }                                                             \
} while(0)

namespace attn_fwd {

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int D  = 128;
constexpr int THREADS = 128;
constexpr float SCALE = 0.08838834764831845f;

__device__ __forceinline__ void mma_m16n8k16_row_col(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

__device__ __forceinline__ void mma_m16n8k16_row_row(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.row.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

__global__ __launch_bounds__(THREADS, 4)
void attention_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S)
{
    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;
    int q_block = blockIdx.y;
    int m_start = q_block * BM;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;

    extern __shared__ char smem[];
    __nv_bfloat16* sQ = (__nv_bfloat16*)smem;
    __nv_bfloat16* sK = sQ + BM * D;
    __nv_bfloat16* sV = sK + BN * D;
    float* sS = (float*)(sV + BN * D);
    __nv_bfloat16* sP = (__nv_bfloat16*)sS;
    float* sAlpha = (float*)(sS + BM * BN);

    int64_t base = ((int64_t)b * H + h) * S * D;

    // Load Q
    for (int i = tid; i < BM; i += THREADS) {
        int m = m_start + i;
        if (m < S) {
            #pragma unroll
            for (int d = 0; d < D; d += 8)
                *(int4*)&sQ[i * D + d] = *(const int4*)&Q[base + (int64_t)m * D + d];
        } else {
            #pragma unroll
            for (int d = 0; d < D; d += 8)
                *(int4*)&sQ[i * D + d] = make_int4(0, 0, 0, 0);
        }
    }

    float my_rowmax = -INFINITY;
    float my_rowsum = 0.0f;

    // O accumulator: D/8=16 tiles, 4 floats each
    float o[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) { o[i][0]=0; o[i][1]=0; o[i][2]=0; o[i][3]=0; }

    __syncthreads();

    // Precompute fragment layout constants
    int row0 = warp_id * 16 + lane / 4;
    int row1 = row0 + 8;
    int col_a = (lane % 4) * 4;
    int col_d = (lane % 4) * 2;

    for (int kv_start = 0; kv_start < S; kv_start += BN) {
        // Load K and V
        for (int i = tid; i < BN; i += THREADS) {
            int kv = kv_start + i;
            if (kv < S) {
                #pragma unroll
                for (int d = 0; d < D; d += 8) {
                    *(int4*)&sK[i * D + d] = *(const int4*)&K[base + (int64_t)kv * D + d];
                    *(int4*)&sV[i * D + d] = *(const int4*)&V[base + (int64_t)kv * D + d];
                }
            } else {
                #pragma unroll
                for (int d = 0; d < D; d += 8) {
                    *(int4*)&sK[i * D + d] = make_int4(0,0,0,0);
                    *(int4*)&sV[i * D + d] = make_int4(0,0,0,0);
                }
            }
        }
        __syncthreads();

        // QK^T: each warp does 1 M-tile x 8 N-tiles, 8 K-steps
        #pragma unroll
        for (int ni = 0; ni < BN / 8; ni++) {
            float s0=0, s1=0, s2=0, s3=0;
            #pragma unroll
            for (int ki = 0; ki < D / 16; ki++) {
                uint32_t a0 = *(uint32_t*)&sQ[row0 * D + ki*16 + col_a];
                uint32_t a1 = *(uint32_t*)&sQ[row0 * D + ki*16 + col_a + 2];
                uint32_t a2 = *(uint32_t*)&sQ[row1 * D + ki*16 + col_a];
                uint32_t a3 = *(uint32_t*)&sQ[row1 * D + ki*16 + col_a + 2];
                int n = ni * 8 + lane / 4;
                uint32_t b0 = *(uint32_t*)&sK[n * D + ki*16 + col_a];
                uint32_t b1 = *(uint32_t*)&sK[n * D + ki*16 + col_a + 2];
                mma_m16n8k16_row_col(s0, s1, s2, s3, a0, a1, a2, a3, b0, b1, s0, s1, s2, s3);
            }
            s0 *= SCALE; s1 *= SCALE; s2 *= SCALE; s3 *= SCALE;
            int sc = ni * 8 + col_d;
            sS[row0 * BN + sc]     = s0;
            sS[row0 * BN + sc + 1] = s1;
            sS[row1 * BN + sc]     = s2;
            sS[row1 * BN + sc + 1] = s3;
        }
        __syncthreads();

        // Softmax: thread tid -> row tid
        int row = tid;
        int m = m_start + row;
        float alpha = 1.0f;
        if (m < S) {
            int kv_end = min(kv_start + BN, S);
            float old_max = my_rowmax;
            float new_max = old_max;
            #pragma unroll
            for (int n = 0; n < BN; n++)
                if (kv_start + n < kv_end) new_max = fmaxf(new_max, sS[row * BN + n]);
            alpha = __expf(old_max - new_max);
            float p_sum = 0.0f;
            #pragma unroll
            for (int n = 0; n < BN; n++) {
                float s_val = (kv_start + n < kv_end) ? sS[row * BN + n] : -INFINITY;
                float p = __expf(s_val - new_max);
                sP[row * BN + n] = __float2bfloat16(p);
                p_sum += p;
            }
            my_rowmax = new_max;
            my_rowsum = my_rowsum * alpha + p_sum;
        }
        sAlpha[row] = alpha;
        __syncthreads();

        // Rescale O in registers by alpha
        float alpha0 = sAlpha[row0];
        float alpha1 = sAlpha[row1];
        #pragma unroll
        for (int ni = 0; ni < D / 8; ni++) {
            o[ni][0] *= alpha0; o[ni][1] *= alpha0;
            o[ni][2] *= alpha1; o[ni][3] *= alpha1;
        }

        // PV: O += P @ V, each warp does 1 M-tile x 16 N-tiles, 4 K-steps
        #pragma unroll
        for (int ni = 0; ni < D / 8; ni++) {
            #pragma unroll
            for (int ki = 0; ki < BN / 16; ki++) {
                uint32_t a0 = *(uint32_t*)&sP[row0 * BN + ki*16 + col_a];
                uint32_t a1 = *(uint32_t*)&sP[row0 * BN + ki*16 + col_a + 2];
                uint32_t a2 = *(uint32_t*)&sP[row1 * BN + ki*16 + col_a];
                uint32_t a3 = *(uint32_t*)&sP[row1 * BN + ki*16 + col_a + 2];
                int kr  = ki * 16 + lane / 4;
                int kr2 = kr + 8;
                uint32_t b0 = *(uint32_t*)&sV[kr  * D + ni*8 + col_d];
                uint32_t b1 = *(uint32_t*)&sV[kr2 * D + ni*8 + col_d];
                mma_m16n8k16_row_row(o[ni][0], o[ni][1], o[ni][2], o[ni][3],
                                     a0, a1, a2, a3, b0, b1,
                                     o[ni][0], o[ni][1], o[ni][2], o[ni][3]);
            }
        }
        __syncthreads();
    }

    // Final normalization
    sAlpha[tid] = (tid < BM && m_start + tid < S) ? (1.0f / my_rowsum) : 0.0f;
    __syncthreads();

    float inv0 = sAlpha[row0];
    float inv1 = sAlpha[row1];
    #pragma unroll
    for (int ni = 0; ni < D / 8; ni++) {
        o[ni][0] *= inv0; o[ni][1] *= inv0;
        o[ni][2] *= inv1; o[ni][3] *= inv1;
    }

    // Store O to global
    #pragma unroll
    for (int ni = 0; ni < D / 8; ni++) {
        int sc = ni * 8 + col_d;
        int gm0 = m_start + row0;
        int gm1 = m_start + row1;
        if (gm0 < S) {
            O[base + (int64_t)gm0 * D + sc]     = __float2bfloat16(o[ni][0]);
            O[base + (int64_t)gm0 * D + sc + 1] = __float2bfloat16(o[ni][1]);
        }
        if (gm1 < S) {
            O[base + (int64_t)gm1 * D + sc]     = __float2bfloat16(o[ni][2]);
            O[base + (int64_t)gm1 * D + sc + 1] = __float2bfloat16(o[ni][3]);
        }
    }

    // Store LSE
    if (tid < BM && m_start + tid < S) {
        LSE[((int64_t)b * H + h) * S + m_start + tid] = my_rowmax + __logf(my_rowsum);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    const int B = 4;
    const int H = 48;
    const int S = static_cast<int>(Q.size(2));

    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data       = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data             = static_cast<float*>(LSE.data_ptr());

    const int num_q_blocks = (S + BM - 1) / BM;
    dim3 grid(B * H, num_q_blocks);
    dim3 block(THREADS);

    size_t smem_size =
        (size_t)(BM * D) * sizeof(__nv_bfloat16)       // sQ
      + (size_t)(BN * D) * sizeof(__nv_bfloat16) * 2    // sK + sV
      + (size_t)(BM * BN) * sizeof(float)               // sS (sP overlays)
      + (size_t)(BM) * sizeof(float);                    // sAlpha

    CUDA_CHECK(cudaFuncSetAttribute(
        attention_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem_size));

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    attention_kernel<<<grid, block, smem_size, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data, B, H, S);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, attn_fwd::run);

} // namespace attn_fwd