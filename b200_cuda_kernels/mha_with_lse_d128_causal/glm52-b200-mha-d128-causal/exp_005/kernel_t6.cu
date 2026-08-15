#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <cstdio>
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

namespace mha_kernel {

constexpr int D = 128;
constexpr int BM = 64;
constexpr int BK = 64;
constexpr int SQ_STRIDE = D + 8;
constexpr int SK_STRIDE = D + 8;
constexpr int SV_STRIDE = BK + 8;
constexpr int SP_STRIDE = BK + 8;
constexpr int NUM_WARPS = 4;
constexpr int THREADS = NUM_WARPS * 32;

__device__ __forceinline__ void mma_m16n8k16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

__global__ void attn_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int H, int S)
{
    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;
    int qb = blockIdx.y;
    int q_start = qb * BM;

    int64_t off = (int64_t)(b * H + h) * (int64_t)S * D;
    const __nv_bfloat16* Qg = Q + off;
    const __nv_bfloat16* Kg = K + off;
    const __nv_bfloat16* Vg = V + off;
    __nv_bfloat16* Og = O + off;
    float* LSEg = LSE + (int64_t)(b * H + h) * (int64_t)S;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid & 31;
    int warp_row = warp_id * 16;

    extern __shared__ char smem_buf[];
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem_buf);
    __nv_bfloat16* sK = sQ + BM * SQ_STRIDE;
    __nv_bfloat16* sV_t = sK + BK * SK_STRIDE;
    __nv_bfloat16* sP = sV_t + D * SV_STRIDE;

    const float scale = 0.08838834764831845f;

    // Load Q tile [BM, D]
    for (int i = tid; i < BM * D; i += THREADS) {
        int r = i / D, d = i % D;
        int gr = q_start + r;
        sQ[r * SQ_STRIDE + d] = (gr < S) ? Qg[(int64_t)gr * D + d] : __float2bfloat16(0.f);
    }
    __syncthreads();

    float o_frag[16][4];
    float m_row[2] = {-INFINITY, -INFINITY};
    float l_row[2] = {0.f, 0.f};

    #pragma unroll
    for (int i = 0; i < 16; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            o_frag[i][j] = 0.f;

    // MMA f32 output layout: d0=row/4,col=(lane%4)*2; d1=row/4+8,col=(lane%4)*2
    //                        d2=row/4,col=(lane%4)*2+1; d3=row/4+8,col=(lane%4)*2+1
    int row0 = warp_row + lane / 4;
    int row1 = warp_row + lane / 4 + 8;
    int q0 = q_start + row0;
    int q1 = q_start + row1;

    int block_max_q = (q_start + BM - 1 < S - 1) ? (q_start + BM - 1) : (S - 1);
    if (block_max_q < 0) block_max_q = 0;
    int last_kb = block_max_q / BK;

    for (int kb = 0; kb <= last_kb; kb++) {
        int k_start = kb * BK;

        // Load K [BK, D] and V transposed [D, BK]
        for (int i = tid; i < BK * D; i += THREADS) {
            int k = i / D, d = i % D;
            int gr = k_start + k;
            bool valid = (gr < S);
            sK[k * SK_STRIDE + d] = valid ? Kg[(int64_t)gr * D + d] : __float2bfloat16(0.f);
            sV_t[d * SV_STRIDE + k] = valid ? Vg[(int64_t)gr * D + d] : __float2bfloat16(0.f);
        }
        __syncthreads();

        // S = Q @ K^T: 8 k-iters (D/16), 8 n-tiles (BK/8)
        float s_frag[8][4];
        #pragma unroll
        for (int i = 0; i < 8; i++)
            #pragma unroll
            for (int j = 0; j < 4; j++)
                s_frag[i][j] = 0.f;

        #pragma unroll
        for (int k_iter = 0; k_iter < 8; k_iter++) {
            int k_off = k_iter * 16;
            // Load A fragment from Q (row-major, M×K=16×16)
            int r = lane / 4;
            int c = (lane % 4) * 2 + k_off;
            uint32_t a0 = *(const uint32_t*)(sQ + (warp_row + r) * SQ_STRIDE + c);
            uint32_t a1 = *(const uint32_t*)(sQ + (warp_row + r + 8) * SQ_STRIDE + c);
            uint32_t a2 = *(const uint32_t*)(sQ + (warp_row + r) * SQ_STRIDE + c + 8);
            uint32_t a3 = *(const uint32_t*)(sQ + (warp_row + r + 8) * SQ_STRIDE + c + 8);

            #pragma unroll
            for (int n_tile = 0; n_tile < 8; n_tile++) {
                // Load B fragment from K^T: B[k][n] = K[n][k], col-major
                int n = lane / 4 + n_tile * 8;
                int k = (lane % 4) * 2 + k_off;
                uint32_t b0 = *(const uint32_t*)(sK + n * SK_STRIDE + k);
                uint32_t b1 = *(const uint32_t*)(sK + n * SK_STRIDE + k + 8);

                mma_m16n8k16(
                    s_frag[n_tile][0], s_frag[n_tile][1],
                    s_frag[n_tile][2], s_frag[n_tile][3],
                    a0, a1, a2, a3, b0, b1,
                    s_frag[n_tile][0], s_frag[n_tile][1],
                    s_frag[n_tile][2], s_frag[n_tile][3]);
            }
        }

        // Apply scale and causal mask
        // Fragment layout: [n_tile][0]=row0,col=(lane%4)*2
        //                  [n_tile][1]=row1,col=(lane%4)*2
        //                  [n_tile][2]=row0,col=(lane%4)*2+1
        //                  [n_tile][3]=row1,col=(lane%4)*2+1
        #pragma unroll
        for (int n_tile = 0; n_tile < 8; n_tile++) {
            int col_even = n_tile * 8 + (lane % 4) * 2;
            int col_odd  = n_tile * 8 + (lane % 4) * 2 + 1;
            int k_g_even = k_start + col_even;
            int k_g_odd  = k_start + col_odd;

            s_frag[n_tile][0] *= scale;
            s_frag[n_tile][1] *= scale;
            s_frag[n_tile][2] *= scale;
            s_frag[n_tile][3] *= scale;

            if (k_g_even > q0 || q0 >= S) s_frag[n_tile][0] = -INFINITY;
            if (k_g_even > q1 || q1 >= S) s_frag[n_tile][1] = -INFINITY;
            if (k_g_odd  > q0 || q0 >= S) s_frag[n_tile][2] = -INFINITY;
            if (k_g_odd  > q1 || q1 >= S) s_frag[n_tile][3] = -INFINITY;
        }

        // Online softmax: row max
        float local_max0 = -INFINITY, local_max1 = -INFINITY;
        #pragma unroll
        for (int n_tile = 0; n_tile < 8; n_tile++) {
            local_max0 = fmaxf(local_max0, s_frag[n_tile][0]);
            local_max0 = fmaxf(local_max0, s_frag[n_tile][2]);
            local_max1 = fmaxf(local_max1, s_frag[n_tile][1]);
            local_max1 = fmaxf(local_max1, s_frag[n_tile][3]);
        }
        local_max0 = fmaxf(local_max0, __shfl_xor_sync(0xffffffff, local_max0, 1));
        local_max0 = fmaxf(local_max0, __shfl_xor_sync(0xffffffff, local_max0, 2));
        local_max1 = fmaxf(local_max1, __shfl_xor_sync(0xffffffff, local_max1, 1));
        local_max1 = fmaxf(local_max1, __shfl_xor_sync(0xffffffff, local_max1, 2));

        float m_new0 = fmaxf(m_row[0], local_max0);
        float m_new1 = fmaxf(m_row[1], local_max1);

        float alpha0 = (m_row[0] == -INFINITY) ? 0.f : __expf(m_row[0] - m_new0);
        float alpha1 = (m_row[1] == -INFINITY) ? 0.f : __expf(m_row[1] - m_new1);
        m_row[0] = m_new0;
        m_row[1] = m_new1;
        l_row[0] *= alpha0;
        l_row[1] *= alpha1;

        #pragma unroll
        for (int n_tile = 0; n_tile < 16; n_tile++) {
            o_frag[n_tile][0] *= alpha0;
            o_frag[n_tile][1] *= alpha1;
            o_frag[n_tile][2] *= alpha0;
            o_frag[n_tile][3] *= alpha1;
        }

        // P = exp(S - m), row sum
        float row_sum0 = 0.f, row_sum1 = 0.f;
        #pragma unroll
        for (int n_tile = 0; n_tile < 8; n_tile++) {
            float p0 = (s_frag[n_tile][0] == -INFINITY) ? 0.f : __expf(s_frag[n_tile][0] - m_row[0]);
            float p1 = (s_frag[n_tile][1] == -INFINITY) ? 0.f : __expf(s_frag[n_tile][1] - m_row[1]);
            float p2 = (s_frag[n_tile][2] == -INFINITY) ? 0.f : __expf(s_frag[n_tile][2] - m_row[0]);
            float p3 = (s_frag[n_tile][3] == -INFINITY) ? 0.f : __expf(s_frag[n_tile][3] - m_row[1]);
            s_frag[n_tile][0] = p0;
            s_frag[n_tile][1] = p1;
            s_frag[n_tile][2] = p2;
            s_frag[n_tile][3] = p3;
            row_sum0 += p0 + p2;
            row_sum1 += p1 + p3;
        }
        row_sum0 += __shfl_xor_sync(0xffffffff, row_sum0, 1);
        row_sum0 += __shfl_xor_sync(0xffffffff, row_sum0, 2);
        row_sum1 += __shfl_xor_sync(0xffffffff, row_sum1, 1);
        row_sum1 += __shfl_xor_sync(0xffffffff, row_sum1, 2);
        l_row[0] += row_sum0;
        l_row[1] += row_sum1;

        // Store P to shared memory as bf16
        #pragma unroll
        for (int n_tile = 0; n_tile < 8; n_tile++) {
            int col_even = n_tile * 8 + (lane % 4) * 2;
            int col_odd  = n_tile * 8 + (lane % 4) * 2 + 1;
            sP[row0 * SP_STRIDE + col_even] = __float2bfloat16(s_frag[n_tile][0]);
            sP[row1 * SP_STRIDE + col_even] = __float2bfloat16(s_frag[n_tile][1]);
            sP[row0 * SP_STRIDE + col_odd]  = __float2bfloat16(s_frag[n_tile][2]);
            sP[row1 * SP_STRIDE + col_odd]  = __float2bfloat16(s_frag[n_tile][3]);
        }
        __syncthreads();

        // O += P @ V: 4 k-iters (BK/16), 16 n-tiles (D/8)
        #pragma unroll
        for (int k_iter = 0; k_iter < 4; k_iter++) {
            int k_off = k_iter * 16;
            // Load A fragment from P (row-major)
            int r = lane / 4;
            int c = (lane % 4) * 2 + k_off;
            uint32_t a0 = *(const uint32_t*)(sP + (warp_row + r) * SP_STRIDE + c);
            uint32_t a1 = *(const uint32_t*)(sP + (warp_row + r + 8) * SP_STRIDE + c);
            uint32_t a2 = *(const uint32_t*)(sP + (warp_row + r) * SP_STRIDE + c + 8);
            uint32_t a3 = *(const uint32_t*)(sP + (warp_row + r + 8) * SP_STRIDE + c + 8);

            #pragma unroll
            for (int n_tile = 0; n_tile < 16; n_tile++) {
                // Load B fragment from V^T: B[k][n] = V[k][n] = sV_t[n][k], col-major
                int n = lane / 4 + n_tile * 8;
                int k = (lane % 4) * 2 + k_off;
                uint32_t b0 = *(const uint32_t*)(sV_t + n * SV_STRIDE + k);
                uint32_t b1 = *(const uint32_t*)(sV_t + n * SV_STRIDE + k + 8);

                mma_m16n8k16(
                    o_frag[n_tile][0], o_frag[n_tile][1],
                    o_frag[n_tile][2], o_frag[n_tile][3],
                    a0, a1, a2, a3, b0, b1,
                    o_frag[n_tile][0], o_frag[n_tile][1],
                    o_frag[n_tile][2], o_frag[n_tile][3]);
            }
        }
        __syncthreads();
    }

    // Write O: fragment layout d0=row0,col_even; d1=row1,col_even; d2=row0,col_odd; d3=row1,col_odd
    if (q0 < S) {
        float inv_l0 = 1.f / l_row[0];
        #pragma unroll
        for (int n_tile = 0; n_tile < 16; n_tile++) {
            int col_even = n_tile * 8 + (lane % 4) * 2;
            int col_odd  = n_tile * 8 + (lane % 4) * 2 + 1;
            Og[(int64_t)q0 * D + col_even] = __float2bfloat16(o_frag[n_tile][0] * inv_l0);
            Og[(int64_t)q0 * D + col_odd]  = __float2bfloat16(o_frag[n_tile][2] * inv_l0);
        }
    }
    if (q1 < S) {
        float inv_l1 = 1.f / l_row[1];
        #pragma unroll
        for (int n_tile = 0; n_tile < 16; n_tile++) {
            int col_even = n_tile * 8 + (lane % 4) * 2;
            int col_odd  = n_tile * 8 + (lane % 4) * 2 + 1;
            Og[(int64_t)q1 * D + col_even] = __float2bfloat16(o_frag[n_tile][1] * inv_l1);
            Og[(int64_t)q1 * D + col_odd]  = __float2bfloat16(o_frag[n_tile][3] * inv_l1);
        }
    }

    // Write LSE
    if (lane % 4 == 0) {
        if (q0 < S) LSEg[q0] = m_row[0] + logf(l_row[0]);
        if (q1 < S) LSEg[q1] = m_row[1] + logf(l_row[1]);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    const int B = 4;
    const int H = 48;
    int S = (int)Q.size(2);

    const __nv_bfloat16* Qd = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* Kd = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* Vd = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* Od = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSEd = static_cast<float*>(LSE.data_ptr());

    int smem_bytes = (int)((BM * SQ_STRIDE + BK * SK_STRIDE + D * SV_STRIDE + BM * SP_STRIDE) * sizeof(__nv_bfloat16));

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(
        attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    dim3 grid(B * H, (S + BM - 1) / BM);
    dim3 block(THREADS);
    attn_kernel<<<grid, block, smem_bytes, stream>>>(Qd, Kd, Vd, Od, LSEd, H, S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel::run);

}  // namespace mha_kernel