#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

namespace tvm_mha {

constexpr int D = 128;
constexpr int BQ = 64;
constexpr int BK = 64;
constexpr int NUM_THREADS = 128;

__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ uint32_t pack_bf16(float a, float b) {
    __nv_bfloat16 ba = __float2bfloat16(a);
    __nv_bfloat16 bb = __float2bfloat16(b);
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};"
        : "=r"(result)
        : "h"(*reinterpret_cast<uint16_t*>(&ba)),
          "h"(*reinterpret_cast<uint16_t*>(&bb)));
    return result;
}

__global__ __launch_bounds__(128, 2) void mha_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S) {

    int bh = blockIdx.x;
    int q_block = blockIdx.y;
    int b = bh / H;
    int h = bh % H;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int64_t offset = (int64_t)b * H * S * D + (int64_t)h * S * D;
    const __nv_bfloat16* Q_base = Q + offset;
    const __nv_bfloat16* K_base = K + offset;
    const __nv_bfloat16* V_base = V + offset;
    __nv_bfloat16* O_base = O + offset;
    float* LSE_base = LSE + (int64_t)b * H * S + (int64_t)h * S;

    int q_start = q_block * BQ;

    extern __shared__ char smem[];
    __nv_bfloat16* Q_smem = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* K_smem[2];
    K_smem[0] = Q_smem + BQ * D;
    K_smem[1] = K_smem[0] + BK * D;
    __nv_bfloat16* V_smem[2];
    V_smem[0] = K_smem[1] + BK * D;
    V_smem[1] = V_smem[0] + BK * D;
    __nv_bfloat16* P_smem = reinterpret_cast<__nv_bfloat16*>(V_smem[1] + BK * D);
    float* m_smem = reinterpret_cast<float*>(P_smem + BQ * BK);
    float* l_smem = m_smem + BQ;

    const float scale = 0.08838834764831845f;

    // Load Q
    {
        constexpr int vpr = D / 8;
        const uint4* Q_vec = reinterpret_cast<const uint4*>(Q_base);
        uint4* Q_smem_vec = reinterpret_cast<uint4*>(Q_smem);
        for (int idx = tid; idx < BQ * vpr; idx += NUM_THREADS) {
            int qi = idx / vpr;
            int vi = idx % vpr;
            Q_smem_vec[idx] = (q_start + qi < S) ? Q_vec[(q_start + qi) * vpr + vi] : make_uint4(0, 0, 0, 0);
        }
    }

    for (int i = tid; i < BQ; i += NUM_THREADS) {
        m_smem[i] = -INFINITY;
        l_smem[i] = 0.0f;
    }

    using namespace nvcuda::wmma;
    fragment<accumulator, 16, 16, 16, float> o_frag[8];
    for (int n = 0; n < 8; n++) fill_fragment(o_frag[n], 0.0f);

    // Issue first K/V load
    {
        constexpr int vpr = D / 8;
        for (int idx = tid; idx < BK * vpr; idx += NUM_THREADS) {
            int ki = idx / vpr;
            int vi = idx % vpr;
            if (ki < S) {
                cp_async_16(&K_smem[0][ki * D + vi * 8], &K_base[ki * D + vi * 8]);
                cp_async_16(&V_smem[0][ki * D + vi * 8], &V_base[ki * D + vi * 8]);
            }
        }
    }
    cp_async_commit();
    __syncthreads();

    fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> b_frag_qk;
    fragment<matrix_b, 16, 16, 16, __nv_bfloat16, row_major> b_frag_pv;
    fragment<accumulator, 16, 16, 16, float> c_frag[4];

    int row_r0 = warp_id * 16 + lane_id / 4;
    int row_r1 = row_r0 + 8;
    int col_base = (lane_id % 4) * 2;
    bool valid_r0 = (row_r0 < BQ) && (q_start + row_r0 < S);
    bool valid_r1 = (row_r1 < BQ) && (q_start + row_r1 < S);

    for (int kv = 0; kv < S; kv += BK) {
        int buf = (kv / BK) % 2;

        // Issue next K/V load
        if (kv + BK < S) {
            constexpr int vpr = D / 8;
            for (int idx = tid; idx < BK * vpr; idx += NUM_THREADS) {
                int ki = idx / vpr;
                int vi = idx % vpr;
                int kv_idx = kv + BK + ki;
                if (kv_idx < S) {
                    cp_async_16(&K_smem[1-buf][ki * D + vi * 8], &K_base[kv_idx * D + vi * 8]);
                    cp_async_16(&V_smem[1-buf][ki * D + vi * 8], &V_base[kv_idx * D + vi * 8]);
                }
            }
            cp_async_commit();
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_group<0>();
        }
        __syncthreads();

        // Q@K^T -> c_frag (4 N-tiles, 8 K-tiles per warp)
        #pragma unroll
        for (int n = 0; n < BK / 16; n++) {
            fill_fragment(c_frag[n], 0.0f);
            #pragma unroll
            for (int k = 0; k < D / 16; k++) {
                load_matrix_sync(a_frag, Q_smem + warp_id * 16 * D + k * 16, D);
                load_matrix_sync(b_frag_qk, K_smem[buf] + n * 16 * D + k * 16, D);
                mma_sync(c_frag[n], a_frag, b_frag_qk, c_frag[n]);
            }
            #pragma unroll
            for (int i = 0; i < 8; i++) c_frag[n].x[i] *= scale;
        }

        // Online softmax from fragments - Step 1: row max
        float max_r0 = -INFINITY, max_r1 = -INFINITY;
        #pragma unroll
        for (int n = 0; n < BK / 16; n++) {
            int nc0 = n * 16 + col_base;
            int nc1 = nc0 + 8, nc2 = nc0 + 1, nc3 = nc0 + 9;
            bool kv0 = (kv + nc0 < S), kv1 = (kv + nc1 < S);
            bool kv2 = (kv + nc2 < S), kv3 = (kv + nc3 < S);

            max_r0 = fmaxf(max_r0, fmaxf(
                fmaxf(kv0 ? c_frag[n].x[0] : -INFINITY, kv2 ? c_frag[n].x[4] : -INFINITY),
                fmaxf(kv1 ? c_frag[n].x[1] : -INFINITY, kv3 ? c_frag[n].x[5] : -INFINITY)));
            max_r1 = fmaxf(max_r1, fmaxf(
                fmaxf(kv0 ? c_frag[n].x[2] : -INFINITY, kv2 ? c_frag[n].x[6] : -INFINITY),
                fmaxf(kv1 ? c_frag[n].x[3] : -INFINITY, kv3 ? c_frag[n].x[7] : -INFINITY)));
        }

        // Warp reduce max (4 threads per row)
        max_r0 = fmaxf(max_r0, __shfl_xor_sync(0xFFFFFFFF, max_r0, 2));
        max_r0 = fmaxf(max_r0, __shfl_xor_sync(0xFFFFFFFF, max_r0, 1));
        max_r1 = fmaxf(max_r1, __shfl_xor_sync(0xFFFFFFFF, max_r1, 2));
        max_r1 = fmaxf(max_r1, __shfl_xor_sync(0xFFFFFFFF, max_r1, 1));

        float m_old_r0 = valid_r0 ? m_smem[row_r0] : -INFINITY;
        float m_old_r1 = valid_r1 ? m_smem[row_r1] : -INFINITY;
        float m_new_r0 = fmaxf(m_old_r0, max_r0);
        float m_new_r1 = fmaxf(m_old_r1, max_r1);
        float rescale_r0 = (m_old_r0 == -INFINITY) ? 0.0f : __expf(m_old_r0 - m_new_r0);
        float rescale_r1 = (m_old_r1 == -INFINITY) ? 0.0f : __expf(m_old_r1 - m_new_r1);

        // Step 2: exp, sum, store P
        float sum_r0 = 0.0f, sum_r1 = 0.0f;
        #pragma unroll
        for (int n = 0; n < BK / 16; n++) {
            int nc0 = n * 16 + col_base;
            int nc1 = nc0 + 8, nc2 = nc0 + 1, nc3 = nc0 + 9;
            bool kv0 = (kv + nc0 < S), kv1 = (kv + nc1 < S);
            bool kv2 = (kv + nc2 < S), kv3 = (kv + nc3 < S);

            float e0 = kv0 ? __expf(c_frag[n].x[0] - m_new_r0) : 0.0f;
            float e1 = kv1 ? __expf(c_frag[n].x[1] - m_new_r0) : 0.0f;
            float e2 = kv2 ? __expf(c_frag[n].x[4] - m_new_r0) : 0.0f;
            float e3 = kv3 ? __expf(c_frag[n].x[5] - m_new_r0) : 0.0f;
            float e4 = kv0 ? __expf(c_frag[n].x[2] - m_new_r1) : 0.0f;
            float e5 = kv1 ? __expf(c_frag[n].x[3] - m_new_r1) : 0.0f;
            float e6 = kv2 ? __expf(c_frag[n].x[6] - m_new_r1) : 0.0f;
            float e7 = kv3 ? __expf(c_frag[n].x[7] - m_new_r1) : 0.0f;

            sum_r0 += e0 + e1 + e2 + e3;
            sum_r1 += e4 + e5 + e6 + e7;

            if (valid_r0) {
                int base = row_r0 * BK + n * 16 + col_base;
                *reinterpret_cast<uint32_t*>(&P_smem[base]) = pack_bf16(e0, e2);
                *reinterpret_cast<uint32_t*>(&P_smem[base + 8]) = pack_bf16(e1, e3);
            }
            if (valid_r1) {
                int base = row_r1 * BK + n * 16 + col_base;
                *reinterpret_cast<uint32_t*>(&P_smem[base]) = pack_bf16(e4, e6);
                *reinterpret_cast<uint32_t*>(&P_smem[base + 8]) = pack_bf16(e5, e7);
            }
        }

        // Warp reduce sum
        sum_r0 += __shfl_xor_sync(0xFFFFFFFF, sum_r0, 2);
        sum_r0 += __shfl_xor_sync(0xFFFFFFFF, sum_r0, 1);
        sum_r1 += __shfl_xor_sync(0xFFFFFFFF, sum_r1, 2);
        sum_r1 += __shfl_xor_sync(0xFFFFFFFF, sum_r1, 1);

        if (valid_r0) {
            l_smem[row_r0] = l_smem[row_r0] * rescale_r0 + sum_r0;
            m_smem[row_r0] = m_new_r0;
        }
        if (valid_r1) {
            l_smem[row_r1] = l_smem[row_r1] * rescale_r1 + sum_r1;
            m_smem[row_r1] = m_new_r1;
        }

        // Rescale O fragments in registers
        #pragma unroll
        for (int n = 0; n < 8; n++) {
            o_frag[n].x[0] *= rescale_r0; o_frag[n].x[1] *= rescale_r0;
            o_frag[n].x[4] *= rescale_r0; o_frag[n].x[5] *= rescale_r0;
            o_frag[n].x[2] *= rescale_r1; o_frag[n].x[3] *= rescale_r1;
            o_frag[n].x[6] *= rescale_r1; o_frag[n].x[7] *= rescale_r1;
        }

        // P@V -> accumulate into O fragments (8 N-tiles, 4 K-tiles per warp)
        #pragma unroll
        for (int n = 0; n < D / 16; n++) {
            #pragma unroll
            for (int k = 0; k < BK / 16; k++) {
                load_matrix_sync(a_frag, P_smem + warp_id * 16 * BK + k * 16, BK);
                load_matrix_sync(b_frag_pv, V_smem[buf] + k * 16 * D + n * 16, D);
                mma_sync(o_frag[n], a_frag, b_frag_pv, o_frag[n]);
            }
        }
    }

    // Epilogue: store O fragments to reused K_smem space, normalize, write to global
    float* O_smem = reinterpret_cast<float*>(K_smem[0]);
    #pragma unroll
    for (int n = 0; n < D / 16; n++) {
        store_matrix_sync(O_smem + warp_id * 16 * D + n * 16, o_frag[n], D, mem_row_major);
    }

    if (tid < BQ && q_start + tid < S) {
        LSE_base[q_start + tid] = m_smem[tid] + logf(l_smem[tid]);
    }

    __syncthreads();

    {
        constexpr int vpr = D / 8;
        for (int idx = tid; idx < BQ * vpr; idx += NUM_THREADS) {
            int row = idx / vpr;
            int vi = idx % vpr;
            if (q_start + row >= S) continue;
            float inv_l = 1.0f / l_smem[row];
            __nv_bfloat16 vals[8];
            #pragma unroll
            for (int d = 0; d < 8; d++) {
                vals[d] = __float2bfloat16(O_smem[row * D + vi * 8 + d] * inv_l);
            }
            *reinterpret_cast<uint4*>(O_base + (q_start + row) * D + vi * 8) =
                *reinterpret_cast<uint4*>(vals);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int S = (int)Q.size(2);

    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    int smem_size = BQ * D * 2 +        // Q
                    2 * BK * D * 2 +     // K (double buffered)
                    2 * BK * D * 2 +     // V (double buffered)
                    BQ * BK * 2 +        // P
                    BQ * 4 * 2;          // m, l

    CUDA_CHECK(cudaFuncSetAttribute(mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    dim3 grid(B * H, (S + BQ - 1) / BQ, 1);
    dim3 block(NUM_THREADS, 1, 1);

    mha_kernel<<<grid, block, smem_size, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data, B, H, S);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_mha::run);

}  // namespace tvm_mha