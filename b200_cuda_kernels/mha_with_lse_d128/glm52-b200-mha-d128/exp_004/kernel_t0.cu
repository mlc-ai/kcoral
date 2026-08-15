#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <mma.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda;

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_mha_cuda {

// Constants
constexpr int D_HEAD = 128;
constexpr int BM = 64;
constexpr int BK = 64;
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int NUM_THREADS = 128;
constexpr int NUM_WARPS = 4;
constexpr int NUM_N_TILES_O = D_HEAD / WMMA_N;  // 8
constexpr int NUM_N_TILES_S = BK / WMMA_N;       // 4
constexpr int NUM_K_TILES_QK = D_HEAD / WMMA_K;  // 8
constexpr int NUM_K_TILES_PV = BK / WMMA_K;      // 4

// Shared memory layout (total ~55KB):
// Q_smem:  BM * D * 2 = 16384 bytes (reused as O_bf16 at end)
// KV_smem: BK * D * 2 = 16384 bytes
// S_smem:  BM * BK * 4 = 16384 bytes
// P_smem:  BM * BK * 2 = 8192 bytes
// m_smem:  BM * 4 = 256 bytes
// l_smem:  BM * 4 = 256 bytes
constexpr int SMEM_SIZE = BM * D_HEAD * 2 + BK * D_HEAD * 2 + BM * BK * 4 + BM * BK * 2 + BM * 4 * 2;

__global__ void attention_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S, int B, int H)
{
    const int bh = blockIdx.y;
    const int b = bh / H;
    const int h = bh % H;
    const int q_block_idx = blockIdx.x;
    const int q_start = q_block_idx * BM;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int m_row_start = warp_id * WMMA_M;

    // Shared memory pointers
    extern __shared__ char smem[];
    __nv_bfloat16* Q_smem = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* KV_smem = Q_smem + BM * D_HEAD;
    float* S_smem = reinterpret_cast<float*>(KV_smem + BK * D_HEAD);
    __nv_bfloat16* P_smem = reinterpret_cast<__nv_bfloat16*>(S_smem + BM * BK);
    float* m_smem = reinterpret_cast<float*>(P_smem + BM * BK);
    float* l_smem = m_smem + BM;

    constexpr float scale = 0.08838834764f; // 1/sqrt(128)

    // Base pointers for this (b, h)
    const size_t bh_offset = (static_cast<size_t>(b) * H + h) * S * D_HEAD;
    const __nv_bfloat16* Q_bh = Q + bh_offset;
    const __nv_bfloat16* K_bh = K + bh_offset;
    const __nv_bfloat16* V_bh = V + bh_offset;
    __nv_bfloat16* O_bh = O + bh_offset;
    float* LSE_bh = LSE + (static_cast<size_t>(b) * H + h) * S;

    // Load Q tile with vectorized uint4 loads (8 bf16 per load)
    {
        const uint4* Q_g = reinterpret_cast<const uint4*>(Q_bh + static_cast<size_t>(q_start) * D_HEAD);
        uint4* Q_s = reinterpret_cast<uint4*>(Q_smem);
        const int total_vecs = BM * D_HEAD / 8;
        for (int i = threadIdx.x; i < total_vecs; i += blockDim.x) {
            const int row = i / (D_HEAD / 8);
            Q_s[i] = (q_start + row < S) ? Q_g[i] : make_uint4(0, 0, 0, 0);
        }
    }

    // Initialize m and l
    for (int i = threadIdx.x; i < BM; i += blockDim.x) {
        m_smem[i] = -INFINITY;
        l_smem[i] = 0.0f;
    }

    // Initialize O fragments to 0 (kept in registers across k_blocks)
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag[NUM_N_TILES_O];
    #pragma unroll
    for (int n = 0; n < NUM_N_TILES_O; n++) {
        wmma::fill_fragment(o_frag[n], 0.0f);
    }

    __syncthreads();

    // Fragment row mapping: each thread owns 2 rows
    const int row0 = m_row_start + lane_id / 4;
    const int row1 = m_row_start + lane_id / 4 + 8;
    const int sub_col = (lane_id % 4) * (BK / 4); // 16 elements per thread per row
    const bool valid0 = (q_start + row0 < S);
    const bool valid1 = (q_start + row1 < S);

    // Main loop over K/V blocks
    for (int k_start = 0; k_start < S; k_start += BK) {
        const int k_len = min(k_start + BK, S) - k_start;

        // Load K tile
        {
            const uint4* K_g = reinterpret_cast<const uint4*>(K_bh + static_cast<size_t>(k_start) * D_HEAD);
            uint4* K_s = reinterpret_cast<uint4*>(KV_smem);
            const int total_vecs = BK * D_HEAD / 8;
            for (int i = threadIdx.x; i < total_vecs; i += blockDim.x) {
                const int row = i / (D_HEAD / 8);
                K_s[i] = (k_start + row < S) ? K_g[i] : make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        // Compute S = Q @ K^T using wmma (unscaled)
        #pragma unroll
        for (int n_tile = 0; n_tile < NUM_N_TILES_S; n_tile++) {
            const int n_col_start = n_tile * WMMA_N;

            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_frag;

            wmma::fill_fragment(s_frag, 0.0f);

            #pragma unroll
            for (int k_tile = 0; k_tile < NUM_K_TILES_QK; k_tile++) {
                const int k_offset = k_tile * WMMA_K;
                wmma::load_matrix_sync(a_frag, Q_smem + m_row_start * D_HEAD + k_offset, D_HEAD);
                wmma::load_matrix_sync(b_frag, KV_smem + n_col_start * D_HEAD + k_offset, D_HEAD);
                wmma::mma_sync(s_frag, a_frag, b_frag, s_frag);
            }

            wmma::store_matrix_sync(S_smem + m_row_start * BK + n_col_start, s_frag, BK, wmma::mem_row_major);
        }

        __syncthreads();

        // Online softmax with integrated scale and masking
        // Phase 1: Read S_smem, apply scale and mask, compute partial max (cache values)
        float s0_cache[16], s1_cache[16];
        float pmax0 = -INFINITY, pmax1 = -INFINITY;

        if (valid0) {
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                float val = S_smem[row0 * BK + sub_col + j];
                val = (sub_col + j >= k_len) ? -INFINITY : val * scale;
                s0_cache[j] = val;
                pmax0 = fmaxf(pmax0, val);
            }
        }
        if (valid1) {
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                float val = S_smem[row1 * BK + sub_col + j];
                val = (sub_col + j >= k_len) ? -INFINITY : val * scale;
                s1_cache[j] = val;
                pmax1 = fmaxf(pmax1, val);
            }
        }

        // Reduce max across 4 threads in the group (xor-based butterfly)
        pmax0 = fmaxf(pmax0, __shfl_xor_sync(0xFFFFFFFF, pmax0, 1));
        pmax0 = fmaxf(pmax0, __shfl_xor_sync(0xFFFFFFFF, pmax0, 2));
        pmax1 = fmaxf(pmax1, __shfl_xor_sync(0xFFFFFFFF, pmax1, 1));
        pmax1 = fmaxf(pmax1, __shfl_xor_sync(0xFFFFFFFF, pmax1, 2));

        // Combine with running max
        const float m_old0 = valid0 ? m_smem[row0] : -INFINITY;
        const float m_old1 = valid1 ? m_smem[row1] : -INFINITY;
        const float m_new0 = fmaxf(m_old0, pmax0);
        const float m_new1 = fmaxf(m_old1, pmax1);

        // Phase 2: Compute P = exp(S - m_new), write bf16 to P_smem, compute partial sum
        float psum0 = 0.0f, psum1 = 0.0f;

        if (valid0) {
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                float p = __expf(s0_cache[j] - m_new0);
                P_smem[row0 * BK + sub_col + j] = __float2bfloat16(p);
                psum0 += p;
            }
        }
        if (valid1) {
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                float p = __expf(s1_cache[j] - m_new1);
                P_smem[row1 * BK + sub_col + j] = __float2bfloat16(p);
                psum1 += p;
            }
        }

        // Reduce sum across 4 threads
        psum0 += __shfl_xor_sync(0xFFFFFFFF, psum0, 1);
        psum0 += __shfl_xor_sync(0xFFFFFFFF, psum0, 2);
        psum1 += __shfl_xor_sync(0xFFFFFFFF, psum1, 1);
        psum1 += __shfl_xor_sync(0xFFFFFFFF, psum1, 2);

        // Rescale O fragments (rows: x[0,1,4,5]=row0, x[2,3,6,7]=row1)
        const float rescale0 = (m_new0 == -INFINITY) ? 1.0f : __expf(m_old0 - m_new0);
        const float rescale1 = (m_new1 == -INFINITY) ? 1.0f : __expf(m_old1 - m_new1);

        #pragma unroll
        for (int n = 0; n < NUM_N_TILES_O; n++) {
            o_frag[n].x[0] *= rescale0;
            o_frag[n].x[1] *= rescale0;
            o_frag[n].x[2] *= rescale1;
            o_frag[n].x[3] *= rescale1;
            o_frag[n].x[4] *= rescale0;
            o_frag[n].x[5] *= rescale0;
            o_frag[n].x[6] *= rescale1;
            o_frag[n].x[7] *= rescale1;
        }

        // Update m and l in shared memory (only group leader writes)
        if (valid0 && (lane_id & 3) == 0) {
            m_smem[row0] = m_new0;
            l_smem[row0] = l_smem[row0] * rescale0 + psum0;
        }
        if (valid1 && (lane_id & 3) == 0) {
            m_smem[row1] = m_new1;
            l_smem[row1] = l_smem[row1] * rescale1 + psum1;
        }

        // Load V tile (KV_smem is safe to overwrite now — K no longer needed)
        {
            const uint4* V_g = reinterpret_cast<const uint4*>(V_bh + static_cast<size_t>(k_start) * D_HEAD);
            uint4* V_s = reinterpret_cast<uint4*>(KV_smem);
            const int total_vecs = BK * D_HEAD / 8;
            for (int i = threadIdx.x; i < total_vecs; i += blockDim.x) {
                const int row = i / (D_HEAD / 8);
                V_s[i] = (k_start + row < S) ? V_g[i] : make_uint4(0, 0, 0, 0);
            }
        }

        __syncthreads();

        // Compute O += P @ V using wmma (accumulate into register fragments)
        #pragma unroll
        for (int n_tile = 0; n_tile < NUM_N_TILES_O; n_tile++) {
            const int n_col_start = n_tile * WMMA_N;

            #pragma unroll
            for (int k_tile = 0; k_tile < NUM_K_TILES_PV; k_tile++) {
                const int k_offset = k_tile * WMMA_K;

                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;

                wmma::load_matrix_sync(a_frag, P_smem + m_row_start * BK + k_offset, BK);
                wmma::load_matrix_sync(b_frag, KV_smem + k_offset * D_HEAD + n_col_start, D_HEAD);

                wmma::mma_sync(o_frag[n_tile], a_frag, b_frag, o_frag[n_tile]);
            }
        }

        __syncthreads();
    }

    // Final normalization
    const float l0 = valid0 ? l_smem[row0] : 0.0f;
    const float l1 = valid1 ? l_smem[row1] : 0.0f;
    const float m0 = valid0 ? m_smem[row0] : -INFINITY;
    const float m1 = valid1 ? m_smem[row1] : -INFINITY;

    const float inv_l0 = (l0 > 0.0f) ? (1.0f / l0) : 0.0f;
    const float inv_l1 = (l1 > 0.0f) ? (1.0f / l1) : 0.0f;

    #pragma unroll
    for (int n = 0; n < NUM_N_TILES_O; n++) {
        o_frag[n].x[0] *= inv_l0;
        o_frag[n].x[1] *= inv_l0;
        o_frag[n].x[2] *= inv_l1;
        o_frag[n].x[3] *= inv_l1;
        o_frag[n].x[4] *= inv_l0;
        o_frag[n].x[5] *= inv_l0;
        o_frag[n].x[6] *= inv_l1;
        o_frag[n].x[7] *= inv_l1;
    }

    // Write LSE (natural log)
    if (valid0 && l0 > 0.0f) {
        LSE_bh[q_start + row0] = m0 + logf(l0);
    }
    if (valid1 && l1 > 0.0f) {
        LSE_bh[q_start + row1] = m1 + logf(l1);
    }

    // Store O fragments to bf16 shared memory (reuse Q_smem since Q no longer needed)
    __nv_bfloat16* O_bf16 = Q_smem;
    const int col_base = (lane_id % 4) * 2;

    #pragma unroll
    for (int n = 0; n < NUM_N_TILES_O; n++) {
        const int n_col = n * WMMA_N;
        // x[0,1]: row=lane/4,     col=col_base,   col_base+1
        // x[2,3]: row=lane/4+8,   col=col_base,   col_base+1
        // x[4,5]: row=lane/4,     col=col_base+8, col_base+9
        // x[6,7]: row=lane/4+8,   col=col_base+8, col_base+9
        if (valid0) {
            O_bf16[(m_row_start + lane_id / 4)     * D_HEAD + n_col + col_base]     = __float2bfloat16(o_frag[n].x[0]);
            O_bf16[(m_row_start + lane_id / 4)     * D_HEAD + n_col + col_base + 1] = __float2bfloat16(o_frag[n].x[1]);
            O_bf16[(m_row_start + lane_id / 4)     * D_HEAD + n_col + col_base + 8] = __float2bfloat16(o_frag[n].x[4]);
            O_bf16[(m_row_start + lane_id / 4)     * D_HEAD + n_col + col_base + 9] = __float2bfloat16(o_frag[n].x[5]);
        }
        if (valid1) {
            O_bf16[(m_row_start + lane_id / 4 + 8) * D_HEAD + n_col + col_base]     = __float2bfloat16(o_frag[n].x[2]);
            O_bf16[(m_row_start + lane_id / 4 + 8) * D_HEAD + n_col + col_base + 1] = __float2bfloat16(o_frag[n].x[3]);
            O_bf16[(m_row_start + lane_id / 4 + 8) * D_HEAD + n_col + col_base + 8] = __float2bfloat16(o_frag[n].x[6]);
            O_bf16[(m_row_start + lane_id / 4 + 8) * D_HEAD + n_col + col_base + 9] = __float2bfloat16(o_frag[n].x[7]);
        }
    }

    __syncthreads();

    // Vectorized store O to global memory
    {
        uint4* O_s = reinterpret_cast<uint4*>(O_bf16);
        uint4* O_g = reinterpret_cast<uint4*>(O_bh + static_cast<size_t>(q_start) * D_HEAD);
        const int total_vecs = BM * D_HEAD / 8;
        for (int i = threadIdx.x; i < total_vecs; i += blockDim.x) {
            const int row = i / (D_HEAD / 8);
            if (q_start + row < S) {
                O_g[i] = O_s[i];
            }
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    const int B = static_cast<int>(Q.size(0));
    const int H = static_cast<int>(Q.size(1));
    const int S = static_cast<int>(Q.size(2));
    const int D = static_cast<int>(Q.size(3));
    (void)D; // D=128 is hardcoded in the kernel

    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());

    const int num_q_blocks = (S + BM - 1) / BM;
    dim3 grid(num_q_blocks, B * H);
    dim3 block(NUM_THREADS);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(attention_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));

    attention_kernel<<<grid, block, SMEM_SIZE, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data, S, B, H);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_cuda::run);

}  // namespace tvm_ffi_mha_cuda