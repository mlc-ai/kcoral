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
constexpr int BK = 128;
constexpr int NUM_THREADS = 128;
constexpr int NUM_WARPS = 4;
constexpr int ROWS_PER_WARP = BQ / NUM_WARPS;

// Padded strides to avoid bank conflicts (must be multiple of 8 for bf16 wmma alignment)
constexpr int D_STRIDE = D + 8;     // 136
constexpr int BK_STRIDE = BK + 8;   // 136

__global__ void mha_kernel(
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
    int q_end = min(q_start + BQ, S);
    int q_len = q_end - q_start;

    extern __shared__ char smem[];
    __nv_bfloat16* Q_smem = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* K_smem = Q_smem + BQ * D_STRIDE;
    __nv_bfloat16* V_smem = K_smem + BK * D_STRIDE;
    float* S_smem = reinterpret_cast<float*>(V_smem + BK * D_STRIDE);
    float* O_smem = S_smem + BQ * BK_STRIDE;
    __nv_bfloat16* P_smem = reinterpret_cast<__nv_bfloat16*>(O_smem + BQ * D_STRIDE);
    float* m_smem = reinterpret_cast<float*>(P_smem + BQ * BK_STRIDE);
    float* l_smem = m_smem + BQ;
    float* rescale_smem = l_smem + BQ;

    const float scale = 0.08838834764831845f;

    // Load Q with padded stride
    {
        const uint4* Q_vec = reinterpret_cast<const uint4*>(Q_base);
        constexpr int vpr = D / 8;
        for (int idx = tid; idx < BQ * vpr; idx += NUM_THREADS) {
            int qi = idx / vpr;
            int vi = idx % vpr;
            uint4 val = (qi < q_len) ? Q_vec[(q_start + qi) * vpr + vi] : make_uint4(0, 0, 0, 0);
            uint4* dst = reinterpret_cast<uint4*>(Q_smem + qi * D_STRIDE + vi * 8);
            *dst = val;
        }
    }

    // Init O, m, l
    for (int i = tid; i < BQ; i += NUM_THREADS) {
        m_smem[i] = -INFINITY;
        l_smem[i] = 0.0f;
    }
    for (int idx = tid; idx < BQ * D; idx += NUM_THREADS) {
        int qi = idx / D;
        int di = idx % D;
        O_smem[qi * D_STRIDE + di] = 0.0f;
    }

    __syncthreads();

    using namespace nvcuda::wmma;
    fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> b_frag_qk;
    fragment<matrix_b, 16, 16, 16, __nv_bfloat16, row_major> b_frag_pv;
    fragment<accumulator, 16, 16, 16, float> c_frag;

    for (int kv_start = 0; kv_start < S; kv_start += BK) {
        int kv_end = min(kv_start + BK, S);
        int kv_len = kv_end - kv_start;

        // Load K with padded stride
        {
            const uint4* K_vec = reinterpret_cast<const uint4*>(K_base);
            constexpr int vpr = D / 8;
            for (int idx = tid; idx < BK * vpr; idx += NUM_THREADS) {
                int ki = idx / vpr;
                int vi = idx % vpr;
                uint4 val = (ki < kv_len) ? K_vec[(kv_start + ki) * vpr + vi] : make_uint4(0, 0, 0, 0);
                uint4* dst = reinterpret_cast<uint4*>(K_smem + ki * D_STRIDE + vi * 8);
                *dst = val;
            }
        }

        // Load V with padded stride
        {
            const uint4* V_vec = reinterpret_cast<const uint4*>(V_base);
            constexpr int vpr = D / 8;
            for (int idx = tid; idx < BK * vpr; idx += NUM_THREADS) {
                int ki = idx / vpr;
                int vi = idx % vpr;
                uint4 val = (ki < kv_len) ? V_vec[(kv_start + ki) * vpr + vi] : make_uint4(0, 0, 0, 0);
                uint4* dst = reinterpret_cast<uint4*>(V_smem + ki * D_STRIDE + vi * 8);
                *dst = val;
            }
        }

        __syncthreads();

        // S = Q @ K^T * scale
        #pragma unroll
        for (int n_tile = 0; n_tile < BK / 16; n_tile++) {
            fill_fragment(c_frag, 0.0f);
            #pragma unroll
            for (int k_tile = 0; k_tile < D / 16; k_tile++) {
                load_matrix_sync(a_frag, Q_smem + warp_id * 16 * D_STRIDE + k_tile * 16, D_STRIDE);
                load_matrix_sync(b_frag_qk, K_smem + k_tile * 16 + n_tile * 16 * D_STRIDE, D_STRIDE);
                mma_sync(c_frag, a_frag, b_frag_qk, c_frag);
            }
            #pragma unroll
            for (int i = 0; i < c_frag.num_elements; i++) {
                c_frag.x[i] *= scale;
            }
            store_matrix_sync(S_smem + warp_id * 16 * BK_STRIDE + n_tile * 16, c_frag, BK_STRIDE, mem_row_major);
        }

        __syncthreads();

        // Parallel online softmax: 2 threads per row, warp shuffle reduction
        int row_base = warp_id * ROWS_PER_WARP;
        int row_in_warp = lane_id / 2;
        int col_half = lane_id % 2;
        int row = row_base + row_in_warp;
        int cols_per_thread = BK / 2;
        int col_start = col_half * cols_per_thread;

        if (row < q_len) {
            float local_max = -INFINITY;
            for (int j = 0; j < cols_per_thread; j++) {
                int col = col_start + j;
                float val = (col < kv_len) ? S_smem[row * BK_STRIDE + col] : -INFINITY;
                local_max = fmaxf(local_max, val);
            }

            float other_max = __shfl_xor_sync(0xFFFFFFFF, local_max, 1);
            float block_max = fmaxf(local_max, other_max);

            float m_old = m_smem[row];
            float m_new = fmaxf(m_old, block_max);
            float rescale = (m_old == -INFINITY) ? 0.0f : __expf(m_old - m_new);

            float local_sum = 0.0f;
            for (int j = 0; j < cols_per_thread; j++) {
                int col = col_start + j;
                if (col < kv_len) {
                    float p = __expf(S_smem[row * BK_STRIDE + col] - m_new);
                    local_sum += p;
                    P_smem[row * BK_STRIDE + col] = __float2bfloat16(p);
                } else {
                    P_smem[row * BK_STRIDE + col] = __float2bfloat16(0.0f);
                }
            }

            float other_sum = __shfl_xor_sync(0xFFFFFFFF, local_sum, 1);
            float block_sum = local_sum + other_sum;

            rescale_smem[row] = rescale;
            l_smem[row] = l_smem[row] * rescale + block_sum;
            m_smem[row] = m_new;
        }

        __syncthreads();

        // Rescale O
        for (int idx = tid; idx < BQ * D; idx += NUM_THREADS) {
            int qi = idx / D;
            int di = idx % D;
            if (qi >= q_len) continue;
            O_smem[qi * D_STRIDE + di] *= rescale_smem[qi];
        }

        __syncthreads();

        // O += P @ V
        #pragma unroll
        for (int n_tile = 0; n_tile < D / 16; n_tile++) {
            load_matrix_sync(c_frag, O_smem + warp_id * 16 * D_STRIDE + n_tile * 16, D_STRIDE, mem_row_major);
            #pragma unroll
            for (int k_tile = 0; k_tile < BK / 16; k_tile++) {
                load_matrix_sync(a_frag, P_smem + warp_id * 16 * BK_STRIDE + k_tile * 16, BK_STRIDE);
                load_matrix_sync(b_frag_pv, V_smem + k_tile * 16 * D_STRIDE + n_tile * 16, D_STRIDE);
                mma_sync(c_frag, a_frag, b_frag_pv, c_frag);
            }
            store_matrix_sync(O_smem + warp_id * 16 * D_STRIDE + n_tile * 16, c_frag, D_STRIDE, mem_row_major);
        }

        __syncthreads();
    }

    // Write LSE
    if (tid < BQ && tid < q_len) {
        float l = l_smem[tid];
        float m = m_smem[tid];
        LSE_base[q_start + tid] = m + logf(l);
    }

    // Write O (normalize by l, convert to bf16, vectorized store)
    {
        constexpr int vpr = D / 8;
        uint4* O_out_vec = reinterpret_cast<uint4*>(O_base + q_start * D);
        for (int idx = tid; idx < BQ * vpr; idx += NUM_THREADS) {
            int qi = idx / vpr;
            int vi = idx % vpr;
            if (qi >= q_len) continue;
            float inv_l = 1.0f / l_smem[qi];
            float* o_row = O_smem + qi * D_STRIDE + vi * 8;
            __nv_bfloat16 o_bf16[8];
            #pragma unroll
            for (int d = 0; d < 8; d++) {
                o_bf16[d] = __float2bfloat16(o_row[d] * inv_l);
            }
            O_out_vec[qi * vpr + vi] = *reinterpret_cast<uint4*>(o_bf16);
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

    int smem_size = BQ * D_STRIDE * sizeof(__nv_bfloat16) +   // Q
                    BK * D_STRIDE * sizeof(__nv_bfloat16) +    // K
                    BK * D_STRIDE * sizeof(__nv_bfloat16) +    // V
                    BQ * BK_STRIDE * sizeof(float) +           // S
                    BQ * D_STRIDE * sizeof(float) +            // O
                    BQ * BK_STRIDE * sizeof(__nv_bfloat16) +   // P
                    BQ * sizeof(float) * 3;                    // m, l, rescale

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