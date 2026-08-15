#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cmath>
#include <cstdint>
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

using namespace nvcuda;

namespace mha_kernel_ns {

constexpr int D = 128;
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int Q_STRIDE = 128;   // Fixed: matches int4 vectorized load stride
constexpr int S_STRIDE = 65;    // BN + 1 padding (float elements) for bank conflict avoidance
constexpr int P_STRIDE = 130;   // 2 * S_STRIDE (bf16 elements, same byte stride)
constexpr float INV_SQRT_D = 0.0883883476483184f;

__device__ __forceinline__ void cp_async_16B(void* smem, const void* gmem) {
    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n");
}

__global__ __launch_bounds__(128, 2)
void mha_wmma_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S) {

    int b = blockIdx.x;
    int h = blockIdx.y;
    int q_block = blockIdx.z;
    int q_start = q_block * BM;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;

    size_t bh_offset = ((size_t)b * H + h) * S * D;
    const __nv_bfloat16* Q_bh = Q + bh_offset;
    const __nv_bfloat16* K_bh = K + bh_offset;
    const __nv_bfloat16* V_bh = V + bh_offset;
    __nv_bfloat16* O_bh = O + bh_offset;
    float* LSE_bh = LSE + ((size_t)b * H + h) * S;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* smem_q = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* smem_k0 = smem_q + BM * Q_STRIDE;
    __nv_bfloat16* smem_k1 = smem_k0 + BN * D;
    __nv_bfloat16* smem_v0 = smem_k1 + BN * D;
    __nv_bfloat16* smem_v1 = smem_v0 + BN * D;
    float* smem_s = reinterpret_cast<float*>(smem_v1 + BN * D);
    __nv_bfloat16* smem_p = reinterpret_cast<__nv_bfloat16*>(smem_s);
    float* smem_rescale = smem_s + BM * S_STRIDE;
    float* smem_rowsum = smem_rescale + BM;

    // Load Q tile using int4 vectorized stores (stride = D = 128, contiguous)
    {
        int total = BM * D / 8;
        const int4* q_src = reinterpret_cast<const int4*>(Q_bh + (size_t)q_start * D);
        int4* q_dst = reinterpret_cast<int4*>(smem_q);
        for (int i = tid; i < total; i += 128) {
            int row = i / (D / 8);
            if (q_start + row < S) {
                q_dst[i] = q_src[i];
            } else {
                q_dst[i] = make_int4(0, 0, 0, 0);
            }
        }
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag_qk;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_frag_pv;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_frag[8];

    #pragma unroll
    for (int j = 0; j < 8; j++) {
        wmma::fill_fragment(o_frag[j], 0.0f);
    }

    float rowmax = -INFINITY;
    float rowsum = 0.0f;

    int num_kv_tiles = (S + BN - 1) / BN;
    int buf = 0;

    // Load first K, V tile using cp.async
    {
        __nv_bfloat16* k_buf = smem_k0;
        __nv_bfloat16* v_buf = smem_v0;
        int total = BN * D / 8;
        for (int i = tid; i < total; i += 128) {
            int row = i / (D / 8);
            int col_chunk = i % (D / 8);
            if (row < S) {
                cp_async_16B(k_buf + i * 8, K_bh + (size_t)row * D + col_chunk * 8);
                cp_async_16B(v_buf + i * 8, V_bh + (size_t)row * D + col_chunk * 8);
            } else {
                int4* k_dst = reinterpret_cast<int4*>(k_buf);
                int4* v_dst = reinterpret_cast<int4*>(v_buf);
                k_dst[i] = make_int4(0, 0, 0, 0);
                v_dst[i] = make_int4(0, 0, 0, 0);
            }
        }
        cp_async_commit();
        cp_async_wait_all();
    }
    __syncthreads();

    for (int kv_idx = 0; kv_idx < num_kv_tiles; kv_idx++) {
        int kv = kv_idx * BN;
        int actual_bn = min(BN, S - kv);
        bool has_next = (kv_idx + 1 < num_kv_tiles);
        int next_buf = 1 - buf;

        // Prefetch next K, V tile using cp.async
        if (has_next) {
            __nv_bfloat16* k_next = (next_buf == 0) ? smem_k0 : smem_k1;
            __nv_bfloat16* v_next = (next_buf == 0) ? smem_v0 : smem_v1;
            int next_kv = (kv_idx + 1) * BN;
            int total = BN * D / 8;
            for (int i = tid; i < total; i += 128) {
                int row = i / (D / 8);
                int col_chunk = i % (D / 8);
                int kv_row = next_kv + row;
                if (kv_row < S) {
                    cp_async_16B(k_next + i * 8, K_bh + (size_t)kv_row * D + col_chunk * 8);
                    cp_async_16B(v_next + i * 8, V_bh + (size_t)kv_row * D + col_chunk * 8);
                } else {
                    int4* k_dst = reinterpret_cast<int4*>(k_next);
                    int4* v_dst = reinterpret_cast<int4*>(v_next);
                    k_dst[i] = make_int4(0, 0, 0, 0);
                    v_dst[i] = make_int4(0, 0, 0, 0);
                }
            }
            cp_async_commit();
        }

        // Current K, V buffers
        __nv_bfloat16* smem_k = (buf == 0) ? smem_k0 : smem_k1;
        __nv_bfloat16* smem_v = (buf == 0) ? smem_v0 : smem_v1;

        // === QK^T using WMMA ===
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            wmma::fill_fragment(s_frag[n], 0.0f);
        }

        #pragma unroll
        for (int k_t = 0; k_t < D / 16; k_t++) {
            wmma::load_matrix_sync(a_frag,
                smem_q + warp_id * 16 * Q_STRIDE + k_t * 16, Q_STRIDE);

            #pragma unroll
            for (int n_t = 0; n_t < 4; n_t++) {
                wmma::load_matrix_sync(b_frag_qk,
                    smem_k + n_t * 16 * D + k_t * 16, D);
                wmma::mma_sync(s_frag[n_t], a_frag, b_frag_qk, s_frag[n_t]);
            }
        }

        // Scale by 1/sqrt(D)
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                s_frag[n].x[i] *= INV_SQRT_D;
            }
        }

        // Store S to shared memory
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            wmma::store_matrix_sync(
                smem_s + warp_id * 16 * S_STRIDE + n * 16,
                s_frag[n], S_STRIDE, wmma::mem_row_major);
        }
        __syncthreads();

        // === Online Softmax ===
        if (tid < BM) {
            int row = tid;
            float* s_row = smem_s + row * S_STRIDE;
            __nv_bfloat16* p_row = smem_p + row * P_STRIDE;

            float old_max = rowmax;
            float new_max = old_max;
            #pragma unroll
            for (int j = 0; j < BN; j++) {
                float val = (j < actual_bn) ? s_row[j] : -INFINITY;
                new_max = fmaxf(new_max, val);
            }

            float rescale = expf(old_max - new_max);
            smem_rescale[row] = rescale;
            rowsum *= rescale;

            #pragma unroll
            for (int j = 0; j < BN; j++) {
                float val = (j < actual_bn) ? s_row[j] : -INFINITY;
                float p = expf(val - new_max);
                p_row[j] = __float2bfloat16(p);
                rowsum += p;
            }

            rowmax = new_max;
            smem_rowsum[row] = rowsum;
        }
        __syncthreads();

        // === Rescale O fragments ===
        {
            int row1 = warp_id * 16 + lane / 4;
            int row2 = warp_id * 16 + lane / 4 + 8;
            float r1 = smem_rescale[row1];
            float r2 = smem_rescale[row2];
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                o_frag[j].x[0] *= r1;
                o_frag[j].x[1] *= r1;
                o_frag[j].x[4] *= r1;
                o_frag[j].x[5] *= r1;
                o_frag[j].x[2] *= r2;
                o_frag[j].x[3] *= r2;
                o_frag[j].x[6] *= r2;
                o_frag[j].x[7] *= r2;
            }
        }

        // === PV using WMMA ===
        #pragma unroll
        for (int k_t = 0; k_t < BN / 16; k_t++) {
            wmma::load_matrix_sync(a_frag,
                smem_p + warp_id * 16 * P_STRIDE + k_t * 16, P_STRIDE);

            #pragma unroll
            for (int n_t = 0; n_t < 8; n_t++) {
                wmma::load_matrix_sync(b_frag_pv,
                    smem_v + k_t * 16 * D + n_t * 16, D);
                wmma::mma_sync(o_frag[n_t], a_frag, b_frag_pv, o_frag[n_t]);
            }
        }

        __syncthreads();

        // Wait for next K, V
        if (has_next) {
            cp_async_wait_all();
            __syncthreads();
            buf = next_buf;
        }
    }

    // === Store O to shared memory (reuse K/V buffers for fp32 staging) ===
    float* smem_o1 = reinterpret_cast<float*>(smem_k0);
    float* smem_o2 = reinterpret_cast<float*>(smem_k1);

    #pragma unroll
    for (int j = 0; j < 8; j++) {
        float* target = (warp_id < 2) ? smem_o1 : smem_o2;
        int row_offset = (warp_id % 2) * 16;
        wmma::store_matrix_sync(target + row_offset * 128 + j * 16,
            o_frag[j], 128, wmma::mem_row_major);
    }
    __syncthreads();

    // Vectorized store to global (int4 = 8 bf16 per write)
    {
        int total = BM * D / 8;
        int4* o_dst = reinterpret_cast<int4*>(O_bh + (size_t)q_start * D);
        for (int i = tid; i < total; i += 128) {
            int row = i / (D / 8);
            int col_start = (i % (D / 8)) * 8;
            if (q_start + row < S) {
                float* src = (row < 32) ? smem_o1 : smem_o2;
                int local_row = row % 32;
                float rs = smem_rowsum[row];
                float inv_rs = (rs > 0.0f) ? (1.0f / rs) : 0.0f;
                __nv_bfloat16 tmp[8];
                #pragma unroll
                for (int d = 0; d < 8; d++) {
                    tmp[d] = __float2bfloat16(src[local_row * 128 + col_start + d] * inv_rs);
                }
                o_dst[i] = *reinterpret_cast<int4*>(tmp);
            }
        }
    }

    // Store LSE
    if (tid < BM && q_start + tid < S) {
        LSE_bh[q_start + tid] = rowmax + logf(rowsum);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = static_cast<int>(Q.size(0));
    int H = static_cast<int>(Q.size(1));
    int S = static_cast<int>(Q.size(2));

    if (S == 0) return;

    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());

    dim3 grid(B, H, (S + BM - 1) / BM);
    dim3 block(128);

    size_t smem_size = (size_t)BM * Q_STRIDE * sizeof(__nv_bfloat16)   // Q
                     + (size_t)2 * BN * D * sizeof(__nv_bfloat16)       // K double buffer
                     + (size_t)2 * BN * D * sizeof(__nv_bfloat16)       // V double buffer
                     + (size_t)BM * S_STRIDE * sizeof(float)             // S/P
                     + (size_t)BM * sizeof(float)                         // rescale
                     + (size_t)BM * sizeof(float);                        // rowsum

    CUDA_CHECK(cudaFuncSetAttribute(mha_wmma_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    mha_wmma_kernel<<<grid, block, smem_size, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data, B, H, S);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel_ns::run);

}  // namespace mha_kernel_ns