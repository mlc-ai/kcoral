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
constexpr int Q_STRIDE = 128;   // contiguous, 16-byte aligned
constexpr int S_STRIDE = 64;    // Fixed: 64*4=256 bytes, multiple of 16
constexpr int P_STRIDE = 128;   // Fixed: 128*2=256 bytes, multiple of 16
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

__global__ __launch_bounds__(128, 1)
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

    size_t bh_offset = ((size_t)b * H + h) * S * D;
    const __nv_bfloat16* Q_bh = Q + bh_offset;
    const __nv_bfloat16* K_bh = K + bh_offset;
    const __nv_bfloat16* V_bh = V + bh_offset;
    __nv_bfloat16* O_bh = O + bh_offset;
    float* LSE_bh = LSE + ((size_t)b * H + h) * S;

    extern __shared__ char smem_raw[];
    // Layout: Q(16KB) | K(16KB, reused for PV) | V(16KB) | S/P(16KB) | O(32KB) | aux(256B)
    __nv_bfloat16* smem_q = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* smem_k = smem_q + BM * Q_STRIDE;
    __nv_bfloat16* smem_v = smem_k + BN * D;
    float* smem_s = reinterpret_cast<float*>(smem_v + BN * D);
    float* smem_o = smem_s + BM * S_STRIDE;
    float* smem_aux = smem_o + BM * D;

    // Aliases for reuse
    float* smem_pv = reinterpret_cast<float*>(smem_k);
    __nv_bfloat16* smem_p = reinterpret_cast<__nv_bfloat16*>(smem_s);

    // Initialize smem_o to 0
    {
        int4* o_dst_i4 = reinterpret_cast<int4*>(smem_o);
        int total = BM * D / 4;
        int4 zero = make_int4(0, 0, 0, 0);
        for (int i = tid; i < total; i += 128) {
            o_dst_i4[i] = zero;
        }
    }

    // Load Q tile
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

    float rowmax = -INFINITY;
    float rowsum = 0.0f;

    int num_kv_tiles = (S + BN - 1) / BN;

    for (int kv_idx = 0; kv_idx < num_kv_tiles; kv_idx++) {
        int kv = kv_idx * BN;
        int actual_bn = min(BN, S - kv);

        // Load K, V tiles
        {
            int total = BN * D / 8;
            for (int i = tid; i < total; i += 128) {
                int row = i / (D / 8);
                int col_chunk = i % (D / 8);
                int kv_row = kv + row;
                if (kv_row < S) {
                    cp_async_16B(smem_k + i * 8, K_bh + (size_t)kv_row * D + col_chunk * 8);
                    cp_async_16B(smem_v + i * 8, V_bh + (size_t)kv_row * D + col_chunk * 8);
                } else {
                    int4* k_dst = reinterpret_cast<int4*>(smem_k);
                    int4* v_dst = reinterpret_cast<int4*>(smem_v);
                    k_dst[i] = make_int4(0, 0, 0, 0);
                    v_dst[i] = make_int4(0, 0, 0, 0);
                }
            }
            cp_async_commit();
            cp_async_wait_all();
        }
        __syncthreads();

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
            for (int j = 0; j < BN; j++) {
                float val = (j < actual_bn) ? s_row[j] : -INFINITY;
                new_max = fmaxf(new_max, val);
            }

            float rescale = expf(old_max - new_max);
            smem_aux[row] = rescale;
            rowsum *= rescale;

            for (int j = 0; j < BN; j++) {
                float val = (j < actual_bn) ? s_row[j] : -INFINITY;
                float p = expf(val - new_max);
                p_row[j] = __float2bfloat16(p);
                rowsum += p;
            }

            rowmax = new_max;
        }
        __syncthreads();

        // === PV using WMMA ===
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            wmma::fill_fragment(o_frag[j], 0.0f);
        }

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

        // === Store PV to shared memory and accumulate into O with rescale ===
        // First half: columns 0-63
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::store_matrix_sync(
                smem_pv + warp_id * 16 * 64 + j * 16,
                o_frag[j], 64, wmma::mem_row_major);
        }
        __syncthreads();

        if (tid < BM) {
            float r = smem_aux[tid];
            float* o_row = smem_o + tid * D;
            float* pv_row = smem_pv + tid * 64;
            #pragma unroll
            for (int d = 0; d < 64; d++) {
                o_row[d] = o_row[d] * r + pv_row[d];
            }
        }
        __syncthreads();

        // Second half: columns 64-127
        #pragma unroll
        for (int j = 4; j < 8; j++) {
            wmma::store_matrix_sync(
                smem_pv + warp_id * 16 * 64 + (j - 4) * 16,
                o_frag[j], 64, wmma::mem_row_major);
        }
        __syncthreads();

        if (tid < BM) {
            float r = smem_aux[tid];
            float* o_row = smem_o + tid * D;
            float* pv_row = smem_pv + tid * 64;
            #pragma unroll
            for (int d = 0; d < 64; d++) {
                o_row[64 + d] = o_row[64 + d] * r + pv_row[d];
            }
        }
        __syncthreads();
    }

    // === Store rowsum for normalization ===
    if (tid < BM) {
        smem_aux[tid] = rowsum;
    }
    __syncthreads();

    // === Normalize and store O to global ===
    {
        int total = BM * D / 8;
        int4* o_dst = reinterpret_cast<int4*>(O_bh + (size_t)q_start * D);
        for (int i = tid; i < total; i += 128) {
            int row = i / (D / 8);
            int col_start = (i % (D / 8)) * 8;
            if (q_start + row < S) {
                float rs = smem_aux[row];
                float inv_rs = (rs > 0.0f) ? (1.0f / rs) : 0.0f;
                __nv_bfloat16 tmp[8];
                #pragma unroll
                for (int d = 0; d < 8; d++) {
                    tmp[d] = __float2bfloat16(smem_o[row * D + col_start + d] * inv_rs);
                }
                o_dst[i] = *reinterpret_cast<int4*>(tmp);
            }
        }
    }

    // === Store LSE ===
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

    size_t smem_size = (size_t)BM * Q_STRIDE * sizeof(__nv_bfloat16)
                     + (size_t)BN * D * sizeof(__nv_bfloat16)
                     + (size_t)BN * D * sizeof(__nv_bfloat16)
                     + (size_t)BM * S_STRIDE * sizeof(float)
                     + (size_t)BM * D * sizeof(float)
                     + (size_t)BM * sizeof(float);

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