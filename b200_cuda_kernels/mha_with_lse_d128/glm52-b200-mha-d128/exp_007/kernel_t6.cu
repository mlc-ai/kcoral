#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda::wmma;

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

namespace attention_kernel {

constexpr int Br = 128;
constexpr int Bc = 128;
constexpr int D = 128;
constexpr int THREADS = 128;

__device__ __forceinline__ void cp_async_16B(void* smem_dst, const void* gmem_src) {
    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem_src));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::: "memory");
}

__global__ __launch_bounds__(THREADS, 1)
void attention_fwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S) {

    const float scale = 0.08838834764831840f;

    int bh = blockIdx.x;
    int q_block = blockIdx.y;
    int q_start = q_block * Br;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int b = bh / H;
    int h = bh % H;
    int64_t offset = (int64_t)b * H * S * D + (int64_t)h * S * D;

    const __nv_bfloat16* Q_base = Q + offset;
    const __nv_bfloat16* K_base = K + offset;
    const __nv_bfloat16* V_base = V + offset;
    __nv_bfloat16* O_base = O + offset;
    float* LSE_base = LSE + (int64_t)b * H * S + (int64_t)h * S;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* Q_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* K_smem = Q_smem + Br * D;
    __nv_bfloat16* V_smem = K_smem + Bc * D;
    float* S_smem = reinterpret_cast<float*>(V_smem + Bc * D);
    float* O_smem = S_smem + Br * Bc;

    // Load Q tile
    for (int i = tid; i < Br * D / 8; i += THREADS) {
        int row = i / (D / 8);
        int col8 = i % (D / 8);
        int q_idx = q_start + row;
        if (q_idx < S) {
            cp_async_16B(&Q_smem[row * D + col8 * 8], &Q_base[(int64_t)q_idx * D + col8 * 8]);
        } else {
            *reinterpret_cast<uint4*>(&Q_smem[row * D + col8 * 8]) = make_uint4(0, 0, 0, 0);
        }
    }
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    // Zero O_smem
    for (int i = tid; i < Br * D / 4; i += THREADS) {
        reinterpret_cast<float4*>(&O_smem[i * 4])[0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    __syncthreads();

    float m_prev = -INFINITY;
    float l_prev = 0.0f;

    int n_blocks = (S + Bc - 1) / Bc;

    for (int kv_block = 0; kv_block < n_blocks; kv_block++) {
        int kv_start = kv_block * Bc;

        // Load K, V tiles
        for (int i = tid; i < Bc * D / 8; i += THREADS) {
            int row = i / (D / 8);
            int col8 = i % (D / 8);
            int kv_idx = kv_start + row;
            if (kv_idx < S) {
                cp_async_16B(&K_smem[row * D + col8 * 8], &K_base[(int64_t)kv_idx * D + col8 * 8]);
                cp_async_16B(&V_smem[row * D + col8 * 8], &V_base[(int64_t)kv_idx * D + col8 * 8]);
            } else {
                *reinterpret_cast<uint4*>(&K_smem[row * D + col8 * 8]) = make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4*>(&V_smem[row * D + col8 * 8]) = make_uint4(0, 0, 0, 0);
            }
        }
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();

        // S = Q @ K^T * scale
        // Each warp handles 32 rows (2 M-tiles)
        {
            fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a_frag;
            fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> b_frag;
            fragment<accumulator, 16, 16, 16, float> s_frag[2][8];

            #pragma unroll
            for (int m = 0; m < 2; m++)
                #pragma unroll
                for (int n = 0; n < 8; n++) fill_fragment(s_frag[m][n], 0.0f);

            #pragma unroll
            for (int k = 0; k < D / 16; k++) {
                #pragma unroll
                for (int n = 0; n < 8; n++) {
                    load_matrix_sync(b_frag, &K_smem[n * 16 * D + k * 16], D);
                    #pragma unroll
                    for (int m = 0; m < 2; m++) {
                        load_matrix_sync(a_frag, &Q_smem[(warp_id * 32 + m * 16) * D + k * 16], D);
                        mma_sync(s_frag[m][n], a_frag, b_frag, s_frag[m][n]);
                    }
                }
            }

            #pragma unroll
            for (int m = 0; m < 2; m++) {
                #pragma unroll
                for (int n = 0; n < 8; n++) {
                    #pragma unroll
                    for (int i = 0; i < s_frag[m][n].num_elements; i++)
                        s_frag[m][n].x[i] *= scale;
                    store_matrix_sync(&S_smem[(warp_id * 32 + m * 16) * Bc + n * 16], s_frag[m][n], Bc, mem_row_major);
                }
            }
        }
        __syncthreads();

        // Softmax: each thread handles one row
        int row = tid;
        int global_row = q_start + row;
        if (global_row < S) {
            float row_max = -INFINITY;
            for (int j = 0; j < Bc; j++) {
                if (kv_start + j < S)
                    row_max = fmaxf(row_max, S_smem[row * Bc + j]);
            }

            float m_new = fmaxf(m_prev, row_max);
            float alpha = __expf(m_prev - m_new);

            float row_sum = 0.0f;
            for (int j = 0; j < Bc; j++) {
                float p = (kv_start + j >= S) ? 0.0f : __expf(S_smem[row * Bc + j] - m_new);
                K_smem[row * Bc + j] = __float2bfloat16(p);
                row_sum += p;
            }
            l_prev = l_prev * alpha + row_sum;
            m_prev = m_new;

            // Rescale O_smem for this row
            for (int d = 0; d < D; d += 4) {
                float4* o_ptr = reinterpret_cast<float4*>(&O_smem[row * D + d]);
                float4 v = o_ptr[0];
                v.x *= alpha; v.y *= alpha; v.z *= alpha; v.w *= alpha;
                o_ptr[0] = v;
            }
        }
        __syncthreads();

        // O += P @ V (P stored in K_smem)
        {
            fragment<accumulator, 16, 16, 16, float> o_frag[2][8];
            #pragma unroll
            for (int m = 0; m < 2; m++)
                #pragma unroll
                for (int n = 0; n < 8; n++)
                    load_matrix_sync(o_frag[m][n], &O_smem[(warp_id * 32 + m * 16) * D + n * 16], D, mem_row_major);

            #pragma unroll
            for (int k = 0; k < Bc / 16; k++) {
                #pragma unroll
                for (int n = 0; n < 8; n++) {
                    fragment<matrix_b, 16, 16, 16, __nv_bfloat16, row_major> v_frag;
                    load_matrix_sync(v_frag, &V_smem[k * 16 * D + n * 16], D);
                    #pragma unroll
                    for (int m = 0; m < 2; m++) {
                        fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> p_frag;
                        load_matrix_sync(p_frag, &K_smem[(warp_id * 32 + m * 16) * Bc + k * 16], Bc);
                        mma_sync(o_frag[m][n], p_frag, v_frag, o_frag[m][n]);
                    }
                }
            }

            #pragma unroll
            for (int m = 0; m < 2; m++)
                #pragma unroll
                for (int n = 0; n < 8; n++)
                    store_matrix_sync(&O_smem[(warp_id * 32 + m * 16) * D + n * 16], o_frag[m][n], D, mem_row_major);
        }

        __syncthreads();
    }

    // Final output
    int row = tid;
    int global_row = q_start + row;
    if (global_row < S) {
        float inv_l = 1.0f / l_prev;
        for (int d = 0; d < D; d += 8) {
            __nv_bfloat162 o01, o23, o45, o67;
            o01.x = __float2bfloat16(O_smem[row * D + d] * inv_l);
            o01.y = __float2bfloat16(O_smem[row * D + d + 1] * inv_l);
            o23.x = __float2bfloat16(O_smem[row * D + d + 2] * inv_l);
            o23.y = __float2bfloat16(O_smem[row * D + d + 3] * inv_l);
            o45.x = __float2bfloat16(O_smem[row * D + d + 4] * inv_l);
            o45.y = __float2bfloat16(O_smem[row * D + d + 5] * inv_l);
            o67.x = __float2bfloat16(O_smem[row * D + d + 6] * inv_l);
            o67.y = __float2bfloat16(O_smem[row * D + d + 7] * inv_l);
            uint4 out;
            out.x = *reinterpret_cast<uint32_t*>(&o01);
            out.y = *reinterpret_cast<uint32_t*>(&o23);
            out.z = *reinterpret_cast<uint32_t*>(&o45);
            out.w = *reinterpret_cast<uint32_t*>(&o67);
            *reinterpret_cast<uint4*>(&O_base[(int64_t)global_row * D + d]) = out;
        }
        LSE_base[global_row] = m_prev + logf(l_prev);
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
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());

    dim3 grid(B * H, (S + Br - 1) / Br);
    dim3 block(THREADS);

    int smem_size = Br * D * 2 + 2 * Bc * D * 2 + Br * Bc * 4 + Br * D * 4;

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(attention_fwd_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    attention_fwd_kernel<<<grid, block, smem_size, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data, B, H, S);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, attention_kernel::run);

}  // namespace attention_kernel