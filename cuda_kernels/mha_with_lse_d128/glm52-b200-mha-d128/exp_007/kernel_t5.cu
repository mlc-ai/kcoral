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

constexpr int Br = 64;
constexpr int Bc = 64;
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

__global__ __launch_bounds__(THREADS, 3)
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

    // O accumulator in registers: 8 N-tiles (D/16), 1 M-tile per warp
    fragment<accumulator, 16, 16, 16, float> o_frag[8];
    #pragma unroll
    for (int n = 0; n < 8; n++) fill_fragment(o_frag[n], 0.0f);

    float m_ref = -INFINITY;
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
        fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a_frag;
        fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> b_frag;
        fragment<accumulator, 16, 16, 16, float> s_frag[4];

        #pragma unroll
        for (int n = 0; n < 4; n++) fill_fragment(s_frag[n], 0.0f);

        #pragma unroll
        for (int k = 0; k < D / 16; k++) {
            load_matrix_sync(a_frag, &Q_smem[warp_id * 16 * D + k * 16], D);
            #pragma unroll
            for (int n = 0; n < 4; n++) {
                load_matrix_sync(b_frag, &K_smem[n * 16 * D + k * 16], D);
                mma_sync(s_frag[n], a_frag, b_frag, s_frag[n]);
            }
        }

        #pragma unroll
        for (int n = 0; n < 4; n++) {
            #pragma unroll
            for (int i = 0; i < s_frag[n].num_elements; i++)
                s_frag[n].x[i] *= scale;
            store_matrix_sync(&S_smem[warp_id * 16 * Bc + n * 16], s_frag[n], Bc, mem_row_major);
        }
        __syncwarp();

        // Softmax: lanes 0-15 handle 16 rows per warp
        float alpha = 1.0f;
        if (lane_id < 16) {
            int row = warp_id * 16 + lane_id;
            int global_row = q_start + row;
            if (global_row < S) {
                float row_max = -INFINITY;
                for (int j = 0; j < Bc; j++) {
                    if (kv_start + j < S)
                        row_max = fmaxf(row_max, S_smem[row * Bc + j]);
                }

                float m_new = fmaxf(m_ref, row_max);
                alpha = __expf(m_ref - m_new);

                float row_sum = 0.0f;
                for (int j = 0; j < Bc; j++) {
                    float p = (kv_start + j >= S) ? 0.0f : __expf(S_smem[row * Bc + j] - m_new);
                    K_smem[row * Bc + j] = __float2bfloat16(p);
                    row_sum += p;
                }
                l_prev = l_prev * alpha + row_sum;
                m_ref = m_new;
            }
        }

        // Broadcast alpha for in-register rescaling of o_frag
        // wmma 16x16 fp32 accumulator layout (sm_80+):
        // frag.x[0,1,4,5] -> row (lane_id/4)*2, frag.x[2,3,6,7] -> row (lane_id/4)*2+1
        float alpha0 = __shfl_sync(0xFFFFFFFF, alpha, (lane_id / 4) * 2);
        float alpha1 = __shfl_sync(0xFFFFFFFF, alpha, (lane_id / 4) * 2 + 1);

        #pragma unroll
        for (int n = 0; n < 8; n++) {
            o_frag[n].x[0] *= alpha0;
            o_frag[n].x[1] *= alpha0;
            o_frag[n].x[4] *= alpha0;
            o_frag[n].x[5] *= alpha0;
            o_frag[n].x[2] *= alpha1;
            o_frag[n].x[3] *= alpha1;
            o_frag[n].x[6] *= alpha1;
            o_frag[n].x[7] *= alpha1;
        }
        __syncwarp();

        // O += P @ V (P stored in K_smem)
        #pragma unroll
        for (int k = 0; k < Bc / 16; k++) {
            fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> p_frag;
            load_matrix_sync(p_frag, &K_smem[warp_id * 16 * Bc + k * 16], Bc);

            #pragma unroll
            for (int n = 0; n < 8; n++) {
                fragment<matrix_b, 16, 16, 16, __nv_bfloat16, row_major> v_frag;
                load_matrix_sync(v_frag, &V_smem[k * 16 * D + n * 16], D);
                mma_sync(o_frag[n], p_frag, v_frag, o_frag[n]);
            }
        }

        __syncthreads();
    }

    // Store O to global in 2 passes (S_smem = 64*64*4 = 16KB, O = 64*128*4 = 32KB)
    // Pass 1: o_frag[0..3] -> S_smem -> global[0..63]
    #pragma unroll
    for (int n = 0; n < 4; n++)
        store_matrix_sync(&S_smem[warp_id * 16 * 64 + n * 16], o_frag[n], 64, mem_row_major);
    __syncwarp();

    if (lane_id < 16) {
        int row = warp_id * 16 + lane_id;
        int global_row = q_start + row;
        if (global_row < S) {
            float inv_l = 1.0f / l_prev;
            for (int d = 0; d < 64; d += 8) {
                __nv_bfloat162 o01, o23, o45, o67;
                o01.x = __float2bfloat16(S_smem[row * 64 + d] * inv_l);
                o01.y = __float2bfloat16(S_smem[row * 64 + d + 1] * inv_l);
                o23.x = __float2bfloat16(S_smem[row * 64 + d + 2] * inv_l);
                o23.y = __float2bfloat16(S_smem[row * 64 + d + 3] * inv_l);
                o45.x = __float2bfloat16(S_smem[row * 64 + d + 4] * inv_l);
                o45.y = __float2bfloat16(S_smem[row * 64 + d + 5] * inv_l);
                o67.x = __float2bfloat16(S_smem[row * 64 + d + 6] * inv_l);
                o67.y = __float2bfloat16(S_smem[row * 64 + d + 7] * inv_l);
                uint4 out;
                out.x = *reinterpret_cast<uint32_t*>(&o01);
                out.y = *reinterpret_cast<uint32_t*>(&o23);
                out.z = *reinterpret_cast<uint32_t*>(&o45);
                out.w = *reinterpret_cast<uint32_t*>(&o67);
                *reinterpret_cast<uint4*>(&O_base[(int64_t)global_row * D + d]) = out;
            }
            LSE_base[global_row] = m_ref + logf(l_prev);
        }
    }
    __syncwarp();

    // Pass 2: o_frag[4..7] -> S_smem -> global[64..127]
    #pragma unroll
    for (int n = 4; n < 8; n++)
        store_matrix_sync(&S_smem[warp_id * 16 * 64 + (n - 4) * 16], o_frag[n], 64, mem_row_major);
    __syncwarp();

    if (lane_id < 16) {
        int row = warp_id * 16 + lane_id;
        int global_row = q_start + row;
        if (global_row < S) {
            float inv_l = 1.0f / l_prev;
            for (int d = 0; d < 64; d += 8) {
                __nv_bfloat162 o01, o23, o45, o67;
                o01.x = __float2bfloat16(S_smem[row * 64 + d] * inv_l);
                o01.y = __float2bfloat16(S_smem[row * 64 + d + 1] * inv_l);
                o23.x = __float2bfloat16(S_smem[row * 64 + d + 2] * inv_l);
                o23.y = __float2bfloat16(S_smem[row * 64 + d + 3] * inv_l);
                o45.x = __float2bfloat16(S_smem[row * 64 + d + 4] * inv_l);
                o45.y = __float2bfloat16(S_smem[row * 64 + d + 5] * inv_l);
                o67.x = __float2bfloat16(S_smem[row * 64 + d + 6] * inv_l);
                o67.y = __float2bfloat16(S_smem[row * 64 + d + 7] * inv_l);
                uint4 out;
                out.x = *reinterpret_cast<uint32_t*>(&o01);
                out.y = *reinterpret_cast<uint32_t*>(&o23);
                out.z = *reinterpret_cast<uint32_t*>(&o45);
                out.w = *reinterpret_cast<uint32_t*>(&o67);
                *reinterpret_cast<uint4*>(&O_base[(int64_t)global_row * D + 64 + d]) = out;
            }
        }
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

    int smem_size = Br * D * 2 + 2 * Bc * D * 2 + Br * Bc * 4;

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