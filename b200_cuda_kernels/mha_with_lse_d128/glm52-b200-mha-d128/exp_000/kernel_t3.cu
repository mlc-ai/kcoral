#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>
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

namespace mha_lse {

constexpr int Br = 128;
constexpr int Bc = 64;
constexpr int D = 128;
constexpr int THREADS = 256;

constexpr int SMEM_SIZE = 
    Br * D * 2 +         // smem_q: 32 KB
    Bc * D * 2 +         // smem_k: 16 KB
    Br * Bc * 4 +        // smem_s: 32 KB
    Br * Bc * 2 +        // smem_p: 16 KB
    Br * D * 4 +         // smem_o: 64 KB
    Br * 32 * 4;         // smem_pv: 16 KB (Total: 176 KB)

__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n");
}

__global__ __launch_bounds__(THREADS, 1)
void mha_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S, float scale) {

    int tid = threadIdx.x;
    int warp_id = tid / 32;

    int n_q_blocks = (S + Br - 1) / Br;
    int grid_idx = blockIdx.x;
    int batch = grid_idx / (H * n_q_blocks);
    int rest = grid_idx % (H * n_q_blocks);
    int head = rest / n_q_blocks;
    int q_block = rest % n_q_blocks;
    int q_start = q_block * Br;

    extern __shared__ char smem[];
    __nv_bfloat16* smem_q = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* smem_k = smem_q + Br * D;
    float* smem_s = reinterpret_cast<float*>(smem_k + Bc * D);
    __nv_bfloat16* smem_p = reinterpret_cast<__nv_bfloat16*>(smem_s + Br * Bc);
    float* smem_o = reinterpret_cast<float*>(smem_p + Br * Bc);
    float* smem_pv = reinterpret_cast<float*>(smem_o + Br * D);

    // Load Q
    const __nv_bfloat16* Q_gptr = Q + ((batch * H + head) * S + q_start) * D;
    for (int i = tid; i < Br * D / 8; i += THREADS) {
        int row = (i * 8) / D;
        if (q_start + row < S) {
            cp_async_16(smem_q + i * 8, Q_gptr + i * 8);
        } else {
            *reinterpret_cast<int4*>(smem_q + i * 8) = make_int4(0, 0, 0, 0);
        }
    }
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    // Init O to 0
    for (int i = tid; i < Br * D; i += THREADS) {
        smem_o[i] = 0.0f;
    }
    __syncthreads();

    float m = -INFINITY;
    float l = 0.0f;

    for (int kv_start = 0; kv_start < S; kv_start += Bc) {
        int actual_Bc = min(Bc, S - kv_start);

        // Load K
        const __nv_bfloat16* K_gptr = K + ((batch * H + head) * S + kv_start) * D;
        for (int i = tid; i < Bc * D / 8; i += THREADS) {
            int row = (i * 8) / D;
            if (kv_start + row < S) {
                cp_async_16(smem_k + i * 8, K_gptr + i * 8);
            } else {
                *reinterpret_cast<int4*>(smem_k + i * 8) = make_int4(0, 0, 0, 0);
            }
        }
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();

        // QK^T
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag[4];
        #pragma unroll
        for (int n = 0; n < 4; n++) wmma::fill_fragment(s_frag[n], 0.0f);

        #pragma unroll
        for (int k = 0; k < 8; k++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, smem_q + warp_id * 16 * D + k * 16, D);
            #pragma unroll
            for (int n = 0; n < 4; n++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
                wmma::load_matrix_sync(b_frag, smem_k + n * 16 * D + k * 16, D);
                wmma::mma_sync(s_frag[n], a_frag, b_frag, s_frag[n]);
            }
        }

        #pragma unroll
        for (int n = 0; n < 4; n++) {
            #pragma unroll
            for (int i = 0; i < s_frag[n].num_elements; i++) {
                s_frag[n].x[i] *= scale;
            }
            wmma::store_matrix_sync(smem_s + warp_id * 16 * Bc + n * 16, s_frag[n], Bc, wmma::mem_row_major);
        }
        __syncthreads();

        // Online Softmax
        if (tid < 128) {
            float* s_row = smem_s + tid * Bc;
            __nv_bfloat16* p_row = smem_p + tid * Bc;
            float* o_row = smem_o + tid * D;

            float m_block = -INFINITY;
            #pragma unroll
            for (int j = 0; j < Bc; j++) {
                if (kv_start + j < S) {
                    m_block = fmaxf(m_block, s_row[j]);
                }
            }
            float m_new = fmaxf(m, m_block);
            float scale_o = expf(m - m_new);

            #pragma unroll
            for (int d = 0; d < D; d += 4) {
                float4 o4 = *reinterpret_cast<float4*>(o_row + d);
                o4.x *= scale_o; o4.y *= scale_o; o4.z *= scale_o; o4.w *= scale_o;
                *reinterpret_cast<float4*>(o_row + d) = o4;
            }

            float l_block = 0.0f;
            #pragma unroll
            for (int j = 0; j < Bc; j++) {
                if (kv_start + j < S) {
                    float p = expf(s_row[j] - m_new);
                    p_row[j] = __float2bfloat16(p);
                    l_block += p;
                } else {
                    p_row[j] = __float2bfloat16(0.0f);
                }
            }
            l = l * scale_o + l_block;
            m = m_new;
        }
        __syncthreads();

        // Load V
        const __nv_bfloat16* V_gptr = V + ((batch * H + head) * S + kv_start) * D;
        for (int i = tid; i < Bc * D / 8; i += THREADS) {
            int row = (i * 8) / D;
            if (kv_start + row < S) {
                cp_async_16(smem_k + i * 8, V_gptr + i * 8);
            } else {
                *reinterpret_cast<int4*>(smem_k + i * 8) = make_int4(0, 0, 0, 0);
            }
        }
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();

        // PV
        #pragma unroll
        for (int n_outer = 0; n_outer < 4; n_outer++) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[2];
            #pragma unroll
            for (int n = 0; n < 2; n++) wmma::fill_fragment(c_frag[n], 0.0f);

            #pragma unroll
            for (int k = 0; k < 4; k++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
                wmma::load_matrix_sync(a_frag, smem_p + warp_id * 16 * Bc + k * 16, Bc);
                #pragma unroll
                for (int n = 0; n < 2; n++) {
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_frag;
                    wmma::load_matrix_sync(b_frag, smem_k + k * 16 * D + (n_outer * 2 + n) * 16, D);
                    wmma::mma_sync(c_frag[n], a_frag, b_frag, c_frag[n]);
                }
            }

            #pragma unroll
            for (int n = 0; n < 2; n++) {
                wmma::store_matrix_sync(smem_pv + warp_id * 16 * 32 + n * 16, c_frag[n], 32, wmma::mem_row_major);
            }
            __syncthreads();

            if (tid < 128) {
                float* o_row = smem_o + tid * D + n_outer * 32;
                float* pv_row = smem_pv + tid * 32;
                #pragma unroll
                for (int d = 0; d < 32; d += 4) {
                    float4 o4 = *reinterpret_cast<float4*>(o_row + d);
                    float4 p4 = *reinterpret_cast<float4*>(pv_row + d);
                    o4.x += p4.x; o4.y += p4.y; o4.z += p4.z; o4.w += p4.w;
                    *reinterpret_cast<float4*>(o_row + d) = o4;
                }
            }
            __syncthreads();
        }
    }

    // Final output
    if (tid < 128 && q_start + tid < S) {
        float inv_l = 1.0f / l;
        float* o_row = smem_o + tid * D;
        __nv_bfloat16* O_gptr = O + ((batch * H + head) * S + q_start + tid) * D;

        #pragma unroll
        for (int d = 0; d < D; d += 8) {
            float4 o4_0 = *reinterpret_cast<float4*>(o_row + d);
            float4 o4_1 = *reinterpret_cast<float4*>(o_row + d + 4);

            __nv_bfloat162 p01 = __float22bfloat162_rn(make_float2(o4_0.x * inv_l, o4_0.y * inv_l));
            __nv_bfloat162 p23 = __float22bfloat162_rn(make_float2(o4_0.z * inv_l, o4_0.w * inv_l));
            __nv_bfloat162 p45 = __float22bfloat162_rn(make_float2(o4_1.x * inv_l, o4_1.y * inv_l));
            __nv_bfloat162 p67 = __float22bfloat162_rn(make_float2(o4_1.z * inv_l, o4_1.w * inv_l));

            float4 out;
            *reinterpret_cast<__nv_bfloat162*>(&out.x) = p01;
            *reinterpret_cast<__nv_bfloat162*>(&out.y) = p23;
            *reinterpret_cast<__nv_bfloat162*>(&out.z) = p45;
            *reinterpret_cast<__nv_bfloat162*>(&out.w) = p67;
            *reinterpret_cast<float4*>(O_gptr + d) = out;
        }

        LSE[(batch * H + head) * S + q_start + tid] = m + logf(l);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int B = 4;
    int H = 48;
    int D = 128;
    int64_t S = Q.size(2);
    float scale = 1.0f / sqrtf((float)D);
    int n_q_blocks = (static_cast<int>(S) + Br - 1) / Br;
    int grid = B * H * n_q_blocks;
    int block = THREADS;

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));

    mha_kernel<<<grid, block, SMEM_SIZE, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<__nv_bfloat16*>(O.data_ptr()),
        static_cast<float*>(LSE.data_ptr()),
        B, H, static_cast<int>(S), scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_lse::run);

}  // namespace mha_lse