#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cmath>
#include <cstdint>
#include <stdio.h>
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

using namespace nvcuda;

namespace mha_attention {

constexpr int D = 128;
constexpr int BR = 128;
constexpr int BC = 64;
constexpr int WARPS = 8;
constexpr int THREADS = WARPS * 32;
constexpr float LN2_RCP = 1.4426950408889634f;

__device__ __forceinline__ void cp_async_16B(uint32_t smem_addr, const void* gmem_ptr) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                 :: "r"(smem_addr), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

__global__ __launch_bounds__(THREADS)
void mha_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S) {

    int bh = blockIdx.x;
    int q_blk = blockIdx.y;
    int b = bh / H;
    int h = bh % H;
    int q_start = q_blk * BR;
    if (q_start >= S) return;

    const __nv_bfloat16* Q_bh = Q + ((size_t)(b * H + h) * S) * D;
    const __nv_bfloat16* K_bh = K + ((size_t)(b * H + h) * S) * D;
    const __nv_bfloat16* V_bh = V + ((size_t)(b * H + h) * S) * D;
    __nv_bfloat16* O_bh = O + ((size_t)(b * H + h) * S) * D;
    float* LSE_bh = LSE + (size_t)(b * H + h) * S;

    extern __shared__ char smem_buf[];
    __nv_bfloat16* smem_Q  = reinterpret_cast<__nv_bfloat16*>(smem_buf);
    __nv_bfloat16* smem_K0 = smem_Q + BR * D;
    __nv_bfloat16* smem_K1 = smem_K0 + BC * D;
    __nv_bfloat16* smem_V0 = smem_K1 + BC * D;
    __nv_bfloat16* smem_V1 = smem_V0 + BC * D;
    float* smem_S          = reinterpret_cast<float*>(smem_V1 + BC * D);
    __nv_bfloat16* smem_P  = reinterpret_cast<__nv_bfloat16*>(smem_S + BR * BC);
    float* smem_O          = reinterpret_cast<float*>(smem_P + BR * BC);

    __shared__ float smem_m[BR];
    __shared__ float smem_l[BR];
    __shared__ float smem_rescale[BR];

    int tid     = threadIdx.x;
    int warp_id = tid / 32;
    int lane    = tid % 32;
    int r       = lane % 16;
    int half    = lane / 16;
    int row     = warp_id * 16 + r;

    // Initialize O, m, l
    for (int i = tid; i < BR * D; i += THREADS) smem_O[i] = 0.0f;
    if (tid < BR) { smem_m[tid] = -INFINITY; smem_l[tid] = 0.0f; }
    __syncthreads();

    // Load Q tile (vectorized int4 = 8 bf16)
    {
        int4* smem_Q_i4 = reinterpret_cast<int4*>(smem_Q);
        const int4* Q_bh_i4 = reinterpret_cast<const int4*>(Q_bh);
        constexpr int i4_per_row = D / 8;
        constexpr int total = BR * i4_per_row;
        for (int i = tid; i < total; i += THREADS) {
            int qrow = i / i4_per_row;
            int qcol8 = i % i4_per_row;
            int q_idx = q_start + qrow;
            smem_Q_i4[i] = (q_idx < S) ? Q_bh_i4[q_idx * i4_per_row + qcol8]
                                       : make_int4(0,0,0,0);
        }
    }
    __syncthreads();

    const float scale = 1.0f / sqrtf((float)D);
    int n_kv = (S + BC - 1) / BC;

    // Load first K, V with cp.async
    {
        int valid_kv = min(BC, S);
        const int4* K_src = reinterpret_cast<const int4*>(K_bh);
        const int4* V_src = reinterpret_cast<const int4*>(V_bh);
        int4* K_dst = reinterpret_cast<int4*>(smem_K0);
        int4* V_dst = reinterpret_cast<int4*>(smem_V0);
        constexpr int i4_per_row = D / 8;
        for (int i = tid; i < valid_kv * i4_per_row; i += THREADS) {
            cp_async_16B((uint32_t)__cvta_generic_to_shared(&K_dst[i]), &K_src[i]);
            cp_async_16B((uint32_t)__cvta_generic_to_shared(&V_dst[i]), &V_src[i]);
        }
    }
    cp_async_commit();
    cp_async_wait_group<0>();
    __syncthreads();

    for (int kv = 0; kv < n_kv; kv++) {
        int cur = kv % 2;
        __nv_bfloat16* smem_K = cur == 0 ? smem_K0 : smem_K1;
        __nv_bfloat16* smem_V = cur == 0 ? smem_V0 : smem_V1;
        int kv_start = kv * BC;
        int valid_kv = min(BC, S - kv_start);

        // Prefetch next K, V
        if (kv + 1 < n_kv) {
            int next = (kv + 1) % 2;
            __nv_bfloat16* nK = next == 0 ? smem_K0 : smem_K1;
            __nv_bfloat16* nV = next == 0 ? smem_V0 : smem_V1;
            int nks = (kv + 1) * BC;
            int nvk = min(BC, S - nks);
            const int4* K_src = reinterpret_cast<const int4*>(K_bh);
            const int4* V_src = reinterpret_cast<const int4*>(V_bh);
            int4* K_dst = reinterpret_cast<int4*>(nK);
            int4* V_dst = reinterpret_cast<int4*>(nV);
            constexpr int i4pr = D / 8;
            for (int i = tid; i < nvk * i4pr; i += THREADS) {
                cp_async_16B((uint32_t)__cvta_generic_to_shared(&K_dst[i]),
                             &K_src[nks * i4pr + i]);
                cp_async_16B((uint32_t)__cvta_generic_to_shared(&V_dst[i]),
                             &V_src[nks * i4pr + i]);
            }
            cp_async_commit();
        }

        // S = Q @ K^T
        #pragma unroll
        for (int n_iter = 0; n_iter < 4; n_iter++) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag;
            wmma::fill_fragment(s_frag, 0.0f);
            #pragma unroll
            for (int k_iter = 0; k_iter < 8; k_iter++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
                wmma::load_matrix_sync(a_frag, &smem_Q[(warp_id*16)*D + k_iter*16], D);
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
                wmma::load_matrix_sync(b_frag, &smem_K[n_iter*16*D + k_iter*16], D);
                wmma::mma_sync(s_frag, a_frag, b_frag, s_frag);
            }
            #pragma unroll
            for (int i = 0; i < s_frag.num_elements; i++) s_frag.x[i] *= scale;
            wmma::store_matrix_sync(&smem_S[(warp_id*16)*BC + n_iter*16], s_frag, BC, wmma::mem_row_major);
        }
        __syncthreads();

        // Softmax: rowmax with inline masking
        float mx = -INFINITY;
        for (int c = half*32; c < half*32+32; c++) {
            float sv = (c < valid_kv) ? smem_S[row*BC + c] : -INFINITY;
            mx = fmaxf(mx, sv);
        }
        mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFF, mx, 16));

        float m_old = smem_m[row];
        float m_new = fmaxf(m_old, mx);
        float rf = (m_old == -INFINITY) ? 1.0f : exp2f((m_old - m_new) * LN2_RCP);
        if (half == 0) { smem_rescale[row] = rf; smem_m[row] = m_new; }
        __syncthreads();

        // Rescale O
        for (int i = tid; i < BR * D; i += THREADS)
            smem_O[i] *= smem_rescale[i / D];

        // Compute P and rowsum
        float mc = m_new;
        float rs = 0.0f;
        for (int c = half*32; c < half*32+32; c++) {
            if (c < valid_kv) {
                float sv = smem_S[row*BC + c];
                float pv = exp2f((sv - mc) * LN2_RCP);
                smem_P[row*BC + c] = __float2bfloat16(pv);
                rs += pv;
            } else {
                smem_P[row*BC + c] = __float2bfloat16(0.0f);
            }
        }
        rs += __shfl_xor_sync(0xFFFFFFFF, rs, 16);
        if (half == 0) { smem_l[row] = smem_l[row] * rf + rs; }
        __syncthreads();

        // O += P @ V
        #pragma unroll
        for (int n_iter = 0; n_iter < 8; n_iter++) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_frag;
            wmma::load_matrix_sync(o_frag, &smem_O[(warp_id*16)*D + n_iter*16], D, wmma::mem_row_major);
            #pragma unroll
            for (int k_iter = 0; k_iter < 4; k_iter++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> p_frag;
                wmma::load_matrix_sync(p_frag, &smem_P[(warp_id*16)*BC + k_iter*16], BC);
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> v_frag;
                wmma::load_matrix_sync(v_frag, &smem_V[k_iter*16*D + n_iter*16], D);
                wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
            }
            wmma::store_matrix_sync(&smem_O[(warp_id*16)*D + n_iter*16], o_frag, D, wmma::mem_row_major);
        }
        __syncthreads();

        if (kv + 1 < n_kv) {
            cp_async_wait_group<0>();
            __syncthreads();
        }
    }

    // Final: LSE and normalize O
    for (int i = tid; i < BR; i += THREADS) {
        int q_idx = q_start + i;
        if (q_idx < S) LSE_bh[q_idx] = smem_m[i] + logf(smem_l[i]);
    }
    for (int i = tid; i < BR * D; i += THREADS) {
        int rrow = i / D;
        int q_idx = q_start + rrow;
        if (q_idx < S) {
            float l = smem_l[rrow];
            float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
            O_bh[q_idx * D + (i % D)] = __float2bfloat16(smem_O[i] * inv_l);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    const int B = 4;
    const int H = 48;
    int64_t S = Q.size(2);

    const __nv_bfloat16* Q_d = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_d = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_d = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_d = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_d = static_cast<float*>(LSE.data_ptr());

    int q_blocks = (S + BR - 1) / BR;
    dim3 grid(B * H, q_blocks);
    dim3 block(THREADS);

    size_t smem_size = (size_t)BR * D * sizeof(__nv_bfloat16)       // Q
                     + (size_t)2 * BC * D * sizeof(__nv_bfloat16)    // K[0,1]
                     + (size_t)2 * BC * D * sizeof(__nv_bfloat16)    // V[0,1]
                     + (size_t)BR * BC * sizeof(float)               // S
                     + (size_t)BR * BC * sizeof(__nv_bfloat16)       // P
                     + (size_t)BR * D * sizeof(float);               // O

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(mha_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size));

    mha_kernel<<<grid, block, smem_size, stream>>>(
        Q_d, K_d, V_d, O_d, LSE_d, B, H, (int)S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_attention::run);

}  // namespace mha_attention