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
    asm volatile("cp.async.cg.shared.global.16 [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

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

    extern __shared__ char smem[];
    __nv_bfloat16* Q_smem = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* K_smem[2];
    K_smem[0] = Q_smem + BQ * D;
    K_smem[1] = K_smem[0] + BK * D;
    __nv_bfloat16* V_smem[2];
    V_smem[0] = K_smem[1] + BK * D;
    V_smem[1] = V_smem[0] + BK * D;
    float* S_smem = reinterpret_cast<float*>(V_smem[1] + BK * D);
    __nv_bfloat16* P_smem = reinterpret_cast<__nv_bfloat16*>(S_smem + BQ * BK);
    float* O_smem = reinterpret_cast<float*>(P_smem + BQ * BK);
    float* m_smem = O_smem + BQ * D;
    float* l_smem = m_smem + BQ;
    float* rescale_smem = l_smem + BQ;

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
    fragment<accumulator, 16, 16, 16, float> c_frag;

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

        // Q@K^T -> S_smem
        for (int n = 0; n < BK / 16; n++) {
            fill_fragment(c_frag, 0.0f);
            for (int k = 0; k < D / 16; k++) {
                load_matrix_sync(a_frag, Q_smem + warp_id * 16 * D + k * 16, D);
                load_matrix_sync(b_frag_qk, K_smem[buf] + n * 16 * D + k * 16, D);
                mma_sync(c_frag, a_frag, b_frag_qk, c_frag);
            }
            for (int i = 0; i < c_frag.num_elements; i++) c_frag.x[i] *= scale;
            store_matrix_sync(S_smem + warp_id * 16 * BK + n * 16, c_frag, BK, mem_row_major);
        }
        __syncthreads();

        // Parallel online softmax: 2 threads per row
        {
            int row = warp_id * 16 + lane_id / 2;
            int col_half = lane_id % 2;
            int col_start = col_half * (BK / 2);

            if (row < BQ && q_start + row < S) {
                float m_local = -INFINITY;
                for (int j = 0; j < BK / 2; j++) {
                    int col = col_start + j;
                    float s_val = (kv + col < S) ? S_smem[row * BK + col] : -INFINITY;
                    m_local = fmaxf(m_local, s_val);
                }
                float m_other = __shfl_xor_sync(0xFFFFFFFF, m_local, 1);
                float m_block = fmaxf(m_local, m_other);

                float m_old = m_smem[row];
                float m_new = fmaxf(m_old, m_block);
                float rescale = (m_old == -INFINITY) ? 0.0f : __expf(m_old - m_new);

                float l_local = 0.0f;
                for (int j = 0; j < BK / 2; j++) {
                    int col = col_start + j;
                    float p = (kv + col < S) ? __expf(S_smem[row * BK + col] - m_new) : 0.0f;
                    P_smem[row * BK + col] = __float2bfloat16(p);
                    l_local += p;
                }
                float l_other = __shfl_xor_sync(0xFFFFFFFF, l_local, 1);
                float l_block = l_local + l_other;

                rescale_smem[row] = rescale;
                l_smem[row] = l_smem[row] * rescale + l_block;
                m_smem[row] = m_new;
            }
        }
        __syncthreads();

        // Rescale O fragments in registers
        {
            int r0_idx = warp_id * 16 + lane_id / 4;
            int r1_idx = r0_idx + 8;
            float r0 = rescale_smem[r0_idx];
            float r1 = rescale_smem[r1_idx];
            for (int n = 0; n < 8; n++) {
                o_frag[n].x[0] *= r0; o_frag[n].x[1] *= r0;
                o_frag[n].x[4] *= r0; o_frag[n].x[5] *= r0;
                o_frag[n].x[2] *= r1; o_frag[n].x[3] *= r1;
                o_frag[n].x[6] *= r1; o_frag[n].x[7] *= r1;
            }
        }

        // P@V -> accumulate into O fragments
        for (int n = 0; n < D / 16; n++) {
            for (int k = 0; k < BK / 16; k++) {
                load_matrix_sync(a_frag, P_smem + warp_id * 16 * BK + k * 16, BK);
                load_matrix_sync(b_frag_pv, V_smem[buf] + k * 16 * D + n * 16, D);
                mma_sync(o_frag[n], a_frag, b_frag_pv, o_frag[n]);
            }
        }
        __syncthreads();
    }

    // Epilogue: store O fragments, normalize, write to global
    for (int n = 0; n < D / 16; n++) {
        store_matrix_sync(O_smem + warp_id * 16 * D + n * 16, o_frag[n], D, mem_row_major);
    }
    __syncthreads();

    if (tid < BQ && q_start + tid < S) {
        LSE_base[q_start + tid] = m_smem[tid] + logf(l_smem[tid]);
    }

    {
        constexpr int vpr = D / 8;
        for (int idx = tid; idx < BQ * vpr; idx += NUM_THREADS) {
            int row = idx / vpr;
            int vi = idx % vpr;
            if (q_start + row >= S) continue;
            float inv_l = 1.0f / l_smem[row];
            __nv_bfloat16 vals[8];
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
                    BQ * BK * 4 +        // S
                    BQ * BK * 2 +        // P
                    BQ * D * 4 +         // O
                    BQ * 4 * 3;          // m, l, rescale

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