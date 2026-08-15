#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", \
                cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

namespace mha_cuda {

using namespace nvcuda;

constexpr int D = 128;
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int NUM_THREADS = 128;
constexpr int NUM_WARPS = 4;
constexpr float SCALE = 0.08838834764831845f;
constexpr float LOG2E = 1.4426950408889634f;

__device__ __forceinline__ float fast_expf(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x * LOG2E));
    return y;
}

__device__ __forceinline__ void cp_async_16(uint32_t smem_addr, const void* gmem_addr) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem_addr));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n");
}

__global__ __launch_bounds__(NUM_THREADS, 2)
void attention_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int H, int S
) {
    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* Q_smem = smem;                          // [BM, D] = 64*128
    __nv_bfloat16* KV_smem = Q_smem + BM * D;              // [2, BN, D] = 2*64*128
    float* P_smem = reinterpret_cast<float*>(KV_smem + 2 * BN * D);  // [NUM_WARPS, 16, 64]
    float* O_smem_f = P_smem + NUM_WARPS * 16 * 64;        // [BM, D] float for epilogue

    int q_start = blockIdx.x * BM;
    int bh = blockIdx.y;
    int b = bh / H;
    int h = bh % H;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int64_t base = (int64_t)(b * H + h) * S * D;
    const __nv_bfloat16* Q_bh = Q + base;
    const __nv_bfloat16* K_bh = K + base;
    const __nv_bfloat16* V_bh = V + base;

    // Load Q to Q_smem
    {
        int total = BM * D / 16;
        for (int i = tid; i < total; i += NUM_THREADS) {
            int row = i / (D / 16), col = i % (D / 16);
            int q_row = q_start + row;
            int4* dst = reinterpret_cast<int4*>(Q_smem + row * D + col * 16);
            if (q_row < S) {
                *dst = *reinterpret_cast<const int4*>(Q_bh + (int64_t)q_row * D + col * 16);
            } else {
                *dst = make_int4(0, 0, 0, 0);
            }
        }
    }
    __syncthreads();

    // Load K_0
    {
        int total = BN * D / 16;
        for (int i = tid; i < total; i += NUM_THREADS) {
            int row = i / (D / 16), col = i % (D / 16);
            int k_row = 0 + row;
            if (k_row < S) {
                uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(KV_smem + row * D + col * 16);
                cp_async_16(smem_addr, K_bh + (int64_t)k_row * D + col * 16);
            }
        }
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();
    }

    float m[16], l[16];
    for (int i = 0; i < 16; i++) { m[i] = -INFINITY; l[i] = 0.0f; }

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_frag[8];
    for (int i = 0; i < 8; i++) wmma::fill_fragment(o_frag[i], 0.0f);

    for (int k_start = 0; k_start < S; k_start += BN) {
        int buf_idx = (k_start / BN) % 2;
        __nv_bfloat16* K_buf = KV_smem + buf_idx * BN * D;
        __nv_bfloat16* V_buf = KV_smem + (1 - buf_idx) * BN * D;
        float* P_smem_warp = P_smem + warp_id * 16 * 64;

        // Issue V_i
        {
            int total = BN * D / 16;
            for (int i = tid; i < total; i += NUM_THREADS) {
                int row = i / (D / 16), col = i % (D / 16);
                int k_row = k_start + row;
                if (k_row < S) {
                    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(V_buf + row * D + col * 16);
                    cp_async_16(smem_addr, V_bh + (int64_t)k_row * D + col * 16);
                }
            }
            cp_async_commit();
        }

        // Compute QK_i
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> qk_frag[4];
        for (int n_tile = 0; n_tile < 4; n_tile++) {
            wmma::fill_fragment(qk_frag[n_tile], 0.0f);
            for (int d_tile = 0; d_tile < 8; d_tile++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
                wmma::load_matrix_sync(a_frag, Q_smem + warp_id * 16 * D + d_tile * 16, D);
                wmma::load_matrix_sync(b_frag, K_buf + n_tile * 16 * D + d_tile * 16, D);
                wmma::mma_sync(qk_frag[n_tile], a_frag, b_frag, qk_frag[n_tile]);
            }
            for (int i = 0; i < qk_frag[n_tile].num_elements; i++) {
                qk_frag[n_tile].x[i] *= SCALE;
            }
            wmma::store_matrix_sync(P_smem_warp + n_tile * 16, qk_frag[n_tile], 64, wmma::mem_row_major);
        }

        // Wait V_i
        cp_async_wait_all();
        __syncthreads();

        // Online softmax - each lane handles one row of the 16-row warp tile
        float scale[16];
        if (lane_id < 16) {
            int r = lane_id;
            float max_val = -INFINITY;
            for (int col = 0; col < 64; col++) {
                int k_idx = k_start + col;
                if (k_idx >= S) {
                    P_smem_warp[r * 64 + col] = -INFINITY;
                } else {
                    max_val = fmaxf(max_val, P_smem_warp[r * 64 + col]);
                }
            }
            float m_new = fmaxf(m[r], max_val);
            scale[r] = (m[r] > -INFINITY) ? fast_expf(m[r] - m_new) : 0.0f;
            l[r] *= scale[r];
            float sum = 0.0f;
            for (int col = 0; col < 64; col++) {
                float p = fast_expf(P_smem_warp[r * 64 + col] - m_new);
                P_smem_warp[r * 64 + col] = p;  // Store as float for PV step
                sum += p;
            }
            m[r] = m_new;
            l[r] += sum;
        }
        __syncwarp();

        // Broadcast scale to all lanes
        float my_scale = (lane_id < 16) ? scale[lane_id] : 0.0f;
        float scale0 = __shfl_sync(0xFFFFFFFF, my_scale, lane_id / 4);
        float scale1 = __shfl_sync(0xFFFFFFFF, my_scale, lane_id / 4 + 8);

        // Rescale O accumulator
        for (int d_tile = 0; d_tile < 8; d_tile++) {
            for (int i = 0; i < 8; i++) {
                if (i % 4 < 2) o_frag[d_tile].x[i] *= scale0;
                else o_frag[d_tile].x[i] *= scale1;
            }
        }

        // Compute PV_i - convert P to bf16 in shared memory first
        // Store P as bf16 in the same P_smem area (reuse space)
        __nv_bfloat16* P_bf16 = reinterpret_cast<__nv_bfloat16*>(P_smem_warp);
        if (lane_id < 16) {
            int r = lane_id;
            for (int col = 0; col < 64; col++) {
                P_bf16[r * 64 + col] = __float2bfloat16(P_smem_warp[r * 64 + col]);
            }
        }
        __syncwarp();

        for (int d_tile = 0; d_tile < 8; d_tile++) {
            for (int n_tile = 0; n_tile < 4; n_tile++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> p_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> v_frag;
                wmma::load_matrix_sync(p_frag, P_bf16 + n_tile * 16, 64);
                wmma::load_matrix_sync(v_frag, V_buf + n_tile * 16 * D + d_tile * 16, D);
                wmma::mma_sync(o_frag[d_tile], p_frag, v_frag, o_frag[d_tile]);
            }
        }

        // Issue K_{i+1}
        bool has_next = (k_start + BN < S);
        if (has_next) {
            int next_k_start = k_start + BN;
            int total = BN * D / 16;
            for (int i = tid; i < total; i += NUM_THREADS) {
                int row = i / (D / 16), col = i % (D / 16);
                int k_row = next_k_start + row;
                if (k_row < S) {
                    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(K_buf + row * D + col * 16);
                    cp_async_16(smem_addr, K_bh + (int64_t)k_row * D + col * 16);
                }
            }
            cp_async_commit();
            cp_async_wait_all();
            __syncthreads();
        }
    }

    // Epilogue: normalize and store
    float my_inv_l = (lane_id < 16) ? (1.0f / l[lane_id]) : 0.0f;
    float inv_l0 = __shfl_sync(0xFFFFFFFF, my_inv_l, lane_id / 4);
    float inv_l1 = __shfl_sync(0xFFFFFFFF, my_inv_l, lane_id / 4 + 8);

    for (int d_tile = 0; d_tile < 8; d_tile++) {
        for (int i = 0; i < 8; i++) {
            if (i % 4 < 2) o_frag[d_tile].x[i] *= inv_l0;
            else o_frag[d_tile].x[i] *= inv_l1;
        }
        // Store to float shared memory
        wmma::store_matrix_sync(O_smem_f + warp_id * 16 * D + d_tile * 16, o_frag[d_tile], D, wmma::mem_row_major);
    }
    __syncthreads();

    // Write LSE
    if (lane_id < 16) {
        int r = lane_id;
        int q_row = q_start + warp_id * 16 + r;
        if (q_row < S) {
            LSE[(int64_t)(b * H + h) * S + q_row] = m[r] + logf(l[r]);
        }
    }

    // Convert float O_smem to bf16 and write to global
    {
        int total = BM * D / 4;  // Each thread handles 4 floats -> 4 bf16
        for (int i = tid; i < total; i += NUM_THREADS) {
            int row = i / (D / 4), col = i % (D / 4);
            int q_row = q_start + row;
            if (q_row < S) {
                float f0 = O_smem_f[row * D + col * 4];
                float f1 = O_smem_f[row * D + col * 4 + 1];
                float f2 = O_smem_f[row * D + col * 4 + 2];
                float f3 = O_smem_f[row * D + col * 4 + 3];
                __nv_bfloat16* out = O + base + (int64_t)q_row * D + col * 4;
                out[0] = __float2bfloat16(f0);
                out[1] = __float2bfloat16(f1);
                out[2] = __float2bfloat16(f2);
                out[3] = __float2bfloat16(f3);
            }
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    const int B = 4;
    const int H = 48;
    int S = static_cast<int>(Q.size(2));

    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());

    dim3 grid((S + BM - 1) / BM, B * H);
    dim3 block(NUM_THREADS);

    // Q_smem: 64*128*2=16KB, KV_smem: 2*64*128*2=32KB, P_smem: 4*16*64*4=16KB, O_smem_f: 64*128*4=32KB
    size_t smem_size = BM * D * sizeof(__nv_bfloat16)
                     + 2 * BN * D * sizeof(__nv_bfloat16)
                     + NUM_WARPS * 16 * 64 * sizeof(float)
                     + BM * D * sizeof(float);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(attention_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_size)));

    attention_kernel<<<grid, block, smem_size, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data, H, S);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_cuda::run);

}  // namespace mha_cuda