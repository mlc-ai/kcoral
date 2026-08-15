#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cmath>
#include <stdio.h>
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
} while (0)

namespace attention_bwd {

constexpr int D_HEAD = 128;
constexpr int BQ = 64;
constexpr int BKV = 128;
constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
constexpr int NWARPS = 8;
constexpr int NTHREADS = 256;

// Tile counts
constexpr int S_TILES = BQ / WMMA_M;      // 4
constexpr int KV_TILES = BKV / WMMA_N;    // 8
constexpr int D_TILES = D_HEAD / WMMA_N;  // 8

// dK/dV: BKV x D = 8x8 = 64 tiles, 8 per warp
// dQ: BQ x D = 4x8 = 32 tiles, 4 per warp
// S/dP: BQ x BKV = 4x8 = 32 tiles, 4 per warp

constexpr int SMEM_SIZE =
    BKV * D_HEAD * 2 +      // Kj = 32KB
    BKV * D_HEAD * 2 +      // Vj = 32KB
    BQ * D_HEAD * 2 +       // Qi = 16KB
    BQ * D_HEAD * 2 +       // dOi = 16KB
    BQ * BKV * 4 +          // S/dP float = 32KB
    BQ * BKV * 2 +          // P_bf16 = 16KB
    BQ * BKV * 2 +          // dS_bf16 = 16KB
    BQ * 4 * 2 +            // Li + Di = 512B
    NWARPS * WMMA_M * WMMA_N * 4; // staging = 8KB
// Total: ~160KB

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

__global__ void compute_D_kernel(
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    float* __restrict__ D,
    int total_rows) {
    int row = blockIdx.x;
    if (row >= total_rows) return;
    int tid = threadIdx.x;
    const __nv_bfloat16* O_ptr = O + (size_t)row * D_HEAD;
    const __nv_bfloat16* dO_ptr = dO + (size_t)row * D_HEAD;
    float sum = 0.0f;
    for (int i = tid; i < D_HEAD; i += blockDim.x)
        sum += __bfloat162float(O_ptr[i]) * __bfloat162float(dO_ptr[i]);
    sum = warp_reduce_sum(sum);
    __shared__ float warp_sums[8];
    int warp_id = tid / 32, lane_id = tid % 32;
    if (lane_id == 0) warp_sums[warp_id] = sum;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (int w = 0; w < blockDim.x / 32; w++) total += warp_sums[w];
        D[row] = total;
    }
}

__global__ void zero_float_kernel(float* ptr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) ptr[idx] = 0.0f;
}

__global__ void convert_f32_to_bf16_kernel(
    const float* __restrict__ src,
    __nv_bfloat16* __restrict__ dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __float2bfloat16(src[idx]);
}

__global__ __launch_bounds__(NTHREADS, 1) void attention_backward_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ D_arr,
    float* __restrict__ dQ_float,
    __nv_bfloat16* __restrict__ dK_out,
    __nv_bfloat16* __restrict__ dV_out,
    int B, int H, int S) {

    extern __shared__ char smem[];
    __nv_bfloat16* Kj_smem = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* Vj_smem = Kj_smem + BKV * D_HEAD;
    __nv_bfloat16* Qi_smem = Vj_smem + BKV * D_HEAD;
    __nv_bfloat16* dOi_smem = Qi_smem + BQ * D_HEAD;
    float* S_smem = reinterpret_cast<float*>(dOi_smem + BQ * D_HEAD);
    __nv_bfloat16* P_bf16 = reinterpret_cast<__nv_bfloat16*>(S_smem + BQ * BKV);
    __nv_bfloat16* dS_bf16 = P_bf16 + BQ * BKV;
    float* Li_smem = reinterpret_cast<float*>(dS_bf16 + BQ * BKV);
    float* Di_smem = Li_smem + BQ;
    float* staging = Di_smem + BQ;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int num_kv_blocks = (S + BKV - 1) / BKV;
    int bh_idx = blockIdx.x / num_kv_blocks;
    int kv_block = blockIdx.x % num_kv_blocks;
    int b = bh_idx / H;
    int h = bh_idx % H;
    int kv_start = kv_block * BKV;

    size_t bh_offset = (size_t)(b * H + h) * S * D_HEAD;
    const __nv_bfloat16* Q_base = Q + bh_offset;
    const __nv_bfloat16* K_base = K + bh_offset;
    const __nv_bfloat16* V_base = V + bh_offset;
    const __nv_bfloat16* dO_base = dO + bh_offset;
    const float* L_base = L + (size_t)(b * H + h) * S;
    const float* D_base = D_arr + (size_t)(b * H + h) * S;
    float* dQ_base = dQ_float + bh_offset;
    __nv_bfloat16* dK_base = dK_out + bh_offset;
    __nv_bfloat16* dV_base = dV_out + bh_offset;

    // Load Kj, Vj using vectorized 128-bit loads
    {
        int4* Kj_v = reinterpret_cast<int4*>(Kj_smem);
        int4* Vj_v = reinterpret_cast<int4*>(Vj_smem);
        int total_vecs = BKV * D_HEAD / 8; // 128*128/8 = 2048
        for (int i = tid; i < total_vecs; i += NTHREADS) {
            int row = i / (D_HEAD / 8);
            int col_vec = i % (D_HEAD / 8);
            int gr = kv_start + row;
            if (gr < S) {
                Kj_v[i] = *reinterpret_cast<const int4*>(&K_base[(size_t)gr * D_HEAD + col_vec * 8]);
                Vj_v[i] = *reinterpret_cast<const int4*>(&V_base[(size_t)gr * D_HEAD + col_vec * 8]);
            } else {
                Kj_v[i] = make_int4(0, 0, 0, 0);
                Vj_v[i] = make_int4(0, 0, 0, 0);
            }
        }
    }
    __syncthreads();

    // dK/dV accumulators: BKV x D = 8x8 = 64 tiles, 8 per warp
    // Warp w: row_base = (w/4)*2, col_base = (w%4)*2
    // 4 tiles: (r,c), (r,c+1), (r+1,c), (r+1,c+1)
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> dK_frag[4];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> dV_frag[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        wmma::fill_fragment(dK_frag[i], 0.0f);
        wmma::fill_fragment(dV_frag[i], 0.0f);
    }

    const float scale = 0.08838834764f;

    for (int q_start = 0; q_start < S; q_start += BQ) {
        // Load Qi, dOi with vectorized loads
        {
            int4* Qi_v = reinterpret_cast<int4*>(Qi_smem);
            int4* dOi_v = reinterpret_cast<int4*>(dOi_smem);
            int total_vecs = BQ * D_HEAD / 8; // 64*128/8 = 1024
            for (int i = tid; i < total_vecs; i += NTHREADS) {
                int row = i / (D_HEAD / 8);
                int col_vec = i % (D_HEAD / 8);
                int gr = q_start + row;
                if (gr < S) {
                    Qi_v[i] = *reinterpret_cast<const int4*>(&Q_base[(size_t)gr * D_HEAD + col_vec * 8]);
                    dOi_v[i] = *reinterpret_cast<const int4*>(&dO_base[(size_t)gr * D_HEAD + col_vec * 8]);
                } else {
                    Qi_v[i] = make_int4(0, 0, 0, 0);
                    dOi_v[i] = make_int4(0, 0, 0, 0);
                }
            }
        }
        if (tid < BQ) {
            int gr = q_start + tid;
            Li_smem[tid] = (gr < S) ? L_base[gr] : 0.0f;
            Di_smem[tid] = (gr < S) ? D_base[gr] : 0.0f;
        }
        __syncthreads();

        // Step 1: S = Qi @ Kj^T (BQ x BKV = 4x8 tiles, 4 per warp, 8 K-steps)
        // Warp w: row = w/8, col_base = (w%8)*1 ... wait, 4x8=32 tiles, 8 warps: 4 per warp
        // Warp w: row_base = w/2, col_base = (w%2)*4
        // Tiles: (r, c), (r, c+1), (r, c+2), (r, c+3)
        {
            int wr = warp_id / 2;  // 0-3
            int wc_base = (warp_id % 2) * 4;  // 0 or 4
            #pragma unroll
            for (int wj = 0; wj < 4; wj++) {
                int ct = wc_base + wj;
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
                wmma::fill_fragment(c_frag, 0.0f);
                #pragma unroll
                for (int kk = 0; kk < D_HEAD / WMMA_K; kk++) {
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag;
                    wmma::load_matrix_sync(a_frag, Qi_smem + wr * 16 * D_HEAD + kk * 16, D_HEAD);
                    wmma::load_matrix_sync(b_frag, Kj_smem + ct * 16 * D_HEAD + kk * 16, D_HEAD);
                    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                }
                wmma::store_matrix_sync(S_smem + wr * 16 * BKV + ct * 16, c_frag, BKV, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // Step 2: P = exp(S * scale - L) -> store as bf16
        for (int i = tid; i < BQ * BKV; i += NTHREADS) {
            int row = i / BKV;
            int col = i % BKV;
            int qi = q_start + row;
            int ki = kv_start + col;
            if (qi < S && ki < S) {
                float s_val = S_smem[i] * scale - Li_smem[row];
                P_bf16[i] = __float2bfloat16(__expf(s_val));
            } else {
                P_bf16[i] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        // Step 3: dP = dOi @ Vj^T (reuse S_smem)
        {
            int wr = warp_id / 2;
            int wc_base = (warp_id % 2) * 4;
            #pragma unroll
            for (int wj = 0; wj < 4; wj++) {
                int ct = wc_base + wj;
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
                wmma::fill_fragment(c_frag, 0.0f);
                #pragma unroll
                for (int kk = 0; kk < D_HEAD / WMMA_K; kk++) {
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag;
                    wmma::load_matrix_sync(a_frag, dOi_smem + wr * 16 * D_HEAD + kk * 16, D_HEAD);
                    wmma::load_matrix_sync(b_frag, Vj_smem + ct * 16 * D_HEAD + kk * 16, D_HEAD);
                    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                }
                wmma::store_matrix_sync(S_smem + wr * 16 * BKV + ct * 16, c_frag, BKV, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // Step 4: dS = P * (dP - D) * scale -> store as bf16
        for (int i = tid; i < BQ * BKV; i += NTHREADS) {
            int row = i / BKV;
            int col = i % BKV;
            int qi = q_start + row;
            int ki = kv_start + col;
            if (qi < S && ki < S) {
                float p_val = __bfloat162float(P_bf16[i]);
                float ds_val = p_val * (S_smem[i] - Di_smem[row]) * scale;
                dS_bf16[i] = __float2bfloat16(ds_val);
            } else {
                dS_bf16[i] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        // Step 5: dV += P^T @ dOi (BKV x D = 8x8 tiles, 4 per warp, 4 K-steps)
        // Warp w: row_base = (w/4)*2, col_base = (w%4)*2
        // Tiles fi: (r+fi/2, c+fi%2)
        {
            int wr_base = (warp_id / 4) * 2;
            int wc_base = (warp_id % 4) * 2;
            #pragma unroll
            for (int fi = 0; fi < 4; fi++) {
                int rt = wr_base + fi / 2;
                int ct = wc_base + fi % 2;
                #pragma unroll
                for (int kk = 0; kk < BQ / WMMA_K; kk++) {
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
                    wmma::load_matrix_sync(a_frag, P_bf16 + kk * 16 * BKV + rt * 16, BKV);
                    wmma::load_matrix_sync(b_frag, dOi_smem + kk * 16 * D_HEAD + ct * 16, D_HEAD);
                    wmma::mma_sync(dV_frag[fi], a_frag, b_frag, dV_frag[fi]);
                }
            }
        }

        // Step 6: dK += dS^T @ Qi (same structure as dV)
        {
            int wr_base = (warp_id / 4) * 2;
            int wc_base = (warp_id % 4) * 2;
            #pragma unroll
            for (int fi = 0; fi < 4; fi++) {
                int rt = wr_base + fi / 2;
                int ct = wc_base + fi % 2;
                #pragma unroll
                for (int kk = 0; kk < BQ / WMMA_K; kk++) {
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
                    wmma::load_matrix_sync(a_frag, dS_bf16 + kk * 16 * BKV + rt * 16, BKV);
                    wmma::load_matrix_sync(b_frag, Qi_smem + kk * 16 * D_HEAD + ct * 16, D_HEAD);
                    wmma::mma_sync(dK_frag[fi], a_frag, b_frag, dK_frag[fi]);
                }
            }
        }

        // Step 7: dQ += dS @ Kj (BQ x D = 4x8 tiles, 4 per warp, 8 K-steps)
        // Warp w: row = w/8, col_base = (w%8)*... 4x8=32 tiles, 8 warps: 4 per warp
        // Warp w: row_base = w/2, col_base = (w%2)*4
        {
            int wr = warp_id / 2;
            int wc_base = (warp_id % 2) * 4;
            #pragma unroll
            for (int wj = 0; wj < 4; wj++) {
                int ct = wc_base + wj;
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> dq_frag;
                wmma::fill_fragment(dq_frag, 0.0f);
                #pragma unroll
                for (int kk = 0; kk < BKV / WMMA_K; kk++) {
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
                    wmma::load_matrix_sync(a_frag, dS_bf16 + wr * 16 * BKV + kk * 16, BKV);
                    wmma::load_matrix_sync(b_frag, Kj_smem + kk * 16 * D_HEAD + ct * 16, D_HEAD);
                    wmma::mma_sync(dq_frag, a_frag, b_frag, dq_frag);
                }

                // Store to staging and atomic add to global
                float* stage = staging + warp_id * WMMA_M * WMMA_N;
                wmma::store_matrix_sync(stage, dq_frag, WMMA_N, wmma::mem_row_major);
                __syncwarp();

                int grb = q_start + wr * 16;
                int gcb = ct * 16;
                for (int i = lane_id; i < WMMA_M * WMMA_N; i += 32) {
                    int r = i / WMMA_N;
                    int c = i % WMMA_N;
                    int gr = grb + r;
                    if (gr < S) {
                        atomicAdd(&dQ_base[(size_t)gr * D_HEAD + gcb + c], stage[i]);
                    }
                }
            }
        }
        __syncthreads();
    }

    // Store dK, dV to global
    {
        int wr_base = (warp_id / 4) * 2;
        int wc_base = (warp_id % 4) * 2;
        float* stage = staging + warp_id * WMMA_M * WMMA_N;
        #pragma unroll
        for (int fi = 0; fi < 4; fi++) {
            int rt = wr_base + fi / 2;
            int ct = wc_base + fi % 2;
            int grb = kv_start + rt * 16;
            int gcb = ct * 16;

            wmma::store_matrix_sync(stage, dK_frag[fi], WMMA_N, wmma::mem_row_major);
            __syncwarp();
            for (int i = lane_id; i < WMMA_M * WMMA_N; i += 32) {
                int r = i / WMMA_N;
                int c = i % WMMA_N;
                int gr = grb + r;
                if (gr < S)
                    dK_base[(size_t)gr * D_HEAD + gcb + c] = __float2bfloat16(stage[i]);
            }

            wmma::store_matrix_sync(stage, dV_frag[fi], WMMA_N, wmma::mem_row_major);
            __syncwarp();
            for (int i = lane_id; i < WMMA_M * WMMA_N; i += 32) {
                int r = i / WMMA_N;
                int c = i % WMMA_N;
                int gr = grb + r;
                if (gr < S)
                    dV_base[(size_t)gr * D_HEAD + gcb + c] = __float2bfloat16(stage[i]);
            }
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {

    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    const int B = 4, H = 48, S = (int)Q.size(2), d = 128;
    int total_elements = B * H * S * d;
    int total_rows = B * H * S;

    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_ptr = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_ptr = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    float *D_buf, *dQ_float;
    CUDA_CHECK(cudaMalloc(&D_buf, (size_t)total_rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dQ_float, (size_t)total_elements * sizeof(float)));

    {
        int threads = 256, blocks = (total_elements + threads - 1) / threads;
        zero_float_kernel<<<blocks, threads, 0, stream>>>(dQ_float, total_elements);
    }
    compute_D_kernel<<<total_rows, 128, 0, stream>>>(O_ptr, dO_ptr, D_buf, total_rows);
    {
        int num_kv_blocks = (S + BKV - 1) / BKV;
        int grid = B * H * num_kv_blocks;
        CUDA_CHECK(cudaFuncSetAttribute(attention_backward_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));
        attention_backward_kernel<<<grid, NTHREADS, SMEM_SIZE, stream>>>(
            Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_buf,
            dQ_float, dK_ptr, dV_ptr, B, H, S);
    }
    {
        int threads = 256, blocks = (total_elements + threads - 1) / threads;
        convert_f32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(dQ_float, dQ_ptr, total_elements);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(D_buf));
    CUDA_CHECK(cudaFree(dQ_float));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, attention_bwd::run);

}  // namespace attention_bwd