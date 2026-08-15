#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda;
using bf16 = __nv_bfloat16;

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", \
                cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

namespace attn_bwd {

constexpr int D = 128;
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int WARPS = 4;
constexpr int THREADS = WARPS * 32;
constexpr float SCALE = 0.08838834764831845f;

using FragA_bf_row = wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major>;
using FragB_bf_col = wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major>;
using FragC_bf = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

using FragA_tf_row = wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major>;
using FragA_tf_col = wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::col_major>;
using FragB_tf_row = wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major>;
using FragC_tf = wmma::fragment<wmma::accumulator, 16, 16, 8, float>;

struct SmemBuf {
    bf16 K_s[BN][D];
    bf16 V_s[BN][D];
    bf16 Q_s[BM][D];
    bf16 dO_s[BM][D];
    float P_f[BM][BN];
    float dS_f[BM][BN];
    float stage_b[WARPS][128];
    float tmp[WARPS][256];
    float L_s[BM];
    float D_s[BM];
};

__global__ void compute_D_kernel(
    const bf16* __restrict__ O,
    const bf16* __restrict__ dO,
    float* __restrict__ D_out,
    int total_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= total_rows) return;
    const bf16* o_ptr = O + row * D;
    const bf16* do_ptr = dO + row * D;
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < D; i++) {
        sum += __bfloat162float(o_ptr[i]) * __bfloat162float(do_ptr[i]);
    }
    D_out[row] = sum;
}

__global__ void convert_kernel(const float* src, bf16* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __float2bfloat16(src[idx]);
}

__device__ __forceinline__ void stage_bf16_to_float(bf16* src, float* dst, int ldm, int lane_id) {
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int idx = lane_id + i * 32;
        int row = idx / 16;
        int col = idx % 16;
        dst[idx] = __bfloat162float(src[row * ldm + col]);
    }
    __syncwarp();
}

__global__ void attn_backward_kernel(
    const bf16* __restrict__ Q_g,
    const bf16* __restrict__ K_g,
    const bf16* __restrict__ V_g,
    const bf16* __restrict__ dO_g,
    const float* __restrict__ L_g,
    const float* __restrict__ D_g,
    float* __restrict__ dQ_buf,
    bf16* __restrict__ dK_out,
    bf16* __restrict__ dV_out,
    int B, int H, int S) {

    extern __shared__ char smem_raw[];
    SmemBuf* smem = reinterpret_cast<SmemBuf*>(smem_raw);

    int bh = blockIdx.x;
    int j_block = blockIdx.y;
    int j_start = j_block * BN;
    int j_end = min(j_start + BN, S);
    int actual_BN = j_end - j_start;
    if (actual_BN <= 0) return;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    {
        int4* k_src = (int4*)(K_g + (bh * S + j_start) * D);
        int4* v_src = (int4*)(V_g + (bh * S + j_start) * D);
        int4* k_dst = (int4*)smem->K_s;
        int4* v_dst = (int4*)smem->V_s;
        int total = actual_BN * D / 8;
        for (int i = tid; i < total; i += THREADS) {
            k_dst[i] = k_src[i];
            v_dst[i] = v_src[i];
        }
        int4 zero = make_int4(0, 0, 0, 0);
        for (int i = actual_BN * D / 8 + tid; i < BN * D / 8; i += THREADS) {
            k_dst[i] = zero;
            v_dst[i] = zero;
        }
    }
    __syncthreads();

    FragC_tf dK_frag[8];
    FragC_tf dV_frag[8];
    #pragma unroll
    for (int tc = 0; tc < 8; tc++) {
        wmma::fill_fragment(dK_frag[tc], 0.0f);
        wmma::fill_fragment(dV_frag[tc], 0.0f);
    }

    int num_q_blocks = (S + BM - 1) / BM;

    for (int i_block = j_block; i_block < num_q_blocks; i_block++) {
        int i_start = i_block * BM;
        int i_end = min(i_start + BM, S);
        int actual_BM = i_end - i_start;
        bool is_causal_block = (i_block == j_block);

        {
            int4* q_src = (int4*)(Q_g + (bh * S + i_start) * D);
            int4* do_src = (int4*)(dO_g + (bh * S + i_start) * D);
            int4* q_dst = (int4*)smem->Q_s;
            int4* do_dst = (int4*)smem->dO_s;
            int total = actual_BM * D / 8;
            for (int i = tid; i < total; i += THREADS) {
                q_dst[i] = q_src[i];
                do_dst[i] = do_src[i];
            }
            int4 zero = make_int4(0, 0, 0, 0);
            for (int i = actual_BM * D / 8 + tid; i < BM * D / 8; i += THREADS) {
                q_dst[i] = zero;
                do_dst[i] = zero;
            }
            for (int i = tid; i < actual_BM; i += THREADS) {
                smem->L_s[i] = L_g[bh * S + i_start + i];
                smem->D_s[i] = D_g[bh * S + i_start + i];
            }
            for (int i = actual_BM + tid; i < BM; i += THREADS) {
                smem->L_s[i] = 0.0f;
                smem->D_s[i] = 0.0f;
            }
        }
        __syncthreads();

        // S = Q @ K^T (BF16 MMA) -> P_f
        {
            FragC_bf s_frag[4];
            #pragma unroll
            for (int tc = 0; tc < 4; tc++)
                wmma::fill_fragment(s_frag[tc], 0.0f);
            FragA_bf_row a_frag;
            FragB_bf_col b_frag;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                wmma::load_matrix_sync(a_frag, &smem->Q_s[warp_id * 16][k * 16], D);
                #pragma unroll
                for (int tc = 0; tc < 4; tc++) {
                    wmma::load_matrix_sync(b_frag, &smem->K_s[tc * 16][k * 16], D);
                    wmma::mma_sync(s_frag[tc], a_frag, b_frag, s_frag[tc]);
                }
            }
            #pragma unroll
            for (int tc = 0; tc < 4; tc++)
                wmma::store_matrix_sync(&smem->P_f[warp_id * 16][tc * 16], s_frag[tc], BN, wmma::mem_row_major);
        }
        __syncthreads();

        // P = exp(S*SCALE - L) with causal mask
        {
            for (int i = tid; i < BM * BN; i += THREADS) {
                int r = i / BN;
                int c = i % BN;
                float s_val = smem->P_f[r][c];
                float l_val = smem->L_s[r];
                if (is_causal_block && (c > r)) {
                    smem->P_f[r][c] = 0.0f;
                } else {
                    smem->P_f[r][c] = expf(s_val * SCALE - l_val);
                }
            }
        }
        __syncthreads();

        // dP = dO @ V^T (BF16 MMA) -> dS_f
        {
            FragC_bf dp_frag[4];
            #pragma unroll
            for (int tc = 0; tc < 4; tc++)
                wmma::fill_fragment(dp_frag[tc], 0.0f);
            FragA_bf_row a_frag;
            FragB_bf_col b_frag;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                wmma::load_matrix_sync(a_frag, &smem->dO_s[warp_id * 16][k * 16], D);
                #pragma unroll
                for (int tc = 0; tc < 4; tc++) {
                    wmma::load_matrix_sync(b_frag, &smem->V_s[tc * 16][k * 16], D);
                    wmma::mma_sync(dp_frag[tc], a_frag, b_frag, dp_frag[tc]);
                }
            }
            #pragma unroll
            for (int tc = 0; tc < 4; tc++)
                wmma::store_matrix_sync(&smem->dS_f[warp_id * 16][tc * 16], dp_frag[tc], BN, wmma::mem_row_major);
        }
        __syncthreads();

        // dS = P * (dP - D) * SCALE -> dS_f
        {
            for (int i = tid; i < BM * BN; i += THREADS) {
                int r = i / BN;
                int c = i % BN;
                float p_val = smem->P_f[r][c];
                float dp_val = smem->dS_f[r][c];
                float d_val = smem->D_s[r];
                smem->dS_f[r][c] = p_val * (dp_val - d_val) * SCALE;
            }
        }
        __syncthreads();

        // dQ += dS @ K (TF32 MMA with K staging)
        {
            for (int tc_batch = 0; tc_batch < 8; tc_batch += 4) {
                FragC_tf dq_frag[4];
                #pragma unroll
                for (int t = 0; t < 4; t++)
                    wmma::fill_fragment(dq_frag[t], 0.0f);
                FragA_tf_row a_frag;
                FragB_tf_row b_frag;
                #pragma unroll
                for (int k = 0; k < 8; k++) {
                    wmma::load_matrix_sync(a_frag, &smem->dS_f[warp_id * 16][k * 8], BN);
                    #pragma unroll
                    for (int t = 0; t < 4; t++) {
                        int tc = tc_batch + t;
                        stage_bf16_to_float(&smem->K_s[k * 8][tc * 16], smem->stage_b[warp_id], D, lane_id);
                        wmma::load_matrix_sync(b_frag, smem->stage_b[warp_id], 16);
                        wmma::mma_sync(dq_frag[t], a_frag, b_frag, dq_frag[t]);
                    }
                }
                #pragma unroll
                for (int t = 0; t < 4; t++) {
                    int tc = tc_batch + t;
                    wmma::store_matrix_sync(smem->tmp[warp_id], dq_frag[t], 16, wmma::mem_row_major);
                    __syncwarp();
                    #pragma unroll
                    for (int i = 0; i < 8; i++) {
                        int idx = lane_id * 8 + i;
                        int row = idx / 16;
                        int col = idx % 16;
                        float val = smem->tmp[warp_id][idx];
                        int g_row = i_start + warp_id * 16 + row;
                        int g_col = tc * 16 + col;
                        if (g_row < S) {
                            atomicAdd(&dQ_buf[(bh * S + g_row) * D + g_col], val);
                        }
                    }
                    __syncwarp();
                }
            }
        }

        // dK += dS^T @ Q (TF32 MMA with Q staging)
        {
            FragA_tf_col a_frag;
            FragB_tf_row b_frag;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                wmma::load_matrix_sync(a_frag, &smem->dS_f[k * 8][warp_id * 16], BN);
                #pragma unroll
                for (int tc = 0; tc < 8; tc++) {
                    stage_bf16_to_float(&smem->Q_s[k * 8][tc * 16], smem->stage_b[warp_id], D, lane_id);
                    wmma::load_matrix_sync(b_frag, smem->stage_b[warp_id], 16);
                    wmma::mma_sync(dK_frag[tc], a_frag, b_frag, dK_frag[tc]);
                }
            }
        }
        __syncthreads();

        // dV += P^T @ dO (TF32 MMA with dO staging)
        {
            FragA_tf_col a_frag;
            FragB_tf_row b_frag;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                wmma::load_matrix_sync(a_frag, &smem->P_f[k * 8][warp_id * 16], BN);
                #pragma unroll
                for (int tc = 0; tc < 8; tc++) {
                    stage_bf16_to_float(&smem->dO_s[k * 8][tc * 16], smem->stage_b[warp_id], D, lane_id);
                    wmma::load_matrix_sync(b_frag, smem->stage_b[warp_id], 16);
                    wmma::mma_sync(dV_frag[tc], a_frag, b_frag, dV_frag[tc]);
                }
            }
        }
        __syncthreads();
    }

    // Store dK, dV
    for (int tc = 0; tc < 8; tc++) {
        wmma::store_matrix_sync(smem->tmp[warp_id], dK_frag[tc], 16, wmma::mem_row_major);
        __syncwarp();
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int idx = lane_id * 8 + i;
            int row = idx / 16;
            int col = idx % 16;
            float val = smem->tmp[warp_id][idx];
            int g_row = j_start + warp_id * 16 + row;
            int g_col = tc * 16 + col;
            if (g_row < S) {
                dK_out[(bh * S + g_row) * D + g_col] = __float2bfloat16(val);
            }
        }
        __syncwarp();
    }

    for (int tc = 0; tc < 8; tc++) {
        wmma::store_matrix_sync(smem->tmp[warp_id], dV_frag[tc], 16, wmma::mem_row_major);
        __syncwarp();
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int idx = lane_id * 8 + i;
            int row = idx / 16;
            int col = idx % 16;
            float val = smem->tmp[warp_id][idx];
            int g_row = j_start + warp_id * 16 + row;
            int g_col = tc * 16 + col;
            if (g_row < S) {
                dV_out[(bh * S + g_row) * D + g_col] = __float2bfloat16(val);
            }
        }
        __syncwarp();
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {

    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = 4, H = 48;
    int64_t S = Q.size(2);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    int total_rows = B * H * (int)S;

    float* D_val;
    CUDA_CHECK(cudaMalloc(&D_val, (size_t)total_rows * sizeof(float)));

    int64_t dQ_size = (int64_t)B * H * S * D;
    float* dQ_buf;
    CUDA_CHECK(cudaMalloc(&dQ_buf, (size_t)dQ_size * sizeof(float)));

    {
        int threads = 256;
        int blocks = (total_rows + threads - 1) / threads;
        compute_D_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const bf16*>(O.data_ptr()),
            static_cast<const bf16*>(dO.data_ptr()),
            D_val, total_rows);
    }

    CUDA_CHECK(cudaMemsetAsync(dQ_buf, 0, (size_t)dQ_size * sizeof(float), stream));

    {
        dim3 grid(B * H, ((int)S + BN - 1) / BN);
        dim3 block(THREADS);
        int smem_size = (int)sizeof(SmemBuf);

        CUDA_CHECK(cudaFuncSetAttribute(attn_backward_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

        attn_backward_kernel<<<grid, block, smem_size, stream>>>(
            static_cast<const bf16*>(Q.data_ptr()),
            static_cast<const bf16*>(K.data_ptr()),
            static_cast<const bf16*>(V.data_ptr()),
            static_cast<const bf16*>(dO.data_ptr()),
            static_cast<const float*>(L.data_ptr()),
            D_val,
            dQ_buf,
            static_cast<bf16*>(dK.data_ptr()),
            static_cast<bf16*>(dV.data_ptr()),
            B, H, (int)S);
    }

    {
        int threads = 256;
        int blocks = ((int)dQ_size + threads - 1) / threads;
        convert_kernel<<<blocks, threads, 0, stream>>>(
            dQ_buf, static_cast<bf16*>(dQ.data_ptr()), (int)dQ_size);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFree(D_val));
    CUDA_CHECK(cudaFree(dQ_buf));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, attn_bwd::run);

} // namespace attn_bwd