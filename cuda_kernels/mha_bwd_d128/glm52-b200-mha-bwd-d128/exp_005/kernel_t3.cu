#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <math.h>
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
} while(0)

namespace attn_bwd {

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int D = 128;
constexpr int THREADS = 128;
constexpr int SMEM_SIZE = 156160;

__global__ void attn_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    float* __restrict__ dQ_float,
    int B, int H, int S)
{
    int kv_block = blockIdx.x;
    int bh = blockIdx.y;
    int h = bh % H;
    int b = bh / H;
    int kv_start = kv_block * BN;

    int tid = threadIdx.x;
    int warp_id = tid / 32;

    const float scale = 1.0f / sqrtf((float)D);

    size_t bh_offset = ((size_t)b * H + h) * S * D;
    size_t stats_offset = ((size_t)b * H + h) * S;

    const __nv_bfloat16* Q_bh = Q + bh_offset;
    const __nv_bfloat16* K_bh = K + bh_offset;
    const __nv_bfloat16* V_bh = V + bh_offset;
    const __nv_bfloat16* O_bh = O + bh_offset;
    const __nv_bfloat16* dO_bh = dO + bh_offset;
    const float* L_bh = L + stats_offset;
    __nv_bfloat16* dK_bh = dK + bh_offset;
    __nv_bfloat16* dV_bh = dV + bh_offset;
    float* dQ_bh = dQ_float + bh_offset;

    extern __shared__ char smem_buffer[];
    char* ptr = smem_buffer;
    float* dK_acc     = reinterpret_cast<float*>(ptr);              ptr += BN * D * 4;
    float* dV_acc     = reinterpret_cast<float*>(ptr);              ptr += BN * D * 4;
    __nv_bfloat16* K_smem  = reinterpret_cast<__nv_bfloat16*>(ptr); ptr += BN * D * 2;
    __nv_bfloat16* V_smem  = reinterpret_cast<__nv_bfloat16*>(ptr); ptr += BN * D * 2;
    __nv_bfloat16* Q_smem  = reinterpret_cast<__nv_bfloat16*>(ptr); ptr += BM * D * 2;
    __nv_bfloat16* dO_smem = reinterpret_cast<__nv_bfloat16*>(ptr); ptr += BM * D * 2;
    float* temp_smem        = reinterpret_cast<float*>(ptr);        ptr += BM * BN * 4;
    __nv_bfloat16* P_bf16   = reinterpret_cast<__nv_bfloat16*>(ptr);
    float* L_smem           = reinterpret_cast<float*>(ptr + BM * BN * 2);
    float* D_smem           = L_smem + BM;

    // Load K, V with vectorized 16-byte loads
    for (int i = tid; i < BN * D / 8; i += THREADS) {
        int row = (i * 8) / D;
        int col = (i * 8) % D;
        int gr = kv_start + row;
        if (gr < S) {
            *(int4*)(&K_smem[i * 8]) = *(int4*)(&K_bh[gr * D + col]);
            *(int4*)(&V_smem[i * 8]) = *(int4*)(&V_bh[gr * D + col]);
        } else {
            *(int4*)(&K_smem[i * 8]) = make_int4(0, 0, 0, 0);
            *(int4*)(&V_smem[i * 8]) = make_int4(0, 0, 0, 0);
        }
    }

    for (int i = tid; i < BN * D; i += THREADS) {
        dK_acc[i] = 0.0f;
        dV_acc[i] = 0.0f;
    }
    __syncthreads();

    int num_q_blocks = (S + BM - 1) / BM;
    for (int qb = 0; qb < num_q_blocks; qb++) {
        int q_start = qb * BM;

        // Load Q, dO with vectorized loads
        for (int i = tid; i < BM * D / 8; i += THREADS) {
            int row = (i * 8) / D;
            int col = (i * 8) % D;
            int gr = q_start + row;
            if (gr < S) {
                *(int4*)(&Q_smem[i * 8]) = *(int4*)(&Q_bh[gr * D + col]);
                *(int4*)(&dO_smem[i * 8]) = *(int4*)(&dO_bh[gr * D + col]);
            } else {
                *(int4*)(&Q_smem[i * 8]) = make_int4(0, 0, 0, 0);
                *(int4*)(&dO_smem[i * 8]) = make_int4(0, 0, 0, 0);
            }
        }

        for (int i = tid; i < BM; i += THREADS) {
            int gr = q_start + i;
            L_smem[i] = (gr < S) ? L_bh[gr] : 0.0f;
        }
        __syncthreads();

        // Compute D[i] = rowsum(dO_i * O_i)
        if (tid < BM) {
            int gr = q_start + tid;
            float d_val = 0.0f;
            if (gr < S) {
                for (int col = 0; col < D; col++) {
                    d_val += __bfloat162float(O_bh[gr * D + col]) *
                             __bfloat162float(dO_smem[tid * D + col]);
                }
            }
            D_smem[tid] = d_val;
        }
        __syncthreads();

        // === S[BM,BN] = Q @ K^T via wmma ===
        {
            int wr = warp_id / 2;
            int wc = warp_id % 2;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> cF[2][2];

            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 2; j++)
                    wmma::fill_fragment(cF[i][j], 0.0f);

            for (int k = 0; k < D; k += 16) {
                for (int i = 0; i < 2; i++) {
                    int row = wr * 32 + i * 16;
                    wmma::load_matrix_sync(a_frag, &Q_smem[row * D + k], D);
                    for (int j = 0; j < 2; j++) {
                        int col = wc * 32 + j * 16;
                        wmma::load_matrix_sync(b_frag, &K_smem[col * D + k], D);
                        wmma::mma_sync(cF[i][j], a_frag, b_frag, cF[i][j]);
                    }
                }
            }
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 2; j++) {
                    int row = wr * 32 + i * 16;
                    int col = wc * 32 + j * 16;
                    wmma::store_matrix_sync(&temp_smem[row * BN + col], cF[i][j], BN, wmma::mem_row_major);
                }
        }
        __syncthreads();

        // P = exp(S * scale - L), convert to bf16
        for (int i = tid; i < BM * BN; i += THREADS) {
            int row = i / BN;
            int col = i % BN;
            int gr = q_start + row;
            int gcol = kv_start + col;
            if (gr < S && gcol < S) {
                float p = expf(temp_smem[i] * scale - L_smem[row]);
                temp_smem[i] = p;
                P_bf16[i] = __float2bfloat16(p);
            } else {
                temp_smem[i] = 0.0f;
                P_bf16[i] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        // === dV[BN,D] += P^T[BN,BM] @ dO[BM,D] via wmma ===
        {
            int wr = warp_id / 2;
            int wc = warp_id % 2;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> cF[2][4];

            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 4; j++) {
                    int row = wr * 32 + i * 16;
                    int col = wc * 64 + j * 16;
                    wmma::load_matrix_sync(cF[i][j], &dV_acc[row * D + col], D, wmma::mem_row_major);
                }

            for (int k = 0; k < BM; k += 16) {
                for (int i = 0; i < 2; i++) {
                    int row = wr * 32 + i * 16;
                    wmma::load_matrix_sync(a_frag, &P_bf16[k * BN + row], BN);
                    for (int j = 0; j < 4; j++) {
                        int col = wc * 64 + j * 16;
                        wmma::load_matrix_sync(b_frag, &dO_smem[k * D + col], D);
                        wmma::mma_sync(cF[i][j], a_frag, b_frag, cF[i][j]);
                    }
                }
            }
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 4; j++) {
                    int row = wr * 32 + i * 16;
                    int col = wc * 64 + j * 16;
                    wmma::store_matrix_sync(&dV_acc[row * D + col], cF[i][j], D, wmma::mem_row_major);
                }
        }
        __syncthreads();

        // === dP[BM,BN] = dO @ V^T via wmma ===
        {
            int wr = warp_id / 2;
            int wc = warp_id % 2;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> cF[2][2];

            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 2; j++)
                    wmma::fill_fragment(cF[i][j], 0.0f);

            for (int k = 0; k < D; k += 16) {
                for (int i = 0; i < 2; i++) {
                    int row = wr * 32 + i * 16;
                    wmma::load_matrix_sync(a_frag, &dO_smem[row * D + k], D);
                    for (int j = 0; j < 2; j++) {
                        int col = wc * 32 + j * 16;
                        wmma::load_matrix_sync(b_frag, &V_smem[col * D + k], D);
                        wmma::mma_sync(cF[i][j], a_frag, b_frag, cF[i][j]);
                    }
                }
            }
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 2; j++) {
                    int row = wr * 32 + i * 16;
                    int col = wc * 32 + j * 16;
                    wmma::store_matrix_sync(&temp_smem[row * BN + col], cF[i][j], BN, wmma::mem_row_major);
                }
        }
        __syncthreads();

        // dS = P * (dP - D) * scale → bf16 (overwrite P_bf16)
        for (int i = tid; i < BM * BN; i += THREADS) {
            int row = i / BN;
            float p = __bfloat162float(P_bf16[i]);
            float dp = temp_smem[i];
            float d = D_smem[row];
            P_bf16[i] = __float2bfloat16(p * (dp - d) * scale);
        }
        __syncthreads();

        // === dK[BN,D] += dS^T[BN,BM] @ Q[BM,D] via wmma ===
        {
            int wr = warp_id / 2;
            int wc = warp_id % 2;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> cF[2][4];

            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 4; j++) {
                    int row = wr * 32 + i * 16;
                    int col = wc * 64 + j * 16;
                    wmma::load_matrix_sync(cF[i][j], &dK_acc[row * D + col], D, wmma::mem_row_major);
                }

            for (int k = 0; k < BM; k += 16) {
                for (int i = 0; i < 2; i++) {
                    int row = wr * 32 + i * 16;
                    wmma::load_matrix_sync(a_frag, &P_bf16[k * BN + row], BN);
                    for (int j = 0; j < 4; j++) {
                        int col = wc * 64 + j * 16;
                        wmma::load_matrix_sync(b_frag, &Q_smem[k * D + col], D);
                        wmma::mma_sync(cF[i][j], a_frag, b_frag, cF[i][j]);
                    }
                }
            }
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 4; j++) {
                    int row = wr * 32 + i * 16;
                    int col = wc * 64 + j * 16;
                    wmma::store_matrix_sync(&dK_acc[row * D + col], cF[i][j], D, wmma::mem_row_major);
                }
        }
        __syncthreads();

        // === dQ[BM,D] += dS[BM,BN] @ K[BN,D] via wmma ===
        {
            int wr = warp_id / 2;
            int wc = warp_id % 2;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> cF[2][4];

            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 4; j++)
                    wmma::fill_fragment(cF[i][j], 0.0f);

            for (int k = 0; k < BN; k += 16) {
                for (int i = 0; i < 2; i++) {
                    int row = wr * 32 + i * 16;
                    wmma::load_matrix_sync(a_frag, &P_bf16[row * BN + k], BN);
                    for (int j = 0; j < 4; j++) {
                        int col = wc * 64 + j * 16;
                        wmma::load_matrix_sync(b_frag, &K_smem[k * D + col], D);
                        wmma::mma_sync(cF[i][j], a_frag, b_frag, cF[i][j]);
                    }
                }
            }

            // Phase 1: store cols 0-63 to temp_smem, atomicAdd to global
            if (wc == 0) {
                for (int i = 0; i < 2; i++)
                    for (int j = 0; j < 4; j++) {
                        int row = wr * 32 + i * 16;
                        int col = j * 16;
                        wmma::store_matrix_sync(&temp_smem[row * 64 + col], cF[i][j], 64, wmma::mem_row_major);
                    }
            }
            __syncthreads();
            for (int i = tid; i < BM * 64; i += THREADS) {
                int row = i / 64;
                int col = i % 64;
                int gr = q_start + row;
                if (gr < S) atomicAdd(&dQ_bh[gr * D + col], temp_smem[i]);
            }
            __syncthreads();

            // Phase 2: store cols 64-127, atomicAdd to global
            if (wc == 1) {
                for (int i = 0; i < 2; i++)
                    for (int j = 0; j < 4; j++) {
                        int row = wr * 32 + i * 16;
                        int col = j * 16;
                        wmma::store_matrix_sync(&temp_smem[row * 64 + col], cF[i][j], 64, wmma::mem_row_major);
                    }
            }
            __syncthreads();
            for (int i = tid; i < BM * 64; i += THREADS) {
                int row = i / 64;
                int col = i % 64;
                int gr = q_start + row;
                if (gr < S) atomicAdd(&dQ_bh[gr * D + col + 64], temp_smem[i]);
            }
        }
        __syncthreads();
    }

    // Store dK, dV to global
    for (int i = tid; i < BN * D / 8; i += THREADS) {
        int row = (i * 8) / D;
        int col = (i * 8) % D;
        int gr = kv_start + row;
        if (gr < S) {
            int4 dk_vals, dv_vals;
            float* dk_f = (float*)&dk_vals;
            float* dv_f = (float*)&dv_vals;
            for (int j = 0; j < 4; j++) {
                dk_f[j] = dK_acc[i * 8 + j * 2];
                dv_f[j] = dV_acc[i * 8 + j * 2];
            }
            // Pack 8 bf16 values into int4
            __nv_bfloat16* dk_bf = (__nv_bfloat16*)&dk_vals;
            __nv_bfloat16* dv_bf = (__nv_bfloat16*)&dv_vals;
            for (int j = 0; j < 8; j++) {
                dk_bf[j] = __float2bfloat16(dK_acc[i * 8 + j]);
                dv_bf[j] = __float2bfloat16(dV_acc[i * 8 + j]);
            }
            *(int4*)(&dK_bh[gr * D + col]) = dk_vals;
            *(int4*)(&dV_bh[gr * D + col]) = dv_vals;
        }
    }
}

__global__ void convert_dQ_kernel(const float* __restrict__ dQ_float,
                                   __nv_bfloat16* __restrict__ dQ, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dQ[idx] = __float2bfloat16(dQ_float[idx]);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = 4;
    int H = 48;
    int S = static_cast<int>(Q.size(2));

    const __nv_bfloat16* Q_data  = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data  = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data  = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_data  = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_data = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_data          = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_data       = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_data       = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_data       = static_cast<__nv_bfloat16*>(dV.data_ptr());

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    size_t total_elems = (size_t)B * H * S * D;

    float* dQ_float = nullptr;
    CUDA_CHECK(cudaMalloc(&dQ_float, total_elems * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(dQ_float, 0, total_elems * sizeof(float), stream));

    int num_kv_blocks = (S + BN - 1) / BN;
    dim3 grid(num_kv_blocks, B * H);
    dim3 block(THREADS);
    int smem_size = SMEM_SIZE;

    CUDA_CHECK(cudaFuncSetAttribute(
        attn_bwd_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    attn_bwd_kernel<<<grid, block, smem_size, stream>>>(
        Q_data, K_data, V_data, O_data, dO_data, L_data,
        dK_data, dV_data, dQ_float, B, H, S);

    CUDA_CHECK(cudaGetLastError());

    int convert_threads = 256;
    int convert_blocks = (total_elems + convert_threads - 1) / convert_threads;
    convert_dQ_kernel<<<convert_blocks, convert_threads, 0, stream>>>(
        dQ_float, dQ_data, total_elems);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(dQ_float));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, attn_bwd::run);

}  // namespace attn_bwd