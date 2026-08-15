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
constexpr int THREADS = 256;

// Shared memory layout (bytes)
// Q_smem:      BM*D*2 = 16384
// dO_smem:     BM*D*2 = 16384
// O/dS_smem:   BM*D*2 = 16384  (O as bf16 at start, dS as float in loop - same size)
// dQ_acc:      BM*D*4 = 32768
// K_smem:      BN*D*2 = 16384
// V_smem:      BN*D*2 = 16384
// temp_smem:   BM*BN*4 = 16384  (S then dP)
// P_smem:      BM*BN*4 = 16384
// L_smem:      BM*4 = 256
// D_smem:      BM*4 = 256
constexpr int SMEM_SIZE = 16384*7 + 32768 + 256*2; // 147968

__global__ void attn_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int B, int H, int S)
{
    int num_q_blocks = (S + BM - 1) / BM;
    int q_block = blockIdx.x % num_q_blocks;
    int bh = blockIdx.x / num_q_blocks;
    int h = bh % H;
    int b = bh / H;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int warp_row = warp_id / 2;
    int warp_col = warp_id % 2;

    int q_start = q_block * BM;
    const float scale = 1.0f / sqrtf((float)D);

    size_t bh_offset = ((size_t)b * H + h) * S * D;
    size_t stats_offset = ((size_t)b * H + h) * S;

    const __nv_bfloat16* Q_bh = Q + bh_offset;
    const __nv_bfloat16* K_bh = K + bh_offset;
    const __nv_bfloat16* V_bh = V + bh_offset;
    const __nv_bfloat16* O_bh = O + bh_offset;
    const __nv_bfloat16* dO_bh = dO + bh_offset;
    const float* L_bh = L + stats_offset;
    __nv_bfloat16* dQ_bh = dQ + bh_offset;
    __nv_bfloat16* dK_bh = dK + bh_offset;
    __nv_bfloat16* dV_bh = dV + bh_offset;

    extern __shared__ char smem_buffer[];
    char* ptr = smem_buffer;
    __nv_bfloat16* Q_smem   = reinterpret_cast<__nv_bfloat16*>(ptr); ptr += BM * D * 2;
    __nv_bfloat16* dO_smem  = reinterpret_cast<__nv_bfloat16*>(ptr); ptr += BM * D * 2;
    // O_smem (bf16) and dS_smem (float) share same region
    __nv_bfloat16* O_smem   = reinterpret_cast<__nv_bfloat16*>(ptr);
    float* dS_smem          = reinterpret_cast<float*>(ptr);         ptr += BM * D * 2;
    float* dQ_acc           = reinterpret_cast<float*>(ptr);         ptr += BM * D * 4;
    __nv_bfloat16* K_smem   = reinterpret_cast<__nv_bfloat16*>(ptr); ptr += BN * D * 2;
    __nv_bfloat16* V_smem   = reinterpret_cast<__nv_bfloat16*>(ptr); ptr += BN * D * 2;
    float* temp_smem        = reinterpret_cast<float*>(ptr);         ptr += BM * BN * 4;
    float* P_smem           = reinterpret_cast<float*>(ptr);         ptr += BM * BN * 4;
    float* L_smem           = reinterpret_cast<float*>(ptr);         ptr += BM * 4;
    float* D_smem           = reinterpret_cast<float*>(ptr);

    // Load Q, dO, O
    for (int i = tid; i < BM * D; i += THREADS) {
        int row = i / D;
        int col = i % D;
        int gr = q_start + row;
        if (gr < S) {
            Q_smem[i]  = Q_bh[gr * D + col];
            dO_smem[i] = dO_bh[gr * D + col];
            O_smem[i]  = O_bh[gr * D + col];
        } else {
            Q_smem[i]  = __float2bfloat16(0.0f);
            dO_smem[i] = __float2bfloat16(0.0f);
            O_smem[i]  = __float2bfloat16(0.0f);
        }
    }
    for (int i = tid; i < BM; i += THREADS) {
        int gr = q_start + i;
        L_smem[i] = (gr < S) ? L_bh[gr] : 0.0f;
    }
    __syncthreads();

    // Compute D[i] = rowsum(dO[i] * O[i])
    for (int i = tid; i < BM; i += THREADS) {
        float sum = 0.0f;
        for (int d = 0; d < D; d++) {
            sum += __bfloat162float(dO_smem[i * D + d]) *
                   __bfloat162float(O_smem[i * D + d]);
        }
        D_smem[i] = sum;
    }
    __syncthreads();

    // Initialize dQ accumulator
    for (int i = tid; i < BM * D; i += THREADS) {
        dQ_acc[i] = 0.0f;
    }
    __syncthreads();

    int num_kv_blocks = (S + BN - 1) / BN;
    for (int kvb = 0; kvb < num_kv_blocks; kvb++) {
        int kv_start = kvb * BN;

        // Load K, V
        for (int i = tid; i < BN * D; i += THREADS) {
            int row = i / D;
            int col = i % D;
            int gr = kv_start + row;
            if (gr < S) {
                K_smem[i] = K_bh[gr * D + col];
                V_smem[i] = V_bh[gr * D + col];
            } else {
                K_smem[i] = __float2bfloat16(0.0f);
                V_smem[i] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        // === S[BM,BN] = Q @ K^T via wmma (64x64, contraction 128) ===
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> cF[2][2];

            for (int tr = 0; tr < 2; tr++)
                for (int tc = 0; tc < 2; tc++)
                    wmma::fill_fragment(cF[tr][tc], 0.0f);

            for (int k = 0; k < D; k += 16) {
                for (int tr = 0; tr < 2; tr++) {
                    int tr_row = warp_row * 32 + tr * 16;
                    wmma::load_matrix_sync(a_frag, &Q_smem[tr_row * D + k], D);
                    for (int tc = 0; tc < 2; tc++) {
                        int tc_col = warp_col * 32 + tc * 16;
                        wmma::load_matrix_sync(b_frag, &K_smem[tc_col * D + k], D);
                        wmma::mma_sync(cF[tr][tc], a_frag, b_frag, cF[tr][tc]);
                    }
                }
            }
            for (int tr = 0; tr < 2; tr++) {
                int tr_row = warp_row * 32 + tr * 16;
                for (int tc = 0; tc < 2; tc++) {
                    int tc_col = warp_col * 32 + tc * 16;
                    wmma::store_matrix_sync(&temp_smem[tr_row * BN + tc_col],
                                            cF[tr][tc], BN, wmma::mem_row_major);
                }
            }
        }
        __syncthreads();

        // === P[i,j] = exp(S[i,j]*scale - L[i]) ===
        for (int i = tid; i < BM * BN; i += THREADS) {
            int row = i / BN;
            int col = i % BN;
            int gcol = kv_start + col;
            if (gcol < S) {
                P_smem[i] = expf(temp_smem[i] * scale - L_smem[row]);
            } else {
                P_smem[i] = 0.0f;
            }
        }
        __syncthreads();

        // === dP[BM,BN] = dO @ V^T via wmma ===
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> cF[2][2];

            for (int tr = 0; tr < 2; tr++)
                for (int tc = 0; tc < 2; tc++)
                    wmma::fill_fragment(cF[tr][tc], 0.0f);

            for (int k = 0; k < D; k += 16) {
                for (int tr = 0; tr < 2; tr++) {
                    int tr_row = warp_row * 32 + tr * 16;
                    wmma::load_matrix_sync(a_frag, &dO_smem[tr_row * D + k], D);
                    for (int tc = 0; tc < 2; tc++) {
                        int tc_col = warp_col * 32 + tc * 16;
                        wmma::load_matrix_sync(b_frag, &V_smem[tc_col * D + k], D);
                        wmma::mma_sync(cF[tr][tc], a_frag, b_frag, cF[tr][tc]);
                    }
                }
            }
            for (int tr = 0; tr < 2; tr++) {
                int tr_row = warp_row * 32 + tr * 16;
                for (int tc = 0; tc < 2; tc++) {
                    int tc_col = warp_col * 32 + tc * 16;
                    wmma::store_matrix_sync(&temp_smem[tr_row * BN + tc_col],
                                            cF[tr][tc], BN, wmma::mem_row_major);
                }
            }
        }
        __syncthreads();

        // === dS[i,j] = P[i,j] * (dP[i,j] - D[i]) ===
        // dP is in temp_smem; dS goes to dS_smem (shares with O_smem, no longer needed)
        for (int i = tid; i < BM * BN; i += THREADS) {
            int row = i / BN;
            int grow = q_start + row;
            if (grow < S) {
                dS_smem[i] = P_smem[i] * (temp_smem[i] - D_smem[row]);
            } else {
                dS_smem[i] = 0.0f;
            }
        }
        __syncthreads();

        // === dV[BN,D] += P^T[BN,BM] @ dO[BM,D] (CUDA cores, atomic add) ===
        {
            int rb = tid / 16;
            int cb = tid % 16;
            int j0 = rb * 4;
            int d0 = cb * 8;
            float acc[4][8];
            for (int jj = 0; jj < 4; jj++) for (int dd = 0; dd < 8; dd++) acc[jj][dd] = 0.0f;

            for (int i = 0; i < BM; i++) {
                float p[4];
                for (int jj = 0; jj < 4; jj++) p[jj] = P_smem[i * BN + j0 + jj];
                float dv[8];
                for (int dd = 0; dd < 8; dd++) dv[dd] = __bfloat162float(dO_smem[i * D + d0 + dd]);
                for (int jj = 0; jj < 4; jj++)
                    for (int dd = 0; dd < 8; dd++)
                        acc[jj][dd] += p[jj] * dv[dd];
            }
            for (int jj = 0; jj < 4; jj++) {
                int gr = kv_start + j0 + jj;
                if (gr >= S) continue;
                for (int dd = 0; dd < 8; dd++) {
                    atomicAdd(&dV_bh[gr * D + d0 + dd], __float2bfloat16(acc[jj][dd]));
                }
            }
        }

        // === dK[BN,D] += dS^T[BN,BM] @ Q[BM,D] * scale (CUDA cores, atomic add) ===
        {
            int rb = tid / 16;
            int cb = tid % 16;
            int j0 = rb * 4;
            int d0 = cb * 8;
            float acc[4][8];
            for (int jj = 0; jj < 4; jj++) for (int dd = 0; dd < 8; dd++) acc[jj][dd] = 0.0f;

            for (int i = 0; i < BM; i++) {
                float ds[4];
                for (int jj = 0; jj < 4; jj++) ds[jj] = dS_smem[i * BN + j0 + jj];
                float qv[8];
                for (int dd = 0; dd < 8; dd++) qv[dd] = __bfloat162float(Q_smem[i * D + d0 + dd]);
                for (int jj = 0; jj < 4; jj++)
                    for (int dd = 0; dd < 8; dd++)
                        acc[jj][dd] += ds[jj] * qv[dd];
            }
            for (int jj = 0; jj < 4; jj++) {
                int gr = kv_start + j0 + jj;
                if (gr >= S) continue;
                for (int dd = 0; dd < 8; dd++) {
                    atomicAdd(&dK_bh[gr * D + d0 + dd], __float2bfloat16(acc[jj][dd] * scale));
                }
            }
        }

        // === dQ[BM,D] += dS[BM,BN] @ K[BN,D] * scale (CUDA cores, local accum) ===
        {
            int rb = tid / 16;
            int cb = tid % 16;
            int i0 = rb * 4;
            int d0 = cb * 8;
            float acc[4][8];
            for (int ii = 0; ii < 4; ii++) for (int dd = 0; dd < 8; dd++) acc[ii][dd] = 0.0f;

            for (int j = 0; j < BN; j++) {
                float ds[4];
                for (int ii = 0; ii < 4; ii++) ds[ii] = dS_smem[(i0 + ii) * BN + j];
                float kv[8];
                for (int dd = 0; dd < 8; dd++) kv[dd] = __bfloat162float(K_smem[j * D + d0 + dd]);
                for (int ii = 0; ii < 4; ii++)
                    for (int dd = 0; dd < 8; dd++)
                        acc[ii][dd] += ds[ii] * kv[dd];
            }
            for (int ii = 0; ii < 4; ii++)
                for (int dd = 0; dd < 8; dd++)
                    dQ_acc[(i0 + ii) * D + d0 + dd] += acc[ii][dd] * scale;
        }

        __syncthreads();
    }

    // Write dQ to global
    __syncthreads();
    for (int i = tid; i < BM * D; i += THREADS) {
        int row = i / D;
        int col = i % D;
        int gr = q_start + row;
        if (gr < S) {
            dQ_bh[gr * D + col] = __float2bfloat16(dQ_acc[i]);
        }
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

    // Zero dK, dV (atomically accumulated)
    size_t total_elems = (size_t)B * H * S * 128;
    CUDA_CHECK(cudaMemsetAsync(dK_data, 0, total_elems * sizeof(__nv_bfloat16), stream));
    CUDA_CHECK(cudaMemsetAsync(dV_data, 0, total_elems * sizeof(__nv_bfloat16), stream));

    int num_q_blocks = (S + BM - 1) / BM;
    int grid_size = B * H * num_q_blocks;
    int smem_size = SMEM_SIZE;

    CUDA_CHECK(cudaFuncSetAttribute(
        attn_bwd_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    attn_bwd_kernel<<<grid_size, THREADS, smem_size, stream>>>(
        Q_data, K_data, V_data, O_data, dO_data, L_data,
        dQ_data, dK_data, dV_data, B, H, S);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, attn_bwd::run);

}  // namespace attn_bwd