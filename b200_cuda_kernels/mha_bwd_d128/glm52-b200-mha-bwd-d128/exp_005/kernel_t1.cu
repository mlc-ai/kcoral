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

constexpr int SMEM_SIZE = 32768*2 + 16384*6 + 16384*2 + 256*2;

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
    int warp_row = warp_id / 2;
    int warp_col = warp_id % 2;

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
    __nv_bfloat16* O_smem   = reinterpret_cast<__nv_bfloat16*>(ptr);
    float* dS_smem          = reinterpret_cast<float*>(ptr);        ptr += BM * D * 2;
    float* temp_smem        = reinterpret_cast<float*>(ptr);        ptr += BM * BN * 4;
    float* P_smem           = reinterpret_cast<float*>(ptr);        ptr += BM * BN * 4;
    float* L_smem           = reinterpret_cast<float*>(ptr);        ptr += BM * 4;
    float* D_smem           = reinterpret_cast<float*>(ptr);

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

    for (int i = tid; i < BN * D; i += THREADS) {
        dK_acc[i] = 0.0f;
        dV_acc[i] = 0.0f;
    }
    __syncthreads();

    int num_q_blocks = (S + BM - 1) / BM;
    for (int qb = 0; qb < num_q_blocks; qb++) {
        int q_start = qb * BM;

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

        for (int i = tid; i < BM; i += THREADS) {
            float sum = 0.0f;
            for (int d = 0; d < D; d++) {
                sum += __bfloat162float(dO_smem[i * D + d]) *
                       __bfloat162float(O_smem[i * D + d]);
            }
            D_smem[i] = sum;
        }
        __syncthreads();

        // S = Q @ K^T
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

        for (int i = tid; i < BM * BN; i += THREADS) {
            int row = i / BN;
            int col = i % BN;
            int gcol = kv_start + col;
            if (gcol < S && (q_start + row) < S) {
                P_smem[i] = expf(temp_smem[i] * scale - L_smem[row]);
            } else {
                P_smem[i] = 0.0f;
            }
        }
        __syncthreads();

        // dP = dO @ V^T
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

        for (int i = tid; i < BM * BN; i += THREADS) {
            int row = i / BN;
            dS_smem[i] = P_smem[i] * (temp_smem[i] - D_smem[row]);
        }
        __syncthreads();

        // dV_acc[BN,D] += P^T[BN,BM] @ dO[BM,D]
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
            for (int jj = 0; jj < 4; jj++)
                for (int dd = 0; dd < 8; dd++)
                    dV_acc[(j0 + jj) * D + d0 + dd] += acc[jj][dd];
        }

        // dK_acc[BN,D] += dS^T[BN,BM] @ Q[BM,D] * scale
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
            for (int jj = 0; jj < 4; jj++)
                for (int dd = 0; dd < 8; dd++)
                    dK_acc[(j0 + jj) * D + d0 + dd] += acc[jj][dd] * scale;
        }

        // dQ += dS @ K * scale (float atomicAdd)
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
            for (int ii = 0; ii < 4; ii++) {
                int gr = q_start + i0 + ii;
                if (gr >= S) continue;
                for (int dd = 0; dd < 8; dd++) {
                    atomicAdd(&dQ_bh[gr * D + d0 + dd], acc[ii][dd] * scale);
                }
            }
        }

        __syncthreads();
    }

    for (int i = tid; i < BN * D; i += THREADS) {
        int row = i / D;
        int col = i % D;
        int gr = kv_start + row;
        if (gr < S) {
            dK_bh[gr * D + col] = __float2bfloat16(dK_acc[i]);
            dV_bh[gr * D + col] = __float2bfloat16(dV_acc[i]);
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