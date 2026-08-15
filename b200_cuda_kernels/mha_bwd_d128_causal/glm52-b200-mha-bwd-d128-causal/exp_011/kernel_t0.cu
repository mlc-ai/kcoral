#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
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

namespace attn_bwd {

using bf16 = __nv_bfloat16;

constexpr int D = 128;
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int THREADS = 256;
constexpr float SCALE = 0.08838834764831845f; // 1/sqrt(128)

struct SmemBuf {
    bf16 K[BN][D];       // 16384
    bf16 V[BN][D];       // 16384
    float dK[BN][D];     // 32768
    float dV[BN][D];     // 32768
    bf16 Q[BM][D];       // 16384
    bf16 dO[BM][D];      // 16384
    float P[BM][BN];     // 16384
    float dS[BM][BN];    // 16384
    float L[BM];         // 256
    float Dv[BM];        // 256
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

__global__ void zero_kernel(float* ptr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) ptr[idx] = 0.0f;
}

__global__ void convert_kernel(const float* src, bf16* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __float2bfloat16(src[idx]);
}

__global__ void attn_backward_kernel(
    const bf16* __restrict__ Q,
    const bf16* __restrict__ K,
    const bf16* __restrict__ V,
    const bf16* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ D_val,
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

    // Load K and V tiles
    {
        int elems = actual_BN * D;
        for (int i = tid; i < elems; i += THREADS) {
            int row = i / D;
            int col = i % D;
            int gidx = (bh * S + j_start + row) * D + col;
            smem->K[row][col] = K[gidx];
            smem->V[row][col] = V[gidx];
        }
    }

    // Initialize dK, dV to 0
    for (int i = tid; i < BN * D; i += THREADS) {
        smem->dK[i / D][i % D] = 0.0f;
        smem->dV[i / D][i % D] = 0.0f;
    }

    __syncthreads();

    int num_q_blocks = (S + BM - 1) / BM;

    for (int i_block = j_block; i_block < num_q_blocks; i_block++) {
        int i_start = i_block * BM;
        int i_end = min(i_start + BM, S);
        int actual_BM = i_end - i_start;

        // Load Q, dO tiles
        {
            int elems = actual_BM * D;
            for (int i = tid; i < elems; i += THREADS) {
                int row = i / D;
                int col = i % D;
                int gidx = (bh * S + i_start + row) * D + col;
                smem->Q[row][col] = Q[gidx];
                smem->dO[row][col] = dO[gidx];
            }
        }

        // Load L, D values
        for (int i = tid; i < actual_BM; i += THREADS) {
            smem->L[i] = L[bh * S + i_start + i];
            smem->Dv[i] = D_val[bh * S + i_start + i];
        }

        __syncthreads();

        // Step 1: S = Q @ K^T, P = exp(S*SCALE - L) with causal mask
        {
            int rb = tid / 16;
            int cb = tid % 16;
            int row_base = rb * 4;
            int col_base = cb * 4;
            float s[4][4];
            #pragma unroll
            for (int ii = 0; ii < 4; ii++)
                #pragma unroll
                for (int jj = 0; jj < 4; jj++)
                    s[ii][jj] = 0.0f;

            #pragma unroll 8
            for (int k = 0; k < D; k++) {
                float q[4], kv[4];
                #pragma unroll
                for (int ii = 0; ii < 4; ii++)
                    q[ii] = __bfloat162float(smem->Q[row_base + ii][k]);
                #pragma unroll
                for (int jj = 0; jj < 4; jj++)
                    kv[jj] = __bfloat162float(smem->K[col_base + jj][k]);
                #pragma unroll
                for (int ii = 0; ii < 4; ii++) {
                    float qi = q[ii];
                    #pragma unroll
                    for (int jj = 0; jj < 4; jj++)
                        s[ii][jj] += qi * kv[jj];
                }
            }

            for (int ii = 0; ii < 4; ii++) {
                int r = row_base + ii;
                if (r >= actual_BM) continue;
                float l = smem->L[r];
                for (int jj = 0; jj < 4; jj++) {
                    int c = col_base + jj;
                    if (c >= actual_BN) continue;
                    if (j_start + c > i_start + r) {
                        smem->P[r][c] = 0.0f;
                    } else {
                        smem->P[r][c] = __expf(s[ii][jj] * SCALE - l);
                    }
                }
            }
        }

        __syncthreads();

        // Step 2: dP = dO @ V^T, dS = P * (dP - D) * SCALE
        {
            int rb = tid / 16;
            int cb = tid % 16;
            int row_base = rb * 4;
            int col_base = cb * 4;
            float dp[4][4];
            #pragma unroll
            for (int ii = 0; ii < 4; ii++)
                #pragma unroll
                for (int jj = 0; jj < 4; jj++)
                    dp[ii][jj] = 0.0f;

            #pragma unroll 8
            for (int k = 0; k < D; k++) {
                float q[4], kv[4];
                #pragma unroll
                for (int ii = 0; ii < 4; ii++)
                    q[ii] = __bfloat162float(smem->dO[row_base + ii][k]);
                #pragma unroll
                for (int jj = 0; jj < 4; jj++)
                    kv[jj] = __bfloat162float(smem->V[col_base + jj][k]);
                #pragma unroll
                for (int ii = 0; ii < 4; ii++) {
                    float qi = q[ii];
                    #pragma unroll
                    for (int jj = 0; jj < 4; jj++)
                        dp[ii][jj] += qi * kv[jj];
                }
            }

            for (int ii = 0; ii < 4; ii++) {
                int r = row_base + ii;
                if (r >= actual_BM) continue;
                float dv = smem->Dv[r];
                for (int jj = 0; jj < 4; jj++) {
                    int c = col_base + jj;
                    if (c >= actual_BN) continue;
                    smem->dS[r][c] = smem->P[r][c] * (dp[ii][jj] - dv) * SCALE;
                }
            }
        }

        __syncthreads();

        // Step 3: dQ += dS @ K  [BM, D] += [BM, BN] @ [BN, D]
        {
            int row = tid / 4;
            int col_base = (tid % 4) * 32;
            if (row < actual_BM) {
                float dq[32];
                #pragma unroll
                for (int i = 0; i < 32; i++) dq[i] = 0.0f;

                for (int kk = 0; kk < actual_BN; kk++) {
                    float ds_val = smem->dS[row][kk];
                    #pragma unroll
                    for (int c = 0; c < 32; c++) {
                        dq[c] += ds_val * __bfloat162float(smem->K[kk][col_base + c]);
                    }
                }

                float* dQ_ptr = dQ_buf + (bh * S + i_start + row) * D + col_base;
                #pragma unroll
                for (int c = 0; c < 32; c++) {
                    atomicAdd(&dQ_ptr[c], dq[c]);
                }
            }
        }

        __syncthreads();

        // Step 4: dK += dS^T @ Q  [BN, D] += [BN, BM] @ [BM, D]
        {
            int row = tid / 4;
            int col_base = (tid % 4) * 32;
            if (row < actual_BN) {
                float dk[32];
                #pragma unroll
                for (int i = 0; i < 32; i++) dk[i] = 0.0f;

                for (int kk = 0; kk < actual_BM; kk++) {
                    float ds_val = smem->dS[kk][row];
                    #pragma unroll
                    for (int c = 0; c < 32; c++) {
                        dk[c] += ds_val * __bfloat162float(smem->Q[kk][col_base + c]);
                    }
                }

                #pragma unroll
                for (int c = 0; c < 32; c++) {
                    smem->dK[row][col_base + c] += dk[c];
                }
            }
        }

        __syncthreads();

        // Step 5: dV += P^T @ dO  [BN, D] += [BN, BM] @ [BM, D]
        {
            int row = tid / 4;
            int col_base = (tid % 4) * 32;
            if (row < actual_BN) {
                float dv[32];
                #pragma unroll
                for (int i = 0; i < 32; i++) dv[i] = 0.0f;

                for (int kk = 0; kk < actual_BM; kk++) {
                    float p_val = smem->P[kk][row];
                    #pragma unroll
                    for (int c = 0; c < 32; c++) {
                        dv[c] += p_val * __bfloat162float(smem->dO[kk][col_base + c]);
                    }
                }

                #pragma unroll
                for (int c = 0; c < 32; c++) {
                    smem->dV[row][col_base + c] += dv[c];
                }
            }
        }

        __syncthreads();
    }

    // Store dK and dV to global memory
    {
        int elems = actual_BN * D;
        for (int i = tid; i < elems; i += THREADS) {
            int row = i / D;
            int col = i % D;
            int gidx = (bh * S + j_start + row) * D + col;
            dK_out[gidx] = __float2bfloat16(smem->dK[row][col]);
            dV_out[gidx] = __float2bfloat16(smem->dV[row][col]);
        }
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

    // Allocate D_val (rowsum of dO * O)
    float* D_val;
    CUDA_CHECK(cudaMalloc(&D_val, (size_t)total_rows * sizeof(float)));

    // Allocate dQ_buf (float for precision)
    int64_t dQ_size = (int64_t)B * H * S * D;
    float* dQ_buf;
    CUDA_CHECK(cudaMalloc(&dQ_buf, (size_t)dQ_size * sizeof(float)));

    // Compute D = rowsum(dO * O)
    {
        int threads = 256;
        int blocks = (total_rows + threads - 1) / threads;
        compute_D_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const bf16*>(O.data_ptr()),
            static_cast<const bf16*>(dO.data_ptr()),
            D_val, total_rows);
    }

    // Zero dQ_buf
    {
        int threads = 256;
        int blocks = ((int)dQ_size + threads - 1) / threads;
        zero_kernel<<<blocks, threads, 0, stream>>>(dQ_buf, (int)dQ_size);
    }

    // Backward kernel
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

    // Convert dQ_buf (float) to dQ (bf16)
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