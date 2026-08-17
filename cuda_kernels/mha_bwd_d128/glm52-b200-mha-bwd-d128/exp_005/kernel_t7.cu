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
constexpr int D_PAD = 136;
constexpr int P_PAD = 72;
constexpr int T_PAD = 72;
constexpr int D_ACC_PAD = 136;
constexpr int THREADS = 128;
constexpr int SMEM_SIZE = 17408*4 + 18432 + 9216 + 34816*2 + 512;

__device__ __forceinline__ float fast_expf(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x * 1.4426950408889634f));
    return y;
}

__global__ void precompute_D_kernel(
    const __nv_bfloat16* __restrict__ dO,
    const __nv_bfloat16* __restrict__ O,
    float* __restrict__ D_out,
    int total_rows)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < total_rows) {
        const __nv_bfloat16* dO_row = dO + (size_t)row * D;
        const __nv_bfloat16* O_row = O + (size_t)row * D;
        float sum = 0.0f;
        #pragma unroll
        for (int d = 0; d < D; d += 8) {
            int4 dO_v = *(const int4*)(dO_row + d);
            int4 O_v = *(const int4*)(O_row + d);
            __nv_bfloat16* dO_h = (__nv_bfloat16*)&dO_v;
            __nv_bfloat16* O_h = (__nv_bfloat16*)&O_v;
            #pragma unroll
            for (int j = 0; j < 8; j++)
                sum += __bfloat162float(dO_h[j]) * __bfloat162float(O_h[j]);
        }
        D_out[row] = sum;
    }
}

__global__ __launch_bounds__(THREADS, 1)
void fused_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ D_pre,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    float* __restrict__ dQ_workspace,
    int B, int H, int S)
{
    int bh = blockIdx.x;
    int kv_block = blockIdx.y;
    int h = bh % H;
    int b = bh / H;
    int kv_start = kv_block * BN;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int wr = warp_id / 2;
    int wc = warp_id % 2;

    const float scale = 1.0f / sqrtf((float)D);
    size_t bh_off = ((size_t)b * H + h) * S * D;
    size_t stat_off = ((size_t)b * H + h) * S;

    const __nv_bfloat16* Q_bh = Q + bh_off;
    const __nv_bfloat16* K_bh = K + bh_off;
    const __nv_bfloat16* V_bh = V + bh_off;
    const __nv_bfloat16* dO_bh = dO + bh_off;
    const float* L_bh = L + stat_off;
    const float* D_bh = D_pre + stat_off;
    __nv_bfloat16* dK_bh = dK + bh_off;
    __nv_bfloat16* dV_bh = dV + bh_off;
    float* dQ_bh = dQ_workspace + bh_off;

    extern __shared__ char smem[];
    char* ptr = smem;
    __nv_bfloat16* K_smem  = (__nv_bfloat16*)ptr;  ptr += BN * D_PAD * 2;
    __nv_bfloat16* V_smem  = (__nv_bfloat16*)ptr;  ptr += BN * D_PAD * 2;
    __nv_bfloat16* Q_smem  = (__nv_bfloat16*)ptr;  ptr += BM * D_PAD * 2;
    __nv_bfloat16* dO_smem = (__nv_bfloat16*)ptr;  ptr += BM * D_PAD * 2;
    float* temp_smem       = (float*)ptr;           ptr += BM * T_PAD * 4;
    __nv_bfloat16* P_smem  = (__nv_bfloat16*)ptr;   ptr += BM * P_PAD * 2;
    float* dV_acc          = (float*)ptr;           ptr += BN * D_ACC_PAD * 4;
    float* dK_acc          = (float*)ptr;           ptr += BN * D_ACC_PAD * 4;
    float* L_smem          = (float*)ptr;           ptr += BM * 4;
    float* D_smem          = (float*)ptr;

    for (int i = tid; i < BN * D_ACC_PAD; i += THREADS) {
        dV_acc[i] = 0.0f;
        dK_acc[i] = 0.0f;
    }

    for (int i = tid; i < BN * D / 8; i += THREADS) {
        int row = (i * 8) / D, col = (i * 8) % D;
        int gr = kv_start + row;
        if (gr < S) {
            *(int4*)(&K_smem[row * D_PAD + col]) = *(int4*)(&K_bh[gr*D + col]);
            *(int4*)(&V_smem[row * D_PAD + col]) = *(int4*)(&V_bh[gr*D + col]);
        } else {
            *(int4*)(&K_smem[row * D_PAD + col]) = make_int4(0,0,0,0);
            *(int4*)(&V_smem[row * D_PAD + col]) = make_int4(0,0,0,0);
        }
    }
    __syncthreads();

    int num_q = (S + BM - 1) / BM;
    for (int qb = 0; qb < num_q; qb++) {
        int q_start = qb * BM;

        for (int i = tid; i < BM * D / 8; i += THREADS) {
            int row = (i * 8) / D, col = (i * 8) % D;
            int gr = q_start + row;
            if (gr < S) {
                *(int4*)(&Q_smem[row * D_PAD + col])  = *(int4*)(&Q_bh[gr*D + col]);
                *(int4*)(&dO_smem[row * D_PAD + col]) = *(int4*)(&dO_bh[gr*D + col]);
            } else {
                *(int4*)(&Q_smem[row * D_PAD + col])  = make_int4(0,0,0,0);
                *(int4*)(&dO_smem[row * D_PAD + col]) = make_int4(0,0,0,0);
            }
        }
        for (int i = tid; i < BM; i += THREADS) {
            int gr = q_start + i;
            L_smem[i] = (gr < S) ? L_bh[gr] : 0.0f;
            D_smem[i] = (gr < S) ? D_bh[gr] : 0.0f;
        }
        __syncthreads();

        // S = Q @ K^T
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> cF[2][2];
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                #pragma unroll
                for (int j = 0; j < 2; j++) wmma::fill_fragment(cF[i][j], 0.0f);
            }
            #pragma unroll
            for (int k = 0; k < D; k += 16) {
                #pragma unroll
                for (int i = 0; i < 2; i++) {
                    wmma::load_matrix_sync(a, &Q_smem[(wr*32+i*16)*D_PAD + k], D_PAD);
                    #pragma unroll
                    for (int j = 0; j < 2; j++) {
                        wmma::load_matrix_sync(b, &K_smem[(wc*32+j*16)*D_PAD + k], D_PAD);
                        wmma::mma_sync(cF[i][j], a, b, cF[i][j]);
                    }
                }
            }
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                #pragma unroll
                for (int j = 0; j < 2; j++)
                    wmma::store_matrix_sync(&temp_smem[(wr*32+i*16)*T_PAD + wc*32+j*16], cF[i][j], T_PAD, wmma::mem_row_major);
            }
        }
        __syncthreads();

        for (int i = tid; i < BM * BN; i += THREADS) {
            int row = i / BN, col = i % BN;
            if (q_start+row < S && kv_start+col < S)
                P_smem[row * P_PAD + col] = __float2bfloat16(fast_expf(temp_smem[row * T_PAD + col] * scale - L_smem[row]));
            else
                P_smem[row * P_PAD + col] = __float2bfloat16(0.0f);
        }
        __syncthreads();

        // dP = dO @ V^T
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> cF[2][2];
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                #pragma unroll
                for (int j = 0; j < 2; j++) wmma::fill_fragment(cF[i][j], 0.0f);
            }
            #pragma unroll
            for (int k = 0; k < D; k += 16) {
                #pragma unroll
                for (int i = 0; i < 2; i++) {
                    wmma::load_matrix_sync(a, &dO_smem[(wr*32+i*16)*D_PAD + k], D_PAD);
                    #pragma unroll
                    for (int j = 0; j < 2; j++) {
                        wmma::load_matrix_sync(b, &V_smem[(wc*32+j*16)*D_PAD + k], D_PAD);
                        wmma::mma_sync(cF[i][j], a, b, cF[i][j]);
                    }
                }
            }
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                #pragma unroll
                for (int j = 0; j < 2; j++)
                    wmma::store_matrix_sync(&temp_smem[(wr*32+i*16)*T_PAD + wc*32+j*16], cF[i][j], T_PAD, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // dS = P * (dP - D) * scale
        for (int i = tid; i < BM * BN; i += THREADS) {
            int row = i / BN, col = i % BN;
            float p = __bfloat162float(P_smem[row * P_PAD + col]);
            float dp = temp_smem[row * T_PAD + col];
            P_smem[row * P_PAD + col] = __float2bfloat16(p * (dp - D_smem[row]) * scale);
        }
        __syncthreads();

        // dV_acc += P^T @ dO (2 col passes)
        for (int col_pass = 0; col_pass < 2; col_pass++) {
            int col_offset = col_pass * 64;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> cF[2][2];
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                #pragma unroll
                for (int j = 0; j < 2; j++) wmma::fill_fragment(cF[i][j], 0.0f);
            }
            #pragma unroll
            for (int k = 0; k < BM; k += 16) {
                #pragma unroll
                for (int i = 0; i < 2; i++) {
                    wmma::load_matrix_sync(a, &P_smem[k*P_PAD + wr*32+i*16], P_PAD);
                    #pragma unroll
                    for (int j = 0; j < 2; j++) {
                        wmma::load_matrix_sync(b, &dO_smem[k*D_PAD + col_offset + wc*32+j*16], D_PAD);
                        wmma::mma_sync(cF[i][j], a, b, cF[i][j]);
                    }
                }
            }
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                    wmma::load_matrix_sync(acc[i][j], &dV_acc[(wr*32+i*16)*D_ACC_PAD + col_offset + wc*32+j*16], D_ACC_PAD, wmma::mem_row_major);
                    #pragma unroll
                    for (int e = 0; e < cF[i][j].num_elements; e++)
                        acc[i][j].x[e] += cF[i][j].x[e];
                    wmma::store_matrix_sync(&dV_acc[(wr*32+i*16)*D_ACC_PAD + col_offset + wc*32+j*16], acc[i][j], D_ACC_PAD, wmma::mem_row_major);
                }
            }
        }
        __syncthreads();

        // dK_acc += dS^T @ Q (2 col passes)
        for (int col_pass = 0; col_pass < 2; col_pass++) {
            int col_offset = col_pass * 64;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> cF[2][2];
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                #pragma unroll
                for (int j = 0; j < 2; j++) wmma::fill_fragment(cF[i][j], 0.0f);
            }
            #pragma unroll
            for (int k = 0; k < BM; k += 16) {
                #pragma unroll
                for (int i = 0; i < 2; i++) {
                    wmma::load_matrix_sync(a, &P_smem[k*P_PAD + wr*32+i*16], P_PAD);
                    #pragma unroll
                    for (int j = 0; j < 2; j++) {
                        wmma::load_matrix_sync(b, &Q_smem[k*D_PAD + col_offset + wc*32+j*16], D_PAD);
                        wmma::mma_sync(cF[i][j], a, b, cF[i][j]);
                    }
                }
            }
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                    wmma::load_matrix_sync(acc[i][j], &dK_acc[(wr*32+i*16)*D_ACC_PAD + col_offset + wc*32+j*16], D_ACC_PAD, wmma::mem_row_major);
                    #pragma unroll
                    for (int e = 0; e < cF[i][j].num_elements; e++)
                        acc[i][j].x[e] += cF[i][j].x[e];
                    wmma::store_matrix_sync(&dK_acc[(wr*32+i*16)*D_ACC_PAD + col_offset + wc*32+j*16], acc[i][j], D_ACC_PAD, wmma::mem_row_major);
                }
            }
        }
        __syncthreads();

        // dQ += dS @ K (2 col passes, atomicAdd to global float)
        for (int col_pass = 0; col_pass < 2; col_pass++) {
            int col_offset = col_pass * 64;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> cF[2][2];
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                #pragma unroll
                for (int j = 0; j < 2; j++) wmma::fill_fragment(cF[i][j], 0.0f);
            }
            #pragma unroll
            for (int k = 0; k < BN; k += 16) {
                #pragma unroll
                for (int i = 0; i < 2; i++) {
                    wmma::load_matrix_sync(a, &P_smem[(wr*32+i*16)*P_PAD + k], P_PAD);
                    #pragma unroll
                    for (int j = 0; j < 2; j++) {
                        wmma::load_matrix_sync(b, &K_smem[k*D_PAD + col_offset + wc*32+j*16], D_PAD);
                        wmma::mma_sync(cF[i][j], a, b, cF[i][j]);
                    }
                }
            }
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                #pragma unroll
                for (int j = 0; j < 2; j++)
                    wmma::store_matrix_sync(&temp_smem[(wr*32+i*16)*T_PAD + wc*32+j*16], cF[i][j], T_PAD, wmma::mem_row_major);
            }
            __syncthreads();
            for (int i = tid; i < BM * 64; i += THREADS) {
                int row = i / 64, col = i % 64;
                int gr = q_start + row;
                if (gr < S) atomicAdd(&dQ_bh[gr*D + col_offset + col], temp_smem[row * T_PAD + col]);
            }
            __syncthreads();
        }
    }

    // Store dK, dV to global
    for (int i = tid; i < BN * D; i += THREADS) {
        int row = i / D, col = i % D;
        int gr = kv_start + row;
        if (gr < S) {
            dK_bh[gr*D + col] = __float2bfloat16(dK_acc[row * D_ACC_PAD + col]);
            dV_bh[gr*D + col] = __float2bfloat16(dV_acc[row * D_ACC_PAD + col]);
        }
    }
}

__global__ void convert_dQ_kernel(const float* __restrict__ dQ_float,
                                   __nv_bfloat16* __restrict__ dQ, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dQ[idx] = __float2bfloat16(dQ_float[idx]);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = 4, H = 48, S = static_cast<int>(Q.size(2));

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

    int total_rows = B * H * S;
    float* D_pre = nullptr;
    CUDA_CHECK(cudaMalloc(&D_pre, total_rows * sizeof(float)));

    int d_threads = 256;
    int d_blocks = (total_rows + d_threads - 1) / d_threads;
    precompute_D_kernel<<<d_blocks, d_threads, 0, stream>>>(
        dO_data, O_data, D_pre, total_rows);

    size_t total_elems = (size_t)B * H * S * D;
    float* dQ_float = nullptr;
    CUDA_CHECK(cudaMalloc(&dQ_float, total_elems * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(dQ_float, 0, total_elems * sizeof(float), stream));

    CUDA_CHECK(cudaFuncSetAttribute(fused_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));

    int num_kv = (S + BN - 1) / BN;
    dim3 grid(B * H, num_kv);
    fused_bwd_kernel<<<grid, THREADS, SMEM_SIZE, stream>>>(
        Q_data, K_data, V_data, dO_data, L_data, D_pre,
        dK_data, dV_data, dQ_float, B, H, S);

    CUDA_CHECK(cudaGetLastError());

    int convert_threads = 256;
    int convert_blocks = (total_elems + convert_threads - 1) / convert_threads;
    convert_dQ_kernel<<<convert_blocks, convert_threads, 0, stream>>>(
        dQ_float, dQ_data, total_elems);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(dQ_float));
    CUDA_CHECK(cudaFree(D_pre));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, attn_bwd::run);

}  // namespace attn_bwd