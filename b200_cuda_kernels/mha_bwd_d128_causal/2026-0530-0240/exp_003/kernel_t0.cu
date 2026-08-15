#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <mma.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>
#include <math.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

using namespace nvcuda;

namespace mha_bwd_d128_causal {

__global__ void precompute_D(const __nv_bfloat16* O, const __nv_bfloat16* dO, float* D, int B, int H, int S, int d) {
    int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int head_idx = blockIdx.y;
    int batch_idx = blockIdx.z;
    if (seq_idx < S) {
        float sum = 0;
        int base = ((batch_idx * H + head_idx) * S + seq_idx) * d;
        for (int i = 0; i < d; i++) {
            float o = __bfloat162float(O[base + i]);
            float do_ = __bfloat162float(dO[base + i]);
            sum += o * do_;
        }
        D[(batch_idx * H + head_idx) * S + seq_idx] = sum;
    }
}

__global__ void convert_dQ(const float* dQ_float, __nv_bfloat16* dQ, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        dQ[idx] = __float2bfloat16(dQ_float[idx]);
    }
}

__global__ void bwd_kernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* O, const __nv_bfloat16* dO, const float* LSE,
    const float* D,
    float* dQ_float, __nv_bfloat16* dK, __nv_bfloat16* dV,
    int B_size, int H, int S, int d) 
{
    int i_start = blockIdx.x * 64; // KV block start
    int head_idx = blockIdx.y;
    int batch_idx = blockIdx.z;
    
    if (i_start >= S) return;
    
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    
    int warp_row = (warp_id / 2) * 32;
    int warp_col = (warp_id % 2) * 64;
    
    // Allocate SMEM
    __shared__ alignas(16) __nv_bfloat16 K_i[64][128]; 
    __shared__ alignas(16) __nv_bfloat16 V_i[64][128]; 
    __shared__ alignas(16) __nv_bfloat16 Q_j[64][128]; 
    __shared__ alignas(16) __nv_bfloat16 dO_j[64][128]; 
    
    __shared__ float D_j[64]; 
    __shared__ float LSE_j[64]; 
    
    __shared__ alignas(16) float S_ij[64][64]; 
    __shared__ alignas(16) float dP_ij[64][64];
    __shared__ alignas(16) __nv_bfloat16 P_ij[64][64]; 
    __shared__ alignas(16) __nv_bfloat16 dS_ij[64][64]; 
    __shared__ alignas(16) float dQ_smem[64][128]; 
    
    // Load K_i, V_i
    int kv_base = ((batch_idx * H + head_idx) * S + i_start) * 128;
    for (int idx = tid; idx < 64 * 128; idx += 128) {
        int r = idx / 128;
        int c = idx % 128;
        if (i_start + r < S) {
            K_i[r][c] = K[kv_base + r * 128 + c];
            V_i[r][c] = V[kv_base + r * 128 + c];
        } else {
            K_i[r][c] = __float2bfloat16(0.0f);
            V_i[r][c] = __float2bfloat16(0.0f);
        }
    }
    
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_dV[2][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_dK[2][4];
    for (int r = 0; r < 2; r++) {
        for (int c = 0; c < 4; c++) {
            wmma::fill_fragment(acc_dV[r][c], 0.0f);
            wmma::fill_fragment(acc_dK[r][c], 0.0f);
        }
    }
    
    __syncthreads();
    
    // Causal mask dictates j >= i
    int j_start_align = (i_start / 64) * 64; 
    
    for (int j_start = j_start_align; j_start < S; j_start += 64) {
        
        // Load Q_j, dO_j
        int q_base = ((batch_idx * H + head_idx) * S + j_start) * 128;
        for (int idx = tid; idx < 64 * 128; idx += 128) {
            int r = idx / 128;
            int c = idx % 128;
            if (j_start + r < S) {
                Q_j[r][c] = Q[q_base + r * 128 + c];
                dO_j[r][c] = dO[q_base + r * 128 + c];
            } else {
                Q_j[r][c] = __float2bfloat16(0.0f);
                dO_j[r][c] = __float2bfloat16(0.0f);
            }
        }
        if (tid < 64) {
            if (j_start + tid < S) {
                int l_base = (batch_idx * H + head_idx) * S + j_start;
                D_j[tid] = D[l_base + tid];
                LSE_j[tid] = LSE[l_base + tid];
            } else {
                D_j[tid] = 0.0f;
                LSE_j[tid] = 0.0f;
            }
        }
        __syncthreads();
        
        // Compute S_ij = Q_j @ K_i^T
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_S[2][2];
        for (int r = 0; r < 2; r++) {
            for (int c = 0; c < 2; c++) {
                wmma::fill_fragment(acc_S[r][c], 0.0f);
            }
        }
        for (int k = 0; k < 128; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b[2];
            
            wmma::load_matrix_sync(a[0], &Q_j[warp_row + 0][k], 128);
            wmma::load_matrix_sync(a[1], &Q_j[warp_row + 16][k], 128);
            
            wmma::load_matrix_sync(b[0], &K_i[warp_col + 0][k], 128);
            wmma::load_matrix_sync(b[1], &K_i[warp_col + 16][k], 128);
            
            for (int r = 0; r < 2; r++) {
                for (int c = 0; c < 2; c++) {
                    wmma::mma_sync(acc_S[r][c], a[r], b[c], acc_S[r][c]);
                }
            }
        }
        for (int r = 0; r < 2; r++) {
            for (int c = 0; c < 2; c++) {
                wmma::store_matrix_sync(&S_ij[warp_row + r * 16][warp_col + c * 16], acc_S[r][c], 64, wmma::mem_row_major);
            }
        }
        __syncthreads();
        
        // Compute P_ij
        float attn_scale = 0.08838834764f; // 1.0 / sqrt(128)
        for (int idx = tid; idx < 64 * 64; idx += 128) {
            int row = idx / 64;
            int col = idx % 64;
            int global_q_idx = j_start + row;
            int global_k_idx = i_start + col;
            
            float val = S_ij[row][col] * attn_scale;
            // Causal mask application
            if (global_q_idx < global_k_idx || global_q_idx >= S || global_k_idx >= S) {
                val = -INFINITY;
            }
            float p = expf(val - LSE_j[row]);
            P_ij[row][col] = __float2bfloat16(p);
        }
        __syncthreads();
        
        // Compute dP_ij = dO_j @ V_i^T
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_dP[2][2];
        for (int r = 0; r < 2; r++) {
            for (int c = 0; c < 2; c++) {
                wmma::fill_fragment(acc_dP[r][c], 0.0f);
            }
        }
        for (int k = 0; k < 128; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b[2];
            
            wmma::load_matrix_sync(a[0], &dO_j[warp_row + 0][k], 128);
            wmma::load_matrix_sync(a[1], &dO_j[warp_row + 16][k], 128);
            
            wmma::load_matrix_sync(b[0], &V_i[warp_col + 0][k], 128);
            wmma::load_matrix_sync(b[1], &V_i[warp_col + 16][k], 128);
            
            for (int r = 0; r < 2; r++) {
                for (int c = 0; c < 2; c++) {
                    wmma::mma_sync(acc_dP[r][c], a[r], b[c], acc_dP[r][c]);
                }
            }
        }
        for (int r = 0; r < 2; r++) {
            for (int c = 0; c < 2; c++) {
                wmma::store_matrix_sync(&dP_ij[warp_row + r * 16][warp_col + c * 16], acc_dP[r][c], 64, wmma::mem_row_major);
            }
        }
        __syncthreads();
        
        // Compute dS_ij
        for (int idx = tid; idx < 64 * 64; idx += 128) {
            int row = idx / 64;
            int col = idx % 64;
            
            float p = __bfloat162float(P_ij[row][col]);
            float dp = dP_ij[row][col];
            float d = D_j[row];
            
            float ds = p * (dp - d) * attn_scale;
            
            int global_q_idx = j_start + row;
            int global_k_idx = i_start + col;
            if (global_q_idx < global_k_idx || global_q_idx >= S || global_k_idx >= S) {
                ds = 0.0f;
            }
            dS_ij[row][col] = __float2bfloat16(ds);
        }
        __syncthreads();
        
        // Compute dV_i += P_ij^T @ dO_j
        for (int k = 0; k < 64; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b[4];
            
            wmma::load_matrix_sync(a[0], &P_ij[k][warp_row + 0], 64);
            wmma::load_matrix_sync(a[1], &P_ij[k][warp_row + 16], 64);
            
            wmma::load_matrix_sync(b[0], &dO_j[k][warp_col + 0], 128);
            wmma::load_matrix_sync(b[1], &dO_j[k][warp_col + 16], 128);
            wmma::load_matrix_sync(b[2], &dO_j[k][warp_col + 32], 128);
            wmma::load_matrix_sync(b[3], &dO_j[k][warp_col + 48], 128);
            
            for (int r = 0; r < 2; r++) {
                for (int c = 0; c < 4; c++) {
                    wmma::mma_sync(acc_dV[r][c], a[r], b[c], acc_dV[r][c]);
                }
            }
        }
        
        // Compute dK_i += dS_ij^T @ Q_j
        for (int k = 0; k < 64; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b[4];
            
            wmma::load_matrix_sync(a[0], &dS_ij[k][warp_row + 0], 64);
            wmma::load_matrix_sync(a[1], &dS_ij[k][warp_row + 16], 64);
            
            wmma::load_matrix_sync(b[0], &Q_j[k][warp_col + 0], 128);
            wmma::load_matrix_sync(b[1], &Q_j[k][warp_col + 16], 128);
            wmma::load_matrix_sync(b[2], &Q_j[k][warp_col + 32], 128);
            wmma::load_matrix_sync(b[3], &Q_j[k][warp_col + 48], 128);
            
            for (int r = 0; r < 2; r++) {
                for (int c = 0; c < 4; c++) {
                    wmma::mma_sync(acc_dK[r][c], a[r], b[c], acc_dK[r][c]);
                }
            }
        }
        
        // Compute dQ_j_partial = dS_ij @ K_i
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_dQ[2][4];
        for (int r = 0; r < 2; r++) {
            for (int c = 0; c < 4; c++) {
                wmma::fill_fragment(acc_dQ[r][c], 0.0f);
            }
        }
        for (int k = 0; k < 64; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b[4];
            
            wmma::load_matrix_sync(a[0], &dS_ij[warp_row + 0][k], 64);
            wmma::load_matrix_sync(a[1], &dS_ij[warp_row + 16][k], 64);
            
            wmma::load_matrix_sync(b[0], &K_i[k][warp_col + 0], 128);
            wmma::load_matrix_sync(b[1], &K_i[k][warp_col + 16], 128);
            wmma::load_matrix_sync(b[2], &K_i[k][warp_col + 32], 128);
            wmma::load_matrix_sync(b[3], &K_i[k][warp_col + 48], 128);
            
            for (int r = 0; r < 2; r++) {
                for (int c = 0; c < 4; c++) {
                    wmma::mma_sync(acc_dQ[r][c], a[r], b[c], acc_dQ[r][c]);
                }
            }
        }
        
        for (int r = 0; r < 2; r++) {
            for (int c = 0; c < 4; c++) {
                wmma::store_matrix_sync(&dQ_smem[warp_row + r * 16][warp_col + c * 16], acc_dQ[r][c], 128, wmma::mem_row_major);
            }
        }
        __syncthreads();
        
        // Write dQ_j_partial out to a global float buffer via atomic add to avoid precision loss
        for (int idx = tid; idx < 64 * 128; idx += 128) {
            int row = idx / 128;
            int col = idx % 128;
            int global_q_idx = j_start + row;
            if (global_q_idx < S) {
                float val = dQ_smem[row][col];
                if (val != 0.0f) {
                    int offset = ((batch_idx * H + head_idx) * S + global_q_idx) * 128 + col;
                    atomicAdd(&dQ_float[offset], val);
                }
            }
        }
        __syncthreads();
    }
    
    // Store fully accumulated dK_i and dV_i out to global memory
    __shared__ alignas(16) float dK_smem[64][128];
    __shared__ alignas(16) float dV_smem[64][128];
    
    for (int r = 0; r < 2; r++) {
        for (int c = 0; c < 4; c++) {
            wmma::store_matrix_sync(&dK_smem[warp_row + r * 16][warp_col + c * 16], acc_dK[r][c], 128, wmma::mem_row_major);
            wmma::store_matrix_sync(&dV_smem[warp_row + r * 16][warp_col + c * 16], acc_dV[r][c], 128, wmma::mem_row_major);
        }
    }
    __syncthreads();
    
    for (int idx = tid; idx < 64 * 128; idx += 128) {
        int row = idx / 128;
        int col = idx % 128;
        int global_k_idx = i_start + row;
        if (global_k_idx < S) {
            int offset = ((batch_idx * H + head_idx) * S + global_k_idx) * 128 + col;
            dK[offset] = __float2bfloat16(dK_smem[row][col]);
            dV[offset] = __float2bfloat16(dV_smem[row][col]);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) 
{
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int d = Q.size(3);

    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_ptr = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_ptr = static_cast<const float*>(L.data_ptr());

    __nv_bfloat16* dQ_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());

    // Allocate workspace for precomputed D
    float* D_ptr;
    CUDA_CHECK(cudaMallocAsync(&D_ptr, B * H * S * sizeof(float), stream));

    // Allocate workspace for high-precision float dQ accumulation to avoid atomic bf16 loss of precision
    float* dQ_float_ptr;
    CUDA_CHECK(cudaMallocAsync(&dQ_float_ptr, B * H * S * d * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(dQ_float_ptr, 0, B * H * S * d * sizeof(float), stream));

    // Launch D precompute
    dim3 grid_D((S + 127) / 128, H, B);
    dim3 block_D(128);
    precompute_D<<<grid_D, block_D, 0, stream>>>(O_ptr, dO_ptr, D_ptr, B, H, S, d);

    // Launch main backward kernel
    dim3 grid_Bwd((S + 63) / 64, H, B);
    dim3 block_Bwd(128);
    bwd_kernel<<<grid_Bwd, block_Bwd, 0, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, D_ptr, 
        dQ_float_ptr, dK_ptr, dV_ptr, B, H, S, d
    );

    // Convert dQ from float workspace to bfloat16 output
    int count = B * H * S * d;
    dim3 grid_Cvt((count + 255) / 256);
    dim3 block_Cvt(256);
    convert_dQ<<<grid_Cvt, block_Cvt, 0, stream>>>(dQ_float_ptr, dQ_ptr, count);

    // Free workspaces
    CUDA_CHECK(cudaFreeAsync(D_ptr, stream));
    CUDA_CHECK(cudaFreeAsync(dQ_float_ptr, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128_causal::run);

}  // namespace mha_bwd_d128_causal