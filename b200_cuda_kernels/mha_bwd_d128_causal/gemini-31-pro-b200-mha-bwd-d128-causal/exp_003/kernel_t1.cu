#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_example_cuda {

// Kernel 1: computes dQ (outer loop i, inner loop j <= i)
__global__ void mha_bwd_causal_kernel_dQ(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    int B, int H, int S, int d) 
{
    int i = blockIdx.x;
    if (i * 64 >= S) return;
    int bh = blockIdx.y;
    int tid = threadIdx.x;

    extern __shared__ char smem[];
    __nv_bfloat16 (*s_Q)[138]  = reinterpret_cast<__nv_bfloat16 (*)[138]>(smem);
    __nv_bfloat16 (*s_O)[138]  = reinterpret_cast<__nv_bfloat16 (*)[138]>(smem + 17664);
    __nv_bfloat16 (*s_dO)[138] = reinterpret_cast<__nv_bfloat16 (*)[138]>(smem + 35328);
    __nv_bfloat16 (*s_K)[138]  = reinterpret_cast<__nv_bfloat16 (*)[138]>(smem + 52992);
    __nv_bfloat16 (*s_V)[138]  = reinterpret_cast<__nv_bfloat16 (*)[138]>(smem + 70656);
    
    // Using 65 for padding to avoid bank conflicts
    float (*s_dS)[65] = reinterpret_cast<float (*)[65]>(smem + 104960);
    
    float *s_L = reinterpret_cast<float *>(smem + 121600);
    float *s_D = reinterpret_cast<float *>(smem + 121856);

    int i_offset_base = bh * S * 128 + i * 64 * 128;
    int base_idx = tid * 16; 

    // 1. Load Q_i, O_i, dO_i
    for(int k=0; k<16; k++) {
        int idx = base_idx + k;
        if (idx < 4096) {
            int r = idx >> 6;
            int c = (idx & 63) << 1;
            int global_r = i * 64 + r;
            if (global_r < S) {
                *(__nv_bfloat162*)&s_Q[r][c]  = *(__nv_bfloat162*)&Q[i_offset_base + r * 128 + c];
                *(__nv_bfloat162*)&s_O[r][c]  = *(__nv_bfloat162*)&O[i_offset_base + r * 128 + c];
                *(__nv_bfloat162*)&s_dO[r][c] = *(__nv_bfloat162*)&dO[i_offset_base + r * 128 + c];
            } else {
                *(__nv_bfloat162*)&s_Q[r][c]  = __floats2bfloat162_rn(0.0f, 0.0f);
                *(__nv_bfloat162*)&s_O[r][c]  = __floats2bfloat162_rn(0.0f, 0.0f);
                *(__nv_bfloat162*)&s_dO[r][c] = __floats2bfloat162_rn(0.0f, 0.0f);
            }
        }
    }
    
    if (tid < 64) {
        if (i * 64 + tid < S) s_L[tid] = L[bh * S + i * 64 + tid];
        else s_L[tid] = 0.0f;
    }
    __syncthreads();

    // Compute D_i = rowsum(dO_i * O_i)
    if (tid < 64) {
        float sum = 0;
        for(int k=0; k<128; k+=2) {
            float2 fdo = __bfloat1622float2(*(__nv_bfloat162*)&s_dO[tid][k]);
            float2 fo  = __bfloat1622float2(*(__nv_bfloat162*)&s_O[tid][k]);
            sum += fdo.x * fo.x + fdo.y * fo.y;
        }
        s_D[tid] = sum;
    }
    __syncthreads();

    int tx1 = tid % 16, ty1 = tid / 16;
    int tx2 = tid % 16, ty2 = tid / 16;
    int row_start = ty2 * 4;
    int col_start = tx2 * 8;
    
    // Register tile for accumulating dQ_i over all j <= i
    float dQ_tile[4][8] = {0};

    // Inner loop over K_j, V_j
    for (int j = 0; j <= i; j++) {
        int j_offset_base = bh * S * 128 + j * 64 * 128;
        
        for(int k=0; k<16; k++) {
            int idx = base_idx + k;
            if (idx < 4096) {
                int r = idx >> 6;
                int c = (idx & 63) << 1;
                int global_r = j * 64 + r;
                if (global_r < S) {
                    *(__nv_bfloat162*)&s_K[r][c] = *(__nv_bfloat162*)&K[j_offset_base + r * 128 + c];
                    *(__nv_bfloat162*)&s_V[r][c] = *(__nv_bfloat162*)&V[j_offset_base + r * 128 + c];
                } else {
                    *(__nv_bfloat162*)&s_K[r][c] = __floats2bfloat162_rn(0.0f, 0.0f);
                    *(__nv_bfloat162*)&s_V[r][c] = __floats2bfloat162_rn(0.0f, 0.0f);
                }
            }
        }
        __syncthreads();

        float S_tile[4][4] = {0};
        float dP_tile[4][4] = {0};
        
        for(int k=0; k<128; k+=2) {
            float2 fq[4], fdo[4], fk[4], fv[4];
            for (int r = 0; r < 4; r++) {
                fq[r]  = __bfloat1622float2(*(__nv_bfloat162*)&s_Q[ty1 * 4 + r][k]);
                fdo[r] = __bfloat1622float2(*(__nv_bfloat162*)&s_dO[ty1 * 4 + r][k]);
                fk[r]  = __bfloat1622float2(*(__nv_bfloat162*)&s_K[tx1 * 4 + r][k]);
                fv[r]  = __bfloat1622float2(*(__nv_bfloat162*)&s_V[tx1 * 4 + r][k]);
            }
            for (int r = 0; r < 4; r++) {
                for (int c = 0; c < 4; c++) {
                    S_tile[r][c]  += fq[r].x * fk[c].x + fq[r].y * fk[c].y;
                    dP_tile[r][c] += fdo[r].x * fv[c].x + fdo[r].y * fv[c].y;
                }
            }
        }

        // Apply causal mask and logsumexp, compute dS_ij
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                int row = ty1 * 4 + r;
                int col = tx1 * 4 + c;
                int global_row = i * 64 + row;
                int global_col = j * 64 + col;
                
                if (global_col > global_row || global_row >= S || global_col >= S) {
                    s_dS[row][col] = 0.0f;
                } else {
                    float p = expf(S_tile[r][c] * 0.0883883476483f - s_L[row]);
                    s_dS[row][col] = p * (dP_tile[r][c] - s_D[row]);
                }
            }
        }
        __syncthreads();

        // Compute dQ_i += dS_ij * K_j
        for (int k = 0; k < 64; k++) {
            float dsv_arr[4];
            for (int r = 0; r < 4; r++) {
                dsv_arr[r] = s_dS[row_start + r][k];
            }
            float2 fk[4];
            for (int c = 0; c < 4; c++) {
                fk[c]  = __bfloat1622float2(*(__nv_bfloat162*)&s_K[k][col_start + c*2]);
            }
            for (int r = 0; r < 4; r++) {
                for (int c = 0; c < 4; c++) {
                    dQ_tile[r][c*2]   += dsv_arr[r] * fk[c].x;
                    dQ_tile[r][c*2+1] += dsv_arr[r] * fk[c].y;
                }
            }
        }
        __syncthreads();
    } 
    
    // Scale and write dQ_i to global memory
    for (int r = 0; r < 4; r++) {
        int global_r = i * 64 + row_start + r;
        if (global_r < S) {
            for (int c = 0; c < 8; c+=2) {
                float dq0 = dQ_tile[r][c]   * 0.0883883476483f;
                float dq1 = dQ_tile[r][c+1] * 0.0883883476483f;
                *(__nv_bfloat162*)&dQ[i_offset_base + (row_start + r) * 128 + col_start + c] = __floats2bfloat162_rn(dq0, dq1);
            }
        }
    }
}

// Kernel 2: computes dK and dV (outer loop j, inner loop i >= j)
__global__ void mha_bwd_causal_kernel_dK_dV(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int B, int H, int S, int d) 
{
    int j = blockIdx.x;
    if (j * 64 >= S) return;
    int bh = blockIdx.y;
    int tid = threadIdx.x;

    extern __shared__ char smem[];
    __nv_bfloat16 (*s_Q)[138]  = reinterpret_cast<__nv_bfloat16 (*)[138]>(smem);
    __nv_bfloat16 (*s_O)[138]  = reinterpret_cast<__nv_bfloat16 (*)[138]>(smem + 17664);
    __nv_bfloat16 (*s_dO)[138] = reinterpret_cast<__nv_bfloat16 (*)[138]>(smem + 35328);
    __nv_bfloat16 (*s_K)[138]  = reinterpret_cast<__nv_bfloat16 (*)[138]>(smem + 52992);
    __nv_bfloat16 (*s_V)[138]  = reinterpret_cast<__nv_bfloat16 (*)[138]>(smem + 70656);
    
    float (*s_P)[65]  = reinterpret_cast<float (*)[65]>(smem + 88320);
    float (*s_dS)[65] = reinterpret_cast<float (*)[65]>(smem + 104960);
    
    float *s_L = reinterpret_cast<float *>(smem + 121600);
    float *s_D = reinterpret_cast<float *>(smem + 121856);

    int j_offset_base = bh * S * 128 + j * 64 * 128;
    int base_idx = tid * 16; 

    // Load K_j, V_j ONCE for the block j
    for(int k=0; k<16; k++) {
        int idx = base_idx + k;
        if (idx < 4096) {
            int r = idx >> 6;
            int c = (idx & 63) << 1;
            int global_r = j * 64 + r;
            if (global_r < S) {
                *(__nv_bfloat162*)&s_K[r][c] = *(__nv_bfloat162*)&K[j_offset_base + r * 128 + c];
                *(__nv_bfloat162*)&s_V[r][c] = *(__nv_bfloat162*)&V[j_offset_base + r * 128 + c];
            } else {
                *(__nv_bfloat162*)&s_K[r][c] = __floats2bfloat162_rn(0.0f, 0.0f);
                *(__nv_bfloat162*)&s_V[r][c] = __floats2bfloat162_rn(0.0f, 0.0f);
            }
        }
    }
    __syncthreads();

    int tx1 = tid % 16, ty1 = tid / 16;
    int tx2 = tid % 16, ty2 = tid / 16;
    int row_start = ty2 * 4;  // Index in j
    int col_start = tx2 * 8;  // Feature index
    
    // Register tiles for accumulating dK_j and dV_j over all i >= j
    float dK_tile[4][8] = {0};
    float dV_tile[4][8] = {0};

    int num_i_blocks = (S + 63) / 64;
    for (int i = j; i < num_i_blocks; i++) {
        int i_offset_base = bh * S * 128 + i * 64 * 128;
        
        // Load Q_i, O_i, dO_i
        for(int k=0; k<16; k++) {
            int idx = base_idx + k;
            if (idx < 4096) {
                int r = idx >> 6;
                int c = (idx & 63) << 1;
                int global_r = i * 64 + r;
                if (global_r < S) {
                    *(__nv_bfloat162*)&s_Q[r][c]  = *(__nv_bfloat162*)&Q[i_offset_base + r * 128 + c];
                    *(__nv_bfloat162*)&s_O[r][c]  = *(__nv_bfloat162*)&O[i_offset_base + r * 128 + c];
                    *(__nv_bfloat162*)&s_dO[r][c] = *(__nv_bfloat162*)&dO[i_offset_base + r * 128 + c];
                } else {
                    *(__nv_bfloat162*)&s_Q[r][c]  = __floats2bfloat162_rn(0.0f, 0.0f);
                    *(__nv_bfloat162*)&s_O[r][c]  = __floats2bfloat162_rn(0.0f, 0.0f);
                    *(__nv_bfloat162*)&s_dO[r][c] = __floats2bfloat162_rn(0.0f, 0.0f);
                }
            }
        }
        
        if (tid < 64) {
            if (i * 64 + tid < S) s_L[tid] = L[bh * S + i * 64 + tid];
            else s_L[tid] = 0.0f;
        }
        __syncthreads();

        // Compute D_i = rowsum(dO_i * O_i)
        if (tid < 64) {
            float sum = 0;
            for(int k=0; k<128; k+=2) {
                float2 fdo = __bfloat1622float2(*(__nv_bfloat162*)&s_dO[tid][k]);
                float2 fo  = __bfloat1622float2(*(__nv_bfloat162*)&s_O[tid][k]);
                sum += fdo.x * fo.x + fdo.y * fo.y;
            }
            s_D[tid] = sum;
        }
        __syncthreads();

        float S_tile[4][4] = {0};
        float dP_tile[4][4] = {0};
        
        for(int k=0; k<128; k+=2) {
            float2 fq[4], fdo[4], fk[4], fv[4];
            for (int r = 0; r < 4; r++) {
                fq[r]  = __bfloat1622float2(*(__nv_bfloat162*)&s_Q[ty1 * 4 + r][k]);
                fdo[r] = __bfloat1622float2(*(__nv_bfloat162*)&s_dO[ty1 * 4 + r][k]);
                fk[r]  = __bfloat1622float2(*(__nv_bfloat162*)&s_K[tx1 * 4 + r][k]);
                fv[r]  = __bfloat1622float2(*(__nv_bfloat162*)&s_V[tx1 * 4 + r][k]);
            }
            for (int r = 0; r < 4; r++) {
                for (int c = 0; c < 4; c++) {
                    S_tile[r][c]  += fq[r].x * fk[c].x + fq[r].y * fk[c].y;
                    dP_tile[r][c] += fdo[r].x * fv[c].x + fdo[r].y * fv[c].y;
                }
            }
        }

        // Apply causal mask and logsumexp, compute P_ij and dS_ij
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                int row = ty1 * 4 + r; // index in i
                int col = tx1 * 4 + c; // index in j
                int global_row = i * 64 + row;
                int global_col = j * 64 + col;
                
                if (global_col > global_row || global_row >= S || global_col >= S) {
                    s_P[row][col]  = 0.0f;
                    s_dS[row][col] = 0.0f;
                } else {
                    float p = expf(S_tile[r][c] * 0.0883883476483f - s_L[row]);
                    s_P[row][col]  = p;
                    s_dS[row][col] = p * (dP_tile[r][c] - s_D[row]);
                }
            }
        }
        __syncthreads();

        // Compute dV_j += P_ij^T * dO_i  and  dK_j += dS_ij^T * Q_i
        for (int k = 0; k < 64; k++) { // k is index in i
            float p_arr[4];
            float dsk_arr[4];
            for (int r = 0; r < 4; r++) {
                // row_start + r is index in j
                p_arr[r]   = s_P[k][row_start + r];
                dsk_arr[r] = s_dS[k][row_start + r];
            }
            
            float2 fdo[4], fq[4];
            for (int c = 0; c < 4; c++) {
                fdo[c] = __bfloat1622float2(*(__nv_bfloat162*)&s_dO[k][col_start + c*2]);
                fq[c]  = __bfloat1622float2(*(__nv_bfloat162*)&s_Q[k][col_start + c*2]);
            }
            
            for (int r = 0; r < 4; r++) {
                for (int c = 0; c < 4; c++) {
                    dV_tile[r][c*2]   += p_arr[r] * fdo[c].x;
                    dV_tile[r][c*2+1] += p_arr[r] * fdo[c].y;
                    
                    dK_tile[r][c*2]   += dsk_arr[r] * fq[c].x;
                    dK_tile[r][c*2+1] += dsk_arr[r] * fq[c].y;
                }
            }
        }
        __syncthreads();
    } 
    
    // Write dV_j and scaled dK_j to global memory
    for (int r = 0; r < 4; r++) {
        int global_r = j * 64 + row_start + r; // index in j
        if (global_r < S) {
            for (int c = 0; c < 8; c+=2) {
                *(__nv_bfloat162*)&dV[j_offset_base + (row_start + r) * 128 + col_start + c] = __floats2bfloat162_rn(dV_tile[r][c], dV_tile[r][c+1]);
                
                float dk0 = dK_tile[r][c]   * 0.0883883476483f;
                float dk1 = dK_tile[r][c+1] * 0.0883883476483f;
                *(__nv_bfloat162*)&dK[j_offset_base + (row_start + r) * 128 + col_start + c] = __floats2bfloat162_rn(dk0, dk1);
            }
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

    const __nv_bfloat16* q_ptr  = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* k_ptr  = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* v_ptr  = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* o_ptr  = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* do_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* l_ptr = static_cast<const float*>(L.data_ptr());

    __nv_bfloat16* dq_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dk_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dv_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());

    dim3 threads(256);
    dim3 blocks((S + 63) / 64, B * H);
    
    // Total shared memory requirement exactly 122112 bytes
    int shared_mem_size = 122112; 
    CUDA_CHECK(cudaFuncSetAttribute(mha_bwd_causal_kernel_dQ, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size));
    CUDA_CHECK(cudaFuncSetAttribute(mha_bwd_causal_kernel_dK_dV, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size));

    // Pass 1: Compute dQ
    mha_bwd_causal_kernel_dQ<<<blocks, threads, shared_mem_size, stream>>>(
        q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, l_ptr, dq_ptr, B, H, S, d
    );
    CUDA_CHECK(cudaGetLastError());
    
    // Pass 2: Compute dK and dV
    mha_bwd_causal_kernel_dK_dV<<<blocks, threads, shared_mem_size, stream>>>(
        q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, l_ptr, dk_ptr, dv_ptr, B, H, S, d
    );
    CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda