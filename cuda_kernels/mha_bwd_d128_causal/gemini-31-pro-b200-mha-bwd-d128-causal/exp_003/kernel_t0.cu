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

__global__ void mha_bwd_causal_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int B, int H, int S, int d) 
{
    // Block config: blockIdx.x = Q block index (i), blockIdx.y = batch * head
    int i = blockIdx.x;
    if (i * 64 >= S) return;
    int bh = blockIdx.y;
    int tid = threadIdx.x; // 0..255

    // Shared memory allocations
    // To avoid bank conflicts, we pad the 128 dimension to 138 for bf16 accesses.
    // 138 bf16 = 276 bytes. 276 % 4 == 0, preserving 4-byte alignment for __nv_bfloat162.
    extern __shared__ char smem[];
    __nv_bfloat16 (*s_Q)[138]  = reinterpret_cast<__nv_bfloat16 (*)[138]>(smem);
    __nv_bfloat16 (*s_O)[138]  = reinterpret_cast<__nv_bfloat16 (*)[138]>(smem + 17664);
    __nv_bfloat16 (*s_dO)[138] = reinterpret_cast<__nv_bfloat16 (*)[138]>(smem + 35328);
    __nv_bfloat16 (*s_K)[138]  = reinterpret_cast<__nv_bfloat16 (*)[138]>(smem + 52992);
    __nv_bfloat16 (*s_V)[138]  = reinterpret_cast<__nv_bfloat16 (*)[138]>(smem + 70656);
    
    // Padded to 65 floats (260 bytes) to avoid bank conflicts on column reads.
    float (*s_P)[65]  = reinterpret_cast<float (*)[65]>(smem + 88320);
    float (*s_dS)[65] = reinterpret_cast<float (*)[65]>(smem + 104960);
    
    float *s_L = reinterpret_cast<float *>(smem + 121600);
    float *s_D = reinterpret_cast<float *>(smem + 121856);

    int i_offset_base = bh * S * 128 + i * 64 * 128;
    int base_idx = tid * 16; 

    // 1. Load Q_i, O_i, dO_i
    for(int k=0; k<16; k++) {
        int idx = base_idx + k;
        if (idx < 4096) {
            int r = idx / 64;
            int c = (idx % 64) * 2;
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
    
    // 2. Load L_i
    if (tid < 64) {
        if (i * 64 + tid < S) {
            s_L[tid] = L[bh * S + i * 64 + tid];
        } else {
            s_L[tid] = 0.0f;
        }
    }
    __syncthreads();

    // 3. Compute D_i = rowsum(dO_i * O_i)
    if (tid < 64) {
        float sum = 0;
        for(int k=0; k<128; k+=2) {
            __nv_bfloat162 do_val = *(__nv_bfloat162*)&s_dO[tid][k];
            __nv_bfloat162 o_val  = *(__nv_bfloat162*)&s_O[tid][k];
            float2 fdo = __bfloat1622float2(do_val);
            float2 fo  = __bfloat1622float2(o_val);
            sum += fdo.x * fo.x + fdo.y * fo.y;
        }
        s_D[tid] = sum;
    }
    __syncthreads();

    // Register tiles for accumulation
    float dQ_tile[4][8] = {0};

    // Inner loop over K, V blocks
    for (int j = 0; j <= i; j++) {
        int j_offset_base = bh * S * 128 + j * 64 * 128;
        
        // 4. Load K_j, V_j
        for(int k=0; k<16; k++) {
            int idx = base_idx + k;
            if (idx < 4096) {
                int r = idx / 64;
                int c = (idx % 64) * 2;
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

        // 5. Compute S_ij and dP_ij
        int tx1 = tid % 16;
        int ty1 = tid / 16;
        float S_tile[4][4] = {0};
        float dP_tile[4][4] = {0};

        for(int k=0; k<128; k+=2) {
            for (int r = 0; r < 4; r++) {
                __nv_bfloat162 q_val  = *(__nv_bfloat162*)&s_Q[ty1 * 4 + r][k];
                __nv_bfloat162 do_val = *(__nv_bfloat162*)&s_dO[ty1 * 4 + r][k];
                float2 fq  = __bfloat1622float2(q_val);
                float2 fdo = __bfloat1622float2(do_val);
                
                for (int c = 0; c < 4; c++) {
                    __nv_bfloat162 k_val = *(__nv_bfloat162*)&s_K[tx1 * 4 + c][k];
                    __nv_bfloat162 v_val = *(__nv_bfloat162*)&s_V[tx1 * 4 + c][k];
                    float2 fk = __bfloat1622float2(k_val);
                    float2 fv = __bfloat1622float2(v_val);
                    
                    S_tile[r][c]  += fq.x * fk.x + fq.y * fk.y;
                    dP_tile[r][c] += fdo.x * fv.x + fdo.y * fv.y;
                }
            }
        }

        // Apply causal mask and logsumexp, compute P_ij and dS_ij
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                int row = ty1 * 4 + r;
                int col = tx1 * 4 + c;
                int global_row = i * 64 + row;
                int global_col = j * 64 + col;
                
                if (global_col > global_row || global_row >= S || global_col >= S) {
                    s_P[row][col]  = 0.0f;
                    s_dS[row][col] = 0.0f;
                } else {
                    float p = expf(S_tile[r][c] * (1.0f / 11.313708499f) - s_L[row]);
                    s_P[row][col]  = p;
                    s_dS[row][col] = p * (dP_tile[r][c] - s_D[row]);
                }
            }
        }
        __syncthreads();

        // 6. Compute dV_j, dQ_i, dK_j
        int tx2 = tid % 16;
        int ty2 = tid / 16;
        int row_start = ty2 * 4;
        int col_start = tx2 * 8;
        
        float dV_tile[4][8] = {0};
        float dK_tile[4][8] = {0};
        
        for (int k = 0; k < 64; k++) {
            for (int r = 0; r < 4; r++) {
                float p    = s_P[k][row_start + r];
                float ds_v = s_dS[row_start + r][k];
                float ds_k = s_dS[k][row_start + r];
                
                for (int c = 0; c < 8; c+=2) {
                    __nv_bfloat162 do_val = *(__nv_bfloat162*)&s_dO[k][col_start + c];
                    float2 fdo = __bfloat1622float2(do_val);
                    dV_tile[r][c]   += p * fdo.x;
                    dV_tile[r][c+1] += p * fdo.y;
                    
                    __nv_bfloat162 k_val = *(__nv_bfloat162*)&s_K[k][col_start + c];
                    float2 fk = __bfloat1622float2(k_val);
                    dQ_tile[r][c]   += ds_v * fk.x;
                    dQ_tile[r][c+1] += ds_v * fk.y;
                    
                    __nv_bfloat162 q_val = *(__nv_bfloat162*)&s_Q[k][col_start + c];
                    float2 fq = __bfloat1622float2(q_val);
                    dK_tile[r][c]   += ds_k * fq.x;
                    dK_tile[r][c+1] += ds_k * fq.y;
                }
            }
        }
        
        // Write dV and dK to global memory
        for (int r = 0; r < 4; r++) {
            int global_r = j * 64 + row_start + r;
            if (global_r < S) {
                for (int c = 0; c < 8; c+=2) {
                    __nv_bfloat162 dv_add = __floats2bfloat162_rn(dV_tile[r][c], dV_tile[r][c+1]);
                    atomicAdd((__nv_bfloat162*)&dV[j_offset_base + (row_start + r) * 128 + col_start + c], dv_add);
                    
                    float dk0 = dK_tile[r][c]   * (1.0f / 11.313708499f);
                    float dk1 = dK_tile[r][c+1] * (1.0f / 11.313708499f);
                    __nv_bfloat162 dk_add = __floats2bfloat162_rn(dk0, dk1);
                    atomicAdd((__nv_bfloat162*)&dK[j_offset_base + (row_start + r) * 128 + col_start + c], dk_add);
                }
            }
        }
        __syncthreads();
    } 
    
    // 7. Write dQ to global memory
    int tx2 = tid % 16;
    int ty2 = tid / 16;
    int row_start = ty2 * 4;
    int col_start = tx2 * 8;
    for (int r = 0; r < 4; r++) {
        int global_r = i * 64 + row_start + r;
        if (global_r < S) {
            for (int c = 0; c < 8; c+=2) {
                float dq0 = dQ_tile[r][c]   * (1.0f / 11.313708499f);
                float dq1 = dQ_tile[r][c+1] * (1.0f / 11.313708499f);
                __nv_bfloat162 dq_out = __floats2bfloat162_rn(dq0, dq1);
                *(__nv_bfloat162*)&dQ[i_offset_base + (row_start + r) * 128 + col_start + c] = dq_out;
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

    // Initialize dK and dV to 0 because we use atomicAdd to accumulate them
    int64_t num_elements = (int64_t)B * H * S * d;
    CUDA_CHECK(cudaMemsetAsync(dk_ptr, 0, num_elements * sizeof(uint16_t), stream));
    CUDA_CHECK(cudaMemsetAsync(dv_ptr, 0, num_elements * sizeof(uint16_t), stream));

    // Calculate grid and block dimensions
    dim3 threads(256);
    dim3 blocks((S + 63) / 64, B * H);
    
    int shared_mem_size = 122112; 
    CUDA_CHECK(cudaFuncSetAttribute(mha_bwd_causal_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size));

    mha_bwd_causal_kernel<<<blocks, threads, shared_mem_size, stream>>>(
        q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, l_ptr, dq_ptr, dk_ptr, dv_ptr, B, H, S, d
    );
    CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda