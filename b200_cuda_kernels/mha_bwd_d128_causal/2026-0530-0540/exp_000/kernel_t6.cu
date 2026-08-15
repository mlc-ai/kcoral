#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <mma.h>
#include <stdio.h>
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

using namespace nvcuda;

// Asynchronous 16-byte Hopper copy instruction (No Driver TMA hassle needed)
__device__ __forceinline__ void cp_async_16B(void* smem, const void* gmem) {
    uint32_t smem_ptr = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" : : "r"(smem_ptr), "l"(gmem) : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;" : : : "memory");
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;" : : "n"(N) : "memory");
}

__device__ __forceinline__ void load_smem_64x128(
    __nv_bfloat16* smem, const __nv_bfloat16* gmem, 
    int global_row, int seq_len, int batch_offset, int tid
) {
    for (int i = 0; i < 8; i++) {
        int idx = tid + i * 128; // 0..1023
        int r = idx / 16;        // 0..63
        int c = (idx % 16) * 8;  // 0, 8, 16..120
        if (global_row + r < seq_len) {
            cp_async_16B(&smem[r * 128 + c], &gmem[batch_offset + (global_row + r) * 128 + c]);
        } else {
            *(uint4*)&smem[r * 128 + c] = {0, 0, 0, 0};
        }
    }
}

// -------------------------------------------------------------
// Kernel 1: Computes dQ (Outer Loop: Q, Inner Loop: KV)
// -------------------------------------------------------------
__global__ __launch_bounds__(128) void sdpa_bwd_dQ_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    int seq_len
) {
    extern __shared__ uint8_t smem[];
    __nv_bfloat16* Q_smem  = (__nv_bfloat16*)smem;                   // 16KB
    __nv_bfloat16* dO_smem = Q_smem + 64 * 128;                      // 16KB
    __nv_bfloat16* O_smem  = dO_smem + 64 * 128;                     // 16KB
    __nv_bfloat16* K_smem  = O_smem + 64 * 128;                      // 32KB
    __nv_bfloat16* V_smem  = K_smem + 2 * 64 * 128;                  // 32KB
    float* S_smem_f        = (float*)(V_smem + 2 * 64 * 128);        // 16KB
    float* dP_smem_f       = S_smem_f + 64 * 64;                     // 16KB
    __nv_bfloat16* dS_smem_bf16 = (__nv_bfloat16*)(dP_smem_f + 64 * 64); // 8KB
    float* L_smem          = (float*)(dS_smem_bf16 + 64 * 64);       // 256B
    float* D_smem          = L_smem + 64;                            // 256B

    int block_i = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int Wy = tid / 32;

    int global_q_row = block_i * 64;
    int batch_offset = (b * gridDim.y + h) * seq_len * 128;
    int l_offset = (b * gridDim.y + h) * seq_len;

    load_smem_64x128(&Q_smem[0], Q, global_q_row, seq_len, batch_offset, tid);
    load_smem_64x128(&dO_smem[0], dO, global_q_row, seq_len, batch_offset, tid);
    load_smem_64x128(&O_smem[0], O, global_q_row, seq_len, batch_offset, tid);
    if (tid < 64) {
        L_smem[tid] = (global_q_row + tid < seq_len) ? L[l_offset + global_q_row + tid] : 0.0f;
    }
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();

    if (tid < 64) {
        float d_val = 0.0f;
        float4* o_row = (float4*)&O_smem[tid * 128];
        float4* do_row = (float4*)&dO_smem[tid * 128];
        for (int k = 0; k < 16; k++) {
            float4 o_vec = o_row[k];
            float4 do_vec = do_row[k];
            __nv_bfloat162* o_bf = (__nv_bfloat162*)&o_vec;
            __nv_bfloat162* do_bf = (__nv_bfloat162*)&do_vec;
            for (int i = 0; i < 4; i++) {
                float2 o_f2 = __bfloat1622float2(o_bf[i]);
                float2 do_f2 = __bfloat1622float2(do_bf[i]);
                d_val += o_f2.x * do_f2.x + o_f2.y * do_f2.y;
            }
        }
        D_smem[tid] = d_val;
    }

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dQ_frag[8];
    for(int i = 0; i < 8; i++) wmma::fill_fragment(dQ_frag[i], 0.0f);

    int k_smem_offset = 0;
    int global_k_row = 0;
    load_smem_64x128(&K_smem[0], K, global_k_row, seq_len, batch_offset, tid);
    load_smem_64x128(&V_smem[0], V, global_k_row, seq_len, batch_offset, tid);
    cp_async_commit();

    for (int block_j = 0; block_j <= block_i; block_j++) {
        cp_async_wait<0>();
        __syncthreads();
        
        int next_k_row = (block_j + 1) * 64;
        int next_offset = k_smem_offset ^ (64 * 128);
        
        if (block_j + 1 <= block_i) {
            load_smem_64x128(&K_smem[next_offset], K, next_k_row, seq_len, batch_offset, tid);
            load_smem_64x128(&V_smem[next_offset], V, next_k_row, seq_len, batch_offset, tid);
            cp_async_commit();
        }

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> S_frag[4];
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> dP_frag[4];
        for(int c = 0; c < 4; c++) { wmma::fill_fragment(S_frag[c], 0.0f); wmma::fill_fragment(dP_frag[c], 0.0f); }

        for (int k = 0; k < 128; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> q_frag;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> do_frag;
            wmma::load_matrix_sync(q_frag, &Q_smem[Wy * 16 * 128 + k], 128);
            wmma::load_matrix_sync(do_frag, &dO_smem[Wy * 16 * 128 + k], 128);

            for (int c = 0; c < 4; c++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> k_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> v_frag;
                wmma::load_matrix_sync(k_frag, &K_smem[k_smem_offset + c * 16 * 128 + k], 128);
                wmma::load_matrix_sync(v_frag, &V_smem[k_smem_offset + c * 16 * 128 + k], 128);
                
                wmma::mma_sync(S_frag[c], q_frag, k_frag, S_frag[c]);
                wmma::mma_sync(dP_frag[c], do_frag, v_frag, dP_frag[c]);
            }
        }

        for (int c = 0; c < 4; c++) {
            wmma::store_matrix_sync(&S_smem_f[(Wy * 16) * 64 + c * 16], S_frag[c], 64, wmma::mem_row_major);
            wmma::store_matrix_sync(&dP_smem_f[(Wy * 16) * 64 + c * 16], dP_frag[c], 64, wmma::mem_row_major);
        }
        __syncwarp();

        for (int i = lane; i < 1024; i += 32) {
            int r = i / 64;       // 0..15
            int col = i % 64;     // 0..63
            int glob_r = global_q_row + Wy * 16 + r;
            int glob_c = block_j * 64 + col;
            
            float ds = 0.0f;
            if (glob_r < seq_len && glob_c < seq_len && glob_c <= glob_r) {
                float s = S_smem_f[(Wy * 16 + r) * 64 + col] * 0.088388347648f;
                float p = expf(s - L_smem[Wy * 16 + r]);
                ds = p * (dP_smem_f[(Wy * 16 + r) * 64 + col] - D_smem[Wy * 16 + r]) * 0.088388347648f;
            }
            dS_smem_bf16[(Wy * 16 + r) * 64 + col] = __float2bfloat16(ds);
        }
        __syncwarp();

        for (int k = 0; k < 64; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> ds_frag;
            wmma::load_matrix_sync(ds_frag, &dS_smem_bf16[Wy * 16 * 64 + k], 64);
            for (int c = 0; c < 8; c++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> k_frag;
                wmma::load_matrix_sync(k_frag, &K_smem[k_smem_offset + k * 128 + c * 16], 128);
                wmma::mma_sync(dQ_frag[c], ds_frag, k_frag, dQ_frag[c]);
            }
        }
        __syncthreads(); 
        k_smem_offset = next_offset;
    }

    float* dQ_smem_f = (float*)Q_smem;
    for (int c = 0; c < 8; c++) {
        wmma::store_matrix_sync(&dQ_smem_f[Wy * 16 * 128 + c * 16], dQ_frag[c], 128, wmma::mem_row_major);
    }
    __syncthreads();

    for(int idx = tid; idx < 4096; idx += 128) {
        int r = idx / 64;
        int c = (idx % 64) * 2;
        if (global_q_row + r < seq_len) {
            float dq0 = dQ_smem_f[r * 128 + c];
            float dq1 = dQ_smem_f[r * 128 + c + 1];
            __nv_bfloat162 dq_bf = __floats2bfloat162_rn(dq0, dq1);
            *(uint32_t*)&dQ[batch_offset + (global_q_row + r) * 128 + c] = *(uint32_t*)&dq_bf;
        }
    }
}

// -------------------------------------------------------------
// Kernel 2: Computes dK, dV (Outer Loop: KV, Inner Loop: Q)
// -------------------------------------------------------------
__global__ __launch_bounds__(128) void sdpa_bwd_dK_dV_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int seq_len
) {
    extern __shared__ uint8_t smem[];
    __nv_bfloat16* K_smem  = (__nv_bfloat16*)smem;                   // 16KB
    __nv_bfloat16* V_smem  = K_smem + 64 * 128;                      // 16KB
    __nv_bfloat16* Q_smem  = V_smem + 64 * 128;                      // 32KB
    __nv_bfloat16* dO_smem = Q_smem + 2 * 64 * 128;                  // 32KB
    __nv_bfloat16* O_smem  = dO_smem + 2 * 64 * 128;                 // 32KB
    float* S_smem_f        = (float*)(O_smem + 2 * 64 * 128);        // 16KB
    float* dP_smem_f       = S_smem_f + 64 * 64;                     // 16KB
    __nv_bfloat16* dS_smem_bf16 = (__nv_bfloat16*)(dP_smem_f + 64 * 64); // 8KB
    __nv_bfloat16* P_smem_bf16  = dS_smem_bf16 + 64 * 64;            // 8KB
    float* L_smem          = (float*)(P_smem_bf16 + 64 * 64);        // 512B
    float* D_smem          = L_smem + 128;                           // 512B

    int block_j = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int Wy = tid / 32;

    int global_k_row = block_j * 64;
    int batch_offset = (b * gridDim.y + h) * seq_len * 128;
    int l_offset = (b * gridDim.y + h) * seq_len;

    load_smem_64x128(&K_smem[0], K, global_k_row, seq_len, batch_offset, tid);
    load_smem_64x128(&V_smem[0], V, global_k_row, seq_len, batch_offset, tid);

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dK_frag[8];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dV_frag[8];
    for(int c = 0; c < 8; c++) { 
        wmma::fill_fragment(dK_frag[c], 0.0f); 
        wmma::fill_fragment(dV_frag[c], 0.0f); 
    }

    int start_i = block_j; 
    int end_i = (seq_len + 63) / 64;

    int q_smem_offset = 0;
    int l_smem_offset = 0;
    int global_q_row = start_i * 64;

    if (start_i < end_i) {
        load_smem_64x128(&Q_smem[0], Q, global_q_row, seq_len, batch_offset, tid);
        load_smem_64x128(&dO_smem[0], dO, global_q_row, seq_len, batch_offset, tid);
        load_smem_64x128(&O_smem[0], O, global_q_row, seq_len, batch_offset, tid);
        if (tid < 64) {
            L_smem[tid] = (global_q_row + tid < seq_len) ? L[l_offset + global_q_row + tid] : 0.0f;
        }
        cp_async_commit();
    }

    for (int block_i = start_i; block_i < end_i; block_i++) {
        cp_async_wait<0>();
        __syncthreads();

        if (tid < 64) {
            float d_val = 0.0f;
            float4* o_row = (float4*)&O_smem[q_smem_offset + tid * 128];
            float4* do_row = (float4*)&dO_smem[q_smem_offset + tid * 128];
            for (int k = 0; k < 16; k++) {
                float4 o_vec = o_row[k];
                float4 do_vec = do_row[k];
                __nv_bfloat162* o_bf = (__nv_bfloat162*)&o_vec;
                __nv_bfloat162* do_bf = (__nv_bfloat162*)&do_vec;
                for (int i = 0; i < 4; i++) {
                    float2 o_f2 = __bfloat1622float2(o_bf[i]);
                    float2 do_f2 = __bfloat1622float2(do_bf[i]);
                    d_val += o_f2.x * do_f2.x + o_f2.y * do_f2.y;
                }
            }
            D_smem[l_smem_offset + tid] = d_val;
        }
        __syncthreads();
        
        int next_q_row = (block_i + 1) * 64;
        int next_q_offset = q_smem_offset ^ (64 * 128);
        int next_l_offset = l_smem_offset ^ 64;
        
        if (block_i + 1 < end_i) {
            load_smem_64x128(&Q_smem[next_q_offset], Q, next_q_row, seq_len, batch_offset, tid);
            load_smem_64x128(&dO_smem[next_q_offset], dO, next_q_row, seq_len, batch_offset, tid);
            load_smem_64x128(&O_smem[next_q_offset], O, next_q_row, seq_len, batch_offset, tid);
            if (tid < 64) {
                L_smem[next_l_offset + tid] = (next_q_row + tid < seq_len) ? L[l_offset + next_q_row + tid] : 0.0f;
            }
            cp_async_commit();
        }

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> S_frag[4];
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> dP_frag[4];
        for(int c = 0; c < 4; c++) { 
            wmma::fill_fragment(S_frag[c], 0.0f); 
            wmma::fill_fragment(dP_frag[c], 0.0f); 
        }

        for (int k = 0; k < 128; k += 16) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> k_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> v_frag;
            wmma::load_matrix_sync(k_frag, &K_smem[Wy * 16 * 128 + k], 128);
            wmma::load_matrix_sync(v_frag, &V_smem[Wy * 16 * 128 + k], 128);

            for (int c = 0; c < 4; c++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> q_frag;
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> do_frag;
                wmma::load_matrix_sync(q_frag, &Q_smem[q_smem_offset + c * 16 * 128 + k], 128);
                wmma::load_matrix_sync(do_frag, &dO_smem[q_smem_offset + c * 16 * 128 + k], 128);
                
                wmma::mma_sync(S_frag[c], q_frag, k_frag, S_frag[c]);
                wmma::mma_sync(dP_frag[c], do_frag, v_frag, dP_frag[c]);
            }
        }

        for (int c = 0; c < 4; c++) {
            wmma::store_matrix_sync(&S_smem_f[(c * 16) * 64 + Wy * 16], S_frag[c], 64, wmma::mem_row_major);
            wmma::store_matrix_sync(&dP_smem_f[(c * 16) * 64 + Wy * 16], dP_frag[c], 64, wmma::mem_row_major);
        }
        __syncwarp();

        for (int i = lane; i < 1024; i += 32) {
            int r = i / 16;       // 0..63
            int col = i % 16;     // 0..15
            int glob_r = (block_i * 64) + r;
            int glob_c = global_k_row + Wy * 16 + col;
            
            float ds = 0.0f;
            float p = 0.0f;
            if (glob_r < seq_len && glob_c < seq_len && glob_c <= glob_r) {
                float s = S_smem_f[r * 64 + Wy * 16 + col] * 0.088388347648f;
                p = expf(s - L_smem[l_smem_offset + r]);
                ds = p * (dP_smem_f[r * 64 + Wy * 16 + col] - D_smem[l_smem_offset + r]) * 0.088388347648f;
            }
            dS_smem_bf16[r * 64 + Wy * 16 + col] = __float2bfloat16(ds);
            P_smem_bf16[r * 64 + Wy * 16 + col]  = __float2bfloat16(p);
        }
        __syncwarp();

        for (int k = 0; k < 64; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> ds_t_frag;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> p_t_frag;
            wmma::load_matrix_sync(ds_t_frag, &dS_smem_bf16[k * 64 + Wy * 16], 64);
            wmma::load_matrix_sync(p_t_frag, &P_smem_bf16[k * 64 + Wy * 16], 64);

            for (int c = 0; c < 8; c++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> q_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> do_frag;
                wmma::load_matrix_sync(q_frag, &Q_smem[q_smem_offset + k * 128 + c * 16], 128);
                wmma::load_matrix_sync(do_frag, &dO_smem[q_smem_offset + k * 128 + c * 16], 128);
                
                wmma::mma_sync(dK_frag[c], ds_t_frag, q_frag, dK_frag[c]);
                wmma::mma_sync(dV_frag[c], p_t_frag, do_frag, dV_frag[c]);
            }
        }
        __syncthreads();
        
        q_smem_offset = next_q_offset;
        l_smem_offset = next_l_offset;
    }

    float* dK_smem_f = (float*)Q_smem;
    float* dV_smem_f = (float*)dO_smem;
    
    for (int c = 0; c < 8; c++) {
        wmma::store_matrix_sync(&dK_smem_f[Wy * 16 * 128 + c * 16], dK_frag[c], 128, wmma::mem_row_major);
        wmma::store_matrix_sync(&dV_smem_f[Wy * 16 * 128 + c * 16], dV_frag[c], 128, wmma::mem_row_major);
    }
    __syncthreads();

    for(int idx = tid; idx < 4096; idx += 128) {
        int r = idx / 64;
        int c = (idx % 64) * 2;
        if (global_k_row + r < seq_len) {
            float dk0 = dK_smem_f[r * 128 + c];
            float dk1 = dK_smem_f[r * 128 + c + 1];
            float dv0 = dV_smem_f[r * 128 + c];
            float dv1 = dV_smem_f[r * 128 + c + 1];
            
            __nv_bfloat162 dk_bf = __floats2bfloat162_rn(dk0, dk1);
            __nv_bfloat162 dv_bf = __floats2bfloat162_rn(dv0, dv1);
            *(uint32_t*)&dK[batch_offset + (global_k_row + r) * 128 + c] = *(uint32_t*)&dk_bf;
            *(uint32_t*)&dV[batch_offset + (global_k_row + r) * 128 + c] = *(uint32_t*)&dv_bf;
        }
    }
}


namespace tvm_ffi_sdpa_causal {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    
    if (Q.size(2) == 0) return;
         
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    
    dim3 grid((S + 63) / 64, H, B);
    dim3 block(128);

    int smem_dQ = 160000;
    int smem_dK = 180000;
    
    CUDA_CHECK(cudaFuncSetAttribute(sdpa_bwd_dQ_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_dQ));
    sdpa_bwd_dQ_kernel<<<grid, block, smem_dQ, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(O.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        S
    );

    CUDA_CHECK(cudaFuncSetAttribute(sdpa_bwd_dK_dV_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_dK));
    sdpa_bwd_dK_dV_kernel<<<grid, block, smem_dK, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(O.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        S
    );
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_sdpa_causal::run);

}