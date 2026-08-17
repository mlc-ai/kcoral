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

__global__ void sdpa_bwd_kernel_wmma(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int seq_len, int B, int H
) {
    extern __shared__ uint8_t shared_memory[];
    
    // Total size ~192.25 KB - Well within Hopper's 227 KB capability
    __nv_bfloat16* Q_smem = (__nv_bfloat16*)shared_memory;       
    __nv_bfloat16* dO_smem = Q_smem + 64 * 128;                  
    __nv_bfloat16* K_smem = dO_smem + 64 * 128;                  
    __nv_bfloat16* V_smem = K_smem + 64 * 128;                   
    __nv_bfloat16* O_smem = V_smem + 64 * 128;                   
    float* S_smem = (float*)(O_smem + 64 * 128);                 
    float* dP_smem = S_smem + 64 * 64;                           
    __nv_bfloat16* P_smem = (__nv_bfloat16*)(dP_smem + 64 * 64); 
    __nv_bfloat16* dS_smem = P_smem + 64 * 64;                   
    float* dK_smem_f = (float*)(dS_smem + 64 * 64);              
    float* dV_smem_f = dK_smem_f + 64 * 128;                     
    float* D_smem = (float*)(dV_smem_f + 64 * 128);              
    float* L_smem = D_smem + 64;                                 
    
    int block_i = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    int tid = threadIdx.x;
    
    // Configure Warp Mappings. 4 warps -> 2x2 grid
    int warp_id = tid / 32;
    int Wy = warp_id / 2;
    int Wx = warp_id % 2;
    
    int batch_offset = (b * H + h) * seq_len * 128;
    int l_offset = (b * H + h) * seq_len;
    
    int global_q_row = block_i * 64;
    
    // 1. Vectorized Memory Loads for Q, dO, O, and L
    for(int idx = tid; idx < 4096; idx += 128) {
        int r = idx / 64;
        int c = (idx % 64) * 2;
        if (global_q_row + r < seq_len) {
            uint32_t* q_ptr = (uint32_t*)&Q[batch_offset + (global_q_row + r) * 128 + c];
            uint32_t* do_ptr = (uint32_t*)&dO[batch_offset + (global_q_row + r) * 128 + c];
            uint32_t* o_ptr = (uint32_t*)&O[batch_offset + (global_q_row + r) * 128 + c];
            
            ((uint32_t*)Q_smem)[idx] = *q_ptr;
            ((uint32_t*)dO_smem)[idx] = *do_ptr;
            ((uint32_t*)O_smem)[idx] = *o_ptr;
        } else {
            ((uint32_t*)Q_smem)[idx] = 0;
            ((uint32_t*)dO_smem)[idx] = 0;
            ((uint32_t*)O_smem)[idx] = 0;
        }
    }
    if (tid < 64) {
        if (global_q_row + tid < seq_len) {
            L_smem[tid] = L[l_offset + global_q_row + tid];
        } else {
            L_smem[tid] = 0.0f;
        }
    }
    __syncthreads();
    
    // 2. Compute the Row sum reduction factors (D_i)
    if (tid < 64) {
        float d_val = 0.0f;
        if (global_q_row + tid < seq_len) {
            for(int k = 0; k < 128; k++) {
                float do_val = __bfloat162float(dO_smem[tid * 128 + k]);
                float o_val = __bfloat162float(O_smem[tid * 128 + k]);
                d_val += do_val * o_val;
            }
        }
        D_smem[tid] = d_val;
    }
    __syncthreads();
    
    // Accumulator fragment for dQ_i to continuously integrate iteratively inside j loop
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dQ_frag[2][4];
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 4; j++) {
            wmma::fill_fragment(dQ_frag[i][j], 0.0f);
        }
    }
    
    for (int block_j = 0; block_j <= block_i; block_j++) {
        int global_k_row = block_j * 64;
        
        // Load K, V Memory
        for(int idx = tid; idx < 4096; idx += 128) {
            int r = idx / 64;
            int c = (idx % 64) * 2;
            if (global_k_row + r < seq_len) {
                uint32_t* k_ptr = (uint32_t*)&K[batch_offset + (global_k_row + r) * 128 + c];
                uint32_t* v_ptr = (uint32_t*)&V[batch_offset + (global_k_row + r) * 128 + c];
                ((uint32_t*)K_smem)[idx] = *k_ptr;
                ((uint32_t*)V_smem)[idx] = *v_ptr;
            } else {
                ((uint32_t*)K_smem)[idx] = 0;
                ((uint32_t*)V_smem)[idx] = 0;
            }
        }
        __syncthreads();
        
        // 3. Tensor Core Pass: S = Q K^T
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> S_frag[2][2];
        for(int i = 0; i < 2; i++) for(int j = 0; j < 2; j++) wmma::fill_fragment(S_frag[i][j], 0.0f);
        for(int k = 0; k < 128; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> q_frag[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> k_frag[2];
            for(int i = 0; i < 2; i++) wmma::load_matrix_sync(q_frag[i], &Q_smem[(Wy * 32 + i * 16) * 128 + k], 128);
            for(int j = 0; j < 2; j++) wmma::load_matrix_sync(k_frag[j], &K_smem[(Wx * 32 + j * 16) * 128 + k], 128); // col_major applies hardware transposition logically!
            for(int i = 0; i < 2; i++) for(int j = 0; j < 2; j++) wmma::mma_sync(S_frag[i][j], q_frag[i], k_frag[j], S_frag[i][j]);
        }
        for(int i = 0; i < 2; i++) {
            for(int j = 0; j < 2; j++) {
                wmma::store_matrix_sync(&S_smem[(Wy * 32 + i * 16) * 64 + (Wx * 32 + j * 16)], S_frag[i][j], 64, wmma::mem_row_major);
            }
        }
        
        // 4. Tensor Core Pass: dP = dO V^T
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> dP_frag[2][2];
        for(int i = 0; i < 2; i++) for(int j = 0; j < 2; j++) wmma::fill_fragment(dP_frag[i][j], 0.0f);
        for(int k = 0; k < 128; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> do_frag[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> v_t_frag[2];
            for(int i = 0; i < 2; i++) wmma::load_matrix_sync(do_frag[i], &dO_smem[(Wy * 32 + i * 16) * 128 + k], 128);
            for(int j = 0; j < 2; j++) wmma::load_matrix_sync(v_t_frag[j], &V_smem[(Wx * 32 + j * 16) * 128 + k], 128);
            for(int i = 0; i < 2; i++) for(int j = 0; j < 2; j++) wmma::mma_sync(dP_frag[i][j], do_frag[i], v_t_frag[j], dP_frag[i][j]);
        }
        for(int i = 0; i < 2; i++) {
            for(int j = 0; j < 2; j++) {
                wmma::store_matrix_sync(&dP_smem[(Wy * 32 + i * 16) * 64 + (Wx * 32 + j * 16)], dP_frag[i][j], 64, wmma::mem_row_major);
            }
        }
        __syncthreads();
        
        // 5. Causal Mask & Derivative Logic 
        for(int idx = tid; idx < 4096; idx += 128) {
            int r = idx / 64;
            int c = idx % 64;
            int glob_r = global_q_row + r;
            int glob_c = global_k_row + c;
            
            float p = 0.0f;
            float ds = 0.0f;
            
            if (glob_r < seq_len && glob_c < seq_len && glob_c <= glob_r) {
                float s = S_smem[idx];
                float dp = dP_smem[idx];
                float scaled_s = s * 0.08838834764f; // S / sqrt(128)
                p = expf(scaled_s - L_smem[r]);
                ds = p * (dp - D_smem[r]) * 0.08838834764f;
            }
            
            P_smem[idx] = __float2bfloat16(p);
            dS_smem[idx] = __float2bfloat16(ds);
        }
        __syncthreads();
        
        // 6. Tensor Core Pass: dQ += dS K
        for(int k = 0; k < 64; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> ds_frag[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> k_frag_b[4];
            for(int i = 0; i < 2; i++) wmma::load_matrix_sync(ds_frag[i], &dS_smem[(Wy * 32 + i * 16) * 64 + k], 64);
            for(int j = 0; j < 4; j++) wmma::load_matrix_sync(k_frag_b[j], &K_smem[k * 128 + (Wx * 64 + j * 16)], 128);
            for(int i = 0; i < 2; i++) for(int j = 0; j < 4; j++) wmma::mma_sync(dQ_frag[i][j], ds_frag[i], k_frag_b[j], dQ_frag[i][j]);
        }
        
        // 7. Tensor Core Pass: dK = dS^T Q
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> dK_frag[2][4];
        for(int i = 0; i < 2; i++) for(int j = 0; j < 4; j++) wmma::fill_fragment(dK_frag[i][j], 0.0f);
        for(int k = 0; k < 64; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> ds_t_frag[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> q_frag_b[4];
            for(int i = 0; i < 2; i++) wmma::load_matrix_sync(ds_t_frag[i], &dS_smem[k * 64 + (Wy * 32 + i * 16)], 64);
            for(int j = 0; j < 4; j++) wmma::load_matrix_sync(q_frag_b[j], &Q_smem[k * 128 + (Wx * 64 + j * 16)], 128);
            for(int i = 0; i < 2; i++) for(int j = 0; j < 4; j++) wmma::mma_sync(dK_frag[i][j], ds_t_frag[i], q_frag_b[j], dK_frag[i][j]);
        }
        
        // 8. Tensor Core Pass: dV = P^T dO
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> dV_frag[2][4];
        for(int i = 0; i < 2; i++) for(int j = 0; j < 4; j++) wmma::fill_fragment(dV_frag[i][j], 0.0f);
        for(int k = 0; k < 64; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> p_t_frag[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> do_frag_b[4];
            for(int i = 0; i < 2; i++) wmma::load_matrix_sync(p_t_frag[i], &P_smem[k * 64 + (Wy * 32 + i * 16)], 64);
            for(int j = 0; j < 4; j++) wmma::load_matrix_sync(do_frag_b[j], &dO_smem[k * 128 + (Wx * 64 + j * 16)], 128);
            for(int i = 0; i < 2; i++) for(int j = 0; j < 4; j++) wmma::mma_sync(dV_frag[i][j], p_t_frag[i], do_frag_b[j], dV_frag[i][j]);
        }
        
        for(int i = 0; i < 2; i++) {
            for(int j = 0; j < 4; j++) {
                wmma::store_matrix_sync(&dK_smem_f[(Wy * 32 + i * 16) * 128 + (Wx * 64 + j * 16)], dK_frag[i][j], 128, wmma::mem_row_major);
                wmma::store_matrix_sync(&dV_smem_f[(Wy * 32 + i * 16) * 128 + (Wx * 64 + j * 16)], dV_frag[i][j], 128, wmma::mem_row_major);
            }
        }
        __syncthreads();
        
        // 9. Scaled BF16 Atomics mapping back cleanly into GMEM layout mappings
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
                
                __nv_bfloat162* global_dK_ptr = (__nv_bfloat162*)&dK[batch_offset + (global_k_row + r) * 128 + c];
                __nv_bfloat162* global_dV_ptr = (__nv_bfloat162*)&dV[batch_offset + (global_k_row + r) * 128 + c];
                
                atomicAdd(global_dK_ptr, dk_bf);
                atomicAdd(global_dV_ptr, dv_bf);
            }
        }
        __syncthreads();
    } // End of Column Block loops J (Causal bounds handled logically)
    
    // Write fully integrated dQ values per Row Block I
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 4; j++) {
            // Repurpose SMEM layout efficiently after operations
            wmma::store_matrix_sync(&dK_smem_f[(Wy * 32 + i * 16) * 128 + (Wx * 64 + j * 16)], dQ_frag[i][j], 128, wmma::mem_row_major);
        }
    }
    __syncthreads();
    
    for(int idx = tid; idx < 4096; idx += 128) {
        int r = idx / 64;
        int c = (idx % 64) * 2;
        if (global_q_row + r < seq_len) {
            float dq0 = dK_smem_f[r * 128 + c];
            float dq1 = dK_smem_f[r * 128 + c + 1];
            __nv_bfloat162 dq_bf = __floats2bfloat162_rn(dq0, dq1);
            
            uint32_t* global_dQ_ptr = (uint32_t*)&dQ[batch_offset + (global_q_row + r) * 128 + c];
            *global_dQ_ptr = *(uint32_t*)&dq_bf;
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
    int64_t d = 128;
    
    // Hardware atomics are additive, required pre-zeroing logic
    CUDA_CHECK(cudaMemsetAsync(dK.data_ptr(), 0, B * H * S * d * sizeof(uint16_t), stream));
    CUDA_CHECK(cudaMemsetAsync(dV.data_ptr(), 0, B * H * S * d * sizeof(uint16_t), stream));

    dim3 grid((S + 63) / 64, H, B);
    dim3 block(128);

    int smem_size = 200000; 
    CUDA_CHECK(cudaFuncSetAttribute(sdpa_bwd_kernel_wmma, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    sdpa_bwd_kernel_wmma<<<grid, block, smem_size, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(O.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        S, B, H
    );
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_sdpa_causal::run);

}