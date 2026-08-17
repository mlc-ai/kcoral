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

__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 256;" ::: "memory");
}

__device__ __forceinline__ bool elect_one_sync_fn() {
    uint32_t pred;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "elect.sync _|p, 0xFFFFFFFF;\n"
        "selp.b32 %0, 1, 0, p;\n"
        "}\n"
        : "=r"(pred));
    return pred != 0;
}

// Employs hopper asynchronous multi-dispatch mechanism decoupled from tensor-maps
__device__ __forceinline__ void cp_async_bulk_1d(void* smem, const void* gmem, int bytes, uint64_t* mbar) {
    uint32_t smem_ptr = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t mbar_ptr = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
        :: "r"(smem_ptr), "l"(gmem), "r"(bytes), "r"(mbar_ptr) : "memory"
    );
}

__global__ void sdpa_bwd_kernel_bulk_wmma(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    float* __restrict__ dK_fp32,
    float* __restrict__ dV_fp32,
    int seq_len
) {
    setmaxnreg_inc_sync_fn();

    extern __shared__ uint4 smem[];
    uint8_t* smem_base = (uint8_t*)smem;
    uint64_t* mbar = (uint64_t*)smem_base;
    uint8_t* tma_base = smem_base + 16;
    
    // Extrema-optimized dynamic overlap structure ensuring 100% capacity within Hopper's limits 
    __nv_bfloat16* Q_smem = (__nv_bfloat16*)tma_base;                // 16KB
    __nv_bfloat16* dO_smem = Q_smem + 64 * 128;                      // 16KB
    __nv_bfloat16* K_smem = dO_smem + 64 * 128;                      // 16KB
    __nv_bfloat16* V_smem = K_smem + 64 * 128;                       // 16KB
    __nv_bfloat16* O_smem = V_smem + 64 * 128;                       // 16KB
    
    float* S_smem = (float*)(O_smem + 64 * 128);                     // 16KB
    float* dP_smem = S_smem + 64 * 64;                               // 16KB
    __nv_bfloat16* P_smem = (__nv_bfloat16*)(dP_smem + 64 * 64);     // 8KB
    __nv_bfloat16* dS_smem = P_smem + 64 * 64;                       // 8KB
    float* D_smem = (float*)(dS_smem + 64 * 64);                     // 256B
    float* L_smem = D_smem + 64;                                     // 256B
    
    int block_i = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int Wy = warp_id / 2;
    int Wx = warp_id % 2;
    
    int global_q_row = block_i * 64;
    int batch_offset = (b * gridDim.y + h) * seq_len * 128;
    int l_offset = (b * gridDim.y + h) * seq_len;
    
    if (elect_one_sync_fn() && warp_id == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(mbar)), "r"(128));
    }
    __syncthreads();
    
    int phase = 0;
    
    int valid_q_rows = seq_len - global_q_row;
    if (valid_q_rows < 0) valid_q_rows = 0;
    if (valid_q_rows > 64) valid_q_rows = 64;
    
    if (tid == 0) {
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" 
            :: "r"((uint32_t)__cvta_generic_to_shared(mbar)), "r"(valid_q_rows * 256 * 3));
    } else {
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"((uint32_t)__cvta_generic_to_shared(mbar)));
    }
    
    if (tid < valid_q_rows) {
        cp_async_bulk_1d(&Q_smem[tid * 128], &Q[batch_offset + (global_q_row + tid) * 128], 256, mbar);
        cp_async_bulk_1d(&dO_smem[tid * 128], &dO[batch_offset + (global_q_row + tid) * 128], 256, mbar);
        cp_async_bulk_1d(&O_smem[tid * 128], &O[batch_offset + (global_q_row + tid) * 128], 256, mbar);
    }
    
    if (tid < 64) {
        L_smem[tid] = (global_q_row + tid < seq_len) ? L[l_offset + global_q_row + tid] : 0.0f;
    }
    
    asm volatile(
        "{\n.reg .pred P;\nWAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(mbar)), "r"(phase));
    phase ^= 1;
    
    if (tid >= valid_q_rows && tid < 64) {
        for (int i = 0; i < 64; i++) {
            ((uint32_t*)Q_smem)[tid * 64 + i] = 0;
            ((uint32_t*)dO_smem)[tid * 64 + i] = 0;
            ((uint32_t*)O_smem)[tid * 64 + i] = 0;
        }
    }
    __syncthreads();
    
    if (tid < 64) {
        float d_val = 0.0f;
        for (int k = 0; k < 128; k++) {
            float do_val = __bfloat162float(dO_smem[tid * 128 + k]);
            float o_val  = __bfloat162float(O_smem[tid * 128 + k]);
            d_val += do_val * o_val;
        }
        D_smem[tid] = d_val;
    }
    __syncthreads();
    
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dQ_frag[2][4];
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(dQ_frag[i][j], 0.0f);
        }
    }
    
    for (int block_j = 0; block_j <= block_i; block_j++) {
        int global_k_row = block_j * 64;
        
        int valid_k_rows = seq_len - global_k_row;
        if (valid_k_rows < 0) valid_k_rows = 0;
        if (valid_k_rows > 64) valid_k_rows = 64;
        
        if (tid == 0) {
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" 
                :: "r"((uint32_t)__cvta_generic_to_shared(mbar)), "r"(valid_k_rows * 256 * 2));
        } else {
            asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"((uint32_t)__cvta_generic_to_shared(mbar)));
        }
        
        if (tid < valid_k_rows) {
            cp_async_bulk_1d(&K_smem[tid * 128], &K[batch_offset + (global_k_row + tid) * 128], 256, mbar);
            cp_async_bulk_1d(&V_smem[tid * 128], &V[batch_offset + (global_k_row + tid) * 128], 256, mbar);
        }
        
        asm volatile(
            "{\n.reg .pred P;\nWAIT_%=:\n"
            "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
            "@!P bra WAIT_%=;\n}\n"
            :: "r"((uint32_t)__cvta_generic_to_shared(mbar)), "r"(phase));
        phase ^= 1;
        
        if (tid >= valid_k_rows && tid < 64) {
            for (int i = 0; i < 64; i++) {
                ((uint32_t*)K_smem)[tid * 64 + i] = 0;
                ((uint32_t*)V_smem)[tid * 64 + i] = 0;
            }
        }
        __syncthreads();
        
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> S_frag[2][2];
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> dP_frag[2][2];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                wmma::fill_fragment(S_frag[i][j], 0.0f);
                wmma::fill_fragment(dP_frag[i][j], 0.0f);
            }
        }
        
        for (int k = 0; k < 128; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> q_frag[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> k_frag[2];
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> do_frag[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> v_frag[2];
            
            for (int i = 0; i < 2; i++) wmma::load_matrix_sync(q_frag[i], &Q_smem[(Wy * 32 + i * 16) * 128 + k], 128);
            for (int j = 0; j < 2; j++) wmma::load_matrix_sync(k_frag[j], &K_smem[(Wx * 32 + j * 16) * 128 + k], 128);
            for (int i = 0; i < 2; i++) wmma::load_matrix_sync(do_frag[i], &dO_smem[(Wy * 32 + i * 16) * 128 + k], 128);
            for (int j = 0; j < 2; j++) wmma::load_matrix_sync(v_frag[j], &V_smem[(Wx * 32 + j * 16) * 128 + k], 128);
            
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    wmma::mma_sync(S_frag[i][j], q_frag[i], k_frag[j], S_frag[i][j]);
                    wmma::mma_sync(dP_frag[i][j], do_frag[i], v_frag[j], dP_frag[i][j]);
                }
            }
        }
        
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                wmma::store_matrix_sync(&S_smem[(Wy * 32 + i * 16) * 64 + (Wx * 32 + j * 16)], S_frag[i][j], 64, wmma::mem_row_major);
                wmma::store_matrix_sync(&dP_smem[(Wy * 32 + i * 16) * 64 + (Wx * 32 + j * 16)], dP_frag[i][j], 64, wmma::mem_row_major);
            }
        }
        __syncthreads();
        
        for (int idx = tid; idx < 4096; idx += 128) {
            int r = idx / 64;
            int c = idx % 64;
            int glob_r = global_q_row + r;
            int glob_c = global_k_row + c;
            
            float p = 0.0f;
            float ds = 0.0f;
            
            if (glob_r < seq_len && glob_c < seq_len && glob_c <= glob_r) {
                float s = S_smem[idx];
                float dp = dP_smem[idx];
                float scaled_s = s * 0.088388347648f; 
                p = expf(scaled_s - L_smem[r]);
                ds = p * (dp - D_smem[r]) * 0.088388347648f;
            }
            
            P_smem[idx] = __float2bfloat16(p);
            dS_smem[idx] = __float2bfloat16(ds);
        }
        __syncthreads();
        
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> dK_frag[2][4];
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> dV_frag[2][4];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 4; j++) {
                wmma::fill_fragment(dK_frag[i][j], 0.0f);
                wmma::fill_fragment(dV_frag[i][j], 0.0f);
            }
        }
        
        for (int k = 0; k < 64; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> ds_frag[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> k_frag_b[4];
            
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> ds_t_frag[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> q_frag_b[4];
            
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> p_t_frag[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> do_frag_b[4];
            
            for (int i = 0; i < 2; i++) {
                wmma::load_matrix_sync(ds_frag[i], &dS_smem[(Wy * 32 + i * 16) * 64 + k], 64);
                wmma::load_matrix_sync(ds_t_frag[i], &dS_smem[k * 64 + (Wy * 32 + i * 16)], 64);
                wmma::load_matrix_sync(p_t_frag[i], &P_smem[k * 64 + (Wy * 32 + i * 16)], 64);
            }
            for (int j = 0; j < 4; j++) {
                wmma::load_matrix_sync(k_frag_b[j], &K_smem[k * 128 + (Wx * 64 + j * 16)], 128);
                wmma::load_matrix_sync(q_frag_b[j], &Q_smem[k * 128 + (Wx * 64 + j * 16)], 128);
                wmma::load_matrix_sync(do_frag_b[j], &dO_smem[k * 128 + (Wx * 64 + j * 16)], 128);
            }
            
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 4; j++) {
                    wmma::mma_sync(dQ_frag[i][j], ds_frag[i], k_frag_b[j], dQ_frag[i][j]);
                    wmma::mma_sync(dK_frag[i][j], ds_t_frag[i], q_frag_b[j], dK_frag[i][j]);
                    wmma::mma_sync(dV_frag[i][j], p_t_frag[i], do_frag_b[j], dV_frag[i][j]);
                }
            }
        }
        
        float* dK_smem_f = (float*)K_smem; 
        float* dV_smem_f = S_smem;         
        
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 4; j++) {
                wmma::store_matrix_sync(&dK_smem_f[(Wy * 32 + i * 16) * 128 + (Wx * 64 + j * 16)], dK_frag[i][j], 128, wmma::mem_row_major);
                wmma::store_matrix_sync(&dV_smem_f[(Wy * 32 + i * 16) * 128 + (Wx * 64 + j * 16)], dV_frag[i][j], 128, wmma::mem_row_major);
            }
        }
        __syncthreads();
        
        for (int idx = tid; idx < 4096; idx += 128) {
            int r = idx / 64;
            int c = (idx % 64) * 2;
            if (global_k_row + r < seq_len) {
                float dk0 = dK_smem_f[r * 128 + c];
                float dk1 = dK_smem_f[r * 128 + c + 1];
                float dv0 = dV_smem_f[r * 128 + c];
                float dv1 = dV_smem_f[r * 128 + c + 1];
                
                float* global_dK_ptr = &dK_fp32[batch_offset + (global_k_row + r) * 128 + c];
                float* global_dV_ptr = &dV_fp32[batch_offset + (global_k_row + r) * 128 + c];
                
                atomicAdd(global_dK_ptr, dk0);
                atomicAdd(global_dK_ptr + 1, dk1);
                atomicAdd(global_dV_ptr, dv0);
                atomicAdd(global_dV_ptr + 1, dv1);
            }
        }
        __syncthreads(); 
    } 
    
    float* dQ_smem_f = (float*)K_smem; 
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            wmma::store_matrix_sync(&dQ_smem_f[(Wy * 32 + i * 16) * 128 + (Wx * 64 + j * 16)], dQ_frag[i][j], 128, wmma::mem_row_major);
        }
    }
    __syncthreads();
    
    for (int idx = tid; idx < 4096; idx += 128) {
        int r = idx / 64;
        int c = (idx % 64) * 2;
        if (global_q_row + r < seq_len) {
            float dq0 = dQ_smem_f[r * 128 + c];
            float dq1 = dQ_smem_f[r * 128 + c + 1];
            __nv_bfloat162 dq_bf = __floats2bfloat162_rn(dq0, dq1);
            
            uint32_t* global_dQ_ptr = (uint32_t*)&dQ[batch_offset + (global_q_row + r) * 128 + c];
            *global_dQ_ptr = *(uint32_t*)&dq_bf;
        }
    }
}

__global__ void convert_fp32_to_bf16(const float* __restrict__ src_dK, const float* __restrict__ src_dV, 
                                     __nv_bfloat16* __restrict__ dst_dK, __nv_bfloat16* __restrict__ dst_dV, 
                                     int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        dst_dK[idx] = __float2bfloat16(src_dK[idx]);
        dst_dV[idx] = __float2bfloat16(src_dV[idx]);
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
    int total_elements = B * H * S * d;
    
    // Asynchronous Workspace allocating guarantees catastrophic precision loss avoidance
    float* dK_fp32;
    float* dV_fp32;
    CUDA_CHECK(cudaMallocAsync(&dK_fp32, total_elements * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&dV_fp32, total_elements * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(dK_fp32, 0, total_elements * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(dV_fp32, 0, total_elements * sizeof(float), stream));

    dim3 grid((S + 63) / 64, H, B);
    dim3 block(128);

    int smem_size = 132000;
    CUDA_CHECK(cudaFuncSetAttribute(sdpa_bwd_kernel_bulk_wmma, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    sdpa_bwd_kernel_bulk_wmma<<<grid, block, smem_size, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(O.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        dK_fp32, dV_fp32,
        S
    );
    
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    convert_fp32_to_bf16<<<blocks, threads, 0, stream>>>(dK_fp32, dV_fp32, 
                                                         static_cast<__nv_bfloat16*>(dK.data_ptr()), 
                                                         static_cast<__nv_bfloat16*>(dV.data_ptr()), 
                                                         total_elements);
    
    CUDA_CHECK(cudaFreeAsync(dK_fp32, stream));
    CUDA_CHECK(cudaFreeAsync(dV_fp32, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_sdpa_causal::run);

}