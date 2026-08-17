#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>
#include <cutlass/wmma/wmma.h>

using namespace cutlass::wmma;

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

// ---- Attention Kernel ----
__global__ void attention_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S, float scale) 
{
    // Dynamic register allocation (requesting 256 regs/thread)
    setmaxnreg_inc_sync_fn<256>();

    int b_h = blockIdx.y;
    int num_q_blocks = (S + 63) / 64;
    int j = blockIdx.x;
    
    if (j >= num_q_blocks) return;
    
    // Allocate 40KB SMEM explicitly via extern to avoid static limits
    extern __shared__ __align__(128) char smem_pool[];
    __nv_bfloat16* smem_Q = (__nv_bfloat16*)smem_pool;                
    __nv_bfloat16* smem_K = (__nv_bfloat16*)(smem_pool + 8192);       
    __nv_bfloat16* smem_V0 = (__nv_bfloat16*)(smem_pool + 16384);     
    __nv_bfloat16* smem_V1 = (__nv_bfloat16*)(smem_pool + 24576);     
    float* p_val = (float*)(smem_pool + 32768);                      
    
    const int D = 128;
    const __nv_bfloat16* Q_bh = Q + b_h * S * D;
    const __nv_bfloat16* K_bh = K + b_h * S * D;
    const __nv_bfloat16* V_bh = V + b_h * S * D;
    __nv_bfloat16* O_bh = O + b_h * S * D;
    float* LSE_bh = LSE + b_h * S;

    // Prolific double-buffering fetch across entire sequence for Q
    for (int step = 0; step < num_q_blocks; step++) {
        uint4* q_ptr0 = (uint4*)(Q_bh + step*64*D + (threadIdx.x / 2) * D + ((threadIdx.x % 2) * 32) + (threadIdx.x % 32) * 8);
        uint4* q_ptr1 = (uint4*)(Q_bh + step*64*D + 64 + (threadIdx.x / 2) * D + ((threadIdx.x % 2) * 32) + (threadIdx.x % 32) * 8);
        *(uint4*)&smem_Q[(step * 64 + threadIdx.x / 2) * 64 + (threadIdx.x % 2) * 32 + (threadIdx.x % 32) * 8] = *q_ptr0;
        *(uint4*)&smem_Q[(step * 64 + threadIdx.x / 2) * 64 + (threadIdx.x % 2) * 32 + (threadIdx.x % 32) * 8 + 64] = *q_ptr1;
    }
    
    for (int step = 0; step < num_q_blocks; step++) {
        uint4* k_ptr0 = (uint4*)(K_bh + step*64*D + (threadIdx.x / 2) * D + ((threadIdx.x % 2) * 32) + (threadIdx.x % 32) * 8);
        uint4* k_ptr1 = (uint4*)(K_bh + step*64*D + 64 + (threadIdx.x / 2) * D + ((threadIdx.x % 2) * 32) + (threadIdx.x % 32) * 8);
        *(uint4*)&smem_K[(step * 64 + threadIdx.x / 2) * 64 + (threadIdx.x % 2) * 32 + (threadIdx.x % 32) * 8] = *k_ptr0;
        *(uint4*)&smem_K[(step * 64 + threadIdx.x / 2) * 64 + (threadIdx.x % 2) * 32 + (threadIdx.x % 32) * 8 + 64] = *k_ptr1;
    }
    
    for (int step = 0; step < num_q_blocks; step++) {
        uint4* v_ptr0 = (uint4*)(V_bh + step*64*D + (threadIdx.x / 2) * D + ((threadIdx.x % 2) * 32) + (threadIdx.x % 32) * 8);
        uint4* v_ptr1 = (uint4*)(V_bh + step*64*D + 64 + (threadIdx.x / 2) * D + ((threadIdx.x % 2) * 32) + (threadIdx.x % 32) * 8);
        *(uint4*)&smem_V0[(step * 64 + threadIdx.x / 2) * 64 + (threadIdx.x % 2) * 32 + (threadIdx.x % 32) * 8] = *v_ptr0;
        *(uint4*)&smem_V1[(step * 64 + threadIdx.x / 2) * 64 + (threadIdx.x % 2) * 32 + (threadIdx.x % 32) * 8] = *v_ptr1;
    }
    
    __syncthreads();

    using Q_frag = fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major>;
    using K_frag = fragment<matrix_b, 16, 16, 16, __nv_bfloat16, row_major>;
    using S_frag = fragment<matrix_d, 16, 16, 16, float>;
    using P_frag = fragment<matrix_a, 16, 16, 16, float, row_major>;
    using V_frag = fragment<matrix_b, 16, 16, 16, __nv_bfloat16, row_major>;
    using O_frag = fragment<matrix_d, 16, 16, 16, float>;

    int warp_id = threadIdx.x / 32;
    
    float global_max[2] = {-INFINITY, -INFINITY};
    float global_sum[2] = {0, 0};
    
    O_frag my_O_result_first_half[4];
    O_frag my_O_result_second_half[4];
    for (int i = 0; i < 4; ++i) {
        my_O_result_first_half[i].fill(0);
        my_O_result_second_half[i].fill(0);
    }

    for (int query_step = 0; query_step < num_q_blocks; ++query_step) {
        
        float rescale_o[2] = {1.0f, 1.0f};
        if (global_sum[0] > 0 || global_sum[1] > 0) {
            rescale_o[0] = __expf(global_max[0]);
            rescale_o[1] = __expf(global_max[1]);
        }
        
        float current_max[2] = {-INFINITY, -INFINITY};
        float current_sum[2] = {0, 0};
        
        for (int i = 0; i < 4; ++i) {
            for (int k = 0; k < 8; ++k) {
                my_O_result_first_half[i][k] *= rescale_o[(i/2)%2];
                my_O_result_second_half[i][k] *= rescale_o[(i/2)%2];
            }
        }
        
        for (int i = 0; i < 4; ++i) {
            for (int k = 0; k < 8; ++k) {
                int row_idx = (i / 2) % 2;
                current_sum[row_idx] *= rescale_o[row_idx];
            }
        }
        
        global_max[0] = 0; 
        global_max[1] = 0; 
        
        S_frag my_S_result[4];
        for (int i = 0; i < 4; ++i) my_S_result[i].fill(0);

        for (int key_step = 0; key_step <= query_step; ++key_step) {
            
            Q_frag my_Q_frag[4];
            K_frag my_K_T_frag[4]; // Stored as K_T directly to save explicit SMEM transpose ops
            for (int i = 0; i < 4; ++i) {
                load_matrix_sync(my_Q_frag[i], &smem_Q[(query_step * 64 + warp_id * 16) * 128 + i * 16], 128);
                load_matrix_sync(my_K_T_frag[i], &smem_K[(key_step * 64) * 128 + warp_id * 16 + i * 16], 128);
            }
            for (int i = 0; i < 4; ++i) {
                mma_sync(my_S_result[i], my_Q_frag[i], my_K_T_frag[i], my_S_result[i], membar::relaxed);
            }
            
            Q_frag my_Q_frag_s1[4];
            K_frag my_K_T_frag_s1[4]; // K_T for the second 64x64 block
            for (int i = 0; i < 4; ++i) {
                load_matrix_sync(my_Q_frag_s1[i], &smem_Q[(query_step * 64 + warp_id * 16) * 128 + 64 + i * 16], 128);
                load_matrix_sync(my_K_T_frag_s1[i], &smem_K[(key_step * 64) * 128 + 64 + warp_id * 16 + i * 16], 128);
            }
            for (int i = 0; i < 4; ++i) {
                mma_sync(my_S_result[i], my_Q_frag_s1[i], my_K_T_frag_s1[i], my_S_result[i], membar::relaxed);
            }
            
            for (int idx = 0; idx < 4; ++idx) {
                int row_idx = (idx / 2) % 2;
                int col_base = (key_step * 64) + (idx % 2) * 16;
                
                float m = -INFINITY;
                for (int k = 0; k < 8; ++k) {
                    int global_col = col_base + ((idx / 2) * 8 + k);
                    int global_row = query_step * 64 + (warp_id * 16 + (idx % 2) * 8 + (k / 4) * 4 + (k % 4));
                    if (global_col > global_row) {
                        my_S_result[idx][k] = -INFINITY;
                    } else {
                        my_S_result[idx][k] *= scale;
                    }
                    m = fmaxf(m, my_S_result[idx][k]);
                }
                
                current_max[row_idx] = fmaxf(current_max[row_idx], m);
                current_max[row_idx] = fmaxf(current_max[row_idx], __shfl_xor_sync(0xffffffff, current_max[row_idx], 1));
                current_max[row_idx] = fmaxf(current_max[row_idx], __shfl_xor_sync(0xffffffff, current_max[row_idx], 2));
                
                float new_max = fmaxf(global_max[row_idx], current_max[row_idx]);
                float r_o = __expf(global_max[row_idx] - new_max);
                float r_s = r_o;
                
                for (int k = 0; k < 8; ++k) {
                    my_O_result_first_half[idx][k] *= r_o;
                    my_O_result_second_half[idx][k] *= r_o;
                }
                current_sum[row_idx] *= r_s;
                current_sum[row_idx] += __shfl_xor_sync(0xffffffff, current_sum[row_idx], 1);
                current_sum[row_idx] += __shfl_xor_sync(0xffffffff, current_sum[row_idx], 2);
                
                float local_s_sum = 0;
                for (int k = 0; k < 8; ++k) {
                    my_S_result[idx][k] = __expf(my_S_result[idx][k] - new_max);
                    local_s_sum += my_S_result[idx][k];
                }
                
                current_sum[row_idx] += local_s_sum;
                current_sum[row_idx] += __shfl_xor_sync(0xffffffff, current_sum[row_idx], 1);
                current_sum[row_idx] += __shfl_xor_sync(0xffffffff, current_sum[row_idx], 2);
                
                global_max[row_idx] = new_max;
            }
            
            P_frag my_P_frag[4];
            for (int i = 0; i < 4; ++i) {
                my_P_frag[i] = my_S_result[i];
            }
            
            V_frag my_V_frag[4];
            for (int i = 0; i < 4; ++i) {
                load_matrix_sync(my_V_frag[i], &smem_V0[(key_step * 64) * 64 + (warp_id * 16) + i * 16], 64);
                mma_sync(my_O_result_first_half[i], my_P_frag[i], my_V_frag[i], my_O_result_first_half[i], membar::relaxed);
            }
            
            V_frag my_V_frag_s1[4];
            for (int i = 0; i < 4; ++i) {
                load_matrix_sync(my_V_frag_s1[i], &smem_V1[(key_step * 64) * 64 + (warp_id * 16) + i * 16], 64);
                mma_sync(my_O_result_second_half[i], my_P_frag[i], my_V_frag_s1[i], my_O_result_second_half[i], membar::relaxed);
            }
        }
        
        for (int idx = 0; idx < 4; ++idx) {
            int row_idx = (idx / 2) % 2;
            global_sum[row_idx] = current_sum[row_idx];
        }
        
        for (int idx = 0; idx < 4; ++idx) {
            int row_idx = (idx / 2) % 2;
            int col_base = (idx % 2) * 16;
            int global_row = query_step * 64 + (warp_id * 16 + (idx % 2) * 8);
            
            if (global_row < S) {
                float2 final_sum = {global_sum[row_idx], global_sum[row_idx]};
                float2 lse = {global_max[row_idx], final_sum.y};
                LSE_bh[query_step * 64 + global_row] = lse.x + __logf(lse.y);
                
                for (int k = 0; k < 8; ++k) {
                    int global_col0 = col_base + ((idx / 2) * 8 + k);
                    int global_col1 = col_base + ((idx / 2) * 8 + k);
                    
                    float2 o0 = __halff22float2(__bfloat1622float2(*reinterpret_cast<uint32_t*>(&O_bh[(query_step * 64 + global_row) * D + global_col0])));
                    float2 o1 = __halff22float2(__bfloat1622float2(*reinterpret_cast<uint32_t*>(&O_bh[(query_step * 64 + global_row) * D + global_col1 + 64])));
                    
                    float2 out0 = {(float)my_O_result_first_half[idx][k] / final_sum.y, o0.y};
                    float2 out1 = {(float)my_O_result_second_half[idx][k] / final_sum.y, o1.y};
                    
                    *reinterpret_cast<uint32_t*>(&O_bh[(query_step * 64 + global_row) * D + global_col0]) = __float22bfloat16(__float2half(out0.x));
                    *reinterpret_cast<uint32_t*>(&O_bh[(query_step * 64 + global_row) * D + global_col1 + 64]) = __float22bfloat16(__float2half(out1.x));
                }
            }
        }
        
        for (int i = 0; i < 4; ++i) {
            my_O_result_first_half[i].fill(0);
            my_O_result_second_half[i].fill(0);
        }
    }
}

namespace tvm_ffi_example_cuda {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3); 
    
    if (S == 0) return;
    
    float scale = 1.0f / sqrtf((float)D);
    
    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());
    
    int num_q_blocks = (S + 63) / 64;
    dim3 grid(num_q_blocks, B * H);
    dim3 block(128);
    
    int smem_size = 40 * 1024;
    CUDA_CHECK(cudaFuncSetAttribute(
        attention_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    ));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    attention_kernel<<<grid, block, smem_size, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data, B, H, S, scale
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_example_cuda