#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <mma.h>
#include <cuda.h>
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

namespace tvm_ffi_mha {

__global__ void mha_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S_seq, int D
) {
    int bx = blockIdx.x; // B * H
    int batch = bx / H;
    int head = bx % H;
    int q_seq_start = blockIdx.y * 64;

    if (q_seq_start >= S_seq) return;

    int q_seq_len = min(64, S_seq - q_seq_start);

    int64_t batch_head_offset = (int64_t)batch * H * S_seq * 128 + (int64_t)head * S_seq * 128;
    const __nv_bfloat16* Q_ptr = Q + batch_head_offset + (int64_t)q_seq_start * 128;
    const __nv_bfloat16* K_ptr_base = K + batch_head_offset;
    const __nv_bfloat16* V_ptr_base = V + batch_head_offset;

    extern __shared__ char smem[];
    __nv_bfloat16* Q_smem = (__nv_bfloat16*)smem;
    __nv_bfloat16* K_smem = Q_smem + 64 * 128;
    __nv_bfloat16* V_smem = K_smem + 64 * 128;
    float* S_smem = (float*)(V_smem + 64 * 128); 
    float* O_smem = S_smem + 64 * 64; 
    float* m_max_smem = O_smem + 64 * 128; 
    float* l_sum_smem = m_max_smem + 64; 
    __nv_bfloat16* P_bf16_smem = (__nv_bfloat16*)S_smem; 

    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    int tid = warp_id * 32 + lane_id;

    for (int i = tid; i < 64; i += 256) {
        m_max_smem[i] = -1e38f;
        l_sum_smem[i] = 0.0f;
    }
    
    float4* O_smem_f4 = (float4*)O_smem;
    for (int i = tid; i < 2048; i += 256) {
        float4 zero;
        zero.x = 0; zero.y = 0; zero.z = 0; zero.w = 0;
        O_smem_f4[i] = zero;
    }
    __syncthreads();

    float4* Q_smem_f4 = (float4*)Q_smem;
    const float4* Q_ptr_f4 = (const float4*)Q_ptr;
    for (int i = tid; i < 64 * 32; i += 256) {
        int row = i / 32;
        if (row < q_seq_len) {
            Q_smem_f4[i] = Q_ptr_f4[i];
        } else {
            float4 zero;
            zero.x = 0; zero.y = 0; zero.z = 0; zero.w = 0;
            Q_smem_f4[i] = zero;
        }
    }
    __syncthreads();

    int num_kv_blocks = (S_seq + 63) / 64;
    float scale = 0.0883883476f; // 1.0 / sqrt(128)

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int kv_seq_start = kv_block * 64;
        int kv_seq_len = min(64, S_seq - kv_seq_start);

        const float4* K_ptr_f4 = (const float4*)(K_ptr_base + kv_seq_start * 128);
        const float4* V_ptr_f4 = (const float4*)(V_ptr_base + kv_seq_start * 128);
        float4* K_smem_f4 = (float4*)K_smem;
        float4* V_smem_f4 = (float4*)V_smem;
        
        for (int i = tid; i < 64 * 32; i += 256) {
            int row = i / 32;
            if (row < kv_seq_len) {
                K_smem_f4[i] = K_ptr_f4[i];
                V_smem_f4[i] = V_ptr_f4[i];
            } else {
                float4 zero;
                zero.x = 0; zero.y = 0; zero.z = 0; zero.w = 0;
                K_smem_f4[i] = zero;
                V_smem_f4[i] = zero;
            }
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> frag_Q;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> frag_K[2];
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_S[2];

        wmma::fill_fragment(frag_S[0], 0.0f);
        wmma::fill_fragment(frag_S[1], 0.0f);

        int warp_row_S = warp_id / 2; 
        int warp_col_S = warp_id % 2; 

        for (int k_step = 0; k_step < 8; k_step++) {
            int k_idx = k_step * 16;
            wmma::load_matrix_sync(frag_Q, &Q_smem[warp_row_S * 16 * 128 + k_idx], 128);
            wmma::load_matrix_sync(frag_K[0], &K_smem[(warp_col_S * 32 + 0) * 128 + k_idx], 128);
            wmma::load_matrix_sync(frag_K[1], &K_smem[(warp_col_S * 32 + 16) * 128 + k_idx], 128);
            
            wmma::mma_sync(frag_S[0], frag_Q, frag_K[0], frag_S[0]);
            wmma::mma_sync(frag_S[1], frag_Q, frag_K[1], frag_S[1]);
        }

        wmma::store_matrix_sync(&S_smem[warp_row_S * 16 * 64 + warp_col_S * 32 + 0], frag_S[0], 64, wmma::mem_row_major);
        wmma::store_matrix_sync(&S_smem[warp_row_S * 16 * 64 + warp_col_S * 32 + 16], frag_S[1], 64, wmma::mem_row_major);
        __syncthreads();

        int row_start = warp_id * 8;
        
        for (int r = 0; r < 8; r++) {
            int row = row_start + r;
            
            float local_max = -1e38f;
            float v1 = S_smem[row * 64 + lane_id];
            float v2 = S_smem[row * 64 + lane_id + 32];
            
            if (lane_id >= kv_seq_len) v1 = -1e38f;
            if (lane_id + 32 >= kv_seq_len) v2 = -1e38f;
            
            if (row >= q_seq_len) {
                v1 = -1e38f;
                v2 = -1e38f;
            }

            if (v1 > -1e37f) v1 *= scale;
            if (v2 > -1e37f) v2 *= scale;
            
            S_smem[row * 64 + lane_id] = v1;
            S_smem[row * 64 + lane_id + 32] = v2;
            
            local_max = max(v1, v2);
            
            #pragma unroll
            for(int offset = 16; offset > 0; offset /= 2)
                local_max = max(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
            
            float row_max = __shfl_sync(0xffffffff, local_max, 0);
            
            float m_old = m_max_smem[row];
            float l_old = l_sum_smem[row];
            float m_new = max(m_old, row_max);
            float exp_diff = expf(m_old - m_new);
            
            float local_sum = 0.0f;
            float p1 = 0.0f, p2 = 0.0f;
            if (v1 > -1e37f) p1 = expf(v1 - m_new);
            if (v2 > -1e37f) p2 = expf(v2 - m_new);
            
            P_bf16_smem[row * 64 + lane_id] = __float2bfloat16(p1);
            P_bf16_smem[row * 64 + lane_id + 32] = __float2bfloat16(p2);
            
            local_sum = p1 + p2;
            
            #pragma unroll
            for(int offset = 16; offset > 0; offset /= 2)
                local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
                
            float row_sum = __shfl_sync(0xffffffff, local_sum, 0);
            float l_new = l_old * exp_diff + row_sum;
            
            if (lane_id == 0) {
                m_max_smem[row] = m_new;
                l_sum_smem[row] = l_new;
            }
            
            for (int c = 0; c < 4; c++) {
                int col = lane_id + c * 32;
                O_smem[row * 128 + col] *= exp_diff;
            }
        }
        __syncthreads();
        
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> frag_P;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> frag_V[4];
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_O[4];
        
        int warp_row_O = warp_id / 2; 
        int warp_col_O = warp_id % 2; 
        
        wmma::load_matrix_sync(frag_O[0], &O_smem[warp_row_O * 16 * 128 + warp_col_O * 64 + 0], 128, wmma::mem_row_major);
        wmma::load_matrix_sync(frag_O[1], &O_smem[warp_row_O * 16 * 128 + warp_col_O * 64 + 16], 128, wmma::mem_row_major);
        wmma::load_matrix_sync(frag_O[2], &O_smem[warp_row_O * 16 * 128 + warp_col_O * 64 + 32], 128, wmma::mem_row_major);
        wmma::load_matrix_sync(frag_O[3], &O_smem[warp_row_O * 16 * 128 + warp_col_O * 64 + 48], 128, wmma::mem_row_major);
        
        for (int k_step = 0; k_step < 4; k_step++) { 
            int k_idx = k_step * 16;
            wmma::load_matrix_sync(frag_P, &P_bf16_smem[warp_row_O * 16 * 64 + k_idx], 64);
            
            wmma::load_matrix_sync(frag_V[0], &V_smem[k_idx * 128 + warp_col_O * 64 + 0], 128);
            wmma::load_matrix_sync(frag_V[1], &V_smem[k_idx * 128 + warp_col_O * 64 + 16], 128);
            wmma::load_matrix_sync(frag_V[2], &V_smem[k_idx * 128 + warp_col_O * 64 + 32], 128);
            wmma::load_matrix_sync(frag_V[3], &V_smem[k_idx * 128 + warp_col_O * 64 + 48], 128);
            
            wmma::mma_sync(frag_O[0], frag_P, frag_V[0], frag_O[0]);
            wmma::mma_sync(frag_O[1], frag_P, frag_V[1], frag_O[1]);
            wmma::mma_sync(frag_O[2], frag_P, frag_V[2], frag_O[2]);
            wmma::mma_sync(frag_O[3], frag_P, frag_V[3], frag_O[3]);
        }
        
        wmma::store_matrix_sync(&O_smem[warp_row_O * 16 * 128 + warp_col_O * 64 + 0], frag_O[0], 128, wmma::mem_row_major);
        wmma::store_matrix_sync(&O_smem[warp_row_O * 16 * 128 + warp_col_O * 64 + 16], frag_O[1], 128, wmma::mem_row_major);
        wmma::store_matrix_sync(&O_smem[warp_row_O * 16 * 128 + warp_col_O * 64 + 32], frag_O[2], 128, wmma::mem_row_major);
        wmma::store_matrix_sync(&O_smem[warp_row_O * 16 * 128 + warp_col_O * 64 + 48], frag_O[3], 128, wmma::mem_row_major);
        __syncthreads();
    }

    __nv_bfloat16* O_ptr = O + batch_head_offset + (int64_t)q_seq_start * 128;
    float* LSE_ptr = LSE + (int64_t)batch * H * S_seq + (int64_t)head * S_seq + q_seq_start;

    uint4* O_ptr_u4 = (uint4*)O_ptr;
    for (int i = tid; i < q_seq_len * 16; i += 256) {
        int row = i / 16;
        int col_block = i % 16; 
        float l = l_sum_smem[row];
        
        uint32_t packed[4];
        for (int k = 0; k < 4; k++) {
            float out0 = O_smem[row * 128 + col_block * 8 + k * 2 + 0] / l;
            float out1 = O_smem[row * 128 + col_block * 8 + k * 2 + 1] / l;
            __nv_bfloat16 bf0 = __float2bfloat16(out0);
            __nv_bfloat16 bf1 = __float2bfloat16(out1);
            
            uint16_t u0 = reinterpret_cast<uint16_t&>(bf0);
            uint16_t u1 = reinterpret_cast<uint16_t&>(bf1);
            packed[k] = ((uint32_t)u1 << 16) | u0;
        }
        
        uint4 out_u4;
        out_u4.x = packed[0];
        out_u4.y = packed[1];
        out_u4.z = packed[2];
        out_u4.w = packed[3];
        O_ptr_u4[i] = out_u4;
    }

    if (tid < 64) {
        if (tid < q_seq_len) {
            float m = m_max_smem[tid];
            float l = l_sum_smem[tid];
            LSE_ptr[tid] = m + logf(l);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S_seq = Q.size(2);
    int64_t D = Q.size(3);

    __nv_bfloat16* q_ptr = static_cast<__nv_bfloat16*>(Q.data_ptr());
    __nv_bfloat16* k_ptr = static_cast<__nv_bfloat16*>(K.data_ptr());
    __nv_bfloat16* v_ptr = static_cast<__nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_ptr = static_cast<float*>(LSE.data_ptr());

    dim3 grid(B * H, (S_seq + 63) / 64);
    dim3 block(32, 8);
    size_t smem_size = 98816;

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    mha_kernel<<<grid, block, smem_size, stream>>>(q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr, B, H, S_seq, D);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha::run);

}  // namespace tvm_ffi_mha