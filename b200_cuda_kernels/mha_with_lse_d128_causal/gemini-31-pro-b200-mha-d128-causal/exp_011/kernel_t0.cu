#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>
#include <math.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

__device__ __forceinline__ void ldmatrix_x4(uint32_t* R, uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3])
        : "r"(addr)
    );
}

__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t* R, uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
        : "=r"(R[0]), "=r"(R[1])
        : "r"(addr)
    );
}

__device__ __forceinline__ void mma_m16n8k16(float* C, uint32_t* A, uint32_t* B) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1])
    );
}

__device__ __forceinline__ uint16_t float_to_bf16(float x) {
    __nv_bfloat16 bf = __float2bfloat16(x);
    return *(uint16_t*)&bf;
}

__device__ __forceinline__ void load_global_to_smem_128(uint16_t* smem, const uint16_t* gmem, int rows, int global_row_start, int seq_len) {
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    for (int r = warp; r < rows; r += 4) {
        int col = lane * 4;
        int g_row = global_row_start + r;
        float2 val = {0.0f, 0.0f};
        if (g_row < seq_len) {
            val = *(float2*)(&gmem[g_row * 128 + col]);
        }
        
        int col_chunk = col / 8;
        int col_rem = col % 8;
        int s_chunk = col_chunk ^ (r % 8);
        int s_idx = r * 128 + s_chunk * 8 + col_rem;
        
        *(float2*)(&smem[s_idx]) = val;
    }
}

__device__ __forceinline__ void store_smem_to_global_128(const uint16_t* smem, uint16_t* gmem, int rows, int global_row_start, int seq_len) {
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    for (int r = warp; r < rows; r += 4) {
        int g_row = global_row_start + r;
        if (g_row < seq_len) {
            int col = lane * 4;
            int col_chunk = col / 8;
            int col_rem = col % 8;
            int s_chunk = col_chunk ^ (r % 8);
            int s_idx = r * 128 + s_chunk * 8 + col_rem;
            
            *(float2*)(&gmem[g_row * 128 + col]) = *(float2*)(&smem[s_idx]);
        }
    }
}

__global__ __launch_bounds__(128) void mha_fwd_kernel(
    const uint16_t* __restrict__ Q,
    const uint16_t* __restrict__ K,
    const uint16_t* __restrict__ V,
    uint16_t* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int seq_len
) {
    int b = blockIdx.x / H;
    int h = blockIdx.x % H;
    int bq = blockIdx.y;
    
    if (bq * 64 >= seq_len) return;
    
    int w = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    
    extern __shared__ uint8_t shared_mem[];
    uint16_t* smem_Q = (uint16_t*)shared_mem;               
    uint16_t* smem_K = (uint16_t*)(shared_mem + 16384);     
    uint16_t* smem_V = (uint16_t*)(shared_mem + 32768);     
    uint16_t* smem_P = (uint16_t*)(shared_mem + 49152);     
    uint16_t* smem_O = smem_K; 

    const uint16_t* q_ptr = Q + b * H * seq_len * 128 + h * seq_len * 128;
    const uint16_t* k_ptr = K + b * H * seq_len * 128 + h * seq_len * 128;
    const uint16_t* v_ptr = V + b * H * seq_len * 128 + h * seq_len * 128;
    uint16_t* o_ptr = O + b * H * seq_len * 128 + h * seq_len * 128;
    float* lse_ptr = LSE + b * H * seq_len + h * seq_len;
    
    load_global_to_smem_128(smem_Q, q_ptr, 64, bq * 64, seq_len);
    __syncthreads();
    
    uint32_t Q_regs[8][4];
    #pragma unroll
    for (int k_chunk = 0; k_chunk < 8; ++k_chunk) {
        int r = lane % 16;
        int c = k_chunk * 16;
        int s_chunk = (c / 8) ^ ((w * 16 + r) % 8);
        int s_idx = (w * 16 + r) * 128 + s_chunk * 8;
        uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem_Q[s_idx]));
        ldmatrix_x4(Q_regs[k_chunk], addr);
    }
    
    float m_i[2] = {-INFINITY, -INFINITY};
    float l_i[2] = {0.0f, 0.0f};
    
    float O_regs[16][4];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        O_regs[i][0] = O_regs[i][1] = O_regs[i][2] = O_regs[i][3] = 0.0f;
    }
    
    float scale = 1.0f / sqrtf(128.0f);
    int r1_local = lane / 4;
    int r3_local = lane / 4 + 8;
    
    for (int bk = 0; bk <= bq; ++bk) {
        load_global_to_smem_128(smem_K, k_ptr, 64, bk * 64, seq_len);
        load_global_to_smem_128(smem_O, v_ptr, 64, bk * 64, seq_len);
        __syncthreads();
        
        for (int tile = threadIdx.x / 32; tile < 32; tile += 4) {
            int tile_r = (tile / 8) * 16;
            int tile_c = (tile % 8) * 16;
            
            int r_in = lane % 16;
            int c_in = lane / 16;
            
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                int c = tile_c + c_in * 8 + i;
                int r = tile_r + r_in;
                
                int s_chunk_O = (c / 8) ^ (r % 8);
                int s_idx_O = r * 128 + s_chunk_O * 8 + (c % 8);
                uint16_t val = smem_O[s_idx_O];
                
                int s_chunk_V = (r / 8) ^ (c % 8);
                int s_idx_V = c * 64 + s_chunk_V * 8 + (r % 8);
                smem_V[s_idx_V] = val;
            }
        }
        __syncthreads();
        
        float S[8][4];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            S[i][0] = S[i][1] = S[i][2] = S[i][3] = 0.0f;
        }
        
        #pragma unroll
        for (int k_chunk = 0; k_chunk < 8; ++k_chunk) {
            #pragma unroll
            for (int n_chunk = 0; n_chunk < 8; ++n_chunk) {
                uint32_t K_regs[2];
                int r = n_chunk * 8 + (lane % 8);
                int c = k_chunk * 16;
                int s_chunk = (c / 8) ^ (r % 8);
                int s_idx = r * 128 + s_chunk * 8;
                uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem_K[s_idx]));
                ldmatrix_x2_trans(K_regs, addr);
                
                mma_m16n8k16(S[n_chunk], Q_regs[k_chunk], K_regs);
            }
        }
        
        float max1 = -INFINITY, max3 = -INFINITY;
        #pragma unroll
        for (int n_chunk = 0; n_chunk < 8; ++n_chunk) {
            int c1 = n_chunk * 8 + (lane % 4) * 2;
            int c2 = c1 + 1;
            
            int global_k_c1 = bk * 64 + c1;
            int global_k_c2 = bk * 64 + c2;
            
            int global_q_r1 = bq * 64 + w * 16 + r1_local;
            int global_q_r3 = bq * 64 + w * 16 + r3_local;
            
            float v1 = (global_q_r1 >= global_k_c1 && global_k_c1 < seq_len) ? (S[n_chunk][0] * scale) : -INFINITY;
            float v2 = (global_q_r1 >= global_k_c2 && global_k_c2 < seq_len) ? (S[n_chunk][1] * scale) : -INFINITY;
            float v3 = (global_q_r3 >= global_k_c1 && global_k_c1 < seq_len) ? (S[n_chunk][2] * scale) : -INFINITY;
            float v4 = (global_q_r3 >= global_k_c2 && global_k_c2 < seq_len) ? (S[n_chunk][3] * scale) : -INFINITY;
            
            S[n_chunk][0] = v1; S[n_chunk][1] = v2;
            S[n_chunk][2] = v3; S[n_chunk][3] = v4;
            
            max1 = fmaxf(max1, fmaxf(v1, v2));
            max3 = fmaxf(max3, fmaxf(v3, v4));
        }
        
        #pragma unroll
        for (int offset = 2; offset > 0; offset /= 2) {
            max1 = fmaxf(max1, __shfl_xor_sync(0xffffffff, max1, offset));
            max3 = fmaxf(max3, __shfl_xor_sync(0xffffffff, max3, offset));
        }
        
        float m_new1 = fmaxf(m_i[0], max1);
        float m_new3 = fmaxf(m_i[1], max3);
        
        float exp_scale1 = expf(m_i[0] - m_new1);
        float exp_scale3 = expf(m_i[1] - m_new3);
        
        l_i[0] *= exp_scale1;
        l_i[1] *= exp_scale3;
        
        float sum1 = 0.0f, sum3 = 0.0f;
        
        #pragma unroll
        for (int n_chunk = 0; n_chunk < 8; ++n_chunk) {
            float p1 = expf(S[n_chunk][0] - m_new1);
            float p2 = expf(S[n_chunk][1] - m_new1);
            float p3 = expf(S[n_chunk][2] - m_new3);
            float p4 = expf(S[n_chunk][3] - m_new3);
            
            sum1 += p1 + p2;
            sum3 += p3 + p4;
            
            uint16_t bf_p1 = float_to_bf16(p1);
            uint16_t bf_p2 = float_to_bf16(p2);
            uint32_t val1 = ((uint32_t)bf_p2 << 16) | bf_p1;
            
            uint16_t bf_p3 = float_to_bf16(p3);
            uint16_t bf_p4 = float_to_bf16(p4);
            uint32_t val3 = ((uint32_t)bf_p4 << 16) | bf_p3;
            
            int c_local = n_chunk * 8 + (lane % 4) * 2;
            
            int s_chunk1 = (c_local / 8) ^ ((w * 16 + r1_local) % 8);
            int s_idx1 = (w * 16 + r1_local) * 64 + s_chunk1 * 8 + (c_local % 8);
            *(uint32_t*)(&smem_P[s_idx1]) = val1;
            
            int s_chunk3 = (c_local / 8) ^ ((w * 16 + r3_local) % 8);
            int s_idx3 = (w * 16 + r3_local) * 64 + s_chunk3 * 8 + (c_local % 8);
            *(uint32_t*)(&smem_P[s_idx3]) = val3;
        }
        
        #pragma unroll
        for (int offset = 2; offset > 0; offset /= 2) {
            sum1 += __shfl_xor_sync(0xffffffff, sum1, offset);
            sum3 += __shfl_xor_sync(0xffffffff, sum3, offset);
        }
        
        l_i[0] += sum1;
        l_i[1] += sum3;
        m_i[0] = m_new1;
        m_i[1] = m_new3;
        
        #pragma unroll
        for (int n_chunk = 0; n_chunk < 16; ++n_chunk) {
            O_regs[n_chunk][0] *= exp_scale1;
            O_regs[n_chunk][1] *= exp_scale1;
            O_regs[n_chunk][2] *= exp_scale3;
            O_regs[n_chunk][3] *= exp_scale3;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int k_chunk = 0; k_chunk < 4; ++k_chunk) {
            uint32_t P_regs[4];
            int r_p = lane % 16;
            int c_p = k_chunk * 16;
            int s_chunk_p = (c_p / 8) ^ ((w * 16 + r_p) % 8);
            int s_idx_p = (w * 16 + r_p) * 64 + s_chunk_p * 8;
            uint32_t addr_p = static_cast<uint32_t>(__cvta_generic_to_shared(&smem_P[s_idx_p]));
            ldmatrix_x4(P_regs, addr_p);
            
            #pragma unroll
            for (int n_chunk = 0; n_chunk < 16; ++n_chunk) {
                uint32_t V_regs[2];
                int r_v = n_chunk * 8 + (lane % 8);
                int c_v = k_chunk * 16;
                int s_chunk_v = (c_v / 8) ^ (r_v % 8);
                int s_idx_v = r_v * 64 + s_chunk_v * 8;
                uint32_t addr_v = static_cast<uint32_t>(__cvta_generic_to_shared(&smem_V[s_idx_v]));
                ldmatrix_x2_trans(V_regs, addr_v);
                
                mma_m16n8k16(O_regs[n_chunk], P_regs, V_regs);
            }
        }
        __syncthreads();
    }
    
    #pragma unroll
    for (int n_chunk = 0; n_chunk < 16; ++n_chunk) {
        int c_local = n_chunk * 8 + (lane % 4) * 2;
        
        uint16_t o1 = float_to_bf16(O_regs[n_chunk][0] / l_i[0]);
        uint16_t o2 = float_to_bf16(O_regs[n_chunk][1] / l_i[0]);
        uint32_t val1 = ((uint32_t)o2 << 16) | o1;
        
        uint16_t o3 = float_to_bf16(O_regs[n_chunk][2] / l_i[1]);
        uint16_t o4 = float_to_bf16(O_regs[n_chunk][3] / l_i[1]);
        uint32_t val3 = ((uint32_t)o4 << 16) | o3;
        
        int s_chunk1 = (c_local / 8) ^ ((w * 16 + r1_local) % 8);
        int s_idx1 = (w * 16 + r1_local) * 128 + s_chunk1 * 8 + (c_local % 8);
        *(uint32_t*)(&smem_O[s_idx1]) = val1;
        
        int s_chunk3 = (c_local / 8) ^ ((w * 16 + r3_local) % 8);
        int s_idx3 = (w * 16 + r3_local) * 128 + s_chunk3 * 8 + (c_local % 8);
        *(uint32_t*)(&smem_O[s_idx3]) = val3;
    }
    __syncthreads();
    
    store_smem_to_global_128(smem_O, o_ptr, 64, bq * 64, seq_len);
    
    if ((lane % 4) == 0) {
        int global_r1 = bq * 64 + w * 16 + r1_local;
        int global_r3 = bq * 64 + w * 16 + r3_local;
        
        if (global_r1 < seq_len) {
            lse_ptr[global_r1] = logf(l_i[0]) + m_i[0];
        }
        if (global_r3 < seq_len) {
            lse_ptr[global_r3] = logf(l_i[1]) + m_i[1];
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
        
        int grid_x = B * H;
        int grid_y = (S + 63) / 64;
        dim3 grid(grid_x, grid_y);
        dim3 block(128);
        
        cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
        
        int smem_size = 57344;
        CUDA_CHECK(cudaFuncSetAttribute(mha_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        
        mha_fwd_kernel<<<grid, block, smem_size, stream>>>(
            static_cast<const uint16_t*>(Q.data_ptr()),
            static_cast<const uint16_t*>(K.data_ptr()),
            static_cast<const uint16_t*>(V.data_ptr()),
            static_cast<uint16_t*>(O.data_ptr()),
            static_cast<float*>(LSE.data_ptr()),
            B, H, S
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    
    TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);
}