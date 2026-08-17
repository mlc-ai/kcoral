#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <mma.h>
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

using namespace nvcuda;

namespace tvm_ffi_example_cuda {

__device__ __forceinline__ void atomicAddBf162(__nv_bfloat162* address, __nv_bfloat162 val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    atomicAdd(address, val);
#else
    unsigned int* address_as_uint = (unsigned int*)address;
    unsigned int old = *address_as_uint, assumed;
    do {
        assumed = old;
        float2 f = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&assumed));
        float2 f_val = __bfloat1622float2(val);
        f.x += f_val.x; f.y += f_val.y;
        __nv_bfloat162 sum = __floats2bfloat162_rn(f.x, f.y);
        old = atomicCAS(address_as_uint, assumed, *reinterpret_cast<unsigned int*>(&sum));
    } while (assumed != old);
#endif
}

// Precomputes D_i = sum(dO_i * O_i)
__global__ void ComputeDKernel(const __nv_bfloat162* O, const __nv_bfloat162* dO, float* D, int B, int H, int S) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * H * S) {
        float sum = 0.0f;
        int offset = idx * 64; // 128 elements = 64 bf162
        for (int k = 0; k < 64; ++k) {
            __nv_bfloat162 o_val = O[offset + k];
            __nv_bfloat162 do_val = dO[offset + k];
            float2 o_f2 = __bfloat1622float2(o_val);
            float2 do_f2 = __bfloat1622float2(do_val);
            sum += o_f2.x * do_f2.x + o_f2.y * do_f2.y;
        }
        D[idx] = sum;
    }
}

struct SharedMemory {
    __nv_bfloat16 s_Q[32][128];
    __nv_bfloat16 s_K[32][128];
    __nv_bfloat16 s_V[32][128];
    __nv_bfloat16 s_dO[32][128];
    __nv_bfloat16 s_P[32][32];
    __nv_bfloat16 s_dS[32][32];
    float s_S_float[32][32];
    float s_dP_float[32][32];
    float s_dQ_float[32][128];
};

// Grid: ( (S+31)/32, H, B )
// Block: 128
__global__ void FABackwardKernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ D,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int S, float scale) 
{
    extern __shared__ char smem_buf[];
    SharedMemory& smem = *reinterpret_cast<SharedMemory*>(smem_buf);

    int j_block = blockIdx.x;
    int h_idx = blockIdx.y;
    int b_idx = blockIdx.z;
    
    int num_blocks = (S + 31) / 32;
    
    size_t head_offset = (b_idx * gridDim.y + h_idx) * (size_t)S * 128;
    size_t l_offset = (b_idx * gridDim.y + h_idx) * (size_t)S;
    
    int j = j_block;
    int global_j_start = j * 32;
    int tid = threadIdx.x;
    
    // Load K_j and V_j
    for (int step = 0; step < 32; ++step) {
        int lin = tid + step * 128;
        if (lin < 32 * 128) {
            int r = lin / 128;
            int c = lin % 128;
            if (global_j_start + r < S) {
                smem.s_K[r][c] = K[head_offset + (global_j_start + r) * 128 + c];
                smem.s_V[r][c] = V[head_offset + (global_j_start + r) * 128 + c];
            } else {
                smem.s_K[r][c] = __float2bfloat16(0.0f);
                smem.s_V[r][c] = __float2bfloat16(0.0f);
            }
        }
    }
    
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dK_frag[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dV_frag[4];
    for (int i=0; i<4; ++i) {
        wmma::fill_fragment(dK_frag[i], 0.0f);
        wmma::fill_fragment(dV_frag[i], 0.0f);
    }
    
    int warp_id = tid / 32;
    int r_off = (warp_id / 2) * 16;
    int c_off = (warp_id % 2) * 64; 
    int w_c_S = (warp_id % 2) * 16; 
    
    for (int i = j; i < num_blocks; ++i) {
        int global_i_start = i * 32;
        
        // Load Q_i and dO_i
        for (int step = 0; step < 32; ++step) {
            int lin = tid + step * 128;
            if (lin < 32 * 128) {
                int r = lin / 128;
                int c = lin % 128;
                if (global_i_start + r < S) {
                    smem.s_Q[r][c] = Q[head_offset + (global_i_start + r) * 128 + c];
                    smem.s_dO[r][c] = dO[head_offset + (global_i_start + r) * 128 + c];
                } else {
                    smem.s_Q[r][c] = __float2bfloat16(0.0f);
                    smem.s_dO[r][c] = __float2bfloat16(0.0f);
                }
            }
        }
        __syncthreads();
        
        // Compute S_ij and dP_ij
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> S_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> dP_frag;
        wmma::fill_fragment(S_frag, 0.0f);
        wmma::fill_fragment(dP_frag, 0.0f);
        
        for (int k = 0; k < 8; ++k) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> q_f;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> k_f; 
            wmma::load_matrix_sync(q_f, &smem.s_Q[r_off][k*16], 128);
            wmma::load_matrix_sync(k_f, &smem.s_K[k*16][w_c_S], 128); 
            wmma::mma_sync(S_frag, q_f, k_f, S_frag);
            
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> do_f;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> v_f;
            wmma::load_matrix_sync(do_f, &smem.s_dO[r_off][k*16], 128);
            wmma::load_matrix_sync(v_f, &smem.s_V[k*16][w_c_S], 128);
            wmma::mma_sync(dP_frag, do_f, v_f, dP_frag);
        }
        
        wmma::store_matrix_sync(&smem.s_S_float[r_off][w_c_S], S_frag, 32, wmma::mem_row_major);
        wmma::store_matrix_sync(&smem.s_dP_float[r_off][w_c_S], dP_frag, 32, wmma::mem_row_major);
        __syncthreads();
        
        // Apply causal mask, LSE, and scale to compute P and dS
        for (int step = 0; step < 8; ++step) {
            int lin = tid + step * 128;
            int r = lin / 32;
            int c = lin % 32;
            
            int global_r = global_i_start + r;
            int global_c = global_j_start + c;
            
            float p_val = 0.0f;
            float ds_val = 0.0f;
            
            if (global_c <= global_r && global_r < S && global_c < S) {
                float s_val = smem.s_S_float[r][c];
                float dp_val = smem.s_dP_float[r][c];
                float l_val = L[l_offset + global_r];
                float d_val = D[l_offset + global_r];
                
                p_val = expf(s_val * scale - l_val);
                ds_val = p_val * (dp_val - d_val) * scale;
            }
            
            smem.s_P[r][c] = __float2bfloat16(p_val);
            smem.s_dS[r][c] = __float2bfloat16(ds_val);
        }
        __syncthreads();
        
        // Update dV_j += P^T * dO_i
        for (int k = 0; k < 2; ++k) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> PT_frag;
            wmma::load_matrix_sync(PT_frag, &smem.s_P[k*16][r_off], 32);
            for (int c_step = 0; c_step < 4; ++c_step) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> dO_frag;
                wmma::load_matrix_sync(dO_frag, &smem.s_dO[k*16][c_off + c_step*16], 128);
                wmma::mma_sync(dV_frag[c_step], PT_frag, dO_frag, dV_frag[c_step]);
            }
        }
        
        // Update dK_j += dS^T * Q_i
        for (int k = 0; k < 2; ++k) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> dST_frag;
            wmma::load_matrix_sync(dST_frag, &smem.s_dS[k*16][r_off], 32);
            for (int c_step = 0; c_step < 4; ++c_step) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> Q_frag;
                wmma::load_matrix_sync(Q_frag, &smem.s_Q[k*16][c_off + c_step*16], 128);
                wmma::mma_sync(dK_frag[c_step], dST_frag, Q_frag, dK_frag[c_step]);
            }
        }
        
        // Compute dQ_i += dS * K_j
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> dQ_frag[4];
        for (int x=0; x<4; ++x) wmma::fill_fragment(dQ_frag[x], 0.0f);
        
        for (int k = 0; k < 2; ++k) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> dS_frag_a;
            wmma::load_matrix_sync(dS_frag_a, &smem.s_dS[r_off][k*16], 32);
            for (int c_step = 0; c_step < 4; ++c_step) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> K_frag_b;
                wmma::load_matrix_sync(K_frag_b, &smem.s_K[k*16][c_off + c_step*16], 128);
                wmma::mma_sync(dQ_frag[c_step], dS_frag_a, K_frag_b, dQ_frag[c_step]);
            }
        }
        
        for (int c_step = 0; c_step < 4; ++c_step) {
            wmma::store_matrix_sync(&smem.s_dQ_float[r_off][c_off + c_step*16], dQ_frag[c_step], 128, wmma::mem_row_major);
        }
        __syncthreads();
        
        // Atomic Add to global dQ
        for(int step=0; step < 16; ++step) {
            int lin = tid + step * 128;
            int r = lin / 64;
            int c_half = lin % 64;
            int c = c_half * 2;
            
            int global_r = global_i_start + r;
            if (global_r < S && c < 128) {
                float f0 = smem.s_dQ_float[r][c];
                float f1 = smem.s_dQ_float[r][c+1];
                __nv_bfloat162 bval = __floats2bfloat162_rn(f0, f1);
                
                __nv_bfloat162* out_ptr = (__nv_bfloat162*)&dQ[head_offset + global_r * 128 + c];
                atomicAddBf162(out_ptr, bval);
            }
        }
        __syncthreads(); 
    }
    
    // Store accumulated dK_j and dV_j
    for (int c_step = 0; c_step < 4; ++c_step) {
        wmma::store_matrix_sync(&smem.s_dQ_float[r_off][c_off + c_step*16], dK_frag[c_step], 128, wmma::mem_row_major);
    }
    __syncthreads();
    
    for (int step = 0; step < 32; ++step) {
        int lin = tid + step * 128;
        if (lin < 32 * 128) {
            int r = lin / 128;
            int c = lin % 128;
            int global_r = global_j_start + r;
            if (global_r < S) {
                dK[head_offset + global_r * 128 + c] = __float2bfloat16(smem.s_dQ_float[r][c]);
            }
        }
    }
    __syncthreads();
    
    for (int c_step = 0; c_step < 4; ++c_step) {
        wmma::store_matrix_sync(&smem.s_dQ_float[r_off][c_off + c_step*16], dV_frag[c_step], 128, wmma::mem_row_major);
    }
    __syncthreads();
    
    for (int step = 0; step < 32; ++step) {
        int lin = tid + step * 128;
        if (lin < 32 * 128) {
            int r = lin / 128;
            int c = lin % 128;
            int global_r = global_j_start + r;
            if (global_r < S) {
                dV[head_offset + global_r * 128 + c] = __float2bfloat16(smem.s_dQ_float[r][c]);
            }
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int d = 128; 
    
    float scale = 1.0f / sqrtf((float)d);
    
    float* d_D = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_D, B * H * S * sizeof(float), stream));
    
    int threads_D = 128;
    int blocks_D = (B * H * S + threads_D - 1) / threads_D;
    ComputeDKernel<<<blocks_D, threads_D, 0, stream>>>(
        static_cast<const __nv_bfloat162*>(O.data_ptr()),
        static_cast<const __nv_bfloat162*>(dO.data_ptr()),
        d_D, B, H, S
    );
    
    CUDA_CHECK(cudaMemsetAsync(dQ.data_ptr(), 0, B * H * S * 128 * sizeof(__nv_bfloat16), stream));
    
    int num_j_blocks = (S + 31) / 32;
    dim3 grid(num_j_blocks, H, B);
    dim3 block(128); 
    
    CUDA_CHECK(cudaFuncSetAttribute(FABackwardKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedMemory)));
    
    FABackwardKernel<<<grid, block, sizeof(SharedMemory), stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        d_D,
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        S, scale
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFreeAsync(d_D, stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda