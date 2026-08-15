#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <mma.h>
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

namespace mha_bwd {

__global__ void kernel_pass1_dQ(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    int B, int H, int S) 
{
    int i_start = blockIdx.x * 64;
    if (i_start >= S) return;
    
    int b = blockIdx.y / H;
    int h = blockIdx.y % H;
    long long batch_head_offset = (long long)(b * H + h) * S * 128;
    
    extern __shared__ char smem[];
    __nv_bfloat16* s_Q = (__nv_bfloat16*)smem;
    __nv_bfloat16* s_dO = s_Q + 64 * 128;
    __nv_bfloat16* s_K = s_dO + 64 * 128;
    __nv_bfloat16* s_V = s_K + 64 * 128;
    float* s_S = (float*)(s_V + 64 * 128);
    float* s_dP = s_S + 64 * 64;
    __nv_bfloat16* s_dS = (__nv_bfloat16*)(s_dP + 64 * 64);
    float* delta = (float*)(s_dS + 64 * 64);
    float* s_delta_reduce = delta + 64;
    
    int tid = threadIdx.x;
    
    int4* s_Q_int4 = (int4*)s_Q;
    int4* s_dO_int4 = (int4*)s_dO;
    int4* s_K_int4 = (int4*)s_K; 
    
    int4* Q_int4 = (int4*)(Q + batch_head_offset + i_start * 128);
    int4* dO_int4 = (int4*)(dO + batch_head_offset + i_start * 128);
    int4* O_int4 = (int4*)(O + batch_head_offset + i_start * 128);
    
    for (int i = 0; i < 8; ++i) {
        int idx = i * 128 + tid; 
        int row = idx / 16;
        if (i_start + row < S) {
            s_Q_int4[idx] = Q_int4[idx];
            s_dO_int4[idx] = dO_int4[idx];
            s_K_int4[idx] = O_int4[idx]; 
        } else {
            s_Q_int4[idx] = make_int4(0, 0, 0, 0);
            s_dO_int4[idx] = make_int4(0, 0, 0, 0);
            s_K_int4[idx] = make_int4(0, 0, 0, 0);
        }
    }
    __syncthreads();
    
    int row = tid / 2;
    int col_start = (tid % 2) * 64;
    float sum = 0.0f;
    for (int c = 0; c < 64; ++c) {
        float do_val = __bfloat162float(s_dO[row * 128 + col_start + c]);
        float o_val = __bfloat162float(s_K[row * 128 + col_start + c]); 
        sum += do_val * o_val;
    }
    s_delta_reduce[tid] = sum;
    __syncthreads();
    
    if (tid % 2 == 0) {
        delta[tid / 2] = s_delta_reduce[tid] + s_delta_reduce[tid + 1];
    }
    __syncthreads();
    
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_dQ[2][4];
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            wmma::fill_fragment(frag_dQ[i][j], 0.0f);
        }
    }
    
    int w = tid / 32;
    int w_r = w / 2;
    int w_c = w % 2;
    
    const float* L_ptr = L + (long long)(b * H + h) * S + i_start;
    
    for (int j_start = 0; j_start <= i_start; j_start += 64) {
        int4* K_int4 = (int4*)(K + batch_head_offset + j_start * 128);
        int4* V_int4 = (int4*)(V + batch_head_offset + j_start * 128);
        int4* s_V_int4 = (int4*)s_V;
        
        for (int i = 0; i < 8; ++i) {
            int idx = i * 128 + tid;
            int r = idx / 16;
            if (j_start + r < S) {
                s_K_int4[idx] = K_int4[idx];
                s_V_int4[idx] = V_int4[idx];
            } else {
                s_K_int4[idx] = make_int4(0, 0, 0, 0);
                s_V_int4[idx] = make_int4(0, 0, 0, 0);
            }
        }
        __syncthreads();
        
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_S[2][2];
        for(int i = 0; i < 2; ++i) for(int j = 0; j < 2; ++j) wmma::fill_fragment(frag_S[i][j], 0.0f);
        
        for (int k = 0; k < 128; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a0, a1;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b0, b1;
            wmma::load_matrix_sync(a0, s_Q + (w_r * 32) * 128 + k, 128);
            wmma::load_matrix_sync(a1, s_Q + (w_r * 32 + 16) * 128 + k, 128);
            wmma::load_matrix_sync(b0, s_K + (w_c * 32) * 128 + k, 128);
            wmma::load_matrix_sync(b1, s_K + (w_c * 32 + 16) * 128 + k, 128);
            
            wmma::mma_sync(frag_S[0][0], a0, b0, frag_S[0][0]);
            wmma::mma_sync(frag_S[0][1], a0, b1, frag_S[0][1]);
            wmma::mma_sync(frag_S[1][0], a1, b0, frag_S[1][0]);
            wmma::mma_sync(frag_S[1][1], a1, b1, frag_S[1][1]);
        }
        
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_dP[2][2];
        for(int i = 0; i < 2; ++i) for(int j = 0; j < 2; ++j) wmma::fill_fragment(frag_dP[i][j], 0.0f);
        
        for (int k = 0; k < 128; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a0, a1;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b0, b1;
            wmma::load_matrix_sync(a0, s_dO + (w_r * 32) * 128 + k, 128);
            wmma::load_matrix_sync(a1, s_dO + (w_r * 32 + 16) * 128 + k, 128);
            wmma::load_matrix_sync(b0, s_V + (w_c * 32) * 128 + k, 128);
            wmma::load_matrix_sync(b1, s_V + (w_c * 32 + 16) * 128 + k, 128);
            
            wmma::mma_sync(frag_dP[0][0], a0, b0, frag_dP[0][0]);
            wmma::mma_sync(frag_dP[0][1], a0, b1, frag_dP[0][1]);
            wmma::mma_sync(frag_dP[1][0], a1, b0, frag_dP[1][0]);
            wmma::mma_sync(frag_dP[1][1], a1, b1, frag_dP[1][1]);
        }
        
        wmma::store_matrix_sync(s_S + (w_r * 32) * 64 + w_c * 32, frag_S[0][0], 64, wmma::mem_row_major);
        wmma::store_matrix_sync(s_S + (w_r * 32) * 64 + w_c * 32 + 16, frag_S[0][1], 64, wmma::mem_row_major);
        wmma::store_matrix_sync(s_S + (w_r * 32 + 16) * 64 + w_c * 32, frag_S[1][0], 64, wmma::mem_row_major);
        wmma::store_matrix_sync(s_S + (w_r * 32 + 16) * 64 + w_c * 32 + 16, frag_S[1][1], 64, wmma::mem_row_major);
        
        wmma::store_matrix_sync(s_dP + (w_r * 32) * 64 + w_c * 32, frag_dP[0][0], 64, wmma::mem_row_major);
        wmma::store_matrix_sync(s_dP + (w_r * 32) * 64 + w_c * 32 + 16, frag_dP[0][1], 64, wmma::mem_row_major);
        wmma::store_matrix_sync(s_dP + (w_r * 32 + 16) * 64 + w_c * 32, frag_dP[1][0], 64, wmma::mem_row_major);
        wmma::store_matrix_sync(s_dP + (w_r * 32 + 16) * 64 + w_c * 32 + 16, frag_dP[1][1], 64, wmma::mem_row_major);
        
        __syncthreads();
        
        for (int i = 0; i < 32; ++i) {
            int idx = tid * 32 + i;
            int r = idx / 64;
            int c = idx % 64;
            
            float s_val = s_S[r * 64 + c];
            float dp_val = s_dP[r * 64 + c];
            
            s_val *= 0.08838834764f; 
            
            int global_row = i_start + r;
            int global_col = j_start + c;
            
            float p_val = 0.0f;
            float l_val = (global_row < S) ? L_ptr[r] : 0.0f;
            if (global_col <= global_row && global_row < S && global_col < S) {
                p_val = expf(s_val - l_val);
            }
            
            float ds_val = p_val * (dp_val - delta[r]);
            s_dS[r * 64 + c] = __float2bfloat16(ds_val);
        }
        __syncthreads();
        
        for (int k = 0; k < 64; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a0, a1;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b0, b1, b2, b3;
            
            wmma::load_matrix_sync(a0, s_dS + (w_r * 32) * 64 + k, 64);
            wmma::load_matrix_sync(a1, s_dS + (w_r * 32 + 16) * 64 + k, 64);
            
            wmma::load_matrix_sync(b0, s_K + k * 128 + w_c * 64, 128);
            wmma::load_matrix_sync(b1, s_K + k * 128 + w_c * 64 + 16, 128);
            wmma::load_matrix_sync(b2, s_K + k * 128 + w_c * 64 + 32, 128);
            wmma::load_matrix_sync(b3, s_K + k * 128 + w_c * 64 + 48, 128);
            
            wmma::mma_sync(frag_dQ[0][0], a0, b0, frag_dQ[0][0]);
            wmma::mma_sync(frag_dQ[0][1], a0, b1, frag_dQ[0][1]);
            wmma::mma_sync(frag_dQ[0][2], a0, b2, frag_dQ[0][2]);
            wmma::mma_sync(frag_dQ[0][3], a0, b3, frag_dQ[0][3]);
            
            wmma::mma_sync(frag_dQ[1][0], a1, b0, frag_dQ[1][0]);
            wmma::mma_sync(frag_dQ[1][1], a1, b1, frag_dQ[1][1]);
            wmma::mma_sync(frag_dQ[1][2], a1, b2, frag_dQ[1][2]);
            wmma::mma_sync(frag_dQ[1][3], a1, b3, frag_dQ[1][3]);
        }
        __syncthreads();
    }
    
    float* s_dQ_float = (float*)s_Q; 
    
    wmma::store_matrix_sync(s_dQ_float + (w_r * 32) * 128 + w_c * 64, frag_dQ[0][0], 128, wmma::mem_row_major);
    wmma::store_matrix_sync(s_dQ_float + (w_r * 32) * 128 + w_c * 64 + 16, frag_dQ[0][1], 128, wmma::mem_row_major);
    wmma::store_matrix_sync(s_dQ_float + (w_r * 32) * 128 + w_c * 64 + 32, frag_dQ[0][2], 128, wmma::mem_row_major);
    wmma::store_matrix_sync(s_dQ_float + (w_r * 32) * 128 + w_c * 64 + 48, frag_dQ[0][3], 128, wmma::mem_row_major);
    
    wmma::store_matrix_sync(s_dQ_float + (w_r * 32 + 16) * 128 + w_c * 64, frag_dQ[1][0], 128, wmma::mem_row_major);
    wmma::store_matrix_sync(s_dQ_float + (w_r * 32 + 16) * 128 + w_c * 64 + 16, frag_dQ[1][1], 128, wmma::mem_row_major);
    wmma::store_matrix_sync(s_dQ_float + (w_r * 32 + 16) * 128 + w_c * 64 + 32, frag_dQ[1][2], 128, wmma::mem_row_major);
    wmma::store_matrix_sync(s_dQ_float + (w_r * 32 + 16) * 128 + w_c * 64 + 48, frag_dQ[1][3], 128, wmma::mem_row_major);
    
    __syncthreads();
    
    __nv_bfloat16* dQ_ptr = dQ + batch_head_offset + i_start * 128;
    for (int i = 0; i < 64; ++i) {
        int idx = i * 128 + tid;
        if (i_start + i < S) {
            dQ_ptr[idx] = __float2bfloat16(s_dQ_float[idx]);
        }
    }
}

__global__ void kernel_pass2_dK_dV(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int B, int H, int S) 
{
    int j_start = blockIdx.x * 64;
    if (j_start >= S) return;
    
    int b = blockIdx.y / H;
    int h = blockIdx.y % H;
    long long batch_head_offset = (long long)(b * H + h) * S * 128;
    
    extern __shared__ char smem[];
    __nv_bfloat16* s_K = (__nv_bfloat16*)smem;
    __nv_bfloat16* s_V = s_K + 64 * 128;
    __nv_bfloat16* s_Q = s_V + 64 * 128;
    __nv_bfloat16* s_dO = s_Q + 64 * 128;
    float* s_S = (float*)(s_dO + 64 * 128);
    float* s_dP = s_S + 64 * 64;
    __nv_bfloat16* s_dS = (__nv_bfloat16*)(s_dP + 64 * 64);
    __nv_bfloat16* s_P = s_dS + 64 * 64;
    float* delta = (float*)(s_P + 64 * 64);
    float* s_delta_reduce = delta + 64;
    
    int tid = threadIdx.x;
    
    int4* s_K_int4 = (int4*)s_K;
    int4* s_V_int4 = (int4*)s_V;
    int4* K_int4 = (int4*)(K + batch_head_offset + j_start * 128);
    int4* V_int4 = (int4*)(V + batch_head_offset + j_start * 128);
    
    for (int i = 0; i < 8; ++i) {
        int idx = i * 128 + tid;
        int row = idx / 16;
        if (j_start + row < S) {
            s_K_int4[idx] = K_int4[idx];
            s_V_int4[idx] = V_int4[idx];
        } else {
            s_K_int4[idx] = make_int4(0, 0, 0, 0);
            s_V_int4[idx] = make_int4(0, 0, 0, 0);
        }
    }
    __syncthreads();
    
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_dK[2][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_dV[2][4];
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            wmma::fill_fragment(frag_dK[i][j], 0.0f);
            wmma::fill_fragment(frag_dV[i][j], 0.0f);
        }
    }
    
    int w = tid / 32;
    int w_r = w / 2;
    int w_c = w % 2;
    
    for (int i_start = j_start; i_start < S; i_start += 64) {
        __nv_bfloat16* s_O = (__nv_bfloat16*)s_S; 
        
        int4* s_Q_int4 = (int4*)s_Q;
        int4* s_dO_int4 = (int4*)s_dO;
        int4* s_O_int4 = (int4*)s_O;
        int4* Q_int4 = (int4*)(Q + batch_head_offset + i_start * 128);
        int4* dO_int4 = (int4*)(dO + batch_head_offset + i_start * 128);
        int4* O_int4 = (int4*)(O + batch_head_offset + i_start * 128);
        
        for (int i = 0; i < 8; ++i) {
            int idx = i * 128 + tid;
            int r = idx / 16;
            if (i_start + r < S) {
                s_Q_int4[idx] = Q_int4[idx];
                s_dO_int4[idx] = dO_int4[idx];
                s_O_int4[idx] = O_int4[idx];
            } else {
                s_Q_int4[idx] = make_int4(0, 0, 0, 0);
                s_dO_int4[idx] = make_int4(0, 0, 0, 0);
                s_O_int4[idx] = make_int4(0, 0, 0, 0);
            }
        }
        __syncthreads();
        
        int row = tid / 2;
        int col_start = (tid % 2) * 64;
        float sum = 0.0f;
        for (int c = 0; c < 64; ++c) {
            float do_val = __bfloat162float(s_dO[row * 128 + col_start + c]);
            float o_val = __bfloat162float(s_O[row * 128 + col_start + c]);
            sum += do_val * o_val;
        }
        s_delta_reduce[tid] = sum;
        __syncthreads();
        if (tid % 2 == 0) {
            delta[tid / 2] = s_delta_reduce[tid] + s_delta_reduce[tid + 1];
        }
        __syncthreads(); 
        
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_S[2][2];
        for(int i = 0; i < 2; ++i) for(int j = 0; j < 2; ++j) wmma::fill_fragment(frag_S[i][j], 0.0f);
        
        for (int k = 0; k < 128; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a0, a1;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b0, b1;
            wmma::load_matrix_sync(a0, s_Q + (w_r * 32) * 128 + k, 128);
            wmma::load_matrix_sync(a1, s_Q + (w_r * 32 + 16) * 128 + k, 128);
            wmma::load_matrix_sync(b0, s_K + (w_c * 32) * 128 + k, 128);
            wmma::load_matrix_sync(b1, s_K + (w_c * 32 + 16) * 128 + k, 128);
            
            wmma::mma_sync(frag_S[0][0], a0, b0, frag_S[0][0]);
            wmma::mma_sync(frag_S[0][1], a0, b1, frag_S[0][1]);
            wmma::mma_sync(frag_S[1][0], a1, b0, frag_S[1][0]);
            wmma::mma_sync(frag_S[1][1], a1, b1, frag_S[1][1]);
        }
        
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_dP[2][2];
        for(int i = 0; i < 2; ++i) for(int j = 0; j < 2; ++j) wmma::fill_fragment(frag_dP[i][j], 0.0f);
        
        for (int k = 0; k < 128; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a0, a1;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b0, b1;
            wmma::load_matrix_sync(a0, s_dO + (w_r * 32) * 128 + k, 128);
            wmma::load_matrix_sync(a1, s_dO + (w_r * 32 + 16) * 128 + k, 128);
            wmma::load_matrix_sync(b0, s_V + (w_c * 32) * 128 + k, 128);
            wmma::load_matrix_sync(b1, s_V + (w_c * 32 + 16) * 128 + k, 128);
            
            wmma::mma_sync(frag_dP[0][0], a0, b0, frag_dP[0][0]);
            wmma::mma_sync(frag_dP[0][1], a0, b1, frag_dP[0][1]);
            wmma::mma_sync(frag_dP[1][0], a1, b0, frag_dP[1][0]);
            wmma::mma_sync(frag_dP[1][1], a1, b1, frag_dP[1][1]);
        }
        
        wmma::store_matrix_sync(s_S + (w_r * 32) * 64 + w_c * 32, frag_S[0][0], 64, wmma::mem_row_major);
        wmma::store_matrix_sync(s_S + (w_r * 32) * 64 + w_c * 32 + 16, frag_S[0][1], 64, wmma::mem_row_major);
        wmma::store_matrix_sync(s_S + (w_r * 32 + 16) * 64 + w_c * 32, frag_S[1][0], 64, wmma::mem_row_major);
        wmma::store_matrix_sync(s_S + (w_r * 32 + 16) * 64 + w_c * 32 + 16, frag_S[1][1], 64, wmma::mem_row_major);
        
        wmma::store_matrix_sync(s_dP + (w_r * 32) * 64 + w_c * 32, frag_dP[0][0], 64, wmma::mem_row_major);
        wmma::store_matrix_sync(s_dP + (w_r * 32) * 64 + w_c * 32 + 16, frag_dP[0][1], 64, wmma::mem_row_major);
        wmma::store_matrix_sync(s_dP + (w_r * 32 + 16) * 64 + w_c * 32, frag_dP[1][0], 64, wmma::mem_row_major);
        wmma::store_matrix_sync(s_dP + (w_r * 32 + 16) * 64 + w_c * 32 + 16, frag_dP[1][1], 64, wmma::mem_row_major);
        
        __syncthreads();
        
        const float* L_ptr = L + (long long)(b * H + h) * S + i_start;
        
        for (int i = 0; i < 32; ++i) {
            int idx = tid * 32 + i;
            int r = idx / 64;
            int c = idx % 64;
            
            float s_val = s_S[r * 64 + c];
            float dp_val = s_dP[r * 64 + c];
            
            s_val *= 0.08838834764f; 
            
            int global_row = i_start + r;
            int global_col = j_start + c;
            
            float p_val = 0.0f;
            float l_val = (global_row < S) ? L_ptr[r] : 0.0f;
            if (global_col <= global_row && global_row < S && global_col < S) {
                p_val = expf(s_val - l_val);
            }
            
            float ds_val = p_val * (dp_val - delta[r]);
            s_dS[r * 64 + c] = __float2bfloat16(ds_val);
            s_P[r * 64 + c] = __float2bfloat16(p_val);
        }
        __syncthreads();
        
        for (int k = 0; k < 64; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a0, a1;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b0, b1, b2, b3;
            
            wmma::load_matrix_sync(a0, s_dS + k * 64 + (w_r * 32), 64);
            wmma::load_matrix_sync(a1, s_dS + k * 64 + (w_r * 32 + 16), 64);
            
            wmma::load_matrix_sync(b0, s_Q + k * 128 + w_c * 64, 128);
            wmma::load_matrix_sync(b1, s_Q + k * 128 + w_c * 64 + 16, 128);
            wmma::load_matrix_sync(b2, s_Q + k * 128 + w_c * 64 + 32, 128);
            wmma::load_matrix_sync(b3, s_Q + k * 128 + w_c * 64 + 48, 128);
            
            wmma::mma_sync(frag_dK[0][0], a0, b0, frag_dK[0][0]);
            wmma::mma_sync(frag_dK[0][1], a0, b1, frag_dK[0][1]);
            wmma::mma_sync(frag_dK[0][2], a0, b2, frag_dK[0][2]);
            wmma::mma_sync(frag_dK[0][3], a0, b3, frag_dK[0][3]);
            
            wmma::mma_sync(frag_dK[1][0], a1, b0, frag_dK[1][0]);
            wmma::mma_sync(frag_dK[1][1], a1, b1, frag_dK[1][1]);
            wmma::mma_sync(frag_dK[1][2], a1, b2, frag_dK[1][2]);
            wmma::mma_sync(frag_dK[1][3], a1, b3, frag_dK[1][3]);
        }
        
        for (int k = 0; k < 64; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a0, a1;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b0, b1, b2, b3;
            
            wmma::load_matrix_sync(a0, s_P + k * 64 + (w_r * 32), 64);
            wmma::load_matrix_sync(a1, s_P + k * 64 + (w_r * 32 + 16), 64);
            
            wmma::load_matrix_sync(b0, s_dO + k * 128 + w_c * 64, 128);
            wmma::load_matrix_sync(b1, s_dO + k * 128 + w_c * 64 + 16, 128);
            wmma::load_matrix_sync(b2, s_dO + k * 128 + w_c * 64 + 32, 128);
            wmma::load_matrix_sync(b3, s_dO + k * 128 + w_c * 64 + 48, 128);
            
            wmma::mma_sync(frag_dV[0][0], a0, b0, frag_dV[0][0]);
            wmma::mma_sync(frag_dV[0][1], a0, b1, frag_dV[0][1]);
            wmma::mma_sync(frag_dV[0][2], a0, b2, frag_dV[0][2]);
            wmma::mma_sync(frag_dV[0][3], a0, b3, frag_dV[0][3]);
            
            wmma::mma_sync(frag_dV[1][0], a1, b0, frag_dV[1][0]);
            wmma::mma_sync(frag_dV[1][1], a1, b1, frag_dV[1][1]);
            wmma::mma_sync(frag_dV[1][2], a1, b2, frag_dV[1][2]);
            wmma::mma_sync(frag_dV[1][3], a1, b3, frag_dV[1][3]);
        }
        __syncthreads();
    }
    
    float* s_buf_float = (float*)s_K;
    
    wmma::store_matrix_sync(s_buf_float + (w_r * 32) * 128 + w_c * 64, frag_dK[0][0], 128, wmma::mem_row_major);
    wmma::store_matrix_sync(s_buf_float + (w_r * 32) * 128 + w_c * 64 + 16, frag_dK[0][1], 128, wmma::mem_row_major);
    wmma::store_matrix_sync(s_buf_float + (w_r * 32) * 128 + w_c * 64 + 32, frag_dK[0][2], 128, wmma::mem_row_major);
    wmma::store_matrix_sync(s_buf_float + (w_r * 32) * 128 + w_c * 64 + 48, frag_dK[0][3], 128, wmma::mem_row_major);
    
    wmma::store_matrix_sync(s_buf_float + (w_r * 32 + 16) * 128 + w_c * 64, frag_dK[1][0], 128, wmma::mem_row_major);
    wmma::store_matrix_sync(s_buf_float + (w_r * 32 + 16) * 128 + w_c * 64 + 16, frag_dK[1][1], 128, wmma::mem_row_major);
    wmma::store_matrix_sync(s_buf_float + (w_r * 32 + 16) * 128 + w_c * 64 + 32, frag_dK[1][2], 128, wmma::mem_row_major);
    wmma::store_matrix_sync(s_buf_float + (w_r * 32 + 16) * 128 + w_c * 64 + 48, frag_dK[1][3], 128, wmma::mem_row_major);
    
    __syncthreads();
    
    __nv_bfloat16* dK_ptr = dK + batch_head_offset + j_start * 128;
    for (int i = 0; i < 64; ++i) {
        int idx = i * 128 + tid;
        if (j_start + i < S) {
            dK_ptr[idx] = __float2bfloat16(s_buf_float[idx]);
        }
    }
    __syncthreads();
    
    wmma::store_matrix_sync(s_buf_float + (w_r * 32) * 128 + w_c * 64, frag_dV[0][0], 128, wmma::mem_row_major);
    wmma::store_matrix_sync(s_buf_float + (w_r * 32) * 128 + w_c * 64 + 16, frag_dV[0][1], 128, wmma::mem_row_major);
    wmma::store_matrix_sync(s_buf_float + (w_r * 32) * 128 + w_c * 64 + 32, frag_dV[0][2], 128, wmma::mem_row_major);
    wmma::store_matrix_sync(s_buf_float + (w_r * 32) * 128 + w_c * 64 + 48, frag_dV[0][3], 128, wmma::mem_row_major);
    
    wmma::store_matrix_sync(s_buf_float + (w_r * 32 + 16) * 128 + w_c * 64, frag_dV[1][0], 128, wmma::mem_row_major);
    wmma::store_matrix_sync(s_buf_float + (w_r * 32 + 16) * 128 + w_c * 64 + 16, frag_dV[1][1], 128, wmma::mem_row_major);
    wmma::store_matrix_sync(s_buf_float + (w_r * 32 + 16) * 128 + w_c * 64 + 32, frag_dV[1][2], 128, wmma::mem_row_major);
    wmma::store_matrix_sync(s_buf_float + (w_r * 32 + 16) * 128 + w_c * 64 + 48, frag_dV[1][3], 128, wmma::mem_row_major);
    
    __syncthreads();
    
    __nv_bfloat16* dV_ptr = dV + batch_head_offset + j_start * 128;
    for (int i = 0; i < 64; ++i) {
        int idx = i * 128 + tid;
        if (j_start + i < S) {
            dV_ptr[idx] = __float2bfloat16(s_buf_float[idx]);
        }
    }
}

void run(tvm::ffi::TensorView Q_tv, tvm::ffi::TensorView K_tv, tvm::ffi::TensorView V_tv, 
         tvm::ffi::TensorView O_tv, tvm::ffi::TensorView dO_tv, tvm::ffi::TensorView L_tv,
         tvm::ffi::TensorView dQ_tv, tvm::ffi::TensorView dK_tv, tvm::ffi::TensorView dV_tv) {
    CUDA_CHECK(cudaSetDevice(Q_tv.device().device_id));
    
    int64_t B = Q_tv.size(0);
    int64_t H = Q_tv.size(1);
    int64_t S = Q_tv.size(2);
    
    __nv_bfloat16* Q = static_cast<__nv_bfloat16*>(Q_tv.data_ptr());
    __nv_bfloat16* K = static_cast<__nv_bfloat16*>(K_tv.data_ptr());
    __nv_bfloat16* V = static_cast<__nv_bfloat16*>(V_tv.data_ptr());
    __nv_bfloat16* O = static_cast<__nv_bfloat16*>(O_tv.data_ptr());
    __nv_bfloat16* dO = static_cast<__nv_bfloat16*>(dO_tv.data_ptr());
    float* L = static_cast<float*>(L_tv.data_ptr());
    __nv_bfloat16* dQ = static_cast<__nv_bfloat16*>(dQ_tv.data_ptr());
    __nv_bfloat16* dK = static_cast<__nv_bfloat16*>(dK_tv.data_ptr());
    __nv_bfloat16* dV = static_cast<__nv_bfloat16*>(dV_tv.data_ptr());
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q_tv.device().device_type, Q_tv.device().device_id));
    
    int grid_x = (S + 63) / 64;
    int grid_y = B * H;
    dim3 grid(grid_x, grid_y);
    dim3 block(128);
    
    size_t smem_pass1 = 110 * 1024;
    size_t smem_pass2 = 120 * 1024;
    
    CUDA_CHECK(cudaFuncSetAttribute(kernel_pass1_dQ, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_pass1));
    CUDA_CHECK(cudaFuncSetAttribute(kernel_pass2_dK_dV, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_pass2));
    
    kernel_pass1_dQ<<<grid, block, smem_pass1, stream>>>(Q, K, V, O, dO, L, dQ, B, H, S);
    kernel_pass2_dK_dV<<<grid, block, smem_pass2, stream>>>(Q, K, V, O, dO, L, dK, dV, B, H, S);
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}