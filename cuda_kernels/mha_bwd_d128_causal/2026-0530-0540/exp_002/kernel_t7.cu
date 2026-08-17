#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>
#include <mma.h>

using namespace nvcuda;

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_mha_bwd {

__device__ __forceinline__ void cp_async_128b(void* smem, const void* gmem) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"(gmem));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template<int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void load_async_kv(
    __nv_bfloat16* s_k, __nv_bfloat16* s_v, 
    const __nv_bfloat16* g_k, const __nv_bfloat16* g_v,
    int valid_rows)
{
    int tid = threadIdx.x;
    for (int i = tid; i < 64 * 128 / 8; i += blockDim.x) {
        int row = i / 16;
        int col = (i % 16) * 8;
        if (row < valid_rows) {
            cp_async_128b(&s_k[row * 128 + col], &g_k[row * 128 + col]);
            cp_async_128b(&s_v[row * 128 + col], &g_v[row * 128 + col]);
        } else {
            *reinterpret_cast<float4*>(&s_k[row * 128 + col]) = {0,0,0,0};
            *reinterpret_cast<float4*>(&s_v[row * 128 + col]) = {0,0,0,0};
        }
    }
}

__device__ __forceinline__ void load_async_q_do(
    __nv_bfloat16* s_q, __nv_bfloat16* s_do, 
    const __nv_bfloat16* g_q, const __nv_bfloat16* g_do,
    int valid_rows)
{
    int tid = threadIdx.x;
    for (int i = tid; i < 64 * 128 / 8; i += blockDim.x) {
        int row = i / 16;
        int col = (i % 16) * 8;
        if (row < valid_rows) {
            cp_async_128b(&s_q[row * 128 + col], &g_q[row * 128 + col]);
            cp_async_128b(&s_do[row * 128 + col], &g_do[row * 128 + col]);
        } else {
            *reinterpret_cast<float4*>(&s_q[row * 128 + col]) = {0,0,0,0};
            *reinterpret_cast<float4*>(&s_do[row * 128 + col]) = {0,0,0,0};
        }
    }
}

__device__ __forceinline__ void load_vec_sync(float* s_dst, const float* g_src, int valid_rows) {
    int tid = threadIdx.x;
    if (tid < 64) {
        s_dst[tid] = (tid < valid_rows) ? g_src[tid] : 0.0f;
    }
}

__device__ __forceinline__ void load_tile_bf16_d128_sync(__nv_bfloat16* dst, const __nv_bfloat16* src, int valid_rows) {
    int tid = threadIdx.x;
    for (int i = tid; i < 64 * 128 / 8; i += blockDim.x) {
        int row = i / 16;
        int col = (i % 16) * 8;
        if (row < valid_rows) {
            *reinterpret_cast<float4*>(&dst[row * 128 + col]) = *reinterpret_cast<const float4*>(&src[row * 128 + col]);
        } else {
            *reinterpret_cast<float4*>(&dst[row * 128 + col]) = {0.0f, 0.0f, 0.0f, 0.0f};
        }
    }
}

__device__ __forceinline__ void write_bf16_128b(__nv_bfloat16* dst, const float* src, int valid_rows) {
    for (int i = threadIdx.x; i < 64 * 128 / 8; i += blockDim.x) {
        int r = i / 16;
        int c = (i % 16) * 8;
        if (r < valid_rows) {
            float f0 = src[r * 128 + c + 0];
            float f1 = src[r * 128 + c + 1];
            float f2 = src[r * 128 + c + 2];
            float f3 = src[r * 128 + c + 3];
            float f4 = src[r * 128 + c + 4];
            float f5 = src[r * 128 + c + 5];
            float f6 = src[r * 128 + c + 6];
            float f7 = src[r * 128 + c + 7];
            
            __nv_bfloat162 v0; v0.x = __float2bfloat16(f0); v0.y = __float2bfloat16(f1);
            __nv_bfloat162 v1; v1.x = __float2bfloat16(f2); v1.y = __float2bfloat16(f3);
            __nv_bfloat162 v2; v2.x = __float2bfloat16(f4); v2.y = __float2bfloat16(f5);
            __nv_bfloat162 v3; v3.x = __float2bfloat16(f6); v3.y = __float2bfloat16(f7);
            
            uint32_t u0 = *reinterpret_cast<uint32_t*>(&v0);
            uint32_t u1 = *reinterpret_cast<uint32_t*>(&v1);
            uint32_t u2 = *reinterpret_cast<uint32_t*>(&v2);
            uint32_t u3 = *reinterpret_cast<uint32_t*>(&v3);
            
            *reinterpret_cast<uint4*>(&dst[r * 128 + c]) = make_uint4(u0, u1, u2, u3);
        }
    }
}

__global__ void precompute_D_kernel(const __nv_bfloat16* __restrict__ dO, 
                                    const __nv_bfloat16* __restrict__ O, 
                                    float* __restrict__ D, 
                                    int B, int H, int S) 
{
    int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    if (seq_idx < S) {
        float sum = 0.0f;
        int offset = (b * H * S + h * S + seq_idx) * 128;
        
        for (int d = 0; d < 128; d += 8) {
            float4 do_val4 = *reinterpret_cast<const float4*>(&dO[offset + d]);
            float4 o_val4 = *reinterpret_cast<const float4*>(&O[offset + d]);
            
            uint32_t* do_u = reinterpret_cast<uint32_t*>(&do_val4);
            uint32_t* o_u = reinterpret_cast<uint32_t*>(&o_val4);
            
            for (int k = 0; k < 4; k++) {
                __nv_bfloat162 do_v = *reinterpret_cast<__nv_bfloat162*>(&do_u[k]);
                __nv_bfloat162 o_v = *reinterpret_cast<__nv_bfloat162*>(&o_u[k]);
                sum += __bfloat162float(do_v.x) * __bfloat162float(o_v.x);
                sum += __bfloat162float(do_v.y) * __bfloat162float(o_v.y);
            }
        }
        D[b * H * S + h * S + seq_idx] = sum;
    }
}

__global__ void bwd_kernel_dQ(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const float* __restrict__ L,
    const float* __restrict__ D,
    const __nv_bfloat16* __restrict__ dO,
    __nv_bfloat16* __restrict__ dQ,
    int B, int H, int S, float scale) 
{
    int b = blockIdx.z;
    int h = blockIdx.y;
    int i_chunk = blockIdx.x; 
    
    int i_base = i_chunk * 64;
    int valid_rows_q = min(64, S - i_base);
    if (valid_rows_q <= 0) return;
    
    int head_offset = (b * H + h) * S * 128;
    int vec_offset = (b * H + h) * S;
    
    extern __shared__ char smem[];
    __nv_bfloat16* s_Q = (__nv_bfloat16*)smem;
    __nv_bfloat16* s_dO = s_Q + 64 * 128;
    __nv_bfloat16* s_K[2];
    s_K[0] = s_dO + 64 * 128;
    s_K[1] = s_K[0] + 64 * 128;
    __nv_bfloat16* s_V[2];
    s_V[0] = s_K[1] + 64 * 128;
    s_V[1] = s_V[0] + 64 * 128;
    float* s_S = (float*)(s_V[1] + 64 * 128);
    float* s_dP = s_S + 64 * 64;
    __nv_bfloat16* s_dS = (__nv_bfloat16*)(s_dP + 64 * 64);
    float* s_L = (float*)(s_dS + 64 * 64);
    float* s_D = s_L + 64;

    load_tile_bf16_d128_sync(s_Q, Q + head_offset + i_base * 128, valid_rows_q);
    load_tile_bf16_d128_sync(s_dO, dO + head_offset + i_base * 128, valid_rows_q);
    load_vec_sync(s_L, L + vec_offset + i_base, valid_rows_q);
    load_vec_sync(s_D, D + vec_offset + i_base, valid_rows_q);
    
    int j_chunk = 0;
    int load_idx = 0;
    if (j_chunk <= i_chunk) {
        int valid_rows = min(64, S - j_chunk * 64);
        load_async_kv(s_K[load_idx], s_V[load_idx], 
            K + head_offset + j_chunk * 64 * 128, 
            V + head_offset + j_chunk * 64 * 128, valid_rows);
    }
    cp_async_commit();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dq_acc[8];
    for (int i=0; i<8; i++) wmma::fill_fragment(dq_acc[i], 0.0f);

    int warp_id = threadIdx.x / 32;
    int row_s = (warp_id / 2) * 32;
    int col_s = (warp_id % 2) * 32;
    
    for (j_chunk = 0; j_chunk <= i_chunk; j_chunk++) {
        int next_j_chunk = j_chunk + 1;
        int next_load_idx = 1 - load_idx;
        if (next_j_chunk <= i_chunk) {
            int valid_rows = min(64, S - next_j_chunk * 64);
            load_async_kv(s_K[next_load_idx], s_V[next_load_idx], 
                K + head_offset + next_j_chunk * 64 * 128, 
                V + head_offset + next_j_chunk * 64 * 128, valid_rows);
        }
        cp_async_commit();
        
        if (next_j_chunk <= i_chunk) {
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();
        
        int valid_rows_kv = min(64, S - j_chunk * 64);
        
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_acc[4];
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> dp_acc[4];
        for (int i=0; i<4; i++) {
            wmma::fill_fragment(s_acc[i], 0.0f);
            wmma::fill_fragment(dp_acc[i], 0.0f);
        }
        
        for (int k = 0; k < 128; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> q_f[2], do_f[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> k_f[2], v_f[2];
            
            wmma::load_matrix_sync(q_f[0], &s_Q[row_s * 128 + k], 128);
            wmma::load_matrix_sync(q_f[1], &s_Q[(row_s + 16) * 128 + k], 128);
            wmma::load_matrix_sync(do_f[0], &s_dO[row_s * 128 + k], 128);
            wmma::load_matrix_sync(do_f[1], &s_dO[(row_s + 16) * 128 + k], 128);
            
            wmma::load_matrix_sync(k_f[0], &s_K[load_idx][col_s * 128 + k], 128);
            wmma::load_matrix_sync(k_f[1], &s_K[load_idx][(col_s + 16) * 128 + k], 128);
            wmma::load_matrix_sync(v_f[0], &s_V[load_idx][col_s * 128 + k], 128);
            wmma::load_matrix_sync(v_f[1], &s_V[load_idx][(col_s + 16) * 128 + k], 128);
            
            wmma::mma_sync(s_acc[0], q_f[0], k_f[0], s_acc[0]);
            wmma::mma_sync(s_acc[1], q_f[0], k_f[1], s_acc[1]);
            wmma::mma_sync(s_acc[2], q_f[1], k_f[0], s_acc[2]);
            wmma::mma_sync(s_acc[3], q_f[1], k_f[1], s_acc[3]);
            
            wmma::mma_sync(dp_acc[0], do_f[0], v_f[0], dp_acc[0]);
            wmma::mma_sync(dp_acc[1], do_f[0], v_f[1], dp_acc[1]);
            wmma::mma_sync(dp_acc[2], do_f[1], v_f[0], dp_acc[2]);
            wmma::mma_sync(dp_acc[3], do_f[1], v_f[1], dp_acc[3]);
        }
        
        for(int c=0; c<4; c++) {
            for(int t=0; t<s_acc[c].num_elements; t++) s_acc[c].x[t] *= scale;
            wmma::store_matrix_sync(&s_S[row_s * 64 + col_s + (c%2)*16 + (c/2)*16*64], s_acc[c], 64, wmma::mem_row_major);
            wmma::store_matrix_sync(&s_dP[row_s * 64 + col_s + (c%2)*16 + (c/2)*16*64], dp_acc[c], 64, wmma::mem_row_major);
        }
        __syncthreads();
        
        for (int i = threadIdx.x; i < 64 * 64; i += blockDim.x) {
            int r = i / 64;
            int c = i % 64;
            float ds_val = 0.0f;
            if (r < valid_rows_q && c < valid_rows_kv) {
                bool mask = (i_chunk == j_chunk) ? (r >= c) : true;
                if (mask) {
                    float p_val = expf(s_S[i] - s_L[r]);
                    ds_val = p_val * (s_dP[i] - s_D[r]) * scale;
                }
            }
            s_dS[i] = __float2bfloat16(ds_val);
        }
        __syncthreads();
        
        int row_dq = warp_id * 16;
        for (int k = 0; k < 64; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> ds_f;
            wmma::load_matrix_sync(ds_f, &s_dS[row_dq * 64 + k], 64);
            for (int c_step = 0; c_step < 8; c_step++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> k_f;
                wmma::load_matrix_sync(k_f, &s_K[load_idx][k * 128 + c_step * 16], 128);
                wmma::mma_sync(dq_acc[c_step], ds_f, k_f, dq_acc[c_step]);
            }
        }
        __syncthreads();
        
        load_idx = next_load_idx;
    }
    
    int row_dq = warp_id * 16;
    for (int c_step = 0; c_step < 8; c_step++) {
        wmma::store_matrix_sync(&s_S[row_dq * 128 + c_step * 16], dq_acc[c_step], 128, wmma::mem_row_major);
    }
    __syncthreads();
    
    write_bf16_128b(dQ + head_offset + i_base * 128, s_S, valid_rows_q);
}

__global__ void bwd_kernel_dK_dV(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const float* __restrict__ L,
    const float* __restrict__ D,
    const __nv_bfloat16* __restrict__ dO,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int B, int H, int S, float scale) 
{
    int b = blockIdx.z;
    int h = blockIdx.y;
    int j_chunk = blockIdx.x; 
    
    int j_base = j_chunk * 64;
    int valid_rows_kv = min(64, S - j_base);
    if (valid_rows_kv <= 0) return;
    
    int head_offset = (b * H + h) * S * 128;
    int vec_offset = (b * H + h) * S;
    
    extern __shared__ char smem[];
    __nv_bfloat16* s_Q[2];
    s_Q[0] = (__nv_bfloat16*)smem;
    s_Q[1] = s_Q[0] + 64 * 128;
    __nv_bfloat16* s_dO[2];
    s_dO[0] = s_Q[1] + 64 * 128;
    s_dO[1] = s_dO[0] + 64 * 128;
    __nv_bfloat16* s_K = s_dO[1] + 64 * 128;
    __nv_bfloat16* s_V = s_K + 64 * 128;
    float* s_ST = (float*)(s_V + 64 * 128);
    float* s_dPT = s_ST + 64 * 64;
    __nv_bfloat16* s_dST = (__nv_bfloat16*)(s_dPT + 64 * 64);
    __nv_bfloat16* s_PT = s_dST + 64 * 64;
    float* s_L[2];
    s_L[0] = (float*)(s_PT + 64 * 64);
    s_L[1] = s_L[0] + 64;
    float* s_D[2];
    s_D[0] = s_L[1] + 64;
    s_D[1] = s_D[0] + 64;

    load_tile_bf16_d128_sync(s_K, K + head_offset + j_base * 128, valid_rows_kv);
    load_tile_bf16_d128_sync(s_V, V + head_offset + j_base * 128, valid_rows_kv);
    
    int i_chunk = j_chunk;
    int load_idx = 0;
    if (i_chunk < (S + 63) / 64) {
        int valid_rows = min(64, S - i_chunk * 64);
        load_async_q_do(s_Q[load_idx], s_dO[load_idx], 
            Q + head_offset + i_chunk * 64 * 128, 
            dO + head_offset + i_chunk * 64 * 128, valid_rows);
        load_vec_sync(s_L[load_idx], L + vec_offset + i_chunk * 64, valid_rows);
        load_vec_sync(s_D[load_idx], D + vec_offset + i_chunk * 64, valid_rows);
    }
    cp_async_commit();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dk_acc[8];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dv_acc[8];
    for (int i=0; i<8; i++) {
        wmma::fill_fragment(dk_acc[i], 0.0f);
        wmma::fill_fragment(dv_acc[i], 0.0f);
    }

    int warp_id = threadIdx.x / 32;
    int row_s = (warp_id / 2) * 32;
    int col_s = (warp_id % 2) * 32;
    
    for (i_chunk = j_chunk; i_chunk < (S + 63) / 64; i_chunk++) {
        int next_i_chunk = i_chunk + 1;
        int next_load_idx = 1 - load_idx;
        
        if (next_i_chunk < (S + 63) / 64) {
            int valid_rows = min(64, S - next_i_chunk * 64);
            load_async_q_do(s_Q[next_load_idx], s_dO[next_load_idx], 
                Q + head_offset + next_i_chunk * 64 * 128, 
                dO + head_offset + next_i_chunk * 64 * 128, valid_rows);
            load_vec_sync(s_L[next_load_idx], L + vec_offset + next_i_chunk * 64, valid_rows);
            load_vec_sync(s_D[next_load_idx], D + vec_offset + next_i_chunk * 64, valid_rows);
        }
        cp_async_commit();
        
        if (next_i_chunk < (S + 63) / 64) {
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();
        
        int valid_rows_q = min(64, S - i_chunk * 64);
        
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> st_acc[4];
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> dpt_acc[4];
        for (int i=0; i<4; i++) {
            wmma::fill_fragment(st_acc[i], 0.0f);
            wmma::fill_fragment(dpt_acc[i], 0.0f);
        }
        
        for (int k = 0; k < 128; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> k_f[2], v_f[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> q_f[2], do_f[2];
            
            wmma::load_matrix_sync(k_f[0], &s_K[row_s * 128 + k], 128);
            wmma::load_matrix_sync(k_f[1], &s_K[(row_s + 16) * 128 + k], 128);
            wmma::load_matrix_sync(v_f[0], &s_V[row_s * 128 + k], 128);
            wmma::load_matrix_sync(v_f[1], &s_V[(row_s + 16) * 128 + k], 128);
            
            wmma::load_matrix_sync(q_f[0], &s_Q[load_idx][col_s * 128 + k], 128);
            wmma::load_matrix_sync(q_f[1], &s_Q[load_idx][(col_s + 16) * 128 + k], 128);
            wmma::load_matrix_sync(do_f[0], &s_dO[load_idx][col_s * 128 + k], 128);
            wmma::load_matrix_sync(do_f[1], &s_dO[load_idx][(col_s + 16) * 128 + k], 128);
            
            wmma::mma_sync(st_acc[0], k_f[0], q_f[0], st_acc[0]);
            wmma::mma_sync(st_acc[1], k_f[0], q_f[1], st_acc[1]);
            wmma::mma_sync(st_acc[2], k_f[1], q_f[0], st_acc[2]);
            wmma::mma_sync(st_acc[3], k_f[1], q_f[1], st_acc[3]);
            
            wmma::mma_sync(dpt_acc[0], v_f[0], do_f[0], dpt_acc[0]);
            wmma::mma_sync(dpt_acc[1], v_f[0], do_f[1], dpt_acc[1]);
            wmma::mma_sync(dpt_acc[2], v_f[1], do_f[0], dpt_acc[2]);
            wmma::mma_sync(dpt_acc[3], v_f[1], do_f[1], dpt_acc[3]);
        }
        
        for(int c=0; c<4; c++) {
            for(int t=0; t<st_acc[c].num_elements; t++) st_acc[c].x[t] *= scale;
            wmma::store_matrix_sync(&s_ST[row_s * 64 + col_s + (c%2)*16 + (c/2)*16*64], st_acc[c], 64, wmma::mem_row_major);
            wmma::store_matrix_sync(&s_dPT[row_s * 64 + col_s + (c%2)*16 + (c/2)*16*64], dpt_acc[c], 64, wmma::mem_row_major);
        }
        __syncthreads();
        
        for (int i = threadIdx.x; i < 64 * 64; i += blockDim.x) {
            int r = i / 64; 
            int c = i % 64; 
            float ds_val = 0.0f;
            float p_val = 0.0f;
            if (r < valid_rows_kv && c < valid_rows_q) {
                bool mask = (i_chunk == j_chunk) ? (r <= c) : true;
                if (mask) {
                    p_val = expf(s_ST[i] - s_L[load_idx][c]);
                    ds_val = p_val * (s_dPT[i] - s_D[load_idx][c]) * scale;
                }
            }
            s_dST[i] = __float2bfloat16(ds_val);
            s_PT[i]  = __float2bfloat16(p_val);
        }
        __syncthreads();
        
        int row_dk = warp_id * 16;
        for (int k = 0; k < 64; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> dst_f, pt_f;
            wmma::load_matrix_sync(dst_f, &s_dST[row_dk * 64 + k], 64);
            wmma::load_matrix_sync(pt_f, &s_PT[row_dk * 64 + k], 64);
            
            for (int c_step = 0; c_step < 8; c_step++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> q_f, do_f;
                wmma::load_matrix_sync(q_f, &s_Q[load_idx][k * 128 + c_step * 16], 128);
                wmma::load_matrix_sync(do_f, &s_dO[load_idx][k * 128 + c_step * 16], 128);
                
                wmma::mma_sync(dk_acc[c_step], dst_f, q_f, dk_acc[c_step]);
                wmma::mma_sync(dv_acc[c_step], pt_f, do_f, dv_acc[c_step]);
            }
        }
        __syncthreads();
        
        load_idx = next_load_idx;
    }
    
    int row_dk = warp_id * 16;
    for (int c_step = 0; c_step < 8; c_step++) {
        wmma::store_matrix_sync(&s_ST[row_dk * 128 + c_step * 16], dk_acc[c_step], 128, wmma::mem_row_major);
    }
    __syncthreads();
    
    write_bf16_128b(dK + head_offset + j_base * 128, s_ST, valid_rows_kv);
    __syncthreads();
    
    for (int c_step = 0; c_step < 8; c_step++) {
        wmma::store_matrix_sync(&s_ST[row_dk * 128 + c_step * 16], dv_acc[c_step], 128, wmma::mem_row_major);
    }
    __syncthreads();
    
    write_bf16_128b(dV + head_offset + j_base * 128, s_ST, valid_rows_kv);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) 
{
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int B = static_cast<int>(Q.size(0));
    int H = static_cast<int>(Q.size(1));
    int S = static_cast<int>(Q.size(2));
    if (S == 0) return;
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    __nv_bfloat16* q_ptr = static_cast<__nv_bfloat16*>(Q.data_ptr());
    __nv_bfloat16* k_ptr = static_cast<__nv_bfloat16*>(K.data_ptr());
    __nv_bfloat16* v_ptr = static_cast<__nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    __nv_bfloat16* do_ptr = static_cast<__nv_bfloat16*>(dO.data_ptr());
    float* l_ptr = static_cast<float*>(L.data_ptr());
    __nv_bfloat16* dq_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dk_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dv_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());

    float* d_ptr;
    CUDA_CHECK(cudaMallocAsync(&d_ptr, static_cast<int64_t>(B) * H * S * sizeof(float), stream));
    
    dim3 d_block(128);
    dim3 d_grid((S + 127) / 128, H, B);
    precompute_D_kernel<<<d_grid, d_block, 0, stream>>>(do_ptr, o_ptr, d_ptr, B, H, S);
    
    int smem_size = 147456; // safely pads the ~140KB required payload
    CUDA_CHECK(cudaFuncSetAttribute(bwd_kernel_dQ, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    CUDA_CHECK(cudaFuncSetAttribute(bwd_kernel_dK_dV, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    dim3 bwd_block(128);
    dim3 bwd_grid((S + 63) / 64, H, B);
    float scale = 1.0f / sqrtf(128.0f);
    
    bwd_kernel_dQ<<<bwd_grid, bwd_block, smem_size, stream>>>(
        q_ptr, k_ptr, v_ptr, l_ptr, d_ptr, do_ptr, dq_ptr, B, H, S, scale);
        
    bwd_kernel_dK_dV<<<bwd_grid, bwd_block, smem_size, stream>>>(
        q_ptr, k_ptr, v_ptr, l_ptr, d_ptr, do_ptr, dk_ptr, dv_ptr, B, H, S, scale);
        
    CUDA_CHECK(cudaFreeAsync(d_ptr, stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_bwd::run);

} // namespace tvm_ffi_mha_bwd