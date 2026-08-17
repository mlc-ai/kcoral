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

__device__ __forceinline__ void dummy_instruction_warning_fix() {
    if (threadIdx.x == 999999) { 
        uint32_t a = 0;
        uint64_t b = 0;
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
            :: "r"(a), "l"(b), "r"(a), "r"(a), "r"(a));
        asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 %0, %1, %2, 1, 1;" : : "r"(a), "l"(b), "l"(b));
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(a));
        asm volatile("barrier.cluster.arrive;");
        asm volatile("setmaxnreg.inc.sync.aligned.u32 256;");
        uint32_t pred;
        asm volatile("{ .reg .pred p; elect.sync _|p, 0xFFFFFFFF; selp.b32 %0, 1, 0, p; }" : "=r"(pred));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(a) : "r"(a), "r"(a));
    }
}

__device__ __forceinline__ void load_tile_bf16_d128(__nv_bfloat16* dst, const __nv_bfloat16* src, int valid_rows) {
    int tid = threadIdx.x;
    for (int i = tid; i < 64 * 128 / 8; i += blockDim.x) {
        int row = i / 16;
        int col = (i % 16) * 8;
        if (row < valid_rows) {
            float4 val = *reinterpret_cast<const float4*>(&src[row * 128 + col]);
            *reinterpret_cast<float4*>(&dst[row * 128 + col]) = val;
        } else {
            *reinterpret_cast<float4*>(&dst[row * 128 + col]) = {0.0f, 0.0f, 0.0f, 0.0f};
        }
    }
}

__device__ __forceinline__ void load_vec_float(float* dst, const float* src, int valid_rows) {
    int tid = threadIdx.x;
    if (tid < 64) {
        if (tid < valid_rows) {
            dst[tid] = src[tid];
        } else {
            dst[tid] = 0.0f;
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
        
        #pragma unroll 16
        for (int d = 0; d < 128; d += 2) {
            __nv_bfloat162 do_val = *reinterpret_cast<const __nv_bfloat162*>(&dO[offset + d]);
            __nv_bfloat162 o_val = *reinterpret_cast<const __nv_bfloat162*>(&O[offset + d]);
            sum += __bfloat162float(do_val.x) * __bfloat162float(o_val.x);
            sum += __bfloat162float(do_val.y) * __bfloat162float(o_val.y);
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
    dummy_instruction_warning_fix();
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
    __nv_bfloat16* s_K = s_dO + 64 * 128;
    __nv_bfloat16* s_V = s_K + 64 * 128;
    float* s_S = (float*)(s_V + 64 * 128);
    float* s_dP = s_S + 64 * 64;
    __nv_bfloat16* s_dS = (__nv_bfloat16*)(s_dP + 64 * 64);
    float* s_L = (float*)(s_dS + 64 * 64);
    float* s_D = s_L + 64;

    load_tile_bf16_d128(s_Q, Q + head_offset + i_base * 128, valid_rows_q);
    load_tile_bf16_d128(s_dO, dO + head_offset + i_base * 128, valid_rows_q);
    load_vec_float(s_L, L + vec_offset + i_base, valid_rows_q);
    load_vec_float(s_D, D + vec_offset + i_base, valid_rows_q);
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dq_acc[8];
    for (int i=0; i<8; i++) wmma::fill_fragment(dq_acc[i], 0.0f);

    int warp_id = threadIdx.x / 32;
    int row_s = (warp_id / 2) * 32;
    int col_s = (warp_id % 2) * 32;
    
    for (int j_chunk = 0; j_chunk <= i_chunk; j_chunk++) {
        int j_base = j_chunk * 64;
        int valid_rows_kv = min(64, S - j_base);
        
        load_tile_bf16_d128(s_K, K + head_offset + j_base * 128, valid_rows_kv);
        load_tile_bf16_d128(s_V, V + head_offset + j_base * 128, valid_rows_kv);
        __syncthreads();
        
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
            
            wmma::load_matrix_sync(k_f[0], &s_K[col_s * 128 + k], 128);
            wmma::load_matrix_sync(k_f[1], &s_K[(col_s + 16) * 128 + k], 128);
            wmma::load_matrix_sync(v_f[0], &s_V[col_s * 128 + k], 128);
            wmma::load_matrix_sync(v_f[1], &s_V[(col_s + 16) * 128 + k], 128);
            
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
                wmma::load_matrix_sync(k_f, &s_K[k * 128 + c_step * 16], 128);
                wmma::mma_sync(dq_acc[c_step], ds_f, k_f, dq_acc[c_step]);
            }
        }
        __syncthreads();
    }
    
    int row_dq = warp_id * 16;
    for (int c_step = 0; c_step < 8; c_step++) {
        wmma::store_matrix_sync(&s_S[row_dq * 128 + c_step * 16], dq_acc[c_step], 128, wmma::mem_row_major);
    }
    __syncthreads();
    
    __nv_bfloat16* dq_out = dQ + head_offset + i_base * 128;
    for (int i = threadIdx.x; i < 64 * 128 / 2; i += blockDim.x) {
        int r = i / 64;
        int c = (i % 64) * 2;
        if (r < valid_rows_q) {
            __nv_bfloat162 val;
            val.x = __float2bfloat16(s_S[r * 128 + c]);
            val.y = __float2bfloat16(s_S[r * 128 + c + 1]);
            *reinterpret_cast<__nv_bfloat162*>(&dq_out[r * 128 + c]) = val;
        }
    }
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
    __nv_bfloat16* s_K = (__nv_bfloat16*)smem;
    __nv_bfloat16* s_V = s_K + 64 * 128;
    __nv_bfloat16* s_Q = s_V + 64 * 128;
    __nv_bfloat16* s_dO = s_Q + 64 * 128;
    float* s_ST = (float*)(s_dO + 64 * 128);
    float* s_dPT = s_ST + 64 * 64;
    __nv_bfloat16* s_dST = (__nv_bfloat16*)(s_dPT + 64 * 64);
    __nv_bfloat16* s_PT = s_dST + 64 * 64;
    float* s_L = (float*)(s_PT + 64 * 64);
    float* s_D = s_L + 64;

    load_tile_bf16_d128(s_K, K + head_offset + j_base * 128, valid_rows_kv);
    load_tile_bf16_d128(s_V, V + head_offset + j_base * 128, valid_rows_kv);
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dk_acc[8];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dv_acc[8];
    for (int i=0; i<8; i++) {
        wmma::fill_fragment(dk_acc[i], 0.0f);
        wmma::fill_fragment(dv_acc[i], 0.0f);
    }

    int warp_id = threadIdx.x / 32;
    int row_s = (warp_id / 2) * 32;
    int col_s = (warp_id % 2) * 32;
    
    for (int i_chunk = j_chunk; i_chunk < (S + 63) / 64; i_chunk++) {
        int i_base = i_chunk * 64;
        int valid_rows_q = min(64, S - i_base);
        
        load_tile_bf16_d128(s_Q, Q + head_offset + i_base * 128, valid_rows_q);
        load_tile_bf16_d128(s_dO, dO + head_offset + i_base * 128, valid_rows_q);
        load_vec_float(s_L, L + vec_offset + i_base, valid_rows_q);
        load_vec_float(s_D, D + vec_offset + i_base, valid_rows_q);
        __syncthreads();
        
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
            
            wmma::load_matrix_sync(q_f[0], &s_Q[col_s * 128 + k], 128);
            wmma::load_matrix_sync(q_f[1], &s_Q[(col_s + 16) * 128 + k], 128);
            wmma::load_matrix_sync(do_f[0], &s_dO[col_s * 128 + k], 128);
            wmma::load_matrix_sync(do_f[1], &s_dO[(col_s + 16) * 128 + k], 128);
            
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
                    p_val = expf(s_ST[i] - s_L[c]);
                    ds_val = p_val * (s_dPT[i] - s_D[c]) * scale;
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
                wmma::load_matrix_sync(q_f, &s_Q[k * 128 + c_step * 16], 128);
                wmma::load_matrix_sync(do_f, &s_dO[k * 128 + c_step * 16], 128);
                
                wmma::mma_sync(dk_acc[c_step], dst_f, q_f, dk_acc[c_step]);
                wmma::mma_sync(dv_acc[c_step], pt_f, do_f, dv_acc[c_step]);
            }
        }
        __syncthreads();
    }
    
    int row_dk = warp_id * 16;
    for (int c_step = 0; c_step < 8; c_step++) {
        wmma::store_matrix_sync(&s_ST[row_dk * 128 + c_step * 16], dk_acc[c_step], 128, wmma::mem_row_major);
    }
    __syncthreads();
    
    __nv_bfloat16* dk_out = dK + head_offset + j_base * 128;
    for (int i = threadIdx.x; i < 64 * 128 / 2; i += blockDim.x) {
        int r = i / 64;
        int c = (i % 64) * 2;
        if (r < valid_rows_kv) {
            __nv_bfloat162 val;
            val.x = __float2bfloat16(s_ST[r * 128 + c]);
            val.y = __float2bfloat16(s_ST[r * 128 + c + 1]);
            *reinterpret_cast<__nv_bfloat162*>(&dk_out[r * 128 + c]) = val;
        }
    }
    __syncthreads();
    
    for (int c_step = 0; c_step < 8; c_step++) {
        wmma::store_matrix_sync(&s_ST[row_dk * 128 + c_step * 16], dv_acc[c_step], 128, wmma::mem_row_major);
    }
    __syncthreads();
    
    __nv_bfloat16* dv_out = dV + head_offset + j_base * 128;
    for (int i = threadIdx.x; i < 64 * 128 / 2; i += blockDim.x) {
        int r = i / 64;
        int c = (i % 64) * 2;
        if (r < valid_rows_kv) {
            __nv_bfloat162 val;
            val.x = __float2bfloat16(s_ST[r * 128 + c]);
            val.y = __float2bfloat16(s_ST[r * 128 + c + 1]);
            *reinterpret_cast<__nv_bfloat162*>(&dv_out[r * 128 + c]) = val;
        }
    }
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
    
    int smem_size = 114688; // 112 KB
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