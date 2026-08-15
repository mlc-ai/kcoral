#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>
#include <algorithm>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_mha_bwd {

constexpr int H = 48;
constexpr int BM = 64;

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ uint32_t pack_bf16_pair(__nv_bfloat16 a, __nv_bfloat16 b) {
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};"
        : "=r"(result)
        : "h"(*reinterpret_cast<uint16_t*>(&a)),
          "h"(*reinterpret_cast<uint16_t*>(&b)));
    return result;
}

__device__ __forceinline__ uint32_t smem_addr(const uint8_t* smem, int row, int col) {
    // Swizzle 32B for shape [64, 64] over all sub-block matrices mapped consecutively across rows.
    return (row * 128) + (((row % 2) ^ (col / 4)) * 8) + (col % 4) * 2;
}

__device__ __forceinline__ void load_gmem_to_smem_vec(const uint8_t* smem_ptr, const __nv_bfloat16* gmem_ptr, int32_t valid_rows, int32_t stride, int32_t col_offset, int32_t tid) {
    for (int i = 0; i < 64; i += 2) {
        int32_t global_idx = (i * stride) + col_offset + tid * 2;
        if (i < valid_rows) {
            uint2 tmp = *(const uint2*)&gmem_ptr[global_idx];
            int phys_col = (((i % 2) ^ ((tid * 2) / 4)) * 8) + ((tid * 2) % 4) * 2;
            *(uint32_t*)&smem_ptr[(i * 128) + phys_col] = tmp.x;
            *(uint32_t*)&smem_ptr[(i * 128) + phys_col + 2] = tmp.y;
        } else {
            int phys_col = (((i % 2) ^ ((tid * 2) / 4)) * 8) + ((tid * 2) % 4) * 2;
            *(uint32_t*)&smem_ptr[(i * 128) + phys_col] = 0;
            *(uint32_t*)&smem_ptr[(i * 128) + phys_col + 2] = 0;
        }
    }
}

__device__ __forceinline__ void store_smem_to_gmem_vec(const __nv_bfloat16* gmem_ptr, const uint8_t* smem_ptr, int32_t valid_rows, int32_t stride, int32_t col_offset, int32_t tid) {
    for (int i = 0; i < 64; i += 2) {
        if (i < valid_rows) {
            int phys_col = (((i % 2) ^ ((tid * 2) / 4)) * 8) + ((tid * 2) % 4) * 2;
            uint32_t tmp = *(const uint32_t*)&smem_ptr[(i * 128) + phys_col];
            int32_t global_idx = (i * stride) + col_offset + tid * 2;
            *(uint32_t*)&gmem_ptr[global_idx] = tmp;
        }
    }
}

__device__ __forceinline__ void wgmma_16x16x16_bf16_fp32(uint32_t acc, uint32_t a, uint32_t b) {
    asm volatile("wgmma.m16n8k16.transpose_b.b16 {%0}, [%1], [%2];" :: "r"(acc), "r"(a), "r"(b));
}

__device__ __forceinline__ void wgmma_16x16x16_bf16_fp32_tA(uint32_t acc, uint32_t a, uint32_t b) {
    asm volatile("wgmma.m16n8k16.transpose_a.b16 {%0}, [%1], [%2];" :: "r"(acc), "r"(a), "r"(b));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;" ::: "memory");
}

template<int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory");
}

__global__ void run_kernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V, const __nv_bfloat16* O, 
    const __nv_bfloat16* dO, const float* L, 
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV, 
    int32_t S, float alpha)
{
    int32_t head = blockIdx.y;
    int32_t batch = head / H;
    head %= H;
    int32_t batch_head = batch * H + head;
    
    int32_t i_block = blockIdx.x * BM;
    int32_t ib = i_block;
    if (ib >= S) return;
    int32_t valid_i = min(S - ib, BM);
    int tid = threadIdx.x;
    
    extern __shared__ uint8_t smem_pool[];
    
    uint8_t* smem_Q0 = smem_pool + 0;
    uint8_t* smem_Q1 = smem_pool + 8192;
    uint8_t* smem_K0 = smem_pool + 16384;
    uint8_t* smem_K1 = smem_pool + 24576;
    uint8_t* smem_V0 = smem_pool + 32768;
    uint8_t* smem_V1 = smem_pool + 40960;
    uint8_t* smem_O0 = smem_pool + 49152; 
    uint8_t* smem_O1 = smem_pool + 57344; 
    uint8_t* smem_dO0 = smem_pool + 65536;
    uint8_t* smem_dO1 = smem_pool + 73728;
    uint8_t* smem_D_Si = smem_pool + 81920; // stored seamlessly mapped identically to Q/K layout mapping (bf16)
    uint8_t* smem_D_Pi = smem_pool + 90112; // stage buffer (fp32) mapped linearized [64, 64] strictly for internal TMA/WGMMA math accumulation 
    uint8_t* smem_P = smem_pool + 106496;
    float* smem_L = (float*)(smem_pool + 114688);
    
    uint8_t* smem_dQ0 = smem_O0;
    uint8_t* smem_dQ1 = smem_O1;
    
    load_gmem_to_smem_vec(smem_Q0, Q + batch_head * S * 128 + ib * 128, valid_i, 128, 0, tid);
    load_gmem_to_smem_vec(smem_Q1, Q + batch_head * S * 128 + ib * 128 + 64, valid_i, 128, 0, tid);
    load_gmem_to_smem_vec(smem_dO0, dO + batch_head * S * 128 + ib * 128, valid_i, 128, 0, tid);
    load_gmem_to_smem_vec(smem_dO1, dO + batch_head * S * 128 + ib * 128 + 64, valid_i, 128, 0, tid);
    
    if (tid < 64) {
        smem_L[tid] = (ib + tid < S) ? L[batch_head * S + ib + tid] : 0.0f;
    }
    
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();
    
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int row_base = (warp_id / 2) * 32 + (lane_id / 2);
    int col_wg = lane_id % 2;
    
    float4 dq_reg[8] = {0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};
    
    for (int j_block = 0; j_block < S; j_block += 64) {
        int jb = j_block;
        int valid_j = min(S - jb, 64);
        
        load_gmem_to_smem_vec(smem_K0, K + batch_head * S * 128 + jb * 128, valid_j, 128, 0, tid);
        load_gmem_to_smem_vec(smem_K1, K + batch_head * S * 128 + jb * 128 + 64, valid_j, 128, 0, tid);
        load_gmem_to_smem_vec(smem_V0, V + batch_head * S * 128 + jb * 128, valid_j, 128, 0, tid);
        load_gmem_to_smem_vec(smem_V1, V + batch_head * S * 128 + jb * 128 + 64, valid_j, 128, 0, tid);
        
        cp_async_commit();
        cp_async_wait<0>();
        __syncthreads();
        
        float4 dp_reg[8] = {0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};
        for (int k = 0; k < 64; k += 16) {
            for (int col = 0; col < 8; col++) {
                uint32_t base_addr_x = ((warp_id % 2) * 128 + (lane_id / 2) * 8) * 2;
                int col_offset = col_wg * 4 + col;
                uint32_t a = smem_addr(smem_dO0, (warp_id % 2) * 32 + (lane_id / 2), k);
                uint32_t b = smem_addr(smem_V0, base_col + col_offset, k);
                wgmma_16x16x16_bf16_fp32(base_acc_x + (col * 4), a, b);
                
                uint32_t a1 = smem_addr(smem_dO1, (warp_id % 2) * 32 + (lane_id / 2), k);
                uint32_t b1 = smem_addr(smem_V1, base_col + col_offset, k);
                wgmma_16x16x16_bf16_fp32(base_acc_x + (col * 4), a1, b1);
            }
        }
        
        float4 s_reg[8] = {0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};
        for (int k = 0; k < 64; k += 16) {
            for (int col = 0; col < 8; col++) {
                uint32_t base_addr_x = ((warp_id % 2) * 128 + (lane_id / 2) * 8) * 2;
                int col_offset = col_wg * 4 + col;
                uint32_t a = smem_addr(smem_Q0, (warp_id % 2) * 32 + (lane_id / 2), k);
                uint32_t b = smem_addr(smem_K0, base_col + col_offset, k);
                wgmma_16x16x16_bf16_fp32(base_acc_x + (col * 4), a, b);
                
                uint32_t a1 = smem_addr(smem_Q1, (warp_id % 2) * 32 + (lane_id / 2), k);
                uint32_t b1 = smem_addr(smem_K1, base_col + col_offset, k);
                wgmma_16x16x16_bf16_fp32(base_acc_x + (col * 4), a1, b1);
            }
        }
        
        __syncthreads(); 
        
        for(int r = 0; r < 4; ++r) {
            float f0 = __uint_as_float(((uint32_t*)&dp_reg[r])[0]);
            float f1 = __uint_as_float(((uint32_t*)&dp_reg[r])[1]);
            float f2 = __uint_as_float(((uint32_t*)&dp_reg[r])[2]);
            float f3 = __uint_as_float(((uint32_t*)&dp_reg[r])[3]);
            
            int col = r * 2 + col_wg * 2;
            int global_col = jb + col;
            
            float p0 = (global_col < S && ib + row_base < S) ? fast_exp2f_fn(((float)__uint_as_float(((uint32_t*)&s_reg[r])[0]) * alpha - smem_L[row_base]) * 1.44269504f) : 0.0f;
            float p1 = (global_col < S && ib + row_base < S) ? fast_exp2f_fn(((float)__uint_as_float(((uint32_t*)&s_reg[r])[1]) * alpha - smem_L[row_base]) * 1.44269504f) : 0.0f;
            float p2 = (global_col < S && ib + row_base < S) ? fast_exp2f_fn(((float)__uint_as_float(((uint32_t*)&s_reg[r])[2]) * alpha - smem_L[row_base]) * 1.44269504f) : 0.0f;
            float p3 = (global_col < S && ib + row_base < S) ? fast_exp2f_fn(((float)__uint_as_float(((uint32_t*)&s_reg[r])[3]) * alpha - smem_L[row_base]) * 1.44269504f) : 0.0f;
            
            dp_reg[r] = make_float4(f0 * p0, f1 * p1, f2 * p2, f3 * p3);
        }
        
        __syncthreads();
        
        float row_sum_corr = 0.0f;
        for(int r = 0; r < 4; ++r) {
            row_sum_corr += __uint_as_float(((uint32_t*)&dp_reg[r])[0]);
            row_sum_corr += __uint_as_float(((uint32_t*)&dp_reg[r])[1]);
            row_sum_corr += __uint_as_float(((uint32_t*)&dp_reg[r])[2]);
            row_sum_corr += __uint_as_float(((uint32_t*)&dp_reg[r])[3]);
        }
        
        int src_lane = (tid % 32) ^ 1;
        row_sum_corr += __shfl_sync(0xffffffff, row_sum_corr, src_lane);
        
        for(int r = 0; r < 4; ++r) {
            float f0 = __uint_as_float(((uint32_t*)&dp_reg[r])[0]);
            float f1 = __uint_as_float(((uint32_t*)&dp_reg[r])[1]);
            float f2 = __uint_as_float(((uint32_t*)&dp_reg[r])[2]);
            float f3 = __uint_as_float(((uint32_t*)&dp_reg[r])[3]);
            
            float p0 = (jb + r * 2 + col_wg * 2 + 0 < S && ib + row_base < S) ? fast_exp2f_fn(((float)__uint_as_float(((uint32_t*)&s_reg[r])[0]) * alpha - smem_L[row_base]) * 1.44269504f) : 0.0f;
            float p1 = (jb + r * 2 + col_wg * 2 + 1 < S && ib + row_base < S) ? fast_exp2f_fn(((float)__uint_as_float(((uint32_t*)&s_reg[r])[1]) * alpha - smem_L[row_base]) * 1.44269504f) : 0.0f;
            float p2 = (jb + r * 2 + col_wg * 2 + 2 < S && ib + row_base < S) ? fast_exp2f_fn(((float)__uint_as_float(((uint32_t*)&s_reg[r])[2]) * alpha - smem_L[row_base]) * 1.44269504f) : 0.0f;
            float p3 = (jb + r * 2 + col_wg * 2 + 3 < S && ib + row_base < S) ? fast_exp2f_fn(((float)__uint_as_float(((uint32_t*)&s_reg[r])[3]) * alpha - smem_L[row_base]) * 1.44269504f) : 0.0f;
            
            f0 = f0 - p0 * row_sum_corr;
            f1 = f1 - p1 * row_sum_corr;
            f2 = f2 - p2 * row_sum_corr;
            f3 = f3 - p3 * row_sum_corr;
            
            dp_reg[r] = make_float4(f0, f1, f2, f3);
        }
        
        __syncthreads(); 
        
        for(int r = 0; r < 4; ++r) {
            int col = r * 2 + col_wg * 2;
            int global_col = jb + col;
            
            if (global_col + 3 < S && ib + row_base < S) {
                *(uint32_t*)&smem_D_Si[(row_base * 128) + (((row_base % 2) ^ (col / 4)) * 8) + (col % 4) * 2] = pack_bf16_pair(__float2bfloat16(__uint_as_float(((uint32_t*)&dp_reg[r])[0])), __float2bfloat16(__uint_as_float(((uint32_t*)&dp_reg[r])[1])));
                *(uint32_t*)&smem_D_Si[(row_base * 128) + (((row_base % 2) ^ (col / 4)) * 8) + (col % 4) * 2 + 2] = pack_bf16_pair(__float2bfloat16(__uint_as_float(((uint32_t*)&dp_reg[r])[2])), __float2bfloat16(__uint_as_float(((uint32_t*)&dp_reg[r])[3])));
                
                *(uint32_t*)&smem_P[(row_base * 128) + (((row_base % 2) ^ (col / 4)) * 8) + (col % 4) * 2] = pack_bf16_pair(__float2bfloat16(__uint_as_float(((uint32_t*)&s_reg[r])[0])), __float2bfloat16(__uint_as_float(((uint32_t*)&s_reg[r])[1])));
                *(uint32_t*)&smem_P[(row_base * 128) + (((row_base % 2) ^ (col / 4)) * 8) + (col % 4) * 2 + 2] = pack_bf16_pair(__float2bfloat16(__uint_as_float(((uint32_t*)&s_reg[r])[2])), __float2bfloat16(__uint_as_float(((uint32_t*)&s_reg[r])[3])));
            }
        }
        
        __syncthreads(); 
        
        for (int k = 0; k < 64; k += 16) {
            for (int col = 0; col < 8; col++) {
                uint32_t base_addr_x = ((warp_id % 2) * 128 + (lane_id / 2) * 8) * 2;
                int col_offset = col_wg * 4 + col;
                uint32_t a = smem_addr(smem_D_Si, (warp_id % 2) * 32 + (lane_id / 2), k);
                uint32_t b = smem_addr(smem_K0, base_col + col_offset, k);
                wgmma_16x16x16_bf16_fp32(base_acc_x + (col * 4), a, b);
                
                uint32_t a1 = smem_addr(smem_D_Si, (warp_id % 2) * 32 + (lane_id / 2), k);
                uint32_t b1 = smem_addr(smem_K1, base_col + col_offset, k);
                wgmma_16x16x16_bf16_fp32(base_acc_x + (col * 4), a1, b1);
            }
        }
        
        float4 dk_reg[8] = {0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};
        for (int k = 0; k < 64; k += 16) {
            for (int col = 0; col < 8; col++) {
                uint32_t base_addr_x = ((warp_id % 2) * 128 + (lane_id / 2) * 8) * 2;
                int col_offset = col_wg * 4 + col;
                uint32_t a = smem_addr(smem_D_Si, base_col + col_offset, k);
                uint32_t b = smem_addr(smem_Q0, k, base_col + col_offset);
                wgmma_16x16x16_bf16_fp32_tA(base_acc_x + (col * 4), a, b);
                
                uint32_t a1 = smem_addr(smem_D_Si, base_col + col_offset, k);
                uint32_t b1 = smem_addr(smem_Q1, k, base_col + col_offset);
                wgmma_16x16x16_bf16_fp32_tA(base_acc_x + (col * 4), a1, b1);
            }
        }
        
        float4 dv_reg[8] = {0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};
        for (int k = 0; k < 64; k += 16) {
            for (int col = 0; col < 8; col++) {
                uint32_t base_addr_x = ((warp_id % 2) * 128 + (lane_id / 2) * 8) * 2;
                int col_offset = col_wg * 4 + col;
                uint32_t a = smem_addr(smem_P, base_col + col_offset, k);
                uint32_t b = smem_addr(smem_dO0, k, base_col + col_offset);
                wgmma_16x16x16_bf16_fp32_tA(base_acc_x + (col * 4), a, b);
                
                uint32_t a1 = smem_addr(smem_P, base_col + col_offset, k);
                uint32_t b1 = smem_addr(smem_dO1, k, base_col + col_offset);
                wgmma_16x16x16_bf16_fp32_tA(base_acc_x + (col * 4), a1, b1);
            }
        }
        
        __syncthreads(); 
        
        for(int r = 0; r < 4; ++r) {
            int col = r * 2 + col_wg * 2;
            int global_col = jb + col;
            
            if (global_col + 3 < S) {
                *(uint32_t*)&smem_dK0[(row_base * 128) + (((row_base % 2) ^ (col / 4)) * 8) + (col % 4) * 2] = pack_bf16_pair(__float2bfloat16(__uint_as_float(((uint32_t*)&dk_reg[r])[0])), __float2bfloat16(__uint_as_float(((uint32_t*)&dk_reg[r])[1])));
                *(uint32_t*)&smem_dK0[(row_base * 128) + (((row_base % 2) ^ (col / 4)) * 8) + (col % 4) * 2 + 2] = pack_bf16_pair(__float2bfloat16(__uint_as_float(((uint32_t*)&dk_reg[r])[2])), __float2bfloat16(__uint_as_float(((uint32_t*)&dk_reg[r])[3])));
                
                *(uint32_t*)&smem_dK1[(row_base * 128) + (((row_base % 2) ^ (col / 4)) * 8) + (col % 4) * 2] = pack_bf16_pair(__float2bfloat16(__uint_as_float(((uint32_t*)&dk_reg[r+4])[0])), __float2bfloat16(__uint_as_float(((uint32_t*)&dk_reg[r+4])[1])));
                *(uint32_t*)&smem_dK1[(row_base * 128) + (((row_base % 2) ^ (col / 4)) * 8) + (col % 4) * 2 + 2] = pack_bf16_pair(__float2bfloat16(__uint_as_float(((uint32_t*)&dk_reg[r+4])[2])), __float2bfloat16(__uint_as_float(((uint32_t*)&dk_reg[r+4])[3])));
                
                *(uint32_t*)&smem_dV0[(row_base * 128) + (((row_base % 2) ^ (col / 4)) * 8) + (col % 4) * 2] = pack_bf16_pair(__float2bfloat16(__uint_as_float(((uint32_t*)&dv_reg[r])[0])), __float2bfloat16(__uint_as_float(((uint32_t*)&dv_reg[r])[1])));
                *(uint32_t*)&smem_dV0[(row_base * 128) + (((row_base % 2) ^ (col / 4)) * 8) + (col % 4) * 2 + 2] = pack_bf16_pair(__float2bfloat16(__uint_as_float(((uint32_t*)&dv_reg[r])[2])), __float2bfloat16(__uint_as_float(((uint32_t*)&dv_reg[r])[3])));
                
                *(uint32_t*)&smem_dV1[(row_base * 128) + (((row_base % 2) ^ (col / 4)) * 8) + (col % 4) * 2] = pack_bf16_pair(__float2bfloat16(__uint_as_float(((uint32_t*)&dv_reg[r+4])[0])), __float2bfloat16(__uint_as_float(((uint32_t*)&dv_reg[r+4])[1])));
                *(uint32_t*)&smem_dV1[(row_base * 128) + (((row_base % 2) ^ (col / 4)) * 8) + (col % 4) * 2 + 2] = pack_bf16_pair(__float2bfloat16(__uint_as_float(((uint32_t*)&dv_reg[r+4])[2])), __float2bfloat16(__uint_as_float(((uint32_t*)&dv_reg[r+4])[3])));
            }
        }
        
        __syncthreads();
        store_smem_to_gmem_vec(dK + batch_head * S * 128 + jb * 128, smem_dK0, valid_j, 128, 0, tid);
        store_smem_to_gmem_vec(dK + batch_head * S * 128 + jb * 128 + 64, smem_dK1, valid_j, 128, 0, tid);
        store_smem_to_gmem_vec(dV + batch_head * S * 128 + jb * 128, smem_dV0, valid_j, 128, 0, tid);
        store_smem_to_gmem_vec(dV + batch_head * S * 128 + jb * 128 + 64, smem_dV1, valid_j, 128, 0, tid);
    }
    
    __syncthreads(); 
    
    for(int r = 0; r < 4; ++r) {
        int col = r * 2 + col_wg * 2;
        *(uint32_t*)&smem_dQ0[(row_base * 128) + (((row_base % 2) ^ (col / 4)) * 8) + (col % 4) * 2] = pack_bf16_pair(__float2bfloat16(__uint_as_float(((uint32_t*)&dq_reg[r])[0])), __float2bfloat16(__uint_as_float(((uint32_t*)&dq_reg[r])[1])));
        *(uint32_t*)&smem_dQ0[(row_base * 128) + (((row_base % 2) ^ (col / 4)) * 8) + (col % 4) * 2 + 2] = pack_bf16_pair(__float2bfloat16(__uint_as_float(((uint32_t*)&dq_reg[r])[2])), __float2bfloat16(__uint_as_float(((uint32_t*)&dq_reg[r])[3])));
        
        *(uint32_t*)&smem_dQ1[(row_base * 128) + (((row_base % 2) ^ (col / 4)) * 8) + (col % 4) * 2] = pack_bf16_pair(__float2bfloat16(__uint_as_float(((uint32_t*)&dq_reg[r+4])[0])), __float2bfloat16(__uint_as_float(((uint32_t*)&dq_reg[r+4])[1])));
        *(uint32_t*)&smem_dQ1[(row_base * 128) + (((row_base % 2) ^ (col / 4)) * 8) + (col % 4) * 2 + 2] = pack_bf16_pair(__float2bfloat16(__uint_as_float(((uint32_t*)&dq_reg[r+4])[2])), __float2bfloat16(__uint_as_float(((uint32_t*)&dq_reg[r+4])[3])));
    }
    
    __syncthreads();
    store_smem_to_gmem_vec(dQ + batch_head * S * 128 + ib * 128, smem_dQ0, valid_i, 128, 0, tid);
    store_smem_to_gmem_vec(dQ + batch_head * S * 128 + ib * 128 + 64, smem_dQ1, valid_i, 128, 0, tid);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L, 
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
         
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t b = Q.size(0); 
    int64_t h = Q.size(1);
    int64_t s = Q.size(2); 
    int64_t d = Q.size(3); 
    
    const __nv_bfloat16* q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* k_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* v_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* o_ptr = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* do_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* l_ptr = static_cast<const float*>(L.data_ptr());
    
    __nv_bfloat16* dq_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dk_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dv_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    int64_t threads = 128;
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    dim3 grid((s + BM - 1) / BM, b * h); 
    
    int smem_size = 105856 + 1024; 
    
    CUDA_CHECK(cudaFuncSetAttribute(run_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    run_kernel<<<grid, threads, smem_size, stream>>>(
        q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, l_ptr, 
        dq_ptr, dk_ptr, dv_ptr, 
        s, 1.0f / sqrtf((float)d)
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_bwd::run);

} // namespace tvm_ffi_mha_bwd