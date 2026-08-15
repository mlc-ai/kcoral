#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
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

namespace tvm_ffi_mha_bwd {

#define D 128
#define BQ_DQ 32
#define BK_DQ 128
#define BQ_DKV 128
#define BK_DKV 32

constexpr float SCALE = 0.08838834764f; // 1.0f / sqrtf(128.0f)

__device__ __forceinline__ float half_warp_reduce(float val) {
    val += __shfl_xor_sync(0x0000FFFF, val, 8);
    val += __shfl_xor_sync(0x0000FFFF, val, 4);
    val += __shfl_xor_sync(0x0000FFFF, val, 2);
    val += __shfl_xor_sync(0x0000FFFF, val, 1);
    return val;
}

__global__ void dQ_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    int B, int H, int S)
{
    int batch_head = blockIdx.x;
    int b = batch_head / H;
    int h = batch_head % H;
    int q_start = blockIdx.y * BQ_DQ;
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    int half_id = lane_id / 16;
    int lane_h = lane_id % 16;
    int my_q = q_start + warp_id * 2 + half_id;
    int col = lane_h * 8;
    
    if (q_start >= S) return;
    
    size_t head_offset = (size_t)(b * H + h) * S * D;
    
    const __nv_bfloat16* Q_h = Q + head_offset;
    const __nv_bfloat16* K_h = K + head_offset;
    const __nv_bfloat16* V_h = V + head_offset;
    const __nv_bfloat16* O_h = O + head_offset;
    const __nv_bfloat16* dO_h = dO + head_offset;
    const float* L_h = L + (size_t)(b * H + h) * S;
    __nv_bfloat16* dQ_h = dQ + head_offset;
    
    bool valid_q = my_q < S;
    
    float Q_reg[8], O_reg[8], dO_reg[8];
    float L_val = 0.f, D_i = 0.f;
    
    if (valid_q) {
        uint4 q_val = *reinterpret_cast<const uint4*>(&Q_h[my_q * D + col]);
        float2 q_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&q_val.x));
        float2 q_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&q_val.y));
        float2 q_lo2 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&q_val.z));
        float2 q_hi2 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&q_val.w));
        Q_reg[0] = q_lo.x; Q_reg[1] = q_lo.y; Q_reg[2] = q_hi.x; Q_reg[3] = q_hi.y;
        Q_reg[4] = q_lo2.x; Q_reg[5] = q_lo2.y; Q_reg[6] = q_hi2.x; Q_reg[7] = q_hi2.y;
        
        uint4 o_val = *reinterpret_cast<const uint4*>(&O_h[my_q * D + col]);
        float2 o_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&o_val.x));
        float2 o_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&o_val.y));
        float2 o_lo2 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&o_val.z));
        float2 o_hi2 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&o_val.w));
        O_reg[0] = o_lo.x; O_reg[1] = o_lo.y; O_reg[2] = o_hi.x; O_reg[3] = o_hi.y;
        O_reg[4] = o_lo2.x; O_reg[5] = o_lo2.y; O_reg[6] = o_hi2.x; O_reg[7] = o_hi2.y;
        
        uint4 do_val = *reinterpret_cast<const uint4*>(&dO_h[my_q * D + col]);
        float2 do_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&do_val.x));
        float2 do_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&do_val.y));
        float2 do_lo2 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&do_val.z));
        float2 do_hi2 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&do_val.w));
        dO_reg[0] = do_lo.x; dO_reg[1] = do_lo.y; dO_reg[2] = do_hi.x; dO_reg[3] = do_hi.y;
        dO_reg[4] = do_lo2.x; dO_reg[5] = do_lo2.y; dO_reg[6] = do_hi2.x; dO_reg[7] = do_hi2.y;
        
        L_val = L_h[my_q];
        D_i = half_warp_reduce(dO_reg[0]*O_reg[0] + dO_reg[1]*O_reg[1] + dO_reg[2]*O_reg[2] + dO_reg[3]*O_reg[3] +
                               dO_reg[4]*O_reg[4] + dO_reg[5]*O_reg[5] + dO_reg[6]*O_reg[6] + dO_reg[7]*O_reg[7]);
    }
    
    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* K_smem = smem;
    __nv_bfloat16* V_smem = K_smem + BK_DQ * D;
    
    float dQ_acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    
    int max_q = min(q_start + BQ_DQ, S) - 1;
    int num_k_blocks = (max_q + BK_DQ) / BK_DQ;
    
    for (int kb = 0; kb < num_k_blocks; ++kb) {
        int j_start = kb * BK_DQ;
        
        for (int idx = threadIdx.y * 32 + threadIdx.x; idx < BK_DQ * D / 8; idx += 512) {
            int row = idx / (D / 8);
            int col8 = idx % (D / 8);
            int j = j_start + row;
            if (j < S) {
                uint4 k_val = *reinterpret_cast<const uint4*>(&K_h[j * D + col8 * 8]);
                *reinterpret_cast<uint4*>(&K_smem[row * D + col8 * 8]) = k_val;
                uint4 v_val = *reinterpret_cast<const uint4*>(&V_h[j * D + col8 * 8]);
                *reinterpret_cast<uint4*>(&V_smem[row * D + col8 * 8]) = v_val;
            }
        }
        __syncthreads();
        
        if (valid_q) {
            for (int j = 0; j < BK_DQ; j++) {
                int j_global = j_start + j;
                if (j_global > my_q) break;
                
                uint4 k_raw = *reinterpret_cast<uint4*>(&K_smem[j * D + col]);
                float2 k_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&k_raw.x));
                float2 k_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&k_raw.y));
                float2 k_lo2 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&k_raw.z));
                float2 k_hi2 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&k_raw.w));
                float K_reg[8] = {k_lo.x, k_lo.y, k_hi.x, k_hi.y, k_lo2.x, k_lo2.y, k_hi2.x, k_hi2.y};
                
                uint4 v_raw = *reinterpret_cast<uint4*>(&V_smem[j * D + col]);
                float2 v_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&v_raw.x));
                float2 v_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&v_raw.y));
                float2 v_lo2 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&v_raw.z));
                float2 v_hi2 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&v_raw.w));
                float V_reg[8] = {v_lo.x, v_lo.y, v_hi.x, v_hi.y, v_lo2.x, v_lo2.y, v_hi2.x, v_hi2.y};
                
                float p_val = Q_reg[0]*K_reg[0] + Q_reg[1]*K_reg[1] + Q_reg[2]*K_reg[2] + Q_reg[3]*K_reg[3] +
                              Q_reg[4]*K_reg[4] + Q_reg[5]*K_reg[5] + Q_reg[6]*K_reg[6] + Q_reg[7]*K_reg[7];
                p_val = half_warp_reduce(p_val);
                float P = expf(p_val * SCALE - L_val);
                
                float dp_val = dO_reg[0]*V_reg[0] + dO_reg[1]*V_reg[1] + dO_reg[2]*V_reg[2] + dO_reg[3]*V_reg[3] +
                               dO_reg[4]*V_reg[4] + dO_reg[5]*V_reg[5] + dO_reg[6]*V_reg[6] + dO_reg[7]*V_reg[7];
                dp_val = half_warp_reduce(dp_val);
                
                float dS = P * (dp_val - D_i);
                
                #pragma unroll
                for (int c = 0; c < 8; c++) {
                    dQ_acc[c] += dS * SCALE * K_reg[c];
                }
            }
        }
        __syncthreads();
    }
    
    if (valid_q) {
        __nv_bfloat16* dQ_ptr = dQ_h + my_q * D + col;
        __nv_bfloat162 dq01 = __floats2bfloat162_rn(dQ_acc[0], dQ_acc[1]);
        __nv_bfloat162 dq23 = __floats2bfloat162_rn(dQ_acc[2], dQ_acc[3]);
        __nv_bfloat162 dq45 = __floats2bfloat162_rn(dQ_acc[4], dQ_acc[5]);
        __nv_bfloat162 dq67 = __floats2bfloat162_rn(dQ_acc[6], dQ_acc[7]);
        uint4 packed;
        packed.x = *reinterpret_cast<uint32_t*>(&dq01);
        packed.y = *reinterpret_cast<uint32_t*>(&dq23);
        packed.z = *reinterpret_cast<uint32_t*>(&dq45);
        packed.w = *reinterpret_cast<uint32_t*>(&dq67);
        *reinterpret_cast<uint4*>(dQ_ptr) = packed;
    }
}

__global__ void dKV_kernel(
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
    int batch_head = blockIdx.x;
    int b = batch_head / H;
    int h = batch_head % H;
    int k_start = blockIdx.y * BK_DKV;
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    int half_id = lane_id / 16;
    int lane_h = lane_id % 16;
    int my_k = k_start + warp_id * 2 + half_id;
    int col = lane_h * 8;
    
    if (k_start >= S) return;
    
    size_t head_offset = (size_t)(b * H + h) * S * D;
    
    const __nv_bfloat16* Q_h = Q + head_offset;
    const __nv_bfloat16* K_h = K + head_offset;
    const __nv_bfloat16* V_h = V + head_offset;
    const __nv_bfloat16* O_h = O + head_offset;
    const __nv_bfloat16* dO_h = dO + head_offset;
    const float* L_h = L + (size_t)(b * H + h) * S;
    __nv_bfloat16* dK_h = dK + head_offset;
    __nv_bfloat16* dV_h = dV + head_offset;
    
    bool valid_k = my_k < S;
    
    float K_reg[8], V_reg[8];
    if (valid_k) {
        uint4 k_val = *reinterpret_cast<const uint4*>(&K_h[my_k * D + col]);
        float2 k_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&k_val.x));
        float2 k_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&k_val.y));
        float2 k_lo2 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&k_val.z));
        float2 k_hi2 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&k_val.w));
        K_reg[0] = k_lo.x; K_reg[1] = k_lo.y; K_reg[2] = k_hi.x; K_reg[3] = k_hi.y;
        K_reg[4] = k_lo2.x; K_reg[5] = k_lo2.y; K_reg[6] = k_hi2.x; K_reg[7] = k_hi2.y;
        
        uint4 v_val = *reinterpret_cast<const uint4*>(&V_h[my_k * D + col]);
        float2 v_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&v_val.x));
        float2 v_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&v_val.y));
        float2 v_lo2 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&v_val.z));
        float2 v_hi2 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&v_val.w));
        V_reg[0] = v_lo.x; V_reg[1] = v_lo.y; V_reg[2] = v_hi.x; V_reg[3] = v_hi.y;
        V_reg[4] = v_lo2.x; V_reg[5] = v_lo2.y; V_reg[6] = v_hi2.x; V_reg[7] = v_hi2.y;
    }
    
    float dK_acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float dV_acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    
    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* Q_smem = smem;
    __nv_bfloat16* O_smem = Q_smem + BQ_DKV * D;
    __nv_bfloat16* dO_smem = O_smem + BQ_DKV * D;
    
    int q_start = k_start;
    while (q_start < S) {
        for (int idx = threadIdx.y * 32 + threadIdx.x; idx < BQ_DKV * D / 8; idx += 512) {
            int row = idx / (D / 8);
            int col8 = idx % (D / 8);
            int q = q_start + row;
            if (q < S) {
                uint4 q_val = *reinterpret_cast<const uint4*>(&Q_h[q * D + col8 * 8]);
                *reinterpret_cast<uint4*>(&Q_smem[row * D + col8 * 8]) = q_val;
                uint4 o_val = *reinterpret_cast<const uint4*>(&O_h[q * D + col8 * 8]);
                *reinterpret_cast<uint4*>(&O_smem[row * D + col8 * 8]) = o_val;
                uint4 do_val = *reinterpret_cast<const uint4*>(&dO_h[q * D + col8 * 8]);
                *reinterpret_cast<uint4*>(&dO_smem[row * D + col8 * 8]) = do_val;
            }
        }
        __syncthreads();
        
        if (valid_k) {
            for (int q = 0; q < BQ_DKV; q++) {
                int q_global = q_start + q;
                if (q_global >= S) break;
                if (q_global < my_k) continue;
                
                uint4 q_raw = *reinterpret_cast<uint4*>(&Q_smem[q * D + col]);
                float2 q_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&q_raw.x));
                float2 q_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&q_raw.y));
                float2 q_lo2 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&q_raw.z));
                float2 q_hi2 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&q_raw.w));
                float Q_reg[8] = {q_lo.x, q_lo.y, q_hi.x, q_hi.y, q_lo2.x, q_lo2.y, q_hi2.x, q_hi2.y};
                
                uint4 o_raw = *reinterpret_cast<uint4*>(&O_smem[q * D + col]);
                float2 o_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&o_raw.x));
                float2 o_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&o_raw.y));
                float2 o_lo2 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&o_raw.z));
                float2 o_hi2 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&o_raw.w));
                float O_reg[8] = {o_lo.x, o_lo.y, o_hi.x, o_hi.y, o_lo2.x, o_lo2.y, o_hi2.x, o_hi2.y};
                
                uint4 do_raw = *reinterpret_cast<uint4*>(&dO_smem[q * D + col]);
                float2 do_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&do_raw.x));
                float2 do_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&do_raw.y));
                float2 do_lo2 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&do_raw.z));
                float2 do_hi2 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&do_raw.w));
                float dO_reg[8] = {do_lo.x, do_lo.y, do_hi.x, do_hi.y, do_lo2.x, do_lo2.y, do_hi2.x, do_hi2.y};
                
                float D_i = half_warp_reduce(dO_reg[0]*O_reg[0] + dO_reg[1]*O_reg[1] + dO_reg[2]*O_reg[2] + dO_reg[3]*O_reg[3] +
                                             dO_reg[4]*O_reg[4] + dO_reg[5]*O_reg[5] + dO_reg[6]*O_reg[6] + dO_reg[7]*O_reg[7]);
                
                float p_val = Q_reg[0]*K_reg[0] + Q_reg[1]*K_reg[1] + Q_reg[2]*K_reg[2] + Q_reg[3]*K_reg[3] +
                              Q_reg[4]*K_reg[4] + Q_reg[5]*K_reg[5] + Q_reg[6]*K_reg[6] + Q_reg[7]*K_reg[7];
                p_val = half_warp_reduce(p_val);
                float P = expf(p_val * SCALE - L_h[q_global]);
                
                float dp_val = dO_reg[0]*V_reg[0] + dO_reg[1]*V_reg[1] + dO_reg[2]*V_reg[2] + dO_reg[3]*V_reg[3] +
                               dO_reg[4]*V_reg[4] + dO_reg[5]*V_reg[5] + dO_reg[6]*V_reg[6] + dO_reg[7]*V_reg[7];
                dp_val = half_warp_reduce(dp_val);
                
                float dS = P * (dp_val - D_i);
                
                #pragma unroll
                for (int c = 0; c < 8; c++) {
                    dV_acc[c] += P * dO_reg[c];
                    dK_acc[c] += dS * SCALE * Q_reg[c];
                }
            }
        }
        q_start += BQ_DKV;
        __syncthreads();
    }
    
    if (valid_k) {
        __nv_bfloat16* dV_ptr = dV_h + my_k * D + col;
        __nv_bfloat162 dv01 = __floats2bfloat162_rn(dV_acc[0], dV_acc[1]);
        __nv_bfloat162 dv23 = __floats2bfloat162_rn(dV_acc[2], dV_acc[3]);
        __nv_bfloat162 dv45 = __floats2bfloat162_rn(dV_acc[4], dV_acc[5]);
        __nv_bfloat162 dv67 = __floats2bfloat162_rn(dV_acc[6], dV_acc[7]);
        uint4 dv_packed;
        dv_packed.x = *reinterpret_cast<uint32_t*>(&dv01);
        dv_packed.y = *reinterpret_cast<uint32_t*>(&dv23);
        dv_packed.z = *reinterpret_cast<uint32_t*>(&dv45);
        dv_packed.w = *reinterpret_cast<uint32_t*>(&dv67);
        *reinterpret_cast<uint4*>(dV_ptr) = dv_packed;
        
        __nv_bfloat16* dK_ptr = dK_h + my_k * D + col;
        __nv_bfloat162 dk01 = __floats2bfloat162_rn(dK_acc[0], dK_acc[1]);
        __nv_bfloat162 dk23 = __floats2bfloat162_rn(dK_acc[2], dK_acc[3]);
        __nv_bfloat162 dk45 = __floats2bfloat162_rn(dK_acc[4], dK_acc[5]);
        __nv_bfloat162 dk67 = __floats2bfloat162_rn(dK_acc[6], dK_acc[7]);
        uint4 dk_packed;
        dk_packed.x = *reinterpret_cast<uint32_t*>(&dk01);
        dk_packed.y = *reinterpret_cast<uint32_t*>(&dk23);
        dk_packed.z = *reinterpret_cast<uint32_t*>(&dk45);
        dk_packed.w = *reinterpret_cast<uint32_t*>(&dk67);
        *reinterpret_cast<uint4*>(dK_ptr) = dk_packed;
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int B = 4, H = 48;
    int64_t S = Q.size(2);

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    dim3 grid_dq(B * H, (S + BQ_DQ - 1) / BQ_DQ);
    dim3 block_dq(32, 16);
    int smem_dq = 2 * BK_DQ * D * sizeof(__nv_bfloat16);
    cudaFuncSetAttribute((void*)dQ_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_dq);
    
    dQ_kernel<<<grid_dq, block_dq, smem_dq, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(O.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        B, H, (int)S
    );
    CUDA_CHECK(cudaGetLastError());

    dim3 grid_dkv(B * H, (S + BK_DKV - 1) / BK_DKV);
    dim3 block_dkv(32, 16);
    int smem_dkv = 3 * BQ_DKV * D * sizeof(__nv_bfloat16);
    cudaFuncSetAttribute((void*)dKV_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_dkv);
    
    dKV_kernel<<<grid_dkv, block_dkv, smem_dkv, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(O.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        B, H, (int)S
    );
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_bwd::run);

}  // namespace tvm_ffi_mha_bwd