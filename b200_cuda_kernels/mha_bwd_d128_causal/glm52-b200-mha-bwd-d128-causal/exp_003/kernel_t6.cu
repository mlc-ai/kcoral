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
#define BQ_DQ 16
#define BK_DQ 32
#define BK_DKV 16
#define BQ_DKV 32

constexpr float SCALE = 0.08838834764f; // 1.0f / sqrtf(128.0f)

__device__ __forceinline__ float warp_reduce(float val) {
    val += __shfl_xor_sync(0xFFFFFFFF, val, 16);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 8);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 4);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 2);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 1);
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
    int my_q = q_start + warp_id;
    int d4 = lane_id;
    int col = d4 * 4;
    
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
    
    float Q_reg[4], O_reg[4], dO_reg[4];
    float L_val = 0.f;
    float D_i = 0.f;
    
    if (valid_q) {
        const __nv_bfloat16* q_ptr = Q_h + my_q * D + col;
        uint2 q_val = *reinterpret_cast<const uint2*>(q_ptr);
        float2 q_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&q_val.x));
        float2 q_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&q_val.y));
        Q_reg[0] = q_lo.x; Q_reg[1] = q_lo.y; Q_reg[2] = q_hi.x; Q_reg[3] = q_hi.y;
        
        const __nv_bfloat16* o_ptr = O_h + my_q * D + col;
        uint2 o_val = *reinterpret_cast<const uint2*>(o_ptr);
        float2 o_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&o_val.x));
        float2 o_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&o_val.y));
        O_reg[0] = o_lo.x; O_reg[1] = o_lo.y; O_reg[2] = o_hi.x; O_reg[3] = o_hi.y;
        
        const __nv_bfloat16* do_ptr = dO_h + my_q * D + col;
        uint2 do_val = *reinterpret_cast<const uint2*>(do_ptr);
        float2 do_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&do_val.x));
        float2 do_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&do_val.y));
        dO_reg[0] = do_lo.x; dO_reg[1] = do_lo.y; dO_reg[2] = do_hi.x; dO_reg[3] = do_hi.y;
        
        L_val = L_h[my_q];
        D_i = warp_reduce(dO_reg[0]*O_reg[0] + dO_reg[1]*O_reg[1] + dO_reg[2]*O_reg[2] + dO_reg[3]*O_reg[3]);
    }
    
    extern __shared__ float smem[];
    float* K_smem = smem;
    float* V_smem = K_smem + BK_DQ * D;
    
    float dQ_acc[4] = {0.f, 0.f, 0.f, 0.f};
    
    int max_q = min(q_start + BQ_DQ, S) - 1;
    int num_k_blocks = (max_q + BK_DQ) / BK_DQ;
    
    for (int kb = 0; kb < num_k_blocks; ++kb) {
        int j_start = kb * BK_DQ;
        
        for (int idx = threadIdx.y * 32 + threadIdx.x; idx < BK_DQ * D / 4; idx += 512) {
            int row = idx / (D / 4);
            int col4 = idx % (D / 4);
            int j = j_start + row;
            if (j < S) {
                const __nv_bfloat16* k_ptr = K_h + j * D + col4 * 4;
                uint2 k_val = *reinterpret_cast<const uint2*>(k_ptr);
                float2 k_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&k_val.x));
                float2 k_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&k_val.y));
                *reinterpret_cast<float4*>(&K_smem[row * D + col4 * 4]) = make_float4(k_lo.x, k_lo.y, k_hi.x, k_hi.y);

                const __nv_bfloat16* v_ptr = V_h + j * D + col4 * 4;
                uint2 v_val = *reinterpret_cast<const uint2*>(v_ptr);
                float2 v_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&v_val.x));
                float2 v_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&v_val.y));
                *reinterpret_cast<float4*>(&V_smem[row * D + col4 * 4]) = make_float4(v_lo.x, v_lo.y, v_hi.x, v_hi.y);
            } else {
                *reinterpret_cast<float4*>(&K_smem[row * D + col4 * 4]) = make_float4(0.f, 0.f, 0.f, 0.f);
                *reinterpret_cast<float4*>(&V_smem[row * D + col4 * 4]) = make_float4(0.f, 0.f, 0.f, 0.f);
            }
        }
        __syncthreads();
        
        if (valid_q) {
            for (int j = 0; j < BK_DQ; j++) {
                int j_global = j_start + j;
                if (j_global > my_q) break;
                
                float K_reg[4], V_reg[4];
                *reinterpret_cast<float4*>(&K_reg[0]) = *reinterpret_cast<float4*>(&K_smem[j * D + col]);
                *reinterpret_cast<float4*>(&V_reg[0]) = *reinterpret_cast<float4*>(&V_smem[j * D + col]);
                
                float p_val = Q_reg[0]*K_reg[0] + Q_reg[1]*K_reg[1] + Q_reg[2]*K_reg[2] + Q_reg[3]*K_reg[3];
                p_val = warp_reduce(p_val);
                float P = expf(p_val * SCALE - L_val);
                
                float dp_val = dO_reg[0]*V_reg[0] + dO_reg[1]*V_reg[1] + dO_reg[2]*V_reg[2] + dO_reg[3]*V_reg[3];
                dp_val = warp_reduce(dp_val);
                
                float dS = P * (dp_val - D_i);
                
                #pragma unroll
                for (int c = 0; c < 4; c++) {
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
        *reinterpret_cast<__nv_bfloat162*>(&dQ_ptr[0]) = dq01;
        *reinterpret_cast<__nv_bfloat162*>(&dQ_ptr[2]) = dq23;
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
    int my_k = k_start + warp_id;
    int d4 = lane_id;
    int col = d4 * 4;
    
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
    
    float K_reg[4], V_reg[4];
    if (valid_k) {
        const __nv_bfloat16* k_ptr = K_h + my_k * D + col;
        uint2 k_val = *reinterpret_cast<const uint2*>(k_ptr);
        float2 k_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&k_val.x));
        float2 k_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&k_val.y));
        K_reg[0] = k_lo.x; K_reg[1] = k_lo.y; K_reg[2] = k_hi.x; K_reg[3] = k_hi.y;
        
        const __nv_bfloat16* v_ptr = V_h + my_k * D + col;
        uint2 v_val = *reinterpret_cast<const uint2*>(v_ptr);
        float2 v_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&v_val.x));
        float2 v_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&v_val.y));
        V_reg[0] = v_lo.x; V_reg[1] = v_lo.y; V_reg[2] = v_hi.x; V_reg[3] = v_hi.y;
    }
    
    float dK_acc[4] = {0.f, 0.f, 0.f, 0.f};
    float dV_acc[4] = {0.f, 0.f, 0.f, 0.f};
    
    extern __shared__ float smem[];
    float* Q_smem = smem;
    float* O_smem = Q_smem + BQ_DKV * D;
    float* dO_smem = O_smem + BQ_DKV * D;
    float* L_smem = dO_smem + BQ_DKV * D;
    
    int q_start = k_start;
    while (q_start < S) {
        for (int idx = threadIdx.y * 32 + threadIdx.x; idx < BQ_DKV * D / 4; idx += 512) {
            int row = idx / (D / 4);
            int col4 = idx % (D / 4);
            int q = q_start + row;
            if (q < S) {
                const __nv_bfloat16* q_ptr = Q_h + q * D + col4 * 4;
                uint2 q_val = *reinterpret_cast<const uint2*>(q_ptr);
                float2 q_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&q_val.x));
                float2 q_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&q_val.y));
                *reinterpret_cast<float4*>(&Q_smem[row * D + col4 * 4]) = make_float4(q_lo.x, q_lo.y, q_hi.x, q_hi.y);

                const __nv_bfloat16* o_ptr = O_h + q * D + col4 * 4;
                uint2 o_val = *reinterpret_cast<const uint2*>(o_ptr);
                float2 o_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&o_val.x));
                float2 o_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&o_val.y));
                *reinterpret_cast<float4*>(&O_smem[row * D + col4 * 4]) = make_float4(o_lo.x, o_lo.y, o_hi.x, o_hi.y);

                const __nv_bfloat16* do_ptr = dO_h + q * D + col4 * 4;
                uint2 do_val = *reinterpret_cast<const uint2*>(do_ptr);
                float2 do_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&do_val.x));
                float2 do_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&do_val.y));
                *reinterpret_cast<float4*>(&dO_smem[row * D + col4 * 4]) = make_float4(do_lo.x, do_lo.y, do_hi.x, do_hi.y);
            } else {
                *reinterpret_cast<float4*>(&Q_smem[row * D + col4 * 4]) = make_float4(0.f, 0.f, 0.f, 0.f);
                *reinterpret_cast<float4*>(&O_smem[row * D + col4 * 4]) = make_float4(0.f, 0.f, 0.f, 0.f);
                *reinterpret_cast<float4*>(&dO_smem[row * D + col4 * 4]) = make_float4(0.f, 0.f, 0.f, 0.f);
            }
        }
        if (threadIdx.y == 0) {
            int q = q_start + threadIdx.x;
            if (q < S) L_smem[threadIdx.x] = L_h[q];
            else L_smem[threadIdx.x] = 0.f;
        }
        __syncthreads();
        
        if (valid_k) {
            for (int q = 0; q < BQ_DKV; q++) {
                int q_global = q_start + q;
                if (q_global >= S) break;
                if (q_global < my_k) continue;
                
                float Q_reg[4], O_reg[4], dO_reg[4];
                *reinterpret_cast<float4*>(&Q_reg[0]) = *reinterpret_cast<float4*>(&Q_smem[q * D + col]);
                *reinterpret_cast<float4*>(&O_reg[0]) = *reinterpret_cast<float4*>(&O_smem[q * D + col]);
                *reinterpret_cast<float4*>(&dO_reg[0]) = *reinterpret_cast<float4*>(&dO_smem[q * D + col]);
                
                float D_i = warp_reduce(dO_reg[0]*O_reg[0] + dO_reg[1]*O_reg[1] + dO_reg[2]*O_reg[2] + dO_reg[3]*O_reg[3]);
                
                float p_val = Q_reg[0]*K_reg[0] + Q_reg[1]*K_reg[1] + Q_reg[2]*K_reg[2] + Q_reg[3]*K_reg[3];
                p_val = warp_reduce(p_val);
                float P = expf(p_val * SCALE - L_smem[q]);
                
                float dp_val = dO_reg[0]*V_reg[0] + dO_reg[1]*V_reg[1] + dO_reg[2]*V_reg[2] + dO_reg[3]*V_reg[3];
                dp_val = warp_reduce(dp_val);
                
                float dS = P * (dp_val - D_i);
                
                #pragma unroll
                for (int c = 0; c < 4; c++) {
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
        *reinterpret_cast<__nv_bfloat162*>(&dV_ptr[0]) = dv01;
        *reinterpret_cast<__nv_bfloat162*>(&dV_ptr[2]) = dv23;
        
        __nv_bfloat16* dK_ptr = dK_h + my_k * D + col;
        __nv_bfloat162 dk01 = __floats2bfloat162_rn(dK_acc[0], dK_acc[1]);
        __nv_bfloat162 dk23 = __floats2bfloat162_rn(dK_acc[2], dK_acc[3]);
        *reinterpret_cast<__nv_bfloat162*>(&dK_ptr[0]) = dk01;
        *reinterpret_cast<__nv_bfloat162*>(&dK_ptr[2]) = dk23;
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
    dim3 block_dq(32, BQ_DQ);
    int smem_dq = 2 * BK_DQ * D * sizeof(float);
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
    dim3 block_dkv(32, BK_DKV);
    int smem_dkv = 3 * BQ_DKV * D * sizeof(float) + BQ_DKV * sizeof(float);
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