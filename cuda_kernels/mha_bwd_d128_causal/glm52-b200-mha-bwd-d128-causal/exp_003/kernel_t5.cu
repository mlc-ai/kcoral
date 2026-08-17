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
#define BLOCK_Y 16

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
    int q = blockIdx.y * blockDim.y + threadIdx.y;
    int d4 = threadIdx.x;
    
    if (q >= S) return;
    
    int col = d4 * 4;
    size_t head_offset = (size_t)(b * H + h) * S * D;
    
    const __nv_bfloat16* Q_h = Q + head_offset;
    const __nv_bfloat16* K_h = K + head_offset;
    const __nv_bfloat16* V_h = V + head_offset;
    const __nv_bfloat16* O_h = O + head_offset;
    const __nv_bfloat16* dO_h = dO + head_offset;
    const float* L_h = L + (size_t)(b * H + h) * S;
    __nv_bfloat16* dQ_h = dQ + head_offset;
    
    float Q_reg[4], O_reg[4], dO_reg[4];
    {
        const __nv_bfloat16* q_ptr = Q_h + q * D + col;
        uint2 q_val = *reinterpret_cast<const uint2*>(q_ptr);
        float2 q_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&q_val.x));
        float2 q_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&q_val.y));
        Q_reg[0] = q_lo.x; Q_reg[1] = q_lo.y; Q_reg[2] = q_hi.x; Q_reg[3] = q_hi.y;
        
        const __nv_bfloat16* o_ptr = O_h + q * D + col;
        uint2 o_val = *reinterpret_cast<const uint2*>(o_ptr);
        float2 o_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&o_val.x));
        float2 o_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&o_val.y));
        O_reg[0] = o_lo.x; O_reg[1] = o_lo.y; O_reg[2] = o_hi.x; O_reg[3] = o_hi.y;
        
        const __nv_bfloat16* do_ptr = dO_h + q * D + col;
        uint2 do_val = *reinterpret_cast<const uint2*>(do_ptr);
        float2 do_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&do_val.x));
        float2 do_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&do_val.y));
        dO_reg[0] = do_lo.x; dO_reg[1] = do_lo.y; dO_reg[2] = do_hi.x; dO_reg[3] = do_hi.y;
    }
    
    float L_val = L_h[q];
    float D_i = warp_reduce(dO_reg[0]*O_reg[0] + dO_reg[1]*O_reg[1] + dO_reg[2]*O_reg[2] + dO_reg[3]*O_reg[3]);
    
    float dQ_acc[4] = {0.f, 0.f, 0.f, 0.f};
    
    for (int j = 0; j <= q; ++j) {
        float K_reg[4], V_reg[4];
        {
            const __nv_bfloat16* k_ptr = K_h + j * D + col;
            uint2 k_val = *reinterpret_cast<const uint2*>(k_ptr);
            float2 k_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&k_val.x));
            float2 k_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&k_val.y));
            K_reg[0] = k_lo.x; K_reg[1] = k_lo.y; K_reg[2] = k_hi.x; K_reg[3] = k_hi.y;
            
            const __nv_bfloat16* v_ptr = V_h + j * D + col;
            uint2 v_val = *reinterpret_cast<const uint2*>(v_ptr);
            float2 v_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&v_val.x));
            float2 v_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&v_val.y));
            V_reg[0] = v_lo.x; V_reg[1] = v_lo.y; V_reg[2] = v_hi.x; V_reg[3] = v_hi.y;
        }
        
        float p_val = Q_reg[0]*K_reg[0] + Q_reg[1]*K_reg[1] + Q_reg[2]*K_reg[2] + Q_reg[3]*K_reg[3];
        p_val = warp_reduce(p_val);
        float P = expf(p_val * SCALE - L_val);
        
        float dp_val = dO_reg[0]*V_reg[0] + dO_reg[1]*V_reg[1] + dO_reg[2]*V_reg[2] + dO_reg[3]*V_reg[3];
        dp_val = warp_reduce(dp_val);
        
        float dS = P * (dp_val - D_i);
        
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            dQ_acc[c] += dS * SCALE * K_reg[c];
        }
    }
    
    __nv_bfloat16* dQ_ptr = dQ_h + q * D + col;
    __nv_bfloat162 dq01 = __floats2bfloat162_rn(dQ_acc[0], dQ_acc[1]);
    __nv_bfloat162 dq23 = __floats2bfloat162_rn(dQ_acc[2], dQ_acc[3]);
    *reinterpret_cast<__nv_bfloat162*>(&dQ_ptr[0]) = dq01;
    *reinterpret_cast<__nv_bfloat162*>(&dQ_ptr[2]) = dq23;
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
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int d4 = threadIdx.x;
    
    if (j >= S) return;
    
    int col = d4 * 4;
    size_t head_offset = (size_t)(b * H + h) * S * D;
    
    const __nv_bfloat16* Q_h = Q + head_offset;
    const __nv_bfloat16* K_h = K + head_offset;
    const __nv_bfloat16* V_h = V + head_offset;
    const __nv_bfloat16* O_h = O + head_offset;
    const __nv_bfloat16* dO_h = dO + head_offset;
    const float* L_h = L + (size_t)(b * H + h) * S;
    __nv_bfloat16* dK_h = dK + head_offset;
    __nv_bfloat16* dV_h = dV + head_offset;
    
    float K_reg[4], V_reg[4];
    {
        const __nv_bfloat16* k_ptr = K_h + j * D + col;
        uint2 k_val = *reinterpret_cast<const uint2*>(k_ptr);
        float2 k_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&k_val.x));
        float2 k_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&k_val.y));
        K_reg[0] = k_lo.x; K_reg[1] = k_lo.y; K_reg[2] = k_hi.x; K_reg[3] = k_hi.y;
        
        const __nv_bfloat16* v_ptr = V_h + j * D + col;
        uint2 v_val = *reinterpret_cast<const uint2*>(v_ptr);
        float2 v_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&v_val.x));
        float2 v_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&v_val.y));
        V_reg[0] = v_lo.x; V_reg[1] = v_lo.y; V_reg[2] = v_hi.x; V_reg[3] = v_hi.y;
    }
    
    float dK_acc[4] = {0.f, 0.f, 0.f, 0.f};
    float dV_acc[4] = {0.f, 0.f, 0.f, 0.f};
    
    for (int q = j; q < S; ++q) {
        float Q_reg[4], O_reg[4], dO_reg[4];
        {
            const __nv_bfloat16* q_ptr = Q_h + q * D + col;
            uint2 q_val = *reinterpret_cast<const uint2*>(q_ptr);
            float2 q_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&q_val.x));
            float2 q_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&q_val.y));
            Q_reg[0] = q_lo.x; Q_reg[1] = q_lo.y; Q_reg[2] = q_hi.x; Q_reg[3] = q_hi.y;
            
            const __nv_bfloat16* o_ptr = O_h + q * D + col;
            uint2 o_val = *reinterpret_cast<const uint2*>(o_ptr);
            float2 o_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&o_val.x));
            float2 o_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&o_val.y));
            O_reg[0] = o_lo.x; O_reg[1] = o_lo.y; O_reg[2] = o_hi.x; O_reg[3] = o_hi.y;
            
            const __nv_bfloat16* do_ptr = dO_h + q * D + col;
            uint2 do_val = *reinterpret_cast<const uint2*>(do_ptr);
            float2 do_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&do_val.x));
            float2 do_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&do_val.y));
            dO_reg[0] = do_lo.x; dO_reg[1] = do_lo.y; dO_reg[2] = do_hi.x; dO_reg[3] = do_hi.y;
        }
        
        float D_i = warp_reduce(dO_reg[0]*O_reg[0] + dO_reg[1]*O_reg[1] + dO_reg[2]*O_reg[2] + dO_reg[3]*O_reg[3]);
        
        float p_val = Q_reg[0]*K_reg[0] + Q_reg[1]*K_reg[1] + Q_reg[2]*K_reg[2] + Q_reg[3]*K_reg[3];
        p_val = warp_reduce(p_val);
        float P = expf(p_val * SCALE - L_h[q]);
        
        float dp_val = dO_reg[0]*V_reg[0] + dO_reg[1]*V_reg[1] + dO_reg[2]*V_reg[2] + dO_reg[3]*V_reg[3];
        dp_val = warp_reduce(dp_val);
        
        float dS = P * (dp_val - D_i);
        
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            dV_acc[c] += P * dO_reg[c];
            dK_acc[c] += dS * SCALE * Q_reg[c];
        }
    }
    
    __nv_bfloat16* dV_ptr = dV_h + j * D + col;
    __nv_bfloat162 dv01 = __floats2bfloat162_rn(dV_acc[0], dV_acc[1]);
    __nv_bfloat162 dv23 = __floats2bfloat162_rn(dV_acc[2], dV_acc[3]);
    *reinterpret_cast<__nv_bfloat162*>(&dV_ptr[0]) = dv01;
    *reinterpret_cast<__nv_bfloat162*>(&dV_ptr[2]) = dv23;
    
    __nv_bfloat16* dK_ptr = dK_h + j * D + col;
    __nv_bfloat162 dk01 = __floats2bfloat162_rn(dK_acc[0], dK_acc[1]);
    __nv_bfloat162 dk23 = __floats2bfloat162_rn(dK_acc[2], dK_acc[3]);
    *reinterpret_cast<__nv_bfloat162*>(&dK_ptr[0]) = dk01;
    *reinterpret_cast<__nv_bfloat162*>(&dK_ptr[2]) = dk23;
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int B = 4, H = 48;
    int64_t S = Q.size(2);

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    dim3 grid_dq(B * H, (S + BLOCK_Y - 1) / BLOCK_Y);
    dim3 block_dq(32, BLOCK_Y);
    
    dQ_kernel<<<grid_dq, block_dq, 0, stream>>>(
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

    dim3 grid_dkv(B * H, (S + BLOCK_Y - 1) / BLOCK_Y);
    dim3 block_dkv(32, BLOCK_Y);
    
    dKV_kernel<<<grid_dkv, block_dkv, 0, stream>>>(
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