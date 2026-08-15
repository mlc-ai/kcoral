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

#define BQ 8
#define BK 64
#define D 128

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
    int q_block = blockIdx.y;
    int q_start = q_block * BQ;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int my_q = q_start + warp_id;

    if (q_start >= S) return;

    size_t head_offset = (size_t)(b * H + h) * S * D;
    const __nv_bfloat16* Q_h = Q + head_offset;
    const __nv_bfloat16* K_h = K + head_offset;
    const __nv_bfloat16* V_h = V + head_offset;
    const __nv_bfloat16* O_h = O + head_offset;
    const __nv_bfloat16* dO_h = dO + head_offset;
    const float* L_h = L + (size_t)(b * H + h) * S;
    __nv_bfloat16* dQ_h = dQ + head_offset;

    float4 q_reg = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 dO_reg = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 O_reg = make_float4(0.f, 0.f, 0.f, 0.f);
    float L_val = 0.f;
    float D_i = 0.f;

    if (my_q < S) {
        const __nv_bfloat16* q_ptr = Q_h + my_q * D + lane_id * 4;
        uint2 q_raw = *reinterpret_cast<const uint2*>(q_ptr);
        float2 q_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&q_raw.x));
        float2 q_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&q_raw.y));
        q_reg = make_float4(q_lo.x, q_lo.y, q_hi.x, q_hi.y);

        const __nv_bfloat16* do_ptr = dO_h + my_q * D + lane_id * 4;
        uint2 do_raw = *reinterpret_cast<const uint2*>(do_ptr);
        float2 do_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&do_raw.x));
        float2 do_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&do_raw.y));
        dO_reg = make_float4(do_lo.x, do_lo.y, do_hi.x, do_hi.y);

        const __nv_bfloat16* o_ptr = O_h + my_q * D + lane_id * 4;
        uint2 o_raw = *reinterpret_cast<const uint2*>(o_ptr);
        float2 o_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&o_raw.x));
        float2 o_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&o_raw.y));
        O_reg = make_float4(o_lo.x, o_lo.y, o_hi.x, o_hi.y);

        L_val = L_h[my_q];
        D_i = warp_reduce(dO_reg.x * O_reg.x + dO_reg.y * O_reg.y + dO_reg.z * O_reg.z + dO_reg.w * O_reg.w);
    }

    extern __shared__ char smem_buffer[];
    float* K_smem = reinterpret_cast<float*>(smem_buffer);
    float* V_smem = reinterpret_cast<float*>(K_smem + BK * D);

    float4 dQ_acc = make_float4(0.f, 0.f, 0.f, 0.f);

    int max_q = min(q_start + BQ, S) - 1;
    int num_key_blocks = max_q / BK + 1;

    for (int kb = 0; kb < num_key_blocks; ++kb) {
        int j_start = kb * BK;
        for (int idx = tid; idx < BK * D / 4; idx += blockDim.x) {
            int row = idx / (D / 4);
            int col4 = idx % (D / 4);
            int j = j_start + row;
            if (j < S) {
                uint2 k_val = *reinterpret_cast<const uint2*>(&K_h[j * D + col4 * 4]);
                float2 k_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&k_val.x));
                float2 k_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&k_val.y));
                *reinterpret_cast<float4*>(&K_smem[row * D + col4 * 4]) = make_float4(k_lo.x, k_lo.y, k_hi.x, k_hi.y);

                uint2 v_val = *reinterpret_cast<const uint2*>(&V_h[j * D + col4 * 4]);
                float2 v_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&v_val.x));
                float2 v_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&v_val.y));
                *reinterpret_cast<float4*>(&V_smem[row * D + col4 * 4]) = make_float4(v_lo.x, v_lo.y, v_hi.x, v_hi.y);
            } else {
                *reinterpret_cast<float4*>(&K_smem[row * D + col4 * 4]) = make_float4(0.f, 0.f, 0.f, 0.f);
                *reinterpret_cast<float4*>(&V_smem[row * D + col4 * 4]) = make_float4(0.f, 0.f, 0.f, 0.f);
            }
        }
        __syncthreads();

        if (my_q < S) {
            for (int j = 0; j < BK; ++j) {
                int j_global = j_start + j;
                if (j_global > my_q || j_global >= S) continue;
                float4 k_val = *reinterpret_cast<float4*>(&K_smem[j * D + lane_id * 4]);
                float4 v_val = *reinterpret_cast<float4*>(&V_smem[j * D + lane_id * 4]);
                
                float s = (q_reg.x * k_val.x + q_reg.y * k_val.y + q_reg.z * k_val.z + q_reg.w * k_val.w) * SCALE;
                s = warp_reduce(s);
                float p = expf(s - L_val);
                
                float dp = (dO_reg.x * v_val.x + dO_reg.y * v_val.y + dO_reg.z * v_val.z + dO_reg.w * v_val.w);
                dp = warp_reduce(dp);
                
                float dS = p * (dp - D_i);
                
                dQ_acc.x += dS * k_val.x * SCALE;
                dQ_acc.y += dS * k_val.y * SCALE;
                dQ_acc.z += dS * k_val.z * SCALE;
                dQ_acc.w += dS * k_val.w * SCALE;
            }
        }
        __syncthreads();
    }

    if (my_q < S) {
        __nv_bfloat16* dQ_ptr = dQ_h + my_q * D + lane_id * 4;
        __nv_bfloat162 b01 = __floats2bfloat162_rn(dQ_acc.x, dQ_acc.y);
        __nv_bfloat162 b23 = __floats2bfloat162_rn(dQ_acc.z, dQ_acc.w);
        uint2 packed;
        packed.x = *reinterpret_cast<uint32_t*>(&b01);
        packed.y = *reinterpret_cast<uint32_t*>(&b23);
        *reinterpret_cast<uint2*>(dQ_ptr) = packed;
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
    int k_block = blockIdx.y;
    int j_start = k_block * BK;
    int tid = threadIdx.x;
    int lane_id = tid % 32;
    int warp_id = tid / 32;

    if (j_start >= S) return;

    size_t head_offset = (size_t)(b * H + h) * S * D;
    const __nv_bfloat16* Q_h = Q + head_offset;
    const __nv_bfloat16* K_h = K + head_offset;
    const __nv_bfloat16* V_h = V + head_offset;
    const __nv_bfloat16* O_h = O + head_offset;
    const __nv_bfloat16* dO_h = dO + head_offset;
    const float* L_h = L + (size_t)(b * H + h) * S;
    __nv_bfloat16* dK_h = dK + head_offset;
    __nv_bfloat16* dV_h = dV + head_offset;

    extern __shared__ char smem_buffer[];
    float* K_smem = reinterpret_cast<float*>(smem_buffer);
    float* V_smem = reinterpret_cast<float*>(K_smem + BK * D);
    float* dV_smem = reinterpret_cast<float*>(V_smem + BK * D);
    float* dK_smem = reinterpret_cast<float*>(dV_smem + BK * D);
    float* Q_smem = reinterpret_cast<float*>(dK_smem + BK * D);
    float* dO_smem = reinterpret_cast<float*>(Q_smem + D);
    float* O_smem = reinterpret_cast<float*>(dO_smem + D);
    float* partial_smem = reinterpret_cast<float*>(O_smem + D);

    int j_local = tid;
    bool valid_j = (j_local < BK) && (j_start + j_local < S);

    for (int idx = tid; idx < BK * D / 4; idx += blockDim.x) {
        int row = idx / (D / 4);
        int col4 = idx % (D / 4);
        int j = j_start + row;
        if (j < S) {
            uint2 k_val = *reinterpret_cast<const uint2*>(&K_h[j * D + col4 * 4]);
            float2 k_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&k_val.x));
            float2 k_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&k_val.y));
            *reinterpret_cast<float4*>(&K_smem[row * D + col4 * 4]) = make_float4(k_lo.x, k_lo.y, k_hi.x, k_hi.y);

            uint2 v_val = *reinterpret_cast<const uint2*>(&V_h[j * D + col4 * 4]);
            float2 v_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&v_val.x));
            float2 v_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&v_val.y));
            *reinterpret_cast<float4*>(&V_smem[row * D + col4 * 4]) = make_float4(v_lo.x, v_lo.y, v_hi.x, v_hi.y);
        } else {
            *reinterpret_cast<float4*>(&K_smem[row * D + col4 * 4]) = make_float4(0.f, 0.f, 0.f, 0.f);
            *reinterpret_cast<float4*>(&V_smem[row * D + col4 * 4]) = make_float4(0.f, 0.f, 0.f, 0.f);
        }
    }

    for (int idx = tid; idx < BK * D; idx += blockDim.x) {
        dV_smem[idx] = 0.f;
        dK_smem[idx] = 0.f;
    }
    __syncthreads();

    for (int q = j_start; q < S; ++q) {
        if (tid < 32) {
            uint2 q_val = *reinterpret_cast<const uint2*>(&Q_h[q * D + tid * 4]);
            float2 q_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&q_val.x));
            float2 q_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&q_val.y));
            *reinterpret_cast<float4*>(&Q_smem[tid * 4]) = make_float4(q_lo.x, q_lo.y, q_hi.x, q_hi.y);

            uint2 do_val = *reinterpret_cast<const uint2*>(&dO_h[q * D + tid * 4]);
            float2 do_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&do_val.x));
            float2 do_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&do_val.y));
            *reinterpret_cast<float4*>(&dO_smem[tid * 4]) = make_float4(do_lo.x, do_lo.y, do_hi.x, do_hi.y);

            uint2 o_val = *reinterpret_cast<const uint2*>(&O_h[q * D + tid * 4]);
            float2 o_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&o_val.x));
            float2 o_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&o_val.y));
            *reinterpret_cast<float4*>(&O_smem[tid * 4]) = make_float4(o_lo.x, o_lo.y, o_hi.x, o_hi.y);
        }
        __syncthreads();

        float D_i = 0.f;
        if (tid < 128) {
            float d = dO_smem[tid] * O_smem[tid];
            d = warp_reduce(d);
            if (lane_id == 0) {
                partial_smem[warp_id] = d;
            }
        }
        __syncthreads();
        
        if (warp_id == 0 && lane_id < 4) {
            float sum = partial_smem[lane_id];
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, 2);
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, 1);
            if (lane_id == 0) {
                partial_smem[0] = sum;
            }
        }
        __syncthreads();
        D_i = partial_smem[0];

        if (valid_j) {
            float p_val = 0.f;
            float dp_val = 0.f;
            float4* K_vec = reinterpret_cast<float4*>(&K_smem[j_local * D]);
            float4* V_vec = reinterpret_cast<float4*>(&V_smem[j_local * D]);
            float4* Q_vec = reinterpret_cast<float4*>(Q_smem);
            float4* dO_vec = reinterpret_cast<float4*>(dO_smem);
            
            for (int d = 0; d < D / 4; ++d) {
                float4 k = K_vec[d];
                float4 v = V_vec[d];
                float4 q = Q_vec[d];
                float4 do_ = dO_vec[d];
                p_val += q.x * k.x + q.y * k.y + q.z * k.z + q.w * k.w;
                dp_val += do_.x * v.x + do_.y * v.y + do_.z * v.z + do_.w * v.w;
            }
            
            float s = p_val * SCALE;
            float p = expf(s - L_h[q]);
            float dS = p * (dp_val - D_i);

            float4* dV_vec = reinterpret_cast<float4*>(&dV_smem[j_local * D]);
            float4* dK_vec = reinterpret_cast<float4*>(&dK_smem[j_local * D]);
            for (int d = 0; d < D / 4; ++d) {
                float4 q = Q_vec[d];
                float4 do_ = dO_vec[d];
                dV_vec[d].x += p * do_.x;
                dV_vec[d].y += p * do_.y;
                dV_vec[d].z += p * do_.z;
                dV_vec[d].w += p * do_.w;
                dK_vec[d].x += dS * q.x * SCALE;
                dK_vec[d].y += dS * q.y * SCALE;
                dK_vec[d].z += dS * q.z * SCALE;
                dK_vec[d].w += dS * q.w * SCALE;
            }
        }
        __syncthreads();
    }

    if (valid_j) {
        int j = j_start + j_local;
        float4* dV_vec = reinterpret_cast<float4*>(&dV_smem[j_local * D]);
        float4* dK_vec = reinterpret_cast<float4*>(&dK_smem[j_local * D]);
        for (int d = 0; d < D / 4; ++d) {
            __nv_bfloat162 dv01 = __floats2bfloat162_rn(dV_vec[d].x, dV_vec[d].y);
            __nv_bfloat162 dv23 = __floats2bfloat162_rn(dV_vec[d].z, dV_vec[d].w);
            uint2 dv_packed;
            dv_packed.x = *reinterpret_cast<uint32_t*>(&dv01);
            dv_packed.y = *reinterpret_cast<uint32_t*>(&dv23);
            *reinterpret_cast<uint2*>(&dV_h[j * D + d * 4]) = dv_packed;

            __nv_bfloat162 dk01 = __floats2bfloat162_rn(dK_vec[d].x, dK_vec[d].y);
            __nv_bfloat162 dk23 = __floats2bfloat162_rn(dK_vec[d].z, dK_vec[d].w);
            uint2 dk_packed;
            dk_packed.x = *reinterpret_cast<uint32_t*>(&dk01);
            dk_packed.y = *reinterpret_cast<uint32_t*>(&dk23);
            *reinterpret_cast<uint2*>(&dK_h[j * D + d * 4]) = dk_packed;
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int B = 4, H = 48, d = 128;
    int64_t S = Q.size(2);

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    dim3 grid_dq(B * H, (S + BQ - 1) / BQ);
    dim3 block_dq(256);
    int smem_dq = 2 * BK * D * sizeof(float);
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

    dim3 grid_dkv(B * H, (S + BK - 1) / BK);
    dim3 block_dkv(128);
    int smem_dkv = 4 * BK * D * sizeof(float) + 3 * D * sizeof(float) + 4 * sizeof(float);
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