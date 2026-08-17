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

#define BQ 16
#define BK 16
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
    int q_local = warp_id;
    int my_q = q_start + q_local;
    int col = lane_id * 4;

    if (q_start >= S) return;

    size_t head_offset = (size_t)(b * H + h) * S * D;
    const __nv_bfloat16* Q_h = Q + head_offset;
    const __nv_bfloat16* K_h = K + head_offset;
    const __nv_bfloat16* V_h = V + head_offset;
    const __nv_bfloat16* O_h = O + head_offset;
    const __nv_bfloat16* dO_h = dO + head_offset;
    const float* L_h = L + (size_t)(b * H + h) * S;
    __nv_bfloat16* dQ_h = dQ + head_offset;

    float q_reg[4] = {0.f, 0.f, 0.f, 0.f};
    float dO_reg[4] = {0.f, 0.f, 0.f, 0.f};
    float O_reg[4] = {0.f, 0.f, 0.f, 0.f};
    float L_val = 0.f;
    float D_i = 0.f;

    if (my_q < S) {
        const __nv_bfloat16* q_ptr = Q_h + my_q * D + col;
        uint2 q_raw = *reinterpret_cast<const uint2*>(q_ptr);
        float2 q_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&q_raw.x));
        float2 q_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&q_raw.y));
        q_reg[0] = q_lo.x; q_reg[1] = q_lo.y; q_reg[2] = q_hi.x; q_reg[3] = q_hi.y;

        const __nv_bfloat16* do_ptr = dO_h + my_q * D + col;
        uint2 do_raw = *reinterpret_cast<const uint2*>(do_ptr);
        float2 do_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&do_raw.x));
        float2 do_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&do_raw.y));
        dO_reg[0] = do_lo.x; dO_reg[1] = do_lo.y; dO_reg[2] = do_hi.x; dO_reg[3] = do_hi.y;

        const __nv_bfloat16* o_ptr = O_h + my_q * D + col;
        uint2 o_raw = *reinterpret_cast<const uint2*>(o_ptr);
        float2 o_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&o_raw.x));
        float2 o_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&o_raw.y));
        O_reg[0] = o_lo.x; O_reg[1] = o_lo.y; O_reg[2] = o_hi.x; O_reg[3] = o_hi.y;

        L_val = L_h[my_q];
        float d = dO_reg[0] * O_reg[0] + dO_reg[1] * O_reg[1] + dO_reg[2] * O_reg[2] + dO_reg[3] * O_reg[3];
        D_i = warp_reduce(d);
    }

    extern __shared__ char smem_buffer[];
    float* K_smem = reinterpret_cast<float*>(smem_buffer);
    float* V_smem = reinterpret_cast<float*>(K_smem + BK * D);

    float dQ_acc[4] = {0.f, 0.f, 0.f, 0.f};

    int max_q = min(q_start + BQ, S);
    int num_k_blocks = (max_q + BK - 1) / BK;

    for (int kb = 0; kb < num_k_blocks; ++kb) {
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
            int j_limit = my_q - j_start + 1;
            if (j_limit > 0) {
                j_limit = min(j_limit, BK);
                for (int j = 0; j < j_limit; ++j) {
                    float p_val = 0.f, dp_val = 0.f;
                    #pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        p_val += q_reg[c] * K_smem[j * D + col + c];
                        dp_val += dO_reg[c] * V_smem[j * D + col + c];
                    }
                    p_val = warp_reduce(p_val);
                    dp_val = warp_reduce(dp_val);
                    float P = expf(p_val * SCALE - L_val);
                    float dS = P * (dp_val - D_i);
                    #pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        dQ_acc[c] += dS * SCALE * K_smem[j * D + col + c];
                    }
                }
            }
        }
        __syncthreads();
    }

    if (my_q < S) {
        __nv_bfloat16* dQ_ptr = dQ_h + my_q * D + col;
        __nv_bfloat162 b01 = __floats2bfloat162_rn(dQ_acc[0], dQ_acc[1]);
        __nv_bfloat162 b23 = __floats2bfloat162_rn(dQ_acc[2], dQ_acc[3]);
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
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int j_local = warp_id;
    int my_k = j_start + j_local;
    int col = lane_id * 4;

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
    float* O_smem = reinterpret_cast<float*>(Q_smem + BQ * D);
    float* dO_smem = reinterpret_cast<float*>(O_smem + BQ * D);
    float* D_smem = reinterpret_cast<float*>(dO_smem + BQ * D);
    float* L_smem = reinterpret_cast<float*>(D_smem + BQ);

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

    int q_start = j_start;
    while (q_start < S) {
        for (int idx = tid; idx < BQ * D / 4; idx += blockDim.x) {
            int row = idx / (D / 4);
            int col4 = idx % (D / 4);
            int q = q_start + row;
            if (q < S) {
                uint2 q_val = *reinterpret_cast<const uint2*>(&Q_h[q * D + col4 * 4]);
                float2 q_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&q_val.x));
                float2 q_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&q_val.y));
                *reinterpret_cast<float4*>(&Q_smem[row * D + col4 * 4]) = make_float4(q_lo.x, q_lo.y, q_hi.x, q_hi.y);

                uint2 o_val = *reinterpret_cast<const uint2*>(&O_h[q * D + col4 * 4]);
                float2 o_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&o_val.x));
                float2 o_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&o_val.y));
                *reinterpret_cast<float4*>(&O_smem[row * D + col4 * 4]) = make_float4(o_lo.x, o_lo.y, o_hi.x, o_hi.y);

                uint2 do_val = *reinterpret_cast<const uint2*>(&dO_h[q * D + col4 * 4]);
                float2 do_lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&do_val.x));
                float2 do_hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&do_val.y));
                *reinterpret_cast<float4*>(&dO_smem[row * D + col4 * 4]) = make_float4(do_lo.x, do_lo.y, do_hi.x, do_hi.y);
            } else {
                *reinterpret_cast<float4*>(&Q_smem[row * D + col4 * 4]) = make_float4(0.f, 0.f, 0.f, 0.f);
                *reinterpret_cast<float4*>(&O_smem[row * D + col4 * 4]) = make_float4(0.f, 0.f, 0.f, 0.f);
                *reinterpret_cast<float4*>(&dO_smem[row * D + col4 * 4]) = make_float4(0.f, 0.f, 0.f, 0.f);
            }
        }
        if (tid < BQ) {
            int q = q_start + tid;
            if (q < S) {
                L_smem[tid] = L_h[q];
            } else {
                L_smem[tid] = 0.f;
            }
        }
        __syncthreads();

        if (warp_id < BQ) {
            float d = 0.f;
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                d += dO_smem[warp_id * D + col + c] * O_smem[warp_id * D + col + c];
            }
            d = warp_reduce(d);
            if (lane_id == 0) {
                D_smem[warp_id] = d;
            }
        }
        __syncthreads();

        if (my_k < S) {
            int i_start = max(0, my_k - q_start);
            for (int i = i_start; i < BQ; ++i) {
                float p_val = 0.f, dp_val = 0.f;
                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    p_val += Q_smem[i * D + col + c] * K_smem[j_local * D + col + c];
                    dp_val += dO_smem[i * D + col + c] * V_smem[j_local * D + col + c];
                }
                p_val = warp_reduce(p_val);
                dp_val = warp_reduce(dp_val);
                float P = expf(p_val * SCALE - L_smem[i]);
                float dS = P * (dp_val - D_smem[i]);
                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    dV_smem[j_local * D + col + c] += P * dO_smem[i * D + col + c];
                    dK_smem[j_local * D + col + c] += dS * SCALE * Q_smem[i * D + col + c];
                }
            }
        }
        q_start += BQ;
        __syncthreads();
    }

    if (my_k < S) {
        float4 dv = *reinterpret_cast<float4*>(&dV_smem[j_local * D + col]);
        __nv_bfloat162 dv01 = __floats2bfloat162_rn(dv.x, dv.y);
        __nv_bfloat162 dv23 = __floats2bfloat162_rn(dv.z, dv.w);
        uint2 dv_packed;
        dv_packed.x = *reinterpret_cast<uint32_t*>(&dv01);
        dv_packed.y = *reinterpret_cast<uint32_t*>(&dv23);
        *reinterpret_cast<uint2*>(&dV_h[my_k * D + col]) = dv_packed;

        float4 dk = *reinterpret_cast<float4*>(&dK_smem[j_local * D + col]);
        __nv_bfloat162 dk01 = __floats2bfloat162_rn(dk.x, dk.y);
        __nv_bfloat162 dk23 = __floats2bfloat162_rn(dk.z, dk.w);
        uint2 dk_packed;
        dk_packed.x = *reinterpret_cast<uint32_t*>(&dk01);
        dk_packed.y = *reinterpret_cast<uint32_t*>(&dk23);
        *reinterpret_cast<uint2*>(&dK_h[my_k * D + col]) = dk_packed;
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

    dim3 grid_dq(B * H, (S + BQ - 1) / BQ);
    dim3 block_dq(512);
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
    dim3 block_dkv(512);
    int smem_dkv = 4 * BK * D * sizeof(float) + 3 * BQ * D * sizeof(float) + 2 * BQ * sizeof(float);
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