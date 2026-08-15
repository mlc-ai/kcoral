#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
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

namespace tvm_ffi_example_cuda {

__device__ __forceinline__ uint32_t pack_bf16_fn(uint32_t fp32_a, uint32_t fp32_b) {
    __nv_bfloat16 a = __float2bfloat16(__uint_as_float(fp32_a));
    __nv_bfloat16 b = __float2bfloat16(__uint_as_float(fp32_b));
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};"
        : "=r"(result)
        : "h"(*reinterpret_cast<uint16_t*>(&a)),
          "h"(*reinterpret_cast<uint16_t*>(&b)));
    return result;
}

__device__ __forceinline__ uint32_t smem_ptr_to_u32(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__global__ void __launch_bounds__(128) mha_kernel(
    const uint16_t* __restrict__ Q_data,
    const uint16_t* __restrict__ K_data,
    const uint16_t* __restrict__ V_data,
    uint16_t* __restrict__ O_data,
    float* __restrict__ LSE_data,
    int S)
{
    int b = blockIdx.z;
    int h = blockIdx.y;
    int q_base = blockIdx.x * 64;

    if (q_base >= S) return;

    int batch_head_offset = b * (48 * S * 128) + h * (S * 128);
    const uint16_t* Q_ptr = Q_data + batch_head_offset;
    const uint16_t* K_ptr = K_data + batch_head_offset;
    const uint16_t* V_ptr = V_data + batch_head_offset;
    uint16_t* O_ptr = O_data + batch_head_offset;
    float* LSE_ptr = LSE_data + b * (48 * S) + h * S + q_base;

    extern __shared__ uint16_t smem[];
    #define SMEM_Q(r, c) smem[(r) * 136 + (c)]
    #define SMEM_K(r, c) smem[8704 + (r) * 136 + (c)]
    #define SMEM_V(r, c) smem[17408 + (r) * 65 + (c)]
    #define SMEM_P(r, c) smem[25728 + (r) * 73 + (c)]

    // Load Q
    for(int i = 0; i < 8; ++i) {
        int idx = i * 128 + threadIdx.x;
        int r = idx / 16;
        int c = (idx % 16) * 8;
        if (q_base + r < S) {
            *(int4*)&SMEM_Q(r, c) = *(const int4*)&Q_ptr[(q_base + r) * 128 + c];
        } else {
            *(int4*)&SMEM_Q(r, c) = make_int4(0, 0, 0, 0);
        }
    }

    float O_acc[16][4];
    for(int j = 0; j < 16; ++j) {
        for(int k = 0; k < 4; ++k) O_acc[j][k] = 0.0f;
    }

    float m_prev[2] = {__int_as_float(0xff800000), __int_as_float(0xff800000)}; // -INFINITY
    float l_prev[2] = {0.0f, 0.0f};

    int w = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    float scale = 0.0883883476f; // 1.0f / sqrt(128.0f)

    for (int k_idx = 0; k_idx < (S + 63) / 64; ++k_idx) {
        int k_base = k_idx * 64;

        // Load K
        for(int i = 0; i < 8; ++i) {
            int idx = i * 128 + threadIdx.x;
            int r = idx / 16;
            int c = (idx % 16) * 8;
            if (k_base + r < S) {
                *(int4*)&SMEM_K(r, c) = *(const int4*)&K_ptr[(k_base + r) * 128 + c];
            } else {
                *(int4*)&SMEM_K(r, c) = make_int4(0, 0, 0, 0);
            }
        }

        // Load V
        for(int r = 0; r < 64; ++r) {
            int c = threadIdx.x;
            if (k_base + r < S) {
                SMEM_V(c, r) = V_ptr[(k_base + r) * 128 + c];
            } else {
                SMEM_V(c, r) = 0;
            }
        }

        __syncthreads();

        float S_acc[8][4];
        for(int j = 0; j < 8; ++j) {
            for(int k = 0; k < 4; ++k) S_acc[j][k] = 0.0f;
        }

        // Compute S = Q @ K^T
        for (int step = 0; step < 8; ++step) {
            uint32_t Q_reg[4];
            int r_q = w * 16 + (lane % 16);
            int c_q = step * 16 + (lane / 16) * 8;
            uint32_t addr_q = smem_ptr_to_u32(&SMEM_Q(r_q, c_q));
            asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                : "=r"(Q_reg[0]), "=r"(Q_reg[1]), "=r"(Q_reg[2]), "=r"(Q_reg[3]) : "r"(addr_q));

            uint32_t K_reg[8][2];
            for(int j = 0; j < 8; ++j) {
                int r_k = j * 8 + (lane % 8);
                int c_k = step * 16 + ((lane % 16) / 8) * 8;
                uint32_t addr_k = smem_ptr_to_u32(&SMEM_K(r_k, c_k));
                asm("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                    : "=r"(K_reg[j][0]), "=r"(K_reg[j][1]) : "r"(addr_k));
            }

            for(int j = 0; j < 8; ++j) {
                asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                    : "=f"(S_acc[j][0]), "=f"(S_acc[j][1]), "=f"(S_acc[j][2]), "=f"(S_acc[j][3])
                    : "r"(Q_reg[0]), "r"(Q_reg[1]), "r"(Q_reg[2]), "r"(Q_reg[3]),
                      "r"(K_reg[j][0]), "r"(K_reg[j][1]),
                      "f"(S_acc[j][0]), "f"(S_acc[j][1]), "f"(S_acc[j][2]), "f"(S_acc[j][3]));
            }
        }

        // Softmax reduction for row max
        float m_local[2] = {__int_as_float(0xff800000), __int_as_float(0xff800000)};
        for(int j = 0; j < 8; ++j) {
            int c0 = j * 8 + (lane % 4) * 2;
            int c1 = c0 + 1;
            
            float v0 = S_acc[j][0] * scale;
            if (k_base + c0 >= S) v0 = __int_as_float(0xff800000);
            float v1 = S_acc[j][1] * scale;
            if (k_base + c1 >= S) v1 = __int_as_float(0xff800000);
            m_local[0] = fmaxf(m_local[0], fmaxf(v0, v1));
            S_acc[j][0] = v0;
            S_acc[j][1] = v1;
            
            float v2 = S_acc[j][2] * scale;
            if (k_base + c0 >= S) v2 = __int_as_float(0xff800000);
            float v3 = S_acc[j][3] * scale;
            if (k_base + c1 >= S) v3 = __int_as_float(0xff800000);
            m_local[1] = fmaxf(m_local[1], fmaxf(v2, v3));
            S_acc[j][2] = v2;
            S_acc[j][3] = v3;
        }

        float m_warp[2] = {m_local[0], m_local[1]};
        for(int i = 0; i < 2; ++i) {
            m_warp[i] = fmaxf(m_warp[i], __shfl_xor_sync(0xffffffff, m_warp[i], 1));
            m_warp[i] = fmaxf(m_warp[i], __shfl_xor_sync(0xffffffff, m_warp[i], 2));
        }

        float m_new[2];
        m_new[0] = fmaxf(m_prev[0], m_warp[0]);
        m_new[1] = fmaxf(m_prev[1], m_warp[1]);

        float sum_local[2] = {0.0f, 0.0f};
        for(int j = 0; j < 8; ++j) {
            float v0 = exp2f((S_acc[j][0] - m_new[0]) * 1.44269504f);
            float v1 = exp2f((S_acc[j][1] - m_new[0]) * 1.44269504f);
            S_acc[j][0] = v0;
            S_acc[j][1] = v1;
            sum_local[0] += v0 + v1;

            float v2 = exp2f((S_acc[j][2] - m_new[1]) * 1.44269504f);
            float v3 = exp2f((S_acc[j][3] - m_new[1]) * 1.44269504f);
            S_acc[j][2] = v2;
            S_acc[j][3] = v3;
            sum_local[1] += v2 + v3;
        }

        float sum_warp[2] = {sum_local[0], sum_local[1]};
        for(int i = 0; i < 2; ++i) {
            sum_warp[i] += __shfl_xor_sync(0xffffffff, sum_warp[i], 1);
            sum_warp[i] += __shfl_xor_sync(0xffffffff, sum_warp[i], 2);
        }

        float exp_m[2];
        exp_m[0] = (m_new[0] == __int_as_float(0xff800000)) ? 0.0f : exp2f((m_prev[0] - m_new[0]) * 1.44269504f);
        exp_m[1] = (m_new[1] == __int_as_float(0xff800000)) ? 0.0f : exp2f((m_prev[1] - m_new[1]) * 1.44269504f);

        for(int j = 0; j < 16; ++j) {
            O_acc[j][0] *= exp_m[0];
            O_acc[j][1] *= exp_m[0];
            O_acc[j][2] *= exp_m[1];
            O_acc[j][3] *= exp_m[1];
        }
        
        m_prev[0] = m_new[0];
        m_prev[1] = m_new[1];
        l_prev[0] = l_prev[0] * exp_m[0] + sum_warp[0];
        l_prev[1] = l_prev[1] * exp_m[1] + sum_warp[1];

        for(int j = 0; j < 8; ++j) {
            int r0 = w * 16 + (lane / 4);
            int c0 = (lane % 4) * 2 + j * 8;
            uint32_t v0 = pack_bf16_fn(*(uint32_t*)&S_acc[j][0], *(uint32_t*)&S_acc[j][1]);
            *(uint32_t*)&SMEM_P(r0, c0) = v0;

            int r1 = w * 16 + (lane / 4) + 8;
            uint32_t v1 = pack_bf16_fn(*(uint32_t*)&S_acc[j][2], *(uint32_t*)&S_acc[j][3]);
            *(uint32_t*)&SMEM_P(r1, c0) = v1;
        }

        __syncwarp();

        // Compute P @ V
        for (int step = 0; step < 4; ++step) {
            uint32_t P_reg[4];
            int r_p = w * 16 + (lane % 16);
            int c_p = step * 16 + (lane / 16) * 8;
            uint32_t addr_p = smem_ptr_to_u32(&SMEM_P(r_p, c_p));
            asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                : "=r"(P_reg[0]), "=r"(P_reg[1]), "=r"(P_reg[2]), "=r"(P_reg[3]) : "r"(addr_p));

            uint32_t V_reg[16][2];
            for(int j = 0; j < 16; ++j) {
                int r_v = j * 8 + (lane % 8);
                int c_v = step * 16 + ((lane % 16) / 8) * 8;
                uint32_t addr_v = smem_ptr_to_u32(&SMEM_V(r_v, c_v));
                asm("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                    : "=r"(V_reg[j][0]), "=r"(V_reg[j][1]) : "r"(addr_v));
            }

            for(int j = 0; j < 16; ++j) {
                asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                    : "=f"(O_acc[j][0]), "=f"(O_acc[j][1]), "=f"(O_acc[j][2]), "=f"(O_acc[j][3])
                    : "r"(P_reg[0]), "r"(P_reg[1]), "r"(P_reg[2]), "r"(P_reg[3]),
                      "r"(V_reg[j][0]), "r"(V_reg[j][1]),
                      "f"(O_acc[j][0]), "f"(O_acc[j][1]), "f"(O_acc[j][2]), "f"(O_acc[j][3]));
            }
        }

        __syncthreads();
    }

    float inv_l[2];
    inv_l[0] = (l_prev[0] > 0.0f) ? (1.0f / l_prev[0]) : 0.0f;
    inv_l[1] = (l_prev[1] > 0.0f) ? (1.0f / l_prev[1]) : 0.0f;

    for(int j = 0; j < 16; ++j) {
        O_acc[j][0] *= inv_l[0];
        O_acc[j][1] *= inv_l[0];
        O_acc[j][2] *= inv_l[1];
        O_acc[j][3] *= inv_l[1];

        int r0 = w * 16 + (lane / 4);
        int c0 = (lane % 4) * 2 + j * 8;
        uint32_t v0 = pack_bf16_fn(*(uint32_t*)&O_acc[j][0], *(uint32_t*)&O_acc[j][1]);
        *(uint32_t*)&SMEM_Q(r0, c0) = v0;

        int r1 = w * 16 + (lane / 4) + 8;
        uint32_t v1 = pack_bf16_fn(*(uint32_t*)&O_acc[j][2], *(uint32_t*)&O_acc[j][3]);
        *(uint32_t*)&SMEM_Q(r1, c0) = v1;
    }

    __syncthreads();

    // Write out O
    for(int i = 0; i < 8; ++i) {
        int idx = i * 128 + threadIdx.x;
        int r = idx / 16;
        int c = (idx % 16) * 8;
        if (q_base + r < S) {
            *(int4*)&O_ptr[(q_base + r) * 128 + c] = *(int4*)&SMEM_Q(r, c);
        }
    }

    // Write out LSE
    if ((lane % 4) == 0) {
        int r0 = w * 16 + (lane / 4);
        if (q_base + r0 < S) {
            LSE_ptr[r0] = (m_prev[0] == __int_as_float(0xff800000)) ? __int_as_float(0xff800000) : (m_prev[0] + logf(l_prev[0]));
        }
        
        int r1 = w * 16 + (lane / 4) + 8;
        if (q_base + r1 < S) {
            LSE_ptr[r1] = (m_prev[1] == __int_as_float(0xff800000)) ? __int_as_float(0xff800000) : (m_prev[1] + logf(l_prev[1]));
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);

    const uint16_t* Q_data = static_cast<const uint16_t*>(Q.data_ptr());
    const uint16_t* K_data = static_cast<const uint16_t*>(K.data_ptr());
    const uint16_t* V_data = static_cast<const uint16_t*>(V.data_ptr());
    uint16_t* O_data = static_cast<uint16_t*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());

    dim3 grid((S + 63) / 64, H, B);
    dim3 block(128);

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    // Request dynamic shared memory: 60800 bytes
    CUDA_CHECK(cudaFuncSetAttribute(mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 60800));
    
    mha_kernel<<<grid, block, 60800, stream>>>(Q_data, K_data, V_data, O_data, LSE_data, S);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}  // namespace tvm_ffi_example_cuda