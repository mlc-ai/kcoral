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

__device__ __forceinline__ uint16_t __float2bfloat16_impl(float f) {
    __nv_bfloat16 b = __float2bfloat16(f);
    return *reinterpret_cast<uint16_t*>(&b);
}

union SharedMem {
    uint16_t Q[64][136];
    uint16_t P[4][16][72];
};

extern __shared__ __align__(16) uint8_t dynamic_smem[];

__global__ __launch_bounds__(128)
void CausalAttentionKernel(
    const uint16_t* __restrict__ Q,
    const uint16_t* __restrict__ K,
    const uint16_t* __restrict__ V,
    uint16_t* __restrict__ O,
    float* __restrict__ LSE,
    int S, int H)
{
    int bx = blockIdx.x; // sequence block
    int by = blockIdx.y; // head
    int bz = blockIdx.z; // batch

    int q_start = bx * 64;
    if (q_start >= S) return;

    // SMEM setup
    SharedMem* smem_QP = (SharedMem*)dynamic_smem;
    uint16_t (*smem_K)[136] = (uint16_t (*)[136])(dynamic_smem + 17408);
    uint16_t (*smem_V)[72] = (uint16_t (*)[72])(dynamic_smem + 17408 + 17408);

    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int r_start = warp_id * 16;

    // Load Q into smem_QP->Q
    for (int i = threadIdx.x; i < 64 * 128; i += 128) {
        int r = i / 128;
        int c = i % 128;
        if (q_start + r < S) {
            smem_QP->Q[r][c] = Q[bz * H * S * 128 + by * S * 128 + (q_start + r) * 128 + c];
        } else {
            smem_QP->Q[r][c] = 0;
        }
    }
    __syncthreads();

    // Load Q from SMEM into registers
    uint32_t Q_regs[8][4];
    int q_row = r_start + (lane % 16);
    for (int k = 0; k < 8; ++k) {
        uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(&smem_QP->Q[q_row][k * 16]);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                     : "=r"(Q_regs[k][0]), "=r"(Q_regs[k][1]), "=r"(Q_regs[k][2]), "=r"(Q_regs[k][3])
                     : "r"(smem_addr));
    }

    float O_acc[16][4];
    for (int i = 0; i < 16; ++i) {
        O_acc[i][0] = O_acc[i][1] = O_acc[i][2] = O_acc[i][3] = 0.0f;
    }

    float m_i[2] = {-INFINITY, -INFINITY};
    float l_i[2] = {0.0f, 0.0f};
    float scale = 1.0f / sqrtf(128.0f);

    for (int k_start = 0; k_start <= q_start; k_start += 64) {
        // Load K, V block
        for (int i = threadIdx.x; i < 64 * 128; i += 128) {
            int r = i / 128;
            int c = i % 128;
            if (k_start + r < S) {
                smem_K[r][c] = K[bz * H * S * 128 + by * S * 128 + (k_start + r) * 128 + c];
                smem_V[c][r] = V[bz * H * S * 128 + by * S * 128 + (k_start + r) * 128 + c]; // Note V is stored transposed in SMEM
            } else {
                smem_K[r][c] = 0;
                smem_V[c][r] = 0;
            }
        }
        __syncthreads();

        float S_acc[8][4];
        for (int i = 0; i < 8; ++i) {
            S_acc[i][0] = S_acc[i][1] = S_acc[i][2] = S_acc[i][3] = 0.0f;
        }

        // Compute Q @ K^T
        for (int k = 0; k < 8; ++k) {
            for (int n_blk = 0; n_blk < 8; ++n_blk) {
                uint32_t B_regs[2];
                int k_row = n_blk * 8 + (lane % 8);
                uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(&smem_K[k_row][k * 16]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
                             : "=r"(B_regs[0]), "=r"(B_regs[1]) : "r"(smem_addr));
                
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                             : "+f"(S_acc[n_blk][0]), "+f"(S_acc[n_blk][1]), "+f"(S_acc[n_blk][2]), "+f"(S_acc[n_blk][3])
                             : "r"(Q_regs[k][0]), "r"(Q_regs[k][1]), "r"(Q_regs[k][2]), "r"(Q_regs[k][3]),
                               "r"(B_regs[0]), "r"(B_regs[1]));
            }
        }

        // Causal mask, Scale, Max
        float m_new[2] = {m_i[0], m_i[1]};
        float local_max[2] = {-INFINITY, -INFINITY};

        for (int n_blk = 0; n_blk < 8; ++n_blk) {
            int r0 = lane / 4;
            int c0 = (lane % 4) * 2;
            int global_r0 = q_start + r_start + r0;
            int global_r1 = q_start + r_start + r0 + 8;
            int global_c0 = k_start + n_blk * 8 + c0;
            
            S_acc[n_blk][0] *= scale;
            S_acc[n_blk][1] *= scale;
            S_acc[n_blk][2] *= scale;
            S_acc[n_blk][3] *= scale;
            
            if (global_c0 > global_r0 || global_r0 >= S || global_c0 >= S) S_acc[n_blk][0] = -INFINITY;
            if (global_c0 + 1 > global_r0 || global_r0 >= S || global_c0 + 1 >= S) S_acc[n_blk][1] = -INFINITY;
            if (global_c0 > global_r1 || global_r1 >= S || global_c0 >= S) S_acc[n_blk][2] = -INFINITY;
            if (global_c0 + 1 > global_r1 || global_r1 >= S || global_c0 + 1 >= S) S_acc[n_blk][3] = -INFINITY;

            local_max[0] = fmaxf(local_max[0], fmaxf(S_acc[n_blk][0], S_acc[n_blk][1]));
            local_max[1] = fmaxf(local_max[1], fmaxf(S_acc[n_blk][2], S_acc[n_blk][3]));
        }

        #pragma unroll
        for (int offset = 2; offset > 0; offset /= 2) {
            local_max[0] = fmaxf(local_max[0], __shfl_xor_sync(0xffffffff, local_max[0], offset));
            local_max[1] = fmaxf(local_max[1], __shfl_xor_sync(0xffffffff, local_max[1], offset));
        }

        m_new[0] = fmaxf(m_i[0], local_max[0]);
        m_new[1] = fmaxf(m_i[1], local_max[1]);

        float m_new_safe[2];
        m_new_safe[0] = (m_new[0] == -INFINITY) ? 0.0f : m_new[0];
        m_new_safe[1] = (m_new[1] == -INFINITY) ? 0.0f : m_new[1];

        float exp_diff[2];
        exp_diff[0] = (m_i[0] == -INFINITY) ? 0.0f : expf(m_i[0] - m_new_safe[0]);
        exp_diff[1] = (m_i[1] == -INFINITY) ? 0.0f : expf(m_i[1] - m_new_safe[1]);

        for (int i = 0; i < 16; ++i) {
            O_acc[i][0] *= exp_diff[0];
            O_acc[i][1] *= exp_diff[0];
            O_acc[i][2] *= exp_diff[1];
            O_acc[i][3] *= exp_diff[1];
        }

        float local_sum[2] = {0.0f, 0.0f};
        for (int n_blk = 0; n_blk < 8; ++n_blk) {
            S_acc[n_blk][0] = expf(S_acc[n_blk][0] - m_new_safe[0]);
            S_acc[n_blk][1] = expf(S_acc[n_blk][1] - m_new_safe[0]);
            S_acc[n_blk][2] = expf(S_acc[n_blk][2] - m_new_safe[1]);
            S_acc[n_blk][3] = expf(S_acc[n_blk][3] - m_new_safe[1]);
            
            local_sum[0] += S_acc[n_blk][0] + S_acc[n_blk][1];
            local_sum[1] += S_acc[n_blk][2] + S_acc[n_blk][3];
        }

        #pragma unroll
        for (int offset = 2; offset > 0; offset /= 2) {
            local_sum[0] += __shfl_xor_sync(0xffffffff, local_sum[0], offset);
            local_sum[1] += __shfl_xor_sync(0xffffffff, local_sum[1], offset);
        }

        float l_new[2];
        l_new[0] = l_i[0] * exp_diff[0] + local_sum[0];
        l_new[1] = l_i[1] * exp_diff[1] + local_sum[1];

        m_i[0] = m_new[0];
        m_i[1] = m_new[1];
        l_i[0] = l_new[0];
        l_i[1] = l_new[1];

        // Store to smem_P to prepare for P @ V
        for (int n_blk = 0; n_blk < 8; ++n_blk) {
            int r0 = lane / 4;
            int c0 = n_blk * 8 + (lane % 4) * 2;
            smem_QP->P[warp_id][r0][c0] = __float2bfloat16_impl(S_acc[n_blk][0]);
            smem_QP->P[warp_id][r0][c0 + 1] = __float2bfloat16_impl(S_acc[n_blk][1]);
            smem_QP->P[warp_id][r0 + 8][c0] = __float2bfloat16_impl(S_acc[n_blk][2]);
            smem_QP->P[warp_id][r0 + 8][c0 + 1] = __float2bfloat16_impl(S_acc[n_blk][3]);
        }
        __syncwarp(); // ensure warp data is visible for ldmatrix

        // Compute P @ V
        for (int d_blk = 0; d_blk < 16; ++d_blk) {
            for (int k_blk = 0; k_blk < 4; ++k_blk) {
                uint32_t P_regs[4];
                int p_row = lane % 16;
                uint32_t smem_addr_P = (uint32_t)__cvta_generic_to_shared(&smem_QP->P[warp_id][p_row][k_blk * 16]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                             : "=r"(P_regs[0]), "=r"(P_regs[1]), "=r"(P_regs[2]), "=r"(P_regs[3])
                             : "r"(smem_addr_P));
                
                uint32_t V_regs[2];
                int v_row = d_blk * 8 + (lane % 8);
                uint32_t smem_addr_V = (uint32_t)__cvta_generic_to_shared(&smem_V[v_row][k_blk * 16]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
                             : "=r"(V_regs[0]), "=r"(V_regs[1]) : "r"(smem_addr_V));
                
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                             : "+f"(O_acc[d_blk][0]), "+f"(O_acc[d_blk][1]), "+f"(O_acc[d_blk][2]), "+f"(O_acc[d_blk][3])
                             : "r"(P_regs[0]), "r"(P_regs[1]), "r"(P_regs[2]), "r"(P_regs[3]),
                               "r"(V_regs[0]), "r"(V_regs[1]));
            }
        }
        __syncthreads(); // Synchronize before overwriting SMEM in next iteration
    }

    // Final Normalize
    for (int d_blk = 0; d_blk < 16; ++d_blk) {
        O_acc[d_blk][0] /= l_i[0];
        O_acc[d_blk][1] /= l_i[0];
        O_acc[d_blk][2] /= l_i[1];
        O_acc[d_blk][3] /= l_i[1];
    }

    // Write to smem_Q for coalesced global store
    for (int d_blk = 0; d_blk < 16; ++d_blk) {
        int r0 = lane / 4;
        int c0 = d_blk * 8 + (lane % 4) * 2;
        smem_QP->Q[r_start + r0][c0] = __float2bfloat16_impl(O_acc[d_blk][0]);
        smem_QP->Q[r_start + r0][c0 + 1] = __float2bfloat16_impl(O_acc[d_blk][1]);
        smem_QP->Q[r_start + r0 + 8][c0] = __float2bfloat16_impl(O_acc[d_blk][2]);
        smem_QP->Q[r_start + r0 + 8][c0 + 1] = __float2bfloat16_impl(O_acc[d_blk][3]);
    }
    __syncthreads();

    // Store Output O
    for (int i = threadIdx.x; i < 64 * 128; i += 128) {
        int r = i / 128;
        int c = i % 128;
        if (q_start + r < S) {
            O[bz * H * S * 128 + by * S * 128 + (q_start + r) * 128 + c] = smem_QP->Q[r][c];
        }
    }

    // Store Log-Sum-Exp (LSE)
    if (lane % 4 == 0) {
        int r0 = lane / 4;
        int global_r0 = q_start + r_start + r0;
        int global_r1 = q_start + r_start + r0 + 8;
        if (global_r0 < S) {
            LSE[bz * H * S + by * S + global_r0] = m_i[0] + logf(l_i[0]);
        }
        if (global_r1 < S) {
            LSE[bz * H * S + by * S + global_r1] = m_i[1] + logf(l_i[1]);
        }
    }
}

namespace tvm_ffi_mha {
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

    dim3 blocks((S + 63) / 64, H, B);
    dim3 threads(128);

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    // Allocate max possible dynamic shared memory footprint for kernel needs
    int smem_size = 53248;
    CUDA_CHECK(cudaFuncSetAttribute(CausalAttentionKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    CausalAttentionKernel<<<blocks, threads, smem_size, stream>>>(Q_data, K_data, V_data, O_data, LSE_data, S, H);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha::run);

} // namespace tvm_ffi_mha