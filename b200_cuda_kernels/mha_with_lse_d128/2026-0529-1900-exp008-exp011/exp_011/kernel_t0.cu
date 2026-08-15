#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <mma.h>
#include <math_constants.h>
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

using namespace nvcuda;

namespace tvm_ffi_mha {

__global__ void flash_attn_fwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S, int D)
{
    int batch_head = blockIdx.x;
    int b = batch_head / H;
    int h = batch_head % H;
    int q_start = blockIdx.y * 64;
    int q_len = min(64, S - q_start);

    if (q_len <= 0) return;

    const __nv_bfloat16* q_ptr = Q + (b * H + h) * S * D + q_start * D;
    const __nv_bfloat16* k_ptr = K + (b * H + h) * S * D;
    const __nv_bfloat16* v_ptr = V + (b * H + h) * S * D;
    __nv_bfloat16* o_ptr = O + (b * H + h) * S * D + q_start * D;
    float* lse_ptr = LSE + (b * H + h) * S + q_start;

    extern __shared__ char smem_buf[];
    // Memory map with padding to avoid bank conflicts. 
    // Strides are padded by 8 elements (16 bytes).
    auto Q_smem = (__nv_bfloat16*)smem_buf;                   // 64 x 136 elements = 17408 B
    auto K_smem = (__nv_bfloat16*)(smem_buf + 17408);         // 64 x 136 elements = 17408 B
    auto V_smem = (__nv_bfloat16*)(smem_buf + 34816);         // 64 x 136 elements = 17408 B
    auto S_smem = (float*)(smem_buf + 52224);                 // 64 x 72 elements  = 18432 B
    auto P_smem_sep = (__nv_bfloat16*)(smem_buf + 70656);     // 64 x 72 elements  = 9216 B
    auto O_smem = (float*)(smem_buf + 79872);                 // 64 x 136 elements = 34816 B
    auto m_smem = (float*)(smem_buf + 114688);                // 64 elements = 256 B
    auto l_smem = (float*)(smem_buf + 114944);                // 64 elements = 256 B

    int tid = threadIdx.x;
    int w = tid / 32;

    if (tid < 64) {
        m_smem[tid] = -CUDART_INF_F;
        l_smem[tid] = 0.0f;
    }
    
    // Zero initialize O_smem (64 * 136 = 8704 floats = 2176 float4s)
    for (int i = tid; i < 2176; i += 128) {
        reinterpret_cast<float4*>(O_smem)[i] = make_float4(0, 0, 0, 0);
    }

    // Load Q and scale by 1 / sqrt(D)
    float scale = 1.0f / sqrtf((float)D);
    for (int i = tid; i < 1024; i += 128) {
        int r = i / 16;
        int c = (i % 16) * 8; 
        
        if (r < q_len) {
            float4 val = reinterpret_cast<const float4*>(q_ptr + r * D)[i % 16];
            __nv_bfloat16* bf_ptr = (__nv_bfloat16*)&val;
            __nv_bfloat16 scaled[8];
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                scaled[k] = __float2bfloat16(__bfloat162float(bf_ptr[k]) * scale);
            }
            reinterpret_cast<float4*>(Q_smem + r * 136 + c)[0] = *reinterpret_cast<float4*>(scaled);
        } else {
            reinterpret_cast<float4*>(Q_smem + r * 136 + c)[0] = make_float4(0, 0, 0, 0);
        }
    }
    __syncthreads();

    int num_tiles = (S + 63) / 64;
    for (int tc = 0; tc < num_tiles; tc++) {
        int kv_start = tc * 64;
        int kv_len = min(64, S - kv_start);

        // Load K and V
        for (int i = tid; i < 1024; i += 128) {
            int r = i / 16;
            int c = (i % 16) * 8;
            if (r < kv_len) {
                float4 k_val = reinterpret_cast<const float4*>(k_ptr + (kv_start + r) * D)[i % 16];
                float4 v_val = reinterpret_cast<const float4*>(v_ptr + (kv_start + r) * D)[i % 16];
                reinterpret_cast<float4*>(K_smem + r * 136 + c)[0] = k_val;
                reinterpret_cast<float4*>(V_smem + r * 136 + c)[0] = v_val;
            } else {
                reinterpret_cast<float4*>(K_smem + r * 136 + c)[0] = make_float4(0, 0, 0, 0);
                reinterpret_cast<float4*>(V_smem + r * 136 + c)[0] = make_float4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        // Q @ K^T
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> accS[4];
        #pragma unroll
        for(int j=0; j<4; j++) wmma::fill_fragment(accS[j], 0.0f);

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fragQ[8];
        #pragma unroll
        for(int k=0; k<8; k++) {
            wmma::load_matrix_sync(fragQ[k], &Q_smem[w*16 * 136 + k*16], 136);
        }

        #pragma unroll
        for(int j=0; j<4; j++) {
            #pragma unroll
            for(int k=0; k<8; k++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fragK;
                wmma::load_matrix_sync(fragK, &K_smem[j*16 * 136 + k*16], 136);
                wmma::mma_sync(accS[j], fragQ[k], fragK, accS[j]);
            }
            wmma::store_matrix_sync(&S_smem[w*16 * 72 + j*16], accS[j], 72, wmma::mem_row_major);
        }
        __syncthreads();

        // Softmax & O scaling
        int row = tid / 2;
        int tid_in_row = tid % 2;
        int c_start = tid_in_row * 32;
        
        float row_max = -CUDART_INF_F;
        float4* s_row = reinterpret_cast<float4*>(S_smem + row * 72 + c_start);
        
        #pragma unroll
        for (int c = 0; c < 8; c++) {
            float4 vals = s_row[c];
            if (c_start + c * 4 + 0 < kv_len) row_max = max(row_max, vals.x);
            if (c_start + c * 4 + 1 < kv_len) row_max = max(row_max, vals.y);
            if (c_start + c * 4 + 2 < kv_len) row_max = max(row_max, vals.z);
            if (c_start + c * 4 + 3 < kv_len) row_max = max(row_max, vals.w);
        }
        row_max = max(row_max, __shfl_xor_sync(0xFFFFFFFF, row_max, 1));

        float m_old = m_smem[row];
        float m_new = max(m_old, row_max);
        float row_sum = 0.0f;
        float2* p_row = reinterpret_cast<float2*>(P_smem_sep + row * 72 + c_start);

        #pragma unroll
        for (int c = 0; c < 8; c++) {
            float4 vals = s_row[c];
            __nv_bfloat16 p[4];
            
            float p0 = 0, p1 = 0, p2 = 0, p3 = 0;
            if (c_start + c * 4 + 0 < kv_len) p0 = expf(vals.x - m_new);
            if (c_start + c * 4 + 1 < kv_len) p1 = expf(vals.y - m_new);
            if (c_start + c * 4 + 2 < kv_len) p2 = expf(vals.z - m_new);
            if (c_start + c * 4 + 3 < kv_len) p3 = expf(vals.w - m_new);
            
            row_sum += p0 + p1 + p2 + p3;
            p[0] = __float2bfloat16(p0);
            p[1] = __float2bfloat16(p1);
            p[2] = __float2bfloat16(p2);
            p[3] = __float2bfloat16(p3);
            
            p_row[c] = *reinterpret_cast<float2*>(p);
        }
        row_sum += __shfl_xor_sync(0xFFFFFFFF, row_sum, 1);

        float l_old = l_smem[row];
        float l_new = l_old * expf(m_old - m_new) + row_sum;

        if (tid_in_row == 0) {
            m_smem[row] = m_new;
            l_smem[row] = l_new;
        }

        float o_scale = (m_old == -CUDART_INF_F) ? 0.0f : expf(m_old - m_new);
        int o_start = tid_in_row * 64;
        float4* o_row = reinterpret_cast<float4*>(O_smem + row * 136 + o_start);
        
        #pragma unroll
        for (int c = 0; c < 16; c++) {
            float4 vals = o_row[c];
            vals.x *= o_scale;
            vals.y *= o_scale;
            vals.z *= o_scale;
            vals.w *= o_scale;
            o_row[c] = vals;
        }
        __syncthreads();

        // P @ V
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> accO[8];
        #pragma unroll
        for(int j=0; j<8; j++) {
            wmma::load_matrix_sync(accO[j], &O_smem[w*16 * 136 + j*16], 136, wmma::mem_row_major);
        }

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fragP[4];
        #pragma unroll
        for(int k=0; k<4; k++) {
            wmma::load_matrix_sync(fragP[k], &P_smem_sep[w*16 * 72 + k*16], 72);
        }

        #pragma unroll
        for(int j=0; j<8; j++) {
            #pragma unroll
            for(int k=0; k<4; k++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> fragV;
                wmma::load_matrix_sync(fragV, &V_smem[k*16 * 136 + j*16], 136);
                wmma::mma_sync(accO[j], fragP[k], fragV, accO[j]);
            }
            wmma::store_matrix_sync(&O_smem[w*16 * 136 + j*16], accO[j], 136, wmma::mem_row_major);
        }
        __syncthreads();
    }

    // Write out O
    for (int i = tid; i < 1024; i += 128) {
        int r = i / 16;
        int c = (i % 16) * 8;
        if (r < q_len) {
            float l_val = l_smem[r];
            __nv_bfloat16 out_vals[8];
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                out_vals[k] = __float2bfloat16(O_smem[r * 136 + c + k] / l_val);
            }
            reinterpret_cast<float4*>(o_ptr + r * D)[i % 16] = *reinterpret_cast<float4*>(out_vals);
        }
    }
    
    // Write out LSE
    if (tid < 64 && tid < q_len) {
        float m_val = m_smem[tid];
        float l_val = l_smem[tid];
        lse_ptr[tid] = m_val + logf(l_val);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int D = Q.size(3);

    auto q_ptr = static_cast<__nv_bfloat16*>(Q.data_ptr());
    auto k_ptr = static_cast<__nv_bfloat16*>(K.data_ptr());
    auto v_ptr = static_cast<__nv_bfloat16*>(V.data_ptr());
    auto o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    auto lse_ptr = static_cast<float*>(LSE.data_ptr());

    dim3 grid(B * H, (S + 63) / 64, 1);
    dim3 block(128, 1, 1);
    int smem_size = 115200;

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(flash_attn_fwd_kernel, 
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    flash_attn_fwd_kernel<<<grid, block, smem_size, stream>>>(
        q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr, B, H, S, D);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha::run);

} // namespace tvm_ffi_mha