#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cfloat>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_causal_d128 {

constexpr int BLOCK_M = 64;  // Q rows per tile
constexpr int GROUP_D = 4;   // Output dims per thread

__global__ void mha_causal_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
          __nv_bfloat16* __restrict__ O,
    float*                 __restrict__ LSE_out,
    int B, int H, int S, int D,
    int64_t q_stride_s, int64_t q_stride_d,
    int64_t k_stride_s, int64_t k_stride_d,
    int64_t v_stride_s, int64_t v_stride_d,
    int64_t o_stride_s, int64_t o_stride_d,
    float scale)
{
    int bh = blockIdx.x;                  // 0..B*H-1
    int b = bh / H;
    int h = bh % H;

    // Base offsets using strides
    int64_t q_base = static_cast<int64_t>(b) * q_stride_s * S + static_cast<int64_t>(h) * q_stride_s;
    int64_t k_base = q_base;  // Same layout assumed
    int64_t v_base = q_base;  // Same layout assumed
    int64_t o_base = (int64_t)b * o_stride_s * S + (int64_t)h * o_stride_s;

    int q_tile_start = blockIdx.y * BLOCK_M;
    if (q_tile_start >= S) return;

    int tid = threadIdx.x;   // 0..63
    int d0 = tid * GROUP_D;  // Output dims [d0, d0+1, d0+2, d0+3]
    if (d0 >= D) return;

    int q_local = tid;
    int q_global = q_tile_start + q_local;
    if (q_global >= S) return;

    // Online softmax state
    float row_max = -FLT_MAX;
    float row_sum = 0.f;
    float acc[GROUP_D] = {0.f, 0.f, 0.f, 0.f};

    // Scan through all key positions
    for (int k = 0; k < S; ++k) {
        // Compute attention logit: s = Q[q_global,:] · K[k,:] * scale
        float s = 0.f;
        int64_t q_off = q_base + (int64_t)q_global * q_stride_s;
        int64_t k_off = k_base + (int64_t)k * k_stride_s;
        
        for (int d = 0; d < D; d += 4) {
            float qa = __bfloat162float(Q[q_off + d]);
            float qb = __bfloat162float(Q[q_off + d + 1]);
            float qc = __bfloat162float(Q[q_off + d + 2]);
            float qd = __bfloat162float(Q[q_off + d + 3]);
            float ka = __bfloat162float(K[k_off + d]);
            float kb = __bfloat162float(K[k_off + d + 1]);
            float kc = __bfloat162float(K[k_off + d + 2]);
            float kd = __bfloat162float(K[k_off + d + 3]);
            s += qa*ka + qb*kb + qc*kc + qd*kd;
        }
        s *= scale;
        
        // Causal mask: only attend to k <= q
        if (k > q_global) {
            s = -FLT_MAX;
        }
        
        // Online softmax update
        if (s > row_max) {
            float ratio = expf(row_max - s);
            for (int di = 0; di < GROUP_D; ++di) {
                acc[di] *= ratio;
            }
            row_sum *= ratio;
            row_max = s;
            row_sum += 1.f;
            
            int64_t v_off = v_base + (int64_t)k * v_stride_s;
            for (int di = 0; di < GROUP_D; ++di) {
                acc[di] += __bfloat162float(V[v_off + d0 + di]);
            }
        } else {
            float alpha = expf(s - row_max);
            row_sum += alpha;
            
            int64_t v_off = v_base + (int64_t)k * v_stride_s;
            for (int di = 0; di < GROUP_D; ++di) {
                acc[di] += alpha * __bfloat162float(V[v_off + d0 + di]);
            }
        }
    }

    // Normalize and write output
    if (row_sum > 0.f && q_global < S) {
        float inv_sum = 1.f / row_sum;
        int64_t o_off = o_base + (int64_t)q_global * o_stride_s;
        for (int di = 0; di < GROUP_D; ++di) {
            O[o_off + d0 + di] = __float2bfloat16(acc[di] * inv_sum);
        }
        
        if (tid == 0) {
            int64_t lse_off = static_cast<int64_t>(bh) * S + q_global;
            LSE_out[lse_off] = row_max + logf(row_sum);
        }
    } else if (q_global < S) {
        int64_t o_off = o_base + (int64_t)q_global * o_stride_s;
        for (int di = 0; di < GROUP_D; ++di) {
            O[o_off + d0 + di] = __float2bfloat16(0.f);
        }
        if (tid == 0) {
            int64_t lse_off = static_cast<int64_t>(bh) * S + q_global;
            LSE_out[lse_off] = -1e10f;
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE)
{
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);

    // Extract strides from tensor view (may differ from unit stride!)
    int64_t q_stride_s = Q.stride(2);
    int64_t q_stride_d = Q.stride(3);
    int64_t k_stride_s = K.stride(2);
    int64_t k_stride_d = K.stride(3);
    int64_t v_stride_s = V.stride(2);
    int64_t v_stride_d = V.stride(3);
    int64_t o_stride_s = O.stride(2);
    int64_t o_stride_d = O.stride(3);

    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());

    int64_t num_bh = B * H;
    int64_t num_m_tiles = (S + BLOCK_M - 1) / BLOCK_M;
    
    dim3 grid(num_bh, num_m_tiles, 1);
    dim3 block(BLOCK_M, 1, 1);

    float scale = rsqrtf(static_cast<float>(D));

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    mha_causal_kernel<<<grid, block, 0, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data,
        static_cast<int>(B), static_cast<int>(H), static_cast<int>(S), static_cast<int>(D),
        q_stride_s, q_stride_d,
        k_stride_s, k_stride_d,
        v_stride_s, v_stride_d,
        o_stride_s, o_stride_d,
        scale);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_causal_d128::run);

}  // namespace mha_causal_d128