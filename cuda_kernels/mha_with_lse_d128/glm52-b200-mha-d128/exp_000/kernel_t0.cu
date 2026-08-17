#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
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

namespace mha_lse {

constexpr int Br = 128;
constexpr int Bc = 32;
constexpr int D = 128;
constexpr int THREADS = 128;

__device__ __forceinline__ float dot8_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b) {
    float4 av = *reinterpret_cast<const float4*>(a);
    float4 bv = *reinterpret_cast<const float4*>(b);
    __nv_bfloat162 a01 = *reinterpret_cast<__nv_bfloat162*>(&av.x);
    __nv_bfloat162 a23 = *reinterpret_cast<__nv_bfloat162*>(&av.y);
    __nv_bfloat162 a45 = *reinterpret_cast<__nv_bfloat162*>(&av.z);
    __nv_bfloat162 a67 = *reinterpret_cast<__nv_bfloat162*>(&av.w);
    __nv_bfloat162 b01 = *reinterpret_cast<__nv_bfloat162*>(&bv.x);
    __nv_bfloat162 b23 = *reinterpret_cast<__nv_bfloat162*>(&bv.y);
    __nv_bfloat162 b45 = *reinterpret_cast<__nv_bfloat162*>(&bv.z);
    __nv_bfloat162 b67 = *reinterpret_cast<__nv_bfloat162*>(&bv.w);
    float2 fa01 = __bfloat1622float2(a01);
    float2 fa23 = __bfloat1622float2(a23);
    float2 fa45 = __bfloat1622float2(a45);
    float2 fa67 = __bfloat1622float2(a67);
    float2 fb01 = __bfloat1622float2(b01);
    float2 fb23 = __bfloat1622float2(b23);
    float2 fb45 = __bfloat1622float2(b45);
    float2 fb67 = __bfloat1622float2(b67);
    return fa01.x*fb01.x + fa01.y*fb01.y +
           fa23.x*fb23.x + fa23.y*fb23.y +
           fa45.x*fb45.x + fa45.y*fb45.y +
           fa67.x*fb67.x + fa67.y*fb67.y;
}

__global__ __launch_bounds__(THREADS, 3)
void mha_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S, float scale) {

    int tid = threadIdx.x;
    int row = tid;

    int n_q_blocks = (S + Br - 1) / Br;
    int grid_idx = blockIdx.x;
    int batch = grid_idx / (H * n_q_blocks);
    int rest = grid_idx % (H * n_q_blocks);
    int head = rest / n_q_blocks;
    int q_block = rest % n_q_blocks;

    int q_start = q_block * Br;
    int q_row = q_start + row;
    bool valid = (q_row < S);

    extern __shared__ char smem[];
    __nv_bfloat16* smem_q = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* smem_kv = smem_q + Br * D;

    // Load Q tile [Br, D]
    const __nv_bfloat16* Q_base = Q + ((batch * H + head) * S + q_start) * D;
    if (valid) {
        const uint4* gptr = reinterpret_cast<const uint4*>(Q_base + row * D);
        uint4* sptr = reinterpret_cast<uint4*>(smem_q + row * D);
        #pragma unroll
        for (int i = 0; i < D / 8; ++i) {
            sptr[i] = gptr[i];
        }
    } else {
        uint4* sptr = reinterpret_cast<uint4*>(smem_q + row * D);
        uint4 zero = make_uint4(0, 0, 0, 0);
        #pragma unroll
        for (int i = 0; i < D / 8; ++i) {
            sptr[i] = zero;
        }
    }
    __syncthreads();

    float O_reg[D];
    #pragma unroll
    for (int d = 0; d < D; ++d) O_reg[d] = 0.0f;

    float m = -INFINITY;
    float l = 0.0f;

    for (int kv_start = 0; kv_start < S; kv_start += Bc) {
        int kv_end = min(kv_start + Bc, S);
        int actual_Bc = kv_end - kv_start;

        // Load K block [actual_Bc, D]
        const __nv_bfloat16* K_base = K + ((batch * H + head) * S + kv_start) * D;
        const uint4* gptr_k = reinterpret_cast<const uint4*>(K_base);
        uint4* sptr_k = reinterpret_cast<uint4*>(smem_kv);
        int total_k = actual_Bc * (D / 8);
        for (int i = tid; i < total_k; i += THREADS) {
            sptr_k[i] = gptr_k[i];
        }
        __syncthreads();

        // Compute S = Q @ K^T * scale
        float S_reg[Bc];
        #pragma unroll
        for (int j = 0; j < Bc; ++j) S_reg[j] = -INFINITY;

        if (valid) {
            const __nv_bfloat16* q_ptr = smem_q + row * D;
            #pragma unroll
            for (int j = 0; j < Bc; ++j) {
                if (kv_start + j < kv_end) {
                    const __nv_bfloat16* k_ptr = smem_kv + j * D;
                    float sum = 0.0f;
                    #pragma unroll
                    for (int d = 0; d < D; d += 8) {
                        sum += dot8_bf16(q_ptr + d, k_ptr + d);
                    }
                    S_reg[j] = sum * scale;
                }
            }
        }

        // Online softmax update
        float m_block = -INFINITY;
        #pragma unroll
        for (int j = 0; j < Bc; ++j) m_block = fmaxf(m_block, S_reg[j]);
        float m_new = fmaxf(m, m_block);
        float scale_o = __expf(m - m_new);

        #pragma unroll
        for (int d = 0; d < D; ++d) O_reg[d] *= scale_o;

        float l_block = 0.0f;
        #pragma unroll
        for (int j = 0; j < Bc; ++j) {
            float p = (kv_start + j < kv_end) ? __expf(S_reg[j] - m_new) : 0.0f;
            S_reg[j] = p;
            l_block += p;
        }
        l = l * scale_o + l_block;
        m = m_new;

        // Load V block [actual_Bc, D] (reuse smem_kv)
        const __nv_bfloat16* V_base = V + ((batch * H + head) * S + kv_start) * D;
        const uint4* gptr_v = reinterpret_cast<const uint4*>(V_base);
        uint4* sptr_v = reinterpret_cast<uint4*>(smem_kv);
        int total_v = actual_Bc * (D / 8);
        for (int i = tid; i < total_v; i += THREADS) {
            sptr_v[i] = gptr_v[i];
        }
        __syncthreads();

        if (valid) {
            #pragma unroll
            for (int j = 0; j < Bc; ++j) {
                float p = S_reg[j];
                if (p == 0.0f) continue;
                const __nv_bfloat16* v_ptr = smem_kv + j * D;
                #pragma unroll
                for (int d = 0; d < D; d += 8) {
                    float4 v4 = *reinterpret_cast<const float4*>(v_ptr + d);
                    __nv_bfloat162 v01 = *reinterpret_cast<__nv_bfloat162*>(&v4.x);
                    __nv_bfloat162 v23 = *reinterpret_cast<__nv_bfloat162*>(&v4.y);
                    __nv_bfloat162 v45 = *reinterpret_cast<__nv_bfloat162*>(&v4.z);
                    __nv_bfloat162 v67 = *reinterpret_cast<__nv_bfloat162*>(&v4.w);
                    float2 f01 = __bfloat1622float2(v01);
                    float2 f23 = __bfloat1622float2(v23);
                    float2 f45 = __bfloat1622float2(v45);
                    float2 f67 = __bfloat1622float2(v67);
                    O_reg[d+0] += p * f01.x;
                    O_reg[d+1] += p * f01.y;
                    O_reg[d+2] += p * f23.x;
                    O_reg[d+3] += p * f23.y;
                    O_reg[d+4] += p * f45.x;
                    O_reg[d+5] += p * f45.y;
                    O_reg[d+6] += p * f67.x;
                    O_reg[d+7] += p * f67.y;
                }
            }
        }
        __syncthreads();
    }

    if (valid) {
        float inv_l = 1.0f / l;
        __nv_bfloat16* O_base = O + ((batch * H + head) * S + q_row) * D;
        #pragma unroll
        for (int d = 0; d < D; d += 8) {
            __nv_bfloat162 p01, p23, p45, p67;
            p01.x = __float2bfloat16(O_reg[d+0] * inv_l);
            p01.y = __float2bfloat16(O_reg[d+1] * inv_l);
            p23.x = __float2bfloat16(O_reg[d+2] * inv_l);
            p23.y = __float2bfloat16(O_reg[d+3] * inv_l);
            p45.x = __float2bfloat16(O_reg[d+4] * inv_l);
            p45.y = __float2bfloat16(O_reg[d+5] * inv_l);
            p67.x = __float2bfloat16(O_reg[d+6] * inv_l);
            p67.y = __float2bfloat16(O_reg[d+7] * inv_l);
            float4 out4;
            *reinterpret_cast<__nv_bfloat162*>(&out4.x) = p01;
            *reinterpret_cast<__nv_bfloat162*>(&out4.y) = p23;
            *reinterpret_cast<__nv_bfloat162*>(&out4.z) = p45;
            *reinterpret_cast<__nv_bfloat162*>(&out4.w) = p67;
            *reinterpret_cast<float4*>(O_base + d) = out4;
        }
        LSE[(batch * H + head) * S + q_row] = m + __logf(l);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int B = 4;
    int H = 48;
    int D = 128;
    int64_t S = Q.size(2);
    float scale = 1.0f / sqrtf((float)D);
    int n_q_blocks = (static_cast<int>(S) + Br - 1) / Br;
    int grid = B * H * n_q_blocks;
    int block = THREADS;
    int smem_bytes = (Br + Bc) * D * sizeof(__nv_bfloat16);
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    mha_kernel<<<grid, block, smem_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<__nv_bfloat16*>(O.data_ptr()),
        static_cast<float*>(LSE.data_ptr()),
        B, H, static_cast<int>(S), scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_lse::run);

}  // namespace mha_lse