#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
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

using namespace nvcuda;

namespace mha_lse {

constexpr int Br = 128;
constexpr int Bc = 64;
constexpr int D = 128;
constexpr int THREADS = 128;
constexpr int Bc_pad = 64;
constexpr int D_pad = 128;

constexpr int SMEM_SIZE = 
    Br * D * 2 +         // smem_q: 32 KB
    2 * Bc * D * 2 +     // smem_kv double buffer: 32 KB
    Br * Bc_pad * 4 +    // smem_s: 32 KB
    Br * Bc_pad * 2 +    // smem_p: 16 KB
    Br * D_pad * 4;      // smem_o: 64 KB
                         // Total: 176 KB

__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

template<int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__global__ __launch_bounds__(THREADS, 1)
void mha_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S, float scale) {

    int tid = threadIdx.x;
    int warp_id = tid / 32;

    int n_q_blocks = (S + Br - 1) / Br;
    int grid_idx = blockIdx.x;
    int batch = grid_idx / (H * n_q_blocks);
    int rest = grid_idx % (H * n_q_blocks);
    int head = rest / n_q_blocks;
    int q_block = rest % n_q_blocks;
    int q_start = q_block * Br;

    extern __shared__ char smem[];
    char* ptr = smem;
    
    __nv_bfloat16* smem_q = reinterpret_cast<__nv_bfloat16*>(ptr); ptr += Br * D * 2;
    __nv_bfloat16* smem_kv0 = reinterpret_cast<__nv_bfloat16*>(ptr);
    __nv_bfloat16* smem_kv1 = smem_kv0 + Bc * D; ptr += 2 * Bc * D * 2;
    float* smem_s = reinterpret_cast<float*>(ptr); ptr += Br * Bc_pad * 4;
    __nv_bfloat16* smem_p = reinterpret_cast<__nv_bfloat16*>(ptr); ptr += Br * Bc_pad * 2;
    float* smem_o = reinterpret_cast<float*>(ptr);

    // Load Q
    const __nv_bfloat16* Q_gptr = Q + ((batch * H + head) * S + q_start) * D;
    for (int i = tid; i < Br * D / 8; i += THREADS) {
        int row = (i * 8) / D;
        if (q_start + row < S) cp_async_16(smem_q + i * 8, Q_gptr + i * 8);
        else *reinterpret_cast<int4*>(smem_q + i * 8) = make_int4(0, 0, 0, 0);
    }
    cp_async_commit();
    cp_async_wait_group<0>();
    __syncthreads();

    // Init smem_o
    for (int i = tid; i < Br * D_pad; i += THREADS) smem_o[i] = 0.0f;
    __syncthreads();

    // Prefetch K[0]
    {
        const __nv_bfloat16* K_gptr = K + ((batch * H + head) * S) * D;
        for (int i = tid; i < Bc * D / 8; i += THREADS) {
            int row = (i * 8) / D;
            if (row < S) cp_async_16(smem_kv0 + i * 8, K_gptr + i * 8);
            else *reinterpret_cast<int4*>(smem_kv0 + i * 8) = make_int4(0, 0, 0, 0);
        }
        cp_async_commit();
    }

    float m = -INFINITY;
    float l = 0.0f;

    for (int kv = 0; kv < S; kv += Bc) {
        int buf = (kv / Bc) % 2;
        __nv_bfloat16* smem_kv = (buf == 0) ? smem_kv0 : smem_kv1;
        __nv_bfloat16* smem_kv_next = (buf == 0) ? smem_kv1 : smem_kv0;

        // Wait for K
        cp_async_wait_group<0>();
        __syncthreads();

        // QK^T: S = Q @ K^T * scale
        // Each warp handles 2 m-tiles (32 rows), 4 n-tiles (64 cols)
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            int m_row = warp_id * 2 + mi;
            #pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
                wmma::fill_fragment(c_frag, 0.0f);
                #pragma unroll
                for (int k = 0; k < 8; k++) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
                    wmma::load_matrix_sync(a_frag, smem_q + m_row * 16 * D + k * 16, D);
                    wmma::load_matrix_sync(b_frag, smem_kv + ni * 16 * D + k * 16, D);
                    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                }
                #pragma unroll
                for (int i = 0; i < c_frag.num_elements; i++) c_frag.x[i] *= scale;
                wmma::store_matrix_sync(smem_s + m_row * 16 * Bc_pad + ni * 16, c_frag, Bc_pad, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // Issue V load (into same buffer, overwriting K)
        {
            const __nv_bfloat16* V_gptr = V + ((batch * H + head) * S + kv) * D;
            for (int i = tid; i < Bc * D / 8; i += THREADS) {
                int row = (i * 8) / D;
                if (kv + row < S) cp_async_16(smem_kv + i * 8, V_gptr + i * 8);
                else *reinterpret_cast<int4*>(smem_kv + i * 8) = make_int4(0, 0, 0, 0);
            }
            cp_async_commit();
        }

        // Online softmax (each thread = one row)
        float scale_o = 1.0f;
        if (tid < Br) {
            float* s_row = smem_s + tid * Bc_pad;
            __nv_bfloat16* p_row = smem_p + tid * Bc_pad;
            bool valid = (q_start + tid < S);
            
            float m_block = -INFINITY;
            #pragma unroll
            for (int j = 0; j < Bc; j++) {
                if (kv + j < S) m_block = fmaxf(m_block, s_row[j]);
            }
            float m_new = valid ? fmaxf(m, m_block) : m;
            scale_o = valid ? __expf(m - m_new) : 1.0f;
            
            float l_block = 0.0f;
            #pragma unroll
            for (int j = 0; j < Bc; j++) {
                float p = (kv + j < S) ? __expf(s_row[j] - m_new) : 0.0f;
                p_row[j] = __float2bfloat16(p);
                l_block += p;
            }
            l = valid ? (l * scale_o + l_block) : l;
            m = valid ? m_new : m;
        }
        __syncthreads();

        // Rescale O cooperatively: each thread handles one column across all rows
        for (int row = 0; row < Br; row++) {
            if (q_start + row < S) {
                smem_o[row * D_pad + tid] *= scale_o;
            }
        }
        __syncthreads();

        // Prefetch next K (into other buffer)
        if (kv + Bc < S) {
            const __nv_bfloat16* K_gptr = K + ((batch * H + head) * S + kv + Bc) * D;
            for (int i = tid; i < Bc * D / 8; i += THREADS) {
                int row = (i * 8) / D;
                if (kv + Bc + row < S) cp_async_16(smem_kv_next + i * 8, K_gptr + i * 8);
                else *reinterpret_cast<int4*>(smem_kv_next + i * 8) = make_int4(0, 0, 0, 0);
            }
            cp_async_commit();
        }

        // Wait for V (allow next K to be pending)
        if (kv + Bc < S) cp_async_wait_group<1>();
        else cp_async_wait_group<0>();
        __syncthreads();

        // PV: O += P @ V using wmma load/store for O
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            int m_row = warp_id * 2 + mi;
            #pragma unroll
            for (int ni = 0; ni < 8; ni++) {
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_frag;
                wmma::load_matrix_sync(o_frag, smem_o + m_row * 16 * D_pad + ni * 16, D_pad, wmma::mem_row_major);
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_frag;
                    wmma::load_matrix_sync(a_frag, smem_p + m_row * 16 * Bc_pad + k * 16, Bc_pad);
                    wmma::load_matrix_sync(b_frag, smem_kv + k * 16 * D + ni * 16, D);
                    wmma::mma_sync(o_frag, a_frag, b_frag, o_frag);
                }
                wmma::store_matrix_sync(smem_o + m_row * 16 * D_pad + ni * 16, o_frag, D_pad, wmma::mem_row_major);
            }
        }
        __syncthreads();
    }

    // Final output: normalize and store
    if (tid < Br && q_start + tid < S) {
        float inv_l = 1.0f / l;
        float* o_row = smem_o + tid * D_pad;
        __nv_bfloat16* O_gptr = O + ((batch * H + head) * S + q_start + tid) * D;
        #pragma unroll
        for (int d = 0; d < D; d += 8) {
            float4 o0 = *reinterpret_cast<float4*>(o_row + d);
            float4 o1 = *reinterpret_cast<float4*>(o_row + d + 4);
            __nv_bfloat162 p01 = __float22bfloat162_rn(make_float2(o0.x * inv_l, o0.y * inv_l));
            __nv_bfloat162 p23 = __float22bfloat162_rn(make_float2(o0.z * inv_l, o0.w * inv_l));
            __nv_bfloat162 p45 = __float22bfloat162_rn(make_float2(o1.x * inv_l, o1.y * inv_l));
            __nv_bfloat162 p67 = __float22bfloat162_rn(make_float2(o1.z * inv_l, o1.w * inv_l));
            float4 out;
            *reinterpret_cast<__nv_bfloat162*>(&out.x) = p01;
            *reinterpret_cast<__nv_bfloat162*>(&out.y) = p23;
            *reinterpret_cast<__nv_bfloat162*>(&out.z) = p45;
            *reinterpret_cast<__nv_bfloat162*>(&out.w) = p67;
            *reinterpret_cast<float4*>(O_gptr + d) = out;
        }
        LSE[(batch * H + head) * S + q_start + tid] = m + __logf(l);
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
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    CUDA_CHECK(cudaFuncSetAttribute(mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));
    mha_kernel<<<grid, block, SMEM_SIZE, stream>>>(
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