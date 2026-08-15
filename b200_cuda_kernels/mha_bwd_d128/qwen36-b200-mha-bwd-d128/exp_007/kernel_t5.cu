#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <stdio.h>
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

namespace mha_bwd_impl {

// Tiled GEMM: C_out[M,N] = alpha * A[M,K] * B[K,N]^T + beta*C_out
// A layout: row-major [bh, M, K], B layout: row-major [bh, N, K] (so B(K)^T means we read along K as reduction)
template<int TM, int TN, int BK, bool k_major_a, bool k_major_b>
__global__ void gemm_kernel(const __nv_bfloat16* __restrict__ A,
                            const __nv_bfloat16* __restrict__ B,
                            float* __restrict__ C_out,
                            int M, int N, int K, float alpha, float beta,
                            uint64_t bh_offset, int n_bh) {
    extern __shared__ __align__(128) unsigned char smem[];
    __nv_bfloat16* sA = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sB = sA + TM * BK + 8; // padding for bank conflicts
    
    int bx = blockIdx.x;
    if (bx >= n_bh) return;
    
    uint64_t offA = static_cast<uint64_t>(bx) * M * K + threadIdx.y * blockDim.z * M + threadIdx.x;
    uint64_t offB = static_cast<uint64_t>(bx) * N * K + threadIdx.y * blockDim.z * N + threadIdx.x;
    uint64_t offC = static_cast<uint64_t>(bx) * M * N + threadIdx.y * blockDim.z * N + threadIdx.x;
    
    float acc[TN];
    #pragma unroll
    for (int j = 0; j < TN; j++) acc[j] = 0.0f;
    
    for (int bk = 0; bk < K; bk += BK) {
        // Load tile of A into shared memory cooperatively
        #pragma unroll
        for (int i = 0; i < TM; i++) {
            for (int j = 0; j < BK; j++) {
                int tid_load = i * BK + j;
                if (tid_load == threadIdx.x && ((threadIdx.y * blockDim.z + threadIdx.x) / K * K + bk + j) < K && ((threadIdx.y * blockDim.z + threadIdx.x) / M * M + threadIdx.y) < M) {
                    // Complex indexing... let me simplify
                }
            }
        }
        
        // Simple approach: just have each thread load its own A/B values directly
        // Cooperative loading requires careful index management
        
        // Direct computation without SMEM is simpler but uses more GMEM bandwidth
        float local_a[BK];
        float local_b[TN][BK];
        
        #pragma unroll
        for (int ii = 0; ii < TM; ii++) {
            int m_idx = blockIdx.y * TM + threadIdx.y * TM + ii;
            if (m_idx < M) {
                const __nv_bfloat16* a_row = A + offA + ii * K + bk;
                
                #pragma unroll
                for (int kk = 0; kk < BK; kk++) {
                    local_a[kk] = __bfloat162float(a_row[kk]);
                }
                
                #pragma unroll
                for (int jj = 0; jj < TN; jj++) {
                    int n_idx = blockIdx.z * TN + threadIdx.z * TN + jj;
                    if (n_idx < N) {
                        const __nv_bfloat16* b_row = B + offB + jj * K + bk;
                        
                        #pragma unroll
                        for (int kk = 0; kk < BK; kk++) {
                            local_b[jj][kk] = __bfloat162float(b_row[kk]);
                        }
                        
                        #pragma unroll
                        for (int kk = 0; kk < BK; kk++) {
                            float c_val = acc[jj] + alpha * local_a[kk] * local_b[jj][kk];
                            if (beta != 0.0f) {
                                c_val += beta * C_out[m_idx * N + n_idx];
                            }
                            acc[jj] = c_val;
                        }
                        
                        if (bk + BK >= K || true) {
                            C_out[m_idx * N + n_idx] = acc[jj] * alpha;
                        }
                    }
                }
            }
        }
    }
    
    // Final write
    #pragma unroll
    for (int ii = 0; ii < TM; ii++) {
        #pragma unroll
        for (int jj = 0; jj < TN; jj++) {
            int m_idx = blockIdx.y * TM + threadIdx.y * TM + ii;
            int n_idx = blockIdx.z * TN + threadIdx.z * TN + jj;
            if (m_idx < M && n_idx < N) {
                float c = acc[jj];
                if (beta != 0.0f) {
                    c += beta * C_out[m_idx * N + n_idx];
                }
                C_out[m_idx * N + n_idx] = c;
            }
        }
    }
}

} // namespace mha_bwd_impl

extern "C" {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = Q.size(3);
    
    float scale = 1.0f / std::sqrt(static_cast<float>(d));
    
    const __nv_bfloat16* Q_ptr  = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr  = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr  = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float*         L_ptr  = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_ptr      = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr      = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr      = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    int64_t n_bh = B * H;
    int threads_per_block = 128; // Warp group size
    
    // Workspace: S*S floats per (b,h) for P_mat and dS_mat
    int64_t elems_per_bh = 2 * S * S;
    size_t max_bytes = 12ULL * 1024 * 1024 * 1024;
    int64_t max_concurrent = max_bytes / (elems_per_bh * sizeof(float));
    if (max_concurrent > n_bh) max_concurrent = n_bh;
    if (max_concurrent < 1) max_concurrent = 1;
    
    int64_t tmp_elems = max_concurrent * elems_per_bh;
    float* tmp_buf;
    CUDA_CHECK(cudaMalloc(&tmp_buf, tmp_elems * sizeof(float)));
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    // Launch one simple kernel per (b,h)
    for (int64_t round_start = 0; round_start < n_bh; round_start += max_concurrent) {
        int64_t this_n = min(n_bh - round_start, max_concurrent);
        
        // Grid: 1D over (b,h) pairs, 128 threads each
        int blocks = static_cast<int>(this_n);
        
        mha_bwd_impl::simple_mha_bwd<<<blocks, threads_per_block, 0, stream>>>(
            Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr,
            dQ_ptr, dK_ptr, dV_ptr,
            tmp_buf + round_start * elems_per_bh,
            B, H, S, d, scale
        );
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    
    CUDA_CHECK(cudaFree(tmp_buf));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // extern "C"