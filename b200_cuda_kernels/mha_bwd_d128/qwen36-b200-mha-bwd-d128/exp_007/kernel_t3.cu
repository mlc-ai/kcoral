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

// ===== Utility: BF16 dot product =====
__device__ __forceinline__ float bf16_dot(const __nv_bfloat16* a, const __nv_bfloat16* b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i += 4) {
        s += __bfloat162float(a[i+0]) * __bfloat162float(b[i+0]);
        s += __bfloat162float(a[i+1]) * __bfloat162float(b[i+1]);
        s += __bfloat162float(a[i+2]) * __bfloat162float(b[i+2]);
        s += __bfloat162float(a[i+3]) * __bfloat162float(b[i+3]);
    }
    return s;
}

// ===== General tiled GEMM: C[M,N] = alpha * A[M,K] * B[K,N] + beta*C =====
template<int TM, int TN, int TK>
__global__ void tile_gemm(float** C_out, int n_blocks,
    const __nv_bfloat16* __restrict__ A,  // [..., M, K], strided by bh_offset
    const __nv_bfloat16* __restrict__ B,  // [..., K, N] (transposed B layout: first dim is inner/reduction)
    int M, int N, int K, float alpha, float beta,
    int bh_start, int bh_stride,
    int base_idx, int elem_A, int elem_B, int elem_C,
    bool transpose_b) 
{
    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* As = smem;
    __nv_bfloat16* Bs = As + TM * TK + 8; // +8 for bank conflict avoidance
    
    int tid = threadIdx.x;
    int tx = tid % TM;
    int ty = tid / TM;
    
    float acc[TN];
    #pragma unroll
    for (int j = 0; j < TN; j++) acc[j] = 0.0f;
    
    // Iterate over K in tiles
    for (int kk = 0; kk < K; kk += TK) {
        // Load tile of A into shared memory: As[ty_sub][k_sub]
        for (int ki = 0; ki < TK; ki++) {
            int k_idx = kk + ki;
            int a_row = ty * TN + ty; // each thread owns one A row for its output TN window
            
            // Better: cooperatively load As[row][col]
            for (int ri = ti; ri < TM * TK; ri += blockDim.x) {
                int r = ri / TK;
                int c = ri % TK;
                int global_k = kk + c;
                As[ri] = A[base_idx + (uint64_t)(elem_A) * (r) + global_k];
            }
            __syncthreads();
            
            for (int ki = 0; ki < TK; ki++) {
                int k_local = ki;
                for (int ni = 0; ni < TN; ni++) {
                    int n_idx = tn * TN + ni;
                    float b_val = __bfloat162float(B[base_idx + (uint64_t)(elem_B) * n_idx + global_k]);
                    acc[ni] += __bfloat162float(As[r * TK + k_local]) * b_val;
                }
            }
            __syncthreads();
        }
        
        // Write result
        for (int ni = 0; ni < TN; ni++) {
            int out_n = bn * TN + ni;
            float val = acc[ni] * alpha;
            if (beta == 0.0f) {
                C_out[tid / TN][tid % TN] = val;
            } else {
                float c_old = C_out[tid / TN][out_n];
                C_out[out_m][out_n] = val + beta * c_old;
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
    int64_t elems_pm = S * S;
    
    // Limit workspace to ~14GB max
    size_t max_tmp = 14ULL * 1024 * 1024 * 1024;
    int64_t elems_per_bh = 2 * elems_pm; // P_mat + dS_mat
    int64_t max_bh = max_tmp / (elems_per_bh * sizeof(float));
    if (max_bh > n_bh) max_bh = n_bh;
    
    int n_rounds = (int)((n_bh + max_bh - 1) / max_bh);
    int64_t tmp_elems = max_bh * elems_per_bh;
    
    float* tmp_buf;
    CUDA_CHECK(cudaMalloc(&tmp_buf, tmp_elems * sizeof(float)));
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    // Process in rounds
    for (int round = 0; round < n_rounds; round++) {
        int64_t bh_start = round * max_bh;
        int64_t bh_end = min(bh_start + max_bh, n_bh);
        int n_bh_this = (int)(bh_end - bh_start);
        
        // Clear workspace
        CUDA_CHECK(cudaMemsetAsync(tmp_buf, 0, tmp_elems * sizeof(float), stream));
        
        // TODO: Launch the actual kernel
        
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    
    CUDA_CHECK(cudaFree(tmp_buf));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // extern "C"