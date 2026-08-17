#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>
#include <algorithm>
#include <vector>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_fa4 {

// Load a 128x128 tile of data from global memory to shared memory using vectorized loads.
__device__ void load_to_smem(char* smem, const __nv_bfloat16* gmem) {
    int tid = threadIdx.x;
    for (int vec = 0; vec < 16; vec++) {
        *(uint4*)(&smem[tid*128 + vec*8]) = *(uint4*)(&gmem[tid*128 + vec*8]);
    }
}

// Compute generic transposed GEMM where threads calculate rows of output matrix directly in shared memory.
__device__ void gemm_transposed(char* s_A, char* s_B, char* smem_out) {
    int tid = threadIdx.x;
    for (int i = 0; i < 128; i++) {
        float sum = 0;
        for (int k = 0; k < 128; k++) {
            __nv_bfloat16 a_val = *( (__nv_bfloat16*) (&s_A[tid*128 + k]) );
            __nv_bfloat16 b_val = *( (__nv_bfloat16*) (&s_B[i*128 + k]) );
            sum += __bfloat162float(a_val) * __bfloat162float(b_val);
        }
        *( (__nv_bfloat16*) (&smem_out[tid*128 + i]) ) = __float2bfloat16(sum);
    }
}

__device__ void apply_softmax(char* s_PT, const float* L_bh, int q_start, int kv_start, float scale) {
    int tid = threadIdx.x;
    for (int i = 0; i < 128; i++) {
        float s = __bfloat162float(*( (__nv_bfloat16*) (&s_PT[tid*128 + i]) )) * scale;
        float p = 0;
        if (kv_start + tid <= q_start + i && q_start + i < S) {
            p = expf(s - L_bh[q_start + i]);
        }
        *( (__nv_bfloat16*) (&s_PT[tid*128 + i]) ) = __float2bfloat16(p);
    }
}

// Element-wise phase transitioning values securely into s_PT space (reusing it completely).
__device__ void compute_dS(char* s_PT, char* s_dPT, float* s_DT, char* s_dST) {
    int tid = threadIdx.x;
    for (int i = 0; i < 128; i++) {
        float p = __bfloat162float(*( (__nv_bfloat16*) (&s_PT[tid*128 + i]) ));
        float dp = __bfloat162float(*( (__nv_bfloat16*) (&s_dPT[tid*128 + i]) ));
        float ds = p * (dp - s_DT[i]);
        *( (__nv_bfloat16*) (&s_dST[tid*128 + i]) ) = __float2bfloat16(ds);
    }
}

__device__ void compute_D(char* s_O, char* s_dO, float* s_DT) {
    int tid = threadIdx.x;
    float sum = 0;
    for (int k = 0; k < 128; k++) {
        sum += __bfloat162float(*( (__nv_bfloat16*) (&s_O[tid*128 + k]) )) * __bfloat162float(*( (__nv_bfloat16*) (&s_dO[tid*128 + k]) ));
    }
    s_DT[tid] = sum;
}

// Accumulates exclusively via local register state before leveraging atomic adds globally.
__device__ void gemm_atomic_add(char* s_A, char* s_B, __nv_bfloat16* global_D, int64_t row_base) {
    int tid = threadIdx.x;
    for (int i = 0; i < 128; i++) {
        float sum = 0;
        for (int k = 0; k < 128; k++) {
            __nv_bfloat16 a_val = *( (__nv_bfloat16*) (&s_A[tid*128 + k]) );
            __nv_bfloat16 b_val = *( (__nv_bfloat16*) (&s_B[i*128 + k]) );
            sum += __bfloat162float(a_val) * __bfloat162float(b_val);
        }
        if (row_base + tid < S) {
            atomicAdd(&global_D[(row_base + tid)*128 + i], __float2bfloat16(sum));
        }
    }
}

// Solves inner-product dynamics leveraging distinct shared-memory tracking structures.
__device__ void gemm_dQ_atomic_add(char* s_A_T, char* s_B, __nv_bfloat16* global_D, int64_t row_base) {
    int tid = threadIdx.x;
    for (int k = 0; k < 128; k++) {
        float sum = 0;
        for (int j = 0; j < 128; j++) {
            __nv_bfloat16 a_val = *( (__nv_bfloat16*) (&s_A_T[j*128 + tid]) );
            __nv_bfloat16 b_val = *( (__nv_bfloat16*) (&s_B[j*128 + k]) );
            sum += __bfloat162float(a_val) * __bfloat162float(b_val);
        }
        if (row_base + tid < S) {
            atomicAdd(&global_D[(row_base + tid)*128 + k], __float2bfloat16(sum));
        }
    }
}

struct SharedStorage {
    char s_Q[128*128*2];
    char s_K[128*128*2];
    char s_V[128*128*2];
    char s_dO[128*128*2];
    char s_PT[128*128*2];
    char s_dPT[128*128*2];
    char s_O[128*128*2];
    char s_dST[128*128*2];
    float s_DT[128];
};

__global__ void bwd_kernel(
    const __nv_bfloat16* __restrict__ Q_bh,
    const __nv_bfloat16* __restrict__ K_bh,
    const __nv_bfloat16* __restrict__ V_bh,
    const __nv_bfloat16* __restrict__ O_bh,
    const __nv_bfloat16* __restrict__ dO_bh,
    const float* __restrict__ L_bh,
    __nv_bfloat16* __restrict__ dQ_bh,
    __nv_bfloat16* __restrict__ dK_bh,
    __nv_bfloat16* __restrict__ dV_bh,
    int64_t S, float scale)
{
    int bh = blockIdx.y;
    int kv_start = blockIdx.x * 128;
    
    extern __shared__ char smem_buf[];
    SharedStorage& smem = *(SharedStorage*)smem_buf;
    
    load_to_smem(smem.s_K, K_bh + kv_start * 128);
    load_to_smem(smem.s_V, V_bh + kv_start * 128);
    __syncthreads();

    for (int q_start = 0; q_start <= kv_start && q_start < S; q_start += 128) {
        load_to_smem(smem.s_Q, Q_bh + q_start * 128);
        load_to_smem(smem.s_dO, dO_bh + q_start * 128);
        load_to_smem(smem.s_O, O_bh + q_start * 128);
        __syncthreads();

        compute_D(smem.s_O, smem.s_dO, smem.s_DT);
        __syncthreads();

        gemm_transposed(smem.s_K, smem.s_Q, smem.s_PT); 
        __syncthreads();

        apply_softmax(smem.s_PT, L_bh, q_start, kv_start, scale);
        __syncthreads();

        gemm_transposed(smem.s_dO, smem.s_V, smem.s_dPT);
        __syncthreads();

        compute_dS(smem.s_PT, smem.s_dPT, smem.s_DT, smem.s_dST);
        __syncthreads();

        gemm_atomic_add(smem.s_PT, smem.s_dO, dV_bh + kv_start * 128, 0);

        gemm_atomic_add(smem.s_dST, smem.s_Q, dK_bh + kv_start * 128, 0);
        
        __syncthreads();
    }

    for (int q_start = kv_start; q_start < S; q_start += 128) {
        load_to_smem(smem.s_Q, Q_bh + q_start * 128);
        load_to_smem(smem.s_dO, dO_bh + q_start * 128);
        load_to_smem(smem.s_O, O_bh + q_start * 128);
        __syncthreads();

        compute_D(smem.s_O, smem.s_dO, smem.s_DT);
        __syncthreads();

        gemm_transposed(smem.s_K, smem.s_Q, smem.s_PT);
        __syncthreads();

        apply_softmax(smem.s_PT, L_bh, q_start, kv_start, scale);
        __syncthreads();

        gemm_transposed(smem.s_dO, smem.s_V, smem.s_dPT);
        __syncthreads();

        compute_dS(smem.s_PT, smem.s_dPT, smem.s_DT, smem.s_dST);
        __syncthreads();

        gemm_dQ_atomic_add(smem.s_dST, smem.s_K, dQ_bh + q_start * 128, 0);
        
        __syncthreads();
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L, 
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
         
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S_val = Q.size(2);
    int64_t d = Q.size(3); 
    
    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_data = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_data = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_data = static_cast<const float*>(L.data_ptr());
    
    __nv_bfloat16* dQ_data = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_data = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_data = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    float scale = 1.0f / sqrtf((float)d);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    // Zero-fill output arrays to guarantee appropriate atomic additions tracking properly across disparate block jurisdictions
    CUDA_CHECK(cudaMemsetAsync(dQ_data, 0, B * H * S_val * d * sizeof(__nv_bfloat16), stream));
    CUDA_CHECK(cudaMemsetAsync(dK_data, 0, B * H * S_val * d * sizeof(__nv_bfloat16), stream));
    CUDA_CHECK(cudaMemsetAsync(dV_data, 0, B * H * S_val * d * sizeof(__nv_bfloat16), stream));
    
    int64_t threads = 128;
    dim3 grid((S_val + 127) / 128, B * H);
    
    CUDA_CHECK(cudaFuncSetAttribute(bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedStorage)));
    
    bwd_kernel<<<grid, threads, sizeof(SharedStorage), stream>>>(
        Q_data, K_data, V_data, O_data, dO_data, L_data, dQ_data, dK_data, dV_data, S_val, scale);
        
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_fa4::run);

} // namespace tvm_ffi_fa4