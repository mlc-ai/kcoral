#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
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

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void fence_mbarrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase));
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ __nv_bfloat16 read_swizzled(const char* smem, int row, int col) {
    int col_bytes = col * 2;
    int chunk = col_bytes / 16;
    int chunk_swizzled = chunk ^ (row % 8);
    int final_col_bytes = (chunk_swizzled * 16) + (col_bytes % 16);
    int idx = row * 128 + final_col_bytes;
    return *( (__nv_bfloat16*) (&smem[idx]) );
}

__device__ __forceinline__ void write_swizzled(char* smem, int row, int col, float val) {
    int col_bytes = col * 2;
    int chunk = col_bytes / 16;
    int chunk_swizzled = chunk ^ (row % 8);
    int final_col_bytes = (chunk_swizzled * 16) + (col_bytes % 16);
    int idx = row * 128 + final_col_bytes;
    __nv_bfloat16 a = __float2bfloat16(val);
    *( (__nv_bfloat16*) (&smem[idx]) ) = a;
}

// C = A x B^T
__device__ void gemm_transposed_64x64(char* s_A0, char* s_A1, char* s_B0, char* s_B1, char* smem_out) {
    int tid = threadIdx.x;
    for (int col = 0; col < 64; col++) {
        float sum = 0;
        for (int k = 0; k < 64; k++) {
            sum += __bfloat162float(read_swizzled(s_A0, tid, k)) * __bfloat162float(read_swizzled(s_B0, col, k));
            sum += __bfloat162float(read_swizzled(s_A1, tid, k)) * __bfloat162float(read_swizzled(s_B1, col, k));
        }
        write_swizzled(smem_out, tid, col, sum);
    }
}

// C += A x B
__device__ void gemm_std_64x64_accum(char* s_A, char* s_B, char* smem_out) {
    int tid = threadIdx.x;
    for (int col = 0; col < 64; col++) {
        float sum = __bfloat162float(read_swizzled(smem_out, tid, col));
        for (int k = 0; k < 64; k++) {
            sum += __bfloat162float(read_swizzled(s_A, tid, k)) * __bfloat162float(read_swizzled(s_B, k, col));
        }
        write_swizzled(smem_out, tid, col, sum);
    }
}

__device__ void transpose_64x64_swizzled(const char* src, char* dst) {
    int tid = threadIdx.x;
    for (int col = 0; col < 64; col++) {
        float val = __bfloat162float(read_swizzled(src, tid, col));
        write_swizzled(dst, col, tid, val);
    }
}

__device__ void compute_D_local(char* s_O0, char* s_O1, char* s_dO0, char* s_dO1, float* s_DT) {
    int tid = threadIdx.x;
    float sum = 0;
    for(int k=0; k<64; ++k) {
        sum += __bfloat162float(read_swizzled(s_O0, tid, k)) * __bfloat162float(read_swizzled(s_dO0, tid, k));
        sum += __bfloat162float(read_swizzled(s_O1, tid, k)) * __bfloat162float(read_swizzled(s_dO1, tid, k));
    }
    s_DT[tid] = sum;
}

__device__ void apply_softmax_local(char* s_PT, char* s_S_T, const float* L_data, int bh, int q_start_local, int kv_start, float scale, int S_val) {
    int tid = threadIdx.x;
    const float* L_bh = L_data + (uint64_t)bh * S_val + q_start_local + tid;
    float lse = L_bh[0];
    for (int col = 0; col < 64; col++) {
        float s = __bfloat162float(read_swizzled(s_S_T, tid, col));
        float p = 0;
        if (kv_start + col <= q_start_local + tid && q_start_local + tid < S_val && kv_start + col < S_val) {
            p = expf(s * scale - lse);
        }
        write_swizzled(s_PT, tid, col, p);
    }
}

__device__ void compute_dS_local(char* s_dST, char* s_PT, char* s_dPT, float* s_DT, int q_start_local, int kv_start, int S_val) {
    int tid = threadIdx.x;
    for (int col = 0; col < 64; col++) {
        float p = __bfloat162float(read_swizzled(s_PT, tid, col));
        float dp = __bfloat162float(read_swizzled(s_dPT, tid, col));
        
        float ds = 0;
        if (kv_start + col <= q_start_local + tid && q_start_local + tid < S_val && kv_start + col < S_val) {
            ds = p * (dp - s_DT[tid]);
        }
        write_swizzled(s_dST, tid, col, ds);
    }
}

__device__ void store_gemm(__nv_bfloat16* global_D, char* smem_0, char* smem_1, int64_t row_base, int S_val) {
    int tid = threadIdx.x;
    for (int col = 0; col < 64; col++) {
        int g_row = row_base + tid;
        if (g_row < S_val) {
            global_D[(uint64_t)g_row * 128 + col] = read_swizzled(smem_0, tid, col);
            global_D[(uint64_t)g_row * 128 + 64 + col] = read_swizzled(smem_1, tid, col);
        }
    }
}

struct SharedStorage {
    __align__(1024) char pad_to_1024[1024];
    __align__(1024) char s_K0[8192];
    __align__(1024) char s_K1[8192];
    __align__(1024) char s_V0[8192];
    __align__(1024) char s_V1[8192];
    
    __align__(1024) char s_Q0[8192];
    __align__(1024) char s_Q1[8192];
    __align__(1024) char s_O0[8192];
    __align__(1024) char s_O1[8192];
    __align__(1024) char s_dO0[8192];
    __align__(1024) char s_dO1[8192];
    
    __align__(1024) char s_PT[8192];
    __align__(1024) char s_dPT[8192];
    __align__(1024) char s_dST[8192];
    __align__(1024) char s_dS[8192];
    
    __align__(1024) char s_dV0[8192];
    __align__(1024) char s_dV1[8192];
    
    __align__(1024) char s_dKT0[8192];
    __align__(1024) char s_dKT1[8192];
    
    __align__(1024) char s_dQ0[8192];
    __align__(1024) char s_dQ1[8192];
    
    __align__(1024) float s_DT[64];
    __align__(1024) uint64_t bar[1];
};

__global__ void bwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* __restrict__ L_data,
    __nv_bfloat16* __restrict__ dQ_bh,
    __nv_bfloat16* __restrict__ dK_bh,
    __nv_bfloat16* __restrict__ dV_bh,
    int64_t S_val, float scale)
{
    int bh = blockIdx.y;
    int kv_start = blockIdx.x * 64;
    
    extern __shared__ char smem_buf[];
    SharedStorage* smem = (SharedStorage*)(((size_t)smem_buf + 1023) & ~1023);

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(smem->bar, 1);
    }
    fence_mbarrier_init_fn();
    __syncthreads();
    
    uint32_t phase = 0;
    int tid = threadIdx.x;

    if (kv_start < S_val) {
        mbarrier_arrive_and_expect_tx_fn(smem->bar, 8192 * 4);
        tma_load_2d_fn(&tma_K, smem->bar, smem->s_K0, 0, bh * S_val + kv_start);
        tma_load_2d_fn(&tma_K, smem->bar, smem->s_K1, 64, bh * S_val + kv_start);

        tma_load_2d_fn(&tma_V, smem->bar, smem->s_V0, 0, bh * S_val + kv_start);
        tma_load_2d_fn(&tma_V, smem->bar, smem->s_V1, 64, bh * S_val + kv_start);
        
        mbarrier_wait_fn(smem->bar, phase);
        fence_proxy_async_fn();
        phase ^= 1;
    }
    
    for (int i = tid; i < 8192; i += blockDim.x) {
        ((char*)smem->s_dV0)[i] = 0;
        ((char*)smem->s_dV1)[i] = 0;
        ((char*)smem->s_dKT0)[i] = 0;
        ((char*)smem->s_dKT1)[i] = 0;
        ((char*)smem->s_dQ0)[i] = 0;
        ((char*)smem->s_dQ1)[i] = 0;
    }
    __syncthreads();

    // ==== Phase 1 & 2 Combined: Accumulate dV, dK and calculate dQ ====
    for (int q_start = kv_start; q_start < S_val; q_start += 64) {
        if (q_start < S_val) {
            mbarrier_arrive_and_expect_tx_fn(smem->bar, 8192 * 6);
            tma_load_2d_fn(&tma_Q, smem->bar, smem->s_Q0, 0, bh * S_val + q_start);
            tma_load_2d_fn(&tma_Q, smem->bar, smem->s_Q1, 64, bh * S_val + q_start);

            tma_load_2d_fn(&tma_dO, smem->bar, smem->s_dO0, 0, bh * S_val + q_start);
            tma_load_2d_fn(&tma_dO, smem->bar, smem->s_dO1, 64, bh * S_val + q_start);

            tma_load_2d_fn(&tma_O, smem->bar, smem->s_O0, 0, bh * S_val + q_start);
            tma_load_2d_fn(&tma_O, smem->bar, smem->s_O1, 64, bh * S_val + q_start);
            
            mbarrier_wait_fn(smem->bar, phase);
            fence_proxy_async_fn();
            phase ^= 1;
        }
        __syncthreads();

        if (q_start < S_val) {
            compute_D_local(smem->s_O0, smem->s_O1, smem->s_dO0, smem->s_dO1, smem->s_DT);
            
            gemm_transposed_64x64(smem->s_K0, smem->s_K1, smem->s_Q0, smem->s_Q1, smem->s_PT);
            
            apply_softmax_local(smem->s_PT, smem->s_PT, L_data, bh, q_start, kv_start, scale, S_val);
            
            gemm_transposed_64x64(smem->s_V0, smem->s_V1, smem->s_dO0, smem->s_dO1, smem->s_dPT);
            
            compute_dS_local(smem->s_dST, smem->s_PT, smem->s_dPT, smem->s_DT, q_start, kv_start, S_val);
            
            gemm_std_64x64_accum(smem->s_PT, smem->s_dO0, smem->s_dV0);
            gemm_std_64x64_accum(smem->s_PT, smem->s_dO1, smem->s_dV1);
            
            gemm_std_64x64_accum(smem->s_dST, smem->s_Q0, smem->s_dKT0);
            gemm_std_64x64_accum(smem->s_dST, smem->s_Q1, smem->s_dKT1);
            
            transpose_64x64_swizzled(smem->s_dST, smem->s_dS);
            
            gemm_std_64x64_accum(smem->s_dS, smem->s_K0, smem->s_dQ0);
            gemm_std_64x64_accum(smem->s_dS, smem->s_K1, smem->s_dQ1);
            
            for (int col = 0; col < 64; col++) {
                int g_row = q_start + tid;
                if (g_row < S_val) {
                     __nv_bfloat16 val0 = read_swizzled(smem->s_dQ0, tid, col);
                     __nv_bfloat16 val1 = read_swizzled(smem->s_dQ1, tid, col);
                     atomicAdd(&dQ_bh[(uint64_t)bh * S_val * 128 + (uint64_t)g_row * 128 + col], val0);
                     atomicAdd(&dQ_bh[(uint64_t)bh * S_val * 128 + (uint64_t)g_row * 128 + 64 + col], val1);
                }
            }
        }
        __syncthreads();
    }
    
    store_gemm(dV_bh + bh * S_val * 128, smem->s_dV0, smem->s_dV1, kv_start, S_val);
    store_gemm(dK_bh + bh * S_val * 128, smem->s_dKT0, smem->s_dKT1, kv_start, S_val);
}

CUresult create_tma_2d_descriptor_2B(
    CUtensorMap* d, void* globalAddress, 
    uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, 
    uint32_t smem_inner_dim, uint32_t smem_outer_dim, 
    CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) 
{
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2, 
        globalAddress,
        globalDim,
        globalStrides,
        boxDim, 
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        l2Promotion,
        oobFill
    );
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
    
    CUDA_CHECK(cudaMemsetAsync(dQ_data, 0, B * H * S_val * d * sizeof(__nv_bfloat16), stream));
    CUDA_CHECK(cudaMemsetAsync(dK_data, 0, B * H * S_val * d * sizeof(__nv_bfloat16), stream));
    CUDA_CHECK(cudaMemsetAsync(dV_data, 0, B * H * S_val * d * sizeof(__nv_bfloat16), stream));
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O, tma_dO;
    create_tma_2d_descriptor_2B(&tma_Q, (void*)Q_data, d, B * H * S_val, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_K, (void*)K_data, d, B * H * S_val, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_V, (void*)V_data, d, B * H * S_val, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_O, (void*)O_data, d, B * H * S_val, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_dO, (void*)dO_data, d, B * H * S_val, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    int64_t threads = 64;
    dim3 grid((S_val + 63) / 64, B * H);
    
    CUDA_CHECK(cudaFuncSetAttribute(bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedStorage)));
    
    bwd_kernel<<<grid, threads, sizeof(SharedStorage), stream>>>(
        tma_Q, tma_K, tma_V, tma_O, tma_dO, L_data, dQ_data, dK_data, dV_data, S_val, scale);
        
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_fa4::run);

} // namespace tvm_ffi_fa4