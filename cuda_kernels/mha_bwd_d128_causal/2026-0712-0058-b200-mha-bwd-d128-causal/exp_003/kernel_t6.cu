#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <math.h>
#include <stdio.h>
#include <algorithm>
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

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
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

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
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

namespace tvm_ffi_example_cuda {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = Q.size(3);

    float scale = 1.0f / sqrtf((float)d);
    int64_t BHS = B * H * S;

    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_ptr = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_ptr = static_cast<const float*>(L.data_ptr());
    
    __nv_bfloat16* dQ_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());

    CUtensorMap tma_Q, tma_K, tma_V, tma_O, tma_dO;

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    create_tma_2d_descriptor_2B(&tma_Q, (void*)((const char*)Q_ptr), d, B * H * S, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_K, (void*)((const char*)K_ptr), d, B * H * S, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_V, (void*)((const char*)V_ptr), d, B * H * S, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_O, (void*)((const char*)O_ptr), d, B * H * S, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_dO, (void*)((const char*)dO_ptr), d, B * H * S, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    int num_S_blocks = (S + 63) / 64;
    dim3 grid(num_S_blocks, B * H);
    dim3 block(128);
    
    int smem_size = 92160;
    CUDA_CHECK(cudaFuncSetAttribute(bwd_dQ_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    CUDA_CHECK(cudaFuncSetAttribute(bwd_dK_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    CUDA_CHECK(cudaFuncSetAttribute(bwd_dV_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    bwd_dQ_kernel<<<grid, block, smem_size, stream>>>(tma_Q, tma_K, tma_V, tma_dO, tma_O, L_ptr, dQ_ptr, S, BHS, scale);
    CUDA_CHECK(cudaGetLastError());
    
    bwd_dK_kernel<<<grid, block, smem_size, stream>>>(tma_Q, tma_K, tma_V, tma_dO, tma_O, L_ptr, dK_ptr, S, BHS, scale);
    CUDA_CHECK(cudaGetLastError());
    
    bwd_dV_kernel<<<grid, block, smem_size, stream>>>(tma_Q, tma_K, tma_V, tma_dO, tma_O, L_ptr, dV_ptr, S, BHS, scale);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

} // namespace tvm_ffi_example_cuda


__global__ void bwd_dQ_kernel(
    const __grid_constant__ CUtensorMap tma_Q, const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V, const __grid_constant__ CUtensorMap tma_dO,
    const __grid_constant__ CUtensorMap tma_O,
    const float* L_gmem,
    __nv_bfloat16* dQ_gmem,
    int64_t S, int64_t BHS, float scale) 
{
    setmaxnreg_inc_sync_fn<256>();
    int q_blk = blockIdx.x;
    int bh = blockIdx.y;
    int tid = threadIdx.x;
    
    extern __shared__ char smem_buf[];
    char* smem_Q_L = (char*)(smem_buf + 0 * 8192);
    char* smem_Q_R = (char*)(smem_buf + 1 * 8192);
    char* smem_K_L = (char*)(smem_buf + 2 * 8192);
    char* smem_K_R = (char*)(smem_buf + 3 * 8192);
    char* smem_V_L = (char*)(smem_buf + 4 * 8192);
    char* smem_V_R = (char*)(smem_buf + 5 * 8192);
    char* smem_O_L = (char*)(smem_buf + 6 * 8192);
    char* smem_O_R = (char*)(smem_buf + 7 * 8192);
    char* smem_dO_L = (char*)(smem_buf + 8 * 8192);
    char* smem_dO_R = (char*)(smem_buf + 9 * 8192);
    
    __align__(16) uint64_t* bar_A = (uint64_t*)(smem_buf + 9 * 8192);
    __align__(16) uint64_t* bar_B = (uint64_t*)(smem_buf + 9 * 8192 + 8);
    float* head_dOO_shared = (float*)(smem_buf + 9 * 8192 + 16);

    if (tid == 0) {
        init_smem_barrier_fn(bar_A, 1);
        init_smem_barrier_fn(bar_B, 1);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t phase_B = 0;
    
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_B, sizeof(__nv_bfloat16) * 64 * 64 * 6);
        tma_load_2d_fn(&tma_Q, bar_B, smem_Q_L, 0, bh * S + q_blk * 64);
        tma_load_2d_fn(&tma_Q, bar_B, smem_Q_R, 64, bh * S + q_blk * 64);
        tma_load_2d_fn(&tma_dO, bar_B, smem_dO_L, 0, bh * S + q_blk * 64);
        tma_load_2d_fn(&tma_dO, bar_B, smem_dO_R, 64, bh * S + q_blk * 64);
        tma_load_2d_fn(&tma_O, bar_B, smem_O_L, 0, bh * S + q_blk * 64);
        tma_load_2d_fn(&tma_O, bar_B, smem_O_R, 64, bh * S + q_blk * 64);
    }
    mbarrier_wait_fn(bar_B, phase_B);
    phase_B ^= 1;
    __syncthreads();

    if (tid < 64) {
        float sum = 0;
        for(int c = 0; c < 64; c++) {
            sum += __bfloat162float((( __nv_bfloat16*)smem_O_L)[tid * 64 + c]) * __bfloat162float((( __nv_bfloat16*)smem_dO_L)[tid * 64 + c]);
            sum += __bfloat162float((( __nv_bfloat16*)smem_O_R)[tid * 64 + c]) * __bfloat162float((( __nv_bfloat16*)smem_dO_R)[tid * 64 + c]);
        }
        head_dOO_shared[tid] = sum;
    }
    __syncthreads();

    float D_Q_L[64] = {0};
    float D_Q_R[64] = {0};
    int num_S_blocks = (S + 63) / 64;

    for (int k_blk = 0; k_blk <= q_blk && k_blk < num_S_blocks; k_blk++) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_B, sizeof(__nv_bfloat16) * 64 * 64 * 4);
            tma_load_2d_fn(&tma_K, bar_B, smem_K_L, 0, bh * S + k_blk * 64);
            tma_load_2d_fn(&tma_K, bar_B, smem_K_R, 64, bh * S + k_blk * 64);
            tma_load_2d_fn(&tma_V, bar_B, smem_V_L, 0, bh * S + k_blk * 64);
            tma_load_2d_fn(&tma_V, bar_B, smem_V_R, 64, bh * S + k_blk * 64);
        }
        mbarrier_wait_fn(bar_B, phase_B);
        phase_B ^= 1;
        __syncthreads();

        float Fixme_S[64 * 64];
        float Fixme_dP[64 * 64];
        
        for (int i = tid; i < 4096; i += blockDim.x) {
            int row = i / 64;
            int col = i % 64;
            
            float s_val = 0;
            for (int k = 0; k < 64; k++) {
                s_val += __bfloat162float((( __nv_bfloat16*)smem_Q_L)[row * 64 + k]) * __bfloat162float((( __nv_bfloat16*)smem_K_L)[col * 64 + k]);
                s_val += __bfloat162float((( __nv_bfloat16*)smem_Q_R)[row * 64 + k]) * __bfloat162float((( __nv_bfloat16*)smem_K_R)[col * 64 + k]);
            }
            Fixme_S[row * 64 + col] = s_val;
            
            float dp_val = 0;
            for (int k = 0; k < 64; k++) {
                dp_val += __bfloat162float((( __nv_bfloat16*)smem_dO_L)[row * 64 + k]) * __bfloat162float((( __nv_bfloat16*)smem_V_L)[col * 64 + k]);
                dp_val += __bfloat162float((( __nv_bfloat16*)smem_dO_R)[row * 64 + k]) * __bfloat162float((( __nv_bfloat16*)smem_V_R)[col * 64 + k]);
            }
            Fixme_dP[row * 64 + col] = dp_val;
        }
        __syncthreads(); 
        
        float Fixme_dS[64 * 64];
        for (int i = tid; i < 4096; i += blockDim.x) {
            int row = i / 64;
            int col = i % 64;
            
            float p_val = expf(Fixme_S[row * 64 + col] * scale - L_gmem[bh * S + q_blk * 64 + row]);
            if (p_val > 1.0f || p_val < 0.0f) p_val = 0.0f;
            if ((q_blk * 64 + row) < (k_blk * 64 + col) || q_blk * 64 + row >= S || k_blk * 64 + col >= S) {
                p_val = 0.0f;
            }
            
            Fixme_dS[row * 64 + col] = p_val * (Fixme_dP[row * 64 + col] - head_dOO_shared[row]) * scale;
        }
        __syncthreads();
        fence_proxy_async_fn();

        for (int i = tid; i < 4096; i += blockDim.x) {
            int row = i / 64;
            int col = i % 64;
            
            float dq_l = 0;
            for (int k = 0; k < 64; k++) {
                dq_l += Fixme_dS[row * 64 + k] * __bfloat162float((( __nv_bfloat16*)smem_K_L)[k * 64 + col]);
            }
            D_Q_L[row * 64 + col] += dq_l;
            
            float dq_r = 0;
            for (int k = 0; k < 64; k++) {
                dq_r += Fixme_dS[row * 64 + k] * __bfloat162float((( __nv_bfloat16*)smem_K_R)[k * 64 + col]);
            }
            D_Q_R[row * 64 + col] += dq_r;
        }
        __syncthreads();
    }

    for (int i = tid; i < 4096; i += blockDim.x) {
        int row = i / 64;
        int col = i % 64;
        
        uint32_t g_row = bh * S + q_blk * 64 + row;
        uint32_t g_col_0 = col;
        uint32_t g_col_1 = col + 64;
        
        if (g_row < BHS && g_col_0 < 128) {
            dQ_gmem[g_row * 128 + g_col_0] = __float2bfloat16(D_Q_L[row * 64 + col]);
        }
        if (g_row < BHS && g_col_1 < 128) {
            dQ_gmem[g_row * 128 + g_col_1] = __float2bfloat16(D_Q_R[row * 64 + col]);
        }
    }
}

__global__ void bwd_dK_kernel(
    const __grid_constant__ CUtensorMap tma_Q, const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V, const __grid_constant__ CUtensorMap tma_dO,
    const __grid_constant__ CUtensorMap tma_O,
    const float* L_gmem,
    __nv_bfloat16* dK_gmem,
    int64_t S, int64_t BHS, float scale) 
{
    setmaxnreg_inc_sync_fn<256>();
    int k_blk = blockIdx.x;
    int bh = blockIdx.y;
    int tid = threadIdx.x;

    extern __shared__ char smem_buf[];
    char* smem_Q_L = (char*)(smem_buf + 0 * 8192);
    char* smem_Q_R = (char*)(smem_buf + 1 * 8192);
    char* smem_K_L = (char*)(smem_buf + 2 * 8192);
    char* smem_K_R = (char*)(smem_buf + 3 * 8192);
    char* smem_V_L = (char*)(smem_buf + 4 * 8192);
    char* smem_V_R = (char*)(smem_buf + 5 * 8192);
    char* smem_O_L = (char*)(smem_buf + 6 * 8192);
    char* smem_O_R = (char*)(smem_buf + 7 * 8192);
    char* smem_dO_L = (char*)(smem_buf + 8 * 8192);
    char* smem_dO_R = (char*)(smem_buf + 9 * 8192);
    
    __align__(16) uint64_t* bar_A = (uint64_t*)(smem_buf + 9 * 8192);
    __align__(16) uint64_t* bar_B = (uint64_t*)(smem_buf + 9 * 8192 + 8);
    float* head_dOO_shared = (float*)(smem_buf + 9 * 8192 + 16);

    if (tid == 0) {
        init_smem_barrier_fn(bar_A, 1);
        init_smem_barrier_fn(bar_B, 1);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t phase_B = 0;
    
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_B, sizeof(__nv_bfloat16) * 64 * 64 * 4);
        tma_load_2d_fn(&tma_K, bar_B, smem_K_L, 0, bh * S + k_blk * 64);
        tma_load_2d_fn(&tma_K, bar_B, smem_K_R, 64, bh * S + k_blk * 64);
        tma_load_2d_fn(&tma_V, bar_B, smem_V_L, 0, bh * S + k_blk * 64);
        tma_load_2d_fn(&tma_V, bar_B, smem_V_R, 64, bh * S + k_blk * 64);
    }
    mbarrier_wait_fn(bar_B, phase_B);
    phase_B ^= 1;
    __syncthreads();

    float D_K_L[64] = {0};
    float D_K_R[64] = {0};
    int num_S_blocks = (S + 63) / 64;

    for (int q_blk = k_blk; q_blk < num_S_blocks; q_blk++) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_B, sizeof(__nv_bfloat16) * 64 * 64 * 6);
            tma_load_2d_fn(&tma_Q, bar_B, smem_Q_L, 0, bh * S + q_blk * 64);
            tma_load_2d_fn(&tma_Q, bar_B, smem_Q_R, 64, bh * S + q_blk * 64);
            tma_load_2d_fn(&tma_dO, bar_B, smem_dO_L, 0, bh * S + q_blk * 64);
            tma_load_2d_fn(&tma_dO, bar_B, smem_dO_R, 64, bh * S + q_blk * 64);
            tma_load_2d_fn(&tma_O, bar_B, smem_O_L, 0, bh * S + q_blk * 64);
            tma_load_2d_fn(&tma_O, bar_B, smem_O_R, 64, bh * S + q_blk * 64);
        }
        mbarrier_wait_fn(bar_B, phase_B);
        phase_B ^= 1;
        __syncthreads();

        if (tid < 64) {
            float sum = 0;
            for(int c = 0; c < 64; c++) {
                sum += __bfloat162float((( __nv_bfloat16*)smem_O_L)[tid * 64 + c]) * __bfloat162float((( __nv_bfloat16*)smem_dO_L)[tid * 64 + c]);
                sum += __bfloat162float((( __nv_bfloat16*)smem_O_R)[tid * 64 + c]) * __bfloat162float((( __nv_bfloat16*)smem_dO_R)[tid * 64 + c]);
            }
            head_dOO_shared[tid] = sum;
        }
        __syncthreads();

        float Fixme_S[64 * 64];
        float Fixme_dP[64 * 64];
        
        for (int i = tid; i < 4096; i += blockDim.x) {
            int row = i / 64;
            int col = i % 64;
            
            float s_val = 0;
            for (int k = 0; k < 64; k++) {
                s_val += __bfloat162float((( __nv_bfloat16*)smem_Q_L)[row * 64 + k]) * __bfloat162float((( __nv_bfloat16*)smem_K_L)[col * 64 + k]);
                s_val += __bfloat162float((( __nv_bfloat16*)smem_Q_R)[row * 64 + k]) * __bfloat162float((( __nv_bfloat16*)smem_K_R)[col * 64 + k]);
            }
            Fixme_S[row * 64 + col] = s_val;
            
            float dp_val = 0;
            for (int k = 0; k < 64; k++) {
                dp_val += __bfloat162float((( __nv_bfloat16*)smem_dO_L)[row * 64 + k]) * __bfloat162float((( __nv_bfloat16*)smem_V_L)[col * 64 + k]);
                dp_val += __bfloat162float((( __nv_bfloat16*)smem_dO_R)[row * 64 + k]) * __bfloat162float((( __nv_bfloat16*)smem_V_R)[col * 64 + k]);
            }
            Fixme_dP[row * 64 + col] = dp_val;
        }
        __syncthreads();
        
        float Fixme_S_T[64 * 64];
        for (int i = tid; i < 4096; i += blockDim.x) {
            int row = i / 64;
            int col = i % 64;
            
            float p_val = expf(Fixme_S[col * 64 + row] * scale - L_gmem[bh * S + q_blk * 64 + col]);
            if (p_val > 1.0f || p_val < 0.0f) p_val = 0.0f;
            if ((q_blk * 64 + col) < (k_blk * 64 + row) || q_blk * 64 + col >= S || k_blk * 64 + row >= S) {
                p_val = 0.0f;
            }
            
            Fixme_S_T[row * 64 + col] = p_val * (Fixme_dP[col * 64 + row] - head_dOO_shared[col]) * scale;
        }
        __syncthreads();
        fence_proxy_async_fn();

        for (int i = tid; i < 4096; i += blockDim.x) {
            int row = i / 64;
            int col = i % 64;
            
            float dk_l = 0;
            for (int col_q = 0; col_q < 64; col_q++) {
                dk_l += Fixme_S_T[row * 64 + col_q] * __bfloat162float((( __nv_bfloat16*)smem_Q_L)[col_q * 64 + col]);
            }
            D_K_L[row * 64 + col] += dk_l;
            
            float dk_r = 0;
            for (int col_q = 0; col_q < 64; col_q++) {
                dk_r += Fixme_S_T[row * 64 + col_q] * __bfloat162float((( __nv_bfloat16*)smem_Q_R)[col_q * 64 + col]);
            }
            D_K_R[row * 64 + col] += dk_r;
        }
        __syncthreads();
    }

    for (int i = tid; i < 4096; i += blockDim.x) {
        int row = i / 64;
        int col = i % 64;
        
        uint32_t g_row = bh * S + k_blk * 64 + row;
        uint32_t g_col_0 = col;
        uint32_t g_col_1 = col + 64;
        
        if (g_row < BHS && g_col_0 < 128) {
            dK_gmem[g_row * 128 + g_col_0] = __float2bfloat16(D_K_L[row * 64 + col]);
        }
        if (g_row < BHS && g_col_1 < 128) {
            dK_gmem[g_row * 128 + g_col_1] = __float2bfloat16(D_K_R[row * 64 + col]);
        }
    }
}

__global__ void bwd_dV_kernel(
    const __grid_constant__ CUtensorMap tma_Q, const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V, const __grid_constant__ CUtensorMap tma_dO,
    const __grid_constant__ CUtensorMap tma_O,
    const float* L_gmem,
    __nv_bfloat16* dV_gmem,
    int64_t S, int64_t BHS, float scale) 
{
    setmaxnreg_inc_sync_fn<256>();
    int k_blk = blockIdx.x;
    int bh = blockIdx.y;
    int tid = threadIdx.x;

    extern __shared__ char smem_buf[];
    char* smem_Q_L = (char*)(smem_buf + 0 * 8192);
    char* smem_Q_R = (char*)(smem_buf + 1 * 8192);
    char* smem_K_L = (char*)(smem_buf + 2 * 8192);
    char* smem_K_R = (char*)(smem_buf + 3 * 8192);
    char* smem_V_L = (char*)(smem_buf + 4 * 8192);
    char* smem_V_R = (char*)(smem_buf + 5 * 8192);
    char* smem_O_L = (char*)(smem_buf + 6 * 8192);
    char* smem_O_R = (char*)(smem_buf + 7 * 8192);
    char* smem_dO_L = (char*)(smem_buf + 8 * 8192);
    char* smem_dO_R = (char*)(smem_buf + 9 * 8192);
    
    __align__(16) uint64_t* bar_A = (uint64_t*)(smem_buf + 9 * 8192);
    __align__(16) uint64_t* bar_B = (uint64_t*)(smem_buf + 9 * 8192 + 8);
    float* head_dOO_shared = (float*)(smem_buf + 9 * 8192 + 16);

    if (tid == 0) {
        init_smem_barrier_fn(bar_A, 1);
        init_smem_barrier_fn(bar_B, 1);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t phase_B = 0;
    
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_B, sizeof(__nv_bfloat16) * 64 * 64 * 4);
        tma_load_2d_fn(&tma_K, bar_B, smem_K_L, 0, bh * S + k_blk * 64);
        tma_load_2d_fn(&tma_K, bar_B, smem_K_R, 64, bh * S + k_blk * 64);
        tma_load_2d_fn(&tma_V, bar_B, smem_V_L, 0, bh * S + k_blk * 64);
        tma_load_2d_fn(&tma_V, bar_B, smem_V_R, 64, bh * S + k_blk * 64);
    }
    mbarrier_wait_fn(bar_B, phase_B);
    phase_B ^= 1;
    __syncthreads();

    float D_V_L[64] = {0};
    float D_V_R[64] = {0};
    int num_S_blocks = (S + 63) / 64;

    for (int q_blk = k_blk; q_blk < num_S_blocks; q_blk++) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_B, sizeof(__nv_bfloat16) * 64 * 64 * 6);
            tma_load_2d_fn(&tma_Q, bar_B, smem_Q_L, 0, bh * S + q_blk * 64);
            tma_load_2d_fn(&tma_Q, bar_B, smem_Q_R, 64, bh * S + q_blk * 64);
            tma_load_2d_fn(&tma_dO, bar_B, smem_dO_L, 0, bh * S + q_blk * 64);
            tma_load_2d_fn(&tma_dO, bar_B, smem_dO_R, 64, bh * S + q_blk * 64);
            tma_load_2d_fn(&tma_O, bar_B, smem_O_L, 0, bh * S + q_blk * 64);
            tma_load_2d_fn(&tma_O, bar_B, smem_O_R, 64, bh * S + q_blk * 64);
        }
        mbarrier_wait_fn(bar_B, phase_B);
        phase_B ^= 1;
        __syncthreads();

        if (tid < 64) {
            float sum = 0;
            for(int c = 0; c < 64; c++) {
                sum += __bfloat162float((( __nv_bfloat16*)smem_O_L)[tid * 64 + c]) * __bfloat162float((( __nv_bfloat16*)smem_dO_L)[tid * 64 + c]);
                sum += __bfloat162float((( __nv_bfloat16*)smem_O_R)[tid * 64 + c]) * __bfloat162float((( __nv_bfloat16*)smem_dO_R)[tid * 64 + c]);
            }
            head_dOO_shared[tid] = sum;
        }
        __syncthreads();

        float Fixme_S[64 * 64];
        float Fixme_dP[64 * 64];
        
        for (int i = tid; i < 4096; i += blockDim.x) {
            int row = i / 64;
            int col = i % 64;
            
            float s_val = 0;
            for (int k = 0; k < 64; k++) {
                s_val += __bfloat162float((( __nv_bfloat16*)smem_Q_L)[row * 64 + k]) * __bfloat162float((( __nv_bfloat16*)smem_K_L)[col * 64 + k]);
                s_val += __bfloat162float((( __nv_bfloat16*)smem_Q_R)[row * 64 + k]) * __bfloat162float((( __nv_bfloat16*)smem_K_R)[col * 64 + k]);
            }
            Fixme_S[row * 64 + col] = s_val;
            
            float dp_val = 0;
            for (int k = 0; k < 64; k++) {
                dp_val += __bfloat162float((( __nv_bfloat16*)smem_dO_L)[row * 64 + k]) * __bfloat162float((( __nv_bfloat16*)smem_V_L)[col * 64 + k]);
                dp_val += __bfloat162float((( __nv_bfloat16*)smem_dO_R)[row * 64 + k]) * __bfloat162float((( __nv_bfloat16*)smem_V_R)[col * 64 + k]);
            }
            Fixme_dP[row * 64 + col] = dp_val;
        }
        __syncthreads();
        
        float Fixme_P_T[64 * 64];
        for (int i = tid; i < 4096; i += blockDim.x) {
            int row = i / 64;
            int col = i % 64;
            
            float p_val = expf(Fixme_S[col * 64 + row] * scale - L_gmem[bh * S + q_blk * 64 + col]);
            if (p_val > 1.0f || p_val < 0.0f) p_val = 0.0f;
            if ((q_blk * 64 + col) < (k_blk * 64 + row) || q_blk * 64 + col >= S || k_blk * 64 + row >= S) {
                p_val = 0.0f;
            }
            
            Fixme_P_T[row * 64 + col] = p_val;
        }
        __syncthreads();
        fence_proxy_async_fn();

        for (int i = tid; i < 4096; i += blockDim.x) {
            int row = i / 64;
            int col = i % 64;
            
            float dv_l = 0;
            for (int col_q = 0; col_q < 64; col_q++) {
                dv_l += Fixme_P_T[row * 64 + col_q] * __bfloat162float((( __nv_bfloat16*)smem_dO_L)[col_q * 64 + col]);
            }
            D_V_L[row * 64 + col] += dv_l;
            
            float dv_r = 0;
            for (int col_q = 0; col_q < 64; col_q++) {
                dv_r += Fixme_P_T[row * 64 + col_q] * __bfloat162float((( __nv_bfloat16*)smem_dO_R)[col_q * 64 + col]);
            }
            D_V_R[row * 64 + col] += dv_r;
        }
        __syncthreads();
    }

    for (int i = tid; i < 4096; i += blockDim.x) {
        int row = i / 64;
        int col = i % 64;
        
        uint32_t g_row = bh * S + k_blk * 64 + row;
        uint32_t g_col_0 = col;
        uint32_t g_col_1 = col + 64;
        
        if (g_row < BHS && g_col_0 < 128) {
            dV_gmem[g_row * 128 + g_col_0] = __float2bfloat16(D_V_L[row * 64 + col]);
        }
        if (g_row < BHS && g_col_1 < 128) {
            dV_gmem[g_row * 128 + g_col_1] = __float2bfloat16(D_V_R[row * 64 + col]);
        }
    }
}